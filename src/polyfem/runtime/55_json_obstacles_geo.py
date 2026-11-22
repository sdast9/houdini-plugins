# =============================================================================
# JSON builders
# =============================================================================


# =============================================================================
# Animated obstacles (2026-10-02)
#
# An obstacle moves when its Translate / Rotate / Scale / Pivot parameters
# are keyframed, like any Houdini object (the display chain already follows
# them). Its rest pose is the pose at the Time tab's Start Frame. A motion
# that only translates is exported as obstacle_displacements tables (value
# [1, 1, 1] times the displacement of each component at every step); any
# other motion as a mesh sequence: the obstacle surface posed at every frame
# of an integer frame rate on which every time step falls (PolyFEM's
# mesh_sequence interpolates vertices linearly between frames and holds the
# last one). A simulated body's transform only places its rest shape, so it
# must not be animated.
# =============================================================================

_TRANSFORM_VECTORS = (("translate", "xform_t__{}"), ("rotate", "xform_r__{}"),
                      ("scale", "xform_s__{}"), ("pivot", "tpivot_{}"),
                      ("pivot_rotate", "rpivot_{}"))
_TRANSFORM_ORDERS = ("xform_xOrd__{}", "xform_rOrd__{}")
# Files of one obstacle's mesh sequence, in its own folder under input/.
OBSTACLE_SEQUENCE_DIR = "obstacle_geo{}"
OBSTACLE_SEQUENCE_FILE = "frame_{:06d}.obj"
# A sequence may hold at most this many frames per time step (and in all);
# beyond that dt needs too high a frame rate.
_MAX_FRAMES_PER_STEP = 10


def _transform_parms(parent, geo):
    parms = []
    for _, name in _TRANSFORM_VECTORS:
        tuple_parm = parent.parmTuple(name.format(geo))
        if tuple_parm is not None:
            parms.extend(tuple_parm)
    for name in _TRANSFORM_ORDERS:
        parm = parent.parm(name.format(geo))
        if parm is not None:
            parms.append(parm)
    return parms


def transform_is_animated(parent, geo):
    """Whether a geometry's transform changes over time (keys, expressions)."""
    return any(parm.isTimeDependent() for parm in _transform_parms(parent, geo))


def geometry_matrix(parent, geo, time):
    """The geometry's transform at Houdini time `time`, built exactly as its
    transform SOP builds it (hou.hmath.buildTransform; row vectors, the
    translation in the last row)."""
    values = {}
    for key, name in _TRANSFORM_VECTORS:
        values[key] = tuple(float(parm.evalAtTime(time))
                            for parm in parent.parmTuple(name.format(geo)))
    orders = []
    for name in _TRANSFORM_ORDERS:
        parm = parent.parm(name.format(geo))
        orders.append(parm.menuItems()[int(parm.evalAtTime(time))])
    matrix = hou.hmath.buildTransform(values, orders[0], orders[1])
    return np.asarray(matrix.asTuple(), dtype=np.float64).reshape(4, 4)


def rest_matrix(parent, geo):
    """The transform PolyFEM receives as the rest pose: the transform SOP's
    own matrix for a static transform (as before), the pose at the Start
    Frame for an animated one."""
    if transform_is_animated(parent, geo):
        return geometry_matrix(
            parent, geo, hou.frameToTime(parent.evalParm("houdini_start_frame")))
    node = parent.node(f"transform_{geo}")
    return np.asarray(node.geometry().attribValue("xform"),
                      dtype=np.float64).reshape(4, 4)


def obstacle_motion(parent, geo, line):
    """How an obstacle moves over the run: {"kind": "static" | "translation"
    | "sequence", "matrices": its transform at every step}. Only translation
    -> per-step displacement of each component in "displacement"."""
    if not transform_is_animated(parent, geo):
        return {"kind": "static"}
    matrices = np.array([geometry_matrix(parent, geo, float(T))
                         for T in line.houdini_times])
    if np.all(matrices == matrices[0]):
        return {"kind": "static", "matrices": matrices}
    if np.all(matrices[:, :3, :3] == matrices[0, :3, :3]):
        return {"kind": "translation", "matrices": matrices,
                "times": line.times,
                "displacement": matrices[:, 3, :3] - matrices[0, 3, :3]}
    return {"kind": "sequence", "matrices": matrices}


def _frame_rate_of(value):
    """Smallest whole number F such that value * F is whole (to 1e-9), or
    None."""
    from fractions import Fraction
    if value == 0:
        return 1
    fraction = Fraction(value).limit_denominator(100000)
    if abs(float(fraction) - value) > 1e-9 * abs(value):
        return None
    return fraction.denominator


def sequence_rate(line):
    """The integer frame rate of an obstacle mesh sequence: every time step
    (t0 + k dt) must be a frame, since PolyFEM's frame index is t * fps
    (frames count from time 0). None when that needs more than
    _MAX_FRAMES_PER_STEP frames per step on average."""
    rates = [_frame_rate_of(line.dt), _frame_rate_of(line.t0)]
    if None in rates:
        return None
    rate = math.lcm(*rates)
    frames = int(round(float(line.times[-1]) * rate)) + 1
    if frames > _MAX_FRAMES_PER_STEP * (line.steps + 1):
        return None
    return rate


def _read_obj_surface(path):
    """Vertices (float64, as written) and triangles of an .obj file; polygons
    are split into fans."""
    vertices, faces = [], []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                index = []
                for token in parts[1:]:
                    number = int(token.split("/")[0])
                    index.append(number - 1 if number > 0
                                 else len(vertices) + number)
                for k in range(1, len(index) - 1):
                    faces.append([index[0], index[k], index[k + 1]])
            elif parts[0] == "l":
                raise hou.NodeError(
                    f"{os.path.basename(path)} has line elements; an "
                    "obstacle that rotates or scales is written as triangle "
                    "frames, so it must be a triangle surface.")
    return (np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
            np.asarray(faces, dtype=np.int64).reshape(-1, 3))


def _tet_boundary(tets):
    """Boundary triangles of a tet mesh (faces used once), outward."""
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]],
                            tets[:, [0, 1, 3]], tets[:, [0, 2, 1]]])
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True,
                                 return_counts=True)
    return faces[first[counts == 1]]


def _hex_boundary_tris(hexes):
    """Outward boundary quads of a hexahedral mesh, split into triangles."""
    quads = np.concatenate((
        hexes[:, [0, 3, 2, 1]],
        hexes[:, [4, 5, 6, 7]],
        hexes[:, [0, 1, 5, 4]],
        hexes[:, [1, 2, 6, 5]],
        hexes[:, [2, 3, 7, 6]],
        hexes[:, [3, 0, 4, 7]],
    ), axis=0)
    keys = np.sort(quads, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True,
                                 return_counts=True)
    boundary = quads[first[counts == 1]]
    return np.concatenate((boundary[:, [0, 1, 2]], boundary[:, [0, 2, 3]]),
                          axis=0)


def obstacle_surface(parent, geo):
    """(vertices, triangles, exact) of an obstacle mesh in its file's own
    coordinates: float64 from the file for .obj and .msh (the surface of a
    volume mesh, as PolyFEM extracts it), Houdini's float32 points otherwise."""
    path = parent.evalParm(f"file_location{geo}")
    extension = os.path.splitext(path)[1].lower()
    if extension == ".obj":
        vertices, faces = _read_obj_surface(path)
        return vertices, faces, True
    if extension == ".msh":
        reader = hou.nodeType(hou.sopNodeTypeCategory(), "MSH_Reader::3.0")
        mesh = reader.hdaModule().read_msh(path)
        cells = mesh["cells"]
        parts = []
        if "tet" in cells:
            parts.append(_tet_boundary(np.asarray(cells["tet"]["corners"])[:, :4]))
        if "hex" in cells:
            parts.append(_hex_boundary_tris(np.asarray(cells["hex"]["corners"])[:, :8]))
        if not parts:
            if "tri" in cells:
                parts.append(np.asarray(cells["tri"]["corners"])[:, :3])
            if "quad" in cells:
                quads = np.asarray(cells["quad"]["corners"])[:, :4]
                parts.append(np.concatenate((quads[:, [0, 1, 2]],
                                             quads[:, [0, 2, 3]]), axis=0))
        if not parts:
            raise hou.NodeError(
                f"Obstacle {geo}: {os.path.basename(path)} has no surface or "
                "volume elements (triangles, quads, tetrahedra or hexahedra).")
        faces = np.concatenate(parts, axis=0)
        used, faces = np.unique(faces, return_inverse=True)
        points = np.asarray(mesh["points"], dtype=np.float64)[used]
        return points, faces.reshape(-1, 3), True
    source = parent.node(f"geo_{geo}").geometry()
    points = np.frombuffer(source.pointFloatAttribValuesAsString("P"),
                           dtype=np.float32).astype(np.float64).reshape(-1, 3)
    faces = []
    for prim in source.prims():
        ids = [vertex.point().number() for vertex in prim.vertices()]
        for k in range(1, len(ids) - 1):
            faces.append([ids[0], ids[k], ids[k + 1]])
    return points, np.asarray(faces, dtype=np.int64).reshape(-1, 3), False


def _write_obj(path, vertices, faces):
    with open(path, "w") as handle:
        handle.write("".join(
            f"v {x!r} {y!r} {z!r}\n" for x, y, z in vertices.tolist()))
        handle.write("".join(
            f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces.tolist()))


def write_obstacle_sequence(parent, geo, line, input_dir, cache=None):
    """Write one obstacle's mesh sequence (frame i = the obstacle posed at
    simulation time i / fps, in simulation coordinates; times before Start
    Time hold the rest pose) and return its geometry entry."""
    rate = sequence_rate(line)
    if rate is None:
        raise hou.NodeError(SEQUENCE_RATE_ERROR.format(
            geo=geo, dt=line.dt, t0=line.t0))
    last = int(round(float(line.times[-1]) * rate))
    vertices, faces, _ = _cached(
        cache, "obstacles", int(geo), lambda: obstacle_surface(parent, geo))
    if not len(faces):
        raise hou.NodeError(f"Obstacle {geo} has no triangles to write.")
    folder = OBSTACLE_SEQUENCE_DIR.format(geo)
    directory = os.path.join(input_dir, folder)
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        if name.startswith("frame_") and name.endswith(".obj"):
            os.remove(os.path.join(directory, name))
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    files = []
    for index in range(last + 1):
        t = max(index / rate, line.t0)
        matrix = geometry_matrix(parent, geo, line.houdini_time(t))
        name = OBSTACLE_SEQUENCE_FILE.format(index)
        _write_obj(os.path.join(directory, name), (homogeneous @ matrix)[:, :3],
                   faces)
        files.append(f"{folder}/{name}")
    return {"type": "mesh_sequence", "mesh_sequence": files, "fps": int(rate),
            "enabled": bool(parent.parm(f"is_enabled{geo}").eval()),
            "is_obstacle": True}


SEQUENCE_RATE_ERROR = (
    "Obstacle {geo} rotates or scales over time, so it is written as one "
    "mesh per frame of a whole-number frame rate on which every time step "
    "falls (PolyFEM counts the frames from time 0). For dt = {dt:g} and "
    "Start Time {t0:g} that needs more than 10 frames per time step; choose "
    "a dt whose inverse is a whole number, e.g. 0.01 or 0.0125, and a Start "
    "Time that is a multiple of it (Time tab).")
ANIMATED_BODY_ERROR = (
    "Geometry {geo} is a simulated body and its transform is animated "
    "(keyframes or a time expression on Translate/Rotate/Scale/Pivot). A "
    "body's transform only places its rest shape; move the body with a "
    "Dirichlet condition (with a Time Curve), or remove the keys. Obstacles "
    "can be animated.")
ANIMATED_OBSTACLE_EXPRESSION_ERROR = (
    "Obstacle {geo} has an animated transform and an Obstacle Displacement "
    "expression; PolyFEM can take only one motion per obstacle. Set Obstacle "
    "Displacement to [0, 0, 0] (or remove the transform keys).")


def _obstacle_displacement_is_zero(parent, geo):
    try:
        value = _read_expr_vector(
            parent, f"obstacle_disp_vec{geo}", f"obstacle_disp{geo}",
            f"obstacle {geo}")
    except hou.Error:
        return False
    return isinstance(value, list) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0
        for v in value)


def build_geometry_and_materials(parent, data, input_dir, plans=None,
                                 motions=None, cache=None):
    num_geos = parent.evalParm("num_geos")
    if num_geos == 0:
        raise hou.NodeError("No input geometry provided!")
    data["geometry"] = []
    data["materials"] = []
    orders = []

    _check_composite_consistency(parent)
    materials, shared_fields = _plan_body_materials(parent)
    export_per_element_materials(
        parent, input_dir,
        os.path.join(os.path.dirname(input_dir), "output"),
        cache=cache, shared_fields=shared_fields)

    # Normally every mesh is already staged in input/; after a change of
    # working directory they are staged again here, under names that two
    # different meshes can never share.
    taken = _staged_meshes(parent, input_dir)
    lazy_timeline = _LazyTimeline(parent)
    for geo in range(1, num_geos + 1):
        is_obstacle = bool(parent.parm(f"is_obstacle{geo}").eval())
        enabled = bool(parent.parm(f"is_enabled{geo}").eval())
        path_parm = parent.parm(f"file_location{geo}").eval()
        if not path_parm:
            raise hou.NodeError(f"No file location for geo {geo}!")
        mesh_ref = os.path.basename(
            _stage_mesh(geo, path_parm, input_dir, taken))

        animated = transform_is_animated(parent, geo)
        if animated and not is_obstacle:
            raise hou.NodeError(ANIMATED_BODY_ERROR.format(geo=geo))
        T, R, S = decompose_rowvec_xyz_4x4(
            rest_matrix(parent, geo).ravel().tolist())
        transformation = {"translation": list(T), "rotation": list(R),
                          "scale": list(S)}

        if is_obstacle:
            motion = obstacle_motion(parent, geo, lazy_timeline.get()) \
                if animated else {"kind": "static"}
            if motions is not None:
                motions[geo] = motion
            if motion["kind"] != "static" \
                    and not _obstacle_displacement_is_zero(parent, geo):
                raise hou.NodeError(ANIMATED_OBSTACLE_EXPRESSION_ERROR.format(
                    geo=geo))
            if motion["kind"] == "sequence":
                data["geometry"].append(write_obstacle_sequence(
                    parent, geo, lazy_timeline.get(), input_dir, cache=cache))
                continue
            data["geometry"].append({
                "mesh": mesh_ref, "enabled": enabled, "is_obstacle": True,
                "surface_selection": obstacle_id(geo),
                "transformation": transformation})
            continue

        num_vols = parent.evalParm(f"num_volumes{geo}")
        volumefile = export_volumes(parent, geo, input_dir)
        surface_sel, point_sel = export_sidesets(
            parent, geo, num_vols, input_dir, plans, cache=cache)

        geo_data = {"mesh": mesh_ref, "enabled": enabled, "is_obstacle": False,
                    "volume_selection": volumefile,
                    "transformation": transformation}
        if surface_sel:
            geo_data["surface_selection"] = surface_sel
        if point_sel:
            geo_data["point_selection"] = point_sel
        data["geometry"].append(geo_data)

        for vol in range(1, num_vols + 1):
            data["materials"].append(materials.get((geo, vol)) if enabled
                                     else build_material(parent, geo, vol))
            orders.append({"id": vol_id(geo, vol),
                           "order": parent.evalParm(f"mainOrder{geo}_{vol}") + 1})
    return orders


_DAMPING_MODELS = frozenset({
    "NeoHookean", "LinearElasticity", "SaintVenant",
    "FixedCorotational", "IsochoricNeoHookean"})

INCOMPRESSIBLE_LINEAR_ERROR = (
    "IncompressibleLinearElasticity requires separate displacement and "
    "pressure FE space IDs (and forbids contact, pressure loads and "
    "MaterialSum nesting), which this node does not configure. Use "
    "NeoHookean, IsochoricNeoHookean or LinearElasticity (with a high "
    "Poisson's ratio such as 0.49) instead.")


