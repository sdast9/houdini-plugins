# =============================================================================
# Sideset / selection export (numpy bulk)
#
# PolyFEM gives every boundary face exactly ONE boundary id: the first surface
# selection that matches it and, inside a selection file, the first matching
# row (utils/Selection.cpp FileSelection::id, mesh/GeometryReader.cpp). Its
# conditions then act per id: a second dirichlet entry for one id is refused,
# a second neumann / normal-aligned / pressure entry is silently ignored, and
# a normal-aligned traction replaces a neumann one on the same id
# (assembler/GenericProblem.cpp). The HDA therefore resolves sidesets itself:
#   * every sideset -- picked faces, "*", or a typed axis/box/sphere/plane
#     selection evaluated here on face barycentres with PolyFEM's own rules --
#     becomes an explicit set of faces (or of nodes, for point sidesets);
#   * all conditions of one sideset share its id, and faces that lie in
#     several sidesets get one combined id carrying the conditions of all of
#     them, merged into the single entries PolyFEM honours;
#   * conflicting prescriptions are refused by name instead of one of them
#     being dropped silently.
# Selection files hold PolyFEM's own vertex numbers (see mesh_vertex_ids).
# =============================================================================


_BC_TYPE_NAMES = ("Dirichlet", "Neumann", "Normal Aligned Neumann",
                  "Pressure Boundary", "Pressure Cavity",
                  "Nodal Force (Nodal Neumann)")
_BC_LIST_KEYS = ("dirichlet_boundary", "neumann_boundary",
                 "normal_aligned_neumann_boundary", "pressure_boundary",
                 "pressure_cavity", "nodal_neumann_boundary")
# Boundary types whose value is a 3-component vector [x, y, z]:
# 0 = Dirichlet, 1 = Neumann, 5 = Nodal Force (Nodal Neumann).
_VECTOR_BC_KINDS = (0, 1, 5)
# Written next to params.json: the sidesets as entered (patterns, group
# types, conditions) so Import can restore them exactly. PolyFEM never reads
# it.
SIDESET_RECORD_FILE = "hda_sidesets.json"
SIDESET_RECORD_SCHEMA = "polyfem-houdini-sidesets"


def sideset_id(geo, vol, sideset):
    """Boundary id shared by every condition of one sideset."""
    return boundary_id(geo, vol, sideset, 1)


def combination_id(geo, vol, index):
    """Boundary id of faces/nodes that lie in several sidesets.

    The sideset field is 0, which no sideset uses, so these never collide
    with sideset ids.
    """
    return boundary_id(geo, vol, 0, index)


def _surface_arrays(surf_geo):
    """Bulk read of the provenance attributes stamped by SURFACE_VEX.

    Returns (entity, face_ids) where face_ids is (N, 4) int64 with -1 in
    the last column for triangles. The ids are node tag - 1; PolyFEM's own
    vertex numbers come from mesh_vertex_ids.
    """
    entity = np.frombuffer(
        surf_geo.primIntAttribValuesAsString("Entity"), dtype=np.int32)
    cols = [np.frombuffer(surf_geo.primIntAttribValuesAsString(f"fp{c}"),
                          dtype=np.int32) for c in range(4)]
    face_ids = np.stack(cols, axis=1).astype(np.int64)
    return entity, face_ids


def mesh_vertex_ids(parent, geo):
    """PolyFEM's vertex number for every node of one geometry's .msh file.

    PolyFEM numbers a node tag - 1 when the node tags are exactly 1..N, and
    by its position in the file otherwise (io/MshReader.cpp). The MSH Reader
    keeps file order and stores the tag in msh_pt_id. Returns
    (vid_of_key, key_of_vid): vid_of_key[tag - 1] is the PolyFEM vertex
    number (-1 for tags no node carries) and key_of_vid[vid] = tag - 1.
    The surface provenance (fp*, msh_pt_id) is tag based and is translated
    with these at export and import.
    """
    source = parent.node(f"geo_{geo}")
    if source is None:
        raise hou.NodeError(
            f"Geometry {geo} has no mesh node; import its mesh again")
    tags = np.frombuffer(
        source.geometry().pointIntAttribValuesAsString("msh_pt_id"),
        dtype=np.int32).astype(np.int64)
    if not len(tags):
        raise hou.NodeError(f"Geometry {geo}: the mesh has no nodes")
    if tags.min() < 1:
        raise hou.NodeError(
            f"Geometry {geo}: the .msh file has node tag {int(tags.min())}; "
            f"Gmsh node tags start at 1")
    count, max_tag = len(tags), int(tags.max())
    vid_of_key = np.full(max_tag, -1, dtype=np.int64)
    if max_tag == count:
        vid_of_key[tags - 1] = tags - 1
        key_of_vid = np.arange(count, dtype=np.int64)
    else:
        vid_of_key[tags - 1] = np.arange(count, dtype=np.int64)
        key_of_vid = tags - 1
    return vid_of_key, key_of_vid


def native_selection_mask(pattern, positions, label):
    """Evaluate a typed selection on face barycentres or node positions.

    Same rules as PolyFEM's own selections (utils/Selection.cpp): box bounds
    inclusive, sphere |p - c|^2 <= r^2, "axis:+z:v" -> z >= v and
    "axis:-z:v" -> z <= v, plane (p - point) . normal >= 0. Positions are the
    transformed ones, as PolyFEM applies the geometry transformation before
    it evaluates selections.
    """
    spec = parse_native_selection(pattern, 0, label)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    try:
        if "axis" in spec:
            axis = str(spec["axis"]).strip().lower()
            if not axis or axis[-1] not in "xyz" or axis[:-1] not in ("", "+", "-"):
                raise ValueError(axis)
            values = points[:, "xyz".index(axis[-1])]
            position = float(spec["position"])
            if axis.startswith("-"):
                return values <= position
            return values >= position
        if "box" in spec:
            low = np.asarray(spec["box"][0], dtype=np.float64).reshape(3)
            high = np.asarray(spec["box"][1], dtype=np.float64).reshape(3)
            return np.all((points >= low) & (points <= high), axis=1)
        if "center" in spec:
            centre = np.asarray(spec["center"], dtype=np.float64).reshape(3)
            delta = points - centre
            radius = float(spec["radius"])
            return np.einsum("ij,ij->i", delta, delta) <= radius * radius
        normal = np.asarray(spec["normal"], dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            raise ValueError("zero normal")
        point = np.asarray(spec["point"], dtype=np.float64).reshape(3)
        return np.einsum("ij,j->i", points - point, normal / length) >= 0.0
    except (ValueError, TypeError, KeyError, IndexError):
        raise hou.NodeError(
            f"Could not evaluate the selection '{pattern}' for {label}. "
            "Examples: axis:+z:0.95 | box:[0,0,0],[1,1,1] | "
            "sphere:[0,0,0],0.5 | plane:[0,0,1],[0,0,0.5]")


