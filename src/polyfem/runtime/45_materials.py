# =============================================================================
# Per-element material data (fiber directions, per-element scalars)
#
# PolyFEM binds per-element material data by GLOBAL element id: the elements of
# every enabled, non-obstacle geometry concatenated in geometry order, keeping
# each mesh's own element order. Everything here is built around that contract:
#   * canonical prim attributes (pf_*) on a per-geometry stamp node, so the
#     viewport shows exactly the data the solver is handed
#   * one global fibers.vtk (one VECTORS array per family) and global-length
#     per-element scalar files
# =============================================================================

# Material menu tokens, in DialogScript menu order (the index is stored in
# saved scenes, so entries may be appended but never reordered).
MATERIAL_TOKENS = (
    "LinearElasticity", "HookeLinearElasticity", "IncompressibleLinearElasticity",
    "NeoHookean", "IncompressibleOgden", "UnconstrainedOgden", "MooneyRivlin",
    "MooneyRivlin3Param", "MooneyRivlin3ParamSymbolic", "SaintVenant",
    "FixedCorotational", "IsochoricNeoHookean", "HGOFiber", "HGODispersion",
    "ActiveFiber", "MaterialSum", "GrowthNeoHookean", "InversionBarrier",
    "AMIPS",
)
# Matrix model menu tokens for MaterialSum (indices 0..12 preserved).
MATRIX_MODEL_TOKENS = (
    MATERIAL_TOKENS[:12] + ("None", "GrowthNeoHookean", "AMIPS")
)
FIBER_MODELS = ("HGOFiber", "HGODispersion", "ActiveFiber")
# What the Material menus show for each PolyFEM type (the DialogScript menus
# carry the same labels; the tokens above are what params.json receives).
MATERIAL_LABELS = {
    "LinearElasticity": "Linear Elastic (small strains only)",
    "HookeLinearElasticity": "Anisotropic Linear Elastic (Hooke)",
    "IncompressibleLinearElasticity": "Incompressible Linear Elastic",
    "NeoHookean": "Neo-Hookean (recommended)",
    "IncompressibleOgden": "Ogden, Incompressible",
    "UnconstrainedOgden": "Ogden, Compressible",
    "MooneyRivlin": "Mooney-Rivlin, 2 Parameters",
    "MooneyRivlin3Param": "Mooney-Rivlin, 3 Parameters",
    "MooneyRivlin3ParamSymbolic": "Mooney-Rivlin, 3 Parameters (Symbolic)",
    "SaintVenant": "Saint Venant-Kirchhoff (Anisotropic)",
    "FixedCorotational": "Fixed Corotational",
    "IsochoricNeoHookean": "Isochoric Neo-Hookean",
    "HGOFiber": "HGO Fiber (aligned)",
    "HGODispersion": "HGO Dispersion (GOH)",
    "ActiveFiber": "Active Fiber (muscle)",
    "MaterialSum": "Composite (Matrix + Fiber Families)",
    "GrowthNeoHookean": "Growth Neo-Hookean (3D)",
    "InversionBarrier": "Inversion Barrier",
    "AMIPS": "AMIPS (Distortion Energy)",
}

# Volume-level parameter names that differ from their family-level base.
_VOLUME_PARM_ALIASES = {"k1": "hgo_k1", "k2": "hgo_k2"}


def material_token(parent, geo, vol):
    index = parent.evalParm(f"materials{geo}_{vol}")
    if not 0 <= index < len(MATERIAL_TOKENS):
        raise hou.NodeError(f"Unknown material menu index {index}")
    return MATERIAL_TOKENS[index]


def _parm_name(geo, vol, fam, base):
    """Parameter name for a material slot.

    A plain fiber material keeps its parameters at volume level (fam is None);
    the fiber families of a composite live in a third multiparm level.
    """
    if fam is None:
        return f"{_VOLUME_PARM_ALIASES.get(base, base)}{geo}_{vol}"
    return f"fam_{base}{geo}_{vol}_{fam}"


def _slot_parm(parent, geo, vol, fam, base):
    """Value of a material-slot parameter; menus evaluate to their token.

    evalParm() would hand back a menu's *index*, which silently compares
    unequal to every token it is tested against.
    """
    name = _parm_name(geo, vol, fam, base)
    parm = parent.parm(name)
    if parm is None:
        raise hou.NodeError(f"Missing parameter '{name}'")
    if parm.parmTemplate().type() == hou.parmTemplateType.Menu:
        return parm.evalAsString()
    return parm.eval()


def material_families(parent, geo, vol):
    """[(fam, model_token)] fiber families of one subdomain.

    fam is None for a plain fiber material, 1..N for a composite's families.
    """
    token = material_token(parent, geo, vol)
    if token in FIBER_MODELS:
        return [(None, token)]
    if token != "MaterialSum":
        return []
    count = parent.evalParm(f"num_fiber_families{geo}_{vol}")
    families = []
    for fam in range(1, count + 1):
        parm = parent.parm(f"fam_model{geo}_{vol}_{fam}")
        if parm is None:
            continue
        families.append((fam, parm.evalAsString()))
    return families


def fiber_attrib(vol, fam, side=""):
    """Canonical prim attribute holding one family's directions.

    Namespaced by subdomain and family: several subdomains of one geometry
    share the stamp node, so a bare pf_fiber would collide between them.
    """
    return f"pf_fiber_{vol}_{0 if fam is None else fam}{side}"


def scalar_attrib(name, vol, fam):
    return f"pf_{name}_{vol}_{0 if fam is None else fam}"


def fiber_field(geo, vol, fam, side=""):
    """VECTORS array name for one family inside the shared fibers.vtk."""
    base = f"FIB_{geo}_{vol}" if fam is None else f"FIB_{geo}_{vol}_{fam}"
    return base + side


def mirror_sides(parent, geo, vol, fam):
    """('p', 'm') for a symmetric +/-theta family, ('',) for a single one."""
    if fam is None:
        return ("",)
    parm = parent.parm(f"fam_mirror{geo}_{vol}_{fam}")
    if parm is not None and parm.eval() \
            and _slot_parm(parent, geo, vol, fam, "fib_source") != "expression":
        return ("p", "m")
    return ("",)


def read_cell_vectors_legacy_vtk(path, field):
    """Per-element vectors from a legacy ASCII VTK, mirroring polyfem's reader.

    Only the CELL_DATA count and the named VECTORS array matter; points and
    cells are not required.
    """
    count = None
    with open(path) as handle:
        tokens = []
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "CELL_DATA":
                count = int(parts[1])
                continue
            if parts[0] == "VECTORS":
                if len(parts) < 2 or parts[1] != field:
                    continue
                if count is None:
                    raise hou.NodeError(
                        f"{path}: VECTORS before CELL_DATA")
                for line in handle:
                    tokens.extend(line.split())
                    if len(tokens) >= 3 * count:
                        break
                if len(tokens) < 3 * count:
                    raise hou.NodeError(
                        f"{path}: expected {count} vectors for '{field}', "
                        f"found {len(tokens) // 3}")
                return np.asarray(tokens[:3 * count], dtype=np.float64).reshape(-1, 3)
    raise hou.NodeError(
        f"VECTORS '{field}' not found under CELL_DATA in {path}")


def _volume_mask(geo_data):
    """Boolean mask of the primary (volume) elements in Houdini prim order."""
    if geo_data.findPrimAttrib("is_volume") is None:
        return None
    return np.frombuffer(
        geo_data.primIntAttribValuesAsString("is_volume"),
        dtype=np.int32).astype(bool)


def solver_element_order(geo_data, is_direct_sop=False, *, expect_msh=False):
    """Indices that gather Houdini volume rows into PolyFEM element order.

    The original Direct SOP already has the order written by the MSH writer.
    After the staged file is read back for display, its MSH provenance
    restores that order just like any other MSH file, including after a Sort.
    """
    if is_direct_sop:
        count = sum(geo_data.countPrimType(kind) for kind in
                    (hou.primType.Tetrahedron, hou.primType.Hexahedron))
        return np.arange(count, dtype=np.int64)
    mask = _volume_mask(geo_data)
    count = int(mask.sum()) if mask is not None else \
        geo_data.intrinsicValue("primitivecount")
    identity = np.arange(count, dtype=np.int64)
    attribute = geo_data.findPrimAttrib("msh_element_order")
    if attribute is None:
        if not expect_msh and geo_data.findPrimAttrib("ElementNum") is None \
                and geo_data.findPointAttrib("msh_pt_id") is None \
                and any(geo_data.countPrimType(kind) == 0 for kind in
                        (hou.primType.Tetrahedron, hou.primType.Hexahedron)):
            return identity
    elif attribute.dataType() == hou.attribData.Int and attribute.size() == 1 \
            and not attribute.isArrayType():
        order = np.frombuffer(
            geo_data.primIntAttribValuesAsString("msh_element_order"),
            dtype=np.int32)
        if mask is not None:
            order = order[mask]
        if np.array_equal(order, identity):
            return identity
        permutation = np.argsort(order, kind="stable")
        if np.array_equal(order[permutation], identity):
            return permutation
    raise hou.NodeError(
        "Invalid msh_element_order: expected one unique solver element index "
        f"from 0 to {count - 1} per volume primitive. Reload the original MSH "
        "and preserve msh_element_order on downstream SOPs.")


def volume_element_count(parent, geo):
    branch = parent.node(f"branch_{geo}")
    if branch is None:
        return 0
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    if mask is None:
        return geo_data.intrinsicValue("primitivecount")
    return int(mask.sum())


def global_element_table(parent):
    """[(geo, offset, count)] over the geometries polyfem actually meshes.

    Obstacles and disabled geometries are absent from the solver's element
    numbering, so they consume no ids -- and per-element data for them is
    simply not exported.
    """
    table = []
    offset = 0
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval():
            continue
        if not parent.parm(f"is_enabled{geo}").eval():
            continue
        count = volume_element_count(parent, geo)
        table.append((geo, offset, count))
        offset += count
    return table, offset


# Mean of each primitive's vertex positions. Run by the Attribute VOP verb at
# 64-bit precision, which sums in the same order as numpy's mean of the
# float32 positions, so the result is bitwise the per-primitive Python loop it
# replaces -- in milliseconds instead of seconds per million elements.
_CENTROID_VEX = """
vector c = {0, 0, 0};
int n = primvertexcount(0, @primnum);
for (int j = 0; j < n; j++)
    c += point(0, "P", vertexpoint(0, vertexindex(0, @primnum, j)));
v@__pf_centroid = n > 0 ? c / n : c;
"""


def _prim_centroids(geo_data, mask):
    """Volume-element centroids in element order (recomputed, not cached).

    MSH_Reader stamps a centroid attribute, but it predates the geometry
    transform; recomputing keeps the match in the space the user is looking at.
    """
    count = geo_data.intrinsicValue("primitivecount")
    centroids = None
    verb = hou.sopNodeTypeCategory().nodeVerb("attribvop")
    if verb is not None and count:
        verb.setParms({"bindclass": 1, "vexsrc": 3,
                       "vexsnippet": _CENTROID_VEX, "vex_precision": "64"})
        result = hou.Geometry()
        try:
            verb.execute(result, [geo_data])
            centroids = np.frombuffer(
                result.primFloatAttribValuesAsString(
                    "__pf_centroid", float_type=hou.numericData.Float64),
                dtype=np.float64).reshape(-1, 3)
        except (hou.Error, ValueError):
            centroids = None
    if centroids is None or len(centroids) != count:
        positions = np.frombuffer(
            geo_data.pointFloatAttribValuesAsString("P"),
            dtype=np.float32).astype(np.float64).reshape(-1, 3)
        centroids = np.empty((count, 3))
        for prim in geo_data.prims():
            pts = [v.point().number() for v in prim.vertices()]
            centroids[prim.number()] = positions[pts].mean(axis=0)
    return centroids[mask] if mask is not None else centroids


def _nearest_centroids_numpy(src_centroids, target_centroids):
    """O(N) pure-NumPy nearest-centroid matcher using a 3D spatial hash.

    Replaces the O(N^2) broadcast matrix when scipy.spatial.cKDTree is absent;
    falls back to chunked exact distances only for targets whose 27-cell
    neighborhood is empty (e.g. mismatched meshes).
    """
    src = np.asarray(src_centroids, dtype=np.float64)
    tgt = np.asarray(target_centroids, dtype=np.float64)
    n_src, n_tgt = len(src), len(tgt)
    if n_src == 0 or n_tgt == 0:
        return (np.empty(n_tgt, dtype=np.float64),
                np.empty(n_tgt, dtype=np.int64))
    origin = np.minimum(src.min(axis=0), tgt.min(axis=0))
    span = np.maximum(src.max(axis=0), tgt.max(axis=0)) - origin
    extent = float(np.linalg.norm(span))
    cell = max(1e-9, extent / max(1.0, n_src ** (1.0 / 3.0)))
    src_cells = np.floor((src - origin) / cell).astype(np.int64)
    tgt_cells = np.floor((tgt - origin) / cell).astype(np.int64)
    buckets = {}
    for idx, key in enumerate(map(tuple, src_cells.tolist())):
        buckets.setdefault(key, []).append(idx)
    buckets = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    tgt_groups = {}
    for idx, key in enumerate(map(tuple, tgt_cells.tolist())):
        tgt_groups.setdefault(key, []).append(idx)
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
    order = np.empty(n_tgt, dtype=np.int64)
    distances = np.empty(n_tgt, dtype=np.float64)
    missing = []
    for key, t_list in tgt_groups.items():
        kx, ky, kz = key
        cand_parts = [buckets[(kx + dx, ky + dy, kz + dz)]
                      for dx, dy, dz in offsets
                      if (kx + dx, ky + dy, kz + dz) in buckets]
        t_idx = np.asarray(t_list, dtype=np.int64)
        if not cand_parts:
            missing.append(t_idx)
            continue
        cands = np.concatenate(cand_parts)
        pts_t = tgt[t_idx]
        pts_c = src[cands]
        best_d2 = np.full(len(t_idx), np.inf, dtype=np.float64)
        best_i = np.zeros(len(t_idx), dtype=np.int64)
        for start in range(0, len(cands), 256):
            block = pts_c[start:start + 256]
            diff = pts_t[:, None, :] - block[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            local_i = d2.argmin(axis=1)
            local_d2 = d2[np.arange(len(t_idx)), local_i]
            better = local_d2 < best_d2
            best_d2[better] = local_d2[better]
            best_i[better] = cands[start + local_i[better]]
        order[t_idx] = best_i
        distances[t_idx] = np.sqrt(best_d2)
    if missing:
        m_idx = np.concatenate(missing)
        pts_m = tgt[m_idx]
        best_d2 = np.full(len(m_idx), np.inf, dtype=np.float64)
        best_i = np.zeros(len(m_idx), dtype=np.int64)
        for start in range(0, n_src, 512):
            block = src[start:start + 512]
            diff = pts_m[:, None, :] - block[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            local_i = d2.argmin(axis=1)
            local_d2 = d2[np.arange(len(m_idx)), local_i]
            better = local_d2 < best_d2
            best_d2[better] = local_d2[better]
            best_i[better] = start + local_i[better]
        order[m_idx] = best_i
        distances[m_idx] = np.sqrt(best_d2)
    return distances, order


def _match_by_centroid(source_geo, target_centroids, label):
    """Map target elements onto source volume prims by nearest centroid.

    Element order of an external SOP need not match the HDA's, so bind on
    position rather than index and refuse a match that is not essentially
    exact.
    """
    src_mask = _volume_mask(source_geo)
    src_centroids = _prim_centroids(source_geo, src_mask)
    src_index = np.flatnonzero(src_mask) if src_mask is not None \
        else np.arange(len(src_centroids))
    if len(src_centroids) != len(target_centroids):
        raise hou.NodeError(
            f"{label}: source has {len(src_centroids)} elements, this "
            f"geometry has {len(target_centroids)}")
    try:
        from scipy.spatial import cKDTree
        distances, order = cKDTree(src_centroids).query(target_centroids)
    except ImportError:
        distances, order = _nearest_centroids_numpy(
            src_centroids, target_centroids)
    extent = float(np.linalg.norm(
        target_centroids.max(axis=0) - target_centroids.min(axis=0)))
    tolerance = max(1e-9, 1e-4 * extent)
    if distances.max() > tolerance:
        raise hou.NodeError(
            f"{label}: element centroids do not line up (worst offset "
            f"{distances.max():.3g} > {tolerance:.3g}); the source SOP must "
            f"be the same mesh as this geometry")
    return src_index[order]


def _read_element_attribute(parent, geo, node_path, attrib_name, components,
                            label):
    """Per-element values from a SOP's prim attribute, in branch prim order.

    PolyFEM's ``geometry[].transformation`` moves the mesh but does not
    transform ``materials[].fiber_direction``.  Match an external Fiber SOP
    against the untransformed imported mesh first, and return its vector values
    verbatim.  A transformed-position fallback keeps already-world-space SOPs
    usable without changing their direction components.
    """
    branch = parent.node(f"branch_{geo}")
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    source_node = parent.node(f"geo_{geo}")
    source_mesh = source_node.geometry() if source_node is not None else geo_data
    source_mask = _volume_mask(source_mesh)
    if node_path:
        source = parent.node(node_path) or hou.node(node_path)
        if source is None:
            raise hou.NodeError(f"{label}: SOP '{node_path}' not found")
        source_geo = source.geometry()
    else:
        # Read attributes before the HDA's geometry Transform SOP. Houdini may
        # otherwise rotate vector-typed attributes, while PolyFEM explicitly
        # leaves a0 unchanged by geometry[].transformation.
        source_geo = source_mesh
    attrib = source_geo.findPrimAttrib(attrib_name)
    if attrib is None:
        raise hou.NodeError(
            f"{label}: no primitive attribute '{attrib_name}' on "
            f"{node_path or 'the imported mesh'}")
    size = attrib.size()
    if size != components:
        raise hou.NodeError(
            f"{label}: attribute '{attrib_name}' has {size} component(s), "
            f"expected {components}")
    if attrib.dataType() == hou.attribData.Int:
        values = np.frombuffer(
            source_geo.primIntAttribValuesAsString(attrib_name),
            dtype=np.int32).astype(np.float64)
    else:
        values = np.frombuffer(
            source_geo.primFloatAttribValuesAsString(attrib_name),
            dtype=np.float32).astype(np.float64)
    values = values.reshape(-1, components) if components > 1 else values

    if source_geo is geo_data:
        return values[mask] if mask is not None else values

    raw_order = solver_element_order(source_mesh)[
        np.argsort(solver_element_order(geo_data))]
    if source_geo is source_mesh:
        raw_values = values[source_mask] if source_mask is not None else values
        return raw_values[raw_order]

    raw_targets = _prim_centroids(source_mesh, source_mask)[raw_order]
    try:
        order = _match_by_centroid(source_geo, raw_targets, label)
    except hou.Error as raw_error:
        # Some users deliberately author the source SOP after applying the same
        # world transform. Support that layout too, while still leaving the
        # vector values themselves untouched.
        try:
            order = _match_by_centroid(
                source_geo, _prim_centroids(geo_data, mask), label)
        except hou.Error:
            raise raw_error
    return values[order]


def _element_centroids(parent, geo):
    """Volume-element centroids in Houdini prim order."""
    geo_data = parent.node(f"branch_{geo}").geometry()
    return _prim_centroids(geo_data, _volume_mask(geo_data))


def cylindrical_frame(centroids, origin, axis, label):
    """Per-element (radial, circumferential, longitudinal) unit frame.

    The natural frame for vessels and other tubular tissue: the axis is the
    lumen centreline, and fibers are usually circumferential (optionally as a
    +/-theta pair about the radial direction).
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise hou.NodeError(f"{label}: the cylinder axis has zero length")
    axis = axis / norm

    relative = np.asarray(centroids, dtype=np.float64) - np.asarray(
        origin, dtype=np.float64)
    # einsum, not "@": the BLAS matrix-vector product macOS links (Accelerate)
    # raises spurious divide-by-zero/overflow warnings on finite input.
    along = np.einsum("ij,j->i", relative, axis)
    radial = relative - along[:, None] * axis
    distance = np.linalg.norm(radial, axis=1)

    # Elements sitting on the axis itself have no radial direction. That is a
    # solid cylinder, not a tube -- say so rather than emitting garbage.
    degenerate = np.flatnonzero(distance < 1e-9 * max(1.0, float(distance.max())))
    if len(degenerate):
        raise hou.NodeError(
            f"{label}: {len(degenerate)} element(s) lie on the cylinder axis "
            f"(e.g. {list(degenerate[:10])}), where the radial and "
            f"circumferential directions are undefined. Move the axis outside "
            f"the material, or use a constant/attribute fiber direction.")
    radial = radial / distance[:, None]
    return radial, np.cross(np.tile(axis, (len(radial), 1)), radial), \
        np.tile(axis, (len(radial), 1))


def curve_tangent(parent, centroids, node_path, label):
    """Tangent of the nearest point on a curve, per element.

    The line-of-action frame for tendon, ligament and muscle: draw (or import)
    a centreline curve and every element follows it.
    """
    if not node_path:
        raise hou.NodeError(f"{label}: no curve SOP specified")
    source = parent.node(node_path) or hou.node(node_path)
    if source is None:
        raise hou.NodeError(f"{label}: SOP '{node_path}' not found")
    curve_geo = source.geometry()

    starts, ends = [], []
    for prim in curve_geo.prims():
        points = [v.point().position() for v in prim.vertices()]
        if prim.isClosed() and len(points) > 2:
            points.append(points[0])
        for first, second in zip(points[:-1], points[1:]):
            starts.append(first)
            ends.append(second)
    if not starts:
        raise hou.NodeError(
            f"{label}: '{node_path}' holds no curve segments (it needs a "
            f"polyline, e.g. a Curve or Resample SOP)")

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    segments = ends - starts
    lengths = np.einsum("ij,ij->i", segments, segments)
    keep = lengths > 1e-24
    starts, segments, lengths = starts[keep], segments[keep], lengths[keep]
    if not len(starts):
        raise hou.NodeError(f"{label}: '{node_path}' has zero-length segments only")

    centroids = np.asarray(centroids, dtype=np.float64)
    tangents = np.empty_like(centroids)
    # Chunked so a fine curve against a large mesh cannot blow up memory.
    chunk = max(1, int(4e6 // max(1, len(starts))))
    for begin in range(0, len(centroids), chunk):
        block = centroids[begin:begin + chunk]
        offset = block[:, None, :] - starts[None, :, :]
        t = np.clip(
            np.einsum("ijk,jk->ij", offset, segments) / lengths[None, :], 0, 1)
        closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        deltas = block[:, None, :] - closest
        nearest = np.einsum("ijk,ijk->ij", deltas, deltas).argmin(axis=1)
        tangents[begin:begin + chunk] = segments[nearest]
    return tangents / np.linalg.norm(tangents, axis=1)[:, None]


def _preset_fibers(parent, geo, vol, fam, source, centroids, label):
    """Procedurally generated per-element directions for the fiber presets."""
    if source == "cylindrical":
        radial, circumferential, longitudinal = cylindrical_frame(
            centroids,
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_origin")).eval(),
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_dir")).eval(),
            label)
        component = _slot_parm(parent, geo, vol, fam, "fib_component")
        return {"radial": radial, "circumferential": circumferential,
                "longitudinal": longitudinal}[component]
    return curve_tangent(
        parent, centroids,
        _slot_parm(parent, geo, vol, fam, "fib_curve"), label)


