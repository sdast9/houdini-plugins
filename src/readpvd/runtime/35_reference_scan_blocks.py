def _scan_reduced(node, mesh, field, reduction):
    """Per-point displayed scalar for one mesh -- mirrors the color VEX."""
    raw = _mesh_field(mesh, field)
    if raw is None:
        return None
    return np.asarray(_reduce_array(raw, reduction), dtype=np.float64)


_REFERENCE_CACHE = OrderedDict()
_REFERENCE_CACHE_MAX_BYTES = 128 << 20
_REFERENCE_CACHE_SIZE = 8


def _reference_reduced(node, path, block, field, reduction):
    """Reference topology and reduced field, or (None, None)."""
    if not node.evalParm("reference_enable"):
        return None, None
    step = frame_step(node, node.evalParm("reference_frame"))
    pvd_key = (_file_key(path), _frame_signature(path, step))
    prefix = (pvd_key, step, block, field, reduction)
    # Look up the last topology for this exact on-disk frame identity before
    # loading a mesh. A VTU larger than the parser's LRU could evict it.
    cached = next((key for key in reversed(_REFERENCE_CACHE)
                   if key[:-1] == prefix), None)
    if cached is not None:
        _REFERENCE_CACHE.move_to_end(cached)
        return cached[-1], _REFERENCE_CACHE[cached]
    mesh = load_block(path, step, block, fallback=False)
    if mesh is None:
        return None, None
    topology = mesh["topo_key"]
    values = _scan_reduced(node, mesh, field, reduction)
    if values is not None and values.nbytes <= _REFERENCE_CACHE_MAX_BYTES:
        # Own the reduced array so a scalar view cannot pin a full field.
        _REFERENCE_CACHE[(*prefix, topology)] = values.copy()
        while (len(_REFERENCE_CACHE) > _REFERENCE_CACHE_SIZE
               or sum(v.nbytes for v in _REFERENCE_CACHE.values())
               > _REFERENCE_CACHE_MAX_BYTES):
            _REFERENCE_CACHE.popitem(last=False)
    return topology, values


def _percent_delta(values, reference):
    finite = np.isfinite(values) & np.isfinite(reference)
    scale = max(np.max(np.abs(values), where=np.isfinite(values), initial=0.0),
                np.max(np.abs(reference), where=np.isfinite(reference), initial=0.0))
    tol = 1e-12 * scale
    delta = values - reference
    baseline = np.abs(reference)
    near_zero = np.isfinite(reference) & (baseline <= tol)
    percent = np.full(len(values), np.nan)
    percent[finite & near_zero & (np.abs(delta) <= tol)] = 0.0
    np.divide(delta, baseline, out=percent, where=finite & ~near_zero)
    percent *= 100.0
    return percent, int(np.count_nonzero(near_zero))


def _apply_reference(values, reference_reduced, mode):
    """Apply the reference comparison the color VEX would (difference/abs/%)."""
    if reference_reduced is None or len(reference_reduced) != len(values):
        return values
    if mode == "percent":
        return _percent_delta(values, reference_reduced)[0]
    delta = values - reference_reduced
    return np.abs(delta) if mode == "absolute" else delta


def _coincidence_groups(points, body=None):
    """Return (group id per point, representative flag) for coincident nodes.

    PolyFEM writes per-element duplicated vertices; `inverse` labels which
    physical vertex each point is, and `representative` is 1 for exactly one
    point per physical vertex (used to draw a single glyph per node). With
    PolyFEM body ids, points of different bodies are never one vertex: two
    bodies touching (or sharing an interface) keep their own values when the
    display averages over a vertex.
    """
    keys = np.round(np.asarray(points, dtype=np.float64), 9)
    if body is not None and len(body) == len(keys):
        keys = np.column_stack((keys, np.rint(np.asarray(
            body, dtype=np.float64).reshape(len(keys), -1)[:, 0])))
    _, index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    representative = np.zeros(len(points), dtype=np.float64)
    representative[index] = 1.0
    return inverse, representative


# Element faces in the order the boundary_display wrangle emits them: the
# build_prims vertex order of each volume prim, indexed by Houdini's
# tet_faceindex / hex_faceindex tables.
_FACE_TABLES = {
    "tet": (np.array([1, 2, 3, 0]),
            np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])),
    "hex": (np.array([0, 1, 3, 2, 4, 5, 7, 6]),
            np.array([[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4],
                      [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]])),
}


def _interior_face_bits(conn, family, inverse, body=None):
    """Per cell, bit f set when face f is shared with another cell.

    PolyFEM gives every element its own copy of its vertices, so faces are
    matched through the coincidence groups (and the body id, so an interface
    between two bodies stays visible when one of them is hidden).
    """
    prim_order, face_index = _FACE_TABLES[family]
    conn = np.asarray(conn, dtype=np.int64)
    if not len(conn):
        return np.zeros(0, dtype=np.int64)
    faces = conn[:, prim_order][:, face_index]          # (cells, faces, k)
    cells, n_faces, _ = faces.shape
    keys = np.sort(inverse[faces].reshape(cells * n_faces, -1), axis=1)
    if body is not None and len(body) == len(inverse):
        body = np.rint(np.asarray(body, dtype=np.float64).ravel())
        keys = np.column_stack((keys, body[faces[:, :, 0].ravel()]))
    order = np.lexsort(keys.T[::-1])
    ordered = keys[order]
    same = np.all(ordered[1:] == ordered[:-1], axis=1)
    shared = np.zeros(len(keys), dtype=bool)
    shared[1:] |= same
    shared[:-1] |= same
    interior = np.empty(len(keys), dtype=bool)
    interior[order] = shared
    weights = np.left_shift(1, np.arange(n_faces, dtype=np.int64))
    return (interior.reshape(cells, n_faces) * weights).sum(axis=1)


def _coincidence_inverse(points, body=None):
    """Group id per point for coincident (duplicated) mesh vertices."""
    return _coincidence_groups(points, body)[0]


def _nodal_average(values, inverse):
    """Average `values` over coincidence groups (FEM nodal recovery)."""
    finite = np.isfinite(values)
    size = int(inverse.max()) + 1 if len(inverse) else 0
    sums = np.bincount(inverse[finite], weights=values[finite], minlength=size)
    counts = np.bincount(inverse[finite], minlength=size)
    averaged = np.divide(sums, counts, out=np.full(size, np.nan, np.float64),
                         where=counts > 0)
    return averaged[inverse]


# Memoized results of the last all-frames scan (Auto Range: All Frames), so a
# repeat is instant. Keyed on everything that changes the per-frame
# value; a settings or file change invalidates it. Bounded so a huge sequence
# is recomputed rather than pinned in RAM.
_SCAN_CACHE = {}
_SCAN_CACHE_MAX_BYTES = 64 << 20
_SCAN_CACHE_MAX_SINGLE_ROW_BYTES = 96 << 20
_SCAN_CACHE_MAX_ROWS = 32768
_RANGE_SAMPLE_MAX_VALUES = (128 << 20) // np.dtype(np.float64).itemsize
_RANGE_CHUNK_VALUES = 65536


def _scan_key(node, path):
    try:
        file_key = _file_key(path)
    except Exception:
        file_key = path
    return (file_key, node.evalParm("data_epoch"), node.evalParm("color_attrib"),
            _menu_token(node, "color_reduction"),
            bool(node.evalParm("smooth_field")), _source_block_token(node),
            bool(node.evalParm("reference_enable")),
            frame_step(node, node.evalParm("reference_frame")),
            _menu_token(node, "reference_mode"),
            node.evalParm("visible_bodies").strip())


def _scan_displayed_values(node, path, cache_rows=True):
    """Yield (frame, timestep, values) of the displayed scalar for every frame,
    reusing the cached result when the scan settings and PVD file are
    unchanged. See _scan_frames for the (uncached) computation."""
    key = _scan_key(node, path)
    if _SCAN_CACHE.get("key") == key:
        for row in _SCAN_CACHE["rows"]:
            yield row
        return
    _SCAN_CACHE.clear()
    rows = [] if cache_rows else None
    total = 0
    for frame, timestep, values in _scan_frames(node, path):
        if rows is not None:
            size = 0 if values is None else values.size * np.dtype(np.float64).itemsize
            total += size
            # Cache a single dense frame under its own hard cap. Sequences
            # with multiple rows use the smaller aggregate budget so they
            # leave more room for the live input frame.
            if (size > _SCAN_CACHE_MAX_SINGLE_ROW_BYTES
                    or (rows and total > _SCAN_CACHE_MAX_BYTES)
                    or len(rows) >= _SCAN_CACHE_MAX_ROWS):
                rows = None
            else:
                rows.append((frame, timestep,
                             None if values is None else values.astype(np.float64)))
        yield frame, timestep, values
    if rows is not None:
        _SCAN_CACHE["key"] = key
        _SCAN_CACHE["rows"] = rows
    else:
        _SCAN_CACHE.clear()


def _scan_frames(node, path, frames=None):
    """Compute (frame, timestep, values) per frame -- pure numpy, an exact
    mirror of the color stage (reduction, reference comparison, smoothing
    `_avg` preference, nodal averaging, body visibility). No network cook and
    no frame change, so the range agrees with the viewport. `frames` limits
    the scan to some frame numbers (default: every frame).
    """
    entries = read_pvd(path)
    field = node.evalParm("color_attrib")
    reduction = _menu_token(node, "color_reduction")
    block = _source_block_token(node)
    comparing = bool(node.evalParm("reference_enable"))
    try:
        reference_topo, reference_reduced = _reference_reduced(
            node, path, block, field, reduction)
    except (OSError, ET.ParseError, VtuParseError, ValueError):
        reference_topo, reference_reduced = None, None
    reference_mode = _menu_token(node, "reference_mode")
    smooth = bool(node.evalParm("smooth_field"))
    visible = _visible_body_set(node)
    inverse = None
    geometry_key = None
    for frame in (range(len(entries)) if frames is None else frames):
        timestep = entries[frame][0]
        try:
            mesh = load_block(path, frame, block)
        except Exception:
            yield frame, timestep, None
            continue
        if mesh is None:
            yield frame, timestep, None
            continue
        matched = (comparing and reference_reduced is not None
                   and mesh["topo_key"] == reference_topo
                   and len(mesh["points"]) == len(reference_reduced))
        if comparing and not matched and frames is None:
            yield frame, timestep, None
            continue
        if matched:
            values = _scan_reduced(node, mesh, field, reduction)
            if values is not None:
                values = _apply_reference(
                    values, reference_reduced, reference_mode)
        else:
            # smoothing prefers PolyFEM's continuous _avg field where present
            effective = field
            if smooth:
                candidate = field + "_avg"
                names = {_field_name(name) for name in mesh["point_data"]}
                if candidate in names:
                    effective = candidate
            values = _scan_reduced(node, mesh, effective, reduction)
        if values is not None and smooth:
            key = _rest_geometry_digest(mesh)
            if key != geometry_key:
                inverse = _coincidence_inverse(
                    mesh["points"], mesh["point_data"].get("body_ids"))
                geometry_key = key
            values = _nodal_average(values, inverse)
        # Restrict the range to the bodies that are displayed.
        if values is not None and visible is not None:
            body = mesh["point_data"].get("body_ids")
            if body is not None:
                shown = np.isin(np.rint(np.asarray(body).ravel()), list(visible))
                values = np.where(shown, values, np.nan)
        yield frame, timestep, values


def _build_block_geometry(mesh, block_name, slug, colors, values, components):
    """Build one supplementary block in a standalone geometry (bulk attrs)."""
    block = hou.Geometry()
    positions = np.ascontiguousarray(mesh["points"], dtype=np.float64)
    block.createPoints(positions.tolist())
    n = len(positions)

    block.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                    create_local_variable=False)
    _set_floats(block.setPointFloatAttribValuesFromString, "Cd", colors)
    block.addAttrib(hou.attribType.Point, "pvd_block", "")
    block.setPointStringAttribValues("pvd_block", (block_name,) * n)

    if values is not None:
        name = f"{slug}_{mesh['_field_name']}"
        if components == 1:
            block.addAttrib(hou.attribType.Point, name, 0.0,
                            create_local_variable=False)
            arr = np.asarray(values, dtype=np.float64)
        else:
            width = 3 if components in (2, 3) else components
            block.addAttrib(hou.attribType.Point, name,
                            tuple(0.0 for _ in range(width)),
                            create_local_variable=False)
            arr = np.zeros((n, width), dtype=np.float64)
            arr[:, :components] = np.asarray(values, dtype=np.float64
                                             ).reshape(n, components)
        _set_floats(block.setPointFloatAttribValuesFromString, name, arr)

    points = block.points()
    prim_group = block.createPrimGroup(f"readpvd_block_{slug}")
    point_group = block.createPointGroup(f"readpvd_block_{slug}")
    point_group.add(points)
    for family, connectivity in mesh["cells"].items():
        # VTK corners into Houdini's vertex order, as build_prims does
        # (_FACE_TABLES): positive tets, untwisted hexes
        if family == "tet":
            for indices in connectivity:
                prim_group.add(block.createTetrahedronInPlace(
                    *(points[int(indices[k])] for k in (1, 2, 3, 0))))
        elif family == "hex":
            for indices in connectivity:
                prim_group.add(block.createHexahedronInPlace(
                    *(points[int(indices[k])]
                      for k in (0, 1, 3, 2, 4, 5, 7, 6))))
        elif family == "line":
            for indices in connectivity:
                prim = block.createPolygon(is_closed=False)
                for i in indices:
                    prim.addVertex(points[int(i)])
                prim_group.add(prim)
        elif family == "point":
            continue  # points already exist; 1-vertex polygons are invalid
        else:  # tri / quad: bulk creation (>=3 verts)
            for prim in block.createPolygons(
                    tuple(tuple(int(i) for i in indices)
                          for indices in connectivity)):
                prim_group.add(prim)
    block.addAttrib(hou.attribType.Prim, "pvd_block", "")
    block.setPrimStringAttribValues(
        "pvd_block", (block_name,) * len(block.prims()))
    return block


def _sample_ramp_vectorized(ramp, normalized, finite, invalid_rgb,
                            lut_size=513):
    """Evaluate a Houdini color ramp over a NumPy array via a uniform LUT.

    Avoids calling ramp.lookup() in a Python loop per vertex.
    """
    n = len(normalized)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    lut = np.asarray(
        [ramp.lookup(i / (lut_size - 1)) for i in range(lut_size)],
        dtype=np.float64)
    scaled = np.where(finite, np.clip(normalized, 0.0, 1.0), 0.0) * (lut_size - 1)
    idx0 = np.floor(scaled).astype(np.intp)
    idx1 = np.minimum(idx0 + 1, lut_size - 1)
    frac = (scaled - idx0)[:, None]
    colors = lut[idx0] * (1.0 - frac) + lut[idx1] * frac
    if not np.all(finite):
        colors[~finite] = invalid_rgb
    return colors


def cook_multi_blocks(node):
    """Generate optional supplementary PVD blocks with independent colors."""
    geo = node.geometry()
    asset = node.parent()
    path = asset.evalParm("PVD_file")
    if not path:
        return
    primary = _source_block_token(asset)
    # Only parse if at least one supplementary block is actually requested.
    wanted = [name for name in ("Volume", "Surface", "Contact", "Points")
              if name != primary
              and asset.parm(f"multi_{name.lower()}_show") is not None
              and asset.evalParm(f"multi_{name.lower()}_show")]
    if not wanted:
        return
    blocks = load_frame(path, entry_index(asset), blocks=wanted)
    invalid_rgb = tuple(float(c) for c in asset.evalParmTuple("invalid_color"))

    for block_name, mesh in blocks.items():
        if block_name not in wanted:
            continue
        slug = block_name.lower()
        positions = np.asarray(mesh["points"], dtype=np.float64)
        solution = _solution_vectors(mesh) if asset.evalParm("show_deformed") else None
        if solution is not None:
            positions = positions + solution
        # _build_block_geometry reads positions back from the mesh dict; keep
        # the (possibly deformed) copy local without mutating the cache.
        mesh = dict(mesh, points=positions)

        field = asset.evalParm(f"multi_{slug}_field")
        values = _mesh_point_field(mesh, field) if field else None
        components = (0 if values is None else
                      1 if np.asarray(values).ndim == 1 else values.shape[1])
        if values is None:
            colors = np.ones((len(positions), 3), dtype=np.float64)
        else:
            reduced = np.asarray(_reduce_array(
                values, _menu_token(asset, f"multi_{slug}_reduction")),
                dtype=np.float64)
            low, high = asset.evalParmTuple(f"multi_{slug}_range")
            denom = high - low
            finite = np.isfinite(reduced)
            if abs(denom) < 1e-20:
                normalized = np.full(len(reduced), 0.5, dtype=np.float64)
            else:
                normalized = np.clip((reduced - low) / denom, 0.0, 1.0)
            ramp = asset.parm(f"multi_{slug}_ramp").evalAsRamp()
            colors = _sample_ramp_vectorized(
                ramp, normalized, finite, invalid_rgb)
        mesh["_field_name"] = field
        geo.merge(_build_block_geometry(
            mesh, block_name, slug, colors, values, components))


