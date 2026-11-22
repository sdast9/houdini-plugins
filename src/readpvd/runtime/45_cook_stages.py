# =============================================================================
# topology stage
# =============================================================================


def cook_topology(node):
    geo = node.geometry()
    asset = _asset_of(node)
    pvd_path = asset.evalParm("PVD_file")
    if not pvd_path:
        return

    if asset.evalParm("remesh_mode"):
        frame_index = entry_index(asset)
    else:
        frame_index = frame_step(asset, asset.evalParm("topo_frame"))

    mesh = load_block(pvd_path, frame_index, _source_block_token(asset))
    if mesh is None:
        raise hou.NodeError("No readable mesh block in this frame.")

    points = mesh["points"]
    geo.createPoints(np.ascontiguousarray(points, dtype=np.float64))
    # A point id that survives the body filter and the clip (which renumber
    # points), so the probe follows its vertex (review 2026-11-14, item 4).
    geo.addAttrib(hou.attribType.Point, "readpvd_point", 0,
                  create_local_variable=False)
    geo.setPointIntAttribValuesFromString(
        "readpvd_point", np.arange(len(points), dtype=np.int32).tobytes())

    # Coincidence groups: PolyFEM writes a discontinuous mesh (per-element
    # duplicated vertices), so a physical vertex appears as several points.
    # Group ids let the field-smoothing stage average a field back onto shared
    # vertices (nodal recovery); the representative flag marks one point per
    # vertex so glyphs draw once per node. Computed once with the cached
    # topology; stored as float (exact for < 16M groups) for fast binary reads.
    # With body ids the groups stay inside one body, so smoothing never mixes
    # two bodies' values where they touch.
    body = mesh["point_data"].get("body_ids")
    inverse, representative = _coincidence_groups(points, body)
    geo.addAttrib(hou.attribType.Point, "coincident_id", 0.0,
                  create_local_variable=False)
    _set_floats(geo.setPointFloatAttribValuesFromString, "coincident_id",
                inverse)
    geo.addAttrib(hou.attribType.Point, "coincident_rep", 0.0,
                  create_local_variable=False)
    _set_floats(geo.setPointFloatAttribValuesFromString, "coincident_rep",
                representative)

    def put(name, array):
        geo.addArrayAttrib(hou.attribType.Global, name, hou.attribData.Int, 1)
        geo.setGlobalAttribValue(
            name, np.ascontiguousarray(array, dtype=np.int64).ravel().tolist())

    for family in ("tet", "hex", "tri", "quad", "line"):
        if family in mesh["cells"]:
            put(f"{family}_conn", mesh["cells"][family])

    for family in ("tet", "hex"):
        if family in mesh["cells"]:
            put(f"{family}_face_bits", _interior_face_bits(
                mesh["cells"][family], family, inverse, body))

    geo.addAttrib(hou.attribType.Global, "topo_key", "")
    geo.setGlobalAttribValue("topo_key", str(mesh["topo_key"]))


# =============================================================================
# per-frame stage
# =============================================================================


_INITIAL_ORDERING_DATA = "readpvd_initial_ordering"
_INITIAL_ORDERING_NOTES = {
    family: (f"Note: over 25% of higher-order sub-{family}s on step 0 appear "
             "inverted or degenerate; this may indicate non-PolyFEM VTK "
             "Lagrange node ordering or severe initial mesh distortion.")
    for family in ("tet", "hex")
}


def _initial_ordering_key(asset):
    path = asset.evalParm("PVD_file")
    block = _source_block_token(asset)
    try:
        files = _frame_block_paths(path, 0)
        selected = files.get(block)
        if not selected or not os.path.isfile(selected):
            selected = next((f for f in files.values() if os.path.isfile(f)), None)
        return json.dumps((_file_key(selected), block)) if selected else None
    except (OSError, ET.ParseError, VtuParseError, ValueError):
        return None


def _set_availability_status(asset, summary):
    raw = asset.userData(_INITIAL_ORDERING_DATA)
    if raw:
        notes = json.loads(raw).get(_initial_ordering_key(asset), [])
        summary = " ".join((summary, *notes)).strip()
    asset.parm("availability_status").set(summary)
    return summary


def _note_initial_cell_orientation(asset, mesh):
    key = _initial_ordering_key(asset)
    if key is None:
        return
    saved = json.loads(asset.userData(_INITIAL_ORDERING_DATA) or "{}")
    if key in saved:
        return
    points = mesh["points"]
    types, sizes = mesh["cell_types"], mesh["cell_sizes"]
    notes = []
    for family, vtk_types, orders, measure in (
            ("tet", (24, 71), (10, 20, 35),
             lambda p: _tet_volumes(*p.transpose(1, 0, 2))),
            ("hex", (29, 72), (27, 64), hex_volumes)):
        sources = mesh["cell_sources"].get(family)
        higher = np.isin(types, vtk_types) & np.isin(sizes, orders)
        if sources is None or not higher.any():
            continue
        corners = 4 if family == "tet" else 8
        conn = mesh["cell_connectivity"]
        starts = mesh["cell_starts"]
        display = mesh["cells"][family]
        invalid = count = 0
        for start in range(0, len(sources), 16384):
            span = slice(start, start + 16384)
            parents = sources[span]
            selected = higher[parents]
            parent_conn = conn[starts[parents[selected], None] + np.arange(corners)]
            sign = np.sign(measure(points[parent_conn]))
            subcells = measure(points[display[span][selected]])
            invalid += np.count_nonzero(~(subcells * sign > 0))
            count += len(subcells)
        if invalid * 4 > count:
            notes.append(_INITIAL_ORDERING_NOTES[family])
    # Retain the most recent checks so switching blocks restores their notes.
    saved[key] = notes
    asset.setUserData(_INITIAL_ORDERING_DATA, json.dumps(dict(list(saved.items())[-8:])))
    status = asset.evalParm("availability_status")
    for note in _INITIAL_ORDERING_NOTES.values():
        status = status.replace(note, "").strip()
    _set_availability_status(asset, status)


def cook_frame(node):
    geo = node.geometry()  # copy of input geometry (points + prims)
    asset = _asset_of(node)
    pvd_path = asset.evalParm("PVD_file")
    if not pvd_path:
        return

    step = entry_index(asset)
    mesh = load_block(pvd_path, step, _source_block_token(asset))
    if mesh is None:
        raise hou.NodeError("No readable mesh block in this frame.")
    # What this frame was made from, so Refresh can tell a cached frame whose
    # files were rewritten (a rerun into the same folder) or that showed a
    # step the run had not reached yet (review 2026-11-14, items 1 and 2).
    _COOKED_FRAMES.setdefault(asset.sessionId(), {})[
        float(node.evalParm("frame"))] = _frame_signature(pvd_path, step)

    if str(mesh["topo_key"]) != geo.attribValue("topo_key"):
        if asset.evalParm("remesh_mode"):
            # topo stage already follows the frame; key mismatch here means
            # cook ordering hiccup -- force it
            raise hou.NodeError("Topology out of sync; re-cook the node.")
        raise hou.NodeError(
            "Mesh topology changes over time (remeshing run). "
            "Enable 'Remeshing Mode' on this node.")

    n_points = geo.intrinsicValue("pointcount")
    points = mesh["points"]
    if len(points) != n_points:
        raise hou.NodeError("Point count mismatch with topology frame")
    if step == 0:
        _note_initial_cell_orientation(asset, mesh)

    # rest positions + per-frame fields, all bulk binary uploads
    _set_floats(geo.setPointFloatAttribValuesFromString, "P", points)

    def upload(attrib_type, find, set_from_string, name, arr):
        arr = np.asarray(arr)
        components = 1 if arr.ndim == 1 else arr.shape[1]
        if components not in (1, 2, 3, 9):
            return
        if components in (2, 3):
            vec = np.zeros((len(arr), 3))
            vec[:, :components] = arr
            arr = vec
        if find(name) is None:
            default = 0.0 if components == 1 else tuple(
                0.0 for _ in range(3 if components in (2, 3) else 9))
            geo.addAttrib(attrib_type, name, default,
                          create_local_variable=False)
        _set_floats(set_from_string, name, arr)

    def upload_point(name, arr):
        upload(hou.attribType.Point, geo.findPointAttrib,
               geo.setPointFloatAttribValuesFromString, name, arr)

    def upload_primitive(name, arr):
        upload(hou.attribType.Prim, geo.findPrimAttrib,
               geo.setPrimFloatAttribValuesFromString, name, arr)

    for raw_name, data in mesh["point_data"].items():
        name = _field_name(raw_name)
        arr = np.asarray(data)
        # some fields (e.g. *_avg) omit obstacle vertices: zero-pad to the
        # point count (matches the implicit zero-fill of the 0.26 viewer)
        if len(arr) != n_points:
            padded = np.zeros((n_points,) + arr.shape[1:], dtype=np.float64)
            padded[:min(len(arr), n_points)] = arr[:n_points]
            arr = padded
        upload_point(name, arr)

    # Preserve VTK CellData on Houdini primitives. A point-averaged copy is
    # added only when no PointData field has the same name, allowing the
    # existing point-coloring and probing tools to display cell fields while
    # retaining the original un-interpolated primitive values.
    family_attrib = geo.findPrimAttrib("source_cell_family")
    index_attrib = geo.findPrimAttrib("source_cell_index")
    if mesh.get("cell_data") and family_attrib is not None and index_attrib is not None:
        prim_families = np.asarray(geo.primStringAttribValues(
            "source_cell_family"), dtype=object)
        prim_indices = np.asarray(geo.primIntAttribValues(
            "source_cell_index"), dtype=np.int64)
        for raw_name, by_family in mesh["cell_data"].items():
            name = _field_name(raw_name)
            sample = next(iter(by_family.values()), None)
            if sample is None:
                continue
            sample = np.asarray(sample)
            shape = () if sample.ndim == 1 else (sample.shape[1],)
            prim_values = np.zeros((len(prim_indices),) + shape,
                                   dtype=np.float64)
            point_sum = np.zeros((n_points,) + shape, dtype=np.float64)
            point_count = np.zeros(n_points, dtype=np.float64)
            for family, values in by_family.items():
                values = np.asarray(values)
                mask = prim_families == family
                if np.any(mask):
                    prim_values[mask] = values[prim_indices[mask]]
                conn = mesh["cells"].get(family)
                if conn is None:
                    continue
                for corner in range(conn.shape[1]):
                    np.add.at(point_sum, conn[:, corner], values)
                    np.add.at(point_count, conn[:, corner], 1.0)
            upload_primitive(name, prim_values)
            if geo.findPointAttrib(name) is None:
                divisor = np.maximum(point_count, 1.0)
                if point_sum.ndim > 1:
                    divisor = divisor[:, None]
                upload_point(name, point_sum / divisor)

    # PolyFEM exports fiber directions as three scalar fields per model; make
    # them a vector so they can be drawn and probed as one thing.
    families = []
    for prefix in _fiber_prefixes(mesh["point_data"]):
        columns = [np.asarray(mesh["point_data"][f"{prefix}fiber_direction_{a}"])
                   for a in "xyz"]
        if any(len(c) != n_points for c in columns):
            continue
        name = _fiber_attrib(prefix, raw=True)
        upload_point(name, np.stack([c.ravel() for c in columns], axis=1))
        families.append(name)
        _DISPLAY_NAMES.setdefault(name, prefix + "fiber_direction")

    # Minimal PolyFEM output may intentionally omit material fields. In that
    # case, use the companion written by the preprocessing HDA. Native solver
    # fields remain authoritative whenever they are present.
    if not families:
        for name, label, values in _companion_point_fibers(pvd_path, mesh):
            upload_point(name, values)
            families.append(name)
            _DISPLAY_NAMES.setdefault(name, label)

    # Sanitized attribute name -> original vtu name, for anything reading the
    # cooked geometry (probe, downstream tools) rather than the module cache.
    if geo.findGlobalAttrib("readpvd_display_names") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_display_names", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue("readpvd_display_names",
                             json.dumps(_DISPLAY_NAMES, sort_keys=True))
    if geo.findGlobalAttrib("readpvd_fiber_families") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_fiber_families", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue("readpvd_fiber_families", " ".join(families))
    if geo.findGlobalAttrib("readpvd_fiber_reference_frame") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_fiber_reference_frame", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue(
        "readpvd_fiber_reference_frame",
        "simulation_world_reference_a0" if families else "")
    # Measure this cooked frame's size right here so cache_frame_limit enforces
    # the memory budget during forward scrubbing (cache1 evaluates maxframes
    # while frame_data is dirty for the new frame, so reading needsToCook()
    # there never saw the first frame's size).
    try:
        frame_bytes = int(geo.intrinsicValue("memoryusage"))
        if frame_bytes > 0:
            _CACHE_FRAME_BYTES[asset.sessionId()] = frame_bytes
    except (hou.Error, TypeError, ValueError):
        pass


