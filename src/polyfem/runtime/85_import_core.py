# =============================================================================
# read_params + legacy import
# =============================================================================


def _looks_legacy(data):
    solver = data.get("solver")
    contact = solver.get("contact", {}) if isinstance(solver, dict) else {}
    if isinstance(contact, dict) and "adaptive_barrier_stiffness_multiplier" in contact:
        return True
    materials = data.get("materials")
    if isinstance(materials, dict):
        return True
    for material in materials if isinstance(materials, list) else []:
        if not isinstance(material, dict):
            continue
        try:
            if int(material.get("id", 0)) < 1000:
                return True
        except (TypeError, ValueError):
            continue
    bc = data.get("boundary_conditions", {})
    if isinstance(bc, dict):
        for key in _BC_LIST_KEYS:
            entries = bc.get(key, [])
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                try:
                    sid = int(entry.get("id", 0))
                except (TypeError, ValueError):
                    continue
                if sid != 0 and abs(sid) < 10000000:
                    return True
    return False


def _restore_remesh(parent, data):
    """Restore /space/remesh, filling omitted optional values from the spec."""
    space = data.get("space", {})
    if isinstance(space, list):
        space = next((entry for entry in space if isinstance(entry, dict)), {})
    remesh = space.get("remesh") if isinstance(space, dict) else None
    if not isinstance(remesh, dict):
        remesh = {}

    parms = {
        "remeshing_enabled": bool(
            remesh.get("enabled", REMESH_DEFAULTS["enabled"])),
        "remesh_type": 1 if remesh.get(
            "type", REMESH_DEFAULTS["type"]) == "sizing_field" else 0,
    }

    def restore_stage(json_name, enabled_parm, fields):
        stage = remesh.get(json_name, {})
        if not isinstance(stage, dict):
            stage = {}
        defaults = REMESH_DEFAULTS[json_name]
        parms[enabled_parm] = bool(stage.get("enabled", defaults["enabled"]))
        for key, parm in fields:
            parms[parm] = stage.get(key, defaults[key])

    restore_stage("split", "split_enabled", (
        ("acceptance_tolerance", "split_acceptance_tol"),
        ("culling_threshold", "split_culling_threshold"),
        ("max_depth", "split_max_depth"),
        ("min_edge_length", "min_edge_length")))
    restore_stage("collapse", "collapse_enabled", (
        ("acceptance_tolerance", "collapse_acceptance_tol"),
        ("culling_threshold", "collapse_culling_threshold"),
        ("max_depth", "collapse_max_depth"),
        ("rel_max_edge_length", "rel_max_edge_length"),
        ("abs_max_edge_length", "abs_max_edge_length")))
    restore_stage("swap", "swap_enabled2", (
        ("acceptance_tolerance", "swap_acceptance_tol2"),
        ("max_depth", "swap_max_depth2")))
    restore_stage("smooth", "smooth_enabled", (
        ("acceptance_tolerance", "smooth_acceptance_tol"),
        ("max_iters", "smooth_max_iters")))

    local = remesh.get("local_relaxation", {})
    if not isinstance(local, dict):
        local = {}
    for key, parm in (
            ("local_mesh_n_ring", "local_mesh_n_ring"),
            ("local_mesh_rel_area", "local_mesh_rel_area"),
            ("max_nl_iterations", "max_nl_iterations")):
        parms[parm] = local.get(key, REMESH_DEFAULTS["local_relaxation"][key])

    parent.setParms(parms)


def _format_vector(value):
    """JSON value -> HDA vector/scalar parm string (round-trips expressions)."""
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(v) for v in value) + "]"
    return json.dumps(value)


def _decode_boundary_id(bid):
    """Inverse of boundary_id: -> (geo, vol, sideset, bc)."""
    vid, rem = divmod(int(bid), 10000)
    sideset, bc = divmod(rem, 100)
    return vid // 1000, vid % 1000, sideset, bc


def _ensure_sideset(parent, geo, vol, j):
    parm = parent.parm(f"sideset_selection{geo}_{vol}")
    if parm is not None and parm.eval() < j:
        parm.set(int(j))


def _restore_obstacle_tables(parent, geo, entry, line, notes):
    """An obstacle moved by displacement tables (a keyed translation when
    the asset wrote it): keys on its Translate at the tables' points."""
    value = entry.get("value")
    if not (isinstance(value, list) and len(value) == 3 and all(
            isinstance(v, (int, float)) for v in value)):
        notes.append(f"Obstacle {geo}: its displacement has a time function "
                     "on an expression; the time function was not restored.")
        return False
    if line is None:
        notes.append(f"Obstacle {geo}: its motion was not restored (the Time "
                     "tab defines no run).")
        return False
    interpolation = entry["interpolation"]
    if isinstance(interpolation, list) and len(interpolation) == 2:
        notes.append(f"Obstacle {geo}: two interpolation tables were supplied "
                     "for three axes. The third axis uses a constant multiplier.")
    functions, note = _interpolation_functions(interpolation, 3, line)
    if note:
        notes.append(f"Obstacle {geo}: {note}")
    for axis, (component, function) in enumerate(zip("xyz", functions)):
        parm = parent.parm(f"xform_t__{geo}{component}")
        rest = parm.eval()
        parm.deleteAllKeyframes()
        if function is None:
            points, values = [line.t0], [1.0]
        else:
            points, values = function.points, function.values
        keys = []
        for point, factor in zip(points, values):
            key = hou.Keyframe()
            key.setFrame(line.frame(point))
            key.setValue(rest + float(value[axis]) * float(factor))
            key.setExpression("linear()", hou.exprLanguage.Hscript)
            keys.append(key)
        parm.setKeyframes(keys)
    return True


def _restore_conditions(parent, data, volume_remap=None, scene=None,
                        notes=None, geometry=()):
    """Initial conditions, body force and obstacle motion from new-format
    json (sideset conditions are restored by _restore_sidesets). An obstacle
    the asset animated gets its transform keyframes back from the scene
    record; displacement tables without one become keys on Translate."""
    ic = data.get("initial_conditions", {})
    counts, entries = {}, []
    for type_idx, key in enumerate(("solution", "velocity", "acceleration")):
        for entry in ic.get(key, []):
            mid = int(entry.get("id", 0))
            geo, vol = mid // 1000, mid % 1000
            vol = (volume_remap or {}).get(geo, {}).get(vol, vol)
            l = counts.get((geo, vol), 0) + 1
            counts[(geo, vol)] = l
            entries.append((geo, vol, l, type_idx, entry.get("value")))
    for (geo, vol), n in counts.items():
        parm = parent.parm(f"initial_conditions{geo}_{vol}")
        if parm is not None:
            parm.set(n)
    for geo, vol, l, type_idx, value in entries:
        if parent.parm(f"conditiontype{geo}_{vol}_{l}") is None:
            continue
        parent.setParms({f"conditiontype{geo}_{vol}_{l}": type_idx})
        _set_expr_vector(parent, f"ic_vec{geo}_{vol}_{l}",
                         f"condition_vector_{geo}_{vol}_{l}", value)

    _restore_obstacle_conditions(parent, data, scene, notes, geometry)


def _restore_body_load(parent, data, notes):
    bc = data.get("boundary_conditions")
    if not isinstance(bc, dict) or "rhs" not in bc:
        return
    rhs = bc["rhs"]
    if isinstance(rhs, list) and any(isinstance(item, dict) for item in rhs):
        notes.append("Per-body boundary_conditions.rhs is not representable "
                     "by the global Body Load boxes; retained in the source JSON.")
    else:
        _set_expr_vector(parent, "rhs_vec", "RHS", rhs)


def _restore_obstacle_conditions(parent, data, scene, notes, geometry=()):
    notes = notes if notes is not None else []
    bc = data.get("boundary_conditions")
    if not isinstance(bc, dict):
        bc = {}
    recorded = (scene or {}).get("obstacles", {})
    for geo_text, obstacle in recorded.items():
        if obstacle.get("motion", "static") == "static":
            continue
        for name, keys in obstacle.get("transform", {}).items():
            _restore_keyframes(parent.parm(name), keys)
    # Legacy projects assigned decimal IDs (for example geometry 2's
    # obstacle was 100021). Its surface selection records the exact ID.
    obstacle_ids = {
        item["surface_selection"]: geo
        for geo, item in enumerate(geometry, 1)
        if item.get("is_obstacle") and isinstance(item.get("surface_selection"), int)
    }
    line = None
    for entry in bc.get("obstacle_displacements", []):
        if not isinstance(entry, dict):
            continue
        sid = int(entry.get("id", 0))
        geo = obstacle_ids.get(sid, (sid - 100000) // 1000)
        parm = parent.parm(f"is_obstacle{geo}")
        if parm is None or not parm.eval():
            notes.append(f"Obstacle displacement id {sid} could not be mapped.")
            continue
        if entry.get("interpolation") is None:
            _set_expr_vector(parent, f"obstacle_disp_vec{geo}",
                             f"obstacle_disp{geo}", entry.get("value"))
            continue
        if recorded.get(str(geo), {}).get("motion") == "translation":
            continue  # the record's keys say it exactly
        if line is None:
            try:
                line = timeline(parent)
            except hou.Error:
                line = None
        try:
            _restore_obstacle_tables(parent, geo, entry, line, notes)
        except (hou.Error, KeyError, TypeError, ValueError) as exc:
            notes.append(f"Obstacle {geo}: its motion was not restored "
                         f"({exc}).")


def _native_pattern(entry):
    """Inverse of parse_native_selection: json entry -> UI pattern string."""
    try:
        if "axis" in entry:
            return f"axis:{entry['axis']}:{entry['position']}"
        if "box" in entry:
            lo, hi = entry["box"]
            return f"box:{list(lo)},{list(hi)}"
        if "center" in entry:
            return f"sphere:{list(entry['center'])},{entry['radius']}"
        if "normal" in entry:
            return f"plane:{list(entry['normal'])},{list(entry['point'])}"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _surface_lookups(parent, geo):
    """Map PolyFEM vertex numbers back to the displayed surface.

    Returns (face_lookup, point_lookup): {sorted vertex numbers of a face:
    surface prim} and {vertex number: surface point}, using the same
    numbering as the export (mesh_vertex_ids).
    """
    surf_geo = parent.node(f"null_{geo}").geometry()
    _, face_keys = _surface_arrays(surf_geo)
    vid_of_key, _ = mesh_vertex_ids(parent, geo)
    count = len(vid_of_key)
    face_lookup = {}
    for prim, row in enumerate(face_keys):
        keys = [int(key) for key in row if key >= 0]
        if keys and all(key < count for key in keys):
            face_lookup[tuple(sorted(int(vid_of_key[key])
                                     for key in keys))] = prim
    point_keys = np.frombuffer(
        surf_geo.pointIntAttribValuesAsString("msh_pt_id"),
        dtype=np.int32).astype(np.int64) - 1
    point_lookup = {int(vid_of_key[key]): point
                    for point, key in enumerate(point_keys)
                    if 0 <= key < count}
    return face_lookup, point_lookup


def _restore_sidesets(parent, data, geometry, input_dir, resource_dir,
                      volume_remap=None):
    """Sidesets and their boundary conditions from a scene this asset wrote.

    Faces and nodes come back from the selection files, through PolyFEM's
    vertex numbering and the surface provenance. When the asset's record
    (hda_sidesets.json) still matches params.json, every sideset is restored
    as it was entered: "*", typed selections, group type and each condition.
    Otherwise the conditions are rebuilt from params.json: an older export's
    per-condition ids keep their slots, and a combined id of overlapping
    sidesets becomes a sideset of its own with the same faces and the merged
    conditions, so the solver input is unchanged. Returns notes for the
    import report.
    """
    notes = []
    record, reason = load_sideset_record(input_dir, data, resource_dir)
    if reason:
        notes.append(f"Sidesets: {reason}.")
    conditions = data.get("boundary_conditions", {})
    if not isinstance(conditions, dict):
        conditions = {}

    for i, g in enumerate(geometry, 1):
        if g.get("is_obstacle") or parent.node(f"null_{i}") is None:
            continue
        try:
            face_lookup, point_lookup = _surface_lookups(parent, i)
        except hou.Error as exc:
            notes.append(f"geo {i}: sidesets were not restored ({exc})")
            continue
        faces_by_id, points_by_id, natives = {}, {}, {}
        selections = g.get("surface_selection", [])
        if not isinstance(selections, list):
            notes.append(f"geo {i}: non-list surface_selection")
            selections = []
        for entry in selections:
            if not isinstance(entry, dict):
                notes.append(f"geo {i}: integer selection {entry} "
                             "(assign via sidesets instead)")
                continue
            if "file" in entry:
                try:
                    rows = np.loadtxt(_selection_file(entry, resource_dir),
                                      dtype=np.int64, ndmin=2)
                except (OSError, ValueError):
                    notes.append(f"geo {i}: unreadable {entry['file']}")
                    continue
                unmatched = 0
                for row in rows:
                    prim = face_lookup.get(
                        tuple(sorted(int(v) for v in row[1:])))
                    if prim is None:
                        unmatched += 1
                        continue
                    faces_by_id.setdefault(int(row[0]), set()).add(prim)
                if unmatched:
                    notes.append(
                        f"geo {i}: {unmatched} faces of {entry['file']} not "
                        "on the current mesh")
            else:
                pattern = _native_pattern(entry)
                if pattern is None or "id" not in entry:
                    notes.append(f"geo {i}: unrecognized selection entry")
                    continue
                natives.setdefault(int(entry["id"]), pattern)
        points = g.get("point_selection", [])
        for entry in points if isinstance(points, list) else []:
            if not (isinstance(entry, dict) and "file" in entry):
                continue
            try:
                rows = np.loadtxt(_selection_file(entry, resource_dir),
                                  dtype=np.int64, ndmin=2)
            except (OSError, ValueError):
                notes.append(f"geo {i}: unreadable {entry['file']}")
                continue
            for row in rows:
                point = point_lookup.get(int(row[1]))
                if point is not None:
                    points_by_id.setdefault(int(row[0]), set()).add(point)

        geo_record = None
        if record is not None:
            geo_record = record.get("geometries", {}).get(str(i))
        remap = (volume_remap or {}).get(i, {})
        if geo_record is not None:
            _restore_recorded_sidesets(parent, i, geo_record, faces_by_id,
                                       points_by_id, notes, remap)
        else:
            try:
                line = timeline(parent)
            except hou.Error:
                line = None
            _restore_sidesets_from_ids(parent, i, conditions, faces_by_id,
                                       points_by_id, natives, notes, remap,
                                       line)
    return notes


def _set_condition_parms(parent, suffix, kind, vector=None, value=None,
                         dimension=None, curve_record=None, functions=None,
                         line=None):
    """One condition's parameters; its time curve from the sideset record
    (exact) or from imported time functions (keys at their points)."""
    if parent.parm(f"boundary_type{suffix}") is None:
        return
    parms = {f"boundary_type{suffix}": int(kind)}
    if value is not None:
        parms[f"value_{suffix}"] = _format_expr_box(value)
    if dimension is not None:
        for axis, flag in zip("xyz", dimension):
            parms[f"{axis}_dimension{suffix}"] = int(bool(flag))
    parent.setParms(parms)
    if vector is not None:
        _set_expr_vector(parent, f"bc_vec{suffix}", f"vector_{suffix}", vector)
    if curve_record is not None:
        _restore_curve_record(parent, suffix, int(kind), curve_record)
    elif functions is not None and line is not None:
        _set_curve_from_functions(parent, suffix, int(kind), functions, line)


def _restore_recorded_sidesets(parent, geo, geo_record, faces_by_id,
                               points_by_id, notes, remap=None):
    """Sidesets exactly as entered, from a record that matches params.json.
    remap: {subdomain number in the files: subdomain here}."""
    remap = remap or {}
    combinations = {}
    for sid, owners in geo_record.get("combinations", {}).items():
        try:
            combinations[int(sid)] = [(remap.get(int(v), int(v)), int(s))
                                      for v, s in owners]
        except (TypeError, ValueError):
            continue
    found = {}
    for kind, by_id in (("faces", faces_by_id), ("nodes", points_by_id)):
        for sid, items in by_id.items():
            owners = combinations.get(sid)
            if owners is None:
                _, vol, j, _ = _decode_boundary_id(sid)
                owners = [(remap.get(vol, vol), j)]
            for owner in owners:
                found.setdefault((kind, owner), set()).update(items)

    sidesets = [entry for entry in geo_record.get("sidesets", [])
                if isinstance(entry, dict)]
    for entry in sidesets:
        entry["subdomain"] = remap.get(int(entry.get("subdomain", 0)),
                                       int(entry.get("subdomain", 0)))
    for entry in sidesets:
        vol, j = int(entry.get("subdomain", 0)), int(entry.get("sideset", 0))
        if j < 1 or parent.parm(f"sideset_selection{geo}_{vol}") is None:
            notes.append(f"geo {geo}: the record names subdomain {vol}, which "
                         "this mesh does not have")
            continue
        _ensure_sideset(parent, geo, vol, j)
    for entry in sidesets:
        vol, j = int(entry.get("subdomain", 0)), int(entry.get("sideset", 0))
        if parent.parm(f"basegroup{geo}_{vol}_{j}") is None:
            continue
        grouptype = int(entry.get("group_type", 0))
        pattern = str(entry.get("pattern", ""))
        text = pattern.strip()
        recorded_conditions = [c for c in entry.get("conditions", [])
                               if isinstance(c, dict)]
        restored = pattern
        # Typed selections and "*" are restored as typed (they are evaluated
        # again on export); picks come back from the faces themselves, which
        # does not depend on primitive numbering.
        if text != "*" and not is_native_selection(text) \
                and recorded_conditions:
            items = found.get(("nodes" if grouptype == 1 else "faces",
                               (vol, j)), set())
            if items:
                restored = list_to_space_str(sorted(items))
            else:
                notes.append(
                    f"geo {geo} subdomain {vol} sideset {j}: none of its "
                    "faces were found on the current mesh; the recorded "
                    "selection was kept")
        parent.setParms({f"basegroup{geo}_{vol}_{j}": restored,
                         f"grouptype{geo}_{vol}_{j}": grouptype})
        count = parent.parm(f"Boundary_Condition__{geo}_{vol}_{j}")
        if count is not None:
            count.set(len(recorded_conditions))
        for k, condition in enumerate(recorded_conditions, 1):
            _set_condition_parms(
                parent, f"{geo}_{vol}_{j}_{k}", condition.get("type", 0),
                vector=condition.get("vector"), value=condition.get("value"),
                dimension=condition.get("dimension")
                if int(condition.get("type", 0)) == 0 else None,
                curve_record=condition)


def _nodal_neumann_file_note(path):
    return (f"File-backed nodal_neumann_boundary entry {path!r} cannot be "
            "reconstructed as point picks; retained in the source JSON.")


def _restore_sidesets_from_ids(parent, geo, conditions, faces_by_id,
                               points_by_id, natives, notes, remap=None,
                               line=None):
    """Sidesets rebuilt from params.json ids (no usable record).
    remap: {subdomain number in the ids: subdomain here}; line: the run's
    timeline, for conditions with time functions."""
    remap = remap or {}

    def decode(sid):
        g, vol, j, k = _decode_boundary_id(sid)
        return g, remap.get(vol, vol), j, k

    entries_by_id = {}
    for kind, key in enumerate(_BC_LIST_KEYS):
        values = conditions.get(key, [])
        for entry in values if isinstance(values, list) else []:
            if not isinstance(entry, dict):
                if key == "nodal_neumann_boundary" and isinstance(entry, str):
                    note = _nodal_neumann_file_note(entry)
                    if note not in notes:
                        notes.append(note)
                continue
            try:
                sid = int(entry.get("id", 0))
            except (TypeError, ValueError):
                continue
            if _decode_boundary_id(sid)[0] == geo:
                entries_by_id.setdefault(sid, []).append((kind, entry))

    ids = set(faces_by_id) | set(points_by_id) | set(natives) \
        | set(entries_by_id)
    regular, combined = {}, []
    for sid in sorted(ids):
        g, vol, j, _ = decode(sid)
        if g != geo or vol < 1 \
                or parent.parm(f"sideset_selection{geo}_{vol}") is None:
            notes.append(f"geo {geo}: boundary id {sid} names subdomain "
                         f"{vol}, which this mesh does not have")
            continue
        if j >= 1:
            regular.setdefault(vol, set()).add(j)
        else:
            combined.append((vol, sid))
    # Number each subdomain's sidesets 1, 2, ... in their original order,
    # then the combined ids of overlapping sidesets (sideset field 0), whose
    # faces and merged conditions become sidesets of their own. A sideset
    # whose faces all lay in overlaps has no id of its own any more; compact
    # numbering keeps it from leaving an empty sideset behind.
    renumber, count = {}, {}
    for vol, numbers in regular.items():
        for new, old in enumerate(sorted(numbers), 1):
            renumber[(vol, old)] = new
        count[vol] = len(numbers)
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            notes.append(
                f"geo {geo} subdomain {vol}: sidesets were renumbered "
                f"({', '.join(str(old) for old in sorted(numbers))} -> "
                f"1-{len(numbers)}); any sideset missing from that list "
                "had all of its faces in overlaps with other sidesets")
    target = {}
    for sid in sorted(ids):
        g, vol, j, _ = decode(sid)
        if j >= 1 and (vol, j) in renumber:
            target[sid] = (vol, renumber[(vol, j)])
    for vol, sid in combined:
        count[vol] = count.get(vol, 0) + 1
        target[sid] = (vol, count[vol])
        notes.append(
            f"geo {geo} subdomain {vol}: the faces shared by several "
            f"sidesets (boundary id {sid}) were restored as sideset "
            f"{count[vol]}, with the combined conditions they received")

    sidesets = {}
    for sid, place in target.items():
        _, _, field_sideset, field_condition = _decode_boundary_id(sid)
        k = field_condition if field_sideset >= 1 else 1
        slot = sidesets.setdefault(place, {"faces": set(), "nodes": set(),
                                           "native": None, "entries": []})
        slot["faces"] |= faces_by_id.get(sid, set())
        slot["nodes"] |= points_by_id.get(sid, set())
        if sid in natives and slot["native"] is None:
            slot["native"] = natives[sid]
        for kind, entry in entries_by_id.get(sid, []):
            slot["entries"].append((k, kind, entry))
    for vol, j in sorted(sidesets):
        _ensure_sideset(parent, geo, vol, j)
    for (vol, j), slot in sorted(sidesets.items()):
        if parent.parm(f"basegroup{geo}_{vol}_{j}") is None:
            continue
        if slot["native"] is not None:
            parent.setParms({f"basegroup{geo}_{vol}_{j}": slot["native"],
                             f"grouptype{geo}_{vol}_{j}": 0})
        elif slot["faces"]:
            parent.setParms({
                f"basegroup{geo}_{vol}_{j}":
                    list_to_space_str(sorted(slot["faces"])),
                f"grouptype{geo}_{vol}_{j}": 0})
        elif slot["nodes"]:
            parent.setParms({
                f"basegroup{geo}_{vol}_{j}":
                    list_to_space_str(sorted(slot["nodes"])),
                f"grouptype{geo}_{vol}_{j}": 1})
        entries = slot["entries"]
        if not entries:
            continue
        numbers = [k for k, _, _ in entries]
        if len(set(numbers)) == len(numbers):
            # one id per condition (exports before 2026-10-01): keep their
            # order; PolyFEM applied only the first id it found on a face
            entries = sorted(entries, key=lambda item: item[0])
            if len(entries) > 1:
                notes.append(
                    f"geo {geo} subdomain {vol} sideset {j}: its "
                    f"{len(entries)} conditions had one boundary id each, so "
                    "the run that wrote this file applied only the first of "
                    "them to the faces; exported again, all of them act on "
                    "the sideset.")
        count_parm = parent.parm(f"Boundary_Condition__{geo}_{vol}_{j}")
        if count_parm is not None:
            count_parm.set(len(entries))
        for k, (_, kind, entry) in enumerate(entries, 1):
            _restore_imported_condition(
                parent, f"{geo}_{vol}_{j}_{k}", kind, entry, line, notes)


def _restore_imported_condition(parent, suffix, kind, entry, line, notes):
    vector = kind in _VECTOR_BC_KINDS
    value = entry.get("value")
    functions = None
    if entry.get("interpolation") is not None:
        where = f"Boundary condition {entry.get('id', suffix)}"
        try:
            functions, note = _interpolation_functions(
                entry["interpolation"], 3 if vector else 1, line)
            if note:
                notes.append(f"{where}: {note}")
            if line is None:
                notes.append(f"{where}: its time function was not restored "
                             "(the Time tab defines no run)")
        except (hou.Error, KeyError, TypeError, ValueError) as exc:
            notes.append(f"{where}: its time function was not restored ({exc})")
    _set_condition_parms(
        parent, suffix, kind, vector=value if vector else None,
        value=value if not vector else None,
        dimension=entry.get("dimension", [True, True, True])
        if kind == 0 else None, functions=functions, line=line)


