def _restore_legacy_scene(parent, data, geometry, input_dir, material_slots,
                          scene=None):
    """Best-effort BC/selection import for 1.x and generic older PolyFEM ids.

    HDA 1.2 encoded geo/volume/sideset/BC as decimal digits. Generic PolyFEM
    scenes commonly use small shared selection ids (10, 20, ...). Both are
    mapped into the current collision-free HDA multiparms here.
    """
    skipped = []
    targets = {}
    next_sideset = {}
    grouped_prims = {}
    grouped_points = {}

    def legacy_target(value):
        text = str(abs(int(value)))
        if len(text) < 4:
            return None
        geo, vol, sideset = (int(text[0]), int(text[1]), int(text[2]))
        bc = int(text[3:] or 1)
        if parent.parm(f"sideset_selection{geo}_{vol}") is None:
            return None
        return geo, vol, sideset, bc

    def target_for(selection_id, geo, vol):
        direct = legacy_target(selection_id)
        if direct is not None:
            target = direct[:3]
        else:
            key = (geo, vol)
            sid_key = (geo, vol, int(selection_id))
            if sid_key not in next_sideset:
                next_sideset[sid_key] = 1 + max(
                    [value for (g, v, _), value in next_sideset.items()
                     if (g, v) == key] or [0])
            target = (geo, vol, next_sideset[sid_key])
        targets.setdefault(int(selection_id), [])
        if target not in targets[int(selection_id)]:
            targets[int(selection_id)].append(target)
        _ensure_sideset(parent, *target)
        return target

    for geo, g in enumerate(geometry, start=1):
        if g.get("is_obstacle") or parent.node(f"null_{geo}") is None:
            continue
        surface_geo = parent.node(f"null_{geo}").geometry()
        entities, _ = _surface_arrays(surface_geo)
        # Selection files hold PolyFEM's vertex numbers (mesh_vertex_ids).
        face_lookup, point_lookup = _surface_lookups(parent, geo)
        volume_count = parent.evalParm(f"num_volumes{geo}")

        selections = g.get("surface_selection", [])
        if not isinstance(selections, list):
            selections = [selections]
        for entry in selections:
            if isinstance(entry, dict) and "file" in entry:
                path = _selection_file(entry, input_dir)
                try:
                    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
                except (OSError, ValueError) as exc:
                    skipped.append(
                        f"Geometry {geo}: unreadable surface selection file "
                        f"{entry.get('file')} ({exc}).")
                    continue
                for row in rows:
                    if len(row) < 4:
                        continue
                    sid = int(row[0])
                    prim = face_lookup.get(tuple(sorted(int(v) for v in row[1:])))
                    if prim is None:
                        skipped.append(
                            f"Geometry {geo}: one face from {entry.get('file')} "
                            "is not on the current mesh.")
                        continue
                    direct = legacy_target(sid)
                    vol = direct[1] if direct is not None else int(entities[prim])
                    target = target_for(sid, geo, vol)
                    grouped_prims.setdefault(target, set()).add(prim)
            elif isinstance(entry, dict) and "id" in entry:
                sid = int(entry["id"])
                pattern = _native_pattern(entry)
                if pattern is None:
                    skipped.append(
                        f"Geometry {geo}: selection id {sid} uses an "
                        "unsupported older selector.")
                    continue
                direct = legacy_target(sid)
                volumes = [direct[1]] if direct is not None \
                    else range(1, volume_count + 1)
                for vol in volumes:
                    target = target_for(sid, geo, vol)
                    parent.setParms({
                        f"basegroup{target[0]}_{target[1]}_{target[2]}": pattern,
                        f"grouptype{target[0]}_{target[1]}_{target[2]}": 0})

        point_selections = g.get("point_selection", [])
        if not isinstance(point_selections, list):
            point_selections = [point_selections]
        for entry in point_selections:
            if not (isinstance(entry, dict) and "file" in entry):
                continue
            path = _selection_file(entry, input_dir)
            try:
                rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
            except (OSError, ValueError) as exc:
                skipped.append(
                    f"Geometry {geo}: unreadable point selection file "
                    f"{entry.get('file')} ({exc}).")
                continue
            for row in rows:
                if len(row) < 2:
                    continue
                sid = int(row[0])
                point = point_lookup.get(int(row[1]))
                if point is None:
                    continue
                direct = legacy_target(sid)
                vol = direct[1] if direct is not None else 1
                target = target_for(sid, geo, vol)
                grouped_points.setdefault(target, set()).add(point)

    for target, prims in grouped_prims.items():
        geo, vol, sideset = target
        parent.setParms({
            f"basegroup{geo}_{vol}_{sideset}":
                list_to_space_str(sorted(prims)),
            f"grouptype{geo}_{vol}_{sideset}": 0})
    for target, points in grouped_points.items():
        geo, vol, sideset = target
        parent.setParms({
            f"basegroup{geo}_{vol}_{sideset}":
                list_to_space_str(sorted(points)),
            f"grouptype{geo}_{vol}_{sideset}": 1})

    # Initial conditions use material ids, not boundary selection ids.
    initial = data.get("initial_conditions", {})
    per_slot = {}
    for type_index, key in enumerate(("solution", "velocity", "acceleration")):
        for entry in initial.get(key, []) if isinstance(initial, dict) else []:
            slots = _material_slots(entry.get("id"), material_slots, True)
            if not slots:
                skipped.append(
                    f"Initial condition id {entry.get('id')} was not mapped.")
                continue
            for slot in slots:
                per_slot.setdefault(slot, []).append((type_index, entry.get("value")))
    for (geo, vol), entries in per_slot.items():
        parent.setParms({f"initial_conditions{geo}_{vol}": len(entries)})
        for index, (type_index, value) in enumerate(entries, start=1):
            parent.setParms({
                f"conditiontype{geo}_{vol}_{index}": type_index})
            _set_expr_vector(parent, f"ic_vec{geo}_{vol}_{index}",
                             f"condition_vector_{geo}_{vol}_{index}", value)

    bc = data.get("boundary_conditions", {})
    if not isinstance(bc, dict):
        return skipped
    try:
        line = timeline(parent)
    except hou.Error:
        line = None
    used_slots = {}
    for type_index, key in enumerate(_BC_LIST_KEYS):
        for entry in bc.get(key, []):
            if not isinstance(entry, dict):
                if key == "nodal_neumann_boundary" and isinstance(entry, str):
                    skipped.append(_nodal_neumann_file_note(entry))
                continue
            sid = int(entry.get("id", 0))
            direct = legacy_target(sid)
            selected_targets = [direct[:3]] if direct is not None \
                else targets.get(sid, [])
            if not selected_targets:
                skipped.append(
                    f"Boundary condition selection id {sid} was not found.")
                continue
            for geo, vol, sideset in selected_targets:
                used = used_slots.setdefault((geo, vol, sideset), set())
                index = direct[3] if direct is not None else 1
                while index in used:
                    index += 1
                used.add(index)
                count = parent.parm(
                    f"Boundary_Condition__{geo}_{vol}_{sideset}")
                if count is not None and count.eval() < index:
                    count.set(index)
                _restore_imported_condition(
                    parent, f"{geo}_{vol}_{sideset}_{index}",
                    type_index, entry, line, skipped)
    _restore_obstacle_conditions(parent, data, scene, skipped, geometry)
    return skipped


def _young_poisson(model):
    young, nu = model.get("young", model.get("E")), model.get("nu")
    if young is not None and nu is not None:
        return young, nu
    lam, mu = model.get("lambda"), model.get("mu")
    if not all(type(v) in (int, float) and math.isfinite(v)
               for v in (lam, mu)) or lam + mu == 0:
        return None
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


def _import_isotropic(parent, geo, vol, model, warnings=None):
    warn = warnings.append if warnings is not None else _message
    token = model.get("type")
    if token in _TENSOR_MODELS + _E_NU_MODELS:
        elastic = _young_poisson(model)
        if token in ("SaintVenant", "HookeLinearElasticity") \
                and "elasticity_tensor" not in model:
            if elastic and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in elastic) and elastic[1] != -1:
                E, nu = elastic
                shear = E / (2 * (1 + nu))
                model = dict(model, E=E, nu=nu,
                             elasticity_tensor=[E]*3 + [nu]*3 + [shear]*3)
            elif any(k in model for k in ("E", "young", "lambda", "mu", "nu")):
                warn(f"{_slot_label(geo, vol, None)}: {token} scalar stiffness "
                     "could not be converted to a numeric Elasticity Tensor.")
                model = {k: v for k, v in model.items() if k not in ("E", "nu")}
        elif token in _E_NU_MODELS and elastic is not None:
            model = dict(model, E=elastic[0], nu=elastic[1])

    parms = {}
    for key, base in (("E", "E"), ("nu", "nu"), ("c1", "c1"), ("c2", "c2"),
                      ("c3", "c3"), ("d1", "d1"), ("k", "bulk"),
                      ("phi", "phi"), ("psi", "psi"),
                      ("JBarrierThreshold", "inv_barrier_thresh"),
                      ("weight", "amips_weight"),
                      ("power", "amips_power"),
                      ("normalized", "amips_normalized")):
        if key in model and parent.parm(f"{base}{geo}_{vol}") is not None:
            parms[f"{base}{geo}_{vol}"] = (
                int(bool(model[key])) if key == "normalized"
                else model[key])
    if model.get("type") == "GrowthNeoHookean" \
            or any(k in model for k in ("g11", "g22", "g33")):
        diag_tuple = parent.parmTuple(f"growth_diag{geo}_{vol}")
        if diag_tuple is not None:
            diag_tuple.set([
                _format_expr_box(model.get("g11", 1)),
                _format_expr_box(model.get("g22", 1)),
                _format_expr_box(model.get("g33", 1))])
    if model.get("type") == "GrowthNeoHookean" \
            or any(k in model for k in ("g12", "g13", "g23")):
        off_tuple = parent.parmTuple(f"growth_offdiag{geo}_{vol}")
        if off_tuple is not None:
            off_tuple.set([
                _format_expr_box(model.get("g12", 0)),
                _format_expr_box(model.get("g13", 0)),
                _format_expr_box(model.get("g23", 0))])
    if model.get("type") == "InversionBarrier" \
            and "JBarrierThreshold" not in model \
            and parent.parm(f"inv_barrier_thresh{geo}_{vol}") is not None:
        parms[f"inv_barrier_thresh{geo}_{vol}"] = 0.5
    if model.get("type") == "AMIPS":
        if "weight" not in model \
                and parent.parm(f"amips_weight{geo}_{vol}") is not None:
            parms[f"amips_weight{geo}_{vol}"] = 1.0
        rest_pose = model.get("use_rest_pose", False)
        if isinstance(rest_pose, str):
            rest_pose = {"true": True, "false": False, "1": True, "0": False
                         }.get(rest_pose.strip().lower())
        if rest_pose not in (True, False):
            warn(f"{_slot_label(geo, vol, None)}: AMIPS use_rest_pose "
                 f"{model['use_rest_pose']!r} cannot be represented by the "
                 "Use Rest Pose boolean toggle; imported with the toggle off.")
            rest_pose = False
        parms[f"amips_use_rest_pose{geo}_{vol}"] = int(rest_pose)
        if "power" not in model \
                and parent.parm(f"amips_power{geo}_{vol}") is not None:
            parms[f"amips_power{geo}_{vol}"] = 1.0
        if "normalized" not in model \
                and parent.parm(f"amips_normalized{geo}_{vol}") is not None:
            parms[f"amips_normalized{geo}_{vol}"] = 0
    if "phi" in model or "psi" in model:
        parms[f"damping{geo}_{vol}"] = 1
    for keys, base in (
            (("elasticity_tensor",), "elast_tensor"),
            (("c", "C"), "ogdenC"), (("m",), "ogdenM"),
            (("alphas",), "ogden_alphas"), (("mus",), "ogden_mus"),
            (("Ds", "ds"), "ogden_ds")):
        key = next((candidate for candidate in keys if candidate in model), None)
        if key is not None and parent.parm(f"{base}{geo}_{vol}") is not None:
            parms[f"{base}{geo}_{vol}"] = json.dumps(model[key])
            if base == "elast_tensor" \
                    and parent.parm(f"elast_mode{geo}_{vol}") is not None:
                vals = model[key]
                if isinstance(vals, list) and len(vals) == 9 \
                        and all(isinstance(v, (int, float)) for v in vals):
                    parms[f"elast_mode{geo}_{vol}"] = 0
                    _set_ortho_parms(parent, geo, vol, vals)
                else:
                    parms[f"elast_mode{geo}_{vol}"] = 1
    if token in ("SaintVenant", "HookeLinearElasticity") \
            and isinstance(model.get("fiber_direction"), list):
        axes = model["fiber_direction"]
        parms[f"material_coord{geo}_{vol}"] = json.dumps(axes, separators=(",", ":"))
    if parms:
        parent.setParms(parms)
    if token == "SaintVenant":
        note = _saint_venant_axes_message(parent, geo, vol)
        if note:
            warn(note)


def _import_scalar(parent, geo, vol, fam, name, value, params_dir,
                   warnings=None):
    """Set a scalar parameter's source and value from imported JSON."""
    prefix = _parm_name(geo, vol, fam, f"{name}_source")
    if parent.parm(prefix) is None:
        return
    if isinstance(value, (int, float)):
        parent.setParms({prefix: "constant",
                         _parm_name(geo, vol, fam, name): float(value)})
        return
    text = str(value)
    source_path = text if os.path.isabs(text) else os.path.join(
        params_dir, text)
    if os.path.isfile(source_path):
        try:
            source_path = _stage_material_file(
                parent, source_path, f"{name} per-element file",
                _parm_name(geo, vol, fam, f"{name}_file"))
        except hou.NodeError as exc:
            if warnings is not None:
                warnings.append(str(exc))
            return
        parent.setParms({
            prefix: "file",
            _parm_name(geo, vol, fam, f"{name}_file"):
                source_path})
    else:
        parent.setParms({prefix: "expression",
                         _parm_name(geo, vol, fam, f"{name}_expr"): text})


def _import_fiber_direction(parent, geo, vol, fam, value, params_dir,
                            warnings=None):
    source = _parm_name(geo, vol, fam, "fib_source")
    if parent.parm(source) is None:
        return
    if isinstance(value, dict) and value.get("type") == "per_element_file":
        relative = value.get("path", FIBER_FILE)
        source_path = relative if os.path.isabs(relative) else os.path.join(
            params_dir, relative)
        try:
            source_path = _stage_material_file(
                parent, source_path, "fiber file",
                _parm_name(geo, vol, fam, "fib_file"))
        except hou.NodeError as exc:
            if warnings is not None:
                warnings.append(str(exc))
            return
        parent.setParms({
            source: "file",
            _parm_name(geo, vol, fam, "fib_file"): source_path,
            _parm_name(geo, vol, fam, "fib_file_field"):
                value.get("field", "FIB_DIR1")})
        return
    if isinstance(value, list) and len(value) == 3:
        if all(isinstance(entry, (int, float)) for entry in value):
            parent.setParms({source: "constant"})
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_dir")).set(
                [float(entry) for entry in value])
            return
        parent.setParms({source: "expression",
                         _parm_name(geo, vol, fam, "fib_expr"):
                             json.dumps(value)})


def _import_fiber_model(parent, geo, vol, fam, model, params_dir,
                        warnings=None):
    token = model.get("type")
    if fam is not None:
        parent.setParms({f"fam_model{geo}_{vol}_{fam}": token})
    if "fiber_direction" in model:
        _import_fiber_direction(parent, geo, vol, fam,
                                model["fiber_direction"], params_dir, warnings)
    parms = {}
    for key in ("k1", "k2", "k_chi", "Tmax"):
        name = _parm_name(geo, vol, fam, key)
        if key in model and parent.parm(name) is not None:
            parms[name] = model[key]
    if "activation" in model:
        name = _parm_name(geo, vol, fam, "activation")
        if parent.parm(name) is not None:
            parms[name] = str(model["activation"])
    if parms:
        parent.setParms(parms)
    if "kappa" in model:
        _import_scalar(parent, geo, vol, fam, "kappa", model["kappa"],
                       params_dir, warnings)


def _import_composite(parent, geo, vol, models, params_dir, warnings=None):
    """Split an imported "models" list back into matrix + fiber families.

    Anything the interface cannot express (an unknown type, a nested sum) is
    kept verbatim in the extra-models field so the round trip stays lossless.
    """
    matrix = "None"
    families, extra, penalty, inv_barrier = [], [], None, None
    for model in models:
        token = model.get("type") if isinstance(model, dict) else None
        if token in FIBER_MODELS:
            families.append(model)
        elif token == "VolumePenalty" and penalty is None:
            penalty = model
        elif token == "InversionBarrier" and inv_barrier is None \
                and parent.parm(f"add_inv_barrier{geo}_{vol}") is not None:
            inv_barrier = model
        elif token in MATRIX_MODEL_TOKENS and token != "None" \
                and matrix == "None" and not families:
            matrix = token
            _import_isotropic(parent, geo, vol, model, warnings)
        else:
            extra.append(model)

    comp_parms = {
        f"matrix_model{geo}_{vol}": matrix,
        f"num_fiber_families{geo}_{vol}": len(families),
        f"volume_penalty{geo}_{vol}": 1 if penalty else 0,
        f"extra_models_json{geo}_{vol}": json.dumps(extra) if extra else ""}
    if parent.parm(f"add_inv_barrier{geo}_{vol}") is not None:
        comp_parms[f"add_inv_barrier{geo}_{vol}"] = 1 if inv_barrier else 0
        if inv_barrier:
            elastic = _young_poisson(inv_barrier)
            if elastic is not None:
                comp_parms[f"comp_inv_E{geo}_{vol}"], \
                    comp_parms[f"comp_inv_nu{geo}_{vol}"] = elastic
            else:
                for key in ("E", "nu"):
                    if key in inv_barrier:
                        comp_parms[f"comp_inv_{key}{geo}_{vol}"] = inv_barrier[key]
            comp_parms[f"comp_inv_thresh{geo}_{vol}"] = \
                inv_barrier.get("JBarrierThreshold", 0.5)
    parent.setParms(comp_parms)
    if penalty and parent.parm(f"vp_k{geo}_{vol}") is not None:
        parent.setParms({f"vp_k{geo}_{vol}": penalty.get("k", 1e8)})
    for index, model in enumerate(families, start=1):
        # A symmetric pair comes in as two independent families: the
        # expanded stack is what PolyFEM sees, so this is value-exact. The
        # scene record puts the pair back when it can (_restore_material_ui).
        parent.setParms({f"fam_mirror{geo}_{vol}_{index}": 0})
        _import_fiber_model(
            parent, geo, vol, index, model, params_dir, warnings)
    if extra:
        message = (
            f"Geo {geo} subdomain {vol}: {len(extra)} model(s) cannot be "
            "shown by the interface; retained in Extra Models (JSON).")
        if warnings is not None:
            warnings.append(message)
        else:
            _message("Import: " + message)


def _selection_file(selection, params_dir):
    """Resolve a PolyFEM file selection to an absolute path, if it is one."""
    value = selection
    if isinstance(selection, dict):
        value = selection.get("file")
    if not isinstance(value, str) or not value:
        return None
    return value if os.path.isabs(value) else os.path.join(params_dir, value)


def _decoded_volume(mid, geo, legacy):
    """Recognize this HDA's current and 1.x body-id encodings."""
    try:
        value = int(mid)
    except (TypeError, ValueError):
        return None
    if value // 1000 == int(geo) and value % 1000 > 0:
        return value % 1000
    text = str(abs(value))
    if legacy and len(text) >= 2 and int(text[0]) == int(geo):
        result = int(text[1:] or 1)
        return result if result > 0 else None
    return None


def _restore_volume_assignments(parent, geo, geometry_data, params_dir,
                                legacy, warnings):
    """Recreate edited Entity values from geometry.volume_selection.

    The MSH file may still contain its original physical groups, while the
    previous Houdini scene re-tagged arbitrary elements into new subdomains.
    The exported volume-selection file is therefore the source of truth.
    Returns {original material/body id: (geo, HDA volume index)}.
    """
    selection = geometry_data.get("volume_selection")
    branch = parent.node(f"branch_{geo}")
    if branch is None:
        warnings.append(
            f"Geometry {geo}: node tree was not created; volume assignments "
            "could not be restored.")
        return {}
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    prim_numbers = np.flatnonzero(mask) if mask is not None else np.arange(
        geo_data.intrinsicValue("primitivecount"))
    if isinstance(selection, (int, float)):
        ids = np.full(len(prim_numbers), int(selection), dtype=np.int64)
    else:
        path = _selection_file(selection, params_dir)
        if path is None:
            if selection is not None:
                warnings.append(
                    f"Geometry {geo}: unsupported volume_selection; retained "
                    "the subdomains stored in the mesh.")
            groups = physical_groups(geo_data)
            return {tag: (geo, vol)
                    for vol, tag in enumerate(groups["tags"], 1)}
        try:
            ids = np.loadtxt(path, dtype=np.int64, ndmin=1).reshape(-1)
        except (OSError, ValueError) as exc:
            warnings.append(
                f"Geometry {geo}: could not read volume_selection "
                f"'{os.path.basename(path)}' ({exc}); retained mesh "
                "subdomains.")
            return {}
    if len(ids) != len(prim_numbers):
        warnings.append(
            f"Geometry {geo}: volume_selection has {len(ids)} rows but the "
            f"mesh has {len(prim_numbers)} volume elements; retained mesh "
            "subdomains.")
        return {}

    ordered_ids = list(dict.fromkeys(int(value) for value in ids))
    ids = ids[np.argsort(solver_element_order(geo_data))]
    decoded = [_decoded_volume(value, geo, legacy) for value in ordered_ids]
    if all(value is not None for value in decoded) \
            and len(set(decoded)) == len(decoded):
        body_to_volume = dict(zip(ordered_ids, decoded))
        if sorted(decoded) != list(range(1, len(decoded) + 1)):
            # Exports from before 2026-10-01 numbered subdomains by physical
            # tag, gaps included (tags 1 and 100 made 100 subdomains). Keep
            # the order, drop the gaps; sidesets and initial conditions
            # follow through the returned lookup.
            compact = {volume: index for index, volume in
                       enumerate(sorted(decoded), start=1)}
            body_to_volume = {body: compact[volume]
                              for body, volume in body_to_volume.items()}
            warnings.append(
                f"Geometry {geo}: subdomains {sorted(decoded)} were "
                f"renumbered 1-{len(decoded)} (subdomain numbers no longer "
                "follow physical tags with gaps); materials, sidesets and "
                "initial conditions moved with them.")
    else:
        # Arbitrary valid PolyFEM material ids cannot be represented verbatim by
        # the HDA's geo*1000+volume convention. Compact them but retain their
        # exact material association through the returned lookup.
        body_to_volume = {
            body: index for index, body in enumerate(ordered_ids, start=1)}
        warnings.append(
            f"Geometry {geo}: arbitrary material ids {ordered_ids} were "
            "compacted to consecutive Houdini subdomains; assignments and "
            "material associations were preserved.")

    num_volumes = max(body_to_volume.values())
    parent.setParms({f"num_volumes{geo}": num_volumes})
    for volume in range(1, num_volumes + 1):
        _set_color_tuple(parent, f"color_{geo}_{volume}",
                         default_color(geo, volume))

    # Build the same lightweight VEX retag chain used by interactive edits,
    # once, then reconstruct the display/selection network from its result.
    upstream = parent.node(f"elements_{geo}")
    for body, volume in body_to_volume.items():
        selected = prim_numbers[ids == body]
        vex = parent.createNode("attribwrangle", f"vex_{geo}_{volume}")
        vex.setParms({
            "group": list_to_space_str(selected), "grouptype": 4, "class": 1,
            "snippet": f"i@Entity = {int(volume)};"})
        vex.setInput(0, upstream)
        upstream = vex
    branch.setInput(0, upstream)
    clear_geo_tree(parent, geo)
    create_geo_tree({"node": parent, "script_multiparm_index": str(geo)})
    parent.setParms({f"subdomain_number_{geo}": 1})
    fetch_elements({"node": parent, "script_multiparm_index": str(geo)})
    return {
        int(body): (int(geo), int(volume))
        for body, volume in body_to_volume.items()}


def _material_slots(mid, material_slots, legacy):
    try:
        value = int(mid)
    except (TypeError, ValueError, OverflowError):
        return []
    if value in material_slots:
        return material_slots[value]
    if legacy:
        text = str(abs(value))
        if len(text) >= 2:
            return [(int(text[0]), int(text[1:]))]
    geo, vol = divmod(value, 1000)
    return [(geo, vol)] if geo > 0 and vol > 0 else []


def _set_import_parms(parent, parms, section, warnings):
    for name, value in parms.items():
        try:
            if value is None or isinstance(value, (list, dict)) or (
                    isinstance(value, float) and not math.isfinite(value)):
                raise ValueError(f"invalid value {value!r}")
            parent.parm(name).set(value)
        except (hou.Error, AttributeError, TypeError, ValueError, OverflowError) as exc:
            warnings.append(f"{section}: could not restore {name} ({_error_text(exc)}).")


def _restore_time(parent, data, warnings):
    time = data.get("time")
    if not isinstance(time, dict):
        return
    parms = {
        "t0": time.get("t0", parent.evalParm("t0")),
        "quasistatic": bool(
            time.get("quasistatic", parent.evalParm("quasistatic")))}
    integrator = time.get("integrator")
    integrators = {
        name: index for index, name in enumerate((
            "ImplicitEuler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5",
            "BDF6", "ImplicitNewmark"))}
    if isinstance(integrator, dict):
        token = integrator.get("type")
        if isinstance(token, str) and token in integrators:
            parms["integrator"] = integrators[token]
        if "beta" in integrator:
            parms["newmark_beta"] = integrator["beta"]
        if "gamma" in integrator:
            parms["newmark_gamma"] = integrator["gamma"]
    elif isinstance(integrator, str) and integrator in integrators:
        parms["integrator"] = integrators[integrator]
    parms["end_time_bool"] = int("tend" in time)
    parms["time_inc_bool"] = int("dt" in time)
    parms["num_timesteps_bool"] = int("time_steps" in time)
    if parent.parm("define_run_by") is not None:
        if "tend" in time and "dt" in time:
            parms["define_run_by"] = 0
        elif "tend" in time and "time_steps" in time:
            parms["define_run_by"] = 1
        elif "dt" in time and "time_steps" in time:
            parms["define_run_by"] = 2
    for key, parm in (
            ("tend", "tend"), ("dt", "dt"),
            ("time_steps", "num_timesteps")):
        if key in time:
            parms[parm] = time[key]
    _set_import_parms(parent, parms, "time", warnings)


def _restore_contact(parent, data, warnings):
    contact = data.get("contact")
    if not isinstance(contact, dict):
        return
    parms = {"enable": bool(contact.get("enabled", False))}
    for key, parm in (
            ("dhat", "dhat"), ("epsv", "epsv"),
            ("friction_coefficient", "cof"),
            ("use_convergent_formulation", "area_weighted"),
            ("use_gcp_formulation", "gcp_enable"),
            ("alpha_n", "alpha_n"), ("alpha_t", "alpha_t"),
            ("min_distance_ratio", "min_dist_ratio"),
            ("use_adaptive_dhat", "adapt_dhat")):
        if key in contact:
            parms[parm] = contact[key]
    # Semi-implicit exports write every convergent switch (convergent_flags):
    # Area Weighted and Remove Duplicate Contacts follow them. Anything else
    # (Adaptive/Fixed, or no convergent formulation) leaves duplicate removal
    # off, as the toggle only applies to Semi-Implicit mode.
    solver_contact = data.get("solver", {}).get("contact", {}) \
        if isinstance(data.get("solver"), dict) else {}
    semi_implicit = not isinstance(solver_contact, dict) or \
        solver_contact.get("barrier_stiffness", "semi_implicit") == "semi_implicit"
    convergent = bool(contact.get("use_convergent_formulation", False))
    if semi_implicit and convergent and "use_area_weighting" in contact:
        parms["area_weighted"] = bool(contact["use_area_weighting"])
    parms["improved_max"] = semi_implicit and convergent and \
        bool(contact.get("use_improved_max_operator", False))
    adhesion = contact.get("adhesion")
    if isinstance(adhesion, dict):
        for key, parm in (
                ("adhesion_enabled", "adhesion_enable"),
                ("dhat_p", "dhat_p"), ("dhat_a", "dhat_a"),
                ("adhesion_strength", "adhesion_strength"),
                ("tangential_adhesion_coefficient", "tangent_coeff"),
                ("epsa", "adhesion_epsa")):
            if key in adhesion:
                parms[parm] = adhesion[key]
    _set_import_parms(parent, parms, "contact", warnings)


