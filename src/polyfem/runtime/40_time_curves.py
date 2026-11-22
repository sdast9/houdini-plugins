class _NormalizeConditionExpression(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not isinstance(node.op, ast.Mult):
            return node
        factors = [node.left, node.right]
        negations = 0
        for i, factor in enumerate(factors):
            if isinstance(factor, ast.UnaryOp) and isinstance(factor.op, ast.USub):
                factors[i] = factor.operand
                negations += 1
        factors.sort(key=ast.dump)
        # Retain the legacy 0*t shorthand. A literal zero times a finite
        # coordinate is identically zero, independent of its input.
        left, right = factors
        if isinstance(left, ast.Constant) and type(left.value) in (int, float) \
                and left.value == 0 and isinstance(right, ast.Name) \
                and right.id in ("x", "y", "z", "t"):
            return ast.Constant(0)
        product = ast.BinOp(left, ast.Mult(), right)
        return ast.UnaryOp(ast.USub(), product) if negations % 2 else product


def _expression_structure(value):
    text = re.sub(r"\bif\s*(?=\()", "_if", str(value).strip())
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
        return ast.dump(_NormalizeConditionExpression().visit(tree))
    except SyntaxError:
        return text


def _same_value(first, second):
    """Compare overlapping prescriptions without sampling variable expressions."""
    if _expression_structure(first) == _expression_structure(second):
        return True
    try:
        a, b = float(first), float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    return a == b and math.isfinite(a)


def _value_text(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(float(value))
    return str(value)


def _added(conditions):
    """Sum of the values of several loads of one kind on the same faces."""
    values = [condition["value"] for condition in conditions]
    if len(values) == 1:
        return values[0]
    vector = isinstance(values[0], list)
    for condition in conditions:
        value = condition["value"]
        if isinstance(value, list) != vector or (vector and len(value) != 3):
            raise hou.NodeError(
                f"{condition['label']}: several "
                f"{_BC_TYPE_NAMES[condition['type']]} conditions act on the "
                f"same faces and are added together, so each must be "
                f"[x, y, z] (vector loads) or a single value (normal loads)")

    def add(parts):
        if all(isinstance(part, (int, float)) and not isinstance(part, bool)
               for part in parts):
            return float(sum(parts))
        return " + ".join(f"({_value_text(part)})" for part in parts)

    if vector:
        return [add([value[axis] for value in values]) for axis in range(3)]
    return add(values)


def _merged_load(conditions, where):
    """One entry for the loads of one kind acting on the same faces or nodes:
    their values added, with their time curves. Loads whose curves differ are
    added into one table per component (value 1 x table), which needs their
    values to be numbers."""
    kind = conditions[0]["type"]
    vector = kind in _VECTOR_BC_KINDS
    width = 3 if vector else 1
    entry = {"value": _added(conditions)}
    first = conditions[0]["curves"]
    if all(_same_curve(first[axis], condition["curves"][axis])
           for condition in conditions for axis in range(width)):
        curves = first
    else:
        curves = []
        values_out = []
        for axis in range(width):
            parts = []
            for condition in conditions:
                value = condition["value"][axis] if vector \
                    else condition["value"]
                if isinstance(value, bool) \
                        or not isinstance(value, (int, float)):
                    raise hou.NodeError(
                        f"{condition['label']}: several "
                        f"{_BC_TYPE_NAMES[kind]} loads with different time "
                        f"curves act on {where}. The asset adds them into one "
                        f"time table, so their values must be numbers, not "
                        f"expressions ({_value_text(value)}). Express the time "
                        f"dependence with the Time Curve instead, or keep the "
                        f"loads on separate faces.")
                parts.append((float(value), condition["curves"][axis]))
            curve_points = [curve.points for _, curve in parts
                            if curve is not None]
            if not curve_points:
                # Constant axis (no time curve on any load for this component).
                values_out.append(sum(value for value, _ in parts))
                curves.append(None)
            else:
                points = np.unique(np.concatenate(curve_points))
                total = sum(value * (np.ones(len(points)) if curve is None
                                     else curve.at(points))
                            for value, curve in parts)
                values_out.append(1.0)
                curves.append(TimeCurve(*_compressed_table(points, total)))
        entry["value"] = values_out if vector else values_out[0]
    if vector:
        interpolation = _vector_interpolation(curves)
    else:
        interpolation = None if curves[0] is None else curves[0].as_json()
    if interpolation is not None:
        entry["interpolation"] = interpolation
    return entry


def merge_conditions(conditions, where):
    """The entries PolyFEM honours for one boundary id: at most one of each kind.

    Dirichlet components are combined, and a component prescribed by two
    conditions must agree, time curve included (PolyFEM refuses a second
    dirichlet entry for an id); each component keeps the time curve of the
    condition prescribing it. Loads of one kind are added (PolyFEM applies
    only the first entry of a kind), see _merged_load. A Neumann and a
    normal-aligned traction cannot share faces: PolyFEM replaces the first by
    the second instead of adding them. A cavity has one pressure. Returns
    [(list key, entry without id)].
    """
    by_kind = [[condition for condition in conditions
                if condition["type"] == kind]
               for kind in range(len(_BC_TYPE_NAMES))]
    entries = []
    dirichlet = by_kind[0]
    if len(dirichlet) == 1:
        entry = {"value": dirichlet[0]["value"],
                 "dimension": list(dirichlet[0]["dims"])}
        interpolation = _vector_interpolation(dirichlet[0]["curves"],
                                              dirichlet[0]["dims"])
        if interpolation is not None:
            entry["interpolation"] = interpolation
        entries.append(("dirichlet_boundary", entry))
    elif dirichlet:
        value, dims, owner = [0, 0, 0], [False, False, False], [None] * 3
        curves = [None, None, None]
        for condition in dirichlet:
            vector = condition["value"]
            if not (isinstance(vector, list) and len(vector) == 3):
                raise hou.NodeError(
                    f"{condition['label']}: several Dirichlet conditions act "
                    f"on {where}; to combine them each must give [x, y, z]")
            for axis in range(3):
                if not condition["dims"][axis]:
                    continue
                if dims[axis]:
                    if not _same_value(value[axis], vector[axis]):
                        raise hou.NodeError(
                            f"{owner[axis]['label']} and {condition['label']} "
                            f"both prescribe the {'xyz'[axis]} displacement on "
                            f"{where}, with different values "
                            f"({_value_text(value[axis])} and "
                            f"{_value_text(vector[axis])}). A face can have only "
                            f"one prescription per component: remove the "
                            f"overlap, or turn {'xyz'[axis]} off on one of them.")
                    if not _same_curve(curves[axis],
                                       condition["curves"][axis]):
                        raise hou.NodeError(
                            f"{owner[axis]['label']} and {condition['label']} "
                            f"both prescribe the {'xyz'[axis]} displacement on "
                            f"{where}, with different time curves. A face can "
                            f"have only one prescription per component: remove "
                            f"the overlap, give both the same Time Curve, or "
                            f"turn {'xyz'[axis]} off on one of them.")
                    continue
                value[axis] = vector[axis]
                dims[axis] = True
                owner[axis] = condition
                curves[axis] = condition["curves"][axis]
        entry = {"value": value, "dimension": dims}
        interpolation = _vector_interpolation(curves, dims)
        if interpolation is not None:
            entry["interpolation"] = interpolation
        entries.append(("dirichlet_boundary", entry))
    neumann, normal = by_kind[1], by_kind[2]
    if neumann and normal:
        raise hou.NodeError(
            f"{neumann[0]['label']} and {normal[0]['label']} act on {where}: "
            f"PolyFEM replaces a Neumann traction by a normal-aligned one on "
            f"the same faces instead of adding them. Use Pressure Boundary "
            f"for the normal load (it adds to the traction), or keep the two "
            f"loads on separate faces.")
    if neumann:
        entries.append(("neumann_boundary", _merged_load(neumann, where)))
    if normal:
        entries.append(("normal_aligned_neumann_boundary",
                        _merged_load(normal, where)))
    if by_kind[3]:
        entries.append(("pressure_boundary",
                        _merged_load(by_kind[3], where)))
    cavities = by_kind[4]
    if len(cavities) > 1:
        raise hou.NodeError(
            f"{cavities[0]['label']} and {cavities[1]['label']} act on "
            f"{where}: a cavity has a single pressure.")
    if cavities:
        entries.append(("pressure_cavity", _merged_load(cavities, where)))
    if by_kind[5]:
        entries.append(("nodal_neumann_boundary",
                        _merged_load(by_kind[5], where)))
    return entries


def resolve_sidesets(parent, geo, num_vols):
    """Every sideset of one geometry as explicit faces/nodes, the boundary id
    each face and node ends up with, and the merged conditions of every id.

    Returns a dict:
      face_ids, point_ids      boundary id per surface face / point (0 = none)
      face_vids, point_vids    their PolyFEM vertex numbers (-1 pads triangles)
      conditions               {id: [(list key, entry without id)]}
      combinations             {id: [[subdomain, sideset], ...]} for faces or
                               nodes that lie in several sidesets
      records                  the sidesets as entered, for the import record
      warnings, notes          problems to show / overlaps that were combined
    """
    plan = {"face_ids": np.zeros(0, dtype=np.int64),
            "face_vids": np.zeros((0, 4), dtype=np.int64),
            "point_ids": np.zeros(0, dtype=np.int64),
            "point_vids": np.zeros(0, dtype=np.int64),
            "conditions": {}, "combinations": {}, "records": [],
            "warnings": [], "notes": []}
    null_node = parent.node(f"null_{geo}")
    if null_node is None:
        return plan
    surf_geo = null_node.geometry()
    entity, face_keys = _surface_arrays(surf_geo)
    n_faces = len(entity)
    point_keys = np.frombuffer(
        surf_geo.pointIntAttribValuesAsString("msh_pt_id"),
        dtype=np.int32).astype(np.int64) - 1
    positions = np.frombuffer(
        surf_geo.pointFloatAttribValuesAsString("P"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3)

    vid_of_key, _ = mesh_vertex_ids(parent, geo)

    def vertex_numbers(keys):
        keys = np.asarray(keys, dtype=np.int64)
        valid = keys >= 0
        if np.any(keys[valid] >= len(vid_of_key)):
            raise hou.NodeError(
                f"Geometry {geo}: the surface refers to nodes the mesh does "
                f"not have; import the mesh again")
        vids = np.full(keys.shape, -1, dtype=np.int64)
        vids[valid] = vid_of_key[keys[valid]]
        if np.any(vids[valid] < 0):
            raise hou.NodeError(
                f"Geometry {geo}: the surface refers to node tags the mesh "
                f"does not have; import the mesh again")
        return vids

    face_vids = vertex_numbers(face_keys)
    point_vids = vertex_numbers(point_keys)

    # Face barycentres in the transformed (world) frame, as PolyFEM computes
    # them for its own selections.
    point_of_key = np.full(len(vid_of_key), -1, dtype=np.int64)
    point_of_key[point_keys] = np.arange(len(point_keys))
    corners = np.where(face_keys >= 0,
                       point_of_key[np.maximum(face_keys, 0)], -1)
    if np.any((face_keys >= 0) & (corners < 0)):
        raise hou.NodeError(
            f"Geometry {geo}: the surface display is out of date; import the "
            f"mesh again")
    used = corners >= 0
    centres = np.zeros((n_faces, 3))
    for column in range(4):
        rows = used[:, column]
        centres[rows] += positions[corners[rows, column]]
    centres /= np.maximum(used.sum(axis=1), 1)[:, None]

    face_sets, point_sets, measured = [], [], []
    lazy_timeline = _LazyTimeline(parent)
    volume_points = {}

    def members_of(vol, j, text, grouptype, conditions, warnings):
        """("nodes" | "faces", the sideset's points or faces)."""
        vol_faces = np.flatnonzero(entity == vol)
        name = f"geo {geo} subdomain {vol} sideset {j}"
        if text != "*" and not is_native_selection(text):
            # An explicit pick is exported as the numbers it lists. Anything
            # else in the pattern used to be left out without a word, which
            # wrote a smaller sideset than the one shown in the viewport.
            unknown = unsupported_group_tokens(text)
            if unknown:
                shown = ", ".join(repr(token) for token in unknown[:5])
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol}: the selection "
                    f"contains {shown}"
                    f"{', ...' if len(unknown) > 5 else ''}, which is not a "
                    f"list of {'point' if grouptype == 1 else 'face'} "
                    f"numbers (such as 0 2 5-8 or 0-10:2). Pick the "
                    f"{'points' if grouptype == 1 else 'faces'} again in "
                    f"the viewport, or type * or an axis/box/sphere/plane "
                    f"selection.")
        if grouptype == 1:  # nodes
            face_loads = [c for c in conditions if c["type"] not in (0, 5)]
            if face_loads:
                raise hou.NodeError(
                    f"{face_loads[0]['label']}: sideset {j} of geo {geo}/vol "
                    f"{vol} selects points, and PolyFEM applies surface "
                    f"tractions and pressures to faces only, so this load "
                    f"would do nothing. Set its Group Type to Primitives, or "
                    f"use Nodal Force (Nodal Neumann) for concentrated nodal "
                    f"forces.")
            if vol not in volume_points:
                touched = corners[vol_faces]
                volume_points[vol] = np.unique(touched[touched >= 0])
            vol_points = volume_points[vol]
            if text == "*":
                members = vol_points
            elif is_native_selection(text):
                members = vol_points[native_selection_mask(
                    text, positions[vol_points], name)]
            else:
                members = np.asarray(expand_group_str(text),
                                     dtype=np.int64)
                outside = members[~np.isin(members, vol_points)]
                if len(outside):
                    raise hou.NodeError(
                        f"Sideset {j} of geo {geo}/vol {vol}: point(s) "
                        f"{list(outside[:10])} are not on this subdomain's surface; "
                        f"pick the points again")
            members = np.unique(members)
            if not len(members):
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol} selects nothing")
            return "nodes", members

        nodal_loads = [c for c in conditions if c["type"] == 5]
        if nodal_loads:
            raise hou.NodeError(
                f"{nodal_loads[0]['label']}: sideset {j} of geo {geo}/vol "
                f"{vol} selects faces (Primitives), and Nodal Force (Nodal "
                f"Neumann) applies to mesh nodes (Points) only. Set its Group "
                f"Type to Points.")

        if text == "*":
            members = vol_faces
        elif is_native_selection(text):
            members = vol_faces[native_selection_mask(
                text, centres[vol_faces], name)]
            if not len(members):
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol} selects nothing: "
                    f"no face centre of subdomain {vol} satisfies '{text}'")
        else:
            idx = np.unique(np.asarray(expand_group_str(text),
                                       dtype=np.int64))
            if len(idx) == 0:
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol} selects nothing")
            outside = idx[(idx < 0) | (idx >= n_faces)]
            if len(outside):
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol}: face(s) "
                    f"{list(outside[:10])} are not on this surface; pick "
                    f"the faces again")
            # Restrict to this volume's faces. An explicit pick that loses
            # faces here was made against another subdomain's surface --
            # report it rather than quietly writing a sideset the user did
            # not select.
            members = idx[entity[idx] == vol]
            if len(members) == 0:
                raise hou.NodeError(
                    f"Sideset {j} of geo {geo}/vol {vol} selects "
                    f"{len(idx)} face(s), none of which are on subdomain "
                    f"{vol}; pick faces on that subdomain instead")
            if len(members) != len(idx):
                warnings.append(
                    f"  geo {geo}, subdomain {vol}, sideset {j}: dropped "
                    f"{len(idx) - len(members)} of {len(idx)} faces")
        members = np.unique(members)
        if not len(members):
            raise hou.NodeError(
                f"Sideset {j} of geo {geo}/vol {vol} selects nothing")
        return "faces", members

    for vol in range(1, num_vols + 1):
        n_sidesets = parent.evalParm(f"sideset_selection{geo}_{vol}")
        if n_sidesets > 99:
            raise hou.NodeError(
                f"Geometry {geo}, subdomain {vol}: at most 99 sidesets are "
                f"supported per subdomain (found {n_sidesets}); the ID scheme "
                "reserves 1..99 per subdomain.")
        for j in range(1, n_sidesets + 1):
            pattern = parent.evalParm(f"basegroup{geo}_{vol}_{j}")
            grouptype = parent.evalParm(f"grouptype{geo}_{vol}_{j}")
            conditions = _sideset_conditions(parent, geo, vol, j,
                                             lazy_timeline)
            record = {"subdomain": vol, "sideset": j,
                      "group_type": int(grouptype), "pattern": pattern,
                      "conditions": [c["record"] for c in conditions]}
            plan["records"].append(record)
            text = pattern.strip()
            if not conditions:
                # Nothing acts on it; it is kept as a record, and when it
                # selects something, as a set Read PVD can sum forces over.
                if text:
                    try:
                        kind, members = members_of(vol, j, text, grouptype,
                                                   conditions, [])
                        measured.append((kind, vol, j, members))
                    except hou.Error:
                        pass
                continue
            kind, members = members_of(vol, j, text, grouptype, conditions,
                                       plan["warnings"])
            record.update(kind=kind, count=int(len(members)))
            (point_sets if kind == "nodes" else face_sets).append(
                (vol, j, members, conditions))

    counters = {}

    def next_combination(vol):
        index = counters.get(vol, 0) + 1
        if index > 99:
            raise hou.NodeError(
                f"Geometry {geo}, subdomain {vol}: more than 99 different "
                f"groups of overlapping sidesets; simplify the selections")
        counters[vol] = index
        return combination_id(geo, vol, index)

    def assign(sets, count, kind, scope):
        ids = np.zeros(count, dtype=np.int64)
        if not sets:
            return ids
        if len(sets) > 63:
            raise hou.NodeError(
                f"{scope}: more than 63 sidesets with conditions on {kind}")
        masks = np.zeros(count, dtype=np.uint64)
        for bit, (_, _, members, _) in enumerate(sets):
            masks[members] |= np.uint64(1 << bit)
        selected = np.flatnonzero(masks)
        for mask in np.unique(masks[selected]):
            items = selected[masks[selected] == mask]
            owners = [sets[bit] for bit in range(len(sets))
                      if (int(mask) >> bit) & 1]
            vol, j = owners[0][0], owners[0][1]
            if len(owners) == 1:
                sid = sideset_id(geo, vol, j)
                where = (f"the {len(items)} {kind} of sideset {j} (geo {geo} "
                         f"subdomain {vol})")
            else:
                sid = next_combination(vol)
                names = ", ".join(
                    str(s) if v == vol else f"{s} (subdomain {v})"
                    for v, s, _, _ in owners)
                where = (f"the {len(items)} {kind} shared by sidesets {names} "
                         f"of geo {geo} subdomain {vol}")
                plan["combinations"][int(sid)] = [
                    [int(v), int(s)] for v, s, _, _ in owners]
                plan["notes"].append(
                    f"geo {geo} subdomain {vol}: {len(items)} {kind} lie in "
                    f"sidesets {names} and receive the conditions of all of "
                    f"them (boundary id {sid})")
            ids[items] = sid
            plan["conditions"][int(sid)] = merge_conditions(
                [condition for owner in owners for condition in owner[3]],
                where)
        return ids

    # A face belongs to one subdomain, so face sidesets combine per subdomain;
    # a node can be shared, so node sidesets combine across the geometry.
    face_ids = np.zeros(n_faces, dtype=np.int64)
    for vol in sorted({s[0] for s in face_sets}):
        face_ids += assign([s for s in face_sets if s[0] == vol], n_faces,
                           "faces", f"Geometry {geo}, subdomain {vol}")
    point_ids = assign(point_sets, len(point_keys), "nodes",
                       f"Geometry {geo}")

    # PolyFEM computes a cavity's volume from the faces that share its id, so
    # a cavity sideset must not be split between ids by an overlap.
    for vol, j, members, conditions in face_sets:
        if not any(condition["type"] == 4 for condition in conditions):
            continue
        ids = np.unique(face_ids[members])
        if len(ids) > 1:
            others = sorted({
                s for sid in ids for v, s in plan["combinations"].get(
                    int(sid), []) if (v, s) != (vol, j)})
            raise hou.NodeError(
                f"Sideset {j} of geo {geo}/vol {vol} carries a Pressure "
                f"Cavity, but part of it is also in sideset(s) "
                f"{', '.join(str(s) for s in others)}. PolyFEM computes a "
                f"cavity's volume from the faces sharing one boundary id, so "
                f"the overlap would split the cavity. Keep the cavity's faces "
                f"out of other sidesets (or give another sideset exactly the "
                f"same faces).")

    plan.update(face_ids=face_ids, face_vids=face_vids, point_ids=point_ids,
                point_vids=point_vids)
    # Each sideset's own faces/nodes (overlaps included), for the force
    # sets Read PVD sums nodal forces over (write_scene_record).
    plan["sets"] = [
        {"subdomain": int(vol), "sideset": int(j), "kind": kind,
         "members": members,
         "conditions": sorted({_BC_TYPE_NAMES[c["type"]] for c in conds})}
        for kind, sets in (("faces", face_sets), ("nodes", point_sets))
        for vol, j, members, conds in sets]
    plan["sets"].extend(
        {"subdomain": int(vol), "sideset": int(j), "kind": kind,
         "members": members, "conditions": []}
        for kind, vol, j, members in measured)
    plan["sets"].sort(key=lambda item: (item["subdomain"], item["sideset"]))
    return plan


_SELECTION_FILE_PATTERNS = ("surface_sidesets{geo}_tri.txt",
                            "surface_sidesets{geo}_quad.txt",
                            "point_sidesets{geo}.txt")


def _cached(cache, bucket, key, factory):
    """Memoize expensive setup/export lookups across Check Setup and Write."""
    if cache is None:
        return factory()
    store = cache.setdefault(bucket, {})
    if key not in store:
        store[key] = factory()
    return store[key]


def export_sidesets(parent, geo, num_vols, input_dir, plans=None, cache=None):
    """Write one geometry's selection files.

    Returns (surface_selection_list, point_selection_list) for the json. The
    resolved plan (ids and merged conditions) is stored in plans[geo] when a
    dict is passed, for build_conditions and the import record.
    """
    plan = _cached(cache, "plans", int(geo),
                   lambda: resolve_sidesets(parent, geo, num_vols))
    if plans is not None:
        plans[int(geo)] = plan
    if plan["warnings"]:
        _message("Some sideset faces belong to a different subdomain than the "
                 "sideset they were assigned to, and were left out:\n"
                 + "\n".join(plan["warnings"]))
    if plan["notes"]:
        _status("Overlapping sidesets combined: " + "; ".join(plan["notes"]))

    written = set()
    surface_sel = []
    face_ids, face_vids = plan["face_ids"], plan["face_vids"]
    selected = np.flatnonzero(face_ids)
    for arity, name in ((3, _SELECTION_FILE_PATTERNS[0]),
                        (4, _SELECTION_FILE_PATTERNS[1])):
        faces = selected[(face_vids[selected, 3] < 0) == (arity == 3)]
        if not len(faces):
            continue
        fname = name.format(geo=geo)
        rows = np.column_stack((face_ids[faces], face_vids[faces, :arity]))
        np.savetxt(os.path.join(input_dir, fname), rows.astype(np.int64),
                   fmt="%d")
        surface_sel.append({"file": fname})
        written.add(fname)

    point_sel = []
    point_ids, point_vids = plan["point_ids"], plan["point_vids"]
    selected = np.flatnonzero(point_ids)
    if len(selected):
        fname = _SELECTION_FILE_PATTERNS[2].format(geo=geo)
        rows = np.column_stack((point_ids[selected], point_vids[selected]))
        np.savetxt(os.path.join(input_dir, fname), rows.astype(np.int64),
                   fmt="%d")
        point_sel.append({"file": fname})
        written.add(fname)

    # Drop this geometry's selection files from an earlier export that this
    # one no longer writes, so input/ only holds what params.json uses.
    for name in _SELECTION_FILE_PATTERNS:
        fname = name.format(geo=geo)
        path = os.path.join(input_dir, fname)
        if fname not in written and os.path.isfile(path):
            os.remove(path)
    return surface_sel, point_sel


def _condition_summary(record):
    kind = record.get("type", 0)
    name = _BC_TYPE_NAMES[kind] if 0 <= kind < len(_BC_TYPE_NAMES) else "?"
    text = record.get("vector") if kind in _VECTOR_BC_KINDS else record.get("value")
    if kind == 0:
        axes = "".join(axis for axis, flag in zip(
            "xyz", record.get("dimension", [1, 1, 1])) if flag)
        return f"{name} {text} on {axes or 'no component'}"
    return f"{name} {text}"


def sideset_report(plans):
    """Readable summary of what an export applied to every sideset."""
    lines = []
    for geo in sorted(plans):
        plan = plans[geo]
        for record in plan["records"]:
            pattern = record["pattern"].strip()
            shown = pattern if pattern == "*" or is_native_selection(pattern) \
                else "picked"
            where = (f"{record['count']} {record['kind']}"
                     if "count" in record else "not exported (no conditions)")
            conditions = "; ".join(_condition_summary(condition)
                                   for condition in record["conditions"])
            lines.append(
                f"geo {geo} subdomain {record['subdomain']} sideset "
                f"{record['sideset']} ({shown}): {where}"
                + (f" -- {conditions}" if conditions else ""))
        lines.extend("  " + note for note in plan["notes"])
        lines.extend("  left out: " + warning.strip()
                     for warning in plan["warnings"])
    return lines


def export_volumes(parent, geo, input_dir):
    """volumes<geo>.txt: one body id per volume element, element order."""
    branch = parent.node(f"branch_{geo}")
    geo_data = branch.geometry()
    entity = np.frombuffer(
        geo_data.primIntAttribValuesAsString("Entity"), dtype=np.int32)
    keep = _volume_mask(geo_data)
    if keep is not None:
        entity = entity[keep]
    entity = entity[solver_element_order(geo_data, expect_msh=True)]
    fname = f"volumes{geo}.txt"
    np.savetxt(os.path.join(input_dir, fname),
               1000 * int(geo) + entity.astype(np.int64), fmt="%d")
    return fname


