# ---- Data Over Time ------------------------------------------------------------

_STATISTICS = (("max", "max"), ("min", "min"), ("mean", "mean"),
               ("amean", "arithmetic mean"), ("integral", "integral"),
               ("sum", "sum"), ("rms", "RMS"), ("std", "SD"),
               ("count", "count"))
_STATISTIC_DEFINITIONS = (
    ("max / min", "largest / smallest value over the region's elements (or "
     "nodes)"),
    ("mean", "volume-weighted mean: sum of value x volume / sum of volumes "
     "(the volume of each element, or the volume lumped to each node)"),
    ("arithmetic mean", "plain mean over the elements (or nodes), whatever "
     "their size"),
    ("integral", "sum of value x volume over the region (unit: the value's "
     "unit x volume)"),
    ("sum", "plain sum over the elements (or nodes); over nodes, a nodal "
     "force's sum is the resultant force"),
    ("RMS", "root mean square over the elements (or nodes)"),
    ("SD", "standard deviation over the elements (or nodes), population "
     "formula"),
    ("Pnn", "the nn-th percentile over the elements (or nodes), linear "
     "interpolation"),
    ("count", "number of elements (or nodes) with a value"),
    ("value", "a region of one element or node has a single column"),
)


def _quantity_definition(node, index):
    p = f"export_q_{{}}{index}"
    field = _quantity_field(node, index)
    token = _quantity_value(node, index)
    basis = _menu_token(node, p.format("basis")) or "auto"
    if basis == "auto":
        name = field[:-4] if field.endswith("_avg") else field
        nodal = name in _NODAL_FIELDS or name.endswith("_forces")
        basis = "nodes" if nodal else "elements"
    statistics = [key for key, _ in _STATISTICS
                  if node.evalParm(p.format(key))]
    percentiles = []
    for text in node.evalParm(p.format("percentiles")).replace(
            ",", " ").split():
        try:
            value = float(text)
        except ValueError:
            raise ExportError(f"Quantity {index}: '{text}' is not a "
                              "percentile (0 to 100).")
        if not 0 <= value <= 100:
            raise ExportError(f"Quantity {index}: percentile {text} is not "
                              "between 0 and 100.")
        percentiles.append(value)
    return {"index": index, "field": field, "value": token, "basis": basis,
            "statistics": statistics, "percentiles": percentiles,
            "where": bool(node.evalParm(p.format("where"))),
            "volume": _menu_token(node, p.format("volume")) or "current",
            "label": node.evalParm(p.format("label")).strip()}


def _quantity_header(quantity, components):
    """'von Mises stress', 'displacement magnitude', 'Cauchy stress xy'."""
    if quantity["label"]:
        return quantity["label"]
    field, token = quantity["field"], quantity["value"]
    name = _HEADER_NAMES.get(field[:-4] if field.endswith("_avg") else field)
    if _is_fiber_quantity(field):
        kind, family = field.split(":", 1)
        base = {"fiber_stretch": "fiber stretch",
                "fiber_isochoric_stretch": "fiber isochoric stretch",
                "fiber_I4": "fiber I4"}[kind]
        name = f"{base} ({_field_label(family)})"
    if name is None:
        name = _display_name(field)
    if components <= 1 or field == "_volume":
        return name
    if token in ("", "auto"):  # the Display tab's Automatic Value
        token = "magnitude"
    if components == 9 and token in _TENSOR_ENTRY_NAMES:
        return f"{name} {_TENSOR_ENTRY_NAMES[token]}"
    if field in PRINCIPAL_VALUE_FIELDS:
        word = {"principal_max": "1st", "principal_middle": "2nd",
                "principal_min": "3rd"}.get(token)
        return f"{name} ({word})" if word else f"{name} magnitude"
    if token.startswith("principal_"):  # "max principal Cauchy stress"
        return f"{_VALUE_NAMES[token]} {name}"
    suffix = _VALUE_NAMES.get(token or "magnitude", token)
    return f"{name} {suffix}"


def _statistics(values, weights, quantity):
    """{statistic: value} over the region's items; NaN where undefined."""
    finite = np.isfinite(values) & np.isfinite(weights)
    v, w = values[finite], weights[finite]
    out = {}
    empty = not len(v)
    if "max" in quantity["statistics"] or quantity["where"]:
        out["max"] = np.nan if empty else float(v.max())
        out["argmax"] = -1 if empty else int(np.flatnonzero(finite)[
            int(np.argmax(v))])
    if "min" in quantity["statistics"]:
        out["min"] = np.nan if empty else float(v.min())
    total = float(w.sum()) if len(w) else 0.0
    if "mean" in quantity["statistics"]:
        out["mean"] = float((v * w).sum() / total) if total else np.nan
    if "amean" in quantity["statistics"]:
        out["amean"] = np.nan if empty else float(v.mean())
    if "integral" in quantity["statistics"]:
        out["integral"] = np.nan if empty else float((v * w).sum())
    if "sum" in quantity["statistics"]:
        out["sum"] = np.nan if empty else float(v.sum())
    if "rms" in quantity["statistics"]:
        out["rms"] = np.nan if empty else float(np.sqrt(np.mean(v * v)))
    if "std" in quantity["statistics"]:
        out["std"] = np.nan if empty else float(v.std())
    if "count" in quantity["statistics"]:
        out["count"] = int(len(v))
    for p in quantity["percentiles"]:
        out[f"P{p:g}"] = np.nan if empty else float(np.percentile(v, p))
    return out


def _export_steps(node, count):
    first = max(0, int(node.evalParm("export_step_first")))
    last = int(node.evalParm("export_step_last"))
    last = count - 1 if last < 0 else min(last, count - 1)
    every = max(1, int(node.evalParm("export_step_every")))
    steps = list(range(first, last + 1, every))
    if not steps:
        raise ExportError(f"No steps between {first} and {last}: the run has "
                          f"steps 0 to {count - 1}.")
    return steps


class _NoProgress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def updateLongProgress(self, *args):
        pass


def _progress(label):
    if hou.isUIAvailable():
        return hou.InterruptableOperation(
            label, long_operation_name=label, open_interrupt_dialog=True)
    return _NoProgress()


def _manifest(pvd_path):
    path = os.path.join(os.path.dirname(os.path.abspath(pvd_path)),
                        "run-manifest.json")
    try:
        with open(path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


class _ForceColumns:
    """The chosen force curves (Analysis > Force Curves) as columns, read
    in the export's own pass over the steps."""

    def __init__(self, node, pvd_path):
        self.note = None
        self.sets = []
        self.values = []  # per step: {set: (reaction, contact, displacement)}
        if not node.evalParm("export_forces"):
            return
        try:
            _, definitions = _force_definitions(node)
        except ForceCurveError as exc:
            self.note = f"Force curves not included: {exc}"
            return
        chosen = {int(t) for t in node.evalParm("export_force_sets").split()
                  if t.isdigit()}
        self.sets = [i for i in range(1, len(definitions) + 1)
                     if not chosen or i in chosen]
        self.definitions = definitions
        self.record_key = _file_key(scene_record_path(pvd_path))
        self.quantities = [(name, label, dimension) for name, label, dimension
                           in (("reaction", "reaction", "force"),
                               ("contact", "contact force", "force"),
                               ("displacement", "mean displacement",
                                "length"))
                           if node.evalParm(f"export_force_{name}")]
        self.components = [axis for axis in ("x", "y", "z", "magnitude")
                           if node.evalParm(f"export_force_{axis}")]
        if not (self.sets and self.quantities and self.components):
            self.note = ("Force curves: nothing chosen (sets, forces or "
                         "components).")
            self.sets = []

    def add(self, mesh):
        """One step (mesh None: the step could not be read)."""
        if not self.sets:
            return
        if mesh is None:
            self.values.append(None)
            return
        nodes = _force_nodes(self.record_key, mesh, self.definitions, self.sets)
        self.values.append(_frame_set_values(mesh, nodes))

    def columns(self, units, run_has_forces):
        if not self.sets:
            return []
        if not any(values is not None for values in self.values) and not run_has_forces():
            self.note = f"Force curves not included: {_MISSING_FORCE_FIELDS}"
            return []
        geometries = len({e.get("geometry") for e in self.definitions
                          if e.get("kind") != "obstacle"})
        out = []
        nothing = (np.full(3, np.nan),) * 3
        for index in self.sets:
            entry = self.definitions[index - 1]
            name = (f"Obstacle {entry.get('geometry')}"
                    if entry.get("kind") == "obstacle"
                    else _short_sideset_label(entry, geometries))
            for position, (_, label, dimension) in enumerate(
                    self.quantities):
                slot = ("reaction", "contact", "displacement").index(
                    self.quantities[position][0])
                factor, unit = units.convert(dimension)
                vectors = np.array([(step or {}).get(index, nothing)[slot]
                                    for step in self.values]) * factor
                for axis in self.components:
                    values = (np.linalg.norm(vectors, axis=1)
                              if axis == "magnitude"
                              else vectors[:, "xyz".index(axis)])
                    out.append((f"{name}: {label} {axis}"
                                + (f" ({unit})" if unit else ""), values))
        return out


def export_data(node, progress=None):
    """Compute and write the Data Over Time workbook; returns a summary."""
    import time as _time
    started = _time.time()
    pvd = node.evalParm("PVD_file")
    if not pvd or not os.path.isfile(pvd):
        raise ExportError("Load a PVD file first.")
    entries = read_pvd(pvd)
    if not entries:
        raise ExportError("The PVD file lists no steps.")
    steps = _export_steps(node, len(entries))
    units = _ExportUnits(node)
    region_count = int(node.evalParm("export_regions"))
    quantity_count = int(node.evalParm("export_quantities"))
    if quantity_count < 1:
        raise ExportError("Add at least one quantity.")
    quantities = [_quantity_definition(node, i)
                  for i in range(1, quantity_count + 1)]
    mesh = volume_mesh(pvd, steps[0])
    topology = volume_topology(mesh)
    identity = _identity(pvd, mesh, topology)
    regions = []
    definitions = [_region_definition(node, i)
                   for i in range(1, region_count + 1)] or [
        {"index": 1, "kind": "all", "name": "", "only": set()}]
    for definition in definitions:
        elements, nodes, label, description = region_members(
            node, definition, mesh, topology, identity, pvd)
        # One Element / One Node give one value: a node on an interface
        # between two bodies has a copy in each (averaged)
        regions.append({"elements": elements, "nodes": nodes,
                        "label": label, "description": description,
                        "kind": definition["kind"],
                        "single": definition["kind"] in ("element", "node")})
    seen = {}
    for region in regions:  # unique labels: they head the columns
        seen[region["label"]] = seen.get(region["label"], 0) + 1
        if seen[region["label"]] > 1:
            region["label"] = f"{region['label']} ({seen[region['label']]})"
    # Validate quantity fields against the exported steps rather than only the
    # playbar frame (a field absent at frame 0 may exist on later exported
    # steps). Scan endpoints first, falling back to intermediate steps only
    # when a requested field is still missing.
    step_fields = {}
    probe_order = list(dict.fromkeys([steps[0], steps[-1]] + list(steps)))
    missing_fields = {q["field"] for q in quantities
                      if q["field"] != "_volume"
                      and not _is_fiber_quantity(q["field"])}
    for step in probe_order:
        if not missing_fields:
            break
        try:
            info = frame_field_info(pvd, step).get("Volume", {})
        except Exception:
            continue
        avail = _available_fields_from_fields(_fields_from_info(info), True)
        step_fields.update(avail)
        missing_fields -= set(avail)
    components = {}
    for q in quantities:
        if q["field"] == "_volume" or _is_fiber_quantity(q["field"]):
            components[q["index"]] = 1
        else:
            components[q["index"]] = step_fields.get(
                q["field"], _field_components(node, q["field"]))
    for q in quantities:
        if not components[q["index"]]:
            raise ExportError(
                f"Quantity {q['index']}: '{q['field']}' is not in this "
                "result (or not on the Volume block).")
        if not q["statistics"] and not q["percentiles"] \
                and q["field"] != "_volume":
            raise ExportError(f"Quantity {q['index']}: choose at least one "
                              "statistic.")
    rest_points = np.asarray(mesh["points"], dtype=np.float64)
    rest_volume = element_volumes(topology, rest_points)
    sign = np.where(rest_volume < 0, -1.0, 1.0)
    rest_volume = rest_volume * sign
    owner, members = topology["owner"], topology["members"]
    point_node = topology["point_node"]

    def node_weights(volume):
        share = (volume / topology["sizes"])[owner]
        return np.bincount(point_node[members], weights=share,
                           minlength=topology["n_nodes"])

    rest_node_volume = node_weights(rest_volume)
    per_item = bool(node.evalParm("export_per_item"))
    item_mode = _menu_token(node, "export_per_item_steps") or "last"
    if item_mode == "all":
        item_steps = list(steps)
    elif item_mode == "screen":
        item_steps = [max(0, min(entry_index(node), len(entries) - 1))]
    else:
        item_steps = [steps[-1]]
    need_current = any(q["volume"] == "current" for q in quantities)
    items = {}       # quantity index -> {step: [values per region]}
    records = []     # per step: {(region, quantity, key): value}
    unreadable = []
    forces = _ForceColumns(node, pvd)
    mapping = time_mapping(node)
    rows_time = []
    # The per element / node sheet may want a step the table does not list
    # (Step on Screen outside First/Last/Every): read it too, for that sheet
    # only (review 2026-11-14, item 7).
    table_steps = set(steps)
    read_steps = sorted(table_steps | set(item_steps)) if per_item \
        else list(steps)
    item_unreadable = []
    total = len(read_steps)
    with (progress or _progress("Exporting data over time")) as operation:
        for position, step in enumerate(read_steps):
            operation.updateLongProgress(
                position / max(1, total),
                f"Step {step} ({position + 1} of {total})")
            in_table = step in table_steps
            time = entries[step][0]
            record = {}  # a step outside the table fills a throwaway record
            if in_table:
                rows_time.append((step, time, entry_frame(node, step)))
                records.append(record)
            try:
                frame_mesh = mesh if step == steps[0] \
                    else volume_mesh(pvd, step)
            except Exception as exc:  # noqa: BLE001 -- reported below
                # e.g. the last step of a run that is still writing
                if in_table:
                    unreadable.append((step, str(exc)))
                    forces.add(None)
                else:
                    item_unreadable.append((step, str(exc)))
                continue
            if (str(frame_mesh.get("topo_key")), len(frame_mesh["points"])) \
                    != topology["key"]:
                raise ExportError(
                    f"The mesh changes at step {step} (a remeshing run): the "
                    "export follows one mesh through the run.")
            if in_table:
                forces.add(frame_mesh)
            current_volume = current_node_volume = None
            if need_current:
                solution = _solution_vectors(frame_mesh)
                frame_points = np.asarray(frame_mesh["points"], dtype=np.float64)
                moved = frame_points if solution is None else frame_points + solution
                current_volume = element_volumes(topology, moved) * sign
                current_node_volume = node_weights(current_volume)
            for q in quantities:
                volume = current_volume if q["volume"] == "current" \
                    else rest_volume
                node_volume = current_node_volume \
                    if q["volume"] == "current" else rest_node_volume
                if q["field"] == "_volume":
                    for r, region in enumerate(regions):
                        record[(r, q["index"], "volume")] = float(
                            volume[region["elements"]].sum())
                    continue
                element_rows = np.unique(np.concatenate(
                    [region["elements"] for region in regions])) \
                    if q["basis"] == "elements" else np.zeros(0, np.int64)
                node_rows = np.unique(np.concatenate(
                    [region["nodes"] for region in regions])) \
                    if q["basis"] == "nodes" else np.zeros(0, np.int64)
                element_values, node_values = point_or_element_values(
                    frame_mesh, topology, q["field"], q["value"],
                    element_rows, node_rows, pvd)
                if q["basis"] == "elements":
                    lookup = np.full(topology["n_elements"], np.nan)
                    lookup[element_rows] = element_values
                    weight_of = volume
                else:
                    lookup = np.full(topology["n_nodes"], np.nan)
                    lookup[node_rows] = node_values
                    weight_of = node_volume
                for r, region in enumerate(regions):
                    rows = region[q["basis"]]
                    values, weights = lookup[rows], weight_of[rows]
                    if region["single"]:
                        finite = values[np.isfinite(values)]
                        record[(r, q["index"], "value")] = \
                            float(finite.mean()) if len(finite) else np.nan
                    else:
                        for key, value in _statistics(values, weights,
                                                      q).items():
                            record[(r, q["index"], key)] = value
                    if per_item and step in item_steps:
                        items.setdefault(q["index"], {}).setdefault(
                            step, []).append(values)
    if len(unreadable) == len(steps):
        raise ExportError("None of the chosen steps could be read.")
    keys = []
    for record in records:
        keys.extend(key for key in record if key not in keys)
    columns = {key: [record.get(key, np.nan) for record in records]
               for key in keys}
    # ---- columns --------------------------------------------------------
    header_units = {}
    for q in quantities:
        dimension = quantity_dimension(q["field"])
        if q["value"] == "determinant" and components[q["index"]] == 9:
            # det(T) scales as the cube of a 3x3 tensor's unit.
            if dimension == "stress":
                dimension = "stress3"
            elif dimension not in ("none", None):
                dimension = None
        header_units[q["index"]] = units.convert(dimension)
    tfactor, tunit = units.convert("time")
    data_header = ["Step", "Time" + (f" ({tunit})" if tunit else ""),
                   "Frame"]
    data_columns = []
    for r, region in enumerate(regions):
        for q in quantities:
            name = _quantity_header(q, components[q["index"]])
            factor, unit = header_units[q["index"]]
            if q["field"] == "_volume":
                vfactor, vunit = units.convert("volume")
                header = f"{region['label']}: {name}" + (
                    " (rest)" if q["volume"] == "rest" else "") + (
                    f" ({vunit})" if vunit else "")
                data_columns.append((header, np.asarray(columns[(
                    r, q["index"], "volume")]) * vfactor))
                continue
            single = (r, q["index"], "value") in columns
            keys = ["value"] if single else (
                [key for key, _ in _STATISTICS if key in q["statistics"]]
                + [f"P{p:g}" for p in q["percentiles"]])
            for key in keys:
                values = np.asarray(columns[(r, q["index"], key)],
                                    dtype=np.float64)
                label = dict(_STATISTICS).get(key, key)
                if key == "count":
                    unit_text = ""
                    values_out = values
                elif key == "integral":
                    if unit is None:
                        values_out = values
                        unit_text = ""
                    else:
                        vfactor, vunit = units.convert("volume")
                        values_out = values * factor * vfactor
                        unit_text = f"{unit}*{vunit}" if unit else vunit
                else:
                    values_out = values * factor
                    unit_text = unit or ""
                header = f"{region['label']}: {name}" + (
                    "" if single else f", {label}") + (
                    f" ({unit_text})" if unit_text else "")
                data_columns.append((header, values_out))
            if q["where"] and not single:
                where = np.asarray(columns[(r, q["index"], "argmax")],
                                   dtype=np.float64)
                where = np.where(np.isfinite(where), where, -1).astype(
                    np.int64)
                rows = region[q["basis"]]
                lfactor, lunit = units.convert("length")
                centroids, node_positions = _rest_positions(mesh, topology)
                positions = centroids if q["basis"] == "elements" \
                    else node_positions
                numbers = identity["element_number"] \
                    if q["basis"] == "elements" else identity["node_number"]
                picked = (np.where(where >= 0, rows[np.maximum(where, 0)], -1)
                          if len(rows) else np.full(len(where), -1,
                                                    dtype=np.int64))
                word = "element" if q["basis"] == "elements" else "node"
                data_columns.append((
                    f"{region['label']}: {name}, max at {word}",
                    np.where(picked >= 0, numbers[np.maximum(picked, 0)],
                             np.nan)))
                for axis, label in enumerate("xyz"):
                    data_columns.append((
                        f"{region['label']}: {name}, max at {label}" + (
                            f" ({lunit})" if lunit else ""),
                        np.where(picked >= 0, positions[np.maximum(
                            picked, 0), axis] * lfactor, np.nan)))
    def other_steps_have_forces():
        names = {f"{term}_forces" for term in FORCE_TERMS}
        for step in range(len(entries)):
            if step in table_steps:
                continue
            try:
                fields = frame_field_info(pvd, step).get("Volume", {})
            except (OSError, VtuParseError, ET.ParseError, ValueError):
                continue
            if names.intersection(fields.get("point_data", {})):
                return True
        return False

    data_columns.extend(forces.columns(units, other_steps_have_forces))
    force_note = forces.note
    # ---- workbook -----------------------------------------------------------
    path = node.evalParm("export_data_file").strip()
    if not path:
        stem = os.path.splitext(os.path.basename(pvd))[0]
        path = os.path.join(export_folder(node), f"{stem}_data.xlsx")
    path = os.path.abspath(os.path.expanduser(path))
    if not path.lower().endswith(".xlsx"):
        path += ".xlsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    book = XlsxWorkbook()
    headers = data_header + [header for header, _ in data_columns]

    def data_rows():
        yield headers
        for position, (step, time, frame) in enumerate(rows_time):
            yield [step, time * tfactor, frame] + [
                values[position] for _, values in data_columns]

    widths = [6, 12, 8] + [min(60, max(12, len(h) * 0.9))
                           for h, _ in data_columns]
    book.add_sheet("Data", data_rows(), widths=widths)
    if per_item:
        lfactor, lunit = units.convert("length")
        centroids, node_positions = _rest_positions(mesh, topology)
        for q in quantities:
            if q["field"] == "_volume":
                continue
            word = "Element" if q["basis"] == "elements" else "Node"
            factor, unit = header_units[q["index"]]
            positions = centroids if word == "Element" else node_positions
            numbers = identity["element_number"] if word == "Element" \
                else identity["node_number"]
            geometry = identity["element_geometry"] if word == "Element" \
                else identity["node_geometry"]
            bodies = topology["body"] if word == "Element" \
                else topology["node_body"]
            count = sum(len(region[q["basis"]]) for region in regions)
            if count + 1 > XLSX_MAX_ROWS:
                raise ExportError(
                    f"Per {word.lower()} values: {count:,} rows do not fit "
                    f"in an Excel sheet ({XLSX_MAX_ROWS:,}); choose a smaller "
                    "region or turn this off.")
            name = _quantity_header(q, components[q["index"]])
            unit_suffix = f" ({unit})" if unit else ""
            header = ["Region", "Geometry", word, "Body"] + [
                f"{axis} ({lunit})" if lunit else axis for axis in "xyz"] + [
                f"Step {step}: t = {entries[step][0] * tfactor:.6g}" + (
                    f" {tunit}" if tunit else "") + unit_suffix
                for step in item_steps]
            by_step = items.get(q["index"], {})

            def item_rows(q=q, header=header, by_step=by_step,
                          positions=positions, numbers=numbers,
                          geometry=geometry, bodies=bodies, factor=factor):
                yield header
                for r, region in enumerate(regions):
                    rows = region[q["basis"]]
                    # an unreadable step keeps its column, empty
                    stacked = [by_step[step][r] * factor if step in by_step
                               else np.full(len(rows), np.nan)
                               for step in item_steps]
                    for k, row in enumerate(rows):
                        yield ([region["label"], int(geometry[row]),
                                int(numbers[row]), int(bodies[row])]
                               + list(positions[row] * lfactor)
                               + [values[k] for values in stacked])

            book.add_sheet(f"{name} per {word.lower()}", item_rows(),
                           widths=[14, 9, 10, 8, 12, 12, 12]
                           + [16] * len(item_steps))
    about = _about_rows(node, pvd, entries, steps, regions, quantities,
                        components, units, identity, force_note, mapping)
    book.add_sheet("About", about, widths=[30, 110], header_rows=0)
    try:
        written = book.save(path, title=f"PolyFEM results: {pvd}",
                            creator="Read PVD 1.0")
    except XlsxLimitError as exc:
        raise ExportError(f"The table does not fit in Excel: {exc}.")
    except PermissionError:
        raise ExportError(f"Could not write {path}: is it open in Excel?")
    seconds = _time.time() - started
    lines = [f"Wrote {path}",
             f"Data: {len(steps)} steps x {len(data_columns)} series "
             f"({seconds:.1f} s)."]
    for region in regions:
        lines.append(f"{region['label']}: {len(region['elements'])} "
                     f"elements, {len(region['nodes'])} nodes.")
    if force_note:
        lines.append(force_note)
    if unreadable:
        lines.append(
            f"Step(s) {', '.join(str(step) for step, _ in unreadable)} could "
            f"not be read (empty cells; a run still writing?): "
            f"{unreadable[0][1]}")
    if item_unreadable:
        lines.append(
            f"Per element / node sheet: step(s) "
            f"{', '.join(str(step) for step, _ in item_unreadable)} could not "
            f"be read (empty column): {item_unreadable[0][1]}")
    if not units.known:
        lines.append(f"No unit labels: {units.why}")
    return {"path": path, "steps": steps, "headers": headers,
            "columns": data_columns, "rows": rows_time, "regions": regions,
            "sheets": written, "status": "\n".join(lines)}


def _about_rows(node, pvd, entries, steps, regions, quantities, components,
                units, identity, force_note, mapping):
    import datetime
    manifest = _manifest(pvd) or {}
    build = (manifest.get("build") or {}).get("sources") or {}
    commit = (build.get("polyfem") or {}).get("commit", "")
    input_file = ((manifest.get("input") or {}).get("file") or {})
    completion = manifest.get("completion") or {}
    rows = [["Read PVD export: data over time"], [],
            ["PVD file", pvd],
            ["Run folder", run_folder(pvd)],
            ["Run id", manifest.get("run_id", "(no run-manifest.json)")],
            ["Run status", completion.get("status", "")],
            ["PolyFEM commit", commit],
            ["Input params.json sha256", input_file.get("sha256", "")],
            ["Houdini scene", hou.hipFile.path()],
            ["Exported", datetime.datetime.now().astimezone().isoformat(
                timespec="seconds")],
            ["Exported by", f"Read PVD 1.0, Houdini "
                            f"{hou.applicationVersionString()}"],
            [],
            ["Steps", f"{len(steps)} of the run's {len(entries)} output "
                      f"steps: {steps[0]} to {steps[-1]}"
                      + (f", every {steps[1] - steps[0]}"
                         if len(steps) > 1 else "")],
            ["Time", "simulation time of each output step (the PVD time)"],
            ["Frame", "the Houdini frame that shows the step ("
             + ("Time Mapping: simulation time" if mapping is not None
                else "one frame per output step") + ")"],
            ["Units", units.describe()],
            [],
            ["Regions", "fixed on the rest shape: the same elements and "
                        "nodes at every step"]]
    for region in regions:
        rows.append([region["label"],
                     f"{region['description']}: {len(region['elements'])} "
                     f"elements, {len(region['nodes'])} nodes"])
    rows.append([])
    rows.append(["Quantities", ""])
    for q in quantities:
        name = _quantity_header(q, components[q["index"]])
        source = "the region's volume" if q["field"] == "_volume" else (
            f"field {_display_name(q['field'])}"
            + (f", value {q['value']}" if components[q["index"]] > 1 else "")
            + f", per {q['basis'][:-1]}")
        rows.append([name, source + f"; volumes: {q['volume']} (" + (
            "deformed, at each step" if q["volume"] == "current"
            else "undeformed") + ")"])
    rows.append([])
    rows.append(["Element value", "the mean of the element's nodal values "
                 "in PolyFEM's output (exact for linear elements, whose "
                 "fields are constant)"])
    rows.append(["Node value", "the mean of the node's copies, one per "
                 "element (the nodal average Smooth Field shows)"])
    rows.append(["Numbering", identity["scheme"]])
    for name, text in _STATISTIC_DEFINITIONS:
        rows.append([name, text])
    rows.append(["Empty cell", "no value at that step (e.g. the field is "
                 "missing in that step's file)"])
    if force_note:
        rows.append(["Force curves", force_note])
    elif node.evalParm("export_forces"):
        rows.append(["Force curves", "reaction: the force the prescribed "
                     "displacement (or an obstacle's motion) applies to the "
                     "body there; contact force: contact + friction; mean "
                     "displacement of the set's nodes (Analysis > Force "
                     "Curves)"])
    return rows


def export_data_button(kwargs):
    """Export Spreadsheet button."""
    node = kwargs["node"]
    try:
        summary = export_data(node)
        status = summary["status"]
    except hou.OperationInterrupted:
        summary, status = None, "Cancelled."
    except (ExportError, OSError, MshParseError, VtuParseError,
            ET.ParseError, ValueError) as exc:
        summary, status = None, f"Not exported: {exc}"
    node.parm("export_data_status").set(status)
    if summary is None and not status.startswith("Cancelled"):
        _message(status)
    elif summary is not None and hou.isUIAvailable():
        hou.ui.setStatusMessage(f"Data written to {summary['path']}",
                                hou.severityType.ImportantMessage)
    return summary


def open_export_data(kwargs):
    """Open Spreadsheet button (exports first when there is no file)."""
    node = kwargs["node"]
    path = node.evalParm("export_data_file").strip()
    if path and not path.lower().endswith(".xlsx"):
        path += ".xlsx"
    if not path and node.evalParm("PVD_file"):
        stem = os.path.splitext(os.path.basename(node.evalParm("PVD_file")))[0]
        path = os.path.join(export_folder(node), f"{stem}_data.xlsx")
    if not path or not os.path.isfile(path):
        summary = export_data_button(kwargs)
        if summary is None:
            return
        path = summary["path"]
    _open_with_system(path)
