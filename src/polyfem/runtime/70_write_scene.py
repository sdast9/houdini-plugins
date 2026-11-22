# =============================================================================
# write / run / log
# =============================================================================


CHECKPOINT_RESUME_NOTE = (
    "Resume uses the checkpoint's saved geometry, materials, time and solver "
    "settings, not this node's physics controls. Working Directory selects "
    "the new output location. Binary, maximum threads and log level still "
    "come from this node. The solver validates checkpoint compatibility.")


def _resume_checkpoint(parent):
    parm = parent.parm("resume_checkpoint")
    value = parm.eval().strip() if parm is not None else ""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value))) if value else ""


def build_params(parent, cache=None):
    working_dir = parent.evalParm("working_dir")
    if not working_dir.strip():
        raise hou.NodeError("Choose a Working Directory first.")
    try:
        os.makedirs(working_dir, exist_ok=True)
    except OSError as exc:
        raise hou.NodeError(
            f"Could not create the working directory {working_dir}: {exc}")
    input_dir = os.path.join(working_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)

    checkpoint = _resume_checkpoint(parent)
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise hou.NodeError(f"Resume Checkpoint not found: {checkpoint}")
        params_path = os.path.join(input_dir, "params.json")
        data = {"input": {"checkpoint": {
            "path": checkpoint, "reorder": bool(parent.evalParm("resume_reorder"))}}}
        with open(params_path, "w") as handle:
            json.dump(data, handle, indent=4)
        report = parent.parm("export_report")
        if report is not None:
            report.set(CHECKPOINT_RESUME_NOTE)
        return params_path

    build_remesh(parent)
    _raise_saint_venant_axes(parent)
    extra_params = _parse_extra_params_json(parent)
    _guard_forward_json(extra_params)
    if cache is None:
        cache = {}
    data = {}
    plans, motions = {}, {}
    orders = build_geometry_and_materials(parent, data, input_dir, plans,
                                          motions, cache=cache)
    build_time(parent, data)
    build_contact(parent, data)
    build_conditions(parent, data, plans, motions)
    build_space(parent, data, orders)
    build_solver(parent, data)
    build_output(parent, data)
    build_provenance(parent, data)
    sideset_override = scene_override = False
    if extra_params:
        pre_time = _digest(data.get("time"))
        pre_override = _digest([data["geometry"], data.get("boundary_conditions")])
        _deep_merge_json(data, extra_params)
        sideset_override = pre_override != _digest(
            [data.get("geometry"), data.get("boundary_conditions")])
        scene_override = sideset_override or pre_time != _digest(data.get("time"))
    _guard_forward_json(data)

    params_path = os.path.join(input_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(data, f, indent=4)
    if sideset_override:
        path = os.path.join(input_dir, SIDESET_RECORD_FILE)
        if os.path.isfile(path):
            os.remove(path)
    else:
        write_sideset_record(input_dir, data, plans)
    if scene_override:
        path = os.path.join(input_dir, SCENE_RECORD_FILE)
        if os.path.isfile(path):
            os.remove(path)
    else:
        write_scene_record(parent, input_dir, data, plans, motions, cache=cache)
    _set_export_report(parent, plans)
    if scene_override:
        note = ("Extra JSON changed geometry, boundary conditions or time. "
                "The HDA scene record was omitted because it would describe "
                "different settings.")
        report = parent.parm("export_report")
        if report is not None:
            report.set("\n".join(filter(None, (report.eval(), note))))
        _status(note)
    return params_path


# Why a method other than Newton is not usable with contact
# (polyfem/docs/qn-contact-investigation-20260922.md).
NON_NEWTON_CONTACT_WARNING = (
    "Contact is on and the nonlinear solver is {method}: only Newton "
    "converges on contact scenes. The contact barrier makes the problem so "
    "stiff that {method} stalls far from the solution; a step finishes only "
    "on the gradient tolerance, which it may never reach (the run then stops "
    "with a named failure). Choose Newton under Solver > Nonlinear unless you "
    "are experimenting.")


def _warn_non_newton_contact(parent, data):
    """Warn (without refusing) when a contact scene uses a method other than
    Newton; returns the message, or None. Write and Run show the same
    warning through Check Setup."""
    method = data.get("solver", {}).get("nonlinear", {}).get("solver", "Newton")
    if method == "Newton" or not data.get("contact", {}).get("enabled", False):
        return None
    message = NON_NEWTON_CONTACT_WARNING.format(method=method)
    _message(message)
    return message


def build_provenance(parent, data):
    """Who wrote this input (RB-12): the solver copies the block verbatim into
    its run manifest (``run-manifest.json``, ``producer``) so a result can be
    traced back to the asset definition and scene that produced it. Needs a
    PolyFEM build that knows ``/provenance`` (1f6f826fa or later; earlier
    strict builds refuse unknown keys)."""
    import datetime
    import hashlib

    definition = parent.type().definition()
    library = definition.libraryFilePath() if definition is not None else ""
    asset_sha = ""
    if library and os.path.isfile(library):
        digest = hashlib.sha256()
        with open(library, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        asset_sha = digest.hexdigest()
    try:
        version = hou.applicationVersionString()
    except Exception:
        version = ".".join(str(v) for v in hou.applicationVersion())
    data["provenance"] = {
        "producer": "houdini",
        "producer_version": version,
        "asset": f"{parent.type().name()} ({os.path.basename(library)})" if library else parent.type().name(),
        "asset_version": (definition.version() if definition is not None else "") or "",
        "asset_sha256": asset_sha,
        "scene": hou.hipFile.path(),
        "exported_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _sideset_condition_lists(data):
    """The sideset part of boundary_conditions, which the record describes."""
    conditions = data.get("boundary_conditions")
    if not isinstance(conditions, dict):
        return {}
    return {key: conditions[key] for key in _BC_LIST_KEYS if key in conditions}


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_file_names(data):
    names = []
    geometry = data.get("geometry")
    for entry in geometry if isinstance(geometry, list) else []:
        if not isinstance(entry, dict):
            continue
        for key in ("surface_selection", "point_selection"):
            selections = entry.get(key)
            if isinstance(selections, list):
                names.extend(
                    selection["file"] for selection in selections
                    if isinstance(selection, dict)
                    and isinstance(selection.get("file"), str))
    return names


def write_sideset_record(input_dir, data, plans):
    """input/hda_sidesets.json: the sidesets as entered, for Import.

    params.json and the selection files hold what PolyFEM needs (one id per
    face, merged conditions); this keeps what the user typed -- "*", typed
    selections, group types and every condition -- with hashes of the files it
    belongs to, so an edited params.json is never contradicted on import.
    """
    record = {
        "schema": SIDESET_RECORD_SCHEMA, "version": 1,
        "boundary_conditions_sha256": _digest(_sideset_condition_lists(data)),
        "selection_files": {
            name: _file_digest(os.path.join(input_dir, name))
            for name in _selection_file_names(data)
            if os.path.isfile(os.path.join(input_dir, name))},
        "geometries": {
            str(geo): {
                "sidesets": plan["records"],
                "combinations": {
                    str(sid): owners
                    for sid, owners in sorted(plan["combinations"].items())}}
            for geo, plan in sorted(plans.items())}}
    path = os.path.join(input_dir, SIDESET_RECORD_FILE)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(record, handle, indent=2)
    os.replace(temporary, path)


# Written next to params.json: what Read PVD and Import need beyond it -- the
# timeline (Houdini frames <-> simulation time), each animated obstacle's
# transform keyframes, and the force sets: every sideset's faces (or nodes)
# and every obstacle's surface in simulation coordinates, over which Read
# PVD sums PolyFEM's nodal forces. PolyFEM never reads it.
SCENE_RECORD_FILE = "hda_scene.json"
SCENE_RECORD_SCHEMA = "polyfem-houdini-scene"


def _vertex_positions(parent, geo, matrix):
    """PolyFEM's vertices of one geometry's .msh in simulation coordinates
    (float64 from the file, PolyFEM's numbering, the rest transform)."""
    reader = hou.nodeType(hou.sopNodeTypeCategory(), "MSH_Reader::3.0")
    mesh = reader.hdaModule().read_msh(parent.evalParm(f"file_location{geo}"))
    points = np.asarray(mesh["points"], dtype=np.float64)
    tags = np.asarray(mesh["node_tags"], dtype=np.int64)
    if len(tags) and int(tags.max()) == len(tags):
        by_vid = np.empty_like(points)
        by_vid[tags - 1] = points  # tag - 1 when the tags are exactly 1..N
    else:
        by_vid = points  # file order otherwise (io/MshReader.cpp)
    return by_vid @ matrix[:3, :3] + matrix[3, :3]


def _force_set_label(geo, entry):
    conditions = ", ".join(entry["conditions"]) or "no condition"
    return (f"Geometry {geo} subdomain {entry['subdomain']} sideset "
            f"{entry['sideset']} ({conditions})")


def write_scene_record(parent, input_dir, data, plans, motions, cache=None):
    """input/hda_scene.json (see SCENE_RECORD_FILE)."""
    try:
        line = timeline(parent)
    except hou.Error:
        line = None
    record = {"schema": SCENE_RECORD_SCHEMA, "version": 1,
              "params_sha256": _digest(data),
              "timeline": line.record() if line is not None else None,
              "obstacles": {}, "force_sets": []}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if not parent.parm(f"is_enabled{geo}").eval() \
                or parent.node(f"geo_{geo}") is None:
            continue
        matrix = rest_matrix(parent, geo)
        if parent.parm(f"is_obstacle{geo}").eval():
            motion = (motions or {}).get(geo, {"kind": "static"})
            vertices, _, exact = _cached(
                cache, "obstacles", int(geo),
                lambda g=geo: obstacle_surface(parent, g))
            vertices = vertices @ matrix[:3, :3] + matrix[3, :3]
            record["obstacles"][str(geo)] = {
                "motion": motion["kind"],
                "mesh": parent.parm(f"file_location{geo}").unexpandedString(),
                "transform": {parm.name(): _keyframes_record(parm)
                              for parm in _transform_parms(parent, geo)}
                if motion["kind"] != "static" else {}}
            record["force_sets"].append({
                "label": f"Obstacle {geo} (force on the obstacle)",
                "kind": "obstacle", "geometry": geo, "body": 0,
                "vertices": vertices.tolist(), "exact": exact})
            continue
        plan = (plans or {}).get(geo)
        if not plan or not plan.get("sets"):
            continue
        positions = _vertex_positions(parent, geo, matrix)
        for entry in plan["sets"]:
            vol = entry["subdomain"]
            if entry["kind"] == "faces":
                corners = plan["face_vids"][entry["members"]]
            else:
                corners = plan["point_vids"][entry["members"]].reshape(-1, 1)
            used, inverse = np.unique(corners[corners >= 0],
                                      return_inverse=True)
            index = np.full(corners.shape, -1, dtype=np.int64)
            index[corners >= 0] = inverse
            record["force_sets"].append({
                "label": _force_set_label(geo, entry),
                "kind": entry["kind"], "geometry": geo, "subdomain": vol,
                "sideset": entry["sideset"], "body": vol_id(geo, vol),
                "conditions": entry["conditions"],
                "order": int(parent.evalParm(f"mainOrder{geo}_{vol}")) + 1,
                "vertices": positions[used].tolist(),
                "faces": index.tolist()})
    record["materials"] = _material_ui_records(parent, input_dir, data)
    path = os.path.join(input_dir, SCENE_RECORD_FILE)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(record, handle)
    os.replace(temporary, path)
    return record


# =============================================================================
# Fiber settings in the scene record (2026-11-14)
#
# params.json holds what PolyFEM reads: a symmetric +/-theta family as two
# expanded models with their own direction arrays, and every per-element
# source (attribute, cylindrical, curve) as a file reference. Import alone can
# only rebuild those as file-based families, which give the same material but
# lose the settings that made them (mirror, half angle, axis, source). The
# scene record keeps each fiber subdomain's Material settings; Import puts
# them back only when they reproduce the imported material block and the
# per-element values in its files exactly, and otherwise keeps the file-based
# families.
# =============================================================================


def _ui_value(parm):
    """A parameter's value in the form setParms takes back."""
    kind = parm.parmTemplate().type()
    if kind == hou.parmTemplateType.Menu:
        return parm.evalAsString()
    if kind == hou.parmTemplateType.String:
        return parm.unexpandedString()
    return parm.eval()


def _material_ui_parms(parent, geo, vol):
    """{name: value} of every setting in one subdomain's Material folder,
    fiber families included (their count under num_fiber_families<g>_<v>)."""
    folder = parent.parmTemplateGroup().find("material#_#")
    values = {}
    skip = (hou.parmTemplateType.Separator, hou.parmTemplateType.Label,
            hou.parmTemplateType.Button)

    def visit(templates):
        for template in templates:
            if template.type() == hou.parmTemplateType.Folder:
                if template.name() != "num_fiber_families#_#":
                    visit(template.parmTemplates())
                continue
            if template.type() in skip:
                continue
            name = template.name().replace("#", str(geo), 1) \
                .replace("#", str(vol), 1)
            tuple_parm = parent.parmTuple(name)
            if tuple_parm is None:
                continue
            for parm in tuple_parm:
                values[parm.name()] = _ui_value(parm)

    if folder is not None:
        visit(folder.parmTemplates())
    count = parent.parm(f"num_fiber_families{geo}_{vol}")
    if count is not None:
        values[count.name()] = count.eval()
        for parm in count.multiParmInstances():
            if parm.parmTemplate().type() not in skip:
                values[parm.name()] = _ui_value(parm)
    return values


def _apply_material_ui(parent, geo, vol, values):
    count = f"num_fiber_families{geo}_{vol}"
    if count in values and parent.parm(count) is not None:
        parent.parm(count).set(values[count])
    parent.setParms({name: value for name, value in values.items()
                     if name != count and parent.parm(name) is not None})


def _material_file_refs(material):
    """Names of the per-element files one material block reads."""
    found = set()

    def walk(value):
        if isinstance(value, dict):
            if value.get("type") == "per_element_file" \
                    and isinstance(value.get("path"), str):
                found.add(value["path"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and _export_writes(value) \
                and value.lower().endswith(".txt"):
            found.add(value)

    walk(material)
    return sorted(found)


def _material_ui_records(parent, input_dir, data):
    """{material id: {"material", "files", "parms"}} of every enabled
    simulated subdomain with a fiber or composite material."""
    by_id = {entry.get("id"): entry for entry in data.get("materials", [])
             if isinstance(entry, dict)}
    records = {}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval() \
                or not parent.parm(f"is_enabled{geo}").eval() \
                or parent.node(f"geo_{geo}") is None:
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            token = material_token(parent, geo, vol)
            material = by_id.get(vol_id(geo, vol))
            if material is None or (token != "MaterialSum"
                                    and token not in FIBER_MODELS):
                continue
            files = {}
            for name in _material_file_refs(material):
                path = os.path.join(input_dir, name)
                if os.path.isfile(path):
                    files[name] = _file_digest(path)
            records[str(vol_id(geo, vol))] = {
                "material": material, "files": files,
                "parms": _material_ui_parms(parent, geo, vol)}
    return records


def _material_ui_reproduces(parent, geo, vol, material, params_dir):
    """Check the material and the file rows owned by this subdomain."""
    built = json.loads(json.dumps(build_material(parent, geo, vol)))
    fields, _ = per_element_data(parent, geo)
    branch = parent.node(f"branch_{geo}").geometry()
    order = solver_element_order(branch, expect_msh=True)
    entities = np.frombuffer(branch.primIntAttribValuesAsString("Entity"), dtype=np.int32)
    mask = _volume_mask(branch)
    owned = (entities[mask] if mask is not None else entities)[order] == vol
    table, total = global_element_table(parent)
    offset = next((start for index, start, _ in table if index == geo), None)
    if offset is None:
        return False
    stored_rows = offset + np.flatnonzero(owned)
    directions = {}
    for fam, _ in material_families(parent, geo, vol):
        for side in mirror_sides(parent, geo, vol, fam):
            values = fields.get(fiber_attrib(vol, fam, side))
            if values is not None:
                directions[fiber_field(geo, vol, fam, side)] = values[order][owned]
        kappa = fields.get(scalar_attrib("kappa", vol, fam))
        if kappa is not None:
            stored = np.loadtxt(os.path.join(
                params_dir, per_element_scalar_file(geo, vol, fam, "kappa")), ndmin=1)
            if len(stored) != total or not np.allclose(
                    kappa[order][owned], stored[stored_rows], rtol=1e-8, atol=0):
                return False

    # The source settings may emit a local field name or a constant while
    # the actual solver input references a shared global column. Compare
    # directions on this subdomain before treating the references as equal.
    local_models = built.get("models", [built])
    stored_models = material.get("models", [material])
    for local, stored_model in zip(local_models, stored_models):
        if not isinstance(local, dict) or not isinstance(stored_model, dict):
            continue
        ref = stored_model.get("fiber_direction")
        if not isinstance(ref, dict) or ref.get("path") != FIBER_FILE:
            continue
        source = local.get("fiber_direction")
        if isinstance(source, dict) and source.get("path") == FIBER_FILE:
            values = directions.get(source.get("field"))
        elif isinstance(source, list) and len(source) == 3 and all(
                type(v) in (int, float) for v in source):
            values = _normalize_fibers(np.asarray([source], dtype=float),
                                       _slot_label(geo, vol, None))
        else:
            values = None
        if values is None:
            return False
        stored = read_cell_vectors_legacy_vtk(
            os.path.join(params_dir, FIBER_FILE), ref["field"])
        if len(stored) != total or not np.allclose(
                values, stored[stored_rows], rtol=0, atol=1e-8):
            return False
        local["fiber_direction"] = ref
    return built == material


def _restore_material_ui(parent, scene, materials, params_dir, warnings):
    """Import: put recorded fiber settings back where they reproduce the
    imported material exactly (see the section comment)."""
    records = (scene or {}).get("materials")
    if not isinstance(records, dict):
        return
    for material in materials:
        if not isinstance(material, dict):
            continue
        entry = records.get(str(material.get("id")))
        if not isinstance(entry, dict) or entry.get("material") != material:
            continue
        geo, vol = divmod(int(material["id"]), 1000)
        if parent.parm(f"materials{geo}_{vol}") is None:
            continue
        files = entry.get("files") or {}
        if any(not os.path.isfile(os.path.join(params_dir, name))
               or _file_digest(os.path.join(params_dir, name)) != digest
               for name, digest in files.items()):
            continue
        imported = _material_ui_parms(parent, geo, vol)
        try:
            _apply_material_ui(parent, geo, vol, entry.get("parms") or {})
            restored = _material_ui_reproduces(parent, geo, vol, material,
                                               params_dir)
        except (hou.Error, OSError, ValueError, TypeError, KeyError):
            restored = False
        if not restored:
            _apply_material_ui(parent, geo, vol, imported)
            warnings.append(
                f"Geometry {geo} subdomain {vol}: its fiber settings as "
                "entered (sources, symmetric +/-theta families) no longer "
                "reproduce the run's fibers here (for example a SOP or file "
                "they read is missing or changed), so its families are "
                "imported as per-element files, which give the same "
                "material.")


def load_scene_record(input_dir, data=None, notes=None):
    path = os.path.join(input_dir, SCENE_RECORD_FILE)
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) \
            or record.get("schema") != SCENE_RECORD_SCHEMA:
        return None
    if data is None:
        data = _read_json(os.path.join(input_dir, "params.json"))
    if not isinstance(data, dict) or record.get("params_sha256") != _digest(data):
        if notes is not None:
            notes.append(f"{SCENE_RECORD_FILE} does not match params.json. "
                         "Obstacle motion and other scene settings were "
                         "restored from the JSON where possible.")
        return None
    return record


def load_sideset_record(input_dir, data, resource_dir=None):
    """(record, note): the HDA's sideset record if it still matches params.json.

    note explains why an existing record was not used; it is None when there
    is no record at all, or when the record was used.
    """
    path = os.path.join(input_dir, SIDESET_RECORD_FILE)
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"{SIDESET_RECORD_FILE} is unreadable ({exc})"
    if not isinstance(record, dict) \
            or record.get("schema") != SIDESET_RECORD_SCHEMA:
        return None, f"{SIDESET_RECORD_FILE} is not a sideset record"
    stale = ("so sidesets and their conditions were rebuilt from params.json "
             "and the selection files instead of from the record of what was "
             "entered")
    if record.get("boundary_conditions_sha256") != _digest(
            _sideset_condition_lists(data)):
        return None, ("the boundary conditions in params.json differ from "
                      f"the ones the asset wrote, {stale}")
    files = record.get("selection_files", {})
    current_files = set(_selection_file_names(data))
    if not isinstance(files, dict) or set(files) != current_files:
        return None, f"the selection files referenced in params.json changed, {stale}"
    for name in sorted(current_files):
        selection_path = name if os.path.isabs(name) else os.path.join(
            resource_dir or input_dir, name)
        current = _file_digest(selection_path) \
            if os.path.isfile(selection_path) else None
        if files.get(name) != current:
            return None, (f"the selection file {name} differs from the one "
                          f"the asset wrote, {stale}")
    return record, None


def _set_export_report(parent, plans):
    parm = parent.parm("export_report")
    if parm is None:
        return
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = sideset_report(plans) \
        or ["No sideset carries a boundary condition."]
    parm.set("\n".join([f"Written {stamp}."] + lines))


# Asked once per node and folder: user data holding the working directory the
# user agreed to overwrite. Import forgets it, so an imported run is never
# replaced without asking.
_OVERWRITE_APPROVED = "polyfem_overwrite_approved"


def previous_run_files(working_dir):
    """What an earlier run left in <working_dir>/output (empty if nothing).

    The fiber companion is the asset's own export file, not a result.
    """
    output = os.path.join(working_dir, "output")
    if not os.path.isdir(output):
        return []
    return [name for name in sorted(os.listdir(output))
            if not name.startswith(".")
            and name not in (FIBER_FAMILY_FILE, FIBER_FAMILY_FILE + ".tmp")]


def next_free_folder(working_dir):
    """<working_dir>_run2, _run3, ...: the first that does not exist."""
    base = os.path.normpath(os.path.abspath(working_dir))
    match = re.match(r"^(.*)_run(\d+)$", base)
    stem, number = (match.group(1), int(match.group(2)) + 1) if match \
        else (base, 2)
    while os.path.exists(f"{stem}_run{number}"):
        number += 1
    return f"{stem}_run{number}/"


def confirm_output_folder(parent):
    """Ask before replacing an earlier run's results; False means cancel.

    Writing replaces the earlier run's input files and the solver then
    overwrites its results, so a folder holding results (for instance one
    just imported) is reused only after the user says so, once per node and
    folder. "Use a New Folder" switches the node to <folder>_runN. Without an
    interface (hython) there is nobody to ask: it proceeds and says so.
    """
    working_dir = parent.evalParm("working_dir")
    if not working_dir or not os.path.isdir(working_dir):
        return True  # build_params reports the missing folder
    found = previous_run_files(working_dir)
    if not found:
        return True
    folder = os.path.normpath(os.path.abspath(working_dir))
    if parent.userData(_OVERWRITE_APPROVED) == folder:
        return True
    shown = ", ".join(found[:6]) + (", ..." if len(found) > 6 else "")
    if not hou.isUIAvailable():
        print(f"[PolyFEM HDA] {folder}/output already holds an earlier run "
              f"({shown}); writing over it (no interface to ask).")
        return True
    choice = hou.ui.displayMessage(
        "This working directory already holds the results of an earlier "
        f"run:\n{folder}/output\n({shown})\n\nWriting or running here "
        "replaces that run's input files, and the solver then overwrites its "
        "results.",
        buttons=("Overwrite", "Use a New Folder", "Cancel"),
        severity=hou.severityType.Warning, default_choice=1, close_choice=2,
        title="PolyFEM: earlier results found")
    if choice == 0:
        parent.setUserData(_OVERWRITE_APPROVED, folder)
        return True
    if choice == 1:
        new_folder = next_free_folder(working_dir)
        os.makedirs(os.path.join(new_folder, "input"), exist_ok=True)
        os.makedirs(os.path.join(new_folder, "output"), exist_ok=True)
        raw_wd = parent.parm("working_dir").unexpandedString().rstrip("/\\")
        stored = new_folder
        if "$" in raw_wd:
            match = re.match(r"^(.*)_run(\d+)$", raw_wd)
            raw_stem = match.group(1) if match else raw_wd
            suffix = re.search(r"(_run\d+)/?$", new_folder)
            if suffix:
                candidate = raw_stem + suffix.group(1) + "/"
                if os.path.normpath(os.path.abspath(
                        hou.text.expandString(candidate))) == os.path.normpath(
                            os.path.abspath(new_folder)):
                    stored = candidate
        parent.parm("working_dir").set(stored)
        _status(f"Working directory changed to {stored}")
        return True
    return False


def write_params_only(kwargs):
    """Write params.json and every input file without running anything."""
    parent = kwargs["node"]
    path = _prepare_inputs(parent, "write")
    if path:
        _status(f"Wrote {path}")
    return path


def _prepare_inputs(parent, purpose):
    """Everything Write and Run do before the solver starts.

    Check Setup first (problems PolyFEM would only report as a failure),
    then refuse a folder whose run is still going, then confirm replacing an
    earlier run's results, then write the inputs. purpose is "write", "run"
    (terminal) or "background". Returns the params.json path, or None when a
    check refused or the user cancelled.
    """
    cache = {}
    if not gate_setup(parent, purpose, cache=cache):
        return None
    running = active_run(parent)
    if running is not None:
        _message(
            f"A PolyFEM run is still writing into {running['output']} "
            f"({_describe_process(running)}). Wait for it, press Stop, or "
            "choose another Working Directory.")
        return None
    if not confirm_output_folder(parent):
        return None
    try:
        return build_params(parent, cache=cache)
    except hou.NodeError as e:
        _message(str(e))
        return None


