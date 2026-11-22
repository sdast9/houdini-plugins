def _sequence_as_obstacle(item, index, scene, resource_dir, warnings):
    """A mesh_sequence obstacle entry as the obstacle it was made from: the
    mesh the scene record names (its keyframes come back with the
    conditions), else the sequence's first frame, static."""
    if not isinstance(item, dict) or item.get("type") != "mesh_sequence":
        return item
    recorded = ((scene or {}).get("obstacles") or {}).get(str(index), {})
    mesh = recorded.get("mesh")
    if mesh:
        mesh = os.path.expandvars(mesh)
        if not os.path.isabs(mesh):
            mesh = os.path.join(resource_dir, mesh)
    if mesh and os.path.isfile(mesh):
        return {"mesh": mesh, "is_obstacle": True,
                "enabled": item.get("enabled", True)}
    frames = item.get("mesh_sequence")
    first = frames[0] if isinstance(frames, list) and frames else None
    if isinstance(first, str):
        warnings.append(
            f"Geometry {index} is a moving obstacle (a mesh sequence) without "
            f"the asset's record of its keyframes; it was imported as its "
            f"first frame, standing still.")
        return {"mesh": first if os.path.isabs(first)
                else os.path.join(resource_dir, first),
                "is_obstacle": True, "enabled": item.get("enabled", True)}
    return item


def _restore_timeline(parent, scene):
    """Start Frame and Time Scale from the scene record (same Houdini time
    at another frame rate)."""
    line = (scene or {}).get("timeline")
    if not isinstance(line, dict):
        return
    try:
        start = (float(line["start_frame"]) - 1.0) / float(line["fps"])
        parent.setParms({"houdini_start_frame": hou.timeToFrame(start),
                         "time_scale": float(line["time_scale"])})
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass


_GLOBAL_RESET_FOLDERS = frozenset({
    "Time", "Contact", "Solver", "Output", "Global", "Remeshing"})


def _revert_global_parms(parent):
    """Reset simulation settings outside Main/Geometry to asset defaults so
    an imported params.json that omits optional keys does not inherit values
    left over from an earlier scene."""
    ptg = parent.parmTemplateGroup()

    def reset_folder(folder):
        for template in folder.parmTemplates():
            if isinstance(template, hou.FolderParmTemplate):
                if template.folderType() in (
                        hou.folderType.MultiparmBlock,
                        hou.folderType.ScrollingMultiparmBlock,
                        hou.folderType.TabbedMultiparmBlock):
                    parm = parent.parm(template.name())
                    if parm is not None:
                        parm.revertToDefaults()
                else:
                    reset_folder(template)
            elif template.type() not in (
                    hou.parmTemplateType.Label,
                    hou.parmTemplateType.Separator,
                    hou.parmTemplateType.Button):
                tuple_parm = parent.parmTuple(template.name())
                if tuple_parm is not None:
                    for p in tuple_parm:
                        if p.keyframes():
                            p.deleteAllKeyframes()
                    tuple_parm.revertToDefaults()

    def walk(container):
        for template in container.parmTemplates():
            if isinstance(template, hou.FolderParmTemplate):
                if template.label() in _GLOBAL_RESET_FOLDERS:
                    reset_folder(template)
                else:
                    walk(template)

    walk(ptg)


# JSON keys the importer maps to controls or reconstructs by convention.
# Object keys are listed by path so unused nested solver parameters can be
# reported even when their enclosing section has supported controls.
_IMPORT_KEYS = {
    "space": "discr_order remesh",
    "space.discr_order": "id order",
    "space.remesh": REMESH_DEFAULTS.keys(),
    **{f"space.remesh.{key}": value.keys() for key, value in REMESH_DEFAULTS.items()
       if isinstance(value, dict)},
    "time": "t0 tend dt time_steps quasistatic integrator",
    "time.integrator": "type beta gamma",
    "contact": ("enabled dhat epsv friction_coefficient use_convergent_formulation "
                "use_gcp_formulation use_area_weighting use_improved_max_operator "
                "use_physical_barrier alpha_n alpha_t min_distance_ratio "
                "use_adaptive_dhat adhesion"),
    "contact.adhesion": ("adhesion_enabled dhat_p dhat_a adhesion_strength "
                         "tangential_adhesion_coefficient epsa"),
    "solver": "max_threads nonlinear contact augmented_lagrangian advanced linear rayleigh_damping",
    "solver.nonlinear": (
        "solver max_iterations x_delta_tol x_delta grad_norm_tol grad_norm "
        "first_grad_norm_tol rel_grad_norm_tol rel_x_delta_tol newton_decrement_tol "
        "allow_non_grad_convergence iterations_per_strategy allow_out_of_iterations "
        "norm_type Newton DenseNewton L-BFGS L-BFGS-B ADAM StochasticADAM "
        "StochasticGradientDescent advanced line_search"),
    "solver.nonlinear.solver": "type",
    **{f"solver.nonlinear.{method}": (
        "residual_tolerance reg_weight_min reg_weight_max reg_weight_inc "
        "force_psd_projection use_psd_projection use_psd_projection_in_regularized")
       for method in ("Newton", "DenseNewton")},
    **{f"solver.nonlinear.{method}": "history_size"
       for method in ("L-BFGS", "L-BFGS-B")},
    **{f"solver.nonlinear.{method}": "alpha beta_1 beta_2 epsilon erase_component_probability"
       for method in ("ADAM", "StochasticADAM")},
    "solver.nonlinear.StochasticGradientDescent": "erase_component_probability",
    "solver.nonlinear.advanced": ("f_delta_tol f_delta f_delta_step_tol "
                                 "derivative_along_delta_x_tol gradient_fd_eps apply_gradient_fd"),
    "solver.nonlinear.line_search": (
        "method use_grad_norm_tol min_step_size max_step_size_iter min_step_size_final "
        "max_step_size_iter_final default_init_step_size step_ratio "
        "accept_energy_decrease_in_grad_mode Armijo RobustArmijo Wolfe"),
    "solver.nonlinear.line_search.Armijo": "c roundoff_tolerance",
    "solver.nonlinear.line_search.RobustArmijo": "delta_relative_tolerance",
    "solver.nonlinear.line_search.Wolfe": ("c2 growth_factor growth_limit max_evaluations "
                                          "max_objective_restarts approximate_wolfe_epsilon"),
    "solver.contact": ("friction_iterations tangential_adhesion_iterations friction_convergence_tol "
                       "CCD barrier_stiffness semi_implicit initial_barrier_stiffness dhat_epsilon_scale"),
    "solver.contact.CCD": "broad_phase tolerance max_iterations resource_limits",
    "solver.contact.CCD.resource_limits": "max_cell_items max_candidate_emissions",
    "solver.contact.semi_implicit": (
        "trim_lower trim_upper trim_factor initial_trim_estimate kappa_spread kappa_min "
        "refresh_interval conditioning_cap controller_interval trial_displacement_cap "
        "force_continuation continuation_max_ratio coefficient_identity friction_lag "
        "band_statistic clamped_contacts gradient_balance_dofs restart"),
    "solver.contact.semi_implicit.restart": (
        "enabled alpha_threshold patience min_iterations soft_iteration_limit max_restarts stall_trim_factor"),
    "solver.augmented_lagrangian": (
        "initial_weight initial_weight_multiplier scaling max_weight eta error lumping nonlinear budget"),
    "solver.augmented_lagrangian.nonlinear": "max_iterations",
    "solver.augmented_lagrangian.budget": (
        "max_passes stagnation_window progress_tolerance snap_tolerance drift_tolerance"),
    "solver.advanced": ("cache_size lump_mass_matrix lagged_regularization_weight "
                        "lagged_regularization_iterations jacobian_threshold check_inversion "
                        "stall_restart line_search_failure_restarts"),
    "solver.advanced.stall_restart": (
        "enabled alpha_threshold patience min_iterations soft_iteration_limit "
        "max_restarts feasible_ratio_threshold alpha_basis remedy"),
    "solver.linear": "solver precond Pardiso Hypre AMGCL",
    "solver.linear.Pardiso": "mtype",
    "solver.linear.Hypre": "max_iter pre_max_iter tolerance theta nodal_coarsening",
    "solver.linear.AMGCL": "solver precond",
    "solver.linear.AMGCL.solver": "maxiter tol type",
    "solver.linear.AMGCL.precond": "class max_levels direct_coarse ncycle relax coarsening",
    "solver.linear.AMGCL.precond.relax": "type degree power_iters higher lower scale",
    "solver.linear.AMGCL.precond.coarsening": "type estimate_spectral_radius relax aggr",
    "solver.linear.AMGCL.precond.coarsening.aggr": "eps_strong",
    "solver.rayleigh_damping": "form stiffness_ratio lagging_iterations",
    "output": "directory paraview data advanced log json checkpoint restart_json",
    "output.checkpoint": "path",
    "output.paraview": "file_name vismesh_rel_area skip_frame high_order_mesh volume surface wireframe points async options fields",
    "output.paraview.options": (
        "use_hdf5 material body_ids contact_forces friction_forces normal_adhesion_forces "
        "tangential_adhesion_forces velocity acceleration scalar_values tensor_values "
        "discretization_order nodes forces force_high_order jacobian_validity obstacle_volume"),
    "output.data": "solution stress_mat diagnostic_state state mises nodes",
    "output.advanced": (
        "timestep_prefix sol_on_grid compute_error sol_at_node vis_boundary_only curved_mesh_size "
        "save_solve_sequence_debug save_ccd_debug_meshes save_time_sequence save_nl_solve_sequence spectrum"),
    "output.log": "level quiet path",
    "geometry": (
        "type mesh mesh_sequence fps enabled is_obstacle volume_selection "
        "surface_selection point_selection transformation"),
    "geometry.transformation": "translation rotation rotation_mode scale",
    "geometry.volume_selection": "file",
    "geometry.surface_selection": "file id axis position box center radius normal point",
    "geometry.point_selection": "file id",
    "boundary_conditions": (*_BC_LIST_KEYS, "rhs", "obstacle_displacements"),
    **{f"boundary_conditions.{kind}": "id value dimension interpolation"
       for kind in (*_BC_LIST_KEYS, "obstacle_displacements")},
}
_IMPORT_KEYS = {path: frozenset(keys.split() if isinstance(keys, str) else keys)
                for path, keys in _IMPORT_KEYS.items()}

_IMPORT_MATERIAL_KEYS = {
    **{token: "E young nu lambda mu phi psi" for token in _E_NU_MODELS},
    **{token: "E young nu lambda mu phi psi elasticity_tensor fiber_direction"
       for token in _TENSOR_MODELS},
    "GrowthNeoHookean": "E young nu lambda mu g11 g12 g13 g22 g23 g33",
    "InversionBarrier": "E young nu lambda mu JBarrierThreshold",
    "AMIPS": "weight power normalized use_rest_pose",
    "IncompressibleOgden": "c C m k",
    "UnconstrainedOgden": "alphas mus Ds ds",
    "MooneyRivlin": "c1 c2 k",
    "MooneyRivlin3Param": "c1 c2 c3 d1",
    "MooneyRivlin3ParamSymbolic": "c1 c2 c3 d1",
    "MooneyRivlin3ParmSymbolic": "c1 c2 c3 d1",
    "HGODispersion": "fiber_direction k1 k2 kappa k_chi",
    "HGOFiber": "fiber_direction k1 k2",
    "ActiveFiber": "fiber_direction Tmax activation",
    "VolumePenalty": "k",
    "MaterialSum": "models",
}


def _unrepresented_import_paths(data):
    def walk(value, schema_path, path, missing=False):
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, schema_path, f"{path}[{index}]", missing)
            return
        if not isinstance(value, dict):
            return
        keys = _IMPORT_KEYS.get(schema_path, ())
        if schema_path == "materials":
            token = value.get("type")
            model_keys = _IMPORT_MATERIAL_KEYS.get(token, "") if isinstance(token, str) else ""
            keys = {"type", "id", "rho", *model_keys.split()}
        elif schema_path == "materials.fiber_direction":
            keys = {"type", "path", "field"}
        elif schema_path.startswith("solver.linear.Eigen::"):
            keys = {"max_iter", "tolerance"}
        elif schema_path.startswith("boundary_conditions.") and schema_path.endswith(".interpolation"):
            keys = {"type", "points", "values", "extend", "from", "to"}
        for key, item in value.items():
            child_schema = f"{schema_path}.{key}"
            child_path = f"{path}.{key}"
            represented = key in keys or (schema_path == "solver.linear" and key.startswith("Eigen::"))
            if schema_path == "solver.advanced" and key == "stall_restart":
                represented = isinstance(item, dict)
            if missing or not represented:
                yield child_path
            if schema_path == "materials" and key == "models":
                child_schema = "materials"
            yield from walk(item, child_schema, child_path, missing or not represented)

    for section in ("space", "solver", "time", "contact", "output",
                    "materials", "geometry", "boundary_conditions"):
        yield from walk(data.get(section), section, section)


def read_params(kwargs):
    parent = kwargs["node"]
    selected_value = parent.evalParm("old_input_dir").strip()
    if not selected_value:
        _message("Please provide the path to a previous input dir")
        return
    selected_dir = os.path.abspath(os.path.expandvars(selected_value))
    # Accept a JSON file directly, the simulation root, or its input directory.
    if os.path.isfile(selected_dir):
        params_path = selected_dir
        input_dir = os.path.dirname(selected_dir)
    elif os.path.isfile(os.path.join(selected_dir, "input", "params.json")):
        input_dir = os.path.join(selected_dir, "input")
        params_path = os.path.join(input_dir, "params.json")
    else:
        input_dir = selected_dir
        params_path = os.path.join(input_dir, "params.json")
    try:
        with open(params_path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        _message(f"Error reading JSON file: {e}")
        return
    if not isinstance(data, dict):
        _message("The root of params.json must be a JSON object.")
        return
    # Native CLI dispatches every object containing states to OptState.
    # Refuse before clearing geometry or changing any imported scene controls.
    if "states" in data:
        _message("Optimization configurations cannot be imported as forward "
                 "simulation scenes. MMG shape-optimization remeshing must be "
                 "configured through PolyFEM's optimization workflow.")
        return
    input_block = data.get("input")
    input_block = input_block if isinstance(input_block, dict) else {}
    checkpoint_block = input_block.get("checkpoint")
    checkpoint_block = (checkpoint_block
                        if isinstance(checkpoint_block, dict) else {})
    checkpoint = checkpoint_block.get("path", "")
    if isinstance(checkpoint, str) and checkpoint.strip():
        checkpoint = os.path.expandvars(os.path.expanduser(checkpoint))
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(input_dir, checkpoint)
        norm_input = os.path.normpath(input_dir)
        working_dir = (os.path.dirname(norm_input)
                       if os.path.basename(norm_input) == "input"
                       else norm_input)
        with hou.undos.group("Import PolyFEM JSON"):
            parent.setParms({"resume_checkpoint": os.path.abspath(checkpoint),
                             "resume_reorder": int(checkpoint_block.get("reorder") is True),
                             "working_dir": working_dir + "/"})
            if parent.userData(_OVERWRITE_APPROVED) is not None:
                parent.destroyUserData(_OVERWRITE_APPROVED)
            report = parent.parm("export_report")
            if report is not None:
                report.set(CHECKPOINT_RESUME_NOTE)
        _message(CHECKPOINT_RESUME_NOTE)
        return
    root_path = data.get("root_path")
    resource_dir = (os.path.abspath(os.path.join(input_dir, os.path.expandvars(root_path)))
                    if isinstance(root_path, str) and root_path.strip()
                    else input_dir)
    raw_geometry = data.get("geometry")
    if not isinstance(raw_geometry, list) or not raw_geometry:
        _message("The geometry section must be a non-empty list of mesh entries.")
        return
    warnings = []
    scene = load_scene_record(input_dir, data, warnings)
    raw_geometry = [
        _sequence_as_obstacle(item, index, scene, resource_dir, warnings)
        for index, item in enumerate(raw_geometry, 1)]
    retained = [(index, item) for index, item in enumerate(raw_geometry, 1)
                if isinstance(item, dict) and "mesh" in item]
    geometry = [item for _, item in retained]
    if scene is not None:
        obstacles = scene.get("obstacles")
        if isinstance(obstacles, dict):
            remapped = {}
            for geo, (index, _) in enumerate(retained, 1):
                obstacle = obstacles.get(str(index))
                if not isinstance(obstacle, dict):
                    continue
                transforms = obstacle.get("transform", {})
                remapped[str(geo)] = dict(obstacle, transform={
                    re.sub(rf"(?<!\d){index}([xyz]?)$", rf"{geo}\g<1>", name): keys
                    for name, keys in transforms.items()}) if isinstance(transforms, dict) else obstacle
            scene = dict(scene, obstacles=remapped)
    ignored = len(raw_geometry) - len(geometry)
    if ignored:
        warnings.append(f"Skipped {ignored} procedural/non-mesh geometry entries.")
    mesh_errors = [_import_mesh_error(item, resource_dir) for item in geometry]
    if not geometry or all(mesh_errors):
        _message("The geometry section contains no usable mesh files.\n" +
                 "\n".join(error for error in mesh_errors if error))
        return

    if hou.isUIAvailable() and _has_loaded_geometry(parent):
        choice = hou.ui.displayMessage(
            "Importing a previous run replaces the current geometries, "
            "materials, sidesets and solver settings on this node.",
            buttons=("Import", "Cancel"),
            default_choice=0, close_choice=1,
            severity=hou.severityType.Warning, title="Import params.json")
        if choice != 0:
            return
    with hou.undos.group("Import PolyFEM JSON"):
        _import_scene(kwargs, parent, data, geometry, scene, params_path,
                      input_dir, resource_dir, warnings, mesh_errors)


def _import_mesh_error(item, resource_dir):
    mesh = item.get("mesh")
    if not isinstance(mesh, str) or not mesh.strip():
        return "mesh path is not a file string."
    path = os.path.abspath(os.path.join(resource_dir, os.path.expandvars(mesh)))
    if not os.path.isfile(path):
        return f"mesh file not found: {path}"
    extension = os.path.splitext(path)[1].lower()
    try:
        if extension == ".msh":
            reader = hou.nodeType(hou.sopNodeTypeCategory(), "MSH_Reader::3.0")
            if reader is None:
                return "MSH Reader 3.0 is unavailable."
            cells = reader.hdaModule().read_msh(path)["cells"]
            has_elements = bool(cells) if item.get("is_obstacle") else any(
                family in cells and len(cells[family]["corners"])
                for family in ("tet", "hex"))
            return None if has_elements else f"mesh has no usable elements: {path}"
        if extension in (".obj", ".stl", ".ply"):
            geometry = hou.Geometry()
            geometry.loadFromFile(path)
            return None if geometry.intrinsicValue("primitivecount") else f"empty mesh: {path}"
    except Exception as exc:  # preserve later geometries even on parser failure
        return f"could not read mesh {path} ({_error_text(exc)})"
    return f"unsupported mesh format: {path}"


def _import_rotation(transform):
    rotation = transform.get("rotation")
    if rotation is None or rotation == []:
        return [0, 0, 0]
    mode = transform.get("rotation_mode", "xyz")
    if not isinstance(mode, str):
        raise ValueError(f"unsupported rotation_mode {mode!r}")
    mode = mode.lower()
    if mode == "axis_angle":
        if not isinstance(rotation, list) or len(rotation) != 4 or not all(
                type(v) in (int, float) and math.isfinite(v) for v in rotation):
            raise ValueError("axis_angle needs [angle_deg, ax, ay, az]")
        angle, *axis = rotation
        if not math.hypot(*axis) > 0:
            raise ValueError("axis_angle needs a nonzero rotation axis")
        return tuple(hou.hmath.buildRotateAboutAxis(axis, angle).extractRotates())
    if mode == "xyz" and isinstance(rotation, list) and len(rotation) == 3 and all(
            type(v) in (int, float) and math.isfinite(v) for v in rotation):
        return rotation
    if mode == "xyz":
        raise ValueError("xyz rotation needs three finite Euler angles")
    raise ValueError(f"unsupported rotation_mode {mode!r}")


def _import_scene(kwargs, parent, data, geometry, scene, params_path,
                  input_dir, resource_dir, warnings, mesh_errors):
    legacy = _looks_legacy(data)
    if legacy:
        warnings.append(
            "Older PolyFEM/HDA id schema detected. Compatible values were "
            "mapped to the current controls.")

    clear_all_geos(kwargs, _confirmed=True)
    _revert_global_parms(parent)
    if parent.parm("resume_checkpoint") is not None:
        parent.setParms({"resume_checkpoint": "", "resume_reorder": 0})
    norm_input = os.path.normpath(input_dir)
    working_dir = (os.path.dirname(norm_input)
                   if os.path.basename(norm_input) == "input"
                   else norm_input)
    parent.setParms({"working_dir": working_dir + "/"})
    if parent.userData(_OVERWRITE_APPROVED) is not None:
        parent.destroyUserData(_OVERWRITE_APPROVED)

    parent.setParms({"num_geos": len(geometry), "geo_int": len(geometry)})

    valid_geometries = []
    # Reserve the entire incoming set before any basename can be copied.
    mesh_paths = [
        os.path.abspath(os.path.join(resource_dir, os.path.expandvars(item["mesh"])))
        if isinstance(item.get("mesh"), str) and item["mesh"].strip() else ""
        for item in geometry]
    destination = os.path.abspath(os.path.join(working_dir, "input"))
    taken = {path: path for path in mesh_paths
             if os.path.dirname(path) == destination}
    real_destination = os.path.realpath(destination)
    for path in filter(None, mesh_paths):
        resolved = os.path.realpath(path)
        if os.path.dirname(resolved) == real_destination:
            taken[os.path.join(destination, os.path.basename(resolved))] = path
    for i, (g, mesh, error) in enumerate(zip(geometry, mesh_paths, mesh_errors), 1):
        if error:
            warnings.append(f"Geometry {i}: {error}")
            continue
        try:
            staged = _stage_mesh(i, mesh, destination, taken)
            parent.setParms({f"file_location{i}": staged,
                             f"is_enabled{i}": bool(g.get("enabled", True)),
                             f"is_obstacle{i}": bool(g.get("is_obstacle", False))})
            create_geo_nodes({"node": parent, "script_multiparm_index": str(i),
                              "parm_name": f"file_location{i}"})
        except (hou.Error, OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"Geometry {i}: could not load {mesh} ({_error_text(exc)})")
            continue
        if parent.node(f"geo_{i}") is None:
            warnings.append(
                f"Geometry {i}: Houdini could not construct the mesh node tree.")
            continue
        valid_geometries.append(i)
        tr = g.get("transformation", {})
        if not isinstance(tr, dict):
            tr = {}

        def vector3(key, default):
            value = tr.get(key, [default] * 3)
            if type(value) in (int, float):
                value = [value] * 3
            if isinstance(value, list) and len(value) == 3 and all(
                    type(v) in (int, float) and math.isfinite(v) for v in value):
                return value
            warnings.append(f"Geometry {i}: invalid transformation.{key}; kept the default.")
            return [default] * 3

        t = vector3("translation", 0)
        s = vector3("scale", 1)
        try:
            r = _import_rotation(tr)
        except (hou.Error, TypeError, ValueError, OverflowError) as exc:
            warnings.append(f"Geometry {i}: {_error_text(exc)}; kept zero rotation.")
            r = [0, 0, 0]
        parent.setParms({
            f"xform_t__{i}x": t[0], f"xform_t__{i}y": t[1], f"xform_t__{i}z": t[2],
            f"xform_r__{i}x": r[0], f"xform_r__{i}y": r[1], f"xform_r__{i}z": r[2],
            f"xform_s__{i}x": s[0], f"xform_s__{i}y": s[1], f"xform_s__{i}z": s[2],
            # params.json stores the already pivot-compensated affine TRS.
            # Keeping create_geo_nodes' centroid pivot would compensate it a
            # second time and make every rotated/scaled round trip drift.
            f"tpivot_{i}x": 0, f"tpivot_{i}y": 0, f"tpivot_{i}z": 0,
            f"rpivot_{i}x": 0, f"rpivot_{i}y": 0, f"rpivot_{i}z": 0})

    # Restore edited Entity assignments before materials, orders, sidesets, or
    # conditions are populated; all of those multiparms depend on volume count.
    material_slots = {}
    for i, g in enumerate(geometry, 1):
        if i in valid_geometries and not bool(g.get("is_obstacle", False)):
            try:
                slots = _restore_volume_assignments(parent, i, g, resource_dir,
                                                    legacy, warnings)
                for body, slot in slots.items():
                    material_slots.setdefault(body, []).append(slot)
            except (hou.Error, OSError, TypeError, ValueError, RuntimeError) as exc:
                warnings.append(f"Geometry {i}: volume assignments were not "
                                f"restored ({_error_text(exc)}).")

    # Subdomain numbers that changed on the way in (compacted gaps):
    # {geo: {number in the file's ids: subdomain here}}.
    volume_remap = {}
    for body, slots in material_slots.items():
        for geo, vol in slots:
            if body // 1000 == geo and body % 1000 != vol:
                volume_remap.setdefault(geo, {})[body % 1000] = vol

    # Also recognize both HDA id encodings when volume_selection was absent.
    for geo in valid_geometries:
        if parent.evalParm(f"is_obstacle{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            material_slots.setdefault(1000 * geo + vol, [(geo, vol)])
            material_slots.setdefault(int(f"{geo}{vol}"), [(geo, vol)])

    # Materials: use the volume-selection lookup first. This preserves the
    # association even for arbitrary material ids and edited multi-material
    # meshes, instead of assuming the ids still match the MSH physical groups.
    type_menu = {name: index for index, name in enumerate(MATERIAL_TOKENS)}
    type_menu.setdefault("MooneyRivlin3ParmSymbolic",
                         type_menu["MooneyRivlin3ParamSymbolic"])
    params_dir = resource_dir
    materials = data.get("materials", [])
    if isinstance(materials, dict):
        materials = [materials]
    elif not isinstance(materials, list):
        materials = []
        warnings.append("The materials section was not a list; defaults remain.")
    all_slots = sorted({slot for slots in material_slots.values() for slot in slots})
    for material_index, m in enumerate(materials):
        if not isinstance(m, dict):
            warnings.append(
                f"Material entry {material_index + 1} was not an object.")
            continue
        mid = m.get("id")
        slots = _material_slots(mid, material_slots, legacy)
        if not slots and material_index < len(all_slots) and (
                mid is None or len(materials) == len(all_slots)):
            geo, vol = all_slots[material_index]
            slots = [(geo, vol)]
            warnings.append(
                f"Material entry {material_index + 1} (id {mid}) was "
                f"associated positionally with geo {geo} subdomain {vol}.")
        if not slots:
            warnings.append(f"Material id {mid} could not be mapped; skipped.")
            continue
        token = m.get("type")
        if not isinstance(token, str) or token not in type_menu:
            warnings.append(f"Unknown material type '{token}' on id {mid}.")
            continue
        for geo, vol in slots:
            if parent.parm(f"materials{geo}_{vol}") is None:
                warnings.append(
                    f"Material id {mid} has no matching geometry/subdomain "
                    f"(geo {geo}, vol {vol}); skipped.")
                continue
            try:
                parent.setParms({f"materials{geo}_{vol}": type_menu[token]})
                if "rho" in m:
                    parent.parm(f"rho{geo}_{vol}").set(m["rho"])
                if token == "MaterialSum":
                    _import_composite(parent, geo, vol, m.get("models", []),
                                      params_dir, warnings)
                elif token in FIBER_MODELS:
                    _import_fiber_model(parent, geo, vol, None, m, params_dir, warnings)
                else:
                    _import_isotropic(parent, geo, vol, m, warnings)
            except (hou.Error, OSError, TypeError, ValueError, RuntimeError) as exc:
                warnings.append(f"Material id {mid} ({_slot_label(geo, vol, None)}): "
                                f"{_error_text(exc)}")
    _restore_material_ui(parent, scene, materials, params_dir, warnings)

    for section, restore, args in (
            ("time", _restore_time, (data, warnings)),
            ("timeline", _restore_timeline, (scene,)),
            ("contact", _restore_contact, (data, warnings)),
            ("space", _restore_space, (data, material_slots, legacy, warnings)),
            ("solver", _restore_solver, (data, legacy, warnings)),
            ("output", _restore_output, (data, warnings)),
            ("units", _restore_units, (data,))):
        try:
            if section in ("time", "contact", "solver", "output") and \
                    section in data and not isinstance(data[section], dict):
                warnings.append(f"Section '{section}' is not a JSON object; kept defaults.")
                continue
            restore(parent, *args)
        except (hou.Error, KeyError, TypeError, ValueError, OverflowError) as exc:
            warnings.append(f"Section '{section}' was partially restored "
                            f"({_error_text(exc)}).")

    # Current collision-free ids round-trip directly. Older HDA decimal ids and
    # generic PolyFEM selection ids are rebuilt through a compatibility mapper.
    _restore_body_load(parent, data, warnings)
    if not legacy:
        try:
            _restore_conditions(parent, data, volume_remap, scene, warnings, geometry)
        except Exception as e:
            warnings.append(f"Conditions were only partially restored: {e}")
        try:
            warnings.extend(_restore_sidesets(
                parent, data, geometry, input_dir, resource_dir,
                volume_remap))
        except Exception as e:
            warnings.append(f"Sidesets were only partially restored: {e}")
    else:
        try:
            warnings.extend(_restore_legacy_scene(
                parent, data, geometry, resource_dir, material_slots, scene))
        except Exception as exc:
            warnings.append(
                f"Older selections/conditions were only partially restored: "
                f"{exc}")
    for i, g in enumerate(geometry, 1):
        if i not in valid_geometries or parent.evalParm(f"is_obstacle{i}"):
            continue
        try:
            for vol in range(1, parent.evalParm(f"num_volumes{i}") + 1):
                if parent.parm(f"sideset_selection{i}_{vol}") is not None:
                    # imported sidesets keep exactly the conditions they had
                    _rebuild_group_chain(parent, str(i), str(vol),
                                         default_condition=False)
        except Exception as e:
            warnings.append(f"Group nodes for geometry {i}: {e}")

    for section in ("time", "contact", "space", "solver", "output"):
        if section not in data:
            warnings.append(
                f"Section '{section}' is absent; current HDA defaults were "
                "retained for it.")
    represented_top_level = {
        "geometry", "materials", "time", "contact", "space", "solver",
        "output", "units", "boundary_conditions", "initial_conditions",
        "root_path", "provenance"}  # provenance is regenerated at export
    unrepresented = sorted(set(data) - represented_top_level)
    if unrepresented:
        warnings.append(
            "Top-level JSON section(s) retained only in the source file (no "
            "equivalent HDA controls): " + ", ".join(unrepresented))

    for path in _unrepresented_import_paths(data):
        warnings.append(f"{path} has no equivalent HDA control and remains in the source JSON.")

    for geo in valid_geometries:
        if not parent.evalParm(f"is_obstacle{geo}"):
            try:
                update_fiber_data(parent, geo)
            except hou.Error as exc:
                warnings.append(
                    f"Geometry {geo}: material preview refresh failed ({exc}).")

    summary = [
        f"Imported {len(valid_geometries)} geometry entr"
        f"{'y' if len(valid_geometries) == 1 else 'ies'} and "
        f"{len(materials)} material entr"
        f"{'y' if len(materials) == 1 else 'ies'} from {params_path}."
    ]
    if warnings:
        summary.append(
            f"{len(warnings)} item{'s' if len(warnings) != 1 else ''} need "
            "attention:")
        summary.extend(f"- {item}" for item in warnings)
    else:
        summary.append("All represented scene and PolyFEM parameters restored.")
    # The node now writes into the imported run's folder; Write and Run ask
    # before replacing its results (confirm_output_folder).
    if previous_run_files(parent.evalParm("working_dir")):
        summary.append(
            f"Working Directory is the imported run's folder "
            f"({parent.evalParm('working_dir')}); Write and Run ask before "
            "replacing its results.")
    report_text = "\n".join(summary)
    report_parm = parent.parm("import_report")
    if report_parm is not None:
        report_parm.set(report_text)
    if warnings:
        _message(
            "Import completed with compatibility notes:\n"
            + "\n".join(f"• {item}" for item in warnings[:20])
            + ("\n• See Import Report for the complete list."
               if len(warnings) > 20 else ""))
    _status(
        f"Imported params.json: {len(valid_geometries)} geometries, "
        f"{len(materials)} materials, {len(warnings)} compatibility notes")


def geo_set(kwargs, val):
    kwargs["node"].setParms({"geo_int": val})
