def check_setup(parent, purpose="check", cache=None, *, raise_shared=False):
    """Find setup problems before PolyFEM does.

    purpose: "check" (the button: everything), "write" (no binary needed) or
    "run"/"background". Returns {"errors", "warnings", "notes"}: lists of
    sentences. Nothing is written. When ``cache`` is passed, resolved
    sideset plans, per-element material arrays, and obstacle surfaces are
    stored in it so a subsequent ``build_params(..., cache=cache)`` reuses
    them without re-parsing meshes or re-evaluating sidesets.
    """
    errors, warnings, notes = [], [], []
    working_dir = parent.evalParm("working_dir")
    resource_dir = os.path.abspath(os.path.join(working_dir, "input"))
    if not working_dir.strip():
        errors.append("Working Directory is empty: choose the folder this "
                      "node writes the simulation's input and output to.")
    elif not os.path.isdir(working_dir):
        notes.append(f"Working Directory {working_dir} does not exist yet; "
                     "it is created when the input is written.")
    if purpose != "write":
        binary = _resolved_binary(parent.evalParm("polyfem_bin").strip())
        if not binary:
            errors.append("PolyFEM Binary is empty: point it at the PolyFEM "
                          "executable (PolyFEM_bin).")
        elif not os.path.isfile(binary):
            errors.append(f"PolyFEM Binary not found: {binary}")
        elif platform.system() != "Windows" and not os.access(binary,
                                                              os.X_OK):
            errors.append(f"PolyFEM Binary is not executable: {binary}")

    checkpoint = _resume_checkpoint(parent)
    if checkpoint:
        if not os.path.isfile(checkpoint):
            errors.append(f"Resume Checkpoint not found: {checkpoint}")
        notes.append(CHECKPOINT_RESUME_NOTE)
        return {"errors": errors, "warnings": warnings, "notes": notes}

    if parent.evalParm("remeshing_enabled"):
        errors.append(LEGACY_REMESH_ERROR)
    if parent.parm("rest_mesh_file") is not None \
            and parent.evalParm("rest_mesh_file"):
        warnings.append(
            "Rest Mesh (.msh file) has been retired in PolyFEM "
            "(output.data.rest_mesh was removed) and is no longer exported; "
            "turn it off to clear this warning.")

    quasistatic = bool(parent.evalParm("quasistatic"))
    contact = bool(parent.evalParm("enable"))
    num_geos = parent.evalParm("num_geos")
    simulated, bodies, obstacles = [], [], []
    if num_geos < 1:
        errors.append("There is no geometry: load a mesh on the Geometry "
                      "tab.")
    for geo in range(1, num_geos + 1):
        path = parent.evalParm(f"file_location{geo}")
        obstacle = bool(parent.evalParm(f"is_obstacle{geo}"))
        source = parent.parm(f"mesh_source{geo}")
        if not obstacle and source is not None and source.evalAsString() == "sop":
            sop_path = parent.evalParm(f"sop_path{geo}")
            sop = (parent.node(sop_path) or hou.node(sop_path)) if sop_path else None
            if isinstance(sop, hou.SopNode):
                try:
                    notes.extend(_sop_subdomain_notes(sop.geometry(), geo))
                except hou.Error:
                    pass  # existing mesh/source checks report cook failures
        if not parent.evalParm(f"is_enabled{geo}"):
            _disabled_geometry_findings(
                parent, geo, obstacle, path, errors, cache=cache)
            continue
        if not path.strip():
            errors.append(f"Geometry {geo} has no mesh file.")
            continue
        if not os.path.isfile(path):
            errors.append(f"Geometry {geo}: mesh file not found: {path}")
        if parent.node(f"geo_{geo}") is None:
            errors.append(f"Geometry {geo}: the mesh is not loaded; pick its "
                          "file again.")
            continue
        source = parent.node(f"geo_{geo}").geometry()
        duplicates = _msh_duplicate_elements(source)
        if duplicates:
            errors.append(
                f"Geometry {geo}: its MSH 2.2 file writes {duplicates} "
                "elements more than once (once per physical group they are "
                "in). Houdini shows each once, but PolyFEM reads every copy "
                "and stops ('Elements 0 and 1 are the same element'). Save "
                "the mesh from Gmsh as MSH "
                "4.1 (each element once, in its first group), or put each "
                "element in a single physical group.")
        if obstacle:
            obstacles.append(geo)
            continue
        if source.countPrimType(hou.primType.Tetrahedron) \
                + source.countPrimType(hou.primType.Hexahedron) == 0:
            errors.append(
                f"Geometry {geo} has no tetrahedra or hexahedra (a 2D or "
                "surface mesh). This node writes 3D simulations: mesh the "
                "volume (tetrahedra or hexahedra), or load it as an "
                "obstacle.")
            continue
        simulated.append(geo)
        if transform_is_animated(parent, geo):
            errors.append(ANIMATED_BODY_ERROR.format(geo=geo))
        determinant = _transform_determinant(parent, geo)
        if determinant is not None and determinant < 0:
            errors.append(
                f"Geometry {geo}: its transform mirrors the mesh (a negative "
                "scale), which turns every element inside out; PolyFEM would "
                "stop with 'element 0 is flipped'. Use positive scales, and "
                "mirror the mesh file itself if you need a mirrored part.")
        elif determinant is not None and abs(determinant) < 1e-12:
            errors.append(f"Geometry {geo}: its transform has a zero scale, "
                          "which flattens the mesh.")
        elif _transform_has_shear(parent, geo):
            errors.append(
                f"Geometry {geo}: its transform introduces shear (for example "
                "non-uniform scale combined with a non-SRT transform order or "
                "rotated pivot). PolyFEM applies Scale, then Rotate, then "
                "Translate without shear, so the simulated shape would differ "
                "from the viewport. Use Transform Order = Scale Rot Trans "
                "(SRT) with zero Pivot Rotate, or bake the transform into the "
                "mesh file.")
        num_vols = parent.evalParm(f"num_volumes{geo}")
        branch = parent.node(f"branch_{geo}")
        if branch is not None:
            try:
                solver_element_order(branch.geometry(),
                                     expect_msh=path.lower().endswith(".msh"))
            except hou.Error as exc:
                errors.append(f"Geometry {geo}: {_error_text(exc)}")
        fibers = False
        for vol in range(1, num_vols + 1):
            _material_findings(parent, geo, vol, quasistatic, errors)
            fibers = fibers or material_token(parent, geo, vol) in \
                FIBER_MODELS + ("MaterialSum",)
        if fibers:
            try:
                _cached(cache, "pe_data", int(geo),
                        lambda g=geo: per_element_data(parent, g))
            except hou.Error as exc:
                errors.append(f"Geometry {geo}: {_error_text(exc)}")
        try:
            _hex_order_findings(parent, geo, errors)
        except hou.Error:
            pass
        try:
            plan = _cached(
                cache, "plans", int(geo),
                lambda g=geo, nv=num_vols: resolve_sidesets(parent, g, nv))
        except hou.Error as exc:
            errors.append(f"Geometry {geo} sidesets: {_error_text(exc)}")
            plan = None
        if plan is None:
            continue
        for warning in plan["warnings"]:
            warnings.append("Some picked faces lie on another subdomain and "
                            "are left out:" + warning)
        for record in plan["records"]:
            if not record["conditions"] and record["pattern"].strip():
                notes.append(
                    f"Geometry {geo} subdomain {record['subdomain']} sideset "
                    f"{record['sideset']} has a selection but no boundary "
                    "condition: PolyFEM applies nothing there (Read PVD's "
                    "Force Curves can still measure the force on it).")
        fixed = [False, False, False]
        for entries in plan["conditions"].values():
            for key, entry in entries:
                val = entry.get("value")
                items = val if isinstance(val, list) else [val]
                for axis_i, comp in enumerate(items):
                    err = _validate_tinyexpr(
                        comp,
                        f"Geometry {geo} {key} (axis {'xyz'[axis_i]})"
                        if isinstance(val, list) and axis_i < 3
                        else f"Geometry {geo} {key}", resource_dir)
                    if err:
                        errors.append(err)
                if key == "dirichlet_boundary":
                    fixed = [a or bool(b) for a, b in
                             zip(fixed, entry.get("dimension", [1, 1, 1]))]
        for vol in range(1, num_vols + 1):
            for l in range(1, parent.evalParm(
                    f"initial_conditions{geo}_{vol}") + 1):
                try:
                    ic_vec = _read_expr_vector(
                        parent, f"ic_vec{geo}_{vol}_{l}",
                        f"condition_vector_{geo}_{vol}_{l}",
                        f"initial condition {geo}/{vol}/{l}")
                except hou.Error as exc:
                    errors.append(f"Geometry {geo} subdomain {vol} initial "
                                  f"condition {l}: {_error_text(exc)}")
                    continue
                for axis_i, comp in enumerate(
                        ic_vec if isinstance(ic_vec, list) else [ic_vec]):
                    err = _validate_tinyexpr(
                        comp, f"Geometry {geo} subdomain {vol} initial "
                        f"condition {l} ({'xyz'[axis_i]})", resource_dir)
                    if err:
                        errors.append(err)
        bodies.append((geo, fixed))

    for geo in obstacles:
        try:
            obs_vec = _read_expr_vector(
                parent, f"obstacle_disp_vec{geo}", f"obstacle_disp{geo}",
                f"obstacle {geo}")
        except hou.Error as exc:
            errors.append(f"Obstacle {geo} displacement: {_error_text(exc)}")
            continue
        for axis_i, comp in enumerate(
                obs_vec if isinstance(obs_vec, list) else [obs_vec]):
            err = _validate_tinyexpr(
                comp, f"Obstacle {geo} displacement ({'xyz'[axis_i]})", resource_dir)
            if err:
                errors.append(err)
    try:
        rhs_vec = _read_expr_vector(parent, "rhs_vec", "RHS", "rhs")
        for axis_i, comp in enumerate(
                rhs_vec if isinstance(rhs_vec, list) else [rhs_vec]):
            err = _validate_tinyexpr(
                comp, f"Global Body Load ({'xyz'[axis_i]})", resource_dir)
            if err:
                errors.append(err)
    except hou.Error as exc:
        errors.append(f"Global Body Load: {_error_text(exc)}")

    if quasistatic:
        for geo, fixed in bodies:
            if all(fixed):
                continue
            free = "".join(axis for axis, flag in zip("xyz", fixed)
                           if not flag)
            held = ("" if not contact else
                    " unless contact with a fixed body or obstacle holds it "
                    "in place")
            warnings.append(
                f"Geometry {geo} is not held in {free}: no Dirichlet "
                f"condition fixes {'it' if free == 'xyz' else free}. A "
                "quasistatic run looks for an equilibrium, and a body free "
                "to move has none, so PolyFEM usually stops with 'Reached "
                f"iteration limit'{held}. Fix part of its surface "
                "(Dirichlet [0, 0, 0]), or make the run dynamic (turn off "
                "Quasistatic on the Time tab).")

    if contact and simulated:
        dhat = parent.evalParm("dhat")
        diagonal = _model_diagonal(parent, simulated + obstacles)
        if diagonal and dhat > 1e-2 * diagonal:
            warnings.append(
                f"The contact distance dhat ({dhat:g}) is {100 * dhat / diagonal:.1f}% "
                f"of the model's size (bounding-box diagonal {diagonal:.4g}): "
                "contact forces start that far from a surface and stiffen "
                f"the response. A typical value is 1e-3 x diagonal = "
                f"{1e-3 * diagonal:.3g} (Contact tab).")
        edge = _min_boundary_edge(parent, simulated)
        if edge and dhat > edge:
            notes.append(
                f"dhat ({dhat:g}) is larger than the shortest boundary edge "
                f"({edge:.4g}); PolyFEM warns about this too. Contact may "
                "act between neighbouring surface nodes.")
        method = _menu_token(parent, "solver_nl")
        if method != "Newton":
            warnings.append(NON_NEWTON_CONTACT_WARNING.format(method=method))
        if parent.evalParm("barrier_mode") == 0 \
                and parent.evalParm("gcp_enable"):
            errors.append(
                "Geometric Contact (GCP) is not supported with the "
                "Semi-Implicit barrier mode: PolyFEM refuses this combination "
                "at start-up. Switch Barrier Stiffness Mode to Adaptive or "
                "Fixed, or turn off Geometric Contact (GCP) (Contact tab).")
        if parent.evalParm("barrier_mode") == 0 \
                and not parent.evalParm("gcp_enable"):
            if parent.evalParm("si_trim_lower") >= parent.evalParm("si_trim_upper"):
                errors.append(
                    "Semi-Implicit Trim Lower Band must be smaller than Trim "
                    "Upper Band (Contact > Barrier (IPC) > Semi-Implicit "
                    "Options).")
            if parent.evalParm("improved_max"):
                if _menu_token(parent, "si_coefficient_identity") == "stencil":
                    errors.append(IMPROVED_MAX_STENCIL_ERROR)
                else:
                    notes.append(
                        "Remove Duplicate Contacts (Improved Max) is on: "
                        "experimental, verified in 2D frictionless tests only "
                        "(3D and friction not yet validated). It needs a "
                        "PolyFEM built from sdast9/polyfem a54509955 or later; "
                        "an older binary refuses it at start-up.")
        if _menu_token(parent, "resource_limits") == "custom":
            bp_idx = int(parent.evalParm("broad_phase"))
            if bp_idx not in (0, 1):
                bp_name = ("Hash Grid", "Brute Force", "Spatial Hash", "BVH",
                           "Sweep and Prune", "Sweep and Tiniest Queue")[bp_idx]
                warnings.append(
                    f"Custom Contact Resource Limits are skipped by the "
                    f"{bp_name} broad phase (ceilings are only enforced by "
                    "Hash Grid and Brute Force). Switch Broad Phase to Hash "
                    "Grid / Brute Force or leave Resource Limits on Automatic "
                    "(Contact > CCD Parameters).")
    if _menu_token(parent, "method") == "Wolfe":
        c1 = parent.evalParm("armijo_c")
        c2 = parent.evalParm("wolfe_c2")
        if not (c1 < c2 < 1.0):
            errors.append(
                f"Strong Wolfe curvature c2 ({c2:g}) must lie strictly "
                f"between Armijo c ({c1:g}) and 1 (Solver > Line Search).")
    shared_error = None
    try:
        _check_composite_consistency(parent)
        _plan_body_materials(parent)
    except _SharedMaterialError as exc:
        shared_error = exc
        errors.append(_error_text(exc))
    except (hou.Error, TypeError, ValueError) as exc:
        message = _error_text(exc)
        if not any(message in error for error in errors):
            errors.append(message)

    t0, tend = parent.evalParm("t0"), parent.evalParm("tend")
    tend_on, dt_on, steps_on = _time_flags(parent)
    if t0 < 0:
        errors.append(f"Start Time ({t0:g}) cannot be negative (Time tab).")
    if tend_on and not tend > t0:
        errors.append(f"End Time ({tend:g}) must be after Start Time "
                      f"({t0:g}) (Time tab).")
    if dt_on and not parent.evalParm("dt") > 0:
        errors.append("The time step dt must be greater than 0 (Time tab).")
    if steps_on and not parent.evalParm("num_timesteps") >= 1:
        errors.append("The number of time steps must be at least 1 "
                      "(Time tab).")
    try:
        _guard_forward_json(_parse_extra_params_json(parent))
    except hou.Error as exc:
        errors.append(_error_text(exc))

    _timeline_findings(parent, obstacles, errors, notes, cache=cache)

    notes.extend(_expert_notes(parent))
    hdf5 = _hdf5_note(parent)
    if hdf5:
        notes.append(hdf5)
    report = {"errors": errors, "warnings": warnings, "notes": notes}
    if shared_error is not None and raise_shared:
        _set_check_report(parent, report)
        raise shared_error
    return report


def _disabled_geometry_findings(parent, geo, obstacle, path, errors,
                                cache=None):
    """Check Setup for a geometry with Enabled off.

    PolyFEM skips its mesh, but Write still exports its settings (so they
    are there again when it is enabled, and Import restores them). What
    Write cannot export stops it, and used to surface only as "Error
    generated by Python node." after Check Setup had passed. Only those
    problems are reported here: the run checks (materials, constraints,
    contact) do not apply to a geometry PolyFEM skips.
    """
    lead = (f"Geometry {geo} is disabled, but Write still exports its "
            "settings (they are kept for when it is enabled again)")
    if not path.strip():
        errors.append(f"{lead}, and it has no mesh file: load one, or "
                      "remove the geometry.")
        return
    if parent.node(f"geo_{geo}") is None:
        errors.append(f"{lead}, and its mesh is not loaded: pick its file "
                      "again, or remove the geometry.")
        return
    if obstacle:
        return
    if transform_is_animated(parent, geo):
        errors.append(f"{lead}: " + ANIMATED_BODY_ERROR.format(geo=geo))
    try:
        num_vols = parent.evalParm(f"num_volumes{geo}")
        _cached(cache, "plans", int(geo),
                lambda g=geo, nv=num_vols: resolve_sidesets(parent, g, nv))
    except hou.Error as exc:
        text = _error_text(exc).rstrip(".")
        errors.append(f"{lead}, and a sideset cannot be exported: {text}. "
                      "Fix that sideset or remove it, even while the "
                      "geometry is disabled.")


def _timeline_findings(parent, obstacles, errors, notes, cache=None):
    """Check Setup: the timeline, animated obstacles, time curves and what
    force curves can include."""
    try:
        line = timeline(parent)
    except hou.Error as exc:
        if not any("Time tab" in error for error in errors):
            errors.append(_error_text(exc))
        line = None
    for geo in obstacles:
        if line is None or not transform_is_animated(parent, geo):
            continue
        motion = obstacle_motion(parent, geo, line)
        if motion["kind"] == "static":
            continue
        if not _obstacle_displacement_is_zero(parent, geo):
            errors.append(ANIMATED_OBSTACLE_EXPRESSION_ERROR.format(geo=geo))
        if motion["kind"] == "translation":
            notes.append(
                f"Obstacle {geo} moves (its transform is keyframed; "
                f"translation only): written as a displacement table over "
                f"the {line.steps} time steps, its rest pose being the pose "
                f"at frame {line.start_frame:g}.")
            continue
        rate = sequence_rate(line)
        if rate is None:
            errors.append(SEQUENCE_RATE_ERROR.format(geo=geo, dt=line.dt,
                                                     t0=line.t0))
            continue
        frames = int(round(float(line.times[-1]) * rate)) + 1
        _, _, exact = _cached(
            cache, "obstacles", int(geo),
            lambda g=geo: obstacle_surface(parent, g))
        notes.append(
            f"Obstacle {geo} rotates or scales over time: it is written as "
            f"{frames} posed copies of its surface (input/"
            f"{OBSTACLE_SEQUENCE_DIR.format(geo)}/, {rate} per simulated "
            f"second), its rest pose being the pose at frame "
            f"{line.start_frame:g}."
            + ("" if exact else " Its points come from Houdini (single "
               "precision); .obj and .msh files are written at full "
               "precision."))
    pressures = set()
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval():
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            count = parent.parm(f"sideset_selection{geo}_{vol}")
            for j in range(1, (count.eval() if count is not None else 0) + 1):
                for k in range(1, parent.evalParm(
                        f"Boundary_Condition__{geo}_{vol}_{j}") + 1):
                    suffix = f"{geo}_{vol}_{j}_{k}"
                    mode = parent.parm(f"time_curve{suffix}").evalAsString()
                    kind = parent.evalParm(f"boundary_type{suffix}")
                    if mode != "entered" and kind in (3, 4):
                        pressures.add(_BC_TYPE_NAMES[kind])
                    if mode != "animated":
                        continue
                    parms = list(parent.parmTuple(
                        f"curve_multiplier{suffix}")) if kind in _VECTOR_BC_KINDS \
                        else [parent.parm(f"curve_scale{suffix}")]
                    if not any(p.isTimeDependent() for p in parms):
                        notes.append(
                            f"Geometry {geo} subdomain {vol} sideset {j} "
                            f"condition {k}: Time Curve is Animated "
                            f"multiplier, but the Multiplier has no keyframes "
                            f"({', '.join(f'{p.eval():g}' for p in parms)} "
                            f"throughout).")
    if pressures:
        notes.append(
            f"A time curve on {' and '.join(sorted(pressures))} needs PolyFEM "
            "ddb4579a3 or later; older builds refuse 'interpolation' on "
            "pressures (invalid input json).")
    if parent.evalParm("forces_fields"):
        missing = []
        if parent.evalParm("enable") and parent.evalParm("adhesion_enable"):
            missing.append("adhesion")
        if parent.evalParm("num_rayleigh") > 0:
            missing.append("Rayleigh damping")
        if missing:
            notes.append(
                f"Force curves leave out {' and '.join(missing)}: PolyFEM "
                f"does not write {'those' if len(missing) > 1 else 'its'} "
                f"nodal forces. Reactions are off by that force where it "
                f"acts.")


# What Show Expert Controls hides: (folder label path or parm names, label,
# whether the export reads it in this setup). Hiding never changes a value.
_EXPERT_FOLDERS = (
    (("Contact", "Barrier (IPC)", "Semi-Implicit Options"),
     lambda p: p.evalParm("enable") and not p.evalParm("gcp_enable")
     and _menu_token(p, "barrier_mode") == "semi_implicit"),
    (("Solver", "Augmented Lagrangian"), lambda p: True),
    (("Solver", "Rayleigh Damping"), lambda p: True),
    (("Solver", "Linear", "AMGCL Precond Settings"),
     lambda p: _menu_token(p, "solver_nl") != "BFGS"
     and _menu_token(p, "solver") == "AMGCL"),
    (("Output", "Advanced"), lambda p: True),
)
_EXPERT_PARMS = (
    (("solver_type",), "Solver > Linear > AMGCL",
     lambda p: _menu_token(p, "solver_nl") != "BFGS"
     and _menu_token(p, "solver") == "AMGCL"),
    (("pre_max_iter_hypre", "theta_hypre",
      "nodal_coarsening_hypre"),
     "Solver > Linear > Hypre",
     lambda p: _menu_token(p, "solver_nl") != "BFGS"
     and _menu_token(p, "solver") == "Hypre"),
    (("wolfe_c2", "wolfe_growth_factor", "wolfe_growth_limit",
      "wolfe_max_evaluations", "wolfe_max_objective_restarts",
      "wolfe_approximate_epsilon"), "Solver > Line Search > Wolfe",
     lambda p: _menu_token(p, "method") == "Wolfe"),
    (("ls_accept_energy_in_grad_mode",), "Solver > Line Search",
     lambda p: True),
    (("dhat_epsilon_scale",), "Contact > Barrier (IPC)",
     lambda p: p.evalParm("enable") and not p.evalParm("gcp_enable")
     and _menu_token(p, "barrier_mode") == "adaptive"),
    (("paraview_async",), "Output > Paraview > Options",
     lambda p: True),
    (("adv_stall_restart_enabled", "adv_stall_alpha_threshold",
      "adv_stall_patience", "adv_stall_min_iterations",
      "adv_stall_soft_limit", "adv_stall_max_restarts",
      "adv_stall_alpha_basis", "adv_stall_feasible_ratio",
      "adv_stall_remedy", "ls_failure_restarts"),
     "Solver > Advanced > Stall Restart",
     lambda p: True),
    (("apply_gradient_fd", "gradient_fd_eps"), "Solver > Advanced > Debugging",
     lambda p: True),
)


def _expert_notes(parent):
    """Expert settings that differ from their defaults while Show Expert
    Controls hides them (they still apply)."""
    if parent.parm("expert_mode") is None or parent.evalParm("expert_mode"):
        return []
    group = parent.parmTemplateGroup()
    changed = []
    multiparm_types = (
        hou.folderType.MultiparmBlock,
        hou.folderType.ScrollingMultiparmBlock,
        hou.folderType.TabbedMultiparmBlock,
    )

    def visit(templates, where):
        for template in templates:
            if template.type() == hou.parmTemplateType.Folder:
                if template.folderType() in multiparm_types:
                    tuple_parm = parent.parmTuple(template.name())
                    if tuple_parm is not None and not tuple_parm.isAtDefault():
                        changed.append(" > ".join(where + (template.label(),)))
                    continue
                visit(template.parmTemplates(),
                      where + (template.label(),))
                continue
            tuple_parm = parent.parmTuple(template.name())
            if tuple_parm is not None and not tuple_parm.isAtDefault():
                changed.append(" > ".join(where + (template.label(),)))

    for path, applies in _EXPERT_FOLDERS:
        folder = group.findFolder(path)
        if folder is not None and applies(parent):
            visit(folder.parmTemplates(), tuple(path))
    for names, where, applies in _EXPERT_PARMS:
        if not applies(parent):
            continue
        for name in names:
            tuple_parm = parent.parmTuple(name)
            if tuple_parm is not None and not tuple_parm.isAtDefault():
                changed.append(f"{where} > {tuple_parm.parmTemplate().label()}")
    if not changed:
        return []
    shown = "; ".join(changed[:12]) + ("; ..." if len(changed) > 12 else "")
    return [f"Show Expert Controls is off, but {len(changed)} expert "
            f"setting{'s' if len(changed) != 1 else ''} differ from the "
            f"default and still apply: {shown} (Main tab: Show Expert "
            "Controls shows them)."]


def setup_report_text(report, when=None):
    import datetime
    stamp = (when or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    errors, warnings, notes = (report["errors"], report["warnings"],
                               report["notes"])
    if not errors and not warnings:
        head = f"Checked {stamp}: no problems found."
    else:
        head = (f"Checked {stamp}: {len(errors)} error"
                f"{'s' if len(errors) != 1 else ''} (PolyFEM cannot run "
                f"this), {len(warnings)} warning"
                f"{'s' if len(warnings) != 1 else ''} (legal, but usually "
                "wrong).")
    lines = [head]
    lines.extend("ERROR: " + item for item in errors)
    lines.extend("WARNING: " + item for item in warnings)
    lines.extend("Note: " + item for item in notes)
    return "\n".join(lines)


def _set_check_report(parent, report):
    parm = parent.parm("check_report")
    if parm is not None:
        with hou.undos.disabler():
            parm.set(setup_report_text(report))


def check_setup_button(kwargs):
    """Check Setup button: list every problem at once."""
    parent = kwargs["node"]
    report = check_setup(parent, "check", raise_shared=True)
    _set_check_report(parent, report)
    if report["errors"] and not _resume_checkpoint(parent):
        _raise_saint_venant_axes(parent)
    if report["errors"] or report["warnings"]:
        _message(setup_report_text(report))
    else:
        _status("Check Setup: no problems found.")
    return report


def gate_setup(parent, purpose, cache=None):
    """Run Check Setup before Write/Run; True means go ahead.

    Errors refuse (PolyFEM could not run the scene). Warnings are confirmed
    before a run, once per node for the same list; writing only reports
    them, and without an interface (hython) they are printed.
    """
    report = check_setup(parent, purpose, cache=cache, raise_shared=True)
    _set_check_report(parent, report)
    if report["errors"]:
        if not _resume_checkpoint(parent):
            _raise_saint_venant_axes(parent)
        _message("PolyFEM cannot run this setup yet (Check Setup on the Main "
                 "tab lists it too):\n\n" + "\n".join(
                     "- " + item for item in report["errors"]),
                 severity=hou.severityType.Error)
        return False
    if not report["warnings"]:
        return True
    text = "\n".join("- " + item for item in report["warnings"])
    if purpose == "write" or not hou.isUIAvailable():
        _status(f"Check Setup: {len(report['warnings'])} warning(s); see the "
                "Check Setup report on the Main tab.")
        if not hou.isUIAvailable():
            print("[PolyFEM HDA] Check Setup warnings:\n" + text)
        return True
    key = _digest(report["warnings"])
    if parent.userData(_WARNINGS_ACCEPTED) == key:
        return True
    choice = hou.ui.displayMessage(
        "Check Setup found something that is legal but usually wrong:\n\n"
        + text, buttons=("Run Anyway", "Cancel"),
        severity=hou.severityType.Warning, default_choice=1, close_choice=1,
        title="PolyFEM: Check Setup")
    if choice != 0:
        return False
    parent.setUserData(_WARNINGS_ACCEPTED, key)
    return True


