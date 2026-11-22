def _restore_space(parent, data, material_slots, legacy, warnings):
    _restore_remesh(parent, data)
    if parent.evalParm("remeshing_enabled"):
        warnings.append(
            "Legacy remeshing values retained for inspection. "
            + LEGACY_REMESH_ERROR)
    space = data.get("space")
    if isinstance(space, list):
        entries = [entry for entry in space if isinstance(entry, dict)]
        space = entries[0] if entries else {}
    if not isinstance(space, dict):
        return
    orders = space.get("discr_order", [])
    if isinstance(orders, (int, float)):
        for geo in range(1, parent.evalParm("num_geos") + 1):
            if parent.evalParm(f"is_obstacle{geo}"):
                continue
            for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
                parent.setParms({f"mainOrder{geo}_{vol}": int(orders) - 1})
    elif isinstance(orders, list):
        for entry in orders:
            if not isinstance(entry, dict) or "order" not in entry:
                continue
            slots = _material_slots(entry.get("id"), material_slots, legacy)
            if not slots:
                warnings.append(
                    f"Discretization order id {entry.get('id')} could not be "
                    "mapped to a geometry/subdomain.")
                continue
            for geo, vol in slots:
                parm = parent.parm(f"mainOrder{geo}_{vol}")
                if parm is not None:
                    parm.set(max(0, int(entry["order"]) - 1))


def _restore_solver(parent, data, legacy, warnings):
    solver = data.get("solver")
    if not isinstance(solver, dict):
        return
    parms = {}
    if "max_threads" in solver:
        parms["max_threads"] = solver["max_threads"]
    nonlinear = solver.get("nonlinear", {})
    if isinstance(nonlinear, dict):
        for keys, parm in (
                (("max_iterations",), "max_iterations"),
                (("x_delta_tol", "x_delta"), "x_delta"),
                (("grad_norm_tol", "grad_norm"), "grad_norm"),
                (("first_grad_norm_tol",), "first_grad_norm_tol"),
                (("rel_grad_norm_tol",), "rel_grad_norm_tol"),
                (("rel_x_delta_tol",), "rel_x_delta_tol"),
                (("newton_decrement_tol",), "newton_decrement_tol"),
                (("allow_non_grad_convergence",),
                 "allow_non_grad_convergence"),
                (("iterations_per_strategy",), "iterations_per_strategy"),
                (("allow_out_of_iterations",), "allow_out_of_iterations")):
            key = next(
                (candidate for candidate in keys if candidate in nonlinear),
                None)
            if key is not None:
                parms[parm] = nonlinear[key]
        norm_types = {"Euclidean": 0, "L2": 1, "Linf": 2}
        if nonlinear.get("norm_type") in norm_types:
            parms["norm_type"] = norm_types[nonlinear["norm_type"]]
        nonlinear_type = nonlinear.get("solver", "Newton")
        nonlinear_types = {
            name: index for index, name in enumerate(
                parent.parm("solver_nl").menuItems())}
        if isinstance(nonlinear_type, list):
            # polysolve also accepts an explicit strategy list; the UI holds
            # the single method, so take the first entry's type.
            first = nonlinear_type[0] if nonlinear_type else {}
            nonlinear_type = first.get("type", "Newton") \
                if isinstance(first, dict) else "Newton"
            warnings.append(
                "solver.nonlinear.solver was a strategy list; only its first "
                "method was restored.")
        if nonlinear_type in nonlinear_types \
                and parent.parm("solver_nl") is not None:
            parms["solver_nl"] = nonlinear_types[nonlinear_type]
        elif isinstance(nonlinear_type, str) and nonlinear_type in _UNSUPPORTED_NONLINEAR:
            warnings.append(
                f"Nonlinear solver '{nonlinear_type}' is not offered by the "
                f"HDA: it {_UNSUPPORTED_NONLINEAR[nonlinear_type]}; Newton "
                "was kept.")
        elif not isinstance(nonlinear_type, str) or nonlinear_type not in nonlinear_types:
            warnings.append(
                f"Nonlinear solver '{nonlinear_type}' is not offered by the "
                "HDA; Newton was kept.")
        # Per-method blocks: read whichever the file carries into the shared
        # UI parameters (Newton/DenseNewton, L-BFGS variants, ADAM variants).
        for block in ("Newton", "DenseNewton"):
            newton = nonlinear.get(block, {})
            if isinstance(newton, dict):
                for key, parm in (
                        ("residual_tolerance", "residual_tolerance"),
                        ("reg_weight_min", "reg_weight_min"),
                        ("reg_weight_max", "reg_weight_max"),
                        ("reg_weight_inc", "reg_weight_inc"),
                        ("force_psd_projection", "force_psd_projection"),
                        ("use_psd_projection", "use_psd_projection"),
                        ("use_psd_projection_in_regularized",
                         "use_psd_projection_reg")):
                    if key in newton:
                        parms[parm] = newton[key]
        for block in ("L-BFGS", "L-BFGS-B"):
            lbfgs = nonlinear.get(block, {})
            if isinstance(lbfgs, dict) and "history_size" in lbfgs:
                parms["history_size"] = lbfgs["history_size"]
        for block in ("ADAM", "StochasticADAM"):
            adam = nonlinear.get(block, {})
            if isinstance(adam, dict):
                for key, parm in (("alpha", "adam_alpha"),
                                  ("beta_1", "adam_beta1"),
                                  ("beta_2", "adam_beta2"),
                                  ("epsilon", "adam_epsilon"),
                                  ("erase_component_probability",
                                   "erase_component_probability")):
                    if key in adam:
                        parms[parm] = adam[key]
        sgd = nonlinear.get("StochasticGradientDescent", {})
        if isinstance(sgd, dict) and "erase_component_probability" in sgd:
            parms["erase_component_probability"] = \
                sgd["erase_component_probability"]
        advanced = nonlinear.get("advanced", {})
        if isinstance(advanced, dict):
            for keys, parm in (
                    (("f_delta_tol", "f_delta"), "f_delta"),
                    (("f_delta_step_tol",), "f_delta_step_tol"),
                    (("derivative_along_delta_x_tol",),
                     "derivative_along_delta_x_tol"),
                    (("gradient_fd_eps",), "gradient_fd_eps")):
                key = next(
                    (candidate for candidate in keys if candidate in advanced),
                    None)
                if key is not None:
                    parms[parm] = advanced[key]
            fd_modes = {name: index for index, name in enumerate((
                "None", "DirectionalDerivative", "FullFiniteDiff"))}
            if advanced.get("apply_gradient_fd") in fd_modes:
                parms["apply_gradient_fd"] = \
                    fd_modes[advanced["apply_gradient_fd"]]
        line_search = nonlinear.get("line_search", {})
        if isinstance(line_search, dict):
            methods = {
                name: index for index, name in enumerate((
                    "Armijo", "RobustArmijo", "Backtracking", "None",
                    "Wolfe"))}
            method = line_search.get("method")
            if method == "none":
                method = "None"
            aliases = {"ArmijoAlt": "Armijo", "MoreThuente": "Backtracking",
                       "ResidualBacktracking": "Backtracking"}
            if isinstance(method, str) and method in aliases:
                warnings.append(
                    f"Line search '{method}' is not available in this "
                    f"PolyFEM; '{aliases[method]}' was restored instead.")
                method = aliases[method]
            if isinstance(method, str) and method in methods:
                parms["method"] = methods[method]
            for key, parm in (
                    ("use_grad_norm_tol", "use_grad_norm_tol"),
                    ("min_step_size", "min_step_size"),
                    ("max_step_size_iter", "max_step_size_iter"),
                    ("min_step_size_final", "min_step_size_final"),
                    ("max_step_size_iter_final", "max_step_size_iter_final"),
                    ("default_init_step_size", "default_init_step_size"),
                    ("step_ratio", "step_ratio")):
                if key in line_search:
                    parms[parm] = line_search[key]
            if "accept_energy_decrease_in_grad_mode" in line_search \
                    and parent.parm("ls_accept_energy_in_grad_mode") is not None:
                parms["ls_accept_energy_in_grad_mode"] = int(bool(
                    line_search["accept_energy_decrease_in_grad_mode"]))
            armijo = line_search.get("Armijo", {})
            robust = line_search.get("RobustArmijo", {})
            wolfe = line_search.get("Wolfe", {})
            if isinstance(armijo, dict) and "c" in armijo:
                parms["armijo_c"] = armijo["c"]
            if isinstance(armijo, dict) and "roundoff_tolerance" in armijo:
                parms["armijo_roundoff_tolerance"] = armijo["roundoff_tolerance"]
            if isinstance(robust, dict) \
                    and "delta_relative_tolerance" in robust:
                parms["delta_relative_tolerance"] = \
                    robust["delta_relative_tolerance"]
            if isinstance(wolfe, dict):
                for key, parm in (
                        ("c2", "wolfe_c2"),
                        ("growth_factor", "wolfe_growth_factor"),
                        ("growth_limit", "wolfe_growth_limit"),
                        ("max_evaluations", "wolfe_max_evaluations"),
                        ("max_objective_restarts",
                         "wolfe_max_objective_restarts"),
                        ("approximate_wolfe_epsilon",
                         "wolfe_approximate_epsilon")):
                    if key in wolfe:
                        parms[parm] = wolfe[key]

    contact = solver.get("contact", {})
    if isinstance(contact, dict):
        for key, parm in (
                ("friction_iterations", "friction_iterations"),
                ("tangential_adhesion_iterations", "tangent_iter"),
                ("friction_convergence_tol", "friction_convergence_tol")):
            if key in contact:
                parms[parm] = contact[key]
        ccd = contact.get("CCD", {})
        if isinstance(ccd, dict):
            broad = {
                name: index for index, name in enumerate((
                    "hash_grid", "brute_force", "spatial_hash", "BVH",
                    "sweep_and_prune", "sweep_and_tiniest_queue"))}
            if ccd.get("broad_phase") in broad:
                parms["broad_phase"] = broad[ccd["broad_phase"]]
            if "tolerance" in ccd:
                parms["CCD_tolerance"] = ccd["tolerance"]
            if "max_iterations" in ccd:
                parms["ccd_max_iterations"] = ccd["max_iterations"]
            limits = ccd.get("resource_limits")
            if isinstance(limits, dict):
                items = limits.get("max_cell_items", -1)
                pairs = limits.get("max_candidate_emissions", -1)
                if items < 0 or pairs < 0:
                    parms["resource_limits"] = 0  # automatic
                elif items == 0 and pairs == 0:
                    parms["resource_limits"] = 1  # off
                else:
                    parms["resource_limits"] = 2  # custom
                    parms["resource_max_cell_items"] = int(items)
                    parms["resource_max_candidate_emissions"] = int(pairs)
        stiffness = contact.get("barrier_stiffness", "semi_implicit")
        if stiffness == "semi_implicit":
            parms["barrier_mode"] = 0
            semi = contact.get("semi_implicit", {})
            if isinstance(semi, dict):
                if semi.get("constraint_floor", 0) != 0:
                    warnings.append(
                        "constraint_floor has been retired and is ignored; "
                        "contact barriers remain active below the former floor.")
                for key, parm in (
                        ("trim_lower", "si_trim_lower"),
                        ("trim_upper", "si_trim_upper"),
                        ("trim_factor", "si_trim_factor"),
                        ("initial_trim_estimate", "si_initial_trim_estimate"),
                        ("kappa_spread", "si_kappa_spread"),
                        ("kappa_min", "si_kappa_min"),
                        ("refresh_interval", "si_refresh_interval"),
                        ("conditioning_cap", "si_conditioning_cap"),
                        ("controller_interval", "si_controller_interval"),
                        ("trial_displacement_cap",
                         "si_trial_displacement_cap"),
                        ("force_continuation", "si_force_continuation"),
                        ("continuation_max_ratio",
                         "si_continuation_max_ratio")):
                    if key in semi:
                        parms[parm] = semi[key]
                identities = {"parent": 0, "stencil": 1}
                if semi.get("coefficient_identity") in identities:
                    parms["si_coefficient_identity"] = \
                        identities[semi["coefficient_identity"]]
                lag_modes = {"realized_force": 0, "follow_stiffness": 1}
                if semi.get("friction_lag") in lag_modes:
                    parms["si_friction_lag"] = lag_modes[semi["friction_lag"]]
                band_statistics = {"rms": 0, "force_weighted": 1}
                if semi.get("band_statistic") in band_statistics:
                    parms["si_band_statistic"] = \
                        band_statistics[semi["band_statistic"]]
                clamped_modes = {
                    "exclude_statistics": 0, "keep": 1,
                    "exclude_collisions": 2}
                if semi.get("clamped_contacts") in clamped_modes \
                        and parent.parm("si_clamped_contacts") is not None:
                    parms["si_clamped_contacts"] = \
                        clamped_modes[semi["clamped_contacts"]]
                gb_dofs_modes = {"all": 0, "free": 1}
                if semi.get("gradient_balance_dofs") in gb_dofs_modes \
                        and parent.parm("si_gradient_balance_dofs") is not None:
                    parms["si_gradient_balance_dofs"] = \
                        gb_dofs_modes[semi["gradient_balance_dofs"]]
                if semi.get("collapse_guard_basis", "pair") != "pair":
                    warnings.append(
                        "collapse_guard_basis {!r} is not supported by the asset: "
                        "the experimental force-weighted band is always exported "
                        "with the pair-basis guard (EF-07).".format(
                            semi.get("collapse_guard_basis")))
                if semi.get("gap_floor", 0) != 0:
                    warnings.append(
                        "gap_floor is experimental and not exposed by the "
                        "HDA; it was dropped on import.")
                restart = semi.get("restart", {})
                if isinstance(restart, dict):
                    for key, parm in (
                            ("enabled", "si_restart_enabled"),
                            ("alpha_threshold", "si_alpha_threshold"),
                            ("patience", "si_patience"),
                            ("min_iterations", "si_min_iterations"),
                            ("soft_iteration_limit", "si_soft_iteration_limit"),
                            ("max_restarts", "si_max_restarts"),
                            ("stall_trim_factor", "si_stall_trim_factor")):
                        if key in restart:
                            parms[parm] = restart[key]
        elif stiffness == "adaptive":
            parms["barrier_mode"] = 1
            if "initial_barrier_stiffness" in contact:
                parms["initial_barrier_stiffness"] = \
                    contact["initial_barrier_stiffness"]
            if "dhat_epsilon_scale" in contact \
                    and parent.parm("dhat_epsilon_scale") is not None:
                parms["dhat_epsilon_scale"] = contact["dhat_epsilon_scale"]
            if "adaptive_barrier_stiffness_multiplier" in contact:
                warnings.append(
                    "solver.contact.adaptive_barrier_stiffness_multiplier has "
                    "no current equivalent; the adaptive stiffness default was "
                    "retained.")
        elif isinstance(stiffness, (int, float)):
            parms["barrier_mode"] = 2
            parms["barrier_stiffness"] = stiffness

    al = solver.get("augmented_lagrangian", {})
    if isinstance(al, dict):
        if al.get("initial_weight") == "hessian_scaled":
            parms["al_hessian_scaled"] = 1
            if "initial_weight_multiplier" in al:
                parms["al_weight_multiplier"] = \
                    al["initial_weight_multiplier"]
        elif "initial_weight" in al:
            parms["al_hessian_scaled"] = 0
            parms["al_initial_weight"] = al["initial_weight"]
        for key, parm in (
                ("scaling", "al_scaling"), ("max_weight", "al_max_weight"),
                ("eta", "al_eta")):
            if key in al:
                parms[parm] = al[key]
        if "error" in al and parent.parm("al_error") is not None:
            parms["al_error"] = al["error"]
        lumpings = {"row_sum": 0, "hrz": 1}
        if al.get("lumping") in lumpings:
            parms["al_lumping"] = lumpings[al["lumping"]]
        al_nonlinear = al.get("nonlinear", {})
        if isinstance(al_nonlinear, dict) \
                and "max_iterations" in al_nonlinear:
            parms["al_max_iterations"] = al_nonlinear["max_iterations"]
        else:
            # Without its own budget the AL phase uses the nonlinear one
            # (build_solver writes it only when they differ); the node's
            # earlier value must not survive the import.
            parms["al_max_iterations"] = parms.get(
                "max_iterations", parent.evalParm("max_iterations"))
        al_budget = al.get("budget", {})
        if isinstance(al_budget, dict) \
                and parent.parm("al_max_passes") is not None:
            for key, parm in (
                    ("max_passes", "al_max_passes"),
                    ("stagnation_window", "al_stagnation_window"),
                    ("progress_tolerance", "al_progress_tol"),
                    ("snap_tolerance", "al_snap_tol"),
                    ("drift_tolerance", "al_drift_tol")):
                if key in al_budget:
                    parms[parm] = al_budget[key]

    advanced = solver.get("advanced", {})
    if isinstance(advanced, dict):
        for key, parm in (
                ("cache_size", "cache_size"),
                ("lump_mass_matrix", "lump_mass_matrix"),
                ("lagged_regularization_weight",
                 "lagged_regularization_weight"),
                ("lagged_regularization_iterations",
                 "lagged_regularization_iterations"),
                ("jacobian_threshold", "jacobian_threshold")):
            if key in advanced:
                parms[parm] = advanced[key]
        inversion = {"Discrete": 0, "Conservative": 1}
        if advanced.get("check_inversion") in inversion:
            parms["inversion_method"] = inversion[advanced["check_inversion"]]
        stall_restart = advanced.get("stall_restart", {})
        if isinstance(stall_restart, dict) \
                and parent.parm("adv_stall_restart_enabled") is not None:
            for key, parm in (
                    ("enabled", "adv_stall_restart_enabled"),
                    ("alpha_threshold", "adv_stall_alpha_threshold"),
                    ("patience", "adv_stall_patience"),
                    ("min_iterations", "adv_stall_min_iterations"),
                    ("soft_iteration_limit", "adv_stall_soft_limit"),
                    ("max_restarts", "adv_stall_max_restarts"),
                    ("feasible_ratio_threshold", "adv_stall_feasible_ratio")):
                if key in stall_restart:
                    parms[parm] = stall_restart[key]
            alpha_bases = {"feasible_bound": 0, "absolute": 1}
            if stall_restart.get("alpha_basis") in alpha_bases:
                parms["adv_stall_alpha_basis"] = \
                    alpha_bases[stall_restart["alpha_basis"]]
            remedies = {"force_psd_projection": 0, "none": 1}
            if stall_restart.get("remedy") in remedies:
                parms["adv_stall_remedy"] = remedies[stall_restart["remedy"]]
        if "line_search_failure_restarts" in advanced \
                and parent.parm("ls_failure_restarts") is not None:
            parms["ls_failure_restarts"] = \
                advanced["line_search_failure_restarts"]

    linear = solver.get("linear", {})
    bfgs = isinstance(nonlinear, dict) and nonlinear.get("solver") == "BFGS"
    if bfgs:
        # The HDA supplies BFGS's dense linear solver itself; the Linear
        # folder keeps its own (sparse) choice for the other methods.
        if isinstance(linear, dict) \
                and linear.get("solver") not in (None, _BFGS_LINEAR_SOLVER):
            warnings.append(
                f"solver.linear.solver '{linear.get('solver')}' was not "
                f"restored: BFGS needs a dense linear solver, and the HDA "
                f"exports {_BFGS_LINEAR_SOLVER} with it.")
    elif isinstance(linear, dict):
        _restore_linear_solver(parent, linear, parms, warnings)
    _set_import_parms(parent, parms, "solver", warnings)

    rayleigh = solver.get("rayleigh_damping", [])
    if isinstance(rayleigh, dict):
        rayleigh = [rayleigh]
    if isinstance(rayleigh, list):
        parent.setParms({"num_rayleigh": len(rayleigh)})
        forms = {"elasticity": 0, "contact": 1, "friction": 2}
        for index, entry in enumerate(rayleigh, start=1):
            if not isinstance(entry, dict):
                continue
            values = {}
            if entry.get("form") in forms:
                values[f"rayleigh_form{index}"] = forms[entry["form"]]
            if "stiffness_ratio" in entry:
                values[f"stiffness_ratio{index}"] = entry["stiffness_ratio"]
            if "lagging_iterations" in entry:
                values[f"rayleigh_lagging{index}"] = \
                    entry["lagging_iterations"]
            if values:
                _set_import_parms(parent, values, f"solver.rayleigh_damping[{index}]", warnings)


def _restore_linear_solver(parent, linear, parms, warnings):
    """Map /solver/linear back onto the Linear folder (menu tokens by name)."""
    solver_parm = parent.parm("solver")
    tokens = list(solver_parm.menuItems())
    name = linear.get("solver", "")
    if isinstance(name, list):
        name = name[0] if name else ""
        warnings.append(
            "solver.linear.solver listed several solvers; only the first "
            "was restored.")
    aliases = {"Eigen:SparseLU": "Eigen::SparseLU",
               "Eigen:PardisoLU": "Eigen::PardisoLU"}
    if not isinstance(name, str):
        warnings.append("solver.linear.solver must be a solver name.")
        return
    name = aliases.get(name, name)
    if not name:
        parms["solver"] = tokens.index("auto")
    elif name in tokens:
        parms["solver"] = tokens.index(name)
    else:
        warnings.append(
            f"Linear solver '{name}' is not offered by the HDA; the "
            "automatic choice was kept.")
        return
    precond = linear.get("precond")
    precond_tokens = list(parent.parm("precond").menuItems())
    if precond in precond_tokens:
        parms["precond"] = precond_tokens.index(precond)
    block = linear.get(name, {}) if isinstance(linear.get(name), dict) else {}
    if name.startswith("Eigen::") and block:
        if "max_iter" in block:
            parms["max_iter"] = block["max_iter"]
        if "tolerance" in block:
            parms["tolerance"] = block["tolerance"]
    pardiso = linear.get("Pardiso", {})
    if isinstance(pardiso, dict) and "mtype" in pardiso:
        mtypes = list(parent.parm("mtype").menuItems())
        if str(pardiso["mtype"]) in mtypes:
            parms["mtype"] = mtypes.index(str(pardiso["mtype"]))
    hypre = linear.get("Hypre", {})
    if isinstance(hypre, dict):
        # `dimension` is not a control: the exporter always writes 3.
        for key, parm in (("max_iter", "max_iter_hypre"),
                          ("pre_max_iter", "pre_max_iter_hypre"),
                          ("tolerance", "tolerance_hypre_AMGCL"),
                          ("theta", "theta_hypre"),
                          ("nodal_coarsening", "nodal_coarsening_hypre")):
            if key in hypre:
                parms[parm] = hypre[key]

    def restore_token(parm_name, value, what):
        if value is None:
            return
        tokens = list(parent.parm(parm_name).menuItems())
        if value in tokens:
            parms[parm_name] = tokens.index(value)
        else:
            warnings.append(
                f"AMGCL {what} '{value}' is not offered by the HDA; the "
                "default was kept.")

    amgcl = linear.get("AMGCL", {})
    if isinstance(amgcl, dict):
        amg_solver = amgcl.get("solver", {})
        if isinstance(amg_solver, dict):
            if "maxiter" in amg_solver:
                parms["max_iter"] = amg_solver["maxiter"]
            if "tol" in amg_solver:
                parms["tolerance_hypre_AMGCL"] = amg_solver["tol"]
            restore_token("solver_type", amg_solver.get("type"), "solver type")
        precond_block = amgcl.get("precond", {})
        if isinstance(precond_block, dict):
            restore_token("class", precond_block.get("class"),
                          "preconditioner class")
            for key, parm in (("max_levels", "max_levels"),
                              ("direct_coarse", "direct_coarse"),
                              ("ncycle", "ncycle")):
                if key in precond_block:
                    parms[parm] = precond_block[key]
            relax = precond_block.get("relax", {})
            if isinstance(relax, dict):
                restore_token("relax_type", relax.get("type"), "relax type")
                for key, parm in (("degree", "degree"),
                                  ("power_iters", "power_iters"),
                                  ("higher", "higher"), ("lower", "lower"),
                                  ("scale", "scale_amgcl")):
                    if key in relax:
                        parms[parm] = relax[key]
            coarsening = precond_block.get("coarsening", {})
            if isinstance(coarsening, dict):
                restore_token("coarsening_type", coarsening.get("type"),
                              "coarsening type")
                for key, parm in (
                        ("estimate_spectral_radius",
                         "estimate_spectral_radius"),
                        ("relax", "coarse_relax")):
                    if key in coarsening:
                        parms[parm] = coarsening[key]
                aggr = coarsening.get("aggr", {})
                if isinstance(aggr, dict) and "eps_strong" in aggr:
                    parms["eps_strong"] = aggr["eps_strong"]


def _restore_output(parent, data, warnings):
    output = data.get("output")
    if not isinstance(output, dict):
        return
    checkpoint_output = output.get("checkpoint")
    checkpoint_output = (checkpoint_output
                         if isinstance(checkpoint_output, dict) else {})
    parms = {
        "output_json": int("json" in output),
        "restart_json": int("restart_json" in output
                            or bool(checkpoint_output.get("path", "")))}
    paraview = output.get("paraview", {})
    if isinstance(paraview, dict):
        filename = paraview.get("file_name")
        if isinstance(filename, str):
            parms["paraview_file_name"] = os.path.basename(filename)
        for key, parm in (
                ("vismesh_rel_area", "vismesh_rel_area"),
                ("skip_frame", "skip_frame"),
                ("high_order_mesh", "high_order_mesh"),
                ("wireframe", "paraview_wireframe"),
                ("points", "paraview_points")):
            if key in paraview:
                parms[parm] = paraview[key]
        if "async" in paraview and parent.parm("paraview_async") is not None:
            parms["paraview_async"] = int(bool(paraview["async"]))
        options = paraview.get("options", {})
        if isinstance(options, dict):
            for key, parm in (
                    ("use_hdf5", "use_hdf5"),
                    ("material", "materials_fields"),
                    ("body_ids", "body_ids_fields"),
                    ("contact_forces", "contact_forces_fields"),
                    ("friction_forces", "friction_forces_fields"),
                    ("normal_adhesion_forces",
                     "normal_adhesion_forces_fields"),
                    ("tangential_adhesion_forces",
                     "tangential_adhesion_forces_fields"),
                    ("velocity", "velocity_fields"),
                    ("acceleration", "acceleration_fields"),
                    ("scalar_values", "scalar_values"),
                    ("tensor_values", "tensor_values"),
                    ("discretization_order", "discr_fields"),
                    ("nodes", "nodes_field"), ("forces", "forces_fields"),
                    ("force_high_order", "forces_HO_output"),
                    ("jacobian_validity", "validity_fields")):
                if key in options:
                    parms[parm] = options[key]
            if "obstacle_volume" in options \
                    and parent.parm("obstacle_volume") is not None:
                parms["obstacle_volume"] = int(bool(options["obstacle_volume"]))
            # Contact forces are written only with contact on: a file without
            # contact says nothing about the toggle, so keep its default.
            contact = data.get("contact")
            if not (isinstance(contact, dict) and contact.get("enabled")):
                parms.pop("contact_forces_fields", None)
        parms["minimal_fields"] = int(
            isinstance(paraview.get("fields"), list)
            and not bool(options.get("material", False)))
        fields = paraview.get("fields")
        if isinstance(fields, list):
            # The surface field toggles only leave a trace in the whitelist.
            parms["normals_fields"] = int("normals" in fields)
            parms["displaced_normals_fields"] = int(
                "displaced_normals" in fields and "normals" not in fields)
            parms["sidesets_fields"] = int("sidesets" in fields)
            parms["solution_grad_fields"] = int("solution_gradient" in fields)
            parms["adaptive_dhat"] = int("adaptive_dhat" in fields)

    output_data = output.get("data", {})
    if isinstance(output_data, dict):
        for key, parm in (
                ("solution", "solution_file"),
                ("stress_mat", "stress_mat_file"),
                ("mises", "mises_file"),
                ("nodes", "nodes")):
            parms[parm] = int(key in output_data)
        # Legacy state key: the per-step restart state was implied by
        # restart_json, not the diagnostic State toggle.
        if "diagnostic_state" in output_data:
            parms["state_file"] = int(bool(
                str(output_data["diagnostic_state"] or "").strip()))
        elif "state" in output_data:
            parms["state_file"] = int(
                output_data["state"] != LEGACY_RESTART_STATE_FILE)
        if parent.parm("rest_mesh_file") is not None:
            parms["rest_mesh_file"] = 0
        if "rest_mesh" in output_data and warnings is not None:
            warnings.append(
                "output.data.rest_mesh has been retired in PolyFEM and was "
                "not restored.")
    advanced = output.get("advanced", {})
    if isinstance(advanced, dict):
        for key, parm in (
                ("timestep_prefix", "timestep_prefix"),
                ("sol_on_grid", "sol_on_grid"),
                ("compute_error", "compute_error"),
                ("sol_at_node", "sol_at_node"),
                ("vis_boundary_only", "vis_boundary_only"),
                ("curved_mesh_size", "curved_mesh_size"),
                ("save_solve_sequence_debug", "save_solve_sequence_debug"),
                ("save_ccd_debug_meshes", "save_ccd_debug_meshes"),
                ("save_time_sequence", "save_time_sequence"),
                ("save_nl_solve_sequence", "save_nl_solve_sequence"),
                ("spectrum", "spectrum")):
            if key in advanced:
                parms[parm] = advanced[key]
    log = output.get("log", {})
    if isinstance(log, dict):
        if "level" in log:
            parms["log_level"] = log["level"]
        if "quiet" in log:
            parms["log_quiet"] = log["quiet"]
        parms["write_log"] = int("path" in log)
    _set_import_parms(parent, parms, "output", warnings)


def _restore_units(parent, data):
    units = data.get("units")
    if not isinstance(units, dict):
        parent.setParms({"units": 0})
        return
    parms = {"units": 1}
    for key, parm in (
            ("length", "length"), ("mass", "mass"), ("time", "time"),
            ("characteristic_length", "char_length")):
        if key in units:
            parms[parm] = units[key]
    parent.setParms(parms)


