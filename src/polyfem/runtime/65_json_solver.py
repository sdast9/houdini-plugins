def build_remesh(parent):
    """Guard retained for callers of the former dynamic-remesh exporter."""
    if parent.evalParm("remeshing_enabled"):
        raise hou.NodeError(LEGACY_REMESH_ERROR)
    return None


def _guard_forward_json(data):
    if not data:
        return
    if "states" in data:
        raise hou.NodeError("Extra JSON states dispatches PolyFEM to optimization; "
                            "this HDA writes forward simulations.")
    space = data.get("space")
    if isinstance(space, dict) and "remesh" in space:
        raise hou.NodeError("Extra JSON space.remesh: " + LEGACY_REMESH_ERROR)
    input_block = data.get("input")
    checkpoint = input_block.get("checkpoint") if isinstance(input_block, dict) else None
    if isinstance(checkpoint, dict) and checkpoint.get("path"):
        raise hou.NodeError("Use Resume Checkpoint on the Main tab instead of "
                            "setting input.checkpoint.path in Extra JSON.")


def build_space(parent, data, orders):
    build_remesh(parent)
    data["space"] = {"discr_order": orders}


def _menu_token(parent, name):
    """Token string of an ordinal menu parameter (evalParm gives the index)."""
    return parent.parm(name).evalAsString()


# Nonlinear methods the HDA no longer offers, and why (BFGS audit stage 5).
# PolyFEM's simulation cannot run them: its forward solve builds PolySolve's
# unconstrained solver, which has no box-constrained strategies, and none of
# its problems assembles a dense Hessian.
_UNSUPPORTED_NONLINEAR = {
    "L-BFGS-B": "is a box-constrained optimizer (PolyFEM uses it for shape "
                "and parameter optimization, not for a simulation step)",
    "MMA": "is a box-constrained optimizer (PolyFEM uses it for shape and "
           "parameter optimization, not for a simulation step)",
    "DenseNewton": "needs a dense Hessian, which PolyFEM's problems do not "
                   "assemble; Newton is the same method on the sparse one"}

# Dense BFGS factorizes a dense n x n matrix; PolySolve refuses any other
# linear solver for it, and the Linear folder offers only sparse ones.
_BFGS_LINEAR_SOLVER = "Eigen::LDLT"


def build_linear_solver(parent):
    """The /solver/linear block: only what the chosen solver actually reads.

    "auto" leaves the choice to PolyFEM/PolySolve (the best direct solver
    compiled into the binary), which is what every scene got before the
    linear settings were wired at all.
    """
    if _menu_token(parent, "solver_nl") == "BFGS":
        return {"solver": _BFGS_LINEAR_SOLVER}
    solver = _menu_token(parent, "solver")
    if solver == "auto":
        return None
    # PolyFEM validates the choice against the solvers compiled into the
    # binary and aborts otherwise; with the overwrite flag it logs a warning
    # and falls back to its default instead.
    linear = {"solver": solver, "enable_overwrite_solver": True}
    iterative_eigen = ("Eigen::ConjugateGradient",
                       "Eigen::LeastSquaresConjugateGradient", "Eigen::DGMRES",
                       "Eigen::BiCGSTAB", "Eigen::GMRES", "Eigen::MINRES")
    if solver in iterative_eigen:
        linear["precond"] = _menu_token(parent, "precond")
        linear[solver] = {"max_iter": parent.evalParm("max_iter"),
                          "tolerance": parent.evalParm("tolerance")}
    elif solver == "Pardiso":
        linear["Pardiso"] = {"mtype": int(_menu_token(parent, "mtype"))}
    elif solver == "Hypre":
        linear["Hypre"] = {
            "max_iter": parent.evalParm("max_iter_hypre"),
            "pre_max_iter": parent.evalParm("pre_max_iter_hypre"),
            "tolerance": parent.evalParm("tolerance_hypre_AMGCL"),
            "theta": parent.evalParm("theta_hypre"),
            "nodal_coarsening":
                bool(parent.evalParm("nodal_coarsening_hypre")),
            # Houdini scenes are 3-D; dimension > 1 switches BoomerAMG to
            # its systems (elasticity) settings, which the default of 1
            # never does.
            "dimension": 3}
    elif solver == "AMGCL":
        linear["AMGCL"] = {
            "solver": {"maxiter": parent.evalParm("max_iter"),
                       "tol": parent.evalParm("tolerance_hypre_AMGCL"),
                       "type": _menu_token(parent, "solver_type")},
            "precond": {
                "class": _menu_token(parent, "class"),
                "max_levels": parent.evalParm("max_levels"),
                "direct_coarse": bool(parent.evalParm("direct_coarse")),
                "ncycle": parent.evalParm("ncycle"),
                "relax": {"degree": parent.evalParm("degree"),
                          "type": _menu_token(parent, "relax_type"),
                          "power_iters": parent.evalParm("power_iters"),
                          "higher": parent.evalParm("higher"),
                          "lower": parent.evalParm("lower"),
                          "scale": bool(parent.evalParm("scale_amgcl"))},
                "coarsening": {
                    "type": _menu_token(parent, "coarsening_type"),
                    "estimate_spectral_radius":
                        bool(parent.evalParm("estimate_spectral_radius")),
                    "relax": parent.evalParm("coarse_relax"),
                    "aggr": {"eps_strong": parent.evalParm("eps_strong")}}}}
    return linear


def build_nonlinear_solver(parent):
    """The /solver/nonlinear block (rescaled-tolerances polysolve spec)."""
    ls = {"method": _menu_token(parent, "method"),
          "use_grad_norm_tol": parent.evalParm("use_grad_norm_tol"),
          "min_step_size": parent.evalParm("min_step_size"),
          "max_step_size_iter": parent.evalParm("max_step_size_iter"),
          "min_step_size_final": parent.evalParm("min_step_size_final"),
          "max_step_size_iter_final": parent.evalParm("max_step_size_iter_final"),
          "default_init_step_size": parent.evalParm("default_init_step_size"),
          "step_ratio": parent.evalParm("step_ratio"),
          "Armijo": {"c": parent.evalParm("armijo_c"),
                     "roundoff_tolerance":
                         parent.evalParm("armijo_roundoff_tolerance")},
          "RobustArmijo": {"delta_relative_tolerance":
                           parent.evalParm("delta_relative_tolerance")},
          "Wolfe": {
              "c2": parent.evalParm("wolfe_c2"),
              "growth_factor": parent.evalParm("wolfe_growth_factor"),
              "growth_limit": parent.evalParm("wolfe_growth_limit"),
              "max_evaluations": parent.evalParm("wolfe_max_evaluations"),
              "max_objective_restarts":
                  parent.evalParm("wolfe_max_objective_restarts"),
              "approximate_wolfe_epsilon":
                  parent.evalParm("wolfe_approximate_epsilon")}}
    if parent.parm("ls_accept_energy_in_grad_mode") is not None \
            and bool(parent.evalParm("ls_accept_energy_in_grad_mode")):
        ls["accept_energy_decrease_in_grad_mode"] = True

    # A scene saved with a withdrawn method (see _UNSUPPORTED_NONLINEAR) loads
    # with Houdini's own warning ("Parameter value ... in solver_nl is
    # invalid. Defaulting to 0."), i.e. as Newton; no withdrawn token reaches
    # this point.
    method = _menu_token(parent, "solver_nl")
    nonlinear = {
        "solver": method,
        "max_iterations": parent.evalParm("max_iterations"),
        "iterations_per_strategy": parent.evalParm("iterations_per_strategy"),
        "allow_out_of_iterations":
            bool(parent.evalParm("allow_out_of_iterations")),
        "x_delta_tol": parent.evalParm("x_delta"),
        "rel_x_delta_tol": parent.evalParm("rel_x_delta_tol"),
        "grad_norm_tol": parent.evalParm("grad_norm"),
        "rel_grad_norm_tol": parent.evalParm("rel_grad_norm_tol"),
        "norm_type": _menu_token(parent, "norm_type"),
        "first_grad_norm_tol": parent.evalParm("first_grad_norm_tol"),
        "newton_decrement_tol": parent.evalParm("newton_decrement_tol"),
        "allow_non_grad_convergence":
            bool(parent.evalParm("allow_non_grad_convergence")),
        "line_search": ls,
        "advanced": {"f_delta_tol": parent.evalParm("f_delta"),
                     "f_delta_step_tol": parent.evalParm("f_delta_step_tol"),
                     "derivative_along_delta_x_tol":
                         parent.evalParm("derivative_along_delta_x_tol"),
                     "apply_gradient_fd":
                         _menu_token(parent, "apply_gradient_fd"),
                     "gradient_fd_eps": parent.evalParm("gradient_fd_eps")}}
    # Per-method settings: written only for the chosen method so the file
    # says exactly what the run used.
    if method == "Newton":
        nonlinear[method] = {
            "residual_tolerance": parent.evalParm("residual_tolerance"),
            "reg_weight_min": parent.evalParm("reg_weight_min"),
            "reg_weight_max": parent.evalParm("reg_weight_max"),
            "reg_weight_inc": parent.evalParm("reg_weight_inc"),
            "force_psd_projection":
                bool(parent.evalParm("force_psd_projection")),
            "use_psd_projection": bool(parent.evalParm("use_psd_projection")),
            "use_psd_projection_in_regularized":
                bool(parent.evalParm("use_psd_projection_reg"))}
    elif method == "L-BFGS":
        nonlinear[method] = {"history_size": parent.evalParm("history_size")}
    elif method in ("ADAM", "StochasticADAM"):
        nonlinear[method] = {"alpha": parent.evalParm("adam_alpha"),
                             "beta_1": parent.evalParm("adam_beta1"),
                             "beta_2": parent.evalParm("adam_beta2"),
                             "epsilon": parent.evalParm("adam_epsilon")}
        if method == "StochasticADAM":
            nonlinear[method]["erase_component_probability"] = \
                parent.evalParm("erase_component_probability")
    elif method == "StochasticGradientDescent":
        nonlinear[method] = {"erase_component_probability":
                             parent.evalParm("erase_component_probability")}
    return nonlinear


def build_solver(parent, data):
    nonlinear = build_nonlinear_solver(parent)

    # augmented lagrangian (un-hardcoded vs 1.2)
    if parent.evalParm("al_hessian_scaled"):
        al = {"initial_weight": "hessian_scaled",
              "initial_weight_multiplier":
                  parent.evalParm("al_weight_multiplier")}
    else:
        al = {"initial_weight": parent.evalParm("al_initial_weight")}
    al.update({"scaling": parent.evalParm("al_scaling"),
               "max_weight": parent.evalParm("al_max_weight"),
               "eta": parent.evalParm("al_eta"),
               "lumping": _menu_token(parent, "al_lumping")})
    if parent.parm("al_error") is not None:
        al_err = float(parent.evalParm("al_error"))
        if not math.isclose(al_err, 0.01, rel_tol=1e-9, abs_tol=1e-12):
            al["error"] = al_err
    # The AL (boundary-condition preparation) phase reuses the nonlinear
    # settings with its own iteration budget.
    if parent.evalParm("al_max_iterations") != nonlinear["max_iterations"]:
        al["nonlinear"] = {
            "max_iterations": parent.evalParm("al_max_iterations")}
    # AL phase budget (RB-07): emit only when non-default so untouched scenes
    # keep identical params.json.
    if parent.parm("al_max_passes") is not None:
        al_max_passes = int(parent.evalParm("al_max_passes"))
        al_stag_win = int(parent.evalParm("al_stagnation_window"))
        al_prog_tol = float(parent.evalParm("al_progress_tol"))
        al_snap_tol = float(parent.evalParm("al_snap_tol"))
        al_drift_tol = float(parent.evalParm("al_drift_tol"))
        if (al_max_passes > 0 or al_stag_win > 0
                or not math.isclose(al_prog_tol, 0.01, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(al_snap_tol, 0.01, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(al_drift_tol, 0.01, rel_tol=1e-9, abs_tol=1e-12)):
            al["budget"] = {
                "max_passes": al_max_passes,
                "stagnation_window": al_stag_win,
                "progress_tolerance": al_prog_tol,
                "snap_tolerance": al_snap_tol,
                "drift_tolerance": al_drift_tol}

    # contact solver block -- current-branch schema
    broad_phase_dict = {0: "hash_grid", 1: "brute_force", 2: "spatial_hash",
                        3: "BVH", 4: "sweep_and_prune",
                        5: "sweep_and_tiniest_queue"}
    # RB-05 resource limits: -1 = automatic (PolyFEM's production ceilings
    # where the broad phase can enforce them), 0 = off, N = custom.
    limits_mode = parent.parm("resource_limits").evalAsString()
    if limits_mode == "off":
        resource_limits = {"max_cell_items": 0, "max_candidate_emissions": 0}
    elif limits_mode == "custom":
        resource_limits = {
            "max_cell_items": int(parent.evalParm("resource_max_cell_items")),
            "max_candidate_emissions":
                int(parent.evalParm("resource_max_candidate_emissions"))}
    else:
        resource_limits = {"max_cell_items": -1, "max_candidate_emissions": -1}
    contact = {
        "CCD": {"broad_phase": broad_phase_dict[parent.evalParm("broad_phase")],
                "tolerance": parent.evalParm("CCD_tolerance"),
                "max_iterations": parent.evalParm("ccd_max_iterations"),
                "resource_limits": resource_limits},
        "friction_iterations": parent.evalParm("friction_iterations"),
        "tangential_adhesion_iterations": parent.evalParm("tangent_iter"),
        "friction_convergence_tol":
            parent.evalParm("friction_convergence_tol")}

    barrier_mode = parent.evalParm("barrier_mode")
    if barrier_mode == 0:  # semi-implicit
        contact["barrier_stiffness"] = "semi_implicit"
        contact["semi_implicit"] = {
            "refresh_interval": parent.evalParm("si_refresh_interval"),
            "trim_lower": parent.evalParm("si_trim_lower"),
            "trim_upper": parent.evalParm("si_trim_upper"),
            "trim_factor": parent.evalParm("si_trim_factor"),
            "band_statistic": _menu_token(parent, "si_band_statistic"),
            "initial_trim_estimate":
                bool(parent.evalParm("si_initial_trim_estimate")),
            "kappa_spread": parent.evalParm("si_kappa_spread"),
            "kappa_min": parent.evalParm("si_kappa_min"),
            "conditioning_cap": parent.evalParm("si_conditioning_cap"),
            "controller_interval": parent.evalParm("si_controller_interval"),
            "trial_displacement_cap":
                parent.evalParm("si_trial_displacement_cap"),
            "force_continuation":
                bool(parent.evalParm("si_force_continuation")),
            "continuation_max_ratio":
                parent.evalParm("si_continuation_max_ratio"),
            "coefficient_identity":
                _menu_token(parent, "si_coefficient_identity"),
            "friction_lag": _menu_token(parent, "si_friction_lag"),
            "restart": {
                "enabled": bool(parent.evalParm("si_restart_enabled")),
                "alpha_threshold": parent.evalParm("si_alpha_threshold"),
                "patience": parent.evalParm("si_patience"),
                "min_iterations": parent.evalParm("si_min_iterations"),
                "soft_iteration_limit":
                    parent.evalParm("si_soft_iteration_limit"),
                "max_restarts": parent.evalParm("si_max_restarts"),
                "stall_trim_factor":
                    parent.evalParm("si_stall_trim_factor")}}
        # The experimental force-weighted band always runs with the EF-07
        # pair-basis downward guard (user decision 2026-09-28); without it the
        # band loops on contact-dense scenes.
        if contact["semi_implicit"]["band_statistic"] == "force_weighted":
            contact["semi_implicit"]["collapse_guard_basis"] = "pair"
        if parent.parm("si_clamped_contacts") is not None:
            clamped = _menu_token(parent, "si_clamped_contacts")
            if clamped != "exclude_statistics":
                contact["semi_implicit"]["clamped_contacts"] = clamped
        if parent.parm("si_gradient_balance_dofs") is not None:
            gb_dofs = _menu_token(parent, "si_gradient_balance_dofs")
            if gb_dofs != "all":
                contact["semi_implicit"]["gradient_balance_dofs"] = gb_dofs
    elif barrier_mode == 1:  # classic adaptive
        contact["barrier_stiffness"] = "adaptive"
        contact["initial_barrier_stiffness"] = \
            parent.evalParm("initial_barrier_stiffness")
        if parent.parm("dhat_epsilon_scale") is not None:
            eps_scale = float(parent.evalParm("dhat_epsilon_scale"))
            if not math.isclose(eps_scale, 1e-9, rel_tol=1e-9, abs_tol=1e-18):
                contact["dhat_epsilon_scale"] = eps_scale
    else:  # fixed
        contact["barrier_stiffness"] = parent.evalParm("barrier_stiffness")

    rayleigh = []
    form_dict = {0: "elasticity", 1: "contact", 2: "friction"}
    for i in range(1, parent.evalParm("num_rayleigh") + 1):
        rayleigh.append({
            "form": form_dict[parent.evalParm(f"rayleigh_form{i}")],
            "stiffness_ratio": parent.evalParm(f"stiffness_ratio{i}"),
            "lagging_iterations": parent.evalParm(f"rayleigh_lagging{i}")})

    advanced = {
        "cache_size": parent.evalParm("cache_size"),
        "lump_mass_matrix": bool(parent.evalParm("lump_mass_matrix")),
        "lagged_regularization_weight":
            parent.evalParm("lagged_regularization_weight"),
        "lagged_regularization_iterations":
            parent.evalParm("lagged_regularization_iterations"),
        "check_inversion": _menu_token(parent, "inversion_method"),
        "jacobian_threshold": parent.evalParm("jacobian_threshold")}
    # Generalized stall restart (D8) and line-search failure recovery:
    # emit only when non-default so untouched scenes keep identical JSON.
    if parent.parm("adv_stall_restart_enabled") is not None:
        sr_enabled = bool(parent.evalParm("adv_stall_restart_enabled"))
        sr_alpha = float(parent.evalParm("adv_stall_alpha_threshold"))
        sr_patience = int(parent.evalParm("adv_stall_patience"))
        sr_min_iters = int(parent.evalParm("adv_stall_min_iterations"))
        sr_soft_limit = int(parent.evalParm("adv_stall_soft_limit"))
        sr_max_restarts = int(parent.evalParm("adv_stall_max_restarts"))
        sr_basis = _menu_token(parent, "adv_stall_alpha_basis")
        sr_ratio = float(parent.evalParm("adv_stall_feasible_ratio"))
        sr_remedy = _menu_token(parent, "adv_stall_remedy")
        if (not sr_enabled
                or not math.isclose(sr_alpha, 0.01, rel_tol=1e-9, abs_tol=1e-12)
                or sr_patience != 5 or sr_min_iters != 5
                or sr_soft_limit != -1 or sr_max_restarts != 20
                or sr_basis != "feasible_bound"
                or not math.isclose(sr_ratio, 0.999, rel_tol=1e-9, abs_tol=1e-12)
                or sr_remedy != "force_psd_projection"):
            advanced["stall_restart"] = {
                "enabled": sr_enabled,
                "alpha_threshold": sr_alpha,
                "patience": sr_patience,
                "min_iterations": sr_min_iters,
                "soft_iteration_limit": sr_soft_limit,
                "max_restarts": sr_max_restarts,
                "alpha_basis": sr_basis,
                "feasible_ratio_threshold": sr_ratio,
                "remedy": sr_remedy}
    if parent.parm("ls_failure_restarts") is not None:
        ls_restarts = int(parent.evalParm("ls_failure_restarts"))
        if ls_restarts != 1:
            advanced["line_search_failure_restarts"] = ls_restarts

    solver = {"max_threads": parent.evalParm("max_threads"),
              "nonlinear": nonlinear,
              "augmented_lagrangian": al,
              "contact": contact,
              "advanced": advanced}
    linear = build_linear_solver(parent)
    if linear is not None:
        solver["linear"] = linear
    if rayleigh:
        solver["rayleigh_damping"] = rayleigh
    data["solver"] = solver


def hdf5_readable_here():
    """(True, how) when the Read PVD node can open VTK-HDF results on this
    computer, else (False, why not).

    It can with h5py installed in Houdini's Python, or with the h5py wheel
    the Read PVD asset bundles for this platform (macOS, Linux x86_64 with
    glibc 2.28+, Windows x64; Python 3.13). The environment variable
    POLYFEM_HDA_HDF5_READER=none makes it answer no (used by the tests).
    """
    if os.environ.get("POLYFEM_HDA_HDF5_READER", "").lower() == "none":
        return False, "POLYFEM_HDA_HDF5_READER=none is set"
    try:
        import importlib.util
        if importlib.util.find_spec("h5py") is not None:
            return True, "h5py is installed in Houdini's Python"
    except (ImportError, ValueError):
        pass
    node_type = hou.nodeType(hou.objNodeTypeCategory(), "readPVD::1.0")
    if node_type is None or node_type.definition() is None:
        # Results are not read in Houdini here (ParaView reads VTK-HDF).
        return True, "Read PVD is not installed; other viewers read VTK-HDF"
    sections = node_type.definition().sections()
    try:
        section, reason = node_type.hdaModule().embedded_h5py_section()
    except AttributeError:
        # A Read PVD from before 2026-10-01 bundles the Apple-silicon reader
        # only.
        section = "h5py-3.16.0-cp313-macos-arm64.whl.b64"
        reason = ("the installed Read PVD asset bundles an HDF5 reader for "
                  "Apple-silicon Macs only; install the current one")
        if not (sys.platform == "darwin" and sys.version_info[:2] == (3, 13)
                and platform.machine().lower() in ("arm64", "aarch64")):
            return False, reason
    if section is None:
        return False, reason
    if section not in sections:
        return False, "the installed Read PVD asset lacks this platform's reader"
    return True, "Read PVD bundles an HDF5 reader for this platform"


def _hdf5_note(parent):
    if not parent.evalParm("use_hdf5"):
        return None
    readable, reason = hdf5_readable_here()
    if readable:
        return None
    return ("Use HDF5 is on, but the Read PVD node cannot read HDF5 on this "
            f"computer ({reason}); the results will be written as XML .vtu "
            "files (larger, readable everywhere). Install h5py into "
            "Houdini's Python to get HDF5 output.")


def _condition_kinds_used(parent):
    """Boundary types (0..4) of every condition on every simulated body."""
    kinds = set()
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval():
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            count = parent.parm(f"sideset_selection{geo}_{vol}")
            for j in range(1, (count.eval() if count is not None else 0) + 1):
                for k in range(1, parent.evalParm(
                        f"Boundary_Condition__{geo}_{vol}_{j}") + 1):
                    kinds.add(int(parent.evalParm(
                        f"boundary_type{geo}_{vol}_{j}_{k}")))
    return kinds


def _material_damping_used(parent):
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if not parent.evalParm(f"is_enabled{geo}") \
                or parent.evalParm(f"is_obstacle{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            if _damping_active(parent, geo, vol):
                return True
    return False


def force_field_names(parent):
    """The nodal-force fields (PolyFEM's <form>_forces) of the forms this
    scene has, which Read PVD's force curves add up. PolyFEM writes zeros
    for a form a scene does not have, so those are not requested."""
    names = ["elastic_forces"]
    kinds = _condition_kinds_used(parent)
    try:
        gravity = _read_expr_vector(parent, "rhs_vec", "RHS", "rhs")
    except hou.Error:
        gravity = None
    no_gravity = isinstance(gravity, list) and all(
        isinstance(v, (int, float)) and v == 0 for v in gravity)
    if kinds & {1, 2, 5} or not no_gravity:
        names.append("body_forces")
    if kinds & {3, 4}:
        names.append("pressure_forces")
    if parent.evalParm("enable"):
        names.append("contact_forces")
        if parent.evalParm("cof") > 0:
            names.append("friction_forces")
    if not parent.evalParm("quasistatic"):
        names.append("inertia_forces")
    if _material_damping_used(parent):
        names.append("damping_forces")
    return names


def build_output(parent, data):
    def b(name):
        return bool(parent.evalParm(name))

    # Contact surface fields are only meaningful (and only written) when their
    # corresponding contact feature is enabled.
    contact_on = b("enable")
    adhesion_on = contact_on and b("adhesion_enable")
    gcp_adapt_on = contact_on and b("gcp_enable") and b("adapt_dhat")
    contact_forces = b("contact_forces_fields") and contact_on
    friction_forces = b("friction_forces_fields") and contact_on
    normal_adh = b("normal_adhesion_forces_fields") and adhesion_on
    tang_adh = b("tangential_adhesion_forces_fields") and adhesion_on
    adaptive_dhat = b("adaptive_dhat") and gcp_adapt_on
    surface = (b("normals_fields") or b("displaced_normals_fields")
               or b("sidesets_fields")
               or b("solution_grad_fields") or contact_forces
               or friction_forces or normal_adh or tang_adh
               or adaptive_dhat)
    jacobian_validity = b("validity_fields")
    volume = (b("velocity_fields") or b("acceleration_fields")
              or b("nodes_field") or b("forces_fields") or jacobian_validity)
    points = b("paraview_points") or jacobian_validity

    # Minimal-fields mode: whitelist only the derivation basis the readPVD
    # HDA needs (solution + F + Cauchy stress) plus whatever force/surface
    # arrays are enabled. Everything readPVD can derive on load (von Mises,
    # PK1/PK2, strains, invariants, and every *_avg variant) is omitted --
    # PolyFEM then skips computing them entirely, shrinking the vtu files.
    # The whitelist is global across the volume/surface/contact blocks.
    # Material-field export uses assembler-specific array names we cannot
    # enumerate safely, so that option falls back to full output.
    minimal = b("minimal_fields") and not b("materials_fields")
    export_fields = []
    if minimal:
        export_fields = ["solution"]
        if b("scalar_values") or b("tensor_values"):
            export_fields += ["F", "cauchy_stess"]  # [sic] PolyFEM spelling
        if b("normals_fields"):
            export_fields += ["normals", "displaced_normals"]
        elif b("displaced_normals_fields"):
            export_fields.append("displaced_normals")
        if b("sidesets_fields"):
            export_fields.append("sidesets")
        if b("solution_grad_fields"):
            export_fields.append("solution_gradient")
        for enabled, name in (
                (contact_forces, "contact_forces"),
                (friction_forces, "friction_forces"),
                (normal_adh, "normal_adhesion_forces"),
                (tang_adh, "tangential_adhesion_forces"),
                (adaptive_dhat, "adaptive_dhat"),
                (b("body_ids_fields"), "body_ids"),
                (b("velocity_fields"), "velocity"),
                (b("acceleration_fields"), "acceleration"),
                (b("discr_fields"), "discr"),
                (b("nodes_field"), "nodes"),
                (jacobian_validity, "validity")):
            if enabled:
                export_fields.append(name)
        # Nodal forces of every form the scene has (Read PVD's force
        # curves); without them in the whitelist PolyFEM wrote none.
        if b("forces_fields"):
            export_fields.extend(name for name in force_field_names(parent)
                                 if name not in export_fields)

    # XML when Read PVD could not open HDF5 here (Check Setup says so).
    use_hdf5 = b("use_hdf5") and hdf5_readable_here()[0]
    options = {
        "use_hdf5": use_hdf5, "material": b("materials_fields"),
        "body_ids": b("body_ids_fields"), "contact_forces": contact_forces,
        "friction_forces": friction_forces,
        "normal_adhesion_forces": normal_adh,
        "tangential_adhesion_forces": tang_adh,
        "velocity": b("velocity_fields"),
        "acceleration": b("acceleration_fields"),
        # the whitelist already excludes von_mises; forcing tensor on /
        # scalar off makes the F + cauchy basis independent of the toggles
        "scalar_values": False if minimal else b("scalar_values"),
        "tensor_values": True if minimal else b("tensor_values"),
        "discretization_order": b("discr_fields"),
        "nodes": b("nodes_field"), "forces": b("forces_fields"),
        "force_high_order": b("forces_HO_output"),
        "jacobian_validity": jacobian_validity}
    if parent.parm("obstacle_volume") is not None and not b("obstacle_volume"):
        options["obstacle_volume"] = False
    paraview = {
        "file_name": "../output/" + parent.evalParm("paraview_file_name"),
        "vismesh_rel_area": parent.evalParm("vismesh_rel_area"),
        "skip_frame": parent.evalParm("skip_frame"),
        "high_order_mesh": b("high_order_mesh"),
        "volume": volume or True,  # always write the volume mesh
        "surface": surface, "wireframe": b("paraview_wireframe"),
        "points": points, "options": options}
    if parent.parm("paraview_async") is not None and not b("paraview_async"):
        paraview["async"] = False
    if export_fields:
        paraview["fields"] = export_fields

    data_output = {}
    for parm, key, fname in (
            ("solution_file", "solution", "solution.txt"),
            ("stress_mat_file", "stress_mat", "stress_mat.txt"),
            ("state_file", "diagnostic_state", "state.txt"),
            ("mises_file", "mises", "mises.txt"),
            ("nodes", "nodes", "nodes.txt")):
        if b(parm):
            data_output[key] = "../output/" + fname

    advanced = {
        "timestep_prefix": parent.evalParm("timestep_prefix"),
        "sol_on_grid": parent.evalParm("sol_on_grid"),
        "compute_error": b("compute_error"),
        "sol_at_node": parent.evalParm("sol_at_node"),
        "vis_boundary_only": b("vis_boundary_only"),
        "curved_mesh_size": b("curved_mesh_size"),
        "save_solve_sequence_debug": b("save_solve_sequence_debug"),
        "save_ccd_debug_meshes": b("save_ccd_debug_meshes"),
        "save_time_sequence": b("save_time_sequence"),
        "save_nl_solve_sequence": b("save_nl_solve_sequence"),
        "spectrum": b("spectrum")}

    data["output"] = {"directory": "../output/", "paraview": paraview,
                      "data": data_output, "advanced": advanced}
    if b("output_json"):
        data["output"]["json"] = "../output/output.json"
    if b("restart_json"):
        data["output"]["checkpoint"] = {"path": RESTART_CHECKPOINT_FILE}
    log = {"level": parent.evalParm("log_level"),
           "quiet": b("log_quiet")}
    if b("write_log"):
        log["path"] = "../output/log.txt"
    data["output"]["log"] = log

    units_on = parent.evalParm("units")
    if units_on:
        data["units"] = {"length": parent.evalParm("length"),
                         "mass": parent.evalParm("mass"),
                         "time": parent.evalParm("time"),
                         "characteristic_length":
                             parent.evalParm("char_length")}


