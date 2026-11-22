def _damping_active(parent, geo, vol):
    """True only when the subdomain uses a direct material that PolyFEM's
    ViscousDamping supports and its Damping toggle is on."""
    if material_token(parent, geo, vol) not in _DAMPING_MODELS:
        return False
    damping_parm = parent.parm(f"damping{geo}_{vol}")
    return damping_parm is not None and bool(damping_parm.eval())


def _damping(parent, geo, vol, material):
    if _damping_active(parent, geo, vol):
        material["phi"] = parent.evalParm(f"phi{geo}_{vol}")
        material["psi"] = parent.evalParm(f"psi{geo}_{vol}")
    return material


def _read_growth_tensor(parent, geo, vol):
    """Read GrowthNeoHookean symmetric 3x3 natural-configuration entries."""
    diag_tuple = parent.parmTuple(f"growth_diag{geo}_{vol}")
    off_tuple = parent.parmTuple(f"growth_offdiag{geo}_{vol}")
    diag_raw = list(diag_tuple.eval()) if diag_tuple is not None \
        else ["1", "1", "1"]
    off_raw = list(off_tuple.eval()) if off_tuple is not None \
        else ["0", "0", "0"]
    out = {}
    for key, raw in zip(("g11", "g22", "g33"), diag_raw):
        text = str(raw).strip()
        out[key] = 1.0 if not text else _parse_expr_component(text, key)
    for key, raw in zip(("g12", "g13", "g23"), off_raw):
        text = str(raw).strip()
        out[key] = 0.0 if not text else _parse_expr_component(text, key)
    return out


def _check_growth_tensor(entries, where, resource_dir=None):
    """Validate GrowthNeoHookean G entries: TinyExpr syntax, plus Sylvester's
    positive-definiteness criterion when all 6 entries are plain numbers."""
    errors = []
    for key in ("g11", "g22", "g33", "g12", "g13", "g23"):
        val = entries[key]
        err = _validate_tinyexpr(val, f"{where} {key}", resource_dir)
        if err:
            errors.append(err)
    if errors:
        return errors
    if all(isinstance(entries[k], (int, float)) and not isinstance(entries[k], bool)
           for k in ("g11", "g22", "g33", "g12", "g13", "g23")):
        g11, g22, g33 = float(entries["g11"]), float(entries["g22"]), float(entries["g33"])
        g12, g13, g23 = float(entries["g12"]), float(entries["g13"]), float(entries["g23"])
        mat = np.array([[g11, g12, g13],
                        [g12, g22, g23],
                        [g13, g23, g33]], dtype=np.float64)
        m1 = g11
        m2 = g11 * g22 - g12 * g12
        m3 = float(np.linalg.det(mat))
        if not (np.all(np.isfinite(mat)) and m1 > 0.0 and m2 > 0.0 and m3 > 0.0):
            errors.append(
                f"{where}: natural-configuration tensor G must be symmetric "
                f"positive definite (leading principal minors are "
                f"{m1:g}, {m2:g}, {m3:g}; all three must be > 0).")
    return errors


_IDENTITY_MATERIAL_AXES = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def _material_axes(raw):
    try:
        axes = np.asarray(ast.literal_eval(raw), dtype=float)
        if axes.shape in ((9,), (3, 3)) and np.isfinite(axes).all():
            return axes.reshape(3, 3)
    except (SyntaxError, TypeError, ValueError, OverflowError):
        pass
    return None


def _material_axes_are_identity(raw):
    axes = _material_axes(raw)
    return axes is not None and np.array_equal(axes, np.eye(3))


def _hooke_axes(parent, geo, vol):
    axes = _material_axes(parent.evalParm(f"material_coord{geo}_{vol}"))
    if axes is None or not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-8) \
            or not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1e-8):
        raise hou.NodeError(
            f"{_slot_label(geo, vol, None)} (HookeLinearElasticity): Material "
            "Axes must be an orthonormal right-handed 3x3 rotation "
            "(identity: [1,0,0,0,1,0,0,0,1]).")
    return axes.ravel().tolist()


def _saint_venant_axes_message(parent, geo, vol):
    if _material_axes_are_identity(parent.evalParm(f"material_coord{geo}_{vol}")):
        return None
    return (f"{_slot_label(geo, vol, None)}: SaintVenant uses global "
            "coordinates and ignores fiber_direction (Material Axes) in "
            "PolyFEM. Reset Material Axes to identity [1,0,0,0,1,0,0,0,1]. "
            "To use rotated stiffness, supply a pre-rotated 21-entry "
            "elasticity_tensor with identity axes, or switch to "
            "HookeLinearElasticity.")


def _saint_venant_axes_findings(parent):
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.evalParm(f"is_obstacle{geo}") or not parent.evalParm(f"is_enabled{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            token = material_token(parent, geo, vol)
            if token == "MaterialSum":
                token = parent.parm(f"matrix_model{geo}_{vol}").evalAsString()
            if token == "SaintVenant":
                note = _saint_venant_axes_message(parent, geo, vol)
                if note:
                    yield note


def warn_saint_venant_axes(parent):
    notes = list(_saint_venant_axes_findings(parent))
    if notes:
        _status(f"{parent.path()}: " + " ".join(notes))


def _raise_saint_venant_axes(parent):
    notes = list(_saint_venant_axes_findings(parent))
    if notes:
        raise hou.NodeError("\n".join(notes))


def _isotropic_model(parent, geo, vol, token, allow_damping=True):
    """Body of an isotropic model: no id/rho, so it also serves as a
    MaterialSum child (children carry neither)."""
    def ep():
        return (parent.evalParm(f"E{geo}_{vol}"),
                parent.evalParm(f"nu{geo}_{vol}"))

    if token == "IncompressibleLinearElasticity":
        raise hou.NodeError(
            f"{_slot_label(geo, vol, None)}: {INCOMPRESSIBLE_LINEAR_ERROR}")
    if token in ("LinearElasticity", "NeoHookean", "FixedCorotational",
                 "IsochoricNeoHookean"):
        E, nu = ep()
        model = {"type": token, "E": E, "nu": nu}
        if allow_damping:
            _damping(parent, geo, vol, model)
        return model
    if token == "GrowthNeoHookean":
        E, nu = ep()
        entries = _read_growth_tensor(parent, geo, vol)
        errs = _check_growth_tensor(entries, _slot_label(geo, vol, None),
                                    os.path.join(parent.evalParm("working_dir"), "input"))
        if errs:
            raise hou.NodeError(errs[0])
        model = {"type": token, "E": E, "nu": nu}
        model.update(entries)
        return model
    if token == "InversionBarrier":
        E, nu = ep()
        thresh = parent.evalParm(f"inv_barrier_thresh{geo}_{vol}")
        if not thresh > 0:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: JBarrierThreshold must be "
                f"greater than 0 (got {thresh:g}).")
        return {"type": token, "E": E, "nu": nu,
                "JBarrierThreshold": float(thresh)}
    if token == "AMIPS":
        weight = parent.evalParm(f"amips_weight{geo}_{vol}")
        if not weight > 0:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: AMIPS weight must be greater "
                f"than 0 (got {weight:g}).")
        model = {"type": token, "weight": float(weight),
                 "use_rest_pose": bool(
                     parent.evalParm(f"amips_use_rest_pose{geo}_{vol}"))}
        if parent.parm(f"amips_power{geo}_{vol}") is not None:
            power = float(parent.evalParm(f"amips_power{geo}_{vol}"))
            if not power > 0:
                raise hou.NodeError(
                    f"{_slot_label(geo, vol, None)}: AMIPS power must be "
                    f"greater than 0 (got {power:g}).")
            if not math.isclose(power, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                model["power"] = power
        if parent.parm(f"amips_normalized{geo}_{vol}") is not None \
                and bool(parent.evalParm(f"amips_normalized{geo}_{vol}")):
            model["normalized"] = True
        return model
    if token in ("HookeLinearElasticity", "SaintVenant"):
        if token == "SaintVenant" and parent.evalParm(f"is_enabled{geo}"):
            note = _saint_venant_axes_message(parent, geo, vol)
            if note:
                raise hou.NodeError(note)
        model = {"type": token,
                 "elasticity_tensor": _read_elast_tensor(parent, geo, vol)}
        if token == "HookeLinearElasticity":
            if parent.evalParm(f"is_enabled{geo}"):
                model["fiber_direction"] = _hooke_axes(parent, geo, vol)
            else:
                model["fiber_direction"] = parse_vector(
                    parent.evalParm(f"material_coord{geo}_{vol}"), "fiber_direction")
        elif allow_damping:
            _damping(parent, geo, vol, model)
        return model
    if token == "IncompressibleOgden":
        return {"type": token,
                "c": parse_vector(parent.evalParm(f"ogdenC{geo}_{vol}"), "c"),
                "m": parse_vector(parent.evalParm(f"ogdenM{geo}_{vol}"), "m"),
                "k": parent.evalParm(f"bulk{geo}_{vol}")}
    if token == "UnconstrainedOgden":
        return {"type": token,
                "alphas": parse_vector(
                    parent.evalParm(f"ogden_alphas{geo}_{vol}"), "alphas"),
                "mus": parse_vector(
                    parent.evalParm(f"ogden_mus{geo}_{vol}"), "mus"),
                "Ds": parse_vector(
                    parent.evalParm(f"ogden_ds{geo}_{vol}"), "Ds")}
    if token == "MooneyRivlin":
        return {"type": token,
                "c1": parent.evalParm(f"c1{geo}_{vol}"),
                "c2": parent.evalParm(f"c2{geo}_{vol}"),
                "k": parent.evalParm(f"bulk{geo}_{vol}")}
    if token in ("MooneyRivlin3Param", "MooneyRivlin3ParamSymbolic"):
        return {"type": token,
                "c1": parent.evalParm(f"c1{geo}_{vol}"),
                "c2": parent.evalParm(f"c2{geo}_{vol}"),
                "c3": parent.evalParm(f"c3{geo}_{vol}"),
                "d1": parent.evalParm(f"d1{geo}_{vol}")}
    raise hou.NodeError(f"Unsupported isotropic material '{token}'")


def _number_or_expression(text, label):
    """A polyfem scalar the user typed: number when it parses as one,
    otherwise the string (an expression of x,y,z,t)."""
    text = str(text).strip()
    if not text:
        raise hou.NodeError(f"{label} is empty")
    try:
        return float(text)
    except ValueError:
        return text


def fiber_direction_json(parent, geo, vol, fam, side=""):
    """"fiber_direction" for one family: constant, expression, or a reference
    into the shared per-element file."""
    source = _slot_parm(parent, geo, vol, fam, "fib_source")
    if source in PER_ELEMENT_FIBER_SOURCES or side:
        return {"type": "per_element_file", "path": FIBER_FILE,
                "field": fiber_field(geo, vol, fam, side)}
    if source == "expression":
        return parse_vector(
            _slot_parm(parent, geo, vol, fam, "fib_expr"), "fiber_direction")
    return list(parent.parmTuple(_parm_name(geo, vol, fam, "fib_dir")).eval())


def scalar_json(parent, geo, vol, fam, name):
    """A scalar material parameter: number, expression, or per-element file."""
    source = _slot_parm(parent, geo, vol, fam, f"{name}_source")
    if source in ("attribute", "file"):
        return per_element_scalar_file(geo, vol, fam, name)
    if source == "expression":
        return _number_or_expression(
            _slot_parm(parent, geo, vol, fam, f"{name}_expr"),
            f"{_slot_label(geo, vol, fam)} {name}")
    return _slot_parm(parent, geo, vol, fam, name)


def _fiber_model(parent, geo, vol, fam, token, side=""):
    """One fiber family as a polyfem model object (no id/rho)."""
    model = {"type": token,
             "fiber_direction": fiber_direction_json(parent, geo, vol, fam, side)}
    if token == "ActiveFiber":
        model["Tmax"] = _slot_parm(parent, geo, vol, fam, "Tmax")
        model["activation"] = _number_or_expression(
            _slot_parm(parent, geo, vol, fam, "activation"),
            f"{_slot_label(geo, vol, fam)} activation")
        return model
    k2 = _slot_parm(parent, geo, vol, fam, "k2")
    if k2 <= 0:
        raise hou.NodeError(
            f"{_slot_label(geo, vol, fam)}: k2 must be greater than zero "
            f"(the model divides by it)")
    model["k1"] = _slot_parm(parent, geo, vol, fam, "k1")
    model["k2"] = k2
    if token == "HGODispersion":
        model["kappa"] = scalar_json(parent, geo, vol, fam, "kappa")
        k_chi = _slot_parm(parent, geo, vol, fam, "k_chi")
        if abs(k_chi - 100.0) > 1e-9:
            model["k_chi"] = k_chi
    return model


def composite_models(parent, geo, vol):
    """The ordered "models" list of a composite, families expanded.

    A symmetric family becomes two children sharing every parameter but their
    rotated directions, which is exactly what polyfem sums.
    """
    models = []
    matrix = parent.parm(f"matrix_model{geo}_{vol}").evalAsString()
    if matrix != "None":
        models.append(_isotropic_model(parent, geo, vol, matrix,
                                       allow_damping=False))
    for fam, token in material_families(parent, geo, vol):
        for side in mirror_sides(parent, geo, vol, fam):
            models.append(_fiber_model(parent, geo, vol, fam, token, side))
    if parent.evalParm(f"volume_penalty{geo}_{vol}"):
        models.append({"type": "VolumePenalty",
                       "k": parent.evalParm(f"vp_k{geo}_{vol}")})
    if parent.parm(f"add_inv_barrier{geo}_{vol}") is not None \
            and parent.evalParm(f"add_inv_barrier{geo}_{vol}"):
        inv_E = parent.evalParm(f"comp_inv_E{geo}_{vol}")
        inv_nu = parent.evalParm(f"comp_inv_nu{geo}_{vol}")
        inv_thresh = parent.evalParm(f"comp_inv_thresh{geo}_{vol}")
        if not inv_thresh > 0:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: composite InversionBarrier "
                f"JBarrierThreshold must be greater than 0 (got {inv_thresh:g}).")
        models.append({"type": "InversionBarrier",
                       "E": inv_E, "nu": inv_nu,
                       "JBarrierThreshold": float(inv_thresh)})
    extra = parent.evalParm(f"extra_models_json{geo}_{vol}").strip()
    if extra:
        try:
            parsed = json.loads(extra)
        except ValueError as error:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: extra models are not valid "
                f"JSON ({error})")
        if not isinstance(parsed, list):
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: extra models must be a JSON "
                f"list of model objects")
        for index, model in enumerate(parsed, 1):
            if not isinstance(model, dict) or not isinstance(
                    model.get("type"), str) or not model["type"].strip():
                raise hou.NodeError(
                    f"{_slot_label(geo, vol, None)}: Extra Models entry "
                    f'{index} must be an object with a non-empty "type".')
        models.extend(parsed)
    return models


def build_material(parent, geo, vol):
    token = material_token(parent, geo, vol)
    material = {"id": vol_id(geo, vol), "rho": parent.evalParm(f"rho{geo}_{vol}")}

    if token == "MaterialSum":
        models = composite_models(parent, geo, vol)
        if not models:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: a composite material needs a "
                f"matrix model or at least one fiber family")
        material["type"] = "MaterialSum"
        material["models"] = models
        return material
    if token in FIBER_MODELS:
        material.update(_fiber_model(parent, geo, vol, None, token))
        return material
    material.update(_isotropic_model(parent, geo, vol, token))
    return material


class _SharedMaterialError(hou.NodeError):
    """An incompatible set of bodies sharing a solver assembler."""


def _plan_body_materials(parent):
    """Validate shared assemblers and plan the global fiber columns they use."""
    materials, owned_fields = {}, {}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.evalParm(f"is_obstacle{geo}") or not parent.evalParm(f"is_enabled{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            body = (geo, vol)
            materials[body] = build_material(parent, *body)
            owned_fields[body] = {
                fiber_field(geo, vol, fam, side)
                for fam, model in material_families(parent, geo, vol)
                for side in mirror_sides(parent, geo, vol, fam)
                if model in ("HGODispersion", "HGOFiber") and
                (_slot_parm(parent, geo, vol, fam, "fib_source")
                 in PER_ELEMENT_FIBER_SOURCES or side)}

    tensors, fibers = {}, {}

    def collect(body, material, slot=()):
        token = material.get("type")
        if token == "MaterialSum":
            for index, child in enumerate(material.get("models", []), 1):
                if isinstance(child, dict):
                    collect(body, child, (*slot, index))
        elif token in _TENSOR_MODELS:
            tensor = material.get("elasticity_tensor")
            if tensor is None or tensor == []:
                tensor = _young_poisson(material)
            tensors.setdefault(token, []).append((body, tensor))
        elif token in ("HGODispersion", "HGOFiber"):
            fibers.setdefault((token, slot), []).append((body, material))

    for body, material in materials.items():
        collect(body, material)

    for token, entries in tensors.items():
        bodies = sorted({body for body, _ in entries})
        if len(bodies) < 2:
            continue
        first = entries[0][1]
        if any(not isinstance(tensor, (list, tuple)) or
               not all(type(value) in (int, float) and math.isfinite(value)
                       for value in tensor) or tuple(tensor) != tuple(first)
               for _, tensor in entries):
            labels = ", ".join(_slot_label(*body, None) for body in bodies)
            raise _SharedMaterialError(
                f"{labels}: PolyFEM shares one elasticity tensor per "
                f"anisotropic model. Every body using {token} must use "
                "the same finite numeric elasticity tensor.")

    shared = {}
    for (token, slot), entries in fibers.items():
        staged, external, constants, expressions = {}, {}, {}, []
        for body, material in entries:
            direction = material.get("fiber_direction")
            if isinstance(direction, dict) and direction.get("type") == "per_element_file":
                name, path = direction.get("field", "FIB_DIR1"), direction.get("path")
                if path == FIBER_FILE and name in owned_fields[body]:
                    staged[body] = name
                else:
                    external[body] = (path, name)
            elif isinstance(direction, list) and len(direction) == 3 and all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in direction):
                constants[body] = direction
            else:
                expressions.append(body)

        if not staged and not external:
            continue
        labels = ", ".join(_slot_label(*body, None) for body, _ in entries)
        where = token + (" MaterialSum slot " + ".".join(map(str, slot)) if slot else "")
        if expressions and (staged or len(entries) > 1):
            raise _SharedMaterialError(
                f"{labels}: {where} cannot mix a per-element fiber file "
                "with a spatially or time-varying fiber expression.")
        if external and (staged or constants or
                         any(ref != next(iter(external.values())) for ref in external.values())):
            raise _SharedMaterialError(
                f"{labels}: {where} has incompatible external per-element "
                "fiber files. Use the HDA Fiber File source to stage the "
                "directions, or use the same external file and field for every body.")
        if not staged:
            continue
        name = next(iter(staged.values()))
        sources = shared.setdefault(name, dict(staged))
        for body, direction in constants.items():
            sources[body] = _normalize_fibers(
                np.asarray([direction], dtype=float), _slot_label(*body, None))[0]
        for _, material in entries:
            material["fiber_direction"] = {
                "type": "per_element_file", "path": FIBER_FILE, "field": name}
    return materials, shared


# PolyFEM formats {:d} with the output step index.
LEGACY_RESTART_STATE_FILE = "../output/state_{:d}.hdf5"
RESTART_CHECKPOINT_FILE = "../output/checkpoint_{:d}.hdf5"
RESTART_STATE_FILE = LEGACY_RESTART_STATE_FILE  # alias for older tests/callers


def build_time(parent, data):
    quasistatic = bool(parent.evalParm("quasistatic"))
    t0 = parent.evalParm("t0")
    integrator = parent.evalParm("integrator")
    integrator_dict = {0: "ImplicitEuler", 1: "BDF1", 2: "BDF2", 3: "BDF3",
                       4: "BDF4", 5: "BDF5", 6: "BDF6", 7: "ImplicitNewmark"}
    if integrator == 7:
        integrator = {"type": "ImplicitNewmark",
                      "beta": parent.evalParm("newmark_beta"),
                      "gamma": parent.evalParm("newmark_gamma")}
    else:
        integrator = integrator_dict[integrator]

    time = {"t0": t0, "integrator": integrator, "quasistatic": quasistatic}
    tend_on, dt_on, steps_on = _time_flags(parent)
    if int(tend_on) + int(dt_on) + int(steps_on) < 2:
        raise hou.NodeError(
            "The Time tab does not define a run: turn on two of End Time, "
            "Time Increment and # of Timesteps.")
    if tend_on:
        time["tend"] = parent.evalParm("tend")
    if dt_on:
        time["dt"] = parent.evalParm("dt")
    if steps_on and not (tend_on and dt_on):
        time["time_steps"] = int(parent.evalParm("num_timesteps"))
    data["time"] = time


def build_contact(parent, data):
    if not parent.evalParm("enable"):
        data["contact"] = {"enabled": False}
        return
    area_weighted = parent.evalParm("area_weighted")

    adhesion = {
        "adhesion_enabled": bool(parent.evalParm("adhesion_enable")),
        "dhat_p": parent.evalParm("dhat_p"),
        "dhat_a": parent.evalParm("dhat_a"),
        "adhesion_strength": parent.evalParm("adhesion_strength"),
        "tangential_adhesion_coefficient": parent.evalParm("tangent_coeff"),
        "epsa": parent.evalParm("adhesion_epsa")}
    data["contact"] = {
        "enabled": True,
        "dhat": parent.evalParm("dhat"),
        "epsv": parent.evalParm("epsv"),
        "friction_coefficient": parent.evalParm("cof"),
        "use_convergent_formulation": bool(area_weighted),
        "use_gcp_formulation": bool(parent.evalParm("gcp_enable")),
        "adhesion": adhesion}
    data["contact"].update(convergent_flags(parent, area_weighted))
    if parent.evalParm("gcp_enable"):
        # Read only by the smooth (GCP) contact form; the IPC barrier form
        # never sees them, so the file carries them only when they apply.
        data["contact"].update({
            "alpha_n": parent.evalParm("alpha_n"),
            "alpha_t": parent.evalParm("alpha_t"),
            "use_adaptive_dhat": bool(parent.evalParm("adapt_dhat")),
            "min_distance_ratio": parent.evalParm("min_dist_ratio")})


IMPROVED_MAX_STENCIL_ERROR = (
    "Remove Duplicate Contacts (Improved Max) needs Coefficient Identity = "
    "Parent Candidate (Contact tab, Semi-Implicit Options): only the parent "
    "law gives the duplicate-removal terms their own stiffness; PolyFEM "
    "refuses the Built Stencil identity with it.")


def convergent_flags(parent, area_weighted):
    """The convergent-formulation switches of the IPC barrier form.

    Adaptive and Fixed modes keep the historical export (use_convergent_
    formulation alone: PolyFEM's defaults then turn on area weighting, the
    improved max operator and the physical barrier). Semi-Implicit mode does
    not support the physical barrier, so it writes every switch: area
    weighting from Area Weighted, the improved max operator from Remove
    Duplicate Contacts (the signed parent law), the physical barrier off.
    GCP does not use the barrier form; nothing extra is written for it.
    """
    if parent.evalParm("gcp_enable") or parent.evalParm("barrier_mode") != 0:
        return {}
    improved_max = bool(parent.evalParm("improved_max"))
    if not area_weighted and not improved_max:
        return {}
    if improved_max and \
            _menu_token(parent, "si_coefficient_identity") == "stencil":
        raise hou.NodeError(IMPROVED_MAX_STENCIL_ERROR)
    return {"use_convergent_formulation": True,
            "use_area_weighting": bool(area_weighted),
            "use_improved_max_operator": improved_max,
            "use_physical_barrier": False}


def obstacle_displacement_entry(parent, geo, motion):
    """The obstacle_displacements entry of one obstacle (None for a mesh
    sequence, which carries its own motion)."""
    if motion["kind"] == "sequence":
        return None
    if motion["kind"] == "translation":
        tables = [TimeCurve(*_compressed_table(motion["times"],
                                               motion["displacement"][:, d]))
                  for d in range(3)]
        return {"id": obstacle_id(geo), "value": [1, 1, 1],
                "interpolation": [table.as_json() for table in tables]}
    vector = _read_expr_vector(
        parent, f"obstacle_disp_vec{geo}", f"obstacle_disp{geo}",
        f"obstacle {geo}")
    return {"id": obstacle_id(geo), "value": vector}


def build_conditions(parent, data, plans=None, motions=None):
    """Initial and boundary conditions.

    Sideset conditions come from the resolved sideset plans (one boundary id
    per sideset or per group of overlapping sidesets, conditions merged; see
    resolve_sidesets). export_sidesets stores the plans while writing the
    selection files; without them they are resolved here.
    """
    solution, velocity, acceleration = [], [], []
    lists = {key: [] for key in _BC_LIST_KEYS}
    obstacle = []
    num_geos = parent.evalParm("num_geos")

    for geo in range(1, num_geos + 1):
        is_obstacle = bool(parent.parm(f"is_obstacle{geo}").eval())
        if is_obstacle:
            motion = (motions or {}).get(geo)
            if motion is None:
                motion = obstacle_motion(parent, geo, timeline(parent)) \
                    if transform_is_animated(parent, geo) \
                    else {"kind": "static"}
            entry = obstacle_displacement_entry(parent, geo, motion)
            if entry is not None:
                obstacle.append(entry)
            continue
        num_vols = parent.evalParm(f"num_volumes{geo}")
        for vol in range(1, num_vols + 1):
            for l in range(1, parent.evalParm(
                    f"initial_conditions{geo}_{vol}") + 1):
                vector = _read_expr_vector(
                    parent, f"ic_vec{geo}_{vol}_{l}",
                    f"condition_vector_{geo}_{vol}_{l}",
                    f"initial condition {geo}/{vol}/{l}")
                ctype = parent.evalParm(f"conditiontype{geo}_{vol}_{l}")
                target = (solution, velocity, acceleration)[ctype]
                target.append({"id": vol_id(geo, vol), "value": vector})
        plan = (plans or {}).get(geo)
        if plan is None:
            plan = resolve_sidesets(parent, geo, num_vols)
        for sid in sorted(plan["conditions"]):
            for key, entry in plan["conditions"][sid]:
                lists[key].append(dict({"id": int(sid)}, **entry))

    if solution or velocity or acceleration:
        data["initial_conditions"] = {}
        for key, lst in (("solution", solution), ("velocity", velocity),
                         ("acceleration", acceleration)):
            if lst:
                data["initial_conditions"][key] = lst

    rhs = _read_expr_vector(parent, "rhs_vec", "RHS", "rhs")
    bc = {}
    for key, lst in [(key, lists[key]) for key in _BC_LIST_KEYS] \
            + [("obstacle_displacements", obstacle)]:
        if lst:
            bc[key] = lst
    if rhs is not None:
        bc["rhs"] = rhs
    if bc:
        data["boundary_conditions"] = bc


REMESH_DEFAULTS = {
    "enabled": False,
    "type": "physics",
    "split": {
        "enabled": True, "acceptance_tolerance": 1e-3,
        "culling_threshold": 0.95, "max_depth": 3,
        "min_edge_length": 1e-6},
    "collapse": {
        "enabled": True, "acceptance_tolerance": -1e-8,
        "culling_threshold": 0.01, "max_depth": 3,
        "rel_max_edge_length": 1.0, "abs_max_edge_length": 1e100},
    "swap": {
        "enabled": False, "acceptance_tolerance": -1e-8, "max_depth": 3},
    "smooth": {
        "enabled": False, "acceptance_tolerance": -1e-8, "max_iters": 1},
    "local_relaxation": {
        "local_mesh_n_ring": 2, "local_mesh_rel_area": 0.01,
        "max_nl_iterations": 1},
}


LEGACY_REMESH_ERROR = (
    "Legacy in-timestep adaptive remeshing is no longer supported. These "
    "controls do not configure MMG shape-optimization remeshing. Disable "
    "legacy remeshing to export a forward simulation.")


