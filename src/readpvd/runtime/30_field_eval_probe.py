def _mesh_point_field(mesh, field):
    """A field of a parsed mesh by its attribute name (cell data averaged to
    the points). `field` is the sanitized name the menus and the cooked
    geometry use ("MaterialSum/HGODispersion/kappa" is
    MaterialSum_HGODispersion_kappa), so raw vtu names are sanitized the
    same way before they are compared."""
    for raw_name, data in mesh["point_data"].items():
        if _field_name(raw_name) == field:
            array = np.asarray(data, dtype=np.float64)
            n_points = len(mesh["points"])
            if len(array) == n_points:
                return array
            # Match cook_frame: fields that omit obstacle vertices (e.g. *_avg)
            # are zero-padded to the full mesh point count.
            if 0 < len(array) < n_points:
                padded = np.zeros((n_points,) + array.shape[1:],
                                  dtype=np.float64)
                padded[:len(array)] = array
                return padded
    for raw_name, by_family in mesh.get("cell_data", {}).items():
        if _field_name(raw_name) != field:
            continue
        sample = next(iter(by_family.values()), None)
        if sample is None:
            return None
        sample = np.asarray(sample)
        shape = () if sample.ndim == 1 else (sample.shape[1],)
        total = np.zeros((len(mesh["points"]),) + shape, dtype=np.float64)
        count = np.zeros(len(mesh["points"]), dtype=np.float64)
        for family, values in by_family.items():
            conn = mesh["cells"].get(family)
            if conn is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            for corner in range(conn.shape[1]):
                np.add.at(total, conn[:, corner], values)
                np.add.at(count, conn[:, corner], 1.0)
        divisor = np.maximum(count, 1.0)
        return total / (divisor[:, None] if total.ndim > 1 else divisor)
    return None


def _finite_matrix_rows(operation, matrices):
    """Run LAPACK on finite rows only and retain missing tuples as NaN."""
    finite = np.isfinite(matrices).all(axis=(1, 2))
    if finite.all():
        return operation(matrices)
    result = operation(matrices[finite])

    def expand(array):
        full = np.full((len(matrices),) + array.shape[1:], np.nan)
        full[finite] = array
        return full

    if isinstance(result, tuple):
        return tuple(expand(array) for array in result)
    return expand(result)


def _reduce_array(values, reduction):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim == 3 or (values.ndim == 2 and values.shape[1] == 9):
        matrices = values.reshape(-1, 3, 3)
        if reduction == "trace":
            return np.trace(matrices, axis1=1, axis2=2)
        if reduction == "determinant":
            return _finite_matrix_rows(np.linalg.det, matrices)
        if reduction in ("principal_max", "principal_middle", "principal_min"):
            eigenvalues = _finite_matrix_rows(np.linalg.eigvalsh,
                0.5 * (matrices + np.swapaxes(matrices, 1, 2)))[:, ::-1]
            return eigenvalues[:, {
                "principal_max": 0, "principal_middle": 1,
                "principal_min": 2}[reduction]]
        if reduction in ("x", "y", "z"):
            index = {"x": 0, "y": 1, "z": 2}[reduction]
            return matrices[:, index, index]
        if reduction in ("xy", "yz", "xz"):
            row, col = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}[reduction]
            return matrices[:, row, col]
        return np.linalg.norm(matrices, axis=(1, 2))
    if reduction in ("principal_max", "principal_middle", "principal_min"):
        return values[:, {
            "principal_max": 0, "principal_middle": 1,
            "principal_min": 2}[reduction]]
    if reduction == "z" and values.shape[1] < 3:
        return np.zeros(len(values), dtype=np.float64)
    if reduction in ("x", "y", "z"):
        index = {"x": 0, "y": 1, "z": 2}[reduction]
        return values[:, min(index, values.shape[1] - 1)]
    return np.linalg.norm(values, axis=1)


def _mesh_field(mesh, field):
    """One field of a parsed mesh: exported, or derived exactly as the derived
    VEX stage does it. Only what the requested field needs is computed (the
    full derived set costs about 1 KB per point), so ranging a large result
    by von Mises does not also decompose every strain tensor."""
    raw = _mesh_point_field(mesh, field)
    if raw is not None:
        return raw
    names = {FIELD_ALIASES.get(name, name): name
             for name in mesh["point_data"]}

    def array(name):
        return np.asarray(mesh["point_data"][names[name]], dtype=np.float64)

    def have(*wanted):
        return all(name in names for name in wanted)

    if field == "solution_mag":
        return np.linalg.norm(array("solution"), axis=1) \
            if have("solution") else None
    if not have("F_1", "F_2", "F_3"):
        return None

    memo = {}

    def once(function):
        def value():
            if function.__name__ not in memo:
                memo[function.__name__] = function()
            return memo[function.__name__]
        return value

    identity = np.eye(3)

    # PolyFEM flattens tensors column-major: the X_i arrays are the COLUMNS
    # of the tensor (stack on axis=2), not the rows.
    @once
    def F():
        return np.stack((array("F_1"), array("F_2"), array("F_3")), axis=2)

    @once
    def jacobian():
        return _finite_matrix_rows(np.linalg.det, F())

    # Right/Left Cauchy-Green deformation tensors and their principal
    # (Lagrangian / Eulerian) axes, reused for the stretch and strain measures
    # that share those axes.
    @once
    def C():
        return np.einsum("nji,njk->nik", F(), F())   # F^T F (Lagrangian)

    @once
    def B():
        return np.einsum("nij,nkj->nik", F(), F())   # F F^T (Eulerian)

    @once
    def c_eigen():
        return _finite_matrix_rows(np.linalg.eigh, C())                    # ascending

    @once
    def b_eigen():
        return _finite_matrix_rows(np.linalg.eigh, B())

    @once
    def lam():
        return np.sqrt(np.clip(c_eigen()[0], 0, None))   # principal stretches

    @once
    def green():
        return 0.5 * (C() - identity)

    @once
    def hencky():
        vecs = c_eigen()[1]
        value = np.einsum("nij,nj,nkj->nik", vecs,
                          np.log(np.clip(lam(), 1e-20, None)), vecs)
        value[np.abs(jacobian()) <= 1e-12] = 0.0   # ln(U) undefined at J = 0
        return value

    # Almansi strain needs B^-1, defined only where B is invertible (J != 0);
    # zero it elsewhere so it matches the guarded VEX display path.
    @once
    def almansi():
        value = 0.5 * (identity - _finite_matrix_rows(np.linalg.pinv, B()))
        value[np.abs(jacobian()) <= 1e-12] = 0.0
        return value

    @once
    def infinitesimal():
        return 0.5 * (F() + np.swapaxes(F(), 1, 2)) - identity

    # Cauchy stress from the exported field, or reconstructed from the 1st
    # Piola-Kirchhoff stress (sigma = (1/J) P F^T) when only that was written.
    @once
    def cauchy():
        if have("cauchy_stress_1", "cauchy_stress_2", "cauchy_stress_3"):
            return np.stack((array("cauchy_stress_1"),
                             array("cauchy_stress_2"),
                             array("cauchy_stress_3")), axis=2)
        if have("pk1_stress_1", "pk1_stress_2", "pk1_stress_3"):
            pk1_tensor = np.stack((array("pk1_stress_1"),
                                   array("pk1_stress_2"),
                                   array("pk1_stress_3")), axis=2)
            jac = jacobian()
            scale = np.divide(1.0, jac, out=np.zeros_like(jac),
                              where=np.abs(jac) > 1e-12)
            return scale[:, None, None] * np.matmul(
                pk1_tensor, np.swapaxes(F(), 1, 2))
        return None

    @once
    def trace():
        return np.trace(cauchy(), axis1=1, axis2=2)

    @once
    def hydro():
        return trace() / 3.0

    @once
    def dev():
        return cauchy() - hydro()[:, None, None] * identity

    @once
    def j2():
        return 0.5 * np.sum(dev() * dev(), axis=(1, 2))

    @once
    def eigen():
        return _finite_matrix_rows(np.linalg.eigvalsh, cauchy())[:, ::-1]

    @once
    def Finv():
        return _finite_matrix_rows(np.linalg.pinv, F())

    @once
    def pk1():
        jac = jacobian()
        valid = np.abs(jac) > 1e-12
        out = np.zeros_like(cauchy())
        out[~np.isfinite(jac)] = np.nan
        if np.any(valid):
            out[valid] = jac[valid, None, None] * np.einsum(
                "nij,njk->nik", cauchy()[valid],
                np.swapaxes(Finv()[valid], 1, 2))
        return out

    @once
    def pk2():
        jac = jacobian()
        valid = np.abs(jac) > 1e-12
        out = np.zeros_like(cauchy())
        out[~np.isfinite(jac)] = np.nan
        if np.any(valid):
            out[valid] = jac[valid, None, None] * np.einsum(
                "nij,njk,nlk->nil", Finv()[valid], cauchy()[valid],
                Finv()[valid])
        return out

    kinematics = {
        "F_mat": F, "J": jacobian,
        "right_cauchy_green": C,
        "right_cauchy_green_eigenvalues": lambda: c_eigen()[0][:, ::-1],
        "left_cauchy_green": B,
        "left_cauchy_green_eigenvalues": lambda: b_eigen()[0][:, ::-1],
        "right_stretch": lambda: np.einsum(
            "nij,nj,nkj->nik", c_eigen()[1], lam(), c_eigen()[1]),
        "left_stretch": lambda: np.einsum(
            "nij,nj,nkj->nik", b_eigen()[1],
            np.sqrt(np.clip(b_eigen()[0], 0, None)), b_eigen()[1]),
        "principal_stretches": lambda: lam()[:, ::-1],
        "green_lagrange_strain": green,
        "green_lagrange_eigenvalues":
            lambda: _finite_matrix_rows(np.linalg.eigvalsh, green())[:, ::-1],
        "almansi_strain": almansi,
        "almansi_strain_eigenvalues":
            lambda: _finite_matrix_rows(np.linalg.eigvalsh, almansi())[:, ::-1],
        "hencky_strain": hencky,
        "hencky_strain_eigenvalues":
            lambda: _finite_matrix_rows(np.linalg.eigvalsh, hencky())[:, ::-1],
        "infinitesimal_strain": infinitesimal,
        "infinitesimal_strain_eigenvalues":
            lambda: _finite_matrix_rows(np.linalg.eigvalsh, infinitesimal())[:, ::-1],
    }
    stresses = {
        "cauchy_mat": cauchy, "cauchy_eigenvalues": eigen,
        "cauchy_trace": trace, "hydrostatic_stress": hydro,
        "deviatoric_stress": dev, "stress_J2": j2,
        "stress_J3": lambda: _finite_matrix_rows(np.linalg.det, dev()),
        "von_mises": lambda: np.sqrt(np.maximum(0, 3 * j2())),
        "von_mises_derived": lambda: np.sqrt(np.maximum(0, 3 * j2())),
        "max_shear_stress": lambda: 0.5 * (eigen()[:, 0] - eigen()[:, 2]),
        "stress_triaxiality": lambda: np.divide(
            hydro(), np.sqrt(np.maximum(0, 3 * j2())),
            out=np.zeros_like(hydro()), where=j2() > 1e-24),
        "pk1": pk1, "pk2": pk2,
        "pk2_eigenvalues": lambda: _finite_matrix_rows(np.linalg.eigvalsh,
            0.5 * (pk2() + np.swapaxes(pk2(), 1, 2)))[:, ::-1],
    }
    if field in kinematics:
        return kinematics[field]()
    if field in stresses:
        return stresses[field]() if cauchy() is not None else None
    return None


def cook_reference_comparison(node):
    geo = node.geometry()
    asset = node.parent()
    if not asset.evalParm("reference_enable"):
        asset.parm("reference_status").set("Comparison is disabled.")
        return
    field = asset.evalParm("color_attrib")
    path = asset.evalParm("PVD_file")
    step = entry_index(asset)
    block, _ = _selected_fields(asset)
    reduction = _menu_token(asset, "color_reduction")
    current_mesh = load_block(path, step, block)
    current = (_scan_reduced(asset, current_mesh, field, reduction)
               if current_mesh is not None else None)
    if current is None or geo.findPointAttrib(field) is None:
        asset.parm("reference_status").set(
            f"Current field '{field}' is unavailable.")
        return
    reference_frame = asset.evalParm("reference_frame")
    reference_step = frame_step(asset, reference_frame)
    try:
        reference_topo, reference = _reference_reduced(
            asset, path, block, field, reduction)
    except (OSError, ET.ParseError, VtuParseError, ValueError) as exc:
        asset.parm("reference_status").set(f"Reference load failed: {exc}")
        return
    if reference is None:
        asset.parm("reference_status").set(
            f"Reference field '{field}' is unavailable.")
        return
    if (reference_topo != current_mesh["topo_key"]
            or len(reference) != len(current)
            or len(current) != geo.intrinsicValue("pointcount")):
        asset.parm("reference_status").set(
            "Reference comparison requires matching point topology.")
        return
    mode = _menu_token(asset, "reference_mode")
    note = ""
    if mode == "percent":
        delta, near_zero = _percent_delta(current, reference)
        if near_zero:
            note = (f" {near_zero:,} of {len(reference):,} points have a "
                    "near-zero reference baseline. Changes above tolerance "
                    "at those points are undefined.")
    else:
        delta = _apply_reference(current, reference, mode)
    if geo.findPointAttrib("reference_comparison") is None:
        geo.addAttrib(hou.attribType.Point, "reference_comparison", 0.0,
                      create_local_variable=False)
    _set_floats(geo.setPointFloatAttribValuesFromString,
                "reference_comparison", delta)
    times = _pvd_times(asset)
    when = (f" (step {reference_step}, t = {times[reference_step]:.6g})"
            if reference_step < len(times) else "")
    asset.parm("reference_status").set(
        f"Comparing {field} with frame {reference_frame}{when} using {mode}."
        + note)


def cook_field_smoothing(node):
    """Nodally average the displayed scalar over coincident vertices.

    PolyFEM writes a discontinuous mesh (per-element duplicated nodes), so a
    field like von Mises stress is constant inside each element and shows as a
    flat color. Averaging `for_color` over each coincidence group (FEM nodal
    recovery) yields a continuous field that interpolates across elements.
    The groups come from the cached topology, so this is one vectorized
    group-average per frame. Non-finite values are excluded so invalid points
    do not contaminate a whole group.
    """
    geo = node.geometry()
    if geo.findPointAttrib("for_color") is None \
            or geo.findPointAttrib("coincident_id") is None:
        return
    values = np.frombuffer(
        geo.pointFloatAttribValuesAsString("for_color"),
        dtype=np.float32).astype(np.float64)
    groups = np.frombuffer(
        geo.pointFloatAttribValuesAsString("coincident_id"),
        dtype=np.float32).astype(np.int64)
    if not len(groups):
        return
    _set_floats(geo.setPointFloatAttribValuesFromString, "for_color",
                _nodal_average(values, groups))


def _result_row(geo, point_id):
    """Index in `geo` of the point whose readpvd_point is `point_id`, or -1.

    Without a body filter the id is the point number (checked first); hidden
    bodies renumber the points, and then the id is looked up."""
    count = geo.intrinsicValue("pointcount")
    if point_id < 0:
        return -1
    if geo.findPointAttrib("readpvd_point") is None:
        return point_id if point_id < count else -1
    if point_id < count \
            and geo.point(point_id).attribValue("readpvd_point") == point_id:
        return point_id
    ids = np.frombuffer(geo.pointIntAttribValuesAsString("readpvd_point"),
                        dtype=np.int32)
    hit = np.flatnonzero(ids == point_id)
    return int(hit[0]) if len(hit) else -1


def _probe_geometry(node):
    """The result before clipping, where probed points keep their ids."""
    result = node.node("OUT_result")
    return result.geometry() if result is not None else None


def probe_pick(node, output_point):
    """Probe the vertex picked in the viewport (output point number).

    The displayed geometry is clipped and sliced, which renumbers its points
    from frame to frame as the deformed body moves through the plane, and
    adds cut points that are no vertex of the model. The probe keeps the
    vertex's readpvd_point id instead: the picked point's own id when it is
    a model vertex, else the nearest model vertex of the same body in the
    undeformed shape."""
    output = node.node("output")
    result = _probe_geometry(node)
    if output is None or result is None:
        return ""
    shown = output.geometry()
    if output_point < 0 or output_point >= len(shown.points()):
        return ""
    picked = shown.point(output_point)
    rest_attrib = shown.findPointAttrib("rest")
    rest = np.asarray(picked.attribValue(rest_attrib) if rest_attrib
                      else picked.position(), dtype=np.float64)
    id_attrib = shown.findPointAttrib("readpvd_point")
    candidate = int(picked.attribValue(id_attrib)) if id_attrib \
        else output_point
    rests = np.frombuffer(result.pointFloatAttribValuesAsString("rest"),
                          dtype=np.float32).reshape(-1, 3).astype(np.float64) \
        if result.findPointAttrib("rest") is not None else np.asarray(
            [p.position() for p in result.points()], dtype=np.float64)
    if not len(rests):
        return ""
    # Clipped points can carry interpolated integer point IDs. Compare in
    # rest space at the local face scale, independent of world translation.
    extent = 0.0
    for prim in picked.prims():
        local = np.asarray([p.attribValue(rest_attrib) if rest_attrib else p.position()
                            for p in prim.points()], dtype=np.float64)
        extent = max(extent, float(np.ptp(local, axis=0).max()))
    row = _result_row(result, candidate)
    if row < 0 or np.abs(rests[row] - rest).max() > 1e-6 * extent:
        # a cut point of the clip (or an id the clip interpolated)
        mask = np.ones(len(rests), dtype=bool)
        body_attrib = shown.findPointAttrib("body_ids")
        if body_attrib is not None \
                and result.findPointAttrib("body_ids") is not None:
            bodies = np.rint(np.frombuffer(
                result.pointFloatAttribValuesAsString("body_ids"),
                dtype=np.float32))
            same = bodies == np.rint(float(picked.attribValue(body_attrib)))
            if same.any():
                mask = same
        rows = np.flatnonzero(mask)
        delta = rests[rows] - rest
        row = int(rows[np.argmin(np.einsum("ij,ij->i", delta, delta))])
        candidate = int(result.point(row).attribValue("readpvd_point")) \
            if result.findPointAttrib("readpvd_point") is not None else row
    return probe_point(node, candidate)


def probe_readout(node, point_number):
    """Current position and readout text for a probed point.

    `point_number` is the vertex's readpvd_point id (its point number in the
    result before clipping). Reads from that geometry, so calling it again
    after the frame changes returns the vertex's *new* position and field
    values, whatever a clip does to the displayed point numbers. Pure: it
    writes no parameters, so it is safe to call every viewport redraw.
    Returns (position, text) or (None, "") if the point is unavailable.
    """
    geo = _probe_geometry(node)
    if geo is None:
        return None, ""
    row = _result_row(geo, int(point_number))
    if row < 0:
        return None, ""
    point = geo.point(row)
    position = point.position()
    digits = int(node.evalParm("legend_digits")) \
        if node.parm("legend_digits") is not None else 5
    notation = _menu_token(node, "legend_number_format") or "automatic"

    def _fmt(v):
        if isinstance(v, (tuple, list)):
            return ", ".join(_format_number(c, digits, notation) for c in v)
        if isinstance(v, float):
            return _format_number(v, digits, notation)
        return str(v)

    lines = [f"Point {point_number}",
             "Position: " + _fmt(tuple(position))]
    block_attrib = geo.findPointAttrib("pvd_block")
    if block_attrib is not None:
        block = point.attribValue(block_attrib)
        if block:
            lines.append(f"PVD Block: {block}")
    color_attrib = node.evalParm("color_attrib")
    units_text = node.evalParm("field_units").strip()
    comparison = geo.findPointAttrib("reference_comparison") is not None
    comparison_units = "%" if comparison and _percent_reference(node) else units_text
    if node.evalParm("smooth_field"):
        for_color = geo.findPointAttrib("for_color")
        if for_color is not None:
            smoothed = point.attribValue(for_color)
            lines.append(
                f"Displayed (smoothed {_field_label(color_attrib)}): "
                f"{_fmt(smoothed)}"
                + (f" {comparison_units}" if comparison_units else ""))
    seen_names = set()
    for name in ("solution", color_attrib,
                 "reference_comparison", "body_ids", "sidesets",
                 "cauchy_eigenvalues", "pk2_eigenvalues",
                 "principal_stretches"):
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        attrib = geo.findPointAttrib(name)
        if attrib is None:
            continue
        value = point.attribValue(attrib)
        text = _fmt(value)
        units = (comparison_units if name == "reference_comparison" else
                 units_text if name == color_attrib else "")
        lines.append(f"{_field_label(name)}: {text}"
                     + (f" {units}" if units else ""))
    return position, "\n".join(lines)


def probe_point(node, point_number):
    """Store the probed point and its readout snapshot (called on pick)."""
    position, readout = probe_readout(node, point_number)
    if position is None:
        return ""
    node.setParms({"probe_point": point_number, "probe_readout": readout})
    return readout


