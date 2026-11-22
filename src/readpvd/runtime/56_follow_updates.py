def autoscale(kwargs):
    """Set the color ramp range from the current frame's color attribute."""
    node = kwargs["node"]
    if node.evalParm("range_lock"):
        _message("Displayed range is locked.")
        return
    color_src = node.node("OUT_result")
    if color_src is None:
        return
    color_src.cook(force=True)
    geo = color_src.geometry()
    if geo.findPointAttrib("for_color") is None:
        _message("Cook the node first (no color attribute yet).")
        return
    fixed = _fixed_color_range(node)
    if fixed is not None:
        node.setParms({"color_min": fixed[0], "color_max": fixed[1]})
        update_color_status(kwargs)
        return
    values = np.frombuffer(
        geo.pointFloatAttribValuesAsString("for_color"), dtype=np.float32)
    _apply_color_range(node, values)
    update_color_status(kwargs)


def _fixed_color_range(node):
    """(min, max) of a field whose color scale is fixed by its definition.

    Fiber dispersion is bounded by 1/d, and the preprocessing HDA colours it
    over that fixed range -- match it so the same kappa reads as the same
    colour before and after the solve. Not for a comparison with a reference
    frame, which shows changes.
    """
    if node.evalParm("reference_enable"):
        return None
    if _display_name(node.evalParm("color_attrib")).endswith("kappa"):
        return 0.0, 1.0 / 3.0
    return None


def auto_range_default(node):
    """Range the colors for a newly loaded result or a newly chosen field.

    Uses the frame on screen and the last frame: frame 0 of a loaded run is
    often all zero (no displacement or stress yet), while the end of the run
    usually holds its largest values, so a range from frame 0 alone would
    draw the whole run in one flat color. Mirrors the color stage in numpy
    (no cook), and does nothing while Lock Displayed Range is on. Returns
    True when it set the range.
    """
    if node.evalParm("range_lock"):
        return False
    fixed = _fixed_color_range(node)
    if fixed is not None:
        node.setParms({"color_min": fixed[0], "color_max": fixed[1]})
        return True
    path = node.evalParm("PVD_file")
    if not path or not node.evalParm("color_attrib"):
        return False
    try:
        count = len(read_pvd(path))
    except Exception:
        return False
    if not count:
        return False
    current = max(0, min(entry_index(node), count - 1))
    frames = [current] if node.evalParm("reference_enable") else sorted({current, count - 1})
    try:
        collected = [values[np.isfinite(values)] for _, _, values in
                     _scan_frames(node, path, frames)
                     if values is not None]
    except Exception:
        return False
    collected = [chunk for chunk in collected if len(chunk)]
    if not collected:
        return False
    return _apply_color_range(node, np.concatenate(collected), quiet=True)


def autoscale_all(kwargs):
    """Scan the displayed value across the PVD sequence with direct numpy.

    Mirrors the color stage (reduction, reference comparison, smoothing) per
    frame instead of force-cooking the result pipeline and stepping the
    playbar -- far faster, and it never disturbs the current frame.
    """
    node = kwargs["node"]
    if node.evalParm("range_lock"):
        _message("Displayed range is locked.")
        return
    path = node.evalParm("PVD_file")
    if not path:
        return
    fixed = _fixed_color_range(node)
    if fixed is not None:
        node.setParms({"color_min": fixed[0], "color_max": fixed[1]})
        update_color_status(kwargs)
        return
    percentiles = tuple(sorted(node.evalParmTuple("range_percentiles"))) \
        if node.evalParm("range_percentile") else (0.0, 100.0)
    interior = percentiles != (0.0, 100.0)
    try:
        limits, sampled = _all_frame_range(
            _scan_displayed_values(node, path, cache_rows=interior),
            percentiles, node.evalParm("color_scale") == 1)
    except Exception as exc:
        _message(f"Could not scan PVD sequence: {exc}")
        return
    if limits is not None:
        _set_color_range(node, *limits)
    else:
        _message("No valid values were found for the selected color scale.")
    update_color_status(kwargs)
    if sampled:
        status = node.parm("color_status")
        status.set(f"{status.eval()} Percentiles estimated from a bounded "
                   f"sample of {sampled:,} values.")


def _all_frame_range(rows, percentiles, positive_only):
    """Exact extrema and exact percentiles until the bounded buffer fills.

    When full, retain every other sample and double the stride. Keep the
    stride aligned to the global valid-value index, independent of frame
    sizes and chunk boundaries. The buffer, bounded scan cache, and small
    filtering/compaction allocations fit within 256 MiB besides the live
    input frame and field reduction.
    """
    low, high = np.inf, -np.inf
    buffer = (np.empty(_RANGE_SAMPLE_MAX_VALUES, dtype=np.float64)
              if percentiles != (0.0, 100.0) else None)
    count = total = 0
    stride = 1
    for _, _, values in rows:
        if values is None:
            continue
        for start in range(0, len(values), _RANGE_CHUNK_VALUES):
            chunk = values[start:start + _RANGE_CHUNK_VALUES]
            valid = np.isfinite(chunk)
            if positive_only:
                valid &= chunk > 0
            low = min(low, np.min(chunk, where=valid, initial=np.inf))
            high = max(high, np.max(chunk, where=valid, initial=-np.inf))
            if buffer is None:
                continue
            chunk = chunk[valid]
            stop = total + len(chunk)
            while count + len(chunk[(-total % stride)::stride]) > len(buffer):
                # Copy in bounded chunks because source and destination overlap.
                kept = (count + 1) // 2
                for offset in range(0, kept, _RANGE_CHUNK_VALUES):
                    end = min(offset + _RANGE_CHUNK_VALUES, kept)
                    buffer[offset:end] = buffer[2 * offset:2 * end:2].copy()
                count = kept
                stride *= 2
            sample = chunk[(-total % stride)::stride]
            buffer[count:count + len(sample)] = sample
            count += len(sample)
            total = stop
    if low > high:
        return None, 0
    if buffer is None:
        return (low, high), 0
    estimates = np.percentile(buffer[:count], percentiles, overwrite_input=True)
    limits = (low if percentiles[0] == 0 else estimates[0],
              high if percentiles[1] == 100 else estimates[1])
    return limits, count if stride > 1 else 0


def _apply_color_range(node, values, quiet=False):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if node.evalParm("color_scale") == 1:
        values = values[values > 0]
    if not len(values):
        if not quiet:
            _message("No valid values were found for the selected color "
                     "scale.")
        return False
    if node.evalParm("range_percentile"):
        low_percentile, high_percentile = node.evalParmTuple(
            "range_percentiles")
        low, high = np.percentile(
            values, (min(low_percentile, high_percentile),
                     max(low_percentile, high_percentile)))
    else:
        low, high = values.min(), values.max()
    _set_color_range(node, low, high)
    return True


def _set_color_range(node, low, high):
    if node.evalParm("range_symmetric") and node.evalParm("color_scale") == 0:
        extent = max(abs(float(low)), abs(float(high)))
        low, high = -extent, extent
    node.setParms({"color_min": float(low), "color_max": float(high)})


def set_diverging_ramp(kwargs):
    node = kwargs["node"]
    ramp = hou.Ramp(
        (hou.rampBasis.Linear, hou.rampBasis.Linear, hou.rampBasis.Linear),
        (0.0, 0.5, 1.0),
        ((0.1, 0.25, 0.9), (1.0, 1.0, 1.0), (0.9, 0.15, 0.1)))
    node.parm("color_ramp").set(ramp)


def update_color_status(kwargs):
    """Report color-field availability from metadata (no network cook)."""
    node = kwargs["node"]
    field = node.evalParm("color_attrib")
    components = _available_fields(node).get(field, 0)
    if not components and _menu_token(node, "field_time_scope") != "current":
        block = _source_block_token(node)
        frames, _ = _sequence_metadata(node)
        for frame_meta in frames:
            components = _available_fields_from_fields(
                frame_meta.get(block, {}), bool(node.evalParm("derived"))
            ).get(field, 0)
            if components:
                break
    if components:
        reduction = REDUCTION_LABELS.get(
            _menu_token(node, "color_reduction"), "Automatic Value")
        status = (
            f"Available: '{field}' ({components} value"
            f"{'s' if components != 1 else ''} per point); "
            f"displaying {reduction}.")
    elif not field:
        status = "No color field is selected."
    else:
        status = (
            f"Unavailable: '{field}'. The result uses the neutral "
            "Unavailable Field Color.")
    parm = node.parm("color_status")
    if parm is not None:
        parm.set(status)


def _scene_viewer():
    if not hou.isUIAvailable():
        return None
    try:
        return hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    except Exception:
        return None


def show_overlay(kwargs):
    """Activate this node's viewer state so the overlay legend (and probe)
    draw without the user pressing Enter in the viewport.

    A Python viewer state only draws while it is the active state, so the
    overlay persists through viewport navigation once activated but ends when
    you select another node or switch tools -- a Houdini limitation for
    state-drawn overlays. For an always-on legend regardless of selection, use
    the Scene Geometry legend mode (renderable geometry).
    """
    node = kwargs["node"]
    viewer = _scene_viewer()
    if viewer is None:
        return
    state_name = node.type().definition().sections()[
        "DefaultState"].contents()
    try:
        viewer.setCurrentNode(node)
        viewer.setCurrentState(state_name, generate=hou.stateGenerateMode.Enter)
    except hou.Error as exc:
        _message(f"Could not activate the viewport overlay: {exc}")


_AUTOPLACED_LEGEND = {}
_AUTOPLACED_GNOMON = {}


def _same_placement(current, reference, tol=1e-6):
    """Compare ((x, y, z), scale) tuples with scale-aware tolerance."""
    if reference is None:
        return False
    (cx, cy, cz), cs = current
    (rx, ry, rz), rs = reference
    eps = max(tol, abs(rs) * 1e-4)
    return (abs(cx - rx) <= eps and abs(cy - ry) <= eps
            and abs(cz - rz) <= eps and abs(cs - rs) <= eps)


def _scene_reference_bbox(node):
    """Bounding box (in `node`'s local space) of this result plus any nearby
    visible scene geometry in /obj so legends and gnomons never overlap what
    is already visible in the viewport."""
    own_bbox = None
    src = node.node("OUT_result")
    if src is not None:
        try:
            cand = src.geometry().boundingBox()
            if cand.isValid():
                own_bbox = cand
        except hou.OperationFailed:
            pass
    if own_bbox is None:
        path = node.evalParm("PVD_file") if node.parm("PVD_file") else ""
        if path:
            try:
                block_paths = _frame_block_paths(path, 0)
                low = np.full(3, np.inf)
                high = np.full(3, -np.inf)
                for block_path in block_paths.values():
                    if not os.path.isfile(block_path):
                        continue
                    pts = np.asarray(read_mesh_points(block_path), dtype=np.float64)
                    if pts.ndim == 2 and len(pts):
                        low = np.minimum(low, pts.min(axis=0))
                        high = np.maximum(high, pts.max(axis=0))
                if np.isfinite(low).all():
                    own_bbox = hou.BoundingBox(
                        float(low[0]), float(low[1]), float(low[2]),
                        float(high[0]), float(high[1]), float(high[2]))
            except Exception:
                pass
    if own_bbox is None or not own_bbox.isValid():
        return None
    own_size = own_bbox.sizevec()
    own_span = max(float(own_size[0]), float(own_size[1]), float(own_size[2]))
    if own_span <= 0.0:
        return None

    bmin = list(own_bbox.minvec())
    bmax = list(own_bbox.maxvec())
    own_center = own_bbox.center()
    parent = node.parent()
    if parent is not None:
        try:
            inv_node = node.worldTransform().inverted()
        except Exception:
            inv_node = None
        if inv_node is not None:
            for sibling in parent.children():
                if sibling == node:
                    continue
                try:
                    if not sibling.isDisplayFlagSet():
                        continue
                    if sibling.childTypeCategory() != hou.sopNodeTypeCategory():
                        continue
                    target = sibling.node("OUT_result") or sibling.displayNode()
                    if target is None:
                        continue
                    sib_bbox = target.geometry().boundingBox()
                    if not sib_bbox.isValid():
                        continue
                    rel = sibling.worldTransform() * inv_node
                    smin = sib_bbox.minvec()
                    smax = sib_bbox.maxvec()
                    corners = [
                        hou.Vector3(
                            smin[0] if i & 1 else smax[0],
                            smin[1] if i & 2 else smax[1],
                            smin[2] if i & 4 else smax[2]) * rel
                        for i in range(8)
                    ]
                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    zs = [c[2] for c in corners]
                    sib_span = max(max(xs) - min(xs), max(ys) - min(ys),
                                   max(zs) - min(zs))
                    sib_center = hou.Vector3(
                        0.5 * (min(xs) + max(xs)),
                        0.5 * (min(ys) + max(ys)),
                        0.5 * (min(zs) + max(zs)))
                    if not (1e-6 * own_span <= sib_span <= 10.0 * own_span):
                        continue
                    if (sib_center - own_center).length() > 10.0 * own_span:
                        continue
                    bmin = [min(bmin[0], min(xs)), min(bmin[1], min(ys)),
                            min(bmin[2], min(zs))]
                    bmax = [max(bmax[0], max(xs)), max(bmax[1], max(ys)),
                            max(bmax[2], max(zs))]
                except Exception:
                    continue
    return hou.BoundingBox(bmin[0], bmin[1], bmin[2],
                           bmax[0], bmax[1], bmax[2])


def overlay_feature_changed(kwargs):
    """Auto-place the scene legend when first enabled, and auto-activate the
    viewer state when an overlay legend or the probe is enabled."""
    node = kwargs["node"]
    mode = node.evalParm("legend_mode")
    if mode in (1, 3):
        autoplace_legend(kwargs, only_if_default=True)
    if mode in (2, 3) or node.evalParm("probe_enabled"):
        show_overlay(kwargs)


def legend_orientation_changed(kwargs):
    """Reposition an uncustomized scene legend when its plane/up changes."""
    node = kwargs["node"]
    if node.evalParm("legend_mode") in (1, 3):
        autoplace_legend(kwargs, only_if_default=True)


def gnomon_show_changed(kwargs):
    """Auto-size and place the scene gnomon beside the model when first shown."""
    node = kwargs["node"]
    if node.evalParm("gnomon_show"):
        autoplace_gnomon(kwargs, only_if_default=True)


def autoplace_legend(kwargs, only_if_default=False):
    """Size and position the scene legend cleanly beside the visible scene
    geometry, respecting the chosen plane/up orientation."""
    node = kwargs["node"]
    sid = node.sessionId()
    if only_if_default:
        current = (tuple(float(v) for v in node.evalParmTuple("legend_translate")),
                   float(node.evalParm("legend_scale")))
        is_factory = _same_placement(current, ((0.0, 0.0, 0.0), 1.0))
        is_auto = _same_placement(current, _AUTOPLACED_LEGEND.get(sid))
        if not (is_factory or is_auto):
            return
    bbox = _scene_reference_bbox(node)
    if bbox is None:
        return
    size = bbox.sizevec()
    center = bbox.center()
    span = max(float(size[0]), float(size[1]), float(size[2]), 1e-9)

    base_rot = legend_rotation(node)
    extra_rot = tuple(float(v) for v in node.evalParmTuple("legend_rotate"))
    rot_mat = hou.hmath.buildRotate((
        base_rot[0] + extra_rot[0],
        base_rot[1] + extra_rot[1],
        base_rot[2] + extra_rot[2],
    ), "xyz")
    right_dir = (hou.Vector3(1.0, 0.0, 0.0) * rot_mat).normalized()
    up_dir = (hou.Vector3(0.0, 1.0, 0.0) * rot_mat).normalized()

    up_extent = (abs(up_dir[0]) * float(size[0])
                 + abs(up_dir[1]) * float(size[1])
                 + abs(up_dir[2]) * float(size[2]))
    right_extent = (abs(right_dir[0]) * float(size[0])
                    + abs(right_dir[1]) * float(size[1])
                    + abs(right_dir[2]) * float(size[2]))

    scale = max(min(max(0.75 * up_extent, 0.45 * span), 0.90 * span), 1e-6)
    margin = max(0.15 * span, 0.08 * scale)
    pos = (center
           + right_dir * (0.5 * right_extent + margin)
           - up_dir * (0.5 * scale))
    translate = (float(pos[0]), float(pos[1]), float(pos[2]))
    node.parmTuple("legend_translate").set(translate)
    node.parm("legend_scale").set(float(scale))
    _AUTOPLACED_LEGEND[sid] = (translate, float(scale))


def autoplace_gnomon(kwargs, only_if_default=False):
    """Size and position the scene gnomon cleanly at the lower-left of the
    visible scene geometry (opposite the scene legend) so all three axes sit
    clear of the model."""
    node = kwargs["node"]
    sid = node.sessionId()
    if only_if_default:
        current = (tuple(float(v) for v in node.evalParmTuple("gnomon_center")),
                   float(node.evalParm("gnomon_scale")))
        is_factory = _same_placement(current, ((0.0, 0.0, 0.0), 1.0))
        is_auto = _same_placement(current, _AUTOPLACED_GNOMON.get(sid))
        if not (is_factory or is_auto):
            return
    bbox = _scene_reference_bbox(node)
    if bbox is None:
        return
    size = bbox.sizevec()
    bmin = bbox.minvec()
    center = bbox.center()
    span = max(float(size[0]), float(size[1]), float(size[2]), 1e-9)

    scale = max(0.18 * span, 1e-6)
    margin = 0.12 * span
    origin = (float(bmin[0] - margin - scale),
              float(bmin[1]),
              float(center[2]))
    node.parmTuple("gnomon_center").set(origin)
    node.parm("gnomon_scale").set(float(scale))
    _AUTOPLACED_GNOMON[sid] = (origin, float(scale))


_GLYPH_VALUE_ATTR = {
    "right_cauchy_green": "right_cauchy_green_eigenvalues",
    "left_cauchy_green": "left_cauchy_green_eigenvalues",
    "green_lagrange": "green_lagrange_eigenvalues",
    "pk2": "pk2_eigenvalues",
    "cauchy": "cauchy_eigenvalues",
}

_GLYPH_MATRIX_ATTR = {
    "right_cauchy_green": "right_cauchy_green",
    "left_cauchy_green": "left_cauchy_green",
    "green_lagrange": "green_lagrange_strain",
    "pk2": "pk2",
    "cauchy": "cauchy_mat",
}

_GLYPH_VECTOR_ATTR = {
    "right_cauchy_green": "right_cauchy_green_eigenvectors",
    "left_cauchy_green": "left_cauchy_green_eigenvectors",
    "green_lagrange": "green_lagrange_eigenvectors",
    "pk2": "pk2_eigenvectors",
    "cauchy": "cauchy_eigenvectors",
}


def cook_fiber_recovery(node):
    """One consistent fiber direction per physical node, when smoothing is on.

    Fibers are LINE fields: a0 and -a0 describe the same fiber, and PolyFEM
    normalizes per element, so the copies of a shared node can disagree in
    sign. Averaging them naively cancels to (near) zero and the line vanishes,
    so flip every member into the first one's half-space before averaging.
    """
    asset = node.parent()
    if not (asset.evalParm("smooth_field") and asset.evalParm("show_fibers")
            and asset.evalParm("has_fiber_data")):
        return
    geo = node.geometry()
    if geo.findPointAttrib("coincident_id") is None:
        return
    try:
        families = geo.attribValue("readpvd_fiber_families").split()
    except hou.OperationFailed:
        return
    groups = np.frombuffer(
        geo.pointFloatAttribValuesAsString("coincident_id"),
        dtype=np.float32).astype(np.int64)
    if not len(groups):
        return
    size = int(groups.max()) + 1
    counts = np.maximum(np.bincount(groups, minlength=size), 1)

    for name in families:
        if geo.findPointAttrib(name) is None:
            continue
        vectors = np.frombuffer(
            geo.pointFloatAttribValuesAsString(name),
            dtype=np.float32).astype(np.float64).reshape(-1, 3)
        # Reference direction per group: the first copy encountered.
        reference = np.zeros((size, 3))
        first = np.full(size, -1, dtype=np.int64)
        order = np.argsort(groups, kind="stable")
        starts = np.searchsorted(groups[order], np.arange(size))
        valid = starts < len(order)
        first[valid] = order[starts[valid]]
        reference[valid] = vectors[first[valid]]

        signs = np.where(
            np.einsum("ij,ij->i", vectors, reference[groups]) < 0.0, -1.0, 1.0)
        flipped = vectors * signs[:, None]
        averaged = np.empty((size, 3))
        for axis in range(3):
            averaged[:, axis] = np.bincount(
                groups, weights=flipped[:, axis], minlength=size) / counts
        norms = np.linalg.norm(averaged, axis=1)
        degenerate = norms < 1e-12
        averaged[degenerate] = reference[degenerate]
        norms = np.maximum(np.linalg.norm(averaged, axis=1), 1e-12)
        averaged /= norms[:, None]
        _set_floats(geo.setPointFloatAttribValuesFromString, name,
                    averaged[groups])


def cook_glyph_recovery(node):
    """One set of principal values/directions per node when smoothing is on.

    On the discontinuous mesh, every duplicated node carries its own element
    tensor, so glyphs at one physical location overlap with conflicting
    orientations (noisy). This nodally averages the glyph tensor over each
    coincidence group and re-decomposes the averaged tensor, writing a single
    consistent eigenvalue/eigenvector pair to every copy of the node. Runs only
    when glyphs are shown and smoothing is on; the glyph VEX then draws one
    glyph per node. Vectorized: nine group-averages plus a batched eigh.
    """
    asset = node.parent()
    if not (asset.evalParm("smooth_field") and asset.evalParm("add_glyphs")
            and asset.evalParm("has_glyph_data")):
        return
    geo = node.geometry()
    tensor = _menu_token(asset, "glyph_tensor")
    matrix_name = _GLYPH_MATRIX_ATTR.get(tensor)
    value_name = _GLYPH_VALUE_ATTR.get(tensor)
    vector_name = _GLYPH_VECTOR_ATTR.get(tensor)
    if not (matrix_name and value_name and vector_name) \
            or geo.findPointAttrib(matrix_name) is None \
            or geo.findPointAttrib("coincident_id") is None:
        return
    matrices = np.frombuffer(
        geo.pointFloatAttribValuesAsString(matrix_name),
        dtype=np.float32).astype(np.float64).reshape(-1, 3, 3)
    groups = np.frombuffer(
        geo.pointFloatAttribValuesAsString("coincident_id"),
        dtype=np.float32).astype(np.int64)
    if not len(groups):
        return
    matrices = 0.5 * (matrices + np.transpose(matrices, (0, 2, 1)))
    size = int(groups.max()) + 1
    counts = np.maximum(np.bincount(groups, minlength=size), 1)
    averaged = np.empty((size, 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            averaged[:, a, b] = np.bincount(
                groups, weights=matrices[:, a, b], minlength=size) / counts
    node_vals, node_vecs = np.linalg.eigh(averaged)
    eigenvalues = np.ascontiguousarray(node_vals[:, ::-1][groups])
    eigenvectors = np.ascontiguousarray(node_vecs[:, :, ::-1][groups])
    _set_floats(geo.setPointFloatAttribValuesFromString, value_name,
                eigenvalues)
    _set_floats(geo.setPointFloatAttribValuesFromString, vector_name,
                eigenvectors)


def autofiber(kwargs):
    """Size the fiber lines to the mesh: about half an average edge.

    Fiber directions are unit vectors, so unlike the glyphs this is purely
    geometric -- no field magnitude to divide out.
    """
    node = kwargs["node"]
    src = node.node("OUT_result")
    if src is None:
        return
    try:
        src.cook(force=False)
        edge_length = src.geometry().averageEdgeLength()
    except hou.OperationFailed:
        return
    if not edge_length or not np.isfinite(edge_length):
        return
    node.parm("fiber_scale").set(0.5 * edge_length)
    _message(f"Fiber line length set to {0.5 * edge_length:.4g}.")


def autoglyph(kwargs):
    """Estimate a glyph scale that sizes glyphs to the mesh, not the units.

    The 0.26 viewer normalized each glyph's eigenvalues by their magnitude, so
    a purely geometric edge-length scale always looked right. The current line
    glyphs scale by the raw principal value, so this divides that geometric
    target by a representative principal-value magnitude of the selected
    tensor -- giving the same 'always sensibly sized' result whether the field
    is stress (~1e6) or strain (~1e-2). Normalized-length glyphs ignore the
    value, so they use the geometric target directly.
    """
    node = kwargs["node"]
    src = node.node("OUT_result")
    if src is None:
        return
    try:
        src.cook(force=False)
        geo = src.geometry()
        edge_length = geo.averageEdgeLength()
    except hou.OperationFailed:
        return
    if not edge_length or not np.isfinite(edge_length):
        return
    target = 0.5 * edge_length            # largest glyphs ~ half an edge
    mode = _menu_token(node, "glyph_length_mode")
    value_attrib = _GLYPH_VALUE_ATTR.get(_menu_token(node, "glyph_tensor"))
    if mode == "normalized" or value_attrib is None \
            or geo.findPointAttrib(value_attrib) is None:
        scale = target
    else:
        values = np.frombuffer(
            geo.pointFloatAttribValuesAsString(value_attrib),
            dtype=np.float32).reshape(-1, 3)
        magnitude = np.abs(values).max(axis=1)
        magnitude = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
        representative = float(np.percentile(magnitude, 95)) if len(
            magnitude) else 0.0
        scale = target / representative if representative > 0 else target
    parms = {"tensor_scale": float(scale)}
    if mode == "clamped":
        parms["glyph_max_length"] = float(edge_length)
    node.setParms(parms)


def auto_glyph_scale(kwargs):
    """Re-estimate the glyph scale when glyphs are enabled or retargeted."""
    node = kwargs["node"]
    if node.evalParm("add_glyphs") and node.evalParm("has_glyph_data"):
        autoglyph(kwargs)


