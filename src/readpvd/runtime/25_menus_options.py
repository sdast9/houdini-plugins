def source_block_menu(kwargs):
    metadata = _frame_metadata(kwargs["node"])
    ordered = [name for name in BLOCK_LABELS if name in metadata]
    ordered.extend(name for name in metadata if name not in BLOCK_LABELS)
    if not ordered:
        return ["", "No Renderable Blocks Found"]
    result = []
    for name in ordered:
        result.extend((name, BLOCK_LABELS.get(name, name)))
    return result


def color_field_menu(kwargs):
    node = kwargs["node"]
    scope = _menu_token(node, "field_time_scope") or "every"
    if scope == "current":
        available = _available_fields(node)
        counts = {name: 1 for name in available}
        frame_count = 1
    else:
        block = _source_block_token(node)
        frames, block_counts = _sequence_metadata(node)
        frame_count = block_counts.get(block, 0)
        available = {}
        counts = {}
        for frame_meta in frames:
            fields = _available_fields_from_fields(
                frame_meta.get(block, {}), bool(node.evalParm("derived")))
            for name, components in fields.items():
                available.setdefault(name, components)
                counts[name] = counts.get(name, 0) + 1
        if scope == "every":
            available = {
                name: components for name, components in available.items()
                if frame_count > 0 and counts.get(name, 0) == frame_count
            }
    if not available:
        return ["", "No Renderable Point Fields Found"]

    preferred = [name for name in FIELD_LABELS if name in available]
    preferred.extend(sorted(name for name in available if name not in preferred))
    result = []
    for name in preferred:
        label = _field_label(name)
        if scope != "current":
            count = counts.get(name, 0)
            label += (
                " [all frames]" if count == frame_count
                else f" [{count}/{frame_count} frames]")
        result.extend((name, label))
    return result


def _available_fields(node):
    _, fields = _selected_fields(node)
    return _available_fields_from_fields(fields, bool(node.evalParm("derived")))


def _available_fields_from_fields(fields, derived):
    available = dict(fields)
    if derived:
        requirements = _requirements(fields)
        for name, requirement in DERIVED_FIELD_REQUIREMENTS.items():
            if requirements[requirement]:
                available[name] = 9 if name in (
                    "F_mat", "right_cauchy_green", "left_cauchy_green",
                    "right_stretch", "left_stretch", "green_lagrange_strain",
                    "almansi_strain", "hencky_strain",
                    "infinitesimal_strain", "cauchy_mat",
                    "deviatoric_stress", "pk1", "pk2") else (
                        3 if name in PRINCIPAL_VALUE_FIELDS else 1)
    return available


def color_reduction_menu(kwargs):
    node = kwargs["node"]
    field = node.evalParm("color_attrib")
    components = _available_fields(node).get(field, 0)
    if not components and _menu_token(node, "field_time_scope") == "any":
        block = _source_block_token(node)
        frames, _ = _sequence_metadata(node)
        for frame_meta in frames:
            components = _available_fields_from_fields(
                frame_meta.get(block, {}), bool(node.evalParm("derived"))
            ).get(field, 0)
            if components:
                break
    if components == 1:
        entries = (("auto", "Scalar Value"),)
    elif components in (2, 3) and field in PRINCIPAL_VALUE_FIELDS:
        entries = (
            ("auto", "Principal-Value Magnitude"),
            ("magnitude", "Principal-Value Magnitude"),
            ("principal_max", "First Principal Value (Largest)"),
            ("principal_middle", "Second Principal Value"),
            ("principal_min", "Third Principal Value (Smallest)"),
        )
    elif components in (2, 3):
        entries = (
            ("auto", "Vector Magnitude"),
            ("magnitude", "Vector Magnitude"),
            ("x", "X Component"),
            ("y", "Y Component"),
            ("z", "Z Component"),
        )
    elif components == 9:
        entries = (
            ("auto", "Frobenius Norm"),
            ("magnitude", "Frobenius Norm"),
            ("x", "XX Entry"),
            ("y", "YY Entry"),
            ("z", "ZZ Entry"),
            ("xy", "XY Entry"),
            ("yz", "YZ Entry"),
            ("xz", "XZ Entry"),
            ("principal_max", "First Principal Value (Largest)"),
            ("principal_middle", "Second Principal Value"),
            ("principal_min", "Third Principal Value (Smallest)"),
            ("trace", "Tensor Trace"),
            ("determinant", "Tensor Determinant"),
        )
    else:
        return ["", "No Displayable Value"]
    result = []
    for token, label in entries:
        result.extend((token, label))
    return result


def glyph_tensor_menu(kwargs):
    node = kwargs["node"]
    _, fields = _selected_fields(node)
    requirements = _requirements(fields)
    available = [
        (token, label) for token, label, requirement in GLYPH_FIELDS
        if node.evalParm("derived") and requirements[requirement]]
    if not available:
        return ["", "No Compatible Tensor Fields Found"]
    result = []
    for token, label in available:
        result.extend((token, label))
    return result


def _available_body_ids(node):
    """Sorted unique PolyFEM body ids in the active frame's selected block."""
    path = node.evalParm("PVD_file")
    if not path:
        return []
    frame = entry_index(node)
    selected = _source_block_token(node)
    # Metadata scan is already cached and costs milliseconds: if this block
    # has no body_ids field at all, avoid reading any mesh arrays from disk.
    try:
        info = frame_field_info(path, frame)
    except Exception:
        info = {}
    if info:
        block_name = selected if selected in info else next(iter(info))
        if "body_ids" not in info[block_name].get("point_data", {}):
            return []
    try:
        block_paths = _frame_block_paths(path, frame)
    except Exception:
        return []
    block_path = block_paths.get(selected)
    if not block_path or not os.path.isfile(block_path):
        block_path = next(
            (p for p in block_paths.values() if os.path.isfile(p)), None)
    if not block_path:
        return []
    try:
        body = read_mesh_point_field(block_path, "body_ids")
    except Exception:
        return []
    if body is None:
        return []
    body = np.asarray(body)
    return sorted(int(value) for value in np.unique(body[np.isfinite(body)]))


def body_id_menu(kwargs):
    """Toggle menu of the bodies present in the data (data-driven)."""
    ids = _available_body_ids(kwargs["node"])
    if not ids:
        return ["", "No Bodies In This Block"]
    result = []
    for body in ids:
        result.extend((str(body), f"Body {body}"))
    return result


def _visible_body_set(node):
    """Set of body ids the user chose to show, or None to show all."""
    text = node.evalParm("visible_bodies").strip()
    if not text:
        return None
    ids = set()
    for token in text.replace(",", " ").split():
        try:
            ids.add(int(float(token)))
        except ValueError:
            pass
    return ids or None


def _menu_tokens(flat_menu):
    return flat_menu[::2]


# The field a result opens with: von Mises stress -- exported, or derived from
# F and the Cauchy stress, which is all the PolyFEM node's default Minimal
# Fields output writes -- else the displacement.
_DEFAULT_COLOR_FIELDS = ("von_mises", "von_mises_derived", "solution_mag",
                         "solution")


def _first_color_field(node):
    tokens = _menu_tokens(color_field_menu({"node": node}))
    for preferred in _DEFAULT_COLOR_FIELDS:
        if preferred in tokens:
            return preferred
    return tokens[0] if tokens and tokens[0] else ""


def _absent_from_completed_sequence(node, block):
    path = node.evalParm("PVD_file")
    if node.evalParm("follow_running") or not _manifest_is_finished(run_manifest(path)):
        return False
    try:
        entries = read_pvd(path)
        # A block listed in a VTM may still be waiting for its data file.
        return bool(entries) and all(
            block not in _frame_block_paths(path, step) for step in range(len(entries)))
    except (OSError, ET.ParseError, VtuParseError, ValueError):
        return False


def sync_available_options(kwargs):
    """Keep all user-selectable data choices valid for the active block."""
    node = kwargs["node"]
    metadata = _frame_metadata(node)
    if not metadata and node.evalParm("PVD_file") \
            and node.evalParm("source_block"):
        # Transient empty read (a running sim mid-write on this frame's
        # files): leave every selection parm exactly as the user set it
        # instead of resetting block/color/visibility to defaults.
        node.setParms({name: 0 for name in (
            "has_solution_data", "has_glyph_data", "has_fiber_data", "has_multibody",
            "has_block_volume", "has_block_surface", "has_block_contact", "has_block_points")})
        node.parm("availability_status").set(
            "Current frame unreadable (sim writing?); keeping settings.")
        return
    block_tokens = _menu_tokens(source_block_menu({"node": node}))
    selected_block = _source_block_token(node)
    if not selected_block or (selected_block not in metadata
                              and _absent_from_completed_sequence(node, selected_block)):
        selected_block = block_tokens[0] if block_tokens else ""
        node.parm("source_block").set(selected_block)

    _, fields = _selected_fields(node)
    requirements = _requirements(fields)
    derived = bool(node.evalParm("derived"))
    color_tokens = _menu_tokens(color_field_menu({"node": node}))
    if node.evalParm("color_attrib") not in color_tokens:
        node.parm("color_attrib").set(_first_color_field(node))
    reduction_tokens = _menu_tokens(color_reduction_menu({"node": node}))
    if node.evalParm("color_reduction") not in reduction_tokens:
        node.parm("color_reduction").set(
            reduction_tokens[0] if reduction_tokens else "")
    glyph_tokens = _menu_tokens(glyph_tensor_menu({"node": node}))
    if node.evalParm("glyph_tensor") not in glyph_tokens:
        node.parm("glyph_tensor").set(glyph_tokens[0] if glyph_tokens else "")

    has_solution = int(requirements["solution"])
    has_glyph = int(bool(glyph_tokens and glyph_tokens[0]))
    fiber_attribs = [_fiber_attrib(prefix)
                     for prefix in _fiber_prefixes(fields)]
    if not fiber_attribs:
        fiber_attribs = _companion_family_names(node.evalParm("PVD_file"))
    node.parm("fiber_attribs").set(" ".join(fiber_attribs))
    if node.evalParm("fiber_family") not in ["all"] + fiber_attribs:
        node.parm("fiber_family").set("all")
    body_ids = _available_body_ids(node)
    flags = {
        "has_solution_data": has_solution,
        "has_glyph_data": has_glyph,
        "has_fiber_data": int(bool(fiber_attribs)),
        "has_multibody": int(len(body_ids) > 1),
    }
    for slug in ("volume", "surface", "contact", "points"):
        block_name = slug.title()
        present = int(block_name in metadata)
        flags[f"has_block_{slug}"] = present
    node.setParms(flags)
    if not has_glyph and node.evalParm("add_glyphs"):
        # Preserve the toggle during temporary field loss. Retain the existing
        # reset for explicit Derived-off and complete sequences with no tensor.
        finished = not node.evalParm("follow_running") and _manifest_is_finished(
            run_manifest(node))
        frames = _sequence_metadata(node)[0] if finished and derived else []
        if not derived or (frames and all(selected_block in frame for frame in frames)
                           and not any(_requirements(frame[selected_block])[
                               "deformation_gradient"] for frame in frames)):
            node.parm("add_glyphs").set(0)
    renderable_count = max(0, len(color_tokens) - (1 if color_tokens == [""] else 0))
    block_label = BLOCK_LABELS.get(selected_block, selected_block or "None")
    summary = (
        f"{block_label}: {renderable_count} renderable point field"
        f"{'s' if renderable_count != 1 else ''}; "
        f"{len(block_tokens) if block_tokens != [''] else 0} available block"
        f"{'s' if len(block_tokens) != 1 else ''}.")
    if fiber_attribs and not requirements["deformation_gradient"] \
            and _menu_token(node, "fiber_frame") != "reference":
        summary += " Deformed fiber directions are unavailable on this block and step."
    _set_availability_status(node, summary)


def _autocenter_clip(node):
    """Initialize the clip/slice plane origin to the scene's center.

    Runs when a PVD file is (re)loaded and only while the origin is still at
    the factory default (0, 0, 0); an origin the user has moved is never
    touched. The center is the bounding-box midpoint of every block in the
    scene's first frame, so the plane starts inside the geometry instead of
    at the world origin. The plane normal, if still at its factory default
    (+X), is aligned to the longest bounding-box axis so the first clip cuts
    across the scene's dominant extent.
    """
    origin_parms = [node.parm("clip_origin" + axis) for axis in "xyz"]
    if any(parm is None for parm in origin_parms):
        return
    if any(parm.eval() != 0.0 for parm in origin_parms):
        return
    path = node.evalParm("PVD_file")
    if not path:
        return
    try:
        block_paths = _frame_block_paths(path, 0)
    except Exception:
        return
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for block_path in block_paths.values():
        if not os.path.isfile(block_path):
            continue
        try:
            points = np.asarray(read_mesh_points(block_path), dtype=np.float64)
        except Exception:
            continue
        if points.ndim == 2 and len(points):
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
    if not np.isfinite(low).all():
        return
    center = 0.5 * (low + high)
    for parm, value in zip(origin_parms, center):
        parm.set(float(value))
    normal_parms = [node.parm("clip_direction" + axis) for axis in "xyz"]
    if all(parm is not None for parm in normal_parms) \
            and [parm.eval() for parm in normal_parms] == [1.0, 0.0, 0.0]:
        longest = int(np.argmax(high - low))
        for i, parm in enumerate(normal_parms):
            parm.set(1.0 if i == longest else 0.0)


def source_changed(kwargs):
    # Same files, different block: the identity-keyed disk caches stay warm;
    # only the cooked-frame cache holds the wrong block's geometry.
    sync_available_options(kwargs)
    clear_cache(kwargs)
    update_color_status(kwargs)


def analysis_options_changed(kwargs):
    node = kwargs["node"]
    sync_available_options(kwargs)
    parm = kwargs.get("parm")
    if parm is not None and parm.name() == "visible_bodies":
        auto_range_default(node)
    update_color_status(kwargs)


def reference_options_changed(kwargs):
    """Comparison tab callback: re-range displayed colors when enabling or
    adjusting reference-frame comparison (unless locked)."""
    node = kwargs["node"]
    auto_range_default(node)
    update_color_status(kwargs)


def color_selection_changed(kwargs):
    """Color Field / Field Value To Display callback: a new quantity needs its
    own range, so the displayed range follows it unless it is locked."""
    sync_available_options(kwargs)
    auto_range_default(kwargs["node"])
    update_color_status(kwargs)


def color_scale_changed(kwargs):
    """Color Scale callback: switching to log10 with a non-positive range
    auto-ranges to positive values unless Lock Displayed Range is on."""
    node = kwargs["node"]
    if (node.evalParm("color_scale") == 1
            and not node.evalParm("range_lock")
            and (node.evalParm("color_min") <= 0
                 or node.evalParm("color_max") <= 0)):
        auto_range_default(node)
    update_color_status(kwargs)


def time_scope_changed(kwargs):
    sync_available_options(kwargs)
    update_color_status(kwargs)


def multi_block_field_menu(kwargs):
    parm_name = kwargs["parm"].name()
    slug = parm_name.removeprefix("multi_").removesuffix("_field")
    block_name = slug.title()
    fields = _frame_metadata(kwargs["node"]).get(block_name, {})
    result = ["", "Solid White"]
    for name in sorted(fields):
        result.extend((name, FIELD_LABELS.get(
            name, name.replace("_", " ").title())))
    return result


