
FIELD_ALIASES = {
    "cauchy_stess_1": "cauchy_stress_1", "cauchy_stess_2": "cauchy_stress_2",
    "cauchy_stess_3": "cauchy_stress_3",
    "cauchy_stess_avg_1": "cauchy_stress_avg_1",
    "cauchy_stess_avg_2": "cauchy_stress_avg_2",
    "cauchy_stess_avg_3": "cauchy_stress_avg_3",
    "pk1_stess_1": "pk1_stress_1", "pk1_stess_2": "pk1_stress_2",
    "pk1_stess_3": "pk1_stress_3",
    "pk2_stess_1": "pk2_stress_1", "pk2_stess_2": "pk2_stress_2",
    "pk2_stess_3": "pk2_stress_3",
}

REDUCTION_LABELS = {
    "auto": "Automatic Value", "magnitude": "Magnitude / Frobenius Norm",
    "x": "X Component / XX Entry", "y": "Y Component / YY Entry",
    "z": "Z Component / ZZ Entry",
    "xy": "XY Entry", "yz": "YZ Entry", "xz": "XZ Entry",
    "principal_max": "First Principal Value (Largest)",
    "principal_middle": "Second Principal Value",
    "principal_min": "Third Principal Value (Smallest)", "trace": "Tensor Trace",
    "determinant": "Tensor Determinant",
}

FIELD_LABELS = {
    "von_mises": "Von Mises Stress",
    "von_mises_derived": "Von Mises Stress (Derived)",
    "solution_mag": "Displacement Magnitude",
    "solution": "Displacement Vector",
    "energy": "Strain Energy Density",
    "energy_avg": "Strain Energy Density (Element Avg)",
    "growth": "Growth Field (theta)",
    "growth_gradient": "Growth Field Gradient",
    "growth_reaction": "Growth Driving Force (Reaction)",
    "growth_multiplier": "Growth Active-Set Multiplier",
    "growth_pinned": "Growth Pinned Mask",
    "growth_activation_step": "Growth Activation Step",
    "cauchy_eigenvalues": "Cauchy Stress Principal Values",
    "pk2_eigenvalues": "Second Piola-Kirchhoff Stress Principal Values",
    "principal_stretches": "Principal Stretches",
    "green_lagrange_eigenvalues": "Green-Lagrange Strain Principal Values",
    "infinitesimal_strain_eigenvalues":
        "Infinitesimal Strain Principal Values",
    "hydrostatic_stress": "Hydrostatic Stress",
    "max_shear_stress": "Maximum Shear Stress",
    "stress_triaxiality": "Stress Triaxiality",
    "stress_J2": "Second Deviatoric Stress Invariant (J2)",
    "stress_J3": "Third Deviatoric Stress Invariant (J3)",
    "J": "Deformation Volume Ratio (J)",
    "cauchy_trace": "Cauchy Stress Trace",
    "F_mat": "Deformation Gradient Tensor",
    "right_cauchy_green": "Right Cauchy-Green Tensor",
    "left_cauchy_green": "Left Cauchy-Green Tensor",
    "right_stretch": "Right Stretch Tensor",
    "left_stretch": "Left Stretch Tensor",
    "cauchy_mat": "Cauchy Stress Tensor",
    "deviatoric_stress": "Deviatoric Cauchy Stress Tensor",
    "pk1": "First Piola-Kirchhoff Stress Tensor",
    "pk2": "Second Piola-Kirchhoff Stress Tensor",
    "green_lagrange_strain": "Green-Lagrange Strain Tensor",
    "almansi_strain": "Almansi Strain Tensor",
    "hencky_strain": "Logarithmic (Hencky) Strain Tensor",
    "infinitesimal_strain": "Infinitesimal Strain Tensor",
}

BLOCK_LABELS = {
    "Volume": "Volume",
    "Surface": "Surface (Normals, Sidesets, and Traction)",
    "Contact": "Contact (Contact and Friction Forces)",
    "Points": "Points",
}

DERIVED_FIELD_REQUIREMENTS = {
    "solution_mag": "solution",
    "F_mat": "deformation_gradient",
    "J": "deformation_gradient",
    "right_cauchy_green": "deformation_gradient",
    "right_cauchy_green_eigenvalues": "deformation_gradient",
    "left_cauchy_green": "deformation_gradient",
    "left_cauchy_green_eigenvalues": "deformation_gradient",
    "right_stretch": "deformation_gradient",
    "left_stretch": "deformation_gradient",
    "principal_stretches": "deformation_gradient",
    "green_lagrange_strain": "deformation_gradient",
    "green_lagrange_eigenvalues": "deformation_gradient",
    "almansi_strain": "deformation_gradient",
    "almansi_strain_eigenvalues": "deformation_gradient",
    "hencky_strain": "deformation_gradient",
    "hencky_strain_eigenvalues": "deformation_gradient",
    "infinitesimal_strain": "deformation_gradient",
    "infinitesimal_strain_eigenvalues": "deformation_gradient",
    "cauchy_mat": "stress_derived",
    "cauchy_eigenvalues": "stress_derived",
    "cauchy_trace": "stress_derived",
    "hydrostatic_stress": "stress_derived",
    "deviatoric_stress": "stress_derived",
    "stress_J2": "stress_derived",
    "stress_J3": "stress_derived",
    "von_mises_derived": "stress_derived",
    "max_shear_stress": "stress_derived",
    "stress_triaxiality": "stress_derived",
    "pk1": "stress_derived",
    "pk2": "stress_derived",
    "pk2_eigenvalues": "stress_derived",
}

GLYPH_FIELDS = (
    ("right_cauchy_green", "Right Cauchy-Green Tensor",
     "deformation_gradient"),
    ("left_cauchy_green", "Left Cauchy-Green Tensor",
     "deformation_gradient"),
    ("green_lagrange", "Green-Lagrange Strain Tensor",
     "deformation_gradient"),
    ("pk2", "Second Piola-Kirchhoff Stress Tensor", "stress_derived"),
    ("cauchy", "Cauchy Stress Tensor", "stress_derived"),
)

PRINCIPAL_VALUE_FIELDS = {
    "principal_stretches",
    "right_cauchy_green_eigenvalues", "left_cauchy_green_eigenvalues",
    "green_lagrange_eigenvalues", "almansi_strain_eigenvalues",
    "hencky_strain_eigenvalues", "infinitesimal_strain_eigenvalues",
    "cauchy_eigenvalues", "pk2_eigenvalues",
}

# Euler angles (degrees, Houdini XYZ rotation order) taking the local XY (+Y up)
# scene legend into each of the 12 (plane, up) orientations.
LEGEND_ROTATIONS = (
    (0, 0, -90), (0, 0, 90), (0, 0, 0), (0, 0, 180),
    (0, 90, 0), (0, 90, 180), (90, 0, 90), (-90, 0, -90),
    (0, -90, -90), (0, 90, 90), (90, 0, 0), (-90, 0, 0),
)

# Block/field metadata for the whole sequence, keyed on the PVD file
# identity. Underlying per-vtu scans live in the embedded parser's own
# caches; this just memoizes the assembled sequence view.
_SEQUENCE_METADATA_CACHE = {}
_FIBER_COMPANION_CACHE = {}
_FIBER_POINT_CACHE = {}
FIBER_COMPANION_NAME = "polyfem_fiber_families.npz"


# Houdini attribute names admit only [A-Za-z0-9_] and must not start with a
# digit, but PolyFEM's material fields are namespaced with slashes
# ("MaterialSum/HGODispersion/kappa"). Uploading those verbatim makes
# addAttrib raise and the whole frame fail to load, so every vtu name is
# sanitized on the way in and mapped back for display.
_ATTRIB_NAME_RE = re.compile(r"[^0-9A-Za-z_]")
_COMPANION_NAME_RE = re.compile(r"hda_g\d+_v\d+_f\d+(?:_[pm])?_fiber_direction")

# sanitized attribute name -> original vtu array name (only when they differ)
_DISPLAY_NAMES = {}
# original vtu array name -> sanitized attribute name (collision-stable)
_ATTRIB_NAMES = {}
_FIELD_NAME_OWNERS = {}
_MECHANICS_INPUT_NAMES = {
    "F_1", "F_2", "F_3", "solution", "body_ids", "von_mises",
    "von_mises_avg", "_volume",
}
for _stress in ("cauchy_stress", "pk1_stress", "pk2_stress"):
    _MECHANICS_INPUT_NAMES.update(
        {_stress, _stress + "_avg",
         *(f"{_stress}{avg}_{i}" for avg in ("", "_avg") for i in (1, 2, 3))})
_RESERVED_FIELD_NAMES = {
    "P", "Pw", "Cd", "Alpha", "N", "uv", "v", "id", "name", "orient",
    "pscale", "scale", "up", "width",
    "rest", "for_color", "coincident_id", "coincident_rep", "pvd_block",
    "source_cell_index", "source_cell_family", "reference_comparison",
    "green_lagrange", "almansi", "hencky", "infinitesimal", "cauchy",
    *DERIVED_FIELD_REQUIREMENTS,
}
_PROTECTED_FIELD_NAMES = _RESERVED_FIELD_NAMES | _MECHANICS_INPUT_NAMES


def _field_name(raw_name):
    """Alias, then sanitize a vtu array name into a legal Houdini attrib name.

    Deterministic and collision-stable: distinct raw names that sanitize to
    the same string get _2, _3, ... suffixes in first-seen order.
    """
    raw_name = FIELD_ALIASES.get(raw_name, raw_name)
    cached = _ATTRIB_NAMES.get(raw_name)
    if cached is not None:
        return cached
    name = _ATTRIB_NAME_RE.sub("_", raw_name)
    if not name or name[0].isdigit():
        name = "_" + name
    if name.startswith("readpvd_"):
        name = "vtu_" + name
    if ((name in _PROTECTED_FIELD_NAMES and raw_name not in _MECHANICS_INPUT_NAMES)
            or _COMPANION_NAME_RE.fullmatch(name)
            or name in _FIELD_NAME_OWNERS):
        suffix = 2
        while (f"{name}_{suffix}" in _FIELD_NAME_OWNERS
               or f"{name}_{suffix}" in _PROTECTED_FIELD_NAMES):
            suffix += 1
        name = f"{name}_{suffix}"
    if name != raw_name:
        _DISPLAY_NAMES[name] = raw_name
    _ATTRIB_NAMES[raw_name] = name
    _FIELD_NAME_OWNERS[name] = raw_name
    return name


def _display_name(name):
    """Original vtu array name for a sanitized attribute name."""
    return _DISPLAY_NAMES.get(name, name)


def _field_label(name):
    """Human label for a field: curated label, else its original vtu name."""
    if name in FIELD_LABELS:
        return FIELD_LABELS[name]
    display = _display_name(name)
    if display != name:
        return display
    return name.replace("_", " ").title()


_FIBER_SUFFIX = "fiber_direction_x"


def _fiber_prefixes(fields):
    """Namespace prefixes of every complete fiber_direction triple.

    The prefix depends on the formulation -- bare for a single fiber material,
    "<Model>/" inside a MaterialSum, "MaterialSum/<Model>/" once several
    bodies force MultiModels, plus "<Model>_<i>/" for repeated families --
    so match on the suffix and group by whatever comes before it.
    """
    prefixes = []
    for name in fields:
        if not name.endswith(_FIBER_SUFFIX):
            continue
        prefix = name[:-len(_FIBER_SUFFIX)]
        if all(f"{prefix}fiber_direction_{axis}" in fields for axis in "yz"):
            prefixes.append(prefix)
    return sorted(prefixes)


def _fiber_attrib(prefix, raw=False):
    """Vector attribute assembled from one prefix's fiber_direction triple.

    Callers reach this from two name spaces: cook_frame works on raw vtu
    names (raw=True, so they still need sanitizing), while the menu/parameter
    side works on the already-sanitized field names. Sanitizing an
    already-sanitized prefix a second time would register it as a *different*
    original name and earn a collision suffix, so only do it once.
    """
    if raw:
        return _field_name(prefix + "fiber_direction")
    return prefix + "fiber_direction"


def _fiber_companion_path(pvd_path):
    return os.path.join(
        os.path.dirname(os.path.abspath(pvd_path)), FIBER_COMPANION_NAME)


def _load_fiber_companion(pvd_path):
    """Validated fiber families exported beside the PolyFEM PVD collection."""
    path = _fiber_companion_path(pvd_path)
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_mtime_ns, stat.st_size)
    cached = _FIBER_COMPANION_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with np.load(path, allow_pickle=False) as archive:
            version = int(np.asarray(archive["schema_version"]).ravel()[0])
            if version >= 2 and "direction_frame" in archive:
                direction_frame = str(
                    np.asarray(archive["direction_frame"]).ravel()[0])
            else:
                # Schema 1 used the same PolyFEM convention but did not record
                # it explicitly.
                direction_frame = "simulation_world_reference_a0"
            result = {
                "version": version,
                "direction_frame": direction_frame,
                "centroids": np.asarray(
                    archive["centroids"], dtype=np.float64),
                "body_ids": np.asarray(
                    archive["body_ids"], dtype=np.int64).ravel(),
                "family_names": [
                    str(value) for value in archive["family_names"].tolist()],
                "family_labels": [
                    str(value) for value in archive["family_labels"].tolist()],
                "family_body_ids": np.asarray(
                    archive["family_body_ids"], dtype=np.int64).ravel(),
                "directions": np.asarray(
                    archive["directions"], dtype=np.float64),
            }
    except (OSError, KeyError, ValueError, IndexError):
        return None
    count = len(result["centroids"])
    families = len(result["family_names"])
    valid = (
        result["version"] in (1, 2)
        and result["direction_frame"] == "simulation_world_reference_a0"
        and result["centroids"].shape == (count, 3)
        and len(result["body_ids"]) == count
        and result["directions"].shape == (families, count, 3)
        and len(result["family_labels"]) == families
        and len(result["family_body_ids"]) == families)
    if not valid:
        return None
    # Keep established family names and disambiguate names that sanitize to
    # the same attribute, without sharing the special menu token "all".
    used = set()
    for i, raw_name in enumerate(result["family_names"]):
        base = (raw_name if _COMPANION_NAME_RE.fullmatch(raw_name) else
                "readpvd_companion_" + (_ATTRIB_NAME_RE.sub("_", raw_name) or "fiber"))
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        result["family_names"][i] = name
        used.add(name)
    if len(_FIBER_COMPANION_CACHE) > 8:
        _FIBER_COMPANION_CACHE.clear()
    _FIBER_COMPANION_CACHE[key] = result
    return result


def _companion_family_names(pvd_path):
    companion = _load_fiber_companion(pvd_path)
    return companion["family_names"] if companion is not None else []


def _nearest_rows(source, targets):
    """Index of the nearest source row for each target, without all-pairs RAM.

    Houdini's Python does not necessarily include scipy.  The previous fallback
    allocated a ``chunk x len(source) x 3`` array and still performed O(N*M)
    work; a 36k-element companion projected onto 144k PVD points could make
    Houdini disappear under memory pressure.  Equal centroid sets are matched
    by sorting, and genuinely different sets use a bounded uniform-grid search.
    """
    source = np.asarray(source, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not len(source):
        raise ValueError("Cannot project fibers from an empty centroid array")
    if not len(targets):
        return np.empty(0, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
        return cKDTree(source).query(targets)[1]
    except ImportError:
        pass

    extent = float(np.linalg.norm(np.ptp(source, axis=0)))
    tolerance = max(1e-9, 1e-4 * extent)

    # PolyFEM's visualization cells normally retain global element order.
    if len(source) == len(targets):
        row_distance = np.linalg.norm(source - targets, axis=1)
        if len(row_distance) and float(row_distance.max()) <= tolerance:
            return np.arange(len(targets), dtype=np.int64)

        # Also handle the same centroid set in a different element order.
        source_order = np.lexsort(
            (source[:, 2], source[:, 1], source[:, 0]))
        target_order = np.lexsort(
            (targets[:, 2], targets[:, 1], targets[:, 0]))
        sorted_distance = np.linalg.norm(
            source[source_order] - targets[target_order], axis=1)
        if len(sorted_distance) and float(sorted_distance.max()) <= tolerance:
            result = np.empty(len(targets), dtype=np.int64)
            result[target_order] = source_order
            return result

    # Dependency-free spatial fallback for sampled/high-order output whose cell
    # centroids differ from the preprocessing mesh.  Bins hold ~8 source rows
    # on average; queries expand only until the current best point is closer
    # than the boundary of the searched cube.
    low = source.min(axis=0)
    spans = np.ptp(source, axis=0)
    active_dim = max(1, int(np.count_nonzero(spans > 1e-12)))
    bins_linear = max(
        1, int(round((len(source) / 8.0) ** (1.0 / active_dim))))
    cell_size = max(float(spans.max()) / bins_linear, 1e-12)
    source_keys = np.floor((source - low) / cell_size).astype(np.int64)
    buckets = {}
    for index, key in enumerate(source_keys):
        buckets.setdefault(tuple(int(value) for value in key), []).append(index)

    source_key_min = source_keys.min(axis=0)
    source_key_max = source_keys.max(axis=0)
    result = np.empty(len(targets), dtype=np.int64)
    for target_index, target in enumerate(targets):
        query = np.floor((target - low) / cell_size).astype(np.int64)
        max_radius = int(np.max(np.maximum(
            np.abs(query - source_key_min),
            np.abs(query - source_key_max)))) + 1
        best_index = -1
        best_squared = np.inf
        for radius in range(max_radius + 1):
            for offset in itertools.product(
                    range(-radius, radius + 1), repeat=3):
                if radius and max(abs(value) for value in offset) != radius:
                    continue
                key = tuple(int(query[axis] + offset[axis])
                            for axis in range(3))
                candidates = buckets.get(key)
                if not candidates:
                    continue
                candidate_array = np.asarray(candidates, dtype=np.int64)
                delta = source[candidate_array] - target
                squared = np.einsum("ij,ij->i", delta, delta)
                local = int(np.argmin(squared))
                if float(squared[local]) < best_squared:
                    best_squared = float(squared[local])
                    best_index = int(candidate_array[local])

            if best_index >= 0:
                lower = low + (query - radius) * cell_size
                upper = low + (query + radius + 1) * cell_size
                outside_distance = float(np.min(np.concatenate(
                    (target - lower, upper - target))))
                if outside_distance >= 0 \
                        and best_squared <= outside_distance * outside_distance:
                    break
        result[target_index] = best_index
    return result


def _companion_point_fibers(pvd_path, mesh):
    """Project reference element directions onto the displayed PVD points."""
    companion = _load_fiber_companion(pvd_path)
    if companion is None:
        return []
    companion_path = _fiber_companion_path(pvd_path)
    try:
        companion_stat = os.stat(companion_path)
        companion_identity = (
            companion_stat.st_mtime_ns, companion_stat.st_size)
    except OSError:
        companion_identity = (0, 0)
    cache_key = (
        companion_path, companion_identity, str(mesh.get("topo_key")),
        _rest_geometry_digest(mesh))
    cached = _FIBER_POINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    points = np.asarray(mesh["points"], dtype=np.float64)
    point_bodies = mesh["point_data"].get("body_ids")
    if point_bodies is not None and len(point_bodies) == len(points):
        point_bodies = np.asarray(point_bodies).reshape(-1)
    else:
        point_bodies = None

    # Prefer element topology over point-to-centroid nearest neighbours.
    # PolyFEM's discontinuous PVD mesh has one visualization cell per solver
    # element in the common case, so the 36k companion rows in a 144k-point tet
    # output map in O(elements), with no 36k x 144k distance problem.
    cell_blocks = [
        np.asarray(connectivity, dtype=np.int64)
        for connectivity in mesh.get("cells", {}).values()
        if len(connectivity)]
    cell_to_source = None
    if cell_blocks:
        cell_centroids = np.concatenate(
            [points[connectivity].mean(axis=1)
             for connectivity in cell_blocks], axis=0)
        cell_to_source = _nearest_rows(
            companion["centroids"], cell_centroids)

    result = []
    for name, label, body_id, directions in zip(
            companion["family_names"], companion["family_labels"],
            companion["family_body_ids"], companion["directions"]):
        norms = np.linalg.norm(directions, axis=1)
        source_mask = (companion["body_ids"] == int(body_id)) & (norms > 1e-12)
        if not np.any(source_mask):
            continue
        target_mask = np.ones(len(points), dtype=bool) if point_bodies is None \
            else point_bodies == int(body_id)
        if not np.any(target_mask):
            continue
        values = np.zeros((len(points), 3), dtype=np.float64)
        if cell_to_source is not None:
            projected = np.asarray(
                directions[cell_to_source], dtype=np.float64)
            projected_norms = np.linalg.norm(projected, axis=1)
            projected_bodies = companion["body_ids"][cell_to_source]
            live_cells = (projected_bodies == int(body_id)) \
                & (projected_norms > 1e-12)
            projected[live_cells] /= projected_norms[live_cells, None]

            begin = 0
            for connectivity in cell_blocks:
                end = begin + len(connectivity)
                local_live = live_cells[begin:end]
                if np.any(local_live):
                    corners = connectivity[local_live].ravel()
                    vectors = np.repeat(
                        projected[begin:end][local_live],
                        connectivity.shape[1], axis=0)
                    values[corners] = vectors
                begin = end
        else:
            # Point-only blocks have no topology to scatter through. The
            # bounded spatial matcher is safe here even without scipy.
            source_points = companion["centroids"][source_mask]
            source_values = directions[source_mask]
            source_values = source_values / np.linalg.norm(
                source_values, axis=1)[:, None]
            nearest = _nearest_rows(source_points, points[target_mask])
            values[target_mask] = source_values[nearest]
        if point_bodies is not None:
            values[point_bodies != int(body_id)] = 0.0
        result.append((name, label, values))

    if len(_FIBER_POINT_CACHE) > 16:
        _FIBER_POINT_CACHE.clear()
    _FIBER_POINT_CACHE[cache_key] = result
    return result


def fiber_family_menu(kwargs):
    node = kwargs["node"]
    output = node.node("output")
    families = []
    if output is not None:
        try:
            families = output.geometry().attribValue(
                "readpvd_fiber_families").split()
        except hou.OperationFailed:
            families = []
    if not families:
        return ["all", "No Fiber Fields Found"]
    result = ["all", "All Families"]
    for name in families:
        result.extend((name, _field_label(name)))
    return result


def _fields_from_info(info):
    """Flatten a field-info dict into {name: components}.

    PointData wins over same-named CellData (cell fields are point-averaged
    for display only). Only renderable component counts are kept.
    """
    fields = {}
    for raw_name, components in info.get("point_data", {}).items():
        if components in (1, 2, 3, 9):
            fields[_field_name(raw_name)] = components
    for raw_name, components in info.get("cell_data", {}).items():
        if components in (1, 2, 3, 9):
            fields.setdefault(_field_name(raw_name), components)
    return fields


def _set_floats(setter, name, arr):
    """Bulk-upload a float attribute from raw float64 bytes (no Python list)."""
    setter(name, np.ascontiguousarray(arr, dtype=np.float64).tobytes(),
           float_type=hou.numericData.Float64)


def _message(text):
    if hou.isUIAvailable():
        hou.ui.displayMessage(str(text))
    else:
        print(f"[readPVD] {text}")


def _asset_of(node):
    return node.parent()


def _menu_token(node, name):
    parm = node.parm(name)
    return parm.evalAsString() if parm is not None else ""


def _format_number(value, digits, notation="automatic"):
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    digits = max(1, int(digits))
    if notation == "scientific":
        return f"{value:.{digits - 1}e}"
    if notation == "fixed":
        return f"{value:.{digits}f}"
    if notation == "engineering" and value != 0:
        exponent = int(math.floor(math.log10(abs(value)) / 3.0) * 3)
        return f"{value / (10 ** exponent):.{digits - 1}f}e{exponent:+d}"
    return f"{value:.{digits}g}"


def _source_block_token(node):
    value = node.evalParm("source_block")
    if isinstance(value, str):
        return value
    return ("Volume", "Surface", "Contact", "Points")[
        max(0, min(int(value), 3))]


