# readPVD 1.0 -- PolyFEM ParaView result viewer.
#
# Architecture (built for >1M element meshes):
#   topo_build  (python, time-independent) parse ONE frame's topology ->
#               bulk points + detail connectivity arrays + topo_key
#   build_prims (VEX detail wrangle)       tets/hexes/tris from arrays
#   boundary    (VEX prim wrangle)         display surface (volume prims
#               removed; runs only when topology changes)
#   frame_data  (python, time-dependent)   per-frame BULK upload of P and
#               every PointData field (generic; polyfem's 'stess' typo
#               aliased); topology rebuild only when topo_key changes
#   deform/derived/glyphs/color            VEX (+ optional OpenCL eigen)
#
# Per-frame cost = vtu parse + attribute upload only.

import math
import os
import json
import re
import itertools

import numpy as np

import hou

# ---- embedded native parser (built from src/common/vtu_parser.py) ----------
# @VTU_PARSER@
# ---- end embedded parser ----------------------------------------------------

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

LEGEND_ROTATIONS = (
    (0, 0, -90), (0, 0, 90), (0, 0, 0), (0, 0, 180),
    (0, 90, 0), (0, 90, 180), (0, 90, -90), (0, 90, 90),
    (90, 0, -90), (90, 0, 90), (90, 0, 0), (-90, 0, 0),
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

# sanitized attribute name -> original vtu array name (only when they differ)
_DISPLAY_NAMES = {}
# original vtu array name -> sanitized attribute name (collision-stable)
_ATTRIB_NAMES = {}


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
    if _DISPLAY_NAMES.get(name, raw_name) != raw_name:
        suffix = 2
        while _DISPLAY_NAMES.get(f"{name}_{suffix}", raw_name) != raw_name:
            suffix += 1
        name = f"{name}_{suffix}"
    if name != raw_name:
        _DISPLAY_NAMES[name] = raw_name
    _ATTRIB_NAMES[raw_name] = name
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
        len(mesh["points"]))
    cached = _FIBER_POINT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    points = np.asarray(mesh["points"], dtype=np.float64)
    point_bodies = mesh["point_data"].get("body_ids")
    if point_bodies is not None and len(point_bodies) == len(points):
        point_bodies = np.asarray(point_bodies).reshape(-1).astype(np.int64)
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


def _frame_metadata(node):
    """Renderable field metadata for the active frame (no array decode)."""
    path = node.evalParm("PVD_file")
    if not path:
        return {}
    frame = int(hou.frame())
    try:
        info = frame_field_info(path, frame)
    except Exception:
        return {}
    return {block: _fields_from_info(block_info)
            for block, block_info in info.items()}


def _sequence_metadata(node):
    """Per-frame field metadata for the whole sequence (no array decode).

    Returns (frames, block_counts) where frames[i] maps block -> fields and
    block_counts maps block -> number of frames it appears in. Cached on the
    PVD's file identity; the underlying scans are byte-level, so a full
    sequence costs milliseconds even for million-element meshes.
    """
    path = node.evalParm("PVD_file")
    if not path:
        return [], {}
    try:
        key = _file_key(path)
        entries = read_pvd(path)
    except Exception:
        return [], {}
    cached = _SEQUENCE_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    frames = []
    for frame in range(len(entries)):
        try:
            info = frame_field_info(path, frame)
        except Exception:
            frames.append({})
            continue
        frames.append({block: _fields_from_info(block_info)
                       for block, block_info in info.items()})
    block_counts = {}
    for frame_meta in frames:
        for block_name in frame_meta:
            block_counts[block_name] = block_counts.get(block_name, 0) + 1
    result = (frames, block_counts)
    if len(_SEQUENCE_METADATA_CACHE) > 8:
        _SEQUENCE_METADATA_CACHE.clear()
    _SEQUENCE_METADATA_CACHE[key] = result
    return result


def _selected_fields(node):
    metadata = _frame_metadata(node)
    if not metadata:
        return "", {}
    selected = _source_block_token(node)
    if selected not in metadata:
        selected = next(iter(metadata))
    return selected, metadata[selected]


def _requirements(fields):
    has_f = all(name in fields for name in ("F_1", "F_2", "F_3"))
    has_cauchy = all(
        name in fields for name in
        ("cauchy_stress_1", "cauchy_stress_2", "cauchy_stress_3"))
    # Cauchy stress can be reconstructed from the 1st PK stress and F, so
    # exporting either (with F) unlocks the full derived-stress set.
    has_pk1 = all(
        name in fields for name in
        ("pk1_stress_1", "pk1_stress_2", "pk1_stress_3"))
    return {
        "solution": "solution" in fields,
        "deformation_gradient": has_f,
        "stress_derived": has_f and (has_cauchy or has_pk1),
    }


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
    try:
        blocks = load_frame(path, int(hou.frame()))
    except Exception:
        return []
    mesh = blocks.get(_source_block_token(node)) or next(
        iter(blocks.values()), None)
    if mesh is None:
        return []
    body = mesh["point_data"].get("body_ids")
    if body is None:
        return []
    return sorted(int(value) for value in np.unique(np.asarray(body)))


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


def _first_color_field(node):
    tokens = _menu_tokens(color_field_menu({"node": node}))
    for preferred in ("von_mises", "solution_mag", "solution"):
        if preferred in tokens:
            return preferred
    return tokens[0] if tokens and tokens[0] else ""


def sync_available_options(kwargs):
    """Keep all user-selectable data choices valid for the active block."""
    node = kwargs["node"]
    metadata = _frame_metadata(node)
    if not metadata and node.evalParm("PVD_file") \
            and node.evalParm("source_block"):
        # Transient empty read (a running sim mid-write on this frame's
        # files): leave every selection parm exactly as the user set it
        # instead of resetting block/color/visibility to defaults.
        node.parm("availability_status").set(
            "Current frame unreadable (sim writing?); keeping settings.")
        return
    block_tokens = _menu_tokens(source_block_menu({"node": node}))
    selected_block = _source_block_token(node)
    if selected_block not in block_tokens:
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
        if not present and node.parm(f"multi_{slug}_show") is not None:
            node.parm(f"multi_{slug}_show").set(0)
    node.setParms(flags)
    if not has_solution:
        node.parm("show_deformed").set(0)
    if not has_glyph:
        node.parm("add_glyphs").set(0)
    if not fiber_attribs:
        node.parm("show_fibers").set(0)
    elif not requirements["deformation_gradient"] \
            and _menu_token(node, "fiber_frame") != "reference":
        # The deformed direction needs F; fall back rather than draw the
        # reference direction while claiming it is deformed.
        node.parm("fiber_frame").set("reference")
        _message("Fibers: this run has no deformation gradient; "
                 "showing the reference direction.")
    # Drop visible-body selections that no longer exist (only when this block
    # actually carries body ids, so switching to a body-less block keeps them).
    if body_ids and node.parm("visible_bodies") is not None:
        valid = {str(body) for body in body_ids}
        current = node.evalParm("visible_bodies").split()
        kept = [token for token in current if token in valid]
        if kept != current:
            node.parm("visible_bodies").set(" ".join(kept))

    renderable_count = max(0, len(color_tokens) - (1 if color_tokens == [""] else 0))
    block_label = BLOCK_LABELS.get(selected_block, selected_block or "None")
    summary = (
        f"{block_label}: {renderable_count} renderable point field"
        f"{'s' if renderable_count != 1 else ''}; "
        f"{len(block_tokens) if block_tokens != [''] else 0} available block"
        f"{'s' if len(block_tokens) != 1 else ''}.")
    node.parm("availability_status").set(summary)


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
        blocks = load_frame(path, 0)
    except Exception:
        return
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for mesh in blocks.values():
        points = np.asarray(mesh.get("points", ()), dtype=np.float64)
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
    sync_available_options(kwargs)
    update_color_status(kwargs)


def color_selection_changed(kwargs):
    sync_available_options(kwargs)
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


def _mesh_point_field(mesh, field):
    for raw_name, data in mesh["point_data"].items():
        if FIELD_ALIASES.get(raw_name, raw_name) == field:
            array = np.asarray(data, dtype=np.float64)
            if len(array) == len(mesh["points"]):
                return array
    for raw_name, by_family in mesh.get("cell_data", {}).items():
        if FIELD_ALIASES.get(raw_name, raw_name) != field:
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


def _reduce_array(values, reduction):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim == 3 or (values.ndim == 2 and values.shape[1] == 9):
        matrices = values.reshape(-1, 3, 3)
        if reduction == "trace":
            return np.trace(matrices, axis1=1, axis2=2)
        if reduction == "determinant":
            return np.linalg.det(matrices)
        if reduction in ("principal_max", "principal_middle", "principal_min"):
            eigenvalues = np.linalg.eigvalsh(
                0.5 * (matrices + np.swapaxes(matrices, 1, 2)))[:, ::-1]
            return eigenvalues[:, {
                "principal_max": 0, "principal_middle": 1,
                "principal_min": 2}[reduction]]
        if reduction in ("x", "y", "z"):
            index = {"x": 0, "y": 1, "z": 2}[reduction]
            return matrices[:, index, index]
        return np.linalg.norm(matrices, axis=(1, 2))
    if reduction in ("principal_max", "principal_middle", "principal_min"):
        return values[:, {
            "principal_max": 0, "principal_middle": 1,
            "principal_min": 2}[reduction]]
    if reduction in ("x", "y", "z"):
        index = {"x": 0, "y": 1, "z": 2}[reduction]
        return values[:, min(index, values.shape[1] - 1)]
    return np.linalg.norm(values, axis=1)


def _mesh_field(mesh, field):
    raw = _mesh_point_field(mesh, field)
    if raw is not None:
        return raw
    point_data = {
        FIELD_ALIASES.get(name, name): np.asarray(values, dtype=np.float64)
        for name, values in mesh["point_data"].items()
    }
    solution = point_data.get("solution")
    if field == "solution_mag" and solution is not None:
        return np.linalg.norm(solution, axis=1)
    if not all(name in point_data for name in ("F_1", "F_2", "F_3")):
        return None
    # PolyFEM flattens tensors column-major: the X_i arrays are the COLUMNS
    # of the tensor (stack on axis=2), not the rows.
    F = np.stack((point_data["F_1"], point_data["F_2"],
                  point_data["F_3"]), axis=2)
    identity = np.eye(3)
    jacobian = np.linalg.det(F)
    # Right/Left Cauchy-Green deformation tensors and their principal
    # (Lagrangian / Eulerian) axes, reused for the stretch and strain measures
    # that share those axes.
    C = np.einsum("nji,njk->nik", F, F)         # F^T F  (right, Lagrangian)
    B = np.einsum("nij,nkj->nik", F, F)         # F F^T  (left, Eulerian)
    c_vals, c_vecs = np.linalg.eigh(C)          # ascending
    b_vals, b_vecs = np.linalg.eigh(B)
    lam = np.sqrt(np.clip(c_vals, 0, None))     # principal stretches
    right_stretch = np.einsum("nij,nj,nkj->nik", c_vecs, lam, c_vecs)
    left_stretch = np.einsum(
        "nij,nj,nkj->nik", b_vecs, np.sqrt(np.clip(b_vals, 0, None)), b_vecs)
    hencky = np.einsum(
        "nij,nj,nkj->nik", c_vecs, np.log(np.clip(lam, 1e-20, None)), c_vecs)
    hencky[np.abs(jacobian) <= 1e-12] = 0.0     # ln(U) undefined at J = 0
    green = 0.5 * (C - identity)
    # Almansi strain needs B^-1, defined only where B is invertible (J != 0);
    # zero it elsewhere so it matches the guarded VEX display path.
    almansi = 0.5 * (identity - np.linalg.pinv(B))
    almansi[np.abs(jacobian) <= 1e-12] = 0.0
    infinitesimal = 0.5 * (F + np.swapaxes(F, 1, 2)) - identity
    derived = {
        "F_mat": F, "J": jacobian,
        "right_cauchy_green": C,
        "right_cauchy_green_eigenvalues": c_vals[:, ::-1],
        "left_cauchy_green": B,
        "left_cauchy_green_eigenvalues": b_vals[:, ::-1],
        "right_stretch": right_stretch, "left_stretch": left_stretch,
        "principal_stretches": lam[:, ::-1],
        "green_lagrange_strain": green,
        "green_lagrange_eigenvalues": np.linalg.eigvalsh(green)[:, ::-1],
        "almansi_strain": almansi,
        "almansi_strain_eigenvalues": np.linalg.eigvalsh(almansi)[:, ::-1],
        "hencky_strain": hencky,
        "hencky_strain_eigenvalues": np.linalg.eigvalsh(hencky)[:, ::-1],
        "infinitesimal_strain": infinitesimal,
        "infinitesimal_strain_eigenvalues":
            np.linalg.eigvalsh(infinitesimal)[:, ::-1],
    }
    # Cauchy stress from the exported field, or reconstructed from the 1st
    # Piola-Kirchhoff stress (sigma = (1/J) P F^T) when only that was written.
    cauchy = None
    if all(name in point_data for name in (
            "cauchy_stress_1", "cauchy_stress_2", "cauchy_stress_3")):
        cauchy = np.stack((point_data["cauchy_stress_1"],
                           point_data["cauchy_stress_2"],
                           point_data["cauchy_stress_3"]), axis=2)
    elif all(name in point_data for name in (
            "pk1_stress_1", "pk1_stress_2", "pk1_stress_3")):
        pk1_tensor = np.stack((point_data["pk1_stress_1"],
                               point_data["pk1_stress_2"],
                               point_data["pk1_stress_3"]), axis=2)
        scale = np.divide(1.0, jacobian, out=np.zeros_like(jacobian),
                          where=np.abs(jacobian) > 1e-12)
        cauchy = scale[:, None, None] * np.matmul(
            pk1_tensor, np.swapaxes(F, 1, 2))
    if cauchy is not None:
        trace = np.trace(cauchy, axis1=1, axis2=2)
        hydro = trace / 3.0
        dev = cauchy - hydro[:, None, None] * identity
        j2 = 0.5 * np.sum(dev * dev, axis=(1, 2))
        eigen = np.linalg.eigvalsh(cauchy)[:, ::-1]
        Finv = np.linalg.pinv(F)
        pk1 = jacobian[:, None, None] * np.einsum(
            "nij,njk->nik", cauchy, np.swapaxes(Finv, 1, 2))
        pk2 = jacobian[:, None, None] * np.einsum(
            "nij,njk,nlk->nil", Finv, cauchy, Finv)
        derived.update({
            "cauchy_mat": cauchy, "cauchy_eigenvalues": eigen,
            "cauchy_trace": trace, "hydrostatic_stress": hydro,
            "deviatoric_stress": dev, "stress_J2": j2,
            "stress_J3": np.linalg.det(dev),
            "von_mises_derived": np.sqrt(np.maximum(0, 3 * j2)),
            "max_shear_stress": 0.5 * (eigen[:, 0] - eigen[:, 2]),
            "stress_triaxiality": np.divide(
                hydro, np.sqrt(np.maximum(0, 3 * j2)),
                out=np.zeros_like(hydro), where=j2 > 1e-24),
            "pk1": pk1, "pk2": pk2,
            "pk2_eigenvalues": np.linalg.eigvalsh(
                0.5 * (pk2 + np.swapaxes(pk2, 1, 2)))[:, ::-1],
        })
    return derived.get(field)


def cook_reference_comparison(node):
    geo = node.geometry()
    asset = node.parent()
    if not asset.evalParm("reference_enable"):
        asset.parm("reference_status").set("Comparison is disabled.")
        return
    field = asset.evalParm("color_attrib")
    attrib = geo.findPointAttrib(field)
    if attrib is None:
        asset.parm("reference_status").set(
            f"Current field '{field}' is unavailable.")
        return
    components = attrib.size()
    current = np.frombuffer(
        geo.pointFloatAttribValuesAsString(field), dtype=np.float32)
    if components > 1:
        current = current.reshape(-1, components)
    current = _reduce_array(current, _menu_token(asset, "color_reduction"))
    try:
        blocks = load_frame(asset.evalParm("PVD_file"),
                            asset.evalParm("reference_frame"))
        reference_mesh = blocks.get(_source_block_token(asset))
        reference = _mesh_field(reference_mesh, field) \
            if reference_mesh is not None else None
    except Exception as exc:
        reference = None
        asset.parm("reference_status").set(f"Reference load failed: {exc}")
    if reference is None:
        asset.parm("reference_status").set(
            f"Reference field '{field}' is unavailable.")
        return
    reference = _reduce_array(
        reference, _menu_token(asset, "color_reduction"))
    if len(reference) != len(current):
        asset.parm("reference_status").set(
            "Reference comparison requires matching point topology.")
        return
    delta = current - reference
    mode = _menu_token(asset, "reference_mode")
    if mode == "absolute":
        delta = np.abs(delta)
    elif mode == "percent":
        delta = np.divide(
            delta * 100.0, np.abs(reference), out=np.zeros_like(delta),
            where=np.abs(reference) > 1e-20)
    if geo.findPointAttrib("reference_comparison") is None:
        geo.addAttrib(hou.attribType.Point, "reference_comparison", 0.0,
                      create_local_variable=False)
    geo.setPointFloatAttribValues(
        "reference_comparison",
        np.ascontiguousarray(delta, dtype=np.float64).tolist())
    asset.parm("reference_status").set(
        f"Comparing {field} with frame {asset.evalParm('reference_frame')} "
        f"using {mode}.")


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


def probe_readout(node, point_number):
    """Current position and readout text for a probed point.

    Reads from the final displayed geometry, so calling it again after the
    frame changes returns the point's *new* position and field values. Pure:
    it writes no parameters, so it is safe to call every viewport redraw.
    Returns (position, text) or (None, "") if the point is unavailable.
    """
    output = node.node("output")
    if output is None:
        return None, ""
    geo = output.geometry()
    if point_number < 0 or point_number >= len(geo.points()):
        return None, ""
    point = geo.point(point_number)
    position = point.position()
    lines = [f"Point {point_number}",
             "Position: " + ", ".join(f"{value:.6g}" for value in position)]
    block_attrib = geo.findPointAttrib("pvd_block")
    if block_attrib is not None:
        block = point.attribValue(block_attrib)
        if block:
            lines.append(f"PVD Block: {block}")
    for name in ("solution", node.evalParm("color_attrib"),
                 "reference_comparison", "body_ids", "sidesets",
                 "cauchy_eigenvalues", "pk2_eigenvalues",
                 "principal_stretches"):
        attrib = geo.findPointAttrib(name)
        if attrib is None:
            continue
        value = point.attribValue(attrib)
        if isinstance(value, tuple):
            text = ", ".join(f"{component:.6g}" for component in value)
        else:
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
        units = node.evalParm("field_units").strip() \
            if name in (node.evalParm("color_attrib"),
                        "reference_comparison") else ""
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


def _nice_ticks(low, high, target=5):
    """Tick values at 1/2/2.5/5 x 10^k steps spanning [low, high]."""
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        return np.array([low if np.isfinite(low) else 0.0])
    span = high - low
    magnitude = 10.0 ** math.floor(math.log10(span / max(1, target)))
    step = 10 * magnitude
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if span / step <= target * 1.4:
            break
    start = math.ceil(low / step - 1e-9) * step
    count = int(math.floor((high - start) / step + 1e-9)) + 1
    return start + step * np.arange(max(1, count))


def _diagnostics_reduced(node, mesh, field, reduction):
    """Per-point displayed scalar for one mesh -- mirrors the color VEX."""
    raw = _mesh_field(mesh, field)
    if raw is None:
        return None
    return np.asarray(_reduce_array(raw, reduction), dtype=np.float64)


def _reference_reduced(node, path, block, field, reduction):
    """Reduced reference-frame field, or None when comparison is off/missing."""
    if not node.evalParm("reference_enable"):
        return None
    try:
        blocks = load_frame(path, node.evalParm("reference_frame"))
        mesh = blocks.get(block) or next(iter(blocks.values()), None)
        return _diagnostics_reduced(node, mesh, field, reduction) \
            if mesh is not None else None
    except Exception:
        return None


def _apply_reference(values, reference_reduced, mode):
    """Apply the reference comparison the color VEX would (difference/abs/%)."""
    if reference_reduced is None or len(reference_reduced) != len(values):
        return values
    delta = values - reference_reduced
    if mode == "absolute":
        return np.abs(delta)
    if mode == "percent":
        return np.divide(delta * 100.0, np.abs(reference_reduced),
                         out=np.zeros_like(delta),
                         where=np.abs(reference_reduced) > 1e-20)
    return delta


def _coincidence_groups(points):
    """Return (group id per point, representative flag) for coincident nodes.

    PolyFEM writes per-element duplicated vertices; `inverse` labels which
    physical vertex each point is, and `representative` is 1 for exactly one
    point per physical vertex (used to draw a single glyph per node).
    """
    _, index, inverse = np.unique(
        np.round(points, 9), axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    representative = np.zeros(len(points), dtype=np.float64)
    representative[index] = 1.0
    return inverse, representative


def _coincidence_inverse(points):
    """Group id per point for coincident (duplicated) mesh vertices."""
    return _coincidence_groups(points)[0]


def _nodal_average(values, inverse):
    """Average `values` over coincidence groups (FEM nodal recovery)."""
    finite = np.isfinite(values)
    size = int(inverse.max()) + 1 if len(inverse) else 0
    sums = np.bincount(inverse[finite], weights=values[finite], minlength=size)
    counts = np.bincount(inverse[finite], minlength=size)
    averaged = np.divide(sums, counts, out=np.full(size, np.nan, np.float64),
                         where=counts > 0)
    return averaged[inverse]


# Memoized results of the last all-frames scan, so the two features that read
# every frame (Auto Range: All Frames, Compute Field Over Time) share the work
# and a repeat is instant. Keyed on everything that changes the per-frame
# value; a settings or file change invalidates it. Bounded so a huge sequence
# is recomputed rather than pinned in RAM.
_SCAN_CACHE = {}
_SCAN_CACHE_MAX_BYTES = 2 * 1024 ** 3


def _scan_key(node, path):
    try:
        file_key = _file_key(path)
    except Exception:
        file_key = path
    return (file_key, node.evalParm("color_attrib"),
            _menu_token(node, "color_reduction"),
            bool(node.evalParm("smooth_field")), _source_block_token(node),
            bool(node.evalParm("reference_enable")),
            int(node.evalParm("reference_frame")),
            _menu_token(node, "reference_mode"),
            node.evalParm("visible_bodies").strip())


def _scan_displayed_values(node, path):
    """Yield (frame, timestep, values) of the displayed scalar for every frame,
    reusing the cached result when the scan settings and PVD file are
    unchanged. See _scan_frames for the (uncached) computation."""
    key = _scan_key(node, path)
    if _SCAN_CACHE.get("key") == key:
        for row in _SCAN_CACHE["rows"]:
            yield row
        return
    rows = []
    for frame, timestep, values in _scan_frames(node, path):
        rows.append((frame, timestep,
                     None if values is None else values.astype(np.float32)))
        yield frame, timestep, values
    total = sum(row[2].nbytes for row in rows if row[2] is not None)
    if total <= _SCAN_CACHE_MAX_BYTES:
        _SCAN_CACHE["key"] = key
        _SCAN_CACHE["rows"] = rows
    else:
        _SCAN_CACHE.clear()


def _scan_frames(node, path):
    """Compute (frame, timestep, values) per frame -- pure numpy, an exact
    mirror of the color stage (reduction, reference comparison, smoothing
    `_avg` preference, nodal averaging, body visibility). No network cook and
    no frame change, so the two consumers agree with the viewport.
    """
    entries = read_pvd(path)
    field = node.evalParm("color_attrib")
    reduction = _menu_token(node, "color_reduction")
    block = _source_block_token(node)
    reference_reduced = _reference_reduced(node, path, block, field, reduction)
    reference_mode = _menu_token(node, "reference_mode")
    smooth = bool(node.evalParm("smooth_field"))
    visible = _visible_body_set(node)
    inverse = None
    for frame in range(len(entries)):
        timestep = entries[frame][0]
        try:
            blocks = load_frame(path, frame)
        except Exception:
            yield frame, timestep, None
            continue
        mesh = blocks.get(block) or next(iter(blocks.values()), None)
        if mesh is None:
            yield frame, timestep, None
            continue
        if reference_reduced is not None:
            values = _diagnostics_reduced(node, mesh, field, reduction)
            if values is not None:
                values = _apply_reference(
                    values, reference_reduced, reference_mode)
        else:
            # smoothing prefers PolyFEM's continuous _avg field where present
            effective = field
            if smooth:
                candidate = field + "_avg"
                names = {FIELD_ALIASES.get(name, name)
                         for name in mesh["point_data"]}
                if candidate in names:
                    effective = candidate
            values = _diagnostics_reduced(node, mesh, effective, reduction)
        if values is not None and smooth:
            if inverse is None or len(inverse) != len(values):
                inverse = _coincidence_inverse(mesh["points"])
            values = _nodal_average(values, inverse)
        # Restrict the range/diagnostics to the bodies that are displayed.
        if values is not None and visible is not None:
            body = mesh["point_data"].get("body_ids")
            if body is not None:
                shown = np.isin(np.asarray(body).astype(np.int64).ravel(),
                                list(visible))
                values = np.where(shown, values, np.nan)
        yield frame, timestep, values


def compute_diagnostics(kwargs):
    """Scan the displayed value over every frame using numpy directly.

    No network cook and no frame changes: each frame is parsed once (shared
    cache) and reduced with the same helpers the color VEX uses, so this is
    far faster than force-cooking the result pipeline per frame and never
    disturbs the current frame or playbar.
    """
    node = kwargs["node"]
    path = node.evalParm("PVD_file")
    if not path:
        return
    try:
        scanned = list(_scan_displayed_values(node, path))
    except Exception as exc:
        node.parm("diagnostics_status").set(f"Timeline scan failed: {exc}")
        return
    rows = []
    for frame, timestep, values in scanned:
        if values is None:
            continue
        finite = values[np.isfinite(values)]
        if len(finite):
            rows.append((int(frame), float(timestep), float(finite.min()),
                         float(finite.mean()), float(finite.max())))
    node.parm("diagnostics_data").set(json.dumps(rows))
    node.parm("diagnostics_status").set(
        f"Stored {len(rows)} of {len(scanned)} frames for "
        f"{legend_title_text(node)}.")


# Timeline-plot palette and depth layering (everything is built in the unit
# plot space then transformed; small z offsets avoid coplanar z-fighting).
_DIAG_MIN_COLOR = (0.25, 0.55, 1.0)     # cool blue
_DIAG_MEAN_COLOR = (0.95, 0.8, 0.25)    # amber
_DIAG_MAX_COLOR = (1.0, 0.4, 0.25)      # warm red
_DIAG_AXIS_COLOR = (0.78, 0.80, 0.84)
_DIAG_GRID_COLOR = (0.33, 0.35, 0.40)
_DIAG_ZERO_COLOR = (0.55, 0.57, 0.62)
_DIAG_BAND_COLOR = (0.26, 0.29, 0.36)
_DIAG_Z_BAND, _DIAG_Z_GRID = -0.002, -0.001
_DIAG_Z_AXIS, _DIAG_Z_CURVE = 0.0, 0.001


def _diag_curve(geo, group, xs, ys, color, z):
    prim = geo.createPolygon(is_closed=False)
    for x_value, y_value in zip(xs, ys):
        prim.addVertex(_point(geo, (x_value, y_value, z), color))
    group.add(prim)


def cook_diagnostics(node):
    """Renderable timeline chart: axes, gridlines, numbered ticks with units,
    min/mean/max curves, an optional spread band, and a legend.

    Built in a unit plot box, then transformed by the diagnostics translate
    and scale so a single bulk point transform positions the whole plot.
    """
    geo = node.geometry()
    asset = node.parent()
    try:
        rows = np.asarray(json.loads(asset.evalParm("diagnostics_data")),
                          dtype=np.float64)
    except Exception:
        return
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 5:
        return

    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    curves = geo.createPrimGroup("readpvd_diagnostics")
    chrome = geo.createPrimGroup("readpvd_diagnostics_axes")

    use_time = _menu_token(asset, "diagnostics_x_axis") != "frame"
    x_data = rows[:, 1] if use_time else rows[:, 0]
    y_min, y_mean, y_max = rows[:, 2], rows[:, 3], rows[:, 4]

    digits = int(asset.evalParm("legend_digits"))
    notation = _menu_token(asset, "legend_number_format")

    x_lo, x_hi = float(x_data.min()), float(x_data.max())
    x_ticks = _nice_ticks(x_lo, x_hi, 6)
    x_ticks = x_ticks[(x_ticks >= x_lo - 1e-9) & (x_ticks <= x_hi + 1e-9)]
    data_lo = float(np.nanmin(y_min))
    data_hi = float(np.nanmax(y_max))
    y_ticks = _nice_ticks(data_lo, data_hi, 5)
    y_lo = min(data_lo, float(y_ticks[0]))
    y_hi = max(data_hi, float(y_ticks[-1]))

    def mx(value):
        return (value - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.5

    def my(value):
        return (value - y_lo) / (y_hi - y_lo) if y_hi > y_lo else 0.5

    xs = np.array([mx(v) for v in x_data])

    # gridlines (behind everything)
    if asset.evalParm("diagnostics_grid"):
        for tick in x_ticks:
            _line(geo, (mx(tick), 0, _DIAG_Z_GRID), (mx(tick), 1, _DIAG_Z_GRID),
                  _DIAG_GRID_COLOR, chrome)
        for tick in y_ticks:
            _line(geo, (0, my(tick), _DIAG_Z_GRID), (1, my(tick), _DIAG_Z_GRID),
                  _DIAG_GRID_COLOR, chrome)

    # optional min-max spread band
    if asset.evalParm("diagnostics_band"):
        band = geo.createPolygon()
        for x_value, y_value in zip(xs, np.array([my(v) for v in y_max])):
            band.addVertex(_point(geo, (x_value, y_value, _DIAG_Z_BAND),
                                  _DIAG_BAND_COLOR))
        for x_value, y_value in zip(xs[::-1],
                                    np.array([my(v) for v in y_min])[::-1]):
            band.addVertex(_point(geo, (x_value, y_value, _DIAG_Z_BAND),
                                  _DIAG_BAND_COLOR))
        chrome.add(band)

    # zero reference line if the value range crosses zero
    if y_lo < 0.0 < y_hi:
        _line(geo, (0, my(0.0), _DIAG_Z_GRID), (1, my(0.0), _DIAG_Z_GRID),
              _DIAG_ZERO_COLOR, chrome)

    # axes (L-shape) and tick marks
    _line(geo, (0, 0, _DIAG_Z_AXIS), (0, 1, _DIAG_Z_AXIS),
          _DIAG_AXIS_COLOR, chrome)
    _line(geo, (0, 0, _DIAG_Z_AXIS), (1, 0, _DIAG_Z_AXIS),
          _DIAG_AXIS_COLOR, chrome)
    for tick in x_ticks:
        _line(geo, (mx(tick), 0, _DIAG_Z_AXIS), (mx(tick), -0.014, _DIAG_Z_AXIS),
              _DIAG_AXIS_COLOR, chrome)
        _merge_into(geo, chrome, _font_geometry(
            _format_number(tick, digits, notation), 0.034, 1, 1,
            (mx(tick), -0.022), _DIAG_AXIS_COLOR))  # center, top
    for tick in y_ticks:
        _line(geo, (0, my(tick), _DIAG_Z_AXIS), (-0.014, my(tick), _DIAG_Z_AXIS),
              _DIAG_AXIS_COLOR, chrome)
        _merge_into(geo, chrome, _font_geometry(
            _format_number(tick, digits, notation), 0.034, 2, 2,
            (-0.022, my(tick)), _DIAG_AXIS_COLOR))  # right, middle

    # curves (in front)
    _diag_curve(geo, curves, xs, [my(v) for v in y_min],
                _DIAG_MIN_COLOR, _DIAG_Z_CURVE)
    _diag_curve(geo, curves, xs, [my(v) for v in y_mean],
                _DIAG_MEAN_COLOR, _DIAG_Z_CURVE)
    _diag_curve(geo, curves, xs, [my(v) for v in y_max],
                _DIAG_MAX_COLOR, _DIAG_Z_CURVE)

    # axis titles + plot title + legend
    units = asset.evalParm("field_units").strip()
    if use_time:
        time_units = asset.evalParm("diagnostics_time_units").strip()
        x_title = "Simulation Time" + (f" [{time_units}]" if time_units else "")
    else:
        x_title = "Frame Index"
    _merge_into(geo, chrome, _font_geometry(
        x_title, 0.045, 1, 1, (0.5, -0.075), _DIAG_AXIS_COLOR))  # center, top
    y_title = "Value" + (f" [{units}]" if units else "")
    _merge_into(geo, chrome, _font_geometry(
        y_title, 0.045, 1, 2, (-0.11, 0.5), _DIAG_AXIS_COLOR, rotate=90))
    _merge_into(geo, chrome, _font_geometry(
        legend_title_text(asset), 0.055, 0, 3, (0.0, 1.15),
        _DIAG_AXIS_COLOR))  # left, bottom

    legend_x = 0.66
    for offset, (label, color) in enumerate((
            ("Max", _DIAG_MAX_COLOR), ("Mean", _DIAG_MEAN_COLOR),
            ("Min", _DIAG_MIN_COLOR))):
        y = 1.15 - 0.06 * offset
        _line(geo, (legend_x, y, _DIAG_Z_CURVE),
              (legend_x + 0.05, y, _DIAG_Z_CURVE), color, chrome)
        _merge_into(geo, chrome, _font_geometry(
            label, 0.034, 0, 2, (legend_x + 0.07, y), color))  # left, middle

    # one bulk transform places the whole plot
    translate = np.asarray(asset.evalParmTuple("diagnostics_translate"),
                           dtype=np.float64)
    scale = float(asset.evalParm("diagnostics_scale"))
    P = np.frombuffer(geo.pointFloatAttribValuesAsString("P"),
                      dtype=np.float32).reshape(-1, 3).astype(np.float64)
    P = translate + scale * P
    geo.setPointFloatAttribValuesFromString(
        "P", np.ascontiguousarray(P).tobytes(),
        float_type=hou.numericData.Float64)

    geo.addAttrib(hou.attribType.Global, "diagnostics_field", "")
    geo.addAttrib(hou.attribType.Global, "diagnostics_min", 0.0)
    geo.addAttrib(hou.attribType.Global, "diagnostics_max", 0.0)
    geo.setGlobalAttribValue("diagnostics_field", legend_title_text(asset))
    geo.setGlobalAttribValue("diagnostics_min", y_lo)
    geo.setGlobalAttribValue("diagnostics_max", y_hi)


def _build_block_geometry(mesh, block_name, slug, colors, values, components):
    """Build one supplementary block in a standalone geometry (bulk attrs)."""
    block = hou.Geometry()
    positions = np.ascontiguousarray(mesh["points"], dtype=np.float64)
    block.createPoints(positions.tolist())
    n = len(positions)

    block.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                    create_local_variable=False)
    _set_floats(block.setPointFloatAttribValuesFromString, "Cd", colors)
    block.addAttrib(hou.attribType.Point, "pvd_block", "")
    block.setPointStringAttribValues("pvd_block", (block_name,) * n)

    if values is not None:
        name = f"{slug}_{mesh['_field_name']}"
        if components == 1:
            block.addAttrib(hou.attribType.Point, name, 0.0,
                            create_local_variable=False)
            arr = np.asarray(values, dtype=np.float64)
        else:
            width = 3 if components in (2, 3) else components
            block.addAttrib(hou.attribType.Point, name,
                            tuple(0.0 for _ in range(width)),
                            create_local_variable=False)
            arr = np.zeros((n, width), dtype=np.float64)
            arr[:, :components] = np.asarray(values, dtype=np.float64
                                             ).reshape(n, components)
        _set_floats(block.setPointFloatAttribValuesFromString, name, arr)

    points = block.points()
    prim_group = block.createPrimGroup(f"readpvd_block_{slug}")
    point_group = block.createPointGroup(f"readpvd_block_{slug}")
    point_group.add(points)
    for family, connectivity in mesh["cells"].items():
        if family == "tet":
            for indices in connectivity:
                prim_group.add(block.createTetrahedronInPlace(
                    *(points[int(i)] for i in indices)))
        elif family == "hex":
            for indices in connectivity:
                prim_group.add(block.createHexahedronInPlace(
                    *(points[int(i)] for i in indices)))
        elif family == "line":
            for indices in connectivity:
                prim = block.createPolygon(is_closed=False)
                for i in indices:
                    prim.addVertex(points[int(i)])
                prim_group.add(prim)
        else:  # tri / quad: bulk creation (>=3 verts)
            for prim in block.createPolygons(
                    tuple(tuple(int(i) for i in indices)
                          for indices in connectivity)):
                prim_group.add(prim)
    block.addAttrib(hou.attribType.Prim, "pvd_block", "")
    block.setPrimStringAttribValues(
        "pvd_block", (block_name,) * len(block.prims()))
    return block


def cook_multi_blocks(node):
    """Generate optional supplementary PVD blocks with independent colors."""
    geo = node.geometry()
    asset = node.parent()
    path = asset.evalParm("PVD_file")
    if not path:
        return
    primary = _source_block_token(asset)
    # Only parse if at least one supplementary block is actually requested.
    wanted = [name for name in ("Volume", "Surface", "Contact", "Points")
              if name != primary
              and asset.parm(f"multi_{name.lower()}_show") is not None
              and asset.evalParm(f"multi_{name.lower()}_show")]
    if not wanted:
        return
    blocks = load_frame(path, int(hou.frame()))

    for block_name, mesh in blocks.items():
        if block_name not in wanted:
            continue
        slug = block_name.lower()
        positions = np.asarray(mesh["points"], dtype=np.float64)
        solution = mesh["point_data"].get("solution")
        if asset.evalParm("show_deformed") and solution is not None:
            positions = positions + np.asarray(solution, dtype=np.float64)
        # _build_block_geometry reads positions back from the mesh dict; keep
        # the (possibly deformed) copy local without mutating the cache.
        mesh = dict(mesh, points=positions)

        field = asset.evalParm(f"multi_{slug}_field")
        values = _mesh_point_field(mesh, field) if field else None
        components = (0 if values is None else
                      1 if np.asarray(values).ndim == 1 else values.shape[1])
        if values is None:
            colors = np.ones((len(positions), 3), dtype=np.float64)
        else:
            reduced = _reduce_array(
                values, _menu_token(asset, f"multi_{slug}_reduction"))
            low, high = asset.evalParmTuple(f"multi_{slug}_range")
            denom = high - low
            normalized = (np.full(len(reduced), 0.5) if abs(denom) < 1e-20
                          else np.clip((reduced - low) / denom, 0.0, 1.0))
            ramp = asset.parm(f"multi_{slug}_ramp").evalAsRamp()
            colors = np.asarray([ramp.lookup(float(v)) for v in normalized])
        mesh["_field_name"] = field
        geo.merge(_build_block_geometry(
            mesh, block_name, slug, colors, values, components))


def legend_values(node):
    count = max(2, int(node.evalParm("legend_ticks")))
    low = float(node.evalParm("color_min"))
    high = float(node.evalParm("color_max"))
    return np.linspace(high, low, count)


def legend_title_text(node):
    custom = node.evalParm("legend_title").strip()
    if custom:
        return custom
    field = node.evalParm("color_attrib")
    reduction = _menu_token(node, "color_reduction")
    label = _field_label(field)
    if reduction not in ("", "auto"):
        label += " - " + REDUCTION_LABELS.get(reduction, reduction)
    units = node.evalParm("field_units").strip()
    if units:
        label += f" [{units}]"
    return label


def legend_rotation(node):
    index = max(0, min(int(node.evalParm("legend_orientation")),
                       len(LEGEND_ROTATIONS) - 1))
    return LEGEND_ROTATIONS[index]


def _point(geo, position, color=None):
    point = geo.createPoint()
    point.setPosition(hou.Vector3(position))
    if color is not None:
        point.setAttribValue("Cd", tuple(float(x) for x in color))
    return point


def _line(geo, a, b, color, group):
    prim = geo.createPolygon()
    prim.setIsClosed(False)
    prim.addVertex(_point(geo, a, color))
    prim.addVertex(_point(geo, b, color))
    group.add(prim)
    return prim


# Scene-legend layout, all in the bar's local space (bar height = 1.0). The
# bar occupies x in [0, BAR_WIDTH]; ticks, labels, and title hang to its right
# / top with fixed gaps so spacing is independent of tick count and digits.
_LEGEND_BAR_WIDTH = 0.12
_LEGEND_TICK_LEN = 0.04
_LEGEND_LABEL_GAP = 0.025
_LEGEND_TITLE_GAP = 0.06


def _font_geometry(text, size, halign, valign, position, color, rotate=0):
    """Render `text` as filled polygons placed/aligned in local space."""
    verb = hou.sopNodeTypeCategory().nodeVerb("font")
    verb.setParms({
        "text": text, "fontsize": float(size), "halign": halign,
        "valign": valign, "type": 2, "hole": True,
        "t": (float(position[0]), float(position[1]), 0.0),
        "r": (0.0, 0.0, float(rotate))})
    geo = hou.Geometry()
    verb.execute(geo, [])
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    if len(geo.points()):
        flat = np.tile(np.asarray(color, dtype=np.float64)[:3],
                       len(geo.points()))
        geo.setPointFloatAttribValuesFromString(
            "Cd", flat.tobytes(), float_type=hou.numericData.Float64)
    return geo


def cook_scene_legend(node):
    """Renderable color strip + tick marks + aligned numeric labels + title.

    Each numeric label is generated as its own font primitive and placed at
    the exact height of its tick (vertically centered), so label spacing
    always matches the colored bar regardless of tick count, digits, or
    number format.
    """
    geo = node.geometry()
    asset = node.parent()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    group = geo.createPrimGroup("readpvd_legend")

    ramp = asset.parm("color_ramp").evalAsRamp()
    segments = 48
    width = _LEGEND_BAR_WIDTH
    for i in range(segments):
        y0 = i / segments
        y1 = (i + 1) / segments
        c0 = ramp.lookup(y0)
        c1 = ramp.lookup(y1)
        prim = geo.createPolygon()
        for pos, color in (((0, y0, 0), c0), ((width, y0, 0), c0),
                           ((width, y1, 0), c1), ((0, y1, 0), c1)):
            prim.addVertex(_point(geo, pos, color))
        group.add(prim)

    tick_color = tuple(asset.evalParmTuple("legend_text_color"))
    n_ticks = max(2, int(asset.evalParm("legend_ticks")))
    digits = asset.evalParm("legend_digits")
    notation = _menu_token(asset, "legend_number_format")
    low = float(asset.evalParm("color_min"))
    high = float(asset.evalParm("color_max"))
    # Tick labels sized to the available spacing so they never overlap.
    label_size = min(0.055, 0.62 / (n_ticks - 1))
    label_x = width + _LEGEND_TICK_LEN + _LEGEND_LABEL_GAP
    for i in range(n_ticks):
        fraction = i / (n_ticks - 1)
        y = fraction
        _line(geo, (width, y, 0), (width + _LEGEND_TICK_LEN, y, 0),
              tick_color, group)
        value = low + fraction * (high - low)
        label = _font_geometry(
            _format_number(value, digits, notation), label_size,
            0, 2, (label_x, y), tick_color)  # left, middle
        _merge_into(geo, group, label)

    title = _font_geometry(
        legend_title_text(asset), 0.075, 0, 3,  # left, bottom
        (0.0, 1.0 + _LEGEND_TITLE_GAP), tick_color)
    _merge_into(geo, group, title)

    geo.addAttrib(hou.attribType.Global, "legend_min", 0.0)
    geo.addAttrib(hou.attribType.Global, "legend_max", 0.0)
    geo.addAttrib(hou.attribType.Global, "legend_title", "")
    geo.setGlobalAttribValue("legend_min", low)
    geo.setGlobalAttribValue("legend_max", high)
    geo.setGlobalAttribValue("legend_title", legend_title_text(asset))


def _merge_into(geo, group, other):
    """Merge `other` into `geo`, adding its new prims to `group`."""
    first = len(geo.prims())
    geo.merge(other)
    group.add(geo.prims()[first:])


def _gnomon_basis(direction):
    """Unit axis direction plus two perpendicular unit vectors."""
    direction = hou.Vector3(direction).normalized()
    helper = hou.Vector3(0, 0, 1)
    if abs(direction.dot(helper)) > 0.9:
        helper = hou.Vector3(0, 1, 0)
    u = direction.cross(helper).normalized()
    v = direction.cross(u).normalized()
    return direction, u, v


def _quad(geo, points, color, group):
    prim = geo.createPolygon()
    for position in points:
        prim.addVertex(_point(geo, position, color))
    group.add(prim)


def _beam(geo, start, end, u, v, radius, color, group, cap_end=True):
    """Square-section solid beam between two points (a thin 3D shaft)."""
    start, end = hou.Vector3(start), hou.Vector3(end)
    offsets = (u * radius + v * radius, -u * radius + v * radius,
               -u * radius - v * radius, u * radius - v * radius)
    s = [start + o for o in offsets]
    e = [end + o for o in offsets]
    for i in range(4):
        j = (i + 1) % 4
        _quad(geo, (s[i], s[j], e[j], e[i]), color, group)
    _quad(geo, (s[3], s[2], s[1], s[0]), color, group)   # start cap
    if cap_end:
        _quad(geo, (e[0], e[1], e[2], e[3]), color, group)


def _cone(geo, tip, base_center, u, v, radius, color, group):
    """Solid 4-sided pyramid: apex at `tip`, square base at `base_center`."""
    corners = [base_center + u * radius + v * radius,
               base_center - u * radius + v * radius,
               base_center - u * radius - v * radius,
               base_center + u * radius - v * radius]
    apex = _point(geo, tip, color)
    base_points = [_point(geo, corner, color) for corner in corners]
    for i in range(4):
        face = geo.createPolygon()
        for point in (apex, base_points[i], base_points[(i + 1) % 4]):
            face.addVertex(point)
        group.add(face)
    base = geo.createPolygon()                            # base cap
    for point in reversed(base_points):
        base.addVertex(point)
    group.add(base)


def _cube(geo, center, half, color, group):
    """Solid axis-aligned cube centered at `center`."""
    center = hou.Vector3(center)
    corner = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner[(sx, sy, sz)] = _point(
                    geo, center + hou.Vector3(sx, sy, sz) * half, color)
    faces = (
        ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),     # +X
        ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),  # -X
        ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),     # +Y
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),  # -Y
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),     # +Z
        ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),  # -Z
    )
    for face in faces:
        prim = geo.createPolygon()
        for sign in face:
            prim.addVertex(corner[sign])
        group.add(prim)


def cook_gnomon(node):
    """Renderable XYZ orientation marker: solid beam shafts with cone
    arrowheads and a solid cube origin marker (a refined take on the 0.26
    gnomon -- 3D heads and cube instead of flat triangles and wire crosses)."""
    geo = node.geometry()
    asset = node.parent()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    group = geo.createPrimGroup("readpvd_gnomon")
    center = hou.Vector3(asset.evalParmTuple("gnomon_center"))
    scale = float(asset.evalParm("gnomon_scale"))
    arrows = bool(asset.evalParm("gnomon_arrows"))
    shaft_radius = 0.012 * scale
    cone_radius = 0.038 * scale
    head_length = 0.22 * scale
    axes = (
        ("gnomon_x", hou.Vector3(1, 0, 0), (0.95, 0.26, 0.26)),
        ("gnomon_y", hou.Vector3(0, 1, 0), (0.36, 0.85, 0.30)),
        ("gnomon_z", hou.Vector3(0, 0, 1), (0.28, 0.48, 1.0)),
    )
    for parm, axis, color in axes:
        if not asset.evalParm(parm):
            continue
        direction, u, v = _gnomon_basis(axis)
        tip = center + direction * scale
        if arrows:
            base = tip - direction * head_length
            _beam(geo, center, base, u, v, shaft_radius, color, group,
                  cap_end=False)
            _cone(geo, tip, base, u, v, cone_radius, color, group)
        else:
            _beam(geo, center, tip, u, v, shaft_radius, color, group)
    if asset.evalParm("gnomon_center_marker"):
        _cube(geo, center, 0.045 * scale, (1.0, 0.85, 0.2), group)


def _block_for(asset, blocks):
    """Pick the vtu block matching the Source parm (Volume/Surface/Points)."""
    wanted = _source_block_token(asset)
    if wanted in blocks:
        return blocks[wanted]
    return next(iter(blocks.values()))


# =============================================================================
# topology stage
# =============================================================================


def cook_topology(node):
    geo = node.geometry()
    asset = _asset_of(node)
    pvd_path = asset.evalParm("PVD_file")
    if not pvd_path:
        return

    if asset.evalParm("remesh_mode"):
        frame_index = int(hou.frame())
    else:
        frame_index = asset.evalParm("topo_frame")

    blocks = load_frame(pvd_path, frame_index)
    mesh = _block_for(asset, blocks)

    points = mesh["points"]
    geo.createPoints(points.tolist())

    # Coincidence groups: PolyFEM writes a discontinuous mesh (per-element
    # duplicated vertices), so a physical vertex appears as several points.
    # Group ids let the field-smoothing stage average a field back onto shared
    # vertices (nodal recovery); the representative flag marks one point per
    # vertex so glyphs draw once per node. Computed once with the cached
    # topology; stored as float (exact for < 16M groups) for fast binary reads.
    inverse, representative = _coincidence_groups(points)
    geo.addAttrib(hou.attribType.Point, "coincident_id", 0.0,
                  create_local_variable=False)
    _set_floats(geo.setPointFloatAttribValuesFromString, "coincident_id",
                inverse)
    geo.addAttrib(hou.attribType.Point, "coincident_rep", 0.0,
                  create_local_variable=False)
    _set_floats(geo.setPointFloatAttribValuesFromString, "coincident_rep",
                representative)

    def put(name, array):
        geo.addArrayAttrib(hou.attribType.Global, name, hou.attribData.Int, 1)
        geo.setGlobalAttribValue(
            name, np.ascontiguousarray(array, dtype=np.int64).ravel().tolist())

    for family in ("tet", "hex", "tri", "quad", "line"):
        if family in mesh["cells"]:
            put(f"{family}_conn", mesh["cells"][family])

    geo.addAttrib(hou.attribType.Global, "topo_key", "")
    geo.setGlobalAttribValue("topo_key", str(mesh["topo_key"]))


# =============================================================================
# per-frame stage
# =============================================================================


def cook_frame(node):
    geo = node.geometry()  # copy of input geometry (points + prims)
    asset = _asset_of(node)
    pvd_path = asset.evalParm("PVD_file")
    if not pvd_path:
        return

    blocks = load_frame(pvd_path, int(hou.frame()))
    mesh = _block_for(asset, blocks)

    if str(mesh["topo_key"]) != geo.attribValue("topo_key"):
        if asset.evalParm("remesh_mode"):
            # topo stage already follows the frame; key mismatch here means
            # cook ordering hiccup -- force it
            raise hou.NodeError("Topology out of sync; re-cook the node.")
        raise hou.NodeError(
            "Mesh topology changes over time (remeshing run). "
            "Enable 'Remeshing Mode' on this node.")

    n_points = geo.intrinsicValue("pointcount")
    points = mesh["points"]
    if len(points) != n_points:
        raise hou.NodeError("Point count mismatch with topology frame")

    # rest positions + per-frame fields, all bulk binary uploads
    _set_floats(geo.setPointFloatAttribValuesFromString, "P", points)

    def upload(attrib_type, find, set_from_string, name, arr):
        arr = np.asarray(arr)
        components = 1 if arr.ndim == 1 else arr.shape[1]
        if components not in (1, 2, 3, 9):
            return
        if components in (2, 3):
            vec = np.zeros((len(arr), 3))
            vec[:, :components] = arr
            arr = vec
        if find(name) is None:
            default = 0.0 if components == 1 else tuple(
                0.0 for _ in range(3 if components in (2, 3) else 9))
            geo.addAttrib(attrib_type, name, default,
                          create_local_variable=False)
        _set_floats(set_from_string, name, arr)

    def upload_point(name, arr):
        upload(hou.attribType.Point, geo.findPointAttrib,
               geo.setPointFloatAttribValuesFromString, name, arr)

    def upload_primitive(name, arr):
        upload(hou.attribType.Prim, geo.findPrimAttrib,
               geo.setPrimFloatAttribValuesFromString, name, arr)

    for raw_name, data in mesh["point_data"].items():
        name = _field_name(raw_name)
        arr = np.asarray(data)
        # some fields (e.g. *_avg) omit obstacle vertices: zero-pad to the
        # point count (matches the implicit zero-fill of the 0.26 viewer)
        if len(arr) != n_points:
            padded = np.zeros((n_points,) + arr.shape[1:], dtype=np.float64)
            padded[:min(len(arr), n_points)] = arr[:n_points]
            arr = padded
        upload_point(name, arr)

    # Preserve VTK CellData on Houdini primitives. A point-averaged copy is
    # added only when no PointData field has the same name, allowing the
    # existing point-coloring and probing tools to display cell fields while
    # retaining the original un-interpolated primitive values.
    family_attrib = geo.findPrimAttrib("source_cell_family")
    index_attrib = geo.findPrimAttrib("source_cell_index")
    if mesh.get("cell_data") and family_attrib is not None and index_attrib is not None:
        prim_families = np.asarray(geo.primStringAttribValues(
            "source_cell_family"), dtype=object)
        prim_indices = np.asarray(geo.primIntAttribValues(
            "source_cell_index"), dtype=np.int64)
        for raw_name, by_family in mesh["cell_data"].items():
            name = _field_name(raw_name)
            sample = next(iter(by_family.values()), None)
            if sample is None:
                continue
            sample = np.asarray(sample)
            shape = () if sample.ndim == 1 else (sample.shape[1],)
            prim_values = np.zeros((len(prim_indices),) + shape,
                                   dtype=np.float64)
            point_sum = np.zeros((n_points,) + shape, dtype=np.float64)
            point_count = np.zeros(n_points, dtype=np.float64)
            for family, values in by_family.items():
                values = np.asarray(values)
                mask = prim_families == family
                if np.any(mask):
                    prim_values[mask] = values[prim_indices[mask]]
                conn = mesh["cells"].get(family)
                if conn is None:
                    continue
                for corner in range(conn.shape[1]):
                    np.add.at(point_sum, conn[:, corner], values)
                    np.add.at(point_count, conn[:, corner], 1.0)
            upload_primitive(name, prim_values)
            if geo.findPointAttrib(name) is None:
                divisor = np.maximum(point_count, 1.0)
                if point_sum.ndim > 1:
                    divisor = divisor[:, None]
                upload_point(name, point_sum / divisor)

    # PolyFEM exports fiber directions as three scalar fields per model; make
    # them a vector so they can be drawn and probed as one thing.
    families = []
    for prefix in _fiber_prefixes(mesh["point_data"]):
        columns = [np.asarray(mesh["point_data"][f"{prefix}fiber_direction_{a}"])
                   for a in "xyz"]
        if any(len(c) != n_points for c in columns):
            continue
        name = _fiber_attrib(prefix, raw=True)
        upload_point(name, np.stack([c.ravel() for c in columns], axis=1))
        families.append(name)
        _DISPLAY_NAMES.setdefault(name, prefix + "fiber_direction")

    # Minimal PolyFEM output may intentionally omit material fields. In that
    # case, use the companion written by the preprocessing HDA. Native solver
    # fields remain authoritative whenever they are present.
    if not families:
        for name, label, values in _companion_point_fibers(pvd_path, mesh):
            upload_point(name, values)
            families.append(name)
            _DISPLAY_NAMES.setdefault(name, label)

    # Sanitized attribute name -> original vtu name, for anything reading the
    # cooked geometry (probe, downstream tools) rather than the module cache.
    if geo.findGlobalAttrib("readpvd_display_names") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_display_names", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue("readpvd_display_names",
                             json.dumps(_DISPLAY_NAMES, sort_keys=True))
    if geo.findGlobalAttrib("readpvd_fiber_families") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_fiber_families", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue("readpvd_fiber_families", " ".join(families))
    if geo.findGlobalAttrib("readpvd_fiber_reference_frame") is None:
        geo.addAttrib(hou.attribType.Global, "readpvd_fiber_reference_frame", "",
                      create_local_variable=False)
    geo.setGlobalAttribValue(
        "readpvd_fiber_reference_frame",
        "simulation_world_reference_a0" if families else "")


# =============================================================================
# UI callbacks (playbar / cache / scaling), ported from 0.26
# =============================================================================


def refresh(kwargs=None, force_clear_cache=False):
    """Pick up new timesteps of the SAME simulation.

    Every disk cache (parsed pvd/vtu, field info, sequence metadata, frame
    scans) is keyed on (path, mtime, size), so files a running sim rewrites
    invalidate themselves and unchanged steps keep hitting the cache -- no
    wholesale clearing here, and neither the cooked-frame cache (cache1)
    nor the current frame position is touched. Only a PVD file change
    (start) resets those.
    """
    node = hou.pwd() if kwargs is None else kwargs["node"]
    pvd_path = node.evalParm("PVD_file")
    try:
        entries = read_pvd(pvd_path)
    except Exception as e:
        # Likely the sim rewriting the .pvd mid-refresh: keep the current
        # playbar, caches, and settings; the next refresh will see it whole.
        _message(f"Failed to read .pvd: {e}")
        return
    if not entries:
        _message("Empty .pvd collection")
        return
    if hou.isUIAvailable():
        hou.playbar.setUseIntegerFrames(True)
        hou.playbar.setFrameIncrement(1)
        hou.playbar.setFrameRange(0, len(entries) - 1)
        hou.playbar.setPlaybackRange(0, len(entries) - 1)
    sync_available_options({"node": node})
    if force_clear_cache:
        clear_cache(kwargs)


def start(kwargs=None):
    # A PVD file change is new data: drop every cache (parsed files AND
    # cooked frames) and jump to frame 0, unlike the Refresh button.
    clear_caches()
    _SEQUENCE_METADATA_CACHE.clear()
    _SCAN_CACHE.clear()
    _FIBER_COMPANION_CACHE.clear()
    _FIBER_POINT_CACHE.clear()
    refresh(kwargs, force_clear_cache=True)
    node = hou.pwd() if kwargs is None else kwargs["node"]
    _autocenter_clip(node)
    if hou.isUIAvailable():
        hou.setFrame(0)


def clear_cache(kwargs=None):
    """Flush the cooked-frame cache (cache1). Never moves the playbar."""
    node = hou.pwd() if kwargs is None else kwargs["node"]
    node.allowEditingOfContents()
    cache_node = node.node("cache1")
    if cache_node is not None:
        cache_node.parm("clear").pressButton()


def toggle_cache(kwargs):
    node = kwargs["node"]
    node.allowEditingOfContents()
    cache_node = node.node("cache1")
    if cache_node is not None:
        cache_node.bypass(not node.evalParm("cache"))


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
    # Fiber dispersion is bounded by 1/d, and the preprocessing HDA colours it
    # over that fixed range -- match it so the same kappa reads as the same
    # colour before and after the solve.
    if _display_name(node.evalParm("color_attrib")).endswith("kappa"):
        node.setParms({"color_min": 0.0, "color_max": 1.0 / 3.0})
        update_color_status(kwargs)
        return
    values = np.frombuffer(
        geo.pointFloatAttribValuesAsString("for_color"), dtype=np.float32)
    _apply_color_range(node, values)
    update_color_status(kwargs)


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
    try:
        collected = [values[np.isfinite(values)]
                     for _, _, values in _scan_displayed_values(node, path)
                     if values is not None]
    except Exception as exc:
        _message(f"Could not scan PVD sequence: {exc}")
        return
    collected = [chunk for chunk in collected if len(chunk)]
    if collected:
        _apply_color_range(node, np.concatenate(collected))
    else:
        _message("No finite values were found for the selected color field.")
    update_color_status(kwargs)


def _apply_color_range(node, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if node.evalParm("color_scale") == 1:
        values = values[values > 0]
    if not len(values):
        _message("No valid values were found for the selected color scale.")
        return
    if node.evalParm("range_percentile"):
        low_percentile, high_percentile = node.evalParmTuple(
            "range_percentiles")
        low, high = np.percentile(
            values, (min(low_percentile, high_percentile),
                     max(low_percentile, high_percentile)))
    else:
        low, high = values.min(), values.max()
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


def overlay_feature_changed(kwargs):
    """Auto-activate the state when an overlay legend or the probe is enabled
    (both are drawn/handled by the viewer state)."""
    node = kwargs["node"]
    if node.evalParm("legend_mode") in (2, 3) or node.evalParm("probe_enabled"):
        show_overlay(kwargs)


def autoplace_legend(kwargs):
    node = kwargs["node"]
    src = node.node("OUT_result")
    if src is None:
        return
    try:
        bbox = src.geometry().boundingBox()
    except hou.OperationFailed:
        return
    size = bbox.sizevec()
    margin = max(float(size[0]), float(size[1]), float(size[2]), 1.0) * 0.1
    scale = max(float(size[1]), float(size[2]), 1.0)
    node.parmTuple("legend_translate").set(
        (bbox.maxvec()[0] + margin, bbox.minvec()[1], bbox.minvec()[2]))
    node.parm("legend_scale").set(scale)


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
    per_point = averaged[groups]
    eigenvalues, eigenvectors = np.linalg.eigh(per_point)
    eigenvalues = eigenvalues[:, ::-1]                     # largest first
    eigenvectors = np.ascontiguousarray(eigenvectors[:, :, ::-1])
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
