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

import base64
import bisect
import hashlib
import importlib
import io
import math
import os
import json
import platform
import re
import itertools
import shutil
import subprocess
import sys
import zipfile

import numpy as np

import hou


# The official h5py 3.16.0 wheels for Houdini 22's CPython 3.13, embedded in
# the asset (base64 sections), one per platform Houdini 22 runs on.
EMBEDDED_H5PY_SECTIONS = {
    ("darwin", "arm64"): "h5py-3.16.0-cp313-macos-arm64.whl.b64",
    ("darwin", "x86_64"): "h5py-3.16.0-cp313-macos-x86_64.whl.b64",
    ("linux", "x86_64"): "h5py-3.16.0-cp313-linux-x86_64.whl.b64",
    ("win32", "amd64"): "h5py-3.16.0-cp313-windows-amd64.whl.b64",
}
# The Linux wheel is manylinux_2_28: it needs glibc 2.28 or newer.
_LINUX_GLIBC = (2, 28)


def h5py_platform_key():
    """(sys.platform, machine) in the spelling of EMBEDDED_H5PY_SECTIONS."""
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "x64": "amd64"}.get(machine, machine)
    if sys.platform == "win32" and machine == "x86_64":
        machine = "amd64"
    if sys.platform != "win32" and machine == "amd64":
        machine = "x86_64"
    return sys.platform, machine


def embedded_h5py_section(key=None, python=None):
    """(section name, None) of the bundled h5py wheel that runs here, or
    (None, why not)."""
    key = key or h5py_platform_key()
    python = python or sys.version_info[:2]
    if tuple(python) != (3, 13):
        return None, (f"the bundled HDF5 reader is built for Houdini 22's "
                      f"Python 3.13, and this is Python "
                      f"{python[0]}.{python[1]}")
    section = EMBEDDED_H5PY_SECTIONS.get(tuple(key))
    if section is None:
        return None, ("there is no bundled HDF5 reader for this platform "
                      f"({key[0]}, {key[1]}); bundled: macOS (Apple silicon "
                      "and Intel), Linux x86_64, Windows x64")
    if key[0] == "linux":
        library, version = platform.libc_ver()
        try:
            found = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:
            found = ()
        if library != "glibc" or found < _LINUX_GLIBC:
            have = f"{library} {version}".strip() or "no glibc"
            return None, ("the bundled Linux HDF5 reader needs glibc 2.28 or "
                          f"newer (this system has {have})")
    return section, None


def _load_embedded_h5py():
    """Install the asset's pinned h5py wheel into a versioned user cache.

    The wheel is embedded in the .hdanc, so reading PolyFEM's default HDF5
    output works offline and does not modify Houdini's own Python installation.
    A normal h5py installation, when present, is preferred by _import_h5py.
    """
    section_name, reason = embedded_h5py_section()
    if section_name is None:
        raise ImportError(
            f"The bundled HDF5 reader cannot be used: {reason}. Install h5py "
            "into Houdini's Python environment, then restart Houdini (or "
            "turn off Use HDF5 on the PolyFEM node).")

    node_type = hou.nodeType(hou.objNodeTypeCategory(), "readPVD::1.0")
    definition = node_type.definition() if node_type is not None else None
    section = (definition.sections().get(section_name)
               if definition is not None else None)
    if section is None:
        raise ImportError(
            "This Read PVD asset does not contain its HDF5 reader. Reinstall "
            "the current object_readPVD.1.0.hdanc asset.")
    wheel = base64.b64decode(section.contents().encode("ascii"))
    digest = hashlib.sha256(wheel).hexdigest()
    pref_dir = hou.expandString("$HOUDINI_USER_PREF_DIR")
    vendor_dir = os.path.join(
        pref_dir, "python3.13libs", f"readpvd_h5py_{digest[:12]}")
    marker = os.path.join(vendor_dir, ".complete")
    if not os.path.isfile(marker):
        if os.path.isdir(vendor_dir):
            shutil.rmtree(vendor_dir)
        os.makedirs(vendor_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            for member in archive.infolist():
                normalized = os.path.normpath(member.filename)
                if os.path.isabs(normalized) or normalized.startswith(".."):
                    raise ImportError(
                        "The embedded HDF5 reader contains an unsafe path.")
            archive.extractall(vendor_dir)
        with open(marker, "w", encoding="ascii") as handle:
            handle.write(digest + "\n")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    importlib.invalidate_caches()
    import h5py
    return h5py

# ---- embedded native parser (built from src/common/vtu_parser.py) ----------
# @VTU_PARSER@
# ---- end embedded parser ----------------------------------------------------

# ---- embedded MSH parser (built from src/common/msh_parser.py) -------------
# The Export tab reads the run's own .msh files and writes moved copies.
# @MSH_PARSER@
# ---- end embedded MSH parser -------------------------------------------------

# ---- embedded .xlsx writer (built from src/common/xlsx_writer.py) ----------
# @XLSX_WRITER@
# ---- end embedded .xlsx writer -----------------------------------------------

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


# =============================================================================
# Timeline: which PVD step a Houdini frame shows (2026-10-02)
#
# The PolyFEM node plays a run on Houdini's timeline: Houdini time is
# simulation time, t = Start Time + (T - T_start) * Time Scale, recorded in
# the run's scene record (<run>/input/hda_scene.json). Time Mapping
# "Automatic" uses that record and falls back to one frame per output step
# (frame k shows step k, as before) for runs without one. On the simulation
# timeline a frame shows the last step at or before its simulation time.
# =============================================================================

SCENE_RECORD_FILE = "hda_scene.json"
SCENE_RECORD_SCHEMA = "polyfem-houdini-scene"
TIME_MAPPING_TOKENS = ("auto", "steps", "time")
_SCENE_RECORD_CACHE = {}


def scene_record_path(pvd_path):
    """<run>/input/hda_scene.json for <run>/output/<name>.pvd."""
    run = os.path.dirname(os.path.dirname(os.path.abspath(pvd_path)))
    return os.path.join(run, "input", SCENE_RECORD_FILE)


def scene_record(node):
    """The PolyFEM node's record of this run, or None."""
    pvd = node.evalParm("PVD_file")
    if not pvd:
        return None
    path = scene_record_path(pvd)
    try:
        key = _file_key(path)
    except OSError:
        return None
    cached = _SCENE_RECORD_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict) \
            or record.get("schema") != SCENE_RECORD_SCHEMA:
        record = None
    if len(_SCENE_RECORD_CACHE) > 8:
        _SCENE_RECORD_CACHE.clear()
    _SCENE_RECORD_CACHE[path] = (key, record)
    return record


def _pvd_times(node):
    path = node.evalParm("PVD_file")
    if not path:
        return []
    try:
        return [entry[0] for entry in read_pvd(path)]
    except Exception:
        return []


def time_mapping(node):
    """(Houdini time of the start, time scale, simulation start time) of the
    playback timeline, or None for one frame per output step."""
    parm = node.parm("time_mapping")
    token = parm.evalAsString() if parm is not None else "steps"
    if token == "steps":
        return None
    record = scene_record(node)
    line = record.get("timeline") if record else None
    if token == "auto":
        if not isinstance(line, dict):
            return None
        try:
            return ((float(line["start_frame"]) - 1.0) / float(line["fps"]),
                    float(line["time_scale"]), float(line["t0"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    if isinstance(line, dict) and "t0" in line:
        t0 = float(line["t0"])
    else:
        times = _pvd_times(node)
        t0 = times[0] if times else 0.0
    scale = float(node.evalParm("time_scale"))
    return (hou.frameToTime(node.evalParm("time_start_frame")),
            scale if scale > 0 else 1.0, t0)


def entry_index(node, frame=None):
    """Index of the PVD step shown at a Houdini frame (default: current)."""
    frame = hou.frame() if frame is None else frame
    mapping = time_mapping(node)
    if mapping is None:
        return int(frame)
    times = _pvd_times(node)
    if not times:
        return 0
    start, scale, t0 = mapping
    t = t0 + (hou.frameToTime(frame) - start) * scale
    tolerance = 1e-9 * max(1.0, abs(times[-1] - times[0]))
    return max(0, bisect.bisect_right(times, t + tolerance) - 1)


def entry_frame(node, index):
    """The first Houdini frame showing PVD step `index`."""
    mapping = time_mapping(node)
    if mapping is None:
        return float(index)
    times = _pvd_times(node)
    if not times:
        return 0.0
    start, scale, t0 = mapping
    index = max(0, min(int(index), len(times) - 1))
    return math.ceil(hou.timeToFrame(start + (times[index] - t0) / scale)
                     - 1e-6)


def playbar_range(node):
    """(first, last) whole frames covering the run on the active mapping."""
    times = _pvd_times(node)
    if not times:
        return 0, 0
    if time_mapping(node) is None:
        return 0, len(times) - 1
    return (int(entry_frame(node, 0)),
            int(max(entry_frame(node, 0), entry_frame(node, len(times) - 1))))


def sim_time_text(node):
    """Read-only Simulation Time line: the step on screen and its time."""
    hou.frame()  # time-dependent, so it follows the playbar
    times = _pvd_times(node)
    if not times:
        return "Load a PVD file."
    index = min(entry_index(node), len(times) - 1)
    if time_mapping(node) is None:
        how = "one frame per output step"
        parm = node.parm("time_mapping")
        if parm is not None and parm.evalAsString() == "auto":
            how += " (this run has no PolyFEM node timeline)"
    else:
        how = "simulation time"
    return (f"Step {index} of {len(times) - 1}: t = {times[index]:.6g}  "
            f"({how})")


def time_mapping_changed(kwargs):
    """A new mapping shows other steps on the same frames: drop the cooked
    frames and set the playbar to the run."""
    clear_cache(kwargs)
    refresh(kwargs)


def _frame_metadata(node):
    """Renderable field metadata for the active frame (no array decode)."""
    path = node.evalParm("PVD_file")
    if not path:
        return {}
    frame = entry_index(node)
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
        blocks = load_frame(path, entry_index(node))
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
    """Color Field / Field Value To Display callback: a new quantity needs its
    own range, so the displayed range follows it unless it is locked."""
    sync_available_options(kwargs)
    auto_range_default(kwargs["node"])
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
    """A field of a parsed mesh by its attribute name (cell data averaged to
    the points). `field` is the sanitized name the menus and the cooked
    geometry use ("MaterialSum/HGODispersion/kappa" is
    MaterialSum_HGODispersion_kappa), so raw vtu names are sanitized the
    same way before they are compared."""
    for raw_name, data in mesh["point_data"].items():
        if _field_name(raw_name) == field:
            array = np.asarray(data, dtype=np.float64)
            if len(array) == len(mesh["points"]):
                return array
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
        return np.linalg.det(F())

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
        return np.linalg.eigh(C())                    # ascending

    @once
    def b_eigen():
        return np.linalg.eigh(B())

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
        value = 0.5 * (identity - np.linalg.pinv(B()))
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
        return np.linalg.eigvalsh(cauchy())[:, ::-1]

    @once
    def Finv():
        return np.linalg.pinv(F())

    @once
    def pk1():
        return jacobian()[:, None, None] * np.einsum(
            "nij,njk->nik", cauchy(), np.swapaxes(Finv(), 1, 2))

    @once
    def pk2():
        return jacobian()[:, None, None] * np.einsum(
            "nij,njk,nlk->nil", Finv(), cauchy(), Finv())

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
            lambda: np.linalg.eigvalsh(green())[:, ::-1],
        "almansi_strain": almansi,
        "almansi_strain_eigenvalues":
            lambda: np.linalg.eigvalsh(almansi())[:, ::-1],
        "hencky_strain": hencky,
        "hencky_strain_eigenvalues":
            lambda: np.linalg.eigvalsh(hencky())[:, ::-1],
        "infinitesimal_strain": infinitesimal,
        "infinitesimal_strain_eigenvalues":
            lambda: np.linalg.eigvalsh(infinitesimal())[:, ::-1],
    }
    stresses = {
        "cauchy_mat": cauchy, "cauchy_eigenvalues": eigen,
        "cauchy_trace": trace, "hydrostatic_stress": hydro,
        "deviatoric_stress": dev, "stress_J2": j2,
        "stress_J3": lambda: np.linalg.det(dev()),
        "von_mises_derived": lambda: np.sqrt(np.maximum(0, 3 * j2())),
        "max_shear_stress": lambda: 0.5 * (eigen()[:, 0] - eigen()[:, 2]),
        "stress_triaxiality": lambda: np.divide(
            hydro(), np.sqrt(np.maximum(0, 3 * j2())),
            out=np.zeros_like(hydro()), where=j2() > 1e-24),
        "pk1": pk1, "pk2": pk2,
        "pk2_eigenvalues": lambda: np.linalg.eigvalsh(
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


def _scan_reduced(node, mesh, field, reduction):
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
        return _scan_reduced(node, mesh, field, reduction) \
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


def _coincidence_groups(points, body=None):
    """Return (group id per point, representative flag) for coincident nodes.

    PolyFEM writes per-element duplicated vertices; `inverse` labels which
    physical vertex each point is, and `representative` is 1 for exactly one
    point per physical vertex (used to draw a single glyph per node). With
    PolyFEM body ids, points of different bodies are never one vertex: two
    bodies touching (or sharing an interface) keep their own values when the
    display averages over a vertex.
    """
    keys = np.round(np.asarray(points, dtype=np.float64), 9)
    if body is not None and len(body) == len(keys):
        keys = np.column_stack((keys, np.rint(np.asarray(
            body, dtype=np.float64).reshape(len(keys), -1)[:, 0])))
    _, index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    representative = np.zeros(len(points), dtype=np.float64)
    representative[index] = 1.0
    return inverse, representative


# Element faces in the order the boundary_display wrangle emits them: the
# build_prims vertex order of each volume prim, indexed by Houdini's
# tet_faceindex / hex_faceindex tables.
_FACE_TABLES = {
    "tet": (np.array([1, 3, 2, 0]),
            np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])),
    "hex": (np.array([0, 1, 3, 2, 4, 5, 7, 6]),
            np.array([[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4],
                      [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]])),
}


def _interior_face_bits(conn, family, inverse, body=None):
    """Per cell, bit f set when face f is shared with another cell.

    PolyFEM gives every element its own copy of its vertices, so faces are
    matched through the coincidence groups (and the body id, so an interface
    between two bodies stays visible when one of them is hidden).
    """
    prim_order, face_index = _FACE_TABLES[family]
    conn = np.asarray(conn, dtype=np.int64)
    if not len(conn):
        return np.zeros(0, dtype=np.int64)
    faces = conn[:, prim_order][:, face_index]          # (cells, faces, k)
    cells, n_faces, _ = faces.shape
    keys = np.sort(inverse[faces].reshape(cells * n_faces, -1), axis=1)
    if body is not None and len(body) == len(inverse):
        body = np.rint(np.asarray(body, dtype=np.float64).ravel())
        keys = np.column_stack(
            (keys, body[faces[:, :, 0].ravel()].astype(np.int64)))
    order = np.lexsort(keys.T[::-1])
    ordered = keys[order]
    same = np.all(ordered[1:] == ordered[:-1], axis=1)
    shared = np.zeros(len(keys), dtype=bool)
    shared[1:] |= same
    shared[:-1] |= same
    interior = np.empty(len(keys), dtype=bool)
    interior[order] = shared
    weights = np.left_shift(1, np.arange(n_faces, dtype=np.int64))
    return (interior.reshape(cells, n_faces) * weights).sum(axis=1)


def _coincidence_inverse(points, body=None):
    """Group id per point for coincident (duplicated) mesh vertices."""
    return _coincidence_groups(points, body)[0]


def _nodal_average(values, inverse):
    """Average `values` over coincidence groups (FEM nodal recovery)."""
    finite = np.isfinite(values)
    size = int(inverse.max()) + 1 if len(inverse) else 0
    sums = np.bincount(inverse[finite], weights=values[finite], minlength=size)
    counts = np.bincount(inverse[finite], minlength=size)
    averaged = np.divide(sums, counts, out=np.full(size, np.nan, np.float64),
                         where=counts > 0)
    return averaged[inverse]


# Memoized results of the last all-frames scan (Auto Range: All Frames), so a
# repeat is instant. Keyed on everything that changes the per-frame
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


def _scan_frames(node, path, frames=None):
    """Compute (frame, timestep, values) per frame -- pure numpy, an exact
    mirror of the color stage (reduction, reference comparison, smoothing
    `_avg` preference, nodal averaging, body visibility). No network cook and
    no frame change, so the range agrees with the viewport. `frames` limits
    the scan to some frame numbers (default: every frame).
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
    for frame in (range(len(entries)) if frames is None else frames):
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
            values = _scan_reduced(node, mesh, field, reduction)
            if values is not None:
                values = _apply_reference(
                    values, reference_reduced, reference_mode)
        else:
            # smoothing prefers PolyFEM's continuous _avg field where present
            effective = field
            if smooth:
                candidate = field + "_avg"
                names = {_field_name(name) for name in mesh["point_data"]}
                if candidate in names:
                    effective = candidate
            values = _scan_reduced(node, mesh, effective, reduction)
        if values is not None and smooth:
            if inverse is None or len(inverse) != len(values):
                inverse = _coincidence_inverse(
                    mesh["points"], mesh["point_data"].get("body_ids"))
            values = _nodal_average(values, inverse)
        # Restrict the range to the bodies that are displayed.
        if values is not None and visible is not None:
            body = mesh["point_data"].get("body_ids")
            if body is not None:
                shown = np.isin(np.asarray(body).astype(np.int64).ravel(),
                                list(visible))
                values = np.where(shown, values, np.nan)
        yield frame, timestep, values


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
    blocks = load_frame(path, entry_index(asset))

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
        frame_index = entry_index(asset)
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
    # With body ids the groups stay inside one body, so smoothing never mixes
    # two bodies' values where they touch.
    body = mesh["point_data"].get("body_ids")
    inverse, representative = _coincidence_groups(points, body)
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

    for family in ("tet", "hex"):
        if family in mesh["cells"]:
            put(f"{family}_face_bits", _interior_face_bits(
                mesh["cells"][family], family, inverse, body))

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

    blocks = load_frame(pvd_path, entry_index(asset))
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
        first, last = playbar_range(node)
        hou.playbar.setUseIntegerFrames(True)
        hou.playbar.setFrameIncrement(1)
        hou.playbar.setFrameRange(first, last)
        hou.playbar.setPlaybackRange(first, last)
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
        hou.setFrame(playbar_range(node)[0])
    # First look: colors spread over the result's values instead of the
    # 0..1 a new node starts with, and the model centered in the viewport.
    auto_range_default(node)
    update_color_status({"node": node})
    _frame_result(node)


def _frame_result(node):
    """Frame the Scene Viewer on the loaded result (interactive sessions)."""
    viewer = _scene_viewer()
    if viewer is None:
        return
    try:
        output = node.node("output")
        output.cook()
        bbox = output.geometry().boundingBox()
        if not bbox.isValid():
            return
        try:
            bbox = bbox * node.worldTransform()
        except (TypeError, AttributeError, hou.Error):
            pass
        viewer.curViewport().frameBoundingBox(bbox)
    except (hou.Error, AttributeError):
        pass


def clear_cache(kwargs=None):
    """Flush the cooked-frame cache (cache1). Never moves the playbar.

    Pressing an internal button is allowed on a locked asset; nothing here
    may unlock the node (allowEditingOfContents), or the instance stops
    receiving later asset rebuilds.
    """
    node = hou.pwd() if kwargs is None else kwargs["node"]
    cache_node = node.node("cache1")
    if cache_node is not None:
        cache_node.parm("clear").pressButton()
    _CACHE_CLEARED.add(node.sessionId())
    _CACHE_CONTENTS.pop(node.sessionId(), None)
    # Cache Status reads this counter, so it updates the moment the button is
    # pressed instead of on the next frame change.
    epoch = node.parm("cache_epoch")
    if epoch is not None:
        epoch.set(epoch.eval() + 1)


def toggle_cache(kwargs):
    # cache_switch follows the toggle by expression; turning the cache off
    # also releases the frames it holds.
    clear_cache(kwargs)


# Measured size of one cached frame per asset instance (bytes), updated
# whenever the cache's input is already cooked.
_CACHE_FRAME_BYTES = {}
CACHE_FRAME_CEILING = 2500   # the Cache SOP's own default Max Frames


def _physical_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 16 << 30


def _cache_budget_bytes(asset):
    parm = asset.parm("cache_memory_gb")
    limit_gb = parm.eval() if parm is not None else 0.0
    if limit_gb > 0:
        return int(limit_gb * (1 << 30))
    return _physical_memory() // 2


def cache_frame_limit(cache_node):
    """Max Frames for cache1: as many frames as fit the memory budget.

    Evaluated by cache1's maxframes expression on every cook. The frame size
    is read only from an input that is already cooked, so a cache hit never
    forces the upstream stages to run.
    """
    asset = cache_node.parent()
    key = asset.sessionId()
    source = cache_node.input(0)
    if source is not None and not source.needsToCook():
        try:
            size = int(source.geometry().intrinsicValue("memoryusage"))
        except (hou.Error, TypeError, ValueError):
            size = 0
        if size > 0:
            _CACHE_FRAME_BYTES[key] = size
    size = _CACHE_FRAME_BYTES.get(key)
    if not size:
        return CACHE_FRAME_CEILING
    return int(max(1, min(CACHE_FRAME_CEILING,
                          _cache_budget_bytes(asset) // size)))


# Last frames/memory the Cache SOP reported, per asset instance, and the
# instances whose cache was cleared since their last new frame.
_CACHE_CONTENTS = {}
_CACHE_CLEARED = set()


def _cache_contents(node, cache_node):
    """(frames cached, memory text) as the Cache SOP reports them.

    Reading the info tree of a node that needs to cook would cook it (and,
    right after Clear Cache, re-cache the frame on screen), so a dirty cache
    reports the last known contents instead.
    """
    key = node.sessionId()
    if cache_node.needsToCook():
        if key in _CACHE_CLEARED:
            return 0, "0 B"
        return _CACHE_CONTENTS.get(key)
    try:
        rows = dict(cache_node.infoTree(verbose=False).branches()[
            "Cache SOP Info"].rows())
        contents = (int(rows["Geometries Cached"].split("/")[0]),
                    rows["Memory Used"])
    except (hou.Error, KeyError, ValueError, AttributeError):
        return _CACHE_CONTENTS.get(key)
    _CACHE_CONTENTS[key] = contents
    if contents[0] > 1:
        _CACHE_CLEARED.discard(key)
    return contents


def _system_memory():
    """(total, available) physical memory in bytes, or None."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        return None


def _houdini_memory_bytes():
    """Resident memory of this Houdini process in bytes, or None."""
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        rss_kb = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=2).stdout.strip()
        return int(rss_kb) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def available_memory_gb():
    """System memory available right now, in GB (the OnCreated default for
    Cache Memory Limit); falls back to half of physical memory."""
    memory = _system_memory()
    available = memory[1] if memory else _physical_memory() // 2
    return round(available / float(1 << 30), 1)


def cache_status_text(node):
    hou.frame()   # time-dependent, so the readout refreshes as frames load
    node.evalParm("cache_epoch")   # ... and when Clear Cache is pressed
    if not node.evalParm("cache"):
        return "Cache off: every frame change reads its result file."
    budget = _cache_budget_bytes(node) / float(1 << 30)
    size = _CACHE_FRAME_BYTES.get(node.sessionId())
    cache_node = node.node("cache1")
    contents = (_cache_contents(node, cache_node)
                if cache_node is not None else None)
    if contents is None:
        held = ""
    elif contents[0] <= 1 and node.sessionId() in _CACHE_CLEARED:
        held = ("Cleared: 0 frames cached. " if contents[0] == 0 else
                f"Cleared: only the frame on screen is cached "
                f"({contents[1]}). ")
    else:
        held = (f"{contents[0]} frame{'s' if contents[0] != 1 else ''} "
                f"cached ({contents[1]}). ")
    if not size:
        return (f"{held}Limit {budget:.1f} GB; the frame size is measured "
                "when the first frame is cached.")
    frames = min(CACHE_FRAME_CEILING, max(1, int(budget * (1 << 30) // size)))
    return (f"{held}Limit {budget:.1f} GB holds {frames} "
            f"frame{'s' if frames != 1 else ''} of "
            f"{size / float(1 << 30):.2f} GB; the oldest are dropped first.")


def memory_status_text(node):
    hou.frame()   # time-dependent, so the readout refreshes as frames load
    gb = float(1 << 30)
    parts = []
    houdini = _houdini_memory_bytes()
    if houdini:
        parts.append(f"Houdini {houdini / gb:.1f} GB")
    memory = _system_memory()
    if memory:
        total, available = memory
        parts.append(f"System {(total - available) / gb:.1f} of "
                     f"{total / gb:.1f} GB used, {available / gb:.1f} GB "
                     "available")
    return " | ".join(parts) or "Memory use is not available on this system."


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
    try:
        collected = [values[np.isfinite(values)] for _, _, values in
                     _scan_frames(node, path, sorted({current, count - 1}))
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
    if node.evalParm("range_symmetric") and node.evalParm("color_scale") == 0:
        extent = max(abs(float(low)), abs(float(high)))
        low, high = -extent, extent
    node.setParms({"color_min": float(low), "color_max": float(high)})
    return True


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


# =============================================================================
# Force curves (review F4, 2026-10-02)
#
# With Nodal Forces on, PolyFEM writes every node's force of every term of
# the energy ("<term>_forces" = -dE/du) on the volume block, once per element
# the node belongs to. At equilibrium the terms balance at every free node;
# where the displacement is prescribed the imbalance is what holds the node
# there. So, over the nodes of a set (each counted once):
#   reaction = -(sum of every term): the force the prescribed displacement --
#              or an obstacle's prescribed motion -- applies there (zero where
#              nothing is prescribed);
#   contact  = sum of the contact and friction forces on those nodes.
# For an obstacle the set is its own output rows (body id 0): contact is the
# force the body applies to the obstacle, and the reaction the force that
# drives the obstacle along its path. The sets (sideset faces and obstacle
# surfaces in simulation coordinates) come from the PolyFEM node's scene
# record. Adhesion and Rayleigh-damping forces are not written by PolyFEM
# and so are not included.
# =============================================================================

FORCE_TERMS = ("elastic", "inertia", "body", "contact", "friction", "damping",
               "pressure", "strain_augmented_lagrangian_lagr",
               "periodic_contact")
FORCE_CSV = "force_curves.csv"
FORCE_QUANTITIES = ("reaction", "contact")
FORCE_COMPONENTS = ("x", "y", "z", "magnitude")
FORCE_AGAINST = ("displacement", "time")
_FORCE_NODES_CACHE = {}
_FORCE_CURVES = {}
_CHART_WINDOWS = {}


class ForceCurveError(Exception):
    """Why force curves cannot be computed (shown to the user as is)."""


def _force_definitions(node):
    record = scene_record(node)
    if record is None:
        raise ForceCurveError(
            "Force curves need the PolyFEM node's record of the run "
            f"(input/{SCENE_RECORD_FILE} next to the output folder), which "
            "PolyFEM (Dev) 2.0 writes with every export since 2026-10-02: "
            "write or run the scene again.")
    sets = record.get("force_sets") or []
    if not sets:
        raise ForceCurveError(
            "The run has no sideset or obstacle to measure: add a sideset (a "
            "selection without a condition is enough) and write the scene "
            "again.")
    return record, sets


def force_set_menu(kwargs):
    """Toggle/replace menu of the record's force sets (1-based tokens)."""
    try:
        _, sets = _force_definitions(kwargs["node"])
    except ForceCurveError:
        return []
    items = []
    for index, entry in enumerate(sets, 1):
        items.extend((str(index), entry.get("label", f"Set {index}")))
    return items


def _selected_force_sets(node, count):
    text = node.evalParm("force_sets").split()
    chosen = sorted({int(token) for token in text if token.isdigit()
                     and 1 <= int(token) <= count})
    return chosen or list(range(1, count + 1))


def _face_lattice(vertices, faces, order):
    """Positions of the Lagrange nodes of straight faces of a given order:
    the corners, plus edge and face nodes from order 2 on."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(len(faces), -1)
    points = [vertices]
    if order < 2 or not len(faces):
        return vertices
    if faces.shape[1] < 4:
        faces = np.column_stack(
            (faces, np.full((len(faces), 4 - faces.shape[1]), -1)))
    triangles = faces[faces[:, 3] < 0][:, :3]
    quads = faces[faces[:, 3] >= 0]
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            if max(i, j, k) == order or not len(triangles):
                continue  # a corner
            points.append((i * vertices[triangles[:, 0]]
                           + j * vertices[triangles[:, 1]]
                           + k * vertices[triangles[:, 2]]) / order)
    for a in range(order + 1):
        for b in range(order + 1):
            if a in (0, order) and b in (0, order) or not len(quads):
                continue
            u, v = a / order, b / order
            points.append((1 - u) * (1 - v) * vertices[quads[:, 0]]
                          + u * (1 - v) * vertices[quads[:, 1]]
                          + u * v * vertices[quads[:, 2]]
                          + (1 - u) * v * vertices[quads[:, 3]])
    return np.concatenate(points)


def _force_nodes(record_key, mesh, definitions, chosen):
    """{set: (output points of its nodes, nodes looked for, not found)}:
    each node once (a representative of its coincidence group), matched by
    position within the set's body. Cached per record and topology."""
    key = (record_key, str(mesh.get("topo_key")), tuple(chosen))
    cached = _FORCE_NODES_CACHE.get(key)
    if cached is not None:
        return cached
    points = np.asarray(mesh["points"], dtype=np.float64)
    body = mesh["point_data"].get("body_ids")
    if body is not None:
        body = np.rint(np.asarray(body, dtype=np.float64).reshape(
            len(points), -1)[:, 0]).astype(np.int64)
    _, representative = _coincidence_groups(points, body)
    reps = np.flatnonzero(representative)
    extent = float(np.linalg.norm(np.ptp(points, axis=0))) or 1.0
    tolerance = 1e-6 * extent
    result = {}
    for index in chosen:
        entry = definitions[index - 1]
        vertices = np.asarray(entry.get("vertices", []),
                              dtype=np.float64).reshape(-1, 3)
        if entry.get("kind") == "faces":
            targets = _face_lattice(vertices, entry.get("faces", []),
                                    int(entry.get("order", 1)))
        else:
            targets = vertices
        candidates = reps if body is None \
            else reps[body[reps] == int(entry.get("body", -1))]
        if not len(candidates) or not len(targets):
            result[index] = (np.zeros(0, dtype=np.int64), len(targets),
                             len(targets))
            continue
        nearest = _nearest_rows(points[candidates], targets)
        distance = np.linalg.norm(points[candidates[nearest]] - targets,
                                  axis=1)
        found = distance <= tolerance
        result[index] = (np.unique(candidates[nearest[found]]), len(targets),
                         int((~found).sum()))
    if len(_FORCE_NODES_CACHE) > 16:
        _FORCE_NODES_CACHE.clear()
    _FORCE_NODES_CACHE[key] = result
    return result


def _point_vectors(mesh, name):
    values = mesh["point_data"].get(name)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    count = len(mesh["points"])
    values = values.reshape(len(values), -1)[:, :3]
    if len(values) != count:  # some fields omit obstacle rows: zero-fill
        padded = np.zeros((count, values.shape[1]))
        padded[:min(len(values), count)] = values[:count]
        values = padded
    return values


def _frame_set_values(mesh, nodes):
    """{set: (reaction, contact, mean displacement)} of one frame."""
    terms = [_point_vectors(mesh, f"{term}_forces") for term in FORCE_TERMS]
    terms = [values for values in terms if values is not None]
    if not terms:
        raise ForceCurveError(
            "This run has no nodal forces: turn on Output > Nodal Forces "
            "(force curves) on the PolyFEM node and run it again.")
    total = np.sum(terms, axis=0)
    contact = [_point_vectors(mesh, name)
               for name in ("contact_forces", "friction_forces")]
    contact = [values for values in contact if values is not None]
    contact = np.sum(contact, axis=0) if contact else np.zeros_like(total)
    displacement = _point_vectors(mesh, "solution")
    if displacement is None:
        displacement = np.zeros_like(total)
    out = {}
    for index, (members, _, _) in nodes.items():
        if not len(members):
            out[index] = (np.full(3, np.nan),) * 3
            continue
        out[index] = (-total[members].sum(axis=0),
                      contact[members].sum(axis=0),
                      displacement[members].mean(axis=0))
    return out


def _volume_block(blocks):
    mesh = blocks.get("Volume")
    if mesh is None:
        raise ForceCurveError(
            "This frame has no Volume block, where PolyFEM writes nodal "
            "forces.")
    return mesh


def force_curves(node):
    """Every step's reaction, contact force and mean displacement of the
    chosen force sets: {"steps", "times", "sets", "labels", "reaction",
    "contact", "displacement" (steps x sets x 3), "nodes", "unmatched"}.
    Memoized on the PVD, the record and the choice of sets."""
    path = node.evalParm("PVD_file")
    if not path or not os.path.isfile(path):
        raise ForceCurveError("Load a PVD file first.")
    record, definitions = _force_definitions(node)
    record_key = _file_key(scene_record_path(path))
    chosen = _selected_force_sets(node, len(definitions))
    entries = read_pvd(path)
    memo_key = (_file_key(path), record_key, tuple(chosen), len(entries))
    memo = _FORCE_CURVES.get(node.sessionId())
    if memo is not None and memo["key"] == memo_key:
        return memo
    reaction = np.full((len(entries), len(chosen), 3), np.nan)
    contact = np.full_like(reaction, np.nan)
    displacement = np.full_like(reaction, np.nan)
    nodes = {}
    for step in range(len(entries)):
        try:
            mesh = _volume_block(load_frame(path, step))
        except ForceCurveError:
            raise
        except Exception:
            continue  # an unreadable step (a run still writing) stays NaN
        nodes = _force_nodes(record_key, mesh, definitions, chosen)
        values = _frame_set_values(mesh, nodes)
        for column, index in enumerate(chosen):
            reaction[step, column], contact[step, column], \
                displacement[step, column] = values[index]
    memo = {"key": memo_key, "steps": np.arange(len(entries)),
            "times": np.array([entry[0] for entry in entries]),
            "sets": chosen,
            "labels": [definitions[i - 1].get("label", f"Set {i}")
                       for i in chosen],
            "kinds": [definitions[i - 1].get("kind") for i in chosen],
            "reaction": reaction, "contact": contact,
            "displacement": displacement,
            "nodes": {i: len(nodes.get(i, ([],))[0]) for i in chosen},
            "unmatched": {i: nodes.get(i, (None, 0, 0))[2] for i in chosen}}
    _FORCE_CURVES[node.sessionId()] = memo
    return memo


def _magnitude(vectors):
    return np.linalg.norm(vectors, axis=-1)


def write_force_csv(curves, path):
    """One row per step: time, then per set its reaction, contact force and
    mean displacement (x, y, z, magnitude)."""
    import csv
    header = ["step", "time"]
    for label in curves["labels"]:
        for quantity in ("reaction", "contact force", "displacement"):
            header.extend(f"{label}: {quantity} {axis}"
                          for axis in FORCE_COMPONENTS)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(len(curves["steps"])):
            row = [int(curves["steps"][step]), repr(float(curves["times"][step]))]
            for column in range(len(curves["sets"])):
                for name in ("reaction", "contact", "displacement"):
                    vector = curves[name][step, column]
                    row.extend(repr(float(v)) for v in vector)
                    row.append(repr(float(_magnitude(vector))))
            writer.writerow(row)
    return path


def force_csv_path(node):
    return os.path.join(os.path.dirname(os.path.abspath(
        node.evalParm("PVD_file"))), FORCE_CSV)


def compute_force_curves(kwargs):
    """Compute Force Curves button: every step, written to the CSV."""
    node = kwargs["node"]
    try:
        curves = force_curves(node)
    except ForceCurveError as exc:
        _message(str(exc))
        return None
    path = force_csv_path(node)
    try:
        write_force_csv(curves, path)
    except OSError as exc:
        _message(f"Could not write {path}: {exc}")
        return curves
    missing = [f"{label}: {curves['unmatched'][index]} nodes not found"
               for index, label in zip(curves["sets"], curves["labels"])
               if curves["unmatched"][index]]
    note = ("\n" + "\n".join(missing)) if missing else ""
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(
            f"Force curves of {len(curves['sets'])} set(s) over "
            f"{len(curves['steps'])} steps written to {path}",
            hou.severityType.ImportantMessage)
    if note:
        _message("Some nodes of the force sets were not found in the "
                 "results (the mesh or the record changed since the run?):"
                 + note)
    _update_chart(node)
    return curves


def _vector_text(vector):
    return "(" + ", ".join(f"{float(v):.5g}" for v in vector) + ")"


def force_readout_text(node):
    """Read-only Force Curves readout: each set's forces at this frame."""
    hou.frame()  # time-dependent, so it follows the playbar
    path = node.evalParm("PVD_file")
    if not path:
        return "Load a PVD file."
    try:
        record, definitions = _force_definitions(node)
        chosen = _selected_force_sets(node, len(definitions))
        step = min(entry_index(node), len(read_pvd(path)) - 1)
        mesh = _volume_block(load_frame(path, step))
        nodes = _force_nodes(_file_key(scene_record_path(path)), mesh,
                             definitions, chosen)
        values = _frame_set_values(mesh, nodes)
    except ForceCurveError as exc:
        return str(exc)
    except Exception as exc:
        return f"Force readout unavailable: {exc}"
    lines = []
    for index in chosen:
        reaction, contact, displacement = values[index]
        label = definitions[index - 1].get("label", f"Set {index}")
        lines.append(
            f"{label}: reaction {_vector_text(reaction)} |{float(_magnitude(reaction)):.5g}|, "
            f"contact {_vector_text(contact)}, displacement "
            f"{_vector_text(displacement)}")
    return "\n".join(lines)


def _chart_series(node, curves):
    """(x, y, x label, y label, title) of the chosen chart."""
    token = node.evalParm("force_chart_set")
    try:
        column = curves["sets"].index(int(token))
    except (ValueError, TypeError):
        column = 0
    quantity = _menu_token(node, "force_chart_quantity")
    component = _menu_token(node, "force_chart_component")
    against = _menu_token(node, "force_chart_against")
    vectors = curves[quantity][:, column]
    if component == "magnitude":
        y = _magnitude(vectors)
    else:
        y = vectors[:, "xyz".index(component)]
    if against == "time":
        x, x_label = curves["times"], "time"
    else:
        moved = curves["displacement"][:, column]
        x = _magnitude(moved) if component == "magnitude" \
            else moved[:, "xyz".index(component)]
        x_label = f"mean displacement {component}"
    name = "reaction" if quantity == "reaction" else "contact force"
    return (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
            x_label, f"{name} {component}", curves["labels"][column])


def _chart_class():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ForceChart(QtWidgets.QWidget):
        """A small line chart of one force curve; the marker follows the
        playbar."""

        def __init__(self, node, parent=None):
            super().__init__(parent, QtCore.Qt.Window)
            self.node_id = node.sessionId()
            self.series = None
            self.marker = None
            self.setWindowTitle(f"Force Curve - {node.name()}")
            self.resize(640, 420)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._follow)
            self.timer.start(200)
            self._frame = None

        def set_series(self, series):
            self.series = series
            self.update()

        def _follow(self):
            node = hou.nodeBySessionId(self.node_id)
            if node is None:
                self.close()
                return
            frame = hou.frame()
            if frame != self._frame:
                self._frame = frame
                self.update()

        def closeEvent(self, event):
            self.timer.stop()
            _CHART_WINDOWS.pop(self.node_id, None)
            super().closeEvent(event)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.fillRect(self.rect(), self.palette().window())
            draw_force_chart(painter, self.rect(), self.series,
                             self._marker_index(), self.palette())
            painter.end()

        def _marker_index(self):
            node = hou.nodeBySessionId(self.node_id)
            if node is None or self.series is None:
                return None
            try:
                return entry_index(node)
            except Exception:
                return None

    return ForceChart


def draw_force_chart(painter, rect, series, marker, palette=None):
    """Draw axes, ticks, the curve and the marker (step `marker`) with a
    QPainter; also used to render the chart headlessly in tests."""
    from PySide6 import QtCore, QtGui
    text = palette.text().color() if palette is not None \
        else QtGui.QColor(30, 30, 30)
    if series is None:
        painter.setPen(text)
        painter.drawText(rect, QtCore.Qt.AlignCenter,
                         "Compute Force Curves first.")
        return
    x, y, x_label, y_label, title = series
    valid = np.isfinite(x) & np.isfinite(y)
    left, top, right, bottom = 70, 34, 18, 46
    area = QtCore.QRectF(rect.left() + left, rect.top() + top,
                         rect.width() - left - right,
                         rect.height() - top - bottom)
    painter.setPen(text)
    painter.drawText(QtCore.QRectF(rect.left(), rect.top() + 6, rect.width(),
                                   20), QtCore.Qt.AlignCenter, title)
    if not valid.any():
        painter.drawText(area, QtCore.Qt.AlignCenter, "No values.")
        return

    def span(values):
        low, high = float(values.min()), float(values.max())
        if high - low <= 1e-300:
            pad = abs(low) * 0.05 or 1.0
            return low - pad, high + pad
        pad = 0.05 * (high - low)
        return low - pad, high + pad

    x_low, x_high = span(x[valid])
    y_low, y_high = span(y[valid])

    def to_screen(px, py):
        return QtCore.QPointF(
            area.left() + (px - x_low) / (x_high - x_low) * area.width(),
            area.bottom() - (py - y_low) / (y_high - y_low) * area.height())

    grid = QtGui.QColor(text)
    grid.setAlpha(60)
    for i in range(6):
        fx = x_low + (x_high - x_low) * i / 5
        fy = y_low + (y_high - y_low) * i / 5
        painter.setPen(grid)
        painter.drawLine(to_screen(fx, y_low), to_screen(fx, y_high))
        painter.drawLine(to_screen(x_low, fy), to_screen(x_high, fy))
        painter.setPen(text)
        point = to_screen(fx, y_low)
        painter.drawText(QtCore.QRectF(point.x() - 40, area.bottom() + 4, 80,
                                       16), QtCore.Qt.AlignCenter,
                         f"{fx:.3g}")
        point = to_screen(x_low, fy)
        painter.drawText(QtCore.QRectF(rect.left(), point.y() - 8, left - 6,
                                       16),
                         QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                         f"{fy:.3g}")
    painter.setPen(text)
    painter.drawRect(area)
    painter.drawText(QtCore.QRectF(area.left(), rect.bottom() - 22,
                                   area.width(), 18), QtCore.Qt.AlignCenter,
                     x_label)
    painter.save()
    painter.translate(rect.left() + 12, area.center().y())
    painter.rotate(-90)
    painter.drawText(QtCore.QRectF(-area.height() / 2, -10, area.height(),
                                   20), QtCore.Qt.AlignCenter, y_label)
    painter.restore()
    curve = QtGui.QPolygonF([to_screen(px, py) for px, py in
                             zip(x[valid], y[valid])])
    pen = QtGui.QPen(QtGui.QColor(31, 119, 180))
    pen.setWidthF(2.0)
    painter.setPen(pen)
    painter.drawPolyline(curve)
    if marker is not None and 0 <= marker < len(x) and valid[marker]:
        point = to_screen(x[marker], y[marker])
        painter.setBrush(QtGui.QColor(214, 39, 40))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(point, 5, 5)


def _update_chart(node):
    window = _CHART_WINDOWS.get(node.sessionId())
    if window is None:
        return
    curves = _FORCE_CURVES.get(node.sessionId())
    window.set_series(_chart_series(node, curves) if curves else None)


def show_force_chart(kwargs):
    """Show Chart button: the chosen curve in a window that follows the
    playbar (computes the curves first if needed)."""
    node = kwargs["node"]
    try:
        curves = force_curves(node)
    except ForceCurveError as exc:
        _message(str(exc))
        return None
    if not hou.isUIAvailable():
        return _chart_series(node, curves)
    window = _CHART_WINDOWS.get(node.sessionId())
    if window is None:
        window = _chart_class()(node, hou.qt.mainWindow())
        _CHART_WINDOWS[node.sessionId()] = window
    window.set_series(_chart_series(node, curves))
    window.show()
    window.raise_()
    return window


def force_chart_changed(kwargs):
    _update_chart(kwargs["node"])


def open_force_csv(kwargs):
    """Open the CSV with the system's default program."""
    node = kwargs["node"]
    path = force_csv_path(node)
    if not os.path.isfile(path):
        if compute_force_curves(kwargs) is None or not os.path.isfile(path):
            return
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)  # noqa: S606 -- the user's own results file
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        _message(f"Could not open {path}: {exc}")


# =============================================================================
# Export: the deformed mesh, and data over time (2026-10-03)
#
# Two output functions on the Export tab:
#  * Deformed Mesh -- for each simulated geometry of the run, the run's own
#    .msh (staged in <run>/input/ by the PolyFEM node) with only its node
#    coordinates moved to one step's deformed positions: the same node tags,
#    elements, physical groups and names, and file format, so the PolyFEM
#    node treats it as the same mesh. Linear elements only (the edge
#    curvature of a P2+ run is reported, not kept). The exported shape is a
#    new rest shape: it carries no stress.
#  * Data Over Time -- per output step, statistics of any quantity Read PVD
#    can color by, over named regions, plus the force curves, as one
#    plot-ready table in an Excel workbook (for presentation plots: no
#    charts in the file), with optional per element / node sheets.
#
# Both rest on PolyFEM's high-order volume output (checked 2026-10-02): one
# volume cell per FE element, in FE order -- the enabled simulated
# geometries of params.json concatenated, each mesh in file order -- every
# element with its own bitwise-identical copies of its nodes, the obstacle
# rows last (body id 0). PolyFEM's rest transform is rebuilt from
# params.json exactly as GeometryReader.cpp applies it.
# =============================================================================

EXPORT_FOLDER = "exports"
_TET_TYPES = (10, 24, 71)
_HEX_TYPES = (12, 25, 72)
_CORNERS = {"tet": 4, "hex": 8}
_PARAMS_CACHE = {}
_TOPOLOGY_CACHE = {}
_MATCH_CACHE = {}
_MSH_CACHE = {}


class ExportError(Exception):
    """Why an export cannot be done (shown to the user as is)."""


def run_folder(pvd_path):
    """The PolyFEM node's working directory of <run>/output/<name>.pvd."""
    return os.path.dirname(os.path.dirname(os.path.abspath(pvd_path)))


def export_folder(node):
    """Export Folder, or <run>/exports when it is empty."""
    text = node.evalParm("export_folder").strip()
    if text:
        return os.path.abspath(os.path.expanduser(text))
    return os.path.join(run_folder(node.evalParm("PVD_file")), EXPORT_FOLDER)


def _open_with_system(path):
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)  # noqa: S606 -- the user's own results
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        _message(f"Could not open {path}: {exc}")


def open_export_folder(kwargs):
    """Open Folder button: the export folder in Finder/Explorer."""
    node = kwargs["node"]
    if not node.evalParm("PVD_file"):
        _message("Load a PVD file first.")
        return
    folder = export_folder(node)
    os.makedirs(folder, exist_ok=True)
    _open_with_system(folder)


def run_params(pvd_path):
    """The run's input/params.json (as the PolyFEM node wrote it), or None."""
    path = os.path.join(run_folder(pvd_path), "input", "params.json")
    try:
        key = _file_key(path)
    except OSError:
        return None
    cached = _PARAMS_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with open(path) as handle:
            params = json.load(handle)
    except (OSError, ValueError):
        params = None
    if not isinstance(params, dict):
        params = None
    if len(_PARAMS_CACHE) > 8:
        _PARAMS_CACHE.clear()
    _PARAMS_CACHE[path] = (key, params)
    return params


def simulated_geometries(pvd_path, params):
    """[{number, entry, path}] of the enabled, simulated (non-obstacle)
    geometries of params.json; number is the entry's 1-based position, the
    PolyFEM node's geometry number (body ids are 1000 x number + subdomain)."""
    input_dir = os.path.join(run_folder(pvd_path), "input")
    entries = params.get("geometry", [])
    if isinstance(entries, dict):
        entries = [entries]
    result = []
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or entry.get("is_obstacle", False) \
                or not entry.get("enabled", True) \
                or entry.get("type", "mesh") != "mesh":
            continue
        mesh = entry.get("mesh")
        if not isinstance(mesh, str) or not mesh:
            continue
        path = mesh if os.path.isabs(mesh) else os.path.join(input_dir, mesh)
        result.append({"number": number, "entry": entry,
                       "path": os.path.normpath(path)})
    return result


def _axis_angle(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def polyfem_rotation(rotation, mode="xyz"):
    """PolyFEM's to_rotation_matrix (utils/JSONUtils.cpp): degrees; for an
    axis-order mode such as "xyz", entry j is the angle about axis j and the
    rotations are applied in the order the mode names them."""
    mode = str(mode or "xyz").lower()
    if isinstance(rotation, (int, float)):
        values = [0.0, 0.0, 0.0]
        values["xyz".index(mode[0])] = float(rotation)
        rotation = values
    values = [float(value) for value in (rotation or [])]
    if not values:
        return np.eye(3)
    if mode == "axis_angle":
        return _axis_angle(math.radians(values[0]), values[1:4])
    if mode == "quaternion":
        x, y, z, w = np.asarray(values[:4]) / np.linalg.norm(values[:4])
        return np.array(
            [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    radians = np.radians(np.asarray(values, dtype=np.float64))
    if mode == "rotation_vector":
        angle = float(np.linalg.norm(radians))
        return np.eye(3) if angle == 0 else _axis_angle(angle, radians / angle)
    matrix = np.eye(3)
    for axis_name in mode:
        axis = "xyz".index(axis_name)
        matrix = _axis_angle(radians[axis], np.eye(3)[axis]) @ matrix
    return matrix


def polyfem_rest_transform(entry, file_points):
    """(A, b) such that PolyFEM's rest positions are A x + b for the mesh
    file's positions x: GeometryReader.cpp's construct_affine_transformation
    -- scale, then rotate about the origin, then translate."""
    number = entry.get("_number", "")
    advanced = entry.get("advanced") or {}
    if advanced.get("normalize_mesh"):
        raise ExportError(
            f"Geometry {number} is normalized by PolyFEM (advanced/"
            "normalize_mesh), which the export cannot reproduce.")
    if entry.get("unit"):
        raise ExportError(
            f"Geometry {number} has its own length unit ('unit' in "
            "params.json), which the export does not convert.")
    if int(entry.get("n_refs", 0) or 0) > 0:
        raise ExportError(
            f"Geometry {number} was refined by PolyFEM (n_refs): the "
            "simulated mesh is not the mesh file.")
    transform = entry.get("transformation") or {}
    dimensions = transform.get("dimensions")
    if isinstance(dimensions, list) and dimensions:
        extent = np.ptp(np.asarray(file_points, dtype=np.float64), axis=0)
        extent[extent == 0] = 1.0
        values = [float(value) for value in dimensions] + [0.0, 0.0, 0.0]
        scale = np.asarray(values[:3]) / extent
    elif isinstance(transform.get("scale"), (int, float)):
        scale = np.full(3, float(transform["scale"]))
    else:
        values = [float(value) for value in transform.get("scale") or []]
        scale = np.ones(3) if not values \
            else np.asarray((values + [0.0, 0.0, 0.0])[:3])
    rotation = polyfem_rotation(transform.get("rotation", []),
                                transform.get("rotation_mode", "xyz"))
    translation = [float(value) for value in transform.get("translation")
                   or []]
    return (rotation @ np.diag(scale),
            np.asarray((translation + [0.0, 0.0, 0.0])[:3]))


def _parsed_mesh(path):
    key = _file_key(path)
    cached = _MSH_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        parsed = read_msh(path)
    except (OSError, MshParseError, ValueError, IndexError) as exc:
        raise ExportError(f"Could not read the mesh {path}: {exc}")
    if len(_MSH_CACHE) > 4:
        _MSH_CACHE.clear()
    _MSH_CACHE[path] = (key, parsed)
    return parsed


def volume_mesh(pvd_path, step):
    """The parsed Volume block of one output step."""
    blocks = _frame_block_paths(pvd_path, step)
    path = blocks.get("Volume")
    if path is None or not os.path.isfile(path):
        raise ExportError(f"Step {step} has no Volume block.")
    return read_mesh_cached(path)


# ---- elements and nodes of the output -------------------------------------

def volume_topology(mesh):
    """Elements (the volume cells, in PolyFEM's element order) and nodes
    (each element's copies of a vertex, merged per body) of a Volume block;
    cached per topology."""
    key = (str(mesh.get("topo_key")), len(mesh["points"]))
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached
    if mesh.get("cell_types") is None:
        raise ExportError("This frame was read by an older Read PVD: press "
                          "Clear Cache and try again.")
    types = np.asarray(mesh["cell_types"], dtype=np.int64)
    is_hex = np.isin(types, _HEX_TYPES)
    cells = np.flatnonzero(np.isin(types, _TET_TYPES) | is_hex)
    if not len(cells):
        raise ExportError("This result has no tetrahedra or hexahedra.")
    starts = np.asarray(mesh["cell_starts"], dtype=np.int64)[cells]
    sizes = np.asarray(mesh["cell_sizes"], dtype=np.int64)[cells]
    connectivity = np.asarray(mesh["cell_connectivity"], dtype=np.int64)
    hexes = is_hex[cells]
    corner_count = np.where(hexes, 8, 4)
    corners = np.full((len(cells), 8), -1, dtype=np.int64)
    for corner in range(8):
        mask = corner < corner_count
        corners[mask, corner] = connectivity[starts[mask] + corner]
    owner = np.repeat(np.arange(len(cells)), sizes)
    within = np.arange(int(sizes.sum())) - np.repeat(
        np.cumsum(sizes) - sizes, sizes)
    members = connectivity[np.repeat(starts, sizes) + within]
    points = np.asarray(mesh["points"], dtype=np.float64)
    body_ids = mesh["point_data"].get("body_ids")
    if body_ids is not None and len(body_ids) == len(points):
        point_body = np.rint(np.asarray(body_ids, dtype=np.float64).reshape(
            len(points), -1)[:, 0]).astype(np.int64)
    else:
        point_body = np.zeros(len(points), dtype=np.int64)
    used = np.unique(members)
    keys = np.column_stack((points[used], point_body[used]))
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    point_node = np.full(len(points), -1, dtype=np.int64)
    point_node[used] = np.asarray(inverse).ravel()
    node_point = used[first]
    point_element = np.full(len(points), -1, dtype=np.int64)
    point_element[members] = owner
    topology = {
        "key": key, "cells": cells, "hex": hexes, "starts": starts,
        "sizes": sizes, "corners": corners, "corner_count": corner_count,
        "owner": owner, "members": members,
        "body": point_body[corners[:, 0]], "point_body": point_body,
        "has_body_ids": body_ids is not None,
        "point_node": point_node, "node_point": node_point,
        "node_body": point_body[node_point], "point_element": point_element,
        "n_elements": len(cells), "n_nodes": len(node_point),
        "connectivity": connectivity,
    }
    if len(_TOPOLOGY_CACHE) > 1:
        _TOPOLOGY_CACHE.clear()
    _TOPOLOGY_CACHE[key] = topology
    return topology


# Quadratic tetrahedron (VTK node order: corners, then the edges (0,1),
# (1,2), (0,2), (0,3), (1,3), (2,3)) and a degree-3 rule, exact for the
# cubic det J of a P2 element.
_P2_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
_DEGREE3_TET_RULE = (
    ((0.25, 0.25, 0.25), -0.8),
    ((0.5, 1 / 6, 1 / 6), 0.45), ((1 / 6, 0.5, 1 / 6), 0.45),
    ((1 / 6, 1 / 6, 0.5), 0.45), ((1 / 6, 1 / 6, 1 / 6), 0.45))
_HEX_REFERENCE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                          dtype=np.float64)


def _tet_volumes(a, b, c, d):
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _p2_tet_volumes(nodes):
    """Exact volume of P2 tets from their 10 node positions (n, 10, 3)."""
    gradients = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    volume = np.zeros(len(nodes))
    for (xi, eta, zeta), weight in _DEGREE3_TET_RULE:
        lam = (1.0 - xi - eta - zeta, xi, eta, zeta)
        dN = np.zeros((10, 3))
        for i in range(4):
            dN[i] = (4.0 * lam[i] - 1.0) * gradients[i]
        for k, (i, j) in enumerate(_P2_EDGES, 4):
            dN[k] = 4.0 * (lam[j] * gradients[i] + lam[i] * gradients[j])
        jacobian = np.einsum("nki,kj->nij", nodes, dN)
        volume += weight / 6.0 * np.linalg.det(jacobian)
    return volume


def hex_volumes(nodes):
    """Exact volume of trilinear hexes from their 8 corners (n, 8, 3) in
    VTK / Gmsh order (2 x 2 x 2 Gauss points)."""
    g = 0.5 / math.sqrt(3.0)
    volume = np.zeros(len(nodes))
    for u in (0.5 - g, 0.5 + g):
        for v in (0.5 - g, 0.5 + g):
            for w in (0.5 - g, 0.5 + g):
                dN = np.empty((8, 3))
                for k, (a, b, c) in enumerate(_HEX_REFERENCE):
                    fu, fv, fw = (u if a else 1 - u), (v if b else 1 - v), \
                        (w if c else 1 - w)
                    su, sv, sw = (1 if a else -1), (1 if b else -1), \
                        (1 if c else -1)
                    dN[k] = (su * fv * fw, fu * sv * fw, fu * fv * sw)
                jacobian = np.einsum("nki,kj->nij", nodes, dN)
                volume += np.linalg.det(jacobian) / 8.0
    return volume


def element_volumes(topology, positions):
    """Signed volume of every element with its output points at
    `positions`: exact for straight and P2 tets and trilinear hexes;
    P3/P4 tets as the sum of Read PVD's display sub-tets."""
    positions = np.asarray(positions, dtype=np.float64)
    volume = np.zeros(topology["n_elements"])
    corners = topology["corners"]
    sizes = topology["sizes"]
    connectivity = topology["connectivity"]
    starts = topology["starts"]
    tets = ~topology["hex"]
    linear = tets & (sizes == 4)
    if linear.any():
        c = corners[linear, :4]
        volume[linear] = _tet_volumes(*(positions[c[:, i]] for i in range(4)))
    quadratic = tets & (sizes == 10)
    if quadratic.any():
        nodes = connectivity[starts[quadratic][:, None] + np.arange(10)]
        volume[quadratic] = _p2_tet_volumes(positions[nodes])
    for size, table in ((20, TET20_SUBDIV), (35, TET35_SUBDIV)):
        mask = tets & (sizes == size)
        if not mask.any():
            continue
        nodes = connectivity[starts[mask][:, None] + np.arange(size)]
        sub = nodes[:, table]                               # (n, s, 4)
        volume[mask] = _tet_volumes(
            *(positions[sub[:, :, i].ravel()] for i in range(4))).reshape(
                len(nodes), -1).sum(axis=1)
    other = tets & ~(linear | quadratic | (sizes == 20) | (sizes == 35))
    if other.any():  # P5+: the corner tet
        c = corners[other, :4]
        volume[other] = _tet_volumes(*(positions[c[:, i]] for i in range(4)))
    hexes = topology["hex"]
    if hexes.any():
        volume[hexes] = hex_volumes(positions[corners[hexes]])
    return volume


# ---- the run's meshes, element by element ---------------------------------

def mesh_match(pvd_path, mesh, topology):
    """Match the output elements to the run's mesh files, element by
    element: which geometry and file element each output element is, and
    which file node each output corner point is. Raises ExportError (with
    the reason) when the output does not match the meshes of params.json.
    """
    params = run_params(pvd_path)
    if params is None:
        raise ExportError(
            "This needs the run's input/params.json (written by the PolyFEM "
            "node) next to the output folder.")
    geometries = simulated_geometries(pvd_path, params)
    if not geometries:
        raise ExportError("params.json lists no simulated geometry.")
    missing = [g for g in geometries if not os.path.isfile(g["path"])]
    if missing:
        raise ExportError(
            "The run's mesh file is missing: "
            + ", ".join(f"geometry {g['number']} ({g['path']})"
                        for g in missing)
            + ". The PolyFEM node keeps a copy in <run>/input/.")
    key = (topology["key"], _file_key(os.path.join(
        run_folder(pvd_path), "input", "params.json")),
        tuple(_file_key(g["path"]) for g in geometries))
    cached = _MATCH_CACHE.get(key)
    if cached is not None:
        return cached
    points = np.asarray(mesh["points"], dtype=np.float64)
    n_elements = topology["n_elements"]
    total = 0
    for geometry in geometries:
        cells = _parsed_mesh(geometry["path"])["cells"]
        total += sum(len(cells[family]["order"]) for family in ("tet", "hex")
                     if family in cells)
    if total != n_elements:
        raise ExportError(
            f"The result has {n_elements} elements and the run's meshes "
            f"{total}: the result is a sampled visualization mesh (Output > "
            "High Order Mesh off on the PolyFEM node, or a curved mesh), or "
            "the meshes changed since the run.")
    point_file_node = np.full(len(points), -1, dtype=np.int64)
    point_geometry = np.zeros(len(points), dtype=np.int64)
    element_geometry = np.zeros(n_elements, dtype=np.int64)
    element_number = np.zeros(n_elements, dtype=np.int64)
    start = 0
    matched = []
    for geometry in geometries:
        number = geometry["number"]
        parsed = _parsed_mesh(geometry["path"])
        cells = parsed["cells"]
        unsupported = [family for family in ("prism", "pyramid")
                       if family in cells]
        if unsupported:
            raise ExportError(
                f"Geometry {number} has {' and '.join(unsupported)} "
                "elements; the export supports tetrahedra and hexahedra.")
        families = [family for family in ("tet", "hex") if family in cells]
        if not families:
            raise ExportError(f"Geometry {number}'s mesh has no tetrahedra "
                              "or hexahedra.")
        curved = [family for family in families
                  if cells[family]["num_nodes"] != _CORNERS[family]]
        count = sum(len(cells[family]["order"]) for family in families)
        is_hex = np.zeros(count, dtype=bool)
        file_corners = np.full((count, 8), -1, dtype=np.int64)
        for family in families:
            order = cells[family]["order"]
            if family == "hex":
                is_hex[order] = True
            file_corners[order, :_CORNERS[family]] = cells[family]["corners"]
        entry = dict(geometry["entry"], _number=number)
        A, b = polyfem_rest_transform(entry, parsed["points"])
        # einsum, not @: Accelerate's matmul raises spurious divide-by-zero
        # warnings on macOS (review P2-12)
        rest = np.einsum("ij,nj->ni", A, np.asarray(
            parsed["points"], dtype=np.float64)) + b
        stop = start + count
        if stop > n_elements or not np.array_equal(
                topology["hex"][start:stop], is_hex):
            raise ExportError(
                f"The result's elements do not follow geometry {number}'s "
                f"mesh ({os.path.basename(geometry['path'])}): the result is "
                "a sampled visualization mesh (Output > High Order Mesh off, "
                "or a curved mesh), or the mesh changed since the run.")
        extent = float(np.linalg.norm(np.ptp(rest, axis=0))) or 1.0
        tolerance = 1e-6 * extent
        output_corners = topology["corners"][start:stop]
        for hexes in (False, True):
            local = np.flatnonzero(is_hex == hexes)
            k = 8 if hexes else 4
            for chunk in range(0, len(local), 20000):
                rows = local[chunk:chunk + 20000]
                out_points = output_corners[rows, :k]
                file_nodes = file_corners[rows, :k]
                distance = np.linalg.norm(
                    points[out_points][:, :, None, :]
                    - rest[file_nodes][:, None, :, :], axis=3)
                nearest = distance.argmin(axis=2)
                worst = float(np.take_along_axis(
                    distance, nearest[:, :, None], axis=2).max())
                if worst > tolerance or np.any(
                        np.sort(nearest, axis=1)
                        != np.arange(k)[None, :]):
                    raise ExportError(
                        f"The result's elements do not sit on geometry "
                        f"{number}'s mesh nodes (off by {worst:.3g}): the "
                        "result is a sampled visualization mesh (Output > "
                        "High Order Mesh off), or the mesh file, its "
                        "transform or the run changed since it was written.")
                point_file_node[out_points.ravel()] = np.take_along_axis(
                    file_nodes, nearest, axis=1).ravel()
        owned = (topology["owner"] >= start) & (topology["owner"] < stop)
        point_geometry[topology["members"][owned]] = number
        element_geometry[start:stop] = number
        element_number[start:stop] = np.arange(1, count + 1)
        matched.append({"number": number, "path": geometry["path"],
                        "entry": geometry["entry"], "parsed": parsed,
                        "A": A, "b": b, "rest": rest, "start": start,
                        "stop": stop, "is_hex": is_hex,
                        "file_corners": file_corners, "curved": curved})
        start = stop
    if start != n_elements:
        raise ExportError(
            f"The result has {n_elements} elements and the run's meshes "
            f"{start}: the result is a sampled visualization mesh (Output > "
            "High Order Mesh off, or a curved mesh), or the meshes changed "
            "since the run.")
    # every copy of a node must be the same file node
    corner_points = np.flatnonzero(point_file_node >= 0)
    pairs = np.unique(np.column_stack((
        topology["point_node"][corner_points],
        point_file_node[corner_points])), axis=0)
    if len(pairs) != len(np.unique(pairs[:, 0])):
        raise ExportError("Copies of one result node match different mesh "
                          "nodes; the result does not belong to these meshes.")
    match = {"geometries": matched, "point_file_node": point_file_node,
             "point_geometry": point_geometry,
             "element_geometry": element_geometry,
             "element_number": element_number}
    if len(_MATCH_CACHE) > 1:
        _MATCH_CACHE.clear()
    _MATCH_CACHE[key] = match
    return match


# ---- Deformed Mesh ---------------------------------------------------------

def export_mesh_step(node, count):
    """The output step the Deformed Mesh export writes."""
    if _menu_token(node, "export_mesh_step_mode") == "number":
        step = int(node.evalParm("export_mesh_step"))
        if step < 0 or step >= count:
            raise ExportError(f"Step {step} does not exist: the run has steps "
                              f"0 to {count - 1}.")
        return step
    return max(0, min(entry_index(node), count - 1))


def export_mesh_step_text(node):
    """Read-only line: the step Export Deformed Mesh writes, and its time."""
    hou.frame()  # time-dependent, so it follows the playbar
    times = _pvd_times(node)
    if not times:
        return "Load a PVD file."
    try:
        step = export_mesh_step(node, len(times))
    except ExportError as exc:
        return str(exc)
    return f"Step {step} of {len(times) - 1}: t = {times[step]:.6g}"


def export_geometry_menu(kwargs):
    """Toggle menu: the run's simulated geometries."""
    path = kwargs["node"].evalParm("PVD_file")
    params = run_params(path) if path else None
    if params is None:
        return []
    items = []
    for geometry in simulated_geometries(path, params):
        items.extend((str(geometry["number"]),
                      f"Geometry {geometry['number']} "
                      f"({os.path.basename(geometry['path'])})"))
    return items


def _solution_vectors(mesh):
    for name in ("solution", "displacement"):
        values = mesh["point_data"].get(name)
        if values is not None:
            values = np.asarray(values, dtype=np.float64).reshape(
                len(values), -1)
            if values.shape[1] >= 3:
                return values[:, :3]
    return None


def _p2_curvature(geometry, topology, points, deformed):
    """Largest distance of a deformed edge/face node from the straight
    element through the deformed corners, as a fraction of the element's
    mean edge length, and the element (file number) where it occurs."""
    rows = np.arange(geometry["start"], geometry["stop"])
    rows = rows[(~topology["hex"][rows]) & (topology["sizes"][rows] > 4)]
    if not len(rows):
        return None
    worst, where = 0.0, 0
    for chunk in range(0, len(rows), 20000):
        part = rows[chunk:chunk + 20000]
        for n in np.unique(topology["sizes"][part]):
            sub = part[topology["sizes"][part] == n]
            nodes = topology["connectivity"][
                topology["starts"][sub][:, None] + np.arange(n)]
            rest = points[nodes]
            moved = deformed[nodes]
            T = np.stack([rest[:, i] - rest[:, 0] for i in (1, 2, 3)], axis=2)
            local = np.linalg.solve(T[:, None, :, :],
                                    (rest[:, 4:] - rest[:, :1])[..., None])
            Tm = np.stack([moved[:, i] - moved[:, 0] for i in (1, 2, 3)],
                          axis=2)
            straight = moved[:, :1] + np.einsum("nij,nkj->nki", Tm,
                                                local[..., 0])
            gap = np.linalg.norm(moved[:, 4:] - straight, axis=2).max(axis=1)
            edges = np.mean([np.linalg.norm(moved[:, i] - moved[:, j],
                                             axis=1)
                             for i, j in _P2_EDGES], axis=0)
            ratio = gap / np.maximum(edges, 1e-300)
            k = int(np.argmax(ratio))
            if ratio[k] > worst:
                worst = float(ratio[k])
                where = int(sub[k] - geometry["start"] + 1)
    return worst, where


def deformed_mesh_positions(geometry, match, topology, solution, coordinates):
    """(positions of the mesh file's nodes, report) for one geometry."""
    parsed = geometry["parsed"]
    count = len(parsed["points"])
    owned = (topology["owner"] >= geometry["start"]) \
        & (topology["owner"] < geometry["stop"])
    corner_points = topology["members"][owned]
    corner_points = corner_points[match["point_file_node"][corner_points] >= 0]
    file_nodes = match["point_file_node"][corner_points]
    displacement = np.zeros((count, 3))
    displacement[file_nodes] = solution[corner_points]
    spread = float(np.abs(displacement[file_nodes]
                          - solution[corner_points]).max()) \
        if len(file_nodes) else 0.0
    scale = float(np.abs(solution[corner_points]).max()) \
        if len(file_nodes) else 0.0
    if spread > 1e-9 * max(scale, 1e-300):
        raise ExportError(
            f"Geometry {geometry['number']}: copies of a node have different "
            f"displacements (by {spread:.3g}); the mesh is not conforming.")
    used = np.zeros(count, dtype=bool)
    used[file_nodes] = True
    rest = geometry["rest"]
    deformed = rest + displacement
    corners = geometry["file_corners"]
    is_hex = geometry["is_hex"]
    ratio = np.empty(len(corners))
    for hexes in (False, True):
        local = np.flatnonzero(is_hex == hexes)
        if not len(local):
            continue
        if hexes:
            before = hex_volumes(rest[corners[local]])
            after = hex_volumes(deformed[corners[local]])
        else:
            c = corners[local, :4]
            before = _tet_volumes(*(rest[c[:, i]] for i in range(4)))
            after = _tet_volumes(*(deformed[c[:, i]] for i in range(4)))
        ratio[local] = after / np.where(before == 0, np.nan, before)
    bad = np.flatnonzero(~(ratio > 0))
    if len(bad):
        listed = ", ".join(str(int(i) + 1) for i in bad[:10])
        more = f" and {len(bad) - 10} more" if len(bad) > 10 else ""
        raise ExportError(
            f"Geometry {geometry['number']}: {len(bad)} element(s) are "
            f"inverted or flat in the deformed shape (elements {listed}"
            f"{more} of {os.path.basename(geometry['path'])}); PolyFEM cannot "
            "use such a mesh. Export an earlier step.")
    if coordinates == "simulation":
        positions = deformed
    else:
        positions = np.array(parsed["points"], dtype=np.float64)
        pulled = np.linalg.solve(geometry["A"], (deformed - geometry["b"]).T).T
        positions[used] = pulled[used]
    report = {"nodes": int(used.sum()), "unused": int((~used).sum()),
              "elements": len(corners),
              "ratio_min": float(np.nanmin(ratio)),
              "ratio_min_at": int(np.nanargmin(ratio)) + 1,
              "ratio_max": float(np.nanmax(ratio)),
              "ratio_max_at": int(np.nanargmax(ratio)) + 1}
    return positions, report


def export_deformed_mesh(node):
    """Write one .msh per simulated geometry at the chosen step; returns
    (paths written, report text)."""
    pvd = node.evalParm("PVD_file")
    if not pvd or not os.path.isfile(pvd):
        raise ExportError("Load a PVD file first.")
    entries = read_pvd(pvd)
    if not entries:
        raise ExportError("The PVD file lists no steps.")
    step = export_mesh_step(node, len(entries))
    mesh = volume_mesh(pvd, step)
    topology = volume_topology(mesh)
    match = mesh_match(pvd, mesh, topology)
    solution = _solution_vectors(mesh)
    if solution is None:
        raise ExportError("This result has no displacement (the 'solution' "
                          "field), so there is no deformed shape.")
    chosen = {int(token) for token in
              node.evalParm("export_mesh_geometries").split()
              if token.isdigit()}
    coordinates = _menu_token(node, "export_mesh_coordinates") or "mesh"
    geometries = [g for g in match["geometries"]
                  if not chosen or g["number"] in chosen]
    if not geometries:
        raise ExportError("None of the chosen geometries is in this run.")
    curved = [g for g in geometries if g["curved"]]
    if curved:
        raise ExportError(
            "Export Deformed Mesh writes linear elements and needs straight "
            "(4-node tet / 8-node hex) mesh files; "
            + ", ".join(f"geometry {g['number']} "
                        f"({os.path.basename(g['path'])})" for g in curved)
            + " has curved (higher-order) elements.")
    results = []
    points = np.asarray(mesh["points"], dtype=np.float64)
    for geometry in geometries:  # check everything before writing anything
        positions, report = deformed_mesh_positions(
            geometry, match, topology, solution, coordinates)
        curvature = _p2_curvature(geometry, topology, points,
                                  points + solution)
        results.append((geometry, positions, report, curvature))
    folder = export_folder(node)
    os.makedirs(folder, exist_ok=True)
    width = max(3, len(str(len(entries) - 1)))
    stems = [os.path.splitext(os.path.basename(g["path"]))[0]
             for g, *_ in results]
    lines = [f"Step {step} of {len(entries) - 1} (t = {entries[step][0]:.6g})"
             f" written to {folder}"]
    written = []
    for (geometry, positions, report, curvature), stem in zip(results, stems):
        if stems.count(stem) > 1:
            stem = f"{stem}_geo{geometry['number']}"
        extension = os.path.splitext(geometry["path"])[1] or ".msh"
        # simulation coordinates get their own name: loading one in place of
        # the other would apply the Transform twice (or not at all)
        suffix = "_simulation" if coordinates == "simulation" else ""
        path = os.path.join(
            folder, f"{stem}_step{step:0{width}d}{suffix}{extension}")
        write_moved_nodes(geometry["path"], positions, path)
        written.append(path)
        lines.append(
            f"Geometry {geometry['number']}: {os.path.basename(path)} -- "
            f"{report['nodes']} nodes moved, {report['elements']} elements, "
            f"element volume ratio {report['ratio_min']:.4g} (element "
            f"{report['ratio_min_at']}) to {report['ratio_max']:.4g} "
            f"(element {report['ratio_max_at']}), none inverted.")
        if report["unused"]:
            lines.append(f"  {report['unused']} node(s) that no element uses "
                         "keep their positions.")
        if curvature is not None:
            lines.append(
                f"  Higher-order run: the deformed edges are curved, the "
                f"exported elements are straight (largest gap "
                f"{100 * curvature[0]:.3g} % of an edge, element "
                f"{curvature[1]}).")
    lines.append(
        "Coordinates: those of the mesh file (the geometry's Transform puts "
        "the shape where it was)." if coordinates != "simulation" else
        "Coordinates: simulation (world); load it with no Transform.")
    lines.append("The exported shape is a new rest shape: it carries no "
                 "stress, strain or velocity.")
    return written, "\n".join(lines)


def export_mesh_button(kwargs):
    """Export Deformed Mesh button."""
    node = kwargs["node"]
    try:
        written, report = export_deformed_mesh(node)
    except ExportError as exc:
        report, written = f"Not exported: {exc}", []
    except (OSError, MshParseError) as exc:
        report, written = f"Not exported: {exc}", []
    node.parm("export_mesh_report").set(report)
    if written and hou.isUIAvailable():
        hou.ui.setStatusMessage(
            f"Deformed mesh written: {', '.join(map(os.path.basename, written))}",
            hou.severityType.ImportantMessage)
    elif not written:
        _message(report)
    return written


# ---- units -----------------------------------------------------------------

EXPORT_UNIT_SYSTEMS = (
    ("auto", "Automatic (as the PolyFEM node recorded)"),
    ("none", "Not Set (no unit labels)"),
    ("m_kg_s", "m, kg, s (N, Pa)"),
    ("mm_t_s", "mm, tonne, s (N, MPa)"),
    ("mm_kg_s", "mm, kg, s (mN, kPa)"),
    ("mm_g_s", "mm, g, s (uN, Pa)"),
)
_UNIT_SYSTEM_NAMES = {"m_kg_s": ("m", "kg", "s"), "mm_t_s": ("mm", "t", "s"),
                      "mm_kg_s": ("mm", "kg", "s"), "mm_g_s": ("mm", "g", "s")}
_LENGTH_SI = {"m": 1.0, "dm": 0.1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6,
              "µm": 1e-6, "micron": 1e-6, "nm": 1e-9, "km": 1e3,
              "in": 0.0254, "ft": 0.3048}
_MASS_SI = {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "t": 1e3, "tonne": 1e3,
            "Mg": 1e3, "lb": 0.45359237}
_TIME_SI = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "min": 60.0,
            "h": 3600.0}
EXPORT_LENGTH_UNITS = ("run", "m", "cm", "mm", "um")
EXPORT_STRESS_UNITS = ("run", "Pa", "kPa", "MPa", "GPa")
EXPORT_FORCE_UNITS = ("run", "N", "mN", "uN", "kN")
EXPORT_TIME_UNITS = ("run", "s", "ms")
_STRESS_SI = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}
_FORCE_SI = {"N": 1.0, "mN": 1e-3, "uN": 1e-6, "kN": 1e3}
_UNIT_TEXT = {"um": "µm", "uN": "µN", "µs": "µs"}


def run_units(node):
    """((length, mass, time) names, why) of the run's unit system, or
    (None, why) when it is not known."""
    token = _menu_token(node, "export_units") or "auto"
    if token == "none":
        return None, "Units are off for this export: no unit labels."
    if token in _UNIT_SYSTEM_NAMES:
        return _UNIT_SYSTEM_NAMES[token], "chosen on this node"
    path = node.evalParm("PVD_file")
    params = run_params(path) if path else None
    units = params.get("units") if params else None
    if isinstance(units, dict) and units.get("length"):
        names = (str(units.get("length", "m")), str(units.get("mass", "kg")),
                 str(units.get("time", "s")))
        if names[0] in _LENGTH_SI and names[1] in _MASS_SI \
                and names[2] in _TIME_SI:
            return names, "recorded by the PolyFEM node"
        return None, (f"The PolyFEM node recorded units {', '.join(names)}, "
                      "which this export does not know: choose the run's "
                      "unit system.")
    return None, ("The run did not record its units (the PolyFEM node's "
                  "Units were off): choose the run's unit system to label "
                  "and convert values.")


def export_units_text(node):
    """Read-only line: the run's unit system, as the export uses it."""
    names, why = run_units(node)
    if names is None:
        return why
    return f"Run: {', '.join(names)} ({why})."


class _ExportUnits:
    """Unit labels and conversion factors of one export."""

    def __init__(self, node):
        self.names, self.why = run_units(node)
        self.known = self.names is not None
        if not self.known:
            return
        length, mass, time = self.names
        L, M, T = _LENGTH_SI[length], _MASS_SI[mass], _TIME_SI[time]
        stress, force = M / (L * T * T), M * L / (T * T)
        self.run = {"length": (L, length), "time": (T, time),
                    "stress": (stress, self._named(stress, _STRESS_SI) or
                               f"{mass}/({length}*{time}^2)"),
                    "force": (force, self._named(force, _FORCE_SI) or
                              f"{mass}*{length}/{time}^2")}
        self.shown = {}
        for kind, parm, table in (
                ("length", "export_unit_length", _LENGTH_SI),
                ("stress", "export_unit_stress", _STRESS_SI),
                ("force", "export_unit_force", _FORCE_SI),
                ("time", "export_unit_time", _TIME_SI)):
            token = _menu_token(node, parm) or "run"
            if token in table:
                self.shown[kind] = (table[token], token)
            else:
                self.shown[kind] = self.run[kind]

    @staticmethod
    def _named(value, table):
        for name, factor in table.items():
            if abs(value / factor - 1.0) < 1e-9:
                return name
        return None

    def ratio(self, kind):
        return self.run[kind][0] / self.shown[kind][0]

    def text(self, kind):
        name = self.shown[kind][1]
        return _UNIT_TEXT.get(name, name)

    def convert(self, dimension):
        """(factor, label) turning run values of a dimension into the
        shown unit; label None = unknown unit, "" = dimensionless."""
        if dimension == "none":
            return 1.0, ""
        if dimension is None or not self.known:
            return 1.0, None
        if dimension in ("length", "stress", "force", "time"):
            return self.ratio(dimension), self.text(dimension)
        if dimension == "volume":
            return self.ratio("length") ** 3, f"{self.text('length')}³"
        if dimension == "velocity":
            return (self.ratio("length") / self.ratio("time"),
                    f"{self.text('length')}/{self.text('time')}")
        if dimension == "acceleration":
            return (self.ratio("length") / self.ratio("time") ** 2,
                    f"{self.text('length')}/{self.text('time')}²")
        if dimension in ("stress2", "stress3"):
            power = 2 if dimension == "stress2" else 3
            return (self.ratio("stress") ** power,
                    f"{self.text('stress')}{'²' if power == 2 else '³'}")
        return 1.0, None

    def describe(self):
        if not self.known:
            return f"Not set: {self.why}"
        shown = ", ".join(f"{kind} {self.text(kind)}"
                          for kind in ("length", "stress", "force", "time"))
        return (f"Run: {', '.join(self.names)} ({self.why}); this workbook: "
                f"{shown}.")


# What each quantity is measured in (by its Houdini attribute name). Fields
# not listed have an unknown unit: their headers carry none and they are
# never converted.
_STRESS_FIELDS = {
    "von_mises", "von_mises_derived", "cauchy_mat", "cauchy_eigenvalues",
    "cauchy_trace", "hydrostatic_stress", "deviatoric_stress",
    "max_shear_stress", "pk1", "pk2", "pk2_eigenvalues",
    "cauchy_stress_1", "cauchy_stress_2", "cauchy_stress_3",
    "pk1_stress_1", "pk1_stress_2", "pk1_stress_3", "pk2_stress_1",
    "pk2_stress_2", "pk2_stress_3", "cauchy_stess", "pk1_stess", "pk2_stess",
    "E", "mu", "lambda", "k1", "c1", "c2", "c3", "d1"}
_DIMENSIONLESS_FIELDS = {
    "F_mat", "F_1", "F_2", "F_3", "J", "right_cauchy_green",
    "right_cauchy_green_eigenvalues", "left_cauchy_green",
    "left_cauchy_green_eigenvalues", "right_stretch", "left_stretch",
    "principal_stretches", "green_lagrange_strain",
    "green_lagrange_eigenvalues", "almansi_strain",
    "almansi_strain_eigenvalues", "hencky_strain", "hencky_strain_eigenvalues",
    "infinitesimal_strain", "infinitesimal_strain_eigenvalues",
    "stress_triaxiality", "nu", "k2", "kappa", "discr"}


def quantity_dimension(field):
    """'length', 'stress', ... or 'none' (dimensionless) or None (unknown)."""
    if field == "_volume":
        return "volume"
    if field.startswith("fiber_"):
        return "none"
    name = field[:-4] if field.endswith("_avg") else field
    base = name.rsplit("_", 1)[-1] if "_" in name else name
    if name in ("solution", "solution_mag", "displacement"):
        return "length"
    if name == "velocity":
        return "velocity"
    if name == "acceleration":
        return "acceleration"
    if name.endswith("_forces"):
        return "force"
    if name in _STRESS_FIELDS or base in _STRESS_FIELDS:
        return "stress"
    if name == "stress_J2":
        return "stress2"
    if name == "stress_J3":
        return "stress3"
    if name in _DIMENSIONLESS_FIELDS or base in _DIMENSIONLESS_FIELDS:
        return "none"
    return None


# ---- quantities --------------------------------------------------------------

# Short names for column headers (slide legends).
_HEADER_NAMES = {
    "von_mises": "von Mises stress", "von_mises_derived": "von Mises stress",
    "solution": "displacement", "displacement": "displacement",
    "solution_mag": "displacement magnitude", "velocity": "velocity",
    "acceleration": "acceleration", "J": "volume ratio J",
    "F_mat": "deformation gradient", "cauchy_mat": "Cauchy stress",
    "cauchy_eigenvalues": "principal Cauchy stress",
    "cauchy_trace": "Cauchy stress trace",
    "hydrostatic_stress": "hydrostatic stress",
    "deviatoric_stress": "deviatoric stress",
    "max_shear_stress": "maximum shear stress",
    "stress_triaxiality": "stress triaxiality", "stress_J2": "J2",
    "stress_J3": "J3", "pk1": "1st Piola-Kirchhoff stress",
    "pk2": "2nd Piola-Kirchhoff stress",
    "pk2_eigenvalues": "principal 2nd Piola-Kirchhoff stress",
    "principal_stretches": "principal stretch",
    "right_cauchy_green": "right Cauchy-Green tensor",
    "left_cauchy_green": "left Cauchy-Green tensor",
    "right_cauchy_green_eigenvalues": "principal right Cauchy-Green",
    "left_cauchy_green_eigenvalues": "principal left Cauchy-Green",
    "right_stretch": "right stretch tensor",
    "left_stretch": "left stretch tensor",
    "green_lagrange_strain": "Green-Lagrange strain",
    "green_lagrange_eigenvalues": "principal Green-Lagrange strain",
    "almansi_strain": "Almansi strain",
    "almansi_strain_eigenvalues": "principal Almansi strain",
    "hencky_strain": "logarithmic strain",
    "hencky_strain_eigenvalues": "principal logarithmic strain",
    "infinitesimal_strain": "small strain",
    "infinitesimal_strain_eigenvalues": "principal small strain",
    "_volume": "volume",
}
_VALUE_NAMES = {
    "magnitude": "magnitude", "x": "x", "y": "y", "z": "z", "xy": "xy",
    "yz": "yz", "xz": "xz", "principal_max": "max principal",
    "principal_middle": "mid principal", "principal_min": "min principal",
    "trace": "trace", "determinant": "determinant",
}
_TENSOR_ENTRY_NAMES = {"x": "xx", "y": "yy", "z": "zz", "magnitude": "norm"}
_NODAL_FIELDS = {"solution", "solution_mag", "displacement", "velocity",
                 "acceleration"}
_FIBER_KINDS = (("fiber_stretch", "Fiber Stretch |F a0|"),
                ("fiber_isochoric_stretch", "Fiber Isochoric Stretch"),
                ("fiber_I4", "Fiber I4 = |F a0|^2"))


def _quantity_field(node, index):
    field = node.evalParm(f"export_q_field{index}")
    return field or node.evalParm("color_attrib")


def _quantity_value(node, index):
    if not node.evalParm(f"export_q_field{index}"):
        return node.evalParm("color_reduction")
    return node.evalParm(f"export_q_value{index}")


def _fiber_menu_items(node):
    path = node.evalParm("PVD_file")
    if not path:
        return []
    try:
        mesh = volume_mesh(path, entry_index(node))
    except Exception:
        return []
    names = [_fiber_attrib(prefix, raw=True)
             for prefix in _fiber_prefixes(mesh["point_data"])]
    if not names:
        names = _companion_family_names(path)
    if not names or not all(f"F_{i}" in mesh["point_data"] for i in (1, 2, 3)):
        return []
    items = []
    for name in names:
        for kind, label in _FIBER_KINDS:
            items.extend((f"{kind}:{name}", f"{label} ({_field_label(name)})"))
    return items


def export_field_menu(kwargs):
    """Quantity menu: the Display tab's field, every color field, the
    region's volume, and fiber stretches on fiber runs."""
    node = kwargs["node"]
    items = ["", "Same as the Display Tab"]
    available = _volume_fields(node)
    ordered = [name for name in FIELD_LABELS if name in available]
    ordered.extend(sorted(name for name in available if name not in ordered))
    for name in ordered:
        items.extend((name, _field_label(name)))
    items.extend(("_volume", "Region Volume"))
    items.extend(_fiber_menu_items(node))
    return items


def _volume_fields(node):
    """{field: components} of the Volume block on screen, derived fields
    included (the export computes them whatever the Display tab shows)."""
    return _available_fields_from_fields(
        _frame_metadata(node).get("Volume", {}), True)


def _field_components(node, field):
    if field == "_volume" or field.startswith("fiber_"):
        return 1
    return _volume_fields(node).get(field, 0)


def export_value_menu(kwargs):
    """Value menu of a quantity: what Field Value To Display offers, plus
    the off-diagonal entries of tensors."""
    node = kwargs["node"]
    index = kwargs["parm"].multiParmInstanceIndices()[0]
    if not node.evalParm(f"export_q_field{index}"):
        return ["", "Same as the Display Tab"]
    field = _quantity_field(node, index)
    components = _field_components(node, field)
    if components == 1:
        return ["auto", "Scalar Value"]
    if components in (2, 3) and field in PRINCIPAL_VALUE_FIELDS:
        entries = (("principal_max", "First Principal Value (Largest)"),
                   ("principal_middle", "Second Principal Value"),
                   ("principal_min", "Third Principal Value (Smallest)"),
                   ("magnitude", "Principal-Value Magnitude"))
    elif components in (2, 3):
        entries = (("magnitude", "Vector Magnitude"), ("x", "X Component"),
                   ("y", "Y Component"), ("z", "Z Component"))
    elif components == 9:
        entries = (("principal_max", "First Principal Value (Largest)"),
                   ("principal_middle", "Second Principal Value"),
                   ("principal_min", "Third Principal Value (Smallest)"),
                   ("x", "XX Entry"), ("y", "YY Entry"), ("z", "ZZ Entry"),
                   ("xy", "XY Entry"), ("yz", "YZ Entry"), ("xz", "XZ Entry"),
                   ("magnitude", "Frobenius Norm"), ("trace", "Tensor Trace"),
                   ("determinant", "Tensor Determinant"))
    else:
        return ["auto", "Not Available in This Frame"]
    result = []
    for token, label in entries:
        result.extend((token, label))
    return result


def export_quantity_changed(kwargs):
    """A new Field: keep the Value valid for it."""
    node = kwargs["node"]
    index = kwargs["parm"].multiParmInstanceIndices()[0]
    value = node.parm(f"export_q_value{index}")
    tokens = _menu_tokens(export_value_menu({"node": node, "parm": value}))
    if value.evalAsString() not in tokens:
        value.set(tokens[0] if tokens else "")


def _export_reduce(values, token):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values
    if token in ("xy", "yz", "xz"):
        if values.ndim == 3 or values.shape[1] == 9:
            matrices = values.reshape(-1, 3, 3)
            i, j = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}[token]
            return matrices[:, i, j]
        return np.full(len(values), np.nan)
    if not token or token == "auto":
        token = "magnitude"
    return _reduce_array(values, token)


def _sub_mesh(mesh, rows):
    """The point data of some output points only (derived fields of a small
    region are computed on its own points)."""
    point_data = {}
    for name, values in mesh["point_data"].items():
        values = np.asarray(values)
        if len(values) > (int(rows.max()) if len(rows) else -1):
            point_data[name] = values[rows]
    return {"points": np.asarray(mesh["points"])[rows],
            "point_data": point_data, "cell_data": {}, "cells": {}}


def _fiber_point_values(mesh, field, rows, pvd_path):
    kind, name = field.split(":", 1)
    point_data = mesh["point_data"]
    a0 = None
    for prefix in _fiber_prefixes(point_data):
        if _fiber_attrib(prefix, raw=True) == name:
            a0 = np.stack([np.asarray(point_data[
                f"{prefix}fiber_direction_{axis}"], dtype=np.float64).ravel()
                for axis in "xyz"], axis=1)
    if a0 is None:
        for family, _, values in _companion_point_fibers(pvd_path, mesh):
            if family == name:
                a0 = np.asarray(values, dtype=np.float64)
    if a0 is None or not all(f"F_{i}" in point_data for i in (1, 2, 3)):
        return np.full(len(rows), np.nan)
    a0 = a0[rows]
    F = np.stack([np.asarray(point_data[f"F_{i}"], dtype=np.float64)[rows]
                  for i in (1, 2, 3)], axis=2)
    length = np.linalg.norm(a0, axis=1)
    direction = np.divide(a0, length[:, None], out=np.zeros_like(a0),
                          where=length[:, None] > 1e-12)
    stretch = np.linalg.norm(np.einsum("nij,nj->ni", F, direction), axis=1)
    stretch[length <= 1e-12] = np.nan
    if kind == "fiber_I4":
        return stretch ** 2
    if kind == "fiber_isochoric_stretch":
        J = np.abs(np.linalg.det(F))
        return stretch / np.cbrt(np.maximum(J, 1e-300))
    return stretch


def _cell_field_raw(mesh, field):
    for raw_name, by_family in mesh.get("cell_data", {}).items():
        if _field_name(raw_name) == field:
            return by_family
    return None


def point_or_element_values(mesh, topology, field, token, element_rows,
                            node_rows, pvd_path):
    """(values per element of element_rows, values per node of node_rows)
    of one quantity at one step. An element's value is the mean over its
    nodes in PolyFEM's output (exact for linear elements, whose fields are
    constant); a node's value is the mean over its copies, one per element
    (the nodal average Smooth Field shows). A cell field gives each element
    its own value and each node the mean of its elements'."""
    owner, members = topology["owner"], topology["members"]
    point_node = topology["point_node"]
    by_family = None if any(_field_name(n) == field
                            for n in mesh["point_data"]) \
        else _cell_field_raw(mesh, field)
    element_values = node_values = None
    if by_family is not None:
        cell_values = np.full(len(mesh["cell_types"]), np.nan)
        for family, values in by_family.items():
            sources = mesh["cell_sources"].get(family)
            if sources is None:
                continue
            reduced = _export_reduce(values, token)
            cell_values[sources] = reduced
        per_element = cell_values[topology["cells"]]
        element_values = per_element[element_rows]
        if len(node_rows):
            total = np.bincount(point_node[members], weights=np.nan_to_num(
                per_element[owner]), minlength=topology["n_nodes"])
            count = np.bincount(point_node[members], weights=np.isfinite(
                per_element[owner]).astype(float),
                minlength=topology["n_nodes"])
            node_values = np.divide(total, count, out=np.full_like(total,
                                    np.nan), where=count > 0)[node_rows]
        return element_values, node_values
    needed = np.zeros(len(mesh["points"]), dtype=bool)
    element_mask = np.zeros(topology["n_elements"], dtype=bool)
    element_mask[element_rows] = True
    in_elements = element_mask[owner]
    needed[members[in_elements]] = True
    node_mask = np.zeros(topology["n_nodes"], dtype=bool)
    node_mask[node_rows] = True
    fe = point_node >= 0
    needed[fe] |= node_mask[point_node[fe]]
    rows = np.flatnonzero(needed)
    if field.startswith("fiber_"):
        values = _fiber_point_values(mesh, field, rows, pvd_path)
    else:
        raw = _mesh_field(_sub_mesh(mesh, rows) if len(rows) < len(needed)
                          else mesh, field)
        if raw is None:
            nothing = np.full(len(element_rows), np.nan)
            return nothing, np.full(len(node_rows), np.nan)
        values = _export_reduce(raw, token)
    point_values = np.full(len(mesh["points"]), np.nan)
    point_values[rows] = values
    if len(element_rows):
        weights = point_values[members[in_elements]]
        total = np.bincount(owner[in_elements], weights=weights,
                            minlength=topology["n_elements"])
        count = np.bincount(owner[in_elements],
                            minlength=topology["n_elements"])
        element_values = (total / np.maximum(count, 1))[element_rows]
    if len(node_rows):
        points = np.flatnonzero(fe & needed)
        total = np.bincount(point_node[points], weights=point_values[points],
                            minlength=topology["n_nodes"])
        count = np.bincount(point_node[points],
                            minlength=topology["n_nodes"])
        node_values = (total / np.maximum(count, 1))[node_rows]
    return element_values, node_values


# ---- regions -----------------------------------------------------------------

EXPORT_REGION_KINDS = (
    ("all", "Whole Model"), ("bodies", "Bodies / Subdomains"),
    ("sideset", "Sideset"), ("box", "Box"), ("sphere", "Sphere"),
    ("plane", "Half Space (side of a plane)"), ("element", "One Element"),
    ("node", "One Node"))


def _body_names(node):
    """{body id: label} from the run's meshes: the PolyFEM node numbers the
    subdomains of geometry g by its Gmsh physical groups (sorted tags,
    untagged elements last) and gives them body id 1000 g + subdomain."""
    path = node.evalParm("PVD_file")
    params = run_params(path) if path else None
    labels = {}
    if params is None:
        return labels
    for geometry in simulated_geometries(path, params):
        try:
            parsed = _parsed_mesh(geometry["path"])
        except (ExportError, OSError):
            continue
        tags = []
        for family in ("tet", "hex"):
            if family in parsed["cells"]:
                tags.extend(int(t) for t in np.unique(
                    parsed["cells"][family]["entity"]))
        tags = sorted(set(tags), key=lambda tag: (tag <= 0, tag))
        names = {tag: name for (dim, tag), name in
                 parsed.get("physical_names", {}).items() if dim == 3}
        for position, tag in enumerate(tags, 1):
            body = 1000 * geometry["number"] + position
            name = names.get(tag)
            labels[body] = (f"Geometry {geometry['number']}, subdomain "
                            f"{position}" + (f" '{name}'" if name else
                                             (f" (group {tag})" if tag > 0
                                              else "")))
    return labels


def _body_short_name(names, body):
    """The Gmsh group name of a body when it has one, else 'Body 1002'."""
    label = names.get(body, "")
    if label.count("'") >= 2:
        return label.split("'")[1]
    return f"Body {body}"


def export_body_menu(kwargs):
    """Toggle menu: the bodies of the result, by name where known."""
    node = kwargs["node"]
    path = node.evalParm("PVD_file")
    if not path:
        return []
    try:
        mesh = volume_mesh(path, entry_index(node))
    except Exception:
        return []
    body = mesh["point_data"].get("body_ids")
    if body is None:
        return []
    names = _body_names(node)
    items = []
    for value in sorted({int(v) for v in np.unique(np.rint(np.asarray(
            body, dtype=np.float64)))} - {0}):
        items.extend((str(value), names.get(value, f"Body {value}")))
    return items


def export_sideset_menu(kwargs):
    """The sidesets of the PolyFEM node's record (obstacles excluded)."""
    try:
        _, sets = _force_definitions(kwargs["node"])
    except ForceCurveError:
        return ["", "No Sidesets Recorded"]
    items = []
    for index, entry in enumerate(sets, 1):
        if entry.get("kind") != "obstacle":
            items.extend((str(index), entry.get("label", f"Set {index}")))
    return items or ["", "No Sidesets Recorded"]


def _short_sideset_label(entry, geometries):
    if geometries <= 1:
        return f"Sideset {entry.get('sideset', '?')}"
    return f"Geometry {entry.get('geometry', '?')} sideset " \
           f"{entry.get('sideset', '?')}"


def _region_definition(node, index):
    p = f"export_region_{{}}{index}"
    kind = _menu_token(node, p.format("kind")) or "all"
    return {
        "index": index, "kind": kind,
        "name": node.evalParm(p.format("name")).strip(),
        "bodies": {int(t) for t in node.evalParm(p.format("bodies")).split()
                   if t.lstrip("-").isdigit()},
        "only": {int(t) for t in node.evalParm(p.format("only")).split()
                 if t.lstrip("-").isdigit()},
        "sideset": node.evalParm(p.format("sideset")),
        "box": (np.asarray(node.evalParmTuple(p.format("box_min"))),
                np.asarray(node.evalParmTuple(p.format("box_max")))),
        "center": np.asarray(node.evalParmTuple(p.format("center"))),
        "radius": float(node.evalParm(p.format("radius"))),
        "point": np.asarray(node.evalParmTuple(p.format("point"))),
        "normal": np.asarray(node.evalParmTuple(p.format("normal"))),
        "geometry": int(node.evalParm(p.format("geometry"))),
        "number": int(node.evalParm(p.format("number"))),
    }


def _identity(pvd_path, mesh, topology):
    """Element and node numbers as the user knows them. For a run written
    by the PolyFEM node: (geometry, element number in its .msh file) and
    (geometry, .msh node tag); edge/face nodes of a P2+ run, which the mesh
    file does not have, are numbered after the file's largest tag.
    Otherwise PolyFEM's element order and a node count, geometry 0."""
    try:
        match = mesh_match(pvd_path, mesh, topology)
    except ExportError as exc:
        match, why = None, str(exc)
    n_nodes = topology["n_nodes"]
    if match is None:
        geometry = np.where(topology["body"] >= 1000,
                            topology["body"] // 1000, 0)
        return {"element_geometry": geometry,
                "element_number": np.arange(1, topology["n_elements"] + 1),
                "node_geometry": np.where(topology["node_body"] >= 1000,
                                          topology["node_body"] // 1000, 0),
                "node_number": np.arange(1, n_nodes + 1), "match": None,
                "scheme": ("Elements are numbered in PolyFEM's order and "
                           "nodes by count (the run's mesh files could not "
                           f"be matched: {why})")}
    node_point = topology["node_point"]
    node_geometry = match["point_geometry"][node_point]
    file_node = match["point_file_node"][node_point]
    node_number = np.zeros(n_nodes, dtype=np.int64)
    for geometry in match["geometries"]:
        tags = np.asarray(geometry["parsed"]["node_tags"], dtype=np.int64)
        mine = node_geometry == geometry["number"]
        corner = mine & (file_node >= 0)
        node_number[corner] = tags[file_node[corner]]
        extra = np.flatnonzero(mine & (file_node < 0))
        node_number[extra] = int(tags.max()) + 1 + np.arange(len(extra))
    return {"element_geometry": match["element_geometry"],
            "element_number": match["element_number"],
            "node_geometry": node_geometry, "node_number": node_number,
            "match": match,
            "scheme": ("Elements: their number in the geometry's .msh file "
                       "(1 = the file's first volume element); nodes: their "
                       ".msh node tag (edge and face nodes of higher-order "
                       "elements, which the file does not have, follow the "
                       "largest tag)")}


def _rest_positions(mesh, topology):
    points = np.asarray(mesh["points"], dtype=np.float64)
    corners = topology["corners"]
    count = topology["corner_count"]
    total = np.zeros((topology["n_elements"], 3))
    for corner in range(8):
        mask = corner < count
        total[mask] += points[corners[mask, corner]]
    return total / count[:, None], points[topology["node_point"]]


def _inside(kind, region, positions):
    if kind == "box":
        low, high = np.minimum(*region["box"]), np.maximum(*region["box"])
        return np.all((positions >= low) & (positions <= high), axis=1)
    if kind == "sphere":
        return np.linalg.norm(positions - region["center"], axis=1) \
            <= region["radius"]
    normal = region["normal"]
    if not np.linalg.norm(normal) > 0:
        raise ExportError("A Half Space region needs a non-zero normal.")
    return np.einsum("ij,j->i", positions - region["point"], normal) >= 0


def region_members(node, region, mesh, topology, identity, pvd_path):
    """(element rows, node rows, label, description) of one region."""
    kind = region["kind"]
    n_elements, n_nodes = topology["n_elements"], topology["n_nodes"]
    elements = np.zeros(n_elements, dtype=bool)
    nodes = np.zeros(n_nodes, dtype=bool)
    centroids, node_positions = _rest_positions(mesh, topology)
    point_node = topology["point_node"]
    if kind == "all":
        elements[:] = True
        nodes[:] = True
        label, description = "Model", "every element and node of the model"
    elif kind == "bodies":
        if not region["bodies"]:
            raise ExportError(f"Region {region['index']}: choose at least one "
                              "body.")
        elements = np.isin(topology["body"], list(region["bodies"]))
        nodes = np.isin(topology["node_body"], list(region["bodies"]))
        names = _body_names(node)
        chosen = sorted(region["bodies"])
        label = " + ".join(_body_short_name(names, b) for b in chosen)
        description = "bodies " + ", ".join(
            f"{b} ({names.get(b, 'no name')})" for b in chosen)
    elif kind == "sideset":
        try:
            record, sets = _force_definitions(node)
        except ForceCurveError as exc:
            raise ExportError(f"Region {region['index']}: {exc}")
        try:
            entry = sets[int(region["sideset"]) - 1]
        except (ValueError, IndexError):
            raise ExportError(f"Region {region['index']}: choose a sideset.")
        geometries = len({e.get("geometry") for e in sets
                          if e.get("kind") != "obstacle"})
        label = _short_sideset_label(entry, geometries)
        chosen = _force_nodes(_file_key(scene_record_path(pvd_path)), mesh,
                              sets, [int(region["sideset"])])
        members, looked_for, missing = chosen[int(region["sideset"])]
        if missing:
            raise ExportError(
                f"Region {region['index']}: {missing} of the {looked_for} "
                f"nodes of {entry.get('label')} are not in the result (the "
                "mesh or the record changed since the run).")
        nodes[point_node[members]] = True
        # elements with a whole face on the sideset
        corner_nodes = np.zeros(n_nodes, dtype=bool)
        corner_nodes[point_node[members]] = True
        faces = (np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
                 np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                           [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]))
        for hexes, table in ((False, faces[0]), (True, faces[1])):
            rows = np.flatnonzero(topology["hex"] == hexes)
            if not len(rows):
                continue
            corner_node = point_node[topology["corners"][rows][:, :8 if hexes
                                                               else 4]]
            on = corner_nodes[corner_node]                    # (n, k)
            elements[rows] = np.any(np.all(on[:, table], axis=2), axis=1)
        description = f"the sideset {entry.get('label')} (its nodes; the " \
                      "elements with a face on it)"
        if entry.get("kind") == "nodes":
            elements[:] = np.isin(np.arange(n_elements),
                                  topology["point_element"][members])
            description = f"the point set {entry.get('label')} (its nodes; " \
                          "the elements that contain them)"
    elif kind in ("box", "sphere", "plane"):
        elements = _inside(kind, region, centroids)
        nodes = _inside(kind, region, node_positions)
        label = {"box": "Box", "sphere": "Sphere", "plane": "Half space"}[kind]
        if kind == "box":
            low, high = np.minimum(*region["box"]), np.maximum(*region["box"])
            description = (f"the box {_vector_text(low)} to "
                           f"{_vector_text(high)} (element centroids and "
                           "nodes in the rest shape)")
        elif kind == "sphere":
            description = (f"the sphere of radius {region['radius']:.6g} "
                           f"around {_vector_text(region['center'])} (rest "
                           "shape)")
        else:
            description = (f"the side of the plane through "
                           f"{_vector_text(region['point'])} that "
                           f"{_vector_text(region['normal'])} points to "
                           "(rest shape)")
    elif kind == "element":
        hit = np.flatnonzero(
            (identity["element_geometry"] == region["geometry"])
            & (identity["element_number"] == region["number"]))
        if not len(hit) and identity["match"] is None:
            hit = np.flatnonzero(identity["element_number"]
                                 == region["number"])
        if not len(hit):
            raise ExportError(f"Region {region['index']}: geometry "
                              f"{region['geometry']} has no element "
                              f"{region['number']}.")
        elements[hit[0]] = True
        # a nodal quantity of one element: the mean over its nodes
        nodes[point_node[topology["members"][topology["owner"] == hit[0]]]] \
            = True
        label = f"Element {region['number']}"
        description = (f"element {region['number']} of geometry "
                       f"{region['geometry']}")
    elif kind == "node":
        hit = np.flatnonzero(
            (identity["node_geometry"] == region["geometry"])
            & (identity["node_number"] == region["number"]))
        if not len(hit) and identity["match"] is None:
            hit = np.flatnonzero(identity["node_number"] == region["number"])
        if not len(hit):
            raise ExportError(f"Region {region['index']}: geometry "
                              f"{region['geometry']} has no node "
                              f"{region['number']}.")
        nodes[hit] = True  # one copy per body at an interface
        # an element quantity at one node: the mean over the elements
        # around it (the nodal average Smooth Field shows)
        around = np.isin(point_node, hit)
        elements[np.unique(topology["point_element"][around])] = True
        label = f"Node {region['number']}"
        description = (f"node {region['number']} of geometry "
                       f"{region['geometry']}")
    else:
        raise ExportError(f"Region {region['index']}: unknown kind {kind}.")
    if region["only"] and kind not in ("bodies", "element", "node"):
        elements &= np.isin(topology["body"], list(region["only"]))
        nodes &= np.isin(topology["node_body"], list(region["only"]))
        description += " in bodies " + ", ".join(map(str,
                                                     sorted(region["only"])))
    return (np.flatnonzero(elements), np.flatnonzero(nodes),
            region["name"] or label, description)


def export_region_from_probe(kwargs):
    """Set From Probe: the probed point's element, node or position."""
    node = kwargs["node"]
    index = kwargs["script_multiparm_index"]
    point = int(node.evalParm("probe_point"))
    output = node.node("output")
    if point < 0 or output is None:
        _message("Enable the probe (Probe tab) and click the result first.")
        return
    geo = output.geometry()
    if point >= len(geo.points()):
        _message("The probed point is not in the result on screen.")
        return
    rest_attrib = geo.findPointAttrib("rest")
    probed = geo.point(point)
    position = np.asarray(probed.attribValue(rest_attrib) if rest_attrib
                          else probed.position(), dtype=np.float64)
    body_attrib = geo.findPointAttrib("body_ids")
    body = int(round(probed.attribValue(body_attrib))) if body_attrib else None
    pvd = node.evalParm("PVD_file")
    try:
        mesh = volume_mesh(pvd, entry_index(node))
        topology = volume_topology(mesh)
        identity = _identity(pvd, mesh, topology)
    except Exception as exc:
        _message(f"Could not read the result: {exc}")
        return
    _, node_positions = _rest_positions(mesh, topology)
    candidates = np.arange(topology["n_nodes"])
    if body is not None and topology["has_body_ids"]:
        candidates = candidates[topology["node_body"] == body]
    if not len(candidates):
        _message("The probed point is not on a simulated body.")
        return
    nearest = candidates[_nearest_rows(node_positions[candidates],
                                       position[None, :])[0]]
    p = f"export_region_{{}}{index}"
    kind = _menu_token(node, p.format("kind"))
    rest = node_positions[nearest]
    if kind == "element":
        # the element the probed copy belongs to (each element has its own
        # copy of a node); the node's first element if the point numbers of
        # the result on screen differ (bodies hidden, a clip)
        if point < len(topology["point_node"]) \
                and topology["point_node"][point] == nearest:
            element = topology["point_element"][point]
        else:
            copies = np.flatnonzero(topology["point_node"] == nearest)
            element = topology["point_element"][copies[0]]
        node.setParms({p.format("geometry"): int(
            identity["element_geometry"][element]),
            p.format("number"): int(identity["element_number"][element])})
    elif kind == "node":
        node.setParms({p.format("geometry"): int(
            identity["node_geometry"][nearest]),
            p.format("number"): int(identity["node_number"][nearest])})
    elif kind == "plane":
        node.parmTuple(p.format("point")).set(tuple(float(v) for v in rest))
    else:
        if kind != "sphere":
            node.parm(p.format("kind")).set("sphere")
        node.parmTuple(p.format("center")).set(tuple(float(v) for v in rest))


# ---- Data Over Time ------------------------------------------------------------

_STATISTICS = (("max", "max"), ("min", "min"), ("mean", "mean"),
               ("amean", "arithmetic mean"), ("integral", "integral"),
               ("sum", "sum"), ("rms", "RMS"), ("std", "SD"),
               ("count", "count"))
_STATISTIC_DEFINITIONS = (
    ("max / min", "largest / smallest value over the region's elements (or "
     "nodes)"),
    ("mean", "volume-weighted mean: sum of value x volume / sum of volumes "
     "(the volume of each element, or the volume lumped to each node)"),
    ("arithmetic mean", "plain mean over the elements (or nodes), whatever "
     "their size"),
    ("integral", "sum of value x volume over the region (unit: the value's "
     "unit x volume)"),
    ("sum", "plain sum over the elements (or nodes); over nodes, a nodal "
     "force's sum is the resultant force"),
    ("RMS", "root mean square over the elements (or nodes)"),
    ("SD", "standard deviation over the elements (or nodes), population "
     "formula"),
    ("Pnn", "the nn-th percentile over the elements (or nodes), linear "
     "interpolation"),
    ("count", "number of elements (or nodes) with a value"),
    ("value", "a region of one element or node has a single column"),
)


def _quantity_definition(node, index):
    p = f"export_q_{{}}{index}"
    field = _quantity_field(node, index)
    token = _quantity_value(node, index)
    basis = _menu_token(node, p.format("basis")) or "auto"
    if basis == "auto":
        name = field[:-4] if field.endswith("_avg") else field
        nodal = name in _NODAL_FIELDS or name.endswith("_forces")
        basis = "nodes" if nodal else "elements"
    statistics = [key for key, _ in _STATISTICS
                  if node.evalParm(p.format(key))]
    percentiles = []
    for text in node.evalParm(p.format("percentiles")).replace(
            ",", " ").split():
        try:
            value = float(text)
        except ValueError:
            raise ExportError(f"Quantity {index}: '{text}' is not a "
                              "percentile (0 to 100).")
        if not 0 <= value <= 100:
            raise ExportError(f"Quantity {index}: percentile {text} is not "
                              "between 0 and 100.")
        percentiles.append(value)
    return {"index": index, "field": field, "value": token, "basis": basis,
            "statistics": statistics, "percentiles": percentiles,
            "where": bool(node.evalParm(p.format("where"))),
            "volume": _menu_token(node, p.format("volume")) or "current",
            "label": node.evalParm(p.format("label")).strip()}


def _quantity_header(quantity, components):
    """'von Mises stress', 'displacement magnitude', 'Cauchy stress xy'."""
    if quantity["label"]:
        return quantity["label"]
    field, token = quantity["field"], quantity["value"]
    name = _HEADER_NAMES.get(field[:-4] if field.endswith("_avg") else field)
    if field.startswith("fiber_"):
        kind, family = field.split(":", 1)
        base = {"fiber_stretch": "fiber stretch",
                "fiber_isochoric_stretch": "fiber isochoric stretch",
                "fiber_I4": "fiber I4"}[kind]
        name = f"{base} ({_field_label(family)})"
    if name is None:
        name = _display_name(field)
    if components <= 1 or field == "_volume":
        return name
    if token in ("", "auto"):  # the Display tab's Automatic Value
        token = "magnitude"
    if components == 9 and token in _TENSOR_ENTRY_NAMES:
        return f"{name} {_TENSOR_ENTRY_NAMES[token]}"
    if field in PRINCIPAL_VALUE_FIELDS:
        word = {"principal_max": "1st", "principal_middle": "2nd",
                "principal_min": "3rd"}.get(token)
        return f"{name} ({word})" if word else f"{name} magnitude"
    if token.startswith("principal_"):  # "max principal Cauchy stress"
        return f"{_VALUE_NAMES[token]} {name}"
    suffix = _VALUE_NAMES.get(token or "magnitude", token)
    return f"{name} {suffix}"


def _statistics(values, weights, quantity):
    """{statistic: value} over the region's items; NaN where undefined."""
    finite = np.isfinite(values) & np.isfinite(weights)
    v, w = values[finite], weights[finite]
    out = {}
    empty = not len(v)
    if "max" in quantity["statistics"] or quantity["where"]:
        out["max"] = np.nan if empty else float(v.max())
        out["argmax"] = -1 if empty else int(np.flatnonzero(finite)[
            int(np.argmax(v))])
    if "min" in quantity["statistics"]:
        out["min"] = np.nan if empty else float(v.min())
    total = float(w.sum()) if len(w) else 0.0
    if "mean" in quantity["statistics"]:
        out["mean"] = float((v * w).sum() / total) if total else np.nan
    if "amean" in quantity["statistics"]:
        out["amean"] = np.nan if empty else float(v.mean())
    if "integral" in quantity["statistics"]:
        out["integral"] = np.nan if empty else float((v * w).sum())
    if "sum" in quantity["statistics"]:
        out["sum"] = np.nan if empty else float(v.sum())
    if "rms" in quantity["statistics"]:
        out["rms"] = np.nan if empty else float(np.sqrt(np.mean(v * v)))
    if "std" in quantity["statistics"]:
        out["std"] = np.nan if empty else float(v.std())
    if "count" in quantity["statistics"]:
        out["count"] = int(len(v))
    for p in quantity["percentiles"]:
        out[f"P{p:g}"] = np.nan if empty else float(np.percentile(v, p))
    return out


def _export_steps(node, count):
    first = max(0, int(node.evalParm("export_step_first")))
    last = int(node.evalParm("export_step_last"))
    last = count - 1 if last < 0 else min(last, count - 1)
    every = max(1, int(node.evalParm("export_step_every")))
    steps = list(range(first, last + 1, every))
    if not steps:
        raise ExportError(f"No steps between {first} and {last}: the run has "
                          f"steps 0 to {count - 1}.")
    return steps


class _NoProgress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def updateLongProgress(self, *args):
        pass


def _progress(label):
    if hou.isUIAvailable():
        return hou.InterruptableOperation(
            label, long_operation_name=label, open_interrupt_dialog=True)
    return _NoProgress()


def _manifest(pvd_path):
    path = os.path.join(os.path.dirname(os.path.abspath(pvd_path)),
                        "run-manifest.json")
    try:
        with open(path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


class _ForceColumns:
    """The chosen force curves (Analysis > Force Curves) as columns, read
    in the export's own pass over the steps."""

    def __init__(self, node, mesh, pvd_path):
        self.note = None
        self.sets = []
        self.values = []  # per step: {set: (reaction, contact, displacement)}
        if not node.evalParm("export_forces"):
            return
        try:
            _, definitions = _force_definitions(node)
        except ForceCurveError as exc:
            self.note = f"Force curves not included: {exc}"
            return
        chosen = {int(t) for t in node.evalParm("export_force_sets").split()
                  if t.isdigit()}
        self.sets = [i for i in range(1, len(definitions) + 1)
                     if not chosen or i in chosen]
        self.definitions = definitions
        self.nodes = _force_nodes(_file_key(scene_record_path(pvd_path)),
                                  mesh, definitions, self.sets)
        self.quantities = [(name, label, dimension) for name, label, dimension
                           in (("reaction", "reaction", "force"),
                               ("contact", "contact force", "force"),
                               ("displacement", "mean displacement",
                                "length"))
                           if node.evalParm(f"export_force_{name}")]
        self.components = [axis for axis in ("x", "y", "z", "magnitude")
                           if node.evalParm(f"export_force_{axis}")]
        if not (self.sets and self.quantities and self.components):
            self.note = ("Force curves: nothing chosen (sets, forces or "
                         "components).")
            self.sets = []

    def add(self, mesh):
        """One step (mesh None: the step could not be read)."""
        if not self.sets:
            return
        if mesh is None:
            self.values.append(None)
            return
        try:
            self.values.append(_frame_set_values(mesh, self.nodes))
        except ForceCurveError as exc:
            self.note = f"Force curves not included: {exc}"
            self.sets = []

    def columns(self, units):
        if not self.sets:
            return []
        geometries = len({e.get("geometry") for e in self.definitions
                          if e.get("kind") != "obstacle"})
        out = []
        nothing = (np.full(3, np.nan),) * 3
        for index in self.sets:
            entry = self.definitions[index - 1]
            name = (f"Obstacle {entry.get('geometry')}"
                    if entry.get("kind") == "obstacle"
                    else _short_sideset_label(entry, geometries))
            for position, (_, label, dimension) in enumerate(
                    self.quantities):
                slot = ("reaction", "contact", "displacement").index(
                    self.quantities[position][0])
                factor, unit = units.convert(dimension)
                vectors = np.array([(step or {}).get(index, nothing)[slot]
                                    for step in self.values]) * factor
                for axis in self.components:
                    values = (np.linalg.norm(vectors, axis=1)
                              if axis == "magnitude"
                              else vectors[:, "xyz".index(axis)])
                    out.append((f"{name}: {label} {axis}"
                                + (f" ({unit})" if unit else ""), values))
        return out


def export_data(node, progress=None):
    """Compute and write the Data Over Time workbook; returns a summary."""
    import time as _time
    started = _time.time()
    pvd = node.evalParm("PVD_file")
    if not pvd or not os.path.isfile(pvd):
        raise ExportError("Load a PVD file first.")
    entries = read_pvd(pvd)
    if not entries:
        raise ExportError("The PVD file lists no steps.")
    steps = _export_steps(node, len(entries))
    units = _ExportUnits(node)
    region_count = int(node.evalParm("export_regions"))
    quantity_count = int(node.evalParm("export_quantities"))
    if quantity_count < 1:
        raise ExportError("Add at least one quantity.")
    quantities = [_quantity_definition(node, i)
                  for i in range(1, quantity_count + 1)]
    mesh = volume_mesh(pvd, steps[0])
    topology = volume_topology(mesh)
    identity = _identity(pvd, mesh, topology)
    regions = []
    definitions = [_region_definition(node, i)
                   for i in range(1, region_count + 1)] or [
        {"index": 1, "kind": "all", "name": "", "only": set()}]
    for definition in definitions:
        elements, nodes, label, description = region_members(
            node, definition, mesh, topology, identity, pvd)
        # One Element / One Node give one value: a node on an interface
        # between two bodies has a copy in each (averaged)
        regions.append({"elements": elements, "nodes": nodes,
                        "label": label, "description": description,
                        "kind": definition["kind"],
                        "single": definition["kind"] in ("element", "node")})
    seen = {}
    for region in regions:  # unique labels: they head the columns
        seen[region["label"]] = seen.get(region["label"], 0) + 1
        if seen[region["label"]] > 1:
            region["label"] = f"{region['label']} ({seen[region['label']]})"
    components = {q["index"]: _field_components(node, q["field"])
                  for q in quantities}
    for q in quantities:
        if not components[q["index"]]:
            raise ExportError(
                f"Quantity {q['index']}: '{q['field']}' is not in this "
                "result (or not on the Volume block).")
        if not q["statistics"] and not q["percentiles"] \
                and q["field"] != "_volume":
            raise ExportError(f"Quantity {q['index']}: choose at least one "
                              "statistic.")
    rest_points = np.asarray(mesh["points"], dtype=np.float64)
    rest_volume = element_volumes(topology, rest_points)
    sign = np.where(rest_volume < 0, -1.0, 1.0)
    rest_volume = rest_volume * sign
    owner, members = topology["owner"], topology["members"]
    point_node = topology["point_node"]

    def node_weights(volume):
        share = (volume / topology["sizes"])[owner]
        return np.bincount(point_node[members], weights=share,
                           minlength=topology["n_nodes"])

    rest_node_volume = node_weights(rest_volume)
    per_item = bool(node.evalParm("export_per_item"))
    item_mode = _menu_token(node, "export_per_item_steps") or "last"
    if item_mode == "all":
        item_steps = list(steps)
    elif item_mode == "screen":
        item_steps = [max(0, min(entry_index(node), len(entries) - 1))]
    else:
        item_steps = [steps[-1]]
    need_current = any(q["volume"] == "current" for q in quantities)
    items = {}       # quantity index -> {step: [values per region]}
    records = []     # per step: {(region, quantity, key): value}
    unreadable = []
    forces = _ForceColumns(node, mesh, pvd)
    mapping = time_mapping(node)
    rows_time = []
    total = len(steps)
    with (progress or _progress("Exporting data over time")) as operation:
        for position, step in enumerate(steps):
            operation.updateLongProgress(
                position / max(1, total),
                f"Step {step} ({position + 1} of {total})")
            time = entries[step][0]
            rows_time.append((step, time, entry_frame(node, step)))
            record = {}
            records.append(record)
            try:
                frame_mesh = mesh if step == steps[0] \
                    else volume_mesh(pvd, step)
            except Exception as exc:  # noqa: BLE001 -- reported below
                # e.g. the last step of a run that is still writing
                unreadable.append((step, str(exc)))
                forces.add(None)
                continue
            if (str(frame_mesh.get("topo_key")), len(frame_mesh["points"])) \
                    != topology["key"]:
                raise ExportError(
                    f"The mesh changes at step {step} (a remeshing run): the "
                    "export follows one mesh through the run.")
            forces.add(frame_mesh)
            current_volume = current_node_volume = None
            if need_current:
                solution = _solution_vectors(frame_mesh)
                moved = rest_points if solution is None \
                    else rest_points + solution
                current_volume = element_volumes(topology, moved) * sign
                current_node_volume = node_weights(current_volume)
            for q in quantities:
                volume = current_volume if q["volume"] == "current" \
                    else rest_volume
                node_volume = current_node_volume \
                    if q["volume"] == "current" else rest_node_volume
                if q["field"] == "_volume":
                    for r, region in enumerate(regions):
                        record[(r, q["index"], "volume")] = float(
                            volume[region["elements"]].sum())
                    continue
                element_rows = np.unique(np.concatenate(
                    [region["elements"] for region in regions])) \
                    if q["basis"] == "elements" else np.zeros(0, np.int64)
                node_rows = np.unique(np.concatenate(
                    [region["nodes"] for region in regions])) \
                    if q["basis"] == "nodes" else np.zeros(0, np.int64)
                element_values, node_values = point_or_element_values(
                    frame_mesh, topology, q["field"], q["value"],
                    element_rows, node_rows, pvd)
                if q["basis"] == "elements":
                    lookup = np.full(topology["n_elements"], np.nan)
                    lookup[element_rows] = element_values
                    weight_of = volume
                else:
                    lookup = np.full(topology["n_nodes"], np.nan)
                    lookup[node_rows] = node_values
                    weight_of = node_volume
                for r, region in enumerate(regions):
                    rows = region[q["basis"]]
                    values, weights = lookup[rows], weight_of[rows]
                    if region["single"]:
                        finite = values[np.isfinite(values)]
                        record[(r, q["index"], "value")] = \
                            float(finite.mean()) if len(finite) else np.nan
                    else:
                        for key, value in _statistics(values, weights,
                                                      q).items():
                            record[(r, q["index"], key)] = value
                    if per_item and step in item_steps:
                        items.setdefault(q["index"], {}).setdefault(
                            step, []).append(values)
    if len(unreadable) == len(steps):
        raise ExportError("None of the chosen steps could be read.")
    keys = []
    for record in records:
        keys.extend(key for key in record if key not in keys)
    columns = {key: [record.get(key, np.nan) for record in records]
               for key in keys}
    # ---- columns --------------------------------------------------------
    header_units = {}
    for q in quantities:
        dimension = quantity_dimension(q["field"])
        header_units[q["index"]] = units.convert(dimension)
    tfactor, tunit = units.convert("time")
    data_header = ["Step", "Time" + (f" ({tunit})" if tunit else ""),
                   "Frame"]
    data_columns = []
    for r, region in enumerate(regions):
        for q in quantities:
            name = _quantity_header(q, components[q["index"]])
            factor, unit = header_units[q["index"]]
            if q["field"] == "_volume":
                vfactor, vunit = units.convert("volume")
                header = f"{region['label']}: {name}" + (
                    " (rest)" if q["volume"] == "rest" else "") + (
                    f" ({vunit})" if vunit else "")
                data_columns.append((header, np.asarray(columns[(
                    r, q["index"], "volume")]) * vfactor))
                continue
            single = (r, q["index"], "value") in columns
            keys = ["value"] if single else (
                [key for key, _ in _STATISTICS if key in q["statistics"]]
                + [f"P{p:g}" for p in q["percentiles"]])
            for key in keys:
                values = np.asarray(columns[(r, q["index"], key)],
                                    dtype=np.float64)
                label = dict(_STATISTICS).get(key, key)
                if key == "count":
                    unit_text = ""
                    values_out = values
                elif key == "integral":
                    vfactor, vunit = units.convert("volume")
                    values_out = values * factor * vfactor
                    unit_text = "" if unit is None else (
                        f"{unit}*{vunit}" if unit else vunit)
                else:
                    values_out = values * factor
                    unit_text = unit or ""
                header = f"{region['label']}: {name}" + (
                    "" if single else f", {label}") + (
                    f" ({unit_text})" if unit_text else "")
                data_columns.append((header, values_out))
            if q["where"] and not single:
                where = np.asarray(columns[(r, q["index"], "argmax")],
                                   dtype=np.float64)
                where = np.where(np.isfinite(where), where, -1).astype(
                    np.int64)
                rows = region[q["basis"]]
                lfactor, lunit = units.convert("length")
                centroids, node_positions = _rest_positions(mesh, topology)
                positions = centroids if q["basis"] == "elements" \
                    else node_positions
                numbers = identity["element_number"] \
                    if q["basis"] == "elements" else identity["node_number"]
                picked = np.where(where >= 0, rows[np.maximum(where, 0)], -1)
                word = "element" if q["basis"] == "elements" else "node"
                data_columns.append((
                    f"{region['label']}: {name}, max at {word}",
                    np.where(picked >= 0, numbers[np.maximum(picked, 0)],
                             np.nan)))
                for axis, label in enumerate("xyz"):
                    data_columns.append((
                        f"{region['label']}: {name}, max at {label}" + (
                            f" ({lunit})" if lunit else ""),
                        np.where(picked >= 0, positions[np.maximum(
                            picked, 0), axis] * lfactor, np.nan)))
    data_columns.extend(forces.columns(units))
    force_note = forces.note
    # ---- workbook -----------------------------------------------------------
    path = node.evalParm("export_data_file").strip()
    if not path:
        stem = os.path.splitext(os.path.basename(pvd))[0]
        path = os.path.join(export_folder(node), f"{stem}_data.xlsx")
    path = os.path.abspath(os.path.expanduser(path))
    if not path.lower().endswith(".xlsx"):
        path += ".xlsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    book = XlsxWorkbook()
    headers = data_header + [header for header, _ in data_columns]

    def data_rows():
        yield headers
        for position, (step, time, frame) in enumerate(rows_time):
            yield [step, time * tfactor, frame] + [
                values[position] for _, values in data_columns]

    widths = [6, 12, 8] + [min(60, max(12, len(h) * 0.9))
                           for h, _ in data_columns]
    book.add_sheet("Data", data_rows(), widths=widths)
    if per_item:
        lfactor, lunit = units.convert("length")
        centroids, node_positions = _rest_positions(mesh, topology)
        for q in quantities:
            if q["field"] == "_volume":
                continue
            word = "Element" if q["basis"] == "elements" else "Node"
            factor, unit = header_units[q["index"]]
            positions = centroids if word == "Element" else node_positions
            numbers = identity["element_number"] if word == "Element" \
                else identity["node_number"]
            geometry = identity["element_geometry"] if word == "Element" \
                else identity["node_geometry"]
            bodies = topology["body"] if word == "Element" \
                else topology["node_body"]
            count = sum(len(region[q["basis"]]) for region in regions)
            if count + 1 > XLSX_MAX_ROWS:
                raise ExportError(
                    f"Per {word.lower()} values: {count:,} rows do not fit "
                    f"in an Excel sheet ({XLSX_MAX_ROWS:,}); choose a smaller "
                    "region or turn this off.")
            name = _quantity_header(q, components[q["index"]])
            header = ["Region", "Geometry", word, "Body"] + [
                f"{axis} ({lunit})" if lunit else axis for axis in "xyz"] + [
                f"Step {step}: t = {entries[step][0] * tfactor:.6g}" + (
                    f" {tunit}" if tunit else "") for step in item_steps]
            by_step = items.get(q["index"], {})

            def item_rows(q=q, header=header, by_step=by_step,
                          positions=positions, numbers=numbers,
                          geometry=geometry, bodies=bodies, factor=factor):
                yield header
                for r, region in enumerate(regions):
                    rows = region[q["basis"]]
                    stacked = [by_step[step][r] * factor
                               for step in item_steps if step in by_step]
                    for k, row in enumerate(rows):
                        yield ([region["label"], int(geometry[row]),
                                int(numbers[row]), int(bodies[row])]
                               + list(positions[row] * lfactor)
                               + [values[k] for values in stacked])

            book.add_sheet(f"{name} per {word.lower()}", item_rows(),
                           widths=[14, 9, 10, 8, 12, 12, 12]
                           + [16] * len(item_steps))
    about = _about_rows(node, pvd, entries, steps, regions, quantities,
                        components, units, identity, force_note, mapping)
    book.add_sheet("About", about, widths=[30, 110], header_rows=0)
    try:
        written = book.save(path, title=f"PolyFEM results: {pvd}",
                            creator="Read PVD 1.0")
    except XlsxLimitError as exc:
        raise ExportError(f"The table does not fit in Excel: {exc}.")
    except PermissionError:
        raise ExportError(f"Could not write {path}: is it open in Excel?")
    seconds = _time.time() - started
    lines = [f"Wrote {path}",
             f"Data: {len(steps)} steps x {len(data_columns)} series "
             f"({seconds:.1f} s)."]
    for region in regions:
        lines.append(f"{region['label']}: {len(region['elements'])} "
                     f"elements, {len(region['nodes'])} nodes.")
    if force_note:
        lines.append(force_note)
    if unreadable:
        lines.append(
            f"Step(s) {', '.join(str(step) for step, _ in unreadable)} could "
            f"not be read (empty cells; a run still writing?): "
            f"{unreadable[0][1]}")
    if not units.known:
        lines.append(f"No unit labels: {units.why}")
    return {"path": path, "steps": steps, "headers": headers,
            "columns": data_columns, "rows": rows_time, "regions": regions,
            "sheets": written, "status": "\n".join(lines)}


def _about_rows(node, pvd, entries, steps, regions, quantities, components,
                units, identity, force_note, mapping):
    import datetime
    manifest = _manifest(pvd) or {}
    build = (manifest.get("build") or {}).get("sources") or {}
    commit = (build.get("polyfem") or {}).get("commit", "")
    input_file = ((manifest.get("input") or {}).get("file") or {})
    completion = manifest.get("completion") or {}
    rows = [["Read PVD export: data over time"], [],
            ["PVD file", pvd],
            ["Run folder", run_folder(pvd)],
            ["Run id", manifest.get("run_id", "(no run-manifest.json)")],
            ["Run status", completion.get("status", "")],
            ["PolyFEM commit", commit],
            ["Input params.json sha256", input_file.get("sha256", "")],
            ["Houdini scene", hou.hipFile.path()],
            ["Exported", datetime.datetime.now().astimezone().isoformat(
                timespec="seconds")],
            ["Exported by", f"Read PVD 1.0, Houdini "
                            f"{hou.applicationVersionString()}"],
            [],
            ["Steps", f"{len(steps)} of the run's {len(entries)} output "
                      f"steps: {steps[0]} to {steps[-1]}"
                      + (f", every {steps[1] - steps[0]}"
                         if len(steps) > 1 else "")],
            ["Time", "simulation time of each output step (the PVD time)"],
            ["Frame", "the Houdini frame that shows the step ("
             + ("Time Mapping: simulation time" if mapping is not None
                else "one frame per output step") + ")"],
            ["Units", units.describe()],
            [],
            ["Regions", "fixed on the rest shape: the same elements and "
                        "nodes at every step"]]
    for region in regions:
        rows.append([region["label"],
                     f"{region['description']}: {len(region['elements'])} "
                     f"elements, {len(region['nodes'])} nodes"])
    rows.append([])
    rows.append(["Quantities", ""])
    for q in quantities:
        name = _quantity_header(q, components[q["index"]])
        source = "the region's volume" if q["field"] == "_volume" else (
            f"field {_display_name(q['field'])}"
            + (f", value {q['value']}" if components[q["index"]] > 1 else "")
            + f", per {q['basis'][:-1]}")
        rows.append([name, source + f"; volumes: {q['volume']} (" + (
            "deformed, at each step" if q["volume"] == "current"
            else "undeformed") + ")"])
    rows.append([])
    rows.append(["Element value", "the mean of the element's nodal values "
                 "in PolyFEM's output (exact for linear elements, whose "
                 "fields are constant)"])
    rows.append(["Node value", "the mean of the node's copies, one per "
                 "element (the nodal average Smooth Field shows)"])
    rows.append(["Numbering", identity["scheme"]])
    for name, text in _STATISTIC_DEFINITIONS:
        rows.append([name, text])
    rows.append(["Empty cell", "no value at that step (e.g. the field is "
                 "missing in that step's file)"])
    if force_note:
        rows.append(["Force curves", force_note])
    elif node.evalParm("export_forces"):
        rows.append(["Force curves", "reaction: the force the prescribed "
                     "displacement (or an obstacle's motion) applies to the "
                     "body there; contact force: contact + friction; mean "
                     "displacement of the set's nodes (Analysis > Force "
                     "Curves)"])
    return rows


def export_data_button(kwargs):
    """Export Spreadsheet button."""
    node = kwargs["node"]
    try:
        summary = export_data(node)
        status = summary["status"]
    except ExportError as exc:
        summary, status = None, f"Not exported: {exc}"
    except hou.OperationInterrupted:
        summary, status = None, "Cancelled."
    except OSError as exc:
        summary, status = None, f"Not exported: {exc}"
    node.parm("export_data_status").set(status)
    if summary is None and not status.startswith("Cancelled"):
        _message(status)
    elif summary is not None and hou.isUIAvailable():
        hou.ui.setStatusMessage(f"Data written to {summary['path']}",
                                hou.severityType.ImportantMessage)
    return summary


def open_export_data(kwargs):
    """Open Spreadsheet button (exports first when there is no file)."""
    node = kwargs["node"]
    path = node.evalParm("export_data_file").strip()
    if not path and node.evalParm("PVD_file"):
        stem = os.path.splitext(os.path.basename(node.evalParm("PVD_file")))[0]
        path = os.path.join(export_folder(node), f"{stem}_data.xlsx")
    if not path or not os.path.isfile(path):
        summary = export_data_button(kwargs)
        if summary is None:
            return
        path = summary["path"]
    _open_with_system(path)
