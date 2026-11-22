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
    source_label = "recorded by the PolyFEM node"
    if params is None and path:
        manifest = run_manifest(path)
        m_units = ((manifest or {}).get("input") or {}).get("units")
        if isinstance(m_units, dict) and m_units.get("length"):
            units = m_units
            source_label = "recorded in run-manifest.json"
    if isinstance(units, dict) and units.get("length"):
        names = (str(units.get("length", "m")), str(units.get("mass", "kg")),
                 str(units.get("time", "s")))
        if names[0] in _LENGTH_SI and names[1] in _MASS_SI \
                and names[2] in _TIME_SI:
            return names, source_label
        return None, (f"The run recorded units {', '.join(names)}, "
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


def _is_fiber_quantity(field):
    return ":" in field and field.startswith("fiber_")


def quantity_dimension(field):
    """'length', 'stress', ... or 'none' (dimensionless) or None (unknown)."""
    if field == "_volume":
        return "volume"
    if _is_fiber_quantity(field):
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
    if field == "_volume" or _is_fiber_quantity(field):
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
    if _is_fiber_quantity(field):
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
        finite = np.isfinite(weights)
        owners = owner[in_elements][finite]
        total = np.bincount(owners, weights=weights[finite],
                            minlength=topology["n_elements"])
        count = np.bincount(owners, minlength=topology["n_elements"])
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        element_values = mean[element_rows]
    if len(node_rows):
        points = np.flatnonzero(fe & needed)
        weights = point_values[points]
        finite = np.isfinite(weights)
        nodes = point_node[points][finite]
        total = np.bincount(nodes, weights=weights[finite],
                            minlength=topology["n_nodes"])
        count = np.bincount(nodes, minlength=topology["n_nodes"])
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        node_values = mean[node_rows]
    return element_values, node_values


