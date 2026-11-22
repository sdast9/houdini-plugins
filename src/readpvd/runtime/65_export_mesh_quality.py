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
_HEX_TYPES = (12, 25, 29, 72)
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

def _rest_geometry_digest(mesh):
    points = np.ascontiguousarray(mesh["points"], dtype=np.float64)
    digest = hashlib.blake2b(len(points).to_bytes(8, "little"), digest_size=12)
    digest.update(points)
    body = mesh["point_data"].get("body_ids")
    if body is not None:
        digest.update(np.ascontiguousarray(body, dtype=np.float64))
    return digest.digest()


def volume_topology(mesh):
    """Elements (the volume cells, in PolyFEM's element order) and nodes
    (each element's copies of a vertex, merged per body) of a Volume block;
    cached per topology (including rest positions and body_ids)."""
    digest = _rest_geometry_digest(mesh)
    key = (str(mesh.get("topo_key")), len(mesh["points"]))
    cache_key = (key, digest)
    cached = _TOPOLOGY_CACHE.get(cache_key)
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
        ids = np.asarray(body_ids).reshape(len(points), -1)[:, 0]
        point_body = np.where(np.isfinite(ids), np.rint(ids), -1).astype(np.int64)
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
        "types": types[cells],
        "sizes": sizes, "corners": corners, "corner_count": corner_count,
        "owner": owner, "members": members,
        "body": point_body[corners[:, 0]], "point_body": point_body,
        "has_body_ids": body_ids is not None,
        "point_node": point_node, "node_point": node_point,
        "node_body": point_body[node_point], "point_element": point_element,
        "n_elements": len(cells), "n_nodes": len(node_point),
        "connectivity": connectivity,
    }
    if len(_TOPOLOGY_CACHE) > 4:
        _TOPOLOGY_CACHE.clear()
    _TOPOLOGY_CACHE[cache_key] = topology
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


def _hex_jacobian_determinants(nodes, u, v, w):
    return np.linalg.det(np.einsum("nki,kj->nij", nodes - nodes[:, :1],
                                   _hex_gradients(u, v, w)))


def _hex_gradients(u, v, w):
    sign = 2 * _HEX_REFERENCE - 1
    a, b, c = (0.5 + sign * (np.array([u, v, w]) - 0.5)).T
    return sign * np.column_stack((b * c, a * c, a * b))


def hex_volumes(nodes):
    """Exact signed volume of trilinear hexes in VTK / Gmsh order."""
    g = 0.5 / math.sqrt(3.0)
    volume = np.zeros(len(nodes))
    for u, v, w in itertools.product((0.5 - g, 0.5 + g), repeat=3):
        volume += _hex_jacobian_determinants(nodes, u, v, w) / 8.0
    return volume


def _inverted_hex_corners(nodes, orientation):
    invalid = np.zeros(len(nodes), dtype=bool)
    for u, v, w in _HEX_REFERENCE:
        invalid |= ~(_hex_jacobian_determinants(nodes, u, v, w) * orientation > 0)
    return invalid


_HEX_BERNSTEIN = np.array([[1, 0, 0], [-0.5, 2, -0.5], [0, 0, 1]])
_HEX_BERNSTEIN_TO_SAMPLES = np.array([[1, 0, 0], [0.25, 0.5, 0.25], [0, 0, 1]])
_HEX_BERNSTEIN_HALVES = (
    np.array([[1, 0, 0], [0.5, 0.5, 0], [0.25, 0.5, 0.25]]),
    np.array([[0.25, 0.5, 0.25], [0, 0.5, 0.5], [0, 0, 1]]))
_HEX_SAMPLE_GRADIENTS = np.array([
    _hex_gradients(u, v, w)
    for u, v, w in itertools.product((0.0, 0.5, 1.0), repeat=3)])


def _hex_tensor_transform(grid, a, b, c):
    return np.einsum("ai,bj,ck,nijk->nabc", a, b, c, grid, optimize=True)


def _hex_inversion_certificate(nodes, orientation):
    """Masks for proven inversion/degeneracy and unresolved Q1 hexes."""
    inverted = np.zeros(len(nodes), dtype=bool)
    unresolved = np.zeros_like(inverted)
    axes = (1, 2, 3)
    for start in range(0, len(nodes), 4096):
        span = slice(start, start + 4096)
        local = nodes[span] - nodes[span, :1]
        jacobian = np.einsum("nki,skj->nsij", local, _HEX_SAMPLE_GRADIENTS)
        samples = (np.linalg.det(jacobian) * orientation[span, None]).reshape(-1, 3, 3, 3)
        bad = np.any(samples <= 0, axis=axes)
        coeff = _hex_tensor_transform(samples, *([_HEX_BERNSTEIN] * 3))
        tol = 64 * np.finfo(np.float64).eps * np.max(np.abs(coeff), axis=axes)
        positive = np.all(coeff > tol[:, None, None, None], axis=axes)
        candidates = np.flatnonzero(~(positive | bad))
        if len(candidates):
            undecided = np.zeros(len(candidates), dtype=bool)
            bounds = tol[candidates, None, None, None]
            parent_coeff = coeff[candidates]
            for halves in itertools.product(_HEX_BERNSTEIN_HALVES, repeat=3):
                child = _hex_tensor_transform(parent_coeff, *halves)
                uncertain = ~np.all(child > bounds, axis=axes)
                if uncertain.any():
                    inner = _hex_tensor_transform(
                        child[uncertain], *([_HEX_BERNSTEIN_TO_SAMPLES] * 3))
                    bad[candidates[uncertain]] |= np.any(inner <= 0, axis=axes)
                undecided |= uncertain
            unresolved[start + candidates] = undecided & ~bad[candidates]
        inverted[span] = bad
    return inverted, unresolved


# VTK_QUADRATIC_HEXAHEDRON puts its 12 edge midpoints after the corners.
_HEX_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                       (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])


def _serendipity_hex_gradients(q):
    # Reference coordinates in [-1, 1]. Corner N = product(1+a_i*q_i)
    # times (a.q-2)/8. An edge along i uses (1-q_i^2) times the two
    # transverse linear factors, divided by four.
    sign = 2 * _HEX_REFERENCE - 1
    edge = sign[_HEX_EDGES].mean(axis=1)
    factors = 1 + sign * q
    a, b, c = factors.T
    corner = sign * (
        np.column_stack((b * c, a * c, a * b)) *
        (np.sum(sign * q, axis=1) - 2)[:, None] + (a * b * c)[:, None]) / 8
    linear = 1 + edge * q
    along = edge == 0
    factors = np.where(along, 1 - q * q, linear)
    derivatives = np.where(along, -2 * q, edge)
    a, b, c = factors.T
    mid = derivatives * np.column_stack((b * c, a * c, a * b)) / 4
    return np.vstack((corner, mid))


def _serendipity_hex_volumes(nodes):
    local = nodes - nodes[:, :1]
    g = math.sqrt(3 / 5)
    rule = ((-g, 5 / 9), (0.0, 8 / 9), (g, 5 / 9))
    volume = np.zeros(len(nodes))
    for (u, a), (v, b), (w, c) in itertools.product(rule, repeat=3):
        gradients = _serendipity_hex_gradients(np.array([u, v, w]))
        volume += a * b * c * np.linalg.det(
            np.einsum("nki,kj->nij", local, gradients))
    return volume


def element_volumes(topology, positions):
    """Signed volume of every element with its output points at
    `positions`: exact for straight and P2 tets, trilinear hexes and
    20-node serendipity hexes;
    P3/P4 tets and Q2/Q3 Lagrange or triquadratic hexes as the sum of
    Read PVD's linear display sub-elements."""
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
    serendipity = hexes & (sizes == 20) & (topology["types"] == 25)
    selected = np.flatnonzero(serendipity)
    for start in range(0, len(selected), 4096):
        part = selected[start:start + 4096]
        nodes = connectivity[starts[part, None] + np.arange(20)]
        volume[part] = _serendipity_hex_volumes(positions[nodes])
    linear_hexes = hexes & (sizes != 27) & (sizes != 64) & ~serendipity
    if linear_hexes.any():
        volume[linear_hexes] = hex_volumes(positions[corners[linear_hexes]])
    triquadratic = topology["types"] == 29
    for mask, table in ((hexes & (sizes == 27) & triquadratic, HEX27_TRIQUAD_SUBDIV),
                        (hexes & (sizes == 27) & ~triquadratic, HEX27_SUBDIV),
                        (hexes & (sizes == 64), HEX64_SUBDIV)):
        selected = np.flatnonzero(mask)
        for start in range(0, len(selected), 4096):
            rows = selected[start:start + 4096]
            sub = connectivity[starts[rows, None, None] + table]
            volume[rows] = hex_volumes(positions[sub.reshape(-1, 8)]).reshape(
                len(rows), len(table)).sum(axis=1)
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
    key = (topology["key"], _rest_geometry_digest(mesh), _file_key(os.path.join(
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


