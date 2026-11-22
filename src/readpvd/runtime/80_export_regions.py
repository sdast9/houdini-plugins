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
        selected = point_node[members]
        nodes[selected[selected >= 0]] = True
        # elements with a whole face on the sideset
        faces = (np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
                 np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                           [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]))
        for hexes, table in ((False, faces[0]), (True, faces[1])):
            rows = np.flatnonzero(topology["hex"] == hexes)
            if not len(rows):
                continue
            corner_node = point_node[topology["corners"][rows][:, :8 if hexes
                                                               else 4]]
            on = nodes[corner_node]                           # (n, k)
            elements[rows] = np.any(np.all(on[:, table], axis=2), axis=1)
        description = f"the sideset {entry.get('label')} (its nodes; the " \
                      "elements with a face on it)"
        if entry.get("kind") == "nodes":
            hits = nodes[point_node[topology["members"]]]
            elements[topology["owner"][hits]] = True
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
        around = nodes[point_node[topology["members"]]]
        elements[topology["owner"][around]] = True
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
    geo = _probe_geometry(node)
    if point < 0 or geo is None:
        _message("Enable the probe (Probe tab) and click the result first.")
        return
    row = _result_row(geo, point)
    if row < 0:
        _message("The probed point is not in the result on screen.")
        return
    rest_attrib = geo.findPointAttrib("rest")
    probed = geo.point(row)
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


