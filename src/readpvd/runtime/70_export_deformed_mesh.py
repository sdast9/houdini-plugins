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
        if values is not None and values.ndim == 2 and values.shape[1] >= 2:
            return _point_vectors(mesh, name)
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
    rest = geometry["rest"]
    rest_extent = float(np.ptp(rest, axis=0).max()) if len(rest) else 1.0
    if spread > 1e-9 * max(scale, 1e-6 * rest_extent, 1e-12):
        raise ExportError(
            f"Geometry {geometry['number']}: copies of a node have different "
            f"displacements (by {spread:.3g}); the mesh is not conforming.")
    used = np.zeros(count, dtype=bool)
    used[file_nodes] = True
    deformed = rest + displacement
    corners = geometry["file_corners"]
    is_hex = geometry["is_hex"]
    ratio = np.empty(len(corners))
    locally_inverted = np.zeros(len(corners), dtype=bool)
    indeterminate = np.zeros_like(locally_inverted)
    for hexes in (False, True):
        local = np.flatnonzero(is_hex == hexes)
        if not len(local):
            continue
        if hexes:
            before = hex_volumes(rest[corners[local]])
            current = deformed[corners[local]]
            after = hex_volumes(current)
            locally_inverted[local], indeterminate[local] = (
                _hex_inversion_certificate(current, np.sign(before)))
        else:
            c = corners[local, :4]
            before = _tet_volumes(*(rest[c[:, i]] for i in range(4)))
            after = _tet_volumes(*(deformed[c[:, i]] for i in range(4)))
        ratio[local] = after / np.where(before == 0, np.nan, before)
    bad = np.flatnonzero((np.isfinite(ratio) & (ratio <= 0)) | locally_inverted)
    if len(bad):
        listed = ", ".join(str(int(i) + 1) for i in bad[:10])
        more = f" and {len(bad) - 10} more" if len(bad) > 10 else ""
        raise ExportError(
            f"Geometry {geometry['number']}: {len(bad)} element(s) are "
            f"inverted or flat in the deformed shape (elements {listed}"
            f"{more} of {os.path.basename(geometry['path'])}); PolyFEM cannot "
            "use such a mesh. Export an earlier step.")
    ambiguous = np.flatnonzero(indeterminate | ~np.isfinite(ratio))
    if len(ambiguous):
        listed = ", ".join(str(int(i) + 1) for i in ambiguous[:10])
        more = f" and {len(ambiguous) - 10} more" if len(ambiguous) > 10 else ""
        raise ExportError(
            f"Geometry {geometry['number']}: inversion check is indeterminate "
            f"(possibly inverted, elements {listed}{more} of "
            f"{os.path.basename(geometry['path'])}). Export an earlier step.")
    if coordinates == "simulation":
        positions = deformed
    else:
        positions = np.array(parsed["points"], dtype=np.float64)
        pulled = np.linalg.solve(geometry["A"], (deformed - geometry["b"]).T).T
        positions[used] = pulled[used]
    report = {"nodes": int(used.sum()), "unused": int((~used).sum()),
              "elements": len(corners),
              "hex_corners_checked": bool(is_hex.any()),
              "ratio_min": float(np.nanmin(ratio)),
              "ratio_min_at": int(np.nanargmin(ratio)) + 1,
              "ratio_max": float(np.nanmax(ratio)),
              "ratio_max_at": int(np.nanargmax(ratio)) + 1}
    return positions, report


def _sampled_curved_inversions(mesh, topology, geometry, deformed):
    """Count source curved cells with a nonpositive sampled subcell orientation."""
    start, stop = geometry["start"], geometry["stop"]
    rows = np.arange(start, stop)
    rows = rows[topology["sizes"][rows] > topology["corner_count"][rows]]
    if not len(rows):
        return None
    bad = np.zeros(len(mesh["cell_types"]), dtype=bool)
    for family in ("tet", "hex"):
        parents = rows[topology["hex"][rows] == (family == "hex")]
        if not len(parents):
            continue
        corners = topology["corners"][parents, :8 if family == "hex" else 4]
        rest = mesh["points"][corners]
        orientation = np.full(len(bad), np.nan)
        parent_cells = topology["cells"][parents]
        orientation[parent_cells] = np.sign(
            hex_volumes(rest) if family == "hex"
            else _tet_volumes(*rest.transpose(1, 0, 2)))
        is_parent = np.zeros(len(bad), dtype=bool)
        is_parent[parent_cells] = True
        source = mesh["cell_sources"][family]
        display = mesh["cells"][family]
        for offset in range(0, len(source), 4096):
            span = slice(offset, offset + 4096)
            mask = is_parent[source[span]]
            if not mask.any():
                continue
            ids = source[span][mask]
            nodes = deformed[display[span][mask]]
            sign = orientation[ids]
            if family == "hex":
                invalid = _inverted_hex_corners(nodes, sign)
                invalid |= ~(_hex_jacobian_determinants(nodes, 0.5, 0.5, 0.5) * sign > 0)
            else:
                invalid = ~(_tet_volumes(*nodes.transpose(1, 0, 2)) * sign > 0)
            bad[ids[invalid]] = True
    return int(np.count_nonzero(bad))


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
    moved = points + solution
    has_curved_output = np.any(topology["sizes"] > topology["corner_count"])
    for geometry in geometries:  # check everything before writing anything
        positions, report = deformed_mesh_positions(
            geometry, match, topology, solution, coordinates)
        curvature = _p2_curvature(geometry, topology, points, moved)
        sampled = (_sampled_curved_inversions(mesh, topology, geometry, moved)
                   if has_curved_output else None)
        results.append((geometry, positions, report, curvature, sampled))
    folder = export_folder(node)
    os.makedirs(folder, exist_ok=True)
    width = max(3, len(str(len(entries) - 1)))
    stems = [os.path.splitext(os.path.basename(g["path"]))[0]
             for g, *_ in results]
    lines = [f"Step {step} of {len(entries) - 1} (t = {entries[step][0]:.6g})"
             f" written to {folder}"]
    written = []
    used_names = set()
    for (geometry, positions, report, curvature, sampled), stem in zip(results, stems):
        if stems.count(stem) > 1:
            stem = f"{stem}_geo{geometry['number']}"
        extension = os.path.splitext(geometry["path"])[1] or ".msh"
        # simulation coordinates get their own name: loading one in place of
        # the other would apply the Transform twice (or not at all)
        suffix = "_simulation" if coordinates == "simulation" else ""
        tail = f"_step{step:0{width}d}{suffix}{extension}"
        filename = stem + tail
        serial = 2
        while filename.casefold() in used_names:
            filename = f"{stem}_{serial}{tail}"
            serial += 1
        used_names.add(filename.casefold())
        path = os.path.join(folder, filename)
        write_moved_nodes(geometry["path"], positions, path)
        written.append(path)
        inversion_check = ("none inverted in the exported linear mesh "
                           "(tet signs and Q1 hex Bernstein certificates)"
                           if report["hex_corners_checked"] else "none inverted")
        if sampled is not None:
            inversion_check = (
                f"{sampled} higher-order source element(s) with suspect orientation "
                "at sampled corner/interior points" if sampled else
                "no inversions detected at sampled corner/interior points "
                "of higher-order source elements")
        lines.append(
            f"Geometry {geometry['number']}: {os.path.basename(path)} -- "
            f"{report['nodes']} nodes moved, {report['elements']} elements, "
            f"element volume ratio {report['ratio_min']:.4g} (element "
            f"{report['ratio_min_at']}) to {report['ratio_max']:.4g} "
            f"(element {report['ratio_max_at']}), {inversion_check}.")
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
    except (ExportError, OSError, MshParseError, VtuParseError,
            ET.ParseError, ValueError) as exc:
        report, written = f"Not exported: {exc}", []
    node.parm("export_mesh_report").set(report)
    if written and hou.isUIAvailable():
        hou.ui.setStatusMessage(
            f"Deformed mesh written: {', '.join(map(os.path.basename, written))}",
            hou.severityType.ImportantMessage)
    elif not written:
        _message(report)
    return written


