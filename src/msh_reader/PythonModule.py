# MSH_Reader 3.0 cook logic.
#
# The heavy lifting happens in two stages:
#   1. this module (called from the internal Python SOP): native numpy MSH
#      parse -> bulk points + detail connectivity/entity arrays
#   2. a detail VEX wrangle downstream builds tet/hex/poly/line prims from the
#      arrays (compiled, ~1M prims/s) and prim wrangles add ElementNum,
#      centroid in parallel.
#
# Attribute contract (matches MSH_Reader 2.2):
#   point  msh_pt_id   original gmsh node tag (int)
#   prim   Entity      physical volume tag (3D groups; +1 shift if any 0)
#   prim   ElementNum  sequential element number
#   prim   centroid    element centroid (vector)
#   prim   msh_physical_tag  the element's physical tag as written in the file
#                      (0 = no physical group), without the Entity shift
#   detail msh_physical_names  JSON {"<tag>": "<name>"} of the primary
#                      dimension's $PhysicalNames (empty object if none)
#   detail msh_duplicate_elements  primary elements an MSH 2.2 file wrote
#                      more than once (once per physical group) and that were
#                      loaded once; PolyFEM reads every copy and stops
# New in 3.0: 2D and 1D meshes are emitted as primary elements. Optional
#   lower-dimensional physical groups in 3D are imported as boundary polys
#   with prim surface_entity (enables gmsh-authored sidesets downstream).

import json
import re

import numpy as np

import hou

# ---- embedded native parser (built from src/common/msh_parser.py) ----------
# @MSH_PARSER@
# ---- end embedded parser ----------------------------------------------------

FAMILY_DIM = {
    "point": 0, "line": 1, "tri": 2, "quad": 2,
    "tet": 3, "hex": 3, "prism": 3, "pyramid": 3,
}
PRIMARY_FAMILIES = {
    1: ("line",),
    2: ("tri", "quad"),
    3: ("tet", "hex"),
}


def _sanitize_group_name(prefix, tag, name):
    """Build a valid Houdini group name from a physical group tag and name."""
    raw = name.strip() if name else str(int(tag))
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = str(int(tag))
    return f"{prefix}_{cleaned}"


def mesh_summary_text(node):
    """Read-only Mesh Summary string for the MSH Reader parameter pane."""
    path = node.evalParm("File").strip()
    if node.parm("import_surfaces") is not None:
        node.evalParm("import_surfaces")
    if not path:
        return "Select a .msh file."
    inner = node.node("read_msh")
    if inner is None:
        return "Not loaded yet."
    try:
        if inner.needsToCook():
            inner.cook(force=False)
        geo = inner.geometry()
        if geo is not None and geo.findGlobalAttrib("msh_summary") is not None:
            return str(geo.attribValue("msh_summary"))
    except hou.Error:
        pass
    return "Mesh failed to load; check node errors."


def cook(node):
    """Cook body of the internal Python SOP."""
    geo = node.geometry()
    asset = node.parent()

    path = asset.evalParm("File")
    if not path:
        return

    try:
        mesh = read_msh(path)
    except (OSError, MshParseError) as e:
        raise hou.NodeError(f"Failed to read MSH file: {e}")
    except Exception as e:  # a parser defect: still say which file
        raise hou.NodeError(
            f"Failed to read MSH file {path}: {type(e).__name__}: {e}")

    points = mesh["points"]
    n_points = len(points)
    if n_points == 0:
        raise hou.NodeError("MSH file contains no nodes")

    # --- bulk point creation + msh_pt_id ---------------------------------
    geo.createPoints(np.ascontiguousarray(points, dtype=np.float64))
    geo.addAttrib(hou.attribType.Point, "msh_pt_id", -1, create_local_variable=False)
    tags = np.asarray(mesh["node_tags"], dtype=np.int64)
    if len(tags) and int(tags.min()) >= -2147483648 and int(tags.max()) <= 2147483647:
        geo.setPointIntAttribValuesFromString(
            "msh_pt_id", np.ascontiguousarray(tags, dtype=np.int32).tobytes(),
            int_type=hou.numericData.Int32)
    else:
        geo.setPointIntAttribValues("msh_pt_id", tags.tolist())

    cells = mesh["cells"]
    if not cells:
        raise hou.NodeError(
            f"MSH file contains {n_points} nodes but no elements: mesh the "
            "geometry in Gmsh (or save all elements) before exporting")

    # Entity convention from 2.2: polyfem ids must be positive, so if any
    # primary element has physical tag 0 (no physical group), shift ALL by +1.
    mesh_dim = max(FAMILY_DIM[f] for f in cells)
    primary = tuple(f for f in PRIMARY_FAMILIES.get(mesh_dim, ()) if f in cells)
    if not primary:
        families = sorted(f for f in cells if FAMILY_DIM[f] == mesh_dim)
        raise hou.NodeError(
            f"Unsupported primary element families in {mesh_dim}D mesh: {families}")
    primary_entities = [cells[f]["entity"] for f in primary]
    shift = 1 if min(int(e.min()) for e in primary_entities) <= 0 else 0

    geo.addAttrib(
        hou.attribType.Global, "mesh_dim", -1, create_local_variable=False)
    geo.setGlobalAttribValue("mesh_dim", mesh_dim)

    def put(name, array, tuple_size):
        geo.addArrayAttrib(hou.attribType.Global, name, hou.attribData.Int, tuple_size)
        geo.setGlobalAttribValue(name, np.ascontiguousarray(array, dtype=np.int64).ravel().tolist())

    def put_strings(name, values):
        geo.addArrayAttrib(hou.attribType.Global, name, hou.attribData.String, 1)
        geo.setGlobalAttribValue(name, list(values))

    create_groups = (asset.parm("create_groups") is not None
                     and bool(asset.evalParm("create_groups")))
    import_surfaces = bool(asset.evalParm("import_surfaces"))
    all_names = mesh.get("physical_names", {})

    used_points = np.zeros(n_points, dtype=bool)
    for family in primary:
        corners = cells[family]["corners"]
        used_points[corners] = True
        put(f"{family}_conn", corners, 1)
        put(f"{family}_entity", cells[family]["entity"] + shift, 1)
        put(f"{family}_physical", cells[family]["entity"], 1)
        if "order" in cells[family]:
            put(f"{family}_order", cells[family]["order"], 1)
        if create_groups:
            ent = cells[family]["entity"]
            lut = {
                int(t): _sanitize_group_name(
                    "phys", t, all_names.get((mesh_dim, int(t)), ""))
                for t in np.unique(ent)}
            put_strings(f"{family}_group", [lut[int(t)] for t in ent.tolist()])

    names = {str(tag): name for (dim, tag), name in
             all_names.items() if dim == mesh_dim}
    geo.addAttrib(hou.attribType.Global, "msh_physical_names", "",
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_physical_names", json.dumps(names))

    surf_dim = mesh_dim - 1
    surf_names = {str(tag): name for (dim, tag), name in
                  all_names.items() if dim == surf_dim and surf_dim >= 1}
    geo.addAttrib(hou.attribType.Global, "msh_surface_physical_names", "",
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_surface_physical_names", json.dumps(surf_names))

    imported_surface_counts = {}
    if mesh_dim == 3 and import_surfaces:
        for family, conn_name, ent_name in (
                ("tri", "tri_conn", "tri_entity"),
                ("quad", "quad_conn", "quad_entity")):
            if family in cells:
                # only import surfaces that belong to a physical group
                keep = cells[family]["entity"] != 0
                if keep.any():
                    s_corners = cells[family]["corners"][keep]
                    s_ent = cells[family]["entity"][keep]
                    used_points[s_corners] = True
                    put(conn_name, s_corners, 1)
                    put(ent_name, s_ent, 1)
                    imported_surface_counts[family] = int(keep.sum())
                    if create_groups:
                        lut = {
                            int(t): _sanitize_group_name(
                                "surf", t, all_names.get((2, int(t)), ""))
                            for t in np.unique(s_ent)}
                        put_strings(f"{family}_group",
                                    [lut[int(t)] for t in s_ent.tolist()])
    elif mesh_dim == 2 and import_surfaces and "line" in cells:
        keep = cells["line"]["entity"] != 0
        if keep.any():
            s_corners = cells["line"]["corners"][keep]
            s_ent = cells["line"]["entity"][keep]
            used_points[s_corners] = True
            put("line_conn", s_corners, 1)
            put("line_entity", s_ent, 1)
            imported_surface_counts["line"] = int(keep.sum())
            if create_groups:
                lut = {
                    int(t): _sanitize_group_name(
                        "surf", t, all_names.get((1, int(t)), ""))
                    for t in np.unique(s_ent)}
                put_strings("line_group",
                            [lut[int(t)] for t in s_ent.tolist()])

    n_loose = int((~used_points).sum())
    geo.addAttrib(hou.attribType.Global, "msh_loose_node_count", 0,
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_loose_node_count", n_loose)
    if n_loose > 0:
        geo.addAttrib(hou.attribType.Point, "__msh_loose", 0,
                      create_local_variable=False)
        geo.setPointIntAttribValuesFromString(
            "__msh_loose",
            np.ascontiguousarray((~used_points).astype(np.int32)).tobytes(),
            int_type=hou.numericData.Int32)

    n_inverted_tets = 0
    if "tet" in primary:
        tc = cells["tet"]["corners"]
        v0 = points[tc[:, 0]]
        vol6 = np.einsum(
            "ij,ij->i",
            np.cross(points[tc[:, 1]] - v0, points[tc[:, 2]] - v0),
            points[tc[:, 3]] - v0)
        n_inverted_tets = int((vol6 <= 0.0).sum())

    supported = set(primary)
    if mesh_dim == 3:
        supported.update(("tri", "quad", "line", "point"))
    elif mesh_dim == 2:
        supported.update(("line", "point"))
    elif mesh_dim == 1:
        supported.add("point")
    warnings = []
    unsupported = sorted(set(cells) - supported)
    if unsupported:
        counts = ", ".join(f"{len(cells[f]['entity'])} {f}"
                           for f in unsupported)
        warnings.append(
            f"Ignored elements this reader does not load: {counts} (only "
            f"{'/'.join(primary)} are loaded as {mesh_dim}D elements), so "
            "part of the mesh is missing.")

    # Elements in more than one physical group. PolyFEM assigns each element
    # one group, the first; MSH 2.2 also writes such an element once per
    # group, and those copies were left out above.
    shown_dims = {mesh_dim}
    if mesh_dim in (2, 3) and import_surfaces:
        shown_dims.add(mesh_dim - 1)

    def label(dim, tag):
        name = all_names.get((dim, tag))
        return f'{tag} "{name}"' if name else str(tag)

    for dim, tags in mesh.get("shared_groups", ()):
        if dim not in shown_dims:
            continue
        others = ", ".join(label(dim, t) for t in tags[1:])
        warnings.append(
            f"Physical groups {', '.join(label(dim, t) for t in tags)} "
            f"share elements; each is loaded in its first group, "
            f"{label(dim, tags[0])}, as PolyFEM assigns it, so {others} "
            "get none of them.")
    duplicates = int(mesh.get("duplicate_elements", {}).get(mesh_dim, 0))
    geo.addAttrib(hou.attribType.Global, "msh_duplicate_elements", 0,
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_duplicate_elements", duplicates)

    # Human-readable mesh summary for the Mesh Summary readout.
    elem_parts = [f"{len(cells[f]['entity']):,} {f}" for f in primary]
    prim_tags = sorted({int(t) for f in primary for t in np.unique(cells[f]["entity"])})
    group_labels = [
        f"{t}:{all_names[(mesh_dim, t)]}" if (mesh_dim, t) in all_names else str(t)
        for t in prim_tags[:8]]
    if len(prim_tags) > 8:
        group_labels.append(f"+{len(prim_tags) - 8} more")
    version_txt = mesh.get("version", "?")
    summary_lines = [
        f"MSH {version_txt} | {mesh_dim}D | {n_points:,} nodes | "
        + ", ".join(elem_parts),
        f"Physical groups ({len(prim_tags)}): " + ", ".join(group_labels),
    ]
    flags = []
    if imported_surface_counts:
        flags.append(
            "imported boundaries: "
            + ", ".join(f"{c:,} {f}" for f, c in imported_surface_counts.items()))
    elif surf_names:
        flags.append(f"{len(surf_names)} named boundary group(s) available")
    if n_loose:
        flags.append(f"{n_loose:,} loose/higher-order nodes (group: msh_loose_nodes)")
    if n_inverted_tets:
        flags.append(f"WARNING: {n_inverted_tets:,} inverted/zero-volume tets")
    if duplicates:
        flags.append(f"WARNING: {duplicates:,} duplicate MSH 2.2 elements")
    if flags:
        summary_lines.append(" | ".join(flags))
    geo.addAttrib(hou.attribType.Global, "msh_summary", "",
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_summary", "\n".join(summary_lines))

    if duplicates:
        warnings.append(
            f"This MSH 2.2 file writes {duplicates} elements twice or more "
            "(once per physical group they are in); each is loaded once. "
            "PolyFEM reads every copy and stops ('the same element'), so the "
            "PolyFEM node refuses this mesh: save it from Gmsh as MSH 4.1 "
            "(each element once, in its first group), or put each element "
            "in a single physical group.")
    if warnings:
        raise hou.NodeWarning("\n".join(warnings))
