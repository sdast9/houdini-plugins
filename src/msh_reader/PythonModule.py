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
    geo.createPoints(points.tolist())
    geo.addAttrib(hou.attribType.Point, "msh_pt_id", -1, create_local_variable=False)
    geo.setPointIntAttribValues("msh_pt_id", mesh["node_tags"].astype(np.int64).tolist())

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

    for family in primary:
        put(f"{family}_conn", cells[family]["corners"], 1)
        put(f"{family}_entity", cells[family]["entity"] + shift, 1)
        put(f"{family}_physical", cells[family]["entity"], 1)

    names = {str(tag): name for (dim, tag), name in
             mesh.get("physical_names", {}).items() if dim == mesh_dim}
    geo.addAttrib(hou.attribType.Global, "msh_physical_names", "",
                  create_local_variable=False)
    geo.setGlobalAttribValue("msh_physical_names", json.dumps(names))

    if mesh_dim == 3 and asset.evalParm("import_surfaces"):
        for family, conn_name, ent_name in (
                ("tri", "tri_conn", "tri_entity"),
                ("quad", "quad_conn", "quad_entity")):
            if family in cells:
                # only import surfaces that belong to a physical group
                keep = cells[family]["entity"] != 0
                if keep.any():
                    put(conn_name, cells[family]["corners"][keep], 1)
                    put(ent_name, cells[family]["entity"][keep], 1)

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
    if mesh_dim == 3 and asset.evalParm("import_surfaces"):
        shown_dims.add(2)
    all_names = mesh.get("physical_names", {})

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
