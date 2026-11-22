"""Write Houdini solid primitives as linear Gmsh 2.2 ASCII elements."""

import os
import tempfile

import hou
import numpy as np


_SOP_VOLUME_TAGS = ("Entity", "entity", "subdomain")
_SOP_GMSH_FACES = {
    4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    5: ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
}


def _sop_volume_tags(sop_geo, names):
    for name in names:
        attr = sop_geo.findPrimAttrib(name)
        if attr is not None and attr.dataType() == hou.attribData.Int \
                and attr.size() == 1 and not attr.isArrayType():
            return np.frombuffer(
                sop_geo.primIntAttribValuesAsString(name, int_type=hou.numericData.Int64),
                dtype=np.int64)
    return None


def _orient_sop_cells(points, vertices, prims, element_type, orient):
    label = "tetrahedron" if element_type == 4 else "hexahedron"
    _, first, counts = np.unique(np.sort(vertices, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    duplicates = np.flatnonzero(counts > 1)
    if len(duplicates):
        raise hou.NodeError(
            f"Duplicate {label} connectivity at primitive {prims[first[duplicates[0]]]}."
        )

    origin = points[vertices[:, 0]]
    second, third = (2, 3) if element_type == 4 else (3, 4)
    with np.errstate(over="ignore", invalid="ignore"):
        volume = np.einsum(
            "ij,ij->i",
            np.cross(points[vertices[:, 1]] - origin,
                     points[vertices[:, second]] - origin),
            points[vertices[:, third]] - origin)
    invalid = np.flatnonzero((volume == 0) | ~np.isfinite(volume))
    if len(invalid):
        raise hou.NodeError(
            f"Primitive {prims[invalid[0]]} is a degenerate {label} "
            "(zero or non-finite signed volume).")
    if orient:
        flipped = volume < 0
        order = (0, 1, 3, 2) if element_type == 4 else (4, 5, 6, 7, 0, 1, 2, 3)
        vertices[flipped] = vertices[flipped][:, order]
    return vertices.tolist()


def _check_sop_skin(skin, cells):
    """Allow the faces emitted alongside the tets by Houdini's tet SOPs."""
    if not skin:
        return
    unmatched = dict(skin)
    for element_type, elements in cells.items():
        for vertices in elements:
            for face in _SOP_GMSH_FACES[element_type]:
                unmatched.pop(tuple(sorted(vertices[index] for index in face)), None)
            if not unmatched:
                return
    primitive = next(iter(unmatched.values()))
    raise hou.NodeError(
        f"Mixed geometry: polygon primitive {primitive} is not a face of a "
        "volume element. Isolate the solid tetrahedra or hexahedra in the Source SOP.")


def _write_sop_volume_msh(sop_geo, dst_path, *, allow_hexes=True, orient=True,
                          tag_attributes=_SOP_VOLUME_TAGS, allow_tag_zero=False):
    """Keep Houdini point and volume primitive order and atomically write MSH."""
    if not sop_geo.intrinsicValue("pointcount"):
        raise hou.NodeError("Source SOP has no points.")
    points = np.frombuffer(
        sop_geo.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float64),
        dtype=np.float64).reshape(-1, 3)
    invalid_points = np.flatnonzero(~np.isfinite(points).all(axis=1))
    if len(invalid_points):
        raise hou.NodeError(f"Point {invalid_points[0]} has a non-finite position.")

    tags = _sop_volume_tags(sop_geo, tag_attributes)
    min_tag = 0 if allow_tag_zero else 1
    volume_types = {hou.primType.Tetrahedron: (4, 4)}
    if allow_hexes:
        volume_types[hou.primType.Hexahedron] = (5, 8)
    shapes = "tetrahedra or hexahedra" if allow_hexes else "tetrahedra"
    cells = {4: [], 5: []}
    prims = {4: [], 5: []}
    rows, skin = [], []
    for prim in sop_geo.prims():
        number, kind = prim.number(), prim.type()
        if kind == hou.primType.Polygon:
            if prim.isClosed() and prim.numVertices() in (3, 4):
                corners = tuple(sorted(vertex.point().number() for vertex in prim.vertices()))
            else:
                # A polygon of any arity is a surface, but only a full triangle
                # or quad can be a face of one of the supported solid elements.
                corners = (number,)
            skin.append((corners, number))
            continue
        if kind not in volume_types:
            raise hou.NodeError(
                f"Unsupported primitive {number} ({kind}). Isolate the {shapes} "
                "in the source geometry.")
        element_type, arity = volume_types[kind]
        vertices = [vertex.point().number() for vertex in prim.vertices()]
        if len(set(vertices)) != arity or len(vertices) != arity \
                or min(vertices) < 0 or max(vertices) >= len(points):
            raise hou.NodeError(f"Primitive {number} must reference {arity} distinct points.")
        if element_type == 5:
            vertices = [vertices[index] for index in (0, 1, 3, 2, 4, 5, 7, 6)]
        tag = int(tags[number]) if tags is not None else min_tag
        rows.append((element_type, len(cells[element_type]), tag))
        cells[element_type].append(vertices)
        prims[element_type].append(number)

    if not rows:
        if skin:
            raise hou.NodeError(
                f"Source SOP contains surface polygons and no {shapes}. "
                "Use Tet Conform or Tet Embed to create a solid tetrahedral mesh.")
        raise hou.NodeError(f"Source SOP has no {shapes}.")

    _check_sop_skin(skin, cells)
    for element_type, elements in cells.items():
        if elements:
            cells[element_type] = _orient_sop_cells(
                points, np.asarray(elements), prims[element_type], element_type, orient)

    group_tags = sorted({tag for _, _, tag in rows})
    shift = max(0, min_tag - group_tags[0])
    if group_tags[-1] + shift > np.iinfo(np.int32).max:
        raise hou.NodeError("Volume tags exceed the Gmsh integer range after shifting.")
    # The standalone writer has always numbered geometrical entities by class.
    # For PolyFEM keep the existing equal physical and geometrical tags.
    geom_tags = {tag: index for index, tag in enumerate(group_tags, 1)} \
        if allow_tag_zero else {tag: tag + shift for tag in group_tags}

    dst_path = os.path.abspath(dst_path)
    directory = os.path.dirname(dst_path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".write_msh_", dir=directory) as stage:
        candidate = os.path.join(stage, os.path.basename(dst_path))
        with open(candidate, "w", encoding="ascii", newline="\n") as handle:
            handle.write(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{len(points)}\n")
            for index, (x, y, z) in enumerate(points.tolist(), 1):
                handle.write(f"{index} {x!r} {y!r} {z!r}\n")
            handle.write(f"$EndNodes\n$Elements\n{len(rows)}\n")
            for eid, (element_type, index, tag) in enumerate(rows, 1):
                vertices = " ".join(str(vertex + 1) for vertex in cells[element_type][index])
                handle.write(f"{eid} {element_type} 2 {tag + shift} {geom_tags[tag]} {vertices}\n")
            handle.write("$EndElements\n")
        os.replace(candidate, dst_path)
    return {"n_points": len(points), "n_tets": len(cells[4]), "n_hexes": len(cells[5]),
            "n_groups": len(group_tags), "n_skin": len(skin)}
