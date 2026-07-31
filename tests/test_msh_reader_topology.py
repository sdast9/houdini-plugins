"""Focused MSH Reader topology/connectivity regression tests."""

import os
import tempfile

import hou


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_msh(path, nodes, elements):
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(nodes)}\n")
        for tag, x, y, z in nodes:
            f.write(f"{tag} {x} {y} {z}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(elements)}\n")
        for row in elements:
            f.write(" ".join(str(value) for value in row) + "\n")
        f.write("$EndElements\n")


def point_tags(prim):
    return tuple(point.attribValue("msh_pt_id") for point in prim.points())


def make_reader(path, name):
    node = hou.node("/obj").createNode("geo", name).createNode(
        "MSH_Reader::3.0")
    node.parm("File").set(path)
    node.cook(force=True)
    return node


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    work = tempfile.mkdtemp(prefix="msh_reader_topology_")

    mesh2d = os.path.join(work, "mixed_2d.msh")
    write_msh(
        mesh2d,
        [(10, 0, 0, 0), (20, 1, 0, 0), (30, 1, 1, 0),
         (40, 0, 1, 0), (50, 2, 0, 0), (60, 2, 1, 0)],
        [
            # element tag, type, tag count, physical/entity tags, node tags
            (101, 2, 2, 7, 70, 10, 20, 40),
            (102, 3, 2, 9, 90, 20, 50, 60, 30),
        ])
    reader = make_reader(mesh2d, "reader_2d")
    geo = reader.geometry()
    assert len(geo.points()) == 6
    assert len(geo.prims()) == 2
    assert [point.attribValue("msh_pt_id") for point in geo.points()] == [
        10, 20, 30, 40, 50, 60]
    assert [point_tags(prim) for prim in geo.prims()] == [
        (10, 20, 40), (20, 50, 60, 30)]
    assert [prim.attribValue("Entity") for prim in geo.prims()] == [7, 9]
    assert [prim.attribValue("ElementNum") for prim in geo.prims()] == [0, 1]
    assert [prim.attribValue("is_volume") for prim in geo.prims()] == [1, 1]

    mesh1d = os.path.join(work, "line_1d.msh")
    write_msh(
        mesh1d,
        [(4, 0, 0, 0), (8, 1, 0, 0), (12, 2, 0, 0)],
        [(20, 1, 2, 0, 5, 4, 8), (21, 1, 2, 0, 5, 8, 12)])
    reader = make_reader(mesh1d, "reader_1d")
    geo = reader.geometry()
    assert len(geo.prims()) == 2
    assert [point_tags(prim) for prim in geo.prims()] == [(4, 8), (8, 12)]
    assert [prim.attribValue("Entity") for prim in geo.prims()] == [1, 1]
    assert [prim.attribValue("ElementNum") for prim in geo.prims()] == [0, 1]

    mesh3d = os.path.join(work, "tet_with_surface.msh")
    write_msh(
        mesh3d,
        [(1, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0), (4, 0, 0, 1)],
        [(1, 2, 2, 12, 12, 1, 2, 3),
         (2, 4, 2, 4, 4, 1, 2, 3, 4)])
    reader = make_reader(mesh3d, "reader_3d")
    reader.parm("import_surfaces").set(1)
    reader.cook(force=True)
    geo = reader.geometry()
    assert len(geo.prims()) == 2
    assert point_tags(geo.prim(0)) == (2, 4, 3, 1)
    assert geo.prim(0).attribValue("Entity") == 4
    assert geo.prim(0).attribValue("ElementNum") == 0
    assert geo.prim(0).attribValue("is_volume") == 1
    assert point_tags(geo.prim(1)) == (1, 2, 3)
    assert geo.prim(1).attribValue("surface_entity") == 12
    assert geo.prim(1).attribValue("ElementNum") == -1
    assert geo.prim(1).attribValue("is_volume") == 0

    print("PASS: MSH Reader non-tet/hex topology and numbering")


if __name__ == "__main__":
    main()
