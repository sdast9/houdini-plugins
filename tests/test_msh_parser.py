"""Format-matrix test of the native MSH parser.

The same mesh written as v2.2 ASCII, v2.2 binary, v4.1 ASCII, and v4.1
binary must parse to identical nodes/elements, at comparable speed.
Also parses a real fTetWild v2.2 ASCII mesh and its binary re-save.

Run: hython tests/test_msh_parser.py  (or any python with numpy + gmsh)
"""

import os
import sys
import tempfile
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "src", "common"))
from msh_parser import MshParseError, read_msh  # noqa: E402

BALLHEAD = os.path.join(ROOT, "test_cases", "input", "ballhead.msh")


def canonical(mesh):
    """Format-independent view: node tags, positions, per-family corner-tag
    sets with entities (element order inside a family is format-stable)."""
    order = np.argsort(mesh["node_tags"])
    out = {"tags": mesh["node_tags"][order],
           "points": mesh["points"][order]}
    for family, cell in mesh["cells"].items():
        corner_tags = mesh["node_tags"][cell["corners"]]
        out[family] = (corner_tags, cell["entity"])
    return out


def assert_same(a, b, label):
    ca, cb = canonical(a), canonical(b)
    assert set(ca) == set(cb), (label, set(ca), set(cb))
    assert np.array_equal(ca["tags"], cb["tags"]), label
    assert np.allclose(ca["points"], cb["points"], atol=1e-12), label
    for key in ca:
        if key in ("tags", "points"):
            continue
        assert np.array_equal(ca[key][0], cb[key][0]), (label, key)
        assert np.array_equal(ca[key][1], cb[key][1]), (label, key)


def main():
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    work = tempfile.mkdtemp(prefix="msh_parser_test_")
    variants = {}

    # one physical-grouped cube, written in every format variant
    gmsh.model.add("cube")
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], 7)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.12)
    gmsh.model.mesh.generate(3)
    for version, binary, name in ((2.2, 0, "v22_ascii"), (2.2, 1, "v22_bin"),
                                  (4.1, 0, "v41_ascii"), (4.1, 1, "v41_bin")):
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.option.setNumber("Mesh.Binary", binary)
        path = os.path.join(work, name + ".msh")
        gmsh.write(path)
        variants[name] = path

    meshes = {}
    for name, path in variants.items():
        t0 = time.time()
        meshes[name] = read_msh(path)
        dt = time.time() - t0
        n_tets = len(meshes[name]["cells"]["tet"]["entity"])
        print(f"{name:10s}: {n_tets} tets in {dt * 1000:.1f} ms")

    reference = meshes["v41_ascii"]
    assert len(reference["cells"]["tet"]["entity"]) > 1000
    assert (reference["cells"]["tet"]["entity"] == 7).all()
    for name, mesh in meshes.items():
        assert_same(mesh, reference, f"{name} vs v41_ascii")
    print("PASS: cube identical across all four format variants")

    # real fTetWild output (v2.2 ASCII) and its binary v2.2 re-save
    if os.path.isfile(BALLHEAD):
        ascii_mesh = read_msh(BALLHEAD)
        gmsh.open(BALLHEAD)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 1)
        resave = os.path.join(work, "ballhead_bin.msh")
        gmsh.write(resave)
        with open(resave, "rb") as f:
            assert f.read(20).startswith(b"$MeshFormat\n2.2 1"), "not binary"
        binary_mesh = read_msh(resave)
        assert_same(binary_mesh, ascii_mesh, "ballhead binary vs ascii")
        n = len(ascii_mesh["cells"]["tet"]["entity"])
        print(f"PASS: fTetWild ballhead.msh ({n} tets) "
              "identical in v2.2 ASCII and binary")
    else:
        print(f"SKIP: {BALLHEAD} not found")

    check_review_cases(gmsh, work)
    gmsh.finalize()
    print("\nPASS: MSH parser format matrix")


def write_all(gmsh, work, name, variants=((2.2, 0), (2.2, 1), (4.1, 0),
                                          (4.1, 1))):
    paths = []
    for version, binary in variants:
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.option.setNumber("Mesh.Binary", binary)
        path = os.path.join(work, f"{name}_{version}_{binary}.msh")
        gmsh.write(path)
        paths.append(path)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    return paths


def refused(path, fragment):
    try:
        read_msh(path)
    except MshParseError as exc:
        assert fragment in str(exc), (path, str(exc))
        return
    raise AssertionError(f"{path} was read")


def check_review_cases(gmsh, work):
    """Cases from the 2026-11-14 MSH Reader review."""
    # a volume in two physical groups: MSH 2.2 writes each tet once per
    # group; every format loads it once, in group 1 (PolyFEM's group)
    gmsh.model.add("two_groups")
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], 1, "a")
    gmsh.model.addPhysicalGroup(3, [box], 2, "b")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(3)
    paths = write_all(gmsh, work, "two_groups")
    gmsh.model.remove()
    meshes = [read_msh(path) for path in paths]
    reference = meshes[-1]
    n = len(reference["cells"]["tet"]["entity"])
    for path, mesh in zip(paths, meshes):
        tet = mesh["cells"]["tet"]
        assert len(tet["entity"]) == n and (tet["entity"] == 1).all(), path
        assert np.array_equal(tet["order"], np.arange(n)), path
        assert_same(mesh, reference, path)
        assert mesh["shared_groups"] == [(3, (1, 2))], mesh["shared_groups"]
        expected = {3: n} if "_2.2_" in path else {}
        assert mesh["duplicate_elements"] == expected, (path, expected)
    print(f"PASS: a volume in two physical groups loads {n} tets once "
          "(v2.2 copies left out and counted)")

    # a group holding a surface with reversed orientation: Gmsh writes -5
    geo = os.path.join(work, "reversed.geo")
    with open(geo, "w") as handle:
        handle.write('SetFactory("OpenCASCADE");\nBox(1) = {0,0,0,1,1,1};\n'
                     'Physical Volume("v", 7) = {1};\n'
                     'Physical Surface("a", 5) = {1, -2};\n'
                     "Mesh.MeshSizeMax = 0.3;\n")
    gmsh.open(geo)
    gmsh.model.mesh.generate(3)
    paths = write_all(gmsh, work, "reversed")
    gmsh.model.remove()
    with open(paths[2]) as handle:
        assert " -5 " in handle.read(), "Gmsh no longer writes -5"
    for path in paths:
        tri = read_msh(path)["cells"]["tri"]["entity"]
        assert len(tri) and (tri == 5).all(), (path, np.unique(tri))
    print("PASS: a physical group with a reversed surface is one group 5")

    # partitioned (with ghost cells) and parametric meshes
    gmsh.model.add("parts")
    a = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    b = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, a)], [(3, b)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [a], 4, "left")
    gmsh.model.addPhysicalGroup(3, [b], 9, "right")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(3)
    plain = [read_msh(path) for path in
             write_all(gmsh, work, "whole", ((4.1, 0), (4.1, 1)))]
    whole = plain[0]
    gmsh.option.setNumber("Mesh.SaveParametric", 1)
    for path, same_encoding in zip(
            write_all(gmsh, work, "parametric", ((4.1, 0), (4.1, 1))), plain):
        mesh = read_msh(path)
        assert np.array_equal(mesh["points"], same_encoding["points"]), path
        assert_same(mesh, same_encoding, path)
    gmsh.option.setNumber("Mesh.SaveParametric", 0)
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)
    gmsh.model.mesh.partition(3)
    partitioned = write_all(gmsh, work, "partitioned", ((4.1, 0), (4.1, 1)))
    for path in partitioned:
        mesh = read_msh(path)
        tet = mesh["cells"]["tet"]["entity"]
        assert sorted(np.unique(tet).tolist()) == [4, 9], (path, tet)
        assert len(tet) == len(whole["cells"]["tet"]["entity"]), path
        assert np.array_equal(np.sort(tet),
                              np.sort(whole["cells"]["tet"]["entity"])), path
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 0)
    v40 = write_all(gmsh, work, "v40", ((4.0, 0),))[0]
    gmsh.model.remove()
    refused(v40, "save the mesh again from Gmsh as version 4.1")
    print("PASS: parametric and partitioned (ghost cells) meshes keep "
          "nodes and groups; MSH 4.0 is refused by name")

    # elements pointing at missing nodes, files cut short
    head = ("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n"
            "2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n")
    for tag in (0, 9):
        path = os.path.join(work, f"bad_node_{tag}.msh")
        with open(path, "w") as handle:
            handle.write(head + f"$Elements\n1\n1 4 2 1 1 1 2 3 {tag}\n"
                         "$EndElements\n")
        refused(path, f"node tag {tag}, which is not a node")
    with open(partitioned[1], "rb") as handle:   # binary, cut short
        data = handle.read()
    start = data.index(b"$Elements")
    path = os.path.join(work, "cut.msh")
    with open(path, "wb") as handle:
        handle.write(data[:start + 300] + b"\n$EndElements\n")
    refused(path, "malformed or cut short")
    with open(paths[0], "rb") as handle:         # v2.2 ASCII, cut short
        data = handle.read()
    with open(path, "wb") as handle:
        handle.write(data[:data.index(b"$Elements") + 300]
                     + b"\n$EndElements\n")
    refused(path, "header declared")
    print("PASS: a missing node tag and a truncated file are named errors")


if __name__ == "__main__":
    main()
