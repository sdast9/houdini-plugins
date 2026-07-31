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
from msh_parser import read_msh  # noqa: E402

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

    gmsh.finalize()
    print("\nPASS: MSH parser format matrix")


if __name__ == "__main__":
    main()
