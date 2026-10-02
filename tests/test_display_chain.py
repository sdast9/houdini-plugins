"""PolyFEM 2.0 display chain: cost per edit, and exactness of what it stamps.

Until 2026-10-01 every cook of fiberdata_<geo> (each translate or rotate edit,
each handle drag) walked every element in Python to recompute its centroid,
and then resolved the fiber/dispersion sources even when nothing displayed
them: 6.8 s per edit at 1.3M tets, 12 s with a cylindrical fiber shown at
384k tets. The centroids now come from a compiled VEX verb (bitwise the old
values) and the material sources are resolved only while something shows
them.

Run: hython tests/test_display_chain.py
No gmsh needed: the meshes are written here.
"""

import json
import os
import tempfile
import time

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"

# Kuhn split of a cube into six tetrahedra around the 0-6 diagonal.
_KUHN = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
         (0, 4, 5, 6), (0, 5, 1, 6))


def write_box(path, n, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    """Structured tet box as MSH 2.2 ASCII, built without Python loops over
    elements (a 384k-tet box is written in about a second)."""
    xs = [np.linspace(lo[a], hi[a], n + 1) for a in range(3)]
    z, y, x = np.meshgrid(xs[2], xs[1], xs[0], indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                          indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()

    def vid(a, b, c):
        return (c * (n + 1) + b) * (n + 1) + a

    corners = np.stack((vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                        vid(i, j + 1, k), vid(i, j, k + 1),
                        vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1),
                        vid(i, j + 1, k + 1)), axis=1)
    tets = np.concatenate([corners[:, list(t)] for t in _KUHN])
    p = points[tets]
    volume = np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                       np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    flip = volume < 0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2], tets[flip, 1].copy()
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(points)}\n")
        handle.write("".join(
            f"{number} {a!r} {b!r} {c!r}\n"
            for number, (a, b, c) in enumerate(points.tolist(), 1)))
        handle.write(f"$EndNodes\n$Elements\n{len(tets)}\n")
        rows = np.column_stack((np.arange(1, len(tets) + 1), tets + 1))
        handle.write("".join(
            f"{e} 4 2 1 1 {a} {b} {c} {d}\n" for e, a, b, c, d in rows.tolist()))
        handle.write("$EndElements\n")
    return len(tets)


def write_mixed(path):
    """One hexahedron and one tetrahedron in one mesh."""
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n9\n"
                     "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
                     "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n9 2 0 0\n"
                     "$EndNodes\n$Elements\n2\n"
                     "1 5 2 1 1 1 2 3 4 5 6 7 8\n"
                     "2 4 2 1 1 2 9 3 6\n$EndElements\n")


def python_centroids(geo_data, prims):
    """The per-element loop the display chain used before 2026-10-01."""
    positions = np.frombuffer(
        geo_data.pointFloatAttribValuesAsString("P"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3)
    return np.array([
        positions[[v.point().number() for v in geo_data.prim(int(p)).vertices()]]
        .mean(axis=0) for p in prims])


def timed_cook(node):
    start = time.time()
    node.cook(force=True)
    return time.time() - start


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    work = tempfile.mkdtemp(prefix="polyfem_display_chain_")
    mesh = os.path.join(work, "box.msh")
    count = write_box(mesh, 40)        # 384,000 tets

    node = hou.node("/obj").createNode(TYPE, "chain")
    module = node.hdaModule()
    node.setParms({"working_dir": work + "/", "file_location1": mesh})
    start = time.time()
    node.parm("file_location1").pressButton()
    print(f"  import of {count} tets: {time.time() - start:.2f} s")
    stamp = node.node("fiberdata_1")

    # --- no fiber material, nothing displayed: a transform edit is cheap ---
    node.parmTuple("xform_t__1").set((0.5, 0.0, 0.0))
    node.parmTuple("xform_r__1").set((0.0, 30.0, 0.0))
    cost = timed_cook(stamp)
    # The per-element loop took ~2 s here; the verb takes milliseconds.
    assert cost < 0.75, f"fiberdata_1 cook took {cost:.2f} s"
    stamped = stamp.geometry()
    assert not [a.name() for a in stamped.primAttribs()
                if a.name().startswith("pf_")]
    # The stamped centroid is the transformed one, bitwise the old values.
    branch = node.node("branch_1").geometry()
    sample = np.linspace(0, count - 1, 997).astype(int)
    expected = python_centroids(branch, sample)
    computed = module._prim_centroids(branch, None)[sample]
    assert np.array_equal(computed, expected), \
        float(np.abs(computed - expected).max())
    attribute = np.frombuffer(stamped.primFloatAttribValuesAsString(
        "centroid"), dtype=np.float32).reshape(-1, 3)[sample]
    assert np.allclose(attribute, expected, atol=1e-5), \
        float(np.abs(attribute - expected).max())
    print(f"PASS: transform edit recooks the stamp in {cost:.3f} s at "
          f"{count} tets; centroids bitwise equal to the per-element loop")

    # --- a fiber material resolves only while something shows it ----------
    node.setParms({
        "materials1_1": module.MATERIAL_TOKENS.index("HGOFiber"),
        "fib_source1_1": "cylindrical"})
    node.parmTuple("fib_axis_origin1_1").set((-2.0, -2.0, 0.0))
    node.parmTuple("fib_axis_dir1_1").set((0.0, 0.0, 1.0))
    cost = timed_cook(stamp)
    assert cost < 0.75, f"stamp with hidden fibers took {cost:.2f} s"
    assert stamp.geometry().findPrimAttrib("pf_fiber_1_0") is None
    node.setParms({"show_fibers1": 1})
    module.fiber_display_changed({
        "node": node, "parm": node.parm("show_fibers1"),
        "parm_name": "show_fibers1", "script_multiparm_index": "1"})
    node.parmTuple("xform_t__1").set((0.25, 0.0, 0.0))
    cost = timed_cook(stamp)
    # 12 s per edit before (cylindrical frame from a Python centroid loop).
    assert cost < 3.0, f"stamp with cylindrical fibers shown took {cost:.2f} s"
    fibers = np.frombuffer(stamp.geometry().primFloatAttribValuesAsString(
        "pf_fiber_1_0"), dtype=np.float32).reshape(-1, 3)
    assert len(fibers) == count and np.allclose(
        np.linalg.norm(fibers, axis=1), 1.0, atol=1e-5)
    node.node("output").cook(force=True)
    assert node.node("fiberviz_1").geometry().intrinsicValue(
        "primitivecount") > 0
    node.setParms({"show_fibers1": 0})
    module.fiber_display_changed({
        "node": node, "parm": node.parm("show_fibers1"),
        "parm_name": "show_fibers1", "script_multiparm_index": "1"})
    assert stamp.geometry().findPrimAttrib("pf_fiber_1_0") is None
    print(f"PASS: cylindrical fibers shown recook in {cost:.3f} s; hidden "
          "fibers are not resolved at all")

    # --- export with per-element fibers -----------------------------------
    start = time.time()
    params = module.write_params_only({"node": node})
    cost = time.time() - start
    assert params and os.path.isfile(os.path.join(work, "input", "fibers.vtk"))
    data = json.load(open(params))
    assert data["materials"][0]["fiber_direction"]["type"] == \
        "per_element_file"
    assert cost < 5.0, f"export took {cost:.2f} s"     # 11 s before
    print(f"PASS: export with per-element fibers in {cost:.2f} s")

    # --- per-element data on a mixed tet/hex mesh is still refused --------
    mixed = os.path.join(work, "mixed.msh")
    write_mixed(mixed)
    other = hou.node("/obj").createNode(TYPE, "mixed")
    other.setParms({"working_dir": os.path.join(work, "mixed") + "/",
                    "file_location1": mixed})
    other.parm("file_location1").pressButton()
    try:
        other.hdaModule()._check_single_element_family(other, 1)
    except hou.NodeError as exc:
        assert "mixes element types ([4, 8] vertices)" in str(exc), exc
    else:
        raise AssertionError("mixed tet/hex mesh accepted")
    print("PASS: per-element data on a mixed tet/hex mesh is refused")
    print("workdir:", work)


if __name__ == "__main__":
    main()
