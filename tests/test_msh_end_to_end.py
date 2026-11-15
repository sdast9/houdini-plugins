"""MSH meshes from Houdini to PolyFEM and back, element by element.

1. Subdomains: the MSH Reader reads physical groups itself (signs dropped,
   partitioned and parametric files read), while PolyFEM reads the staged
   .msh with its own reader and takes its body ids from the node's
   volumes<g>.txt. For each mesh variant the solver's elements must be
   Houdini's, in the same order, with the subdomain Houdini shows.
2. Orientation (2026-11-15): tetrahedra built by the MSH Reader and by Read
   PVD are positive by Houdini's convention (measuredvolume > 0; they were
   inside out), boundary faces point outward, and the PolyFEM node's surface
   keeps the face numbering of the old vertex order, which sideset picks
   store.

Run: hython tests/test_msh_end_to_end.py
Uses gmsh (bundled with hython) and ../polyfem/build/PolyFEM_bin.
"""

import os
import subprocess
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
# POLYFEM_BIN overrides the sibling build (a binary other than the shared one).
POLYFEM_BIN = os.environ.get(
    "POLYFEM_BIN", os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin"))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"

# Houdini's tet_faceindex: face f of a tetrahedron, as vertex positions
TET_FACEINDEX = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
OLD_TET_ORDER = [1, 3, 2, 0]   # Gmsh corners per Houdini vertex, before


def make_meshes(work):
    """{label: path}: two unit boxes side by side, groups 4 "left" and
    9 "right", in the variants whose reading changed."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    def save(name, version=4.1, binary=0, **options):
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.option.setNumber("Mesh.Binary", binary)
        path = os.path.join(work, name + ".msh")
        gmsh.write(path)
        for key in options:
            gmsh.option.setNumber(key, 0)
        gmsh.option.setNumber("Mesh.Binary", 0)
        return path

    meshes = {}
    gmsh.model.add("boxes")
    left = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    right = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, left)], [(3, right)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [left], 4, "left")
    gmsh.model.addPhysicalGroup(3, [right], 9, "right")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.35)
    gmsh.model.mesh.generate(3)
    meshes["plain 4.1"] = save("plain41")
    meshes["plain 2.2"] = save("plain22", version=2.2)
    meshes["parametric"] = save("parametric", **{"Mesh.SaveParametric": 1})
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)
    gmsh.model.mesh.partition(3)
    meshes["partitioned ascii"] = save("part_ascii")
    meshes["partitioned binary"] = save("part_binary", binary=1)
    gmsh.model.remove()

    geo = os.path.join(work, "negative.geo")
    with open(geo, "w") as handle:
        handle.write(
            'SetFactory("OpenCASCADE");\n'
            "Box(1) = {0, 0, 0, 1, 1, 1};\nBox(2) = {1, 0, 0, 1, 1, 1};\n"
            "BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }\n"
            'Physical Volume("left", 4) = {1};\n'
            'Physical Volume("right", 9) = {-2};\n'
            "Mesh.MeshSizeMax = 0.35;\n")
    gmsh.open(geo)
    gmsh.model.mesh.generate(3)
    meshes["negative volume tag"] = save("negative")
    with open(meshes["negative volume tag"]) as handle:
        assert " -9 " in handle.read().split("$EndEntities")[0], \
            "Gmsh no longer writes the reversed group as -9"
    gmsh.model.remove()

    gmsh.model.add("hexes")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setTransfiniteAutomatic()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(3)
    meshes["hexahedra"] = save("hexes")
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.finalize()
    return meshes


def measured_volumes(geo, prim_type):
    return np.array([prim.intrinsicValue("measuredvolume")
                     for prim in geo.prims() if prim.type() == prim_type])


def outward(geo, prims, center):
    """Fraction of polygon prims whose normal points away from `center`."""
    good = 0
    for prim in prims:
        centroid = np.mean([v.point().position() for v in prim.vertices()],
                           axis=0)
        good += float(np.dot(np.array(prim.normal()),
                             centroid - center)) > 0
    return good / max(len(prims), 1)


def setup(label, mesh, work):
    node = hou.node("/obj").createNode(TYPE, "e2e_" + label.replace(" ", "_")
                                       .replace(".", ""))
    folder = os.path.join(work, label.replace(" ", "_"))
    os.makedirs(folder)
    node.setParms({"working_dir": folder + "/", "polyfem_bin": POLYFEM_BIN,
                   "log_level": 2})
    node.setParms({"file_location1": mesh})
    node.parm("file_location1").pressButton()
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 1.0, "enable": 0,
                   "body_ids_fields": 1})
    for vol in range(1, node.evalParm("num_volumes1") + 1):
        node.setParms({f"materials1_{vol}": 3, f"E1_{vol}": 1e5,
                       f"nu1_{vol}": 0.3})
    node.setParms({"sideset_selection1_1": 1})
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({"basegroup1_1_1": "axis:-z:0.01",
                   "Boundary_Condition__1_1_1": 1,
                   "boundary_type1_1_1_1": 0, "vector_1_1_1_1": "[0, 0, 0]"})
    return node, folder


def run(node, folder):
    mod = node.hdaModule()
    report = mod.check_setup(node, "run")
    assert not report["errors"], report["errors"]
    params = mod.build_params(node)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "error", "--max_threads", "1"],
        cwd=os.path.dirname(params), capture_output=True, text=True,
        timeout=900)
    assert result.returncode == 0, (result.stdout + result.stderr)[-500:]
    out = os.path.join(folder, "output")
    frames = sorted(name for name in os.listdir(out)
                    if name.endswith((".hdf", ".h5", ".hdf5", ".vtu")))
    pvds = [name for name in os.listdir(out) if name.endswith(".pvd")]
    assert frames and pvds, os.listdir(out)
    return os.path.join(out, frames[0]), os.path.join(out, pvds[0])


def check_subdomains(meshes, work, reader):
    results = {}
    for label, mesh in meshes.items():
        if label == "hexahedra":
            continue
        node, folder = setup(label, mesh, work)
        branch = node.node("branch_1").geometry()
        volume = np.array(branch.primIntAttribValues("is_volume"), bool)
        entity = np.array(branch.primIntAttribValues("Entity"))[volume]
        tags = sorted(set(np.array(branch.primIntAttribValues(
            "msh_physical_tag"))[volume].tolist()))
        assert tags == [4, 9], (label, tags)
        assert node.evalParm("num_volumes1") == 2, label
        prims = [p for p, keep in zip(branch.prims(), volume) if keep]
        centroids = np.array([np.mean([v.point().position()
                                       for v in p.vertices()], axis=0)
                              for p in prims])
        frame, pvd = run(node, folder)
        result = reader.read_mesh_cached(frame)
        cells = result["cells"]["tet"]
        body = result["point_data"]["body_ids"].ravel()[cells[:, 0]]
        solved = result["points"][cells[:, :4]].mean(axis=1)
        assert len(cells) == len(entity), (label, len(cells), len(entity))
        assert np.allclose(solved, centroids, atol=1e-5), label
        assert np.array_equal(body.astype(int), 1000 + entity), label
        results[label] = (node, pvd)
        print(f"PASS: {label}: groups 4/9 -> subdomains 1/2; PolyFEM has "
              f"Houdini's {len(cells)} elements, in order, with its body ids")
    return results


def check_orientation(meshes, results, work):
    # MSH Reader: positive tetrahedra and hexahedra
    for label in ("plain 4.1", "plain 2.2", "hexahedra"):
        reader = hou.node("/obj").createNode("geo").createNode(
            "MSH_Reader::3.0")
        reader.parm("File").set(meshes[label])
        geo = reader.geometry()
        for kind, name in ((hou.primType.Tetrahedron, "tetrahedra"),
                           (hou.primType.Hexahedron, "hexahedra")):
            volumes = measured_volumes(geo, kind)
            if len(volumes):
                assert (volumes > 0).all(), (label, name, volumes.min())
                total = 2.0 if label != "hexahedra" else 1.0
                assert np.isclose(volumes.sum(), total), (label, volumes.sum())
    print("PASS: MSH Reader tetrahedra and hexahedra are positive "
          "(measuredvolume sums to the box volume)")

    # PolyFEM node surface: outward faces, numbering of the old order
    node, _ = results["plain 4.1"]
    surface = node.node("null_1").geometry()
    faces = [p for p in surface.prims() if p.type() == hou.primType.Polygon]
    share = outward(surface, faces, np.array([1.0, 0.5, 0.5]))
    assert share == 1.0, share
    branch = node.node("branch_1").geometry()
    gmsh_corners = []                 # node tag - 1, Gmsh corner order
    for prim in branch.prims():
        if prim.type() != hou.primType.Tetrahedron:
            continue
        tags = [v.point().attribValue("msh_pt_id") - 1
                for v in prim.vertices()]
        corners = [0] * 4
        for position, corner in enumerate([1, 2, 3, 0]):
            corners[corner] = tags[position]
        gmsh_corners.append(corners)
    old = np.array(gmsh_corners)[:, OLD_TET_ORDER][:, TET_FACEINDEX]
    keys = np.sort(old.reshape(-1, 3), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    expected = keys[counts[inverse.ravel()] == 1]
    got = np.sort(np.stack([np.array(surface.primIntAttribValues(f"fp{c}"))
                            for c in range(3)], axis=1), axis=1)
    assert np.array_equal(got, expected), "surface face numbering changed"
    print(f"PASS: the PolyFEM node's {len(got)} surface faces point outward "
          "and keep the numbering of the old vertex order (sideset picks)")

    # Read PVD: positive tetrahedra, outward exterior faces
    _, pvd = results["plain 4.1"]
    viewer = hou.node("/obj").createNode("readPVD::1.0", "e2e_viewer")
    module = viewer.hdaModule()
    hou.setFrame(0)
    viewer.setParms({"PVD_file": pvd})
    module.start({"node": viewer})
    prims = viewer.node("build_prims").geometry()
    volumes = measured_volumes(prims, hou.primType.Tetrahedron)
    assert len(volumes) and (volumes > 0).all(), volumes.min()
    assert np.isclose(volumes.sum(), 2.0), volumes.sum()
    centers = {}
    for prim in prims.prims():
        if prim.type() == hou.primType.Tetrahedron:
            centers[prim.attribValue("source_cell_index")] = np.mean(
                [v.point().position() for v in prim.vertices()], axis=0)
    boundary = viewer.node("boundary_display").geometry()
    faces = boundary.prims()
    # every face of every element (interior ones are dropped downstream),
    # each pointing away from its own element
    assert len(faces) == 4 * len(centers), (len(faces), len(centers))
    away = sum(float(np.dot(np.array(face.normal()), np.mean(
        [v.point().position() for v in face.vertices()], axis=0)
        - centers[face.attribValue("source_cell_index")])) > 0
        for face in faces)
    assert away == len(faces), (away, len(faces))
    interior = boundary.findPrimGroup("readpvd_interior")
    shown = len(faces) - (len(interior.prims()) if interior else 0)
    # the box surface, plus the interface between the bodies from both sides
    assert shown > len(got), (shown, len(got))
    print(f"PASS: Read PVD tetrahedra are positive and all {len(faces)} of "
          f"their faces point outward ({shown} shown)")

    # Read PVD supplementary blocks build volume cells the same way
    block = module._build_block_geometry(
        {"points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
                             [2, 0, 1], [3, 0, 1], [3, 1, 1], [2, 1, 1]],
                            dtype=np.float64),
         "cells": {"tet": np.array([[0, 1, 2, 3]]),
                   "hex": np.array([[4, 5, 6, 7, 8, 9, 10, 11]])},
         "point_data": {}},
        "Obstacle", "obstacle", np.ones((12, 3)), None, 0)
    tet = measured_volumes(block, hou.primType.Tetrahedron)
    hexa = measured_volumes(block, hou.primType.Hexahedron)
    assert np.allclose(tet, [1 / 6]) and np.allclose(hexa, [1.0]), (tet, hexa)
    print("PASS: Read PVD supplementary blocks build positive tetrahedra and "
          "untwisted hexahedra")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    for library in ("sop_MSH_Reader.3.0.hdanc",
                    "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                    "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, library),
                            force_use_assets=True)
    work = tempfile.mkdtemp(prefix="msh_end_to_end_")
    reader = hou.node("/obj").createNode("readPVD::1.0", "e2e_reader")
    meshes = make_meshes(work)
    results = check_subdomains(meshes, work, reader.hdaModule())
    check_orientation(meshes, results, work)
    print("\nPASS: MSH meshes reach PolyFEM element by element, positively "
          "oriented in Houdini")


if __name__ == "__main__":
    main()
