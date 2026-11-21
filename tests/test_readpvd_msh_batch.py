"""Regression & end-to-end tests for Read PVD & MSH Reader Batch (2026-11-20):
  1. Higher-order Q2 (27-node) & Q3 (64-node) Lagrange hex + TriQuadratic hex (type 29)
     subdivision in vtu_parser & Read PVD (verified against real PolyFEM_bin Q2/Q3 runs)
  2. Read PVD Follow Running Simulation (F7): poll interval gating, live new-step
     refresh + jump to latest frame, and automatic stop when run-manifest completes
  3. MSH Reader F10 & mixed-mesh order: msh_element_order, msh_surface_physical_names,
     surface_entity = -1 on primary prims, 1D boundary lines on 2D meshes,
     create_groups (phys_*/surf_*), msh_loose_nodes point group, & read-only Mesh Summary

Run via hython:
  hython tests/test_readpvd_msh_batch.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import hou
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src", "common"))
import vtu_parser

POLYFEM_BIN = os.environ.get(
    "POLYFEM_BIN",
    os.path.abspath(
        os.path.join(BASE, "..", "polyfem", "build", "PolyFEM_bin")))


def _hex_volumes(points, hexes):
    """Exact volume of linear P1 hexes by splitting each into 5 tets (or corner det)."""
    vols = []
    # Split VTK hex [0,1,2,3,4,5,6,7] into 6 positive tets around diagonal 0-6
    tets_idx = (
        (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
    )
    for h in hexes:
        p = points[h]
        v = 0.0
        for a, b, c, d in tets_idx:
            v += np.linalg.det(np.column_stack([p[b] - p[a], p[c] - p[a], p[d] - p[a]])) / 6.0
        vols.append(v)
    return np.asarray(vols, dtype=np.float64)


def test_higher_order_hex_subdivision(work):
    # 1. Real PolyFEM_bin Q2 (27-node) and Q3 (64-node) Lagrange hex runs
    msh = os.path.join(work, "unit_hex.msh")
    with open(msh, "w") as handle:
        handle.write(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n8\n"
            "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
            "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
            "$EndNodes\n"
            "$Elements\n1\n1 5 2 1 1 1 2 3 4 5 6 7 8\n$EndElements\n")

    for order, expected_hexes in ((2, 8), (3, 27)):
        out_dir = os.path.join(work, f"hex_q{order}")
        os.makedirs(out_dir, exist_ok=True)
        cfg = {
            "geometry": [{"mesh": msh, "volume_selection": 1}],
            "materials": [{"id": 1, "type": "NeoHookean", "E": 1e5,
                           "nu": 0.3, "rho": 1000}],
            "space": {"discr_order": order},
            "boundary_conditions": {
                "dirichlet_boundary": [{"id": 1, "value": [0, 0, 0]}],
            },
            "output": {
                "directory": out_dir,
                "paraview": {
                    "file_name": "sim.vtu",
                    "high_order_mesh": True,
                    "vismesh_rel_area": 10.0,
                    "options": {"use_hdf5": False},
                },
            },
        }
        jpath = os.path.join(out_dir, "params.json")
        with open(jpath, "w") as handle:
            json.dump(cfg, handle)
        proc = subprocess.run(
            [POLYFEM_BIN, "-j", jpath],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr[-500:]
        vtu_path = os.path.join(out_dir, "sim.vtu")
        mesh = vtu_parser.read_vtu(vtu_path)
        hexes = mesh["cells"]["hex"]
        assert len(hexes) == expected_hexes, (order, len(hexes))
        vols = _hex_volumes(mesh["points"], hexes)
        assert np.all(vols > 0.0), f"Q{order} sub-hex inverted: {vols}"
        assert abs(vols.sum() - 1.0) < 1e-6, (order, vols.sum())

        # Also cook through Read PVD 1.0 with surface_only=0 to verify Houdini hexes
        pvd_path = os.path.join(out_dir, "sim.pvd")
        with open(pvd_path, "w") as handle:
            handle.write(
                '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                '  <Collection><DataSet timestep="0" file="sim.vtu"/></Collection>\n'
                '</VTKFile>\n')
        node = hou.node("/obj").createNode("readPVD::1.0", f"pvd_q{order}")
        node.setParms({"PVD_file": pvd_path, "surface_only": 0})
        node.hdaModule().start({"node": node})
        out = node.node("OUT_result")
        out.cook(force=True)
        g = out.geometry()
        assert g.intrinsicValue("primitivecount") == expected_hexes, (
            order, g.intrinsicValue("primitivecount"))
        node.destroy()

    # 2. VTK_TRIQUADRATIC_HEXAHEDRON (type 29, 27 nodes, nodes 18<->19 swapped)
    # Build canonical 27-node VTK type 29 coordinates on [0,1]^3
    pts29 = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0],
        [0.5, 0, 1], [1, 0.5, 1], [0.5, 1, 1], [0, 0.5, 1],
        [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5],
        [0, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 1, 0.5],
        [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, 0.5, 0.5],
    ], dtype=np.float64)
    mesh29 = vtu_parser._mesh_from_arrays(
        pts29, np.arange(27, dtype=np.int64), np.array([27], dtype=np.int64),
        np.array([29], dtype=np.int64), {}, {})
    vols29 = _hex_volumes(mesh29["points"], mesh29["cells"]["hex"])
    assert len(vols29) == 8 and np.allclose(vols29, 0.125, atol=1e-9), vols29
    print("PASS: Q2 (8 sub-hexes), Q3 (27 sub-hexes), and TriQuadratic type 29 hex subdivision")


def _write_ascii_tet_vtu(path, z_shift):
    pts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.0, 1.0 + z_shift),
    ]
    pts_str = " ".join(f"{x} {y} {z}" for x, y, z in pts)
    sol_str = " ".join(f"0 0 {z_shift}" for _ in pts)
    vm_str = " ".join(f"{10.0 * (1.0 + z_shift)}" for _ in pts)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        '  <UnstructuredGrid>\n'
        '    <Piece NumberOfPoints="4" NumberOfCells="1">\n'
        '      <PointData>\n'
        f'        <DataArray type="Float64" Name="solution" NumberOfComponents="3" format="ascii">{sol_str}</DataArray>\n'
        f'        <DataArray type="Float64" Name="von_mises" NumberOfComponents="1" format="ascii">{vm_str}</DataArray>\n'
        '      </PointData>\n'
        '      <Points>\n'
        f'        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{pts_str}</DataArray>\n'
        '      </Points>\n'
        '      <Cells>\n'
        '        <DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>\n'
        '        <DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>\n'
        '        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>\n'
        '      </Cells>\n'
        '    </Piece>\n'
        '  </UnstructuredGrid>\n'
        '</VTKFile>\n'
    )
    with open(path, "w") as handle:
        handle.write(xml)


def test_follow_running_simulation(work):
    run_dir = os.path.join(work, "live_run")
    os.makedirs(run_dir, exist_ok=True)
    _write_ascii_tet_vtu(os.path.join(run_dir, "step_0.vtu"), 0.0)
    pvd_path = os.path.join(run_dir, "sim.pvd")
    with open(pvd_path, "w") as handle:
        handle.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '  <Collection>\n'
            '    <DataSet timestep="0.0" file="step_0.vtu"/>\n'
            '  </Collection>\n</VTKFile>\n')

    node = hou.node("/obj").createNode("readPVD::1.0", "live_viewer")
    mod = node.hdaModule()
    node.setParms({"PVD_file": pvd_path})
    mod.start({"node": node})

    # Poll Interval disabled while follow_running is off
    assert node.evalParm("follow_running") == 0
    node.updateParmStates()
    assert node.parm("follow_interval").isDisabled()

    # Real PolyFEM writes completion={"status": "running", ...} from step 0.
    # Follow Running Simulation MUST stay checked while status == "running".
    manifest_path = os.path.join(run_dir, "run-manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump({
            "schema": "polyfem-run-manifest",
            "completion": {
                "status": "running",
                "exit_status": None,
                "finished_at": None,
                "message": None,
                "peak_rss_mb": None,
                "wall_seconds": None,
            },
        }, handle)

    # Enable Follow Running Simulation
    node.parm("follow_running").set(1)
    node.updateParmStates()
    assert not node.parm("follow_interval").isDisabled()
    mod.toggle_follow_running({"node": node})
    assert node.evalParm("follow_running") == 1, \
        "follow_running must stay enabled while manifest status is 'running'"
    assert node.sessionId() in mod._FOLLOW_WATCHERS

    # Unchanged PVD -> noop (and still following)
    res0 = mod.poll_running_simulation(node, force=True, jump_frame=True)
    assert res0["action"] == "noop" and res0["steps"] == 1, res0
    assert node.evalParm("follow_running") == 1

    # Simulate solver writing step 1
    _write_ascii_tet_vtu(os.path.join(run_dir, "step_1.vtu"), 0.5)
    with open(pvd_path, "w") as handle:
        handle.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '  <Collection>\n'
            '    <DataSet timestep="0.0" file="step_0.vtu"/>\n'
            '    <DataSet timestep="0.1" file="step_1.vtu"/>\n'
            '  </Collection>\n</VTKFile>\n')
    res1 = mod.poll_running_simulation(node, force=True, jump_frame=True)
    assert res1["action"] == "refreshed" and res1["steps"] == 2, res1
    assert int(hou.frame()) == 1, hou.frame()
    assert node.evalParm("follow_running") == 1

    # Simulate solver writing finalized run-manifest.json -> auto-stop following
    with open(manifest_path, "w") as handle:
        json.dump({
            "schema": "polyfem-run-manifest",
            "completion": {"status": "completed", "exit_status": 0},
        }, handle)
    res2 = mod.poll_running_simulation(node, force=True, jump_frame=True)
    assert res2["action"] == "finished" and res2["steps"] == 2, res2
    assert node.evalParm("follow_running") == 0
    assert node.sessionId() not in mod._FOLLOW_WATCHERS
    node.destroy()
    print("PASS: Read PVD Follow Running Simulation (live refresh, jump, auto-stop)")


def test_single_block_loading(work):
    mb_dir = os.path.join(work, "multiblock")
    os.makedirs(mb_dir, exist_ok=True)
    vol = os.path.join(mb_dir, "step_0.vtu")
    surf = os.path.join(mb_dir, "step_0_surf.vtu")
    contact = os.path.join(mb_dir, "step_0_contact.vtu")
    _write_ascii_tet_vtu(vol, 0.0)
    _write_ascii_tet_vtu(surf, 0.1)
    _write_ascii_tet_vtu(contact, 0.2)
    vtm = os.path.join(mb_dir, "step_0.vtm")
    with open(vtm, "w") as handle:
        handle.write(
            '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
            '  <vtkMultiBlockDataSet>\n'
            '    <Block index="0" name="Volume"><DataSet index="0" file="step_0.vtu"/></Block>\n'
            '    <Block index="1" name="Surface"><DataSet index="0" file="step_0_surf.vtu"/></Block>\n'
            '    <Block index="2" name="Contact"><DataSet index="0" file="step_0_contact.vtu"/></Block>\n'
            '  </vtkMultiBlockDataSet>\n</VTKFile>\n')
    pvd = os.path.join(mb_dir, "sim.pvd")
    with open(pvd, "w") as handle:
        handle.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '  <Collection><DataSet timestep="0.0" file="step_0.vtm"/></Collection>\n'
            '</VTKFile>\n')

    node = hou.node("/obj").createNode("readPVD::1.0", "single_block_viewer")
    mod = node.hdaModule()
    mod._VTU_CACHE.clear()
    mesh = mod.load_block(pvd, 0, "Volume")
    assert mesh is not None
    cached_names = [os.path.basename(k[0]) for k in mod._VTU_CACHE.keys()]
    assert cached_names == ["step_0.vtu"], cached_names
    node.destroy()
    print("PASS: Read PVD single-block load avoids parsing unused sibling VTM blocks")


def test_msh_reader_introspection_and_order(work):
    # 1. Mixed 3D tet+hex mesh with:
    #    - 1 positive tet (file order 0, phys 1 "body_a")
    #    - 1 hex          (file order 1, phys 2 "body_b")
    #    - 1 inverted tet (file order 2, phys 1 "body_a")
    #    - 1 physical 2D surface tri (phys 10 "bottom face")
    #    - 1 loose unreferenced node (node 15)
    msh3d = os.path.join(work, "mixed_3d.msh")
    with open(msh3d, "w") as handle:
        handle.write("\n".join([
            "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
            "$PhysicalNames", "3",
            '2 10 "bottom face"',
            '3 1 "body_a"',
            '3 2 "body_b"',
            "$EndPhysicalNames",
            "$Nodes", "15",
            # Tet 0 (positive)
            "1 0 0 0", "2 1 0 0", "3 0 1 0", "4 0 0 1",
            # Hex 1 (positive cube shifted in x)
            "5 2 0 0", "6 3 0 0", "7 3 1 0", "8 2 1 0",
            "9 2 0 1", "10 3 0 1", "11 3 1 1", "12 2 1 1",
            # Extra nodes for inverted tet 2 (swapped 3<->4 orientation)
            "13 0 2 0", "14 0 2 1",
            # Loose unreferenced node (e.g. higher-order mid-node)
            "15 5 5 5",
            "$EndNodes",
            "$Elements", "4",
            "1 4 2 1 1 1 2 3 4",                # order 0: positive tet
            "2 5 2 2 2 5 6 7 8 9 10 11 12",     # order 1: hex
            "3 4 2 1 1 1 3 2 4",                # order 2: inverted tet
            "4 2 2 10 10 1 2 3",                # 2D surface tri
            "$EndElements",
        ]) + "\n")

    geo_obj = hou.node("/obj").createNode("geo", "msh_batch_test")
    reader = geo_obj.createNode("MSH_Reader::3.0", "msh3d")
    reader.setParms({"File": msh3d, "import_surfaces": 1, "create_groups": 1})
    reader.cook(force=True)
    g = reader.geometry()

    # msh_element_order preserves exact file order across mixed families
    orders = [prim.attribValue("msh_element_order") for prim in g.prims()]
    # build_prims emits tets first (orders 0, 2), then hex (order 1), then surface tri (-1)
    assert orders == [0, 2, 1, -1], orders

    # surface_entity defaults to -1 on volume prims, 10 on surface tri
    surf_ents = [prim.attribValue("surface_entity") for prim in g.prims()]
    assert surf_ents == [-1, -1, -1, 10], surf_ents

    # msh_surface_physical_names preserved
    snames = json.loads(g.attribValue("msh_surface_physical_names"))
    assert snames == {"10": "bottom face"}, snames

    # Named Houdini prim groups created when create_groups == 1
    group_names = {grp.name() for grp in g.primGroups()}
    assert {"phys_body_a", "phys_body_b", "surf_bottom_face"} <= group_names, group_names
    assert len(g.findPrimGroup("phys_body_a").prims()) == 2
    assert len(g.findPrimGroup("phys_body_b").prims()) == 1
    assert len(g.findPrimGroup("surf_bottom_face").prims()) == 1

    # Loose node tagged in msh_loose_nodes point group (only point index 14 / node 15)
    assert g.attribValue("msh_loose_node_count") == 3  # nodes 13, 14, 15 unused
    loose_grp = g.findPointGroup("msh_loose_nodes")
    assert loose_grp is not None and len(loose_grp.points()) == 3
    assert g.findPointAttrib("__msh_loose") is None  # cleaned up

    # Read-only Mesh Summary reports version, counts, loose nodes, and inverted tet
    reader.updateParmStates()
    assert reader.parm("mesh_summary").isDisabled()
    summary = reader.evalParm("mesh_summary")
    assert "MSH 2.2" in summary and "3D" in summary, summary
    assert "2 tet" in summary and "1 hex" in summary, summary
    assert "1 inverted/zero-volume tets" in summary, summary
    assert "loose/higher-order nodes" in summary, summary

    # 2. 2D triangle mesh with 1D physical boundary line
    msh2d = os.path.join(work, "tri_2d.msh")
    with open(msh2d, "w") as handle:
        handle.write("\n".join([
            "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
            "$PhysicalNames", "2",
            '1 5 "inlet"',
            '2 1 "domain"',
            "$EndPhysicalNames",
            "$Nodes", "3",
            "1 0 0 0", "2 1 0 0", "3 0 1 0",
            "$EndNodes",
            "$Elements", "2",
            "1 2 2 1 1 1 2 3",
            "2 1 2 5 5 1 2",
            "$EndElements",
        ]) + "\n")
    r2d = geo_obj.createNode("MSH_Reader::3.0", "msh2d")
    r2d.setParms({"File": msh2d, "import_surfaces": 1, "create_groups": 1})
    r2d.cook(force=True)
    g2 = r2d.geometry()
    assert g2.intrinsicValue("primitivecount") == 2
    assert g2.prim(0).attribValue("is_volume") == 1
    assert g2.prim(0).attribValue("surface_entity") == -1
    assert g2.prim(0).attribValue("msh_element_order") == 0
    assert g2.prim(1).attribValue("is_volume") == 0
    assert g2.prim(1).attribValue("surface_entity") == 5
    assert g2.prim(1).attribValue("msh_element_order") == -1
    assert json.loads(g2.attribValue("msh_surface_physical_names")) == {"5": "inlet"}
    assert g2.findPrimGroup("surf_inlet") is not None
    assert g2.findPrimGroup("phys_domain") is not None
    print("PASS: MSH Reader msh_element_order, surface names/lines, groups, loose nodes, & summary")


def main():
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_readPVD.1.0.hdanc"), force_use_assets=True)
    work = tempfile.mkdtemp(prefix="readpvd_msh_batch_")
    try:
        test_higher_order_hex_subdivision(work)
        test_follow_running_simulation(work)
        test_single_block_loading(work)
        test_msh_reader_introspection_and_order(work)
        print("ALL READPVD & MSH READER BATCH TESTS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
