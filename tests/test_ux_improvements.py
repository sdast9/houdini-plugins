"""Regression tests for the 2026-11-19 UX improvements across PolyFEM 2.0
and Read PVD 1.0.

Run: hython tests/test_ux_improvements.py
Needs no PolyFEM binary and no gmsh.
"""

import json
import os
import shutil
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_tet_msh(path):
    text = (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n4\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
        "$EndNodes\n"
        "$Elements\n1\n"
        "1 4 2 1 1 1 2 3 4\n"
        "$EndElements\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_polyfem_ux(root):
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)

    work = os.path.join(root, "pf_ux")
    os.makedirs(work, exist_ok=True)
    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "pf_ux")
    mod = node.hdaModule()
    node.parm("working_dir").set(work + "/")
    mod.working_dir_check({"node": node})

    msh = _write_tet_msh(os.path.join(work, "tet.msh"))
    node.parm("file_location1").set(msh)
    mod.create_geo_nodes({"node": node, "script_multiparm_index": "1"})

    # 1. Three-box BC expression inputs (unquoted, quoted, numeric, empty)
    node.parm("sideset_selection1_1").set(1)
    mod.create_group({"node": node, "script_multiparm_index": "1",
                      "script_multiparm_index2": "1"})
    node.parm("basegroup1_1_1").set("*")
    node.parm("Boundary_Condition__1_1_1").set(2)

    # Vector BC: unquoted TinyExpr, quoted TinyExpr, and plain number
    node.parmTuple("bc_vec1_1_1_1").set(
        ("0", '"-0.01*t"', "1e-3*smoothstep(t)*sin(pi*x)"))
    mod.sync_expr_vector({"node": node,
                          "parm": node.parmTuple("bc_vec1_1_1_1")[0]})
    # Scalar BC: unquoted TinyExpr
    node.parm("boundary_type1_1_1_2").set(3)  # Pressure Boundary
    node.parm("value_1_1_1_2").set("if(t-0.5, 1000, 2000*t)")

    # Initial condition via 3-box ic_vec
    node.parm("initial_conditions1_1").set(1)
    node.parm("conditiontype1_1_1").set(1)  # Velocity
    node.parmTuple("ic_vec1_1_1").set(("", "deg2rad(90)*y", "rotate_2D_x(x,y,t)"))

    # Body load via 3-box rhs_vec
    node.parmTuple("rhs_vec").set(("0", "9.81", "0"))

    data = {}
    mod.build_conditions(node, data)
    bc = data["boundary_conditions"]
    assert bc["rhs"] == [0, 9.81, 0], bc["rhs"]
    assert bc["dirichlet_boundary"][0]["value"] == [
        0, "-0.01*t", "1e-3*smoothstep(t)*sin(pi*x)"
    ], bc["dirichlet_boundary"]
    assert bc["pressure_boundary"][0]["value"] == "if(t-0.5, 1000, 2000*t)", \
        bc["pressure_boundary"]
    assert data["initial_conditions"]["velocity"][0]["value"] == [
        0, "deg2rad(90)*y", "rotate_2D_x(x,y,t)"
    ], data["initial_conditions"]

    # Force fields detect non-zero rhs_vec
    assert "body_forces" in mod.force_field_names(node)

    # Resetting bc_vec back to 0,0,0 via callback overrides legacy vector_
    node.parmTuple("bc_vec1_1_1_1").set(("0", "0", "0"))
    mod.sync_expr_vector({"node": node,
                          "parm": node.parmTuple("bc_vec1_1_1_1")[0]})
    data2 = {}
    mod.build_conditions(node, data2)
    assert data2["boundary_conditions"]["dirichlet_boundary"][0]["value"] == [
        0, 0, 0]

    # 2. TinyExpr validation in Check Setup
    assert mod._validate_tinyexpr("1e-5*smoothstep(t) + min(x, max(y, z))", "ok") is None
    assert mod._validate_tinyexpr("if(t-1, sign(x), half_smoothstep(t))", "ok") is None
    assert "pow" in mod._validate_tinyexpr("t**2", "bad")
    assert "brackets" in mod._validate_tinyexpr("[0, -0.1*t, 0]", "bad")
    assert "unclosed" in mod._validate_tinyexpr("sin(t", "bad")
    assert "unknown_fn" in mod._validate_tinyexpr("unknown_fn(t)", "bad")

    node.parmTuple("bc_vec1_1_1_1").set(("0", "t**2", "bogus_var*t"))
    report = mod.check_setup(node)
    joined = "\n".join(report["errors"])
    assert "uses '**'" in joined, report["errors"]
    assert "bogus_var" in joined, report["errors"]
    node.parmTuple("bc_vec1_1_1_1").set(("0", "0", "0"))
    mod.sync_expr_vector({"node": node,
                          "parm": node.parmTuple("bc_vec1_1_1_1")[0]})

    # 3. Legacy scene migration (OnLoaded helper)
    node.parmTuple("rhs_vec").revertToDefaults()
    node.parm("RHS").set('[0, 9.81, "-1*t"]')
    mod.migrate_legacy_expr_vectors(node)
    assert node.parmTuple("rhs_vec").eval() == ("0", "9.81", "-1*t"), \
        node.parmTuple("rhs_vec").eval()

    # 4. Suggest dhat (0.001 * bounding-box diagonal of unit tet = sqrt(3)*1e-3)
    suggested = mod.suggest_dhat({"node": node})
    assert suggested is not None and abs(suggested - 1e-3 * (3.0 ** 0.5)) < 1e-6
    assert abs(node.evalParm("dhat") - suggested) < 1e-9

    # 5. Viewport dimming when is_enabled1 == 0
    ec = node.node("entitycolor_1")
    ec.cook(force=True)
    assert np.allclose(ec.geometry().primFloatAttribValues("Alpha"), 1.0)
    node.parm("is_enabled1").set(0)
    ec.cook(force=True)
    assert np.allclose(ec.geometry().primFloatAttribValues("Alpha"), 0.25)
    node.parm("is_enabled1").set(1)

    # 6. Slider ranges, solver labels, editable unit presets, Hypre/AMGCL visible
    assert node.parmTuple("xform_r__1").parmTemplate().maxValue() == 180
    assert node.parmTuple("xform_s__1").parmTemplate().maxValue() == 10
    assert node.parm("solver").parmTemplate().label() == "Linear Solver"
    assert node.parm("solver_nl").parmTemplate().label() == "Nonlinear Solver"
    assert "mm" in node.parm("length").parmTemplate().menuItems()
    assert "tonne" in node.parm("mass").parmTemplate().menuItems()
    node.setParms({"expert_mode": 0, "solver": "Hypre"})
    node.updateParmStates()
    assert node.parm("max_iter_hypre").isVisible()
    assert node.parm("tolerance_hypre_AMGCL").isVisible()

    # 7. Distinct non-dark subdomain colors
    c1, c2, c3 = (mod.default_color(1, v) for v in (1, 2, 3))
    assert c1 != c2 and c2 != c3 and max(c3) >= 0.6

    node.destroy()
    print("PASS: PolyFEM 2.0 UX improvements verified")


def test_readpvd_ux(root):
    hou.hda.installFile(
        os.path.join(BASE, "object_readPVD.1.0.hdanc"), force_use_assets=True)
    pvd_dir = os.path.join(root, "pvd_ux")
    os.makedirs(pvd_dir, exist_ok=True)
    # Write a 1-tet VTU with asymmetric stress tensor (non-zero xy/yz/xz)
    vtu_text = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="4" NumberOfCells="1">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
0 0 0  1 0 0  0 1 0  0 0 1</DataArray></Points>
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">4</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
</Cells>
<PointData>
<DataArray type="Float64" Name="F_1" NumberOfComponents="3" format="ascii">
1 0 0  1 0 0  1 0 0  1 0 0</DataArray>
<DataArray type="Float64" Name="F_2" NumberOfComponents="3" format="ascii">
0 1 0  0 1 0  0 1 0  0 1 0</DataArray>
<DataArray type="Float64" Name="F_3" NumberOfComponents="3" format="ascii">
0 0 1  0 0 1  0 0 1  0 0 1</DataArray>
<DataArray type="Float64" Name="cauchy_stress_1" NumberOfComponents="3" format="ascii">
10 2 3  10 2 3  10 2 3  10 2 3</DataArray>
<DataArray type="Float64" Name="cauchy_stress_2" NumberOfComponents="3" format="ascii">
2 20 5  2 20 5  2 20 5  2 20 5</DataArray>
<DataArray type="Float64" Name="cauchy_stress_3" NumberOfComponents="3" format="ascii">
3 5 30  3 5 30  3 5 30  3 5 30</DataArray>
</PointData></Piece></UnstructuredGrid></VTKFile>
"""
    with open(os.path.join(pvd_dir, "step_0.vtu"), "w", encoding="utf-8") as f:
        f.write(vtu_text)
    with open(os.path.join(pvd_dir, "step_1.vtu"), "w", encoding="utf-8") as f:
        f.write(vtu_text)
    pvd_path = os.path.join(pvd_dir, "sim.pvd")
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '<Collection>\n'
            '<DataSet timestep="0.0" file="step_0.vtu"/>\n'
            '<DataSet timestep="1.0" file="step_1.vtu"/>\n'
            '</Collection></VTKFile>\n'
        )
    # Write a realistic run-manifest.json (standalone CLI run: no input/params.json)
    manifest = {
        "completion": {"status": "completed", "exit_status": 0,
                       "wall_seconds": 125.0, "peak_rss_mb": 2048.0},
        "input": {"units": {"length": "mm", "mass": "tonne", "time": "s"}},
        "steps": [
            {"step": 1, "outcome": "accepted", "stall_retunes": 2,
             "al_stagnation": None,
             "contact": {"history_sensitivity": {"continuation": {
                 "losses": {"orientation": 1, "reentry": 2, "other": 0},
                 "split_pairs": 4}}}}
        ],
    }
    with open(os.path.join(pvd_dir, "run-manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f)

    node = hou.node("/obj").createNode("readPVD::1.0", "pvd_ux")
    mod = node.hdaModule()
    node.parm("PVD_file").set(pvd_path)
    mod.sync_available_options({"node": node})

    # 1. Run Summary & per-step diagnostic tag
    summary = mod.run_summary_text(node)
    assert "completed (exit 0)" in summary, summary
    assert "2m 5s" in summary and "2.00 GB" in summary, summary
    assert "stall retunes: 2" in summary and "continuation losses: 3" in summary, summary
    node.updateParmStates()
    assert node.parm("run_summary_status").isDisabled()

    hou.setFrame(1)
    sim_txt = mod.sim_time_text(node)
    assert "[2 stall retunes]" in sim_txt, sim_txt
    hou.setFrame(0)

    # 2. CLI run_units fallback to run-manifest.json
    names, why = mod.run_units(node)
    assert names == ("mm", "tonne", "s") and "run-manifest.json" in why, (names, why)

    # 3. Off-diagonal tensor entries xy, yz, xz in VEX and Python
    node.setParms({"derived": 1, "color_attrib": "cauchy_mat"})
    menu = mod.color_reduction_menu({"node": node})
    assert "xy" in menu and "yz" in menu and "xz" in menu, menu
    out = node.node("OUT_result")
    for red, expected in (("xy", 2.0), ("yz", 5.0), ("xz", 3.0)):
        node.parm("color_reduction").set(red)
        out.cook(force=True)
        vals = out.geometry().pointFloatAttribValues("for_color")
        assert np.allclose(vals, expected), (red, vals, expected)
        mesh = mod.read_vtu(os.path.join(pvd_dir, "step_0.vtu"))
        py_vals = mod._reduce_array(mod._mesh_field(mesh, "cauchy_mat"), red)
        assert np.allclose(py_vals, expected), (red, py_vals, expected)

    # 4. update_playbar toggle exists and defaults to True
    assert node.evalParm("update_playbar") == 1

    # 5. Scoped cache invalidation preserves unrelated PVD entries
    other_key = ("/some/other/run/step_0.vtu", 1, 1)
    mod._VTU_CACHE[other_key] = {"dummy": True}
    mod._invalidate_pvd_caches(pvd_path)
    assert other_key in mod._VTU_CACHE
    mod._VTU_CACHE.pop(other_key, None)

    node.destroy()
    print("PASS: Read PVD 1.0 UX improvements verified")


def main():
    root = tempfile.mkdtemp(prefix="hda_ux_improvements_")
    try:
        test_polyfem_ux(root)
        test_readpvd_ux(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL UX IMPROVEMENT TESTS PASSED")


if __name__ == "__main__":
    main()
