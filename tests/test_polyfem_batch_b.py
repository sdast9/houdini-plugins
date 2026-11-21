"""Regression & end-to-end tests for PolyFEM Workflow Batch B (2026-11-20):
  1. Smart defaults ($POLYFEM_BIN, $HIP/polyfem/$OS/, non-destructive polyfem_bin_check)
  2. Output drawer promotions (use_hdf5, minimal_fields, forces_fields directly under Paraview)
  3. Direct Houdini SOP input (F1): TetConform volume mesh with subdomains + real PolyFEM_bin solve,
     plus polygon SOP as rigid obstacle surface
  4. Rebuild Display Network button + OnLoaded display-version upgrade

Run via hython:
  hython tests/test_polyfem_batch_b.py
"""
import json
import os
import shutil
import subprocess
import tempfile

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
POLYFEM_BIN = os.environ.get(
    "POLYFEM_BIN",
    os.path.abspath(
        os.path.join(BASE, "..", "polyfem", "build", "PolyFEM_bin")))


def test_smart_defaults_and_non_destructive_bin_check():
    prev_env = os.environ.get("POLYFEM_BIN")
    os.environ["POLYFEM_BIN"] = POLYFEM_BIN
    try:
        node = hou.node("/obj").createNode(TYPE, "batch_b_defaults")
        assert node.parm("polyfem_bin").eval() == POLYFEM_BIN, (
            node.parm("polyfem_bin").eval())
        assert node.parm("working_dir").unexpandedString() == "$HIP/polyfem/$OS/", (
            node.parm("working_dir").unexpandedString())
        assert node.evalParm("forces_fields") == 1
        assert node.userData("polyfem_display_version") == str(
            node.hdaModule().DISPLAY_CHAIN_VERSION)

        # Mistyped binary path warns without wiping the user's text
        typo = "/nonexistent/path/PolyFEM_bin_typo"
        node.parm("polyfem_bin").set(typo)
        node.hdaModule().polyfem_bin_check({"node": node})
        assert node.parm("polyfem_bin").eval() == typo, (
            "polyfem_bin_check wiped the user's typed path")
        node.destroy()
    finally:
        if prev_env is None:
            os.environ.pop("POLYFEM_BIN", None)
        else:
            os.environ["POLYFEM_BIN"] = prev_env
    print("PASS: smart defaults ($POLYFEM_BIN, $HIP/polyfem/$OS/) & non-destructive bin check")


def test_output_drawer_promotions(node):
    for parm_name in ("use_hdf5", "minimal_fields", "forces_fields"):
        p = node.parm(parm_name)
        assert p is not None, f"missing {parm_name}"
        folders = p.containingFolders()
        assert "Paraview" in folders, (parm_name, folders)
        assert "Options" not in folders, (
            f"{parm_name} is still buried inside Options: {folders}")
        assert "Volume Specific Data" not in folders, (
            f"{parm_name} is still buried inside Volume Specific Data: {folders}")
    # Conditional still works after promotion
    node.parm("materials_fields").set(1)
    node.updateParmStates()
    assert node.parm("minimal_fields").isDisabled()
    node.parm("materials_fields").set(0)
    node.updateParmStates()
    assert not node.parm("minimal_fields").isDisabled()
    print("PASS: use_hdf5, minimal_fields, and forces_fields promoted under Paraview")


def test_direct_sop_input_and_rebuild_display(work):
    # Build a Houdini TetConform volume SOP with 2 subdomains via prim Entity
    src_obj = hou.node("/obj").createNode("geo", "batch_b_sops")
    box = src_obj.createNode("box", "unit_box")
    box.parmTuple("size").set((1.0, 1.0, 1.0))
    box.parmTuple("t").set((0.5, 0.5, 0.5))
    tc = src_obj.createNode("tetconform", "tets")
    tc.setInput(0, box)
    ent = src_obj.createNode("attribwrangle", "split_entity")
    ent.setInput(0, tc)
    ent.parm("class").set(1)
    ent.parm("snippet").set('i@Entity = (@P.y > 0.5) ? 2 : 1;')
    ent.cook(force=True)

    # Also build a polygon surface SOP for obstacle testing
    grid = src_obj.createNode("grid", "floor_grid")
    grid.setParms({"rows": 2, "cols": 2, "sizex": 2.0, "sizey": 2.0})
    grid.parmTuple("t").set((0.5, -0.5, 0.5))
    grid.cook(force=True)

    node = hou.node("/obj").createNode(TYPE, "batch_b_sop_sim")
    mod = node.hdaModule()
    node.setParms({
        "working_dir": work + "/",
        "polyfem_bin": POLYFEM_BIN,
        "quasistatic": 1,
        "define_run_by": 0,
        "tend": 1.0,
        "dt": 1.0,
        "enable": 0,
        "use_hdf5": 0,
        "log_level": 2,
    })

    # 1. Surface-only SOP rejected as simulated volume with clear guidance
    node.setParms({"mesh_source1": 1, "sop_path1": grid.path()})
    assert mod.load_sop_mesh({"node": node, "script_multiparm_index": "1"}) is False
    assert node.node("geo_1") is None

    # 2. TetConform SOP loads as 2-subdomain volume mesh (auto-oriented positive)
    node.parm("sop_path1").set(ent.path())
    assert mod.load_sop_mesh({"node": node, "script_multiparm_index": "1"}) is True
    assert node.node("geo_1") is not None
    assert node.evalParm("num_volumes1") == 2, node.evalParm("num_volumes1")
    node.updateParmStates()
    assert not node.parm("sop_path1").isDisabled()
    assert node.parm("file_location1").isDisabled()

    # Add a Dirichlet BC on bottom (y <= 0.05) and top (y >= 0.95) across subdomains
    for vol, pat, vec in ((1, "axis:-y:0.05", ("0", "0", "0")),
                          (2, "axis:+y:0.95", ("0", "-0.05*t", "0"))):
        node.setParms({f"sideset_selection1_{vol}": 1})
        mod.create_group({
            "node": node, "script_multiparm_index2": "1",
            "script_multiparm_index": str(vol)})
        node.setParms({f"basegroup1_{vol}_1": pat})
        node.parmTuple(f"bc_vec1_{vol}_1_1").set(vec)
        mod.sync_expr_vector({
            "node": node, "parm": node.parm(f"bc_vec1_{vol}_1_1x")})

    # 3. Add Geometry 2 as rigid obstacle from polygon SOP
    node.parm("num_geos").set(2)
    node.setParms({
        "is_obstacle2": 1,
        "mesh_source2": 1,
        "sop_path2": grid.path(),
    })
    assert mod.load_sop_mesh({"node": node, "script_multiparm_index": "2"}) is True
    assert node.node("geo_2") is not None
    assert node.node("null_2").geometry().intrinsicValue("primitivecount") == 2

    # 4. Write & run real PolyFEM_bin (verifies positive tet orientation!)
    params_path = mod._prepare_inputs(node, "write")
    assert params_path is not None and os.path.isfile(params_path)
    if os.path.isfile(POLYFEM_BIN):
        proc = subprocess.run(
            [POLYFEM_BIN, "-j", params_path],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, (
            f"PolyFEM_bin failed on Houdini TetConform SOP mesh:\n"
            f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
        print("PASS: direct Houdini TetConform SOP (2 subdomains) + obstacle SOP solved in PolyFEM_bin")
    else:
        print("PASS: direct Houdini TetConform SOP + obstacle SOP exported cleanly")

    # 5. Rebuild Display Network & OnLoaded version check preserve everything
    ec = node.node("entitycolor_1")
    ec.parm("snippet").set("// stale v0 snippet")
    node.setUserData("polyfem_display_version", "0")
    upgraded = mod.check_display_version(node)
    assert upgraded == 2, upgraded
    assert node.userData("polyfem_display_version") == str(mod.DISPLAY_CHAIN_VERSION)
    assert "stale v0" not in node.node("entitycolor_1").evalParm("snippet")
    assert "f@Alpha = 0.25" in node.node("entitycolor_1").evalParm("snippet")
    # Sidesets and materials survived untouched
    assert node.evalParm("basegroup1_1_1") == "axis:-y:0.05"
    assert node.evalParm("basegroup1_2_1") == "axis:+y:0.95"
    assert node.node("group_1_1_1") is not None
    assert node.node("group_1_2_1") is not None

    # Second check_display_version is a no-op once at current version
    assert mod.check_display_version(node) == 0
    # Manual Rebuild Display Network button rebuilds cleanly
    assert mod.rebuild_display_network({"node": node}) == 2
    print("PASS: Rebuild Display Network & OnLoaded version upgrade preserve all settings")


def main():
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)
    work = tempfile.mkdtemp(prefix="polyfem_batch_b_")
    try:
        test_smart_defaults_and_non_destructive_bin_check()
        probe = hou.node("/obj").createNode(TYPE, "batch_b_output_probe")
        test_output_drawer_promotions(probe)
        probe.destroy()
        test_direct_sop_input_and_rebuild_display(work)
        print("ALL BATCH B TESTS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
