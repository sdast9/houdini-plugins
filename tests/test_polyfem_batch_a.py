"""Regression tests for PolyFEM Workflow Batch A (2026-11-20):
  1. Time tab 'Define Run By' menu + legacy toggle migration & Import round-trip
  2. Structured orthotropic elasticity inputs (E1..E3, nu, G) + PD validation
  3. Unit-aware Gravity (-Y) & Clear Body Load helpers
  4. Extra params.json deep-merge passthrough + Check Setup validation
  5. Create Sidesets from Gmsh 2D Physical Surfaces (idempotent)

Run via hython:
  hython tests/test_polyfem_batch_a.py
"""
import json
import os
import shutil
import tempfile

import hou
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def _write_cube_msh_with_surfaces(path):
    """Write a 6-tet unit cube MSH 2.2 with two named 2D physical surfaces:
    tag 10 'bottom_fixed' (z=0, 2 tris) and tag 20 'top_load' (z=1, 2 tris),
    and one 3D physical volume tag 1 'body'."""
    corners = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
    ]
    # 6 positive-volume tets partitioning the unit cube
    tets = [
        (1, 2, 3, 7), (1, 3, 4, 7), (1, 4, 8, 7),
        (1, 8, 5, 7), (1, 5, 6, 7), (1, 6, 2, 7),
    ]
    # Boundary tris on z=0 (tag 10) and z=1 (tag 20)
    tris = [
        (10, (1, 2, 3)), (10, (1, 3, 4)),   # bottom_fixed
        (20, (5, 6, 7)), (20, (5, 7, 8)),   # top_load
    ]
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        "3",
        '2 10 "bottom_fixed"',
        '2 20 "top_load"',
        '3 1 "body"',
        "$EndPhysicalNames",
        "$Nodes",
        str(len(corners)),
    ]
    for tag, x, y, z in corners:
        lines.append(f"{tag} {x} {y} {z}")
    lines.extend(["$EndNodes", "$Elements", str(len(tris) + len(tets))])
    eid = 1
    for ptag, (a, b, c) in tris:
        lines.append(f"{eid} 2 2 {ptag} {ptag} {a} {b} {c}")
        eid += 1
    for a, b, c, d in tets:
        lines.append(f"{eid} 4 2 1 1 {a} {b} {c} {d}")
        eid += 1
    lines.append("$EndElements")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def test_time_define_run_by(node, mod, work):
    # Default: tend_dt (0)
    assert node.parm("define_run_by").evalAsString() == "tend_dt"
    node.updateParmStates()
    assert not node.parm("tend").isDisabled()
    assert not node.parm("dt").isDisabled()
    assert node.parm("num_timesteps").isDisabled()

    # Switch to tend_steps (1) via callback
    node.parm("define_run_by").set(1)
    mod.sync_time_mode({"node": node})
    node.updateParmStates()
    assert not node.parm("tend").isDisabled()
    assert node.parm("dt").isDisabled()
    assert not node.parm("num_timesteps").isDisabled()
    node.setParms({"t0": 0.0, "tend": 2.0, "num_timesteps": 10})
    data = {}
    mod.build_time(node, data)
    assert data["time"]["tend"] == 2.0
    assert data["time"]["time_steps"] == 10
    assert "dt" not in data["time"]

    # Switch to dt_steps (2) via callback
    node.parm("define_run_by").set(2)
    mod.sync_time_mode({"node": node})
    node.updateParmStates()
    assert node.parm("tend").isDisabled()
    assert not node.parm("dt").isDisabled()
    assert not node.parm("num_timesteps").isDisabled()
    node.setParms({"dt": 0.05, "num_timesteps": 20})
    data2 = {}
    mod.build_time(node, data2)
    assert data2["time"]["dt"] == 0.05
    assert data2["time"]["time_steps"] == 20
    assert "tend" not in data2["time"]

    # Headless set of define_run_by without callback still works
    node.setParms({"end_time_bool": 1, "time_inc_bool": 1,
                   "num_timesteps_bool": 0, "define_run_by": 1})
    assert mod._time_flags(node) == (True, False, True)

    # Legacy scene migration (bools set, define_run_by still at default 0)
    node.setParms({"define_run_by": 0, "end_time_bool": 0,
                   "time_inc_bool": 1, "num_timesteps_bool": 1})
    mod.migrate_legacy_time_mode(node)
    assert node.parm("define_run_by").evalAsString() == "dt_steps"

    # Reset back to tend_dt
    node.parm("define_run_by").set(0)
    mod.sync_time_mode({"node": node})
    print("PASS: Time tab Define Run By menu, export, and legacy migration")


def test_structured_orthotropic(node, mod, work):
    tokens = mod.MATERIAL_TOKENS
    node.setParms({
        "materials1_1": tokens.index("HookeLinearElasticity"),
        "elast_mode1_1": 0,
    })
    node.updateParmStates()
    assert not node.parmTuple("ortho_E1_1").isDisabled()
    assert node.parm("elast_tensor1_1").isDisabled()
    # Valid out-of-the-box defaults pass Check Setup
    report = mod.check_setup(node, "write")
    assert not report["errors"], report["errors"]

    # Custom structured constants export cleanly
    node.parmTuple("ortho_E1_1").set((2e5, 3e5, 4e5))
    node.parmTuple("ortho_nu1_1").set((0.25, 0.2, 0.15))
    node.parmTuple("ortho_G1_1").set((5e4, 6e4, 7e4))
    mod.sync_ortho_tensor({
        "node": node, "script_multiparm_index2": "1",
        "script_multiparm_index": "1"})
    mat = mod.build_material(node, 1, 1)
    assert mat["type"] == "HookeLinearElasticity"
    assert np.allclose(
        mat["elasticity_tensor"],
        [2e5, 3e5, 4e5, 0.25, 0.2, 0.15, 5e4, 6e4, 7e4]), mat

    # Non-positive-definite orthotropic constants caught in Check Setup
    node.parmTuple("ortho_E1_1").set((1e5, 5e5, 5e5))
    node.parmTuple("ortho_nu1_1").set((0.49, 0.49, 0.49))
    bad_report = mod.check_setup(node, "write")
    assert any("positive definite" in e.lower() for e in bad_report["errors"]), bad_report

    # Raw mode accepts 5-entry transversely isotropic list
    node.setParms({
        "elast_mode1_1": 1,
        "elast_tensor1_1": "[1e5, 2e5, 0.3, 0.25, 4e4]",
    })
    node.updateParmStates()
    assert node.parmTuple("ortho_E1_1").isDisabled()
    assert not node.parm("elast_tensor1_1").isDisabled()
    raw_report = mod.check_setup(node, "write")
    assert not raw_report["errors"], raw_report["errors"]
    assert mod.build_material(node, 1, 1)["elasticity_tensor"] == [
        1e5, 2e5, 0.3, 0.25, 4e4]

    # Import round-trip: 9-number tensor restores orthotropic mode, 5-number restores raw
    mod._import_isotropic(node, 1, 1, {
        "elasticity_tensor": [1.5e5, 2.5e5, 3.5e5, 0.2, 0.2, 0.2, 6e4, 7e4, 8e4],
    })
    assert node.parm("elast_mode1_1").evalAsString() == "orthotropic"
    assert np.allclose(node.parmTuple("ortho_E1_1").eval(), (1.5e5, 2.5e5, 3.5e5))
    assert np.allclose(node.parmTuple("ortho_G1_1").eval(), (6e4, 7e4, 8e4))

    mod._import_isotropic(node, 1, 1, {
        "elasticity_tensor": [1e5, 2e5, 0.3, 0.25, 4e4],
    })
    assert node.parm("elast_mode1_1").evalAsString() == "raw"

    # Reset material to NeoHookean for downstream tests
    node.setParms({"materials1_1": tokens.index("NeoHookean")})
    print("PASS: structured orthotropic inputs, PD validation, and Import round-trip")


def test_gravity_helper(node, mod):
    node.setParms({"units": 0})
    mod.set_gravity_y({"node": node})
    assert node.parmTuple("rhs_vec").eval() == ("0", "9.81", "0")

    node.setParms({"units": 1, "length": "mm", "time": "s"})
    mod.set_gravity_y({"node": node})
    assert node.parmTuple("rhs_vec").eval() == ("0", "9810", "0")

    node.setParms({"units": 1, "length": "cm", "time": "s"})
    mod.set_gravity_y({"node": node})
    assert node.parmTuple("rhs_vec").eval() == ("0", "981", "0")

    node.setParms({"units": 1, "length": "mm", "time": "ms"})
    mod.set_gravity_y({"node": node})
    assert node.parmTuple("rhs_vec").eval() == ("0", "0.00981", "0")

    mod.clear_body_load({"node": node})
    assert node.parmTuple("rhs_vec").eval() == ("0", "0", "0")
    node.setParms({"units": 0})
    print("PASS: unit-aware Gravity (-Y) and Clear Body Load helpers")


def test_extra_params_json(node, mod, work):
    # Invalid JSON caught in Check Setup
    node.parm("extra_params_json").set("{not valid json")
    rep = mod.check_setup(node, "write")
    assert any("Extra params.json" in e for e in rep["errors"]), rep["errors"]

    # Non-object JSON caught in Check Setup
    node.parm("extra_params_json").set("[1, 2, 3]")
    rep2 = mod.check_setup(node, "write")
    assert any("JSON object" in e for e in rep2["errors"]), rep2["errors"]

    # Valid deep-merge JSON overrides nested keys without clobbering siblings
    node.parm("extra_params_json").set(json.dumps({
        "solver": {"advanced": {"stall_restart": {"enabled": True, "min_iters": 4}}},
        "custom_experimental_flag": "batch_a",
    }))
    rep3 = mod.check_setup(node, "write")
    assert not rep3["errors"], rep3["errors"]
    params_path = mod.build_params(node)
    with open(params_path) as handle:
        exported = json.load(handle)
    assert exported["custom_experimental_flag"] == "batch_a"
    assert exported["solver"]["advanced"]["stall_restart"] == {
        "enabled": True, "min_iters": 4}
    # Existing sibling keys inside solver were preserved
    assert "nonlinear" in exported["solver"] and "augmented_lagrangian" in exported["solver"]
    node.parm("extra_params_json").set("")
    print("PASS: Extra params.json deep-merge passthrough and validation")


def test_gmsh_physical_surfaces_to_sidesets(node, mod):
    assert node.evalParm("sideset_selection1_1") == 0
    created = mod.create_gmsh_sidesets({
        "node": node, "script_multiparm_index": "1"})
    assert len(created) == 2, created
    by_tag = {item["tag"]: item for item in created}
    assert set(by_tag) == {10, 20}
    assert by_tag[10]["name"] == "bottom_fixed" and by_tag[10]["faces"] == 2
    assert by_tag[20]["name"] == "top_load" and by_tag[20]["faces"] == 2
    assert node.evalParm("sideset_selection1_1") == 2

    # Idempotent: pressing again adds zero duplicates
    created_again = mod.create_gmsh_sidesets({
        "node": node, "script_multiparm_index": "1"})
    assert created_again == [], created_again
    assert node.evalParm("sideset_selection1_1") == 2

    # Resolved sidesets & exported params.json contain both surface groups
    plan = mod.resolve_sidesets(node, 1, 1)
    assert len(plan["sets"]) == 2
    assert sum(len(s["members"]) for s in plan["sets"]) == 4
    params_path = mod.build_params(node)
    with open(params_path) as handle:
        exported = json.load(handle)
    assert "surface_selection" in exported["geometry"][0]
    print("PASS: Create Sidesets from Gmsh 2D Physical Surfaces (with idempotency)")


def main():
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)
    work = tempfile.mkdtemp(prefix="polyfem_batch_a_")
    try:
        msh_path = os.path.join(work, "cube_surfaces.msh")
        _write_cube_msh_with_surfaces(msh_path)
        node = hou.node("/obj").createNode(TYPE, "batch_a")
        mod = node.hdaModule()
        node.setParms({"working_dir": work + "/", "file_location1": msh_path})
        node.parm("file_location1").pressButton()

        test_time_define_run_by(node, mod, work)
        test_structured_orthotropic(node, mod, work)
        test_gravity_helper(node, mod)
        test_gmsh_physical_surfaces_to_sidesets(node, mod)
        test_extra_params_json(node, mod, work)
        print("ALL BATCH A TESTS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
