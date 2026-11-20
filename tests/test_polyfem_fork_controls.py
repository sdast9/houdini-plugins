"""Regression & end-to-end tests for Selective PolyFEM Fork Controls (2026-11-20):
  1. GrowthNeoHookean (standalone + MaterialSum + SPD validation + import + real solve)
  2. InversionBarrier (standalone + MaterialSum Add Inversion Barrier + validation + import + real solve)
  3. AMIPS (standalone + MaterialSum + validation + import + real solve)
  4. Nodal Force / nodal_neumann_boundary (Points sideset, Faces refusal, time curve, import, real solve)
  5. Solver & Contact Robustness controls (Expert Mode):
     - AL Phase Budget (RB-07): solver.augmented_lagrangian.budget
     - Generalized Stall Restart (D8) + line_search_failure_restarts: solver.advanced.*
     - Semi-Implicit Controller Options: clamped_contacts, gradient_balance_dofs
     - Omitted when at defaults; exported, Check-Setup-noted, round-tripped, and accepted by PolyFEM_bin when set.

Run via hython:
  hython tests/test_polyfem_fork_controls.py
"""
import json
import os
import shutil
import subprocess
import tempfile

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
POLYFEM_BIN = os.path.abspath(
    os.path.join(BASE, "..", "polyfem", "build", "PolyFEM_bin"))


def _write_unit_cube_msh(path):
    """5-tetrahedron unit cube [0,1]^3 in Gmsh MSH 2.2 ASCII with positive volume."""
    nodes = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    # Standard 5-tet decomposition of [0,1]^3 with positive signed volume
    tets = [
        (1, 2, 4, 5),
        (2, 3, 4, 7),
        (2, 5, 6, 7),
        (4, 5, 7, 8),
        (2, 4, 5, 7),
    ]
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$Nodes",
        str(len(nodes)),
    ]
    for nid, x, y, z in nodes:
        lines.append(f"{nid} {x} {y} {z}")
    lines.extend(["$EndNodes", "$Elements", str(len(tets))])
    for eid, (a, b, c, d) in enumerate(tets, 1):
        lines.append(f"{eid} 4 2 1 1 {a} {b} {c} {d}")
    lines.append("$EndElements")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _new_cube_node(name, msh_path, workdir):
    node = hou.node("/obj").createNode(TYPE, name)
    mod = node.hdaModule()
    node.setParms({
        "working_dir": workdir,
        "polyfem_bin": POLYFEM_BIN,
        "quasistatic": 1,
        "define_run_by": 1,  # End Time + # of Timesteps
        "tend": 1.0,
        "num_timesteps": 1,
        "enable": 0,
        "use_hdf5": 0,
        "num_geos": 1,
    })
    node.setParms({"file_location1": msh_path})
    mod.create_geo_nodes({"node": node, "script_multiparm_index": "1"})
    # Fix bottom face (y = 0)
    node.setParms({
        "sideset_selection1_1": 1,
        "basegroup1_1_1": "axis:-y:0.01",
        "grouptype1_1_1": 0,
        "Boundary_Condition__1_1_1": 1,
        "boundary_type1_1_1_1": 0,
    })
    node.parmTuple("bc_vec1_1_1_1").set(("0", "0", "0"))
    return node


def _import_into(node, params_path, workdir):
    os.makedirs(workdir, exist_ok=True)
    node.setParms({
        "working_dir": workdir,
        "old_input_dir": os.path.dirname(params_path),
    })
    node.hdaModule().read_params({"node": node})


def _run_polyfem(params_path, workdir):
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    res = subprocess.run(
        [POLYFEM_BIN, "-j", params_path, "--ns"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, (
        f"PolyFEM_bin failed (rc={res.returncode}):\n"
        f"STDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    return res


def test_growth_neohookean(msh_path, tmp):
    workdir = os.path.join(tmp, "growth")
    os.makedirs(workdir, exist_ok=True)
    node = _new_cube_node("fork_growth", msh_path, workdir)
    mod = node.hdaModule()

    # Select GrowthNeoHookean standalone
    node.parm("materials1_1").set("GrowthNeoHookean")
    node.setParms({"E1_1": 1e5, "nu1_1": 0.3})
    node.parmTuple("growth_diag1_1").set(("1 + 0.05*t", "1", "1"))
    node.parmTuple("growth_offdiag1_1").set(("0.01*t", "0", "0"))

    report = mod.check_setup(node, "write")
    assert not report["errors"], report

    params_path = mod.write_params_only({"node": node})
    data = json.load(open(params_path))
    mat = data["materials"][0]
    assert mat["type"] == "GrowthNeoHookean", mat
    assert mat["E"] == 1e5 and mat["nu"] == 0.3, mat
    assert mat["g11"] == "1 + 0.05*t" and mat["g22"] == 1 and mat["g33"] == 1, mat
    assert mat["g12"] == "0.01*t" and mat["g13"] == 0 and mat["g23"] == 0, mat

    # Real PolyFEM_bin solve!
    _run_polyfem(params_path, workdir)

    # Round-trip import into a fresh node
    imported = hou.node("/obj").createNode(TYPE, "fork_growth_imp")
    _import_into(imported, params_path, os.path.join(tmp, "growth_imp"))
    assert imported.parm("materials1_1").evalAsString() == "GrowthNeoHookean"
    assert list(imported.parmTuple("growth_diag1_1").eval()) == ["1 + 0.05*t", "1", "1"]
    assert list(imported.parmTuple("growth_offdiag1_1").eval()) == ["0.01*t", "0", "0"]
    imported.destroy()

    # Non-SPD numeric G must be rejected by Check Setup and export
    node.parmTuple("growth_diag1_1").set(("1", "1", "1"))
    node.parmTuple("growth_offdiag1_1").set(("2", "0", "0"))
    bad_report = mod.check_setup(node, "write")
    assert any("positive definite" in e for e in bad_report["errors"]), bad_report

    # MaterialSum with GrowthNeoHookean as matrix_model
    node.parm("materials1_1").set("MaterialSum")
    node.parm("matrix_model1_1").set("GrowthNeoHookean")
    node.parmTuple("growth_diag1_1").set(("1.1", "1.0", "1.0"))
    node.parmTuple("growth_offdiag1_1").set(("0", "0", "0"))
    comp_path = mod.write_params_only({"node": node})
    comp_data = json.load(open(comp_path))
    assert comp_data["materials"][0]["type"] == "MaterialSum"
    assert comp_data["materials"][0]["models"][0]["type"] == "GrowthNeoHookean"
    assert comp_data["materials"][0]["models"][0]["g11"] == 1.1

    node.destroy()
    print("PASS: GrowthNeoHookean (standalone, MaterialSum, SPD validation, import, PolyFEM_bin solve)")


def test_inversion_barrier(msh_path, tmp):
    workdir = os.path.join(tmp, "inv_barrier")
    os.makedirs(workdir, exist_ok=True)
    node = _new_cube_node("fork_inv", msh_path, workdir)
    mod = node.hdaModule()

    # Standalone InversionBarrier
    node.parm("materials1_1").set("InversionBarrier")
    node.setParms({"E1_1": 5e4, "nu1_1": 0.4, "inv_barrier_thresh1_1": 0.35})
    params_path = mod.write_params_only({"node": node})
    data = json.load(open(params_path))
    mat = data["materials"][0]
    assert mat["type"] == "InversionBarrier", mat
    assert mat["E"] == 5e4 and mat["nu"] == 0.4 and abs(mat["JBarrierThreshold"] - 0.35) < 1e-9, mat

    # Round-trip standalone
    imp = hou.node("/obj").createNode(TYPE, "fork_inv_imp")
    _import_into(imp, params_path, os.path.join(tmp, "inv_imp"))
    assert imp.parm("materials1_1").evalAsString() == "InversionBarrier"
    assert abs(imp.evalParm("inv_barrier_thresh1_1") - 0.35) < 1e-6
    imp.destroy()

    # UI hard lower bound clamps <= 0 threshold to positive 1e-6; invalid nu caught by Check Setup
    node.parm("inv_barrier_thresh1_1").set(0.0)
    assert node.evalParm("inv_barrier_thresh1_1") >= 1e-6
    node.parm("nu1_1").set(0.5)
    bad = mod.check_setup(node, "write")
    assert any("Poisson" in e for e in bad["errors"]), bad
    node.parm("nu1_1").set(0.4)

    # MaterialSum: NeoHookean matrix + Add Inversion Barrier -> real PolyFEM_bin solve!
    node.parm("materials1_1").set("MaterialSum")
    node.parm("matrix_model1_1").set("NeoHookean")
    node.setParms({
        "E1_1": 1e5,
        "nu1_1": 0.3,
        "num_fiber_families1_1": 0,
        "add_inv_barrier1_1": 1,
        "comp_inv_E1_1": 2e4,
        "comp_inv_nu1_1": 0.35,
        "comp_inv_thresh1_1": 0.4,
    })
    # Add small top traction so the solve does real work
    node.setParms({
        "sideset_selection1_1": 2,
        "basegroup1_1_2": "axis:+y:0.99",
        "grouptype1_1_2": 0,
        "Boundary_Condition__1_1_2": 1,
        "boundary_type1_1_2_1": 1,
    })
    node.parmTuple("bc_vec1_1_2_1").set(("0", "-500", "0"))
    comp_path = mod.write_params_only({"node": node})
    comp_data = json.load(open(comp_path))
    models = comp_data["materials"][0]["models"]
    assert [m["type"] for m in models] == ["NeoHookean", "InversionBarrier"], models
    assert abs(models[1]["JBarrierThreshold"] - 0.4) < 1e-9

    _run_polyfem(comp_path, workdir)

    # Round-trip composite MaterialSum with InversionBarrier
    imp2 = hou.node("/obj").createNode(TYPE, "fork_inv_comp_imp")
    _import_into(imp2, comp_path, os.path.join(tmp, "inv_comp_imp"))
    assert imp2.parm("materials1_1").evalAsString() == "MaterialSum"
    assert imp2.parm("matrix_model1_1").evalAsString() == "NeoHookean"
    assert imp2.evalParm("add_inv_barrier1_1") == 1
    assert abs(imp2.evalParm("comp_inv_E1_1") - 2e4) < 1e-6
    assert abs(imp2.evalParm("comp_inv_thresh1_1") - 0.4) < 1e-6
    assert imp2.evalParm("extra_models_json1_1").strip() == ""
    imp2.destroy()

    node.destroy()
    print("PASS: InversionBarrier (standalone, MaterialSum, validation, import, PolyFEM_bin solve)")


def test_amips(msh_path, tmp):
    workdir = os.path.join(tmp, "amips")
    os.makedirs(workdir, exist_ok=True)
    node = _new_cube_node("fork_amips", msh_path, workdir)
    mod = node.hdaModule()

    node.parm("materials1_1").set("AMIPS")
    node.setParms({"amips_weight1_1": 2.5, "amips_use_rest_pose1_1": 1})
    node.setParms({
        "sideset_selection1_1": 2,
        "basegroup1_1_2": "axis:+y:0.99",
        "grouptype1_1_2": 0,
        "Boundary_Condition__1_1_2": 1,
        "boundary_type1_1_2_1": 0,
    })
    node.parmTuple("bc_vec1_1_2_1").set(("0", "0.05", "0"))

    params_path = mod.write_params_only({"node": node})
    data = json.load(open(params_path))
    mat = data["materials"][0]
    assert mat["type"] == "AMIPS", mat
    assert abs(mat["weight"] - 2.5) < 1e-9 and mat["use_rest_pose"] is True, mat

    _run_polyfem(params_path, workdir)

    imp = hou.node("/obj").createNode(TYPE, "fork_amips_imp")
    _import_into(imp, params_path, os.path.join(tmp, "amips_imp"))
    assert imp.parm("materials1_1").evalAsString() == "AMIPS"
    assert abs(imp.evalParm("amips_weight1_1") - 2.5) < 1e-6
    assert imp.evalParm("amips_use_rest_pose1_1") == 1
    imp.destroy()

    # UI hard lower bound clamps <= 0 weight to positive 1e-9
    node.parm("amips_weight1_1").set(0.0)
    assert node.evalParm("amips_weight1_1") >= 1e-9

    # Composite with AMIPS matrix
    node.parm("amips_weight1_1").set(1.5)
    node.parm("materials1_1").set("MaterialSum")
    node.parm("matrix_model1_1").set("AMIPS")
    comp_path = mod.write_params_only({"node": node})
    comp_data = json.load(open(comp_path))
    assert comp_data["materials"][0]["models"][0]["type"] == "AMIPS"
    assert abs(comp_data["materials"][0]["models"][0]["weight"] - 1.5) < 1e-9

    node.destroy()
    print("PASS: AMIPS (standalone, MaterialSum, validation, import, PolyFEM_bin solve)")


def test_nodal_neumann(msh_path, tmp):
    workdir = os.path.join(tmp, "nodal_neumann")
    os.makedirs(workdir, exist_ok=True)
    node = _new_cube_node("fork_nodal", msh_path, workdir)
    mod = node.hdaModule()

    # Add a Points sideset on top nodes with Nodal Force (Nodal Neumann)
    node.setParms({
        "sideset_selection1_1": 2,
        "basegroup1_1_2": "axis:+y:0.99",
        "grouptype1_1_2": 1,  # Points
        "Boundary_Condition__1_1_2": 1,
        "boundary_type1_1_2_1": 5,  # Nodal_Neumann
        "time_curve1_1_2_1": 1,     # Ramp
    })
    node.parmTuple("bc_vec1_1_2_1").set(("0", "-100", "0"))
    node.parmTuple("rhs_vec").set(("0", "0", "0"))

    # force_field_names should include body_forces
    assert "body_forces" in mod.force_field_names(node), mod.force_field_names(node)

    # Check Setup passes and exports nodal_neumann_boundary + point_selection
    report = mod.check_setup(node, "write")
    assert not report["errors"], report
    params_path = mod.write_params_only({"node": node})
    data = json.load(open(params_path))
    nnb = data["boundary_conditions"].get("nodal_neumann_boundary")
    assert nnb and len(nnb) == 1, data["boundary_conditions"]
    assert nnb[0]["value"] == [0, -100, 0], nnb[0]
    assert "interpolation" in nnb[0], nnb[0]
    assert data["geometry"][0].get("point_selection"), data["geometry"][0]

    # Real PolyFEM_bin solve!
    _run_polyfem(params_path, workdir)

    # Round-trip via hda_sidesets.json
    imp = hou.node("/obj").createNode(TYPE, "fork_nodal_imp")
    _import_into(imp, params_path, os.path.join(tmp, "nodal_imp"))
    assert imp.evalParm("grouptype1_1_2") == 1
    assert imp.evalParm("boundary_type1_1_2_1") == 5
    assert list(imp.parmTuple("bc_vec1_1_2_1").eval()) == ["0", "-100", "0"]
    imp.destroy()

    # Round-trip from bare params.json (without hda_sidesets.json)
    sidesets_rec = os.path.join(os.path.dirname(params_path), mod.SIDESET_RECORD_FILE)
    if os.path.isfile(sidesets_rec):
        os.remove(sidesets_rec)
    imp_bare = hou.node("/obj").createNode(TYPE, "fork_nodal_bare")
    _import_into(imp_bare, params_path, os.path.join(tmp, "nodal_bare_imp"))
    assert imp_bare.evalParm("grouptype1_1_2") == 1
    assert imp_bare.evalParm("boundary_type1_1_2_1") == 5
    assert list(imp_bare.parmTuple("bc_vec1_1_2_1").eval()) == ["0", "-100", "0"]
    imp_bare.destroy()

    # Nodal Force on Primitives (Faces) sideset must be refused cleanly
    node.parm("grouptype1_1_2").set(0)
    bad_face = mod.check_setup(node, "write")
    assert any("Points" in e for e in bad_face["errors"]), bad_face

    # Face traction (Neumann) on Points sideset must still be refused
    node.setParms({"grouptype1_1_2": 1, "boundary_type1_1_2_1": 1})
    bad_pts = mod.check_setup(node, "write")
    assert any("faces only" in e for e in bad_pts["errors"]), bad_pts

    node.destroy()
    print("PASS: Nodal Force (nodal_neumann_boundary: export, refusals, curves, import, PolyFEM_bin solve)")


def test_solver_and_contact_robustness(msh_path, tmp):
    workdir = os.path.join(tmp, "robustness")
    os.makedirs(workdir, exist_ok=True)
    node = _new_cube_node("fork_robust", msh_path, workdir)
    mod = node.hdaModule()

    # 1. Untouched defaults must NOT emit optional blocks
    default_dir = os.path.join(tmp, "robustness_default")
    os.makedirs(default_dir, exist_ok=True)
    node.setParms({"working_dir": default_dir, "enable": 1, "barrier_mode": 0})
    default_path = mod.write_params_only({"node": node})
    default_data = json.load(open(default_path))
    assert "budget" not in default_data["solver"]["augmented_lagrangian"], default_data["solver"]
    assert "stall_restart" not in default_data["solver"]["advanced"], default_data["solver"]
    assert "line_search_failure_restarts" not in default_data["solver"]["advanced"], default_data["solver"]
    semi_default = default_data["solver"]["contact"]["semi_implicit"]
    assert "clamped_contacts" not in semi_default, semi_default
    assert "gradient_balance_dofs" not in semi_default, semi_default

    # 2. Set non-default values for all three robustness groups
    node.setParms({"working_dir": workdir})
    node.setParms({
        "expert_mode": 0,
        # AL Phase Budget (RB-07)
        "al_max_passes": 12,
        "al_stagnation_window": 3,
        "al_progress_tol": 0.02,
        "al_snap_tol": 0.005,
        "al_drift_tol": 0.03,
        # Generalized Stall Restart (D8) + Line-Search Recovery
        "adv_stall_restart_enabled": 1,
        "adv_stall_alpha_threshold": 0.005,
        "adv_stall_patience": 4,
        "adv_stall_min_iterations": 3,
        "adv_stall_soft_limit": 50,
        "adv_stall_max_restarts": 8,
        "adv_stall_alpha_basis": 1,   # absolute
        "adv_stall_feasible_ratio": 0.95,
        "adv_stall_remedy": 1,        # none
        "ls_failure_restarts": 2,
        # Semi-Implicit Controller Options
        "si_clamped_contacts": 2,     # exclude_collisions
        "si_gradient_balance_dofs": 1,  # free
    })

    # Check Setup surfaces hidden non-default expert controls when expert_mode == 0
    report = mod.check_setup(node, "write")
    assert not report["errors"], report
    notes_text = " ".join(report["notes"])
    assert "Max AL Passes" in notes_text, report["notes"]
    assert "Stall Patience" in notes_text or "Line-Search Failure Restarts" in notes_text, report["notes"]
    assert "Clamped Contacts Handling" in notes_text, report["notes"]

    # Export and verify exact JSON structure
    custom_path = mod.write_params_only({"node": node})
    data = json.load(open(custom_path))
    al_budget = data["solver"]["augmented_lagrangian"]["budget"]
    assert al_budget == {
        "max_passes": 12,
        "stagnation_window": 3,
        "progress_tolerance": 0.02,
        "snap_tolerance": 0.005,
        "drift_tolerance": 0.03,
    }, al_budget

    adv = data["solver"]["advanced"]
    assert adv["line_search_failure_restarts"] == 2, adv
    assert adv["stall_restart"] == {
        "enabled": True,
        "alpha_threshold": 0.005,
        "patience": 4,
        "min_iterations": 3,
        "soft_iteration_limit": 50,
        "max_restarts": 8,
        "alpha_basis": "absolute",
        "feasible_ratio_threshold": 0.95,
        "remedy": "none",
    }, adv["stall_restart"]

    semi = data["solver"]["contact"]["semi_implicit"]
    assert semi["clamped_contacts"] == "exclude_collisions", semi
    assert semi["gradient_balance_dofs"] == "free", semi

    # Real PolyFEM_bin solve accepts every single new JSON key!
    _run_polyfem(custom_path, workdir)

    # Round-trip import restores every control
    imp = hou.node("/obj").createNode(TYPE, "fork_robust_imp")
    _import_into(imp, custom_path, os.path.join(tmp, "robust_imp"))
    assert imp.evalParm("al_max_passes") == 12
    assert imp.evalParm("al_stagnation_window") == 3
    assert abs(imp.evalParm("al_progress_tol") - 0.02) < 1e-6
    assert abs(imp.evalParm("al_snap_tol") - 0.005) < 1e-6
    assert abs(imp.evalParm("al_drift_tol") - 0.03) < 1e-6
    assert abs(imp.evalParm("adv_stall_alpha_threshold") - 0.005) < 1e-6
    assert imp.evalParm("adv_stall_patience") == 4
    assert imp.evalParm("adv_stall_min_iterations") == 3
    assert imp.evalParm("adv_stall_soft_limit") == 50
    assert imp.evalParm("adv_stall_max_restarts") == 8
    assert imp.parm("adv_stall_alpha_basis").evalAsString() == "absolute"
    assert abs(imp.evalParm("adv_stall_feasible_ratio") - 0.95) < 1e-6
    assert imp.parm("adv_stall_remedy").evalAsString() == "none"
    assert imp.evalParm("ls_failure_restarts") == 2
    assert imp.parm("si_clamped_contacts").evalAsString() == "exclude_collisions"
    assert imp.parm("si_gradient_balance_dofs").evalAsString() == "free"

    # Re-importing the default JSON cleanly resets all of them back to defaults
    _import_into(imp, default_path, os.path.join(tmp, "robust_def_imp"))
    assert imp.evalParm("al_max_passes") == 0
    assert imp.evalParm("al_stagnation_window") == 0
    assert imp.evalParm("adv_stall_patience") == 5
    assert imp.evalParm("ls_failure_restarts") == 1
    assert imp.parm("si_clamped_contacts").evalAsString() == "exclude_statistics"
    assert imp.parm("si_gradient_balance_dofs").evalAsString() == "all"

    imp.destroy()
    node.destroy()
    print("PASS: Solver & Contact Robustness controls (AL budget, D8 stall restart, SI controller, import, PolyFEM_bin solve)")


def main():
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"),
        force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)

    tmp = tempfile.mkdtemp(prefix="test_fork_controls_")
    try:
        msh_path = os.path.join(tmp, "cube.msh")
        _write_unit_cube_msh(msh_path)
        test_growth_neohookean(msh_path, tmp)
        test_inversion_barrier(msh_path, tmp)
        test_amips(msh_path, tmp)
        test_nodal_neumann(msh_path, tmp)
        test_solver_and_contact_robustness(msh_path, tmp)
        print("ALL POLYFEM FORK CONTROLS TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
