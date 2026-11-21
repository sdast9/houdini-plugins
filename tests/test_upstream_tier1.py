"""Tests for PolyFEM upstream merge compatibility + Tier 1 HDA controls.

Covers:
  1. Retired output.data.rest_mesh: not emitted, Check Setup warns when enabled,
     legacy import clears rest_mesh_file and notes retirement.
  2. output.data.diagnostic_state: emitted from state_file, accepted by
     PolyFEM_bin, round-tripped + legacy state import compatibility.
  3. AMIPS power & normalized: omitted at default (1, False), emitted when
     non-default, round-tripped, validated, and solved in PolyFEM_bin.
  4. ParaView options: paraview.async (omitted when True, emitted when False)
     and options.obstacle_volume (omitted when True, emitted when False),
     round-tripped, Expert-noted for paraview_async, accepted by PolyFEM_bin.
  5. Expert solver/contact controls: solver.contact.dhat_epsilon_scale,
     solver.augmented_lagrangian.error, and
     solver.nonlinear.line_search.accept_energy_decrease_in_grad_mode —
     omitted at defaults, emitted when non-default, Check-Setup Expert-noted,
     round-tripped, and accepted by PolyFEM_bin.
  6. Read PVD FIELD_LABELS for energy, energy_avg, and GrowthElasticity fields.

Run: hython tests/test_upstream_tier1.py
"""

import json
import os
import subprocess
import tempfile

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
PVD_TYPE = "readPVD::1.0"
POLYFEM_BIN = os.environ.get(
    "POLYFEM_BIN",
    os.path.abspath(
        os.path.join(BASE, "..", "polyfem", "build", "PolyFEM_bin")))


def _write_unit_cube_msh(path):
    """5-tet unit cube [0,1]^3 in Gmsh MSH 2.2 ASCII."""
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
    ]
    tets = [
        (1, 1, 2, 4, 5), (2, 2, 3, 4, 7),
        (3, 2, 5, 6, 7), (4, 4, 5, 7, 8),
        (5, 2, 4, 5, 7),
    ]
    lines = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$PhysicalNames", "1", '3 1 "cube"', "$EndPhysicalNames",
        "$Nodes", str(len(nodes)),
    ]
    for nid, x, y, z in nodes:
        lines.append(f"{nid} {x} {y} {z}")
    lines += ["$EndNodes", "$Elements", str(len(tets))]
    for eid, n1, n2, n3, n4 in tets:
        lines.append(f"{eid} 4 2 1 1 {n1} {n2} {n3} {n4}")
    lines += ["$EndElements"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _make_scene(work, name, extra_parms=None):
    mesh = os.path.join(work, "cube.msh")
    if not os.path.isfile(mesh):
        _write_unit_cube_msh(mesh)
    node = hou.node("/obj").createNode(TYPE, name)
    mod = node.hdaModule()
    node.setParms({
        "working_dir": work + "/",
        "polyfem_bin": POLYFEM_BIN,
        "file_location1": mesh,
        "quasistatic": 1,
        "end_time_bool": 1, "tend": 0.1,
        "time_inc_bool": 1, "dt": 0.1,
        "use_hdf5": 0,
        "minimal_fields": 0,
    })
    node.parm("file_location1").pressButton()
    node.setParms({
        "materials1_1": mod.MATERIAL_TOKENS.index("NeoHookean"),
        "E1_1": 1e5, "nu1_1": 0.3, "rho1_1": 1000.0,
        "sideset_selection1_1": 2,
        "grouptype1_1_1": 1, "basegroup1_1_1": "0 1 2 3",
        "Boundary_Condition__1_1_1": 1,
        "boundary_type1_1_1_1": 0,
        "grouptype1_1_2": 1, "basegroup1_1_2": "4 5 6 7",
        "Boundary_Condition__1_1_2": 1,
        "boundary_type1_1_2_1": 0,
    })
    node.parmTuple("bc_vec1_1_1_1").set(("0", "0", "0"))
    node.parmTuple("bc_vec1_1_2_1").set(("0", "0", "0.05*t"))
    if extra_parms:
        node.setParms(extra_parms)
    return node, mod


def _import_into(work, name):
    node = hou.node("/obj").createNode(TYPE, name)
    mod = node.hdaModule()
    node.parm("old_input_dir").set(os.path.join(work, "input"))
    mod.read_params({"node": node})
    return node, mod


def _run_polyfem(params_path):
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    res = subprocess.run(
        [POLYFEM_BIN, "-j", params_path, "-o", "../output/", "--log_level", "error"],
        cwd=os.path.dirname(params_path),
        capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, (
        f"PolyFEM_bin failed (rc={res.returncode}):\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def test_retired_rest_mesh_and_diagnostic_state():
    work = tempfile.mkdtemp(prefix="polyfem_tier1_out_")
    node, mod = _make_scene(work, "tier1_out", {
        "state_file": 1,
        "rest_mesh_file": 1,
    })
    setup = mod.check_setup(node, "write")
    assert not setup["errors"], setup
    assert any("rest_mesh" in w for w in setup["warnings"]), setup["warnings"]

    params_path = mod.build_params(node)
    with open(params_path) as f:
        data = json.load(f)
    out_data = data["output"]["data"]
    assert out_data.get("diagnostic_state") == "../output/state.txt", out_data
    assert "state" not in out_data, out_data
    assert "rest_mesh" not in out_data, out_data

    _run_polyfem(params_path)

    # Round-trip diagnostic_state and verify legacy rest_mesh is ignored + warned.
    data["output"]["data"]["rest_mesh"] = "../output/rest_mesh.msh"
    with open(params_path, "w") as f:
        json.dump(data, f)
    imp, _ = _import_into(work, "tier1_out_imp")
    assert imp.evalParm("state_file") == 1
    assert imp.evalParm("rest_mesh_file") == 0
    assert "rest_mesh" in imp.evalParm("import_report")
    print("PASS: diagnostic_state emitted/solved and retired rest_mesh handled")


def test_amips_power_and_normalized():
    work = tempfile.mkdtemp(prefix="polyfem_tier1_amips_")
    node, mod = _make_scene(work, "tier1_amips")
    amips_idx = mod.MATERIAL_TOKENS.index("AMIPS")
    node.setParms({
        "materials1_1": amips_idx,
        "amips_weight1_1": 1e3,
        "amips_use_rest_pose1_1": 1,
    })
    # Defaults omitted
    p0 = mod.build_params(node)
    with open(p0) as f:
        m0 = json.load(f)["materials"][0]
    assert m0["type"] == "AMIPS"
    assert "power" not in m0 and "normalized" not in m0, m0

    # Non-default power & normalized emitted, solved, round-tripped
    node.setParms({"amips_power1_1": 2.0, "amips_normalized1_1": 1})
    p1 = mod.build_params(node)
    with open(p1) as f:
        m1 = json.load(f)["materials"][0]
    assert m1["power"] == 2.0 and m1["normalized"] is True, m1
    _run_polyfem(p1)

    imp, _ = _import_into(work, "tier1_amips_imp")
    assert imp.evalParm("materials1_1") == amips_idx
    assert abs(imp.evalParm("amips_power1_1") - 2.0) < 1e-12
    assert imp.evalParm("amips_normalized1_1") == 1
    print("PASS: AMIPS power and normalized exported, solved, and round-tripped")


def test_paraview_and_expert_controls():
    work = tempfile.mkdtemp(prefix="polyfem_tier1_expert_")
    node, mod = _make_scene(work, "tier1_expert", {
        "enable": 1,
        "barrier_mode": 1,  # adaptive
        "dhat": 1e-3,
    })
    # At defaults, all new optional Tier 1 keys are omitted.
    p0 = mod.build_params(node)
    with open(p0) as f:
        d0 = json.load(f)
    assert "async" not in d0["output"]["paraview"]
    assert "obstacle_volume" not in d0["output"]["paraview"]["options"]
    assert "dhat_epsilon_scale" not in d0["solver"]["contact"]
    assert "error" not in d0["solver"]["augmented_lagrangian"]
    assert "accept_energy_decrease_in_grad_mode" not in d0["solver"]["nonlinear"]["line_search"]

    # Set non-default values while expert_mode == 0 -> Check Setup notes them.
    node.setParms({
        "expert_mode": 0,
        "paraview_async": 0,
        "obstacle_volume": 0,
        "dhat_epsilon_scale": 1e-7,
        "al_error": 1e-3,
        "ls_accept_energy_in_grad_mode": 1,
    })
    setup = mod.check_setup(node, "write")
    assert not setup["errors"], setup
    expert_text = " ".join(setup["notes"])
    assert "Async Output Writer" in expert_text, setup["notes"]
    assert "Adaptive Distance Scale" in expert_text, setup["notes"]
    assert "Accept Energy Decrease in Grad Mode" in expert_text, setup["notes"]
    assert "Error Tolerance" in expert_text, setup["notes"]

    p1 = mod.build_params(node)
    with open(p1) as f:
        d1 = json.load(f)
    assert d1["output"]["paraview"]["async"] is False
    assert d1["output"]["paraview"]["options"]["obstacle_volume"] is False
    assert abs(d1["solver"]["contact"]["dhat_epsilon_scale"] - 1e-7) < 1e-15
    assert abs(d1["solver"]["augmented_lagrangian"]["error"] - 1e-3) < 1e-15
    assert d1["solver"]["nonlinear"]["line_search"]["accept_energy_decrease_in_grad_mode"] is True

    _run_polyfem(p1)

    imp, _ = _import_into(work, "tier1_expert_imp")
    assert imp.evalParm("paraview_async") == 0
    assert imp.evalParm("obstacle_volume") == 0
    assert abs(imp.evalParm("dhat_epsilon_scale") - 1e-7) < 1e-15
    assert abs(imp.evalParm("al_error") - 1e-3) < 1e-15
    assert imp.evalParm("ls_accept_energy_in_grad_mode") == 1
    print("PASS: ParaView async/obstacle_volume and Tier 1 expert controls solved & round-tripped")


def test_readpvd_new_field_labels():
    pvd = hou.node("/obj").createNode(PVD_TYPE, "tier1_readpvd")
    rm = pvd.hdaModule()
    for field in ("energy", "energy_avg", "growth", "growth_gradient",
                  "growth_reaction", "growth_multiplier", "growth_pinned",
                  "growth_activation_step"):
        assert field in rm.FIELD_LABELS, f"missing FIELD_LABELS[{field!r}]"
        assert rm._field_label(field) == rm.FIELD_LABELS[field]
    print("PASS: Read PVD curated labels present for energy and growth fields")


def main():
    hou.hda.installFile(os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"))
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))

    test_retired_rest_mesh_and_diagnostic_state()
    test_amips_power_and_normalized()
    test_paraview_and_expert_controls()
    test_readpvd_new_field_labels()
    print("ALL UPSTREAM TIER 1 TESTS PASSED")


if __name__ == "__main__":
    main()
