"""End-to-end headless test of the PolyFEM 2.0 HDA.

Builds a contact scene programmatically, writes params.json through the HDA,
then runs the actual PolyFEM binary (strict json validation) on it.

Run: hython tests/test_polyfem_hda.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")


def make_cube_msh(path):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cube")
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(path)
    gmsh.finalize()


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_hda_test_")
    os.makedirs(os.path.join(work, "input"), exist_ok=True)
    cube = os.path.join(work, "cube.msh")
    make_cube_msh(cube)
    slab = os.path.join(work, "slab.obj")
    shutil.copy(os.path.join(
        ROOT, "polyfem", "scenes", "semi-implicit", "slab.obj"), slab)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_test")
    mod = node.hdaModule()
    assert node.parm("si_constraint_floor") is None, "retired control still exposed"
    node.addSpareParmTuple(hou.FloatParmTemplate(
        "si_constraint_floor", "Legacy saved floor", 1, default_value=(1e-4,)))
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN})

    # --- geometry 1: simulated cube -------------------------------------
    node.setParms({"file_location1": cube})
    node.parm("file_location1").pressButton()
    assert node.node("geo_1") is not None, "geo chain not built"
    assert node.node("surface_1") is not None, "surface chain not built"
    assert node.evalParm("num_volumes1") == 1

    # NeoHookean material
    node.setParms({"materials1_1": 3, "E1_1": 1e6, "nu1_1": 0.45,
                   "rho1_1": 1000})

    # sideset 1: native axis selection on the top face -> Dirichlet push-down
    node.setParms({"sideset_selection1_1": 1})
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({"basegroup1_1_1": "axis:+z:0.99", "grouptype1_1_1": 0,
                   "Boundary_Condition__1_1_1": 1})
    node.setParms({"boundary_type1_1_1_1": 0,
                   "vector_1_1_1_1": '["0", "0", "-0.25*t"]',
                   "x_dimension1_1_1_1": 1, "y_dimension1_1_1_1": 1,
                   "z_dimension1_1_1_1": 1})

    # sideset 2: hand-picked pattern (all faces) -> file-based export path,
    # zero Neumann (no physical effect, exercises the txt pipeline)
    node.setParms({"sideset_selection1_1": 2})
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({"basegroup1_1_2": "*", "grouptype1_1_2": 0,
                   "Boundary_Condition__1_1_2": 1})
    node.setParms({"boundary_type1_1_2_1": 1,
                   "vector_1_1_2_1": "[0, 0, 0]"})

    # --- geometry 2: obstacle slab ---------------------------------------
    node.setParms({"geo_int": 2, "num_geos": 2})
    node.setParms({"is_obstacle2": 1, "file_location2": slab})
    node.parm("file_location2").pressButton()
    node.setParms({"obstacle_disp2": "[0, 0, 0]"})

    # --- global settings --------------------------------------------------
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 0.5,
                   "time_inc_bool": 1, "dt": 0.125})
    node.setParms({"enable": 1, "dhat": 1e-3})
    # barrier_mode default 0 (semi-implicit); al_hessian_scaled default 1

    # --- write + validate ---------------------------------------------------
    params_path = mod.write_params_only({"node": node})
    assert params_path and os.path.isfile(params_path), "params.json missing"

    import json
    with open(params_path) as f:
        data = json.load(f)
    contact = data["solver"]["contact"]
    assert contact["barrier_stiffness"] == "semi_implicit", contact
    assert "semi_implicit" in contact
    assert "constraint_floor" not in contact["semi_implicit"]
    assert "adaptive_barrier_stiffness_multiplier" not in contact
    # GCP-only settings live on the GCP tab and are exported only with GCP
    gcp_keys = {"use_adaptive_dhat", "min_distance_ratio", "alpha_n", "alpha_t"}
    assert not (gcp_keys & set(data["contact"])), data["contact"]
    ptg = node.parmTemplateGroup()
    for name in ("adapt_dhat", "min_dist_ratio"):
        folder = ptg.containingFolder(name)
        assert folder.label() == "Geometric Contact (GCP)", (name, folder.label())
    al = data["solver"]["augmented_lagrangian"]
    assert al["initial_weight"] == "hessian_scaled", al
    # minimal-fields mode (default): whitelist the derivation basis, drop
    # everything readPVD derives (von Mises, PK stresses, *_avg variants)
    pv = data["output"]["paraview"]
    assert {"solution", "F", "cauchy_stess"} <= set(pv["fields"]), pv["fields"]
    assert not any(f.startswith(("von_mises", "pk1", "pk2"))
                   or f.endswith("_avg") for f in pv["fields"]), pv["fields"]
    assert pv["options"]["scalar_values"] is False
    assert pv["options"]["tensor_values"] is True
    assert data["space"]["remesh"] == {"enabled": False}
    assert data["materials"][0]["id"] == 1001
    geo1 = data["geometry"][0]
    sel = geo1["surface_selection"]
    assert any(isinstance(s, dict) and s.get("axis") for s in sel), sel
    assert any(isinstance(s, dict) and str(s.get("file", "")).endswith("_tri.txt")
               for s in sel), sel
    assert data["geometry"][1]["surface_selection"] == 102000  # obstacle_id(2)

    input_dir = os.path.dirname(params_path)
    tri_files = [s["file"] for s in sel if isinstance(s, dict) and "file" in s]
    for f_name in tri_files + [geo1["volume_selection"]]:
        assert os.path.isfile(os.path.join(input_dir, f_name)), f_name

    # provenance (RB-12): the asset identifies itself in the export
    prov = data["provenance"]
    assert prov["producer"] == "houdini", prov
    assert prov["producer_version"] == hou.applicationVersionString(), prov
    assert prov["asset"].startswith("stevenabramowitch::dev::PolyFEM::2.0"), prov
    assert len(prov["asset_sha256"]) == 64, prov
    assert prov["exported_at"].endswith("Z"), prov

    # --- run the real binary (strict validation is the schema test) -------
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info"],
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    sys.stdout.write(result.stdout[-2000:])
    sys.stderr.write(result.stderr[-2000:])
    assert result.returncode == 0, "PolyFEM run failed"

    out_dir = os.path.join(work, "output")
    pvds = [f for f in os.listdir(out_dir) if f.endswith(".pvd")]
    assert pvds, f"no .pvd written in {out_dir}"

    # ... and the solver's run manifest carries it back as the producer
    with open(os.path.join(out_dir, "run-manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["completion"]["status"] == "completed", manifest["completion"]
    assert manifest["producer"] == prov, (manifest["producer"], prov)
    assert manifest["input"]["file"]["sha256"], manifest["input"]["file"]

    # the Dirichlet push must actually move the cube (guards selection order)
    import re as _re
    linf = [float(m.group(1)) for m in _re.finditer(
        r"-- Linf error: ([0-9.e+-]+)", result.stdout)]
    assert linf and linf[-1] > 0.05, f"no displacement; Linf={linf}"
    # Import a legacy positive floor; it must be dropped on re-export.
    data["solver"]["contact"]["semi_implicit"]["constraint_floor"] = 1e-4
    with open(params_path, "w") as f:
        json.dump(data, f)
    # --- round-trip: import the params back into a fresh node -------------
    node2 = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_roundtrip")
    node2.setParms({"old_input_dir": input_dir})
    node2.hdaModule().read_params({"node": node2})
    assert node2.evalParm("num_geos") == 2
    assert node2.evalParm("num_volumes1") == 1
    assert node2.evalParm("sideset_selection1_1") == 2, \
        node2.evalParm("sideset_selection1_1")
    assert node2.evalParm("basegroup1_1_1") == "axis:+z:0.99", \
        node2.evalParm("basegroup1_1_1")
    assert node2.evalParm("Boundary_Condition__1_1_1") == 1
    assert node2.evalParm("boundary_type1_1_1_1") == 0
    assert "-0.25*t" in node2.evalParm("vector_1_1_1_1"), \
        node2.evalParm("vector_1_1_1_1")
    assert node2.evalParm("z_dimension1_1_1_1") == 1
    assert node2.evalParm("boundary_type1_1_2_1") == 1  # Neumann
    # the file-based sideset selected every face; matching through the
    # provenance attributes must recover all of them
    surf = node2.node("null_1").geometry()
    restored = node2.hdaModule().expand_group_str(
        node2.evalParm("basegroup1_1_2"))
    assert len(restored) == surf.intrinsicValue("primitivecount"), \
        (len(restored), surf.intrinsicValue("primitivecount"))
    assert node2.evalParm("is_obstacle2") == 1
    assert node2.evalParm("obstacle_disp2").replace(" ", "") == "[0,0,0]", \
        node2.evalParm("obstacle_disp2")
    assert node2.parm("si_constraint_floor") is None
    roundtrip_path = node2.hdaModule().write_params_only({"node": node2})
    with open(roundtrip_path) as f:
        roundtrip_data = json.load(f)
    assert "constraint_floor" not in roundtrip_data["solver"]["contact"]["semi_implicit"]
    print("PASS: params.json round-trip import; retired floor dropped")

    # --- RB-05 resource limits: automatic by default, off / custom round-trip
    limits = data["solver"]["contact"]["CCD"]["resource_limits"]
    assert limits == {"max_cell_items": -1, "max_candidate_emissions": -1}, limits
    assert node2.parm("resource_limits").evalAsString() == "automatic"
    for parm_name in ("resource_limits", "resource_max_cell_items",
                      "resource_max_candidate_emissions", "broad_phase"):
        assert node.parm(parm_name).parmTemplate().help(), f"no tooltip on {parm_name}"
    assert "exit status 3" in node.parm("resource_limits").parmTemplate().help()
    node2.parm("resource_limits").set("custom")
    node2.setParms({"resource_max_cell_items": 12345,
                    "resource_max_candidate_emissions": 0})
    custom_path = node2.hdaModule().write_params_only({"node": node2})
    with open(custom_path) as f:
        custom = json.load(f)["solver"]["contact"]["CCD"]["resource_limits"]
    assert custom == {"max_cell_items": 12345, "max_candidate_emissions": 0}, custom
    node3 = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_limits_roundtrip")
    node3.setParms({"old_input_dir": os.path.dirname(custom_path)})
    node3.hdaModule().read_params({"node": node3})
    assert node3.parm("resource_limits").evalAsString() == "custom"
    assert node3.evalParm("resource_max_cell_items") == 12345
    assert node3.evalParm("resource_max_candidate_emissions") == 0
    node3.parm("resource_limits").set("off")
    off_path = node3.hdaModule().write_params_only({"node": node3})
    with open(off_path) as f:
        off = json.load(f)["solver"]["contact"]["CCD"]["resource_limits"]
    assert off == {"max_cell_items": 0, "max_candidate_emissions": 0}, off
    node3.setParms({"old_input_dir": os.path.dirname(off_path)})
    node3.hdaModule().read_params({"node": node3})
    assert node3.parm("resource_limits").evalAsString() == "off"
    node3.destroy()
    print("PASS: resource limits automatic by default; custom/off round-trip")

    # --- RB-10 friction defaults: budget 2, realized-force lag; round-trip --
    semi = data["solver"]["contact"]["semi_implicit"]
    assert semi["friction_lag"] == "realized_force", semi
    assert data["solver"]["contact"]["friction_iterations"] == 2, data["solver"]["contact"]
    for parm_name in ("si_friction_lag", "friction_iterations"):
        assert node.parm(parm_name).parmTemplate().help(), f"no tooltip on {parm_name}"
    assert "trim" in node.parm("si_friction_lag").parmTemplate().help()
    node2.parm("si_friction_lag").set("follow_stiffness")
    node2.setParms({"friction_iterations": 1})
    lag_path = node2.hdaModule().write_params_only({"node": node2})
    with open(lag_path) as f:
        lag_contact = json.load(f)["solver"]["contact"]
    assert lag_contact["semi_implicit"]["friction_lag"] == "follow_stiffness", lag_contact
    assert lag_contact["friction_iterations"] == 1, lag_contact
    node4 = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_friction_lag_roundtrip")
    node4.setParms({"old_input_dir": os.path.dirname(lag_path)})
    node4.hdaModule().read_params({"node": node4})
    assert node4.parm("si_friction_lag").evalAsString() == "follow_stiffness"
    assert node4.evalParm("friction_iterations") == 1
    node4.destroy()
    node2.parm("si_friction_lag").set("realized_force")
    node2.setParms({"friction_iterations": 2})
    print("PASS: friction lag realized-force / budget 2 by default; follow/1 round-trip")

    # --- solver panel: nothing wired may be hidden --------------------------
    def walk(templates, path):
        for t in templates:
            if isinstance(t, hou.FolderParmTemplate):
                assert not t.isHidden(), f"hidden tab {path}/{t.label()}"
                walk(t.parmTemplates(), path + "/" + t.label())
            elif not isinstance(t, hou.LabelParmTemplate):
                assert not t.isHidden(), f"hidden control {path}/{t.name()}"
    solver_folder = node.parmTemplateGroup().findFolder("Solver")
    assert solver_folder is not None
    walk(solver_folder.parmTemplates(), "Solver")
    print("PASS: no hidden control or tab in the Solver folder")

    # --- Hypre + the new convergence controls: export, validate, run ------
    node.setParms({"tend": 0.25})
    node.parm("solver").set("Hypre")
    node.setParms({"nodal_coarsening_hypre": 1, "theta_hypre": 0.4,
                   "rel_grad_norm_tol": 1e-9, "rel_x_delta_tol": 1e-12,
                   "allow_non_grad_convergence": 1,
                   "newton_decrement_tol": 1e-14})
    node.parm("norm_type").set("Linf")
    node.parm("al_lumping").set("hrz")
    params_path = mod.write_params_only({"node": node})
    with open(params_path) as f:
        data = json.load(f)
    linear = data["solver"]["linear"]
    assert linear["solver"] == "Hypre" and linear["enable_overwrite_solver"], linear
    assert linear["Hypre"]["dimension"] == 3, linear
    assert linear["Hypre"]["nodal_coarsening"] is True
    assert linear["Hypre"]["theta"] == 0.4
    nl = data["solver"]["nonlinear"]
    assert nl["norm_type"] == "Linf" and nl["rel_grad_norm_tol"] == 1e-9, nl
    assert nl["rel_x_delta_tol"] == 1e-12 and nl["newton_decrement_tol"] == 1e-14
    assert nl["allow_non_grad_convergence"] is True
    assert data["solver"]["augmented_lagrangian"]["lumping"] == "hrz"
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info"],
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    sys.stdout.write(result.stdout[-1500:])
    assert result.returncode == 0, "PolyFEM run with Hypre failed"
    assert "falling back" not in result.stdout, "Hypre missing from binary"
    print("PASS: Hypre (dimension 3, nodal coarsening) export and run")

    # --- a solver the binary may lack must warn and fall back, not abort --
    node.parm("solver").set("AMGCL")
    params_path = mod.write_params_only({"node": node})
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info"],
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    sys.stdout.write(result.stdout[-1500:])
    assert result.returncode == 0, "run with an unbuilt solver aborted"
    assert "invalid input json" not in result.stdout
    print("PASS: unbuilt linear solver falls back instead of aborting"
          + (" (fallback taken)" if "falling back" in result.stdout
             else " (AMGCL present in this binary)"))

    # --- the new controls round-trip through the importer ------------------
    node.parm("solver").set("Hypre")
    params_path = mod.write_params_only({"node": node})
    node3 = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_roundtrip_solver")
    node3.setParms({"old_input_dir": input_dir})
    node3.hdaModule().read_params({"node": node3})
    assert node3.parm("solver").evalAsString() == "Hypre"
    assert node3.evalParm("nodal_coarsening_hypre") == 1
    assert abs(node3.evalParm("theta_hypre") - 0.4) < 1e-12
    assert node3.parm("norm_type").evalAsString() == "Linf"
    assert abs(node3.evalParm("rel_grad_norm_tol") - 1e-9) < 1e-20
    assert abs(node3.evalParm("rel_x_delta_tol") - 1e-12) < 1e-24
    assert abs(node3.evalParm("newton_decrement_tol") - 1e-14) < 1e-26
    assert node3.evalParm("allow_non_grad_convergence") == 1
    assert node3.parm("al_lumping").evalAsString() == "hrz"
    print("PASS: solver-panel controls round-trip through import")

    # --- BFGS audit stage 3: feasibility-respecting Strong Wolfe ----------
    # Append Wolfe to the menu so the saved ordinal values of all older
    # choices remain stable. Exercise its full JSON contract, importer and a
    # real dense-BFGS solve rather than accepting UI presence as sufficient.
    line_searches = list(node.parm("method").menuItems())
    assert line_searches == ["Armijo", "RobustArmijo", "Backtracking", "None",
                             "Wolfe"], line_searches
    node.parm("method").set("Wolfe")
    node.parm("solver_nl").set("BFGS")
    node.setParms({"wolfe_c2": 0.8, "wolfe_growth_factor": 1.75,
                   "wolfe_growth_limit": 12.0,
                   "wolfe_max_evaluations": 17,
                   "wolfe_max_objective_restarts": 3,
                   "wolfe_approximate_epsilon": 2e-6})
    params_path = mod.write_params_only({"node": node})
    with open(params_path) as f:
        data = json.load(f)
    line_search = data["solver"]["nonlinear"]["line_search"]
    assert line_search["method"] == "Wolfe", line_search
    assert line_search["Wolfe"] == {
        "c2": 0.8, "growth_factor": 1.75, "growth_limit": 12.0,
        "max_evaluations": 17, "max_objective_restarts": 3,
        "approximate_wolfe_epsilon": 2e-6}, line_search
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info"],
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    combined = result.stdout + result.stderr
    assert "invalid input json" not in combined, combined[-1500:]
    assert "[BFGS][Wolfe]" in combined, combined[-1500:]
    wolfe_back = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_wolfe_roundtrip")
    wolfe_back.setParms({"old_input_dir": input_dir})
    wolfe_back.hdaModule().read_params({"node": wolfe_back})
    assert wolfe_back.parm("method").evalAsString() == "Wolfe"
    assert abs(wolfe_back.evalParm("wolfe_c2") - 0.8) < 1e-12
    assert abs(wolfe_back.evalParm("wolfe_growth_factor") - 1.75) < 1e-12
    assert abs(wolfe_back.evalParm("wolfe_growth_limit") - 12.0) < 1e-12
    assert wolfe_back.evalParm("wolfe_max_evaluations") == 17
    assert wolfe_back.evalParm("wolfe_max_objective_restarts") == 3
    assert abs(wolfe_back.evalParm("wolfe_approximate_epsilon") - 2e-6) < 1e-18
    wolfe_back.destroy()
    node.parm("method").set("RobustArmijo")
    print("PASS: BFGS Strong Wolfe exports, runs and round-trips")

    # --- BFGS audit stage 5: forward nonlinear methods ------------------------
    # The menu offers only what PolyFEM's simulation can run; every offered
    # method exports JSON that round-trips through the importer and that the
    # binary constructs and iterates with (convergence is not required: the
    # first-order methods do not converge on this scene in one step).
    offered = list(node.parm("solver_nl").menuItems())
    assert offered == ["Newton", "GradientDescent", "ADAM", "StochasticADAM",
                       "StochasticGradientDescent", "L-BFGS", "BFGS"], offered
    ran_as = {"Newton": "[SparseNewton]", "GradientDescent": "[GradientDescent]",
              "ADAM": "[ADAM]", "StochasticADAM": "[StochasticADAM]",
              "StochasticGradientDescent": "[StochasticGradientDescent]",
              "L-BFGS": "[L-BFGS]", "BFGS": "[BFGS]"}
    refusals = ("Unrecognized solver type", "is a box-constrained method",
                "must be dense", "must be sparse", "Dense Hessian not implemented",
                "invalid input json")
    node.parm("solver").set("auto")
    node.setParms({"tend": 0.125, "max_iterations": 30})
    for method in offered:
        node.parm("solver_nl").set(offered.index(method))
        params_path = mod.write_params_only({"node": node})
        with open(params_path) as f:
            data = json.load(f)
        assert data["solver"]["nonlinear"]["solver"] == method, method
        linear = data["solver"].get("linear", {})
        if method == "BFGS":
            assert linear == {"solver": "Eigen::LDLT"}, linear
        else:
            assert linear.get("solver") != "Eigen::LDLT", linear
        result = subprocess.run(
            [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
             "--log_level", "info"],
            cwd=input_dir, capture_output=True, text=True, timeout=900)
        found = [r for r in refusals if r in result.stdout + result.stderr]
        assert not found, (method, found, result.stdout[-1500:])
        assert ran_as[method] in result.stdout, (method, result.stdout[-1500:])
        back = hou.node("/obj").createNode(
            "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_method_" +
            method.replace("-", "_"))
        back.setParms({"old_input_dir": input_dir})
        back.hdaModule().read_params({"node": back})
        assert back.parm("solver_nl").evalAsString() == method, method
        report = back.evalParm("import_report")
        assert "solver.linear" not in report and "Nonlinear solver" not in report, report
        back.destroy()
        print(f"  {method}: exported, run by PolyFEM "
              f"(exit {result.returncode}), round-tripped")
    print("PASS: every offered nonlinear method runs in PolyFEM and round-trips")

    # A withdrawn method in an imported file: Newton is kept, with the reason;
    # and PolyFEM itself names it if the file is run as it is.
    node.parm("solver_nl").set(0)
    params_path = mod.write_params_only({"node": node})
    with open(params_path) as f:
        data = json.load(f)
    for withdrawn, why in (("L-BFGS-B", "box-constrained optimizer"),
                           ("MMA", "box-constrained optimizer"),
                           ("DenseNewton", "needs a dense Hessian")):
        data["solver"]["nonlinear"]["solver"] = withdrawn
        with open(params_path, "w") as f:
            json.dump(data, f)
        back = hou.node("/obj").createNode(
            "stevenabramowitch::dev::PolyFEM::2.0", "polyfem_withdrawn")
        back.setParms({"old_input_dir": input_dir})
        back.hdaModule().read_params({"node": back})
        assert back.parm("solver_nl").evalAsString() == "Newton", withdrawn
        report = back.evalParm("import_report")
        assert f"Nonlinear solver '{withdrawn}' is not offered" in report \
            and why in report, report
        back.destroy()
        result = subprocess.run(
            [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
             "--log_level", "info"],
            cwd=input_dir, capture_output=True, text=True, timeout=900)
        assert result.returncode != 0, withdrawn
        named = {"L-BFGS-B": "L-BFGS-B is a box-constrained method",
                 "MMA": "MMA is a box-constrained method",
                 "DenseNewton": "the dense Newton strategies"}[withdrawn]
        assert named in result.stdout + result.stderr, \
            (withdrawn, result.stdout[-1500:])
    print("PASS: withdrawn methods import as Newton with the reason; "
          "PolyFEM refuses them by name")

    print("\nPASS: end-to-end PolyFEM 2.0 HDA test")
    print("workdir:", work)
    return work


if __name__ == "__main__":
    main()
