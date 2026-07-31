"""Legacy params.json import test (old fork schema -> PolyFEM 2.0 UI)."""

import json
import os
import tempfile

import hou

from test_polyfem_hda import BASE, make_cube_msh


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_legacy_test_")
    input_dir = os.path.join(work, "input")
    os.makedirs(input_dir)
    cube = os.path.join(input_dir, "cube.msh")
    make_cube_msh(cube)

    legacy = {
        "geometry": [{
            "mesh": "cube.msh", "enabled": True, "is_obstacle": False,
            "volume_selection": "volumes1.txt",
            "transformation": {"translation": [0.5, 0, 0],
                               "rotation": [0, 10, 0], "scale": [1, 1, 2]}}],
        "materials": [{"id": 11, "type": "NeoHookean", "E": 5e6, "nu": 0.4,
                       "rho": 1100}],
        "time": {"t0": 0, "tend": 2.0, "dt": 0.5, "quasistatic": True},
        "contact": {"enabled": True, "dhat": 5e-4,
                    "friction_coefficient": 0.2},
        "solver": {
            "contact": {"barrier_stiffness": "adaptive",
                        "adaptive_barrier_stiffness_multiplier": 100.0},
            "augmented_lagrangian": {"initial_weight": 1e8}},
    }
    with open(os.path.join(input_dir, "params.json"), "w") as f:
        json.dump(legacy, f)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "legacy_import")
    node.setParms({"old_input_dir": input_dir})
    node.hdaModule().read_params({"node": node})

    assert node.evalParm("num_geos") == 1
    assert node.evalParm("file_location1").endswith("cube.msh")
    assert abs(node.evalParm("xform_t__1x") - 0.5) < 1e-9
    assert abs(node.evalParm("xform_s__1z") - 2.0) < 1e-9
    assert node.evalParm("materials1_1") == 3  # NeoHookean
    assert abs(node.evalParm("E1_1") - 5e6) < 1
    assert node.evalParm("quasistatic") == 1
    assert abs(node.evalParm("tend") - 2.0) < 1e-9
    assert node.evalParm("enable") == 1
    assert abs(node.evalParm("dhat") - 5e-4) < 1e-12
    # legacy adaptive + fork multiplier -> adaptive mode, multiplier dropped
    assert node.evalParm("barrier_mode") == 1
    assert node.evalParm("al_hessian_scaled") == 0
    assert abs(node.evalParm("al_initial_weight") - 1e8) < 1

    print("PASS: legacy HDA params.json import")

    # Generic older PolyFEM scenes use small material/selection ids rather than
    # either HDA's encoding. Reconstruct those into current multiparms too.
    generic = {
        "geometry": [{
            "mesh": "cube.msh", "volume_selection": 1,
            "surface_selection": [
                {"id": 10, "axis": "-z", "position": 0.001}]}],
        "materials": [{
            "id": 1, "type": "NeoHookean", "E": 3e6, "nu": 0.35,
            "rho": 1020}],
        "initial_conditions": {
            "velocity": [{"id": 1, "value": [0, 0, 0.2]}]},
        "boundary_conditions": {
            "dirichlet_boundary": [{
                "id": 10, "value": [0, 0, 0],
                "dimension": [True, False, True]}]},
        "time": {"t0": 0, "dt": 0.2, "time_steps": 3},
        "contact": {"enabled": False},
        "space": {"discr_order": [{"id": 1, "order": 2}]},
        "solver": {
            "nonlinear": {
                "solver": "Newton", "max_iterations": 31,
                "x_delta": 2e-8, "grad_norm": 3e-7,
                "first_grad_norm_tol": 4e-5,
                "line_search": {"method": "none"}},
            "contact": {"barrier_stiffness": 5000},
            "augmented_lagrangian": {"initial_weight": 2e6}},
        "output": {
            "paraview": {
                "file_name": "old.pvd", "options": {"material": False}},
            "log": {"level": "warning"}},
    }
    with open(os.path.join(input_dir, "params.json"), "w") as f:
        json.dump(generic, f)
    generic_node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "generic_old_import")
    generic_node.setParms({"old_input_dir": input_dir})
    generic_node.hdaModule().read_params({"node": generic_node})
    assert generic_node.evalParm("num_volumes1") == 1
    assert generic_node.evalParm("materials1_1") == 3
    assert generic_node.evalParm("sideset_selection1_1") == 1
    assert generic_node.evalParm("basegroup1_1_1") == "axis:-z:0.001"
    assert generic_node.evalParm("Boundary_Condition__1_1_1") == 1
    assert generic_node.evalParm("boundary_type1_1_1_1") == 0
    assert generic_node.evalParm("y_dimension1_1_1_1") == 0
    assert generic_node.evalParm("initial_conditions1_1") == 1
    assert generic_node.evalParm("conditiontype1_1_1") == 1
    assert generic_node.evalParm("mainOrder1_1") == 1
    assert abs(generic_node.evalParm("x_delta") - 2e-8) < 1e-15
    assert abs(generic_node.evalParm("grad_norm") - 3e-7) < 1e-15
    assert generic_node.evalParm("method") == 5
    print("PASS: generic older PolyFEM ids/selections import")


if __name__ == "__main__":
    main()
