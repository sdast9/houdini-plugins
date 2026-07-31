"""Current-schema folder import: edited entities and parameter reconstruction."""

import json
import os
import tempfile

import hou
import numpy as np

from test_polyfem_hda import BASE, make_cube_msh


def volume_entities(node):
    geo = node.node("branch_1").geometry()
    entities = np.asarray(geo.primIntAttribValues("Entity"), dtype=np.int64)
    mask = np.asarray(geo.primIntAttribValues("is_volume"), dtype=bool)
    return entities[mask]


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_import_reconstruction_")
    os.makedirs(os.path.join(work, "input"))
    mesh = os.path.join(work, "cube.msh")
    make_cube_msh(mesh)

    source = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "roundtrip_source")
    source.setParms({"working_dir": work + "/", "file_location1": mesh})
    source.parm("file_location1").pressButton()
    phm = source.hdaModule()

    # The MSH has one physical volume. Re-tag half of its elements as a second
    # material exactly as a user would in the HDA.
    geo = source.node("branch_1").geometry()
    mask = np.asarray(geo.primIntAttribValues("is_volume"), dtype=bool)
    volume_prims = np.flatnonzero(mask)
    edited = volume_prims[len(volume_prims) // 2:]
    source.setParms({
        "subdomain_number_1": 2,
        "elements_1": " ".join(str(value) for value in edited)})
    phm.update_entities(
        {"node": source, "script_multiparm_index": "1"})
    original_entities = volume_entities(source)
    assert set(original_entities) == {1, 2}

    source.setParms({
        "materials1_1": phm.MATERIAL_TOKENS.index("NeoHookean"),
        "E1_1": 2.5e6, "nu1_1": 0.43, "rho1_1": 1050,
        "materials1_2": phm.MATERIAL_TOKENS.index("HGOFiber"),
        "hgo_k11_2": 12345, "hgo_k21_2": 7.5, "rho1_2": 990,
        "fib_source1_2": "constant",
        "xform_t__1x": 0.25, "xform_t__1y": -0.5, "xform_t__1z": 1.0,
        "xform_r__1x": 12.0, "xform_r__1y": -7.0, "xform_r__1z": 18.0,
        "xform_s__1x": 1.1, "xform_s__1y": 0.9, "xform_s__1z": 1.2,
        "integrator": 7, "newmark_beta": 0.31, "newmark_gamma": 0.61,
        "t0": 0.2, "end_time_bool": 1, "tend": 2.4,
        "time_inc_bool": 1, "dt": 0.08, "quasistatic": 0,
        "enable": 1, "dhat": 0.002, "epsv": 0.0003, "cof": 0.27,
        "area_weighted": 1, "gcp_enable": 1, "alpha_n": 0.8,
        "alpha_t": 0.7, "min_dist_ratio": 0.15, "adapt_dhat": 1,
        "adhesion_enable": 1, "dhat_p": 0.03, "dhat_a": 0.015,
        "adhesion_strength": 2.1, "tangent_coeff": 0.45,
        "adhesion_epsa": 0.0002,
        "remeshing_enabled": 1, "remesh_type": 1,
        "split_enabled": 1, "split_acceptance_tol": 0.004,
        "collapse_enabled": 1, "collapse_max_depth": 4,
        "swap_enabled2": 1, "swap_max_depth2": 5,
        "smooth_enabled": 1, "smooth_max_iters": 3,
        "local_mesh_n_ring": 4, "local_mesh_rel_area": 0.02,
        "max_nl_iterations": 3,
        "method": 2, "use_grad_norm_tol": 1,
        "max_iterations": 77, "x_delta": 1e-8, "grad_norm": 2e-7,
        "first_grad_norm_tol": 3e-5, "armijo_c": 0.002,
        "delta_relative_tolerance": 0.14,
        "broad_phase": 3, "CCD_tolerance": 1e-7,
        "ccd_max_iterations": 88, "friction_iterations": 4,
        "tangent_iter": 5, "friction_convergence_tol": 4e-4,
        "barrier_mode": 2, "barrier_stiffness": 7e5,
        "al_hessian_scaled": 0, "al_initial_weight": 8e6,
        "al_scaling": 9.0, "al_max_weight": 1e10, "al_eta": 0.2,
        "cache_size": 23, "lump_mass_matrix": 1,
        "num_rayleigh": 2,
        "rayleigh_form1": 0, "stiffness_ratio1": 0.03,
        "rayleigh_lagging1": 2,
        "rayleigh_form2": 2, "stiffness_ratio2": 0.04,
        "rayleigh_lagging2": 3,
        "paraview_file_name": "roundtrip.pvd",
        "vismesh_rel_area": 333.0, "skip_frame": 2,
        "high_order_mesh": 0, "paraview_wireframe": 1,
        "paraview_points": 1, "materials_fields": 1,
        "body_ids_fields": 1, "velocity_fields": 1,
        "scalar_values": 1, "tensor_values": 1,
        "minimal_fields": 0, "output_json": 1, "restart_json": 1,
        "solution_file": 1, "state_file": 1, "nodes": 1,
        "timestep_prefix": "frame_", "sol_on_grid": 0.5,
        "compute_error": 0, "sol_at_node": 2.0,
        "vis_boundary_only": 1, "save_time_sequence": 1,
        "log_level": "debug", "log_quiet": 1, "write_log": 1,
        "units": 1, "length": "mm", "mass": "kg", "time": "s",
        "char_length": 12.5})
    source.parmTuple("fib_dir1_2").set((0.2, 0.9, 0.3))

    params_path = phm.build_params(source)
    with open(params_path) as handle:
        original_json = json.load(handle)

    restored = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "roundtrip_restored")
    # Root-folder selection is supported in addition to selecting input itself.
    restored.setParms({"old_input_dir": work})
    restored.hdaModule().read_params({"node": restored})
    restored_entities = volume_entities(restored)
    assert np.array_equal(restored_entities, original_entities)
    assert restored.evalParm("num_volumes1") == 2
    assert restored.evalParm("materials1_2") == \
        phm.MATERIAL_TOKENS.index("HGOFiber")
    assert restored.evalParm("integrator") == 7
    assert abs(restored.evalParm("newmark_beta") - 0.31) < 1e-12
    assert restored.evalParm("remeshing_enabled") == 1
    assert restored.evalParm("method") == 2
    assert restored.evalParm("broad_phase") == 3
    assert restored.evalParm("num_rayleigh") == 2
    assert restored.evalParm("materials_fields") == 1
    assert restored.evalParm("units") == 1
    assert "All represented scene" in restored.evalParm("import_report")

    restored_path = restored.hdaModule().build_params(restored)
    with open(restored_path) as handle:
        restored_json = json.load(handle)
    for section in (
            "materials", "time", "contact", "space", "solver", "output",
            "units"):
        assert restored_json[section] == original_json[section], (
            section, restored_json[section], original_json[section])
    assert np.allclose(
        restored_json["geometry"][0]["transformation"]["translation"],
        original_json["geometry"][0]["transformation"]["translation"])
    assert np.allclose(
        restored_json["geometry"][0]["transformation"]["rotation"],
        original_json["geometry"][0]["transformation"]["rotation"])
    assert np.allclose(
        restored_json["geometry"][0]["transformation"]["scale"],
        original_json["geometry"][0]["transformation"]["scale"])
    assert np.array_equal(
        np.loadtxt(os.path.join(work, "input", "volumes1.txt"), dtype=np.int64),
        1000 + restored_entities)
    print("PASS: edited entities and represented PolyFEM parameters round-trip")
    print("workdir:", work)


if __name__ == "__main__":
    main()
