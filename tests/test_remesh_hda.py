"""Legacy remeshing import preservation and explicit refusal test.

Run: hython tests/test_remesh_hda.py
"""

import json
import os
import tempfile

import hou

from test_polyfem_hda import BASE, make_cube_msh


REMESH_PARMS = {
    "remeshing_enabled": 1,
    "remesh_type": 1,
    "split_enabled": 1,
    "split_acceptance_tol": 0.002,
    "split_culling_threshold": 0.8,
    "split_max_depth": 4,
    "min_edge_length": 2e-6,
    "collapse_enabled": 0,
    "collapse_acceptance_tol": -2e-8,
    "collapse_culling_threshold": 0.02,
    "collapse_max_depth": 5,
    "rel_max_edge_length": 0.7,
    "abs_max_edge_length": 10.0,
    "swap_enabled2": 1,
    "swap_acceptance_tol2": -3e-8,
    "swap_max_depth2": 2,
    "smooth_enabled": 1,
    "smooth_acceptance_tol": -4e-8,
    "smooth_max_iters": 2,
    "local_mesh_n_ring": 3,
    "local_mesh_rel_area": 0.02,
    "max_nl_iterations": 2,
}

EXPECTED_REMESH = {
    "enabled": True,
    "type": "sizing_field",
    "split": {
        "enabled": True,
        "acceptance_tolerance": 0.002,
        "culling_threshold": 0.8,
        "max_depth": 4,
        "min_edge_length": 2e-6,
    },
    "collapse": {"enabled": False},
    "swap": {
        "enabled": True,
        "acceptance_tolerance": -3e-8,
        "max_depth": 2,
    },
    "smooth": {
        "enabled": True,
        "acceptance_tolerance": -4e-8,
        "max_iters": 2,
    },
    "local_relaxation": {
        "local_mesh_n_ring": 3,
        "local_mesh_rel_area": 0.02,
        "max_nl_iterations": 2,
    },
}


def make_node(name, work, mesh):
    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", name)
    node.setParms({"working_dir": work + "/", "file_location1": mesh})
    node.parm("file_location1").pressButton()
    return node


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"), force_use_assets=True)

    work = tempfile.mkdtemp(prefix="polyfem_remesh_test_")
    mesh = os.path.join(work, "cube.msh")
    make_cube_msh(mesh)

    source = make_node("remesh_source", work, mesh)
    module = source.hdaModule()
    params_path = module.build_params(source)
    baseline = load_json(params_path)
    assert "remesh" not in baseline["space"]
    legacy = json.loads(json.dumps(baseline))
    legacy["space"]["remesh"] = EXPECTED_REMESH
    with open(params_path, "w") as handle:
        json.dump(legacy, handle)

    restored = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "remesh_restored")
    restored.setParms({"old_input_dir": os.path.dirname(params_path)})
    rm = restored.hdaModule()
    rm.read_params({"node": restored})
    for parm, expected in REMESH_PARMS.items():
        if parm.startswith("collapse_") or parm in (
                "rel_max_edge_length", "abs_max_edge_length"):
            continue
        assert restored.evalParm(parm) == expected, parm
    assert restored.evalParm("collapse_enabled") == 0
    assert restored.evalParm("collapse_acceptance_tol") == -1e-8
    assert "no longer supported" in restored.evalParm("import_report")
    restored.setParms({"area_weighted": 0})
    before = open(params_path, "rb").read()
    for purpose in ("check", "write", "run", "background"):
        assert any("no longer supported" in error
                   for error in rm.check_setup(restored, purpose)["errors"])
    for action in (lambda: rm.build_params(restored),
                   lambda: rm.build_remesh(restored)):
        try:
            action()
        except hou.NodeError as exc:
            assert "no longer supported" in str(exc)
        else:
            raise AssertionError("Legacy remeshing was exported")
    assert open(params_path, "rb").read() == before
    assert restored.evalParm("area_weighted") == 0
    contact = {}
    rm.build_contact(restored, contact)
    assert restored.evalParm("area_weighted") == 0

    # Native checkpoint settings override stale legacy UI settings.
    checkpoint = os.path.join(work, "placeholder.hdf5")
    open(checkpoint, "wb").close()
    restored.setParms({"resume_checkpoint": checkpoint})
    assert not any("no longer supported" in error for error in
                   rm.check_setup(restored, "write")["errors"])
    wrapper = load_json(rm.build_params(restored))
    assert wrapper == {"input": {"checkpoint": {"path": checkpoint,
                                                "reorder": False}}}
    restored.setParms({"resume_checkpoint": "", "remeshing_enabled": 0})
    assert "remesh" not in load_json(rm.build_params(restored))["space"]

    # Missing/null space and missing remesh must clear stale legacy controls.
    for candidate in ({}, {"space": None}, {"space": {}}):
        restored.setParms({"remeshing_enabled": 1, "split_max_depth": 9})
        rm._restore_space(restored, candidate, {}, False, [])
        assert restored.evalParm("remeshing_enabled") == 0
        assert restored.evalParm("split_max_depth") == 3

    # Optimization inputs must not destructively import as forward scenes.
    prior_children = tuple(child.path() for child in restored.children())
    prior_parms = restored.evalParm("num_geos")
    with open(params_path, "w") as handle:
        json.dump({"states": [], "remeshing": {"trigger": "periodic"}}, handle)
    rm.read_params({"node": restored})
    assert tuple(child.path() for child in restored.children()) == prior_children
    assert restored.evalParm("num_geos") == prior_parms
    print("PASS: legacy remeshing retained, explicitly refused, no silent MMG mapping")


if __name__ == "__main__":
    main()
