"""PolyFEM 2.0 remesh JSON write/read round-trip test.

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
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_remesh_test_")
    mesh = os.path.join(work, "cube.msh")
    make_cube_msh(mesh)

    source = make_node("remesh_source", work, mesh)
    source.setParms(REMESH_PARMS)
    params_path = source.hdaModule().write_params_only({"node": source})
    assert load_json(params_path)["space"]["remesh"] == EXPECTED_REMESH

    restored = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "remesh_restored")
    restored.setParms({"old_input_dir": os.path.dirname(params_path)})
    restored.hdaModule().read_params({"node": restored})
    for parm, expected in REMESH_PARMS.items():
        if parm.startswith("collapse_") or parm in (
                "rel_max_edge_length", "abs_max_edge_length"):
            continue  # disabled stages restore schema defaults
        assert restored.evalParm(parm) == expected, (
            parm, restored.evalParm(parm), expected)
    assert restored.evalParm("collapse_enabled") == 0
    assert restored.evalParm("collapse_acceptance_tol") == -1e-8
    roundtrip_path = restored.hdaModule().write_params_only({"node": restored})
    assert load_json(roundtrip_path)["space"]["remesh"] == EXPECTED_REMESH

    # Disabled remeshing is explicit in newly written JSON.
    restored.setParms({"remeshing_enabled": 0})
    disabled_path = restored.hdaModule().write_params_only({"node": restored})
    assert load_json(disabled_path)["space"]["remesh"] == {"enabled": False}

    # Older JSON files may omit remesh entirely; reading one must clear stale UI.
    older = load_json(disabled_path)
    del older["space"]["remesh"]
    with open(disabled_path, "w") as f:
        json.dump(older, f)
    restored.setParms({"remeshing_enabled": 1, "remesh_type": 1,
                       "split_max_depth": 9})
    restored.hdaModule().read_params({"node": restored})
    assert restored.evalParm("remeshing_enabled") == 0
    assert restored.evalParm("remesh_type") == 0
    assert restored.evalParm("split_max_depth") == 3

    print("PASS: PolyFEM remesh JSON round trip")


if __name__ == "__main__":
    main()
