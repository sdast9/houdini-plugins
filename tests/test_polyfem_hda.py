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
    assert "adaptive_barrier_stiffness_multiplier" not in contact
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

    # the Dirichlet push must actually move the cube (guards selection order)
    import re as _re
    linf = [float(m.group(1)) for m in _re.finditer(
        r"-- Linf error: ([0-9.e+-]+)", result.stdout)]
    assert linf and linf[-1] > 0.05, f"no displacement; Linf={linf}"
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
    print("PASS: params.json round-trip import")

    print("\nPASS: end-to-end PolyFEM 2.0 HDA test")
    print("workdir:", work)
    return work


if __name__ == "__main__":
    main()
