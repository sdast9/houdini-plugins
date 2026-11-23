"""Analytical nodal loading and checkpoint inertia through the Houdini HDAs."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import hou
import numpy as np

from test_model_oracle_pipeline import (
    BODY_ID, MSH_READER, POLYFEM, UNIT_TET, check_close, exact_definition,
    read_json, tetrahedron_sop, vectors,
)
from test_simulation_oracle_pipeline import add_condition, face_prim, msh22_tets, pipeline_inputs, run_reader


DT = 0.125
TIMES = [step * DT for step in range(6)]
SPLIT = 2


def new_node(work, binary):
    work.mkdir()
    node = hou.node("/obj").createNode("stevenabramowitch::dev::PolyFEM::2.0", work.name)
    exact_definition(node, POLYFEM)
    node.setParms({"working_dir": str(work) + "/", "polyfem_bin": str(binary),
                   "max_threads": 1, "log_level": "info"})
    return node


def setup(work, binary, lumped):
    sop = tetrahedron_sop(work.name + "_input", UNIT_TET, [(0, 1, 2, 3)], [(0., 0., 1.)])
    node = new_node(work, binary)
    node.setParms({"mesh_source1": "sop", "sop_path1": sop.path(), "enable": 0,
                   "units": 1, "length": "m", "mass": "kg", "time": "s",
                   "quasistatic": 0, "integrator": 0, "end_time_bool": 1, "t0": 0, "tend": TIMES[-1],
                   "time_inc_bool": 1, "dt": DT, "use_hdf5": 0,
                   "lump_mass_matrix": int(lumped), "restart_json": 1,
                   "velocity_fields": 1, "acceleration_fields": 1,
                   "body_ids_fields": 1, "forces_fields": 1,
                   "vis_boundary_only": 0, "sol_at_node": 1})
    module = node.hdaModule()
    assert module.load_sop_mesh({"node": node, "script_multiparm_index": "1"})
    exact_definition(node.node("geo_1"), MSH_READER)
    assert node.evalParm("num_volumes1") == 1
    node.setParms({"materials1_1": module.MATERIAL_TOKENS.index("LinearElasticity"),
                   "E1_1": 30000, "nu1_1": 0.25, "rho1_1": 120})
    base = face_prim(node, 1, lambda p: p[2] == 0)
    slope = face_prim(node, 1, lambda p: sum(p) == 1)
    add_condition(node, 1, 1, base, 0, [0, 0, 0], (1, 1, 1))
    add_condition(node, 1, 2, slope, 0, [0, 0, 0], (0, 1, 1))
    force = add_condition(node, 1, 3, "axis:+z:0.99", 5, [30, 0, 0], (1, 1, 1), ramp=True, points=True)
    return node, force


def check_export(work, force, lumped):
    inp = work / "input"
    args = read_json(inp / "params.json")
    time = args["time"]
    assert (time["t0"], time["tend"], time["dt"], time["integrator"], time["quasistatic"]) == (
        0, TIMES[-1], DT, "ImplicitEuler", False), time
    assert {key: args["units"][key] for key in ("length", "mass", "time")} == {
        "length": "m", "mass": "kg", "time": "s"}
    assert args["contact"]["enabled"] is False
    assert args["solver"]["advanced"]["lump_mass_matrix"] == lumped
    material, = args["materials"]
    assert (material["type"], material["id"], material["E"], material["nu"], material["rho"]) == (
        "LinearElasticity", BODY_ID, 30000, 0.25, 120), material
    geometry, = args["geometry"]
    points, _ = msh22_tets(inp / geometry["mesh"], count=1)
    selection, = geometry["point_selection"]
    row, = np.loadtxt(inp / selection["file"], dtype=int, ndmin=2)
    assert row[0] == force and points[row[1]] == UNIT_TET[3], (row, points)
    load, = args["boundary_conditions"]["nodal_neumann_boundary"]
    assert load["id"] == force and load["value"] == [30, 0, 0], load
    assert load["interpolation"] == [
        {"type": "piecewise_linear", "points": [0, TIMES[-1]], "values": [0, 1], "extend": "constant"}
    ], load


def expected_fields(lumped):
    # K = G V on the unit tet. Consistent mass has m_aa=2 kg and
    # m_ia=1 kg for i!=a. Row-sum lumping puts 5 kg on the diagonal.
    stiffness = 2000
    diagonal, off_diagonal = (5, 0) if lumped else (2, 1)
    u = v = a = 0
    result = []
    for t in TIMES:
        force = 48 * t
        if t > 0:
            u_next = (diagonal * (u + DT * v) + DT**2 * force) / (diagonal + DT**2 * stiffness)
            v_next = (u_next - u) / DT
            a = (v_next - v) / DT
            u, v = u_next, v_next
        f = stiffness * u
        result.append({
            "solution": [(0, 0, 0)] * 3 + [(u, 0, 0)],
            "velocity": [(0, 0, 0)] * 3 + [(v, 0, 0)],
            "acceleration": [(0, 0, 0)] * 3 + [(a, 0, 0)],
            "elastic_forces": [(f, 0, f), (0, 0, -f), (0, 0, 0), (-f, 0, 0)],
            "body_forces": [(0, 0, 0)] * 3 + [(force, 0, 0)],
            "inertia_forces": [(-off_diagonal * a, 0, 0)] * 3 + [(-diagonal * a, 0, 0)],
        })
    return result


def check_frames(reader, work, reference, first=0):
    pvd = Path(reader.evalParm("PVD_file"))
    sets = ET.parse(pvd).findall("./Collection/DataSet")
    check_close(work.name + " PVD times (s)", [float(s.attrib["timestep"]) for s in sets], TIMES, 1e-12)
    measured = []
    for step in range(first, len(TIMES)):
        assert (pvd.parent / sets[step].attrib["file"]).is_file(), sets[step].attrib
        hou.setFrame(reader.hdaModule().entry_frame(reader, step))
        sop = reader.node("OUT_result")
        sop.cook(force=True)
        geo = sop.geometry()
        assert len(geo.points()) == 4, sop.errors()
        positions = vectors(geo, "P")
        indices = []
        for p in UNIT_TET:
            index, = np.flatnonzero(np.linalg.norm(positions - p, axis=1) < 1e-9)
            assert int(geo.point(int(index)).attribValue("body_ids")) == BODY_ID
            indices.append(index)

        actual = {name: vectors(geo, name)[indices] for name in reference[step]}
        for name, values in reference[step].items():
            atol = {"solution": 3e-8, "velocity": 3e-7, "acceleration": 3e-6}.get(name, 3e-5)
            check_close(f"{work.name} step {step} {name}", actual[name], values, atol)
        measured.append({"step": step, "time_s": TIMES[step],
                         "apex_u_x_m": float(actual["solution"][3, 0]),
                         "apex_v_x_m_per_s": float(actual["velocity"][3, 0]),
                         "apex_a_x_m_per_s2": float(actual["acceleration"][3, 0]),
                         "apex_inertia_x_N": float(actual["inertia_forces"][3, 0]),
                         "apex_applied_x_N": float(actual["body_forces"][3, 0])})
    (work / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")


def check_run(root, binary, identity, lumped):
    label = "lumped" if lumped else "consistent"
    full = root / (label + "_full")
    node, force = setup(full, binary, lumped)
    reader = run_reader(node, full, binary, identity)
    check_export(full, force, lumped)
    reference = expected_fields(lumped)
    check_frames(reader, full, reference)

    resume = root / (label + "_resume")
    resumed = new_node(resume, binary)
    checkpoint = (full / "output" / f"checkpoint_{SPLIT}.hdf5").resolve(strict=True)
    resumed.setParms({"resume_checkpoint": str(checkpoint), "num_geos": 0})
    reader = run_reader(resumed, resume, binary, identity)
    assert read_json(resume / "input" / "params.json") == {
        "input": {"checkpoint": {"path": str(checkpoint), "reorder": False}}}
    # The collection retains global indices. The resumed directory starts at
    # the checkpoint, which is the first available topology frame there.
    reader.parm("topo_frame").set(int(reader.hdaModule().entry_frame(reader, SPLIT)))
    check_frames(reader, resume, reference, SPLIT)
    check_close(label + " repeated checkpoint inertia (N)",
                read_json(resume / "measurements.json")[0]["apex_inertia_x_N"],
                read_json(full / "measurements.json")[SPLIT]["apex_inertia_x_N"], 1e-7)
    print(f"PASS {label} nodal load and checkpoint readback", flush=True)


def main():
    root, binary, identity = pipeline_inputs(__doc__)
    for lumped in (False, True):
        check_run(root, binary, identity, lumped)
    print("2/2 Houdini transient and restart cases passed", flush=True)


if __name__ == "__main__":
    main()
