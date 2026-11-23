"""Analytical one-contact equilibrium through native SOP, PolyFEM and Read PVD."""

import json
import math

import hou
import numpy as np

from test_model_oracle_pipeline import BODY_ID, MSH_READER, check_close, exact_definition, read_json, tetrahedron_sop, vectors
from test_simulation_oracle_pipeline import add_condition, face_prim, pipeline_inputs, run_reader
from test_transient_load_restart import new_node


START_GAP = 3 / 32
MOTION = -1 / 32
SUPPORT = 1 / 8
BARRIER_STIFFNESS = 10000
APEX_STIFFNESS = 6000
BASE = START_GAP + 1
SOLID = [(0., 0., BASE), (1., 0., BASE), (0., 1., BASE), (0., 0., START_GAP)]


def contact_force(gap):
    # For the clamped log B(s)=-(s-h)^2 log(s/h), s=gap^2 and h=dhat^2.
    # The force on the solid apex is -k dB/dgap. All distances are in m.
    s, h = gap**2, SUPPORT**2
    b_prime = -2 * (s - h) * math.log(s / h) - (s - h)**2 / s
    return -2 * gap * BARRIER_STIFFNESS * b_prime


def setup(work, binary, applied):
    node = new_node(work, binary)
    sop = tetrahedron_sop(work.name + "_input", SOLID, [(0, 1, 2, 3)], [(0., 0., 1.)])
    node.setParms({"mesh_source1": "sop", "sop_path1": sop.path(),
                   "enable": 1, "barrier_mode": 2, "barrier_stiffness": BARRIER_STIFFNESS,
                   "dhat": SUPPORT, "cof": 0., "use_hdf5": 0,
                   "quasistatic": 1, "end_time_bool": 1, "t0": 0., "tend": 1.,
                   "time_inc_bool": 1, "dt": 1., "units": 1,
                   "length": "m", "mass": "kg", "time": "s",
                   "forces_fields": 1, "tensor_values": 1, "body_ids_fields": 1,
                   "vis_boundary_only": 0, "sol_at_node": 1})
    module = node.hdaModule()
    assert module.load_sop_mesh({"node": node, "script_multiparm_index": "1"})
    exact_definition(node.node("geo_1"), MSH_READER)
    assert node.evalParm("num_volumes1") == 1
    node.setParms({"materials1_1": module.MATERIAL_TOKENS.index("LinearElasticity"),
                   "E1_1": 30000., "nu1_1": 0.25})
    plane = work / "rigid_triangle.obj"
    plane.write_text("v -2 -2 0\nv 6 -2 0\nv -2 6 0\nf 1 2 3\n")
    node.setParms({"num_geos": 2, "geo_int": 2, "is_obstacle2": 1, "file_location2": str(plane)})
    node.parm("file_location2").pressButton()

    base = face_prim(node, 1, lambda p: p[2] == BASE)
    slope = face_prim(node, 1, lambda p: p[0] + p[1] + BASE - p[2] == 1.)
    add_condition(node, 1, 1, base, 0, [0, 0, 0], (1, 1, 1))
    add_condition(node, 1, 2, slope, 0, [0, 0, 0], (1, 1, 0))
    load = add_condition(node, 1, 3, "axis:-z:0.1", 5, [0, 0, applied], (1, 1, 1), points=True)
    return node, load


def check_export(work, load, applied):
    inp = work / "input"
    args = read_json(inp / "params.json")
    assert args["time"]["quasistatic"]
    assert args["contact"]["enabled"] and args["contact"]["dhat"] == SUPPORT
    assert args["solver"]["contact"]["barrier_stiffness"] == BARRIER_STIFFNESS
    assert not args["contact"].get("use_physical_barrier", False)
    assert not args["contact"].get("use_area_weighting", False)
    assert {k: args["units"][k] for k in ("length", "mass", "time")} == {
        "length": "m", "mass": "kg", "time": "s"}
    solid, obstacle = args["geometry"]
    assert obstacle["is_obstacle"] and not solid["is_obstacle"], args["geometry"]
    point_selection, = solid["point_selection"]
    row, = np.loadtxt(inp / point_selection["file"], dtype=int, ndmin=2)
    assert tuple(row) == (load, 3), row
    entry, = args["boundary_conditions"]["nodal_neumann_boundary"]
    assert entry["id"] == load
    check_close("exported applied z force (N)", entry["value"], [0, 0, applied], 1e-10)


def check_readback(reader, work, applied, contact):
    module = reader.hdaModule()
    pvd = reader.evalParm("PVD_file")
    times = [t for t, *_ in module.read_pvd(pvd)]
    check_close("Read PVD times (s)", times, (0, 1), 1e-12)
    hou.setFrame(module.entry_frame(reader, 1))
    sop = reader.node("OUT_result")
    sop.cook(force=True)
    geo = sop.geometry()
    pos = vectors(geo, "P")
    index, = np.flatnonzero(np.linalg.norm(pos - SOLID[3], axis=1) < 1e-9)
    assert int(geo.point(int(index)).attribValue("body_ids")) == BODY_ID

    apex = {key: vectors(geo, key)[index] for key in ("solution", "body_forces", "elastic_forces", "contact_forces")}
    check_close("apex displacement (m)", apex["solution"], (0, 0, MOTION), 2e-7)
    check_close("apex gap (m)", START_GAP + apex["solution"][2], START_GAP + MOTION, 2e-7)
    check_close("apex applied force (N)", apex["body_forces"], (0, 0, applied), 2e-5)
    check_close("apex elastic force (N)", apex["elastic_forces"], (0, 0, -APEX_STIFFNESS * MOTION), 2e-3)
    check_close("apex contact force (N)", apex["contact_forces"], (0, 0, contact), 2e-3)
    check_close("apex equilibrium (N)", apex["body_forces"] + apex["elastic_forces"] + apex["contact_forces"],
                (0, 0, 0), 2e-3)
    check_close("apex stress zz (Pa)", vectors(geo, "cauchy_stress_3")[index, 2], -36000 * MOTION, 0.012)

    record = read_json(work / "input" / "hda_scene.json")
    curves = module.force_curves(reader)
    base, = [i for i, entry in enumerate(record["force_sets"], 1)
             if entry["body"] == BODY_ID and entry["sideset"] == 1]
    obstacle, = [i for i, entry in enumerate(record["force_sets"], 1) if entry["kind"] == "obstacle"]
    assert curves["nodes"][base] == curves["nodes"][obstacle] == 3, curves
    assert curves["unmatched"][base] == curves["unmatched"][obstacle] == 0, curves
    support_reaction = curves["reaction"][-1, curves["sets"].index(base)]
    obstacle_contact = curves["contact"][-1, curves["sets"].index(obstacle)]
    check_close("base support reaction (N)", support_reaction, (0, 0, -APEX_STIFFNESS * MOTION), 2e-3)
    check_close("contact on the obstacle (N)", obstacle_contact, (0, 0, -contact), 2e-3)
    data = {name: list(map(float, value)) for name, value in apex.items()}
    data.update(base_reaction_N=support_reaction.tolist(), obstacle_contact_N=obstacle_contact.tolist())
    (work / "measurements.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    root, binary, identity = pipeline_inputs(__doc__)
    # The base is fixed and the axial P1 mode has K=(lambda+2*mu)V = 6000 N/m.
    # Manufacture a downward nodal load for the selected equilibrium gap.
    contact = contact_force(START_GAP + MOTION)
    applied = APEX_STIFFNESS * MOTION - contact
    work = root / "contact_equilibrium"
    node, load = setup(work, binary, applied)
    reader = run_reader(node, work, binary, identity)
    check_export(work, load, applied)
    check_readback(reader, work, applied, contact)
    print("1/1 active contact Houdini case passed", flush=True)


if __name__ == "__main__":
    main()
