"""Analytical two-body traction and reaction checks through the Houdini HDAs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import hou
import numpy as np

from test_model_oracle_pipeline import (
    MSH_READER, POLYFEM, READ_PVD, UNIT_TET, check_close, exact_definition,
    install_libraries, read_json, stress_columns, vectors,
)


# Physical tags, input x offsets, E in Pa, and final free-apex loads in N.
# Both bodies use nu=1/4, so G=E/2.5.
BODIES = [(17, 0, 30000, 24), (53, 3, 50000, -30)]
TIMES = [0, 0.5, 1]


def body_points(offset):
    return [(offset + x, y, z) for x, y, z in UNIT_TET]


def source_sop(name, order):
    coordinates = [point for _, x, _, _ in BODIES for point in body_points(x)]
    elements = [(BODIES[i][0], [4 * i + j for j in (1, 2, 3, 0)]) for i in order]
    obj = hou.node("/obj").createNode("geo", name, run_init_scripts=False)
    sop = obj.createNode("python", "two_tets")
    sop.parm("python").set(f"""
import hou
geo = hou.pwd().geometry()
points = geo.createPoints({coordinates!r})
tag = geo.addAttrib(hou.attribType.Prim, "Entity", 0)
for body, corners in {elements!r}:
    tet = geo.createTetrahedronInPlace(*(points[i] for i in corners))
    tet.setAttribValue(tag, body)
""")
    assert [prim.attribValue("Entity") for prim in sop.geometry().prims()] == [e[0] for e in elements]
    return sop


def face_prim(node, volume, keep):
    found = [str(prim.number()) for prim in node.node("null_1").geometry().prims()
             if prim.attribValue("Entity") == volume
             and all(keep(vertex.point().position()) for vertex in prim.vertices())]
    prim, = found
    return prim


def add_condition(node, volume, sideset, pattern, kind, value, dims, ramp=False, points=False):
    node.parm(f"sideset_selection1_{volume}").set(sideset)
    node.parm(f"sideset_selection1_{volume}").pressButton()
    suffix = f"1_{volume}_{sideset}"
    node.setParms({f"basegroup{suffix}": pattern, f"grouptype{suffix}": int(points),
                   f"Boundary_Condition__{suffix}": 1,
                   f"boundary_type{suffix}_1": kind,
                   f"vector_{suffix}_1": json.dumps(value),
                   f"time_curve{suffix}_1": int(ramp)})
    for axis, is_fixed in zip("xyz", dims):
        node.parm(f"{axis}_dimension{suffix}_1").set(is_fixed)
    return node.hdaModule().sideset_id(1, volume, sideset)


def setup(work, binary, order):
    work.mkdir()
    sop = source_sop(work.name + "_input", order)
    node = hou.node("/obj").createNode("stevenabramowitch::dev::PolyFEM::2.0", work.name)
    exact_definition(node, POLYFEM)
    node.setParms({"working_dir": str(work) + "/", "polyfem_bin": str(binary),
                   "mesh_source1": "sop", "sop_path1": sop.path(),
                   "enable": 0, "max_threads": 1, "log_level": "info",
                   "units": 1, "length": "m", "mass": "kg", "time": "s",
                   "quasistatic": 1, "end_time_bool": 1, "t0": 0, "tend": 1,
                   "time_inc_bool": 1, "dt": 0.5, "use_hdf5": 0,
                   "tensor_values": 1, "body_ids_fields": 1, "forces_fields": 1,
                   "vis_boundary_only": 0, "sol_at_node": 1})
    assert node.hdaModule().load_sop_mesh({"node": node, "script_multiparm_index": "1"})
    exact_definition(node.node("geo_1"), MSH_READER)
    assert node.evalParm("num_volumes1") == len(BODIES)

    faces = {}
    for volume, (tag, x, young, load) in enumerate(BODIES, 1):
        assert f"physical group {tag}" in node.evalParm(f"subdomain_info1_{volume}")
        token = node.hdaModule().MATERIAL_TOKENS.index("LinearElasticity")
        node.setParms({f"materials1_{volume}": token, f"E1_{volume}": young, f"nu1_{volume}": 0.25})
        specs = [
            (lambda p: p[2] == 0, 0, [0, 0, 0], (1, 1, 1), False, (0, 1, 2)),
            (lambda p: sum(p) - x == 1, 0, [0, 0, 0], (0, 1, 1), False, (1, 2, 3)),
            (lambda p: p[0] == x, 1, [6 * load, 0, 0], (1, 1, 1), True, (0, 2, 3)),
        ]
        for j, (keep, kind, value, dims, ramp, corners) in enumerate(specs, 1):
            pattern = face_prim(node, volume, keep)
            tag = add_condition(node, volume, j, pattern, kind, value, dims, ramp)
            faces[tag] = (sorted(body_points(x)[i] for i in corners), kind, value, dims)
    return node, faces


def msh22_tets(path, count=2):
    # The fixtures stage disconnected tetrahedra in an ASCII Gmsh 2.2 file.
    text = path.read_text()
    assert text.startswith("$MeshFormat\n2.2 0 8\n"), path
    points = {}
    for row in text.split("$Nodes\n")[1].split("$EndNodes")[0].splitlines()[1:]:
        index, *xyz = row.split()
        points[int(index) - 1] = tuple(map(float, xyz))
    elements = []
    for row in text.split("$Elements\n")[1].split("$EndElements")[0].splitlines()[1:]:
        _, element_type, ntags, *data = map(int, row.split())
        assert element_type == 4 and len(data) == ntags + 4, row
        elements.append(sorted(points[index - 1] for index in data[ntags:]))
    assert len(points) == 4 * count and len(elements) == count, (path, points, elements)
    return points, elements


def check_export(work, order, expected_faces):
    input_dir = work / "input"
    params = read_json(input_dir / "params.json")
    assert {unit: params["units"][unit] for unit in ("length", "mass", "time")} == {
        "length": "m", "mass": "kg", "time": "s"}
    assert params["time"]["quasistatic"] and params["time"]["dt"] == 0.5
    assert params["contact"]["enabled"] is False and params["solver"]["max_threads"] == 1
    assert [(m["id"], m["type"], m["E"], m["nu"]) for m in params["materials"]] == [
        (1001 + i, "LinearElasticity", body[2], 0.25) for i, body in enumerate(BODIES)]
    geom, = params["geometry"]
    points, elements = msh22_tets(input_dir / geom["mesh"])
    assert elements == [sorted(body_points(BODIES[i][1])) for i in order], elements
    np.testing.assert_array_equal(
        np.loadtxt(input_dir / geom["volume_selection"], dtype=int, ndmin=1),
        [1001 + i for i in order])
    side, = geom["surface_selection"]
    face_rows = np.loadtxt(input_dir / side["file"], dtype=int, ndmin=2)
    assert len(face_rows) == len(expected_faces), face_rows
    assert {int(tag): sorted(points[v] for v in corners) for tag, *corners in face_rows} == {
        tag: settings[0] for tag, settings in expected_faces.items()}
    conditions = params["boundary_conditions"]
    assert conditions.get("rhs", [0, 0, 0]) == [0, 0, 0], conditions
    for kind, name in enumerate(("dirichlet_boundary", "neumann_boundary")):
        entries = {b["id"]: b for b in conditions[name]}
        assert set(entries) == {tag for tag, value in expected_faces.items() if value[1] == kind}
        for tag, entry in entries.items():
            _, _, value, dims = expected_faces[tag]
            assert entry["value"] == value, entry
            if kind == 0:
                assert tuple(entry.get("dimension", (1, 1, 1))) == dims, entry
            else:
                assert entry["interpolation"] == [
                    {"type": "piecewise_linear", "points": [0, 1], "values": [0, 1], "extend": "constant"}
                ], entry


def run_reader(node, work, binary, identity):
    module = node.hdaModule()
    run = module.run_background({"node": node})
    assert run and Path(run["binary"]).resolve() == binary, run
    process = run["popen"]
    try:
        code = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    assert code == 0, Path(run["log"]).read_text()
    manifest = read_json(work / "output" / "run-manifest.json")
    assert manifest["completion"]["exit_status"] == 0, manifest["completion"]
    assert manifest["build"] == identity["build_info"]
    exe = manifest["process"]["executable"]
    assert Path(exe["path"]).resolve() == binary and exe["sha256"] == identity["solver_sha256"]
    reader = module.open_results({"node": node})
    assert reader is not None
    exact_definition(reader, READ_PVD)
    pvd = Path(reader.evalParm("PVD_file")).resolve()
    assert pvd.parent == work / "output", pvd
    reader.parm("show_deformed").set(0)
    return reader


def check_readback(reader, work):
    module = reader.hdaModule()
    curves = module.force_curves(reader)
    check_close("Read PVD times (s)", curves["times"], TIMES, 1e-12)
    record = read_json(work / "input" / "hda_scene.json")
    reactions = {}
    for volume, (_, _, _, load) in enumerate(BODIES, 1):
        index, = [i for i, definition in enumerate(record["force_sets"], 1)
                  if definition["body"] == 1000 + volume and definition["sideset"] == 1]
        assert curves["nodes"][index] == 3 and curves["unmatched"][index] == 0, curves
        reaction = curves["reaction"][:, curves["sets"].index(index), :]
        check_close(f"body {volume} base reaction (N)", reaction,
                    [(-3 * load * t, 0, 0) for t in TIMES], 1e-8)
        reactions[volume] = reaction[:, 0]

    measurements = []
    for i, time in enumerate(TIMES):
        hou.setFrame(module.entry_frame(reader, i))
        sop = reader.node("OUT_result")
        sop.cook(force=True)
        geo = sop.geometry()
        assert len(geo.points()) == 8, (time, sop.errors())
        positions = vectors(geo, "P")
        body_ids = [int(p.attribValue("body_ids")) for p in geo.points()]
        stress = stress_columns(lambda name: vectors(geo, name), "cauchy_stress")
        for volume, (_, x, young, load) in enumerate(BODIES, 1):
            rows = []
            for point in body_points(x):
                index, = np.flatnonzero(np.linalg.norm(positions - point, axis=1) < 1e-9)
                assert body_ids[index] == 1000 + volume, (point, body_ids[index])
                rows.append(index)
            f = load * time
            tau = 6 * f
            u = tau / (young / 2.5)
            stress_tensor = [[0, 0, tau], [0, 0, 0], [tau, 0, 0]]
            displacement = vectors(geo, "solution")[rows]
            elastic = vectors(geo, "elastic_forces")[rows]
            applied = vectors(geo, "body_forces")[rows]
            label = f"time {time:g} body {volume}"
            check_close(label + " displacement (m)", displacement, [(0, 0, 0)] * 3 + [(u, 0, 0)], 3e-8)
            check_close(label + " Cauchy stress (Pa)", stress[rows], stress_tensor, 3e-5)
            check_close(label + " elastic forces (N)", elastic, [(f, 0, f), (0, 0, -f), (0, 0, 0), (-f, 0, 0)], 3e-5)
            check_close(label + " applied forces (N)", applied, [(f, 0, 0), (0, 0, 0), (f, 0, 0), (f, 0, 0)], 3e-5)
            measurements.append({"time_s": time, "body_id": 1000 + volume,
                                 "apex_u_x_m": float(displacement[3, 0]),
                                 "apex_elastic_x_N": float(elastic[3, 0]),
                                 "sigma_xz_Pa": float(stress[rows[3], 0, 2]),
                                 "base_reaction_x_N": float(reactions[volume][i])})
    (work / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")


def pipeline_inputs(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    binary = args.solver.resolve(strict=True)
    prefs = Path(hou.homeHoudiniDirectory()).resolve()
    assert os.environ.get("HOUDINI_USER_PREF_DIR") and prefs.is_relative_to(root.parent), prefs
    assert not hou.isUIAvailable() and not hou.node("/obj").children()
    root.mkdir()
    info = json.loads(subprocess.check_output([binary, "--build_info"], cwd=root, text=True, timeout=15))
    version = info["sources"]["polyfem"]
    assert version["commit"] == args.expect_commit and not version["dirty"], version
    identity = {"build_info": info, "solver": str(binary),
                "solver_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "hda_sha256": install_libraries(), "houdini": hou.applicationVersionString(),
                "preferences": str(prefs)}
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return root, binary, identity


def main():
    root, binary, identity = pipeline_inputs(__doc__)
    for label, order in (("sop_order_ab", (0, 1)), ("sop_order_ba", (1, 0))):
        work = root / label
        node, expected_faces = setup(work, binary, order)
        reader = run_reader(node, work, binary, identity)
        check_export(work, order, expected_faces)
        check_readback(reader, work)
        print(f"PASS {label}", flush=True)
    print("2/2 Houdini cases passed", flush=True)


if __name__ == "__main__":
    main()
