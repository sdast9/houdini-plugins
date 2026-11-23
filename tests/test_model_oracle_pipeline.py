"""Analytical material checks through the published PolyFEM and Read PVD HDAs.

Run with hython in a fresh preference directory. See model_oracle_pipeline.md.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import hou
import numpy as np


BASE = Path(__file__).resolve().parents[1]
POLYFEM = "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"
READ_PVD = "object_readPVD.1.0.hdanc"
MSH_READER = "sop_MSH_Reader.3.0.hdanc"
BODY_ID = 1001
UNIT_TET = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def exact_definition(node, library):
    actual = Path(node.type().definition().libraryFilePath()).resolve()
    assert actual == (BASE / library).resolve(), (node.path(), actual)


def install_libraries():
    for name in (POLYFEM, MSH_READER, READ_PVD):
        path = BASE / name
        hou.hda.installFile(str(path), oplibraries_file=None, force_use_assets=True)
        definitions = hou.hda.definitionsInFile(str(path))
        assert definitions and all(d.isCurrent() for d in definitions), path
    return {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
            for name in (POLYFEM, MSH_READER, READ_PVD)}


def tetrahedron_sop(label, coords, cells, directors):
    """A deterministic native tetrahedral SOP with Entity and fiber attributes."""
    obj = hou.node("/obj").createNode("geo", label + "_source", run_init_scripts=False)
    sop = obj.createNode("python", "analytical_tets")
    sop.parm("python").set(f"""
import hou
geo = hou.pwd().geometry()
points = geo.createPoints({coords!r})
entity = geo.addAttrib(hou.attribType.Prim, "Entity", 17)
director = geo.addAttrib(hou.attribType.Prim, "director", (0., 0., 0.))
for cell, a in zip({cells!r}, {directors!r}):
    tet = geo.createTetrahedronInPlace(*(points[cell[i]] for i in (1, 2, 3, 0)))
    tet.setAttribValue(entity, 17)
    tet.setAttribValue(director, a)
""")
    assert sop.geometry().countPrimType(hou.primType.Tetrahedron) == len(cells), sop.errors()
    return sop


def new_case(path, binary, sop, displacement):
    path.mkdir()
    node = hou.node("/obj").createNode("stevenabramowitch::dev::PolyFEM::2.0", path.name)
    exact_definition(node, POLYFEM)
    node.setParms({"working_dir": str(path) + "/", "polyfem_bin": str(binary),
                   "mesh_source1": "sop", "sop_path1": sop.path(),
                   "enable": 0, "max_threads": 1, "log_level": "info",
                   "units": 1, "length": "m", "mass": "kg", "time": "s",
                   "quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 1.0,
                   "use_hdf5": 0, "tensor_values": 1, "materials_fields": 1,
                   "body_ids_fields": 1, "forces_fields": 1, "output_json": 1,
                   "vis_boundary_only": 0, "sol_at_node": 1})
    mod = node.hdaModule()
    assert mod.load_sop_mesh({"node": node, "script_multiparm_index": "1"})
    exact_definition(node.node("geo_1"), MSH_READER)
    assert node.evalParm("num_volumes1") == 1
    node.parm("sideset_selection1_1").set(1)
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({"basegroup1_1_1": "*", "grouptype1_1_1": 0,
                   "Boundary_Condition__1_1_1": 1, "boundary_type1_1_1_1": 0,
                   "vector_1_1_1_1": json.dumps(displacement),
                   "x_dimension1_1_1_1": 1, "y_dimension1_1_1_1": 1,
                   "z_dimension1_1_1_1": 1})
    return node


def read_json(path):
    return json.loads(path.read_text())


def run_case(node, work, binary, commit, displacement, element_count):
    mod = node.hdaModule()
    run = mod.run_background({"node": node})
    assert run and Path(run["binary"]).resolve() == binary, run
    process = run["popen"]
    try:
        code = process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    assert code == 0, Path(run["log"]).read_text()

    # Audit what the HDA actually exported and which process consumed it.
    params = read_json(work / "input" / "params.json")
    manifest = read_json(work / "output" / "run-manifest.json")
    record = read_json(work / "output" / ".polyfem_run.json")
    assert Path(record["binary"]).resolve() == binary
    assert manifest["completion"]["exit_status"] == 0, manifest["completion"]
    version = manifest["build"]["sources"]["polyfem"]
    assert version["commit"] == commit and not version["dirty"], version
    assert {unit: params["units"][unit] for unit in ("length", "mass", "time")} == {
        "length": "m", "mass": "kg", "time": "s"}
    material, = params["materials"]
    geometry, = params["geometry"]
    assert material["id"] == BODY_ID
    assert (work / "input" / geometry["mesh"]).is_file()
    np.testing.assert_array_equal(
        np.loadtxt(work / "input" / geometry["volume_selection"], dtype=int, ndmin=1),
        np.full(element_count, BODY_ID))
    constraint, = params["boundary_conditions"]["dirichlet_boundary"]
    assert constraint["value"] == displacement, constraint
    assert all(constraint.get("dimension", [True] * 3))
    sideset, = geometry["surface_selection"]
    faces = np.loadtxt(work / "input" / sideset["file"], dtype=int, ndmin=2)
    assert faces.shape[1] == 4 and np.all(faces[:, 0] == constraint["id"]), faces

    reader = mod.open_results({"node": node})
    assert reader is not None
    exact_definition(reader, READ_PVD)
    pvd = Path(reader.evalParm("PVD_file")).resolve()
    assert pvd.parent == work / "output", pvd
    steps = ET.parse(pvd).findall("./Collection/DataSet")
    assert len(steps) == 2, steps  # initial frame and the one prescribed increment
    frame_path = pvd.parent / steps[-1].attrib["file"]
    if frame_path.suffix == ".vtm":
        volume = ET.parse(frame_path).find("./vtkMultiBlockDataSet/Block[@name='Volume']/DataSet")
        assert volume is not None, frame_path
        frame_path = frame_path.parent / volume.attrib["file"]
    raw = reader.hdaModule().read_mesh_cached(str(frame_path))
    np.testing.assert_array_equal(raw["point_data"]["body_ids"], BODY_ID)
    hou.setFrame(reader.hdaModule().entry_frame(reader, len(steps) - 1))
    result = reader.node("OUT_result")
    result.cook(force=True)
    geo = result.geometry()
    assert geo is not None and geo.points(), result.errors()
    print(f"{work.name}: {frame_path.name}, {len(geo.points())} Read PVD points")
    return material, raw, geo, reader


def vectors(geo, name):
    assert geo.findPointAttrib(name) is not None, (name, [a.name() for a in geo.pointAttribs()])
    return np.asarray(geo.pointFloatAttribValues(name), dtype=np.float64).reshape(len(geo.points()), -1)


def stress_columns(field, stem):
    # VTK and Read PVD provide three column vectors. PK1 is nonsymmetric
    # in the fiber shear case, which makes the column convention observable.
    return np.stack([field(f"{stem}_{i}") for i in (1, 2, 3)], axis=-1)


def check_close(label, actual, expected, atol):
    actual = np.asarray(actual)
    expected = np.broadcast_to(expected, actual.shape)
    error = np.max(np.abs(actual - expected))
    print(f"  {label}: max error {error:.3g}, range [{np.min(actual):.9g}, {np.max(actual):.9g}]")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=label)


def check_axis(label, actual, expected):
    difference = np.linalg.norm(actual - expected, axis=-1)
    flipped = np.linalg.norm(actual + expected, axis=-1)
    check_close(label, np.minimum(difference, flipped), 0, 5e-7)


def check_shear_displacement(geo, gamma, j):
    pos = vectors(geo, "P")
    # In either rest or displayed deformed coordinates, coordinate j is unchanged.
    expected = np.zeros_like(pos)
    expected[:, 0] = gamma * pos[:, j]
    assert np.max(expected[:, 0]) >= gamma * 0.99
    check_close("Read PVD displacement (m)", vectors(geo, "solution"), expected, 5e-8)


def fiber_case(root, binary, commit):
    label = "dispersed_fiber_shear"
    offsets = (0, 3, 6)
    axes = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0]], dtype=float)
    coords = [(x + offset, y, z) for offset in offsets for x, y, z in UNIT_TET]
    cells = [tuple(range(i, i + 4)) for i in range(0, len(coords), 4)]
    sop = tetrahedron_sop(label, coords, cells, [tuple(a.tolist()) for a in axes])
    gamma = 0.25
    bc = ["0.25*y", 0, 0]
    work = root / label
    node = new_case(work, binary, sop, bc)
    mod = node.hdaModule()
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("HGODispersion"),
                   "hgo_k11_1": 100., "hgo_k21_1": 5., "kappa1_1": 0.,
                   "fib_source1_1": "attribute", "fib_sop1_1": sop.path(),
                   "fib_attrib1_1": "director"})
    material, raw, geo, reader = run_case(node, work, binary, commit, bc, len(cells))
    assert material["type"] == "HGODispersion"
    assert (material["k1"], material["k2"], material["kappa"], material.get("k_chi", 100)) == (100, 5, 0, 100)
    ref = material["fiber_direction"]
    assert ref == {"type": "per_element_file", "path": "fibers.vtk", "field": "FIB_1_1"}, ref
    rows = mod.read_cell_vectors_legacy_vtk(str(work / "input" / ref["path"]), ref["field"])
    np.testing.assert_array_equal(rows, axes)
    check_shear_displacement(geo, gamma, 1)

    # F = I + gamma e_x (x) e_y, J = 1. For the full-invariant kappa=0
    # law, s = |F a|^2 - 1. The y-aligned families have s=gamma^2,
    # including the flipped director. The x-aligned family has s=0.
    F = np.eye(3)
    F[0, 1] = gamma
    read_positions = vectors(geo, "P")
    reader.setParms({"show_fibers": 1, "fiber_frame": "deformed", "fiber_stride": 1, "smooth_field": 0})
    reader.hdaModule().autofiber({"node": reader})
    glyphs = reader.node("fiber_only").geometry().prims()
    assert len(glyphs) == len(read_positions)
    by_region = [[] for _ in axes]
    for prim in glyphs:
        endpoints = np.array([tuple(vertex.point().position()) for vertex in prim.vertices()])
        assert endpoints.shape == (2, 3)
        center = endpoints.mean(axis=0)
        nearest = read_positions[np.argmin(np.linalg.norm(read_positions - center, axis=1))]
        assert np.linalg.norm(nearest - center) < 2e-6
        by_region[int(nearest[0] / 3 + 1e-8)].append(prim)

    for region, a in enumerate(axes):
        pushed = F @ a
        s = float(pushed @ pushed) - 1
        chi = 1 / (1 + math.exp(-100 * s))
        e = math.expm1(5 * s * s)
        W = 10 * chi * e
        dW_ds = 100 * chi * s * (e + 1) + 1000 * chi * (1 - chi) * e
        P = 2 * dW_ds * np.outer(pushed, a)
        sigma = P @ F.T

        # The disjoint tets occupy x in [0,1], [3,4], and [6,7]. The shear
        # moves x by at most 1/4 m, so the same masks work in either geometry.
        raw_mask = np.floor(raw["points"][:, 0] / 3 + 1e-8) == region
        mask = np.floor(read_positions[:, 0] / 3 + 1e-8) == region
        assert raw_mask.sum() >= 4 and mask.sum() >= 4
        check_close(f"raw fiber {region} energy (J/m^3)", raw["point_data"]["energy"][raw_mask], W, 1e-10)
        check_close(f"Read PVD fiber {region} energy (J/m^3)", vectors(geo, "energy")[mask], W, 2e-6)
        check_close(f"Read PVD fiber {region} Cauchy stress (Pa)",
                    stress_columns(lambda name: vectors(geo, name)[mask], "cauchy_stress"), sigma, 3e-5)
        check_close(f"Read PVD fiber {region} first Piola stress (Pa)",
                    stress_columns(lambda name: vectors(geo, name)[mask], "pk1_stress"), P, 3e-5)
        read_axis = np.column_stack([vectors(geo, f"fiber_direction_{c}")[mask, 0] for c in "xyz"])
        check_axis(f"Read PVD fiber {region} reference axis", read_axis, a)
        group = by_region[region]
        assert len(group) == mask.sum()
        assert all(p.attribValue("fiber_drawn_frame") == "current_normalized_Fa0" for p in group)
        check_axis(f"Read PVD fiber {region} current axis",
                   np.array([p.attribValue("fiber_current_direction") for p in group]), pushed / np.linalg.norm(pushed))
        check_close(f"Read PVD fiber {region} I4bar", [p.attribValue("fiber_I4bar") for p in group], s + 1, 5e-7)


def hooke_case(root, binary, commit, rotated):
    label = "rotated_hooke_shear" if rotated else "aligned_hooke_control"
    # Four exterior vertices plus a free interior vertex. The exact affine
    # solution follows from constant stress, zero body force, and u on all faces.
    cells = [(4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)]
    sop = tetrahedron_sop(label, UNIT_TET + [(0.25, 0.25, 0.25)], cells, [(1., 0., 0.)] * 4)
    gamma = 0.01
    bc = ["0.01*z", 0, 0]
    work = root / label
    node = new_case(work, binary, sop, bc)
    E = (92e6, 54e6, 36e6)
    nu = (0.24, 0.16, 0.19)
    G = (3.8e6, 8.3e6, 6.1e6)  # Gyz, Gzx, Gxy in material axes
    R = [0, -1, 0, 1, 0, 0, 0, 0, 1] if rotated else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    node.setParms({"materials1_1": node.hdaModule().MATERIAL_TOKENS.index("HookeLinearElasticity"),
                   "material_coord1_1": json.dumps(R)})
    for name, values in (("ortho_E", E), ("ortho_nu", nu), ("ortho_G", G)):
        node.parmTuple(name + "1_1").set(values)
    material, raw, geo, _ = run_case(node, work, binary, commit, bc, len(cells))
    assert material["type"] == "HookeLinearElasticity" and material["fiber_direction"] == R
    np.testing.assert_array_equal(material["elasticity_tensor"], E + nu + G)
    check_shear_displacement(geo, gamma, 2)
    pos = vectors(geo, "P")
    interior = np.isclose(pos[:, 1], 0.25) & np.isclose(pos[:, 2], 0.25)
    assert np.any(interior)
    check_close("Read PVD free interior displacement (m)", vectors(geo, "solution")[interior, 0], 0.0025, 5e-8)

    tau = (G[0] if rotated else G[1]) * gamma
    W = 0.5 * tau * gamma
    sigma = np.zeros((3, 3))
    sigma[0, 2] = sigma[2, 0] = tau
    print(f"  expected sigma_xz = {tau:.0f} Pa, energy = {W:.0f} J/m^3")
    check_close("raw Hooke energy (J/m^3)", raw["point_data"]["energy"], W, 2e-8)
    check_close("Read PVD Hooke energy (J/m^3)", vectors(geo, "energy"), W, 2e-3)
    check_close("Read PVD Hooke Cauchy stress (Pa)",
                stress_columns(lambda name: vectors(geo, name), "cauchy_stress"), sigma, 0.05)

    # The original z=1 apex hat function equals z on the exterior. Splitting
    # the tet at its centroid keeps that boundary trace, so the integral of
    # its gradient stays e_z/6. PolyFEM exports the negative energy gradient.
    apex = np.isclose(pos[:, 1], 0.0) & np.isclose(pos[:, 2], 1.0)
    assert np.any(apex)
    check_close("Read PVD apex elastic force (N)", vectors(geo, "elastic_forces")[apex],
                np.array([-tau / 6.0, 0.0, 0.0]), 5e-3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    binary = args.solver.resolve(strict=True)
    prefs = Path(hou.homeHoudiniDirectory()).resolve()
    assert os.environ.get("HOUDINI_USER_PREF_DIR") and prefs.is_relative_to(root.parent), prefs
    assert not hou.isUIAvailable() and not hou.node("/obj").children()
    root.mkdir()
    info = json.loads(subprocess.check_output([binary, "--build_info"], cwd=root, text=True))
    source = info["sources"]["polyfem"]
    assert source["commit"] == args.expect_commit and not source["dirty"], source
    receipt = {"solver": str(binary), "solver_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
               "build_info": info, "houdini": hou.applicationVersionString(),
               "preferences": str(prefs), "hda_sha256": install_libraries()}
    (root / "identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fiber_case(root, binary, args.expect_commit)
    hooke_case(root, binary, args.expect_commit, rotated=True)
    hooke_case(root, binary, args.expect_commit, rotated=False)
    print("PASS: dispersed fiber, rotated Hooke and aligned orientation control")


if __name__ == "__main__":
    main()
