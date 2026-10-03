"""Read PVD's Export > Deformed Mesh (2026-10-03).

The export writes each simulated geometry's own .msh (the copy the PolyFEM
node staged in <run>/input/) with only its node coordinates moved to one
step's deformed positions, pulled back into the mesh file's coordinates.
Checked with the real solver:

* round trip: a two-subdomain bar (physical tags 7 and 3, named) that the
  PolyFEM node rotates, scales unevenly and translates is pulled and sheared;
  the exported file, picked again on the same geometry of the node (which
  keeps the Transform and the subdomain materials), starts a new run whose
  rest positions are the first run's deformed positions (to 1e-12), with the
  same body ids -- and with zero stress, while the exported step was loaded;
* the file is the staged mesh with only $Nodes changed (tags, elements,
  physical groups and names identical); Simulation coordinates are the
  same shape with the Transform applied;
* Q1 hexahedra, and a binary MSH 4.1 mesh, round trip the same way;
* a P2 run exports straight elements with exact corners and reports the
  edge curvature it drops;
* two simulated geometries and an obstacle: one file per geometry, the
  obstacle skipped, a chosen geometry alone;
* the 2.2 / 4.1, ASCII / binary node writers change nothing but $Nodes;
* refused by name: an inverted element, a missing mesh, a curved mesh,
  sampled output (High Order Mesh off), a run without params.json.

Run: hython tests/test_export_mesh.py
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

import hou

sys.dont_write_bytecode = True
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
sys.path.insert(0, os.path.join(BASE, "src", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msh_parser  # noqa: E402
from test_force_curves import write_hex_box  # noqa: E402
from test_sideset_conditions import write_box  # noqa: E402

COUNT = [0]


def with_names(path, names):
    """Add a $PhysicalNames section (3D groups) to an MSH 2.2 ASCII file."""
    with open(path) as handle:
        text = handle.read()
    block = "$PhysicalNames\n%d\n%s$EndPhysicalNames\n" % (
        len(names), "".join(f'3 {tag} "{name}"\n'
                            for tag, name in sorted(names.items())))
    with open(path, "w") as handle:
        handle.write(text.replace("$EndMeshFormat\n",
                                  "$EndMeshFormat\n" + block, 1))
    return path


def to_v41_binary(source, target):
    """The same mesh as MSH 4.1 binary (one entity block per family/tag)."""
    parsed = msh_parser.read_msh(source)
    points, tags = parsed["points"], parsed["node_tags"]
    cells = parsed["cells"]["tet"]
    entities = sorted(set(int(t) for t in cells["entity"]))
    with open(target, "wb") as handle:
        handle.write(b"$MeshFormat\n4.1 1 8\n" + struct.pack("<i", 1)
                     + b"\n$EndMeshFormat\n")
        handle.write(b"$Entities\n" + struct.pack("<4Q", 0, 0, 0,
                                                  len(entities)))
        for tag in entities:
            handle.write(struct.pack("<i6dQi Q", tag, 0, 0, 0, 1, 1, 1, 1,
                                     tag, 0))
        handle.write(b"\n$EndEntities\n$Nodes\n")
        handle.write(struct.pack("<4Q", 1, len(points), int(tags.min()),
                                 int(tags.max())))
        handle.write(struct.pack("<iiiQ", 3, entities[0], 0, len(points)))
        handle.write(np.asarray(tags, dtype="<u8").tobytes())
        handle.write(np.asarray(points, dtype="<f8").tobytes())
        handle.write(b"\n$EndNodes\n$Elements\n")
        handle.write(struct.pack("<4Q", len(entities), len(cells["entity"]),
                                 1, len(cells["entity"])))
        number = 0
        for tag in entities:
            rows = np.flatnonzero(cells["entity"] == tag)
            handle.write(struct.pack("<iiiQ", 3, tag, 4, len(rows)))
            for row in rows:
                number += 1
                handle.write(struct.pack(
                    "<5Q", number, *tags[cells["corners"][row]]))
        handle.write(b"\n$EndElements\n")
    return target


def make_node(root, label, mesh, transform=None, materials=None):
    """A PolyFEM node on `mesh`, quasistatic, no contact, linear output."""
    COUNT[0] += 1
    work = os.path.join(root, f"{COUNT[0]:02d}_{label}")
    os.makedirs(work)
    node = hou.node("/obj").createNode(TYPE, f"m{COUNT[0]}")
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN,
                   "use_hdf5": 0, "file_location1": mesh})
    node.parm("file_location1").pressButton()
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 0.25, "enable": 0})
    for (name, value) in (transform or {}).items():
        node.parmTuple(name).set(value)
    for vol, young in (materials or {1: 1e5}).items():
        items = node.parm(f"materials1_{vol}").menuItems()
        node.setParms({f"materials1_{vol}": items.index("NeoHookean"),
                       f"E1_{vol}": young, f"nu1_{vol}": 0.3})
    return node, work


def plane(node, height, top):
    """A typed selection of the bar's top (or bottom) faces in simulation
    coordinates: the transformed plane z = height - 0.05 (or 0.05)."""
    matrix = node.hdaModule().rest_matrix(node, 1)
    A, b = matrix[:3, :3].T, matrix[3, :3]
    normal = np.linalg.solve(A.T, np.array([0.0, 0.0, 1.0 if top else -1.0]))
    normal /= np.linalg.norm(normal)
    point = A @ np.array([0.5, 0.5, height - 0.05 if top else 0.05]) + b
    return ("plane:[%r,%r,%r],[%r,%r,%r]"
            % (*(float(v) for v in normal), *(float(v) for v in point)))


def sideset(node, vol, pattern, value):
    j = node.evalParm(f"sideset_selection1_{vol}") + 1
    node.parm(f"sideset_selection1_{vol}").set(j)
    node.parm(f"sideset_selection1_{vol}").pressButton()
    suffix = f"1_{vol}_{j}_1"
    node.setParms({f"basegroup1_{vol}_{j}": pattern,
                   f"Boundary_Condition__1_{vol}_{j}": 1,
                   f"boundary_type{suffix}": 0, f"vector_{suffix}": value})
    for axis in "xyz":
        node.setParms({f"{axis}_dimension{suffix}": 1})


def run(node):
    module = node.hdaModule()
    path = module.write_params_only({"node": node})
    assert path, "export failed"
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "warning", "--max_threads", "1"],
        cwd=os.path.dirname(path), capture_output=True, text=True,
        timeout=1800)
    assert result.returncode == 0, result.stdout[-3000:]
    reader = module.open_results({"node": node})
    assert reader is not None
    return reader


def show_last(reader):
    """Put the playbar on the run's last step (Step on Screen)."""
    module = reader.hdaModule()
    count = len(module.read_pvd(reader.evalParm("PVD_file")))
    hou.setFrame(module.entry_frame(reader, count - 1))
    assert module.entry_index(reader) == count - 1


def export(reader, step=None, coordinates=0, geometries=""):
    if step is None:
        reader.parm("export_mesh_step_mode").set(0)
    else:
        reader.setParms({"export_mesh_step_mode": 1, "export_mesh_step": step})
    reader.setParms({"export_mesh_coordinates": coordinates,
                     "export_mesh_geometries": geometries})
    return reader.hdaModule().export_deformed_mesh(reader)


def refused(reader, words, **options):
    module = reader.hdaModule()
    try:
        export(reader, **options)
    except module.ExportError as exc:
        text = str(exc)
        for word in words:
            assert word in text, (word, text)
        return text
    raise AssertionError(f"not refused (expected {words})")


def nodes_section_span(data):
    sections = msh_parser._find_sections(data)
    return sections["Nodes"]


def same_outside_nodes(original, exported):
    with open(original, "rb") as handle:
        a = handle.read()
    with open(exported, "rb") as handle:
        b = handle.read()
    sa, sb = nodes_section_span(a), nodes_section_span(b)
    return a[:sa[0]] == b[:sb[0]] and a[sa[1]:] == b[sb[1]:]


def last_volume(reader, step=-1):
    module = reader.hdaModule()
    path = reader.evalParm("PVD_file")
    entries = module.read_pvd(path)
    return module.volume_mesh(path, step % len(entries)), len(entries)


def corner_rows(mesh, module):
    topology = module.volume_topology(mesh)
    return np.unique(topology["corners"][topology["corners"] >= 0])


def check_round_trip(root):
    mesh = with_names(write_box(
        os.path.join(root, "bar_src.msh"), n=2, hi=(1.0, 1.0, 2.0),
        tag_of=lambda x, y, z: 7 if z < 1.0 else 3),
        {3: "upper", 7: "lower"})
    transform = {"xform_t__1": (0.1, -0.2, 0.3), "xform_r__1": (10, 20, 30),
                 "xform_s__1": (1.5, 0.8, 1.2)}
    node, work = make_node(root, "round_trip", mesh, transform,
                           {1: 1e5, 2: 3e5})
    # subdomain 1 = group 3 'upper', 2 = group 7 'lower' (sorted tags)
    top, bottom = plane(node, 2.0, True), plane(node, 2.0, False)
    sideset(node, 2, bottom, "[0, 0, 0]")
    sideset(node, 1, top, '["0.05*t", "-0.02*t", "0.2*t"]')
    reader = run(node)
    module = reader.hdaModule()
    first_pvd = reader.evalParm("PVD_file")
    old, count = last_volume(reader)
    written, report = export(reader, step=count - 1)
    assert len(written) == 1 and written[0].endswith("_step004.msh"), written
    assert "none inverted" in report and "no stress" in report, report
    assert os.path.dirname(written[0]) == os.path.join(work, "exports")
    staged = os.path.join(work, "input",
                          json.load(open(os.path.join(
                              work, "input", "params.json")))
                          ["geometry"][0]["mesh"])
    assert same_outside_nodes(staged, written[0])
    before, after = msh_parser.read_msh(staged), msh_parser.read_msh(
        written[0])
    assert np.array_equal(before["node_tags"], after["node_tags"])
    assert np.array_equal(before["cells"]["tet"]["corners"],
                          after["cells"]["tet"]["corners"])
    assert np.array_equal(before["cells"]["tet"]["entity"],
                          after["cells"]["tet"]["entity"])
    assert after["physical_names"] == {(3, 3): "upper", (3, 7): "lower"}
    print("PASS: the deformed mesh is the staged mesh with only $Nodes "
          "changed: node tags, elements, physical groups and names kept")

    # Simulation coordinates: the same shape with the Transform applied
    params = json.load(open(os.path.join(work, "input", "params.json")))
    A, b = module.polyfem_rest_transform(params["geometry"][0],
                                         before["points"])
    written_sim, _ = export(reader, step=count - 1, coordinates=1)
    assert written_sim[0].endswith("_step004_simulation.msh"), written_sim
    simulated = msh_parser.read_msh(written_sim[0])["points"]
    assert np.allclose(simulated, after["points"] @ A.T + b, rtol=0,
                       atol=1e-12 * np.abs(simulated).max())

    # pick the exported file on the same geometry, in a new folder
    young = (node.evalParm("E1_1"), node.evalParm("E1_2"))
    rest_before = node.hdaModule().rest_matrix(node, 1)
    second = os.path.join(root, "round_trip_second")
    os.makedirs(second)
    node.setParms({"working_dir": second + "/",
                   "file_location1": written[0]})
    node.parm("file_location1").pressButton()
    assert (node.evalParm("E1_1"), node.evalParm("E1_2")) == young
    assert np.allclose(node.hdaModule().rest_matrix(node, 1), rest_before)
    for j in range(1, node.evalParm("sideset_selection1_1") + 1):
        node.setParms({f"vector_1_1_{j}_1": "[0, 0, 0]"})
    node.setParms({"tend": 0.25})
    new_reader = run(node)
    new, _ = last_volume(new_reader, 0)
    fe = np.asarray(old["point_data"]["body_ids"]).ravel() > 0
    deformed = old["points"] + old["point_data"]["solution"]
    corners = corner_rows(old, module)
    scale = np.abs(deformed[corners]).max()
    gap = np.abs(new["points"][corners] - deformed[corners]).max()
    assert gap <= 1e-12 * scale, gap
    assert np.array_equal(np.asarray(new["point_data"]["body_ids"])[fe],
                          np.asarray(old["point_data"]["body_ids"])[fe])
    stress_new = np.abs(module._mesh_field(new, "von_mises_derived")).max()
    stress_old = np.abs(module._mesh_field(old, "von_mises_derived")).max()
    assert stress_old > 1e3 and stress_new < 1e-9 * stress_old, \
        (stress_old, stress_new)
    print("PASS: picked again on the same geometry (Transform and subdomain "
          "materials kept), the exported shape starts a new run at the "
          f"deformed positions (gap {gap:.2e} of {scale:.3g}), same body ids, "
          f"and stress-free (von Mises {stress_new:.1e} vs {stress_old:.4g})")
    # Open Results may reuse that Read PVD node for the second run: give
    # the first run its own reader
    first = hou.node("/obj").createNode("readPVD::1.0", "first_run")
    first.parm("PVD_file").set(first_pvd)
    first.hdaModule().start({"node": first})
    return first


def check_hexes_and_formats(root):
    mesh = write_hex_box(os.path.join(root, "hex_src.msh"), n=2,
                         hi=(1.0, 1.0, 2.0))
    node, work = make_node(root, "hexes", mesh,
                           {"xform_r__1": (0, 45, 10), "xform_s__1": (2, 2, 2)})
    sideset(node, 1, plane(node, 2.0, False), "[0, 0, 0]")
    sideset(node, 1, plane(node, 2.0, True), '["0.1*t", "0", "0.2*t"]')
    reader = run(node)
    old, count = last_volume(reader)
    show_last(reader)
    written, report = export(reader)
    assert written[0].endswith(f"_step{count - 1:03d}.msh"), written
    second = os.path.join(root, "hexes_second")
    os.makedirs(second)
    node.setParms({"working_dir": second + "/", "file_location1": written[0],
                   "tend": 0.25})
    node.parm("file_location1").pressButton()
    for j in range(1, node.evalParm("sideset_selection1_1") + 1):
        node.setParms({f"vector_1_1_{j}_1": "[0, 0, 0]"})
    new, _ = last_volume(run(node), 0)
    deformed = old["points"] + old["point_data"]["solution"]
    gap = np.abs(new["points"] - deformed)[
        np.asarray(old["point_data"]["body_ids"]).ravel() > 0].max()
    assert gap <= 1e-12 * np.abs(deformed).max(), gap
    print(f"PASS: Q1 hexahedra round trip (gap {gap:.2e})")

    binary = to_v41_binary(write_box(os.path.join(root, "v41_src.msh"), n=2,
                                     hi=(1.0, 1.0, 2.0)),
                           os.path.join(root, "bar_v41_binary.msh"))
    node, work = make_node(root, "v41_binary", binary,
                           {"xform_r__1": (0, 0, 30)})
    sideset(node, 1, plane(node, 2.0, False), "[0, 0, 0]")
    sideset(node, 1, plane(node, 2.0, True), '["0", "0.05*t", "0.1*t"]')
    reader = run(node)
    old, count = last_volume(reader)
    show_last(reader)
    written, _ = export(reader)
    staged = os.path.join(work, "input", "bar_v41_binary.msh")
    assert same_outside_nodes(staged, written[0])
    with open(written[0], "rb") as handle:
        assert handle.read(30).startswith(b"$MeshFormat\n4.1 1 8")
    second = os.path.join(root, "v41_second")
    os.makedirs(second)
    node.setParms({"working_dir": second + "/", "file_location1": written[0],
                   "tend": 0.25})
    node.parm("file_location1").pressButton()
    for j in range(1, node.evalParm("sideset_selection1_1") + 1):
        node.setParms({f"vector_1_1_{j}_1": "[0, 0, 0]"})
    new, _ = last_volume(run(node), 0)
    deformed = old["points"] + old["point_data"]["solution"]
    gap = np.abs(new["points"] - deformed)[
        np.asarray(old["point_data"]["body_ids"]).ravel() > 0].max()
    assert gap <= 1e-12 * np.abs(deformed).max(), gap
    print(f"PASS: a binary MSH 4.1 mesh is written back as binary 4.1 with "
          f"only its coordinates changed, and round trips (gap {gap:.2e})")


def check_p2(root):
    mesh = write_box(os.path.join(root, "p2_src.msh"), n=2, hi=(1, 1, 2))
    node, work = make_node(root, "p2", mesh)
    node.parm("mainOrder1_1").set(1)  # order 2
    sideset(node, 1, "axis:-z:0.01", "[0, 0, 0]")
    sideset(node, 1, "axis:+z:1.99", '["0.3*t", "0", "0.4*t"]')
    reader = run(node)
    module = reader.hdaModule()
    old, count = last_volume(reader)
    show_last(reader)
    written, report = export(reader)
    assert "Higher-order run" in report and "% of an edge" in report, report
    exported = msh_parser.read_msh(written[0])
    staged = msh_parser.read_msh(os.path.join(work, "input", "p2_src.msh"))
    assert exported["cells"]["tet"]["num_nodes"] == 4
    topology = module.volume_topology(old)
    match = module.mesh_match(reader.evalParm("PVD_file"), old, topology)
    corners = np.flatnonzero(match["point_file_node"] >= 0)
    expected = old["points"][corners] + old["point_data"]["solution"][corners]
    got = exported["points"][match["point_file_node"][corners]]
    assert np.allclose(got, expected, rtol=0, atol=1e-12)
    assert np.array_equal(exported["node_tags"], staged["node_tags"])
    print("PASS: a P2 run exports straight (4-node) elements with exact "
          "corners and reports the edge curvature it drops: "
          + report.splitlines()[2].strip())


def check_geometries(root):
    first = write_box(os.path.join(root, "left.msh"), n=2, hi=(1, 1, 1))
    second = write_box(os.path.join(root, "right.msh"), n=2,
                       lo=(2.0, 0.0, 0.0), hi=(3.0, 1.0, 1.0))
    plate = os.path.join(root, "plate.obj")
    with open(plate, "w") as handle:
        for x, y in ((-0.5, -0.5), (3.5, -0.5), (3.5, 1.5), (-0.5, 1.5)):
            handle.write(f"v {x} {y} 1.5\n")
        handle.write("f 1 3 2\nf 1 4 3\n")
    node, work = make_node(root, "geometries", first)
    node.setParms({"num_geos": 3, "geo_int": 3, "file_location2": second})
    node.parm("file_location2").pressButton()
    node.setParms({"is_obstacle3": 1, "file_location3": plate})
    node.parm("file_location3").pressButton()
    node.setParms({"materials2_1": node.evalParm("materials1_1"),
                   "E2_1": 1e5, "nu2_1": 0.3})
    for geo in (1, 2):
        j = node.evalParm(f"sideset_selection{geo}_1") + 1
        node.parm(f"sideset_selection{geo}_1").set(j)
        node.parm(f"sideset_selection{geo}_1").pressButton()
        suffix = f"{geo}_1_{j}_1"
        node.setParms({f"basegroup{geo}_1_{j}": "axis:-z:0.01",
                       f"Boundary_Condition__{geo}_1_{j}": 1,
                       f"boundary_type{suffix}": 0,
                       f"vector_{suffix}": '["0", "0", "0"]'})
        j += 1
        node.parm(f"sideset_selection{geo}_1").set(j)
        node.parm(f"sideset_selection{geo}_1").pressButton()
        suffix = f"{geo}_1_{j}_1"
        node.setParms({f"basegroup{geo}_1_{j}": "axis:+z:0.99",
                       f"Boundary_Condition__{geo}_1_{j}": 1,
                       f"boundary_type{suffix}": 0,
                       f"vector_{suffix}": '["0", "0", "0.1*t"]'})
    reader = run(node)
    show_last(reader)
    written, report = export(reader)
    names = sorted(os.path.basename(p) for p in written)
    assert names == ["left_step004.msh", "right_step004.msh"], names
    written, _ = export(reader, geometries="2")
    assert [os.path.basename(p) for p in written] == ["right_step004.msh"]
    old, _ = last_volume(reader)
    moved = msh_parser.read_msh(written[0])["points"]
    assert abs(moved[:, 2].max() - 1.1) < 1e-9, moved[:, 2].max()
    print("PASS: two simulated geometries give one file each, the obstacle "
          "is skipped, and Geometries picks one")


def check_writers(root):
    """The four node layouts: only $Nodes changes, and it reads back."""
    source = write_box(os.path.join(root, "w22.msh"), n=2)
    parsed = msh_parser.read_msh(source)
    moved = parsed["points"] * 1.5 + 0.25
    variants = {"2.2 ascii": source,
                "4.1 binary": to_v41_binary(source,
                                            os.path.join(root, "w41b.msh"))}
    # 2.2 binary and 4.1 ascii, written from the parsed mesh
    path = os.path.join(root, "w22b.msh")
    with open(path, "wb") as handle:
        handle.write(b"$MeshFormat\n2.2 1 8\n" + struct.pack("<i", 1)
                     + b"\n$EndMeshFormat\n$Nodes\n%d\n" % len(moved))
        for tag, xyz in zip(parsed["node_tags"], parsed["points"]):
            handle.write(struct.pack("<i3d", int(tag), *xyz))
        cells = parsed["cells"]["tet"]
        handle.write(b"\n$EndNodes\n$Elements\n%d\n" % len(cells["entity"]))
        handle.write(struct.pack("<iii", 4, len(cells["entity"]), 2))
        for number, corners in enumerate(cells["corners"], 1):
            handle.write(struct.pack("<7i", number, 1, 1,
                                     *parsed["node_tags"][corners]))
        handle.write(b"\n$EndElements\n")
    variants["2.2 binary"] = path
    path = os.path.join(root, "w41a.msh")
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Entities\n"
                     "0 0 0 1\n1 0 0 0 1 1 1 0 0\n$EndEntities\n$Nodes\n")
        handle.write(f"1 {len(moved)} 1 {len(moved)}\n3 1 0 {len(moved)}\n")
        handle.write("".join(f"{int(t)}\n" for t in parsed["node_tags"]))
        handle.write("".join(f"{x!r} {y!r} {z!r}\n" for x, y, z in
                             parsed["points"].tolist()))
        cells = parsed["cells"]["tet"]
        handle.write(f"$EndNodes\n$Elements\n1 {len(cells['entity'])} 1 "
                     f"{len(cells['entity'])}\n3 1 4 {len(cells['entity'])}\n")
        for number, corners in enumerate(cells["corners"], 1):
            handle.write(f"{number} " + " ".join(
                str(int(t)) for t in parsed["node_tags"][corners]) + "\n")
        handle.write("$EndElements\n")
    variants["4.1 ascii"] = path
    for name, path in variants.items():
        out = msh_parser.write_moved_nodes(path, moved, path + ".moved.msh")
        back = msh_parser.read_msh(out)
        assert np.array_equal(back["points"], moved), name
        assert np.array_equal(back["node_tags"], parsed["node_tags"]), name
        assert np.array_equal(back["cells"]["tet"]["corners"],
                              parsed["cells"]["tet"]["corners"]), name
        assert same_outside_nodes(path, out), name
        try:
            msh_parser.write_moved_nodes(path, moved[:-1], path + ".bad")
        except msh_parser.MshParseError:
            pass
        else:
            raise AssertionError(f"{name}: wrong node count accepted")
    print("PASS: MSH 2.2 / 4.1, ASCII / binary: only $Nodes changes and the "
          "moved nodes read back exactly; a wrong node count is refused")


def check_refusals(root, reader):
    module = reader.hdaModule()
    pvd = reader.evalParm("PVD_file")
    work = os.path.dirname(os.path.dirname(pvd))
    # an inverted element: one node pushed through its neighbours
    mesh, count = last_volume(reader)
    topology = module.volume_topology(mesh)
    match = module.mesh_match(pvd, mesh, topology)
    solution = np.zeros((len(mesh["points"]), 3))
    centre = int(np.argmin(np.linalg.norm(
        mesh["points"][topology["node_point"]]
        - mesh["points"][topology["node_point"]].mean(axis=0), axis=1)))
    # every copy of that mesh node (it may sit on the subdomain interface,
    # where each body has its own copies)
    file_node = match["point_file_node"][topology["node_point"][centre]]
    copies = np.flatnonzero(match["point_file_node"] == file_node)
    solution[copies] = [0.0, 0.0, 50.0]
    try:
        module.deformed_mesh_positions(match["geometries"][0], match,
                                       topology, solution, "mesh")
    except module.ExportError as exc:
        assert "inverted" in str(exc) and "elements" in str(exc), exc
    else:
        raise AssertionError("an inverted element was accepted")

    # copies of the run: the mesh missing / curved; no params.json
    def copy_run(label):
        target = os.path.join(root, label)
        shutil.copytree(work, target)
        copy = hou.node("/obj").createNode("readPVD::1.0", label)
        copy.parm("PVD_file").set(os.path.join(target, "output", "sim.pvd"))
        copy.hdaModule().start({"node": copy})
        return copy, target

    copy, target = copy_run("missing_mesh")
    params = json.load(open(os.path.join(target, "input", "params.json")))
    mesh_name = params["geometry"][0]["mesh"]
    os.remove(os.path.join(target, "input", mesh_name))
    refused(copy, ["missing", mesh_name])

    copy, target = copy_run("curved_mesh")
    staged = os.path.join(target, "input", mesh_name)
    parsed = msh_parser.read_msh(staged)
    with open(staged, "w") as handle:  # the same corners as 10-node tets
        tags = parsed["node_tags"]
        points = parsed["points"]
        tets = parsed["cells"]["tet"]
        extra = {}
        lines = []
        for corners in tets["corners"]:
            ids = list(tags[corners])
            for a, b in ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)):
                key = tuple(sorted((ids[a], ids[b])))
                if key not in extra:
                    extra[key] = int(tags.max()) + 1 + len(extra)
                ids.append(extra[key])
            lines.append(ids)
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        handle.write(f"{len(points) + len(extra)}\n")
        position = dict(zip(tags.tolist(), points.tolist()))
        for tag, xyz in position.items():
            handle.write(f"{tag} {xyz[0]!r} {xyz[1]!r} {xyz[2]!r}\n")
        for (a, b), tag in extra.items():
            mid = (np.asarray(position[a]) + np.asarray(position[b])) / 2
            handle.write(f"{tag} {mid[0]!r} {mid[1]!r} {mid[2]!r}\n")
        handle.write(f"$EndNodes\n$Elements\n{len(lines)}\n")
        for number, (ids, tag) in enumerate(zip(lines, tets["entity"]), 1):
            handle.write(f"{number} 11 2 {tag} {tag} "
                         + " ".join(map(str, ids)) + "\n")
        handle.write("$EndElements\n")
    refused(copy, ["curved"])

    copy, target = copy_run("no_params")
    os.remove(os.path.join(target, "input", "params.json"))
    refused(copy, ["params.json"])

    # sampled output: High Order Mesh off
    mesh = write_box(os.path.join(root, "sampled_src.msh"), n=2)
    node, _ = make_node(root, "sampled", mesh)
    node.parm("mainOrder1_1").set(1)
    node.parm("high_order_mesh").set(0)
    sideset(node, 1, "axis:-z:0.01", "[0, 0, 0]")
    sideset(node, 1, "axis:+z:0.99", '["0", "0", "0.1*t"]')
    refused(run(node), ["sampled"])

    # the button: the report says why nothing was written
    reader.parm("export_mesh_step_mode").set(1)
    reader.parm("export_mesh_step").set(99)
    reader.parm("export_mesh").pressButton()
    assert reader.evalParm("export_mesh_report").startswith(
        "Not exported: Step 99 does not exist"), \
        reader.evalParm("export_mesh_report")
    reader.parm("export_mesh_step").set(count - 1)
    reader.parm("export_mesh").pressButton()
    assert "written to" in reader.evalParm("export_mesh_report")
    print("PASS: refused by name: an inverted element, a missing mesh, a "
          "curved mesh, a run without params.json, sampled output; the "
          "button reports what it wrote or why not")


def check_interface(reader):
    templates = []

    def walk(entries):
        for template in entries:
            templates.append(template)
            if isinstance(template, hou.FolderParmTemplate):
                walk(template.parmTemplates())

    walk(reader.parmTemplateGroup().entries())
    tabs = [t for t in templates if t.type() == hou.parmTemplateType.Folder
            and t.label() == "Export"]
    assert len(tabs) == 1, "no Export tab"
    templates.clear()
    walk(tabs[0].parmTemplates())
    # (folders, multiparm blocks included, cannot carry a tooltip)
    missing = [t.name() for t in templates
               if t.type() != hou.parmTemplateType.Folder and not t.help()]
    assert not missing, f"no tooltip: {missing}"
    reader.updateParmStates()
    assert reader.parm("export_mesh_report").isDisabled()
    print(f"PASS: all {len(templates)} Export tab parameters have a tooltip")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    for library in ("sop_MSH_Reader.3.0.hdanc",
                    "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                    "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, library),
                            force_use_assets=True)
    hou.setFps(24)
    root = tempfile.mkdtemp(prefix="readpvd_export_mesh_")
    check_writers(root)
    reader = check_round_trip(root)
    check_interface(reader)
    check_hexes_and_formats(root)
    check_p2(root)
    check_geometries(root)
    check_refusals(root, reader)
    print("workdir:", root)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
