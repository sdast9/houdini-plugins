"""Sideset and boundary-condition export: what PolyFEM actually applies.

PolyFEM gives every boundary face ONE boundary id (the first matching
selection, the first matching row of a selection file) and honours one
dirichlet entry and the first load entry of each kind per id. Until 2026-10-01
the asset gave every condition its own id, so a second condition on a sideset,
or a later sideset overlapping an earlier one, silently did nothing (exit 0).
This test runs the real binary and checks, by comparing displacements with
reference scenes, that every condition the user attaches to a face acts on it.
It also covers PolyFEM's vertex numbering for .msh files whose node tags are
not 1..N, mesh staging under clashing file names, the confirmation before an
earlier run's results are replaced, and the import of these scenes (with and
without the asset's sideset record, and from the old per-condition format).

Run: hython tests/test_sideset_conditions.py
No gmsh needed: the meshes are written here.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_box(path, n=3, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0),
              tags="contiguous", tag_of=None):
    """Structured tet box as MSH 2.2 ASCII; node tags 1..N, offset by 100,
    with gaps (2, 4, 6, ...) or 1..N written in shuffled file order."""
    xs = [np.linspace(lo[a], hi[a], n + 1) for a in range(3)]

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i

    points = np.array([(x, y, z) for z in xs[2] for y in xs[1]
                       for x in xs[0]])
    cube = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
            (0, 4, 5, 6), (0, 5, 1, 6)]
    tets, physical = [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                     vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                     vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)]
                centre = points[c].mean(axis=0)
                for t in cube:
                    a, b, cc, d = (c[v] for v in t)
                    if np.linalg.det(np.array([points[b] - points[a],
                                               points[cc] - points[a],
                                               points[d] - points[a]])) < 0:
                        b, cc = cc, b
                    tets.append((a, b, cc, d))
                    physical.append(tag_of(*centre) if tag_of else 1)
    count = len(points)
    if tags == "offset":
        tag = np.arange(count) + 101
    elif tags == "gaps":
        tag = 2 * np.arange(count) + 2
    else:
        tag = np.arange(count) + 1
    order = np.arange(count)
    if tags == "shuffled":
        order = np.random.default_rng(7).permutation(count)
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{count}\n")
        for index in order:
            x, y, z = points[index]
            handle.write(f"{tag[index]} {x:.17g} {y:.17g} {z:.17g}\n")
        handle.write(f"$EndNodes\n$Elements\n{len(tets)}\n")
        for number, (t, group) in enumerate(zip(tets, physical), 1):
            handle.write(f"{number} 4 2 {group} {group} "
                         + " ".join(str(tag[v]) for v in t) + "\n")
        handle.write("$EndElements\n")
    return path


class Scene:
    """One PolyFEM node with a box mesh, quasistatic, one step, no contact."""

    count = 0

    def __init__(self, root, label, tags="contiguous", n=3, tag_of=None):
        Scene.count += 1
        self.work = os.path.join(root, f"{Scene.count:02d}_{label}")
        os.makedirs(self.work)
        mesh = write_box(os.path.join(self.work, "box.msh"), n=n, tags=tags,
                         tag_of=tag_of)
        self.node = hou.node("/obj").createNode(TYPE, f"s{Scene.count}")
        self.mod = self.node.hdaModule()
        self.node.setParms({"working_dir": self.work + "/",
                            "polyfem_bin": POLYFEM_BIN})
        self.node.setParms({"file_location1": mesh})
        self.node.parm("file_location1").pressButton()
        self.node.setParms({
            "quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
            "time_inc_bool": 1, "dt": 1.0, "enable": 0,
            "materials1_1": 0, "E1_1": 1e5, "nu1_1": 0.3})

    def faces(self, predicate, vol=1):
        """Surface prim numbers of subdomain `vol` whose centre passes."""
        surface = self.node.node("null_1").geometry()
        out = []
        for prim in surface.prims():
            if prim.attribValue("Entity") != vol:
                continue
            centre = sum((v.point().position() for v in prim.vertices()),
                         hou.Vector3()) / len(prim.vertices())
            if predicate(*centre):
                out.append(prim.number())
        return " ".join(str(p) for p in out)

    def points(self, predicate):
        surface = self.node.node("null_1").geometry()
        return " ".join(str(p.number()) for p in surface.points()
                        if predicate(*p.position()))

    def sideset(self, pattern, *conditions, vol=1, points=False):
        """conditions: (kind, text, dims) with kind 0..4 (Dirichlet ..
        Pressure Cavity); text is the Vector (0, 1) or Value (2-4)."""
        node = self.node
        j = node.evalParm(f"sideset_selection1_{vol}") + 1
        node.parm(f"sideset_selection1_{vol}").set(j)
        node.parm(f"sideset_selection1_{vol}").pressButton()
        node.setParms({f"basegroup1_{vol}_{j}": pattern,
                       f"grouptype1_{vol}_{j}": 1 if points else 0,
                       f"Boundary_Condition__1_{vol}_{j}": len(conditions)})
        for k, (kind, text, dims) in enumerate(conditions, 1):
            parms = {f"boundary_type1_{vol}_{j}_{k}": kind}
            parms[(f"vector_1_{vol}_{j}_{k}" if kind in (0, 1)
                   else f"value_1_{vol}_{j}_{k}")] = text
            for axis, flag in zip("xyz", dims):
                parms[f"{axis}_dimension1_{vol}_{j}_{k}"] = flag
            node.setParms(parms)
        return j

    def write(self):
        path = self.mod.write_params_only({"node": self.node})
        assert path, "export failed"
        with open(path) as handle:
            return path, json.load(handle)

    def run(self):
        path, data = self.write()
        result = subprocess.run(
            [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
             "--log_level", "info", "--max_threads", "1"],
            cwd=os.path.dirname(path), capture_output=True, text=True,
            timeout=900)
        linf = [float(m.group(1)) for m in re.finditer(
            r"-- Linf error: ([0-9.e+-]+)", result.stdout)]
        assert result.returncode == 0, (
            f"PolyFEM failed in {self.work}:\n{result.stdout[-1500:]}"
            f"\n{result.stderr[-800:]}")
        return linf[-1], data


ALL, X, Z, XZ = (1, 1, 1), (1, 0, 0), (0, 0, 1), (1, 0, 1)
FIXED = (0, "[0, 0, 0]", ALL)
PUSH = (0, '[0, 0, "-0.1*t"]', ALL)


def bottom(scene):
    return scene.sideset("axis:-z:0.01", FIXED)


def expect_error(text, action):
    try:
        action()
    except hou.Error as exc:
        assert text in str(exc), f"expected '{text}' in: {exc}"
        return str(exc)
    raise AssertionError(f"no error containing '{text}'")


def face_conditions(work, data):
    """{sorted vertex numbers: conditions that act on that face} (ids
    replaced by the entries they carry), to compare two exports."""
    by_id = {}
    for key, entries in data.get("boundary_conditions", {}).items():
        if key in ("rhs", "obstacle_displacements"):
            continue
        for entry in entries:
            rest = {k: v for k, v in entry.items() if k != "id"}
            by_id.setdefault(entry["id"], []).append(
                (key, json.dumps(rest, sort_keys=True)))
    mapping = {}
    for selection in data["geometry"][0].get("surface_selection", []):
        rows = np.loadtxt(os.path.join(work, "input", selection["file"]),
                          dtype=np.int64, ndmin=2)
        for row in rows:
            mapping[tuple(sorted(row[1:].tolist()))] = sorted(
                by_id.get(int(row[0]), []))
    return mapping


def check_conditions_on_one_sideset(root):
    # reference: one Dirichlet that holds x and pushes z
    ref = Scene(root, "ref_xz")
    bottom(ref)
    ref.sideset("axis:+z:0.99", (0, '[0, 0, "-0.1*t"]', XZ))
    expected, _ = ref.run()
    assert expected > 0.05, expected

    # the same, entered as two conditions on one sideset
    two = Scene(root, "two_dirichlet")
    bottom(two)
    j = two.sideset("axis:+z:0.99", (0, "[0, 0, 0]", X),
                    (0, '[0, 0, "-0.1*t"]', Z))
    linf, data = two.run()
    assert linf == expected, (linf, expected)
    sid = two.mod.sideset_id(1, 1, j)
    dirichlet = [d for d in data["boundary_conditions"]["dirichlet_boundary"]
                 if d["id"] == sid]
    assert dirichlet == [{"id": sid, "value": [0, 0, "-0.1*t"],
                          "dimension": [True, False, True]}], dirichlet
    rows = np.loadtxt(os.path.join(two.work, "input",
                                   "surface_sidesets1_tri.txt"),
                      dtype=np.int64, ndmin=2)
    assert set(rows[:, 0]) == {two.mod.sideset_id(1, 1, 1), sid}, \
        set(rows[:, 0])
    print("PASS: two Dirichlet conditions on one sideset both act "
          f"(Linf {linf:.6g} = one combined condition)")

    # a Dirichlet and a Neumann on one sideset act together, exactly as
    # when the two sit on two sidesets with the same faces
    load = (1, "[0, 0, -2000]", ALL)
    same = Scene(root, "dirichlet_neumann_one")
    bottom(same)
    j = same.sideset("axis:+z:0.99", (0, "[0, 0, 0]", X), load)
    linf_one, data = same.run()
    sid = same.mod.sideset_id(1, 1, j)
    kinds = sorted(key for key, entries in data["boundary_conditions"].items()
                   if key not in ("rhs", "obstacle_displacements")
                   for entry in entries if entry["id"] == sid)
    assert kinds == ["dirichlet_boundary", "neumann_boundary"], kinds
    split = Scene(root, "dirichlet_neumann_two")
    bottom(split)
    split.sideset("axis:+z:0.99", (0, "[0, 0, 0]", X))
    split.sideset("axis:+z:0.99", load)
    linf_two, _ = split.run()
    assert linf_one == linf_two and linf_one > 1e-3, (linf_one, linf_two)
    print("PASS: Dirichlet + Neumann on one sideset = on two identical "
          f"sidesets (Linf {linf_one:.6g})")

    # two loads of one kind are added
    added = Scene(root, "two_neumann")
    bottom(added)
    j = added.sideset("axis:+z:0.99", (1, "[0, 0, -1000]", ALL),
                      (1, '[0, 0, "-1000*t"]', ALL))
    _, data = added.write()
    sid = added.mod.sideset_id(1, 1, j)
    neumann = [n for n in data["boundary_conditions"]["neumann_boundary"]
               if n["id"] == sid]
    assert len(neumann) == 1, neumann
    assert neumann[0]["value"][2] == "(-1000.0) + (-1000*t)", neumann
    print("PASS: two Neumann loads on one sideset are added")


def check_overlapping_sidesets(root):
    control = Scene(root, "control")
    bottom(control)
    top = control.faces(lambda x, y, z: z > 0.99)
    control.sideset(top, PUSH)
    expected, _ = control.run()

    overlap = Scene(root, "overlap")
    bottom(overlap)
    overlap.sideset("*", (1, "[0, 0, 0]", ALL))
    overlap.sideset(top, PUSH)
    linf, data = overlap.run()
    assert linf == expected, (linf, expected)
    plan = overlap.mod.resolve_sidesets(overlap.node, 1, 1)
    owners = sorted(map(tuple, map(tuple, (
        sorted(map(tuple, o)) for o in plan["combinations"].values()))))
    assert ((1, 1), (1, 2)) in owners and ((1, 2), (1, 3)) in owners, owners
    report = overlap.node.evalParm("export_report")
    assert "receive the conditions of all of them" in report, report
    print(f"PASS: a picked sideset inside a '*' sideset keeps its push "
          f"(Linf {linf:.6g} = control)")


def check_refusals(root):
    scene = Scene(root, "refusals")
    bottom(scene)
    j = scene.sideset("axis:+z:0.99", (0, '[0, 0, "-0.1*t"]', Z),
                      (0, '[0, 0, "-0.2*t"]', Z))
    expect_error("both prescribe the z displacement",
                 lambda: scene.mod.build_params(scene.node))
    # the same value spelled differently is not a conflict
    scene.node.setParms({f"vector_1_1_{j}_2": '[0, 0, "-t*0.1"]'})
    scene.mod.build_params(scene.node)
    scene.node.setParms({f"boundary_type1_1_{j}_1": 1,
                         f"vector_1_1_{j}_1": "[0, 0, -100]",
                         f"boundary_type1_1_{j}_2": 2,
                         f"value_1_1_{j}_2": "100"})
    expect_error("replaces a Neumann traction",
                 lambda: scene.mod.build_params(scene.node))
    scene.node.setParms({f"boundary_type1_1_{j}_2": 1,
                         f"grouptype1_1_{j}": 1})
    expect_error("selects points",
                 lambda: scene.mod.build_params(scene.node))
    scene.node.setParms({f"grouptype1_1_{j}": 0,
                         f"basegroup1_1_{j}": "axis:+z:7"})
    expect_error("selects nothing",
                 lambda: scene.mod.build_params(scene.node))

    cavity = Scene(root, "cavity")
    bottom(cavity)
    cavity.sideset("*", (4, "1000", ALL))
    expect_error("Pressure Cavity",
                 lambda: cavity.mod.build_params(cavity.node))
    # a cavity that shares all of its faces with another sideset is fine
    whole = Scene(root, "cavity_whole")
    whole.sideset("*", (4, "1000", ALL))
    whole.sideset("*", (3, "10", ALL))
    whole.mod.build_params(whole.node)
    print("PASS: conflicting Dirichlet values, Neumann + normal traction, "
          "loads on points, empty typed selections and split cavities are "
          "refused by name")


def check_node_numbering(root):
    results = {}
    for tags in ("contiguous", "offset", "gaps", "shuffled"):
        scene = Scene(root, f"tags_{tags}")
        scene.sideset(scene.faces(lambda x, y, z: z < 0.01), FIXED)
        scene.sideset(scene.faces(lambda x, y, z: z > 0.99), PUSH)
        # a node sideset: hold the x = 0 side in x (Dirichlet on points)
        scene.sideset(scene.points(lambda x, y, z: x < 0.01),
                      (0, "[0, 0, 0]", X), points=True)
        results[tags], _ = scene.run()
    assert len(set(results.values())) == 1 and results["contiguous"] > 0.05, \
        results
    print("PASS: picked face and node sidesets select the same faces for "
          "node tags 1..N, offset, with gaps and shuffled "
          f"(Linf {results['contiguous']:.6g})")


def check_mesh_staging(root):
    work = os.path.join(root, "staging")
    os.makedirs(os.path.join(work, "a"))
    os.makedirs(os.path.join(work, "b"))
    first = write_box(os.path.join(work, "a", "part.msh"), n=2)
    second = write_box(os.path.join(work, "b", "part.msh"), n=2,
                       lo=(5, 0, 0), hi=(6, 1, 1))
    node = hou.node("/obj").createNode(TYPE, "staging")
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN})
    for geo, path in ((1, first), (2, second)):
        node.setParms({"geo_int": geo, "num_geos": geo,
                       f"file_location{geo}": path})
        node.parm(f"file_location{geo}").pressButton()
    staged = [node.evalParm(f"file_location{g}") for g in (1, 2)]
    assert os.path.basename(staged[0]) == "part.msh", staged
    assert os.path.basename(staged[1]) == "part_geo2.msh", staged
    assert open(staged[0]).read() == open(first).read()
    assert open(staged[1]).read() == open(second).read()
    data = json.load(open(node.hdaModule().write_params_only({"node": node})))
    assert [g["mesh"] for g in data["geometry"]] == \
        ["part.msh", "part_geo2.msh"], data["geometry"]
    for geo, low in ((1, 0.0), (2, 5.0)):
        mesh = node.node(f"geo_{geo}")
        mesh.cook(force=True)
        assert abs(mesh.geometry().boundingBox().minvec()[0] - low) < 1e-9

    # a new working directory stages again, still without clashes
    moved = os.path.join(root, "staging_moved")
    os.makedirs(moved)
    node.setParms({"working_dir": moved + "/"})
    data = json.load(open(node.hdaModule().write_params_only({"node": node})))
    meshes = [g["mesh"] for g in data["geometry"]]
    assert len(set(meshes)) == 2, meshes
    contents = [open(os.path.join(moved, "input", m)).read() for m in meshes]
    assert contents == [open(first).read(), open(second).read()]
    print("PASS: meshes named alike are staged under distinct names")


class FakeUI:
    """Stands in for hou.ui so the confirmation can be answered in hython."""

    def __init__(self, answer):
        self.answer = answer
        self.asked = 0

    def displayMessage(self, text, buttons=("OK",), **kwargs):
        if "Overwrite" in buttons:
            self.asked += 1
            return self.answer
        return 0

    def setStatusMessage(self, *args, **kwargs):
        pass


def with_answer(answer, action):
    real_available = hou.isUIAvailable
    fake = FakeUI(answer)
    hou.ui = fake
    hou.isUIAvailable = lambda: True
    try:
        return action(), fake.asked
    finally:
        hou.isUIAvailable = real_available
        del hou.ui


def check_overwrite_guard(root):
    scene = Scene(root, "guard")
    bottom(scene)
    scene.sideset("axis:+z:0.99", PUSH)
    scene.run()   # leaves results in output/
    mod, node = scene.mod, scene.node
    assert "sim.pvd" in mod.previous_run_files(scene.work)
    params = os.path.join(scene.work, "input", "params.json")
    before = os.stat(params).st_mtime_ns

    # Cancel: nothing is written
    result, asked = with_answer(2, lambda: mod.write_params_only(
        {"node": node}))
    assert result is None and asked == 1
    assert os.stat(params).st_mtime_ns == before

    # Use a New Folder: the node moves on to <folder>_run2
    result, asked = with_answer(1, lambda: mod.write_params_only(
        {"node": node}))
    assert asked == 1 and result is not None
    assert os.path.normpath(node.evalParm("working_dir")) == \
        os.path.normpath(scene.work) + "_run2", node.evalParm("working_dir")
    assert os.path.dirname(os.path.dirname(result)) == \
        os.path.normpath(scene.work) + "_run2"
    assert os.stat(params).st_mtime_ns == before

    # Overwrite: asked once per node and folder
    node.setParms({"working_dir": scene.work + "/"})
    result, asked = with_answer(0, lambda: mod.write_params_only(
        {"node": node}))
    assert asked == 1 and result == params
    result, asked = with_answer(0, lambda: mod.write_params_only(
        {"node": node}))
    assert asked == 0, "asked again after Overwrite"

    # Importing a run asks again before replacing it
    imported = hou.node("/obj").createNode(TYPE, "guard_import")
    imported.setUserData(mod._OVERWRITE_APPROVED,
                         os.path.normpath(scene.work))
    imported.setParms({"old_input_dir": scene.work})
    imported.hdaModule().read_params({"node": imported})
    assert imported.userData(mod._OVERWRITE_APPROVED) is None
    report = imported.evalParm("import_report")
    assert "ask before replacing its results" in report, report
    result, asked = with_answer(2, lambda: imported.hdaModule()
                                .write_params_only({"node": imported}))
    assert result is None and asked == 1
    assert mod.next_free_folder(scene.work) == \
        os.path.normpath(scene.work) + "_run3/"
    print("PASS: replacing an earlier run is confirmed (cancel / new folder "
          "/ overwrite once), and an imported run asks again")


def build_round_trip_scene(root):
    scene = Scene(root, "round_trip", tags="gaps",
                  tag_of=lambda x, y, z: 1 if x < 0.5 else 2)
    node = scene.node
    assert node.evalParm("num_volumes1") == 2
    scene.sideset("axis:-z:0.01", FIXED, vol=1)
    scene.sideset("axis:-z:0.01", FIXED, vol=2)
    scene.sideset("*", (1, "[0, 0, 0]", ALL), vol=2)
    top = scene.faces(lambda x, y, z: z > 0.99, vol=2)
    scene.sideset(top, (0, "[0, 0, 0]", X), (0, '[0, 0, "-0.1*t"]', Z),
                  vol=2)
    scene.sideset("box:[-1,-1,0.3],[2,2,0.7]", (3, '"-50*t"', ALL), vol=2)
    return scene


def node_sidesets(node):
    """The sideset interface of geometry 1 as plain data."""
    out = []
    for vol in range(1, node.evalParm("num_volumes1") + 1):
        for j in range(1, node.evalParm(f"sideset_selection1_{vol}") + 1):
            conditions = []
            for k in range(1, node.evalParm(
                    f"Boundary_Condition__1_{vol}_{j}") + 1):
                s = f"1_{vol}_{j}_{k}"
                kind = node.evalParm(f"boundary_type{s}")
                conditions.append((
                    kind, node.evalParm(f"vector_{s}") if kind in (0, 1)
                    else node.evalParm(f"value_{s}"),
                    tuple(node.evalParm(f"{a}_dimension{s}") for a in "xyz")
                    if kind == 0 else None))
            pattern = node.evalParm(f"basegroup1_{vol}_{j}")
            module = node.hdaModule()
            if pattern.strip() != "*" \
                    and not module.is_native_selection(pattern):
                # picks: the same faces, whatever the range notation
                pattern = tuple(sorted(module.expand_group_str(pattern)))
            out.append((vol, j, node.evalParm(f"grouptype1_{vol}_{j}"),
                        pattern, conditions))
    return out


def check_import(root):
    scene = build_round_trip_scene(root)
    # Only the export matters here (a pressure patch on an open surface is
    # not a solvable load for PolyFEM, and it does not need to be).
    _, data = scene.write()
    original_faces = face_conditions(scene.work, data)
    original_ui = node_sidesets(scene.node)

    # with the record: exactly what was entered
    restored = hou.node("/obj").createNode(TYPE, "restored")
    restored.setParms({"old_input_dir": scene.work})
    restored.hdaModule().read_params({"node": restored})
    assert "All represented scene" in restored.evalParm("import_report"), \
        restored.evalParm("import_report")
    assert node_sidesets(restored) == original_ui, (
        node_sidesets(restored), original_ui)
    again = os.path.join(root, "round_trip_again")
    os.makedirs(again)
    restored.setParms({"working_dir": again + "/"})
    data_again = json.load(open(restored.hdaModule().write_params_only(
        {"node": restored})))
    assert data_again["boundary_conditions"] == data["boundary_conditions"]
    assert face_conditions(again, data_again) == original_faces
    print("PASS: import restores '*', typed selections, picks and every "
          "condition as entered; the export is unchanged")

    # without the record: same solver input, overlaps as their own sidesets
    os.remove(os.path.join(scene.work, "input", "hda_sidesets.json"))
    rebuilt = hou.node("/obj").createNode(TYPE, "rebuilt")
    rebuilt.setParms({"old_input_dir": scene.work})
    rebuilt.hdaModule().read_params({"node": rebuilt})
    report = rebuilt.evalParm("import_report")
    assert "restored as sideset" in report, report
    third = os.path.join(root, "round_trip_without_record")
    os.makedirs(third)
    rebuilt.setParms({"working_dir": third + "/"})
    data_third = json.load(open(rebuilt.hdaModule().write_params_only(
        {"node": rebuilt})))
    assert face_conditions(third, data_third) == original_faces
    print("PASS: without the record the rebuilt scene gives every face the "
          "same conditions")

    # an edited params.json is followed, not the record
    edited = os.path.join(root, "round_trip_edited")
    shutil.copytree(os.path.join(again, "input"),
                    os.path.join(edited, "input"))
    path = os.path.join(edited, "input", "params.json")
    changed = json.load(open(path))
    for entry in changed["boundary_conditions"]["pressure_boundary"]:
        entry["value"] = "-75*t"
    json.dump(changed, open(path, "w"))
    follower = hou.node("/obj").createNode(TYPE, "follower")
    follower.setParms({"old_input_dir": edited})
    follower.hdaModule().read_params({"node": follower})
    report = follower.evalParm("import_report")
    assert "differ from the ones the asset wrote" in report, report
    values = [c[1] for s in node_sidesets(follower) for c in s[4]
              if c[0] == 3]
    assert values == ['"-75*t"'], values
    print("PASS: a params.json edited after export wins over the record")


def check_old_format_import(root):
    """Files written before 2026-10-01: one id per condition, faces repeated."""
    scene = Scene(root, "old_format")
    bottom(scene)
    top = scene.faces(lambda x, y, z: z > 0.99)
    j = scene.sideset(top, (0, "[0, 0, 0]", X), (0, '[0, 0, "-0.1*t"]', Z))
    path, data = scene.write()
    work_input = os.path.dirname(path)
    rows = np.loadtxt(os.path.join(work_input, "surface_sidesets1_tri.txt"),
                      dtype=np.int64, ndmin=2)
    sid = scene.mod.sideset_id(1, 1, j)
    mine = rows[rows[:, 0] == sid]
    second = mine.copy()
    second[:, 0] = scene.mod.boundary_id(1, 1, j, 2)
    np.savetxt(os.path.join(work_input, "surface_sidesets1_tri.txt"),
               np.vstack([rows, second]), fmt="%d")
    dirichlet = data["boundary_conditions"]["dirichlet_boundary"]
    dirichlet[:] = [d for d in dirichlet if d["id"] != sid] + [
        {"id": sid, "value": [0, 0, 0], "dimension": [True, False, False]},
        {"id": scene.mod.boundary_id(1, 1, j, 2),
         "value": [0, 0, "-0.1*t"], "dimension": [False, False, True]}]
    json.dump(data, open(path, "w"))
    os.remove(os.path.join(work_input, "hda_sidesets.json"))

    old = hou.node("/obj").createNode(TYPE, "old_import")
    old.setParms({"old_input_dir": scene.work})
    old.hdaModule().read_params({"node": old})
    report = old.evalParm("import_report")
    assert "applied only the first of them" in report, report
    assert old.evalParm(f"Boundary_Condition__1_1_{j}") == 2
    assert old.evalParm(f"z_dimension1_1_{j}_2") == 1
    fresh = os.path.join(root, "old_format_again")
    os.makedirs(fresh)
    old.setParms({"working_dir": fresh + "/"})
    data_again = json.load(open(old.hdaModule().write_params_only(
        {"node": old})))
    merged = [d for d in data_again["boundary_conditions"]
              ["dirichlet_boundary"] if d["id"] == sid]
    assert merged == [{"id": sid, "value": [0, 0, "-0.1*t"],
                       "dimension": [True, False, True]}], merged
    print("PASS: an old per-condition file imports with a note, and both "
          "conditions act once exported again")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    root = tempfile.mkdtemp(prefix="polyfem_sideset_conditions_")
    check_conditions_on_one_sideset(root)
    check_overlapping_sidesets(root)
    check_refusals(root)
    check_node_numbering(root)
    check_mesh_staging(root)
    check_overwrite_guard(root)
    check_import(root)
    check_old_format_import(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
