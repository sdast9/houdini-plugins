"""Subdomains from Gmsh physical groups; reloading and duplicating geometry.

Until 2026-10-01 subdomain n was physical tag n: tags {1, 100} made 100
subdomains (98 empty materials exported), tags of 1000 or more collided with
the next geometry's ids, and one untagged element renumbered every subdomain
(the MSH Reader's +1 shift). Picking a geometry's mesh again reset subdomain
2 onwards and every sideset, and Duplicate Geometry copied only the material
type and a few values. Now subdomains follow the sorted physical groups
(untagged elements last, tag and name shown in each subdomain's header), a
reload keeps every setting, matched by physical group, and says what it
could not keep, and Duplicate copies everything.

Run: hython tests/test_subdomains_reload.py
Uses ../polyfem/build/PolyFEM_bin; no gmsh needed.
"""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_box(path, tag_of, n=2, names=None):
    """Structured tet box; tag_of(cx, cy, cz) gives each cell's physical
    tag (0 = none). names: {tag: name} for $PhysicalNames."""
    xs = np.linspace(0.0, 1.0, n + 1)

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i

    points = [(x, y, z) for z in xs for y in xs for x in xs]
    kuhn = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
            (0, 4, 5, 6), (0, 5, 1, 6)]
    tets, tags = [], []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                     vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                     vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)]
                centre = np.mean([points[v] for v in c], axis=0)
                for t in kuhn:
                    a, b, cc, d = (c[v] for v in t)
                    p = np.array([points[a], points[b], points[cc], points[d]])
                    if np.linalg.det(p[1:] - p[0]) < 0:
                        b, cc = cc, b
                    tets.append((a, b, cc, d))
                    tags.append(tag_of(*centre))
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if names:
            handle.write(f"$PhysicalNames\n{len(names)}\n")
            for tag, name in names.items():
                handle.write(f'3 {tag} "{name}"\n')
            handle.write("$EndPhysicalNames\n")
        handle.write(f"$Nodes\n{len(points)}\n")
        for number, (x, y, z) in enumerate(points, 1):
            handle.write(f"{number} {float(x)!r} {float(y)!r} {float(z)!r}\n")
        handle.write(f"$EndNodes\n$Elements\n{len(tets)}\n")
        for number, (tet, tag) in enumerate(zip(tets, tags), 1):
            handle.write(f"{number} 4 2 {tag} {tag} "
                         + " ".join(str(v + 1) for v in tet) + "\n")
        handle.write("$EndElements\n")
    return path


def halves(left, right):
    return lambda x, y, z: left if x < 0.5 else right


class Messages:
    """Collects what the node would show in dialogs (hython has no UI)."""

    def __init__(self):
        self.seen = []

    def __enter__(self):
        import builtins
        self.real = builtins.print
        outer = self

        def capture(*args, **kwargs):
            outer.seen.append(" ".join(str(a) for a in args))
            outer.real(*args, **kwargs)
        builtins.print = capture
        return self

    def __exit__(self, *exc):
        import builtins
        builtins.print = self.real

    def text(self):
        return "\n".join(self.seen)


def new_node(work, name):
    os.makedirs(work, exist_ok=True)
    node = hou.node("/obj").createNode(TYPE, name)
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN,
                   "log_level": 2, "quasistatic": 1, "end_time_bool": 1,
                   "tend": 1.0, "time_inc_bool": 1, "dt": 1.0, "enable": 0})
    return node


def load(node, path, geo=1):
    if geo > node.evalParm("num_geos"):
        node.setParms({"geo_int": geo, "num_geos": geo})
    node.setParms({f"file_location{geo}": path})
    node.parm(f"file_location{geo}").pressButton()


def add_sideset(node, vol, pattern, kind, text, dims=(1, 1, 1), geo=1):
    j = node.evalParm(f"sideset_selection{geo}_{vol}") + 1
    node.parm(f"sideset_selection{geo}_{vol}").set(j)
    node.parm(f"sideset_selection{geo}_{vol}").pressButton()
    node.setParms({f"basegroup{geo}_{vol}_{j}": pattern,
                   f"Boundary_Condition__{geo}_{vol}_{j}": 1,
                   f"boundary_type{geo}_{vol}_{j}_1": kind})
    node.setParms({(f"vector_{geo}_{vol}_{j}_1" if kind in (0, 1)
                    else f"value_{geo}_{vol}_{j}_1"): text})
    for axis, flag in zip("xyz", dims):
        node.setParms({f"{axis}_dimension{geo}_{vol}_{j}_1": flag})
    return j


def picked_faces(node, vol, predicate, geo=1):
    surface = node.node(f"null_{geo}").geometry()
    out = []
    for prim in surface.prims():
        if prim.attribValue("Entity") != vol:
            continue
        centre = sum((v.point().position() for v in prim.vertices()),
                     hou.Vector3()) / len(prim.vertices())
        if predicate(*centre):
            out.append(prim.number())
    return " ".join(str(p) for p in out)


def solve(params):
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info", "--max_threads", "1"],
        cwd=os.path.dirname(params), capture_output=True, text=True,
        timeout=900)
    assert result.returncode == 0, result.stdout[-1500:]


def configure_two(node, toks):
    """Distinct settings on both subdomains of a two-subdomain geometry."""
    node.setParms({"materials1_1": toks.index("NeoHookean"), "E1_1": 2e5,
                   "nu1_1": 0.3,
                   "materials1_2": toks.index("MooneyRivlin"),
                   "c11_2": 7000.0, "c21_2": 1000.0, "bulk1_2": 1e6,
                   "mainOrder1_2": 1})
    add_sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    add_sideset(node, 2, picked_faces(node, 2, lambda x, y, z: z > 0.99),
                0, '[0, 0, "-0.05*t"]')
    node.setParms({"initial_conditions1_2": 1, "conditiontype1_2_1": 1,
                   "condition_vector_1_2_1": "[0, 0, 1]"})
    node.parmTuple("color_1_2").set((0.1, 0.2, 0.3))


def settings_of(node, vol, geo=1):
    """A subdomain's parameters (its read-only Elements line excluded)."""
    module = node.hdaModule()
    return [(name, tail, values) for name, depth, is_count, tail, values in
            module.snapshot_instance(node, (geo, vol))
            if name != "subdomain_info#_#"]


def check_physical_groups(root):
    work = os.path.join(root, "tags")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "tags.msh"), halves(1, 100),
                     names={1: "matrix", 100: "insert"})
    node = new_node(work, "tags")
    load(node, mesh)
    assert node.evalParm("num_volumes1") == 2, node.evalParm("num_volumes1")
    info = [node.evalParm(f"subdomain_info1_{v}") for v in (1, 2)]
    assert info[0] == ('Gmsh physical group 1 "matrix", 24 elements.'), info
    assert info[1] == ('Gmsh physical group 100 "insert", 24 elements.'), info
    add_sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    add_sideset(node, 2, "axis:-z:0.01", 0, "[0, 0, 0]")
    add_sideset(node, 2, "axis:+z:0.99", 0, '[0, 0, "-0.05*t"]')
    params = node.hdaModule().write_params_only({"node": node})
    data = json.load(open(params))
    assert [m["id"] for m in data["materials"]] == [1001, 1002], \
        data["materials"]
    volumes = np.loadtxt(os.path.join(work, "input", "volumes1.txt"))
    assert set(volumes.astype(int)) == {1001, 1002}
    solve(params)
    print("PASS: physical tags {1, 100} make two subdomains with their "
          "names in the headers, and the run solves")

    # untagged elements come last instead of renumbering the tagged ones
    work = os.path.join(root, "untagged")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "untagged.msh"),
                     lambda x, y, z: 2 if x > 0.5 else (1 if y > 0.5 else 0))
    node = new_node(work, "untagged")
    load(node, mesh)
    texts = [node.evalParm(f"subdomain_info1_{v}") for v in (1, 2, 3)]
    assert texts[0].startswith("Gmsh physical group 1,"), texts
    assert texts[1].startswith("Gmsh physical group 2,"), texts
    assert texts[2].startswith("Elements without a Gmsh physical group"), \
        texts
    print("PASS: untagged elements form the last subdomain; tags 1 and 2 "
          "stay subdomains 1 and 2")


def check_reload(root):
    work = os.path.join(root, "reload")
    os.makedirs(work)
    mesh_a = write_box(os.path.join(work, "a.msh"), halves(1, 2))
    node = new_node(work, "reload")
    toks = node.hdaModule().MATERIAL_TOKENS
    load(node, mesh_a)
    configure_two(node, toks)
    # split subdomain 1 with Apply Subdomain Change: a third subdomain
    module = node.hdaModule()
    node.setParms({"elements_1": "0-5", "subdomain_number_1": 3})
    module.update_entities({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm("num_volumes1") == 3
    node.setParms({"materials1_3": toks.index("LinearElasticity"),
                   "E1_3": 3e5})
    before = {vol: settings_of(node, vol) for vol in (1, 2, 3)}
    export_before = json.load(open(module.write_params_only({"node": node})))

    # the same file again: nothing changes, the reassignment survives
    with Messages() as messages:
        node.parm("file_location1").pressButton()
    assert "every subdomain kept its settings" in messages.text(), \
        messages.text()
    assert node.evalParm("num_volumes1") == 3
    for vol in (1, 2, 3):
        assert settings_of(node, vol) == before[vol], vol
    assert node.node("vex_1_3") is not None
    export_after = json.load(open(module.write_params_only({"node": node})))
    for key in ("materials", "boundary_conditions", "initial_conditions"):
        assert export_after[key] == export_before[key], key
    print("PASS: loading the same mesh again keeps every subdomain's "
          "materials, sidesets, initial conditions and element reassignments")

    # a changed file: group 2 became group 7, the nodes are the same
    mesh_b = write_box(os.path.join(work, "b.msh"), halves(1, 7))
    node.setParms({"file_location1": mesh_b})
    with Messages() as messages:
        node.parm("file_location1").pressButton()
    text = messages.text()
    assert "subdomain 2 (Gmsh physical group 7) is new" in text, text
    assert "subdomain 2 (Gmsh physical group 2) of the previous mesh is not " \
        "in this one" in text, text
    assert "Apply Subdomain Change were removed" in text, text
    assert node.evalParm("num_volumes1") == 2
    assert settings_of(node, 1) == before[1]
    assert node.parm("materials1_2").evalAsString() == "NeoHookean"
    print("PASS: a changed mesh keeps the settings of the physical groups it "
          "still has and lists what it could not keep")

    # picks follow their faces into a re-meshed file with the same nodes
    node.setParms({"file_location1": mesh_a})
    node.parm("file_location1").pressButton()
    configure_two(node, toks)
    picks = node.evalParm("basegroup1_2_1")
    shuffled = os.path.join(work, "a_reordered.msh")
    lines = open(mesh_a).read().split("$Elements\n")
    head, rest = lines[0], lines[1].split("\n")
    count, elements = int(rest[0]), rest[1:1 + int(rest[0])]
    elements = elements[::-1]   # same elements, reversed order
    with open(shuffled, "w") as handle:
        handle.write(head + "$Elements\n" + f"{count}\n"
                     + "\n".join(elements) + "\n$EndElements\n")
    node.setParms({"file_location1": shuffled})
    node.parm("file_location1").pressButton()
    moved = node.evalParm("basegroup1_2_1")
    surface = node.node("null_1").geometry()
    centres = [np.mean([v.point().position() for v in
                        surface.prim(int(p)).vertices()], axis=0)
               for p in node.hdaModule().expand_group_str(moved)]
    assert len(centres) == len(node.hdaModule().expand_group_str(picks))
    assert all(c[2] > 0.99 for c in centres), "picks moved to other faces"
    print("PASS: picked faces follow the mesh when the file is re-ordered")


def check_legacy_numbering(root):
    """A scene from before 2026-10-01: no subdomains_<geo> node, subdomain
    number = physical tag (100 subdomains for tags 1 and 100)."""
    work = os.path.join(root, "legacy")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "legacy.msh"), halves(1, 100))
    node = new_node(work, "legacy")
    toks = node.hdaModule().MATERIAL_TOKENS
    load(node, mesh)
    # rebuild the old chain: the reader feeds attrib_1 directly
    node.node("attrib_1").setInput(0, node.node("geo_1"))
    node.node("subdomains_1").destroy()
    node.setParms({"num_volumes1": 100})
    node.setParms({"materials1_100": toks.index("MooneyRivlin"),
                   "c11_100": 7000.0, "c21_100": 1000.0})
    add_sideset(node, 100, "axis:-z:0.01", 0, "[0, 0, 0]")
    with Messages() as messages:
        node.parm("file_location1").pressButton()
    text = messages.text()
    assert "100 -> 2" in text and "98 empty subdomains" in text, text
    assert node.evalParm("num_volumes1") == 2
    assert node.parm("materials1_2").evalAsString() == "MooneyRivlin"
    assert node.evalParm("c11_2") == 7000.0
    assert node.evalParm("sideset_selection1_2") == 1
    assert node.evalParm("basegroup1_2_1") == "axis:-z:0.01"
    print("PASS: a scene numbered by physical tag is renumbered on reload, "
          "and subdomain 100's settings move to subdomain 2")


def check_old_export_import(root):
    work = os.path.join(root, "export")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "two.msh"), halves(1, 100))
    node = new_node(work, "export")
    toks = node.hdaModule().MATERIAL_TOKENS
    load(node, mesh)
    configure_two(node, toks)
    params = node.hdaModule().write_params_only({"node": node})
    # Make it an export from before 2026-10-01: subdomain 2 was "100".
    data = json.load(open(params))

    def old(value):
        value = int(value)
        if value == 1002:
            return 1100
        if 10020000 <= value < 10030000:
            return value - 10020000 + 11000000
        return value
    for material in data["materials"]:
        material["id"] = old(material["id"])
    for entry in data["space"]["discr_order"]:
        entry["id"] = old(entry["id"])
    for key, entries in data["boundary_conditions"].items():
        if key != "rhs" and isinstance(entries, list):
            for entry in entries:
                entry["id"] = old(entry["id"])
    for entries in data.get("initial_conditions", {}).values():
        for entry in entries:
            entry["id"] = old(entry["id"])
    json.dump(data, open(params, "w"), indent=2)
    input_dir = os.path.dirname(params)
    volumes = np.loadtxt(os.path.join(input_dir, "volumes1.txt"), dtype=int)
    np.savetxt(os.path.join(input_dir, "volumes1.txt"),
               np.vectorize(old)(volumes), fmt="%d")
    for name in os.listdir(input_dir):
        if name.startswith(("surface_sidesets", "point_sidesets")):
            rows = np.loadtxt(os.path.join(input_dir, name), dtype=np.int64,
                              ndmin=2)
            rows[:, 0] = np.vectorize(old)(rows[:, 0])
            np.savetxt(os.path.join(input_dir, name), rows, fmt="%d")
    os.remove(os.path.join(input_dir, "hda_sidesets.json"))

    restored = hou.node("/obj").createNode(TYPE, "restored")
    restored.setParms({"old_input_dir": work})
    restored.hdaModule().read_params({"node": restored})
    report = restored.evalParm("import_report")
    assert "were renumbered 1-2" in report, report
    assert restored.evalParm("num_volumes1") == 2
    assert restored.parm("materials1_2").evalAsString() == "MooneyRivlin"
    assert restored.evalParm("c11_2") == 7000.0
    assert restored.evalParm("mainOrder1_2") == 1
    assert restored.evalParm("sideset_selection1_2") == 1
    assert restored.evalParm("initial_conditions1_2") == 1
    print("PASS: an export numbered by physical tag (ids 1100, 1100xxxx) "
          "imports as subdomains 1 and 2 with every setting in place")


def check_duplicate(root):
    work = os.path.join(root, "duplicate")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "dup.msh"), halves(1, 2))
    node = new_node(work, "duplicate")
    module = node.hdaModule()
    toks = module.MATERIAL_TOKENS
    load(node, mesh)
    configure_two(node, toks)
    node.setParms({"materials1_1": toks.index("MaterialSum"),
                   "num_fiber_families1_1": 1})
    node.setParms({"fam_model1_1_1": "HGODispersion", "fam_k11_1_1": 55.0,
                   "fam_kappa1_1_1": 0.1, "fam_mirror1_1_1": 1,
                   "fam_theta1_1_1": 30.0})
    node.parmTuple("fam_fib_dir1_1_1").set((0, 1, 0))
    node.parmTuple("xform_t__1").set((2.0, 0.0, 0.0))
    node.setParms({"elements_1": "0-3", "subdomain_number_1": 3})
    module.update_entities({"node": node, "script_multiparm_index": "1"})
    dst = module.geo_duplicate({"node": node, "script_multiparm_index": "1"})
    assert dst == 2 and node.evalParm("num_geos") == 2
    original = settings_of(node, 1, geo=1)
    for vol in (1, 2, 3):
        assert settings_of(node, vol, geo=2) == settings_of(node, vol, geo=1), vol
    assert node.parmTuple("xform_t__2").eval() == (2.0, 0.0, 0.0)
    geo1 = np.frombuffer(node.node("branch_1").geometry()
                         .primIntAttribValuesAsString("Entity"), np.int32)
    geo2 = np.frombuffer(node.node("branch_2").geometry()
                         .primIntAttribValuesAsString("Entity"), np.int32)
    assert np.array_equal(geo1, geo2), "subdomain assignment not copied"
    data = json.load(open(module.write_params_only({"node": node})))
    by_id = {m["id"]: json.dumps({k: v for k, v in m.items() if k != "id"},
                                 sort_keys=True)
             for m in data["materials"]}
    for vol in (1, 2, 3):
        # per-element fiber fields are named after their geometry (FIB_<geo>_)
        assert by_id[1000 + vol].replace('"FIB_1_', '"FIB_2_') == \
            by_id[2000 + vol], (by_id[1000 + vol], by_id[2000 + vol])
    assert original
    print("PASS: Duplicate Geometry copies every setting: composite fiber "
          "families, sidesets, initial conditions, transform and element "
          "reassignments")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    root = tempfile.mkdtemp(prefix="polyfem_subdomains_")
    check_physical_groups(root)
    check_reload(root)
    check_legacy_numbering(root)
    check_old_export_import(root)
    check_duplicate(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
