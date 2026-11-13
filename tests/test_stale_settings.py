"""Settings that outlived what they belonged to (review of 2026-11-13).

Removing a sideset or a condition with its own buttons always worked. These
are the other ways out: every check below left sidesets, conditions or
subdomains behind, or brought one back, before the fix it names.

  * emptying the Mesh File field (or a load that failed after the old chain
    was removed) kept subdomains 2+ with their sidesets and conditions, whose
    picked face numbers were then applied to the next mesh;
  * a sideset whose conditions were removed got the default Dirichlet back
    from any rebuild of the display chain;
  * Restore Mesh Subdomains kept the subdomains Apply Subdomain Change made;
  * turning Obstacle on and off again swapped or reset subdomain settings
    when the physical tags were not 1..N;
  * Clear All kept geometry 1's transform keyframes, obstacle displacement,
    initial conditions and Enabled state;
  * a mesh regenerated over its staged copy was reloaded as "the same file";
  * selection tokens other than N and N-M were left out of the export.

Run: hython tests/test_stale_settings.py
Needs no PolyFEM binary and no gmsh (nothing is solved).
"""

import os
import tempfile

import numpy as np

import hou

from test_subdomains_reload import (
    BASE, Messages, add_sideset, halves, load, new_node, picked_faces,
    settings_of, write_box)


def press(node, name):
    node.parm(name).pressButton()


def two_subdomains(node, toks):
    """Distinct materials, sidesets and an initial condition on both."""
    node.setParms({"materials1_1": toks.index("NeoHookean"), "E1_1": 2e5,
                   "nu1_1": 0.3,
                   "materials1_2": toks.index("MooneyRivlin"),
                   "c11_2": 7000.0, "c21_2": 1000.0, "bulk1_2": 1e6})
    add_sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    add_sideset(node, 2, picked_faces(node, 2, lambda x, y, z: z > 0.99),
                1, "[0, 0, -10]")
    node.setParms({"initial_conditions1_2": 1, "conditiontype1_2_1": 1,
                   "condition_vector_1_2_1": "[0, 0, 1]"})


def assert_clean_subdomains(node, count):
    assert node.evalParm("num_volumes1") == count, node.evalParm("num_volumes1")
    for vol in range(1, count + 1):
        assert node.evalParm(f"sideset_selection1_{vol}") == 0, vol
        assert node.evalParm(f"initial_conditions1_{vol}") == 0, vol
    for vol in range(2, count + 1):
        assert node.parm(f"materials1_{vol}").isAtDefault(), \
            f"subdomain {vol} kept the material of the mesh that was removed"
    assert not [n.name() for n in node.children()
                if n.name().startswith(("group_1_", "solo_1_"))], \
        "sideset group nodes of the removed mesh are still there"


def check_cleared_mesh(root):
    work = os.path.join(root, "cleared")
    os.makedirs(work)
    first = write_box(os.path.join(work, "first.msh"), halves(1, 2))
    second = write_box(os.path.join(work, "second.msh"), halves(1, 2), n=3)
    node = new_node(work, "cleared")
    toks = node.hdaModule().MATERIAL_TOKENS
    load(node, first)
    two_subdomains(node, toks)

    # empty the Mesh File field, then load another mesh
    node.setParms({"file_location1": ""})
    press(node, "file_location1")
    assert node.node("geo_1") is None, "the emptied geometry kept its nodes"
    load(node, second)
    assert_clean_subdomains(node, 2)
    print("PASS: after the Mesh File field is emptied, the next mesh starts "
          "with no sidesets, conditions or initial conditions on any "
          "subdomain")

    # the same after a file this node cannot load
    two_subdomains(node, toks)
    node.setParms({"file_location1": os.path.join(work, "notes.txt")})
    with Messages():
        press(node, "file_location1")
    assert node.node("geo_1") is None
    load(node, first)
    assert_clean_subdomains(node, 2)
    print("PASS: the same after picking a file that is not a mesh")


def check_default_condition(root):
    work = os.path.join(root, "default")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "one.msh"), lambda x, y, z: 1)
    node = new_node(work, "default")
    module = node.hdaModule()
    load(node, mesh)
    node.parm("sideset_selection1_1").set(1)
    press(node, "sideset_selection1_1")
    assert node.evalParm("Boundary_Condition__1_1_1") == 1, \
        "a new sideset no longer starts with its default condition"
    node.setParms({"Boundary_Condition__1_1_1": 0})   # the user removes it

    def cleared(after):
        assert node.evalParm("sideset_selection1_1") == 1, after
        assert node.evalParm("Boundary_Condition__1_1_1") == 0, \
            f"the removed condition came back after {after}"

    node.setParms({"elements_1": "0-5", "subdomain_number_1": 2})
    module.update_entities({"node": node, "script_multiparm_index": "1"})
    cleared("Apply Subdomain Change")
    assert node.evalParm("num_volumes1") == 2
    add_sideset(node, 2, "*", 0, "[0, 0, 0]")

    module.revert_entities({"node": node, "script_multiparm_index": "1"})
    cleared("Restore Mesh Subdomains")
    assert node.evalParm("num_volumes1") == 1, \
        "Restore Mesh Subdomains kept the subdomain it had added"
    assert node.parm("sideset_selection1_2") is None
    assert not [n.name() for n in node.children()
                if n.name().startswith(("group_1_2_", "solo_1_2"))]
    print("PASS: Restore Mesh Subdomains removes the subdomains Apply "
          "Subdomain Change added, with their sidesets")

    press(node, "file_location1")   # the same mesh again
    cleared("reloading the mesh")
    dst = module.geo_duplicate({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm(f"Boundary_Condition__{dst}_1_1") == 0, \
        "the duplicate got a condition the original does not have"
    cleared("Duplicate Geometry")

    # adding a sideset still gives it the default condition
    node.parm("sideset_selection1_1").set(2)
    press(node, "sideset_selection1_1")
    assert node.evalParm("Boundary_Condition__1_1_2") == 1
    assert node.evalParm("Boundary_Condition__1_1_1") == 0
    print("PASS: a sideset whose conditions were removed stays without one "
          "through subdomain edits, a reload and Duplicate; a new sideset "
          "still starts with the default Dirichlet")


def check_obstacle_toggle(root):
    for name, tag_of, count in (
            ("tags23", halves(2, 3), 2),
            ("untagged", halves(0, 1), 2)):
        work = os.path.join(root, name)
        os.makedirs(work)
        mesh = write_box(os.path.join(work, name + ".msh"), tag_of)
        node = new_node(work, name)
        toks = node.hdaModule().MATERIAL_TOKENS
        load(node, mesh)
        assert node.evalParm("num_volumes1") == count
        two_subdomains(node, toks)
        before = {vol: settings_of(node, vol) for vol in range(1, count + 1)}

        node.setParms({"is_obstacle1": 1})
        press(node, "is_obstacle1")
        assert node.node("subdomains_1") is None
        node.setParms({"is_obstacle1": 0})
        with Messages() as messages:
            press(node, "is_obstacle1")
        assert node.evalParm("num_volumes1") == count
        for vol in range(1, count + 1):
            assert settings_of(node, vol) == before[vol], \
                (name, vol, messages.text())
        assert "renumbered" not in messages.text(), messages.text()
        assert "starts with default settings" not in messages.text(), \
            messages.text()
    print("PASS: turning Obstacle on and off again leaves every subdomain's "
          "settings where they were (physical tags 2 and 3; untagged "
          "elements with tag 1)")


def check_clear_all(root):
    work = os.path.join(root, "clearall")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "block.msh"), lambda x, y, z: 1)
    node = new_node(work, "clearall")
    module = node.hdaModule()
    load(node, mesh)
    node.setParms({"is_obstacle1": 1})
    press(node, "is_obstacle1")
    keyframes = []
    for frame, value in ((1, 0.0), (24, 1.0)):
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression("linear()", hou.exprLanguage.Hscript)
        keyframes.append(key)
    node.parm("xform_t__1z").setKeyframes(keyframes)
    node.setParms({"obstacle_disp1": '[0, 0, "-0.1*t"]', "is_enabled1": 0,
                   "initial_conditions1_1": 1})
    assert module.transform_is_animated(node, 1)

    module.clear_all_geos({"node": node})
    assert not node.parm("xform_t__1z").keyframes(), \
        "Clear All kept geometry 1's transform keyframes"
    assert node.evalParm("xform_t__1z") == 0
    assert node.parm("obstacle_disp1").isAtDefault()
    assert node.evalParm("is_enabled1") == 1
    assert node.evalParm("initial_conditions1_1") == 0
    load(node, mesh)
    assert not module.transform_is_animated(node, 1), \
        "the mesh loaded after Clear All inherited an animated transform"
    print("PASS: Clear All removes geometry 1's transform keyframes, "
          "obstacle displacement and initial conditions, and enables it")


def check_overwritten_in_place(root):
    work = os.path.join(root, "inplace")
    os.makedirs(work)
    source = write_box(os.path.join(work, "part.msh"), halves(1, 2))
    node = new_node(work, "inplace")
    module = node.hdaModule()
    load(node, source)
    staged = node.evalParm("file_location1")
    assert os.path.dirname(os.path.normpath(staged)) == \
        os.path.normpath(os.path.join(work, "input")), staged
    picks = picked_faces(node, 2, lambda x, y, z: z > 0.99)
    add_sideset(node, 2, picks, 0, "[0, 0, 0]")

    # regenerate the mesh over its staged copy: same nodes, elements reversed
    head, rest = open(staged).read().split("$Elements\n")
    rest = rest.split("\n")
    count = int(rest[0])
    with open(staged, "w") as handle:
        handle.write(head + "$Elements\n" + f"{count}\n"
                     + "\n".join(rest[1:1 + count][::-1]) + "\n$EndElements\n")
    press(node, "file_location1")
    moved = node.evalParm("basegroup1_2_1")
    surface = node.node("null_1").geometry()
    centres = [np.mean([v.point().position() for v in
                        surface.prim(int(p)).vertices()], axis=0)
               for p in module.expand_group_str(moved)]
    assert len(centres) == len(module.expand_group_str(picks)), \
        (moved, picks)
    assert all(c[2] > 0.99 for c in centres), \
        "a mesh overwritten in place was reloaded as the same file: the " \
        "picks now name other faces"
    print("PASS: a mesh regenerated over its staged copy is reloaded as a "
          "changed file, and picked faces follow it")


def check_selection_patterns(root):
    work = os.path.join(root, "patterns")
    os.makedirs(work)
    mesh = write_box(os.path.join(work, "one.msh"), lambda x, y, z: 1)
    node = new_node(work, "patterns")
    module = node.hdaModule()
    expand = module.expand_group_str
    assert expand("0 2 5-8") == [0, 2, 5, 6, 7, 8]
    assert expand("0,2,5-8") == [0, 2, 5, 6, 7, 8]
    assert expand("0-10:2") == [0, 2, 4, 6, 8, 10]
    assert expand("0-6:2,3") == [0, 1, 3, 4, 6]
    assert module.unsupported_group_tokens("0 2 5-8 0-10:2 0-6:2,3") == []
    assert module.unsupported_group_tokens("3 top !5 4") == ["top", "!5"]

    # Houdini's own reading of the stepped forms, on the surface itself
    load(node, mesh)
    surface = node.node("null_1").geometry()
    for pattern in ("0-10:2", "0-6:2,3", "1 4-9:5"):
        own = sorted(prim.number() for prim in surface.globPrims(pattern))
        assert sorted(expand(pattern)) == own, (pattern, expand(pattern), own)

    j = add_sideset(node, 1, "0-10:2", 0, "[0, 0, 0]")
    plan = module.resolve_sidesets(node, 1, 1)
    record = [r for r in plan["records"] if r["sideset"] == j][0]
    assert record["count"] == 6, record
    node.setParms({f"basegroup1_1_{j}": "0-3 top"})
    report = module.check_setup(node, "write")
    assert any("'top'" in error and "not a list of face numbers" in error
               for error in report["errors"]), report["errors"]
    print("PASS: stepped selections (0-10:2, 0-6:2,3) are exported as "
          "Houdini reads them, and a token that is not a number list is "
          "refused by name instead of being left out")


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    root = tempfile.mkdtemp(prefix="polyfem_stale_")
    check_cleared_mesh(root)
    check_default_condition(root)
    check_obstacle_toggle(root)
    check_clear_all(root)
    check_overwritten_in_place(root)
    check_selection_patterns(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
