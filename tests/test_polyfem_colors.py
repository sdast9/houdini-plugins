"""Headless regression test for PolyFEM 2.0 viewport color bindings.

Run: hython tests/test_polyfem_colors.py
"""

import os
import tempfile

import hou
import numpy as np

from test_polyfem_hda import BASE


def make_two_volume_msh(path):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("two_boxes")
    first = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    second = gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [first], 1)
    gmsh.model.addPhysicalGroup(3, [second], 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(path)
    gmsh.finalize()


def assert_close(actual, expected, message):
    assert np.allclose(actual, expected), f"{message}: {actual} != {expected}"


def displayed_entity_colors(node, geo):
    geometry = node.node(f"entitycolor_{geo}").geometry()
    entities = np.asarray(geometry.primIntAttribValues("Entity"))
    colors = np.asarray(geometry.primFloatAttribValues("Cd")).reshape(-1, 3)
    return {int(entity): colors[entities == entity][0]
            for entity in np.unique(entities)}


def displayed_obstacle_color(node, geo):
    # Obstacles color per-PRIM (matching the volume chains) so that merging
    # never mixes attribute classes.
    values = node.node(f"color_{geo}").geometry().primFloatAttribValues("Cd")
    return np.asarray(values).reshape(-1, 3)[0]


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_color_test_")
    first_mesh = os.path.join(work, "two_boxes_1.msh")
    second_mesh = os.path.join(work, "two_boxes_2.msh")
    obstacle = os.path.join(work, "obstacle.obj")
    make_two_volume_msh(first_mesh)
    make_two_volume_msh(second_mesh)
    with open(obstacle, "w") as obj:
        obj.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "color_test")
    mod = node.hdaModule()
    node.setParms({"working_dir": work + "/"})

    node.setParms({"num_geos": 3, "geo_int": 3})
    for geo, mesh in ((1, first_mesh), (2, second_mesh)):
        node.setParms({f"file_location{geo}": mesh})
        node.parm(f"file_location{geo}").pressButton()
        assert node.evalParm(f"num_volumes{geo}") == 2

    assert_close(node.parmTuple("color_1_1").eval(), (1, 0, 0),
                 "geometry 1 default")
    assert_close(node.parmTuple("color_1_2").eval(), (0.5, 0, 0),
                 "geometry 1 subdomain 2 default")
    assert_close(node.parmTuple("color_2_1").eval(), (0, 1, 0),
                 "geometry 2 default")
    assert_close(node.parmTuple("color_2_2").eval(), (0, 0.5, 0),
                 "geometry 2 subdomain 2 default")

    custom = {
        "color_1_1": (0.1, 0.2, 0.3),
        "color_1_2": (0.4, 0.5, 0.6),
        "color_2_1": (0.7, 0.2, 0.1),
        "color_2_2": (0.3, 0.8, 0.4),
    }
    for name, value in custom.items():
        node.parmTuple(name).set(value)

    colors_1 = displayed_entity_colors(node, 1)
    colors_2 = displayed_entity_colors(node, 2)
    assert_close(colors_1[1], custom["color_1_1"], "geometry 1 subdomain 1")
    assert_close(colors_1[2], custom["color_1_2"], "geometry 1 subdomain 2")
    assert_close(colors_2[1], custom["color_2_1"], "geometry 2 subdomain 1")
    assert_close(colors_2[2], custom["color_2_2"], "geometry 2 subdomain 2")

    # Re-importing a mesh rebuilds its full network and must preserve colors.
    node.parm("file_location2").pressButton()
    assert_close(displayed_entity_colors(node, 2)[2], custom["color_2_2"],
                 "color lost during geometry re-import")

    # Changing geometry 2 must not affect geometry 1.
    node.parmTuple("color_2_1").set((0.9, 0.8, 0.7))
    assert_close(displayed_entity_colors(node, 1)[1], custom["color_1_1"],
                 "geometry 2 color leaked into geometry 1")

    # Entity edits rebuild the display chain; custom colors must survive.
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    assert_close(displayed_entity_colors(node, 1)[2], custom["color_1_2"],
                 "color lost during entity display rebuild")

    node.setParms({"is_obstacle3": 1, "file_location3": obstacle})
    node.parm("file_location3").pressButton()
    node.parmTuple("color_3").set((0.25, 0.5, 0.75))
    assert_close(displayed_obstacle_color(node, 3), (0.25, 0.5, 0.75),
                 "obstacle color")

    # Regression: merging obstacle and volume geometry must not mix Cd
    # attribute classes (point Cd from one side blacks out the other's
    # prim colors in the viewport).
    merged = node.node("all").geometry()
    assert merged.findPointAttrib("Cd") is None, \
        "merged output carries point Cd (class conflict with prim colors)"
    m_ent = np.asarray(merged.primIntAttribValues("Entity"))
    m_geo = np.asarray(merged.primIntAttribValues("geometry_num"))
    m_cd = np.asarray(merged.primFloatAttribValues("Cd")).reshape(-1, 3)
    sel = (m_geo == 1) & (m_ent == 1)
    assert_close(m_cd[sel][0], custom["color_1_1"],
                 "volume prim color lost in merged output with obstacle")

    # Regression: tagging elements with a NEW subdomain number must grow
    # the per-volume UI and color mapping (previously rendered grey).
    node.setParms({"subdomain_number_1": 3, "elements_1": "0-5"})
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm("num_volumes1") == 3, \
        "num_volumes did not grow with new subdomain"
    assert node.parmTuple("color_1_3") is not None
    colors_after = displayed_entity_colors(node, 1)
    assert 3 in colors_after, "new subdomain not present in display"
    assert not np.allclose(colors_after[3], (0.5, 0.5, 0.5)), \
        "new subdomain rendered grey (color mapping not grown)"
    # restore subdomain tags/count for the remaining checks
    mod.revert_entities({"node": node, "script_multiparm_index": "1"})
    node.setParms({"num_volumes1": 2, "subdomain_number_1": 1})

    # Regression: creating a sideset auto-creates one zero-displacement
    # Dirichlet BC (1.x behavior).
    node.setParms({"sideset_selection1_1": 1})
    mod.create_group({"node": node, "script_multiparm_index2": "1",
                      "script_multiparm_index": "1"})
    assert node.evalParm("Boundary_Condition__1_1_1") == 1, \
        "new sideset did not auto-create a boundary condition"
    assert node.evalParm("boundary_type1_1_1_1") == 0, \
        "auto-created BC is not Dirichlet"

    # Duplicate geometry 1 after the obstacle; destination is geometry 4.
    mod.geo_duplicate({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm("num_volumes4") == 2
    assert_close(node.parmTuple("color_4_1").eval(), custom["color_1_1"],
                 "duplicated subdomain 1 color")
    assert_close(node.parmTuple("color_4_2").eval(), custom["color_1_2"],
                 "duplicated subdomain 2 color")
    assert_close(displayed_entity_colors(node, 4)[2], custom["color_1_2"],
                 "duplicated display color")

    mod.geo_duplicate({"node": node, "script_multiparm_index": "3"})
    assert_close(node.parmTuple("color_5").eval(), (0.25, 0.5, 0.75),
                 "duplicated obstacle color")
    assert_close(displayed_obstacle_color(node, 5), (0.25, 0.5, 0.75),
                 "duplicated obstacle display color")

    print("PASS: PolyFEM viewport colors")


if __name__ == "__main__":
    main()
