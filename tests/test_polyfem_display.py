"""Focused regressions for PolyFEM 2.0 material preview and transforms.

Run: hython tests/test_polyfem_display.py
"""

import os
import tempfile

import gmsh
import hou
import numpy as np


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_cube_msh(path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("display_cube")
    volume = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [volume], 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(path)
    gmsh.finalize()


def callback(module, node, parm_name):
    module.fiber_display_changed({
        "node": node,
        "parm": node.parm(parm_name),
        "parm_name": parm_name,
        "script_multiparm_index": "1",
    })


def xform_matrix(node):
    return np.asarray(
        node.node("transform_1").geometry().attribValue("xform"),
        dtype=np.float64).reshape(4, 4)


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(
            BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_display_test_")
    mesh = os.path.join(work, "cube.msh")
    make_cube_msh(mesh)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "display_test")
    module = node.hdaModule()
    node.setParms({"working_dir": work + "/", "file_location1": mesh})
    node.parm("file_location1").pressButton()
    count = module.volume_element_count(node, 1)

    # A constant fiber is a JSON constant, but must still expand to an
    # element-wise viewport field and one line per element.
    node.setParms({
        "materials1_1": module.MATERIAL_TOKENS.index("HGOFiber"),
        "show_fibers1": 1,
    })
    node.parmTuple("fib_dir1_1").set((1, 0, 0))
    callback(module, node, "show_fibers1")
    stamped = node.node("fiberdata_1").geometry()
    assert stamped.findPrimAttrib("pf_fiber_1_0") is not None
    assert node.node("fiberviz_1").geometry().intrinsicValue(
        "primitivecount") == count

    # Surface and line colors must use the same attribute class after merging.
    merged = node.node("all").geometry()
    assert merged.findPointAttrib("Cd") is None
    assert merged.findPrimAttrib("Cd") is not None

    # Fibers sit inside a closed volume; the surface becomes translucent while
    # they are shown so the lines are not depth-occluded.
    surface = node.node("entitycolor_1").geometry()
    alpha = np.asarray(surface.primFloatAttribValues("Alpha"))
    assert len(alpha) and np.allclose(alpha, 0.2)

    # Uniform fiber color is authored on line primitives, not points.
    node.setParms({"fiber_color_mode1": 2})
    node.parmTuple("fiber_color1").set((0.25, 0.5, 0.75))
    callback(module, node, "fiber_color_mode1")
    lines = node.node("fiberviz_1").geometry()
    line_cd = np.asarray(lines.primFloatAttribValues("Cd")).reshape(-1, 3)
    assert len(line_cd) and np.allclose(line_cd, (0.25, 0.5, 0.75))

    # Selecting a surface field must perform the structural refresh itself.
    node.parm("color_by1").set("fiber_rgb")
    callback(module, node, "color_by1")
    surface_cd = np.asarray(
        node.node("entitycolor_1").geometry().primFloatAttribValues("Cd")
    ).reshape(-1, 3)
    assert len(surface_cd) and np.allclose(surface_cd, (1, 0, 0))

    # Material callbacks use the outer geometry multiparm index and update an
    # already-visible preview without reconstructing the whole geometry tree.
    node.parmTuple("fib_dir1_1").set((0, 1, 0))
    module.material_display_changed({
        "node": node,
        "script_multiparm_index2": "1",
        "script_multiparm_index": "1",
    })
    surface_cd = np.asarray(
        node.node("entitycolor_1").geometry().primFloatAttribValues("Cd")
    ).reshape(-1, 3)
    assert len(surface_cd) and np.allclose(surface_cd, (0, 1, 0))

    # PolyFEM applies geometry[].transformation to mesh points only. The HDA
    # must show and export the unchanged simulation/world reference a0, rather
    # than silently rotating it with the mesh.
    node.parmTuple("xform_r__1").set((0, 0, 90))
    node.node("fiberviz_1").cook(force=True)
    lines = node.node("fiberviz_1").geometry()
    reference = np.asarray(
        lines.primFloatAttribValues("fiber_reference_direction")
    ).reshape(-1, 3)
    assert len(reference) and np.allclose(reference, (0, 1, 0)), reference[:3]
    assert set(lines.primStringAttribValues("fiber_frame")) == {
        "simulation_world_reference_a0"}
    assert node.node("fiberdata_1").geometry().attribValue(
        "polyfem_fiber_reference_frame") == \
        "simulation_world_reference_a0"
    material = module.build_material(node, 1, 1)
    assert np.allclose(material["fiber_direction"], (0, 1, 0))

    # A Fiber SOP can remain in the mesh's original coordinates. Element
    # correspondence is resolved before the HDA transform, while its vector
    # values are passed through unchanged, matching PolyFEM.
    node.allowEditingOfContents()
    source = node.createNode("attribwrangle", "untransformed_fiber_source")
    source.setInput(0, node.node("geo_1"))
    source.setParms({"class": 1, "snippet": "v@fiber_world = {1, 0, 0};"})
    node.setParms({
        "fib_source1_1": "attribute",
        "fib_sop1_1": "untransformed_fiber_source",
        "fib_attrib1_1": "fiber_world",
    })
    fields, resolved = module.per_element_data(node, 1)
    assert resolved == count
    assert np.allclose(fields["pf_fiber_1_0"], (1, 0, 0))
    node.setParms({"fib_source1_1": "constant"})

    # Constant kappa must also be available to the viewport without becoming a
    # per-element solver file.
    node.setParms({
        "materials1_1": module.MATERIAL_TOKENS.index("HGODispersion"),
        "kappa_source1_1": "constant",
        "kappa1_1": 0.2,
        "color_by1": "kappa",
    })
    callback(module, node, "color_by1")
    assert node.node("fiberdata_1").geometry().findPrimAttrib(
        "pf_kappa_1_0") is not None

    # The symbolic Mooney-Rivlin token must agree across the menu, builder and
    # PolyFEM schema, including when used as a composite matrix.
    node.setParms({
        "materials1_1": module.MATERIAL_TOKENS.index("MaterialSum"),
        "matrix_model1_1": "MooneyRivlin3ParamSymbolic",
        "num_fiber_families1_1": 0,
    })
    material = module.build_material(node, 1, 1)
    assert material["models"][0]["type"] == "MooneyRivlin3ParamSymbolic"

    # Build-time callback injection keeps a visible preview live, while the
    # rotation tuple no longer carries the destructive legacy callback.
    assert "material_display_changed" in node.parm(
        "materials1_1").parmTemplate().scriptCallback()
    assert "material_display_changed" in node.parmTuple(
        "fib_dir1_1").parmTemplate().scriptCallback()
    assert not node.parmTuple(
        "xform_r__1").parmTemplate().scriptCallback()
    damping_condition = node.parm(
        "damping1_1").parmTemplate().conditionals()[hou.parmCondType.HideWhen]
    assert "matrix_model1_1 != FixedCorotational" in damping_condition

    # params.json stores an effective, pivot-compensated TRS. Import must clear
    # the automatic centroid pivot so that the affine matrix round-trips.
    node.setParms({
        "materials1_1": module.MATERIAL_TOKENS.index("NeoHookean"),
        "show_fibers1": 0,
        "color_by1": "subdomains",
    })
    node.parmTuple("xform_t__1").set((2, 3, 4))
    node.parmTuple("xform_r__1").set((10, 20, 30))
    node.parmTuple("xform_s__1").set((1.5, 2, 0.5))
    before = xform_matrix(node)
    params_path = module.write_params_only({"node": node})

    restored = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "display_restored")
    restored.setParms({"old_input_dir": os.path.dirname(params_path)})
    restored.hdaModule().read_params({"node": restored})
    after = xform_matrix(restored)
    assert np.allclose(after, before, atol=1e-9), (before, after)
    assert np.allclose(restored.parmTuple("tpivot_1").eval(), (0, 0, 0))

    print("PASS: PolyFEM fiber/material display and transform regressions")
    print("workdir:", work)


if __name__ == "__main__":
    main()
