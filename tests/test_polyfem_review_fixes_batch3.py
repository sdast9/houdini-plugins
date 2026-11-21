"""Batch 3 regressions. Run with Houdini 22's hython after build_all.py."""

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import hou
import numpy as np

from test_polyfem_hda import BASE, POLYFEM_BIN


ASSET = "stevenabramowitch::dev::PolyFEM::2.0"
IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]
ROTATED = [0, -1, 0, 1, 0, 0, 0, 0, 1]
RAMP = {"type": "piecewise_linear", "points": [0, 1], "values": [0, 1]}


def write_msh(path, mixed=False):
    """A hex followed by a disconnected tet (the MSH Reader shows tets first)."""
    tet = [(2, 0, 0), (3, 0, 0), (2, 1, 0), (2, 0, 1)]
    hex_points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                  (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    points = hex_points + tet if mixed else tet
    cells = (["1 5 2 22 22 1 2 3 4 5 6 7 8", "2 4 2 11 11 9 10 11 12"]
             if mixed else ["1 4 2 11 11 1 2 3 4"])
    rows = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(points))]
    rows.extend(f"{i} {x} {y} {z}" for i, (x, y, z) in enumerate(points, 1))
    rows += ["$EndNodes", "$Elements", str(len(cells)), *cells, "$EndElements"]
    path.write_text("\n".join(rows) + "\n")
    return path


def write_obj(path):
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


class Batch3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for asset in ("sop_MSH_Reader.3.0.hdanc",
                      "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                      "object_readPVD.1.0.hdanc"):
            hou.hda.installFile(str(Path(BASE) / asset), force_use_assets=True)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="polyfem_batch3_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.mesh = write_msh(self.root / "tet.msh")
        self.node = self.new_node()
        self.mod = self.node.hdaModule()

    def new_node(self):
        node = hou.node("/obj").createNode(ASSET)
        self.addCleanup(node.destroy)
        node.setParms({"working_dir": str(self.root), "polyfem_bin": POLYFEM_BIN})
        return node

    def load_mesh(self, path=None):
        self.node.parm("file_location1").set(str(path or self.mesh))
        self.node.parm("file_location1").pressButton()
        self.assertIsNotNone(self.node.node("branch_1"))

    def import_data(self, data, node=None):
        node = node or self.node
        params = self.input / "params.json"
        params.write_text(json.dumps(data))
        node.parm("old_input_dir").set(str(params))
        node.hdaModule().read_params({"node": node})
        return node.evalParm("import_report")

    def data(self, material="NeoHookean", **properties):
        return {
            "geometry": [{"mesh": str(self.mesh), "volume_selection": 1001}],
            "materials": [dict(id=1001, type=material, **properties)],
            "time": {"t0": 0, "dt": .5, "time_steps": 2},
        }

    def build(self):
        return json.loads(Path(self.mod.build_params(self.node)).read_text())

    def test_import_preflight_and_undo_group(self):
        self.load_mesh()
        self.node.parm("E1_1").set(123456)
        old_branch = self.node.node("branch_1").sessionId()
        broken = self.root / "broken.msh"
        broken.write_text("garbage")
        for geometry in (None, {}, [], [None, {"type": "box"}], [{"mesh": True}],
                         [{"mesh": "missing.msh"}], [{"mesh": str(broken)}]):
            with self.subTest(geometry=geometry):
                self.import_data({"geometry": geometry})
                self.assertEqual(self.node.node("branch_1").sessionId(), old_branch)
                self.assertEqual(self.node.evalParm("E1_1"), 123456)
                self.assertEqual(Path(self.node.evalParm("working_dir")), self.root)

        entered = []
        undo_group = hou.undos.group

        @contextmanager
        def group(name):
            with undo_group(name):
                entered.append(name)
                yield

        original_clear = self.mod.clear_all_geos
        def clear(*args, **kwargs):
            self.assertEqual(entered, ["Import PolyFEM JSON"])
            return original_clear(*args, **kwargs)

        with mock.patch.object(hou.undos, "group", group), mock.patch.dict(
                self.mod.read_params.__globals__, {"clear_all_geos": clear}):
            self.import_data(self.data(E=53700, nu=.28))
        self.assertEqual(entered, ["Import PolyFEM JSON"])
        self.assertEqual(self.node.evalParm("E1_1"), 53700)

    def test_growth_precision_and_scalar_stiffness(self):
        self.import_data(self.data("GrowthNeoHookean", E=5e4, nu=.3,
                                   g12=.9999999, g33=.12345678912345678))
        material = self.build()["materials"][0]
        self.assertEqual(material["g12"], .9999999)
        self.assertEqual(material["g33"], .12345678912345678)
        self.assertEqual(self.mod._format_expr_box(123456789123456789),
                         "123456789123456789")
        self.assertEqual(self.mod._format_expr_box(1.0), "1")

        for name, properties, young, nu, shear in (
                ("SaintVenant", {"E": 84000, "nu": .4}, 84000, .4, 30000),
                ("HookeLinearElasticity", {"young": 50000, "nu": .25},
                 50000, .25, 20000),
                ("SaintVenant", {"lambda": 20000, "mu": 20000},
                 50000, .25, 20000),
                ("HookeLinearElasticity", {"lambda": 20000, "mu": 20000},
                 50000, .25, 20000)):
            with self.subTest(material=name, properties=properties):
                self.import_data(self.data(name, **properties))
                self.assertEqual(self.node.evalParm("elast_mode1_1"), 0)
                expected = [young] * 3 + [nu] * 3 + [shear] * 3
                np.testing.assert_allclose(
                    json.loads(self.node.evalParm("elast_tensor1_1")), expected)
                np.testing.assert_allclose(
                    self.build()["materials"][0]["elasticity_tensor"], expected)

        for name, fields in (
                ("InversionBarrier", {"lambda": 20000, "mu": 20000}),
                ("MaterialSum", {"models": [
                    {"type": "InversionBarrier", "lambda": 20000, "mu": 20000}]})):
            self.import_data(self.data(name, **fields))
            material = self.build()["materials"][0]
            if name == "MaterialSum":
                material = material["models"][0]
            self.assertEqual((material["E"], material["nu"]), (50000, .25))

    def test_legacy_migration_keeps_expressions_and_keyframes(self):
        self.load_mesh()
        self.node.setParms({"initial_conditions1_1": 1, "sideset_selection1_1": 1})
        self.node.parm("Boundary_Condition__1_1_1").set(1)
        vectors = {
            "rhs_vec": ("RHS", "[$F, 0, 0]"),
            "obstacle_disp_vec1": ("obstacle_disp1", "[`$F + 1`, 0, 0]"),
            "ic_vec1_1_1": ("condition_vector_1_1_1", "[$F, 0, 0]"),
            "bc_vec1_1_1_1": ("vector_1_1_1_1", "[$F, 0, 0]"),
        }
        for legacy, raw in vectors.values():
            self.node.parm(legacy).set(raw)
        bc = self.node.parm("vector_1_1_1_1")
        bc.setExpression('"[" + str(int(hou.frame())) + ",0,0]"',
                         hou.exprLanguage.Python)
        keys = bc.keyframes()
        original_frame = hou.frame()
        self.addCleanup(hou.setFrame, original_frame)
        hou.setFrame(3)
        status = io.StringIO()
        with redirect_stdout(status):
            self.mod.migrate_legacy_expr_vectors(self.node)
            self.mod.migrate_legacy_expr_vectors(self.node)
        self.assertEqual(status.getvalue().count("expression-driven legacy"), 1)
        for frame in (3, 11):
            hou.setFrame(frame)
            for base, (legacy, raw) in vectors.items():
                self.assertTrue(self.node.parmTuple(base).isAtDefault())
                result = self.mod._read_expr_vector(self.node, base, legacy, base)
                self.assertEqual(result, [frame + (legacy == "obstacle_disp1"), 0, 0])
                if legacy != bc.name():
                    self.assertEqual(self.node.parm(legacy).unexpandedString(), raw)
        self.assertEqual(bc.keyframes()[0].expression(), keys[0].expression())

    def test_saint_venant_axes_import_load_ui_and_export_gates(self):
        report = self.import_data(self.data("SaintVenant", E=90000, nu=.25,
                                            fiber_direction=ROTATED))
        self.assertIn("SaintVenant uses global coordinates", report)
        self.assertEqual(json.loads(self.node.evalParm("material_coord1_1")), ROTATED)
        self.node.updateParmStates()
        axes = self.node.parm("material_coord1_1")
        self.assertFalse(axes.isDisabled())
        self.assertFalse(axes.isHidden())
        section = self.node.type().definition().sections()["OnLoaded"].contents()
        status = io.StringIO()
        with redirect_stdout(status):
            exec(compile(section, "OnLoaded", "exec"), {"kwargs": {"node": self.node}})
        self.assertIn("ignores fiber_direction", status.getvalue())
        report = self.mod.check_setup(self.node)
        self.assertTrue(any("21-entry elasticity_tensor" in e for e in report["errors"]))
        for action in (self.mod.check_setup_button, self.mod.write_params,
                       self.mod.run_background, self.mod.write_params_only):
            with self.subTest(action=action.__name__):
                with self.assertRaisesRegex(hou.NodeError, "ignores fiber_direction"):
                    action({"node": self.node})
        axes.set(json.dumps([float(v) for v in IDENTITY]))
        self.node.updateParmStates()
        self.assertTrue(axes.isDisabled())
        self.assertFalse(axes.isHidden())
        self.assertNotIn("fiber_direction", self.build()["materials"][0])
        self.import_data(self.data("HookeLinearElasticity", E=90000, nu=.25,
                                   fiber_direction=ROTATED))
        self.assertEqual(self.build()["materials"][0]["fiber_direction"], ROTATED)
        self.node.updateParmStates()
        self.assertFalse(self.node.parm("material_coord1_1").isDisabled())

    def test_amips_boolean_and_spatial_expression(self):
        for raw, expected in (("true", True), ("false", False), ("0", False),
                              ("1", True), (True, True), ("if(x-1,1,0)", False)):
            with self.subTest(use_rest_pose=raw):
                report = self.import_data(self.data("AMIPS", use_rest_pose=raw))
                self.assertEqual(self.build()["materials"][0]["use_rest_pose"], expected)
                if raw == "if(x-1,1,0)":
                    self.assertIn("cannot be represented by the Use Rest Pose", report)
                    self.assertIn(raw, report)

    def test_scene_digest_and_extra_json_sidesets(self):
        data = self.data()
        obj = write_obj(self.root / "triangle.obj")
        data["geometry"].append({"mesh": str(obj), "is_obstacle": True})
        data["boundary_conditions"] = {"obstacle_displacements": [
            {"id": 102000, "value": [1, 0, 0], "interpolation": RAMP}]}
        self.import_data(data)
        self.assertTrue(self.node.parm("xform_t__2x").keyframes())
        exported = self.build()
        record = self.mod.load_scene_record(str(self.input))
        self.assertEqual(record["params_sha256"], self.mod._digest(exported))
        displacement = exported["boundary_conditions"]["obstacle_displacements"][0]
        displacement["value"] = [3, 0, 0]
        other = self.new_node()
        report = self.import_data(exported, other)
        self.assertIn("hda_scene.json does not match params.json", report)
        self.assertAlmostEqual(other.parm("xform_t__2x").evalAtFrame(
            self.mod.timeline(other).frame(1)), 3)
        self.assertIsNone(self.mod.load_scene_record(str(self.input)))

        self.import_data(self.data())
        self.node.setParms({"sideset_selection1_1": 1})
        self.node.setParms({"basegroup1_1_1": "*", "Boundary_Condition__1_1_1": 1})
        self.node.parm("boundary_type1_1_1_1").set(0)
        self.mod._set_expr_vector(self.node, "bc_vec1_1_1_1", "vector_1_1_1_1",
                                  [1, 0, 0])
        original = self.build()
        sideset = self.input / "hda_sidesets.json"
        self.assertTrue(sideset.is_file())
        entry = original["boundary_conditions"]["dirichlet_boundary"][0]
        self.node.parm("extra_params_json").set(json.dumps({
            "boundary_conditions": {"dirichlet_boundary": [dict(entry, value=[2, 0, 0])]}}))
        edited = self.build()
        self.assertIsNone(self.mod.load_sideset_record(str(self.input), edited)[0])
        restored = self.new_node()
        self.import_data(edited, restored)
        self.assertEqual(self.mod._read_expr_vector(
            restored, "bc_vec1_1_1_1", "vector_1_1_1_1", "bc"), [2, 0, 0])

        self.node.parm("extra_params_json").set("")
        self.build()
        self.assertTrue(sideset.is_file())
        geometry = [dict(original["geometry"][0], surface_selection=[])]
        self.node.parm("extra_params_json").set(json.dumps({"geometry": geometry}))
        self.build()
        self.assertFalse(sideset.exists())
        self.assertFalse((self.input / "hda_scene.json").exists())

        for override, problem in (({"states": []}, "states"),
                                  ({"space": {"remesh": {"enabled": True}}}, "space.remesh"),
                                  ({"input": {"checkpoint": {"path": "resume.bin"}}},
                                   "Resume Checkpoint")):
            with self.subTest(override=override):
                self.node.parm("extra_params_json").set(json.dumps(override))
                self.assertTrue(any(problem in e for e in
                                    self.mod.check_setup(self.node)["errors"]))
                with self.assertRaisesRegex(hou.NodeError, problem):
                    self.build()

    def test_legacy_curves_nodal_file_and_per_body_rhs(self):
        data = self.data()
        data["materials"][0]["id"] = 11
        data["geometry"][0].update(
            volume_selection=11, surface_selection=[{"id": 1111, "axis": "-x",
                                                     "position": 2.01}])
        data["geometry"].append({"mesh": str(write_obj(self.root / "obs.obj")),
                                 "is_obstacle": True, "surface_selection": 100021})
        interpolation = {"type": "piecewise_linear", "points": [0, .5, 1],
                         "values": [0, .25, 1]}
        bc = {"dirichlet_boundary": [
                  {"id": 1111, "value": [2, 0, 0], "interpolation": [RAMP]}],
              "obstacle_displacements": [
                  {"id": 100021, "value": [3, 0, 0], "interpolation": interpolation}],
              "nodal_neumann_boundary": ["point_loads.txt"],
              "rhs": [{"id": 11, "value": [0, -9.8, 0]}]}
        data["boundary_conditions"] = bc
        report = self.import_data(data)
        self.assertIn("point_loads.txt", report)
        self.assertIn("Per-body boundary_conditions.rhs", report)
        self.assertTrue(self.node.parmTuple("rhs_vec").isAtDefault())
        self.assertEqual(self.node.parm("time_curve1_1_1_1").evalAsString(), "ramp")
        line = self.mod.timeline(self.node)
        tx = self.node.parm("xform_t__2x")
        self.assertEqual(len(tx.keyframes()), 3)
        self.assertAlmostEqual(tx.evalAtFrame(line.frame(.5)), .75)
        self.assertAlmostEqual(tx.evalAtFrame(line.frame(1)), 3)
        out = self.build()["boundary_conditions"]["dirichlet_boundary"][0]
        self.assertIn("interpolation", out)
        self.assertEqual(out["value"], [2, 0, 0])
        modern = self.data()
        modern["boundary_conditions"] = {
            "nodal_neumann_boundary": ["modern.txt"],
            "rhs": [{"id": 1001, "value": [1, 0, 0]}],
        }
        report = self.import_data(modern)
        self.assertIn("modern.txt", report)
        self.assertIn("Per-body boundary_conditions.rhs", report)
        self.assertTrue(self.node.parmTuple("rhs_vec").isAtDefault())

    def test_medium_validation_and_restore(self):
        self.load_mesh(write_msh(self.root / "mixed.msh", mixed=True))
        user = self.node.createNode("null", "custom_1_work")
        self.mod.clear_geo_tree(self.node, 1)
        self.assertIsNotNone(self.node.node(user.name()))
        self.assertIsNotNone(self.node.node("branch_1"))
        self.mod.create_geo_tree({"node": self.node, "script_multiparm_index": "1"})

        self.node.setParms({"sideset_selection1_1": 1})
        surface = self.node.node("null_1").geometry()
        tet_points, hex_points = (set() for _ in range(2))
        for face in surface.prims():
            target = tet_points if face.intAttribValue("Entity") == 1 else hex_points
            target.update(vertex.point().number() for vertex in face.vertices())
        wrong = min(hex_points - tet_points)
        self.node.setParms({"Boundary_Condition__1_1_1": 1,
                            "boundary_type1_1_1_1": 5, "grouptype1_1_1": 1,
                            "basegroup1_1_1": str(wrong)})
        with self.assertRaisesRegex(hou.NodeError, "subdomain's surface"):
            self.mod.resolve_sidesets(self.node, 1, 2)
        self.node.parm("basegroup1_1_1").set(str(min(tet_points)))
        self.mod.resolve_sidesets(self.node, 1, 2)

        self.assertFalse(self.mod._same_value(0, "t*(t-0.37)*(t-1.93)"))
        self.assertTrue(self.mod._same_value(0, "0*t"))
        self.mod._set_expr_vector(self.node, "rhs_vec", "RHS", [None, 2, None])
        self.assertEqual(json.loads(self.node.parm("RHS").unexpandedString()), [0, 2, 0])
        self.assertFalse(self.mod._looks_legacy({"solver": None}))
        data = self.data()
        data.update(solver=None, output={"data": {"diagnostic_state": " "}})
        self.import_data(data)
        self.assertFalse(self.node.evalParm("state_file"))
        data["output"]["data"]["diagnostic_state"] = "state.txt"
        self.import_data(data)
        self.assertTrue(self.node.evalParm("state_file"))
        self.node.parm("materials1_1").set("MaterialSum")
        for malformed in ("[{}]", '["bad"]', '[{"type": 3}]', '[{"type": ""}]'):
            self.node.parm("extra_models_json1_1").set(malformed)
            with self.assertRaisesRegex(hou.NodeError, 'Extra Models entry 1'):
                self.mod._check_composite_consistency(self.node)

        checkpoint = self.input / "checkpoint.h5"
        checkpoint.touch()
        self.import_data({"input": {"checkpoint": {
            "path": checkpoint.name, "reorder": True}}})
        self.assertTrue(self.build()["input"]["checkpoint"]["reorder"])
        self.import_data(self.data())
        self.assertFalse(self.node.evalParm("resume_reorder"))

        parm = mock.MagicMock()
        text = "solver: $F and `pwd`"
        parm.unexpandedString.return_value = text
        node = mock.Mock()
        node.parm.return_value = parm
        self.mod._set_run_status(node, text)
        parm.set.assert_not_called()
        parm.eval.assert_not_called()
        self.mod._set_run_status(node, text + " different")
        parm.set.assert_called_once_with(text + " different")

        for text, error in (("bogus(t)/2", "unknown name"),
                            ("-0.01*t/(1+", "unclosed")):
            self.assertIn(error, self.mod._validate_tinyexpr(text, "test"))
        for text in ("-0.01*t/(1+t)", "/tmp/input", "./data/field", "../field",
                     "~/data/scalar", r"C:\data\field", "relative/field.csv"):
            self.assertIsNone(self.mod._validate_tinyexpr(text, "test"), text)

    def source_geo(self):
        source = hou.node("/obj").createNode("geo")
        self.addCleanup(source.destroy)
        return source

    def load_sop(self, sop, obstacle=False):
        self.node.setParms({"mesh_source1": "sop", "sop_path1": sop.path(),
                            "is_obstacle1": int(obstacle)})
        return self.mod.load_sop_mesh({"node": self.node, "script_multiparm_index": "1"})

    def test_sop_duplicate_and_failed_reload(self):
        source = self.source_geo()
        tet = source.createNode("MSH_Reader::3.0")
        tet.parm("File").set(str(self.mesh))
        surface = source.createNode("grid")
        for sop, obstacle, ext, stem in ((tet, False, "msh", "sop_geo"),
                                         (surface, True, "obj", "sop_obs")):
            with self.subTest(obstacle=obstacle):
                self.mod.clear_all_geos({"node": self.node}, _confirmed=True)
                self.assertTrue(self.load_sop(sop, obstacle))
                self.assertEqual(self.mod.geo_duplicate({
                    "node": self.node, "script_multiparm_index": "1"}), 2)
                files = [Path(self.node.evalParm(f"file_location{index}"))
                         for index in (1, 2)]
                self.assertEqual([path.name for path in files],
                                 [f"{stem}1.{ext}", f"{stem}2.{ext}"])
                copied = files[1].read_bytes()
                self.assertEqual(files[0].read_bytes(), copied)
                xform = source.createNode("xform")
                xform.setInput(0, sop)
                xform.parm("tx").set(1)
                self.assertTrue(self.load_sop(xform, obstacle))
                self.assertNotEqual(files[0].read_bytes(), copied)
                self.assertEqual(files[1].read_bytes(), copied)
                self.node.parm("sop_path1").set("/obj/missing_sop")
                self.assertFalse(self.mod.load_sop_mesh({
                    "node": self.node, "script_multiparm_index": "1"}))
                self.assertEqual(self.node.evalParm("file_location1"), "")
                self.assertTrue(any("Geometry 1 has no mesh" in error
                                    for error in self.mod.check_setup(self.node)["errors"]))
                self.assertEqual(files[1].read_bytes(), copied)

    def sorted_prims(self, source):
        sort = source.parent().createNode("sort")
        sort.setInput(0, source)
        order = sort.parm("primsort")
        order.set(order.menuLabels().index("Reverse"))
        return sort

    def set_cell_data(self, source):
        attributes = source.parent().createNode("attribwrangle")
        attributes.setInput(0, source)
        attributes.setParms({"class": 1, "snippet": (
            "v@fiber = @numvtx == 8 ? set(0,1,0) : set(0,0,1);\n"
            "f@dispersion = @numvtx == 8 ? 0.2 : 0.1;")})
        for vol in (1, 2):
            slot = f"1_{vol}"
            self.node.setParms({
                f"materials{slot}": "HGODispersion",
                f"fib_source{slot}": "attribute", f"fib_sop{slot}": attributes.path(),
                f"fib_attrib{slot}": "fiber",
                f"kappa_source{slot}": "attribute", f"kappa_sop{slot}": attributes.path(),
                f"kappa_attrib{slot}": "dispersion"})

    def assert_solver_cell_files(self, node, body_ids=(1002, 1001)):
        model = node.hdaModule()
        volume_file = model.export_volumes(node, 1, str(self.input))
        np.testing.assert_array_equal(
            np.loadtxt(self.input / volume_file, dtype=int), body_ids)
        model.export_per_element_materials(node, str(self.input), str(self.root / "output"))
        for vol in (1, 2):
            fiber = model.read_cell_vectors_legacy_vtk(
                str(self.input / "fibers.vtk"), f"FIB_1_{vol}")
            np.testing.assert_allclose(fiber, [[0, 1, 0], [0, 0, 1]])
            np.testing.assert_allclose(
                np.loadtxt(self.input / f"pe_kappa_1_{vol}.txt"), [.2, .1], atol=1e-8)
        with np.load(self.root / "output" / model.FIBER_FAMILY_FILE) as companion:
            np.testing.assert_array_equal(companion["body_ids"], body_ids)
            np.testing.assert_allclose(companion["centroids"],
                                       [[.5, .5, .5], [2.25, .25, .25]])
            np.testing.assert_allclose(
                companion["directions"], [[[0, 0, 0], [0, 0, 1]],
                                          [[0, 1, 0], [0, 0, 0]]])

    def assert_mixed_solver_binding(self):
        for vol in (1, 2):
            self.node.parm(f"sideset_selection1_{vol}").set(1)
            self.node.setParms({f"basegroup1_{vol}_1": "*",
                                f"Boundary_Condition__1_{vol}_1": 1})
        self.node.setParms({"materials_fields": 1, "minimal_fields": 0,
                            "body_ids_fields": 1, "high_order_mesh": 1,
                            "enable": 0, "tend": .1, "dt": .1})
        node = hou.node("/obj").createNode("readPVD::1.0")
        self.addCleanup(node.destroy)
        readpvd = node.hdaModule()
        output = self.root / "output"
        # PolyFEM requires a common fiber file/field when several material
        # entries supply per-element directions. Verify each body's data in
        # turn with a single per-element source per solve.
        for fiber_vol, axis, kappa in ((1, "z", .1), (2, "y", .2)):
            self.node.setParms({
                f"materials1_{fiber_vol}": "HGODispersion",
                f"materials1_{3 - fiber_vol}": "NeoHookean"})
            self.build()
            result = subprocess.run(
                [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
                 "--max_threads", "2", "--log_level", "warning"],
                cwd=self.input, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            frame = next(path for path in output.glob("step_0.*")
                         if path.suffix in (".vtu", ".hdf", ".hdf5"))
            mesh = readpvd.read_mesh_cached(str(frame))
            point = mesh["point_data"]
            hex_points = mesh["points"][:, 0] < 1.5
            np.testing.assert_array_equal(point["body_ids"].ravel(),
                                          np.where(hex_points, 1002, 1001))
            owned = hex_points if fiber_vol == 2 else ~hex_points
            np.testing.assert_allclose(
                point[f"HGODispersion/fiber_direction_{axis}"].ravel(), owned)
            np.testing.assert_allclose(point["HGODispersion/kappa"].ravel(),
                                       np.where(owned, kappa, 0))
            topology = readpvd.volume_topology(mesh)
            np.testing.assert_array_equal(topology["hex"], [True, False])
            match = readpvd.mesh_match(str(output / "sim.pvd"), mesh, topology)
            np.testing.assert_array_equal(match["element_number"], [1, 2])

    def test_mixed_msh_sort_and_inverse_import(self):
        mixed = write_msh(self.root / "mixed.msh", mixed=True)
        self.load_mesh(mixed)
        branch = self.node.node("branch_1")
        self.set_cell_data(self.node.node("geo_1"))
        np.testing.assert_array_equal(self.mod.solver_element_order(branch.geometry()),
                                      [1, 0])
        self.assert_solver_cell_files(self.node)

        original_input = branch.inputs()[0]
        sort = self.sorted_prims(original_input)
        branch.setInput(0, sort)
        np.testing.assert_array_equal(
            sort.geometry().primIntAttribValues("msh_element_order"), [0, 1])
        np.testing.assert_array_equal(self.mod.solver_element_order(branch.geometry()),
                                      [0, 1])
        self.assert_solver_cell_files(self.node)
        params = self.build()

        # The fallback has only solver-order vectors, kappa and body IDs.
        (self.input / "hda_scene.json").unlink()
        other = self.new_node()
        self.import_data(params, other)
        prims = other.node("branch_1").geometry().primIntAttribValues("Entity")
        self.assertEqual(prims, (1, 2))
        fields, count = self.mod.per_element_data(other, 1)
        self.assertEqual(count, 2)
        np.testing.assert_allclose(fields[self.mod.fiber_attrib(1, None)],
                                   [[0, 0, 1], [0, 1, 0]])
        self.assert_solver_cell_files(other)

        broken = self.node.createNode("attribwrangle")
        broken.setInput(0, sort)
        branch.setInput(0, broken)
        for snippet in ("i@msh_element_order = -1;", "i@msh_element_order = 0;",
                        "i@msh_element_order = @primnum * 2;"):
            broken.setParms({"class": 1, "snippet": snippet})
            with self.assertRaisesRegex(hou.NodeError, "msh_element_order"):
                self.mod.export_volumes(self.node, 1, str(self.input))
        branch.setInput(0, sort)
        self.assert_mixed_solver_binding()

    def test_mixed_direct_sop_uses_authored_prim_order(self):
        source = self.source_geo()
        reader = source.createNode("MSH_Reader::3.0")
        reader.parm("File").set(str(write_msh(self.root / "mixed.msh", mixed=True)))
        sop = self.sorted_prims(reader)
        self.assertEqual(sop.geometry().prims()[0].type(), hou.primType.Hexahedron)
        np.testing.assert_array_equal(self.mod.solver_element_order(
            sop.geometry(), is_direct_sop=True), [0, 1])
        self.assertTrue(self.load_sop(sop))
        self.set_cell_data(sop)
        output_mesh = Path(self.node.evalParm("file_location1"))
        rows = output_mesh.read_text().split("$Elements\n", 1)[1].splitlines()
        self.assertEqual([row.split()[1] for row in rows[1:3]], ["5", "4"])
        self.assert_solver_cell_files(self.node)
        self.assert_mixed_solver_binding()

    def test_stop_refuses_pid_reuse_and_accepts_matching_run(self):
        output = self.root / "output"
        output.mkdir(exist_ok=True)
        decoy_dir = self.root / "other" / "input"
        decoy_dir.mkdir(parents=True)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", "-j", "params.json"],
            cwd=decoy_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def clean_child():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
        self.addCleanup(clean_child)

        import psutil
        process = psutil.Process(child.pid)
        record = {"kind": "background", "pid": child.pid, "started": time.time(),
                  "binary": process.exe(), "process_created": process.create_time()}
        (output / self.mod._RUN_RECORD).write_text(json.dumps(record))
        manifest = {"process": {"pid": child.pid,
                                 "executable": {"path": process.exe()}},
                    "completion": {"status": "running"}}
        (output / "run-manifest.json").write_text(json.dumps(manifest))
        run = self.mod.reattach_run(self.node)
        self.assertIsNotNone(run)
        self.assertIsNotNone(run["result"])
        self.addCleanup(self.mod._runs().pop, str(output), None)
        with mock.patch.object(self.mod.os, "kill") as kill, \
                mock.patch.object(self.mod.os, "killpg") as killpg:
            self.assertFalse(self.mod.stop_run({"node": self.node}))
            kill.assert_not_called()
            killpg.assert_not_called()
        self.assertIsNone(child.poll())

        expected = str(decoy_dir / "params.json")
        identity = {"cwd": str(decoy_dir), "binary": process.exe()}
        self.assertTrue(self.mod._process_alive(
            child.pid, expected, created=process.create_time(), **identity))
        self.assertFalse(self.mod._process_alive(
            child.pid, expected, created=process.create_time() - 60, **identity))
        self.assertFalse(self.mod._process_alive(
            child.pid, str(decoy_dir / "different.json"), **identity))
        self.assertFalse(self.mod._process_alive(
            child.pid, expected, cwd=str(self.input), binary=process.exe()))

        # Matching folder, argv and creation time survive reattachment. Only
        # the final signal is mocked, to prove which PID the action selects.
        matching_output = decoy_dir.parent / "output"
        matching_output.mkdir()
        (matching_output / self.mod._RUN_RECORD).write_text(json.dumps(record))
        self.node.parm("working_dir").set(str(decoy_dir.parent))
        run = self.mod.reattach_run(self.node)
        self.addCleanup(self.mod._runs().pop, str(matching_output), None)
        self.assertIsNone(run["result"])
        with mock.patch.object(self.mod.os, "kill") as kill:
            self.assertTrue(self.mod.stop_run({"node": self.node}))
            self.assertEqual(kill.call_args.args[0], child.pid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
