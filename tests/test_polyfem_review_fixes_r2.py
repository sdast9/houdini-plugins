"""Round 2 Batch 3 regressions. Run with Houdini 22 hython after build_all.py."""

import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import hou
import numpy as np

from test_polyfem_hda import BASE, POLYFEM_BIN
from test_polyfem_review_fixes_batch3 import ASSET, write_msh, write_obj


class Round2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for asset in ("sop_MSH_Reader.3.0.hdanc",
                      "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                      "object_readPVD.1.0.hdanc"):
            hou.hda.installFile(str(Path(BASE) / asset), force_use_assets=True)

    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="polyfem_round2_")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.node = hou.node("/obj").createNode(ASSET)
        self.addCleanup(self.node.destroy)
        self.node.setParms({"working_dir": str(self.root), "polyfem_bin": POLYFEM_BIN})
        self.mod = self.node.hdaModule()
        self.mesh = write_msh(self.root / "tet.msh")

    def load(self, geo=1, mesh=None):
        if self.node.evalParm("num_geos") < geo:
            self.node.parm("num_geos").set(geo)
        parm = self.node.parm(f"file_location{geo}")
        parm.set(str(mesh or self.mesh))
        parm.pressButton()
        self.assertIsNotNone(self.node.node(f"geo_{geo}"))

    def import_scene(self, data):
        path = self.input / "params.json"
        path.write_text(json.dumps(data))
        self.node.parm("old_input_dir").set(str(path))
        self.mod.read_params({"node": self.node})
        return self.node.evalParm("import_report")

    def build(self):
        return json.loads(Path(self.mod.build_params(self.node)).read_text())

    def fiber_file(self):
        file = self.root / "local-fibers.txt"
        np.savetxt(file, [[0, 0, 1]])
        return str(file)

    def test_real_multibody_hgo_solver(self):
        self.load()
        self.load(2)
        self.node.setParms({
            "materials1_1": "HGODispersion", "materials2_1": "HGODispersion",
            "fib_source1_1": "file", "fib_file1_1": self.fiber_file(),
            "xform_t__2x": 3, "enable": 0, "quasistatic": 0,
            "tend": .2, "dt": .1, "max_threads": 2,
            "materials_fields": 1, "minimal_fields": 0, "body_ids_fields": 1,
            "use_hdf5": 0,
        })
        self.node.parmTuple("fib_dir2_1").set((0, 1, 0))
        relative_binary = os.path.relpath(POLYFEM_BIN)
        self.node.parm("polyfem_bin").set(relative_binary)
        self.assertFalse(self.mod.check_setup(self.node, "background")["errors"])
        data = self.build()
        first, second = (m["fiber_direction"] for m in data["materials"])
        self.assertEqual(first, second)
        np.testing.assert_array_equal(
            self.mod.read_cell_vectors_legacy_vtk(self.input / "fibers.vtk", first["field"]),
            [[0, 0, 1], [0, 1, 0]])
        self.assertEqual((self.input / "fibers.vtk").read_text().count("VECTORS "), 1)
        cwd, command = self.mod._launch_command(self.node, str(self.input / "params.json"))
        self.assertEqual(command[0], os.path.abspath(POLYFEM_BIN))
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
        evidence = os.environ.get("POLYFEM_R2_EVIDENCE_DIR")
        if evidence:
            (Path(evidence) / "multibody-solver.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "output"
        manifest = json.loads((output / "run-manifest.json").read_text())
        self.assertEqual(manifest["completion"]["status"], "completed")
        frames = sorted(output.glob("*.vtu"))
        self.assertTrue(frames, list(output.iterdir()))
        if evidence:
            for artifact, filename in ((self.input / "params.json", "params.json"),
                                       (self.input / "fibers.vtk", "fibers.vtk"),
                                       (output / "run-manifest.json", "run-manifest.json"),
                                       (frames[-1], "result.vtu")):
                shutil.copyfile(artifact, Path(evidence) / f"multibody-{filename}")
        reader = hou.node("/obj").createNode("readPVD::1.0")
        self.addCleanup(reader.destroy)
        points = reader.hdaModule().read_mesh_cached(str(frames[-1]))["point_data"]
        bodies = points["body_ids"].ravel()
        fields = [name for name in points if name.endswith("fiber_direction_x")]
        self.assertEqual(len(fields), 1, sorted(points))
        prefix = fields[0][:-1]
        directions = np.column_stack([points[prefix + axis].ravel() for axis in "xyz"])
        counts = {}
        for body, direction in ((1001, [0, 0, 1]), (2001, [0, 1, 0])):
            owned = directions[bodies == body]
            self.assertGreaterEqual(len(owned), 4)
            np.testing.assert_array_equal(owned, np.tile(direction, (len(owned), 1)))
            counts[body] = len(owned)
        print(f"Verified exact PolyFEM VTU fiber directions for both bodies: {counts}")
        report = self.import_scene(data)
        self.assertNotIn("no longer reproduce", report)
        self.assertEqual(self.node.parm("fib_source1_1").evalAsString(), "file")
        self.assertEqual(self.node.parm("fib_source2_1").evalAsString(), "constant")
        self.assertEqual(self.node.parmTuple("fib_dir2_1").eval(), (0, 1, 0))

    def test_shared_composite_fiber_slots_and_refusals(self):
        self.load()
        self.load(2)
        for geo in (1, 2):
            self.node.setParms({
                f"materials{geo}_1": "MaterialSum",
                f"matrix_model{geo}_1": "NeoHookean",
                f"num_fiber_families{geo}_1": 2,
                f"fam_model{geo}_1_1": "HGOFiber",
                f"fam_model{geo}_1_2": "HGOFiber",
            })
        attribute = self.node.createNode("attribwrangle", "fiber_attribute")
        attribute.setInput(0, self.node.node("branch_1"))
        attribute.setParms({"class": 1, "snippet": "v@r2_fiber = {0, 0, 1};"})
        self.node.setParms({"fam_fib_source1_1_1": "attribute",
                            "fam_fib_sop1_1_1": attribute.path(),
                            "fam_fib_attrib1_1_1": "r2_fiber"})
        self.node.parmTuple("fam_fib_dir2_1_1").set((0, 1, 0))
        first, second = (m["models"] for m in self.build()["materials"])
        reference = first[1]["fiber_direction"]
        self.assertEqual(reference, second[1]["fiber_direction"])
        np.testing.assert_array_equal(
            self.mod.read_cell_vectors_legacy_vtk(self.input / "fibers.vtk", reference["field"]),
            [[0, 0, 1], [0, 1, 0]])
        self.assertIsInstance(second[2]["fiber_direction"], list)
        self.node.setParms({"fam_fib_source2_1_1": "expression",
                            "fam_fib_expr2_1_1": '["1+t",0,0]'})
        with self.assertRaisesRegex(hou.NodeError, r"geo 1.*geo 2.*HGOFiber MaterialSum slot 2.*expression"):
            self.mod.build_params(self.node)
        for geo in (1, 2):
            external = {"type": "HGODispersion", "fiber_direction": {
                "type": "per_element_file", "path": f"external{geo}.vtk", "field": "fiber"}}
            self.node.setParms({f"num_fiber_families{geo}_1": 0,
                                f"extra_models_json{geo}_1": json.dumps([external])})
        with self.assertRaisesRegex(hou.NodeError, "incompatible external per-element"):
            self.mod.check_setup_button({"node": self.node})

    def test_shared_anisotropic_refusal_in_check_write_and_run(self):
        self.load()
        self.load(2)
        for model in ("SaintVenant", "HookeLinearElasticity"):
            with self.subTest(model=model):
                self.node.setParms({"materials1_1": model, "materials2_1": "MaterialSum",
                                    "matrix_model2_1": model})
                self.node.parmTuple("ortho_E2_1").set((2e5, 2e5, 2e5))
                pattern = rf"geo 1 subdomain 1, geo 2 subdomain 1.*shares one elasticity tensor.*{model}"
                for action in (self.mod.check_setup_button, self.mod.write_params_only,
                               self.mod.write_params, self.mod.run_background):
                    with self.assertRaisesRegex(hou.NodeError, pattern):
                        action({"node": self.node})
                for geo in (1, 2):
                    self.node.setParms({f"elast_mode{geo}_1": "raw",
                                        f"elast_tensor{geo}_1": '[1,1,1,0,0,0,1,1,"x"]'})
                with self.assertRaisesRegex(hou.NodeError, pattern):
                    self.mod.build_params(self.node)
                for geo in (1, 2):
                    self.node.parm(f"elast_mode{geo}_1").set("orthotropic")
                    self.node.parm(f"elast_tensor{geo}_1").revertToDefaults()
        self.assertFalse((self.input / "params.json").exists())

    def test_sop_subdomain_detection_and_homogeneous_provenance(self):
        source = hou.node("/obj").createNode("geo")
        self.addCleanup(source.destroy)
        reader = source.createNode("MSH_Reader::3.0")
        reader.parm("File").set(str(self.mesh))
        wrangle = source.createNode("attribwrangle")
        wrangle.setInput(0, reader)
        wrangle.setParms({"class": 1,
                          "snippet": "i@class=17; i@subdomain=21; i@entity=35;"})
        wrangle_geo = wrangle.geometry().freeze()
        wrangle_geo.renamePrimAttrib("Entity", "old_Entity")
        wrangle_geo.renamePrimAttrib("entity", "old_entity")
        wrangle_geo.addAttrib(hou.attribType.Prim, "entity", (40, 41))
        path = self.root / "sop.msh"
        self.mod._write_sop_volume_msh(wrangle_geo, str(path))
        self.assertIn("1 4 2 21 21", path.read_text())
        wrangle_geo.renamePrimAttrib("subdomain", "old_subdomain")
        self.mod._write_sop_volume_msh(wrangle_geo, str(path))
        self.assertIn("1 4 2 1 1", path.read_text())
        self.assertTrue(any("integer tuple" in n for n in
                            self.mod._sop_subdomain_notes(wrangle_geo, 1)))

        self.node.setParms({"mesh_source1": "sop", "sop_path1": wrangle.path()})
        self.assertTrue(self.mod.load_sop_mesh({"node": self.node}))
        self.assertTrue(any("ignored primitive 'class'" in n for n in
                            self.mod.check_setup(self.node)["notes"]))
        branch = self.node.node("branch_1")
        bad = self.node.createNode("attribwrangle", "strip_msh_order")
        bad.setInput(0, branch.input(0))
        bad.setParms({"class": 0, "snippet": 'removeprimattrib(0, "msh_element_order");'})
        branch.setInput(0, bad)
        self.assertTrue(any("Invalid msh_element_order" in e for e in
                            self.mod.check_setup(self.node, "write")["errors"]))
        with self.assertRaisesRegex(hou.NodeError, "Invalid msh_element_order"):
            self.mod.build_params(self.node)

    def test_vector_markers_and_live_legacy_tensor(self):
        self.load()
        node = self.node
        node.setParms({"sideset_selection1_1": 1, "initial_conditions1_1": 1})
        node.parm("Boundary_Condition__1_1_1").set(1)
        pairs = (("rhs_vec", "RHS"), ("ic_vec1_1_1", "condition_vector_1_1_1"),
                 ("bc_vec1_1_1_1", "vector_1_1_1_1"),
                 ("obstacle_disp_vec1", "obstacle_disp1"))
        for _, legacy in pairs:
            node.parm(legacy).set("[1,2,3]")
        raw = node.parm("elast_tensor1_1")
        node.addSpareParmTuple(hou.StringParmTemplate("tensor_source", "Tensor", 1))
        node.parm("tensor_source").set("[10,10,10,0,0,0,2,2,2]")
        raw.setExpression('chs("tensor_source")', hou.exprLanguage.Hscript)
        # Exercise the shipped callbacks, including a later reload/update pass.
        sections = node.type().definition().sections()
        for event in ("OnLoaded", "OnUpdated"):
            exec(compile(sections[event].contents(), event, "exec"), {"kwargs": {"node": node}})
        for visible, legacy in pairs:
            self.assertEqual(node.parmTuple(visible).eval(), ("1", "2", "3"))
            node.parmTuple(visible).revertToDefaults()
            node.parm(legacy).set("[4,5,6]")
        self.mod.migrate_legacy_expr_vectors(node)
        for visible, legacy in pairs:
            self.assertEqual(self.mod._read_expr_vector(node, visible, legacy, visible), [0, 0, 0])
        node.parm("tensor_source").set("[20,20,20,0,0,0,2,2,2]")
        self.assertTrue(raw.keyframes())
        self.assertEqual(self.mod._read_elast_tensor(node, 1, 1)[0], 20)

    def test_nested_import_paths_and_mesh_reservations(self):
        outside = self.root / "outside"
        outside.mkdir()
        external = write_msh(outside / "part.msh")
        local = write_msh(self.input / "part.msh")
        local.write_text(local.read_text().replace("2 3 0 0", "2 7 0 0"))
        original = local.read_bytes()
        self.assertNotEqual(original, external.read_bytes())
        report = self.import_scene({
            "geometry": [{"mesh": str(external), "volume_selection": 23, "unit": "mm",
                          "transformation": {"dimensions": [1, 1, 1]}},
                         {"mesh": local.name, "volume_selection": 23}],
            "materials": [{"id": 23, "type": "NeoHookean", "lambda": 20, "mu": 30,
                           "extension": {"a": 1}}],
            "space": {"basis_type": "spline", "discr_order": [{"id": 23, "order": 2}]},
            "solver": {"advanced": {"stall_restart": {"undocumented": True}}},
            "time": {"integrator": {"extra": 1}},
            "contact": {"adhesion": {"extra": 1}},
            "output": {"paraview": {"options": {"extra": 1}}},
            "boundary_conditions": {"dirichlet_boundary": [{"id": 1111, "extra": 1}]},
        })
        for path in ("space.basis_type", "solver.advanced.stall_restart.undocumented",
                     "geometry[0].unit", "geometry[0].transformation.dimensions",
                     "materials[0].extension.a", "time.integrator.extra",
                     "contact.adhesion.extra", "output.paraview.options.extra",
                     "boundary_conditions.dirichlet_boundary[0].extra"):
            self.assertIn(path, report)
        self.assertEqual(local.read_bytes(), original)
        self.assertNotEqual(self.node.evalParm("file_location1"), str(local))
        self.assertEqual(self.node.evalParm("file_location2"), str(local))
        self.assertEqual(self.node.evalParm("extra_params_json"), "")
        for geo in (1, 2):
            self.assertEqual(self.node.parm(f"materials{geo}_1").evalAsString(), "NeoHookean")
            self.assertAlmostEqual(self.node.evalParm(f"E{geo}_1"), 72)
            self.assertAlmostEqual(self.node.evalParm(f"nu{geo}_1"), .2)
            self.assertEqual(self.node.evalParm(f"mainOrder{geo}_1"), 1)

    def test_lame_import_for_isotropic_and_tensor_models(self):
        for token in ("LinearElasticity", "FixedCorotational", "IsochoricNeoHookean",
                      "SaintVenant", "HookeLinearElasticity"):
            with self.subTest(token=token):
                self.import_scene({
                    "geometry": [{"mesh": str(self.mesh), "volume_selection": 1001}],
                    "materials": [{"id": 1001, "type": token, "lambda": 20, "mu": 30}]})
                model = self.build()["materials"][0]
                if token in ("SaintVenant", "HookeLinearElasticity"):
                    np.testing.assert_allclose(model["elasticity_tensor"], [72]*3 + [.2]*3 + [30]*3)
                else:
                    self.assertAlmostEqual(model["E"], 72)
                    self.assertAlmostEqual(model["nu"], .2)

    def test_axis_angle_obstacle_alignment_and_malformed_import(self):
        broken = self.root / "broken.msh"
        broken.write_bytes(b"\xff\x00")
        obstacle = write_obj(self.root / "obstacle.obj")
        data = {
            "geometry": [{"type": "box"}, {"mesh": broken.name},
                         {"mesh": str(obstacle), "is_obstacle": True,
                          "surface_selection": 103000,
                          "transformation": {"rotation_mode": "axis_angle",
                                             "rotation": [90, 0, 0, 2]}},
                         {"mesh": str(self.mesh), "volume_selection": 1001,
                          "transformation": {"rotation_mode": "unknown", "rotation": [1, 2, 3]}}],
            "time": {"dt": [], "tend": 1, "integrator": {}},
            "contact": {"dhat": {}},
            "solver": {"max_threads": {}},
            "output": {"paraview": {"skip_frame": []}},
            "boundary_conditions": {"obstacle_displacements": [
                {"id": 103000, "value": [1, 2, 3]}]},
        }
        (self.input / broken.name).write_bytes(broken.read_bytes())
        report = self.import_scene(data)
        for issue in ("could not read mesh", "unsupported rotation_mode",
                      "time: could not restore dt", "contact: could not restore dhat",
                      "solver: could not restore max_threads", "output: could not restore skip_frame"):
            self.assertIn(issue, report)
        self.assertIsNotNone(self.node.node("geo_3"))
        matrix = self.mod.rest_matrix(self.node, 2)
        np.testing.assert_allclose(np.array([1, 0, 0]) @ matrix[:3, :3], [0, 1, 0], atol=1e-6)
        self.assertEqual(self.node.parmTuple("obstacle_disp_vec2").eval(), ("1", "2", "3"))
        path = self.input / "params.json"
        path.write_bytes(b"\xff")
        output = io.StringIO()
        with redirect_stdout(output):
            self.mod.read_params({"node": self.node})
        self.assertIn("Error reading JSON file", output.getvalue())
        self.assertIsNotNone(self.node.node("geo_3"))

    def test_remapped_obstacle_sidecar_and_interpolation_tables(self):
        from test_polyfem_review_fixes_batch3 import RAMP
        obstacle = write_obj(self.root / "obstacle.obj")
        self.node.setParms({"is_obstacle1": 1})
        self.load(mesh=obstacle)
        keys = []
        for frame, value in ((1, 0), (2, 1)):
            key = hou.Keyframe()
            key.setFrame(frame)
            key.setValue(value)
            key.setExpression("linear()", hou.exprLanguage.Hscript)
            keys.append(key)
        self.node.parm("xform_t__1x").setKeyframes(keys)
        self.load(2)
        data = self.build()
        record_path = self.input / "hda_scene.json"
        record = json.loads(record_path.read_text())
        obstacle_record = record["obstacles"].pop("1")
        obstacle_record["transform"] = {
            name.replace("1", "2"): value for name, value in obstacle_record["transform"].items()}
        record["obstacles"]["2"] = obstacle_record
        data["geometry"].insert(0, {"type": "box"})
        if isinstance(data["geometry"][1].get("surface_selection"), int):
            data["geometry"][1]["surface_selection"] = 102000
        for displacement in data["boundary_conditions"].get("obstacle_displacements", []):
            displacement["id"] = 102000
        record["params_sha256"] = self.mod._digest(data)
        record_path.write_text(json.dumps(record))
        self.import_scene(data)
        self.assertEqual(len(self.node.parm("xform_t__1x").keyframes()), 2)
        self.assertFalse(self.node.parm("xform_t__2x").keyframes())

        data["time"] = {"t0": 0, "dt": .5, "time_steps": 2}
        data["geometry"][1]["surface_selection"] = 102000
        entry = {"id": 102000, "value": [1, 2, 3], "interpolation": [RAMP, RAMP]}
        data["boundary_conditions"]["obstacle_displacements"] = [entry]
        report = self.import_scene(data)
        self.assertIn("third axis uses a constant multiplier", report)
        self.assertEqual(self.node.evalParm("xform_t__1z"), 3)
        for points, values in (([], []), ([1, 0], [0, 1]), ([0, 1], [1])):
            entry["interpolation"] = dict(RAMP, points=points, values=values)
            report = self.import_scene(data)
            self.assertIn("piecewise_linear interpolation needs", report)
            self.assertIsNotNone(self.node.node("geo_2"))

    def test_dirichlet_overlap_and_legacy_multiple_kinds(self):
        cond = {"type": 0, "dims": [True, False, False], "curves": [None]*3}
        zero = dict(cond, label="first", value=[0, 0, 0])
        late = dict(cond, label="second", value=["(abs(t-5)+t-5)/2", 0, 0])
        with self.assertRaisesRegex(hou.NodeError, "first and second.*different values"):
            self.mod.merge_conditions([zero, late], "overlapping faces")
        self.assertTrue(self.mod._same_value("sin( t )", "(sin(t))"))
        self.assertTrue(self.mod._same_value("0.0", 0))

        self.import_scene({
            "geometry": [{"mesh": str(self.mesh), "volume_selection": 11,
                          "surface_selection": [{"id": 1111, "axis": "x", "position": 2}]}],
            "materials": [{"id": 11, "type": "NeoHookean", "E": 1e4, "nu": .25}],
            "boundary_conditions": {"dirichlet_boundary": [{"id": 1111, "value": [1, 0, 0]}],
                                    "pressure_boundary": [{"id": 1111, "value": 6}]}},
        )
        self.assertEqual(self.node.evalParm("Boundary_Condition__1_1_1"), 2)
        self.assertEqual(self.node.evalParm("boundary_type1_1_1_1"), 0)
        self.assertEqual(self.node.evalParm("boundary_type1_1_1_2"), 3)
        self.assertEqual(float(self.node.evalParm("value_1_1_1_2")), 6)

    def test_sidecar_selection_deletion_and_time_override(self):
        self.load()
        self.node.parm("sideset_selection1_1").set(1)
        self.node.parm("sideset_selection1_1").pressButton()
        self.node.parm("basegroup1_1_1").set("0")
        data = self.build()
        record, _ = self.mod.load_sideset_record(str(self.input), data)
        self.assertIsNotNone(record)
        self.assertTrue(record["selection_files"])
        data["geometry"][0].pop("surface_selection")
        record, reason = self.mod.load_sideset_record(str(self.input), data)
        self.assertIsNone(record)
        self.assertIn("selection files referenced", reason)
        self.assertTrue((self.input / "hda_scene.json").exists())
        self.node.parm("extra_params_json").set('{"time":{"dt":0.0325}}')
        data = self.build()
        self.assertEqual(data["time"]["dt"], .0325)
        self.assertFalse((self.input / "hda_scene.json").exists())
        self.assertTrue((self.input / "hda_sidesets.json").exists())
        self.assertIn("Extra JSON changed", self.node.evalParm("export_report"))

    def test_axes_kappa_small_pulse_and_expression_resource_root(self):
        self.load()
        self.load(2)
        self.node.setParms({"materials1_1": "HookeLinearElasticity",
                            "materials2_1": "SaintVenant", "is_enabled2": 0,
                            "material_coord2_1": "[2,0,0,0,2,0,0,0,2]"})
        axes = self.node.parm("material_coord1_1")
        for identity in ("[1,0,0,0,1,0,0,0,1]", "[[1,0,0],[0,1,0],[0,0,1]]"):
            axes.set(identity)
            self.assertTrue(self.mod._material_axes_are_identity(identity))
            self.assertFalse(self.mod.check_setup(self.node, "write")["errors"])
            self.assertEqual(self.build()["materials"][0]["fiber_direction"], [1,0,0,0,1,0,0,0,1])
        for invalid in ("[2,0,0,0,2,0,0,0,2]", "[[-1,0,0],[0,1,0],[0,0,1]]"):
            axes.set(invalid)
            self.assertTrue(any("orthonormal right-handed" in msg for msg in
                                self.mod.check_setup(self.node, "write")["errors"]))
            with self.assertRaisesRegex(hou.NodeError, "orthonormal right-handed"):
                self.mod.build_params(self.node)
        self.node.setParms({"materials1_1": "HGODispersion", "kappa_source1_1": "file"})
        bad = self.root / "bad-kappa.txt"
        bad.write_text("nan\n")
        self.node.parm("kappa_file1_1").set(str(bad))
        with self.assertRaisesRegex(hou.NodeError, "finite and within"):
            self.mod.build_params(self.node)
        _, values = self.mod._compressed_table([0, 1, 2], [0, 1e-13, 0])
        np.testing.assert_array_equal(values, [0, 1e-13, 0])
        folder = self.input / "resources"
        folder.mkdir()
        (folder / "value").write_text("1\n")
        for expression, error in (("1e999", "finite"), ("t+", "syntax"),
                                  ("2x", "syntax"), ("sin()", "empty function")):
            self.assertIn(error, self.mod._validate_tinyexpr(expression, "test"))
        self.node.parm("kappa_source1_1").set("constant")
        self.node.parmTuple("rhs_vec").set(("resources/value", "0", "0"))
        self.assertFalse(self.mod.check_setup(self.node, "write")["errors"])

    @unittest.skipUnless(os.name == "posix", "POSIX ps required")
    def test_startup_pid_liveness_fallback_and_stop_refusal(self):
        self.load()
        output = self.root / "output"
        output.mkdir(exist_ok=True)
        params = str(self.input / "params.json")
        binary = "/bin/sh"
        started = time.time()
        child = subprocess.Popen([binary, "-c", "sleep 30 & wait", "test-polyfem",
                                  "-j", params], cwd=self.input, start_new_session=True)

        def cleanup():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)

        self.addCleanup(cleanup)
        self.addCleanup(self.mod._runs().pop, str(output), None)
        record = {"kind": "background", "pid": child.pid, "binary": binary, "started": started}
        record_path = output / self.mod._RUN_RECORD
        record_path.write_text(json.dumps(record))
        unavailable = (mock.patch.dict(sys.modules, {"psutil": None}),
                       mock.patch("psutil.Process", side_effect=__import__("psutil").AccessDenied))
        for adapter in unavailable:
            with adapter, self.subTest(adapter=adapter):
                self.mod._runs().pop(str(output), None)
                self.assertEqual(self.mod.active_run(self.node)["pid"], child.pid)
                self.assertFalse((output / "run-manifest.json").exists())
                self.assertIsNone(self.mod.write_params_only({"node": self.node}))
                self.assertFalse((self.input / "params.json").exists())
                run = self.mod.reattach_run(self.node)
                self.assertIsNone(run["result"])
                self.assertFalse(self.mod._process_alive(
                    child.pid, params + ".old", binary=binary, earliest=started-2))
                self.assertFalse(self.mod._process_alive(
                    child.pid, params, binary=binary, earliest=started-60, latest=started-55))
                messages = io.StringIO()
                with mock.patch.object(self.mod.os, "kill") as kill, \
                        mock.patch.object(self.mod.os, "killpg") as killpg, redirect_stdout(messages):
                    self.assertFalse(self.mod.stop_run({"node": self.node}))
                    kill.assert_not_called()
                    killpg.assert_not_called()
                self.assertIn("identity unknown", messages.getvalue())

    def test_viewer_restores_bars_and_preserves_keyed_obstacles(self):
        source = self.node.type().definition().sections()["ViewerStateModule"].contents()
        scope = {"__name__": "test_polyfem_viewer_state"}
        exec(compile(source, "ViewerStateModule", "exec"), scope)
        state = scope["State"].__new__(scope["State"])
        state.scene_viewer = mock.Mock()
        state.scene_viewer.isShowingOperationBar.return_value = False
        state.scene_viewer.isShowingSelectionBar.return_value = True
        state.handle = mock.Mock()
        state.handle_settings = ""
        state.geo_of_prim = 1
        state.node = None
        state.onStateToHandle({"parms": {}})
        state.onHandleToState({"parms": {}})
        state.onEnter({"node": self.node})
        self.node.parm("is_obstacle1").set(1)
        parm = self.node.parm("xform_t__1x")
        key = hou.Keyframe()
        key.setFrame(1)
        key.setValue(2)
        parm.setKeyframe(key)
        state.onHandleToState({"parms": {f"{kind}{axis}": 10 for kind in "trs" for axis in "xyz"}})
        self.assertEqual(parm.keyframes()[0].value(), 2)
        self.assertIn("keyed transform", state.scene_viewer.setPromptMessage.call_args.args[0])
        state.onExit({})
        state.scene_viewer.showOperationBar.assert_called_with(False)
        state.scene_viewer.showSelectionBar.assert_called_with(True)
        self.assertIsNone(state.node)


if __name__ == "__main__":
    unittest.main(verbosity=2)
