"""Exercise the packaged SOP import and standalone writeMSH under hython."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import hou
import numpy as np

from test_polyfem_hda import BASE, POLYFEM_BIN

sys.path.insert(0, str(Path(BASE) / "src"))
from common.msh_parser import read_msh


class SopMeshImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("sop_MSH_Reader.3.0.hdanc",
                     "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                     "sop_writeMSH.2.hdanc"):
            path = str(Path(BASE) / name)
            hou.hda.installFile(path, oplibraries_file=None, force_use_assets=True)
            assert hou.hda.definitionsInFile(path)[0].isCurrent(), path

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="polyfem SOP import ")
        self.root = Path(temp.name).resolve()
        self.addCleanup(temp.cleanup)
        hou.hipFile.setName(str(self.root / "test scene.hipnc"))
        obj = hou.node("/obj")
        self.source = obj.createNode("geo", "source_sops")
        self.addCleanup(self.source.destroy)
        self.box = self.source.createNode("box")

    def new_case(self):
        node = hou.node("/obj").createNode("stevenabramowitch::dev::PolyFEM::2.0")
        self.addCleanup(node.destroy)
        node.setParms({
            "polyfem_bin": POLYFEM_BIN, "quasistatic": 1, "define_run_by": 0,
            "tend": 0.1, "dt": 0.1, "enable": 0, "use_hdf5": 0,
        })
        return node

    def new_tet_sop(self, operator):
        node = self.source.createNode(operator)
        node.setInput(0, self.box)
        self.assertEqual(len(node.outputLabels()), 1)
        return node

    def run_solver(self, node, expected_mesh):
        module = node.hdaModule()
        for vol in range(1, node.evalParm("num_volumes1") + 1):
            node.setParms({f"sideset_selection1_{vol}": 1})
            module.create_group({"node": node, "script_multiparm_index2": "1",
                                 "script_multiparm_index": str(vol)})
            node.setParms({f"basegroup1_{vol}_1": "*",
                           f"Boundary_Condition__1_{vol}_1": 1})
        params = Path(module._prepare_inputs(node, "write"))
        self.assertEqual(params.parent, expected_mesh.parent)
        geometry, = json.loads(params.read_text())["geometry"]
        self.assertEqual(geometry["mesh"], expected_mesh.name)
        self.assertEqual((params.parent / geometry["mesh"]).resolve(), expected_mesh)
        result = subprocess.run(
            [POLYFEM_BIN, "-j", params.name, "--max_threads", "2"],
            cwd=params.parent, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((params.parent.parent / "output/sim.pvd").is_file())

    def check_tet_file(self, path, source, expected_tags, orient=True):
        self.assertTrue(path.read_bytes().startswith(b"$MeshFormat\n2.2 0 8\n"))
        geom = source.geometry()
        tets = [prim for prim in geom.prims() if prim.type() == hou.primType.Tetrahedron]
        original = np.array([[vertex.point().number() for vertex in prim.vertices()]
                             for prim in tets])
        self.assertGreater(len(tets), 0)
        mesh = read_msh(path)
        self.assertEqual(set(mesh["cells"]), {"tet"})
        np.testing.assert_array_equal(mesh["node_tags"], np.arange(1, len(geom.points()) + 1))
        np.testing.assert_array_equal(mesh["points"], [tuple(p.position()) for p in geom.points()])
        tet_data = mesh["cells"]["tet"]
        np.testing.assert_array_equal(tet_data["order"], np.arange(len(tets)))
        np.testing.assert_array_equal(tet_data["entity"], expected_tags)
        corners = tet_data["corners"]
        sort_corners = np.sort(corners, axis=1)
        np.testing.assert_array_equal(sort_corners, np.sort(original, axis=1))
        self.assertEqual(len(np.unique(sort_corners, axis=0)), len(tets))
        if orient:
            coords = mesh["points"][corners]
            signed_volume = np.linalg.det(coords[:, 1:] - coords[:, :1])
            self.assertTrue((signed_volume > 0).all())
        else:
            np.testing.assert_array_equal(corners, original)
        return len(tets)

    def load_sop(self, case, path):
        case.setParms({"mesh_source1": "sop", "sop_path1": path})
        case.parm("sop_path1").pressButton()

    def test_tet_conform_and_embed_direct_import(self):
        for operator in ("tetconform", "tetembed"):
            with self.subTest(operator=operator):
                source = self.new_tet_sop(operator)
                case = self.new_case()
                sop_path = source.path() if operator == "tetconform" else \
                    "op:" + case.relativePathTo(source)
                for skin in (False, True):
                    source.parm("outputsurftri").set(skin)
                    self.assertEqual(bool(source.geometry().countPrimType(hou.primType.Polygon)),
                                     skin)
                    self.load_sop(case, sop_path)
                    expected = self.root / "polyfem" / case.name() / "input/sop_geo1.msh"
                    self.assertEqual(case.parm("working_dir").unexpandedString(),
                                     "$HIP/polyfem/$OS/")
                    self.assertEqual(case.parm("file_location1").unexpandedString(),
                                     f"$HIP/polyfem/{case.name()}/input/sop_geo1.msh")
                    self.assertEqual(Path(case.evalParm("file_location1")), expected)
                    count = source.geometry().countPrimType(hou.primType.Tetrahedron)
                    self.assertEqual(self.check_tet_file(expected, source, np.ones(count)), count)
                    imported = case.node("geo_1")
                    self.assertEqual(Path(imported.evalParm("File")), expected)
                    self.assertEqual(imported.geometry().countPrimType(hou.primType.Tetrahedron),
                                     count)
                original = expected.read_bytes()
                case.parm("load_sop1").pressButton()
                self.assertEqual(expected.read_bytes(), original)
                self.run_solver(case, expected)
                print(f"{operator}: {count} tetrahedra, direct SOP load and PolyFEM solve")

    def test_invalid_sources_keep_staged_file(self):
        source = self.new_tet_sop("tetconform")
        case = self.new_case()
        self.load_sop(case, source.path())
        dest = Path(case.evalParm("file_location1"))
        original = dest.read_bytes()
        old_reader = case.node("geo_1")

        empty = self.source.createNode("add")
        packed = self.source.createNode("pack")
        packed.setInput(0, self.box)
        mixed = self.source.createNode("merge")
        mixed.setInput(0, source)
        mixed.setInput(1, self.box)
        flat = self.source.createNode("attribwrangle")
        flat.setInput(0, source)
        flat.parm("snippet").set("@P.y = 0;")
        duplicate = self.source.createNode("attribwrangle")
        duplicate.setInput(0, source)
        duplicate.setParms({"class": 0, "snippet": 'addprim(0, "tet", primpoints(0, 0));'})
        over_limit = self.source.createNode("attribwrangle")
        over_limit.setInput(0, self.new_tet_sop("tetembed"))
        over_limit.setParms({"class": 1, "snippet": "i@Entity = @primnum + 1;"})

        for path, message in (("/obj/missing_sop", "SOP node not found"),
                              (empty.path(), "no points"),
                              (self.box.path(), "surface polygons"),
                              (packed.path(), "Unsupported primitive"),
                              (mixed.path(), "Mixed geometry"),
                              (flat.path(), "degenerate tetrahedron"),
                              (duplicate.path(), "Duplicate tetrahedron"),
                              (over_limit.path(), "supports up to 999 subdomains")):
            with self.subTest(path=path):
                log = io.StringIO()
                with redirect_stdout(log):
                    self.load_sop(case, path)
                self.assertIn(message, log.getvalue())
                self.assertEqual(case.evalParm("file_location1"), "")
                self.assertEqual(case.node("geo_1"), old_reader)
                self.assertEqual(dest.read_bytes(), original)
                self.assertEqual(list(dest.parent.iterdir()), [dest])
        self.load_sop(case, source.path())
        self.assertEqual(Path(case.evalParm("file_location1")), dest)
        self.assertEqual(dest.read_bytes(), original)

    def test_polygon_obstacle_with_default_case_folder(self):
        case = self.new_case()
        case.parm("is_obstacle1").set(1)
        self.load_sop(case, case.relativePathTo(self.box))
        target = Path(case.evalParm("file_location1"))
        self.assertEqual(target, self.root / "polyfem" / case.name() / "input/sop_obs1.obj")
        mesh = case.node("geo_1")
        self.assertEqual(Path(mesh.evalParm("file")), target)
        self.assertEqual(mesh.geometry().countPrimType(hou.primType.Polygon), 12)

    def test_standalone_write_msh_and_file_import(self):
        for operator in ("tetconform", "tetembed"):
            with self.subTest(operator=operator):
                sop = self.new_tet_sop(operator)
                sop.parm("outputsurftri").set(1)
                tagged = self.source.createNode("attribwrangle")
                tagged.setInput(0, sop)
                tagged.setParms({"class": 1, "snippet": "i@class = (@primnum % 2) * 9; i@Entity = 7;"})
                writer = self.source.createNode("writeMSH::2")
                writer.setInput(0, tagged)
                path = self.root / f"{operator} standalone.msh"
                writer.parm("file_to_save").set(str(path))
                self.assertEqual(writer.evalParm("flip"), 1)
                self.assertEqual(writer.geometry().intrinsicValue("primitivecount"),
                                 tagged.geometry().intrinsicValue("primitivecount"))
                self.assertFalse(path.exists())
                writer.parm("export").pressButton()
                classes = [p.intAttribValue("class") for p in tagged.geometry().prims()
                           if p.type() == hou.primType.Tetrahedron]
                count = self.check_tet_file(path, tagged, classes)
                original = path.read_bytes()
                writer.parm("export").pressButton()
                self.assertEqual(path.read_bytes(), original)

                writer.parm("flip").set(0)
                writer.parm("export").pressButton()
                self.check_tet_file(path, tagged, classes, orient=False)
                writer.parm("flip").set(1)
                writer.parm("export").pressButton()
                writer.setInput(0, self.box)
                with self.assertRaisesRegex(hou.NodeError, "surface polygons"):
                    writer.hdaModule().save_geo(writer)
                self.assertEqual(path.read_bytes(), original)

                # Normal mesh-file callback, using the other spelling of $OS.
                case = self.new_case()
                case.parm("working_dir").set("$HIP/standalone/${OS}/")
                case.parm("file_location1").set(str(path))
                case.parm("file_location1").pressButton()
                expected = self.root / "standalone" / case.name() / "input" / path.name
                self.assertEqual(Path(case.evalParm("file_location1")), expected)
                self.assertEqual(case.evalParm("num_volumes1"), 2)
                self.assertEqual(expected.read_bytes(), original)
                self.run_solver(case, expected)
                print(f"{operator}: {count} tetrahedra, writeMSH and PolyFEM file solve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
