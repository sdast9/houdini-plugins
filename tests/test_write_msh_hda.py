"""Check the project-built writeMSH::2 asset. Run under hython."""

from pathlib import Path
import sys
import tempfile
import unittest

import hou
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
LIBRARY = BASE / "sop_writeMSH.2.hdanc"
sys.path.insert(0, str(SRC))
from common.msh_parser import read_msh
from common.section_build import manifest_entries
from write_msh.runtime_manifest import PARTS, TYPE_NAME


class WriteMshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hou.hda.installFile(str(LIBRARY), oplibraries_file=None, force_use_assets=True)
        cls.definition, = hou.hda.definitionsInFile(str(LIBRARY))
        assert cls.definition.isCurrent()
        assert cls.definition.nodeTypeName() == TYPE_NAME

    def setUp(self):
        target = tempfile.TemporaryDirectory(prefix="writeMSH test ")
        self.addCleanup(target.cleanup)
        self.path = Path(target.name) / "solid.msh"
        self.geo = hou.node("/obj").createNode("geo", "source", run_init_scripts=False)
        self.addCleanup(self.geo.destroy)
        self.box = self.geo.createNode("box")
        self.writer = self.geo.createNode(TYPE_NAME, exact_type_name=True)
        self.assertEqual(Path(self.writer.type().definition().libraryFilePath()), LIBRARY)
        self.assertEqual(self.writer.evalParm("flip"), 1)
        self.writer.parm("file_to_save").set(str(self.path))

    def make_tets(self, operator, snippet):
        solid = self.geo.createNode(operator)
        solid.setInput(0, self.box)
        solid.parm("outputsurftri").set(1)
        tagged = self.geo.createNode("attribwrangle")
        tagged.setInput(0, solid)
        tagged.setParms({"class": 1, "snippet": snippet})
        upstream = tagged.geometry()
        self.assertGreater(upstream.countPrimType(hou.primType.Polygon), 0)
        return tagged

    def test_embedded_sections_match_source(self):
        sections = self.definition.sections()
        entries = manifest_entries(PARTS)
        self.assertEqual({key for key in sections if key.startswith("py_")},
                         {name for _, _, name in entries})
        for _, source, name in entries:
            self.assertEqual(sections[name].contents(),
                             SRC.joinpath(*source).read_text(encoding="utf-8"))
        for flag in ("IsPython", "IsScript"):
            self.assertTrue(self.definition.extraFileOptions()[f"PythonModule/{flag}"])

    def test_real_tet_conform_and_embed(self):
        for operator in ("tetconform", "tetembed"):
            with self.subTest(operator=operator):
                self.path = self.path.with_name(f"{operator} standalone.msh")
                self.writer.setParms({"file_to_save": str(self.path), "flip": 1})
                source = self.make_tets(operator,
                                        "i@class = 9 * (@primnum % 2); i@Entity = 7;")
                self.writer.setInput(0, source)
                geo = self.writer.geometry()
                self.assertEqual(geo.intrinsicValue("primitivecount"),
                                 source.geometry().intrinsicValue("primitivecount"))
                self.assertFalse(self.path.exists())
                self.writer.parm("export").pressButton()
                original = self.path.read_bytes()
                self.assertTrue(original.startswith(b"$MeshFormat\n2.2 0 8\n"))
                mesh = read_msh(self.path)
                self.assertEqual(set(mesh["cells"]), {"tet"})
                tet = mesh["cells"]["tet"]
                tets = [prim for prim in geo.prims() if prim.type() == hou.primType.Tetrahedron]
                self.assertGreater(len(tets), 0)
                indices = [[vertex.point().number() for vertex in prim.vertices()] for prim in tets]
                tags = [prim.intAttribValue("class") for prim in tets]
                np.testing.assert_array_equal(tet["entity"], tags)
                np.testing.assert_array_equal(np.sort(tet["corners"], axis=1),
                                              np.sort(indices, axis=1))
                points = mesh["points"][tet["corners"]]
                self.assertTrue((np.linalg.det(points[:, 1:] - points[:, :1]) > 0).all())
                self.writer.parm("export").pressButton()
                self.assertEqual(self.path.read_bytes(), original)

                self.writer.parm("flip").set(0)
                self.writer.parm("export").pressButton()
                raw = read_msh(self.path)["cells"]["tet"]["corners"]
                np.testing.assert_array_equal(raw, indices)
                print(f"{operator}: {len(tets)} tetrahedra, class tags preserved")

    def test_tag_fallback_and_surface_rejection_preserve_file(self):
        source = self.make_tets("tetconform", "i@Entity = 17;")
        self.writer.setInput(0, source)
        self.writer.parm("export").pressButton()
        original = self.path.read_bytes()
        self.assertEqual(set(read_msh(self.path)["cells"]["tet"]["entity"]), {17})
        self.writer.setInput(0, self.box)
        with self.assertRaisesRegex(hou.NodeError, "surface polygons"):
            self.writer.hdaModule().save_geo(self.writer)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
