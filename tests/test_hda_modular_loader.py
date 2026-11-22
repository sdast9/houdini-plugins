"""Verify runtime sections with hython after building the modular MSH Reader."""

import hashlib
import importlib.util
import linecache
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
READER = "sop_MSH_Reader.3.0.hdanc"
TYPE_NAME = "MSH_Reader::3.0"

TRIANGLE = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
1
1 2 2 1 1 1 2 3
$EndElements
"""


def _exercise_installed_reader(path):
    """Child process entry point. Never import any code from src here."""
    import hou

    check = unittest.TestCase()
    check.assertFalse(any(Path(entry).resolve().is_relative_to(SRC) for entry in sys.path))
    hou.hda.installFile(str(path), force_use_assets=True)
    parent = hou.node("/obj").createNode("geo", "modular_reader_test", run_init_scripts=False)
    try:
        node = parent.createNode(TYPE_NAME, exact_type_name=True)
        definition = node.type().definition()
        check.assertEqual(os.path.realpath(definition.libraryFilePath()), os.path.realpath(path))
        sections = definition.sections()
        expected_sections = {"py_00_msh_parser.py", "py_01_cook.py"}
        check.assertTrue(expected_sections <= sections.keys(),
                         "Run this test against a built modular MSH Reader")
        flags = definition.extraFileOptions()
        for flag in ("IsPython", "IsScript"):
            check.assertTrue(flags.get(f"PythonModule/{flag}"))
            for section in expected_sections:
                check.assertFalse(flags.get(f"{section}/{flag}"))

        mod = node.hdaModule()
        names = set(dir(mod))
        namespace = mod.cook.__globals__
        expected_names = {"cook", "reload", "read_msh", "write_moved_nodes"}
        # Some HOM versions present a proxy whose dir() lists wrapper fields.
        check.assertTrue(expected_names <= (names if "cook" in names else namespace.keys()))
        check.assertFalse({"_bootstrap_hda_parts", "linecache"} & (names | namespace.keys()))
        for name in expected_names:
            check.assertTrue(callable(getattr(mod, name)))

        missing = path.parent / "missing.msh"
        try:
            mod.read_msh(str(missing))
        except FileNotFoundError as error:
            frame = next(frame for frame in traceback.extract_tb(error.__traceback__)
                         if frame.name == "read_msh")
            text = sections["py_00_msh_parser.py"].contents()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            check.assertEqual(frame.filename, f"<msh_reader:common/msh_parser.py@{digest}>")
            source_line = linecache.getline(frame.filename, frame.lineno)
            check.assertEqual(source_line, text.splitlines(keepends=True)[frame.lineno - 1])
            check.assertIn('with open(path, "rb")', source_line)
        else:
            check.fail("The parser did not raise for a missing file")

        mesh = path.parent / "triangle.msh"
        mesh.write_text(TRIANGLE, encoding="utf-8")
        node.parm("File").set(str(mesh))
        # The Python SOP invokes cook from the second part. Its read_msh
        # lookup must see replacements on the common HDA module.
        parser = Mock(wraps=mod.read_msh)
        with patch.dict(namespace, read_msh=parser):
            node.node("read_msh").cook(force=True)
            node.cook(force=True)
            geometry = node.geometry()
            parser.assert_any_call(str(mesh))
        check.assertEqual(len(geometry.points()), 3)
        check.assertEqual(len(geometry.prims()), 1)
        check.assertFalse(any(Path(entry).resolve().is_relative_to(SRC) for entry in sys.path))
    finally:
        parent.destroy()


class ModularLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SRC))
        cls.addClassCleanup(sys.path.remove, str(SRC))

    def test_current_hdas_pass_static_checks(self):
        from common import check_parts

        errors = check_parts.check_all()
        self.assertEqual(errors, [], "\n".join(map(str, errors)))

    def test_checker_reports_all_four_rules(self):
        from common.check_parts import SourcePart, check_sources

        parts = [
            SourcePart("a.py", "import math\ndef f(value=later):\n    return missing\n"),
            SourcePart("b.py", "import math\nlater = 1\ndef f():\n    pass\n"),
        ]
        callbacks = [SourcePart("DialogScript", 'script_callback "hou.phm().absent(kwargs)"')]
        errors = check_sources("synthetic", parts, callbacks)
        self.assertCountEqual(
            [(error.rule, error.path, error.line, error.name) for error in errors],
            [
                ("duplicate-definition", "b.py", 3, "f"),
                ("load-order", "a.py", 2, "later"),
                ("undefined-name", "a.py", 3, "missing"),
                ("missing-callback", "DialogScript", 1, "absent"),
            ],
        )
        duplicates = check_sources("synthetic", [
            SourcePart("a.py", "(one, *rest) = (1, 2)\none: int = 3\n"),
        ])
        self.assertEqual([(error.rule, error.name) for error in duplicates],
                         [("duplicate-definition", "one")])

    def test_checker_tracks_class_and_function_scopes(self):
        from common.check_parts import SourcePart, check_sources

        source = """\
import contextlib
total = 0
class Example:
    shared = 1
    comprehension = [value for value in range(shared)]
    def method(self, argument=shared):
        global total
        counter = argument
        def advance(step):
            nonlocal counter
            with contextlib.nullcontext(step) as value:
                try:
                    for item in (value,):
                        if (next_count := item + counter):
                            counter = next_count
                except ValueError as error:
                    return str(error)
            return (lambda offset=counter: offset + total)()
        return advance(1) + later()
def later():
    return 2
"""
        self.assertEqual(check_sources("synthetic", [SourcePart("a.py", source)]), [])
        broken = """\
class Example(Base, metaclass=Meta):
    before = later
    later = 1
    def method(self):
        return later
@decorate
def f(arg: Kind) -> Kind:
    return [value for value in data if missing]
Base, Meta, Kind, decorate, data = object, type, int, str, ()
"""
        errors = check_sources("synthetic", [SourcePart("a.py", broken)])
        self.assertCountEqual(
            [(error.rule, error.name) for error in errors],
            [("load-order", name) for name in ("Base", "Meta", "decorate", "Kind", "Kind")]
            + [("undefined-name", name) for name in ("later", "later", "missing")],
        )

    def test_generated_loader_uses_the_shared_source_reader(self):
        from common.section_build import build_runtime_sections

        reader = Mock(side_effect=[
            '"""Standalone part docstring."""\ndef helper():\n    return 1\n',
            "def caller():\n    return helper()\n",
        ])
        parts = [("helper", ("common", "helper.py")), ("caller", ("asset", "caller.py"))]
        with patch.dict(sys.modules, hda_build=SimpleNamespace(read_source=reader)):
            sections, metadata = build_runtime_sections("asset", "example::1.0", parts)
            with self.assertRaisesRegex(ValueError, "Duplicate runtime part key"):
                build_runtime_sections("asset", "example::1.0", [parts[0], parts[0]])
            with self.assertRaisesRegex(ValueError, "Invalid runtime part key"):
                build_runtime_sections("asset", "example::1.0", [("a/b", ("a.py",))])
        self.assertEqual(reader.call_args_list, [call(*part[1]) for part in parts])

        entries = {
            name: SimpleNamespace(contents=lambda value=text: value)
            for name, text in sections.items()
        }
        definition = SimpleNamespace(sections=lambda: entries)
        node_type = SimpleNamespace(definition=lambda: definition)
        namespace = {"__name__": "example", "kwargs": {"type": node_type}}
        exec(sections["PythonModule"], namespace)
        self.assertEqual(set(namespace),
                         {"__builtins__", "__name__", "kwargs", "helper", "caller"})
        self.assertEqual(namespace["caller"](), 1)
        namespace["helper"] = lambda: 2
        self.assertEqual(namespace["caller"](), 2)
        part = metadata[0]
        filename = namespace["caller"].__code__.co_filename
        self.assertEqual(filename, f"<asset:asset/caller.py@{metadata[1].digest}>")
        self.assertEqual(linecache.getline(filename, 2), "    return helper()\n")
        self.assertRegex(part.digest, r"^[0-9a-f]{12}$")

        lookup = Mock(return_value=node_type)
        hou = SimpleNamespace(nodeType=lookup, nodeTypeCategories=lambda: {"Sop": "sop"})
        with patch.dict(sys.modules, hou=hou):
            exec(sections["PythonModule"], {"__name__": "example"})
        lookup.assert_called_once_with("sop", "example::1.0")

    @unittest.skipUnless(
        "hython" in Path(sys.executable).name.lower() and importlib.util.find_spec("hou"),
        "requires hython and a rebuilt MSH Reader",
    )
    def test_isolated_embedded_reader(self):
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from common import hda_build, section_build
        from msh_reader import runtime_manifest as msh_manifest
        import hou

        with tempfile.TemporaryDirectory(prefix="modular_hda_") as temporary:
            work = Path(temporary)
            otls = work / "otls"
            otls.mkdir()
            path = otls / READER
            shutil.copyfile(BASE / READER, path)

            # Verify embedded py_* sections match the current manifest sources.
            hou.hda.installFile(str(path), force_use_assets=True)
            node_type = hou.nodeType(hou.sopNodeTypeCategory(), msh_manifest.TYPE_NAME)
            definition = node_type.definition()
            self.assertEqual(os.path.realpath(definition.libraryFilePath()), os.path.realpath(path))
            sections = definition.sections()
            for _, relpath, sec_name in section_build.manifest_entries(msh_manifest.PARTS):
                self.assertIn(sec_name, sections)
                self.assertEqual(sections[sec_name].contents(), hda_build.read_source(*relpath))

            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update({
                "HOUDINI_USER_PREF_DIR": str(work / "prefs" / "__HVER__"),
                "HOUDINI_NO_ENV_FILE": "1",
                "HOUDINI_PACKAGE_SKIP": "1",
                "HOUDINI_PATH": os.pathsep.join(("$HOUDINI_USER_PREF_DIR", "$HFS/houdini")),
                "HOUDINI_OTLSCAN_PATH": os.pathsep.join((str(otls), "$HFS/houdini/otls")),
                "HOUDINI_OPLIBRARIES_PATH": str(otls),
            })
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--embedded-reader", str(path)],
                cwd=work, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--embedded-reader":
        _exercise_installed_reader(Path(sys.argv[2]))
    else:
        unittest.main()
