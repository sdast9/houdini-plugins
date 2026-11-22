"""Check runtime sections, with isolated Houdini loads after rebuilding."""

import hashlib
import importlib.util
import linecache
import os
from pathlib import Path
import shutil
import signal
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
READ_PVD = "object_readPVD.1.0.hdanc"
POLYFEM = "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"
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


def _exercise_installed_readpvd(path):
    """Load Read PVD from the isolated library without importing src."""
    import hou

    check = unittest.TestCase()
    check.assertFalse(any(Path(entry).resolve().is_relative_to(SRC) for entry in sys.path))
    hou.hda.installFile(str(path), force_use_assets=True)
    node = hou.node("/obj").createNode("readPVD::1.0", "modular_readpvd_test",
                                      run_init_scripts=False, exact_type_name=True)
    try:
        definition = node.type().definition()
        check.assertEqual(os.path.realpath(definition.libraryFilePath()), os.path.realpath(path))
        mod = node.hdaModule()
        namespace = mod.refresh.__globals__
        check.assertNotIn("_bootstrap_hda_parts", namespace)
        check.assertIs(mod._pvd_times.__globals__, mod.read_pvd.__globals__)
        check.assertIs(namespace["ET"], mod.ET)
        check.assertTrue(callable(mod.read_msh))
        check.assertTrue(callable(mod.XlsxWorkbook))

        malformed = path.parent.parent / "malformed.pvd"
        malformed.write_text("<VTKFile>", encoding="utf-8")
        try:
            mod.read_pvd(str(malformed))
        except mod.ET.ParseError as error:
            frame = next(frame for frame in traceback.extract_tb(error.__traceback__)
                         if frame.name == "read_pvd")
            text = definition.sections()["py_01_vtu_parser.py"].contents()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            check.assertEqual(frame.filename, f"<readpvd:common/vtu_parser.py@{digest}>")
            check.assertEqual(linecache.getline(frame.filename, frame.lineno),
                              text.splitlines(keepends=True)[frame.lineno - 1])
            check.assertEqual(frame.line, "tree = ET.parse(path)")
        else:
            check.fail("The parser accepted malformed PVD XML")

        collection = path.parent.parent / "collection.pvd"
        collection.write_text(
            '<VTKFile type="Collection"><Collection>'
            '<DataSet timestep="1.25" file="mesh.vtu"/>'
            '</Collection></VTKFile>', encoding="utf-8",
        )
        timeline = SimpleNamespace(evalParm=Mock(return_value=str(collection)))
        with patch.object(mod.ET, "parse", wraps=mod.ET.parse) as parse:
            check.assertEqual(mod._pvd_times(timeline), [1.25])
        timeline.evalParm.assert_called_once_with("PVD_file")
        parse.assert_called_once_with(str(collection))
    finally:
        node.destroy()


def _exercise_installed_polyfem(path):
    """Load PolyFEM from the isolated library without importing src."""
    import hou

    check = unittest.TestCase()
    check.assertFalse(any(Path(entry).resolve().is_relative_to(SRC) for entry in sys.path))
    hou.hda.installFile(str(path), force_use_assets=True)
    node = hou.node("/obj").createNode("stevenabramowitch::dev::PolyFEM::2.0",
                                      "modular_polyfem_test", run_init_scripts=False,
                                      exact_type_name=True)
    try:
        definition = node.type().definition()
        check.assertEqual(Path(definition.libraryFilePath()).resolve(), path.resolve())
        mod = node.hdaModule()
        namespace = mod.stop_run.__globals__
        check.assertNotIn("_bootstrap_hda_parts", namespace)
        check.assertIs(mod._message.__globals__, namespace)
        check.assertIs(mod.read_cell_vectors_legacy_vtk.__globals__, namespace)
        check.assertIs(namespace["os"], mod.os)

        work = path.parent.parent
        run = {"pid": 12345, "output": str(work / "output")}
        active_run = Mock(return_value=run)
        matches = Mock(return_value=True)
        message = Mock()
        with (patch.dict(namespace, active_run=active_run,
                         _run_pid_matches=matches, _message=message),
              patch.object(mod.platform, "system", return_value="Linux"),
              patch.object(mod.os, "kill", side_effect=OSError("test signal failure")) as kill):
            check.assertFalse(mod.stop_run({"node": node}))
        active_run.assert_called_once_with(node)
        matches.assert_called_once_with(run, run["pid"], for_signal=True)
        kill.assert_called_once_with(run["pid"], signal.SIGTERM)
        message.assert_called_once_with("Could not stop the run: test signal failure")

        try:
            mod.read_cell_vectors_legacy_vtk(str(work / "missing.vtk"), "fibers")
        except FileNotFoundError as error:
            frame = next(frame for frame in traceback.extract_tb(error.__traceback__)
                         if frame.name == "read_cell_vectors_legacy_vtk")
            source = definition.sections()["py_10_45_materials.py"].contents()
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
            check.assertEqual(frame.filename,
                              f"<polyfem:polyfem/runtime/45_materials.py@{digest}>")
            check.assertEqual(linecache.getline(frame.filename, frame.lineno),
                              source.splitlines(keepends=True)[frame.lineno - 1])
            check.assertEqual(frame.line, "with open(path) as handle:")
        else:
            check.fail("The VTK reader did not raise for a missing file")
        check.assertFalse(any(Path(entry).resolve().is_relative_to(SRC) for entry in sys.path))
    finally:
        node.destroy()


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

    def test_module_compilation_preserves_bytecode_and_local_tracebacks(self):
        from common.hda_fingerprint import _callable_code_hashes
        from common.section_build import build_runtime_sections

        header = "import math\n"
        logic = '''\
"""Standalone part docstring."""
def calculate():
    class Calculation:
        def square_root(self):
            return math.sqrt(-1)
    return Calculation().square_root()
'''
        parts = [("header", ("header.py",)), ("logic", ("logic.py",))]
        reader = Mock(side_effect=[header, logic])
        with patch.dict(sys.modules, hda_build=SimpleNamespace(read_source=reader)):
            sections, metadata = build_runtime_sections(
                "example", "example::1.0", parts, compile_line_gaps={"logic": 3})
        entries = {name: SimpleNamespace(contents=lambda text=text: text)
                   for name, text in sections.items()}
        definition = SimpleNamespace(sections=lambda: entries)
        node_type = SimpleNamespace(definition=lambda: definition)
        baseline = {"__name__": "example", "kwargs": {"type": node_type}}
        modular = baseline.copy()
        exec(header + "\n" * 3 + logic, baseline)
        exec(sections["PythonModule"], modular)
        self.assertEqual(set(modular), set(baseline))
        self.assertEqual(_callable_code_hashes(modular), _callable_code_hashes(baseline))

        try:
            modular["calculate"]()
        except ValueError as error:
            frame = traceback.extract_tb(error.__traceback__)[-1]
            self.assertEqual(frame.filename, f"<example:logic.py@{metadata[1].digest}>")
            self.assertEqual(frame.lineno, 5)
            self.assertEqual(frame.line, "return math.sqrt(-1)")
        else:
            self.fail("Expected math.sqrt to raise for a negative input")

    @unittest.skipUnless(
        "hython" in Path(sys.executable).name.lower() and importlib.util.find_spec("hou"),
        "requires hython and a rebuilt MSH Reader",
    )
    def test_isolated_embedded_reader(self):
        from msh_reader import runtime_manifest as msh_manifest

        self._check_isolated_asset(READER, msh_manifest, "--embedded-reader")

    @unittest.skipUnless(
        "hython" in Path(sys.executable).name.lower() and importlib.util.find_spec("hou"),
        "requires hython and a rebuilt Read PVD",
    )
    def test_isolated_embedded_readpvd(self):
        from readpvd import runtime_manifest

        self._check_isolated_asset(READ_PVD, runtime_manifest, "--embedded-readpvd")

    @unittest.skipUnless(
        "hython" in Path(sys.executable).name.lower() and importlib.util.find_spec("hou"),
        "requires hython and a rebuilt PolyFEM",
    )
    def test_isolated_embedded_polyfem(self):
        from polyfem import runtime_manifest

        viewer_sections = ("ViewerStateModule", "ViewerStateInstall", "ViewerStateUninstall")
        script_sections = ("PythonModule", "OnCreated", "OnLoaded", "OnUpdated", *viewer_sections)
        flags = {f"{name}/{flag}": True for name in script_sections
                 for flag in ("IsPython", "IsScript")}
        flags.update({f"{name}/IsViewerState": True for name in viewer_sections})
        self._check_isolated_asset(POLYFEM, runtime_manifest, "--embedded-polyfem",
                                   expected_flags=flags)

    def _check_isolated_asset(self, filename, manifest, child_mode, *, expected_flags=None):
        from common import hda_build, section_build
        import hou

        with tempfile.TemporaryDirectory(prefix="modular_hda_") as temporary:
            work = Path(temporary)
            otls = work / "otls"
            otls.mkdir()
            path = otls / filename
            shutil.copyfile(BASE / filename, path)

            # Verify embedded py_* sections match the current manifest sources.
            definition, = hou.hda.definitionsInFile(str(path))
            sections = definition.sections()
            entries = section_build.manifest_entries(manifest.PARTS)
            self.assertEqual({name for name in sections if name.startswith("py_")},
                             {name for _, _, name in entries})
            flags = definition.extraFileOptions()
            if expected_flags is not None:
                self.assertEqual(flags, expected_flags)
            for flag in ("IsPython", "IsScript"):
                self.assertTrue(flags.get(f"PythonModule/{flag}"))
            for _, relpath, name in entries:
                self.assertEqual(sections[name].contents(), hda_build.read_source(*relpath), name)
                self.assertFalse(flags.get(f"{name}/IsScript"), name)

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
                [sys.executable, str(Path(__file__).resolve()), child_mode, str(path)],
                cwd=work, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--embedded-reader":
        _exercise_installed_reader(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--embedded-readpvd":
        _exercise_installed_readpvd(Path(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--embedded-polyfem":
        _exercise_installed_polyfem(Path(sys.argv[2]))
    else:
        unittest.main()
