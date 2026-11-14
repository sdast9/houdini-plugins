"""Material files staged into input/ keep their own content (2026-11-14).

Before this fix a fiber or dispersion file was copied to input/<its name>:
* two files of the same name from different folders became one input file,
  so one body was exported with the other body's fiber directions;
* a file named fibers.vtk (or pe_*.txt) took the name of a file Write
  creates, so the first Write replaced it and every later Write and Check
  Setup failed ("VECTORS 'FIB_DIR1' not found").

Run: hython tests/test_material_file_staging.py
No gmsh and no PolyFEM binary needed.
"""

import os
import tempfile

import numpy as np

import hou

from test_check_setup import write_box

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_fibers(path, count, direction, field="FIB_DIR1"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("# vtk DataFile Version 3.0\nuser fibres\nASCII\n"
                     f"DATASET UNSTRUCTURED_GRID\nCELL_DATA {count}\n"
                     f"VECTORS {field} float\n")
        for _ in range(count):
            handle.write("%r %r %r\n" % tuple(float(v) for v in direction))
    return path


def exported_rows(input_dir, field):
    """The rows of one VECTORS array of the exported input/fibers.vtk."""
    lines = open(os.path.join(input_dir, "fibers.vtk")).read().splitlines()
    count = int(next(line for line in lines
                     if line.startswith("CELL_DATA")).split()[1])
    start = lines.index(f"VECTORS {field} float") + 1
    return np.loadtxt(lines[start:start + count], ndmin=2)


def scene(root, label, bodies=1):
    work = os.path.join(root, label)
    os.makedirs(work)
    mesh = write_box(os.path.join(root, f"{label}_box.msh"))
    node = hou.node("/obj").createNode(TYPE, label)
    node.setParms({"working_dir": work + "/", "geo_int": bodies,
                   "num_geos": bodies})
    mod = node.hdaModule()
    for geo in range(1, bodies + 1):
        node.setParms({f"file_location{geo}": mesh})
        node.parm(f"file_location{geo}").pressButton()
        node.setParms({f"xform_t__{geo}x": 2.0 * (geo - 1),
                       f"materials{geo}_1":
                           mod.MATERIAL_TOKENS.index("HGOFiber")})
    node.setParms({"enable": 0, "quasistatic": 1})
    return node, mod, work


def pick(node, mod, parm_name, path):
    """What the UI does when a file is chosen in a material file field."""
    node.parm(parm_name).set(path)
    mod.material_file_changed({"node": node, "parm": node.parm(parm_name),
                               "script_multiparm_index": "1",
                               "script_multiparm_index2": "1"})
    return node.evalParm(parm_name)


def check_user_file_named_fibers_vtk(root):
    node, mod, work = scene(root, "named_fibers_vtk")
    input_dir = os.path.join(work, "input")
    count = mod.volume_element_count(node, 1)
    user = write_fibers(os.path.join(root, "userdata", "fibers.vtk"), count,
                        (0, 0, 1))
    node.setParms({"fib_source1_1": "file"})
    staged = pick(node, mod, "fib_file1_1", user)
    assert os.path.dirname(staged) == input_dir, staged
    assert os.path.basename(staged) != "fibers.vtk", staged
    for attempt in (1, 2, 3):
        mod.build_params(node)
        rows = exported_rows(input_dir, "FIB_1_1")
        assert np.allclose(rows, [0, 0, 1]), (attempt, rows[:2])
        assert not mod.check_setup(node, "write")["errors"], attempt
    print("PASS: a user fiber file named fibers.vtk is staged under another "
          "name and survives repeated Writes")
    return node, mod, work


def check_same_name_two_bodies(root):
    node, mod, work = scene(root, "same_name", bodies=2)
    input_dir = os.path.join(work, "input")
    count = mod.volume_element_count(node, 1)
    first = write_fibers(os.path.join(root, "bodyA", "dirs.vtk"), count,
                         (1, 0, 0))
    second = write_fibers(os.path.join(root, "bodyB", "dirs.vtk"), count,
                          (0, 1, 0))
    node.setParms({"fib_source1_1": "file", "fib_source2_1": "file"})
    staged1 = pick(node, mod, "fib_file1_1", first)
    staged2 = pick(node, mod, "fib_file2_1", second)
    assert staged1 != staged2, (staged1, staged2)
    mod.build_params(node)
    rows1 = exported_rows(input_dir, "FIB_1_1")[:count]
    rows2 = exported_rows(input_dir, "FIB_2_1")[count:]
    assert np.allclose(rows1, [1, 0, 0]), rows1[:2]
    assert np.allclose(rows2, [0, 1, 0]), rows2[:2]

    # Picking geometry 1's file again (unchanged) keeps its staged copy, and
    # the same file chosen for both bodies is staged once.
    assert pick(node, mod, "fib_file1_1", first) == staged1
    shared = pick(node, mod, "fib_file2_1", first)
    assert shared == staged1, (shared, staged1)
    print("PASS: same-named fiber files from two folders stay two files; "
          "each body is exported with its own directions")


def check_dispersion_file_name(root):
    node, mod, work = scene(root, "pe_name")
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("HGODispersion"),
                   "fib_source1_1": "constant", "kappa_source1_1": "file"})
    count = mod.volume_element_count(node, 1)
    user = os.path.join(root, "userdata2", "pe_kappa_1_1.txt")
    os.makedirs(os.path.dirname(user))
    np.savetxt(user, np.full(count, 0.2))
    staged = pick(node, mod, "kappa_file1_1", user)
    assert os.path.basename(staged) != "pe_kappa_1_1.txt", staged
    for attempt in (1, 2):
        mod.build_params(node)
        assert not mod.check_setup(node, "write")["errors"], attempt
    print("PASS: a dispersion file named like Write's own pe_*.txt is "
          "staged under another name")


def check_old_scene(root):
    """A scene saved before the fix: the parameter points at input/fibers.vtk,
    which still holds the user's file. Write moves it aside first."""
    node, mod, work = scene(root, "old_scene")
    input_dir = os.path.join(work, "input")
    count = mod.volume_element_count(node, 1)
    old = write_fibers(os.path.join(input_dir, "fibers.vtk"), count,
                       (0, 1, 0))
    node.setParms({"fib_source1_1": "file", "fib_file1_1": old})
    for attempt in (1, 2):
        mod.build_params(node)
        assert os.path.basename(node.evalParm("fib_file1_1")) != \
            "fibers.vtk", node.evalParm("fib_file1_1")
        assert np.allclose(exported_rows(input_dir, "FIB_1_1"), [0, 1, 0])
    print("PASS: a scene pointing at input/fibers.vtk has its file moved "
          "aside before Write overwrites that name")

    # Already overwritten (an earlier Write under the old code): the check
    # names what happened instead of a bare missing-array message.
    clobbered = write_fibers(os.path.join(input_dir, "fibers.vtk"), count,
                             (0, 1, 0), field="FIB_1_1")
    node.setParms({"fib_file1_1": clobbered})
    errors = mod.check_setup(node, "write")["errors"]
    assert any("Pick your original fiber file" in e for e in errors), errors
    print("PASS: a fiber file already overwritten by an earlier Write is "
          "reported by name")


def check_import(root, source_work):
    """Importing a run reads its fibers.vtk; the node must not keep pointing
    at the name Write rewrites."""
    node = hou.node("/obj").createNode(TYPE, "reimport")
    node.setParms({"old_input_dir": source_work})
    mod = node.hdaModule()
    mod.read_params({"node": node})
    staged = node.evalParm("fib_file1_1")
    assert node.parm("fib_source1_1").evalAsString() == "file"
    assert os.path.basename(staged) != "fibers.vtk", staged
    assert node.evalParm("fib_file_field1_1") == "FIB_1_1"
    for attempt in (1, 2):
        mod.build_params(node)
        assert np.allclose(exported_rows(
            os.path.join(source_work, "input"), "FIB_1_1"), [0, 0, 1])
    print("PASS: an imported run's fibers.vtk is staged under another name "
          "and Write reproduces its directions")


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    root = tempfile.mkdtemp(prefix="polyfem_material_staging_")
    _, _, named_work = check_user_file_named_fibers_vtk(root)
    check_same_name_two_bodies(root)
    check_dispersion_file_name(root)
    check_old_scene(root)
    check_import(root, named_work)
    print("workdir:", root)


if __name__ == "__main__":
    main()
