"""Animated obstacles from keyframed transforms (2026-10-02).

An obstacle moves when its Translate / Rotate / Scale / Pivot are keyframed,
like any Houdini object; its rest pose is the pose at the Time tab's Start
Frame. A translation is exported as obstacle_displacements tables, any other
motion as a mesh sequence (the obstacle posed at every frame of an integer
frame rate on which every time step falls). Checked with the real solver:

* a keyed translation pushing on a body equals the same motion written as an
  expression of t (same solution to roundoff);
* a keyed rotation + push (mesh sequence) puts the obstacle exactly where the
  same motion written as expressions puts it, at every step, and the body
  responds the same; the sequence's frame rate follows dt;
* a .msh (tet) obstacle is written as its surface, matching PolyFEM's own
  reading of the file;
* refusals by name: a simulated body with an animated transform, keys plus a
  displacement expression on one obstacle, a dt no reasonable frame rate fits;
* Import: keyframes back from the scene record (translation and sequence),
  translation tables without it, a sequence without it as a static obstacle.

Run: hython tests/test_animated_obstacles.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
sys.path.insert(0, os.path.join(BASE, "src", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtu_parser  # noqa: E402
from test_sideset_conditions import write_box  # noqa: E402


def plate(path, half=0.7, z=1.05, centre=(0.5, 0.5)):
    """A square of two triangles above the unit cube."""
    cx, cy = centre
    with open(path, "w") as handle:
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            handle.write(f"v {cx + half * x} {cy + half * y} {z}\n")
        handle.write("f 1 3 2\nf 1 4 3\n")
    return path


def keys(parm, points):
    keyframes = []
    for frame, value in points:
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression("linear()", hou.exprLanguage.Hscript)
        keyframes.append(key)
    parm.setKeyframes(keyframes)


class Scene:
    """Unit cube fixed at the bottom, an obstacle plate above it, contact."""

    count = 0

    def __init__(self, root, label, obstacle=None, half=0.7, dt=0.125,
                 centre=(0.5, 0.5)):
        Scene.count += 1
        self.work = os.path.join(root, f"{Scene.count:02d}_{label}")
        os.makedirs(self.work)
        node = self.node = hou.node("/obj").createNode(TYPE,
                                                       f"o{Scene.count}")
        self.mod = node.hdaModule()
        node.setParms({"working_dir": self.work + "/",
                       "polyfem_bin": POLYFEM_BIN, "use_hdf5": 0,
                       "num_geos": 2, "geo_int": 2})
        node.setParms({"file_location1": write_box(
            os.path.join(self.work, "box.msh"), n=2)})
        node.parm("file_location1").pressButton()
        node.setParms({"is_obstacle2": 1, "file_location2": obstacle or plate(
            os.path.join(self.work, "plate.obj"), half=half, centre=centre)})
        node.parm("file_location2").pressButton()
        node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                       "time_inc_bool": 1, "dt": dt, "enable": 1,
                       "dhat": 0.01, "cof": 0.0, "materials1_1": 0,
                       "E1_1": 1e5, "nu1_1": 0.3})
        node.parm("sideset_selection1_1").set(1)
        node.parm("sideset_selection1_1").pressButton()
        node.setParms({"basegroup1_1_1": "axis:-z:0.01",
                       "Boundary_Condition__1_1_1": 1,
                       "boundary_type1_1_1_1": 0,
                       "vector_1_1_1_1": "[0, 0, 0]"})

    def write(self):
        path = self.mod.write_params_only({"node": self.node})
        assert path, f"export failed in {self.work}"
        with open(path) as handle:
            return path, json.load(handle)

    def run(self):
        path, data = self.write()
        return run_input(os.path.dirname(path)), data


def run_input(input_dir):
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "warning", "--max_threads", "1"],
        cwd=input_dir, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, (
        f"PolyFEM failed in {input_dir}:\n{result.stdout[-2000:]}"
        f"\n{result.stderr[-800:]}")
    pvd = os.path.join(os.path.dirname(input_dir), "output", "sim.pvd")
    out = []
    for index in range(len(vtu_parser.read_pvd(pvd))):
        mesh = vtu_parser.load_frame(pvd, index)["Volume"]
        body = np.asarray(mesh["point_data"]["body_ids"]).ravel()
        out.append((np.asarray(mesh["points"]),
                    np.asarray(mesh["point_data"]["solution"]), body))
    return out


def split(frames):
    """(body solutions, obstacle positions) per step (obstacle = body 0)."""
    bodies = [solution[body != 0] for _, solution, body in frames]
    obstacles = [points[body == 0] + solution[body == 0]
                 for points, solution, body in frames]
    return bodies, obstacles


def max_difference(first, second):
    return max(float(np.abs(a - b).max()) for a, b in zip(first, second))


def expect_error(text, action):
    try:
        action()
    except hou.Error as exc:
        assert text in str(exc), f"expected '{text}' in: {exc}"
        return str(exc)
    raise AssertionError(f"no error containing '{text}'")


def check_translation(root):
    keyed = Scene(root, "keyed_translation")
    keys(keyed.node.parm("xform_t__2z"), [(1, 0.0), (13, -0.08), (25, -0.04)])
    report = keyed.mod.check_setup(keyed.node)
    assert any("translation only" in note for note in report["notes"]), report
    frames, data = keyed.run()
    (entry,) = data["boundary_conditions"]["obstacle_displacements"]
    assert entry["value"] == [1, 1, 1] and len(entry["interpolation"]) == 3
    assert entry["interpolation"][2]["points"] == [0.0, 0.5, 1.0], entry
    assert "mesh" in data["geometry"][1]
    expression = Scene(root, "expression_translation")
    expression.node.parm("obstacle_disp2").set(
        '["0", "0", "-0.16*min(t,0.5) + 0.08*max(t-0.5,0)"]')
    reference, _ = expression.run()
    bodies, obstacles = split(frames)
    reference_bodies, reference_obstacles = split(reference)
    assert max_difference(obstacles, reference_obstacles) < 1e-12
    assert max_difference(bodies, reference_bodies) < 1e-12
    assert max(float(np.abs(b).max()) for b in bodies) > 1e-3
    print("PASS: a keyed translation equals the same motion as an expression "
          "of t (body and obstacle identical to roundoff); exported as "
          "displacement tables")
    return keyed


def check_rotation(root):
    # The default semi-implicit barrier adapts its stiffness with discrete
    # decisions, which turn the 1e-16 difference between two ways of
    # computing the same rotation into 1e-6..1e-4 in the body (nudging the
    # expression scene's pivot by 1e-15 moves the body by 1e-5). A fixed
    # barrier stiffness responds smoothly (the same nudge: 8e-17), so the
    # comparison uses it, with a plate that covers the cube's top at every
    # angle and no plate edge over a node of the cube.
    keyed = Scene(root, "keyed_rotation", half=1.0, centre=(0.63, 0.5))
    node = keyed.node
    node.parm("barrier_mode").set("fixed")
    node.parmTuple("tpivot_2").set((0.5, 0.5, 1.05))
    keys(node.parm("xform_r__2z"), [(1, 0.0), (25, 90.0)])
    keys(node.parm("xform_t__2z"), [(1, 0.0), (25, -0.08)])
    report = keyed.mod.check_setup(node)
    assert any("rotates or scales" in note for note in report["notes"]), report
    frames, data = keyed.run()
    sequence = data["geometry"][1]
    assert sequence["type"] == "mesh_sequence" and sequence["fps"] == 8 \
        and len(sequence["mesh_sequence"]) == 9, sequence
    assert "obstacle_displacements" not in data["boundary_conditions"]
    expression = Scene(root, "expression_rotation", half=1.0,
                       centre=(0.63, 0.5))
    expression.node.parm("barrier_mode").set("fixed")
    expression.node.parm("obstacle_disp2").set(json.dumps([
        "(cos(deg2rad(90*t))-1)*(x-0.5) - sin(deg2rad(90*t))*(y-0.5)",
        "sin(deg2rad(90*t))*(x-0.5) + (cos(deg2rad(90*t))-1)*(y-0.5)",
        "-0.08*t"]))
    reference, _ = expression.run()
    bodies, obstacles = split(frames)
    reference_bodies, reference_obstacles = split(reference)
    obstacle_error = max_difference(obstacles, reference_obstacles)
    body_error = max_difference(bodies, reference_bodies)
    assert obstacle_error < 1e-12, obstacle_error
    assert body_error < 1e-12, body_error
    assert max(float(np.abs(b).max()) for b in bodies) > 1e-3
    print("PASS: a keyed rotation + push (mesh sequence, 8 frames per "
          "simulated second) puts the obstacle where the expression scene "
          f"does at every step ({obstacle_error:.1e}) and the body responds "
          f"the same ({body_error:.1e})")

    # the frame rate follows dt: 0.03 -> 100 frames/s, 3 per step
    node.setParms({"dt": 0.03})
    _, data = keyed.write()
    sequence = data["geometry"][1]
    assert sequence["fps"] == 100 and len(sequence["mesh_sequence"]) == 103, \
        (sequence["fps"], len(sequence["mesh_sequence"]))
    files = sorted(os.listdir(os.path.join(keyed.work, "input",
                                           "obstacle_geo2")))
    assert len(files) == 103, len(files)
    # a dt that would need too many frames per step is refused by name
    node.setParms({"dt": 0.0123})
    report = keyed.mod.check_setup(node)
    assert any("whole number" in error for error in report["errors"]), report
    expect_error("whole number", lambda: keyed.mod.build_params(node))
    print("PASS: the sequence's frame rate follows dt (0.03 s -> 100 frames "
          "per second); a dt needing more than 10 frames per step is refused")
    node.setParms({"dt": 0.125})
    return keyed


def check_msh_obstacle(root):
    work = os.path.join(root, "msh_obstacle")
    os.makedirs(work)
    ball = write_box(os.path.join(work, "block.msh"), n=2,
                     lo=(0.2, 0.2, 1.05), hi=(0.8, 0.8, 1.35))
    static = Scene(root, "msh_static", obstacle=ball)
    frames, _ = static.run()
    _, static_obstacle = split(frames)
    keyed = Scene(root, "msh_rotating", obstacle=ball)
    node = keyed.node
    node.parmTuple("tpivot_2").set((0.5, 0.5, 1.2))
    keys(node.parm("xform_r__2z"), [(1, 0.0), (25, 30.0)])
    _, data = keyed.write()
    first = os.path.join(keyed.work, "input",
                         data["geometry"][1]["mesh_sequence"][0])
    vertices, faces = keyed.mod._read_obj_surface(first)
    # the surface of the 48-tet block: PolyFEM's own reading has the same
    # points (26 surface nodes of 27), in some order
    rest = static_obstacle[0]
    assert len(vertices) == len(rest) == 26 and len(faces) == 48, \
        (len(vertices), len(rest), len(faces))
    distance = [float(np.min(np.linalg.norm(rest - v, axis=1)))
                for v in vertices]
    assert max(distance) < 1e-14, max(distance)
    print("PASS: a rotating .msh (tet) obstacle is written as its surface, "
          "matching PolyFEM's own reading of the file")


def check_refusals(root):
    scene = Scene(root, "refusals")
    node = scene.node
    keys(node.parm("xform_t__1x"), [(1, 0.0), (25, 0.1)])
    report = scene.mod.check_setup(node)
    assert any("simulated body" in error for error in report["errors"]), report
    expect_error("simulated body", lambda: scene.mod.build_params(node))
    node.parm("xform_t__1x").deleteAllKeyframes()
    node.parm("xform_t__1x").set(0)
    keys(node.parm("xform_t__2z"), [(1, 0.0), (25, -0.05)])
    node.parm("obstacle_disp2").set('[0, 0, "-0.01*t"]')
    report = scene.mod.check_setup(node)
    assert any("only one motion" in error for error in report["errors"]), report
    expect_error("only one motion", lambda: scene.mod.build_params(node))
    print("PASS: refused by name: an animated simulated body; keys plus a "
          "displacement expression on one obstacle")


def import_into(root, source_input, label):
    target = hou.node("/obj").createNode(TYPE, label)
    target.setParms({"old_input_dir": source_input})
    target.hdaModule().read_params({"node": target})
    work = os.path.join(root, label)
    os.makedirs(work)
    target.setParms({"working_dir": work + "/", "use_hdf5": 0,
                     "polyfem_bin": POLYFEM_BIN})
    return target


def check_import(root, translated, rotated):
    for scene, label in ((translated, "translation"), (rotated, "sequence")):
        path, data = scene.write()
        target = import_into(root, os.path.dirname(path),
                             f"import_{label}")
        for name in ("xform_t__2z", "xform_r__2z"):
            source = [k.asJSON() for k in scene.node.parm(name).keyframes()]
            restored = [k.asJSON() for k in target.parm(name).keyframes()]
            assert restored == source, (name, restored, source)
        again = target.hdaModule().write_params_only({"node": target})
        with open(again) as handle:
            reexported = json.load(handle)
        assert reexported["geometry"][1].get("type") == \
            data["geometry"][1].get("type")
        assert reexported["boundary_conditions"].get(
            "obstacle_displacements") == data["boundary_conditions"].get(
            "obstacle_displacements")
    print("PASS: Import restores an animated obstacle's keyframes from the "
          "scene record (translation tables and mesh sequence alike)")

    path, data = translated.write()
    os.remove(os.path.join(os.path.dirname(path), "hda_scene.json"))
    target = import_into(root, os.path.dirname(path),
                         "import_translation_without_record")
    again = target.hdaModule().write_params_only({"node": target})
    with open(again) as handle:
        reexported = json.load(handle)
    first = data["boundary_conditions"]["obstacle_displacements"][0]
    second = reexported["boundary_conditions"]["obstacle_displacements"][0]
    for a, b in zip(first["interpolation"], second["interpolation"]):
        assert np.allclose(np.interp(np.linspace(0, 1, 9), a["points"],
                                     a["values"]),
                           np.interp(np.linspace(0, 1, 9), b["points"],
                                     b["values"]), atol=1e-14), (a, b)
    print("PASS: without the record, displacement tables come back as linear "
          "keys on Translate that export the same tables")

    path, data = rotated.write()
    os.remove(os.path.join(os.path.dirname(path), "hda_scene.json"))
    target = import_into(root, os.path.dirname(path),
                         "import_sequence_without_record")
    report = target.evalParm("import_report")
    assert "first frame" in report, report
    assert target.evalParm("is_obstacle2") \
        and not target.parm("xform_r__2z").keyframes()
    print("PASS: a mesh sequence without the record imports as its first "
          "frame, standing still, with a note")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"),
                        force_use_assets=True)
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)
    hou.setFps(24)
    root = tempfile.mkdtemp(prefix="polyfem_animated_obstacles_")
    translated = check_translation(root)
    rotated = check_rotation(root)
    check_msh_obstacle(root)
    check_refusals(root)
    check_import(root, translated, rotated)
    print("workdir:", root)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
