"""Time curves for boundary conditions and the simulation timeline (2026-10-02).

Review item F3. Every sideset condition has a Time Curve: Value as entered
(as before), Ramp from 0 to the value, or Animated multiplier (keyframed on
Houdini's timeline). Houdini time is simulation time: t = Start Time +
(T - T_start) * Time Scale. The curve multiplies the value, as PolyFEM's
interpolation does, and is exported as a piecewise-linear table sampled at
every time step (the only times PolyFEM evaluates a condition).

Checked with the real solver:
* a keyframed (ease) Dirichlet multiplier is reproduced at every step, per
  component; halving dt keeps the curve in time;
* a ramp equals the same ramp written as an expression of t;
* Start Frame and Time Scale move the curve as stated;
* a keyed Neumann load, two loads with different curves on one sideset, a
  normal traction and a pressure (PolyFEM ddb4579a3 or later) equal the same
  loads written as expressions of t;
* refusals: ramp end before its start, a zero Time Scale, two curves on one
  Dirichlet component, added loads with different curves and expression
  values;
* Import: exact keyframes from the sideset record; from params.json alone,
  linear keys reproducing the tables at every step, including PolyFEM's
  piecewise_cubic / piecewise_constant / linear_ramp time functions (checked
  against PolyFEM's own evaluation by running them).

Run: hython tests/test_time_curves.py
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

DIRICHLET, NEUMANN, NORMAL, PRESSURE = 0, 1, 2, 3
ENTERED, RAMP, ANIMATED = 0, 1, 2


class Scene:
    """A unit cube (48 tets), quasistatic, bottom fixed, XML output."""

    count = 0

    def __init__(self, root, label, dt=0.1, tend=1.0, fixed=True):
        Scene.count += 1
        self.work = os.path.join(root, f"{Scene.count:02d}_{label}")
        os.makedirs(self.work)
        mesh = write_box(os.path.join(self.work, "box.msh"), n=2)
        self.node = hou.node("/obj").createNode(TYPE, f"s{Scene.count}")
        self.mod = self.node.hdaModule()
        self.node.setParms({"working_dir": self.work + "/",
                            "polyfem_bin": POLYFEM_BIN, "use_hdf5": 0,
                            "file_location1": mesh})
        self.node.parm("file_location1").pressButton()
        self.node.setParms({
            "quasistatic": 1, "end_time_bool": 1, "tend": tend,
            "time_inc_bool": 1, "dt": dt, "num_timesteps_bool": 0,
            "enable": 0, "materials1_1": 0, "E1_1": 1e5, "nu1_1": 0.3})
        if fixed:
            self.sideset("axis:-z:0.01", (DIRICHLET, "[0, 0, 0]"))

    def sideset(self, pattern, *conditions):
        """conditions: (kind, text[, dims]); returns (sideset, suffixes)."""
        node = self.node
        j = node.evalParm("sideset_selection1_1") + 1
        node.parm("sideset_selection1_1").set(j)
        node.parm("sideset_selection1_1").pressButton()
        node.setParms({f"basegroup1_1_{j}": pattern,
                       f"Boundary_Condition__1_1_{j}": len(conditions)})
        suffixes = []
        for k, condition in enumerate(conditions, 1):
            kind, text = condition[0], condition[1]
            dims = condition[2] if len(condition) > 2 else (1, 1, 1)
            suffix = f"1_1_{j}_{k}"
            node.setParms({f"boundary_type{suffix}": kind,
                           (f"vector_{suffix}" if kind in (0, 1)
                            else f"value_{suffix}"): text})
            for axis, flag in zip("xyz", dims):
                node.setParms({f"{axis}_dimension{suffix}": flag})
            suffixes.append(suffix)
        return j, suffixes

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
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, (
        f"PolyFEM failed in {input_dir}:\n{result.stdout[-2000:]}"
        f"\n{result.stderr[-800:]}")
    pvd = os.path.join(os.path.dirname(input_dir), "output", "sim.pvd")
    entries = vtu_parser.read_pvd(pvd)
    steps = []
    for index, (time, _) in enumerate(entries):
        mesh = vtu_parser.load_frame(pvd, index)["Volume"]
        steps.append((time, np.asarray(mesh["points"]),
                      np.asarray(mesh["point_data"]["solution"])))
    return steps


def top(steps, axis=2):
    """Mean displacement component of the top face (z = 1) per step."""
    return np.array([solution[points[:, 2] > 0.999, axis].mean()
                     for _, points, solution in steps])


def keys(parm, points, expression="linear()"):
    keyframes = []
    for frame, value in points:
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression(expression, hou.exprLanguage.Hscript)
        keyframes.append(key)
    parm.setKeyframes(keyframes)


def expect_error(text, action):
    try:
        action()
    except hou.Error as exc:
        assert text in str(exc), f"expected '{text}' in: {exc}"
        return str(exc)
    raise AssertionError(f"no error containing '{text}'")


def entry_of(data, key, sid):
    entries = [entry for entry in data["boundary_conditions"].get(key, [])
               if entry["id"] == sid]
    assert len(entries) == 1, entries
    return entries[0]


def close(a, b, tolerance=1e-12):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(1.0, float(np.abs(a).max()), float(np.abs(b).max()))
    return float(np.abs(a - b).max()) <= tolerance * scale


def check_timeline_rules(root):
    scene = Scene(root, "timeline", fixed=False)
    node, mod = scene.node, scene.mod
    node.setParms({"t0": 0.5, "tend": 1.3, "dt": 0.25})
    line = mod.timeline(node)  # tend + dt: ceil((1.3 - 0.5) / 0.25) = 4
    assert line.steps == 4 and close(line.times, 0.5 + 0.25 * np.arange(5)), \
        line.times
    node.setParms({"time_inc_bool": 0, "num_timesteps_bool": 1,
                   "num_timesteps": 8})
    line = mod.timeline(node)  # tend + steps
    assert line.steps == 8 and close(line.dt, 0.1), (line.steps, line.dt)
    node.setParms({"end_time_bool": 0, "time_inc_bool": 1, "dt": 0.05})
    line = mod.timeline(node)  # dt + steps
    assert line.steps == 8 and close(line.times[-1], 0.9), line.times
    node.setParms({"houdini_start_frame": 11, "time_scale": 0.5})
    line = mod.timeline(node)
    assert close(line.houdini_times[0], hou.frameToTime(11))
    assert close(line.houdini_times[-1], hou.frameToTime(11) + 0.4 / 0.5)
    text = node.evalParm("timeline_info")
    assert "Frame 11 = t 0.5" in text, text
    node.setParms({"time_scale": 0})
    expect_error("Time Scale", lambda: mod.timeline(node))
    node.setParms({"end_time_bool": 0, "time_inc_bool": 0,
                   "num_timesteps_bool": 1, "time_scale": 1})
    expect_error("two of End Time", lambda: mod.timeline(node))
    print("PASS: step times follow PolyFEM's rules (tend+dt with ceil, "
          "tend+steps, dt+steps); Start Frame and Time Scale map them")


def check_keyed_dirichlet(root):
    scene = Scene(root, "keyed_dirichlet")
    j, (suffix,) = scene.sideset("axis:+z:0.99",
                                 (DIRICHLET, "[0.002, 0, -0.01]"))
    node = scene.node
    node.parm(f"time_curve{suffix}").set(ANIMATED)
    multiplier = node.parmTuple(f"curve_multiplier{suffix}")
    keys(multiplier[2], [(1, 0.0), (13, 1.0), (25, 0.25)], "ease()")
    keys(multiplier[0], [(1, 1.0), (25, -1.0)])
    steps, data = scene.run()
    entry = entry_of(data, "dirichlet_boundary", scene.mod.sideset_id(1, 1, j))
    curves = entry["interpolation"]
    assert len(curves) == 3 and curves[1] == {"type": "none"}, curves
    # the linear x keys compress to their end points; the ease z keys keep
    # every step
    assert curves[0]["points"] == [0.0, 1.0], curves[0]
    assert len(curves[2]["points"]) == 11, curves[2]
    line = scene.mod.timeline(node)
    expected_z = [-0.01 * multiplier[2].evalAtTime(float(T))
                  for T in line.houdini_times]
    expected_x = [0.002 * multiplier[0].evalAtTime(float(T))
                  for T in line.houdini_times]
    # PolyFEM writes the initial (unloaded) state at the start time and
    # applies the conditions from the first step on
    assert top(steps)[0] == 0.0 and top(steps, 0)[0] == 0.0
    assert close(top(steps)[1:], expected_z[1:]), (top(steps), expected_z)
    assert close(top(steps, 0)[1:], expected_x[1:]), \
        (top(steps, 0), expected_x)
    # the curve is really curved: not the line through the keys
    assert abs(expected_z[3] + 0.01 * 0.6) > 1e-4, expected_z[3]
    print("PASS: a keyframed Dirichlet multiplier (ease keys on z, linear on "
          "x) is reproduced at every step: "
          f"max error {np.abs(top(steps)[1:] - expected_z[1:]).max():.1e}")

    # halving dt samples the same channel: same displacement at every time
    # both runs reach
    fine = Scene(root, "keyed_dirichlet_fine", dt=0.05)
    _, (fine_suffix,) = fine.sideset("axis:+z:0.99",
                                     (DIRICHLET, "[0.002, 0, -0.01]"))
    fine.node.parm(f"time_curve{fine_suffix}").set(ANIMATED)
    fine_multiplier = fine.node.parmTuple(f"curve_multiplier{fine_suffix}")
    keys(fine_multiplier[2], [(1, 0.0), (13, 1.0), (25, 0.25)], "ease()")
    keys(fine_multiplier[0], [(1, 1.0), (25, -1.0)])
    fine_steps, _ = fine.run()
    assert close(top(fine_steps)[::2], top(steps)), \
        (top(fine_steps)[::2], top(steps))
    print("PASS: halving dt keeps the curve in time (the coarse run's steps "
          "equal every second fine step)")


def check_ramp(root):
    ramp = Scene(root, "ramp")
    _, (suffix,) = ramp.sideset("axis:+z:0.99", (DIRICHLET, "[0, 0, -0.01]"))
    node = ramp.node
    node.parm(f"time_curve{suffix}").set(RAMP)
    # by default the ramp spans the whole run (Start Time .. last step)
    node.setParms({"t0": 0.5, "tend": 1.5})
    steps, data = ramp.run()
    reference = Scene(root, "ramp_expression")
    reference.node.setParms({"t0": 0.5, "tend": 1.5})
    reference.sideset("axis:+z:0.99", (DIRICHLET, '[0, 0, "-0.01*(t-0.5)"]'))
    expected, _ = reference.run()
    assert close(top(steps), top(expected)), (top(steps), top(expected))
    # custom times; typing them never touches the Time tab
    node.setParms({"t0": 0.0, "tend": 1.0, f"ramp_custom{suffix}": 1,
                   f"ramp_start{suffix}": 0.2, f"ramp_end{suffix}": 0.8})
    assert (node.evalParm("t0"), node.evalParm("tend")) == (0.0, 1.0)
    steps, data = ramp.run()
    reference = Scene(root, "ramp_custom_expression")
    reference.sideset("axis:+z:0.99", (
        DIRICHLET, '[0, 0, "-0.01*min(max((t-0.2)/0.6,0),1)"]'))
    expected, _ = reference.run()
    assert close(top(steps), top(expected)), (top(steps), top(expected))
    print("PASS: Ramp from 0 to the value (over the whole run, and with "
          "custom times) equals the same ramp written as an expression of t")
    node.setParms({f"ramp_end{suffix}": 0.2})
    expect_error("Ramp End", lambda: ramp.mod.build_params(node))
    report = ramp.mod.check_setup(node)
    assert any("Ramp End" in error for error in report["errors"]), report
    print("PASS: a ramp that ends before it starts is refused by name")


def check_start_frame_and_scale(root):
    scene = Scene(root, "start_scale", dt=0.25)
    j, (suffix,) = scene.sideset("axis:+z:0.99", (DIRICHLET, "[0, 0, -0.01]"))
    node = scene.node
    node.setParms({"houdini_start_frame": 11, "time_scale": 0.5,
                   f"time_curve{suffix}": ANIMATED})
    multiplier = node.parmTuple(f"curve_multiplier{suffix}")[2]
    # frame 11 = t 0; 0.5 simulated s per Houdini s: t 1 = 2 s = frame 59
    keys(multiplier, [(11, 0.0), (59, 1.0)])
    _, data = scene.write()
    table = entry_of(data, "dirichlet_boundary",
                     scene.mod.sideset_id(1, 1, j))["interpolation"][-1]
    assert close(table["points"], [0.0, 1.0]) \
        and close(table["values"], [0.0, 1.0]), table
    keys(multiplier, [(11, 0.0), (35, 1.0)])  # t 0.5 at frame 35
    _, data = scene.write()
    table = entry_of(data, "dirichlet_boundary",
                     scene.mod.sideset_id(1, 1, j))["interpolation"][-1]
    assert close(table["points"], [0.0, 0.5, 1.0]) \
        and close(table["values"], [0.0, 1.0, 1.0]), table
    print("PASS: Start Frame and Time Scale place the keys in simulation "
          "time (frame 11 = t 0, frame 35 = t 0.5 at a scale of 0.5)")
    node.setParms({"time_scale": 0})
    report = scene.mod.check_setup(node)
    assert any("Time Scale" in error for error in report["errors"]), report
    print("PASS: a zero Time Scale is refused by Check Setup")


def check_loads(root):
    # a keyed Neumann load = the same load as an expression of t
    keyed = Scene(root, "keyed_neumann")
    _, (suffix,) = keyed.sideset("axis:+z:0.99", (NEUMANN, "[0, 0, -2000]"))
    keyed.node.parm(f"time_curve{suffix}").set(ANIMATED)
    keys(keyed.node.parmTuple(f"curve_multiplier{suffix}")[2],
         [(1, 0.0), (13, 1.0), (25, 0.5)])
    steps, _ = keyed.run()
    explicit = Scene(root, "expression_neumann")
    explicit.sideset("axis:+z:0.99",
                     (NEUMANN, '[0, 0, "-2000*min(2*t,1.5-t)"]'))
    expected, _ = explicit.run()
    assert close(top(steps), top(expected)) and abs(top(steps)).max() > 1e-3
    print("PASS: a keyed Neumann load equals the same load as an expression "
          "of t")

    # two loads with different curves on one sideset are added into one
    # table per component
    two = Scene(root, "two_loads")
    j, (first, second) = two.sideset(
        "axis:+z:0.99", (NEUMANN, "[0, 0, -2000]"), (NEUMANN, "[800, 0, 0]"))
    node = two.node
    node.setParms({f"time_curve{first}": RAMP, f"time_curve{second}": ANIMATED})
    keys(node.parmTuple(f"curve_multiplier{second}")[0],
         [(1, 0.0), (13, 1.0), (25, 0.0)])
    steps, data = two.run()
    entry = entry_of(data, "neumann_boundary", two.mod.sideset_id(1, 1, j))
    assert entry["value"] == [1.0, 1.0, 1.0], entry
    sum_reference = Scene(root, "two_loads_expression")
    sum_reference.sideset("axis:+z:0.99", (
        NEUMANN, '["800*min(2*t,2-2*t)", 0, "-2000*t"]'))
    expected, _ = sum_reference.run()
    for axis in (0, 2):
        assert close(top(steps, axis), top(expected, axis)), axis
    print("PASS: two loads with different curves on one sideset are added "
          "into one table per component (= the sum as an expression)")

    # ... which needs numbers, not expressions, as values
    node.setParms({f"vector_{first}": '[0, 0, "-2000*x"]'})
    expect_error("must be numbers", lambda: two.mod.build_params(node))
    print("PASS: added loads with different curves and an expression value "
          "are refused by name")

    # scalar conditions: normal traction and pressure (PolyFEM ddb4579a3+)
    for kind, label, faces in ((NORMAL, "normal", "axis:+z:0.99"),
                               (PRESSURE, "pressure", "box:[-1,-1,0.01],[2,2,2]")):
        keyed = Scene(root, f"keyed_{label}")
        _, (suffix,) = keyed.sideset(faces, (kind, "300"))
        keyed.node.parm(f"time_curve{suffix}").set(ANIMATED)
        keys(keyed.node.parm(f"curve_scale{suffix}"),
             [(1, 0.0), (13, 1.0), (25, 0.5)])
        steps, data = keyed.run()
        key = {NORMAL: "normal_aligned_neumann_boundary",
               PRESSURE: "pressure_boundary"}[kind]
        assert isinstance(data["boundary_conditions"][key][0]["interpolation"],
                          dict)
        explicit = Scene(root, f"expression_{label}")
        explicit.sideset(faces, (kind, '"300*min(2*t,1.5-t)"'))
        expected, _ = explicit.run()
        assert close(top(steps), top(expected)) \
            and abs(top(steps)).max() > 1e-5, (top(steps), top(expected))
        print(f"PASS: a keyed {label} curve equals the same {label} as an "
              "expression of t")


def check_dirichlet_conflict(root):
    scene = Scene(root, "conflict")
    _, (first, second) = scene.sideset(
        "axis:+z:0.99", (DIRICHLET, "[0, 0, -0.01]", (0, 0, 1)),
        (DIRICHLET, "[0, 0, -0.01]", (0, 0, 1)))
    scene.node.parm(f"time_curve{first}").set(RAMP)
    expect_error("different time curves",
                 lambda: scene.mod.build_params(scene.node))
    scene.node.parm(f"time_curve{second}").set(RAMP)
    scene.write()
    print("PASS: one Dirichlet component with two different time curves is "
          "refused; with the same curve the conditions combine")


def check_import_with_record(root):
    scene = Scene(root, "import_record")
    j, (suffix,) = scene.sideset("axis:+z:0.99",
                                 (DIRICHLET, "[0, 0, -0.01]"))
    node = scene.node
    node.parm(f"time_curve{suffix}").set(ANIMATED)
    keys(node.parmTuple(f"curve_multiplier{suffix}")[2],
         [(1, 0.0), (13, 1.0), (25, 0.25)], "bezier()")
    node.setParms({"houdini_start_frame": 1, "time_scale": 1})
    path, data = scene.write()
    source_keys = [k.asJSON() for k in
                   node.parmTuple(f"curve_multiplier{suffix}")[2].keyframes()]

    target = hou.node("/obj").createNode(TYPE, "import_record_target")
    target.setParms({"old_input_dir": os.path.dirname(path)})
    target.hdaModule().read_params({"node": target})
    suffix2 = f"1_1_{j}_1"
    assert target.parm(f"time_curve{suffix2}").evalAsString() == "animated"
    restored = [k.asJSON() for k in
                target.parmTuple(f"curve_multiplier{suffix2}")[2].keyframes()]
    assert restored == source_keys, (restored, source_keys)
    copy = os.path.join(root, "import_record_copy")
    os.makedirs(copy)
    target.setParms({"working_dir": copy + "/"})
    again = target.hdaModule().write_params_only({"node": target})
    with open(again) as handle:
        reexported = json.load(handle)
    sid = scene.mod.sideset_id(1, 1, j)
    assert entry_of(reexported, "dirichlet_boundary", sid) == \
        entry_of(data, "dirichlet_boundary", sid)
    print("PASS: Import restores the exact keyframes from the sideset record "
          "and exports the same table again")


def check_import_tables(root):
    """No record: params.json tables become linear keys that reproduce them
    at every step; PolyFEM's other time functions are evaluated exactly
    (checked by running them)."""
    for label, function in (
            ("cubic", {"type": "piecewise_cubic", "points": [0, 0.3, 0.7, 1],
                       "values": [0, 1, 0.4, 0.8], "extend": "constant"}),
            ("constant", {"type": "piecewise_constant",
                          "points": [0, 0.25, 0.55, 1],
                          "values": [0.1, 1, 0.3, 0.6], "extend": "constant"}),
            ("ramp", {"type": "linear_ramp", "from": 0.15, "to": 0.65}),
            ("repeat", {"type": "piecewise_linear", "points": [0, 0.35],
                        "values": [0, 1], "extend": "repeat_offset"})):
        scene = Scene(root, f"import_{label}")
        j, _ = scene.sideset("axis:+z:0.99", (DIRICHLET, "[0, 0, -0.01]"))
        path, data = scene.write()
        sid = scene.mod.sideset_id(1, 1, j)
        entry_of(data, "dirichlet_boundary", sid)["interpolation"] = [function]
        with open(path, "w") as handle:
            json.dump(data, handle, indent=2)
        os.remove(os.path.join(os.path.dirname(path), "hda_sidesets.json"))
        polyfem_steps = run_input(os.path.dirname(path))

        target = hou.node("/obj").createNode(TYPE, f"import_{label}_target")
        target.setParms({"old_input_dir": os.path.dirname(path)})
        target.hdaModule().read_params({"node": target})
        assert target.parm(f"time_curve1_1_{j}_1").evalAsString() in \
            ("animated", "ramp")
        rerun = os.path.join(root, f"import_{label}_rerun")
        os.makedirs(rerun)
        target.setParms({"working_dir": rerun + "/", "use_hdf5": 0,
                         "polyfem_bin": POLYFEM_BIN})
        again = target.hdaModule().write_params_only({"node": target})
        steps = run_input(os.path.dirname(again))
        assert close(top(steps), top(polyfem_steps)), \
            (label, top(steps), top(polyfem_steps))
        print(f"PASS: an imported {function['type']} time function "
              f"({function.get('extend', 'ramp')}) comes back as keyframes "
              "that reproduce PolyFEM's own run at every step")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"),
                        force_use_assets=True)
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)
    hou.setFps(24)
    root = tempfile.mkdtemp(prefix="polyfem_time_curves_")
    check_timeline_rules(root)
    check_keyed_dirichlet(root)
    check_ramp(root)
    check_start_frame_and_scale(root)
    check_loads(root)
    check_dirichlet_conflict(root)
    check_import_with_record(root)
    check_import_tables(root)
    print("workdir:", root)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
