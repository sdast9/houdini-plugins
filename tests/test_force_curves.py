"""Force curves per sideset and obstacle; Read PVD on the simulation timeline.

Review item F4 (2026-10-02). With Output > Nodal Forces on, PolyFEM writes
every node's force of every term of the energy. Read PVD sums them over the
nodes of each sideset and obstacle recorded by the PolyFEM node
(input/hda_scene.json): the reaction (minus the sum of every term, the force
the prescribed displacement or obstacle motion applies there) and the contact
force. Checked with the real solver:

* uniaxial bar, linear elastic, nu = 0.3 on symmetry planes: the grip's
  reaction is E A strain at every step (uniform stress, exact), the bottom's
  is equal and opposite, the symmetry planes carry none;
* NeoHookean (nu = 0): mu (lambda - 1/lambda) A, and E A strain within the
  geometric nonlinearity;
* equilibrium: the reactions balance a traction load and gravity;
* an obstacle pushing on a body: the force on the obstacle equals minus the
  body's support reaction;
* a dynamic run: the reaction includes inertia (minus the sum of the applied
  and inertia forces over the body);
* P2 tetrahedra (edge nodes found) and Q1 hexahedra (quad faces);
* a sideset without a condition measures (zero reaction, contact if any);
* Read PVD's time mapping (Automatic from the record, one frame per step,
  custom), the readout, the CSV and an offscreen render of the chart;
* runs without the record, or without nodal forces, are explained.

Run: hython tests/test_force_curves.py
"""

import csv
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
READER = "readPVD::1.0"
sys.path.insert(0, os.path.join(BASE, "src", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sideset_conditions import write_box  # noqa: E402

X, Y, Z, ALL = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)


def write_hex_box(path, n=2, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    """Structured hexahedral box as MSH 2.2 ASCII (element type 5)."""
    xs = [np.linspace(lo[a], hi[a], n + 1) for a in range(3)]

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i + 1

    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{(n + 1) ** 3}\n")
        for k in range(n + 1):
            for j in range(n + 1):
                for i in range(n + 1):
                    handle.write(f"{vid(i, j, k)} {xs[0][i]:.17g} "
                                 f"{xs[1][j]:.17g} {xs[2][k]:.17g}\n")
        handle.write(f"$EndNodes\n$Elements\n{n ** 3}\n")
        number = 0
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    number += 1
                    corners = (vid(i, j, k), vid(i + 1, j, k),
                               vid(i + 1, j + 1, k), vid(i, j + 1, k),
                               vid(i, j, k + 1), vid(i + 1, j, k + 1),
                               vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1))
                    handle.write(f"{number} 5 2 1 1 "
                                 + " ".join(map(str, corners)) + "\n")
        handle.write("$EndElements\n")
    return path


class Scene:
    """One body (a box of the given height), quasistatic, XML output."""

    count = 0

    def __init__(self, root, label, height=2.0, n=2, hexes=False, dt=0.25):
        Scene.count += 1
        self.work = os.path.join(root, f"{Scene.count:02d}_{label}")
        os.makedirs(self.work)
        writer = write_hex_box if hexes else write_box
        mesh = writer(os.path.join(self.work, "bar.msh"), n=n,
                      hi=(1.0, 1.0, height))
        node = self.node = hou.node("/obj").createNode(TYPE,
                                                       f"f{Scene.count}")
        self.mod = node.hdaModule()
        node.setParms({"working_dir": self.work + "/",
                       "polyfem_bin": POLYFEM_BIN, "use_hdf5": 0,
                       "file_location1": mesh})
        node.parm("file_location1").pressButton()
        node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                       "time_inc_bool": 1, "dt": dt, "enable": 0,
                       "materials1_1": 0, "E1_1": 1e5, "nu1_1": 0.3})

    def material(self, token, nu=None):
        items = self.node.parm("materials1_1").menuItems()
        self.node.parm("materials1_1").set(items.index(token))
        if nu is not None:
            self.node.parm("nu1_1").set(nu)

    def sideset(self, pattern, *conditions):
        node = self.node
        j = node.evalParm("sideset_selection1_1") + 1
        node.parm("sideset_selection1_1").set(j)
        node.parm("sideset_selection1_1").pressButton()
        node.setParms({f"basegroup1_1_{j}": pattern,
                       f"Boundary_Condition__1_1_{j}": len(conditions)})
        for k, (kind, text, dims) in enumerate(conditions, 1):
            suffix = f"1_1_{j}_{k}"
            node.setParms({f"boundary_type{suffix}": kind,
                           (f"vector_{suffix}" if kind in (0, 1)
                            else f"value_{suffix}"): text})
            for axis, flag in zip("xyz", dims):
                node.setParms({f"{axis}_dimension{suffix}": flag})
        return j

    def run(self):
        path = self.mod.write_params_only({"node": self.node})
        assert path, f"export failed in {self.work}"
        result = subprocess.run(
            [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
             "--log_level", "warning", "--max_threads", "1"],
            cwd=os.path.dirname(path), capture_output=True, text=True,
            timeout=1800)
        assert result.returncode == 0, (
            f"PolyFEM failed in {self.work}:\n{result.stdout[-2000:]}")
        reader = self.mod.open_results({"node": self.node})
        assert reader is not None
        return reader

    def curves(self):
        reader = self.run()
        curves = reader.hdaModule().force_curves(reader)
        return reader, curves


def column(curves, prefix):
    """Index of the force set whose label starts with prefix."""
    matches = [i for i, label in enumerate(curves["labels"])
               if label.startswith(prefix)]
    assert len(matches) == 1, (prefix, curves["labels"])
    return matches[0]


def check_uniaxial(root):
    # linear elastic, nu = 0.3, symmetry planes: uniform uniaxial stress
    scene = Scene(root, "uniaxial_linear")
    scene.material("LinearElasticity")
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", Z))
    scene.sideset("axis:-x:0.01", (0, "[0, 0, 0]", X))
    scene.sideset("axis:-y:0.01", (0, "[0, 0, 0]", Y))
    scene.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', Z))
    scene.sideset("axis:+x:0.99")  # measurement only
    reader, curves = scene.curves()
    bottom = column(curves, "Geometry 1 subdomain 1 sideset 1 ")
    side_x = column(curves, "Geometry 1 subdomain 1 sideset 2 ")
    grip = column(curves, "Geometry 1 subdomain 1 sideset 4 ")
    free = column(curves, "Geometry 1 subdomain 1 sideset 5 ")
    expected = 1e5 * 1.0 * 0.001 * curves["times"]  # E A strain
    reaction = curves["reaction"]
    assert np.allclose(reaction[:, grip, 2], expected, rtol=1e-8,
                       atol=1e-8), (reaction[:, grip, 2], expected)
    assert np.allclose(reaction[:, bottom, 2], -expected, rtol=1e-8,
                       atol=1e-8)
    # no lateral stress: the symmetry plane's normal reaction and a free
    # face's are zero (nodes on an edge between two sidesets count in both,
    # so their z components carry the edges' share of the grip and support)
    assert np.abs(reaction[:, side_x, 0]).max() < 1e-7
    assert np.abs(reaction[:, free, :2]).max() < 1e-7
    assert curves["nodes"][curves["sets"][grip]] == 9
    assert np.allclose(curves["displacement"][:, grip, 2],
                       0.002 * curves["times"], atol=1e-12)
    print("PASS: uniaxial bar (linear elastic, nu = 0.3, symmetry planes): "
          f"grip reaction = E A strain at every step ({reaction[-1, grip, 2]:.9g}"
          f" vs {expected[-1]:.9g}), bottom equal and opposite, no "
          "normal reaction on a symmetry plane or a free face")
    return reader, curves


def check_neohookean(root):
    scene = Scene(root, "uniaxial_neohookean", dt=1.0)
    scene.material("NeoHookean", nu=0.0)
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    scene.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', ALL))
    _, curves = scene.curves()
    stretch = 1.001
    analytic = 1e5 / 2 * (stretch - 1 / stretch)  # mu (l - 1/l) A, lambda 0
    grip = column(curves, "Geometry 1 subdomain 1 sideset 2 ")
    force = curves["reaction"][-1, grip, 2]
    # the uniform state is exact; what is left is the default Newton
    # tolerance on this single step (about 1e-6)
    assert abs(force - analytic) < 1e-5 * analytic, (force, analytic)
    assert abs(force - 100.0) < 0.06, force  # E A strain, 0.05 % nonlinear
    print(f"PASS: NeoHookean bar: reaction {force:.9g} = mu (lambda - "
          f"1/lambda) A = {analytic:.9g} (within the solver tolerance), and "
          "E A strain = 100 within the 0.05 % geometric nonlinearity")


def check_equilibrium(root):
    scene = Scene(root, "traction_and_gravity", height=1.0)
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    scene.sideset("axis:+z:0.99", (1, "[300, 0, -2000]", ALL))
    scene.node.setParms({"RHS": "[0, 0, 9.81]", "rho1_1": 1000.0})
    _, curves = scene.curves()
    bottom = column(curves, "Geometry 1 subdomain 1 sideset 1 ")
    weight = 1000.0 * 9.81 * 1.0  # rho g V, gravity along -z
    expected = np.array([-300.0, 0.0, 2000.0 + weight])
    reaction = curves["reaction"][-1, bottom]
    assert np.allclose(reaction, expected, rtol=1e-6, atol=1e-6), \
        (reaction, expected)
    print("PASS: the bottom's reaction balances a traction (300, 0, -2000) "
          f"and the weight: {np.round(reaction, 6)}")


def check_obstacle(root):
    scene = Scene(root, "obstacle_push", height=1.0, dt=0.125)
    node = scene.node
    plate = os.path.join(scene.work, "plate.obj")
    with open(plate, "w") as handle:
        for x, y in ((-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)):
            handle.write(f"v {x} {y} 1.05\n")
        handle.write("f 1 3 2\nf 1 4 3\n")
    node.setParms({"num_geos": 2, "geo_int": 2, "is_obstacle2": 1,
                   "file_location2": plate})
    node.parm("file_location2").pressButton()
    node.setParms({"enable": 1, "dhat": 0.01, "cof": 0.0})
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    scene.sideset("axis:+z:0.99")  # the contact face, measured
    keys = []
    for frame, value in ((1, 0.0), (25, -0.08)):
        key = hou.Keyframe()
        key.setFrame(frame)
        key.setValue(value)
        key.setExpression("linear()", hou.exprLanguage.Hscript)
        keys.append(key)
    node.parm("xform_t__2z").setKeyframes(keys)
    _, curves = scene.curves()
    bottom = column(curves, "Geometry 1 subdomain 1 sideset 1 ")
    face = column(curves, "Geometry 1 subdomain 1 sideset 2 ")
    obstacle = column(curves, "Obstacle 2")
    on_obstacle = curves["contact"][:, obstacle]
    assert on_obstacle[-1, 2] > 100, on_obstacle[:, 2]
    # the body pushes the obstacle up exactly as hard as the support pushes
    # the body up
    assert np.allclose(on_obstacle, curves["reaction"][:, bottom],
                       rtol=1e-6, atol=1e-6 * on_obstacle[-1, 2])
    assert np.allclose(curves["contact"][:, face], -on_obstacle,
                       rtol=1e-6, atol=1e-6 * on_obstacle[-1, 2])
    assert np.allclose(curves["reaction"][:, obstacle], -on_obstacle)
    assert np.allclose(curves["displacement"][:, obstacle, 2],
                       -0.08 * curves["times"], atol=1e-12)
    print("PASS: obstacle push: force on the obstacle = the support's "
          f"reaction = minus the contact force on the face (last step "
          f"{on_obstacle[-1, 2]:.6g}); the obstacle's displacement is its "
          "keyed motion")


def check_dynamic(root):
    scene = Scene(root, "dynamic", height=1.0, dt=0.002)
    node = scene.node
    node.setParms({"quasistatic": 0, "tend": 0.02, "rho1_1": 1000.0})
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    scene.sideset("axis:+z:0.99", (1, "[0, 0, -5000]", ALL))
    reader, curves = scene.curves()
    bottom = column(curves, "Geometry 1 subdomain 1 sideset 1 ")
    module = reader.hdaModule()
    path = reader.evalParm("PVD_file")
    for step in (3, len(curves["steps"]) - 1):
        mesh = module.load_frame(path, step)["Volume"]
        points = np.asarray(mesh["points"])
        _, representative = module._coincidence_groups(points)
        nodes = np.flatnonzero(representative)
        applied = np.zeros(3)
        for name in ("body_forces", "inertia_forces"):
            applied += np.asarray(mesh["point_data"][name])[nodes].sum(axis=0)
        reaction = curves["reaction"][step, bottom]
        assert np.allclose(reaction, -applied, rtol=1e-6,
                           atol=1e-6 * abs(applied).max()), (reaction, applied)
    early = curves["reaction"][3, bottom, 2]
    assert abs(early - 5000.0) > 1.0, early  # inertia counts
    print("PASS: dynamic run: the reaction is minus the applied and inertia "
          "forces over the body (and differs from the static 5000)")


def check_elements(root):
    p2 = Scene(root, "p2", dt=1.0)
    p2.material("LinearElasticity", nu=0.0)
    p2.node.parm("mainOrder1_1").set(1)  # order 2
    p2.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    p2.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', ALL))
    _, curves = p2.curves()
    grip = column(curves, "Geometry 1 subdomain 1 sideset 2 ")
    assert curves["nodes"][curves["sets"][grip]] == 25, curves["nodes"]
    assert abs(curves["reaction"][-1, grip, 2] - 100.0) < 1e-6
    print("PASS: P2 tetrahedra: the grip's 25 nodes (corners and edge "
          "midpoints) give E A strain = 100")

    hexes = Scene(root, "q1", hexes=True, dt=1.0)
    hexes.material("LinearElasticity", nu=0.0)
    hexes.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    hexes.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', ALL))
    _, curves = hexes.curves()
    grip = column(curves, "Geometry 1 subdomain 1 sideset 2 ")
    assert curves["nodes"][curves["sets"][grip]] == 9
    assert abs(curves["reaction"][-1, grip, 2] - 100.0) < 1e-6
    print("PASS: Q1 hexahedra (quad faces): E A strain = 100")


def check_reader(root, reader, curves):
    module = reader.hdaModule()
    # Automatic: the PolyFEM node's timeline (frame 1 = t 0, 24 fps)
    assert reader.parm("time_mapping").evalAsString() == "auto"
    assert module.playbar_range(reader) == (1, 25)
    for frame, step in ((1, 0), (6, 0), (7, 1), (24, 3), (25, 4), (40, 4)):
        assert module.entry_index(reader, frame) == step, (frame, step)
    hou.setFrame(13)
    status = reader.evalParm("sim_time_status")
    assert status.startswith("Step 2 of 4: t = 0.5"), status
    readout = reader.evalParm("force_readout")
    assert "sideset 4" in readout and "reaction" in readout, readout
    # one frame per output step, as before
    reader.parm("time_mapping").set(1)
    assert module.entry_index(reader, 3) == 3
    assert module.playbar_range(reader) == (0, 4)
    # a custom timeline: frame 101 = t 0, 2 simulated s per Houdini s
    reader.setParms({"time_mapping": 2, "time_start_frame": 101,
                     "time_scale": 2.0})
    assert module.entry_index(reader, 104) == 1, module.entry_index(reader, 104)
    assert module.playbar_range(reader) == (101, 113)
    reader.parm("time_mapping").set(0)
    print("PASS: Read PVD plays the run on the PolyFEM node's timeline "
          "(Automatic), one frame per step, or a custom timeline; the "
          "Simulation Time line and the readout follow the frame")

    reader.parm("force_compute").pressButton()
    path = module.force_csv_path(reader)
    assert os.path.isfile(path)
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:2] == ["step", "time"] and len(rows) == 6, rows[0]
    grip = column(curves, "Geometry 1 subdomain 1 sideset 4 ")
    name = f"{curves['labels'][grip]}: reaction z"
    values = [float(row[rows[0].index(name)]) for row in rows[1:]]
    assert np.allclose(values, curves["reaction"][:, grip, 2], rtol=0,
                       atol=0), values
    print("PASS: force_curves.csv holds every step's reaction, contact force "
          "and displacement per set")

    reader.setParms({"force_chart_set": str(curves["sets"][grip]),
                     "force_chart_quantity": 0, "force_chart_component": 2,
                     "force_chart_against": 0})
    x, y, x_label, y_label, title = module._chart_series(reader, curves)
    assert np.allclose(y, curves["reaction"][:, grip, 2]) \
        and x_label == "mean displacement z", (x_label, y_label)
    try:
        from PySide6 import QtGui
    except ImportError:
        print("SKIP: no PySide6 here; the chart was not rendered")
        return
    app = QtGui.QGuiApplication.instance()
    if app is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QtGui.QGuiApplication(["force_chart_test"])
    image = QtGui.QImage(480, 320, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(255, 255, 255))
    painter = QtGui.QPainter(image)
    module.draw_force_chart(painter, image.rect(),
                            (x, y, x_label, y_label, title), 2)
    painter.end()
    pixels = [image.pixelColor(i, j) for i in range(0, 480, 2)
              for j in range(0, 320, 2)]
    blue = sum(1 for c in pixels if c.blue() > 150 and c.red() < 80)
    red = sum(1 for c in pixels if c.red() > 180 and c.green() < 80)
    assert blue > 20 and red > 2, (blue, red)
    print("PASS: the chart renders (curve and step marker drawn offscreen)")


def check_explanations(root):
    scene = Scene(root, "no_forces", dt=1.0)
    scene.node.parm("forces_fields").set(0)
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", ALL))
    scene.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', ALL))
    reader = scene.run()
    module = reader.hdaModule()
    try:
        module.force_curves(reader)
    except module.ForceCurveError as exc:
        assert "Nodal Forces" in str(exc), exc
    else:
        raise AssertionError("no explanation without nodal forces")
    assert "Nodal Forces" in reader.evalParm("force_readout")
    os.remove(os.path.join(scene.work, "input", "hda_scene.json"))
    module._SCENE_RECORD_CACHE.clear()
    try:
        module.force_curves(reader)
    except module.ForceCurveError as exc:
        assert "hda_scene.json" in str(exc), exc
    else:
        raise AssertionError("no explanation without the record")
    # without the record Automatic falls back to one frame per step
    assert module.time_mapping(reader) is None
    print("PASS: a run without nodal forces, or without the PolyFEM node's "
          "record, is explained by name; Automatic then shows one frame per "
          "step")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    for library in ("sop_MSH_Reader.3.0.hdanc",
                    "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                    "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, library),
                            force_use_assets=True)
    hou.setFps(24)
    root = tempfile.mkdtemp(prefix="polyfem_force_curves_")
    reader, curves = check_uniaxial(root)
    check_reader(root, reader, curves)
    check_neohookean(root)
    check_equilibrium(root)
    check_obstacle(root)
    check_dynamic(root)
    check_elements(root)
    check_explanations(root)
    print("workdir:", root)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
