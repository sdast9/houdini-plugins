"""Check Setup: setup problems found before PolyFEM runs.

Each case is checked twice: Check Setup must name it, and the real solver,
given the same setup anyway (bypassing the check), must fail the way the
check says -- so the check never refuses a setup PolyFEM would run. Before
2026-10-01 these only surfaced as solver messages that do not name the
cause ("Reached iteration limit", a JSON type error, "element 0 is
flipped"), or not at all (a 2D mesh).

Run: hython tests/test_check_setup.py
Uses ../polyfem/build/PolyFEM_bin; no gmsh needed.
"""

import os
import subprocess
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_box(path, n=2):
    xs = np.linspace(0.0, 1.0, n + 1)

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i

    points = [(x, y, z) for z in xs for y in xs for x in xs]
    kuhn = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
            (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                     vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                     vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)]
                for t in kuhn:
                    a, b, cc, d = (c[v] for v in t)
                    p = np.array([points[a], points[b], points[cc], points[d]])
                    if np.linalg.det(p[1:] - p[0]) < 0:
                        b, cc = cc, b
                    tets.append((a, b, cc, d))
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(points)}\n")
        for number, (x, y, z) in enumerate(points, 1):
            handle.write(f"{number} {float(x)!r} {float(y)!r} {float(z)!r}\n")
        handle.write(f"$EndNodes\n$Elements\n{len(tets)}\n")
        for number, tet in enumerate(tets, 1):
            handle.write(f"{number} 4 2 1 1 "
                         + " ".join(str(v + 1) for v in tet) + "\n")
        handle.write("$EndElements\n")
    return path


def write_two_hexes(path):
    """Two unit hexahedra sharing a face, physical groups 1 and 2."""
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1),
              (1, 0, 1), (1, 1, 1), (0, 1, 1), (2, 0, 0), (2, 1, 0),
              (2, 0, 1), (2, 1, 1)]
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(points)}\n")
        for number, (x, y, z) in enumerate(points, 1):
            handle.write(f"{number} {x} {y} {z}\n")
        handle.write("$EndNodes\n$Elements\n2\n"
                     "1 5 2 1 1 1 2 3 4 5 6 7 8\n"
                     "2 5 2 2 2 2 9 10 3 6 11 12 7\n$EndElements\n")
    return path


def write_square(path):
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n"
                     "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
                     "$Elements\n2\n1 2 2 1 1 1 2 3\n2 2 2 1 1 1 3 4\n"
                     "$EndElements\n")
    return path


class Scene:
    count = 0

    def __init__(self, root, label, mesh_writer=write_box):
        Scene.count += 1
        self.work = os.path.join(root, f"{Scene.count:02d}_{label}")
        os.makedirs(self.work)
        mesh = mesh_writer(os.path.join(self.work, "mesh.msh"))
        self.node = hou.node("/obj").createNode(TYPE, f"c{Scene.count}")
        self.mod = self.node.hdaModule()
        self.node.setParms({"working_dir": self.work + "/",
                            "polyfem_bin": POLYFEM_BIN, "log_level": 2})
        self.node.setParms({"file_location1": mesh})
        self.node.parm("file_location1").pressButton()
        self.node.setParms({
            "quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
            "time_inc_bool": 1, "dt": 1.0, "enable": 0})
        for vol in range(1, self.node.evalParm("num_volumes1") + 1):
            self.node.setParms({f"materials1_{vol}": 3, f"E1_{vol}": 1e5,
                                f"nu1_{vol}": 0.3})

    def sideset(self, pattern, kind, text, vol=1, dims=(1, 1, 1)):
        node = self.node
        j = node.evalParm(f"sideset_selection1_{vol}") + 1
        node.parm(f"sideset_selection1_{vol}").set(j)
        node.parm(f"sideset_selection1_{vol}").pressButton()
        node.setParms({f"basegroup1_{vol}_{j}": pattern,
                       f"Boundary_Condition__1_{vol}_{j}": 1,
                       f"boundary_type1_{vol}_{j}_1": kind})
        node.setParms({(f"vector_1_{vol}_{j}_1" if kind in (0, 1)
                        else f"value_1_{vol}_{j}_1"): text})
        for axis, flag in zip("xyz", dims):
            node.setParms({f"{axis}_dimension1_{vol}_{j}_1": flag})

    def check(self, purpose="run"):
        return self.mod.check_setup(self.node, purpose)

    def solve_anyway(self):
        """Write the input without the check and run the real solver."""
        params = self.mod.build_params(self.node)
        result = subprocess.run(
            [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
             "--log_level", "info", "--max_threads", "1"],
            cwd=os.path.dirname(params), capture_output=True, text=True,
            timeout=900)
        return result.returncode, result.stdout + result.stderr


def only(report, kind, fragment):
    found = [item for item in report[kind] if fragment in item]
    assert found, f"no {kind[:-1]} containing {fragment!r}: {report}"
    return found[0]


def check_cases_against_solver(root):
    # an unconstrained body in a quasistatic run: warning (legal physics)
    scene = Scene(root, "unconstrained")
    scene.sideset("axis:+z:0.99", 1, "[0, 0, -100]")
    report = scene.check()
    assert not report["errors"], report
    only(report, "warnings", "Geometry 1 is not held in xyz")
    code, log = scene.solve_anyway()
    assert code == 1 and "Reached iteration limit" in log, (code, log[-500:])
    # fixed in z only: still free in x and y
    scene.sideset("axis:-z:0.01", 0, "[0, 0, 0]", dims=(0, 0, 1))
    only(scene.check(), "warnings", "is not held in xy")
    scene.sideset("axis:-z:0.01", 0, "[0, 0, 0]")
    assert not scene.check()["warnings"], scene.check()
    code, log = scene.solve_anyway()
    assert code == 0, log[-500:]
    print("PASS: an unconstrained quasistatic body is a warning; PolyFEM "
          "indeed stops with 'Reached iteration limit', and runs once fixed")

    # Hooke / Saint Venant left at the placeholder tensor: error
    tokens = scene.mod.MATERIAL_TOKENS
    for model in ("HookeLinearElasticity", "SaintVenant"):
        scene.node.setParms({"materials1_1": tokens.index(model)})
        only(scene.check(), "errors", "Replace the placeholder names")
        code, log = scene.solve_anyway()
        assert code == 1 and "type must be number" in log, (model, code,
                                                             log[-400:])
        scene.node.setParms({
            "elast_tensor1_1": "[1e5, 2e5, 3e5, 0.3, 0.3, 0.3, 4e4, 4e4, 4e4]"})
        assert not scene.check()["errors"], scene.check()
        code, log = scene.solve_anyway()
        assert code == 0, (model, log[-400:])
        scene.node.parm("elast_tensor1_1").revertToDefaults()
    scene.node.setParms({"materials1_1": 3})
    print("PASS: the placeholder elasticity tensor is an error (PolyFEM: "
          "JSON type error); a numeric tensor runs")

    # a mirrored transform: error
    scene.node.setParms({"xform_s__1x": -1.0})
    only(scene.check(), "errors", "mirrors the mesh")
    code, log = scene.solve_anyway()
    assert code == 1 and "is flipped" in log, (code, log[-400:])
    scene.node.setParms({"xform_s__1x": 1.0})
    print("PASS: a negative scale is an error (PolyFEM: element 0 is "
          "flipped)")

    # hexahedral orders
    hexes = Scene(root, "hexahedra", write_two_hexes)
    assert hexes.node.evalParm("num_volumes1") == 2
    hexes.sideset("*", 0, "[0, 0, 0]", vol=1)
    hexes.sideset("*", 0, "[0, 0, 0]", vol=2)
    assert not hexes.check()["errors"], hexes.check()
    code, log = hexes.solve_anyway()
    assert code == 0, log[-400:]
    hexes.node.setParms({"mainOrder1_2": 1})            # Q1 next to Q2
    only(hexes.check(), "errors", "touches subdomain 1")
    code, log = hexes.solve_anyway()
    assert code == 1 and "Mixed per-element hexahedral orders" in log, \
        (code, log[-400:])
    hexes.node.setParms({"mainOrder1_1": 1})            # both Q2: fine
    assert not hexes.check()["errors"], hexes.check()
    hexes.node.setParms({"mainOrder1_1": 3, "mainOrder1_2": 3})   # Q4
    only(hexes.check(), "errors", "hexahedra of order 4 are not available")
    code, log = hexes.solve_anyway()
    assert code == 1 and "Q4 hexahedral bases are not available" in log, \
        (code, log[-400:])
    print("PASS: mixed hexahedral orders and Q4 hexahedra are errors, as "
          "PolyFEM refuses them; equal orders run")


def check_other_findings(root):
    flat = Scene(root, "two_d", write_square)
    only(flat.check("write"), "errors", "no tetrahedra or hexahedra")
    assert flat.mod.write_params_only({"node": flat.node}) is None

    scene = Scene(root, "findings")
    scene.sideset("axis:-z:0.01", 0, "[0, 0, 0]")
    assert scene.check() == {"errors": [], "warnings": [], "notes": []} or \
        not (scene.check()["errors"] or scene.check()["warnings"])
    node = scene.node
    # binary: needed to run, not to write
    node.setParms({"polyfem_bin": os.path.join(root, "missing_bin")})
    only(scene.check("run"), "errors", "PolyFEM Binary not found")
    assert not scene.check("write")["errors"]
    node.setParms({"polyfem_bin": POLYFEM_BIN})
    # material values PolyFEM cannot use
    node.setParms({"E1_1": 0.0})
    only(scene.check(), "errors", "Young's modulus E must be greater than 0")
    node.setParms({"E1_1": 1e5, "nu1_1": 0.5})
    only(scene.check(), "errors", "Poisson's ratio must lie between")
    node.setParms({"nu1_1": 0.3, "materials1_1":
                   scene.mod.MATERIAL_TOKENS.index("MooneyRivlin")})
    only(scene.check(), "errors", "C1 and C2 are both 0")
    node.setParms({"materials1_1": 3, "quasistatic": 0, "rho1_1": 0.0})
    only(scene.check(), "errors", "density must be greater than 0")
    node.setParms({"quasistatic": 1, "rho1_1": 1000.0})
    # time
    node.setParms({"tend": 0.0})
    only(scene.check(), "errors", "End Time")
    node.setParms({"tend": 1.0})
    # contact distance and method
    node.setParms({"enable": 1, "dhat": 0.05})
    only(scene.check(), "warnings", "of the model's size")
    node.setParms({"dhat": 1e-3})
    assert not any("model's size" in w for w in scene.check()["warnings"])
    node.parm("solver_nl").set("L-BFGS")
    only(scene.check(), "warnings", "only Newton converges")
    node.parm("solver_nl").set("Newton")
    node.setParms({"enable": 0})
    # a sideset conflict (refused at export since 2026-10-01) is listed too
    scene.sideset("axis:+z:0.99", 0, '[0, 0, "-0.1*t"]', dims=(0, 0, 1))
    j = node.evalParm("sideset_selection1_1")
    node.setParms({f"Boundary_Condition__1_1_{j}": 2,
                   f"boundary_type1_1_{j}_2": 0,
                   f"vector_1_1_{j}_2": '[0, 0, "-0.2*t"]',
                   f"x_dimension1_1_{j}_2": 0, f"y_dimension1_1_{j}_2": 0})
    only(scene.check(), "errors", "both prescribe the z displacement")
    node.setParms({f"Boundary_Condition__1_1_{j}": 1})
    # a sideset with faces but no condition
    k = node.evalParm("sideset_selection1_1") + 1
    node.parm("sideset_selection1_1").set(k)
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({f"basegroup1_1_{k}": "axis:+x:0.99",
                   f"Boundary_Condition__1_1_{k}": 0})
    only(scene.check(), "notes", "has a selection but no boundary condition")
    print("PASS: 2D meshes, missing binary, impossible material values, "
          "time, contact distance, solver method, sideset conflicts and "
          "idle sidesets are reported")
    return scene


class FakeUI:
    def __init__(self, answer):
        self.answer, self.asked = answer, 0

    def displayMessage(self, text, buttons=("OK",), **kwargs):
        if "Run Anyway" in buttons:
            self.asked += 1
            return self.answer
        return 0

    def setStatusMessage(self, *args, **kwargs):
        pass


def with_ui(answer, action):
    real = hou.isUIAvailable
    fake = FakeUI(answer)
    hou.ui = fake
    hou.isUIAvailable = lambda: True
    try:
        return action(), fake.asked
    finally:
        hou.isUIAvailable = real
        del hou.ui


def check_gate(scene):
    node, mod = scene.node, scene.mod
    node.setParms({"tend": 1.0, "dt": 1.0, "enable": 0})
    for vol_j in range(1, node.evalParm("sideset_selection1_1") + 1):
        node.setParms({f"Boundary_Condition__1_1_{vol_j}": 0})
    node.setParms({f"Boundary_Condition__1_1_1": 1,
                   "basegroup1_1_1": "axis:+z:0.99",
                   "boundary_type1_1_1_1": 1,
                   "vector_1_1_1_1": "[0, 0, -100]"})
    report = mod.check_setup(node, "background")
    assert not report["errors"] and report["warnings"], report
    # the button writes the report
    mod.check_setup_button({"node": node})
    assert "WARNING: Geometry 1 is not held" in node.evalParm("check_report")
    # without an interface warnings are printed and the write goes ahead
    assert mod.write_params_only({"node": node})
    # with one, a run asks once: Cancel stops, Run Anyway is remembered
    result, asked = with_ui(1, lambda: mod.gate_setup(node, "background"))
    assert result is False and asked == 1
    result, asked = with_ui(0, lambda: mod.gate_setup(node, "background"))
    assert result is True and asked == 1
    result, asked = with_ui(0, lambda: mod.gate_setup(node, "background"))
    assert result is True and asked == 0, "asked again for the same warnings"
    # writing only reports warnings, never asks
    result, asked = with_ui(1, lambda: mod.gate_setup(node, "write"))
    assert result is True and asked == 0
    # errors refuse every action
    node.setParms({"E1_1": -1.0})
    assert mod.write_params_only({"node": node}) is None
    assert "ERROR: Geometry 1 subdomain 1" in node.evalParm("check_report")
    print("PASS: Write and Run run the check: errors refuse; warnings are "
          "confirmed before a run (once per set) and only reported by Write")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    root = tempfile.mkdtemp(prefix="polyfem_check_setup_")
    check_cases_against_solver(root)
    scene = check_other_findings(root)
    check_gate(scene)
    print("workdir:", root)


if __name__ == "__main__":
    main()
