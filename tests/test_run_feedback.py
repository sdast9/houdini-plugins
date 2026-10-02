"""Running PolyFEM from the node: status, stop, logs, results, and paths.

Until 2026-10-01 Run in Background returned at once and nothing read the
exit status; a second click started another run into the same folder; Run
PolyFEM joined the command unquoted into nested AppleScript quotes (a space
in the binary path or an apostrophe in the folder broke it); and "$HIP/sim"
was rewritten to an absolute path, so a moved .hip lost its inputs.

Run: hython tests/test_run_feedback.py
Uses ../polyfem/build/PolyFEM_bin; no gmsh needed. No terminal window is
opened: the terminal script is run with bash directly.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_box(path, n=2):
    """Structured tet box (six tets per cell) as MSH 2.2 ASCII."""
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


def scene(work, name, binary=POLYFEM_BIN, steps=2):
    """A pushed cube: bottom fixed, top pushed down over `steps` steps."""
    os.makedirs(work, exist_ok=True)
    node = hou.node("/obj").createNode(TYPE, name)
    node.setParms({"working_dir": work + "/", "polyfem_bin": binary,
                   "log_level": 2})
    mesh = write_box(os.path.join(work, "cube.msh"))
    node.setParms({"file_location1": mesh})
    node.parm("file_location1").pressButton()
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 1.0 / steps, "enable": 0,
                   "materials1_1": 3, "E1_1": 1e5, "nu1_1": 0.3})
    for pattern, vector in (("axis:-z:0.01", "[0, 0, 0]"),
                            ("axis:+z:0.99", '[0, 0, "-0.1*t"]')):
        j = node.evalParm("sideset_selection1_1") + 1
        node.parm("sideset_selection1_1").set(j)
        node.parm("sideset_selection1_1").pressButton()
        node.setParms({f"basegroup1_1_{j}": pattern,
                       f"Boundary_Condition__1_1_{j}": 1,
                       f"boundary_type1_1_{j}_1": 0,
                       f"vector_1_1_{j}_1": vector})
    return node


def wait(module, node, limit=300.0, until=None):
    start = time.time()
    while time.time() - start < limit:
        active = module.poll_runs()
        if until is not None and until(node.evalParm("run_status")):
            return node.evalParm("run_status")
        if not active and until is None:
            return node.evalParm("run_status")
        time.sleep(0.25)
    raise AssertionError(f"timed out; status: {node.evalParm('run_status')}")


def check_background(root):
    node = scene(os.path.join(root, "background"), "background")
    module = node.hdaModule()
    run = module.run_background({"node": node})
    assert run is not None and run["kind"] == "background"
    status = wait(module, node)
    assert status.startswith("Completed in") and "2 time steps" in status, \
        status
    record = json.load(open(os.path.join(root, "background", "output",
                                         ".polyfem_run.json")))
    assert record["kind"] == "background" and record["pid"] == run["pid"]
    print(f"PASS: Run in Background is followed to the end: {status!r}")

    # A named failure: the status carries the meaning and PolyFEM's reason.
    failing = scene(os.path.join(root, "failure"), "failure")
    failing.setParms({"max_iterations": 1})
    failing.hdaModule().run_background({"node": failing})
    status = wait(module, failing)
    assert status.startswith("Failed before the first time step was done: "
                             "exit status 1, a named failure (not a crash)"), \
        status
    assert "PolyFEM stopped: " in status and "iteration limit" in status, \
        status
    print(f"PASS: a failed run says what exit status 1 means and why: "
          f"{status[:90]}...")
    return node


def check_stop_and_refusal(root):
    node = scene(os.path.join(root, "long"), "long", steps=2000)
    module = node.hdaModule()
    run = module.run_background({"node": node})
    wait(module, node, until=lambda s: "time steps done" in s)
    # A second run into the same folder is refused while the first goes on.
    assert module.run_background({"node": node}) is None
    assert module.write_params_only({"node": node}) is None
    assert module.active_run(node) is not None
    # So is a run from another node pointed at the same folder.
    other = hou.node("/obj").createNode(TYPE, "same_folder")
    other.setParms({"working_dir": node.evalParm("working_dir")})
    assert other.hdaModule().active_run(other) is not None
    # After a reload the node finds its run again (OnLoaded).
    module._runs().clear()
    module.reattach_run(node)
    assert "Running" in node.evalParm("run_status"), \
        node.evalParm("run_status")
    assert module.stop_run({"node": node})
    status = wait(module, node)
    assert status.startswith("Stopped by you after "), status
    assert not module._process_alive(run["pid"], "params.json")
    assert module.active_run(node) is None
    print(f"PASS: a second run into a running folder is refused; Stop ends "
          f"the run: {status[:80]}...")


def check_terminal_script(root):
    """Run in Terminal from a folder whose name has a space and an
    apostrophe, with a binary path that has a space: the script is run by
    bash exactly as Terminal would run it."""
    folder = os.path.join(root, "o'brien's runs", "work dir")
    bindir = os.path.join(root, "bin dir")
    os.makedirs(bindir)
    binary = os.path.join(bindir, "PolyFEM_bin")
    os.symlink(POLYFEM_BIN, binary)
    node = scene(folder, "terminal", binary=binary)
    module = node.hdaModule()
    launched = []
    real_popen = subprocess.Popen

    def fake_popen(command, *args, **kwargs):
        if command[:3] == ["open", "-a", "Terminal"]:
            launched.append(command)
            return real_popen(["/bin/bash", command[3]])
        return real_popen(command, *args, **kwargs)

    subprocess.Popen = fake_popen
    try:
        run = module.write_params({"node": node})
    finally:
        subprocess.Popen = real_popen
    assert run is not None and run["kind"] == "terminal"
    script = os.path.join(folder, "output", "run_polyfem.command")
    assert launched == [["open", "-a", "Terminal", script]], launched
    status = wait(module, node)
    assert status.startswith("Completed in") and "2 time steps" in status, \
        status
    output = os.path.join(folder, "output")
    assert open(os.path.join(output, ".polyfem_exit_status")).read().strip() \
        == "0"
    assert "2/2" in open(os.path.join(output, "terminal_log.txt")).read()
    assert module.log_path_for(node) == os.path.join(output,
                                                     "terminal_log.txt")
    print("PASS: Run in Terminal works with a space and an apostrophe in "
          "the paths, and is followed to the end")

    # The Windows and Linux scripts quote the same way.
    args = [binary, "-j", "params.json", "-o", "../output/"]
    bat = module.write_launch_script(
        os.path.join(folder, "input"), output, args, system="Windows")
    text = open(bat, newline="").read()
    assert f'cd /d "{os.path.join(folder, "input")}"' in text, text
    assert f'"{binary}" "-j" "params.json"' in text, text
    assert "\r\n" in text
    assert module.terminal_command(bat, system="Windows") == \
        ["cmd", "/c", "start", "", "cmd", "/k", bat]
    sh = module.write_launch_script(
        os.path.join(folder, "input"), output, args, system="Linux")
    result = subprocess.run(["/bin/bash", "-n", sh], capture_output=True)
    assert result.returncode == 0, result.stderr
    print("PASS: the Windows .bat and Linux .sh scripts quote every path")


def check_meanings(module):
    assert module.exit_status_meaning(0) == "completed"
    assert "not a crash" in module.exit_status_meaning(1)
    assert "resource limit" in module.exit_status_meaning(3)
    assert "SIGSEGV" in module.exit_status_meaning(None, 11)
    run = {"output": "/nonexistent", "started": 0.0, "ended": 75.0,
           "log": None, "kind": "background", "result": (3, None)}
    text = module.run_status_text(run)
    assert text.startswith("Stopped before the first time step was done: "
                           "exit status 3, a resource limit"), text
    run["result"] = (None, 11)
    assert module.run_status_text(run).startswith(
        "Crashed before the first time step was done: killed by SIGSEGV")
    print("PASS: exit statuses 0, 1, 3 and signals are explained")


def check_results_and_log(root, node):
    module = node.hdaModule()
    path = module.log_path_for(node)
    assert path == os.path.join(root, "background", "output", "log.txt"), path
    reader = module.open_results({"node": node})
    assert reader is not None and reader.type().name() == "readPVD::1.0"
    assert reader.evalParm("PVD_file").endswith("output/sim.pvd")
    assert len(reader.hdaModule().read_pvd(reader.evalParm("PVD_file"))) == 3
    assert reader.evalParm("color_attrib") == "von_mises_derived"
    assert reader.evalParm("color_max") > 1.0, reader.evalParm("color_max")
    again = module.open_results({"node": node})
    assert again.path() == reader.path(), "Open Results made a second node"
    print("PASS: Open Results loads the run in one Read PVD node (reused), "
          "and Open Log finds the log")


def check_hip_paths(root):
    project = os.path.join(root, "project")
    os.makedirs(project)
    hou.hipFile.clear(suppress_save_prompt=True)
    install_assets()
    hou.hipFile.save(os.path.join(project, "scene.hip"))
    node = hou.node("/obj").createNode(TYPE, "hip")
    node.parm("working_dir").set("$HIP/sim")
    node.hdaModule().working_dir_check({"node": node})
    assert node.parm("working_dir").unexpandedString() == "$HIP/sim/", \
        node.parm("working_dir").unexpandedString()
    mesh = write_box(os.path.join(project, "part.msh"))
    node.parm("file_location1").set("$HIP/part.msh")
    node.parm("file_location1").pressButton()
    staged = node.parm("file_location1").unexpandedString()
    assert staged == "$HIP/sim/input/part.msh", staged
    assert node.node("geo_1").parm("File").unexpandedString() == staged
    hou.hipFile.save()
    # Move the whole project: the scene still finds its staged mesh.
    moved = os.path.join(root, "moved project")
    shutil.move(project, moved)
    hou.hipFile.load(os.path.join(moved, "scene.hip"),
                     suppress_save_prompt=True)
    install_assets()
    node = hou.node("/obj/hip")
    expected = os.path.join(moved, "sim", "input", "part.msh")
    assert node.evalParm("file_location1") == expected, \
        node.evalParm("file_location1")
    reader = node.node("geo_1")
    reader.cook(force=True)
    assert len(reader.geometry().prims()) > 0
    assert os.path.isfile(mesh) is False   # it moved with the project
    print("PASS: '$HIP/sim' stays relative, and a moved project keeps its "
          "staged mesh")


def install_assets():
    # force: a cleared or loaded scene would otherwise fall back to whichever
    # definition of these types Houdini prefers (e.g. one in the user otls)
    for name in ("sop_MSH_Reader.3.0.hdanc",
                 "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                 "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, name), force_use_assets=True)


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    install_assets()
    root = tempfile.mkdtemp(prefix="polyfem_run_feedback_")
    node = check_background(root)
    check_meanings(node.hdaModule())
    check_results_and_log(root, node)
    check_stop_and_refusal(root)
    check_terminal_script(root)
    check_hip_paths(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
