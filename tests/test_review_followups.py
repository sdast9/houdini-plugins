"""PolyFEM node: follow-ups of the 2026-11-14 review.

* The Time tab counts the steps as PolyFEM does since sdast9/polyfem's
  2026-11-14 fix: End Time 0.14 with dt 0.02 is 7 steps (ceil of the
  floating-point ratio 7.000000000000001 made it 8, past End Time).
* Import sets Max Iterations (AL Phase): a file without its own AL budget
  used to leave the node's earlier value in place.
* Import keeps fiber settings as entered (a symmetric +/-theta family, a
  cylindrical source) when they reproduce the run's material exactly, from
  the scene record; it used to bring them back as two file-based families.
  When they no longer reproduce it, the file-based families stay.

Save Log with Run in Background is covered in tests/test_run_feedback.py.

Run: hython tests/test_review_followups.py
No gmsh and no PolyFEM binary needed.
"""

import json
import os
import tempfile

import hou

from test_check_setup import write_box

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def install():
    for name in ("sop_MSH_Reader.3.0.hdanc",
                 "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, name), force_use_assets=True)


def scene(root, label):
    work = os.path.join(root, label)
    os.makedirs(work)
    mesh = write_box(os.path.join(root, f"{label}_box.msh"))
    node = hou.node("/obj").createNode(TYPE, label)
    node.setParms({"working_dir": work + "/"})
    node.setParms({"file_location1": mesh})
    node.parm("file_location1").pressButton()
    node.setParms({"enable": 0, "quasistatic": 1})
    return node, node.hdaModule(), work


def import_into(work, label):
    node = hou.node("/obj").createNode(TYPE, label)
    node.setParms({"old_input_dir": work})
    node.hdaModule().read_params({"node": node})
    return node


def params(node):
    with open(node.hdaModule().build_params(node)) as handle:
        data = json.load(handle)
    data.pop("provenance", None)
    return data


def check_step_count(root):
    node, mod, _ = scene(root, "steps")
    for tend, dt, steps in ((0.14, 0.02, 7), (4.35, 0.05, 87), (3.3, 1.1, 3),
                            (1.0, 0.1, 10), (1.05, 0.1, 11)):
        node.setParms({"end_time_bool": 1, "time_inc_bool": 1,
                       "num_timesteps_bool": 0, "t0": 0.0, "tend": tend,
                       "dt": dt})
        assert mod.simulation_steps(node)[2] == steps, \
            (tend, dt, mod.simulation_steps(node))
        assert f"{steps} time steps" in mod.timeline_text(node), \
            mod.timeline_text(node)
    print("PASS: End Time / dt a whole number up to roundoff gives that many "
          "steps (0.14 / 0.02 = 7), as PolyFEM counts them")


def check_al_iterations(root):
    source, _, work = scene(root, "al_source")
    source.setParms({"max_iterations": 300, "al_max_iterations": 300})
    written = params(source)
    assert "nonlinear" not in written["solver"]["augmented_lagrangian"]
    target = hou.node("/obj").createNode(TYPE, "al_target")
    target.setParms({"al_max_iterations": 77, "old_input_dir": work})
    target.hdaModule().read_params({"node": target})
    assert target.evalParm("al_max_iterations") == 300, \
        target.evalParm("al_max_iterations")
    print("PASS: Import sets Max Iterations (AL Phase) also when the file "
          "has no AL budget of its own")


def check_adhesion_slip_default(root):
    """Adhesion Slip Velocity defaulted to 1.0 against PolyFEM's 0.001 (and
    epsv's 0.001); user decision 2026-11-15: 0.001."""
    node, _, _ = scene(root, "adhesion")
    template = node.parm("adhesion_epsa").parmTemplate()
    assert template.defaultValue()[0] == 0.001, template.defaultValue()
    node.setParms({"enable": 1, "adhesion_enable": 1})
    written = params(node)["contact"]["adhesion"]
    assert written["epsa"] == 0.001, written
    print("PASS: Adhesion Slip Velocity defaults to 0.001, PolyFEM's default")


def fiber_scene(root, label):
    node, mod, work = scene(root, label)
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("MaterialSum"),
                   "matrix_model1_1": "NeoHookean",
                   "num_fiber_families1_1": 2})
    node.setParms({"fam_model1_1_1": "HGODispersion",
                   "fam_fib_source1_1_1": "cylindrical",
                   "fam_fib_component1_1_1": "circumferential",
                   "fam_mirror1_1_1": 1, "fam_theta1_1_1": 30,
                   "fam_axis_source1_1_1": "cylindrical",
                   "fam_model1_1_2": "HGOFiber",
                   "fam_fib_source1_1_2": "constant"})
    node.parmTuple("fam_fib_axis_origin1_1_1").set((0.5, 0.0, 0.5))
    node.parmTuple("fam_fib_axis_dir1_1_1").set((0, 1, 0))
    node.parmTuple("fam_fib_dir1_1_2").set((0, 1, 0))
    return node, mod, work


SETTINGS = ("num_fiber_families1_1", "fam_mirror1_1_1", "fam_theta1_1_1",
            "fam_fib_source1_1_1", "fam_axis_source1_1_1",
            "fam_fib_component1_1_1", "fam_fib_source1_1_2")


def settings(node):
    return {name: (node.parm(name).evalAsString()
                   if node.parm(name).parmTemplate().type()
                   == hou.parmTemplateType.Menu else node.parm(name).eval())
            for name in SETTINGS}


def check_fiber_settings(root):
    node, _, work = fiber_scene(root, "fibers")
    first = params(node)
    imported = import_into(work, "fibers_import")
    assert settings(imported) == settings(node), \
        (settings(imported), settings(node))
    second = params(imported)
    assert second["materials"] == first["materials"]
    print("PASS: Import keeps a symmetric +/-theta family and a cylindrical "
          "source as entered, and writes the same material")

    # A recorded setting that no longer reproduces the run (the axis was
    # changed after the run was written): the file-based import stays.
    path = os.path.join(work, "input", "hda_scene.json")
    record = json.load(open(path))
    entry = record["materials"]["1001"]
    entry["parms"]["fam_fib_axis_dir1_1_1y"] = 0.0
    entry["parms"]["fam_fib_axis_dir1_1_1z"] = 1.0
    json.dump(record, open(path, "w"))
    fallback = import_into(work, "fibers_fallback")
    assert fallback.evalParm("num_fiber_families1_1") == 3, \
        fallback.evalParm("num_fiber_families1_1")
    assert fallback.parm("fam_fib_source1_1_1").evalAsString() == "file"
    report = fallback.evalParm("import_report")
    assert "no longer reproduce" in report, report
    assert params(fallback)["materials"][0]["models"][1]["fiber_direction"] \
        ["type"] == "per_element_file"
    print("PASS: recorded fiber settings that no longer reproduce the run "
          "are not used; the families import as per-element files, with a "
          "note")


def main():
    install()
    root = tempfile.mkdtemp(prefix="polyfem_review_followups_")
    check_step_count(root)
    check_al_iterations(root)
    check_adhesion_slip_default(root)
    check_fiber_settings(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
