"""Remove Duplicate Contacts (Improved Max): export, refusal, import, run.

The toggle turns on IPC's improved max operator under Semi-Implicit
stiffness (PolyFEM's signed parent law, sdast9/polyfem a54509955; record
polyfem/docs/improved-max-signed-parents-20261018.md). Semi-Implicit exports
write every convergent switch with the physical barrier off; Adaptive and
Fixed keep the historical export (use_convergent_formulation alone).

Runs the real binary (POLYFEM_BIN, default ../polyfem/build/PolyFEM_bin); it
must be a54509955 or later, an older one refuses the operator at start-up.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import hou

from test_polyfem_hda import BASE, POLYFEM_BIN, ROOT, make_cube_msh

CONVERGENT_KEYS = ("use_area_weighting", "use_improved_max_operator",
                   "use_physical_barrier")


def exported_contact(mod, node):
    with open(mod.write_params_only({"node": node})) as handle:
        data = json.load(handle)
    return data, data["contact"]


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_improved_max_")
    os.makedirs(os.path.join(work, "input"), exist_ok=True)
    cube = os.path.join(work, "cube.msh")
    make_cube_msh(cube)
    slab = os.path.join(work, "slab.obj")
    shutil.copy(os.path.join(
        ROOT, "polyfem", "scenes", "semi-implicit", "slab.obj"), slab)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "improved_max_test")
    mod = node.hdaModule()
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN})
    node.setParms({"file_location1": cube})
    node.parm("file_location1").pressButton()
    node.setParms({"materials1_1": 3, "E1_1": 1e6, "nu1_1": 0.45,
                   "rho1_1": 1000})
    node.setParms({"sideset_selection1_1": 1})
    node.parm("sideset_selection1_1").pressButton()
    node.setParms({"basegroup1_1_1": "axis:+z:0.99", "grouptype1_1_1": 0,
                   "Boundary_Condition__1_1_1": 1})
    node.setParms({"boundary_type1_1_1_1": 0,
                   "vector_1_1_1_1": '["0", "0", "-0.25*t"]',
                   "x_dimension1_1_1_1": 1, "y_dimension1_1_1_1": 1,
                   "z_dimension1_1_1_1": 1})
    node.setParms({"geo_int": 2, "num_geos": 2})
    node.setParms({"is_obstacle2": 1, "file_location2": slab})
    node.parm("file_location2").pressButton()
    node.setParms({"obstacle_disp2": "[0, 0, 0]"})
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 0.5,
                   "time_inc_bool": 1, "dt": 0.125})
    node.setParms({"enable": 1, "dhat": 1e-3})

    # --- the control: visible, off by default, Semi-Implicit only ---------
    parm = node.parm("improved_max")
    assert parm is not None
    assert not parm.parmTemplate().isHidden()
    assert node.evalParm("improved_max") == 0
    folder = node.parmTemplateGroup().containingFolder("improved_max")
    assert folder.label() == node.parmTemplateGroup().containingFolder(
        "area_weighted").label(), folder.label()
    help_text = parm.parmTemplate().help()
    assert "Semi-Implicit mode only, experimental" in help_text
    assert "Parent Candidate" in help_text
    conditions = parm.parmTemplate().conditionals()
    disable = conditions.get(hou.parmCondType.DisableWhen, "")
    assert "barrier_mode != semi_implicit" in disable, disable

    # --- exports -----------------------------------------------------------
    # Nothing set: unchanged (no convergent switches beyond the historical one)
    _, contact = exported_contact(mod, node)
    assert contact["use_convergent_formulation"] is False, contact
    assert not set(CONVERGENT_KEYS) & set(contact), contact

    # Semi-Implicit + Remove Duplicate Contacts
    node.setParms({"improved_max": 1})
    data, contact = exported_contact(mod, node)
    assert data["solver"]["contact"]["barrier_stiffness"] == "semi_implicit"
    assert {k: contact[k] for k in ("use_convergent_formulation",)
            + CONVERGENT_KEYS} == {
        "use_convergent_formulation": True, "use_area_weighting": False,
        "use_improved_max_operator": True,
        "use_physical_barrier": False}, contact

    # ... with Area Weighted too
    node.setParms({"area_weighted": 1})
    _, contact = exported_contact(mod, node)
    assert contact["use_area_weighting"] is True, contact
    assert contact["use_improved_max_operator"] is True, contact
    assert contact["use_physical_barrier"] is False, contact

    # Area Weighted alone under Semi-Implicit: area weighting only, which
    # PolyFEM accepts (the full convergent defaults were refused there)
    node.setParms({"improved_max": 0})
    _, contact = exported_contact(mod, node)
    assert contact["use_convergent_formulation"] is True, contact
    assert contact["use_area_weighting"] is True, contact
    assert contact["use_improved_max_operator"] is False, contact
    assert contact["use_physical_barrier"] is False, contact

    # Adaptive / Fixed: the historical export, the toggle ignored
    for mode in (1, 2):
        node.setParms({"barrier_mode": mode, "improved_max": 1,
                       "area_weighted": 1})
        _, contact = exported_contact(mod, node)
        assert contact["use_convergent_formulation"] is True, contact
        assert not set(CONVERGENT_KEYS) & set(contact), (mode, contact)
        node.setParms({"area_weighted": 0})
        _, contact = exported_contact(mod, node)
        assert contact["use_convergent_formulation"] is False, contact
        assert not set(CONVERGENT_KEYS) & set(contact), (mode, contact)
    node.setParms({"barrier_mode": 0, "area_weighted": 0, "improved_max": 1})

    # --- the stencil identity is refused by name, at export and in Check Setup
    identity = node.parm("si_coefficient_identity")
    tokens = identity.parmTemplate().menuItems()
    node.setParms({"si_coefficient_identity": tokens.index("stencil")})
    # The Write button stops at Check Setup's error (nothing written) ...
    assert not mod.write_params_only({"node": node}), \
        "stencil identity with improved max was written"
    # ... and the exporter itself refuses by name
    try:
        mod.build_params(node)
    except hou.Error as exc:
        assert "needs Coefficient Identity = Parent Candidate" in str(exc), exc
    else:
        raise AssertionError("stencil identity with improved max was exported")
    report = mod.check_setup(node, purpose="write")
    assert any("needs Coefficient Identity = Parent Candidate" in e
               for e in report["errors"]), report
    node.setParms({"si_coefficient_identity": tokens.index("parent")})
    report = mod.check_setup(node, purpose="write")
    assert not any("Coefficient Identity" in e for e in report["errors"]), report
    assert any("Remove Duplicate Contacts (Improved Max) is on" in n
               for n in report["notes"]), report

    # --- import round trip ---------------------------------------------------
    node.setParms({"area_weighted": 1})
    params_path = mod.write_params_only({"node": node})
    restored = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "improved_max_restored")
    restored.setParms({"old_input_dir": os.path.dirname(params_path)})
    restored.hdaModule().read_params({"node": restored})
    assert restored.evalParm("improved_max") == 1
    assert restored.evalParm("area_weighted") == 1
    assert restored.evalParm("barrier_mode") == 0
    restored.destroy()
    node.setParms({"area_weighted": 0})
    params_path = mod.write_params_only({"node": node})

    # --- the real binary accepts and runs it -------------------------------
    input_dir = os.path.dirname(params_path)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info", "--max_threads", "1"],
        cwd=input_dir, capture_output=True, text=True, timeout=900)
    sys.stdout.write(result.stdout[-1500:])
    sys.stderr.write(result.stderr[-1500:])
    assert "supports the improved max operator" not in result.stdout + \
        result.stderr, ("this PolyFEM predates the signed parent law "
                        "(sdast9/polyfem a54509955); set POLYFEM_BIN")
    assert result.returncode == 0, "PolyFEM run failed"
    with open(os.path.join(work, "output", "run-manifest.json")) as handle:
        manifest = json.load(handle)
    assert manifest["completion"]["status"] == "completed", \
        manifest["completion"]
    compact = json.dumps(manifest, separators=(",", ":"))
    assert '"improved_max_operator":true' in compact, \
        "the run manifest does not report the improved max operator"

    print("test_improved_max: all checks passed")


if __name__ == "__main__":
    main()
