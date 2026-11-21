"""Read PVD review of 2026-11-14: seven defects, each reproduced before the
fix (hda-review-work/scripts/t_readpvd_review_20261114*.py).

1. Refresh kept cached frames of a run rewritten into the same folder.
2. A frame past the last step of a running simulation kept the clamped step
   after Refresh.
3. Legend ticks were linear on the logarithmic color scale.
4. The probe stored a point number of the clipped display, which a clip
   renumbers as the deformed body moves through the plane.
5. Reference Frame (and Topology Frame) were used as step numbers although
   they are Houdini frames.
6. Principal values of a non-symmetric tensor (F) differed between the
   viewport (VEX) and Auto Range / export (numpy).
7. The per-element sheet with Step on Screen outside the exported steps had
   a header and no values.

Run: hython tests/test_readpvd_review_fixes.py
Uses ../smoke-out/quasistatic-semi (run-smoke.sh) and synthetic files.
"""

import os
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
SMOKE_ROOT = os.environ.get("POLYFEM_TEST_SMOKE_ROOT", os.path.join(ROOT, "smoke-out"))
SMOKE = os.path.join(SMOKE_ROOT, "quasistatic-semi",
                     "quasistatic-semi.pvd")
TET = "0 0 0  1 0 0  0 1 0  0 0 1"


def write_vtu(folder, name, value):
    with open(os.path.join(folder, name), "w") as handle:
        handle.write(
            '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
            '<Piece NumberOfPoints="4" NumberOfCells="1"><Points>'
            '<DataArray type="Float64" NumberOfComponents="3" '
            f'format="ascii">{TET}</DataArray></Points><Cells>'
            '<DataArray type="Int64" Name="connectivity" format="ascii">'
            '0 1 2 3</DataArray><DataArray type="Int64" Name="offsets" '
            'format="ascii">4</DataArray><DataArray type="UInt8" '
            'Name="types" format="ascii">10</DataArray></Cells>'
            '<PointData><DataArray type="Float64" Name="solution" '
            'NumberOfComponents="3" format="ascii">'
            + f"{value} 0 0  " * 4 + '</DataArray></PointData>'
            '</Piece></UnstructuredGrid></VTKFile>')


def write_run(folder, values):
    for step, value in enumerate(values):
        write_vtu(folder, f"g{step}.vtu", value)
    with open(os.path.join(folder, "run.pvd"), "w") as handle:
        handle.write('<VTKFile type="Collection"><Collection>'
                     + "".join(f'<DataSet timestep="{i}" file="g{i}.vtu"/>'
                               for i in range(len(values)))
                     + '</Collection></VTKFile>')
    return os.path.join(folder, "run.pvd")


def shown(node):
    output = node.node("output")
    output.cook()
    return float(np.frombuffer(output.geometry().pointFloatAttribValuesAsString(
        "solution"), dtype=np.float32)[0])


def load(name, pvd, mapping="steps"):
    node = hou.node("/obj").createNode("readPVD::1.0", name)
    node.setParms({"PVD_file": pvd, "time_mapping": mapping})
    node.hdaModule().start({"node": node})
    return node, node.hdaModule()


def check_rerun_and_growing(root):
    folder = os.path.join(root, "rerun")
    os.makedirs(folder)
    pvd = write_run(folder, [0.0, 1.0, 2.0, 3.0])
    node, module = load("rerun", pvd)
    assert node.evalParm("cache") == 1
    before = []
    for frame in range(4):
        hou.setFrame(frame)
        before.append(shown(node))
    assert before == [0.0, 1.0, 2.0, 3.0], before
    # an unchanged Refresh keeps the cooked frames
    assert module.stale_cooked_frames(node) == []
    time.sleep(1.1)
    write_run(folder, [100.0, 101.0, 102.0, 103.0])  # a rerun, same names
    hou.setFrame(2)
    assert module.stale_cooked_frames(node), "rewritten files not noticed"
    module.refresh({"node": node})
    assert hou.frame() == 2, "Refresh moved the frame"
    after = []
    for frame in range(4):
        hou.setFrame(frame)
        after.append(shown(node))
    assert after == [100.0, 101.0, 102.0, 103.0], (
        f"Refresh after a rerun still shows {after}")
    node.destroy()
    print("PASS 1: Refresh after a rerun into the same folder shows the new "
          "files on every frame (it showed the old cached frames)")

    folder = os.path.join(root, "growing")
    os.makedirs(folder)
    pvd = write_run(folder, [0.0, 1.0])
    node, module = load("growing", pvd)
    hou.setFrame(0)
    shown(node)
    hou.setFrame(3)
    assert shown(node) == 1.0          # past the end: the last step
    write_run(folder, [0.0, 1.0, 2.0, 3.0, 4.0])
    module.refresh({"node": node})
    assert shown(node) == 3.0, "frame 3 kept the clamped step 1"
    hou.setFrame(0)
    assert shown(node) == 0.0
    # a step that is not rewritten and still shown leaves the cache alone
    assert module.stale_cooked_frames(node) == []
    node.destroy()
    print("PASS 2: a frame shown past the end of a running simulation "
          "shows its step after Refresh (it kept the clamped step)")


def check_log_legend():
    node = hou.node("/obj").createNode("readPVD::1.0", "legend")
    module = node.hdaModule()
    node.setParms({"color_scale": "log10", "color_min": 1.0,
                   "color_max": 100.0, "legend_ticks": 3})
    assert np.allclose(module.legend_values(node), [100.0, 10.0, 1.0])
    assert np.allclose(module.legend_tick_values(node), [1.0, 10.0, 100.0])
    # a non-positive range cannot be shown on the log scale: linear labels
    node.setParms({"color_min": 0.0})
    assert np.allclose(module.legend_values(node), [100.0, 50.0, 0.0])
    node.setParms({"color_scale": "linear", "color_min": 1.0})
    assert np.allclose(module.legend_values(node), [100.0, 50.5, 1.0])
    legend = node.node("legend_bar")
    legend.cook(force=True)
    assert not legend.errors(), legend.errors()
    node.destroy()
    print("PASS 3: legend ticks on the log scale are 1, 10, 100 (were 1, "
          "50.5, 100); linear unchanged")


def check_probe_under_clip():
    node, module = load("probe", SMOKE)
    last = len(module.read_pvd(SMOKE)) - 1
    output = node.node("output")
    node.setParms({"clip_mode": 1, "show_deformed": 1, "clip_fill": 0,
                   "clip_originx": 0.5, "clip_originy": 0.5,
                   "clip_originz": 0.5, "clip_directionx": 1.0,
                   "clip_directiony": 0.0, "clip_directionz": 0.0,
                   "probe_enabled": 1})
    hou.setFrame(0)
    output.cook(force=True)
    geo = output.geometry()
    rest = np.frombuffer(geo.pointFloatAttribValuesAsString("rest"),
                         dtype=np.float32).reshape(-1, 3)
    # a model vertex of the kept half, away from the plane
    candidates = np.flatnonzero(np.abs(rest[:, 0] - 0.5) > 0.2)
    assert len(candidates), "the clip kept no vertex away from the plane"
    picked = int(candidates[len(candidates) // 2])
    want = rest[picked].astype(np.float64)
    module.probe_pick(node, picked)
    point_id = node.evalParm("probe_point")
    for frame in (0, last):
        hou.setFrame(frame)
        output.cook(force=True)
        position, text = module.probe_readout(node, point_id)
        assert position is not None and f"Point {point_id}" in text
        result = node.node("OUT_result").geometry()
        row = module._result_row(result, point_id)
        got = np.asarray(result.point(row).attribValue("rest"))
        assert np.allclose(got, want, atol=1e-6), (frame, got, want)
        assert np.allclose(position, result.point(row).position())
    # a cut point of the clip snaps to the nearest model vertex
    hou.setFrame(0)
    output.cook(force=True)
    geo = output.geometry()
    rest = np.frombuffer(geo.pointFloatAttribValuesAsString("rest"),
                         dtype=np.float32).reshape(-1, 3)
    ids = np.frombuffer(geo.pointIntAttribValuesAsString("readpvd_point"),
                        dtype=np.int32)
    result = node.node("OUT_result").geometry()
    result_rest = np.frombuffer(result.pointFloatAttribValuesAsString(
        "rest"), dtype=np.float32).reshape(-1, 3)
    cut = [i for i in range(len(rest)) if ids[i] >= len(result_rest)
           or not np.allclose(result_rest[ids[i]], rest[i], atol=1e-6)]
    assert cut, "the clip made no cut points"
    module.probe_pick(node, cut[0])
    snapped = node.evalParm("probe_point")
    distance = np.linalg.norm(result_rest - rest[cut[0]], axis=1)
    assert np.isclose(distance[snapped], distance.min(), atol=1e-6)
    # without a clip the probe number is the displayed point number, as before
    node.setParms({"clip_mode": 0})
    output.cook(force=True)
    position, _ = module.probe_readout(node, 770)
    assert np.allclose(position, output.geometry().point(770).position())
    node.destroy()
    print("PASS 4: the probe follows its vertex while a clip renumbers the "
          "displayed points (it jumped to another vertex); a cut point snaps "
          "to the nearest vertex")


def check_frame_parameters(root):
    folder = os.path.join(root, "mapping")
    os.makedirs(folder)
    pvd = write_run(folder, [0.0, 1.0, 2.0, 3.0])
    node, module = load("mapping", pvd, mapping="time")
    frame = module.entry_frame(node, 1)
    assert frame > 1, "the mapping is one frame per step"
    hou.setFrame(frame)
    assert module.entry_index(node) == 1
    node.setParms({"color_attrib": "solution", "color_reduction": "x",
                   "reference_enable": 1, "reference_frame": int(frame)})
    result = node.node("OUT_result")
    result.cook(force=True)
    delta = np.frombuffer(result.geometry().pointFloatAttribValuesAsString(
        "reference_comparison"), dtype=np.float32)
    assert np.allclose(delta, 0.0), delta
    status = node.evalParm("reference_status")
    assert status.startswith("Comparing") and "step 1" in status, status
    # frame 0 is before the run's first step: step 0
    assert module.frame_step(node, 0) == 0
    # the all-frames scan compares with the same step
    node.setParms({"reference_frame": 0})
    values = {frame_index: vals for frame_index, _, vals in
              module._scan_frames(node, pvd)}
    assert np.allclose(values[3], 3.0), values[3]
    # Topology Frame goes through the mapping too
    node.setParms({"reference_enable": 0, "topo_frame": int(frame)})
    topo = node.node("topo_build")
    assert topo.evalParm("topoframe") == 1
    node.destroy()
    print("PASS 5: Reference Frame and Topology Frame are Houdini frames on "
          "the Simulation Time mapping (were step numbers)")


def check_nonsymmetric_principal():
    node, module = load("principal", SMOKE)
    last = len(module.read_pvd(SMOKE)) - 1
    hou.setFrame(last)
    mesh = module._block_for(node, module.load_frame(SMOKE, last))
    for field in ("F_mat", "pk1", "cauchy_mat"):
        for reduction in ("principal_max", "principal_middle",
                          "principal_min"):
            node.setParms({"derived": 1, "color_attrib": field,
                           "color_reduction": reduction, "smooth_field": 0})
            result = node.node("OUT_result")
            result.cook(force=True)
            viewport = np.frombuffer(
                result.geometry().pointFloatAttribValuesAsString("for_color"),
                dtype=np.float32).astype(np.float64)
            mirror = module._scan_reduced(node, mesh, field, reduction)
            scale = np.abs(mirror).max()
            worst = np.abs(viewport - mirror).max() / scale
            assert worst < 1e-5, (field, reduction, worst)
    node.destroy()
    print("PASS 6: principal values of F, P and sigma agree between the "
          "viewport and Auto Range / export (F differed by 8 %)")


def check_per_item_screen_step(root):
    node, module = load("per_item", SMOKE)
    hou.setFrame(0)
    target = os.path.join(root, "per_item.xlsx")
    node.setParms({"export_step_first": 2, "export_step_last": -1,
                   "export_per_item": 1, "export_per_item_steps": "screen",
                   "export_data_file": target, "export_forces": 0})
    summary = module.export_data(node)
    assert summary["steps"] == [2, 3, 4], summary["steps"]
    assert [row[0] for row in summary["rows"]] == [2, 3, 4]
    with zipfile.ZipFile(target) as archive:
        sheet = archive.read("xl/worksheets/sheet2.xml").decode()
    rows = sheet.split("<row")[1:]
    header = re.findall(r"<t[^>]*>([^<]*)</t>", rows[0])
    assert header[-1].startswith("Step 0"), header
    first = re.findall(r'<c r="([A-Z]+)\d+"[^>]*>(?:<v>([^<]*)</v>)?', rows[1])
    column = dict(first)
    assert column.get("H"), f"Step 0 column is empty: {rows[1][:300]}"
    # step 0 is the undeformed start: zero stress
    assert abs(float(column["H"])) < 1e-6
    node.destroy()
    print("PASS 7: the per-element sheet fills a Step on Screen outside the "
          "exported steps (it had a header and no values)")


def main():
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"),
                        force_use_assets=True)
    hou.setFps(24)
    assert os.path.isfile(SMOKE), f"missing {SMOKE} (run run-smoke.sh)"
    root = tempfile.mkdtemp(prefix="readpvd_review_")
    try:
        check_rerun_and_growing(root)
        check_log_legend()
        check_probe_under_clip()
        check_frame_parameters(root)
        check_nonsymmetric_principal()
        check_per_item_screen_step(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL PASS")


main()
