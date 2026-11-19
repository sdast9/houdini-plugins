"""Regression tests for Patch 2 fixes across PolyFEM 2.0, Read PVD 1.0,
MSH Reader 3.0, and src/common/ (2026-11-19 review).

Run: hython tests/test_patch2_fixes.py
Needs no PolyFEM binary and no gmsh.
"""

import base64
import json
import os
import shutil
import struct
import sys
import tempfile
import zlib

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src", "common"))
import msh_parser  # noqa: E402
import vtu_parser  # noqa: E402


def _write_tet_msh(path, empty=False):
    """Minimal Gmsh 2.2 ASCII tet mesh (or 0-element mesh)."""
    if empty:
        text = (
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n4\n"
            "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
            "$EndNodes\n"
            "$Elements\n0\n$EndElements\n"
        )
    else:
        text = (
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n4\n"
            "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
            "$EndNodes\n"
            "$Elements\n1\n"
            "1 4 2 1 1 1 2 3 4\n"
            "$EndElements\n"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_polyfem_batch_2a(root):
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True,
    )
    work = os.path.join(root, "pf_work")
    os.makedirs(work, exist_ok=True)
    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "pf_patch2")
    mod = node.hdaModule()
    node.parm("working_dir").set(work + "/")
    mod.working_dir_check({"node": node})

    # 1. DialogScript UI / label / range / default fixes
    node.updateParmStates()
    ptg = node.parmTemplateGroup()
    assert ptg.findFolder(("Contact", "Barrier (IPC)")) is not None
    assert ptg.findFolder(("Contact", "Barrier (ICP)")) is None
    assert node.parm("min_edge").isDisabled()
    nt_tmpl = node.parm("num_timesteps").parmTemplate()
    assert nt_tmpl.type() == hou.parmTemplateType.Int
    assert nt_tmpl.minValue() == 1
    assert node.parm("lower").parmTemplate().minValue() > 0
    assert node.evalParm("swap_max_depth2") == 3

    node.parm("quasistatic").set(1)
    node.updateParmStates()
    assert node.parm("integrator").isDisabled()
    node.parm("quasistatic").set(0)

    node.parm("minimal_fields").set(1)
    node.parm("materials_fields").set(1)
    node.updateParmStates()
    assert node.parm("minimal_fields").isDisabled()
    assert not node.parm("scalar_values").isDisabled()
    assert not node.parm("tensor_values").isDisabled()
    node.parm("materials_fields").set(0)
    node.parm("minimal_fields").set(0)

    node.parm("solver").set("AMGCL")
    node.updateParmStates()
    assert node.parm("max_iter").isVisible()
    assert not node.parm("max_iter").isDisabled()

    # Output > Advanced labels are readable (not raw snake_case)
    for pname in ("sol_on_grid", "compute_error", "sol_at_node",
                  "vis_boundary_only", "curved_mesh_size", "spectrum"):
        label = node.parm(pname).parmTemplate().label()
        assert "_" not in label, (pname, label)

    # 2. ViewerStateModule checks (source level for headless hython)
    vsm = node.type().definition().sections()["ViewerStateModule"].contents()
    assert "showSelectionBar(True)" in vsm
    assert "PolyFEM::2.0" in vsm
    assert "worldTransform().inverted()" in vsm
    assert 'hou.undos.group("PolyFEM Transform Handle")' in vsm
    assert "hou.uiEventReason.Picked" in vsm

    # 3. Load a valid tet mesh, test reload safety & 0-element guard
    good_msh = _write_tet_msh(os.path.join(work, "good.msh"))
    empty_msh = _write_tet_msh(os.path.join(work, "empty.msh"), empty=True)
    node.parm("file_location1").set(good_msh)
    mod.create_geo_nodes({"node": node, "script_multiparm_index": "1"})
    assert node.node("geo_1") is not None
    staged_good = node.parm("file_location1").unexpandedString()

    # Missing file preserves existing geo_1 and restores file_location1
    node.parm("file_location1").set(os.path.join(work, "does_not_exist.msh"))
    mod.create_geo_nodes({"node": node, "script_multiparm_index": "1"})
    assert node.node("geo_1") is not None
    assert node.parm("file_location1").unexpandedString() == staged_good

    # 0-element mesh preserves existing geo_1 and restores file_location1
    node.parm("file_location1").set(empty_msh)
    mod.create_geo_nodes({"node": node, "script_multiparm_index": "1"})
    assert node.node("geo_1") is not None
    assert node.parm("file_location1").unexpandedString() == staged_good

    # 4. reset_transform removes keyframes
    tx = node.parm("xform_t__1x")
    for frame, val in ((1, 5.0), (10, 15.0)):
        kf = hou.Keyframe()
        kf.setFrame(frame)
        kf.setValue(val)
        tx.setKeyframe(kf)
    assert len(tx.keyframes()) == 2 and tx.isTimeDependent()
    mod.reset_transform({"node": node, "script_multiparm_index": "1"})
    assert len(tx.keyframes()) == 0 and not tx.isTimeDependent()
    assert tx.eval() == 0.0

    # 5. create_group on obstacle resets sideset_selection without orphan node
    node.parm("is_obstacle1").set(1)
    node.parm("sideset_selection1_1").set(1)
    mod.create_group({"node": node, "script_multiparm_index": "1",
                      "script_multiparm_index2": "1"})
    assert node.evalParm("sideset_selection1_1") == 0
    assert node.node("group_1_1_1") is None
    node.parm("is_obstacle1").set(0)

    # 6. simulation_steps requires >= 2 enabled toggles
    node.setParms({"end_time_bool": 1, "time_inc_bool": 0,
                   "num_timesteps_bool": 0})
    try:
        mod.simulation_steps(node)
        raise AssertionError("expected NodeError when <2 time toggles enabled")
    except hou.NodeError:
        pass
    node.setParms({"end_time_bool": 1, "time_inc_bool": 1,
                   "num_timesteps_bool": 0})

    # 7. resolve_sidesets rejects >99 sidesets
    node.parm("sideset_selection1_1").set(100)
    try:
        mod.resolve_sidesets(node, 1, 1)
        raise AssertionError("expected NodeError when sidesets > 99")
    except hou.NodeError as exc:
        assert "99" in str(exc)
    node.parm("sideset_selection1_1").set(0)

    # 8. _read_element_file strips # comments with commas & catches bad values
    elem_txt = os.path.join(work, "elems.txt")
    with open(elem_txt, "w", encoding="utf-8") as f:
        f.write("# header, with, commas\n1.0 2.0 3.0\n4.0 5.0 6.0\n")
    vals = mod._read_element_file(elem_txt, 2, 3, "", "test")
    assert vals.shape == (2, 3)
    bad_txt = os.path.join(work, "bad_elems.txt")
    with open(bad_txt, "w", encoding="utf-8") as f:
        f.write("not_a_number 2.0 3.0\n")
    try:
        mod._read_element_file(bad_txt, 1, 3, "", "test")
        raise AssertionError("expected NodeError on invalid element file")
    except hou.NodeError:
        pass

    # 9. build_output gates contact-dependent fields when enable == 0
    node.setParms({
        "enable": 0,
        "contact_forces_fields": 1,
        "friction_forces_fields": 1,
        "normal_adhesion_forces_fields": 1,
        "tangential_adhesion_forces_fields": 1,
        "adaptive_dhat": 1,
    })
    data = {}
    mod.build_output(node, data)
    opts = data["output"]["paraview"]["options"]
    for key in ("contact_forces", "friction_forces",
                "normal_adhesion_forces", "tangential_adhesion_forces",
                "adaptive_dhat"):
        assert opts.get(key) is not True, (key, opts)

    # 10. check_setup catches invalid trim band, Wolfe c2, negative t0,
    #     and warns on custom resource limits with BVH
    node.setParms({
        "enable": 1, "barrier_mode": 0, "gcp_enable": 0,
        "si_trim_lower": 0.9, "si_trim_upper": 0.5,
        "broad_phase": 3, "resource_limits": 2,
        "method": "Wolfe", "armijo_c": 0.5, "wolfe_c2": 0.4,
        "t0": -1.0,
    })
    report = mod.check_setup(node)
    errs, warns = report["errors"], report["warnings"]
    joined_errs = "\n".join(errs)
    joined_warns = "\n".join(warns)
    assert "Trim Lower Band" in joined_errs, errs
    assert "Strong Wolfe" in joined_errs, errs
    assert node.evalParm("t0") >= 0.0
    assert "Custom Contact Resource Limits" in joined_warns, warns
    # Reset back to valid defaults
    node.setParms({
        "si_trim_lower": 0.2, "si_trim_upper": 0.8,
        "resource_limits": 0, "method": "RobustArmijo",
        "armijo_c": 1e-4, "wolfe_c2": 0.9, "t0": 0.0,
    })

    # 11. _expert_notes: Rayleigh Damping detected; AMGCL ignores Hypre parms;
    #     BFGS ignores linear solver parms
    node.setParms({"expert_mode": 0, "num_rayleigh": 1,
                   "solver_nl": "Newton", "solver": "AMGCL",
                   "max_iter_hypre": 99})
    notes = "\n".join(mod._expert_notes(node))
    assert "Rayleigh Damping" in notes, notes
    assert "Hypre" not in notes, notes

    node.setParms({"num_rayleigh": 0, "solver_nl": "BFGS",
                   "solver": "AMGCL", "solver_type": "cg"})
    notes_bfgs = "\n".join(mod._expert_notes(node))
    assert "AMGCL" not in notes_bfgs, notes_bfgs

    # 12. _clear_run_files removes logs; active_run / reattach_run handle
    #     finished entries cleanly
    out_dir = os.path.join(work, "output")
    os.makedirs(out_dir, exist_ok=True)
    for fname in ("log.txt", mod._CONSOLE_LOG, mod._TERMINAL_LOG,
                  mod._EXIT_STATUS_FILE):
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("old\n")
    mod._clear_run_files(out_dir)
    for fname in ("log.txt", mod._CONSOLE_LOG, mod._TERMINAL_LOG,
                  mod._EXIT_STATUS_FILE):
        assert not os.path.exists(os.path.join(out_dir, fname)), fname

    norm_out = os.path.normpath(out_dir)
    mod._runs()[norm_out] = {"output": norm_out, "result": 0, "node": "/gone"}
    with open(os.path.join(out_dir, mod._RUN_RECORD), "w") as f:
        json.dump({"kind": "background", "pid": 99999999, "started": 1.0,
                   "log": os.path.join(out_dir, "log.txt")}, f)
    reattached = mod.reattach_run(node)
    assert reattached is not None
    mod._runs().pop(norm_out, None)

    node.destroy()
    print("PASS: Batch 2A (PolyFEM 2.0) fixes verified")


def _write_vtu_xml(path, points, cells_conn, cells_offs, cells_types,
                   point_arrays, byte_order="LittleEndian", fmt="ascii",
                   compressor="", appended_encoding="raw",
                   two_stream=False, add_comment=False):
    endian = ">" if byte_order == "BigEndian" else "<"
    header_dtype = np.dtype(endian + "u4")

    def encode_payload(arr_bytes, for_appended=False):
        if compressor == "vtkZLibDataCompressor":
            comp = zlib.compress(arr_bytes)
            hdr = np.array([1, len(arr_bytes), len(arr_bytes), len(comp)],
                           dtype=header_dtype).tobytes()
            if (for_appended and appended_encoding == "base64") or (
                    not for_appended and fmt == "binary"):
                if two_stream:
                    return base64.b64encode(hdr) + base64.b64encode(comp)
                return base64.b64encode(hdr + comp)
            return hdr + comp
        hdr = np.array([len(arr_bytes)], dtype=header_dtype).tobytes()
        if (for_appended and appended_encoding == "base64") or (
                not for_appended and fmt == "binary"):
            if two_stream:
                return base64.b64encode(hdr) + base64.b64encode(arr_bytes)
            return base64.b64encode(hdr + arr_bytes)
        return hdr + arr_bytes

    appended_chunks = []
    appended_offset = 0

    def array_xml(name, arr, vtk_type, ncomp=1):
        nonlocal appended_offset
        np_dt = np.dtype(vtu_parser._VTK_TO_NP[vtk_type]).newbyteorder(endian)
        typed = np.asarray(arr, dtype=np_dt)
        name_attr = f' Name="{name}"' if name else ""
        comp_attr = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
        if fmt == "ascii":
            body = " ".join(str(x) for x in np.asarray(arr).ravel())
            return (f'<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
                    f'format="ascii">{body}</DataArray>').encode("ascii")
        if fmt == "binary":
            body = encode_payload(typed.tobytes(), for_appended=False)
            return (f'<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
                    f'format="binary">').encode("ascii") + body + b"</DataArray>"
        chunk = encode_payload(typed.tobytes(), for_appended=True)
        off = appended_offset
        appended_chunks.append(chunk)
        appended_offset += len(chunk)
        return (f'<DataArray type="{vtk_type}"{name_attr}{comp_attr} '
                f'format="appended" offset="{off}"/>').encode("ascii")

    comp_attr = f' compressor="{compressor}"' if compressor else ""
    parts = [
        f'<?xml version="1.0"?>\n'
        f'<VTKFile type="UnstructuredGrid" version="0.1" '
        f'byte_order="{byte_order}" header_type="UInt32"{comp_attr}>\n'
        f'<UnstructuredGrid><Piece NumberOfPoints="{len(points)}" '
        f'NumberOfCells="{len(cells_types)}">\n'.encode("ascii")
    ]
    if add_comment:
        parts.append(
            b'<!-- comment containing <DataArray Name="fake" format="ascii">'
            b'1 2 3</DataArray> -->\n')
    parts.append(b"<Points>")
    parts.append(array_xml("", points, "Float64", ncomp=3))
    parts.append(b"</Points>\n<Cells>")
    parts.append(array_xml("connectivity", cells_conn, "Int32"))
    parts.append(array_xml("offsets", cells_offs, "Int32"))
    parts.append(array_xml("types", cells_types, "UInt8"))
    parts.append(b"</Cells>\n<PointData>")
    for pname, (arr, vtype, nc) in point_arrays.items():
        parts.append(array_xml(pname, arr, vtype, ncomp=nc))
    parts.append(b"</PointData>\n</Piece></UnstructuredGrid>\n")
    if fmt == "appended":
        parts.append(
            f'<AppendedData encoding="{appended_encoding}">_'.encode("ascii"))
        parts.extend(appended_chunks)
        parts.append(b"</AppendedData>\n")
    parts.append(b"</VTKFile>\n")
    with open(path, "wb") as handle:
        handle.write(b"".join(parts))
    return path


def test_readpvd_batch_2b(root):
    hou.hda.installFile(
        os.path.join(BASE, "object_readPVD.1.0.hdanc"),
        force_use_assets=True,
    )
    node = hou.node("/obj").createNode("readPVD::1.0", "pvd_patch2")
    mod = node.hdaModule()

    # 1. Read-only status fields, multiline probe_readout, and default Time Mapping
    node.updateParmStates()
    assert node.parm("time_mapping").evalAsString() == "steps"
    for name in ("availability_status", "color_status", "reference_status",
                 "probe_readout"):
        assert node.parm(name).isDisabled(), f"{name} should be read-only"
    assert "editorlines" in node.parm("probe_readout").parmTemplate().tags()

    # 2. DisableWhen conditionals
    node.parm("remesh_mode").set(1)
    node.parm("range_percentile").set(0)
    node.parm("color_scale").set("log10")
    node.parm("range_lock").set(1)
    node.parm("reference_enable").set(0)
    node.parm("clip_mode").set("off")
    node.parm("legend_mode").set("off")
    node.parm("gnomon_show").set(0)
    node.parm("probe_enabled").set(0)
    node.updateParmStates()
    for pname in ("topo_frame", "range_percentiles", "range_symmetric",
                  "color_min", "color_max", "autoscale", "autoscale_all",
                  "reference_frame", "reference_mode", "clip_origin",
                  "clip_keep", "slice_thickness", "legend_title",
                  "legend_scale", "overlay_margin", "gnomon_scale",
                  "probe_point", "autofiber", "autoglyph"):
        pt = node.parmTuple(pname)
        assert pt is not None and pt[0].isDisabled(), (
            f"{pname} should be disabled")
    # Reset
    node.setParms({"remesh_mode": 0, "color_scale": "linear", "range_lock": 0})

    # 3. multi_block_display has dataepoch spare parm
    assert node.node("multi_block_display").parm("dataepoch") is not None
    assert node.node("reference_comparison").parm("dataepoch") is not None

    # 4. ViewerStateModule world-to-local ray & probe marker xform
    vsm = node.type().definition().sections()["ViewerStateModule"].contents()
    assert "worldTransform().inverted()" in vsm
    assert '"xform": self.node.worldTransform()' in vsm

    # 5. Synthetic 2-step PVD with unit cube (x in [0, 1]) + body_ids +
    #    F + cauchy_stress (no von_mises) + intermittent field on step 1
    pvd_dir = os.path.join(root, "pvd_case")
    os.makedirs(pvd_dir, exist_ok=True)
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    conn = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    offs = np.array([8], dtype=np.int32)
    types = np.array([12], dtype=np.uint8)  # VTK_HEXAHEDRON
    F1 = np.tile([1.0, 0.0, 0.0], (8, 1))
    F2 = np.tile([0.0, 1.0, 0.0], (8, 1))
    F3 = np.tile([0.0, 0.0, 1.0], (8, 1))
    S1 = np.tile([10.0, 0.0, 0.0], (8, 1))
    S2 = np.tile([0.0, 0.0, 0.0], (8, 1))
    S3 = np.tile([0.0, 0.0, 0.0], (8, 1))
    sol0 = np.zeros((8, 3))
    sol1 = np.tile([0.1, 0.0, 0.0], (8, 1))
    bodies = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    _write_vtu_xml(
        os.path.join(pvd_dir, "step_0.vtu"), pts, conn, offs, types,
        {"solution": (sol0, "Float64", 3), "body_ids": (bodies, "Float64", 1),
         "F_1": (F1, "Float64", 3), "F_2": (F2, "Float64", 3),
         "F_3": (F3, "Float64", 3), "cauchy_stress_1": (S1, "Float64", 3),
         "cauchy_stress_2": (S2, "Float64", 3),
         "cauchy_stress_3": (S3, "Float64", 3)})
    _write_vtu_xml(
        os.path.join(pvd_dir, "step_1.vtu"), pts, conn, offs, types,
        {"solution": (sol1, "Float64", 3), "body_ids": (bodies, "Float64", 1),
         "F_1": (F1, "Float64", 3), "F_2": (F2, "Float64", 3),
         "F_3": (F3, "Float64", 3), "cauchy_stress_1": (S1, "Float64", 3),
         "cauchy_stress_2": (S2, "Float64", 3),
         "cauchy_stress_3": (S3, "Float64", 3),
         "late_scalar": (np.arange(1, 9, dtype=np.float64), "Float64", 1)})
    pvd_path = os.path.join(pvd_dir, "sim.pvd")
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '<Collection>\n'
            '<DataSet timestep="0.0" file="step_0.vtu"/>\n'
            '<DataSet timestep="1.0" file="step_1.vtu"/>\n'
            '</Collection></VTKFile>\n'
        )
    hou.setFrame(0)
    node.parm("PVD_file").set(pvd_path)
    mod.sync_available_options({"node": node})

    # 6. Symmetric Thin Slice around x = 0.5 with thickness 0.2 -> [0.4, 0.6]
    node.setParms({
        "show_deformed": 0,
        "clip_mode": "slice",
        "clip_originx": 0.5, "clip_originy": 0.5, "clip_originz": 0.5,
        "clip_directionx": 1.0, "clip_directiony": 0.0, "clip_directionz": 0.0,
        "slice_thickness": 0.2,
    })
    out = node.node("output")
    out.cook(force=True)
    xs = [pt.position()[0] for pt in out.geometry().points()]
    assert min(xs) >= 0.39 and max(xs) <= 0.61, (min(xs), max(xs))
    assert min(xs) < 0.45 and max(xs) > 0.55, (min(xs), max(xs))
    node.parm("clip_mode").set("off")

    # 7. _mesh_field von_mises fallback & pk1/pk2 |J|<=1e-12 zeroing
    mesh0 = mod.read_vtu(os.path.join(pvd_dir, "step_0.vtu"))
    vm = mod._mesh_field(mesh0, "von_mises")
    assert vm is not None and np.allclose(vm, 10.0)
    mesh_deg = dict(mesh0, point_data=dict(mesh0["point_data"]))
    mesh_deg["point_data"]["F_3"] = np.zeros((8, 3))
    pk1 = mod._mesh_field(mesh_deg, "pk1")
    pk2 = mod._mesh_field(mesh_deg, "pk2")
    assert np.allclose(pk1, 0.0) and np.allclose(pk2, 0.0)

    # 8. probe_readout honors digits/format, deduplicates solution, shows smoothed
    node.setParms({"color_attrib": "solution", "smooth_field": 1,
                   "legend_digits": 3, "legend_number_format": "scientific"})
    out.cook(force=True)
    _, readout = mod.probe_readout(node, 0)
    assert readout.count("Displacement Vector:") == 1, readout
    assert "Displayed (smoothed" in readout, readout
    assert "e+00" in readout or "e-0" in readout, readout

    # 9. sim_time_text clamps negative frame
    hou.setFrame(-5)
    text = mod.sim_time_text(node)
    assert "Step 0 of 1" in text, text
    hou.setFrame(0)

    # 10. color_scale_changed auto-ranges to positive on log10 switch
    node.setParms({"color_attrib": "von_mises", "color_min": 0.0,
                   "color_max": 10.0, "range_lock": 0, "color_scale": "log10"})
    mod.color_scale_changed({"node": node})
    assert node.evalParm("color_min") > 0.0, node.evalParm("color_min")
    node.parm("color_scale").set("linear")

    # 11. export_data accepts late_scalar present on exported step 1 even
    #     while playbar is on step 0
    xlsx_base = os.path.join(pvd_dir, "late_out")
    node.setParms({
        "export_regions": 1, "export_region_kind1": "all",
        "export_quantities": 1, "export_q_field1": "late_scalar",
        "export_q_value1": "auto", "export_forces": 0,
        "export_step_first": 1, "export_step_last": 1,
        "export_data_file": xlsx_base,
    })
    summary = mod.export_data(node)
    assert os.path.isfile(summary["path"])
    assert summary["path"].endswith(".xlsx")

    # 12. 2D VTU with F_1/F_2 only (no F_3) cooks cleanly
    vtu_2d = os.path.join(pvd_dir, "planar.vtu")
    _write_vtu_xml(
        vtu_2d,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([5], dtype=np.uint8),
        {"F_1": (np.tile([1, 0, 0], (3, 1)), "Float64", 3),
         "F_2": (np.tile([0, 1, 0], (3, 1)), "Float64", 3)},
    )
    pvd_2d = os.path.join(pvd_dir, "planar.pvd")
    with open(pvd_2d, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '<Collection><DataSet timestep="0.0" file="planar.vtu"/>'
            '</Collection></VTKFile>\n'
        )
    node.parm("PVD_file").set(pvd_2d)
    out.cook(force=True)
    assert len(out.geometry().points()) == 3

    # 13. Points block + comma-separated visible_bodies + NaN multi-block color
    vtu_pts = os.path.join(pvd_dir, "pts.vtu")
    _write_vtu_xml(
        vtu_pts,
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([1, 1, 1], dtype=np.uint8),  # VTK_VERTEX -> "point"
        {"body_ids": (np.array([0.0, 1.0, 2.0]), "Float64", 1),
         "val": (np.array([1.0, np.nan, 2.0]), "Float64", 1),
         "displacement": (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                   dtype=np.float64), "Float64", 3)},
    )
    pvd_pts = os.path.join(pvd_dir, "pts.pvd")
    with open(pvd_pts, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
            '<Collection><DataSet timestep="0.0" file="pts.vtu"/>'
            '</Collection></VTKFile>\n'
        )
    node.setParms({"PVD_file": pvd_pts, "visible_bodies": "0, 2",
                   "show_deformed": 0})
    out.cook(force=True)
    assert len(out.geometry().points()) == 2, len(out.geometry().points())

    # Multi-block builder skips "point" cells and handles NaN colors
    mesh_pts = mod.read_vtu(vtu_pts)
    blk = mod._build_block_geometry(
        mesh_pts, "Points", "points",
        np.ones((3, 3)), None, 0)
    assert len(blk.points()) == 3 and len(blk.prims()) == 0

    node.destroy()
    print("PASS: Batch 2B (Read PVD 1.0) fixes verified")


def test_common_batch_2c(root):
    vtu_dir = os.path.join(root, "vtu_formats")
    os.makedirs(vtu_dir, exist_ok=True)
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   dtype=np.float64)
    conn = np.array([0, 1, 2, 3], dtype=np.int32)
    offs = np.array([4], dtype=np.int32)
    types = np.array([10], dtype=np.uint8)
    scalars = np.array([1.5, -2.25, 3.75, 4.0], dtype=np.float64)

    # 1. BigEndian inline binary (single-stream & two-stream), uncompressed & zlib
    for comp in ("", "vtkZLibDataCompressor"):
        for two in (False, True):
            p = _write_vtu_xml(
                os.path.join(vtu_dir, f"be_bin_{bool(comp)}_{two}.vtu"),
                pts, conn, offs, types, {"s": (scalars, "Float64", 1)},
                byte_order="BigEndian", fmt="binary", compressor=comp,
                two_stream=two, add_comment=True,
            )
            mesh = vtu_parser.read_vtu(p)
            assert np.allclose(mesh["points"], pts)
            assert np.allclose(mesh["point_data"]["s"], scalars)
            assert mesh["point_data"]["s"].dtype.byteorder in ("=", "<")

    # 2. AppendedData encoding="base64" (uncompressed & zlib, 1- & 2-stream)
    for comp in ("", "vtkZLibDataCompressor"):
        for two in (False, True):
            p = _write_vtu_xml(
                os.path.join(vtu_dir, f"app_b64_{bool(comp)}_{two}.vtu"),
                pts, conn, offs, types, {"s": (scalars, "Float64", 1)},
                byte_order="BigEndian", fmt="appended", compressor=comp,
                appended_encoding="base64", two_stream=two,
            )
            mesh = vtu_parser.read_vtu(p)
            assert np.allclose(mesh["points"], pts)
            assert np.allclose(mesh["point_data"]["s"], scalars)

    # 3. Non-ZLib compressor raises explicit VtuParseError
    bad_comp = os.path.join(vtu_dir, "lz4.vtu")
    with open(bad_comp, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian" compressor="vtkLZ4DataCompressor">'
            '<UnstructuredGrid/></VTKFile>'
        )
    try:
        vtu_parser.read_vtu(bad_comp)
        raise AssertionError("expected VtuParseError on vtkLZ4DataCompressor")
    except vtu_parser.VtuParseError as exc:
        assert "vtkLZ4DataCompressor" in str(exc)

    # 4. msh_parser supports Gmsh P5 triangle (type 25) and P5 line (type 28)
    assert msh_parser.GMSH_ELEM_TYPES[25] == ("tri", 21, 3)
    assert msh_parser.GMSH_ELEM_TYPES[28] == ("line", 6, 2)
    p5_msh = os.path.join(vtu_dir, "p5.msh")
    node_lines = "\n".join(f"{i} {i * 0.1} 0 0" for i in range(1, 22))
    tri_nodes = " ".join(str(i) for i in range(1, 22))
    line_nodes = " ".join(str(i) for i in range(1, 7))
    with open(p5_msh, "w", encoding="utf-8") as f:
        f.write(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            f"$Nodes\n21\n{node_lines}\n$EndNodes\n"
            f"$Elements\n2\n"
            f"1 25 2 1 1 {tri_nodes}\n"
            f"2 28 2 2 2 {line_nodes}\n"
            "$EndElements\n"
        )
    parsed = msh_parser.read_msh(p5_msh)
    assert parsed["cells"]["tri"]["corners"].shape == (1, 3)
    assert parsed["cells"]["line"]["corners"].shape == (1, 2)

    # 5. build_all.py uses hou.homeHoudiniDirectory() (portable)
    with open(os.path.join(BASE, "src", "build_all.py"),
              encoding="utf-8") as f:
        build_src = f.read()
    assert "homeHoudiniDirectory" in build_src
    assert "~/Library/Preferences/houdini" not in build_src

    print("PASS: Batch 2C (vtu_parser, msh_parser, build tooling) verified")


def main():
    root = tempfile.mkdtemp(prefix="patch2_fixes_")
    try:
        test_polyfem_batch_2a(root)
        test_readpvd_batch_2b(root)
        test_common_batch_2c(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL PATCH 2 REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
