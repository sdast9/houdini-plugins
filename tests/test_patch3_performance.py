"""Regression & performance tests for Patch 3 (2026-11-20).

Run via hython:
  hython tests/test_patch3_performance.py
"""
import base64
import json
import os
import shutil
import struct
import sys
import tempfile
import time

import hou
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src", "common"))
import vtu_parser


def _b64_raw(arr, header_type="UInt32"):
    raw = np.ascontiguousarray(arr).tobytes()
    hdr = struct.pack("<I" if header_type == "UInt32" else "<Q", len(raw))
    return base64.b64encode(hdr + raw).decode("ascii")


def _write_inline_binary_vtu(path, points, tets, point_data):
    """Write an uncompressed inline-binary VTU (same layout PolyFEM writes)."""
    n_pts = len(points)
    n_cells = len(tets)
    conn = np.ascontiguousarray(tets, dtype=np.int32).ravel()
    offsets = np.arange(4, 4 * n_cells + 1, 4, dtype=np.int32)
    types = np.full(n_cells, 10, dtype=np.uint8)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">',
        '      <PointData>',
    ]
    for name, arr in point_data.items():
        arr = np.asarray(arr, dtype=np.float64)
        comps = 1 if arr.ndim == 1 else arr.shape[1]
        lines.append(
            f'        <DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{comps}" format="binary">'
            f'{_b64_raw(arr)}</DataArray>')
    lines.extend([
        '      </PointData>',
        '      <CellData/>',
        '      <Points>',
        '        <DataArray type="Float64" NumberOfComponents="3" '
        f'format="binary">{_b64_raw(np.asarray(points, dtype=np.float64))}'
        '</DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int32" Name="connectivity" format="binary">'
        f'{_b64_raw(conn)}</DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="binary">'
        f'{_b64_raw(offsets)}</DataArray>',
        '        <DataArray type="UInt8" Name="types" format="binary">'
        f'{_b64_raw(types)}</DataArray>',
        '      </Cells>',
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ])
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _make_discontinuous_cube_pvd(work, steps=(0.0, 0.25, 0.5)):
    """Create a 3-step discontinuous 6-tet cube PVD with F, cauchy, von_mises,
    solution, and nodal forces (for derived, cache, glyph, and force tests)."""
    corners = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    cube_tets = np.array([
        [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
        [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
    ], dtype=np.int32)
    # Discontinuous mesh: duplicate vertices per tet (24 points, 8 unique nodes)
    pts = corners[cube_tets.ravel()]
    tets = np.arange(len(pts), dtype=np.int32).reshape(-1, 4)

    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1">',
        '  <Collection>',
    ]
    for s_idx, t in enumerate(steps):
        fname = f"step_{s_idx}.vtu"
        n = len(pts)
        # Non-trivial deformation gradient F (symmetric + slight rotation)
        scale_z = 1.0 + 0.15 * (s_idx + 1)
        F_mat = np.array([
            [1.0 + 0.05 * (s_idx + 1), 0.03 * (s_idx + 1), 0.0],
            [-0.01 * (s_idx + 1), 0.96, 0.02 * (s_idx + 1)],
            [0.0, 0.0, scale_z],
        ], dtype=np.float64)
        # Vary slightly per element so nodal recovery has non-trivial averaging
        F_all = np.broadcast_to(F_mat, (n, 3, 3)).copy()
        for elem in range(len(tets)):
            F_all[4 * elem:4 * elem + 4, 0, 0] += 0.01 * elem
        # PolyFEM exports columns F_1, F_2, F_3
        cauchy_base = np.array([
            [10.0 * (s_idx + 1), 2.5, -1.0],
            [2.5, -4.0 * (s_idx + 1), 1.5],
            [-1.0, 1.5, 6.0 * (s_idx + 1)],
        ], dtype=np.float64)
        cauchy_all = np.broadcast_to(cauchy_base, (n, 3, 3)).copy()
        for elem in range(len(tets)):
            cauchy_all[4 * elem:4 * elem + 4, 0, 0] += 0.8 * elem
            cauchy_all[4 * elem:4 * elem + 4, 1, 2] += 0.2 * elem
            cauchy_all[4 * elem:4 * elem + 4, 2, 1] += 0.2 * elem
        tr = np.trace(cauchy_all, axis1=1, axis2=2)
        dev = cauchy_all - tr[:, None, None] * np.eye(3) / 3.0
        vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
        sol = pts * np.array([0.05 * (s_idx + 1), -0.02, 0.15 * (s_idx + 1)])
        forces = np.zeros((n, 3), dtype=np.float64)
        forces[pts[:, 2] > 0.5, 2] = 12.5 * (s_idx + 1)
        contact_f = np.zeros((n, 3), dtype=np.float64)
        contact_f[pts[:, 2] > 0.5, 2] = -2.0 * (s_idx + 1)
        body_ids = np.full(n, 1001.0, dtype=np.float64)
        _write_inline_binary_vtu(
            os.path.join(work, fname), pts, tets, {
                "solution": sol,
                "von_mises": vm,
                "body_ids": body_ids,
                "F_1": F_all[:, :, 0],
                "F_2": F_all[:, :, 1],
                "F_3": F_all[:, :, 2],
                "cauchy_stress_1": cauchy_all[:, :, 0],
                "cauchy_stress_2": cauchy_all[:, :, 1],
                "cauchy_stress_3": cauchy_all[:, :, 2],
                "nodal_forces": forces,
                "contact_forces": contact_f,
            })
        pvd_lines.append(f'    <DataSet timestep="{t}" file="{fname}"/>')
    pvd_lines.extend(['  </Collection>', '</VTKFile>'])
    pvd_path = os.path.join(work, "sim.pvd")
    with open(pvd_path, "w") as handle:
        handle.write("\n".join(pvd_lines))

    # Write a minimal hda_scene.json in ../input for force_readout_text tests
    input_dir = os.path.join(os.path.dirname(work), "input")
    os.makedirs(input_dir, exist_ok=True)
    top_nodes = corners[corners[:, 2] > 0.5]
    scene = {
        "schema": "polyfem-houdini-scene",
        "version": 1,
        "params_sha256": "",
        "timeline": None,
        "obstacles": {},
        "force_sets": [{
            "label": "Top Face",
            "kind": "nodes",
            "geometry": 1,
            "body": 1001,
            "vertices": top_nodes.tolist(),
            "exact": True,
        }],
    }
    with open(os.path.join(input_dir, "hda_scene.json"), "w") as handle:
        json.dump(scene, handle)
    return pvd_path


def _write_simple_msh(path):
    """Write a small valid MSH 2.2 single-tet mesh."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n4\n"
                "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
                "$EndNodes\n")
        f.write("$Elements\n1\n1 4 2 1 1 1 2 3 4\n$EndElements\n")


def _write_simple_obj(path):
    with open(path, "w") as f:
        f.write("v 0 0 2\nv 1 0 2\nv 0 1 2\nf 1 2 3\n")


def test_vtu_field_info_mmap_and_skip(work):
    # 1. Appended VTU with a multi-MB appended binary blob
    appended_path = os.path.join(work, "appended.vtu")
    header = (
        b'<?xml version="1.0"?>\n'
        b'<VTKFile type="UnstructuredGrid" version="0.1" '
        b'byte_order="LittleEndian">\n'
        b'  <UnstructuredGrid>\n'
        b'    <Piece NumberOfPoints="100" NumberOfCells="10">\n'
        b'      <!-- <DataArray Name="fake_comment" NumberOfComponents="9"/> -->\n'
        b'      <PointData>\n'
        b'        <DataArray type="Float64" Name="solution" '
        b'NumberOfComponents="3" format="appended" offset="0"/>\n'
        b'        <DataArray type="Float64" Name="von_mises" '
        b'format="appended" offset="2400"/>\n'
        b'      </PointData>\n'
        b'      <CellData>\n'
        b'        <DataArray type="Int32" Name="mat_id" '
        b'format="appended" offset="3200"/>\n'
        b'      </CellData>\n'
        b'      <Points>\n'
        b'        <DataArray type="Float64" NumberOfComponents="3" '
        b'format="appended" offset="3240"/>\n'
        b'      </Points>\n'
        b'    </Piece>\n'
        b'  </UnstructuredGrid>\n'
        b'  <AppendedData encoding="raw">\n_'
    )
    # 8 MB blob containing fake <DataArray> bytes that MUST NOT be scanned
    blob = b"X" * (4 << 20) + b'<DataArray Name="leak" NumberOfComponents="3"/>' + b"Y" * (4 << 20)
    with open(appended_path, "wb") as handle:
        handle.write(header + blob + b'\n  </AppendedData>\n</VTKFile>\n')
    vtu_parser._FIELD_INFO_CACHE.clear()
    vtu_parser._VTU_CACHE.clear()
    t0 = time.perf_counter()
    info = vtu_parser.read_vtu_field_info(appended_path)
    elapsed = time.perf_counter() - t0
    assert info == {
        "point_data": {"solution": 3, "von_mises": 1},
        "cell_data": {"mat_id": 1},
    }, info
    assert elapsed < 0.2, f"appended field_info took {elapsed:.3f}s"

    # 2. Inline-binary VTU with large uncompressed payloads
    inline_path = os.path.join(work, "inline_big.vtu")
    big_pts = np.linspace(0.0, 1.0, 60000 * 3, dtype=np.float64).reshape(-1, 3)
    big_sol = np.ones_like(big_pts)
    big_vm = np.ones(len(big_pts), dtype=np.float64)
    _write_inline_binary_vtu(
        inline_path, big_pts, np.array([[0, 1, 2, 3]], dtype=np.int32),
        {"solution": big_sol, "von_mises": big_vm})
    vtu_parser._FIELD_INFO_CACHE.clear()
    t0 = time.perf_counter()
    info2 = vtu_parser.read_vtu_field_info(inline_path)
    elapsed2 = time.perf_counter() - t0
    assert info2 == {
        "point_data": {"solution": 3, "von_mises": 1},
        "cell_data": {},
    }, info2
    assert elapsed2 < 0.2, f"inline binary field_info took {elapsed2:.3f}s"
    print(f"PASS: read_vtu_field_info mmap+skip ({elapsed*1e3:.2f}ms appended, "
          f"{elapsed2*1e3:.2f}ms inline)")


def test_readpvd_patch3(work):
    hou.hda.installFile(
        os.path.join(BASE, "object_readPVD.1.0.hdanc"),
        force_use_assets=True)
    out_dir = os.path.join(work, "pvd_run", "output")
    os.makedirs(out_dir, exist_ok=True)
    pvd_path = _make_discontinuous_cube_pvd(out_dir)

    node = hou.node("/obj").createNode("readPVD::1.0", "p3_readpvd")
    mod = node.hdaModule()
    node.parm("PVD_file").set(pvd_path)
    mod.start({"node": node})

    # 1. derived_mode defaults to on_demand and disables when derived == 0
    assert node.parm("derived_mode").evalAsString() == "on_demand"
    node.parm("derived").set(0)
    node.updateParmStates()
    assert node.parm("derived_mode").isDisabled()
    node.parm("derived").set(1)
    node.updateParmStates()
    assert not node.parm("derived_mode").isDisabled()

    # 2. On-demand allocates zero unused heavy tensor attributes for von_mises
    hou.setFrame(1)
    node.setParms({
        "color_attrib": "von_mises",
        "add_glyphs": 0,
        "show_fibers": 0,
        "probe_enabled": 0,
        "probe_point": -1,
    })
    out_res = node.node("OUT_result")
    out_res.cook(force=True)
    g_demand = out_res.geometry()
    heavy_unused = (
        "right_cauchy_green", "left_cauchy_green", "right_stretch",
        "left_stretch", "green_lagrange_strain", "almansi_strain",
        "hencky_strain", "infinitesimal_strain", "cauchy_mat",
        "deviatoric_stress", "pk1", "pk2", "F_mat", "J",
    )
    present_unused = [a for a in heavy_unused if g_demand.findPointAttrib(a) is not None]
    assert not present_unused, f"on_demand allocated unused attribs: {present_unused}"

    # 3. Switching color_attrib to green_lagrange_eigenvalues computes only it
    #    and matches 0.5 * (C_eigs - 1) and derived_mode='all'
    node.parm("color_attrib").set("green_lagrange_eigenvalues")
    out_res.cook(force=True)
    g_gl = out_res.geometry()
    assert g_gl.findPointAttrib("green_lagrange_eigenvalues") is not None
    assert g_gl.findPointAttrib("hencky_strain") is None
    assert g_gl.findPointAttrib("pk1") is None
    gl_demand = np.frombuffer(
        g_gl.pointFloatAttribValuesAsString("green_lagrange_eigenvalues"),
        dtype=np.float32).reshape(-1, 3)

    node.parm("derived_mode").set("all")
    out_res.cook(force=True)
    g_all = out_res.geometry()
    gl_all = np.frombuffer(
        g_all.pointFloatAttribValuesAsString("green_lagrange_eigenvalues"),
        dtype=np.float32).reshape(-1, 3)
    rcg_all = np.frombuffer(
        g_all.pointFloatAttribValuesAsString("right_cauchy_green_eigenvalues"),
        dtype=np.float32).reshape(-1, 3)
    assert np.allclose(gl_demand, gl_all, atol=1e-6)
    assert np.allclose(gl_all, 0.5 * (rcg_all - 1.0), atol=1e-5)

    # 4. Glyphs in on_demand mode allocate only the chosen glyph tensor
    node.setParms({
        "derived_mode": "on_demand",
        "color_attrib": "von_mises",
        "add_glyphs": 1,
        "glyph_tensor": "cauchy",
        "smooth_field": 1,
    })
    out_res.cook(force=True)
    g_glyph = out_res.geometry()
    for needed in ("cauchy_mat", "cauchy_eigenvalues", "cauchy_eigenvectors"):
        assert g_glyph.findPointAttrib(needed) is not None, f"missing {needed}"
    assert g_glyph.findPointAttrib("left_cauchy_green") is None
    assert g_glyph.findPointAttrib("pk1") is None

    # 5. Glyph recovery unique-node eigh matches exact per-group averaged tensor
    rec = node.node("glyph_tensor_recovery")
    rec.cook(force=True)
    g_rec = rec.geometry()
    c_mats = np.frombuffer(
        g_rec.pointFloatAttribValuesAsString("cauchy_mat"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3, 3)
    c_vals = np.frombuffer(
        g_rec.pointFloatAttribValuesAsString("cauchy_eigenvalues"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3)
    c_vecs = np.frombuffer(
        g_rec.pointFloatAttribValuesAsString("cauchy_eigenvectors"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3, 3)
    groups = np.frombuffer(
        g_rec.pointFloatAttribValuesAsString("coincident_id"),
        dtype=np.float32).astype(np.int64)
    # Every point sharing a coincidence group must have identical eigenvalues/vecs
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        assert np.allclose(c_vals[idx], c_vals[idx[0]], atol=1e-6)
        assert np.allclose(c_vecs[idx], c_vecs[idx[0]], atol=1e-6)
        m_avg = c_mats[idx].mean(axis=0)
        m_avg = 0.5 * (m_avg + m_avg.T)
        recon = c_vecs[idx[0]] @ np.diag(c_vals[idx[0]]) @ c_vecs[idx[0]].T
        assert np.allclose(recon, m_avg, atol=1e-4)

    # 6. Probe in on_demand mode computes principal_stretches, cauchy & pk2 eigs
    node.setParms({"add_glyphs": 0, "smooth_field": 0, "probe_enabled": 1})
    out_res.cook(force=True)
    g_probe = out_res.geometry()
    for needed in ("principal_stretches", "cauchy_eigenvalues", "pk2_eigenvalues"):
        assert g_probe.findPointAttrib(needed) is not None, f"probe missing {needed}"
    assert g_probe.findPointAttrib("almansi_strain") is None
    _, readout = mod.probe_readout(node, 0)
    assert "Principal Stretches" in readout and "Cauchy Stress Principal Values" in readout, readout
    node.setParms({"probe_enabled": 0, "probe_point": -1})
    print("PASS: DERIVED_VEX on_demand, Green-Lagrange reuse, & unique-node glyph eigh")

    # 7. Step-keyed cache1 under Simulation Time (no duplicate entries)
    node.setParms({
        "time_mapping": 2,  # Simulation Time (24 fps -> steps at frames 0, 6, 12)
        "time_start_frame": 0.0,
        "time_scale": 1.0,
        "cache": 1,
    })
    mod.clear_cache({"node": node})
    frame_node = node.node("frame_data")
    cache_node = node.node("cache1")
    before = frame_node.cookCount()
    for f in range(0, 12):  # 12 frames covering only steps 0 and 1!
        hou.setFrame(f)
        out_res.cook(force=False)
    after = frame_node.cookCount()
    assert after - before == 2, (
        f"Expected 2 frame_data cooks for 2 steps across 12 frames, got {after - before}")
    hou.setFrame(0)
    rows = dict(cache_node.infoTree(verbose=False).branches()[
        "Cache SOP Info"].rows())
    cached_count = int(rows["Geometries Cached"].split("/")[0])
    assert cached_count == 2, f"Expected 2 cached geometries, got {cached_count}"
    print(f"PASS: step-keyed cache1 deduplicated 12 frames -> {cached_count} entries")

    # 8. force_readout_text caching (frame cache + _FORCE_CURVES reuse)
    node.setParms({"time_mapping": 1})  # 1 frame per step
    hou.setFrame(0)
    mod_globals = mod.force_readout_text.__globals__
    orig_load = mod_globals["load_frame"]
    calls = [0]

    def counting_load(*args, **kwargs):
        calls[0] += 1
        return orig_load(*args, **kwargs)

    mod._FORCE_FRAME_CACHE.clear()
    mod._FORCE_CURVES.pop(node.sessionId(), None)
    mod_globals["load_frame"] = counting_load
    try:
        txt0 = mod.force_readout_text(node)
        assert "Top Face:" in txt0 and calls[0] == 1, (txt0, calls[0])
        txt0_again = mod.force_readout_text(node)
        assert txt0_again == txt0 and calls[0] == 1, (
            f"force_readout_text did not hit frame cache (calls={calls[0]})")
        # Compute full curves -> subsequent step 1 readout hits _FORCE_CURVES
        mod.force_curves(node)
        calls_after_curves = calls[0]
        hou.setFrame(1)
        txt1 = mod.force_readout_text(node)
        assert "Top Face:" in txt1 and calls[0] == calls_after_curves, (
            f"force_readout_text did not reuse _FORCE_CURVES "
            f"({calls[0]} vs {calls_after_curves})")
    finally:
        mod_globals["load_frame"] = orig_load
    # Scoped invalidation clears _FORCE_FRAME_CACHE entries for this PVD
    assert len(mod._FORCE_FRAME_CACHE) > 0
    mod._invalidate_pvd_caches(pvd_path)
    assert len(mod._FORCE_FRAME_CACHE) == 0
    print("PASS: force_readout_text frame cache & _FORCE_CURVES reuse")

    # 9. Vectorized ramp LUT accuracy & NaN handling
    ramp = hou.Ramp(
        (hou.rampBasis.Linear, hou.rampBasis.Linear, hou.rampBasis.Linear),
        (0.0, 0.5, 1.0),
        ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)))
    vals = np.linspace(0.0, 1.0, 2048, dtype=np.float64)
    vals[10] = np.nan
    vals[20] = np.inf
    finite = np.isfinite(vals)
    invalid_rgb = (0.5, 0.25, 0.75)
    vec_colors = mod._sample_ramp_vectorized(ramp, vals, finite, invalid_rgb)
    ref_colors = np.asarray([
        ramp.lookup(float(v)) if ok else invalid_rgb
        for v, ok in zip(vals, finite)], dtype=np.float64)
    assert np.allclose(vec_colors, ref_colors, atol=2e-3), (
        float(np.abs(vec_colors - ref_colors).max()))
    assert np.allclose(vec_colors[10], invalid_rgb)
    assert np.allclose(vec_colors[20], invalid_rgb)
    print("PASS: _sample_ramp_vectorized matches ramp.lookup & handles NaN/Inf")

    # 10. ViewerStateModule overlay bar/tick cache keys
    vs_src = node.type().definition().sections()["ViewerStateModule"].contents()
    vs_ns = {}
    exec(compile(vs_src, "ViewerStateModule.py", "exec"), vs_ns)
    bk1 = vs_ns["_bar_cache_key"](node, 10, 20, 24, 200)
    bk2 = vs_ns["_bar_cache_key"](node, 10, 20, 24, 200)
    bk3 = vs_ns["_bar_cache_key"](node, 12, 20, 24, 200)
    assert bk1 == bk2 and bk1 != bk3
    tk1 = vs_ns["_tick_cache_key"](node, 10, 20, 24, 200, False)
    tk2 = vs_ns["_tick_cache_key"](node, 10, 20, 24, 200, False)
    node.parm("legend_ticks").set(node.evalParm("legend_ticks") + 1)
    tk3 = vs_ns["_tick_cache_key"](node, 10, 20, 24, 200, False)
    assert tk1 == tk2 and tk1 != tk3
    print("PASS: ViewerStateModule bar/tick cache keys")


def test_polyfem_and_msh_patch3(work):
    hou.hda.installFile(
        os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"), force_use_assets=True)
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"),
        force_use_assets=True)

    # 1. MSH_Reader bulk load correctness
    msh_path = os.path.join(work, "tet.msh")
    _write_simple_msh(msh_path)
    obj_path = os.path.join(work, "obs.obj")
    _write_simple_obj(obj_path)
    geo_container = hou.node("/obj").createNode("geo", "p3_msh")
    reader = geo_container.createNode("MSH_Reader::3.0", "msh")
    reader.parm("File").set(msh_path)
    reader.cook(force=True)
    msh_geo = reader.geometry()
    assert len(msh_geo.points()) == 4
    assert len(msh_geo.prims()) == 1
    assert msh_geo.pointIntAttribValues("msh_pt_id") == (1, 2, 3, 4)
    print("PASS: MSH_Reader bulk NumPy createPoints + msh_pt_id upload")

    # 2. PolyFEM O(N) pure-NumPy centroid matcher (without scipy)
    pf = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "p3_pf")
    mod = pf.hdaModule()
    rng = np.random.default_rng(42)
    src = rng.uniform(-5.0, 5.0, size=(5000, 3))
    perm = rng.permutation(len(src))
    tgt = src[perm] + rng.normal(scale=1e-12, size=src.shape)
    t0 = time.perf_counter()
    dists, order = mod._nearest_centroids_numpy(src, tgt)
    elapsed = time.perf_counter() - t0
    assert np.array_equal(order, perm), "spatial hash mismatch on 5000 centroids"
    assert float(dists.max()) < 1e-10
    assert elapsed < 2.0, f"spatial hash too slow: {elapsed:.2f}s"
    # Mismatched target far from source triggers chunked fallback accurately
    far_tgt = np.array([[100.0, 100.0, 100.0]])
    d_far, i_far = mod._nearest_centroids_numpy(src, far_tgt)
    exact_d = np.linalg.norm(src - far_tgt[0], axis=1)
    assert int(i_far[0]) == int(exact_d.argmin())
    assert abs(float(d_far[0]) - float(exact_d.min())) < 1e-9
    print(f"PASS: _nearest_centroids_numpy 5000 centroids in {elapsed*1e3:.1f}ms")

    # 3. PolyFEM Write shared pre-flight cache (no redundant sideset/pe/obstacle calls)
    pf_work = os.path.join(work, "pf_run")
    os.makedirs(pf_work, exist_ok=True)
    pf.setParms({"working_dir": pf_work + "/", "file_location1": msh_path})
    pf.parm("file_location1").pressButton()
    # Add a fiber material so check_setup also validates per_element_data
    pf.setParms({
        "materials1_1": mod.MATERIAL_TOKENS.index("HGOFiber"),
        "fib_source1_1": "cylindrical",
    })
    # Add a second geometry as an obstacle
    pf.parm("num_geos").set(2)
    pf.setParms({"is_obstacle2": 1, "file_location2": obj_path})
    pf.parm("file_location2").pressButton()

    counts = {"sidesets": 0, "pe": 0, "obstacle": 0}
    pf_globals = mod._prepare_inputs.__globals__
    orig_rs = pf_globals["resolve_sidesets"]
    orig_pe = pf_globals["per_element_data"]
    orig_obs = pf_globals["obstacle_surface"]

    def wrap_rs(*a, **kw):
        counts["sidesets"] += 1
        return orig_rs(*a, **kw)

    def wrap_pe(*a, **kw):
        counts["pe"] += 1
        return orig_pe(*a, **kw)

    def wrap_obs(*a, **kw):
        counts["obstacle"] += 1
        return orig_obs(*a, **kw)

    pf_globals["resolve_sidesets"] = wrap_rs
    pf_globals["per_element_data"] = wrap_pe
    pf_globals["obstacle_surface"] = wrap_obs
    try:
        params_path = mod._prepare_inputs(pf, "write")
    finally:
        pf_globals["resolve_sidesets"] = orig_rs
        pf_globals["per_element_data"] = orig_pe
        pf_globals["obstacle_surface"] = orig_obs

    assert params_path is not None and os.path.isfile(params_path)
    assert counts["sidesets"] == 1, f"resolve_sidesets ran {counts['sidesets']}x"
    assert counts["pe"] == 1, f"per_element_data ran {counts['pe']}x"
    assert counts["obstacle"] == 1, f"obstacle_surface ran {counts['obstacle']}x"
    print("PASS: PolyFEM _prepare_inputs shares sideset/per-element/obstacle cache (1x each)")


def main():
    work = tempfile.mkdtemp(prefix="patch3_perf_")
    try:
        test_vtu_field_info_mmap_and_skip(work)
        test_readpvd_patch3(work)
        test_polyfem_and_msh_patch3(work)
        print("ALL PATCH 3 TESTS PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
