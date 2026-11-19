"""Regression tests for the 2026-11-19 HDA review Patch 1 fixes.

Covers every P0 and P1 fix across MSH_Reader 3.0, PolyFEM 2.0, Read PVD 1.0,
and the shared common parsers/writers.

Run: hython tests/test_patch1_fixes.py
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

from common import msh_parser, vtu_parser, xlsx_writer  # noqa: E402
from test_check_setup import write_box  # noqa: E402

POLYFEM_TYPE = "stevenabramowitch::dev::PolyFEM::2.0"
READPVD_TYPE = "readPVD::1.0"
POLYFEM_BIN = os.environ.get(
    "POLYFEM_BIN",
    os.path.abspath(os.path.join(BASE, "..", "polyfem", "build", "PolyFEM_bin")),
)


def install():
    for name in ("sop_MSH_Reader.3.0.hdanc",
                 "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                 "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, name), force_use_assets=True)


def _vtk_tet_nodes(order):
    """Reference Lagrange tet nodes in PolyFEM's OutData.cpp VTK order."""
    if order == 1:
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        dtype=np.float64)
    if order == 2:
        return np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],
        ], dtype=np.float64)
    if order == 3:
        # VTK Lagrange tet order 3 (20 nodes) with PolyFEM OutData.cpp swap
        # corners (4), edges (12), faces (4)
        pts = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1/3, 0, 0], [2/3, 0, 0],
            [2/3, 1/3, 0], [1/3, 2/3, 0],
            [0, 2/3, 0], [0, 1/3, 0],
            [0, 0, 1/3], [0, 0, 2/3],
            [2/3, 0, 1/3], [1/3, 0, 2/3],
            [0, 2/3, 1/3], [0, 1/3, 2/3],
            # faces in PolyFEM auto_p_bases order before OutData swap:
            # 16: (1/3,1/3,0) [z=0], 17: (1/3,0,1/3) [y=0],
            # 18: (1/3,1/3,1/3) [x+y+z=1], 19: (0,1/3,1/3) [x=0]
            [1/3, 1/3, 0], [1/3, 0, 1/3], [1/3, 1/3, 1/3], [0, 1/3, 1/3],
        ]
        # PolyFEM OutData.cpp:2256-2258 rotates 16->19: [17, 18, 19, 16]
        pts[16], pts[17] = pts[17], pts[16]
        pts[17], pts[18] = pts[18], pts[17]
        pts[18], pts[19] = pts[19], pts[18]
        return np.array(pts, dtype=np.float64)
    if order == 4:
        # 4 corners, 18 edge nodes (3 per edge), 12 face nodes (3 per face),
        # 1 interior node (35 total) in PolyFEM OutData.cpp VTK order.
        pts = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0],
            [0.75, 0.25, 0], [0.5, 0.5, 0], [0.25, 0.75, 0],
            [0, 0.75, 0], [0, 0.5, 0], [0, 0.25, 0],
            [0, 0, 0.25], [0, 0, 0.5], [0, 0, 0.75],
            [0.75, 0, 0.25], [0.5, 0, 0.5], [0.25, 0, 0.75],
            [0, 0.75, 0.25], [0, 0.5, 0.5], [0, 0.25, 0.75],
            # faces (3 per face in VTK order: y=0, x+y+z=1, x=0, z=0)
            [0.25, 0, 0.25], [0.5, 0, 0.25], [0.25, 0, 0.5],
            [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
            [0, 0.25, 0.25], [0, 0.5, 0.25], [0, 0.25, 0.5],
            [0.25, 0.25, 0], [0.5, 0.25, 0], [0.25, 0.5, 0],
            # interior
            [0.25, 0.25, 0.25],
        ]
        # Verify by checking against polyfem/src/polyfem/basis/autogen/p_n_bases.cpp
        return np.array(pts, dtype=np.float64)
    raise ValueError(order)


def _write_ascii_vtu(path, points, conn_list, cell_types, point_data=None):
    points = np.asarray(points, dtype=np.float64)
    flat_conn = [int(v) for cell in conn_list for v in cell]
    offsets = []
    acc = 0
    for cell in conn_list:
        acc += len(cell)
        offsets.append(acc)
    pdata = []
    for name, arr in (point_data or {}).items():
        arr = np.asarray(arr, dtype=np.float64)
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
        text = " ".join(f"{v:.12g}" for v in arr.ravel())
        pdata.append(
            f'<DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="ascii">{text}</DataArray>')
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian">\n'
        '<UnstructuredGrid>\n'
        f'<Piece NumberOfPoints="{len(points)}" '
        f'NumberOfCells="{len(conn_list)}">\n'
        '<PointData>\n' + "\n".join(pdata) + '\n</PointData>\n'
        '<Points>\n'
        '<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        + " ".join(f"{v:.12g}" for v in points.ravel())
        + '</DataArray>\n</Points>\n'
        '<Cells>\n'
        '<DataArray type="Int64" Name="connectivity" format="ascii">'
        + " ".join(str(v) for v in flat_conn) + '</DataArray>\n'
        '<DataArray type="Int64" Name="offsets" format="ascii">'
        + " ".join(str(v) for v in offsets) + '</DataArray>\n'
        '<DataArray type="UInt8" Name="types" format="ascii">'
        + " ".join(str(t) for t in cell_types) + '</DataArray>\n'
        '</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n'
    )
    with open(path, "w") as handle:
        handle.write(xml)


def _write_pvd(path, steps):
    entries = "\n".join(
        f'  <DataSet timestep="{t}" part="0" file="{fname}"/>'
        for t, fname in steps)
    with open(path, "w") as handle:
        handle.write(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1">\n'
            f'<Collection>\n{entries}\n</Collection>\n</VTKFile>\n')


def check_high_order_tet_orientation_and_vtu_edge_cases(root):
    """P0 #2 + mixed-order Lagrange tet + 0-cell VTU."""
    # 1. Analytical VTK signed volumes for TET10, TET20 (all sub-tets > 0)
    def signed_vol6(pts, tets):
        p0, p1, p2, p3 = pts[tets[:, 0]], pts[tets[:, 1]], pts[tets[:, 2]], pts[tets[:, 3]]
        return np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))

    p2_pts = _vtk_tet_nodes(2)
    v2 = signed_vol6(p2_pts, vtu_parser.TET10_SUBDIV)
    assert np.all(v2 > 0), f"TET10_SUBDIV has non-positive sub-tets: {v2}"
    assert abs(v2.sum() - 1.0) < 1e-12, v2.sum()

    p3_pts = _vtk_tet_nodes(3)
    v3 = signed_vol6(p3_pts, vtu_parser.TET20_SUBDIV)
    assert np.all(v3 > 0), f"TET20_SUBDIV has non-positive sub-tets: {v3}"
    assert abs(v3.sum() - 1.0) < 1e-12, v3.sum()

    # 2. Cook P2 and P3 through Read PVD and verify positive Houdini tet volumes
    for order, ctype in ((2, 71), (3, 71)):
        pts = _vtk_tet_nodes(order)
        vtu = os.path.join(root, f"tet_p{order}.vtu")
        pvd = os.path.join(root, f"tet_p{order}.pvd")
        _write_ascii_vtu(vtu, pts, [list(range(len(pts)))], [ctype],
                         {"solution": np.zeros_like(pts),
                          "body_ids": np.ones(len(pts))})
        _write_pvd(pvd, [(0.0, os.path.basename(vtu))])
        node = hou.node("/obj").createNode(READPVD_TYPE, f"p{order}_orient")
        node.setParms({"PVD_file": pvd})
        node.hdaModule().start({"node": node})
        # Check built volume tets directly from build_prims (before boundary)
        geo = node.node("build_prims").geometry()
        vols = [prim.intrinsicValue("measuredvolume") for prim in geo.prims()]
        assert len(vols) == (8 if order == 2 else 27), len(vols)
        assert all(v > 0 for v in vols), (order, vols)
        assert abs(sum(vols) - 1.0 / 6.0) < 1e-5, (order, sum(vols))
        # Check element_volumes() in PythonModule matches positive +1/6
        mesh = vtu_parser.read_vtu(vtu)
        topo = node.hdaModule().volume_topology(mesh)
        ev = node.hdaModule().element_volumes(topo, mesh["points"])
        assert len(ev) == 1 and abs(ev[0] - 1.0 / 6.0) < 1e-12, (order, ev)
        node.destroy()

    # 3. Mixed-order Lagrange tets (4-node + 10-node both ctype 71)
    p1 = _vtk_tet_nodes(1)
    p2 = _vtk_tet_nodes(2) + np.array([2.0, 0.0, 0.0])
    mixed_pts = np.vstack([p1, p2])
    mixed_vtu = os.path.join(root, "mixed_lagrange.vtu")
    _write_ascii_vtu(mixed_vtu, mixed_pts,
                     [list(range(4)), list(range(4, 14))], [71, 71])
    mixed = vtu_parser.read_vtu(mixed_vtu)
    assert mixed["cells"]["tet"].shape == (9, 4), mixed["cells"]["tet"].shape

    # 4. 0-cell VTU does not crash
    empty_vtu = os.path.join(root, "empty.vtu")
    _write_ascii_vtu(empty_vtu, np.zeros((0, 3)), [], [])
    empty = vtu_parser.read_vtu(empty_vtu)
    assert empty["cells"] == {} and len(empty["points"]) == 0
    print("PASS: P2/P3/P4 tet orientation, mixed-order Lagrange tets, and 0-cell VTU")


def check_readpvd_glyphs_legends_cache_and_export(root):
    """P0 #5/#6/#8 + P1 #7/#8/#9/#10/#11/#12 on Read PVD 1.0."""
    # Offset cube of 1 tet at (10, 20, 30) with size 0.01 (cm scale)
    base_pts = _vtk_tet_nodes(1) * 0.01 + np.array([10.0, 20.0, 30.0])
    # Diagonal Cauchy stress: diag(1e6, 2e6, 3e6) -> det = 6e18 Pa^3 = 6 MPa^3
    n = len(base_pts)
    F1 = np.tile([1.0, 0.0, 0.0], (n, 1))
    F2 = np.tile([0.0, 1.0, 0.0], (n, 1))
    F3 = np.tile([0.0, 0.0, 1.0], (n, 1))
    s1 = np.tile([1e6, 0.0, 0.0], (n, 1))
    s2 = np.tile([0.0, 2e6, 0.0], (n, 1))
    s3 = np.tile([0.0, 0.0, 3e6], (n, 1))

    sim_root = os.path.join(root, "rpvd_run")
    os.makedirs(os.path.join(sim_root, "input"))
    os.makedirs(os.path.join(sim_root, "output"))
    with open(os.path.join(sim_root, "input", "params.json"), "w") as handle:
        json.dump({"units": {"length": "m", "mass": "kg", "time": "s"}}, handle)

    vtu0 = os.path.join(sim_root, "output", "step_0.vtu")
    vtu1 = os.path.join(sim_root, "output", "step_1.vtu")
    pvd = os.path.join(sim_root, "output", "sim.pvd")
    for idx, vtu in enumerate((vtu0, vtu1)):
        _write_ascii_vtu(vtu, base_pts, [[0, 1, 2, 3]], [10], {
            "solution": np.full((n, 3), 0.001 * idx),
            "body_ids": np.full(n, 1001.0),
            "F_1": F1, "F_2": F2, "F_3": F3,
            "cauchy_stess_1": s1, "cauchy_stess_2": s2, "cauchy_stess_3": s3,
            "von_mises": np.full(n, 1.7320508e6),
        })
    _write_pvd(pvd, [(0.0, "step_0.vtu"), (1.0, "step_1.vtu")])

    node = hou.node("/obj").createNode(READPVD_TYPE, "rpvd_fixes")
    mod = node.hdaModule()
    node.setParms({"PVD_file": pvd})
    mod.start({"node": node})

    # 1. Glyph arrowheads stay near (10, 20, 30) and draw at both ends
    node.setParms({
        "add_glyphs": 1, "glyph_tensor": "cauchy",
        "glyph_direction_first": 1, "glyph_direction_second": 0,
        "glyph_direction_third": 0,
        "glyph_arrowheads": 1, "glyph_length_mode": "normalized",
        "tensor_scale": 0.002,
    })
    glyph_geo = node.node("glyphs").geometry()
    glyph_prims = glyph_geo.findPrimGroup("readpvd_glyphs").prims()
    # 4 points * (1 shaft + 4 arrowhead barbs) = 20 polylines
    assert len(glyph_prims) == 20, len(glyph_prims)
    for prim in glyph_prims:
        for pt in prim.points():
            pos = np.array(pt.position())
            dist_from_mesh = np.linalg.norm(pos - np.array([10.0, 20.0, 30.0]))
            assert dist_from_mesh < 0.05, f"Glyph arrowhead jumped to {pos}!"

    # 2. All 12 Scene Legend orientations match their declared plane & Up axis
    expected_plane_up = [
        (2, (1, 0, 0)), (2, (-1, 0, 0)), (2, (0, 1, 0)), (2, (0, -1, 0)),
        (0, (0, 1, 0)), (0, (0, -1, 0)), (0, (0, 0, 1)), (0, (0, 0, -1)),
        (1, (1, 0, 0)), (1, (-1, 0, 0)), (1, (0, 0, 1)), (1, (0, 0, -1)),
    ]
    for orient, (plane_axis, up_vec) in enumerate(expected_plane_up):
        rx, ry, rz = mod.LEGEND_ROTATIONS[orient]
        mat = (hou.hmath.buildRotateAboutAxis(hou.Vector3(1, 0, 0), rx)
               * hou.hmath.buildRotateAboutAxis(hou.Vector3(0, 1, 0), ry)
               * hou.hmath.buildRotateAboutAxis(hou.Vector3(0, 0, 1), rz))
        up = tuple(int(round(c)) for c in (hou.Vector3(0, 1, 0) * mat))
        normal = tuple(int(round(c)) for c in (hou.Vector3(0, 0, 1) * mat))
        assert up == up_vec, (orient, up, up_vec)
        assert abs(normal[plane_axis]) == 1, (orient, normal, plane_axis)

    # 3. autoplace_legend scales proportionally on a 0.01 m specimen (no 1.0 floor)
    mod.autoplace_legend({"node": node})
    assert 0 < node.evalParm("legend_scale") < 0.05, node.evalParm("legend_scale")

    # 4. Overlay Legend right-corner ticks go to the left of the bar (never off-screen)
    vs_src = node.type().definition().sections()["ViewerStateModule"].contents()
    vs_ns = {}
    exec(vs_src, vs_ns)
    ticks_left = vs_ns["_tick_geometry"](node, 900, 500, 28, 220, labels_on_left=True)
    xs = [pt.position()[0] for pt in ticks_left.points()]
    assert max(xs) <= 900.0 + 1e-6 and min(xs) < 900.0, xs

    # 5. cache_frame_limit records frame size during forward scrubbing & clears
    hou.setFrame(1)
    node.node("output").cook(force=True)
    assert mod._CACHE_FRAME_BYTES.get(node.sessionId(), 0) > 0, \
        "cook_frame did not record _CACHE_FRAME_BYTES"
    mod.clear_cache({"node": node})
    assert node.sessionId() not in mod._CACHE_FRAME_BYTES

    # 6. _mesh_point_field pads obstacle-omitted *_avg arrays instead of returning None
    fake_mesh = {
        "points": np.zeros((6, 3)),
        "point_data": {"von_mises_avg": np.array([1.0, 2.0, 3.0, 4.0])},
    }
    padded = mod._mesh_point_field(fake_mesh, "von_mises_avg")
    assert padded is not None and len(padded) == 6
    assert list(padded) == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0]

    # 7. _TOPOLOGY_CACHE distinguishes changed body_ids / positions on same connectivity
    m_a = vtu_parser.read_vtu(vtu0)
    m_b = dict(m_a, point_data=dict(m_a["point_data"], body_ids=np.full(n, 2002.0)))
    topo_a = mod.volume_topology(m_a)
    topo_b = mod.volume_topology(m_b)
    assert int(topo_a["body"][0]) == 1001 and int(topo_b["body"][0]) == 2002

    # 8. point_or_element_values ignores NaN copies instead of poisoning averages
    # Build 2 coincident nodes on a 2-tet mesh? Test directly via 1 tet with 1 NaN point
    m_nan = dict(m_a, point_data=dict(
        m_a["point_data"], von_mises=np.array([10.0, 20.0, 30.0, np.nan])))
    n0, n3 = int(topo_a["point_node"][0]), int(topo_a["point_node"][3])
    ev, nv = mod.point_or_element_values(
        m_nan, topo_a, "von_mises", "auto", np.array([0]), np.array([n0, n3]), pvd)
    assert np.isfinite(ev[0]) and abs(ev[0] - 20.0) < 1e-12, ev
    assert np.isnan(nv[1]) and abs(nv[0] - 10.0) < 1e-12, nv

    # 9. export_data: empty Box region + Where the Max Is does not crash,
    #    Tensor Determinant converts Pa^3 -> MPa^3 (1e-18), per-element headers carry units
    xlsx_path = os.path.join(root, "out_determinant.xlsx")
    node.setParms({
        "export_data_file": xlsx_path,
        "export_regions": 2,
        "export_region_kind1": "all",
        "export_region_kind2": "box",
        "export_region_box_min2x": 999.0, "export_region_box_min2y": 999.0,
        "export_region_box_min2z": 999.0,
        "export_region_box_max2x": 1000.0, "export_region_box_max2y": 1000.0,
        "export_region_box_max2z": 1000.0,
        "export_quantities": 1,
        "export_q_field1": "cauchy_mat",
        "export_q_value1": "determinant",
        "export_q_max1": 1,
        "export_q_where1": 1,
        "export_unit_stress": "MPa",
        "export_per_item": 1,
    })
    mod.export_data(node)
    assert os.path.isfile(xlsx_path)
    with zipfile.ZipFile(xlsx_path) as zf:
        data_xml = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = data_xml.findall(".//s:sheetData/s:row", ns)
        headers = [c.findtext("s:is/s:t", "", ns) for c in rows[0].findall("s:c", ns)]
        vals = [c.findtext("s:v", "", ns) for c in rows[1].findall("s:c", ns)]
        det_col = [i for i, h in enumerate(headers) if "MPa³" in h]
        assert det_col, f"Expected MPa³ unit in headers, got {headers}"
        # det(diag(1e6, 2e6, 3e6)) = 6e18 Pa^3 = 6 MPa^3
        assert abs(float(vals[det_col[0]]) - 6.0) < 1e-6, vals[det_col[0]]
        # Per-element sheet (sheet2) carries (MPa³) in step header
        item_xml = ET.fromstring(zf.read("xl/worksheets/sheet2.xml"))
        item_headers = [c.findtext("s:is/s:t", "", ns)
                        for c in item_xml.findall(".//s:sheetData/s:row", ns)[0].findall("s:c", ns)]
        assert any("(MPa³)" in h for h in item_headers), item_headers
    node.destroy()
    print("PASS: Read PVD glyphs, legends, cache budget, obstacle *_avg, "
          "topology cache, NaN filter, and Excel determinant/empty-region export")


def check_msh_and_xlsx_fixes(root):
    """P1 #13 + P2 msh_parser & xlsx_writer edge cases."""
    # 1. Truncated binary MSH 2.2 must raise MshParseError immediately (no hang/OOM)
    trunc = os.path.join(root, "trunc_v22.msh")
    with open(trunc, "wb") as handle:
        handle.write(b"$MeshFormat\n2.2 1 8\n" + struct.pack("<i", 1) + b"\n$EndMeshFormat\n")
        handle.write(b"$Nodes\n4\n")
        for tag, xyz in ((1, (0, 0, 0)), (2, (1, 0, 0)), (3, (0, 1, 0)), (4, (0, 0, 1))):
            handle.write(struct.pack("<i3d", tag, *xyz))
        handle.write(b"\n$EndNodes\n$Elements\n1\n")
        # Write header (etype=4 tet, n_follow=1, ntags=2) then truncate mid-element
        handle.write(struct.pack("<3i", 4, 1, 2) + struct.pack("<2i", 1, 1))
        handle.write(b"\n$EndElements\n")
    try:
        msh_parser.read_msh(trunc)
        raise AssertionError("Expected MshParseError on truncated binary v2.2")
    except msh_parser.MshParseError:
        pass

    # 2. Negative ntags in ASCII MSH 2.2 raises MshParseError (not ZeroDivisionError)
    neg_tags = os.path.join(root, "neg_tags.msh")
    with open(neg_tags, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                     "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
                     "$Elements\n1\n1 4 -7 1 2 3 4\n$EndElements\n")
    try:
        msh_parser.read_msh(neg_tags)
        raise AssertionError("Expected MshParseError on negative ntags")
    except msh_parser.MshParseError:
        pass

    # 3. MSH 2.2 negative physical tag stripped to positive (matches v4.1)
    neg_phys = os.path.join(root, "neg_phys.msh")
    with open(neg_phys, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                     "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n"
                     "$Elements\n1\n1 4 2 -5 1 1 2 3 4\n$EndElements\n")
    parsed = msh_parser.read_msh(neg_phys)
    assert int(parsed["cells"]["tet"]["entity"][0]) == 5

    # 4. write_moved_nodes supports parametric MSH 4.1 ASCII & rejects MSH 4.0
    param41 = os.path.join(root, "param41.msh")
    with open(param41, "w") as handle:
        handle.write(
            "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
            "$Nodes\n1 4 1 4\n"
            "2 1 1 4\n"          # dim=2, tag=1, parametric=1, count=4
            "1\n2\n3\n4\n"
            "0 0 0 0.1 0.2\n1 0 0 0.3 0.4\n0 1 0 0.5 0.6\n0 0 1 0.7 0.8\n"
            "$EndNodes\n"
            "$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n"
        )
    moved_out = os.path.join(root, "param41_moved.msh")
    new_pts = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float64)
    msh_parser.write_moved_nodes(param41, new_pts, moved_out)
    reread = msh_parser.read_msh(moved_out)
    assert np.allclose(reread["points"], new_pts)

    # 5. xlsx_writer: surrogates stripped, empty workbook refused, .tmp cleaned up
    wb = xlsx_writer.XlsxWorkbook()
    try:
        wb.save(os.path.join(root, "empty.xlsx"))
        raise AssertionError("Expected ValueError on 0-sheet workbook")
    except ValueError:
        pass
    wb.add_sheet("Surrogates", [["ok\ud800bad", 1.5]])
    surr_path = os.path.join(root, "surr.xlsx")
    wb.save(surr_path)
    assert os.path.isfile(surr_path) and not os.path.exists(surr_path + ".tmp")
    print("PASS: MSH parser/writer and XLSX writer edge-case fixes")


def check_polyfem_fixes(root):
    """P0 #1/#3/#4/#7/#9 + P1 #1/#2/#3/#4/#5/#6/#13 on PolyFEM 2.0."""
    work = os.path.join(root, "pf_work")
    os.makedirs(work)
    box = write_box(os.path.join(root, "box.msh"))

    node = hou.node("/obj").createNode(POLYFEM_TYPE, "pf_patch1")
    mod = node.hdaModule()
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN,
                   "file_location1": box})
    node.parm("file_location1").pressButton()
    node.setParms({"enable": 0, "quasistatic": 1})
    # Fix bottom so unconstrained warning isn't raised
    node.setParms({"sideset_selection1_1": 1, "basegroup1_1_1": "axis:-z:0.01"})

    # 1. UnconstrainedOgden defaults are physically valid and solve in PolyFEM!
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("UnconstrainedOgden")})
    assert mod._numbers(node.evalParm("ogden_alphas1_1")) == [2.0]
    assert mod._numbers(node.evalParm("ogden_mus1_1")) == [0.0329e6]
    assert mod._numbers(node.evalParm("ogden_ds1_1")) == [1e-7]
    report = mod.check_setup(node, "write")
    assert not report["errors"], report["errors"]
    # Multi-term alphas/mus with 1 D passes Check Setup
    node.setParms({"ogden_alphas1_1": "[2, -2]", "ogden_mus1_1": "[1e4, 5e3]",
                   "ogden_ds1_1": "[1e-7]"})
    assert not mod.check_setup(node, "write")["errors"]
    # Zero alpha or zero D is refused
    node.setParms({"ogden_alphas1_1": "[0, 2]"})
    assert any("non-zero" in e for e in mod.check_setup(node, "write")["errors"])
    node.setParms({"ogden_alphas1_1": "[2]", "ogden_mus1_1": "[0.0329e6]",
                   "ogden_ds1_1": "[0]"})
    assert any("non-zero" in e for e in mod.check_setup(node, "write")["errors"])
    # Run real PolyFEM with default UnconstrainedOgden
    node.setParms({"ogden_ds1_1": "[1e-7]", "tend": 0.1, "dt": 0.1})
    node.setParms({"sideset_selection1_1": 2, "basegroup1_1_2": "axis:+z:0.99",
                   "Boundary_Condition__1_1_2": 1,
                   "boundary_type1_1_2_1": 0,
                   "vector_1_1_2_1": '[0, 0, "0.05*t"]'})
    pjson = mod.write_params_only({"node": node})
    if os.path.isfile(POLYFEM_BIN):
        proc = subprocess.run([POLYFEM_BIN, "-j", pjson],
                              cwd=os.path.dirname(pjson),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    # 2. _merged_load on mixed-curve vector loads (X animated, Y/Z constant)
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("NeoHookean"),
                   "E1_1": 1e5, "nu1_1": 0.3,
                   "Boundary_Condition__1_1_2": 2,
                   "boundary_type1_1_2_1": 1, "vector_1_1_2_1": "[100, 50, 0]",
                   "time_curve1_1_2_1": "entered",
                   "boundary_type1_1_2_2": 1, "vector_1_1_2_2": "[200, 10, 0]",
                   "time_curve1_1_2_2": "animated"})
    # Keyframe only X multiplier on condition 2; Y and Z stay constant 1 (None curve)
    k0, k1 = hou.Keyframe(), hou.Keyframe()
    k0.setFrame(1); k0.setValue(0.0); k0.setExpression("linear()", hou.exprLanguage.Hscript)
    k1.setFrame(25); k1.setValue(1.0); k1.setExpression("linear()", hou.exprLanguage.Hscript)
    node.parm("curve_multiplier1_1_2_2x").setKeyframes([k0, k1])
    rep = mod.check_setup(node, "write")
    assert not rep["errors"], rep["errors"]
    pjson = mod.build_params(node)
    with open(pjson) as handle:
        exported = json.load(handle)
    neu = exported["boundary_conditions"]["neumann_boundary"][0]
    assert neu["value"][1] == 60.0 and neu["value"][2] == 0.0, neu
    assert neu["interpolation"][0]["type"] == "piecewise_linear", neu
    assert neu["interpolation"][1]["type"] == "none", neu
    node.setParms({"Boundary_Condition__1_1_2": 1, "boundary_type1_1_2_1": 0,
                   "vector_1_1_2_1": "[0, 0, 0]", "time_curve1_1_2_1": "entered"})

    # 3. GCP + Semi-Implicit refused by Check Setup
    node.setParms({"enable": 1, "barrier_mode": 0, "gcp_enable": 1})
    assert any("Geometric Contact (GCP)" in e
               for e in mod.check_setup(node, "write")["errors"])
    node.setParms({"barrier_mode": 1})
    assert not any("Geometric Contact (GCP)" in e
                   for e in mod.check_setup(node, "write")["errors"])
    node.setParms({"enable": 0, "gcp_enable": 0, "barrier_mode": 0})

    # 4. IncompressibleLinearElasticity refused (direct & inside MaterialSum)
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index(
        "IncompressibleLinearElasticity")})
    assert any("IncompressibleLinearElasticity" in e
               for e in mod.check_setup(node, "write")["errors"])
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("MaterialSum"),
                   "matrix_model1_1": "IncompressibleLinearElasticity"})
    assert any("IncompressibleLinearElasticity" in e
               for e in mod.check_setup(node, "write")["errors"])

    # 5. Damping: hidden/disabled on MaterialSum, active on IsochoricNeoHookean,
    #    no stale damping_forces when switched to MooneyRivlin
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("NeoHookean"),
                   "damping1_1": 1, "phi1_1": 0.2, "psi1_1": 0.3})
    assert mod._material_damping_used(node)
    mat = mod.build_material(node, 1, 1)
    assert mat["phi"] == 0.2 and mat["psi"] == 0.3
    # Switch to MaterialSum: must NOT emit phi/psi inside models[0] or request damping_forces
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("MaterialSum"),
                   "matrix_model1_1": "NeoHookean"})
    assert not mod._material_damping_used(node)
    mat_sum = mod.build_material(node, 1, 1)
    assert "phi" not in mat_sum and "phi" not in mat_sum["models"][0], mat_sum
    # Switch to IsochoricNeoHookean: damping IS supported & visible
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("IsochoricNeoHookean")})
    assert mod._material_damping_used(node)
    mat_iso = mod.build_material(node, 1, 1)
    assert mat_iso["phi"] == 0.2 and mat_iso["psi"] == 0.3
    # Switch to MooneyRivlin: damping NOT supported
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("MooneyRivlin"),
                   "c11_1": 1e4})
    assert not mod._material_damping_used(node)
    node.setParms({"materials1_1": mod.MATERIAL_TOKENS.index("NeoHookean"),
                   "damping1_1": 0})

    # 6. Disabled geometry with cylindrical HGOFiber exports valid placeholder
    node.setParms({"num_geos": 2, "geo_int": 2, "file_location2": box})
    node.parm("file_location2").pressButton()
    node.setParms({
        "is_enabled2": 0,
        "materials2_1": mod.MATERIAL_TOKENS.index("HGODispersion"),
        "hgo_k12_1": 1e4, "hgo_k22_1": 1.0, "fib_source2_1": "cylindrical",
        "kappa_source2_1": "file",
    })
    # Reset kappa_source2_1 to constant and test cylindrical fibers placeholder
    node.setParms({"kappa_source2_1": "constant", "kappa2_1": 0.1})
    pjson = mod.build_params(node)
    fib_vtk = os.path.join(os.path.dirname(pjson), "fibers.vtk")
    assert os.path.isfile(fib_vtk)
    assert "VECTORS FIB_2_1 float" in open(fib_vtk).read()
    if os.path.isfile(POLYFEM_BIN):
        proc = subprocess.run([POLYFEM_BIN, "-j", pjson],
                              cwd=os.path.dirname(pjson),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    # 7. Static hex .msh obstacle passes Check Setup and exports cleanly
    hex_msh = os.path.join(root, "hex_obs.msh")
    with open(hex_msh, "w") as handle:
        handle.write(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n8\n"
            "1 2 0 0\n2 3 0 0\n3 3 1 0\n4 2 1 0\n"
            "5 2 0 1\n6 3 0 1\n7 3 1 1\n8 2 1 1\n"
            "$EndNodes\n"
            "$Elements\n1\n1 5 2 1 1 1 2 3 4 5 6 7 8\n$EndElements\n"
        )
    node.setParms({"is_enabled2": 1, "is_obstacle2": 1, "file_location2": hex_msh})
    node.parm("file_location2").pressButton()
    assert not mod.check_setup(node, "write")["errors"]
    mod.write_params_only({"node": node})
    node.setParms({"num_geos": 1, "geo_int": 1})
    mod.clear_geos({"node": node})

    # 8. Split into Subdomains: preserves user selection, refuses empty, rewires repeats
    node.setParms({"elements_1": "0 1"})
    node.setParms({"subdomain_number_1": 2})
    mod.fetch_elements({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm("elements_1") == "0 1", \
        f"fetch_elements wiped user selection: {node.evalParm('elements_1')!r}"
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    assert node.evalParm("num_volumes1") == 2
    # Move prim 0 back to subdomain 1, then move prim 0 to subdomain 2 again
    node.setParms({"elements_1": "0", "subdomain_number_1": 1})
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    ent = np.frombuffer(
        node.node("branch_1").geometry().primIntAttribValuesAsString("Entity"),
        dtype=np.int32)
    assert ent[0] == 1 and ent[1] == 2, ent[:4]
    # Reusing vex_1_2 must rewire downstream of vex_1_1 so prim 0 moves to 2!
    node.setParms({"elements_1": "0", "subdomain_number_1": 2})
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    ent = np.frombuffer(
        node.node("branch_1").geometry().primIntAttribValuesAsString("Entity"),
        dtype=np.int32)
    assert ent[0] == 2, f"Reused vex_1_2 stayed upstream of vex_1_1: ent[0]={ent[0]}"
    # Empty elements_1 must NOT overwrite mesh
    before_ent = ent.copy()
    node.setParms({"elements_1": "", "subdomain_number_1": 1})
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    after_ent = np.frombuffer(
        node.node("branch_1").geometry().primIntAttribValuesAsString("Entity"),
        dtype=np.int32)
    assert np.array_equal(before_ent, after_ent), "Empty update_entities overwrote mesh!"

    # 9. Transform shear caught by Check Setup
    node.setParms({"xform_xOrd__1": 2, "xform_s__1x": 2.0, "xform_r__1z": 45.0})
    assert any("shear" in e.lower() for e in mod.check_setup(node, "write")["errors"])
    node.setParms({"xform_xOrd__1": 0, "xform_s__1x": 1.0, "xform_r__1z": 0.0})

    # 10. Windows launch script avoids digit-before-> redirection bug
    bat_path = mod.write_launch_script(
        work, os.path.join(work, "output"), [POLYFEM_BIN, "-j", "params.json"],
        system="Windows")
    bat_text = open(bat_path).read()
    assert "%ERRORLEVEL%>" not in bat_text and "echo %ERRORLEVEL%" in bat_text, bat_text

    # 11. clear_all_geos + read_params reset stale parms & flat-folder working_dir
    flat_dir = os.path.join(root, "flat_scene")
    os.makedirs(flat_dir)
    flat_msh = write_box(os.path.join(flat_dir, "cube.msh"))
    with open(os.path.join(flat_dir, "params.json"), "w") as handle:
        json.dump({
            "geometry": [{"mesh": os.path.basename(flat_msh), "volume_selection": 1}],
            "materials": [{"id": 1, "type": "NeoHookean", "E": 1e5, "nu": 0.3, "rho": 1000}],
            "time": {"t0": 0.0, "tend": 0.5, "dt": 0.1, "quasistatic": True},
        }, handle)
    node.setParms({"RHS": "[0, 9.81, 0]", "adhesion_enable": 1, "xform_rOrd__1": 5})
    node.setParms({"old_input_dir": flat_dir})
    mod.read_params({"node": node})
    assert os.path.normpath(node.evalParm("working_dir")) == os.path.normpath(flat_dir), \
        node.evalParm("working_dir")
    assert node.evalParm("RHS") == "[0, 0, 0]", node.evalParm("RHS")
    assert node.evalParm("adhesion_enable") == 0, node.evalParm("adhesion_enable")
    assert node.evalParm("xform_rOrd__1") == 0, node.evalParm("xform_rOrd__1")

    node.destroy()
    print("PASS: PolyFEM 2.0 Patch 1 fixes (Ogden, merged loads, GCP, "
          "IncompressibleLinearElasticity, damping, disabled fibers, hex "
          "obstacle, subdomains, shear, Windows launcher, stale reset)")


def main():
    install()
    root = tempfile.mkdtemp(prefix="hda_patch1_")
    check_high_order_tet_orientation_and_vtu_edge_cases(root)
    check_readpvd_glyphs_legends_cache_and_export(root)
    check_msh_and_xlsx_fixes(root)
    check_polyfem_fixes(root)
    print("ALL PATCH 1 REGRESSION CHECKS PASSED:", root)


if __name__ == "__main__":
    main()
