"""Read PVD Batch 2 regressions using the rebuilt asset and small VTK fixtures.

Run with hython. Reference-node checks also use the sibling PolyFEM sources.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import itertools
import json
from pathlib import Path
import re
import struct
import sys
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
from xml.sax.saxutils import quoteattr
import zipfile
import zlib

import hou
import numpy as np

from test_export_data import export, read_xlsx, set_quantity, set_region, table

BASE = Path(__file__).resolve().parents[1]
POLYFEM = BASE.parent / "polyfem/src/polyfem"
TET = np.vstack((np.zeros(3), np.eye(3)))
HEX = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], dtype=float)


def check_raises(error_type, text, function, *args):
    try:
        function(*args)
    except error_type as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {text}")


def data_array(name, values, dtype="Float64", *, binary=False, info=False):
    arrays = {"Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}
    array = np.asarray(values, dtype=arrays[dtype])
    attr = f" Name={quoteattr(name)}" if name else ""
    if array.ndim == 2:
        attr += f" NumberOfComponents='{array.shape[1]}'"
    if binary:
        payload = base64.b64encode(struct.pack("<I", array.nbytes)
                                   + array.tobytes()).decode()
    else:
        payload = " ".join(map(str, array.ravel()))
    if info:
        payload += "<InformationKey name='L2_NORM_RANGE'><Value>1234</Value></InformationKey>"
    fmt = "binary" if binary else "ascii"
    return f"<DataArray type='{dtype}' format='{fmt}'{attr}>{payload}</DataArray>"


def piece(points, ctype=10, point_fields=None, cell_fields=None, conn=None):
    conn = list(range(len(points))) if conn is None else conn
    point_fields = point_fields or {}
    cell_fields = cell_fields or {}
    return (
        f"<Piece NumberOfPoints='{len(points)}' NumberOfCells='1'>"
        "<Points>" + data_array("", points) + "</Points><Cells>"
        + data_array("connectivity", conn, "Int64")
        + data_array("offsets", [len(conn)], "Int64")
        + data_array("types", [ctype], "UInt8")
        + "</Cells><PointData>"
        + "".join(data_array(name, arr) for name, arr in point_fields.items())
        + "</PointData><CellData>"
        + "".join(data_array(name, arr) for name, arr in cell_fields.items())
        + "</CellData></Piece>")


def write_vtu(path, *pieces, appended="", attrs=""):
    path.write_text(f"<VTKFile type='UnstructuredGrid' byte_order='LittleEndian' {attrs}>"
                    "<UnstructuredGrid>" + "".join(pieces) + "</UnstructuredGrid>"
                    + appended + "</VTKFile>")
    return path


def write_pvd(path, *children):
    path.write_text("<VTKFile type='Collection'><Collection>"
                    + "".join(f"<DataSet timestep='{i}' file={quoteattr(p.name)}/>"
                              for i, p in enumerate(children))
                    + "</Collection></VTKFile>")
    return path


def new_reader(name, pvd):
    hou.setFrame(0)
    node = hou.node("/obj").createNode("readPVD::1.0", name)
    node.setParms({"PVD_file": str(pvd), "time_mapping": "steps"})
    return node, node.hdaModule()


def module_hooks(mod, **hooks):
    # HDAModule is a HOM proxy. Patch the functions' Python globals directly.
    return patch.dict(mod._file_key.__globals__, hooks)


def polyfem_reference_nodes():
    """Read the actual generated coordinates and applied OutData permutations."""
    outdata = (POLYFEM / "io/OutData.cpp").read_text()
    build = outdata.split("void OutGeometryData::build_high_order_vis_mesh(", 1)[1]
    assert "autogen::p_nodes_3d(output_orders(i), ref_pts)" in build
    assert "autogen::q_nodes_3d(output_orders(i), ref_pts)" in build
    build = build.split("bs.eval_geom_mapping(ref_pts, mapped);", 1)[1]
    build = build.split("if (!error_msg.empty())", 1)[0]
    live_code = re.sub(r"//[^\n]*", "", build)
    swaps = r"std::swap\(elements\[i\]\.vertices\[(\d+)\], elements\[i\]\.vertices\[(\d+)\]\)"
    simplex = live_code.split("if (output_orders(i) >= 3)", 1)[1].split("}", 1)[0]
    cube = live_code.split("if (output_orders(i) == 2)", 1)[1].split("}", 1)[0]

    def nodes(family, order, permute=""):
        filename = "auto_p_bases.cpp" if family == "p" else "auto_q_bases_3d_nodes.cpp"
        source = (POLYFEM / "autogen" / filename).read_text()
        initializer = re.search(rf"void {family}_{order}_nodes_3d\([^)]*\)\s*{{"
                                r"\s*res.resize\(\d+,\s*3\);\s*res\s*<<([^;]+);", source)[1]
        result = np.array([float(n) / float(d) if sep else float(n)
                           for n, sep, d in (s.strip().partition("/")
                                             for s in initializer.split(","))]).reshape(-1, 3)
        for a, b in ((int(a), int(b)) for a, b in re.findall(swaps, permute)):
            result[[a, b]] = result[[b, a]]
        return result

    return nodes("p", 3, simplex), nodes("p", 4, simplex), nodes("q", 2, cube), nodes("q", 3)


def canonical_vtk_p4():
    # VTK's vtkHigherOrderTetra::BarycentricIndex orders vertices, edges,
    # then faces. The triangular face interiors follow the oriented corners.
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    faces = [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]
    quarters = [TET * 4]
    quarters += [np.array([(4 - j) * TET[a] + j * TET[b]
                          for j in (1, 2, 3)]) for a, b in edges]
    quarters += [TET[list(face)] + TET[list(face)].sum(axis=0) for face in faces]
    return np.vstack((*quarters, [[1, 1, 1]])) / 4


def test_subdivision_and_workbook(root, mod):
    p3, p4, q2, q3 = polyfem_reference_nodes()
    for order, coordinates, subdiv in ((3, p3, mod.TET20_SUBDIV),
                                       (4, p4, mod.TET35_SUBDIV)):
        cells = coordinates[subdiv]
        volume = np.linalg.det(cells[:, 1:] - cells[:, :1]) / 6
        assert len(cells) == order**3
        np.testing.assert_allclose(volume, 1 / (6 * order**3), rtol=1e-12)
        assert len(np.unique(subdiv)) == len(coordinates)
        faces = np.concatenate([np.sort(subdiv[:, face], axis=1)
                                for face in itertools.combinations(range(4), 3)])
        _, counts = np.unique(faces, axis=0, return_counts=True)
        assert np.count_nonzero(counts == 1) == 4 * order**2
        assert counts.max() == 2

    tri = q2.copy()
    tri[[18, 19]] = tri[[19, 18]]
    for order, ctype, coordinates, subdiv in (
            (2, 72, q2, mod.HEX27_SUBDIV), (3, 72, q3, mod.HEX64_SUBDIV),
            (2, 29, tri, mod.HEX27_TRIQUAD_SUBDIV)):
        cells = coordinates[subdiv]
        assert len(np.unique(cells[:, 0], axis=0)) == order**3
        np.testing.assert_allclose(cells - cells[:, :1],
                                   np.broadcast_to(HEX / order, cells.shape), atol=1e-15)
        moved = coordinates.copy()
        top = np.flatnonzero((coordinates[:, 2] == 1)
                             & np.all((coordinates[:, :2] > 0)
                                      & (coordinates[:, :2] < 1), axis=1))[0]
        moved[top, 2] += 0.12
        mesh = mod._mesh_from_arrays(coordinates, np.arange(len(coordinates)),
                                     [len(coordinates)], [ctype], {}, {})
        topology = mod.volume_topology(mesh)
        expected = 1 + 0.12 / order**2
        np.testing.assert_allclose(mod.element_volumes(topology, coordinates), [1])
        np.testing.assert_allclose(mod.element_volumes(topology, moved), [expected])

    mesh_file = write_vtu(root / "type29.vtu", piece(
        tri, 29, {"solution": moved - tri, "fiber_stress": np.full(len(tri), 7.25)}))
    node, _ = new_reader("type29", write_pvd(root / "type29.pvd", mesh_file))
    node.setParms({"export_regions": 1, "export_quantities": 3, "export_forces": 0})
    set_region(node, 1, "all", "Hex")
    set_quantity(node, 1, "_volume", volume=1)
    set_quantity(node, 2, "_volume", volume=0)
    set_quantity(node, 3, "fiber_stress", "auto", stats=("mean",))
    _, sheets = export(node)
    values = table(sheets["Data"])
    np.testing.assert_allclose(values["Hex: volume (rest)"], [1])
    np.testing.assert_allclose(values["Hex: volume"], [expected])
    np.testing.assert_allclose(values["Hex: fiber_stress, mean"], [7.25])
    node.destroy()

    canonical = canonical_vtk_p4()
    cells = canonical[mod.TET35_SUBDIV]
    assert np.count_nonzero(np.linalg.det(cells[:, 1:] - cells[:, :1]) <= 0) > 16
    for name, coordinates in (("polyfem_p4", p4), ("vtk_p4", canonical)):
        vtu = write_vtu(root / f"{name}.vtu", piece(coordinates, 71))
        node, _ = new_reader(name, write_pvd(root / f"{name}.pvd", vtu))
        node.node("frame_data").cook(force=True)
        assert not node.node("frame_data").errors()
        status = node.evalParm("availability_status")
        assert ("over 25%" in status) == (name == "vtk_p4"), status
        if name == "vtk_p4":
            assert "may indicate" in status and "severe initial mesh distortion" in status
        node.destroy()


def test_xml_and_hdf(root, mod):
    # Both XML readers must support quotes, escaped names and inline metadata.
    first = piece(TET, point_fields={"stress & strain": [4] * 4},
                  cell_fields={"quality": [11]}).replace(
        "<PointData>", "<PointData>" +
        data_array("inline", [1, 2, 3, 4], binary=True, info=True)
        + data_array("inline_ascii", [4, 3, 2, 1], info=True).replace(
            ">4.0", "><InformationKey name='leading_metadata'/>4.0", 1))
    second = piece(TET + 4, point_fields={"stress & strain": [8] * 4},
                   cell_fields={"quality": [22], "second_only": [30]})
    vtu = write_vtu(root / "pieces.vtu", first, second)
    assert mod.read_vtu_field_info(vtu) == {
        "point_data": {"inline": 1, "inline_ascii": 1, "stress & strain": 1},
        "cell_data": {"quality": 1, "second_only": 1},
    }
    mesh = mod.read_vtu(vtu)
    np.testing.assert_array_equal(mesh["cells"]["tet"], [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_array_equal(mesh["cell_starts"], [0, 4])
    np.testing.assert_equal(mesh["point_data"]["inline"], [1, 2, 3, 4] + [np.nan] * 4)
    np.testing.assert_equal(mesh["point_data"]["inline_ascii"][:4], [4, 3, 2, 1])
    np.testing.assert_equal(mesh["point_data"]["stress & strain"], [4] * 4 + [8] * 4)
    np.testing.assert_equal(mesh["cell_data"]["quality"]["tet"], [11, 22])
    np.testing.assert_equal(mesh["cell_data"]["second_only"]["tet"], [np.nan, 30])

    quad = write_vtu(root / "quad.vtu", piece(TET, 9))
    tet = write_vtu(root / "tet.vtu", piece(TET))
    assert mod.read_vtu(quad)["topo_key"] != mod.read_vtu(tet)["topo_key"]

    # The first encoding is a single stream whose byte count and one-byte
    # payload share a base64 quad. The second uses separate VTK streams.
    cell = piece(TET).replace("<CellData>", "<CellData>"
                             "<DataArray Name='flag' type='UInt8' format='appended' offset='0'/>")
    for encoded in ("AQAAABs=", "AQAAAA==Gw=="):
        appended = f"<AppendedData encoding='base64'>_{encoded}</AppendedData>"
        binary = write_vtu(root / "byte.vtu", cell, appended=appended)
        np.testing.assert_equal(mod.read_vtu(binary)["cell_data"]["flag"]["tet"], [27])
    bad_inline = base64.b64encode(struct.pack("<I", 32) + b"\0" * 16).decode()
    bad = (f"<DataArray Name='broken' type='Float64' format='binary'>"
           f"{bad_inline}</DataArray>")
    truncated = write_vtu(root / "truncated.vtu",
                          piece(TET).replace("<PointData>", "<PointData>" + bad))
    check_raises(mod.VtuParseError, "Truncated", mod.read_vtu, truncated)
    short_tuples = data_array("broken", [1, 2], binary=True).replace(
        'Name="broken"', "Name='broken' NumberOfTuples='4'")
    check_raises(mod.VtuParseError, "wrong tuple count", mod.read_vtu,
                 write_vtu(root / "short_tuples.vtu",
                           piece(TET).replace("<PointData>", "<PointData>" + short_tuples)))
    check_raises(mod.VtuParseError, "Truncated", mod.read_vtu,
                 write_vtu(root / "short_appended.vtu", cell,
                           appended="<AppendedData encoding='base64'>_AgAAAA==Gw==</AppendedData>"))
    compressed = zlib.compress(struct.pack("<4d", 2.0, 4.0, 6.0, 8.0))
    header = struct.pack("<4I", 1, 32, 32, len(compressed))
    for block in (base64.b64encode(header + compressed),
                  base64.b64encode(header) + base64.b64encode(compressed)):
        data = ("<DataArray Name='compressed' type='Float64' format='binary'>"
                + block.decode() + "</DataArray>")
        binary = write_vtu(root / "compressed.vtu",
                           piece(TET).replace("<PointData>", "<PointData>" + data),
                           attrs="compressor='vtkZLibDataCompressor'")
        np.testing.assert_equal(mod.read_vtu(binary)["point_data"]["compressed"], [2, 4, 6, 8])

    class TruncatingVTU(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= mod._VTU_SCAN_CHUNK
            chunk = super().read(size)
            if self.tell() >= mod._VTU_SCAN_CHUNK:
                self.truncate(self.tell())
            return chunk

    partial = (b"<VTKFile><UnstructuredGrid><Piece><PointData>"
               b"<DataArray Name='during_write' format='ascii'>"
               + b"123 " * mod._VTU_SCAN_CHUNK)
    with module_hooks(mod, open=lambda *args: TruncatingVTU(partial)):
        assert mod.read_vtu_field_info(tet)["point_data"] == {"during_write": 1}

    multiblock = root / "pieces.vtm"
    text = ("<VTKFile><vtkMultiBlockDataSet><Block name='Volume'>"
            "<DataSet file='tet.vtu'/><DataSet file='quad.vtu'/></Block>"
            "<Block><DataSet name='Surface' file='quad.vtu'/>"
            "<DataSet name='Volume' file='tet.vtu'/></Block>"
            "<DataSet name='Surface' file='tet.vtu'/>"
            "</vtkMultiBlockDataSet></VTKFile>")
    multiblock.write_text(text)
    with patch.object(mod.ET, "parse", wraps=mod.ET.parse) as parse:
        blocks = mod.read_vtm(multiblock)
        assert set(blocks) == {"Volume", "Volume_2", "Surface", "Volume_3", "Surface_2"}
        assert blocks["Volume_2"] == str(quad)
        assert mod.read_vtm(multiblock) is blocks and parse.call_count == 1
        multiblock.write_text(text.replace("Surface", "Contact data"))
        assert "Contact data_2" in mod.read_vtm(multiblock) and parse.call_count == 2

    hdf = root / "partitions.vtkhdf"
    with mod._import_h5py().File(hdf, "w") as handle:
        vtk = handle.create_group("VTKHDF")
        vtk.attrs["Type"] = "UnstructuredGrid"
        for name, values in {
                "Points": np.vstack((TET, TET + 4)), "Connectivity": [0, 1, 2, 3] * 2,
                "Offsets": [0, 4, 0, 0, 0, 4], "Types": [10, 0, 10],
                "NumberOfPoints": [4, 0, 4], "NumberOfCells": [1, 1, 1],
                "NumberOfConnectivityIds": [4, 0, 4], "PointData/marker": np.arange(8),
                "CellData/material": [3, 0, 9]}.items():
            vtk.create_dataset(name, data=values)
    assert mod.read_mesh_field_info(hdf)["point_data"] == {"marker": 1}
    pvd = write_pvd(root / "partitions.pvd", hdf)
    mesh = mod.load_block(str(pvd), 0, "Volume")
    np.testing.assert_equal(mesh["cell_starts"], [0, 4, 4])
    np.testing.assert_equal(mesh["cell_connectivity"], np.arange(8))
    np.testing.assert_equal(mesh["cell_data"]["material"]["tet"], [3, 9])


def test_fields_reference_and_options(root, mod):
    stress = [1.234567894567, 0, 0, 10]
    fields = {"a_b": [10] * 4, "a/b": [20] * 4, "P": TET + 100, "Cd": TET,
              "N": TET, "readpvd_point": [91] * 4, "readpvd_body": [99] * 4,
              "reference_comparison": [75] * 4, "stress": stress,
              "solution": np.zeros_like(TET), "vec2": [[3, 7]] * 4,
              "F_1": [[1, 0, 0]] * 4, "F_2": [[0, 1, 0]] * 4,
              "F_3": [[0, 0, 1]] * 4}
    n0 = write_vtu(root / "names0.vtu", piece(
        TET, point_fields=fields, cell_fields={"source_cell_index": [99]}))
    next_fields = dict(fields, stress=[stress[0], 2, 0, 9])
    n1 = write_vtu(root / "names1.vtu", piece(TET, point_fields=next_fields))
    n2 = write_vtu(root / "names2.vtu", piece(TET, point_fields=fields, conn=[0, 2, 1, 3]))
    n3 = write_vtu(root / "names3.vtu", piece(TET, point_fields={"stress": stress}))
    metadata = root / "names0.vtm"
    metadata.write_text("<VTKFile><vtkMultiBlockDataSet>"
                        "<DataSet name='Volume' file='names0.vtu'/>"
                        "<DataSet name='Surface' file='names0.vtu'/>"
                        "</vtkMultiBlockDataSet></VTKFile>")
    pvd = write_pvd(root / "names.pvd", metadata, n1, n2, n3)
    node, _ = new_reader("field_collisions", pvd)
    node.setParms({"field_time_scope": "any", "reference_enable": 1,
                   "reference_frame": 0, "color_attrib": "stress",
                   "remesh_mode": 1, "field_units": "kPa",
                   "export_folder": str(root)})
    mod.sync_available_options({"node": node})

    menu = node.parm("color_attrib").menuItems()
    assert mod._field_name("a_b") == "a_b" and mod._field_name("a/b") == "a_b_2"
    assert {"a_b", "a_b_2", mod._field_name("P")} <= set(menu)
    assert len({mod._field_name(name) for name in fields}) == len(fields)
    for name in ("P", "Cd", "N", "readpvd_point", "readpvd_body",
                 "source_cell_index", "reference_comparison"):
        mapped = mod._field_name(name)
        assert name != mapped and mod._display_name(mapped) == name

    frame = node.node("frame_data")
    frame.cook(force=True)
    geo = frame.geometry()
    np.testing.assert_equal(np.array([tuple(p.position()) for p in geo.points()]), TET)
    np.testing.assert_equal(geo.pointIntAttribValues("readpvd_point"), [0, 1, 2, 3])
    np.testing.assert_equal(geo.pointFloatAttribValues("a_b"), [10] * 4)
    np.testing.assert_equal(geo.pointFloatAttribValues("a_b_2"), [20] * 4)
    nprims = geo.intrinsicValue("primitivecount")
    assert nprims > 0
    np.testing.assert_equal(geo.primIntAttribValues("source_cell_index"), [0] * nprims)
    np.testing.assert_equal(geo.primFloatAttribValues(mod._field_name("source_cell_index")),
                            [99] * nprims)
    node.parm("color_attrib").set("a_b_2")
    assert "a/b: 20" in mod.probe_readout(node, 0)[1]
    node.setParms({"export_regions": 1, "export_quantities": 1, "export_forces": 0})
    set_region(node, 1, "all", "Tet")
    set_quantity(node, 1, "a_b_2", "auto", stats=("mean",))
    # Only step zero has the original topology and all chosen fields.
    node.setParms({"export_step_first": 0, "export_step_last": 0})
    _, sheets = export(node)
    assert table(sheets["Data"])["Tet: a/b, mean"] == [20]
    mesh0 = mod.read_vtu(n0)
    np.testing.assert_equal(mod._mesh_field(mesh0, "a_b_2"), [20] * 4)

    node.parm("color_attrib").set("stress")
    reference = node.node("reference_comparison")
    reference.cook(force=True)
    np.testing.assert_equal(reference.geometry().pointFloatAttribValues("reference_comparison"), [0] * 4)
    hou.setFrame(1)
    node.parm("reference_mode").set("percent")
    reference.cook(force=True)
    np.testing.assert_equal(reference.geometry().pointFloatAttribValues("reference_comparison"),
                            [0, np.nan, 0, -10])
    assert mod.legend_title_text(node).endswith("[%]")
    probe = mod.probe_readout(node, 1)[1].splitlines()
    reference_line = next(line for line in probe if "Reference Comparison" in line)
    assert reference_line.endswith("%") and "kPa" not in reference_line
    _, _, scan = next(mod._scan_frames(node, str(pvd), [1]))
    np.testing.assert_equal(scan, [0, np.nan, 0, -10])

    hou.setFrame(2)
    reference.cook(force=True)
    assert "matching point topology" in node.evalParm("reference_status")
    assert reference.geometry().findPointAttrib("reference_comparison") is None
    _, _, scan = next(mod._scan_frames(node, str(pvd), [2]))
    np.testing.assert_equal(scan, stress)
    original = n0.read_text()
    n0.write_text("<VTKFile><UnstructuredGrid><VTU_is_incomplete")
    try:
        reference.cook(force=True)
        status = node.evalParm("reference_status")
        assert status.startswith(f"Reference load failed: Invalid VTU file {n0}"), status
    finally:
        n0.write_text(original)

    assert mod._reduce_array(np.array([[3, 7], [8, 10]]), "z").dtype == np.float64
    np.testing.assert_equal(mod._reduce_array(mod._mesh_field(mesh0, "vec2"), "z"), [0] * 4)
    hou.setFrame(0)
    node.setParms({"reference_enable": 0, "color_attrib": "vec2", "color_reduction": "z"})
    node.node("color_reduce").cook(force=True)
    np.testing.assert_equal(node.node("color_reduce").geometry().pointFloatAttribValues("for_color"), [0] * 4)

    toggles = {"multi_surface_show": 1, "multi_contact_show": 1,
               "multi_points_show": 1, "show_deformed": 1, "add_glyphs": 1,
               "show_fibers": 1, "source_block": "Surface"}
    (root / "run-manifest.json").write_text('{"completion":{"status":"completed"}}')
    node.setParms(toggles)
    hou.setFrame(3)
    mod.sync_available_options({"node": node})
    assert {name: node.evalParm(name) for name in toggles} == toggles
    assert not any(node.evalParm(name) for name in (
        "has_block_surface", "has_glyph_data", "has_fiber_data", "has_solution_data"))
    hou.setFrame(0)
    mod.sync_available_options({"node": node})
    assert node.evalParm("source_block") == "Surface"
    assert node.evalParm("has_block_surface") and node.evalParm("has_glyph_data")
    node.parm("source_block").set("Contact")
    mod.sync_available_options({"node": node})
    assert node.evalParm("source_block") == "Volume"
    node.destroy()


def test_mesh_inversion_and_names(root, mod):
    moved = HEX.copy()
    moved[6] = (0.2, 0.2, 0.2)
    assert mod.hex_volumes(moved[None])[0] > 0
    assert mod._inverted_hex_corners(moved[None], np.ones(1)).tolist() == [True]
    mesh = mod._mesh_from_arrays(HEX, np.arange(8), [8], [12], {}, {})
    topology = mod.volume_topology(mesh)
    match = {"point_file_node": np.arange(8)}
    geometry = {"parsed": {"points": HEX}, "rest": HEX, "start": 0, "stop": 1,
                "file_corners": np.arange(8).reshape(1, 8),
                "is_hex": np.ones(1, dtype=bool), "path": "hex.msh", "number": 1}
    _, hex_report = mod.deformed_mesh_positions(
        geometry, match, topology, np.zeros_like(HEX), "simulation")
    assert hex_report["hex_corners_checked"]
    check_raises(mod.ExportError, "1 element(s) are inverted or flat",
                 mod.deformed_mesh_positions, geometry, match, topology, moved - HEX, "simulation")

    vtu = write_vtu(root / "mesh_names.vtu", piece(
        TET, point_fields={"solution": np.zeros_like(TET)}))
    node, _ = new_reader("mesh_names", write_pvd(root / "mesh_names.pvd", vtu))
    node.parm("export_folder").set(str(root / "moved_meshes"))
    geometries = [{"number": i, "curved": False, "path": str(root / str(i) / name)}
                  for i, name in enumerate(("part.msh", "part.msh", "part_geo1.msh"), 1)]
    report = {"nodes": 4, "unused": 0, "elements": 1, "hex_corners_checked": False, "ratio_min": 1,
              "ratio_max": 1, "ratio_min_at": 1, "ratio_max_at": 1}

    def fake_write(source, positions, destination):
        Path(destination).write_text(source)

    with module_hooks(mod, mesh_match=Mock(return_value={"geometries": geometries}),
                      deformed_mesh_positions=Mock(return_value=(TET, report)),
                      _p2_curvature=Mock(return_value=None), write_moved_nodes=fake_write):
        paths, _ = mod.export_deformed_mesh(node)
    assert len(set(paths)) == 3
    assert [Path(p).read_text() for p in paths] == [g["path"] for g in geometries]
    node.destroy()


def test_force_refresh_and_scan(root, mod):
    run = root / "force_run"
    output = run / "output"
    output.mkdir(parents=True)
    (run / "input").mkdir()
    definitions = [{"vertex_count": 1, "label": "Anchor", "kind": "nodes",
                    "body": 8, "vertices": [[0, 0, 0]]}]
    record = run / "input/hda_scene.json"
    record.write_text(json.dumps({"schema": mod.SCENE_RECORD_SCHEMA,
                                  "force_sets": definitions}))
    key = mod._file_key(record)
    bodies = [8, 8, 9, 9]
    a = mod._mesh_from_arrays(TET, np.arange(4), [4], [10], {"body_ids": bodies}, {})
    reordered = TET[[1, 0, 2, 3]]
    b = dict(a, points=reordered)
    c = dict(b, point_data={"body_ids": [8, 9, 8, 9]})
    np.testing.assert_equal(mod._force_nodes(key, a, definitions, [1])[1][0], [0])
    np.testing.assert_equal(mod._force_nodes(key, b, definitions, [1])[1][0], [1])
    assert not len(mod._force_nodes(key, c, definitions, [1])[1][0])

    vtu = output / "forces.vtu"
    pvd = write_pvd(output / "forces.pvd", vtu)
    initial_key = mod._file_key(pvd)
    forces = np.arange(12).reshape(4, 3)
    write_vtu(vtu, piece(TET, point_fields={"body_ids": bodies, "elastic_forces": forces,
                                            "scalar": [1] * 4}))
    node, _ = new_reader("force_refresh", pvd)
    node.parm("color_attrib").set("scalar")
    readout = mod.force_readout_text(node)
    assert "Anchor" in readout and mod._FORCE_FRAME_CACHE, readout
    assert mod.force_curves(node)["reaction"][0, 0, 0] == 0
    assert list(mod._scan_displayed_values(node, str(pvd)))[0][2][0] == 1
    mod._sequence_metadata(node)
    assert initial_key in mod._SEQUENCE_METADATA_CACHE
    assert pvd.is_file() and mod._file_key(pvd) == initial_key
    write_vtu(vtu, piece(reordered, point_fields={"body_ids": bodies, "elastic_forces": forces,
                                                   "scalar": [2] * 4, "new_field": [9] * 4}))
    mod.refresh({"node": node})
    assert node.sessionId() not in mod._FORCE_CURVES
    assert not any(mod._cache_path(k) == str(pvd) for k in mod._FORCE_FRAME_CACHE)
    assert not any(mod._cache_path(k) == str(record) for k in mod._FORCE_NODES_CACHE)
    assert not mod._SCAN_CACHE
    assert "new_field" in mod._sequence_metadata(node)[0][0]["Volume"]
    assert "Anchor" in mod.force_readout_text(node)
    assert mod.force_curves(node)["reaction"][0, 0, 0] == -3
    assert list(mod._scan_displayed_values(node, str(pvd)))[0][2][0] == 2
    assert mod._file_key(pvd) == initial_key

    # The first cached copy is released mid-iteration, and later frames are
    # yielded without any further cache-sized float32 copies.
    import weakref
    copies = []

    class Tracked(np.ndarray):
        def astype(self, dtype, *args, **kwargs):
            array = super().astype(dtype, *args, **kwargs)
            copies.append(weakref.ref(array))
            return array

    rows = [(i, float(i), np.ones(3).view(Tracked)) for i in range(5)]
    with module_hooks(mod, _SCAN_CACHE_MAX_BYTES=13,
                      _scan_key=Mock(return_value=("over_cap",)),
                      _scan_frames=Mock(return_value=iter(rows))):
        scan = mod._scan_displayed_values(node, str(pvd))
        assert next(scan)[0] == 0
        assert len(copies) == 1 and copies[0]() is not None
        assert next(scan)[0] == 1
        assert len(copies) == 1 and copies[0]() is None
        assert [row[0] for row in scan] == [2, 3, 4]
        assert len(copies) == 1 and not mod._SCAN_CACHE
    node.destroy()


def test_follow_and_scene_reload(root, mod):
    folder = root / "follow"
    folder.mkdir()
    vtu = write_vtu(folder / "step.vtu", piece(TET, point_fields={"s": [1] * 4}))
    pvd = write_pvd(folder / "follow.pvd", vtu, vtu, vtu)
    manifest = folder / "run-manifest.json"

    def set_manifest(status, **identity):
        manifest.write_text(json.dumps({**identity, "completion": {"status": status}}))

    set_manifest("completed", run_id="previous")
    node, _ = new_reader("follow_batch2", pvd)
    node.parm("follow_running").set(1)
    mod.toggle_follow_running({"node": node})
    sid = node.sessionId()
    assert node.evalParm("follow_running")
    assert mod._FOLLOW_WATCHERS[sid]["baseline"] == ("run_id", "previous")
    assert node.evalParm("availability_status") == "Waiting for new run (previous run completed)"
    set_manifest("completed", run_id="previous", started_at="still_the_same")
    assert mod.poll_running_simulation(node, force=True)["action"] == "noop"
    assert node.evalParm("follow_running")

    # A new run can start and finish between two polls. Even an unchanged
    # collection identity must be refreshed exactly once.
    set_manifest("completed", run_id="very_fast")
    original_key = mod._file_key(pvd)
    refresh = Mock(wraps=mod.refresh)
    with module_hooks(mod, refresh=refresh):
        assert mod.poll_running_simulation(node, force=True) == {"action": "finished", "steps": 3}
        assert refresh.call_count == 1
    assert mod._file_key(pvd) == original_key
    assert not node.evalParm("follow_running") and sid not in mod._FOLLOW_WATCHERS

    for identity in ({"started_at": "previous start"}, {}):
        set_manifest("completed", **identity)
        node.parm("follow_running").set(1)
        mod.toggle_follow_running({"node": node})
        baseline = mod._FOLLOW_WATCHERS[sid]["baseline"]
        assert baseline[0] == ("started_at" if identity else "file"), baseline
        set_manifest("running", run_id="running_now")
        write_pvd(pvd, vtu, vtu)
        assert mod.poll_running_simulation(node, force=True) == {"action": "refreshed", "steps": 2}
        write_pvd(pvd, vtu)
        assert mod.poll_running_simulation(node, force=True) == {"action": "refreshed", "steps": 1}
        assert mod._FOLLOW_WATCHERS[sid]["last_pvd_key"] == mod._file_key(pvd)
        node.parm("follow_running").set(0)
        mod.toggle_follow_running({"node": node})

    set_manifest("completed", run_id="previous_again")
    node.parm("follow_running").set(1)
    mod.toggle_follow_running({"node": node})
    set_manifest("failed", run_id="failed_before_step_0")
    pvd.unlink()
    assert mod.poll_running_simulation(node, force=True) == {"action": "finished", "steps": 0}
    assert not node.evalParm("follow_running")

    # Initial polls with transient errors stay armed. Fatal failures remove
    # the watcher and synchronize the parameter even in a headless session.
    set_manifest("running", run_id="new")
    node.parm("follow_running").set(1)
    with module_hooks(mod, poll_running_simulation=Mock(side_effect=OSError("busy"))):
        mod.toggle_follow_running({"node": node})
    assert node.evalParm("follow_running") and sid in mod._FOLLOW_WATCHERS
    assert "busy" in node.evalParm("availability_status")
    with module_hooks(mod, poll_running_simulation=Mock(side_effect=RuntimeError("fatal"))):
        mod.toggle_follow_running({"node": node})
    assert not node.evalParm("follow_running") and sid not in mod._FOLLOW_WATCHERS
    assert "fatal" in node.evalParm("availability_status")

    definition = node.type().definition()
    event = definition.sections()["OnLoaded"].contents()
    options = definition.extraFileOptions()
    assert options["OnLoaded/IsPython"] and options["OnLoaded/IsScript"]
    node.parm("follow_running").set(1)
    toggle = Mock()
    defer = Mock()
    proxy = SimpleNamespace(
        parm=node.parm,
        hdaModule=lambda: SimpleNamespace(toggle_follow_running=toggle))
    with (patch.object(hou, "isUIAvailable", return_value=True),
          patch.dict(sys.modules, {"hdefereval": SimpleNamespace(executeDeferred=defer)})):
        exec(event, {"kwargs": {"node": proxy}})
        defer.assert_called_once()
        toggle.assert_not_called()
        restore, deferred_node = defer.call_args.args
        restore(deferred_node)
        toggle.assert_called_once_with({"node": proxy}, restoring=True)

    node_path = node.path()
    scene = str(folder / "follow.hipnc")
    hou.hipFile.save(scene)
    for status in ("running", None, "completed"):
        if status is None:
            manifest.unlink()
        else:
            set_manifest(status, run_id=f"reload_{status}")
        hou.hipFile.load(scene, suppress_save_prompt=True)
        reloaded = hou.node(node_path)
        armed = status != "completed"
        assert bool(reloaded.evalParm("follow_running")) == armed
        assert (reloaded.sessionId() in reloaded.hdaModule()._FOLLOW_WATCHERS) == armed
    hou.node(node_path).destroy()


def test_probe_interior(root, mod):
    vtu = write_vtu(root / "probe.vtu", piece(TET, point_fields={"signal": [1] * 4}))
    node, _ = new_reader("probe_interior", write_pvd(root / "probe.pvd", vtu))
    node.parm("probe_enabled").set(1)
    state_code = node.type().definition().sections()["ViewerStateModule"].contents()
    namespace = {"__name__": "batch2_probe"}
    exec(compile(state_code, "ViewerStateModule", "exec"), namespace)
    state = namespace["State"].__new__(namespace["State"])
    state.scene_viewer = None
    geo = hou.Geometry()
    # The nearer face is a legend decoration and must be skipped.
    for name, z in (("readpvd_legend", 1), ("readpvd_interior", 0)):
        polygon = geo.createPolygon()
        for x, y in ((0, 0), (1, 0), (0, 1)):
            point = geo.createPoint()
            point.setPosition((x, y, z))
            polygon.addVertex(point)
        geo.createPrimGroup(name).add(polygon)
    picked = Mock()
    proxy = SimpleNamespace(
        evalParm=node.evalParm, worldTransform=node.worldTransform,
        node=lambda name: SimpleNamespace(geometry=lambda: geo),
        hdaModule=lambda: SimpleNamespace(probe_pick=picked))
    state.node = proxy
    event = SimpleNamespace(reason=lambda: hou.uiEventReason.Picked,
                            ray=lambda: (hou.Vector3(0.1, 0.1, 2), hou.Vector3(0, 0, -1)))
    assert state.onMouseEvent({"ui_event": event})
    picked.assert_called_once_with(proxy, 3)
    node.destroy()


def test_concurrent_xlsx(root, mod):
    path = root / "parallel.xlsx"
    both_writing = Barrier(2)
    both_inspected = Barrier(2)

    def write(value):
        def rows():
            both_writing.wait(10)
            assert len(list(root.glob(".parallel.xlsx.*.tmp"))) == 2
            both_inspected.wait(10)
            yield ["Field"]
            yield [value]

        workbook = mod.XlsxWorkbook()
        workbook.add_sheet("Data", rows())
        return workbook.save(path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(write, (17, 41))) == [{"Data": 2}] * 2
    assert not list(root.glob(".parallel.xlsx.*.tmp"))
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    assert read_xlsx(path)["Data"][1][0] in (17, 41)

    def interrupted():
        yield ["Incomplete"]
        raise ValueError("intentional interruption")

    workbook = mod.XlsxWorkbook()
    workbook.add_sheet("Data", interrupted())
    check_raises(ValueError, "intentional interruption", workbook.save, path)
    assert not list(root.glob(".parallel.xlsx.*.tmp"))
    assert read_xlsx(path)["Data"][1][0] in (17, 41)


def main():
    hou.hda.installFile(str(BASE / "object_readPVD.1.0.hdanc"), force_use_assets=True)
    bootstrap = hou.node("/obj").createNode("readPVD::1.0", "batch2_bootstrap")
    mod = bootstrap.hdaModule()
    bootstrap.destroy()
    with tempfile.TemporaryDirectory(prefix="readpvd_batch2_") as tmp:
        root = Path(tmp)
        for test in (test_xml_and_hdf, test_subdivision_and_workbook,
                     test_fields_reference_and_options, test_mesh_inversion_and_names,
                     test_force_refresh_and_scan, test_probe_interior,
                     test_concurrent_xlsx, test_follow_and_scene_reload):
            work = root / test.__name__
            work.mkdir()
            test(work, mod)
            print(f"PASS {test.__name__}", flush=True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
