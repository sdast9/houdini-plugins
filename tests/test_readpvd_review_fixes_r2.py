"""Round 2 Batch 2 regressions against the rebuilt Read PVD 1.0 asset."""

import base64
import gc
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock
import weakref
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import hou
import numpy as np

from test_export_data import export, read_xlsx, set_quantity, set_region, table
from test_readpvd_review_fixes_batch2 import (
    BASE, HEX, TET, check_raises, module_hooks, new_reader, piece,
    polyfem_reference_nodes, write_pvd, write_vtu,
)


def bundle(path, **blocks):
    path.write_text("<VTKFile type='vtkMultiBlockDataSet'><vtkMultiBlockDataSet>"
                    + "".join(f"<DataSet index='{i}' name={quoteattr(name)} "
                              f"file={quoteattr(file.name)}/>"
                              for i, (name, file) in enumerate(blocks.items()))
                    + "</vtkMultiBlockDataSet></VTKFile>")
    return path


def force_record(run, mod, targets=((0, 0, 0),)):
    path = run / "input" / "hda_scene.json"
    path.parent.mkdir(exist_ok=True)
    sets = [{"geometry": 1, "label": "Anchor", "kind": "nodes", "body": 8,
             "vertices": targets, "vertex_count": len(targets), "sideset": 4}]
    path.write_text(json.dumps({"schema": mod.SCENE_RECORD_SCHEMA,
                                "force_sets": sets}))
    return path, sets


def gmsh_geometry(run, points, cell_type):
    folder = run / "input"
    folder.mkdir(exist_ok=True)
    path = folder / "source.msh"
    coordinates = "".join(f"{i} {x} {y} {z}\n"
                          for i, (x, y, z) in enumerate(points, 1))
    connectivity = " ".join(map(str, range(1, len(points) + 1)))
    path.write_text(f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{len(points)}\n"
                    f"{coordinates}$EndNodes\n$Elements\n1\n"
                    f"1 {cell_type} 2 8 8 {connectivity}\n$EndElements\n")
    (folder / "params.json").write_text(json.dumps({"geometry": [{"mesh": path.name}]}))
    return path


def test_streaming_range(root, mod):
    pvd = write_pvd(root / "range.pvd",
                    write_vtu(root / "ref.vtu", piece(TET, point_fields={"scalar": [1] * 4})))
    node, _ = new_reader("r2_range", pvd)
    node.parm("color_attrib").set("scalar")
    calls = []
    released = []

    def streamed(*_):
        for i in range(12):
            if i > 3:
                gc.collect()
                assert released[i - 3]() is None, "an earlier full row was retained"
            row = np.array([i + 0.123456789012, i + 1.0000000001, -i, np.nan])
            released.append(weakref.ref(row))
            calls.append(i)
            yield i, float(i), row

    for pct in (0, 1):
        calls.clear()
        released.clear()
        mod._SCAN_CACHE.clear()
        node.setParms({"range_percentile": pct, "color_scale": 1})
        node.parmTuple("range_percentiles").set((0, 100))
        with module_hooks(mod, _scan_frames=streamed):
            mod.autoscale_all({"node": node})
        assert calls == list(range(12)) and not mod._SCAN_CACHE
        assert node.evalParm("color_min") == 0.123456789012
        assert node.evalParm("color_max") == 12.0000000001

    calls.clear()
    released.clear()
    node.parm("color_scale").set(0)
    with module_hooks(mod, _scan_frames=streamed):
        mod.autoscale_all({"node": node})
    assert node.evalParm("color_min") == -11
    assert node.evalParm("color_max") == 12.0000000001

    values = np.r_[-99, np.nan, np.arange(1, 43, dtype=float) + 2.5e-10]

    def rows(parts):
        return [(i, float(i), row) for i, row in enumerate(parts)]

    node.setParms({"range_percentile": 1, "color_scale": 1})
    node.parmTuple("range_percentiles").set((25, 100))
    scans = Mock(side_effect=lambda *_: iter(rows(np.array_split(values, 3))))
    with module_hooks(mod, _RANGE_SAMPLE_MAX_VALUES=64, _scan_frames=scans):
        mod.autoscale_all({"node": node})
        first = (node.evalParm("color_min"), node.evalParm("color_max"))
        mod.autoscale_all({"node": node})
        assert first == (node.evalParm("color_min"), node.evalParm("color_max"))
        assert scans.call_count == 1
        for (_, _, array), (_, _, expected) in zip(
                mod._SCAN_CACHE["rows"], rows(np.array_split(values, 3))):
            assert array.dtype == np.float64
            np.testing.assert_array_equal(array, expected)
        np.testing.assert_array_equal(first, np.percentile(values[2:], (25, 100)))

    # The global stride is eight for 42 values with a capacity of seven.
    expected = np.percentile(values[2::8], 25)
    ranges = []
    for cut in (2, 5, 11):
        mod._SCAN_CACHE.clear()
        with module_hooks(mod, _RANGE_SAMPLE_MAX_VALUES=7, _RANGE_CHUNK_VALUES=3,
                          _scan_frames=lambda *_: iter(rows(np.array_split(values, cut)))):
            mod.autoscale_all({"node": node})
            ranges.append((node.evalParm("color_min"), node.evalParm("color_max")))
        assert "estimated from a bounded sample of 6 values" in node.evalParm("color_status")
    np.testing.assert_array_equal(ranges, [(expected, values[-1])] * 3)

    mod._SCAN_CACHE.clear()
    one = Mock(return_value=iter(rows([np.array([1.0000000001, 2., 3.])])))
    with module_hooks(mod, _SCAN_CACHE_MAX_BYTES=16, _SCAN_CACHE_MAX_SINGLE_ROW_BYTES=32,
                      _scan_frames=one):
        list(mod._scan_displayed_values(node, str(pvd)))
        assert mod._SCAN_CACHE["rows"][0][2].nbytes == 24
    mod._SCAN_CACHE.clear()
    too_large = Mock(return_value=iter(rows([np.arange(5, dtype=float)])))
    with module_hooks(mod, _SCAN_CACHE_MAX_BYTES=16, _SCAN_CACHE_MAX_SINGLE_ROW_BYTES=32,
                      _scan_frames=too_large):
        list(mod._scan_displayed_values(node, str(pvd)))
        assert not mod._SCAN_CACHE
    node.destroy()


def test_bernstein_certificate(root, mod):
    u, v, w = HEX.T

    def product(a, b):
        return np.column_stack(((w - a) * u, (w - b) * v, w))

    def square(a, epsilon):
        off = np.sqrt(epsilon)
        return np.column_stack(((w - a) * u - off * v, off * u + (w - a) * v, w))

    center = product(0.4, 0.6)
    quarter = product(0.2, 0.3)
    for nodes, integral in ((center, 11 / 150), (quarter, 43 / 300)):
        np.testing.assert_allclose(mod.hex_volumes(nodes[None]), [integral])
        assert not mod._inverted_hex_corners(nodes[None], np.ones(1))[0]
    assert np.isclose(mod._hex_jacobian_determinants(center[None], .5, .5, .5)[0], -.01)
    np.testing.assert_allclose(mod._hex_jacobian_determinants(quarter[None], .5, .5, .25),
                               [-.0025])
    for a in (0, .5, 1):
        for b in (0, .5, 1):
            for c in (0, .5, 1):
                assert mod._hex_jacobian_determinants(quarter[None], a, b, c)[0] > 0

    translated = HEX + [1e6, -1e6, 1e6]
    mirrored = HEX * [-1, 1, 1]
    positive_after_split = square(.5, .01)
    ambiguous = square(.25, .01)
    nodes = np.array([center, quarter, translated, mirrored,
                      positive_after_split, ambiguous])
    inverted, unresolved = mod._hex_inversion_certificate(nodes, np.array([1, 1, 1, -1, 1, 1]))
    np.testing.assert_equal(inverted, [True, True, False, False, False, False])
    np.testing.assert_equal(unresolved, [False, False, False, False, False, True])

    mesh = mod._mesh_from_arrays(HEX, np.arange(8), [8], [12], {}, {})
    topology = mod.volume_topology(mesh)
    geometry = {"parsed": {"points": HEX}, "rest": HEX, "start": 0, "stop": 1,
                "file_corners": np.arange(8)[None], "is_hex": np.ones(1, dtype=bool),
                "number": 1, "path": str(root / "my_hex.msh")}
    match = {"point_file_node": np.arange(8)}
    for moved in (center, quarter):
        check_raises(mod.ExportError, "elements 1 of my_hex.msh", mod.deformed_mesh_positions,
                     geometry, match, topology, moved - HEX, "simulation")
    check_raises(mod.ExportError, "indeterminate (possibly inverted",
                 mod.deformed_mesh_positions, geometry, match, topology,
                 ambiguous - HEX, "simulation")


def test_linear_and_curved_export_reports(root, mod):
    _, _, q2, q3 = polyfem_reference_nodes()
    edges = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    p2 = np.vstack((TET, [TET[list(edge)].mean(axis=0) for edge in edges]))
    for i, (corners, gmsh_type, points, vtk_type) in enumerate(
            ((HEX, 5, HEX, 12), (HEX, 5, q2, 72), (HEX, 5, q3, 72), (TET, 4, p2, 24))):
        run = root / f"mesh_{i}"
        output = run / "output"
        output.mkdir(parents=True)
        gmsh_geometry(run, corners, gmsh_type)
        step = write_vtu(output / "step.vtu", piece(points, vtk_type, {
            "solution": np.zeros_like(points), "body_ids": np.full(len(points), 8)}))
        node, _ = new_reader(f"r2_mesh_report_{i}", write_pvd(output / "run.pvd", step))
        paths, report = mod.export_deformed_mesh(node)
        assert len(paths) == 1 and Path(paths[0]).is_file()
        if len(points) == len(corners):
            assert "Q1 hex Bernstein certificates" in report
        else:
            assert "no inversions detected at sampled corner/interior points" in report
            assert "none inverted" not in report
        node.destroy()


def test_reference_cache_legend_and_baseline(root, mod):
    base = np.array([0, 3e-9, 1e-9, 1e4])
    current = np.array([2e-9, 1, 3e-9, 1.25e4])
    a = write_vtu(root / "a.vtu", piece(TET, point_fields={"stress": base}))
    b = write_vtu(root / "b.vtu", piece(TET, point_fields={"stress": current}))
    c = write_vtu(root / "c.vtu", piece(TET, point_fields={"stress": [800] * 4},
                                       conn=[1, 0, 2, 3]))
    d = write_vtu(root / "d.vtu", piece(TET, point_fields={"stress": 2 * current}))
    pvd = write_pvd(root / "ref.pvd", a, b, c, d)
    node, _ = new_reader("r2_reference", pvd)
    node.setParms({"reference_enable": 1, "reference_mode": "percent",
                   "reference_frame": 0, "color_attrib": "stress",
                   "field_units": "Pa", "remesh_mode": 1})
    reference = node.node("reference_comparison")
    legend = node.node("legend_bar")
    mod._REFERENCE_CACHE.clear()
    loader = Mock(wraps=mod.load_block)
    upload = Mock(wraps=mod._set_floats)
    with module_hooks(mod, load_block=loader, _set_floats=upload,
                      _VTU_CACHE_SIZE=1, _REFERENCE_CACHE_SIZE=2):
        hou.setFrame(1)
        reference.cook(force=True)
        np.testing.assert_equal(reference.geometry().pointFloatAttribValues("reference_comparison"),
                                [0, np.nan, 0, 25])
        assert "3 of 4 points have a near-zero" in node.evalParm("reference_status")
        comparison_uploads = [c.args for c in upload.call_args_list
                              if c.args[1] == "reference_comparison"]
        assert comparison_uploads[0][0].__name__ == "setPointFloatAttribValuesFromString"
        assert comparison_uploads[0][2].dtype == np.float64
        assert mod.legend_title_text(node).endswith("[%]")
        assert legend.geometry().attribValue("legend_title").endswith("[%]")
        hou.setFrame(2)
        assert mod.legend_title_text(node).endswith("[Pa]")
        assert legend.geometry().attribValue("legend_title").endswith("[Pa]")
        assert reference.geometry().findPointAttrib("reference_comparison") is None
        hou.setFrame(3)
        reference.cook(force=True)
        ref_reads = [call for call in loader.call_args_list
                     if call.args[1] == 0 and call.kwargs.get("fallback") is False]
        assert len(ref_reads) == 1, ref_reads
        key, value = next(iter(mod._REFERENCE_CACHE.items()))
        assert key[-1] == mod.read_vtu(a)["topo_key"] and value.dtype == np.float64
        for reduction in ("trace", "magnitude"):
            mod._reference_reduced(node, str(pvd), "Volume", "stress", reduction)
        assert len(mod._REFERENCE_CACHE) == 2 and key not in mod._REFERENCE_CACHE

    rows = list(mod._scan_frames(node, str(pvd)))
    assert rows[2][2] is None
    mod.autoscale_all({"node": node})
    assert node.evalParm("color_max") == 150
    np.testing.assert_array_equal(
        mod._apply_reference(np.array([2e-40, 0]), np.array([1e-40, 0]), "percent"),
        [100, 0])
    changed = write_vtu(a, piece(TET, point_fields={"stress": base * 2}))
    assert changed == a
    _, reduced = mod._reference_reduced(node, str(pvd), "Volume", "stress", "auto")
    np.testing.assert_array_equal(reduced, base * 2)
    node.destroy()


def test_initial_ordering_and_display_choices(root, mod):
    p3, p4, q2, q3 = polyfem_reference_nodes()
    tri = q2.copy()
    tri[[18, 19]] = tri[[19, 18]]
    edges = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    p2 = np.vstack((TET, [TET[list(edge)].mean(axis=0) for edge in edges]))
    for i, (family, ctype, vertices, corners) in enumerate(
            (("tet", 24, p2, 4), ("tet", 71, p3, 4), ("tet", 71, p4, 4),
             ("hex", 29, tri, 8), ("hex", 72, q2, 8), ("hex", 72, q3, 8))):
        collapsed = vertices.copy()
        collapsed[corners:] = vertices[:corners].mean(axis=0)
        vtu = write_vtu(root / f"bad_{i}.vtu", piece(collapsed, ctype))
        normal = write_vtu(root / f"normal_{i}.vtu", piece(vertices, ctype))
        mirrored = write_vtu(root / f"mirror_{i}.vtu",
                             piece(vertices * [-1, 1, 1], ctype))
        pvd = write_pvd(root / f"step0_{i}.pvd", vtu, vtu)
        node, _ = new_reader(f"r2_order_{i}", pvd)
        node.node("frame_data").cook(force=True)
        assert not node.node("frame_data").errors(), node.node("frame_data").errors()
        warning = f"higher-order sub-{family}s on step 0 appear"
        assert warning in node.evalParm("availability_status"), (i, node.evalParm("availability_status"))
        assert node.userData(mod._INITIAL_ORDERING_DATA)
        hou.setFrame(1)
        mod.sync_available_options({"node": node})
        assert warning in node.evalParm("availability_status")
        mod._follow_status(node, "Following next step")
        assert node.evalParm("availability_status").startswith("Following next step")
        assert warning in node.evalParm("availability_status")
        for file in (normal, mirrored):
            hou.setFrame(0)
            node.parm("PVD_file").set(str(write_pvd(pvd, file)))
            node.node("frame_data").cook(force=True)
            mod.sync_available_options({"node": node})
            assert warning not in node.evalParm("availability_status")
        node.destroy()

    fibers = np.tile([1., 0., 0.], (4, 1))
    fiber_fields = {f"fiber_direction_{axis}": fibers[:, i]
                    for i, axis in enumerate("xyz")}
    volume = write_vtu(root / "body.vtu", piece(TET, point_fields={
        "body_ids": [8, 8, 9, 9], "F_1": fibers, "F_2": np.roll(fibers, 1, axis=1),
        "F_3": np.roll(fibers, 2, axis=1), **fiber_fields}))
    surface = write_vtu(root / "surface.vtu", piece(TET[:3], 5, {
        "body_ids": [8, 8, 8], **{key: v[:3] for key, v in fiber_fields.items()}}))
    body_pvd = write_pvd(root / "choices.pvd",
                         bundle(root / "choices.vtm", Volume=volume, Surface=surface))
    node, _ = new_reader("r2_choices", body_pvd)
    node.setParms({"visible_bodies": "8 9", "fiber_frame": "deformed", "source_block": "Surface"})
    messages = Mock()
    with module_hooks(mod, _message=messages):
        mod.sync_available_options({"node": node})
    messages.assert_not_called()
    assert node.evalParm("visible_bodies") == "8 9"
    assert node.parm("fiber_frame").evalAsString() == "deformed"
    assert node.evalParm("has_fiber_data")
    assert "Deformed fiber directions are unavailable" in node.evalParm("availability_status")
    node.destroy()


def test_match_incident_elements_and_volume(root, mod):
    output = root / "output"
    output.mkdir()
    force_record(root, mod)
    gmsh_geometry(root, TET, 4)
    a = write_vtu(output / "start.vtu", piece(TET, point_fields={"body_ids": [8] * 4}))
    pvd = write_pvd(output / "mesh.pvd", a)
    mesh_a = mod.volume_mesh(str(pvd), 0)
    ta = mod.volume_topology(mesh_a)
    match_a = mod.mesh_match(str(pvd), mesh_a, ta)
    write_vtu(a, piece(TET[[1, 0, 2, 3]], point_fields={"body_ids": [8] * 4}))
    mesh_b = mod.volume_mesh(str(pvd), 0)
    tb = mod.volume_topology(mesh_b)
    assert ta["key"] == tb["key"] and ta is not tb
    match_b = mod.mesh_match(str(pvd), mesh_b, tb)
    np.testing.assert_equal(match_a["point_file_node"], [0, 1, 2, 3])
    np.testing.assert_equal(match_b["point_file_node"], [1, 0, 2, 3])

    # Two disconnected output cells share the physical vertex at the origin.
    points = np.concatenate((TET, -TET))
    mesh = mod._mesh_from_arrays(points, np.arange(8), [4, 8], [10, 10],
                                 {"body_ids": np.full(8, 8)}, {})
    topology = mod.volume_topology(mesh)
    assert topology["point_node"][0] == topology["point_node"][4]
    identity = {"node_number": np.arange(1, topology["n_nodes"] + 1),
                "node_geometry": np.ones(topology["n_nodes"]), "match": None}
    node, _ = new_reader("r2_regions", pvd)
    for kind, extra in (("sideset", {"sideset": "1"}), ("node", {
            "geometry": 1, "number": int(topology["point_node"][0] + 1)})):
        elements, nodes, *_ = mod.region_members(
            node, dict(index=1, kind=kind, name="", only=[], **extra),
            mesh, topology, identity, str(pvd))
        np.testing.assert_equal(elements, [0, 1])
        np.testing.assert_equal(nodes, [topology["point_node"][0]])
    node.destroy()

    edge_indices = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    quadratic = np.vstack((HEX, [HEX[list(edge)].mean(axis=0) for edge in edge_indices]))
    mesh = mod._mesh_from_arrays(quadratic, np.arange(20), [20], [25], {}, {})
    topology = mod.volume_topology(mesh)
    lifted = quadratic.copy()
    lifted[12, 2] += 0.12
    np.testing.assert_allclose(mod.element_volumes(topology, quadratic), [1], atol=1e-13)
    np.testing.assert_allclose(mod.element_volumes(topology, lifted), [1.04], atol=1e-13)
    np.testing.assert_allclose(mod.element_volumes(topology, lifted * [-1, 1, 1]),
                               [-1.04], atol=1e-13)


def test_export_step_geometry_and_forces(root, mod):
    output = root / "output"
    output.mkdir()
    force_record(root, mod)
    elastic = np.array([[1, 2], [7, 8], [9, 10], [11, 12]])
    contact = np.tile([3, 4], (4, 1))
    solution = np.tile([.125, .25], (4, 1))
    fields = {"body_ids": [8] * 4, "elastic_forces": elastic,
              "contact_forces": contact, "solution": solution}
    a = write_vtu(output / "a.vtu", piece(TET, point_fields=fields))
    # An even row permutation moves the anchor to row 2 without flipping
    # this tetrahedron. Scaling each axis by two multiplies volume by eight.
    b = write_vtu(output / "b.vtu", piece(2 * TET[[1, 2, 0, 3]], point_fields=fields))
    c = write_vtu(output / "c.vtu", piece(3 * TET, point_fields={"body_ids": [8] * 4}))
    surf = write_vtu(output / "surface.vtu",
                     piece(TET[:3], 5, {"solution": solution[:3]}))
    a_multi = bundle(output / "a.vtm", Volume=a, Surface=surf)
    pvd = write_pvd(output / "forces.pvd", a_multi, b, c)
    node, _ = new_reader("r2_forces", pvd)
    node.setParms({"export_regions": 1, "export_quantities": 1, "export_forces": 1,
                   "export_force_reaction": 1, "export_force_contact": 1,
                   "export_force_displacement": 1, "export_force_x": 1,
                   "export_force_y": 1, "export_force_z": 1, "export_units": 1,
                   "export_step_first": 0, "export_step_last": 2})
    set_region(node, 1, "all", "Model")
    set_quantity(node, 1, "_volume", volume=0)
    curves = mod.force_curves(node)
    np.testing.assert_equal(curves["reaction"][:, 0], [[-4, -6, 0], [-12, -14, 0],
                                                      [np.nan] * 3])
    np.testing.assert_equal(curves["contact"][:, 0], [[3, 4, 0]] * 2 + [[np.nan] * 3])
    np.testing.assert_equal(curves["displacement"][:, 0],
                            [[.125, .25, 0]] * 2 + [[np.nan] * 3])
    _, sheets = export(node)
    data = table(sheets["Data"])
    np.testing.assert_allclose(data["Model: volume"], np.array([1, 8, 27]) / 6)
    for key, label in (("reaction", "reaction"), ("contact", "contact force"),
                       ("displacement", "mean displacement")):
        for i, axis in enumerate("xyz"):
            column = data[f"Sideset 4: {label} {axis}"]
            assert column[:2] == list(curves[key][:2, 0, i])
            assert column[2] is None

    node.setParms({"export_step_first": 2, "export_step_last": 2})
    _, sheets = export(node)
    assert table(sheets["Data"])["Sideset 4: reaction x"] == [None]
    hou.setFrame(2)
    assert "unavailable at this step" in mod.force_readout_text(node)
    hou.setFrame(0)
    node.setParms({"show_multi_blocks": 1, "multi_surface_show": 1, "show_deformed": 1})
    geo = node.node("multi_block_display").geometry()
    np.testing.assert_allclose([point.position() for point in geo.points()],
                               TET[:3] + [.125, .25, 0])
    node.parm("PVD_file").set(str(write_pvd(output / "no_forces.pvd", c)))
    check_raises(mod.ForceCurveError, "no nodal forces", mod.force_curves, node)
    node.destroy()


def test_companion_cache_and_remesh_smoothing(root, mod):
    pvd = write_pvd(root / "fiber.pvd", write_vtu(root / "a.vtu", piece(TET)))
    name = root / mod.FIBER_COMPANION_NAME
    directions = np.array([[[1, 0, 0], [0, 1, 0]]] * 3)
    np.savez(name, schema_version=[1], centroids=[[0, 0, 0], [1, 0, 0]],
             body_ids=[8, 8], directions=directions, family_body_ids=[8, 8, 8],
             family_names=["1st/fiber", "1st fiber", "F_1"],
             family_labels=["First", "Second", "Reserved"])
    mesh = {"points": np.array([[0., 0, 0], [1, 0, 0]]), "cells": {},
            "topo_key": "same", "point_data": {"body_ids": np.array([8, 8])}}
    swapped = dict(mesh, points=mesh["points"][::-1])
    rebody = dict(mesh, point_data={"body_ids": np.array([9, 8])})
    first = mod._companion_point_fibers(str(pvd), mesh)
    second = mod._companion_point_fibers(str(pvd), swapped)
    third = mod._companion_point_fibers(str(pvd), rebody)
    names = [row[0] for row in first]
    assert names == ["readpvd_companion_1st_fiber", "readpvd_companion_1st_fiber_2",
                     "readpvd_companion_F_1"]
    np.testing.assert_equal(first[0][2], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(second[0][2], [[0, 1, 0], [1, 0, 0]])
    np.testing.assert_equal(third[0][2], [[0, 0, 0], [0, 1, 0]])
    node, _ = new_reader("r2_companion", pvd)
    mod.refresh({"node": node})
    assert not any(key[0] == str(name) for key in mod._FIBER_POINT_CACHE)
    node.destroy()

    def frames(stem, offset):
        return write_vtu(root / f"{stem}.vtu",
                         piece(TET, point_fields={"scalar": [2] * 4}),
                         piece(TET + offset, point_fields={"scalar": [8] * 4}))
    coincident = frames("coincident", 0)
    separated = frames("separated", 10)
    node, _ = new_reader("r2_smoothing", write_pvd(root / "smooth.pvd", coincident, separated))
    node.setParms({"smooth_field": 1, "remesh_mode": 1, "color_attrib": "scalar"})
    rows = list(mod._scan_frames(node, node.evalParm("PVD_file")))
    np.testing.assert_array_equal(rows[0][2], [5] * 8)
    np.testing.assert_array_equal(rows[1][2], [2] * 4 + [8] * 4)
    node.destroy()


def test_vtk_parsers_and_field_reservations(root, mod):
    # Both encodings would borrow bytes from the next header without bounds.
    def binary_array(name, offset):
        return f'<DataArray type="Float64" Name="{name}" format="appended" offset="{offset}"/>'

    head = struct.pack("<I", 8)
    short = b"\x00" * 4
    complete = struct.pack("<d", -13.0)
    for encoding in ("raw", "base64"):
        if encoding == "raw":
            first, second = head + short, head + complete
        else:
            first = base64.b64encode(head) + base64.b64encode(short)
            second = base64.b64encode(head) + base64.b64encode(complete)
        # Declaration order is opposite to physical offset order.
        arrays = binary_array("complete", len(first)) + binary_array("short", 0)
        path = root / f"{encoding}.vtu"
        path.write_bytes((f'<VTKFile><UnstructuredGrid>{arrays}</UnstructuredGrid>'
                          f'<AppendedData encoding="{encoding}">_').encode()
                         + first + second + b"</AppendedData></VTKFile>")
        reader = mod._Reader(path)
        assert isinstance(reader.appended, memoryview) and reader.appended.obj is reader.data
        good, bad = reader.root.findall(".//DataArray")
        np.testing.assert_array_equal(reader.read_data_array(good), [-13])
        check_raises(mod.VtuParseError, "Truncated VTK binary payload",
                     reader.read_data_array, bad)

    class TracedBytes(bytes):
        def find(self, sub, start=0, end=None):
            if sub == b"<!--":
                comment_scans.append((start, end))
            return super().find(sub, start, len(self) if end is None else end)

    comment_scans = []
    xml = TracedBytes(b"<VTKFile><!-- <DataArray pretend /> -->"
                      + b'<DataArray type="Float64"> -5 <!-- 800 --> 7 </DataArray>' * 20
                      + b"</VTKFile>")
    _, ranges = mod._strip_inline_payloads(xml)
    assert len(ranges) == 20
    assert sum(end is None or end == len(xml) for _, end in comment_scans) == 1
    path = root / "comments.vtu"
    path.write_bytes(xml)
    reader = mod._Reader(path)
    for array in reader.root.findall("DataArray"):
        np.testing.assert_equal(reader.read_data_array(array), [-5, 7])

    hdf = root / "scalars.vtkhdf"
    h5py = mod._import_h5py()
    with h5py.File(hdf, "w") as handle:
        vtk = handle.create_group("VTKHDF")
        vtk.attrs["Type"] = "UnstructuredGrid"
        for field, values in {"Points": TET, "Connectivity": [0, 1, 2, 3],
                              "Offsets": [0, 4], "Types": [10],
                              "PointData/signed": [[-2], [-7], [0], [1]],
                              "CellData/signed_cell": [[-3]]}.items():
            vtk.create_dataset(field, data=values)
        vtk.create_group("Steps").attrs["NSteps"] = 1
    mesh = mod.read_hdf(hdf)
    assert mesh["point_data"]["signed"].shape == (4,)
    assert mesh["cell_data"]["signed_cell"]["tet"].shape == (1,)
    np.testing.assert_equal(mod._reduce_array(np.array([[-2], [-7]]), "auto"), [-2, -7])
    pvd = write_pvd(root / "hdf.pvd", hdf)
    node, _ = new_reader("r2_hdf", pvd)
    node.parm("color_attrib").set("signed")
    np.testing.assert_equal(next(mod._scan_frames(node, str(pvd)))[2], [-2, -7, 0, 1])
    np.testing.assert_equal(node.node("color_reduce").geometry().pointFloatAttribValues("for_color"),
                            [-2, -7, 0, 1])
    node.destroy()
    with h5py.File(hdf, "r+") as handle:
        handle["VTKHDF/Steps"].attrs["NSteps"] = 2
    for read in (mod.read_hdf, mod.read_hdf_field_info):
        check_raises(mod.VtuParseError, "Temporal multi-step VTKHDF is unsupported", read, hdf)
    with h5py.File(hdf, "r+") as handle:
        steps = handle["VTKHDF/Steps"]
        steps.attrs["NSteps"] = 1
        steps.create_dataset("Values", data=[0, 1])
    check_raises(mod.VtuParseError, "Temporal multi-step VTKHDF is unsupported", mod.read_hdf, hdf)

    axes = np.tile(np.eye(3), (4, 1, 1))
    tensor = np.tile(np.diag([1., 2., 3.]).ravel(), (4, 1))
    fields = {"F/1": np.full((4, 3), 999), "F_1": axes[:, 0],
              "F_2": axes[:, 1], "F_3": axes[:, 2], "T": tensor,
              "body_ids": [8] * 4, "J": [4] * 4, "solution": TET}
    multi = write_vtu(root / "missing_fields.vtu", piece(TET, point_fields=fields),
                      piece(TET + 5, point_fields={"marker": [1] * 4}))
    pvd = write_pvd(root / "multi.pvd", multi)
    node, _ = new_reader("r2_multi", pvd)
    mod.sync_available_options({"node": node})
    assert mod._available_body_ids(node) == [8]
    node.node("frame_data").cook(force=True)
    assert not node.node("frame_data").errors(), node.node("frame_data").errors()
    geo = node.node("frame_data").geometry()
    assert mod._field_name("F/1") != "F_1" and mod._field_name("F_1") == "F_1"
    assert geo.findPointAttrib("F_1").size() == 3
    assert mod._field_name("J") != "J"
    assert np.array_equal(geo.pointFloatAttribValues("F_1")[:3], [1, 0, 0])
    parsed = mod.read_vtu(multi)
    np.testing.assert_equal(mod._reduce_array(parsed["point_data"]["T"], "principal_max"),
                            [3] * 4 + [np.nan] * 4)
    np.testing.assert_equal(mod._mesh_field(parsed, "J"), [1] * 4 + [np.nan] * 4)
    for field in ("principal_stretches", "almansi_strain_eigenvalues", "hencky_strain_eigenvalues"):
        values = mod._mesh_field(parsed, field)
        assert np.isfinite(values[:4]).all() and np.isnan(values[4:]).all(), field
    node.destroy()


def test_translated_probe(root, mod):
    result, shown = hou.Geometry(), hou.Geometry()
    for geo in (result, shown):
        geo.addAttrib(hou.attribType.Point, "rest", (0., 0., 0.))
        geo.addAttrib(hou.attribType.Point, "readpvd_point", 0)
    rests = np.array([[1e6, 0, 0], [1e6 + 1, 0, 0], [1e6, 1, 0]])
    for i, point in enumerate(result.createPoints(rests)):
        point.setAttribValue("rest", tuple(rests[i]))
        point.setAttribValue("readpvd_point", i)
    clipped = np.array([[1e6 + .75, 0, 0], [1e6 + 1, 0, 0], [1e6 + .75, .25, 0]])
    prim = shown.createPolygon()
    for i, point in enumerate(shown.createPoints(clipped)):
        point.setAttribValue("rest", tuple(clipped[i]))
        point.setAttribValue("readpvd_point", i)
        prim.addVertex(point)
    proxy = SimpleNamespace(node=lambda name: SimpleNamespace(
        geometry=lambda: shown if name == "output" else result))
    pick = Mock()
    with module_hooks(mod, probe_point=pick):
        mod.probe_pick(proxy, 0)
    pick.assert_called_once_with(proxy, 1)


def test_follow_and_export_errors(root, mod):
    frame = write_vtu(root / "test.vtu", piece(TET))
    pvd = write_pvd(root / "follow.pvd", frame, frame)
    manifest = root / "run-manifest.json"

    def update(run, status):
        manifest.write_text(json.dumps({"run_id": run, "completion": {"status": status}}))

    update("previous", "completed")
    node, _ = new_reader("r2_follow", pvd)
    node.parm("follow_running").set(1)
    mod.toggle_follow_running({"node": node})
    original = mod._file_key(pvd)
    refresh = Mock(wraps=mod.refresh)
    with module_hooks(mod, refresh=refresh):
        update("new", "failed")
        assert mod.poll_running_simulation(node, force=True) == {"action": "finished", "steps": 0}
        refresh.assert_not_called()
    assert not node.evalParm("follow_running")
    assert mod._file_key(pvd) == original

    update("active", "running")
    node.parm("follow_running").set(1)
    mod.toggle_follow_running({"node": node})
    scene = str(root / "active.hipnc")
    node_path = node.path()
    hou.hipFile.save(scene)
    assert json.loads(node.userData(mod._STEP_COUNT_DATA))[1] == 2
    write_pvd(pvd, frame, frame, frame)
    hou.hipFile.load(scene, suppress_save_prompt=True)
    node = hou.node(node_path)
    assert node.evalParm("follow_running")
    assert json.loads(node.userData(mod._STEP_COUNT_DATA))[1] == 3
    assert mod._FOLLOW_WATCHERS[node.sessionId()]["last_count"] == 3

    for handler, underlying, status in (
            (mod.export_mesh_button, "export_deformed_mesh", "export_mesh_report"),
            (mod.export_data_button, "export_data", "export_data_status")):
        for error in (mod.ExportError, OSError, mod.MshParseError, mod.VtuParseError,
                      ET.ParseError, ValueError):
            failing = Mock(side_effect=error("incomplete frame"))
            with module_hooks(mod, **{underlying: failing, "_message": Mock()}):
                handler({"node": node})
            assert node.evalParm(status) == "Not exported: incomplete frame"
    mod._stop_following(node)
    node.destroy()


def test_xlsx_edge_cases(root, mod):
    path = root / "output.xlsx"
    book = mod.XlsxWorkbook()
    book.add_sheet("hIsToRy", [["left", np.nan, "right"], ["x" * 32767]])
    book.add_sheet("History", [[1]])
    old_mask = os.umask(0o027)
    try:
        book.save(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
        path.chmod(0o664)
        book.save(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o664
    finally:
        os.umask(old_mask)
    sheets = read_xlsx(path)
    assert all(name.lower() != "history" for name in sheets)
    assert sheets["hIsToRy (2)"][0] == ["left", None, "right"]
    assert len(sheets["hIsToRy (2)"][1][0]) == 32767
    original = path.read_bytes()
    invalid = mod.XlsxWorkbook()
    invalid.add_sheet("Data", [["x" * 32768]])
    check_raises(mod.XlsxLimitError, "Cell A1 exceeds Excel's 32,767-character limit",
                 invalid.save, path)
    assert path.read_bytes() == original
    assert not list(root.glob(".output.xlsx.*.tmp"))


def main():
    hou.hda.installFile(str(BASE / "object_readPVD.1.0.hdanc"), force_use_assets=True)
    bootstrap = hou.node("/obj").createNode("readPVD::1.0", "r2_bootstrap")
    mod = bootstrap.hdaModule()
    bootstrap.destroy()
    checks = (test_streaming_range, test_bernstein_certificate, test_linear_and_curved_export_reports,
              test_reference_cache_legend_and_baseline, test_initial_ordering_and_display_choices,
              test_match_incident_elements_and_volume, test_export_step_geometry_and_forces,
              test_companion_cache_and_remesh_smoothing, test_vtk_parsers_and_field_reservations,
              test_translated_probe, test_follow_and_export_errors, test_xlsx_edge_cases)
    with tempfile.TemporaryDirectory(prefix="readpvd_r2_") as tmp:
        for check in checks:
            work = Path(tmp) / check.__name__
            work.mkdir()
            check(work, mod)
            print(f"PASS {check.__name__}", flush=True)
    print("ALL PASS")


if __name__ == "__main__":
    main()
