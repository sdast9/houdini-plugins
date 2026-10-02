"""Read PVD's first look at a PolyFEM result, and the output it gets.

Until 2026-10-01 a freshly loaded result kept the 0..1 color range a new node
starts with (one flat color for a 4 mm displacement), opened on displacement
because the PolyFEM node's default Minimal Fields output has no exported
von_mises, and kept the old range when the field changed. Fields whose
PolyFEM name contains "/" (material output) broke Auto Range: All Frames and
the reference comparison. Body ids were off by default, so Visible Bodies
was disabled, and smoothing averaged across touching bodies. Contact forces
were off by default.

Run: hython tests/test_readpvd_first_look.py
Uses ../polyfem/build/PolyFEM_bin; no gmsh needed.
"""

import json
import os
import subprocess
import tempfile

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
TYPE = "stevenabramowitch::dev::PolyFEM::2.0"


def write_box(path, n=3, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    """Structured tet box (six tets per cell) as MSH 2.2 ASCII."""
    xs = [np.linspace(lo[a], hi[a], n + 1) for a in range(3)]

    def vid(i, j, k):
        return (k * (n + 1) + j) * (n + 1) + i

    points = np.array([(x, y, z) for z in xs[2] for y in xs[1]
                       for x in xs[0]])
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
                    if np.linalg.det(np.array([points[b] - points[a],
                                               points[cc] - points[a],
                                               points[d] - points[a]])) < 0:
                        b, cc = cc, b
                    tets.append((a, b, cc, d))
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(points)}\n")
        for number, (x, y, z) in enumerate(points.tolist(), 1):
            handle.write(f"{number} {x!r} {y!r} {z!r}\n")
        handle.write(f"$EndNodes\n$Elements\n{len(tets)}\n")
        for number, tet in enumerate(tets, 1):
            handle.write(f"{number} 4 2 1 1 "
                         + " ".join(str(v + 1) for v in tet) + "\n")
        handle.write("$EndElements\n")
    return path


def new_scene(work, name):
    node = hou.node("/obj").createNode(TYPE, name)
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN})
    return node


def load(node, geo, path):
    if geo > node.evalParm("num_geos"):
        node.setParms({"geo_int": geo, "num_geos": geo})
    node.setParms({f"file_location{geo}": path})
    node.parm(f"file_location{geo}").pressButton()


def sideset(node, geo, pattern, kind, text, dims=(1, 1, 1)):
    j = node.evalParm(f"sideset_selection{geo}_1") + 1
    node.parm(f"sideset_selection{geo}_1").set(j)
    node.parm(f"sideset_selection{geo}_1").pressButton()
    node.setParms({f"basegroup{geo}_1_{j}": pattern,
                   f"Boundary_Condition__{geo}_1_{j}": 1,
                   f"boundary_type{geo}_1_{j}_1": kind})
    node.setParms({(f"vector_{geo}_1_{j}_1" if kind in (0, 1)
                    else f"value_{geo}_1_{j}_1"): text})
    for axis, flag in zip("xyz", dims):
        node.setParms({f"{axis}_dimension{geo}_1_{j}_1": flag})


def run(node):
    params = node.hdaModule().write_params_only({"node": node})
    assert params, "export failed"
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "info", "--max_threads", "1"],
        cwd=os.path.dirname(params), capture_output=True, text=True,
        timeout=900)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    return params, os.path.join(os.path.dirname(os.path.dirname(params)),
                                "output", "sim.pvd")


def for_color(viewer):
    result = viewer.node("OUT_result")
    result.cook(force=True)
    return np.frombuffer(result.geometry().pointFloatAttribValuesAsString(
        "for_color"), dtype=np.float32)


def check_first_look(root):
    work = os.path.join(root, "first_look")
    os.makedirs(work)
    node = new_scene(work, "first_look")
    load(node, 1, write_box(os.path.join(work, "box.msh"), n=3))
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 0.5, "enable": 0,
                   "materials1_1": 3, "E1_1": 1e5, "nu1_1": 0.3})
    sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    sideset(node, 1, "axis:+z:0.99", 0, '[0, 0, "-0.1*t"]')
    params, pvd = run(node)
    options = json.load(open(params))["output"]["paraview"]
    # PolyFEM node defaults: Minimal Fields (no exported von Mises), body ids
    # on, contact forces only with contact on.
    assert "von_mises" not in options["fields"], options["fields"]
    assert options["options"]["body_ids"] is True, options["options"]
    assert "body_ids" in options["fields"], options["fields"]
    assert options["options"]["contact_forces"] is False, options["options"]

    viewer = hou.node("/obj").createNode("readPVD::1.0", "first_look_viewer")
    module = viewer.hdaModule()
    hou.setFrame(0)
    viewer.setParms({"PVD_file": pvd})
    module.start({"node": viewer})
    last = len(module.read_pvd(pvd)) - 1
    assert viewer.evalParm("color_attrib") == "von_mises_derived", \
        viewer.evalParm("color_attrib")
    hou.setFrame(last)
    final = for_color(viewer)
    hou.setFrame(0)
    initial = for_color(viewer)
    low, high = viewer.evalParm("color_min"), viewer.evalParm("color_max")
    assert final.max() > 1.0, final.max()
    assert np.isclose(high, final.max(), rtol=1e-5), (high, final.max())
    assert np.isclose(low, min(final.min(), initial.min()), atol=1e-6), low
    print(f"PASS: a loaded result opens on von Mises (derived) with the range "
          f"{low:.4g}..{high:.4g} of its values (was 0..1)")

    # A new field (or value to display) gets its own range ...
    viewer.parm("color_attrib").set("solution_mag")
    module.color_selection_changed({"node": viewer})
    hou.setFrame(last)
    final = for_color(viewer)
    assert np.isclose(viewer.evalParm("color_max"), final.max(), rtol=1e-5)
    assert viewer.evalParm("color_min") == 0.0
    viewer.parm("color_attrib").set("solution")
    viewer.parm("color_reduction").set("z")
    module.color_selection_changed({"node": viewer})
    final = for_color(viewer)
    assert np.isclose(viewer.evalParm("color_min"), final.min(), rtol=1e-5)
    assert viewer.evalParm("color_max") == 0.0
    # ... unless the range is locked
    viewer.setParms({"range_lock": 1})
    locked = (viewer.evalParm("color_min"), viewer.evalParm("color_max"))
    viewer.parm("color_attrib").set("von_mises_derived")
    module.color_selection_changed({"node": viewer})
    assert (viewer.evalParm("color_min"),
            viewer.evalParm("color_max")) == locked
    print("PASS: the range follows a new field or display value unless "
          "Lock Displayed Range is on")
    return work


def check_slash_fields(root):
    work = os.path.join(root, "material_fields")
    os.makedirs(work)
    node = new_scene(work, "material_fields")
    load(node, 1, write_box(os.path.join(work, "box.msh"), n=3))
    module = node.hdaModule()
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 0.5, "enable": 0,
                   "materials1_1": module.MATERIAL_TOKENS.index("MaterialSum"),
                   "num_fiber_families1_1": 1})
    node.setParms({"fam_model1_1_1": "HGODispersion", "fam_kappa1_1_1": 0.1,
                   "fam_k11_1_1": 1234.0})
    node.parmTuple("fam_fib_dir1_1_1").set((0, 0, 1))
    sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    sideset(node, 1, "axis:+z:0.99", 0, '[0, 0, "0.2*t"]')
    node.setParms({"materials_fields": 1})
    _, pvd = run(node)

    viewer = hou.node("/obj").createNode("readPVD::1.0", "material_viewer")
    phm = viewer.hdaModule()
    hou.setFrame(0)
    viewer.setParms({"PVD_file": pvd})
    phm.start({"node": viewer})
    tokens = phm._menu_tokens(phm.color_field_menu({"node": viewer}))
    k1 = next(t for t in tokens if t.endswith("_k1"))
    kappa = next(t for t in tokens if t.endswith("_kappa"))
    assert "/" in phm._display_name(k1), (k1, phm._display_name(k1))
    viewer.parm("color_attrib").set(k1)
    phm.color_selection_changed({"node": viewer})
    phm.autoscale_all({"node": viewer})
    assert np.isclose(viewer.evalParm("color_min"), 1234.0) and np.isclose(
        viewer.evalParm("color_max"), 1234.0), (
        viewer.evalParm("color_min"), viewer.evalParm("color_max"))
    mesh = phm.load_frame(pvd, 1)["Volume"]
    assert phm._mesh_point_field(mesh, k1) is not None
    viewer.setParms({"reference_enable": 1, "reference_frame": 0})
    hou.setFrame(1)
    viewer.node("OUT_result").cook(force=True)
    status = viewer.evalParm("reference_status")
    assert status.startswith(f"Comparing {k1} with frame 0"), status
    viewer.setParms({"reference_enable": 0})
    # Dispersion keeps its fixed 0..1/3 scale on every range button.
    viewer.parm("color_attrib").set(kappa)
    phm.color_selection_changed({"node": viewer})
    assert (viewer.evalParm("color_min"), viewer.evalParm("color_max")) == \
        (0.0, 1.0 / 3.0)
    viewer.setParms({"color_max": 5.0})
    phm.autoscale_all({"node": viewer})
    assert viewer.evalParm("color_max") == 1.0 / 3.0
    print(f"PASS: '{phm._display_name(k1)}' ranges over all frames and "
          "compares with a reference frame")


def write_vtu(path, points, tets, point_data):
    def ascii(values):
        return " ".join(repr(float(v)) for v in np.ravel(values))
    arrays = "".join(
        f'<DataArray type="Float64" Name="{name}" '
        f'NumberOfComponents="{1 if np.ndim(values) == 1 else 3}" '
        f'format="ascii">{ascii(values)}</DataArray>'
        for name, values in point_data.items())
    with open(path, "w") as handle:
        handle.write(
            '<?xml version="1.0"?><VTKFile type="UnstructuredGrid" '
            'version="0.1" byte_order="LittleEndian"><UnstructuredGrid>'
            f'<Piece NumberOfPoints="{len(points)}" '
            f'NumberOfCells="{len(tets)}"><Points><DataArray type="Float64" '
            f'NumberOfComponents="3" format="ascii">{ascii(points)}'
            '</DataArray></Points><Cells><DataArray type="Int32" '
            f'Name="connectivity" format="ascii">'
            f'{" ".join(str(i) for i in np.ravel(tets))}</DataArray>'
            '<DataArray type="Int32" Name="offsets" format="ascii">'
            f'{" ".join(str(4 * (i + 1)) for i in range(len(tets)))}'
            '</DataArray><DataArray type="UInt8" Name="types" '
            f'format="ascii">{" ".join("10" for _ in tets)}</DataArray>'
            f'</Cells><PointData>{arrays}</PointData></Piece>'
            '</UnstructuredGrid></VTKFile>')


def check_smoothing_per_body(root):
    """Two bodies share a face (coincident points, PolyFEM duplicates every
    element's vertices); smoothing must average within each body only."""
    work = os.path.join(root, "two_bodies")
    os.makedirs(work)
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    points = np.vstack([base, [[0, 0, 1]],        # body 1: apex above
                        base, [[0, 0, -1]],       # body 2: apex below
                        base, [[1, 1, 1]]])       # body 1, second element
    tets = np.array([[0, 1, 2, 3], [4, 6, 5, 7], [8, 9, 10, 11]])
    body = np.array([1] * 4 + [2] * 4 + [1] * 4, dtype=float)
    stress = np.array([10.0] * 4 + [100.0] * 4 + [20.0] * 4)
    write_vtu(os.path.join(work, "f0.vtu"), points, tets,
              {"solution": np.zeros((12, 3)), "body_ids": body,
               "von_mises": stress})
    pvd = os.path.join(work, "two.pvd")
    with open(pvd, "w") as handle:
        handle.write('<?xml version="1.0"?><VTKFile type="Collection" '
                     'version="0.1"><Collection><DataSet timestep="0" '
                     'file="f0.vtu"/></Collection></VTKFile>')
    viewer = hou.node("/obj").createNode("readPVD::1.0", "two_bodies")
    phm = viewer.hdaModule()
    hou.setFrame(0)
    viewer.setParms({"PVD_file": pvd, "surface_only": 0})
    phm.start({"node": viewer})
    assert viewer.evalParm("has_multibody") == 1
    viewer.setParms({"color_attrib": "von_mises", "smooth_field": 1})
    result = viewer.node("OUT_result")
    result.cook(force=True)
    geo = result.geometry()
    values = np.frombuffer(geo.pointFloatAttribValuesAsString("for_color"),
                           dtype=np.float32)
    bodies = np.frombuffer(geo.pointFloatAttribValuesAsString("body_ids"),
                           dtype=np.float32)
    positions = np.frombuffer(geo.pointFloatAttribValuesAsString("P"),
                              dtype=np.float32).reshape(-1, 3)
    shared = np.all(np.isclose(positions[:, None], base[None]), axis=2).any(1)
    # On the shared face: body 1 averages 10 and 20, body 2 keeps 100.
    assert np.allclose(values[shared & (bodies == 1)], 15.0), values
    assert np.allclose(values[shared & (bodies == 2)], 100.0), values
    phm.autoscale_all({"node": viewer})
    assert (viewer.evalParm("color_min"), viewer.evalParm("color_max")) == \
        (10.0, 100.0), (viewer.evalParm("color_min"),
                        viewer.evalParm("color_max"))
    print("PASS: smoothing averages within one body where two bodies touch")


def check_contact_forces_default(root):
    work = os.path.join(root, "contact_default")
    os.makedirs(work)
    node = new_scene(work, "contact_default")
    load(node, 1, write_box(os.path.join(work, "box.msh"), n=2))
    sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    for enabled in (1, 0):
        node.setParms({"enable": enabled})
        params = node.hdaModule().write_params_only({"node": node})
        options = json.load(open(params))["output"]["paraview"]
        assert options["options"]["contact_forces"] is bool(enabled), options
        assert ("contact_forces" in options["fields"]) is bool(enabled)
    assert node.evalParm("contact_forces_fields") == 1
    print("PASS: contact forces are written by default with contact on, and "
          "not without contact")


# SHA-256 of the official PyPI h5py 3.16.0 wheels the asset embeds.
WHEEL_DIGESTS = {
    "h5py-3.16.0-cp313-macos-arm64.whl.b64": (
        "macosx_11_0_arm64",
        "42108e93326c50c2810025aade9eac9d6827524cdccc7d4b75a546e5ab308edb"),
    "h5py-3.16.0-cp313-macos-x86_64.whl.b64": (
        "macosx_10_13_x86_64",
        "370a845f432c2c9619db8eed334d1e610c6015796122b0e57aa46312c22617d9"),
    "h5py-3.16.0-cp313-linux-x86_64.whl.b64": (
        "manylinux_2_28_x86_64",
        "9300ad32dea9dfc5171f94d5f6948e159ed93e4701280b0f508773b3f582f402"),
    "h5py-3.16.0-cp313-windows-amd64.whl.b64": (
        "win_amd64",
        "18f2bbcd545e6991412253b98727374c356d67caa920e68dc79eab36bf5fedad"),
}


def check_hdf5_portability(root):
    import base64
    import hashlib
    import io
    import zipfile
    reader_type = hou.nodeType(hou.objNodeTypeCategory(), "readPVD::1.0")
    phm = reader_type.hdaModule()
    sections = reader_type.definition().sections()
    assert set(phm.EMBEDDED_H5PY_SECTIONS.values()) == set(WHEEL_DIGESTS)
    for name, (tag, digest) in WHEEL_DIGESTS.items():
        wheel = base64.b64decode(sections[name].contents())
        assert hashlib.sha256(wheel).hexdigest() == digest, name
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            meta = next(n for n in archive.namelist()
                        if n.endswith(".dist-info/WHEEL"))
            assert f"Tag: cp313-cp313-{tag}" in archive.read(meta).decode()
    assert phm.embedded_h5py_section(("darwin", "arm64"))[0] == \
        "h5py-3.16.0-cp313-macos-arm64.whl.b64"
    assert phm.embedded_h5py_section(("darwin", "x86_64"))[0] == \
        "h5py-3.16.0-cp313-macos-x86_64.whl.b64"
    assert phm.embedded_h5py_section(("win32", "amd64"))[0] == \
        "h5py-3.16.0-cp313-windows-amd64.whl.b64"
    section, reason = phm.embedded_h5py_section(("sunos5", "sparc"))
    assert section is None and "no bundled HDF5 reader" in reason
    section, reason = phm.embedded_h5py_section(python=(3, 11))
    assert section is None and "Python 3.13" in reason
    print("PASS: Read PVD carries the official h5py wheels for macOS (both), "
          "Linux x86_64 and Windows x64, and picks the platform's one")

    # Where Read PVD could not open HDF5, the PolyFEM node writes XML.
    work = os.path.join(root, "xml_fallback")
    os.makedirs(work)
    node = new_scene(work, "xml_fallback")
    load(node, 1, write_box(os.path.join(work, "box.msh"), n=2))
    node.setParms({"quasistatic": 1, "end_time_bool": 1, "tend": 1.0,
                   "time_inc_bool": 1, "dt": 1.0, "enable": 0})
    sideset(node, 1, "axis:-z:0.01", 0, "[0, 0, 0]")
    sideset(node, 1, "axis:+z:0.99", 0, '[0, 0, "-0.05*t"]')
    module = node.hdaModule()
    assert module.hdf5_readable_here()[0], module.hdf5_readable_here()
    os.environ["POLYFEM_HDA_HDF5_READER"] = "none"
    try:
        report = module.check_setup(node, "write")
        assert any("written as XML .vtu files" in note
                   for note in report["notes"]), report
        params, pvd = run(node)
    finally:
        del os.environ["POLYFEM_HDA_HDF5_READER"]
    options = json.load(open(params))["output"]["paraview"]["options"]
    assert options["use_hdf5"] is False, options
    output = os.path.dirname(pvd)
    assert any(name.endswith(".vtu") for name in os.listdir(output))
    assert not any(name.endswith(".hdf") for name in os.listdir(output))
    viewer = hou.node("/obj").createNode("readPVD::1.0", "xml_viewer")
    viewer.setParms({"PVD_file": pvd})
    viewer.hdaModule().start({"node": viewer})
    assert viewer.evalParm("color_max") > 0.0
    assert node.evalParm("use_hdf5") == 1, "the toggle itself was changed"
    print("PASS: without an HDF5 reader the node writes XML (and says so); "
          "Use HDF5 itself stays on")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(os.path.join(
        BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"))
    root = tempfile.mkdtemp(prefix="readpvd_first_look_")
    check_first_look(root)
    check_slash_fields(root)
    check_smoothing_per_body(root)
    check_contact_forces_default(root)
    check_hdf5_portability(root)
    print("workdir:", root)


if __name__ == "__main__":
    main()
