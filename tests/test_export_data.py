"""Read PVD's Export > Data Over Time (2026-10-03): an Excel workbook of
statistics per output step, for presentation plots.

Checked with the real solver, on a bar under uniaxial stress (linear
elastic, symmetry planes: the stress is the same in every element, E strain,
and the displacement field is known in closed form):

* every statistic of von Mises over the whole model equals E strain at every
  step (max, min, both means, percentiles; SD 0; the count of elements);
  the integral equals E strain x the region's volume (current or rest,
  differing by J); the rest volume is the bar's volume exactly;
* nodal displacement: the largest magnitude is the closed-form corner value,
  the sideset's mean z displacement the prescribed one; one node by its .msh
  tag and one element by number give a single value column;
* regions: whole model, bodies (named after the Gmsh group), a sideset, a
  box, one element, one node, Set From Probe -- element and node counts and
  values against an independent computation;
* time columns: step, simulation time and the Houdini frame of the
  PolyFEM node's timeline;
* units: none when the run did not record any; chosen system and kPa / mm
  convert exactly and label the headers; the PolyFEM node's recorded units
  are used automatically;
* the force curves as columns equal Analysis > Force Curves;
* the workbook: one header row, numbers only (NaN = empty cell), the per
  element sheet with positions, the About sheet with the run id and input
  hash; read back by our own reader and, when LibreOffice is installed, by
  LibreOffice (every sheet);
* P2 tetrahedra and Q1 hexahedra: exact region volumes.

Run: hython tests/test_export_data.py
"""

import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import hou

sys.dont_write_bytecode = True
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_force_curves import Scene, X, Y, Z, column  # noqa: E402

E, NU = 1e5, 0.3
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATION = ("{http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships}id")


def read_xlsx(path):
    """{sheet name: rows} with numbers as floats, text as str, empty = None."""
    archive = zipfile.ZipFile(path)
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target") for r in relations}
    sheets = {}
    for sheet in workbook.iter(MAIN + "sheet"):
        root = ET.fromstring(archive.read("xl/" + targets[sheet.get(
            RELATION)]))
        rows = []
        for number, row in enumerate(root.iter(MAIN + "row"), 1):
            assert int(row.get("r")) == number
            cells = {}
            for cell in row:
                letters = re.match(r"[A-Z]+", cell.get("r")).group(0)
                index = 0
                for letter in letters:
                    index = index * 26 + ord(letter) - 64
                if cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(MAIN + "t"))
                else:
                    value = float(cell.find(MAIN + "v").text)
                cells[index - 1] = value
            width = max(cells) + 1 if cells else 0
            rows.append([cells.get(i) for i in range(width)])
        sheets[sheet.get("name")] = rows
    return sheets


def table(rows):
    """Data sheet rows -> {header: column values (None for empty)}."""
    header = rows[0]
    out = {}
    for index, name in enumerate(header):
        out[name] = [row[index] if index < len(row) else None
                     for row in rows[1:]]
    return out


def set_quantity(reader, index, field, value="", basis=0, stats=(),
                 percentiles="", volume=0, label="", where=False):
    names = ("max", "min", "mean", "amean", "integral", "sum", "rms", "std",
             "count")
    reader.setParms({f"export_q_field{index}": field,
                     f"export_q_value{index}": value,
                     f"export_q_basis{index}": basis,
                     f"export_q_percentiles{index}": percentiles,
                     f"export_q_volume{index}": volume,
                     f"export_q_label{index}": label,
                     f"export_q_where{index}": int(where)})
    for name in names:
        reader.parm(f"export_q_{name}{index}").set(int(name in stats))


def set_region(reader, index, kind, name="", **values):
    kinds = ("all", "bodies", "sideset", "box", "sphere", "plane", "element",
             "node")
    reader.setParms({f"export_region_kind{index}": kinds.index(kind),
                     f"export_region_name{index}": name})
    for key, value in values.items():
        parm = reader.parmTuple(f"export_region_{key}{index}")
        parm.set(value if isinstance(value, (tuple, list)) else (value,))


def export(reader):
    summary = reader.hdaModule().export_data(reader)
    sheets = read_xlsx(summary["path"])
    return summary, sheets


def uniaxial(root):
    """The bar of test_force_curves.check_uniaxial: linear elastic, nu 0.3,
    symmetry planes, top pulled by 0.002 t (strain 0.001 t), 4 steps."""
    scene = Scene(root, "uniaxial")
    scene.material("LinearElasticity")
    scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", Z))
    scene.sideset("axis:-x:0.01", (0, "[0, 0, 0]", X))
    scene.sideset("axis:-y:0.01", (0, "[0, 0, 0]", Y))
    scene.sideset("axis:+z:1.99", (0, '[0, 0, "0.002*t"]', Z))
    reader = scene.run()
    return scene, reader


def check_statistics(root):
    scene, reader = uniaxial(root)
    module = reader.hdaModule()
    path = reader.evalParm("PVD_file")
    times = np.array([t for t, _ in module.read_pvd(path)])
    strain = 0.001 * times
    stress = E * strain
    reader.setParms({"export_regions": 1, "export_quantities": 4,
                     "export_forces": 0})
    set_region(reader, 1, "all", "Bar")
    set_quantity(reader, 1, "von_mises_derived", "auto", stats=(
        "max", "min", "mean", "amean", "integral", "sum", "rms", "std",
        "count"), percentiles="50 95")
    set_quantity(reader, 2, "cauchy_mat", "z", stats=("mean",),
                 label="sigma_zz")
    set_quantity(reader, 3, "solution", "magnitude", stats=("max",))
    set_quantity(reader, 4, "_volume", volume=0)
    summary, sheets = export(reader)
    assert list(sheets) == ["Data", "About"], list(sheets)
    data = table(sheets["Data"])
    assert sheets["Data"][0][:3] == ["Step", "Time", "Frame"], \
        sheets["Data"][0][:3]
    assert data["Step"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.allclose(data["Time"], times, rtol=0, atol=0)
    assert data["Frame"] == [1.0, 7.0, 13.0, 19.0, 25.0], data["Frame"]
    header = "Bar: von Mises stress"
    count = 6 * 2 ** 3  # write_box(n=2): 8 cubes of 6 tets
    for name in ("max", "min", "mean", "arithmetic mean", "RMS", "P50",
                 "P95"):
        values = np.array(data[f"{header}, {name}"])
        assert np.allclose(values, stress, rtol=1e-7, atol=1e-9), \
            (name, values, stress)
    assert np.allclose(data[f"{header}, SD"], 0, atol=1e-7 * stress.max())
    assert data[f"{header}, count"] == [float(count)] * 5
    assert np.allclose(data[f"{header}, sum"], count * stress, rtol=1e-7,
                       atol=1e-9)
    volume = np.array(data["Bar: volume"])
    jacobian = (1 + strain) * (1 - NU * strain) ** 2
    assert np.allclose(volume, 2.0 * jacobian, rtol=1e-12), (volume,
                                                             jacobian)
    assert np.allclose(data[f"{header}, integral"], stress * volume,
                       rtol=1e-7, atol=1e-9)
    assert np.allclose(data["Bar: sigma_zz, mean"], stress, rtol=1e-7,
                       atol=1e-9)
    corner = strain * np.sqrt(4 + 2 * NU ** 2)  # u at (1, 1, 2)
    assert np.allclose(data["Bar: displacement magnitude, max"], corner,
                       rtol=1e-7, atol=1e-12), (
        data["Bar: displacement magnitude, max"], corner)
    print("PASS: uniform stress: max = min = both means = P50 = P95 = RMS = "
          "E strain at every step, SD 0, count = elements, sum = count x "
          "value, integral = value x current volume; the current volume is "
          "2 J; the largest displacement is the closed-form corner value")

    # rest volumes: the integral uses the undeformed volume
    set_quantity(reader, 1, "von_mises_derived", "auto",
                 stats=("integral",), volume=1)
    set_quantity(reader, 4, "_volume", volume=1)
    _, sheets = export(reader)
    data = table(sheets["Data"])
    assert np.allclose(data["Bar: volume (rest)"], 2.0, rtol=1e-14)
    assert np.allclose(data["Bar: von Mises stress, integral"], 2.0 * stress,
                       rtol=1e-7, atol=1e-9)
    print("PASS: with rest volumes the volume is exactly the bar's 2 and the "
          "integral E strain x 2")
    return scene, reader


def check_regions(scene, reader):
    module = reader.hdaModule()
    path = reader.evalParm("PVD_file")
    times = np.array([t for t, _ in module.read_pvd(path)])
    strain = 0.001 * times
    reader.setParms({"export_regions": 5, "export_quantities": 2})
    set_region(reader, 1, "bodies", "", bodies="1001")
    set_region(reader, 2, "sideset")
    sets = module._force_definitions(reader)[1]
    grip = [i for i, entry in enumerate(sets, 1)
            if entry.get("sideset") == 4][0]
    reader.parm("export_region_sideset2").set(str(grip))
    set_region(reader, 3, "box", "Lower half", box_min=(-1, -1, -1),
               box_max=(2, 2, 1.0))
    set_region(reader, 4, "node", "Corner", geometry=1, number=27)
    set_region(reader, 5, "element", "", geometry=1, number=1)
    set_quantity(reader, 1, "solution", "z", basis=2, stats=("mean", "count"))
    set_quantity(reader, 2, "von_mises_derived", "auto", basis=1,
                 stats=("max", "count"))
    summary, sheets = export(reader)
    data = table(sheets["Data"])
    regions = summary["regions"]
    counts = [(len(r["elements"]), len(r["nodes"])) for r in regions]
    # bodies: everything (48 tets, 27 nodes); the top sideset: 9 nodes and
    # 8 tets (2 x 2 cubes x 2 triangles, one tet each); the lower half:
    # centroids z <= 1 (24 tets), nodes z <= 1 (18)
    assert counts[0] == (48, 27), counts
    assert counts[1][1] == 9 and counts[1][0] == 8, counts
    assert counts[2] == (24, 18), counts
    assert counts[3][1] == 1 and counts[4][0] == 1, counts
    label = regions[0]["label"]
    assert data[f"{label}: displacement z, count"] == [27.0] * 5
    top = data["Sideset 4: displacement z, mean"]
    assert np.allclose(top, 0.002 * times, rtol=0, atol=1e-12), top
    # node 27 = (1, 1, 2), the top corner of write_box(n=2, hi z=2)
    assert np.allclose(data["Corner: displacement z"], 0.002 * times,
                       atol=1e-12), data["Corner: displacement z"]
    assert np.allclose(data["Element 1: von Mises stress"], E * strain,
                       rtol=1e-7, atol=1e-9)
    assert data["Lower half: von Mises stress, count"] == [24.0] * 5
    # set from the probe: the probed point's node and its rest position
    output = reader.node("output")
    geo = output.geometry()
    rest = np.array([p.attribValue("rest") for p in geo.points()])
    corner = int(np.argmin(np.linalg.norm(rest - [1.0, 1.0, 2.0], axis=1)))
    reader.parm("probe_point").set(corner)
    reader.parm("export_region_kind4").set(7)  # node
    module.export_region_from_probe({"node": reader,
                                     "script_multiparm_index": "4"})
    assert reader.evalParm("export_region_number4") == 27
    reader.parm("export_region_kind3").set(4)  # sphere
    module.export_region_from_probe({"node": reader,
                                     "script_multiparm_index": "3"})
    assert np.allclose(reader.evalParmTuple("export_region_center3"),
                       (1.0, 1.0, 2.0))
    print("PASS: regions: bodies, the grip sideset (9 nodes, 8 elements "
          "with a face on it, mean z displacement = 0.002 t), a box (24 "
          "elements, 18 nodes), node 27 by its .msh tag, element 1; Set From "
          "Probe fills the node and a sphere's centre")


def check_units_and_forces(scene, reader):
    module = reader.hdaModule()
    path = reader.evalParm("PVD_file")
    times = np.array([t for t, _ in module.read_pvd(path)])
    stress = E * 0.001 * times
    reader.setParms({"export_regions": 1, "export_quantities": 3,
                     "export_forces": 1, "export_units": 0})
    set_region(reader, 1, "all", "Bar")
    set_quantity(reader, 1, "von_mises_derived", "auto", stats=("max",
                                                                 "integral"))
    set_quantity(reader, 2, "solution", "z", basis=2, stats=("max",))
    set_quantity(reader, 3, "_volume")
    _, sheets = export(reader)
    header = sheets["Data"][0]
    assert not any("(" in str(h) for h in header if h.startswith("Bar")), \
        header
    assert "units" in reader.evalParm("export_units_status").lower()
    # forces as columns equal Analysis > Force Curves
    curves = module.force_curves(reader)
    grip = column(curves, "Geometry 1 subdomain 1 sideset 4 ")
    data = table(sheets["Data"])
    assert np.allclose(data["Sideset 4: reaction z"],
                       curves["reaction"][:, grip, 2], rtol=0, atol=0)
    assert np.allclose(data["Sideset 4: mean displacement z"],
                       curves["displacement"][:, grip, 2], rtol=0, atol=0)
    assert np.allclose(data["Sideset 4: reaction z"], stress * 1.0,
                       rtol=1e-6, atol=1e-6)
    print("PASS: without recorded units no header carries a unit; the force "
          "curves come along as columns equal to Analysis > Force Curves "
          "(grip reaction = E A strain)")

    # chosen system, shown in kPa and mm
    reader.setParms({"export_units": 2, "export_unit_stress": 2,
                     "export_unit_length": 3, "export_unit_force": 2})
    _, sheets = export(reader)
    data = table(sheets["Data"])
    assert np.allclose(data["Bar: von Mises stress, max (kPa)"],
                       stress / 1e3, rtol=1e-7)
    assert np.allclose(data["Bar: displacement z, max (mm)"],
                       0.002 * times * 1e3, rtol=1e-9, atol=1e-12)
    volume = np.array(data["Bar: volume (mm³)"])
    assert np.allclose(volume / 1e9, 2.0 * (1 + 0.001 * times)
                       * (1 - NU * 0.001 * times) ** 2, rtol=1e-12)
    assert np.allclose(data["Bar: von Mises stress, integral (kPa*mm³)"],
                       stress / 1e3 * volume, rtol=1e-7, atol=1e-9)
    assert np.allclose(data["Sideset 4: reaction z (mN)"],
                       data["Sideset 4: reaction z (mN)"][-1] / times[-1]
                       * times, rtol=1e-6)
    print("PASS: a chosen unit system with kPa / mm / mN converts exactly and "
          "labels every header (volume mm³, integral kPa*mm³)")

    # recorded by the PolyFEM node: mm, g, s -> stress in Pa, force in uN
    scene.node.setParms({"units": 1, "length": "mm", "mass": "g",
                         "time": "s"})
    scene.mod.write_params_only({"node": scene.node})
    module._PARAMS_CACHE.clear()
    reader.setParms({"export_units": 0, "export_unit_stress": 0,
                     "export_unit_length": 0, "export_unit_force": 0})
    status = reader.evalParm("export_units_status")
    assert status.startswith("Run: mm, g, s (recorded"), status
    _, sheets = export(reader)
    header = sheets["Data"][0]
    assert "Bar: von Mises stress, max (Pa)" in header, header
    assert "Bar: displacement z, max (mm)" in header, header
    assert "Sideset 4: reaction z (µN)" in header, header
    assert "Time (s)" in header, header
    print("PASS: the PolyFEM node's recorded units (mm, g, s) label the "
          "headers automatically: Pa, mm, µN, s")
    return sheets


def check_workbook(reader):
    module = reader.hdaModule()
    reader.setParms({"export_regions": 2, "export_quantities": 1,
                     "export_per_item": 1, "export_per_item_steps": 0,
                     "export_forces": 0})
    set_region(reader, 1, "all", "Bar")
    set_region(reader, 2, "box", "Corner box", box_min=(0.5, 0.5, 1.5),
               box_max=(1.1, 1.1, 2.1))
    set_quantity(reader, 1, "von_mises_derived", "auto", stats=("max",),
                 where=True)
    summary, sheets = export(reader)
    assert list(sheets) == ["Data", "von Mises stress per element",
                            "About"], list(sheets)
    items = sheets["von Mises stress per element"]
    header = items[0]
    assert header[:4] == ["Region", "Geometry", "Element", "Body"], header
    assert header[4].startswith("x") and header[-1].startswith("Step 4"), \
        header
    bar = [row for row in items[1:] if row[0] == "Bar"]
    assert len(bar) == 48 and [row[2] for row in bar] == list(
        map(float, range(1, 49)))
    assert np.allclose([row[-1] for row in bar], E * 0.001, rtol=1e-7)
    data = table(sheets["Data"])
    assert "Bar: von Mises stress, max at element" in data
    about = {row[0]: row[1] for row in sheets["About"] if len(row) == 2}
    manifest = json.load(open(os.path.join(os.path.dirname(
        reader.evalParm("PVD_file")), "run-manifest.json")))
    assert about["Run id"] == manifest["run_id"], about.get("Run id")
    assert about["Input params.json sha256"] == \
        manifest["input"]["file"]["sha256"]
    assert "Corner box" in about and "elements" in about["Corner box"]
    # the same settings give the same headers
    again, sheets_again = export(reader)
    assert sheets_again["Data"][0] == sheets["Data"][0]
    # a step missing a field: empty cells, not zeros
    reader.setParms({"export_quantities": 1})
    set_quantity(reader, 1, "velocity", "magnitude", basis=2, stats=("max",))
    try:
        module.export_data(reader)
    except module.ExportError as exc:
        assert "velocity" in str(exc), exc
    else:
        raise AssertionError("a field the result lacks was accepted")
    print("PASS: the per element sheet lists every element with its position "
          "and value, Where the Max Is adds its element, the About sheet "
          "carries the run id and input hash, headers are stable, and a "
          "field the result lacks is refused by name")
    return summary["path"], sheets


def check_unreadable_step(root, reader):
    """A damaged last step (a run still writing) gives empty cells."""
    work = os.path.dirname(os.path.dirname(reader.evalParm("PVD_file")))
    copy = os.path.join(root, "damaged_copy")
    shutil.copytree(work, copy)
    module = reader.hdaModule()
    last = module.read_pvd(os.path.join(copy, "output", "sim.pvd"))[-1][1]
    block = module.read_vtm(last)["Volume"] if last.endswith(".vtm") \
        else last
    with open(block, "r+b") as handle:
        handle.truncate(200)
    damaged = hou.node("/obj").createNode("readPVD::1.0", "damaged")
    damaged.parm("PVD_file").set(os.path.join(copy, "output", "sim.pvd"))
    module.start({"node": damaged})
    damaged.setParms({"export_regions": 1, "export_quantities": 1,
                      "export_forces": 1, "export_per_item": 0,
                      "export_units": 1})  # no unit labels
    set_region(damaged, 1, "all", "Bar")
    set_quantity(damaged, 1, "von_mises_derived", "auto", stats=("max",))
    summary, sheets = export(damaged)
    data = table(sheets["Data"])
    assert data["Bar: von Mises stress, max"][-1] is None
    assert data["Bar: von Mises stress, max"][-2] is not None
    assert data["Sideset 4: reaction z"][-1] is None
    assert data["Step"][-1] == 4.0
    assert "could not be read" in summary["status"], summary["status"]
    print("PASS: a step that cannot be read (a run still writing) gives "
          "empty cells and a note; the other steps are exported")


def check_libreoffice(path, sheets):
    soffice = shutil.which("soffice") or (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if os.path.isfile("/Applications/LibreOffice.app/Contents/MacOS/"
                          "soffice") else None)
    if soffice is None:
        print("SKIP: LibreOffice is not installed; the workbook was read "
              "back by this test's own reader only")
        return
    out = tempfile.mkdtemp(prefix="xlsx_lo_")
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    result = subprocess.run(
        [soffice, f"-env:UserInstallation=file://{profile}", "--headless",
         "--convert-to", "csv:Text - txt - csv (StarCalc):44,34,76,1,,0,"
         "false,true,false,false,false,-1", "--outdir", out, path],
        capture_output=True, text=True, timeout=300)
    produced = {os.path.basename(p): p for p in glob.glob(
        os.path.join(out, "*.csv"))}
    assert result.returncode == 0 and len(produced) == len(sheets), \
        (result.stdout, result.stderr, produced)
    stem = os.path.splitext(os.path.basename(path))[0]
    worst = 0.0
    for name, rows in sheets.items():
        with open(produced[f"{stem}-{name}.csv"], newline="",
                  encoding="utf-8") as handle:
            lo = list(csv.reader(handle))
        for mine, theirs in zip(rows, lo):
            for a, b in zip(mine, theirs):
                if isinstance(a, float):
                    worst = max(worst, abs(float(b) - a)
                                / max(abs(a), 1e-300))
                elif a is None:
                    assert b == "", (name, a, b)
                else:
                    assert a == b, (name, a, b)
    assert worst < 1e-12, worst
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(profile, ignore_errors=True)
    print(f"PASS: LibreOffice reads every sheet back (largest relative "
          f"difference {worst:.1e})")


def check_elements(root):
    for label, hexes, order in (("p2", False, 2), ("q1", True, 1)):
        scene = Scene(root, f"volume_{label}", hexes=hexes, dt=1.0)
        scene.material("LinearElasticity", nu=0.0)
        if order == 2:
            scene.node.parm("mainOrder1_1").set(1)
        scene.sideset("axis:-z:0.01", (0, "[0, 0, 0]", Z))
        scene.sideset("axis:+z:1.99", (0, '["0.1*t", "0", "0.2*t"]', (1, 1, 1)))
        reader = scene.run()
        reader.setParms({"export_regions": 1, "export_quantities": 2,
                         "export_forces": 0})
        set_region(reader, 1, "all", "Bar")
        set_quantity(reader, 1, "_volume", volume=1)
        set_quantity(reader, 2, "_volume", volume=0)
        _, sheets = export(reader)
        data = table(sheets["Data"])
        assert np.allclose(data["Bar: volume (rest)"], 2.0, rtol=1e-13)
        module = reader.hdaModule()
        mesh = module.volume_mesh(reader.evalParm("PVD_file"), 1)
        topology = module.volume_topology(mesh)
        moved = mesh["points"] + mesh["point_data"]["solution"]
        # the current volume against Read PVD's display sub-cells (P2) or a
        # fine Gauss rule (hexes are exact either way)
        if order == 2:
            cells = mesh["cells"]["tet"]
            reference = np.abs(module._tet_volumes(
                *(moved[cells[:, i]] for i in range(4)))).sum()
            assert abs(data["Bar: volume"][-1] - reference) \
                < 1e-3 * reference, (data["Bar: volume"][-1], reference)
        else:
            del topology
        print(f"PASS: {label.upper()} elements: rest volume of the bar exactly "
              f"2, current volume {data['Bar: volume'][-1]:.9g}")


def check_hex_volume(module):
    """A twisted trilinear hex: 2 x 2 x 2 Gauss is exact (against 5 x 5 x 5)."""
    rng = np.random.default_rng(3)
    corners = module._HEX_REFERENCE + 0.2 * rng.standard_normal((8, 3))
    points, weights = np.polynomial.legendre.leggauss(5)
    points, weights = 0.5 * (points + 1), 0.5 * weights
    volume = 0.0
    for u, wu in zip(points, weights):
        for v, wv in zip(points, weights):
            for w, ww in zip(points, weights):
                dN = np.empty((8, 3))
                for k, (a, b, c) in enumerate(module._HEX_REFERENCE):
                    fu, fv, fw = (u if a else 1 - u), (v if b else 1 - v), \
                        (w if c else 1 - w)
                    dN[k] = ((1 if a else -1) * fv * fw,
                             fu * (1 if b else -1) * fw,
                             fu * fv * (1 if c else -1))
                volume += wu * wv * ww * np.linalg.det(corners.T @ dN)
    assert abs(module.hex_volumes(corners[None])[0] - volume) < 1e-14
    print("PASS: a twisted trilinear hex's volume is exact (2 x 2 x 2 Gauss "
          "equals 5 x 5 x 5)")


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    for library in ("sop_MSH_Reader.3.0.hdanc",
                    "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
                    "object_readPVD.1.0.hdanc"):
        hou.hda.installFile(os.path.join(BASE, library),
                            force_use_assets=True)
    hou.setFps(24)
    root = tempfile.mkdtemp(prefix="readpvd_export_data_")
    scene, reader = check_statistics(root)
    check_regions(scene, reader)
    check_units_and_forces(scene, reader)
    path, sheets = check_workbook(reader)
    check_libreoffice(path, sheets)
    check_unreadable_step(root, reader)
    check_elements(root)
    check_hex_volume(reader.hdaModule())
    print("workdir:", root)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
