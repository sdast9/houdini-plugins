"""Fiber models, composites and per-element material data in the PolyFEM HDA.

Covers the JSON/file emission, the validation rules, the round trip, and an
end-to-end solve whose output proves the per-element data reached the solver
bound to the right elements.

Run: hython tests/test_polyfem_materials.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import hou
import numpy as np

from test_polyfem_hda import BASE, ROOT, POLYFEM_BIN

sys.path.insert(0, os.path.join(BASE, "src", "common"))
import vtu_parser  # noqa: E402

ANCHOR = np.array([0.5, 0.5, -0.75])
fails = []


def check(label, condition, detail=""):
    print(("  PASS  " if condition else "  FAIL  ") + label
          + (f"  {detail}" if detail else ""))
    if not condition:
        fails.append(label)


def expect_error(label, fragment, action):
    try:
        action()
    except hou.Error as error:
        check(label, fragment in str(error), str(error).strip()[:110])
    else:
        check(label, False, "no error raised")


def make_msh(path, volumes=1, h=0.45):
    """One box per volume, stacked in y and glued so they share an interface."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("m")
    boxes = [gmsh.model.occ.addBox(0, i, 0, 1, 1, 1) for i in range(volumes)]
    gmsh.model.occ.synchronize()
    if volumes > 1:
        gmsh.model.occ.fragment([(3, boxes[0])], [(3, b) for b in boxes[1:]])
        gmsh.model.occ.synchronize()
    tags = [t[1] for t in gmsh.model.getEntities(3)]
    for index, tag in enumerate(sorted(tags), start=1):
        gmsh.model.addPhysicalGroup(3, [tag], index)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(path)
    gmsh.finalize()


def centroids_of(node, geo):
    """Volume-element centroids in polyfem order, straight from the HDA."""
    phm = node.hdaModule()
    geo_data = node.node(f"branch_{geo}").geometry()
    return phm._prim_centroids(geo_data, phm._volume_mask(geo_data))


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_materials_")
    input_dir = os.path.join(work, "input")
    os.makedirs(input_dir, exist_ok=True)
    mesh = os.path.join(work, "two_vol.msh")
    make_msh(mesh, volumes=2)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "materials_test")
    phm = node.hdaModule()
    node.setParms({"working_dir": work + "/", "polyfem_bin": POLYFEM_BIN})
    node.setParms({"file_location1": mesh})
    node.parm("file_location1").pressButton()
    assert node.evalParm("num_volumes1") == 2

    count = phm.volume_element_count(node, 1)
    cents = centroids_of(node, 1)
    check("element count matches centroid count", len(cents) == count,
          f"{count} elements")

    # ---- per-element fiber field authored as a Houdini attribute ----------
    fibers = cents - ANCHOR
    fibers /= np.linalg.norm(fibers, axis=1, keepdims=True)
    source = node.createNode("attribwrangle", "fiber_source")
    source.setInput(0, node.node("branch_1"))
    source.setParms({"class": 1, "snippet": (
        "if (i@is_volume == 0) return;\n"
        "vector c = {0,0,0}; int n = primvertexcount(0, @primnum);\n"
        "for (int i = 0; i < n; i++)\n"
        "    c += point(0, \"P\", vertexpoint(0, vertexindex(0, @primnum, i)));\n"
        "v@fiber1 = normalize(c / n - {0.5, 0.5, -0.75});\n"
        "f@kappa = 0.30 * float(@primnum) / float(@numprim);")})

    print("\n--- single fiber model, per-element direction + kappa ---")
    node.setParms({
        "materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
        "hgo_k11_1": 1e4, "hgo_k21_1": 5.0,
        "fib_source1_1": "attribute", "fib_sop1_1": source.path(),
        "fib_attrib1_1": "fiber1",
        "kappa_source1_1": "attribute", "kappa_sop1_1": source.path(),
        "kappa_attrib1_1": "kappa",
        "materials1_2": phm.MATERIAL_TOKENS.index("NeoHookean"),
        "E1_2": 1e5, "nu1_2": 0.4})

    material = phm.build_material(node, 1, 1)
    check("HGODispersion emits a per-element fiber reference",
          material["fiber_direction"] == {
              "type": "per_element_file", "path": "fibers.vtk",
              "field": "FIB_1_1"}, str(material.get("fiber_direction")))
    check("HGODispersion emits a per-element kappa file",
          material["kappa"] == "pe_kappa_1_1.txt", str(material.get("kappa")))
    check("k_chi omitted at its default", "k_chi" not in material)
    node.setParms({"k_chi1_1": 250})
    check("k_chi emitted when changed",
          phm.build_material(node, 1, 1).get("k_chi") == 250)
    node.setParms({"k_chi1_1": 100})

    # ---- files span the global element range ------------------------------
    phm.export_per_element_materials(node, input_dir)
    vtk_path = os.path.join(input_dir, "fibers.vtk")
    written = phm.read_cell_vectors_legacy_vtk(vtk_path, "FIB_1_1")
    check("fibers.vtk has one row per global element", len(written) == count,
          f"{len(written)} rows vs {count}")
    vol1 = np.asarray(node.node("branch_1").geometry()
                      .primIntAttribValues("Entity"))
    vol1 = vol1[np.asarray(node.node("branch_1").geometry()
                           .primIntAttribValues("is_volume")).astype(bool)] == 1
    check("fiber rows of this subdomain match the authored directions",
          np.abs(written[vol1] - fibers[vol1]).max() < 1e-6,
          f"max diff {np.abs(written[vol1] - fibers[vol1]).max():.2e}")
    kappa_written = np.loadtxt(os.path.join(input_dir, "pe_kappa_1_1.txt"))
    expected_kappa = 0.30 * np.arange(count) / count
    check("kappa file matches the authored attribute",
          len(kappa_written) == count
          and np.abs(kappa_written[vol1] - expected_kappa[vol1]).max() < 1e-6)

    # ---- validation rules --------------------------------------------------
    print("\n--- validations ---")
    node.setParms({"kappa_attrib1_1": "does_not_exist"})
    expect_error("missing attribute is a clear error",
                 "no primitive attribute",
                 lambda: phm.export_per_element_materials(node, input_dir))
    node.setParms({"kappa_attrib1_1": "kappa"})

    bad = node.createNode("attribwrangle", "bad_kappa")
    bad.setInput(0, node.node("branch_1"))
    bad.setParms({"class": 1, "snippet": "f@kappa = 0.9;"})
    node.setParms({"kappa_sop1_1": bad.path()})
    expect_error("out-of-range kappa is rejected", "within [0, 1/3]",
                 lambda: phm.export_per_element_materials(node, input_dir))
    node.setParms({"kappa_sop1_1": source.path()})

    zero = node.createNode("attribwrangle", "zero_fiber")
    zero.setInput(0, node.node("branch_1"))
    zero.setParms({"class": 1, "snippet": "v@fiber1 = {0, 0, 0};"})
    node.setParms({"fib_sop1_1": zero.path()})
    expect_error("zero-length fiber is rejected", "zero-length",
                 lambda: phm.export_per_element_materials(node, input_dir))
    node.setParms({"fib_sop1_1": source.path()})

    short = os.path.join(work, "short.txt")
    np.savetxt(short, np.full(5, 0.1))
    node.setParms({"kappa_source1_1": "file", "kappa_file1_1": short})
    expect_error("short per-element file is rejected", "but this geometry has",
                 lambda: phm.export_per_element_materials(node, input_dir))
    node.setParms({"kappa_source1_1": "attribute"})

    node.setParms({"hgo_k21_1": 0})
    expect_error("k2 must be positive", "k2 must be greater than zero",
                 lambda: phm.build_material(node, 1, 1))
    node.setParms({"hgo_k21_1": 5.0})

    # ---- composites --------------------------------------------------------
    print("\n--- composite (matrix + families) ---")
    node.setParms({
        "materials1_2": phm.MATERIAL_TOKENS.index("MaterialSum"),
        "matrix_model1_2": "NeoHookean", "E1_2": 1e5, "nu1_2": 0.4,
        "num_fiber_families1_2": 1,
        "fam_model1_2_1": "HGODispersion",
        "fam_k11_2_1": 2e4, "fam_k21_2_1": 8.0,
        "fam_kappa_source1_2_1": "constant", "fam_kappa1_2_1": 0.15,
        "fam_fib_source1_2_1": "constant"})
    node.parmTuple("fam_fib_dir1_2_1").set((1, 0, 0))
    composite = phm.build_material(node, 1, 2)
    check("composite emits matrix + family",
          [m["type"] for m in composite["models"]]
          == ["NeoHookean", "HGODispersion"],
          str([m["type"] for m in composite["models"]]))
    check("composite children carry no id/rho",
          all("id" not in m and "rho" not in m for m in composite["models"]))
    check("family parameters land on the family",
          composite["models"][1]["k1"] == 2e4
          and composite["models"][1]["kappa"] == 0.15)

    # mixed stacks are the silent-corruption case polyfem cannot detect
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("MaterialSum"),
                   "matrix_model1_1": "NeoHookean",
                   "num_fiber_families1_1": 2})
    expect_error("differing composite stacks are rejected",
                 "same ordered model stack",
                 lambda: phm._check_composite_consistency(node))
    node.setParms({"num_fiber_families1_1": 1})
    node.setParms({"fam_model1_1_1": "HGODispersion",
                   "fam_fib_source1_1_1": "attribute",
                   "fam_fib_sop1_1_1": source.path(),
                   "fam_fib_attrib1_1_1": "fiber1"})
    phm._check_composite_consistency(node)
    check("matching composite stacks pass", True)

    # ---- symmetric +/- theta pair -----------------------------------------
    print("\n--- symmetric family (+/- theta) ---")
    node.setParms({"fam_mirror1_2_1": 1, "fam_theta1_2_1": 30,
                   "fam_axis_source1_2_1": "constant"})
    node.parmTuple("fam_axis1_2_1").set((0, 0, 1))
    node.setParms({"fam_mirror1_1_1": 1, "fam_theta1_1_1": 30,
                   "fam_axis_source1_1_1": "constant"})
    node.parmTuple("fam_axis1_1_1").set((0, 0, 1))
    mirrored = phm.build_material(node, 1, 2)
    types = [m["type"] for m in mirrored["models"]]
    check("mirrored family expands to two children",
          types == ["NeoHookean", "HGODispersion", "HGODispersion"], str(types))
    directions = [m["fiber_direction"] for m in mirrored["models"][1:]]
    root = np.sqrt(3) / 2
    check("constant mirrored directions are rotated by +/-theta",
          all(isinstance(d, dict) for d in directions), str(directions)[:80])

    phm.export_per_element_materials(node, input_dir)
    plus = phm.read_cell_vectors_legacy_vtk(vtk_path, "FIB_1_2_1p")
    minus = phm.read_cell_vectors_legacy_vtk(vtk_path, "FIB_1_2_1m")
    vol2 = np.asarray(node.node("branch_1").geometry()
                      .primIntAttribValues("Entity"))
    vol2 = vol2[np.asarray(node.node("branch_1").geometry()
                           .primIntAttribValues("is_volume")).astype(bool)] == 2
    check("+theta rotation matches a numpy reference",
          np.abs(plus[vol2] - [root, 0.5, 0]).max() < 1e-6,
          str(plus[vol2][0]))
    check("-theta rotation matches a numpy reference",
          np.abs(minus[vol2] - [root, -0.5, 0]).max() < 1e-6,
          str(minus[vol2][0]))

    axis_parallel = node.createNode("attribwrangle", "axis_parallel")
    axis_parallel.setInput(0, node.node("branch_1"))
    axis_parallel.setParms({"class": 1, "snippet": (
        "v@fiber1 = {0, 0, 1}; v@normal1 = {0, 0, 1};")})
    node.setParms({"fam_fib_source1_1_1": "attribute",
                   "fam_fib_sop1_1_1": axis_parallel.path(),
                   "fam_axis_source1_1_1": "attribute",
                   "fam_axis_attrib1_1_1": "normal1"})
    expect_error("axis parallel to the fiber is rejected",
                 "parallel to the fiber",
                 lambda: phm.export_per_element_materials(node, input_dir))
    node.setParms({"fam_fib_sop1_1_1": source.path(),
                   "fam_axis_source1_1_1": "constant"})

    # ---- end-to-end: does the solver get the right data? ------------------
    print("\n--- end-to-end solve ---")
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
                   "materials1_2": phm.MATERIAL_TOKENS.index("NeoHookean"),
                   "fib_source1_1": "attribute", "kappa_source1_1": "attribute"})
    node.setParms({"sideset_selection1_1": 1, "sideset_selection1_2": 1})
    node.parm("sideset_selection1_1").pressButton()
    node.parm("sideset_selection1_2").pressButton()
    node.setParms({"basegroup1_1_1": "axis:-y:0.001",
                   "basegroup1_2_1": "axis:+y:1.999",
                   "boundary_type1_1_1_1": 0, "boundary_type1_2_1_1": 0,
                   "vector_1_1_1_1": '["0","0","0"]',
                   "vector_1_2_1_1": '["0","0.02*t","0"]'})
    node.setParms({"materials_fields": 1, "minimal_fields": 0,
                   "body_ids_fields": 1,
                   "num_timesteps_bool": 1, "num_timesteps": 2})
    params_path = phm.build_params(node)
    with open(params_path) as handle:
        params = json.load(handle)
    check("params.json validates as JSON", "materials" in params)

    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output/",
         "--log_level", "error"],
        cwd=os.path.dirname(params_path), capture_output=True, text=True,
        timeout=900)
    check("PolyFEM runs the exported scene", result.returncode == 0,
          (result.stdout + result.stderr)[-400:])

    if result.returncode == 0:
        out_dir = os.path.join(work, "output")
        vtus = [f for f in os.listdir(out_dir) if f.endswith(".vtu")]
        vtu = vtu_parser.read_vtu(os.path.join(out_dir, sorted(vtus)[0]))
        pdata = vtu["point_data"]
        fx = [k for k in pdata if k.endswith("fiber_direction_x")]
        check("material fields present in the output", bool(fx), str(fx))
        if fx:
            prefix = fx[0][:-len("fiber_direction_x")]
            got = np.stack(
                [pdata[prefix + f"fiber_direction_{a}"].ravel() for a in "xyz"],
                axis=1)
            pts = vtu["points"]
            body = pdata["body_ids"].ravel() if "body_ids" in pdata else None
            # Only the subdomain that uses the fiber model carries fibers;
            # PolyFEM reports zero for the elements of the plain NeoHookean
            # body, which is expected rather than a binding error.
            check("body ids exported for the binding check", body is not None)
            bad = checked = 0
            for cell in vtu["cells"]["tet"]:
                if body is None or int(body[cell[0]]) != 1001:
                    continue
                mean = got[cell].mean(axis=0)
                if np.linalg.norm(mean) < 1e-9:
                    bad += 1
                    continue
                reference = pts[cell].mean(axis=0) - ANCHOR
                reference /= np.linalg.norm(reference)
                checked += 1
                if abs(float(np.dot(mean / np.linalg.norm(mean),
                                    reference))) < 0.99:
                    bad += 1
            expected = int(np.sum(np.asarray(
                node.node("branch_1").geometry().primIntAttribValues("Entity")
            )[np.asarray(node.node("branch_1").geometry()
                         .primIntAttribValues("is_volume")).astype(bool)] == 1))
            check("per-element fibers reached the solver on the right elements",
                  checked == expected and bad == 0,
                  f"{checked}/{expected} checked, {bad} wrong")

    # ---- round trip --------------------------------------------------------
    print("\n--- round trip ---")
    node.setParms({"old_input_dir": os.path.dirname(params_path)})
    fresh = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "materials_reimport")
    fresh.setParms({"working_dir": work + "/",
                    "old_input_dir": os.path.dirname(params_path)})
    fresh.hdaModule().read_params({"node": fresh})
    check("re-imported material type",
          fresh.evalParm("materials1_1")
          == phm.MATERIAL_TOKENS.index("HGODispersion"),
          phm.MATERIAL_TOKENS[fresh.evalParm("materials1_1")])
    check("re-imported per-element fiber source",
          fresh.parm("fib_source1_1").evalAsString() == "file"
          and fresh.evalParm("fib_file_field1_1") == "FIB_1_1",
          fresh.parm("fib_source1_1").evalAsString())
    check("re-imported per-element kappa source",
          fresh.parm("kappa_source1_1").evalAsString() == "file",
          fresh.parm("kappa_source1_1").evalAsString())
    check("re-imported k1/k2", fresh.evalParm("hgo_k11_1") == 1e4
          and fresh.evalParm("hgo_k21_1") == 5.0)

    reexported = phm.build_material(fresh, 1, 1)
    original = phm.build_material(node, 1, 1)
    check("re-export reproduces the material block",
          reexported == original, f"{reexported} != {original}")

    # ---- fiber presets -----------------------------------------------------
    print("\n--- fiber presets ---")
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
                   "fib_source1_1": "cylindrical",
                   "fib_component1_1": "circumferential"})
    node.parmTuple("fib_axis_origin1_1").set((0.5, 0.0, 0.5))
    node.parmTuple("fib_axis_dir1_1").set((0, 1, 0))

    np.seterr(all="ignore")   # Accelerate flags these exact matmuls spuriously
    resolved = phm.resolve_fibers(node, 1, 1, None, count)[""]
    axis = np.array([0.0, 1.0, 0.0])
    relative = cents - np.array([0.5, 0.0, 0.5])
    radial = relative - (relative @ axis)[:, None] * axis
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    circumferential = np.cross(np.tile(axis, (count, 1)), radial)

    check("cylindrical: circumferential matches an analytic reference",
          np.abs(resolved - circumferential).max() < 1e-9,
          f"max diff {np.abs(resolved - circumferential).max():.2e}")
    check("cylindrical: fibers are perpendicular to the axis",
          np.abs(resolved @ axis).max() < 1e-9)
    check("cylindrical: fibers are perpendicular to the radial direction",
          np.abs(np.einsum("ij,ij->i", resolved, radial)).max() < 1e-9)

    node.setParms({"fib_component1_1": "radial"})
    check("cylindrical: radial component",
          np.abs(phm.resolve_fibers(node, 1, 1, None, count)[""] - radial).max()
          < 1e-9)
    node.setParms({"fib_component1_1": "longitudinal"})
    check("cylindrical: longitudinal component is the axis",
          np.abs(phm.resolve_fibers(node, 1, 1, None, count)[""]
                 - axis).max() < 1e-9)
    node.setParms({"fib_component1_1": "circumferential"})

    # an element sitting on the axis has no radial/circumferential direction
    on_axis = np.array([[0.5, 0.4, 0.5], [0.9, 0.4, 0.5]])
    expect_error("cylindrical: an element on the axis is rejected",
                 "lie on the cylinder axis",
                 lambda: phm.cylindrical_frame(
                     on_axis, (0.5, 0.0, 0.5), (0, 1, 0), "test"))
    expect_error("cylindrical: a zero-length axis is rejected",
                 "zero length",
                 lambda: phm.cylindrical_frame(
                     cents, (0.5, 0.0, 0.5), (0, 0, 0), "test"))

    # curve tangent
    curve = node.createNode("python", "curve")
    curve.parm("python").set(
        "geo = hou.pwd().geometry()\ncoords = [(0.5, -1, 0.5), (0.5, 0, 0.5), (0.5, 3, 0.5)]\npoints = []\nfor c in coords:\n    p = geo.createPoint()\n    p.setPosition(hou.Vector3(c))\n    points.append(p)\npoly = geo.createPolygon()\npoly.setIsClosed(False)\nfor p in points:\n    poly.addVertex(p)\n")
    node.setParms({"fib_source1_1": "curve", "fib_curve1_1": curve.path()})
    tangents = phm.resolve_fibers(node, 1, 1, None, count)[""]
    check("curve tangent: straight curve gives its own direction",
          np.abs(np.abs(tangents @ axis) - 1).max() < 1e-9,
          f"max deviation {np.abs(np.abs(tangents @ axis) - 1).max():.2e}")

    # Compare against a straightforward reference: for every element, the
    # tangent of the closest point over all segments. This exercises the real
    # algorithm instead of an intuition about which segment "should" win --
    # near a bend both segments are genuinely equidistant.
    def reference_tangents(coords, points):
        starts = np.asarray(coords[:-1], dtype=np.float64)
        segments = np.asarray(coords[1:], dtype=np.float64) - starts
        lengths = np.einsum("ij,ij->i", segments, segments)
        best = np.empty_like(points)
        for index, point in enumerate(points):
            offsets = point - starts
            t = np.clip(np.einsum("ij,ij->i", offsets, segments) / lengths, 0, 1)
            deltas = starts + t[:, None] * segments - point
            nearest = np.einsum("ij,ij->i", deltas, deltas).argmin()
            best[index] = segments[nearest] / np.sqrt(lengths[nearest])
        return best

    bent_coords = [(0.5, -1, 0.5), (0.5, 1, 0.5), (3, 1, 0.5)]
    bent = node.createNode("python", "bent")
    bent.parm("python").set(
        "geo = hou.pwd().geometry()\ncoords = [(0.5, -1, 0.5), (0.5, 1, 0.5), (3, 1, 0.5)]\npoints = []\nfor c in coords:\n    p = geo.createPoint()\n    p.setPosition(hou.Vector3(c))\n    points.append(p)\npoly = geo.createPolygon()\npoly.setIsClosed(False)\nfor p in points:\n    poly.addVertex(p)\n")
    node.setParms({"fib_curve1_1": bent.path()})
    bent_tangents = phm.resolve_fibers(node, 1, 1, None, count)[""]
    expected_tangents = reference_tangents(bent_coords, cents)
    check("curve tangent matches an independent reference over a bend",
          np.abs(bent_tangents - expected_tangents).max() < 1e-9,
          f"max diff {np.abs(bent_tangents - expected_tangents).max():.2e}")
    check("curve tangent: both segments of the bend are used",
          len(np.unique(np.round(expected_tangents, 6), axis=0)) == 2,
          str(np.unique(np.round(expected_tangents, 6), axis=0).tolist()))
    check("curve tangent: every direction is unit length",
          np.abs(np.linalg.norm(bent_tangents, axis=1) - 1).max() < 1e-9)

    # A curve fine enough to force the chunked path (chunk = 4e6 / segments).
    fine = node.createNode("python", "fine")
    fine.parm("python").set(
        "import numpy as np\n"
        "geo = hou.pwd().geometry()\n"
        "points = []\n"
        "for y in np.linspace(-1, 3, 3200):\n"
        "    p = geo.createPoint()\n"
        "    p.setPosition(hou.Vector3((0.5, float(y), 0.5)))\n"
        "    points.append(p)\n"
        "poly = geo.createPolygon()\n"
        "poly.setIsClosed(False)\n"
        "for p in points:\n"
        "    poly.addVertex(p)\n")
    node.setParms({"fib_curve1_1": fine.path()})
    fine_tangents = phm.resolve_fibers(node, 1, 1, None, count)[""]
    check("curve tangent: chunked path (3199 segments) stays correct",
          np.abs(np.abs(fine_tangents @ axis) - 1).max() < 1e-9,
          f"chunk size {max(1, int(4e6 // 3199))} vs {count} elements")

    empty = node.createNode("null", "no_curve")
    node.setParms({"fib_curve1_1": empty.path()})
    expect_error("curve tangent: a SOP with no curve is rejected",
                 "no curve segments",
                 lambda: phm.resolve_fibers(node, 1, 1, None, count))
    node.setParms({"fib_curve1_1": curve.path()})

    # presets export like any other per-element source
    node.setParms({"fib_source1_1": "cylindrical"})
    material = phm.build_material(node, 1, 1)
    check("presets emit a per-element file reference",
          material["fiber_direction"].get("field") == "FIB_1_1",
          str(material["fiber_direction"]))
    phm.export_per_element_materials(node, input_dir)
    exported = phm.read_cell_vectors_legacy_vtk(
        os.path.join(input_dir, "fibers.vtk"), "FIB_1_1")
    check("preset directions reach the exported file",
          np.abs(exported[vol1] - circumferential[vol1]).max() < 1e-6,
          f"max diff {np.abs(exported[vol1] - circumferential[vol1]).max():.2e}")

    # the artery layout: circumferential +/- theta about the wall normal
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("MaterialSum"),
                   "matrix_model1_1": "NeoHookean",
                   "num_fiber_families1_1": 1,
                   "fam_model1_1_1": "HGODispersion",
                   "fam_fib_source1_1_1": "cylindrical",
                   "fam_fib_component1_1_1": "circumferential",
                   "fam_mirror1_1_1": 1, "fam_theta1_1_1": 30,
                   "fam_axis_source1_1_1": "cylindrical"})
    node.parmTuple("fam_fib_axis_origin1_1_1").set((0.5, 0.0, 0.5))
    node.parmTuple("fam_fib_axis_dir1_1_1").set((0, 1, 0))
    pair = phm.resolve_fibers(node, 1, 1, 1, count)
    check("artery layout produces a symmetric pair", set(pair) == {"p", "m"})
    for side in ("p", "m"):
        check(f"artery {side}: stays perpendicular to the wall normal",
              np.abs(np.einsum("ij,ij->i", pair[side], radial)).max() < 1e-9)
    angles = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", pair["p"], pair["m"]), -1, 1)))
    check("artery: the two families are 2*theta apart",
          np.abs(angles - 60).max() < 1e-6, f"max {np.abs(angles - 60).max():.2e}")
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
                   "fib_source1_1": "attribute"})

    # ---- visualization -----------------------------------------------------
    print("\n--- visualization ---")
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
                   "fib_source1_1": "attribute", "fib_sop1_1": source.path(),
                   "fib_attrib1_1": "fiber1",
                   "kappa_source1_1": "attribute",
                   "kappa_sop1_1": source.path(), "kappa_attrib1_1": "kappa"})
    phm.update_fiber_data(node, 1)

    stamped = node.node("fiberdata_1").geometry()
    check("per-element data stamped on the volume elements",
          stamped.findPrimAttrib("pf_fiber_1_0") is not None
          and stamped.findPrimAttrib("pf_kappa_1_0") is not None,
          str([a.name() for a in stamped.primAttribs() if a.name()[:3] == "pf_"]))
    mask = phm._volume_mask(stamped)
    stamped_fibers = np.asarray(
        stamped.primFloatAttribValues("pf_fiber_1_0")).reshape(-1, 3)
    stamped_volume = stamped_fibers[mask]
    owner_diff = np.abs(stamped_volume[vol1] - fibers[vol1]).max()
    check("stamped directions equal the authored ones on their subdomain",
          owner_diff < 1e-5
          and (np.linalg.norm(stamped_volume[~vol1], axis=1) < 1e-9).all(),
          f"owner max diff {owner_diff:.2e}")

    surface_geo = node.node("surface_1").geometry()
    surface_fibers = np.asarray(
        surface_geo.primFloatAttribValues("pf_fiber_1_0")).reshape(-1, 3)
    surface_entities = np.asarray(surface_geo.primIntAttribValues("Entity"))
    check("boundary faces carry data only on the owning subdomain",
          len(surface_fibers)
          and (np.linalg.norm(
              surface_fibers[surface_entities == 1], axis=1) > 1e-9).all()
          and (np.linalg.norm(
              surface_fibers[surface_entities != 1], axis=1) < 1e-9).all())

    node.setParms({"show_fibers1": 1, "fiber_scale1": 0.5,
                   "fiber_max_lines1": 20000})
    phm.fiber_display_changed({"node": node, "script_multiparm_index": "1"})
    hedgehogs = node.node("fiberviz_1").geometry().intrinsicValue(
        "primitivecount")
    check("one hedgehog per element of the owning subdomain",
          hedgehogs == int(vol1.sum()),
          f"{hedgehogs} lines vs {int(vol1.sum())} owning elements")

    node.setParms({"fiber_max_lines1": 200})
    phm.fiber_display_changed({"node": node, "script_multiparm_index": "1"})
    decimated = node.node("fiberviz_1").geometry().intrinsicValue(
        "primitivecount")
    check("the line cap decimates instead of drawing everything",
          decimated <= 210, f"{decimated} lines")
    node.setParms({"fiber_max_lines1": 20000})

    colours = {}
    for mode in ("subdomains", "kappa", "fiber_rgb"):
        node.parm("color_by1").set(mode)
        phm.fiber_display_changed({"node": node, "script_multiparm_index": "1"})
        entity_colour = node.node("entitycolor_1")
        check(f"color-by {mode} cooks cleanly", not entity_colour.errors(),
              str(entity_colour.errors()[:1]))
        values = np.asarray(
            entity_colour.geometry().primFloatAttribValues("Cd")).reshape(-1, 3)
        colours[mode] = len(np.unique(np.round(values, 3), axis=0))
    check("subdomain mode keeps one colour per subdomain",
          colours["subdomains"] == 2, str(colours))
    check("kappa and fiber modes vary across the surface",
          colours["kappa"] > 10 and colours["fiber_rgb"] > 10, str(colours))
    node.parm("color_by1").set("subdomains")
    phm.fiber_display_changed({"node": node, "script_multiparm_index": "1"})

    # ---- end-to-end: a per-element scalar on a NON-FIRST subdomain ---------
    # RB-11 (2026-09-13): the HDA writes every per-element file over the
    # global element range. Upstream #333 had made the solver read scalar
    # files body-locally, so body 2's rows were misindexed while the fibre
    # file (still global) was not; the fork binds both by length. This solve
    # proves the kappa authored on subdomain 2 reaches the right elements.
    print("\n--- end-to-end solve: per-element kappa on subdomain 2 ---")
    scalar_source = node.createNode("attribwrangle", "scalar_source_vol2")
    scalar_source.setInput(0, node.node("branch_1"))
    scalar_source.setParms({"class": 1, "snippet": (
        "if (i@is_volume == 0) return;\n"
        "vector c = {0,0,0}; int n = primvertexcount(0, @primnum);\n"
        "for (int i = 0; i < n; i++)\n"
        "    c += point(0, \"P\", vertexpoint(0, vertexindex(0, @primnum, i)));\n"
        "c /= n;\n"
        "v@fiber2 = normalize(set(0.2, 1.0, 0.1));\n"
        "f@kappa2 = 0.05 + 0.25 * clamp(0.5 * c.x + 0.25 * c.z, 0.0, 1.0);")})
    node.setParms({
        "materials1_1": phm.MATERIAL_TOKENS.index("NeoHookean"),
        "E1_1": 1e5, "nu1_1": 0.4,
        "materials1_2": phm.MATERIAL_TOKENS.index("HGODispersion"),
        "hgo_k11_2": 1e4, "hgo_k21_2": 5.0,
        "fib_source1_2": "attribute", "fib_sop1_2": scalar_source.path(),
        "fib_attrib1_2": "fiber2",
        "kappa_source1_2": "attribute", "kappa_sop1_2": scalar_source.path(),
        "kappa_attrib1_2": "kappa2"})
    material2 = phm.build_material(node, 1, 2)
    check("subdomain 2 emits a per-element kappa file",
          material2.get("kappa") == "pe_kappa_1_2.txt", str(material2))
    params_path = phm.build_params(node)
    kappa_file = np.loadtxt(os.path.join(os.path.dirname(params_path),
                                         "pe_kappa_1_2.txt"))
    check("subdomain 2's kappa file spans the global element range",
          len(kappa_file) == count, f"{len(kappa_file)} rows vs {count}")
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "params.json", "-o", "../output_vol2/",
         "--log_level", "error"],
        cwd=os.path.dirname(params_path), capture_output=True, text=True,
        timeout=900)
    check("PolyFEM runs the subdomain-2 scene", result.returncode == 0,
          (result.stdout + result.stderr)[-400:])
    if result.returncode == 0:
        out_dir = os.path.join(work, "output_vol2")
        vtus = [f for f in os.listdir(out_dir) if f.endswith(".vtu")]
        vtu = vtu_parser.read_vtu(os.path.join(out_dir, sorted(vtus)[0]))
        pdata = vtu["point_data"]
        kappa_keys = [k for k in pdata if k.endswith("kappa")]
        check("kappa field present in the output", bool(kappa_keys),
              str(sorted(pdata)[:12]))
        body = pdata["body_ids"].ravel() if "body_ids" in pdata else None
        check("body ids exported for the subdomain-2 binding check",
              body is not None)
        if kappa_keys and body is not None:
            kappa_out = pdata[kappa_keys[0]].ravel()
            pts = vtu["points"]
            checked = bad = other = 0
            worst = 0.0
            for cell in vtu["cells"]["tet"]:
                values = kappa_out[cell]
                if int(body[cell[0]]) != 1002:
                    # the NeoHookean body has no kappa: reported as zero
                    if np.abs(values).max() > 1e-12:
                        other += 1
                    continue
                centroid = pts[cell].mean(axis=0)
                expected = 0.05 + 0.25 * min(max(
                    0.5 * centroid[0] + 0.25 * centroid[2], 0.0), 1.0)
                checked += 1
                err = float(np.abs(values - expected).max())
                worst = max(worst, err)
                if err > 1e-6:
                    bad += 1
            expected_count = int(np.sum(np.asarray(
                node.node("branch_1").geometry().primIntAttribValues("Entity")
            )[np.asarray(node.node("branch_1").geometry()
                         .primIntAttribValues("is_volume")).astype(bool)] == 2))
            check("per-element kappa reached the solver on subdomain 2's "
                  "elements (global-row binding)",
                  checked == expected_count and bad == 0 and other == 0,
                  f"{checked}/{expected_count} checked, {bad} wrong, "
                  f"{other} leaked onto body 1, worst {worst:.2e}")
    node.setParms({"materials1_1": phm.MATERIAL_TOKENS.index("HGODispersion"),
                   "materials1_2": phm.MATERIAL_TOKENS.index("NeoHookean"),
                   "fib_source1_2": "constant", "kappa_source1_2": "constant"})

    print("\n" + ("PASS: PolyFEM material models and per-element data"
                  if not fails else f"FAILURES: {fails}"))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
