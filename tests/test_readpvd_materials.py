"""readPVD ingestion + visualization of PolyFEM material fields.

PolyFEM namespaces material output with slashes ("MaterialSum/HGODispersion/
kappa"), which are illegal in Houdini attribute names -- uploading them
verbatim makes addAttrib raise and the whole frame fail to load. Covers the
sanitizer/display-name map (R0) and the fiber visualization built on it (R1).

Run: hython tests/test_readpvd_materials.py
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


fails = []


def check(label, condition, detail=""):
    print(("  PASS  " if condition else "  FAIL  ") + label
          + (f"  {detail}" if detail else ""))
    if not condition:
        fails.append(label)


def make_cube(path, x0, size=1.0, h=0.5):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(os.path.basename(path))
    box = gmsh.model.occ.addBox(x0, 0, 0, size, size, size)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(path)
    gmsh.finalize()


def tet_centroids(msh_path):
    import sys
    sys.path.insert(0, os.path.join(BASE, "src", "common"))
    import msh_parser
    mesh = msh_parser.read_msh(msh_path)
    return mesh["points"][mesh["cells"]["tet"]["corners"]].mean(axis=1)


ANCHOR = np.array([0.5, 0.5, -0.75])


def ensure_material_run():
    """Two-body run: MaterialSum(NeoHookean + HGODispersion with per-element
    fibers) and a plain NeoHookean body -> MultiModels, slash-named fields."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-materials")
    pvd = os.path.join(out_dir, "mats.pvd")
    if os.path.isfile(pvd):
        return out_dir, pvd
    os.makedirs(out_dir, exist_ok=True)
    cube1 = os.path.join(out_dir, "cube1.msh")
    cube2 = os.path.join(out_dir, "cube2.msh")
    make_cube(cube1, 0.0)
    make_cube(cube2, 2.0)

    fibers = np.concatenate([tet_centroids(cube1), tet_centroids(cube2)]) - ANCHOR
    fibers /= np.linalg.norm(fibers, axis=1, keepdims=True)
    with open(os.path.join(out_dir, "fibers.vtk"), "w") as f:
        f.write("# vtk DataFile Version 3.0\nfibers\nASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")
        f.write(f"CELL_DATA {len(fibers)}\nVECTORS FIB_DIR1 float\n")
        for v in fibers:
            f.write(f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")

    scene = {
        "geometry": [
            {"mesh": "cube1.msh", "volume_selection": 1,
             "surface_selection": [{"id": 10, "axis": "-z", "position": 0.001},
                                   {"id": 20, "axis": "+z", "position": 0.999}]},
            {"mesh": "cube2.msh", "volume_selection": 2,
             "surface_selection": [{"id": 10, "axis": "-z", "position": 0.001},
                                   {"id": 20, "axis": "+z", "position": 0.999}]},
        ],
        "materials": [
            {"id": 1, "type": "MaterialSum", "rho": 1000, "models": [
                {"type": "NeoHookean", "E": 1e5, "nu": 0.4},
                {"type": "HGODispersion", "k1": 1e4, "k2": 5.0, "kappa": 0.2,
                 "fiber_direction": {"type": "per_element_file",
                                     "path": "fibers.vtk"}}]},
            {"id": 2, "type": "NeoHookean", "E": 1e5, "nu": 0.4, "rho": 1000},
        ],
        "boundary_conditions": {"dirichlet_boundary": [
            {"id": 10, "value": [0, 0, 0]},
            {"id": 20, "value": [0, 0, "0.05 * t"]}]},
        # time-dependent so PolyFEM emits a .pvd sequence (readPVD's input)
        "time": {"t0": 0, "dt": 0.5, "time_steps": 2, "quasistatic": True},
        "output": {"log": {"level": "error"}, "paraview": {
            "file_name": "mats.pvd", "vismesh_rel_area": 1e10,
            "options": {"material": True, "body_ids": True}}},
    }
    scene_path = os.path.join(out_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", "scene.json", "-o", "."],
        cwd=out_dir, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert os.path.isfile(pvd), os.listdir(out_dir)
    return out_dir, pvd


def main():
    assert os.path.isfile(POLYFEM_BIN), f"missing {POLYFEM_BIN}"
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"))
    out_dir, pvd = ensure_material_run()

    node = hou.node("/obj").createNode("readPVD::1.0", "mat_viewer")
    phm = node.hdaModule()
    node.setParms({"PVD_file": pvd, "source_block": "Volume"})
    hou.setFrame(1)

    # ---- R0: the load must succeed and expose sanitized names --------------
    geo = node.node("output").geometry()   # raises if cook_frame failed
    names = {a.name() for a in geo.pointAttribs()}

    raw_fields = phm.read_vtu_field_info(
        os.path.join(out_dir, [f for f in os.listdir(out_dir)
                               if f.endswith(".vtu")][0]))
    raw_point = raw_fields.get("Volume", raw_fields).get("point_data", {})
    slashed = [n for n in raw_point if "/" in n]
    assert slashed, f"fixture has no slash-named fields: {sorted(raw_point)}"

    for raw in slashed:
        sanitized = phm._field_name(raw)
        assert "/" not in sanitized, sanitized
        assert sanitized in names, \
            f"{raw} -> {sanitized} missing from loaded attribs"
        assert phm._display_name(sanitized) == raw, \
            f"display name lost for {raw}"
    print(f"  R0: {len(slashed)} slash-named fields loaded and mapped back")

    # display-name map is also stored on the geometry
    stored = json.loads(geo.attribValue("readpvd_display_names"))
    for raw in slashed:
        assert stored.get(phm._field_name(raw)) == raw

    # menu lists them with their original names as labels
    menu = phm.color_field_menu({"node": node})
    tokens, labels = menu[0::2], menu[1::2]
    for raw in slashed:
        sanitized = phm._field_name(raw)
        assert sanitized in tokens, f"{sanitized} not offered for coloring"
        label = labels[tokens.index(sanitized)]
        assert raw in label, f"label {label!r} does not show {raw!r}"
    print("  R0: color menu offers them with original labels")

    # sanitizer is deterministic, legal, and collision-stable
    assert phm._field_name("MaterialSum/HGODispersion/kappa") == \
        "MaterialSum_HGODispersion_kappa"
    assert phm._field_name("9lives") == "_9lives"
    a = phm._field_name("a/b")
    b = phm._field_name("a_b")
    assert a != b and phm._display_name(a) == "a/b", (a, b)
    assert phm._field_name("a/b") == a, "sanitizer not stable across calls"
    print("  R0: sanitizer legal, stable, collision-safe")

    # a slash-named field is selectable and actually colors the mesh
    kappa = phm._field_name(
        next(n for n in slashed if n.endswith("kappa")))
    node.setParms({"color_attrib": kappa, "color_reduction": "auto"})
    node.node("output").cook(force=True)
    kap = np.asarray(node.node("output").geometry()
                     .pointFloatAttribValues(kappa))
    assert np.isclose(kap[kap != 0].max(), 0.2, atol=1e-6), kap.max()
    print(f"  R0: kappa field colors the mesh (max {kap.max():.3f})")

    # ---- R1: fiber families, hedgehogs, deformed directions ---------------
    print()
    node.parm("sync").pressButton() if node.parm("sync") else None
    phm.sync_available_options({"node": node})
    families = node.node("output").geometry().attribValue(
        "readpvd_fiber_families").split()
    check("fiber family assembled from the x/y/z triple", len(families) == 1,
          str(families))
    check("has_fiber_data flag set", node.evalParm("has_fiber_data") == 1)

    fiber_menu = phm.fiber_family_menu({"node": node})
    check("family menu offers All + the family",
          fiber_menu[0] == "all" and families[0] in fiber_menu, str(fiber_menu))

    geo = node.node("output").geometry()
    a0 = np.asarray(geo.pointFloatAttribValues(families[0])).reshape(-1, 3)
    check("assembled fiber vectors are unit length",
          np.abs(np.linalg.norm(a0[np.linalg.norm(a0, axis=1) > 1e-9], axis=1)
                 - 1).max() < 1e-5)

    # hedgehog geometry
    node.setParms({"show_fibers": 1, "fiber_frame": "reference",
                   "fiber_stride": 1, "smooth_field": 0})
    phm.autofiber({"node": node})
    node.node("output").cook(force=True)
    fiber_only = node.node("fiber_only")
    lines = fiber_only.geometry().intrinsicValue("primitivecount")
    # Only the body using the fiber model has fibers; PolyFEM reports zero for
    # the plain-NeoHookean body's nodes, which are correctly not drawn.
    result_geo = node.node("OUT_result").geometry()
    rep_flags = np.asarray(result_geo.pointFloatAttribValues("coincident_rep"))
    fiber_vectors = np.asarray(
        result_geo.pointFloatAttribValues(families[0])).reshape(-1, 3)
    drawn = int(np.sum((rep_flags > 0.5)
                       & (np.linalg.norm(fiber_vectors, axis=1) > 1e-12)))
    check("one reference line per fiber-carrying physical node",
          lines == drawn and drawn > 0, f"{lines} lines vs {drawn} nodes")

    node.setParms({"fiber_frame": "both"})
    node.node("output").cook(force=True)
    check("'both' draws reference and deformed",
          fiber_only.geometry().intrinsicValue("primitivecount") == 2 * lines,
          str(fiber_only.geometry().intrinsicValue("primitivecount")))

    # T-R3: the deformed direction must be normalize(F a0), with PolyFEM's
    # column-major flattening undone. This is the check that catches a
    # row/column mix-up in the wrangle.
    node.setParms({"fiber_frame": "deformed", "fiber_stride": 7})
    node.node("output").cook(force=True)
    lines_geo = fiber_only.geometry()
    src = node.node("OUT_result").geometry()
    positions = np.asarray(src.pointFloatAttribValues("P")).reshape(-1, 3)
    rep = np.asarray(src.pointFloatAttribValues("coincident_rep"))
    fibers = np.asarray(src.pointFloatAttribValues(families[0])).reshape(-1, 3)
    cols = [np.asarray(src.pointFloatAttribValues(f"F_{i}")).reshape(-1, 3)
            for i in (1, 2, 3)]

    checked = wrong = wrong_metrics = 0
    metric_failures = {}
    line_pts = np.asarray(lines_geo.pointFloatAttribValues("P")).reshape(-1, 3)
    for prim in lines_geo.prims():
        a, b = [v.point().number() for v in prim.vertices()]
        mid = 0.5 * (line_pts[a] + line_pts[b])
        direction = line_pts[b] - line_pts[a]
        if np.linalg.norm(direction) < 1e-12:
            continue
        direction /= np.linalg.norm(direction)
        # match the drawn line back to its node by position
        node_index = int(np.argmin(np.linalg.norm(positions - mid, axis=1)))
        if rep[node_index] < 0.5:
            continue
        # F from the exported COLUMNS
        F = np.stack([cols[0][node_index], cols[1][node_index],
                      cols[2][node_index]], axis=1)
        pushed = F @ fibers[node_index]
        stretch = np.linalg.norm(pushed)
        if stretch < 1e-12:
            continue
        expected = pushed / stretch
        checked += 1
        if abs(float(np.dot(direction, expected))) < 1 - 1e-4:
            wrong += 1
        current = np.asarray(
            prim.attribValue("fiber_current_direction"), dtype=np.float64)
        reference = np.asarray(
            prim.attribValue("fiber_reference_direction"), dtype=np.float64)
        angle = np.degrees(np.arccos(np.clip(
            abs(float(np.dot(reference, current))), 0.0, 1.0)))
        J = max(abs(float(np.linalg.det(F))), 1e-12)
        isochoric_stretch = stretch / J ** (1.0 / 3.0)
        metrics = {
            "current": abs(float(np.dot(current, expected))) > 1 - 1e-4,
            "stretch": np.isclose(
                prim.attribValue("fiber_stretch"), stretch,
                rtol=1e-4, atol=1e-5),
            "I4": np.isclose(
                prim.attribValue("fiber_I4"), stretch ** 2,
                rtol=1e-4, atol=1e-5),
            "isochoric_stretch": np.isclose(
                prim.attribValue("fiber_isochoric_stretch"),
                isochoric_stretch, rtol=1e-4, atol=1e-5),
            "I4bar": np.isclose(
                prim.attribValue("fiber_I4bar"),
                isochoric_stretch ** 2, rtol=1e-4, atol=1e-5),
            "angle": np.isclose(
                prim.attribValue("fiber_direction_change_degrees"),
                angle, rtol=1e-4, atol=5e-2),
            "frame": prim.attribValue("fiber_drawn_frame")
            == "current_normalized_Fa0",
        }
        if not all(metrics.values()):
            wrong_metrics += 1
            for name, passed in metrics.items():
                if not passed:
                    metric_failures[name] = metric_failures.get(name, 0) + 1
    check("deformed direction is normalize(F a0) (column-major undone)",
          checked > 0 and wrong == 0, f"{checked} checked, {wrong} wrong")
    check("fiber lines expose PolyFEM I4/I4bar kinematics",
          checked > 0 and wrong_metrics == 0,
          f"{checked} checked, {wrong_metrics} wrong {metric_failures}")
    color_modes = node.parm(
        "fiber_color_mode").parmTemplate().menuItems()
    check("fiber coloring exposes stretch and direction change",
          "stretch" in color_modes and "angle" in color_modes,
          str(color_modes))

    # T-R5: sign-coherent nodal recovery
    node.setParms({"smooth_field": 1, "fiber_stride": 1,
                   "fiber_frame": "reference"})
    node.node("output").cook(force=True)
    recovered = np.asarray(
        node.node("fiber_recovery").geometry()
        .pointFloatAttribValues(families[0])).reshape(-1, 3)
    live = np.linalg.norm(recovered, axis=1) > 1e-9
    check("smoothed fibers stay unit length (no sign cancellation)",
          np.abs(np.linalg.norm(recovered[live], axis=1) - 1).max() < 1e-4,
          f"min norm {np.linalg.norm(recovered[live], axis=1).min():.3f}")

    # explicit +a0/-a0 cancellation case
    flipped = fibers.copy()
    ids = np.asarray(src.pointFloatAttribValues("coincident_id")).astype(int)
    seen = {}
    for index, group in enumerate(ids):
        if group in seen:
            flipped[index] = -flipped[index]
        else:
            seen[group] = index
    naive = np.zeros((ids.max() + 1, 3))
    for index, group in enumerate(ids):
        naive[group] += flipped[index]
    cancelled = int(np.sum(np.linalg.norm(naive, axis=1) < 1e-6))
    check("the +/-a0 case really would cancel without sign alignment",
          cancelled > 0, f"{cancelled} groups cancel under naive averaging")

    node.setParms({"show_fibers": 0, "smooth_field": 0})
    print("\n" + ("PASS: readPVD material fields and fiber visualization "
                  "(R0 + R1)" if not fails else f"FAILURES: {fails}"))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
