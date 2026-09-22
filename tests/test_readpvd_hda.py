"""Headless test of the readPVD 1.0 HDA against real PolyFEM output.

Run: hython tests/test_readpvd_hda.py
"""

import json
import os
import subprocess
import time

import numpy as np

import hou

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
POLYFEM_BIN = os.path.join(ROOT, "polyfem", "build", "PolyFEM_bin")
SMOKE = os.path.join(ROOT, "smoke-out", "quasistatic-semi")


def cached_run_is_current(pvd):
    """A derived run of the smoke scene is only comparable with the full run
    in SMOKE if the same solver produced both: regenerate it when it is
    older than the full run's last step or than the binary (RB-12: the
    July minimal run was being compared with a September full run, 4e-4 of
    displacement apart, and failed the PK2 comparison for that reason)."""
    if not os.path.isfile(pvd):
        return False
    age = os.path.getmtime(pvd)
    newer = [f for f in (os.path.join(SMOKE, "step_4.vtu"), POLYFEM_BIN)
             if os.path.isfile(f) and os.path.getmtime(f) > age]
    if newer:
        print(f"regenerating {os.path.basename(pvd)}: older than "
              + ", ".join(os.path.basename(f) for f in newer))
    return not newer


def ensure_surface_run():
    """Produce output with a Surface block (contact forces) if missing."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-surface")
    pvd = os.path.join(out_dir, "surface.pvd")
    if cached_run_is_current(pvd):
        return pvd
    os.makedirs(out_dir, exist_ok=True)
    scene_dir = os.path.join(ROOT, "polyfem", "scenes", "semi-implicit")
    with open(os.path.join(scene_dir, "quasistatic-semi.json")) as f:
        scene = json.load(f)
    scene["materials"]["E"] = 1e6
    scene["output"]["paraview"] = {
        "file_name": "surface.pvd", "surface": True,
        "options": {"contact_forces": True, "friction_forces": True}}
    scene_path = os.path.join(out_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", scene_path, "-o", out_dir, "--log_level", "error"],
        cwd=scene_dir, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    assert os.path.isfile(pvd)
    return pvd


def ensure_minimal_run():
    """Re-run the smoke scene with the fields whitelist the PolyFEM HDA
    emits in minimal mode (solution + F + Cauchy stress only); everything
    else is derived by readPVD on load."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-minimal")
    pvd = os.path.join(out_dir, "minimal.pvd")
    if cached_run_is_current(pvd):
        return pvd
    os.makedirs(out_dir, exist_ok=True)
    scene_dir = os.path.join(ROOT, "polyfem", "scenes", "semi-implicit")
    with open(os.path.join(scene_dir, "quasistatic-semi.json")) as f:
        scene = json.load(f)
    scene["output"]["paraview"] = {
        "file_name": "minimal.pvd",
        "fields": ["solution", "F", "cauchy_stess"],
        "options": {"scalar_values": False, "tensor_values": True}}
    scene_path = os.path.join(out_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", scene_path, "-o", out_dir, "--log_level", "error"],
        cwd=scene_dir, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    assert os.path.isfile(pvd)
    return pvd


def ensure_hdf_run():
    """Produce the same smoke scene in PolyFEM's VTK-HDF format."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-hdf")
    pvd = os.path.join(out_dir, "hdf.pvd")
    if cached_run_is_current(pvd) and os.path.isfile(
            os.path.join(out_dir, "step_4.hdf")):
        return pvd
    os.makedirs(out_dir, exist_ok=True)
    scene_dir = os.path.join(ROOT, "polyfem", "scenes", "semi-implicit")
    with open(os.path.join(scene_dir, "quasistatic-semi.json")) as f:
        scene = json.load(f)
    scene["output"]["paraview"] = {
        "file_name": "hdf.pvd", "options": {"use_hdf5": True}}
    scene_path = os.path.join(out_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f)
    result = subprocess.run(
        [POLYFEM_BIN, "-j", scene_path, "-o", out_dir, "--log_level", "error"],
        cwd=scene_dir, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    assert os.path.isfile(pvd)
    assert os.path.isfile(os.path.join(out_dir, "step_4.hdf"))
    return pvd


def ensure_two_body_run():
    """Create a tiny PVD with two bodies (body_ids 0 and 1) for show/hide."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-two-body")
    os.makedirs(out_dir, exist_ok=True)
    pvd = os.path.join(out_dir, "two_body.pvd")
    points = ("0 0 0  1 0 0  0 1 0  0 0 1  1 1 1  "        # body 0 (pts 0-4)
              "3 0 0  4 0 0  3 1 0  3 0 1  4 1 1")          # body 1 (pts 5-9)
    text = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="10" NumberOfCells="4">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
{points}</DataArray></Points>
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii">0 1 2 3  1 2 3 4  5 6 7 8  6 7 8 9</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">4 8 12 16</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">10 10 10 10</DataArray>
</Cells>
<PointData>
<DataArray type="Float64" Name="solution" NumberOfComponents="3"
format="ascii">{"0 0 0  " * 10}</DataArray>
<DataArray type="Float64" Name="body_ids" format="ascii">0 0 0 0 0 1 1 1 1 1</DataArray>
<DataArray type="Float64" Name="von_mises" format="ascii">10 10 10 10 10 20 20 20 20 20</DataArray>
</PointData></Piece></UnstructuredGrid></VTKFile>"""
    with open(os.path.join(out_dir, "f0.vtu"), "w") as file:
        file.write(text)
    with open(pvd, "w") as file:
        file.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection><DataSet timestep="0" file="f0.vtu"/></Collection></VTKFile>""")
    return pvd


def ensure_cell_data_run():
    """Create a tiny two-frame PVD with scalar/vector CellData."""
    out_dir = os.path.join(ROOT, "smoke-out", "readpvd-cell-data")
    os.makedirs(out_dir, exist_ok=True)
    pvd = os.path.join(out_dir, "cell_data.pvd")
    points = "0 0 0  1 0 0  0 1 0  0 0 1  1 1 1"
    connectivity = "0 1 2 3  1 2 3 4"
    for frame in range(2):
        intermittent = (
            '<DataArray type="Float64" Name="intermittent" '
            'format="ascii">2 4</DataArray>' if frame == 0 else "")
        text = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="5" NumberOfCells="2">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
{points}</DataArray></Points>
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii">{connectivity}</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">4 8</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">10 10</DataArray>
</Cells>
<PointData><DataArray type="Float64" Name="solution"
NumberOfComponents="3" format="ascii">0 0 0  0 0 0  0 0 0  0 0 0  0 0 0
</DataArray></PointData>
<CellData>
<DataArray type="Float64" Name="cell_scalar" format="ascii">{1+frame} {3+frame}</DataArray>
<DataArray type="Float64" Name="cell_vector" NumberOfComponents="3"
format="ascii">1 2 3  4 5 6</DataArray>{intermittent}
</CellData>
</Piece></UnstructuredGrid></VTKFile>"""
        with open(os.path.join(out_dir, f"frame_{frame}.vtu"), "w") as file:
            file.write(text)
    with open(pvd, "w") as file:
        file.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
<DataSet timestep="0" file="frame_0.vtu"/>
<DataSet timestep="1" file="frame_1.vtu"/>
</Collection></VTKFile>""")
    return pvd


def count_types(geo):
    from collections import Counter
    return Counter(p.intrinsicValue("typename") for p in geo.prims())


def pf(geo, name):
    return np.frombuffer(geo.pointFloatAttribValuesAsString(name),
                         dtype=np.float32)


def assert_close(actual, expected, label, rtol=2e-4, atol=2e-5):
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"{label}: max absolute error "
        f"{np.nanmax(np.abs(actual - expected)):.6g}")


def main():
    hou.hda.installFile(os.path.join(BASE, "object_readPVD.1.0.hdanc"))
    node = hou.node("/obj").createNode("readPVD::1.0", "viewer")
    out = node.node("output")
    result = node.node("OUT_result")
    topo = node.node("topo_build")

    # A new node gets Houdini's "Infra-Red" ramp preset as the default color
    # ramp (set by the OnCreated handler): blue -> cyan -> green -> yellow ->
    # red, with no black or white ends.
    default_ramp = node.parm("color_ramp").evalAsRamp()
    assert len(default_ramp.keys()) == 5, "default ramp is not the infra-red preset"
    low, high = default_ramp.lookup(0.0), default_ramp.lookup(1.0)
    assert low[2] > 0.8 and low[0] < 0.5         # blue at the bottom
    assert high[0] > 0.8 and high[1] < 0.2 and high[2] < 0.2   # red at the top
    assert max(low) > 0.5 and min(high) < 0.5    # neither end is black or white

    # The inherited object folders (Transform/Render/Misc) are hidden from the
    # parameter list; the custom Read PVD folder stays visible.
    folders = {pt.label(): pt.isHidden()
               for pt in node.parmTemplateGroup().entries()
               if pt.type() == hou.parmTemplateType.Folder}
    assert folders.get("Transform") and folders.get("Render") \
        and folders.get("Misc"), "object folders should be hidden"
    assert not folders["Read PVD 1.0 (PolyFEM results)"]
    assert node.evalParm("tx") == 0  # transform parms still exist and work

    pvd = os.path.join(SMOKE, "quasistatic-semi.pvd")
    node.setParms({"PVD_file": pvd})
    node.hdaModule().sync_available_options({"node": node})

    # Availability scope defaults to current-frame so dropping a file does not
    # scan the whole sequence; the all-frames scopes are labelled as slow.
    assert node.parm("field_time_scope").evalAsString() == "current"
    scope_labels = node.parm("field_time_scope").menuLabels()
    assert any("slow" in label.lower() for label in scope_labels[1:])
    # field info is free when the frame is already parsed (reuses the LRU,
    # no second file read); exercised via the HDA's embedded parser
    phm = node.hdaModule()
    step0 = os.path.join(SMOKE, "step_0.vtu")
    phm.clear_caches()
    phm.read_vtu_cached(step0)
    info = phm.read_vtu_field_info(step0)
    assert "von_mises" in info["point_data"]

    # PolyFEM now defaults to compressed VTK-HDF output. The asset carries a
    # compatible h5py wheel, so a clean Houdini 22 installation can load it
    # without a separate package install. Compare the parsed contract against
    # the same XML smoke frame, then exercise a full HDA cook from the HDF PVD.
    hdf_pvd = ensure_hdf_run()
    xml_blocks = phm.load_frame(pvd, 4)
    hdf_blocks = phm.load_frame(hdf_pvd, 4)
    assert set(hdf_blocks) == set(xml_blocks)
    for block_name in xml_blocks:
        xml_mesh, hdf_mesh = xml_blocks[block_name], hdf_blocks[block_name]
        assert np.allclose(hdf_mesh["points"], xml_mesh["points"])
        assert set(hdf_mesh["cells"]) == set(xml_mesh["cells"])
        for family in xml_mesh["cells"]:
            assert np.array_equal(
                hdf_mesh["cells"][family], xml_mesh["cells"][family])
        assert set(hdf_mesh["point_data"]) == set(xml_mesh["point_data"])
        assert set(hdf_mesh["cell_data"]) == set(xml_mesh["cell_data"])
    hdf_info = phm.frame_field_info(hdf_pvd, 4)
    assert hdf_info == phm.frame_field_info(pvd, 4)
    hdf_node = hou.node("/obj").createNode("readPVD::1.0", "viewer_hdf")
    hdf_node.setParms({"PVD_file": hdf_pvd})
    hdf_node.hdaModule().sync_available_options({"node": hdf_node})
    hou.setFrame(4)
    hdf_node.node("output").cook(force=True)
    assert (hdf_node.node("output").geometry().intrinsicValue("pointcount")
            == 1540)
    hdf_node.destroy()
    print("PASS: embedded HDF5 reader matches VTU and cooks the HDA")

    # Menus expose only data that exists in the active PVD frame and block.
    assert node.parm("source_block").menuItems() == ("Volume",)
    volume_fields = set(node.parm("color_attrib").menuItems())
    assert "von_mises" in volume_fields and "cauchy_mat" in volume_fields
    assert "contact_forces" not in volume_fields
    assert set(node.parm("glyph_tensor").menuItems()) == {
        "right_cauchy_green", "left_cauchy_green", "green_lagrange",
        "pk2", "cauchy"}
    node.setParms({"derived": 0, "add_glyphs": 1})
    node.hdaModule().sync_available_options({"node": node})
    assert "cauchy_mat" not in node.parm("color_attrib").menuItems()
    assert node.parm("glyph_tensor").menuItems() == ("",)
    assert node.evalParm("add_glyphs") == 0
    node.setParms({"derived": 1})
    node.hdaModule().sync_available_options({"node": node})

    hou.setFrame(0)
    out.cook(force=True)
    geo0 = out.geometry()
    n_pts = geo0.intrinsicValue("pointcount")
    assert n_pts == 1540, n_pts
    # boundary-surface display by default: polys only, no tets
    counts0 = count_types(geo0)
    assert counts0.get("Tetrahedron", 0) == 0 and counts0.get("Poly", 0) > 0, counts0

    topo_cooks_before = topo.cookCount()

    hou.setFrame(4)
    t0 = time.time()
    out.cook(force=False)
    frame_cook = time.time() - t0
    geo4 = out.geometry()

    # topology stage must NOT have recooked on the frame change
    assert topo.cookCount() == topo_cooks_before, "topology cache missed"

    # The explicit Cache SOP stores imported fields plus stable derived
    # mechanics, not the finished display. Recoloring or clipping previously
    # cached frames must reuse both the PVD upload and derived calculation.
    cache_node = node.node("cache1")
    frame_node = node.node("frame_data")
    derived_node = node.node("derived")
    assert cache_node.inputs()[0] == node.node("derived_switch")
    assert node.node("deform").inputs()[0] == cache_node
    node.setParms({"cache": 1})
    node.hdaModule().toggle_cache({"node": node})
    node.hdaModule().clear_cache({"node": node})
    for cached_frame in (0, 4, 0):
        hou.setFrame(cached_frame)
        out.cook(force=False)
    cached_cooks = (frame_node.cookCount(), derived_node.cookCount())

    # Exercise the actual UI callback path for a field change, then revisit
    # both cached frames. Neither stable stage may cook again.
    old_color_field = node.evalParm("color_attrib")
    node.parm("color_attrib").set("J")
    node.hdaModule().color_selection_changed({"node": node})
    for cached_frame in (0, 4, 0):
        hou.setFrame(cached_frame)
        out.cook(force=False)
    assert (frame_node.cookCount(), derived_node.cookCount()) == cached_cooks, (
        "changing the color field invalidated calculated frame data")

    # Clipping is presentation-only and must have the same behavior in both
    # directions (the regression reported by the production fiber scene).
    for clip_mode in (1, 0):
        node.parm("clip_mode").set(clip_mode)
        for cached_frame in (4, 0):
            hou.setFrame(cached_frame)
            out.cook(force=False)
        assert (frame_node.cookCount(),
                derived_node.cookCount()) == cached_cooks, (
            "toggling clipping invalidated calculated frame data")

    # Range/ramp edits are downstream as well.
    old_color_max = node.evalParm("color_max")
    node.parm("color_max").set(old_color_max + 0.123)
    for cached_frame in (4, 0):
        hou.setFrame(cached_frame)
        out.cook(force=False)
    assert (frame_node.cookCount(), derived_node.cookCount()) == cached_cooks, (
        "editing the color range invalidated calculated frame data")

    node.parm("color_attrib").set(old_color_field)
    node.hdaModule().color_selection_changed({"node": node})
    node.parm("color_max").set(old_color_max)
    node.setParms({"cache": 0})
    node.hdaModule().toggle_cache({"node": node})
    hou.setFrame(4)
    out.cook(force=False)

    # Fields: generic forwarding, typo alias, and expanded mechanics values.
    derived_names = (
        "solution_mag", "F_mat", "J",
        "right_cauchy_green", "right_cauchy_green_eigenvalues",
        "left_cauchy_green", "left_cauchy_green_eigenvalues",
        "right_stretch", "left_stretch",
        "principal_stretches", "green_lagrange_strain",
        "green_lagrange_eigenvalues", "almansi_strain",
        "almansi_strain_eigenvalues", "hencky_strain",
        "hencky_strain_eigenvalues", "infinitesimal_strain",
        "infinitesimal_strain_eigenvalues", "cauchy_mat",
        "cauchy_eigenvalues", "cauchy_trace", "hydrostatic_stress",
        "deviatoric_stress", "stress_J2", "stress_J3",
        "von_mises_derived", "max_shear_stress", "stress_triaxiality",
        "pk1", "pk2", "pk2_eigenvalues")
    for name in ("solution", "von_mises", "cauchy_stress_1", "rest",
                 *derived_names):
        assert geo4.findPointAttrib(name) is not None, f"missing {name}"
    # deformation applied: P != rest and max |solution| matches the push
    P = pf(geo4, "P").reshape(-1, 3)
    rest = pf(geo4, "rest").reshape(-1, 3)
    sol = pf(geo4, "solution").reshape(-1, 3)
    assert np.allclose(P, rest + sol, atol=1e-5)
    assert abs(sol[:, 2].min() + 0.25) < 0.02, sol[:, 2].min()

    # Verify the derived continuum-mechanics quantities, not only their names.
    # PolyFEM flattens tensors column-major: X_i arrays are tensor COLUMNS.
    F = np.stack((pf(geo4, "F_1").reshape(-1, 3),
                  pf(geo4, "F_2").reshape(-1, 3),
                  pf(geo4, "F_3").reshape(-1, 3)), axis=2)
    identity = np.eye(3)
    right_cauchy_green = np.einsum("nji,njk->nik", F, F)
    left_cauchy_green = np.einsum("nij,nkj->nik", F, F)
    assert_close(pf(geo4, "J"), np.linalg.det(F), "deformation volume ratio J")
    # Right/Left Cauchy-Green deformation tensors are exported directly.
    assert_close(pf(geo4, "right_cauchy_green").reshape(-1, 3, 3),
                 right_cauchy_green, "Right Cauchy-Green tensor")
    assert_close(pf(geo4, "left_cauchy_green").reshape(-1, 3, 3),
                 left_cauchy_green, "Left Cauchy-Green tensor")
    # Stretch tensors satisfy U^2 = C and V^2 = B (polar decomposition).
    right_stretch = pf(geo4, "right_stretch").reshape(-1, 3, 3)
    left_stretch = pf(geo4, "left_stretch").reshape(-1, 3, 3)
    assert_close(np.matmul(right_stretch, right_stretch), right_cauchy_green,
                 "right stretch U (U^2 = C)", rtol=1e-4)
    assert_close(np.matmul(left_stretch, left_stretch), left_cauchy_green,
                 "left stretch V (V^2 = B)", rtol=1e-4)
    assert_close(
        pf(geo4, "green_lagrange_strain").reshape(-1, 3, 3),
        0.5 * (right_cauchy_green - identity), "Green-Lagrange strain")
    almansi_ref = 0.5 * (identity - np.linalg.pinv(left_cauchy_green))
    almansi_ref[np.abs(np.linalg.det(F)) <= 1e-12] = 0.0
    assert_close(pf(geo4, "almansi_strain").reshape(-1, 3, 3), almansi_ref,
                 "Almansi strain", rtol=1e-4)
    # Hencky (logarithmic) strain = matrix log of the right stretch tensor,
    # zeroed where a stretch vanishes (J = 0) to match the guarded HDA path.
    hencky_vals, hencky_vecs = np.linalg.eigh(right_cauchy_green)
    hencky = np.einsum("nij,nj,nkj->nik", hencky_vecs,
                       np.log(np.clip(np.sqrt(np.clip(hencky_vals, 0, None)),
                                      1e-20, None)),
                       hencky_vecs)
    hencky[np.abs(np.linalg.det(F)) <= 1e-12] = 0.0
    assert_close(pf(geo4, "hencky_strain").reshape(-1, 3, 3), hencky,
                 "logarithmic (Hencky) strain", rtol=1e-4)
    expected_stretches = np.sqrt(
        np.clip(np.linalg.eigvalsh(right_cauchy_green)[:, ::-1], 0, None))
    assert_close(pf(geo4, "principal_stretches").reshape(-1, 3),
                 expected_stretches, "principal stretches")

    cauchy = np.stack((pf(geo4, "cauchy_stress_1").reshape(-1, 3),
                       pf(geo4, "cauchy_stress_2").reshape(-1, 3),
                       pf(geo4, "cauchy_stress_3").reshape(-1, 3)), axis=2)
    stress_trace = np.trace(cauchy, axis1=1, axis2=2)
    deviatoric = cauchy - stress_trace[:, None, None] * identity / 3.0
    stress_j2 = 0.5 * np.sum(deviatoric * deviatoric, axis=(1, 2))
    assert_close(pf(geo4, "cauchy_trace"), stress_trace, "Cauchy stress trace")
    assert_close(pf(geo4, "hydrostatic_stress"), stress_trace / 3.0,
                 "hydrostatic stress")
    assert_close(pf(geo4, "stress_J2"), stress_j2,
                 "second deviatoric stress invariant J2", rtol=3e-4)
    assert_close(pf(geo4, "von_mises_derived"), np.sqrt(3.0 * stress_j2),
                 "derived von Mises stress", rtol=3e-4)
    for values_name in ("green_lagrange_eigenvalues",
                        "infinitesimal_strain_eigenvalues",
                        "cauchy_eigenvalues", "pk2_eigenvalues"):
        values = pf(geo4, values_name).reshape(-1, 3)
        assert np.all(values[:, :-1] >= values[:, 1:] - 1e-4), (
            f"{values_name} is not sorted maximum to minimum")

    # Eigenvectors used by the glyphs must be true eigenvectors, stored as the
    # matrix COLUMNS, paired with the descending eigenvalue in the same column
    # (guards the sorted_eigen row/column convention the glyph directions rely
    # on). Verify M @ v_c == lambda_c v_c and orthonormal columns.
    for mat_name, val_name, vec_name in (
            ("right_cauchy_green", "right_cauchy_green_eigenvalues",
             "right_cauchy_green_eigenvectors"),
            ("left_cauchy_green", "left_cauchy_green_eigenvalues",
             "left_cauchy_green_eigenvectors"),
            ("green_lagrange_strain", "green_lagrange_eigenvalues",
             "green_lagrange_eigenvectors"),
            ("pk2", "pk2_eigenvalues", "pk2_eigenvectors"),
            ("cauchy_mat", "cauchy_eigenvalues", "cauchy_eigenvectors")):
        tens = pf(geo4, mat_name).reshape(-1, 3, 3)
        tens = 0.5 * (tens + np.transpose(tens, (0, 2, 1)))    # symmetric part
        evals = pf(geo4, val_name).reshape(-1, 3)
        evecs = pf(geo4, vec_name).reshape(-1, 3, 3)           # column c = v_c
        good = np.abs(np.linalg.det(tens)) > 1e-9              # skip J~0 points
        tg, lg, vg = tens[good], evals[good], evecs[good]
        span = max(np.abs(lg).max(), 1e-12)
        for c in range(3):
            vc = vg[:, :, c]
            residual = np.einsum("nij,nj->ni", tg, vc) - lg[:, c][:, None] * vc
            assert np.abs(residual).max() / span < 1e-4, (
                f"{vec_name} column {c} is not the matching eigenvector")
        gram = np.einsum("nij,nik->njk", vg, vg)               # columns V^T V
        assert np.abs(gram - np.eye(3)).max() < 1e-3, (
            f"{vec_name} columns are not orthonormal")

    # ---- minimal-fields equivalence -------------------------------------
    # The PolyFEM HDA's minimal output mode stops writing von Mises and the
    # PK stresses; readPVD derives them from F + Cauchy. PolyFEM's own
    # exported arrays are the ground truth the derivations must reproduce.
    vm_file = pf(geo4, "von_mises")
    assert_close(pf(geo4, "von_mises_derived"), vm_file,
                 "derived von Mises vs PolyFEM von_mises",
                 rtol=2e-3, atol=1e-5 * np.abs(vm_file).max())
    pk1_file = np.stack([pf(geo4, f"pk1_stress_{i}").reshape(-1, 3)
                         for i in (1, 2, 3)], axis=2)
    assert_close(pf(geo4, "pk1").reshape(-1, 3, 3), pk1_file,
                 "derived PK1 vs PolyFEM pk1_stess",
                 rtol=2e-3, atol=1e-5 * np.abs(pk1_file).max())
    pk2_file = np.stack([pf(geo4, f"pk2_stress_{i}").reshape(-1, 3)
                         for i in (1, 2, 3)], axis=2)
    assert_close(pf(geo4, "pk2").reshape(-1, 3, 3), pk2_file,
                 "derived PK2 vs PolyFEM pk2_stess",
                 rtol=2e-3, atol=1e-5 * np.abs(pk2_file).max())

    # End-to-end: a run exporting ONLY the whitelist must unlock the whole
    # derived set with values matching the full run's PolyFEM arrays.
    minimal_pvd = ensure_minimal_run()
    minimal_step4 = os.path.join(os.path.dirname(minimal_pvd), "step_4.vtu")
    minfo = node.hdaModule().read_vtu_field_info(minimal_step4)
    minimal_names = set(minfo["point_data"])
    assert not any(n.startswith(("von_mises", "pk1", "pk2"))
                   for n in minimal_names), minimal_names
    mnode = hou.node("/obj").createNode("readPVD::1.0", "viewer_minimal")
    mnode.setParms({"PVD_file": minimal_pvd})
    mnode.hdaModule().sync_available_options({"node": mnode})
    minimal_menu = set(mnode.parm("color_attrib").menuItems())
    assert set(derived_names) <= minimal_menu, (
        set(derived_names) - minimal_menu)
    hou.setFrame(4)
    mout = mnode.node("output")
    mout.cook(force=True)
    mgeo = mout.geometry()
    assert_close(pf(mgeo, "von_mises_derived"), vm_file,
                 "minimal-run derived von Mises vs full-run PolyFEM",
                 rtol=2e-3, atol=1e-5 * np.abs(vm_file).max())
    assert_close(pf(mgeo, "pk2").reshape(-1, 3, 3), pk2_file,
                 "minimal-run derived PK2 vs full-run PolyFEM",
                 rtol=2e-3, atol=1e-5 * np.abs(pk2_file).max())
    full_size = os.path.getsize(os.path.join(SMOKE, "step_4.vtu"))
    minimal_size = os.path.getsize(minimal_step4)
    assert minimal_size < 0.5 * full_size, (minimal_size, full_size)
    print(f"PASS: minimal-fields equivalence "
          f"(vtu {full_size//1024} KB -> {minimal_size//1024} KB)")
    mnode.destroy()

    # Scalar/vector/tensor reductions drive one authoritative color value.
    node.setParms({"color_attrib": "cauchy_mat", "color_reduction": "trace"})
    result.cook(force=True)
    assert_close(pf(result.geometry(), "for_color"), stress_trace,
                 "tensor trace color reduction")
    node.setParms({"color_reduction": "determinant"})
    result.cook(force=True)
    assert_close(pf(result.geometry(), "for_color"), np.linalg.det(cauchy),
                 "tensor determinant color reduction", rtol=5e-4)
    cauchy_principal = pf(
        result.geometry(), "cauchy_eigenvalues").reshape(-1, 3)
    for reduction, component, label in (
            ("principal_max", 0, "first"),
            ("principal_middle", 1, "second"),
            ("principal_min", 2, "third")):
        node.setParms({"color_reduction": reduction})
        result.cook(force=True)
        assert_close(pf(result.geometry(), "for_color"),
                     cauchy_principal[:, component],
                     f"{label} principal-value color reduction")
    node.setParms({"color_attrib": "solution", "color_reduction": "x"})
    result.cook(force=True)
    assert_close(pf(result.geometry(), "for_color"), sol[:, 0],
                 "vector X-component color reduction")
    for reduction, expected in (
            ("y", sol[:, 1]), ("z", sol[:, 2]),
            ("magnitude", np.linalg.norm(sol, axis=1))):
        node.setParms({"color_reduction": reduction})
        result.cook(force=True)
        assert_close(pf(result.geometry(), "for_color"), expected,
                     f"vector {reduction} color reduction")
    node.setParms({"color_reduction": "z", "color_scale": 1,
                   "color_min": 1e-6, "color_max": 1.0})
    result.cook(force=True)
    assert result.geometry().attribValue("invalid_color_value_count") > 0
    node.setParms({"color_scale": 0})

    # Invalid/stale selections are replaced before they can error the node.
    node.setParms({"source_block": "Contact",
                   "color_attrib": "field_that_is_not_present",
                   "glyph_tensor": "not_a_tensor"})
    node.hdaModule().sync_available_options({"node": node})
    assert node.evalParm("source_block") == "Volume"
    assert node.evalParm("color_attrib") in node.parm("color_attrib").menuItems()
    assert node.evalParm("glyph_tensor") in node.parm("glyph_tensor").menuItems()
    result.cook(force=True)
    node.hdaModule().update_color_status({"node": node})
    assert node.evalParm("color_status").startswith("Available:")

    # Display-value choices also match the selected field's data shape.
    node.setParms({"color_attrib": "von_mises"})
    node.hdaModule().sync_available_options({"node": node})
    assert node.parm("color_reduction").menuItems() == ("auto",)
    node.setParms({"color_attrib": "solution"})
    node.hdaModule().sync_available_options({"node": node})
    assert set(node.parm("color_reduction").menuItems()) == {
        "auto", "magnitude", "x", "y", "z"}
    assert set(node.parm("color_reduction").menuLabels()) == {
        "Vector Magnitude", "X Component", "Y Component", "Z Component"}
    node.setParms({"color_attrib": "cauchy_mat"})
    node.hdaModule().sync_available_options({"node": node})
    assert "trace" in node.parm("color_reduction").menuItems()
    assert {
        "First Principal Value (Largest)", "Second Principal Value",
        "Third Principal Value (Smallest)",
    }.issubset(set(node.parm("color_reduction").menuLabels()))

    # Current-frame and sequence-wide range calculations use for_color.
    node.setParms({"color_attrib": "solution_mag", "color_reduction": "auto"})
    node.hdaModule().autoscale({"node": node})
    result.cook(force=True)
    current_values = pf(result.geometry(), "for_color")
    assert_close(np.array([node.evalParm("color_min"),
                           node.evalParm("color_max")]),
                 np.array([current_values.min(), current_values.max()]),
                 "current-frame automatic range")
    node.hdaModule().autoscale_all({"node": node})
    assert node.evalParm("color_min") <= current_values.min() + 1e-6
    assert node.evalParm("color_max") >= current_values.max() - 1e-6
    # All-frames auto range is a numpy scan that mirrors the displayed value
    # exactly (no per-frame cook): it must equal the range of the cooked
    # for_color gathered across every frame.
    truth = []
    for frame in range(len(node.hdaModule().read_pvd(pvd))):
        hou.setFrame(frame)
        result.cook(force=True)
        values = pf(result.geometry(), "for_color")
        truth.append(values[np.isfinite(values)])
    truth = np.concatenate(truth)
    node.hdaModule().autoscale_all({"node": node})
    assert_close(np.array([node.evalParm("color_min"),
                           node.evalParm("color_max")]),
                 np.array([truth.min(), truth.max()]),
                 "all-frames auto range matches displayed values")
    # The every-frame scan is memoized: it caches the per-frame values keyed on
    # the scan settings, so a same-settings run (e.g. diagnostics right after
    # auto-range) reuses them instead of re-parsing. Poison the cached values
    # and confirm the next same-settings scan reads them back.
    phm = node.hdaModule()
    assert phm._SCAN_CACHE.get("key") == phm._scan_key(node, pvd)
    phm._SCAN_CACHE["rows"] = [(frame, timestep, np.full(4, 42.0, np.float32))
                               for frame, timestep, _ in phm._SCAN_CACHE["rows"]]
    phm.compute_diagnostics({"node": node})
    poisoned = json.loads(node.evalParm("diagnostics_data"))
    assert poisoned and all(abs(row[4] - 42.0) < 1e-3 for row in poisoned), (
        "all-frames scan did not reuse the cache")
    # changing the scanned field invalidates the cache (different key)
    key_before = phm._scan_key(node, pvd)
    node.setParms({"color_attrib": "solution", "color_reduction": "magnitude"})
    assert phm._scan_key(node, pvd) != key_before
    node.setParms({"color_attrib": "solution_mag", "color_reduction": "auto"})
    phm._SCAN_CACHE.clear()
    hou.setFrame(4)
    node.setParms({"range_symmetric": 1, "range_percentile": 1,
                   "range_percentilesx": 5.0, "range_percentilesy": 95.0})
    node.hdaModule().autoscale({"node": node})
    assert abs(node.evalParm("color_min") + node.evalParm("color_max")) < 1e-6
    locked_range = (node.evalParm("color_min"), node.evalParm("color_max"))
    node.setParms({"range_lock": 1})
    node.hdaModule().autoscale({"node": node})
    assert locked_range == (node.evalParm("color_min"), node.evalParm("color_max"))
    node.setParms({"range_lock": 0, "range_symmetric": 0,
                   "range_percentile": 0})
    node.hdaModule().set_diverging_ramp({"node": node})
    assert len(node.parm("color_ramp").evalAsRamp().values()) == 3

    # full-volume display mode
    node.setParms({"surface_only": 0})
    out.cook(force=True)
    assert count_types(out.geometry()).get("Tetrahedron", 0) > 0
    node.setParms({"surface_only": 1})

    # Field smoothing: PolyFEM writes a discontinuous mesh, so raw element
    # quantities are constant per element (coincident duplicate nodes disagree).
    # The smoothing pass averages the displayed scalar onto shared vertices.
    def coincident_spread(geo):
        values = pf(geo, "for_color")
        groups = pf(geo, "coincident_id").astype(np.int64)
        order = np.argsort(groups)
        values, groups = values[order], groups[order]
        edges = np.flatnonzero(np.diff(groups)) + 1
        return max((seg.max() - seg.min())
                   for seg in np.split(values, edges) if len(seg) > 1)

    # a derived field with no PolyFEM _avg variant -> generic nodal averaging
    node.setParms({"color_attrib": "J", "color_reduction": "auto",
                   "smooth_field": 0})
    result.cook(force=True)
    assert coincident_spread(result.geometry()) > 1e-9, "expected discontinuity"
    node.setParms({"smooth_field": 1})
    result.cook(force=True)
    assert coincident_spread(result.geometry()) < 1e-6, "smoothing not applied"
    # von_mises smoothing must reuse PolyFEM's exact continuous von_mises_avg
    node.setParms({"color_attrib": "von_mises", "color_reduction": "auto"})
    result.cook(force=True)
    smoothed = pf(result.geometry(), "for_color")
    von_mises_avg = np.asarray(
        node.hdaModule().load_frame(pvd, 4)["Volume"]["point_data"][
            "von_mises_avg"])
    assert np.allclose(np.sort(smoothed), np.sort(von_mises_avg), rtol=1e-3)
    node.setParms({"smooth_field": 0})

    # All selectable principal-direction glyph tensors produce the isolated
    # glyph group, with stride limiting density.
    for glyph_tensor in ("right_cauchy_green", "left_cauchy_green",
                         "green_lagrange", "pk2", "cauchy"):
        node.setParms({"add_glyphs": 1, "glyph_tensor": glyph_tensor,
                       "glyph_stride": 25, "tensor_scale": 0.01})
        out.cook(force=True)
        glyph_group = out.geometry().findPrimGroup("readpvd_glyphs")
        assert glyph_group is not None and len(glyph_group.prims()) > 0
        assert len(glyph_group.prims()) <= int(np.ceil(n_pts / 25.0)) * 3

    # Auto glyph-scale is unit-agnostic: the largest glyphs span ~half an edge
    # whether the tensor is stress (~1e6) or stretch (~1). A flat edge/N scale
    # (the naive approach) would explode stress glyphs and hide strain glyphs.
    node.setParms({"add_glyphs": 1, "glyph_length_mode": "value"})
    src = node.node("OUT_result")
    src.cook(force=True)
    edge = src.geometry().averageEdgeLength()
    scales = {}
    for glyph_tensor, values_name in (
            ("cauchy", "cauchy_eigenvalues"),
            ("green_lagrange", "green_lagrange_eigenvalues")):
        node.setParms({"glyph_tensor": glyph_tensor})
        node.hdaModule().autoglyph({"node": node})
        scale = node.evalParm("tensor_scale")
        scales[glyph_tensor] = scale
        eig = pf(src.geometry(), values_name).reshape(-1, 3)
        value_95 = np.percentile(np.abs(eig).max(axis=1), 95)
        # largest glyphs land near the geometric target (half an edge)
        assert abs(value_95 * scale - 0.5 * edge) < 0.1 * edge, glyph_tensor
    # stress scale must be far smaller than strain scale (orders of magnitude)
    assert scales["cauchy"] < scales["green_lagrange"] * 1e-3
    node.setParms({"glyph_tensor": "cauchy", "tensor_scale": 0.01})

    # Glyphs color their own points and must NOT create a primitive Cd: a
    # prim Cd would override the object's point-Cd field coloring (regression:
    # adding glyphs with full-volume display wiped the result color).
    for surface_only in (1, 0):
        node.setParms({"add_glyphs": 0, "surface_only": surface_only,
                       "color_attrib": "von_mises"})
        out.cook(force=True)
        base = out.geometry()
        base_n = base.intrinsicValue("pointcount")
        base_cd = pf(base, "Cd").reshape(-1, 3)[:base_n]
        node.setParms({"add_glyphs": 1, "glyph_tensor": "cauchy",
                       "glyph_stride": 10})
        out.cook(force=True)
        glyphed = out.geometry()
        assert glyphed.findPrimAttrib("Cd") is None, (
            "glyphs promoted Cd to a primitive attribute")
        assert np.allclose(base_cd, pf(glyphed, "Cd").reshape(-1, 3)[:base_n]), (
            "glyphs changed the object's field coloring")
    # Smoothing gives each node one recovered tensor, so glyphs at duplicated
    # nodes agree (not noisy) and exactly one glyph is drawn per physical node.
    node.setParms({"add_glyphs": 1, "glyph_tensor": "cauchy",
                   "glyph_stride": 1, "tensor_scale": 0.01,
                   "glyph_direction_first": 1, "glyph_direction_second": 1,
                   "glyph_direction_third": 1, "smooth_field": 0})
    node.hdaModule().sync_available_options({"node": node})
    recovery = node.node("glyph_tensor_recovery")

    def group_spread(values, ids):
        order = np.argsort(ids)
        values, ids = values[order], ids[order]
        edges = np.flatnonzero(np.diff(ids)) + 1
        return max((np.ptp(seg, axis=0).max())
                   for seg in np.split(values, edges) if len(seg) > 1)

    recovery.cook(force=True)
    eig_raw = pf(recovery.geometry(), "cauchy_eigenvalues").reshape(-1, 3)
    cid = pf(recovery.geometry(), "coincident_id").astype(np.int64)
    n_nodes = len(np.unique(cid))
    assert group_spread(eig_raw, cid) > 1.0  # noisy: per-element values disagree
    node.setParms({"smooth_field": 1})
    recovery.cook(force=True)
    eig_sm = pf(recovery.geometry(), "cauchy_eigenvalues").reshape(-1, 3)
    cid_sm = pf(recovery.geometry(), "coincident_id").astype(np.int64)
    # one consistent recovered eigenvalue set per node
    assert group_spread(eig_sm, cid_sm) < 1e-3
    # exactly one glyph (3 directions) per physical node
    out.cook(force=True)
    smooth_glyphs = len(out.geometry().findPrimGroup("readpvd_glyphs").prims())
    assert smooth_glyphs == 3 * n_nodes, (smooth_glyphs, n_nodes)
    node.setParms({"smooth_field": 0})

    # leave glyphs enabled for the following direction tests
    node.setParms({"add_glyphs": 1, "surface_only": 1, "glyph_stride": 25})

    # Principal glyph directions can be displayed independently.
    glyph_counts = []
    for active_index in range(3):
        direction_values = {
            "glyph_direction_first": int(active_index == 0),
            "glyph_direction_second": int(active_index == 1),
            "glyph_direction_third": int(active_index == 2),
        }
        node.setParms(direction_values)
        out.cook(force=True)
        group = out.geometry().findPrimGroup("readpvd_glyphs")
        assert group is not None and len(group.prims()) > 0
        glyph_counts.append(len(group.prims()))
    assert max(glyph_counts) - min(glyph_counts) <= 1, glyph_counts
    node.setParms({"glyph_direction_first": 1, "glyph_direction_second": 0,
                   "glyph_direction_third": 0, "glyph_length_mode": 1,
                   "glyph_arrowheads": 0})
    out.cook(force=True)
    plain_count = len(out.geometry().findPrimGroup("readpvd_glyphs").prims())
    node.setParms({"glyph_arrowheads": 1})
    out.cook(force=True)
    arrow_count = len(out.geometry().findPrimGroup("readpvd_glyphs").prims())
    assert arrow_count == plain_count * 3
    node.setParms({"glyph_direction_first": 0, "glyph_direction_second": 0,
                   "glyph_direction_third": 0})
    out.cook(force=True)
    empty_group = out.geometry().findPrimGroup("readpvd_glyphs")
    assert empty_group is None or len(empty_group.prims()) == 0
    node.setParms({"glyph_direction_first": 1, "glyph_direction_second": 1,
                   "glyph_direction_third": 1, "glyph_arrowheads": 0,
                   "glyph_length_mode": 0})
    node.setParms({"add_glyphs": 0})

    # Renderable scene decorations are grouped and merged only at final output.
    assert result.geometry().findPrimGroup("readpvd_legend") is None
    node.setParms({"legend_mode": 1, "gnomon_show": 1})
    out.cook(force=True)
    decorated = out.geometry()
    for group_name in ("readpvd_legend", "readpvd_gnomon"):
        group = decorated.findPrimGroup(group_name)
        assert group is not None and len(group.prims()) > 0, group_name
    node.setParms({"legend_mode": 0, "gnomon_show": 0})

    # User-facing controls use full mechanics terminology, and the HDA embeds
    # a registered overlay viewer state rather than the legacy unfinished one.
    ptg = node.type().definition().parmTemplateGroup()
    assert "Second Piola-Kirchhoff Stress Tensor" in (
        node.parm("glyph_tensor").menuLabels())
    assert "maximum to minimum" in ptg.find("color_reduction").help()
    sections = node.type().definition().sections()
    assert "ViewerStateModule" in sections
    assert "readpvd_overlay_legend_bar" in sections["ViewerStateModule"].contents()

    # Surface block with contact forces
    surf_pvd = ensure_surface_run()
    node.setParms({"PVD_file": surf_pvd, "source_block": "Contact"})
    node.hdaModule().sync_available_options({"node": node})
    assert set(node.parm("source_block").menuItems()) == {
        "Volume", "Surface", "Contact"}
    # Upstream (varforms) only writes friction_forces when friction is
    # active and adds gradient/displacement arrays; require the essentials
    # instead of an exact set so output-schema additions don't break us.
    assert {"contact_forces", "solution", "solution_mag"} <= set(
        node.parm("color_attrib").menuItems())
    assert node.parm("glyph_tensor").menuItems() == ("",)
    node.setParms({"add_glyphs": 1})
    node.hdaModule().sync_available_options({"node": node})
    assert node.evalParm("add_glyphs") == 0
    node.setParms({"color_attrib": "contact_forces",
                   "color_reduction": "magnitude"})
    hou.setFrame(4)
    out.cook(force=True)
    sgeo = out.geometry()
    assert sgeo.findPointAttrib("contact_forces") is not None
    cf = pf(sgeo, "contact_forces").reshape(-1, 3)
    assert np.abs(cf).max() > 0, "contact forces all zero"

    # Supplementary blocks display together with independent block groups.
    node.setParms({
        "source_block": "Volume", "show_multi_blocks": 1,
        "multi_surface_show": 1, "multi_contact_show": 1,
        "multi_surface_field": "sidesets",
        "multi_contact_field": "contact_forces"})
    node.hdaModule().sync_available_options({"node": node})
    out.cook(force=True)
    multi_geo = out.geometry()
    for name in ("readpvd_block_surface", "readpvd_block_contact"):
        group = multi_geo.findPrimGroup(name)
        assert group is not None and len(group.prims()) > 0, name
    node.setParms({"show_multi_blocks": 0})

    # Reference comparison, probing, clipping/slicing, and timeline plot.
    node.setParms({"color_attrib": "solution", "color_reduction": "magnitude",
                   "reference_enable": 1, "reference_frame": 0,
                   "reference_mode": 0})
    hou.setFrame(4)
    result.cook(force=True)
    comparison = pf(result.geometry(), "reference_comparison")
    assert np.abs(comparison).max() > 0
    assert node.evalParm("reference_status").startswith("Comparing")
    readout = node.hdaModule().probe_point(node, 0)
    assert "Point 0" in readout and "Displacement Vector" in readout
    # The probe marker/readout follow the point as the frame changes: the
    # live position must match the deformed point and differ across frames.
    probe_pt = 770
    node.setParms({"show_deformed": 1})
    hou.setFrame(0)
    out.cook(force=True)
    pos0, _ = node.hdaModule().probe_readout(node, probe_pt)
    hou.setFrame(4)
    out.cook(force=True)
    pos4, text4 = node.hdaModule().probe_readout(node, probe_pt)
    assert pos0 is not None and pos4 is not None
    assert not np.allclose(pos0, pos4), "probe marker did not move with frame"
    assert np.allclose(pos4, out.geometry().point(probe_pt).position()), (
        "probe marker position does not match the live point")
    node.setParms({"reference_enable": 0})
    out.cook(force=True)
    unclipped_prims = len(out.geometry().prims())
    node.setParms({"clip_mode": 1, "clip_originx": 0.5,
                   "clip_directionx": 1.0, "clip_directiony": 0.0,
                   "clip_directionz": 0.0})
    out.cook(force=True)
    assert len(out.geometry().prims()) < unclipped_prims
    node.setParms({"clip_mode": 2, "slice_thickness": 0.05})
    out.cook(force=True)
    assert len(out.geometry().prims()) > 0
    node.setParms({"clip_mode": 0})

    # Clipping must cut the model but leave glyphs intact (they merge in after
    # the clip stage). The glyph prim count is identical clipped vs unclipped.
    node.setParms({"add_glyphs": 1, "glyph_tensor": "cauchy",
                   "glyph_stride": 5, "clip_mode": 0})
    out.cook(force=True)
    glyphs_unclipped = len(out.geometry().findPrimGroup("readpvd_glyphs").prims())
    model_unclipped = len(out.geometry().prims()) - glyphs_unclipped
    node.setParms({"clip_mode": 1, "clip_originx": 0.5, "clip_directionx": 1.0,
                   "clip_directiony": 0.0, "clip_directionz": 0.0})
    out.cook(force=True)
    clipped = out.geometry()
    glyphs_clipped = len(clipped.findPrimGroup("readpvd_glyphs").prims())
    model_clipped = len(clipped.prims()) - glyphs_clipped
    assert glyphs_clipped == glyphs_unclipped, "clipping cut the glyphs"
    assert model_clipped < model_unclipped, "clipping did not cut the model"
    node.setParms({"clip_mode": 0, "add_glyphs": 0})

    # Loading a PVD auto-centers the clip/slice plane on the scene's
    # bounding-box center (only while the origin is still at 0,0,0).
    cnode = hou.node("/obj").createNode("readPVD::1.0", "viewer_autocenter")
    cnode.setParms({"PVD_file": pvd})
    cnode.hdaModule().start({"node": cnode})
    blocks0 = cnode.hdaModule().load_frame(pvd, 0)
    pts = np.concatenate([np.asarray(m["points"], dtype=np.float64)
                          for m in blocks0.values()])
    expected_center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    origin = [cnode.evalParm("clip_origin" + axis) for axis in "xyz"]
    assert np.allclose(origin, expected_center, atol=1e-6), (
        origin, expected_center)
    assert not np.allclose(origin, 0), "scene center should not be the origin"
    # the default +X normal gets aligned to the longest bounding-box axis
    extent = pts.max(axis=0) - pts.min(axis=0)
    expected_normal = np.eye(3)[int(np.argmax(extent))]
    normal = [cnode.evalParm("clip_direction" + axis) for axis in "xyz"]
    assert np.allclose(normal, expected_normal), (normal, extent)
    # user-moved origin and normal survive a reload of the same file
    cnode.setParms({"clip_originx": 9.0,
                    "clip_directionx": 0.0, "clip_directiony": 0.0,
                    "clip_directionz": 1.0})
    cnode.hdaModule().start({"node": cnode})
    assert cnode.evalParm("clip_originx") == 9.0
    assert cnode.evalParm("clip_directionz") == 1.0
    assert cnode.evalParm("clip_directionx") == 0.0
    cnode.destroy()

    # ---- growing-sim refresh: caches and frame position survive ---------
    grow_dir = os.path.join(ROOT, "smoke-out", "readpvd-growing")
    os.makedirs(grow_dir, exist_ok=True)
    tet_points = "0 0 0  1 0 0  0 1 0  0 0 1"

    def write_grow_vtu(name, value):
        with open(os.path.join(grow_dir, name), "w") as file:
            file.write(
                '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
                '<Piece NumberOfPoints="4" NumberOfCells="1"><Points>'
                '<DataArray type="Float64" NumberOfComponents="3" '
                f'format="ascii">{tet_points}</DataArray></Points><Cells>'
                '<DataArray type="Int64" Name="connectivity" format="ascii">'
                '0 1 2 3</DataArray><DataArray type="Int64" Name="offsets" '
                'format="ascii">4</DataArray><DataArray type="UInt8" '
                'Name="types" format="ascii">10</DataArray></Cells>'
                '<PointData><DataArray type="Float64" Name="solution" '
                'NumberOfComponents="3" format="ascii">'
                + f"{value} 0 0  " * 4 + '</DataArray></PointData>'
                '</Piece></UnstructuredGrid></VTKFile>')

    def write_grow_pvd(frames):
        with open(os.path.join(grow_dir, "grow.pvd"), "w") as file:
            file.write(
                '<VTKFile type="Collection"><Collection>'
                + "".join(f'<DataSet timestep="{i}" file="g{i}.vtu"/>'
                          for i in range(frames))
                + '</Collection></VTKFile>')

    write_grow_vtu("g0.vtu", 0.0)
    write_grow_pvd(1)
    grow_pvd = os.path.join(grow_dir, "grow.pvd")
    gnode = hou.node("/obj").createNode("readPVD::1.0", "viewer_growing")
    gphm = gnode.hdaModule()
    gnode.setParms({"PVD_file": grow_pvd, "cache": 1})
    gphm.toggle_cache({"node": gnode})
    gphm.start({"node": gnode})
    g0_key = gphm._file_key(os.path.join(grow_dir, "g0.vtu"))
    gphm.load_frame(grow_pvd, 0)
    assert g0_key in gphm._VTU_CACHE, "frame 0 not in the parsed-file cache"

    # the sim writes a new timestep: refresh must extend the sequence while
    # keeping the parsed-file cache and the current frame position intact
    write_grow_vtu("g1.vtu", 1.0)
    write_grow_pvd(2)
    hou.setFrame(1)
    gphm.refresh({"node": gnode})
    assert hou.frame() == 1, "refresh moved the current frame"
    assert g0_key in gphm._VTU_CACHE, "refresh dropped an unchanged frame"
    assert len(gphm.read_pvd(grow_pvd)) == 2
    gout = gnode.node("output")
    gout.cook(force=True)
    assert abs(pf(gout.geometry(), "solution")[0] - 1.0) < 1e-6, (
        "frame 1 data not visible after refresh")

    # a mid-write blip (pvd lists a file that is not on disk yet) must not
    # reset the user's block/color selections
    write_grow_pvd(3)                       # g2.vtu does not exist yet
    hou.setFrame(2)
    prev_block = gnode.evalParm("source_block")
    prev_color = gnode.evalParm("color_attrib")
    gphm.refresh({"node": gnode})
    assert gnode.evalParm("source_block") == prev_block
    assert gnode.evalParm("color_attrib") == prev_color
    assert "keeping settings" in gnode.evalParm("availability_status")
    hou.setFrame(1)

    # loading a DIFFERENT pvd (start) is the one path that flushes caches
    gphm._VTU_CACHE["sentinel"] = {}
    gnode.setParms({"PVD_file": pvd})
    gphm.start({"node": gnode})
    assert "sentinel" not in gphm._VTU_CACHE, "start did not flush caches"
    gnode.destroy()

    node.hdaModule().compute_diagnostics({"node": node})
    # numpy scan stores frame, timestep, min, mean, max per frame
    diag_rows = json.loads(node.evalParm("diagnostics_data"))
    assert diag_rows and all(len(row) == 5 for row in diag_rows)
    assert [row[2] <= row[3] <= row[4] for row in diag_rows]  # min<=mean<=max
    node.setParms({"diagnostics_show": 1, "field_units": "m",
                   "legend_number_format": 2, "diagnostics_grid": 1,
                   "diagnostics_band": 1})
    out.cook(force=True)
    diagnostic_group = out.geometry().findPrimGroup("readpvd_diagnostics")
    assert diagnostic_group is not None and len(diagnostic_group.prims()) == 3
    # axes, gridlines, numbered tick labels, band, titles, and legend
    chrome = out.geometry().findPrimGroup("readpvd_diagnostics_axes")
    assert chrome is not None and len(chrome.prims()) > 20
    assert "[m]" in node.hdaModule().legend_title_text(node)
    node.setParms({"diagnostics_show": 0})

    # CellData remains on primitives, is promoted for display, and temporal
    # field menus distinguish every-frame and intermittent data.
    cell_pvd = ensure_cell_data_run()
    node.setParms({"PVD_file": cell_pvd, "source_block": "Volume",
                   "surface_only": 0, "field_time_scope": 1})
    hou.setFrame(0)
    node.hdaModule().sync_available_options({"node": node})
    assert "cell_scalar" in node.parm("color_attrib").menuItems()
    assert "intermittent" not in node.parm("color_attrib").menuItems()
    node.setParms({"field_time_scope": 2})
    node.hdaModule().sync_available_options({"node": node})
    assert "intermittent" in node.parm("color_attrib").menuItems()
    assert any("1/2 frames" in label
               for label in node.parm("color_attrib").menuLabels())
    node.setParms({"color_attrib": "cell_scalar",
                   "color_reduction": "auto"})
    out.cook(force=True)
    cell_geo = out.geometry()
    assert cell_geo.findPrimAttrib("cell_scalar") is not None
    assert cell_geo.findPointAttrib("cell_scalar") is not None
    assert set(round(value, 5) for value in
               cell_geo.primFloatAttribValues("cell_scalar")) == {1.0, 3.0}

    # Show/hide bodies: the menu is data-driven, isolation filters the result
    # to the chosen bodies, and the all-frames range follows what is visible.
    two_body = ensure_two_body_run()
    node.setParms({"PVD_file": two_body, "source_block": "Volume",
                   "surface_only": 0, "field_time_scope": 0,
                   "color_attrib": "von_mises", "color_reduction": "auto",
                   "visible_bodies": ""})
    hou.setFrame(0)
    node.hdaModule().sync_available_options({"node": node})
    assert node.parm("visible_bodies").menuItems() == ("0", "1")
    assert node.evalParm("has_multibody") == 1

    def body_set(geo):
        ids = pf(geo, "body_ids").astype(np.int64)
        return set(np.unique(ids[ids >= 0]).tolist())

    out.cook(force=True)
    assert body_set(out.geometry()) == {0, 1}
    node.setParms({"visible_bodies": "1"})
    out.cook(force=True)
    assert body_set(out.geometry()) == {1}, "isolation did not hide body 0"
    node.setParms({"visible_bodies": "0"})
    out.cook(force=True)
    assert body_set(out.geometry()) == {0}
    # auto range follows the visible body (body 0 -> 10, all -> 20)
    node.hdaModule().autoscale_all({"node": node})
    assert abs(node.evalParm("color_max") - 10.0) < 1e-4
    node.setParms({"visible_bodies": ""})
    node.hdaModule().autoscale_all({"node": node})
    assert abs(node.evalParm("color_max") - 20.0) < 1e-4

    # Higher-order Lagrange tets are subdivided for display (not collapsed to a
    # corner tet): P2/P3/P4 -> 8/27/64 P1 sub-tets, field data preserved.
    phm = node.hdaModule()
    ho_dir = os.path.join(ROOT, "smoke-out", "readpvd-higher-order")
    os.makedirs(ho_dir, exist_ok=True)
    for n_nodes, expected in ((10, 8), (20, 27), (35, 64)):
        conn = " ".join(str(i) for i in range(n_nodes))
        pts = " ".join("%.4f" % v for v in np.random.rand(n_nodes * 3))
        vm = " ".join("%.3f" % v for v in np.random.rand(n_nodes))
        path = os.path.join(ho_dir, f"tet{n_nodes}.vtu")
        with open(path, "w") as file:
            file.write(
                '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
                f'<Piece NumberOfPoints="{n_nodes}" NumberOfCells="1">'
                f'<Points><DataArray type="Float64" NumberOfComponents="3" '
                f'format="ascii">{pts}</DataArray></Points><Cells>'
                f'<DataArray type="Int64" Name="connectivity" format="ascii">'
                f'{conn}</DataArray><DataArray type="Int64" Name="offsets" '
                f'format="ascii">{n_nodes}</DataArray><DataArray type="UInt8" '
                'Name="types" format="ascii">71</DataArray></Cells><PointData>'
                f'<DataArray type="Float64" Name="von_mises" format="ascii">'
                f'{vm}</DataArray></PointData></Piece></UnstructuredGrid>'
                '</VTKFile>')
        mesh = phm.read_vtu(path)
        sub_tets = mesh["cells"]["tet"]
        assert len(sub_tets) == expected, (n_nodes, len(sub_tets))
        assert sub_tets.max() < n_nodes              # only references real nodes
        assert len(mesh["point_data"]["von_mises"]) == n_nodes

    # Cauchy stress is reconstructed from F + 1st PK stress when Cauchy itself
    # was not exported (sigma = (1/J) P F^T), so a lighter PolyFEM output that
    # drops the redundant Cauchy/PK2 tensors still yields the full derived set.
    np.random.seed(0)
    F = np.eye(3) + 0.15 * np.random.rand(6, 3, 3)
    sym = np.random.rand(6, 3, 3)
    sigma = 0.5 * (sym + np.transpose(sym, (0, 2, 1))) * 1e6
    jac = np.linalg.det(F)
    pk1 = jac[:, None, None] * np.matmul(
        sigma, np.transpose(np.linalg.inv(F), (0, 2, 1)))
    fields = "".join(
        '<DataArray type="Float64" Name="%s_%d" NumberOfComponents="3" '
        'format="ascii">%s</DataArray>' % (
            # PolyFEM flattens column-major: X_i arrays are tensor COLUMNS
            name, i + 1, " ".join("%.9g" % v for v in tensor[:, :, i].ravel()))
        for name, tensor in (("F", F), ("pk1_stess", pk1)) for i in range(3))
    pk1_pvd = os.path.join(ROOT, "smoke-out", "readpvd-pk1", "e.vtu")
    os.makedirs(os.path.dirname(pk1_pvd), exist_ok=True)
    with open(pk1_pvd, "w") as file:
        file.write(
            '<VTKFile type="UnstructuredGrid"><UnstructuredGrid>'
            '<Piece NumberOfPoints="6" NumberOfCells="1"><Points>'
            '<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
            + " ".join("%.4f" % v for v in np.random.rand(18))
            + '</DataArray></Points><Cells><DataArray type="Int64" '
            'Name="connectivity" format="ascii">0 1 2 3</DataArray>'
            '<DataArray type="Int64" Name="offsets" format="ascii">4</DataArray>'
            '<DataArray type="UInt8" Name="types" format="ascii">10</DataArray>'
            '</Cells><PointData>' + fields
            + '</PointData></Piece></UnstructuredGrid></VTKFile>')
    pk1_mesh = phm.read_vtu(pk1_pvd)
    assert not any("cauchy" in name for name in pk1_mesh["point_data"])
    assert_close(phm._mesh_field(pk1_mesh, "cauchy_mat").reshape(-1, 3, 3),
                 sigma, "Cauchy reconstructed from F + pk1", rtol=1e-4, atol=1)
    assert phm._requirements({"F_1": 3, "F_2": 3, "F_3": 3, "pk1_stress_1": 3,
                              "pk1_stress_2": 3, "pk1_stress_3": 3})[
        "stress_derived"]

    # Large inline VTUs: the whole document can exceed expat's int-sized feed
    # (OverflowError) and building a full ElementTree of the base64 text can
    # exhaust memory. The parser feeds the XML in chunks and strips inline
    # <DataArray> payloads (decoding them on demand). Exercise both paths.
    doc = b'<VTKFile a="1"><X>' + b'y' * 40 + b'</X><Z>zz</Z></VTKFile>'
    saved_chunk = phm._XML_FEED_CHUNK
    try:
        phm._XML_FEED_CHUNK = 8                       # force chunked feeding
        root = phm._parse_xml(doc)
    finally:
        phm._XML_FEED_CHUNK = saved_chunk
    assert root.tag == "VTKFile" and [c.tag for c in root] == ["X", "Z"]
    skeleton, spans = phm._strip_inline_payloads(
        b'<r><DataArray>PAYLOAD</DataArray><DataArray o="1"/></r>')
    assert b"PAYLOAD" not in skeleton                 # inline payload removed
    assert spans[0] is not None and spans[1] is None  # inline span, self-closing

    print(f"\nPASS: readPVD 1.0 ({frame_cook*1000:.0f} ms/frame on this mesh; "
          f"contact-force max {np.abs(cf).max():.3g})")


if __name__ == "__main__":
    main()
