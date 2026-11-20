"""Build readPVD::1.0 (.hdanc). Run under hython via src/build_all.py."""

import base64
import os
import sys

import hou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import hda_build  # noqa: E402

TYPE_NAME = "readPVD::1.0"
LABEL = "Read PVD 1.0 (PolyFEM results)"
# A conditional must compare a parameter: "{ 1 == 1 }" names none, so Houdini
# ignores it -- until 2026-10-01 the internal parameters below were visible
# and the status fields editable. Internal parameters now carry the
# template's hidden flag; read-only fields compare the file path with a value
# no path takes.
READ_ONLY = "{ PVD_file != __readpvd_read_only__ }"
TIME_MAPPING_TOKENS = ("auto", "steps", "time")
# Official h5py wheels from PyPI (SHA-256 in vendor/README.md), embedded as
# base64 sections; the section names match EMBEDDED_H5PY_SECTIONS in the
# PythonModule.
H5PY_WHEELS = {
    "h5py-3.16.0-cp313-macos-arm64.whl.b64":
        "h5py-3.16.0-cp313-cp313-macosx_11_0_arm64.whl",
    "h5py-3.16.0-cp313-macos-x86_64.whl.b64":
        "h5py-3.16.0-cp313-cp313-macosx_10_13_x86_64.whl",
    "h5py-3.16.0-cp313-linux-x86_64.whl.b64":
        "h5py-3.16.0-cp313-cp313-manylinux_2_28_x86_64.whl",
    "h5py-3.16.0-cp313-windows-amd64.whl.b64":
        "h5py-3.16.0-cp313-cp313-win_amd64.whl",
}

TOPO_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("topoframe")  # cook dependencies
node.evalParm("dataepoch")
node.parent().hdaModule().cook_topology(node)
"""

FRAME_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("frame")  # cook dependencies
node.evalParm("dataepoch")
node.parent().hdaModule().cook_frame(node)
"""

LEGEND_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_scene_legend(node)
"""

GNOMON_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_gnomon(node)
"""

MULTI_BLOCK_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("frame")
node.evalParm("dataepoch")
node.parent().hdaModule().cook_multi_blocks(node)
"""

REFERENCE_SOP_CODE = """\
node = hou.pwd()
node.evalParm("frame")
node.evalParm("dataepoch")
node.parent().hdaModule().cook_reference_comparison(node)
"""

SMOOTH_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_field_smoothing(node)
"""

GLYPH_RECOVERY_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_glyph_recovery(node)
"""

FIBER_RECOVERY_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_fiber_recovery(node)
"""

BUILD_PRIMS_VEX = """\
int tet[] = detail(0, "tet_conn");
int tet_bits[] = detail(0, "tet_face_bits");
for (int i = 0; i < len(tet); i += 4) {
    // Gmsh/VTK corners [1,2,3,0]: positive by Houdini's convention
    int prim = addprim(0, "tet", tet[i+1], tet[i+2], tet[i+3], tet[i]);
    setprimattrib(0, "source_cell_index", prim, i / 4);
    setprimattrib(0, "source_cell_family", prim, "tet");
    if (len(tet_bits))
        setprimattrib(0, "readpvd_face_bits", prim, tet_bits[i / 4]);
}
int hexc[] = detail(0, "hex_conn");
int hex_bits[] = detail(0, "hex_face_bits");
for (int i = 0; i < len(hexc); i += 8) {
    int prim = addprim(0, "hex", hexc[i], hexc[i+1], hexc[i+3], hexc[i+2],
                      hexc[i+4], hexc[i+5], hexc[i+7], hexc[i+6]);
    setprimattrib(0, "source_cell_index", prim, i / 8);
    setprimattrib(0, "source_cell_family", prim, "hex");
    if (len(hex_bits))
        setprimattrib(0, "readpvd_face_bits", prim, hex_bits[i / 8]);
}
int tri[] = detail(0, "tri_conn");
for (int i = 0; i < len(tri); i += 3) {
    int prim = addprim(0, "poly", tri[i], tri[i+1], tri[i+2]);
    setprimattrib(0, "source_cell_index", prim, i / 3);
    setprimattrib(0, "source_cell_family", prim, "tri");
}
int quad[] = detail(0, "quad_conn");
for (int i = 0; i < len(quad); i += 4) {
    int prim = addprim(0, "poly", quad[i], quad[i+1], quad[i+2], quad[i+3]);
    setprimattrib(0, "source_cell_index", prim, i / 4);
    setprimattrib(0, "source_cell_family", prim, "quad");
}
int lin[] = detail(0, "line_conn");
for (int i = 0; i < len(lin); i += 2) {
    int prim = addprim(0, "polyline", lin[i], lin[i+1]);
    setprimattrib(0, "source_cell_index", prim, i / 2);
    setprimattrib(0, "source_cell_family", prim, "line");
}
"""

BOUNDARY_VEX = """\
// Element faces for display; runs only on topology changes (placed before
// the per-frame attribute update). PolyFEM duplicates vertices per element,
// so tet_adjacent/hex_adjacent see every face as boundary: the topology stage
// matches faces by coincident vertex (and body) instead and passes one bit
// per face in i@readpvd_face_bits. Interior faces are kept in a group so the
// display can drop them, except while clipping.
int vcount = primvertexcount(0, @primnum);
int bits = i@readpvd_face_bits;
if (vcount == 8) {
    for (int f = 0, mask = 1; f < 6; f++, mask *= 2) {
        int pts[];
        for (int j = 0; j < 4; j++)
            append(pts, primpoint(0, @primnum, hex_faceindex(f, j)));
        int face = addprim(0, "poly", pts[0], pts[1], pts[2], pts[3]);
        setprimattrib(0, "source_cell_index", face, i@source_cell_index);
        setprimattrib(0, "source_cell_family", face, s@source_cell_family);
        setprimgroup(0, "readpvd_interior", face, (bits / mask) % 2);
    }
    removeprim(0, @primnum, 0);
} else if (vcount == 4 && primintrinsic(0, "typename", @primnum) == "Tetrahedron") {
    // faces 0,2,1,3: the same faces in the same order as the tet order
    // [1,3,2,0] used before 2026-11-15 gave (see build_prims)
    int tet_faces[] = array(0, 2, 1, 3);
    int face_mask[] = array(1, 2, 4, 8);     // bit f of readpvd_face_bits
    foreach (int f; tet_faces) {
        int mask = face_mask[f];
        int pts[];
        for (int j = 0; j < 3; j++)
            append(pts, primpoint(0, @primnum, tet_faceindex(f, j)));
        int face = addprim(0, "poly", pts[0], pts[1], pts[2]);
        setprimattrib(0, "source_cell_index", face, i@source_cell_index);
        setprimattrib(0, "source_cell_family", face, s@source_cell_family);
        setprimgroup(0, "readpvd_interior", face, (bits / mask) % 2);
    }
    removeprim(0, @primnum, 0);
}
"""

DROP_INTERIOR_VEX = """\
if (inprimgroup(0, "readpvd_interior", @primnum))
    removeprim(0, @primnum, 0);
"""

DEFORM_VEX = """\
v@rest = v@P;
if (chi("../show_deformed") && haspointattrib(0, "solution"))
    v@P += v@solution;
"""

DERIVED_PREPARE_VEX = """\
// Detail wrangle: allocate only the derived point attributes needed by the
// current display, glyphs, fibers, and probe (or every derived field when
// Derived Fields Scope is set to 'All Mechanics Fields'). Static @attrib
// references in a point wrangle would pre-allocate every attribute even inside
// untaken branches (~780 B/pt); dynamic creation here keeps on-demand mode
// near zero memory overhead.
int all_fields = chs("../derived_mode") == "all";
string color = chs("../color_attrib");
int glyphs = chi("../add_glyphs");
string gtensor = glyphs ? chs("../glyph_tensor") : "";
int probe = chi("../probe_enabled") || chi("../probe_point") >= 0;
int fibers = chi("../show_fibers");
matrix3 zero_m = 0;
vector zero_v = {0, 0, 0};

setdetailattrib(0, "__derived_fill_von_mises", 0, "set");

if (haspointattrib(0, "solution") && (all_fields || color == "solution_mag"))
    addpointattrib(0, "solution_mag", 0.0);

int have_F = haspointattrib(0, "F_1") && haspointattrib(0, "F_2")
    && haspointattrib(0, "F_3");
if (!have_F)
    return;

if (all_fields || fibers || color == "F_mat")
    addpointattrib(0, "F_mat", zero_m);
if (all_fields || color == "J")
    addpointattrib(0, "J", 0.0);

if (all_fields || probe || color == "principal_stretches")
    addpointattrib(0, "principal_stretches", zero_v);
if (all_fields || gtensor == "right_cauchy_green"
    || color == "right_cauchy_green")
    addpointattrib(0, "right_cauchy_green", zero_m);
if (all_fields || gtensor == "right_cauchy_green"
    || color == "right_cauchy_green_eigenvalues")
    addpointattrib(0, "right_cauchy_green_eigenvalues", zero_v);
if (all_fields || gtensor == "right_cauchy_green"
    || color == "right_cauchy_green_eigenvectors")
    addpointattrib(0, "right_cauchy_green_eigenvectors", zero_m);
if (all_fields || color == "right_stretch")
    addpointattrib(0, "right_stretch", zero_m);
if (all_fields || color == "hencky_strain")
    addpointattrib(0, "hencky_strain", zero_m);
if (all_fields || color == "hencky_strain_eigenvalues")
    addpointattrib(0, "hencky_strain_eigenvalues", zero_v);
if (all_fields || gtensor == "green_lagrange"
    || color == "green_lagrange_strain")
    addpointattrib(0, "green_lagrange_strain", zero_m);
if (all_fields || gtensor == "green_lagrange"
    || color == "green_lagrange_eigenvalues")
    addpointattrib(0, "green_lagrange_eigenvalues", zero_v);
if (all_fields || gtensor == "green_lagrange"
    || color == "green_lagrange_eigenvectors")
    addpointattrib(0, "green_lagrange_eigenvectors", zero_m);

if (all_fields || gtensor == "left_cauchy_green"
    || color == "left_cauchy_green")
    addpointattrib(0, "left_cauchy_green", zero_m);
if (all_fields || gtensor == "left_cauchy_green"
    || color == "left_cauchy_green_eigenvalues")
    addpointattrib(0, "left_cauchy_green_eigenvalues", zero_v);
if (all_fields || gtensor == "left_cauchy_green"
    || color == "left_cauchy_green_eigenvectors")
    addpointattrib(0, "left_cauchy_green_eigenvectors", zero_m);
if (all_fields || color == "left_stretch")
    addpointattrib(0, "left_stretch", zero_m);
if (all_fields || color == "almansi_strain")
    addpointattrib(0, "almansi_strain", zero_m);
if (all_fields || color == "almansi_strain_eigenvalues")
    addpointattrib(0, "almansi_strain_eigenvalues", zero_v);

if (all_fields || color == "infinitesimal_strain")
    addpointattrib(0, "infinitesimal_strain", zero_m);
if (all_fields || color == "infinitesimal_strain_eigenvalues")
    addpointattrib(0, "infinitesimal_strain_eigenvalues", zero_v);
if (all_fields || color == "infinitesimal_strain_eigenvectors")
    addpointattrib(0, "infinitesimal_strain_eigenvectors", zero_m);

int have_cauchy = haspointattrib(0, "cauchy_stress_1")
    && haspointattrib(0, "cauchy_stress_2")
    && haspointattrib(0, "cauchy_stress_3");
int have_pk1 = haspointattrib(0, "pk1_stress_1")
    && haspointattrib(0, "pk1_stress_2")
    && haspointattrib(0, "pk1_stress_3");
if (!(have_cauchy || have_pk1))
    return;

if (all_fields || gtensor == "cauchy" || color == "cauchy_mat")
    addpointattrib(0, "cauchy_mat", zero_m);
if (all_fields || probe || gtensor == "cauchy" || color == "cauchy_eigenvalues")
    addpointattrib(0, "cauchy_eigenvalues", zero_v);
if (all_fields || gtensor == "cauchy" || color == "cauchy_eigenvectors")
    addpointattrib(0, "cauchy_eigenvectors", zero_m);
if (all_fields || color == "cauchy_trace")
    addpointattrib(0, "cauchy_trace", 0.0);
if (all_fields || color == "hydrostatic_stress")
    addpointattrib(0, "hydrostatic_stress", 0.0);
if (all_fields || color == "deviatoric_stress")
    addpointattrib(0, "deviatoric_stress", zero_m);
if (all_fields || color == "stress_J2")
    addpointattrib(0, "stress_J2", 0.0);
if (all_fields || color == "stress_J3")
    addpointattrib(0, "stress_J3", 0.0);
if (all_fields || color == "von_mises_derived")
    addpointattrib(0, "von_mises_derived", 0.0);
if (all_fields || color == "max_shear_stress")
    addpointattrib(0, "max_shear_stress", 0.0);
if (all_fields || color == "stress_triaxiality")
    addpointattrib(0, "stress_triaxiality", 0.0);
if (!haspointattrib(0, "von_mises") && (all_fields || color == "von_mises")) {
    addpointattrib(0, "von_mises", 0.0);
    setdetailattrib(0, "__derived_fill_von_mises", 1, "set");
}
if (all_fields || color == "pk1")
    addpointattrib(0, "pk1", zero_m);
if (all_fields || gtensor == "pk2" || color == "pk2")
    addpointattrib(0, "pk2", zero_m);
if (all_fields || probe || gtensor == "pk2" || color == "pk2_eigenvalues")
    addpointattrib(0, "pk2_eigenvalues", zero_v);
if (all_fields || gtensor == "pk2" || color == "pk2_eigenvectors")
    addpointattrib(0, "pk2_eigenvectors", zero_m);
"""

DERIVED_VEX = """\
float matrix_trace(matrix3 m) {
    return getcomp(m, 0, 0) + getcomp(m, 1, 1) + getcomp(m, 2, 2);
}

float matrix_norm_squared(matrix3 m) {
    float result = 0.0;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) {
            float value = getcomp(m, row, column);
            result += value * value;
        }
    return result;
}

void sorted_eigen(matrix3 m; export vector vals; export matrix3 vecs) {
    vector raw_vals = {0, 0, 0};
    matrix3 raw_vecs = diagonalizesymmetric(m, raw_vals);
    float unsorted[] = array(raw_vals.x, raw_vals.y, raw_vals.z);
    int order[] = argsort(unsorted);
    for (int j = 0; j < 3; j++) {
        int src = order[2-j];
        setcomp(vals, getcomp(raw_vals, src), j);
        // diagonalizesymmetric returns eigenvectors as ROWS; store eigenvector
        // 'src' (row src of raw_vecs) as COLUMN j so that
        // vecs * diag(vals) * transpose(vecs) reconstructs the tensor.
        for (int r = 0; r < 3; r++)
            setcomp(vecs, getcomp(raw_vecs, src, r), r, j);
    }
}

// Rebuild a symmetric tensor from a function applied to its eigenvalues,
// Q diag(fvals) Q^T (eigenvectors are the columns of vecs). Used for the
// stretch tensors sqrt(C)/sqrt(B) and the logarithmic (Hencky) strain.
matrix3 rebuild_symmetric(vector fvals; matrix3 vecs) {
    matrix3 diag = 0;
    setcomp(diag, fvals.x, 0, 0);
    setcomp(diag, fvals.y, 1, 1);
    setcomp(diag, fvals.z, 2, 2);
    return vecs * diag * transpose(vecs);
}

if (haspointattrib(0, "solution_mag") && haspointattrib(0, "solution")) {
    vector sol = point(0, "solution", @ptnum);
    setpointattrib(0, "solution_mag", @ptnum, length(sol));
}

if (!(haspointattrib(0, "F_1") && haspointattrib(0, "F_2")
      && haspointattrib(0, "F_3")))
    return;

int want_F_mat = haspointattrib(0, "F_mat");
int want_J = haspointattrib(0, "J");

int want_rcg = haspointattrib(0, "right_cauchy_green");
int want_rcg_vals = haspointattrib(0, "right_cauchy_green_eigenvalues");
int want_rcg_vecs = haspointattrib(0, "right_cauchy_green_eigenvectors");
int want_stretches = haspointattrib(0, "principal_stretches");
int want_right_stretch = haspointattrib(0, "right_stretch");
int want_hencky = haspointattrib(0, "hencky_strain");
int want_hencky_vals = haspointattrib(0, "hencky_strain_eigenvalues");
int want_green = haspointattrib(0, "green_lagrange_strain");
int want_green_vals = haspointattrib(0, "green_lagrange_eigenvalues");
int want_green_vecs = haspointattrib(0, "green_lagrange_eigenvectors");
int need_C_eigen = want_rcg_vals || want_rcg_vecs || want_stretches
    || want_right_stretch || want_hencky || want_hencky_vals
    || want_green_vals || want_green_vecs;
int need_C = need_C_eigen || want_rcg || want_green;

int want_lcg = haspointattrib(0, "left_cauchy_green");
int want_lcg_vals = haspointattrib(0, "left_cauchy_green_eigenvalues");
int want_lcg_vecs = haspointattrib(0, "left_cauchy_green_eigenvectors");
int want_left_stretch = haspointattrib(0, "left_stretch");
int want_almansi = haspointattrib(0, "almansi_strain");
int want_almansi_vals = haspointattrib(0, "almansi_strain_eigenvalues");
int need_B_eigen = want_lcg_vals || want_lcg_vecs || want_left_stretch
    || want_almansi_vals;
int need_B = need_B_eigen || want_lcg || want_almansi;

int want_inf = haspointattrib(0, "infinitesimal_strain");
int want_inf_vals = haspointattrib(0, "infinitesimal_strain_eigenvalues");
int want_inf_vecs = haspointattrib(0, "infinitesimal_strain_eigenvectors");
int need_inf_eigen = want_inf_vals || want_inf_vecs;
int need_inf = need_inf_eigen || want_inf;

int want_cauchy_mat = haspointattrib(0, "cauchy_mat");
int want_cauchy_vals = haspointattrib(0, "cauchy_eigenvalues");
int want_cauchy_vecs = haspointattrib(0, "cauchy_eigenvectors");
int want_cauchy_trace = haspointattrib(0, "cauchy_trace");
int want_hydro = haspointattrib(0, "hydrostatic_stress");
int want_dev = haspointattrib(0, "deviatoric_stress");
int want_J2 = haspointattrib(0, "stress_J2");
int want_J3 = haspointattrib(0, "stress_J3");
int want_vm_derived = haspointattrib(0, "von_mises_derived");
int want_max_shear = haspointattrib(0, "max_shear_stress");
int want_triax = haspointattrib(0, "stress_triaxiality");
int fill_vm = int(detail(0, "__derived_fill_von_mises", 0));
int want_pk1 = haspointattrib(0, "pk1");
int want_pk2 = haspointattrib(0, "pk2");
int want_pk2_vals = haspointattrib(0, "pk2_eigenvalues");
int want_pk2_vecs = haspointattrib(0, "pk2_eigenvectors");
int need_cauchy_eigen = want_cauchy_vals || want_cauchy_vecs || want_max_shear;
int need_pk_eigen = want_pk2_vals || want_pk2_vecs;
int need_pk = want_pk1 || want_pk2 || need_pk_eigen;
int need_dev = want_dev || want_J2 || want_J3 || want_vm_derived
    || want_triax || fill_vm;
int need_stress = want_cauchy_mat || need_cauchy_eigen || want_cauchy_trace
    || want_hydro || need_dev || need_pk;

if (!(want_F_mat || want_J || need_C || need_B || need_inf || need_stress))
    return;

// PolyFEM flattens tensors column-major: the exported X_i arrays are
// the COLUMNS of the tensor, while VEX set(a, b, c) fills rows --
// transpose to recover the true matrix (verified against PolyFEM's own
// pk1_stess export, which is asymmetric and exposes the convention).
vector F1 = point(0, "F_1", @ptnum);
vector F2 = point(0, "F_2", @ptnum);
vector F3 = point(0, "F_3", @ptnum);
matrix3 F = transpose(set(F1, F2, F3));
if (want_F_mat)
    setpointattrib(0, "F_mat", @ptnum, F);
float J_val = determinant(F);
if (want_J)
    setpointattrib(0, "J", @ptnum, J_val);

matrix3 I = ident();

if (need_C) {
    // Right Cauchy-Green deformation tensor C = F^T F (Lagrangian) and its
    // Lagrangian principal directions, reused for the stretch/strain measures
    // that share those axes (right stretch U, Green-Lagrange, Hencky).
    matrix3 C = transpose(F) * F;
    if (want_rcg)
        setpointattrib(0, "right_cauchy_green", @ptnum, C);
    if (want_green)
        setpointattrib(0, "green_lagrange_strain", @ptnum, 0.5 * (C - I));
    if (need_C_eigen) {
        vector rcg_vals = {0, 0, 0};
        matrix3 rcg_vecs = 0;
        sorted_eigen(C, rcg_vals, rcg_vecs);
        vector lam = sqrt(max(rcg_vals, {0, 0, 0}));   // principal stretches
        if (want_stretches)
            setpointattrib(0, "principal_stretches", @ptnum, lam);
        if (want_rcg_vals)
            setpointattrib(0, "right_cauchy_green_eigenvalues", @ptnum,
                           rcg_vals);
        if (want_rcg_vecs)
            setpointattrib(0, "right_cauchy_green_eigenvectors", @ptnum,
                           rcg_vecs);
        if (want_right_stretch)
            setpointattrib(0, "right_stretch", @ptnum,
                           rebuild_symmetric(lam, rcg_vecs));
        // E = 0.5 * (C - I) shares C's eigenvectors and shifts/scales its
        // eigenvalues directly -- no second eigendecomposition needed.
        if (want_green_vals)
            setpointattrib(0, "green_lagrange_eigenvalues", @ptnum,
                           0.5 * (rcg_vals - {1, 1, 1}));
        if (want_green_vecs)
            setpointattrib(0, "green_lagrange_eigenvectors", @ptnum, rcg_vecs);
        // Material logarithmic (Hencky) strain = 0.5 ln C = sum ln(lambda) n (x) n,
        // defined only where every stretch is positive (J != 0).
        if (want_hencky || want_hencky_vals) {
            if (abs(J_val) > 1e-12) {
                vector ln_lam = log(max(lam, 1e-20));
                if (want_hencky)
                    setpointattrib(0, "hencky_strain", @ptnum,
                                   rebuild_symmetric(ln_lam, rcg_vecs));
                if (want_hencky_vals)
                    setpointattrib(0, "hencky_strain_eigenvalues", @ptnum,
                                   ln_lam);
            }
        }
    }
}

if (need_B) {
    // Left Cauchy-Green deformation tensor B = F F^T (Eulerian) and its
    // Eulerian principal directions, shared with the left stretch V and the
    // Almansi strain.
    matrix3 B = F * transpose(F);
    if (want_lcg)
        setpointattrib(0, "left_cauchy_green", @ptnum, B);
    if (need_B_eigen) {
        vector lcg_vals = {0, 0, 0};
        matrix3 lcg_vecs = 0;
        sorted_eigen(B, lcg_vals, lcg_vecs);
        if (want_lcg_vals)
            setpointattrib(0, "left_cauchy_green_eigenvalues", @ptnum, lcg_vals);
        if (want_lcg_vecs)
            setpointattrib(0, "left_cauchy_green_eigenvectors", @ptnum,
                           lcg_vecs);
        if (want_left_stretch)
            setpointattrib(0, "left_stretch", @ptnum,
                           rebuild_symmetric(sqrt(max(lcg_vals, {0, 0, 0})),
                                             lcg_vecs));
        if (want_almansi_vals && abs(J_val) > 1e-12) {
            setpointattrib(0, "almansi_strain_eigenvalues", @ptnum, set(
                0.5 * (1.0 - 1.0 / max(lcg_vals.x, 1e-20)),
                0.5 * (1.0 - 1.0 / max(lcg_vals.y, 1e-20)),
                0.5 * (1.0 - 1.0 / max(lcg_vals.z, 1e-20))));
        }
    }
    // Almansi (Eulerian) strain e = 0.5 (I - B^-1); defined only where B is
    // invertible (J != 0).
    if (want_almansi && abs(J_val) > 1e-12)
        setpointattrib(0, "almansi_strain", @ptnum, 0.5 * (I - invert(B)));
}

if (need_inf) {
    matrix3 infinitesimal = 0.5 * (F + transpose(F)) - I;
    if (want_inf)
        setpointattrib(0, "infinitesimal_strain", @ptnum, infinitesimal);
    if (need_inf_eigen) {
        vector inf_vals = {0, 0, 0};
        matrix3 inf_vecs = 0;
        sorted_eigen(infinitesimal, inf_vals, inf_vecs);
        if (want_inf_vals)
            setpointattrib(0, "infinitesimal_strain_eigenvalues", @ptnum,
                           inf_vals);
        if (want_inf_vecs)
            setpointattrib(0, "infinitesimal_strain_eigenvectors", @ptnum,
                           inf_vecs);
    }
}

if (need_stress) {
    // Cauchy stress from the exported field, or reconstructed from the 1st
    // Piola-Kirchhoff stress (sigma = (1/J) P F^T) when only that was written,
    // so PolyFEM can export just F + pk1 and everything else is derived.
    int have_cauchy = haspointattrib(0, "cauchy_stress_1")
        && haspointattrib(0, "cauchy_stress_2")
        && haspointattrib(0, "cauchy_stress_3");
    int have_pk1 = haspointattrib(0, "pk1_stress_1")
        && haspointattrib(0, "pk1_stress_2")
        && haspointattrib(0, "pk1_stress_3");
    if (have_cauchy || (have_pk1 && abs(J_val) > 1e-12)) {
        matrix3 cauchy;
        if (have_cauchy) {
            // symmetric, so the column-major transpose is a no-op; kept for
            // consistency with the F / pk1 assembly convention above
            vector s1 = point(0, "cauchy_stress_1", @ptnum);
            vector s2 = point(0, "cauchy_stress_2", @ptnum);
            vector s3 = point(0, "cauchy_stress_3", @ptnum);
            cauchy = transpose(set(s1, s2, s3));
        } else {
            vector p1 = point(0, "pk1_stress_1", @ptnum);
            vector p2 = point(0, "pk1_stress_2", @ptnum);
            vector p3 = point(0, "pk1_stress_3", @ptnum);
            matrix3 P = transpose(set(p1, p2, p3));
            cauchy = (1.0 / J_val) * P * transpose(F);
        }
        if (want_cauchy_mat)
            setpointattrib(0, "cauchy_mat", @ptnum, cauchy);
        if (need_cauchy_eigen) {
            vector c_vals = {0, 0, 0};
            matrix3 c_vecs = 0;
            sorted_eigen(cauchy, c_vals, c_vecs);
            if (want_cauchy_vals)
                setpointattrib(0, "cauchy_eigenvalues", @ptnum, c_vals);
            if (want_cauchy_vecs)
                setpointattrib(0, "cauchy_eigenvectors", @ptnum, c_vecs);
            if (want_max_shear)
                setpointattrib(0, "max_shear_stress", @ptnum,
                               0.5 * (c_vals.x - c_vals.z));
        }

        if (want_cauchy_trace || want_hydro || need_dev) {
            float tr = matrix_trace(cauchy);
            float hydro = tr / 3.0;
            if (want_cauchy_trace)
                setpointattrib(0, "cauchy_trace", @ptnum, tr);
            if (want_hydro)
                setpointattrib(0, "hydrostatic_stress", @ptnum, hydro);
            if (need_dev) {
                matrix3 dev = cauchy - hydro * I;
                if (want_dev)
                    setpointattrib(0, "deviatoric_stress", @ptnum, dev);
                float j2 = 0.5 * matrix_norm_squared(dev);
                if (want_J2)
                    setpointattrib(0, "stress_J2", @ptnum, j2);
                if (want_J3)
                    setpointattrib(0, "stress_J3", @ptnum, determinant(dev));
                if (want_vm_derived || want_triax || fill_vm) {
                    float vm = sqrt(max(0.0, 3.0 * j2));
                    if (want_vm_derived)
                        setpointattrib(0, "von_mises_derived", @ptnum, vm);
                    if (want_triax)
                        setpointattrib(0, "stress_triaxiality", @ptnum,
                                       vm > 1e-12 ? hydro / vm : 0.0);
                    if (fill_vm)
                        setpointattrib(0, "von_mises", @ptnum, vm);
                }
            }
        }

        if (need_pk) {
            matrix3 pk2_mat = 0;
            if (abs(J_val) > 1e-12) {
                matrix3 Finv = invert(F);
                matrix3 FinvT = transpose(Finv);
                if (want_pk1)
                    setpointattrib(0, "pk1", @ptnum, J_val * cauchy * FinvT);
                if (want_pk2 || need_pk_eigen) {
                    pk2_mat = J_val * Finv * cauchy * FinvT;
                    if (want_pk2)
                        setpointattrib(0, "pk2", @ptnum, pk2_mat);
                }
            }
            if (need_pk_eigen) {
                vector pk2_vals = {0, 0, 0};
                matrix3 pk2_vecs = 0;
                sorted_eigen(pk2_mat, pk2_vals, pk2_vecs);
                if (want_pk2_vals)
                    setpointattrib(0, "pk2_eigenvalues", @ptnum, pk2_vals);
                if (want_pk2_vecs)
                    setpointattrib(0, "pk2_eigenvectors", @ptnum, pk2_vecs);
            }
        }
    }
}
"""

# The colour stage is split so an optional nodal-averaging pass can sit
# between the field reduction and the ramp mapping:
#   color_reduce  -> for_color (the displayed scalar)
#   field_smoothing (optional, switched) -> averages for_color per vertex
#   color_map     -> for_color to Cd (ramp / log / invalid / missing)
COLOR_REDUCE_VEX = """\
float matrix_trace(matrix3 m) {
    return getcomp(m, 0, 0) + getcomp(m, 1, 1) + getcomp(m, 2, 2);
}

float matrix_norm_squared(matrix3 m) {
    float result = 0.0;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) {
            float value = getcomp(m, row, column);
            result += value * value;
        }
    return result;
}

void sorted_eigen(matrix3 m; export vector vals; export matrix3 vecs) {
    vector raw_vals = {0, 0, 0};
    matrix3 raw_vecs = diagonalizesymmetric(m, raw_vals);
    float unsorted[] = array(raw_vals.x, raw_vals.y, raw_vals.z);
    int order[] = argsort(unsorted);
    for (int j = 0; j < 3; j++) {
        int src = order[2-j];
        setcomp(vals, getcomp(raw_vals, src), j);
        // diagonalizesymmetric returns eigenvectors as ROWS; store eigenvector
        // 'src' (row src of raw_vecs) as COLUMN j so that
        // vecs * diag(vals) * transpose(vecs) reconstructs the tensor.
        for (int r = 0; r < 3; r++)
            setcomp(vecs, getcomp(raw_vecs, src, r), r, j);
    }
}
// When smoothing is on, prefer PolyFEM's continuous nodal-averaged field
// (e.g. von_mises -> von_mises_avg) when one exists; the generic averaging
// pass downstream handles fields that have no _avg variant.
string base = chs("../color_attrib");
string field = chi("../smooth_field") && haspointattrib(0, base + "_avg")
    ? base + "_avg" : base;
string attr = chi("../reference_enable")
    && haspointattrib(0, "reference_comparison")
    ? "reference_comparison" : field;
string reduction = chs("../color_reduction");
int size = pointattribsize(0, attr);
int present = haspointattrib(0, attr) && size > 0;
float value = 0.0;
if (present) {
    if (size == 1) {
        value = point(0, attr, @ptnum);
    } else if (size >= 9) {
        matrix3 m = point(0, attr, @ptnum);
        if (reduction == "trace")
            value = matrix_trace(m);
        else if (reduction == "determinant")
            value = determinant(m);
        else if (reduction == "principal_max"
                 || reduction == "principal_middle"
                 || reduction == "principal_min") {
            // Principal values of the symmetric part, as the numpy mirror
            // (_reduce_array: Auto Range, export) computes them:
            // diagonalizesymmetric reads only part of a non-symmetric F or
            // P (review 2026-11-14, item 6).
            vector evals = {0, 0, 0};
            matrix3 evecs = 0;
            sorted_eigen(0.5 * (m + transpose(m)), evals, evecs);
            int component = reduction == "principal_max" ? 0
                : reduction == "principal_middle" ? 1 : 2;
            value = getcomp(evals, component);
        } else if (reduction == "x" || reduction == "y" || reduction == "z") {
            int component = reduction == "x" ? 0 : reduction == "y" ? 1 : 2;
            value = getcomp(m, component, component);
        } else if (reduction == "xy") {
            value = getcomp(m, 0, 1);
        } else if (reduction == "yz") {
            value = getcomp(m, 1, 2);
        } else if (reduction == "xz") {
            value = getcomp(m, 0, 2);
        }
        else
            value = sqrt(matrix_norm_squared(m));
    } else {
        vector v = point(0, attr, @ptnum);
        if (reduction == "x" || reduction == "principal_max")
            value = v.x;
        else if (reduction == "y" || reduction == "principal_middle")
            value = v.y;
        else if (reduction == "z" || reduction == "principal_min")
            value = v.z;
        else
            value = length(v);
    }
}
f@for_color = value;
if (@ptnum == 0)
    setdetailattrib(0, "color_field_missing", present ? 0 : 1, "set");
"""

COLOR_MAP_VEX = """\
if (detail(0, "color_field_missing", 0) == 1) {
    v@Cd = chv("../missing_color");
} else {
    float value = f@for_color;
    float low = ch("../color_min");
    float high = ch("../color_max");
    int valid = isfinite(value);
    if (chs("../color_scale") == "log10") {
        valid = valid && value > 0 && low > 0 && high > 0;
        if (valid) {
            value = log10(value);
            low = log10(low);
            high = log10(high);
        }
    }
    if (valid) {
        float t = abs(high - low) > 1e-20
            ? fit(value, low, high, 0, 1) : 0.5;
        v@Cd = chramp("../color_ramp", clamp(t, 0, 1));
    } else {
        v@Cd = chv("../invalid_color");
        setdetailattrib(0, "invalid_color_value_count", 1, "add");
    }
}
"""

# Show/hide PolyFEM bodies: keep only prims whose body_ids is in the selected
# set. Empty selection (or a block without body_ids) shows everything.
BODY_FILTER_VEX = """\
if (chs("../visible_bodies") == "") return;
if (!haspointattrib(0, "body_ids")) return;
string tokens[] = split(re_replace(",", " ", chs("../visible_bodies")));
if (len(tokens) == 0) return;
float body_value = point(0, "body_ids", primpoint(0, @primnum, 0));
int body = (int)rint(body_value);
foreach (string token; tokens)
    if (atoi(token) == body) return;
removeprim(0, @primnum, 1);
"""

BODY_FILTER_POINTS_VEX = """\
if (nprimitives(0) > 0) return;
if (chs("../visible_bodies") == "") return;
if (!haspointattrib(0, "body_ids")) return;
string tokens[] = split(re_replace(",", " ", chs("../visible_bodies")));
if (len(tokens) == 0) return;
float body_value = point(0, "body_ids", @ptnum);
int body = (int)rint(body_value);
foreach (string token; tokens)
    if (atoi(token) == body) return;
removepoint(0, @ptnum);
"""

FIBER_VEX = """\
// One line per physical node per family. PolyFEM exports the unchanged
// simulation/world reference direction a0; geometry[].transformation never
// rotates it. Fibers are LINE fields (a0 and -a0 are the same fiber), so
// colour by |direction| and never by sign.
if (chi("../show_fibers") == 0) return;
if (haspointattrib(0, "coincident_rep")
    && float(point(0, "coincident_rep", @ptnum)) < 0.5)
    return;
int sample_id = @ptnum;
if (haspointattrib(0, "coincident_id")) {
    float cid = point(0, "coincident_id", @ptnum);
    sample_id = (int)rint(cid);
}
if (sample_id % max(1, chi("../fiber_stride")) != 0) return;

string wanted = chs("../fiber_family");
string names[] = split(chs("../fiber_attribs"));
string frame = chs("../fiber_frame");
float scale = chf("../fiber_scale");
int mode = chi("../fiber_color_mode");

// PolyFEM flattens tensors column-major: F_1..F_3 are the COLUMNS, while
// set(a, b, c) fills rows -- transpose to recover the true matrix (same
// convention as the derived-fields wrangle).
matrix3 F = ident();
int has_f = 0;
if (haspointattrib(0, "F_mat")) {
    F = matrix3(point(0, "F_mat", @ptnum));
    has_f = 1;
} else if (haspointattrib(0, "F_1") && haspointattrib(0, "F_2") && haspointattrib(0, "F_3")) {
    F = transpose(set(point(0, "F_1", @ptnum), point(0, "F_2", @ptnum),
                      point(0, "F_3", @ptnum)));
    has_f = 1;
}

int family = 0;
foreach (string name; names) {
    family++;
    if (wanted != "all" && wanted != name) continue;
    if (!haspointattrib(0, name)) continue;
    vector a0 = point(0, name, @ptnum);
    if (length(a0) < 1e-12) continue;
    a0 = normalize(a0);

    // Fiber kinematics used by PolyFEM's constitutive invariants:
    //   current direction = normalize(F a0)
    //   I4 = a0^T C a0 = |F a0|^2
    // HGOFiber/ActiveFiber use isochoric I4bar; HGODispersion uses full I4.
    vector current = a0;
    float stretch = 1.0;
    float isochoric_stretch = 1.0;
    float direction_change = 0.0;
    if (has_f) {
        // F * a0 in column convention; VEX multiplies row-vectors, so the
        // matching product is a0 * transpose(F).
        vector pushed = a0 * transpose(F);
        stretch = length(pushed);
        if (stretch > 1e-12) {
            current = pushed / stretch;
            float J = max(abs(determinant(F)), 1e-12);
            isochoric_stretch = stretch / pow(J, 1.0 / 3.0);
            // Fibers are unoriented lines: compare a0 with +/-current.
            direction_change = degrees(acos(
                clamp(abs(dot(a0, current)), 0.0, 1.0)));
        }
    }

    vector dirs[] = {};
    float dims[] = {};
    string frames[] = {};
    if (frame == "reference" || frame == "both") {
        append(dirs, a0);
        append(dims, frame == "both" ? 0.45 : 1.0);
        append(frames, "reference_a0");
    }
    if ((frame == "deformed" || frame == "both") && has_f) {
        if (stretch > 1e-12) {
            append(dirs, current);
            append(dims, 1.0);
            append(frames, "current_normalized_Fa0");
        }
    }

    int index = 0;
    foreach (vector d; dirs) {
        vector colour;
        if (mode == 0)
            colour = abs(d);
        else if (mode == 1)
            colour = set(float(family % 3 == 1), float(family % 3 == 2),
                         float(family % 3 == 0));
        else if (mode == 3)
            // Blue contraction -> white unchanged -> red extension.
            colour = stretch < 1.0
                ? lerp(set(0.0, 0.2, 1.0), set(1.0, 1.0, 1.0),
                       clamp(stretch, 0.0, 1.0))
                : lerp(set(1.0, 1.0, 1.0), set(1.0, 0.1, 0.0),
                       clamp(stretch - 1.0, 0.0, 1.0));
        else if (mode == 4)
            // Blue=no reorientation, red=90 degrees (line-sign invariant).
            colour = lerp(set(0.0, 0.2, 1.0), set(1.0, 0.0, 0.0),
                          clamp(direction_change / 90.0, 0.0, 1.0));
        else
            colour = chv("../fiber_color");
        colour *= dims[index];
        int a = addpoint(0, @P - 0.5 * scale * d);
        int b = addpoint(0, @P + 0.5 * scale * d);
        int line = addprim(0, "polyline", a, b);
        setpointattrib(0, "Cd", a, colour, "set");
        setpointattrib(0, "Cd", b, colour, "set");
        setprimattrib(0, "fiber_family", line, name, "set");
        setprimattrib(0, "fiber_drawn_frame", line, frames[index], "set");
        setprimattrib(0, "fiber_reference_direction", line, a0, "set");
        setprimattrib(0, "fiber_current_direction", line, current, "set");
        setprimattrib(0, "fiber_stretch", line, stretch, "set");
        setprimattrib(0, "fiber_isochoric_stretch", line,
                      isochoric_stretch, "set");
        setprimattrib(0, "fiber_I4", line, stretch * stretch, "set");
        setprimattrib(0, "fiber_I4bar", line,
                      isochoric_stretch * isochoric_stretch, "set");
        setprimattrib(0, "fiber_direction_change_degrees", line,
                      direction_change, "set");
        setprimgroup(0, "readpvd_fibers", line, 1, "set");
        index++;
    }
}
"""

GLYPH_VEX = """\
// With smoothing on, every physical node has one recovered tensor (the same
// on all its duplicated copies); draw a single glyph per node, not one per
// duplicate, so coincident glyphs do not stack up.
if (chi("../smooth_field") && haspointattrib(0, "coincident_rep")
    && float(point(0, "coincident_rep", @ptnum)) < 0.5)
    return;
int sample_id = @ptnum;
if (chi("../smooth_field") && haspointattrib(0, "coincident_id")) {
    float cid = point(0, "coincident_id", @ptnum);
    sample_id = (int)rint(cid);
}
if (sample_id % max(1, chi("../glyph_stride")) != 0) return;
string tensor = chs("../glyph_tensor");
string vec_name =
      tensor == "right_cauchy_green" ? "right_cauchy_green_eigenvectors"
    : tensor == "left_cauchy_green" ? "left_cauchy_green_eigenvectors"
    : tensor == "green_lagrange" ? "green_lagrange_eigenvectors"
    : tensor == "pk2" ? "pk2_eigenvectors"
    : tensor == "cauchy" ? "cauchy_eigenvectors" : "";
string val_name =
      tensor == "right_cauchy_green" ? "right_cauchy_green_eigenvalues"
    : tensor == "left_cauchy_green" ? "left_cauchy_green_eigenvalues"
    : tensor == "green_lagrange" ? "green_lagrange_eigenvalues"
    : tensor == "pk2" ? "pk2_eigenvalues"
    : tensor == "cauchy" ? "cauchy_eigenvalues" : "";
if (!haspointattrib(0, vec_name) || !haspointattrib(0, val_name)) return;
float scale = ch("../tensor_scale");
matrix3 vecs = point(0, vec_name, @ptnum);
vector vals = point(0, val_name, @ptnum);
for (int j = 0; j < 3; j++) {
    if ((j == 0 && !chi("../glyph_direction_first"))
        || (j == 1 && !chi("../glyph_direction_second"))
        || (j == 2 && !chi("../glyph_direction_third")))
        continue;
    vector dir = normalize(set(getcomp(vecs, 0, j), getcomp(vecs, 1, j),
                               getcomp(vecs, 2, j)));
    float direction_scale = j == 0 ? chf("../glyph_scale_first")
        : j == 1 ? chf("../glyph_scale_second")
        : chf("../glyph_scale_third");
    float mag = chs("../glyph_length_mode") == "normalized"
        ? scale * direction_scale
        : abs(vals[j]) * scale * direction_scale;
    if (chs("../glyph_length_mode") == "clamped")
        mag = min(mag, ch("../glyph_max_length"));
    // Color the glyph POINTS (not prims): writing a primitive Cd here would
    // promote Cd to a prim attribute, and a prim Cd overrides the object's
    // point-Cd color mapping in the viewport (turning the result a flat
    // color). Point Cd keeps the field coloring intact alongside the glyphs.
    vector color = chi("../glyph_sign_color")
        && (tensor == "pk2" || tensor == "cauchy")
        ? (vals[j] >= 0 ? {1.0, 0.2, 0.2} : {0.2, 0.4, 1.0})
        : chv("../glyph_color");
    vector pos_a = v@P - dir * mag * 0.5;
    vector pos_b = v@P + dir * mag * 0.5;
    int a = addpoint(0, pos_a);
    int b = addpoint(0, pos_b);
    setpointattrib(0, "Cd", a, color, "set");
    setpointattrib(0, "Cd", b, color, "set");
    int prim = addprim(0, "polyline", a, b);
    setprimgroup(0, "readpvd_glyphs", prim, 1, "set");
    if (chi("../glyph_arrowheads") && mag > 1e-12) {
        vector helper = abs(dir.z) < 0.9 ? {0, 0, 1} : {0, 1, 0};
        float arrow = ch("../glyph_arrow_size");
        vector side = normalize(cross(dir, helper)) * mag * arrow;
        vector step = dir * mag * arrow * 2;
        // Arrowheads at both ends (eigenvectors are unoriented: +dir == -dir).
        vector back_b = pos_b - step;
        int cb = addpoint(0, back_b + side);
        int db = addpoint(0, back_b - side);
        setpointattrib(0, "Cd", cb, color, "set");
        setpointattrib(0, "Cd", db, color, "set");
        setprimgroup(0, "readpvd_glyphs", addprim(0, "polyline", b, cb), 1, "set");
        setprimgroup(0, "readpvd_glyphs", addprim(0, "polyline", b, db), 1, "set");
        vector back_a = pos_a + step;
        int ca = addpoint(0, back_a + side);
        int da = addpoint(0, back_a - side);
        setpointattrib(0, "Cd", ca, color, "set");
        setpointattrib(0, "Cd", da, color, "set");
        setprimgroup(0, "readpvd_glyphs", addprim(0, "polyline", a, ca), 1, "set");
        setprimgroup(0, "readpvd_glyphs", addprim(0, "polyline", a, da), 1, "set");
    }
}
"""

# Houdini's built-in "Infra-Red" ramp preset (blue -> cyan -> green -> yellow
# -> red) used as the default Displayed Color Ramp -- no black or white ends.
_INFRARED_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_INFRARED_COLORS = ((0.2, 0.0, 1.0), (0.0, 0.85, 1.0), (0.0, 1.0, 0.1),
                    (0.95, 1.0, 0.0), (1.0, 0.0, 0.0))

# Ramp colour defaults can't be expressed on a RampParmTemplate, so a new node
# gets the infra-red palette via an OnCreated handler (runs once at creation,
# leaving ramps in saved scenes untouched).
ON_CREATED_CODE = f"""\
import hou
_limit = kwargs["node"].parm("cache_memory_gb")
if _limit is not None:
    _limit.set(kwargs["node"].hdaModule().available_memory_gb())
_parm = kwargs["node"].parm("color_ramp")
if _parm is not None:
    _parm.set(hou.Ramp(
        [hou.rampBasis.Linear] * {len(_INFRARED_POSITIONS)},
        {list(_INFRARED_POSITIONS)},
        {[list(color) for color in _INFRARED_COLORS]}))
"""


def _force_curve_folder():
    """Analysis > Force Curves (review F4): reaction and contact force per
    sideset or obstacle, from the nodal forces PolyFEM writes."""
    folder = hou.FolderParmTemplate(
        "force_folder", "Force Curves", folder_type=hou.folderType.Simple)
    sets = hou.StringParmTemplate(
        "force_sets", "Force Sets", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which sidesets and obstacles to measure (empty = all of them). "
             "They come from the PolyFEM node that wrote the run: every "
             "sideset, with or without a condition (a sideset without a "
             "condition can be made just to measure a force), and every "
             "obstacle.")
    sets.setItemGeneratorScript("hou.phm().force_set_menu(kwargs)")
    sets.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(sets)
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_compute", "Compute Force Curves",
        script_callback="hou.phm().compute_force_curves(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Read every time step and write force_curves.csv next to the "
             "PVD file: per step and per set, the reaction (the force the "
             "prescribed displacement, or an obstacle's prescribed motion, "
             "applies there: minus the sum of every nodal force PolyFEM "
             "writes), the contact force (contact + friction) and the mean "
             "displacement, as x, y, z and magnitude. Needs Output > Nodal "
             "Forces on the PolyFEM node. Reads every result file, like Auto "
             "Range: All Frames. Adhesion and Rayleigh-damping forces are not "
             "written by PolyFEM and are not included."))
    readout = hou.StringParmTemplate(
        "force_readout", "At This Frame", 1,
        default_expression=(
            "hou.pwd().hdaModule().force_readout_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        tags={"editor": "1", "editorlines": "2-6"},
        help="Read-only: each force set's reaction, contact force and mean "
             "displacement at the step on screen. For a fixed or moved "
             "sideset the reaction is the support force; for an obstacle "
             "the contact force is the force the body applies to it.")
    readout.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    folder.addParmTemplate(readout)
    chart_set = hou.StringParmTemplate(
        "force_chart_set", "Chart", 1, default_value=("1",),
        menu_type=hou.menuType.Normal,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which force set the chart shows.")
    chart_set.setItemGeneratorScript("hou.phm().force_set_menu(kwargs)")
    chart_set.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(chart_set)
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_quantity", "Force", ("reaction", "contact"),
        menu_labels=("Reaction (support force)", "Contact Force"),
        default_value=0,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Reaction: the force the prescribed displacement (or the "
             "obstacle's prescribed motion) applies to the body there. "
             "Contact Force: contact plus friction forces on those nodes."))
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_component", "Component", ("x", "y", "z", "magnitude"),
        menu_labels=("X", "Y", "Z", "Magnitude"), default_value=3,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which component of the force (and of the displacement it is "
             "plotted against) the chart shows."))
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_against", "Against", ("displacement", "time"),
        menu_labels=("Mean Displacement", "Time"), default_value=0,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Plot the force against the set's mean displacement (same "
             "component: a force-displacement curve) or against simulation "
             "time."))
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_show_chart", "Show Chart",
        script_callback="hou.phm().show_force_chart(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        join_with_next=True,
        help="Open a chart of the chosen curve in its own window; a dot "
             "marks the step on screen and follows the playbar. Computes the "
             "curves first if needed."))
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_open_csv", "Open CSV",
        script_callback="hou.phm().open_force_csv(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Open force_curves.csv with the system's default program "
             "(computing it first if needed)."))
    return folder


def _callback(name):
    return {"script_callback": f"hou.phm().{name}(kwargs)",
            "script_callback_language": hou.scriptLanguage.Python}


def _menu_script(template, name):
    template.setItemGeneratorScript(f"hou.phm().{name}(kwargs)")
    template.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    return template


def _read_only_text(name, label, expression, help_text, lines=None):
    template = hou.StringParmTemplate(
        name, label, 1, default_expression=(expression,),
        default_expression_language=(hou.scriptLanguage.Python,),
        tags={"editor": "1", "editorlines": lines} if lines else {},
        help=help_text)
    template.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    return template


def _status_text(name, label, help_text, lines="2-8"):
    template = hou.StringParmTemplate(
        name, label, 1, default_value=("",),
        tags={"editor": "1", "editorlines": lines}, help=help_text)
    template.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    return template


def _export_tab():
    """Export (2026-10-03): the deformed mesh as a .msh, and data over time
    as an Excel workbook for presentation plots."""
    tab = hou.FolderParmTemplate("export_folder_tab", "Export",
                                 folder_type=hou.folderType.Tabs)
    tab.addParmTemplate(hou.StringParmTemplate(
        "export_folder", "Export Folder", 1, default_value=("",),
        string_type=hou.stringParmType.FileReference,
        file_type=hou.fileType.Directory,
        help="Where both exports are written. Empty (the default) = the "
             "run's 'exports' folder, next to its 'input' and 'output' "
             "folders: outside 'output', so running the simulation again "
             "into the same folder leaves the exports alone."))
    tab.addParmTemplate(hou.ButtonParmTemplate(
        "export_open_folder", "Open Folder", **_callback("open_export_folder"),
        help="Show the export folder in Finder / Explorer (it is created if "
             "needed)."))

    # ---- Deformed Mesh ------------------------------------------------------
    mesh = hou.FolderParmTemplate("export_mesh_folder", "Deformed Mesh",
                                  folder_type=hou.folderType.Simple)
    mesh.addParmTemplate(hou.MenuParmTemplate(
        "export_mesh_step_mode", "Step", ("screen", "number"),
        menu_labels=("Step on Screen", "Step Number"), default_value=0,
        help="Which output step's shape to export: the one the playbar shows "
             "now, or a step you type."))
    step = hou.IntParmTemplate(
        "export_mesh_step", "Step Number", 1, default_value=(0,), min=0,
        max=1000,
        help="The output step to export (0 = the start of the run, before "
             "any load). The Simulation Time line on the Main tab shows the "
             "step on screen.")
    step.setConditional(hou.parmCondType.HideWhen,
                        "{ export_mesh_step_mode != number }")
    mesh.addParmTemplate(step)
    mesh.addParmTemplate(_read_only_text(
        "export_mesh_step_status", "Exported Step",
        "hou.pwd().hdaModule().export_mesh_step_text(hou.pwd())",
        "Read-only: the step Export Deformed Mesh writes, and its "
        "simulation time."))
    geometries = hou.StringParmTemplate(
        "export_mesh_geometries", "Geometries", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Which of the run's simulated geometries to export (one .msh "
             "each); empty = all of them. Obstacles are not exported: they "
             "keep their own files.")
    mesh.addParmTemplate(_menu_script(geometries, "export_geometry_menu"))
    mesh.addParmTemplate(hou.MenuParmTemplate(
        "export_mesh_coordinates", "Coordinates", ("mesh", "simulation"),
        menu_labels=("Mesh File (keep the geometry's Transform)",
                     "Simulation (world)"), default_value=0,
        help="Mesh File: the coordinates and units of the original mesh "
             "file; give it the same Transform on the PolyFEM node (as when "
             "you pick it on the geometry that made the run) and it sits "
             "where the deformed body was. Simulation: the positions in the "
             "simulation, with the Transform already applied (the file name "
             "ends in _simulation); load it with no Transform."))
    mesh.addParmTemplate(hou.ButtonParmTemplate(
        "export_mesh", "Export Deformed Mesh", **_callback("export_mesh_button"),
        help="Write each geometry's deformed shape at the chosen step as a "
             ".msh file in the export folder (<mesh name>_step<N>.msh): the "
             "run's own mesh file with only its node positions moved, so "
             "node numbers, elements, Gmsh physical groups and names (the "
             "subdomains) stay the same, and the PolyFEM node keeps a "
             "geometry's materials and picked sidesets when you load the "
             "file on it. Linear elements: for a P2 or higher run the "
             "corners are moved and the curvature of the edges, which the "
             "report measures, is not kept. Inverted elements are refused. "
             "The exported shape is a new rest shape: a simulation that "
             "starts from it starts without stress, strain or velocity."))
    mesh.addParmTemplate(_status_text(
        "export_mesh_report", "Report",
        "Read-only: what the last Export Deformed Mesh wrote (files, nodes "
        "moved, the range of element volume change, the curvature a "
        "higher-order run loses) or why it wrote nothing."))
    tab.addParmTemplate(mesh)

    # ---- Data Over Time --------------------------------------------------------
    data = hou.FolderParmTemplate("export_data_folder", "Data Over Time",
                                  folder_type=hou.folderType.Simple)
    regions = hou.FolderParmTemplate(
        "export_regions", "Regions", folder_type=hou.folderType.MultiparmBlock,
        default_value=1)
    kind_condition = "{{ export_region_kind# != {} }}"
    regions.addParmTemplate(hou.StringParmTemplate(
        "export_region_name#", "Name", 1, default_value=("",),
        help="The region's name in the column headers (e.g. 'Cartilage'); "
             "empty = a name from its kind. Each region (Regions above sets "
             "how many) gives the spreadsheet its own columns; with no region "
             "the whole model is measured."))
    kind = hou.MenuParmTemplate(
        "export_region_kind#", "Region",
        ("all", "bodies", "sideset", "box", "sphere", "plane", "element",
         "node"),
        menu_labels=("Whole Model", "Bodies / Subdomains", "Sideset", "Box",
                     "Sphere", "Half Space (side of a plane)", "One Element",
                     "One Node"),
        default_value=0,
        help="Whole Model: every element and node. Bodies: chosen bodies "
             "(the PolyFEM node's subdomains, named after their Gmsh groups). "
             "Sideset: a sideset of the PolyFEM node (its nodes, and the "
             "elements with a face on it). Box, Sphere, Half Space: the "
             "elements whose centre and the nodes that lie inside, in the "
             "undeformed shape (simulation coordinates). One Element / One "
             "Node: by number (Set From Probe fills it in from a point "
             "clicked with the Probe tab's probe).")
    regions.addParmTemplate(kind)
    bodies = hou.StringParmTemplate(
        "export_region_bodies#", "Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="The bodies of the region (needs Body IDs in the PolyFEM node's "
             "output, on by default).")
    bodies.setConditional(hou.parmCondType.HideWhen,
                          kind_condition.format("bodies"))
    regions.addParmTemplate(_menu_script(bodies, "export_body_menu"))
    sideset = hou.StringParmTemplate(
        "export_region_sideset#", "Sideset", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        help="One of the sidesets the PolyFEM node recorded for the run "
             "(input/hda_scene.json, written since 2026-10-02). A sideset "
             "without a condition can be made just to measure there.")
    sideset.setConditional(hou.parmCondType.HideWhen,
                           kind_condition.format("sideset"))
    regions.addParmTemplate(_menu_script(sideset, "export_sideset_menu"))
    for name, label, default, what, help_text in (
            ("box_min", "Box Minimum", (0.0, 0.0, 0.0), "box",
             "One corner of the box (simulation coordinates, undeformed)."),
            ("box_max", "Box Maximum", (1.0, 1.0, 1.0), "box",
             "The opposite corner of the box."),
            ("center", "Center", (0.0, 0.0, 0.0), "sphere",
             "The sphere's centre (simulation coordinates, undeformed); Set "
             "From Probe puts it on the probed point."),
            ("point", "Point on Plane", (0.0, 0.0, 0.0), "plane",
             "A point of the plane (simulation coordinates, undeformed)."),
            ("normal", "Normal", (0.0, 0.0, 1.0), "plane",
             "The region is the side of the plane this normal points to.")):
        template = hou.FloatParmTemplate(
            f"export_region_{name}#", label, 3, default_value=default,
            help=help_text)
        template.setConditional(hou.parmCondType.HideWhen,
                                kind_condition.format(what))
        regions.addParmTemplate(template)
    radius = hou.FloatParmTemplate(
        "export_region_radius#", "Radius", 1, default_value=(0.1,), min=0.0,
        max=10.0, help="The sphere's radius (simulation units).")
    radius.setConditional(hou.parmCondType.HideWhen,
                          kind_condition.format("sphere"))
    regions.addParmTemplate(radius)
    one = ("{ export_region_kind# != element export_region_kind# != node }")
    geometry = hou.IntParmTemplate(
        "export_region_geometry#", "Geometry", 1, default_value=(1,), min=1,
        max=20,
        help="The geometry (as numbered on the PolyFEM node) of the element "
             "or node.")
    geometry.setConditional(hou.parmCondType.HideWhen, one)
    regions.addParmTemplate(geometry)
    number = hou.IntParmTemplate(
        "export_region_number#", "Number", 1, default_value=(1,), min=1,
        max=100000,
        help="Element: its number in the geometry's .msh file (1 = the "
             "file's first volume element). Node: its node tag in the .msh "
             "file. Set From Probe fills it in from the probed point.")
    number.setConditional(hou.parmCondType.HideWhen, one)
    regions.addParmTemplate(number)
    probe = hou.ButtonParmTemplate(
        "export_region_probe#", "Set From Probe",
        **_callback("export_region_from_probe"),
        help="Use the point last clicked with the probe (Probe tab): its "
             "node or element for One Node / One Element, its undeformed "
             "position as the centre of a Sphere, or as the point of a "
             "Half Space.")
    probe.setConditional(
        hou.parmCondType.HideWhen,
        "{ export_region_kind# == all } { export_region_kind# == bodies } "
        "{ export_region_kind# == sideset } { export_region_kind# == box }")
    regions.addParmTemplate(probe)
    only = hou.StringParmTemplate(
        "export_region_only#", "Only in Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Keep only the elements and nodes of these bodies (empty = any "
             "body), e.g. the part of a box that is cartilage.")
    only.setConditional(
        hou.parmCondType.HideWhen,
        "{ export_region_kind# == bodies } { export_region_kind# == element }"
        " { export_region_kind# == node }")
    regions.addParmTemplate(_menu_script(only, "export_body_menu"))
    data.addParmTemplate(regions)

    quantities = hou.FolderParmTemplate(
        "export_quantities", "Quantities",
        folder_type=hou.folderType.MultiparmBlock, default_value=1)
    field = hou.StringParmTemplate(
        "export_q_field#", "Field", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        **_callback("export_quantity_changed"),
        help="What to measure in every region (Quantities above sets how "
             "many; each quantity's statistics become columns, one per "
             "region). Any field Read PVD can color by (exported or derived: "
             "displacement, von Mises, principal stresses, strains, "
             "stretches, J ...), Region Volume, or a fiber stretch on fiber "
             "runs. Same as the Display Tab follows the Display tab's Color "
             "Field and Field Value To Display at export time.")
    quantities.addParmTemplate(_menu_script(field, "export_field_menu"))
    value = hou.StringParmTemplate(
        "export_q_value#", "Value", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        help="One number per element or node: a vector's magnitude or a "
             "component; a tensor's principal value, an entry (xx ... xz), "
             "Frobenius norm, trace or determinant.")
    quantities.addParmTemplate(_menu_script(value, "export_value_menu"))
    quantities.addParmTemplate(hou.MenuParmTemplate(
        "export_q_basis#", "Per", ("auto", "elements", "nodes"),
        menu_labels=("Automatic", "Element", "Node"), default_value=0,
        help="What one value belongs to. Element: the element's value (the "
             "mean of its nodes in PolyFEM's output, exact for linear "
             "elements); Node: the node's value (the mean of its copies, one "
             "per element, as Smooth Field shows). Automatic: nodes for "
             "displacement, velocity, acceleration and nodal forces, "
             "elements otherwise (stress, strain)."))
    for name, label, default, help_text in (
            ("max", "Max", True, "Largest value in the region."),
            ("min", "Min", False, "Smallest value in the region."),
            ("mean", "Mean (volume-weighted)", True,
             "Volume-weighted mean: sum of value x volume / sum of volumes, "
             "so small elements do not count more than large ones."),
            ("amean", "Arithmetic Mean", False,
             "Plain mean over the elements (or nodes), whatever their size."),
            ("integral", "Integral", False,
             "Sum of value x volume over the region (the volume integral)."),
            ("sum", "Sum", False,
             "Plain sum over the elements or nodes; over the nodes of a "
             "region, a nodal force's sum is the resultant force."),
            ("rms", "RMS", False, "Root mean square."),
            ("std", "Standard Deviation", False,
             "Standard deviation over the elements (or nodes)."),
            ("count", "Count", False,
             "How many elements (or nodes) had a value at the step.")):
        quantities.addParmTemplate(hou.ToggleParmTemplate(
            f"export_q_{name}#", label, default_value=default,
            join_with_next=name not in ("amean", "count"), help=help_text))
    quantities.addParmTemplate(hou.StringParmTemplate(
        "export_q_percentiles#", "Percentiles", 1, default_value=("",),
        help="Percentiles to add, e.g. '95 99' (a robust peak: the largest "
             "element values at corners and contact edges depend on the "
             "mesh)."))
    quantities.addParmTemplate(hou.ToggleParmTemplate(
        "export_q_where#", "Where the Max Is", default_value=False,
        help="Add columns with the element or node that holds the max at "
             "each step and its undeformed position."))
    quantities.addParmTemplate(hou.MenuParmTemplate(
        "export_q_volume#", "Volumes", ("current", "rest"),
        menu_labels=("Current (deformed)", "Rest (undeformed)"),
        default_value=0,
        help="The volumes used for the volume-weighted mean, the integral "
             "and Region Volume. Current: each element's deformed volume at "
             "that step, right for quantities per deformed volume such as "
             "the Cauchy stress. Rest: the undeformed volume, right for "
             "quantities per undeformed volume such as Piola-Kirchhoff "
             "stresses, Green-Lagrange strain and strain energy density. "
             "They differ by the volume ratio J."))
    quantities.addParmTemplate(hou.StringParmTemplate(
        "export_q_label#", "Column Label", 1, default_value=("",),
        help="Name of the quantity in the column headers (e.g. 'Stress'); "
             "empty = a name from the field and value."))
    data.addParmTemplate(quantities)

    data.addParmTemplate(hou.ToggleParmTemplate(
        "export_forces", "Add Force Curves", default_value=True,
        help="Add the Force Curves (Analysis tab) as columns of the same "
             "table: per sideset and obstacle, the reaction, the contact "
             "force and the mean displacement, so force against "
             "displacement plots from one table. Needs a run written with "
             "Nodal Forces on (the default); otherwise the export notes it "
             "and goes on."))
    force_sets = hou.StringParmTemplate(
        "export_force_sets", "Force Sets", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Which sets to add (empty = all).")
    force_sets.setConditional(hou.parmCondType.DisableWhen,
                              "{ export_forces == 0 }")
    data.addParmTemplate(_menu_script(force_sets, "force_set_menu"))
    for name, label, default, join, help_text in (
            ("reaction", "Reaction", True, True,
             "The force the prescribed displacement (or an obstacle's motion) "
             "applies to the body there."),
            ("contact", "Contact Force", False, True,
             "Contact plus friction forces on the set's nodes."),
            ("displacement", "Mean Displacement", True, False,
             "The mean displacement of the set's nodes."),
            ("x", "X", True, True, "The x component."),
            ("y", "Y", True, True, "The y component."),
            ("z", "Z", True, True, "The z component."),
            ("magnitude", "Magnitude", True, False, "The magnitude.")):
        template = hou.ToggleParmTemplate(
            f"export_force_{name}", label, default_value=default,
            join_with_next=join, help=help_text)
        template.setConditional(hou.parmCondType.DisableWhen,
                                "{ export_forces == 0 }")
        data.addParmTemplate(template)

    data.addParmTemplate(hou.MenuParmTemplate(
        "export_units", "Run Units",
        ("auto", "none", "m_kg_s", "mm_t_s", "mm_kg_s", "mm_g_s"),
        menu_labels=("Automatic (as the PolyFEM node recorded)",
                     "Not Set (no unit labels)", "m, kg, s (N, Pa)",
                     "mm, tonne, s (N, MPa)", "mm, kg, s (mN, kPa)",
                     "mm, g, s (uN, Pa)"),
        default_value=0,
        help="The unit system the run's numbers are in. Automatic: the "
             "units the PolyFEM node recorded (its Units are on); a run "
             "without them gets no unit labels until you choose its system "
             "here -- PolyFEM itself does not know whether you entered "
             "millimetres or metres."))
    data.addParmTemplate(_read_only_text(
        "export_units_status", "Units",
        "hou.pwd().hdaModule().export_units_text(hou.pwd())",
        "Read-only: the run's unit system as the export uses it, and where "
        "it comes from."))
    for name, label, tokens, labels in (
            ("length", "Length", ("run", "m", "cm", "mm", "um"),
             ("As the Run", "m", "cm", "mm", "µm")),
            ("stress", "Stress", ("run", "Pa", "kPa", "MPa", "GPa"),
             ("As the Run", "Pa", "kPa", "MPa", "GPa")),
            ("force", "Force", ("run", "N", "mN", "uN", "kN"),
             ("As the Run", "N", "mN", "µN", "kN")),
            ("time", "Time", ("run", "s", "ms"), ("As the Run", "s", "ms"))):
        template = hou.MenuParmTemplate(
            f"export_unit_{name}", f"{label} in", tokens, menu_labels=labels,
            default_value=0, join_with_next=name in ("length", "force"),
            help=f"The unit of every {name.lower()} in the spreadsheet "
                 "(volumes, velocities and integrals follow); values are "
                 "converted. Needs the run's unit system.")
        template.setConditional(hou.parmCondType.DisableWhen,
                                "{ export_units == none }")
        data.addParmTemplate(template)

    for name, label, default, minimum, help_text in (
            ("first", "First Step", 0, 0,
             "The first output step in the table (0 = the start of the "
             "run)."),
            ("last", "Last Step", -1, -1,
             "The last output step in the table; -1 = the run's last step."),
            ("every", "Every", 1, 1,
             "Take every Nth output step (1 = every step).")):
        data.addParmTemplate(hou.IntParmTemplate(
            f"export_step_{name}", label, 1, default_value=(default,),
            min=minimum, max=1000, join_with_next=name != "every",
            help=help_text))
    data.addParmTemplate(hou.ToggleParmTemplate(
        "export_per_item", "Values per Element / Node", default_value=False,
        help="Also write, per quantity, a sheet with the value of every "
             "element (or node) of every region and its undeformed position "
             "-- for profiles (value against x) and histograms. Excel holds "
             "about a million rows."))
    item_steps = hou.MenuParmTemplate(
        "export_per_item_steps", "At", ("last", "screen", "all"),
        menu_labels=("Last Exported Step", "Step on Screen",
                     "Every Exported Step"), default_value=0,
        help="The steps of the per element / node sheets: one column each.")
    item_steps.setConditional(hou.parmCondType.DisableWhen,
                              "{ export_per_item == 0 }")
    data.addParmTemplate(item_steps)
    data.addParmTemplate(hou.StringParmTemplate(
        "export_data_file", "Spreadsheet", 1, default_value=("",),
        string_type=hou.stringParmType.FileReference,
        tags={"filechooser_pattern": "*.xlsx", "filechooser_mode": "write"},
        help="The Excel file to write; empty = <export folder>/<PVD "
             "name>_data.xlsx. An existing file is replaced."))
    data.addParmTemplate(hou.ButtonParmTemplate(
        "export_data", "Export Spreadsheet", **_callback("export_data_button"),
        join_with_next=True,
        help="Read every chosen step (like Auto Range: All Frames) and write "
             "the workbook: a Data sheet with one row per step (Step, Time, "
             "Frame) and one column per region, quantity and statistic, "
             "ready to plot (numbers only, units in the headers, no charts); "
             "the optional per element / node sheets; and an About sheet "
             "(the run, its id and input hash, the regions, the units and "
             "what every column means)."))
    data.addParmTemplate(hou.ButtonParmTemplate(
        "export_data_open", "Open Spreadsheet",
        **_callback("open_export_data"),
        help="Open the spreadsheet with the system's default program "
             "(exporting it first when it does not exist yet)."))
    data.addParmTemplate(_status_text(
        "export_data_status", "Status",
        "Read-only: what the last export wrote (file, steps, columns, "
        "elements and nodes per region) or why it wrote nothing."))
    tab.addParmTemplate(data)
    return tab


def _parms():
    ptg = hou.ParmTemplateGroup()

    main = hou.FolderParmTemplate("main_folder", "Main",
                                  folder_type=hou.folderType.Tabs)
    main.addParmTemplate(hou.StringParmTemplate(
        "PVD_file", "PVD File", 1,
        string_type=hou.stringParmType.FileReference,
        tags={"filechooser_pattern": "*.pvd"},
        script_callback="hou.phm().start(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="The .pvd file PolyFEM wrote in its output folder (sim.pvd by "
             "default). It lists one result file per saved time step; those "
             "steps are mapped to Houdini frames when you press Refresh / Set "
             "Playbar."))
    main.addParmTemplate(hou.ButtonParmTemplate(
        "refresh", "Refresh / Set Playbar",
        script_callback="hou.phm().refresh(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        join_with_next=True,
        help="Re-read the PVD file (e.g. while a simulation is still writing "
             "steps), rebuild the field menus, and optionally set the playbar "
             "range to the available time steps."))
    main.addParmTemplate(hou.ToggleParmTemplate(
        "update_playbar", "Set Playbar Range on Refresh", default_value=True,
        help="Update Houdini's playbar range when loading a PVD or pressing "
             "Refresh. Turn off when you have several Read PVD nodes or a "
             "PolyFEM animation timeline you do not want overwritten."))
    main.addParmTemplate(hou.MenuParmTemplate(
        "time_mapping", "Time Mapping", TIME_MAPPING_TOKENS,
        menu_labels=("Automatic", "One Frame per Output Step",
                     "Simulation Time"),
        default_value=1,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which time step each Houdini frame shows. One Frame per Output "
             "Step (the default): frame k shows step k, so the playbar has "
             "exactly one frame per solved output step (0..N-1). Automatic: "
             "Simulation Time with the PolyFEM node's own Start Frame and "
             "Time Scale when the run was written by it (input/hda_scene.json), "
             "one frame per output step otherwise. Simulation Time: Houdini "
             "time is simulation time (simulation time = Start Time + "
             "(frame's time - Start Frame's time) x Time Scale), so the "
             "results play back in step with keyframed loads and obstacle "
             "motions; a frame shows the last step at or before its "
             "simulation time."))
    start_frame = hou.FloatParmTemplate(
        "time_start_frame", "Start Frame", 1, default_value=(1.0,),
        min=0.0, max=240.0,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Simulation Time mapping: the Houdini frame at which the run's "
             "first step (its start time) is shown.")
    start_frame.setConditional(hou.parmCondType.HideWhen,
                               "{ time_mapping != time }")
    main.addParmTemplate(start_frame)
    time_scale = hou.FloatParmTemplate(
        "time_scale", "Time Scale", 1, default_value=(1.0,),
        min=0.001, max=10.0,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Simulation Time mapping: simulated time per unit of Houdini "
             "time (1 = one simulated second per Houdini second).")
    time_scale.setConditional(hou.parmCondType.HideWhen,
                              "{ time_mapping != time }")
    main.addParmTemplate(time_scale)
    sim_time = hou.StringParmTemplate(
        "sim_time_status", "Simulation Time", 1,
        default_expression=(
            "hou.pwd().hdaModule().sim_time_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: the time step on screen and its simulation time.")
    sim_time.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(sim_time)
    source_block = hou.StringParmTemplate(
        "source_block", "Source Block", 1, default_value=("Volume",),
        menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().source_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which part of the result to load as the main geometry: Volume "
             "(the solid elements), Surface (boundary faces with normals, "
             "sidesets, tractions), Contact (contact and friction forces per "
             "surface node) or Points (FEM nodes). Only blocks present in the "
             "current PVD frame are listed; enable the matching options in "
             "the PolyFEM node's Output tab to produce the others.")
    source_block.setItemGeneratorScript(
        "hou.phm().source_block_menu(kwargs)")
    source_block.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    main.addParmTemplate(source_block)
    avail = hou.StringParmTemplate(
        "availability_status", "Available Data", 1,
        default_value=("Select a PVD file to inspect its renderable data.",),
        help="Read-only summary of what the loaded result contains (blocks, "
             "fields, bodies, fibers) and therefore which controls below are "
             "usable.")
    avail.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(avail)
    main.addParmTemplate(_read_only_text(
        "run_summary_status", "Run Summary",
        "hou.pwd().hdaModule().run_summary_text(hou.pwd())",
        "Read-only summary from PolyFEM's run-manifest.json (status, exit "
        "code, wall time, peak memory, rollbacks, AL stagnations, and "
        "contact history-sensitivity counters when present).",
        lines="2-5"))
    main.addParmTemplate(hou.MenuParmTemplate(
        "field_time_scope", "Field Availability Scope",
        ("current", "every", "any"),
        menu_labels=("Current Frame (fast)",
                     "Every Frame (scans whole sequence - slow)",
                     "Any Frame (scans whole sequence - slow)"),
        default_value=0,
        script_callback="hou.phm().time_scope_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Controls which fields appear in menus. Current Frame only "
             "inspects the displayed frame and is fast; a stale selection "
             "simply shows the Unavailable Field Color, never an error. Every "
             "Frame and Any Frame must inspect every result file in the sequence to find "
             "intermittent fields and label their coverage -- this can take a "
             "while for long sequences of large meshes and there is no way "
             "around it (the field list lives inside each file)."))
    main.addParmTemplate(hou.ToggleParmTemplate(
        "remesh_mode", "Remeshing Mode", default_value=False,
        help="Rebuild topology every frame (needed for remeshing runs). This "
             "makes every frame change re-parse and rebuild the mesh, so "
             "scrubbing is much slower; leave off unless the topology actually "
             "changes over time. Off = topology parsed once and cached."))
    topo_frame = hou.IntParmTemplate(
        "topo_frame", "Topology Frame", 1, default_value=(0,),
        help="When Remeshing Mode is off, the mesh connectivity is read once "
             "from the time step this Houdini frame shows (through Time "
             "Mapping, like the playbar) and reused for every other frame "
             "(only positions and fields change). Leave at 0 (the first "
             "step) unless that step is missing or damaged.")
    topo_frame.setConditional(
        hou.parmCondType.DisableWhen, "{ remesh_mode == 1 }")
    main.addParmTemplate(topo_frame)
    main.addParmTemplate(hou.ToggleParmTemplate(
        "cache", "Cache Frames", default_value=True,
        script_callback="hou.phm().toggle_cache(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Keep each frame's imported data in memory once it has been "
             "read, so returning to a frame never reads its file again. "
             "Derived fields and every display setting (deformation, colors, "
             "comparison, smoothing, visibility, glyphs, fibers, clipping) "
             "are recalculated live from the cached data, so changing them "
             "never re-reads the files. The cache is rebuilt automatically "
             "when the PVD file, source block, topology mode, remeshing "
             "settings or Display Boundary Surface Only change; Clear Cache "
             "empties it, and so does turning this off. "
             "Cached frames are limited by Cache Memory Limit."))
    cache_limit = hou.FloatParmTemplate(
        "cache_memory_gb", "Cache Memory Limit (GB)", 1,
        default_value=(0.0,), min=0.0, max=512.0,
        help="Most memory the frame cache may use. A new node starts at the "
             "system memory available when it was created; 0 = half of this "
             "computer's memory. Each frame's size is measured when it is "
             "cached, and once the limit is reached the oldest cached frames "
             "are dropped first, so a long run of a large mesh does not push "
             "the computer into swapping. Lower it if other programs need "
             "the memory.")
    cache_limit.setConditional(hou.parmCondType.DisableWhen, "{ cache == 0 }")
    main.addParmTemplate(cache_limit)
    cache_status = hou.StringParmTemplate(
        "cache_status", "Cache Status", 1,
        default_expression=(
            "hou.pwd().hdaModule().cache_status_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: how many frames the cache holds and the memory "
             "they use, and how many fit within Cache Memory Limit at the "
             "measured size of one frame. Clear Cache resets it to zero.")
    cache_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(cache_status)
    memory_status = hou.StringParmTemplate(
        "memory_status", "Memory", 1,
        default_expression=(
            "hou.pwd().hdaModule().memory_status_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: memory used by Houdini, and the computer's used, "
             "total and available memory (available = what can still be "
             "handed out without swapping).")
    memory_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(memory_status)
    cache_epoch = hou.IntParmTemplate(
        "cache_epoch", "cache_epoch", 1, default_value=(0,),
        help="Internal, hidden: counts Clear Cache presses so Cache Status "
             "refreshes immediately.")
    cache_epoch.hide(True)
    main.addParmTemplate(cache_epoch)
    data_epoch = hou.IntParmTemplate(
        "data_epoch", "data_epoch", 1, default_value=(0,),
        help="Internal, hidden: counts Refreshes that found rewritten result "
             "files, so the topology and frame stages read them again.")
    data_epoch.hide(True)
    main.addParmTemplate(data_epoch)
    main.addParmTemplate(hou.ButtonParmTemplate(
        "clear_cache", "Clear Cache",
        script_callback="hou.phm().clear_cache(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Drop every cached frame so the next frame change re-reads the "
             "files. Use it after a simulation has been re-run into the same "
             "output folder."))
    ptg.append(main)

    disp = hou.FolderParmTemplate("display_folder", "Display",
                                  folder_type=hou.folderType.Tabs)
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "surface_only", "Display Boundary Surface Only", default_value=True,
        help="Show only the outer surface of the volume mesh instead of "
             "every solid element. Much faster to display for large meshes "
             "and looks the same from outside. Clipping Plane and Thin Slice "
             "automatically reveal interior faces while active; turn this "
             "off only when you need full solid tetrahedra/hexahedra "
             "downstream or unclipped interior access."))
    show_deformed = hou.ToggleParmTemplate(
        "show_deformed", "Show Deformed", default_value=True,
        help="Move the mesh by the computed displacement so you see the "
             "deformed shape. Off shows the undeformed (rest) mesh with the "
             "fields painted on it. The rest positions are always kept in "
             "the 'rest' point attribute.")
    show_deformed.setConditional(
        hou.parmCondType.DisableWhen, "{ has_solution_data == 0 }")
    disp.addParmTemplate(show_deformed)
    visible_bodies = hou.StringParmTemplate(
        "visible_bodies", "Visible Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        script_callback="hou.phm().analysis_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Pick which bodies to show (by PolyFEM body id); leave empty "
             "to show all of them. Handy to hide an obstacle or isolate one "
             "part. The color range, probe, glyphs and clipping only consider "
             "the visible bodies. Needs Body IDs exported from the PolyFEM "
             "node and more than one body in the result.")
    visible_bodies.setItemGeneratorScript("hou.phm().body_id_menu(kwargs)")
    visible_bodies.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    visible_bodies.setConditional(
        hou.parmCondType.DisableWhen, "{ has_multibody == 0 }")
    disp.addParmTemplate(visible_bodies)
    color_attrib = hou.StringParmTemplate(
        "color_attrib", "Color Field", 1, default_value=("von_mises",),
        menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().color_selection_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which result field colors the mesh (displacement, stress, "
             "strain, contact force ...). Only fields that exist in the "
             "loaded block are listed; derived fields appear when the data "
             "they need was exported.")
    color_attrib.setItemGeneratorScript(
        "hou.phm().color_field_menu(kwargs)")
    color_attrib.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    disp.addParmTemplate(color_attrib)
    color_reduction = hou.StringParmTemplate(
        "color_reduction", "Field Value To Display", 1,
        default_value=("auto",), menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().color_selection_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="A color can only show one number per point, so vector and "
             "tensor fields must be reduced: magnitude or a component for "
             "vectors; von Mises-style norm, a diagonal entry, a principal "
             "value (principal values are ordered maximum to minimum; for a "
             "non-symmetric tensor such as the deformation gradient or the "
             "1st Piola-Kirchhoff stress, those of its symmetric part), trace "
             "or determinant for tensors. "
             "Only reductions that make sense for the chosen field are "
             "listed.")
    color_reduction.setItemGeneratorScript(
        "hou.phm().color_reduction_menu(kwargs)")
    color_reduction.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    disp.addParmTemplate(color_reduction)
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "smooth_field", "Smooth Field Across Elements", default_value=False,
        help="PolyFEM writes a discontinuous mesh, so element-wise quantities "
             "(stress, strain) show one flat color per element. This nodally "
             "averages the displayed value onto shared vertices for a "
             "continuous, interpolated field. PolyFEM's own continuous '_avg' "
             "fields (e.g. von_mises_avg) are preferred automatically when "
             "they exist; other fields use a generic nodal average. Linear "
             "elements have no true sub-element variation -- raise PolyFEM's "
             "vismesh sampling or element order for that."))
    color_status = hou.StringParmTemplate(
        "color_status", "Color Field Status", 1,
        default_value=("Cook the result to validate the selected field.",),
        help="Read-only: whether the chosen Color Field was found in the "
             "current frame, and the reduction that is actually displayed.")
    color_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    disp.addParmTemplate(color_status)
    missing = hou.FloatParmTemplate(
        "missing_color", "Unavailable Field Color", 3,
        default_value=(0.35, 0.35, 0.35),
        help="Flat color painted on the mesh when the selected Color Field "
             "does not exist in the current frame (for example a field that "
             "PolyFEM only wrote for some steps).")
    disp.addParmTemplate(missing)
    invalid = hou.FloatParmTemplate(
        "invalid_color", "Invalid / NaN Value Color", 3,
        default_value=(1.0, 0.0, 1.0),
        help="Color used for points whose value is not a number (NaN), "
             "infinite, or not allowed by the color scale (e.g. zero or "
             "negative with a logarithmic scale), so they stand out instead "
             "of being hidden.")
    disp.addParmTemplate(invalid)
    disp.addParmTemplate(hou.MenuParmTemplate(
        "color_scale", "Color Scale", ("linear", "log10"),
        menu_labels=("Linear", "Logarithmic (Base 10)"), default_value=0,
        script_callback="hou.phm().color_scale_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Linear maps values evenly between the displayed minimum and "
             "maximum. Logarithmic spreads the colors by powers of ten, which "
             "helps when values span several orders of magnitude (stress "
             "concentrations). Log needs positive values: zero, negative, "
             "NaN and infinite values get the Invalid Value Color."))
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "range_lock", "Lock Displayed Range", default_value=False,
        help="Freeze the displayed minimum and maximum so the Auto Range "
             "buttons and frame changes cannot alter them. Use it to compare "
             "frames or runs with an identical color scale."))
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "range_percentile", "Ignore Range Outliers", default_value=False,
        help="When auto-ranging, ignore the most extreme values and use "
             "percentile limits instead of the absolute minimum and maximum, "
             "so a few outliers (a single hot element) do not wash out the "
             "colors everywhere else."))
    range_percentiles = hou.FloatParmTemplate(
        "range_percentiles", "Range Percentiles", 2,
        default_value=(1.0, 99.0), min=0.0, max=100.0,
        help="Lower and upper percentiles used by Ignore Range Outliers: "
             "1 and 99 map the color range to the middle 98% of the values.")
    range_percentiles.setConditional(
        hou.parmCondType.DisableWhen, "{ range_percentile == 0 }")
    disp.addParmTemplate(range_percentiles)
    range_symmetric = hou.ToggleParmTemplate(
        "range_symmetric", "Symmetric Range Around Zero", default_value=False,
        help="Make the displayed range symmetric around zero (e.g. -5 to +5) "
             "so that a diverging ramp puts zero exactly in the middle. "
             "Useful for signed fields such as principal stresses.")
    range_symmetric.setConditional(
        hou.parmCondType.DisableWhen, "{ color_scale == log10 }")
    disp.addParmTemplate(range_symmetric)
    color_min = hou.FloatParmTemplate(
        "color_min", "Displayed Minimum", 1, default_value=(0.0,),
        help="Value mapped to the left end of the color ramp. Values below "
             "it are clamped to that color. Set by the Auto Range buttons or "
             "type your own.")
    color_min.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(color_min)
    color_max = hou.FloatParmTemplate(
        "color_max", "Displayed Maximum", 1, default_value=(1.0,),
        help="Value mapped to the right end of the color ramp. Values above "
             "it are clamped to that color.")
    color_max.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(color_max)
    autoscale_btn = hou.ButtonParmTemplate(
        "autoscale", "Auto Range: Current Frame",
        script_callback="hou.phm().autoscale(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Set Displayed Minimum/Maximum from the values in the frame "
             "currently shown (instant). Respects Ignore Range Outliers and "
             "Symmetric Range; blocked by Lock Displayed Range.")
    autoscale_btn.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(autoscale_btn)
    autoscale_all_btn = hou.ButtonParmTemplate(
        "autoscale_all", "Auto Range: All Frames (scans sequence)",
        script_callback="hou.phm().autoscale_all(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Scans the selected field over the complete PVD sequence to find "
             "the global range. This must read every frame once, so it can "
             "take a while for long sequences of large meshes -- unavoidable, "
             "the values live inside each file. Use Auto Range: Current Frame "
             "for an instant range from the displayed frame.")
    autoscale_all_btn.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(autoscale_all_btn)
    ramp = hou.RampParmTemplate(
        "color_ramp", "Displayed Color Ramp", hou.rampParmType.Color,
        help="Colors used from Displayed Minimum (left) to Displayed Maximum "
             "(right). Edit it like any Houdini color ramp, or use the "
             "Set Signed Diverging Ramp button for blue-white-red.")
    disp.addParmTemplate(ramp)
    disp.addParmTemplate(hou.ButtonParmTemplate(
        "set_diverging_ramp", "Set Signed Diverging Ramp",
        script_callback="hou.phm().set_diverging_ramp(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Replace the ramp with blue-white-red, the usual choice for "
             "signed values (negative = blue, zero = white, positive = red). "
             "Pair it with Symmetric Range Around Zero."))
    ptg.append(disp)

    ana = hou.FolderParmTemplate("analysis_folder", "Analysis",
                                 folder_type=hou.folderType.Tabs)
    derived_toggle = hou.ToggleParmTemplate(
        "derived", "Compute Derived Quantities", default_value=True,
        script_callback="hou.phm().analysis_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Compute extra mechanics fields from the exported deformation "
             "gradient and stress so you can color and probe them: volume "
             "ratio, the Cauchy-Green tensors, Green-Lagrange / Almansi / "
             "logarithmic / small strains, both Piola-Kirchhoff stresses, "
             "principal values, von Mises, invariants and more. Turn off to "
             "load faster when you only need the exported fields.")
    ana.addParmTemplate(derived_toggle)
    derived_mode = hou.MenuParmTemplate(
        "derived_mode", "Derived Fields Scope",
        ("on_demand", "all"),
        menu_labels=("Active Display / Glyphs / Probe (fast)",
                     "All Mechanics Fields (downstream SOPs)"),
        default_value=0,
        help="Controls which derived mechanics attributes are computed and "
             "attached to the geometry. 'Active Display / Glyphs / Probe' "
             "computes only the fields needed by the current Color Field, "
             "glyphs, fibers, and probe (much faster and far less memory on "
             "large meshes). Switch to 'All Mechanics Fields' when downstream "
             "SOPs need every derived tensor and invariant simultaneously.")
    derived_mode.setConditional(
        hou.parmCondType.DisableWhen, "{ derived == 0 }")
    ana.addParmTemplate(derived_mode)
    glyph_toggle = hou.ToggleParmTemplate(
        "add_glyphs", "Show Principal-Direction Glyphs", default_value=False,
        script_callback="hou.phm().auto_glyph_scale(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Draw small line glyphs at the nodes showing the three "
             "principal directions of the chosen tensor (e.g. the directions "
             "of largest and smallest stress). By default the line length "
             "is proportional to the principal value; enabling glyphs picks "
             "a length scale that fits the mesh.")
    glyph_toggle.setConditional(
        hou.parmCondType.DisableWhen, "{ has_glyph_data == 0 }")
    ana.addParmTemplate(glyph_toggle)
    glyph_tensor = hou.StringParmTemplate(
        "glyph_tensor", "Glyph Tensor", 1, default_value=("cauchy",),
        menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().auto_glyph_scale(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which tensor the glyphs represent: Cauchy stress, second "
             "Piola-Kirchhoff stress, Green-Lagrange strain or the right/left "
             "Cauchy-Green tensor. Only tensors that can be computed from the "
             "loaded data are listed; switching re-estimates the glyph scale.")
    glyph_tensor.setItemGeneratorScript(
        "hou.phm().glyph_tensor_menu(kwargs)")
    glyph_tensor.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    glyph_off = "{ has_glyph_data == 0 } { add_glyphs == 0 }"
    glyph_tensor.setConditional(hou.parmCondType.DisableWhen, glyph_off)
    ana.addParmTemplate(glyph_tensor)
    direction_folder = hou.FolderParmTemplate(
        "glyph_directions", "Principal Directions",
        folder_type=hou.folderType.Simple)
    for name, label in (
            ("glyph_direction_first", "Show First Principal Direction"),
            ("glyph_direction_second", "Show Second Principal Direction"),
            ("glyph_direction_third", "Show Third Principal Direction")):
        direction = hou.ToggleParmTemplate(name, label, default_value=True,
                                           help=(
            "Draw this principal direction. Principal directions are ordered "
            "by principal value from largest to smallest."))
        direction.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        direction_folder.addParmTemplate(direction)
    ana.addParmTemplate(direction_folder)
    for template in (
            hou.FloatParmTemplate(
                "tensor_scale", "Principal-Value Length Scale", 1,
                default_value=(0.01,),
                help="Multiplier from principal value to line length (scene "
                     "units per unit of the tensor). Press Auto-Estimate "
                     "Glyph Scale to get a value that fits the mesh; then "
                     "adjust by eye."),
            hou.MenuParmTemplate(
                "glyph_length_mode", "Glyph Length Mode",
                ("value", "normalized", "clamped"),
                menu_labels=("Scale by Principal Value",
                             "Normalized Directions",
                             "Scale by Value with Maximum Length"),
                default_value=0,
                help="How long each glyph line is. Scale by Principal Value: "
                     "length proportional to the magnitude (shows where the "
                     "field is strong). Normalized: all lines the same length "
                     "(shows only direction). Scale by Value with Maximum "
                     "Length: proportional but capped, so a few huge values "
                     "do not hide everything else.")):
        template.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(template)
    glyph_max_length = hou.FloatParmTemplate(
        "glyph_max_length", "Maximum Glyph Length", 1, default_value=(1.0,),
        min=0.0, max=1e9,
        help="Longest line allowed (scene units) when Glyph Length Mode is "
             "Scale by Value with Maximum Length.")
    glyph_max_length.setConditional(
        hou.parmCondType.DisableWhen,
        glyph_off + " { glyph_length_mode != clamped }")
    ana.addParmTemplate(glyph_max_length)
    for name, label, which in (
            ("glyph_scale_first", "First Direction Length Multiplier",
             "largest"),
            ("glyph_scale_second", "Second Direction Length Multiplier",
             "middle"),
            ("glyph_scale_third", "Third Direction Length Multiplier",
             "smallest")):
        gscale = hou.FloatParmTemplate(
            name, label, 1, default_value=(1.0,), min=0.0, max=1e9,
            help=f"Extra length factor for the glyphs of the {which} "
                 "principal value only, e.g. to emphasize one direction "
                 "(1 = no change).")
        gscale.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(gscale)
    glyph_arrowheads = hou.ToggleParmTemplate(
        "glyph_arrowheads", "Show Glyph Arrowheads", default_value=False,
        help="Draw a small arrowhead at both ends of each glyph line so the "
             "direction reads clearly from a distance.")
    glyph_arrowheads.setConditional(hou.parmCondType.DisableWhen, glyph_off)
    ana.addParmTemplate(glyph_arrowheads)
    glyph_arrow_size = hou.FloatParmTemplate(
        "glyph_arrow_size", "Glyph Arrowhead Size", 1,
        default_value=(0.08,), min=0.001, max=0.5,
        help="Size of the arrowheads as a fraction of the glyph length.")
    glyph_arrow_size.setConditional(
        hou.parmCondType.DisableWhen,
        glyph_off + " { glyph_arrowheads == 0 }")
    ana.addParmTemplate(glyph_arrow_size)
    for template in (
            hou.IntParmTemplate(
                "glyph_stride", "Glyph Sampling Stride", 1,
                default_value=(1,), min=1, max=100000,
                help="Draw a glyph only at every Nth point, to thin them out "
                     "on dense meshes (1 = every point)."),
            hou.ToggleParmTemplate(
                "glyph_sign_color", "Stress Sign Colors", default_value=True,
                help="For stress tensors, color glyphs red where the "
                     "principal stress is tension (positive) and blue where "
                     "it is compression (negative). Strain tensors use the "
                     "uniform color."),
            hou.FloatParmTemplate(
                "glyph_color", "Uniform Glyph Color", 3,
                default_value=(1, 1, 1),
                help="Color of the glyph lines when Stress Sign Colors is off "
                     "(or for strain tensors, which have no sign coloring)."),
            hou.ButtonParmTemplate(
                "autoglyph", "Auto-Estimate Glyph Scale",
                script_callback="hou.phm().autoglyph(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Size the glyphs to the mesh: the largest glyphs span "
                     "about half an average edge. For value-scaled glyphs the "
                     "estimate accounts for the selected tensor's "
                     "principal-value magnitude, so it works for stress and "
                     "strain alike.")):
        template.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(template)
    fiber_folder = hou.FolderParmTemplate(
        "fiber_folder", "Fibers", folder_type=hou.folderType.Simple)
    show_fibers = hou.ToggleParmTemplate(
        "show_fibers", "Show Fiber Directions", default_value=False,
        help="Draw the material fiber direction at each node as a short "
             "line (only for results whose materials have fibers). Fiber "
             "Frame chooses between the direction you set up (a0) and the "
             "direction after deformation. The fiber data comes from the "
             "result's material fields, or from the PolyFEM node's fiber "
             "file when those were not exported.")
    show_fibers.setConditional(
        hou.parmCondType.DisableWhen, "{ has_fiber_data == 0 }")
    fiber_folder.addParmTemplate(show_fibers)

    family = hou.StringParmTemplate(
        "fiber_family", "Fiber Family", 1, default_value=("all",),
        menu_items=("all",), menu_labels=("All Families",),
        item_generator_script="hou.phm().fiber_family_menu(kwargs)",
        item_generator_script_language=hou.scriptLanguage.Python,
        help="Which fiber family to draw when a composite material has "
             "several (each family is exported as its own field). All "
             "Families draws every one.")
    family.setConditional(hou.parmCondType.DisableWhen, "{ show_fibers == 0 }")
    fiber_folder.addParmTemplate(family)

    frame = hou.MenuParmTemplate(
        "fiber_frame", "Fiber Frame", ("reference", "deformed", "both"),
        menu_labels=("Reference a0 (simulation/world)",
                     "Current normalize(F a0)",
                     "Both (reference dimmed)"),
        default_value=1,
        help="Reference draws the fiber direction as it was defined for "
             "the simulation (a0; PolyFEM does not rotate it with the "
             "geometry transform). Current draws where that fiber points "
             "after the deformation of this frame (the direction a0 is "
             "carried to by the deformation gradient F). Both shows the two "
             "together with the reference dimmed.")
    frame.setConditional(hou.parmCondType.DisableWhen, "{ show_fibers == 0 }")
    fiber_folder.addParmTemplate(frame)

    for template in (
            hou.FloatParmTemplate("fiber_scale", "Fiber Line Length", 1,
                                  default_value=(0.01,), min=0.0, max=1e9,
                                  help="Length of each drawn fiber line in "
                                       "scene units. Press Auto-Estimate "
                                       "Fiber Scale for a value that fits "
                                       "the mesh."),
            hou.IntParmTemplate("fiber_stride", "Fiber Sampling Stride", 1,
                                default_value=(1,), min=1, max=100000,
                                help="Draw a fiber line only at every Nth "
                                     "node, to thin them out on dense "
                                     "meshes (1 = every node)."),
            hou.MenuParmTemplate(
                "fiber_color_mode", "Fiber Color",
                ("direction_rgb", "family", "uniform", "stretch", "angle"),
                menu_labels=("Direction (RGB)", "Per Family", "Uniform",
                             "Fiber Stretch |F a0|",
                             "Direction Change Angle"),
                default_value=0,
                help="How fiber lines are colored: by direction (x/y/z as "
                     "red/green/blue), by fiber family, one uniform color, "
                     "by how much each fiber has stretched (|F a0|, 1 = "
                     "unstretched), or by the angle between its current and "
                     "reference direction."),
            hou.FloatParmTemplate("fiber_color", "Uniform Fiber Color", 3,
                                  default_value=(1, 1, 1),
                                  help="Color of the fiber lines when Fiber "
                                       "Color is set to Uniform.")):
        template.setConditional(
            hou.parmCondType.DisableWhen,
            "{ show_fibers == 0 } { fiber_color_mode != uniform }"
            if template.name() == "fiber_color" else "{ show_fibers == 0 }")
        fiber_folder.addParmTemplate(template)

    autofiber = hou.ButtonParmTemplate(
        "autofiber", "Auto-Estimate Fiber Scale",
        script_callback="hou.phm().autofiber(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Size the fiber lines to the mesh (about half an average edge).")
    autofiber.setConditional(
        hou.parmCondType.DisableWhen, "{ show_fibers == 0 }")
    fiber_folder.addParmTemplate(autofiber)
    fiber_attribs = hou.StringParmTemplate(
        "fiber_attribs", "fiber_attribs", 1,
        default_value=("",),
        help="Internal, hidden: the fiber vector attributes assembled on "
             "load.")
    fiber_attribs.hide(True)
    fiber_folder.addParmTemplate(fiber_attribs)
    ana.addParmTemplate(fiber_folder)
    ana.addParmTemplate(_force_curve_folder())
    ptg.append(ana)
    ptg.append(_export_tab())

    multi = hou.FolderParmTemplate("multi_block_folder", "Multi-Block",
                                   folder_type=hou.folderType.Tabs)
    multi.addParmTemplate(hou.ToggleParmTemplate(
        "show_multi_blocks", "Show Additional PVD Blocks",
        default_value=False,
        help="Show other parts of the result (Volume, Surface, Contact, "
             "Points) together with the main Source Block, e.g. the contact "
             "forces drawn over the deformed volume. Each extra block has its "
             "own field, range and color ramp in the folders below."))
    for slug, label in (
            ("volume", "Volume"), ("surface", "Surface"),
            ("contact", "Contact"), ("points", "Points")):
        # Each control is disabled when the block is absent from the PVD
        # (folder-level DisableWhen isn't honored, so gate per-control).
        absent = f"{{ has_block_{slug} == 0 }}"
        folder = hou.FolderParmTemplate(
            f"multi_{slug}_folder", label, folder_type=hou.folderType.Simple)
        show = hou.ToggleParmTemplate(
            f"multi_{slug}_show", f"Show Additional {label}",
            default_value=False,
            help=f"Draw the {label} block on top of the main result, with "
                 "its own field, range and ramp below. Only available when "
                 "the PVD actually contains a {label} block.")
        show.setConditional(hou.parmCondType.DisableWhen, absent)
        folder.addParmTemplate(show)
        field = hou.StringParmTemplate(
            f"multi_{slug}_field", f"{label} Color Field", 1,
            default_value=("",), menu_type=hou.menuType.StringReplace,
            help=f"Field used to color the additional {label} block. Only "
                 "fields present in that block are listed.")
        field.setItemGeneratorScript(
            "hou.phm().multi_block_field_menu(kwargs)")
        field.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
        field.setConditional(
            hou.parmCondType.DisableWhen,
            f"{{ has_block_{slug} == 0 }} {{ multi_{slug}_show == 0 }}")
        folder.addParmTemplate(field)
        reduction = hou.MenuParmTemplate(
            f"multi_{slug}_reduction", f"{label} Field Value",
            ("magnitude", "x", "y", "z"),
            menu_labels=("Magnitude / Scalar", "X Component", "Y Component",
                         "Z Component"), default_value=0,
            help=f"For a vector field on the {label} block, show its length "
                 "(magnitude) or one component; scalar fields always show "
                 "their value.")
        reduction.setConditional(
            hou.parmCondType.DisableWhen, f"{{ multi_{slug}_show == 0 }}")
        folder.addParmTemplate(reduction)
        rng = hou.FloatParmTemplate(
            f"multi_{slug}_range", f"{label} Displayed Range", 2,
            default_value=(0.0, 1.0),
            help=f"Minimum and maximum value mapped to the {label} color "
                 "ramp. This block has its own range, independent of the "
                 "main result's.")
        rng.setConditional(
            hou.parmCondType.DisableWhen, f"{{ multi_{slug}_show == 0 }}")
        folder.addParmTemplate(rng)
        block_ramp = hou.RampParmTemplate(
            f"multi_{slug}_ramp", f"{label} Color Ramp",
            hou.rampParmType.Color,
            help=f"Color ramp used for the additional {label} block.")
        block_ramp.setConditional(
            hou.parmCondType.DisableWhen,
            f"{{ has_block_{slug} == 0 }} {{ multi_{slug}_show == 0 }}")
        folder.addParmTemplate(block_ramp)
        multi.addParmTemplate(folder)
    ptg.append(multi)

    compare = hou.FolderParmTemplate("comparison_folder", "Comparison",
                                     folder_type=hou.folderType.Tabs)
    compare.addParmTemplate(hou.ToggleParmTemplate(
        "reference_enable", "Compare With Reference Frame",
        default_value=False,
        script_callback="hou.phm().reference_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Colors the active field by its change from a selected frame. "
             "Requires matching point topology. Parses the reference frame in "
             "addition to the current one, so each frame change does a little "
             "more work while this is on."))
    ref_frame = hou.IntParmTemplate(
        "reference_frame", "Reference Frame", 1, default_value=(0,), min=0,
        script_callback="hou.phm().reference_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Houdini frame to compare against: the time step that frame "
             "shows (through Time Mapping, like the playbar) is the "
             "reference; the status line names the step. 0 shows the first "
             "step, usually the undeformed start. The mesh must have the same "
             "points in both frames.")
    ref_frame.setConditional(
        hou.parmCondType.DisableWhen, "{ reference_enable == 0 }")
    compare.addParmTemplate(ref_frame)
    ref_mode = hou.MenuParmTemplate(
        "reference_mode", "Comparison Value",
        ("difference", "absolute", "percent"),
        menu_labels=("Current Minus Reference", "Absolute Difference",
                     "Percentage Change"), default_value=0,
        script_callback="hou.phm().reference_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="What to color: the signed change since the reference frame, "
             "its magnitude, or the change as a percentage of the reference "
             "value.")
    ref_mode.setConditional(
        hou.parmCondType.DisableWhen, "{ reference_enable == 0 }")
    compare.addParmTemplate(ref_mode)
    ref_status = hou.StringParmTemplate(
        "reference_status", "Reference Comparison Status", 1,
        default_value=("Comparison is disabled.",),
        help="Read-only: whether the reference frame could be loaded and "
             "matched to the current one.")
    ref_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    compare.addParmTemplate(ref_status)
    ptg.append(compare)

    probe = hou.FolderParmTemplate("probe_folder", "Probe",
                                   folder_type=hou.folderType.Tabs)
    probe.addParmTemplate(hou.ToggleParmTemplate(
        "probe_enabled", "Enable Viewport Probe", default_value=False,
        script_callback="hou.phm().overlay_feature_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Click the displayed result to inspect position, selected field, "
             "vector/tensor values, IDs, and reference comparison. Enabling "
             "this activates the node's viewer state (no need to press Enter "
             "in the viewport); the probe works while that state is active."))
    probe_pt = hou.IntParmTemplate(
        "probe_point", "Probed Point Number", 1, default_value=(-1,), min=-1,
        help="The last probed vertex (-1 = none): its point number in the "
             "result before clipping (the point number in the loaded block's "
             "result file). The marker follows this vertex as frames change, "
             "also while a clip or slice is active; you can also type a "
             "number to probe it directly.")
    probe_pt.setConditional(
        hou.parmCondType.DisableWhen, "{ probe_enabled == 0 }")
    probe.addParmTemplate(probe_pt)
    probe_readout = hou.StringParmTemplate(
        "probe_readout", "Probe Readout", 1,
        default_value=("Enable the viewport probe and click the result.",),
        tags={"editor": "1", "editorlines": "4-10"},
        help="Read-only text with the probed point's position, displacement, "
             "field value, body and sideset ids, principal values and "
             "reference comparison.")
    probe_readout.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    probe.addParmTemplate(probe_readout)
    ptg.append(probe)

    section = hou.FolderParmTemplate("section_folder", "Clip / Slice",
                                     folder_type=hou.folderType.Tabs)
    section.addParmTemplate(hou.MenuParmTemplate(
        "clip_mode", "Section Mode", ("off", "clip", "slice"),
        menu_labels=("Off", "Clipping Plane", "Thin Slice"),
        default_value=0,
        help="Cut into the model to see inside. Clipping Plane removes one "
             "side of a plane; Thin Slice keeps only a slab centered on the "
             "plane. Glyphs and fibers are added after the cut and are never "
             "removed by it."))
    for template in (
            hou.FloatParmTemplate(
                "clip_origin", "Plane Origin", 3, default_value=(0, 0, 0),
                help="Auto-set to the scene's bounding-box center when a PVD "
                     "file is loaded (only while still at 0,0,0; your edits "
                     "are kept)."),
            hou.FloatParmTemplate(
                "clip_direction", "Plane Normal", 3, default_value=(1, 0, 0),
                help="Auto-aligned to the scene's longest bounding-box axis "
                     "when a PVD file is loaded (only while still at the +X "
                     "default; your edits are kept).")):
        template.setConditional(
            hou.parmCondType.DisableWhen, "{ clip_mode == off }")
        section.addParmTemplate(template)
    clip_keep = hou.MenuParmTemplate(
        "clip_keep", "Clip Side To Keep", ("above", "below"),
        menu_labels=("Above Plane", "Below Plane"), default_value=0,
        help="Which side of the clipping plane stays visible (above = the "
             "side the Plane Normal points to).")
    clip_keep.setConditional(
        hou.parmCondType.DisableWhen, "{ clip_mode != clip }")
    section.addParmTemplate(clip_keep)
    clip_fill = hou.ToggleParmTemplate(
        "clip_fill", "Fill Clipped Surface", default_value=True,
        help="Creates a colored cut surface when clipping closed geometry.")
    clip_fill.setConditional(
        hou.parmCondType.DisableWhen, "{ clip_mode == off }")
    section.addParmTemplate(clip_fill)
    slice_thickness = hou.FloatParmTemplate(
        "slice_thickness", "Slice Thickness", 1, default_value=(0.01,),
        min=1e-8, max=1e9,
        help="Total thickness of the kept slab (centered on Plane Origin, "
             "scene units) in Thin Slice mode.")
    slice_thickness.setConditional(
        hou.parmCondType.DisableWhen, "{ clip_mode != slice }")
    section.addParmTemplate(slice_thickness)
    ptg.append(section)


    legend = hou.FolderParmTemplate("legend_folder", "Legend",
                                    folder_type=hou.folderType.Tabs)
    legend_mode = hou.MenuParmTemplate(
        "legend_mode", "Legend Display Mode",
        ("off", "scene", "overlay", "both"),
        menu_labels=("Off", "Scene Geometry (Renderable, always visible)",
                     "Viewport Overlay (screen-fixed)",
                     "Scene Geometry and Overlay"),
        default_value=0,
        script_callback="hou.phm().overlay_feature_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Scene Geometry is renderable, positioned beside the model, and "
             "always visible regardless of selection. The Viewport Overlay is "
             "screen-fixed but is drawn by this node's viewer state, so it only "
             "shows while that state is active: selecting an overlay mode "
             "activates it automatically (and the 'Show Overlay In Viewport' "
             "button re-activates it), and it persists through viewport "
             "navigation until you select another node or change tools.")
    legend.addParmTemplate(legend_mode)
    overlay_button = hou.ButtonParmTemplate(
        "show_overlay", "Show Overlay In Viewport",
        script_callback="hou.phm().show_overlay(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Activate this node's viewer state so the overlay legend and "
             "click-probe appear without pressing Enter in the viewport.")
    overlay_button.setConditional(
        hou.parmCondType.DisableWhen,
        "{ legend_mode != 2 legend_mode != 3 probe_enabled == 0 }")
    legend.addParmTemplate(overlay_button)
    legend_off = "{ legend_mode == off }"
    legend_and_probe_off = "{ legend_mode == off probe_enabled == 0 }"
    scene_legend_off = "{ legend_mode == off } { legend_mode == overlay }"
    overlay_legend_off = "{ legend_mode == off } { legend_mode == scene }"
    for template, cond in (
            (hou.StringParmTemplate(
                "legend_title", "Custom Legend Title", 1, default_value=("",),
                help="Leave empty to use the resolved color field and "
                     "displayed value."), legend_off),
            (hou.IntParmTemplate(
                "legend_ticks", "Number of Tick Labels", 1, default_value=(6,),
                min=2, max=21,
                help="How many labeled values are written along the color bar "
                     "(including both ends)."), legend_off),
            (hou.IntParmTemplate(
                "legend_digits", "Significant Digits", 1, default_value=(5,),
                min=1, max=12,
                help="Number of significant digits shown in the legend labels "
                     "and probe readout."), legend_and_probe_off),
            (hou.MenuParmTemplate(
                "legend_number_format", "Legend Number Format",
                ("automatic", "scientific", "engineering", "fixed"),
                menu_labels=("Automatic", "Scientific", "Engineering",
                             "Fixed"),
                default_value=0,
                help="How legend and probe numbers are written: Automatic "
                     "picks per value; Scientific uses powers of ten "
                     "(1.2e+05); Engineering uses powers of a thousand "
                     "(120e+03); Fixed uses plain decimals."),
             legend_and_probe_off),
            (hou.StringParmTemplate(
                "field_units", "Displayed Field Units", 1, default_value=("",),
                help="Optional units appended to legend titles and probe "
                     "values."), legend_and_probe_off),
            (hou.FloatParmTemplate(
                "legend_text_color", "Legend Text Color", 3,
                default_value=(1, 1, 1),
                help="Color of the legend title and tick labels (scene legend "
                     "and viewport overlay)."), legend_off)):
        template.setConditional(hou.parmCondType.DisableWhen, cond)
        legend.addParmTemplate(template)
    legend_rotate = hou.FloatParmTemplate(
        "legend_rotate", "Additional Scene Legend Rotation", 3,
        default_value=(0, 0, 0),
        help="Extra rotation (degrees) applied to the scene legend after the "
             "plane/up choice, for fine adjustment.")
    legend_rotate.setLook(hou.parmLook.Angle)
    for template in (
            hou.MenuParmTemplate(
                "legend_orientation", "Scene Legend Plane and Up Direction",
                tuple(str(i) for i in range(12)),
                menu_labels=("XY Plane, +X Up", "XY Plane, -X Up",
                             "XY Plane, +Y Up", "XY Plane, -Y Up",
                             "YZ Plane, +Y Up", "YZ Plane, -Y Up",
                             "YZ Plane, +Z Up", "YZ Plane, -Z Up",
                             "XZ Plane, +X Up", "XZ Plane, -X Up",
                             "XZ Plane, +Z Up", "XZ Plane, -Z Up"),
                default_value=2,
                help="Which world plane the scene legend lies in and which "
                     "way its 'up' points. Choose the plane facing your usual "
                     "camera."),
            hou.FloatParmTemplate(
                "legend_translate", "Scene Legend Translation", 3,
                default_value=(0, 0, 0),
                help="Position of the scene legend geometry (scene units). "
                     "Place Scene Legend Beside Model sets it automatically."),
            legend_rotate,
            hou.FloatParmTemplate(
                "legend_scale", "Scene Legend Scale", 1, default_value=(1.0,),
                min=0.0001, max=1000,
                help="Overall size of the scene legend geometry."),
            hou.ButtonParmTemplate(
                "legend_autoplace", "Place Scene Legend Beside Model",
                script_callback="hou.phm().autoplace_legend(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Move and size the scene legend so it sits next to the "
                     "model's bounding box at a readable size.")):
        template.setConditional(hou.parmCondType.DisableWhen, scene_legend_off)
        legend.addParmTemplate(template)
    for template in (
            hou.MenuParmTemplate(
                "overlay_corner", "Overlay Screen Corner",
                ("upper_left", "upper_right", "lower_left", "lower_right"),
                menu_labels=("Upper Left", "Upper Right", "Lower Left",
                             "Lower Right"),
                default_value=1,
                help="Corner of the viewport where the screen-fixed overlay "
                     "legend is drawn."),
            hou.IntParmTemplate(
                "overlay_margin", "Overlay Margin (Pixels)", 1,
                default_value=(24,), min=0, max=1000,
                help="Distance in pixels between the overlay legend and the "
                     "viewport edges."),
            hou.IntParmTemplate(
                "overlay_width", "Overlay Bar Width (Pixels)", 1,
                default_value=(28,), min=8, max=500,
                help="Width of the overlay color bar in pixels."),
            hou.IntParmTemplate(
                "overlay_height", "Overlay Bar Height (Pixels)", 1,
                default_value=(260,), min=40, max=2000,
                help="Height of the overlay color bar in pixels.")):
        template.setConditional(
            hou.parmCondType.DisableWhen, overlay_legend_off)
        legend.addParmTemplate(template)
    ptg.append(legend)

    gnomon = hou.FolderParmTemplate("gnomon_folder", "Gnomon",
                                    folder_type=hou.folderType.Tabs)
    gnomon.addParmTemplate(hou.ToggleParmTemplate(
        "gnomon_show", "Show Scene Gnomon", default_value=False,
        help="Adds a renderable XYZ orientation marker to the final geometry."))
    gnomon_off = "{ gnomon_show == 0 }"
    for template in (
            hou.FloatParmTemplate(
                "gnomon_center", "Gnomon Center", 3, default_value=(0, 0, 0),
                help="World position of the gnomon's origin (scene units)."),
            hou.FloatParmTemplate(
                "gnomon_scale", "Gnomon Axis Length", 1, default_value=(1.0,),
                min=0.0001, max=1000,
                help="Length of each gnomon axis line (scene units).")):
        template.setConditional(hou.parmCondType.DisableWhen, gnomon_off)
        gnomon.addParmTemplate(template)
    for name, label, what in (
            ("gnomon_x", "Show X Axis", "Draw the red X axis line."),
            ("gnomon_y", "Show Y Axis", "Draw the green Y axis line."),
            ("gnomon_z", "Show Z Axis", "Draw the blue Z axis line."),
            ("gnomon_arrows", "Show Arrowheads",
             "Draw arrowheads at the tip of each axis."),
            ("gnomon_center_marker", "Show Center Marker",
             "Draw a small marker at the gnomon origin.")):
        g_toggle = hou.ToggleParmTemplate(
            name, label, default_value=True, help=what)
        g_toggle.setConditional(hou.parmCondType.DisableWhen, gnomon_off)
        gnomon.addParmTemplate(g_toggle)
    ptg.append(gnomon)

    hidden_flags = ["has_solution_data", "has_glyph_data", "has_multibody",
                    "has_fiber_data"]
    hidden_flags += [f"has_block_{slug}"
                     for slug in ("volume", "surface", "contact", "points")]
    for name in hidden_flags:
        flag = hou.ToggleParmTemplate(
            name, name, default_value=False,
            help="Internal, hidden: set on load to record whether this kind "
                 "of data exists, so the controls that need it can be "
                 "enabled or disabled.")
        flag.hide(True)
        ptg.append(flag)
    return ptg


def build(out_dir):
    hda_path = os.path.join(out_dir, "object_readPVD.1.0.hdanc")
    asset = hda_build.new_asset("GeoObject", TYPE_NAME, LABEL, hda_path)

    def wrangle(name, klass, snippet, inp):
        node = asset.createNode("attribwrangle", name)
        node.setParms({"class": klass, "snippet": snippet})
        if inp is not None:
            node.setNextInput(inp)
        return node

    topo = asset.createNode("python", "topo_build")
    topo.parm("python").set(TOPO_SOP_CODE)
    # spare parms establish cook dependencies on the asset parms
    spare = hou.StringParmTemplate("pvdpath", "pvdpath", 1)
    topo.addSpareParmTuple(spare)
    topo.parm("pvdpath").setExpression('chs("../PVD_file")',
                                       hou.exprLanguage.Hscript)
    spare_f = hou.FloatParmTemplate("topoframe", "topoframe", 1)
    topo.addSpareParmTuple(spare_f)
    # Remeshing Mode follows the playbar; otherwise the fixed Topology Frame.
    # An expression (not a callback editing the node) keeps instances locked.
    # Python short-circuits, so hou.frame() -- and with it time dependence --
    # is only reached in Remeshing Mode (an Hscript if() evaluates both).
    # Topology Frame is a Houdini frame, read through the time mapping like
    # the playbar (review 2026-11-14, item 5).
    topo.parm("topoframe").setExpression(
        "hou.pwd().parent().hdaModule().entry_index(hou.pwd().parent()) "
        "if hou.pwd().parent().evalParm('remesh_mode') "
        "else hou.pwd().parent().hdaModule().frame_step(hou.pwd().parent(), "
        "hou.pwd().parent().evalParm('topo_frame'))",
        hou.exprLanguage.Python)

    prims = wrangle("build_prims", 0, BUILD_PRIMS_VEX, topo)

    boundary = wrangle("boundary_display", 1, BOUNDARY_VEX, prims)
    surf_switch = asset.createNode("switch", "surface_switch")
    surf_switch.parm("input").setExpression('1 - ch("../surface_only")')
    surf_switch.setNextInput(boundary)   # input 0: surface only
    surf_switch.setNextInput(prims)      # input 1: full volume prims

    frame = asset.createNode("python", "frame_data")
    frame.parm("python").set(FRAME_SOP_CODE)
    frame.addSpareParmTuple(hou.StringParmTemplate("pvdpath", "pvdpath", 1))
    frame.parm("pvdpath").setExpression('chs("../PVD_file")',
                                        hou.exprLanguage.Hscript)
    frame.addSpareParmTuple(hou.FloatParmTemplate("frame", "frame", 1))
    frame.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    frame.setNextInput(surf_switch)
    # Refresh bumps data_epoch when result files were rewritten in place.
    for stage in (topo, frame):
        stage.addSpareParmTuple(hou.IntParmTemplate("dataepoch", "dataepoch", 1))
        stage.parm("dataepoch").setExpression('ch("../data_epoch")',
                                              hou.exprLanguage.Hscript)

    # Cache only what comes from disk: topology, P, and the imported
    # PVD/companion attributes (~1 GB per frame at 2.4M points). Everything
    # else -- derived mechanics included -- is recalculated downstream, which
    # is fast next to a file read, so no setting change (Derived, color,
    # clipping, deformation display, smoothing, visibility, glyphs, fibers)
    # ever makes a cached frame read its file again. Keep the historical node
    # name because callbacks and existing scenes address it directly.
    cache = asset.createNode("cache", "cache1")
    cache.setNextInput(frame)
    # Frames beyond the memory budget are dropped oldest-first instead of
    # growing until the machine swaps.
    cache.parm("maxframes").setExpression(
        "hou.pwd().parent().hdaModule().cache_frame_limit(hou.pwd())",
        hou.exprLanguage.Python)
    # Deduplicate cache entries by output step: under Simulation Time (or past
    # the last step) several Houdini frames show the same PVD step, so shifting
    # cache lookups to the step's canonical frame stores one copy per step.
    step_shift = asset.createNode("timeshift", "step_shift")
    step_shift.setNextInput(cache)
    step_shift.parm("frame").setExpression(
        "hou.pwd().parent().hdaModule().cache_frame(hou.pwd().parent())",
        hou.exprLanguage.Python)
    # Cache Frames selects the cached stream by expression. Bypassing cache1
    # from a callback would need allowEditingOfContents(), which unlocks the
    # instance so it silently stops receiving asset updates.
    cache_switch = asset.createNode("switch", "cache_switch")
    cache_switch.parm("input").setExpression('ch("../cache")')
    cache_switch.setNextInput(frame)            # 0: read every frame
    cache_switch.setNextInput(step_shift)       # 1: cached frames (per step)

    # Mechanics are independent of whether the user draws the rest or
    # displaced positions, so calculate them on the raw PVD geometry.
    # derived_prepare allocates only the needed point attributes; derived runs
    # multi-threaded over points to fill them.
    derived_prepare = wrangle(
        "derived_prepare", 0, DERIVED_PREPARE_VEX, cache_switch)
    derived = wrangle("derived", 2, DERIVED_VEX, derived_prepare)
    derived_switch = asset.createNode("switch", "derived_switch")
    derived_switch.parm("input").setExpression('ch("../derived")')
    derived_switch.setNextInput(cache_switch)
    derived_switch.setNextInput(derived)

    deform = wrangle("deform", 2, DEFORM_VEX, derived_switch)  # class 2 = point

    reference = asset.createNode("python", "reference_comparison")
    reference.parm("python").set(REFERENCE_SOP_CODE)
    reference.addSpareParmTuple(hou.FloatParmTemplate("frame", "frame", 1))
    reference.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    reference.addSpareParmTuple(hou.IntParmTemplate("dataepoch", "dataepoch", 1))
    reference.parm("dataepoch").setExpression('ch("../data_epoch")',
                                              hou.exprLanguage.Hscript)
    reference.setNextInput(deform)

    # Field reduction -> optional nodal averaging -> ramp mapping.
    color_reduce = wrangle("color_reduce", 2, COLOR_REDUCE_VEX, reference)
    field_smoothing = asset.createNode("python", "field_smoothing")
    field_smoothing.parm("python").set(SMOOTH_SOP_CODE)
    field_smoothing.setNextInput(color_reduce)
    smooth_switch = asset.createNode("switch", "smooth_switch")
    smooth_switch.parm("input").setExpression('ch("../smooth_field")')
    smooth_switch.setNextInput(color_reduce)      # 0: raw per-element value
    smooth_switch.setNextInput(field_smoothing)   # 1: nodally averaged
    color = wrangle("color_map", 2, COLOR_MAP_VEX, smooth_switch)

    # Show/hide bodies. Filtering before OUT_result means the range, probe,
    # glyphs, and clip all act on the visible bodies only.
    body_filter = wrangle("body_filter", 1, BODY_FILTER_VEX, color)
    body_filter_pts = wrangle("body_filter_points", 2, BODY_FILTER_POINTS_VEX,
                              body_filter)

    out_result = asset.createNode("null", "OUT_result")
    out_result.setNextInput(body_filter_pts)

    # Glyphs are kept as their own stream (blast everything but the glyph
    # group) so the clip/slice stage below never cuts them.
    glyph_recovery = asset.createNode("python", "glyph_tensor_recovery")
    glyph_recovery.parm("python").set(GLYPH_RECOVERY_SOP_CODE)
    glyph_recovery.setNextInput(out_result)
    glyphs = wrangle("glyphs", 2, GLYPH_VEX, glyph_recovery)
    glyph_only = asset.createNode("blast", "glyph_only")
    glyph_only.setNextInput(glyphs)
    glyph_only.setParms({"group": "readpvd_glyphs", "grouptype": 4,
                         "negate": 1})       # keep only the glyph primitives
    empty_glyphs = asset.createNode("null", "empty_glyphs")
    glyph_switch = asset.createNode("switch", "glyph_switch")
    glyph_switch.parm("input").setExpression(
        'ch("../add_glyphs") && ch("../has_glyph_data")')
    glyph_switch.setNextInput(empty_glyphs)
    glyph_switch.setNextInput(glyph_only)

    # Fibers get their own stream for the same reason glyphs do: a clipped
    # fiber line means nothing, and the lines must not be re-coloured by the
    # field ramp.
    fiber_recovery = asset.createNode("python", "fiber_recovery")
    fiber_recovery.parm("python").set(FIBER_RECOVERY_SOP_CODE)
    fiber_recovery.setNextInput(out_result)
    fibers = wrangle("fibers", 2, FIBER_VEX, fiber_recovery)
    fiber_only = asset.createNode("blast", "fiber_only")
    fiber_only.setNextInput(fibers)
    fiber_only.setParms({"group": "readpvd_fibers", "grouptype": 4,
                         "negate": 1})
    empty_fibers = asset.createNode("null", "empty_fibers")
    fiber_switch = asset.createNode("switch", "fiber_switch")
    fiber_switch.parm("input").setExpression(
        'ch("../show_fibers") && ch("../has_fiber_data")')
    fiber_switch.setNextInput(empty_fibers)
    fiber_switch.setNextInput(fiber_only)

    multi_blocks = asset.createNode("python", "multi_block_display")
    multi_blocks.parm("python").set(MULTI_BLOCK_SOP_CODE)
    multi_blocks.addSpareParmTuple(hou.StringParmTemplate(
        "pvdpath", "pvdpath", 1))
    multi_blocks.parm("pvdpath").setExpression(
        'chs("../PVD_file")', hou.exprLanguage.Hscript)
    multi_blocks.addSpareParmTuple(hou.FloatParmTemplate(
        "frame", "frame", 1))
    multi_blocks.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    multi_blocks.addSpareParmTuple(hou.IntParmTemplate(
        "dataepoch", "dataepoch", 1))
    multi_blocks.parm("dataepoch").setExpression(
        'ch("../data_epoch")', hou.exprLanguage.Hscript)
    empty_multi = asset.createNode("null", "empty_multi_blocks")
    multi_switch = asset.createNode("switch", "multi_block_switch")
    multi_switch.parm("input").setExpression('ch("../show_multi_blocks")')
    multi_switch.setNextInput(empty_multi)
    multi_switch.setNextInput(multi_blocks)

    # Clip/slice the result and supplementary blocks (glyphs excluded).
    # The topology stage keeps every element face and groups the interior
    # ones, so the cached frame serves both views: the outer surface alone,
    # or -- while a clip or slice is active -- every face, so the cut shows
    # values inside the body. Points are kept (probe, range, glyphs).
    interior = wrangle("drop_interior_faces", 1, DROP_INTERIOR_VEX,
                       out_result)
    surface_view = asset.createNode("switch", "surface_view_switch")
    surface_view.parm("input").setExpression(
        'ch("../surface_only") && ch("../clip_mode") == 0')
    surface_view.setNextInput(out_result)   # 0: every face
    surface_view.setNextInput(interior)     # 1: outer surface only

    clip_input = asset.createNode("merge", "clip_input")
    clip_input.setNextInput(surface_view)
    clip_input.setNextInput(multi_switch)
    clip_one = asset.createNode("clip::2.0", "clip_plane")
    clip_one.setNextInput(clip_input)
    clip_two = asset.createNode("clip::2.0", "slice_second_plane")
    clip_two.setNextInput(clip_one)
    for clip_node in (clip_one, clip_two):
        clip_node.parm("dirtype").set(0)
        for axis in "xyz":
            clip_node.parm("dir" + axis).setExpression(
                f'ch("../clip_direction{axis}")', hou.exprLanguage.Hscript)
        clip_node.parm("dofill").setExpression(
            'ch("../clip_fill")', hou.exprLanguage.Hscript)
    for axis in "xyz":
        idx = "xyz".index(axis)
        clip_one.parm("origin" + axis).setExpression(
            "hou.pwd().parent().evalParmTuple('clip_origin')"
            f"[{idx}] - ("
            "hou.Vector3(hou.pwd().parent().evalParmTuple('clip_direction'))"
            f".normalized()[{idx}] * 0.5 * "
            "hou.pwd().parent().evalParm('slice_thickness') "
            "if hou.pwd().parent().evalParm('clip_mode') == 2 else 0.0)",
            hou.exprLanguage.Python)
        clip_two.parm("origin" + axis).setExpression(
            "hou.pwd().parent().evalParmTuple('clip_origin')"
            f"[{idx}] + "
            "hou.Vector3(hou.pwd().parent().evalParmTuple('clip_direction'))"
            f".normalized()[{idx}] * 0.5 * "
            "hou.pwd().parent().evalParm('slice_thickness')",
            hou.exprLanguage.Python)
    clip_one.parm("clipop").setExpression(
        "'above' if hou.pwd().parent().evalParm('clip_mode') == 2 "
        "else hou.pwd().parent().evalParm('clip_keep')",
        hou.exprLanguage.Python)
    clip_two.parm("clipop").set("below")
    section_switch = asset.createNode("switch", "section_switch")
    section_switch.parm("input").setExpression('ch("../clip_mode")')
    section_switch.setNextInput(clip_input)
    section_switch.setNextInput(clip_one)
    section_switch.setNextInput(clip_two)

    # Merge the (possibly clipped) result with the unclipped glyphs.
    display_blocks = asset.createNode("merge", "display_blocks")
    display_blocks.setNextInput(section_switch)
    display_blocks.setNextInput(glyph_switch)
    display_blocks.setNextInput(fiber_switch)

    # Compatibility alias retained for existing callbacks/tools.
    out_src = asset.createNode("null", "OUT_color_source")
    out_src.setNextInput(display_blocks)

    # Renderable scene-space legend. The color bar, tick marks, per-tick
    # numeric labels, and title are all generated in Python (cook_scene_legend)
    # in a single local space, so label spacing always matches the bar.
    legend_bar = asset.createNode("python", "legend_bar")
    legend_bar.parm("python").set(LEGEND_SOP_CODE)
    legend_xform = asset.createNode("xform", "legend_scene_transform")
    legend_xform.setNextInput(legend_bar)
    for axis, index in zip("xyz", range(3)):
        legend_xform.parm("t" + axis).setExpression(
            f'ch("../legend_translate{axis}")', hou.exprLanguage.Hscript)
        legend_xform.parm("r" + axis).setExpression(
            "hou.pwd().parent().hdaModule().legend_rotation("
            f"hou.pwd().parent())[{index}] + "
            f"hou.pwd().parent().evalParm('legend_rotate{axis}')",
            hou.exprLanguage.Python)
        legend_xform.parm("s" + axis).setExpression(
            'ch("../legend_scale")', hou.exprLanguage.Hscript)
    empty_legend = asset.createNode("null", "empty_legend")
    legend_switch = asset.createNode("switch", "legend_scene_switch")
    legend_switch.parm("input").setExpression(
        'if(ch("../legend_mode")==1 || ch("../legend_mode")==3, 1, 0)')
    legend_switch.setNextInput(empty_legend)
    legend_switch.setNextInput(legend_xform)

    # Optional renderable XYZ gnomon.
    gnomon = asset.createNode("python", "gnomon_geometry")
    gnomon.parm("python").set(GNOMON_SOP_CODE)
    empty_gnomon = asset.createNode("null", "empty_gnomon")
    gnomon_switch = asset.createNode("switch", "gnomon_switch")
    gnomon_switch.parm("input").setExpression('ch("../gnomon_show")')
    gnomon_switch.setNextInput(empty_gnomon)
    gnomon_switch.setNextInput(gnomon)


    final_merge = asset.createNode("merge", "display_merge")
    final_merge.setNextInput(out_src)
    final_merge.setNextInput(legend_switch)
    final_merge.setNextInput(gnomon_switch)

    out = asset.createNode("output", "output")
    out.setNextInput(final_merge)
    asset.layoutChildren()

    module = hda_build.read_source("readpvd", "PythonModule.py")
    for marker, source in (("# @VTU_PARSER@", "vtu_parser.py"),
                           ("# @MSH_PARSER@", "msh_parser.py"),
                           ("# @XLSX_WRITER@", "xlsx_writer.py")):
        if marker not in module:
            raise RuntimeError(
                f"Missing {marker} marker in readpvd/PythonModule.py")
        module = module.replace(
            marker, hda_build.read_source("common", source), 1)

    state_name = TYPE_NAME
    definition = asset.type().definition()
    definition.updateFromNode(asset)
    definition.setParmTemplateGroup(_parms())
    definition.addSection("PythonModule", module)
    for section_name, wheel in H5PY_WHEELS.items():
        wheel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vendor", wheel)
        with open(wheel_path, "rb") as wheel_file:
            definition.addSection(
                section_name,
                base64.b64encode(wheel_file.read()).decode("ascii"))
    definition.addSection("OnCreated", ON_CREATED_CODE)
    definition.addSection("Help", hda_build.read_source("readpvd", "Help"))
    definition.addSection(
        "ViewerStateModule",
        hda_build.read_source("readpvd", "ViewerStateModule.py"))
    definition.addSection("DefaultState", state_name)
    definition.addSection(
        "ViewerStateInstall",
        "__import__('viewerstate.utils', fromlist=[None])"
        ".register_pystate_embedded(kwargs['type'])")
    definition.addSection(
        "ViewerStateUninstall",
        "__import__('viewerstate.utils', fromlist=[None])"
        ".unregister_pystate_embedded(kwargs['type'])")
    for section in ("PythonModule", "OnCreated", "ViewerStateModule",
                    "ViewerStateInstall", "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsPython", True)
        definition.setExtraFileOption(f"{section}/IsScript", True)
    for section in ("ViewerStateModule", "ViewerStateInstall",
                    "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsViewerState", True)
    definition.save(definition.libraryFilePath())

    # Hide the inherited object folders (Transform / Render / Misc) from the
    # parameter list. They are merged in by Houdini at save time, so this runs
    # as a second pass on the materialized interface, then re-saves. The parms
    # still exist (transforms/handles keep working); they are just not shown.
    ptg = definition.parmTemplateGroup()
    hidden_any = False
    for folder in ptg.entries():
        if (folder.type() == hou.parmTemplateType.Folder
                and folder.label() in ("Transform", "Render", "Misc")):
            ptg.hide(folder, True)
            hidden_any = True
    if hidden_any:
        definition.setParmTemplateGroup(ptg)
        definition.save(definition.libraryFilePath())
    return hda_build.commit_asset(asset)


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print("built:", build(out_dir))
