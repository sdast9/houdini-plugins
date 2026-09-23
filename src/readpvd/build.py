"""Build readPVD::1.0 (.hdanc). Run under hython via src/build_all.py."""

import base64
import os
import sys

import hou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import hda_build  # noqa: E402

TYPE_NAME = "readPVD::1.0"
LABEL = "Read PVD 1.0 (PolyFEM results)"
H5PY_WHEEL = "h5py-3.16.0-cp313-cp313-macosx_11_0_arm64.whl"
H5PY_SECTION = "h5py-3.16.0-cp313-macos-arm64.whl.b64"

TOPO_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("topoframe")  # cook dependencies
node.parent().hdaModule().cook_topology(node)
"""

FRAME_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("frame")  # cook dependencies
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
node.parent().hdaModule().cook_multi_blocks(node)
"""

REFERENCE_SOP_CODE = """\
node = hou.pwd()
node.evalParm("frame")
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
    int prim = addprim(0, "tet", tet[i+1], tet[i+3], tet[i+2], tet[i]);
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
    for (int f = 0, mask = 1; f < 4; f++, mask *= 2) {
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

if (haspointattrib(0, "solution"))
    f@solution_mag = length(v@solution);

if (haspointattrib(0, "F_1")) {
    // PolyFEM flattens tensors column-major: the exported X_i arrays are
    // the COLUMNS of the tensor, while VEX set(a, b, c) fills rows --
    // transpose to recover the true matrix (verified against PolyFEM's own
    // pk1_stess export, which is asymmetric and exposes the convention).
    matrix3 F = transpose(set(v@F_1, v@F_2, v@F_3));
    3@F_mat = F;
    f@J = determinant(F);

    matrix3 I = ident();

    // Right Cauchy-Green deformation tensor C = F^T F (Lagrangian) and its
    // Lagrangian principal directions, reused for the stretch/strain measures
    // that share those axes (right stretch U, Green-Lagrange, Hencky).
    matrix3 C = transpose(F) * F;
    vector rcg_vals = {0, 0, 0};
    matrix3 rcg_vecs = 0;
    sorted_eigen(C, rcg_vals, rcg_vecs);
    vector lam = sqrt(max(rcg_vals, {0, 0, 0}));   // principal stretches
    v@principal_stretches = lam;
    3@right_cauchy_green = C;
    v@right_cauchy_green_eigenvalues = rcg_vals;
    3@right_cauchy_green_eigenvectors = rcg_vecs;
    3@right_stretch = rebuild_symmetric(lam, rcg_vecs);
    // Material logarithmic (Hencky) strain = 0.5 ln C = sum ln(lambda) n (x) n,
    // defined only where every stretch is positive (J != 0).
    if (abs(f@J) > 1e-12) {
        vector ln_lam = log(max(lam, 1e-20));
        3@hencky_strain = rebuild_symmetric(ln_lam, rcg_vecs);
        v@hencky_strain_eigenvalues = ln_lam;
    } else {
        3@hencky_strain = 0;
        v@hencky_strain_eigenvalues = 0;
    }

    // Left Cauchy-Green deformation tensor B = F F^T (Eulerian) and its
    // Eulerian principal directions, shared with the left stretch V and the
    // Almansi strain.
    matrix3 B = F * transpose(F);
    vector lcg_vals = {0, 0, 0};
    matrix3 lcg_vecs = 0;
    sorted_eigen(B, lcg_vals, lcg_vecs);
    3@left_cauchy_green = B;
    v@left_cauchy_green_eigenvalues = lcg_vals;
    3@left_cauchy_green_eigenvectors = lcg_vecs;
    3@left_stretch = rebuild_symmetric(sqrt(max(lcg_vals, {0, 0, 0})),
                                       lcg_vecs);
    // Almansi (Eulerian) strain e = 0.5 (I - B^-1); defined only where B is
    // invertible (J != 0). Its eigenvalues share B's (Eulerian) axes, so
    // 0.5 (1 - 1/lambda^2) in descending order.
    if (abs(f@J) > 1e-12) {
        3@almansi_strain = 0.5 * (I - invert(B));
        v@almansi_strain_eigenvalues = set(
            0.5 * (1.0 - 1.0 / lcg_vals.x),
            0.5 * (1.0 - 1.0 / lcg_vals.y),
            0.5 * (1.0 - 1.0 / lcg_vals.z));
    } else {
        3@almansi_strain = 0;
        v@almansi_strain_eigenvalues = 0;
    }

    matrix3 green = 0.5 * (C - I);
    3@green_lagrange_strain = green;
    v@green_lagrange_eigenvalues = {0, 0, 0};
    3@green_lagrange_eigenvectors = 0;
    sorted_eigen(green, v@green_lagrange_eigenvalues,
                 3@green_lagrange_eigenvectors);

    matrix3 infinitesimal = 0.5 * (F + transpose(F)) - I;
    3@infinitesimal_strain = infinitesimal;
    v@infinitesimal_strain_eigenvalues = {0, 0, 0};
    3@infinitesimal_strain_eigenvectors = 0;
    sorted_eigen(infinitesimal, v@infinitesimal_strain_eigenvalues,
                 3@infinitesimal_strain_eigenvectors);

    // Cauchy stress from the exported field, or reconstructed from the 1st
    // Piola-Kirchhoff stress (sigma = (1/J) P F^T) when only that was written,
    // so PolyFEM can export just F + pk1 and everything else is derived.
    int have_cauchy = haspointattrib(0, "cauchy_stress_1");
    int have_pk1 = haspointattrib(0, "pk1_stress_1");
    if (have_cauchy || (have_pk1 && abs(f@J) > 1e-12)) {
        matrix3 cauchy;
        if (have_cauchy)
            // symmetric, so the column-major transpose is a no-op; kept for
            // consistency with the F / pk1 assembly convention above
            cauchy = transpose(set(v@cauchy_stress_1, v@cauchy_stress_2,
                                   v@cauchy_stress_3));
        else {
            matrix3 P = transpose(set(v@pk1_stress_1, v@pk1_stress_2,
                                      v@pk1_stress_3));
            cauchy = (1.0 / f@J) * P * transpose(F);
        }
        3@cauchy_mat = cauchy;
        v@cauchy_eigenvalues = {0, 0, 0};
        3@cauchy_eigenvectors = 0;
        sorted_eigen(cauchy, v@cauchy_eigenvalues, 3@cauchy_eigenvectors);

        f@cauchy_trace = matrix_trace(cauchy);
        f@hydrostatic_stress = f@cauchy_trace / 3.0;
        matrix3 dev = cauchy - f@hydrostatic_stress * I;
        3@deviatoric_stress = dev;
        f@stress_J2 = 0.5 * matrix_norm_squared(dev);
        f@stress_J3 = determinant(dev);
        f@von_mises_derived = sqrt(max(0.0, 3.0 * f@stress_J2));
        f@max_shear_stress =
            0.5 * (v@cauchy_eigenvalues.x - v@cauchy_eigenvalues.z);
        f@stress_triaxiality = f@von_mises_derived > 1e-12
            ? f@hydrostatic_stress / f@von_mises_derived : 0.0;
        if (!haspointattrib(0, "von_mises"))
            f@von_mises = f@von_mises_derived;

        if (abs(f@J) > 1e-12) {
            matrix3 Finv = invert(F);
            3@pk1 = f@J * cauchy * transpose(Finv);
            3@pk2 = f@J * Finv * cauchy * transpose(Finv);
        } else {
            3@pk1 = 0;
            3@pk2 = 0;
        }
        v@pk2_eigenvalues = {0, 0, 0};
        3@pk2_eigenvectors = 0;
        sorted_eigen(3@pk2, v@pk2_eigenvalues, 3@pk2_eigenvectors);
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
            vector evals = {0, 0, 0};
            matrix3 evecs = 0;
            sorted_eigen(m, evals, evecs);
            int component = reduction == "principal_max" ? 0
                : reduction == "principal_middle" ? 1 : 2;
            value = getcomp(evals, component);
        } else if (reduction == "x" || reduction == "y" || reduction == "z") {
            int component = reduction == "x" ? 0 : reduction == "y" ? 1 : 2;
            value = getcomp(m, component, component);
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
string tokens[] = split(chs("../visible_bodies"));
if (len(tokens) == 0) return;
float body_value = point(0, "body_ids", primpoint(0, @primnum, 0));
int body = (int)rint(body_value);
foreach (string token; tokens)
    if (atoi(token) == body) return;
removeprim(0, @primnum, 1);
"""

FIBER_VEX = """\
// One line per physical node per family. PolyFEM exports the unchanged
// simulation/world reference direction a0; geometry[].transformation never
// rotates it. Fibers are LINE fields (a0 and -a0 are the same fiber), so
// colour by |direction| and never by sign.
if (chi("../show_fibers") == 0) return;
if (haspointattrib(0, "coincident_rep")
    && point(0, "coincident_rep", @ptnum) < 0.5)
    return;
if (@ptnum % max(1, chi("../fiber_stride")) != 0) return;

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
} else if (haspointattrib(0, "F_1")) {
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
    && point(0, "coincident_rep", @ptnum) < 0.5)
    return;
if (@ptnum % max(1, chi("../glyph_stride")) != 0) return;
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
    int a = addpoint(0, v@P - dir * mag * 0.5);
    int b = addpoint(0, v@P + dir * mag * 0.5);
    setpointattrib(0, "Cd", a, color, "set");
    setpointattrib(0, "Cd", b, color, "set");
    int prim = addprim(0, "polyline", a, b);
    setprimgroup(0, "readpvd_glyphs", prim, 1, "set");
    if (chi("../glyph_arrowheads") && mag > 1e-12) {
        vector helper = abs(dir.z) < 0.9 ? {0, 0, 1} : {0, 1, 0};
        vector side = normalize(cross(dir, helper))
            * mag * ch("../glyph_arrow_size");
        vector back = point(0, "P", b) - dir * mag
            * ch("../glyph_arrow_size") * 2;
        int c = addpoint(0, back + side);
        int d = addpoint(0, back - side);
        setpointattrib(0, "Cd", c, color, "set");
        setpointattrib(0, "Cd", d, color, "set");
        int arrow_c = addprim(0, "polyline", b, c);
        int arrow_d = addprim(0, "polyline", b, d);
        setprimgroup(0, "readpvd_glyphs", arrow_c, 1, "set");
        setprimgroup(0, "readpvd_glyphs", arrow_d, 1, "set");
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
        help="Re-read the PVD file (e.g. while a simulation is still writing "
             "steps), rebuild the field menus, and set the playbar range to "
             "the available time steps."))
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
    main.addParmTemplate(hou.StringParmTemplate(
        "availability_status", "Available Data", 1,
        default_value=("Select a PVD file to inspect its renderable data.",),
        help="Read-only summary of what the loaded result contains (blocks, "
             "fields, bodies, fibers) and therefore which controls below are "
             "usable."))
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
    main.addParmTemplate(hou.IntParmTemplate(
        "topo_frame", "Topology Frame", 1, default_value=(0,),
        help="When Remeshing Mode is off, the mesh connectivity is read once "
             "from this frame and reused for every other frame (only "
             "positions and fields change). Leave at 0 unless the first "
             "frame is missing or damaged."))
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
    cache_status.setConditional(hou.parmCondType.DisableWhen, "{ 1 == 1 }")
    main.addParmTemplate(cache_status)
    memory_status = hou.StringParmTemplate(
        "memory_status", "Memory", 1,
        default_expression=(
            "hou.pwd().hdaModule().memory_status_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: memory used by Houdini, and the computer's used, "
             "total and available memory (available = what can still be "
             "handed out without swapping).")
    memory_status.setConditional(hou.parmCondType.DisableWhen, "{ 1 == 1 }")
    main.addParmTemplate(memory_status)
    cache_epoch = hou.IntParmTemplate(
        "cache_epoch", "cache_epoch", 1, default_value=(0,),
        help="Internal, hidden: counts Clear Cache presses so Cache Status "
             "refreshes immediately.")
    cache_epoch.setConditional(hou.parmCondType.HideWhen, "{ 1 == 1 }")
    main.addParmTemplate(cache_epoch)
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
             "and looks the same from outside; turn off when you need to "
             "clip into the interior or probe interior points."))
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
             "value (principal values are ordered maximum to minimum), trace "
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
    disp.addParmTemplate(hou.StringParmTemplate(
        "color_status", "Color Field Status", 1,
        default_value=("Cook the result to validate the selected field.",),
        help="Read-only: whether the chosen Color Field was found in the "
             "current frame, and the reduction that is actually displayed."))
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
    disp.addParmTemplate(hou.FloatParmTemplate(
        "range_percentiles", "Range Percentiles", 2,
        default_value=(1.0, 99.0), min=0.0, max=100.0,
        help="Lower and upper percentiles used by Ignore Range Outliers: "
             "1 and 99 map the color range to the middle 98% of the values."))
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "range_symmetric", "Symmetric Range Around Zero", default_value=False,
        help="Make the displayed range symmetric around zero (e.g. -5 to +5) "
             "so that a diverging ramp puts zero exactly in the middle. "
             "Useful for signed fields such as principal stresses."))
    disp.addParmTemplate(hou.FloatParmTemplate(
        "color_min", "Displayed Minimum", 1, default_value=(0.0,),
        help="Value mapped to the left end of the color ramp. Values below "
             "it are clamped to that color. Set by the Auto Range buttons or "
             "type your own."))
    disp.addParmTemplate(hou.FloatParmTemplate(
        "color_max", "Displayed Maximum", 1, default_value=(1.0,),
        help="Value mapped to the right end of the color ramp. Values above "
             "it are clamped to that color."))
    disp.addParmTemplate(hou.ButtonParmTemplate(
        "autoscale", "Auto Range: Current Frame",
        script_callback="hou.phm().autoscale(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Set Displayed Minimum/Maximum from the values in the frame "
             "currently shown (instant). Respects Ignore Range Outliers and "
             "Symmetric Range; blocked by Lock Displayed Range."))
    disp.addParmTemplate(hou.ButtonParmTemplate(
        "autoscale_all", "Auto Range: All Frames (scans sequence)",
        script_callback="hou.phm().autoscale_all(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Scans the selected field over the complete PVD sequence to find "
             "the global range. This must read every frame once, so it can "
             "take a while for long sequences of large meshes -- unavoidable, "
             "the values live inside each file. Use Auto Range: Current Frame "
             "for an instant range from the displayed frame."))
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
    glyph_tensor.setConditional(
        hou.parmCondType.DisableWhen, "{ has_glyph_data == 0 }")
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
        direction.setConditional(
            hou.parmCondType.DisableWhen, "{ has_glyph_data == 0 }")
        direction_folder.addParmTemplate(direction)
    ana.addParmTemplate(direction_folder)
    ana.addParmTemplate(hou.FloatParmTemplate(
        "tensor_scale", "Principal-Value Length Scale", 1,
        default_value=(0.01,),
        help="Multiplier from principal value to line length (scene units "
             "per unit of the tensor). Press Auto-Estimate Glyph Scale to get "
             "a value that fits the mesh; then adjust by eye."))
    ana.addParmTemplate(hou.MenuParmTemplate(
        "glyph_length_mode", "Glyph Length Mode",
        ("value", "normalized", "clamped"),
        menu_labels=("Scale by Principal Value", "Normalized Directions",
                     "Scale by Value with Maximum Length"),
        default_value=0,
        help="How long each glyph line is. Scale by Principal Value: length "
             "proportional to the magnitude (shows where the field is "
             "strong). Normalized: all lines the same length (shows only "
             "direction). Scale by Value with Maximum Length: proportional "
             "but capped, so a few huge values do not hide everything else."))
    ana.addParmTemplate(hou.FloatParmTemplate(
        "glyph_max_length", "Maximum Glyph Length", 1, default_value=(1.0,),
        min=0.0, max=1e9,
        help="Longest line allowed (scene units) when Glyph Length Mode is "
             "Scale by Value with Maximum Length."))
    for name, label, which in (
            ("glyph_scale_first", "First Direction Length Multiplier",
             "largest"),
            ("glyph_scale_second", "Second Direction Length Multiplier",
             "middle"),
            ("glyph_scale_third", "Third Direction Length Multiplier",
             "smallest")):
        ana.addParmTemplate(hou.FloatParmTemplate(
            name, label, 1, default_value=(1.0,), min=0.0, max=1e9,
            help=f"Extra length factor for the glyphs of the {which} "
                 "principal value only, e.g. to emphasize one direction "
                 "(1 = no change)."))
    ana.addParmTemplate(hou.ToggleParmTemplate(
        "glyph_arrowheads", "Show Glyph Arrowheads", default_value=False,
        help="Draw a small arrowhead at both ends of each glyph line so the "
             "direction reads clearly from a distance."))
    ana.addParmTemplate(hou.FloatParmTemplate(
        "glyph_arrow_size", "Glyph Arrowhead Size", 1,
        default_value=(0.08,), min=0.001, max=0.5,
        help="Size of the arrowheads as a fraction of the glyph length."))
    ana.addParmTemplate(hou.IntParmTemplate(
        "glyph_stride", "Glyph Sampling Stride", 1, default_value=(1,),
        min=1, max=100000,
        help="Draw a glyph only at every Nth point, to thin them out on "
             "dense meshes (1 = every point)."))
    ana.addParmTemplate(hou.ToggleParmTemplate(
        "glyph_sign_color", "Stress Sign Colors", default_value=True,
        help="For stress tensors, color glyphs red where the principal "
             "stress is tension (positive) and blue where it is compression "
             "(negative). Strain tensors use the uniform color."))
    glyph_color = hou.FloatParmTemplate(
        "glyph_color", "Uniform Glyph Color", 3, default_value=(1, 1, 1),
        help="Color of the glyph lines when Stress Sign Colors is off (or "
             "for strain tensors, which have no sign coloring).")
    ana.addParmTemplate(glyph_color)
    ana.addParmTemplate(hou.ButtonParmTemplate(
        "autoglyph", "Auto-Estimate Glyph Scale",
        script_callback="hou.phm().autoglyph(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Size the glyphs to the mesh: the largest glyphs span about half "
             "an average edge. For value-scaled glyphs the estimate accounts "
             "for the selected tensor's principal-value magnitude, so it works "
             "for stress and strain alike."))
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
            hou.parmCondType.DisableWhen, "{ show_fibers == 0 }")
        fiber_folder.addParmTemplate(template)

    fiber_folder.addParmTemplate(hou.ButtonParmTemplate(
        "autofiber", "Auto-Estimate Fiber Scale",
        script_callback="hou.phm().autofiber(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Size the fiber lines to the mesh (about half an average edge)."))
    fiber_folder.addParmTemplate(hou.StringParmTemplate(
        "fiber_attribs", "fiber_attribs", 1,
        default_value=("",),
        help="Internal: the fiber vector attributes assembled on load."))
    ana.addParmTemplate(fiber_folder)
    ptg.append(ana)

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
        folder.addParmTemplate(block_ramp)
        multi.addParmTemplate(folder)
    ptg.append(multi)

    compare = hou.FolderParmTemplate("comparison_folder", "Comparison",
                                     folder_type=hou.folderType.Tabs)
    compare.addParmTemplate(hou.ToggleParmTemplate(
        "reference_enable", "Compare With Reference Frame",
        default_value=False,
        help="Colors the active field by its change from a selected frame. "
             "Requires matching point topology. Parses the reference frame in "
             "addition to the current one, so each frame change does a little "
             "more work while this is on."))
    compare.addParmTemplate(hou.IntParmTemplate(
        "reference_frame", "Reference Frame", 1, default_value=(0,), min=0,
        help="Houdini frame to compare against (0 is usually the undeformed "
             "start). The mesh must have the same points in both frames."))
    compare.addParmTemplate(hou.MenuParmTemplate(
        "reference_mode", "Comparison Value",
        ("difference", "absolute", "percent"),
        menu_labels=("Current Minus Reference", "Absolute Difference",
                     "Percentage Change"), default_value=0,
        help="What to color: the signed change since the reference frame, "
             "its magnitude, or the change as a percentage of the reference "
             "value."))
    compare.addParmTemplate(hou.StringParmTemplate(
        "reference_status", "Reference Comparison Status", 1,
        default_value=("Comparison is disabled.",),
        help="Read-only: whether the reference frame could be loaded and "
             "matched to the current one."))
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
    probe.addParmTemplate(hou.IntParmTemplate(
        "probe_point", "Probed Point Number", 1, default_value=(-1,), min=-1,
        help="Point number of the last probed vertex (-1 = none). The marker "
             "follows this point as frames change; you can also type a point "
             "number to probe it directly."))
    probe.addParmTemplate(hou.StringParmTemplate(
        "probe_readout", "Probe Readout", 1,
        default_value=("Enable the viewport probe and click the result.",),
        help="Read-only text with the probed point's position, displacement, "
             "field value, body and sideset ids, principal values and "
             "reference comparison."))
    ptg.append(probe)

    section = hou.FolderParmTemplate("section_folder", "Clip / Slice",
                                     folder_type=hou.folderType.Tabs)
    section.addParmTemplate(hou.MenuParmTemplate(
        "clip_mode", "Section Mode", ("off", "clip", "slice"),
        menu_labels=("Off", "Clipping Plane", "Thin Slice"),
        default_value=0,
        help="Cut into the model to see inside. Clipping Plane removes one "
             "side of a plane; Thin Slice keeps only a slab around the plane. "
             "Glyphs and fibers are added after the cut and are never "
             "removed by it."))
    section.addParmTemplate(hou.FloatParmTemplate(
        "clip_origin", "Plane Origin", 3, default_value=(0, 0, 0),
        help="Auto-set to the scene's bounding-box center when a PVD file "
             "is loaded (only while still at 0,0,0; your edits are kept)."))
    section.addParmTemplate(hou.FloatParmTemplate(
        "clip_direction", "Plane Normal", 3, default_value=(1, 0, 0),
        help="Auto-aligned to the scene's longest bounding-box axis when a "
             "PVD file is loaded (only while still at the +X default; your "
             "edits are kept)."))
    section.addParmTemplate(hou.MenuParmTemplate(
        "clip_keep", "Clip Side To Keep", ("above", "below"),
        menu_labels=("Above Plane", "Below Plane"), default_value=0,
        help="Which side of the clipping plane stays visible (above = the "
             "side the Plane Normal points to)."))
    section.addParmTemplate(hou.ToggleParmTemplate(
        "clip_fill", "Fill Clipped Surface", default_value=True,
        help="Creates a colored cut surface when clipping closed geometry."))
    section.addParmTemplate(hou.FloatParmTemplate(
        "slice_thickness", "Slice Thickness", 1, default_value=(0.01,),
        min=1e-8, max=1e9,
        help="Thickness of the kept slab (scene units) in Thin Slice mode."))
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
    legend.addParmTemplate(hou.StringParmTemplate(
        "legend_title", "Custom Legend Title", 1, default_value=("",),
        help="Leave empty to use the resolved color field and displayed value."))
    legend.addParmTemplate(hou.IntParmTemplate(
        "legend_ticks", "Number of Tick Labels", 1, default_value=(6,),
        min=2, max=21,
        help="How many labeled values are written along the color bar "
             "(including both ends)."))
    legend.addParmTemplate(hou.IntParmTemplate(
        "legend_digits", "Significant Digits", 1, default_value=(5,),
        min=1, max=12,
        help="Number of significant digits shown in the legend labels and "
             "probe values."))
    legend.addParmTemplate(hou.MenuParmTemplate(
        "legend_number_format", "Legend Number Format",
        ("automatic", "scientific", "engineering", "fixed"),
        menu_labels=("Automatic", "Scientific", "Engineering", "Fixed"),
        default_value=0,
        help="How legend numbers are written: Automatic picks per value; "
             "Scientific uses powers of ten (1.2e+05); Engineering uses "
             "powers of a thousand (120e+03); Fixed uses plain decimals."))
    legend.addParmTemplate(hou.StringParmTemplate(
        "field_units", "Displayed Field Units", 1, default_value=("",),
        help="Optional units appended to legend titles and probe values."))
    text_color = hou.FloatParmTemplate(
        "legend_text_color", "Legend Text Color", 3, default_value=(1, 1, 1),
        help="Color of the legend title and tick labels (scene legend and "
             "viewport overlay).")
    legend.addParmTemplate(text_color)
    legend.addParmTemplate(hou.MenuParmTemplate(
        "legend_orientation", "Scene Legend Plane and Up Direction",
        tuple(str(i) for i in range(12)),
        menu_labels=("XY Plane, +X Up", "XY Plane, -X Up",
                     "XY Plane, +Y Up", "XY Plane, -Y Up",
                     "YZ Plane, +Y Up", "YZ Plane, -Y Up",
                     "YZ Plane, +Z Up", "YZ Plane, -Z Up",
                     "XZ Plane, +X Up", "XZ Plane, -X Up",
                     "XZ Plane, +Z Up", "XZ Plane, -Z Up"),
        default_value=2,
        help="Which world plane the scene legend lies in and which way its "
             "'up' points. Choose the plane facing your usual camera."))
    legend.addParmTemplate(hou.FloatParmTemplate(
        "legend_translate", "Scene Legend Translation", 3,
        default_value=(0, 0, 0),
        help="Position of the scene legend geometry (scene units). Place "
             "Scene Legend Beside Model sets it automatically."))
    legend_rotate = hou.FloatParmTemplate(
        "legend_rotate", "Additional Scene Legend Rotation", 3,
        default_value=(0, 0, 0),
        help="Extra rotation (degrees) applied to the scene legend after the "
             "plane/up choice, for fine adjustment.")
    legend_rotate.setLook(hou.parmLook.Angle)
    legend.addParmTemplate(legend_rotate)
    legend.addParmTemplate(hou.FloatParmTemplate(
        "legend_scale", "Scene Legend Scale", 1, default_value=(1.0,),
        min=0.0001, max=1000,
        help="Overall size of the scene legend geometry."))
    legend.addParmTemplate(hou.ButtonParmTemplate(
        "legend_autoplace", "Place Scene Legend Beside Model",
        script_callback="hou.phm().autoplace_legend(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Move and size the scene legend so it sits next to the model's "
             "bounding box at a readable size."))
    legend.addParmTemplate(hou.MenuParmTemplate(
        "overlay_corner", "Overlay Screen Corner",
        ("upper_left", "upper_right", "lower_left", "lower_right"),
        menu_labels=("Upper Left", "Upper Right", "Lower Left", "Lower Right"),
        default_value=1,
        help="Corner of the viewport where the screen-fixed overlay legend "
             "is drawn."))
    legend.addParmTemplate(hou.IntParmTemplate(
        "overlay_margin", "Overlay Margin (Pixels)", 1, default_value=(24,),
        min=0, max=1000,
        help="Distance in pixels between the overlay legend and the "
             "viewport edges."))
    legend.addParmTemplate(hou.IntParmTemplate(
        "overlay_width", "Overlay Bar Width (Pixels)", 1, default_value=(28,),
        min=8, max=500,
        help="Width of the overlay color bar in pixels."))
    legend.addParmTemplate(hou.IntParmTemplate(
        "overlay_height", "Overlay Bar Height (Pixels)", 1,
        default_value=(260,), min=40, max=2000,
        help="Height of the overlay color bar in pixels."))
    ptg.append(legend)

    gnomon = hou.FolderParmTemplate("gnomon_folder", "Gnomon",
                                    folder_type=hou.folderType.Tabs)
    gnomon.addParmTemplate(hou.ToggleParmTemplate(
        "gnomon_show", "Show Scene Gnomon", default_value=False,
        help="Adds a renderable XYZ orientation marker to the final geometry."))
    gnomon.addParmTemplate(hou.FloatParmTemplate(
        "gnomon_center", "Gnomon Center", 3, default_value=(0, 0, 0),
        help="World position of the gnomon's origin (scene units)."))
    gnomon.addParmTemplate(hou.FloatParmTemplate(
        "gnomon_scale", "Gnomon Axis Length", 1, default_value=(1.0,),
        min=0.0001, max=1000,
        help="Length of each gnomon axis line (scene units)."))
    for name, label, what in (
            ("gnomon_x", "Show X Axis", "Draw the red X axis line."),
            ("gnomon_y", "Show Y Axis", "Draw the green Y axis line."),
            ("gnomon_z", "Show Z Axis", "Draw the blue Z axis line."),
            ("gnomon_arrows", "Show Arrowheads",
             "Draw arrowheads at the tip of each axis."),
            ("gnomon_center_marker", "Show Center Marker",
             "Draw a small marker at the gnomon origin.")):
        gnomon.addParmTemplate(hou.ToggleParmTemplate(
            name, label, default_value=True, help=what))
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
        flag.setConditional(hou.parmCondType.HideWhen, "{ 1 == 1 }")
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
    topo.parm("topoframe").setExpression(
        "hou.frame() if hou.pwd().parent().evalParm('remesh_mode') "
        "else hou.pwd().parent().evalParm('topo_frame')",
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
    # Cache Frames selects the cached stream by expression. Bypassing cache1
    # from a callback would need allowEditingOfContents(), which unlocks the
    # instance so it silently stops receiving asset updates.
    cache_switch = asset.createNode("switch", "cache_switch")
    cache_switch.parm("input").setExpression('ch("../cache")')
    cache_switch.setNextInput(frame)            # 0: read every frame
    cache_switch.setNextInput(cache)            # 1: cached frames

    # Mechanics are independent of whether the user draws the rest or
    # displaced positions, so calculate them on the raw PVD geometry.
    derived = wrangle("derived", 2, DERIVED_VEX, cache_switch)
    derived_switch = asset.createNode("switch", "derived_switch")
    derived_switch.parm("input").setExpression('ch("../derived")')
    derived_switch.setNextInput(cache_switch)
    derived_switch.setNextInput(derived)

    deform = wrangle("deform", 2, DEFORM_VEX, derived_switch)  # class 2 = point

    reference = asset.createNode("python", "reference_comparison")
    reference.parm("python").set(REFERENCE_SOP_CODE)
    reference.addSpareParmTuple(hou.FloatParmTemplate("frame", "frame", 1))
    reference.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
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

    out_result = asset.createNode("null", "OUT_result")
    out_result.setNextInput(body_filter)

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
        clip_one.parm("origin" + axis).setExpression(
            f'ch("../clip_origin{axis}")', hou.exprLanguage.Hscript)
        clip_two.parm("origin" + axis).setExpression(
            "hou.pwd().parent().evalParmTuple('clip_origin')"
            f"[{'xyz'.index(axis)}] + "
            "hou.Vector3(hou.pwd().parent().evalParmTuple('clip_direction'))"
            ".normalized()"
            f"[{'xyz'.index(axis)}] * "
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

    module = hda_build.read_source("readpvd", "PythonModule.py").replace(
        "# @VTU_PARSER@", hda_build.read_source("common", "vtu_parser.py"))

    state_name = TYPE_NAME
    definition = asset.type().definition()
    definition.updateFromNode(asset)
    definition.setParmTemplateGroup(_parms())
    definition.addSection("PythonModule", module)
    wheel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vendor", H5PY_WHEEL)
    with open(wheel_path, "rb") as wheel_file:
        definition.addSection(
            H5PY_SECTION,
            base64.b64encode(wheel_file.read()).decode("ascii"))
    definition.addSection("OnCreated", ON_CREATED_CODE)
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
    return definition.libraryFilePath()


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print("built:", build(out_dir))
