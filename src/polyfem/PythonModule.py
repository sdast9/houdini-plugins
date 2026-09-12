# PolyFEM 2.0 HDA module.
#
# Targets the semi-implicit-barrier PolyFEM build (strict JSON validation):
#   * barrier_stiffness: "semi_implicit" | "adaptive" | number
#   * /solver/contact/semi_implicit/* and augmented_lagrangian "hessian_scaled"
#
# Engineering changes vs 1.2:
#   * collision-free ID scheme: volume id = 1000*geo + vol,
#     boundary id = (1000*geo + vol)*10000 + sideset*100 + bc,
#     obstacle id = 100000 + 1000*geo
#   * sideset export is numpy-bulk from VEX-stamped face provenance
#     (msh vertex ids stored per boundary face at build time); separate
#     files per face arity so mixed tet/hex meshes export correctly
#   * one Entity-colored surface chain per geometry instead of per-volume
#     node forests (scales to millions of elements); each subdomain's sideset
#     group nodes sit on a dead-end branch behind a Visibility SOP that hides
#     the other subdomains, which is what 1.2's per-volume split branches gave
#     for free -- picking stays scoped to one subdomain without renumbering
#     the geometry-wide prims that basegroup and the export refer to
#   * native polyfem selections can be typed into any sideset group field:
#     "axis:+z:0.95", "box:[0,0,0],[1,1,1]", "sphere:[0,0,0],0.5",
#     "plane:[0,0,1],[0,0,0.5]"  (normal, point) -- emitted directly into
#     the json, no selection files needed
#   * ast.literal_eval instead of eval for vector parms
#   * background run with log capture, alongside the terminal launch

import ast
import json
import math
import os
import platform
import re
import shutil
import subprocess

import numpy as np

import hou

# =============================================================================
# VEX snippets (embedded into wrangles by the node-tree builders)
# =============================================================================

# Extract boundary faces of tets/hexes with provenance: per face we stamp the
# gmsh vertex ids (msh_pt_id - 1, polyfem's 0-based node ids) so sideset
# export is a bulk attribute read instead of per-prim python. Volume prims
# are removed afterwards -- this wrangle's output is the display/selection
# surface only.
SURFACE_VEX = """
if (@primnum == 0) {
    addattrib(0, "prim", "fp0", -1);
    addattrib(0, "prim", "fp1", -1);
    addattrib(0, "prim", "fp2", -1);
    addattrib(0, "prim", "fp3", -1);
}

int vcount = primvertexcount(0, @primnum);
if (vcount == 8) {
    for (int f = 0; f < 6; f++) {
        if (hex_adjacent(0, @primnum, f) == -1) {
            int pts[]; int ids[];
            for (int j = 0; j < 4; j++) {
                int p = primpoint(0, @primnum, hex_faceindex(f, j));
                append(pts, p);
                append(ids, point(0, "msh_pt_id", p) - 1);
            }
            int prim = addprim(0, "poly", pts[0], pts[1], pts[2], pts[3]);
            setprimattrib(0, "Entity", prim, i@Entity, "set");
            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");
            setprimattrib(0, "fp0", prim, ids[0], "set");
            setprimattrib(0, "fp1", prim, ids[1], "set");
            setprimattrib(0, "fp2", prim, ids[2], "set");
            setprimattrib(0, "fp3", prim, ids[3], "set");
        }
    }
} else if (vcount == 4) {
    for (int f = 0; f < 4; f++) {
        if (tet_adjacent(0, @primnum, f) == -1) {
            int pts[]; int ids[];
            for (int j = 0; j < 3; j++) {
                int p = primpoint(0, @primnum, tet_faceindex(f, j));
                append(pts, p);
                append(ids, point(0, "msh_pt_id", p) - 1);
            }
            int prim = addprim(0, "poly", pts[0], pts[1], pts[2]);
            setprimattrib(0, "Entity", prim, i@Entity, "set");
            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");
            setprimattrib(0, "fp0", prim, ids[0], "set");
            setprimattrib(0, "fp1", prim, ids[1], "set");
            setprimattrib(0, "fp2", prim, ids[2], "set");
        }
    }
}
removeprim(0, @primnum, 1);
"""

# Per-prim minimum edge length (prim wrangle; parallel-safe). The button
# callback reduces over the prim attribute with numpy.
MIN_EDGE_VEX = """
float m = 1e30;
int pts[] = primpoints(0, @primnum);
int n = len(pts);
for (int i = 0; i < n; i++) {
    vector a = point(0, "P", pts[i]);
    vector b = point(0, "P", pts[(i + 1) % n]);
    m = min(m, distance(a, b));
}
f@min_edge_length = m;
"""

# =============================================================================
# IDs and small utilities
# =============================================================================



def _message(text, severity=None):
    """UI dialog in the GUI, stdout in hython/headless runs."""
    if hou.isUIAvailable():
        hou.ui.displayMessage(
            str(text), severity=severity or hou.severityType.Warning)
    else:
        print(f"[PolyFEM HDA] {text}")


def _status(text):
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(str(text), hou.severityType.ImportantMessage)
    else:
        print(f"[PolyFEM HDA] {text}")


def vol_id(geo, vol):
    """Collision-free volume/body id (geo, vol are 1-based)."""
    return 1000 * int(geo) + int(vol)


def boundary_id(geo, vol, sideset, bc):
    return vol_id(geo, vol) * 10000 + int(sideset) * 100 + int(bc)


def obstacle_id(geo):
    return 100000 + 1000 * int(geo)


def bool_check(var):
    return bool(var)


def parse_vector(text, name):
    """Parse a user vector/scalar parm. Numeric literals via ast; entries that
    are expression strings (polyfem supports them) pass through unchanged."""
    text = text.strip()
    if not text:
        raise hou.NodeError(f"Empty value for {name}")
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    # allow ["0", "-0.1*t", "0"] style lists with expression strings
    if text.startswith("["):
        inner = text.strip("[]")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(ast.literal_eval(p))
            except (ValueError, SyntaxError):
                out.append(p.strip("'\""))
        return out
    return text  # single expression string


def colors():
    return {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 4: (1, 1, 0),
            5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 0.5, 0), 8: (0.5, 0, 1),
            9: (0, 0.5, 0.5), 10: (0.6, 0.3, 0.1)}


def default_color(geo, vol=1):
    palette = colors()
    base = palette[(int(geo) - 1) % len(palette) + 1]
    return tuple(channel / int(vol) for channel in base)


def _color_tuple(parent, name):
    parm = parent.parmTuple(name)
    return tuple(parm.eval()) if parm is not None else None


def _set_color_tuple(parent, name, value):
    parm = parent.parmTuple(name)
    if parm is not None:
        parm.set(value)


# Perceptually-ordered ramp for scalar material fields. Shared with readPVD's
# kappa coloring so the same value reads as the same colour before and after
# a solve.
_KAPPA_RAMP = (
    "{0.267, 0.005, 0.329}", "{0.229, 0.322, 0.545}", "{0.128, 0.567, 0.551}",
    "{0.369, 0.789, 0.383}", "{0.993, 0.906, 0.144}")


def _entity_color_vex(geo, num_vols, mode="subdomains", fiber_attribs=(),
                      kappa_attribs=()):
    """One wrangle per geometry, driven by the visible subdomain swatches.

    The alternate modes recolor the same faces by the per-element material
    data copied onto them, so a fiber field can be inspected on the surface
    without leaving the setup.
    """
    alpha = f"""\
f@Alpha = chi("../show_fibers{int(geo)}")
    ? clamp(chf("../fiber_surface_alpha{int(geo)}"), 0.0, 1.0)
    : 1.0;
"""
    channels = ",\n    ".join(
        f'chv("../color_{int(geo)}_{vol}")' for vol in range(1, num_vols + 1))
    subdomain = f"""\
vector entity_colors[] = array(
    {channels});
int color_index = i@Entity - 1;
v@Cd = color_index >= 0 && color_index < len(entity_colors)
    ? entity_colors[color_index]
    : {{0.5, 0.5, 0.5}};
"""
    if mode == "fiber_rgb" and fiber_attribs:
        lookups = "\n".join(
            f'if (length(d) < 1e-12 && hasprimattrib(0, "{name}")) '
            f'd = prim(0, "{name}", @primnum);' for name in fiber_attribs)
        return f"""\
vector d = {{0, 0, 0}};
{lookups}
v@Cd = length(d) > 1e-12 ? abs(normalize(d)) : {{0.5, 0.5, 0.5}};
""" + alpha
    if mode == "kappa" and kappa_attribs:
        lookups = "\n".join(
            f'if (k < 0 && hasprimattrib(0, "{name}")) '
            f'k = prim(0, "{name}", @primnum);' for name in kappa_attribs)
        ramp = ", ".join(_KAPPA_RAMP)
        return f"""\
float k = -1;
{lookups}
if (k < 0) {{
    v@Cd = {{0.5, 0.5, 0.5}};
}} else {{
    vector ramp[] = array({ramp});
    // kappa is bounded by 1/d (1/3 in 3D); fix the range so the same value
    // reads as the same colour on every geometry.
    float u = clamp(k / (1.0 / 3.0), 0.0, 1.0) * (len(ramp) - 1);
    int lo = int(floor(u));
    int hi = min(lo + 1, len(ramp) - 1);
    v@Cd = lerp(ramp[lo], ramp[hi], u - lo);
}}
""" + alpha
    return subdomain + alpha


def expand_group_str(group_input):
    """'0 2 5-8' -> [0, 2, 5, 6, 7, 8] (Houdini group pattern subset)."""
    out = []
    for tokenized in str(group_input).replace(",", " ").split():
        m = re.fullmatch(r"(\d+)-(\d+)", tokenized)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tokenized.isdigit():
            out.append(int(tokenized))
    return out


def list_to_space_str(numbers):
    """Compact int list -> Houdini group pattern with ranges."""
    if not len(numbers):
        return ""
    numbers = sorted(set(int(n) for n in numbers))
    parts = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        start = prev = n
    parts.append(f"{start}-{prev}" if prev > start else str(start))
    return " ".join(parts)


# native selection syntax: axis:+z:0.95 | box:[...],[...] | sphere:[c],r |
# plane:[normal],[point]
_NATIVE_SELECTION_RE = re.compile(r"^\s*(axis|box|sphere|plane)\s*:", re.I)


def is_native_selection(pattern):
    return bool(_NATIVE_SELECTION_RE.match(pattern or ""))


def parse_native_selection(pattern, sel_id, name):
    kind, _, rest = pattern.partition(":")
    kind = kind.strip().lower()
    rest = rest.strip()
    try:
        if kind == "axis":
            axis, _, position = rest.partition(":")
            return {"id": sel_id, "axis": axis.strip(),
                    "position": float(position)}
        payload = ast.literal_eval("[" + rest + "]")
        if kind == "box":
            return {"id": sel_id, "box": [list(payload[0]), list(payload[1])]}
        if kind == "sphere":
            return {"id": sel_id, "center": list(payload[0]),
                    "radius": float(payload[1])}
        if kind == "plane":
            return {"id": sel_id, "normal": list(payload[0]),
                    "point": list(payload[1])}
    except Exception:
        pass
    raise hou.NodeError(
        f"Could not parse native selection '{pattern}' for {name}. Examples: "
        "axis:+z:0.95 | box:[0,0,0],[1,1,1] | sphere:[0,0,0],0.5 | "
        "plane:[0,0,1],[0,0,0.5]")


# =============================================================================
# Transform / pivot helpers (UI parity with 1.2)
# =============================================================================


def decompose_rowvec_xyz_4x4(arr16):
    m = hou.Matrix4([arr16[0:4], arr16[4:8], arr16[8:12], arr16[12:16]])
    t = m.extractTranslates()
    r = m.extractRotates()
    s = m.extractScales()
    return (t[0], t[1], t[2]), (r[0], r[1], r[2]), (s[0], s[1], s[2])


def pivot_to_centroid(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    attrib_node = parent.node(f"attrib_{geo_number}")
    if attrib_node:
        bbox = attrib_node.geometry().boundingBox()
        pivot = bbox.center()
        parent.setParms({f"tpivot_{geo_number}x": pivot[0],
                         f"tpivot_{geo_number}y": pivot[1],
                         f"tpivot_{geo_number}z": pivot[2],
                         f"rpivot_{geo_number}x": 0,
                         f"rpivot_{geo_number}y": 0,
                         f"rpivot_{geo_number}z": 0})
        return bbox


def pivot_to_origin(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    parent.setParms({f"tpivot_{geo_number}x": 0, f"tpivot_{geo_number}y": 0,
                     f"tpivot_{geo_number}z": 0, f"rpivot_{geo_number}x": 0,
                     f"rpivot_{geo_number}y": 0, f"rpivot_{geo_number}z": 0})


def geometry_to_origin(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    attrib_node = parent.node(f"attrib_{geo_number}")
    if attrib_node:
        pivot = attrib_node.geometry().boundingBox().center()
        parent.setParms({f"xform_t__{geo_number}x": -pivot[0],
                         f"xform_t__{geo_number}y": -pivot[1],
                         f"xform_t__{geo_number}z": -pivot[2]})


def reset_transform(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    parent.setParms({f"xform_t__{geo_number}x": 0, f"xform_t__{geo_number}y": 0,
                     f"xform_t__{geo_number}z": 0, f"xform_r__{geo_number}x": 0,
                     f"xform_r__{geo_number}y": 0, f"xform_r__{geo_number}z": 0,
                     f"xform_s__{geo_number}x": 1, f"xform_s__{geo_number}y": 1,
                     f"xform_s__{geo_number}z": 1})


def reset_zero(kwargs):
    reset_transform(kwargs)
    pivot_to_origin(kwargs)


# =============================================================================
# Geometry node-tree management
# =============================================================================

_PROTECTED_NODES = {"output", "all"}


def _geo_index_of(name):
    parts = name.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def destroy_geo_nodes(parent, geo_number):
    for item in parent.children():
        if item.name() in _PROTECTED_NODES:
            continue
        if _geo_index_of(item.name()) == int(geo_number):
            item.destroy()


def working_dir_check(kwargs):
    parent = kwargs["node"]
    working_dir = parent.evalParm("working_dir")
    if not working_dir:
        _message("Please provide working directory location.")
        return False
    if not working_dir.endswith("/"):
        working_dir += "/"
    os.makedirs(os.path.join(working_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)
    parent.setParms({"working_dir": working_dir})
    return True


def _stage_material_file(parent, source, label="material file"):
    """Copy a user-selected material data file into ``working_dir/input``.

    PolyFEM resolves material file references relative to params.json.  Keeping
    the UI parameter pointed at the staged copy makes the viewport preview and
    the solver consume the exact same file, and makes the input folder portable.
    """
    source = os.path.abspath(os.path.expandvars(source))
    if not os.path.isfile(source):
        raise hou.NodeError(f"{label} not found: {source}")
    working_dir = parent.evalParm("working_dir")
    if not working_dir:
        raise hou.NodeError(
            f"Choose a working directory before selecting a {label}.")
    input_dir = os.path.join(working_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    destination = os.path.join(input_dir, os.path.basename(source))
    if os.path.abspath(source) != os.path.abspath(destination):
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            raise hou.NodeError(
                f"Could not copy {label} into the input folder: {exc}")
    return destination


def material_file_changed(kwargs):
    """Stage fiber/scalar files, then refresh their material preview."""
    parent = kwargs["node"]
    parm = kwargs.get("parm")
    if parm is not None:
        source = parm.eval().strip()
        if source:
            try:
                staged = _stage_material_file(
                    parent, source,
                    "fiber file" if "fib_file" in parm.name()
                    else "per-element scalar file")
            except hou.NodeError as exc:
                _message(str(exc))
                return
            if os.path.abspath(source) != os.path.abspath(staged):
                parm.set(staged)
                _status(f"Staged {os.path.basename(staged)} in "
                        f"{os.path.dirname(staged)}")
    material_display_changed(kwargs)


def polyfem_bin_check(kwargs):
    parent = kwargs["node"]
    polyfem_bin = parent.evalParm("polyfem_bin")
    if polyfem_bin and not os.path.isfile(polyfem_bin):
        _message("PolyFEM binary not found at that path.")
        parent.parm("polyfem_bin").revertToDefaults()


def create_geo_nodes(kwargs):
    """File parm callback: (re)build the import chain for one geometry."""
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    geo_number = kwargs["script_multiparm_index"]
    path_parm = parent.parm("file_location" + geo_number).eval()
    if not working_dir_check(kwargs):
        parent.parm("file_location" + geo_number).revertToDefaults()
        return

    root, extension = os.path.splitext(path_parm)
    extension = extension.lower()
    non_msh = extension in (".obj", ".stl", ".ply")
    if not non_msh and extension != ".msh":
        _message(
            "Only .msh is supported for simulated geometry; "
            ".obj/.stl/.ply/.msh for obstacles.")
        parent.parm("file_location" + geo_number).revertToDefaults()
        destroy_geo_nodes(parent, geo_number)
        return
    if non_msh:
        parent.setParms({"is_obstacle" + geo_number: 1})

    existing_geo = parent.node("geo_" + geo_number) is not None
    old_num_vols = parent.evalParm(f"num_volumes{geo_number}")
    saved_volume_colors = {
        vol: _color_tuple(parent, f"color_{geo_number}_{vol}")
        for vol in range(1, old_num_vols + 1)
    } if existing_geo else {}
    saved_obstacle_color = _color_tuple(
        parent, f"color_{geo_number}") if existing_geo else None

    parent.setParms({f"num_volumes{geo_number}": 1,
                     f"sideset_selection{geo_number}_1": 0,
                     f"initial_conditions{geo_number}_1": 0})
    obstacle_check = parent.parm("is_obstacle" + geo_number).eval()
    destroy_geo_nodes(parent, geo_number)

    # stage a copy in <working_dir>/input/
    working_dir = parent.evalParm("working_dir")
    dst = os.path.join(working_dir, "input", os.path.basename(path_parm))
    if os.path.abspath(dst) != os.path.abspath(path_parm):
        try:
            shutil.copy(path_parm, dst)
        except OSError as e:
            _message(f"Could not copy mesh into working dir: {e}")
            return
    parent.setParms({f"file_location{geo_number}": dst})

    if obstacle_check == 0 or not non_msh:
        file_node = parent.createNode("MSH_Reader::3.0", "geo_" + geo_number)
        file_node.parm("File").set(dst)
    else:
        file_node = parent.createNode("file", "geo_" + geo_number)
        file_node.parm("file").set(dst)
    try:
        file_node.cook(force=True)
    except hou.OperationFailed:
        _message("Failed to read the mesh; check the file.")
        parent.parm("file_location" + geo_number).revertToDefaults()
        return

    attrib_node = parent.createNode("attribcreate::2.0", "attrib_" + geo_number)
    attrib_node.setParms({"name1": "geometry_num", "class1": 1, "type1": 1,
                          "value1v1": int(geo_number)})
    transform_node = parent.createNode("xform", "transform_" + geo_number)
    for parm, channel in (
            ("xOrd", "xform_xOrd__{0}"), ("rOrd", "xform_rOrd__{0}"),
            ("tx", "xform_t__{0}x"), ("ty", "xform_t__{0}y"), ("tz", "xform_t__{0}z"),
            ("rx", "xform_r__{0}x"), ("ry", "xform_r__{0}y"), ("rz", "xform_r__{0}z"),
            ("sx", "xform_s__{0}x"), ("sy", "xform_s__{0}y"), ("sz", "xform_s__{0}z"),
            ("px", "tpivot_{0}x"), ("py", "tpivot_{0}y"), ("pz", "tpivot_{0}z"),
            ("prx", "rpivot_{0}x"), ("pry", "rpivot_{0}y"), ("prz", "rpivot_{0}z")):
        transform_node.parm(parm).setExpression(
            f'ch("../{channel.format(geo_number)}")')
    transform_node.setParms({"addattrib": 1})

    element_select = parent.createNode("groupcreate", "elements_" + geo_number)
    branch = parent.createNode("null", "branch_" + geo_number)
    attrib_node.setNextInput(file_node)
    transform_node.setNextInput(attrib_node)
    element_select.setNextInput(transform_node)
    branch.setNextInput(element_select)

    if obstacle_check == 0:
        if file_node.geometry().findPrimAttrib("Entity") is None:
            _message("Mesh has no Entity attribute.")
            return
        ent = np.frombuffer(
            file_node.geometry().primIntAttribValuesAsString("Entity"),
            dtype=np.int32)
        num_vols = int(ent.max())
        parent.setParms({f"num_volumes{geo_number}": num_vols})
        for vol in range(1, num_vols + 1):
            _set_color_tuple(
                parent, f"color_{geo_number}_{vol}",
                saved_volume_colors.get(vol) or default_color(geo_number, vol))
        create_geo_tree(kwargs)
        fetch_elements(kwargs)
    else:
        _set_color_tuple(
            parent, f"color_{geo_number}",
            saved_obstacle_color or default_color(geo_number))
        create_obstacle_tree(parent, geo_number)

    bbox = pivot_to_centroid(kwargs)
    sv = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer) \
        if hou.isUIAvailable() else None
    if sv and bbox:
        sv.curViewport().frameBoundingBox(bbox)


def _merge_all(parent):
    merge_node = parent.node("all")
    if merge_node is None:
        merge_node = parent.createNode("merge", "all")
        out = parent.node("output")
        if out is not None:
            out.setInput(0, merge_node)
    return merge_node


FIBER_DATA_SOP_CODE = """
node = hou.pwd()
node.parent().hdaModule().cook_fiber_data(node)
"""


def create_geo_tree(kwargs):
    """Volume geometry display chain: one Entity-colored boundary surface."""
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    branch = parent.node("branch_" + geo_number)

    # Per-element material data is stamped onto the volume elements here, so
    # the viewport draws exactly what the export writes out.
    fiberdata = parent.createNode("python", "fiberdata_" + geo_number)
    fiberdata.parm("python").set(FIBER_DATA_SOP_CODE)
    fiberdata.setNextInput(branch)

    surface = parent.createNode("attribwrangle", "surface_" + geo_number)
    surface.setParms({"class": 1,
                      "snippet": _surface_vex(parent, geo_number)})
    surface.setNextInput(fiberdata)

    color = parent.createNode("attribwrangle", "entitycolor_" + geo_number)
    color.setParms({
        "class": 1,
        "snippet": _entity_color_vex(
            geo_number, parent.evalParm(f"num_volumes{geo_number}"))})
    color.setNextInput(surface)

    # min-edge readout source (suggested dhat)
    mindist = parent.createNode("attribwrangle", "mindist_" + geo_number)
    mindist.setParms({"class": 1, "snippet": MIN_EDGE_VEX})
    mindist.setNextInput(surface)

    _build_fiber_viz(parent, geo_number, fiberdata)

    null_node = parent.createNode("null", "null_" + geo_number)
    null_node.setNextInput(color)
    _merge_all(parent).setNextInput(null_node)

    # clear_geo_tree destroys the sideset group nodes with the rest of the
    # display chain; rebuild them so viewport reselection keeps working
    # after subdomain edits.
    for vol in range(1, parent.evalParm(f"num_volumes{geo_number}") + 1):
        if parent.parm(f"sideset_selection{geo_number}_{vol}") is not None:
            _rebuild_group_chain(parent, geo_number, str(vol))
    parent.layoutChildren()


def create_obstacle_tree(parent, geo_number):
    branch = parent.node("branch_" + geo_number)

    # Seed the swatch with this geometry's palette color (distinct color per
    # obstacle, matching 1.2) without clobbering a user-customized value.
    swatch = parent.parmTuple(f"color_{geo_number}")
    if swatch is not None and all(p.isAtDefault() for p in swatch):
        _set_color_tuple(parent, f"color_{geo_number}",
                         default_color(geo_number))

    # Volumes color per-PRIM; if an obstacle carries per-POINT Cd (painted
    # .obj/.ply), the merged output ends up with both attribute classes and
    # the volume geometry's points fall back to default-black point Cd.
    # Promote any point Cd to prims first so a single class survives the
    # merge, then tint (preserving promoted file colors).
    promote_node = parent.createNode("attribpromote", "cdpromote_" + geo_number)
    promote_node.setParms({"inname": "Cd", "inclass": 2, "outclass": 1,
                           "deletein": 1})
    promote_node.setNextInput(branch)

    color_node = parent.createNode("attribwrangle", "color_" + geo_number)
    color_node.setParms({"class": 1, "snippet": (
        "if (!hasprimattrib(0, 'Cd'))\n"
        f"    v@Cd = chv('../color_{geo_number}');")})
    color_node.setNextInput(promote_node)
    null_node = parent.createNode("null", "null_" + geo_number)
    null_node.setNextInput(color_node)
    _merge_all(parent).setNextInput(null_node)
    parent.layoutChildren()


def clear_geo_tree(parent, geo_number):
    """Remove the display chain (keeps geo/attrib/transform/elements/branch/vex)."""
    keep = {"geo", "attrib", "transform", "elements", "vex", "branch"}
    for item in parent.children():
        name = item.name()
        if name in _PROTECTED_NODES:
            continue
        if _geo_index_of(name) == int(geo_number) and name.split("_")[0] not in keep:
            item.destroy()
    merge_node = parent.node("all")
    if merge_node is not None:
        for i, inp in enumerate(merge_node.inputs()):
            if inp is None:
                continue
            if _geo_index_of(inp.name()) == int(geo_number):
                merge_node.setInput(i, None)


def clear_all_geos(kwargs):
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    for item in parent.children():
        if item.name() not in _PROTECTED_NODES:
            item.destroy()
    parent.setParms({
        "geo_int": 1, "num_geos": 1, "num_volumes1": 1,
        "sideset_selection1_1": 0, "file_location1": "", "is_obstacle1": 0,
        "xform_t__1x": 0, "xform_t__1y": 0, "xform_t__1z": 0,
        "xform_r__1x": 0, "xform_r__1y": 0, "xform_r__1z": 0,
        "xform_s__1x": 1, "xform_s__1y": 1, "xform_s__1z": 1,
        "tpivot_1x": 0, "tpivot_1y": 0, "tpivot_1z": 0,
        "rpivot_1x": 0, "rpivot_1y": 0, "rpivot_1z": 0,
        "subdomain_number_1": 1, "elements_1": ""})
    if hou.isUIAvailable():
        sv = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer)
        if sv:
            sv.curViewport().home()


def eval_geo_num(kwargs):
    return kwargs["node"].evalParm("geo_int")


def clear_geos(kwargs):
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    num_geos = eval_geo_num(kwargs)
    parent.setParms({"num_geos": int(num_geos)})
    for item in parent.children():
        if item.name() in _PROTECTED_NODES:
            continue
        idx = _geo_index_of(item.name())
        if idx is not None and idx > int(num_geos):
            item.destroy()
    if hou.isUIAvailable():
        pane_tab = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.Parm, 0)
        if pane_tab:
            pane_tab.setMultiParmTab("num_geos", int(num_geos))


def fetch_elements(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    entity_number = parent.evalParm("subdomain_number_" + geo_number)
    branch = parent.node("branch_" + geo_number)
    if branch is None:
        return
    entity = np.frombuffer(
        branch.geometry().primIntAttribValuesAsString("Entity"), dtype=np.int32)
    elements = np.nonzero(entity == int(entity_number))[0]
    parent.setParms({"elements_" + geo_number: list_to_space_str(elements)})


def update_entities(kwargs, call=None):
    """Re-tag selected elements with the chosen subdomain number (VEX)."""
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    if parent.node("geo_" + geo_number) is None:
        return
    entity_number = parent.evalParm("subdomain_number_" + geo_number)
    if entity_number < 1:
        _message("Subdomain numbers start at 1.")
        return

    # A brand-new subdomain number must also grow the per-volume UI
    # (material tabs, color swatches) and the color mapping, or the new
    # subdomain renders grey and has no visible material entry.
    num_vols = parent.evalParm(f"num_volumes{geo_number}")
    if entity_number > num_vols:
        parent.setParms({f"num_volumes{geo_number}": int(entity_number)})
        for vol in range(num_vols + 1, int(entity_number) + 1):
            _set_color_tuple(parent, f"color_{geo_number}_{vol}",
                             default_color(geo_number, vol))

    elements = list_to_space_str(
        expand_group_str(parent.evalParm("elements_" + geo_number)))
    vex_node = parent.node(f"vex_{geo_number}_{entity_number}")
    new_node = vex_node is None
    if new_node:
        vex_node = parent.createNode(
            "attribwrangle", f"vex_{geo_number}_{entity_number}")
    vex_node.setParms({"group": elements, "grouptype": 4, "class": 1,
                       "snippet": f"i@Entity = {int(entity_number)};"})
    if new_node:
        branch = parent.node("branch_" + geo_number)
        upstream = branch.inputs()[0]
        vex_node.setInput(0, upstream)
        branch.setInput(0, vex_node)
    clear_geo_tree(parent, geo_number)
    create_geo_tree(kwargs)


def revert_entities(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    for item in parent.children():
        name = item.name()
        if name.split("_")[0] == "vex" and _geo_index_of(name) == int(geo_number):
            item.destroy()
    # reconnect branch to elements node
    branch = parent.node("branch_" + geo_number)
    elements = parent.node("elements_" + geo_number)
    if branch is not None and elements is not None:
        branch.setInput(0, elements)
    fetch_elements(kwargs)
    clear_geo_tree(parent, geo_number)
    create_geo_tree(kwargs)


def BC_check(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index2"]
    if parent.parm("is_obstacle" + str(geo_number)).eval():
        _message("Obstacles take a displacement, not sidesets.")


def create_group(kwargs):
    """Sideset count callback: maintain groupcreate nodes in the surface chain."""
    BC_check(kwargs)
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    geo_number = kwargs["script_multiparm_index2"]
    vol_num = kwargs["script_multiparm_index"]
    if parent.node("geo_" + geo_number) is None:
        _message("No input geometry provided to apply sideset!")
        parent.setParms({f"sideset_selection{geo_number}_{vol_num}": 0})
        return
    _rebuild_group_chain(parent, geo_number, vol_num)


def _rebuild_group_chain(parent, geo_number, vol_num):
    sideset_number = parent.evalParm(
        f"sideset_selection{geo_number}_{vol_num}")

    prefix = f"group_{geo_number}_{vol_num}_"
    existing = [n for n in parent.children() if n.name().startswith(prefix)]
    for node in existing:
        if int(node.name().split("_")[-1]) > sideset_number:
            node.destroy()

    color_node = parent.node("entitycolor_" + geo_number)
    null_node = parent.node("null_" + geo_number)
    if color_node is None or null_node is None:
        return

    # The group nodes hang off entitycolor on a dead-end branch of their own
    # rather than being spliced into the display chain. Splicing them inline
    # meant each volume's rebuild re-pointed null_ at its own last group node,
    # orphaning every other volume's chain; and nothing downstream consumes
    # the groups anyway -- they exist only to host the viewport selection.
    null_node.setInput(0, color_node)

    # Restrict picking to this volume. soputils.selectGroupParm display-flags
    # the group node's input 0, so every entity visible there is pickable: on
    # a multi-material mesh the other subdomains stayed in the way, hiding the
    # interface faces behind them, and any face picked on another entity was
    # then silently dropped by export_sidesets' per-volume filter. 1.2 got this
    # for free by giving each volume its own split branch; do it without the
    # per-volume node forest by hiding the other entities instead of deleting
    # them -- Visibility leaves primitive numbering untouched, which the
    # geometry-wide prim numbers in basegroup depend on.
    solo_name = f"solo_{geo_number}_{vol_num}"
    solo_node = parent.node(solo_name)
    if sideset_number == 0:
        if solo_node is not None:
            solo_node.destroy()
        parent.layoutChildren()
        return
    if solo_node is None:
        solo_node = parent.createNode("visibility", solo_name)
    solo_node.setParms({"group": f"@Entity!={int(vol_num)}",
                        "action": 0, "applyto": 0})
    solo_node.setInput(0, color_node)

    upstream = solo_node
    for num in range(1, sideset_number + 1):
        name = f"{prefix}{num}"
        group_node = parent.node(name)
        if group_node is None:
            group_node = parent.createNode("groupcreate", name)
            # A freshly created sideset defaults to one zero-displacement
            # Dirichlet BC (all dims fixed), matching the 1.x behavior --
            # but never overwrite a sideset that already carries BCs or a
            # selection (e.g. during params import).
            bc_count = parent.parm(
                f"Boundary_Condition__{geo_number}_{vol_num}_{num}")
            basegroup = parent.evalParm(
                f"basegroup{geo_number}_{vol_num}_{num}")
            if bc_count is not None and bc_count.eval() == 0 \
                    and not basegroup.strip():
                bc_count.set(1)
            group_node.parm("grouptype").setExpression(
                f'ch("../grouptype{geo_number}_{vol_num}_{num}")')
            # native selection syntax types into the same field; feed the
            # group SOP an empty pattern in that case so it does not error
            group_node.parm("basegroup").setExpression(
                f"p = hou.node('..').evalParm('basegroup{geo_number}_{vol_num}_{num}')\n"
                "return '' if p.strip().lower().startswith("
                "('axis:', 'box:', 'sphere:', 'plane:')) else p",
                hou.exprLanguage.Python)
        group_node.setInput(0, upstream)
        upstream = group_node
    parent.layoutChildren()


def geo_duplicate(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    if parent.node("geo_" + geo_number) is None:
        _message("Import geometry, then duplicate.")
        return
    src = int(geo_number)
    dst = parent.evalParm("num_geos") + 1
    parent.setParms({"num_geos": dst, "geo_int": dst})
    if hou.isUIAvailable():
        pane_tab = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.Parm, 0)
        if pane_tab:
            pane_tab.setMultiParmTab("num_geos", dst)

    parent.setParms({f"file_location{dst}": parent.evalParm(f"file_location{src}"),
                     f"is_obstacle{dst}": parent.evalParm(f"is_obstacle{src}"),
                     f"is_enabled{dst}": parent.evalParm(f"is_enabled{src}")})
    parent.parm(f"file_location{dst}").pressButton()

    copies = {}
    for base in ("xform_t__{}x", "xform_t__{}y", "xform_t__{}z",
                 "xform_r__{}x", "xform_r__{}y", "xform_r__{}z",
                 "xform_s__{}x", "xform_s__{}y", "xform_s__{}z",
                 "tpivot_{}x", "tpivot_{}y", "tpivot_{}z",
                 "rpivot_{}x", "rpivot_{}y", "rpivot_{}z"):
        copies[base.format(dst)] = parent.evalParm(base.format(src))
    for suffix in "rgb":
        copies[f"color_{dst}{suffix}"] = parent.evalParm(f"color_{src}{suffix}")
    parent.setParms(copies)

    if not parent.evalParm(f"is_obstacle{src}"):
        num_vols = parent.evalParm(f"num_volumes{src}")
        mat_parms = {}
        for v in range(1, num_vols + 1):
            for base in ("mainOrder{}_{}", "materials{}_{}", "rho{}_{}",
                         "E{}_{}", "nu{}_{}", "damping{}_{}", "phi{}_{}",
                         "psi{}_{}", "bulk{}_{}", "c1{}_{}", "c2{}_{}",
                         "c3{}_{}", "d1{}_{}", "color_{}_{}r",
                         "color_{}_{}g", "color_{}_{}b"):
                parm = parent.parm(base.format(src, v))
                if parm is not None and parent.parm(base.format(dst, v)) is not None:
                    mat_parms[base.format(dst, v)] = parm.eval()
        parent.setParms(mat_parms)


def deter_min_edge(kwargs):
    parent = kwargs["node"]
    total_min = None
    for node in parent.glob("mindist_*"):
        try:
            node.cook(force=False)
        except hou.OperationFailed:
            _message(f"Could not cook {node.name()}: {node.errors()}")
            continue
        geo = node.geometry()
        if geo is None or geo.findPrimAttrib("min_edge_length") is None:
            continue
        values = np.frombuffer(
            geo.primFloatAttribValuesAsString("min_edge_length"),
            dtype=np.float32)
        if len(values):
            local_min = float(values.min())
            total_min = local_min if total_min is None \
                else min(total_min, local_min)
    if total_min is None:
        _message("No surface geometry found; import a volume mesh first.")
        return
    parent.parm("min_edge").set(str(total_min))


# =============================================================================
# Sideset / selection export (numpy bulk)
# =============================================================================


def _surface_arrays(surf_geo):
    """Bulk read of the provenance attributes stamped by SURFACE_VEX.

    Returns (entity, face_ids) where face_ids is (N, 4) int64 with -1 in
    the last column for triangles.
    """
    entity = np.frombuffer(
        surf_geo.primIntAttribValuesAsString("Entity"), dtype=np.int32)
    cols = [np.frombuffer(surf_geo.primIntAttribValuesAsString(f"fp{c}"),
                          dtype=np.int32) for c in range(4)]
    face_ids = np.stack(cols, axis=1).astype(np.int64)
    return entity, face_ids


def export_sidesets(parent, geo, num_vols, input_dir):
    """Write per-arity sideset files + native selections for one geometry.

    Returns (surface_selection_list, point_selection_list) for the json.
    """
    null_node = parent.node(f"null_{geo}")
    if null_node is None:
        return [], []
    surf_geo = null_node.geometry()
    entity, face_ids = _surface_arrays(surf_geo)
    n_prims = len(entity)

    surface_rows = {3: [], 4: []}  # arity -> list of [id, v0, v1, v2(, v3)]
    point_rows = []
    native_selections = []
    cross_volume = []  # explicit picks that landed on another subdomain

    for vol in range(1, num_vols + 1):
        sidesets = parent.evalParm(f"sideset_selection{geo}_{vol}")
        for j in range(1, sidesets + 1):
            num_bcs = parent.evalParm(f"Boundary_Condition__{geo}_{vol}_{j}")
            pattern = parent.evalParm(f"basegroup{geo}_{vol}_{j}")
            grouptype = parent.evalParm(f"grouptype{geo}_{vol}_{j}")
            for k in range(1, num_bcs + 1):
                sel_id = boundary_id(geo, vol, j, k)
                if is_native_selection(pattern):
                    native_selections.append(parse_native_selection(
                        pattern, sel_id, f"geo {geo} vol {vol} sideset {j}"))
                    continue
                if grouptype == 1:  # points
                    if pattern.strip() == "*":
                        idx = np.arange(surf_geo.intrinsicValue("pointcount"))
                    else:
                        idx = np.asarray(expand_group_str(pattern), dtype=np.int64)
                    msh = np.frombuffer(
                        surf_geo.pointIntAttribValuesAsString("msh_pt_id"),
                        dtype=np.int32)
                    for m in msh[idx] - 1:
                        point_rows.append((sel_id, int(m)))
                    continue
                # primitive sideset
                wildcard = pattern.strip() == "*"
                if wildcard:
                    idx = np.arange(n_prims)
                else:
                    idx = np.asarray(expand_group_str(pattern), dtype=np.int64)
                if len(idx) == 0:
                    raise hou.NodeError(
                        f"Sideset {j} of geo {geo}/vol {vol} selects nothing")
                # Restrict to this volume's faces; "*" is defined as "every
                # face of this subdomain", so filtering it is not a mistake.
                # An explicit pick that loses faces here is one made against
                # another subdomain's surface -- report it rather than
                # quietly writing a sideset the user did not select.
                kept = idx[entity[idx] == vol]
                if len(kept) == 0:
                    raise hou.NodeError(
                        f"Sideset {j} of geo {geo}/vol {vol} selects "
                        f"{len(idx)} face(s), none of which are on subdomain "
                        f"{vol}; pick faces on that subdomain instead")
                if not wildcard and len(kept) != len(idx):
                    cross_volume.append(
                        f"  geo {geo}, subdomain {vol}, sideset {j}: dropped "
                        f"{len(idx) - len(kept)} of {len(idx)} faces")
                ids = face_ids[kept]
                tri_mask = ids[:, 3] < 0
                for row in ids[tri_mask, :3]:
                    surface_rows[3].append((sel_id,) + tuple(int(v) for v in row))
                for row in ids[~tri_mask]:
                    surface_rows[4].append((sel_id,) + tuple(int(v) for v in row))

    if cross_volume:
        _message("Some sideset faces belong to a different subdomain than the "
                 "sideset they were assigned to, and were left out:\n"
                 + "\n".join(cross_volume))

    # Native selections first: polyfem applies the FIRST selection in the
    # list that matches a face, and explicit axis/box/plane/sphere picks
    # should win over broad face-list files.
    surface_sel = list(native_selections)
    for arity, rows in surface_rows.items():
        if not rows:
            continue
        fname = f"surface_sidesets{geo}_{'tri' if arity == 3 else 'quad'}.txt"
        np.savetxt(os.path.join(input_dir, fname),
                   np.asarray(rows, dtype=np.int64), fmt="%d")
        surface_sel.append({"file": fname})

    point_sel = []
    if point_rows:
        fname = f"point_sidesets{geo}.txt"
        np.savetxt(os.path.join(input_dir, fname),
                   np.asarray(point_rows, dtype=np.int64), fmt="%d")
        point_sel.append({"file": fname})
    return surface_sel, point_sel


def export_volumes(parent, geo, input_dir):
    """volumes<geo>.txt: one body id per volume element, element order."""
    branch = parent.node(f"branch_{geo}")
    geo_data = branch.geometry()
    entity = np.frombuffer(
        geo_data.primIntAttribValuesAsString("Entity"), dtype=np.int32)
    if geo_data.findPrimAttrib("is_volume") is not None:
        keep = np.frombuffer(
            geo_data.primIntAttribValuesAsString("is_volume"),
            dtype=np.int32).astype(bool)
        entity = entity[keep]
    fname = f"volumes{geo}.txt"
    np.savetxt(os.path.join(input_dir, fname),
               1000 * int(geo) + entity.astype(np.int64), fmt="%d")
    return fname


# =============================================================================
# Per-element material data (fiber directions, per-element scalars)
#
# PolyFEM binds per-element material data by GLOBAL element id: the elements of
# every enabled, non-obstacle geometry concatenated in geometry order, keeping
# each mesh's own element order. Everything here is built around that contract:
#   * canonical prim attributes (pf_*) on a per-geometry stamp node, so the
#     viewport shows exactly the data the solver is handed
#   * one global fibers.vtk (one VECTORS array per family) and global-length
#     per-element scalar files
# =============================================================================

# Material menu tokens, in DialogScript menu order (the index is stored in
# saved scenes, so entries may be appended but never reordered).
MATERIAL_TOKENS = (
    "LinearElasticity", "HookeLinearElasticity", "IncompressibleLinearElasticity",
    "NeoHookean", "IncompressibleOgden", "UnconstrainedOgden", "MooneyRivlin",
    "MooneyRivlin3Param", "MooneyRivlin3ParamSymbolic", "SaintVenant",
    "FixedCorotational", "IsochoricNeoHookean", "HGOFiber", "HGODispersion",
    "ActiveFiber", "MaterialSum",
)
FIBER_MODELS = ("HGOFiber", "HGODispersion", "ActiveFiber")

# Volume-level parameter names that differ from their family-level base.
_VOLUME_PARM_ALIASES = {"k1": "hgo_k1", "k2": "hgo_k2"}


def material_token(parent, geo, vol):
    index = parent.evalParm(f"materials{geo}_{vol}")
    if not 0 <= index < len(MATERIAL_TOKENS):
        raise hou.NodeError(f"Unknown material menu index {index}")
    return MATERIAL_TOKENS[index]


def _parm_name(geo, vol, fam, base):
    """Parameter name for a material slot.

    A plain fiber material keeps its parameters at volume level (fam is None);
    the fiber families of a composite live in a third multiparm level.
    """
    if fam is None:
        return f"{_VOLUME_PARM_ALIASES.get(base, base)}{geo}_{vol}"
    return f"fam_{base}{geo}_{vol}_{fam}"


def _slot_parm(parent, geo, vol, fam, base):
    """Value of a material-slot parameter; menus evaluate to their token.

    evalParm() would hand back a menu's *index*, which silently compares
    unequal to every token it is tested against.
    """
    name = _parm_name(geo, vol, fam, base)
    parm = parent.parm(name)
    if parm is None:
        raise hou.NodeError(f"Missing parameter '{name}'")
    if parm.parmTemplate().type() == hou.parmTemplateType.Menu:
        return parm.evalAsString()
    return parm.eval()


def material_families(parent, geo, vol):
    """[(fam, model_token)] fiber families of one subdomain.

    fam is None for a plain fiber material, 1..N for a composite's families.
    """
    token = material_token(parent, geo, vol)
    if token in FIBER_MODELS:
        return [(None, token)]
    if token != "MaterialSum":
        return []
    count = parent.evalParm(f"num_fiber_families{geo}_{vol}")
    families = []
    for fam in range(1, count + 1):
        parm = parent.parm(f"fam_model{geo}_{vol}_{fam}")
        if parm is None:
            continue
        families.append((fam, parm.evalAsString()))
    return families


def fiber_attrib(vol, fam, side=""):
    """Canonical prim attribute holding one family's directions.

    Namespaced by subdomain and family: several subdomains of one geometry
    share the stamp node, so a bare pf_fiber would collide between them.
    """
    return f"pf_fiber_{vol}_{0 if fam is None else fam}{side}"


def scalar_attrib(name, vol, fam):
    return f"pf_{name}_{vol}_{0 if fam is None else fam}"


def fiber_field(geo, vol, fam, side=""):
    """VECTORS array name for one family inside the shared fibers.vtk."""
    base = f"FIB_{geo}_{vol}" if fam is None else f"FIB_{geo}_{vol}_{fam}"
    return base + side


def mirror_sides(parent, geo, vol, fam):
    """('p', 'm') for a symmetric +/-theta family, ('',) for a single one."""
    if fam is None:
        return ("",)
    parm = parent.parm(f"fam_mirror{geo}_{vol}_{fam}")
    if parm is not None and parm.eval() \
            and _slot_parm(parent, geo, vol, fam, "fib_source") != "expression":
        return ("p", "m")
    return ("",)


def read_cell_vectors_legacy_vtk(path, field):
    """Per-element vectors from a legacy ASCII VTK, mirroring polyfem's reader.

    Only the CELL_DATA count and the named VECTORS array matter; points and
    cells are not required.
    """
    count = None
    with open(path) as handle:
        tokens = []
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "CELL_DATA":
                count = int(parts[1])
                continue
            if parts[0] == "VECTORS":
                if len(parts) < 2 or parts[1] != field:
                    continue
                if count is None:
                    raise hou.NodeError(
                        f"{path}: VECTORS before CELL_DATA")
                for line in handle:
                    tokens.extend(line.split())
                    if len(tokens) >= 3 * count:
                        break
                if len(tokens) < 3 * count:
                    raise hou.NodeError(
                        f"{path}: expected {count} vectors for '{field}', "
                        f"found {len(tokens) // 3}")
                return np.asarray(tokens[:3 * count], dtype=np.float64).reshape(-1, 3)
    raise hou.NodeError(
        f"VECTORS '{field}' not found under CELL_DATA in {path}")


def _volume_mask(geo_data):
    """Boolean mask of the primary (volume) elements, in polyfem order."""
    if geo_data.findPrimAttrib("is_volume") is None:
        return None
    return np.frombuffer(
        geo_data.primIntAttribValuesAsString("is_volume"),
        dtype=np.int32).astype(bool)


def volume_element_count(parent, geo):
    branch = parent.node(f"branch_{geo}")
    if branch is None:
        return 0
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    if mask is None:
        return geo_data.intrinsicValue("primitivecount")
    return int(mask.sum())


def global_element_table(parent):
    """[(geo, offset, count)] over the geometries polyfem actually meshes.

    Obstacles and disabled geometries are absent from the solver's element
    numbering, so they consume no ids -- and per-element data for them is
    simply not exported.
    """
    table = []
    offset = 0
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval():
            continue
        if not parent.parm(f"is_enabled{geo}").eval():
            continue
        count = volume_element_count(parent, geo)
        table.append((geo, offset, count))
        offset += count
    return table, offset


def _prim_centroids(geo_data, mask):
    """Volume-element centroids in element order (recomputed, not cached).

    MSH_Reader stamps a centroid attribute, but it predates the geometry
    transform; recomputing keeps the match in the space the user is looking at.
    """
    positions = np.frombuffer(
        geo_data.pointFloatAttribValuesAsString("P"),
        dtype=np.float32).astype(np.float64).reshape(-1, 3)
    centroids = np.empty((geo_data.intrinsicValue("primitivecount"), 3))
    for prim in geo_data.prims():
        pts = [v.point().number() for v in prim.vertices()]
        centroids[prim.number()] = positions[pts].mean(axis=0)
    return centroids[mask] if mask is not None else centroids


def _match_by_centroid(source_geo, target_centroids, label):
    """Map target elements onto source volume prims by nearest centroid.

    Element order of an external SOP need not match the HDA's, so bind on
    position rather than index and refuse a match that is not essentially
    exact.
    """
    src_mask = _volume_mask(source_geo)
    src_centroids = _prim_centroids(source_geo, src_mask)
    src_index = np.flatnonzero(src_mask) if src_mask is not None \
        else np.arange(len(src_centroids))
    if len(src_centroids) != len(target_centroids):
        raise hou.NodeError(
            f"{label}: source has {len(src_centroids)} elements, this "
            f"geometry has {len(target_centroids)}")
    try:
        from scipy.spatial import cKDTree
        distances, order = cKDTree(src_centroids).query(target_centroids)
    except ImportError:
        deltas = target_centroids[:, None, :] - src_centroids[None, :, :]
        squared = np.einsum("ijk,ijk->ij", deltas, deltas)
        order = squared.argmin(axis=1)
        distances = np.sqrt(squared[np.arange(len(order)), order])
    extent = float(np.linalg.norm(
        target_centroids.max(axis=0) - target_centroids.min(axis=0)))
    tolerance = max(1e-9, 1e-4 * extent)
    if distances.max() > tolerance:
        raise hou.NodeError(
            f"{label}: element centroids do not line up (worst offset "
            f"{distances.max():.3g} > {tolerance:.3g}); the source SOP must "
            f"be the same mesh as this geometry")
    return src_index[order]


def _read_element_attribute(parent, geo, node_path, attrib_name, components,
                            label):
    """Per-element values from a SOP's prim attribute, in polyfem order.

    PolyFEM's ``geometry[].transformation`` moves the mesh but does not
    transform ``materials[].fiber_direction``.  Match an external Fiber SOP
    against the untransformed imported mesh first, and return its vector values
    verbatim.  A transformed-position fallback keeps already-world-space SOPs
    usable without changing their direction components.
    """
    branch = parent.node(f"branch_{geo}")
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    source_node = parent.node(f"geo_{geo}")
    source_mesh = source_node.geometry() if source_node is not None else geo_data
    source_mask = _volume_mask(source_mesh)
    if node_path:
        source = parent.node(node_path) or hou.node(node_path)
        if source is None:
            raise hou.NodeError(f"{label}: SOP '{node_path}' not found")
        source_geo = source.geometry()
    else:
        # Read attributes before the HDA's geometry Transform SOP. Houdini may
        # otherwise rotate vector-typed attributes, while PolyFEM explicitly
        # leaves a0 unchanged by geometry[].transformation.
        source_geo = source_mesh
    attrib = source_geo.findPrimAttrib(attrib_name)
    if attrib is None:
        raise hou.NodeError(
            f"{label}: no primitive attribute '{attrib_name}' on "
            f"{node_path or 'the imported mesh'}")
    size = attrib.size()
    if size != components:
        raise hou.NodeError(
            f"{label}: attribute '{attrib_name}' has {size} component(s), "
            f"expected {components}")
    if attrib.dataType() == hou.attribData.Int:
        values = np.frombuffer(
            source_geo.primIntAttribValuesAsString(attrib_name),
            dtype=np.int32).astype(np.float64)
    else:
        values = np.frombuffer(
            source_geo.primFloatAttribValuesAsString(attrib_name),
            dtype=np.float32).astype(np.float64)
    values = values.reshape(-1, components) if components > 1 else values

    if source_geo is source_mesh:
        return values[source_mask] if source_mask is not None else values
    if source_geo is geo_data:
        return values[mask] if mask is not None else values

    raw_targets = _prim_centroids(source_mesh, source_mask)
    try:
        order = _match_by_centroid(source_geo, raw_targets, label)
    except hou.Error as raw_error:
        # Some users deliberately author the source SOP after applying the same
        # world transform. Support that layout too, while still leaving the
        # vector values themselves untouched.
        try:
            order = _match_by_centroid(
                source_geo, _prim_centroids(geo_data, mask), label)
        except hou.Error:
            raise raw_error
    return values[order]


def _element_centroids(parent, geo):
    """Volume-element centroids of a geometry, in polyfem element order."""
    geo_data = parent.node(f"branch_{geo}").geometry()
    return _prim_centroids(geo_data, _volume_mask(geo_data))


def cylindrical_frame(centroids, origin, axis, label):
    """Per-element (radial, circumferential, longitudinal) unit frame.

    The natural frame for vessels and other tubular tissue: the axis is the
    lumen centreline, and fibers are usually circumferential (optionally as a
    +/-theta pair about the radial direction).
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise hou.NodeError(f"{label}: the cylinder axis has zero length")
    axis = axis / norm

    relative = np.asarray(centroids, dtype=np.float64) - np.asarray(
        origin, dtype=np.float64)
    along = relative @ axis
    radial = relative - along[:, None] * axis
    distance = np.linalg.norm(radial, axis=1)

    # Elements sitting on the axis itself have no radial direction. That is a
    # solid cylinder, not a tube -- say so rather than emitting garbage.
    degenerate = np.flatnonzero(distance < 1e-9 * max(1.0, float(distance.max())))
    if len(degenerate):
        raise hou.NodeError(
            f"{label}: {len(degenerate)} element(s) lie on the cylinder axis "
            f"(e.g. {list(degenerate[:10])}), where the radial and "
            f"circumferential directions are undefined. Move the axis outside "
            f"the material, or use a constant/attribute fiber direction.")
    radial = radial / distance[:, None]
    return radial, np.cross(np.tile(axis, (len(radial), 1)), radial), \
        np.tile(axis, (len(radial), 1))


def curve_tangent(parent, centroids, node_path, label):
    """Tangent of the nearest point on a curve, per element.

    The line-of-action frame for tendon, ligament and muscle: draw (or import)
    a centreline curve and every element follows it.
    """
    if not node_path:
        raise hou.NodeError(f"{label}: no curve SOP specified")
    source = parent.node(node_path) or hou.node(node_path)
    if source is None:
        raise hou.NodeError(f"{label}: SOP '{node_path}' not found")
    curve_geo = source.geometry()

    starts, ends = [], []
    for prim in curve_geo.prims():
        points = [v.point().position() for v in prim.vertices()]
        if prim.isClosed() and len(points) > 2:
            points.append(points[0])
        for first, second in zip(points[:-1], points[1:]):
            starts.append(first)
            ends.append(second)
    if not starts:
        raise hou.NodeError(
            f"{label}: '{node_path}' holds no curve segments (it needs a "
            f"polyline, e.g. a Curve or Resample SOP)")

    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    segments = ends - starts
    lengths = np.einsum("ij,ij->i", segments, segments)
    keep = lengths > 1e-24
    starts, segments, lengths = starts[keep], segments[keep], lengths[keep]
    if not len(starts):
        raise hou.NodeError(f"{label}: '{node_path}' has zero-length segments only")

    centroids = np.asarray(centroids, dtype=np.float64)
    tangents = np.empty_like(centroids)
    # Chunked so a fine curve against a large mesh cannot blow up memory.
    chunk = max(1, int(4e6 // max(1, len(starts))))
    for begin in range(0, len(centroids), chunk):
        block = centroids[begin:begin + chunk]
        offset = block[:, None, :] - starts[None, :, :]
        t = np.clip(
            np.einsum("ijk,jk->ij", offset, segments) / lengths[None, :], 0, 1)
        closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        deltas = block[:, None, :] - closest
        nearest = np.einsum("ijk,ijk->ij", deltas, deltas).argmin(axis=1)
        tangents[begin:begin + chunk] = segments[nearest]
    return tangents / np.linalg.norm(tangents, axis=1)[:, None]


def _preset_fibers(parent, geo, vol, fam, source, centroids, label):
    """Procedurally generated per-element directions for the fiber presets."""
    if source == "cylindrical":
        radial, circumferential, longitudinal = cylindrical_frame(
            centroids,
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_origin")).eval(),
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_dir")).eval(),
            label)
        component = _slot_parm(parent, geo, vol, fam, "fib_component")
        return {"radial": radial, "circumferential": circumferential,
                "longitudinal": longitudinal}[component]
    return curve_tangent(
        parent, centroids,
        _slot_parm(parent, geo, vol, fam, "fib_curve"), label)


def _read_element_file(path, count, components, field, label, parent=None,
                       geo=None):
    path = os.path.expandvars(path)
    if not os.path.isfile(path):
        raise hou.NodeError(f"{label}: file not found: {path}")
    if components == 3 and path.lower().endswith(".vtk"):
        values = read_cell_vectors_legacy_vtk(path, field or "FIB_DIR1")
    else:
        text = open(path).read()
        delimiter = "," if ("," in text and ".vtk" not in path.lower()) else None
        values = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        if components == 1:
            values = values.reshape(-1)
        elif values.shape[1] != components:
            raise hou.NodeError(
                f"{label}: file has {values.shape[1]} columns, "
                f"expected {components}")
    if len(values) != count and parent is not None and geo is not None:
        # This HDA writes one shared, global-length fibers.vtk because PolyFEM
        # indexes per-element fields across all enabled simulated geometries.
        # On re-import, recover the slice belonging to this geometry so the
        # preview/re-export path remains valid in multi-geometry scenes.
        table, total = global_element_table(parent)
        if len(values) == total:
            match = next(
                ((offset, size) for index, offset, size in table
                 if int(index) == int(geo)), None)
            if match is not None:
                offset, size = match
                values = values[offset:offset + size]
    if len(values) != count:
        raise hou.NodeError(
            f"{label}: file has {len(values)} rows but this geometry has "
            f"{count} elements")
    return values


def _rotate_about_axis(vectors, axes, degrees, label):
    """Rodrigues rotation of per-element fibers about per-element axes."""
    theta = math.radians(degrees)
    axis_norm = np.linalg.norm(axes, axis=1)
    bad = np.flatnonzero(axis_norm < 1e-12)
    if len(bad):
        raise hou.NodeError(
            f"{label}: zero-length rotation axis on element(s) "
            f"{list(bad[:10])}")
    axes = axes / axis_norm[:, None]
    parallel = np.flatnonzero(
        np.abs(np.einsum("ij,ij->i", axes, vectors)) > 1 - 1e-9)
    if len(parallel):
        raise hou.NodeError(
            f"{label}: rotation axis is parallel to the fiber on element(s) "
            f"{list(parallel[:10])}, so the two families would coincide")
    cos, sin = math.cos(theta), math.sin(theta)
    dot = np.einsum("ij,ij->i", axes, vectors)[:, None]
    cross = np.cross(axes, vectors)
    return vectors * cos + cross * sin + axes * dot * (1.0 - cos)


def _normalize_fibers(vectors, label):
    norms = np.linalg.norm(vectors, axis=1)
    bad = np.flatnonzero(~np.isfinite(norms) | (norms < 1e-12))
    if len(bad):
        raise hou.NodeError(
            f"{label}: zero-length or non-finite fiber direction on "
            f"element(s) {list(bad[:10])}")
    return vectors / norms[:, None]


# Fiber sources that produce one direction per element (and therefore a file).
PER_ELEMENT_FIBER_SOURCES = ("attribute", "file", "cylindrical", "curve")


def resolve_fibers(parent, geo, vol, fam, count):
    """{side: (count, 3) unit directions} for one family, or None if constant.

    Returns None when the family's direction is a plain constant or expression
    (nothing per-element to stamp or export).
    """
    source = _slot_parm(parent, geo, vol, fam, "fib_source")
    sides = mirror_sides(parent, geo, vol, fam)
    label = _slot_label(geo, vol, fam)
    if source not in PER_ELEMENT_FIBER_SOURCES and sides == ("",):
        return None

    # Mirroring (and with it the rotation axis) only exists for the fiber
    # families of a composite, so do not look for those parameters otherwise.
    axis_source = _slot_parm(parent, geo, vol, fam, "axis_source") \
        if sides != ("",) else ""

    centroids = None
    if source in ("cylindrical", "curve") or axis_source == "cylindrical":
        centroids = _element_centroids(parent, geo)

    if source == "attribute":
        base = _read_element_attribute(
            parent, geo,
            _slot_parm(parent, geo, vol, fam, "fib_sop"),
            _slot_parm(parent, geo, vol, fam, "fib_attrib"), 3,
            f"{label} fiber")
    elif source == "file":
        base = _read_element_file(
            _slot_parm(parent, geo, vol, fam, "fib_file"), count, 3,
            _slot_parm(parent, geo, vol, fam, "fib_file_field"),
            f"{label} fiber", parent, geo)
    elif source in ("cylindrical", "curve"):
        base = _preset_fibers(parent, geo, vol, fam, source, centroids,
                              f"{label} fiber")
    else:  # constant direction, mirrored into a per-element pair
        direction = np.asarray(
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_dir")).eval(),
            dtype=np.float64)
        base = np.tile(direction, (count, 1))
    base = _normalize_fibers(np.asarray(base, dtype=np.float64), f"{label} fiber")

    if sides == ("",):
        return {"": base}

    if axis_source == "attribute":
        axes = _read_element_attribute(
            parent, geo,
            _slot_parm(parent, geo, vol, fam, "fib_sop"),
            _slot_parm(parent, geo, vol, fam, "axis_attrib"), 3,
            f"{label} rotation axis")
        axes = np.asarray(axes, dtype=np.float64)
    elif axis_source == "cylindrical":
        # The artery layout: circumferential fibers wound +/-theta about the
        # wall normal, which is the cylinder's own radial direction.
        axes = cylindrical_frame(
            centroids,
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_origin")).eval(),
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_axis_dir")).eval(),
            f"{label} rotation axis")[0]
    else:
        axis = np.asarray(
            parent.parmTuple(_parm_name(geo, vol, fam, "axis")).eval(),
            dtype=np.float64)
        axes = np.tile(axis, (count, 1))
    theta = _slot_parm(parent, geo, vol, fam, "theta")
    return {"p": _rotate_about_axis(base, axes, theta, label),
            "m": _rotate_about_axis(base, axes, -theta, label)}


def resolve_scalar(parent, geo, vol, fam, name, count):
    """Per-element values for a scalar parameter, or None if not per-element."""
    source = _slot_parm(parent, geo, vol, fam, f"{name}_source")
    label = f"{_slot_label(geo, vol, fam)} {name}"
    if source == "attribute":
        values = _read_element_attribute(
            parent, geo,
            _slot_parm(parent, geo, vol, fam, f"{name}_sop"),
            _slot_parm(parent, geo, vol, fam, f"{name}_attrib"), 1, label)
    elif source == "file":
        values = _read_element_file(
            _slot_parm(parent, geo, vol, fam, f"{name}_file"), count, 1,
            None, label, parent, geo)
    else:
        return None
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if name == "kappa":
        limit = 1.0 / 3.0
        out = np.flatnonzero((values < 0) | (values > limit + 1e-9))
        if len(out):
            raise hou.NodeError(
                f"{label}: dispersion must be within [0, 1/3]; element(s) "
                f"{list(out[:10])} are outside (min {values.min():.4g}, "
                f"max {values.max():.4g})")
    return values


def _slot_label(geo, vol, fam):
    if fam is None:
        return f"geo {geo} subdomain {vol}"
    return f"geo {geo} subdomain {vol} family {fam}"


def per_element_data(parent, geo):
    """{attrib_name: (n, c) array} of every per-element field of one geometry.

    Single place where the sources are read, so the stamped attributes the
    viewport draws and the files the solver reads can never disagree.
    """
    count = volume_element_count(parent, geo)
    if not count:
        return {}, 0
    fields = {}
    for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
        for fam, model in material_families(parent, geo, vol):
            fibers = resolve_fibers(parent, geo, vol, fam, count)
            if fibers:
                for side, values in fibers.items():
                    fields[fiber_attrib(vol, fam, side)] = values
            if model == "HGODispersion":
                kappa = resolve_scalar(parent, geo, vol, fam, "kappa", count)
                if kappa is not None:
                    fields[scalar_attrib("kappa", vol, fam)] = kappa
    if fields:
        _check_single_element_family(parent, geo)
    return fields, count


def display_material_data(parent, geo):
    """Per-element fields used by the viewport.

    PolyFEM accepts constant directions and dispersion values directly in the
    JSON, so they intentionally do not belong in ``fibers.vtk``/``pe_*.txt``.
    The viewport still needs one value per element, however. Expand constants
    here, downstream of the solver-export resolver, so previewing a constant
    material does not silently change the files handed to PolyFEM.

    General x/y/z/t expressions are left to PolyFEM: evaluating its expression
    language in Houdini would risk showing a field different from the solve.
    """
    fields, count = per_element_data(parent, geo)
    if not count:
        return fields, count

    for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
        for fam, model in material_families(parent, geo, vol):
            sides = mirror_sides(parent, geo, vol, fam)
            source = _slot_parm(parent, geo, vol, fam, "fib_source")
            if source == "constant" and sides == ("",):
                direction = np.asarray(
                    parent.parmTuple(
                        _parm_name(geo, vol, fam, "fib_dir")).eval(),
                    dtype=np.float64)
                values = _normalize_fibers(
                    np.tile(direction, (count, 1)),
                    f"{_slot_label(geo, vol, fam)} fiber")
                fields[fiber_attrib(vol, fam)] = values

            if model == "HGODispersion" and _slot_parm(
                    parent, geo, vol, fam, "kappa_source") == "constant":
                kappa = float(_slot_parm(parent, geo, vol, fam, "kappa"))
                if not math.isfinite(kappa) or kappa < 0 \
                        or kappa > 1.0 / 3.0 + 1e-9:
                    raise hou.NodeError(
                        f"{_slot_label(geo, vol, fam)} kappa: dispersion must "
                        f"be within [0, 1/3], got {kappa}")
                fields[scalar_attrib("kappa", vol, fam)] = np.full(count, kappa)
    return fields, count


def _check_single_element_family(parent, geo):
    """Per-element data needs Houdini and polyfem to agree on element order.

    MSH_Reader emits all tets and then all hexes, while polyfem's reader keeps
    the file's entity-block order; on a mixed mesh the two orders differ and
    per-element data would silently bind to the wrong elements.
    """
    geo_data = parent.node(f"branch_{geo}").geometry()
    mask = _volume_mask(geo_data)
    sizes = set()
    for prim in geo_data.prims():
        if mask is None or mask[prim.number()]:
            sizes.add(len(prim.vertices()))
    if len(sizes) > 1:
        raise hou.NodeError(
            f"Geometry {geo} mixes element types ({sorted(sizes)} vertices), "
            f"and per-element material data cannot be bound safely on a mixed "
            f"mesh: Houdini groups elements by type while PolyFEM keeps the "
            f"file order. Use a single element type, or set the material "
            f"parameters to constants/expressions.")


# One line per drawn reference fiber, centred on the transformed element.
# PolyFEM keeps a0 in simulation/world coordinates when geometry is transformed,
# so only the line centre follows geometry[].transformation. Fibers are line
# fields, so the DTI convention (colour = |direction|) is sign-free.
FIBER_VIZ_VEX = """
if (i@is_volume == 0 || @primnum % max(1, chi("stride")) != 0) {
    removeprim(0, @primnum, 1);
    return;
}
string names[] = split(chs("fiber_attribs"));
float scale = chf("scale");
int mode = chi("color_mode");
vector uniform = chv("uniform_color");
int family = 0;
foreach (string name; names) {
    family++;
    if (!hasprimattrib(0, name)) continue;
    vector d = prim(0, name, @primnum);
    if (length(d) < 1e-12) continue;
    d = normalize(d);
    vector centre = v@centroid;
    int a = addpoint(0, centre - 0.5 * scale * d);
    int b = addpoint(0, centre + 0.5 * scale * d);
    int line = addprim(0, "polyline", a, b);
    vector colour = mode == 0 ? abs(d)
        : mode == 1 ? set(float(family % 3 == 1), float(family % 3 == 2),
                          float(family % 3 == 0))
        : uniform;
    setprimattrib(0, "Cd", line, colour, "set");
    setprimattrib(0, "geometry_num", line, i@geometry_num, "set");
    setprimattrib(0, "fiber_reference_direction", line, d, "set");
    setprimattrib(0, "fiber_source_attribute", line, name, "set");
    setprimattrib(0, "fiber_frame", line,
                  "simulation_world_reference_a0", "set");
}
removeprim(0, @primnum, 1);
"""


def geo_fiber_attribs(parent, geo):
    """Canonical per-element attribute names currently in use on a geometry."""
    names = []
    num_vols_parm = parent.parm(f"num_volumes{geo}")
    if num_vols_parm is None:
        return names
    for vol in range(1, num_vols_parm.eval() + 1):
        for fam, _model in material_families(parent, geo, vol):
            # PolyFEM expressions are not evaluated in Houdini; do not ask the
            # viewport to read an attribute that cannot be reproduced exactly.
            if _slot_parm(parent, geo, vol, fam, "fib_source") == "expression":
                continue
            for side in mirror_sides(parent, geo, vol, fam):
                names.append(fiber_attrib(vol, fam, side))
    return names


def geo_scalar_attribs(parent, geo):
    names = []
    num_vols_parm = parent.parm(f"num_volumes{geo}")
    if num_vols_parm is None:
        return names
    for vol in range(1, num_vols_parm.eval() + 1):
        for fam, model in material_families(parent, geo, vol):
            if model == "HGODispersion" and _slot_parm(
                    parent, geo, vol, fam, "kappa_source") != "expression":
                names.append(scalar_attrib("kappa", vol, fam))
    return names


def _surface_vex(parent, geo):
    """SURFACE_VEX plus per-element material data carried onto the faces.

    The boundary faces are new primitives, so anything stamped on the volume
    elements has to be copied across explicitly for the surface to be able to
    show it.
    """
    fibers = geo_fiber_attribs(parent, geo)
    scalars = geo_scalar_attribs(parent, geo)
    if not fibers and not scalars:
        return SURFACE_VEX

    reads, writes = [], []
    for index, name in enumerate(fibers):
        reads.append(f'vector _v{index} = hasprimattrib(0, "{name}") ? '
                     f'prim(0, "{name}", @primnum) : {{0, 0, 0}};')
        writes.append(f'    setprimattrib(0, "{name}", prim, _v{index}, "set");')
    for index, name in enumerate(scalars):
        reads.append(f'float _f{index} = hasprimattrib(0, "{name}") ? '
                     f'prim(0, "{name}", @primnum) : -1.0;')
        writes.append(f'    setprimattrib(0, "{name}", prim, _f{index}, "set");')

    snippet = SURFACE_VEX.replace(
        "int vcount = primvertexcount(0, @primnum);",
        "\n".join(reads) + "\n\nint vcount = primvertexcount(0, @primnum);")
    # every face creation site gets the same copies
    return snippet.replace(
        '            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");',
        '            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");\n'
        + "\n".join("        " + line.strip() for line in writes))


def _build_fiber_viz(parent, geo, fiberdata):
    """Hedgehog branch: lines are display-only, so they hang off the stamp
    node rather than sitting in the surface chain."""
    viz = parent.node(f"fiberviz_{geo}")
    if viz is None:
        viz = parent.createNode("attribwrangle", f"fiberviz_{geo}")
        for name, template in (
                ("stride", hou.IntParmTemplate("stride", "stride", 1,
                                               default_value=(1,))),
                ("scale", hou.FloatParmTemplate("scale", "scale", 1,
                                                default_value=(0.1,))),
                ("color_mode", hou.IntParmTemplate("color_mode", "color_mode",
                                                   1, default_value=(0,))),
                ("fiber_attribs", hou.StringParmTemplate(
                    "fiber_attribs", "fiber_attribs", 1)),
                ("uniform_color", hou.FloatParmTemplate(
                    "uniform_color", "uniform_color", 3,
                    default_value=(1, 1, 1)))):
            if viz.parm(name) is None and viz.parmTuple(name) is None:
                viz.addSpareParmTuple(template)
    viz.setParms({"class": 1, "snippet": FIBER_VIZ_VEX})
    viz.setInput(0, fiberdata)

    switch = parent.node(f"fibervizswitch_{geo}")
    if switch is None:
        switch = parent.createNode("switch", f"fibervizswitch_{geo}")
        empty = parent.createNode("null", f"nofiberviz_{geo}")
        switch.setInput(0, empty)
        switch.setInput(1, viz)
        switch.parm("input").setExpression(f'ch("../show_fibers{geo}")')
    _merge_all(parent).setNextInput(switch)
    update_fiber_viz(parent, geo)


def update_fiber_viz(parent, geo):
    """Push the current family list and sizing onto the hedgehog wrangle."""
    viz = parent.node(f"fiberviz_{geo}")
    if viz is None:
        return
    names = geo_fiber_attribs(parent, geo)
    count = volume_element_count(parent, geo)
    limit = max(1, parent.evalParm(f"fiber_max_lines{geo}"))
    stride = max(1, -(-count // limit)) if count else 1

    scale = parent.evalParm(f"fiber_scale{geo}")
    mindist = parent.node(f"mindist_{geo}")
    edge = 0.0
    if mindist is not None:
        try:
            values = np.frombuffer(
                mindist.geometry().primFloatAttribValuesAsString(
                    "min_edge_length"), dtype=np.float32)
            if len(values):
                edge = float(values.mean())
        except (hou.OperationFailed, AttributeError):
            edge = 0.0
    viz.setParms({
        "fiber_attribs": " ".join(names),
        "stride": stride,
        "scale": scale * edge if edge else scale,
        "color_mode": parent.evalParm(f"fiber_color_mode{geo}")})
    viz.parmTuple("uniform_color").set(
        parent.parmTuple(f"fiber_color{geo}").eval())


def cook_fiber_data(node):
    """Stamp per-element material data onto the volume elements (Python SOP).

    Failures are warnings, not errors: a mistyped attribute name must not take
    the whole display chain (and with it sideset selection) down. The export
    resolves the same sources and does raise.
    """
    geo_data = node.geometry()
    parent = node.parent()
    geo = _geo_index_of(node.name())
    if geo is None:
        return

    # Recomputed after the transform, and needed by the hedgehog wrangle.
    mask = _volume_mask(geo_data)
    centroids = _prim_centroids(geo_data, None)
    if geo_data.findPrimAttrib("centroid") is None:
        geo_data.addAttrib(hou.attribType.Prim, "centroid", (0.0, 0.0, 0.0),
                           create_local_variable=False)
    geo_data.setPrimFloatAttribValuesFromString(
        "centroid", np.ascontiguousarray(centroids, dtype=np.float64).tobytes(),
        float_type=hou.numericData.Float64)

    try:
        fields, _count = display_material_data(parent, geo)
    except hou.Error as error:
        raise hou.NodeWarning(str(error))
    if not fields:
        return

    if any(np.asarray(values).ndim == 2 for values in fields.values()):
        if geo_data.findGlobalAttrib("polyfem_fiber_reference_frame") is None:
            geo_data.addAttrib(
                hou.attribType.Global, "polyfem_fiber_reference_frame", "",
                create_local_variable=False)
        geo_data.setGlobalAttribValue(
            "polyfem_fiber_reference_frame",
            "simulation_world_reference_a0")

    total = geo_data.intrinsicValue("primitivecount")
    index = np.flatnonzero(mask) if mask is not None else np.arange(total)
    entity_attrib = geo_data.findPrimAttrib("Entity")
    entity_values = np.asarray(
        geo_data.primIntAttribValues("Entity")) if entity_attrib is not None \
        else None
    for name, values in fields.items():
        values = np.asarray(values, dtype=np.float64)
        target = index
        source = values
        owner = re.match(r"pf_(?:fiber|kappa)_(\d+)_", name)
        if owner is not None and entity_values is not None:
            belongs = entity_values[index] == int(owner.group(1))
            target = index[belongs]
            source = values[belongs]
        if values.ndim == 1:
            full = np.full(total, -1.0)
            full[target] = source
            default = -1.0
        else:
            full = np.zeros((total, 3))
            full[target] = source
            default = (0.0, 0.0, 0.0)
        if geo_data.findPrimAttrib(name) is None:
            geo_data.addAttrib(hou.attribType.Prim, name, default,
                               create_local_variable=False)
        geo_data.setPrimFloatAttribValuesFromString(
            name, np.ascontiguousarray(full, dtype=np.float64).tobytes(),
            float_type=hou.numericData.Float64)


def update_fiber_data(parent, geo):
    """Re-resolve the per-element sources and refresh everything showing them."""
    surface = parent.node(f"surface_{geo}")
    if surface is not None:
        surface.parm("snippet").set(_surface_vex(parent, geo))
    fiberdata = parent.node(f"fiberdata_{geo}")
    if fiberdata is not None:
        fiberdata.cook(force=True)
    update_fiber_viz(parent, geo)
    update_surface_color(parent, geo)


def update_surface_color(parent, geo):
    color = parent.node(f"entitycolor_{geo}")
    if color is None:
        return
    mode_parm = parent.parm(f"color_by{geo}")
    color.parm("snippet").set(_entity_color_vex(
        geo, parent.evalParm(f"num_volumes{geo}"),
        mode_parm.evalAsString() if mode_parm is not None else "subdomains",
        geo_fiber_attribs(parent, geo), geo_scalar_attribs(parent, geo)))


def fiber_display_changed(kwargs):
    """Refresh styling and self-heal stale material display data.

    Showing fibers or changing the surface field is the point where users need
    the material structure to be current. Rebuild the generated surface VEX
    and re-cook the stamp SOP here instead of requiring the separate
    "Refresh Per-Element Data" button after every material/source change.
    Pure styling changes keep the lightweight path.
    """
    parent = kwargs["node"]
    geo = kwargs.get("script_multiparm_index")
    if geo is None or parent.node(f"branch_{geo}") is None:
        return
    parent.allowEditingOfContents()
    parm = kwargs.get("parm")
    parm_name = kwargs.get("parm_name") or (
        parm.name() if parm is not None else "")
    if not parm_name or parm_name.startswith(("show_fibers", "color_by")):
        update_fiber_data(parent, geo)
    else:
        update_fiber_viz(parent, geo)
        update_surface_color(parent, geo)


def material_display_changed(kwargs):
    """Keep an already-visible preview live while material controls change."""
    parent = kwargs["node"]
    geo = None
    for key in ("script_multiparm_index3", "script_multiparm_index2",
                "script_multiparm_index"):
        candidate = kwargs.get(key)
        if candidate is not None \
                and parent.parm(f"file_location{candidate}") is not None:
            geo = str(candidate)
            break
    if geo is None or parent.node(f"branch_{geo}") is None:
        return
    color_mode = parent.parm(f"color_by{geo}")
    if parent.evalParm(f"show_fibers{geo}") or (
            color_mode is not None
            and color_mode.evalAsString() != "subdomains"):
        parent.allowEditingOfContents()
        update_fiber_data(parent, geo)


def refresh_material_data(kwargs):
    """Resolve this subdomain's per-element sources now and report what they
    hold, so a bad attribute or file surfaces here instead of at export."""
    parent = kwargs["node"]
    geo = kwargs.get("script_multiparm_index2") or kwargs.get(
        "script_multiparm_index")
    if parent.node(f"branch_{geo}") is None:
        _message("Import geometry first.")
        return
    fields, count = per_element_data(parent, geo)
    update_fiber_data(parent, geo)
    if not fields:
        _status(f"Geometry {geo}: no per-element material data "
                f"({count} elements).")
        return
    summary = []
    for name in sorted(fields):
        values = fields[name]
        if values.ndim == 1:
            summary.append(f"{name}: {values.min():.4g}..{values.max():.4g}")
        else:
            summary.append(f"{name}: {len(values)} directions")
    _status(f"Geometry {geo} ({count} elements) -- " + "; ".join(summary))


FIBER_FILE = "fibers.vtk"
FIBER_FAMILY_FILE = "polyfem_fiber_families.npz"


def per_element_scalar_file(geo, vol, fam, name):
    suffix = "" if fam is None else f"_{fam}"
    return f"pe_{name}_{geo}_{vol}{suffix}.txt"


def _check_composite_consistency(parent):
    """All composite subdomains must declare the same ordered child types.

    PolyFEM builds ONE summed assembler for every body using MaterialSum (from
    whichever body it sees first) and then feeds each body's parameters into
    those children positionally, so a differing stack silently applies the
    wrong parameters to the wrong model instead of failing.
    """
    stacks = {}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if parent.parm(f"is_obstacle{geo}").eval() \
                or not parent.parm(f"is_enabled{geo}").eval():
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            if material_token(parent, geo, vol) != "MaterialSum":
                continue
            stacks[(geo, vol)] = tuple(
                model["type"] for model in composite_models(parent, geo, vol))
    if len(set(stacks.values())) > 1:
        listing = "\n".join(
            f"  geo {geo} subdomain {vol}: {' + '.join(stack)}"
            for (geo, vol), stack in sorted(stacks.items()))
        raise hou.NodeError(
            "Every composite subdomain must use the same ordered model stack "
            "(PolyFEM shares one summed assembler across all of them), but "
            f"they differ:\n{listing}")
    for (geo, vol), stack in stacks.items():
        repeated = {t for t in stack if stack.count(t) > 1}
        if repeated:
            _status(
                f"Note: geo {geo} subdomain {vol} sums two or more "
                f"{'/'.join(sorted(repeated))} models. They are summed "
                f"correctly; their material fields appear in the output as "
                f"<type>_0, <type>_1 (needs the SumModel naming patch).")


def export_per_element_materials(parent, input_dir, output_dir=None):
    """Write solver fields and a ReadPVD companion fiber-family file.

    Every file spans the GLOBAL element range: polyfem indexes them by global
    element id, and with mixed models it evaluates a body's parameters over
    every element when writing output. Elements outside the owning subdomain
    get a harmless placeholder.

    ``fibers.vtk`` remains the solver input. Its directions are reference a0
    values in simulation/world coordinates and, matching PolyFEM, are not
    changed by ``geometry[].transformation``. The compressed companion in the
    output folder additionally records transformed element centroids, this
    coordinate-frame contract, body ids, and every resolvable family (including
    constant directions). ReadPVD uses it only when the VTU does not contain
    PolyFEM material fields.
    """
    table, total = global_element_table(parent)
    if not total:
        return
    fiber_fields = {}
    family_names, family_labels, family_body_ids, family_values = [], [], [], []
    all_centroids = np.zeros((total, 3), dtype=np.float64)
    all_body_ids = np.zeros(total, dtype=np.int64)
    for geo, offset, count in table:
        fields, resolved = per_element_data(parent, geo)
        if resolved != count:
            raise hou.NodeError(
                f"Geometry {geo}: element count changed while exporting "
                f"({resolved} vs {count}); re-cook the node and try again")
        branch_geo = parent.node(f"branch_{geo}").geometry()
        volume_mask = _volume_mask(branch_geo)
        entities = np.frombuffer(
            branch_geo.primIntAttribValuesAsString("Entity"),
            dtype=np.int32)
        if volume_mask is not None:
            entities = entities[volume_mask]
        centroids = _element_centroids(parent, geo)
        all_centroids[offset:offset + count] = centroids
        all_body_ids[offset:offset + count] = \
            1000 * int(geo) + entities.astype(np.int64)

        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            for fam, model in material_families(parent, geo, vol):
                for side in mirror_sides(parent, geo, vol, fam):
                    values = fields.get(fiber_attrib(vol, fam, side))
                    if values is not None:
                        column = fiber_fields.setdefault(
                            fiber_field(geo, vol, fam, side),
                            np.tile([1.0, 0.0, 0.0], (total, 1)))
                        column[offset:offset + count] = values
                    elif _slot_parm(
                            parent, geo, vol, fam, "fib_source") == "constant":
                        direction = np.asarray(parent.parmTuple(
                            _parm_name(geo, vol, fam, "fib_dir")).eval(),
                            dtype=np.float64)
                        values = _normalize_fibers(
                            np.tile(direction, (count, 1)),
                            f"{_slot_label(geo, vol, fam)} fiber")

                    # Expressions are evaluated inside PolyFEM and cannot be
                    # reproduced safely at preprocessing time. Native material
                    # fields will still be picked up when output.material=true.
                    if values is not None:
                        companion = np.zeros((total, 3), dtype=np.float64)
                        owned = entities == int(vol)
                        local = np.asarray(values, dtype=np.float64).copy()
                        local[~owned] = 0.0
                        companion[offset:offset + count] = local
                        family = 0 if fam is None else int(fam)
                        suffix = f"_{side}" if side else ""
                        family_names.append(
                            f"hda_g{geo}_v{vol}_f{family}{suffix}"
                            "_fiber_direction")
                        angle = {"p": " (+theta)", "m": " (-theta)"}.get(
                            side, "")
                        family_labels.append(
                            f"Geometry {geo} / Subdomain {vol} / "
                            f"{model} family {family}{angle}")
                        family_body_ids.append(1000 * int(geo) + int(vol))
                        family_values.append(companion)
                kappa = fields.get(scalar_attrib("kappa", vol, fam))
                if kappa is not None:
                    padded = np.full(
                        total, _slot_parm(parent, geo, vol, fam, "kappa"))
                    padded[offset:offset + count] = kappa
                    np.savetxt(
                        os.path.join(
                            input_dir,
                            per_element_scalar_file(geo, vol, fam, "kappa")),
                        padded, fmt="%.9g")

    if not fiber_fields:
        solver_path = os.path.join(input_dir, FIBER_FILE)
        if os.path.isfile(solver_path):
            os.remove(solver_path)
    else:
        with open(os.path.join(input_dir, FIBER_FILE), "w") as handle:
            handle.write("# vtk DataFile Version 3.0\n")
            handle.write("PolyFEM HDA per-element fiber directions\n")
            handle.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
            handle.write(f"CELL_DATA {total}\n")
            for name in sorted(fiber_fields):
                handle.write(f"VECTORS {name} float\n")
                np.savetxt(handle, fiber_fields[name], fmt="%.9f")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        companion_path = os.path.join(output_dir, FIBER_FAMILY_FILE)
        if family_values:
            temporary = companion_path + ".tmp"
            with open(temporary, "wb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray([2], dtype=np.int32),
                    direction_frame=np.asarray(
                        ["simulation_world_reference_a0"], dtype=np.str_),
                    centroids=all_centroids,
                    body_ids=all_body_ids,
                    family_names=np.asarray(family_names, dtype=np.str_),
                    family_labels=np.asarray(family_labels, dtype=np.str_),
                    family_body_ids=np.asarray(
                        family_body_ids, dtype=np.int64),
                    directions=np.stack(family_values, axis=0))
            os.replace(temporary, companion_path)
        elif os.path.isfile(companion_path):
            os.remove(companion_path)


# =============================================================================
# JSON builders
# =============================================================================


def build_geometry_and_materials(parent, data, input_dir):
    num_geos = parent.evalParm("num_geos")
    if num_geos == 0:
        raise hou.NodeError("No input geometry provided!")
    data["geometry"] = []
    data["materials"] = []
    orders = []

    _check_composite_consistency(parent)
    export_per_element_materials(
        parent, input_dir,
        os.path.join(os.path.dirname(input_dir), "output"))

    for geo in range(1, num_geos + 1):
        is_obstacle = bool(parent.parm(f"is_obstacle{geo}").eval())
        enabled = bool(parent.parm(f"is_enabled{geo}").eval())
        path_parm = parent.parm(f"file_location{geo}").eval()
        if not path_parm:
            raise hou.NodeError(f"No file location for geo {geo}!")
        mesh_ref = os.path.basename(path_parm)
        if not os.path.isfile(os.path.join(input_dir, mesh_ref)):
            shutil.copy(path_parm, os.path.join(input_dir, mesh_ref))

        transform_node = parent.node(f"transform_{geo}")
        T, R, S = decompose_rowvec_xyz_4x4(
            transform_node.geometry().attribValue("xform"))
        transformation = {"translation": list(T), "rotation": list(R),
                          "scale": list(S)}

        if is_obstacle:
            data["geometry"].append({
                "mesh": mesh_ref, "enabled": enabled, "is_obstacle": True,
                "surface_selection": obstacle_id(geo),
                "transformation": transformation})
            continue

        num_vols = parent.evalParm(f"num_volumes{geo}")
        volumefile = export_volumes(parent, geo, input_dir)
        surface_sel, point_sel = export_sidesets(
            parent, geo, num_vols, input_dir)

        geo_data = {"mesh": mesh_ref, "enabled": enabled, "is_obstacle": False,
                    "volume_selection": volumefile,
                    "transformation": transformation}
        if surface_sel:
            geo_data["surface_selection"] = surface_sel
        if point_sel:
            geo_data["point_selection"] = point_sel
        data["geometry"].append(geo_data)

        for vol in range(1, num_vols + 1):
            data["materials"].append(build_material(parent, geo, vol))
            orders.append({"id": vol_id(geo, vol),
                           "order": parent.evalParm(f"mainOrder{geo}_{vol}") + 1})
    return orders


def _damping(parent, geo, vol, material):
    damping_parm = parent.parm(f"damping{geo}_{vol}")
    if damping_parm is not None and not damping_parm.isDisabled() \
            and damping_parm.eval():
        material["phi"] = parent.evalParm(f"phi{geo}_{vol}")
        material["psi"] = parent.evalParm(f"psi{geo}_{vol}")
    return material


def _isotropic_model(parent, geo, vol, token):
    """Body of an isotropic model: no id/rho, so it also serves as a
    MaterialSum child (children carry neither)."""
    def ep():
        return (parent.evalParm(f"E{geo}_{vol}"),
                parent.evalParm(f"nu{geo}_{vol}"))

    if token in ("LinearElasticity", "NeoHookean", "FixedCorotational",
                 "IsochoricNeoHookean", "IncompressibleLinearElasticity"):
        E, nu = ep()
        model = {"type": token, "E": E, "nu": nu}
        if token != "IncompressibleLinearElasticity":
            _damping(parent, geo, vol, model)
        return model
    if token in ("HookeLinearElasticity", "SaintVenant"):
        model = {"type": token,
                 "elasticity_tensor": parse_vector(
                     parent.evalParm(f"elast_tensor{geo}_{vol}"),
                     "elasticity_tensor"),
                 "fiber_direction": parse_vector(
                     parent.evalParm(f"material_coord{geo}_{vol}"),
                     "fiber_direction")}
        if token == "SaintVenant":
            _damping(parent, geo, vol, model)
        return model
    if token == "IncompressibleOgden":
        return {"type": token,
                "c": parse_vector(parent.evalParm(f"ogdenC{geo}_{vol}"), "c"),
                "m": parse_vector(parent.evalParm(f"ogdenM{geo}_{vol}"), "m"),
                "k": parent.evalParm(f"bulk{geo}_{vol}")}
    if token == "UnconstrainedOgden":
        return {"type": token,
                "alphas": parse_vector(
                    parent.evalParm(f"ogden_alphas{geo}_{vol}"), "alphas"),
                "mus": parse_vector(
                    parent.evalParm(f"ogden_mus{geo}_{vol}"), "mus"),
                "Ds": parse_vector(
                    parent.evalParm(f"ogden_ds{geo}_{vol}"), "Ds")}
    if token == "MooneyRivlin":
        return {"type": token,
                "c1": parent.evalParm(f"c1{geo}_{vol}"),
                "c2": parent.evalParm(f"c2{geo}_{vol}"),
                "k": parent.evalParm(f"bulk{geo}_{vol}")}
    if token in ("MooneyRivlin3Param", "MooneyRivlin3ParamSymbolic"):
        return {"type": token,
                "c1": parent.evalParm(f"c1{geo}_{vol}"),
                "c2": parent.evalParm(f"c2{geo}_{vol}"),
                "c3": parent.evalParm(f"c3{geo}_{vol}"),
                "d1": parent.evalParm(f"d1{geo}_{vol}")}
    raise hou.NodeError(f"Unsupported isotropic material '{token}'")


def _number_or_expression(text, label):
    """A polyfem scalar the user typed: number when it parses as one,
    otherwise the string (an expression of x,y,z,t)."""
    text = str(text).strip()
    if not text:
        raise hou.NodeError(f"{label} is empty")
    try:
        return float(text)
    except ValueError:
        return text


def fiber_direction_json(parent, geo, vol, fam, side=""):
    """"fiber_direction" for one family: constant, expression, or a reference
    into the shared per-element file."""
    source = _slot_parm(parent, geo, vol, fam, "fib_source")
    if source in PER_ELEMENT_FIBER_SOURCES or side:
        return {"type": "per_element_file", "path": FIBER_FILE,
                "field": fiber_field(geo, vol, fam, side)}
    if source == "expression":
        return parse_vector(
            _slot_parm(parent, geo, vol, fam, "fib_expr"), "fiber_direction")
    return list(parent.parmTuple(_parm_name(geo, vol, fam, "fib_dir")).eval())


def scalar_json(parent, geo, vol, fam, name):
    """A scalar material parameter: number, expression, or per-element file."""
    source = _slot_parm(parent, geo, vol, fam, f"{name}_source")
    if source in ("attribute", "file"):
        return per_element_scalar_file(geo, vol, fam, name)
    if source == "expression":
        return _number_or_expression(
            _slot_parm(parent, geo, vol, fam, f"{name}_expr"),
            f"{_slot_label(geo, vol, fam)} {name}")
    return _slot_parm(parent, geo, vol, fam, name)


def _fiber_model(parent, geo, vol, fam, token, side=""):
    """One fiber family as a polyfem model object (no id/rho)."""
    model = {"type": token,
             "fiber_direction": fiber_direction_json(parent, geo, vol, fam, side)}
    if token == "ActiveFiber":
        model["Tmax"] = _slot_parm(parent, geo, vol, fam, "Tmax")
        model["activation"] = _number_or_expression(
            _slot_parm(parent, geo, vol, fam, "activation"),
            f"{_slot_label(geo, vol, fam)} activation")
        return model
    k2 = _slot_parm(parent, geo, vol, fam, "k2")
    if k2 <= 0:
        raise hou.NodeError(
            f"{_slot_label(geo, vol, fam)}: k2 must be greater than zero "
            f"(the model divides by it)")
    model["k1"] = _slot_parm(parent, geo, vol, fam, "k1")
    model["k2"] = k2
    if token == "HGODispersion":
        model["kappa"] = scalar_json(parent, geo, vol, fam, "kappa")
        k_chi = _slot_parm(parent, geo, vol, fam, "k_chi")
        if abs(k_chi - 100.0) > 1e-9:
            model["k_chi"] = k_chi
    return model


def composite_models(parent, geo, vol):
    """The ordered "models" list of a composite, families expanded.

    A symmetric family becomes two children sharing every parameter but their
    rotated directions, which is exactly what polyfem sums.
    """
    models = []
    matrix = parent.parm(f"matrix_model{geo}_{vol}").evalAsString()
    if matrix != "None":
        models.append(_isotropic_model(parent, geo, vol, matrix))
    for fam, token in material_families(parent, geo, vol):
        for side in mirror_sides(parent, geo, vol, fam):
            models.append(_fiber_model(parent, geo, vol, fam, token, side))
    if parent.evalParm(f"volume_penalty{geo}_{vol}"):
        models.append({"type": "VolumePenalty",
                       "k": parent.evalParm(f"vp_k{geo}_{vol}")})
    extra = parent.evalParm(f"extra_models_json{geo}_{vol}").strip()
    if extra:
        try:
            parsed = json.loads(extra)
        except ValueError as error:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: extra models are not valid "
                f"JSON ({error})")
        if not isinstance(parsed, list):
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: extra models must be a JSON "
                f"list of model objects")
        models.extend(parsed)
    return models


def build_material(parent, geo, vol):
    token = material_token(parent, geo, vol)
    material = {"id": vol_id(geo, vol), "rho": parent.evalParm(f"rho{geo}_{vol}")}

    if token == "MaterialSum":
        models = composite_models(parent, geo, vol)
        if not models:
            raise hou.NodeError(
                f"{_slot_label(geo, vol, None)}: a composite material needs a "
                f"matrix model or at least one fiber family")
        material["type"] = "MaterialSum"
        material["models"] = models
        return material
    if token in FIBER_MODELS:
        material.update(_fiber_model(parent, geo, vol, None, token))
        return material
    material.update(_isotropic_model(parent, geo, vol, token))
    return material


def build_time(parent, data):
    quasistatic = bool(parent.evalParm("quasistatic"))
    t0 = parent.evalParm("t0")
    integrator = parent.evalParm("integrator")
    integrator_dict = {0: "ImplicitEuler", 1: "BDF1", 2: "BDF2", 3: "BDF3",
                       4: "BDF4", 5: "BDF5", 6: "BDF6", 7: "ImplicitNewmark"}
    if integrator == 7:
        integrator = {"type": "ImplicitNewmark",
                      "beta": parent.evalParm("newmark_beta"),
                      "gamma": parent.evalParm("newmark_gamma")}
    else:
        integrator = integrator_dict[integrator]

    time = {"t0": t0, "integrator": integrator, "quasistatic": quasistatic}
    tend_on = parent.evalParm("end_time_bool")
    dt_on = parent.evalParm("time_inc_bool")
    steps_on = parent.evalParm("num_timesteps_bool")
    if tend_on:
        time["tend"] = parent.evalParm("tend")
    if dt_on:
        time["dt"] = parent.evalParm("dt")
    if steps_on and not (tend_on and dt_on):
        time["time_steps"] = int(parent.evalParm("num_timesteps"))
    if "tend" not in time and "time_steps" not in time:
        time["tend"] = parent.evalParm("tend")
    if "dt" not in time and "time_steps" not in time:
        time["dt"] = parent.evalParm("dt")
    data["time"] = time


def build_contact(parent, data):
    if not parent.evalParm("enable"):
        data["contact"] = {"enabled": False}
        return
    remesh_enabled = parent.evalParm("remeshing_enabled")
    area_weighted = parent.evalParm("area_weighted")
    if not area_weighted and remesh_enabled:
        _message(
            "Note: Convergent formulation selected to support remeshing.")
    if remesh_enabled:
        area_weighted = 1
        parent.setParms({"area_weighted": 1})

    adhesion = {
        "adhesion_enabled": bool(parent.evalParm("adhesion_enable")),
        "dhat_p": parent.evalParm("dhat_p"),
        "dhat_a": parent.evalParm("dhat_a"),
        "adhesion_strength": parent.evalParm("adhesion_strength"),
        "tangential_adhesion_coefficient": parent.evalParm("tangent_coeff"),
        "epsa": parent.evalParm("adhesion_epsa")}
    data["contact"] = {
        "enabled": True,
        "dhat": parent.evalParm("dhat"),
        "epsv": parent.evalParm("epsv"),
        "friction_coefficient": parent.evalParm("cof"),
        "use_convergent_formulation": bool(area_weighted),
        "use_gcp_formulation": bool(parent.evalParm("gcp_enable")),
        "alpha_n": parent.evalParm("alpha_n"),
        "alpha_t": parent.evalParm("alpha_t"),
        "min_distance_ratio": parent.evalParm("min_dist_ratio"),
        "use_adaptive_dhat": bool(parent.evalParm("adapt_dhat")),
        "adhesion": adhesion}


def build_conditions(parent, data):
    solution, velocity, acceleration = [], [], []
    dirichlet, neumann, neumann_normal = [], [], []
    pressure_boundary, pressure_cavity, obstacle = [], [], []
    num_geos = parent.evalParm("num_geos")

    for geo in range(1, num_geos + 1):
        is_obstacle = bool(parent.parm(f"is_obstacle{geo}").eval())
        if is_obstacle:
            vector = parse_vector(
                parent.evalParm(f"obstacle_disp{geo}"), f"obstacle {geo}")
            obstacle.append({"id": obstacle_id(geo), "value": vector})
            continue
        num_vols = parent.evalParm(f"num_volumes{geo}")
        for vol in range(1, num_vols + 1):
            for l in range(1, parent.evalParm(
                    f"initial_conditions{geo}_{vol}") + 1):
                vector = parse_vector(
                    parent.evalParm(f"condition_vector_{geo}_{vol}_{l}"),
                    f"initial condition {geo}/{vol}/{l}")
                ctype = parent.evalParm(f"conditiontype{geo}_{vol}_{l}")
                target = (solution, velocity, acceleration)[ctype]
                target.append({"id": vol_id(geo, vol), "value": vector})
            sidesets = parent.evalParm(f"sideset_selection{geo}_{vol}")
            for j in range(1, sidesets + 1):
                for k in range(1, parent.evalParm(
                        f"Boundary_Condition__{geo}_{vol}_{j}") + 1):
                    bid = boundary_id(geo, vol, j, k)
                    btype = parent.evalParm(
                        f"boundary_type{geo}_{vol}_{j}_{k}")
                    if btype in (0, 1):
                        vector = parse_vector(
                            parent.evalParm(f"vector_{geo}_{vol}_{j}_{k}"),
                            f"BC {geo}/{vol}/{j}/{k}")
                        if btype == 0:
                            dims = [bool(parent.evalParm(
                                f"{ax}_dimension{geo}_{vol}_{j}_{k}"))
                                for ax in "xyz"]
                            dirichlet.append({"id": bid, "value": vector,
                                              "dimension": dims})
                        else:
                            neumann.append({"id": bid, "value": vector})
                    else:
                        value = parse_vector(
                            parent.evalParm(f"value_{geo}_{vol}_{j}_{k}"),
                            f"BC {geo}/{vol}/{j}/{k}")
                        target = {2: neumann_normal, 3: pressure_boundary,
                                  4: pressure_cavity}[btype]
                        target.append({"id": bid, "value": value})

    if solution or velocity or acceleration:
        data["initial_conditions"] = {}
        for key, lst in (("solution", solution), ("velocity", velocity),
                         ("acceleration", acceleration)):
            if lst:
                data["initial_conditions"][key] = lst

    rhs = parent.evalParm("RHS")
    bc = {}
    for key, lst in (("dirichlet_boundary", dirichlet),
                     ("neumann_boundary", neumann),
                     ("normal_aligned_neumann_boundary", neumann_normal),
                     ("pressure_boundary", pressure_boundary),
                     ("pressure_cavity", pressure_cavity),
                     ("obstacle_displacements", obstacle)):
        if lst:
            bc[key] = lst
    if rhs:
        bc["rhs"] = parse_vector(rhs, "rhs")
    if bc:
        data["boundary_conditions"] = bc


REMESH_DEFAULTS = {
    "enabled": False,
    "type": "physics",
    "split": {
        "enabled": True, "acceptance_tolerance": 1e-3,
        "culling_threshold": 0.95, "max_depth": 3,
        "min_edge_length": 1e-6},
    "collapse": {
        "enabled": True, "acceptance_tolerance": -1e-8,
        "culling_threshold": 0.01, "max_depth": 3,
        "rel_max_edge_length": 1.0, "abs_max_edge_length": 1e100},
    "swap": {
        "enabled": False, "acceptance_tolerance": -1e-8, "max_depth": 3},
    "smooth": {
        "enabled": False, "acceptance_tolerance": -1e-8, "max_iters": 1},
    "local_relaxation": {
        "local_mesh_n_ring": 2, "local_mesh_rel_area": 0.01,
        "max_nl_iterations": 1},
}


def _remesh_stage(parent, enabled_parm, fields):
    enabled = bool(parent.evalParm(enabled_parm))
    stage = {"enabled": enabled}
    if enabled:
        stage.update({key: parent.evalParm(parm) for key, parm in fields})
    return stage


def build_remesh(parent):
    """Serialize the HDA remesh controls to PolyFEM's /space/remesh schema."""
    enabled = bool(parent.evalParm("remeshing_enabled"))
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "type": "physics" if parent.evalParm("remesh_type") == 0
                else "sizing_field",
        "split": _remesh_stage(parent, "split_enabled", (
            ("acceptance_tolerance", "split_acceptance_tol"),
            ("culling_threshold", "split_culling_threshold"),
            ("max_depth", "split_max_depth"),
            ("min_edge_length", "min_edge_length"))),
        "collapse": _remesh_stage(parent, "collapse_enabled", (
            ("acceptance_tolerance", "collapse_acceptance_tol"),
            ("culling_threshold", "collapse_culling_threshold"),
            ("max_depth", "collapse_max_depth"),
            ("rel_max_edge_length", "rel_max_edge_length"),
            ("abs_max_edge_length", "abs_max_edge_length"))),
        "swap": _remesh_stage(parent, "swap_enabled2", (
            ("acceptance_tolerance", "swap_acceptance_tol2"),
            ("max_depth", "swap_max_depth2"))),
        "smooth": _remesh_stage(parent, "smooth_enabled", (
            ("acceptance_tolerance", "smooth_acceptance_tol"),
            ("max_iters", "smooth_max_iters"))),
        "local_relaxation": {
            "local_mesh_n_ring": parent.evalParm("local_mesh_n_ring"),
            "local_mesh_rel_area": parent.evalParm("local_mesh_rel_area"),
            "max_nl_iterations": parent.evalParm("max_nl_iterations")},
    }


def build_space(parent, data, orders):
    space = {"discr_order": orders,
             "remesh": build_remesh(parent)}
    data["space"] = space


def _menu_token(parent, name):
    """Token string of an ordinal menu parameter (evalParm gives the index)."""
    return parent.parm(name).evalAsString()


def build_linear_solver(parent):
    """The /solver/linear block: only what the chosen solver actually reads.

    "auto" leaves the choice to PolyFEM/PolySolve (the best direct solver
    compiled into the binary), which is what every scene got before the
    linear settings were wired at all.
    """
    solver = _menu_token(parent, "solver")
    if solver == "auto":
        return None
    # PolyFEM validates the choice against the solvers compiled into the
    # binary and aborts otherwise; with the overwrite flag it logs a warning
    # and falls back to its default instead.
    linear = {"solver": solver, "enable_overwrite_solver": True}
    iterative_eigen = ("Eigen::ConjugateGradient",
                       "Eigen::LeastSquaresConjugateGradient", "Eigen::DGMRES",
                       "Eigen::BiCGSTAB", "Eigen::GMRES", "Eigen::MINRES")
    if solver in iterative_eigen:
        linear["precond"] = _menu_token(parent, "precond")
        linear[solver] = {"max_iter": parent.evalParm("max_iter"),
                          "tolerance": parent.evalParm("tolerance")}
    elif solver == "Pardiso":
        linear["Pardiso"] = {"mtype": int(_menu_token(parent, "mtype"))}
    elif solver == "Hypre":
        linear["Hypre"] = {
            "max_iter": parent.evalParm("max_iter_hypre"),
            "pre_max_iter": parent.evalParm("pre_max_iter_hypre"),
            "tolerance": parent.evalParm("tolerance_hypre_AMGCL"),
            "theta": parent.evalParm("theta_hypre"),
            "nodal_coarsening":
                bool(parent.evalParm("nodal_coarsening_hypre")),
            # Houdini scenes are 3-D; dimension > 1 switches BoomerAMG to
            # its systems (elasticity) settings, which the default of 1
            # never does.
            "dimension": 3}
    elif solver == "AMGCL":
        linear["AMGCL"] = {
            "solver": {"maxiter": parent.evalParm("max_iter"),
                       "tol": parent.evalParm("tolerance_hypre_AMGCL"),
                       "type": _menu_token(parent, "solver_type")},
            "precond": {
                "class": _menu_token(parent, "class"),
                "max_levels": parent.evalParm("max_levels"),
                "direct_coarse": bool(parent.evalParm("direct_coarse")),
                "ncycle": parent.evalParm("ncycle"),
                "relax": {"degree": parent.evalParm("degree"),
                          "type": _menu_token(parent, "relax_type"),
                          "power_iters": parent.evalParm("power_iters"),
                          "higher": parent.evalParm("higher"),
                          "lower": parent.evalParm("lower"),
                          "scale": bool(parent.evalParm("scale_amgcl"))},
                "coarsening": {
                    "type": _menu_token(parent, "coarsening_type"),
                    "estimate_spectral_radius":
                        bool(parent.evalParm("estimate_spectral_radius")),
                    "relax": parent.evalParm("coarse_relax"),
                    "aggr": {"eps_strong": parent.evalParm("eps_strong")}}}}
    return linear


def build_nonlinear_solver(parent):
    """The /solver/nonlinear block (rescaled-tolerances polysolve spec)."""
    ls = {"method": _menu_token(parent, "method"),
          "use_grad_norm_tol": parent.evalParm("use_grad_norm_tol"),
          "min_step_size": parent.evalParm("min_step_size"),
          "max_step_size_iter": parent.evalParm("max_step_size_iter"),
          "min_step_size_final": parent.evalParm("min_step_size_final"),
          "max_step_size_iter_final": parent.evalParm("max_step_size_iter_final"),
          "default_init_step_size": parent.evalParm("default_init_step_size"),
          "step_ratio": parent.evalParm("step_ratio"),
          "Armijo": {"c": parent.evalParm("armijo_c"),
                     "roundoff_tolerance":
                         parent.evalParm("armijo_roundoff_tolerance")},
          "RobustArmijo": {"delta_relative_tolerance":
                           parent.evalParm("delta_relative_tolerance")}}

    method = _menu_token(parent, "solver_nl")
    nonlinear = {
        "solver": method,
        "max_iterations": parent.evalParm("max_iterations"),
        "iterations_per_strategy": parent.evalParm("iterations_per_strategy"),
        "allow_out_of_iterations":
            bool(parent.evalParm("allow_out_of_iterations")),
        "x_delta_tol": parent.evalParm("x_delta"),
        "rel_x_delta_tol": parent.evalParm("rel_x_delta_tol"),
        "grad_norm_tol": parent.evalParm("grad_norm"),
        "rel_grad_norm_tol": parent.evalParm("rel_grad_norm_tol"),
        "norm_type": _menu_token(parent, "norm_type"),
        "first_grad_norm_tol": parent.evalParm("first_grad_norm_tol"),
        "newton_decrement_tol": parent.evalParm("newton_decrement_tol"),
        "allow_non_grad_convergence":
            bool(parent.evalParm("allow_non_grad_convergence")),
        "line_search": ls,
        "advanced": {"f_delta_tol": parent.evalParm("f_delta"),
                     "f_delta_step_tol": parent.evalParm("f_delta_step_tol"),
                     "derivative_along_delta_x_tol":
                         parent.evalParm("derivative_along_delta_x_tol"),
                     "apply_gradient_fd":
                         _menu_token(parent, "apply_gradient_fd"),
                     "gradient_fd_eps": parent.evalParm("gradient_fd_eps")}}
    # Per-method settings: written only for the chosen method so the file
    # says exactly what the run used.
    if method in ("Newton", "DenseNewton"):
        nonlinear[method] = {
            "residual_tolerance": parent.evalParm("residual_tolerance"),
            "reg_weight_min": parent.evalParm("reg_weight_min"),
            "reg_weight_max": parent.evalParm("reg_weight_max"),
            "reg_weight_inc": parent.evalParm("reg_weight_inc"),
            "force_psd_projection":
                bool(parent.evalParm("force_psd_projection")),
            "use_psd_projection": bool(parent.evalParm("use_psd_projection")),
            "use_psd_projection_in_regularized":
                bool(parent.evalParm("use_psd_projection_reg"))}
    elif method in ("L-BFGS", "L-BFGS-B"):
        nonlinear[method] = {"history_size": parent.evalParm("history_size")}
    elif method in ("ADAM", "StochasticADAM"):
        nonlinear[method] = {"alpha": parent.evalParm("adam_alpha"),
                             "beta_1": parent.evalParm("adam_beta1"),
                             "beta_2": parent.evalParm("adam_beta2"),
                             "epsilon": parent.evalParm("adam_epsilon")}
        if method == "StochasticADAM":
            nonlinear[method]["erase_component_probability"] = \
                parent.evalParm("erase_component_probability")
    elif method == "StochasticGradientDescent":
        nonlinear[method] = {"erase_component_probability":
                             parent.evalParm("erase_component_probability")}
    return nonlinear


def build_solver(parent, data):
    nonlinear = build_nonlinear_solver(parent)

    # augmented lagrangian (un-hardcoded vs 1.2)
    if parent.evalParm("al_hessian_scaled"):
        al = {"initial_weight": "hessian_scaled",
              "initial_weight_multiplier":
                  parent.evalParm("al_weight_multiplier")}
    else:
        al = {"initial_weight": parent.evalParm("al_initial_weight")}
    al.update({"scaling": parent.evalParm("al_scaling"),
               "max_weight": parent.evalParm("al_max_weight"),
               "eta": parent.evalParm("al_eta"),
               "lumping": _menu_token(parent, "al_lumping")})
    # The AL (boundary-condition preparation) phase reuses the nonlinear
    # settings with its own iteration budget.
    if parent.evalParm("al_max_iterations") != nonlinear["max_iterations"]:
        al["nonlinear"] = {
            "max_iterations": parent.evalParm("al_max_iterations")}

    # contact solver block -- current-branch schema
    broad_phase_dict = {0: "hash_grid", 1: "brute_force", 2: "spatial_hash",
                        3: "BVH", 4: "sweep_and_prune",
                        5: "sweep_and_tiniest_queue"}
    contact = {
        "CCD": {"broad_phase": broad_phase_dict[parent.evalParm("broad_phase")],
                "tolerance": parent.evalParm("CCD_tolerance"),
                "max_iterations": parent.evalParm("ccd_max_iterations")},
        "friction_iterations": parent.evalParm("friction_iterations"),
        "tangential_adhesion_iterations": parent.evalParm("tangent_iter"),
        "friction_convergence_tol":
            parent.evalParm("friction_convergence_tol")}

    barrier_mode = parent.evalParm("barrier_mode")
    if barrier_mode == 0:  # semi-implicit
        contact["barrier_stiffness"] = "semi_implicit"
        contact["semi_implicit"] = {
            "refresh_interval": parent.evalParm("si_refresh_interval"),
            "trim_lower": parent.evalParm("si_trim_lower"),
            "trim_upper": parent.evalParm("si_trim_upper"),
            "trim_factor": parent.evalParm("si_trim_factor"),
            "kappa_spread": parent.evalParm("si_kappa_spread"),
            "kappa_min": parent.evalParm("si_kappa_min"),
            "conditioning_cap": parent.evalParm("si_conditioning_cap"),
            "controller_interval": parent.evalParm("si_controller_interval"),
            "trial_displacement_cap":
                parent.evalParm("si_trial_displacement_cap"),
            "force_continuation":
                bool(parent.evalParm("si_force_continuation")),
            "continuation_max_ratio":
                parent.evalParm("si_continuation_max_ratio"),
            "coefficient_identity":
                _menu_token(parent, "si_coefficient_identity"),
            "restart": {
                "enabled": bool(parent.evalParm("si_restart_enabled")),
                "alpha_threshold": parent.evalParm("si_alpha_threshold"),
                "patience": parent.evalParm("si_patience"),
                "min_iterations": parent.evalParm("si_min_iterations"),
                "soft_iteration_limit":
                    parent.evalParm("si_soft_iteration_limit"),
                "max_restarts": parent.evalParm("si_max_restarts"),
                "stall_trim_factor":
                    parent.evalParm("si_stall_trim_factor")}}
    elif barrier_mode == 1:  # classic adaptive
        contact["barrier_stiffness"] = "adaptive"
        contact["initial_barrier_stiffness"] = \
            parent.evalParm("initial_barrier_stiffness")
    else:  # fixed
        contact["barrier_stiffness"] = parent.evalParm("barrier_stiffness")

    rayleigh = []
    form_dict = {0: "elasticity", 1: "contact", 2: "friction"}
    for i in range(1, parent.evalParm("num_rayleigh") + 1):
        rayleigh.append({
            "form": form_dict[parent.evalParm(f"rayleigh_form{i}")],
            "stiffness_ratio": parent.evalParm(f"stiffness_ratio{i}"),
            "lagging_iterations": parent.evalParm(f"rayleigh_lagging{i}")})

    solver = {"max_threads": parent.evalParm("max_threads"),
              "nonlinear": nonlinear,
              "augmented_lagrangian": al,
              "contact": contact,
              "advanced": {
                  "cache_size": parent.evalParm("cache_size"),
                  "lump_mass_matrix":
                      bool(parent.evalParm("lump_mass_matrix")),
                  "lagged_regularization_weight":
                      parent.evalParm("lagged_regularization_weight"),
                  "lagged_regularization_iterations":
                      parent.evalParm("lagged_regularization_iterations"),
                  "check_inversion": _menu_token(parent, "inversion_method"),
                  "jacobian_threshold":
                      parent.evalParm("jacobian_threshold")}}
    linear = build_linear_solver(parent)
    if linear is not None:
        solver["linear"] = linear
    if rayleigh:
        solver["rayleigh_damping"] = rayleigh
    data["solver"] = solver


def build_output(parent, data):
    def b(name):
        return bool(parent.evalParm(name))

    contact_forces = b("contact_forces_fields")
    friction_forces = b("friction_forces_fields")
    normal_adh = b("normal_adhesion_forces_fields")
    tang_adh = b("tangential_adhesion_forces_fields")
    surface = (b("normals_fields") or b("displaced_normals_fields")
               or b("sidesets_fields")
               or b("solution_grad_fields") or contact_forces
               or friction_forces or normal_adh or tang_adh
               or b("adaptive_dhat"))
    jacobian_validity = b("validity_fields")
    volume = (b("velocity_fields") or b("acceleration_fields")
              or b("nodes_field") or b("forces_fields") or jacobian_validity)
    points = b("paraview_points") or jacobian_validity

    # Minimal-fields mode: whitelist only the derivation basis the readPVD
    # HDA needs (solution + F + Cauchy stress) plus whatever force/surface
    # arrays are enabled. Everything readPVD can derive on load (von Mises,
    # PK1/PK2, strains, invariants, and every *_avg variant) is omitted --
    # PolyFEM then skips computing them entirely, shrinking the vtu files.
    # The whitelist is global across the volume/surface/contact blocks.
    # Material-field export uses assembler-specific array names we cannot
    # enumerate safely, so that option falls back to full output.
    minimal = b("minimal_fields") and not b("materials_fields")
    export_fields = []
    if minimal:
        export_fields = ["solution"]
        if b("scalar_values") or b("tensor_values"):
            export_fields += ["F", "cauchy_stess"]  # [sic] PolyFEM spelling
        if b("normals_fields"):
            export_fields += ["normals", "displaced_normals"]
        elif b("displaced_normals_fields"):
            export_fields.append("displaced_normals")
        if b("sidesets_fields"):
            export_fields.append("sidesets")
        if b("solution_grad_fields"):
            export_fields.append("solution_gradient")
        for enabled, name in (
                (contact_forces, "contact_forces"),
                (friction_forces, "friction_forces"),
                (normal_adh, "normal_adhesion_forces"),
                (tang_adh, "tangential_adhesion_forces"),
                (b("adaptive_dhat"), "adaptive_dhat"),
                (b("body_ids_fields"), "body_ids"),
                (b("velocity_fields"), "velocity"),
                (b("acceleration_fields"), "acceleration"),
                (b("discr_fields"), "discr"),
                (b("nodes_field"), "nodes"),
                (jacobian_validity, "validity")):
            if enabled:
                export_fields.append(name)

    options = {
        "use_hdf5": b("use_hdf5"), "material": b("materials_fields"),
        "body_ids": b("body_ids_fields"), "contact_forces": contact_forces,
        "friction_forces": friction_forces,
        "normal_adhesion_forces": normal_adh,
        "tangential_adhesion_forces": tang_adh,
        "velocity": b("velocity_fields"),
        "acceleration": b("acceleration_fields"),
        # the whitelist already excludes von_mises; forcing tensor on /
        # scalar off makes the F + cauchy basis independent of the toggles
        "scalar_values": False if minimal else b("scalar_values"),
        "tensor_values": True if minimal else b("tensor_values"),
        "discretization_order": b("discr_fields"),
        "nodes": b("nodes_field"), "forces": b("forces_fields"),
        "force_high_order": b("forces_HO_output"),
        "jacobian_validity": jacobian_validity}
    paraview = {
        "file_name": "../output/" + parent.evalParm("paraview_file_name"),
        "vismesh_rel_area": parent.evalParm("vismesh_rel_area"),
        "skip_frame": parent.evalParm("skip_frame"),
        "high_order_mesh": b("high_order_mesh"),
        "volume": volume or True,  # always write the volume mesh
        "surface": surface, "wireframe": b("paraview_wireframe"),
        "points": points, "options": options}
    if export_fields:
        paraview["fields"] = export_fields

    data_output = {}
    for parm, key, fname in (
            ("solution_file", "solution", "solution.txt"),
            ("stress_mat_file", "stress_mat", "stress_mat.txt"),
            ("state_file", "state", "state.txt"),
            ("rest_mesh_file", "rest_mesh", "rest_mesh.msh"),
            ("mises_file", "mises", "mises.txt"),
            ("nodes", "nodes", "nodes.txt")):
        if b(parm):
            data_output[key] = "../output/" + fname

    advanced = {
        "timestep_prefix": parent.evalParm("timestep_prefix"),
        "sol_on_grid": parent.evalParm("sol_on_grid"),
        "compute_error": b("compute_error"),
        "sol_at_node": parent.evalParm("sol_at_node"),
        "vis_boundary_only": b("vis_boundary_only"),
        "curved_mesh_size": b("curved_mesh_size"),
        "save_solve_sequence_debug": b("save_solve_sequence_debug"),
        "save_ccd_debug_meshes": b("save_ccd_debug_meshes"),
        "save_time_sequence": b("save_time_sequence"),
        "save_nl_solve_sequence": b("save_nl_solve_sequence"),
        "spectrum": b("spectrum")}

    data["output"] = {"directory": "../output/", "paraview": paraview,
                      "data": data_output, "advanced": advanced}
    if b("output_json"):
        data["output"]["json"] = "../output/output.json"
    if b("restart_json"):
        data["output"]["restart_json"] = "../output/restart.json"
    log = {"level": parent.evalParm("log_level"),
           "quiet": b("log_quiet")}
    if b("write_log"):
        log["path"] = "../output/log.txt"
    data["output"]["log"] = log

    units_on = parent.evalParm("units")
    if units_on:
        data["units"] = {"length": parent.evalParm("length"),
                         "mass": parent.evalParm("mass"),
                         "time": parent.evalParm("time"),
                         "characteristic_length":
                             parent.evalParm("char_length")}


# =============================================================================
# write / run / log
# =============================================================================


def build_params(parent):
    working_dir = parent.evalParm("working_dir")
    if not os.path.isdir(working_dir):
        raise hou.NodeError("Working directory is not valid!")
    input_dir = os.path.join(working_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)

    data = {}
    orders = build_geometry_and_materials(parent, data, input_dir)
    build_time(parent, data)
    build_contact(parent, data)
    build_conditions(parent, data)
    build_space(parent, data, orders)
    build_solver(parent, data)
    build_output(parent, data)

    params_path = os.path.join(input_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(data, f, indent=4)
    return params_path


def write_params_only(kwargs):
    parent = kwargs["node"]
    try:
        path = build_params(parent)
    except hou.NodeError as e:
        _message(str(e))
        return None
    _status(f"Wrote {path}")
    return path


def _launch_command(parent, params_path):
    polyfem_bin = parent.evalParm("polyfem_bin")
    if not polyfem_bin or not os.path.isfile(polyfem_bin):
        raise hou.NodeError("No valid PolyFEM binary provided!")
    input_dir = os.path.dirname(params_path)
    log_level = parent.evalParm("log_level")
    max_threads = parent.evalParm("max_threads")
    args = [polyfem_bin, "-j", "params.json", "-o", "../output/",
            "--log_level", str(log_level)]
    if max_threads > 0:
        args += ["--max_threads", str(max_threads)]
    return input_dir, args


def write_params(kwargs):
    """Write params.json and launch PolyFEM in a terminal (1.2 behavior)."""
    parent = kwargs["node"]
    try:
        params_path = build_params(parent)
        input_dir, args = _launch_command(parent, params_path)
    except hou.NodeError as e:
        _message(str(e))
        return
    shell = " ".join(args)
    system = platform.system()
    if system == "Darwin":
        command = f"cd '{input_dir}' && {shell}"
        escaped = command.replace('"', '\\"')
        os.system(
            f'osascript -e \'tell application "Terminal" to do script "{escaped}"\'')
    elif system == "Linux":
        subprocess.Popen(
            ["xterm", "-hold", "-e",
             f"bash -c 'cd \"{input_dir}\" && {shell}'"])
    elif system == "Windows":
        subprocess.Popen(
            ["cmd", "/c", "start", "", "cmd", "/k",
             f'cd /d "{input_dir}" && {shell}'])


def run_background(kwargs):
    """Write params.json and run PolyFEM headless with log capture."""
    parent = kwargs["node"]
    try:
        params_path = build_params(parent)
        input_dir, args = _launch_command(parent, params_path)
    except hou.NodeError as e:
        _message(str(e))
        return
    log_path = os.path.join(os.path.dirname(input_dir), "output", "log.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(args, cwd=input_dir, stdout=log_file,
                            stderr=subprocess.STDOUT)
    hou.session.__dict__.setdefault("polyfem_runs", []).append(
        {"pid": proc.pid, "log": log_path})
    _status(f"PolyFEM running in background (pid {proc.pid}); log: {log_path}")


def show_log(kwargs):
    parent = kwargs["node"]
    runs = getattr(hou.session, "polyfem_runs", [])
    log_path = runs[-1]["log"] if runs else os.path.join(
        parent.evalParm("working_dir"), "output", "log.txt")
    if not os.path.isfile(log_path):
        _message(f"No log found at {log_path}")
        return
    with open(log_path) as f:
        tail = f.readlines()[-60:]
    if hou.isUIAvailable():
        hou.ui.displayMessage("".join(tail) or "(log empty)",
                              title=os.path.basename(log_path))
    else:
        print("".join(tail))


# =============================================================================
# read_params + legacy import
# =============================================================================


def _looks_legacy(data):
    contact = data.get("solver", {}).get("contact", {})
    if "adaptive_barrier_stiffness_multiplier" in contact:
        return True
    for material in data.get("materials", []):
        if not isinstance(material, dict):
            continue
        try:
            if int(material.get("id", 0)) < 1000:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _restore_remesh(parent, data):
    """Restore /space/remesh, filling omitted optional values from the spec."""
    space = data.get("space", {})
    if isinstance(space, list):
        space = next((entry for entry in space if isinstance(entry, dict)), {})
    remesh = space.get("remesh") if isinstance(space, dict) else None
    if not isinstance(remesh, dict):
        remesh = {}

    parms = {
        "remeshing_enabled": bool(
            remesh.get("enabled", REMESH_DEFAULTS["enabled"])),
        "remesh_type": 1 if remesh.get(
            "type", REMESH_DEFAULTS["type"]) == "sizing_field" else 0,
    }

    def restore_stage(json_name, enabled_parm, fields):
        stage = remesh.get(json_name, {})
        if not isinstance(stage, dict):
            stage = {}
        defaults = REMESH_DEFAULTS[json_name]
        parms[enabled_parm] = bool(stage.get("enabled", defaults["enabled"]))
        for key, parm in fields:
            parms[parm] = stage.get(key, defaults[key])

    restore_stage("split", "split_enabled", (
        ("acceptance_tolerance", "split_acceptance_tol"),
        ("culling_threshold", "split_culling_threshold"),
        ("max_depth", "split_max_depth"),
        ("min_edge_length", "min_edge_length")))
    restore_stage("collapse", "collapse_enabled", (
        ("acceptance_tolerance", "collapse_acceptance_tol"),
        ("culling_threshold", "collapse_culling_threshold"),
        ("max_depth", "collapse_max_depth"),
        ("rel_max_edge_length", "rel_max_edge_length"),
        ("abs_max_edge_length", "abs_max_edge_length")))
    restore_stage("swap", "swap_enabled2", (
        ("acceptance_tolerance", "swap_acceptance_tol2"),
        ("max_depth", "swap_max_depth2")))
    restore_stage("smooth", "smooth_enabled", (
        ("acceptance_tolerance", "smooth_acceptance_tol"),
        ("max_iters", "smooth_max_iters")))

    local = remesh.get("local_relaxation", {})
    if not isinstance(local, dict):
        local = {}
    for key, parm in (
            ("local_mesh_n_ring", "local_mesh_n_ring"),
            ("local_mesh_rel_area", "local_mesh_rel_area"),
            ("max_nl_iterations", "max_nl_iterations")):
        parms[parm] = local.get(key, REMESH_DEFAULTS["local_relaxation"][key])

    parent.setParms(parms)


def _format_vector(value):
    """JSON value -> HDA vector/scalar parm string (round-trips expressions)."""
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(v) for v in value) + "]"
    return json.dumps(value)


def _decode_boundary_id(bid):
    """Inverse of boundary_id: -> (geo, vol, sideset, bc)."""
    vid, rem = divmod(int(bid), 10000)
    sideset, bc = divmod(rem, 100)
    return vid // 1000, vid % 1000, sideset, bc


def _ensure_sideset(parent, geo, vol, j):
    parm = parent.parm(f"sideset_selection{geo}_{vol}")
    if parm is not None and parm.eval() < j:
        parm.set(int(j))


def _restore_conditions(parent, data):
    """Fill BC / initial-condition / obstacle parms from new-format json."""
    ic = data.get("initial_conditions", {})
    counts, entries = {}, []
    for type_idx, key in enumerate(("solution", "velocity", "acceleration")):
        for entry in ic.get(key, []):
            mid = int(entry.get("id", 0))
            geo, vol = mid // 1000, mid % 1000
            l = counts.get((geo, vol), 0) + 1
            counts[(geo, vol)] = l
            entries.append((geo, vol, l, type_idx, entry.get("value")))
    for (geo, vol), n in counts.items():
        parm = parent.parm(f"initial_conditions{geo}_{vol}")
        if parm is not None:
            parm.set(n)
    for geo, vol, l, type_idx, value in entries:
        if parent.parm(f"conditiontype{geo}_{vol}_{l}") is None:
            continue
        parent.setParms({
            f"conditiontype{geo}_{vol}_{l}": type_idx,
            f"condition_vector_{geo}_{vol}_{l}": _format_vector(value)})

    bc = data.get("boundary_conditions", {})
    if "rhs" in bc and parent.parm("RHS") is not None:
        parent.setParms({"RHS": _format_vector(bc["rhs"])})
    for entry in bc.get("obstacle_displacements", []):
        geo = (int(entry.get("id", 0)) - 100000) // 1000
        if parent.parm(f"obstacle_disp{geo}") is not None:
            parent.setParms(
                {f"obstacle_disp{geo}": _format_vector(entry.get("value"))})

    type_names = ("dirichlet_boundary", "neumann_boundary",
                  "normal_aligned_neumann_boundary", "pressure_boundary",
                  "pressure_cavity")
    sideset_max, bc_max, bc_entries = {}, {}, []
    for btype, key in enumerate(type_names):
        for entry in bc.get(key, []):
            geo, vol, j, k = _decode_boundary_id(entry.get("id", 0))
            if j < 1 or k < 1:
                continue
            sideset_max[(geo, vol)] = max(sideset_max.get((geo, vol), 0), j)
            bc_max[(geo, vol, j)] = max(bc_max.get((geo, vol, j), 0), k)
            bc_entries.append((geo, vol, j, k, btype, entry))
    for (geo, vol), n in sideset_max.items():
        _ensure_sideset(parent, geo, vol, n)
    for (geo, vol, j), n in bc_max.items():
        parm = parent.parm(f"Boundary_Condition__{geo}_{vol}_{j}")
        if parm is not None:
            parm.set(n)
    for geo, vol, j, k, btype, entry in bc_entries:
        if parent.parm(f"boundary_type{geo}_{vol}_{j}_{k}") is None:
            continue
        parms = {f"boundary_type{geo}_{vol}_{j}_{k}": btype}
        if btype in (0, 1):
            parms[f"vector_{geo}_{vol}_{j}_{k}"] = \
                _format_vector(entry.get("value"))
            if btype == 0:
                dims = entry.get("dimension", [True, True, True])
                for ax, flag in zip("xyz", dims):
                    parms[f"{ax}_dimension{geo}_{vol}_{j}_{k}"] = \
                        int(bool(flag))
        else:
            parms[f"value_{geo}_{vol}_{j}_{k}"] = \
                _format_vector(entry.get("value"))
        parent.setParms(parms)


def _native_pattern(entry):
    """Inverse of parse_native_selection: json entry -> UI pattern string."""
    try:
        if "axis" in entry:
            return f"axis:{entry['axis']}:{entry['position']}"
        if "box" in entry:
            lo, hi = entry["box"]
            return f"box:{list(lo)},{list(hi)}"
        if "center" in entry:
            return f"sphere:{list(entry['center'])},{entry['radius']}"
        if "normal" in entry:
            return f"plane:{list(entry['normal'])},{list(entry['point'])}"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _restore_selections(parent, geometry, input_dir):
    """Rebuild sideset selection patterns from the exported native entries
    and per-arity sideset files (faces matched back through the surface
    provenance attributes)."""
    skipped = []
    for i, g in enumerate(geometry, 1):
        if g.get("is_obstacle"):
            continue
        sel = g.get("surface_selection", [])
        if not isinstance(sel, list):
            skipped.append(f"geo {i}: non-list surface_selection")
            continue

        null_node = parent.node(f"null_{i}")
        face_lookup = {}
        point_lookup = {}
        if null_node is not None:
            try:
                surf_geo = null_node.geometry()
                _, face_ids = _surface_arrays(surf_geo)
                for prim, row in enumerate(face_ids):
                    key = tuple(sorted(int(v) for v in row if v >= 0))
                    face_lookup[key] = prim
                msh = np.frombuffer(
                    surf_geo.pointIntAttribValuesAsString("msh_pt_id"),
                    dtype=np.int32)
                point_lookup = {int(m): p for p, m in enumerate(msh)}
            except hou.Error:
                pass

        sideset_prims = {}
        for entry in sel:
            if not isinstance(entry, dict):
                skipped.append(f"geo {i}: integer selection {entry} "
                               "(assign via sidesets instead)")
                continue
            if "file" in entry:
                path = os.path.join(input_dir, entry["file"])
                try:
                    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
                except (OSError, ValueError):
                    skipped.append(f"geo {i}: unreadable {entry['file']}")
                    continue
                unmatched = 0
                for row in rows:
                    _, vol, j, _ = _decode_boundary_id(row[0])
                    prim = face_lookup.get(
                        tuple(sorted(int(v) for v in row[1:])))
                    if prim is None:
                        unmatched += 1
                        continue
                    sideset_prims.setdefault((vol, j), set()).add(prim)
                if unmatched:
                    skipped.append(
                        f"geo {i}: {unmatched} faces of {entry['file']} not "
                        "on the current mesh")
            else:
                pattern = _native_pattern(entry)
                _, vol, j, _ = _decode_boundary_id(entry.get("id", 0))
                if pattern is None or j < 1:
                    skipped.append(f"geo {i}: unrecognized selection entry")
                    continue
                _ensure_sideset(parent, i, vol, j)
                parent.setParms({f"basegroup{i}_{vol}_{j}": pattern,
                                 f"grouptype{i}_{vol}_{j}": 0})
        for (vol, j), prims in sideset_prims.items():
            _ensure_sideset(parent, i, vol, j)
            parent.setParms({
                f"basegroup{i}_{vol}_{j}": list_to_space_str(sorted(prims)),
                f"grouptype{i}_{vol}_{j}": 0})

        # point sidesets
        for entry in g.get("point_selection", []) \
                if isinstance(g.get("point_selection", []), list) else []:
            if not (isinstance(entry, dict) and "file" in entry):
                continue
            path = os.path.join(input_dir, entry["file"])
            try:
                rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
            except (OSError, ValueError):
                skipped.append(f"geo {i}: unreadable {entry['file']}")
                continue
            sideset_points = {}
            for row in rows:
                _, vol, j, _ = _decode_boundary_id(row[0])
                pt = point_lookup.get(int(row[1]) + 1)
                if pt is not None:
                    sideset_points.setdefault((vol, j), set()).add(pt)
            for (vol, j), pts in sideset_points.items():
                _ensure_sideset(parent, i, vol, j)
                parent.setParms({
                    f"basegroup{i}_{vol}_{j}": list_to_space_str(sorted(pts)),
                    f"grouptype{i}_{vol}_{j}": 1})
    return skipped


def _restore_legacy_scene(parent, data, geometry, input_dir, material_slots):
    """Best-effort BC/selection import for 1.x and generic older PolyFEM ids.

    HDA 1.2 encoded geo/volume/sideset/BC as decimal digits. Generic PolyFEM
    scenes commonly use small shared selection ids (10, 20, ...). Both are
    mapped into the current collision-free HDA multiparms here.
    """
    skipped = []
    targets = {}
    next_sideset = {}
    grouped_prims = {}
    grouped_points = {}

    def legacy_target(value):
        text = str(abs(int(value)))
        if len(text) < 4:
            return None
        geo, vol, sideset = (int(text[0]), int(text[1]), int(text[2]))
        bc = int(text[3:] or 1)
        if parent.parm(f"sideset_selection{geo}_{vol}") is None:
            return None
        return geo, vol, sideset, bc

    def target_for(selection_id, geo, vol):
        direct = legacy_target(selection_id)
        if direct is not None:
            target = direct[:3]
        else:
            key = (geo, vol)
            sid_key = (geo, vol, int(selection_id))
            if sid_key not in next_sideset:
                next_sideset[sid_key] = 1 + max(
                    [value for (g, v, _), value in next_sideset.items()
                     if (g, v) == key] or [0])
            target = (geo, vol, next_sideset[sid_key])
        targets.setdefault(int(selection_id), [])
        if target not in targets[int(selection_id)]:
            targets[int(selection_id)].append(target)
        _ensure_sideset(parent, *target)
        return target

    for geo, g in enumerate(geometry, start=1):
        if g.get("is_obstacle") or parent.node(f"null_{geo}") is None:
            continue
        surface_geo = parent.node(f"null_{geo}").geometry()
        entities, face_ids = _surface_arrays(surface_geo)
        face_lookup = {
            tuple(sorted(int(value) for value in row if value >= 0)): index
            for index, row in enumerate(face_ids)}
        msh_ids = np.frombuffer(
            surface_geo.pointIntAttribValuesAsString("msh_pt_id"),
            dtype=np.int32)
        point_lookup = {int(value): index for index, value in enumerate(msh_ids)}
        volume_count = parent.evalParm(f"num_volumes{geo}")

        selections = g.get("surface_selection", [])
        if not isinstance(selections, list):
            selections = [selections]
        for entry in selections:
            if isinstance(entry, dict) and "file" in entry:
                path = _selection_file(entry, input_dir)
                try:
                    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
                except (OSError, ValueError) as exc:
                    skipped.append(
                        f"Geometry {geo}: unreadable surface selection file "
                        f"{entry.get('file')} ({exc}).")
                    continue
                for row in rows:
                    if len(row) < 4:
                        continue
                    sid = int(row[0])
                    prim = face_lookup.get(tuple(sorted(int(v) for v in row[1:])))
                    if prim is None:
                        skipped.append(
                            f"Geometry {geo}: one face from {entry.get('file')} "
                            "is not on the current mesh.")
                        continue
                    direct = legacy_target(sid)
                    vol = direct[1] if direct is not None else int(entities[prim])
                    target = target_for(sid, geo, vol)
                    grouped_prims.setdefault(target, set()).add(prim)
            elif isinstance(entry, dict) and "id" in entry:
                sid = int(entry["id"])
                pattern = _native_pattern(entry)
                if pattern is None:
                    skipped.append(
                        f"Geometry {geo}: selection id {sid} uses an "
                        "unsupported older selector.")
                    continue
                direct = legacy_target(sid)
                volumes = [direct[1]] if direct is not None \
                    else range(1, volume_count + 1)
                for vol in volumes:
                    target = target_for(sid, geo, vol)
                    parent.setParms({
                        f"basegroup{target[0]}_{target[1]}_{target[2]}": pattern,
                        f"grouptype{target[0]}_{target[1]}_{target[2]}": 0})

        point_selections = g.get("point_selection", [])
        if not isinstance(point_selections, list):
            point_selections = [point_selections]
        for entry in point_selections:
            if not (isinstance(entry, dict) and "file" in entry):
                continue
            path = _selection_file(entry, input_dir)
            try:
                rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
            except (OSError, ValueError) as exc:
                skipped.append(
                    f"Geometry {geo}: unreadable point selection file "
                    f"{entry.get('file')} ({exc}).")
                continue
            for row in rows:
                if len(row) < 2:
                    continue
                sid = int(row[0])
                point = point_lookup.get(int(row[1]) + 1)
                if point is None:
                    continue
                direct = legacy_target(sid)
                vol = direct[1] if direct is not None else 1
                target = target_for(sid, geo, vol)
                grouped_points.setdefault(target, set()).add(point)

    for target, prims in grouped_prims.items():
        geo, vol, sideset = target
        parent.setParms({
            f"basegroup{geo}_{vol}_{sideset}":
                list_to_space_str(sorted(prims)),
            f"grouptype{geo}_{vol}_{sideset}": 0})
    for target, points in grouped_points.items():
        geo, vol, sideset = target
        parent.setParms({
            f"basegroup{geo}_{vol}_{sideset}":
                list_to_space_str(sorted(points)),
            f"grouptype{geo}_{vol}_{sideset}": 1})

    # Initial conditions use material ids, not boundary selection ids.
    initial = data.get("initial_conditions", {})
    per_slot = {}
    for type_index, key in enumerate(("solution", "velocity", "acceleration")):
        for entry in initial.get(key, []) if isinstance(initial, dict) else []:
            slot = _material_slot(entry.get("id"), material_slots, True)
            if slot is None:
                skipped.append(
                    f"Initial condition id {entry.get('id')} was not mapped.")
                continue
            per_slot.setdefault(slot, []).append(
                (type_index, entry.get("value")))
    for (geo, vol), entries in per_slot.items():
        parent.setParms({f"initial_conditions{geo}_{vol}": len(entries)})
        for index, (type_index, value) in enumerate(entries, start=1):
            parent.setParms({
                f"conditiontype{geo}_{vol}_{index}": type_index,
                f"condition_vector_{geo}_{vol}_{index}":
                    _format_vector(value)})

    bc = data.get("boundary_conditions", {})
    if not isinstance(bc, dict):
        return skipped
    if "rhs" in bc:
        parent.setParms({"RHS": _format_vector(bc["rhs"])})
    type_names = (
        "dirichlet_boundary", "neumann_boundary",
        "normal_aligned_neumann_boundary", "pressure_boundary",
        "pressure_cavity")
    next_bc = {}
    for type_index, key in enumerate(type_names):
        for entry in bc.get(key, []):
            sid = int(entry.get("id", 0))
            direct = legacy_target(sid)
            selected_targets = [direct[:3]] if direct is not None \
                else targets.get(sid, [])
            if not selected_targets:
                skipped.append(
                    f"Boundary condition selection id {sid} was not found.")
                continue
            for geo, vol, sideset in selected_targets:
                if direct is not None:
                    index = direct[3]
                else:
                    key_tuple = (geo, vol, sideset)
                    index = next_bc.get(key_tuple, 0) + 1
                    next_bc[key_tuple] = index
                count = parent.parm(
                    f"Boundary_Condition__{geo}_{vol}_{sideset}")
                if count is not None and count.eval() < index:
                    count.set(index)
                parms = {
                    f"boundary_type{geo}_{vol}_{sideset}_{index}": type_index}
                if type_index in (0, 1):
                    parms[f"vector_{geo}_{vol}_{sideset}_{index}"] = \
                        _format_vector(entry.get("value"))
                    if type_index == 0:
                        for axis, flag in zip(
                                "xyz", entry.get(
                                    "dimension", [True, True, True])):
                            parms[
                                f"{axis}_dimension{geo}_{vol}_{sideset}_{index}"
                            ] = int(bool(flag))
                else:
                    parms[f"value_{geo}_{vol}_{sideset}_{index}"] = \
                        _format_vector(entry.get("value"))
                parent.setParms(parms)
    for entry in bc.get("obstacle_displacements", []):
        value = int(entry.get("id", 0))
        geo = (value - 100000) // 1000
        if parent.parm(f"obstacle_disp{geo}") is not None:
            parent.setParms({
                f"obstacle_disp{geo}": _format_vector(entry.get("value"))})
    return skipped


def _import_isotropic(parent, geo, vol, model):
    parms = {}
    for key, base in (("E", "E"), ("nu", "nu"), ("c1", "c1"), ("c2", "c2"),
                      ("c3", "c3"), ("d1", "d1"), ("k", "bulk"),
                      ("phi", "phi"), ("psi", "psi")):
        if key in model and parent.parm(f"{base}{geo}_{vol}") is not None:
            parms[f"{base}{geo}_{vol}"] = model[key]
    if "phi" in model or "psi" in model:
        parms[f"damping{geo}_{vol}"] = 1
    for keys, base in (
            (("elasticity_tensor",), "elast_tensor"),
            (("c", "C"), "ogdenC"), (("m",), "ogdenM"),
            (("alphas",), "ogden_alphas"), (("mus",), "ogden_mus"),
            (("Ds", "ds"), "ogden_ds")):
        key = next((candidate for candidate in keys if candidate in model), None)
        if key is not None and parent.parm(f"{base}{geo}_{vol}") is not None:
            parms[f"{base}{geo}_{vol}"] = json.dumps(model[key])
    if "fiber_direction" in model and isinstance(model["fiber_direction"], list) \
            and parent.parm(f"material_coord{geo}_{vol}") is not None:
        parms[f"material_coord{geo}_{vol}"] = json.dumps(model["fiber_direction"])
    if parms:
        parent.setParms(parms)


def _import_scalar(parent, geo, vol, fam, name, value, params_dir,
                   warnings=None):
    """Set a scalar parameter's source and value from imported JSON."""
    prefix = _parm_name(geo, vol, fam, f"{name}_source")
    if parent.parm(prefix) is None:
        return
    if isinstance(value, (int, float)):
        parent.setParms({prefix: "constant",
                         _parm_name(geo, vol, fam, name): float(value)})
        return
    text = str(value)
    source_path = text if os.path.isabs(text) else os.path.join(
        params_dir, text)
    if os.path.isfile(source_path):
        try:
            source_path = _stage_material_file(
                parent, source_path, f"{name} per-element file")
        except hou.NodeError as exc:
            if warnings is not None:
                warnings.append(str(exc))
            return
        parent.setParms({
            prefix: "file",
            _parm_name(geo, vol, fam, f"{name}_file"):
                source_path})
    else:
        parent.setParms({prefix: "expression",
                         _parm_name(geo, vol, fam, f"{name}_expr"): text})


def _import_fiber_direction(parent, geo, vol, fam, value, params_dir,
                            warnings=None):
    source = _parm_name(geo, vol, fam, "fib_source")
    if parent.parm(source) is None:
        return
    if isinstance(value, dict) and value.get("type") == "per_element_file":
        relative = value.get("path", FIBER_FILE)
        source_path = relative if os.path.isabs(relative) else os.path.join(
            params_dir, relative)
        try:
            source_path = _stage_material_file(
                parent, source_path, "fiber file")
        except hou.NodeError as exc:
            if warnings is not None:
                warnings.append(str(exc))
            return
        parent.setParms({
            source: "file",
            _parm_name(geo, vol, fam, "fib_file"): source_path,
            _parm_name(geo, vol, fam, "fib_file_field"):
                value.get("field", "FIB_DIR1")})
        return
    if isinstance(value, list) and len(value) == 3:
        if all(isinstance(entry, (int, float)) for entry in value):
            parent.setParms({source: "constant"})
            parent.parmTuple(_parm_name(geo, vol, fam, "fib_dir")).set(
                [float(entry) for entry in value])
            return
        parent.setParms({source: "expression",
                         _parm_name(geo, vol, fam, "fib_expr"):
                             json.dumps(value)})


def _import_fiber_model(parent, geo, vol, fam, model, params_dir,
                        warnings=None):
    token = model.get("type")
    if fam is not None:
        parent.setParms({f"fam_model{geo}_{vol}_{fam}": token})
    if "fiber_direction" in model:
        _import_fiber_direction(parent, geo, vol, fam,
                                model["fiber_direction"], params_dir, warnings)
    parms = {}
    for key in ("k1", "k2", "k_chi", "Tmax"):
        name = _parm_name(geo, vol, fam, key)
        if key in model and parent.parm(name) is not None:
            parms[name] = model[key]
    if "activation" in model:
        name = _parm_name(geo, vol, fam, "activation")
        if parent.parm(name) is not None:
            parms[name] = str(model["activation"])
    if parms:
        parent.setParms(parms)
    if "kappa" in model:
        _import_scalar(parent, geo, vol, fam, "kappa", model["kappa"],
                       params_dir, warnings)


def _import_composite(parent, geo, vol, models, params_dir, warnings=None):
    """Split an imported "models" list back into matrix + fiber families.

    Anything the interface cannot express (an unknown type, a nested sum) is
    kept verbatim in the extra-models field so the round trip stays lossless.
    """
    matrix = "None"
    families, extra, penalty = [], [], None
    for model in models:
        token = model.get("type")
        if token in FIBER_MODELS:
            families.append(model)
        elif token == "VolumePenalty" and penalty is None:
            penalty = model
        elif token in MATERIAL_TOKENS and token != "MaterialSum" \
                and matrix == "None" and not families:
            matrix = token
            _import_isotropic(parent, geo, vol, model)
        else:
            extra.append(model)

    parent.setParms({
        f"matrix_model{geo}_{vol}": matrix,
        f"num_fiber_families{geo}_{vol}": len(families),
        f"volume_penalty{geo}_{vol}": 1 if penalty else 0,
        f"extra_models_json{geo}_{vol}": json.dumps(extra) if extra else ""})
    if penalty and parent.parm(f"vp_k{geo}_{vol}") is not None:
        parent.setParms({f"vp_k{geo}_{vol}": penalty.get("k", 1e8)})
    for index, model in enumerate(families, start=1):
        # A symmetric pair round-trips as two independent families: the
        # expanded stack is what PolyFEM sees, so this is value-exact.
        parent.setParms({f"fam_mirror{geo}_{vol}_{index}": 0})
        _import_fiber_model(
            parent, geo, vol, index, model, params_dir, warnings)
    if extra:
        message = (
            f"Geo {geo} subdomain {vol}: {len(extra)} model(s) cannot be "
            "shown by the interface; retained in Extra Models (JSON).")
        if warnings is not None:
            warnings.append(message)
        else:
            _message("Import: " + message)


def _selection_file(selection, params_dir):
    """Resolve a PolyFEM file selection to an absolute path, if it is one."""
    value = selection
    if isinstance(selection, dict):
        value = selection.get("file")
    if not isinstance(value, str) or not value:
        return None
    return value if os.path.isabs(value) else os.path.join(params_dir, value)


def _decoded_volume(mid, geo, legacy):
    """Recognize this HDA's current and 1.x body-id encodings."""
    try:
        value = int(mid)
    except (TypeError, ValueError):
        return None
    if value // 1000 == int(geo) and value % 1000 > 0:
        return value % 1000
    text = str(abs(value))
    if legacy and len(text) >= 2 and int(text[0]) == int(geo):
        result = int(text[1:] or 1)
        return result if result > 0 else None
    return None


def _restore_volume_assignments(parent, geo, geometry_data, params_dir,
                                legacy, warnings):
    """Recreate edited Entity values from geometry.volume_selection.

    The MSH file may still contain its original physical groups, while the
    previous Houdini scene re-tagged arbitrary elements into new subdomains.
    The exported volume-selection file is therefore the source of truth.
    Returns {original material/body id: (geo, HDA volume index)}.
    """
    selection = geometry_data.get("volume_selection")
    branch = parent.node(f"branch_{geo}")
    if branch is None:
        warnings.append(
            f"Geometry {geo}: node tree was not created; volume assignments "
            "could not be restored.")
        return {}
    geo_data = branch.geometry()
    mask = _volume_mask(geo_data)
    prim_numbers = np.flatnonzero(mask) if mask is not None else np.arange(
        geo_data.intrinsicValue("primitivecount"))
    if isinstance(selection, (int, float)):
        ids = np.full(len(prim_numbers), int(selection), dtype=np.int64)
    else:
        path = _selection_file(selection, params_dir)
        if path is None:
            warnings.append(
                f"Geometry {geo}: no file/scalar volume_selection; retained "
                "the subdomains stored in the mesh.")
            return {}
        try:
            ids = np.loadtxt(path, dtype=np.int64, ndmin=1).reshape(-1)
        except (OSError, ValueError) as exc:
            warnings.append(
                f"Geometry {geo}: could not read volume_selection "
                f"'{os.path.basename(path)}' ({exc}); retained mesh "
                "subdomains.")
            return {}
    if len(ids) != len(prim_numbers):
        warnings.append(
            f"Geometry {geo}: volume_selection has {len(ids)} rows but the "
            f"mesh has {len(prim_numbers)} volume elements; retained mesh "
            "subdomains.")
        return {}

    ordered_ids = list(dict.fromkeys(int(value) for value in ids))
    decoded = [_decoded_volume(value, geo, legacy) for value in ordered_ids]
    if all(value is not None for value in decoded) \
            and len(set(decoded)) == len(decoded):
        body_to_volume = dict(zip(ordered_ids, decoded))
    else:
        # Arbitrary valid PolyFEM material ids cannot be represented verbatim by
        # the HDA's geo*1000+volume convention. Compact them but retain their
        # exact material association through the returned lookup.
        body_to_volume = {
            body: index for index, body in enumerate(ordered_ids, start=1)}
        warnings.append(
            f"Geometry {geo}: arbitrary material ids {ordered_ids} were "
            "compacted to consecutive Houdini subdomains; assignments and "
            "material associations were preserved.")

    num_volumes = max(body_to_volume.values())
    parent.setParms({f"num_volumes{geo}": num_volumes})
    for volume in range(1, num_volumes + 1):
        _set_color_tuple(parent, f"color_{geo}_{volume}",
                         default_color(geo, volume))

    # Build the same lightweight VEX retag chain used by interactive edits,
    # once, then reconstruct the display/selection network from its result.
    upstream = parent.node(f"elements_{geo}")
    for body, volume in body_to_volume.items():
        selected = prim_numbers[ids == body]
        vex = parent.createNode("attribwrangle", f"vex_{geo}_{volume}")
        vex.setParms({
            "group": list_to_space_str(selected), "grouptype": 4, "class": 1,
            "snippet": f"i@Entity = {int(volume)};"})
        vex.setInput(0, upstream)
        upstream = vex
    branch.setInput(0, upstream)
    clear_geo_tree(parent, geo)
    create_geo_tree({"node": parent, "script_multiparm_index": str(geo)})
    parent.setParms({f"subdomain_number_{geo}": 1})
    fetch_elements({"node": parent, "script_multiparm_index": str(geo)})
    return {
        int(body): (int(geo), int(volume))
        for body, volume in body_to_volume.items()}


def _material_slot(mid, material_slots, legacy):
    try:
        value = int(mid)
    except (TypeError, ValueError):
        return None
    if value in material_slots:
        return material_slots[value]
    if legacy:
        text = str(abs(value))
        if len(text) >= 2:
            return int(text[0]), int(text[1:] or 1)
    geo, vol = divmod(value, 1000)
    return (geo, vol) if geo > 0 and vol > 0 else None


def _restore_time(parent, data):
    time = data.get("time")
    if not isinstance(time, dict):
        return
    parms = {
        "t0": time.get("t0", parent.evalParm("t0")),
        "quasistatic": bool(
            time.get("quasistatic", parent.evalParm("quasistatic")))}
    integrator = time.get("integrator")
    integrators = {
        name: index for index, name in enumerate((
            "ImplicitEuler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5",
            "BDF6", "ImplicitNewmark"))}
    if isinstance(integrator, dict):
        token = integrator.get("type")
        if token in integrators:
            parms["integrator"] = integrators[token]
        if "beta" in integrator:
            parms["newmark_beta"] = integrator["beta"]
        if "gamma" in integrator:
            parms["newmark_gamma"] = integrator["gamma"]
    elif integrator in integrators:
        parms["integrator"] = integrators[integrator]
    parms["end_time_bool"] = int("tend" in time)
    parms["time_inc_bool"] = int("dt" in time)
    parms["num_timesteps_bool"] = int("time_steps" in time)
    for key, parm in (
            ("tend", "tend"), ("dt", "dt"),
            ("time_steps", "num_timesteps")):
        if key in time:
            parms[parm] = time[key]
    parent.setParms(parms)


def _restore_contact(parent, data):
    contact = data.get("contact")
    if not isinstance(contact, dict):
        return
    parms = {"enable": bool(contact.get("enabled", False))}
    for key, parm in (
            ("dhat", "dhat"), ("epsv", "epsv"),
            ("friction_coefficient", "cof"),
            ("use_convergent_formulation", "area_weighted"),
            ("use_gcp_formulation", "gcp_enable"),
            ("alpha_n", "alpha_n"), ("alpha_t", "alpha_t"),
            ("min_distance_ratio", "min_dist_ratio"),
            ("use_adaptive_dhat", "adapt_dhat")):
        if key in contact:
            parms[parm] = contact[key]
    adhesion = contact.get("adhesion")
    if isinstance(adhesion, dict):
        for key, parm in (
                ("adhesion_enabled", "adhesion_enable"),
                ("dhat_p", "dhat_p"), ("dhat_a", "dhat_a"),
                ("adhesion_strength", "adhesion_strength"),
                ("tangential_adhesion_coefficient", "tangent_coeff"),
                ("epsa", "adhesion_epsa")):
            if key in adhesion:
                parms[parm] = adhesion[key]
    parent.setParms(parms)


def _restore_space(parent, data, material_slots, legacy, warnings):
    space = data.get("space")
    if isinstance(space, list):
        entries = [entry for entry in space if isinstance(entry, dict)]
        space = entries[0] if entries else {}
    if not isinstance(space, dict):
        return
    orders = space.get("discr_order", [])
    if isinstance(orders, (int, float)):
        for geo in range(1, parent.evalParm("num_geos") + 1):
            if parent.evalParm(f"is_obstacle{geo}"):
                continue
            for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
                parent.setParms({f"mainOrder{geo}_{vol}": int(orders) - 1})
    elif isinstance(orders, list):
        for entry in orders:
            if not isinstance(entry, dict) or "order" not in entry:
                continue
            slot = _material_slot(entry.get("id"), material_slots, legacy)
            if slot is None:
                warnings.append(
                    f"Discretization order id {entry.get('id')} could not be "
                    "mapped to a geometry/subdomain.")
                continue
            geo, vol = slot
            parm = parent.parm(f"mainOrder{geo}_{vol}")
            if parm is not None:
                parm.set(max(0, int(entry["order"]) - 1))
    _restore_remesh(parent, data)


def _restore_solver(parent, data, legacy, warnings):
    solver = data.get("solver")
    if not isinstance(solver, dict):
        return
    parms = {}
    if "max_threads" in solver:
        parms["max_threads"] = solver["max_threads"]
    nonlinear = solver.get("nonlinear", {})
    if isinstance(nonlinear, dict):
        for keys, parm in (
                (("max_iterations",), "max_iterations"),
                (("x_delta_tol", "x_delta"), "x_delta"),
                (("grad_norm_tol", "grad_norm"), "grad_norm"),
                (("first_grad_norm_tol",), "first_grad_norm_tol"),
                (("rel_grad_norm_tol",), "rel_grad_norm_tol"),
                (("rel_x_delta_tol",), "rel_x_delta_tol"),
                (("newton_decrement_tol",), "newton_decrement_tol"),
                (("allow_non_grad_convergence",),
                 "allow_non_grad_convergence"),
                (("iterations_per_strategy",), "iterations_per_strategy"),
                (("allow_out_of_iterations",), "allow_out_of_iterations")):
            key = next(
                (candidate for candidate in keys if candidate in nonlinear),
                None)
            if key is not None:
                parms[parm] = nonlinear[key]
        norm_types = {"Euclidean": 0, "L2": 1, "Linf": 2}
        if nonlinear.get("norm_type") in norm_types:
            parms["norm_type"] = norm_types[nonlinear["norm_type"]]
        nonlinear_type = nonlinear.get("solver", "Newton")
        nonlinear_types = {
            name: index for index, name in enumerate((
                "Newton", "DenseNewton", "GradientDescent", "ADAM",
                "StochasticADAM", "StochasticGradientDescent", "L-BFGS",
                "BFGS", "L-BFGS-B", "MMA"))}
        if isinstance(nonlinear_type, list):
            # polysolve also accepts an explicit strategy list; the UI holds
            # the single method, so take the first entry's type.
            first = nonlinear_type[0] if nonlinear_type else {}
            nonlinear_type = first.get("type", "Newton") \
                if isinstance(first, dict) else "Newton"
            warnings.append(
                "solver.nonlinear.solver was a strategy list; only its first "
                "method was restored.")
        if nonlinear_type in nonlinear_types \
                and parent.parm("solver_nl") is not None:
            parms["solver_nl"] = nonlinear_types[nonlinear_type]
        elif nonlinear_type not in nonlinear_types:
            warnings.append(
                f"Nonlinear solver '{nonlinear_type}' is not offered by the "
                "HDA; Newton was kept.")
        # Per-method blocks: read whichever the file carries into the shared
        # UI parameters (Newton/DenseNewton, L-BFGS variants, ADAM variants).
        for block in ("Newton", "DenseNewton"):
            newton = nonlinear.get(block, {})
            if isinstance(newton, dict):
                for key, parm in (
                        ("residual_tolerance", "residual_tolerance"),
                        ("reg_weight_min", "reg_weight_min"),
                        ("reg_weight_max", "reg_weight_max"),
                        ("reg_weight_inc", "reg_weight_inc"),
                        ("force_psd_projection", "force_psd_projection"),
                        ("use_psd_projection", "use_psd_projection"),
                        ("use_psd_projection_in_regularized",
                         "use_psd_projection_reg")):
                    if key in newton:
                        parms[parm] = newton[key]
        for block in ("L-BFGS", "L-BFGS-B"):
            lbfgs = nonlinear.get(block, {})
            if isinstance(lbfgs, dict) and "history_size" in lbfgs:
                parms["history_size"] = lbfgs["history_size"]
        for block in ("ADAM", "StochasticADAM"):
            adam = nonlinear.get(block, {})
            if isinstance(adam, dict):
                for key, parm in (("alpha", "adam_alpha"),
                                  ("beta_1", "adam_beta1"),
                                  ("beta_2", "adam_beta2"),
                                  ("epsilon", "adam_epsilon"),
                                  ("erase_component_probability",
                                   "erase_component_probability")):
                    if key in adam:
                        parms[parm] = adam[key]
        sgd = nonlinear.get("StochasticGradientDescent", {})
        if isinstance(sgd, dict) and "erase_component_probability" in sgd:
            parms["erase_component_probability"] = \
                sgd["erase_component_probability"]
        advanced = nonlinear.get("advanced", {})
        if isinstance(advanced, dict):
            for keys, parm in (
                    (("f_delta_tol", "f_delta"), "f_delta"),
                    (("f_delta_step_tol",), "f_delta_step_tol"),
                    (("derivative_along_delta_x_tol",),
                     "derivative_along_delta_x_tol"),
                    (("gradient_fd_eps",), "gradient_fd_eps")):
                key = next(
                    (candidate for candidate in keys if candidate in advanced),
                    None)
                if key is not None:
                    parms[parm] = advanced[key]
            fd_modes = {name: index for index, name in enumerate((
                "None", "DirectionalDerivative", "FullFiniteDiff"))}
            if advanced.get("apply_gradient_fd") in fd_modes:
                parms["apply_gradient_fd"] = \
                    fd_modes[advanced["apply_gradient_fd"]]
        line_search = nonlinear.get("line_search", {})
        if isinstance(line_search, dict):
            methods = {
                name: index for index, name in enumerate((
                    "Armijo", "RobustArmijo", "Backtracking", "None"))}
            method = line_search.get("method")
            if method == "none":
                method = "None"
            aliases = {"ArmijoAlt": "Armijo", "MoreThuente": "Backtracking",
                       "ResidualBacktracking": "Backtracking"}
            if method in aliases:
                warnings.append(
                    f"Line search '{method}' is not available in this "
                    f"PolyFEM; '{aliases[method]}' was restored instead.")
                method = aliases[method]
            if method in methods:
                parms["method"] = methods[method]
            for key, parm in (
                    ("use_grad_norm_tol", "use_grad_norm_tol"),
                    ("min_step_size", "min_step_size"),
                    ("max_step_size_iter", "max_step_size_iter"),
                    ("min_step_size_final", "min_step_size_final"),
                    ("max_step_size_iter_final", "max_step_size_iter_final"),
                    ("default_init_step_size", "default_init_step_size"),
                    ("step_ratio", "step_ratio")):
                if key in line_search:
                    parms[parm] = line_search[key]
            armijo = line_search.get("Armijo", {})
            robust = line_search.get("RobustArmijo", {})
            if isinstance(armijo, dict) and "c" in armijo:
                parms["armijo_c"] = armijo["c"]
            if isinstance(armijo, dict) and "roundoff_tolerance" in armijo:
                parms["armijo_roundoff_tolerance"] = armijo["roundoff_tolerance"]
            if isinstance(robust, dict) \
                    and "delta_relative_tolerance" in robust:
                parms["delta_relative_tolerance"] = \
                    robust["delta_relative_tolerance"]

    contact = solver.get("contact", {})
    if isinstance(contact, dict):
        for key, parm in (
                ("friction_iterations", "friction_iterations"),
                ("tangential_adhesion_iterations", "tangent_iter"),
                ("friction_convergence_tol", "friction_convergence_tol")):
            if key in contact:
                parms[parm] = contact[key]
        ccd = contact.get("CCD", {})
        if isinstance(ccd, dict):
            broad = {
                name: index for index, name in enumerate((
                    "hash_grid", "brute_force", "spatial_hash", "BVH",
                    "sweep_and_prune", "sweep_and_tiniest_queue"))}
            if ccd.get("broad_phase") in broad:
                parms["broad_phase"] = broad[ccd["broad_phase"]]
            if "tolerance" in ccd:
                parms["CCD_tolerance"] = ccd["tolerance"]
            if "max_iterations" in ccd:
                parms["ccd_max_iterations"] = ccd["max_iterations"]
        stiffness = contact.get("barrier_stiffness", "semi_implicit")
        if stiffness == "semi_implicit":
            parms["barrier_mode"] = 0
            semi = contact.get("semi_implicit", {})
            if isinstance(semi, dict):
                if semi.get("constraint_floor", 0) != 0:
                    warnings.append(
                        "constraint_floor has been retired and is ignored; "
                        "contact barriers remain active below the former floor.")
                for key, parm in (
                        ("trim_lower", "si_trim_lower"),
                        ("trim_upper", "si_trim_upper"),
                        ("trim_factor", "si_trim_factor"),
                        ("kappa_spread", "si_kappa_spread"),
                        ("kappa_min", "si_kappa_min"),
                        ("refresh_interval", "si_refresh_interval"),
                        ("conditioning_cap", "si_conditioning_cap"),
                        ("controller_interval", "si_controller_interval"),
                        ("trial_displacement_cap",
                         "si_trial_displacement_cap"),
                        ("force_continuation", "si_force_continuation"),
                        ("continuation_max_ratio",
                         "si_continuation_max_ratio")):
                    if key in semi:
                        parms[parm] = semi[key]
                identities = {"parent": 0, "stencil": 1}
                if semi.get("coefficient_identity") in identities:
                    parms["si_coefficient_identity"] = \
                        identities[semi["coefficient_identity"]]
                if semi.get("gap_floor", 0) != 0:
                    warnings.append(
                        "gap_floor is experimental and not exposed by the "
                        "HDA; it was dropped on import.")
                restart = semi.get("restart", {})
                if isinstance(restart, dict):
                    for key, parm in (
                            ("enabled", "si_restart_enabled"),
                            ("alpha_threshold", "si_alpha_threshold"),
                            ("patience", "si_patience"),
                            ("min_iterations", "si_min_iterations"),
                            ("soft_iteration_limit", "si_soft_iteration_limit"),
                            ("max_restarts", "si_max_restarts"),
                            ("stall_trim_factor", "si_stall_trim_factor")):
                        if key in restart:
                            parms[parm] = restart[key]
        elif stiffness == "adaptive":
            parms["barrier_mode"] = 1
            if "initial_barrier_stiffness" in contact:
                parms["initial_barrier_stiffness"] = \
                    contact["initial_barrier_stiffness"]
            if "adaptive_barrier_stiffness_multiplier" in contact:
                warnings.append(
                    "solver.contact.adaptive_barrier_stiffness_multiplier has "
                    "no current equivalent; the adaptive stiffness default was "
                    "retained.")
        elif isinstance(stiffness, (int, float)):
            parms["barrier_mode"] = 2
            parms["barrier_stiffness"] = stiffness

    al = solver.get("augmented_lagrangian", {})
    if isinstance(al, dict):
        if al.get("initial_weight") == "hessian_scaled":
            parms["al_hessian_scaled"] = 1
            if "initial_weight_multiplier" in al:
                parms["al_weight_multiplier"] = \
                    al["initial_weight_multiplier"]
        elif "initial_weight" in al:
            parms["al_hessian_scaled"] = 0
            parms["al_initial_weight"] = al["initial_weight"]
        for key, parm in (
                ("scaling", "al_scaling"), ("max_weight", "al_max_weight"),
                ("eta", "al_eta")):
            if key in al:
                parms[parm] = al[key]
        lumpings = {"row_sum": 0, "hrz": 1}
        if al.get("lumping") in lumpings:
            parms["al_lumping"] = lumpings[al["lumping"]]
        al_nonlinear = al.get("nonlinear", {})
        if isinstance(al_nonlinear, dict) \
                and "max_iterations" in al_nonlinear:
            parms["al_max_iterations"] = al_nonlinear["max_iterations"]

    advanced = solver.get("advanced", {})
    if isinstance(advanced, dict):
        for key, parm in (
                ("cache_size", "cache_size"),
                ("lump_mass_matrix", "lump_mass_matrix"),
                ("lagged_regularization_weight",
                 "lagged_regularization_weight"),
                ("lagged_regularization_iterations",
                 "lagged_regularization_iterations"),
                ("jacobian_threshold", "jacobian_threshold")):
            if key in advanced:
                parms[parm] = advanced[key]
        inversion = {"Discrete": 0, "Conservative": 1}
        if advanced.get("check_inversion") in inversion:
            parms["inversion_method"] = inversion[advanced["check_inversion"]]

    linear = solver.get("linear", {})
    if isinstance(linear, dict):
        _restore_linear_solver(parent, linear, parms, warnings)
    parent.setParms(parms)

    rayleigh = solver.get("rayleigh_damping", [])
    if isinstance(rayleigh, dict):
        rayleigh = [rayleigh]
    if isinstance(rayleigh, list):
        parent.setParms({"num_rayleigh": len(rayleigh)})
        forms = {"elasticity": 0, "contact": 1, "friction": 2}
        for index, entry in enumerate(rayleigh, start=1):
            if not isinstance(entry, dict):
                continue
            values = {}
            if entry.get("form") in forms:
                values[f"rayleigh_form{index}"] = forms[entry["form"]]
            if "stiffness_ratio" in entry:
                values[f"stiffness_ratio{index}"] = entry["stiffness_ratio"]
            if "lagging_iterations" in entry:
                values[f"rayleigh_lagging{index}"] = \
                    entry["lagging_iterations"]
            if values:
                parent.setParms(values)


def _restore_linear_solver(parent, linear, parms, warnings):
    """Map /solver/linear back onto the Linear folder (menu tokens by name)."""
    solver_parm = parent.parm("solver")
    tokens = list(solver_parm.menuItems())
    name = linear.get("solver", "")
    if isinstance(name, list):
        name = name[0] if name else ""
        warnings.append(
            "solver.linear.solver listed several solvers; only the first "
            "was restored.")
    aliases = {"Eigen:SparseLU": "Eigen::SparseLU",
               "Eigen:PardisoLU": "Eigen::PardisoLU"}
    name = aliases.get(name, name)
    if not name:
        parms["solver"] = tokens.index("auto")
    elif name in tokens:
        parms["solver"] = tokens.index(name)
    else:
        warnings.append(
            f"Linear solver '{name}' is not offered by the HDA; the "
            "automatic choice was kept.")
        return
    precond = linear.get("precond")
    precond_tokens = list(parent.parm("precond").menuItems())
    if precond in precond_tokens:
        parms["precond"] = precond_tokens.index(precond)
    block = linear.get(name, {}) if isinstance(linear.get(name), dict) else {}
    if name.startswith("Eigen::") and block:
        if "max_iter" in block:
            parms["max_iter"] = block["max_iter"]
        if "tolerance" in block:
            parms["tolerance"] = block["tolerance"]
    pardiso = linear.get("Pardiso", {})
    if isinstance(pardiso, dict) and "mtype" in pardiso:
        mtypes = list(parent.parm("mtype").menuItems())
        if str(pardiso["mtype"]) in mtypes:
            parms["mtype"] = mtypes.index(str(pardiso["mtype"]))
    hypre = linear.get("Hypre", {})
    if isinstance(hypre, dict):
        # `dimension` is not a control: the exporter always writes 3.
        for key, parm in (("max_iter", "max_iter_hypre"),
                          ("pre_max_iter", "pre_max_iter_hypre"),
                          ("tolerance", "tolerance_hypre_AMGCL"),
                          ("theta", "theta_hypre"),
                          ("nodal_coarsening", "nodal_coarsening_hypre")):
            if key in hypre:
                parms[parm] = hypre[key]

    def restore_token(parm_name, value, what):
        if value is None:
            return
        tokens = list(parent.parm(parm_name).menuItems())
        if value in tokens:
            parms[parm_name] = tokens.index(value)
        else:
            warnings.append(
                f"AMGCL {what} '{value}' is not offered by the HDA; the "
                "default was kept.")

    amgcl = linear.get("AMGCL", {})
    if isinstance(amgcl, dict):
        amg_solver = amgcl.get("solver", {})
        if isinstance(amg_solver, dict):
            if "maxiter" in amg_solver:
                parms["max_iter"] = amg_solver["maxiter"]
            if "tol" in amg_solver:
                parms["tolerance_hypre_AMGCL"] = amg_solver["tol"]
            restore_token("solver_type", amg_solver.get("type"), "solver type")
        precond_block = amgcl.get("precond", {})
        if isinstance(precond_block, dict):
            restore_token("class", precond_block.get("class"),
                          "preconditioner class")
            for key, parm in (("max_levels", "max_levels"),
                              ("direct_coarse", "direct_coarse"),
                              ("ncycle", "ncycle")):
                if key in precond_block:
                    parms[parm] = precond_block[key]
            relax = precond_block.get("relax", {})
            if isinstance(relax, dict):
                restore_token("relax_type", relax.get("type"), "relax type")
                for key, parm in (("degree", "degree"),
                                  ("power_iters", "power_iters"),
                                  ("higher", "higher"), ("lower", "lower"),
                                  ("scale", "scale_amgcl")):
                    if key in relax:
                        parms[parm] = relax[key]
            coarsening = precond_block.get("coarsening", {})
            if isinstance(coarsening, dict):
                restore_token("coarsening_type", coarsening.get("type"),
                              "coarsening type")
                for key, parm in (
                        ("estimate_spectral_radius",
                         "estimate_spectral_radius"),
                        ("relax", "coarse_relax")):
                    if key in coarsening:
                        parms[parm] = coarsening[key]
                aggr = coarsening.get("aggr", {})
                if isinstance(aggr, dict) and "eps_strong" in aggr:
                    parms["eps_strong"] = aggr["eps_strong"]


def _restore_output(parent, data):
    output = data.get("output")
    if not isinstance(output, dict):
        return
    parms = {
        "output_json": int("json" in output),
        "restart_json": int("restart_json" in output)}
    paraview = output.get("paraview", {})
    if isinstance(paraview, dict):
        filename = paraview.get("file_name")
        if isinstance(filename, str):
            parms["paraview_file_name"] = os.path.basename(filename)
        for key, parm in (
                ("vismesh_rel_area", "vismesh_rel_area"),
                ("skip_frame", "skip_frame"),
                ("high_order_mesh", "high_order_mesh"),
                ("wireframe", "paraview_wireframe"),
                ("points", "paraview_points")):
            if key in paraview:
                parms[parm] = paraview[key]
        options = paraview.get("options", {})
        if isinstance(options, dict):
            for key, parm in (
                    ("use_hdf5", "use_hdf5"),
                    ("material", "materials_fields"),
                    ("body_ids", "body_ids_fields"),
                    ("contact_forces", "contact_forces_fields"),
                    ("friction_forces", "friction_forces_fields"),
                    ("normal_adhesion_forces",
                     "normal_adhesion_forces_fields"),
                    ("tangential_adhesion_forces",
                     "tangential_adhesion_forces_fields"),
                    ("velocity", "velocity_fields"),
                    ("acceleration", "acceleration_fields"),
                    ("scalar_values", "scalar_values"),
                    ("tensor_values", "tensor_values"),
                    ("discretization_order", "discr_fields"),
                    ("nodes", "nodes_field"), ("forces", "forces_fields"),
                    ("force_high_order", "forces_HO_output"),
                    ("jacobian_validity", "validity_fields")):
                if key in options:
                    parms[parm] = options[key]
        parms["minimal_fields"] = int(
            isinstance(paraview.get("fields"), list)
            and not bool(options.get("material", False)))
        fields = paraview.get("fields")
        if isinstance(fields, list):
            # The surface field toggles only leave a trace in the whitelist.
            parms["normals_fields"] = int("normals" in fields)
            parms["displaced_normals_fields"] = int(
                "displaced_normals" in fields and "normals" not in fields)
            parms["sidesets_fields"] = int("sidesets" in fields)
            parms["solution_grad_fields"] = int("solution_gradient" in fields)
            parms["adaptive_dhat"] = int("adaptive_dhat" in fields)

    output_data = output.get("data", {})
    if isinstance(output_data, dict):
        for key, parm in (
                ("solution", "solution_file"),
                ("stress_mat", "stress_mat_file"), ("state", "state_file"),
                ("rest_mesh", "rest_mesh_file"), ("mises", "mises_file"),
                ("nodes", "nodes")):
            parms[parm] = int(key in output_data)
    advanced = output.get("advanced", {})
    if isinstance(advanced, dict):
        for key, parm in (
                ("timestep_prefix", "timestep_prefix"),
                ("sol_on_grid", "sol_on_grid"),
                ("compute_error", "compute_error"),
                ("sol_at_node", "sol_at_node"),
                ("vis_boundary_only", "vis_boundary_only"),
                ("curved_mesh_size", "curved_mesh_size"),
                ("save_solve_sequence_debug", "save_solve_sequence_debug"),
                ("save_ccd_debug_meshes", "save_ccd_debug_meshes"),
                ("save_time_sequence", "save_time_sequence"),
                ("save_nl_solve_sequence", "save_nl_solve_sequence"),
                ("spectrum", "spectrum")):
            if key in advanced:
                parms[parm] = advanced[key]
    log = output.get("log", {})
    if isinstance(log, dict):
        if "level" in log:
            parms["log_level"] = log["level"]
        if "quiet" in log:
            parms["log_quiet"] = log["quiet"]
        parms["write_log"] = int("path" in log)
    parent.setParms(parms)


def _restore_units(parent, data):
    units = data.get("units")
    if not isinstance(units, dict):
        parent.setParms({"units": 0})
        return
    parms = {"units": 1}
    for key, parm in (
            ("length", "length"), ("mass", "mass"), ("time", "time"),
            ("characteristic_length", "char_length")):
        if key in units:
            parms[parm] = units[key]
    parent.setParms(parms)


def read_params(kwargs):
    parent = kwargs["node"]
    selected_value = parent.evalParm("old_input_dir").strip()
    if not selected_value:
        _message("Please provide the path to a previous input dir")
        return
    selected_dir = os.path.abspath(os.path.expandvars(selected_value))
    # Accept either the simulation root or its input directory.
    if os.path.isfile(os.path.join(selected_dir, "input", "params.json")):
        input_dir = os.path.join(selected_dir, "input")
    else:
        input_dir = selected_dir
    params_path = os.path.join(input_dir, "params.json")
    try:
        with open(params_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _message(f"Error reading JSON file: {e}")
        return
    if not isinstance(data, dict):
        _message("The root of params.json must be a JSON object.")
        return
    root_path = data.get("root_path")
    if isinstance(root_path, str) and root_path.strip():
        expanded_root = os.path.expandvars(root_path)
        resource_dir = expanded_root if os.path.isabs(expanded_root) \
            else os.path.join(input_dir, expanded_root)
        resource_dir = os.path.abspath(resource_dir)
    else:
        resource_dir = input_dir

    warnings = []
    legacy = _looks_legacy(data)
    if legacy:
        warnings.append(
            "Older PolyFEM/HDA id schema detected. Compatible values were "
            "mapped to the current controls.")

    clear_all_geos(kwargs)
    working_dir = os.path.dirname(os.path.normpath(input_dir))
    parent.setParms({"working_dir": working_dir + "/"})

    raw_geometry = data.get("geometry", [])
    if not isinstance(raw_geometry, list):
        _message("The geometry section is not a list.")
        return
    geometry = [
        item for item in raw_geometry
        if isinstance(item, dict) and "mesh" in item]
    ignored_geometry = len(raw_geometry) - len(geometry)
    if ignored_geometry:
        warnings.append(
            f"{ignored_geometry} procedural/non-mesh geometry entr"
            f"{'y was' if ignored_geometry == 1 else 'ies were'} not "
            "representable by this HDA and were skipped.")
    if not geometry:
        _message("No geometry in JSON file")
        return
    parent.setParms({"num_geos": len(geometry), "geo_int": len(geometry)})

    valid_geometries = []
    for i, g in enumerate(geometry, 1):
        mesh = g.get("mesh", "")
        if not isinstance(mesh, str) or not mesh:
            warnings.append(f"Geometry {i}: mesh path is not a file string.")
            continue
        if not os.path.isabs(mesh):
            mesh = os.path.join(resource_dir, mesh)
        if not os.path.isfile(mesh):
            warnings.append(f"Geometry {i}: mesh file not found: {mesh}")
            continue
        parent.setParms({f"file_location{i}": mesh,
                         f"is_enabled{i}": bool(g.get("enabled", True)),
                         f"is_obstacle{i}": bool(g.get("is_obstacle", False))})
        parent.parm(f"file_location{i}").pressButton()
        if parent.node(f"geo_{i}") is None:
            warnings.append(
                f"Geometry {i}: Houdini could not construct the mesh node tree.")
            continue
        valid_geometries.append(i)
        tr = g.get("transformation", {})
        if not isinstance(tr, dict):
            tr = {}

        def vector3(value, default):
            if isinstance(value, (int, float)):
                return [value, value, value]
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                return list(default)
            return list(value[:3])

        t = vector3(tr.get("translation"), [0, 0, 0])
        r = vector3(tr.get("rotation"), [0, 0, 0])
        s = vector3(tr.get("scale"), [1, 1, 1])
        if isinstance(s, (int, float)):
            s = [s, s, s]
        parent.setParms({
            f"xform_t__{i}x": t[0], f"xform_t__{i}y": t[1], f"xform_t__{i}z": t[2],
            f"xform_r__{i}x": r[0], f"xform_r__{i}y": r[1], f"xform_r__{i}z": r[2],
            f"xform_s__{i}x": s[0], f"xform_s__{i}y": s[1], f"xform_s__{i}z": s[2],
            # params.json stores the already pivot-compensated affine TRS.
            # Keeping create_geo_nodes' centroid pivot would compensate it a
            # second time and make every rotated/scaled round trip drift.
            f"tpivot_{i}x": 0, f"tpivot_{i}y": 0, f"tpivot_{i}z": 0,
            f"rpivot_{i}x": 0, f"rpivot_{i}y": 0, f"rpivot_{i}z": 0})

    # Restore edited Entity assignments before materials, orders, sidesets, or
    # conditions are populated; all of those multiparms depend on volume count.
    material_slots = {}
    for i, g in enumerate(geometry, 1):
        if i in valid_geometries and not bool(g.get("is_obstacle", False)):
            material_slots.update(_restore_volume_assignments(
                parent, i, g, resource_dir, legacy, warnings))

    # Also recognize both HDA id encodings when volume_selection was absent.
    for geo in valid_geometries:
        if parent.evalParm(f"is_obstacle{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            material_slots.setdefault(1000 * geo + vol, (geo, vol))
            material_slots.setdefault(
                int(f"{geo}{vol}"), (geo, vol))

    # Materials: use the volume-selection lookup first. This preserves the
    # association even for arbitrary material ids and edited multi-material
    # meshes, instead of assuming the ids still match the MSH physical groups.
    type_menu = {name: index for index, name in enumerate(MATERIAL_TOKENS)}
    type_menu.setdefault("MooneyRivlin3ParmSymbolic",
                         type_menu["MooneyRivlin3ParamSymbolic"])
    params_dir = resource_dir
    materials = data.get("materials", [])
    if not isinstance(materials, list):
        materials = []
        warnings.append("The materials section was not a list; defaults remain.")
    all_slots = sorted(set(material_slots.values()))
    for material_index, m in enumerate(materials):
        if not isinstance(m, dict):
            warnings.append(
                f"Material entry {material_index + 1} was not an object.")
            continue
        mid = m.get("id")
        slot = _material_slot(mid, material_slots, legacy)
        if slot is None and mid is None and material_index < len(all_slots):
            slot = all_slots[material_index]
            warnings.append(
                f"Material entry {material_index + 1} had no id and was "
                f"associated positionally with geo {slot[0]} subdomain "
                f"{slot[1]}.")
        if slot is None and len(materials) == len(all_slots) \
                and material_index < len(all_slots):
            slot = all_slots[material_index]
            warnings.append(
                f"Material id {mid} did not match this HDA's id convention "
                f"and was associated positionally with geo {slot[0]} "
                f"subdomain {slot[1]}.")
        if slot is None:
            warnings.append(f"Material id {mid} could not be mapped; skipped.")
            continue
        geo, vol = slot
        if parent.parm(f"materials{geo}_{vol}") is None:
            warnings.append(
                f"Material id {mid} has no matching geometry/subdomain "
                f"(geo {geo}, vol {vol}); skipped.")
            continue
        token = m.get("type")
        if token not in type_menu:
            warnings.append(
                f"Unknown material type '{token}' on id {mid}; its parameters "
                "could not be shown in the interface.")
            continue
        parms = {f"materials{geo}_{vol}": type_menu[token]}
        if "rho" in m:
            parms[f"rho{geo}_{vol}"] = m["rho"]
        parent.setParms(parms)

        if token == "MaterialSum":
            _import_composite(parent, geo, vol, m.get("models", []),
                              params_dir, warnings)
        elif token in FIBER_MODELS:
            _import_fiber_model(
                parent, geo, vol, None, m, params_dir, warnings)
        else:
            _import_isotropic(parent, geo, vol, m)

    _restore_time(parent, data)
    _restore_contact(parent, data)
    _restore_space(parent, data, material_slots, legacy, warnings)
    _restore_solver(parent, data, legacy, warnings)
    _restore_output(parent, data)
    _restore_units(parent, data)

    # Current collision-free ids round-trip directly. Older HDA decimal ids and
    # generic PolyFEM selection ids are rebuilt through a compatibility mapper.
    if not legacy:
        try:
            _restore_conditions(parent, data)
        except Exception as e:
            warnings.append(f"Conditions were only partially restored: {e}")
        try:
            warnings.extend(_restore_selections(
                parent, geometry, resource_dir))
        except Exception as e:
            warnings.append(f"Selections were only partially restored: {e}")
    else:
        try:
            warnings.extend(_restore_legacy_scene(
                parent, data, geometry, resource_dir, material_slots))
        except Exception as exc:
            warnings.append(
                f"Older selections/conditions were only partially restored: "
                f"{exc}")
    for i, g in enumerate(geometry, 1):
        if g.get("is_obstacle"):
            continue
        try:
            for vol in range(1, parent.evalParm(f"num_volumes{i}") + 1):
                if parent.parm(f"sideset_selection{i}_{vol}") is not None:
                    _rebuild_group_chain(parent, str(i), str(vol))
        except Exception as e:
            warnings.append(f"Group nodes for geometry {i}: {e}")

    for section in ("time", "contact", "space", "solver", "output"):
        if section not in data:
            warnings.append(
                f"Section '{section}' is absent; current HDA defaults were "
                "retained for it.")
    represented_top_level = {
        "geometry", "materials", "time", "contact", "space", "solver",
        "output", "units", "boundary_conditions", "initial_conditions",
        "root_path"}
    unrepresented = sorted(set(data) - represented_top_level)
    if unrepresented:
        warnings.append(
            "Top-level JSON section(s) retained only in the source file (no "
            "equivalent HDA controls): " + ", ".join(unrepresented))

    for geo in valid_geometries:
        if not parent.evalParm(f"is_obstacle{geo}"):
            try:
                update_fiber_data(parent, geo)
            except hou.Error as exc:
                warnings.append(
                    f"Geometry {geo}: material preview refresh failed ({exc}).")

    summary = [
        f"Imported {len(valid_geometries)} geometry entr"
        f"{'y' if len(valid_geometries) == 1 else 'ies'} and "
        f"{len(materials)} material entr"
        f"{'y' if len(materials) == 1 else 'ies'} from {params_path}."
    ]
    if warnings:
        summary.append(
            f"{len(warnings)} item{'s' if len(warnings) != 1 else ''} need "
            "attention:")
        summary.extend(f"- {item}" for item in warnings)
    else:
        summary.append("All represented scene and PolyFEM parameters restored.")
    report_text = "\n".join(summary)
    report_parm = parent.parm("import_report")
    if report_parm is not None:
        report_parm.set(report_text)
    if warnings:
        _message(
            "Import completed with compatibility notes:\n"
            + "\n".join(f"• {item}" for item in warnings[:20])
            + ("\n• See Import Report for the complete list."
               if len(warnings) > 20 else ""))
    _status(
        f"Imported params.json: {len(valid_geometries)} geometries, "
        f"{len(materials)} materials, {len(warnings)} compatibility notes")


def geo_set(kwargs, val):
    kwargs["node"].setParms({"geo_int": val})
