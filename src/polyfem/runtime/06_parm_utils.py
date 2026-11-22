def _ortho_parms_at_default(parent, geo, vol):
    for base in ("ortho_E", "ortho_nu", "ortho_G"):
        pt = parent.parmTuple(f"{base}{geo}_{vol}")
        if pt is not None and not pt.isAtDefault():
            return False
    return True


def _set_ortho_parms(parent, geo, vol, vals):
    """Populate structured orthotropic tuples from 9 constants."""
    if len(vals) != 9:
        return
    e_pt = parent.parmTuple(f"ortho_E{geo}_{vol}")
    nu_pt = parent.parmTuple(f"ortho_nu{geo}_{vol}")
    g_pt = parent.parmTuple(f"ortho_G{geo}_{vol}")
    if e_pt is not None:
        e_pt.set(tuple(float(v) for v in vals[0:3]))
    if nu_pt is not None:
        nu_pt.set(tuple(float(v) for v in vals[3:6]))
    if g_pt is not None:
        g_pt.set(tuple(float(v) for v in vals[6:9]))


def _orthotropic_values(parent, geo, vol):
    e = parent.parmTuple(f"ortho_E{geo}_{vol}").eval()
    nu = parent.parmTuple(f"ortho_nu{geo}_{vol}").eval()
    g = parent.parmTuple(f"ortho_G{geo}_{vol}").eval()
    return [float(e[0]), float(e[1]), float(e[2]),
            float(nu[0]), float(nu[1]), float(nu[2]),
            float(g[0]), float(g[1]), float(g[2])]


def sync_ortho_tensor(kwargs):
    """Keep elast_tensor synced whenever structured orthotropic inputs edit."""
    parent = kwargs["node"]
    geo = kwargs.get("script_multiparm_index2")
    vol = kwargs.get("script_multiparm_index")
    if not (geo and vol):
        return
    mode_parm = parent.parm(f"elast_mode{geo}_{vol}")
    if mode_parm is None or mode_parm.evalAsString() != "orthotropic":
        return
    et_parm = parent.parm(f"elast_tensor{geo}_{vol}")
    if et_parm is not None and parent.parmTuple(f"ortho_E{geo}_{vol}") is not None:
        vals = _orthotropic_values(parent, geo, vol)
        et_parm.set(json.dumps(vals))


def _check_orthotropic_pd(values):
    """Return an error string if 9 orthotropic constants are not positive
    definite, or None when physically valid."""
    e1, e2, e3, nu12, nu13, nu23, g23, g31, g12 = [float(v) for v in values]
    if not (e1 > 0 and e2 > 0 and e3 > 0):
        return (f"Young's moduli E1, E2, E3 must all be > 0 "
                f"(got {e1:g}, {e2:g}, {e3:g}).")
    if not (g23 > 0 and g31 > 0 and g12 > 0):
        return (f"Shear moduli G23, G31, G12 must all be > 0 "
                f"(got {g23:g}, {g31:g}, {g12:g}).")
    nu21 = nu12 * e2 / e1
    nu31 = nu13 * e3 / e1
    nu32 = nu23 * e3 / e2
    if not (1.0 - nu12 * nu21 > 0 and 1.0 - nu13 * nu31 > 0
            and 1.0 - nu23 * nu32 > 0):
        return (
            "Orthotropic Poisson's ratios violate positive definiteness "
            "(1 - nu_ij * nu_ji <= 0); lower |nu12|, |nu13|, or |nu23|.")
    delta = (1.0 - nu12 * nu21 - nu23 * nu32 - nu13 * nu31
             - 2.0 * nu21 * nu32 * nu13)
    if not delta > 0:
        return (
            f"Orthotropic compliance matrix is not positive definite "
            f"(det factor = {delta:.4g} <= 0); adjust Poisson's ratios or "
            "Young's moduli.")
    return None


def _use_raw_elast_tensor(parent, geo, vol):
    """True when Hooke/SaintVenant should read elast_tensor string instead of
    structured ortho_E/nu/G tuples."""
    mode_parm = parent.parm(f"elast_mode{geo}_{vol}")
    if mode_parm is None or parent.parmTuple(f"ortho_E{geo}_{vol}") is None:
        return True
    if mode_parm.evalAsString() == "raw":
        return True
    et_parm = parent.parm(f"elast_tensor{geo}_{vol}")
    if et_parm is not None and not et_parm.isAtDefault() \
            and _ortho_parms_at_default(parent, geo, vol):
        return True
    return False


def _read_elast_tensor(parent, geo, vol):
    """Return elasticity_tensor list from structured orthotropic inputs or raw
    elast_tensor string."""
    if _use_raw_elast_tensor(parent, geo, vol):
        return parse_vector(
            parent.evalParm(f"elast_tensor{geo}_{vol}"), "elasticity_tensor")
    return _orthotropic_values(parent, geo, vol)


def _deep_merge_json(dst, src):
    """Recursively merge dict src into dict dst in place."""
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_merge_json(dst[key], val)
        else:
            dst[key] = val
    return dst


def _parse_extra_params_json(parent):
    """Parse optional Global > Extra params.json override object (or None)."""
    parm = parent.parm("extra_params_json")
    if parm is None:
        return None
    text = parm.eval().strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise hou.NodeError(
            f"Global > Extra params.json is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise hou.NodeError(
            "Global > Extra params.json must be a JSON object ({...}).")
    return parsed


def _compact_index_ranges(indices):
    """Compress sorted/unsorted integer indices into Houdini range syntax
    (e.g. [0,1,2,5,7,8] -> '0-2 5 7-8')."""
    ordered = sorted({int(i) for i in indices})
    if not ordered:
        return ""
    parts = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            parts.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = idx
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return " ".join(parts)


def create_gmsh_sidesets(kwargs):
    """Create PolyFEM sidesets from 2D physical surface groups in a .msh file.

    Matches each Gmsh 2D physical surface (tri/quad elements with entity != 0)
    to this geometry's boundary faces in null_<geo> by sorted Gmsh node-tag
    sets, groups them by (subdomain, physical tag), and appends any sideset
    that is not already present.
    """
    parent = kwargs["node"]
    geo = int(kwargs.get("script_multiparm_index", 1))
    if parent.parm(f"is_obstacle{geo}") is not None \
            and parent.evalParm(f"is_obstacle{geo}"):
        _message("Obstacles take a displacement, not sidesets.")
        return []
    null_node = parent.node(f"null_{geo}")
    if parent.node(f"geo_{geo}") is None or null_node is None:
        _message(f"Geometry {geo}: load a mesh file first.")
        return []
    path = parent.evalParm(f"file_location{geo}").strip()
    if not path or not os.path.isfile(path):
        _message(f"Geometry {geo}: mesh file not found: {path}")
        return []
    if os.path.splitext(path)[1].lower() != ".msh":
        _message(f"Geometry {geo}: Gmsh physical surfaces require a .msh file.")
        return []
    reader = hou.nodeType(hou.sopNodeTypeCategory(), "MSH_Reader::3.0")
    if reader is None:
        _message("MSH_Reader::3.0 is not installed.")
        return []
    try:
        mesh = reader.hdaModule().read_msh(path)
    except Exception as exc:
        _message(f"Geometry {geo}: could not read {path}: {exc}")
        return []
    cells = mesh.get("cells", {})
    node_tags = np.asarray(mesh["node_tags"], dtype=np.int64)
    phys_names = mesh.get("physical_names", {})

    # Build lookup: sorted 0-based gmsh node-tag tuple -> (surface prim, Entity)
    surf_geo = null_node.geometry()
    surf_entity, face_keys = _surface_arrays(surf_geo)
    face_lookup = {}
    for prim_idx in range(len(face_keys)):
        row = face_keys[prim_idx]
        key = tuple(sorted(int(k) for k in row if k >= 0))
        if key:
            face_lookup[key] = (prim_idx, int(surf_entity[prim_idx]))

    grouped = {}
    unmatched = 0
    for family in ("tri", "quad"):
        if family not in cells:
            continue
        corners = cells[family]["corners"]
        entities = cells[family]["entity"]
        for row_idx in range(len(entities)):
            tag = int(entities[row_idx])
            if tag == 0:
                continue
            key = tuple(sorted(int(node_tags[c]) - 1 for c in corners[row_idx]))
            hit = face_lookup.get(key)
            if hit is None:
                unmatched += 1
                continue
            prim_idx, vol = hit
            grouped.setdefault((vol, tag), []).append(prim_idx)

    if not grouped:
        _message(
            f"Geometry {geo}: no 2D physical surface groups matched boundary "
            f"faces in {os.path.basename(path)}."
            + (f" ({unmatched} interior/unmatched surface elements skipped.)"
               if unmatched else ""))
        return []

    parent.allowEditingOfContents()
    created = []
    skipped = 0
    for (vol, tag), prims in sorted(grouped.items()):
        ss_parm = parent.parm(f"sideset_selection{geo}_{vol}")
        if ss_parm is None:
            continue
        target_set = set(prims)
        pattern = _compact_index_ranges(prims)
        existing_count = int(ss_parm.eval())
        already = False
        for s_idx in range(1, existing_count + 1):
            gt = parent.parm(f"grouptype{geo}_{vol}_{s_idx}")
            bg = parent.parm(f"basegroup{geo}_{vol}_{s_idx}")
            if gt is not None and gt.eval() == 0 and bg is not None:
                raw_pat = bg.eval().strip()
                if raw_pat == pattern:
                    already = True
                    break
                if raw_pat:
                    try:
                        parsed = set(expand_group_str(raw_pat))
                        if parsed == target_set:
                            already = True
                            break
                    except Exception:
                        pass
        if already:
            skipped += 1
            continue
        new_idx = existing_count + 1
        ss_parm.set(new_idx)
        _rebuild_group_chain(parent, str(geo), str(vol), default_condition=True)
        parent.setParms({
            f"grouptype{geo}_{vol}_{new_idx}": 0,
            f"basegroup{geo}_{vol}_{new_idx}": pattern,
        })
        name = phys_names.get((2, tag), "")
        created.append({
            "geometry": geo, "subdomain": vol, "sideset": new_idx,
            "tag": tag, "name": name, "faces": len(target_set),
            "pattern": pattern,
        })

    if created:
        parts = []
        for item in created:
            label = f"'{item['name']}' (tag {item['tag']})" if item["name"] \
                else f"tag {item['tag']}"
            parts.append(
                f"Subdomain {item['subdomain']} Sideset {item['sideset']} = "
                f"{label} ({item['faces']} faces)")
        note = (f"; {skipped} already present" if skipped else "")
        _status(f"Geometry {geo}: created {len(created)} sideset(s) from Gmsh "
                f"surfaces — " + "; ".join(parts) + note)
    elif skipped:
        _status(
            f"Geometry {geo}: all {skipped} Gmsh physical surface sideset(s) "
            "already exist.")
    return created


def colors():
    return {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 4: (1, 1, 0),
            5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 0.5, 0), 8: (0.5, 0, 1),
            9: (0, 0.5, 0.5), 10: (0.6, 0.3, 0.1)}


def default_color(geo, vol=1):
    """Distinct bright palette color per (geometry, subdomain)."""
    palette = colors()
    idx = (int(geo) - 1 + 3 * (int(vol) - 1)) % len(palette) + 1
    return palette[idx]


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
if (!chi("../is_enabled{int(geo)}")) {{
    v@Cd = lerp(v@Cd, {{0.35, 0.35, 0.35}}, 0.7);
    f@Alpha = 0.25;
}}
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


# The element numbers of a Houdini group pattern as the viewport writes it:
# "7", "5-8", and the stepped ranges a regular selection is compacted to,
# "0-10:2" (every 2nd) and "0-10:2,3" (the first 2 of every 3).
_GROUP_TOKEN_RE = re.compile(r"(\d+)(?:-(\d+)(?::(\d+)(?:,(\d+))?)?)?")


def _group_tokens(group_input):
    """Whitespace-separated tokens; a comma inside a stepped range belongs
    to it ("0-10:2,3"), any other comma separates."""
    tokens = []
    for chunk in str(group_input).split():
        parts = [part for part in chunk.split(",") if part]
        index = 0
        while index < len(parts):
            token = parts[index]
            if re.fullmatch(r"\d+-\d+:\d+", token) \
                    and index + 1 < len(parts) and parts[index + 1].isdigit():
                token += "," + parts[index + 1]
                index += 1
            tokens.append(token)
            index += 1
    return tokens


def _expand_group_token(token):
    """The numbers one token selects, or None when it is not understood."""
    m = _GROUP_TOKEN_RE.fullmatch(token)
    if m is None:
        return None
    first = int(m.group(1))
    if m.group(2) is None:
        return [first]
    last = int(m.group(2))
    if m.group(3) is None:
        return range(first, last + 1)
    if m.group(4) is None:
        step = int(m.group(3))
        return range(first, last + 1, step) if step > 0 else None
    keep, step = int(m.group(3)), int(m.group(4))
    if step <= 0:
        return None
    return [n for n in range(first, last + 1) if (n - first) % step < keep]


def expand_group_str(group_input):
    """'0 2 5-8 10-20:5' -> [0, 2, 5, 6, 7, 8, 10, 15, 20] (the numeric part
    of Houdini's group syntax; see unsupported_group_tokens for the rest)."""
    out = []
    for token in _group_tokens(group_input):
        numbers = _expand_group_token(token)
        if numbers is not None:
            out.extend(numbers)
    return out


def unsupported_group_tokens(group_input):
    """The tokens of a pattern expand_group_str cannot turn into numbers
    (group names, !, ^, @attribute, ...), which it leaves out."""
    return [token for token in _group_tokens(group_input)
            if _expand_group_token(token) is None]


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


