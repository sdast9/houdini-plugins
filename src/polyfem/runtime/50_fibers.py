def _read_element_file(path, count, components, field, label, parent=None,
                       geo=None):
    path = os.path.expandvars(path)
    if not os.path.isfile(path):
        raise hou.NodeError(f"{label}: file not found: {path}")
    if components == 3 and path.lower().endswith(".vtk"):
        try:
            values = read_cell_vectors_legacy_vtk(path, field or "FIB_DIR1")
        except hou.NodeError:
            if os.path.basename(path).lower() != FIBER_FILE \
                    or os.path.basename(os.path.dirname(
                        os.path.normpath(path))) != "input":
                raise
            raise hou.NodeError(
                f"{label}: array '{field or 'FIB_DIR1'}' is not in {path}. "
                f"input/{FIBER_FILE} is the file Write creates for the "
                "solver, and an earlier Write replaced the fiber file that "
                "was staged under that name. Pick your original fiber file "
                "again (it is now staged under a name Write never uses).")
    else:
        delimiter = None
        if ".vtk" not in path.lower():
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.split("#", 1)[0].strip()
                    if stripped:
                        delimiter = "," if "," in stripped else None
                        break
        try:
            values = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        except (OSError, ValueError) as exc:
            raise hou.NodeError(f"{label}: could not read {path} ({exc})")
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
    if parent is not None and geo is not None:
        branch_geo = parent.node(f"branch_{geo}").geometry()
        values = values[np.argsort(solver_element_order(branch_geo))]
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
        out = np.flatnonzero(~np.isfinite(values) | (values < 0) | (values > limit + 1e-9))
        if len(out):
            raise hou.NodeError(
                f"{label}: dispersion must be finite and within [0, 1/3]; element(s) "
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
    centroids = _prim_centroids(geo_data, None)
    if geo_data.findPrimAttrib("centroid") is None:
        geo_data.addAttrib(hou.attribType.Prim, "centroid", (0.0, 0.0, 0.0),
                           create_local_variable=False)
    geo_data.setPrimFloatAttribValuesFromString(
        "centroid", np.ascontiguousarray(centroids, dtype=np.float64).tobytes(),
        float_type=hou.numericData.Float64)

    # Every transform edit recooks this node. Resolving fiber and dispersion
    # sources (files, SOP attributes, cylindrical frames) is only worth it
    # while something shows them; the export resolves them on its own.
    if not material_display_active(parent, geo):
        return
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

    mask = _volume_mask(geo_data)
    total = geo_data.intrinsicValue("primitivecount")
    index = np.flatnonzero(mask) if mask is not None else np.arange(total)
    entity_attrib = geo_data.findPrimAttrib("Entity")
    entity_values = np.frombuffer(
        geo_data.primIntAttribValuesAsString("Entity"), dtype=np.int32) \
        if entity_attrib is not None else None
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


def material_display_active(parent, geo):
    """Whether the viewport shows per-element material data for a geometry:
    fiber lines, or a surface colored by dispersion or fiber direction."""
    show = parent.parm(f"show_fibers{geo}")
    mode = parent.parm(f"color_by{geo}")
    return bool(show is not None and show.eval()) or (
        mode is not None and mode.evalAsString() != "subdomains")


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
    if material_display_active(parent, geo):
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


def export_per_element_materials(parent, input_dir, output_dir=None,
                                 cache=None, shared_fields=None):
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
    _restage_export_named_files(parent, input_dir)
    table, total = global_element_table(parent)
    if not total:
        return
    shared_fields = shared_fields or {}
    shared_sources = {source for sources in shared_fields.values()
                      for source in sources.values() if isinstance(source, str)}
    fiber_fields = {name: np.tile([1.0, 0.0, 0.0], (total, 1))
                    for name in shared_fields}
    shared_by_geo = {}
    for name, sources in shared_fields.items():
        for (geo, vol), source in sources.items():
            shared_by_geo.setdefault(geo, []).append((name, vol, source))
    family_names, family_labels, family_body_ids, family_values = [], [], [], []
    all_centroids = np.zeros((total, 3), dtype=np.float64)
    all_body_ids = np.zeros(total, dtype=np.int64)
    for geo, offset, count in table:
        fields, resolved = _cached(
            cache, "pe_data", int(geo), lambda g=geo: per_element_data(parent, g))
        if resolved != count:
            raise hou.NodeError(
                f"Geometry {geo}: element count changed while exporting "
                f"({resolved} vs {count}); re-cook the node and try again")
        branch_geo = parent.node(f"branch_{geo}").geometry()
        volume_mask = _volume_mask(branch_geo)
        order = solver_element_order(branch_geo, expect_msh=True)
        entities = np.frombuffer(
            branch_geo.primIntAttribValuesAsString("Entity"),
            dtype=np.int32)
        if volume_mask is not None:
            entities = entities[volume_mask]
        entities = entities[order]
        all_centroids[offset:offset + count] = _element_centroids(parent, geo)[order]
        all_body_ids[offset:offset + count] = \
            1000 * int(geo) + entities.astype(np.int64)

        local_fields = {}
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            for fam, model in material_families(parent, geo, vol):
                for side in mirror_sides(parent, geo, vol, fam):
                    values = fields.get(fiber_attrib(vol, fam, side))
                    if values is not None:
                        values = values[order]
                        name = fiber_field(geo, vol, fam, side)
                        local_fields[name] = values
                        if name not in shared_sources:
                            column = fiber_fields.setdefault(
                                name, np.tile([1.0, 0.0, 0.0], (total, 1)))
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
                    padded[offset:offset + count] = kappa[order]
                    np.savetxt(
                        os.path.join(
                            input_dir,
                            per_element_scalar_file(geo, vol, fam, "kappa")),
                        padded, fmt="%.9g")

        for name, vol, source in shared_by_geo.get(geo, ()):
            owned = entities == vol
            values = local_fields[source][owned] if isinstance(source, str) else source
            fiber_fields[name][offset:offset + count][owned] = values

    # Disabled geometries still have their material block written to
    # params.json (so Import restores them); emit valid length-`total`
    # placeholders for any per-element fiber/kappa files those blocks name
    # so PolyFEM does not fail with missing files/fields.
    active_geos = {geo for geo, _, _ in table}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if geo in active_geos or parent.evalParm(f"is_obstacle{geo}"):
            continue
        for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
            for fam, model in material_families(parent, geo, vol):
                src = _slot_parm(parent, geo, vol, fam, "fib_source")
                for side in mirror_sides(parent, geo, vol, fam):
                    if src in ("attribute", "file", "cylindrical", "curve") \
                            or side:
                        fiber_fields.setdefault(
                            fiber_field(geo, vol, fam, side),
                            np.tile([1.0, 0.0, 0.0], (total, 1)))
                if model == "HGODispersion" and _slot_parm(
                        parent, geo, vol, fam, "kappa_source") in (
                            "attribute", "file"):
                    padded = np.full(
                        total, _slot_parm(parent, geo, vol, fam, "kappa"))
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


