def legend_tick_values(node, count=None):
    """Tick values from the bottom of the bar to the top, spaced the way the
    colors are mapped: evenly on the Linear scale, by equal factors on the
    Logarithmic one (1, 10, 100 -- not 1, 50.5, 100)."""
    count = max(2, int(node.evalParm("legend_ticks")) if count is None
                else int(count))
    low = float(node.evalParm("color_min"))
    high = float(node.evalParm("color_max"))
    if _menu_token(node, "color_scale") == "log10" and low > 0 and high > 0:
        return np.geomspace(low, high, count)
    return np.linspace(low, high, count)


def legend_values(node):
    """Tick values from the top of the bar to the bottom."""
    return legend_tick_values(node)[::-1]


def _percent_reference(node, comparison=None):
    if not node.evalParm("reference_enable") or _menu_token(node, "reference_mode") != "percent":
        return False
    if comparison is not None:
        return comparison
    try:
        geo = _probe_geometry(node)
        return geo is not None and geo.findPointAttrib("reference_comparison") is not None
    except hou.Error:
        return False


def legend_title_text(node, comparison=None):
    custom = node.evalParm("legend_title").strip()
    if custom:
        return custom
    field = node.evalParm("color_attrib")
    reduction = _menu_token(node, "color_reduction")
    label = _field_label(field)
    if reduction not in ("", "auto"):
        label += " - " + REDUCTION_LABELS.get(reduction, reduction)
    units = "%" if _percent_reference(node, comparison) else node.evalParm("field_units").strip()
    if units:
        label += f" [{units}]"
    return label


def legend_rotation(node):
    index = max(0, min(int(node.evalParm("legend_orientation")),
                       len(LEGEND_ROTATIONS) - 1))
    return LEGEND_ROTATIONS[index]


def _point(geo, position, color=None):
    point = geo.createPoint()
    point.setPosition(hou.Vector3(position))
    if color is not None:
        point.setAttribValue("Cd", tuple(float(x) for x in color))
    return point


def _line(geo, a, b, color, group):
    prim = geo.createPolygon()
    prim.setIsClosed(False)
    prim.addVertex(_point(geo, a, color))
    prim.addVertex(_point(geo, b, color))
    group.add(prim)
    return prim


# Scene-legend layout, all in the bar's local space (bar height = 1.0). The
# bar occupies x in [0, BAR_WIDTH]; ticks, labels, and title hang to its right
# / top with fixed gaps so spacing is independent of tick count and digits.
_LEGEND_BAR_WIDTH = 0.12
_LEGEND_TICK_LEN = 0.04
_LEGEND_LABEL_GAP = 0.025
_LEGEND_TITLE_GAP = 0.06


def _font_geometry(text, size, halign, valign, position, color, rotate=0):
    """Render `text` as pre-triangulated filled polygons in local space.

    Generates glyphs at unit scale (fontsize=1.0), triangulates concave
    bridged-hole glyph N-gons via the Divide SOP while vertices are well
    above float32 precision limits, and then scales/positions to target size.
    This avoids Houdini's viewport ear-clipper artifacts on small-scale text.
    """
    verb = hou.sopNodeTypeCategory().nodeVerb("font")
    verb.setParms({
        "text": text, "fontsize": 1.0, "halign": halign,
        "valign": valign, "type": 2, "hole": True,
        "t": (0.0, 0.0, 0.0), "r": (0.0, 0.0, 0.0)})
    raw = hou.Geometry()
    verb.execute(raw, [])
    if len(raw.prims()):
        geo = hou.Geometry()
        div = hou.sopNodeTypeCategory().nodeVerb("divide")
        div.setParms({"convex": True, "numsides": 3, "usemaxsides": True})
        div.execute(geo, [raw])
        xform = (hou.hmath.buildScale(float(size), float(size), 1.0)
                 * hou.hmath.buildRotate((0.0, 0.0, float(rotate)), "xyz")
                 * hou.hmath.buildTranslate(
                     float(position[0]), float(position[1]), 0.0))
        geo.transform(xform)
    else:
        geo = raw
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    if len(geo.points()):
        flat = np.tile(np.asarray(color, dtype=np.float64)[:3],
                       len(geo.points()))
        geo.setPointFloatAttribValuesFromString(
            "Cd", flat.tobytes(), float_type=hou.numericData.Float64)
    return geo


def cook_scene_legend(node):
    """Renderable color strip + tick marks + aligned numeric labels + title.

    Each numeric label is generated as its own font primitive and placed at
    the exact height of its tick (vertically centered), so label spacing
    always matches the colored bar regardless of tick count, digits, or
    number format.
    """
    geo = node.geometry()
    asset = node.parent()
    comparison = geo.findPointAttrib("reference_comparison") is not None
    geo.clear()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    group = geo.createPrimGroup("readpvd_legend")

    ramp = asset.parm("color_ramp").evalAsRamp()
    segments = 48
    width = _LEGEND_BAR_WIDTH
    for i in range(segments):
        y0 = i / segments
        y1 = (i + 1) / segments
        c0 = ramp.lookup(y0)
        c1 = ramp.lookup(y1)
        prim = geo.createPolygon()
        for pos, color in (((0, y0, 0), c0), ((width, y0, 0), c0),
                           ((width, y1, 0), c1), ((0, y1, 0), c1)):
            prim.addVertex(_point(geo, pos, color))
        group.add(prim)

    tick_color = tuple(asset.evalParmTuple("legend_text_color"))
    n_ticks = max(2, int(asset.evalParm("legend_ticks")))
    digits = asset.evalParm("legend_digits")
    notation = _menu_token(asset, "legend_number_format")
    low = float(asset.evalParm("color_min"))
    high = float(asset.evalParm("color_max"))
    ticks = legend_tick_values(asset, n_ticks)
    # Tick labels sized to the available spacing so they never overlap.
    label_size = min(0.055, 0.62 / (n_ticks - 1))
    label_x = width + _LEGEND_TICK_LEN + _LEGEND_LABEL_GAP
    for i in range(n_ticks):
        fraction = i / (n_ticks - 1)
        y = fraction
        _line(geo, (width, y, 0), (width + _LEGEND_TICK_LEN, y, 0),
              tick_color, group)
        value = float(ticks[i])
        label = _font_geometry(
            _format_number(value, digits, notation), label_size,
            0, 2, (label_x, y), tick_color)  # left, middle
        _merge_into(geo, group, label)

    title_text = legend_title_text(asset, comparison)
    title = _font_geometry(
        title_text, 0.075, 0, 3,  # left, bottom
        (0.0, 1.0 + _LEGEND_TITLE_GAP), tick_color)
    _merge_into(geo, group, title)

    geo.addAttrib(hou.attribType.Global, "legend_min", 0.0)
    geo.addAttrib(hou.attribType.Global, "legend_max", 0.0)
    geo.addAttrib(hou.attribType.Global, "legend_title", "")
    geo.setGlobalAttribValue("legend_min", low)
    geo.setGlobalAttribValue("legend_max", high)
    geo.setGlobalAttribValue("legend_title", title_text)


def _merge_into(geo, group, other):
    """Merge `other` into `geo`, adding its new prims to `group`."""
    first = len(geo.prims())
    geo.merge(other)
    group.add(geo.prims()[first:])


def _gnomon_basis(direction):
    """Unit axis direction plus two perpendicular unit vectors."""
    direction = hou.Vector3(direction).normalized()
    helper = hou.Vector3(0, 0, 1)
    if abs(direction.dot(helper)) > 0.9:
        helper = hou.Vector3(0, 1, 0)
    u = direction.cross(helper).normalized()
    v = direction.cross(u).normalized()
    return direction, u, v


def _quad(geo, points, color, group):
    prim = geo.createPolygon()
    for position in points:
        prim.addVertex(_point(geo, position, color))
    group.add(prim)


def _beam(geo, start, end, u, v, radius, color, group, cap_end=True):
    """Square-section solid beam between two points (a thin 3D shaft)."""
    start, end = hou.Vector3(start), hou.Vector3(end)
    offsets = (u * radius + v * radius, -u * radius + v * radius,
               -u * radius - v * radius, u * radius - v * radius)
    s = [start + o for o in offsets]
    e = [end + o for o in offsets]
    for i in range(4):
        j = (i + 1) % 4
        _quad(geo, (s[i], s[j], e[j], e[i]), color, group)
    _quad(geo, (s[3], s[2], s[1], s[0]), color, group)   # start cap
    if cap_end:
        _quad(geo, (e[0], e[1], e[2], e[3]), color, group)


def _cone(geo, tip, base_center, u, v, radius, color, group):
    """Solid 4-sided pyramid: apex at `tip`, square base at `base_center`."""
    corners = [base_center + u * radius + v * radius,
               base_center - u * radius + v * radius,
               base_center - u * radius - v * radius,
               base_center + u * radius - v * radius]
    apex = _point(geo, tip, color)
    base_points = [_point(geo, corner, color) for corner in corners]
    for i in range(4):
        face = geo.createPolygon()
        for point in (apex, base_points[i], base_points[(i + 1) % 4]):
            face.addVertex(point)
        group.add(face)
    base = geo.createPolygon()                            # base cap
    for point in reversed(base_points):
        base.addVertex(point)
    group.add(base)


def _cube(geo, center, half, color, group):
    """Solid axis-aligned cube centered at `center`."""
    center = hou.Vector3(center)
    corner = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner[(sx, sy, sz)] = _point(
                    geo, center + hou.Vector3(sx, sy, sz) * half, color)
    faces = (
        ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),     # +X
        ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),  # -X
        ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),     # +Y
        ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),  # -Y
        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),     # +Z
        ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),  # -Z
    )
    for face in faces:
        prim = geo.createPolygon()
        for sign in face:
            prim.addVertex(corner[sign])
        group.add(prim)


def cook_gnomon(node):
    """Renderable XYZ orientation marker: solid beam shafts with cone
    arrowheads and a solid cube origin marker (a refined take on the 0.26
    gnomon -- 3D heads and cube instead of flat triangles and wire crosses)."""
    geo = node.geometry()
    asset = node.parent()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    group = geo.createPrimGroup("readpvd_gnomon")
    center = hou.Vector3(asset.evalParmTuple("gnomon_center"))
    scale = float(asset.evalParm("gnomon_scale"))
    arrows = bool(asset.evalParm("gnomon_arrows"))
    shaft_radius = 0.012 * scale
    cone_radius = 0.038 * scale
    head_length = 0.22 * scale
    axes = (
        ("gnomon_x", hou.Vector3(1, 0, 0), (0.95, 0.26, 0.26)),
        ("gnomon_y", hou.Vector3(0, 1, 0), (0.36, 0.85, 0.30)),
        ("gnomon_z", hou.Vector3(0, 0, 1), (0.28, 0.48, 1.0)),
    )
    for parm, axis, color in axes:
        if not asset.evalParm(parm):
            continue
        direction, u, v = _gnomon_basis(axis)
        tip = center + direction * scale
        if arrows:
            base = tip - direction * head_length
            _beam(geo, center, base, u, v, shaft_radius, color, group,
                  cap_end=False)
            _cone(geo, tip, base, u, v, cone_radius, color, group)
        else:
            _beam(geo, center, tip, u, v, shaft_radius, color, group)
    if asset.evalParm("gnomon_center_marker"):
        _cube(geo, center, 0.045 * scale, (1.0, 0.85, 0.2), group)


def _block_for(asset, blocks):
    """Pick the vtu block matching the Source parm (Volume/Surface/Points)."""
    wanted = _source_block_token(asset)
    if wanted in blocks:
        return blocks[wanted]
    return next(iter(blocks.values()))


