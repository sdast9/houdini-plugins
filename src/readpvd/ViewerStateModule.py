"""Viewport-overlay legend for readPVD::1.0.

The overlay bar is drawn with a hou.GeometryDrawable whose geometry is built
directly in viewport-pixel coordinates, with ``screen_space`` set to the whole
viewport (0, 0, width, height). This is the same convention Houdini's own
selection-region drawable uses (resourceutils.SelectionRegionDrawable), so the
bar lands exactly where the pixel-positioned tick/title text is drawn.
"""

import hou
import viewerstate.utils as su


# pixel gaps that tie the labels to the bar (kept in one place so the bar,
# ticks, and title stay visually consistent)
_TICK_LEN = 6
_LABEL_GAP = 8
_TITLE_GAP = 10


def _bar_geometry(node, left, bottom, width, height, segments=48):
    """Colored bar as filled quads in viewport-pixel coordinates."""
    geo = hou.Geometry()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    ramp = node.parm("color_ramp").evalAsRamp()
    for index in range(segments):
        y0 = bottom + height * (index / segments)
        y1 = bottom + height * ((index + 1) / segments)
        color0 = ramp.lookup(index / segments)
        color1 = ramp.lookup((index + 1) / segments)
        polygon = geo.createPolygon()
        for position, color in (
                ((left, y0, 0), color0), ((left + width, y0, 0), color0),
                ((left + width, y1, 0), color1), ((left, y1, 0), color1)):
            point = geo.createPoint()
            point.setPosition(hou.Vector3(position))
            point.setAttribValue("Cd", tuple(float(c) for c in color))
            polygon.addVertex(point)
    return geo


def _tick_geometry(node, left, bottom, width, height, labels_on_left=False):
    """Tick marks as open polylines (drawn by the Line drawable)."""
    geo = hou.Geometry()
    geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0),
                  create_local_variable=False)
    tick_color = tuple(float(c) for c in node.evalParmTuple(
        "legend_text_color"))
    n_ticks = max(2, int(node.evalParm("legend_ticks")))
    x0, x1 = ((left, left - _TICK_LEN) if labels_on_left
              else (left + width, left + width + _TICK_LEN))
    for index in range(n_ticks):
        y = bottom + height * (index / (n_ticks - 1))
        polyline = geo.createPolygon(is_closed=False)
        for position in ((x0, y, 0), (x1, y, 0)):
            point = geo.createPoint()
            point.setPosition(hou.Vector3(position))
            point.setAttribValue("Cd", tick_color)
            polyline.addVertex(point)
    return geo


def _bar_cache_key(node, left, bottom, width, height):
    """Cheap signature of everything _bar_geometry depends on."""
    ramp = node.parm("color_ramp").evalAsRamp()
    return (left, bottom, width, height,
            ramp.keys(), ramp.values(), tuple(str(b) for b in ramp.basis()))


def _tick_cache_key(node, left, bottom, width, height, labels_on_left):
    """Cheap signature of everything _tick_geometry depends on."""
    tick_color = tuple(float(c) for c in node.evalParmTuple(
        "legend_text_color"))
    n_ticks = max(2, int(node.evalParm("legend_ticks")))
    return (left, bottom, width, height, bool(labels_on_left),
            n_ticks, tick_color)


class State:
    """Draw a screen-fixed legend while this node's viewer state is active."""

    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.node = None
        self._bar_cache_key = None
        self._tick_cache_key = None
        self._probe_marker_pos = None
        self.bar = hou.GeometryDrawable(
            scene_viewer, hou.drawableGeometryType.Face,
            "readpvd_overlay_legend_bar")
        self.ticks = hou.GeometryDrawable(
            scene_viewer, hou.drawableGeometryType.Line,
            "readpvd_overlay_legend_ticks")
        self.text = hou.TextDrawable(scene_viewer, "readpvd_overlay_legend_text")
        self.probe_text = hou.TextDrawable(
            scene_viewer, "readpvd_probe_readout")
        self.probe_marker = hou.GeometryDrawable(
            scene_viewer, hou.drawableGeometryType.Point,
            "readpvd_probe_marker",
            params={"color1": hou.Vector4(1, 1, 0, 1), "radius": 8.0,
                    "fade_factor": 0.6})

    def onEnter(self, kwargs):
        self.node = kwargs["node"]
        self._bar_cache_key = None
        self._tick_cache_key = None
        self._probe_marker_pos = None
        for drawable in (self.bar, self.ticks, self.text,
                         self.probe_text, self.probe_marker):
            drawable.show(True)

    def onExit(self, kwargs):
        for drawable in (self.bar, self.ticks, self.text,
                         self.probe_text, self.probe_marker):
            drawable.show(False)

    def onMouseEvent(self, kwargs):
        if self.node is None or not self.node.evalParm("probe_enabled"):
            return False
        ui_event = kwargs["ui_event"]
        if ui_event.reason() != hou.uiEventReason.Picked:
            return False
        output = self.node.node("output")
        if output is None:
            return False
        geo = output.geometry()
        origin, direction = ui_event.ray()
        inv = self.node.worldTransform().inverted()
        local_origin = origin * inv
        local_direction = (
            (origin + direction) * inv - local_origin).normalized()
        # Probe the primary result only. Excluding the readpvd_* decoration
        # groups (glyphs, legend, gnomon, supplementary blocks)
        # keeps the pick from snapping to a decoration point that carries no
        # field data -- the cause of "position is right but values are zero".
        intersector = su.GeometryIntersector(
            geo, scene_viewer=self.scene_viewer, pattern="* ^readpvd_*")
        intersector.intersect(local_origin, local_direction)
        if not intersector.intersected or intersector.prim_num < 0:
            return False
        # Snap to the nearest corner of the hit primitive: a real model vertex
        # with field values, not whatever point happens to be closest in space.
        prim = geo.prim(intersector.prim_num)
        hit = intersector.position
        point = min(prim.points(),
                    key=lambda candidate: (candidate.position() - hit).length())
        # Store the vertex (its id survives clipping); onDraw redraws the
        # marker from the live geometry each frame so it follows the vertex
        # as the mesh deforms.
        self.node.hdaModule().probe_pick(self.node, point.number())
        return True

    def _layout(self):
        viewport = self.scene_viewer.curViewport()
        _, _, viewport_width, viewport_height = viewport.size()
        margin = max(0, int(self.node.evalParm("overlay_margin")))
        bar_width = max(8, int(self.node.evalParm("overlay_width")))
        bar_height = max(40, int(self.node.evalParm("overlay_height")))
        corner = self.node.evalParm("overlay_corner")

        left = margin if corner in (0, 2) else (
            viewport_width - margin - bar_width)
        bottom = margin if corner in (2, 3) else (
            viewport_height - margin - bar_height)
        return (viewport_width, viewport_height,
                left, bottom, bar_width, bar_height)

    def onDraw(self, kwargs):
        if self.node is None:
            return

        handle = kwargs["draw_handle"]
        if self.node.evalParm("probe_enabled") and self.node.evalParm(
                "probe_point") >= 0:
            # Re-read the probed point from the live geometry so the marker and
            # readout track it as the frame (and the deformation) changes.
            position, readout = self.node.hdaModule().probe_readout(
                self.node, int(self.node.evalParm("probe_point")))
            if position is not None:
                pos_key = tuple(position)
                if pos_key != self._probe_marker_pos:
                    marker_geo = hou.Geometry()
                    marker_geo.createPoint().setPosition(position)
                    self.probe_marker.setGeometry(marker_geo)
                    self._probe_marker_pos = pos_key
                self.probe_marker.setTransform(self.node.worldTransform())
                self.probe_marker.draw(handle)
                self.probe_text.draw(handle, {
                    "text": readout.replace("\n", "<br>"),
                    "multi_line": True,
                    "translate": hou.Vector3(18, 18, 0),
                    "origin": hou.drawableTextOrigin.BottomLeft,
                    "color1": hou.Color(1, 1, 0.75),
                })
        if self.node.evalParm("legend_mode") not in (2, 3):
            return

        (viewport_width, viewport_height,
         left, bottom, width, height) = self._layout()
        labels_on_left = self.node.evalParm("overlay_corner") in (1, 3)
        # screen_space covers the whole viewport; geometry is in pixels.
        full_viewport = (0, 0, viewport_width, viewport_height, 0, 0)
        bar_key = _bar_cache_key(self.node, left, bottom, width, height)
        if bar_key != self._bar_cache_key:
            self.bar.setGeometry(
                _bar_geometry(self.node, left, bottom, width, height))
            self._bar_cache_key = bar_key
        self.bar.draw(handle, {
            "screen_space": full_viewport,
            "use_cd": True,
            "backface_culling": False,
        })
        tick_key = _tick_cache_key(
            self.node, left, bottom, width, height, labels_on_left)
        if tick_key != self._tick_cache_key:
            self.ticks.setGeometry(
                _tick_geometry(self.node, left, bottom, width, height,
                               labels_on_left=labels_on_left))
            self._tick_cache_key = tick_key
        self.ticks.draw(handle, {
            "screen_space": full_viewport,
            "use_cd": True,
        })

        module = self.node.hdaModule()
        values = module.legend_values(self.node)
        digits = int(self.node.evalParm("legend_digits"))
        notation = self.node.parm("legend_number_format").evalAsString()
        color = hou.Color(self.node.evalParmTuple("legend_text_color"))
        # legend_values runs high -> low; the bar runs low (bottom) -> high.
        if labels_on_left:
            text_x = left - _TICK_LEN - _LABEL_GAP
            tick_origin = hou.drawableTextOrigin.RightCenter
            title_x = left + width
            title_origin = hou.drawableTextOrigin.BottomRight
        else:
            text_x = left + width + _TICK_LEN + _LABEL_GAP
            tick_origin = hou.drawableTextOrigin.LeftCenter
            title_x = left
            title_origin = hou.drawableTextOrigin.BottomLeft
        n = len(values)
        for index, value in enumerate(values):
            fraction = index / max(1, n - 1)
            y = bottom + height * (1.0 - fraction)
            self.text.draw(handle, {
                "text": module._format_number(value, digits, notation),
                "translate": hou.Vector3(text_x, y, 0),
                "origin": tick_origin,
                "color1": color,
            })

        self.text.draw(handle, {
            "text": f"<b>{module.legend_title_text(self.node)}</b>",
            "translate": hou.Vector3(title_x, bottom + height + _TITLE_GAP, 0),
            "origin": title_origin,
            "color1": color,
        })


def createViewerStateTemplate():
    state_name = kwargs["type"].definition().sections()["DefaultState"].contents()
    template = hou.ViewerStateTemplate(
        state_name, "Read PVD Viewport Legend", hou.objNodeTypeCategory())
    template.bindFactory(State)
    template.bindIcon(kwargs["type"].icon())
    return template
