"""
State:          Stevenabramowitch::dev::PolyFEM::2.0
State type:     stevenabramowitch::dev::PolyFEM::2.0
Description:    Stevenabramowitch::dev::PolyFEM::2.0
Author:         stevenabramowitch
"""

import hou
import viewerstate.utils as su


class State(object):
    MSG = ("Select specific geometry by clicking on it. Press Y to cycle "
           "handle types (translate, scale, etc.). Hit 'esc' to exit.")

    def __init__(self, state_name, scene_viewer):
        current_tab = 0
        try:
            desktop = hou.ui.curDesktop()
            pane_tab = (desktop.paneTabOfType(hou.paneTabType.Parm, 0)
                        if desktop is not None else None)
            if pane_tab is not None:
                tab = pane_tab.multiParmTab("num_geos")
                if isinstance(tab, int) and tab >= 0:
                    current_tab = tab
        except hou.Error:
            current_tab = 0

        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.handle = hou.Handle(scene_viewer, "Transform")
        self.node = None
        self.geo_of_prim = current_tab + 1
        self.handle_settings = (
            "translate(1) scale(1) rotate(1) snap_to_selection(1)")

    # --- helpers --------------------------------------------------------------
    def _read_pivot_local(self, idx: int) -> hou.Vector3:
        xf = self.node.node(f"transform_{idx}") if self.node is not None else None
        if not xf:
            return hou.Vector3(0, 0, 0)
        return hou.Vector3(
            xf.parm("px").eval() if xf.parm("px") else 0,
            xf.parm("py").eval() if xf.parm("py") else 0,
            xf.parm("pz").eval() if xf.parm("pz") else 0,
        )

    # --- handle placement -----------------------------------------------------
    def _apply_pivot(self, parms: dict):
        idx = int(self.geo_of_prim)
        c = self._read_pivot_local(idx)
        p_world = c * self.node.worldTransform()
        parms["px"] = float(p_world[0])
        parms["py"] = float(p_world[1])
        parms["pz"] = float(p_world[2])

    # --- state lifecycle ------------------------------------------------------
    def onEnter(self, kwargs):
        self.node = kwargs["node"]
        self.scene_viewer.setPromptMessage(State.MSG)
        self.scene_viewer.showOperationBar(False)
        self.scene_viewer.showSelectionBar(False)
        self.handle.show(True)
        self.handle.applySettings(self.handle_settings)
        self.handle.update()
        self.geo_of_prim = int(self.geo_of_prim) or 1

    def onExit(self, kwargs):
        self.scene_viewer.showOperationBar(True)
        self.scene_viewer.showSelectionBar(True)

    # --- interaction ----------------------------------------------------------
    def onMouseEvent(self, kwargs):
        ui_event = kwargs["ui_event"]
        if ui_event.reason() != hou.uiEventReason.Picked:
            return False
        if self.node is None:
            return False
        all_sop = self.node.node("all")
        if all_sop is None:
            return False
        try:
            collisiongeo = all_sop.geometry()
            if collisiongeo is None:
                return False
            origin, direction = ui_event.ray()
            # Transform world ray into object local space before intersecting.
            inv = self.node.worldTransform().inverted()
            local_origin = origin * inv
            local_dir = direction.multiplyAsDir(inv)
            gi = su.GeometryIntersector(
                collisiongeo, scene_viewer=self.scene_viewer)
            gi.intersect(local_origin, local_dir)
            if not gi.intersected or gi.prim_num < 0:
                return False
            prim = collisiongeo.prim(gi.prim_num)
            if prim is None:
                return False
            new_idx = int(prim.attribValue("geometry_num"))
            if new_idx != int(self.geo_of_prim):
                self.geo_of_prim = new_idx
                desktop = hou.ui.curDesktop()
                pane_tab = (desktop.paneTabOfType(hou.paneTabType.Parm, 0)
                            if desktop is not None else None)
                if pane_tab is not None:
                    pane_tab.setMultiParmTab("num_geos", max(0, new_idx - 1))
                self.handle.update(True)
            return True
        except hou.Error:
            return False

    # --- handle <-> state -----------------------------------------------------
    def onHandleToState(self, kwargs):
        parms = kwargs["parms"]
        self._apply_pivot(parms)
        if self.node is None:
            return
        try:
            idx = int(self.geo_of_prim)
            with hou.undos.group("PolyFEM Transform Handle"):
                self.node.setParms({
                    f"xform_t__{idx}x": parms["tx"],
                    f"xform_t__{idx}y": parms["ty"],
                    f"xform_t__{idx}z": parms["tz"],
                    f"xform_r__{idx}x": parms["rx"],
                    f"xform_r__{idx}y": parms["ry"],
                    f"xform_r__{idx}z": parms["rz"],
                    f"xform_s__{idx}x": parms["sx"],
                    f"xform_s__{idx}y": parms["sy"],
                    f"xform_s__{idx}z": parms["sz"],
                })
        except hou.Error:
            pass

    def onStateToHandle(self, kwargs):
        parms = kwargs["parms"]
        idx = int(self.geo_of_prim)

        def safe(name, default=0.0):
            p = self.node.parm(name) if self.node is not None else None
            return p.evalAsFloat() if p is not None else default

        parms["tx"] = safe(f"xform_t__{idx}x")
        parms["ty"] = safe(f"xform_t__{idx}y")
        parms["tz"] = safe(f"xform_t__{idx}z")
        parms["rx"] = safe(f"xform_r__{idx}x")
        parms["ry"] = safe(f"xform_r__{idx}y")
        parms["rz"] = safe(f"xform_r__{idx}z")
        parms["sx"] = safe(f"xform_s__{idx}x", 1.0)
        parms["sy"] = safe(f"xform_s__{idx}y", 1.0)
        parms["sz"] = safe(f"xform_s__{idx}z", 1.0)
        self._apply_pivot(parms)


def createViewerStateTemplate():
    state_typename = kwargs["type"].definition().sections()[
        "DefaultState"].contents()
    state_label = "Stevenabramowitch::dev::PolyFEM::2.0"
    state_cat = hou.objNodeTypeCategory()
    template = hou.ViewerStateTemplate(state_typename, state_label, state_cat)
    template.bindFactory(State)
    template.bindIcon(kwargs["type"].icon())
    template.bindHandle("xform", "Transform", cache_previous_parms=False)
    return template
