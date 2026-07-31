"""
State:          Stevenabramowitch::dev::PolyFEM::1.0
State type:     stevenabramowitch::dev::PolyFEM::1.0
Description:    Stevenabramowitch::dev::PolyFEM::1.0
Author:         stevenabramowitch
Date Created:   March 19, 2023 - 22:23:39
"""

import hou
import viewerstate.utils as su

class State(object):
    MSG = "Select specific geometry by clicking on it. Press Y to cycle handle types (translate, scale, etc.). Hit 'esc' to exit."

    def __init__(self, state_name, scene_viewer):
        desktop = hou.ui.curDesktop()
        pane_tab = desktop.paneTabOfType(hou.paneTabType.Parm, 0)
        current_tab = pane_tab.multiParmTab("num_geos")

        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.handle = hou.Handle(scene_viewer, "Transform")
        self.node = None
        self.geo_of_prim = current_tab + 1
        self.handle_settings = "translate(1) scale(1) rotate(1) snap_to_selection(1)"

    # --- helpers --------------------------------------------------------------
    def _read_pivot_local(self, idx: int) -> hou.Vector3:
        xf = self.node.node(f"transform_{idx}")
        if not xf: return hou.Vector3(0,0,0)
        return hou.Vector3(
            xf.parm("px").eval() if xf.parm("px") else 0,
            xf.parm("py").eval() if xf.parm("py") else 0,
            xf.parm("pz").eval() if xf.parm("pz") else 0,
        )
        
    # --- handle placement -----------------------------------------------------
    def _apply_pivot(self, parms: dict):
        idx = int(self.geo_of_prim)
        # Always recompute centroid live:
        c = self._read_pivot_local(idx)
        p_world = c * self.node.worldTransform()
        parms["px"], parms["py"], parms["pz"] = float(p_world[0]), float(p_world[1]), float(p_world[2])

    # --- state lifecycle ------------------------------------------------------
    def onEnter(self, kwargs):
        self.node = kwargs['node']
        self.scene_viewer.setPromptMessage(State.MSG)
        self.scene_viewer.showOperationBar(False)
        self.scene_viewer.showSelectionBar(False)
        all_sop = self.node.node("all")
        self.collisiongeo = all_sop.geometry() if all_sop else None
        self.handle.show(True)
        self.handle.applySettings(self.handle_settings)
        self.handle.update()
        self.geo_of_prim = int(self.geo_of_prim) or 1

    def onExit(self, kwargs):
        self.scene_viewer.showOperationBar(True)

    # --- interaction ----------------------------------------------------------
    def onMouseEvent(self, kwargs):
        ui_event = kwargs['ui_event']
        origin, direction = ui_event.ray()
        reason = ui_event.reason()
        try:
            if self.collisiongeo is not None:
                gi = su.GeometryIntersector(self.collisiongeo, scene_viewer=self.scene_viewer)
                gi.intersect(origin, direction)
                if reason == hou.uiEventReason.Picked:
                    prim = self.collisiongeo.nearestPrim(gi.position)
                    new_idx = int(prim[0].attribValue('geometry_num'))
                    if new_idx != int(self.geo_of_prim):
                        self.geo_of_prim = new_idx
                        desktop = hou.ui.curDesktop()
                        pane_tab = desktop.paneTabOfType(hou.paneTabType.Parm, 0)
                        pane_tab.setMultiParmTab("num_geos", new_idx-1)
                        self.handle.update(True)
        except: pass

    # --- handle <-> state -----------------------------------------------------
    def onHandleToState(self, kwargs):
        parms = kwargs["parms"]
        self._apply_pivot(parms)
        try:
            idx = self.geo_of_prim
            self.node.parm(f'xform_t__{idx}x').set(parms['tx'])
            self.node.parm(f'xform_t__{idx}y').set(parms['ty'])
            self.node.parm(f'xform_t__{idx}z').set(parms['tz'])
            self.node.parm(f'xform_r__{idx}x').set(parms['rx'])
            self.node.parm(f'xform_r__{idx}y').set(parms['ry'])
            self.node.parm(f'xform_r__{idx}z').set(parms['rz'])
            self.node.parm(f'xform_s__{idx}x').set(parms['sx'])
            self.node.parm(f'xform_s__{idx}y').set(parms['sy'])
            self.node.parm(f'xform_s__{idx}z').set(parms['sz'])
        except: pass

    def onStateToHandle(self, kwargs):
        parms = kwargs["parms"]
        idx = self.geo_of_prim
        def safe(name,default=0): 
            p=self.node.parm(name)
            return p.evalAsFloat() if p else default
        parms['tx'],parms['ty'],parms['tz'] = safe(f'xform_t__{idx}x'),safe(f'xform_t__{idx}y'),safe(f'xform_t__{idx}z')
        parms['rx'],parms['ry'],parms['rz'] = safe(f'xform_r__{idx}x'),safe(f'xform_r__{idx}y'),safe(f'xform_r__{idx}z')
        parms['sx'],parms['sy'],parms['sz'] = safe(f'xform_s__{idx}x',1),safe(f'xform_s__{idx}y',1),safe(f'xform_s__{idx}z',1)
        self._apply_pivot(parms)

def createViewerStateTemplate():
    state_typename = kwargs["type"].definition().sections()["DefaultState"].contents()
    state_label = "Stevenabramowitch::dev::PolyFEM::1.0"
    state_cat = hou.objNodeTypeCategory()
    template = hou.ViewerStateTemplate(state_typename,state_label,state_cat)
    template.bindFactory(State)
    template.bindIcon(kwargs["type"].icon())
    template.bindHandle("xform","Transform",cache_previous_parms=False)
    return template