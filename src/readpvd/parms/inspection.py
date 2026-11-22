"""Probe and Clip / Slice tab parameter templates."""

import hou

from .helpers import READ_ONLY


def build_probe_tab():
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
    return probe


def build_clip_tab():
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
    return section
