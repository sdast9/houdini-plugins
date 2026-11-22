"""Multi-Block and Comparison tab parameter templates."""

import hou

from .helpers import READ_ONLY


def build_multi_block_tab():
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
    return multi


def build_comparison_tab():
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
    return compare
