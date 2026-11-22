"""Analysis tab parameter templates."""

import hou

from .force_curve import _force_curve_folder
from .helpers import _menu_script


def build_analysis_tab():
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
    derived_mode = hou.MenuParmTemplate(
        "derived_mode", "Derived Fields Scope",
        ("on_demand", "all"),
        menu_labels=("Active Display / Glyphs / Probe (fast)",
                     "All Mechanics Fields (downstream SOPs)"),
        default_value=0,
        help="Controls which derived mechanics attributes are computed and "
             "attached to the geometry. 'Active Display / Glyphs / Probe' "
             "computes only the fields needed by the current Color Field, "
             "glyphs, fibers, and probe (much faster and far less memory on "
             "large meshes). Switch to 'All Mechanics Fields' when downstream "
             "SOPs need every derived tensor and invariant simultaneously.")
    derived_mode.setConditional(
        hou.parmCondType.DisableWhen, "{ derived == 0 }")
    ana.addParmTemplate(derived_mode)
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
    glyph_off = "{ has_glyph_data == 0 } { add_glyphs == 0 }"
    glyph_tensor.setConditional(hou.parmCondType.DisableWhen, glyph_off)
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
        direction.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        direction_folder.addParmTemplate(direction)
    ana.addParmTemplate(direction_folder)
    for template in (
            hou.FloatParmTemplate(
                "tensor_scale", "Principal-Value Length Scale", 1,
                default_value=(0.01,),
                help="Multiplier from principal value to line length (scene "
                     "units per unit of the tensor). Press Auto-Estimate "
                     "Glyph Scale to get a value that fits the mesh; then "
                     "adjust by eye."),
            hou.MenuParmTemplate(
                "glyph_length_mode", "Glyph Length Mode",
                ("value", "normalized", "clamped"),
                menu_labels=("Scale by Principal Value",
                             "Normalized Directions",
                             "Scale by Value with Maximum Length"),
                default_value=0,
                help="How long each glyph line is. Scale by Principal Value: "
                     "length proportional to the magnitude (shows where the "
                     "field is strong). Normalized: all lines the same length "
                     "(shows only direction). Scale by Value with Maximum "
                     "Length: proportional but capped, so a few huge values "
                     "do not hide everything else.")):
        template.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(template)
    glyph_max_length = hou.FloatParmTemplate(
        "glyph_max_length", "Maximum Glyph Length", 1, default_value=(1.0,),
        min=0.0, max=1e9,
        help="Longest line allowed (scene units) when Glyph Length Mode is "
             "Scale by Value with Maximum Length.")
    glyph_max_length.setConditional(
        hou.parmCondType.DisableWhen,
        glyph_off + " { glyph_length_mode != clamped }")
    ana.addParmTemplate(glyph_max_length)
    for name, label, which in (
            ("glyph_scale_first", "First Direction Length Multiplier",
             "largest"),
            ("glyph_scale_second", "Second Direction Length Multiplier",
             "middle"),
            ("glyph_scale_third", "Third Direction Length Multiplier",
             "smallest")):
        gscale = hou.FloatParmTemplate(
            name, label, 1, default_value=(1.0,), min=0.0, max=1e9,
            help=f"Extra length factor for the glyphs of the {which} "
                 "principal value only, e.g. to emphasize one direction "
                 "(1 = no change).")
        gscale.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(gscale)
    glyph_arrowheads = hou.ToggleParmTemplate(
        "glyph_arrowheads", "Show Glyph Arrowheads", default_value=False,
        help="Draw a small arrowhead at both ends of each glyph line so the "
             "direction reads clearly from a distance.")
    glyph_arrowheads.setConditional(hou.parmCondType.DisableWhen, glyph_off)
    ana.addParmTemplate(glyph_arrowheads)
    glyph_arrow_size = hou.FloatParmTemplate(
        "glyph_arrow_size", "Glyph Arrowhead Size", 1,
        default_value=(0.08,), min=0.001, max=0.5,
        help="Size of the arrowheads as a fraction of the glyph length.")
    glyph_arrow_size.setConditional(
        hou.parmCondType.DisableWhen,
        glyph_off + " { glyph_arrowheads == 0 }")
    ana.addParmTemplate(glyph_arrow_size)
    for template in (
            hou.IntParmTemplate(
                "glyph_stride", "Glyph Sampling Stride", 1,
                default_value=(1,), min=1, max=100000,
                help="Draw a glyph only at every Nth point, to thin them out "
                     "on dense meshes (1 = every point)."),
            hou.ToggleParmTemplate(
                "glyph_sign_color", "Stress Sign Colors", default_value=True,
                help="For stress tensors, color glyphs red where the "
                     "principal stress is tension (positive) and blue where "
                     "it is compression (negative). Strain tensors use the "
                     "uniform color."),
            hou.FloatParmTemplate(
                "glyph_color", "Uniform Glyph Color", 3,
                default_value=(1, 1, 1),
                help="Color of the glyph lines when Stress Sign Colors is off "
                     "(or for strain tensors, which have no sign coloring)."),
            hou.ButtonParmTemplate(
                "autoglyph", "Auto-Estimate Glyph Scale",
                script_callback="hou.phm().autoglyph(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Size the glyphs to the mesh: the largest glyphs span "
                     "about half an average edge. For value-scaled glyphs the "
                     "estimate accounts for the selected tensor's "
                     "principal-value magnitude, so it works for stress and "
                     "strain alike.")):
        template.setConditional(hou.parmCondType.DisableWhen, glyph_off)
        ana.addParmTemplate(template)
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
            hou.parmCondType.DisableWhen,
            "{ show_fibers == 0 } { fiber_color_mode != uniform }"
            if template.name() == "fiber_color" else "{ show_fibers == 0 }")
        fiber_folder.addParmTemplate(template)

    autofiber = hou.ButtonParmTemplate(
        "autofiber", "Auto-Estimate Fiber Scale",
        script_callback="hou.phm().autofiber(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Size the fiber lines to the mesh (about half an average edge).")
    autofiber.setConditional(
        hou.parmCondType.DisableWhen, "{ show_fibers == 0 }")
    fiber_folder.addParmTemplate(autofiber)
    fiber_attribs = hou.StringParmTemplate(
        "fiber_attribs", "fiber_attribs", 1,
        default_value=("",),
        help="Internal, hidden: the fiber vector attributes assembled on "
             "load.")
    fiber_attribs.hide(True)
    fiber_folder.addParmTemplate(fiber_attribs)
    ana.addParmTemplate(fiber_folder)
    ana.addParmTemplate(_force_curve_folder())
    return ana
