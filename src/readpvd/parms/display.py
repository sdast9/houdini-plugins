"""Display tab parameter templates."""

import hou

from .helpers import READ_ONLY


def build_display_tab():
    disp = hou.FolderParmTemplate("display_folder", "Display",
                                  folder_type=hou.folderType.Tabs)
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "surface_only", "Display Boundary Surface Only", default_value=True,
        help="Show only the outer surface of the volume mesh instead of "
             "every solid element. Much faster to display for large meshes "
             "and looks the same from outside. Clipping Plane and Thin Slice "
             "automatically reveal interior faces while active; turn this "
             "off only when you need full solid tetrahedra/hexahedra "
             "downstream or unclipped interior access."))
    show_deformed = hou.ToggleParmTemplate(
        "show_deformed", "Show Deformed", default_value=True,
        help="Move the mesh by the computed displacement so you see the "
             "deformed shape. Off shows the undeformed (rest) mesh with the "
             "fields painted on it. The rest positions are always kept in "
             "the 'rest' point attribute.")
    show_deformed.setConditional(
        hou.parmCondType.DisableWhen, "{ has_solution_data == 0 }")
    disp.addParmTemplate(show_deformed)
    visible_bodies = hou.StringParmTemplate(
        "visible_bodies", "Visible Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        script_callback="hou.phm().analysis_options_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Pick which bodies to show (by PolyFEM body id); leave empty "
             "to show all of them. Handy to hide an obstacle or isolate one "
             "part. The color range, probe, glyphs and clipping only consider "
             "the visible bodies. Needs Body IDs exported from the PolyFEM "
             "node and more than one body in the result.")
    visible_bodies.setItemGeneratorScript("hou.phm().body_id_menu(kwargs)")
    visible_bodies.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    visible_bodies.setConditional(
        hou.parmCondType.DisableWhen, "{ has_multibody == 0 }")
    disp.addParmTemplate(visible_bodies)
    color_attrib = hou.StringParmTemplate(
        "color_attrib", "Color Field", 1, default_value=("von_mises",),
        menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().color_selection_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which result field colors the mesh (displacement, stress, "
             "strain, contact force ...). Only fields that exist in the "
             "loaded block are listed; derived fields appear when the data "
             "they need was exported.")
    color_attrib.setItemGeneratorScript(
        "hou.phm().color_field_menu(kwargs)")
    color_attrib.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    disp.addParmTemplate(color_attrib)
    color_reduction = hou.StringParmTemplate(
        "color_reduction", "Field Value To Display", 1,
        default_value=("auto",), menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().color_selection_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="A color can only show one number per point, so vector and "
             "tensor fields must be reduced: magnitude or a component for "
             "vectors; von Mises-style norm, a diagonal entry, a principal "
             "value (principal values are ordered maximum to minimum; for a "
             "non-symmetric tensor such as the deformation gradient or the "
             "1st Piola-Kirchhoff stress, those of its symmetric part), trace "
             "or determinant for tensors. "
             "Only reductions that make sense for the chosen field are "
             "listed.")
    color_reduction.setItemGeneratorScript(
        "hou.phm().color_reduction_menu(kwargs)")
    color_reduction.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    disp.addParmTemplate(color_reduction)
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "smooth_field", "Smooth Field Across Elements", default_value=False,
        help="PolyFEM writes a discontinuous mesh, so element-wise quantities "
             "(stress, strain) show one flat color per element. This nodally "
             "averages the displayed value onto shared vertices for a "
             "continuous, interpolated field. PolyFEM's own continuous '_avg' "
             "fields (e.g. von_mises_avg) are preferred automatically when "
             "they exist; other fields use a generic nodal average. Linear "
             "elements have no true sub-element variation -- raise PolyFEM's "
             "vismesh sampling or element order for that."))
    color_status = hou.StringParmTemplate(
        "color_status", "Color Field Status", 1,
        default_value=("Cook the result to validate the selected field.",),
        help="Read-only: whether the chosen Color Field was found in the "
             "current frame, and the reduction that is actually displayed.")
    color_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    disp.addParmTemplate(color_status)
    missing = hou.FloatParmTemplate(
        "missing_color", "Unavailable Field Color", 3,
        default_value=(0.35, 0.35, 0.35),
        help="Flat color painted on the mesh when the selected Color Field "
             "does not exist in the current frame (for example a field that "
             "PolyFEM only wrote for some steps).")
    disp.addParmTemplate(missing)
    invalid = hou.FloatParmTemplate(
        "invalid_color", "Invalid / NaN Value Color", 3,
        default_value=(1.0, 0.0, 1.0),
        help="Color used for points whose value is not a number (NaN), "
             "infinite, or not allowed by the color scale (e.g. zero or "
             "negative with a logarithmic scale), so they stand out instead "
             "of being hidden.")
    disp.addParmTemplate(invalid)
    disp.addParmTemplate(hou.MenuParmTemplate(
        "color_scale", "Color Scale", ("linear", "log10"),
        menu_labels=("Linear", "Logarithmic (Base 10)"), default_value=0,
        script_callback="hou.phm().color_scale_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Linear maps values evenly between the displayed minimum and "
             "maximum. Logarithmic spreads the colors by powers of ten, which "
             "helps when values span several orders of magnitude (stress "
             "concentrations). Log needs positive values: zero, negative, "
             "NaN and infinite values get the Invalid Value Color."))
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "range_lock", "Lock Displayed Range", default_value=False,
        help="Freeze the displayed minimum and maximum so the Auto Range "
             "buttons and frame changes cannot alter them. Use it to compare "
             "frames or runs with an identical color scale."))
    disp.addParmTemplate(hou.ToggleParmTemplate(
        "range_percentile", "Ignore Range Outliers", default_value=False,
        help="When auto-ranging, ignore the most extreme values and use "
             "percentile limits instead of the absolute minimum and maximum, "
             "so a few outliers (a single hot element) do not wash out the "
             "colors everywhere else."))
    range_percentiles = hou.FloatParmTemplate(
        "range_percentiles", "Range Percentiles", 2,
        default_value=(1.0, 99.0), min=0.0, max=100.0,
        help="Lower and upper percentiles used by Ignore Range Outliers: "
             "1 and 99 map the color range to the middle 98% of the values.")
    range_percentiles.setConditional(
        hou.parmCondType.DisableWhen, "{ range_percentile == 0 }")
    disp.addParmTemplate(range_percentiles)
    range_symmetric = hou.ToggleParmTemplate(
        "range_symmetric", "Symmetric Range Around Zero", default_value=False,
        help="Make the displayed range symmetric around zero (e.g. -5 to +5) "
             "so that a diverging ramp puts zero exactly in the middle. "
             "Useful for signed fields such as principal stresses.")
    range_symmetric.setConditional(
        hou.parmCondType.DisableWhen, "{ color_scale == log10 }")
    disp.addParmTemplate(range_symmetric)
    color_min = hou.FloatParmTemplate(
        "color_min", "Displayed Minimum", 1, default_value=(0.0,),
        help="Value mapped to the left end of the color ramp. Values below "
             "it are clamped to that color. Set by the Auto Range buttons or "
             "type your own.")
    color_min.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(color_min)
    color_max = hou.FloatParmTemplate(
        "color_max", "Displayed Maximum", 1, default_value=(1.0,),
        help="Value mapped to the right end of the color ramp. Values above "
             "it are clamped to that color.")
    color_max.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(color_max)
    autoscale_btn = hou.ButtonParmTemplate(
        "autoscale", "Auto Range: Current Frame",
        script_callback="hou.phm().autoscale(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Set Displayed Minimum/Maximum from the values in the frame "
             "currently shown (instant). Respects Ignore Range Outliers and "
             "Symmetric Range; blocked by Lock Displayed Range.")
    autoscale_btn.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(autoscale_btn)
    autoscale_all_btn = hou.ButtonParmTemplate(
        "autoscale_all", "Auto Range: All Frames (scans sequence)",
        script_callback="hou.phm().autoscale_all(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Scans the selected field over the complete PVD sequence to find "
             "the global range. This must read every frame once, so it can "
             "take a while for long sequences of large meshes -- unavoidable, "
             "the values live inside each file. Use Auto Range: Current Frame "
             "for an instant range from the displayed frame.")
    autoscale_all_btn.setConditional(
        hou.parmCondType.DisableWhen, "{ range_lock == 1 }")
    disp.addParmTemplate(autoscale_all_btn)
    ramp = hou.RampParmTemplate(
        "color_ramp", "Displayed Color Ramp", hou.rampParmType.Color,
        help="Colors used from Displayed Minimum (left) to Displayed Maximum "
             "(right). Edit it like any Houdini color ramp, or use the "
             "Set Signed Diverging Ramp button for blue-white-red.")
    disp.addParmTemplate(ramp)
    disp.addParmTemplate(hou.ButtonParmTemplate(
        "set_diverging_ramp", "Set Signed Diverging Ramp",
        script_callback="hou.phm().set_diverging_ramp(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Replace the ramp with blue-white-red, the usual choice for "
             "signed values (negative = blue, zero = white, positive = red). "
             "Pair it with Symmetric Range Around Zero."))
    return disp
