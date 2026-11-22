"""Legend and Gnomon tab parameter templates."""

import hou


def build_legend_tab():
    legend = hou.FolderParmTemplate("legend_folder", "Legend",
                                    folder_type=hou.folderType.Tabs)
    legend_mode = hou.MenuParmTemplate(
        "legend_mode", "Legend Display Mode",
        ("off", "scene", "overlay", "both"),
        menu_labels=("Off", "Scene Geometry (Renderable, always visible)",
                     "Viewport Overlay (screen-fixed)",
                     "Scene Geometry and Overlay"),
        default_value=0,
        script_callback="hou.phm().overlay_feature_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Scene Geometry is renderable, positioned beside the model, and "
             "always visible regardless of selection. The Viewport Overlay is "
             "screen-fixed but is drawn by this node's viewer state, so it only "
             "shows while that state is active: selecting an overlay mode "
             "activates it automatically (and the 'Show Overlay In Viewport' "
             "button re-activates it), and it persists through viewport "
             "navigation until you select another node or change tools.")
    legend.addParmTemplate(legend_mode)
    overlay_button = hou.ButtonParmTemplate(
        "show_overlay", "Show Overlay In Viewport",
        script_callback="hou.phm().show_overlay(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Activate this node's viewer state so the overlay legend and "
             "click-probe appear without pressing Enter in the viewport.")
    overlay_button.setConditional(
        hou.parmCondType.DisableWhen,
        "{ legend_mode != 2 legend_mode != 3 probe_enabled == 0 }")
    legend.addParmTemplate(overlay_button)
    legend_off = "{ legend_mode == off }"
    legend_and_probe_off = "{ legend_mode == off probe_enabled == 0 }"
    scene_legend_off = "{ legend_mode == off } { legend_mode == overlay }"
    overlay_legend_off = "{ legend_mode == off } { legend_mode == scene }"
    for template, cond in (
            (hou.StringParmTemplate(
                "legend_title", "Custom Legend Title", 1, default_value=("",),
                help="Leave empty to use the resolved color field and "
                     "displayed value."), legend_off),
            (hou.IntParmTemplate(
                "legend_ticks", "Number of Tick Labels", 1, default_value=(6,),
                min=2, max=21,
                help="How many labeled values are written along the color bar "
                     "(including both ends)."), legend_off),
            (hou.IntParmTemplate(
                "legend_digits", "Significant Digits", 1, default_value=(5,),
                min=1, max=12,
                help="Number of significant digits shown in the legend labels "
                     "and probe readout."), legend_and_probe_off),
            (hou.MenuParmTemplate(
                "legend_number_format", "Legend Number Format",
                ("automatic", "scientific", "engineering", "fixed"),
                menu_labels=("Automatic", "Scientific", "Engineering",
                             "Fixed"),
                default_value=0,
                help="How legend and probe numbers are written: Automatic "
                     "picks per value; Scientific uses powers of ten "
                     "(1.2e+05); Engineering uses powers of a thousand "
                     "(120e+03); Fixed uses plain decimals."),
             legend_and_probe_off),
            (hou.StringParmTemplate(
                "field_units", "Displayed Field Units", 1, default_value=("",),
                help="Optional units appended to legend titles and probe "
                     "values."), legend_and_probe_off),
            (hou.FloatParmTemplate(
                "legend_text_color", "Legend Text Color", 3,
                default_value=(1, 1, 1),
                help="Color of the legend title and tick labels (scene legend "
                     "and viewport overlay)."), legend_off)):
        template.setConditional(hou.parmCondType.DisableWhen, cond)
        legend.addParmTemplate(template)
    legend_rotate = hou.FloatParmTemplate(
        "legend_rotate", "Additional Scene Legend Rotation", 3,
        default_value=(0, 0, 0),
        help="Extra rotation (degrees) applied to the scene legend after the "
             "plane/up choice, for fine adjustment.")
    legend_rotate.setLook(hou.parmLook.Angle)
    for template in (
            hou.MenuParmTemplate(
                "legend_orientation", "Scene Legend Plane and Up Direction",
                tuple(str(i) for i in range(12)),
                menu_labels=("XY Plane, +X Up", "XY Plane, -X Up",
                             "XY Plane, +Y Up", "XY Plane, -Y Up",
                             "YZ Plane, +Y Up", "YZ Plane, -Y Up",
                             "YZ Plane, +Z Up", "YZ Plane, -Z Up",
                             "XZ Plane, +X Up", "XZ Plane, -X Up",
                             "XZ Plane, +Z Up", "XZ Plane, -Z Up"),
                default_value=2,
                script_callback="hou.phm().legend_orientation_changed(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Which world plane the scene legend lies in and which "
                     "way its 'up' points. Choose the plane facing your usual "
                     "camera."),
            hou.FloatParmTemplate(
                "legend_translate", "Scene Legend Translation", 3,
                default_value=(0, 0, 0),
                help="Position of the scene legend geometry (scene units). "
                     "Place Scene Legend Beside Model sets it automatically."),
            legend_rotate,
            hou.FloatParmTemplate(
                "legend_scale", "Scene Legend Scale", 1, default_value=(1.0,),
                min=0.0001, max=1000,
                help="Overall size of the scene legend geometry."),
            hou.ButtonParmTemplate(
                "legend_autoplace", "Place Scene Legend Beside Model",
                script_callback="hou.phm().autoplace_legend(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Move and size the scene legend so it sits next to the "
                     "model's bounding box at a readable size.")):
        template.setConditional(hou.parmCondType.DisableWhen, scene_legend_off)
        legend.addParmTemplate(template)
    for template in (
            hou.MenuParmTemplate(
                "overlay_corner", "Overlay Screen Corner",
                ("upper_left", "upper_right", "lower_left", "lower_right"),
                menu_labels=("Upper Left", "Upper Right", "Lower Left",
                             "Lower Right"),
                default_value=1,
                help="Corner of the viewport where the screen-fixed overlay "
                     "legend is drawn."),
            hou.IntParmTemplate(
                "overlay_margin", "Overlay Margin (Pixels)", 1,
                default_value=(24,), min=0, max=1000,
                help="Distance in pixels between the overlay legend and the "
                     "viewport edges."),
            hou.IntParmTemplate(
                "overlay_width", "Overlay Bar Width (Pixels)", 1,
                default_value=(28,), min=8, max=500,
                help="Width of the overlay color bar in pixels."),
            hou.IntParmTemplate(
                "overlay_height", "Overlay Bar Height (Pixels)", 1,
                default_value=(260,), min=40, max=2000,
                help="Height of the overlay color bar in pixels.")):
        template.setConditional(
            hou.parmCondType.DisableWhen, overlay_legend_off)
        legend.addParmTemplate(template)
    return legend


def build_gnomon_tab():
    gnomon = hou.FolderParmTemplate("gnomon_folder", "Gnomon",
                                    folder_type=hou.folderType.Tabs)
    gnomon.addParmTemplate(hou.ToggleParmTemplate(
        "gnomon_show", "Show Scene Gnomon", default_value=False,
        script_callback="hou.phm().gnomon_show_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Adds a renderable XYZ orientation marker to the final geometry. "
             "Automatically scales and positions beside the scene geometry "
             "when first enabled."))
    gnomon_off = "{ gnomon_show == 0 }"
    for template in (
            hou.FloatParmTemplate(
                "gnomon_center", "Gnomon Center", 3, default_value=(0, 0, 0),
                help="World position of the gnomon's origin (scene units). "
                     "Place Gnomon Beside Model sets it automatically."),
            hou.FloatParmTemplate(
                "gnomon_scale", "Gnomon Axis Length", 1, default_value=(1.0,),
                min=0.0001, max=1000,
                help="Length of each gnomon axis line (scene units)."),
            hou.ButtonParmTemplate(
                "gnomon_autoplace", "Place Gnomon Beside Model",
                script_callback="hou.phm().autoplace_gnomon(kwargs)",
                script_callback_language=hou.scriptLanguage.Python,
                help="Move and size the gnomon so it sits cleanly beside the "
                     "scene geometry at a compact, readable scale.")):
        template.setConditional(hou.parmCondType.DisableWhen, gnomon_off)
        gnomon.addParmTemplate(template)
    for name, label, what in (
            ("gnomon_x", "Show X Axis", "Draw the red X axis line."),
            ("gnomon_y", "Show Y Axis", "Draw the green Y axis line."),
            ("gnomon_z", "Show Z Axis", "Draw the blue Z axis line."),
            ("gnomon_arrows", "Show Arrowheads",
             "Draw arrowheads at the tip of each axis."),
            ("gnomon_center_marker", "Show Center Marker",
             "Draw a small marker at the gnomon origin.")):
        g_toggle = hou.ToggleParmTemplate(
            name, label, default_value=True, help=what)
        g_toggle.setConditional(hou.parmCondType.DisableWhen, gnomon_off)
        gnomon.addParmTemplate(g_toggle)
    return gnomon
