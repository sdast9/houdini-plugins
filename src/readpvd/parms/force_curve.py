"""Force Curves folder for the Analysis tab."""

import hou

from .helpers import READ_ONLY


def _force_curve_folder():
    """Analysis > Force Curves (review F4): reaction and contact force per
    sideset or obstacle, from the nodal forces PolyFEM writes."""
    folder = hou.FolderParmTemplate(
        "force_folder", "Force Curves", folder_type=hou.folderType.Simple)
    sets = hou.StringParmTemplate(
        "force_sets", "Force Sets", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which sidesets and obstacles to measure (empty = all of them). "
             "They come from the PolyFEM node that wrote the run: every "
             "sideset, with or without a condition (a sideset without a "
             "condition can be made just to measure a force), and every "
             "obstacle.")
    sets.setItemGeneratorScript("hou.phm().force_set_menu(kwargs)")
    sets.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(sets)
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_compute", "Compute Force Curves",
        script_callback="hou.phm().compute_force_curves(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Read every time step and write force_curves.csv next to the "
             "PVD file: per step and per set, the reaction (the force the "
             "prescribed displacement, or an obstacle's prescribed motion, "
             "applies there: minus the sum of every nodal force PolyFEM "
             "writes), the contact force (contact + friction) and the mean "
             "displacement, as x, y, z and magnitude. Needs Output > Nodal "
             "Forces on the PolyFEM node. Reads every result file, like Auto "
             "Range: All Frames. Adhesion and Rayleigh-damping forces are not "
             "written by PolyFEM and are not included."))
    readout = hou.StringParmTemplate(
        "force_readout", "At This Frame", 1,
        default_expression=(
            "hou.pwd().hdaModule().force_readout_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        tags={"editor": "1", "editorlines": "2-6"},
        help="Read-only: each force set's reaction, contact force and mean "
             "displacement at the step on screen. For a fixed or moved "
             "sideset the reaction is the support force; for an obstacle "
             "the contact force is the force the body applies to it.")
    readout.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    folder.addParmTemplate(readout)
    chart_set = hou.StringParmTemplate(
        "force_chart_set", "Chart", 1, default_value=("1",),
        menu_type=hou.menuType.Normal,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which force set the chart shows.")
    chart_set.setItemGeneratorScript("hou.phm().force_set_menu(kwargs)")
    chart_set.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    folder.addParmTemplate(chart_set)
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_quantity", "Force", ("reaction", "contact"),
        menu_labels=("Reaction (support force)", "Contact Force"),
        default_value=0,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Reaction: the force the prescribed displacement (or the "
             "obstacle's prescribed motion) applies to the body there. "
             "Contact Force: contact plus friction forces on those nodes."))
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_component", "Component", ("x", "y", "z", "magnitude"),
        menu_labels=("X", "Y", "Z", "Magnitude"), default_value=3,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which component of the force (and of the displacement it is "
             "plotted against) the chart shows."))
    folder.addParmTemplate(hou.MenuParmTemplate(
        "force_chart_against", "Against", ("displacement", "time"),
        menu_labels=("Mean Displacement", "Time"), default_value=0,
        script_callback="hou.phm().force_chart_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Plot the force against the set's mean displacement (same "
             "component: a force-displacement curve) or against simulation "
             "time."))
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_show_chart", "Show Chart",
        script_callback="hou.phm().show_force_chart(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        join_with_next=True,
        help="Open a chart of the chosen curve in its own window; a dot "
             "marks the step on screen and follows the playbar. Computes the "
             "curves first if needed."))
    folder.addParmTemplate(hou.ButtonParmTemplate(
        "force_open_csv", "Open CSV",
        script_callback="hou.phm().open_force_csv(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Open force_curves.csv with the system's default program "
             "(computing it first if needed)."))
    return folder
