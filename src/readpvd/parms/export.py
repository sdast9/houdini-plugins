"""Export tab parameter templates."""

import hou

from .helpers import _callback, _menu_script, _read_only_text, _status_text


def _export_tab():
    """Export (2026-10-03): the deformed mesh as a .msh, and data over time
    as an Excel workbook for presentation plots."""
    tab = hou.FolderParmTemplate("export_folder_tab", "Export",
                                 folder_type=hou.folderType.Tabs)
    tab.addParmTemplate(hou.StringParmTemplate(
        "export_folder", "Export Folder", 1, default_value=("",),
        string_type=hou.stringParmType.FileReference,
        file_type=hou.fileType.Directory,
        help="Where both exports are written. Empty (the default) = the "
             "run's 'exports' folder, next to its 'input' and 'output' "
             "folders: outside 'output', so running the simulation again "
             "into the same folder leaves the exports alone."))
    tab.addParmTemplate(hou.ButtonParmTemplate(
        "export_open_folder", "Open Folder", **_callback("open_export_folder"),
        help="Show the export folder in Finder / Explorer (it is created if "
             "needed)."))

    # ---- Deformed Mesh ------------------------------------------------------
    mesh = hou.FolderParmTemplate("export_mesh_folder", "Deformed Mesh",
                                  folder_type=hou.folderType.Simple)
    mesh.addParmTemplate(hou.MenuParmTemplate(
        "export_mesh_step_mode", "Step", ("screen", "number"),
        menu_labels=("Step on Screen", "Step Number"), default_value=0,
        help="Which output step's shape to export: the one the playbar shows "
             "now, or a step you type."))
    step = hou.IntParmTemplate(
        "export_mesh_step", "Step Number", 1, default_value=(0,), min=0,
        max=1000,
        help="The output step to export (0 = the start of the run, before "
             "any load). The Simulation Time line on the Main tab shows the "
             "step on screen.")
    step.setConditional(hou.parmCondType.HideWhen,
                        "{ export_mesh_step_mode != number }")
    mesh.addParmTemplate(step)
    mesh.addParmTemplate(_read_only_text(
        "export_mesh_step_status", "Exported Step",
        "hou.pwd().hdaModule().export_mesh_step_text(hou.pwd())",
        "Read-only: the step Export Deformed Mesh writes, and its "
        "simulation time."))
    geometries = hou.StringParmTemplate(
        "export_mesh_geometries", "Geometries", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Which of the run's simulated geometries to export (one .msh "
             "each); empty = all of them. Obstacles are not exported: they "
             "keep their own files.")
    mesh.addParmTemplate(_menu_script(geometries, "export_geometry_menu"))
    mesh.addParmTemplate(hou.MenuParmTemplate(
        "export_mesh_coordinates", "Coordinates", ("mesh", "simulation"),
        menu_labels=("Mesh File (keep the geometry's Transform)",
                     "Simulation (world)"), default_value=0,
        help="Mesh File: the coordinates and units of the original mesh "
             "file; give it the same Transform on the PolyFEM node (as when "
             "you pick it on the geometry that made the run) and it sits "
             "where the deformed body was. Simulation: the positions in the "
             "simulation, with the Transform already applied (the file name "
             "ends in _simulation); load it with no Transform."))
    mesh.addParmTemplate(hou.ButtonParmTemplate(
        "export_mesh", "Export Deformed Mesh", **_callback("export_mesh_button"),
        help="Write each geometry's deformed shape at the chosen step as a "
             ".msh file in the export folder (<mesh name>_step<N>.msh): the "
             "run's own mesh file with only its node positions moved, so "
             "node numbers, elements, Gmsh physical groups and names (the "
             "subdomains) stay the same, and the PolyFEM node keeps a "
             "geometry's materials and picked sidesets when you load the "
             "file on it. Linear elements: for a P2 or higher run the "
             "corners are moved and the curvature of the edges, which the "
             "report measures, is not kept. Inverted elements are refused. "
             "The exported shape is a new rest shape: a simulation that "
             "starts from it starts without stress, strain or velocity."))
    mesh.addParmTemplate(_status_text(
        "export_mesh_report", "Report",
        "Read-only: what the last Export Deformed Mesh wrote (files, nodes "
        "moved, the range of element volume change, the curvature a "
        "higher-order run loses) or why it wrote nothing."))
    tab.addParmTemplate(mesh)

    # ---- Data Over Time --------------------------------------------------------
    data = hou.FolderParmTemplate("export_data_folder", "Data Over Time",
                                  folder_type=hou.folderType.Simple)
    regions = hou.FolderParmTemplate(
        "export_regions", "Regions", folder_type=hou.folderType.MultiparmBlock,
        default_value=1)
    kind_condition = "{{ export_region_kind# != {} }}"
    regions.addParmTemplate(hou.StringParmTemplate(
        "export_region_name#", "Name", 1, default_value=("",),
        help="The region's name in the column headers (e.g. 'Cartilage'); "
             "empty = a name from its kind. Each region (Regions above sets "
             "how many) gives the spreadsheet its own columns; with no region "
             "the whole model is measured."))
    kind = hou.MenuParmTemplate(
        "export_region_kind#", "Region",
        ("all", "bodies", "sideset", "box", "sphere", "plane", "element",
         "node"),
        menu_labels=("Whole Model", "Bodies / Subdomains", "Sideset", "Box",
                     "Sphere", "Half Space (side of a plane)", "One Element",
                     "One Node"),
        default_value=0,
        help="Whole Model: every element and node. Bodies: chosen bodies "
             "(the PolyFEM node's subdomains, named after their Gmsh groups). "
             "Sideset: a sideset of the PolyFEM node (its nodes, and the "
             "elements with a face on it). Box, Sphere, Half Space: the "
             "elements whose centre and the nodes that lie inside, in the "
             "undeformed shape (simulation coordinates). One Element / One "
             "Node: by number (Set From Probe fills it in from a point "
             "clicked with the Probe tab's probe).")
    regions.addParmTemplate(kind)
    bodies = hou.StringParmTemplate(
        "export_region_bodies#", "Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="The bodies of the region (needs Body IDs in the PolyFEM node's "
             "output, on by default).")
    bodies.setConditional(hou.parmCondType.HideWhen,
                          kind_condition.format("bodies"))
    regions.addParmTemplate(_menu_script(bodies, "export_body_menu"))
    sideset = hou.StringParmTemplate(
        "export_region_sideset#", "Sideset", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        help="One of the sidesets the PolyFEM node recorded for the run "
             "(input/hda_scene.json, written since 2026-10-02). A sideset "
             "without a condition can be made just to measure there.")
    sideset.setConditional(hou.parmCondType.HideWhen,
                           kind_condition.format("sideset"))
    regions.addParmTemplate(_menu_script(sideset, "export_sideset_menu"))
    for name, label, default, what, help_text in (
            ("box_min", "Box Minimum", (0.0, 0.0, 0.0), "box",
             "One corner of the box (simulation coordinates, undeformed)."),
            ("box_max", "Box Maximum", (1.0, 1.0, 1.0), "box",
             "The opposite corner of the box."),
            ("center", "Center", (0.0, 0.0, 0.0), "sphere",
             "The sphere's centre (simulation coordinates, undeformed); Set "
             "From Probe puts it on the probed point."),
            ("point", "Point on Plane", (0.0, 0.0, 0.0), "plane",
             "A point of the plane (simulation coordinates, undeformed)."),
            ("normal", "Normal", (0.0, 0.0, 1.0), "plane",
             "The region is the side of the plane this normal points to.")):
        template = hou.FloatParmTemplate(
            f"export_region_{name}#", label, 3, default_value=default,
            help=help_text)
        template.setConditional(hou.parmCondType.HideWhen,
                                kind_condition.format(what))
        regions.addParmTemplate(template)
    radius = hou.FloatParmTemplate(
        "export_region_radius#", "Radius", 1, default_value=(0.1,), min=0.0,
        max=10.0, help="The sphere's radius (simulation units).")
    radius.setConditional(hou.parmCondType.HideWhen,
                          kind_condition.format("sphere"))
    regions.addParmTemplate(radius)
    one = ("{ export_region_kind# != element export_region_kind# != node }")
    geometry = hou.IntParmTemplate(
        "export_region_geometry#", "Geometry", 1, default_value=(1,), min=1,
        max=20,
        help="The geometry (as numbered on the PolyFEM node) of the element "
             "or node.")
    geometry.setConditional(hou.parmCondType.HideWhen, one)
    regions.addParmTemplate(geometry)
    number = hou.IntParmTemplate(
        "export_region_number#", "Number", 1, default_value=(1,), min=1,
        max=100000,
        help="Element: its number in the geometry's .msh file (1 = the "
             "file's first volume element). Node: its node tag in the .msh "
             "file. Set From Probe fills it in from the probed point.")
    number.setConditional(hou.parmCondType.HideWhen, one)
    regions.addParmTemplate(number)
    probe = hou.ButtonParmTemplate(
        "export_region_probe#", "Set From Probe",
        **_callback("export_region_from_probe"),
        help="Use the point last clicked with the probe (Probe tab): its "
             "node or element for One Node / One Element, its undeformed "
             "position as the centre of a Sphere, or as the point of a "
             "Half Space.")
    probe.setConditional(
        hou.parmCondType.HideWhen,
        "{ export_region_kind# == all } { export_region_kind# == bodies } "
        "{ export_region_kind# == sideset } { export_region_kind# == box }")
    regions.addParmTemplate(probe)
    only = hou.StringParmTemplate(
        "export_region_only#", "Only in Bodies", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Keep only the elements and nodes of these bodies (empty = any "
             "body), e.g. the part of a box that is cartilage.")
    only.setConditional(
        hou.parmCondType.HideWhen,
        "{ export_region_kind# == bodies } { export_region_kind# == element }"
        " { export_region_kind# == node }")
    regions.addParmTemplate(_menu_script(only, "export_body_menu"))
    data.addParmTemplate(regions)

    quantities = hou.FolderParmTemplate(
        "export_quantities", "Quantities",
        folder_type=hou.folderType.MultiparmBlock, default_value=1)
    field = hou.StringParmTemplate(
        "export_q_field#", "Field", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        **_callback("export_quantity_changed"),
        help="What to measure in every region (Quantities above sets how "
             "many; each quantity's statistics become columns, one per "
             "region). Any field Read PVD can color by (exported or derived: "
             "displacement, von Mises, principal stresses, strains, "
             "stretches, J ...), Region Volume, or a fiber stretch on fiber "
             "runs. Same as the Display Tab follows the Display tab's Color "
             "Field and Field Value To Display at export time.")
    quantities.addParmTemplate(_menu_script(field, "export_field_menu"))
    value = hou.StringParmTemplate(
        "export_q_value#", "Value", 1, default_value=("",),
        menu_type=hou.menuType.StringReplace,
        help="One number per element or node: a vector's magnitude or a "
             "component; a tensor's principal value, an entry (xx ... xz), "
             "Frobenius norm, trace or determinant.")
    quantities.addParmTemplate(_menu_script(value, "export_value_menu"))
    quantities.addParmTemplate(hou.MenuParmTemplate(
        "export_q_basis#", "Per", ("auto", "elements", "nodes"),
        menu_labels=("Automatic", "Element", "Node"), default_value=0,
        help="What one value belongs to. Element: the element's value (the "
             "mean of its nodes in PolyFEM's output, exact for linear "
             "elements); Node: the node's value (the mean of its copies, one "
             "per element, as Smooth Field shows). Automatic: nodes for "
             "displacement, velocity, acceleration and nodal forces, "
             "elements otherwise (stress, strain)."))
    for name, label, default, help_text in (
            ("max", "Max", True, "Largest value in the region."),
            ("min", "Min", False, "Smallest value in the region."),
            ("mean", "Mean (volume-weighted)", True,
             "Volume-weighted mean: sum of value x volume / sum of volumes, "
             "so small elements do not count more than large ones."),
            ("amean", "Arithmetic Mean", False,
             "Plain mean over the elements (or nodes), whatever their size."),
            ("integral", "Integral", False,
             "Sum of value x volume over the region (the volume integral)."),
            ("sum", "Sum", False,
             "Plain sum over the elements or nodes; over the nodes of a "
             "region, a nodal force's sum is the resultant force."),
            ("rms", "RMS", False, "Root mean square."),
            ("std", "Standard Deviation", False,
             "Standard deviation over the elements (or nodes)."),
            ("count", "Count", False,
             "How many elements (or nodes) had a value at the step.")):
        quantities.addParmTemplate(hou.ToggleParmTemplate(
            f"export_q_{name}#", label, default_value=default,
            join_with_next=name not in ("amean", "count"), help=help_text))
    quantities.addParmTemplate(hou.StringParmTemplate(
        "export_q_percentiles#", "Percentiles", 1, default_value=("",),
        help="Percentiles to add, e.g. '95 99' (a robust peak: the largest "
             "element values at corners and contact edges depend on the "
             "mesh)."))
    quantities.addParmTemplate(hou.ToggleParmTemplate(
        "export_q_where#", "Where the Max Is", default_value=False,
        help="Add columns with the element or node that holds the max at "
             "each step and its undeformed position."))
    quantities.addParmTemplate(hou.MenuParmTemplate(
        "export_q_volume#", "Volumes", ("current", "rest"),
        menu_labels=("Current (deformed)", "Rest (undeformed)"),
        default_value=0,
        help="The volumes used for the volume-weighted mean, the integral "
             "and Region Volume. Current: each element's deformed volume at "
             "that step, right for quantities per deformed volume such as "
             "the Cauchy stress. Rest: the undeformed volume, right for "
             "quantities per undeformed volume such as Piola-Kirchhoff "
             "stresses, Green-Lagrange strain and strain energy density. "
             "They differ by the volume ratio J."))
    quantities.addParmTemplate(hou.StringParmTemplate(
        "export_q_label#", "Column Label", 1, default_value=("",),
        help="Name of the quantity in the column headers (e.g. 'Stress'); "
             "empty = a name from the field and value."))
    data.addParmTemplate(quantities)

    data.addParmTemplate(hou.ToggleParmTemplate(
        "export_forces", "Add Force Curves", default_value=True,
        help="Add the Force Curves (Analysis tab) as columns of the same "
             "table: per sideset and obstacle, the reaction, the contact "
             "force and the mean displacement, so force against "
             "displacement plots from one table. Needs a run written with "
             "Nodal Forces on (the default); otherwise the export notes it "
             "and goes on."))
    force_sets = hou.StringParmTemplate(
        "export_force_sets", "Force Sets", 1, default_value=("",),
        menu_type=hou.menuType.StringToggle,
        help="Which sets to add (empty = all).")
    force_sets.setConditional(hou.parmCondType.DisableWhen,
                              "{ export_forces == 0 }")
    data.addParmTemplate(_menu_script(force_sets, "force_set_menu"))
    for name, label, default, join, help_text in (
            ("reaction", "Reaction", True, True,
             "The force the prescribed displacement (or an obstacle's motion) "
             "applies to the body there."),
            ("contact", "Contact Force", False, True,
             "Contact plus friction forces on the set's nodes."),
            ("displacement", "Mean Displacement", True, False,
             "The mean displacement of the set's nodes."),
            ("x", "X", True, True, "The x component."),
            ("y", "Y", True, True, "The y component."),
            ("z", "Z", True, True, "The z component."),
            ("magnitude", "Magnitude", True, False, "The magnitude.")):
        template = hou.ToggleParmTemplate(
            f"export_force_{name}", label, default_value=default,
            join_with_next=join, help=help_text)
        template.setConditional(hou.parmCondType.DisableWhen,
                                "{ export_forces == 0 }")
        data.addParmTemplate(template)

    data.addParmTemplate(hou.MenuParmTemplate(
        "export_units", "Run Units",
        ("auto", "none", "m_kg_s", "mm_t_s", "mm_kg_s", "mm_g_s"),
        menu_labels=("Automatic (as the PolyFEM node recorded)",
                     "Not Set (no unit labels)", "m, kg, s (N, Pa)",
                     "mm, tonne, s (N, MPa)", "mm, kg, s (mN, kPa)",
                     "mm, g, s (uN, Pa)"),
        default_value=0,
        help="The unit system the run's numbers are in. Automatic: the "
             "units the PolyFEM node recorded (its Units are on); a run "
             "without them gets no unit labels until you choose its system "
             "here -- PolyFEM itself does not know whether you entered "
             "millimetres or metres."))
    data.addParmTemplate(_read_only_text(
        "export_units_status", "Units",
        "hou.pwd().hdaModule().export_units_text(hou.pwd())",
        "Read-only: the run's unit system as the export uses it, and where "
        "it comes from."))
    for name, label, tokens, labels in (
            ("length", "Length", ("run", "m", "cm", "mm", "um"),
             ("As the Run", "m", "cm", "mm", "µm")),
            ("stress", "Stress", ("run", "Pa", "kPa", "MPa", "GPa"),
             ("As the Run", "Pa", "kPa", "MPa", "GPa")),
            ("force", "Force", ("run", "N", "mN", "uN", "kN"),
             ("As the Run", "N", "mN", "µN", "kN")),
            ("time", "Time", ("run", "s", "ms"), ("As the Run", "s", "ms"))):
        template = hou.MenuParmTemplate(
            f"export_unit_{name}", f"{label} in", tokens, menu_labels=labels,
            default_value=0, join_with_next=name in ("length", "force"),
            help=f"The unit of every {name.lower()} in the spreadsheet "
                 "(volumes, velocities and integrals follow); values are "
                 "converted. Needs the run's unit system.")
        template.setConditional(hou.parmCondType.DisableWhen,
                                "{ export_units == none }")
        data.addParmTemplate(template)

    for name, label, default, minimum, help_text in (
            ("first", "First Step", 0, 0,
             "The first output step in the table (0 = the start of the "
             "run)."),
            ("last", "Last Step", -1, -1,
             "The last output step in the table; -1 = the run's last step."),
            ("every", "Every", 1, 1,
             "Take every Nth output step (1 = every step).")):
        data.addParmTemplate(hou.IntParmTemplate(
            f"export_step_{name}", label, 1, default_value=(default,),
            min=minimum, max=1000, join_with_next=name != "every",
            help=help_text))
    data.addParmTemplate(hou.ToggleParmTemplate(
        "export_per_item", "Values per Element / Node", default_value=False,
        help="Also write, per quantity, a sheet with the value of every "
             "element (or node) of every region and its undeformed position "
             "-- for profiles (value against x) and histograms. Excel holds "
             "about a million rows."))
    item_steps = hou.MenuParmTemplate(
        "export_per_item_steps", "At", ("last", "screen", "all"),
        menu_labels=("Last Exported Step", "Step on Screen",
                     "Every Exported Step"), default_value=0,
        help="The steps of the per element / node sheets: one column each.")
    item_steps.setConditional(hou.parmCondType.DisableWhen,
                              "{ export_per_item == 0 }")
    data.addParmTemplate(item_steps)
    data.addParmTemplate(hou.StringParmTemplate(
        "export_data_file", "Spreadsheet", 1, default_value=("",),
        string_type=hou.stringParmType.FileReference,
        tags={"filechooser_pattern": "*.xlsx", "filechooser_mode": "write"},
        help="The Excel file to write; empty = <export folder>/<PVD "
             "name>_data.xlsx. An existing file is replaced."))
    data.addParmTemplate(hou.ButtonParmTemplate(
        "export_data", "Export Spreadsheet", **_callback("export_data_button"),
        join_with_next=True,
        help="Read every chosen step (like Auto Range: All Frames) and write "
             "the workbook: a Data sheet with one row per step (Step, Time, "
             "Frame) and one column per region, quantity and statistic, "
             "ready to plot (numbers only, units in the headers, no charts); "
             "the optional per element / node sheets; and an About sheet "
             "(the run, its id and input hash, the regions, the units and "
             "what every column means)."))
    data.addParmTemplate(hou.ButtonParmTemplate(
        "export_data_open", "Open Spreadsheet",
        **_callback("open_export_data"),
        help="Open the spreadsheet with the system's default program "
             "(exporting it first when it does not exist yet)."))
    data.addParmTemplate(_status_text(
        "export_data_status", "Status",
        "Read-only: what the last export wrote (file, steps, columns, "
        "elements and nodes per region) or why it wrote nothing."))
    tab.addParmTemplate(data)
    return tab
