"""Main tab parameter templates."""

import hou

from .helpers import READ_ONLY, TIME_MAPPING_TOKENS, _read_only_text


def build_main_tab():
    main = hou.FolderParmTemplate("main_folder", "Main",
                                  folder_type=hou.folderType.Tabs)
    main.addParmTemplate(hou.StringParmTemplate(
        "PVD_file", "PVD File", 1,
        string_type=hou.stringParmType.FileReference,
        tags={"filechooser_pattern": "*.pvd"},
        script_callback="hou.phm().start(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="The .pvd file PolyFEM wrote in its output folder (sim.pvd by "
             "default). It lists one result file per saved time step; those "
             "steps are mapped to Houdini frames when you press Refresh / Set "
             "Playbar."))
    main.addParmTemplate(hou.ButtonParmTemplate(
        "refresh", "Refresh / Set Playbar",
        script_callback="hou.phm().refresh(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        join_with_next=True,
        help="Re-read the PVD file (e.g. while a simulation is still writing "
             "steps), rebuild the field menus, and optionally set the playbar "
             "range to the available time steps."))
    main.addParmTemplate(hou.ToggleParmTemplate(
        "update_playbar", "Set Playbar Range on Refresh", default_value=True,
        help="Update Houdini's playbar range when loading a PVD or pressing "
             "Refresh. Turn off when you have several Read PVD nodes or a "
             "PolyFEM animation timeline you do not want overwritten."))
    main.addParmTemplate(hou.ToggleParmTemplate(
        "follow_running", "Follow Running Simulation", default_value=False,
        join_with_next=True,
        script_callback="hou.phm().toggle_follow_running(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Periodically check the PVD file for new output steps while "
             "PolyFEM is running, refresh the display, and jump to the "
             "latest step. Turns itself off automatically when "
             "run-manifest.json reports that the run has finished."))
    follow_interval = hou.FloatParmTemplate(
        "follow_interval", "Poll Interval (s)", 1, default_value=(3.0,),
        min=0.5, max=60.0, min_is_strict=True,
        help="Seconds between checks while Follow Running Simulation is on.")
    follow_interval.setConditional(
        hou.parmCondType.DisableWhen, "{ follow_running == 0 }")
    main.addParmTemplate(follow_interval)
    main.addParmTemplate(hou.MenuParmTemplate(
        "time_mapping", "Time Mapping", TIME_MAPPING_TOKENS,
        menu_labels=("Automatic", "One Frame per Output Step",
                     "Simulation Time"),
        default_value=1,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which time step each Houdini frame shows. One Frame per Output "
             "Step (the default): frame k shows step k, so the playbar has "
             "exactly one frame per solved output step (0..N-1). Automatic: "
             "Simulation Time with the PolyFEM node's own Start Frame and "
             "Time Scale when the run was written by it (input/hda_scene.json), "
             "one frame per output step otherwise. Simulation Time: Houdini "
             "time is simulation time (simulation time = Start Time + "
             "(frame's time - Start Frame's time) x Time Scale), so the "
             "results play back in step with keyframed loads and obstacle "
             "motions; a frame shows the last step at or before its "
             "simulation time."))
    start_frame = hou.FloatParmTemplate(
        "time_start_frame", "Start Frame", 1, default_value=(1.0,),
        min=0.0, max=240.0,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Simulation Time mapping: the Houdini frame at which the run's "
             "first step (its start time) is shown.")
    start_frame.setConditional(hou.parmCondType.HideWhen,
                               "{ time_mapping != time }")
    main.addParmTemplate(start_frame)
    time_scale = hou.FloatParmTemplate(
        "time_scale", "Time Scale", 1, default_value=(1.0,),
        min=0.001, max=10.0,
        script_callback="hou.phm().time_mapping_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Simulation Time mapping: simulated time per unit of Houdini "
             "time (1 = one simulated second per Houdini second).")
    time_scale.setConditional(hou.parmCondType.HideWhen,
                              "{ time_mapping != time }")
    main.addParmTemplate(time_scale)
    sim_time = hou.StringParmTemplate(
        "sim_time_status", "Simulation Time", 1,
        default_expression=(
            "hou.pwd().hdaModule().sim_time_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: the time step on screen and its simulation time.")
    sim_time.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(sim_time)
    source_block = hou.StringParmTemplate(
        "source_block", "Source Block", 1, default_value=("Volume",),
        menu_type=hou.menuType.StringReplace,
        script_callback="hou.phm().source_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Which part of the result to load as the main geometry: Volume "
             "(the solid elements), Surface (boundary faces with normals, "
             "sidesets, tractions), Contact (contact and friction forces per "
             "surface node) or Points (FEM nodes). Only blocks present in the "
             "current PVD frame are listed; enable the matching options in "
             "the PolyFEM node's Output tab to produce the others.")
    source_block.setItemGeneratorScript(
        "hou.phm().source_block_menu(kwargs)")
    source_block.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    main.addParmTemplate(source_block)
    avail = hou.StringParmTemplate(
        "availability_status", "Available Data", 1,
        default_value=("Select a PVD file to inspect its renderable data.",),
        help="Read-only summary of what the loaded result contains (blocks, "
             "fields, bodies, fibers) and therefore which controls below are "
             "usable.")
    avail.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(avail)
    main.addParmTemplate(_read_only_text(
        "run_summary_status", "Run Summary",
        "hou.pwd().hdaModule().run_summary_text(hou.pwd())",
        "Read-only summary from PolyFEM's run-manifest.json (status, exit "
        "code, wall time, peak memory, rollbacks, AL stagnations, and "
        "contact history-sensitivity counters when present).",
        lines="2-5"))
    main.addParmTemplate(hou.MenuParmTemplate(
        "field_time_scope", "Field Availability Scope",
        ("current", "every", "any"),
        menu_labels=("Current Frame (fast)",
                     "Every Frame (scans whole sequence - slow)",
                     "Any Frame (scans whole sequence - slow)"),
        default_value=0,
        script_callback="hou.phm().time_scope_changed(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Controls which fields appear in menus. Current Frame only "
             "inspects the displayed frame and is fast; a stale selection "
             "simply shows the Unavailable Field Color, never an error. Every "
             "Frame and Any Frame must inspect every result file in the sequence to find "
             "intermittent fields and label their coverage -- this can take a "
             "while for long sequences of large meshes and there is no way "
             "around it (the field list lives inside each file)."))
    main.addParmTemplate(hou.ToggleParmTemplate(
        "remesh_mode", "Remeshing Mode", default_value=False,
        help="Rebuild topology every frame (needed for remeshing runs). This "
             "makes every frame change re-parse and rebuild the mesh, so "
             "scrubbing is much slower; leave off unless the topology actually "
             "changes over time. Off = topology parsed once and cached."))
    topo_frame = hou.IntParmTemplate(
        "topo_frame", "Topology Frame", 1, default_value=(0,),
        help="When Remeshing Mode is off, the mesh connectivity is read once "
             "from the time step this Houdini frame shows (through Time "
             "Mapping, like the playbar) and reused for every other frame "
             "(only positions and fields change). Leave at 0 (the first "
             "step) unless that step is missing or damaged.")
    topo_frame.setConditional(
        hou.parmCondType.DisableWhen, "{ remesh_mode == 1 }")
    main.addParmTemplate(topo_frame)
    main.addParmTemplate(hou.ToggleParmTemplate(
        "cache", "Cache Frames", default_value=True,
        script_callback="hou.phm().toggle_cache(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Keep each frame's imported data in memory once it has been "
             "read, so returning to a frame never reads its file again. "
             "Derived fields and every display setting (deformation, colors, "
             "comparison, smoothing, visibility, glyphs, fibers, clipping) "
             "are recalculated live from the cached data, so changing them "
             "never re-reads the files. The cache is rebuilt automatically "
             "when the PVD file, source block, topology mode, remeshing "
             "settings or Display Boundary Surface Only change; Clear Cache "
             "empties it, and so does turning this off. "
             "Cached frames are limited by Cache Memory Limit."))
    cache_limit = hou.FloatParmTemplate(
        "cache_memory_gb", "Cache Memory Limit (GB)", 1,
        default_value=(0.0,), min=0.0, max=512.0,
        help="Most memory the frame cache may use. A new node starts at the "
             "system memory available when it was created; 0 = half of this "
             "computer's memory. Each frame's size is measured when it is "
             "cached, and once the limit is reached the oldest cached frames "
             "are dropped first, so a long run of a large mesh does not push "
             "the computer into swapping. Lower it if other programs need "
             "the memory.")
    cache_limit.setConditional(hou.parmCondType.DisableWhen, "{ cache == 0 }")
    main.addParmTemplate(cache_limit)
    cache_status = hou.StringParmTemplate(
        "cache_status", "Cache Status", 1,
        default_expression=(
            "hou.pwd().hdaModule().cache_status_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: how many frames the cache holds and the memory "
             "they use, and how many fit within Cache Memory Limit at the "
             "measured size of one frame. Clear Cache resets it to zero.")
    cache_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(cache_status)
    memory_status = hou.StringParmTemplate(
        "memory_status", "Memory", 1,
        default_expression=(
            "hou.pwd().hdaModule().memory_status_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        help="Read-only: memory used by Houdini, and the computer's used, "
             "total and available memory (available = what can still be "
             "handed out without swapping).")
    memory_status.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    main.addParmTemplate(memory_status)
    cache_epoch = hou.IntParmTemplate(
        "cache_epoch", "cache_epoch", 1, default_value=(0,),
        help="Internal, hidden: counts Clear Cache presses so Cache Status "
             "refreshes immediately.")
    cache_epoch.hide(True)
    main.addParmTemplate(cache_epoch)
    data_epoch = hou.IntParmTemplate(
        "data_epoch", "data_epoch", 1, default_value=(0,),
        help="Internal, hidden: counts Refreshes that found rewritten result "
             "files, so the topology and frame stages read them again.")
    data_epoch.hide(True)
    main.addParmTemplate(data_epoch)
    main.addParmTemplate(hou.ButtonParmTemplate(
        "clear_cache", "Clear Cache",
        script_callback="hou.phm().clear_cache(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Drop every cached frame so the next frame change re-reads the "
             "files. Use it after a simulation has been re-run into the same "
             "output folder."))
    return main
