"""Build the internal SOP network for Read PVD."""

import hou

import hda_build
from . import scripts


def build_network(asset):
    def wrangle(name, klass, vex_file, inp):
        snippet = hda_build.read_source("readpvd", "vex", vex_file)
        node = asset.createNode("attribwrangle", name)
        node.setParms({"class": klass, "snippet": snippet})
        if inp is not None:
            node.setNextInput(inp)
        return node

    topo = asset.createNode("python", "topo_build")
    topo.parm("python").set(scripts.TOPO_SOP_CODE)
    # spare parms establish cook dependencies on the asset parms
    spare = hou.StringParmTemplate("pvdpath", "pvdpath", 1)
    topo.addSpareParmTuple(spare)
    topo.parm("pvdpath").setExpression('chs("../PVD_file")',
                                       hou.exprLanguage.Hscript)
    spare_f = hou.FloatParmTemplate("topoframe", "topoframe", 1)
    topo.addSpareParmTuple(spare_f)
    # Remeshing Mode follows the playbar; otherwise the fixed Topology Frame.
    # An expression (not a callback editing the node) keeps instances locked.
    # Python short-circuits, so hou.frame() -- and with it time dependence --
    # is only reached in Remeshing Mode (an Hscript if() evaluates both).
    # Topology Frame is a Houdini frame, read through the time mapping like
    # the playbar (review 2026-11-14, item 5).
    topo.parm("topoframe").setExpression(
        "hou.pwd().parent().hdaModule().entry_index(hou.pwd().parent()) "
        "if hou.pwd().parent().evalParm('remesh_mode') "
        "else hou.pwd().parent().hdaModule().frame_step(hou.pwd().parent(), "
        "hou.pwd().parent().evalParm('topo_frame'))",
        hou.exprLanguage.Python)

    prims = wrangle("build_prims", 0, "build_prims.vex", topo)

    boundary = wrangle("boundary_display", 1, "boundary.vex", prims)
    surf_switch = asset.createNode("switch", "surface_switch")
    surf_switch.parm("input").setExpression('1 - ch("../surface_only")')
    surf_switch.setNextInput(boundary)   # input 0: surface only
    surf_switch.setNextInput(prims)      # input 1: full volume prims

    frame = asset.createNode("python", "frame_data")
    frame.parm("python").set(scripts.FRAME_SOP_CODE)
    frame.addSpareParmTuple(hou.StringParmTemplate("pvdpath", "pvdpath", 1))
    frame.parm("pvdpath").setExpression('chs("../PVD_file")',
                                        hou.exprLanguage.Hscript)
    frame.addSpareParmTuple(hou.FloatParmTemplate("frame", "frame", 1))
    frame.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    frame.setNextInput(surf_switch)
    # Refresh bumps data_epoch when result files were rewritten in place.
    for stage in (topo, frame):
        stage.addSpareParmTuple(hou.IntParmTemplate("dataepoch", "dataepoch", 1))
        stage.parm("dataepoch").setExpression('ch("../data_epoch")',
                                              hou.exprLanguage.Hscript)

    # Cache only what comes from disk: topology, P, and the imported
    # PVD/companion attributes (~1 GB per frame at 2.4M points). Everything
    # else -- derived mechanics included -- is recalculated downstream, which
    # is fast next to a file read, so no setting change (Derived, color,
    # clipping, deformation display, smoothing, visibility, glyphs, fibers)
    # ever makes a cached frame read its file again. Keep the historical node
    # name because callbacks and existing scenes address it directly.
    cache = asset.createNode("cache", "cache1")
    cache.setNextInput(frame)
    # Frames beyond the memory budget are dropped oldest-first instead of
    # growing until the machine swaps.
    cache.parm("maxframes").setExpression(
        "hou.pwd().parent().hdaModule().cache_frame_limit(hou.pwd())",
        hou.exprLanguage.Python)
    # Deduplicate cache entries by output step: under Simulation Time (or past
    # the last step) several Houdini frames show the same PVD step, so shifting
    # cache lookups to the step's canonical frame stores one copy per step.
    step_shift = asset.createNode("timeshift", "step_shift")
    step_shift.setNextInput(cache)
    step_shift.parm("frame").setExpression(
        "hou.pwd().parent().hdaModule().cache_frame(hou.pwd().parent())",
        hou.exprLanguage.Python)
    # Cache Frames selects the cached stream by expression. Bypassing cache1
    # from a callback would need allowEditingOfContents(), which unlocks the
    # instance so it silently stops receiving asset updates.
    cache_switch = asset.createNode("switch", "cache_switch")
    cache_switch.parm("input").setExpression('ch("../cache")')
    cache_switch.setNextInput(frame)            # 0: read every frame
    cache_switch.setNextInput(step_shift)       # 1: cached frames (per step)

    # Mechanics are independent of whether the user draws the rest or
    # displaced positions, so calculate them on the raw PVD geometry.
    # derived_prepare allocates only the needed point attributes; derived runs
    # multi-threaded over points to fill them.
    derived_prepare = wrangle(
        "derived_prepare", 0, "derived_prepare.vex", cache_switch)
    derived = wrangle("derived", 2, "derived.vex", derived_prepare)
    derived_switch = asset.createNode("switch", "derived_switch")
    derived_switch.parm("input").setExpression('ch("../derived")')
    derived_switch.setNextInput(cache_switch)
    derived_switch.setNextInput(derived)

    deform = wrangle("deform", 2, "deform.vex", derived_switch)  # class 2 = point

    reference = asset.createNode("python", "reference_comparison")
    reference.parm("python").set(scripts.REFERENCE_SOP_CODE)
    reference.addSpareParmTuple(hou.FloatParmTemplate("frame", "frame", 1))
    reference.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    reference.addSpareParmTuple(hou.IntParmTemplate("dataepoch", "dataepoch", 1))
    reference.parm("dataepoch").setExpression('ch("../data_epoch")',
                                              hou.exprLanguage.Hscript)
    reference.setNextInput(deform)

    # Field reduction -> optional nodal averaging -> ramp mapping.
    color_reduce = wrangle("color_reduce", 2, "color_reduce.vex", reference)
    field_smoothing = asset.createNode("python", "field_smoothing")
    field_smoothing.parm("python").set(scripts.SMOOTH_SOP_CODE)
    field_smoothing.setNextInput(color_reduce)
    smooth_switch = asset.createNode("switch", "smooth_switch")
    smooth_switch.parm("input").setExpression('ch("../smooth_field")')
    smooth_switch.setNextInput(color_reduce)      # 0: raw per-element value
    smooth_switch.setNextInput(field_smoothing)   # 1: nodally averaged
    color = wrangle("color_map", 2, "color_map.vex", smooth_switch)

    # Show/hide bodies. Filtering before OUT_result means the range, probe,
    # glyphs, and clip all act on the visible bodies only.
    body_filter = wrangle("body_filter", 1, "body_filter.vex", color)
    body_filter_pts = wrangle("body_filter_points", 2, "body_filter_points.vex",
                              body_filter)

    out_result = asset.createNode("null", "OUT_result")
    out_result.setNextInput(body_filter_pts)

    # Glyphs are kept as their own stream (blast everything but the glyph
    # group) so the clip/slice stage below never cuts them.
    glyph_recovery = asset.createNode("python", "glyph_tensor_recovery")
    glyph_recovery.parm("python").set(scripts.GLYPH_RECOVERY_SOP_CODE)
    glyph_recovery.setNextInput(out_result)
    glyphs = wrangle("glyphs", 2, "glyph.vex", glyph_recovery)
    glyph_only = asset.createNode("blast", "glyph_only")
    glyph_only.setNextInput(glyphs)
    glyph_only.setParms({"group": "readpvd_glyphs", "grouptype": 4,
                         "negate": 1})       # keep only the glyph primitives
    empty_glyphs = asset.createNode("null", "empty_glyphs")
    glyph_switch = asset.createNode("switch", "glyph_switch")
    glyph_switch.parm("input").setExpression(
        'ch("../add_glyphs") && ch("../has_glyph_data")')
    glyph_switch.setNextInput(empty_glyphs)
    glyph_switch.setNextInput(glyph_only)

    # Fibers get their own stream for the same reason glyphs do: a clipped
    # fiber line means nothing, and the lines must not be re-coloured by the
    # field ramp.
    fiber_recovery = asset.createNode("python", "fiber_recovery")
    fiber_recovery.parm("python").set(scripts.FIBER_RECOVERY_SOP_CODE)
    fiber_recovery.setNextInput(out_result)
    fibers = wrangle("fibers", 2, "fiber.vex", fiber_recovery)
    fiber_only = asset.createNode("blast", "fiber_only")
    fiber_only.setNextInput(fibers)
    fiber_only.setParms({"group": "readpvd_fibers", "grouptype": 4,
                         "negate": 1})
    empty_fibers = asset.createNode("null", "empty_fibers")
    fiber_switch = asset.createNode("switch", "fiber_switch")
    fiber_switch.parm("input").setExpression(
        'ch("../show_fibers") && ch("../has_fiber_data")')
    fiber_switch.setNextInput(empty_fibers)
    fiber_switch.setNextInput(fiber_only)

    multi_blocks = asset.createNode("python", "multi_block_display")
    multi_blocks.parm("python").set(scripts.MULTI_BLOCK_SOP_CODE)
    multi_blocks.addSpareParmTuple(hou.StringParmTemplate(
        "pvdpath", "pvdpath", 1))
    multi_blocks.parm("pvdpath").setExpression(
        'chs("../PVD_file")', hou.exprLanguage.Hscript)
    multi_blocks.addSpareParmTuple(hou.FloatParmTemplate(
        "frame", "frame", 1))
    multi_blocks.parm("frame").setExpression("$F", hou.exprLanguage.Hscript)
    multi_blocks.addSpareParmTuple(hou.IntParmTemplate(
        "dataepoch", "dataepoch", 1))
    multi_blocks.parm("dataepoch").setExpression(
        'ch("../data_epoch")', hou.exprLanguage.Hscript)
    empty_multi = asset.createNode("null", "empty_multi_blocks")
    multi_switch = asset.createNode("switch", "multi_block_switch")
    multi_switch.parm("input").setExpression('ch("../show_multi_blocks")')
    multi_switch.setNextInput(empty_multi)
    multi_switch.setNextInput(multi_blocks)

    # Clip/slice the result and supplementary blocks (glyphs excluded).
    # The topology stage keeps every element face and groups the interior
    # ones, so the cached frame serves both views: the outer surface alone,
    # or -- while a clip or slice is active -- every face, so the cut shows
    # values inside the body. Points are kept (probe, range, glyphs).
    interior = wrangle("drop_interior_faces", 1, "drop_interior.vex",
                       out_result)
    surface_view = asset.createNode("switch", "surface_view_switch")
    surface_view.parm("input").setExpression(
        'ch("../surface_only") && ch("../clip_mode") == 0')
    surface_view.setNextInput(out_result)   # 0: every face
    surface_view.setNextInput(interior)     # 1: outer surface only

    clip_input = asset.createNode("merge", "clip_input")
    clip_input.setNextInput(surface_view)
    clip_input.setNextInput(multi_switch)
    clip_one = asset.createNode("clip::2.0", "clip_plane")
    clip_one.setNextInput(clip_input)
    clip_two = asset.createNode("clip::2.0", "slice_second_plane")
    clip_two.setNextInput(clip_one)
    for clip_node in (clip_one, clip_two):
        clip_node.parm("dirtype").set(0)
        for axis in "xyz":
            clip_node.parm("dir" + axis).setExpression(
                f'ch("../clip_direction{axis}")', hou.exprLanguage.Hscript)
        clip_node.parm("dofill").setExpression(
            'ch("../clip_fill")', hou.exprLanguage.Hscript)
    for axis in "xyz":
        idx = "xyz".index(axis)
        clip_one.parm("origin" + axis).setExpression(
            "hou.pwd().parent().evalParmTuple('clip_origin')"
            f"[{idx}] - ("
            "hou.Vector3(hou.pwd().parent().evalParmTuple('clip_direction'))"
            f".normalized()[{idx}] * 0.5 * "
            "hou.pwd().parent().evalParm('slice_thickness') "
            "if hou.pwd().parent().evalParm('clip_mode') == 2 else 0.0)",
            hou.exprLanguage.Python)
        clip_two.parm("origin" + axis).setExpression(
            "hou.pwd().parent().evalParmTuple('clip_origin')"
            f"[{idx}] + "
            "hou.Vector3(hou.pwd().parent().evalParmTuple('clip_direction'))"
            f".normalized()[{idx}] * 0.5 * "
            "hou.pwd().parent().evalParm('slice_thickness')",
            hou.exprLanguage.Python)
    clip_one.parm("clipop").setExpression(
        "'above' if hou.pwd().parent().evalParm('clip_mode') == 2 "
        "else hou.pwd().parent().evalParm('clip_keep')",
        hou.exprLanguage.Python)
    clip_two.parm("clipop").set("below")
    section_switch = asset.createNode("switch", "section_switch")
    section_switch.parm("input").setExpression('ch("../clip_mode")')
    section_switch.setNextInput(clip_input)
    section_switch.setNextInput(clip_one)
    section_switch.setNextInput(clip_two)

    # Merge the (possibly clipped) result with the unclipped glyphs.
    display_blocks = asset.createNode("merge", "display_blocks")
    display_blocks.setNextInput(section_switch)
    display_blocks.setNextInput(glyph_switch)
    display_blocks.setNextInput(fiber_switch)

    # Compatibility alias retained for existing callbacks/tools.
    out_src = asset.createNode("null", "OUT_color_source")
    out_src.setNextInput(display_blocks)

    # Renderable scene-space legend. The color bar, tick marks, per-tick
    # numeric labels, and title are all generated in Python (cook_scene_legend)
    # in a single local space, so label spacing always matches the bar.
    legend_bar = asset.createNode("python", "legend_bar")
    legend_bar.parm("python").set(scripts.LEGEND_SOP_CODE)
    legend_bar.setNextInput(out_result)
    legend_xform = asset.createNode("xform", "legend_scene_transform")
    legend_xform.setNextInput(legend_bar)
    for axis, index in zip("xyz", range(3)):
        legend_xform.parm("t" + axis).setExpression(
            f'ch("../legend_translate{axis}")', hou.exprLanguage.Hscript)
        legend_xform.parm("r" + axis).setExpression(
            "hou.pwd().parent().hdaModule().legend_rotation("
            f"hou.pwd().parent())[{index}] + "
            f"hou.pwd().parent().evalParm('legend_rotate{axis}')",
            hou.exprLanguage.Python)
        legend_xform.parm("s" + axis).setExpression(
            'ch("../legend_scale")', hou.exprLanguage.Hscript)
    empty_legend = asset.createNode("null", "empty_legend")
    legend_switch = asset.createNode("switch", "legend_scene_switch")
    legend_switch.parm("input").setExpression(
        'if(ch("../legend_mode")==1 || ch("../legend_mode")==3, 1, 0)')
    legend_switch.setNextInput(empty_legend)
    legend_switch.setNextInput(legend_xform)

    # Optional renderable XYZ gnomon.
    gnomon = asset.createNode("python", "gnomon_geometry")
    gnomon.parm("python").set(scripts.GNOMON_SOP_CODE)
    empty_gnomon = asset.createNode("null", "empty_gnomon")
    gnomon_switch = asset.createNode("switch", "gnomon_switch")
    gnomon_switch.parm("input").setExpression('ch("../gnomon_show")')
    gnomon_switch.setNextInput(empty_gnomon)
    gnomon_switch.setNextInput(gnomon)

    final_merge = asset.createNode("merge", "display_merge")
    final_merge.setNextInput(out_src)
    final_merge.setNextInput(legend_switch)
    final_merge.setNextInput(gnomon_switch)

    out = asset.createNode("output", "output")
    out.setNextInput(final_merge)
    asset.layoutChildren()
