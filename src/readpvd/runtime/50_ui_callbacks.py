# =============================================================================
# UI callbacks (playbar / cache / scaling), ported from 0.26
# =============================================================================


# Houdini frame -> _frame_signature of the data cook_frame read for it, per
# asset instance: the frames the cooked-frame cache (cache1) may hold.
_COOKED_FRAMES = {}


def _frame_signature(pvd_path, step):
    """(step shown, identity of every file it is read from) of one step.

    The step is clamped the way load_frame clamps it, so a frame past the
    end of a running simulation records the step it really showed."""
    try:
        entries = read_pvd(pvd_path)
    except Exception:
        return None
    if not entries:
        return None
    step = max(0, min(int(step), len(entries) - 1))
    paths = [entries[step][1]]
    try:
        paths.extend(_frame_block_paths(pvd_path, step).values())
    except Exception:
        pass
    keys = []
    for path in paths:
        try:
            keys.append(_file_key(path))
        except OSError:
            keys.append(None)
    return step, tuple(keys)


def stale_cooked_frames(node):
    """Houdini frames whose cooked data no longer matches the files on disk:
    their step's files were rewritten, or the frame now shows another step
    (the run reached it, or the scene record changed)."""
    path = node.evalParm("PVD_file")
    cooked = _COOKED_FRAMES.get(node.sessionId(), {})
    stale = []
    for frame, signature in sorted(cooked.items()):
        try:
            step = entry_index(node, frame)
        except Exception:
            step = None
        if step is None or _frame_signature(path, step) != signature:
            stale.append(frame)
    return stale


def _bump_data_epoch(node):
    """Make the topology and frame stages read their files again."""
    epoch = node.parm("data_epoch")
    if epoch is not None:
        epoch.set(epoch.eval() + 1)


def refresh(kwargs=None, force_clear_cache=False):
    """Pick up new timesteps of the SAME simulation.

    Parsed files are keyed on (path, mtime, size). Aggregate caches are
    invalidated because referenced meshes can change without the PVD changing.
    The current frame stays in place. The cooked-frame cache is kept as long as every frame
    it may hold still shows the same step from the same files; a frame made
    from rewritten files (a rerun into the same folder) or from a step the
    run had not reached yet empties it, since the Cache SOP cannot drop
    single frames. Only a PVD file change (start) resets everything.
    """
    node = hou.pwd() if kwargs is None else kwargs["node"]
    pvd_path = node.evalParm("PVD_file")
    _invalidate_pvd_aggregates(pvd_path)
    try:
        entries = read_pvd(pvd_path)
    except Exception as e:
        # Likely the sim rewriting the .pvd mid-refresh: keep the current
        # playbar, caches, and settings; the next refresh will see it whole.
        _message(f"Failed to read .pvd: {e}")
        return
    if not entries:
        _message("Empty .pvd collection")
        return
    _save_step_count(node, pvd_path, len(entries))
    stale = stale_cooked_frames(node)
    if stale:
        force_clear_cache = True
        _bump_data_epoch(node)
        if hou.isUIAvailable():
            hou.ui.setStatusMessage(
                f"Read PVD: {len(stale)} cached frame(s) no longer match the "
                "result files (rerun or new steps); the cache was cleared.",
                hou.severityType.Message)
    if hou.isUIAvailable() and (
            node.parm("update_playbar") is None
            or node.evalParm("update_playbar")):
        first, last = playbar_range(node)
        hou.playbar.setUseIntegerFrames(True)
        hou.playbar.setFrameIncrement(1)
        hou.playbar.setFrameRange(first, last)
        hou.playbar.setPlaybackRange(first, last)
    sync_available_options({"node": node})
    if force_clear_cache:
        clear_cache(kwargs)


def _cache_path(key):
    while isinstance(key, tuple) and key:
        key = key[0]
    return os.path.abspath(key) if isinstance(key, str) else None


def _invalidate_pvd_aggregates(pvd_path):
    if not pvd_path:
        return
    path = os.path.abspath(pvd_path)
    for cache in (_SEQUENCE_METADATA_CACHE, _FORCE_FRAME_CACHE, _REFERENCE_CACHE):
        for key in list(cache):
            if _cache_path(key) == path:
                del cache[key]
    companion_path = _fiber_companion_path(path)
    for key in list(_FIBER_POINT_CACHE):
        if key[0] == companion_path:
            del _FIBER_POINT_CACHE[key]
    for sid, memo in list(_FORCE_CURVES.items()):
        if _cache_path(memo["key"]) == path:
            del _FORCE_CURVES[sid]
    record = scene_record_path(path)
    for key in list(_FORCE_NODES_CACHE):
        if _cache_path(key) == record:
            del _FORCE_NODES_CACHE[key]
    if _cache_path(_SCAN_CACHE.get("key")) == path:
        _SCAN_CACHE.clear()


def _invalidate_pvd_caches(pvd_path):
    """Drop cached disk data belonging to this PVD's folder without evicting
    other Read PVD nodes in the same Houdini session."""
    if not pvd_path:
        return
    folder = os.path.dirname(os.path.abspath(pvd_path))
    prefix = folder + os.sep
    _invalidate_pvd_aggregates(pvd_path)
    for cache in (_PVD_CACHE, _VTM_CACHE, _FIELD_INFO_CACHE, _VTU_CACHE,
                  _FIBER_COMPANION_CACHE, _FIBER_POINT_CACHE, _MANIFEST_CACHE):
        for key in list(cache):
            if (_cache_path(key) or "").startswith(prefix):
                del cache[key]


def start(kwargs=None):
    # A PVD file change is new data: drop this PVD's disk caches (leaving
    # other Read PVD nodes warm) and cooked frames, then jump to first frame
    # when Set Playbar Range on Refresh is enabled.
    node = hou.pwd() if kwargs is None else kwargs["node"]
    watcher = _FOLLOW_WATCHERS.get(node.sessionId())
    if watcher is not None:
        watcher["last_pvd_key"] = None
        watcher["last_count"] = 0
    pvd_path = node.evalParm("PVD_file")
    _invalidate_pvd_caches(pvd_path)
    # Jump to the first frame before inspecting frame options so start() does
    # not inspect Houdini's default frame 1 only to discard it for frame 0.
    if hou.isUIAvailable() and pvd_path and (
            node.parm("update_playbar") is None
            or node.evalParm("update_playbar")):
        try:
            if read_pvd(pvd_path):
                first, last = playbar_range(node)
                hou.playbar.setUseIntegerFrames(True)
                hou.playbar.setFrameIncrement(1)
                hou.playbar.setFrameRange(first, last)
                hou.playbar.setPlaybackRange(first, last)
                hou.setFrame(first)
        except Exception:
            pass
    refresh(kwargs, force_clear_cache=True)
    # First look: colors spread over the result's values instead of the
    # 0..1 a new node starts with, and the model centered in the viewport.
    # Run auto_range_default before _autocenter_clip so frame 0's active block
    # is already warm in _VTU_CACHE when _autocenter_clip inspects its bounds.
    auto_range_default(node)
    _autocenter_clip(node)
    if node.parm("legend_mode") is not None and node.evalParm("legend_mode") in (1, 3):
        autoplace_legend({"node": node}, only_if_default=True)
    if node.parm("gnomon_show") is not None and node.evalParm("gnomon_show"):
        autoplace_gnomon({"node": node}, only_if_default=True)
    update_color_status({"node": node})
    _frame_result(node)


