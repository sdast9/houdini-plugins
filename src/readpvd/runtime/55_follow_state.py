# =============================================================================
# Follow Running Simulation (periodic auto-refresh while PolyFEM is writing)
# =============================================================================

_FOLLOW_WATCHERS = {}


def _stop_following(node_or_id):
    """Unregister any active Follow Running Simulation watcher for a node."""
    sid = node_or_id if isinstance(node_or_id, int) else node_or_id.sessionId()
    watcher = _FOLLOW_WATCHERS.pop(sid, None)
    if watcher is None:
        return
    callback = watcher.get("callback")
    if callback is not None and hou.isUIAvailable():
        try:
            hou.ui.removeEventLoopCallback(callback)
        except Exception:
            pass


def _manifest_is_finished(manifest):
    """True only once run-manifest.json records a finalized completion state.

    PolyFEM writes completion={"status": "running", "exit_status": null,
    "finished_at": null, ...} as soon as a run starts, so a non-empty
    completion dict alone does NOT mean the run has finished.
    """
    if not isinstance(manifest, dict):
        return False
    completion = manifest.get("completion")
    if not isinstance(completion, dict) or not completion:
        return False
    status = str(completion.get("status") or "").strip().lower()
    if status in ("running", "in_progress", "active"):
        return False
    if status:
        return True
    return (completion.get("finished_at") is not None
            or completion.get("exit_status") is not None)


def _follow_identity(path, manifest):
    if not manifest:
        return None
    for key in ("run_id", "started_at"):
        if manifest.get(key):
            return key, manifest[key]
    try:
        manifest_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                     "run-manifest.json")
        return "file", _file_key(manifest_path)[1:]
    except OSError:
        return None


def _follow_status(node, message):
    message = _set_availability_status(node, message)
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(message, hou.severityType.Message)


_STEP_COUNT_DATA = "readpvd_step_count"


def _save_step_count(node, path, count):
    node.setUserData(_STEP_COUNT_DATA, json.dumps((os.path.abspath(path), count)))


def poll_running_simulation(node, force=False, jump_frame=None):
    """Check once whether a running simulation wrote new steps or finished."""
    sid = node.sessionId()
    pvd_path = node.evalParm("PVD_file").strip()
    if not node.evalParm("follow_running") or not pvd_path:
        _stop_following(sid)
        return {"action": "stopped", "steps": 0}
    watcher = _FOLLOW_WATCHERS.setdefault(sid, {
        "last_poll": 0.0, "path": pvd_path, "last_pvd_key": None,
        "last_count": 0, "baseline": None, "run_identity": None,
        "prior_run_pvd_key": None,
    })
    if watcher["path"] != pvd_path:
        watcher.update(path=pvd_path, last_pvd_key=None, last_count=0,
                       baseline=None, run_identity=None, prior_run_pvd_key=None)
    now = time.monotonic()
    interval = max(0.5, float(node.evalParm("follow_interval")))
    if not force and now - watcher["last_poll"] < interval:
        return {"action": "noop", "steps": watcher["last_count"]}
    watcher["last_poll"] = now

    manifest = run_manifest(pvd_path)
    identity = _follow_identity(pvd_path, manifest)
    baseline = watcher["baseline"]
    if baseline is not None and (identity is None or identity == baseline):
        _follow_status(node, "Waiting for new run (previous run completed)")
        return {"action": "noop", "steps": watcher["last_count"]}

    previous = watcher["run_identity"]
    if identity is not None:
        if identity != previous and (baseline is not None or previous is not None):
            watcher.update(prior_run_pvd_key=(watcher["last_pvd_key"]
                                              or watcher["prior_run_pvd_key"]),
                           last_pvd_key=None, last_count=0)
            _invalidate_pvd_caches(pvd_path)
            _bump_data_epoch(node)
            clear_cache({"node": node})
        watcher.update(baseline=None, run_identity=identity)
    finished = _manifest_is_finished(manifest)
    action = "noop"
    steps = watcher["last_count"]
    try:
        pvd_key = _file_key(pvd_path)
    except FileNotFoundError:
        pvd_key = None
        watcher.update(last_pvd_key=None, last_count=0)
        steps = 0

    prior = watcher["prior_run_pvd_key"]
    completion = (manifest or {}).get("completion") or {}
    status = str(completion.get("status") or "").strip().lower()
    failed = (status not in ("", "completed", "success", "succeeded")
              or completion.get("exit_status") not in (None, 0))
    if pvd_key is not None and pvd_key == prior and (not finished or failed):
        # This is still the previous run's collection.
        if not finished:
            _follow_status(node, "Waiting for the new run to write its PVD.")
            return {"action": "noop", "steps": 0}
        pvd_key = None
        steps = 0
    if pvd_key is not None and (pvd_key != watcher["last_pvd_key"] or finished):
        try:
            entries = read_pvd(pvd_path)
        except (OSError, ET.ParseError, VtuParseError, ValueError):
            # The solver may be replacing the collection. Retry next tick.
            return {"action": "noop", "steps": steps}
        previous_count = steps
        steps = len(entries)
        if steps < previous_count:
            _bump_data_epoch(node)
            clear_cache({"node": node})
            watcher.update(last_pvd_key=None, last_count=0)
        if entries:
            refresh({"node": node})
            do_jump = hou.isUIAvailable() if jump_frame is None else bool(jump_frame)
            if do_jump and (steps != previous_count or force):
                hou.setFrame(int(entry_frame(node, steps - 1)))
            action = "refreshed"
        watcher.update(last_pvd_key=pvd_key, last_count=steps, prior_run_pvd_key=None)

    if finished:
        node.parm("follow_running").set(0)
        _stop_following(sid)
        _follow_status(node, f"Run finished ({steps} steps) — stopped following.")
        action = "finished"
    return {"action": action, "steps": steps}


def toggle_follow_running(kwargs, restoring=False):
    """Arm Follow explicitly, or restore the saved setting after scene load."""
    node = kwargs["node"]
    sid = node.sessionId()
    _stop_following(sid)
    if not node.evalParm("follow_running"):
        return
    path = node.evalParm("PVD_file").strip()
    if not path:
        node.parm("follow_running").set(0)
        _follow_status(node, "Select a PVD file before enabling Follow Running Simulation.")
        return
    manifest = run_manifest(path)
    finished = _manifest_is_finished(manifest)
    if restoring and finished:
        node.parm("follow_running").set(0)
        return
    identity = _follow_identity(path, manifest)
    try:
        pvd_key = _file_key(path)
        count = len(read_pvd(path))
    except (OSError, ET.ParseError, VtuParseError, ValueError):
        pvd_key, count = None, 0
    saved_path, saved_count = json.loads(node.userData(_STEP_COUNT_DATA) or "[null, 0]")
    catch_up = restoring and count > (saved_count if saved_path == os.path.abspath(path) else 0)
    if not catch_up and count:
        _save_step_count(node, path, count)

    def tick(force=False):
        try:
            poll_running_simulation(node, force=force)
        except hou.ObjectWasDeleted:
            _stop_following(sid)
        except (OSError, ET.ParseError, VtuParseError) as error:
            _follow_status(node, f"Waiting for simulation output: {error}")
        except Exception as error:
            _stop_following(sid)
            node.parm("follow_running").set(0)
            _follow_status(node, f"Follow Running Simulation stopped: {error}")

    _FOLLOW_WATCHERS[sid] = {
        "last_poll": 0.0, "path": path, "last_pvd_key": None if catch_up else pvd_key,
        "last_count": saved_count if catch_up else count,
        "baseline": identity if finished else None, "prior_run_pvd_key": None,
        "run_identity": None if finished else identity,
        "callback": tick if hou.isUIAvailable() else None,
    }
    if hou.isUIAvailable():
        try:
            hou.ui.addEventLoopCallback(tick)
        except Exception as error:
            _stop_following(sid)
            node.parm("follow_running").set(0)
            _follow_status(node, f"Follow Running Simulation stopped: {error}")
            return
    tick(force=True)


def _frame_result(node):
    """Frame the Scene Viewer on the loaded result (interactive sessions)."""
    viewer = _scene_viewer()
    if viewer is None:
        return
    try:
        output = node.node("output")
        output.cook()
        bbox = output.geometry().boundingBox()
        if not bbox.isValid():
            return
        try:
            bbox = bbox * node.worldTransform()
        except (TypeError, AttributeError, hou.Error):
            pass
        viewer.curViewport().frameBoundingBox(bbox)
    except (hou.Error, AttributeError):
        pass


def clear_cache(kwargs=None):
    """Flush the cooked-frame cache (cache1). Never moves the playbar.

    Pressing an internal button is allowed on a locked asset; nothing here
    may unlock the node (allowEditingOfContents), or the instance stops
    receiving later asset rebuilds.
    """
    node = hou.pwd() if kwargs is None else kwargs["node"]
    cache_node = node.node("cache1")
    if cache_node is not None:
        cache_node.parm("clear").pressButton()
    _COOKED_FRAMES.pop(node.sessionId(), None)
    _CACHE_CLEARED.add(node.sessionId())
    _CACHE_CONTENTS.pop(node.sessionId(), None)
    _CACHE_FRAME_BYTES.pop(node.sessionId(), None)
    # Cache Status reads this counter, so it updates the moment the button is
    # pressed instead of on the next frame change.
    epoch = node.parm("cache_epoch")
    if epoch is not None:
        epoch.set(epoch.eval() + 1)


def toggle_cache(kwargs):
    # cache_switch follows the toggle by expression; turning the cache off
    # also releases the frames it holds.
    clear_cache(kwargs)


# Measured size of one cached frame per asset instance (bytes), updated
# whenever the cache's input is already cooked.
_CACHE_FRAME_BYTES = {}
CACHE_FRAME_CEILING = 2500   # the Cache SOP's own default Max Frames


def _physical_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 16 << 30


def _cache_budget_bytes(asset):
    parm = asset.parm("cache_memory_gb")
    limit_gb = parm.eval() if parm is not None else 0.0
    if limit_gb > 0:
        return int(limit_gb * (1 << 30))
    return _physical_memory() // 2


def cache_frame_limit(cache_node):
    """Max Frames for cache1: as many frames as fit the memory budget.

    Evaluated by cache1's maxframes expression on every cook. The frame size
    is read only from an input that is already cooked, so a cache hit never
    forces the upstream stages to run.
    """
    asset = cache_node.parent()
    key = asset.sessionId()
    source = cache_node.input(0)
    if source is not None and not source.needsToCook():
        try:
            size = int(source.geometry().intrinsicValue("memoryusage"))
        except (hou.Error, TypeError, ValueError):
            size = 0
        if size > 0:
            _CACHE_FRAME_BYTES[key] = size
    size = _CACHE_FRAME_BYTES.get(key)
    if not size:
        return CACHE_FRAME_CEILING
    return int(max(1, min(CACHE_FRAME_CEILING,
                          _cache_budget_bytes(asset) // size)))


# Last frames/memory the Cache SOP reported, per asset instance, and the
# instances whose cache was cleared since their last new frame.
_CACHE_CONTENTS = {}
_CACHE_CLEARED = set()


def _cache_contents(node, cache_node):
    """(frames cached, memory text) as the Cache SOP reports them.

    Reading the info tree of a node that needs to cook would cook it (and,
    right after Clear Cache, re-cache the frame on screen), so a dirty cache
    reports the last known contents instead.
    """
    key = node.sessionId()
    if cache_node.needsToCook():
        if key in _CACHE_CLEARED:
            return 0, "0 B"
        return _CACHE_CONTENTS.get(key)
    try:
        rows = dict(cache_node.infoTree(verbose=False).branches()[
            "Cache SOP Info"].rows())
        contents = (int(rows["Geometries Cached"].split("/")[0]),
                    rows["Memory Used"])
    except (hou.Error, KeyError, ValueError, AttributeError):
        return _CACHE_CONTENTS.get(key)
    _CACHE_CONTENTS[key] = contents
    if contents[0] > 1:
        _CACHE_CLEARED.discard(key)
    return contents


def _system_memory():
    """(total, available) physical memory in bytes, or None."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        return None


def _houdini_memory_bytes():
    """Resident memory of this Houdini process in bytes, or None."""
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        rss_kb = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, timeout=2).stdout.strip()
        return int(rss_kb) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def available_memory_gb():
    """System memory available right now, in GB (the OnCreated default for
    Cache Memory Limit); falls back to half of physical memory."""
    memory = _system_memory()
    available = memory[1] if memory else _physical_memory() // 2
    return round(available / float(1 << 30), 1)


def cache_status_text(node):
    hou.frame()   # time-dependent, so the readout refreshes as frames load
    node.evalParm("cache_epoch")   # ... and when Clear Cache is pressed
    if not node.evalParm("cache"):
        return "Cache off: every frame change reads its result file."
    budget = _cache_budget_bytes(node) / float(1 << 30)
    size = _CACHE_FRAME_BYTES.get(node.sessionId())
    cache_node = node.node("cache1")
    contents = (_cache_contents(node, cache_node)
                if cache_node is not None else None)
    if contents is None:
        held = ""
    elif contents[0] <= 1 and node.sessionId() in _CACHE_CLEARED:
        held = ("Cleared: 0 frames cached. " if contents[0] == 0 else
                f"Cleared: only the frame on screen is cached "
                f"({contents[1]}). ")
    else:
        held = (f"{contents[0]} frame{'s' if contents[0] != 1 else ''} "
                f"cached ({contents[1]}). ")
    if not size:
        return (f"{held}Limit {budget:.1f} GB; the frame size is measured "
                "when the first frame is cached.")
    frames = min(CACHE_FRAME_CEILING, max(1, int(budget * (1 << 30) // size)))
    return (f"{held}Limit {budget:.1f} GB holds {frames} "
            f"frame{'s' if frames != 1 else ''} of "
            f"{size / float(1 << 30):.2f} GB; the oldest are dropped first.")


def memory_status_text(node):
    hou.frame()   # time-dependent, so the readout refreshes as frames load
    gb = float(1 << 30)
    parts = []
    houdini = _houdini_memory_bytes()
    if houdini:
        parts.append(f"Houdini {houdini / gb:.1f} GB")
    memory = _system_memory()
    if memory:
        total, available = memory
        parts.append(f"System {(total - available) / gb:.1f} of "
                     f"{total / gb:.1f} GB used, {available / gb:.1f} GB "
                     "available")
    return " | ".join(parts) or "Memory use is not available on this system."


