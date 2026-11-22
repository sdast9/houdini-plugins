# =============================================================================
# Timeline: which PVD step a Houdini frame shows (2026-10-02)
#
# The PolyFEM node plays a run on Houdini's timeline: Houdini time is
# simulation time, t = Start Time + (T - T_start) * Time Scale, recorded in
# the run's scene record (<run>/input/hda_scene.json). Time Mapping
# "Automatic" uses that record and falls back to one frame per output step
# (frame k shows step k, as before) for runs without one. On the simulation
# timeline a frame shows the last step at or before its simulation time.
# =============================================================================

SCENE_RECORD_FILE = "hda_scene.json"
SCENE_RECORD_SCHEMA = "polyfem-houdini-scene"
TIME_MAPPING_TOKENS = ("auto", "steps", "time")
_SCENE_RECORD_CACHE = {}


def scene_record_path(pvd_path):
    """<run>/input/hda_scene.json for <run>/output/<name>.pvd."""
    run = os.path.dirname(os.path.dirname(os.path.abspath(pvd_path)))
    return os.path.join(run, "input", SCENE_RECORD_FILE)


def scene_record(node):
    """The PolyFEM node's record of this run, or None."""
    pvd = node.evalParm("PVD_file")
    if not pvd:
        return None
    path = scene_record_path(pvd)
    try:
        key = _file_key(path)
    except OSError:
        return None
    cached = _SCENE_RECORD_CACHE.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict) \
            or record.get("schema") != SCENE_RECORD_SCHEMA:
        record = None
    if len(_SCENE_RECORD_CACHE) > 8:
        _SCENE_RECORD_CACHE.clear()
    _SCENE_RECORD_CACHE[path] = (key, record)
    return record


def _pvd_times(node):
    path = node.evalParm("PVD_file")
    if not path:
        return []
    try:
        return [entry[0] for entry in read_pvd(path)]
    except Exception:
        return []


def time_mapping(node):
    """(Houdini time of the start, time scale, simulation start time) of the
    playback timeline, or None for one frame per output step."""
    parm = node.parm("time_mapping")
    token = parm.evalAsString() if parm is not None else "steps"
    if token == "steps":
        return None
    record = scene_record(node)
    line = record.get("timeline") if record else None
    if token == "auto":
        if not isinstance(line, dict):
            return None
        try:
            return ((float(line["start_frame"]) - 1.0) / float(line["fps"]),
                    float(line["time_scale"]), float(line["t0"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    if isinstance(line, dict) and "t0" in line:
        t0 = float(line["t0"])
    else:
        times = _pvd_times(node)
        t0 = times[0] if times else 0.0
    scale = float(node.evalParm("time_scale"))
    return (hou.frameToTime(node.evalParm("time_start_frame")),
            scale if scale > 0 else 1.0, t0)


def entry_index(node, frame=None):
    """Index of the PVD step shown at a Houdini frame (default: current)."""
    frame = hou.frame() if frame is None else frame
    mapping = time_mapping(node)
    if mapping is None:
        return int(frame)
    times = _pvd_times(node)
    if not times:
        return 0
    start, scale, t0 = mapping
    t = t0 + (hou.frameToTime(frame) - start) * scale
    tolerance = 1e-9 * max(1.0, abs(times[-1] - times[0]))
    return max(0, bisect.bisect_right(times, t + tolerance) - 1)


def frame_step(node, frame):
    """The output step Houdini frame `frame` shows, clamped to the run.

    Frame parameters (Reference Frame, Topology Frame) go through this, so
    on the Simulation Time mapping they mean the frame, not a step number
    (review 2026-11-14, item 5)."""
    count = len(_pvd_times(node))
    if not count:
        return 0
    return max(0, min(entry_index(node, frame), count - 1))


def entry_frame(node, index):
    """The first Houdini frame showing PVD step `index`."""
    mapping = time_mapping(node)
    if mapping is None:
        return float(index)
    times = _pvd_times(node)
    if not times:
        return 0.0
    start, scale, t0 = mapping
    index = max(0, min(int(index), len(times) - 1))
    return math.ceil(hou.timeToFrame(start + (times[index] - t0) / scale)
                     - 1e-6)


def cache_frame(node):
    """Canonical Houdini frame for the current PVD step.

    Drives step_shift downstream of cache1 so every Houdini frame mapping to
    the same output step (common under Simulation Time or past the end of a
    sequence) reuses one cache entry instead of storing duplicate copies.
    """
    times = _pvd_times(node)
    if not times:
        return float(hou.frame())
    step = max(0, min(entry_index(node), len(times) - 1))
    canonical = float(entry_frame(node, step))
    if max(0, min(entry_index(node, canonical), len(times) - 1)) == step:
        return canonical
    return float(hou.frame())


def playbar_range(node):
    """(first, last) whole frames covering the run on the active mapping."""
    times = _pvd_times(node)
    if not times:
        return 0, 0
    if time_mapping(node) is None:
        return 0, len(times) - 1
    return (int(entry_frame(node, 0)),
            int(max(entry_frame(node, 0), entry_frame(node, len(times) - 1))))


_MANIFEST_CACHE = {}


def run_manifest(node_or_path):
    """Parsed run-manifest.json next to the PVD file, cached by file identity,
    or None when absent or unreadable."""
    if isinstance(node_or_path, str):
        pvd = node_or_path
    else:
        pvd = node_or_path.evalParm("PVD_file") if node_or_path is not None else ""
    if not pvd:
        return None
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(pvd)),
                                 "run-manifest.json")
    try:
        key = _file_key(manifest_path)
    except OSError:
        return None
    cached = _MANIFEST_CACHE.get(manifest_path)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        data = None
    if len(_MANIFEST_CACHE) > 8:
        _MANIFEST_CACHE.clear()
    _MANIFEST_CACHE[manifest_path] = (key, data)
    return data


def _format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


def run_summary_text(node):
    """Read-only Run Summary from PolyFEM's run-manifest.json."""
    if node.parm("data_epoch") is not None:
        node.evalParm("data_epoch")
    pvd = node.evalParm("PVD_file")
    if not pvd:
        return "Load a PVD file."
    manifest = run_manifest(pvd)
    if not manifest:
        return "No run-manifest.json found next to this PVD."
    completion = manifest.get("completion") or {}
    status = completion.get("status") or "unknown"
    exit_code = completion.get("exit_status")
    wall = completion.get("wall_seconds")
    rss = completion.get("peak_rss_mb")
    head = [f"Status: {status}" + (
        f" (exit {exit_code})" if exit_code is not None else "")]
    if isinstance(wall, (int, float)):
        head.append(f"Wall: {_format_duration(wall)}")
    if isinstance(rss, (int, float)):
        head.append(f"Peak RSS: {rss / 1024.0:.2f} GB" if rss >= 1024
                    else f"Peak RSS: {rss:.0f} MB")
    lines = [" | ".join(head)]
    steps = manifest.get("steps")
    if isinstance(steps, list) and steps:
        n_steps = len(steps)
        non_accepted = sum(
            1 for s in steps
            if isinstance(s, dict) and s.get("outcome") not in (None, "accepted"))
        retunes = sum(
            int(s.get("stall_retunes") or 0) for s in steps
            if isinstance(s, dict))
        stagnations = sum(
            1 for s in steps
            if isinstance(s, dict) and s.get("al_stagnation"))
        losses = 0
        splits = 0
        for s in steps:
            if not isinstance(s, dict):
                continue
            hs = (s.get("contact") or {}).get("history_sensitivity") or {}
            cont = hs.get("continuation") or {}
            loss_map = cont.get("losses") or {}
            if isinstance(loss_map, dict):
                losses += sum(int(v or 0) for v in loss_map.values())
            splits += int(cont.get("split_pairs") or 0)
        parts = [f"Steps logged: {n_steps}"]
        if non_accepted:
            parts.append(f"non-accepted: {non_accepted}")
        if retunes:
            parts.append(f"stall retunes: {retunes}")
        if stagnations:
            parts.append(f"AL stagnations: {stagnations}")
        if losses or splits:
            parts.append(f"continuation losses: {losses}, split pairs: {splits}")
        lines.append(", ".join(parts))
    msg = completion.get("message")
    if msg:
        lines.append(f"Message: {msg}")
    return "\n".join(lines)


def _step_health_tag(node, step_index):
    """Compact per-step diagnostic tag from run-manifest.json, or ''."""
    manifest = run_manifest(node)
    if not manifest:
        return ""
    steps = manifest.get("steps")
    if not isinstance(steps, list) or not steps:
        return ""
    match = None
    for entry in steps:
        if isinstance(entry, dict) and entry.get("step") == step_index:
            match = entry
            break
    if match is None:
        return ""
    flags = []
    outcome = match.get("outcome")
    if outcome and outcome != "accepted":
        flags.append(str(outcome))
    if match.get("al_stagnation"):
        flags.append("AL stagnation")
    retunes = int(match.get("stall_retunes") or 0)
    if retunes:
        flags.append(f"{retunes} stall retune{'s' if retunes != 1 else ''}")
    return f"  [{', '.join(flags)}]" if flags else ""


def sim_time_text(node):
    """Read-only Simulation Time line: the step on screen and its time."""
    hou.frame()  # time-dependent, so it follows the playbar
    times = _pvd_times(node)
    if not times:
        return "Load a PVD file."
    index = max(0, min(entry_index(node), len(times) - 1))
    if time_mapping(node) is None:
        how = "one frame per output step"
        parm = node.parm("time_mapping")
        if parm is not None and parm.evalAsString() == "auto":
            how += " (this run has no PolyFEM node timeline)"
    else:
        how = "simulation time"
    return (f"Step {index} of {len(times) - 1}: t = {times[index]:.6g}  "
            f"({how}){_step_health_tag(node, index)}")


def time_mapping_changed(kwargs):
    """A new mapping shows other steps on the same frames: drop the cooked
    frames and set the playbar to the run."""
    clear_cache(kwargs)
    refresh(kwargs)


def _frame_metadata(node):
    """Renderable field metadata for the active frame (no array decode)."""
    path = node.evalParm("PVD_file")
    if not path:
        return {}
    frame = entry_index(node)
    try:
        info = frame_field_info(path, frame)
    except Exception:
        return {}
    return {block: _fields_from_info(block_info)
            for block, block_info in info.items()}


def _sequence_metadata(node):
    """Per-frame field metadata for the whole sequence (no array decode).

    Returns (frames, block_counts) where frames[i] maps block -> fields and
    block_counts maps block -> number of frames it appears in. Cached on the
    PVD's file identity; the underlying scans are byte-level, so a full
    sequence costs milliseconds even for million-element meshes.
    """
    path = node.evalParm("PVD_file")
    if not path:
        return [], {}
    try:
        key = _file_key(path)
        entries = read_pvd(path)
    except Exception:
        return [], {}
    cached = _SEQUENCE_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    frames = []
    for frame in range(len(entries)):
        try:
            info = frame_field_info(path, frame)
        except Exception:
            frames.append({})
            continue
        frames.append({block: _fields_from_info(block_info)
                       for block, block_info in info.items()})
    block_counts = {}
    for frame_meta in frames:
        for block_name in frame_meta:
            block_counts[block_name] = block_counts.get(block_name, 0) + 1
    result = (frames, block_counts)
    if len(_SEQUENCE_METADATA_CACHE) > 8:
        _SEQUENCE_METADATA_CACHE.clear()
    _SEQUENCE_METADATA_CACHE[key] = result
    return result


def _selected_fields(node):
    metadata = _frame_metadata(node)
    if not metadata:
        return "", {}
    selected = _source_block_token(node)
    if selected not in metadata:
        selected = next(iter(metadata))
    return selected, metadata[selected]


def _requirements(fields):
    has_f = all(name in fields for name in ("F_1", "F_2", "F_3"))
    has_cauchy = all(
        name in fields for name in
        ("cauchy_stress_1", "cauchy_stress_2", "cauchy_stress_3"))
    # Cauchy stress can be reconstructed from the 1st PK stress and F, so
    # exporting either (with F) unlocks the full derived-stress set.
    has_pk1 = all(
        name in fields for name in
        ("pk1_stress_1", "pk1_stress_2", "pk1_stress_3"))
    return {
        "solution": "solution" in fields,
        "deformation_gradient": has_f,
        "stress_derived": has_f and (has_cauchy or has_pk1),
    }


