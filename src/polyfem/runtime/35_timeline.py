# =============================================================================
# Timeline and time curves (review F3, 2026-10-02)
#
# Houdini time is simulation time: t = Start Time + (T - T_start) * Time
# Scale, with T Houdini's time ($T, seconds at the scene's frame rate) and
# T_start the time of the Time tab's Start Frame. Keyframed condition
# multipliers and obstacle transforms are read on that timeline at the step
# times PolyFEM evaluates conditions at (t_k = t0 + k*dt, State.cpp) and
# exported as piecewise-linear interpolation tables, which are exact there.
# PolyFEM's interpolation multiplies a condition's value; vector conditions
# (Dirichlet, Neumann, obstacle motion) take a list of one function for all
# components or one per component, scalar ones (normal traction, pressure,
# cavity) a single function.
# =============================================================================

TIME_CURVE_TOKENS = ("entered", "ramp", "animated")


def simulation_steps(parent):
    """(t0, dt, steps): the run's time steps as PolyFEM derives them from
    the keys build_time writes (State.cpp). Raises hou.NodeError when the
    Time tab does not define a run."""
    t0 = float(parent.evalParm("t0"))
    tend = float(parent.evalParm("tend"))
    dt = float(parent.evalParm("dt"))
    steps = int(parent.evalParm("num_timesteps"))
    tend_on, dt_on, steps_on = _time_flags(parent)
    if int(tend_on) + int(dt_on) + int(steps_on) < 2:
        raise hou.NodeError(
            "The Time tab does not define a run: turn on two of End Time, "
            "Time Increment and # of Timesteps.")
    has_steps = steps_on and not (tend_on and dt_on)
    has_tend = tend_on
    has_dt = dt_on
    if has_tend and has_dt:
        if not (dt > 0 and tend > t0):
            raise hou.NodeError(
                f"The Time tab does not define a run: End Time ({tend:g}) "
                f"must be after Start Time ({t0:g}) and dt ({dt:g}) above 0.")
        # PolyFEM's rule (State.cpp, sdast9/polyfem 2026-11-14): a whole
        # number of steps up to roundoff is that number, not one more past
        # End Time ((0.14 - 0) / 0.02 = 7.000000000000001 is 7 steps).
        ratio = (tend - t0) / dt
        nearest = round(ratio)
        steps = int(nearest) if abs(ratio - nearest) <= 1e-12 * max(
            1.0, nearest) else int(math.ceil(ratio))
    elif has_tend and has_steps:
        if not (steps >= 1 and tend > t0):
            raise hou.NodeError(
                f"The Time tab does not define a run: End Time ({tend:g}) "
                f"must be after Start Time ({t0:g}) and the number of time "
                "steps at least 1.")
        dt = (tend - t0) / steps
    elif has_dt and has_steps:
        if not (dt > 0 and steps >= 1):
            raise hou.NodeError(
                f"The Time tab does not define a run: dt ({dt:g}) must be "
                "above 0 and the number of time steps at least 1.")
    else:
        raise hou.NodeError(
            "The Time tab does not define a run: turn on two of End Time, "
            "Time Increment and # of Timesteps.")
    return t0, dt, steps


class Timeline:
    """The run's step times and the Houdini times they fall on."""

    def __init__(self, t0, dt, steps, start_frame, scale):
        self.t0, self.dt, self.steps = float(t0), float(dt), int(steps)
        self.start_frame, self.scale = float(start_frame), float(scale)
        k = np.arange(self.steps + 1, dtype=np.float64)
        self.times = self.t0 + k * self.dt  # PolyFEM: t0 + dt * k
        self.start_time = hou.frameToTime(self.start_frame)
        self.houdini_times = self.start_time + (k * self.dt) / self.scale

    def houdini_time(self, t):
        """Houdini time (seconds) at which the simulation is at time t."""
        return self.start_time + (float(t) - self.t0) / self.scale

    def frame(self, t):
        return hou.timeToFrame(self.houdini_time(t))

    def record(self):
        """The mapping as written to the scene record (Read PVD, Import)."""
        return {"start_frame": self.start_frame, "time_scale": self.scale,
                "fps": hou.fps(), "t0": self.t0, "dt": self.dt,
                "time_steps": self.steps,
                "end_time": float(self.times[-1])}


def timeline(parent):
    scale = float(parent.evalParm("time_scale"))
    if not scale > 0:
        raise hou.NodeError(
            f"Time Scale ({scale:g}, Time tab) must be greater than 0.")
    t0, dt, steps = simulation_steps(parent)
    return Timeline(t0, dt, steps, parent.evalParm("houdini_start_frame"),
                    scale)


class _LazyTimeline:
    """The node's timeline, computed on first use (only animated curves and
    obstacles need it, and only they report a Time tab that defines no run)."""

    def __init__(self, parent):
        self.parent, self.value = parent, None

    def get(self):
        if self.value is None:
            self.value = timeline(self.parent)
        return self.value


def timeline_text(parent):
    """Time tab readout: which frames the run spans."""
    try:
        line = timeline(parent)
    except hou.Error as exc:
        return _error_text(exc)
    last = hou.timeToFrame(float(line.houdini_times[-1]))
    per_step = line.dt / line.scale * hou.fps()
    return (f"Frame {line.start_frame:g} = t {line.t0:g}; frame {last:.6g} = "
            f"t {line.times[-1]:.6g} (end); {line.steps} time steps, "
            f"{per_step:.4g} frames per step at {hou.fps():g} fps")


class TimeCurve:
    """A time function multiplying a condition's value, as PolyFEM's
    interpolation does: piecewise linear through (points, values), constant
    before the first point and after the last ("extend": "constant")."""

    def __init__(self, points, values):
        self.points = np.asarray(points, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    def at(self, times):
        return np.interp(np.asarray(times, dtype=np.float64), self.points,
                         self.values)

    def as_json(self):
        return {"type": "piecewise_linear",
                "points": [float(p) for p in self.points],
                "values": [float(v) for v in self.values],
                "extend": "constant"}


def _compressed_table(points, values):
    """Drop samples that lie on the line through their neighbours, keeping
    the first and last; the linear table through the rest reproduces every
    sample (checked, else the full table is kept)."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(p) <= 2:
        return p, v
    slopes = np.diff(v) / np.diff(p)
    size = float(np.abs(v).max())
    floor = size / max(float(p[-1] - p[0]), 1e-300)
    bend = np.abs(np.diff(slopes)) > 1e-9 * np.maximum(
        np.maximum(np.abs(slopes[1:]), np.abs(slopes[:-1])), floor)
    keep = np.concatenate(([True], bend, [True]))
    kept_p, kept_v = p[keep], v[keep]
    if np.allclose(np.interp(p, kept_p, kept_v), v, rtol=1e-12,
                   atol=1e-12 * size):
        return kept_p, kept_v
    return p, v


def _same_curve(first, second):
    """Whether two time functions (None = constant 1) are equal everywhere:
    piecewise-linear functions with constant ends are equal iff they agree
    at the union of their breakpoints."""
    if first is None and second is None:
        return True
    points = np.union1d(first.points if first is not None else [],
                        second.points if second is not None else [])
    a = np.ones(len(points)) if first is None else first.at(points)
    b = np.ones(len(points)) if second is None else second.at(points)
    return bool(np.allclose(a, b, rtol=1e-12, atol=1e-15))


def _vector_interpolation(curves, dims=None):
    """The interpolation list of a vector condition (None if no curve):
    [f] when every relevant component has the same function, else
    [fx, fy, fz] with {"type": "none"} for components without one."""
    relevant = [curve for axis, curve in enumerate(curves)
                if dims is None or dims[axis]]
    if all(curve is None for curve in relevant):
        return None
    first = relevant[0]
    if all(_same_curve(first, curve) for curve in relevant):
        return [first.as_json() if first is not None else {"type": "none"}]
    return [curve.as_json() if curve is not None else {"type": "none"}
            for curve in curves]


def _condition_curves(parent, suffix, kind, lazy_timeline, label):
    """The time function of every component of one condition (None =
    constant 1): three for Dirichlet/Neumann/Nodal Neumann, one for the scalar kinds."""
    width = 3 if kind in _VECTOR_BC_KINDS else 1
    parm = parent.parm(f"time_curve{suffix}")
    mode = parm.evalAsString() if parm is not None else "entered"
    if mode == "ramp":
        if parent.evalParm(f"ramp_custom{suffix}"):
            start = float(parent.evalParm(f"ramp_start{suffix}"))
            end = float(parent.evalParm(f"ramp_end{suffix}"))
        else:  # the whole run: 0 at the start, the value at the last step
            line = lazy_timeline.get()
            start, end = line.t0, float(line.times[-1])
        if not end > start:
            raise hou.NodeError(
                f"{label}: Ramp End ({end:g}) must be later than Ramp Start "
                f"({start:g}).")
        return [TimeCurve([start, end], [0.0, 1.0])] * width
    if mode != "animated":
        return [None] * width
    line = lazy_timeline.get()
    parms = list(parent.parmTuple(f"curve_multiplier{suffix}")) \
        if width == 3 else [parent.parm(f"curve_scale{suffix}")]
    curves = []
    for component in parms:
        values = np.array([component.evalAtTime(float(T))
                           for T in line.houdini_times], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise hou.NodeError(
                f"{label}: the Multiplier ({component.name()}) is not a finite "
                "number at every time step.")
        if np.all(values == 1.0):
            curves.append(None)  # constant 1: no function needed
            continue
        curves.append(TimeCurve(*_compressed_table(line.times, values)))
    return curves


def _keyframes_record(parm):
    """A parameter's animation for the sideset/scene records: its keyframes
    (exact, through Houdini's own JSON form) and its current value."""
    record = {"value": parm.eval()}
    keys = parm.keyframes()
    if keys:
        record["keys"] = [key.asJSON() for key in keys]
    return record


def _restore_keyframes(parm, record):
    """Inverse of _keyframes_record."""
    if parm is None or not isinstance(record, dict):
        return
    parm.deleteAllKeyframes()
    keys = record.get("keys")
    if keys:
        restored = []
        for data in keys:
            key = hou.Keyframe()
            key.fromJSON(data)
            restored.append(key)
        parm.setKeyframes(restored)
    elif "value" in record:
        parm.set(record["value"])


def _curve_record(parent, suffix, kind):
    """What the sideset record keeps of a condition's time curve, so Import
    restores it exactly (keyframes, ramp times or their defaults)."""
    parm = parent.parm(f"time_curve{suffix}")
    mode = parm.evalAsString() if parm is not None else "entered"
    record = {"time_curve": mode}
    if mode == "ramp":
        record["ramp"] = {
            name: parent.evalParm(f"{name}{suffix}")
            for name in ("ramp_custom", "ramp_start", "ramp_end")}
    elif mode == "animated":
        parms = list(parent.parmTuple(f"curve_multiplier{suffix}")) \
            if kind in _VECTOR_BC_KINDS else [parent.parm(f"curve_scale{suffix}")]
        record["multiplier"] = [_keyframes_record(p) for p in parms]
    return record


def _restore_curve_record(parent, suffix, kind, record):
    """Inverse of _curve_record (missing fields: Value as entered)."""
    parm = parent.parm(f"time_curve{suffix}")
    if parm is None:
        return
    mode = record.get("time_curve", "entered") \
        if isinstance(record, dict) else "entered"
    if mode not in TIME_CURVE_TOKENS:
        mode = "entered"
    parm.set(TIME_CURVE_TOKENS.index(mode))
    if mode == "ramp":
        for name, value in (record.get("ramp") or {}).items():
            ramp = parent.parm(f"{name}{suffix}")
            if ramp is not None and value is not None:
                ramp.deleteAllKeyframes()
                ramp.set(value)
    elif mode == "animated":
        parms = list(parent.parmTuple(f"curve_multiplier{suffix}")) \
            if kind in _VECTOR_BC_KINDS else [parent.parm(f"curve_scale{suffix}")]
        for component, data in zip(parms, record.get("multiplier") or []):
            _restore_keyframes(component, data)


def _interpolation_functions(interpolation, width, line=None):
    """PolyFEM interpolation json -> per-component functions (None = 1), or
    (functions, note) where the note says what was approximated. A table
    other than piecewise_linear with constant ends is sampled at the step
    times of `line` (exact where PolyFEM evaluates conditions)."""
    if interpolation is None:
        return [None] * width, None
    items = interpolation if isinstance(interpolation, list) \
        else [interpolation]
    if not items:
        return [None] * width, None
    if len(items) == 1:
        items = items * width
    functions, notes = [], []
    for item in items[:width]:
        kind = item.get("type", "none") if isinstance(item, dict) else "none"
        if kind == "piecewise_linear":
            try:
                p = np.asarray(item.get("points"), dtype=float)
                v = np.asarray(item.get("values"), dtype=float)
                if (p.ndim != 1 or p.size == 0 or p.shape != v.shape or
                        not np.isfinite(p).all() or not np.isfinite(v).all() or
                        np.any(np.diff(p) <= 0)):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                raise hou.NodeError(
                    "piecewise_linear interpolation needs equal-length, nonempty "
                    "finite points and values, with strictly increasing points.")
        if kind == "none":
            functions.append(None)
            continue
        if kind == "piecewise_linear" \
                and item.get("extend", "constant") == "constant":
            functions.append(TimeCurve(item.get("points", []),
                                       item.get("values", [])))
            continue
        if line is None:
            raise hou.NodeError(f"time function '{kind}' needs the run's "
                                "time steps")
        functions.append(TimeCurve(line.times,
                                   _evaluate_interpolation(item, line.times)))
        notes.append(kind)
    note = None
    if notes:
        note = ("its time function ({}) was turned into keyframes at the "
                "time steps, exact where PolyFEM evaluates it".format(
                    ", ".join(sorted(set(notes)))))
    functions.extend([None] * (width - len(functions)))
    return functions, note


def _cubic_coefficients(points, values, extend):
    """PolyFEM's PiecewiseCubicInterpolation::init: one cubic per segment
    (rows = segments, columns a, b, c, d of a t^3 + b t^2 + c t + d)."""
    n = len(points) - 1
    a = np.zeros((4 * n, 4 * n))
    b = np.zeros(4 * n)
    for i in range(n):
        for j in range(2):
            t = points[i + j]
            a[2 * i + j, 4 * i:4 * i + 4] = (t ** 3, t ** 2, t, 1.0)
            b[2 * i + j] = values[i + j]
    row = 2 * n
    for i in range(n - 1):
        t = points[i + 1]
        a[row + i, 4 * i:4 * i + 3] = (3 * t ** 2, 2 * t, 1.0)
        a[row + i, 4 * (i + 1):4 * (i + 1) + 3] = (-3 * t ** 2, -2 * t, -1.0)
    row += n - 1
    for i in range(n - 1):
        t = points[i + 1]
        a[row + i, 4 * i:4 * i + 2] = (6 * t, 2.0)
        a[row + i, 4 * (i + 1):4 * (i + 1) + 2] = (-6 * t, -2.0)
    row += n - 1
    t0, tn, last = points[0], points[n], 4 * (n - 1)
    if extend == "constant":
        a[row, 0:3] = (3 * t0 ** 2, 2 * t0, 1.0)
        a[row + 1, last:last + 3] = (3 * tn ** 2, 2 * tn, 1.0)
    elif extend == "extrapolate":
        a[row, 0:2] = (6 * t0, 2.0)
        a[row + 1, last:last + 2] = (6 * tn, 2.0)
    else:
        a[row, 0:3] = (3 * t0 ** 2, 2 * t0, 1.0)
        a[row, last:last + 3] = (-3 * tn ** 2, -2 * tn, -1.0)
        a[row + 1, 0:2] = (6 * t0, 2.0)
        a[row + 1, last:last + 2] = (-6 * tn, -2.0)
    return np.linalg.solve(a, b).reshape(n, 4)


def _evaluate_interpolation(item, times):
    """PolyFEM's Interpolation::eval (utils/Interpolation.cpp), for the time
    functions an imported params.json may hold."""
    times = np.asarray(times, dtype=np.float64)
    kind = item.get("type", "none")
    if kind == "none":
        return np.ones_like(times)
    if kind == "linear":
        return times.copy()
    if kind == "linear_ramp":
        start, end = float(item.get("from", 0.0)), float(item["to"])
        return np.array([end - start if t >= end else 0.0 if t <= start
                         else t - start for t in times])
    points = [float(p) for p in item.get("points", [])]
    values = [float(v) for v in item.get("values", [])]
    extend = item.get("extend", "constant")
    if not points:
        return np.ones_like(times)
    n = len(points) - 1
    coefficients = _cubic_coefficients(points, values, extend) \
        if kind == "piecewise_cubic" and n >= 1 else None

    def piece(t, i):
        if kind == "piecewise_constant":
            return values[i]
        if kind == "piecewise_cubic":
            c = coefficients[i]
            return ((c[0] * t + c[1]) * t + c[2]) * t + c[3]
        alpha = (t - points[i]) / (points[i + 1] - points[i])
        return (values[i + 1] - values[i]) * alpha + values[i]

    def slope(t, i):
        if kind == "piecewise_constant":
            return 0.0
        if kind == "piecewise_cubic":
            c = coefficients[i]
            return (3 * c[0] * t + 2 * c[1]) * t + c[2]
        return (values[i + 1] - values[i]) / (points[i + 1] - points[i])

    def dy_dt(t):
        for i in range(n):
            if points[i] <= t <= points[i + 1]:
                return slope(t, i)
        return 0.0

    def evaluate(t):
        if t < points[0] or t >= points[-1]:
            return extended(t)
        for i in range(n):
            if points[i] <= t < points[i + 1]:
                return piece(t, i)
        return values[0]

    def extended(t):
        t0, tn, y0, yn = points[0], points[-1], values[0], values[-1]
        if extend == "extrapolate":
            return dy_dt(t0) * (t - t0) + y0 if t < t0 \
                else dy_dt(tn) * (t - tn) + yn
        if extend in ("repeat", "repeat_offset") and tn > t0:
            offset = math.floor((t - t0) / (tn - t0)) * (yn - y0) \
                if extend == "repeat_offset" else 0.0
            mod = math.fmod(t - t0, tn - t0)
            mod += tn if mod < 0 else t0
            return evaluate(mod) + offset
        return y0 if t < t0 else yn

    return np.array([evaluate(float(t)) for t in times])


def _set_curve_from_functions(parent, suffix, kind, functions, line):
    """Show imported time functions on a condition: no function -> Value as
    entered; the 0 -> 1 ramp the asset writes -> Ramp; anything else ->
    Animated multiplier with linear keys at the table's points."""
    parm = parent.parm(f"time_curve{suffix}")
    if parm is None:
        return
    width = 3 if kind in _VECTOR_BC_KINDS else 1
    functions = (list(functions) + [None] * width)[:width]
    if all(function is None for function in functions):
        parm.set(TIME_CURVE_TOKENS.index("entered"))
        return
    first = functions[0]
    if first is not None and all(_same_curve(first, f) for f in functions) \
            and len(first.points) == 2 and first.values[0] == 0.0 \
            and first.values[1] == 1.0:
        parm.set(TIME_CURVE_TOKENS.index("ramp"))
        whole_run = np.allclose(first.points, [line.t0, line.times[-1]],
                                rtol=1e-12, atol=1e-15)
        parent.parm(f"ramp_custom{suffix}").set(0 if whole_run else 1)
        for name, value in (("ramp_start", first.points[0]),
                            ("ramp_end", first.points[1])):
            ramp = parent.parm(f"{name}{suffix}")
            ramp.deleteAllKeyframes()
            ramp.set(float(value))
        return
    parm.set(TIME_CURVE_TOKENS.index("animated"))
    parms = list(parent.parmTuple(f"curve_multiplier{suffix}")) \
        if width == 3 else [parent.parm(f"curve_scale{suffix}")]
    for component, function in zip(parms, functions):
        component.deleteAllKeyframes()
        if function is None:
            component.set(1.0)
            continue
        keys = []
        for point, value in zip(function.points, function.values):
            key = hou.Keyframe()
            key.setFrame(line.frame(point))
            key.setValue(float(value))
            key.setExpression("linear()", hou.exprLanguage.Hscript)
            keys.append(key)
        component.setKeyframes(keys)


def _sideset_conditions(parent, geo, vol, sideset, lazy_timeline=None):
    """The conditions entered on one sideset, parsed, in interface order.

    Each carries "curves": the time function of every component (None =
    constant 1; see _condition_curves)."""
    if lazy_timeline is None:
        lazy_timeline = _LazyTimeline(parent)
    conditions = []
    count = parent.evalParm(f"Boundary_Condition__{geo}_{vol}_{sideset}")
    for k in range(1, count + 1):
        suffix = f"{geo}_{vol}_{sideset}_{k}"
        kind = parent.evalParm(f"boundary_type{suffix}")
        if not 0 <= kind < len(_BC_TYPE_NAMES):
            raise hou.NodeError(f"Unknown boundary type {kind} on {suffix}")
        label = (f"geo {geo} subdomain {vol} sideset {sideset} "
                 f"condition {k} ({_BC_TYPE_NAMES[kind]})")
        if kind in _VECTOR_BC_KINDS:
            parsed_val = _read_expr_vector(
                parent, f"bc_vec{suffix}", f"vector_{suffix}",
                f"BC {geo}/{vol}/{sideset}/{k}")
            vector_text = _format_vector(parsed_val)
            value_text = parent.evalParm(f"value_{suffix}")
        else:
            vector_text = _format_vector(_read_expr_vector(
                parent, f"bc_vec{suffix}", f"vector_{suffix}",
                f"BC {geo}/{vol}/{sideset}/{k}"))
            value_text = parent.evalParm(f"value_{suffix}")
            parsed_val = parse_vector(value_text, f"BC {geo}/{vol}/{sideset}/{k}")
        condition = {
            "k": k, "type": kind, "label": label,
            "value": parsed_val,
            "curves": _condition_curves(parent, suffix, kind, lazy_timeline,
                                        label),
            "record": {"type": int(kind), "vector": vector_text,
                       "value": value_text}}
        condition["record"].update(_curve_record(parent, suffix, kind))
        if kind == 0:
            condition["dims"] = [
                bool(parent.evalParm(f"{axis}_dimension{suffix}"))
                for axis in "xyz"]
            condition["record"]["dimension"] = [
                int(flag) for flag in condition["dims"]]
        conditions.append(condition)
    return conditions


