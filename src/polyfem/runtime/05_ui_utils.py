# =============================================================================
# IDs and small utilities
# =============================================================================



def _message(text, severity=None):
    """UI dialog in the GUI, stdout in hython/headless runs."""
    if hou.isUIAvailable():
        hou.ui.displayMessage(
            str(text), severity=severity or hou.severityType.Warning)
    else:
        print(f"[PolyFEM HDA] {text}")


def _status(text):
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(str(text), hou.severityType.ImportantMessage)
    else:
        print(f"[PolyFEM HDA] {text}")


def vol_id(geo, vol):
    """Collision-free volume/body id (geo, vol are 1-based)."""
    return 1000 * int(geo) + int(vol)


def boundary_id(geo, vol, sideset, bc):
    return vol_id(geo, vol) * 10000 + int(sideset) * 100 + int(bc)


def obstacle_id(geo):
    return 100000 + 1000 * int(geo)


def bool_check(var):
    return bool(var)


_TINYEXPR_NAMES = frozenset({
    "x", "y", "z", "t", "pi", "e",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "ln", "log", "log10",
    "pow", "sqrt", "abs", "ceil", "floor", "fac", "ncr", "npr",
    # PolyFEM custom helpers in src/polyfem/utils/ExpressionValue.cpp
    "min", "max", "smoothstep", "half_smoothstep", "deg2rad",
    "rotate_2D_x", "rotate_2D_y", "if", "compare", "smooth_abs", "sign",
})
_EXPR_IDENT_RE = re.compile(r"(?<![0-9.])\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _strip_outer_quotes(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _parse_expr_component(raw, name="value"):
    """Parse one axis/scalar entry: plain number -> int/float, expression
    string (with or without outer quotes) -> clean string, empty -> 0."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw
    text = str(raw).strip()
    if not text:
        return 0
    unquoted = _strip_outer_quotes(text)
    if not unquoted:
        return 0
    try:
        val = ast.literal_eval(unquoted)
        if isinstance(val, (int, float)) and not isinstance(val, bool) \
                and math.isfinite(float(val)):
            return val
    except (ValueError, SyntaxError):
        pass
    return unquoted


def _format_expr_box(value):
    """Format one JSON scalar/expression value for an unquoted UI box."""
    if value is None:
        return "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return _strip_outer_quotes(str(value)) or "0"


def _validate_tinyexpr(value, label, resource_dir=None):
    """Return an error message if a string expression has obvious TinyExpr
    syntax errors, or None when valid (or a numeric literal / file path)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return f"{label}: value must be finite (got {value})."
        return None
    if not isinstance(value, str):
        return None
    expr = _strip_outer_quotes(value)
    if not expr:
        return None
    # PolyFEM also allows a value-file path in expression fields.
    if (expr.startswith(("/", "./", "../", "~")) or "\\" in expr
            or re.match(r"^[A-Za-z]:/", expr)
            or expr.lower().endswith((".txt", ".csv", ".dat", ".npy", ".json"))
            or (resource_dir and os.path.isfile(os.path.join(resource_dir, expr)))):
        return None
    if "[" in expr or "]" in expr:
        return (f"{label}: '{expr}' contains brackets — enter one scalar "
                "value or expression per axis box without brackets or quotes.")
    if "**" in expr:
        return (f"{label}: '{expr}' uses '**' for power; PolyFEM's TinyExpr "
                "uses '^' or pow(a, b) instead.")
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return f"{label}: '{expr}' has an unmatched ')'."
    if depth != 0:
        return f"{label}: '{expr}' has an unclosed '('."
    literals = re.findall(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", expr)
    if any(not math.isfinite(float(number)) for number in literals):
        return f"{label}: numeric literals must be finite in '{expr}'."
    # Strip scientific-notation exponents (e.g. 1e-5, 2.5E+3) before checking
    # identifiers so 'e' inside a numeric literal isn't misread.
    cleaned = re.sub(r"(?<![A-Za-z0-9_])(\d+\.?\d*|\.\d+)[eE][+-]?\d+",
                     "0", expr)
    unknown = [tok for tok in _EXPR_IDENT_RE.findall(cleaned)
               if tok not in _TINYEXPR_NAMES]
    if unknown:
        bad = ", ".join(dict.fromkeys(unknown))
        return (f"{label}: unknown name(s) {bad} in '{expr}'. PolyFEM "
                "expressions allow x, y, z, t, pi, e, standard math "
                "(sin, cos, exp, pow, sqrt, abs, ...) and helpers "
                "(min, max, smoothstep, half_smoothstep, deg2rad, "
                "rotate_2D_x, rotate_2D_y, if, compare, smooth_abs, sign).")
    # TinyExpr's conditional helper is named 'if', a Python keyword.
    syntax = re.sub(r"\bif\s*(?=\()", "_if", expr).replace("^", "**")
    try:
        tree = ast.parse(syntax, mode="eval")
    except SyntaxError:
        return f"{label}: invalid TinyExpr syntax in '{expr}'."
    if any(isinstance(node, ast.Call) and not node.args and
           (not isinstance(node.func, ast.Name) or node.func.id not in {"e", "pi"})
           for node in ast.walk(tree)):
        return f"{label}: empty function call in '{expr}'."
    return None


def parse_vector(text, name):
    """Parse a user vector/scalar parm. Numeric literals via ast; entries that
    are expression strings (polyfem supports them) pass through unchanged,
    with optional outer quotes stripped."""
    text = text.strip()
    if not text:
        raise hou.NodeError(f"Empty value for {name}")
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            return [_parse_expr_component(item, name) for item in val]
        if isinstance(val, str):
            return _parse_expr_component(val, name)
        return val
    except (ValueError, SyntaxError):
        pass
    # allow ["0", "-0.1*t", "0"] or [0, -0.1*t, 0] lists with expressions,
    # splitting only on top-level commas (not inside quotes or function calls
    # such as if(t-0.5, -0.02, -0.04*t) or min(t, 1)).
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        parts, buf, depth, quote = [], [], 0, None
        for ch in inner:
            if quote is not None:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in ("'", '"'):
                quote = ch
                buf.append(ch)
            elif ch in "([{":
                depth += 1
                buf.append(ch)
            elif ch in ")]}":
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == "," and depth == 0:
                token = "".join(buf).strip()
                if token:
                    parts.append(token)
                buf = []
            else:
                buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return [_parse_expr_component(p, name) for p in parts]
    return _parse_expr_component(text, name)


def _read_expr_vector(parent, vec_base, legacy_name, label):
    """Read a 3-component expression vector from the 3-box UI parm, falling
    back to the hidden legacy JSON-string parm for older scenes / tests."""
    vec_tuple = parent.parmTuple(vec_base)
    legacy = parent.parm(legacy_name)
    if vec_tuple is not None and (not vec_tuple.isAtDefault() or
                                  parent.userData(f"polyfem_migrated_{vec_base}")):
        return [_parse_expr_component(p.eval(), label) for p in vec_tuple]
    if legacy is not None and not legacy.isAtDefault():
        parsed = parse_vector(legacy.eval(), label)
        if isinstance(parsed, list):
            return [_parse_expr_component(item, label) for item in parsed]
        return parsed
    if vec_tuple is not None:
        return [_parse_expr_component(p.eval(), label) for p in vec_tuple]
    if legacy is not None:
        return parse_vector(legacy.eval(), label)
    return [0, 0, 0]


def _set_expr_vector(parent, vec_base, legacy_name, value):
    """Populate both the visible 3-box parm and the hidden legacy JSON string."""
    if isinstance(value, str):
        try:
            parsed = parse_vector(value, vec_base)
        except hou.Error:
            parsed = [0, 0, 0]
    else:
        parsed = value
    items = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    items = [0 if item is None else item for item in items[:3]]
    while len(items) < 3:
        items.append(0)
    vec_tuple = parent.parmTuple(vec_base)
    if vec_tuple is not None:
        vec_tuple.set(tuple(_format_expr_box(v) for v in items[:3]))
        parent.setUserData(f"polyfem_migrated_{vec_base}", "1")
    legacy = parent.parm(legacy_name)
    if legacy is not None:
        legacy.set(_format_vector(items[:3]))


def _legacy_name_for_expr_vec(tuple_name):
    if tuple_name.startswith("bc_vec"):
        return "vector_" + tuple_name[len("bc_vec"):]
    if tuple_name.startswith("ic_vec"):
        return "condition_vector_" + tuple_name[len("ic_vec"):]
    if tuple_name.startswith("obstacle_disp_vec"):
        return "obstacle_disp" + tuple_name[len("obstacle_disp_vec"):]
    if tuple_name == "rhs_vec":
        return "RHS"
    return None


def sync_expr_vector(kwargs):
    """Callback on the 3-box expression parms: keep hidden legacy JSON parm
    in sync so resetting a box back to '0' never falls back to a stale value."""
    parent = kwargs["node"]
    parm = kwargs.get("parm")
    if parm is None:
        return
    vec_tuple = parm.tuple()
    legacy_name = _legacy_name_for_expr_vec(vec_tuple.name())
    if not legacy_name:
        return
    legacy = parent.parm(legacy_name)
    if legacy is None:
        return
    vals = [_parse_expr_component(p.eval(), vec_tuple.name())
            for p in vec_tuple]
    legacy.set(_format_vector(vals))
    parent.setUserData(f"polyfem_migrated_{vec_tuple.name()}", "1")


def _has_legacy_expression(parm):
    if parm.keyframes():
        return True
    if parm.parmTemplate().type() == hou.parmTemplateType.String:
        raw = parm.unexpandedString()
        if "$" in raw or "`" in raw or raw != parm.eval():
            return True
    try:
        return bool(parm.expression())
    except hou.OperationFailed:
        return False


def migrate_legacy_expr_vectors(parent):
    """Migrate literal vectors while leaving live legacy channels intact."""
    skipped = []

    def migrate(base, legacy_name):
        visible = parent.parmTuple(base)
        legacy = parent.parm(legacy_name)
        if visible is None or legacy is None or legacy.isAtDefault() \
                or parent.userData(f"polyfem_migrated_{base}"):
            return
        if not visible.isAtDefault():
            parent.setUserData(f"polyfem_migrated_{base}", "1")
            return
        if _has_legacy_expression(legacy):
            skipped.append(legacy_name)
            return
        _set_expr_vector(parent, base, legacy_name, legacy.eval())

    migrate("rhs_vec", "RHS")
    num_geos = parent.evalParm("num_geos") \
        if parent.parm("num_geos") is not None else 0
    for geo in range(1, num_geos + 1):
        migrate(f"obstacle_disp_vec{geo}", f"obstacle_disp{geo}")
        nvol_parm = parent.parm(f"num_volumes{geo}")
        num_vols = nvol_parm.eval() if nvol_parm is not None else 0
        for vol in range(1, num_vols + 1):
            ic_parm = parent.parm(f"initial_conditions{geo}_{vol}")
            for index in range(1, (ic_parm.eval() if ic_parm else 0) + 1):
                suffix = f"{geo}_{vol}_{index}"
                migrate(f"ic_vec{suffix}", f"condition_vector_{suffix}")
            ss_parm = parent.parm(f"sideset_selection{geo}_{vol}")
            for j in range(1, (ss_parm.eval() if ss_parm else 0) + 1):
                bc_parm = parent.parm(f"Boundary_Condition__{geo}_{vol}_{j}")
                for k in range(1, (bc_parm.eval() if bc_parm else 0) + 1):
                    suffix = f"{geo}_{vol}_{j}_{k}"
                    migrate(f"bc_vec{suffix}", f"vector_{suffix}")
            # Migrate legacy orthotropic raw elast_tensor to structured inputs
            mode_parm = parent.parm(f"elast_mode{geo}_{vol}")
            et_parm = parent.parm(f"elast_tensor{geo}_{vol}")
            if mode_parm is not None and et_parm is not None \
                    and mode_parm.isAtDefault() and not et_parm.isAtDefault() \
                    and _ortho_parms_at_default(parent, geo, vol):
                if _has_legacy_expression(et_parm):
                    skipped.append(et_parm.name())
                    continue
                vals = _numbers(et_parm.eval())
                if vals is not None and len(vals) == 9:
                    _set_ortho_parms(parent, geo, vol, vals)
                elif vals is not None:
                    mode_parm.set(1)  # raw mode (5 or 21 numbers)
    if skipped and not parent.cachedUserData("polyfem_legacy_expr_notice"):
        parent.setCachedUserData("polyfem_legacy_expr_notice", True)
        _status(f"{parent.path()}: kept animated or expression-driven legacy "
                "parameters active (" + ", ".join(skipped) + ").")
    migrate_legacy_time_mode(parent)


_TIME_MODE_FLAGS = {
    "tend_dt": (True, True, False),
    "tend_steps": (True, False, True),
    "dt_steps": (False, True, True),
}


def sync_time_mode(kwargs):
    """Callback for Define Run By menu: mirror selection into legacy bools."""
    parent = kwargs["node"]
    parm = parent.parm("define_run_by")
    if parm is None:
        return
    token = parm.evalAsString()
    flags = _TIME_MODE_FLAGS.get(token)
    if flags is None:
        return
    parent.setParms({
        "end_time_bool": int(flags[0]),
        "time_inc_bool": int(flags[1]),
        "num_timesteps_bool": int(flags[2]),
    })


def _time_flags(parent):
    """Return (tend_on, dt_on, steps_on), honoring Define Run By and legacy
    toggles alike."""
    tend_on = bool(parent.evalParm("end_time_bool"))
    dt_on = bool(parent.evalParm("time_inc_bool"))
    steps_on = bool(parent.evalParm("num_timesteps_bool"))
    mode_parm = parent.parm("define_run_by")
    if mode_parm is not None:
        token = mode_parm.evalAsString()
        if token != "tend_dt" and (tend_on, dt_on, steps_on) == (True, True, False):
            return _TIME_MODE_FLAGS.get(token, (tend_on, dt_on, steps_on))
    return tend_on, dt_on, steps_on


def migrate_legacy_time_mode(parent):
    """Sync Define Run By from legacy Time toggles on reopened scenes."""
    mode_parm = parent.parm("define_run_by")
    if mode_parm is None or not mode_parm.isAtDefault():
        return
    flags = (bool(parent.evalParm("end_time_bool")),
             bool(parent.evalParm("time_inc_bool")),
             bool(parent.evalParm("num_timesteps_bool")))
    if flags == (True, False, True):
        mode_parm.set(1)
    elif flags == (False, True, True):
        mode_parm.set(2)


_LENGTH_PER_METER = {
    "m": 1.0, "cm": 100.0, "mm": 1000.0, "um": 1e6,
    "in": 1.0 / 0.0254, "ft": 1.0 / 0.3048,
}
_SECONDS_PER_TIME = {
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "min": 60.0,
}


def set_gravity_y(kwargs):
    """Set Body Load to standard gravity along -Y (rhs = -a = +g in +Y),
    automatically scaled to Global > Units when Units is enabled."""
    parent = kwargs["node"]
    g = 9.81
    unit_note = "m/s^2"
    if parent.parm("units") is not None and parent.evalParm("units"):
        length = parent.evalParm("length").strip().lower()
        time_u = parent.evalParm("time").strip().lower()
        l_scale = _LENGTH_PER_METER.get(length)
        t_scale = _SECONDS_PER_TIME.get(time_u)
        if l_scale is not None and t_scale is not None:
            g = 9.81 * l_scale * (t_scale ** 2)
            unit_note = f"{length}/{time_u}^2"
        else:
            unit_note = f"{length or '?'}/{time_u or '?'}^2 (unrecognized unit; used 9.81)"
    g_clean = float(f"{g:.8g}")
    _set_expr_vector(parent, "rhs_vec", "RHS", [0, g_clean, 0])
    _status(
        f"Body Load set to [0, {g_clean:g}, 0] ({unit_note}, gravity pulling "
        "along -Y; PolyFEM uses rhs = -acceleration).")


def clear_body_load(kwargs):
    """Reset Body Load (rhs) to zero."""
    parent = kwargs["node"]
    _set_expr_vector(parent, "rhs_vec", "RHS", [0, 0, 0])
    _status("Body Load cleared to [0, 0, 0].")


