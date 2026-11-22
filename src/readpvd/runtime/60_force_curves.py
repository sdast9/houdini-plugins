# =============================================================================
# Force curves (review F4, 2026-10-02)
#
# With Nodal Forces on, PolyFEM writes every node's force of every term of
# the energy ("<term>_forces" = -dE/du) on the volume block, once per element
# the node belongs to. At equilibrium the terms balance at every free node;
# where the displacement is prescribed the imbalance is what holds the node
# there. So, over the nodes of a set (each counted once):
#   reaction = -(sum of every term): the force the prescribed displacement --
#              or an obstacle's prescribed motion -- applies there (zero where
#              nothing is prescribed);
#   contact  = sum of the contact and friction forces on those nodes.
# For an obstacle the set is its own output rows (body id 0): contact is the
# force the body applies to the obstacle, and the reaction the force that
# drives the obstacle along its path. The sets (sideset faces and obstacle
# surfaces in simulation coordinates) come from the PolyFEM node's scene
# record. Adhesion and Rayleigh-damping forces are not written by PolyFEM
# and so are not included.
# =============================================================================

FORCE_TERMS = ("elastic", "inertia", "body", "contact", "friction", "damping",
               "pressure", "strain_augmented_lagrangian_lagr",
               "periodic_contact")
FORCE_CSV = "force_curves.csv"
FORCE_QUANTITIES = ("reaction", "contact")
FORCE_COMPONENTS = ("x", "y", "z", "magnitude")
FORCE_AGAINST = ("displacement", "time")
_FORCE_NODES_CACHE = OrderedDict()
_FORCE_CURVES = {}
_CHART_WINDOWS = {}


class ForceCurveError(Exception):
    """Why force curves cannot be computed (shown to the user as is)."""


def _force_definitions(node):
    record = scene_record(node)
    if record is None:
        raise ForceCurveError(
            "Force curves need the PolyFEM node's record of the run "
            f"(input/{SCENE_RECORD_FILE} next to the output folder), which "
            "PolyFEM (Dev) 2.0 writes with every export since 2026-10-02: "
            "write or run the scene again.")
    sets = record.get("force_sets") or []
    if not sets:
        raise ForceCurveError(
            "The run has no sideset or obstacle to measure: add a sideset (a "
            "selection without a condition is enough) and write the scene "
            "again.")
    return record, sets


def force_set_menu(kwargs):
    """Toggle/replace menu of the record's force sets (1-based tokens)."""
    try:
        _, sets = _force_definitions(kwargs["node"])
    except ForceCurveError:
        return []
    items = []
    for index, entry in enumerate(sets, 1):
        items.extend((str(index), entry.get("label", f"Set {index}")))
    return items


def _selected_force_sets(node, count):
    text = node.evalParm("force_sets").split()
    chosen = sorted({int(token) for token in text if token.isdigit()
                     and 1 <= int(token) <= count})
    return chosen or list(range(1, count + 1))


def _face_lattice(vertices, faces, order):
    """Positions of the Lagrange nodes of straight faces of a given order:
    the corners, plus edge and face nodes from order 2 on."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(len(faces), -1)
    points = [vertices]
    if order < 2 or not len(faces):
        return vertices
    if faces.shape[1] < 4:
        faces = np.column_stack(
            (faces, np.full((len(faces), 4 - faces.shape[1]), -1)))
    triangles = faces[faces[:, 3] < 0][:, :3]
    quads = faces[faces[:, 3] >= 0]
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            if max(i, j, k) == order or not len(triangles):
                continue  # a corner
            points.append((i * vertices[triangles[:, 0]]
                           + j * vertices[triangles[:, 1]]
                           + k * vertices[triangles[:, 2]]) / order)
    for a in range(order + 1):
        for b in range(order + 1):
            if a in (0, order) and b in (0, order) or not len(quads):
                continue
            u, v = a / order, b / order
            points.append((1 - u) * (1 - v) * vertices[quads[:, 0]]
                          + u * (1 - v) * vertices[quads[:, 1]]
                          + u * v * vertices[quads[:, 2]]
                          + (1 - u) * v * vertices[quads[:, 3]])
    return np.concatenate(points)


def _force_nodes(record_key, mesh, definitions, chosen):
    """{set: (output points of its nodes, nodes looked for, not found)}:
    each node once (a representative of its coincidence group), matched by
    position within the set's body. Cached per record and rest geometry."""
    points = np.ascontiguousarray(mesh["points"], dtype=np.float64)
    body = mesh["point_data"].get("body_ids")
    if body is not None:
        body = np.rint(np.asarray(body, dtype=np.float64).reshape(
            len(points), -1)[:, 0])
    key = (record_key, str(mesh.get("topo_key")), tuple(chosen),
           _rest_geometry_digest(mesh))
    cached = _FORCE_NODES_CACHE.get(key)
    if cached is not None:
        _FORCE_NODES_CACHE.move_to_end(key)
        return cached
    _, representative = _coincidence_groups(points, body)
    reps = np.flatnonzero(representative)
    extent = float(np.linalg.norm(np.ptp(points, axis=0))) or 1.0
    tolerance = 1e-6 * extent
    result = {}
    for index in chosen:
        entry = definitions[index - 1]
        vertices = np.asarray(entry.get("vertices", []),
                              dtype=np.float64).reshape(-1, 3)
        if entry.get("kind") == "faces":
            targets = _face_lattice(vertices, entry.get("faces", []),
                                    int(entry.get("order", 1)))
        else:
            targets = vertices
        candidates = reps if body is None \
            else reps[body[reps] == int(entry.get("body", -1))]
        if not len(candidates) or not len(targets):
            result[index] = (np.zeros(0, dtype=np.int64), len(targets),
                             len(targets))
            continue
        nearest = _nearest_rows(points[candidates], targets)
        distance = np.linalg.norm(points[candidates[nearest]] - targets,
                                  axis=1)
        found = distance <= tolerance
        result[index] = (np.unique(candidates[nearest[found]]), len(targets),
                         int((~found).sum()))
    _FORCE_NODES_CACHE[key] = result
    if len(_FORCE_NODES_CACHE) > 16:
        _FORCE_NODES_CACHE.popitem(last=False)
    return result


def _point_vectors(mesh, name):
    values = mesh["point_data"].get(name)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    count = len(mesh["points"])
    values = values[:, None] if values.ndim == 1 else values[:, :3]
    if values.shape != (count, 3):  # also zero-fill omitted obstacle rows
        padded = np.zeros((count, 3))
        padded[:min(len(values), count), :values.shape[1]] = values[:count]
        values = padded
    return values


_MISSING_FORCE_FIELDS = (
    "This run has no nodal forces: turn on Output > Nodal Forces "
    "(force curves) on the PolyFEM node and run it again.")


def _frame_set_values(mesh, nodes):
    """{set: (reaction, contact, mean displacement)} of one frame."""
    terms = [_point_vectors(mesh, f"{term}_forces") for term in FORCE_TERMS]
    terms = [values for values in terms if values is not None]
    if not terms:
        return None
    total = np.sum(terms, axis=0)
    contact = [_point_vectors(mesh, name)
               for name in ("contact_forces", "friction_forces")]
    contact = [values for values in contact if values is not None]
    contact = np.sum(contact, axis=0) if contact else np.zeros_like(total)
    displacement = _point_vectors(mesh, "solution")
    if displacement is None:
        displacement = np.zeros_like(total)
    out = {}
    for index, (members, _, _) in nodes.items():
        if not len(members):
            out[index] = (np.full(3, np.nan),) * 3
            continue
        out[index] = (-total[members].sum(axis=0),
                      contact[members].sum(axis=0),
                      displacement[members].mean(axis=0))
    return out


def _volume_block(blocks):
    mesh = blocks.get("Volume")
    if mesh is None:
        raise ForceCurveError(
            "This frame has no Volume block, where PolyFEM writes nodal "
            "forces.")
    return mesh


def force_curves(node):
    """Every step's reaction, contact force and mean displacement of the
    chosen force sets: {"steps", "times", "sets", "labels", "reaction",
    "contact", "displacement" (steps x sets x 3), "nodes", "unmatched"}.
    Memoized on the PVD, the record and the choice of sets."""
    path = node.evalParm("PVD_file")
    if not path or not os.path.isfile(path):
        raise ForceCurveError("Load a PVD file first.")
    record, definitions = _force_definitions(node)
    record_key = _file_key(scene_record_path(path))
    chosen = _selected_force_sets(node, len(definitions))
    entries = read_pvd(path)
    memo_key = (_file_key(path), record_key, tuple(chosen), len(entries))
    memo = _FORCE_CURVES.get(node.sessionId())
    if memo is not None and memo["key"] == memo_key:
        return memo
    reaction = np.full((len(entries), len(chosen), 3), np.nan)
    contact = np.full_like(reaction, np.nan)
    displacement = np.full_like(reaction, np.nan)
    nodes = {}
    found_forces = False
    for step in range(len(entries)):
        try:
            mesh = _volume_block(load_frame(path, step, blocks=("Volume",)))
        except ForceCurveError:
            raise
        except Exception:
            continue  # an unreadable step (a run still writing) stays NaN
        nodes = _force_nodes(record_key, mesh, definitions, chosen)
        values = _frame_set_values(mesh, nodes)
        if values is None:
            continue
        found_forces = True
        for column, index in enumerate(chosen):
            reaction[step, column], contact[step, column], \
                displacement[step, column] = values[index]
    if not found_forces:
        raise ForceCurveError(_MISSING_FORCE_FIELDS)
    memo = {"key": memo_key, "steps": np.arange(len(entries)),
            "times": np.array([entry[0] for entry in entries]),
            "sets": chosen,
            "labels": [definitions[i - 1].get("label", f"Set {i}")
                       for i in chosen],
            "kinds": [definitions[i - 1].get("kind") for i in chosen],
            "reaction": reaction, "contact": contact,
            "displacement": displacement,
            "nodes": {i: len(nodes.get(i, ([],))[0]) for i in chosen},
            "unmatched": {i: nodes.get(i, (None, 0, 0))[2] for i in chosen}}
    _FORCE_CURVES[node.sessionId()] = memo
    return memo


def _magnitude(vectors):
    return np.linalg.norm(vectors, axis=-1)


def write_force_csv(curves, path):
    """One row per step: time, then per set its reaction, contact force and
    mean displacement (x, y, z, magnitude)."""
    import csv
    header = ["step", "time"]
    for label in curves["labels"]:
        for quantity in ("reaction", "contact force", "displacement"):
            header.extend(f"{label}: {quantity} {axis}"
                          for axis in FORCE_COMPONENTS)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(len(curves["steps"])):
            row = [int(curves["steps"][step]), repr(float(curves["times"][step]))]
            for column in range(len(curves["sets"])):
                for name in ("reaction", "contact", "displacement"):
                    vector = curves[name][step, column]
                    row.extend(repr(float(v)) for v in vector)
                    row.append(repr(float(_magnitude(vector))))
            writer.writerow(row)
    return path


def force_csv_path(node):
    return os.path.join(os.path.dirname(os.path.abspath(
        node.evalParm("PVD_file"))), FORCE_CSV)


def compute_force_curves(kwargs):
    """Compute Force Curves button: every step, written to the CSV."""
    node = kwargs["node"]
    try:
        curves = force_curves(node)
    except ForceCurveError as exc:
        _message(str(exc))
        return None
    path = force_csv_path(node)
    try:
        write_force_csv(curves, path)
    except OSError as exc:
        _message(f"Could not write {path}: {exc}")
        return curves
    missing = [f"{label}: {curves['unmatched'][index]} nodes not found"
               for index, label in zip(curves["sets"], curves["labels"])
               if curves["unmatched"][index]]
    note = ("\n" + "\n".join(missing)) if missing else ""
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(
            f"Force curves of {len(curves['sets'])} set(s) over "
            f"{len(curves['steps'])} steps written to {path}",
            hou.severityType.ImportantMessage)
    if note:
        _message("Some nodes of the force sets were not found in the "
                 "results (the mesh or the record changed since the run?):"
                 + note)
    _update_chart(node)
    return curves


def _vector_text(vector):
    return "(" + ", ".join(f"{float(v):.5g}" for v in vector) + ")"


_FORCE_FRAME_CACHE = OrderedDict()
_FORCE_FRAME_CACHE_MAX = 64


def force_readout_text(node):
    """Read-only Force Curves readout: each set's forces at this frame."""
    hou.frame()  # time-dependent, so it follows the playbar
    path = node.evalParm("PVD_file")
    if not path:
        return "Load a PVD file."
    try:
        record, definitions = _force_definitions(node)
        chosen = _selected_force_sets(node, len(definitions))
        entries = read_pvd(path)
        step = max(0, min(entry_index(node), len(entries) - 1))
        pvd_key = _file_key(path)
        record_key = _file_key(scene_record_path(path))
        memo_key = (pvd_key, record_key, tuple(chosen), len(entries))
        values = None
        memo = _FORCE_CURVES.get(node.sessionId())
        if memo is not None and memo["key"] == memo_key \
                and 0 <= step < len(memo["steps"]) \
                and np.all(np.isfinite(memo["reaction"][step])):
            values = {
                index: (memo["reaction"][step, col],
                        memo["contact"][step, col],
                        memo["displacement"][step, col])
                for col, index in enumerate(chosen)
            }
        else:
            step_key = (pvd_key, record_key, step, tuple(chosen))
            cached = _FORCE_FRAME_CACHE.get(step_key)
            if cached is not None:
                _FORCE_FRAME_CACHE.move_to_end(step_key)
                values = cached
            else:
                mesh = _volume_block(load_frame(path, step, blocks=("Volume",)))
                nodes = _force_nodes(record_key, mesh, definitions, chosen)
                values = _frame_set_values(mesh, nodes)
                if values is None:
                    return "Nodal Forces are unavailable at this step."
                _FORCE_FRAME_CACHE[step_key] = values
                while len(_FORCE_FRAME_CACHE) > _FORCE_FRAME_CACHE_MAX:
                    _FORCE_FRAME_CACHE.popitem(last=False)
    except ForceCurveError as exc:
        return str(exc)
    except Exception as exc:
        return f"Force readout unavailable: {exc}"
    lines = []
    for index in chosen:
        reaction, contact, displacement = values[index]
        label = definitions[index - 1].get("label", f"Set {index}")
        lines.append(
            f"{label}: reaction {_vector_text(reaction)} |{float(_magnitude(reaction)):.5g}|, "
            f"contact {_vector_text(contact)}, displacement "
            f"{_vector_text(displacement)}")
    return "\n".join(lines)


def _chart_series(node, curves):
    """(x, y, x label, y label, title) of the chosen chart."""
    token = node.evalParm("force_chart_set").strip()
    try:
        column = curves["sets"].index(int(token))
    except (ValueError, TypeError):
        if token and node.evalParm("force_sets").strip():
            available = ", ".join(str(s) for s in curves["sets"])
            raise ForceCurveError(
                f"Chart set '{token}' is not in the computed Force Sets "
                f"({available}). Clear Force Sets or include '{token}' and "
                "press Compute Force Curves.")
        column = 0
    quantity = _menu_token(node, "force_chart_quantity")
    component = _menu_token(node, "force_chart_component")
    against = _menu_token(node, "force_chart_against")
    vectors = curves[quantity][:, column]
    if component == "magnitude":
        y = _magnitude(vectors)
    else:
        y = vectors[:, "xyz".index(component)]
    if against == "time":
        x, x_label = curves["times"], "time"
    else:
        moved = curves["displacement"][:, column]
        x = _magnitude(moved) if component == "magnitude" \
            else moved[:, "xyz".index(component)]
        x_label = f"mean displacement {component}"
    name = "reaction" if quantity == "reaction" else "contact force"
    return (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
            x_label, f"{name} {component}", curves["labels"][column])


def _chart_class():
    from PySide6 import QtCore, QtGui, QtWidgets

    class ForceChart(QtWidgets.QWidget):
        """A small line chart of one force curve; the marker follows the
        playbar."""

        def __init__(self, node, parent=None):
            super().__init__(parent, QtCore.Qt.Window)
            self.node_id = node.sessionId()
            self.series = None
            self.marker = None
            self.setWindowTitle(f"Force Curve - {node.name()}")
            self.resize(640, 420)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._follow)
            self.timer.start(200)
            self._frame = None

        def set_series(self, series):
            self.series = series
            self.update()

        def _follow(self):
            node = hou.nodeBySessionId(self.node_id)
            if node is None:
                self.close()
                return
            frame = hou.frame()
            if frame != self._frame:
                self._frame = frame
                self.update()

        def closeEvent(self, event):
            self.timer.stop()
            _CHART_WINDOWS.pop(self.node_id, None)
            super().closeEvent(event)

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.fillRect(self.rect(), self.palette().window())
            draw_force_chart(painter, self.rect(), self.series,
                             self._marker_index(), self.palette())
            painter.end()

        def _marker_index(self):
            node = hou.nodeBySessionId(self.node_id)
            if node is None or self.series is None or not len(self.series[0]):
                return None
            try:
                return max(0, min(entry_index(node), len(self.series[0]) - 1))
            except Exception:
                return None

    return ForceChart


def draw_force_chart(painter, rect, series, marker, palette=None):
    """Draw axes, ticks, the curve and the marker (step `marker`) with a
    QPainter; also used to render the chart headlessly in tests."""
    from PySide6 import QtCore, QtGui
    text = palette.text().color() if palette is not None \
        else QtGui.QColor(30, 30, 30)
    if series is None:
        painter.setPen(text)
        painter.drawText(rect, QtCore.Qt.AlignCenter,
                         "Compute Force Curves first.")
        return
    x, y, x_label, y_label, title = series
    valid = np.isfinite(x) & np.isfinite(y)
    left, top, right, bottom = 70, 34, 18, 46
    area = QtCore.QRectF(rect.left() + left, rect.top() + top,
                         rect.width() - left - right,
                         rect.height() - top - bottom)
    painter.setPen(text)
    painter.drawText(QtCore.QRectF(rect.left(), rect.top() + 6, rect.width(),
                                   20), QtCore.Qt.AlignCenter, title)
    if not valid.any():
        painter.drawText(area, QtCore.Qt.AlignCenter, "No values.")
        return

    def span(values):
        low, high = float(values.min()), float(values.max())
        if high - low <= 1e-300:
            pad = abs(low) * 0.05 or 1.0
            return low - pad, high + pad
        pad = 0.05 * (high - low)
        return low - pad, high + pad

    x_low, x_high = span(x[valid])
    y_low, y_high = span(y[valid])

    def to_screen(px, py):
        return QtCore.QPointF(
            area.left() + (px - x_low) / (x_high - x_low) * area.width(),
            area.bottom() - (py - y_low) / (y_high - y_low) * area.height())

    grid = QtGui.QColor(text)
    grid.setAlpha(60)
    for i in range(6):
        fx = x_low + (x_high - x_low) * i / 5
        fy = y_low + (y_high - y_low) * i / 5
        painter.setPen(grid)
        painter.drawLine(to_screen(fx, y_low), to_screen(fx, y_high))
        painter.drawLine(to_screen(x_low, fy), to_screen(x_high, fy))
        painter.setPen(text)
        point = to_screen(fx, y_low)
        painter.drawText(QtCore.QRectF(point.x() - 40, area.bottom() + 4, 80,
                                       16), QtCore.Qt.AlignCenter,
                         f"{fx:.3g}")
        point = to_screen(x_low, fy)
        painter.drawText(QtCore.QRectF(rect.left(), point.y() - 8, left - 6,
                                       16),
                         QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                         f"{fy:.3g}")
    painter.setPen(text)
    painter.drawRect(area)
    painter.drawText(QtCore.QRectF(area.left(), rect.bottom() - 22,
                                   area.width(), 18), QtCore.Qt.AlignCenter,
                     x_label)
    painter.save()
    painter.translate(rect.left() + 12, area.center().y())
    painter.rotate(-90)
    painter.drawText(QtCore.QRectF(-area.height() / 2, -10, area.height(),
                                   20), QtCore.Qt.AlignCenter, y_label)
    painter.restore()
    curve = QtGui.QPolygonF([to_screen(px, py) for px, py in
                             zip(x[valid], y[valid])])
    pen = QtGui.QPen(QtGui.QColor(31, 119, 180))
    pen.setWidthF(2.0)
    painter.setPen(pen)
    painter.drawPolyline(curve)
    if marker is not None and 0 <= marker < len(x) and valid[marker]:
        point = to_screen(x[marker], y[marker])
        painter.setBrush(QtGui.QColor(214, 39, 40))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(point, 5, 5)


def _update_chart(node):
    window = _CHART_WINDOWS.get(node.sessionId())
    if window is None:
        return
    curves = _FORCE_CURVES.get(node.sessionId())
    if not curves:
        window.set_series(None)
        return
    try:
        window.set_series(_chart_series(node, curves))
    except ForceCurveError as exc:
        window.set_series(None)
        _message(str(exc))


def show_force_chart(kwargs):
    """Show Chart button: the chosen curve in a window that follows the
    playbar (computes the curves first if needed)."""
    node = kwargs["node"]
    try:
        curves = force_curves(node)
        series = _chart_series(node, curves)
    except ForceCurveError as exc:
        _message(str(exc))
        return None
    if not hou.isUIAvailable():
        return series
    window = _CHART_WINDOWS.get(node.sessionId())
    if window is None:
        window = _chart_class()(node, hou.qt.mainWindow())
        _CHART_WINDOWS[node.sessionId()] = window
    window.set_series(series)
    window.show()
    window.raise_()
    return window


def force_chart_changed(kwargs):
    _update_chart(kwargs["node"])


def open_force_csv(kwargs):
    """Open the CSV with the system's default program."""
    node = kwargs["node"]
    path = force_csv_path(node)
    if not os.path.isfile(path):
        if compute_force_curves(kwargs) is None or not os.path.isfile(path):
            return
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)  # noqa: S606 -- the user's own results file
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        _message(f"Could not open {path}: {exc}")


