# ---- reload --------------------------------------------------------------


def _surface_keys(parent, geo):
    """Mesh node tags of every displayed surface face and point, so picks can
    follow faces into a changed mesh file."""
    null_node = parent.node(f"null_{geo}")
    if null_node is None:
        return None, None
    surf = null_node.geometry()
    try:
        _, face_keys = _surface_arrays(surf)
        point_keys = np.frombuffer(
            surf.pointIntAttribValuesAsString("msh_pt_id"), dtype=np.int32)
    except hou.Error:
        return None, None
    return face_keys, point_keys.astype(np.int64)


def _retag_chain(parent, geo):
    """[(subdomain, element pattern)] of the Apply Subdomain Change wrangles,
    upstream first."""
    chain = []
    node = parent.node(f"branch_{geo}")
    node = node.input(0) if node is not None else None
    while node is not None and node.name().startswith(f"vex_{geo}_"):
        try:
            number = int(node.name().rsplit("_", 1)[1])
        except ValueError:
            break
        chain.append((number, node.parm("group").eval()))
        node = node.input(0)
    return list(reversed(chain))


def _mesh_digest_key(geo):
    """User data holding the digest of geometry `geo`'s mesh as loaded."""
    return f"polyfem_mesh_digest_{int(geo)}"


def _capture_previous_mesh(parent, geo):
    """Everything a reload of geometry `geo` needs from the chain it
    replaces: the mesh file's digest, the subdomain number of every physical
    group, the subdomains' parameters, the retag wrangles and the surface's
    face identities."""
    geo = str(geo)
    num_vols = parent.evalParm(f"num_volumes{geo}")
    # file_location already holds the newly picked file when this runs; the
    # previous one is still on the reader node about to be replaced.
    reader = parent.node(f"geo_{geo}")
    path_parm = reader.parm("File") or reader.parm("file") \
        if reader is not None else None
    staged = path_parm.eval() if path_parm is not None else ""
    volumes = snapshot_instances(parent, (int(geo),), level=2)
    # The digest taken when the mesh was loaded. Hashing the staged file now
    # compares a mesh regenerated over its staged copy with itself, so the
    # reload saw "the same file" and kept picks and element reassignments
    # that name the old mesh. Scenes saved before the digest was remembered
    # fall back to the file.
    loaded_digest = parent.userData(_mesh_digest_key(geo))
    if not loaded_digest:
        loaded_digest = _file_digest(staged) if os.path.isfile(staged) \
            else None
    previous = {
        "digest": loaded_digest,
        "num_vols": num_vols,
        "volumes": {vol: volumes.get(vol, [])
                    for vol in range(1, num_vols + 1)},
        "retags": _retag_chain(parent, geo), "number_of_tag": {},
        "names": {}}
    previous["face_keys"], previous["point_keys"] = _surface_keys(parent, geo)
    # The stage right after the physical groups were turned into subdomains:
    # subdomains_<geo> since 2026-10-01, the MSH Reader itself before.
    stage = parent.node(f"subdomains_{geo}") or parent.node(f"geo_{geo}")
    try:
        geo_data = stage.geometry() if stage is not None else None
    except hou.Error:
        geo_data = None
    if parent.node(f"subdomains_{geo}") is None \
            and parent.node(f"surface_{geo}") is None:
        # An obstacle chain (no subdomain stage, no body surface): the
        # reader's Entity is the raw physical tag there, not a subdomain
        # number, so reading it as one handed subdomains each other's
        # settings when Obstacle was turned off again (tags 2 and 3: the
        # first took the second's, the second was reset). The subdomain
        # parameters were left as they were while it was an obstacle, and
        # they follow the sorted physical groups.
        try:
            if geo_data is not None \
                    and geo_data.findPrimAttrib("Entity") is not None:
                groups = physical_groups(geo_data)
                previous["number_of_tag"] = {
                    int(tag): number
                    for number, tag in enumerate(groups["tags"], 1)}
        except hou.Error:
            pass
    elif geo_data is not None and geo_data.findPrimAttrib("Entity") \
            is not None and geo_data.findPrimAttrib("msh_physical_tag") \
            is not None:
        entity = np.frombuffer(geo_data.primIntAttribValuesAsString("Entity"),
                               dtype=np.int32)
        tags = np.frombuffer(geo_data.primIntAttribValuesAsString(
            "msh_physical_tag"), dtype=np.int32)
        mask = _volume_mask(geo_data)
        if mask is not None:
            entity, tags = entity[mask], tags[mask]
        pairs = np.unique(np.stack((tags, entity), axis=1), axis=0)
        previous["number_of_tag"] = {int(t): int(v) for t, v in pairs}
    return previous


def _apply_subdomains(parent, geo, groups, previous, digest):
    """Set this geometry's subdomains for a freshly built chain; on a reload
    carry every subdomain's parameters over (matched by physical group) and
    say what could not be kept. Returns notes for the user."""
    geo = str(geo)
    count = len(groups["tags"])
    if previous is None:
        # A first load has no previous mesh to match subdomains against. The
        # geometry can still hold subdomains of a mesh that is gone (the Mesh
        # File field was emptied, or a load failed after the old chain was
        # removed): their sidesets name faces of that mesh, so they must not
        # be carried onto this one. Dropping to one subdomain removes them;
        # the subdomains added back are new instances with default settings.
        parent.setParms({f"num_volumes{geo}": 1})
        parent.setParms({f"num_volumes{geo}": count,
                         f"sideset_selection{geo}_1": 0,
                         f"initial_conditions{geo}_1": 0})
        for vol in range(1, count + 1):
            _set_color_tuple(parent, f"color_{geo}_{vol}",
                             default_color(geo, vol))
        return []

    notes = []
    same_file = digest is not None and digest == previous["digest"]
    old_number = previous["number_of_tag"]
    # new subdomain -> old subdomain whose settings it takes
    source = {}
    for new, tag in enumerate(groups["tags"], 1):
        old = old_number.get(tag)
        if old is not None and old in previous["volumes"]:
            source[new] = old
        else:
            notes.append(f"subdomain {new} ({subdomain_description(tag, groups['names'][new - 1])}) is new and starts with default settings")
    mesh_old = set(old_number.values())
    # subdomains made by Apply Subdomain Change: kept with the same elements
    # when the file is unchanged, numbered after the mesh's own
    retags = previous["retags"]
    made = sorted({number for number, _ in retags if number not in mesh_old})
    renumber = {old: new for new, old in source.items()}
    if retags and same_file:
        for offset, old in enumerate(made, 1):
            renumber[old] = count + offset
            if old in previous["volumes"]:
                source[count + offset] = old
    elif retags:
        notes.append("the element reassignments made with Apply Subdomain "
                     "Change were removed, because the mesh file changed")
    total = max([count] + list(source))
    taken = set(source.values())
    dropped = [old for old in previous["volumes"]
               if old not in taken and (old in mesh_old or old in made)]
    empty = [old for old in previous["volumes"]
             if old not in taken and old not in mesh_old and old not in made]
    if empty:
        notes.append(f"{len(empty)} empty subdomain"
                     f"{'s' if len(empty) != 1 else ''} of the previous "
                     "numbering (physical tag numbers no element used) "
                     f"{'were' if len(empty) != 1 else 'was'} removed")
    for old in sorted(dropped):
        tag = next((t for t, v in old_number.items() if v == old), None)
        what = f" ({subdomain_description(tag, '')})" if tag is not None \
            else ""
        notes.append(f"subdomain {old}{what} of the previous mesh is not in "
                     "this one; its settings were removed")
    # Write the settings: keep the parameters of subdomains that stay where
    # they are, move the others from their snapshot (taken before anything
    # changed), and give new subdomains defaults.
    parent.setParms({f"num_volumes{geo}": total})
    for new in range(1, total + 1):
        old = source.get(new)
        if old == new:
            continue
        if old is None:
            revert_instance(parent, (int(geo), new))
            _set_color_tuple(parent, f"color_{geo}_{new}",
                             default_color(geo, new))
            continue
        restore_instance(parent, (int(geo), new), previous["volumes"][old])
    if any(old != new for new, old in source.items()):
        moves = ", ".join(f"{old} -> {new}" for new, old in sorted(
            source.items()) if old != new)
        notes.append(f"subdomains were renumbered to follow the sorted "
                     f"physical groups ({moves}); their settings moved with "
                     "them")
    # Element reassignments of an unchanged file are rebuilt as they were.
    if retags and same_file:
        upstream = parent.node(f"elements_{geo}")
        branch = parent.node(f"branch_{geo}")
        for number, pattern in retags:
            target = renumber.get(number, number)
            vex = parent.node(f"vex_{geo}_{target}") or parent.createNode(
                "attribwrangle", f"vex_{geo}_{target}")
            vex.setParms({"group": pattern, "grouptype": 4, "class": 1,
                          "snippet": f"i@Entity = {int(target)};"})
            vex.setInput(0, upstream)
            upstream = vex
        branch.setInput(0, upstream)
    return notes


def _remap_picks(parent, geo, previous):
    """Picked faces and points refer to the surface's numbering; after the
    mesh file changed, follow them by their mesh node tags."""
    notes = []
    old_faces, old_points = previous.get("face_keys"), \
        previous.get("point_keys")
    new_faces, new_points = _surface_keys(parent, geo)
    if old_faces is None or new_faces is None:
        return notes
    face_index = {tuple(sorted(int(k) for k in row if k >= 0)): prim
                  for prim, row in enumerate(new_faces)}
    point_index = {int(key): point for point, key in enumerate(new_points)}
    for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
        for j in range(1, parent.evalParm(f"sideset_selection{geo}_{vol}") + 1):
            parm = parent.parm(f"basegroup{geo}_{vol}_{j}")
            pattern = parm.unexpandedString().strip()
            if not pattern or pattern == "*" or is_native_selection(pattern):
                continue
            points = parent.evalParm(f"grouptype{geo}_{vol}_{j}") == 1
            members = expand_group_str(pattern)
            kept = []
            for member in members:
                if points:
                    if 0 <= member < len(old_points) and \
                            int(old_points[member]) in point_index:
                        kept.append(point_index[int(old_points[member])])
                elif 0 <= member < len(old_faces):
                    key = tuple(sorted(int(k) for k in old_faces[member]
                                       if k >= 0))
                    if key in face_index:
                        kept.append(face_index[key])
            parm.set(list_to_space_str(kept))
            if len(kept) != len(members):
                notes.append(
                    f"subdomain {vol} sideset {j}: {len(members) - len(kept)} "
                    f"of {len(members)} picked "
                    f"{'points' if points else 'faces'} are not on the new "
                    "mesh and were removed")
    return notes


def _merge_all(parent):
    merge_node = parent.node("all")
    if merge_node is None:
        merge_node = parent.createNode("merge", "all")
        out = parent.node("output")
        if out is not None:
            out.setInput(0, merge_node)
    return merge_node


FIBER_DATA_SOP_CODE = """
node = hou.pwd()
node.parent().hdaModule().cook_fiber_data(node)
"""


DISPLAY_CHAIN_VERSION = 2


def create_geo_tree(kwargs):
    """Volume geometry display chain: one Entity-colored boundary surface."""
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    branch = parent.node("branch_" + geo_number)

    # Per-element material data is stamped onto the volume elements here, so
    # the viewport draws exactly what the export writes out.
    fiberdata = parent.createNode("python", "fiberdata_" + geo_number)
    fiberdata.parm("python").set(FIBER_DATA_SOP_CODE)
    fiberdata.setNextInput(branch)

    surface = parent.createNode("attribwrangle", "surface_" + geo_number)
    surface.setParms({"class": 1,
                      "snippet": _surface_vex(parent, geo_number)})
    surface.setNextInput(fiberdata)

    color = parent.createNode("attribwrangle", "entitycolor_" + geo_number)
    color.setParms({
        "class": 1,
        "snippet": _entity_color_vex(
            geo_number, parent.evalParm(f"num_volumes{geo_number}"))})
    color.setNextInput(surface)

    # min-edge readout source (suggested dhat)
    mindist = parent.createNode("attribwrangle", "mindist_" + geo_number)
    mindist.setParms({"class": 1, "snippet": MIN_EDGE_VEX})
    mindist.setNextInput(surface)

    _build_fiber_viz(parent, geo_number, fiberdata)

    null_node = parent.createNode("null", "null_" + geo_number)
    null_node.setNextInput(color)
    _merge_all(parent).setNextInput(null_node)

    # clear_geo_tree destroys the sideset group nodes with the rest of the
    # display chain; rebuild them so viewport reselection keeps working
    # after subdomain edits. The sidesets themselves are not new: one whose
    # conditions were removed must not get the default condition back.
    for vol in range(1, parent.evalParm(f"num_volumes{geo_number}") + 1):
        if parent.parm(f"sideset_selection{geo_number}_{vol}") is not None:
            _rebuild_group_chain(parent, geo_number, str(vol),
                                 default_condition=False)
    parent.setUserData("polyfem_display_version", str(DISPLAY_CHAIN_VERSION))
    parent.layoutChildren()


def create_obstacle_tree(parent, geo_number):
    branch = parent.node("branch_" + geo_number)

    # Seed the swatch with this geometry's palette color (distinct color per
    # obstacle, matching 1.2) without clobbering a user-customized value.
    swatch = parent.parmTuple(f"color_{geo_number}")
    if swatch is not None and all(p.isAtDefault() for p in swatch):
        _set_color_tuple(parent, f"color_{geo_number}",
                         default_color(geo_number))

    # Volumes color per-PRIM; if an obstacle carries per-POINT Cd (painted
    # .obj/.ply), the merged output ends up with both attribute classes and
    # the volume geometry's points fall back to default-black point Cd.
    # Promote any point Cd to prims first so a single class survives the
    # merge, then tint (preserving promoted file colors).
    promote_node = parent.createNode("attribpromote", "cdpromote_" + geo_number)
    promote_node.setParms({"inname": "Cd", "inclass": 2, "outclass": 1,
                           "deletein": 1})
    promote_node.setNextInput(branch)

    color_node = parent.createNode("attribwrangle", "color_" + geo_number)
    color_node.setParms({"class": 1, "snippet": (
        "if (!hasprimattrib(0, 'Cd'))\n"
        f"    v@Cd = chv('../color_{geo_number}');\n"
        "f@Alpha = 1.0;\n"
        f"if (!chi('../is_enabled{geo_number}')) {{\n"
        "    v@Cd = lerp(v@Cd, {0.35, 0.35, 0.35}, 0.7);\n"
        "    f@Alpha = 0.25;\n"
        "}")})
    color_node.setNextInput(promote_node)
    null_node = parent.createNode("null", "null_" + geo_number)
    null_node.setNextInput(color_node)
    _merge_all(parent).setNextInput(null_node)
    parent.setUserData("polyfem_display_version", str(DISPLAY_CHAIN_VERSION))
    parent.layoutChildren()


def rebuild_display_network(kwargs, quiet=False):
    """Rebuild the internal viewport display SOPs for every loaded geometry
    from the current asset definition, keeping all transforms, materials,
    subdomains, element reassignments, sidesets, and boundary conditions."""
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    rebuilt = 0
    num_geos = max(0, int(parent.evalParm("num_geos")))
    for geo in range(1, num_geos + 1):
        geo_str = str(geo)
        if parent.node(f"branch_{geo_str}") is None:
            continue
        clear_geo_tree(parent, geo_str)
        if parent.parm(f"is_obstacle{geo_str}") is not None \
                and parent.evalParm(f"is_obstacle{geo_str}"):
            create_obstacle_tree(parent, geo_str)
        else:
            create_geo_tree({"node": parent, "script_multiparm_index": geo_str})
        rebuilt += 1
    parent.setUserData("polyfem_display_version", str(DISPLAY_CHAIN_VERSION))
    if not quiet:
        if rebuilt:
            _status(
                f"Rebuilt display network for {rebuilt} "
                f"geometr{'y' if rebuilt == 1 else 'ies'} (all materials, "
                "transforms, subdomains, and sidesets preserved).")
        else:
            _status("No loaded geometries to rebuild.")
    return rebuilt


def check_display_version(node):
    """Upgrade the internal display chain of an older saved scene on load."""
    if not _has_loaded_geometry(node):
        return 0
    raw = node.userData("polyfem_display_version")
    try:
        current = int(raw) if raw is not None else 0
    except ValueError:
        current = 0
    if current >= DISPLAY_CHAIN_VERSION:
        return 0
    rebuilt = rebuild_display_network({"node": node}, quiet=True)
    if rebuilt:
        print(
            f"[PolyFEM HDA] {node.path()}: upgraded display network "
            f"(v{current} -> v{DISPLAY_CHAIN_VERSION}) for {rebuilt} "
            "geometries.")
    return rebuilt


def clear_geo_tree(parent, geo_number):
    """Remove the display chain (keeps the import chain: geo, subdomains,
    attrib, transform, elements, vex retags and branch)."""
    for item in parent.children():
        name = item.name()
        if (_geo_index_of(name) == int(geo_number)
                and name.split("_")[0] in _GEOMETRY_DISPLAY_PREFIXES):
            item.destroy()
    merge_node = parent.node("all")
    if merge_node is not None:
        for i, inp in enumerate(merge_node.inputs()):
            if inp is None:
                continue
            if _geo_index_of(inp.name()) == int(geo_number):
                merge_node.setInput(i, None)


_LAST_FETCHED_ELEMENTS = {}


def _has_loaded_geometry(parent):
    return any(parent.node(f"geo_{g}") is not None
               for g in range(1, max(1, parent.evalParm("num_geos")) + 1))


def clear_all_geos(kwargs, _confirmed=False):
    parent = kwargs["node"]
    if not _confirmed and hou.isUIAvailable() and _has_loaded_geometry(parent):
        choice = hou.ui.displayMessage(
            "Clear All removes every geometry, material, sideset and boundary "
            "condition on this node and cannot be undone.",
            buttons=("Clear All", "Cancel"),
            default_choice=1, close_choice=1,
            severity=hou.severityType.Warning, title="Clear All Geometries")
        if choice != 0:
            return
    parent.allowEditingOfContents()
    for item in parent.children():
        if item.name() not in _PROTECTED_NODES:
            item.destroy()
    # Geometry 1 is the one entry that stays, so remove channels first and
    # revert the entire instance to defaults (including transform/rotate
    # orders, materials, display toggles and digest userData).
    old_num_geos = max(1, parent.evalParm("num_geos"))
    for parm in _transform_parms(parent, 1):
        if parm.keyframes():
            parm.deleteAllKeyframes()
    parent.setParms({"geo_int": 1, "num_geos": 1})
    revert_instance(parent, (1,))
    for g in range(1, old_num_geos + 1):
        key = _mesh_digest_key(g)
        if parent.userData(key) is not None:
            parent.destroyUserData(key)
    for key in [k for k in _LAST_FETCHED_ELEMENTS if k[0] == parent.sessionId()]:
        _LAST_FETCHED_ELEMENTS.pop(key, None)
    if hou.isUIAvailable():
        sv = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer)
        if sv:
            sv.curViewport().home()


def eval_geo_num(kwargs):
    return kwargs["node"].evalParm("geo_int")


def clear_geos(kwargs):
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    old_geos = int(parent.evalParm("num_geos"))
    num_geos = max(1, int(eval_geo_num(kwargs)))
    if num_geos < old_geos and hou.isUIAvailable():
        dropped = [g for g in range(num_geos + 1, old_geos + 1)
                   if parent.node(f"geo_{g}") is not None]
        if dropped:
            choice = hou.ui.displayMessage(
                f"Lowering Number of Geometries from {old_geos} to {num_geos} "
                f"will remove geometry {', '.join(str(g) for g in dropped)} "
                "and all of its materials and sidesets.",
                buttons=("Remove", "Cancel"),
                default_choice=1, close_choice=1,
                severity=hou.severityType.Warning,
                title="Remove Geometries")
            if choice != 0:
                parent.setParms({"geo_int": old_geos})
                return
    parent.setParms({"num_geos": num_geos})
    for item in parent.children():
        if item.name() in _PROTECTED_NODES:
            continue
        idx = _geo_index_of(item.name())
        if idx is not None and idx > num_geos:
            item.destroy()
    if hou.isUIAvailable():
        pane_tab = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.Parm, 0)
        if pane_tab:
            pane_tab.setMultiParmTab("num_geos", max(0, num_geos - 1))


def fetch_elements(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    entity_number = parent.evalParm("subdomain_number_" + geo_number)
    branch = parent.node("branch_" + geo_number)
    if branch is None:
        return
    # Do not wipe a selection the user just picked or typed for moving into
    # the chosen target subdomain; only refresh when the field is empty or
    # still holds the last auto-fetched list.
    key = (parent.sessionId(), str(geo_number))
    current = parent.evalParm("elements_" + geo_number).strip()
    if current and current != _LAST_FETCHED_ELEMENTS.get(key, ""):
        return
    entity = np.frombuffer(
        branch.geometry().primIntAttribValuesAsString("Entity"), dtype=np.int32)
    elements = np.nonzero(entity == int(entity_number))[0]
    fetched = list_to_space_str(elements)
    _LAST_FETCHED_ELEMENTS[key] = fetched.strip()
    parent.setParms({"elements_" + geo_number: fetched})


def update_entities(kwargs, call=None):
    """Re-tag selected elements with the chosen subdomain number (VEX)."""
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    if parent.node("geo_" + geo_number) is None:
        return
    entity_number = int(parent.evalParm("subdomain_number_" + geo_number))
    if entity_number < 1 or entity_number > 999:
        _message("Subdomain numbers must be between 1 and 999.")
        return

    raw_elements = parent.evalParm("elements_" + geo_number).strip()
    elements = list_to_space_str(expand_group_str(raw_elements))
    if not elements:
        _message("Select at least one element in 'Elements to Move' first "
                 "(empty selections are refused so the whole mesh is not "
                 "overwritten).")
        return

    # A brand-new subdomain number must also grow the per-volume UI
    # (material tabs, color swatches) and the color mapping, or the new
    # subdomain renders grey and has no visible material entry.
    num_vols = parent.evalParm(f"num_volumes{geo_number}")
    if entity_number > num_vols:
        parent.setParms({f"num_volumes{geo_number}": entity_number})
        for vol in range(num_vols + 1, entity_number + 1):
            _set_color_tuple(parent, f"color_{geo_number}_{vol}",
                             default_color(geo_number, vol))

    vex_node = parent.node(f"vex_{geo_number}_{entity_number}")
    branch = parent.node("branch_" + geo_number)
    if vex_node is None:
        vex_node = parent.createNode(
            "attribwrangle", f"vex_{geo_number}_{entity_number}")
    else:
        # Splice an existing wrangle out of its old chain position so this
        # edit always runs downstream of earlier subdomain edits.
        old_in = vex_node.inputs()[0] if vex_node.inputs() else None
        for out_node in list(vex_node.outputs()):
            for idx, inp in enumerate(out_node.inputs()):
                if inp == vex_node:
                    out_node.setInput(idx, old_in)
        vex_node.setInput(0, None)
    vex_node.setParms({"group": elements, "grouptype": 4, "class": 1,
                       "snippet": f"i@Entity = {entity_number};"})
    upstream = branch.inputs()[0]
    vex_node.setInput(0, upstream)
    branch.setInput(0, vex_node)
    _LAST_FETCHED_ELEMENTS.pop((parent.sessionId(), str(geo_number)), None)
    clear_geo_tree(parent, geo_number)
    create_geo_tree(kwargs)
    _update_subdomain_info(parent, geo_number)


def revert_entities(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    has_edits = any(
        item.name().split("_")[0] == "vex"
        and _geo_index_of(item.name()) == int(geo_number)
        for item in parent.children())
    if has_edits and hou.isUIAvailable():
        choice = hou.ui.displayMessage(
            f"Restore Mesh Subdomains on Geometry {geo_number} discards all "
            "element reassignments and removes any added subdomains with "
            "their materials and sidesets.",
            buttons=("Restore", "Cancel"),
            default_choice=1, close_choice=1,
            severity=hou.severityType.Warning,
            title="Restore Mesh Subdomains")
        if choice != 0:
            return
    for item in parent.children():
        name = item.name()
        if name.split("_")[0] == "vex" and _geo_index_of(name) == int(geo_number):
            item.destroy()
    # reconnect branch to elements node
    branch = parent.node("branch_" + geo_number)
    elements = parent.node("elements_" + geo_number)
    if branch is not None and elements is not None:
        branch.setInput(0, elements)
        # The subdomains Apply Subdomain Change added beyond the mesh's own
        # are empty again. They used to stay, with their material, sidesets
        # and conditions (and no way to remove them: the subdomain count is
        # not shown), and a sideset left on one refused every export.
        try:
            entity = np.frombuffer(
                branch.geometry().primIntAttribValuesAsString("Entity"),
                dtype=np.int32)
            mask = _volume_mask(branch.geometry())
            if mask is not None:
                entity = entity[mask]
            own = int(entity.max()) if len(entity) else 0
        except hou.Error:
            own = 0
        num_vols = parent.evalParm(f"num_volumes{geo_number}")
        if 1 <= own < num_vols:
            parent.setParms({f"num_volumes{geo_number}": own})
            _status(f"Geometry {geo_number}: the {num_vols - own} "
                    "subdomain(s) added by Apply Subdomain Change were "
                    "removed with their settings.")
    fetch_elements(kwargs)
    clear_geo_tree(parent, geo_number)
    create_geo_tree(kwargs)
    _update_subdomain_info(parent, geo_number)


def BC_check(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index2"]
    if parent.parm("is_obstacle" + str(geo_number)).eval():
        _message("Obstacles take a displacement, not sidesets.")


def create_group(kwargs):
    """Sideset count callback: maintain groupcreate nodes in the surface chain."""
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    geo_number = kwargs["script_multiparm_index2"]
    vol_num = kwargs["script_multiparm_index"]
    if parent.parm("is_obstacle" + str(geo_number)).eval():
        _message("Obstacles take a displacement, not sidesets.")
        parent.setParms({f"sideset_selection{geo_number}_{vol_num}": 0})
        return
    if parent.node("geo_" + geo_number) is None:
        _message("No input geometry provided to apply sideset!")
        parent.setParms({f"sideset_selection{geo_number}_{vol_num}": 0})
        return
    _rebuild_group_chain(parent, geo_number, vol_num)


def _rebuild_group_chain(parent, geo_number, vol_num, default_condition=True):
    sideset_number = parent.evalParm(
        f"sideset_selection{geo_number}_{vol_num}")

    prefix = f"group_{geo_number}_{vol_num}_"
    existing = [n for n in parent.children() if n.name().startswith(prefix)]
    for node in existing:
        if int(node.name().split("_")[-1]) > sideset_number:
            node.destroy()

    color_node = parent.node("entitycolor_" + geo_number)
    null_node = parent.node("null_" + geo_number)
    if color_node is None or null_node is None:
        return

    # The group nodes hang off entitycolor on a dead-end branch of their own
    # rather than being spliced into the display chain. Splicing them inline
    # meant each volume's rebuild re-pointed null_ at its own last group node,
    # orphaning every other volume's chain; and nothing downstream consumes
    # the groups anyway -- they exist only to host the viewport selection.
    null_node.setInput(0, color_node)

    # Restrict picking to this volume. soputils.selectGroupParm display-flags
    # the group node's input 0, so every entity visible there is pickable: on
    # a multi-material mesh the other subdomains stayed in the way, hiding the
    # interface faces behind them, and any face picked on another entity was
    # then silently dropped by export_sidesets' per-volume filter. 1.2 got this
    # for free by giving each volume its own split branch; do it without the
    # per-volume node forest by hiding the other entities instead of deleting
    # them -- Visibility leaves primitive numbering untouched, which the
    # geometry-wide prim numbers in basegroup depend on.
    solo_name = f"solo_{geo_number}_{vol_num}"
    solo_node = parent.node(solo_name)
    if sideset_number == 0:
        if solo_node is not None:
            solo_node.destroy()
        parent.layoutChildren()
        return
    if solo_node is None:
        solo_node = parent.createNode("visibility", solo_name)
    solo_node.setParms({"group": f"@Entity!={int(vol_num)}",
                        "action": 0, "applyto": 0})
    solo_node.setInput(0, color_node)

    upstream = solo_node
    for num in range(1, sideset_number + 1):
        name = f"{prefix}{num}"
        group_node = parent.node(name)
        if group_node is None:
            group_node = parent.createNode("groupcreate", name)
            # A freshly created sideset defaults to one zero-displacement
            # Dirichlet BC (all dims fixed), matching the 1.x behavior --
            # but never overwrite a sideset that already carries BCs or a
            # selection (e.g. during params import).
            bc_count = parent.parm(
                f"Boundary_Condition__{geo_number}_{vol_num}_{num}")
            basegroup = parent.evalParm(
                f"basegroup{geo_number}_{vol_num}_{num}")
            if default_condition and bc_count is not None \
                    and bc_count.eval() == 0 and not basegroup.strip():
                bc_count.set(1)
            group_node.parm("grouptype").setExpression(
                f'ch("../grouptype{geo_number}_{vol_num}_{num}")')
            # native selection syntax types into the same field; feed the
            # group SOP an empty pattern in that case so it does not error
            group_node.parm("basegroup").setExpression(
                f"p = hou.node('..').evalParm('basegroup{geo_number}_{vol_num}_{num}')\n"
                "return '' if p.strip().lower().startswith("
                "('axis:', 'box:', 'sphere:', 'plane:')) else p",
                hou.exprLanguage.Python)
        group_node.setInput(0, upstream)
        upstream = group_node
    parent.layoutChildren()


def geo_duplicate(kwargs):
    """Duplicate Geometry: a new geometry with the same mesh file and every
    setting of this one -- transform, subdomain assignments, materials (fiber
    sources and families included), sidesets with their conditions, initial
    conditions, obstacle motion and display options."""
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    if parent.node("geo_" + geo_number) is None:
        _message("Import geometry, then duplicate.")
        return
    src = int(geo_number)
    snapshot = snapshot_instance(parent, (src,))
    retags = _retag_chain(parent, src)
    dst = parent.evalParm("num_geos") + 1
    mesh_parm = parent.parm(f"file_location{src}")
    mesh = mesh_parm.unexpandedString()
    is_obstacle = parent.evalParm(f"is_obstacle{src}")
    if parent.parm(f"mesh_source{src}").evalAsString() == "sop":
        try:
            mesh = _sop_staging_path(parent, dst, is_obstacle)
            source = mesh_parm.eval()
            if not source:
                raise hou.NodeError("Reload the Source SOP before duplicating it.")
            shutil.copy2(source, mesh)
        except (OSError, hou.Error) as exc:
            _message(f"Could not duplicate SOP geometry {src}: {exc}")
            return
    parent.setParms({"num_geos": dst, "geo_int": dst})
    if hou.isUIAvailable():
        pane_tab = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.Parm, 0)
        if pane_tab:
            pane_tab.setMultiParmTab("num_geos", max(0, dst - 1))

    # The mesh first: building its chain sets this geometry's subdomains
    # from the file; everything else is then copied over that.
    parent.parm(f"file_location{dst}").set(mesh)
    parent.setParms({f"is_obstacle{dst}": is_obstacle,
                     f"is_enabled{dst}": parent.evalParm(f"is_enabled{src}")})
    parent.parm(f"file_location{dst}").pressButton()
    if parent.node(f"geo_{dst}") is None:
        return
    restore_instance(parent, (dst,), snapshot,
                     skip=("file_location#", "is_obstacle#", "is_enabled#"))
    if parent.evalParm(f"is_obstacle{dst}"):
        return dst
    if retags:
        upstream = parent.node(f"elements_{dst}")
        for number, pattern in retags:
            vex = parent.createNode("attribwrangle", f"vex_{dst}_{number}")
            vex.setParms({"group": pattern, "grouptype": 4, "class": 1,
                          "snippet": f"i@Entity = {int(number)};"})
            vex.setInput(0, upstream)
            upstream = vex
        parent.node(f"branch_{dst}").setInput(0, upstream)
    clear_geo_tree(parent, dst)
    create_geo_tree({"node": parent, "script_multiparm_index": str(dst)})
    update_fiber_data(parent, str(dst))
    _update_subdomain_info(parent, dst)
    _status(f"Geometry {src} duplicated as geometry {dst} (same mesh and "
            "settings, same transform: move it before running).")
    return dst


def deter_min_edge(kwargs):
    parent = kwargs["node"]
    total_min = None
    for node in parent.glob("mindist_*"):
        try:
            node.cook(force=False)
        except hou.OperationFailed:
            _message(f"Could not cook {node.name()}: {node.errors()}")
            continue
        geo = node.geometry()
        if geo is None or geo.findPrimAttrib("min_edge_length") is None:
            continue
        values = np.frombuffer(
            geo.primFloatAttribValuesAsString("min_edge_length"),
            dtype=np.float32)
        if len(values):
            local_min = float(values.min())
            total_min = local_min if total_min is None \
                else min(total_min, local_min)
    if total_min is None:
        _message("No surface geometry found; import a volume mesh first.")
        return
    parent.parm("min_edge").set(str(total_min))


def suggest_dhat(kwargs):
    """Set dhat to 0.001 x bounding-box diagonal of loaded enabled meshes."""
    parent = kwargs["node"]
    geos = [g for g in range(1, parent.evalParm("num_geos") + 1)
            if parent.parm(f"is_enabled{g}").eval()
            and parent.node(f"transform_{g}") is not None]
    diagonal = _model_diagonal(parent, geos)
    if not diagonal or diagonal <= 0:
        _message("No loaded enabled geometry found; import a mesh first.")
        return None
    suggested = max(1e-12, 1e-3 * diagonal)
    parent.parm("dhat").set(suggested)
    _status(f"Set dhat = {suggested:.4g} (0.001 × model diagonal {diagonal:.4g}).")
    return suggested


