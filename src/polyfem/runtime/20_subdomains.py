# =============================================================================
# Subdomains from Gmsh physical groups, and keeping settings on reload
#
# Subdomain n of a geometry holds the elements of the n-th physical group in
# sorted tag order, elements without a group last: tags {1, 100} give
# subdomains 1 and 2 (they gave 100, 98 of them empty), and untagged elements
# no longer renumber the tagged ones (the MSH Reader's +1 shift). The
# subdomains_<geo> wrangle applies the order recorded when the mesh was
# loaded; each subdomain's header shows its tag and name.
# =============================================================================


def physical_groups(geo_data):
    """{"tags", "names", "counts", "source"} of the primary elements' Gmsh
    physical groups, in subdomain order."""
    source = "msh_physical_tag" \
        if geo_data.findPrimAttrib("msh_physical_tag") is not None \
        else "Entity"      # an MSH Reader older than 2026-10-01
    values = np.frombuffer(geo_data.primIntAttribValuesAsString(source),
                           dtype=np.int32).astype(np.int64)
    mask = _volume_mask(geo_data)
    if mask is not None:
        values = values[mask]
    unique, counts = np.unique(values, return_counts=True)
    order = sorted(range(len(unique)),
                   key=lambda i: (int(unique[i]) <= 0, int(unique[i])))
    names = {}
    if geo_data.findGlobalAttrib("msh_physical_names") is not None:
        try:
            names = json.loads(geo_data.attribValue("msh_physical_names")
                               or "{}")
        except ValueError:
            names = {}
    tags = [int(unique[i]) for i in order]
    return {"tags": tags, "names": [names.get(str(t), "") for t in tags],
            "counts": [int(counts[i]) for i in order], "source": source}


def _subdomain_vex(groups):
    listing = ", ".join(str(tag) for tag in groups["tags"])
    return f"""\
// Subdomain = position of the element's Gmsh physical group in the sorted
// list read when the mesh was loaded (elements without a group last).
int tags[] = array({listing});
int index = find(tags, i@{groups['source']});
if (index < 0)
    error("Element %d is in physical group %d, which this mesh did not have "
          + "when it was loaded; pick the mesh file again", @primnum,
          i@{groups['source']});
i@Entity = index + 1;
"""


def subdomain_description(tag, name):
    if tag <= 0:
        return "elements without a Gmsh physical group"
    return f"Gmsh physical group {tag}" + (f" \"{name}\"" if name else "")


def _update_subdomain_info(parent, geo):
    """Fill each subdomain's header line: its physical group (tag, name) and
    element count, or what Apply Subdomain Change made of it."""
    branch = parent.node(f"branch_{geo}")
    source_node = parent.node(f"geo_{geo}")
    if branch is None or source_node is None:
        return
    geo_data = branch.geometry()
    if geo_data.findPrimAttrib("Entity") is None:
        return
    mask = _volume_mask(geo_data)
    entity = np.frombuffer(geo_data.primIntAttribValuesAsString("Entity"),
                           dtype=np.int32)
    tag_name = "msh_physical_tag" \
        if geo_data.findPrimAttrib("msh_physical_tag") is not None else None
    tags = np.frombuffer(geo_data.primIntAttribValuesAsString(tag_name),
                         dtype=np.int32) if tag_name else None
    if mask is not None:
        entity = entity[mask]
        tags = tags[mask] if tags is not None else None
    names = {}
    source = source_node.geometry()
    if source.findGlobalAttrib("msh_physical_names") is not None:
        try:
            names = json.loads(source.attribValue("msh_physical_names") or "{}")
        except ValueError:
            names = {}
    for vol in range(1, parent.evalParm(f"num_volumes{geo}") + 1):
        parm = parent.parm(f"subdomain_info{geo}_{vol}")
        if parm is None:
            continue
        mine = entity == vol
        count = int(mine.sum())
        if not count:
            text = "No elements (an empty subdomain)."
        elif tags is None:
            text = f"{count:,} elements."
        else:
            present = np.unique(tags[mine])
            if len(present) == 1:
                tag = int(present[0])
                whole = int((tags == tag).sum()) == count
                text = (("" if whole else "Part of ")
                        + subdomain_description(tag, names.get(str(tag), ""))
                        + f", {count:,} elements"
                        + ("" if whole else
                           " (the rest moved by Apply Subdomain Change)")
                        + ".")
            else:
                listed = ", ".join(str(int(t)) for t in present[:6])
                text = (f"{count:,} elements from Gmsh physical groups "
                        f"{listed}{', ...' if len(present) > 6 else ''}, "
                        "combined by Apply Subdomain Change.")
        text = text[:1].upper() + text[1:]
        if parm.eval() != text:
            parm.set(text)


# ---- one generic copy of a geometry's or subdomain's parameters ------------

_MULTIPARM_FOLDERS = (hou.folderType.MultiparmBlock,
                      hou.folderType.TabbedMultiparmBlock,
                      hou.folderType.ScrollingMultiparmBlock)
_NO_VALUE = (hou.parmTemplateType.Button, hou.parmTemplateType.Label,
             hou.parmTemplateType.Separator, hou.parmTemplateType.FolderSet)


def _geometry_patterns(parent):
    """[(regex, name, depth, is_count)] of every parameter inside the
    Geometry multiparm ("#" names), outer levels first."""
    folder = parent.parmTemplateGroup().find("num_geos")
    found = []

    def walk(templates):
        for template in templates:
            if template.type() == hou.parmTemplateType.Folder:
                if template.folderType() in _MULTIPARM_FOLDERS:
                    found.append((template.name(), True))
                walk(template.parmTemplates())
            elif template.type() not in _NO_VALUE:
                found.append((template.name(), False))

    walk(folder.parmTemplates())
    patterns = []
    for name, is_count in found:
        regex = re.compile("^" + re.escape(name).replace(r"\#", r"(\d+)")
                           + "$")
        patterns.append((regex, name, name.count("#"), is_count))
    patterns.sort(key=lambda item: (item[2], not item[3]))
    return patterns


def _capture_parm(parm):
    try:
        keys = parm.keyframes()
    except hou.Error:
        keys = ()
    if keys:
        return ("keys", keys)
    if parm.parmTemplate().type() == hou.parmTemplateType.String:
        return ("string", parm.unexpandedString())
    return ("value", parm.eval())


def _apply_parm(parm, captured):
    kind, value = captured
    if parm.keyframes():
        parm.deleteAllKeyframes()
    if kind == "keys":
        parm.setKeyframes(value)
    else:
        parm.set(value)


def snapshot_instances(parent, prefix, level=None):
    """Values of every parameter of Geometry-multiparm instances, in one pass.

    prefix: (geo,) for a whole geometry, (geo, vol) for one subdomain. With
    level=2 and prefix (geo,), returns {vol: snapshot} for every subdomain of
    the geometry. A snapshot is [(pattern, depth, is_count, tail indices,
    [captured values])] with counts before the parameters they hold and
    outer levels first.
    """
    prefix = tuple(int(i) for i in prefix)
    level = level or len(prefix)
    patterns = {}
    for regex, name, depth, is_count in _geometry_patterns(parent):
        if depth >= level:
            patterns.setdefault(depth, []).append(
                (name.split("#", 1)[0], regex, name, is_count))
    found = {}
    for tuple_parm in parent.parmTuples():
        if not len(tuple_parm):
            continue
        first = tuple_parm[0]
        if not first.isMultiParmInstance():
            continue
        indices = tuple(first.multiParmInstanceIndices())
        if len(indices) < level or indices[:len(prefix)] != prefix:
            continue
        tuple_name = tuple_parm.name()
        for literal, regex, name, is_count in patterns.get(len(indices), ()):
            if not tuple_name.startswith(literal):
                continue
            match = regex.match(tuple_name)
            if match and tuple(int(g) for g in match.groups()) == indices:
                key = indices[:level]
                found.setdefault(key, []).append(
                    (name, len(indices), is_count, indices[level:],
                     [_capture_parm(p) for p in tuple_parm]))
                break
    for snapshot in found.values():
        snapshot.sort(key=lambda item: (item[1], not item[2]))
    if level == len(prefix):
        return found.get(prefix, [])
    return {key[-1]: snapshot for key, snapshot in found.items()}


def snapshot_instance(parent, prefix):
    return snapshot_instances(parent, prefix)


def revert_instance(parent, prefix):
    """Put one instance (prefix (geo,) or (geo, vol)) back to the asset's
    defaults (multiparm counts first)."""
    prefix = tuple(int(i) for i in prefix)
    for name, depth, is_count, tail, values in snapshot_instance(parent,
                                                                 prefix):
        target = name
        for index in prefix + tuple(tail):
            target = target.replace("#", str(index), 1)
        tuple_parm = parent.parmTuple(target)
        if tuple_parm is not None:
            tuple_parm.revertToDefaults()


def restore_instance(parent, prefix, snapshot, skip=()):
    """Write a snapshot into another instance (prefix (geo,) or (geo, vol)):
    multiparm counts first, so the instances they hold exist."""
    prefix = tuple(int(i) for i in prefix)
    for name, depth, is_count, tail, values in snapshot:
        if name in skip:
            continue
        target = name
        for index in prefix + tuple(tail):
            target = target.replace("#", str(index), 1)
        tuple_parm = parent.parmTuple(target)
        if tuple_parm is None or len(tuple_parm) != len(values):
            continue
        for parm, captured in zip(tuple_parm, values):
            try:
                _apply_parm(parm, captured)
            except (hou.Error, TypeError):
                pass


