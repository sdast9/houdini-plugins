# =============================================================================
# Geometry node-tree management
# =============================================================================

_PROTECTED_NODES = {"output", "all"}
_GEOMETRY_IMPORT_PREFIXES = {
    "geo", "subdomains", "attrib", "transform", "elements", "vex", "branch",
}
_GEOMETRY_DISPLAY_PREFIXES = {
    "fiberdata", "surface", "entitycolor", "mindist", "null", "cdpromote",
    "color", "solo", "group", "fiberviz", "fibervizswitch", "nofiberviz",
}


def _geo_index_of(name):
    parts = name.split("_")
    if (len(parts) >= 2 and parts[1].isdigit()
            and parts[0] in _GEOMETRY_IMPORT_PREFIXES | _GEOMETRY_DISPLAY_PREFIXES):
        return int(parts[1])
    return None


def destroy_geo_nodes(parent, geo_number):
    for item in parent.children():
        if item.name() in _PROTECTED_NODES:
            continue
        if _geo_index_of(item.name()) == int(geo_number):
            item.destroy()


def working_dir_check(kwargs):
    """Working Directory callback: create input/ and output/, and end the
    path with a slash -- keeping the text as typed, so "$HIP/sim" stays
    relative to the scene file and a moved or shared .hip still finds it."""
    parent = kwargs["node"]
    parm = parent.parm("working_dir")
    working_dir = parm.eval()
    if not working_dir:
        _message("Please set Main > Working Directory before loading a mesh.")
        return False
    try:
        os.makedirs(os.path.join(working_dir, "input"), exist_ok=True)
        os.makedirs(os.path.join(working_dir, "output"), exist_ok=True)
    except OSError as exc:
        _message(f"Could not create the working directory {working_dir}: "
                 f"{exc}")
        return False
    raw = parm.unexpandedString()
    if not raw.endswith(("/", "\\")):
        parm.set(raw + "/")
    return True


def portable_path(parent, path):
    """`path` written relative to the Working Directory as the user typed it.

    Files the node stages into <working_dir>/input are stored as e.g.
    "$HIP/sim/input/part.msh" when the working directory is "$HIP/sim/", so
    moving the .hip together with its simulation folder keeps them. Paths
    outside the working directory are returned unchanged.
    """
    raw = parent.parm("working_dir").unexpandedString()
    expanded = parent.evalParm("working_dir")
    if not raw or not expanded or raw == expanded:
        return path
    base = os.path.normpath(os.path.abspath(expanded))
    target = os.path.normpath(os.path.abspath(path))
    if not target.startswith(base + os.sep):
        return path
    relative = os.path.relpath(target, base).replace(os.sep, "/")
    return raw.rstrip("/\\") + "/" + relative


# Files Write itself creates in input/ (rewritten at every Write). A material
# file staged under one of these names would be overwritten by the export.
_EXPORT_FILE_RE = re.compile(
    r"^(fibers\.vtk|params\.json|hda_sidesets\.json|hda_scene\.json"
    r"|pe_[a-z]+_\d+_\d+(_\d+)?\.txt|volumes\d+\.txt"
    r"|surface_sidesets\d+_(tri|quad)\.txt|point_sidesets\d+\.txt)$",
    re.IGNORECASE)
# The per-element material file parameters (fiber files, dispersion files).
_MATERIAL_FILE_PARM_RE = re.compile(r"^(fam_)?(fib|kappa)_file\d+_\d+(_\d+)?$")


def _export_writes(name):
    """Whether Write creates a file of this name in input/."""
    return bool(_EXPORT_FILE_RE.match(name))


def _material_file_parms(parent):
    return [parm for parm in parent.parms()
            if _MATERIAL_FILE_PARM_RE.match(parm.name())]


def _held_input_files(parent, input_dir, exclude_parm=None):
    """{normcased path: path} of the files in input_dir that the node's
    meshes and material-file parameters (except `exclude_parm`) point at."""
    held = {}
    for path in _staged_meshes(parent, input_dir):
        held[os.path.normcase(path)] = path
    for parm in _material_file_parms(parent):
        if parm.name() == exclude_parm:
            continue
        value = parm.eval().strip()
        if not value:
            continue
        path = os.path.normpath(os.path.abspath(value))
        if os.path.dirname(path) == input_dir:
            held[os.path.normcase(path)] = path
    return held


def _stage_material_file(parent, source, label="material file",
                         parm_name=None):
    """Copy a user-selected material data file into ``working_dir/input``.

    PolyFEM resolves material file references relative to params.json.  Keeping
    the UI parameter pointed at the staged copy makes the viewport preview and
    the solver consume the exact same file, and makes the input folder portable.

    The copy never takes a name that Write creates itself (fibers.vtk, the
    pe_*.txt dispersion files, ...), which the next Write would overwrite,
    and never replaces a different file that another mesh or material file
    parameter of the node uses: two files of the same name from different
    folders used to become one input file, so one body was simulated with
    the other's fibers. Such a name gets _<parameter name> appended instead.
    `parm_name` is the parameter the file is staged for.
    """
    source = os.path.normpath(os.path.abspath(hou.text.expandString(source)))
    if not os.path.isfile(source):
        raise hou.NodeError(f"{label} not found: {source}")
    working_dir = parent.evalParm("working_dir")
    if not working_dir:
        raise hou.NodeError(
            f"Choose a working directory before selecting a {label}.")
    input_dir = os.path.normpath(os.path.abspath(
        os.path.join(working_dir, "input")))
    os.makedirs(input_dir, exist_ok=True)
    if os.path.dirname(source) == input_dir \
            and not _export_writes(os.path.basename(source)):
        return source
    held = _held_input_files(parent, input_dir, exclude_parm=parm_name)
    stem, extension = os.path.splitext(os.path.basename(source))
    tag = parm_name or "file"
    names = [stem + extension, f"{stem}_{tag}{extension}"] + [
        f"{stem}_{tag}_{number}{extension}" for number in range(2, 100)]
    for name in names:
        if _export_writes(name):
            continue
        target = os.path.join(input_dir, name)
        same = os.path.isfile(target) and _same_file(target, source)
        if os.path.normcase(target) in held and not same:
            continue  # another input's different file has this name
        if not same:
            try:
                shutil.copy2(source, target)
            except OSError as exc:
                raise hou.NodeError(
                    f"Could not copy {label} into the input folder: {exc}")
        return target
    raise hou.NodeError(
        f"No free file name for {stem}{extension} in {input_dir}")


def _restage_export_named_files(parent, input_dir):
    """Move material files off names that Write is about to overwrite.

    Scenes saved before 2026-11-14 (and runs imported then) can point a fiber
    or dispersion file parameter at input/fibers.vtk or input/pe_*.txt, which
    the export rewrites. Copy each such file to a free name first and point
    the parameter at the copy, so the file read is still the one meant.
    """
    input_dir = os.path.normpath(os.path.abspath(input_dir))
    for parm in _material_file_parms(parent):
        value = parm.eval().strip()
        if not value:
            continue
        path = os.path.normpath(os.path.abspath(value))
        if os.path.dirname(path) != input_dir \
                or not _export_writes(os.path.basename(path)) \
                or not os.path.isfile(path):
            continue
        staged = _stage_material_file(parent, path, "material file",
                                      parm.name())
        parm.set(portable_path(parent, staged))


def _same_file(first, second):
    try:
        return filecmp.cmp(first, second, shallow=False)
    except OSError:
        return False


def _staged_meshes(parent, input_dir, exclude=None):
    """{path: path} for every geometry whose mesh already sits in input_dir
    (except geometry `exclude`); the starting point for _stage_mesh."""
    input_dir = os.path.normpath(os.path.abspath(input_dir))
    taken = {}
    for geo in range(1, parent.evalParm("num_geos") + 1):
        if exclude is not None and int(geo) == int(exclude):
            continue
        parm = parent.parm(f"file_location{geo}")
        path = parm.eval() if parm is not None else ""
        if not path:
            continue
        path = os.path.normpath(os.path.abspath(path))
        if os.path.dirname(path) == input_dir:
            taken[path] = path
    return taken


def _stage_mesh(geo, source, input_dir, taken):
    """Copy a mesh into input_dir under a name no other geometry uses for a
    different file, and return the staged path.

    Staging used to go by file name alone, so part.msh from two folders became
    one input/part.msh and both geometries simulated the second mesh. A name
    already holding a different mesh gets _geo<N> appended instead. `taken`
    maps staged paths to the file each holds; it is updated.
    """
    source = os.path.normpath(os.path.abspath(os.path.expandvars(source)))
    input_dir = os.path.normpath(os.path.abspath(input_dir))
    if not os.path.isfile(source):
        raise hou.NodeError(f"Mesh file not found: {source}")
    if os.path.dirname(source) == input_dir:
        taken[source] = source
        return source
    os.makedirs(input_dir, exist_ok=True)
    stem, extension = os.path.splitext(os.path.basename(source))
    names = [stem + extension, f"{stem}_geo{int(geo)}{extension}"] + [
        f"{stem}_geo{int(geo)}_{number}{extension}"
        for number in range(2, 100)]
    for name in names:
        target = os.path.join(input_dir, name)
        holder = taken.get(target)
        if holder is not None and holder != source \
                and not _same_file(holder, source):
            continue  # another geometry's different mesh has this name
        if not (os.path.isfile(target) and _same_file(target, source)):
            shutil.copy2(source, target)
        taken[target] = source
        return target
    raise hou.NodeError(
        f"No free file name for {stem}{extension} in {input_dir}")


def material_file_changed(kwargs):
    """Stage fiber/scalar files, then refresh their material preview."""
    parent = kwargs["node"]
    parm = kwargs.get("parm")
    if parm is not None:
        source = parm.eval().strip()
        if source:
            try:
                staged = _stage_material_file(
                    parent, source,
                    "fiber file" if "fib_file" in parm.name()
                    else "per-element scalar file", parm.name())
            except hou.NodeError as exc:
                _message(str(exc))
                return
            if os.path.abspath(source) != os.path.abspath(staged):
                parm.set(portable_path(parent, staged))
                _status(f"Staged {os.path.basename(staged)} in "
                        f"{os.path.dirname(staged)}")
    material_display_changed(kwargs)


def _resolved_binary(binary):
    return os.path.abspath(shutil.which(binary) or binary) if binary else ""


def polyfem_bin_check(kwargs):
    parent = kwargs["node"]
    polyfem_bin = parent.evalParm("polyfem_bin").strip()
    if polyfem_bin and not os.path.isfile(_resolved_binary(polyfem_bin)):
        _message(f"PolyFEM binary not found at: {polyfem_bin}")


def _write_sop_volume_msh(sop_geo, dst_path):
    """Export tetrahedra/hexahedra from a Houdini SOP Geometry to MSH 2.2.

    Auto-orients every element to strictly positive Gmsh/PolyFEM volume
    (Houdini's TetConform creates negative-signed tets relative to Gmsh) and
    preserves integer subdomain tags from prim attribute Entity/entity/
    subdomain when present as a scalar integer.
    """
    n_pts = sop_geo.intrinsicValue("pointcount")
    if not n_pts:
        raise hou.NodeError("Source SOP has no points.")
    pts = np.frombuffer(
        sop_geo.pointFloatAttribValuesAsString(
            "P", float_type=hou.numericData.Float64),
        dtype=np.float64).reshape(-1, 3)

    entity_vals = None
    for attr_name in ("Entity", "entity", "subdomain"):
        attr = sop_geo.findPrimAttrib(attr_name)
        if attr is not None and attr.dataType() == hou.attribData.Int \
                and attr.size() == 1 and not attr.isArrayType():
            entity_vals = np.frombuffer(
                sop_geo.primIntAttribValuesAsString(attr_name),
                dtype=np.int32).astype(np.int64)
            break

    tets, hexes = [], []
    rows = []
    for prim in sop_geo.prims():
        ptype = prim.type()
        tag = int(entity_vals[prim.number()]) if entity_vals is not None else 1
        if ptype == hou.primType.Tetrahedron:
            verts = [v.point().number() for v in prim.vertices()]
            if len(verts) == 4:
                rows.append((4, len(tets), tag))
                tets.append(verts)
        elif ptype == hou.primType.Hexahedron:
            verts = [v.point().number() for v in prim.vertices()]
            if len(verts) == 8:
                rows.append((5, len(hexes), tag))
                # Houdini hex vertex order [0,1,3,2,4,5,7,6] -> Gmsh [0..7]
                hexes.append([verts[0], verts[1], verts[3], verts[2],
                              verts[4], verts[5], verts[7], verts[6]])

    if not tets and not hexes:
        return {"n_points": n_pts, "n_tets": 0, "n_hexes": 0}

    shift = max(0, 1 - min(tag for _, _, tag in rows))

    tets_arr = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    if len(tets_arr):
        v0 = pts[tets_arr[:, 0]]
        vol6 = np.einsum(
            "ij,ij->i",
            np.cross(pts[tets_arr[:, 1]] - v0, pts[tets_arr[:, 2]] - v0),
            pts[tets_arr[:, 3]] - v0)
        flip = vol6 < 0
        if np.any(flip):
            tmp = tets_arr[flip, 2].copy()
            tets_arr[flip, 2] = tets_arr[flip, 3]
            tets_arr[flip, 3] = tmp

    hexes_arr = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    if len(hexes_arr):
        v0 = pts[hexes_arr[:, 0]]
        vol6 = np.einsum(
            "ij,ij->i",
            np.cross(pts[hexes_arr[:, 1]] - v0, pts[hexes_arr[:, 3]] - v0),
            pts[hexes_arr[:, 4]] - v0)
        flip = vol6 < 0
        if np.any(flip):
            hexes_arr[flip] = hexes_arr[flip][:, [4, 5, 6, 7, 0, 1, 2, 3]]

    n_elems = len(tets_arr) + len(hexes_arr)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{n_pts}\n")
        for idx, (x, y, z) in enumerate(pts.tolist(), 1):
            handle.write(f"{idx} {x!r} {y!r} {z!r}\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{n_elems}\n")
        cells = {4: tets_arr.tolist(), 5: hexes_arr.tolist()}
        for row in solver_element_order(sop_geo, is_direct_sop=True):
            kind, index, tag = rows[row]
            eid = row + 1
            ptag = tag + shift
            vertices = " ".join(str(v + 1) for v in cells[kind][index])
            handle.write(f"{eid} {kind} 2 {ptag} {ptag} {vertices}\n")
        handle.write("$EndElements\n")
    os.replace(tmp_path, dst_path)
    return {"n_points": n_pts, "n_tets": len(tets_arr), "n_hexes": len(hexes_arr)}


def _write_sop_surface_obj(sop_geo, dst_path):
    """Export polygon faces from a Houdini SOP Geometry to a triangulated OBJ."""
    n_pts = sop_geo.intrinsicValue("pointcount")
    if not n_pts:
        raise hou.NodeError("Source SOP has no points.")
    pts = np.frombuffer(
        sop_geo.pointFloatAttribValuesAsString(
            "P", float_type=hou.numericData.Float64),
        dtype=np.float64).reshape(-1, 3)
    tris = []
    for prim in sop_geo.prims():
        if prim.type() != hou.primType.Polygon:
            continue
        verts = [v.point().number() for v in prim.vertices()]
        if len(verts) < 3:
            continue
        v0 = verts[0]
        for i in range(1, len(verts) - 1):
            tris.append((v0, verts[i], verts[i + 1]))
    if not tris:
        return {"n_points": n_pts, "n_tris": 0}
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "w") as handle:
        for x, y, z in pts.tolist():
            handle.write(f"v {x!r} {y!r} {z!r}\n")
        for a, b, c in tris:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
    os.replace(tmp_path, dst_path)
    return {"n_points": n_pts, "n_tris": len(tris)}


def _sop_staging_path(parent, geo, is_obstacle):
    name = f"sop_obs{geo}.obj" if is_obstacle else f"sop_geo{geo}.msh"
    input_dir = os.path.abspath(os.path.join(parent.evalParm("working_dir"), "input"))
    path = os.path.join(input_dir, name)
    if path in _staged_meshes(parent, input_dir, exclude=geo):
        raise hou.NodeError(
            f"Geometry {geo}: {path} is already used by another geometry. "
            "Reload or replace that geometry's mesh first.")
    return path


def load_sop_mesh(kwargs):
    """Stage a SOP mesh. An unsuccessful reload invalidates its solver input."""
    parent = kwargs["node"]
    geo = str(kwargs.get("script_multiparm_index", "1"))
    if _load_sop_mesh(kwargs):
        return True
    parent.parm(f"file_location{geo}").revertToDefaults()
    return False


def _load_sop_mesh(kwargs):
    parent = kwargs["node"]
    geo_number = str(kwargs.get("script_multiparm_index", "1"))
    if not working_dir_check(kwargs):
        return False
    sop_path = parent.evalParm(f"sop_path{geo_number}").strip()
    if not sop_path:
        destroy_geo_nodes(parent, geo_number)
        parent.parm(f"file_location{geo_number}").revertToDefaults()
        _status(f"Geometry {geo_number}: Source SOP cleared.")
        return False
    sop_node = parent.node(sop_path) or hou.node(sop_path)
    if sop_node is None or sop_node.type().category() != hou.sopNodeTypeCategory():
        _message(f"Geometry {geo_number}: SOP node not found: {sop_path}")
        return False
    try:
        sop_node.cook(force=False)
        sop_geo = sop_node.geometry()
    except hou.OperationFailed as exc:
        _message(f"Geometry {geo_number}: failed to cook SOP {sop_path}: {exc}")
        return False

    is_obstacle = bool(parent.evalParm(f"is_obstacle{geo_number}"))
    try:
        dst_abs = _sop_staging_path(parent, geo_number, is_obstacle)
        if is_obstacle:
            stats = _write_sop_surface_obj(sop_geo, dst_abs)
            if stats["n_tris"] == 0:
                _message(
                    f"Geometry {geo_number}: SOP '{sop_path}' has no polygon "
                    "faces to export as an obstacle surface.")
                return False
        else:
            stats = _write_sop_volume_msh(sop_geo, dst_abs)
            if stats["n_tets"] == 0 and stats["n_hexes"] == 0:
                _message(
                    f"Geometry {geo_number}: SOP '{sop_path}' has no "
                    "tetrahedra or hexahedra (only surface polygons). Use a "
                    "Tet Conform / Tet Embed SOP for a simulated volume mesh, "
                    "or enable 'Obstacle (rigid)' to use this surface as an "
                    "obstacle.")
                return False
    except (OSError, hou.Error) as exc:
        _message(f"Geometry {geo_number}: could not export SOP {sop_path}: {exc}")
        return False

    parent.setParms({f"file_location{geo_number}": dst_abs})
    if not create_geo_nodes({
        "node": parent,
        "script_multiparm_index": geo_number,
        "parm_name": f"file_location{geo_number}",
    }):
        return False
    if is_obstacle:
        _status(
            f"Geometry {geo_number}: loaded obstacle surface from SOP "
            f"{sop_node.path()} ({stats['n_tris']} triangles).")
    else:
        _status(
            f"Geometry {geo_number}: loaded volume mesh from SOP "
            f"{sop_node.path()} ({stats['n_tets']} tets, "
            f"{stats['n_hexes']} hexes).")
    if not is_obstacle:
        for note in _sop_subdomain_notes(sop_geo, geo_number):
            _status(note)
    return True


def _sop_subdomain_notes(sop_geo, geo):
    if sop_geo.findPrimAttrib("class") is not None:
        yield (f"Geometry {geo}: ignored primitive 'class' for subdomain "
               "assignment. Use a scalar integer Entity, entity or subdomain.")
    for name in ("Entity", "entity", "subdomain"):
        attr = sop_geo.findPrimAttrib(name)
        if attr is not None and attr.dataType() == hou.attribData.Int \
                and (attr.size() != 1 or attr.isArrayType()):
            yield f"Geometry {geo}: ignored '{name}' because it is an integer tuple or array."


def create_geo_nodes(kwargs):
    """File parm callback: (re)build the import chain for one geometry.

    Picking a mesh for a geometry that already has one is a reload: the
    subdomains keep their materials, sidesets and initial conditions, matched
    by Gmsh physical group (subdomain numbers follow the sorted groups, see
    physical_groups), and anything that could not be kept is reported.
    """
    parent = kwargs["node"]
    parent.allowEditingOfContents()
    geo_number = str(kwargs["script_multiparm_index"])
    # If the user toggled Obstacle while Mesh Source is Houdini SOP, re-export
    # the SOP in the new format (.obj surface vs .msh volume).
    src_parm = parent.parm(f"mesh_source{geo_number}")
    if kwargs.get("parm_name") == f"is_obstacle{geo_number}" \
            and src_parm is not None and src_parm.evalAsString() == "sop" \
            and parent.evalParm(f"sop_path{geo_number}").strip():
        load_sop_mesh(kwargs)
        return
    path_parm = parent.parm("file_location" + geo_number).eval()
    if not working_dir_check(kwargs):
        parent.parm("file_location" + geo_number).revertToDefaults()
        return

    if not path_parm.strip():
        # The field was emptied: the geometry has no mesh any more. Its
        # subdomains are reset when the next mesh is loaded (a first load,
        # see _apply_subdomains).
        destroy_geo_nodes(parent, geo_number)
        _status(f"Geometry {geo_number}: mesh removed. Its sidesets, "
                "conditions and extra subdomains are cleared when the next "
                "mesh is loaded.")
        return

    existing_node = parent.node("geo_" + geo_number)
    existing_geo = existing_node is not None
    prev_path = None
    if existing_geo:
        src_parm = existing_node.parm("File") or existing_node.parm("file")
        if src_parm is not None:
            prev_path = src_parm.unexpandedString()

    def _restore_file_parm():
        if prev_path:
            parent.setParms({f"file_location{geo_number}": prev_path})
        else:
            parent.parm("file_location" + geo_number).revertToDefaults()

    root, extension = os.path.splitext(path_parm)
    extension = extension.lower()
    non_msh = extension in (".obj", ".stl", ".ply")
    if not non_msh and extension != ".msh":
        _message(
            "Only .msh is supported for simulated geometry; "
            ".obj/.stl/.ply/.msh for obstacles.")
        parent.parm("file_location" + geo_number).revertToDefaults()
        destroy_geo_nodes(parent, geo_number)
        return

    obstacle_check = 1 if non_msh else parent.parm(
        "is_obstacle" + geo_number).eval()

    # Stage and validate the new mesh before destroying the existing chain,
    # so a missing, unreadable or invalid file does not wipe loaded geometry.
    working_dir = parent.evalParm("working_dir")
    input_dir = os.path.join(working_dir, "input")
    try:
        dst_abs = _stage_mesh(
            geo_number, path_parm, input_dir,
            _staged_meshes(parent, input_dir, exclude=geo_number))
    except (OSError, hou.Error) as e:
        _message(f"Could not copy mesh into working dir: {e}")
        _restore_file_parm()
        return
    digest = _file_digest(dst_abs)
    dst = portable_path(parent, dst_abs)

    stale_probe = parent.node("__staging_probe")
    if stale_probe is not None:
        stale_probe.destroy()
    if obstacle_check == 0 or not non_msh:
        file_node = parent.createNode("MSH_Reader::3.0", "__staging_probe")
        file_node.parm("File").set(dst)
    else:
        file_node = parent.createNode("file", "__staging_probe")
        file_node.parm("file").set(dst)
    try:
        file_node.cook(force=True)
    except hou.OperationFailed:
        file_node.destroy()
        _message("Failed to read the mesh; check the file.")
        _restore_file_parm()
        return

    groups = None
    if obstacle_check == 0:
        if file_node.geometry().findPrimAttrib("Entity") is None:
            file_node.destroy()
            _message("Mesh has no Entity attribute.")
            _restore_file_parm()
            return
        groups = physical_groups(file_node.geometry())
        if not groups["tags"]:
            file_node.destroy()
            _message("Mesh has no elements.")
            _restore_file_parm()
            return
        if len(groups["tags"]) > 999:
            file_node.destroy()
            _message(f"The mesh has {len(groups['tags'])} physical groups; "
                     "this node supports up to 999 subdomains per geometry.")
            _restore_file_parm()
            return

    saved_obstacle_color = _color_tuple(
        parent, f"color_{geo_number}") if existing_geo else None
    # What a reload must keep, captured before the old chain goes away.
    previous = _capture_previous_mesh(parent, geo_number) \
        if existing_geo else None
    destroy_geo_nodes(parent, geo_number)
    file_node.setName("geo_" + geo_number)

    if non_msh:
        parent.setParms({"is_obstacle" + geo_number: 1})
    # Remembered on the node: a reload compares the new file with the one
    # that was loaded, which may since have been overwritten in place.
    parent.setUserData(_mesh_digest_key(geo_number), digest)
    parent.setParms({f"file_location{geo_number}": dst})

    upstream = file_node
    if obstacle_check == 0:
        upstream = parent.createNode("attribwrangle",
                                     "subdomains_" + geo_number)
        upstream.setParms({"class": 1, "group": "@is_volume==1",
                           "grouptype": 4,
                           "snippet": _subdomain_vex(groups)})
        upstream.setInput(0, file_node)

    attrib_node = parent.createNode("attribcreate::2.0", "attrib_" + geo_number)
    attrib_node.setParms({"name1": "geometry_num", "class1": 1, "type1": 1,
                          "value1v1": int(geo_number)})
    transform_node = parent.createNode("xform", "transform_" + geo_number)
    for parm, channel in (
            ("xOrd", "xform_xOrd__{0}"), ("rOrd", "xform_rOrd__{0}"),
            ("tx", "xform_t__{0}x"), ("ty", "xform_t__{0}y"), ("tz", "xform_t__{0}z"),
            ("rx", "xform_r__{0}x"), ("ry", "xform_r__{0}y"), ("rz", "xform_r__{0}z"),
            ("sx", "xform_s__{0}x"), ("sy", "xform_s__{0}y"), ("sz", "xform_s__{0}z"),
            ("px", "tpivot_{0}x"), ("py", "tpivot_{0}y"), ("pz", "tpivot_{0}z"),
            ("prx", "rpivot_{0}x"), ("pry", "rpivot_{0}y"), ("prz", "rpivot_{0}z")):
        transform_node.parm(parm).setExpression(
            f'ch("../{channel.format(geo_number)}")')
    transform_node.setParms({"addattrib": 1})

    element_select = parent.createNode("groupcreate", "elements_" + geo_number)
    branch = parent.createNode("null", "branch_" + geo_number)
    attrib_node.setNextInput(upstream)
    transform_node.setNextInput(attrib_node)
    element_select.setNextInput(transform_node)
    branch.setNextInput(element_select)

    if obstacle_check == 0:
        notes = _apply_subdomains(parent, geo_number, groups, previous, digest)
        create_geo_tree(kwargs)
        if previous is not None and digest != previous["digest"]:
            # picks name faces of the surface just rebuilt
            notes.extend(_remap_picks(parent, geo_number, previous))
        fetch_elements(kwargs)
        _update_subdomain_info(parent, geo_number)
        if notes:
            _message(f"Geometry {geo_number} was loaded again:\n"
                     + "\n".join("- " + note for note in notes))
        elif previous is not None:
            _status(f"Geometry {geo_number} was loaded again; every subdomain "
                    "kept its settings.")
        source = file_node.geometry()
        if source.countPrimType(hou.primType.Tetrahedron) \
                + source.countPrimType(hou.primType.Hexahedron) == 0:
            _message(f"Geometry {geo_number} has no tetrahedra or hexahedra "
                     "(a 2D or surface mesh). This node writes 3D "
                     "simulations: load a volume mesh, or turn on Obstacle "
                     "to use this surface as a rigid obstacle.")
    else:
        _set_color_tuple(
            parent, f"color_{geo_number}",
            saved_obstacle_color or default_color(geo_number))
        create_obstacle_tree(parent, geo_number)

    # A reload keeps the pivot the user may have placed; a new mesh gets its
    # centroid.
    bbox = pivot_to_centroid(kwargs) if previous is None else \
        parent.node(f"attrib_{geo_number}").geometry().boundingBox()
    sv = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer) \
        if hou.isUIAvailable() else None
    if sv and bbox:
        sv.curViewport().frameBoundingBox(bbox)
    return True


