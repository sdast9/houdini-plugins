"""Helpers for building HDAs programmatically under hython.

Pipeline: build the network inside a fresh subnet, convert it to a digital
asset, attach parameter templates + named sections (PythonModule, etc.),
then save the .hdanc. Reproducible alternative to hand-editing assets.
"""

from contextlib import contextmanager
import os
import shutil
import stat
import tempfile

import hou


_STAGED = {}


def _remove_stage(tmp_path):
    # Houdini can save backup copies of a temporary definition as it edits it.
    # The staging directory owns those copies as well as the library itself.
    shutil.rmtree(os.path.dirname(tmp_path))


def _detach_asset(tmp_path, cleanup_node):
    try:
        if cleanup_node is not None:
            cleanup_node.destroy()
    except hou.ObjectWasDeleted:
        pass
    finally:
        canonical = os.path.realpath(tmp_path)
        for loaded_path in hou.hda.loadedFiles():
            if os.path.realpath(loaded_path) == canonical:
                hou.hda.uninstallFile(loaded_path)


def _discard_asset(staged):
    tmp_path, _, cleanup_node = staged
    try:
        _detach_asset(tmp_path, cleanup_node)
    finally:
        _remove_stage(tmp_path)


@contextmanager
def build_transaction():
    """Discard any staged assets left by a builder, including on exceptions."""
    previous = set(_STAGED)
    original = None
    try:
        yield
    except BaseException as error:
        original = error
        raise
    finally:
        failures = []
        for asset_id in _STAGED.keys() - previous:
            staged = _STAGED.pop(asset_id)
            try:
                _discard_asset(staged)
            except BaseException as error:
                failures.append((staged[0], error))
        if failures:
            first = original if original is not None else failures[0][1]
            for path, error in failures:
                first.add_note(f"Could not discard staged HDA {path}: {error!r}")
            if original is None:
                raise first


def set_replacement_mode(source, destination):
    """Set the mode before replacing a destination with a staged file."""
    try:
        mode = stat.S_IMODE(os.stat(destination).st_mode)
    except FileNotFoundError:
        # Let the OS apply the umask without changing it for other threads.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(source)) as probe_dir:
            fd = os.open(os.path.join(probe_dir, "mode"),
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            try:
                mode = stat.S_IMODE(os.fstat(fd).st_mode)
            finally:
                os.close(fd)
    os.chmod(source, mode)


def new_asset(category, type_name, label, hda_path, min_inputs=0, max_inputs=0):
    """Create an asset in a private temporary library next to its destination.

    Use within build_transaction() and finish with commit_asset() or finalize().
    category: 'Object', 'GeoObject', or 'Sop'.
    """
    if category not in ("Sop", "Object", "GeoObject"):
        raise ValueError(category)
    hda_path = os.path.abspath(hda_path)
    filename = os.path.basename(hda_path)
    stage_dir = tempfile.mkdtemp(
        prefix=f".{filename}.{os.getpid()}.", suffix=".tmp",
        dir=os.path.dirname(hda_path))
    tmp_path = os.path.join(stage_dir, filename + ".tmp")
    cleanup_node = None
    staged = False
    try:
        if category == "Sop":
            cleanup_node = hou.node("/obj").createNode("geo", "hda_build_container")
            subnet = cleanup_node.createNode("subnet", "asset_src")
        else:
            node_type = "subnet" if category == "Object" else "geo"
            subnet = cleanup_node = hou.node("/obj").createNode(node_type, "asset_src")
        asset = subnet.createDigitalAsset(
            name=type_name,
            hda_file_name=tmp_path,
            description=label,
            min_num_inputs=min_inputs,
            max_num_inputs=max_inputs,
            ignore_external_references=True,
            change_node_type=True,
            create_backup=False,
        )
        if category != "Sop":
            cleanup_node = asset
        asset.type().definition().enableCreateBackups(False)
        asset.allowEditingOfContents()
        _STAGED[asset.sessionId()] = (tmp_path, hda_path, cleanup_node)
        staged = True
        return asset
    finally:
        if not staged:
            _discard_asset((tmp_path, hda_path, cleanup_node))


def commit_asset(asset):
    """Clean up temporary build nodes and atomically publish the saved asset."""
    staged = _STAGED.pop(asset.sessionId(), None)
    if staged is None:
        return asset.type().definition().libraryFilePath()
    tmp_path, hda_path, cleanup_node = staged
    try:
        _detach_asset(tmp_path, cleanup_node)
        set_replacement_mode(tmp_path, hda_path)
        os.replace(tmp_path, hda_path)
        return hda_path
    finally:
        _remove_stage(tmp_path)


def read_source(*relpath):
    """Read a source file relative to houdini_HDAs/src/."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, *relpath), encoding="utf-8") as f:
        return f.read()


def finalize(asset, parm_template_group=None, sections=None, tools_shelf=None,
             default_state=None):
    """Attach parm templates and sections, then write the asset definition."""
    asset_id = asset.sessionId()
    try:
        definition = asset.type().definition()

        if parm_template_group is not None:
            definition.setParmTemplateGroup(parm_template_group)

        for name, text in (sections or {}).items():
            definition.addSection(name, text)

        if tools_shelf is not None:
            definition.addSection("Tools.shelf", tools_shelf)

        if default_state is not None:
            definition.addSection("DefaultState", default_state)

        definition.updateFromNode(asset)
        definition.save(definition.libraryFilePath(), template_node=asset,
                        create_backup=False)
        return commit_asset(asset)
    finally:
        staged = _STAGED.pop(asset_id, None)
        if staged is not None:
            _discard_asset(staged)
