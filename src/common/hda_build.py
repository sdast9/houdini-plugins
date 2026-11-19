"""Helpers for building HDAs programmatically under hython.

Pipeline: build the network inside a fresh subnet, convert it to a digital
asset, attach parameter templates + named sections (PythonModule, etc.),
then save the .hdanc. Reproducible alternative to hand-editing assets.
"""

import os

import hou


_STAGED = {}


def new_asset(category, type_name, label, hda_path, min_inputs=0, max_inputs=0):
    """Create an empty digital asset staged at ``<hda_path>.tmp`` and return
    its instance node. Call ``commit_asset(asset)`` (or ``finalize(...)``)
    after saving so the finished asset atomically replaces ``hda_path`` and
    temporary build nodes are removed.

    category: 'Object', 'GeoObject', or 'Sop'.
    """
    tmp_path = hda_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    container = None
    if category == "Sop":
        container = hou.node("/obj").createNode("geo", "hda_build_container")
        subnet = container.createNode("subnet", "asset_src")
    elif category == "Object":
        subnet = hou.node("/obj").createNode("subnet", "asset_src")
    elif category == "GeoObject":
        # Object-level asset whose children are SOPs (derived from geo)
        subnet = hou.node("/obj").createNode("geo", "asset_src")
    else:
        raise ValueError(category)

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
    asset.allowEditingOfContents()
    _STAGED[asset.sessionId()] = (tmp_path, hda_path, container or asset)
    return asset


def commit_asset(asset):
    """Clean up temporary build nodes and atomically move the staged HDA into
    its final path."""
    staged = _STAGED.pop(asset.sessionId(), None)
    if staged is None:
        return asset.type().definition().libraryFilePath()
    tmp_path, hda_path, cleanup_node = staged
    try:
        cleanup_node.destroy()
    except hou.ObjectWasDeleted:
        pass
    try:
        hou.hda.uninstallFile(tmp_path)
    except hou.OperationFailed:
        pass
    os.replace(tmp_path, hda_path)
    return hda_path


def read_source(*relpath):
    """Read a source file relative to houdini_HDAs/src/."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, *relpath), encoding="utf-8") as f:
        return f.read()


def finalize(asset, parm_template_group=None, sections=None, tools_shelf=None,
             default_state=None):
    """Attach parm templates and sections, then write the asset definition."""
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
    definition.save(definition.libraryFilePath(), template_node=asset)
    return commit_asset(asset)
