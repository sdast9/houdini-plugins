def save_geo(node):
    """Write on demand and leave the SOP's pass-through geometry intact."""
    try:
        path = node.evalParm("file_to_save").strip()
        if not path:
            raise hou.NodeError("Choose an output .msh file in File to Save.")
        if node.input(0) is None:
            raise hou.NodeError("Connect a tetrahedral mesh to the input.")
        stats = _write_sop_volume_msh(
            node.geometry(), path, allow_hexes=False, orient=node.evalParm("flip"),
            tag_attributes=("class", *_SOP_VOLUME_TAGS), allow_tag_zero=True)
    except (hou.Error, OSError) as exc:
        if hou.isUIAvailable():
            hou.ui.displayMessage(str(exc), severity=hou.severityType.Error)
            return False
        raise hou.NodeError(str(exc)) from exc

    message = (f"Wrote {stats['n_tets']} tetrahedra in {stats['n_groups']} "
               f"regions to {path}")
    if hou.isUIAvailable():
        hou.ui.setStatusMessage(message, severity=hou.severityType.ImportantMessage)
    else:
        print(message)
    return True
