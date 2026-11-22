# =============================================================================
# Transform / pivot helpers (UI parity with 1.2)
# =============================================================================


def decompose_rowvec_xyz_4x4(arr16):
    m = hou.Matrix4([arr16[0:4], arr16[4:8], arr16[8:12], arr16[12:16]])
    t = m.extractTranslates()
    r = m.extractRotates()
    s = m.extractScales()
    return (t[0], t[1], t[2]), (r[0], r[1], r[2]), (s[0], s[1], s[2])


def pivot_to_centroid(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    attrib_node = parent.node(f"attrib_{geo_number}")
    if attrib_node:
        bbox = attrib_node.geometry().boundingBox()
        pivot = bbox.center()
        parent.setParms({f"tpivot_{geo_number}x": pivot[0],
                         f"tpivot_{geo_number}y": pivot[1],
                         f"tpivot_{geo_number}z": pivot[2],
                         f"rpivot_{geo_number}x": 0,
                         f"rpivot_{geo_number}y": 0,
                         f"rpivot_{geo_number}z": 0})
        return bbox


def pivot_to_origin(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    parent.setParms({f"tpivot_{geo_number}x": 0, f"tpivot_{geo_number}y": 0,
                     f"tpivot_{geo_number}z": 0, f"rpivot_{geo_number}x": 0,
                     f"rpivot_{geo_number}y": 0, f"rpivot_{geo_number}z": 0})


def geometry_to_origin(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    attrib_node = parent.node(f"attrib_{geo_number}")
    if attrib_node:
        pivot = attrib_node.geometry().boundingBox().center()
        parent.setParms({f"xform_t__{geo_number}x": -pivot[0],
                         f"xform_t__{geo_number}y": -pivot[1],
                         f"xform_t__{geo_number}z": -pivot[2]})


def reset_transform(kwargs):
    parent = kwargs["node"]
    geo_number = kwargs["script_multiparm_index"]
    for prefix in ("xform_t__", "xform_r__", "xform_s__"):
        for axis in "xyz":
            parm = parent.parm(f"{prefix}{geo_number}{axis}")
            if parm is not None and parm.keyframes():
                parm.deleteAllKeyframes()
    parent.setParms({f"xform_t__{geo_number}x": 0, f"xform_t__{geo_number}y": 0,
                     f"xform_t__{geo_number}z": 0, f"xform_r__{geo_number}x": 0,
                     f"xform_r__{geo_number}y": 0, f"xform_r__{geo_number}z": 0,
                     f"xform_s__{geo_number}x": 1, f"xform_s__{geo_number}y": 1,
                     f"xform_s__{geo_number}z": 1})


def reset_zero(kwargs):
    reset_transform(kwargs)
    pivot_to_origin(kwargs)


