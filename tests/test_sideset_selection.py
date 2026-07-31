"""Headless regression test for multi-subdomain sideset selection.

Covers the 2.0 regression where every entity of a geometry stayed visible and
pickable while assigning faces to one subdomain's sideset (1.2 isolated the
subdomain via its per-volume split branch), and where chaining the group nodes
through the display chain orphaned all but the last-rebuilt subdomain.

Run: hython tests/test_sideset_selection.py
"""

import os
import tempfile

import hou
import numpy as np

from test_polyfem_hda import BASE
from test_polyfem_colors import make_two_volume_msh


def entity_array(node, name):
    geometry = node.node(name).geometry()
    return np.asarray(geometry.primIntAttribValues("Entity"))


def add_sideset(node, mod, geo, vol, count):
    node.setParms({f"sideset_selection{geo}_{vol}": count})
    mod.create_group({"node": node, "script_multiparm_index2": str(geo),
                      "script_multiparm_index": str(vol)})


def main():
    hou.hda.installFile(os.path.join(BASE, "sop_MSH_Reader.3.0.hdanc"))
    hou.hda.installFile(
        os.path.join(BASE, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc"))

    work = tempfile.mkdtemp(prefix="polyfem_sideset_test_")
    input_dir = os.path.join(work, "input")
    os.makedirs(input_dir, exist_ok=True)
    mesh = os.path.join(work, "two_boxes.msh")
    make_two_volume_msh(mesh)

    node = hou.node("/obj").createNode(
        "stevenabramowitch::dev::PolyFEM::2.0", "sideset_test")
    mod = node.hdaModule()
    node.setParms({"working_dir": work + "/"})
    node.setParms({"file_location1": mesh})
    node.parm("file_location1").pressButton()
    assert node.evalParm("num_volumes1") == 2

    add_sideset(node, mod, 1, 1, 2)
    add_sideset(node, mod, 1, 2, 1)

    surface = entity_array(node, "entitycolor_1")
    faces = {vol: int((surface == vol).sum()) for vol in (1, 2)}
    assert faces[1] and faces[2], f"expected faces on both subdomains: {faces}"

    # soputils.selectGroupParm display-flags the group node's input 0 and picks
    # from it, so that node decides what is visible and selectable.
    for vol, num, expected in ((1, 1, "solo_1_1"), (1, 2, "group_1_1_1"),
                               (2, 1, "solo_1_2")):
        group = node.node(f"group_1_{vol}_{num}")
        assert group is not None, f"group_1_{vol}_{num} missing"
        selection = group.inputs()[0]
        assert selection is not None and selection.name() == expected, \
            (f"group_1_{vol}_{num} picks from "
             f"{selection.name() if selection else None}, expected {expected}")

    for vol in (1, 2):
        solo = node.node(f"solo_1_{vol}")
        assert solo is not None, f"solo_1_{vol} missing"
        assert solo.type().name() == "visibility", \
            "subdomain isolation must hide, not delete (prim numbers matter)"
        assert solo.inputs()[0].name() == "entitycolor_1"
        assert solo.evalParm("group") == f"@Entity!={vol}", \
            f"solo_1_{vol} does not hide the other subdomains"
        # Visibility is display-only: the export indexes basegroup prim
        # numbers straight into the geometry-wide surface arrays.
        assert np.array_equal(entity_array(node, f"solo_1_{vol}"), surface), \
            f"solo_1_{vol} changed primitive numbering"

    # The display chain keeps every entity; the group chains hang off it.
    assert node.node("null_1").inputs()[0].name() == "entitycolor_1", \
        "sideset group chain was spliced into the display chain"
    for vol in (1, 2):
        assert int((entity_array(node, "output") == vol).sum()) == faces[vol], \
            f"subdomain {vol} missing from the displayed output"

    # Both subdomains' chains survive each other's rebuild (orphaning fix).
    add_sideset(node, mod, 1, 1, 2)
    add_sideset(node, mod, 1, 2, 1)
    for vol, num in ((1, 1), (1, 2), (2, 1)):
        group = node.node(f"group_1_{vol}_{num}")
        upstream = group
        while upstream.inputs() and upstream.inputs()[0] is not None:
            upstream = upstream.inputs()[0]
        assert upstream.name() == "geo_1", \
            f"group_1_{vol}_{num} was orphaned from the geometry"

    # A pick made entirely on another subdomain used to be filtered away into
    # an empty sideset; it must fail loudly instead.
    vol1 = np.nonzero(surface == 1)[0]
    vol2 = np.nonzero(surface == 2)[0]
    node.setParms({"grouptype1_2_1": 0,
                   "basegroup1_2_1": " ".join(str(p) for p in vol1[:8]),
                   "grouptype1_1_1": 0,
                   "basegroup1_1_1": " ".join(str(p) for p in vol1[:4]),
                   "basegroup1_1_2": "*"})
    try:
        mod.export_sidesets(node, 1, 2, input_dir)
    except hou.NodeError as exc:
        assert "none of which are on subdomain" in str(exc), str(exc)
    else:
        raise AssertionError(
            "cross-subdomain sideset selection exported silently")

    # The same selection made correctly still exports.
    node.setParms({"basegroup1_2_1": " ".join(str(p) for p in vol2[:8])})
    surface_sel, _ = mod.export_sidesets(node, 1, 2, input_dir)
    assert surface_sel, "valid sideset selection produced no export"

    # "*" keeps meaning "every face of this subdomain".
    rows = np.loadtxt(os.path.join(input_dir, "surface_sidesets1_tri.txt"),
                      dtype=np.int64, ndmin=2)
    wildcard_id = mod.boundary_id(1, 1, 2, 1)
    assert int((rows[:, 0] == wildcard_id).sum()) == faces[1], \
        '"*" no longer selects the whole subdomain'

    # Dropping a subdomain's sidesets tears its branch down and leaves the
    # others alone.
    add_sideset(node, mod, 1, 2, 0)
    assert node.node("solo_1_2") is None, "solo node leaked"
    assert node.node("group_1_2_1") is None, "group node leaked"
    assert node.node("solo_1_1") is not None, "wrong subdomain torn down"
    assert node.node("null_1").inputs()[0].name() == "entitycolor_1"

    # Rebuilding the display chain (subdomain edits) restores the branches.
    mod.update_entities({"node": node, "script_multiparm_index": "1"})
    assert node.node("solo_1_1") is not None, \
        "solo node not rebuilt after display-chain rebuild"
    assert node.node("group_1_1_1").inputs()[0].name() == "solo_1_1"
    assert node.node("null_1").inputs()[0].name() == "entitycolor_1"

    print("PASS: PolyFEM sideset selection isolation")


if __name__ == "__main__":
    main()
