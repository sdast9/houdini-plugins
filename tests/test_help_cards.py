"""Help cards: every asset has one, Houdini's help renderer parses it, and
it carries what a first-time user needs.

Until 2026-10-01 the PolyFEM node's Help section was empty and Read PVD and
the MSH Reader had none.

Run: hython tests/test_help_cards.py
"""

import os

import hou
from houdinihelp import api

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS = (
    ("sop_MSH_Reader.3.0.hdanc", hou.sopNodeTypeCategory, "MSH_Reader::3.0",
     ("msh_physical_tag", "msh_physical_names", "Import Physical Surfaces")),
    ("object_stevenabramowitch.dev.PolyFEM.2.0.hdanc",
     hou.objNodeTypeCategory, "stevenabramowitch::dev::PolyFEM::2.0",
     ("== Quick start ==", "Check Setup", "Open Results",
      "`[0, 9.81, 0]` (in m/s^2), __not__ `[0, -9.81, 0]`",
      "exit status 1", "exit status 3", "killed by a signal",
      "docs/hdas.md", "Time Curve", "Start Frame", "Moving obstacles",
      "Nodal Forces")),
    ("object_readPVD.1.0.hdanc", hou.objNodeTypeCategory, "readPVD::1.0",
     ("== Quick start ==", "Auto Range: All Frames", "h5py",
      "docs/hdas.md", "Time Mapping", "Force Curves", "force_curves.csv",
      "== Export ==", "Export Spreadsheet", "__rest shape__")),
)


def block_types(block, found):
    if isinstance(block, dict):
        found.add(block.get("type"))
        for value in block.values():
            block_types(value, found)
    elif isinstance(block, list):
        for value in block:
            block_types(value, found)
    return found


def main():
    pages = api.get_pages()
    for library, category, name, phrases in ASSETS:
        hou.hda.installFile(os.path.join(BASE, library))
        node_type = hou.nodeType(category(), name)
        text = node_type.embeddedHelp()
        assert text.strip(), f"{name}: no help card"
        for phrase in phrases:
            assert phrase in text, (name, phrase)
        assert f"#internal: {name}" in text, name
        data = pages.string_to_json(f"/nodes/{name}", text,
                                    postprocess=False)
        found = block_types(data.get("body", []), set())
        assert {"title", "summary", "h", "link"} <= found, (name, found)
        html = pages.preview(f"/nodes/{name}", text)
        assert "github.com/sdast9/houdini-plugins" in html, name
        print(f"PASS: {name} help card ({len(text)} characters) parses: "
              f"{', '.join(sorted(t for t in found if t))}")


if __name__ == "__main__":
    main()
