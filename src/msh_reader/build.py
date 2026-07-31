"""Build MSH_Reader::3.0 (.hdanc). Run under hython via src/build_all.py."""

import os
import sys

import hou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import hda_build  # noqa: E402

TYPE_NAME = "MSH_Reader::3.0"
LABEL = "MSH Reader 3.0"

PYTHON_SOP_CODE = """\
# Thin shim: all logic lives in the asset's PythonModule section.
hou.pwd().parent().hdaModule().cook(hou.pwd())
"""


def build(out_dir):
    hda_path = os.path.join(out_dir, "sop_MSH_Reader.3.0.hdanc")
    asset = hda_build.new_asset("Sop", TYPE_NAME, LABEL, hda_path)

    # ---- internal network -------------------------------------------------
    py = asset.createNode("python", "read_msh")
    py.parm("python").set(PYTHON_SOP_CODE)

    build_prims = asset.createNode("attribwrangle", "build_prims")
    build_prims.setParms({
        "class": 0,  # detail
        "snippet": hda_build.read_source("msh_reader", "vex", "build_prims.vex"),
    })
    build_prims.setNextInput(py)

    element_attrs = asset.createNode("attribwrangle", "element_attrs")
    element_attrs.setParms({
        "class": 1,  # primitive
        "snippet": hda_build.read_source("msh_reader", "vex", "element_attrs.vex"),
    })
    element_attrs.setNextInput(build_prims)

    cleanup = asset.createNode("attribdelete", "cleanup_arrays")
    cleanup.setParms({
        "dtldel": "tet_conn tet_entity hex_conn hex_entity "
                  "tri_conn tri_entity quad_conn quad_entity "
                  "line_conn line_entity mesh_dim",
    })
    cleanup.setNextInput(element_attrs)

    out = asset.createNode("output", "output0")
    out.setNextInput(cleanup)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    asset.layoutChildren()

    # ---- parameters -------------------------------------------------------
    ptg = hou.ParmTemplateGroup()
    file_parm = hou.StringParmTemplate(
        "File", "MSH File", 1,
        string_type=hou.stringParmType.FileReference,
        file_type=hou.fileType.Geometry,
        tags={"filechooser_pattern": "*.msh"},
        help="Gmsh .msh file (v2.2 or v4.1, ASCII or binary; fTetWild "
             "output in any mode). Parsed natively with numpy; no gmsh "
             "installation required.")
    reload_parm = hou.ButtonParmTemplate(
        "reload", "Reload",
        script_callback="hou.phm().reload(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Re-read the file from disk.")
    surf_parm = hou.ToggleParmTemplate(
        "import_surfaces", "Import Physical Surfaces", default_value=False,
        help="For 3D meshes, also import 2D physical groups "
             "(gmsh-authored sidesets) as polygons with a surface_entity "
             "attribute. Triangles and quads are always imported as primary "
             "elements for 2D meshes.")
    ptg.append(file_parm)
    ptg.append(reload_parm)
    ptg.append(surf_parm)

    # ---- PythonModule: embed the native parser ----------------------------
    parser_src = hda_build.read_source("common", "msh_parser.py")
    module_src = hda_build.read_source("msh_reader", "PythonModule.py")
    module_src = module_src.replace("# @MSH_PARSER@", parser_src)
    module_src += """

def reload(kwargs):
    node = kwargs["node"]
    inner = node.node("read_msh")
    if inner is not None:
        inner.cook(force=True)
"""

    path = hda_build.finalize(
        asset,
        parm_template_group=ptg,
        sections={"PythonModule": module_src},
    )
    return path


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")
    print("built:", build(os.path.abspath(out_dir)))
