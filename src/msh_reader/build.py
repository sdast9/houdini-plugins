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
    with hda_build.build_transaction():
        return _build(out_dir)


def _build(out_dir):
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
        "ptdel": "__msh_loose",
        "dtldel": "tet_conn tet_entity tet_physical tet_order tet_group "
                  "hex_conn hex_entity hex_physical hex_order hex_group "
                  "tri_conn tri_entity tri_physical tri_order tri_group "
                  "quad_conn quad_entity quad_physical quad_order quad_group "
                  "line_conn line_entity line_physical line_order line_group "
                  "mesh_dim",
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
        help="The Gmsh mesh file to load (.msh version 2.2 or 4.1, text or "
             "binary, including fTetWild output). Nothing needs to be "
             "installed: the file is read directly. Tetrahedra/hexahedra "
             "become Houdini primitives, and the file's physical groups "
             "become the 'Entity' attribute that the PolyFEM node uses as "
             "subdomains.")
    reload_parm = hou.ButtonParmTemplate(
        "reload", "Reload",
        script_callback="hou.phm().reload(kwargs)",
        script_callback_language=hou.scriptLanguage.Python,
        help="Read the file again, e.g. after regenerating the mesh with "
             "the same name.")
    surf_parm = hou.ToggleParmTemplate(
        "import_surfaces", "Import Physical Surfaces", default_value=False,
        help="Also load lower-dimensional physical boundary groups tagged in "
             "Gmsh (2D surfaces on 3D meshes, or 1D boundary edges on 2D "
             "meshes) as primitives carrying a 'surface_entity' attribute "
             "(-1 on primary elements). 2D meshes always load their "
             "triangles/quads as the main elements.")
    groups_parm = hou.ToggleParmTemplate(
        "create_groups", "Create Physical Groups", default_value=False,
        help="Create named Houdini primitive groups for each Gmsh physical "
             "group (phys_<name> for primary elements, surf_<name> for "
             "imported boundary patches).")
    summary_parm = hou.StringParmTemplate(
        "mesh_summary", "Mesh Summary", 1,
        default_expression=(
            "hou.pwd().hdaModule().mesh_summary_text(hou.pwd())",),
        default_expression_language=(hou.scriptLanguage.Python,),
        tags={"editor": "1", "editorlines": "2-4"},
        help="Read-only summary of the loaded Gmsh mesh: format version, "
             "dimension, node and element counts, physical groups, loose "
             "higher-order nodes, and basic inverted-tet quality check.")
    summary_parm.setConditional(
        hou.parmCondType.DisableWhen, "{ File != __msh_read_only_sentinel__ }")
    ptg.append(file_parm)
    ptg.append(reload_parm)
    ptg.append(surf_parm)
    ptg.append(groups_parm)
    ptg.append(summary_parm)

    # ---- PythonModule: embed the native parser ----------------------------
    parser_src = hda_build.read_source("common", "msh_parser.py")
    module_src = hda_build.read_source("msh_reader", "PythonModule.py")
    if "# @MSH_PARSER@" not in module_src:
        raise RuntimeError("Missing # @MSH_PARSER@ marker in msh_reader/PythonModule.py")
    module_src = module_src.replace("# @MSH_PARSER@", parser_src, 1)
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
        sections={"PythonModule": module_src,
                  "Help": hda_build.read_source("msh_reader", "Help"),
                  # show the reader's errors AND warnings on the asset
                  # (an inner node's warning does not reach it otherwise)
                  "MessageNodes": "read_msh"},
    )
    return path


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")
    print("built:", build(os.path.abspath(out_dir)))
