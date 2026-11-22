"""Build readPVD::1.0 (.hdanc). Run under hython via src/build_all.py."""

import base64
import os
import sys

import hou

if not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import hda_build  # noqa: E402
from readpvd import network, parms, scripts  # noqa: E402

TYPE_NAME = "readPVD::1.0"
LABEL = "Read PVD 1.0 (PolyFEM results)"

# Official h5py wheels from PyPI (SHA-256 in vendor/README.md), embedded as
# base64 sections; the section names match EMBEDDED_H5PY_SECTIONS in the
# PythonModule.
H5PY_WHEELS = {
    "h5py-3.16.0-cp313-macos-arm64.whl.b64":
        "h5py-3.16.0-cp313-cp313-macosx_11_0_arm64.whl",
    "h5py-3.16.0-cp313-macos-x86_64.whl.b64":
        "h5py-3.16.0-cp313-cp313-macosx_10_13_x86_64.whl",
    "h5py-3.16.0-cp313-linux-x86_64.whl.b64":
        "h5py-3.16.0-cp313-cp313-manylinux_2_28_x86_64.whl",
    "h5py-3.16.0-cp313-windows-amd64.whl.b64":
        "h5py-3.16.0-cp313-cp313-win_amd64.whl",
}


def build(out_dir):
    hda_path = os.path.join(out_dir, "object_readPVD.1.0.hdanc")
    asset = hda_build.new_asset("GeoObject", TYPE_NAME, LABEL, hda_path)

    network.build_network(asset)

    module = hda_build.read_source("readpvd", "PythonModule.py")
    for marker, source in (("# @VTU_PARSER@", "vtu_parser.py"),
                           ("# @MSH_PARSER@", "msh_parser.py"),
                           ("# @XLSX_WRITER@", "xlsx_writer.py")):
        if marker not in module:
            raise RuntimeError(
                f"Missing {marker} marker in readpvd/PythonModule.py")
        module = module.replace(
            marker, hda_build.read_source("common", source), 1)

    state_name = TYPE_NAME
    definition = asset.type().definition()
    definition.updateFromNode(asset)
    definition.setParmTemplateGroup(parms.build_parm_templates())
    definition.addSection("PythonModule", module)
    for section_name, wheel in H5PY_WHEELS.items():
        wheel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vendor", wheel)
        with open(wheel_path, "rb") as wheel_file:
            definition.addSection(
                section_name,
                base64.b64encode(wheel_file.read()).decode("ascii"))
    definition.addSection("OnCreated", scripts.ON_CREATED_CODE)
    definition.setExtraFileOption("OnLoaded/IsPython", True)
    definition.setExtraFileOption("OnLoaded/IsScript", True)
    definition.addSection("OnLoaded", scripts.ON_LOADED_CODE)
    definition.addSection("Help", hda_build.read_source("readpvd", "Help"))
    definition.addSection(
        "ViewerStateModule",
        hda_build.read_source("readpvd", "ViewerStateModule.py"))
    definition.addSection("DefaultState", state_name)
    definition.addSection(
        "ViewerStateInstall",
        "__import__('viewerstate.utils', fromlist=[None])"
        ".register_pystate_embedded(kwargs['type'])")
    definition.addSection(
        "ViewerStateUninstall",
        "__import__('viewerstate.utils', fromlist=[None])"
        ".unregister_pystate_embedded(kwargs['type'])")
    for section in ("PythonModule", "OnCreated", "ViewerStateModule",
                    "ViewerStateInstall", "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsPython", True)
        definition.setExtraFileOption(f"{section}/IsScript", True)
    for section in ("ViewerStateModule", "ViewerStateInstall",
                    "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsViewerState", True)
    definition.save(definition.libraryFilePath())

    # Hide the inherited object folders (Transform / Render / Misc) from the
    # parameter list. They are merged in by Houdini at save time, so this runs
    # as a second pass on the materialized interface, then re-saves. The parms
    # still exist (transforms/handles keep working); they are just not shown.
    ptg = definition.parmTemplateGroup()
    hidden_any = False
    for folder in ptg.entries():
        if (folder.type() == hou.parmTemplateType.Folder
                and folder.label() in ("Transform", "Render", "Misc")):
            ptg.hide(folder, True)
            hidden_any = True
    if hidden_any:
        definition.setParmTemplateGroup(ptg)
        definition.save(definition.libraryFilePath())
    return hda_build.commit_asset(asset)


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print("built:", build(out_dir))
