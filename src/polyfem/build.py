"""Build stevenabramowitch::dev::PolyFEM::2.0 (.hdanc). Run under hython."""

import os
import re
import sys

import hou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import hda_build  # noqa: E402

TYPE_NAME = "stevenabramowitch::dev::PolyFEM::2.0"
LABEL = "PolyFEM (Dev) 2.0"
STATE_NAME = "stevenabramowitch::dev::PolyFEM::2.0"

# These controls can change the set or values of the canonical pf_* attributes
# consumed by the fiber/material preview.  Keep the long generated DialogScript
# readable and inject one consistent callback while building the HDA.
_MATERIAL_DISPLAY_CALLBACK_PARMS = {
    "materials#_#", "num_fiber_families#_#", "fam_model#_#_#",
    "kappa_source#_#", "kappa#_#", "kappa_expr#_#", "kappa_sop#_#",
    "kappa_attrib#_#", "kappa_file#_#",
    "fib_source#_#", "fib_dir#_#", "fib_expr#_#", "fib_sop#_#",
    "fib_attrib#_#", "fib_file#_#", "fib_axis_origin#_#",
    "fib_axis_dir#_#", "fib_component#_#", "fib_curve#_#",
    "fib_file_field#_#",
    "fam_kappa_source#_#_#", "fam_kappa#_#_#",
    "fam_kappa_expr#_#_#", "fam_kappa_sop#_#_#",
    "fam_kappa_attrib#_#_#", "fam_kappa_file#_#_#",
    "fam_fib_source#_#_#", "fam_fib_dir#_#_#",
    "fam_fib_expr#_#_#", "fam_fib_sop#_#_#",
    "fam_fib_attrib#_#_#", "fam_fib_file#_#_#",
    "fam_fib_axis_origin#_#_#", "fam_fib_axis_dir#_#_#",
    "fam_fib_component#_#_#", "fam_fib_curve#_#_#",
    "fam_fib_file_field#_#_#", "fam_mirror#_#_#",
    "fam_theta#_#_#", "fam_axis_source#_#_#",
    "fam_axis#_#_#", "fam_axis_attrib#_#_#",
}

_MATERIAL_FILE_CALLBACK_PARMS = {
    "kappa_file#_#", "fib_file#_#",
    "fam_kappa_file#_#_#", "fam_fib_file#_#_#",
}


def _inject_material_display_callbacks(text):
    """Attach preview-refresh callbacks to selected parm/multiparm blocks."""
    lines = text.splitlines()
    insertions = []
    for start, line in enumerate(lines):
        if line.strip() not in ("parm {", "multiparm {"):
            continue
        indent = line[:len(line) - len(line.lstrip())]
        direct = indent + "    "
        end = None
        for index in range(start + 1, len(lines)):
            if lines[index] == indent + "}":
                end = index
                break
        if end is None:
            continue
        name = None
        has_callback = False
        has_language = False
        for candidate in lines[start + 1:end]:
            candidate_indent = candidate[
                :len(candidate) - len(candidate.lstrip())]
            if candidate_indent != direct:
                continue
            match = re.match(r'\s*name\s+"([^"]+)"', candidate)
            if match and name is None:
                name = match.group(1)
            if '"script_callback"' in candidate:
                has_callback = True
            if '"script_callback_language"' in candidate:
                has_language = True
        if name not in _MATERIAL_DISPLAY_CALLBACK_PARMS or has_callback:
            continue
        callback = "material_file_changed" if name in \
            _MATERIAL_FILE_CALLBACK_PARMS else "material_display_changed"
        additions = [
            direct + 'parmtag { "script_callback" '
            f'"hou.phm().{callback}(kwargs)" }}']
        if not has_language:
            additions.append(
                direct + 'parmtag { "script_callback_language" "python" }')
        insertions.append((end, additions))

    # A containing multiparm starts before its child parms but ends after them;
    # sort by insertion point rather than discovery order so earlier inserts
    # cannot invalidate a containing block's saved index.
    for index, additions in sorted(
            insertions, key=lambda item: item[0], reverse=True):
        lines[index:index] = additions
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _expand_visibility_macros(text):
    """Expand @SHOW_FOR(...)@ / @SHOW_FOR_DIRECT(...)@ in the DialogScript.

    Houdini conditionals are written as the negative ("hide when ..."), groups
    AND inside braces and OR between them, so "show this parameter for models
    A or B" already needs one !=-per-model. Composite materials double that:
    an isotropic parameter must also show when the volume is a MaterialSum
    whose MATRIX is one of those models, i.e. hide = A and (B or C), which
    expands to two groups. Writing that by hand for every parameter is where
    conditional bugs come from, so the source carries the model list and this
    generates the condition.

        @SHOW_FOR(NeoHookean LinearElasticity)@
            -> visible for those models directly, or as a composite matrix
        @SHOW_FOR_DIRECT(HGOFiber HGODispersion)@
            -> visible only when the volume itself is one of those models
               (fiber families inside a composite have their own parameters)
    """
    def group(parm, tokens):
        return " ".join(f"{parm} != {token}" for token in tokens)

    def direct(match):
        return "{ " + group("materials#_#", match.group(1).split()) + " }"

    def composite(match):
        tokens = match.group(1).split()
        not_listed = group("materials#_#", tokens)
        return (f"{{ {not_listed} materials#_# != MaterialSum }} "
                f"{{ {not_listed} {group('matrix_model#_#', tokens)} }}")

    text = re.sub(r"@SHOW_FOR_DIRECT\(([^)]*)\)@", direct, text)
    text = re.sub(r"@SHOW_FOR\(([^)]*)\)@", composite, text)
    leftover = re.search(r"@SHOW_FOR[A-Z_]*\(", text)
    if leftover:
        raise RuntimeError(
            f"unexpanded visibility macro at offset {leftover.start()}")
    return text


def _patched_dialog_script():
    text = hda_build.read_source("polyfem", "sections", "DialogScript")
    text = _expand_visibility_macros(text)
    text = _inject_material_display_callbacks(text)
    text = re.sub(
        r"^# Dialog script for .* automatically generated$",
        f"# Dialog script for {TYPE_NAME} automatically generated",
        text, count=1, flags=re.M)
    text = re.sub(r"^(\s*name\s+)stevenabramowitch::dev::PolyFEM::1\.2\s*$",
                  rf"\g<1>{TYPE_NAME}", text, count=1, flags=re.M)
    text = re.sub(r"^(\s*script\s+)stevenabramowitch::dev::box_maker::1\.0\s*$",
                  rf"\g<1>{TYPE_NAME}", text, count=1, flags=re.M)
    text = re.sub(r'^(\s*label\s+)"PolyFEM \(Dev\)"\s*$',
                  rf'\g<1>"{LABEL}"', text, count=1, flags=re.M)
    return text


def build(out_dir):
    hda_path = os.path.join(
        out_dir, "object_stevenabramowitch.dev.PolyFEM.2.0.hdanc")
    asset = hda_build.new_asset("GeoObject", TYPE_NAME, LABEL, hda_path)

    # skeleton network (OnCreated also ensures this on instances)
    merge_node = asset.createNode("merge", "all")
    output_node = asset.createNode("output", "output")
    output_node.setInput(0, merge_node)
    asset.layoutChildren()

    definition = asset.type().definition()
    definition.updateFromNode(asset)

    sections = {
        "DialogScript": _patched_dialog_script(),
        "PythonModule": hda_build.read_source("polyfem", "PythonModule.py"),
        "ViewerStateModule": hda_build.read_source(
            "polyfem", "sections", "ViewerStateModule.py"),
        "OnCreated": hda_build.read_source("polyfem", "OnCreated.py"),
        "DefaultState": STATE_NAME,
        "ViewerStateInstall":
            "__import__('viewerstate.utils', fromlist=[None])"
            ".register_pystate_embedded(kwargs['type'])",
        "ViewerStateUninstall":
            "__import__('viewerstate.utils', fromlist=[None])"
            ".unregister_pystate_embedded(kwargs['type'])",
        "Help": hda_build.read_source("polyfem", "sections", "Help"),
        "Tools.shelf": hda_build.read_source(
            "polyfem", "sections", "Tools.shelf"),
    }
    for name, text in sections.items():
        definition.addSection(name, text)

    for section in ("PythonModule", "OnCreated", "ViewerStateModule",
                    "ViewerStateInstall", "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsPython", True)
        definition.setExtraFileOption(f"{section}/IsScript", True)
    for section in ("ViewerStateModule", "ViewerStateInstall",
                    "ViewerStateUninstall"):
        definition.setExtraFileOption(f"{section}/IsViewerState", True)

    definition.save(definition.libraryFilePath())
    return definition.libraryFilePath()


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    print("built:", build(out_dir))
