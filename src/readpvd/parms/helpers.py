"""Shared parameter templates and the follow-running restore handler."""

import hou

# A conditional must compare a parameter: "{ 1 == 1 }" names none, so Houdini
# ignores it -- until 2026-10-01 the internal parameters below were visible
# and the status fields editable. Internal parameters now carry the
# template's hidden flag; read-only fields compare the file path with a value
# no path takes.
READ_ONLY = "{ PVD_file != __readpvd_read_only__ }"
TIME_MAPPING_TOKENS = ("auto", "steps", "time")


def _callback(name):
    return {"script_callback": f"hou.phm().{name}(kwargs)",
            "script_callback_language": hou.scriptLanguage.Python}


def _menu_script(template, name):
    template.setItemGeneratorScript(f"hou.phm().{name}(kwargs)")
    template.setItemGeneratorScriptLanguage(hou.scriptLanguage.Python)
    return template


def _read_only_text(name, label, expression, help_text, lines=None):
    template = hou.StringParmTemplate(
        name, label, 1, default_expression=(expression,),
        default_expression_language=(hou.scriptLanguage.Python,),
        tags={"editor": "1", "editorlines": lines} if lines else {},
        help=help_text)
    template.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    return template


def _status_text(name, label, help_text, lines="2-8"):
    template = hou.StringParmTemplate(
        name, label, 1, default_value=("",),
        tags={"editor": "1", "editorlines": lines}, help=help_text)
    template.setConditional(hou.parmCondType.DisableWhen, READ_ONLY)
    return template


def _restore_follow(node):
    try:
        follow = node.parm("follow_running")
        if follow is not None and follow.eval():
            node.hdaModule().toggle_follow_running({"node": node}, restoring=True)
    except hou.ObjectWasDeleted:
        pass
