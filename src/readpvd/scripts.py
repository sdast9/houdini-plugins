"""Python SOP snippets and lifecycle event scripts embedded in Read PVD."""

from inspect import getsource

from .parms.helpers import _restore_follow


TOPO_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("topoframe")  # cook dependencies
node.evalParm("dataepoch")
node.parent().hdaModule().cook_topology(node)
"""

FRAME_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("frame")  # cook dependencies
node.evalParm("dataepoch")
node.parent().hdaModule().cook_frame(node)
"""

LEGEND_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_scene_legend(node)
"""

GNOMON_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_gnomon(node)
"""

MULTI_BLOCK_SOP_CODE = """\
node = hou.pwd()
node.evalParm("pvdpath"); node.evalParm("frame")
node.evalParm("dataepoch")
node.parent().hdaModule().cook_multi_blocks(node)
"""

REFERENCE_SOP_CODE = """\
node = hou.pwd()
node.evalParm("frame")
node.evalParm("dataepoch")
node.parent().hdaModule().cook_reference_comparison(node)
"""

SMOOTH_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_field_smoothing(node)
"""

GLYPH_RECOVERY_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_glyph_recovery(node)
"""

FIBER_RECOVERY_SOP_CODE = """\
node = hou.pwd()
node.parent().hdaModule().cook_fiber_recovery(node)
"""


# Houdini's built-in "Infra-Red" ramp preset (blue -> cyan -> green -> yellow
# -> red) used as the default Displayed Color Ramp -- no black or white ends.
_INFRARED_POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_INFRARED_COLORS = ((0.2, 0.0, 1.0), (0.0, 0.85, 1.0), (0.0, 1.0, 0.1),
                    (0.95, 1.0, 0.0), (1.0, 0.0, 0.0))

# Ramp colour defaults can't be expressed on a RampParmTemplate, so a new node
# gets the infra-red palette via an OnCreated handler (runs once at creation,
# leaving ramps in saved scenes untouched).
ON_CREATED_CODE = f"""\
import hou
_limit = kwargs["node"].parm("cache_memory_gb")
if _limit is not None:
    _limit.set(kwargs["node"].hdaModule().available_memory_gb())
_parm = kwargs["node"].parm("color_ramp")
if _parm is not None:
    _parm.set(hou.Ramp(
        [hou.rampBasis.Linear] * {len(_INFRARED_POSITIONS)},
        {list(_INFRARED_POSITIONS)},
        {[list(color) for color in _INFRARED_COLORS]}))
"""

# Embed the helper so the event section is self-contained in the installed HDA.
ON_LOADED_CODE = """\
import hou

""" + getsource(_restore_follow) + """
_node = kwargs["node"]
if hou.isUIAvailable():
    import hdefereval
    hdefereval.executeDeferred(_restore_follow, _node)
else:
    _restore_follow(_node)
"""
