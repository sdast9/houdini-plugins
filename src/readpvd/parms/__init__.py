"""Assemble the Read PVD parameter interface in display order."""

import hou

from .analysis import build_analysis_tab
from .display import build_display_tab
from .export import _export_tab
from .helpers import READ_ONLY, TIME_MAPPING_TOKENS
from .inspection import build_clip_tab, build_probe_tab
from .main import build_main_tab
from .overlays import build_gnomon_tab, build_legend_tab
from .sources import build_comparison_tab, build_multi_block_tab

__all__ = ["READ_ONLY", "TIME_MAPPING_TOKENS", "build_parm_templates"]


def build_parm_templates():
    ptg = hou.ParmTemplateGroup()
    ptg.append(build_main_tab())
    ptg.append(build_display_tab())
    ptg.append(build_analysis_tab())
    ptg.append(_export_tab())
    ptg.append(build_multi_block_tab())
    ptg.append(build_comparison_tab())
    ptg.append(build_probe_tab())
    ptg.append(build_clip_tab())
    ptg.append(build_legend_tab())
    ptg.append(build_gnomon_tab())

    hidden_flags = ["has_solution_data", "has_glyph_data", "has_multibody",
                    "has_fiber_data"]
    hidden_flags += [f"has_block_{slug}"
                     for slug in ("volume", "surface", "contact", "points")]
    for name in hidden_flags:
        flag = hou.ToggleParmTemplate(
            name, name, default_value=False,
            help="Internal, hidden: set on load to record whether this kind "
                 "of data exists, so the controls that need it can be "
                 "enabled or disabled.")
        flag.hide(True)
        ptg.append(flag)
    return ptg
