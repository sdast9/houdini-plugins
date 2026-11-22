"""Ordered Python runtime sections for the PolyFEM HDA."""

ASSET_TAG = "polyfem"
TYPE_NAME = "stevenabramowitch::dev::PolyFEM::2.0"
PARTS = [
    ("00_header_vex", ("polyfem", "runtime", "00_header_vex.py")),
    ("05_ui_utils", ("polyfem", "runtime", "05_ui_utils.py")),
    ("06_parm_utils", ("polyfem", "runtime", "06_parm_utils.py")),
    ("10_transforms", ("polyfem", "runtime", "10_transforms.py")),
    ("15_geo_nodes", ("polyfem", "runtime", "15_geo_nodes.py")),
    ("20_subdomains", ("polyfem", "runtime", "20_subdomains.py")),
    ("25_reload_geometry", ("polyfem", "runtime", "25_reload_geometry.py")),
    ("30_sidesets", ("polyfem", "runtime", "30_sidesets.py")),
    ("35_timeline", ("polyfem", "runtime", "35_timeline.py")),
    ("40_time_curves", ("polyfem", "runtime", "40_time_curves.py")),
    ("45_materials", ("polyfem", "runtime", "45_materials.py")),
    ("50_fibers", ("polyfem", "runtime", "50_fibers.py")),
    ("55_json_obstacles_geo", ("polyfem", "runtime", "55_json_obstacles_geo.py")),
    ("60_json_materials_bcs", ("polyfem", "runtime", "60_json_materials_bcs.py")),
    ("65_json_solver", ("polyfem", "runtime", "65_json_solver.py")),
    ("70_write_scene", ("polyfem", "runtime", "70_write_scene.py")),
    ("75_check_setup_geo", ("polyfem", "runtime", "75_check_setup_geo.py")),
    ("76_check_setup_solver", ("polyfem", "runtime", "76_check_setup_solver.py")),
    ("80_runner", ("polyfem", "runtime", "80_runner.py")),
    ("85_import_core", ("polyfem", "runtime", "85_import_core.py")),
    ("90_import_materials", ("polyfem", "runtime", "90_import_materials.py")),
    ("95_import_legacy", ("polyfem", "runtime", "95_import_legacy.py")),
    ("96_import_apply", ("polyfem", "runtime", "96_import_apply.py")),
    ("sop_msh_writer", ("common", "sop_msh_writer.py")),
]

# The parts preserve the original source byte for byte, with no include gaps.
COMPILE_LINE_GAPS = {}
