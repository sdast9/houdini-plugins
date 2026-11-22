"""Ordered Python runtime sections for the Read PVD HDA."""

ASSET_TAG = "readpvd"
TYPE_NAME = "readPVD::1.0"
PARTS = [
    ("00_header_h5py", ("readpvd", "runtime", "00_header_h5py.py")),
    ("vtu_parser", ("common", "vtu_parser.py")),
    ("msh_parser", ("common", "msh_parser.py")),
    ("xlsx_writer", ("common", "xlsx_writer.py")),
    ("10_fields_fibers", ("readpvd", "runtime", "10_fields_fibers.py")),
    ("20_timeline_meta", ("readpvd", "runtime", "20_timeline_meta.py")),
    ("25_menus_options", ("readpvd", "runtime", "25_menus_options.py")),
    ("30_field_eval_probe", ("readpvd", "runtime", "30_field_eval_probe.py")),
    ("35_reference_scan_blocks", ("readpvd", "runtime", "35_reference_scan_blocks.py")),
    ("40_legend_gnomon", ("readpvd", "runtime", "40_legend_gnomon.py")),
    ("45_cook_stages", ("readpvd", "runtime", "45_cook_stages.py")),
    ("50_ui_callbacks", ("readpvd", "runtime", "50_ui_callbacks.py")),
    ("55_follow_state", ("readpvd", "runtime", "55_follow_state.py")),
    ("56_follow_updates", ("readpvd", "runtime", "56_follow_updates.py")),
    ("60_force_curves", ("readpvd", "runtime", "60_force_curves.py")),
    ("65_export_mesh_quality", ("readpvd", "runtime", "65_export_mesh_quality.py")),
    ("70_export_deformed_mesh", ("readpvd", "runtime", "70_export_deformed_mesh.py")),
    ("75_export_units_fields", ("readpvd", "runtime", "75_export_units_fields.py")),
    ("80_export_regions", ("readpvd", "runtime", "80_export_regions.py")),
    ("85_export_data", ("readpvd", "runtime", "85_export_data.py")),
]

# The old builder replaced each include marker in place. Preserve the source
# lines taken up by its surrounding comments, blank lines, and marker newlines.
COMPILE_LINE_GAPS = {
    "vtu_parser": 1,
    "msh_parser": 5,
    "xlsx_writer": 4,
    "10_fields_fibers": 2,
}
