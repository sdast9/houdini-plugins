"""Ordered Python runtime sections for the MSH Reader HDA."""

ASSET_TAG = "msh_reader"
TYPE_NAME = "MSH_Reader::3.0"
PARTS = [
    ("msh_parser", ("common", "msh_parser.py")),
    ("cook", ("msh_reader", "PythonModule.py")),
]
