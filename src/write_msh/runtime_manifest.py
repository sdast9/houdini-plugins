"""Embedded Python for the standalone writeMSH SOP."""

ASSET_TAG = "write_msh"
TYPE_NAME = "writeMSH::2"
PARTS = [
    ("sop_msh_writer", ("common", "sop_msh_writer.py")),
    ("save", ("write_msh", "save.py")),
]
