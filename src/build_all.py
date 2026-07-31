"""Build all PolyFEM pipeline HDAs.

Usage: hython src/build_all.py
Emits sop_MSH_Reader.3.0.hdanc, object_stevenabramowitch.dev.PolyFEM.2.0.hdanc,
object_readPVD.1.0.hdanc next to the legacy assets in houdini_HDAs/, and
installs a copy of each into the Houdini user otls directory so running
sessions pick them up after a Refresh Asset Libraries / restart.
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

for pkg in ("msh_reader", "polyfem", "readpvd"):
    sys.path.insert(0, os.path.join(HERE, pkg))

import importlib

for pkg in ("msh_reader", "polyfem", "readpvd"):
    module = importlib.import_module(f"{pkg}.build") if False else None

# import explicitly (each build.py defines build(out_dir))
sys.path.insert(0, HERE)
from msh_reader import build as msh_build  # noqa: E402
from polyfem import build as polyfem_build  # noqa: E402
from readpvd import build as readpvd_build  # noqa: E402

OTLS_DIR = os.path.expanduser("~/Library/Preferences/houdini/22.0/otls")

for builder in (msh_build, polyfem_build, readpvd_build):
    built = builder.build(OUT)
    print("built:", built)
    if os.path.isdir(OTLS_DIR):
        shutil.copy(built, OTLS_DIR)
        print("installed:", os.path.join(OTLS_DIR, os.path.basename(built)))
    else:
        print(f"WARNING: {OTLS_DIR} not found; HDA not installed")
