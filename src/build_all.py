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
import tempfile

import hou

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

for pkg in ("msh_reader", "polyfem", "readpvd"):
    sys.path.insert(0, os.path.join(HERE, pkg))

# import explicitly (each build.py defines build(out_dir))
sys.path.insert(0, HERE)
from msh_reader import build as msh_build  # noqa: E402
from polyfem import build as polyfem_build  # noqa: E402
from readpvd import build as readpvd_build  # noqa: E402
import hda_build  # noqa: E402 (shared by the builders)

OTLS_DIR = os.path.join(hou.homeHoudiniDirectory(), "otls")


def _install_asset(built, otls_dir):
    destination = os.path.join(otls_dir, os.path.basename(built))
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(built)}.", suffix=".tmp", dir=otls_dir)
    os.close(fd)
    try:
        shutil.copy2(built, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return destination


def main():
    built_assets = []
    for builder in (msh_build, polyfem_build, readpvd_build):
        with hda_build.build_transaction():
            built = builder.build(OUT)
        built_assets.append(built)
        print("built:", built)

    if OTLS_DIR == os.path.join(hou.homeHoudiniDirectory(), "otls") \
            and os.path.isdir(hou.homeHoudiniDirectory()):
        os.makedirs(OTLS_DIR, exist_ok=True)
    if not os.path.isdir(OTLS_DIR):
        message = f"{OTLS_DIR} not found; HDAs were built but not installed"
        if os.environ.get("POLYFEM_REQUIRE_INSTALL") == "1":
            print(f"ERROR: {message}", file=sys.stderr)
            return 1
        print(f"WARNING: {message}", file=sys.stderr)
        return 0

    for built in built_assets:
        print("installed:", _install_asset(built, OTLS_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
