"""Build the project HDAs.

Usage: hython src/build_all.py
Builds MSH Reader, PolyFEM, Read PVD, and writeMSH in houdini_HDAs/.
The three pipeline assets also publish to the Houdini user otls directory.
The standalone writeMSH build is project-local.
"""

import os
from pathlib import Path
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
from write_msh import build as write_msh_build  # noqa: E402
from common import check_parts  # noqa: E402
import hda_build  # noqa: E402 (shared by the builders)

OTLS_DIR = os.path.join(hou.homeHoudiniDirectory(), "otls")
PIPELINE_BUILDERS = (msh_build, polyfem_build, readpvd_build)


def _build_stage():
    excluded = [Path(directory).resolve() for directory in
                (OUT, OTLS_DIR, *hou.houdiniPath("HOUDINI_OTLSCAN_PATH"))]
    for directory in (tempfile.gettempdir(), "/tmp"):
        parent = Path(directory).resolve()
        if parent.is_dir() and not any(parent.is_relative_to(p) for p in excluded):
            return tempfile.mkdtemp(prefix="polyfem_hda_stage_", dir=parent)
    raise RuntimeError("No system temporary directory outside the HDA output and OTL scan paths")


def _publish_assets(destinations):
    staging = []
    replacements = []
    published = []
    recovery = set()
    try:
        for directory, built_assets in destinations:
            if any(os.path.samefile(directory, target) for target, _ in staging):
                continue
            scratch = tempfile.mkdtemp(prefix=".polyfem_hda_publish_", dir=directory)
            staging.append((directory, scratch))
            for built in built_assets:
                name = os.path.basename(built)
                destination = os.path.join(directory, name)
                temporary = os.path.join(scratch, name + ".new")
                shutil.copyfile(built, temporary)
                hda_build.set_replacement_mode(temporary, destination)
                backup = None
                if os.path.lexists(destination):
                    backup = os.path.join(scratch, name + ".old")
                    shutil.copy2(destination, backup, follow_symlinks=False)
                replacements.append((temporary, destination, backup))

        for temporary, destination, backup in replacements:
            os.replace(temporary, destination)
            published.append((destination, backup))
    except BaseException as error:
        for destination, backup in reversed(published):
            try:
                if backup is None:
                    os.unlink(destination)
                else:
                    os.replace(backup, destination)
            except OSError as rollback_error:
                error.add_note(f"Could not restore {destination}: {rollback_error}")
                if backup is not None:
                    recovery.add(os.path.dirname(backup))
                    error.add_note(f"Previous asset retained at {backup}")
        raise
    finally:
        for _, scratch in staging:
            if scratch not in recovery:
                shutil.rmtree(scratch)


def main():
    errors = check_parts.check_all(HERE)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    stage_dir = _build_stage()
    try:
        pipeline_assets = []
        for builder in PIPELINE_BUILDERS:
            with hda_build.build_transaction():
                pipeline_assets.append(builder.build(stage_dir))
        with hda_build.build_transaction():
            project_assets = [*pipeline_assets, write_msh_build.build(stage_dir)]

        home = hou.homeHoudiniDirectory()
        if OTLS_DIR == os.path.join(home, "otls") and os.path.isdir(home):
            os.makedirs(OTLS_DIR, exist_ok=True)
        install = os.path.isdir(OTLS_DIR)
        if not install and os.environ.get("POLYFEM_REQUIRE_INSTALL") == "1":
            print(f"ERROR: {OTLS_DIR} not found; publication requires installation",
                  file=sys.stderr)
            return 1

        destinations = [(OUT, project_assets)]
        if install:
            destinations.append((OTLS_DIR, pipeline_assets))
        _publish_assets(destinations)
        for built in project_assets:
            name = os.path.basename(built)
            print("built:", os.path.join(OUT, name))
            if install and built in pipeline_assets:
                print("installed:", os.path.join(OTLS_DIR, name))
        if not install:
            print(f"WARNING: {OTLS_DIR} not found; HDAs were built but not installed",
                  file=sys.stderr)
        return 0
    finally:
        shutil.rmtree(stage_dir)


if __name__ == "__main__":
    raise SystemExit(main())
