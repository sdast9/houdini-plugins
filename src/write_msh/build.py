"""Build the standalone writeMSH::2 Python SOP from project sources."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import ctime

import hou

if __package__:
    from . import runtime_manifest
else:
    import runtime_manifest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import hda_build  # noqa: E402
import section_build  # noqa: E402
from check_parts import check_hda  # noqa: E402

FILENAME = "sop_writeMSH.2.hdanc"


def build(out_dir):
    target = Path(out_dir).resolve() / FILENAME
    errors = check_hda(HERE.parent, "write_msh")
    if errors:
        raise ValueError("\n".join(map(str, errors)))
    sections, _ = section_build.build_runtime_sections(
        runtime_manifest.ASSET_TAG, runtime_manifest.TYPE_NAME, runtime_manifest.PARTS)
    sections["Help"] = hda_build.read_source("write_msh", "Help")

    # hotl assembles the original Python SOP shape from text sections. Using
    # the Houdini SDK tool keeps the node type compatible with existing scenes.
    hotl = Path(hou.getenv("HFS")) / "bin" / ("hotl.exe" if os.name == "nt" else "hotl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".write_msh_build_", dir=target.parent) as directory:
        work = Path(directory)
        definition = work / "definition"
        shutil.copytree(HERE / "definition", definition)
        library_index = definition / "INDEX__SECTION"
        library_index.write_text(
            library_index.read_text(encoding="utf-8").replace("@BUILD_TIME@", ctime()),
            encoding="utf-8", newline="\n")
        operator = definition / "Sop_1writeMSH_8_82"
        with (operator / "Sections.list").open("a", encoding="utf-8", newline="\n") as index:
            for name, content in sections.items():
                (operator / name).write_text(content, encoding="utf-8", newline="\n")
                index.write(f"{name}\t{name}\n")
        candidate = work / FILENAME
        subprocess.run([str(hotl), "-l", str(definition), str(candidate)], check=True)
        hda_build.set_replacement_mode(candidate, target)
        os.replace(candidate, target)
    return str(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[1])
    args = parser.parse_args()
    print("built:", build(args.output_dir))
