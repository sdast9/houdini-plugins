# readPVD 1.0 -- PolyFEM ParaView result viewer.
#
# Architecture (built for >1M element meshes):
#   topo_build  (python, time-independent) parse ONE frame's topology ->
#               bulk points + detail connectivity arrays + topo_key
#   build_prims (VEX detail wrangle)       tets/hexes/tris from arrays
#   boundary    (VEX prim wrangle)         display surface (volume prims
#               removed; runs only when topology changes)
#   frame_data  (python, time-dependent)   per-frame BULK upload of P and
#               every PointData field (generic; polyfem's 'stess' typo
#               aliased); topology rebuild only when topo_key changes
#   deform/derived/glyphs/color            VEX (+ optional OpenCL eigen)
#
# Per-frame cost = vtu parse + attribute upload only.

import base64
import bisect
import hashlib
import importlib
import io
import math
import os
import json
import platform
import re
import itertools
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

import hou


# The official h5py 3.16.0 wheels for Houdini 22's CPython 3.13, embedded in
# the asset (base64 sections), one per platform Houdini 22 runs on.
EMBEDDED_H5PY_SECTIONS = {
    ("darwin", "arm64"): "h5py-3.16.0-cp313-macos-arm64.whl.b64",
    ("darwin", "x86_64"): "h5py-3.16.0-cp313-macos-x86_64.whl.b64",
    ("linux", "x86_64"): "h5py-3.16.0-cp313-linux-x86_64.whl.b64",
    ("win32", "amd64"): "h5py-3.16.0-cp313-windows-amd64.whl.b64",
}
# The Linux wheel is manylinux_2_28: it needs glibc 2.28 or newer.
_LINUX_GLIBC = (2, 28)


def h5py_platform_key():
    """(sys.platform, machine) in the spelling of EMBEDDED_H5PY_SECTIONS."""
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "x64": "amd64"}.get(machine, machine)
    if sys.platform == "win32" and machine == "x86_64":
        machine = "amd64"
    if sys.platform != "win32" and machine == "amd64":
        machine = "x86_64"
    return sys.platform, machine


def embedded_h5py_section(key=None, python=None):
    """(section name, None) of the bundled h5py wheel that runs here, or
    (None, why not)."""
    key = key or h5py_platform_key()
    python = python or sys.version_info[:2]
    if tuple(python) != (3, 13):
        return None, (f"the bundled HDF5 reader is built for Houdini 22's "
                      f"Python 3.13, and this is Python "
                      f"{python[0]}.{python[1]}")
    section = EMBEDDED_H5PY_SECTIONS.get(tuple(key))
    if section is None:
        return None, ("there is no bundled HDF5 reader for this platform "
                      f"({key[0]}, {key[1]}); bundled: macOS (Apple silicon "
                      "and Intel), Linux x86_64, Windows x64")
    if key[0] == "linux":
        library, version = platform.libc_ver()
        try:
            found = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:
            found = ()
        if library != "glibc" or found < _LINUX_GLIBC:
            have = f"{library} {version}".strip() or "no glibc"
            return None, ("the bundled Linux HDF5 reader needs glibc 2.28 or "
                          f"newer (this system has {have})")
    return section, None


def _load_embedded_h5py():
    """Install the asset's pinned h5py wheel into a versioned user cache.

    The wheel is embedded in the .hdanc, so reading PolyFEM's default HDF5
    output works offline and does not modify Houdini's own Python installation.
    A normal h5py installation, when present, is preferred by _import_h5py.
    """
    section_name, reason = embedded_h5py_section()
    if section_name is None:
        raise ImportError(
            f"The bundled HDF5 reader cannot be used: {reason}. Install h5py "
            "into Houdini's Python environment, then restart Houdini (or "
            "turn off Use HDF5 on the PolyFEM node).")

    node_type = hou.nodeType(hou.objNodeTypeCategory(), "readPVD::1.0")
    definition = node_type.definition() if node_type is not None else None
    section = (definition.sections().get(section_name)
               if definition is not None else None)
    if section is None:
        raise ImportError(
            "This Read PVD asset does not contain its HDF5 reader. Reinstall "
            "the current object_readPVD.1.0.hdanc asset.")
    wheel = base64.b64decode(section.contents().encode("ascii"))
    digest = hashlib.sha256(wheel).hexdigest()
    pref_dir = hou.expandString("$HOUDINI_USER_PREF_DIR")
    vendor_dir = os.path.join(
        pref_dir, "python3.13libs", f"readpvd_h5py_{digest[:12]}")
    marker = os.path.join(vendor_dir, ".complete")
    if not os.path.isfile(marker):
        if os.path.isdir(vendor_dir):
            shutil.rmtree(vendor_dir)
        os.makedirs(vendor_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            for member in archive.infolist():
                normalized = os.path.normpath(member.filename)
                if os.path.isabs(normalized) or normalized.startswith(".."):
                    raise ImportError(
                        "The embedded HDF5 reader contains an unsafe path.")
            archive.extractall(vendor_dir)
        with open(marker, "w", encoding="ascii") as handle:
            handle.write(digest + "\n")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    importlib.invalidate_caches()
    import h5py
    return h5py

