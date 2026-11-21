"""Checkpoint metadata and midpoint parity checks; needs h5py/numpy, not hou."""
import json
import os
import shutil
import subprocess
import sys

work, POLYFEM_BIN = sys.argv[1:3]
out_dir = os.path.join(work, "output")

# Self-contained checkpoint metadata replaces the old two-file restart.
try:
    import h5py
except ImportError:
    # Fall back to Read PVD's bundled h5py wheel when run under hython.
    import hou
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hou.hda.installFile(os.path.join(base, "object_readPVD.1.0.hdanc"))
    pvd_node = hou.node("/obj").createNode("readPVD::1.0", "_ckpt_h5py")
    h5py = pvd_node.hdaModule()._import_h5py()
    pvd_node.destroy()

import numpy as np

for step in range(1, 5):
    checkpoint_path = os.path.join(out_dir, f"checkpoint_{step}.hdf5")
    assert os.path.isfile(checkpoint_path), checkpoint_path
    with h5py.File(checkpoint_path) as checkpoint:
        meta = checkpoint["checkpoint/metadata"]
        assert int(meta["output_index"][()]) == step
        assert float(meta["time"][()]) == step * 0.125
        assert float(meta["dt"][()]) == 0.125
        assert int(meta["remaining_steps"][()]) == 4 - step

# Relocated checkpoint plus minimal JSON wrapper exercises the public
# input/checkpoint/path route, independent of original launch directory.
resumed_dir = os.path.join(work, "resumed", "output")
os.makedirs(resumed_dir, exist_ok=True)
relocated = os.path.join(resumed_dir, "resume.hdf5")
shutil.copy2(os.path.join(out_dir, "checkpoint_2.hdf5"), relocated)
wrapper = os.path.join(resumed_dir, "params.json")
with open(wrapper, "w") as handle:
    json.dump({"input": {"checkpoint": {"path": relocated, "reorder": False}}}, handle)
resumed = subprocess.run(
    [POLYFEM_BIN, "-j", wrapper, "-o", resumed_dir, "--log_level", "error"],
    cwd=resumed_dir, capture_output=True, text=True, timeout=900)
assert resumed.returncode == 0, resumed.stdout + resumed.stderr

with h5py.File(os.path.join(out_dir, "checkpoint_4.hdf5")) as full, \
        h5py.File(os.path.join(resumed_dir, "checkpoint_4.hdf5")) as continued:
    a = full["checkpoint/state/solution"][()]
    b = continued["checkpoint/state/solution"][()]
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    for key in ("time", "dt", "remaining_steps", "output_index"):
        np.testing.assert_array_equal(full["checkpoint/metadata/" + key][()],
                                      continued["checkpoint/metadata/" + key][()])
print("HDA checkpoint parity passed")

if len(sys.argv) > 3:
    with h5py.File(os.path.join(out_dir, "checkpoint_4.hdf5")) as full, \
            h5py.File(os.path.join(sys.argv[3], "checkpoint_4.hdf5")) as ui:
        np.testing.assert_allclose(full["checkpoint/state/solution"][()],
                                   ui["checkpoint/state/solution"][()], rtol=1e-10, atol=1e-10)
    print("Actual HDA UI-command parity passed")
