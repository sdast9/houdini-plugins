# PolyFEM 2.0 OnCreated: ensure the internal merge/output skeleton exists.
# (1.2 also pip-installed gmsh here; 2.0 has no external dependencies.)
import os

node = kwargs["node"]
node.allowEditingOfContents()

merge_node = node.node("all")
if merge_node is None:
    merge_node = node.createNode("merge", "all")
output_node = node.node("output")
if output_node is None:
    output_node = node.createNode("output", "output")
if output_node.input(0) is None:
    output_node.setInput(0, merge_node)
node.layoutChildren()
# New nodes write the nodal forces Read PVD's force curves need; the
# parameter's default stays off so scenes saved before keep their output.
forces = node.parm("forces_fields")
if forces is not None:
    forces.set(1)
# Smart defaults for new nodes (saved scenes keep their own values):
# pre-fill PolyFEM Binary from $POLYFEM_BIN when set, and default Working
# Directory to a portable per-node folder under $HIP.
env_bin = os.environ.get("POLYFEM_BIN", "").strip()
bin_parm = node.parm("polyfem_bin")
if env_bin and bin_parm is not None and not bin_parm.unexpandedString().strip():
    bin_parm.set(env_bin)
wd_parm = node.parm("working_dir")
if wd_parm is not None and not wd_parm.unexpandedString().strip():
    wd_parm.set("$HIP/polyfem/$OS/")
try:
    node.setUserData(
        "polyfem_display_version",
        str(node.hdaModule().DISPLAY_CHAIN_VERSION))
except Exception:
    pass
