# PolyFEM 2.0 OnCreated: ensure the internal merge/output skeleton exists.
# (1.2 also pip-installed gmsh here; 2.0 has no external dependencies.)
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
