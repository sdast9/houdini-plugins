# PolyFEM 2.0 OnUpdated: when an already-open scene picks up an updated HDA
# definition (e.g. Assets > Refresh Asset Libraries), migrate any hidden legacy
# expression vectors into the visible 3-box [x, y, z] parameters.
node = kwargs["node"]
try:
    node.hdaModule().migrate_legacy_expr_vectors(node)
except Exception as exc:
    print(f"[PolyFEM HDA] {node.path()}: could not migrate expression vectors: {exc}")
