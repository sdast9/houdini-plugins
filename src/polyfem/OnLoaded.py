# PolyFEM 2.0 OnLoaded: a scene reopened while one of its runs is still going
# (or after it ended) picks the run up again, so Run Status, Stop and Open Log
# keep working across a Houdini restart.
node = kwargs["node"]
try:
    node.hdaModule().migrate_legacy_expr_vectors(node)
except Exception as exc:  # loading the scene must never fail on this
    print(f"[PolyFEM HDA] {node.path()}: could not migrate expression vectors: {exc}")
try:
    node.hdaModule().check_display_version(node)
except Exception as exc:
    print(f"[PolyFEM HDA] {node.path()}: could not refresh display network: {exc}")
try:
    node.hdaModule().reattach_run(node)
except Exception as exc:  # loading the scene must never fail on this
    print(f"[PolyFEM HDA] {node.path()}: could not follow its last run: {exc}")
