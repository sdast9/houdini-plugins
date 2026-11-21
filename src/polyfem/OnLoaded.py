# PolyFEM 2.0 OnLoaded: a scene reopened while one of its runs is still going
# (or after it ended) picks the run up again, so Run Status, Stop and Open Log
# keep working across a Houdini restart.
import hou

node = kwargs["node"]
try:
    node.hdaModule().migrate_legacy_expr_vectors(node)
except Exception as exc:  # loading the scene must never fail on this
    print(f"[PolyFEM HDA] {node.path()}: could not migrate expression vectors: {exc}")
# Never destroy/recreate child SOPs synchronously inside OnLoaded: Houdini's
# OP_Network::loadIntoNetwork still holds child pointers while syncing node
# versions and crashes (SIGBUS) if children are deleted mid-load. Defer display
# network upgrades until after the hip file finishes loading in interactive UI.
if hou.isUIAvailable():
    try:
        import hdefereval
        _node_path = node.path()

        def _deferred_display_upgrade(path=_node_path):
            target = hou.node(path)
            if target is not None:
                try:
                    target.hdaModule().check_display_version(target)
                except Exception as exc:
                    print(f"[PolyFEM HDA] {path}: could not refresh display network: {exc}")

        hdefereval.executeDeferred(_deferred_display_upgrade)
    except Exception:
        pass
try:
    node.hdaModule().reattach_run(node)
except Exception as exc:  # loading the scene must never fail on this
    print(f"[PolyFEM HDA] {node.path()}: could not follow its last run: {exc}")
