"""Record or compare the interface and network of the three built HDAs.

Run with hython and point --dir at the exact library files to inspect.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import types


ASSETS = (
    ("sop_MSH_Reader.3.0.hdanc", "Sop", "MSH_Reader::3.0", "cook"),
    ("object_stevenabramowitch.dev.PolyFEM.2.0.hdanc", "Object",
     "stevenabramowitch::dev::PolyFEM::2.0", "check_setup_button"),
    ("object_readPVD.1.0.hdanc", "Object", "readPVD::1.0", "refresh"),
)
# Sections that legitimately vary between legacy single-file and modular builds
# or contain Houdini build timestamps / archive headers.
_VOLATILE_SECTIONS = {
    "PythonModule", "Contents.gz", "CreateScript", "ExtraFileOptions",
    "IndexFile", "Sections.List", "InternalFileOptions", "TypePropertiesOptions",
}
_PART_SECTION = re.compile(r"py_\d+_[a-z0-9_]+\.py")


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256(text):
    if isinstance(text, bytes):
        return _sha256_bytes(text)
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_const(value):
    if isinstance(value, types.CodeType):
        return _canonical_code(value)
    if isinstance(value, tuple):
        return ["tuple", [_canonical_const(v) for v in value]]
    if isinstance(value, frozenset):
        items = [_canonical_const(v) for v in value]
        return ["frozenset", sorted(items, key=lambda x: json.dumps(x, sort_keys=True))]
    if isinstance(value, bytes):
        return ["bytes", _sha256_bytes(value)]
    if isinstance(value, (type(None), bool, int, float, str)):
        return [type(value).__name__, repr(value)]
    return [type(value).__name__, repr(value)]


def _canonical_code(code):
    """Hashable representation of a code object independent of filename/lineno."""
    return {
        "name": code.co_name,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "code": _sha256_bytes(code.co_code),
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "consts": [_canonical_const(c) for c in code.co_consts],
    }


def _callable_code_hashes(runtime_globals):
    """Fingerprint every Python function defined in the HDA module."""
    hashes = {}
    for name in sorted(runtime_globals):
        value = runtime_globals[name]
        if isinstance(value, types.FunctionType):
            payload = {
                "code": _canonical_code(value.__code__),
                "defaults": _canonical_const(value.__defaults__),
                "kwdefaults": (
                    {k: _canonical_const(v) for k, v in sorted(value.__kwdefaults__.items())}
                    if value.__kwdefaults__ else None
                ),
            }
            hashes[name] = _sha256(json.dumps(payload, sort_keys=True))
    return hashes


def _verify_library(node_type, path):
    definition = node_type.definition() if node_type is not None else None
    actual = definition.libraryFilePath() if definition is not None else None
    if actual is None or os.path.realpath(actual) != os.path.realpath(path):
        raise AssertionError(f"Expected active definition from {path}, found {actual!r}")
    return definition


def _connections(node, normalize):
    records = []
    for connection in sorted(node.inputConnections(), key=lambda wire: wire.inputIndex()):
        item = connection.inputItem()
        indirect = connection.subnetIndirectInput()
        source = {
            "item_path": normalize(item.path()) if item is not None and indirect is None else None,
            "indirect_input": indirect.number() if indirect is not None else None,
        }
        records.append({
            "input": connection.inputIndex(),
            "source": source,
            "output": connection.inputItemOutputIndex(),
        })
    return records


def _parameters(node, normalize):
    records = {}
    for parm in sorted(node.parms(), key=lambda parm: parm.name()):
        name = parm.name()
        animated = bool(parm.keyframes())
        if name in {"snippet", "python"}:
            records[name] = {"source": normalize(parm.rawValue())}
            if not animated:
                continue
        elif parm.isAtDefault(compare_temporary_defaults=False, compare_expressions=True) and not animated:
            continue

        # HOM serializes values, expression language, and all keyframes.
        code = parm.asCode(save_flag_values=False, save_aliases=False, function_name="set_value")
        records.setdefault(name, {})["as_code"] = normalize(code)
    return records


def _flag(node, method):
    getter = getattr(node, method, None)
    return bool(getter()) if getter is not None else None


def _internal_nodes(instance, temporary_parent):
    root = instance.path()
    temporary_paths = [(root, "$ASSET")]
    if temporary_parent is not None:
        temporary_paths.append((temporary_parent.path(), "$ASSET_PARENT"))
    substitutions = [
        (re.compile(re.escape(path) + r"(?![\w.:-])"), token)
        for path, token in temporary_paths
    ]

    def normalize(text):
        for pattern, token in substitutions:
            text = pattern.sub(token, text)
        return text

    nodes = instance.allSubChildren(recurse_in_locked_nodes=True, sync_delayed_definition=True)
    records = {}
    for node in sorted(nodes, key=lambda node: node.path()):
        relative_path = node.path().removeprefix(root + "/")
        records[relative_path] = {
            "type_name": node.type().name(),
            "input_connections": _connections(node, normalize),
            "display_flag": _flag(node, "isDisplayFlagSet"),
            "render_flag": _flag(node, "isRenderFlagSet"),
            "non_default_parms_and_expressions": _parameters(node, normalize),
        }
    return records


def _public_callables(module, names):
    signatures = {}
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        if callable(value):
            try:
                signatures[name] = str(inspect.signature(value))
            except (ValueError, TypeError):
                signatures[name] = None
    return signatures


def _fingerprint_asset(hou, path, category, type_name, runtime_entry):
    if not path.is_file():
        raise FileNotFoundError(path)
    hou.hda.installFile(str(path), force_use_assets=True)
    node_type = hou.nodeType(hou.nodeTypeCategories()[category], type_name)
    definition = _verify_library(node_type, path)

    temporary_parent = instance = None
    try:
        parent = hou.node("/obj")
        if category == "Sop":
            temporary_parent = parent.createNode("geo", "fingerprint_geo", run_init_scripts=False)
            parent = temporary_parent
        # Inspect the saved network. Event code is fingerprinted separately.
        instance = parent.createNode(type_name, "fingerprint_asset", run_init_scripts=False,
                                     exact_type_name=True)
        _verify_library(instance.type(), path)
        module = instance.hdaModule()
        names = sorted(dir(module))
        # HOM may expose the functions through an HDAModule proxy whose dir()
        # only lists proxy attributes. Check the underlying globals as well.
        runtime_globals = getattr(module, runtime_entry).__globals__
        runtime_names = sorted(runtime_globals)
        extra_opts = {
            key: value
            for key, value in sorted(definition.extraFileOptions().items())
            if not _PART_SECTION.match(key.split("/", 1)[0])
        }
        record = {
            "as_dialog_script_sha256": _sha256(definition.parmTemplateGroup().asDialogScript()),
            "sections": {
                name: _sha256(section.binaryContents())
                for name, section in sorted(definition.sections().items())
                if name not in _VOLATILE_SECTIONS and not _PART_SECTION.fullmatch(name)
            },
            "extra_file_options": extra_opts,
            "internal_nodes": _internal_nodes(instance, temporary_parent),
            "module_dir": names,
            "public_callables": _public_callables(
                module, sorted(set(names) | set(runtime_names))),
            "function_code_hashes": _callable_code_hashes(runtime_globals),
        }
        if runtime_entry not in names:
            record["runtime_dir"] = runtime_names
        return record
    finally:
        if temporary_parent is not None:
            temporary_parent.destroy()
        elif instance is not None:
            instance.destroy()


def fingerprint_directory(directory):
    import hou

    directory = Path(directory).resolve()
    return {
        filename: _fingerprint_asset(hou, directory / filename, category, type_name, entry)
        for filename, category, type_name, entry in ASSETS
    }


def differences(expected, actual, path=()):
    """Yield field paths and values for all mismatches."""
    label = " / ".join(path) or "<root>"
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child = (*path, key)
            child_label = " / ".join(child)
            if key not in expected:
                yield f"{child_label}: added {actual[key]!r}"
            elif key not in actual:
                yield f"{child_label}: removed {expected[key]!r}"
            else:
                yield from differences(expected[key], actual[key], child)
    elif isinstance(expected, list) and isinstance(actual, list):
        if path and path[-1] in {"module_dir", "runtime_dir"} and set(expected) != set(actual):
            for name in sorted(set(expected) - set(actual)):
                yield f"{label}: removed {name!r}"
            for name in sorted(set(actual) - set(expected)):
                yield f"{label}: added {name!r}"
        else:
            for index, (before, after) in enumerate(zip(expected, actual)):
                yield from differences(before, after, (*path, str(index)))
            for index in range(len(actual), len(expected)):
                yield f"{label} / {index}: removed {expected[index]!r}"
            for index in range(len(expected), len(actual)):
                yield f"{label} / {index}: added {actual[index]!r}"
    elif type(expected) is not type(actual) or expected != actual:
        yield f"{label}: expected {expected!r}, got {actual!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=Path, metavar="OUT_JSON")
    mode.add_argument("--compare", type=Path, metavar="BASELINE_JSON")
    args = parser.parse_args()
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    actual = fingerprint_directory(args.dir)
    if args.record:
        args.record.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Recorded {len(actual)} asset fingerprints to {args.record}")
        return 0

    mismatches = list(differences(baseline, actual))
    if mismatches:
        for mismatch in mismatches:
            print(mismatch, file=sys.stderr)
        return 1
    print(f"All {len(actual)} asset fingerprints match {args.compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
