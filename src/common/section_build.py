"""Pack Python sources into data sections sharing one HDA module namespace."""

from dataclasses import dataclass
import hashlib
import re


_EXECUTE_IN_PARTS = '''\
    # A part's standalone docstring did not become the HDA module docstring
    # when that part was embedded in the middle of the legacy PythonModule.
    doc_missing = object()
    original_doc = ns.get("__doc__", doc_missing)
    try:
        for section, relpath, digest in parts:
            src = part_sections[section].contents()
            filename = f"<{asset_tag}:{relpath}@{digest}>"
            linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
            exec(compile(src, filename, "exec", dont_inherit=True), ns)
    finally:
        if original_doc is doc_missing:
            ns.pop("__doc__", None)
        else:
            ns["__doc__"] = original_doc
'''

# Phase 3 parity checks showed that compiling with COMPILE_LINE_GAPS preserves
# CPython 3.13 bytecode and class __firstlineno__ values. Relocating the nested
# code objects then points function tracebacks at the individual part files.
_EXECUTE_AS_MODULE = '''\
    from bisect import bisect_right
    from types import CodeType

    chunks, starts, filenames = [], [], []
    line = 1
    for (section, relpath, digest), gap in zip(parts, line_gaps, strict=True):
        src = part_sections[section].contents()
        filename = f"<{asset_tag}:{relpath}@{digest}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        line += gap
        starts.append(line)
        filenames.append(filename)
        chunk = src if src.endswith("\\n") else src + "\\n"
        chunks.extend(("\\n" * gap, chunk))
        line += chunk.count("\\n")

    # CPython optimizes calls using module-wide import bindings and can store
    # absolute source lines in class constants. Compile with the legacy layout
    # first, then make each nested code object point to its own source section.
    source = "".join(chunks)
    linecache.cache[module_filename] = (
        len(source), None, source.splitlines(True), module_filename)
    module_code = compile(source, module_filename, "exec", dont_inherit=True)

    def relocate(code, filename, offset):
        return code.replace(
            co_filename=filename,
            co_firstlineno=code.co_firstlineno - offset,
            co_consts=tuple(relocate(value, filename, offset)
                            if isinstance(value, CodeType) else value
                            for value in code.co_consts),
        )

    constants = []
    for value in module_code.co_consts:
        if isinstance(value, CodeType):
            index = bisect_right(starts, value.co_firstlineno) - 1
            value = relocate(value, filenames[index], starts[index] - 1)
        constants.append(value)
    exec(module_code.replace(co_consts=tuple(constants)), ns)
'''


@dataclass(frozen=True)
class PartMetadata:
    part_key: str
    relpath: tuple[str, ...]
    section_name: str
    digest: str


def section_name_for(index, part_key):
    if not isinstance(part_key, str) or not re.fullmatch(r"[a-z0-9_]+", part_key):
        raise ValueError(f"Invalid runtime part key {part_key!r}; use [a-z0-9_]+")
    return f"py_{index:02d}_{part_key}.py"


def manifest_entries(parts):
    """Validate the keys and return entries with their generated section names."""
    entries = []
    seen = set()
    for index, (key, relpath) in enumerate(parts):
        section_name = section_name_for(index, key)
        if key in seen:
            raise ValueError(f"Duplicate runtime part key {key!r}")
        if (not isinstance(relpath, tuple) or not relpath
                or not all(isinstance(seg, str) and seg for seg in relpath)):
            raise TypeError(
                f"Runtime part {key!r} path must be a non-empty tuple of path "
                f"segments, got {relpath!r}"
            )
        seen.add(key)
        entries.append((key, relpath, section_name))
    return entries


def build_runtime_sections(asset_tag, type_name, parts, *, compile_line_gaps=None):
    """Pack runtime sections, optionally compiling with legacy gaps before parts."""
    # Use the same module object as the builders, including patched readers.
    import hda_build

    sections = {}
    metadata = []
    for key, relpath, name in manifest_entries(parts):
        source = hda_build.read_source(*relpath)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        sections[name] = source
        metadata.append(PartMetadata(key, relpath, name, digest))

    filenames = "\n".join(
        f"        ({part.section_name!r}, {'/'.join(part.relpath)!r}, {part.digest!r}),"
        for part in metadata
    )
    execute = _EXECUTE_IN_PARTS
    if compile_line_gaps is not None:
        gaps = tuple(compile_line_gaps.get(part.part_key, 0) for part in metadata)
        identity = (gaps, tuple(part.digest for part in metadata))
        digest = hashlib.sha256(repr(identity).encode("ascii")).hexdigest()[:12]
        module_filename = f"<{asset_tag}:PythonModule@{digest}>"
        execute = (f"    line_gaps = {gaps!r}\n"
                   f"    module_filename = {module_filename!r}\n"
                   + _EXECUTE_AS_MODULE)
    sections["PythonModule"] = f'''# Generated from the asset's runtime manifest.
def _bootstrap_hda_parts(ns, kwargs):
    import linecache

    asset_tag = {asset_tag!r}
    if "type" in kwargs:
        definition = kwargs["type"].definition()
    else:
        import hou

        matches = [
            node_type.definition()
            for category in hou.nodeTypeCategories().values()
            if (node_type := hou.nodeType(category, {type_name!r})) is not None
            and node_type.definition() is not None
        ]
        if len(matches) != 1:
            raise RuntimeError("Expected one active HDA definition for " + {type_name!r})
        definition, = matches

    part_sections = definition.sections()
    parts = (
{filenames}
    )
    missing = [section for section, _, _ in parts if section not in part_sections]
    if missing:
        raise RuntimeError(
            f"{{asset_tag}} ({type_name}) is missing embedded runtime section(s): "
            + ", ".join(missing)
            + ". Rebuild the HDA with hython src/build_all.py."
        )
{execute}

try:
    _bootstrap_hda_parts(globals(), globals().get("kwargs", {{}}))
finally:
    del _bootstrap_hda_parts
'''
    return sections, metadata
