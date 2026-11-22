"""Native numpy Gmsh MSH parser (v2.2 ASCII + binary, v4.1 ASCII + binary).

No gmsh dependency: pure numpy token/buffer parsing, fast enough for
multi-million element meshes. Used by the MSH_Reader 3.0 HDA and the
headless test suite.

read_msh(path) returns a dict:
    points          (N, 3) float64
    node_tags       (N,)   int64   original gmsh node tags (msh_pt_id)
    cells           {family: {"corners": (M, k) int64 point INDICES,
                              "entity":  (M,)  int32 physical tag (0 if none),
                              "num_nodes": int nodes per element in the file,
                              "gmsh_types": sorted list of Gmsh type codes,
                              "order":   (M,)  int64 position among the file's
                                         elements of the same dimension}}
                    families: tet, hex, tri, quad, line, point, prism, pyramid
    physical_names  {(dim, tag): name}
    duplicate_elements {dim: count} of extra copies left out (MSH 2.2 writes
                    an element once per physical group it is in)
    shared_groups   [(dim, (tag, ...))] physical groups sharing elements; an
                    element keeps its first group (the one PolyFEM assigns)

Physical tags are read without sign (Gmsh writes -t for a group holding an
entity with reversed orientation). Partitioned 4.1 meshes take their groups
from $PartitionedEntities; parametric node blocks are read.

write_moved_nodes(path, positions, out_path) copies a mesh file with only its
node coordinates replaced (Read PVD's deformed mesh export).
"""

import os
import re
import stat
import struct
import tempfile

import numpy as np

# gmsh element type -> (family, nodes_per_elem, n_corners)
GMSH_ELEM_TYPES = {
    1: ("line", 2, 2), 8: ("line", 3, 2), 26: ("line", 4, 2), 27: ("line", 5, 2),
    28: ("line", 6, 2),
    15: ("point", 1, 1),
    2: ("tri", 3, 3), 9: ("tri", 6, 3), 20: ("tri", 9, 3), 21: ("tri", 10, 3),
    22: ("tri", 12, 3), 23: ("tri", 15, 3), 25: ("tri", 21, 3),
    3: ("quad", 4, 4), 10: ("quad", 9, 4), 16: ("quad", 8, 4),
    4: ("tet", 4, 4), 11: ("tet", 10, 4), 29: ("tet", 20, 4), 30: ("tet", 35, 4),
    31: ("tet", 56, 4),
    5: ("hex", 8, 8), 12: ("hex", 27, 8), 17: ("hex", 20, 8),
    6: ("prism", 6, 6), 13: ("prism", 18, 6), 18: ("prism", 15, 6),
    7: ("pyramid", 5, 5), 14: ("pyramid", 14, 5), 19: ("pyramid", 13, 5),
}


class MshParseError(RuntimeError):
    pass


_MESH_SECTIONS = {
    "MeshFormat", "PhysicalNames", "Entities", "PartitionedEntities",
    "Nodes", "Elements",
}
_SECTION_START = re.compile(rb"^\$([A-Za-z][A-Za-z0-9_]*)[ \t]*\r?\n", re.M)
_BINARY_SECTIONS = {"Nodes", "Elements", "Entities", "PartitionedEntities"}
_BINARY_DATA_SECTIONS = {"NodeData", "ElementData", "ElementNodeData"}
_INT64_MAX = np.iinfo(np.int64).max
_INT32_MAX = np.iinfo(np.int32).max


def _find_sections(data):
    """Map section name -> (start, end) byte offsets of the section body."""
    return _scan_sections(data)[0]


def _scan_sections(data, collect_elements=False):
    sections = {}
    version, binary_format, probe_separator = None, None, b""
    element_blocks = []
    view = memoryview(data)
    pos = 0
    while match := _SECTION_START.search(data, pos):
        name = match[1].decode("ascii")
        body_start = match.end()
        if name in _MESH_SECTIONS and name in sections:
            raise MshParseError(f"Duplicate ${name} section")
        marker = rb"\$End" + match[1] + rb"[ \t]*\r?(?:\n|\Z)"
        if name == "MeshFormat":
            header_end = data.find(b"\n", body_start)
            if header_end < 0:
                raise MshParseError("Truncated $MeshFormat header")
            version, encoding, data_size = _parse_format_header(data[body_start:header_end])
            if encoding:
                payload_end = header_end + 5  # newline and the four-byte endian probe
                binary_format = _binary_format(data[header_end + 1:payload_end], data_size, version)
                end = _binary_end(data, marker, payload_end, name)
                probe_separator = end[1] or b""
            else:
                end = re.compile(rb"^" + marker, re.M).search(data, header_end + 1)
        elif binary_format and name in _BINARY_SECTIONS:
            src = _BinarySection(view[body_start:], name, binary_format)
            if name == "Nodes":
                if version == "2.2":
                    _v22_binary_nodes(src, data, body_start)
                else:
                    for _ in _v41_binary_nodes(src):
                        pass
            elif name == "Elements":
                blocks = (_v22_binary_element_blocks(src, data, body_start) if version == "2.2"
                          else _v41_binary_element_blocks(src))
                for block in blocks:
                    if collect_elements:
                        element_blocks.append(block)
            else:
                for _ in _binary_entities(src):
                    pass
            payload_end = body_start + src.pos
            end = _binary_end(data, marker, payload_end, name, probe_separator)
        elif binary_format and name in _BINARY_DATA_SECTIONS:
            # Post-processing data is opaque to this reader. MshIO omits
            # the separator here as well as in the mesh sections.
            end = re.compile(marker).search(data, body_start)
        else:
            end = re.compile(rb"^" + marker, re.M).search(data, body_start)
        if not end:
            raise MshParseError(f"Unterminated section ${name}")
        sections[name] = (body_start, end.start())
        pos = end.end()
    return sections, version, binary_format, element_blocks


def _binary_end(data, marker, payload_end, name, probe_separator=b""):
    end = re.compile(rb"(\r?\n)?" + marker).match(data, payload_end)
    if end:
        separator = end[1] or b""
        # A short payload can swallow part or all of the newline. Resolve
        # this ambiguity using the separator written after the endian probe.
        if (name == "Nodes" and len(separator) < len(probe_separator)
                and data.endswith(probe_separator, 0, end.start() + len(separator))):
            end = None
    if not end:
        raise MshParseError(f"Binary ${name} section is malformed or cut short (invalid end or trailing data)")
    return end


def _tokens(data, span):
    return data[slice(*span)].split()


def _take_tokens(tok, p, count, section):
    if count < 0:
        raise MshParseError(f"Negative count in ${section}")
    if p + count > len(tok):
        raise MshParseError(f"Truncated ASCII ${section} section")
    return tok[p:p + count]


def _int64_tags(values, section):
    """Convert ASCII integer tokens or binary unsigned tags without wrapping."""
    if isinstance(values, np.ndarray) and values.dtype.kind == "u":
        if values.size and int(values.max()) > _INT64_MAX:
            raise MshParseError(f"Integer tag in ${section} is outside signed Int64 range")
    try:
        return np.asarray(values, dtype=np.int64)
    except (ValueError, OverflowError) as exc:
        raise MshParseError(f"Invalid or out-of-range integer tag in ${section}: {exc}") from exc


def _parse_format_header(header):
    version, encoding, data_size = header.decode("ascii").split()
    if version not in ("2.2", "4.1"):
        raise MshParseError(
            f"MSH version {version} is not supported (MSH 2.2 and 4.1 are, as for PolyFEM): "
            "save the mesh again from Gmsh as version 4.1 (Mesh.MshFileVersion = 4.1)")
    encoding = int(encoding)
    if encoding not in (0, 1):
        raise MshParseError(f"Invalid MSH file encoding {encoding}")
    return version, encoding, int(data_size)


def _binary_format(probe, data_size, version):
    if data_size not in ((4, 8) if version == "4.1" else (8,)):
        raise MshParseError(f"Unsupported binary MSH {version} data size {data_size}")
    if probe == b"\x01\x00\x00\x00":
        endian = "<"
    elif probe == b"\x00\x00\x00\x01":
        endian = ">"
    else:
        raise MshParseError("Invalid endianness probe in binary $MeshFormat")
    return endian, endian + f"u{data_size}"


def _read_buffer(buf, dtype, count, offset, section):
    dtype = np.dtype(dtype)
    if count < 0 or offset < 0 or offset + count * dtype.itemsize > len(buf):
        raise MshParseError(f"Binary ${section} section is malformed or cut short")
    return np.frombuffer(buf, dtype, count, offset)


class _BinarySection:
    """Sequential, bounds-checked reads of MSH 4.1 binary fields."""

    def __init__(self, buf, name, binary_format):
        self.buf = buf
        self.name = name
        endian, self.size_dtype = binary_format
        self.int_dtype = endian + "i4"
        self.float_dtype = endian + "f8"
        self.pos = 0

    def read(self, dtype, count, skip=False):
        if not skip:
            values = _read_buffer(self.buf, dtype, count, self.pos, self.name)
        self.pos += count * np.dtype(dtype).itemsize
        if self.pos > len(self.buf):
            raise MshParseError(f"Truncated binary ${self.name} section")
        if not skip:
            return values

    def sizes(self, count):
        return tuple(int(v) for v in self.read(self.size_dtype, count))


def _parse_physical_names(data, sections):
    names = {}
    if "PhysicalNames" not in sections:
        return names
    text = data[slice(*sections["PhysicalNames"])].decode("utf-8", "replace")
    lines = text.strip().split("\n")
    count = int(lines[0])
    _check_total("PhysicalNames", count, len(lines) - 1)
    for line in lines[1:]:
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise MshParseError("Malformed row in $PhysicalNames")
        dim, tag, name = int(parts[0]), int(parts[1]), parts[2].strip().strip('"')
        if (dim, tag) in names:
            raise MshParseError(f"Duplicate entry {dim}/{tag} in $PhysicalNames")
        names[(dim, tag)] = name
    return names


def _physical_list(tags):
    """An entity's physical groups as written, without the sign Gmsh puts on
    a group that holds the entity with reversed orientation
    (Physical Surface(5) = {1, -2} writes -5 for surface 2), duplicates
    removed, file order kept."""
    out = []
    for tag in tags:
        tag = abs(int(tag))
        if tag not in out:
            out.append(tag)
    return out


def _ascii_entities(tok, section):
    p = 0

    def take(count):
        nonlocal p
        values = _take_tokens(tok, p, count, section)
        p += count
        return values

    def count():
        value = int(take(1)[0])
        if value < 0:
            raise MshParseError(f"Negative count in ${section}")
        return value

    partitioned = section == "PartitionedEntities"
    if partitioned:
        count()  # number of partitions
        take(2 * count())  # ghost entity/partition pairs
    counts = [count() for _ in range(4)]
    for dim, num_entities in enumerate(counts):
        for _ in range(num_entities):
            tag = int(take(1)[0])
            parent = None
            if partitioned:
                parent = tuple(map(int, take(2)))
                take(count())  # partitions of this entity
            take(3 if dim == 0 else 6)  # point coordinates or bounding box
            phys = take(count())
            if dim > 0:
                take(count())  # bounding entities
            yield dim, tag, parent, phys
    if p != len(tok):
        raise MshParseError(f"Unexpected trailing tokens in ${section}")


def _binary_entities(src):
    partitioned = src.name == "PartitionedEntities"
    if partitioned:
        _num_partitions, n_ghosts = src.sizes(2)
        src.read(src.int_dtype, 2 * n_ghosts, skip=True)
    for dim, count in enumerate(src.sizes(4)):
        for _ in range(count):
            tag = int(src.read(src.int_dtype, 1)[0])
            parent = None
            if partitioned:
                parent = tuple(map(int, src.read(src.int_dtype, 2)))
                n_parts, = src.sizes(1)
                src.read(src.int_dtype, n_parts, skip=True)
            src.read(src.float_dtype, 3 if dim == 0 else 6, skip=True)
            n_phys, = src.sizes(1)
            phys = src.read(src.int_dtype, n_phys)
            if dim > 0:
                n_brep, = src.sizes(1)
                src.read(src.int_dtype, n_brep, skip=True)
            yield dim, tag, parent, phys


def _entity_groups(data, sections, binary_format):
    model_phys, partition_phys = {}, {}
    for section in ("Entities", "PartitionedEntities"):
        if section not in sections:
            continue
        if binary_format:
            src = _BinarySection(memoryview(data)[slice(*sections[section])], section, binary_format)
            entities = _binary_entities(src)
        else:
            entities = _ascii_entities(_tokens(data, sections[section]), section)
        for dim, tag, parent, tags in entities:
            key = (dim, tag)
            phys = _physical_list(tags)
            if parent is None:
                model_phys[key] = phys
            else:
                # Parent tags refer to model entities even when a partition
                # entity has reused the same (dimension, tag) key.
                if not phys and parent[0] == dim:
                    phys = model_phys.get(parent, [])
                partition_phys[key] = phys
    return {**model_phys, **partition_phys}


def _tag_index_map(node_tags):
    """Return a function mapping arrays of gmsh node tags -> point indices."""
    n = len(node_tags)
    if n and int(node_tags.min()) < 1:
        raise MshParseError(f"Invalid node tag {int(node_tags.min())} in $Nodes; tags must be >= 1")

    def unknown(tags, bad):
        tag = int(np.asarray(tags)[bad][0])
        return MshParseError(
            f"an element references node tag {tag}, which is not a node of "
            f"the file (its {n} nodes have tags "
            f"{int(node_tags.min()) if n else 0} to "
            f"{int(node_tags.max()) if n else 0})")

    if n and node_tags[0] == 1 and node_tags[-1] == n \
            and np.array_equal(node_tags, np.arange(1, n + 1)):
        def contiguous(tags):  # fast path
            bad = (tags < 1) | (tags > n)
            if bad.any():
                raise unknown(tags, bad)
            return tags - 1
        return contiguous
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    repeated = sorted_tags[1:] == sorted_tags[:-1]
    if repeated.any():
        tag = int(sorted_tags[1:][repeated][0])
        raise MshParseError(f"Duplicate node tag {tag} in $Nodes")

    def lookup(tags):
        if not n:
            if len(tags):
                raise unknown(tags, np.ones(len(tags), dtype=bool))
            return np.zeros(0, dtype=np.int64)
        idx = np.searchsorted(sorted_tags, tags)
        bad = (idx >= n) | (sorted_tags[np.minimum(idx, n - 1)] != tags)
        if bad.any():
            raise unknown(tags, bad)
        return order[idx]

    return lookup


FAMILY_DIMENSION = {
    "point": 0, "line": 1, "tri": 2, "quad": 2,
    "tet": 3, "hex": 3, "prism": 3, "pyramid": 3,
}


def _accumulate_cells(cells, etype, node_tags, entity, elementary=None):
    """Store full connectivity by element type, with dimension-relative order."""
    family, _, _ = GMSH_ELEM_TYPES[etype]
    dim = FAMILY_DIMENSION[family]
    counts = cells.setdefault(None, {})
    start = counts.get(dim, 0)
    counts[dim] = start + len(entity)
    slot = cells.setdefault(etype, [])
    slot.append((node_tags, entity, elementary, start))


def _v22_copies(node_tags, entity, elementary):
    """Mask of the extra copies MSH 2.2 writes of an element in more than one
    physical group (Gmsh writes it once per group, with the same nodes and
    elementary entity). The copy in the lowest group is kept, the group
    PolyFEM's reader assigns (MshIO collects an entity's groups sorted), and
    the groups that shared elements are returned as tag tuples."""
    drop = np.zeros(len(entity), dtype=bool)
    if not len(entity) or not (elementary > 0).any():
        return drop, set()
    # (elementary entity, physical group) pairs as one 1D key; an entity
    # listed with more than one group is what makes Gmsh write copies
    group_min = int(entity.min())
    group = entity.astype(np.int64) - group_min
    width = int(group.max()) + 1
    # Keep the fast packed key even for ASCII meshes with wide entity IDs.
    entity_ids = elementary.astype(np.int64, copy=False)
    max_packed_entity = (_INT64_MAX - (width - 1)) // width
    if int(elementary.min()) < -max_packed_entity or int(elementary.max()) > max_packed_entity:
        _, entity_ids = np.unique(elementary, return_inverse=True)
    pairs = np.unique(entity_ids * width + group)
    ids, counts = np.unique(pairs // width, return_counts=True)
    multi = ids[counts > 1]
    if not len(multi):
        return drop, set()
    candidates = np.flatnonzero(np.isin(entity_ids, multi))
    rows = node_tags[candidates]
    # a 64-bit hash of (entity, ordered nodes) brings copies together; equal
    # neighbours are then compared in full, so a hash collision can only
    # leave a copy in, never drop a distinct element
    with np.errstate(over="ignore"):
        digest = elementary[candidates].astype(np.uint64) \
            * np.uint64(0x9E3779B97F4A7C15)
        for column in range(rows.shape[1]):
            digest = (digest ^ rows[:, column].astype(np.uint64)) \
                * np.uint64(0xBF58476D1CE4E5B9)
    # by hash, then group, then file position: the first of each run stays
    order = np.lexsort((candidates, entity[candidates], digest))
    rows = rows[order]
    same = (digest[order][1:] == digest[order][:-1]) \
        & np.all(rows[1:] == rows[:-1], axis=1) \
        & (elementary[candidates][order][1:]
           == elementary[candidates][order][:-1])
    repeat = np.zeros(len(order), dtype=bool)
    repeat[1:] = same
    duplicates = candidates[order[repeat]]
    drop[duplicates] = True
    groups = set()
    pair_entity, pair_group = pairs // width, pairs % width + group_min
    duplicate_entities = np.unique(entity_ids[duplicates])
    starts = np.searchsorted(pair_entity, duplicate_entities)
    ends = np.searchsorted(pair_entity, duplicate_entities, side="right")
    for start, end in zip(starts, ends):
        groups.add(tuple(int(t) for t in pair_group[start:end]))
    return drop, groups


def _finalize_cells(cells, lookup, notes):
    parts = {}
    for etype, blocks in cells.items():
        if etype is None:
            continue
        family, n_nodes, n_corners = GMSH_ELEM_TYPES[etype]
        dim = FAMILY_DIMENSION[family]
        tags, entities, elementary, starts = zip(*blocks)
        node_tags = np.concatenate(tags).astype(np.int64, copy=False)
        entity = np.concatenate(entities)
        order = np.concatenate([np.arange(start, start + len(block), dtype=np.int64)
                                for start, block in zip(starts, tags)])
        # Validate every reference, even on elements that will be deduplicated
        # and on midside nodes that Houdini does not use for linear primitives.
        indices = lookup(node_tags.ravel()).reshape(node_tags.shape)
        corners = indices if n_nodes == n_corners else indices[:, :n_corners].copy()
        if any(e is not None for e in elementary):
            drop, groups = _v22_copies(node_tags, entity, np.concatenate([
                np.zeros(len(block), dtype=np.int64) if e is None else e
                for block, e in zip(tags, elementary)]))
            if drop.any():
                notes["duplicates"][dim] = notes["duplicates"].get(dim, 0) + int(drop.sum())
                notes["shared_groups"].update((dim, g) for g in groups)
                keep = ~drop
                corners, entity, order = corners[keep], entity[keep], order[keep]
        parts.setdefault(family, []).append((corners, entity, order, n_nodes, etype))

    out = {}
    for family, blocks in parts.items():
        corners, entities, orders, widths, gmsh_types = zip(*blocks)
        if len(blocks) == 1:
            corners, entity, order = corners[0], entities[0], orders[0]
        else:
            # Multiple element types may map to the same linear family.
            order = np.concatenate(orders)
            permutation = np.argsort(order, kind="stable")
            corners = np.concatenate(corners)[permutation]
            entity = np.concatenate(entities)[permutation]
            order = order[permutation]
        out[family] = {"corners": corners, "entity": entity,
                       "num_nodes": max(widths), "gmsh_types": sorted(gmsh_types),
                       "order": order}
    # With copies left out, renumber positions densely per dimension.
    for dim in notes["duplicates"]:
        dimension_cells = [cell for family, cell in out.items()
                           if FAMILY_DIMENSION[family] == dim]
        ranks = np.sort(np.concatenate([cell["order"] for cell in dimension_cells]))
        for cell in dimension_cells:
            cell["order"] = np.searchsorted(ranks, cell["order"])
    return out


# ----------------------------------------------------------------------------
# v4.1
# ----------------------------------------------------------------------------

def _block_physical(ent_phys, dim, etag, notes):
    """First physical group of an element block's entity (0 if none), the
    group PolyFEM's reader assigns; an entity in several groups is noted."""
    phys = ent_phys.get((dim, etag), ())
    if len(phys) > 1:
        notes["shared_groups"].add((dim, tuple(phys)))
    tag = phys[0] if phys else 0
    if tag > _INT32_MAX:
        raise MshParseError(f"Physical tag {tag} is outside signed Int32 range")
    return tag


def _v41_node_header(dim, parametric):
    if not 0 <= dim <= 3 or parametric not in (0, 1):
        raise MshParseError(f"Invalid $Nodes block dimension/parametric flag: {dim}/{parametric}")
    return 3 + (dim if parametric else 0)


def _check_node_block_count(total, row, count):
    if count < 0 or row + count > total:
        raise MshParseError(f"Invalid $Nodes block count {count} for {total} declared nodes")


def _check_total(section, declared, parsed):
    if parsed != declared:
        raise MshParseError(
            f"${section} blocks hold {parsed}, but the header declared {declared} "
            "(the section is malformed or cut short)")


def _v41_ascii_node_blocks(tok):
    header = _take_tokens(tok, 0, 4, "Nodes")
    num_blocks, total = map(int, header[:2])
    if total > 0:
        _int64_tags(header[2:], "Nodes")
    if num_blocks < 0 or total < 0:
        raise MshParseError("Negative count in $Nodes header")
    p = 4
    row = 0
    for _ in range(num_blocks):
        header = _take_tokens(tok, p, 4, "Nodes")
        dim, _, parametric, count = map(int, header)
        _check_node_block_count(total, row, count)
        stride = _v41_node_header(dim, parametric)
        p += 4
        tags = _take_tokens(tok, p, count, "Nodes")
        p += count
        coords = _take_tokens(tok, p, stride * count, "Nodes")
        p += stride * count
        yield header, tags, coords, count, stride
        row += count
    _check_total("Nodes", total, row)
    if p != len(tok):
        raise MshParseError("Unexpected trailing tokens in $Nodes")


def _v41_binary_nodes(src, expected_count=None):
    num_blocks, total, min_tag, max_tag = src.sizes(4)
    if total and max(min_tag, max_tag) > _INT64_MAX:
        raise MshParseError("Integer tag in $Nodes header is outside signed Int64 range")
    block_header_size = 3 * np.dtype(src.int_dtype).itemsize + np.dtype(src.size_dtype).itemsize
    if num_blocks * block_header_size > len(src.buf) - src.pos:
        raise MshParseError("Binary $Nodes section is malformed or cut short (invalid block count)")
    if expected_count is not None:
        _check_positions_count(total, expected_count)
    row = 0
    for _ in range(num_blocks):
        dim, _, parametric = (int(v) for v in src.read(src.int_dtype, 3))
        count, = src.sizes(1)
        _check_node_block_count(total, row, count)
        stride = _v41_node_header(dim, parametric)
        tags = src.read(src.size_dtype, count)
        coords = src.read(src.float_dtype, stride * count).reshape(count, stride)
        yield tags, coords
        row += count
    _check_total("Nodes", total, row)


def _read_v41_ascii(data, sections, ent_phys, notes):
    tok = _tokens(data, sections["Nodes"])
    node_blocks = _v41_ascii_node_blocks(tok)
    node_tags, points = [], []
    for _header, tags, coords, count, stride in node_blocks:
        node_tags.append(_int64_tags(tags, "Nodes"))
        points.append(_v41_ascii_points(coords, count, stride))

    tok = _tokens(data, sections["Elements"])
    header = _take_tokens(tok, 0, 4, "Elements")
    num_blocks, total = map(int, header[:2])
    if total > 0:
        _int64_tags(header[2:], "Elements")
    if num_blocks < 0 or total < 0:
        raise MshParseError("Negative count in $Elements header")
    p = 4
    cells = {}
    parsed = 0
    for _ in range(num_blocks):
        dim, etag, etype, count = map(int, _take_tokens(tok, p, 4, "Elements"))
        p += 4
        stride = _element_stride(etype, dim)
        if not count:
            continue
        fields = _take_tokens(tok, p, count * stride, "Elements")
        block = _int64_tags(fields, "Elements").reshape(count, stride)
        p += count * stride
        _accumulate_cells(cells, etype, block[:, 1:],
                          np.full(count, _block_physical(ent_phys, dim, etag, notes),
                                  dtype=np.int32))
        parsed += count
    _check_total("Elements", total, parsed)
    if p != len(tok):
        raise MshParseError("Unexpected trailing tokens in $Elements")
    return _join_nodes(node_tags, points, cells)


def _read_v41_binary(data, sections, ent_phys, notes, binary_format, element_blocks):
    src = _BinarySection(memoryview(data)[slice(*sections["Nodes"])], "Nodes", binary_format)
    node_tags, points = [], []
    for tags, coords in _v41_binary_nodes(src):
        node_tags.append(_int64_tags(tags, "Nodes"))
        points.append(coords[:, :3])

    cells = {}
    for dim, etag, etype, fields in element_blocks:
        block = _int64_tags(fields, "Elements")
        _accumulate_cells(cells, etype, block[:, 1:],
                          np.full(len(block), _block_physical(ent_phys, dim, etag, notes),
                                  dtype=np.int32))
    return _join_nodes(node_tags, points, cells)


def _v41_binary_element_blocks(src):
    num_blocks, total, min_tag, max_tag = src.sizes(4)
    if total and max(min_tag, max_tag) > _INT64_MAX:
        raise MshParseError("Integer tag in $Elements header is outside signed Int64 range")
    parsed = 0
    for _ in range(num_blocks):
        dim, etag, etype = (int(v) for v in src.read(src.int_dtype, 3))
        count, = src.sizes(1)
        stride = _element_stride(etype, dim)
        if not count:
            continue
        yield dim, etag, etype, src.read(src.size_dtype, count * stride).reshape(count, stride)
        parsed += count
    _check_total("Elements", total, parsed)


def _element_stride(etype, dim=None):
    if etype not in GMSH_ELEM_TYPES:
        raise MshParseError(f"Unsupported gmsh element type {etype}")
    family, num_nodes, _ = GMSH_ELEM_TYPES[etype]
    if dim is not None and dim != FAMILY_DIMENSION[family]:
        raise MshParseError(
            f"$Elements block dimension {dim} disagrees with gmsh element type "
            f"{etype} ({family}, dimension {FAMILY_DIMENSION[family]})")
    return 1 + num_nodes


def _v41_ascii_points(coords, count, stride):
    return np.asarray(coords, dtype=np.float64).reshape(count, stride)[:, :3]


def _check_coordinates(points):
    if not np.isfinite(points).all():
        raise MshParseError("Non-finite coordinates in $Nodes")


def _join_nodes(tags, points, cells):
    if not tags:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64), cells
    return np.concatenate(tags), np.concatenate(points), cells


def _v22_ascii_nodes(tok):
    count = int(_take_tokens(tok, 0, 1, "Nodes")[0])
    fields = _take_tokens(tok, 1, 4 * count, "Nodes")
    if 1 + len(fields) != len(tok):
        raise MshParseError("Unexpected trailing tokens in $Nodes")
    return count, fields


def _matching_run(view, columns, expected):
    """Only inspect the matching prefix, with bounded work past a mismatch."""
    end = 1
    # A scalar check avoids a NumPy allocation for each alternating record.
    if len(view) > 1 and all(view[1, col] == value for col, value in zip(columns, expected)):
        end = 2
        while end < len(view):
            chunk = view[end:2 * end]
            matches = np.ones(len(chunk), dtype=bool)
            for col, value in zip(columns, expected):
                matches &= chunk[:, col] == value
            if not matches.all():
                return end + int(matches.argmin())
            end += len(chunk)
    return end


def _v22_elements(cells, block, etype, ntags, start_tags):
    if ntags:
        signed = block[:, start_tags]
        if (int(signed.min()) < -_INT32_MAX or int(signed.max()) > _INT32_MAX):
            raise MshParseError("Physical tag in $Elements is outside signed Int32 range")
        entity = np.abs(signed).astype(np.int32)
    else:
        entity = np.zeros(len(block), dtype=np.int32)
    elementary = block[:, start_tags + 1] if ntags >= 2 else None
    _accumulate_cells(cells, etype, block[:, start_tags + ntags:], entity, elementary)


def _read_v22_ascii(data, sections):
    num_nodes, fields = _v22_ascii_nodes(_tokens(data, sections["Nodes"]))
    node_tags = _int64_tags(fields[::4], "Nodes")
    points = _v22_ascii_points(num_nodes, fields)

    tok = _int64_tags(_tokens(data, sections["Elements"]), "Elements")
    num_elems = int(tok[0])
    cells = {}
    p = 1
    parsed = 0
    while p < len(tok):
        if p + 3 > len(tok):
            raise MshParseError("Truncated ASCII $Elements section")
        etype, ntags = int(tok[p + 1]), int(tok[p + 2])
        if ntags < 0:
            raise MshParseError(f"Negative element tag count {ntags}")
        stride = 2 + ntags + _element_stride(etype)
        max_run = (len(tok) - p) // stride
        if max_run == 0:
            raise MshParseError("Truncated ASCII $Elements section")
        view = tok[p:p + max_run * stride].reshape(max_run, stride)
        run = _matching_run(view, (1, 2), (etype, ntags))
        _v22_elements(cells, view[:run], etype, ntags, start_tags=3)
        p += run * stride
        parsed += run
    _check_total("Elements", num_elems, parsed)
    return node_tags, points, cells


def _v22_ascii_points(count, fields):
    points = np.empty((count, 3), dtype=np.float64)
    for col in range(3):
        points[:, col] = fields[col + 1::4]
    return points


def _v22_binary_count(src, data, start):
    nl = data.find(b"\n", start, start + len(src.buf))
    if nl < 0:
        raise MshParseError(f"Truncated binary ${src.name} header")
    src.pos = nl + 1 - start
    return int(data[start:nl])


def _v22_binary_nodes(src, data, start):
    count = _v22_binary_count(src, data, start)
    record = np.dtype([("tag", src.int_dtype), ("xyz", src.float_dtype, (3,))])
    return src.read(record, count)


def _v22_binary_element_blocks(src, data, start):
    num_elems = _v22_binary_count(src, data, start)
    ints = _read_buffer(src.buf, src.int_dtype, (len(src.buf) - src.pos) // 4,
                        src.pos, "Elements")
    p = 0
    parsed = 0
    while parsed < num_elems:
        if p + 3 > len(ints):
            raise MshParseError("Truncated binary $Elements section")
        etype, n_follow, ntags = map(int, ints[p:p + 3])
        if n_follow <= 0:
            raise MshParseError(f"Invalid binary element block count {n_follow}")
        if ntags < 0:
            raise MshParseError(f"Negative element tag count {ntags}")
        stride = ntags + _element_stride(etype)
        if n_follow == 1:
            # Gmsh may write one header per element.
            stride += 3
            max_run = min((len(ints) - p) // stride, num_elems - parsed)
            if max_run == 0:
                raise MshParseError("Truncated binary $Elements section")
            view = ints[p:p + max_run * stride].reshape(max_run, stride)
            run = _matching_run(view, (0, 1, 2), (etype, 1, ntags))
            block = view[:run, 3:]
            p += run * stride
            parsed += run
        else:
            end = p + 3 + n_follow * stride
            if end > len(ints) or parsed + n_follow > num_elems:
                raise MshParseError("Invalid binary $Elements block size")
            block = ints[p + 3:end].reshape(n_follow, stride)
            p = end
            parsed += n_follow
        yield etype, block, ntags
    _check_total("Elements", num_elems, parsed)
    src.pos += 4 * p


def _read_v22_binary(data, sections, binary_format, element_blocks):
    start, end = sections["Nodes"]
    src = _BinarySection(memoryview(data)[start:end], "Nodes", binary_format)
    nodes = _v22_binary_nodes(src, data, start)
    cells = {}
    for etype, block, ntags in element_blocks:
        _v22_elements(cells, block, etype, ntags, start_tags=1)
    return nodes["tag"].astype(np.int64), nodes["xyz"].astype(np.float64), cells


# ----------------------------------------------------------------------------

def read_msh(path):
    """Parse a Gmsh .msh file (see the module docstring). Anything wrong
    with its content -- including a file cut short -- raises MshParseError
    naming the problem; only a missing or unreadable file raises OSError."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        return _read_msh_data(data)
    except MshParseError:
        raise
    except (ValueError, IndexError, TypeError, OverflowError) as exc:
        raise MshParseError(
            f"the file is malformed or cut short ({exc})") from exc


def _read_msh_data(data):
    sections, version, binary_format, element_blocks = _scan_sections(data, collect_elements=True)
    if version is None:
        raise MshParseError("Not a Gmsh MSH file (missing $MeshFormat)")
    if "Nodes" not in sections or "Elements" not in sections:
        raise MshParseError("MSH file has no $Nodes/$Elements sections")

    physical_names = _parse_physical_names(data, sections)
    notes = {"duplicates": {}, "shared_groups": set()}

    if version == "4.1":
        ent_phys = _entity_groups(data, sections, binary_format)
        if binary_format:
            node_tags, points, cells = _read_v41_binary(
                data, sections, ent_phys, notes, binary_format, element_blocks)
        else:
            node_tags, points, cells = _read_v41_ascii(data, sections, ent_phys, notes)
    elif binary_format:
        node_tags, points, cells = _read_v22_binary(data, sections, binary_format, element_blocks)
    else:
        node_tags, points, cells = _read_v22_ascii(data, sections)

    _check_coordinates(points)
    lookup = _tag_index_map(node_tags)
    finalized = _finalize_cells(cells, lookup, notes)
    return {
        "version": version,
        "points": points,
        "node_tags": node_tags,
        "cells": finalized,
        "physical_names": physical_names,
        # elements written once per physical group (MSH 2.2) and left out
        # here: {dim: count}
        "duplicate_elements": notes["duplicates"],
        # sorted [(dim, (tag, ...))] of physical groups sharing elements;
        # each element keeps the first (2.2: lowest) of its groups
        "shared_groups": sorted(notes["shared_groups"]),
    }


# ----------------------------------------------------------------------------
# A copy of a mesh with its nodes moved (Read PVD's deformed mesh export)
# ----------------------------------------------------------------------------

def _coordinate_text(xyz):
    # repr of a Python float is the shortest text that reads back exactly
    return " ".join(repr(float(value)) for value in xyz).encode("ascii")


def _check_positions_count(total, count):
    if total != count:
        raise MshParseError(f"$Nodes lists {total} nodes, {count} positions given")


def _moved_nodes_v22_ascii(body, positions):
    tok = body.split()
    count, fields = _v22_ascii_nodes(tok)
    _check_positions_count(count, len(positions))
    _tag_index_map(_int64_tags(fields[::4], "Nodes"))
    _check_coordinates(_v22_ascii_points(count, fields))
    lines = [tok[0]]
    for tag, xyz in zip(fields[::4], positions):
        lines.append(tag + b" " + _coordinate_text(xyz))
    return b"\n".join(lines) + b"\n"


def _moved_nodes_v41_ascii(body, positions):
    tok = body.split()
    hdr = _take_tokens(tok, 0, 4, "Nodes")
    _check_positions_count(int(hdr[1]), len(positions))
    lines = [b" ".join(hdr)]
    node_tags = []
    row = 0
    for header, tags, coords, count, stride in _v41_ascii_node_blocks(tok):
        node_tags.append(_int64_tags(tags, "Nodes"))
        _check_coordinates(_v41_ascii_points(coords, count, stride))
        lines.append(b" ".join(header))
        lines.extend(tags)
        for k, xyz in enumerate(positions[row:row + count]):
            tail = coords[k * stride + 3:(k + 1) * stride]
            lines.append(b" ".join([_coordinate_text(xyz), *tail]))
        row += count
    _validate_node_blocks(node_tags)
    return b"\n".join(lines) + b"\n"


def _validate_node_blocks(node_tags):
    if node_tags:
        _tag_index_map(np.concatenate(node_tags))


def _moved_nodes_v22_binary(data, span, positions, binary_format):
    out = bytearray(data)
    start, end = span
    src = _BinarySection(memoryview(out)[start:end], "Nodes", binary_format)
    nodes = _v22_binary_nodes(src, data, start)
    _check_positions_count(len(nodes), len(positions))
    _tag_index_map(nodes["tag"])
    _check_coordinates(nodes["xyz"])
    nodes["xyz"] = positions
    return out


def _moved_nodes_v41_binary(data, span, positions, binary_format):
    out = bytearray(data)
    src = _BinarySection(memoryview(out)[slice(*span)], "Nodes", binary_format)
    node_tags = []
    row = 0
    for tags, coords in _v41_binary_nodes(src, expected_count=len(positions)):
        node_tags.append(_int64_tags(tags, "Nodes"))
        count = len(coords)
        _check_coordinates(coords[:, :3])
        coords[:, :3] = positions[row:row + count]
        row += count
    _validate_node_blocks(node_tags)
    return out


def _move_node_data(data, positions):
    sections, version, binary_format, _ = _scan_sections(data)
    if version is None or "Nodes" not in sections:
        raise MshParseError("Not a Gmsh MSH file with nodes")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MshParseError("node positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise MshParseError("node positions must be finite numbers")
    span = sections["Nodes"]
    if binary_format:
        move = _moved_nodes_v41_binary if version == "4.1" else _moved_nodes_v22_binary
        return move(data, span, positions, binary_format)
    move = _moved_nodes_v41_ascii if version == "4.1" else _moved_nodes_v22_ascii
    body = move(data[slice(*span)], positions)
    return data[:span[0]] + body + data[span[1]:]


def _destination_mode(out_path, directory):
    try:
        return stat.S_IMODE(os.stat(out_path).st_mode)
    except FileNotFoundError:
        # Ask the OS for the default creation mode without changing the
        # process-wide umask while other threads may be creating files.
        with tempfile.TemporaryDirectory(dir=directory) as probe_dir:
            fd = os.open(os.path.join(probe_dir, "mode"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            try:
                return stat.S_IMODE(os.fstat(fd).st_mode)
            finally:
                os.close(fd)


def write_moved_nodes(path, positions, out_path):
    """Copy the mesh file `path` to `out_path` with every node moved to
    `positions` ((N, 3), in the order of the file's $Nodes section, which is
    read_msh's order). Everything else -- node tags, elements, physical
    groups and names, entities, the version and encoding -- is copied as it
    is; an ASCII $Nodes section is rewritten in the standard layout."""
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        result = _move_node_data(data, positions)
    except (ValueError, IndexError, TypeError, OverflowError, struct.error) as exc:
        raise MshParseError(f"the file is malformed or cut short ({exc})") from exc

    destination = os.path.abspath(os.fsdecode(out_path))
    directory = os.path.dirname(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=f".{os.path.basename(destination)}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(result)
        os.chmod(temporary, _destination_mode(destination, directory))
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return out_path


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    mesh = read_msh(sys.argv[1])
    dt = time.time() - t0
    print(f"parsed in {dt:.3f}s: {len(mesh['points'])} nodes")
    for family, cell in mesh["cells"].items():
        ents = np.unique(cell["entity"])
        print(f"  {family}: {len(cell['entity'])} elements "
              f"(file order {cell['num_nodes']} nodes/elem), entities {ents}")
    if mesh["physical_names"]:
        print("  physical names:", mesh["physical_names"])
