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
_INT64_MAX = np.iinfo(np.int64).max
_INT32_MAX = np.iinfo(np.int32).max


def _find_sections(data):
    """Map section name -> (start, end) byte offsets of the section body."""
    sections = {}
    pos = 0
    while match := _SECTION_START.search(data, pos):
        name = match[1].decode("ascii")
        body_start = match.end()
        end_marker = re.compile(rb"^\$End" + match[1] + rb"[ \t]*\r?(?:\n|\Z)", re.M)
        end = end_marker.search(data, body_start)
        if not end:
            raise MshParseError(f"Unterminated section ${name}")
        if name in _MESH_SECTIONS and name in sections:
            raise MshParseError(f"Duplicate ${name} section")
        sections[name] = (body_start, end.start())
        pos = end.end()
    return sections


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


def _binary_format(fmt_body, data_size, version):
    if data_size not in ((4, 8) if version >= 4.0 else (8,)):
        raise MshParseError(f"Unsupported binary MSH {version:g} data size {data_size}")
    nl = fmt_body.find(b"\n")
    probe = fmt_body[nl + 1:nl + 5]
    if nl >= 0 and probe == b"\x01\x00\x00\x00":
        endian = "<"
    elif nl >= 0 and probe == b"\x00\x00\x00\x01":
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
    lines = text.strip().splitlines()
    for line in lines[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3:
            dim, tag, name = int(parts[0]), int(parts[1]), parts[2].strip().strip('"')
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


def _parse_entities_ascii(tok):
    """(dim, entity_tag) -> list of physical tags, from a tokenized
    $Entities."""
    ent_phys = {}
    p = 0
    counts = [int(v) for v in tok[p:p + 4]]
    p += 4
    # points: tag x y z numPhys [phys...]
    for _ in range(counts[0]):
        tag = int(tok[p])
        n_phys = int(tok[p + 4])
        if n_phys > 0:
            ent_phys[(0, tag)] = _physical_list(tok[p + 5:p + 5 + n_phys])
        p += 5 + n_phys
    # curves/surfaces/volumes: tag 6*bbox numPhys [phys...] numBrep [tags...]
    for dim, count in ((1, counts[1]), (2, counts[2]), (3, counts[3])):
        for _ in range(count):
            tag = int(tok[p])
            n_phys = int(tok[p + 7])
            if n_phys > 0:
                ent_phys[(dim, tag)] = _physical_list(
                    tok[p + 8:p + 8 + n_phys])
            p += 8 + n_phys
            n_brep = int(tok[p])
            p += 1 + n_brep
    return ent_phys


def _parse_entities_binary(buf, binary_format):
    ent_phys = {}
    src = _BinarySection(buf, "Entities", binary_format)
    for dim, count in enumerate(src.sizes(4)):
        for _ in range(count):
            tag = int(src.read(src.int_dtype, 1)[0])
            src.read(src.float_dtype, 3 if dim == 0 else 6, skip=True)
            n_phys, = src.sizes(1)
            if n_phys:
                ent_phys[(dim, tag)] = _physical_list(src.read(src.int_dtype, n_phys))
            if dim > 0:
                n_brep, = src.sizes(1)
                src.read(src.int_dtype, n_brep, skip=True)
    return ent_phys


def _parse_partitioned_ascii(tok, ent_phys):
    """Add a partitioned mesh's entities ($PartitionedEntities): its elements
    lie on these, each carrying its own physical groups (or, if it lists
    none, those of the model entity it was cut from)."""
    p = 1                                   # numPartitions
    n_ghosts = int(tok[p]); p += 1 + 2 * n_ghosts
    counts = [int(v) for v in tok[p:p + 4]]; p += 4
    for dim in range(4):
        for _ in range(counts[dim]):
            tag, parent_dim, parent_tag = (int(v) for v in tok[p:p + 3])
            n_parts = int(tok[p + 3]); p += 4 + n_parts
            p += 3 if dim == 0 else 6       # point xyz / bounding box
            n_phys = int(tok[p])
            phys = _physical_list(tok[p + 1:p + 1 + n_phys]); p += 1 + n_phys
            if dim > 0:
                p += 1 + int(tok[p])        # bounding entities
            if not phys and parent_dim == dim:
                phys = ent_phys.get((parent_dim, parent_tag), [])
            ent_phys[(dim, tag)] = phys


def _parse_partitioned_binary(buf, ent_phys, binary_format):
    src = _BinarySection(buf, "PartitionedEntities", binary_format)
    _num_partitions, n_ghosts = src.sizes(2)
    src.read(src.int_dtype, 2 * n_ghosts, skip=True)
    for dim, count in enumerate(src.sizes(4)):
        for _ in range(count):
            tag, parent_dim, parent_tag = (int(v) for v in src.read(src.int_dtype, 3))
            n_parts, = src.sizes(1)
            src.read(src.int_dtype, n_parts, skip=True)
            src.read(src.float_dtype, 3 if dim == 0 else 6, skip=True)
            n_phys, = src.sizes(1)
            phys = _physical_list(src.read(src.int_dtype, n_phys))
            if dim > 0:
                n_brep, = src.sizes(1)
                src.read(src.int_dtype, n_brep, skip=True)
            if not phys and parent_dim == dim:
                phys = ent_phys.get((parent_dim, parent_tag), [])
            ent_phys[(dim, tag)] = phys


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
    entity_ids = elementary
    max_packed_entity = (_INT64_MAX - (width - 1)) // width
    if int(elementary.min()) < -max_packed_entity or int(elementary.max()) > max_packed_entity:
        _, entity_ids = np.unique(elementary, return_inverse=True)
    pairs = np.unique(entity_ids * width + group)
    ids, counts = np.unique(pairs // width, return_counts=True)
    multi = ids[counts > 1]
    if not len(multi):
        return drop, set()
    candidates = np.flatnonzero(np.isin(entity_ids, multi))
    rows = np.sort(node_tags[candidates], axis=1)
    # a 64-bit hash of (entity, sorted nodes) brings copies together; equal
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
    drop[candidates[order[repeat]]] = True
    groups = set()
    pair_entity, pair_group = pairs // width, pairs % width + group_min
    # Sorted pairs are already grouped by entity. Avoid scanning the entire
    # array for each entity that has more than one physical group.
    starts = np.searchsorted(pair_entity, multi)
    for start, count in zip(starts, counts[counts > 1]):
        groups.add(tuple(int(t) for t in pair_group[start:start + count]))
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
        parts.setdefault(family, []).append((corners, entity, order, n_nodes))

    out = {}
    for family, blocks in parts.items():
        corners, entities, orders, widths = zip(*blocks)
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
                       "num_nodes": max(widths), "order": order}
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


def _v41_binary_node_blocks(src, num_blocks, total):
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
        points.append(np.asarray(coords, dtype=np.float64).reshape(count, stride)[:, :3])

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
        if not count:
            continue
        stride = _element_stride(etype)
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


def _read_v41_binary(data, sections, ent_phys, notes, binary_format):
    src = _BinarySection(memoryview(data)[slice(*sections["Nodes"])], "Nodes", binary_format)
    num_blocks, total, min_tag, max_tag = src.sizes(4)
    if total and max(min_tag, max_tag) > _INT64_MAX:
        raise MshParseError("Integer tag in $Nodes header is outside signed Int64 range")
    node_tags, points = [], []
    for tags, coords in _v41_binary_node_blocks(src, num_blocks, total):
        node_tags.append(_int64_tags(tags, "Nodes"))
        points.append(coords[:, :3])

    src = _BinarySection(memoryview(data)[slice(*sections["Elements"])], "Elements", binary_format)
    num_blocks, total, min_tag, max_tag = src.sizes(4)
    if total and max(min_tag, max_tag) > _INT64_MAX:
        raise MshParseError("Integer tag in $Elements header is outside signed Int64 range")
    cells = {}
    parsed = 0
    for _ in range(num_blocks):
        dim, etag, etype = (int(v) for v in src.read(src.int_dtype, 3))
        count, = src.sizes(1)
        if not count:
            continue
        stride = _element_stride(etype)
        fields = src.read(src.size_dtype, count * stride)
        block = _int64_tags(fields, "Elements").reshape(count, stride)
        _accumulate_cells(cells, etype, block[:, 1:],
                          np.full(count, _block_physical(ent_phys, dim, etag, notes),
                                  dtype=np.int32))
        parsed += count
    _check_total("Elements", total, parsed)
    if bytes(src.buf[src.pos:]) not in (b"", b"\n", b"\r\n"):
        raise MshParseError("Unexpected trailing data in binary $Elements")
    return _join_nodes(node_tags, points, cells)


def _element_stride(etype):
    if etype not in GMSH_ELEM_TYPES:
        raise MshParseError(f"Unsupported gmsh element type {etype}")
    return 1 + GMSH_ELEM_TYPES[etype][1]


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
    points = np.empty((num_nodes, 3), dtype=np.float64)
    for col in range(3):
        points[:, col] = fields[col + 1::4]

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


def _v22_binary_nodes(buf, endian):
    nl = buf.find(b"\n")
    if nl < 0:
        raise MshParseError("Truncated binary $Nodes header")
    count = int(buf[:nl])
    record = np.dtype([("tag", endian + "i4"), ("xyz", endian + "f8", (3,))])
    return _read_buffer(buf, record, count, nl + 1, "Nodes"), nl + 1


def _read_v22_binary(data, sections, binary_format):
    """Binary MSH v2.2 (fTetWild / PyMesh MshSaver default output)."""
    endian, _ = binary_format
    nodes, _ = _v22_binary_nodes(data[slice(*sections["Nodes"])], endian)
    node_tags = nodes["tag"].astype(np.int64)
    points = nodes["xyz"].astype(np.float64)

    buf = data[slice(*sections["Elements"])]
    nl = buf.find(b"\n")
    if nl < 0:
        raise MshParseError("Truncated binary $Elements header")
    num_elems = int(buf[:nl])
    ints = _read_buffer(buf, endian + "i4", (len(buf) - nl - 1) // 4, nl + 1, "Elements")
    p = 0
    cells = {}
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
        _v22_elements(cells, block, etype, ntags, start_tags=1)
    _check_total("Elements", num_elems, parsed)
    if buf[nl + 1 + 4 * p:] not in (b"", b"\n", b"\r\n"):
        raise MshParseError("Unexpected trailing data in binary $Elements")
    return node_tags, points, cells


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
    sections = _find_sections(data)
    if "MeshFormat" not in sections:
        raise MshParseError("Not a Gmsh MSH file (missing $MeshFormat)")
    fmt_body = data[slice(*sections["MeshFormat"])]
    fmt = fmt_body.split(b"\n", 1)[0].split()
    version = float(fmt[0])
    encoding, data_size = int(fmt[1]), int(fmt[2])
    if encoding not in (0, 1):
        raise MshParseError(f"Invalid MSH file encoding {encoding}")
    is_binary = encoding == 1
    if 4.0 <= version < 4.1 or version >= 5.0 or 2.3 <= version < 4.0 \
            or version < 2.0:
        raise MshParseError(
            f"MSH version {fmt[0].decode('ascii', 'replace')} is not "
            "supported (MSH 2.2 and 4.1 are, as for PolyFEM): save the mesh "
            "again from Gmsh as version 4.1 (Mesh.MshFileVersion = 4.1)")

    if "Nodes" not in sections or "Elements" not in sections:
        raise MshParseError("MSH file has no $Nodes/$Elements sections")

    binary_format = _binary_format(fmt_body, data_size, version) if is_binary else None
    physical_names = _parse_physical_names(data, sections)
    notes = {"duplicates": {}, "shared_groups": set()}

    if version >= 4.0:
        ent_phys = {}
        if "Entities" in sections:
            if is_binary:
                ent_phys = _parse_entities_binary(
                    memoryview(data)[slice(*sections["Entities"])], binary_format)
            else:
                ent_phys = _parse_entities_ascii(_tokens(data, sections["Entities"]))
        if "PartitionedEntities" in sections:
            if is_binary:
                _parse_partitioned_binary(
                    memoryview(data)[slice(*sections["PartitionedEntities"])], ent_phys, binary_format)
            else:
                _parse_partitioned_ascii(
                    _tokens(data, sections["PartitionedEntities"]), ent_phys)
        if is_binary:
            node_tags, points, cells = _read_v41_binary(
                data, sections, ent_phys, notes, binary_format)
        else:
            node_tags, points, cells = _read_v41_ascii(
                data, sections, ent_phys, notes)
    else:
        if is_binary:
            node_tags, points, cells = _read_v22_binary(data, sections, binary_format)
        else:
            node_tags, points, cells = _read_v22_ascii(data, sections)

    lookup = _tag_index_map(node_tags)
    finalized = _finalize_cells(cells, lookup, notes)
    return {
        "version": fmt[0].decode("ascii", "replace"),
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


def _check_positions_count(total, positions):
    if total != len(positions):
        raise MshParseError(f"$Nodes lists {total} nodes, {len(positions)} positions given")


def _moved_nodes_v22_ascii(body, positions):
    tok = body.split()
    count, fields = _v22_ascii_nodes(tok)
    _check_positions_count(count, positions)
    lines = [tok[0]]
    for tag, xyz in zip(fields[::4], positions):
        lines.append(tag + b" " + _coordinate_text(xyz))
    return b"\n".join(lines) + b"\n"


def _moved_nodes_v41_ascii(body, positions):
    tok = body.split()
    hdr = _take_tokens(tok, 0, 4, "Nodes")
    _check_positions_count(int(hdr[1]), positions)
    lines = [b" ".join(hdr)]
    row = 0
    for header, tags, coords, count, stride in _v41_ascii_node_blocks(tok):
        lines.append(b" ".join(header))
        lines.extend(tags)
        for k, xyz in enumerate(positions[row:row + count]):
            tail = coords[k * stride + 3:(k + 1) * stride]
            lines.append(b" ".join([_coordinate_text(xyz), *tail]))
        row += count
    return b"\n".join(lines) + b"\n"


def _moved_nodes_v22_binary(data, span, positions, binary_format):
    endian, _ = binary_format
    nodes, offset = _v22_binary_nodes(data[slice(*span)], endian)
    _check_positions_count(len(nodes), positions)
    out = bytearray(data)
    out_nodes = _read_buffer(out, nodes.dtype, len(nodes), span[0] + offset, "Nodes")
    out_nodes["xyz"] = positions
    return out


def _moved_nodes_v41_binary(data, span, positions, binary_format):
    out = bytearray(data)
    src = _BinarySection(memoryview(out)[slice(*span)], "Nodes", binary_format)
    blocks, total, _, _ = src.sizes(4)
    _check_positions_count(total, positions)
    row = 0
    for _, coords in _v41_binary_node_blocks(src, blocks, total):
        count = len(coords)
        coords[:, :3] = positions[row:row + count]
        row += count
    return out


def write_moved_nodes(path, positions, out_path):
    """Copy the mesh file `path` to `out_path` with every node moved to
    `positions` ((N, 3), in the order of the file's $Nodes section, which is
    read_msh's order). Everything else -- node tags, elements, physical
    groups and names, entities, the version and encoding -- is copied as it
    is; an ASCII $Nodes section is rewritten in the standard layout."""
    with open(path, "rb") as handle:
        data = handle.read()
    sections = _find_sections(data)
    if "MeshFormat" not in sections or "Nodes" not in sections:
        raise MshParseError("Not a Gmsh MSH file with nodes")
    fmt_body = data[slice(*sections["MeshFormat"])]
    fmt = fmt_body.split(b"\n", 1)[0].split()
    version, binary = float(fmt[0]), int(fmt[1]) == 1
    if not (2.0 <= version < 3.0 or abs(version - 4.1) < 1e-6):
        raise MshParseError(
            f"Unsupported MSH version {version:g}: only 2.2 and 4.1 are "
            "supported.")
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MshParseError("node positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise MshParseError("node positions must be finite numbers")
    span = sections["Nodes"]
    if binary:
        binary_format = _binary_format(fmt_body, int(fmt[2]), version)
        move = _moved_nodes_v41_binary if version >= 4.0 else _moved_nodes_v22_binary
        result = move(data, span, positions, binary_format)
    else:
        body = data[span[0]:span[1]]
        body = (_moved_nodes_v41_ascii(body, positions) if version >= 4.0
                else _moved_nodes_v22_ascii(body, positions))
        result = data[:span[0]] + body + data[span[1]:]
    temporary = os.fspath(out_path) + ".tmp"
    try:
        with open(temporary, "wb") as handle:
            handle.write(result)
        os.replace(temporary, out_path)
    except Exception:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass
        raise
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
