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

import io
import os
import struct

import numpy as np

# gmsh element type -> (family, nodes_per_elem, n_corners)
GMSH_ELEM_TYPES = {
    1: ("line", 2, 2), 8: ("line", 3, 2), 26: ("line", 4, 2), 27: ("line", 5, 2),
    15: ("point", 1, 1),
    2: ("tri", 3, 3), 9: ("tri", 6, 3), 20: ("tri", 9, 3), 21: ("tri", 10, 3),
    22: ("tri", 12, 3), 23: ("tri", 15, 3),
    3: ("quad", 4, 4), 10: ("quad", 9, 4), 16: ("quad", 8, 4),
    4: ("tet", 4, 4), 11: ("tet", 10, 4), 29: ("tet", 20, 4), 30: ("tet", 35, 4),
    31: ("tet", 56, 4),
    5: ("hex", 8, 8), 12: ("hex", 27, 8), 17: ("hex", 20, 8),
    6: ("prism", 6, 6), 13: ("prism", 18, 6), 18: ("prism", 15, 6),
    7: ("pyramid", 5, 5), 14: ("pyramid", 14, 5), 19: ("pyramid", 13, 5),
}


class MshParseError(RuntimeError):
    pass


def _find_sections(data):
    """Map section name -> (start, end) byte offsets of the section body."""
    sections = {}
    pos = 0
    while True:
        start = data.find(b"$", pos)
        if start < 0:
            break
        eol = data.find(b"\n", start)
        if eol < 0:
            break
        name = data[start + 1:eol].strip().decode("ascii", "replace")
        if name.startswith("End"):
            pos = eol + 1
            continue
        end_marker = b"$End" + name.encode("ascii")
        end = data.find(end_marker, eol)
        if end < 0:
            raise MshParseError(f"Unterminated section ${name}")
        sections[name] = (eol + 1, end)
        pos = end + len(end_marker)
    return sections


def _tokens(data, span):
    """Whole-section ASCII tokenization to a float64 array (C-speed split)."""
    text = data[span[0]:span[1]].decode("ascii")
    return np.array(text.split(), dtype=np.float64)


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
    counts = tok[p:p + 4].astype(np.int64)
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


def _parse_entities_binary(buf):
    ent_phys = {}
    p = 0
    counts = np.frombuffer(buf, dtype="<u8", count=4, offset=p)
    p += 32
    for _ in range(int(counts[0])):  # points
        tag = int(np.frombuffer(buf, "<i4", 1, p)[0])
        n_phys = int(np.frombuffer(buf, "<u8", 1, p + 4 + 24)[0])
        if n_phys > 0:
            ent_phys[(0, tag)] = _physical_list(
                np.frombuffer(buf, "<i4", n_phys, p + 4 + 24 + 8))
        p += 4 + 24 + 8 + 4 * n_phys
    for dim, count in ((1, counts[1]), (2, counts[2]), (3, counts[3])):
        for _ in range(int(count)):
            tag = int(np.frombuffer(buf, "<i4", 1, p)[0])
            n_phys = int(np.frombuffer(buf, "<u8", 1, p + 4 + 48)[0])
            if n_phys > 0:
                ent_phys[(dim, tag)] = _physical_list(
                    np.frombuffer(buf, "<i4", n_phys, p + 4 + 48 + 8))
            p += 4 + 48 + 8 + 4 * n_phys
            n_brep = int(np.frombuffer(buf, "<u8", 1, p)[0])
            p += 8 + 4 * n_brep
    return ent_phys


def _parse_partitioned_ascii(tok, ent_phys):
    """Add a partitioned mesh's entities ($PartitionedEntities): its elements
    lie on these, each carrying its own physical groups (or, if it lists
    none, those of the model entity it was cut from)."""
    p = 1                                   # numPartitions
    n_ghosts = int(tok[p]); p += 1 + 2 * n_ghosts
    counts = tok[p:p + 4].astype(np.int64); p += 4
    for dim in range(4):
        for _ in range(counts[dim]):
            tag, parent_dim, parent_tag = (int(v) for v in tok[p:p + 3])
            n_parts = int(tok[p + 3]); p += 4 + n_parts
            p += 3 if dim == 0 else 6       # point xyz / bounding box
            n_phys = int(tok[p])
            phys = _physical_list(tok[p + 1:p + 1 + n_phys]); p += 1 + n_phys
            if dim > 0:
                p += 1 + int(tok[p])        # bounding entities
            ent_phys[(dim, tag)] = phys or ent_phys.get(
                (parent_dim, parent_tag), [])


def _parse_partitioned_binary(buf, ent_phys):
    p = 8                                   # numPartitions
    n_ghosts = int(np.frombuffer(buf, "<u8", 1, p)[0]); p += 8 + 8 * n_ghosts
    counts = np.frombuffer(buf, "<u8", 4, p); p += 32
    for dim in range(4):
        for _ in range(int(counts[dim])):
            tag, parent_dim, parent_tag = (
                int(v) for v in np.frombuffer(buf, "<i4", 3, p))
            n_parts = int(np.frombuffer(buf, "<u8", 1, p + 12)[0])
            p += 20 + 4 * n_parts
            p += 24 if dim == 0 else 48
            n_phys = int(np.frombuffer(buf, "<u8", 1, p)[0])
            phys = _physical_list(np.frombuffer(buf, "<i4", n_phys, p + 8))
            p += 8 + 4 * n_phys
            if dim > 0:
                p += 8 + 4 * int(np.frombuffer(buf, "<u8", 1, p)[0])
            ent_phys[(dim, tag)] = phys or ent_phys.get(
                (parent_dim, parent_tag), [])


def _tag_index_map(node_tags):
    """Return a function mapping arrays of gmsh node tags -> point indices."""
    n = len(node_tags)

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


def _accumulate_cells(cells, family, num_nodes, corners, entity,
                      elementary=None):
    """elementary: the v2.2 elementary entity tag of each element (None in
    v4.1, which writes every element once)."""
    slot = cells.setdefault(
        family, {"corners": [], "entity": [], "elementary": [],
                 "num_nodes": num_nodes})
    slot["corners"].append(corners)
    slot["entity"].append(entity)
    slot["elementary"].append(
        np.zeros(len(entity), dtype=np.int64) if elementary is None
        else np.asarray(elementary, dtype=np.int64))
    slot["num_nodes"] = max(slot["num_nodes"], num_nodes)
    # Blocks in file order (key None), so every element's position among
    # the elements of its dimension can be recovered: PolyFEM reads a mesh's
    # elements in that order, whatever their family.
    cells.setdefault(None, []).append((family, len(entity)))


def _v22_copies(corner_tags, entity, elementary):
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
    group = entity.astype(np.int64) - int(entity.min())
    width = int(group.max()) + 1
    pairs = np.unique(elementary * width + group)
    ids, counts = np.unique(pairs // width, return_counts=True)
    multi = ids[counts > 1]
    if not len(multi):
        return drop, set()
    candidates = np.flatnonzero(np.isin(elementary, multi))
    rows = np.sort(corner_tags[candidates], axis=1)
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
    pair_entity, pair_group = pairs // width, pairs % width + int(entity.min())
    for tag in multi:
        groups.add(tuple(int(t) for t in pair_group[pair_entity == tag]))
    return drop, groups


def _finalize_cells(cells, lookup, notes):
    out = {}
    seen = {}
    orders = {}
    for family, count in cells.get(None, ()):
        dim = FAMILY_DIMENSION[family]
        start = seen.get(dim, 0)
        orders.setdefault(family, []).append(
            np.arange(start, start + count, dtype=np.int64))
        seen[dim] = start + count
    kept_orders = {}
    for family, slot in cells.items():
        if family is None:
            continue
        corners_tags = np.vstack(slot["corners"]).astype(np.int64)
        entity = np.concatenate(slot["entity"]).astype(np.int32)
        order = np.concatenate(orders[family])
        drop, groups = _v22_copies(
            corners_tags, entity, np.concatenate(slot["elementary"]))
        if drop.any():
            dim = FAMILY_DIMENSION[family]
            notes["duplicates"][dim] = notes["duplicates"].get(dim, 0) \
                + int(drop.sum())
            notes["shared_groups"].update((dim, g) for g in groups)
            keep = ~drop
            corners_tags, entity, order = (
                corners_tags[keep], entity[keep], order[keep])
        corners = lookup(corners_tags.ravel()).reshape(corners_tags.shape)
        out[family] = {
            "corners": corners,
            "entity": entity,
            "num_nodes": slot["num_nodes"],
            # position among the file's elements of the same dimension
            "order": order,
        }
        kept_orders.setdefault(FAMILY_DIMENSION[family], []).append(order)
    # with copies left out, renumber positions densely per dimension
    if notes["duplicates"]:
        for dim, parts in kept_orders.items():
            ranks = np.sort(np.concatenate(parts))
            for family, cell in out.items():
                if FAMILY_DIMENSION[family] == dim:
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
    return phys[0] if phys else 0


def _read_v41_ascii(data, sections, ent_phys, notes):
    tok_nodes = _tokens(data, sections["Nodes"])
    p = 0
    num_blocks = int(tok_nodes[p]); num_nodes = int(tok_nodes[p + 1]); p += 4
    node_tags = np.empty(num_nodes, dtype=np.int64)
    points = np.empty((num_nodes, 3), dtype=np.float64)
    filled = 0
    for _ in range(num_blocks):
        dim = int(tok_nodes[p]); parametric = int(tok_nodes[p + 2])
        n_in_block = int(tok_nodes[p + 3]); p += 4
        if filled + n_in_block > num_nodes:
            raise MshParseError(f"$Nodes declares {num_nodes} nodes, but its "
                                "blocks hold more")
        node_tags[filled:filled + n_in_block] = tok_nodes[p:p + n_in_block].astype(np.int64)
        p += n_in_block
        # parametric blocks follow x y z with dim parametric coordinates
        stride = 3 + (dim if parametric else 0)
        coords = tok_nodes[p:p + stride * n_in_block].reshape(
            n_in_block, stride)[:, :3]
        points[filled:filled + n_in_block] = coords
        p += stride * n_in_block
        filled += n_in_block
    if filled != num_nodes:
        raise MshParseError(f"$Nodes declares {num_nodes} nodes, but its "
                            f"blocks hold {filled}")

    tok_elems = _tokens(data, sections["Elements"])
    p = 0
    num_blocks = int(tok_elems[p]); p += 4
    cells = {}
    for _ in range(num_blocks):
        dim = int(tok_elems[p]); etag = int(tok_elems[p + 1])
        etype = int(tok_elems[p + 2]); n_in_block = int(tok_elems[p + 3]); p += 4
        if etype not in GMSH_ELEM_TYPES:
            raise MshParseError(f"Unsupported gmsh element type {etype}")
        family, n_nodes, n_corners = GMSH_ELEM_TYPES[etype]
        block = tok_elems[p:p + n_in_block * (1 + n_nodes)].reshape(
            n_in_block, 1 + n_nodes).astype(np.int64)
        p += n_in_block * (1 + n_nodes)
        phys = _block_physical(ent_phys, dim, etag, notes)
        _accumulate_cells(
            cells, family, n_nodes, block[:, 1:1 + n_corners],
            np.full(n_in_block, phys, dtype=np.int32))
    return node_tags, points, cells


def _read_v41_binary(data, sections, ent_phys, notes):
    buf = data[slice(*sections["Nodes"])]
    p = 0
    hdr = np.frombuffer(buf, "<u8", 4, p); p += 32
    num_blocks, num_nodes = int(hdr[0]), int(hdr[1])
    node_tags = np.empty(num_nodes, dtype=np.int64)
    points = np.empty((num_nodes, 3), dtype=np.float64)
    filled = 0
    for _ in range(num_blocks):
        dim, _tag, parametric = (int(v) for v in np.frombuffer(buf, "<i4", 3, p))
        n_in_block = int(np.frombuffer(buf, "<u8", 1, p + 12)[0]); p += 20
        if filled + n_in_block > num_nodes:
            raise MshParseError(f"$Nodes declares {num_nodes} nodes, but its "
                                "blocks hold more")
        node_tags[filled:filled + n_in_block] = np.frombuffer(buf, "<u8", n_in_block, p)
        p += 8 * n_in_block
        stride = 3 + (dim if parametric else 0)
        points[filled:filled + n_in_block] = np.frombuffer(
            buf, "<f8", stride * n_in_block, p).reshape(
                n_in_block, stride)[:, :3]
        p += 8 * stride * n_in_block
        filled += n_in_block
    if filled != num_nodes:
        raise MshParseError(f"$Nodes declares {num_nodes} nodes, but its "
                            f"blocks hold {filled}")

    buf = data[slice(*sections["Elements"])]
    p = 0
    hdr = np.frombuffer(buf, "<u8", 4, p); p += 32
    num_blocks = int(hdr[0])
    cells = {}
    for _ in range(num_blocks):
        dim = int(np.frombuffer(buf, "<i4", 1, p)[0])
        etag = int(np.frombuffer(buf, "<i4", 1, p + 4)[0])
        etype = int(np.frombuffer(buf, "<i4", 1, p + 8)[0])
        n_in_block = int(np.frombuffer(buf, "<u8", 1, p + 12)[0])
        p += 20
        if etype not in GMSH_ELEM_TYPES:
            raise MshParseError(f"Unsupported gmsh element type {etype}")
        family, n_nodes, n_corners = GMSH_ELEM_TYPES[etype]
        block = np.frombuffer(buf, "<u8", n_in_block * (1 + n_nodes), p).reshape(
            n_in_block, 1 + n_nodes).astype(np.int64)
        p += 8 * n_in_block * (1 + n_nodes)
        phys = _block_physical(ent_phys, dim, etag, notes)
        _accumulate_cells(
            cells, family, n_nodes, block[:, 1:1 + n_corners],
            np.full(n_in_block, phys, dtype=np.int32))
    return node_tags, points, cells


# ----------------------------------------------------------------------------
# v2.2 (ASCII)
# ----------------------------------------------------------------------------

def _read_v22_ascii(data, sections):
    tok = _tokens(data, sections["Nodes"])
    num_nodes = int(tok[0])
    block = tok[1:1 + 4 * num_nodes].reshape(num_nodes, 4)
    node_tags = block[:, 0].astype(np.int64)
    points = block[:, 1:4].copy()

    tok = _tokens(data, sections["Elements"]).astype(np.int64)
    num_elems = int(tok[0])
    cells = {}
    p = 1
    # Group consecutive runs of identical (type, ntags) so the hot path is
    # vectorized; gmsh writes elements grouped by type, so runs are long.
    while p < len(tok):
        etype = int(tok[p + 1])
        ntags = int(tok[p + 2])
        if etype not in GMSH_ELEM_TYPES:
            raise MshParseError(f"Unsupported gmsh element type {etype}")
        family, n_nodes, n_corners = GMSH_ELEM_TYPES[etype]
        stride = 3 + ntags + n_nodes
        # extend the run while the type/ntags columns keep matching
        max_run = (len(tok) - p) // stride
        view = tok[p:p + max_run * stride].reshape(max_run, stride)
        same = (view[:, 1] == etype) & (view[:, 2] == ntags)
        run = int(np.argmin(same)) if not same.all() else max_run
        if run == 0:
            run = 1
        block = view[:run]
        phys = block[:, 3].astype(np.int32) if ntags >= 1 \
            else np.zeros(run, dtype=np.int32)
        elementary = block[:, 4] if ntags >= 2 else None
        corners = block[:, 3 + ntags:3 + ntags + n_corners]
        _accumulate_cells(cells, family, n_nodes, corners, phys, elementary)
        p += run * stride
    total = sum(sum(len(e) for e in c["entity"])
                for family, c in cells.items() if family is not None)
    if total != num_elems:
        raise MshParseError(
            f"Parsed {total} elements, header declared {num_elems}")
    return node_tags, points, cells


def _read_v22_binary(data, sections):
    """Binary MSH v2.2 (fTetWild / PyMesh MshSaver default output).

    Same bulk-frombuffer strategy as the v4.1 binary reader: one structured
    read for all nodes, one int32 read per element block (one block per
    element type in practice), so it parses at the same speed.
    """
    # Endianness probe: the binary int 1 written after the ASCII
    # "2.2 1 data-size" line inside $MeshFormat.
    fmt_body = data[slice(*sections["MeshFormat"])]
    nl = fmt_body.find(b"\n")
    probe = fmt_body[nl + 1:nl + 5]
    if len(probe) == 4 and np.frombuffer(probe, "<i4")[0] == 1:
        endian = "<"
    elif len(probe) == 4 and np.frombuffer(probe, ">i4")[0] == 1:
        endian = ">"
    else:
        raise MshParseError("Bad endianness probe in binary MSH v2.2 header")

    buf = data[slice(*sections["Nodes"])]
    nl = buf.find(b"\n")
    num_nodes = int(buf[:nl])
    # packed record: int32 node tag + 3x float64 coordinates (28 bytes)
    record = np.dtype({"names": ["tag", "xyz"],
                       "formats": [endian + "i4", (endian + "f8", (3,))],
                       "offsets": [0, 4], "itemsize": 28})
    nodes = np.frombuffer(buf, record, num_nodes, nl + 1)
    node_tags = nodes["tag"].astype(np.int64)
    points = nodes["xyz"].astype(np.float64)

    buf = data[slice(*sections["Elements"])]
    nl = buf.find(b"\n")
    num_elems = int(buf[:nl])
    ints = np.frombuffer(buf, endian + "i4", (len(buf) - nl - 1) // 4, nl + 1)
    p = 0
    cells = {}
    parsed = 0
    while parsed < num_elems:
        etype = int(ints[p]); n_follow = int(ints[p + 1]); ntags = int(ints[p + 2])
        if etype not in GMSH_ELEM_TYPES:
            raise MshParseError(f"Unsupported gmsh element type {etype}")
        family, n_nodes, n_corners = GMSH_ELEM_TYPES[etype]
        if n_follow == 1:
            # gmsh writes one header PER ELEMENT: coalesce the consecutive
            # run of identical (type, ntags) single-element blocks and
            # process it as one reshape, like the ASCII v2.2 reader.
            stride = 4 + ntags + n_nodes  # header + elm-number + tags + nodes
            max_run = min((len(ints) - p) // stride, num_elems - parsed)
            view = ints[p:p + max_run * stride].reshape(max_run, stride)
            same = ((view[:, 0] == etype) & (view[:, 1] == 1)
                    & (view[:, 2] == ntags))
            run = max_run if same.all() else max(int(np.argmin(same)), 1)
            block = view[:run, 3:]
            p += run * stride
            parsed += run
        else:
            # block-grouped writer (PyMesh/fTetWild): one bulk read
            stride = 1 + ntags + n_nodes  # elm-number, tags..., node tags...
            block = ints[p + 3:p + 3 + n_follow * stride].reshape(
                n_follow, stride)
            p += 3 + n_follow * stride
            parsed += n_follow
        # first tag is the physical group, matching the ASCII v2.2 path
        phys = block[:, 1].astype(np.int32) if ntags >= 1 \
            else np.zeros(len(block), dtype=np.int32)
        elementary = block[:, 2] if ntags >= 2 else None
        corners = block[:, 1 + ntags:1 + ntags + n_corners].astype(np.int64)
        _accumulate_cells(cells, family, n_nodes, corners, phys, elementary)
    if parsed != num_elems:
        raise MshParseError(
            f"Parsed {parsed} elements, header declared {num_elems}")
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
    except (ValueError, IndexError, TypeError, OverflowError,
            struct.error) as exc:
        raise MshParseError(
            f"the file is malformed or cut short ({exc})") from exc


def _read_msh_data(data):
    sections = _find_sections(data)
    if "MeshFormat" not in sections:
        raise MshParseError("Not a Gmsh MSH file (missing $MeshFormat)")
    fmt = data[slice(*sections["MeshFormat"])].split()
    version = float(fmt[0])
    is_binary = int(fmt[1]) == 1
    if 4.0 <= version < 4.1 or version >= 5.0 or 2.3 <= version < 4.0 \
            or version < 2.0:
        raise MshParseError(
            f"MSH version {fmt[0].decode('ascii', 'replace')} is not "
            "supported (MSH 2.2 and 4.1 are, as for PolyFEM): save the mesh "
            "again from Gmsh as version 4.1 (Mesh.MshFileVersion = 4.1)")

    if "Nodes" not in sections or "Elements" not in sections:
        raise MshParseError("MSH file has no $Nodes/$Elements sections")

    physical_names = _parse_physical_names(data, sections)
    notes = {"duplicates": {}, "shared_groups": set()}

    if version >= 4.0:
        ent_phys = {}
        if "Entities" in sections:
            if is_binary:
                ent_phys = _parse_entities_binary(data[slice(*sections["Entities"])])
            else:
                ent_phys = _parse_entities_ascii(_tokens(data, sections["Entities"]))
        if "PartitionedEntities" in sections:
            if is_binary:
                _parse_partitioned_binary(
                    data[slice(*sections["PartitionedEntities"])], ent_phys)
            else:
                _parse_partitioned_ascii(
                    _tokens(data, sections["PartitionedEntities"]), ent_phys)
        if is_binary:
            node_tags, points, cells = _read_v41_binary(
                data, sections, ent_phys, notes)
        else:
            node_tags, points, cells = _read_v41_ascii(
                data, sections, ent_phys, notes)
    else:
        if is_binary:
            node_tags, points, cells = _read_v22_binary(data, sections)
        else:
            node_tags, points, cells = _read_v22_ascii(data, sections)

    lookup = _tag_index_map(node_tags)
    finalized = _finalize_cells(cells, lookup, notes)
    return {
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


def _moved_nodes_v22_ascii(body, positions):
    tok = body.split()
    count = int(tok[0])
    if count != len(positions) or len(tok) < 1 + 4 * count:
        raise MshParseError(
            f"$Nodes lists {count} nodes, {len(positions)} positions given")
    lines = [tok[0]]
    for row in range(count):
        lines.append(tok[1 + 4 * row] + b" " + _coordinate_text(positions[row]))
    return b"\n".join(lines) + b"\n"


def _moved_nodes_v41_ascii(body, positions):
    tok = body.split()
    blocks, total = int(tok[0]), int(tok[1])
    if total != len(positions):
        raise MshParseError(
            f"$Nodes lists {total} nodes, {len(positions)} positions given")
    lines = [b" ".join(tok[:4])]
    p, row = 4, 0
    for _ in range(blocks):
        count = int(tok[p + 3])
        if int(tok[p + 2]):
            raise MshParseError("parametric node coordinates are not "
                                "supported")
        lines.append(b" ".join(tok[p:p + 4]))
        p += 4
        lines.extend(tok[p:p + count])                      # node tags
        p += count + 3 * count
        lines.extend(_coordinate_text(xyz)
                     for xyz in positions[row:row + count])
        row += count
    if row != total:
        raise MshParseError("$Nodes blocks do not add up to the node count")
    return b"\n".join(lines) + b"\n"


def _moved_nodes_v22_binary(data, span, fmt_span, positions):
    fmt_body = data[fmt_span[0]:fmt_span[1]]
    nl = fmt_body.find(b"\n")
    probe = fmt_body[nl + 1:nl + 5]
    endian = "<" if np.frombuffer(probe, "<i4")[0] == 1 else ">"
    nl = data.find(b"\n", span[0])
    count = int(data[span[0]:nl])
    if count != len(positions):
        raise MshParseError(
            f"$Nodes lists {count} nodes, {len(positions)} positions given")
    out = bytearray(data)
    base = nl + 1
    coordinates = np.ascontiguousarray(positions, dtype=endian + "f8")
    for row in range(count):
        offset = base + 28 * row + 4          # int32 tag, then 3 float64
        out[offset:offset + 24] = coordinates[row].tobytes()
    return bytes(out)


def _moved_nodes_v41_binary(data, span, positions):
    p = span[0]
    blocks, total = (int(v) for v in np.frombuffer(data, "<u8", 2, p))
    if total != len(positions):
        raise MshParseError(
            f"$Nodes lists {total} nodes, {len(positions)} positions given")
    p += 32
    out = bytearray(data)
    coordinates = np.ascontiguousarray(positions, dtype="<f8")
    row = 0
    for _ in range(blocks):
        parametric = int(np.frombuffer(data, "<i4", 1, p + 8)[0])
        count = int(np.frombuffer(data, "<u8", 1, p + 12)[0])
        if parametric:
            raise MshParseError("parametric node coordinates are not "
                                "supported")
        p += 20 + 8 * count
        out[p:p + 24 * count] = coordinates[row:row + count].tobytes()
        p += 24 * count
        row += count
    if row != total:
        raise MshParseError("$Nodes blocks do not add up to the node count")
    return bytes(out)


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
    fmt = data[slice(*sections["MeshFormat"])].split()
    version, binary = float(fmt[0]), int(fmt[1]) == 1
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(positions).all():
        raise MshParseError("node positions must be finite numbers")
    span = sections["Nodes"]
    if binary:
        result = (_moved_nodes_v41_binary(data, span, positions)
                  if version >= 4.0 else
                  _moved_nodes_v22_binary(data, span,
                                          sections["MeshFormat"], positions))
    else:
        body = data[span[0]:span[1]]
        body = (_moved_nodes_v41_ascii(body, positions) if version >= 4.0
                else _moved_nodes_v22_ascii(body, positions))
        result = data[:span[0]] + body + data[span[1]:]
    temporary = out_path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(result)
    os.replace(temporary, out_path)
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
