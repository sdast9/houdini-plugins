"""MSH Reader 3.0 review regressions. Run with hython after src/build_all.py."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import hou
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / "src"), str(BASE / "src" / "common")]

import build_all
import hda_build
import msh_parser


TRI = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def section(name, body, separator=b"\n"):
    name = name.encode()
    return b"$" + name + b"\n" + body + separator + b"$End" + name + b"\n"


def row(values):
    return " ".join(map(str, values)).encode()


def msh22(points, elements, tags=None, names=(), binary=False, separator=b"\n"):
    tags = list(range(1, len(points) + 1)) if tags is None else tags
    if binary:
        fmt = b"2.2 1 8\n" + struct.pack("<i", 1)
        nodes = row([len(points)]) + b"\n" + b"".join(
            struct.pack("<i3d", tag, *xyz) for tag, xyz in zip(tags, points))
        records = []
        for i, (etype, element_tags, conn) in enumerate(elements, 1):
            fields = (etype, 1, len(element_tags), i, *element_tags, *conn)
            records.append(struct.pack(f"<{len(fields)}i", *fields))
        elems = row([len(elements)]) + b"\n" + b"".join(records)
    else:
        fmt = b"2.2 0 8"
        nodes = b"\n".join([row([len(points)]),
                            *(row((tag, *xyz)) for tag, xyz in zip(tags, points))])
        elems = b"\n".join([row([len(elements)]),
                            *(row((i, etype, len(element_tags), *element_tags, *conn))
                              for i, (etype, element_tags, conn) in enumerate(elements, 1))])
    data = section("MeshFormat", fmt, separator)
    if names:
        name_rows = [f'{dim} {tag} "{name}"'.encode() for dim, tag, name in names]
        data += section("PhysicalNames", b"\n".join([row([len(names)]), *name_rows]))
    return data + section("Nodes", nodes, separator) + section("Elements", elems, separator)


def partitioned41(endian=None, size=8, tags=(10, 11, 12, 13), declared=3,
                  separator=b"\n", points=(*TRI, (0, 0, 1)), parent_collision=False):
    """A tet inherits its volume's group. A face on the same parent is an
    untagged interface, and a second face explicitly has a physical tag."""
    size_code = {4: "I", 8: "Q"}[size]

    def pack(fmt, *values):
        if endian is None:
            return row(values) + b"\n"
        return struct.pack(endian + fmt.replace("S", size_code), *values)

    fmt = b"4.1 1 " + row([size]) + b"\n" + pack("i", 1) if endian else b"4.1 0 8"
    bbox = (0, 0, 0, 1, 1, 1)
    entities = pack("4S", 0, 0, 0, 1) + pack("i6dSiS", 10, *bbox, 1, 123, 0)
    partitions = pack("2S2i4S", 2, 1, 44, 2, 0, 0, 2, 1 + parent_collision)
    volumes = [(10, (999,)), (40, ())] if parent_collision else [(40, ())]
    for tag, phys in [(30, ()), (31, (-17,)), *volumes]:
        partitions += (pack("3iSi6dS", tag, 3, 10, 1, 1, *bbox, len(phys))
                       + (pack("i", *phys) if phys else b"") + pack("S", 0))

    node_payload = pack("4S3iS4S", 1, 4, min(tags), max(tags), 3, 40, 0, 4, *tags)
    xyz = np.array(points, dtype=float)
    node_payload += pack("12d", *xyz.ravel())
    # The empty hex block must not leave a family slot. This also exercises
    # multiple blocks whose declared total is independent of the block count.
    elems = (pack("4S", 4, declared, 1, 3) + pack("3iS", 3, 40, 5, 0)
             + pack("3iS5S", 3, 40, 4, 1, 1, *tags)
             + pack("3iS4S", 2, 30, 2, 1, 2, *tags[:3])
             + pack("3iS4S", 2, 31, 2, 1, 3, tags[0], tags[1], tags[3]))
    return (section("MeshFormat", fmt, separator) + section("Entities", entities, separator)
            + section("PartitionedEntities", partitions, separator)
            + section("Nodes", node_payload, separator) + section("Elements", elems, separator)), xyz


class ScratchTest(unittest.TestCase):
    def setUp(self):
        scratch = self.enterContext(tempfile.TemporaryDirectory(prefix="msh_review_"))
        self.work = Path(scratch)

    def mesh_file(self, contents):
        path = self.work / "input.msh"
        path.write_bytes(contents)
        return path

    def parse(self, contents):
        return msh_parser.read_msh(self.mesh_file(contents))

    def refused(self, contents, fragment):
        with self.assertRaisesRegex(msh_parser.MshParseError, fragment):
            self.parse(contents)


class ParserTests(ScratchTest):
    def refused_read_and_move(self, contents, points, fragment):
        self.refused(contents, fragment)
        out = self.work / "moved.msh"
        out.write_bytes(b"previous export")
        with self.assertRaisesRegex(msh_parser.MshParseError, fragment):
            msh_parser.write_moved_nodes(self.work / "input.msh", points, out)
        self.assertEqual(out.read_bytes(), b"previous export")

    def test_bundled_mshio_fixture(self):
        path = (BASE.parent / "upstream-merge-work" / "20261120-candidate" /
                "dependency-cache" / "mshio" / "d857" / "data" / "test_2.2_bin.msh")
        data = path.read_bytes()
        self.assertIn(b"\x01\x00\x00\x00$EndMeshFormat", data)
        for name in (b"Nodes", b"Elements"):
            self.assertNotIn(b"\n$End" + name, data)
        mesh = msh_parser.read_msh(path)
        self.assertEqual(mesh["version"], "2.2")
        self.assertEqual(mesh["node_tags"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(mesh["cells"]["quad"]["corners"].shape, (2, 4))
        out = self.work / "moved.msh"
        moved = mesh["points"] + 1
        msh_parser.write_moved_nodes(path, moved, out)
        np.testing.assert_array_equal(msh_parser.read_msh(out)["points"], moved)
        self.assertEqual(data.split(b"$EndNodes")[1], out.read_bytes().split(b"$EndNodes")[1])

    def test_binary_section_boundaries_follow_record_lengths(self):
        # An end-marker-shaped byte sequence can occur inside binary floating
        # point values. The final data byte can also be a literal LF.
        v22_points = np.asarray(TRI, dtype="<f8")
        v22_points[-1].view("u1")[:10] = np.frombuffer(b"$EndNodes\n", dtype="u1")
        v22_points[-1].view("u1")[-1] = 10
        v22 = msh22(v22_points, [(2, (7,), (1, 2, 3))], binary=True, separator=b"")
        v41, xyz = partitioned41(">", 4, separator=b"")
        start = v41.index(b"$Entities\n") + len(b"$Entities\n") + 5 * 4
        v41 = v41[:start] + b"$EndEntities\n" + v41[start + len(b"$EndEntities\n"):]
        for data, expected in ((v22, v22_points), (v41, xyz)):
            mesh = self.parse(data)
            np.testing.assert_array_equal(mesh["points"], expected)
            out = self.work / "moved.msh"
            msh_parser.write_moved_nodes(self.work / "input.msh", expected + 1, out)
            np.testing.assert_array_equal(msh_parser.read_msh(out)["points"], expected + 1)
        self.assertEqual(mesh["cells"]["tet"]["entity"].tolist(), [123])
        self.assertEqual(mesh["cells"]["tri"]["entity"].tolist(), [0, 17])

    def test_binary_node_lengths_for_read_and_move(self):
        for separator in (b"", b"\n", b"\r\n"):
            v22 = msh22(TRI, [(2, (), (1, 2, 3))], binary=True, separator=separator)
            v41, xyz = partitioned41("<", separator=separator)
            for data, points in ((v22, TRI), (v41, xyz)):
                end = data.index(separator + b"$EndNodes")
                with self.subTest(separator=separator, version=data[12:15]):
                    np.testing.assert_array_equal(self.parse(data)["points"], points)
                    self.refused_read_and_move(data[:end - 1] + data[end:], points, r"\$Nodes")
                    for extra in (b"x", b"\n" + separator if separator else b"\n\n"):
                        self.refused_read_and_move(data[:end] + extra + data[end:], points, r"\$Nodes")

    def test_v22_copy_keys_use_wide_entity_ids_and_exact_node_order(self):
        points = [*TRI, (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0)]
        first = (1, 2, 3, 4, 5, 6)
        different_corners = (1, 2, 6, 4, 5, 3)
        inverted = (1, 3, 2, 6, 5, 4)
        entity = 1200000000
        elements = [(9, (3, entity), first), (9, (2, entity), different_corners),
                    (9, (2, entity), inverted), (9, (1, entity), first)]
        for binary in (False, True):
            mesh = self.parse(msh22(points, elements, binary=binary))
            self.assertEqual(mesh["duplicate_elements"], {2: 1})
            self.assertEqual(mesh["shared_groups"], [(2, (1, 2, 3))])
            cell = mesh["cells"]["tri"]
            self.assertEqual(cell["entity"].tolist(), [2, 2, 1])
            self.assertEqual(cell["corners"].tolist(), [[0, 1, 5], [0, 2, 1], [0, 1, 2]])

    def test_v22_shared_groups_exclude_entities_without_removed_copies(self):
        triangles = [(2, (2, 10), (1, 2, 3)), (2, (1, 10), (1, 2, 3)),
                     (2, (4, 20), (1, 3, 4)), (2, (5, 20), (5, 6, 7))]
        for binary in (False, True):
            with self.subTest(binary=binary):
                mesh = self.parse(msh22(CUBE, triangles, binary=binary))
                self.assertEqual(mesh["duplicate_elements"], {2: 1})
                self.assertEqual(mesh["shared_groups"], [(2, (1, 2))])
                self.assertEqual(mesh["cells"]["tri"]["entity"].tolist(), [1, 4, 5])

    def test_partition_groups_inherit_from_model_even_when_tags_collide(self):
        for endian, size in ((None, 8), ("<", 4), (">", 8)):
            with self.subTest(endian=endian, size=size):
                data, _ = partitioned41(endian, size, parent_collision=True)
                mesh = self.parse(data)
                self.assertEqual(mesh["cells"]["tet"]["entity"].tolist(), [123])
                self.assertEqual(mesh["cells"]["tet"]["gmsh_types"], [4])
                self.assertNotIn("hex", mesh["cells"])

    def test_nonfinite_coordinates_in_all_node_formats(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            points = [*TRI[:2], (0, 1, bad)]
            for binary in (False, True):
                with self.subTest(bad=bad, binary=binary):
                    self.refused_read_and_move(
                        msh22(points, [(2, (), (1, 2, 3))], binary=binary),
                        TRI, "Non-finite coordinates")
                    v41, _ = partitioned41("<" if binary else None, points=(*points, (0, 0, 1)))
                    self.refused_read_and_move(v41, [*TRI, (0, 0, 1)], "Non-finite coordinates")

    def test_v41_element_type_must_match_block_dimension(self):
        for endian in (None, "<"):
            data, _ = partitioned41(endian)
            for count in (0, 1):
                with self.subTest(endian=endian, count=count):
                    if endian:
                        body = struct.pack("<4Q3iQ", 1, count, 1, 1, 3, 40, 2, count)
                        if count:
                            body += struct.pack("<4Q", 1, 10, 11, 12)
                    else:
                        body = row((1, count, 1, 1)) + b"\n" + row((3, 40, 2, count))
                        if count:
                            body += b"\n1 10 11 12"
                    self.refused(data[:data.index(b"$Elements\n")] + section("Elements", body),
                                 "block dimension 3 disagrees with gmsh element type 2")

    def test_ascii_metadata_count_and_list_validation(self):
        data, _ = partitioned41()
        fixtures = {
            "Entities": (
                b"0 0 0", b"0 0 -1 0", b"1 0 0 0\n1 0 0 0 -1",
                b"1 0 0 0\n1 0 0 0 2 7", b"0 1 0 0\n1 0 0 0 1 1 1 0 -1",
                b"0 1 0 0\n1 0 0 0 1 1 1 0 2 1", b"0 0 0 0 99",
            ),
            "PartitionedEntities": (
                b"-1 0 0 0 0 0", b"1 -1 0 0 0 0", b"1 1 5", b"1 0 0 0 -1 0",
                b"1 0 1 0 0 0\n1 0 1 -1", b"1 0 1 0 0 0\n1 0 1 2 1",
                b"1 0 1 0 0 0\n1 0 1 0 0 0 0 -1", b"1 0 1 0 0 0\n1 0 1 0 0 0 0 2 7",
                b"1 0 0 1 0 0\n1 1 1 0 0 0 0 1 1 1 0 -1",
                b"1 0 0 1 0 0\n1 1 1 0 0 0 0 1 1 1 0 2 3", b"1 0 0 0 0 0 99",
            ),
            "PhysicalNames": (b"-1", b'0\n2 17 "face"', b'2\n2 17 "face"', b"1\n2 17",
                              b'2\n2 17 "face"\n2 17 "duplicate"'),
        }
        for name, bodies in fixtures.items():
            encoded = name.encode()
            if name == "PhysicalNames":
                before, after = data, b""
            else:
                before = data[:data.index(b"$" + encoded + b"\n")]
                after = data[data.index(b"$End" + encoded + b"\n") + len(encoded) + 5:]
            for body in bodies:
                with self.subTest(section=name, body=body):
                    self.refused(before + section(name, body) + after, rf"\${name}")

    def test_strict_version_encoding_and_malformed_input_errors(self):
        good = msh22(TRI, [(2, (), (1, 2, 3))])
        for version in (b"2.0", b"2.1", b"2.20", b"4.0", b"4.2", b"nan"):
            with self.subTest(version=version):
                self.refused_read_and_move(good.replace(b"2.2 0 8", version + b" 0 8"),
                                           TRI, "version .* is not supported")
        for flag in (b"-1", b"2"):
            self.refused_read_and_move(good.replace(b"2.2 0 8", b"2.2 " + flag + b" 8"),
                                       TRI, "Invalid MSH file encoding")
        for data in (good.replace(b"2.2 0 8", b"2.2"), good.replace(b"2.2 0 8", b"2.2 0 x"),
                     good.replace(b"3\n1 0 0 0", b"bad\n1 0 0 0"),
                     good.replace(b"3\n1 0 0 0", b"3\n1 bad 0 0")):
            self.refused_read_and_move(data, TRI, "malformed or cut short")

    def test_moved_nodes_rejects_invalid_and_duplicate_source_tags(self):
        for binary in (False, True):
            for tags, fragment in (((1, 2, 0), "Invalid node tag"), ((1, 2, 2), "Duplicate node tag")):
                with self.subTest(binary=binary, tags=tags):
                    self.refused_read_and_move(msh22(TRI, [(2, (), (1, 2, 3))], tags, binary=binary),
                                               TRI, fragment)
                    v41, xyz = partitioned41("<" if binary else None, tags=(*tags, 4))
                    self.refused_read_and_move(v41, xyz, fragment)

    def test_moved_nodes_atomic_replace_cleanup_and_modes(self):
        path = self.mesh_file(msh22(TRI, [(2, (), (1, 2, 3))]))
        out = self.work / "moved.msh"
        old_temp = out.with_suffix(".msh.tmp")
        old_temp.write_bytes(b"unrelated file")
        out.write_bytes(b"previous export")
        out.chmod(0o640)
        seen = []
        replace = os.replace

        def fail_replace(source, dest):
            seen.append(Path(source))
            self.assertEqual(Path(source).parent, self.work)
            self.assertEqual(Path(dest).read_bytes(), b"previous export")
            raise OSError("replace interrupted")

        with patch.object(msh_parser.os, "replace", side_effect=fail_replace):
            with self.assertRaisesRegex(OSError, "replace interrupted"):
                msh_parser.write_moved_nodes(path, TRI, out)
        self.assertEqual(out.read_bytes(), b"previous export")
        self.assertFalse(seen[0].exists())

        def record_replace(source, dest):
            seen.append(Path(source))
            replace(source, dest)

        with patch.object(msh_parser.os, "replace", side_effect=record_replace):
            self.assertEqual(msh_parser.write_moved_nodes(path, TRI, out), out)
        self.assertNotEqual(seen[0], seen[1])
        self.assertFalse(seen[1].exists())
        self.assertEqual(old_temp.read_bytes(), b"unrelated file")
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o640)

        new_out = self.work / "new.msh"
        control = self.work / "default-mode"
        control.touch()
        msh_parser.write_moved_nodes(path, TRI, new_out)
        self.assertEqual(stat.S_IMODE(new_out.stat().st_mode), stat.S_IMODE(control.stat().st_mode))
        self.assertEqual(set(self.work.iterdir()), {path, out, old_temp, new_out, control})

    def test_ascii_tag_precision_and_integrity(self):
        high = 2**53 + 1
        for tags in ((high, high + 1, high + 2), (3, 1, 2), (1, 2, 2**63 - 1)):
            v22 = msh22(TRI, [(2, (), tags)], tags)
            v41 = partitioned41(tags=(*tags, 4))[0]
            for data, expected in ((v22, tags), (v41, (*tags, 4))):
                with self.subTest(tags=expected):
                    mesh = self.parse(data)
                    self.assertEqual(mesh["node_tags"].dtype, np.dtype(np.int64))
                    self.assertEqual(mesh["node_tags"].tolist(), list(expected))
                    self.assertEqual(mesh["cells"]["tri"]["corners"][0].tolist(), [0, 1, 2])

        for tag, error in ((0, "must be >= 1"), (-1, "must be >= 1"),
                           (2, "Duplicate node tag 2"), (2**63, "range"),
                           ("1.5", "Invalid or out-of-range integer tag")):
            tags = (1, 2, tag)
            with self.subTest(tag=tag):
                self.refused(msh22(TRI, [(2, (), (1, 2, 3))], tags), error)
                v41 = partitioned41()[0]
                nodes = section("Nodes", b"1 3 1 3\n2 1 0 3\n"
                                + row(tags) + b"\n0 0 0\n1 0 0\n0 1 0")
                start = v41.index(b"$Nodes\n")
                end = v41.index(b"$Elements\n")
                self.refused(v41[:start] + nodes + v41[end:], error)

        for bad_ref in (2**63, 1.5, 2**53 + 2):
            data = msh22(TRI, [(2, (), (1, 2, bad_ref))], (1, 2, high))
            self.refused(data, r"integer tag in \$Elements|references node tag")
        for tags, fragment in (((1, 0, 3, 4), "must be >= 1"),
                               ((1, 2, 2, 4), "Duplicate node tag"),
                               ((1, 2, 3, 2**63), "outside signed Int64")):
            self.refused(partitioned41(">", tags=tags)[0], fragment)

    def test_binary_format_partition_interfaces_and_empty_blocks(self):
        formats = ((None, 8), ("<", 4), ("<", 8), (">", 4), (">", 8))
        for endian, size in formats:
            with self.subTest(endian=endian, size=size):
                high = 2**32 - 4 if size == 4 else 2**53 + 1
                tags = tuple(range(high, high + 4))
                data, xyz = partitioned41(endian, size, tags)
                mesh = self.parse(data)
                np.testing.assert_array_equal(mesh["points"], xyz)
                self.assertEqual(mesh["node_tags"].tolist(), list(tags))
                self.assertEqual(set(mesh["cells"]), {"tri", "tet"})
                self.assertEqual(mesh["cells"]["tet"]["entity"].tolist(), [123])
                self.assertEqual(mesh["cells"]["tri"]["entity"].tolist(), [0, 17])
                self.assertEqual(mesh["cells"]["tri"]["corners"].tolist(), [[0, 1, 2], [0, 1, 3]])
                out = self.work / "moved.msh"
                msh_parser.write_moved_nodes(self.work / "input.msh", xyz + 0.25, str(out))
                self.assertEqual(msh_parser.read_msh(out)["node_tags"].tolist(), list(tags))
                np.testing.assert_array_equal(msh_parser.read_msh(out)["points"], xyz + 0.25)
                node_end = b"$EndNodes\n"
                # Moving nodes preserves all other sections, even in binary.
                self.assertEqual(data.split(node_end)[1], out.read_bytes().split(node_end)[1])
                for declared in (2, 4):
                    self.refused(partitioned41(endian, size, tags, declared)[0],
                                 rf"\$Elements blocks hold 3, but the header declared {declared}")

        good = partitioned41(">", 4)[0]
        self.refused(good.replace(b"4.1 1 4", b"4.1 1 2"), "data size 2")
        self.refused(good.replace(struct.pack(">i", 1), struct.pack(">i", 3), 1),
                     "endianness probe")
        for name in (b"Entities", b"PartitionedEntities", b"Nodes", b"Elements"):
            with self.subTest(truncated=name):
                end = good.index(b"\n$End" + name + b"\n")
                self.refused(good[:end - 4] + good[end:], name.decode())
        # An impossible header count cannot drive allocation before the read.
        data = partitioned41("<")[0]
        start = data.index(b"$Nodes\n") + len(b"$Nodes\n")
        self.refused(data[:start] + struct.pack("<QQ", 2**63, 2**63) + data[start + 16:],
                     r"Binary \$Nodes")

    def test_duplicate_sections_and_line_anchored_names(self):
        data = msh22(TRI, [(2, (7,), (1, 2, 3))],
                     names=[(2, 7, "name $EndPhysicalNames then $Nodes")])
        data = data.replace(b"\n", b"\r\n").rstrip(b"\r\n")
        self.assertEqual(self.parse(data)["physical_names"],
                         {(2, 7): "name $EndPhysicalNames then $Nodes"})

        data = partitioned41()[0]
        data += section("PhysicalNames", b"1\n2 17 \"face\"")
        for name in ("MeshFormat", "Nodes", "Elements", "Entities",
                     "PartitionedEntities", "PhysicalNames"):
            start = data.index(("$" + name + "\n").encode())
            marker = ("$End" + name + "\n").encode()
            end = data.index(marker, start) + len(marker)
            self.refused(data + data[start:end], rf"Duplicate \${name} section")

    def test_full_higher_order_references_and_deduplication(self):
        points = [*TRI, (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0), (0.1, 0.5, 0)]
        first = (1, 2, 3, 4, 5, 6)
        second = (1, 2, 3, 4, 5, 7)
        elems = [(9, (3, 20), first), (2, (8, 21), (1, 2, 3)),
                 (9, (2, 20), second), (9, (1, 20), first)]
        for binary in (False, True):
            mesh = self.parse(msh22(points, elems, binary=binary))
            self.assertEqual(mesh["duplicate_elements"], {2: 1})
            cell = mesh["cells"]["tri"]
            self.assertEqual(cell["entity"].tolist(), [8, 2, 1])
            self.assertEqual(cell["order"].tolist(), [0, 1, 2])
            self.assertEqual(cell["num_nodes"], 6)
            self.assertEqual(cell["gmsh_types"], [2, 9])
            bad = [(9, (3, 20), (*first[:5], 99)), (9, (1, 20), first)]
            self.refused(msh22(points, bad, binary=binary), "references node tag 99")

        for endian in (None, ">"):
            data = partitioned41(endian)[0]
            start = data.index(b"$Elements\n")
            if endian is None:
                body = b"1 1 1 1\n2 31 9 1\n1 10 11 12 10 11 99"
            else:
                body = struct.pack(">4Q3iQ7Q", 1, 1, 1, 1, 2, 31, 9, 1,
                                   1, 10, 11, 12, 10, 11, 99)
            self.refused(data[:start] + section("Elements", body), "references node tag 99")

    def test_alternating_v22_scans_only_a_linear_number_of_rows(self):
        # Measure inspected rows, not wall time, to catch repeated full-tail
        # scans independently of hardware load.
        class CountedView:
            def __init__(self, view, counter):
                self.view, self.counter = view, counter

            def __len__(self):
                return len(self.view)

            def __getitem__(self, key):
                result = self.view[key]
                self.counter[0] += len(result) if isinstance(key, slice) else 1
                return result

        n = 4096
        run_scanner = msh_parser._matching_run
        for binary in (False, True):
            for case, pair in (("type", [(2, (), (1, 2, 3)), (3, (), (1, 2, 3, 4))]),
                               ("tag count", [(2, (), (1, 2, 3)), (2, (7,), (1, 2, 3))])):
                with self.subTest(binary=binary, case=case):
                    counter = [0]

                    def measure(view, columns, expected):
                        return run_scanner(CountedView(view, counter), columns, expected)

                    data = msh22(CUBE, pair * (n // 2), binary=binary)
                    with patch.object(msh_parser, "_matching_run", side_effect=measure) as scan:
                        mesh = self.parse(data)
                    self.assertEqual(scan.call_count, n)
                    self.assertLessEqual(counter[0], 3 * n)
                    if case == "type":
                        np.testing.assert_array_equal(
                            mesh["cells"]["tri"]["order"], np.arange(0, n, 2))
                        np.testing.assert_array_equal(
                            mesh["cells"]["quad"]["order"], np.arange(1, n, 2))
                    else:
                        self.assertEqual(mesh["cells"]["tri"]["entity"][:4].tolist(), [0, 7, 0, 7])

    def test_moved_nodes_shape_finiteness_and_truncation(self):
        data = msh22(TRI, [(2, (), (1, 2, 3))])
        path = self.mesh_file(data)
        out = self.work / "out.msh"
        out.write_bytes(b"old output")
        for positions, error in ((np.zeros(9), "shape"), (np.zeros((1, 3, 3)), "shape"),
                                 (np.zeros((3, 2)), "shape"), (np.zeros((4, 3)), "positions given"),
                                 (np.full((3, 3), np.nan), "finite"),
                                 (np.full((3, 3), np.inf), "finite")):
            with self.assertRaisesRegex(msh_parser.MshParseError, error):
                msh_parser.write_moved_nodes(path, positions, str(out))
            self.assertEqual(out.read_bytes(), b"old output")

        start = data.index(b"$Nodes\n")
        end = data.index(b"$Elements\n")
        malformed = [b"", b"3\n1 0 0 0\n2 1 0 0\n3"]
        for body in malformed:
            with self.assertRaisesRegex(msh_parser.MshParseError, r"Truncated ASCII \$Nodes"):
                path.write_bytes(data[:start] + section("Nodes", body) + data[end:])
                msh_parser.write_moved_nodes(path, TRI, str(out))
        data = data.replace(b"2.2 0 8", b"4.1 0 8")
        malformed = [b"1 3", b"1 3 1 3\n2 1 1", b"1 3 1 3\n2 1 1 3\n1 2",
                     b"1 3 1 3\n2 1 1 3\n1 2 3\n0 0 0 0 0\n1 0 0 0 0\n0 1 0 0"]
        for body in malformed:
            with self.assertRaisesRegex(msh_parser.MshParseError, r"Truncated ASCII \$Nodes"):
                path.write_bytes(data[:start] + section("Nodes", body) + data[end:])
                msh_parser.write_moved_nodes(path, TRI, str(out))
        good_nodes = b"1 3 1 3\n2 1 1 3\n1 2 3\n0 0 0 1 2\n1 0 0 3 4\n0 1 0 5 6"
        path.write_bytes(data[:start] + section("Nodes", good_nodes) + data[end:])
        msh_parser.write_moved_nodes(path, np.asarray(TRI) + 2, str(out))
        self.assertIn(b"2.0 3.0 2.0 5 6\n", out.read_bytes())
        binary = msh22(TRI, [(2, (), (1, 2, 3))], binary=True)
        path.write_bytes(binary)
        msh_parser.write_moved_nodes(path, np.asarray(TRI) + 2, str(out))
        np.testing.assert_array_equal(msh_parser.read_msh(out)["points"], np.asarray(TRI) + 2)


class ReaderTests(ScratchTest):
    @classmethod
    def setUpClass(cls):
        cls.library = BASE / "sop_MSH_Reader.3.0.hdanc"
        hou.hda.installFile(str(cls.library), force_use_assets=True)

    def reader(self, data, surfaces=True, groups=True):
        container = hou.node("/obj").createNode("geo")
        self.addCleanup(container.destroy)
        node = container.createNode("MSH_Reader::3.0")
        self.assertEqual(Path(node.type().definition().libraryFilePath()), self.library)
        node.setParms({"File": str(self.mesh_file(data)), "import_surfaces": surfaces,
                       "create_groups": groups})
        return node

    def test_nonfinite_tet_volumes_count_as_invalid(self):
        unit_tet = [*TRI, (0, 0, 1)]
        data = msh22(unit_tet, [(4, (), (1, 2, 3, 4))])
        module = self.reader(data).hdaModule()
        points = np.asarray([*unit_tet, (1e200, 0, 0), (0, 1e200, 0),
                             (0, 0, 1e200), (1e200, 1e200, 0)])
        corners = np.array([[0, 4, 5, 7], [0, 4, 5, 6], [0, 5, 4, 6], [0, 1, 2, 3]])
        with np.errstate(over="ignore", invalid="ignore"):
            volumes = module._tet_volume6(points[corners])
        np.testing.assert_array_equal(volumes, [np.nan, np.inf, -np.inf, 1.0])
        self.assertEqual(module._primary_quality(points, {"tet": {"corners": corners}}, ("tet",)),
                         (3, 0))

    def test_2d_winding_boundary_attributes_and_family_order(self):
        points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)]
        elems = [(2, (7,), (1, 2, 4)), (3, (8,), (1, 2, 3, 4)),
                 (2, (7,), (2, 5, 3)), (1, (10,), (1, 2))]
        node = self.reader(msh22(points, elems))
        prims = node.geometry().prims()
        self.assertEqual([p.attribValue("msh_element_order") for p in prims], [0, 2, 1, -1])
        self.assertEqual([p.attribValue("msh_physical_tag") for p in prims], [7, 7, 8, 10])
        self.assertEqual([p.attribValue("surface_entity") for p in prims], [-1, -1, -1, 10])
        self.assertEqual([p.attribValue("is_volume") for p in prims], [1, 1, 1, 0])
        self.assertEqual([p.normal()[2] for p in prims[:3]], [1.0, 1.0, 1.0])

    def test_3d_boundary_tags_winding_and_group_collisions(self):
        elems = [(4, (6,), (1, 2, 4, 5)), (5, (5,), tuple(range(1, 9))),
                 (2, (7,), (1, 2, 4)), (3, (8,), (1, 2, 3, 4)),
                 (2, (9,), (5, 6, 8))]
        names = [(3, 6, "5"), (2, 7, "wall-A"), (2, 8, "wall A"), (2, 9, "wall_A_7")]
        node = self.reader(msh22(CUBE, elems, names=names))
        geo = node.geometry()
        prims = geo.prims()
        self.assertEqual([p.attribValue("msh_physical_tag") for p in prims], [6, 5, 7, 9, 8])
        self.assertEqual([p.attribValue("surface_entity") for p in prims], [-1, -1, 7, 9, 8])
        self.assertEqual([p.attribValue("msh_element_order") for p in prims], [0, 1, -1, -1, -1])
        self.assertEqual([p.attribValue("is_volume") for p in prims], [1, 1, 0, 0, 0])
        self.assertEqual([[p.number() + 1 for p in prim.points()] for prim in prims[2:]],
                         [[1, 2, 4], [5, 6, 8], [1, 2, 3, 4]])
        groups = {g.name(): [p.number() for p in g.prims()] for g in geo.primGroups()}
        self.assertEqual(groups, {"phys_5_6": [0], "phys_5_5": [1], "surf_wall_A_7_": [2],
                                  "surf_wall_A_7": [3], "surf_wall_A_8": [4]})
        summary = geo.attribValue("msh_summary")
        self.assertIn("Disambiguated 3D physical group name phys_5", summary)
        self.assertIn("Disambiguated 2D physical group name surf_wall_A", summary)
        self.assertIn("surf_wall_A_7_", summary)
        self.assertEqual(node.evalParm("mesh_summary"), summary)

        # Evaluate the readout directly, before explicitly cooking the geometry.
        node.parm("create_groups").set(0)
        self.assertNotIn("Disambiguated", node.evalParm("mesh_summary"))
        node.parm("create_groups").set(1)
        self.assertEqual(node.evalParm("mesh_summary"), summary)
        probe = Mock(wraps=node)
        self.assertEqual(node.hdaModule().mesh_summary_text(probe), summary)
        probe.evalParm.assert_any_call("create_groups")

        node.parm("import_surfaces").set(0)
        self.assertEqual(len(node.geometry().prims()), 2)
        self.assertNotIn("surf_wall_A", node.geometry().attribValue("msh_summary"))

    def test_houdini_signed_int_limits(self):
        max_int32 = 2**31 - 1
        tags = (1, 2, max_int32)
        data = msh22(TRI, [(2, (max_int32,), tags)], tags)
        node = self.reader(data)
        geo = node.geometry()
        self.assertEqual(geo.pointIntAttribValues("msh_pt_id"), tags)
        self.assertEqual(geo.prim(0).attribValue("Entity"), max_int32)
        warning = "PolyFEM's MshReader requires node tags < 2147483647"
        self.assertIn(warning, geo.attribValue("msh_summary"))
        self.assertIn(warning, node.evalParm("mesh_summary"))
        self.assertIn(warning, " ".join(node.node("read_msh").warnings()))
        self.assertIn(warning, " ".join(node.warnings()))

        tags_below_limit = (1, 2, max_int32 - 1)
        near_limit = self.reader(msh22(TRI, [(2, (max_int32,), tags_below_limit)], tags_below_limit))
        self.assertNotIn(warning, near_limit.geometry().attribValue("msh_summary"))
        self.assertFalse(near_limit.warnings())

        cases = [
            (msh22(TRI, [(2, (), (1, 2, tag))], (1, 2, tag)),
             f"MSH node tag {tag} exceeds Houdini msh_pt_id's signed Int32 limit")
            for tag in (2**31, 2**53 + 1)
        ]
        cases.append((msh22(TRI, [(2, (), tags), (2, (max_int32,), tags)], tags),
                      "Cannot shift Entity tags by +1"))
        for data, error in cases:
            with self.subTest(error=error):
                node = self.reader(data)
                self.assertIsNone(node.geometry())
                self.assertIn(error, " ".join(node.node("read_msh").errors()))
                self.assertIn(error, " ".join(node.errors()))

    def test_polyfem_type_warning_uses_exact_primary_gmsh_codes(self):
        points = [*CUBE, *((0.5, 0.5, 0.5),) * 12]
        hex20 = (17, (1,), tuple(range(1, 21)))
        hex8 = (5, (1,), tuple(range(1, 9)))
        boundary_quad8 = (16, (2,), (1, 2, 3, 4, 9, 10, 11, 12))
        for binary in (False, True):
            mixed = msh22(points, [hex20, hex8, boundary_quad8], binary=binary)
            mesh = self.parse(mixed)
            self.assertEqual(mesh["cells"]["hex"]["gmsh_types"], [5, 17])
            self.assertEqual(mesh["cells"]["quad"]["gmsh_types"], [16])
            self.assertEqual(mesh["cells"]["hex"]["num_nodes"], 20)
            for elements, consequence in (([hex20, hex8, boundary_quad8], "skip these elements"),
                                          ([hex20, boundary_quad8], "reject this mesh")):
                with self.subTest(binary=binary, consequence=consequence):
                    node = self.reader(msh22(points, elements, binary=binary))
                    summary = node.geometry().attribValue("msh_summary")
                    self.assertEqual(node.evalParm("mesh_summary"), summary)
                    for report in (summary, *node.node("read_msh").warnings(), *node.warnings()):
                        self.assertIn("primary Gmsh element type(s) 17 (hex20)", report)
                        self.assertIn(consequence, report)
                        self.assertNotIn("16 (quad8)", report)
                    self.assertTrue(node.node("read_msh").warnings())
                    self.assertTrue(node.warnings())

        incomplete = [(22, (1,), tuple(range(1, 13))), (20, (1,), tuple(range(1, 10))),
                      boundary_quad8]
        node = self.reader(msh22(points, incomplete))
        self.assertIn("16 (quad8), 20 (tri9), 22 (tri12)",
                      node.geometry().attribValue("msh_summary"))
        supported = self.reader(msh22(points, [hex8, boundary_quad8]))
        self.assertNotIn("primary Gmsh element", supported.geometry().attribValue("msh_summary"))
        self.assertFalse(supported.warnings())

    def test_reader_warns_only_for_groups_on_entities_with_removed_copies(self):
        triangles = [(2, (2, 10), (1, 2, 3)), (2, (1, 10), (1, 2, 3)),
                     (2, (4, 20), (1, 3, 4)), (2, (5, 20), (5, 6, 7))]
        node = self.reader(msh22(CUBE, triangles))
        self.assertEqual(node.geometry().attribValue("msh_duplicate_elements"), 1)
        warnings = " ".join(node.warnings())
        self.assertIn("Physical groups 1, 2 share elements", warnings)
        self.assertNotIn("Physical groups 4, 5 share elements", warnings)

    def test_empty_higher_dimension_block_does_not_change_mesh_dimension(self):
        data = partitioned41()[0]
        start = data.index(b"$Elements\n")
        data = data[:start] + section("Elements", b"2 1 1 1\n3 40 4 0\n2 31 2 1\n1 10 11 12")
        node = self.reader(data)
        prims = node.geometry().prims()
        self.assertEqual(len(prims), 1)
        self.assertEqual(prims[0].attribValue("is_volume"), 1)
        self.assertEqual(prims[0].normal()[2], 1.0)

    def test_float64_quality_and_float32_direct_sop_warning(self):
        base = np.array((2**28, 2**28, 2**28), dtype=np.float64)
        points = np.array([*TRI, (0, 0, 1)]) + base
        # One valid tet collapses after rounding. The second was inverted in
        # float64 already, so it remains in the primary quality count only.
        elems = [(4, (2,), (1, 2, 3, 4)), (4, (2,), (1, 3, 2, 4))]
        node = self.reader(msh22(points, elems))
        geo = node.geometry()
        self.assertEqual(len(geo.prims()), 2)
        self.assertEqual(len(set(tuple(p.position()) for p in geo.points())), 1)
        summary = geo.attribValue("msh_summary")
        self.assertIn("1 inverted/zero-volume tets", summary)
        warning = "1 element(s) valid in the float64 MSH collapse or invert"
        self.assertIn(warning, summary)
        self.assertIn("Direct SOP", summary)
        self.assertIn("Recenter", summary)
        self.assertIn(warning, " ".join(node.node("read_msh").warnings()))
        self.assertIn("Direct SOP", " ".join(node.warnings()))
        self.assertIn(warning, node.evalParm("mesh_summary"))

        centered = self.reader(msh22(points - base, elems))
        self.assertNotIn("float32 P", centered.geometry().attribValue("msh_summary"))
        self.assertFalse(centered.node("read_msh").warnings())

        # All supported primary families are checked. This triangle inverts
        # on the storage grid without coincident rounded points.
        inverted_tri = np.array([(0, 0, 0), (1.49, 0.51, 0), (1.51, 1.49, 0)]) + (2**23, 2**23, 0)
        cases = ((2, inverted_tri, (1, 2, 3)),
                 (1, points[:2], (1, 2)), (3, np.array(CUBE[:4]) + base, (1, 2, 3, 4)),
                 (5, np.array(CUBE) + base, tuple(range(1, 9))))
        for etype, xyz, conn in cases:
            with self.subTest(etype=etype):
                mesh = self.reader(msh22(xyz, [(etype, (1,), conn)])).geometry()
                self.assertIn(warning, mesh.attribValue("msh_summary"))
                if etype == 2:
                    self.assertEqual(len(set(tuple(p.position()) for p in mesh.points())), 3)
                    self.assertEqual(mesh.prim(0).normal()[2], -1.0)


class BuildTests(ScratchTest):
    def setUp(self):
        super().setUp()
        self.children = {n.sessionId() for n in hou.node("/obj").children()}
        self.libraries = set(hou.hda.loadedFiles())
        self.destination = self.work / "test.hdanc"
        self.destination.write_bytes(b"old library")

    def verify_cleanup(self, directory=None):
        self.assertEqual({n.sessionId() for n in hou.node("/obj").children()}, self.children)
        self.assertEqual(set(hou.hda.loadedFiles()), self.libraries)
        self.assertFalse(list((directory or self.work).glob(".*.tmp")))
        self.assertFalse((self.work / "backup").exists())
        self.assertFalse(hda_build._STAGED)

    def new_asset(self, suffix, destination=None, category="Sop"):
        return hda_build.new_asset(
            category, f"msh_review_{suffix}::1.0", "MSH Review Test",
            str(destination or self.destination))

    def test_hda_temporaries_are_unique_and_build_failure_cleans_up(self):
        stale = self.destination.with_suffix(".hdanc.tmp")
        stale.write_bytes(b"other process")
        with self.assertRaisesRegex(RuntimeError, "network failure"):
            with hda_build.build_transaction():
                stages = [self.new_asset(suffix) for suffix in ("a", "b")]
                paths = [Path(a.type().definition().libraryFilePath()) for a in stages]
                self.assertEqual(len(set(paths)), len(paths))
                for path in paths:
                    self.assertEqual(path.parent.parent, self.work)
                    self.assertTrue(path.is_file())
                    backup = path.parent / "backup"
                    backup.mkdir(exist_ok=True)
                    (backup / "temporary-copy").touch()
                raise RuntimeError("network failure")
        self.verify_cleanup()
        self.assertEqual(self.destination.read_bytes(), b"old library")
        self.assertEqual(stale.read_bytes(), b"other process")

    def test_hda_creation_save_and_commit_failure_leave_old_file(self):
        with patch.object(hou.OpNode, "createDigitalAsset", side_effect=RuntimeError("create failed")):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                self.new_asset("creation", category="GeoObject")
        self.verify_cleanup()

        asset = self.new_asset("save")
        with patch.object(hou.HDADefinition, "save", side_effect=RuntimeError("save failed")):
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                hda_build.finalize(asset)
        self.verify_cleanup()

        asset = self.new_asset("commit", category="GeoObject")
        with patch.object(hda_build.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                hda_build.commit_asset(asset)
        self.verify_cleanup()
        self.assertEqual(self.destination.read_bytes(), b"old library")

        asset = self.new_asset("success")
        self.assertEqual(Path(hda_build.finalize(asset)), self.destination)
        self.assertEqual(len(hou.hda.definitionsInFile(str(self.destination))), 1)
        self.verify_cleanup()

    def test_standalone_reader_build_cleans_up_network_failure(self):
        destination = self.work / "sop_MSH_Reader.3.0.hdanc"
        destination.write_bytes(b"previous asset")
        with patch.object(hda_build, "read_source", side_effect=RuntimeError("source missing")):
            with self.assertRaisesRegex(RuntimeError, "source missing"):
                build_all.msh_build.build(self.work)
        self.assertEqual(destination.read_bytes(), b"previous asset")
        self.verify_cleanup()

    def test_detach_matches_canonical_library_paths(self):
        canonical = self.work.resolve() / "asset.hdanc"
        canonical.touch()
        alias_dir = self.work / "linked"
        alias_dir.symlink_to(canonical.parent, target_is_directory=True)
        aliases = [alias_dir / canonical.name]
        if canonical.is_relative_to("/private/var") and Path("/var").is_symlink():
            aliases.append(Path("/var") / canonical.relative_to("/private/var"))
        unrelated = canonical.with_name("other.hdanc")
        for alias in aliases:
            for tmp, loaded in ((alias, canonical), (canonical, alias)):
                with self.subTest(tmp=tmp, loaded=loaded):
                    self.assertNotEqual(str(tmp), str(loaded))
                    node = Mock()
                    with (patch.object(hou.hda, "loadedFiles", return_value=(str(unrelated), str(loaded))),
                          patch.object(hou.hda, "uninstallFile") as uninstall):
                        hda_build._detach_asset(str(tmp), node)
                    node.destroy.assert_called_once_with()
                    uninstall.assert_called_once_with(str(loaded))

    def test_transaction_discards_all_assets_and_preserves_original_error(self):
        discard_asset = hda_build._discard_asset
        for body_fails in (True, False):
            with self.subTest(body_fails=body_fails):
                discarded = []
                original = RuntimeError("build failed")
                cleanup_error = OSError("discard failed")

                def fail_first_discard(staged):
                    discard_asset(staged)
                    discarded.append(staged[0])
                    if len(discarded) == 1:
                        raise cleanup_error

                with patch.object(hda_build, "_discard_asset", side_effect=fail_first_discard):
                    with self.assertRaises(RuntimeError if body_fails else OSError) as caught:
                        with hda_build.build_transaction():
                            assets = [self.new_asset(suffix) for suffix in ("one", "two", "three")]
                            staged_paths = [a.type().definition().libraryFilePath() for a in assets]
                            if body_fails:
                                raise original
                self.assertIs(caught.exception, original if body_fails else cleanup_error)
                self.assertIn("discard failed", " ".join(caught.exception.__notes__))
                self.assertCountEqual(discarded, staged_paths)
                self.verify_cleanup()

    def test_commit_preserves_mode_or_uses_umask(self):
        self.destination.chmod(0o664)
        mode_cases = ((self.destination, 0o077, 0o664),
                      (self.work / "new.hdanc", 0o022, 0o644),
                      (self.work / "new2.hdanc", 0o002, 0o664))
        for i, (destination, mask, expected) in enumerate(mode_cases):
            with self.subTest(destination=destination, mask=mask):
                asset = self.new_asset(f"mode_{i}", destination)
                Path(asset.type().definition().libraryFilePath()).chmod(0o600)
                old_mask = os.umask(mask)
                try:
                    hda_build.commit_asset(asset)
                finally:
                    os.umask(old_mask)
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), expected)
                self.verify_cleanup()

    @staticmethod
    def snapshot(directory):
        if not directory.exists():
            return None
        return {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
                for path in directory.iterdir()}

    @contextmanager
    def mocked_batch(self, *, fail_builder=None, have_otls=True, default_otls=False):
        with tempfile.TemporaryDirectory(dir=self.work) as root, ExitStack() as stack:
            out = Path(root) / "out"
            home = Path(root) / "home"
            otls = home / "otls"
            out.mkdir()
            home.mkdir()
            if have_otls:
                otls.mkdir()
            pipeline_count = len(build_all.PIPELINE_BUILDERS)
            for target, mode, count in ((out, 0o644, pipeline_count + 1),
                                        (otls, 0o664, pipeline_count)):
                if target.exists():
                    for i in range(count):
                        asset = target / f"asset{i}.hdanc"
                        asset.write_bytes(f"old {target.name} {i}".encode())
                        asset.chmod(mode)
                    (target / "unrelated.txt").write_bytes(b"leave unchanged")
            calls = []
            stack.enter_context(patch.object(build_all, "OUT", str(out)))
            stack.enter_context(patch.object(build_all, "OTLS_DIR", str(otls)))
            if default_otls:
                stack.enter_context(patch.object(hou, "homeHoudiniDirectory", return_value=str(home)))
            stack.enter_context(redirect_stdout(io.StringIO()))

            def build(i):
                def run(stage_dir):
                    stage = Path(stage_dir).resolve()
                    for target in (out, otls, *map(Path, hou.houdiniPath("HOUDINI_OTLSCAN_PATH"))):
                        self.assertFalse(stage.is_relative_to(target.resolve()))
                    if calls:
                        self.assertEqual(stage, calls[0][1])
                    calls.append((i, stage))
                    if i == fail_builder:
                        self.new_asset("failed", stage / "failed.hdanc", category="GeoObject")
                        raise RuntimeError(f"builder {i + 1} failed")
                    path = stage / f"asset{i}.hdanc"
                    path.write_bytes(f"new {i}".encode())
                    path.chmod(0o600)
                    return str(path)
                return run

            builders = (*build_all.PIPELINE_BUILDERS, build_all.write_msh_build)
            for i, builder in enumerate(builders):
                stack.enter_context(patch.object(builder, "build", side_effect=build(i)))
            yield out, otls, calls

    def check_batch_cleanup(self, calls, count):
        self.assertEqual([i for i, _ in calls], list(range(count)))
        self.assertFalse(calls[0][1].exists())
        self.verify_cleanup()

    def test_builder_failures_leave_both_destinations_unchanged(self):
        for failed in (1, 2, 3):
            with self.subTest(builder=failed + 1), self.mocked_batch(fail_builder=failed) as batch:
                out, otls, calls = batch
                before = (self.snapshot(out), self.snapshot(otls))
                with self.assertRaisesRegex(RuntimeError, f"builder {failed + 1} failed"):
                    build_all.main()
                self.assertEqual((self.snapshot(out), self.snapshot(otls)), before)
                self.check_batch_cleanup(calls, failed + 1)

    def test_stage_avoids_a_temp_directory_in_the_houdini_scan_path(self):
        with self.mocked_batch() as (out, otls, calls):
            scan_path = self.work / "scan"
            scan_path.mkdir()
            with (patch.object(hou, "houdiniPath", return_value=(str(scan_path),)),
                  patch.object(tempfile, "gettempdir", return_value=str(scan_path))):
                self.assertEqual(build_all.main(), 0)
            self.check_batch_cleanup(calls, 4)

    def test_batch_stages_every_replacement_and_preserves_modes(self):
        with self.mocked_batch() as (out, otls, calls):
            outputs = ((out, 4), (otls, 3))
            for target, _ in outputs:
                (target / "asset2.hdanc").unlink()
            replace = os.replace
            replacements = []

            def check_replace(source, destination):
                path = Path(destination)
                self.assertEqual(len(calls), 4)
                if not replacements:
                    for target, count in outputs:
                        self.assertEqual(len(list(target.glob(".*/*.new"))), count)
                        self.assertEqual(len(list(target.glob(".*/*.old"))), count - 1)
                self.assertEqual(Path(source).stat().st_dev, path.parent.stat().st_dev)
                replacements.append(path)
                replace(source, destination)

            old_mask = os.umask(0o027)
            try:
                with patch.object(build_all.os, "replace", side_effect=check_replace):
                    self.assertEqual(build_all.main(), 0)
            finally:
                os.umask(old_mask)
            self.assertEqual(replacements, [target / f"asset{i}.hdanc"
                                            for target, count in outputs for i in range(count)])
            for target, existing_mode in ((out, 0o644), (otls, 0o664)):
                count = 4 if target == out else 3
                after = self.snapshot(target)
                self.assertEqual(sorted(after), [*(f"asset{i}.hdanc" for i in range(count)), "unrelated.txt"])
                for i in range(count):
                    self.assertEqual(after[f"asset{i}.hdanc"],
                                     (f"new {i}".encode(), 0o640 if i == 2 else existing_mode))
            self.check_batch_cleanup(calls, 4)

    def test_rename_failures_roll_back_outputs_and_installs(self):
        for install in (False, True):
            for index in (1, 2):
                with self.subTest(install=install, rename=index + 1), self.mocked_batch() as batch:
                    out, otls, calls = batch
                    target = otls if install else out
                    if index == 2:
                        (target / "asset0.hdanc").unlink()
                    before = (self.snapshot(out), self.snapshot(otls))
                    replace = os.replace
                    failure_path = target / f"asset{index}.hdanc"

                    def interrupted_replace(source, destination):
                        if Path(destination) == failure_path and Path(source).suffix == ".new":
                            self.assertEqual((target / "asset0.hdanc").read_bytes(), b"new 0")
                            if install:
                                self.assertEqual((out / "asset2.hdanc").read_bytes(), b"new 2")
                            raise OSError("rename interrupted")
                        replace(source, destination)

                    with patch.object(build_all.os, "replace", side_effect=interrupted_replace):
                        with self.assertRaisesRegex(OSError, "rename interrupted"):
                            build_all.main()
                    self.assertEqual((self.snapshot(out), self.snapshot(otls)), before)
                    self.check_batch_cleanup(calls, 4)

    def test_copy_failure_leaves_both_destinations_unchanged(self):
        with self.mocked_batch() as (out, otls, calls):
            before = (self.snapshot(out), self.snapshot(otls))
            copyfile = build_all.shutil.copyfile

            def interrupted_copy(source, stage, **kwargs):
                stage = Path(stage)
                if stage.parent.parent == otls and stage.name == "asset1.hdanc.new":
                    stage.write_bytes(b"partial")
                    raise OSError("copy interrupted")
                return copyfile(source, stage, **kwargs)

            with (patch.object(build_all.shutil, "copyfile", side_effect=interrupted_copy),
                  patch.object(build_all.os, "replace", wraps=os.replace) as replace):
                with self.assertRaisesRegex(OSError, "copy interrupted"):
                    build_all.main()
                replace.assert_not_called()
            self.assertEqual((self.snapshot(out), self.snapshot(otls)), before)
            self.check_batch_cleanup(calls, 4)

    def test_build_all_absent_otls_can_be_required(self):
        for required, code, line in (("0", 0, "WARNING"), ("1", 1, "ERROR")):
            with self.subTest(required=required), self.mocked_batch(have_otls=False) as batch:
                out, otls, calls = batch
                before = self.snapshot(out)
                with (patch.dict(os.environ, POLYFEM_REQUIRE_INSTALL=required),
                      redirect_stderr(io.StringIO()) as stderr):
                    self.assertEqual(build_all.main(), code)
                self.assertIn(f"{line}: {otls} not found", stderr.getvalue())
                self.assertFalse(otls.exists())
                if required == "1":
                    self.assertEqual(self.snapshot(out), before)
                else:
                    self.assertEqual((out / "asset3.hdanc").read_bytes(), b"new 3")
                self.check_batch_cleanup(calls, 4)

    def test_build_all_creates_default_otls_under_existing_houdini_home(self):
        with self.mocked_batch(have_otls=False, default_otls=True) as (out, otls, calls):
            with patch.dict(os.environ, POLYFEM_REQUIRE_INSTALL="1"):
                self.assertEqual(build_all.main(), 0)
            for i in range(3):
                self.assertEqual((otls / f"asset{i}.hdanc").read_bytes(),
                                 (out / f"asset{i}.hdanc").read_bytes())
            self.assertTrue((out / "asset3.hdanc").is_file())
            self.assertFalse((otls / "asset3.hdanc").exists())
            self.check_batch_cleanup(calls, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
