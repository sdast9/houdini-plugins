"""MSH Reader 3.0 review regressions. Run with hython after src/build_all.py."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

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


def section(name, body):
    name = name.encode()
    return b"$" + name + b"\n" + body + b"\n$End" + name + b"\n"


def row(values):
    return " ".join(map(str, values)).encode()


def msh22(points, elements, tags=None, names=(), binary=False):
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
    data = section("MeshFormat", fmt)
    if names:
        name_rows = [f'{dim} {tag} "{name}"'.encode() for dim, tag, name in names]
        data += section("PhysicalNames", b"\n".join([row([len(names)]), *name_rows]))
    return data + section("Nodes", nodes) + section("Elements", elems)


def partitioned41(endian=None, size=8, tags=(10, 11, 12, 13), declared=3):
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
    partitions = pack("2S2i4S", 2, 1, 44, 2, 0, 0, 2, 1)
    for tag, phys in ((30, ()), (31, (-17,)), (40, ())):
        partitions += (pack("3iSi6dS", tag, 3, 10, 1, 1, *bbox, len(phys))
                       + (pack("i", *phys) if phys else b"") + pack("S", 0))

    node_payload = pack("4S3iS4S", 1, 4, min(tags), max(tags), 3, 40, 0, 4, *tags)
    xyz = np.array([*TRI, (0, 0, 1)], dtype=float)
    node_payload += pack("12d", *xyz.ravel())
    # The empty hex block must not leave a family slot. This also exercises
    # multiple blocks whose declared total is independent of the block count.
    elems = (pack("4S", 4, declared, 1, 3) + pack("3iS", 3, 40, 5, 0)
             + pack("3iS5S", 3, 40, 4, 1, 1, *tags)
             + pack("3iS4S", 2, 30, 2, 1, 2, *tags[:3])
             + pack("3iS4S", 2, 31, 2, 1, 3, tags[0], tags[1], tags[3]))
    return (section("MeshFormat", fmt) + section("Entities", entities)
            + section("PartitionedEntities", partitions) + section("Nodes", node_payload)
            + section("Elements", elems)), xyz


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

    def test_build_all_defers_install_and_uses_atomic_replace(self):
        otls = self.work / "otls"
        otls.mkdir()
        outputs = [self.work / f"asset{i}.hdanc" for i in range(3)]
        installed = [otls / path.name for path in outputs]
        for p in installed:
            p.write_bytes(b"old")
        calls = []

        def build(i):
            def run(out):
                self.assertEqual(Path(out), self.work)
                self.assertEqual([p.read_bytes() for p in installed], [b"old"] * 3)
                calls.append(i)
                outputs[i].write_bytes(f"new {i}".encode())
                return str(outputs[i])
            return run

        def fail(_):
            self.new_asset("last", category="GeoObject")
            raise RuntimeError("third build failed")

        builders = (build_all.msh_build, build_all.polyfem_build, build_all.readpvd_build)
        with ExitStack() as stack:
            stack.enter_context(patch.object(build_all, "OUT", str(self.work)))
            stack.enter_context(patch.object(build_all, "OTLS_DIR", str(otls)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for i, builder in enumerate(builders):
                stack.enter_context(patch.object(builder, "build", side_effect=build(i)))
            with patch.object(builders[-1], "build", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "third build failed"):
                    build_all.main()
                self.assertEqual(calls, [0, 1])
                self.assertEqual([p.read_bytes() for p in installed], [b"old"] * 3)
                self.verify_cleanup()

            replace = os.replace
            replacements = []

            def assert_atomic(source, dest):
                path = Path(dest)
                self.assertEqual(calls, [0, 1, 2])
                self.assertEqual(Path(source).parent, otls)
                self.assertEqual(path.read_bytes(), b"old")
                i = installed.index(path)
                self.assertEqual(Path(source).read_bytes(), f"new {i}".encode())
                replacements.append(i)
                replace(source, dest)

            calls.clear()
            with patch.object(build_all.os, "replace", side_effect=assert_atomic):
                self.assertEqual(build_all.main(), 0)
            self.assertEqual(replacements, [0, 1, 2])
            self.assertEqual([p.read_bytes() for p in installed], [b"new 0", b"new 1", b"new 2"])
            self.verify_cleanup(otls)

        def interrupted_copy(_source, stage):
            Path(stage).write_bytes(b"partial")
            raise OSError("copy interrupted")

        with patch.object(build_all.shutil, "copy2", side_effect=interrupted_copy):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                build_all._install_asset(str(outputs[0]), str(otls))
        self.assertEqual(installed[0].read_bytes(), b"new 0")
        self.verify_cleanup(otls)

    def test_build_all_absent_otls_can_be_required(self):
        otls = self.work / "missing"
        with ExitStack() as stack:
            stack.enter_context(patch.object(build_all, "OTLS_DIR", str(otls)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            for builder in (build_all.msh_build, build_all.polyfem_build, build_all.readpvd_build):
                stack.enter_context(patch.object(builder, "build", return_value="built.hdanc"))
            for required, code, line in (("0", 0, "WARNING"), ("1", 1, "ERROR")):
                with patch.dict(os.environ, POLYFEM_REQUIRE_INSTALL=required):
                    with redirect_stderr(io.StringIO()) as stderr:
                        self.assertEqual(build_all.main(), code)
                    self.assertIn(f"{line}: {otls} not found", stderr.getvalue())
            self.assertFalse(otls.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
