"""Native numpy VTK parser (VTU/VTK-HDF/VTM/PVD) for PolyFEM output.

Replaces meshio for the readPVD HDA: zero dependencies beyond numpy,
vectorized decode of ascii / inline-base64 / appended data arrays with
optional zlib compression, and vectorized linearization of higher-order
(Lagrange) tets into P1 sub-tets.
"""

import base64
import binascii
import hashlib
import html
import os
import re
import xml.etree.ElementTree as ET
import zlib
from collections import OrderedDict

import numpy as np

_VTK_TO_NP = {
    "Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
    "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64,
    "UInt64": np.uint64, "Float32": np.float32, "Float64": np.float64,
}

# VTK cell types -> (family, n_corners). Higher-order entries linearize to
# their corner family.
VTK_CELLS = {
    3: ("line", 2), 5: ("tri", 3), 9: ("quad", 4),
    10: ("tet", 4), 24: ("tet", 4), 71: ("tet", 4),
    12: ("hex", 8), 25: ("hex", 8), 29: ("hex", 8), 72: ("hex", 8),
    21: ("line", 2), 68: ("line", 2), 22: ("tri", 3), 69: ("tri", 3),
    23: ("quad", 4), 70: ("quad", 4), 1: ("point", 1),
}

# Higher-order Lagrange tet subdivision tables: each row indexes the element's
# node array in PolyFEM's OutGeometryData::build_high_order_vis_mesh ordering
# to build one P1 sub-tet with positive VTK orientation (same
# convention as a P1 tet [0, 1, 2, 3], so BUILD_PRIMS_VEX's [1, 2, 3, 0]
# produces positive Houdini tetrahedra for P1 through P4 alike).
TET10_SUBDIV = np.array([          # P2 (10 nodes) -> 8 sub-tets
    [0, 4, 6, 7], [4, 1, 5, 8], [5, 6, 9, 2], [8, 7, 3, 9],
    [6, 5, 9, 8], [7, 6, 9, 8], [8, 6, 4, 7], [8, 6, 5, 4]], dtype=np.int64)

TET20_SUBDIV = np.array([          # P3 (20 nodes) -> 27 sub-tets
    [0, 4, 9, 10], [3, 15, 13, 11], [1, 5, 12, 6], [2, 7, 14, 8],
    [10, 16, 18, 11], [10, 9, 18, 4], [10, 16, 4, 18], [4, 16, 19, 18],
    [4, 9, 18, 19], [8, 9, 19, 18], [4, 5, 19, 16], [5, 6, 19, 12],
    [5, 16, 12, 19], [15, 11, 17, 13], [16, 19, 18, 17], [13, 11, 17, 16],
    [13, 12, 16, 17], [19, 12, 17, 16], [19, 6, 17, 12], [16, 11, 17, 18],
    [11, 15, 17, 18], [15, 14, 17, 18], [8, 14, 18, 17], [8, 19, 17, 18],
    [7, 8, 17, 14], [7, 8, 19, 17], [6, 7, 19, 17]], dtype=np.int64)

TET35_SUBDIV = np.array([          # P4 (35 nodes) -> 64 sub-tets
    [2, 10, 9, 18], [18, 20, 31, 30], [10, 9, 18, 30], [20, 32, 29, 21],
    [21, 3, 15, 17], [10, 31, 30, 18], [20, 31, 30, 32], [20, 32, 30, 29],
    [17, 32, 21, 29], [15, 21, 17, 32], [17, 26, 15, 32], [17, 29, 26, 32],
    [26, 32, 14, 15], [25, 32, 14, 26], [25, 33, 14, 32], [25, 33, 13, 14],
    [25, 33, 4, 13], [33, 13, 12, 4], [0, 12, 13, 4], [22, 33, 12, 4],
    [22, 11, 12, 33], [23, 11, 22, 33], [31, 11, 23, 33], [31, 10, 23, 11],
    [30, 10, 23, 31], [30, 9, 23, 10], [30, 34, 31, 23], [31, 34, 33, 23],
    [32, 34, 33, 31], [31, 34, 30, 32], [29, 34, 32, 30], [29, 34, 26, 32],
    [26, 34, 25, 32], [25, 34, 33, 32], [25, 4, 33, 22], [25, 34, 22, 33],
    [22, 34, 23, 33], [4, 5, 22, 25], [34, 5, 25, 22], [34, 5, 22, 24],
    [34, 23, 24, 22], [34, 30, 24, 23], [8, 30, 23, 24], [9, 30, 23, 8],
    [24, 30, 28, 8], [28, 30, 24, 34], [29, 30, 28, 34], [28, 16, 29, 34],
    [29, 16, 26, 34], [26, 16, 29, 17], [26, 16, 27, 34], [27, 25, 26, 34],
    [25, 27, 5, 34], [27, 24, 5, 34], [28, 24, 27, 34], [27, 16, 28, 34],
    [27, 19, 28, 16], [27, 6, 28, 19], [27, 5, 24, 6], [27, 24, 28, 6],
    [6, 24, 28, 7], [7, 24, 28, 8], [7, 19, 28, 6], [6, 19, 1, 7]],
    dtype=np.int64)

# Higher-order Lagrange/TriQuadratic hexahedron subdivision tables: each row
# indexes the element's node array to build one positive-orientation P1
# sub-hex in VTK hex corner order [0..7] (matching BUILD_PRIMS_VEX's
# [0,1,3,2,4,5,7,6] Houdini hex order). These match PolyFEM's
# OutGeometryData::build_high_order_vis_mesh reference-node ordering: Q2
# includes the output permutation, and Q3 uses the q_nodes_3d ordering.
HEX27_SUBDIV = np.array([          # Q2 Lagrange (27 nodes) -> 8 sub-hexes
    [0, 8, 24, 11, 16, 22, 26, 20], [8, 1, 9, 24, 22, 17, 21, 26],
    [11, 24, 10, 3, 20, 26, 23, 18], [24, 9, 2, 10, 26, 21, 19, 23],
    [16, 22, 26, 20, 4, 12, 25, 15], [22, 17, 21, 26, 12, 5, 13, 25],
    [20, 26, 23, 18, 15, 25, 14, 7], [26, 21, 19, 23, 25, 13, 6, 14]],
    dtype=np.int64)

# VTK_TRIQUADRATIC_HEXAHEDRON (type 29) uses the same 27 nodes as Q2 Lagrange
# except vertical mid-edge nodes 18 and 19 are swapped.
HEX27_TRIQUAD_SUBDIV = np.where(
    HEX27_SUBDIV == 18, 19, np.where(HEX27_SUBDIV == 19, 18, HEX27_SUBDIV))

HEX64_SUBDIV = np.array([          # Q3 Lagrange (64 nodes) -> 27 sub-hexes
    [0, 8, 48, 15, 16, 40, 56, 35], [8, 9, 50, 48, 40, 42, 60, 56],
    [9, 1, 10, 50, 42, 19, 36, 60], [15, 48, 49, 14, 35, 56, 58, 33],
    [48, 50, 51, 49, 56, 60, 62, 58], [50, 10, 11, 51, 60, 36, 38, 62],
    [14, 49, 13, 3, 33, 58, 44, 23], [49, 51, 12, 13, 58, 62, 46, 44],
    [51, 11, 2, 12, 62, 38, 21, 46], [16, 40, 56, 35, 17, 41, 57, 34],
    [40, 42, 60, 56, 41, 43, 61, 57], [42, 19, 36, 60, 43, 18, 37, 61],
    [35, 56, 58, 33, 34, 57, 59, 32], [56, 60, 62, 58, 57, 61, 63, 59],
    [60, 36, 38, 62, 61, 37, 39, 63], [33, 58, 44, 23, 32, 59, 45, 22],
    [58, 62, 46, 44, 59, 63, 47, 45], [62, 38, 21, 46, 63, 39, 20, 47],
    [17, 41, 57, 34, 4, 24, 52, 31], [41, 43, 61, 57, 24, 25, 54, 52],
    [43, 18, 37, 61, 25, 5, 26, 54], [34, 57, 59, 32, 31, 52, 53, 30],
    [57, 61, 63, 59, 52, 54, 55, 53], [61, 37, 39, 63, 54, 26, 27, 55],
    [32, 59, 45, 22, 30, 53, 29, 7], [59, 63, 47, 45, 53, 55, 28, 29],
    [63, 39, 20, 47, 55, 27, 6, 28]],
    dtype=np.int64)


class VtuParseError(RuntimeError):
    pass


def _import_h5py():
    """Import h5py, using the HDA's embedded wheel when available."""
    try:
        import h5py
        return h5py
    except ImportError as original:
        loader = globals().get("_load_embedded_h5py")
        if loader is not None:
            try:
                return loader()
            except Exception as embedded_error:
                raise VtuParseError(
                    "This VTK-HDF result needs h5py. The bundled reader could "
                    f"not be loaded: {embedded_error}") from embedded_error
        raise VtuParseError(
            "This VTK-HDF result needs h5py in Houdini's Python environment."
        ) from original


# expat's feed() takes an int-sized length, so ElementTree.fromstring (one
# feed of the whole buffer) raises OverflowError on documents over ~2 GiB --
# e.g. the large inline-binary ("format=binary") VTUs PolyFEM writes for
# multi-million-element meshes. Feeding in sub-INT_MAX chunks avoids it.
_XML_FEED_CHUNK = 1 << 30  # 1 GiB


def _parse_xml(xml_bytes):
    if len(xml_bytes) < _XML_FEED_CHUNK:
        return ET.fromstring(xml_bytes)
    parser = ET.XMLParser()
    view = memoryview(xml_bytes)
    for offset in range(0, len(view), _XML_FEED_CHUNK):
        parser.feed(view[offset:offset + _XML_FEED_CHUNK])
    return parser.close()


def _localname(tag):
    return tag.rsplit("}", 1)[-1]


_ROOT_ATTRS_RE = re.compile(rb"<VTKFile\b([^>]*)>")
_XML_ATTR_RE = re.compile(rb'''([\w:.-]+)\s*=\s*(["'])(.*?)\2''', re.DOTALL)
_XML_TAG_RE = re.compile(rb'''(?:"[^"]*"|'[^']*'|[^"'>])*?>''')
_ARRAY_METADATA_RE = re.compile(
    rb"<!--.*?-->|<InformationKey\b[^>]*(?:/>|(?<!/)>.*?</InformationKey>)", re.DOTALL)
_WHITESPACE = b" \t\r\n"


def _xml_attrs(tag):
    return {name: value for name, _, value in _XML_ATTR_RE.findall(tag)}


def _raw_binary_header(data):
    """(header_size, byteorder) when inline binary payloads are uncompressed.

    Uncompressed inline-binary VTK encodes the byte-count header and the data
    as one base64 stream, so the payload length follows from its first few
    characters. Returns None when the layout is not known to be predictable.
    """
    match = _ROOT_ATTRS_RE.search(data, 0, 4096)
    if match is None:
        return None
    attrs = _xml_attrs(match.group(1))
    if attrs.get(b"compressor"):
        return None
    header = {b"UInt32": 4, b"UInt64": 8}.get(
        attrs.get(b"header_type", b"UInt32"))
    if header is None:
        return None
    order = ("big" if attrs.get(b"byte_order") == b"BigEndian" else "little")
    return header, order


def _predicted_close(data, begin, raw_header):
    """Offset of this payload's </DataArray> from its header, or None.

    Saves a byte search through the whole payload (about a second per 2 GB
    frame). Any mismatch -- wrapped base64, ascii, compressed data -- returns
    None and the caller falls back to searching.
    """
    header_size, order = raw_header
    start = begin
    while data[start:start + 1] and data[start:start + 1] in _WHITESPACE:
        start += 1
    chars = 4 * -(-header_size // 3)
    try:
        head = binascii.a2b_base64(data[start:start + chars])
    except binascii.Error:
        return None
    if len(head) < header_size:
        return None
    nbytes = int.from_bytes(head[:header_size], order)
    close = start + 4 * -(-(header_size + nbytes) // 3)
    while data[close:close + 1] and data[close:close + 1] in _WHITESPACE:
        close += 1
    return close if data[close:close + 12] == b"</DataArray>" else None


def _strip_inline_payloads(data):
    """Remove inline <DataArray> payloads, returning (skeleton, ranges).

    An inline-binary/ascii VTU keeps each array's (possibly huge) base64/ascii
    text inside its <DataArray> element, so parsing the whole document into an
    ElementTree needs gigabytes of RAM. This scans the raw bytes, blanks out
    each payload (leaving a small tag skeleton to parse) and records the
    payload's byte range so it can be decoded on demand from the original
    buffer. Base64/ascii payloads never contain '<', so the tag scan is safe.
    Uncompressed binary payloads are skipped by their encoded length instead
    of scanned. `ranges[i]` is a byte span, or a list of spans when metadata
    interrupts the text. Self-closing arrays have no inline payload.
    """
    close_tag = b"</DataArray>"
    raw_header = _raw_binary_header(data)
    out = bytearray()
    ranges = []
    pos = 0
    while True:
        start = data.find(b"<DataArray", pos)
        comment = data.find(b"<!--", pos)
        if comment >= 0 and (start < 0 or comment < start):
            end_comment = data.find(b"-->", comment + 4)
            if end_comment < 0:
                out += data[pos:]
                break
            out += data[pos:end_comment + 3]
            pos = end_comment + 3
            continue
        if start < 0:
            out += data[pos:]
            break
        tag = _XML_TAG_RE.match(data, start)
        if tag is None:
            out += data[pos:]
            break
        tag_end = tag.end() - 1
        if data[tag_end - 1:tag_end] == b"/":            # <DataArray .../>
            out += data[pos:tag_end + 1]
            ranges.append(None)
            pos = tag_end + 1
            continue
        predicted = (_predicted_close(data, tag_end + 1, raw_header)
                     if raw_header is not None and _xml_attrs(
                         data[start:tag_end]).get(b"format") == b"binary"
                     else None)
        close = predicted
        first_tag = -1
        if close is None:
            first_tag = data.find(b"<", tag_end + 1)
            close = data.find(close_tag, first_tag) if first_tag >= 0 else -1
        if close < 0:
            out += data[pos:]
            break
        out += data[pos:tag_end + 1]                      # opening tag
        payload_start = tag_end + 1
        spans = []
        if 0 <= first_tag < close:
            for metadata in _ARRAY_METADATA_RE.finditer(data, first_tag, close):
                spans.append((payload_start, metadata.start()))
                payload_start = metadata.end()
        tail = (payload_start, close)
        if spans:
            spans.append(tail)
            ranges.append(spans)
        else:
            ranges.append(tail)
        out += close_tag                                  # closing tag
        pos = close + len(close_tag)
    return bytes(out), ranges


# All caches are keyed by (path, mtime, size) so external file changes
# (e.g. a still-running simulation appending frames) invalidate naturally.
_PVD_CACHE = {}
_VTM_CACHE = OrderedDict()
_FIELD_INFO_CACHE = {}
_VTU_CACHE = OrderedDict()
# Parsed frames are large; hold enough for the stages that re-read the same
# frame in one cook (topology + frame + multi-block + reference comparison),
# but never more than _VTU_CACHE_BYTES beyond the newest entry: a 2.4M-point
# frame parses to ~2 GB, and the cooked-frame cache (cache1) already keeps
# every visited frame.
_VTU_CACHE_SIZE = 6
_VTU_CACHE_BYTES = 4 << 30


def _file_key(path):
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def clear_caches():
    _PVD_CACHE.clear()
    _VTM_CACHE.clear()
    _FIELD_INFO_CACHE.clear()
    _VTU_CACHE.clear()


class _Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        # appended section (raw or base64): everything after the first '_'
        # following <AppendedData ...>
        self.appended = None
        self.appended_encoding = "raw"
        idx = self.data.find(b"<AppendedData")
        if idx >= 0:
            tag_end = self.data.find(b">", idx)
            if tag_end >= 0:
                app_attrs = _xml_attrs(self.data[idx:tag_end])
                enc = app_attrs.get(b"encoding", b"raw").decode(
                    "ascii", "ignore").lower()
                if enc:
                    self.appended_encoding = enc
            start = self.data.find(b"_", max(idx, tag_end))
            end = self.data.rfind(b"</AppendedData>")
            if tag_end < 0 or start < 0 or end <= start:
                raise VtuParseError("Incomplete AppendedData section")
            self.appended = self.data[start + 1:end]
            # ElementTree chokes on raw bytes inside AppendedData: replace
            # the payload with nothing for the XML parse.
            xml_bytes = self.data[:start + 1] + self.data[end:]
        else:
            xml_bytes = self.data
        # Strip inline array payloads before parsing so a multi-GB inline
        # VTU doesn't build a multi-GB ElementTree; decode them on demand.
        skeleton, payload_ranges = _strip_inline_payloads(xml_bytes)
        self._payload_source = xml_bytes
        self.root = _parse_xml(skeleton)
        self._payload = {}
        data_arrays = [elem for elem in self.root.iter()
                       if _localname(elem.tag) == "DataArray"]
        for elem, span in zip(data_arrays, payload_ranges):
            if span is not None:
                self._payload[id(elem)] = span
        self.byte_order = self.root.get("byte_order", "LittleEndian")
        self._endian = ">" if self.byte_order == "BigEndian" else "<"
        self.header_dtype = np.dtype(
            _VTK_TO_NP[self.root.get("header_type", "UInt32")]
        ).newbyteorder(self._endian)
        comp = self.root.get("compressor", "").strip()
        if not comp:
            self.compressed = False
        elif "ZLib" in comp:
            self.compressed = True
        else:
            raise VtuParseError(
                f"Unsupported VTK compressor '{comp}' "
                "(only vtkZLibDataCompressor is supported)")

    @staticmethod
    def _decode_inline_base64(payload):
        """Decode single-stream or VTK two-stream (header_b64 + data_b64)."""
        compact = b"".join(payload.split())
        if not compact:
            return b""
        first_eq = compact.find(b"=")
        if 0 <= first_eq < len(compact) - 2:
            split_at = first_eq
            while split_at < len(compact) and compact[split_at:split_at + 1] == b"=":
                split_at += 1
            return (binascii.a2b_base64(compact[:split_at])
                    + binascii.a2b_base64(compact[split_at:]))
        return binascii.a2b_base64(compact)

    def _decode_appended_base64(self, buf):
        """Decode one array from <AppendedData encoding='base64'>."""
        hsize = self.header_dtype.itemsize
        if not self.compressed:
            h_chars = 4 * ((hsize + 2) // 3)
            head = binascii.a2b_base64(buf[:h_chars])
            nbytes = int(np.frombuffer(head, self.header_dtype, count=1)[0])
            # A combined header and one-byte payload can also end in '='.
            # Only a header that decodes to exactly hsize is standalone.
            if len(head) == hsize and hsize % 3:
                p_chars = 4 * ((nbytes + 2) // 3)
                raw = head + binascii.a2b_base64(buf[h_chars:h_chars + p_chars])
            else:
                total_chars = 4 * ((hsize + nbytes + 2) // 3)
                raw = binascii.a2b_base64(buf[:total_chars])
            return self._decode_block(raw)
        hdr3_chars = 4 * hsize  # 3 * hsize is always divisible by 3
        hdr3 = np.frombuffer(
            binascii.a2b_base64(buf[:hdr3_chars]), self.header_dtype, count=3)
        nblocks = int(hdr3[0])
        total_hbytes = (3 + nblocks) * hsize
        h_chars = 4 * ((total_hbytes + 2) // 3)
        full_hdr = binascii.a2b_base64(buf[:h_chars])
        sizes = np.frombuffer(
            full_hdr, self.header_dtype, count=nblocks, offset=3 * hsize)
        total_comp = sum(map(int, sizes))
        if len(full_hdr) == total_hbytes and total_hbytes % 3:
            c_chars = 4 * ((total_comp + 2) // 3)
            raw = full_hdr + binascii.a2b_base64(buf[h_chars:h_chars + c_chars])
        else:
            total_chars = 4 * ((total_hbytes + total_comp + 2) // 3)
            raw = binascii.a2b_base64(buf[:total_chars])
        return self._decode_block(raw)

    def _decode_block(self, raw):
        """Decode (possibly multi-block zlib) payload -> bytes."""
        if not self.compressed:
            hsize = self.header_dtype.itemsize
            n = int(np.frombuffer(raw, self.header_dtype, count=1)[0])
            if len(raw) < hsize + n:
                raise VtuParseError("Truncated VTK binary payload")
            return raw[hsize:hsize + n]
        hdr = np.frombuffer(raw, self.header_dtype, count=3)
        nblocks = int(hdr[0])
        sizes = np.frombuffer(
            raw, self.header_dtype, count=nblocks,
            offset=3 * self.header_dtype.itemsize)
        offset = (3 + nblocks) * self.header_dtype.itemsize
        out = []
        for index, size in enumerate(sizes):
            end = offset + int(size)
            if end > len(raw):
                raise VtuParseError("Truncated VTK compressed payload")
            block = zlib.decompress(raw[offset:end])
            expected = int(hdr[2] or hdr[1]) if index == nblocks - 1 else int(hdr[1])
            if len(block) != expected:
                raise VtuParseError("VTK compressed block has the wrong length")
            out.append(block)
            offset = end
        return b"".join(out)

    def read_data_array(self, elem):
        try:
            return self._read_data_array(elem)
        except (KeyError, ValueError, OverflowError, zlib.error) as error:
            raise VtuParseError(
                f"Invalid VTK DataArray {elem.get('Name', '(unnamed)')}: {error}") from error

    def _read_data_array(self, elem):
        native_dtype = np.dtype(_VTK_TO_NP[elem.get("type")])
        dtype = native_dtype.newbyteorder(self._endian)
        ncomp = int(elem.get("NumberOfComponents", "1"))
        if ncomp < 1:
            raise VtuParseError("VTK DataArray requires at least one component")
        fmt = elem.get("format", "ascii")
        if fmt == "appended":
            if self.appended is None:
                raise VtuParseError("appended format without AppendedData")
            offset = int(elem.get("offset", "0"))
            if offset < 0 or offset >= len(self.appended):
                raise VtuParseError("VTK appended offset lies outside the payload")
            payload = memoryview(self.appended)[offset:]
            if self.appended_encoding == "base64":
                raw = self._decode_appended_base64(payload)
            elif self.appended_encoding == "raw":
                raw = bytes(self._decode_block(payload))
            else:
                raise VtuParseError(
                    f"Unsupported AppendedData encoding '{self.appended_encoding}'")
            arr = np.frombuffer(raw, dtype=dtype).astype(
                native_dtype, copy=False)
        else:
            span = self._payload.get(id(elem))
            if isinstance(span, list):
                payload = b"".join(self._payload_source[start:end] for start, end in span)
            elif span is not None:
                payload = self._payload_source[span[0]:span[1]]
            else:
                payload = b""
            if fmt == "ascii":
                arr = np.array(payload.split(), dtype=native_dtype)
            elif fmt == "binary":
                raw = self._decode_block(self._decode_inline_base64(payload))
                arr = np.frombuffer(raw, dtype=dtype).astype(
                    native_dtype, copy=False)
            else:
                raise VtuParseError(f"Unknown DataArray format {fmt}")
        if ncomp > 1:
            arr = arr.reshape(-1, ncomp)
        tuples = elem.get("NumberOfTuples")
        if tuples is not None and len(arr) != int(tuples):
            raise VtuParseError(f"VTK DataArray {elem.get('Name', '')} has the wrong tuple count")
        return arr


def _mesh_from_arrays(points, connectivity, offsets, types, point_data,
                      raw_cell_data, offsets_include_zero=False):
    """Build the common readPVD mesh contract from VTK cell arrays."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    connectivity = np.asarray(connectivity, dtype=np.int64).ravel()
    offsets = np.asarray(offsets, dtype=np.int64).ravel()
    types = np.asarray(types, dtype=np.int64).ravel()
    if offsets_include_zero:
        if len(offsets) != len(types) + 1 or (len(offsets) and offsets[0] != 0):
            raise VtuParseError(
                "VTK-HDF Offsets must start at zero and contain one entry "
                "more than Types")
        starts = offsets[:-1]
        sizes = np.diff(offsets)
    else:
        if len(offsets) != len(types):
            raise VtuParseError(
                "VTU offsets and cell types have different lengths")
        if len(offsets) == 0:
            starts = np.empty(0, dtype=np.int64)
            sizes = np.empty(0, dtype=np.int64)
        else:
            starts = np.concatenate(([0], offsets[:-1]))
            sizes = offsets - starts

    if np.any(sizes < 0) or (
            (int(offsets[-1]) if len(offsets) else 0) != len(connectivity)):
        raise VtuParseError("VTK cell offsets do not match connectivity")
    if len(connectivity) and (connectivity.min() < 0 or
                              connectivity.max() >= len(points)):
        raise VtuParseError("VTK cell connectivity references a missing point")
    cells = {}
    cell_sources = {}
    for ctype in np.unique(types):
        info = VTK_CELLS.get(int(ctype))
        if info is None:
            continue  # skip exotic cells rather than fail
        family, n_corners = info
        type_mask = types == ctype
        for n_nodes in np.unique(sizes[type_mask]):
            n_nodes = int(n_nodes)
            if n_nodes < n_corners:
                raise VtuParseError(f"VTK cell type {ctype} has too few connectivity ids")
            mask = type_mask & (sizes == n_nodes)
            source_indices = np.flatnonzero(mask)
            c_starts = starts[mask]
            conn = connectivity[
                c_starts[:, None] + np.arange(n_nodes)[None, :]]
            if family == "tet":
                subdiv = (TET10_SUBDIV if n_nodes == 10 else
                          TET20_SUBDIV if n_nodes == 20 else
                          TET35_SUBDIV if n_nodes == 35 else None)
            elif family == "hex":
                subdiv = (HEX27_TRIQUAD_SUBDIV if int(ctype) == 29 and n_nodes == 27 else
                          HEX27_SUBDIV if n_nodes == 27 else
                          HEX64_SUBDIV if n_nodes == 64 else None)
            else:
                subdiv = None
            if subdiv is not None:
                # vectorized higher-order tet/hex -> P1 sub-elements
                conn = conn[:, subdiv].reshape(-1, n_corners)
                source_indices = np.repeat(source_indices, len(subdiv))
            elif family == "tet" and n_nodes > 4:
                conn = conn[:, :4]  # other high orders: corner tet fallback
            else:
                conn = conn[:, :n_corners]
            prev = cells.get(family)
            cells[family] = conn if prev is None else np.vstack([prev, conn])
            prev_sources = cell_sources.get(family)
            cell_sources[family] = (
                source_indices if prev_sources is None
                else np.concatenate((prev_sources, source_indices)))

    cell_data = {}
    for name, raw in raw_cell_data.items():
        raw = np.asarray(raw)
        cell_data[name] = {
            family: raw[source_indices]
            for family, source_indices in cell_sources.items()
        }

    topo = hashlib.blake2b(digest_size=16)
    for array in (types, starts, sizes, connectivity):
        topo.update(len(array).to_bytes(8, "little"))
        topo.update(np.ascontiguousarray(array, dtype="<i8"))
    topo.update(len(points).to_bytes(8, "little"))
    topo_key = topo.hexdigest()

    # The file's own cells as well: every node of every cell (cells above
    # keep corners or sub-tets only), and the cell each row of cells[family]
    # came from. Read PVD's exports use them for element and node identity.
    return {"points": points, "cells": cells, "point_data": point_data,
            "cell_data": cell_data, "topo_key": topo_key,
            "cell_types": types, "cell_starts": starts,
            "cell_sizes": sizes, "cell_connectivity": connectivity,
            "cell_sources": cell_sources}


def read_vtu(path):
    """Parse an UnstructuredGrid .vtu into the readPVD mesh contract."""
    try:
        reader = _Reader(path)
    except (ET.ParseError, KeyError) as error:
        raise VtuParseError(f"Invalid VTU file {path}: {error}") from error
    pieces = reader.root.findall(".//{*}Piece")
    if not pieces:
        raise VtuParseError(f"No <Piece> in {path}")

    def fields(piece, section):
        return {da.get("Name"): reader.read_data_array(da)
                for da in piece.findall(f"{{*}}{section}/{{*}}DataArray")
                if da.get("Name")}

    parts = []
    for piece in pieces:
        points_elem = piece.find("{*}Points/{*}DataArray")
        if points_elem is None:
            raise VtuParseError("VTU Piece is missing its Points DataArray")
        points = reader.read_data_array(points_elem)
        if points.size % 3 or points.ndim > 1 and points.shape[1] != 3:
            raise VtuParseError("VTU Points must contain XYZ tuples")
        points = points.reshape(-1, 3)
        arrays = fields(piece, "Cells")
        try:
            conn, offsets, types = (arrays[name].ravel() for name in
                                    ("connectivity", "offsets", "types"))
        except KeyError as error:
            raise VtuParseError(f"VTU Piece missing Cells array {error}") from error
        for label, count in (("Points", len(points)), ("Cells", len(types))):
            if int(piece.get(f"NumberOf{label}", count)) != count:
                raise VtuParseError(f"VTU Piece has the wrong number of {label}")
        if (len(offsets) != len(types)
                or (int(offsets[-1]) if len(offsets) else 0) != len(conn)
                or len(conn) and (conn.min() < 0 or conn.max() >= len(points))):
            raise VtuParseError("VTU Piece has invalid cell offsets or connectivity")
        point_data, cell_data = fields(piece, "PointData"), fields(piece, "CellData")
        if any(len(array) != len(types) for array in cell_data.values()):
            raise VtuParseError("VTU CellData length does not match its Piece")
        parts.append((points, conn, offsets, types, point_data, cell_data))
    if len(parts) == 1:
        return _mesh_from_arrays(*parts[0])

    points, conns, offsets, types, point_data, cell_data = zip(*parts)
    point_base = np.cumsum([0] + [len(p) for p in points[:-1]], dtype=np.int64)
    conn_base = np.cumsum([0] + [len(c) for c in conns[:-1]], dtype=np.int64)
    return _mesh_from_arrays(
        np.concatenate(points),
        np.concatenate([c.astype(np.int64, copy=False) + b for c, b in zip(conns, point_base)]),
        np.concatenate([o.astype(np.int64, copy=False) + b for o, b in zip(offsets, conn_base)]),
        np.concatenate(types), _merge_piece_fields(point_data, points),
        _merge_piece_fields(cell_data, types))


def _merge_piece_fields(fields, owners):
    """Align field tuples with concatenated pieces, marking absent data NaN."""
    merged = {}
    for name in dict.fromkeys(name for part in fields for name in part):
        sample = next(part[name] for part in fields if name in part)
        shape = sample.shape[1:]
        arrays = []
        for part, owner in zip(fields, owners):
            array = part.get(name)
            if array is None:
                array = np.full((len(owner),) + shape, np.nan)
            elif array.shape[1:] != shape or len(array) > len(owner):
                raise VtuParseError(f"Incompatible VTU Piece field {name!r}")
            elif len(array) < len(owner):
                # PolyFEM's averaged fields can omit obstacle points.
                padding = np.zeros((len(owner) - len(array),) + shape,
                                   dtype=array.dtype)
                array = np.concatenate((array, padding))
            arrays.append(array)
        merged[name] = np.concatenate(arrays)
    return merged


def _hdf_datasets(group):
    """Read every dataset below an HDF5 group, preserving slash names."""
    h5py = _import_h5py()
    arrays = {}

    def collect(name, value):
        if isinstance(value, h5py.Dataset):
            arrays[name] = value[...]

    group.visititems(collect)
    return arrays


def read_hdf(path):
    """Parse PolyFEM's VTK-HDF UnstructuredGrid output."""
    h5py = _import_h5py()
    try:
        with h5py.File(path, "r") as handle:
            if "VTKHDF" not in handle:
                raise VtuParseError(f"No /VTKHDF group in {path}")
            root = handle["VTKHDF"]
            vtk_type = root.attrs.get("Type", "")
            if isinstance(vtk_type, bytes):
                vtk_type = vtk_type.decode(errors="replace")
            if str(vtk_type).rstrip("\x00") != "UnstructuredGrid":
                raise VtuParseError(
                    f"Unsupported VTK-HDF Type {vtk_type!r} in {path}")
            required = ("Points", "Connectivity", "Offsets", "Types")
            missing = [name for name in required if name not in root]
            if missing:
                raise VtuParseError(
                    f"Missing VTK-HDF datasets in {path}: {', '.join(missing)}")
            point_data = (_hdf_datasets(root["PointData"])
                          if "PointData" in root else {})
            cell_data = (_hdf_datasets(root["CellData"])
                         if "CellData" in root else {})
            points, connectivity, offsets, types = (root[name][...] for name in required)
            counts = ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")
            if all(name in root for name in counts):
                n_points, n_cells, n_conn = (np.asarray(root[name], dtype=np.int64).ravel()
                                              for name in counts)
                if (len(n_points) != len(n_cells) or len(n_cells) != len(n_conn)
                        or any(np.any(n < 0) for n in (n_points, n_cells, n_conn))
                        or (n_points.sum(), n_cells.sum(), n_conn.sum(),
                            n_cells.sum() + len(n_cells)) !=
                           (len(points), len(types), len(connectivity), len(offsets))):
                    raise VtuParseError("Inconsistent VTK-HDF partition counts")
                if len(n_cells) != 1:
                    connectivity, offsets = _merge_hdf_partitions(
                        connectivity, offsets, n_points, n_cells, n_conn)
            return _mesh_from_arrays(points, connectivity, offsets, types,
                                     point_data, cell_data, offsets_include_zero=True)
    except OSError as error:
        raise VtuParseError(f"Could not read VTK-HDF file {path}: {error}") \
            from error


def _merge_hdf_partitions(connectivity, offsets, n_points, n_cells, n_conn):
    """Rebase partition-local offsets and point indices to the merged mesh."""
    connectivity = np.asarray(connectivity, dtype=np.int64).copy()
    merged_offsets = np.empty(int(n_cells.sum()) + 1, dtype=np.int64)
    merged_offsets[0] = 0
    point_base = cell_base = conn_base = offset_base = 0
    for points, cells, ids in zip(n_points, n_cells, n_conn):
        part = offsets[offset_base:offset_base + cells + 1]
        conn = connectivity[conn_base:conn_base + ids]
        if (part[0] != 0 or part[-1] != ids or np.any(np.diff(part.astype(np.int64)) < 0)
                or len(conn) and (conn.min() < 0 or conn.max() >= points)):
            raise VtuParseError("Invalid VTK-HDF partition offsets or connectivity")
        merged_offsets[cell_base + 1:cell_base + cells + 1] = part[1:] + conn_base
        conn += point_base
        point_base += points
        cell_base += cells
        conn_base += ids
        offset_base += cells + 1
    return connectivity, merged_offsets


def _mesh_nbytes(mesh):
    """Approximate memory held by one parsed mesh dict."""
    total = getattr(mesh.get("points"), "nbytes", 0)
    for name in ("cell_types", "cell_starts", "cell_sizes",
                 "cell_connectivity"):
        total += getattr(mesh.get(name), "nbytes", 0)
    for group in ("cells", "point_data", "cell_sources"):
        total += sum(getattr(v, "nbytes", 0) for v in mesh.get(group, {}).values())
    for by_family in mesh.get("cell_data", {}).values():
        total += sum(getattr(v, "nbytes", 0) for v in by_family.values())
    return total


def _remember(key, mesh):
    _VTU_CACHE[key] = mesh
    while len(_VTU_CACHE) > 1 and (
            len(_VTU_CACHE) > _VTU_CACHE_SIZE
            or sum(_mesh_nbytes(m) for m in _VTU_CACHE.values())
            > _VTU_CACHE_BYTES):
        _VTU_CACHE.popitem(last=False)


def read_vtu_cached(path):
    """read_vtu with a small LRU keyed on file identity.

    The returned dict is shared between callers and must not be mutated.
    """
    key = _file_key(path)
    mesh = _VTU_CACHE.get(key)
    if mesh is None:
        mesh = read_vtu(path)
        _remember(key, mesh)
    else:
        _VTU_CACHE.move_to_end(key)
    return mesh


def _is_hdf(path):
    return os.path.splitext(path)[1].lower() in (".hdf", ".h5", ".hdf5", ".vtkhdf")


def read_mesh_cached(path):
    """Read either VTK XML or VTK-HDF through the shared small LRU."""
    key = _file_key(path)
    mesh = _VTU_CACHE.get(key)
    if mesh is None:
        mesh = read_hdf(path) if _is_hdf(path) else read_vtu(path)
        _remember(key, mesh)
    else:
        _VTU_CACHE.move_to_end(key)
    return mesh


def _field_info_from_mesh(mesh):
    """Field-info dict from an already-parsed mesh (no file read)."""
    info = {"point_data": {}, "cell_data": {}}
    for name, data in mesh["point_data"].items():
        array = np.asarray(data)
        info["point_data"][name] = 1 if array.ndim == 1 else int(array.shape[1])
    for name, by_family in mesh.get("cell_data", {}).items():
        sample = next(iter(by_family.values()), None)
        if sample is None:
            continue
        sample = np.asarray(sample)
        info["cell_data"][name] = 1 if sample.ndim == 1 else int(sample.shape[1])
    return info


_VTU_SCAN_CHUNK = 64 * 1024


def _vtu_tags(handle):
    """Walk XML headers with bounded reads, skipping large binary arrays."""
    raw_header = _raw_binary_header(handle.read(4096))
    handle.seek(0)
    buffer = b""
    while True:
        lt = buffer.find(b"<")
        if lt < 0:
            buffer = handle.read(_VTU_SCAN_CHUNK)
            if not buffer:
                return
            continue
        buffer = buffer[lt:]
        if len(buffer) < 4:
            buffer += handle.read(4 - len(buffer))
        if buffer.startswith(b"<!--"):
            end = buffer.find(b"-->")
            while end < 0:
                chunk = handle.read(_VTU_SCAN_CHUNK)
                if not chunk:
                    return
                buffer = buffer[-2:] + chunk
                end = buffer.find(b"-->")
            buffer = buffer[end + 3:]
            continue
        match = _XML_TAG_RE.match(buffer)
        if match is None:
            buffer += handle.read(_VTU_SCAN_CHUNK)
            match = _XML_TAG_RE.match(buffer)
            if match is None:
                return
        head = buffer[1:match.end() - 1]
        buffer = buffer[match.end():]
        yield head
        if (raw_header is None or not head.startswith(b"DataArray")
                or head.endswith(b"/") or _xml_attrs(head).get(b"format") != b"binary"):
            continue
        # Probe the length at the start, then verify the predicted closing tag
        # before seeking past the payload. Wrapped or partial arrays fall back
        # to chunked scanning.
        start = handle.tell() - len(buffer)
        handle.seek(start)
        prefix = handle.read(256)
        leading = len(prefix) - len(prefix.lstrip(_WHITESPACE))
        header_size, byteorder = raw_header
        try:
            raw = binascii.a2b_base64(prefix[leading:leading + 4 * ((header_size + 2) // 3)])
        except binascii.Error:
            raw = b""
        buffer = b""
        if len(raw) >= header_size:
            length = header_size + int.from_bytes(raw[:header_size], byteorder)
            end = start + leading + 4 * ((length + 2) // 3)
            if end < 2**63:
                handle.seek(end)
                buffer = handle.read(256)
                if buffer.lstrip(_WHITESPACE).startswith(b"</DataArray>"):
                    continue
        handle.seek(start)
        buffer = b""


def _scan_vtu_field_info(handle):
    info = {"point_data": {}, "cell_data": {}}
    section = None
    in_array = False
    for head in _vtu_tags(handle):
        parts = head.rstrip(b"/").split(None, 1)
        tag = parts[0].rsplit(b":", 1)[-1] if parts else b""
        if tag == b"AppendedData":
            break
        if tag == b"/DataArray":
            in_array = False
        elif in_array:
            continue
        elif tag in (b"PointData", b"CellData") and not head.endswith(b"/"):
            section = "point_data" if tag == b"PointData" else "cell_data"
        elif tag in (b"/PointData", b"/CellData"):
            section = None
        elif tag == b"DataArray":
            in_array = not head.endswith(b"/")
            if section is not None:
                attrs = _xml_attrs(head)
                name = attrs.get(b"Name")
                if name:
                    info[section][html.unescape(name.decode())] = int(
                        attrs.get(b"NumberOfComponents", b"1"))
    return info


def read_vtu_field_info(path):
    """Field names -> component counts without decoding any array data.

    Reuse the parsed mesh when warm. Otherwise read bounded XML header chunks
    without mapping the file, even when a solver may truncate it concurrently.
    """
    key = _file_key(path)
    cached = _FIELD_INFO_CACHE.get(key)
    if cached is not None:
        return cached
    parsed = _VTU_CACHE.get(key)
    if parsed is not None:
        info = _field_info_from_mesh(parsed)
        _FIELD_INFO_CACHE[key] = info
        return info
    with open(path, "rb") as f:
        info = _scan_vtu_field_info(f)
    if len(_FIELD_INFO_CACHE) > 4096:
        _FIELD_INFO_CACHE.clear()
    _FIELD_INFO_CACHE[key] = info
    return info


def read_hdf_field_info(path):
    """Field names and component counts without loading HDF5 array values."""
    key = _file_key(path)
    cached = _FIELD_INFO_CACHE.get(key)
    if cached is not None:
        return cached
    parsed = _VTU_CACHE.get(key)
    if parsed is not None:
        info = _field_info_from_mesh(parsed)
        _FIELD_INFO_CACHE[key] = info
        return info
    h5py = _import_h5py()
    info = {"point_data": {}, "cell_data": {}}
    with h5py.File(path, "r") as handle:
        if "VTKHDF" not in handle:
            raise VtuParseError(f"No /VTKHDF group in {path}")
        root = handle["VTKHDF"]
        for group_name, out_key in (("PointData", "point_data"),
                                    ("CellData", "cell_data")):
            if group_name not in root:
                continue

            def collect(name, value, destination=info[out_key]):
                if isinstance(value, h5py.Dataset):
                    destination[name] = (1 if len(value.shape) < 2
                                         else int(value.shape[1]))

            root[group_name].visititems(collect)
    if len(_FIELD_INFO_CACHE) > 4096:
        _FIELD_INFO_CACHE.clear()
    _FIELD_INFO_CACHE[key] = info
    return info


def read_mesh_field_info(path):
    return read_hdf_field_info(path) if _is_hdf(path) \
        else read_vtu_field_info(path)


def read_vtm(path):
    """Parse a vtkMultiBlockDataSet: returns {block_name: vtu_path}."""
    key = _file_key(path)
    cached = _VTM_CACHE.get(key)
    if cached is not None:
        _VTM_CACHE.move_to_end(key)
        return cached
    tree = ET.parse(key[0])
    base = os.path.dirname(key[0])
    blocks = {}
    next_suffix = {}

    def collect(parent, block_name=None):
        for child in parent:
            tag = _localname(child.tag)
            if tag == "DataSet" and child.get("file"):
                name = block_name or child.get("name") or f"block{len(blocks)}"
                unique = name
                suffix = next_suffix.get(name, 2)
                while unique in blocks:
                    unique = f"{name}_{suffix}"
                    suffix += 1
                next_suffix[name] = suffix
                blocks[unique] = os.path.join(base, child.get("file"))
            else:
                collect(child, child.get("name") if tag == "Block" else block_name)

    collect(tree.getroot())
    _VTM_CACHE[key] = blocks
    if len(_VTM_CACHE) > 1024:
        _VTM_CACHE.popitem(last=False)
    return blocks


def read_pvd(path):
    """Parse a ParaView collection: [(timestep, abs_file_path), ...]."""
    key = _file_key(path)
    cached = _PVD_CACHE.get(key)
    if cached is not None:
        return cached
    tree = ET.parse(path)
    base = os.path.dirname(path)
    out = []
    for ds in tree.getroot().iter():
        if ds.tag.endswith("DataSet") and ds.get("file"):
            out.append((float(ds.get("timestep", len(out))),
                        os.path.join(base, ds.get("file"))))
    out.sort(key=lambda t: t[0])
    if len(_PVD_CACHE) > 32:
        _PVD_CACHE.clear()
    _PVD_CACHE[key] = out
    return out


def _frame_block_paths(pvd_path, frame_index):
    entries = read_pvd(pvd_path)
    if not entries:
        raise VtuParseError("Empty PVD collection")
    frame_index = max(0, min(int(frame_index), len(entries) - 1))
    _, path = entries[frame_index]
    if os.path.splitext(path)[1].lower() == ".vtm":
        return read_vtm(path)
    return {"Volume": path}


def load_frame(pvd_path, frame_index, blocks=None):
    """pvd -> vtm -> vtu/hdf frame -> {block_name: parsed mesh dict}.

    When `blocks` is provided (an iterable of block names), only those blocks
    are parsed from disk, avoiding unnecessary I/O and cache eviction of other
    blocks in multi-block VTM outputs.
    """
    paths = _frame_block_paths(pvd_path, frame_index)
    if blocks is not None:
        wanted = set(blocks)
        paths = {name: p for name, p in paths.items() if name in wanted}
    return {name: read_mesh_cached(p) for name, p in paths.items()
            if os.path.isfile(p)}


def load_block(pvd_path, frame_index, block_name=None, fallback=True):
    """Load only one named block from a PVD frame, or None if unavailable.

    Reads just that block's file through read_mesh_cached instead of decoding
    every sibling block in a VTM. When `fallback` is True and `block_name` is
    absent, falls back to the first existing block in the frame.
    """
    paths = _frame_block_paths(pvd_path, frame_index)
    if block_name and block_name in paths and os.path.isfile(paths[block_name]):
        return read_mesh_cached(paths[block_name])
    if fallback:
        for p in paths.values():
            if os.path.isfile(p):
                return read_mesh_cached(p)
    return None


def read_mesh_points(path):
    """Read only the Points array of a mesh file (reusing cache when warm).

    For VTK-HDF files not already cached, reads only /VTKHDF/Points instead of
    inflating every PointData/CellData dataset and element connectivity.
    """
    key = _file_key(path)
    cached = _VTU_CACHE.get(key)
    if cached is not None:
        return cached["points"]
    if _is_hdf(path):
        h5py = _import_h5py()
        with h5py.File(path, "r") as handle:
            if "VTKHDF" not in handle or "Points" not in handle["VTKHDF"]:
                return np.zeros((0, 3), dtype=np.float64)
            pts = np.asarray(handle["VTKHDF"]["Points"][...], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] < 3:
                padded = np.zeros((len(pts), 3), dtype=np.float64)
                padded[:, :pts.shape[1]] = pts
                pts = padded
            return pts
    return read_mesh_cached(path)["points"]


def read_mesh_point_field(path, field_name):
    """Read one PointData array from a mesh file (reusing cache when warm).

    For VTK-HDF files not already cached, reads only that single dataset.
    """
    key = _file_key(path)
    cached = _VTU_CACHE.get(key)
    if cached is not None:
        return cached["point_data"].get(field_name)
    if _is_hdf(path):
        h5py = _import_h5py()
        with h5py.File(path, "r") as handle:
            root = handle.get("VTKHDF")
            if root is None or "PointData" not in root:
                return None
            pd = root["PointData"]
            if field_name not in pd:
                return None
            return pd[field_name][...]
    return read_mesh_cached(path)["point_data"].get(field_name)


def frame_field_info(pvd_path, frame_index):
    """Metadata-only load_frame: {block_name: field-info dict}."""
    blocks = _frame_block_paths(pvd_path, frame_index)
    return {name: read_mesh_field_info(p) for name, p in blocks.items()
            if os.path.isfile(p)}


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    frames = read_pvd(sys.argv[1])
    blocks = load_frame(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(f"{len(frames)} frames; frame loaded in {time.time()-t0:.3f}s")
    for name, mesh in blocks.items():
        print(f"  block '{name}': {len(mesh['points'])} pts, "
              f"cells={{ {', '.join(f'{f}: {len(c)}' for f, c in mesh['cells'].items())} }}, "
              f"fields={sorted(mesh['point_data'])}")
