"""Native numpy VTK parser (VTU/VTK-HDF/VTM/PVD) for PolyFEM output.

Replaces meshio for the readPVD HDA: zero dependencies beyond numpy,
vectorized decode of ascii / inline-base64 / appended data arrays with
optional zlib compression, and vectorized linearization of higher-order
(Lagrange) tets into P1 sub-tets.
"""

import base64
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
    12: ("hex", 8), 25: ("hex", 8), 72: ("hex", 8),
    21: ("line", 2), 68: ("line", 2), 22: ("tri", 3), 69: ("tri", 3),
    23: ("quad", 4), 70: ("quad", 4), 1: ("point", 1),
}

# Higher-order Lagrange tet subdivision tables: each row indexes the element's
# node array (corners, then edge, then face, then interior nodes, in VTK
# Lagrange order) to build one P1 sub-tet. These tables are the readPVD 0.26
# subdivisions, regenerated verbatim from that asset's hand-built tables, so
# tetra10/20/35 (P2/P3/P4) all render with their curvature instead of
# collapsing to a corner tet.
TET10_SUBDIV = np.array([          # P2 (10 nodes) -> 8 sub-tets
    [0, 4, 7, 6], [4, 1, 8, 5], [5, 6, 2, 9], [8, 7, 9, 3],
    [6, 5, 8, 9], [7, 6, 8, 9], [8, 6, 7, 4], [8, 6, 4, 5]], dtype=np.int64)

TET20_SUBDIV = np.array([          # P3 (20 nodes) -> 27 sub-tets
    [0, 4, 10, 9], [3, 15, 11, 13], [1, 5, 6, 12], [2, 7, 8, 14],
    [10, 16, 11, 18], [10, 9, 4, 18], [10, 16, 18, 4], [4, 16, 18, 19],
    [4, 9, 19, 18], [8, 9, 18, 19], [4, 5, 16, 19], [5, 6, 12, 19],
    [5, 16, 19, 12], [15, 11, 13, 17], [16, 19, 17, 18], [13, 11, 16, 17],
    [13, 12, 17, 16], [19, 12, 16, 17], [19, 6, 12, 17], [16, 11, 18, 17],
    [11, 15, 18, 17], [15, 14, 18, 17], [8, 14, 17, 18], [8, 19, 18, 17],
    [7, 8, 14, 17], [7, 8, 17, 19], [6, 7, 17, 19]], dtype=np.int64)

TET35_SUBDIV = np.array([          # P4 (35 nodes) -> 64 sub-tets
    [2, 10, 18, 9], [18, 20, 30, 31], [10, 9, 30, 18], [20, 32, 21, 29],
    [21, 3, 17, 15], [10, 31, 18, 30], [20, 31, 32, 30], [20, 32, 29, 30],
    [17, 32, 29, 21], [15, 21, 32, 17], [17, 26, 32, 15], [17, 29, 32, 26],
    [26, 32, 15, 14], [25, 32, 26, 14], [25, 33, 32, 14], [25, 33, 14, 13],
    [25, 33, 13, 4], [33, 13, 4, 12], [0, 12, 4, 13], [22, 33, 4, 12],
    [22, 11, 33, 12], [23, 11, 33, 22], [31, 11, 33, 23], [31, 10, 11, 23],
    [30, 10, 31, 23], [30, 9, 10, 23], [30, 34, 23, 31], [31, 34, 23, 33],
    [32, 34, 31, 33], [31, 34, 32, 30], [29, 34, 30, 32], [29, 34, 32, 26],
    [26, 34, 32, 25], [25, 34, 32, 33], [25, 4, 22, 33], [25, 34, 33, 22],
    [22, 34, 33, 23], [4, 5, 25, 22], [34, 5, 22, 25], [34, 5, 24, 22],
    [34, 23, 22, 24], [34, 30, 23, 24], [8, 30, 24, 23], [9, 30, 8, 23],
    [24, 30, 8, 28], [28, 30, 34, 24], [29, 30, 34, 28], [28, 16, 34, 29],
    [29, 16, 34, 26], [26, 16, 17, 29], [26, 16, 34, 27], [27, 25, 34, 26],
    [25, 27, 34, 5], [27, 24, 34, 5], [28, 24, 34, 27], [27, 16, 34, 28],
    [27, 19, 16, 28], [27, 6, 19, 28], [27, 5, 6, 24], [27, 24, 6, 28],
    [6, 24, 7, 28], [7, 24, 8, 28], [7, 19, 6, 28], [6, 19, 7, 1]],
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


def _strip_inline_payloads(data):
    """Remove inline <DataArray> payloads, returning (skeleton, ranges).

    An inline-binary/ascii VTU keeps each array's (possibly huge) base64/ascii
    text inside its <DataArray> element, so parsing the whole document into an
    ElementTree needs gigabytes of RAM. This scans the raw bytes, blanks out
    each payload (leaving a small tag skeleton to parse) and records the
    payload's byte range so it can be decoded on demand from the original
    buffer. Base64/ascii payloads never contain '<', so the tag scan is safe.
    `ranges[i]` is the byte span of the i-th <DataArray> in document order, or
    None for a self-closing tag (appended format).
    """
    close_tag = b"</DataArray>"
    out = bytearray()
    ranges = []
    pos = 0
    while True:
        start = data.find(b"<DataArray", pos)
        if start < 0:
            out += data[pos:]
            break
        tag_end = data.find(b">", start)
        if tag_end < 0:
            out += data[pos:]
            break
        if data[tag_end - 1:tag_end] == b"/":            # <DataArray .../>
            out += data[pos:tag_end + 1]
            ranges.append(None)
            pos = tag_end + 1
            continue
        close = data.find(close_tag, tag_end)
        if close < 0:
            out += data[pos:]
            break
        out += data[pos:tag_end + 1]                      # opening tag
        ranges.append((tag_end + 1, close))               # payload span
        out += close_tag                                  # closing tag
        pos = close + len(close_tag)
    return bytes(out), ranges


# All caches are keyed by (path, mtime, size) so external file changes
# (e.g. a still-running simulation appending frames) invalidate naturally.
_PVD_CACHE = {}
_FIELD_INFO_CACHE = {}
_VTU_CACHE = OrderedDict()
# parsed frames are large; hold enough for the stages that re-read the same
# frame in one cook (topology + frame + multi-block + reference comparison)
_VTU_CACHE_SIZE = 3


def _file_key(path):
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def clear_caches():
    _PVD_CACHE.clear()
    _FIELD_INFO_CACHE.clear()
    _VTU_CACHE.clear()


class _Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        # appended section (raw): everything after the first '_' following
        # <AppendedData
        self.appended = None
        idx = self.data.find(b"<AppendedData")
        if idx >= 0:
            start = self.data.find(b"_", idx)
            end = self.data.rfind(b"</AppendedData>")
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
        self.header_dtype = np.dtype(
            _VTK_TO_NP[self.root.get("header_type", "UInt32")])
        comp = self.root.get("compressor", "")
        self.compressed = "ZLib" in comp

    def _decode_block(self, raw):
        """Decode (possibly multi-block zlib) payload -> bytes."""
        if not self.compressed:
            hsize = self.header_dtype.itemsize
            n = int(np.frombuffer(raw, self.header_dtype, count=1)[0])
            return raw[hsize:hsize + n]
        hdr = np.frombuffer(raw, self.header_dtype, count=3)
        nblocks = int(hdr[0])
        sizes = np.frombuffer(
            raw, self.header_dtype, count=nblocks,
            offset=3 * self.header_dtype.itemsize)
        offset = (3 + nblocks) * self.header_dtype.itemsize
        out = []
        for size in sizes:
            out.append(zlib.decompress(raw[offset:offset + int(size)]))
            offset += int(size)
        return b"".join(out)

    def read_data_array(self, elem):
        dtype = np.dtype(_VTK_TO_NP[elem.get("type")])
        ncomp = int(elem.get("NumberOfComponents", "1"))
        fmt = elem.get("format", "ascii")
        if fmt == "appended":
            if self.appended is None:
                raise VtuParseError("appended format without AppendedData")
            offset = int(elem.get("offset", "0"))
            arr = np.frombuffer(
                self._decode_block(self.appended[offset:]), dtype=dtype)
        else:
            span = self._payload.get(id(elem))
            payload = (self._payload_source[span[0]:span[1]]
                       if span is not None else b"")
            if fmt == "ascii":
                arr = np.array(payload.split(), dtype=dtype)
            elif fmt == "binary":
                arr = np.frombuffer(
                    self._decode_block(base64.b64decode(payload)), dtype=dtype)
            else:
                raise VtuParseError(f"Unknown DataArray format {fmt}")
        if ncomp > 1:
            arr = arr.reshape(-1, ncomp)
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
        starts = np.concatenate(([0], offsets[:-1]))
        sizes = offsets - starts

    cells = {}
    cell_sources = {}
    for ctype in np.unique(types):
        info = VTK_CELLS.get(int(ctype))
        if info is None:
            continue  # skip exotic cells rather than fail
        family, n_corners = info
        mask = types == ctype
        source_indices = np.flatnonzero(mask)
        c_starts = starts[mask]
        c_sizes = sizes[mask]
        n_nodes = int(c_sizes[0])
        if not np.all(c_sizes == n_nodes):
            # ragged same-type cells: fall back per-cell (rare)
            conn = np.vstack([
                connectivity[s:s + n_corners]
                for s, sz in zip(c_starts, c_sizes)])
        else:
            conn = connectivity[
                c_starts[:, None] + np.arange(n_nodes)[None, :]]
        subdiv = (TET10_SUBDIV if family == "tet" and n_nodes == 10 else
                  TET20_SUBDIV if family == "tet" and n_nodes == 20 else
                  TET35_SUBDIV if family == "tet" and n_nodes == 35 else None)
        if subdiv is not None:
            # vectorized higher-order tet -> P1 sub-tets (P2/P3/P4)
            conn = conn[:, subdiv].reshape(-1, 4)
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

    topo = b"".join(
        cells[f].tobytes() for f in sorted(cells)) + str(len(points)).encode()
    topo_key = hash(topo)

    return {"points": points, "cells": cells, "point_data": point_data,
            "cell_data": cell_data, "topo_key": topo_key}


def read_vtu(path):
    """Parse an UnstructuredGrid .vtu into the readPVD mesh contract."""
    reader = _Reader(path)
    piece = reader.root.find(".//{*}Piece") or reader.root.find(".//Piece")
    if piece is None:
        raise VtuParseError(f"No <Piece> in {path}")

    def find(parent, tag):
        e = parent.find(tag)
        return e if e is not None else parent.find("{*}" + tag)

    points_elem = find(piece, "Points")
    points = reader.read_data_array(
        points_elem.find("DataArray")
        if points_elem.find("DataArray") is not None
        else points_elem.find("{*}DataArray"))

    cells_elem = find(piece, "Cells")
    arrays = {da.get("Name"): reader.read_data_array(da)
              for da in cells_elem}

    point_data = {}
    pd = find(piece, "PointData")
    if pd is not None:
        for da in pd:
            name = da.get("Name")
            if name:
                point_data[name] = reader.read_data_array(da)

    raw_cell_data = {}
    cd = find(piece, "CellData")
    if cd is not None:
        for da in cd:
            name = da.get("Name")
            if name:
                raw_cell_data[name] = reader.read_data_array(da)

    return _mesh_from_arrays(
        points, arrays["connectivity"], arrays["offsets"], arrays["types"],
        point_data, raw_cell_data)


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
            return _mesh_from_arrays(
                root["Points"][...], root["Connectivity"][...],
                root["Offsets"][...], root["Types"][...], point_data,
                cell_data, offsets_include_zero=True)
    except OSError as error:
        raise VtuParseError(f"Could not read VTK-HDF file {path}: {error}") \
            from error


def read_vtu_cached(path):
    """read_vtu with a small LRU keyed on file identity.

    The returned dict is shared between callers and must not be mutated.
    """
    key = _file_key(path)
    mesh = _VTU_CACHE.get(key)
    if mesh is None:
        mesh = read_vtu(path)
        _VTU_CACHE[key] = mesh
        while len(_VTU_CACHE) > _VTU_CACHE_SIZE:
            _VTU_CACHE.popitem(last=False)
    else:
        _VTU_CACHE.move_to_end(key)
    return mesh


def _is_hdf(path):
    return os.path.splitext(path)[1].lower() in (".hdf", ".h5", ".hdf5")


def read_mesh_cached(path):
    """Read either VTK XML or VTK-HDF through the shared small LRU."""
    key = _file_key(path)
    mesh = _VTU_CACHE.get(key)
    if mesh is None:
        mesh = read_hdf(path) if _is_hdf(path) else read_vtu(path)
        _VTU_CACHE[key] = mesh
        while len(_VTU_CACHE) > _VTU_CACHE_SIZE:
            _VTU_CACHE.popitem(last=False)
    else:
        _VTU_CACHE.move_to_end(key)
    return mesh


_DATA_ARRAY_RE = re.compile(rb"<DataArray\b[^>]*>")
_XML_ATTR_RE = re.compile(rb'(\w+)="([^"]*)"')


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


def read_vtu_field_info(path):
    """Field names -> component counts without decoding any array data.

    Returns {"point_data": {name: components}, "cell_data": {...}}. If the
    frame is already fully parsed (in the LRU) the info is taken from it for
    free; otherwise it is a raw byte scan of the PointData/CellData sections
    (inline payloads contain no '<', so only real DataArray tags match).
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
        data = f.read()
    info = {"point_data": {}, "cell_data": {}}
    for tag, out_key in ((b"PointData", "point_data"),
                         (b"CellData", "cell_data")):
        start = data.find(b"<" + tag)
        if start < 0:
            continue
        open_end = data.find(b">", start)
        if open_end > 0 and data[open_end - 1:open_end] == b"/":
            continue  # <PointData/> : empty section
        end = data.find(b"</" + tag + b">", start)
        if end < 0:
            end = len(data)
        for match in _DATA_ARRAY_RE.finditer(data, open_end, end):
            attrs = dict(_XML_ATTR_RE.findall(match.group(0)))
            name = attrs.get(b"Name")
            if name:
                info[out_key][name.decode()] = int(
                    attrs.get(b"NumberOfComponents", b"1"))
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
    tree = ET.parse(path)
    base = os.path.dirname(path)
    blocks = {}
    for block in tree.getroot().iter():
        if block.tag.endswith("Block"):
            name = block.get("name", f"block{len(blocks)}")
            ds = block.find("DataSet")
            if ds is None:
                ds = block.find("{*}DataSet")
            if ds is not None and ds.get("file"):
                blocks[name] = os.path.join(base, ds.get("file"))
    # flat vtm (DataSets without Block wrappers)
    if not blocks:
        for ds in tree.getroot().iter():
            if ds.tag.endswith("DataSet") and ds.get("file"):
                blocks[ds.get("name", f"block{len(blocks)}")] = \
                    os.path.join(base, ds.get("file"))
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
    if path.endswith(".vtm"):
        return read_vtm(path)
    return {"Volume": path}


def load_frame(pvd_path, frame_index):
    """pvd -> vtm -> vtu/hdf frame -> {block_name: parsed mesh dict}."""
    blocks = _frame_block_paths(pvd_path, frame_index)
    return {name: read_mesh_cached(p) for name, p in blocks.items()
            if os.path.isfile(p)}


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
