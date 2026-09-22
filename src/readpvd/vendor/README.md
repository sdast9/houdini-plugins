# Embedded HDF5 reader

`h5py-3.16.0-cp313-cp313-macosx_11_0_arm64.whl` is the official h5py wheel
from PyPI for Houdini 22's CPython 3.13 runtime on Apple-silicon macOS.

SHA-256:
`42108e93326c50c2810025aade9eac9d6827524cdccc7d4b75a546e5ab308edb`

The readPVD builder base64-embeds the wheel in the HDA. On the first HDF5
load, the asset extracts it into a content-addressed folder below Houdini's
user preference directory. The wheel includes h5py's BSD license and the HDF5
library license under its `dist-info/licenses/` directory.
