# Embedded HDF5 reader

The official h5py 3.16.0 wheels from PyPI for Houdini 22's CPython 3.13
runtime, one per platform Houdini 22 runs on:

| wheel | SHA-256 |
| --- | --- |
| `h5py-3.16.0-cp313-cp313-macosx_11_0_arm64.whl` (Apple silicon) | `42108e93326c50c2810025aade9eac9d6827524cdccc7d4b75a546e5ab308edb` |
| `h5py-3.16.0-cp313-cp313-macosx_10_13_x86_64.whl` (Intel Mac) | `370a845f432c2c9619db8eed334d1e610c6015796122b0e57aa46312c22617d9` |
| `h5py-3.16.0-cp313-cp313-manylinux_2_28_x86_64.whl` (Linux x86_64, glibc 2.28+) | `9300ad32dea9dfc5171f94d5f6948e159ed93e4701280b0f508773b3f582f402` |
| `h5py-3.16.0-cp313-cp313-win_amd64.whl` (Windows x64) | `18f2bbcd545e6991412253b98727374c356d67caa920e68dc79eab36bf5fedad` |

The Apple-silicon wheel was added on 2026-09-22, the other three on
2026-10-01 (downloaded from files.pythonhosted.org and checked against the
digests PyPI publishes for h5py 3.16.0).

The readPVD builder base64-embeds every wheel in the HDA. On the first HDF5
load, the asset extracts the one for the running platform into a
content-addressed folder below Houdini's user preference directory. Each
wheel includes h5py's BSD license and the HDF5 library license under its
`dist-info/licenses/` directory. Only the Apple-silicon wheel is exercised by
the tests on this Mac; the others are the unmodified PyPI files.
