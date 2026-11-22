# HDA source map

**MSH Reader** turns
Gmsh meshes into Houdini geometry. **PolyFEM** prepares simulation inputs and
manages runs. **Read PVD** loads, displays, and exports the results.
**writeMSH** exports tetrahedral meshes from Houdini. Each asset has a
`build.py` and a `runtime_manifest.py` under its source directory.

## Embedded Python and parameter interfaces

[`common/section_build.py`](common/section_build.py) reads parts in manifest
order and packs them as `py_*` data sections. It generates a `PythonModule`
bootstrap that gets the active HDA definition and executes the embedded source
in one shared module namespace. It registers the source with `linecache` so
tracebacks show part-local filenames and source lines. Read PVD and PolyFEM
compile the complete module with `COMPILE_LINE_GAPS` before relocating its
function code objects. This preserves CPython 3.13 bytecode and class line
metadata. Keep statements in the original order when moving part boundaries.

[`common/dialogscript.py`](common/dialogscript.py) assembles PolyFEM's parameter
interface from [`DialogScript.in`](polyfem/sections/dialog/DialogScript.in).
Each `@INCLUDE file.ds@` resolves relative to the file that contains it. The
expander inserts fragments verbatim and checks for include cycles. The builder
then expands material visibility macros and injects material display callbacks.

## MSH Reader

- [`msh_reader/PythonModule.py`](msh_reader/PythonModule.py) handles SOP cooking,
  reload, and geometry output. [`common/msh_parser.py`](common/msh_parser.py)
  parses Gmsh files for both MSH Reader and Read PVD.
- [`msh_reader/build.py`](msh_reader/build.py) defines the parameters and SOP
  network. [`msh_reader/vex/`](msh_reader/vex/) builds primitives and stamps
  element attributes.

## PolyFEM

| Files in [`polyfem/runtime/`](polyfem/runtime/) | Where to look |
| --- | --- |
| `00_header_vex`, `05_ui_utils`, `06_parm_utils` | Imports, surface and minimum-edge VEX, IDs, UI feedback, parameter helpers and migration |
| `10_transforms`, `15_geo_nodes`, `20_subdomains`, `25_reload_geometry` | Transforms, pivots, geometry networks, subdomain assignment, preserving settings on reload |
| `30_sidesets`, `35_timeline`, `40_time_curves` | Selections, simulation time, load curves, merging boundary conditions, sideset and volume export |
| `45_materials`, `50_fibers` | Material models, element order, per-element data, fiber display and export |
| `55_json_obstacles_geo`, `60_json_materials_bcs`, `65_json_solver` | Geometry and obstacle motion, materials and boundary conditions, solver and output settings |
| `70_write_scene`, `75_check_setup_geo`, `76_check_setup_solver`, `80_runner` | Input and scene records, preflight checks, launching, monitoring, stopping, and opening results |
| `85_import_core`, `90_import_materials`, `95_import_legacy`, `96_import_apply` | Reconstructing sidesets and materials, restoring settings, import coverage (`_IMPORT_KEYS`), and applying imported scenes |

The [DialogScript fragments](polyfem/sections/dialog/) group Geometry controls
under `geometry/` and Solver controls under `solver/`. Global, Time, Contact,
Remeshing, Transform, Render, and Output have their own `.ds` files. Event
hooks live beside the builder, and the viewport picker lives in
[`polyfem/sections/ViewerStateModule.py`](polyfem/sections/ViewerStateModule.py).

## Read PVD

- [`readpvd/runtime/`](readpvd/runtime/) starts with HDF5 support, field and
  fiber discovery, timeline metadata, menus, probes, reference scans, and
  viewport overlays (`00`–`40`). Cook stages, callbacks, and live following
  are in `45`–`56`. Force curves and mesh, field, region, and data exports
  are in `60`–`85`.
- [`readpvd/parms/`](readpvd/parms/) defines templates for sources, display,
  overlays, inspection, analysis, force curves, and export.
- [`readpvd/network.py`](readpvd/network.py) builds the SOP network.
  [`readpvd/vex/`](readpvd/vex/) handles topology, deformation, derived
  fields, coloring, filters, and glyphs. [`readpvd/scripts.py`](readpvd/scripts.py)
  supplies event hooks and SOP callbacks.

## Shared code and verification

[`write_msh/definition/`](write_msh/definition/) contains the editable
Python SOP definition for `writeMSH::2`. The builder uses Houdini's
`hotl` to package those sections with the runtime generated from
`write_msh`'s [runtime manifest](write_msh/runtime_manifest.py). PolyFEM's
direct volume SOP import and `writeMSH` both embed
[`common/sop_msh_writer.py`](common/sop_msh_writer.py). Run
`hython src/write_msh/build.py` to rebuild its project library on its own.

[`common/`](common/) contains build helpers plus the shared Gmsh and VTU
parsers and XLSX writer. From the HDA project directory, run the static check
with plain Python. Use Houdini's `hython` for builds and saved-asset checks.

```sh
python3 src/common/check_parts.py
hython src/build_all.py
hython src/common/hda_fingerprint.py --dir . --compare /path/to/baseline_hda_fingerprint.json
```

The batch build writes all four project libraries. Its install step copies
MSH Reader, PolyFEM and Read PVD into the user `otls/` directory when available.
To capture a fingerprint for those three pipeline libraries, use
`--record /path/to/fingerprint.json` with `--dir`. The
[isolated loader tests](../tests/test_hda_modular_loader.py) check the embedded
part contents and exercise loading and tracebacks under `hython`. The
[`writeMSH` smoke test](../tests/test_write_msh_hda.py) checks its embedded
sources and the export button on real tetrahedral SOPs.
