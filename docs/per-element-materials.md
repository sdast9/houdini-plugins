# Spec: Fiber Models, HGO Dispersion, Model Sums & Per-Element Material Data in the PolyFEM 2.0 HDA

Status: **implemented design record** (written 2026-07-30; status reviewed 2026-09-05).
Phases 0–4 and the readPVD material/fiber pipeline are present in source and tests,
including sign-aware smoothing and dispersion coloring. The 2.0 generation was
published in `sdast9/houdini-plugins` at `5b8c5e9`; PolyFEM's material-file caching,
bounds guard, and duplicate-model output names landed in `3503148d7` and are on
`main`. See [README.md](hdas.md) for the user-facing feature summary.

Sections below retain the original implementation plan, sketches, and July 30
upstream observations. Phrases such as "today", "must add", "crashes", and
"Phase 0 required" describe that pre-implementation baseline, not outstanding
bugs in the current fork. The live source and tests take precedence over sketches;
the phasing table is a historical plan. Deferred questions remain in section 10.
The September 5 review inspected source; it did not rerun the HDA test suite.

Target asset: `object_stevenabramowitch.dev.PolyFEM.2.0.hdanc`, source of truth in
`houdini_HDAs/src/polyfem/` (`PythonModule.py`, `sections/DialogScript`), rebuilt with
`hython src/build_all.py` (installs into `~/Library/Preferences/houdini/22.0/otls/`).

The original plan below records the upstream contract and implementation rationale.
Everything in §1 was **verified live on 2026-07-30** against
`polyfem/build/PolyFEM_bin` (built 2026-07-30 10:59 from the merged tree, commit
`a65e2f11`); the original verification recipes are reproduced in §8; implemented coverage lives
in the source test suite.

---

## 0. Goal

The upstream merge added three capabilities the HDA does not yet expose:

1. **New material models** — `HGODispersion` (GOH fiber model with dispersion `kappa`
   and a smooth tension/compression switch), joining the existing fiber models
   `HGOFiber` and `ActiveFiber`, plus `IsochoricNeoHookean`.
2. **Per-element material data** — fiber directions per element (legacy-VTK
   `CELL_DATA VECTORS` file), and per-element values for *any* scalar material
   parameter (plain text file, one value per element).
3. **Model summation** — `MaterialSum` composes any number of elastic models into one
   material (canonical soft-tissue use: isotropic matrix + N fiber families).

The HDA must let users: pick these models per subdomain; author/import fiber fields and
per-element scalars; **see** those fields in the viewport before running; and export a
`params.json` + sidecar files that run under polyfem's strict JSON validation.
On the results side, readPVD must ingest the material fields polyfem exports
(currently it **crashes** on them — §7.0) and visualize fibers (reference and
deformed) and per-element scalars on the loaded timesteps (§7).

---

## 1. Verified upstream contract (facts the implementation relies on)

### 1.1 Model catalog additions

| type (JSON)        | required            | optional                                        | notes |
|--------------------|---------------------|--------------------------------------------------|-------|
| `HGOFiber`         | `k1`, `k2`          | `id`, `rho`, `fiber_direction`                   | classic aligned HGO-2000: isochoric I4bar, hard tension-only cutoff |
| `HGODispersion`    | `k1`, `k2`          | `id`, `rho`, `kappa`, `k_chi`, `fiber_direction` | E4 = kappa·I1 + (1 − d·kappa)·I4 − 1 on FULL invariants; logistic switch centered at E4=0, smoothness `k_chi` (default 100); `kappa` default 0 (= aligned limit); kappa ∈ [0, 1/d], d = spatial dim |
| `ActiveFiber`      | `activation`        | `id`, `rho`, `Tmax`, `fiber_direction`           | active tension along fiber; `activation` ∈ [0,1], typically an expression of `t`; `Tmax` in stress units |
| `IsochoricNeoHookean` | `E`,`nu` (or `lambda`,`mu`) | `id`, `rho`, `phi`, `psi`             | good matrix for fiber sums |
| `MaterialSum`      | `type`              | `id`, `models`, `rho`                            | `models` = array of full material objects (each with its own `type` and params, **no** `id`/`rho` inside children); energies/gradients/Hessians add |

Units: `k1`, `Tmax` are stresses (Pa in the HDA's unit regime); `k2`, `kappa`,
`activation`, `k_chi` dimensionless. Source: `src/polyfem/assembler/HGODispersion.{hpp,cpp}`,
`HGOFiber.hpp`, `ActiveFiber.hpp`, `SumModel.{hpp,cpp}`,
`json-specs/elastic-material-parameters.json`.

Mixing **different** material types across bodies automatically selects the
`MultiModels` formulation (`varforms/VarFormFactory.cpp:formulation_from_args`), and
`MaterialSum` is a legal participant (verified live: body 1 `MaterialSum`, body 2
`NeoHookean`). Material `id` may be an **array** (`"id": [1001, 1002]`) — verified.

### 1.2 Per-element fiber directions

```json
"fiber_direction": {"type": "per_element_file", "path": "fibers.vtk", "field": "FIB_DIR1"}
```

* File format: legacy **ASCII** VTK. The reader
  (`MatParams.cpp:read_cell_vectors_legacy_vtk`) only scans for two tokens — a
  `CELL_DATA <n>` line, then a `VECTORS <field> <dtype>` line followed by `n`
  whitespace-separated `x y z` triples. Points/cells sections are *not* required. A
  minimal file is valid:

  ```
  # vtk DataFile Version 3.0
  fibers
  ASCII
  DATASET UNSTRUCTURED_GRID
  CELL_DATA 2229
  VECTORS FIB_DIR1 float
  0.123 0.456 0.789
  ...
  ```
* `field` defaults to `FIB_DIR1`. One file can carry several `VECTORS` arrays
  (distinct field names) — the reader scans until the named one.
* Vectors are **normalized at load**; a zero-length vector is a fatal
  `log_and_throw_error`. NaNs poison silently — the HDA must pre-validate (V4).
* `path` resolves relative to the input JSON (the HDA runs polyfem with
  `cwd = <working_dir>/input/` and bare basenames — keep that).
* **Binding is by GLOBAL element id**: element ids run over the concatenation of all
  FEM meshes in `geometry`-array order (obstacles and `"enabled": false` entries do
  **not** consume ids). Within one `.msh`, element order = file (entity-block) order —
  polyfem's `MshReader` appends blocks without sorting. Verified live: fibers
  constructed as `normalize(centroid − anchor)` for 1101 + 1128 tets across two meshes
  came back aligned (dot > 0.999) for **all** body-1 elements via the paraview
  `material` output.
* The per-element branch sets `has_rotation_ = false` and returns a column vector, so
  it behaves exactly like a constant `[x,y,z]` fiber elsewhere in the code.

### 1.3 Per-element scalars (kappa, k1, k2, E, nu, rho, Tmax, activation, …)

Any scalar material parameter that goes through `GenericMatParam` / `LameParameters` /
`Density` accepts, besides a number:

* an **expression string** in `x, y, z, t` (tinyexpr; helpers: `min, max, smoothstep,
  half_smoothstep, deg2rad, rotate_2D_x/y, if, compare, smooth_abs, sign`);
* a **file path string**: if the string resolves (relative to the JSON) to an existing
  regular file, `ExpressionValue::init` loads it with `read_matrix` and evaluation
  returns `mat_(el_id)` — i.e. **row el_id of the file**, with el_id the same global
  element id as §1.2. Format: plain text, one value per line (Eigen dense reader; a
  single column is the safe shape).

Verified live: `"kappa": "kappa.txt"` with `kappa_i = 0.30·i/n` over 2229 global
elements reproduced exactly (per-element constant within cells, unique-value multiset
match at 5e-10) in the output field `MaterialSum/HGODispersion/kappa`.

**Gotcha**: a file shorter than the max el_id indexes out of bounds with **no** bounds
check (release build = memory garbage or crash). The HDA must never emit short files
(V2), and the Phase-0 fork patch adds a guard.

### 1.4 `MaterialSum` mechanics and constraints

`SumModel::add_multimaterial` (called once **per element** via
`Assembler::set_materials`) creates its child assemblers **once**, from the first
element that carries a `models` array; every subsequent element only asserts
`assemblers_.size() == models.size()` (assert = debug only) and forwards
`models[i]` into child `i` **by position**.

Consequences (encode as validation rule V1):

* Every body whose material is `MaterialSum` must declare the **same ordered list of
  child types**. Different lists across bodies = silent parameter corruption in
  release builds. Same list with different parameter *values* per body is fine
  (verified: the per-element `add_multimaterial` carries each body's own values).
* **Duplicate child types (two fiber families) are solve-correct today** — each
  `models` entry gets its own assembler instance. Verified live (2026-07-30):
  `[NeoHookean, HGODispersion(a0=x̂), HGODispersion(a0=ẑ)]` under 5% z-stretch gives
  mean σzz 7435 Pa vs 6117 Pa with the ẑ family removed, and a measurably different
  displacement field. What breaks is only the **output channel**:
  `SumModel::parameters()` keys fields by child type name, so the second same-type
  child overwrites the first in the exported vtu — verified: the two-family run
  exports a single `HGODispersion/fiber_direction_*` set equal to `[0,0,1]` (the
  *last* child). Physics right, post-viz misleading → fixed by fork patch **P0.4**.
* Output field prefix depends on the formulation: single-body `MaterialSum` runs
  export `<ChildType>/<param>` (e.g. `HGODispersion/kappa`); when `MultiModels` is
  active (mixed bodies) the wrapper adds its prefix: `MaterialSum/<ChildType>/<param>`
  (both observed live). Consumers (readPVD, tests) must match by suffix, not exact
  prefix.
* `rho` lives on the sum object, not the children.

### 1.5 Output / post-visualization channel

`output.paraview.options.material = true` (already wired to the HDA's
`materials_fields` toggle) exports **every** assembler parameter as point-associated
fields, including `…/fiber_direction_x|y|z` and `…/kappa`. Note the exported
fiber_direction is the **reference** direction a0 in simulation/world coordinates.
PolyFEM applies `geometry[].transformation` to mesh points only; it does not apply
that transform to a0. The current direction is `normalize(F·a0)`, derivable in
readPVD from the exported `F_*` columns
(remember: F columns are stored column-major per the project's established convention —
see memory note; any consumer must transpose).

With `MultiModels`, the union of all active models' parameters is exported and each
param function is evaluated on **every** element — including elements of bodies that
don't use that model. This is another reason per-element files must span the *global*
element count (§1.2, §1.3): output evaluation would otherwise throw (fiber file, which
is range-checked) or read garbage (scalar file, which is not).

### 1.6 Performance hazard (quadratic loading) — fork patch required for large meshes

Because `add_multimaterial` runs once per element:

* `FiberDirection::add_multimaterial` **re-reads the whole VTK once per element** of
  every body using it (measured: 1101 reads for an 1101-element body).
* `ExpressionValue::init` re-runs `read_matrix` per element, **and each element's
  `ExpressionValue` instance keeps its own full copy of the matrix** — O(n²) time *and*
  memory. 100k elements × 100k rows × 8 B ≈ 80 GB. Unusable beyond ~5–10k elements
  without the Phase-0 patch.

At HDA-test scale (≤ ~3k elements) both are harmless — Phase 0 is not a blocker for
implementing and testing the HDA, but it is a blocker for production meshes.

---

## 2. Phase 0 — fork-side patches (polyfem repo, not the HDA)

Four small patches in `src/polyfem/`. Keep them in the fork; propose upstream later.

**P0.1 — cache per-element fiber files.** `assembler/MatParams.{hpp,cpp}`,
`FiberDirection::add_multimaterial`, in the `per_element_file` branch: return
immediately if already loaded with the same `(resolved path, field)`; otherwise load
via a `static std::map<std::pair<std::string,std::string>, std::shared_ptr<const
std::vector<Eigen::Vector3d>>>` + mutex, and store the `shared_ptr` in the member
(`per_el_fibers_` becomes the shared pointer). Semantics note: today, if two bodies
name *different* files for the same assembler instance, the last body processed wins
for **all** elements — the cache keyed per instance must therefore also
`log_and_throw_error` when a second, different `(path, field)` reaches the same
`FiberDirection`. That turns silent misbehavior into an error and makes the HDA's
single-global-file contract (§5.3) the enforced one.

**P0.2 — cache scalar matrix files.** `utils/ExpressionValue.{hpp,cpp}`: change
`Eigen::MatrixXd mat_` to `std::shared_ptr<const Eigen::MatrixXd>`; in `init(string)`
load through a `static std::unordered_map<std::string, std::shared_ptr<const
Eigen::MatrixXd>>` keyed by resolved path (+ mutex). Fixes both the O(n²) time and
memory.

**P0.3 — bounds guard.** In `ExpressionValue::operator()` where `result =
mat_(index)`: if `index < 0 || index >= mat_.size()`,
`log_and_throw_error("Value file {} has {} rows but element {} was requested", …)`.
(The fiber path already has this check; the scalar path does not.)

**P0.4 — disambiguate duplicate child types in `SumModel::parameters()`.** Required
for the ±θ two-family helper's output/visualization (solve needs nothing — see §1.4).
`assembler/SumModel.cpp`:

```cpp
std::map<std::string, Assembler::ParamFunc> SumModel::parameters() const
{
    std::map<std::string, int> total, seen;
    for (const auto &a : assemblers_) total[a->name()]++;
    std::map<std::string, Assembler::ParamFunc> params;
    for (const auto &a : assemblers_) {
        std::string prefix = a->name();
        if (total[a->name()] > 1)
            prefix += "_" + std::to_string(seen[a->name()]++);
        for (auto &it : a->parameters())
            params[prefix + "/" + it.first] = it.second;
    }
    return params;
}
```

Naming contract: unique child types keep `<ChildType>/<param>` (backward compatible);
repeated types become `<ChildType>_0/…`, `<ChildType>_1/…` in `models`-array order
(0-based occurrence index among same-named children). Under `MultiModels` the wrapper
prefix still applies on top (`MaterialSum/HGODispersion_1/kappa`).

Rebuild `PolyFEM_bin`, re-run `run-smoke.sh` and the §8 verification scripts before
touching the HDA. Also re-run `tests/test_hgo_dispersion` (`ctest -R hgo`).

---

## 3. UI design (DialogScript)

Existing conventions to follow: per-volume parms are `name{geo}_{vol}` written as
`name#_#` inside the two nested multiparms; third-level multiparm parms are `name#_#_#`
(pattern already used by sidesets/BCs). Menus are token-based; `hidewhen`/`disablewhen`
compare against tokens. Braced groups in `hidewhen` OR together; conditions inside one
group AND together.

### 3.1 Material menu (`materials#_#`) — append four entries

Keep existing token order (indices 0–10 are load-bearing for import); append:

| index | token | label |
|---|---|---|
| 11 | `IsochoricNeoHookean` | Isochoric NeoHookean |
| 12 | `HGOFiber` | HGO Fiber (aligned) |
| 13 | `HGODispersion` | HGO Dispersion (GOH) |
| 14 | `ActiveFiber` | Active Fiber (muscle) |
| 15 | `MaterialSum` | Composite (Matrix + Fiber Families) |

`IsochoricNeoHookean` reuses the existing `E#_#`/`nu#_#` (extend their
hide/disablewhen lists). Obstacles: `BC_check`-style guard — fiber/composite materials
on an `is_obstacle` geometry are meaningless; obstacles don't emit materials at all
today, so nothing to do beyond leaving the material folder hidden as it already is.

### 3.2 Fiber-source parameter block (reused in two nesting depths)

A reusable block of parms; instantiate once at volume level (`…#_#`, shown when
`materials#_#` is one of `HGOFiber HGODispersion ActiveFiber`) and once at family
level (`…#_#_#`, inside the family multiparm of the composite). Prefix `fib_`
(volume level) / `fam_fib_` (family level).

| parm | type | default | shown when | meaning |
|---|---|---|---|---|
| `fib_source#_#` | ordinal menu | `constant` | fiber model active | tokens: `constant`, `expression`, `attribute`, `file` |
| `fib_dir#_#` | float vec3 | `1 0 0` | source == constant | constant material-frame direction |
| `fib_expr#_#` | string | `[1, 0, 0]` | source == expression | 3-entry list; entries numbers or `x,y,z,t` tinyexpr strings, passed through `parse_vector` |
| `fib_sop#_#` | oppath | `""` | source == attribute | SOP node to sample; empty = the HDA's own volume geometry (`branch_#`) |
| `fib_attrib#_#` | string | `fiber1` | source == attribute | prim **vector** attribute holding per-element fiber |
| `fib_file#_#` | file | `""` | source == file | external per-geometry file: `.vtk` (legacy, CELL_DATA VECTORS) or `.txt/.csv` (n×3) |
| `fib_file_field#_#` | string | `FIB_DIR1` | source == file, path endswith .vtk | VECTORS array name to read |
| `fib_refresh#_#` | button | | source in attribute/file | re-stamp fiber data (see §4.2) |

`attribute` and `file` are the **per-element** sources; both are resolved to a stamped
prim attribute on the HDA's volume geometry (§4.2) so visualization and export share
one code path ("what you see is what the solver gets").

### 3.3 HGO / Active parameters

Volume level (shown per the material menu):

| parm | type | default | range | shown for |
|---|---|---|---|---|
| `hgo_k1#_#` | float | `10000` | 0 … 1e7 | HGOFiber, HGODispersion |
| `hgo_k2#_#` | float | `5.0` | 1e-6 … 1e3 (min enforced: k1/(2·k2) singular at 0) | HGOFiber, HGODispersion |
| `kappa#_#` | float | `0.0` | 0 … 0.3333 | HGODispersion, kappa_source == constant |
| `kappa_source#_#` | ordinal | `constant` | tokens `constant`, `expression`, `attribute`, `file` | HGODispersion |
| `kappa_expr#_#` | string | `""` | | HGODispersion, source == expression |
| `kappa_sop#_#` / `kappa_attrib#_#` (default `kappa`) / `kappa_file#_#` | oppath / string / file | | | HGODispersion, per-element sources; `kappa_file` is per-geometry: one value per element, that mesh's element order |
| `k_chi#_#` | float | `100` | 1 … 1e4, **advanced folder** | HGODispersion |
| `Tmax#_#` | float | `100000` | 0 … 1e7 | ActiveFiber |
| `activation#_#` | string | `"1"` | | ActiveFiber; number or expression of `t` (e.g. `"0.5*(1-cos(2*3.14159*t))"`); required non-empty (V7) |

Family level: same set with `fam_` prefix and `#_#_#` indices, shown per
`fam_model#_#_#`.

### 3.4 Composite (Matrix + Fiber Families), `materials#_# == MaterialSum`

* `matrix_model#_#` — ordinal menu, tokens: `NeoHookean` (default),
  `IsochoricNeoHookean`, `LinearElasticity`, `MooneyRivlin`, `MooneyRivlin3Param`,
  `MooneyRivlin3ParamSymbolic`, `UnconstrainedOgden`, `IncompressibleOgden`,
  `FixedCorotational`, `None` (fiber-only; warn on export). The matrix's scalar
  parameters **reuse the existing per-volume parms** (`E#_#`, `nu#_#`, `c1#_#`, …,
  `bulk#_#`, `ogden*#_#`): extend each one's `hidewhen` with an OR-group
  `{ materials#_# == MaterialSum matrix_model#_# == <relevant tokens> }`.
* `num_fiber_families#_#` — int multiparm count (default 1, min 0). Inside
  (`#_#_#` = geo, vol, family):
  * `fam_model#_#_#` — menu: `HGOFiber`, `HGODispersion` (default), `ActiveFiber`.
  * the `fam_fib_*` fiber-source block (§3.2) and `fam_` params (§3.3).
  * **±θ pair helper** (the standard GOH two-family layout — e.g. arterial
    collagen at ±θ about the circumferential direction; decided 2026-07-30):
    | parm | type | default | shown when | meaning |
    |---|---|---|---|---|
    | `fam_mirror#_#_#` | toggle | 0 | always (per family) | emit this family twice, base direction rotated by +θ and −θ about a local axis; both children share k1/k2/kappa/k_chi |
    | `fam_theta#_#_#` | float (deg) | 30 | mirror on | half-angle between the families |
    | `fam_axis_source#_#_#` | ordinal | `constant` | mirror on | tokens `constant`, `attribute` — rotation axis (sheet/wall normal) |
    | `fam_axis#_#_#` | float vec3 | `0 0 1` | mirror on, axis constant | constant rotation axis |
    | `fam_axis_attrib#_#_#` | string | `normal1` | mirror on, axis attribute | per-element prim vector attribute; sampled from the same source SOP as `fam_fib_sop` (centroid-matched like the fiber attribute) |

    Constraints: mirror requires the fiber source to be `constant`, `attribute` or
    `file` (an `expression` source cannot be rotated symbolically → validation error
    V10). The rotation happens **HDA-side at stamp time** (§4.2), so polyfem sees two
    ordinary children.
* `volume_penalty#_#` toggle + `vp_k#_#` float (adds `{"type": "VolumePenalty",
  "k": …}` child) — Phase 4, optional.
* `extra_models_json#_#` — string, default `""`, advanced folder. Verbatim JSON array
  appended to `models` (parsed with `json.loads`, must be a list; validated by
  polyfem's strict validation at run). Escape hatch for anything the UI can't express;
  also the round-trip fallback target (§6).

### 3.5 Per-geometry visualization parms (sibling of the existing color swatches)

| parm | type | default | meaning |
|---|---|---|---|
| `show_fibers#` | toggle | 0 | display fiber hedgehogs for this geometry |
| `fiber_scale#` | float | 0.5 | line length as fraction of mean element edge |
| `fiber_max_lines#` | int | 20000 | decimation cap (stride = ceil(n/max)) |
| `color_by#` | ordinal | `subdomains` | tokens `subdomains`, `kappa`, `fiber_rgb` — recolors the surface chain (§4.3) |

---

## 4. Node-graph & data pipeline

### 4.1 Where per-element data lives

Canonical storage: **prim attributes on the volume geometry**, stamped onto a new
node `fiberdata_<geo>` (Python SOP) inserted between `branch_<geo>` and
`surface_<geo>` in `create_geo_tree` (it must also survive `clear_geo_tree` — add
`"fiberdata"` to the keep-set or rebuild it with the display chain; rebuilding is
simpler and matches how sideset groups are handled).

Attribute naming (all prim, on volume prims only):

* `pf_fiber` / `pf_fiber2` / `pf_fiber3` … — vector; family k of the composite uses
  `pf_fiber<k>`, the single fiber models use `pf_fiber`. (Fixed names — the `fib_attrib`
  parm names the *source* attribute; the stamp copies it to the canonical name.)
* `pf_fiber<k>p` / `pf_fiber<k>m` — the two rotated directions of a mirrored (±θ)
  family; replace `pf_fiber<k>` for that family.
* `pf_kappa`, `pf_kappa<k>` — float, only when kappa source is per-element.

### 4.2 Stamping (`update_fiber_data(kwargs)` in PythonModule.py)

Runs on: `fib_refresh` buttons, before export (always), and after params import.
For each fiber-bearing material slot of geometry `geo`:

1. Resolve `n =` number of volume prims of `branch_<geo>` (prims with
   `is_volume == 1`; equals the msh element count — cross-check against
   `export_volumes`' count, assert equal).
2. Source == `attribute`: read prim attrib `fib_attrib` from `fib_sop` (empty path =
   `branch_<geo>` itself, i.e. attribute arrived with the mesh or from a previous
   stamp).
   * If the source SOP is external: sample via **centroid matching** — build a KD tree
     (scipy is available in hython; else brute-force numpy in chunks) over source
     volume-prim centroids, map each HDA element centroid to nearest source prim,
     error if `max distance > 1e-4 × bbox diagonal` (V6) — robust to prim order, works
     with a re-meshed source only when geometry actually matches; exact-count +
     near-zero-distance fast path skips the tree.
   * Vectors are taken as-is (material frame = the post-transform space the user sees;
     polyfem's reference configuration is the transformed mesh, so no re-basis needed).
3. Source == `file`:
   * `.vtk` → parse `CELL_DATA` + named `VECTORS` (mirror of §1.2's reader — write
     `read_cell_vectors_legacy_vtk()` in Python, ~20 lines, reusable in tests);
   * `.txt`/`.csv` → `np.loadtxt` (n×3), delimiter auto (whitespace or comma);
   * row count must equal `n` (V2-geo), rows are **that mesh's element order**
     (= volume-prim order; V3 guards the mixed-family caveat).
4. Scalar per-element sources (`kappa_*`): same flow, 1 column.
5. Normalize fiber vectors; **error** listing element numbers if any norm < 1e-12 or
   NaN (V4).
5b. Mirrored (±θ) families: resolve the rotation axis (constant vector, or
   per-element attribute via the same centroid matching as step 2), normalize it,
   then produce both directions with the Rodrigues formula (numpy, vectorized):
   `a± = a0·cosθ ± (n × a0)·sinθ + n·(n·a0)·(1 − cosθ)`. Validate per element:
   axis norm ≥ 1e-12 and |n·a0| < 1 − 1e-9 (axis ∥ fiber makes the pair degenerate)
   — error with element numbers otherwise (V10). Store as `pf_fiber<k>p/m`.
6. Write `pf_*` attribs (numpy bulk `setPrimFloatAttribValuesAsString` on the
   fiberdata node's stashed geometry — the Python SOP's cook copies input and applies
   stored arrays; stash the arrays as node user data / cachedUserData keyed by
   attrib name, so cooks are deterministic and .hip-persistent via the parm values
   that regenerate them).

Implementation note: a Python SOP cook that re-derives everything from parms each cook
is the simplest persistent design (no user-data persistence issues): cook = "read
parms → resolve sources → stamp". Refresh button just forces a re-cook
(`fiberdata.cook(force=True)`). External SOPs are read inside the cook via
`hou.node(path).geometry()` — acceptable because the sources are explicit user actions,
mirroring the HDA's existing button-driven style.

### 4.3 Visualization

Two additions, both per geometry, built in `create_geo_tree` (they participate in the
existing rebuild lifecycle automatically):

**Fiber hedgehogs** — `fiberviz_<geo>` (attribwrangle, prim class, input =
`fiberdata_<geo>`), bypassed unless `show_fibers#`. VEX (detail-independent,
parallel-safe — creates geometry, deletes source prims):

```vex
// prim wrangle on volume prims: one polyline per (kept) element per family
if (i@is_volume == 0 || @primnum % chi("../stride") != 0) { removeprim(0, @primnum, 1); return; }
float scale = chf("../scale");                    // absolute length, precomputed
string names[] = split(chs("../fib_attribs"));    // e.g. "pf_fiber pf_fiber2p pf_fiber2m",
                                                  // written by Python at stamp time
foreach (string an; names) {
    if (!hasprimattrib(0, an)) continue;
    vector d = normalize(prim(0, an, @primnum));
    vector c = v@centroid;
    int p0 = addpoint(0, c - 0.5 * scale * d);
    int p1 = addpoint(0, c + 0.5 * scale * d);
    int ln = addprim(0, "polyline", p0, p1);
    setprimattrib(0, "Cd", ln, abs(d), "set");    // DTI-style |dir| RGB
    setprimattrib(0, "geometry_num", ln, i@geometry_num, "set");
}
removeprim(0, @primnum, 1);
```

Channels `stride`, `scale` are parm expressions: `stride =
max(1, ceil(nvolprims / fiber_max_lines#))` (Python parm expr), `scale = fiber_scale# ×
mean edge` (reuse the `mindist_<geo>` machinery: mean of `min_edge_length` is
acceptable; do not force-cook at build). `fib_attribs` is a plain string parm on the
wrangle that `update_fiber_data` rewrites with the current canonical attrib names
(including `p`/`m` pairs of mirrored families), so the wrangle needs no knowledge of
the material layout. Output feeds a `fibviznull_<geo>` null that is
an extra input to the `all` merge (hedgehogs render on top of the shaded surface).
`centroid` comes from MSH_Reader; note it survives `transform_<geo>`'s point transform
only if recomputed — **recompute centroid in `fiberdata_<geo>`** (`v@centroid` = mean of
prim points) so hedgehogs sit on the transformed mesh.

**Color-by modes** — extend the `entitycolor_<geo>` wrangle snippet
(`_entity_color_vex`) with a mode switch channel bound to `color_by#`:

* `subdomains` (0): current behavior (per-Entity swatch colors).
* `kappa` (1): surface prims inherit their source volume element's `pf_kappa` (the
  SURFACE_VEX wrangle must copy `pf_kappa*`/`pf_fiber*` from the parent volume prim
  onto emitted boundary faces — one extra `setprimattrib` per attribute, guarded by
  `hasprimattrib`), colored with a fixed viridis-like ramp over [0, 1/3] plus a
  min/max readout printed to the node comment; grey where absent.
* `fiber_rgb` (2): `Cd = abs(normalize(pf_fiber))` per face (first family).

`color_by#` change callback: rebuild only the color wrangle snippet (no full tree
rebuild). The existing `test_polyfem_colors.py` guards subdomain colors; new modes get
their own assertions (§8 T5).

### 4.4 Suggested-fiber presets — IMPLEMENTED

Shipped as **explicit `fib_source` menu tokens** (not as a special case of the
`attribute` source — a distinct token keeps the parameter conditionals honest and the
UI self-describing), at both volume and family level:

* **`cylindrical`** — parms `fib_axis_origin`, `fib_axis_dir`, `fib_component`
  (circumferential / radial / longitudinal). `cylindrical_frame()` returns the whole
  orthonormal frame; elements lying *on* the axis are an error naming them, since the
  radial direction is undefined there (that is a solid cylinder, not a tube).
* **`curve`** — parm `fib_curve` (oppath). `curve_tangent()` takes the tangent of the
  nearest point over every segment of the referenced polyline, chunked so a fine curve
  against a large mesh cannot blow up memory. No Resample required: segments are read
  straight off the SOP's primitives (closed curves included).
* **Artery layout**: `fam_axis_source` also gained `cylindrical`, so a ±θ pair can be
  wound about the *same* cylinder's radial direction (the wall normal) — circumferential
  collagen at ±θ, which is the case the two-family helper exists for.

Both resolve inside `resolve_fibers` and therefore flow through the same stamping,
visualization and export path as any other per-element source (`PER_ELEMENT_FIBER_SOURCES`
is the single list that decides "this needs a file"). They round-trip as `file`, since
what is exported is the baked direction field — value-exact, like the ±θ pair.

---

## 5. Export (`PythonModule.py`)

### 5.1 Global element table

New helper, computed at export time (and cached on the node for stamping/validation):

```python
def global_element_table(parent):
    """[(geo, offset, count)] over enabled, non-obstacle geometries in geo order.
    offset/count are in GLOBAL polyfem element ids (§1.2)."""
```

Count = volume-prim count of `branch_<geo>` (assert equal to the row count
`export_volumes` writes for the same geometry — single source of truth for "element
order" already shipping). Skip `is_enabled == 0` and obstacles — **el_ids shift when a
geometry is disabled**; because the table is derived per export, this is handled
automatically, but per-element data of a *disabled* geometry is simply not exported.

### 5.2 Validation rules (hard errors unless noted; run before writing any file)

| rule | check | message sketch |
|---|---|---|
| V1 | all `MaterialSum` volumes produce the identical ordered child-type list `[matrix] + families` — compared on the **expanded** list (a ±θ family counts as its two children) | "Composite materials must use the same model stack on every subdomain (polyfem shares one summed assembler). geo 1/vol 2 has […]; geo 2/vol 1 has […]" |
| V1b (warn) | duplicate child types within one sum are solve-correct (§1.4) but their output fields collide **unless the binary carries P0.4**; the HDA cannot detect the patch, so warn once per export when duplicates are present | "Two same-type families: material fields in the output show only the last family unless the fork's SumModel naming patch (P0.4) is in the binary" |
| V2 | every emitted per-element file has exactly `total_global_elements` rows; external per-geometry inputs have exactly that geometry's element count | counts in message |
| V3 | any geometry using per-element data has a single volume-element family (all tet or all hex) | "Mixed tet/hex meshes reorder elements on import; per-element data would bind to the wrong elements. (Planned fix: file-order stamp in MSH_Reader.)" |
| V4 | fiber vectors: no zero/NaN after normalize | list first 10 offending element numbers |
| V5 | per-element kappa ∈ [0, 1/3] (3D) | min/max found |
| V6 | attribute source resolves: node exists, attrib exists, is prim vector (or float for scalars), centroid match within tolerance | what was found instead |
| V7 | `activation` non-empty; `k2 > 0`; `k1 ≥ 0`; `Tmax ≥ 0` | |
| V8 (warn) | composite with `matrix_model == None` and 0 families | "emits nothing; pick a matrix or add a family" (then emit plain matrix or error if truly empty) |
| V10 | ±θ mirror: fiber source is not `expression`; rotation axis non-degenerate per element (norm ≥ 1e-12, not parallel to the base fiber) | "Mirrored families need a rotatable fiber source (constant/attribute/file)" / offending element numbers |

### 5.3 File emission

* **One global fiber file**: `input/fibers.vtk`, one `VECTORS` array per distinct
  family field:
  * field names: `FIB_<geo>_<vol>` (single fiber models), `FIB_<geo>_<vol>_<k>`
    (composite family k), `FIB_<geo>_<vol>_<k>p` / `FIB_<geo>_<vol>_<k>m` (the two
    members of a mirrored family). Deterministic, collision-free, and legible in
    ParaView.
  * each array is full global length; rows outside the owning geometry's slice get
    the placeholder `1 0 0` (never evaluated for the solve; harmlessly evaluated by
    the output writer, §1.5).
  * write with numpy (`np.savetxt` into an open handle after the header lines);
    9-decimal fixed format.
* **Scalar files**: `input/pe_<param>_<geo>_<vol>[_<fam>].txt` — full global length,
  one float per line; padding value = the parm's constant value (sane values if ever
  evaluated cross-body).
* Rewritten on every export (same lifecycle as `volumes<geo>.txt` / sideset files).

### 5.4 `build_material()` extensions

New menu indices (§3.1). Helper for the fiber-source triple:

```python
def fiber_direction_json(parent, geo, vol, fam=None):
    src = <fib_source parm>            # constant | expression | attribute | file
    if src == constant:   return [x, y, z]                    # from fib_dir
    if src == expression: return parse_vector(fib_expr, ...)  # 3 entries, numbers/strings
    # attribute | file  -> per-element, exported to the global vtk by §5.3
    return {"type": "per_element_file", "path": "fibers.vtk",
            "field": field_name(geo, vol, fam)}
```

and the analogous `scalar_json(...)` returning float | expression-string |
`"pe_<...>.txt"`. Then:

* `m == 11` → `{"id": mid, "type": "IsochoricNeoHookean", "E": E, "nu": nu, "rho": rho}`
* `m == 12` → `{"id": mid, "type": "HGOFiber", "k1": k1, "k2": k2, "rho": rho,
  "fiber_direction": fiber_direction_json(...)}`
* `m == 13` → HGODispersion: same + `"kappa": scalar_json(kappa…)`; emit `"k_chi"`
  only when ≠ 100 (keep JSON minimal).
* `m == 14` → ActiveFiber: `"activation"` = number-if-parseable else string;
  `"Tmax"`; `fiber_direction`.
* `m == 15` → `{"id": mid, "type": "MaterialSum", "rho": rho, "models": models}`
  where `models = [matrix_json] + [family_json(k) for k] (+ volume penalty)
  (+ json.loads(extra_models_json))`; `matrix_json` reuses the existing isotropic
  emitters (factor the m==0..10 bodies into small `def`s so both paths share them),
  **without** `id`/`rho` keys. A mirrored family expands to **two** children sharing
  k1/k2/kappa/k_chi: constant base + constant axis → two constant
  `fiber_direction` lists (rotated in numpy at export, no VTK involvement);
  per-element base or axis → two `per_element_file` refs to the `…p`/`…m` fields.

`orders` handling (`mainOrder`) is unchanged — it keys off `vol_id`, independent of
material type.

### 5.5 Output side

No changes needed for the run itself. When any per-element/fiber material is present
and the user has `minimal_fields` on **with** `materials_fields` off, print a one-line
status hint that enabling *Material Fields* exports fibers/kappa for post-viz
(`MaterialSum/<Child>/fiber_direction_*`, `…/kappa`) — do not silently change their
output settings.

---

## 6. Import / round-trip (`import_params`)

Extend the `type_menu` map with the four new types (11–14) and `MaterialSum` → 15.
Per material `m`:

* `fiber_direction`:
  * list of 3 numbers → source `constant`, set `fib_dir`;
  * list with any string entry → source `expression`, set `fib_expr` to the JSON list
    re-serialized;
  * dict `per_element_file` → source `file`, `fib_file` = path resolved into the
    params.json's directory, `fib_file_field` = field; then run `update_fiber_data`
    so the imported fibers are immediately **visible** (§4.2 step 3 reads the VTK; the
    global-file slice mapping on import: rows `[offset, offset+n)` of the imported
    geometry per the §5.1 table — when the file is exactly this HDA's own export the
    round-trip is exact; foreign files with per-geometry length also accepted).
* scalar values (kappa, k1, k2, Tmax, activation, E, nu, …): number → parm; string →
  file-exists? (relative to json dir) → source `file` + stamp; else source
  `expression`.
* `MaterialSum`: `models[0]` matching an isotropic type → `matrix_model` + its parms;
  remaining fiber-typed models → families in order (duplicate family types are fine —
  they import as separate explicit families; a pair that was exported via the ±θ
  mirror toggle round-trips **value-exact** but comes back as two explicit families
  with the toggle off, which is acceptable since the expanded child list — what V1
  and polyfem see — is identical); any child that doesn't fit (unknown type, nested
  sum) → serialize those children into `extra_models_json` verbatim and `_message` a
  note (round-trip preserved even when the UI can't represent it).
* Unknown top-level material type (future upstream additions) → keep current behavior
  (warn + default), but ALSO stash the full material dict into `extra_models_json`?
  **No** — that changes semantics (extra_models only exists under composite). Warn and
  skip, as today.

Legacy (1.x) imports: nothing new — old files predate these features.

### 6.1 `input-spec.json` sync note

`import_params` and `build_*` must track `polyfem/json-specs/` (established project
practice — see the `pressure_discr_order` removal precedent). The four model schemas
this spec bakes in are listed in §1.1; re-audit against
`json-specs/elastic-material-parameters.json` at implementation time and whenever the
fork merges upstream again.

---

## 7. readPVD: fiber & material-field visualization

Target: `object_readPVD.1.0`, source in `src/readpvd/` (`PythonModule.py` for cook
logic, `build.py` for the network **and** all parms — readPVD builds its parameter
interface programmatically in `_parms()` with `hou.ParmTemplate`, no DialogScript
section). Architecture recap (header of `PythonModule.py`): `topo_build` →
`build_prims` → `boundary` → `frame_data` (per-frame bulk upload of every PointData
field) → `deform` → `derived_switch` → `reference_comparison` → `color_reduce` →
`smooth_switch` → `color_map` → `body_filter` → `OUT_result`, with glyphs as their own
blast-isolated stream (`glyph_tensor_recovery` → `glyphs` → `glyph_only` →
`glyph_switch`) merged after the clip stage so clipping never cuts them. The mesh is
**discontinuous** (points duplicated per element; `coincident_id` marks physical
nodes). `show_deformed` adds `solution` to rest positions in the `deform` wrangle.

### 7.0 R0 — slash-named fields crash the loader (bugfix, do first)

`cook_frame` uploads vtu PointData names verbatim:
`geo.addAttrib(hou.attribType.Point, name, …)`. Material fields under `MaterialSum` /
`MultiModels` contain slashes (`HGODispersion/kappa`, `MaterialSum/NeoHookean/E`) and
`addAttrib` **raises** on them (verified 2026-07-30 in hython 22.0: "Could not add
attribute 'HGODispersion/kappa'"). Today, enabling *Material Fields* on any multi-body
or summed-material run produces a vtu that readPVD **fails to load at all**. This is a
live bug independent of everything else in this spec.

Fix in `cook_frame` (and the parallel cell-data path):

```python
def _attrib_name(raw):          # module level, used by upload + menus + probe
    name = re.sub(r"[^0-9A-Za-z_]", "_", raw)
    return name if not name[:1].isdigit() else "_" + name
```

* Apply after `FIELD_ALIASES`; keep a per-load `display_names` dict
  `{attrib_name: raw_vtu_name}` stored as a JSON detail attribute
  (`readpvd_display_names`) by `cook_frame`, refreshed every frame.
* Everything that shows field names to the user — `color_field_menu`,
  `glyph_tensor_menu`, `probe_readout`, legend title (`legend_title_text`),
  diagnostics — looks up the display name and falls back to the attrib name. The
  sanitized name is the menu **token** (stable, expression-safe); the raw name is the
  **label**.
* `_fields_from_info` / `_sequence_metadata` (menu population when scanning frames
  without loading) must run the same sanitizer so tokens agree between scanned and
  loaded paths.
* Collisions (two raw names sanitizing identically) are theoretically possible, not
  expected from polyfem output; on collision append `_2`, `_3`, … and map both.

With R0 alone, **scalar material fields already visualize for free**: the color-by
menu enumerates every uploaded point field generically, so
`MaterialSum_HGODispersion_kappa` becomes selectable, ramp-mapped, legended, probed
— no further work. Labels read from the display-name map (e.g. shown as
"MaterialSum/HGODispersion/kappa [all frames]").

### 7.1 Fiber field detection & assembly

polyfem exports fibers as **three scalar fields** `<prefix>fiber_direction_x|y|z`
where `<prefix>` is any of: `` (plain fiber formulation), `<Child>/` (single-body
`MaterialSum`), `MaterialSum/<Child>/` (`MultiModels`), and with P0.4 indexed
`<Child>_<i>/` variants — always match by **suffix**, group by prefix.

In `cook_frame`, after the generic upload loop: find every prefix with all three
components present and assemble a vector attribute
`fib_<sanitized_prefix>` (e.g. `fib_HGODispersion`, `fib_MaterialSum_HGODispersion_1`;
plain prefix → `fib_default`). Numpy stack, one `upload_point` call each. Record the
family list (attrib name + display label derived from the raw prefix, e.g.
"HGODispersion (family 2)") in the `readpvd_display_names` detail JSON under a
`fiber_families` key. Mirror the `has_glyph_data` pattern: `sync_available_options`
sets a hidden `has_fiber_data` toggle parm and populates a `fiber_family` ordinal
menu (`All Families` + one entry per family).

These are **reference** directions a0 in simulation/world coordinates, constant per
element in time (polyfem evaluates the material param, not the deformed state), and
duplicated onto every copy of an element's nodes — element-constant on the
discontinuous mesh. A JSON geometry transform changes their positions but never
rotates or scales their vector components.

### 7.2 UI (new "Fibers" subfolder in the Analysis folder, after the glyph block)

| parm | type | default | meaning |
|---|---|---|---|
| `show_fibers` | toggle | 0 | master switch (disabled when `has_fiber_data == 0`, same conditional style as glyphs) |
| `fiber_family` | ordinal menu | `all` | which family(ies) to draw; tokens from §7.1 |
| `fiber_frame` | ordinal menu | `deformed` | tokens `reference` (a0), `deformed` (normalize(F·a0)), `both` (reference drawn dimmed) |
| `fiber_scale` | float | 0.01 | line length (absolute, like `tensor_scale`) |
| `fiber_stride` | int | 1 | draw every Nth node (same semantics as `glyph_stride`) |
| `fiber_color_mode` | ordinal | `direction_rgb` | tokens `direction_rgb` (Cd = abs(dir), DTI-style — matches the preprocessing HDA §4.3), `family` (fixed palette per family), `uniform` (`fiber_color` vec3) |
| `fiber_color` | vec3 | 1 1 1 | uniform color |
| `autofiber` | button | | sets `fiber_scale` so lines span ~half an average edge (reuse `autoglyph`'s edge estimate) |

Smoothing needs no separate toggle: when the global `smooth_field` is on, fiber
recovery (§7.3) nodally averages like glyph recovery does; off = raw per-element
directions.

### 7.3 Network additions (mirror the glyph stream exactly)

Between `glyph_switch` and the display merge, add a parallel stream off `OUT_result`:

* `fiber_recovery` (python) — analogous to `cook_glyph_recovery`: when
  `smooth_field && show_fibers && has_fiber_data`, average each family's vectors over
  every `coincident_id` group and write the result back to all copies.
  **Fibers are line fields, not arrows**: a0 and −a0 are the same fiber, and polyfem's
  per-element normalization can flip signs between neighboring elements. Naive
  averaging cancels. Per group: take the first member as reference, flip members with
  negative dot against it, average, renormalize (vectorized: `np.bincount` on flipped
  components, like the tensor path). Zero-norm averages (balanced ±) fall back to the
  reference member.
* `fibers` (attribwrangle, point class, input = `fiber_recovery`) — VEX sketch:

```vex
// point wrangle: line-field hedgehogs; one line per physical node (coincident
// copies deduplicated exactly like GLYPH_VEX: draw only on the group's first copy).
if (chi("../show_fibers") == 0) return;
if (@ptnum % max(1, chi("../fiber_stride")) != 0) return;
if (haspointattrib(0, "coincident_first") && i@coincident_first == 0) return;
string fams[] = split(chs("../fiber_attribs"));   // rewritten by sync_available_options
float scale = chf("../fiber_scale");
string mode = chs("../fiber_frame");
int fam_idx = -1;
foreach (string fam; fams) {
    fam_idx++;
    if (chs("../fiber_family") != "all" && chs("../fiber_family") != fam) continue;
    if (!haspointattrib(0, fam)) continue;
    vector a0 = normalize(point(0, fam, @ptnum));
    vector dirs[]; float dims[] = {};
    if (mode == "reference" || mode == "both") { append(dirs, a0); append(dims, mode == "both" ? 0.45 : 1.0); }
    if (mode == "deformed" || mode == "both") {
        // F_mat exists when Derived Fields is on; else assemble from the raw
        // columns (PolyFEM flattens column-major -- transpose, see derived VEX).
        matrix3 F = haspointattrib(0, "F_mat") ? matrix3(point(0, "F_mat", @ptnum))
                  : transpose(set(point(0, "F_1", @ptnum),
                                  point(0, "F_2", @ptnum),
                                  point(0, "F_3", @ptnum)));
        append(dirs, normalize(a0 * F));   // VEX row-vector convention: a*F == F.a0 column form
        append(dims, 1.0);
    }
    int di = -1;
    foreach (vector d; dirs) {
        di++;
        vector c = @P;
        int p0 = addpoint(0, c - 0.5 * scale * d);
        int p1 = addpoint(0, c + 0.5 * scale * d);
        int ln = addprim(0, "polyline", p0, p1);
        vector col = chs("../fiber_color_mode") == "direction_rgb" ? abs(d)
                   : chs("../fiber_color_mode") == "family" ? fiber_palette(fam_idx)
                   : chv("../fiber_color");
        setpointattrib(0, "Cd", p0, col * dims[di], "set");
        setpointattrib(0, "Cd", p1, col * dims[di], "set");
        setprimgroup(0, "readpvd_fibers", ln, 1, "set");
    }
}
```

  Implementation notes for the wrangle: color the **points** not the prims (same
  reasoning as GLYPH_VEX — a prim Cd would win over the field coloring of the mesh);
  `fiber_palette()` is a small VEX function with ~8 fixed distinct colors;
  **deformed-direction math must be validated against numpy in T-R3** — the VEX
  row-vector convention (`a * F` vs `F * a`) combined with the column-major
  transpose is exactly the kind of thing that silently produces plausible-looking
  wrong directions; don't trust the sketch, trust the test. `coincident_first` (1 on
  the first copy of each physical node) should be stamped by `topo_build`/
  `cook_glyph_recovery`-adjacent code if not already available — check how GLYPH_VEX
  deduplicates ("draw a single glyph per node, not one per duplicate") and reuse that
  exact mechanism.
  Each fiber-line primitive also records `fiber_reference_direction`,
  `fiber_current_direction`, `fiber_stretch = |F a0|`,
  `fiber_isochoric_stretch`, `fiber_I4`, `fiber_I4bar`, and the sign-invariant
  direction-change angle. Stretch and angle are available as fiber color modes.
* `fiber_only` (blast, keep group `readpvd_fibers`) → `fiber_switch` (on
  `show_fibers && has_fiber_data`) → merged into `display_blocks` alongside
  `glyph_switch`'s stream (i.e. **excluded from clip/slice**, same as glyphs — a
  clipped fiber line is meaningless).
* Hedgehog positions: the stream branches off `OUT_result`, which is downstream of
  `deform` — so lines automatically sit on the deformed or rest mesh per
  `show_deformed`, consistent with everything else.

Deformed-fiber requirements: `fiber_frame != reference` needs F fields
(`F_1..F_3` or derived `F_mat`). Wire it like `DERIVED_FIELD_REQUIREMENTS`:
`sync_available_options` disables the `deformed`/`both` tokens (or falls back to
`reference` with a status message) when the loaded frame lacks
`deformation_gradient` data — polyfem's Minimal Fields preset **does** export `F`, so
the common case works, but a fields-pruned export may not.

### 7.4 Scalar material fields (kappa & friends)

Covered by R0 + the generic pipeline (§7.0). Two small polish items:

* Add friendly `FIELD_LABELS`-style handling for the common suffixes: a name ending
  `_kappa` labels as "Fiber Dispersion κ (<child>)", `_k1`/`_k2` as "Fiber Stiffness
  k1/k2 (<child>)", using the display-name map (dynamic — do **not** hardcode the
  prefix list; P0.4 indexing must keep working).
* kappa ramp: when the selected color field's raw name ends in `kappa`, default the
  ramp range to [0, 1/3] instead of autoscale (still user-overridable) so
  preprocessing (§4.3) and post both read the same colors for the same kappa.

### 7.5 Probe & diagnostics

`probe_readout` and the timeline diagnostics operate on attrib names — they pick up
sanitized material fields automatically after R0. Ensure probe display uses the
display-name map for its labels (one lookup), so a probed value reads
"MaterialSum/HGODispersion/kappa: 0.213" not the underscore soup.

### 7.6 Tests (`tests/test_readpvd_hda.py` additions)

Fixture: the §8 T2 export (two-body MaterialSum + NeoHookean with per-element fibers
and kappa) provides the vtu; keep it as a generated fixture, not a checked-in binary.

* **T-R0** — regression: load a Material-Fields vtu containing slash-named fields;
  cook succeeds; sanitized attribs exist; `readpvd_display_names` maps them back to
  the raw names; color menu lists them with raw-name labels. (This test fails on
  today's build — it pins the bug.)
* **T-R1** — family detection: `fib_MaterialSum_HGODispersion` (or P0.4-indexed
  variant) vector attrib assembled from the x/y/z triple; `has_fiber_data == 1`;
  family menu has All + 1 entry.
* **T-R2** — hedgehog geometry: `show_fibers=1`, count `readpvd_fibers` prims ==
  physical nodes / stride (± the dedup rule); `both` mode doubles it; lines excluded
  from an active clip.
* **T-R3** — deformed-direction correctness: for 20 random drawn lines, recompute
  `normalize(F·a0)` in numpy from the vtu's own `F_*` columns (transposed!) at the
  matched node and assert the VEX line direction agrees within 1e-5 (up to sign).
  This is the test that catches a wrong row/column convention in the wrangle.
* **T-R4** — kappa coloring: select the kappa field, assert ramp default range
  [0, 1/3] and per-point Cd matches the ramp of the known per-element kappa values;
  probe readout shows the raw slash name.
* **T-R5** — sign-coherent smoothing: craft a two-element fixture whose shared nodes
  carry a0 and −a0; with `smooth_field=1` the recovered node direction is ±a0 (not
  near-zero).

### 7.7 Phasing

* **R0** — sanitizer + display-name map + T-R0. Small, independent, fixes a live
  crash; ship first (can land before any preprocessing-HDA work).
* **R1** — family assembly, Fibers UI folder, hedgehog stream, T-R1..T-R3.
* **R2** — polish: labels, kappa ramp default, probe labels, smoothing recovery,
  T-R4/T-R5.

readPVD phases only depend on Phase 0/P0.4 for the *indexed* prefix variants; they
work against today's binary output otherwise.

---

## 8. Tests (`tests/test_polyfem_materials.py`, headless hython; follow repo style)

Reuse `make_two_volume_msh` from `test_polyfem_colors.py`. The binary at
`polyfem/build/PolyFEM_bin` is required (same skip/assert pattern as
`test_polyfem_hda.py`).

* **T1 — JSON emission.** Two-subdomain mesh. vol 1 = HGODispersion (constant fiber
  [0.7,0.7,0], kappa 0.2); vol 2 = Composite NeoHookean + 1×HGODispersion family with
  per-element fibers from an authored prim attribute (create a wrangle SOP in the test
  scene: `v@fiber1 = normalize(v@centroid - {0.5,0.5,-0.75})`) and per-element kappa
  from a generated txt. Assert: exact `materials` JSON structure; `fibers.vtk` has
  `CELL_DATA == total`, both `VECTORS` fields present, placeholder rows outside slices;
  `pe_kappa_*.txt` row count == total; V-rule triggers (V1: second composite volume
  with a different family list must raise; V2: external file with wrong count; V4:
  zero fiber).
* **T2 — end-to-end solve + binding proof.** Run PolyFEM_bin on T1's export (small
  dirichlet stretch, static). Parse `out.vtu` with `src/common/vtu_parser.py`:
  centroid-alignment check (every composite-body tet: |mean fiber · normalize(rest
  centroid − anchor)| > 0.999 — rest centroid = deform-corrected or use tiny 0.5%
  stretch and tolerance), kappa multiset match to the emitted file, field names
  `MaterialSum/HGODispersion/{kappa, fiber_direction_x}` present. (This reproduces the
  2026-07-30 verification: scratchpad scripts `verify_upstream_materials.py` /
  `verify_per_element_kappa.py`.)
* **T3 — round-trip.** Export → wipe node → import params.json → assert menu indices,
  scalar parms, fiber sources (`file` + field name), stamped `pf_fiber*` attribs equal
  the VTK slice; re-export → byte-identical `materials` block and identical
  `fibers.vtk` field data.
* **T4 — ordering/limits.** Synthetic mixed tet+hex msh (`test_msh_reader_topology.py`
  has machinery) + per-element source → V3 error. 2-geometry scene with geometry 1
  disabled → global table offsets shift; `fibers.vtk` covers only geometry 2's
  elements and solve runs.
* **T5 — viz.** `show_fibers1=1` → hedgehog prim count == ceil(n/stride) polylines;
  `color_by1=kappa` → surface Cd varies and matches ramp endpoints for min/max kappa;
  `color_by1=subdomains` restores swatch colors (extend, don't break,
  `test_polyfem_colors.py`).
* **T6 — ActiveFiber smoke.** vol 1 ActiveFiber (`activation="t"`, Tmax modest) +
  NeoHookean matrix via composite; 3 time steps; solver converges; json validates.
* **T7 — ±θ two-family.** Composite NeoHookean + one mirrored HGODispersion family
  (constant base x̂, axis ẑ, θ=30°): exported `models` has 3 children with
  fiber_directions `[cos30, ±sin30, 0]`; solve runs; response is stiffer than the
  single-family export in both rotated directions (or simply: displacement differs
  from single-family, mirroring the 2026-07-30 verification which measured mean σzz
  7435 vs 6117 Pa for a two- vs one-family sum under 5% z-stretch). With a P0.4
  binary additionally assert distinct `HGODispersion_0/…` and `HGODispersion_1/…`
  output fields carrying the two directions; without it, assert the collision warning
  (V1b) fired. Per-element variant: mirrored family with attribute base + attribute
  axis → `fibers.vtk` carries both `…p`/`…m` VECTORS fields, rotated rows match a
  numpy Rodrigues reference.

Acceptance: all new tests + the existing suite
(`test_polyfem_hda`, `test_polyfem_colors`, `test_legacy_import`,
`test_sideset_selection`, `test_msh_reader_topology`, `test_remesh_hda`) pass; rebuild
via `src/build_all.py` installs cleanly.

---

## 9. Milestones

| phase | content | size |
|---|---|---|
| 0 | fork patches P0.1–P0.4 + rebuild + smoke (§2) | small C++, big rebuild time |
| 1 | menu additions 11–14, scalar parms, constant/expression fiber sources, `build_material` + import for single models, T1/T3 subset | core, mostly mechanical |
| 2 | per-element pipeline: `fiberdata_` stamp node, attribute/file sources, global table, `fibers.vtk` + `pe_*.txt` emission, validations V1–V9, T1/T2/T4 | the heart of the work |
| 3 | composite UI (matrix + families multiparm), ±θ mirror helper, sum emission/import, T1 composite cases, T6, T7 | DialogScript-heavy |
| 4 | viz (hedgehogs + color-by), presets (cylindrical/curve), T5 | independent, parallelizable |
| R0 | readPVD slash-name crash fix (§7.0) + T-R0 | **do first — fixes a live loader crash; independent of everything above** |
| R1–R2 | readPVD fiber hedgehogs + material-field polish (§7.1–§7.7) | independent of Phases 1–4; only P0.4 needed for indexed prefixes |

Update `README.md` (feature section + the "sidesets are order-free / volumes are
order-based" ordering contract note) and the project memory when phases land.

## 10. Open questions (small; don't block Phases 0–2)

1. Should `kappa`'s per-element file path also be exposed for `k1`/`k2` (regional
   stiffness from imaging)? The plumbing (§5.4 `scalar_json`) makes it a 3-parm-per-
   param cost; deferred until asked.
2. ~~Preset priority for §4.4~~ — both shipped 2026-07-30 (§4.4).

(Resolved 2026-07-30: the ±θ two-family helper is **in scope** — §3.4, §4.2 step 5b,
V10, P0.4, T7. Verified beforehand that same-type children already sum correctly in
the solver; only the output naming needed the P0.4 patch.)
