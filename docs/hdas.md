# PolyFEM Houdini Pipeline — 2.0 Generation

Modernized HDAs targeting the local PolyFEM fork on **main**, including its
semi-implicit barrier mode (`polyfem/build/PolyFEM_bin`). The build installs
versioned definitions alongside legacy assets so existing nodes can retain their
old definitions; installed-library state should be checked in Houdini.

| asset | file | replaces |
| --- | --- | --- |
| `MSH_Reader::3.0` (SOP) | `sop_MSH_Reader.3.0.hdanc` | 2.2 |
| `stevenabramowitch::dev::PolyFEM::2.0` (Object) | `object_stevenabramowitch.dev.PolyFEM.2.0.hdanc` | 1.2 |
| `readPVD::1.0` (Object) | `object_readPVD.1.0.hdanc` | readPVD_higher_order 0.26 |

Design notes: [per-element-materials.md](per-element-materials.md) —
fiber models, composites, and per-element material data. **Implemented**
2026-07-30 (phases 0–4 and readPVD material/fiber support, including sign-aware
smoothing and dispersion coloring); the spec is kept as the rationale and
upstream-contract reference.

Everything is **built from source**: `src/` holds the Python modules, VEX,
OpenCL, and DialogScript sources; rebuild any time with

```bash
/Applications/Houdini/Current/Frameworks/Houdini.framework/Versions/Current/Resources/bin/hython src/build_all.py
```

Tests (headless, also exercise the real PolyFEM binary):

```bash
hython tests/test_polyfem_hda.py         # end-to-end: scene -> json -> sim -> pvd
hython tests/test_readpvd_hda.py         # result loading incl. Contact block
hython tests/test_legacy_import.py       # old params.json migration
hython tests/test_polyfem_materials.py   # fiber models, composites, per-element data
hython tests/test_readpvd_materials.py   # material fields + fiber visualization
```

## MSH_Reader 3.0

* **No gmsh dependency** (the pip auto-install flow is gone): native numpy
  parser for MSH v2.2 ASCII and v4.1 ASCII/binary, validated against the gmsh
  API on multi-volume meshes with physical groups.
* numpy bulk point creation + compiled VEX prim construction. The highest
  supported mesh dimension is emitted as the primary topology: tet/hex,
  tri/quad, or line.
* Same attribute contract as 2.2 (`msh_pt_id`, `Entity` with the +1 shift,
  `ElementNum`, `centroid` — hex centroids now use all 8 corners; 2.2 used 7).
* New: optional import of **2D physical groups** as polygons with a
  `surface_entity` attribute (gmsh-authored sidesets visible in Houdini;
  automatic mapping into PolyFEM sidesets is a planned follow-up).
* ~1M tets: **1.1 s** (2.2: 11 s, plus the gmsh install requirement).

## PolyFEM 2.0

Targets the current build's strict-validated schema — the old fork keys
(`adaptive_barrier_stiffness_multiplier`) are gone.

* **Barrier Stiffness Mode** menu: *Semi-Implicit* (default; per-contact
  stiffness with gap-band trim + stall restart — the robust mode),
  *Adaptive* (classic IPC), *Fixed*. Advanced semi-implicit knobs in a
  collapsed folder.
* **Augmented Lagrangian**: `hessian_scaled` initial weight (default on) or a
  manual value; `scaling` / `max_weight` / `eta` are real controls now
  (1.2 hardcoded them).
* **New collision-free IDs**: volume id = `1000*geo + vol`; boundary id =
  `vol_id*10000 + sideset*100 + bc`; obstacle id = `100000 + 1000*geo`.
  (1.2's string concatenation collided: geo 1/vol 12 == geo 11/vol 2.)
* **Sideset export rewritten**: boundary faces carry their gmsh vertex ids
  (stamped by VEX at build time), export is a bulk numpy dump; tri and quad
  faces go to **separate files** (mixed tet/hex meshes were broken before);
  files are emitted as `{"file": ...}` selection-list entries.
* **Sideset picking is scoped to one subdomain**: on a multi-material mesh,
  picking faces for a subdomain's sideset hides that geometry's *other*
  subdomains for the duration of the pick (a Visibility SOP per subdomain, so
  primitive numbering is untouched). Without it every entity stayed in the way
  — the material interface was unreachable behind the outer subdomain, and a
  face picked on the wrong entity was silently dropped at export. Such a pick
  now warns, and one made *entirely* on another subdomain is an error instead
  of an empty sideset. `*` still means "every face of this subdomain".
* **Fiber models and composites**: `HGOFiber`, `HGODispersion` (dispersion
  `kappa`, smooth switch `k_chi`), `ActiveFiber`, `IsochoricNeoHookean`, and
  **Composite (Matrix + Fiber Families)** = `MaterialSum`. Fiber families
  support a ±θ symmetric pair (one toggle emits two rotated children sharing
  k1/k2/kappa — the standard GOH layout).
* **Per-element material data**: fiber directions and any dispersion value can
  come from a Houdini prim attribute (centroid-matched, so any SOP-authored
  field works), an external file (legacy `.vtk` `CELL_DATA VECTORS`, or
  csv/txt), or one of two procedural presets — **Cylindrical** (radial /
  circumferential / longitudinal about an axis; the vessel frame, and the ±θ
  rotation axis can follow the same cylinder's wall normal for the classic
  artery layout) and **Curve Tangent** (every element follows a centreline
  curve — the tendon/ligament/muscle line of action). Exported as one shared
  `input/fibers.vtk` (one `VECTORS` array per
  family) plus `input/pe_*.txt`, indexed by PolyFEM's global element id.
  Validated before writing: missing attributes, wrong row counts, zero/NaN
  directions, out-of-range kappa, degenerate ±θ axes, and mixed tet/hex meshes
  (where Houdini's and PolyFEM's element orders diverge) all fail with the
  offending element numbers.
* **See what the solver gets**: per-element data is stamped onto the elements
  (`fiberdata_<geo>`) and drawn as fiber hedgehogs (*Fiber / Material Display*
  per geometry, with a line cap that decimates automatically); the surface can
  be colored by subdomain, dispersion, or fiber direction. The same data is
  what the exporter writes, so the viewport cannot disagree with the run.
  This deliberately follows PolyFEM's coordinate convention: the geometry
  transform moves the mesh only, while reference fibers `a0` remain unchanged
  in simulation/world coordinates. Fiber-SOP element matching is performed
  against the untransformed mesh first, so applying a geometry transform cannot
  invalidate an otherwise matching source field or silently rotate its values.
* **Native selections without picking**: type into any sideset *Base Group*
  field — `axis:+z:0.99`, `box:[0,0,0],[1,1,1]`, `sphere:[0,0,0],0.5`,
  `plane:[0,0,1],[0,0,0.5]` — emitted directly as polyfem selection objects
  (these take precedence over face-list files).
* **Run buttons**: *Run PolyFEM* (terminal, as before), *Run in Background*
  (headless with `output/log.txt` + *Show Log*), *Write params.json Only*.
* Adaptive remeshing settings write the current `/space/remesh` schema and
  round-trip when importing a previous `params.json`.
* Display chain rebuilt for scale: one Entity-colored boundary surface per
  geometry instead of per-volume node forests; object/subdomain color
  swatches remain independently editable.
* `read_params` restores the new schema and **imports legacy params.json**
  (geometry/transforms/materials/time/contact/solver mapped; sidesets must be
  re-selected since the id scheme changed).
* Tolerance defaults updated to the rescaled-tolerances solver semantics.
* **Provenance** (2026-09-14, RB-12): every exported `params.json` carries a
  `provenance` block — Houdini version, the asset type and library file with
  its SHA-256, the scene file and the export time — which the solver copies
  into `output/run-manifest.json` (`producer`), next to the build identity,
  executable hash, input/mesh hashes and per-step history it records for
  every run. Needs a PolyFEM build of `1f6f826fa` or later; earlier strict
  builds refuse the unknown key.

## readPVD 1.0

* **No meshio dependency**: native numpy VTK-XML parser (ascii / inline
  base64 / appended, zlib-compressed or not). Handles multi-GB inline-binary
  VTUs: the XML is fed to expat in chunks (a >2 GiB document overflows a single
  feed) and inline `<DataArray>` payloads are stripped before parsing and
  decoded on demand, so a 2 GB file does not build a 2 GB ElementTree.
* **Topology caching**: topology is parsed once (configurable frame) and
  only P + fields upload per frame. ~1M tets: first cook 1.75 s, then
  **0.25 s per frame change**. A *Remeshing Mode* toggle rebuilds topology
  every frame for remeshing runs.
* **All point and cell fields forwarded generically** (no hardcoded name
  lists); original CellData remains on primitives and receives a point-averaged
  display copy only when no same-named PointData exists. PolyFEM's
  `cauchy_stess` spelling is aliased to `cauchy_stress*`. Field names are
  sanitized into legal Houdini attribute names and mapped back for display —
  PolyFEM namespaces material output with slashes
  (`MaterialSum/HGODispersion/kappa`), which **failed to load at all** before.
* **Fiber visualization**: fiber directions arrive as three scalar fields per
  model and are reassembled into vectors, listed per family in an *Analysis ▸
  Fibers* folder. Lines can be drawn as PolyFEM's unchanged simulation/world
  reference direction `a0`, the current direction `normalize(F·a0)` (undoing
  PolyFEM's column-major tensor flattening), or both;
  they live in their own stream so clipping never cuts them. Nodal smoothing is
  sign-aware — fibers are line fields, so a0 and −a0 are the same fiber and
  naive averaging would cancel them to nothing. Coloring a `kappa` field
  defaults to the [0, 1/3] range the preprocessing HDA uses, so the same value
  reads as the same color before and after a solve. Fiber lines also carry
  inspectable `fiber_stretch`, `fiber_isochoric_stretch`, `fiber_I4`,
  `fiber_I4bar`, and sign-invariant direction-change attributes, with color
  modes for stretch and direction change.
* **Source Block menu**: Volume / Surface (normals, sidesets, traction) /
  **Contact (contact & friction forces)** / Points — the Contact block is
  the one to watch when validating the semi-implicit barrier work.
* Derived quantities in VEX. From the deformation gradient `F` alone:
  volume ratio `J`; the right and left Cauchy-Green deformation tensors
  (`C = FᵀF`, `B = FFᵀ`); the right and left stretch tensors (`U = √C`,
  `V = √B`); principal stretches; and the standard strain measures —
  Green-Lagrange (`½(C−I)`), Almansi (`½(I−B⁻¹)`), logarithmic/Hencky
  (`½ ln C`), and infinitesimal. Adding a stress field yields the Cauchy-stress
  invariants, maximum shear stress, stress triaxiality, and both
  Piola-Kirchhoff stresses, plus sorted principal values for every tensor.
* The full stress set is derived from `F` plus *either* the Cauchy stress *or*
  the 1st Piola-Kirchhoff stress (Cauchy is reconstructed as `σ = (1/J) P Fᵀ`),
  so PolyFEM output can drop the redundant Cauchy/PK2 tensors and export just
  `F + pk1` — far less to write and parse — with everything still computed here.
* Principal-direction **line glyphs** can visualize the right or left
  Cauchy-Green tensor, Green-Lagrange strain, second Piola-Kirchhoff stress, or
  Cauchy stress. The first, second, and
  third principal directions can be shown
  independently. Glyphs support value-scaled, normalized, and maximum-length
  modes, per-direction length multipliers, optional arrowheads, sampling
  stride, and red tension / blue compression colors. Enabling glyphs (or
  switching tensors) auto-estimates a scale sized to the mesh: the estimate
  divides a geometric target (~half an average edge) by a representative
  principal-value magnitude of the selected tensor, so glyphs are sensibly
  sized whether the field is stress (~1e6) or strain (~1e-2) — the same
  unit-agnostic behavior as the 0.26 viewer's normalized ellipsoid glyphs.
  With **field smoothing** on, glyphs use **nodal tensor recovery**: the glyph
  tensor is averaged over each set of coincident (duplicated) vertices and
  re-decomposed, so every physical node has one consistent set of principal
  values/directions and a single glyph is drawn per node instead of a noisy
  stack of conflicting per-element glyphs.
* **Show/hide bodies**: a data-driven *Visible Bodies* toggle list (built from
  the PVD `body_ids`) isolates one or more bodies — leave it empty to show all,
  or toggle bodies on to hide the rest (e.g. drop an obstacle, isolate one
  part). Filtering happens before the result null, so the color range, probe,
  glyphs, and clipping all act on the visible bodies. The control is enabled
  only when the data has more than one body.
* **Field smoothing across elements** (optional): PolyFEM writes a
  discontinuous mesh (per-element duplicated nodes), so element-wise
  quantities such as stress are constant within each element and render as a
  flat color. A *Smooth Field Across Elements* toggle nodally averages the
  displayed scalar onto shared vertices (FEM recovery) for a continuous,
  interpolated field — preferring PolyFEM's own continuous `_avg` fields when
  one exists and falling back to a generic per-vertex average (which also
  smooths the reader's derived quantities). The coincidence groups are
  precomputed with the cached topology, so the per-frame cost is one
  vectorized group-average (~0.06 ms at 1540 points). Linear elements have no
  true sub-element variation; for that, raise PolyFEM's `vismesh_rel_area`
  sampling or element order so it writes a denser sampled mesh (handled
  natively).
* Color any scalar, vector, or tensor point field. Every vector field offers
  vector magnitude plus X, Y, and Z components. Tensor reductions include
  Frobenius norm, diagonal entries, first/second/third principal values
  ordered largest to smallest, trace, and determinant. *Auto Range: Current
  Frame* and *Auto Range: All Frames* both use the displayed reduced value.
  Range controls include outlier-resistant percentiles, symmetric signed
  ranges, range locking, logarithmic scaling, explicit invalid-value coloring,
  and a signed diverging-ramp preset. New nodes default to Houdini's "Infra-Red"
  ramp preset (blue→cyan→green→yellow→red, no black/white ends); saved scenes
  keep whatever ramp they had.
* Source-block, color-field, displayed-value, and glyph-tensor menus are
  generated from the PVD data. Availability scope defaults to **Current Frame**
  (fast: only the displayed frame is inspected on load, and a stale selection
  shows the Unavailable Field Color rather than erroring). Every Frame / Any
  Frame additionally scan the whole sequence for intermittent fields with
  coverage labels — these read every `.vtu` and are labelled as slow, since the
  field list lives inside each file and cannot be discovered any other way.
  Options that were not exported are not shown; stale selections are replaced
  safely. Controls such as deformation and glyphs are disabled when their
  required data is unavailable. Operations that must scan the whole sequence
  (Auto Range: All Frames, Compute Field Over Time, the all-frames scopes) and
  Remeshing Mode are labelled to set expectations, as their cost is inherent.
* Optional **multi-block display** merges supplementary Volume, Surface,
  Contact, and Points blocks with the primary result. Each supplementary block
  has independent visibility, field, reduction, range, ramp, and geometry
  group.
* The embedded viewer state provides a **click probe** for positions, selected
  values, displacement, body/sideset IDs, principal values, block names, and
  reference comparison values. The probe picks the primary result only (it
  ignores glyphs and other decorations), so it always reports real field data
  at the clicked vertex, and the marker follows that vertex as frames advance.
* Reference-frame comparison supports current-minus-reference, absolute
  difference, and percentage change. Interactive clipping planes and thin
  slices expose internal colored sections; glyphs are merged after the clip so
  they are never cut.
* Timeline diagnostics scan the selected displayed value over the PVD
  sequence (vectorized numpy reduction per frame — no per-frame network cook
  or frame changes) and optionally add a renderable minimum/mean/maximum plot
  with numbered, unit-labeled axes, gridlines, an optional min–max band, a
  curve legend, and a time-vs-frame horizontal axis choice. The two operations
  that read every frame — *Auto Range: All Frames* and *Compute Field Over
  Time* — share a memoized per-frame result keyed on the scan settings and PVD
  file, so running one right after the other (or re-running) is instant; it
  invalidates on a settings/file change and is bounded so a very large
  sequence is recomputed rather than pinned in memory.
* Matching legends can be shown as renderable scene geometry, a screen-fixed
  viewport overlay, or both. The Scene Geometry legend is real geometry and is
  always visible (and renders); the Viewport Overlay is screen-fixed but is
  drawn by the node's viewer state, so it shows only while that state is active.
  Selecting an overlay mode (or enabling the probe) activates the state
  automatically — no need to press Enter in the viewport — and a *Show Overlay
  In Viewport* button re-activates it; it persists through navigation until you
  select another node or change tools. For an always-on legend regardless of
  selection, use Scene Geometry. Legends support optional units and automatic,
  scientific, engineering, or fixed number formatting. A separate renderable
  XYZ gnomon is also available. Higher-order Lagrange tets are subdivided for
  display using vectorized index tables — tetra10/20/35 (P2/P3/P4) become
  8/27/64 P1 sub-tets that capture the element curvature, with all point fields
  carried onto the sub-tets (the 0.26 hand-built subdivisions, regenerated
  verbatim and vectorized). Orders above P4 fall back to the corner tet; for
  those, PolyFEM's `vismesh_rel_area` sampled output covers visualization.
* Playbar mapping from `.pvd` timesteps, plus a calculated-frame cache. The
  cache stores topology, positions, imported PVD/fiber attributes, and enabled
  derived mechanics. Deformation display, reference comparison, color,
  smoothing, visibility, glyph, fiber, and clipping controls remain live
  downstream of it.

## Performance summary

| operation | old | new |
| --- | --- | --- |
| MSH import, 1M tets | 11 s (+gmsh install) | **1.1 s** |
| MSH parse only, 1M tets | — | 0.45 s |
| readPVD frame change, 1M tets | ~30–60 s (meshio + python loops) | **0.25 s** |
| sideset export | per-prim Python loops | numpy bulk |

## GUI checklist (needs human eyes)

1. Drop a `PolyFEM (Dev) 2.0` node; set binary + working dir; import a .msh —
   Entity-colored boundary surface appears; transform handle state works.
2. Add a sideset; pick faces with the group selector; also try a native
   `axis:+z:...` pattern. Add a Dirichlet BC. Run both launch buttons.
3. Drop a `Read PVD 1.0` node on the output; scrub the playbar; toggle
   deformation, the four glyph tensor choices, Contact block, scene legend,
   viewport overlay legend, and scene gnomon; try both Auto Range buttons;
   toggle field smoothing; click-probe a point and scrub; clip with glyphs on;
   on a multi-body result, toggle Visible Bodies to isolate/hide bodies.

### Provenance block (2026-09-14, RB-12)

* `build_params` appends `provenance` (producer `houdini`, `producer_version`,
  `asset` = type name and `.hdanc` file, `asset_version`, `asset_sha256`,
  `scene`, `exported_at`) to the exported JSON; `read_params` treats the key
  as represented (it is regenerated at the next export). The end-to-end test
  checks that the exported file carries it and that the solver's manifest
  reproduces it as `producer`.

### Friction lag and budget (2026-09-13, RB-10)

* **Friction Iterations now defaults to 2** and the Semi-Implicit Options tab
  gained **Friction Lag** (Realized Force / Follow Stiffness), exported as
  `solver/contact/semi_implicit/friction_lag`. Friction is lagged one step
  behind the contact pressures; the RB-10 study measured that a single solve
  per step understates the friction work by a few percent on steady sliding
  and by about half when the load doubles each step, and that rescaling the
  lagged pressures with the barrier's trim (the historical behavior) can
  double the friction capacity for the rest of a step after a trim bump. Both
  controls carry tooltips; an untouched old scene now exports the new
  defaults, and importing a params.json that names the old settings restores
  them.

### Contact resource limits (2026-09-12, RB-05)

* **New on the Contact ▸ CCD Parameters tab: Resource Limits** (Automatic /
  Off / Custom) with *Max Grid Items* and *Max Candidate Pairs* for Custom.
  Before every trial step PolyFEM lists the surface-element pairs that might
  touch; with the Hash Grid or Brute Force broad phase that list can need
  gigabytes when a few points move very far in one Newton trial, and the OS
  then kills the run without a message (macOS never reports an allocation
  failure to the program). *Automatic* (the default, exported as `-1`) uses
  PolyFEM's built-in ceilings — 100 million grid items (≈ 2.4 GB) and
  50 million candidate pairs per pass (≈ 1.75 GB), thousands of times above
  what real scenes need — when the broad phase can enforce them; with BVH
  (this node's default broad phase), Spatial Hash or Sweep and Prune they are
  skipped with a note in the log, and BVH does not have the memory problem in
  the first place. *Off* exports `0`, *Custom* the two values. Every tooltip
  explains this for a first-time user; the Broad Phase tooltip now says which
  methods can blow up.
* **PolyFEM exit statuses are meaningful now.** A reached ceiling stops the
  run with exit status **3** and a plain-language explanation at the end of
  the log (steps already written are kept: reduce the time step or load
  increment, switch to BVH, or raise the ceiling); any other named failure
  (invalid input, a solver that stopped as configured, an option the binary
  lacks) exits **1** with `PolyFEM stopped: …`; an abort signal (−6 / 134) is
  a real crash. Use *Show Log* on the node to read the explanation.
* The end-to-end test checks the automatic default in the exported
  `params.json`, the Custom/Off round trips and that the new controls carry
  tooltips.

### Solver panel completion (2026-09-12)

* **The wired controls were not actually reachable.** The 2026-09-11 wiring
  inherited the 1.2 asset's visibility flags: the whole *Linear* tab, the
  *Line Search* tab and a hidden *Advanced* tab (the gradient finite-difference
  check) were `invisibletab`, and 21 individual controls were `invisible` —
  every per-method setting of the nonlinear solver (Newton residual tolerance
  and regularization weights, the PSD toggles, L-BFGS history, ADAM, erase
  probability) plus iterations per strategy, allow-out-of-iterations, the AL
  phase iteration limit, lagged regularization and the Jacobian threshold.
  Their values were exported, but only ever at their defaults. All are visible
  now; the hidden gradient-check tab was merged into Solver ▸ *Advanced* under
  a *Debugging* heading, and two hidden output toggles (*High Order Mesh*,
  *Jacobian Validity*) are visible as well. The end-to-end test now fails if
  any control or tab in the Solver folder is hidden.
* **Choosing a solver the binary lacks no longer aborts the run.** PolyFEM
  validates the linear solver against the list compiled into the binary;
  before, picking AMGCL, Pardiso or any `Eigen::Pardiso*` on a build without
  them stopped at startup with `invalid input json`. The exporter now writes
  `enable_overwrite_solver: true` with any non-automatic choice, so PolyFEM
  logs a warning and falls back to its default instead (the tooltip says so).
  The test exercises this with AMGCL on the reference build.
* **Hypre** gained *Strength Threshold* (theta) and *Nodal Coarsening*, and
  the exporter always writes `dimension: 3`: without it PolySolve builds
  scalar AMG for a 3-D elasticity system. On the quasistatic smoke, dimension 3
  with nodal coarsening halved the first solve (0.81 s vs 1.50 s) and cut the
  "large linear solve residual" events from 9 to 3, so *Nodal Coarsening*
  defaults on (PolySolve's own default is off). *Pre Max Iterations* is kept
  for compatibility; its tooltip now says the wrapper does not use it.
* **Nonlinear stopping criteria completed:** *Gradient Norm Type* (Euclidean /
  L2 / Linf), *Relative Gradient Norm Tolerance*, *Relative X Delta
  Tolerance*, *Newton Decrement Tolerance* and *Allow Non-Gradient
  Convergence* — every top-level key of PolySolve's nonlinear spec except
  `box_constraints` (optimization only). Augmented Lagrangian gained *Mass
  Lumping* (row_sum / hrz).
* **AMGCL** type menus, previously one entry each, now list AMGCL's runtime
  catalog (solver: cg, bicgstab, bicgstabl, gmres, lgmres, fgmres, idrs;
  smoother: chebyshev, gauss_seidel, damped_jacobi, spai0/1, ilu0/k/t;
  coarsening: smoothed_aggregation, aggregation, ruge_stuben,
  smoothed_aggr_emin). The Chebyshev-only settings hide for other smoothers
  and the coarsening controls moved into their previously empty group.
  Untested beyond export: the reference binary is built without AMGCL.
* The importer restores all of the above (menu tokens by name; `dimension`
  and `enable_overwrite_solver` are not controls).
* **Adaptive dhat / Min dist ratio moved to the GCP tab.** PolyFEM passes
  `use_adaptive_dhat` and `min_distance_ratio` only to the smooth (GCP)
  contact form; on the Barrier (ICP) tab they were a silent no-op. They now
  sit under *Geometric Contact (GCP)* next to the angle controls, disabled
  unless GCP is on, and the four GCP-only keys (`alpha_n`, `alpha_t`,
  `use_adaptive_dhat`, `min_distance_ratio`) are written to `params.json`
  only when `use_gcp_formulation` is true. The adaptive dhat is computed once
  from the rest shape (per element: min of dhat and ratio × rest distance to
  the nearest other surface); the tooltips say so.
* Validation: the 13 HDA test scripts pass; the end-to-end test additionally
  exports and runs Hypre (dimension 3), runs an unbuilt solver through the
  fallback, and round-trips the new controls through import.

### Solver wiring, new contact controls and tooltips (2026-09-11)

* **Every parameter now has a tooltip** written for a first-time user (what the
  control does, when to change it, units and typical values) on all three
  assets — 336 controls on PolyFEM 2.0, 133 on readPVD 1.0, 3 on MSH Reader
  3.0. Houdini's stock Transform/Render/Misc object parameters keep Houdini's
  own documentation.
* **The Solver folder is fully wired.** The linear solver (Automatic default,
  direct/iterative choice, preconditioner, per-solver iterations/tolerance,
  Pardiso matrix type, Hypre and AMGCL settings), the nonlinear method
  (Newton/Dense Newton regularization and PSD toggles, L-BFGS history, ADAM,
  stochastic erase probability), iterations per strategy, allow-out-of-
  iterations, the finite-difference gradient check, lagged regularization,
  inversion detection and the Jacobian threshold are now exported to
  `params.json` and restored on import. Previously these controls existed in
  the UI but were never read; they also carried invalid defaults
  (`residual_tolerance` 1e200, regularization weights −1) that were replaced
  by the PolyFEM spec defaults, so an untouched scene exports the same run as
  before. The AL folder's *Max Iterations* maps to the AL phase's own
  `nonlinear.max_iterations`. The line-search menu lost `ArmijoAlt` and
  `MoreThuente`, which this PolyFEM rejects; imports of old files map them to
  Armijo/Backtracking with a note.
* **New semi-implicit controls** (Contact ▸ Barrier ▸ Semi-Implicit Options):
  *Force Continuation* (default on) carries each contact's stiffness across
  steps so contact forces are continuous at step boundaries; *Continuation Max
  Ratio*; *Coefficient Identity* (parent candidate vs. historical stencil);
  *Trial Displacement Cap*; *Minimum Contact Stiffness*; and the stall
  restart's *Min Iterations* and *Stall Trim Factor*. The line search gained
  *Roundoff Tolerance* (PolySolve's energy-roundoff fallback). Rayleigh damping
  can now target friction.
* **Removed four dead controls** that had no PolyFEM key in this fork:
  `Force?` (AL), the `Traction Force` and `Pressure` output toggles, and the
  `Restart?` toggle (restart-from-state is not implemented for the elastic
  form; *Restart JSON* output remains). A duplicate PSD-projection toggle in
  the Solver folder was merged into the one under Nonlinear Settings.
  *Displaced Normals* is now honored on its own (before it only came along
  with *Normals*).
* Validation: the 13 HDA test scripts pass, including the end-to-end run of
  the real `PolyFEM_bin` with the fully exported solver block.

### Constraint floor retired (2026-09-07)

The Constraint Floor control and JSON export have been removed. Old saved/spare
node values are not read during export; imported JSON with a nonzero floor
produces a compatibility note and drops the setting on export. PolyFEM also
accepts the old key as ignored compatibility data, with a warning for nonzero
values. It cannot reactivate barrier deletion or direction projection.

Refresh Asset Libraries or restart Houdini to load the rebuilt definition.
CCD and the separate trial-displacement cap remain active. Retiring the floor
removes its known force/energy and prescribed-DOF defects; it does not certify
all remaining contact coefficients or whole-scene physical accuracy.

Validation: all 13 HDA test scripts passed, including legacy-floor import/export
and a leftover positive spare parameter. Solver validation passed 22 focused
cases / 1,189 assertions, five standard smokes and a legacy-positive input smoke.
