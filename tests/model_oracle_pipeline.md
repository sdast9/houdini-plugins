# Analytical PolyFEM and Read PVD checks

Run these checks with Houdini's `hython`, a built PolyFEM solver and a clean
preference directory. Run from the Houdini plugins checkout. Use the absolute
path to the solver and its full Git commit as reported by `--build_info`.

```sh
oracle_run=$(mktemp -d /tmp/hda-model-oracles.XXXXXX)
HOUDINI_USER_PREF_DIR="$oracle_run/prefs/__HVER__" \
HOUDINI_PATH='$HOUDINI_USER_PREF_DIR:$HFS/houdini' \
HOUDINI_MAXTHREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
hython tests/test_model_oracle_pipeline.py \
  --solver /absolute/path/to/PolyFEM_bin \
  --expect-commit FULL_SOLVER_COMMIT \
  --output "$oracle_run/results"
```

The test creates native tetrahedral geometry in a fresh scene using Python SOPs.
It loads the PolyFEM, MSH Reader and Read PVD libraries from this checkout into
the headless session and checks the resolved definition paths. Each case stages
the SOP geometry, writes the scene through PolyFEM's HDA, starts the HDA's
background runner and opens its results in Read PVD. The run directory contains
the exported inputs, solver logs, manifests and result data. `identity.json`
records the resolved solver and HDA hashes. The harness uses a single
quasistatic increment with contact and body load disabled and SI units enabled.

## Dispersed fibers under shear

Three disconnected unit tetrahedra share the same material. Their primitive
directors are `+ey`, `ex` and `-ey`, respectively. The HDA maps the primitive
attribute to the `FIB_1_1` cell field in `fibers.vtk`. Every displacement is
prescribed as `u = (0.25 y, 0, 0)`, which gives `Fxy = 0.25` and `J = 1`.

With `kappa = 0`, `s = |Fa|² - 1`, `chi = 1 / (1 + exp(-100 s))`, and the
energy density is `W = 10 chi expm1(5 s²)` in J/m³. Differentiating gives
`P = 2 (dW/ds) (Fa) a^T` and `sigma = P F^T`. For either transverse director,
`s = 0.0625`, `W = 0.1968523127763 J/m³`, and the nonzero entries of `P` are
`Pxy = 3.199459410649 Pa` and `Pyy = 12.797837642595 Pa`. The Cauchy entries
are `sigma_xx = 0.799864852662 Pa`, `sigma_xy = sigma_yx = 3.199459410649 Pa`,
and `sigma_yy = 12.797837642595 Pa`. For the axial control `s = 0` and both
stress tensors and the energy are zero. The two transverse fiber axes are
equivalent up to sign. Their current unit director is parallel to `(0.25, 1, 0)`.

The test checks the exported field, the raw VTU energy, and Read PVD geometry
attributes for displacement, energy and both stress tensors. It checks the
reference axis on the mesh and the current axis and `I4bar` on the fiber glyphs.
The components of first Piola stress distinguish a tensor from its transpose.

## Orthotropic Hooke shear with a free interior node

A unit tetrahedron is split around its centroid. Its exterior displacement is
prescribed as `u = (0.01 z, 0, 0)`. Constant material, zero body load and an
affine displacement imply constant stress and an equilibrium solution. The
one interior node is free and should move `0.0025 m` along x. The global XZ
engineering shear is `gamma = 0.01`. A 90° rotation about global z puts this
shear in the material YZ plane. With the rotated axes, `Gyz = 3.8 MPa` gives
`sigma_xz = 38,000 Pa` and `W = 190 J/m³`. With aligned axes, `Gzx = 8.3 MPa`
gives `sigma_xz = 83,000 Pa` and `W = 415 J/m³`. These follow from
`sigma_xz = G gamma` and `W = G gamma² / 2`.

The test checks the exported orthotropic constants and axes, the raw VTU
energy, and the displacement, energy, Cauchy tensor and nodal elastic force
on Read PVD geometry. At the original vertex `z=1`, the integral of its shape
gradient is `e_z/6 m²` because its boundary trace equals `z`. PolyFEM
exports the negative energy gradient, so the apex force is `(-tau/6, 0, 0)`
in N. The rotated and aligned cases give `-6333.333333 N` and
`-13833.333333 N` in x. Input body IDs, sideset face IDs and SI units are
checked in all three cases.

The readback checks allow `5e-8 m` for displacement, `5e-7` for unit-axis
distance and `I4bar`, `3e-5 Pa` and `2e-6 J/m³` for fiber stress and energy,
and `0.05 Pa`, `2e-3 J/m³` and `5e-3 N` for Hooke stress, energy and force.
These absolute tolerances allow for single-precision SOP attributes and
leave a wide margin to the orientation controls. Raw VTU energies use
double-precision tolerances.
