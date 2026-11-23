# Analytical two-body traction in Houdini

Run the suite with Houdini's `hython`, an explicit PolyFEM executable and
fresh preferences. The script creates native tetrahedral SOP geometry,
resolves all three HDA definitions from this checkout in the headless
session, and runs the solver through the PolyFEM node's background runner.
The HDA stages the inputs and Read PVD opens the original results.

```sh
oracle_run=$(mktemp -d /tmp/hda-simulation-oracles.XXXXXX)
HOUDINI_USER_PREF_DIR="$oracle_run/prefs/__HVER__" \
HOUDINI_PATH='$HOUDINI_USER_PREF_DIR:$HFS/houdini' \
HOUDINI_MAXTHREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
hython tests/test_simulation_oracle_pipeline.py \
  --solver /absolute/path/to/PolyFEM_bin \
  --expect-commit FULL_SOLVER_COMMIT \
  --output "$oracle_run/results"
```

`sop_order_ab` and `sop_order_ba` each create two disconnected 1 m unit
tetrahedra in one SOP. They differ in primitive order. Physical
entities 17 and 53 correspond to HDA body IDs 1001 and 1002.
The test checks the staged Gmsh element order, volume assignment,
boundary face vertex sets, material constants, SI units, and the
Neumann ramp table. It checks each subprocess's executable hash
and build identity in its manifest.

Both base faces are fixed. The sloping faces also constrain y and z,
leaving one free apex x degree of freedom per body. At `t = 1 s`,
uniform x tractions on the faces at the bodies' minimum x coordinates
are `+144 Pa` and `-180 Pa`. Each face has area `1/2 m²`. The
analytical integrated load per corner is `T area/3`, so the apex
loads are `+24 N` and `-30 N`.

The shear moduli are `12 kPa` and `20 kPa` from `E = 30 kPa` and
`50 kPa` at `nu = 0.25`. For vertex spacing `L = 1 m`, the only
engineering strain is `u_x/L`, the stiffness at the apex is
`G V/L²`, and the shear stress is `G u_x/L`. The base supplies
the negative of the full traction resultant, `-3 F_apex`, in x.
The ramp begins at zero and at `0.5 s` each response is half of
its endpoint value.

Both cases verify actual `OUT_result` geometry and Read PVD's
force-curve data. Houdini 22.0.508 and PolyFEM
`4c5c45aa40f9350e4135ff0f1cdd811072395003` gave these endpoint
values in both primitive orders.

| Body ID | Apex x displacement (mm) | Cauchy XZ (Pa) | Apex elastic x (N) | Base reaction x (N) |
| --- | ---: | ---: | ---: | ---: |
| 1001 | 12.0000001 | 144 | -24 | -72 |
| 1002 | -8.99999961 | -180 | 30 | 90 |

Across all frames, displacement error was at most `3.88e-10 m`,
and force-curve error was at most `1.43e-14 N`. Cooked stress and
nodal forces matched the analytical values to the stored precision.
The test allows `3e-8 m`, `3e-5 Pa` and `3e-5 N` for the
single-precision SOP attributes, and `1e-8 N` for force curves.
Swapping the material constants would give `+7.2 mm` and `-15 mm`.
This suite covers the constant isotropic materials, face loads and
separate support reactions exposed by the HDA. The HDA Material Axes
field accepts numerical rotation matrices. Run the matrix-expression
case with the solver's CLI reproducer.

## Transient nodal load and checkpoint continuation

Use the command above with `tests/test_transient_load_restart.py` and a new
output directory. The test uses one native P1 tetrahedron with its base fixed
and its apex free in x. The PolyFEM node's Nodal Force condition applies a
linear ramp from 0 to 30 N in 0.625 s at the apex only. The test verifies the
exported node selection, load, interpolation, Implicit Euler settings, and
physical units before comparing the output. The node exports a checkpoint at
each 0.125 s step.

The shear modulus is 12 kPa and the apex stiffness is 2,000 N/m. The
consistent mass matrix has 2 kg on the diagonal and 1 kg off the diagonal.
Row-sum lumping gives 5 kg on the diagonal. The analytical step solves
`m (q_next - q - h v)/h² + K q_next = 48 t_next N`. The discrete velocity
and acceleration are the backward differences of position and velocity.
The initial state is at rest. The expected inertia on each vertex is
`-M_ia a`.

Two fresh HDA nodes resume their respective mass cases from the checkpoint
at 0.25 s. Each resumed PVD keeps the global frame indices. Set Read PVD's
Topology Frame to the first emitted frame, step 2, since the files for
steps 0 and 1 are in the original output directory. The test uses that
existing reader control and reads the unedited result files through the
`OUT_result` SOP.

Both original and resumed series are checked against the same analytical
displacement, velocity, acceleration and three nodal force fields. At the
repeated checkpoint frame, expected apex inertia is `-0.043416813 N` for
consistent mass and `-0.228299643 N` for lumped mass. The comparisons use
absolute tolerances of 3e-8 m, 3e-7 m/s, 3e-6 m/s² and 3e-5 N for the
single-precision SOP attributes.
