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
