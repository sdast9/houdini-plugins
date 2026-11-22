# PolyFEM 2.0 HDA module.
#
# Targets the semi-implicit-barrier PolyFEM build (strict JSON validation):
#   * barrier_stiffness: "semi_implicit" | "adaptive" | number
#   * /solver/contact/semi_implicit/* and augmented_lagrangian "hessian_scaled"
#
# Engineering changes vs 1.2:
#   * collision-free ID scheme: volume id = 1000*geo + vol,
#     boundary id = (1000*geo + vol)*10000 + sideset*100 + bc,
#     obstacle id = 100000 + 1000*geo
#   * sideset export is numpy-bulk from VEX-stamped face provenance
#     (msh vertex ids stored per boundary face at build time); separate
#     files per face arity so mixed tet/hex meshes export correctly
#   * one Entity-colored surface chain per geometry instead of per-volume
#     node forests (scales to millions of elements); each subdomain's sideset
#     group nodes sit on a dead-end branch behind a Visibility SOP that hides
#     the other subdomains, which is what 1.2's per-volume split branches gave
#     for free -- picking stays scoped to one subdomain without renumbering
#     the geometry-wide prims that basegroup and the export refer to
#   * native polyfem selections can be typed into any sideset group field:
#     "axis:+z:0.95", "box:[0,0,0],[1,1,1]", "sphere:[0,0,0],0.5",
#     "plane:[0,0,1],[0,0,0.5]"  (normal, point) -- emitted directly into
#     the json, no selection files needed
#   * ast.literal_eval instead of eval for vector parms
#   * background run with log capture, alongside the terminal launch

import ast
import filecmp
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

import hou

# =============================================================================
# VEX snippets (embedded into wrangles by the node-tree builders)
# =============================================================================

# Extract boundary faces of tets/hexes with provenance: per face we stamp the
# gmsh vertex ids (msh_pt_id - 1, polyfem's 0-based node ids) so sideset
# export is a bulk attribute read instead of per-prim python. Volume prims
# are removed afterwards -- this wrangle's output is the display/selection
# surface only.
SURFACE_VEX = """
if (@primnum == 0) {
    addattrib(0, "prim", "fp0", -1);
    addattrib(0, "prim", "fp1", -1);
    addattrib(0, "prim", "fp2", -1);
    addattrib(0, "prim", "fp3", -1);
}

int vcount = primvertexcount(0, @primnum);
if (vcount == 8) {
    for (int f = 0; f < 6; f++) {
        if (hex_adjacent(0, @primnum, f) == -1) {
            int pts[]; int ids[];
            for (int j = 0; j < 4; j++) {
                int p = primpoint(0, @primnum, hex_faceindex(f, j));
                append(pts, p);
                append(ids, point(0, "msh_pt_id", p) - 1);
            }
            int prim = addprim(0, "poly", pts[0], pts[1], pts[2], pts[3]);
            setprimattrib(0, "Entity", prim, i@Entity, "set");
            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");
            setprimattrib(0, "fp0", prim, ids[0], "set");
            setprimattrib(0, "fp1", prim, ids[1], "set");
            setprimattrib(0, "fp2", prim, ids[2], "set");
            setprimattrib(0, "fp3", prim, ids[3], "set");
        }
    }
} else if (vcount == 4) {
    // Faces in the order 0,2,1,3: the MSH Reader builds tets as Gmsh corners
    // [1,2,3,0] (positive) since 2026-11-15, [1,3,2,0] (inside out) before;
    // this order emits the same faces in the same order as before, so the
    // surface prim numbers saved in sideset picks keep their faces.
    int tet_faces[] = array(0, 2, 1, 3);
    foreach (int f; tet_faces) {
        if (tet_adjacent(0, @primnum, f) == -1) {
            int pts[]; int ids[];
            for (int j = 0; j < 3; j++) {
                int p = primpoint(0, @primnum, tet_faceindex(f, j));
                append(pts, p);
                append(ids, point(0, "msh_pt_id", p) - 1);
            }
            int prim = addprim(0, "poly", pts[0], pts[1], pts[2]);
            setprimattrib(0, "Entity", prim, i@Entity, "set");
            setprimattrib(0, "geometry_num", prim, i@geometry_num, "set");
            setprimattrib(0, "fp0", prim, ids[0], "set");
            setprimattrib(0, "fp1", prim, ids[1], "set");
            setprimattrib(0, "fp2", prim, ids[2], "set");
        }
    }
}
removeprim(0, @primnum, 1);
"""

# Per-prim minimum edge length (prim wrangle; parallel-safe). The button
# callback reduces over the prim attribute with numpy.
MIN_EDGE_VEX = """
float m = 1e30;
int pts[] = primpoints(0, @primnum);
int n = len(pts);
for (int i = 0; i < n; i++) {
    vector a = point(0, "P", pts[i]);
    vector b = point(0, "P", pts[(i + 1) % n]);
    m = min(m, distance(a, b));
}
f@min_edge_length = m;
"""

