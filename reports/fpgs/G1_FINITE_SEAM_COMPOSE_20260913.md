# G1 current sparse/packed plus qualified terrain query

This experimental branch starts at current G1 `bd1cc095`, not the older
cell-only finite-query baseline. It composes the original finite owner,
positive-face fallback correction and flat internal-seam qualification.
The sole merge conflict retained BOTH the packed-pair and finite-query
constructor flags. Sparse dynamics, packed midphase, public-state ownership,
solver allowances, collision cadence and capacities remain unchanged.

Runtime composition checkpoint: `7e6cb1bd`; paired outcomes follow below.
Original Newton worktrees and fixed Isaac Lab53ee are unchanged.

## Loaded geometry gate on seam source2edf4b2b

Existing paired owner/loaded runner, three CUDA selectors/card, zero skips,
failures or errors, normal child exits and final source/idle guards passed.
Manifest: `/tmp/fpgs-heightfield-internal-seams-physical-paired-20260913-01/manifest.json`
SHA256 `2191cf3e5fc7ec9703ad30e5228a9899edcd2ed62ac40fc78e667ccc54acbc1a`.
Pins894 SHA256 `8d5f621d58ac1daa14a60fda7829cb20449066820f1dc22799f95c964d829b47`.
Original parent155eba was executed with one explicit in-memory WELD1
environment insertion; executed source SHA256
`6053ef59ceaa2d030561ef29748f2d70fc6f6d18a9cde3682b90375c3ea5acb5`.
Original parent file and runner969c26 were not edited.

At unchanged dt.0025 and eight sweeps, sliding stopping distances in metres:

| GPU | Corrected original query | Corrected finite query |
|---|---:|---:|
| RTX | .50720936 | .50719726 |
| GB300 | .50720692 | .50719184 |

The authored continuous Coulomb prediction is .509684m; the corresponding
semi-implicit timestep prediction is approximately .507185m. Previous
unfiltered queries stopped prematurely at .337–.408m in these controls.
All support, tilted support, sliding and finite-border physical gates pass.
Maximum scaled momentum mismatch is7.24e-7; force vanishes beyond the border.
Three same-state rebounds per variant/card give energy ratios .359944–.360000
for authored restitution.6, normal velocity1.799859–1.800000m/s, spin<=.009844,
corner normal residual<=.001401m/s, angular impulse closure<=4.01e-8.
This removes the known sliding blocker, not a general convergence theorem.

## Composed physical check and complete cost discovery

Composed source6e08 passes saved96 current geometry/lifecycle and rebound on
both GPUs. RTX also passes all eight reduced loaded cases. GB finite support
fails the original mean-force check once: five decimated tail samples average
21.603198N versus22.0725N weight (-2.12618%, limit2%). One sample captures a
three-contact transient. Preserve failed manifest
`/tmp/fpgs-g1-finite-seam-compose-physical-paired-20260913-01/manifest.json`,
SHA256 `adfd53e1110c0d06d41b7239aa0271ade31ec295dd1d5c2eea275a8be72c8053`.
Do not attribute this specifically to packing: the older2edf lineage also
predates the shared contact-slot reservation correction. Finite/WELD geometry
is byte-identical; sparse43 is not active in the small loaded fixture.

The subsequent complete timing was explicitly cost discovery while that
support check remained open. BOTH arms use source6e08, SPARSE1/CELL1/PACK1/
WELD1, fixed Lab53ee and original16K200/40/40 protocol/capacities. Only FINITE
changes0->1. There is no comparison against slower register experiments or
against an uncorrected collision arm.

| GPU | Corrected original physics | Finite physics | Physics ratio | Original wall | Finite wall |
|---|---:|---:|---:|---:|---:|
| RTX | 24.714743ms | 22.905959ms | 1.078966x | 38.500910ms | 38.089658ms |
| GB300 | 38.007732ms | 27.702692ms | 1.371987x | 52.684721ms | 41.359652ms |

One discovery round, not repeated timing acceptance. All four normal child
exits, current owner activation, sparse valid16384/status0, capacity and final
source/idle checks pass. Manifest
`/tmp/fpgs-g1-finite-seam-compose-live-paired16k-20260913-01/manifest.json`,
SHA256 `52edb41a4d847ab096fa7a3360f03dd314e440b3248227b2d70128172d18c78d`.
Versioned existing three-file helper aa354e4c/29b1e45a/df441c4b advances the
checked-capture pin3bea->42 and reads actual four feature booleans at the same
two untimed boundaries. Old helpers and their measured inputs remain intact.

## Direct support-history diagnosis

Test-only f37595ef adds all80 tail steps161–240 to the EXISTING loaded selector,
calling original public_force exactly once. Runtime and original failures,
five-sample calculation,2% tolerance and final assertion are unchanged.
The repeated eight loaded cases pass on both cards, with actual full-tail mean
force relative error<=1.508e-5 (0.001508%) and momentum closure<=6.01e-8.
GB finite records a2.5ms three-contact event and5ms force excursion; peak spin
.1153rad/s recovers to.000043rad/s. Do not hide the real short transient.

Manifest `/tmp/fpgs-g1-finite-seam-support-tail-paired-20260914-01/manifest.json`,
SHA256 `6801124d0757f490c3e6b8b6c72557de7c2a5ae4a7c8c810e05f60bd2a7558bc`.
Direct full-history force balance supports sampling bias, rather than
persistent under-support; the first failed five-sample result remains above.
This is bounded loaded qualification, not all-task or training acceptance.

Finite/weld remain default-off. Next compare corrected MJWarp with the SAME
shared terrain changes and repeat the current timing if retained. No current
MJWarp ratio or4x-across-tasks claim follows from the FPGS-variant table.
