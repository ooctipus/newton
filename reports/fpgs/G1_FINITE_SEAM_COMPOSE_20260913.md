# G1 current sparse/packed plus qualified terrain query

This experimental branch starts at current G1 `bd1cc095`, not the older
cell-only finite-query baseline. It composes the original finite owner,
positive-face fallback correction and flat internal-seam qualification.
The sole merge conflict retained BOTH the packed-pair and finite-query
constructor flags. Sparse dynamics, packed midphase, public-state ownership,
solver allowances, collision cadence and capacities remain unchanged.

Runtime composition checkpoint: `7e6cb1bd`. No whole-physics gain yet.
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

The composed branch still needs its current saved-scene/lifecycle gate and
complete paired timing. Retain default-off finite/weld flags until then.
If retained, compare corrected MJWarp with the SAME shared terrain changes;
do not reuse an older MJWarp denominator or claim the old GB10.28ms saving.
