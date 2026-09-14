# G1 current packets to local sparse rows

Experimental, default off; not a measured improvement. Base `4a0bf739`
retains sparse43 and corrected finite/seam/packed collision. The separate
coordinate-register and Gram losses remain closed.

Fresh same-source complete FPGS/MJ costs are 22.861354/40.869610 ms RTX and
27.668025/44.941774 ms GB. Current three-step attribution gives sparse rows
plus GS 9.103631/8.588925 ms exclusive. The first milestone requires at least
2.286135 ms RTX whole-step saving: complete replacement plus retained row
services must fit 6.817495 ms. This does not reach the 4x ceilings
10.217402/11.235444 ms; collision and other dynamics still require removal.

Replace global per-row Z/incident with current raw-contact/direction or limit
keys in existing private support storage. Retain original allocation,
reservation order, metadata, bias, capacity and public-force ownership.
Within the existing one-warp/world solve, form each contact triplet or limit
column once into row-major shared Z[100][18], then execute the original
sparse-coordinate PGS, current-friction sibling law, eight-sweep allowance,
early stationarity and W-transpose decode. The held W/current geometry and
force distinction is unchanged. Unsupported constructors remain original;
changed ownership after admission retains the existing reconstruct/recapture
error contract.

The existing prefix and contact producer dispatches become key producers.
No new queue, maximum allocation, scan or solver dispatch. No dense43 rows,
Gram matrix, static unrolled row recurrence or per-visit W reconstruction.
Separate restitution is folded into row initialization. Global Z and incident
are retired; public row metadata and impulses remain canonical.

Old GS loads global Z every visited sweep and again on sibling correction.
At the current RTX 329356-row boundary, 18 slots imply 23.714 MB of logical
row traffic per complete sweep/substep, up to 189.709 MB for eight sweeps,
plus sibling loads. These are upper logical requests, not measured DRAM
traffic: shorter supports, early exit and caches reduce physical traffic.
Local staging changes those global loads to shared loads, not zero work.
Formation still performs all current sparse contact whitening arithmetic.
Expected shared storage is approximately 9 KB/world versus the old 700 B;
lost residency and serialized contact setup are explicit risks.

Use existing actual-tree current/held/action, full-step continuing graph,
reservation and loaded boundary physical controls, with original physical
eight as the oracle. No coefficient-bit or new certification framework.
Then one complete paired16K 200/40/40 A/B using the existing whole owner,
all flags/capacities unchanged, before any attribution or corrective change.
Native plus first paired physical checkpoint: 02:15 UTC, 2026-09-14.

Current attribution source:
`/tmp/fpgs-g1-finite-seam-current-nodes-paired16k-20260914-01`.
The original parent remains failed solely for its inherited auxiliary-graph
analyzer refusal. Existing strict reader `74c5928b`, with only three exact
finite owner aliases, retains all 12 physics and three auxiliary roots,
memory and overlap. These nodes are diagnostic, not throughput acceptance.

## CPU/native checkpoint, 01:20 UTC

The integrated candidate is ready for paired physical review; it has no GPU
physical or whole-step timing result yet. Enable with both
`FEATHER_PGS_SPARSE_FACTOR=1` and `FEATHER_PGS_SPARSE_PACKETS=1`. The admitted
owner exposes `packet_rows=True` only at the funded capacity of 100 rows;
other constructor capacities retain the original sparse rows. Packet storage
is `(worlds, capacity)`;
`W` remains `(worlds, 434)` with original per-world valid/status arrays.
Global `Z` and incident are allocated directly as `(1, 1, 1)` and `(1, 1)`
dummies, so constructing the packet owner does not allocate and discard the
old global sparse-row panels. Contact reservation, metadata and all other
retained services keep their original owners and allowances.

The existing dispatches select `sparse_packet_prefix43`, `packet_contacts`
and `sparse_packet_gs43_s18_c100`. Current contact or limit keys become sparse
shared rows once at the start of the existing solve. Both native targets
compiled ahead of time with CUDA hidden, without launching GPU kernels.
Compiler artifacts are in `/tmp/fpgs-packet-native-20260914-czj2z4xj`.
The actual solve cubins use 11,020 B static shared memory, 63 registers on
SM100 and 64 on SM120, and no local/stack storage. The static shared figure
includes compiler overhead beyond the 8,972 B of declared local-row arrays;
the card's residency risk therefore remains explicit.

Nine CPU tests pass across `TestSparsePacketRows` and `TestSparseFactor`.
They include actual G1 current geometry and held W, an 18-entry dynamic-body
union with shared anchors, 100 rows, empty reuse, delayed friction siblings,
status-preserving no-publication, the original restitution end-gap slop, and
fallback to original sparse rows for an unsupported capacity of 128.
An in-memory regression control that replaced the original `1e-6` native
slop with zero failed the physical-eight velocity check with scaled error
0.741894718; the unchanged source passes. An independent read-only review
found no actionable geometry, recurrence or lifecycle discrepancy.
`uvx pre-commit run -a` passes. No Isaac Lab source or pointer changed.

The existing paired physical owner remains
`/tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py` (SHA256
`155eba000c50e28e5e88522897f89aae519b92d9b6de0b7c97ddb2a95e156e0c`),
with `/tmp/fpgs-heightfield-finite-physical-kzU2tu/run_checks.py` (SHA256
`027da7a49d25bad8e83e94e1ecad87f9a773c5cf663f0d19841a22d2d7d6859e`).
Use its existing explicit selectors:

- `tools.fpgs_bench.test_sparse_factor:TestSparseFactorCUDA.test_actual_operator_current_rows_and_held_reuse`
- `tools.fpgs_bench.test_sparse_factor:TestSparseFactorCUDA.test_complete_owner_two_steps_and_graph`
- `tools.fpgs_bench.test_sparse_packet_rows:TestSparsePacketCUDA.test_current_geometry_held_and_packet_lifetime`
- `tools.fpgs_bench.test_sparse_packet_rows:TestSparsePacketCUDA.test_full_packet_allowance`

The root-owned paired invocation is below; supply the frozen implementation
commit and the existing wrapper's complete source/fixture pin JSON before
launching under the paired GPU lease. The output directory must be fresh.

```sh
FEATHER_PGS_SPARSE_FACTOR=1 FEATHER_PGS_SPARSE_PACKETS=1 \
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python /tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py \
  --newton /home/octi/Projects/newton-fpgs-g1-sparse-packet-20260914 \
  --wrapper /tmp/fpgs-heightfield-finite-physical-kzU2tu/run_checks.py \
  --output <fresh-paired-output> -- \
  --revision <frozen-commit> --pins-json <root-owned-pins-json> \
  --test tools.fpgs_bench.test_sparse_factor:TestSparseFactorCUDA.test_actual_operator_current_rows_and_held_reuse \
  --test tools.fpgs_bench.test_sparse_factor:TestSparseFactorCUDA.test_complete_owner_two_steps_and_graph \
  --test tools.fpgs_bench.test_sparse_packet_rows:TestSparsePacketCUDA.test_current_geometry_held_and_packet_lifetime \
  --test tools.fpgs_bench.test_sparse_packet_rows:TestSparsePacketCUDA.test_full_packet_allowance
```

The first selector retains original refresh/current-force/held checks and
uses independently rebuilt physical rows for packet mode, including changing
77/83-row limit/contact prefixes. The second retains original full-step
continuing graphs and both serial and parallel drive controls. Subsequent
paired whole16K timing must retain the original 200/40/40 protocol and fixed
recipes. The 02:15 UTC checkpoint and 6.817495 ms RTX complete-family
stop/go budget remain unchanged; native compilation is not acceptance.
