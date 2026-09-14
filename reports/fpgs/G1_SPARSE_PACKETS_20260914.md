# G1 current packets to local sparse rows

Experimental, default off; measured whole-physics loss, not promoted. Base `4a0bf739`
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
analyzer refusal. Existing strict reader `74c5928b`, with only observed
finite owner aliases, retains all 12 physics and three auxiliary roots,
memory and overlap. These nodes are diagnostic, not throughput acceptance.

## CPU/native checkpoint, 01:20 UTC

At this checkpoint the integrated candidate was ready for paired physical
review, before the GPU results recorded below. Enable with both
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
The ahead-of-time solve cubins use 11,020 B static shared memory, 63 registers on
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

The executed paired invocation used the checked adapter below. The original
owner removes inherited `FEATHER_` variables, so outer shell flags alone are
insufficient. The adapter makes four recorded, reversible in-memory owner
substitutions: fixed Lab checkout, dependency pins, both flags in each
recorded child environment, and executed-source provenance in the manifest.
The original process owner and child files are unchanged. The output
directory must be fresh and the frozen tree/pins must match.

```sh
uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python /tmp/fpgs-g1-sparse-packet-physical-ready-Pg9XRHzK/run_packet_physical.py \
  --newton /home/octi/Projects/newton-fpgs-g1-sparse-packet-20260914 \
  --wrapper /tmp/fpgs-heightfield-finite-physical-kzU2tu/run_checks.py \
  --output <fresh-paired-output> -- \
  --revision 7ecb233dc11feab618c4d854cd87b6faf8d90cb2 \
  --pins-json /tmp/fpgs-g1-sparse-packet-physical-ready-Pg9XRHzK/pins.json \
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

## Reaped physical, whole-physics and node results

Frozen implementation `7ecb233dc11feab618c4d854cd87b6faf8d90cb2` passed all
four physical selectors on both devices (RTX 37.236 s, GB 13.648 s), without
skips, cleanup signals or failed source/capacity checks. Both children and
the parent were reaped, then both GPUs were idle. Fixed Isaac Lab was
`53ee6b44c2334341305dbdf385a3916c6b140799` at `contact-reset-20260913`.
Physical output: `/tmp/fpgs-g1-sparse-packet-physical-paired-20260914-01`.
Manifest SHA256:
`7aeaee70d28819def5305ee1db20badbbc7e051940d695d44b0f9aa0a60283e4`.
Executed parent SHA256:
`cd7f2db9f7e517b643b53d44ebc456cb86d4e282861ec7212556e900ea5c2bee`.
Adapter SHA256:
`efd166e1d0fca3a91251f6a0a513cbbe9fe016f8306e78a927a36da9678f12f1`.
Dependency pin-file SHA256:
`76274d27f607c459d7cb71997f67bb4aad1cb80ad1bf590cb80c3b4c2b7fc9dc`.

The existing paired16K 200/40/40 first whole-physics screen was a loss:

| Device | Same-source baseline | Packet | Delta |
| --- | ---: | ---: | ---: |
| RTX | 22.873005600 ms | 25.547120225 ms | +2.674114625 ms |
| GB | 27.789969050 ms | 28.111900800 ms | +0.321931750 ms |

Whole output: `/tmp/fpgs-g1-packet-live-paired16k-20260914-01`.
Manifest SHA256:
`91bc1e7b064b0e9f2d469aab6259b6ee9393010f00fe2b0f0f90151fc43e5680`.
Original source/capacity and final idle guards pass; all owners are reaped.
This first screen is not a speedup or promotion.

Candidate nodes: `/tmp/fpgs-g1-packet-nodes-paired16k-20260914-01`.
Manifest SHA256:
`6838113baa8956a213c81df8d822beeeac70a2811962f45cba44e6eb04cc0769`.
The original node parent remains failed for the inherited auxiliary
analyzer refusal. Root independently confirmed its original source guard,
source-bound packet/capacity observer and final idle guard; this does not
rewrite that parent status. The existing strict reader
`/tmp/fpgs-g1-sparse-finite-compose-ATSRjnR2/audit_sparse_nodes.py`, SHA256
`74c5928beff9685e341135728d6213d9febbdd8fd821e2e91dc6ae020191e589`,
accepted both baseline and packet traces. Only these observed owner-set
additions were made in memory; no process/correlation or interval law changed:

- Collision: `create_query_kernel__locals__heightfield_finite_contacts`,
  `mesh_triangle_contacts_to_reducer_kernel_finite_fallback`.
- Rows: `sparse_packet_prefix43`, `packet_contacts`.
- GS: `sparse_packet_gs43_s18_c100`.

The reader already recognizes the retained packed collision owners. Each
capture has 12 physics and three auxiliary roots, zero unproven nodes,
and all graph memory nodes retained. Physics nodes fall from 1,020 to 996.
Three-step exclusive busy time per environment step is diagnostic only:

| Family | RTX baseline | RTX packet | GB baseline | GB packet |
| --- | ---: | ---: | ---: | ---: |
| Rows and retained services | 4.180311 | 0.836139 | 3.816914 | 0.739657 |
| GS including local formation | 4.923319 | 11.000920 | 4.772011 | 8.213632 |
| Complete rows plus GS | 9.103631 | 11.837059 | 8.588925 | 8.953289 |

All entries are milliseconds. Complete-family deltas are +2.733428 ms RTX
and +0.364364 ms GB; other stages are nearly unchanged. Runtime resources
show 700 to 9,100 B static shared memory, 40 to 64 registers/thread RTX
and 37 to 63 GB, zero local memory, and unchanged 16,384 blocks of 32
threads. The runtime shared figure, not the default-block AOT metadata,
describes these launches. Longer per-world formation before GS and shared
state lifetime are source-backed hypotheses, not occupancy-counter claims.

Raw inputs are under `round_01_g1_fpgs_gpu{0,1}` in the baseline/current
and candidate node directories above. SHA256 pins, in file order
`capture.sqlite`, `capture.json`, `capture_checks.json`:

```text
baseline GPU0
fdd4e664be346bb0b9b56cf7f163f21e4745d72dac4a8c360f4dff88e096459c
838c893d7ff2edc40b0cfd52d31840f59c07b0ebfdc6a122d461ad8cfd33e451
97f4496b7326b978970a8d5199aac0a0b4d693f6f0c89a9b96e883a89b87b054
baseline GPU1
ecc83bd3ec26dce11bdfee3d8fcdf7cdac099c7fe218d1aebb9a20fbfaf26244
704946d1e7b246e61d5a4172834ade0de754cc0f1cbb92c05629d6082298ebf6
b4a93e76ec1a019581352ef9a07b462a9917685b4c493909561e1c2cecc31406
packet GPU0
b2c830bd746b22cd166c5d1cfc051bb32ab9fee7b8bfa22b5ddf397446ee6904
42f29de46c9f5f5afcb43fca1b724fba4f1085398ec85a186371b19ee80284c4
3524f8691ea62a5464423ab26a8ce43b8a6c526d2c7596f6f35440461c5a77e3
packet GPU1
c78485a01b9a353c4596ff491b29c06b182727325eb681ad9369d5704925c409
fb87e9e178386ebf2cb2c7f803c80cb455d422cd4e6d48651bb1541804eb4963
8a9f035aaa04bc208f7d65f1fcc6750121bf944586f478edfaaffbfb2f6d1a19
```

## Funded targeted correction: parallel limits, original global rows

One default-off correction now isolates the successful stable-ballot prefix
from the failed all-row shared lifetime. Retain original global Z/incident,
parallel raw-contact producer, restitution, eight-sweep GS and publication.
Replace only `build_limit_prefix` (2.250646 ms RTX / 1.811637 ms GB) with
three stable candidate ballots and independent active-lane emission of the
original 18-entry W column, incident, diagonal and complete row metadata.
No extra allocation, queue or dispatch; capacity and candidate order remain
original. New flag: `FEATHER_PGS_SPARSE_PARALLEL_LIMITS=1`, with sparse factor
enabled and packet mode disabled. The actual constructor marker is
`parallel_limit_prefix`; the kernel is `sparse_factor_parallel_limit_prefix43`.
Admission is limited to the tested capacity100; other capacities fall back.

The measured keys-only prefix costs 0.171157 ms RTX. Restoring its omitted
Z/incident/diagonal work must be charged: the maximum indicated saving is
2.079489 ms, not a promised 10% whole gain. A roughly 2 ms RTX whole saving
is the funded substantial correction target; the 10% milestone is a
prioritization aid, not a hard rejection threshold. Existing current/held,
continuing-graph, stable limit order and capacity controls precede one
complete paired16K screen. Stop if the integrated saving is not substantial;
no follow-on shared-memory layout sweep is funded. CPU/freeze checkpoint is
25 minutes from authorization, before any GPU lease.

Runtime capture device limits make the loss hypothesis concrete. Ignoring
allocation rounding/reservations, RTX has 102400 B shared/SM, 24 block slots
and 48 warp slots: old700 B permits the 24-block ceiling, while packet9100 B
permits at most11 blocks. GB has 233472 B, 32 blocks and 64 warps: old permits
32 blocks, packet at most25. Registers do not tighten those ceilings.
These are theoretical upper bounds from `TARGET_INFO_GPU`, not measured
occupancy or a claim about the actual shared-memory carveout.

### Parallel-prefix CPU/native readiness

Five original/parallel-prefix CPU controls pass with the new option active;
the full original packet, sparse-factor and new prefix CPU suite also passes
(11 tests). An in-memory control using the actual frozen7ecb constructor
fails the new owner-admission test because that constructor has no parallel
prefix owner. Stable lower/upper order spans all three candidate batches;
the direct prefix test compares every original output at capacity100 and a
deliberate capacity7 buffer while retaining the original uncapped count74.
It also checks disabled limits, changed current velocity and held W.
Capacity128 constructor admission retains the original producer. Requesting
both packet and parallel-global-row owners at capacity100 raises explicitly.

CPU-only AOT compilation with CUDA hidden passes for SM100/SM120, using
block32 metadata. Artifacts:
`/tmp/fpgs-parallel-limits-native-20260914-9da98uwz`.
`cuobjdump` reports 40/48 registers, zero local/stack and 1152 B shared for
the new prefix cubins; there are no explicitly declared shared arrays.
Actual launch resources and performance remain unmeasured. Contact and GS
factories/source are unchanged; the existing measured GS still uses700 B.

Reuse the original two actual-current/held and complete continuing-graph
physical selectors, now with explicit new-owner assertions when the flag is
requested. Replace only the paired run's two packet-specific selectors with:

- `tools.fpgs_bench.test_sparse_factor:TestSparseParallelLimitsCUDA.test_parallel_limit_owner_admission`
- `tools.fpgs_bench.test_sparse_factor:TestSparseParallelLimitsCUDA.test_parallel_limit_prefix_order_and_capacity`

The recorded child environment must set sparse factor1, parallel limits1,
packets0. The earlier four packet selectors and their physical result belong
to frozen7ecb; they are not the new correction's GPU acceptance. Keep the
original155 parent/027 child and fixed Lab53ee, updating only the established
in-memory explicit environment, source pins and selected tests. No new
benchmark or observer framework is introduced.
