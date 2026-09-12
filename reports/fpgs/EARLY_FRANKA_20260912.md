# Early contactless Franka completion experiment

Base: `fb0f075935c4abcfcf8f3932ae615774551a2d95`. This is a default-off
scheduling experiment, not a measured improvement or a claim of 20% savings.
Isaac Lab, capacities, timestep, eight sweeps and current constraint law stay
unchanged. Enable only with the existing local row-packet owner.

## Boundary and cost hypothesis

After current prefix/raw allocation and the original free6 velocity-limit
allocation, admit only original SINGLE9 worlds with zero MF rows and zero
current raw endpoint incidence (including rejected contacts). Run the existing
native9 solve and original integration/FK/body-finalization math on a private
stream while all contact, MF, owner40 and other-articulation work remains late.
Original parallel current force, mass and predictor producers are retained.
No new serial world solver or FK representation is introduced.

The pinned prior 16K node window contains 0.943/0.725 ms of **all-articulation**
publication (RTX/GB), an upper ceiling rather than the selected primary cost.
A duration-preserving counterfactual gives median 184/207 us from the proposed
ready boundary to owner40 completion; original SINGLE9 plus all-art publication
takes 148/133 us. This leaves nominal 36/74 us before incidence, compaction,
events, masked/fixed-grid launch costs and bandwidth contention. Historical
512 snapshots admit 486/482 worlds; this is not current 16K coverage evidence.
The original local9 solve is already hidden by owner40: do not count it as
deleted work. A meaningful first target is >=10% whole RTX improvement with
no GB regression, not the unachieved fixed fourfold target.

## Ownership and validation

Private current count/owner views prevent later queue publication from racing
the early solve. Initialize velocity once before the fork; late impulse clearing
and late SINGLE solve exclude early worlds. Guard compact publication indices
before any source/cache access. All free6 and prescribed articulations stay
late. Early and late qdd, integration, FK and finalizer writes are disjoint;
join the persistent done event before whole-state consumers and return. Reject
in-place/overlapping states and unsupported cache/debug/solver modes.

Validate actual same-input native lambda/velocity/diagonal and original public
q/qd/body pose/twist/cache outputs; test stale prior ownership, raw incidence,
MF and prefix transitions, zero/all/mixed cohorts, partial grids, refresh/reuse,
notification/cold paths, and two captured graphs. Negative controls must expose
late lambda/velocity overwrites. CPU/source tests and offline compilation precede
root-owned actual GPU controls and early complete 512/16K timing. All routing,
publication, fallback, streams and joins are charged. Diagnose a first timing
loss once; no block/lane grid or silent serial-math rewrite.

## Source-ready checkpoint

`FEATHER_PGS_EARLY_FRANKA=1` additionally requires
`FEATHER_PGS_LOCAL_ROW_PACKETS=1` and the original admitted direct-cache law.
The 512 snapshot has 1,536 articulations and 10,752 coordinates. Exact added
device array storage is 67,596 bytes at 512 and 2,162,700 bytes (2.063 MiB) at
16K, including masks/private solve views, valid early/late lists, and a DOF-to-art
map. The model's existing joint-to-art map is reused. Public capacities do not
change; list storage is bounded by actual worlds/articulations, not contacts.

All nine source-adapted publication kernels compile on CPU. Offline NVRTC and
ptxas pass all 20 module/architecture combinations (sm120/sm103), with source
guard, in `/tmp/fpgs-early-franka-ready-Hz08S9do/offline03`. Finalizer forward
uses 40 registers, zero stack/spills; FK retains callable math with a 72-byte
stack and no spills. These resources are not a performance result. Four
portable ownership/source regressions, the completed-event lifecycle regression,
and the existing notification regressions pass. The complete source-ready suite
has 34 CPU passes and four CUDA-only skips (38 registered tests). Independent
CPU checks exercise the actual A-entry/no-sentinel constructor and all 15
publication/cache outputs on four saved inputs. The former completed-event
implementation fails the new lifecycle test when restored in memory; the
original unmasked impulse clearer fails the nonzero early-lambda control.
Full precommit passes. Actual native/graph controls and timing remain pending.

The done event is joined before whole-state publication and `active` is then
cleared, so a new capture does not inherit a prior graph's event. Untimed
admission metadata should inspect the persistent device cohort counts/masks,
not interpret the deliberately cleared host `active` flag as non-admission.

## Completed device controls and rejected timing

The frozen runtime is `b217a144e2fa79c8fc3287712452aa5589bcb9e5`.
Both actual GPUs pass all **58 registered controls** (old 45 plus new 13), without
skips; root reaped the test processes before the live runs. These include the
saved-input/native two-graph, publication, ownership, and negative overwrite
controls described above. They establish bounded implementation correctness,
not universal convergence or identical task trajectories.

The 512 screen completes with real early admission but loses physics time:
RTX 2.615774 → 2.696586 ms; GB 2.826325 → 3.056757 ms. The intended 16K screen
also completes and is **rejected**, not promoted:

| Current 16K node window, ms/env-step | RTX baseline | RTX early | GB baseline | GB early |
| --- | ---: | ---: | ---: | ---: |
| Complete graph span | 5.889696 | 6.121368 | 5.409374 | 5.835510 |
| Device-work union | 5.636928 | 5.919330 | 5.117989 | 5.599465 |
| Publication union | .924107 | 1.583607 | .689553 | 1.381625 |
| Publication-exclusive busy | .924107 | .747971 | .686513 | .674520 |
| Publication + SINGLE9 + cohort exclusive busy | .924107 | .949709 | .686513 | .910584 |

This is one paired, three-profile-step node window: 12 process-correlated graph
roots and 24 Newton substeps per arm, not repeated whole-graph promotion.
The existing 40 synchronized wall samples are a separate measurement.

Actual early counts at the two untimed boundaries are 15,569/15,608 RTX and
15,569/15,574 GB, of 16,384. All selected/complement lists are disjoint and
exhaustive. Both boundaries of all four arms have zero solver and collision
sticky flags. Public contacts 4,000,000, dense 192, MF 64, propagation 192, seed 0,
200 warmup steps, sim_dt 1/120, two Newton substeps, decimation 4 and the original
maximum eight GS sweeps remain unchanged. Parent completion, source and idle
guards all pass; no Lab or model/task parameter changes were made.

### Why intended overlap did not yield a gain

Early publication really overlaps and completes before late publication starts
in all 24 substeps on each card. Maximum measured early overhang after late
publication starts is zero: this is not a final-join tail failure. The cost is
extra publication work, new exposed SINGLE9, and inflated concurrent preparation.

Early publication costs 1.021377/.851527 ms union (RTX/GB), of which
.185741/.144422 ms is exclusive busy. Late publication remains
.562230/.530098 ms, entirely exposed. The new cohort prelude adds
.099307/.114720 ms summed work; SINGLE9, previously hidden under other local
owners, now has .102453/.122016 ms exclusive busy. Added routing/dispatch plus
early and late publication increase device work from 628 to 724 operations per
environment step.

The unchanged body inverse grows .120928→.196373 ms RTX and
.082645→.104094 ms GB; contact production grows .175776→.228853 and
.213259→.261653 ms. This supports contention as a cause, not a hardware-counter
claim that uniquely identifies bandwidth or occupancy. Contact trajectories
differ, and RTX owner40 becomes .101344 ms cheaper, so not every duration delta
can be assigned to scheduling.

Late compact articulation lists mix rare primary 9 arms with mandatory free and
prescribed trees; late qdd/integration/finalization also retain masked global
grids. Early/late FK retain the original equations and zero reported local
memory, with 72 RTX registers (GB early64/late72). Actual grids and all per-call
resources are recorded in the causal evidence. Count guards protect ownership
but do not remove launched slots or mixed-tree loop costs.

A narrow late-list correction is not selected: even deleting the entire
.562230 ms RTX late-publication owner cannot recover the .820642 ms required
for a 10% gain over this paired baseline. The separately assessed earlier
prefix-owned readiness variant has only about .022 ms modeled RTX margin
under optimistic overlap/cost assumptions; root rejected implementation as too
fragile. This scheduling branch is closed without a layout grid or runtime
retry. The fixed fourfold backend target remains unachieved.

### Frozen provenance

Baseline `064ec8ac455fc4cde557a3b54a1a62624cf56441`; tools
`5a9d8d76dc060b11fbd9672803c902595488caed`; unchanged Lab
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`.

- Runtime early_franka.py SHA256:
  `0708cbe11fd0f750d079204611b236167e5ac72ced9164fef0416f92d394f242`.
- Runtime solver SHA256:
  `0c6d2947ff81fbf9daaf276235b3998885e0d3f0d80d6c4954e340cd997dc13b`.
- Completed16K parent:
  `/tmp/fpgs-fourx-early-franka16k-20260912-01`; manifest SHA256
  `fc73025143f4eca53960c9460c3b8114055069aefd970adc4eb13b1868ad9f5f`.
- Completed512 parent:
  `/tmp/fpgs-fourx-early-franka512-20260912-01`; manifest SHA256
  `11851674e9d10540248cda55c6172b6326b30acbbd072250b1ef0184e3376598`.
- Independent causal audit:
  `/tmp/fpgs-early-franka16k-causal-uRiwdfIv/{RESULTS.md,evidence.json,audit.py}`.
  Evidence SHA256:
  `5eaa821442e618d37bef8e6b51ef25dd54a23ea8d0a3ec5e2cafb88e3076dce9`;
  audit SHA256:
  `b07b0792f8ac3603539807b357fe58f2f3628cb5fac6a2d4984400ff773ec695`;
  results SHA256:
  `f1c96d23dd7144477d6b93e72e9a288232f5789f7632c807e118108f5f331d79`.
- Unimplemented expanded assessment:
  `/tmp/fpgs-early-franka-prefix-assessment-Ugjf2rzT/CARD.md`, SHA256
  `f20d5fa8da5e0696e141cebac4659091d5f51a4cb396f026c5f8d48749fe0a3f`.

This checkpoint changes only this report. Frozen runtime, original captures
and all parent dependency pointers are preserved.
