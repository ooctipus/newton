# ANYmal contiguous branch panels: one causal correction

2026-09-14. Experimental successor from frozen eec1b91a, on a new branch;
the failed whole measurement and its sources remain unchanged. CPU/source
implementation only at this checkpoint. No CUDA physical or cost result yet.
The governing contract remains CROSS_TASK_20260911.md. No Lab, task, capacity,
iteration, physical tolerance or benchmark code changes accompany this work.

## Why this correction is funded

The actual source-matched first branch owner lost whole physics:
9.244660→14.167829ms RTX,9.521509→16.000075ms GB. Its exact node attribution
identified parallel32/48 as the main loss:4.079283→8.641411ms RTX and
4.455154→10.849011ms GB exclusive. Original float4 contractions were replaced
for twelve leg coordinates by repeated membership/FFS/tag→slot scalar gathers.
Smaller response storage did not preserve efficient execution. Register counts
also rose, but no counter or individual instruction bottleneck was established.

The source/evidence card is retained at
/tmp/fpgs-anymal-branch-physical-m2e7L8gI/NODE_FINDINGS.md and nodes_evidence.json
(SHA53f13e3688912072c25622bd9dd47e08c6faebba5ebdbdc7448f47214c8d2e91).
The three-step node parent is
/tmp/fpgs-anymal-branch-nodes-paired16k-20260913-01, all four children exited0,
final source/idle guards passed. These node values are not accepted throughput.

The target remains NET≥10% whole RTX versus original50dfa, not recovery from
the failed candidate. With the failed candidate's other costs unchanged, the
node planning envelope requires solve≤2.655ms RTX/3.408ms GB. That is roughly
69% below the failed solve and35%/24% below original. No such gain is promised.
One complete mapping is funded; no block/register/tolerance sweep follows.

## Changed representation and read lifetimes

The direct current-to-held L117 producer, sparse predictor, current geometry,
original row identities/order, full EX1,24 parallel sweeps, friction siblings,
Nesterov restart/stopping, >48/MF fallback and public publication are unchanged.
The representation correction is entirely within the complete branch factory:

- Build/whiten each physical row as before. After every row has finished reading
  the shared physical scratch, an all-lane fence permits its reuse.
- Ballot each row's one/two leg memberships once. Popcounts assign ascending
  original-row ranks, including the second warp. No activity-based row is dropped.
- Keep six root planes in original row order. Pack three coordinate planes for
  each leg in contiguous ascending-row segments. Their total padded length is
  LP=2*AM+28: at most2*AM memberships, at most12 rounding entries and four spare
  float4s. All padding is zeroed. Offsets and the common LP stride are float4
  aligned. The response scratch is one overlaid allocation, not dual panels.
- EX1 matches the same one/two leg tags and reads the packed row positions.
  This packing and remaining one-time EX1 map cost is charged, not omitted.
- Initialize each leg's extrapolated-impulse alias from the actual incoming
  warm impulse. Publish updated aliases beside the original s_y store, before
  the existing sweep fence. No extra sweep synchronization is added.
- Every root and leg contraction uses the original float4/four-accumulator
  scheme. NT32 retains its one chunk and direct s_dv output; NT64 retains three
  chunks across54 coordinate lanes. Leg chunk lengths follow actual leg counts.
  There is no FFS or tag→slot walk in this contraction.
- Final impulse deltas use the same packed aliases/panels before the unchanged
  held transpose action and canonical velocity/impulse publication.

The row-local projection still uses its own two leg tags; the claim is removal
of repeated membership walks and scalar gathers from transposed contractions,
not removal of all integer metadata. Leg packing changes FP32 association where
structural zeros were skipped. Original physical allowances remain mandatory.

The first candidate also retired J ownership with H ownership, reviving full
J zeroing. This correction replaces both H bank entries with the one-element
mass placeholder while retaining the ORIGINAL J ping-pong banks, active-row
clear stream/events and alias updates. No second row producer or max panel is
created. Original H-clear launches now touch only the placeholder. All remaining
memory work is included in complete timing. This is part of correct factor
retirement, not a standalone small-clear tuning experiment.

## Checks and resources

The existing factory regression first failed on the missing contiguous-panel
identity before implementation. Eight CPU tests now pass; exactly three CUDA
selectors skip with devices hidden. CPU controls cover retained J ownership,
actual direct factor/predictor/fallback math, held reuse and topology guard.
Native-source controls require contiguous float4 contractions and prohibit FFS.

The same three actual CUDA selectors are retained in test_branch_response.py.
The warm two-leg control now covers3/30/48 rows, all four adjacent leg pairs,
float4 padding and second-warp ranks. Actual constructor/graph/reset/public-body
and >48 original-row-production tests remain at unchanged physical tolerances.
No new runner or fixture corpus is introduced.

Correct-launch offline build artifacts:
/tmp/fpgs-anymal-contiguous-offline-launch-h3iu5x20. Both sm120/sm103 compile;
all stack/spill bytes are zero. Warp AOT defaults were first observed to use256
threads; the reported builds explicitly use the actual32/64 launch options.
The initial PTX assembly failure from its trailing NUL is preserved; passing the
same generated PTX minus that terminator through stdin resolves only formatting.

| Tier | RTX registers | GB registers | Shared bytes |
| --- | ---: | ---: | ---: |
|32|142|118|4896|
|48|130|128|6980|

For comparison, failed eec1 was118/102 registers,4072B at32 and150/130,5828B
at48. Original50dfa was86/80,5408B and80/80,7492B. Tier32 registers increase;
there is no resource-derived speedup claim. Physical then one complete paired
whole A/B against50dfa is the decision. Root owns GPU authorization. Bench961's
existing checked boundary must prove actual branch activation, all valid L117
factors and status zero; the complete cost includes every retained stage.
