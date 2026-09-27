# Structural continuation after capacity correction

The user explicitly requested continued optimization after the correctness
checkpoint, not a stop at benchmark repair. This work starts from `a3c69ca0`
on the ooctipus fork, retaining the original handoff and
[capacity/numerical evidence](CAPACITY_20260912.md). Isaac Lab remains unchanged.
No new performance gain is claimed by this plan.

## Contract and first decisions

Follow [the cross-task gate](CROSS_TASK_20260911.md): fixed timestep, substeps
and iteration allowances; physical/numerical validation rather than bit
identity; paired RTX PRO 6000/GB300 with one owned process per device. Count
all new producers, representation conversion, fallback and publication in
whole-physics measurements. Do not compare a candidate with a slow prototype
or count the corrected dropped-row baseline as an optimization gain.

At 02:30 UTC the renewed study opened three bounded source/profile cards.
Fresh checked node captures at clean Newton `108459ec` cover all five tasks.
Keyboard uses its calibrated 704 rows, 147,456 contacts and 57,344 broad output;
the other four FPGS node controls retain their existing recipes/allocations
for attribution, not a claim that those allocations are minimal. Every
completed capture passes both supported capacity checks and source guards.
Three node-profile steps are attribution, not balanced whole-step timing.

Keyboard's RTX/GB node physics spans are 11.560/11.090 ms. Independent SQLite
joins reproduce 12 physics graph roots per device. The final generic FK kernel
has 2.825/2.796 ms union duration, about one quarter of the span. FK plus body
finalization has 3.244/3.016 ms union duration. These are observed dependencies,
not a counterfactual proof that every microsecond can be removed.

The broad-phase neighbor-list idea was closed **before implementation**:
scanning all 14.8M pairs costs only 0.370/0.434 ms per environment step. Even
free removal cannot meet the first 10% milestone. Pair counts alone would
have selected the wrong owner. This is a concrete measured exclusion, not
rejection after an unexplained slow implementation.

## Selected first prototype: scalar-prismatic publication

Keyboard already has diagonal key mass/limits, but generic FK still serially
visits all 108 independent key branches twice per articulation. Admit actual
independent prismatic topology, retain the original arm/factor/contact solver,
and write original canonical key/cache/public outputs through a body-parallel
producer. For supported immobile bases, translation and COM velocity follow
the scalar joint state directly; arbitrary-axis/frame/COM, prescribed motion,
model notification, reset and refresh/reuse contracts must be respected.

The first implementation changes Stage7 ownership, not the mass algorithm or
iteration budget. It is experimental and default-off. Source study and exact
admission/physical test card: `/tmp/fpgs-prismatic-producer-zi3nD8/CANDIDATE.md`;
independent node gate: sibling `NODE_GATE.json`. Implementation worktree is
`newton-fpgs-structural-v2-20260912`. First checkpoint is 90 minutes from
02:39 UTC, with an earlier integrated test as soon as the producer is ready.

## Independent response-owner experiment

The scalar sparse-contact response builder inherits a 4,096-worker limit
from a warp-per-contact Jacobian producer. The latter launches 32 lanes per
worker; the scalar builder launches only one. The fresh capture therefore
shows just 16 CTAs for all current contacts, costing 1.509/1.281 ms per step.
This work-unit mismatch is a specific causal lead, not a launch-parameter grid.

A separate worktree, `newton-fpgs-response-owner-20260912`, tests one corrected
scalar-contact ownership mapping with all arithmetic and output routing
unchanged. It adds no allocation, kernel or host count read. The candidate must
cover the full bounded prefix, including more than 4,096 contacts, empty/reused
storage and actual CUDA graph replay. Its first paired whole-step check is due
around 03:10 UTC. A component gain is not a whole-step gain; if it falls short
of the substantial milestone, do not start sub-percent follow-up tuning.

## Target and next gate

The corrected Keyboard single graph-only reference is 11.186/10.801 ms; the
new node spans above are not substituted as easier timing baselines. A 10%
first milestone is approximately 1.12/1.08 ms net whole-step saving. The 2x/4x
targets still require roughly half/three-quarters of the complete step removed.
The proposed producer and response changes do not alone promise that result.
Fresh Allegro/ANYmal solver/collision ownership determines the next broad
algorithm candidate while these integrated experiments run.

If a theory/timing mismatch appears, diagnose its actual source and allow one
targeted corrective experiment per new cause. No silent extension beyond two
hours without a revised measurable hypothesis. Record measured integrated gain,
concrete diagnosed loss and unvalidated ideas separately. Capacity-safe
reset-inclusive quality and balanced three-round whole timing precede promotion.

Local captures (large files stay local):

- `/tmp/fpgs-checked-keyboard4k-nodes-20260912-01`
- `/tmp/fpgs-checked-cross-task16k-nodes-20260912-01`
- `/tmp/fpgs-collision-next-card-jtaHFV/RESULT.md`
- `/tmp/fpgs-structural-resume-20260912-zh6cmy/run_checked_arm.py`

The paired runner composes the committed checked entrypoint, source guards,
process-group cleanup and unchanged Lab analyzer. It never changes Lab code or
adds a simulator callback. Node attribution and graph timing remain distinct.
