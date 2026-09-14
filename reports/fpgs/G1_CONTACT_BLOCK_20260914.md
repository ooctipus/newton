# G1 contact-block numerical solve

Design checkpoint before implementation, September14 03:36 UTC. Base is
frozen runtime3ca99cc3, report-only0e7f9580. This is a different numerical
iteration, not another exact-GS cache or thread-mapping experiment. It remains
experimental/default-off and cannot change Lab code, substeps or the maximum
eight solver iterations. The user's contract is correct convergence and
physical behavior, not identical finite GS iterates.

## Work hypothesis and complete budget

Current G1 has approximately4.84ms RTX whole-step GS cost. Each normal and its
two tangents separately recompute a residual, project, broadcast the impulse
transaction, update shared kinetic velocity and synchronize. A contact block
can replace these three sequential transactions with one simultaneous local
solve/update when the contact is sticking. Limits stay scalar. Sliding or
numerically unsafe local solves retain the original scalar triplet path.
No generic associated cone projection may introduce normal dilatancy.

Construct only each contact's symmetric3x3 self response, not an N-by-N Gram
matrix. Three off-diagonal products supplement the existing diagonal. They
are formed lazily on a contact's first non-open visit and stored as one float
per reserved row:400 coefficient bytes plus16 ready-word bytes at the existing
100-row capacity, not a new world-scaled maximum allocation. Charge these products,
loads, local factor/solve, stick admission, scalar fallback and final decode.
No old contact, factor, restitution or publication producer is duplicated.

A first whole-step milestone is at least2ms saved on RTX: original GS4.84ms
must become at most2.84ms including its new setup and fallbacks, with all other
owners accounted for. This is an unproved approximately1.11x incremental
milestone from20.354962ms, not the4x backend target. The current same-collision
MJ reference40.869610ms requires10.217402ms total physics for4x; making this
solver free still cannot meet that target alone.

## Numerical contract and first decision

Use matched current rows from the existing clean original G1 operator
captures after200 warm steps: operator1600 (refresh) and1601 (held reuse),
both GPUs, selected worlds at16K. Keep that selected-sample scope explicit;
it cannot establish population fractions or runtime cost. Do not use the
later corrupted compact-row failure as an acceptance reference.

Evaluate a3x3 sticking proposal against the normal complementarity and
Coulomb disk conditions. Preserve denominator-only CFM as iteration
regularization rather than silently adding compliance to the physical
residual. Preserve incoming-impulse/kinetic-delta initialization, delayed
friction and current geometry/held operator distinction. Bound inadmissible
or ill-conditioned blocks with original row fallback; no row is omitted.

The first checkpoint is CPU matched-state convergence at the same eight
outer iterations, measuring residual/complementarity and final physical
velocity against the original eight-iteration result and a converged
reference. Include coupled loaded contacts, active limits, refresh/reuse,
and simple independent stick/slip/open fixtures. Do not require vector
identity. A harmful CPU result must be diagnosed before one targeted
correction; it is not a speed result.

If numerically promising, implement the smallest default-off integrated
owner, reuse physical and complete paired16K benchmark tools, and measure
whole physics promptly. All capacities remain unchanged. Initial90-minute
checkpoint; no silent extension past two hours. If a complete timing loss
occurs, one causal attribution/correction is allowed, not a tile sweep.

## Initial CPU decision and native readiness

The existing16 selected world/epoch payloads contain452 rows and99 complete
contact triplets. The guarded block replay remains finite and cone-feasible
at eight iterations. In the most difficult RTX refresh case, the natural
residual changes7.0939e-4->7.5689e-4 (6.7% worse), while normal complementarity
improves1.294e-3->2.853e-6. Its held case residual4.1338e-3->3.7908e-3 improves.
Other selected cases are effectively equal or better. This supports a
guarded integrated trial, not a universal accuracy improvement.

Across a fixed eight-iteration CPU replay,44 sticking blocks,673 open blocks
and75 slip fallbacks occur. These are not representative runtime hit rates:
the sample includes selected minimum/maximum-row worlds and the fixed loop
overcounts fully stationary worlds relative to the original early exit.
Open blocks with zero incoming triplet impulse and nonnegative current normal
residual safely omit the zero-radius friction transactions. A fully open
world may lose because admission is charged against only one original sweep;
mixed worlds may benefit over several sweeps. To avoid unused intermediate
work, initially open contacts do not construct a self block. A shared ready
bit admits construction only on the first later non-open visit. The warp
reads one lane-zero broadcast decision before publishing the cache and ready
bit. Complete timing decides.

The constructor regression first failed on missing`block_contacts` before
implementation. It now passes alongside original level controls. Native
source preserves the flag-off scalar body and uses its unchanged signature,
adds416B shared self-block coefficients/ready words atC100, and publishes actual
`sparse_contact_block43_s18_c100`. Rejected proposals do not mutate lambda or
kinetic velocity; accepted updates use rounded next-minus-old impulses.
Independent native review finds no math, warp-epoch or fallback blocker.

Offline NVRTC/ptxas compilation succeeds onSM120/SM100: original40/37 registers
and700B shared, candidate62/62 registers and1100B shared, zero stack/spills.
Artifacts `/tmp/fpgs-g1-contact-block-native-VH50B32n/offline`, produced by
the existing be47d960 compiler with only the kernel list and target pair
substituted in memory. Executed compiler SHA256
`03c5975f83133755e90c418c9a38c4e1d3268204ac54523fa48bffd79eb5407c`.
These are compilation facts, not runtime occupancy or speed measurements.
The above eager-cache prototype was not run on a GPU. Before freezing the
candidate, unused open-contact setup was removed with the lazy cache described
above. Final compilation under`offline_lazy` passes: SM12062 registers,
SM10061 registers,1116B shared, zero stack/spills. The same executed compiler
hash applies. Independent review found no cache-publication or lifetime issue:
Z is fixed within one solve and every launch/graph replay clears ready words.

Five CPU checks pass (two numerical, constructor admission, two level-update
controls). The native numerical selectors cover stick/open/slip/unsafe cases,
denominator-only CFM, incoming impulses, delayed friction, all16 saved epochs,
held W with current geometry, graph replay and empty follow-up. GPU numerical
and complete-cost checks remain pending. The saved numerical screen allows
the explicitly reported6.7% residual degradation in one case, not a claim
that the new method dominates every finite-eight result.

## First paired native run and test correction

Frozen643cc8cb ran four selectors on each GPU in
`/tmp/fpgs-g1-contact-block-physical-paired-20260914-01`. Simple numerical cases,
current/held graph replay and constructor admission pass on both. Eight of16
saved cases per card fail before entering the block solver, solely at a newly
copied elementwise Z-versus-FP64-WJ gate (4.4e-6 to1.72e-5 absolute mismatch).
The parent exits1 and its final source/idle guard passes. This is not a complete
physical pass or a measured solver loss.

The warm-replay test erroneously reintroduced the intermediate barrier that
the prior original-response cancellation diagnosis had already separated from
final physical acceptance. The follow-up records the same3e-5/3e-6 coefficient
diagnostic with finite checks, executes the unchanged own-rounded-Z block law,
impulse and decoded-velocity checks, and evaluates independent held-H/current-J
momentum defect<2e-6 plus unchanged cone checks. No coefficient tolerance or
native arithmetic is changed. Original logs, source and pin files remain
untouched; follow-up uses a new source revision and fresh physical folder.
This correction allows the new solve to be assessed, not an advance claim that
every coefficient mismatch or final result is harmless.
