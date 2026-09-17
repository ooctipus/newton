# G1 row-owned local formation — 2026-09-17

Status: bounded structural experiment; default off and not accepted as a gain.
The inherited handoff remains in this branch. Isaac Lab is unchanged.

## Hypothesis and complete-cost gate

The compact register-residual experiment at `f489a7ce` remained slower than the
accepted corrected-limit owner: 12.5091 → 12.8689 ms RTX and 14.4305 → 14.8613 ms
GB300. This experiment must beat the accepted owner, not recover that loss and
call the recovery progress.

The candidate forms each eligible row in its owning lane, with up to 32 rows
concurrently in the existing 32 × 44 shared tile. Physical J is converted to WJ
in place, visiting output nodes in descending order and original input nodes in
ascending order. The checked lower-triangular support makes this safe without a
second tile. The existing corrected prefix, spectral contacts, iteration budget,
applied-impulse accounting and physical decode are retained.

Current RTX contact-triplet, value-emitting limit-prefix and restitution nodes
cost 1.437483, 0.557088 and 0.072464 ms respectively. Their 2.067035 ms sum is an
upper bound on displaced work, not a promised saving: large/rejected worlds
still need it. Allocation (0.156459 ms) and metadata (0.315643 ms) remain. The
compact register solve already owes 0.352563 ms versus the accepted solve. A
1 ms whole saving therefore leaves at most 0.714472 ms for new formation,
key production, filtered fallback and dispatch; assuming 90% of the displaced
time were eligible would reduce that allowance to about 0.508 ms. Row-count
eligibility has not established time eligibility.

This differs from the closed September 14 serial packet mapping: rows are formed
concurrently, not contact-by-contact by one warp. It is also not a memory-only
claim: geometry duplication and producer reductions may disappear, while sparse
W/index loads can increase across directions. No hardware-counter or roofline
claim is made.

## Ownership and fallback

`FEATHER_PGS_SPARSE_REGISTER_PACKETS=1` requires the corrected register-residual
owner and excludes the older body-basis row mode. Original `build_rows` source is
unchanged for all other modes; the candidate binds a separate method.

1. Original allocation and row metadata retain their order and calibrated
   capacity. Prefix/contact producers publish current row keys instead of Z.
2. The small solve rewrites routing every call, forms current geometry against
   the current/held W, and applies the original restitution trigger locally.
3. Success publishes canonical scalar row fields, impulses and physical velocity.
   Global Z is not produced for successful worlds. No cross-call response cache
   or global Gram is introduced.
4. Failure must leave keys, RHS, impulses and velocity untouched. Filtered original
   prefix/contact materialization and restitution then precede the unchanged
   corrected-limit fallback solve, on the same stream.

Global Z/incident allocations remain large enough for any world to fall back.
This retires production/traffic, not capacity. Private accepted-world Z is stale
and must not be used as a physical test oracle. Public contact-force export uses
canonical row identities and impulses, not that private Z.

## Qualification and timing

Pending. The complete prototype is timeboxed to an initial checkpoint at
05:56 UTC and a whole-path result by approximately 06:15 UTC. No source-width,
MMA or layout sweep is funded. Numerical and physical behavior, not bit identity,
are the gate. Timings will include every new launch and fallback cost, use the
existing idle/source/capacity guards, and compare against the accepted corrected
limits with unchanged Lab, timestep, substeps, maximum iterations and buffers.

The pre-implementation factory/owner tests failed with the missing new native
module, as expected. Existing register-residual tests remain distinct from the
new packet tests. No speedup is claimed until complete measurement.

## Native checkpoint and test-oracle diagnosis

Runtime SHA256 `7ea45204f07a845ea767517719310963aec65f47010f054603121e1af59f5826`;
filtered services `871d2b41c5debc77b4535e4cb565fb446b079a0f8b2da74134fe8bae7ce1222f`.
Independent source review checks geometry, sorted-support triangular in-place
formation, transactional publication and fallback. Existing 13 CPU controls and
four new capture/dispatch controls pass. Initial current/held geometry, anchors
and restitution native selector passes on both cards.

Hidden-device AOT `/tmp/fpgs-register-packets-offline-boa5nu9z/offline01` reports
118 registers RTX /112 SM103, 5760 shared bytes and zero stack/spills. Executable
text is 82176/82304 bytes, versus compact 62976/62848. Resources increased; the
unchanged tile size is not an unchanged-occupancy or speed prediction.

Full native02 exposed a test-oracle mismatch in two saved states (RTX-source
world7410 at1600/1601), on both cards. A newly added component check substituted
independently reconstructed FP64 J for the inherited rounded production Z but
kept the old coefficient-level tolerance. Candidate and original control have
EXACTLY the same failing differences, 4.0260e-5 /3.7645e-5, and tolerance ratios
5.6129/2.6684. This is not a candidate-only physics discrepancy. Both cards'
sixteen-case diagnostic outputs agree: maximum inherited rounded-Z tolerance
ratio is0.073022 and maximum independent H backward defect is4.2574e-8, below
the existing2e-6 gate.

Test `75c98556fb27fd1e9d35954613720e604948dab9f3ac8f91deab86675c48370f`
restores the original coefficient check using original Z saved BEFORE candidate
sentinel poisoning, with unchanged rtol3e-5/atol3e-6. Independent physical J/H,
cone, metadata, scalar fields and original-output controls remain. FP64-J
component diagnostics and failed02/control logs are preserved; no tolerance or
runtime change was used to resolve this mismatch. Native03 passes all six groups
on both cards before whole timing (3.968s RTX /3.948s GB including load/tests,
not performance). Logs share prefix
`/tmp/fpgs-g1-row-owned-packets-`, with `native-{rtx,gb}-20260917-02.log`,
`momentum-control-{rtx,gb}-20260917-01.log`, and final native03.
