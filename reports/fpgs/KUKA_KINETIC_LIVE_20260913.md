# Live Kuka kinetic integration checkpoint

Implementation GO: 2026-09-13 04:58 UTC. First integrated checkpoint: 06:28
UTC (90 minutes); reassess explicitly before any continuation past 06:58.
This is implementation in progress, not accepted performance or convergence.

The isolated branch `ooctipus/fpgs-demand-state-20260913` starts at
`685d1df5a2492deafeb2fd16617903ea981e083a`, including the handoff and current
reports. Its starting runtime is byte-identical to accepted Kuka
`50dfa28d3aabe51f1b5b75721450efac2c35c5f1`. Original worktrees and Isaac Lab
remain unchanged. The Newton remote is the ooctipus fork.

## Measured motivation and limits

The compact coupled recurrence plus contact-triplet producer reduced the fresh
Kuka prototype replacement section from 10.396544456 to 8.328000069 ms on RTX
and 9.253376007 to 7.966592073 ms on GB300: 1.25x / 1.16x. These complete-section
measurements do not include live collision generation, callbacks, cold repair
or Lab execution. Do not call them whole-task performance, combine them with
old outside costs, or update the accepted task table. The additional 2x/4x
targets remain unmet. Detailed evidence is in
[the lifetime report](KUKA_KINETIC_LIFETIME_20260913.md).

The live port must enter before the retired producers. A new representation
beside the old full dynamics/response pipeline would not realize the hypothesis.
No new numerical algorithm, iteration change, buffer inflation or Lab edit is
authorized by this implementation checkpoint.

## Parallel ownership

- Native port: `kinetic_*` types and kernel factories, relative imports and
  function-source guards. Preserve generated arithmetic/synchronization first;
  keep explicit corrective changes separate and tested.
- Plan/bindings: actual Model/State/Control/Contacts inputs and existing public
  storage; no fixture/capture import, numerical seed, extra maximum-sized panel
  or snapshot facade.
- Lifecycle: solver entry, graph-safe current/geometric/held epochs, ordinary
  reset and odd-request repair, original fallback transition, stream joins and
  publication/contact-export hooks.
- Root: paired actual eager/graph physical checks and whole-workload A/B. RTX
  remains primary; one owned process per GPU, sources frozen during each run.

## Required first gate

Demonstrate real construction and repeated stepping without snapshot restore:
initial refresh, reuse, subset reset on reuse, odd requested refresh, alternating
State objects, and fresh force/target inputs. CPU alias/construction checkpoint
is 05:45. Both primary and original free refresh must consume the same original
request before it is cleared. Reset does not imply an extra held refresh.

Keep dt1/240, two substeps, eight GS sweeps maximum, zero velocity passes,
mass interval2 and original capacities192/64/192/raw4,000,000. Preserve complete
public state, actual `update_contacts` cadence, original anomaly snapshots and
callbacks between physical calls. No hidden two-call solver loop.

Before first admission unsupported input stays on the original path. After
private ownership, an eager demotion must recover canonical factors from the
actual held augmented operator and invalidate current FK; current geometry or
rounded T is not a historical held substitute. A newly unsupported contract
during graph capture may reject before writes with an explicit recapture
requirement at this experimental checkpoint. That is not a claim of completed
general fallback support. No stronger transactional state-staging contract is
being introduced.

Later acceptance still requires no-restore graph replays, notification and
transition coverage, live Lab readers/reset/task behavior, and repeated balanced
whole-physics A/B on both GPUs with wall time reported separately. Approximately
10% RTX whole improvement is a milestone to test, not a promised transfer of
the section gain or completion of the 2x/4x objective.

## Frozen design references

- Lifecycle: `/tmp/fpgs-kuka-live-integration-IvdDlfQT/NOTE.md`, SHA256
  `8f3708a5580f92d052bc345cd178e5a86ac19eb3f5a6a1a58aa3d1a6d06563a4`.
- Full binding map: `/tmp/fpgs-kuka-live-input-binding-VZ7mNfbe/NOTE.md`, SHA256
  `24b332d9dcf9a23b53dfe8a23ef3f0511b4225a33a95f8c2f144993ba31bb287`.

An independent activity census exposed a shared sibling transaction hazard.
The first failed runs remain diagnostic evidence, not speedups or accepted
counts. The eventual explicit correction is recorded below; numerical tolerances
and physical checks were not relaxed to obtain the census.

## Resumed isolated live gate, 11:18 UTC

The original dirty demand-state worktree is preserved. The new worktree
`/home/octi/Projects/newton-fpgs-kinetic-live-20260913`, branch
`ooctipus/fpgs-kinetic-live-20260913`, starts at
`fe61a6528755ab770ac098de051beb8de0ecb8cc`. This inherits the reports and
fixed-root contact-reset handoff, with the same starting Newton runtime as
`685d1df5`. All 30 original candidate files were initially transferred
byte-for-byte: 21 native/type/binding modules, five Solver hook locations in one
existing file, seven diagnostic/test tools and this report. Only the probe's
backend-selection guards/tests and this note change in the resumed checkpoint.

This commit is **WIP and unvalidated in live GPU execution**. It is not an
accepted solver revision, whole-task speedup, convergence result or parent
submodule pointer. The next experiment is the actual 512-world eager lifecycle
probe, followed by real no-restore graph/whole-cost gates if it passes. Do not
repeat archived component tests as a substitute. The first checkpoint is
12:48 UTC; root owns all GPU leases and promotion decisions.

### Shared-impulse correction provenance

The transferred `kinetic_solve.py` SHA256 is
`6e7634e1104cf6a4702d3075f678b020990e592eecd885f8a9989632c48c537a`.
Its explicit reversible edit log removes three dead temporary own-row stores,
adds three old-sibling read-complete warp fences and five final-impulse
read-complete fences. It changes neither the floating recurrence nor the
eight-sweep budget. The compact coupled factory inherits the same recurrence.

The independent diagnostic counterpart is
`/tmp/fpgs-kuka-offset-transaction-gZfso9iY/offset_transaction.py`; its final
paired run is
`/tmp/fpgs-kuka-offset-transaction-paired16k-20260913-01/manifest.json`, SHA256
`98ceac1345a91c2420c986bb078fd268a77d58ff82b2db5c881b66a889ba266e`.
Both cards passed warm/sibling controls and the fresh two-call, same-emitted-order
original-eight comparison. This was an untimed diagnostic, not a live-owner
validation. Earlier failures included correct velocity but doubled published
prefix impulses; they are not dismissed as harmless instrumentation noise.

The checked edit-log SHA256 is
`fdcabd747dd84c9549e0d344d11b95e1fe8909df4ade79ba457bfa515e654913`.
CPU source recovery reverses only these eleven edits and matches all 244
original native/descriptor records. It also checks the corrected offset and
compact generated-source hashes separately. No new transaction hazard is
known from this source transfer; actual live GPU coverage is still missing.
No new independent peer review of the transfer is claimed.

### Probe source selection and scope

`run_kinetic_live_probe.py` requires explicit `--backend-isaaclab` and selects
only `isaaclab_newton` from Lab
`53ee6b44c2334341305dbdf385a3916c6b140799`. Core/tasks, interpreter and the
unchanged `run_profiled.py` harness stay at Lab
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`. No Lab files are edited. Package
origins are checked before imports, all loaded backend sources must be pinned,
and the actual manager/articulation imports are required after execution.
Unused lazy rigid-object imports are not artificially required. A shared `.venv`
symlink to the selected interpreter is the only permitted fixed-Lab untracked
entry. Other source dirt, mixed packages and scratch imports are rejected.

The paired owner supplies one GPU UUID, `--newton`, old `--isaaclab`, distinct
fresh `--output`/`--audit-output`; forwarded arguments supply the fixed backend,
explicit pins, the WIP commit, `--num-envs 512 --perturb`. The probe retains
three actual environment steps (24 physical calls), cold/reuse/alternating-state
ownership, force/target changes, subset reset, odd refresh, capacity checks and
independent public FK checks. Its printed times include readbacks and are not
performance data. Contact-eight, convergence, timing and whole-physics acceptance
remain explicitly false even if the lifecycle probe passes.

### CPU readiness and honest budget

The missing-live-module regression failed before the runtime transfer; the
fixed-backend API regression failed before its guard was added. Afterward, all
38 focused CPU tests pass (bindings, lifecycle, native source recovery, observer
and runner), including a repeat after formatting. The required full-tree
`uvx pre-commit run -a` passes; its only formatting change was the owned probe.
Actual selected package-spec/source checks also pass without
executing Lab or GPU code. This is not a replacement for the first live test.

The old 8.328000069 ms RTX section sits only 0.157446896 ms below the planning
8.485446965 ms replacement ceiling for a 10% whole-physics gain; new live
bookkeeping/dilation and the later synchronization correction are not measured.
The old GB section is already above its corresponding 10% ceiling. Thus this
candidate has a narrow RTX milestone hypothesis, not an established 10% gain
on both cards. It does not by itself reach 2x-handoff or 4x-MJ whole physics.

### First actual probe attempt: import guard, before physics

`/tmp/fpgs-kuka-kinetic-live-eager-paired512-20260913-01` failed on both cards
before task construction. Its audits preserve `success=false`, `complete=false`
and `source_guard_pass=true`. The guard rejected the fileless
`newton.solvers.experimental` namespace that the pinned `newton/solvers.py`
deliberately constructs. This is not a native solver or numerical failure.

The runner-only successor inspects module dictionaries without triggering lazy
imports. It permits only the exact experimental namespace and coupled proxy
objects owned by the selected pinned `newton.solvers`; arbitrary/mixed fileless
namespaces still fail. An actual full Solver import reproduced the old failure
before this correction, and a forged-namespace control remains rejected. The
same physical and failure-audit requirements are retained for attempt02. No
native, binding, lifecycle, Lab, budget or tolerance changes are made.

### Second actual attempt: current-world gravity support gap

Attempt02 used clean runner-corrected commit
`9342b616505d954ff6951d16b60a19b6ceef447b`. The paired parent
`/tmp/fpgs-kuka-kinetic-live-eager-paired512-20260913-02/manifest.json`, SHA256
`26478712fdcd9add801bafd7a603b7e12c9db075e044729aee113abb5c67a98f`,
failed before the first Solver.step on both cards; both audits retain zero
observed calls and successful source guards. Parent final source/idle checks
passed. The actual reset's MODEL_PROPERTIES notification reached the private
binding, which rejected its nonuniform full gravity array. This was an actual
input-support gap, not a performance or contact-solve failure.

Model.gravity is documented as W local-world vectors plus a separate global
world -1 tail. The unchanged Lift event writes local reset IDs only. Its
deterministic curriculum can therefore differ from the unused global tail even
when all owned local worlds agree. The old admission checked the whole array;
both primary and retained free-body force consumers used gravity[0]. Merely
removing the uniformity guard would incorrectly apply world0 gravity to other
local worlds once they differ.

The scoped correction supplies a register-only one-element view of the current
owned-world gravity at the start of the existing construct/finish kernel. Both
unchanged force consumers receive that view. The singleton implicit-world case
retains index0; explicit local worlds use their own index and never consume the
global tail. Static/body ownership is already checked against world-major local
world IDs. Model notification still invalidates current caches. Gravity remains
independent of held geometric mass, so no extra held refresh is introduced.
No Lab/task change, new kernel, device allocation, panel, gravity copy, or
floating force-law rewrite is added. All view work is inside the future charged
producer/finish boundary. Original fallback kernels are unchanged; this is not
a claim of a general FPGS per-world-gravity correction outside the private owner.

The new regression failed on the old notification guard, then passed with three
different owned-world gravities. It independently sums generalized gravitational
force and applies the held operator in FP64 for primary23 and free6, using the
existing 2^-17 scaled action bound. It also checks unchanged held T/free inverse,
read-only gravity aliases and zero effect from changing the global tail.
The current-to-next CPU lifetime test now uses those nonuniform local gravities.
The source oracle retains all original244 records after reversing just the
one view assignment, plus five separately pinned helper/closure records.
Actual live GPU execution of this correction remains pending until root's
next lease; no performance or convergence gate is relaxed.

The authorized attempt03 also labels a correctness-only `world_gravity` event at
physical call7. It temporarily sets world1 gravity to world0 plus a fixed small
three-axis offset, sends the ordinary MODEL_PROPERTIES notification, and restores
the original array plus notification after public/epoch checks. Cleanup also
restores it if the diagnostic aborts. The task configuration and normal reset
randomization remain untouched. This adds no call to Solver.step and changes no
benchmark recipe; all reported harness timings remain inadmissible. The explicit
event/restore CPU regression failed first, then passed. The full focused suite
now has 42 passing tests; the next GPU outcome remains pending.

### Third actual attempt: scoped eager lifecycle pass

Clean commit `50bb891f5bcbd95581ad4f08aaf7c4ccab8783fe` passed on both cards.
The paired parent was reaped exit0; final source and idle guards passed.
Manifest `/tmp/fpgs-kuka-kinetic-live-eager-paired512-20260913-03/manifest.json`
SHA256 `4edff9053b127510f63e0ef3613d735dbb153b2fd53c005383983f4fc183adde`;
RTX audit `80d6905928a24d0138f6ee6dfbbef3ca1b05d9f09e6d80680a424b44833461f2`;
GB audit `01e11b392f87591a689b3af5ac9a092fa6f490e3ee3e34d3e1c1a483e97cd205`.

Each card completed 24 actual private calls and 12 public force exports, with
zero retired-stage calls. All four explicit perturbations ran, including the
temporary world1 gravity, which was restored with the ordinary notification.
Dense count reached90; MF count remained0, an explicit coverage limitation.
The unchanged 3e-5 public-FK gate passed: maximum position error1.525879e-5
on both, scaled velocity error1.735008e-5 RTX /1.274609e-5 GB. This admits only
the actual eager lifecycle/publication scope. Contact-eight, convergence,
timing and whole-physics acceptance remain false.

The next thin continuation adds graph and discovery-performance modes to the
same pinned runner. `kinetic_live_measure.py` composes with the TWO existing
post-warmup/post-profile metadata boundaries. Host eager/capture path observers
are removed before the first timed step; no device graph, arithmetic, counter,
or physics call is added. The512 graph gate uses2 warm/3 measured/2 profile
steps and the same public-FK tolerance. The16K discovery uses200 warm/40 wall/
40 profile steps, checking finite states, current status, capacities, unchanged
graph identity and advanced device epochs only outside timing. It makes no
new full contact-eight or convergence claim; it does not compare unrelated
atomic row orderings bitwise. A matched actual50dfa arm and integrated trace
are still required before any whole-physics performance conclusion.

### First graph attempt: capture-allocation lifetime fault

Graph01 at `f1379fbcb96458fc95c005b957dec72a8c1677ff` failed before its first
boundary on both GPUs. Source and final idle guards passed. Manifest SHA256
`2a16e9c756782524ffe41a1d6504f87b94822ba0e9afa11925e6832a2eada1ec`.
The exact RTX case was rerun under memcheck only; no task or native changes.
Its parent `/tmp/fpgs-kuka-kinetic-live-graph-memcheck-rtx512-20260913-01/manifest.json`
has SHA256 `e3aea9904e0f95be3b8c6f65c2b29762272ae464a1d34872d2fb4f32ef331be8`;
`gpu0/memcheck.log` is `d4e633636b93e0945091e91658b7d06408fc0afa7902f35ac38786bc4cd94a48`.
The FIRST fault is `_invalidate_current_15188213_cuda_kernel_forward+0xb30`,
a four-byte global write by thread96/block0 to unallocated address0x624009b180.
The diagnostic parent was reaped and its final source/idle guards passed.

Actual setup captures the two solver substeps before reset-buffer construction.
The owner first allocated its per-State current/geometric/epoch arrays inside
that capture; ordinary reset then wrote those graph-allocation addresses before
the graph's first replay. This is a Newton ownership/lifetime defect, not a
contact-row arithmetic or publication-tolerance failure. The fixed Lab manager
documents the same CUDA allocation behavior for another solver, but no Lab
replay workaround is introduced here.

The scoped correction moves exactly the standard two State banks and two
directed-call status banks to owner construction. The actual A/B and B/A State
identities are bound later without device allocation. This is the same storage
already demanded by ordinary ping-pong execution, not an enlarged row/contact
panel or an arbitrary reserve of States. Extra actual eager States allocate
only on demand outside capture. Unprepared extra capture States/directions
raise before private writes and require eager preparation/recapture. No kernel,
floating arithmetic, solver budget or task changes occur. A regression failed
first on the original capture-time `wp.zeros`, then passed with both directions,
unseen capture rejection and additional eager-State reuse. The observer also
stores `bool(device.is_capturing)` instead of a reference to Warp's mutable
capture dictionary; this corrects only diagnostic labeling, not the fault.
