# Current local row-packet experiment — pre-timing card

Branch: `ooctipus/fpgs-franka-row-packets-20260912`, accepted runtime base
`42f1492ab99e2e8eedd157bd9893ed9b2fd4de61`. This is an unpromoted, default-off
experiment. No speedup or completed live physical-quality claim is made.
Set `FEATHER_PGS_LOCAL_ROW_PACKETS=1` to request the guarded owner.

## Complete boundary and cost gate

The previous frozen design is `/tmp/fpgs-franka-local-row-card-k9580FSh/CARD.md`
(`0f07fe62688d1d46d27b4a6413c2f0e4327b6dfb8b6351b839efb14ff82cfbb3`). Its
same-window accepted42 complete row/response/local/queue union is
2.140044 ms RTX / 2.367823 ms GB, versus whole 6.375587 / 6.122442 ms.
The existing local solve union is already only .509035 / .609301 ms; the
larger sum of overlapping local kernels is not a removable budget.
Twenty percent RTX whole-time removal requires 1.275117 ms saved, leaving
at most .864927 ms for the complete replacement boundary. This is aggressive
and unproven; even reaching it would not finish the fixed 4× task target.

This revision removes the separate mimic allocation/group-J population,
joint-limit J population, two prefix snapshots, dense-count finalization,
local classification and two queue-compaction passes. One current sparse
prefix owner and one finalization/classification/queue owner replace them.
Current dense contacts write a bounded 15-coordinate packet plus canonical
metadata, RHS and CFM seeds; the existing local native response/8-GS kernels
read these packets and sparse prefix coefficients instead of canonical J.
No original local response or recurrence arithmetic is renamed as a new gain.

The GENERAL path remains complete: a compact current-world queue clears all
active group-J rows, publishes its prefix, reprojects raw contact endpoints,
executes the original held-L triangular action, publishes world J/Y, and
computes J·Y+CFM. This includes rows above the local bound and unsupported
contact topology. All empty fallback launches and packet traffic are charged.
The previous separate dense bias/restitution/CFM passes and local restitution
J copies are unnecessary because current producers own these outputs. The
double-buffer J-clear/count-copy is omitted only for this owner; GENERAL
current rows are completely initialized before use, while H/event cadence
is unchanged.

Retained work includes original raw allocation and collision, every mass/FK
stage, original MF row/limit setup and body inverse, dense impulse preparation,
all original eight sweeps and their convergence/projection law, integration,
public contact forces and state publication. No Lab, timestep, substep,
iteration, model, collision capacity or maximum-row setting is changed.
The producer launches bounded workers with a current-count stride, never a
raw-capacity-sized coefficient grid. New private storage is 45 MiB at16K:
37.5 MiB J packets, 2.5 MiB row tokens, and 5 MiB sparse prefix coefficients.
Original canonical fallback allocations are retained. This is not a net
memory-saving claim and does not allocate the rejected 720 MB raw packet.

## Admission, epochs and fallbacks

Admission is actual topology/mode, not task name: one nine-DOF primary per
world, original 20-row pair /40-row residual /12-row MF local capacities,
all prefix-producing groups of size9, no connect rows, at most two TOTAL
mimic-list entries per primary, and declared mimic world equal to its
articulation world. Missing mimic plans and all unsupported constructors
retain the original implementation. Current enabled flags, coefficients,
q, limits, body poses, motion columns and prescribed velocities are live.
Structural mappings and kinematic membership changes require solver
reconstruction rather than silently reusing captured private mappings.

The mode is the existing immediate matrix-free, augmented-drive,
interleaved/current-friction local path, with no gradient, velocity pass,
warm start, regularization, pre-elimination, grouped/debug/INK/world-row,
sparse/paired-whitening or tiled/diagonal-response conflict. The original
held L is used on refresh and reuse; no fresh mass substitution is made.
Every step initializes the private row map, so inactive tails are unread.
Original current-slot capacity status stays sticky; local overflow is not
a capacity reduction and routes to the complete original general solver.

## Evidence before root GPU gates

Four actual accepted42 512-world captures are pinned under
`/tmp/fpgs-franka-current-rows-paired512-20260912-01`. CPU tests match current
prefix count/order/J/RHS and all original owner/queue sets at both GPUs'
refresh and reuse checkpoints. Constructor ordering, disabled-leading mimic,
declared-world admission, topology reconstruction, exact 9/20/40/12 bounds,
foreign MF endpoints and sticky overflow have focused regressions.

The independent contact tests execute the original metadata/J/bias/restitution
kernels on those inputs: current J/metadata match exactly on CPU, and RHS
maximum difference is 2.38419e-7. Raw-FP64 versus world-origin FP32 tails and
CPU-versus-recorded-GPU differences remain documented, not tolerance-waived,
in `/tmp/fpgs-franka-contact-ready-s9g890kL/READY.md`.

Independent forced GENERAL controls cover all512 worlds of all four captures:
current J, FP64 action of held L·Lᵀ, world mapping, diagonals and poisoned tails.
The CUDA-only controls compare actual produced packets and poisoned canonical
J against a private original full 8-GS control, with lambda/velocity/diagonal,
MF action and seeded graph owner withdrawal/re-admission. These are not
rollout, actual reset or sensor-quality tests. Recorded original replay is
restricted to its matching source GPU UUID.

Both architectures compile offline without spills. Prefix uses40/32 registers;
private local9/20/40 use110/96/88 RTX and107/80/88 GB, with11,776/9,232/25,104
bytes shared respectively. These are compiled resources, not occupancy or
runtime measurements. Full source pins and commands are in
`/tmp/fpgs-franka-packets-ready-Cm9kutFp/READY.md`.

Root executed the exact frozen three suites on each source GPU UUID: all20
tests PASS on RTX and all20 PASS on GB300, including the three actual CUDA
controls. Logs remain local at
`/tmp/fpgs-franka-packets-native-rtx-20260912-01.log` and the corresponding
`-gb-` path. These are same-input numerical/graph controls, not live rollout
or performance acceptance.

Next: actual constructor/loaded512 and whole16K complete-cost screen.
Any lost timing requires current attribution;
do not infer a win from kernel counts, private-row share or these resource
figures. No block/grid sweep is authorized by this card.
