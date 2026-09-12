# Franka private-state owner: numerical pass, cost miss

2026-09-12, 19:28 UTC. The four-times-corrected-MJWarp objective remains
unmet. This experiment is not integrated or enabled in the accepted runtime.
It passed the finite numerical controls below, but its measured cost cannot
deliver the proposed large whole-physics gain. A phase diagnostic is next;
the architectural hypothesis is not rejected merely because its first mapping
missed an unmeasured suballocation.

Follow-up: the [completed phase diagnosis and cooperative retry](FRANKA_COOPERATIVE_STATE_20260912.md)
now measure a 1.25x RTX / 1.17x GB component improvement in a matched window.
The complete-boundary allowance and whole-physics objective still remain unmet.

## What changed in the prototype

One 32-thread world owns current force preparation, held primary mass action,
the complete private prefix, all 21 generalized coordinates, and next public
body state/private kinetic cache. Current subtree external wrenches are
transformed once per body and summed before projection, instead of repeating
the transform for each ancestor axis. Arbitrary current forces remain supported.

Primary internal COM/S/V/A/f/spatial-inertia publication is omitted. Public
q/qd and body poses/COM velocities remain complete. Free/prescribed canonical
fields needed by the next free-body service are published. The private prefix
supports all 20 current limit/mimic rows, uses the qualified rank case only
when its signed conditions hold, and otherwise executes the original eight
PGS sweeps. Raw-positive, invalid-cache, unsupported or newly free-velocity-limit
worlds reject before next state/cache publication.

This is materially more than the earlier held-inverse component: it includes
the free-body predictor and generalized finish, next-state handoff, and private
body/cache ownership. Comparing its cost directly with that partial component
does not establish a complete-solver regression or gain.

## Numerical gate and preserved test failure

Eight CPU controls passed. The separate root-owned CUDA suite then passed on
both physical GPUs: one actual CUDA method per GPU, zero skips, failures or
errors. It covers actual current refresh/reuse inputs, arbitrary changed
forces/COM/gravity, physical held-mass action, original current row law, original
eight-sweep velocity/residual/momentum/bounds, all-21-DOF integration, public
body state, next physical bias/H, poisoned primary canonical caches, rejected
state nonpublication, and a real two-state graph replayed twice. The 20-row
case retains the original finite-eight residual tails; this is not a universal
convergence or full-trajectory claim.

The first paired GPU attempt failed in the reference-test host adapter after
its first physical case passed. Rejected worlds deliberately retained a
sentinel row count of 91; the old oracle binder expanded those worlds before
its device owner guard and indexed beyond the private 20-row array. A CPU
regression reproduced that exact error. The repair zeros only rejected counts
in an oracle-only descriptor, preserves selected counts/metadata and every
live sentinel byte, and rejects selected counts above nine for the local9
oracle. The full20 control uses its independent original-eight oracle. No
runtime code, numerical tolerance, or frozen reference dependency changed.

## Measured complete private owner, not complete Newton boundary

| GPU | Two-owner event median | Eight-substep equivalent | Selected worlds per owner |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 0.647199988 ms | **2.588799953 ms** | 15,552 / 16,384 |
| GB300 | 0.660543978 ms | **2.642175913 ms** | 15,424 / 16,384 |

Each GPU uses its own actual512 saved scene pair, materially replicated 32
times with independently rebased arrays, not modulo-indexed inputs. The first
owner's actual next state and bias feed the second, with shared held L/G and
correct refresh/reuse cadence. Two physical calls per event are multiplied by
four for the unchanged eight-substep environment budget. Forty event samples
follow ten warmups. Every mutable allocation is restored outside each event;
all eager/graph outputs and read-only hashes match. Both parents' exact source,
UUID and final idle guards pass. This is not a fresh16K collision rollout.

The expanded original family's measured exclusive cost was 2.783904 ms RTX.
A 20% reduction of the accepted approximately5.746 ms whole-physics reference
requires a complete replacement no slower than 1.634704 ms. The new owner
alone exceeds that entire allowance by 0.954096 ms RTX, not merely its earlier
1.20 ms suballocation. Even free remaining services would not qualify this
mapping for that large-gain target.

Missing charges include actual free-L6 refresh (the component receives a
captured held factor), current raw counting, demand queues/clears/row/MF
services, unresolved canonical preparation, cold/reset/odd-refresh repair,
private-cache reseeding and fences. Current free forcing/held response,
transport, integration and next free inertia are already timed. The retained
raw-contact/local20/40/general tail lies outside the selected-family allowance;
it must not be double-charged, nor may its possible dilation be ignored.

Next: measure a source-identical phase clone, including probe overhead, to
separate primary preparation/factor work, the newly serialized free6 service,
private prefix construction/solve, generalized integration and next kinetics.
No tile sweep, percent-level polish, new capacity, timestep or iteration change
is authorized by this failed measurement. A future mapping must first fit its
complete charged boundary and then pass repeated whole-physics comparisons.

## Frozen local reproduction evidence

Accepted source remains064ec8ac455fc4cde557a3b54a1a62624cf56441; this report's
branch runtime remains50dfa28d3aabe51f1b5b75721450efac2c35c5f1. Isaac Lab remains
1d8feb82d17dbfab8f0772de56f84deae2cb7974, clean and unchanged. Substeps, dt,
iterations, mass-refresh cadence and all benchmark capacities are unchanged.

- Prototype: `/tmp/fpgs-franka-private-state-tfwYwXuC/private_state.py`,
  SHA256 `10f3f87f85d5fe11129395a5381bc3e201c84112006176cf60371914a98179fa`.
- Corrected test: same directory `test_private_state.py`,
  `edd4538c17838b620be31027a42f0fe907b874857f5d99be43eced8c772215bb`.
- Root runners, chain and explicit205-file source/input pins:
  `/tmp/fpgs-franka-full-owner-root-M1VDkNbt/`, `pins02.json`.
  Reports additionally pin that pin file itself (206 checked entries).
- Preserved first failure:
  `/tmp/fpgs-franka-full-owner-correctness-paired512-20260912-01/`.
- Completed GPU correctness:
  `/tmp/fpgs-franka-full-owner-correctness-paired512-20260912-02/`.
- Completed cost: `/tmp/fpgs-franka-full-owner-cost-paired16k-20260912-01/`,
  manifest `fd7ab5a83026ae09c398467f6bfa94175882d036ba21ba4d390b4cdf0f2f069f`;
  RTX audit `fa314b741e315e3b4836662f02e73b10ce867a84a97fe4cdae1cd5f8cbdc39cb`;
  GB audit `1d12350dc3b8dbb4f5f7a829f2f552d017c262a449ede82b1adf595d774b38ce`.
- Complete-family cost/reader card:
  `/tmp/fpgs-franka-full-state-owner-edYkdU9D/CARD.md`.
- Live reader/cache review:
  `/tmp/fpgs-franka-private-state-live-audit-PUK1E60U/FINDINGS.md`,
  `922c17dd` prefix. No fundamental equation conflict was found, but the
  all-world old scheduler cannot safely consume deliberately stale selected
  spatial caches. Integration is not performance-qualified.

Large saved inputs, generated binaries and scratch experimental helpers remain
local. The completed audit files contain their exact source/input hashes and
the parent commands. No experimental runtime promotion is implied by this report.
