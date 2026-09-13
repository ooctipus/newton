# Structural 4x work window

User-authorized window: 2026-09-13 11:13 UTC through 2026-09-14 11:13 UTC.
The Keyboard contact/termination discrepancy is explicitly parked, not solved.
Optimization resumes on the fixed Lab backend, with task physics unchanged.

## North star and fixed contract

Target at least 4x corrected MJWarp **whole physics graph** throughput on
Franka, KukaAllegro, Allegro, ANYmal-D, G1 Rough and SO101 Keyboard. Report
environment wall time separately; neither number is full RL training speed.
RTX PRO6000 is primary; every GPU experiment has a paired GB300 command and
one owned process per device. Preserve timestep, substeps and iteration
allowances. Numerical convergence and physical behavior, not bit identity,
govern solver changes. Never drop rows/contacts or inflate buffers blindly.

Eliminate unnecessary producers, intermediate representations and conversions
across a complete path before tuning the remaining CUDA work. A replacement
must retire the original producers. Include dispatch, refresh, fallback,
collision, synchronization and public-state publication in live measurements.
No standalone sub-percent or small mapping projects.

This branch starts at `fe61a6528755ab770ac098de051beb8de0ecb8cc`, retaining the
handoff and all reports. Its runtime is accepted Newton
`50dfa28d3aabe51f1b5b75721450efac2c35c5f1`; remote is `ooctipus/newton`.
Original worktrees and large local captures are preserved. The fixed Lab
backend is `53ee6b44c2334341305dbdf385a3916c6b140799`, with unchanged prepared
core/tasks/harness from `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
Explicit runtime import checks must prevent accidentally using the old backend.
No Lab source or parent dependency pointer is changed by this optimization.

## Starting evidence and budget

Historical clean comparisons, ms per batched environment step, establish
priorities, not fresh post-fix claims. Revalidate denominators where shared
collision work or the fixed backend changes them. See the existing lifetime,
representative-task and dataflow reports for exact recipes and provenance.

| Task | RTX FPGS / MJWarp | GB300 FPGS / MJWarp | RTX / GB 4x ceiling |
| --- | ---: | ---: | ---: |
| Franka | 5.746087 / 10.708025 | 5.195039 / 10.348531 | 2.677006 / 2.587133 |
| KukaAllegro | 12.862500 / 27.464577 | 12.177994 / 25.954644 | 6.866144 / 6.488661 |
| ANYmal-D | 9.202900 / 36.635007 | 9.388857 / 41.892997 | 9.158752 / 10.473249 |
| G1 Rough | 43.717024 / 52.264290 | 72.387317 / 78.266767 | 13.066073 / 19.566692 |
| Keyboard, post-fix | 7.460734 / 27.728768 | 6.371108 / 27.987691 | 6.932192 / 6.996923 |

Keyboard uses 4096 worlds, the others 16384. Its post-fix values are three
alternating rounds at `/tmp/fpgs-keyboard-postfix-performance-20260913-01`;
the reset/contact gap persists. Allegro's warning-free current reference must
be recovered before reporting a ratio; old warning-affected numbers are not
valid targets. The original-handoff improvement is a separate denominator.

## First bounded experiments

1. Finish the already implemented Kuka live kinetic owner. Preserve original
   dirty candidate tree, freeze a new branch, then actual eager lifecycle and
   graph checks followed promptly by live whole-physics A/B. The previous
   8.328/7.966592 ms section beat a slow prototype, not accepted Newton. Its
   RTX 10%-whole screening margin is only about 0.157 ms before live overhead;
   GB does not clear the analogous screen. It cannot alone deliver 4x MJWarp.
   First checkpoint 12:48 UTC: measured live result or a concrete blocker/cause,
   not another offline component timing campaign.
2. Attribute the current G1 graph on both cards. Existing graph-only captures
   cannot identify collision versus dynamics/solver cost. Reuse calibrated
   capacities and the existing owner; capture only missing node evidence.
   Select a removal boundary with plausible >=10% whole-physics savings and a
   concrete route toward the much larger 4x gap before implementing it.
3. Independently recover Franka's remaining producer/representation costs and
   failed architecture diagnoses. Reopen an experiment only for new causal
   evidence, not another layout/worker-count sweep.

Subsequent blocks prioritize the best measured structural opportunity,
implement its smallest complete path, then transfer/generalize only where
topology and physical checks support it. ANYmal/Keyboard remain cross-task
regression gates; unsupported fallback is not a speedup. Refresh Allegro
alongside the next representative sweep rather than inventing a denominator.

## Checkpoints and prevention of another zero-progress run

- Every candidate records exact removed/added work, whole-step millisecond
  budget, numerical gates, a falsifiable decision and a 90-minute checkpoint.
- A theory/timing mismatch gets a causal diagnosis and one targeted correction.
  No silent extension beyond two hours or repeated variants without new cause.
- Classify results as measured integrated gain, diagnosed loss, or unvalidated
  hypothesis. Finite state and section-level gains are not promotion gates.
- Before promotion, repeat balanced paired live timing, verify convergence,
  capacity/warning gates, reset/fallback transitions and representative tasks.
- Keep exact source pins, commands and failures in reports; large captures stay
  local. Never replace accepted numbers with prototype or instrumented timings.

The session goal controller still reports an older unfinished blocked goal;
creating a replacement returned an unfinished-goal error. It was not falsely
marked complete. Work is continuing in the active session; this report does
not assert that an autonomous background schedule was successfully installed.
