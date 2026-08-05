# Nut/bolt tunneling diagnosis

Screens contact parameters for nut-through-thread tunneling in seconds, instead of
training a policy and discovering it learned to hammer the nut through.

A run is ~0.3–3 s. The whole five-arm × six-load table for one thread size is about
two minutes.

## Scripts

| script | what it answers |
|---|---|
| `nut_bolt_tunneling.py` | one config, one load -> how far did the nut sink |
| `nut_bolt_curves.py` | penetration vs load, per contact configuration |
| `nut_bolt_critical.py` | bisects the load a configuration survives |
| `nut_bolt_hydro_search.py` | scores hydroelastic `kh` on the same scale |

```bash
python nut_bolt_curves.py m16_tight      # m4_tight / m8_tight / m12_tight / m16_tight
```

## What it measures

The nut is lowered coaxially onto a **static** bolt (so this isolates
nut-through-thread from bolt displacement), allowed to settle, then pressed toward
the bolt base by a position-controlled arm with an effort ceiling.

The metric is **penetration past geometric rest, in thread pitches**. An unrotated
nut cannot legally descend at all -- the thread crest blocks it -- so sinking a full
pitch means the solver let it eat into solid material. Every arm is scored against
one shared reference height, measured on a stiff well-resolved contact, so a soft
configuration settling deeper already reads as penetration instead of redefining
its own zero.

Requiring the nut to traverse the whole bolt is far too strict: the task's reward
already reads a pitch of false descent as progress, which is what the policy farms.

## Calibration against known behavior

The harness is only trustworthy if it reproduces what training already showed. On
`m16_tight`, penetration in pitches:

| arm | 50N | 100N | 200N | 400N | 800N | 1600N |
|---|---|---|---|---|---|---|
| baseline 200 Hz collide / 3200 Hz solver / ke 2.56e6 | 0.03 | 0.03 | 0.04 | 0.06 | 0.07 | 0.12 |
| 1600 Hz solver (substeps 8) | 0.04 | 0.06 | 0.07 | 0.10 | 0.21 | 0.41 |
| 800 Hz solver (substeps 4) | 0.07 | 0.10 | 0.24 | 0.41 | 1.16 | 2.72 |
| 100 Hz collide | 24.2 | 4.48 | 3.46 | 1.86 | 2.92 | 2.63 |
| default soft ke (2500/100) | 1.88 | 3.54 | 6.39 | 12.4 | 15.0 | 16.0 |

Baseline holds (<= 0.12 pitch at 1600 N); 100 Hz collide and default stiffness fail
outright. Halving the solver rate is a clear 3-4x degradation but does not cross a
pitch on m16 -- it does on `m8_tight` (1.18 pitches at 400 N), where the finer thread
is more sensitive. The ordering baseline < 1600 Hz < 800 Hz < 100 Hz ~ soft holds
across sizes.

## Findings

**The ke/kd -> solref mapping is the mechanism.** Confirmed against the built model:
`geom_solref = (2/kd, (kd/2)/sqrt(ke))`. ke 2.56e6 / kd 3200 gives (6.25e-4, 1.0), a
0.625 ms time constant. MuJoCo's REFSAFE clamps the time constant to >= 2*dt, so at a
3200 Hz solver it is exactly at the limit and at 1600 Hz it is clamped -- the contact
silently softens. Lowering the solver rate does not just integrate more coarsely, it
changes the contact you asked for.

**Newton's real ShapeConfig defaults are ke=2500, kd=100** -> solref (0.02, 1.0), a
20 ms time constant. That is the "default stiffness" failure.

**mu must not be small while cone is pyramidal.** Same config, only friction differing:
mu=0.01 NaNs, mu=0.75 (the task's value) is stable. This is the degeneracy the task's
own comment documents -- mu -> 0 zeroes the pyramidal row invweight, efc_D floors at
~1e15, the float32 Hessian degenerates. The stock `example_nut_bolt_sdf.py` uses
mu=0.01 and only survives because it runs `elliptic`. Now reproducible in 3 s.

**The M16-validated parameters do not transfer down.** `m8_tight` loses the solver-rate
margin, and on `m4_tight` the baseline itself fails (2.32 pitches at 200 N, NaN above).
Adding m4/m8/m12 to the task needs its own operating point, not m16's.

**Force is the wrong control variable for a free nut.** 4 kN on a 33 g nut is a 37 m/s
velocity jump in one substep -- the solver explodes before any contact resolves, and a
blow-up scored as "held" silently inverts the metric. The rig drives a position target
with an effort ceiling instead, and treats NaN as a failure, never a pass.

## Operating point per thread size

MuJoCo clamps the contact time constant to >= 2*dt, so the stiffest critically-damped
contact a solver rate can actually hold is `kd = solver_hz`, `ke = (kd/2)**2`. Asking
for more stiffness than the rate supports yields a silently clamped contact, not a
stiffer one. Walking that family upward until penetration stays under a pitch at
1600 N (`nut_bolt_operating_point.py`):

| size | cheapest safe | solver | ke | kd | worst |
|---|---|---|---|---|---|
| m16 | 200 Hz collide x 16 | 3200 Hz | 2.56e6 | 3200 | 0.11p |
| m12 | 200 Hz collide x 32 | 6400 Hz | 1.02e7 | 6400 | 0.00p |
| m8 | 200 Hz collide x 16 | 3200 Hz | 2.56e6 | 3200 | 0.14p |
| m4 | 400 Hz collide x 32 | 12800 Hz | 4.1e7 | 12800 | 0.24p |

The rule reproduces the shipping M16 setting exactly, which is the main reason to
trust it: it was derived from REFSAFE, not fitted to the answer.

m4 needs 4x the collision rate and 2x the substeps of m16 -- 16x the solver work per
environment. Adding it to the task is a real compute decision, not a config tweak.

m12 failing at the m16 rung (1.51 pitches) while m8 passes (0.14) does not follow
thread size monotonically, so it is likely specific to that mesh's clearance rather
than a general size trend. Worth confirming against the task's own USD before acting
on it.

## Hydroelastic

Scored on the same scale, `m16_tight`, worst penetration over 100/400/1600 N:

| kh | 100N | 400N | 1600N |
|---|---|---|---|
| 1e9 | 16.7 | 27.7 | 23.8 |
| 1e10 | 6.75 | 16.0 | 22.9 |
| **1e11** | **2.52** | **8.47** | **12.4** |
| 1e12 | 24.1 | 25.8 | 45.2 |
| 1e13 | 24.1 | 9.04 | 45.2 |

No `kh` reaches SDF's 0.12. The best is 1e11 -- the value the hydro example ships --
and it is still ~100x worse. This is not an SDF-resolution artifact: the non-hydro
control at resolution 128 (what hydro cooks at) gives 0.06/0.08/0.14, matching
resolution 512.

Only `kh` was searched. Hydroelastic has other knobs -- pressure-field resolution, the
pressure callback, gap/margin -- so the finding is "no `kh` alone fixes it", not
"hydroelastic cannot work here".

## Known limits

- The nut rests on the bolt chamfer rather than engaged in the thread, so the rig is
  *less* sensitive than the real task. Screwing the nut on first was attempted and
  abandoned: low torque does not engage, high torque launches the nut.
- Penetrations above ~50 pitches are numerical divergence, not physical descent.
- The 100 Hz collide arm is non-monotone in load; it fails everywhere, but the
  per-load numbers in that row should not be read as a trend.
