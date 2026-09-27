# Franka typed free-state owner: physical pass, cost miss

This continues [the cooperative owner](FRANKA_COOPERATIVE_STATE_20260912.md).
The complete selected free-body lifetime has been replaced in an isolated
prototype, not promoted into Newton dispatch. Original Isaac Lab, timestep,
eight physical substeps, maximum eight GS sweeps, mass-refresh cadence and
capacities remain unchanged. **No new whole-physics gain is established.**

## Mechanism and physical scope

For the supported unregularized dynamic free root, store the model-frame 3x3
inertia inverse and the orientation of the actual held response. Refresh that
orientation inside the scheduled current owner; reuse it on the next physical
call. Compute current force/gyro and COM translation directly. Integrate the
dynamic root with the original quaternion, COM transport and damping law; copy
the prescribed root. Publish complete public poses/COM velocities and joint
state, while leaving all thirteen bodies' internal spatial caches untouched
and all three broad FK-valid flags false.

This removes the selected generic free S/bias/L6 action, generic free integration,
kinematics and spatial finalization. It does not silently replace held response
with current geometry. Actual original grouped R6 is checked, not inferred from
a triangular factor. Nonroot/regularized/driven/inconsistently prescribed models
are unsupported. Dirty model/frame/COM notifications require reseeding; current
mass/inertia drift and a stale held epoch reject before next-state publication.
The primary force/factor/complete-eight-sweep arithmetic is unchanged, as is
primary next geometric mass/bias production.

Cold model inverse/frame seeding, current raw-count production, demand/MF rows,
unresolved canonical repair, reset/odd-refresh repair and live synchronization
are still excluded from this component experiment. The current free held-state
refresh is now included; this was excluded from the predecessor's L6 fixture.

Seven native CPU controls pass. Both target architectures compile without
spills, but resource usage increases from the cooperative 96 registers and
72-byte stack to 104/113 registers (RTX/GB) and a 128-byte stack; shared storage
remains 3712 bytes. This is evidence of an unfavorable mapping tradeoff, not
proof that register pressure alone causes the measured loss.

The independent physical suite has eleven CPU passes and one actual-CUDA skip.
The strict paired GPU entry subsequently passes on both cards without skips,
errors or failures. It retains all seven existing actual-GPU physical cases,
including current refresh/reuse, actual chained state, arbitrary forces/COM,
complete twenty-row/original-eight behavior, zero/one/nonopposed constraints,
held actions, public state, next physical H/bias, repeated graphs and rejection.
New dirty-frame/COM, stale-held and poisoned-old-free-service controls pass.
The only oracle changes concern deliberately omitted internal caches; reversing
those changes recovers the previous numerical checks exactly. No numerical
tolerance was widened. Existing finite-eight residual tails remain reported.

The first GPU launch stopped before physics because the root wrapper omitted
the cooperative module's import path. Its failed output and wrapper remain
frozen. A new wrapper adds that explicit path; the native candidate and physical
suite were not changed to obtain the second run's pass.

## Balanced complete-owner measurement

Two consecutive owners, with real state/cache handoff and shared held free
rotation/epoch, are captured per event. All mutable fields are restored before
each replay, all readonly fields remain unchanged, and every eager/graph output
matches. Both cards execute forty samples per arm after ten warmups, alternating
AB/BA, with 16,384 materialized worlds and a fixed 4x environment multiplier.
These are saved current-contact populations, not a new closed-loop16K rollout.

| GPU | Cooperative owner ms/env | Typed-free owner ms/env | Old/new throughput |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | 2.064448 | 2.158656 | 0.956358x |
| GB300 | 2.257536 | 2.327040 | 0.970132x |

The measured candidate event time is 4.56%/3.08% higher. This comparison is
conservative: only the candidate pays its held-free response refresh. It is
not a complete apples-to-apples free-service lifetime comparison, and it does
not prove the complete live replacement would be slower by these percentages.
Nevertheless, candidate owner time alone exceeds the **entire** 1.634704ms
expanded-family allowance by .523952/.692336ms before its missing services.
Thus this mapping cannot establish the proposed20% whole-physics milestone.
The separate1.20ms owner allocation is only a planning allocation, not a second
mathematical rejection bound.

Do not promote or polish this isolated path. Preserve its physically tested
typed free-state mechanism for a broader producer/consumer design. The next
question is whether the primary held dynamics, constraint response and next
kinetic state can avoid complete generic materializations together. Neither
factor-only changes nor a body-lane count establish the required savings.
The accepted Franka whole-physics numbers remain5.746087ms RTX/5.195039ms GB,
or1.863533x/1.992003x versus the fixed corrected MJWarp discovery reference.
The requested4x target is unmet.

## Reproduction and pins

Unchanged accepted Newton tree064ec8ac455fc4cde557a3b54a1a62624cf56441;
unchanged Lab1d8feb82d17dbfab8f0772de56f84deae2cb7974.

- Native directory `/tmp/fpgs-franka-typed-free-Cb87547z`:
  `typed_free_state.py`40833a103e82e2e0f6a85b96eeea6e5b9993b850be699b7320e27fdf6fa4ce00;
  `READY.md`23629bf30a7728ea75bf0efe2769d4f1f5b8b5b5e08b972a7d3610b40d0a4d4d;
  `test_typed_source.py`6d5719357f11f4d8599db955f6a005a756dc4124249c86dcdeb084379055c44e;
  `compile_offline.py`a8d538e9cb5a15ddb04fc986db9213d5e984ce14820e910f7369eeaf89f0e213;
  `offline02/report.json`f6a264536089b787b1c1f600885afecfa11d8830a80c19657347ca022706998b.
- Physical adapter `/tmp/fpgs-franka-typed-physical-4cjutzxg`:
  `typed_physical_check.py`03874b5651f6b703f33c8ead44b45f98bbb41cc3dc24a5c311e8dc1d2fd26022;
  `test_typed_physical.py`ba00c7bb9a7e7c4d19f4d1a23765f9dab28d8897a4dbe6afcbd58fa30d73eb3b.
- Root helpers `/tmp/fpgs-franka-typed-root-ybZbRXti`:
  `harness.py`9aa90b1bf1a2e0fd4c4dd96a30f53dd86b14a5126159631b07ad7ee69422fac6;
  `run_cost.py`0789f9f5a1f4b1276698e768e7389197e7fe4c8461f10633b94114c315299f81;
  `run_matched.py`8f39e2816fcc4cf497d7cdf3e2a01b410e3363f3c4d075940024e11525f22b2b;
  corrected `run_checked02.py`07a9ec61a951f957a8197a8ee844cf7e2ec7740d40bdeaeead545cd4c9094376;
  `pins02.json`2f36caf000d2e3a3c12bb87eecf04811b2195f70c20f475d2ffbe8cc3ffacb41.
  Three root CPU harness/cadence/source-recovery tests pass, independently
  rerun; matching arrays include both shared held-state fields. Ruff passes.
- Physical success `/tmp/fpgs-franka-typed-correctness-paired512-20260912-02`:
  manifest c431e9f4862d780025dd7a779c287d60d0737007111eeb03613629d14c189849;
  RTX audit46c739c7e53d386f4acc799d4ec08b0b4ecf4a0c413dd06b7f49174da097871f;
  GB audit e333df87e39a53f74b043d27f9685620ee86b7feac97d7b05533117793069b06.
- Matched cost `/tmp/fpgs-franka-typed-matched-paired16k-20260912-01`:
  manifest8f2d6dd20e5b696302ed48413b750965cc9ff3ceeff601cef54128fee26d31ad;
  RTX audit753d5b493e15d49908d8263428747ffe8dd8bcd0ad26524767376692b45fc57f;
  GB audit ce0d83dfcadd1c1d8f7768c042284969fbf3e0fb65796abfcb444a58a3e35683.
- Failed import-only physical01 manifest:
  01758764e8f307b2cdfebdf445713d6c98ff8e5c2f20c94200bf2f5afb8917d6.

The successful runs verify238 predeclared source/input pins plus the pin file,
original source cleanliness and final GPU idle. Their manifests retain exact
paired commands and all forty raw event durations per arm. Experimental sources,
binaries and large captures remain local; this checkpoint changes reports only.
