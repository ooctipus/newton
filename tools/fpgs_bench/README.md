# FeatherPGS bench: the kernel record and the gates

Two things live here:

- **the kernel record** (`kernel_identity.py`, `kernel_sass.py`, `kernel_flow.py`): what a change does to the
  generated and compiled code of the FeatherPGS kernels, without a GPU;
- **the gates** (`gates/`): the fast landing gate, the CUDA test gate, the suite runner and grader, the big-scene
  runner and the path-selection diff.

Timing studies, other engines and one-off experiments are not kept here; their conclusions live in the workstream's
notes outside the repository.

## The kernel record

Run from the root of a Newton checkout, on the CPU, with the `FEATHER_PGS_*` switches unset and
`PYTHONHASHSEED=0`. `--root` imports Newton from another checkout, so one copy of the tools records both sides.

```sh
python tools/fpgs_bench/kernel_identity.py record parent.json --root /path/to/parent
python tools/fpgs_bench/kernel_identity.py record child.json
python tools/fpgs_bench/kernel_identity.py diff parent.json child.json

python tools/fpgs_bench/kernel_sass.py record parent.json --root /path/to/parent --arch 120 --jobs 8 --ptx ptx-parent
python tools/fpgs_bench/kernel_sass.py record child.json --arch 120 --jobs 8 --ptx ptx-child
python tools/fpgs_bench/kernel_sass.py diff parent.json child.json

python tools/fpgs_bench/kernel_flow.py diff ptx-parent ptx-child
```

The rungs, from strongest to weakest:

1. **identity** (`kernel_identity.py`): Warp's content hash and digests of the generated CUDA and CPU code of every
   kernel of the inventory (module kernels, pinned factory calls, the held-tree layout grid, the exit-voting census,
   the walk's readers). No hash changed: every kernel hits its existing cache entry. Hashes changed but no code digest:
   only the Python text moved.
2. **SASS** (`kernel_sass.py`): NVRTC's PTX at the launch block, assembled by the toolkit's `ptxas` for one
   architecture (`--arch 120` is the RTX PRO 6000, `--arch 103` the GB300), read back by `cuobjdump`. Identical SASS is
   identical machine code. `ptxas` and `cuobjdump` must be on `PATH`. `--family` selects the inventory
   (`walk`, `pins`, `factories`, `held_tree`, ...), `--select` filters labels.
3. **dataflow** (`kernel_flow.py`): where SASS moved, the PTX compared by what it computes: *identical*,
   *schedule-only* (the same floating-point operations reach memory in another order: results bitwise) or *rounding*.

`diff` exits 0 when nothing shared changed. The tests (`test_kernel_identity.py`, `test_kernel_sass.py`,
`test_kernel_flow.py`) run on the CPU: `python -m unittest discover -s tools/fpgs_bench -p 'test_kernel_*.py'`.

## The gates

| file | what |
|---|---|
| `gates/gate_fast.sh` | landing gate, about 15 minutes of one GPU: one same-window A/B pass per friction law over the 12 route-covering tasks (bands 0.97-1.03), then coins and hardware at 128 worlds |
| `gates/cuda_gate.py` | the CUDA test gate: runs the listed `newton/tests` modules on one GPU and compares failures with the known list |
| `gates/sel_pro.txt`, `gates/sel_gb300.txt` | the CUDA gate's two halves; a new test module the gate should run is added by hand |
| `gates/run_suite_par.py` | the suite runner: Isaac Lab's `check_suite.py` scenes, several workers, timed loops alone (`bench_site/sitecustomize.py` stops the other workers) |
| `gates/grade_suite.py` | grades a candidate arm against a reference arm: time against bands, physics against the reference spread, capacity flags |
| `gates/complex/run_complex.py` | the big-scene runner (coins, hardware, dishes, cube, pyramid) with `scenes.py`, `metrics.py` and `analysis.py` |
| `gates/record_diff.py` | per task, the path-selection record fields that differ between two suite runs (`--routes`: route choices only) |

```sh
# fast gate: head tree and sha, track tree and sha (clean, at those shas)
tools/fpgs_bench/gates/gate_fast.sh LABEL /path/to/head HEADSHA8 /path/to/track TRACKSHA8

# CUDA gate, PRO half, on a frozen tree, inside a gpu_locks.sh test slot
python3 tools/fpgs_bench/gates/cuda_gate.py --repo /path/to/tree --base BASE --half pro --run \
    --gpu "$FPGS_PRO_UUID" --shared-gpu --out-dir OUT_DIR
# against a run of the same list at the base
python3 tools/fpgs_bench/gates/cuda_gate.py --report OUT_DIR/junit.xml --baseline-report BASE_DIR/junit.xml

python3 tools/fpgs_bench/gates/record_diff.py "$FPGS_OUT/ab-LABEL-head-1" "$FPGS_OUT/ab-LABEL-track-1" --routes
```

`gate_fast.sh` takes `GPU=gb300`, `GATE_TASKS="..."`, `BIG=0` (no big scenes), `SEED=off` (cold kernel cache) and
`HEAD_PATH`/`TRACK_PATH` (a directory prepended to that arm's `PYTHONPATH`). It writes `ab-LABEL[-pt]-{head,track}-1/`,
`ab-LABEL[-pt]-pass-1.log` and `bands_*.json` under `FPGS_OUT`, and the big-scene logs under `FPGS_OUT/complex/`.
The verdict is the grader's table per law (grade `HIT`/`BEAT`/`MISS`, physics `in` or the deviation in reference
spreads, flags) and the big scenes' median ms/frame and physics flags.

### Machine paths

Each is an environment variable; the default is this workstation's value.

| variable | default | used by |
|---|---|---|
| `FPGS_OUT` | `/home/octi/Projects/IsaacLab.wt/output/fpgs-ws-20261015` | every output, `bench_memory.json`, `tests-status/known_failures.txt`, kernel caches |
| `FPGS_LAB` | `/home/octi/Projects/IsaacLab.wt/isaaclab-fpgs-keyboard-capacity-20261011` | the Isaac Lab tree: its `.venv`, `source/` and `scripts/benchmarks/fpgs_profile/` |
| `FPGS_PROFILE_DIR` | `$FPGS_LAB/scripts/benchmarks/fpgs_profile` | `check_suite.py` and `suite21.json` (a proposed checker change applied to both arms) |
| `FPGS_PY` | `/home/octi/Projects/IsaacLab.wt/newton-fpgs-framework-cleanup-20261014/.venv/bin/python` | the big-scene runner's Python (Warp and the gpu_components pin; `--newton` picks the tree) |
| `FPGS_GPU_LOCKS` | `/tmp/claude-1002/-home-octi-Projects-IsaacLab-wt/9a31c1a9-57ae-4e84-8704-55d2ba3b5586/scratchpad/gpu_locks.sh` | the GPU queue (`gpu_exclusive`, `gb300_exclusive`, `gpu_shared`); it stays outside the repository |
| `FPGS_PRO_UUID` | `GPU-883586b6-3100-0610-81e5-3b4c26f45639` | the RTX PRO 6000 |
| `GB300` | `GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4` (set by the queue script) | the GB300, a shared gate card |
