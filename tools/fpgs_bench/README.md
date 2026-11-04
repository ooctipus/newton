# FeatherPGS bench: the kernel record

The kernel record (`kernel_identity.py`, `kernel_sass.py`, `kernel_flow.py`) shows what a change does to the generated
and compiled code of the FeatherPGS kernels, without a GPU.

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
