# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record and compare the compiled SASS of FeatherPGS kernels without a GPU.

For every kernel of a ``kernel_identity`` inventory family (by default the
held-tree layout grid), ``record`` builds the binary a launch on the suite's
GPUs runs, without a device: Warp's NVRTC emits the module's PTX for one CUDA
architecture at the block size the kernel launches with, and the toolkit's
``ptxas`` assembles it, as the driver's JIT does (Warp emits PTX whenever the
driver is at least as new as its NVRTC). ``cuobjdump`` reads the binary back,
and the record stores, for the kernel's own forward function:

* the resources ptxas assigned: registers, stack, shared and local memory;
* the instruction count and a digest of the SASS with addresses, encodings
  and the kernel's hash suffix removed;
* a digest of the CPU code with the CUDA-only branches of native snippets
  resolved away (``#if defined(__CUDA_ARCH__)``), so that a change confined to
  the CUDA side of a native leaves the CPU twin's digest unchanged.

Assembled by ptxas 13.2, the 20 held-tree finish and repair kernels that nsys
profiled on the suite at b7932853 have the registers and shared memory nsys
reported (NVRTC's own CUBIN gave two of them one more register). A driver
whose JIT is another ptxas release can assign other registers.

A held-tree kernel compiles at the block its owner launches: 32 threads, or 64
for a tree of more than 32 bodies (``TreeLayout.block``; a chain plan whose
padded chains need a second warp also launches 64, which no grid layout
models), and a classification, one thread per repair block, at the owner's
classification block (:func:`classify_block`, 64 threads). The held-tree
owner's scalar-branch group has no layout: its kernels are module-level kernels
of ``scalar_branches`` in the pins family (``--family pins --select
scalar_branches.``), and its repair compiles at the block the group launches it
with (:func:`scalar_branch_blocks`, 128 threads). Other kernels compile at
their module's block size. Warp sizes some static shared memory by the block,
so other block sizes give other binaries.

``diff`` compares two records. Identical SASS is identical machine code: the
change cannot alter speed or results. Otherwise the report lists the resource
changes, the direction of any register count that crosses an allocation
granule (eight registers per thread), and the blocks each changed kernel keeps
resident per SM (:func:`resident_blocks`), flagging kernels that keep fewer.

Run from the root of a Newton checkout with the import-time switches
(``FEATHER_PGS_*``) unset; ``ptxas`` and ``cuobjdump`` must be on ``PATH``::

    python tools/fpgs_bench/kernel_sass.py record child.json --jobs 8
    python tools/fpgs_bench/kernel_sass.py record parent.json --root /path/to/parent --jobs 8
    python tools/fpgs_bench/kernel_sass.py record child103.json --arch 103 --jobs 8
    python tools/fpgs_bench/kernel_sass.py diff parent.json child.json

``--root`` imports Newton from another checkout, as in ``kernel_identity.py``;
``--ptx DIR`` keeps each kernel's PTX for ``kernel_flow.py``, the next rung
when the SASS of a kernel changed.
``diff`` exits 0 when no shared kernel changed, 1 when one changed or a kernel
disappeared, and 2 when a kernel also keeps fewer blocks resident or spills
more to local memory.
"""

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fpgs_bench import kernel_identity as identity


@dataclasses.dataclass(frozen=True)
class Target:
    """Per-SM limits of one CUDA architecture."""

    name: str
    blocks: int
    """Resident blocks per SM."""
    threads: int
    """Resident threads per SM."""
    shared: int
    """Shared memory per SM [bytes]."""


# The suite's GPUs. Block limits and allocation units follow cuda_occupancy.h
# (CUDA 13.2); the SM sizes are the architectures' largest shared-memory
# carveouts, which the driver selects when shared memory limits occupancy.
TARGETS = {
    120: Target("RTX PRO 6000 Blackwell", blocks=24, threads=1536, shared=100 * 1024),
    103: Target("GB300", blocks=32, threads=2048, shared=228 * 1024),
}
ARCH = 120
WARP = 32
# Registers are allocated per warp in units of 256 (eight per thread) from four
# sub-partitions of 16K registers each; shared memory in units of 128 bytes.
REGISTER_UNIT = 256
GRANULE = REGISTER_UNIT // WARP
SUB_PARTITIONS = 4
SUB_PARTITION_REGISTERS = 16384
SHARED_UNIT = 128
RESOURCES = ("reg", "stack", "shared", "local")

_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_CUDA_TEST = re.compile(r"^\s*(!?)\s*defined\s*\(?\s*__CUDA_ARCH__\s*\)?\s*$")
_INSTRUCTION = re.compile(r"^\s+/\*[0-9a-f]{4,6}\*/\s+(.*?)\s*;")
_KERNEL_NAME = re.compile(r"\b\w+_[0-9a-f]{8}_cuda_kernel_(forward|backward)\b")
_USAGE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
# "Function : name" in a SASS listing, " Function name:" in a resource usage listing.
_FUNCTION = re.compile(r"^\s*Function\s*:?\s*(\S+?)\s*:?\s*$")


def _cuda_condition(kind, expression):
    """Return True or False for a conditional on ``__CUDA_ARCH__`` alone being defined, else None."""
    expression = expression.split("//", 1)[0].strip()
    if kind == "ifdef":
        return True if expression == "__CUDA_ARCH__" else None
    if kind == "ifndef":
        return False if expression == "__CUDA_ARCH__" else None
    match = _CUDA_TEST.match(expression)
    if match is None:
        return None
    return not match.group(1)


def resolve_cuda_branches(source: str, cuda: bool) -> str:
    """Keep the side of every ``__CUDA_ARCH__`` conditional that ``cuda`` selects.

    Conditionals on anything else are kept as written, with their nesting.
    """
    lines, stack = [], []
    for line in source.splitlines():
        directive = _DIRECTIVE.match(line)
        if directive is None:
            if all(keep for _, keep in stack):
                lines.append(line)
            continue
        kind, expression = directive.group(1), directive.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            condition = _cuda_condition(kind, expression)
            if condition is None:
                stack.append(("other", True))
                if all(keep for _, keep in stack):
                    lines.append(line)
            else:
                stack.append(("cuda", condition == cuda))
            continue
        if not stack:
            raise ValueError(f"Unbalanced preprocessor directive: {line.strip()}")
        if stack[-1][0] == "other":
            if kind == "endif":
                stack.pop()
            if all(keep for _, keep in stack):
                lines.append(line)
            continue
        if kind == "elif":
            raise ValueError("An #elif after a __CUDA_ARCH__ test is not resolved")
        if kind == "else":
            stack[-1] = ("cuda", not stack[-1][1])
        else:
            stack.pop()
    if stack:
        raise ValueError("Unterminated preprocessor conditional")
    return "\n".join(lines)


def function_sections(listing: str) -> dict:
    """Split a cuobjdump SASS or resource usage listing into ``{function name: its lines}``."""
    sections, name, lines = {}, None, []
    for line in listing.splitlines():
        match = _FUNCTION.match(line)
        if match:
            if name is not None:
                sections[name] = "\n".join(lines)
            name, lines = match.group(1), []
        elif name is not None:
            lines.append(line)
    if name is not None:
        sections[name] = "\n".join(lines)
    return sections


def sass_instructions(sass: str) -> list[str]:
    """Return the instructions of a cuobjdump listing without addresses, encodings or kernel hash suffixes."""
    instructions = []
    for line in sass.splitlines():
        match = _INSTRUCTION.match(line)
        if match:
            instructions.append(_KERNEL_NAME.sub(r"KERNEL_\1", match.group(1)))
    return instructions


def parse_usage(usage: str) -> dict:
    """Read the registers, stack, shared and local memory of one function's resource usage."""
    matches = _USAGE.findall(usage)
    if len(matches) != 1:
        raise ValueError(f"Expected one function's resource usage, found {len(matches)}: {usage[:200]!r}")
    return dict(zip(RESOURCES, map(int, matches[0]), strict=True))


def resident_blocks(reg: int, shared: int, threads: int, arch: int = ARCH) -> int:
    """Return how many blocks of ``threads`` threads one SM keeps resident, counted as cuda_occupancy.h counts.

    Args:
        reg: Registers per thread.
        shared: Shared memory per block [bytes] as cuobjdump reports it for
            these targets, which includes the driver's 1 KB per-block
            reservation (the cubin's ``.nv.reservedSmem`` section).
        threads: Threads per block.
        arch: Compute capability, a key of :data:`TARGETS`.
    """
    target = TARGETS[arch]
    warps = -(-threads // WARP)
    per_warp = -(-reg * WARP // REGISTER_UNIT) * REGISTER_UNIT
    by_registers = SUB_PARTITION_REGISTERS // per_warp * SUB_PARTITIONS // warps
    allocated = -(-shared // SHARED_UNIT) * SHARED_UNIT
    by_shared = target.shared // allocated if allocated else target.blocks
    return min(target.blocks, target.threads // WARP // warps, by_registers, by_shared)


def _run(command):
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr.strip()[:2000]}")
    return result.stdout


def ptxas_version() -> str:
    """Return the version line of the ``ptxas`` that assembles the records."""
    return _run(["ptxas", "--version"]).strip().splitlines()[-2].strip()


def compile_module(module, *, arch=ARCH, block_dim=None, ptx=None):
    """Build a module's binary as a launch at ``block_dim`` threads (default: its own) does; return its listings.

    NVRTC emits the module's PTX and ``ptxas`` assembles it, as the driver's
    JIT does. Returns the ``cuobjdump`` resource usage and SASS listings; a
    ``ptx`` list receives the module's PTX.
    """
    import warp as wp  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        # The options Module.load resolves for a launch at this block size.
        options = module.resolve_options(wp.config, block_dim=block_dim)
        module._compile(None, directory, output_arch=arch, use_ptx=True, options=options)
        sources = sorted(Path(directory).rglob("*.ptx"))
        if len(sources) != 1:
            raise RuntimeError(f"Expected one PTX file for module {module.name}, found {len(sources)}")
        # Warp stores the PTX with the terminating NUL its loader passes on.
        source, cubin = Path(directory) / "module.ptx", Path(directory) / "module.cubin"
        source.write_bytes(sources[0].read_bytes().rstrip(b"\0"))
        if ptx is not None:
            ptx.append(source.read_text())
        _run(["ptxas", f"-arch=sm_{arch}", str(source), "-o", str(cubin)])
        usage = _run(["cuobjdump", "--dump-resource-usage", str(cubin)])
        sass = _run(["cuobjdump", "-sass", str(cubin)])
    return usage, sass


def kernel_resources(kernel, usage, sass) -> dict:
    """Return one kernel's resources and SASS digest from its module's listings."""
    import warp as wp  # noqa: PLC0415
    from warp._src.context import ModuleHasher  # noqa: PLC0415

    if kernel.hash is None:
        ModuleHasher([kernel], kernel.module.resolve_options(wp.config))
    name = f"{kernel.get_mangled_name()}_cuda_kernel_forward"
    usages, listings = function_sections(usage), function_sections(sass)
    if name not in usages or name not in listings:
        raise KeyError(f"No function {name} in the compiled module")
    instructions = sass_instructions(listings[name])
    return {
        **parse_usage(usages[name]),
        "instructions": len(instructions),
        "sass": identity.digest("\n".join(instructions)),
    }


def compile_kernel(kernel, arch=ARCH, block_dim=None) -> dict:
    """Compile one kernel's module offline and return the kernel's resources and SASS digest."""
    return kernel_resources(kernel, *compile_module(kernel.module, arch=arch, block_dim=block_dim))


def cpu_twin_digest(kernel) -> str:
    """Digest the kernel's generated CPU code with every CUDA-only native branch removed."""
    import warp as wp  # noqa: PLC0415
    from warp._src.context import ModuleHasher  # noqa: PLC0415

    module = kernel.module
    module.get_module_hash()
    options = dict(module.resolve_options(wp.config), output_arch=None)
    ModuleHasher([kernel], options)
    builder = identity._build(kernel, options, set())
    source = resolve_cuda_branches(builder.codegen("cpu"), cuda=False)
    functions = identity.stable_function_names(builder.functions)
    return identity.digest(identity.normalize(source, (kernel.hash.hex()[:8],), functions))


def classify_block() -> int:
    """Threads per block of a held-tree classification launch, one thread per repair block, as the owner launches it.

    Checkouts that predate the owner's constant launch it at 64 threads too.
    """
    from newton._src.solvers.feather_pgs import tree_kinetic_owner  # noqa: PLC0415

    return getattr(tree_kinetic_owner, "_CLASSIFY_BLOCK_DIM", 64)


def scalar_branch_blocks() -> dict:
    """Return ``{label: threads per block}`` of the scalar-branch kernels that do not launch at their module's block.

    The held-tree owner's scalar-branch group launches its repair one warp per
    articulation, four per block (``scalar_branches._REPAIR_BLOCK_DIM``); its
    predict, finish and listed drive kernels launch at their module's block.
    Checkouts that predate scalar branches have none.
    """
    try:
        from newton._src.solvers.feather_pgs import scalar_branches  # noqa: PLC0415
    except ImportError:
        return {}
    return {"scalar_branches.repair_scalar_branches": scalar_branches._REPAIR_BLOCK_DIM}


def launch_blocks(families):
    """Return ``{label: threads per block}`` for the inventory kernels whose launch block is known.

    A held-tree classification launches :func:`classify_block` threads per
    block; every other held-tree kernel launches ``TreeLayout.block``, as do the
    walk family's predictor halves (:func:`.kernel_identity.walk_calls`); its walk solve launches
    ``WALK_SOLVE_THREADS`` and its refresh ``WALK_BLOCK``. In the
    pins family, the scalar-branch repair launches at
    :func:`scalar_branch_blocks`'s block.
    """
    blocks = {}
    if "held_tree" in families:
        classify = classify_block()
        for name, _, args, kwargs in identity.held_tree_calls():
            block = classify if name.endswith(".get_classify_kernel") else identity._value(kwargs["layout"]).block
            blocks[identity.label(name, args, kwargs)] = block
    if "walk" in families:
        from newton._src.solvers.feather_pgs import tree_factor, tree_kinetic_owner  # noqa: PLC0415

        # The predictor's halves launch at their layout's block, the walk's solve at the owner's, the refresh at the
        # factor's (its lane groups fit in it).
        for name, _, args, kwargs in identity.walk_calls():
            key = identity.label(name, args, kwargs)
            if name.endswith((".get_force_kernel", ".get_predictor_kernel")):
                blocks[key] = identity._value(kwargs["layout"]).block
            elif name.endswith(".get_walk_solve_kernel"):
                blocks[key] = tree_kinetic_owner.WALK_SOLVE_THREADS
            elif name.endswith(".walk_step_kernel"):
                blocks[key] = tree_factor.WALK_BLOCK
    if "pins" in families:
        blocks.update(scalar_branch_blocks())
    return blocks


def _module_groups(kernels, blocks):
    """Group inventory labels by module and launch block, so that each module compiles once per block size."""
    groups = {}
    for key in sorted(kernels):
        module = kernels[key][1].module
        block_dim = blocks.get(key, module.options["block_dim"])
        groups.setdefault((id(module), block_dim), []).append(key)
    # Ordered by label, the same in every process.
    return sorted(((keys, block_dim) for (_, block_dim), keys in groups.items()), key=lambda group: group[0][0])


def _worker(root, families, select, arch, part, parts, ptx_dir=None):
    """Record every ``parts``-th module group of the inventory, starting at ``part``; with ``ptx_dir``, store each
    kernel's PTX entry there (:func:`.kernel_flow.ptx_path`)."""
    identity._import_newton(root)
    import warp as wp  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    kernels, rejected = identity.inventory(families, select)
    rows = {}
    for keys, block_dim in _module_groups(kernels, launch_blocks(families))[part::parts]:
        module = kernels[keys[0]][1].module
        started = time.perf_counter()
        ptx = []
        try:
            listings, failure = compile_module(module, arch=arch, block_dim=block_dim, ptx=ptx), None
        except Exception as error:
            listings, failure = None, f"{type(error).__name__}: {error}"
        seconds = round(time.perf_counter() - started, 1)
        for key in keys:
            family, kernel = kernels[key]
            row = {"family": family, "key": kernel.key, "block_dim": block_dim}
            if failure is not None:
                row["error"] = failure
            else:
                try:
                    row.update(kernel_resources(kernel, *listings))
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                if ptx_dir and ptx:
                    from tools.fpgs_bench import kernel_flow  # noqa: PLC0415

                    entry = kernel_flow.entry(ptx[0], f"{kernel.get_mangled_name()}_cuda_kernel_forward")
                    kernel_flow.ptx_path(ptx_dir, key).write_text(entry)
            row["cpu_twin"] = cpu_twin_digest(kernel)
            row["seconds"] = seconds
            rows[key] = row
    return rows, rejected


def record(path, *, root=None, families=("held_tree",), select=(), jobs=1, arch=ARCH, ptx_dir=None):
    if arch not in TARGETS:
        raise SystemExit(f"Unsupported architecture {arch}; record one of {sorted(TARGETS)}")
    switches = identity._switches()
    if switches:
        raise SystemExit(f"Unset the import-time switches: {', '.join(switches)}")
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        if ptx_dir:
            Path(ptx_dir).mkdir(parents=True, exist_ok=True)
        futures = [pool.submit(_worker, root, families, select, arch, part, jobs, ptx_dir) for part in range(jobs)]
        results = [future.result() for future in futures]
    rows, rejected = {}, {}
    for part_rows, part_rejected in results:
        rows.update(part_rows)
        rejected.update(part_rejected)
    result = {
        "arch": arch,
        "ptxas": ptxas_version(),
        "root": str(Path(root).resolve()) if root else str(Path(__file__).resolve().parents[2]),
        "families": list(families),
        "select": list(select),
        "seconds": round(time.perf_counter() - started, 1),
        "kernels": dict(sorted(rows.items())),
        "rejected": dict(sorted(rejected.items())),
    }
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    errors = sum("error" in row for row in rows.values())
    print(f"Recorded {len(rows)} kernels ({errors} failed to compile) for sm_{arch} in {result['seconds']} s: {path}")
    return result


def _blocks(row, arch):
    if arch not in TARGETS or not all(name in row for name in ("reg", "shared", "block_dim")):
        return None
    return resident_blocks(row["reg"], row["shared"], row["block_dim"], arch)


def compare(parent, child):
    """Return ``{label: change}`` for shared kernels whose SASS, resources or CPU twin differ, plus added/removed.

    A change records each differing column as ``(before, after)``. A changed
    register count adds ``granule``, the direction (``"up"``, ``"down"`` or
    ``None``) in which it crosses an allocation granule; a changed register
    count, shared memory or block size adds ``blocks``, the blocks resident per
    SM before and after (:func:`resident_blocks`), when both rows compiled.
    """
    old, new = parent["kernels"], child["kernels"]
    arch = child.get("arch", parent.get("arch", ARCH))
    changes = {}
    for key in sorted(old.keys() & new.keys()):
        before, after = old[key], new[key]
        change = {}
        for column in ("sass", "cpu_twin", "error", "block_dim", *RESOURCES, "instructions"):
            if before.get(column) != after.get(column):
                change[column] = (before.get(column), after.get(column))
        if not change:
            continue
        if "reg" in change and None not in change["reg"]:
            low, high = (-(-count // GRANULE) for count in change["reg"])
            change["granule"] = "up" if high > low else "down" if high < low else None
        if change.keys() & {"reg", "shared", "block_dim"}:
            blocks = (_blocks(before, parent.get("arch", arch)), _blocks(after, arch))
            if None not in blocks:
                change["blocks"] = blocks
        changes[key] = change
    return {
        "changed": changes,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
    }


def _regressed(change):
    """Whether a change keeps fewer blocks resident or spills more to local memory."""
    blocks = change.get("blocks")
    local = change.get("local")
    return bool((blocks and blocks[1] < blocks[0]) or (local and None not in local and local[1] > local[0]))


def report(parent, child):
    """Print the comparison; return 2 for an occupancy or spill regression, 1 for any other change or removal."""
    arch = (parent.get("arch", ARCH), child.get("arch", ARCH))
    if arch[0] != arch[1]:
        print(f"warning: the records compile for different architectures: sm_{arch[0]} against sm_{arch[1]}")
    if parent.get("ptxas") != child.get("ptxas"):
        print(f"warning: the records assemble with different ptxas: {parent.get('ptxas')} against {child.get('ptxas')}")
    result = compare(parent, child)
    shared = len(parent["kernels"].keys() & child["kernels"].keys())
    changes = result["changed"].values()
    directions = [change.get("granule") for change in changes]
    regressions = sum(_regressed(change) for change in changes)
    print(
        f"{shared} shared kernels (sm_{arch[1]}): {len(changes)} changed, register granule up in "
        f"{directions.count('up')} and down in {directions.count('down')}, fewer resident blocks or more local "
        f"memory in {regressions}"
    )
    for key, change in result["changed"].items():
        parts = []
        for column in (*RESOURCES, "instructions", "block_dim"):
            if column in change:
                parts.append(f"{column} {change[column][0]}->{change[column][1]}")
        if change.get("granule"):
            parts.append(f"granule {change['granule']}")
        if "blocks" in change:
            parts.append(f"blocks/SM {change['blocks'][0]}->{change['blocks'][1]}")
        if _regressed(change):
            parts.append("REGRESSION")
        if "sass" in change and not any(column in change for column in (*RESOURCES, "instructions")):
            parts.append("SASS differs at equal resources")
        if "cpu_twin" in change:
            parts.append("CPU twin code changed")
        if "error" in change:
            parts.append(f"error {change['error']}")
        print(f"  {key}: {', '.join(parts)}")
    for field in ("added", "removed"):
        print(f"{len(result[field])} {field}")
        for key in result[field]:
            print(f"  {field}: {key}")
    if regressions:
        return 2
    return int(bool(result["changed"] or result["removed"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="compile and record the kernels of one checkout")
    record_parser.add_argument("output", help="JSON file to write")
    record_parser.add_argument("--root", help="Newton checkout to import (default: the one holding this script)")
    record_parser.add_argument(
        "--family", action="append", choices=identity.FAMILIES[:4], help="inventory families (default: held_tree)"
    )
    record_parser.add_argument("--select", action="append", default=[], help="keep labels containing this text")
    record_parser.add_argument(
        "--arch", type=int, default=ARCH, choices=sorted(TARGETS), help=f"compute capability (default: {ARCH})"
    )
    record_parser.add_argument("--jobs", type=int, default=1, help="compiling processes")
    record_parser.add_argument("--ptx", help="directory for each kernel's PTX entry, for kernel_flow.py")
    diff_parser = commands.add_parser("diff", help="compare two records")
    diff_parser.add_argument("parent")
    diff_parser.add_argument("child")
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        if argv is None and os.environ.get("PYTHONHASHSEED") != "0":
            # Warp orders some loop-carried variables by string hash, which Python salts per process.
            os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
        record(
            arguments.output,
            root=arguments.root,
            families=tuple(arguments.family or ("held_tree",)),
            select=tuple(arguments.select),
            jobs=max(1, arguments.jobs),
            arch=arguments.arch,
            ptx_dir=arguments.ptx,
        )
        return 0
    parent, child = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (arguments.parent, arguments.child))
    return report(parent, child)


if __name__ == "__main__":
    sys.exit(main())
