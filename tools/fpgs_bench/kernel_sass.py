# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record and compare the compiled SASS of FeatherPGS kernels without a GPU.

For every kernel of a ``kernel_identity`` inventory family (by default the
held-tree layout grid), ``record`` compiles the kernel for one CUDA
architecture with NVRTC (``wp.compile_aot_module``, no device needed), reads
the binary back with ``cuobjdump`` and stores, per kernel:

* the resources ptxas assigned: registers, stack, shared and local memory;
* the instruction count and a digest of the SASS with addresses, encodings
  and the kernel's hash suffix removed;
* a digest of the CPU code with the CUDA-only branches of native snippets
  resolved away (``#if defined(__CUDA_ARCH__)``), so that a change confined to
  the CUDA side of a native leaves the CPU twin's digest unchanged.

``diff`` compares two records. Identical SASS is identical machine code: the
change cannot alter speed or results. Otherwise the report lists the resource
changes, and register counts that cross an allocation granule (eight
registers per thread) are flagged, since they can change occupancy.

Run from the root of a Newton checkout with the import-time switches
(``FEATHER_PGS_*``) unset; ``cuobjdump`` must be on ``PATH``::

    python tools/fpgs_bench/kernel_sass.py record child.json --jobs 8
    python tools/fpgs_bench/kernel_sass.py record parent.json --root /path/to/parent --jobs 8
    python tools/fpgs_bench/kernel_sass.py diff parent.json child.json

``--root`` imports Newton from another checkout, as in ``kernel_identity.py``.
"""

import argparse
import concurrent.futures
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

ARCH = 120
# ptxas allocates registers per warp in units of 256, eight per thread.
GRANULE = 8
RESOURCES = ("reg", "stack", "shared", "local")

_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
_CUDA_TEST = re.compile(r"^\s*(!?)\s*defined\s*\(?\s*__CUDA_ARCH__\s*\)?\s*$")
_INSTRUCTION = re.compile(r"^\s+/\*[0-9a-f]{4,6}\*/\s+(.*?)\s*;")
_KERNEL_NAME = re.compile(r"\b\w+_[0-9a-f]{8}_cuda_kernel_(forward|backward)\b")
_USAGE = re.compile(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")


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


def sass_instructions(sass: str) -> list[str]:
    """Return the instructions of a cuobjdump listing without addresses, encodings or kernel hash suffixes."""
    instructions = []
    for line in sass.splitlines():
        match = _INSTRUCTION.match(line)
        if match:
            instructions.append(_KERNEL_NAME.sub(r"KERNEL_\1", match.group(1)))
    return instructions


def parse_usage(usage: str) -> dict:
    match = _USAGE.search(usage)
    if match is None:
        raise ValueError(f"No resource usage in cuobjdump output: {usage[:200]!r}")
    return dict(zip(RESOURCES, map(int, match.groups()), strict=True))


def compile_kernel(kernel, arch=ARCH) -> dict:
    """Compile one kernel's module offline and return its resources and SASS digest."""
    import warp as wp  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        wp.compile_aot_module(kernel.module, arch=arch, module_dir=directory, use_ptx=False)
        cubins = sorted(Path(directory).glob("*.cubin"))
        if len(cubins) != 1:
            raise RuntimeError(f"Expected one cubin for {kernel.key}, found {len(cubins)}")
        usage = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", str(cubins[0])], capture_output=True, text=True, check=True
        ).stdout
        sass = subprocess.run(["cuobjdump", "-sass", str(cubins[0])], capture_output=True, text=True, check=True).stdout
    instructions = sass_instructions(sass)
    return {
        **parse_usage(usage),
        "instructions": len(instructions),
        "sass": identity.digest("\n".join(instructions)),
    }


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


def _worker(root, families, select, part, parts):
    """Record every ``parts``-th kernel of the inventory, starting at ``part``."""
    identity._import_newton(root)
    import warp as wp  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    kernels, rejected = identity.inventory(families, select)
    rows = {}
    for key in sorted(kernels)[part::parts]:
        family, kernel = kernels[key]
        started = time.perf_counter()
        row = {"family": family, "key": kernel.key}
        try:
            row.update(compile_kernel(kernel))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["cpu_twin"] = cpu_twin_digest(kernel)
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows[key] = row
    return rows, rejected


def record(path, *, root=None, families=("held_tree",), select=(), jobs=1):
    switches = identity._switches()
    if switches:
        raise SystemExit(f"Unset the import-time switches: {', '.join(switches)}")
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        futures = [pool.submit(_worker, root, families, select, part, jobs) for part in range(jobs)]
        results = [future.result() for future in futures]
    rows, rejected = {}, {}
    for part_rows, part_rejected in results:
        rows.update(part_rows)
        rejected.update(part_rejected)
    result = {
        "arch": ARCH,
        "root": str(Path(root).resolve()) if root else str(Path(__file__).resolve().parents[2]),
        "families": list(families),
        "select": list(select),
        "seconds": round(time.perf_counter() - started, 1),
        "kernels": dict(sorted(rows.items())),
        "rejected": dict(sorted(rejected.items())),
    }
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    errors = sum("error" in row for row in rows.values())
    print(f"Recorded {len(rows)} kernels ({errors} failed to compile) in {result['seconds']} s: {path}")
    return result


def compare(parent, child):
    """Return ``{label: change}`` for shared kernels whose SASS, resources or CPU twin differ, plus added/removed."""
    old, new = parent["kernels"], child["kernels"]
    changes = {}
    for key in sorted(old.keys() & new.keys()):
        before, after = old[key], new[key]
        change = {}
        for column in ("sass", "cpu_twin", "error", *RESOURCES, "instructions"):
            if before.get(column) != after.get(column):
                change[column] = (before.get(column), after.get(column))
        if change:
            if "reg" in change and None not in change["reg"]:
                low, high = change["reg"]
                change["granule"] = -(-low // GRANULE) != -(-high // GRANULE)
            changes[key] = change
    return {
        "changed": changes,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
    }


def report(parent, child):
    """Print the comparison; return 1 when any shared kernel's SASS or CPU twin changed or a kernel disappeared."""
    result = compare(parent, child)
    shared = len(parent["kernels"].keys() & child["kernels"].keys())
    print(f"{shared} shared kernels: {len(result['changed'])} changed")
    for key, change in result["changed"].items():
        parts = []
        for column in (*RESOURCES, "instructions"):
            if column in change:
                parts.append(f"{column} {change[column][0]}->{change[column][1]}")
        if change.get("granule"):
            parts.append("REGISTER GRANULE CROSSED")
        if "sass" in change and not parts:
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
    return int(bool(result["changed"] or result["removed"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="compile and record the kernels of one checkout")
    record_parser.add_argument("output", help="JSON file to write")
    record_parser.add_argument("--root", help="Newton checkout to import (default: the one holding this script)")
    record_parser.add_argument(
        "--family", action="append", choices=identity.FAMILIES[:3], help="inventory families (default: held_tree)"
    )
    record_parser.add_argument("--select", action="append", default=[], help="keep labels containing this text")
    record_parser.add_argument("--jobs", type=int, default=1, help="compiling processes")
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
        )
        return 0
    parent, child = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (arguments.parent, arguments.child))
    return report(parent, child)


if __name__ == "__main__":
    sys.exit(main())
