# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record and compare the generated code of FeatherPGS kernels without a GPU.

For every kernel of a fixed inventory, ``record`` stores Warp's content hash and
digests of the CUDA and CPU code Warp generates for it; ``diff`` compares two
records. The content hash is the kernel's cache key and the value that
``newton/tests/test_feather_pgs_point_streams.py`` pins. The code digests ignore
comments (generated code quotes every Python line with its line number),
``#line`` directives, whitespace outside string literals and the hash suffix of
kernel names. A refactor is then exact:

* by hash when no hash changed: every kernel hits its existing cache entry, so
  the binaries are the ones already built;
* by generated code when hashes changed but no code digest did, as when only
  the Python text of a kernel changed (formatting, comments, its position).

Warp's generated code also depends on the process that generates it: user
functions are numbered by how many functions of their name the process created
before, a function emits its adjoint once any earlier build used it backward,
and some loop-carried variables are ordered by Python's salted string hash. The
record removes all three: it renames functions by build order within the
kernel, builds each kernel as a fresh process would, and runs with
``PYTHONHASHSEED=0``.

The inventory is the point-stream pin inventory, the factory variants those
pins do not reach (:func:`factory_calls`) and a held-tree layout grid
(:func:`held_tree_calls`). Nothing is compiled and no device is needed; a full
record takes about half a minute on the CPU.

Run from the root of a Newton checkout with the import-time switches
(``FEATHER_PGS_*``) unset::

    python tools/fpgs_bench/kernel_identity.py record child.json
    python tools/fpgs_bench/kernel_identity.py record parent.json --root /path/to/parent
    python tools/fpgs_bench/kernel_identity.py diff parent.json child.json

``--root`` imports Newton from another checkout, for example a ``git archive``
of the parent commit, so one copy of this script records both sides.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ARCH = "120"
FAMILIES = ("pins", "factories", "held_tree")
COLUMNS = ("hash", "cuda", "cpu")
# Import-time switches select other generated kernels (environment_flags.PREFIXES).
SWITCH_PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_")

_TOKENS = re.compile(
    r"(?P<string>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*')|//[^\n]*|/\*.*?\*/|[ \t\f\v]+",
    re.S,
)


def normalize(source: str, suffixes=(), functions=None) -> str:
    """Return generated code without comments, ``#line`` directives, layout whitespace or process-dependent names.

    Args:
        source: Generated CUDA or CPU code.
        suffixes: Hash prefixes that Warp appends to kernel names (eight hex
            digits of the content hash), replaced wherever they end an identifier
            segment, as in ``name_<hash>_cuda_kernel_forward`` and ``wp_args_name_<hash>``.
        functions: ``{native name: stable name}`` of the user functions built.
            Warp numbers a user function by how many functions of its name the
            process created before it, so ``native_26`` in one process is
            ``native_0`` in another; the forward, ``adj_``, ``replay_`` and
            ``_callable_`` names of each are renamed.
    """
    text = _TOKENS.sub(lambda match: match.group("string") or " ", source)
    if suffixes:
        text = re.sub("_(?:" + "|".join(map(re.escape, suffixes)) + ")(?![0-9A-Za-z])", "_HASH", text)
    if functions:
        names = "|".join(map(re.escape, sorted(functions, key=len, reverse=True)))
        pattern = r"(?<![0-9A-Za-z_])((?:adj_|replay_)?)(" + names + r")(?=_callable_|(?![0-9A-Za-z_]))"
        text = re.sub(pattern, lambda match: match.group(1) + functions[match.group(2)], text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line and not line.startswith("#line"))


def stable_function_names(functions):
    """Name built user functions by key and build order instead of by process-wide creation count."""
    names, counts = {}, {}
    for function in functions:
        if function.is_builtin():
            continue
        position = counts[function.key] = counts.get(function.key, -1) + 1
        names[function.native_func] = f"{function.key}__fn{position}"
    return names


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Named:
    """A factory argument that labels show by name instead of by value."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name


def label(name, args=(), kwargs=None):
    """Name a factory call as ``test_feather_pgs_point_streams`` names its pins."""
    kwargs = kwargs or {}
    parts = [repr(arg) for arg in args] + [f"{key}={kwargs[key]!r}" for key in sorted(kwargs)]
    return f"{name}({', '.join(parts)})"


def _value(argument):
    return argument.value if isinstance(argument, Named) else argument


def _unwrapped(factory):
    """Bypass a factory's cache so that every record builds its kernels anew."""
    return getattr(factory, "__wrapped__", factory)


def factory_calls():
    """Yield ``(name, factory, args, kwargs)`` for factory variants the point-stream pins do not reach.

    Each matrix-free Gauss-Seidel flag is toggled alone on two bases, then the
    shapes the benchmark suite's routes select are added; a second keyed-row
    coordinate layout (one articulation, no secondary block) and the Cholesky
    factor families follow. Calls a factory rejects are recorded as rejected.
    """
    from newton._src.solvers.feather_pgs import keyed_rows, small_cholesky  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import solver_feather_pgs as solver  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout  # noqa: PLC0415

    name = "solver_feather_pgs."
    mf_gs = _unwrapped(solver._get_pgs_solve_mf_gs_kernel)
    toggles = (
        {},
        {"friction_mode": "bisection"},
        {"friction_mode": "bisection_desaxce"},
        {"friction_mode": "coulomb_newton"},
        {"software_pipeline": False},
        {"shared_metadata": False},
        {"stream_mf_impulses": True},
        {"fuse_vel_limits": True},
        {"contact_torsion": True},
        {"skip_local_internal_worlds": True, "local_internal_max_constraints": 19},
        {"skip_incremental_rows": 32},
        {"skip_response_block": True},
        {"dense_support_tiles": True},
        {"dense_support_tiles": True, "stage_velocities": True},
        {"factor_coordinates": True},
        {"factor_coordinates": True, "independent_components": True},
    )
    for base in ({}, {"has_drive_rows": False, "has_dense_velocity_limit_rows": False}):
        for change in toggles:
            yield name + "_get_pgs_solve_mf_gs_kernel", mf_gs, (64, 32, 12, ARCH), {**base, **change}
    no_drives = {"has_drive_rows": False, "has_dense_velocity_limit_rows": False}
    for args, kwargs in (
        ((192, 64, 22, ARCH), {"has_drive_rows": False}),
        ((192, 64, 22, ARCH), {"has_drive_rows": False, "skip_incremental_rows": 128}),
        ((72, 32, 18, ARCH), {**no_drives, "skip_incremental_rows": 48}),
        ((192, 64, 15, ARCH), {**no_drives, "skip_local_internal_worlds": True, "local_internal_max_constraints": 19}),
        ((192, 64, 29, ARCH), {**no_drives, "factor_coordinates": True, "independent_components": True}),
    ):
        yield name + "_get_pgs_solve_mf_gs_kernel", mf_gs, args, kwargs

    def parallel(layout, *args, **kwargs):
        factory = keyed_rows.get_parallel_factory.__wrapped__(solver._get_pgs_solve_parallel_kernel, layout)
        return factory(*args, **kwargs)

    layout = Named("KeyedLayout(18, 0, 0, 0, 48, False, 0)", KeyedLayout(18, 0, 0, 0, 48, False, 0))
    yield "keyed_rows.get_fallback_kernel", _unwrapped(keyed_rows.get_fallback_kernel), (layout,), {}
    yield "keyed_rows.get_patch_fallback_kernel", _unwrapped(keyed_rows.get_patch_fallback_kernel), (layout,), {}
    for (rows, min_rows), linked in ((tier, linked) for tier in ((32, 0), (48, 32)) for linked in (False, True)):
        kwargs = {
            "rows": rows,
            "min_rows": min_rows,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
        }
        if linked:
            kwargs["linked_rows"] = True
        yield "keyed_rows.get_parallel_factory", parallel, (layout, 72, 32, 18, ARCH), kwargs

    for size in (6, 9, 16):
        index = Named(f"range({size * size})", tuple(range(size * size)))
        small = _unwrapped(small_cholesky.get_kernel)
        yield "small_cholesky.get_kernel", small, (size,), {}
        yield "small_cholesky.get_kernel", small, (size,), {"geometric": True}
        yield "small_cholesky.get_kernel", small, (size,), {"geometric": True, "geometric_index": index}
        yield (
            "small_cholesky.get_kernel",
            small,
            (size,),
            {"geometric": True, "geometric_index": index, "geometric_stride": size * size + 1, "geometric_group": True},
        )
        for factory in (
            "_get_crba_cholesky_warp_kernel",
            "_get_crba_cholesky_kernel",
            "_get_cholesky_kernel",
            "_get_cholesky_inverse_tiled_kernel",
            "_get_inverse_cholesky_register_kernel",
            "_get_triangular_solve_kernel",
        ):
            yield name + factory, _unwrapped(getattr(solver, factory)), (size, ARCH), {}
        factory = _unwrapped(solver._get_inverse_cholesky_register_kernel)
        yield name + "_get_inverse_cholesky_register_kernel", factory, (size, ARCH), {"lower_only": True}


def held_tree_layouts():
    """Return ``{name: (layout, chain_scan, response, repair_options)}`` shaped after the suite's held trees.

    Chain plans with free and fixed roots (one at 64 lanes, one with a stored
    factor permutation), kinetic plans at 4, 8, 16, 32 and 64 lanes, and forests
    with a free and a prescribed root, with sparse, Cholesky and diagonal
    responses, copied-state reuse and an alternate repair motion.
    """
    from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout  # noqa: PLC0415

    def reverse(dofs):
        return tuple(range(dofs - 1, -1, -1))

    def tree(bodies, dofs, depth, root, *, lanes=0, entries=None, permutation=None, **forest):
        entries = dofs * (dofs + 1) // 2 if entries is None else entries
        permutation = reverse(dofs) if permutation is None else permutation
        return TreeLayout(bodies, dofs, entries, dofs, depth, root, permutation, lanes, **forest)

    motion = {"motion_velocity": True}
    copies = {"copy_reuse": True}
    return {
        "chain_free9": (tree(9, 14, 2, 6, entries=63), True, "sparse", {}),
        "chain_free13": (tree(13, 18, 5, 6, entries=120), True, "sparse", {}),
        "chain_free20": (tree(20, 25, 5, 6, entries=180), True, "sparse", {}),
        "chain_free44_l64": (tree(44, 43, 10, 6, entries=434, permutation=tuple(range(43))), True, "sparse", {}),
        "chain_fixed8": (tree(8, 6, 6, 0), True, "sparse", motion),
        "chain_fixed11": (tree(11, 9, 8, 0), True, "sparse", motion),
        "kinetic_fixed3_l4": (tree(3, 2, 2, 0, lanes=4), False, "cholesky", copies),
        "kinetic_free1_l8": (tree(1, 6, 0, 6, lanes=8), False, "cholesky", motion),
        "kinetic_fixed11_l16": (tree(11, 9, 4, 0, lanes=16), False, "cholesky", copies),
        "kinetic_fixed9_l16_diagonal": (tree(9, 4, 4, 0, lanes=16), False, "diagonal", copies),
        "kinetic_free17": (tree(17, 18, 4, 6), False, "cholesky", {}),
        "kinetic_fixed21": (tree(21, 16, 4, 0), False, "cholesky", motion),
        "kinetic_free44_l64": (tree(44, 43, 10, 6), False, "sparse", {}),
        "forest13_l16": (
            tree(
                13,
                21,
                9,
                0,
                lanes=16,
                entries=87,
                roots=(0, 11, 12),
                root_coordinates=(0, 6, 6),
                root_offsets=(0, 9, 15),
                root_families=(9, 6, 0),
            ),
            False,
            "cholesky",
            {},
        ),
        "forest32": (
            tree(
                32,
                35,
                10,
                0,
                entries=318,
                roots=(0, 30, 31),
                root_coordinates=(0, 6, 6),
                root_offsets=(0, 23, 29),
                root_families=(23, 6, 0),
            ),
            False,
            "cholesky",
            {},
        ),
        "forest10_free": (
            tree(
                10,
                15,
                8,
                0,
                lanes=16,
                entries=66,
                roots=(0, 9),
                root_coordinates=(0, 6),
                root_offsets=(0, 9),
                root_families=(9, 6),
            ),
            False,
            "cholesky",
            {},
        ),
    }


def held_tree_calls():
    """Yield the repair, finish and predictor of every held-tree layout, plus the switch variants of three."""
    from newton._src.solvers.feather_pgs import tree_kinetic_state as native  # noqa: PLC0415

    state = _unwrapped(native.get_state_kernel)
    predictor = _unwrapped(native.get_predictor_kernel)
    switched = ("chain_free13", "kinetic_fixed11_l16", "forest13_l16")
    for layout_name, (tree, chain, response, repair_options) in held_tree_layouts().items():
        layout = {"layout": Named(layout_name, tree)}
        # The alternate motion is a repair-only contract, body velocity a state-kernel one.
        finish_options = {key: value for key, value in repair_options.items() if key != "motion_velocity"}
        variants = [{}]
        if layout_name in switched:
            variants += [{"reference": True}, {"clocked": True}, {"body_velocity": True}]
        for options in variants:
            yield "tree_kinetic_state.get_state_kernel", state, (False, chain), {**layout, **repair_options, **options}
            yield "tree_kinetic_state.get_state_kernel", state, (True, chain), {**layout, **finish_options, **options}
            if "body_velocity" not in options:
                kwargs = {**layout, "response": response, **options}
                yield "tree_kinetic_state.get_predictor_kernel", predictor, (chain,), kwargs


def _switches():
    return sorted(name for name in os.environ if name.startswith(SWITCH_PREFIXES))


def _import_newton(root):
    """Import Newton from ``root`` (default: the checkout holding this script) and return its package path."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    sys.path.insert(0, str(root))
    import newton  # noqa: PLC0415

    package = Path(newton.__file__).resolve().parent
    if package.parent != root:
        raise SystemExit(f"Imported Newton from {package}, not from {root}")
    return package


def inventory(families, select=()):
    """Return ``({label: (family, kernel)}, {label: rejection})`` for the chosen families."""
    import warp as wp  # noqa: PLC0415

    kernels, rejected = {}, {}
    if "pins" in families:
        from newton.tests import test_feather_pgs_point_streams as pins  # noqa: PLC0415

        kernels.update((name, ("pins", kernel)) for name, kernel in pins._inventory().items())
    for family, calls in (("factories", factory_calls), ("held_tree", held_tree_calls)):
        if family not in families:
            continue
        for name, factory, args, kwargs in calls():
            key = label(name, args, kwargs)
            if key in kernels or (select and not any(text in key for text in select)):
                continue
            try:
                built = factory(*map(_value, args), **{k: _value(v) for k, v in kwargs.items()})
            except (ValueError, NotImplementedError) as error:
                rejected[key] = f"{type(error).__name__}: {error}"
                continue
            if isinstance(built, wp.Kernel):
                kernels[key] = (family, built)
            else:
                for position, kernel in enumerate(built):
                    kernels[f"{key}[{position}]"] = (family, kernel)
    if select:
        kernels = {key: value for key, value in kernels.items() if any(text in key for text in select)}
    return kernels, rejected


def kernel_digests(kernel, *, codegen=True):
    """Return the kernel key, its Warp content hash and its normalized CUDA and CPU code digests."""
    import warp as wp  # noqa: PLC0415
    from warp._src.context import ModuleHasher  # noqa: PLC0415

    module = kernel.module
    # Resolves deferred wp.static expressions; the content hash of one kernel
    # does not depend on the other kernels of its module.
    module.get_module_hash()
    options = module.resolve_options(wp.config)
    ModuleHasher([kernel], options)
    row = {"key": kernel.key, "hash": kernel.hash.hex()}
    if codegen:
        built = set()
        # Warp builds CPU modules with output_arch None, which tile builtins read.
        for target, arch in (("cuda", int(ARCH)), ("cpu", None)):
            target_options = dict(options, output_arch=arch)
            try:
                builder = _build(kernel, target_options, built)
                source = builder.codegen(target)
                functions = stable_function_names(builder.functions)
                row[target] = digest(normalize(source, (row["hash"][:8],), functions))
            except Exception as error:
                # A code generation failure is recorded, and compared, like a digest.
                row[target] = f"error: {type(error).__name__}: {error}"
    return row


def _build(kernel, options, built):
    """Build one kernel as a fresh process would, whatever this process built before.

    A function stays marked as used by a backward kernel once any build in the
    process marked it, and a marked function also emits its adjoint in later
    forward-only kernels. Clear the mark on every function the kernel builds,
    then build again whenever the kernel reached a function not cleared yet.
    """
    from warp._src.context import ModuleBuilder, ModuleHasher  # noqa: PLC0415

    while True:
        for function in built:
            function.adj.used_by_backward_kernel = False
        builder = ModuleBuilder(kernel.module, options, hasher=ModuleHasher([kernel], options))
        if built.issuperset(builder.functions):
            return builder
        built.update(builder.functions)


def record(path, *, root=None, families=FAMILIES, select=(), codegen=True, allow_switches=False):
    switches = _switches()
    if switches and not allow_switches:
        raise SystemExit(f"Unset the import-time switches, or pass --allow-switches: {', '.join(switches)}")
    start = time.perf_counter()
    package = _import_newton(root)
    import warp as wp  # noqa: PLC0415

    wp.config.log_level = wp.LOG_WARNING
    if codegen:
        # Code generation of some tile builtins (Cholesky, triangular solves) queries the runtime.
        wp.init()
    kernels, rejected = inventory(families, select)
    rows = {}
    for key in sorted(kernels):
        family, kernel = kernels[key]
        rows[key] = {"family": family, **kernel_digests(kernel, codegen=codegen)}
    result = {
        "warp": wp.config.version,
        "newton": str(package),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "switches": {name: os.environ[name] for name in switches},
        "families": list(families),
        "select": list(select),
        "codegen": codegen,
        "seconds": round(time.perf_counter() - start, 1),
        "kernels": rows,
        "rejected": dict(sorted(rejected.items())),
    }
    Path(path).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"Recorded {len(rows)} kernels and {len(rejected)} rejected calls in {result['seconds']} s: {path}")
    return result


def compare(parent, child, columns=COLUMNS):
    """Split two records into changed kernels (with their changed columns), added and removed ones."""
    old, new = parent["kernels"], child["kernels"]
    changed = {}
    for key in sorted(old.keys() & new.keys()):
        differ = [c for c in columns if c in old[key] and c in new[key] and old[key][c] != new[key][c]]
        if differ:
            changed[key] = differ
    rejections = sorted(
        key
        for key in parent.get("rejected", {}).keys() | child.get("rejected", {}).keys()
        if parent.get("rejected", {}).get(key) != child.get("rejected", {}).get(key)
    )
    return {
        "changed": changed,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "rejections": rejections,
    }


def report(parent, child, columns=COLUMNS):
    """Print the comparison; return 1 when a shared kernel changed or a kernel disappeared, else 0."""
    for field in ("warp", "hash_seed", "switches", "families", "select", "codegen"):
        if parent.get(field) != child.get(field):
            print(f"warning: the records differ in {field}: {parent.get(field)!r} against {child.get(field)!r}")
    result = compare(parent, child, columns)
    shared = len(parent["kernels"].keys() & child["kernels"].keys())
    print(f"{shared} shared kernels compared on {', '.join(columns)}: {len(result['changed'])} changed")
    for key, differ in result["changed"].items():
        kind = "hash only, generated code identical" if set(differ) == {"hash"} else "generated code changed"
        print(f"  changed ({kind}): {key}")
    for field, item in (("added", "added"), ("removed", "removed"), ("rejections", "rejection changed")):
        print(f"{len(result[field])} {field}")
        for key in result[field]:
            print(f"  {item}: {key}")
    return int(bool(result["changed"] or result["removed"] or result["rejections"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="record the kernels of one checkout")
    record_parser.add_argument("output", help="JSON file to write")
    record_parser.add_argument("--root", help="Newton checkout to import (default: the one holding this script)")
    record_parser.add_argument("--family", action="append", choices=FAMILIES, help="restrict to these families")
    record_parser.add_argument("--select", action="append", default=[], help="keep labels containing this text")
    record_parser.add_argument("--hash-only", action="store_true", help="skip the generated code digests")
    record_parser.add_argument("--allow-switches", action="store_true", help="record under FEATHER_PGS_* switches")
    diff_parser = commands.add_parser("diff", help="compare two records")
    diff_parser.add_argument("parent")
    diff_parser.add_argument("child")
    diff_parser.add_argument(
        "--columns",
        default=",".join(COLUMNS),
        help="columns that must match: 'hash,cuda,cpu' (default) or 'cuda,cpu' for generated-code identity",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        if argv is None and os.environ.get("PYTHONHASHSEED") != "0":
            # Warp orders some loop-carried variables by string hash, which Python salts per process.
            os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": "0"})
        record(
            arguments.output,
            root=arguments.root,
            families=tuple(arguments.family or FAMILIES),
            select=tuple(arguments.select),
            codegen=not arguments.hash_only,
            allow_switches=arguments.allow_switches,
        )
        return 0
    columns = tuple(column for column in arguments.columns.split(",") if column)
    if not columns or any(column not in COLUMNS for column in columns):
        parser.error(f"--columns takes a comma-separated subset of {', '.join(COLUMNS)}")
    parent, child = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (arguments.parent, arguments.child))
    return report(parent, child, columns)


if __name__ == "__main__":
    sys.exit(main())
