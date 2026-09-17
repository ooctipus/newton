# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental four-full-warp scheduling of unchanged G1 row owners.

Each warp owns one complete world, with the original 32-lane collectives,
global rows and per-world shared state. No factor, contact producer, numerical
policy or publication is changed. The CPU prefix keeps its serial world ABI.
"""

import ast
import hashlib
import inspect
import linecache
import textwrap
from functools import cache

from . import sparse_factor_rows, sparse_spectral_tangents


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Full-warp grouping source seam changed: {old!r}")
    return source.replace(old, new)


def _group_source(source, solve):
    """Change only complete-world indexing and disjoint shared storage."""
    if "__syncthreads" in source:
        raise RuntimeError("Full-warp grouping cannot admit a CTA collective")
    if solve:
        source = _replace(
            source,
            "const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];",
            """const int warp=threadIdx.x>>5, lane=threadIdx.x&31, group=pack*4+warp;
    if(group>=p.group_to_art.shape[0])return;
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];""",
        )
        source = _replace(
            source,
            "__shared__ float du[43], lam[100];",
            """__shared__ float du_storage[4][43], lam_storage[4][100];
    float* du=du_storage[warp];
    float* lam=lam_storage[warp];""",
        )
        source = _replace(
            source,
            "__shared__ float contact_cross[100];",
            """__shared__ float contact_cross_storage[4][100];
    float* contact_cross=contact_cross_storage[warp];""",
        )
        source = _replace(
            source,
            "__shared__ unsigned int contact_ready[4];",
            """__shared__ unsigned int contact_ready_storage[4][4];
    unsigned int* contact_ready=contact_ready_storage[warp];""",
        )
    else:
        source = _replace(
            source,
            "const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];",
            """#if defined(__CUDA_ARCH__)
    const int group=pack*4+(threadIdx.x>>5);
#else
    const int group=pack;
#endif
    if(group>=p.group_to_art.shape[0])return;
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];""",
        )
    return source


def _factory(original, solve, key):
    """Wrap the existing source generator without duplicating numerical code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    factory = tree.body[0]
    factory.name = "_grouped_factory"
    factory.decorator_list = []
    natives = names = 0
    for index, statement in enumerate(factory.body):
        if isinstance(statement, ast.FunctionDef) and statement.name == "native":
            if statement.args.args[0].arg != "group":
                raise RuntimeError("Full-warp native group ABI changed")
            statement.args.args[0].arg = "pack"
            factory.body.insert(index, ast.parse(f"source = _group_source(source, {solve})").body[0])
            natives += 1
            break
    target_name = "solve" if solve else "prefix"
    for statement in factory.body:
        if isinstance(statement, ast.Assign) and any(
            ast.unparse(target) == f"{target_name}.__name__" for target in statement.targets
        ):
            statement.value = ast.Constant(value=key)
            names += 1
    if natives != 1 or names != 1:
        raise RuntimeError("Full-warp factory ownership seam changed")
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    filename = f"<sparse-fullwarp-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    namespace["_group_source"] = _group_source
    exec(compile(source, filename, "exec"), namespace)
    return namespace[factory.name]


@cache
def get_solve_kernel():
    """Return the unchanged spectral ABI with four independent full warps."""
    original = sparse_spectral_tangents.get_solve_kernel().func.__globals__["get_solve_kernel"]
    factory = _factory(original, True, "sparse_spectral_tangent43_s18_c100_w4")
    return factory(100, False, metric_tangents=True)


@cache
def get_prefix_kernel():
    """Return the unchanged parallel-prefix ABI with four independent warps."""
    factory = _factory(
        sparse_factor_rows.get_parallel_limit_kernel.__wrapped__, False, "sparse_factor_parallel_limit_prefix43_w4"
    )
    return factory()
