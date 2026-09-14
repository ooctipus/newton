# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain the existing ANYmal EX1 Gram products in per-row registers."""

import ast
import functools
import hashlib
import inspect
import linecache
import os
import textwrap


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Register Gram original native seam changed")
    return source.replace(old, new)


def _rewrite_native(source, rows):
    """Replace only EX1 product lifetime and the repeated residual contraction."""
    declarations = "\n".join(f"    float rg_a{j} = 0.0f;" for j in range(rows))
    source = _replace_once(source, "    float Jr[18];", "    float Jr[18];\n" + declarations)
    old = """            for (int j = 0; j < n_rows; ++j) {
                if (s_kind[j] < 0) continue;
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < 18; ++d) acc += Ji[d] * s_Yt[d * YS + j];
                if (j == i) aii = acc;
                bound += fabsf(acc);
            }"""
    # Keep the compiler from loading a four-column, 72-value Z tile before
    # consuming it; that temporary exceeds the retained coefficients' lifetime.
    products = ["            const volatile float* rg_Z = s_Yt;"]
    for j in range(rows):
        # Frozen rows still contribute to the residual if their incoming impulse
        # is nonzero. Only the majorizer excludes them, as in the original law.
        products.append(f"""            if ({j} < n_rows) {{
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < 18; ++d) acc += Ji[d] * rg_Z[d * YS + {j}];
                rg_a{j} = acc;
                if (s_kind[{j}] >= 0) {{
                    if ({j} == i) aii = acc;
                    bound += fabsf(acc);
                }}
            }}""")
    source = _replace_once(source, old, "\n".join(products))
    start = source.index("#if 1\n        {\n            // dv = Y^T y")
    end = source.index("#endif\n        #pragma unroll", start) + len("#endif\n")
    source = (
        source[:start] + "        // Retained EX1 coefficients replace the factor-space reduction.\n" + source[end:]
    )
    old = """                    const float* Ji = Jr;
                    float r = s_rhs[i];
                    #pragma unroll
                    for (int d = 0; d < 18; ++d) r += Ji[d] * s_dv[d];"""
    product = [
        "                    float r0 = s_rhs[i], r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;",
        "                    const float4* y4 = reinterpret_cast<const float4*>(s_y);",
    ]
    for j in range(0, rows, 4):
        product.append(f"""                    if ({j} < n_rows) {{
                        const float4 yv = y4[{j // 4}];
                        r0 += rg_a{j} * yv.x;
                        r1 += rg_a{j + 1} * yv.y;
                        r2 += rg_a{j + 2} * yv.z;
                        r3 += rg_a{j + 3} * yv.w;
                    }}""")
    product.append("                    const float r = (r0 + r1) + (r2 + r3);")
    return _replace_once(source, old, "\n".join(product))


def _supported(arguments, globals_):
    return (
        arguments["max_world_dofs"] == 18
        and arguments["rows"] in (32, 48)
        and arguments["min_rows"] == (0 if arguments["rows"] == 32 else 32)
        and arguments["sweeps"] == 24
        and arguments["nesterov"]
        and arguments["matrix_free"]
        and arguments["inkernel_response"] == (18, 0, 0, 0)
        and arguments["exact_row_sums"]
        and arguments["world_rows"]
        and not any(
            arguments[name]
            for name in (
                "has_drive_rows",
                "has_dense_velocity_limit_rows",
                "skip_local_internal_worlds",
                "lean_sweep",
                "bound_finger",
            )
        )
        and not any(
            globals_.get(name, False) for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING")
        )
        and int(os.environ.get("FEATHER_PGS_WR_STOP", "0")) == 0
    )


@functools.cache
def get_parallel_factory(original):
    """Wrap the original ABI, preserving unsupported and default factories."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    factory = tree.body[0]
    factory.name = "register_gram_parallel_factory"
    factory.decorator_list = []
    native = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.FunctionDef) and node.name == "pgs_solve_parallel_native"
    )
    factory.body[native:native] = ast.parse("snippet = _rewrite_native(snippet, AM)").body
    naming = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "__name__" for target in node.targets)
    )
    factory.body[naming:naming] = ast.parse("name += '_rg1'").body
    final = next(i for i, node in enumerate(factory.body) if isinstance(node, ast.Return))
    factory.body[final:final] = ast.parse(
        "kernel._fpgs_register_gram = True\nkernel._fpgs_register_gram_native = snippet"
    ).body
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<register-gram-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(original.__globals__)
    namespace["_rewrite_native"] = _rewrite_native
    exec(compile(generated, filename, "exec"), namespace)
    successor = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def selected(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return successor(*args, **kwargs)

    return selected


def validate_solver(solver, source):
    """Reject a requested mode that cannot own the ordinary ANYmal tiers."""
    if not (
        solver.model.device.is_cuda
        and solver.grouped_dynamics
        and solver.max_world_dofs == 18
        and solver.mf_gs_parallel_rows == 48
        and solver.mf_gs_parallel_sweeps == 24
        and solver.mf_gs_parallel_nesterov
        and solver.mf_gs_parallel_matrix_free
        and solver.mf_gs_response_block_rows == 0
        and solver.dense_max_constraints >= 48
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and not solver.fuse_joint_velocity_limits
        and solver.drive_mode != "physx_pgs"
        and not solver.pgs_warmstart
        and not solver._paired_factor_coordinates
        and not solver._local_internal_fast_path
        and source._INK_ON
        and source._WR_ON
        and source._MF_EXACT_ROWSUM
        and not any(
            getattr(source, name)
            for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN")
        )
        and int(os.environ.get("FEATHER_PGS_WR_STOP", "0")) == 0
    ):
        raise ValueError("Register Gram requires ordinary ANYmal18 INK/WR/EX1 matrix-free Nesterov24 tiers32/48")
