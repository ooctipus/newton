# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental two-world warp ownership of the unchanged sparse metric GS law.

Each half warp owns a complete world, including all eighteen support entries.
The source-pinned generator changes ownership and reductions only; the original
metric root, scalar fallback, row order, stopping rule, and decode are retained.
"""

import ast
import hashlib
import inspect
import linecache
import re
import textwrap
from functools import cache
from types import MethodType

import warp as wp

from . import sparse_factor_rows, sparse_metric_tangents

_SOLVE_SHA = "a360992bc13d3a08b0bf55673b026dc30895de3d3901228457d87d2fa4cf8c1a"
_METRIC_SHA = "d56bfaed9e634191395ea91acad0987466cdb9b292a124b2bd18c84bb03d4938"
_KEY = "sparse_paired_metric_tangent43_s18_c100"


def _source(function, expected):
    source = textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError("Paired sparse GS original source changed")
    return source


def _paired_source(source):
    """Map the pinned complete scalar/metric transaction to two half warps."""

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError(f"Paired sparse GS source seam changed: {old}")
        source = source.replace(old, new)

    replace(
        "const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];",
        """const int half=threadIdx.x>>4, lane=threadIdx.x&15;
    const int group=pack*2+half;
    if(group>=p.group_to_art.shape[0])return;
    const unsigned int mask=0xffffu<<(half*16);
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];""",
    )
    replace(
        "__shared__ float du[43], lam[100];",
        """__shared__ float du_storage[86], lam_storage[200];
    float* du=du_storage+half*43;
    float* lam=lam_storage+half*100;""",
    )
    replace(
        "__shared__ float contact_cross[100];",
        """__shared__ float contact_cross_storage[200];
    float* contact_cross=contact_cross_storage+half*100;""",
    )
    replace(
        "__shared__ unsigned int contact_ready[4];",
        """__shared__ unsigned int contact_ready_storage[8];
    unsigned int* contact_ready=contact_ready_storage+half*4;""",
    )
    # The scalar and metric scopes each own their secondary support element.
    replace(
        "const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;",
        """const int node=lane<length?p.support_nodes.data[tpl*18+lane]:-1;
                    const int node_tail=lane+16<length?p.support_nodes.data[tpl*18+lane+16]:-1;""",
        count=2,
    )
    replace(
        "const float z=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;",
        """const float z=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;
            const float z_tail=lane+16<length?d.Z.data[(base+row)*18+lane+16]:0.0f;""",
    )
    replace(
        "float dot=lane<length?z*du[node]:0.0f;",
        "float dot=(lane<length?z*du[node]:0.0f)+(lane+16<length?z_tail*du[node_tail]:0.0f);",
    )
    replace(
        "if(lane<sn)du[p.support_nodes.data[st*18+lane]]+=d.Z.data[(base+sibling)*18+lane]*sibling_delta;",
        """if(lane<sn)du[p.support_nodes.data[st*18+lane]]+=d.Z.data[(base+sibling)*18+lane]*sibling_delta;
                if(lane+16<sn)du[p.support_nodes.data[st*18+lane+16]]+=d.Z.data[(base+sibling)*18+lane+16]*sibling_delta;""",
    )
    replace(
        "if(delta!=0.0f) { if(lane<length)du[node]+=z*delta;changed=1; }",
        """if(delta!=0.0f) {
                if(lane<length)du[node]+=z*delta;
                if(lane+16<length)du[node_tail]+=z_tail*delta;
                changed=1;
            }""",
    )
    replace(
        "const float z0=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;",
        """const float z0=lane<length?d.Z.data[(base+row)*18+lane]:0.0f;
                    const float z0_tail=lane+16<length?d.Z.data[(base+row)*18+lane+16]:0.0f;""",
    )
    replace(
        "float r0=lane<length?z0*du[node]:0.0f;",
        "float r0=(lane<length?z0*du[node]:0.0f)+(lane+16<length?z0_tail*du[node_tail]:0.0f);",
    )
    for direction in (1, 2):
        replace(
            f"const float z{direction}=lane<length && (radius>0.0f || old{direction}!=0.0f)\n"
            f"                            ?d.Z.data[(base+row+{direction})*18+lane]:0.0f;",
            f"""const float z{direction}=lane<length && (radius>0.0f || old{direction}!=0.0f)
                            ?d.Z.data[(base+row+{direction})*18+lane]:0.0f;
                        const float z{direction}_tail=lane+16<length && (radius>0.0f || old{direction}!=0.0f)
                            ?d.Z.data[(base+row+{direction})*18+lane+16]:0.0f;""",
        )
    replace(
        "float a01=z0*z1,a02=z0*z2,a12=z1*z2;",
        "float a01=z0*z1+z0_tail*z1_tail,a02=z0*z2+z0_tail*z2_tail,a12=z1*z2+z1_tail*z2_tail;",
    )
    replace(
        "float r1=lane<length?z1*du[node]:0.0f,r2=lane<length?z2*du[node]:0.0f;",
        """float r1=(lane<length?z1*du[node]:0.0f)+(lane+16<length?z1_tail*du[node_tail]:0.0f);
                            float r2=(lane<length?z2*du[node]:0.0f)+(lane+16<length?z2_tail*du[node_tail]:0.0f);""",
    )
    replace(
        "const int bad_coeff=lane<length && (!isfinite(z0) || !isfinite(z1) || !isfinite(z2));",
        """const int bad_coeff=(lane<length && (!isfinite(z0) || !isfinite(z1) || !isfinite(z2))) ||
                            (lane+16<length && (!isfinite(z0_tail) || !isfinite(z1_tail) || !isfinite(z2_tail)));""",
    )
    replace(
        "if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;",
        """if(lane<length)du[node]+=z0*change0+z1*change1+z2*change2;
                                if(lane+16<length)du[node_tail]+=z0_tail*change0+z1_tail*change1+z2_tail*change2;""",
    )
    # Keep bitmap word=row/32 and bit=row%32 intact: those are not work strides.
    replace("r+=32", "r+=16", count=3)
    replace("k+=32", "k+=16")
    replace("col+=32", "col+=16")
    replace("int s=16;", "int s=8;")
    replace("int shift=16;", "int shift=8;", count=3)
    source = re.sub(r"__shfl_down_sync\(0xffffffff,(\w+),(\w+)\)", r"__shfl_down_sync(mask,\1,\2,16)", source)
    source = re.sub(r"__shfl_sync\(0xffffffff,(\w+),0\)", r"__shfl_sync(mask,\1,0,16)", source)
    source = source.replace("__ballot_sync(0xffffffff,", "__ballot_sync(mask,")
    source = source.replace("__syncwarp()", "__syncwarp(mask)")
    if "0xffffffff" in source or "__syncthreads" in source or "+=32" in source:
        raise RuntimeError("Paired sparse GS retains a whole-warp ownership operation")
    return source


@cache
def get_solve_kernel():
    """Return the two-world, fourteen-argument original metric-solve ABI."""
    _source(sparse_metric_tangents.get_fragments, _METRIC_SHA)
    original = sparse_factor_rows.get_solve_kernel.__wrapped__
    tree = ast.parse(_source(original, _SOLVE_SHA))
    factory = tree.body[0]
    factory.name = "_paired_factory"
    factory.decorator_list = []
    native_count = 0
    rename_count = 0
    for index, statement in enumerate(factory.body):
        if isinstance(statement, ast.FunctionDef) and statement.name == "native":
            statement.args.args[0].arg = "pack"
            factory.body.insert(index, ast.parse("source = _paired_source(source)").body[0])
            native_count += 1
            break
    for statement in factory.body:
        if isinstance(statement, ast.Assign) and any(
            ast.unparse(target) == "solve.__name__" for target in statement.targets
        ):
            statement.value = ast.Constant(value=_KEY)
            rename_count += 1
    if native_count != 1 or rename_count != 1:
        raise RuntimeError("Paired sparse GS factory seam changed")
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    filename = f"<sparse-paired-gs-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    namespace["_paired_source"] = _paired_source
    exec(compile(source, filename, "exec"), namespace)
    return namespace[factory.name](100, False, metric_tangents=True)


def _solve(owner, rhs, iterations, omega, friction_start):
    s = owner.solver
    wp.launch_tiled(
        owner.kernels.solve,
        dim=[(s.world_count + 1) // 2],
        inputs=[
            owner.plan,
            owner.data,
            s.constraint_count,
            rhs,
            s.diag,
            s.impulses,
            s.row_type,
            s.row_parent,
            s.row_mu,
            iterations,
            omega,
            friction_start,
            s.v_hat,
            s.v_out,
        ],
        block_dim=32,
        device=s.model.device,
    )


def install(owner):
    """Replace only an unmodified capacity100 metric owner and its launch."""
    if getattr(owner, "paired_gs", False):
        return True
    if (
        not owner.metric_tangents
        or owner.packet_rows
        or owner.block_contacts
        or owner.solver.dense_max_constraints != 100
        or owner.kernels.solve is not sparse_factor_rows.get_solve_kernel(100, False, metric_tangents=True)
        or "solve" in owner.__dict__
    ):
        return False
    owner.kernels.solve = get_solve_kernel()
    owner.solve = MethodType(_solve, owner)
    owner.paired_gs = True
    return True
