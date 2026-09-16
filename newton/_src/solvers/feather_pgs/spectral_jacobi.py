# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental concurrent spectral contacts with a permanent half-step latch.

Retain original staging, whitening, capacities, physical decode and unsupported
fallback. Cold admitted worlds replace EX1/Nesterov with at most 24 simultaneous
normal-first spectral proposals and one transpose/row action pair per pass.
There is no ordered contact loop, local root or Schur preparation.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path

from . import coupled_contact, coupled_jacobi, spectral_contact

SOURCE_PINS = {
    coupled_contact.__file__: "3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015",
    coupled_jacobi.__file__: "3da318644dad50d0f56955bd72ad042ab07ca21cfc7dc85d3b13f0098c75cf3d",
    spectral_contact.__file__: "aedd42a5ef76041620c7333bf34029c06d12d1410aeed1deaf366befa8f69943",
}

# Reuse only cold admission and the three physical cross/spectral coefficients,
# not the ordered spectral pass or its separate residual stopping scan.
_SEAM = "    if(!sg_bad){\n        if(lane<18)s_dv[lane]=0.0f;"
if spectral_contact._NATIVE.count(_SEAM) != 1:
    raise RuntimeError("Frozen spectral admission seam changed")
_INITIAL = spectral_contact._NATIVE.split(_SEAM)[0]
_INITIAL = _INITIAL.replace(
    "    __shared__ float sg_diag[AM],sg_physical[AM],sg_cross[AM],sg_spectral[AM];\n    __shared__ int sg_changed;",
    "    __shared__ float sg_diag[AM],sg_physical[AM],sg_cross[AM];",
)
_INITIAL = _INITIAL.replace(";sg_spectral[lane]=norm", "")
_INITIAL = _INITIAL.replace("            sg_spectral[lane+1]=spectral;sg_spectral[lane+2]=spectral;\n", "")

_PROPOSAL = r"""
    if(!sg_bad){
        float sj_local_scale=0.0f;
        if(lane<n_rows)sj_local_scale=fabsf(s_rhs[lane])/sqrtf(sg_diag[lane]);
        const float sj_scale=1.0f+sg_max(sj_local_scale);
        auto sj_propose = [&](int row) -> wp::vec3 {
            const wp::vec3 old(s_x[row],s_x[row+1],s_x[row+2]);
            const wp::vec3 residual(s_rhs[row],s_rhs[row+1],s_rhs[row+2]);
            const wp::vec4 coefficients(sg_diag[row],sg_cross[row+2],sg_cross[row],sg_cross[row+1]);
            return sg_project(old,residual,coefficients,s_mu[row+1]);
        };
"""

# Preserve exactly the already-tested precision-consistent merit, permanent
# latch, two operators, padded float4 input and affine midpoint dataflow.
_MEASURE_SEAM = "        const float cone_tolerance=3.0e-5f;"
if coupled_jacobi._JACOBI_NATIVE.count(_MEASURE_SEAM) != 1:
    raise RuntimeError("Frozen concurrent Jacobi recurrence seam changed")
_RECURRENCE = _MEASURE_SEAM + coupled_jacobi._JACOBI_NATIVE.split(_MEASURE_SEAM)[1]
for _old, _new in (("cj_", "sj_"), ("cc_diag", "sg_diag"), ("cc_scale", "sj_scale"), ("cc_max", "sg_max")):
    _RECURRENCE = _RECURRENCE.replace(_old, _new)


def _rewrite_native(source):
    """Replace the admitted solve region without retaining old/root work."""
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    replacement = spectral_contact._LOCAL_NATIVE + _INITIAL + _PROPOSAL + _RECURRENCE
    source = coupled_contact._replace_once(source, start, replacement + "\n    if (sg_bad) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    source = coupled_contact._replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = coupled_contact._replace_once(source, "    float t_k = 1.0f;", "    if (sg_bad) {\n    float t_k = 1.0f;")
    return coupled_contact._replace_once(
        source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)"
    )


@functools.cache
def get_parallel_factory(original):
    """Return the opt-in ABI-identical root-free Jacobi factory."""
    for name, digest in SOURCE_PINS.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen native dependency changed: {name}")
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "spectral_jacobi_parallel_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_sj24"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_fpgs_coupled_contact"):
            node.attr = node.attr.replace("_fpgs_coupled_contact", "_fpgs_spectral_jacobi")
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<spectral-jacobi-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(successor.__globals__, _rewrite_native=_rewrite_native)
    exec(compile(generated, filename, "exec"), namespace)
    implementation = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not coupled_contact._supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)

    return wrapped
