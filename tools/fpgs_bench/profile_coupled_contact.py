# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only coupled owner; never use this clone for timing.

The unchanged ABI's unused ``world_row_w`` argument must be a private float32
array of the original [world_count, max_constraints] shape, initialized to -1.
Regularization must be disabled, as required by the production fast admission.
Each admitted tier writes COUNTERS in that world's prefix AFTER all physical
outputs. Untouched worlds retain -1. Repeated launches overwrite, not accumulate.
Keep all original argument/buffer leases alive through graph execution.

Instrumentation strips byte-exactly to the frozen native source. It nevertheless
changes compiler resource allocation; compare physical outputs with the frozen
owner and do not interpret its latency. root_probes counts all 2x2 evaluations,
including zero and upper endpoints; stop categories count rejecting sweeps, not
rows. last_stop_mask uses bits nonfinite/normal/complementarity/MDP/natural/cone.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import re
import textwrap
from pathlib import Path

from newton._src.solvers.feather_pgs import coupled_contact

RUNTIME_SHA = "3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015"
COUNTERS = (
    "consumed",
    "done",
    "admission_bad",
    "block_calls",
    "root_calls",
    "root_probes",
    "metric_fallbacks",
    "upper_rejects",
    "schur_rejects",
    "root_exhausted",
    "local_guard_rejects",
    "stop_checks",
    "stop_nonfinite",
    "stop_normal",
    "stop_complementarity",
    "stop_mdp",
    "stop_natural",
    "stop_cone",
    "last_stop_mask",
    "n_rows",
    "changed_last",
    "scalar_fallbacks",
)


def _tag(code):
    return "/*CC_DIAG_BEGIN*/" + code + "/*CC_DIAG_END*/"


def strip_instrumentation(source):
    return re.sub(r"/\*CC_DIAG_BEGIN\*/.*?/\*CC_DIAG_END\*/", "", source, flags=re.S)


def instrument_native(source):
    """Insert output-only counters without changing a physical expression."""
    original = source

    def insert(old, before="", after=""):
        nonlocal source
        source = coupled_contact._replace_once(
            source, old, (_tag(before) if before else "") + old + (_tag(after) if after else "")
        )

    insert(
        "auto cc_norm2 =",
        before="""
int diag_blocks=0,diag_roots=0,diag_probes=0,diag_metric=0,diag_upper=0;
int diag_schur=0,diag_exhausted=0,diag_guard=0,diag_scalar=0;
int diag_stop_count=0,diag_stops[6]={0,0,0,0,0,0};
unsigned diag_last=0;
__shared__ unsigned diag_warp_stop[2];
""",
    )
    insert("    wp::vec_t<5,float> out(0.0f);", after="\n++diag_blocks;")
    insert("    if (!(Q[3]>0.0f)) return out;", before="\nif(!(Q[3]>0.0f))++diag_schur;\n")
    insert("    auto eval = [&](float gamma, float& n, float& x, float& y, float& g) {", after="\n++diag_probes;")
    insert("    const float rmin=mu*(-b[0])/(a+mu*cc_norm2(h0,h1));", before="\n++diag_roots;\n")
    insert("    if (!eval(upper,un,ux,uy,ug) || ug>1.0e-12f*wp::max(rmin,1.0f)) return out;", before="")
    # Braces are tagged too, leaving the original single-statement branch exact.
    old = "if (!eval(upper,un,ux,uy,ug) || ug>1.0e-12f*wp::max(rmin,1.0f)) return out;"
    source = coupled_contact._replace_once(
        source, old, old.replace("return out;", _tag("{++diag_upper;") + "return out;" + _tag("}"))
    )
    insert("            return out;\n        }\n        if(g>0.0f)", before="")
    source = coupled_contact._replace_once(
        source,
        "            return out;\n        }\n        if(g>0.0f)",
        _tag("if(out[3]==0.0f)++diag_guard;\n") + "            return out;\n        }\n        if(g>0.0f)",
    )
    insert("    return out;\n};\n", before="\n++diag_exhausted;\n")
    insert(
        "                                const wp::vec4 metric=cc_metric(H,residual,old,s_mu[row+1]);",
        before="\n++diag_metric;\n",
    )
    insert(
        "                                // Original scalar triplet after metric rejection.",
        before="\n++diag_scalar;\n",
    )
    insert("            int failed=0;", after="\nunsigned diag_mask=0;")
    insert("                failed=!isfinite(r)||!isfinite(value);", after="\nif(failed)diag_mask|=1u;")
    insert(
        "                    const float correction=value-fmaxf(0.0f,value-r/cc_diag[lane]);",
        after="""
if(value< -1.0e-10f)diag_mask|=32u;
if(r< -3.0e-5f)diag_mask|=2u;
if(fabsf(value*r)>3.0e-5f)diag_mask|=4u;
if(sqrtf(cc_diag[lane])*fabsf(correction)>1.0e-5f*cc_scale)diag_mask|=16u;
""",
    )
    insert(
        "                    const float mdp=fabsf(value*r+other*rr+radius*hypotf(r,rr));",
        after="""
if(!isfinite(mdp))diag_mask|=1u;
if(hypotf(value,other)-radius>1.0e-10f)diag_mask|=32u;
if(mdp>3.0e-5f)diag_mask|=8u;
if(sqrtf(cc_diag[lane])*fabsf(value-x*factor)>1.0e-5f*cc_scale||
   sqrtf(cc_diag[lane+1])*fabsf(other-y*factor)>1.0e-5f*cc_scale)diag_mask|=16u;
""",
    )
    insert(
        "            cc_done=cc_any(failed)==0;",
        before="""
for(int shift=16;shift>0;shift>>=1)diag_mask|=__shfl_xor_sync(MASK,diag_mask,shift);
if((lane&31)==0)diag_warp_stop[lane>>5]=diag_mask;
SYNC();
diag_last=diag_warp_stop[0]|(NT>32?diag_warp_stop[1]:0u);
++diag_stop_count;
for(int bit=0;bit<6;++bit)diag_stops[bit]+=(int)((diag_last>>bit)&1u);
SYNC();
""",
    )
    insert(
        "#undef A_AT",
        before="""
SYNC();
if(lane==0){
    const int values[22]={cc_consumed,(int)cc_done,cc_bad,diag_blocks,diag_roots,diag_probes,
        diag_metric,diag_upper,diag_schur,diag_exhausted,diag_guard,diag_stop_count,
        diag_stops[0],diag_stops[1],diag_stops[2],diag_stops[3],diag_stops[4],diag_stops[5],
        (int)diag_last,n_rows,cc_consumed>0?cc_changed:0,diag_scalar};
    for(int field=0;field<22;++field)world_row_w.data[off_dense+field]=(float)values[field];
}
""",
    )
    if strip_instrumentation(source) != original:
        raise RuntimeError("Diagnostic instrumentation changed original native bytes")
    return source


@functools.cache
def get_parallel_factory(original):
    """Clone only the frozen candidate factory; same arguments, diagnostic key."""
    if hashlib.sha256(Path(coupled_contact.__file__).read_bytes()).hexdigest() != RUNTIME_SHA:
        raise RuntimeError("Coupled diagnostic runtime pin changed")
    selected = coupled_contact.get_parallel_factory(original)
    successor = inspect.getclosurevars(selected).nonlocals["successor"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    factory = tree.body[0]
    factory.name = "diagnostic_coupled_contact_factory"
    for node in ast.walk(factory):
        if isinstance(node, ast.Constant) and node.value == "_cc6":
            node.value = "_cc6_diag"
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<coupled-contact-diagnostic-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(successor.__globals__)
    namespace["_rewrite_native"] = lambda source: instrument_native(coupled_contact._rewrite_native(source))
    exec(compile(generated, filename, "exec"), namespace)
    diagnostic = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not coupled_contact._supported(bound.arguments, original.__globals__):
            raise ValueError("Diagnostic requires the frozen admitted coupled configuration")
        if bound.arguments["max_constraints"] < len(COUNTERS):
            raise ValueError("Diagnostic counter bank exceeds original row capacity")
        return diagnostic(*args, **kwargs)

    return wrapped
