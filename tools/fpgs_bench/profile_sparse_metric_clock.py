# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source-pinned, sampled clocks for the retained G1 metric owner only.

This is an external diagnostic factory, not a production solver option. The
six exclusive cycle fields sum to total_cycles. They measure lane-zero elapsed
SM cycles (including waiting), not GPU wall time. Total ends before diagnostic
stores. Callers own a separate zeroed (world_count, len(FIELDS)) uint64 buffer,
same-input output checks, and original-versus-observed event timing.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path

ROWS_SHA = "ec1303a009bab879b67e11d6521173f9441c3aee9e6e77da05c068c33677564b"
METRIC_SHA = "344461fb7e1e256801766b757f095742247ee0e047193d43919757ae7f105bd5"
FIELDS = (
    "total_cycles",
    "remainder_cycles",
    "metric_normal_cycles",
    "tangent_prepare_cycles",
    "metric_disk_cycles",
    "metric_commit_cycles",
    "scalar_cycles",
    "rows",
    "sweeps",
    "metric_visits",
    "positive_radius",
    "self_block_builds",
    "scalar_row_visits",
    "sliding_roots",
    "root_probes",
    "metric_accepted",
    "metric_rejected",
    "clock_reads",
)
BEGIN = "\n// METRIC_CLOCK_BEGIN\n"
END = "\n// METRIC_CLOCK_END\n"


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Frozen clock insertion seam changed: {old[:100]!r}")
    return source.replace(old, new)


def _block(text):
    return BEGIN + text + END


def strip_instrumentation(source):
    """Recover every original native byte by removing only marked insertions."""
    while BEGIN in source:
        start = source.index(BEGIN)
        end = source.index(END, start) + len(END)
        source = source[:start] + source[end:]
    return source


def instrument_native(source):
    """Add clocks at existing uniform boundaries without adding synchronization."""
    original = source
    declarations = r"""
    const bool mc_sample=lane==0 && sample_stride>0 && world%sample_stride==0;
    unsigned long long mc_start=0,mc_last=0;
    unsigned long long mc_rest=0,mc_normal=0,mc_tangent=0,mc_disk=0,mc_commit=0,mc_scalar=0;
    unsigned int mc_sweeps=0,mc_visits=0,mc_positive=0,mc_blocks=0,mc_scalar_rows=0;
    unsigned int mc_roots=0,mc_probes=0,mc_accepted=0,mc_rejected=0,mc_reads=0;
    int mc_phase=0;
    if(mc_sample) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(mc_start) :: "memory");
        mc_last=mc_start;mc_reads=1;
    }
    auto mc_mark=[&](int next) {
        if(mc_sample && mc_phase!=next) {
            unsigned long long now;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
            const unsigned long long elapsed=now-mc_last;
            if(mc_phase==0)mc_rest+=elapsed;
            else if(mc_phase==1)mc_normal+=elapsed;
            else if(mc_phase==2)mc_tangent+=elapsed;
            else if(mc_phase==3)mc_disk+=elapsed;
            else if(mc_phase==4)mc_commit+=elapsed;
            else if(mc_phase==5)mc_scalar+=elapsed;
            mc_last=now;mc_phase=next;++mc_reads;
        }
    };
"""
    anchor = "    if (!d.valid.data[world] || d.status.data[world] || count>100) return;"
    source = _once(source, anchor, _block(declarations) + anchor)
    anchor = "    for(int iteration=0;iteration<iterations;++iteration) {"
    source = _once(source, anchor, anchor + _block("        if(mc_sample)++mc_sweeps;\n"))
    anchor = "        for(int row=0;row<count;++row) {"
    source = _once(source, anchor, anchor + _block("            mc_mark(0);\n"))
    anchor = "                    const int tpl=d.support.data[base+row],length=p.support_count.data[tpl];"
    source = _once(source, anchor, _block("                    mc_mark(1);if(mc_sample)++mc_visits;\n") + anchor)
    anchor = "                        const float z1=lane<length && (radius>0.0f || old1!=0.0f)"
    source = _once(source, anchor, _block("                        mc_mark(2);\n") + anchor)
    anchor = "                        if(radius>0.0f) {"
    source = _once(source, anchor, anchor + _block("                            if(mc_sample)++mc_positive;\n"))
    anchor = "                            if(needs_block) {"
    source = _once(source, anchor, anchor + _block("                                if(mc_sample)++mc_blocks;\n"))
    anchor = (
        "                            if(lane==0) {\n                                r1+=contact_cross[row]*change0;"
    )
    source = _once(source, anchor, _block("                            mc_mark(3);\n") + anchor)
    anchor = "                                    if(isfinite(magnitude) && magnitude>radius && isfinite(beta_norm)) {"
    source = _once(
        source, anchor, anchor + _block("                                        if(mc_sample)++mc_roots;\n")
    )
    anchor = "                                            for(int probe=0;probe<16;++probe) {"
    source = _once(
        source, anchor, anchor + _block("                                                if(mc_sample)++mc_probes;\n")
    )
    anchor = "                        accepted=__shfl_sync(0xffffffff,accepted,0);"
    source = _once(source, anchor, _block("                        mc_mark(4);\n") + anchor)
    anchor = "                            if(lane==0) {lam[row]=next0;lam[row+1]=next1;lam[row+2]=next2;}"
    source = _once(source, anchor, _block("                            if(mc_sample)++mc_accepted;\n") + anchor)
    anchor = "                            __syncwarp();row+=2;continue;"
    source = _once(
        source,
        anchor,
        "                            __syncwarp();"
        + _block("                            mc_mark(0);\n")
        + "row+=2;continue;",
    )
    anchor = "                    // Rejection retains the original complete scalar triplet."
    source = _once(source, anchor, _block("                    if(mc_sample)++mc_rejected;mc_mark(0);\n") + anchor)
    anchor = "            if(type==2 && iteration<friction_start)"
    source = _once(source, anchor, _block("            mc_mark(5);if(mc_sample)++mc_scalar_rows;\n") + anchor)
    anchor = "        if(iteration>=friction_start && __ballot_sync(0xffffffff,changed)!=0u)continue;"
    source = _once(source, anchor, _block("        mc_mark(0);\n") + anchor)
    values = (
        "mc_last-mc_start",
        "mc_rest",
        "mc_normal",
        "mc_tangent",
        "mc_disk",
        "mc_commit",
        "mc_scalar",
        "count",
        "mc_sweeps",
        "mc_visits",
        "mc_positive",
        "mc_blocks",
        "mc_scalar_rows",
        "mc_roots",
        "mc_probes",
        "mc_accepted",
        "mc_rejected",
        "mc_reads",
    )
    publish = (
        "    mc_mark(-1);\n    if(mc_sample) {\n"
        + "".join(
            f"        diagnostic.data[(size_t)world*{len(FIELDS)}+{i}]={value};\n" for i, value in enumerate(values)
        )
        + "    }\n"
    )
    source = _once(source, "#endif", _block(publish) + "#endif")
    if strip_instrumentation(source) != original:
        raise AssertionError("Diagnostic changed original numerical source")
    for barrier in ("__syncwarp(", "__syncthreads(", "__ballot_sync("):
        if source.count(barrier) != original.count(barrier):
            raise AssertionError("Diagnostic changed original synchronization")
    return source


@functools.cache
def build_clock_kernel(capacity=100):
    """Return (separate kernel, metadata), appending diagnostic and sample_stride."""
    from newton._src.solvers.feather_pgs import sparse_factor_rows, sparse_metric_tangents  # noqa: PLC0415

    if capacity != 100:
        raise ValueError("Diagnostic only covers the retained capacity100 metric owner")
    for module, expected in ((sparse_factor_rows, ROWS_SHA), (sparse_metric_tangents, METRIC_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime source changed: {module.__name__}")
    original_factory = sparse_factor_rows.get_solve_kernel
    original = original_factory(capacity, metric_tangents=True)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original_factory)))
    factory = tree.body[0]
    factory.name = "metric_clock_factory"
    factory.decorator_list = []
    native_index = next(
        i for i, node in enumerate(factory.body) if isinstance(node, ast.FunctionDef) and node.name == "native"
    )
    factory.body[native_index:native_index] = ast.parse("source = _clock_rewrite(source)").body
    for node in factory.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("native", "solve"):
            node.args.args.extend(
                (
                    ast.arg(arg="diagnostic", annotation=ast.parse("wp.array2d[wp.uint64]", mode="eval").body),
                    ast.arg(arg="sample_stride", annotation=ast.Name(id="int", ctx=ast.Load())),
                )
            )
    calls = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "native"
    ]
    if len(calls) != 1:
        raise ValueError("Native call ABI seam changed")
    calls[0].args.extend((ast.Name(id="diagnostic", ctx=ast.Load()), ast.Name(id="sample_stride", ctx=ast.Load())))
    naming_index = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) for target in node.targets)
    )
    factory.body[naming_index:naming_index] = ast.parse("name += '_clock18'").body
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    digest = hashlib.sha256(generated.encode()).hexdigest()
    filename = f"<sparse-metric-clock-{digest}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    metadata = {
        "fields": FIELDS,
        "rows_sha256": ROWS_SHA,
        "metric_sha256": METRIC_SHA,
        "factory_sha256": digest,
        "original_kernel": original,
    }

    def rewrite(source):
        metadata["original_native"] = source
        result = instrument_native(source)
        metadata["instrumented_native"] = result
        metadata["native_sha256"] = hashlib.sha256(result.encode()).hexdigest()
        return result

    namespace = dict(original_factory.__wrapped__.__globals__)
    namespace["_clock_rewrite"] = rewrite
    exec(compile(generated, filename, "exec"), namespace)
    kernel = namespace[factory.name](capacity, metric_tangents=True)
    if list(inspect.signature(kernel.func).parameters) != [
        *inspect.signature(original.func).parameters,
        "diagnostic",
        "sample_stride",
    ]:
        raise AssertionError("Diagnostic changed original kernel argument order")
    return kernel, metadata
