# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the PTX of FeatherPGS kernels by dataflow: a schedule-only change against a rounding one.

The rung after ``kernel_sass.py``: where a kernel's SASS changed, its PTX
entries (``kernel_sass.py record --ptx DIR`` on both sides) are compared by
what they compute, not by their text. Every instruction's result is named by its
opcode (with modifiers) and its operands' names, a register by its latest
definition in program order; the operands of commutative operations (add, mul,
min, max, and, or, xor) and ``fma``'s two factors are sorted, so commutation is
exact. A listing's signature is the multiset of its stores (address, value,
predicate) and of its floating-point results:

* **identical**: the PTX text is equal (up to the kernel's hash suffix);
* **schedule-only**: the same values reach memory by the same floating-point
  operations, in another order or registers: results bitwise, registers and
  speed may move;
* **rounding**: the floating-point results differ, an operation changed (a
  contraction moved).

Exact on straight-line code; inside a loop a register is taken at its latest
definition, so a reordered loop body can read "rounding" with equal operation
counts, which the report prints for that case.

    python tools/fpgs_bench/kernel_flow.py diff parent_ptx/ child_ptx/
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

COMMUTATIVE = ("add", "mul", "min", "max", "and", "or", "xor")
_INSTRUCTION = re.compile(r"^\s*(@!?%p\d+\s+)?([a-z][\w.]*)\s*(.*?);\s*$")
_FP = re.compile(r"(fma|add|sub|mul|div|rcp|sqrt|rsqrt|neg|abs)\.")
_SUFFIX = re.compile(r"_[0-9a-f]{7,8}(?=[_A-Za-z0-9])")


def _name(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def ptx_path(directory, label) -> Path:
    """The file of one inventory label's PTX entry in ``directory`` (the label's digest; ``index.json`` names it)."""
    directory = Path(directory)
    name = _name(label) + ".ptx"
    index = directory / f"index-{name}.json"
    index.write_text(json.dumps({"label": label, "file": name}))
    return directory / name


def entry(ptx: str, function: str) -> str:
    """One function's entry of a module's PTX, its hash suffixes removed."""
    start = ptx.index(function)
    end = ptx.find("\n}\n", start)
    return _SUFFIX.sub("_H", ptx[start : end if end >= 0 else len(ptx)])


def signature(text: str):
    """Return the multisets of a listing's stores and floating-point results."""
    defs = {}
    stores, floats = collections.Counter(), collections.Counter()

    def value(token):
        token = token.strip()
        if token.startswith("%"):
            return defs.get(token, "in:" + re.sub(r"\d+", "", token))
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            offset = re.match(r"(%\w+)\s*\+\s*(-?\d+)", inner)
            if offset:
                return _name("addr", value(offset.group(1)), offset.group(2))
            return _name("addr", value(inner), "0")
        return "imm:" + token

    for line in text.splitlines():
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        predicate, op, args = match.group(1), match.group(2), match.group(3)
        if op.startswith(("bra", "ret", "bar", "exit", "membar", ".")):
            continue
        operands = [a.strip() for a in re.split(r",(?![^{]*})", args) if a.strip()]
        guard = value(predicate.strip().lstrip("@!")) if predicate else ""
        if op.startswith("st."):
            stores[_name(op, value(operands[0]), value(operands[1]), guard)] += 1
            continue
        if not operands:
            continue
        target, sources = operands[0], [value(a) for a in operands[1:]]
        base = op.split(".")[0]
        if base in COMMUTATIVE and len(sources) == 2:
            sources = sorted(sources)
        if base == "fma" and len(sources) == 3:
            sources = sorted(sources[:2]) + sources[2:]
        result = _name(op, guard, *sources)
        if target.startswith("{"):
            for k, register in enumerate(target.strip("{}").split(",")):
                defs[register.strip()] = _name(result, str(k))
        else:
            defs[target] = result
        if ".f32" in op or ".f64" in op:
            floats[_name(op, *sources)] += 1
    return stores, floats


def operation_counts(text: str) -> collections.Counter:
    """Floating-point arithmetic instructions by opcode."""
    return collections.Counter(
        m.group(2)
        for line in text.splitlines()
        if (m := _INSTRUCTION.match(line)) and _FP.match(m.group(2)) and (".f32" in m.group(2) or ".f64" in m.group(2))
    )


def classify(parent: str, child: str) -> tuple[str, str]:
    """Return ``(class, detail)`` of two PTX entries: identical, schedule-only or rounding."""
    if parent == child:
        return "identical", ""
    (stores_a, floats_a), (stores_b, floats_b) = signature(parent), signature(child)
    counts_a, counts_b = operation_counts(parent), operation_counts(child)
    moved = {
        k: (counts_a[k], counts_b[k]) for k in sorted(counts_a.keys() | counts_b.keys()) if counts_a[k] != counts_b[k]
    }
    detail = (
        f"stores {'same' if stores_a == stores_b else 'differ'} ({sum(stores_a.values())}/{sum(stores_b.values())}), "
        f"fp results {'same' if floats_a == floats_b else 'differ'}, fp op counts {moved or 'same'}"
    )
    if stores_a == stores_b and floats_a == floats_b:
        return "schedule-only", detail
    return "rounding", detail


def _labels(directory):
    return {
        json.loads(p.read_text())["label"]: Path(directory) / json.loads(p.read_text())["file"]
        for p in Path(directory).glob("index-*.json")
    }


def report(parent_dir, child_dir) -> int:
    """Print each shared kernel's class; return 1 when one is rounding, else 0."""
    parent, child = _labels(parent_dir), _labels(child_dir)
    classes = collections.Counter()
    for label in sorted(parent.keys() & child.keys()):
        kind, detail = classify(parent[label].read_text(), child[label].read_text())
        classes[kind] += 1
        if kind != "identical":
            print(f"  {kind}: {label}: {detail}")
    print(f"{sum(classes.values())} shared kernels: " + ", ".join(f"{n} {k}" for k, n in sorted(classes.items())))
    return int(classes["rounding"] > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    diff_parser = commands.add_parser("diff", help="compare two kernel_sass --ptx directories")
    diff_parser.add_argument("parent")
    diff_parser.add_argument("child")
    arguments = parser.parse_args(argv)
    return report(arguments.parent, arguments.child)


if __name__ == "__main__":
    sys.exit(main())
