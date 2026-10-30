# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One source per physical law, for Warp kernels and native CUDA sweeps alike.

A law is written once, as a Python function in the subset of Warp's kernel
language described below, and registered with a :class:`LawFamily`. The family
gives it two renderings:

* the Warp function: :meth:`LawFamily.law` returns ``wp.func`` of the
  definition, which Warp kernels call like any other Warp function. Warp
  compiles it for every device, in the calling module's options (``fuse_fp``
  included);
* the CUDA function: a static member of the family's struct, which native
  sweeps call as ``Family::law(...)`` after declaring the family's device
  prelude, :meth:`LawFamily.prelude` (whose CPU twin serves native bodies that
  also run on the CPU, ``device_preludes.dual_native``). :func:`render_cuda`
  writes it as a sweep's author writes CUDA: one statement per Python
  statement, one operation per Python operation in the same order, plain
  ``if``/``else``, loops and component subscripts, so it compiles to the code
  of the hand-written text it replaces.

Warp kernels call the Warp function rather than reach the CUDA text through
``wp.func_native``: the Warp function keeps their results bit for bit. The two
renderings compute the same operations in the same order but are lowered
differently, Warp merging each branch's values with ``wp::where``, so the CUDA
compiler may fuse a multiply-add in one and not in the other, and Warp negates a scalar by subtracting it from zero. Laws without
an iterative solve return bitwise equal results up to the sign of a zero; the
iterative ones (the friction pair, the Coulomb bisection) agree to their
solve's tolerance.

The subset: scalar (``float``, ``int``, ``bool``, ``wp.float64``), vector,
matrix, quaternion and transform arguments and locals, read-only Warp arrays,
arithmetic, comparisons, ``and``/``or``/``not``, conditional expressions,
``if``/``elif``/``else``, ``for`` over ``range`` (``wp.static`` bounds allowed),
``while``, ``break``, ``continue`` and ``return`` of one value or a tuple, calls
of Warp builtins and of registered laws, and module constants. A local is
declared at its first assignment and lives in its block, as in C++, so a value
used after an ``if`` is assigned before it. A tuple-valued law renders a result
struct and is called only in a tuple assignment.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude

__all__ = ["LawFamily", "LawSignature", "law_preludes", "law_signature", "render_cuda"]

# Function qualifiers by inlining choice. A family's prelude sits after the native snippet's architecture guard
# (``device_preludes.cuda_native``); its CPU twin, for bodies ``dual_native`` composes, is the same text inline.
_INLINE = {
    "force": "__device__ __forceinline__",
    "default": "__device__ inline",
    "never": "__device__ __noinline__",
}
_CPU_INLINE = "inline"

# C++ keywords a Python name may collide with; such names take a trailing underscore.
_CPP_KEYWORDS = frozenset(
    "alignas alignof asm auto bool break case catch char class const constexpr const_cast continue decltype default "
    "delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long "
    "mutable namespace new noexcept nullptr operator private protected public register reinterpret_cast return short "
    "signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef "
    "typeid typename union unsigned using virtual void volatile wchar_t while".split()
)

# Python operators and their C++ spelling and precedence (higher binds tighter).
_BINARY = {
    ast.Add: ("+", 9),
    ast.Sub: ("-", 9),
    ast.Mult: ("*", 10),
    ast.Div: ("/", 10),
    ast.FloorDiv: ("/", 10),  # Warp's integer floor division is C++ division
    ast.Mod: ("%", 10),
    ast.BitAnd: ("&", 5),
    ast.BitOr: ("|", 3),
    ast.BitXor: ("^", 4),
    ast.LShift: ("<<", 8),
    ast.RShift: (">>", 8),
}
_COMPARE = {
    ast.Lt: ("<", 7),
    ast.LtE: ("<=", 7),
    ast.Gt: (">", 7),
    ast.GtE: (">=", 7),
    ast.Eq: ("==", 6),
    ast.NotEq: ("!=", 6),
}
_PREC_TERNARY, _PREC_OR, _PREC_AND, _PREC_UNARY, _PREC_POSTFIX, _PREC_ATOM = 0, 1, 2, 11, 12, 13

# Python builtins Warp maps onto its own functions.
_PYTHON_BUILTINS = {"min": "wp::min", "max": "wp::max", "abs": "wp::abs"}
# Scalar casts.
_CASTS = {"float": "float", "int": "int", "bool": "bool"}
_WP_CASTS = {"float32": "float", "float64": "double", "int32": "int", "int64": "int64_t", "uint32": "uint32_t"}


def _scalar_ctype(dtype) -> str:
    if dtype in (float, wp.float32):
        return "float"
    if dtype is wp.float64:
        return "double"
    if dtype in (int, wp.int32):
        return "int"
    if dtype is wp.int64:
        return "int64_t"
    if dtype is wp.uint32:
        return "uint32_t"
    if dtype in (bool, wp.bool):
        return "bool"
    raise TypeError(f"unsupported scalar type {dtype!r} in a law")


_NAMED_TYPES = {
    wp.vec2: "wp::vec2",
    wp.vec3: "wp::vec3",
    wp.vec4: "wp::vec4",
    wp.spatial_vector: "wp::spatial_vector",
    wp.mat22: "wp::mat22",
    wp.mat33: "wp::mat33",
    wp.quat: "wp::quat",
    wp.transform: "wp::transform",
}


def _ctype(annotation) -> str:
    """C++ type of a law argument or result annotation."""
    if annotation in _NAMED_TYPES:
        return _NAMED_TYPES[annotation]
    if wp.types.is_array(annotation):
        return f"const wp::array_t<{_ctype(annotation.dtype)}>&"
    if wp.types.type_is_vector(annotation):
        return f"wp::vec_t<{annotation._length_}, {_scalar_ctype(annotation._wp_scalar_type_)}>"
    if wp.types.type_is_matrix(annotation):
        rows, cols = annotation._shape_
        return f"wp::mat_t<{rows}, {cols}, {_scalar_ctype(annotation._wp_scalar_type_)}>"
    return _scalar_ctype(annotation)


def _float_literal(value: float, ctype: str = "float") -> str:
    """The C++ literal of ``value`` rounded as Warp rounds a Python float constant of that type."""
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    if math.isnan(value):
        return "NAN"
    if ctype == "double":
        text = repr(float(value))
        return text if any(ch in text for ch in ".en") else text + ".0"
    # Warp rounds a Python float constant to float32; print the shortest decimal of that float32,
    # which a C++ float literal reads back exactly.
    text = str(np.float32(value))
    if not any(ch in text for ch in ".en"):
        text += ".0"
    return text + "f"


def _literal(value) -> str:
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return _float_literal(float(value), "double" if isinstance(value, np.float64 | wp.float64) else "float")
    raise TypeError(f"unsupported constant {value!r} in a law")


@dataclass
class _Law:
    """One registered law: its definition, Warp function and CUDA rendering choices."""

    definition: Callable
    function: Any
    family: LawFamily
    inline: str
    cuda: bool
    signature: inspect.Signature
    result_annotations: tuple  # one result type, or one per tuple member
    results: tuple[str, ...]  # their C++ types

    @property
    def name(self) -> str:
        return self.definition.__name__

    @property
    def result_struct(self) -> str:
        return f"{self.name}_result"


# Every registered law by its Warp function and by its definition, so a law's body resolves the laws it calls.
_LAWS: dict[int, _Law] = {}


def _law_of(value) -> _Law | None:
    return _LAWS.get(id(value))


@dataclass(eq=False)
class LawFamily:
    """Laws whose CUDA renderings share one struct, ``name``, in one device prelude.

    Args:
        name: The C++ struct of the family's CUDA functions; native code calls ``name::law(...)``.
        requires: Families whose laws this family's laws call; their preludes come first.
    """

    name: str
    requires: tuple[LawFamily, ...] = ()
    _laws: list[_Law] = field(default_factory=list, repr=False)
    _prelude: DevicePrelude | None = field(default=None, repr=False)

    def law(self, definition: Callable | None = None, *, inline: str = "force", cuda: bool = True):
        """Register ``definition`` as a law of this family and return its Warp function.

        Args:
            definition: The law, in the subset of Warp's kernel language this module renders.
            inline: Inlining of the CUDA function: ``"force"`` (``__forceinline__``), ``"default"``
                (the compiler's choice) or ``"never"`` (``__noinline__``, for large rarely taken laws).
            cuda: Render a CUDA function. A law whose arithmetic contract needs its Warp callers'
                module options (``fuse_fp`` off, say) has no CUDA form.
        """
        if definition is None:
            return lambda fn: self.law(fn, inline=inline, cuda=cuda)
        if inline not in _INLINE:
            raise ValueError(f"inline must be one of {tuple(_INLINE)}, got {inline!r}")
        if self._prelude is not None:
            raise RuntimeError(f"{self.name} rendered its prelude before {definition.__name__} was registered")
        signature = inspect.signature(definition)
        annotation = signature.return_annotation
        if annotation is inspect.Signature.empty:
            raise TypeError(f"law {definition.__name__} must annotate its result")
        annotations = tuple(get_args(annotation)) if get_origin(annotation) is tuple else (annotation,)
        results = tuple(_ctype(member) for member in annotations)
        for parameter in signature.parameters.values():
            if parameter.annotation is inspect.Parameter.empty:
                raise TypeError(f"law {definition.__name__} must annotate argument {parameter.name}")
        function = wp.func(definition)
        entry = _Law(definition, function, self, inline, cuda, signature, annotations, results)
        self._laws.append(entry)
        _LAWS[id(function)] = entry
        _LAWS[id(definition)] = entry
        return function

    @property
    def laws(self) -> tuple[str, ...]:
        """Names of the family's laws, in registration order."""
        return tuple(entry.name for entry in self._laws)

    def prelude(self) -> DevicePrelude:
        """The device prelude declaring the family's struct of CUDA functions (rendered once)."""
        if self._prelude is None:
            self._prelude = DevicePrelude(self.name, render_family(self), cpu=render_family(self, cpu=True))
        return self._prelude

    def preludes(self) -> tuple[DevicePrelude, ...]:
        """This family's prelude after those of the families it requires, each once."""
        return law_preludes(self)


@dataclass(frozen=True)
class LawSignature:
    """A registered law's family, name, arguments and results."""

    family: str
    """The C++ struct of the law's CUDA function."""
    name: str
    parameters: dict
    """Argument name to type."""
    results: tuple
    """Result types: one, or one per tuple member."""
    cuda: bool
    """Whether the law has a CUDA form."""


def law_signature(function) -> LawSignature:
    """Return the signature of a registered law, given its Warp function."""
    entry = _law_of(function)
    if entry is None:
        raise ValueError(f"{function!r} is not a registered law")
    parameters = {name: parameter.annotation for name, parameter in entry.signature.parameters.items()}
    return LawSignature(entry.family.name, entry.name, parameters, entry.result_annotations, entry.cuda)


def law_preludes(*families: LawFamily) -> tuple[DevicePrelude, ...]:
    """The preludes of ``families`` and the families they require, dependencies first, each once."""
    ordered: list[LawFamily] = []

    def visit(family: LawFamily, path: tuple[str, ...]):
        if family.name in path:
            raise ValueError(f"law families require each other: {' -> '.join((*path, family.name))}")
        for required in family.requires:
            visit(required, (*path, family.name))
        if all(existing is not family for existing in ordered):
            ordered.append(family)

    for family in families:
        visit(family, ())
    return tuple(family.prelude() for family in ordered)


def _required(family: LawFamily) -> list[LawFamily]:
    """The families ``family`` requires, directly or through them."""
    found: list[LawFamily] = []
    pending = list(family.requires)
    while pending:
        required = pending.pop()
        if all(existing is not required for existing in found):
            found.append(required)
            pending.extend(required.requires)
    return found


def render_family(family: LawFamily, *, cpu: bool = False) -> str:
    """Return the C++ text of ``family``'s struct: result structs, then one static function per law.

    ``cpu`` writes the CPU twin, whose functions are plain inline functions.
    """
    laws = [entry for entry in family._laws if entry.cuda]
    lines = [f"    struct {family.name} {{"]
    for entry in laws:
        if len(entry.results) > 1:
            members = " ".join(f"{ctype} v{k};" for k, ctype in enumerate(entry.results))
            lines.append(f"        struct {entry.result_struct} {{ {members} }};")
    for entry in laws:
        lines.append(render_cuda(entry.function, indent=" " * 8, cpu=cpu))
    lines.append("    };")
    return "\n".join(lines)


def render_cuda(function, *, indent: str = "", cpu: bool = False) -> str:
    """Return the CUDA definition of a registered law: a static member of its family's struct.

    ``cpu`` writes it for the CPU twin of the family's prelude, the same text as a plain inline function.
    """
    entry = _law_of(function)
    if entry is None:
        raise ValueError(f"{function!r} is not a registered law")
    if not entry.cuda:
        raise ValueError(f"law {entry.name} has no CUDA form")
    return _Renderer(entry).render(indent, cpu)


class LawSourceError(SyntaxError):
    """A law uses Python outside the subset this module renders."""


# Value kinds the renderer tracks: scalar arithmetic renders infix, every other kind through Warp's own
# functions (wp::add, wp::mul, ...), as Warp's code generator writes it.
_SCALAR, _VECTOR, _MATRIX, _QUAT, _TRANSFORM, _ARRAY = "scalar", "vector", "matrix", "quat", "transform", "array"
_FIRST = "first argument"
_SECOND = "second argument"
_BUILTIN_KINDS = {
    "length": _SCALAR,
    "length_sq": _SCALAR,
    "dot": _SCALAR,
    "sqrt": _SCALAR,
    "isfinite": _SCALAR,
    "isnan": _SCALAR,
    "isinf": _SCALAR,
    "determinant": _SCALAR,
    "trace": _SCALAR,
    "abs": _FIRST,
    "max": _FIRST,
    "min": _FIRST,
    "clamp": _FIRST,
    "cw_div": _FIRST,
    "cw_mul": _FIRST,
    "normalize": _FIRST,
    "cross": _VECTOR,
    "transform_point": _VECTOR,
    "transform_vector": _VECTOR,
    "transform_get_translation": _VECTOR,
    "quat_rotate": _VECTOR,
    "quat_rotate_inv": _VECTOR,
    "spatial_top": _VECTOR,
    "spatial_bottom": _VECTOR,
    "vec2": _VECTOR,
    "vec3": _VECTOR,
    "vec4": _VECTOR,
    "spatial_vector": _VECTOR,
    "outer": _MATRIX,
    "mat22": _MATRIX,
    "mat33": _MATRIX,
    "mat44": _MATRIX,
    "transpose": _MATRIX,
    "skew": _MATRIX,
    "quat_to_matrix": _MATRIX,
    "inverse": _FIRST,
    "where": _SECOND,
    "transform_get_rotation": _QUAT,
    "quat": _QUAT,
    "quat_from_axis_angle": _QUAT,
    "transform": _TRANSFORM,
    "transform_inverse": _TRANSFORM,
    "transform_multiply": _TRANSFORM,
}
_OPERATOR_FUNCTIONS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Mod: "mod"}
# Warp builtins with several results, of float arguments: their C++ form takes the results as output arguments.
_TUPLE_BUILTINS = {"eig3": (("wp::mat33", _MATRIX), ("wp::vec3", _VECTOR))}


def _kind(annotation) -> str:
    """Kind of a value of a law argument or result type."""
    if wp.types.is_array(annotation):
        return _ARRAY
    if annotation is wp.quat or getattr(annotation, "_wp_generic_type_str_", "") == "quat_t":
        return _QUAT
    if annotation is wp.transform or getattr(annotation, "_wp_generic_type_str_", "") == "transform_t":
        return _TRANSFORM
    if wp.types.type_is_vector(annotation):
        return _VECTOR
    if wp.types.type_is_matrix(annotation):
        return _MATRIX
    return _SCALAR


def _product_kind(left: str, right: str) -> str:
    if left == _SCALAR:
        return right
    if right == _SCALAR:
        return left
    if _MATRIX in (left, right) and _VECTOR in (left, right):
        return _VECTOR
    return left


class _Renderer:
    """Write one law's CUDA function from its Python source."""

    def __init__(self, entry: _Law):
        self.entry = entry
        self.definition = entry.definition
        self.globals = entry.definition.__globals__
        source = textwrap.dedent(inspect.getsource(entry.definition))
        tree = ast.parse(source)
        self.node = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.first_line = entry.definition.__code__.co_firstlineno - 1
        self.params = {name: parameter.annotation for name, parameter in entry.signature.parameters.items()}
        self.scopes: list[dict[str, str]] = []  # name -> kind, per C++ block
        self.lines: list[str] = []
        self.counter = 0

    # --- errors ---------------------------------------------------------------------------------------------
    def fail(self, node, message: str):
        line = self.first_line + (getattr(node, "lineno", 1) if node is not None else 1)
        raise LawSourceError(f"{self.entry.name} ({self.definition.__code__.co_filename}:{line}): {message}")

    # --- names and scopes -----------------------------------------------------------------------------------
    @staticmethod
    def cname(name: str) -> str:
        return name + "_" if name in _CPP_KEYWORDS else name

    def kind_of(self, name: str) -> str | None:
        if name in self.params:
            return _kind(self.params[name])
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def declared(self, name: str) -> bool:
        return self.kind_of(name) is not None

    def declare(self, name: str, kind: str):
        self.scopes[-1][name] = kind

    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"_{stem}{self.counter}"

    # --- rendering ------------------------------------------------------------------------------------------
    def render(self, indent: str, cpu: bool = False) -> str:
        entry = self.entry
        arguments = ", ".join(f"{_ctype(annotation)} {self.cname(name)}" for name, annotation in self.params.items())
        result = entry.result_struct if len(entry.results) > 1 else entry.results[0]
        summary = (inspect.getdoc(self.definition) or "").split("\n", 1)[0]
        self.lines = []
        if summary:
            self.lines.append(f"{indent}// {summary}")
        qualifiers = _CPU_INLINE if cpu else _INLINE[entry.inline]
        self.lines.append(f"{indent}static {qualifiers} {result} {entry.name}({arguments}) {{")
        body = self.node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        self.block(body, indent + "    ")
        self.lines.append(f"{indent}}}")
        return "\n".join(self.lines)

    def emit(self, indent: str, text: str):
        self.lines.append(indent + text)

    def block(self, statements, indent: str, names: dict[str, str] | None = None):
        self.scopes.append(dict(names or {}))
        for statement in statements:
            self.statement(statement, indent)
        self.scopes.pop()

    def statement(self, node, indent: str):
        if isinstance(node, ast.Return):
            self.render_return(node, indent)
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                self.fail(node, "chained assignment")
            self.render_assign(node.targets[0], node.value, indent, node)
        elif isinstance(node, ast.AnnAssign):
            if node.value is None:
                self.fail(node, "declaration without a value")
            self.render_assign(node.target, node.value, indent, node)
        elif isinstance(node, ast.AugAssign):
            if type(node.op) not in _BINARY:
                self.fail(node, f"augmented operator {type(node.op).__name__}")
            target, target_kind = self.target(node.target)
            value = self.expression(node.value)
            if (target_kind, value[2]) == (_SCALAR, _SCALAR):
                self.emit(indent, f"{target} {_BINARY[type(node.op)][0]}= {value[0]};")
            else:
                text, _, _ = self.binary(node.op, (target, _PREC_POSTFIX, target_kind), value, node)
                self.emit(indent, f"{target} = {text};")
        elif isinstance(node, ast.If):
            self.render_if(node, indent)
        elif isinstance(node, ast.For):
            self.render_for(node, indent)
        elif isinstance(node, ast.While):
            if node.orelse:
                self.fail(node, "while-else")
            self.emit(indent, f"while ({self.expr(node.test)}) {{")
            self.block(node.body, indent + "    ")
            self.emit(indent, "}")
        elif isinstance(node, ast.Break):
            self.emit(indent, "break;")
        elif isinstance(node, ast.Continue):
            self.emit(indent, "continue;")
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass  # a string statement is a comment
        else:
            self.fail(node, f"statement {type(node).__name__}")

    def render_return(self, node: ast.Return, indent: str):
        results = self.entry.results
        if node.value is None:
            self.fail(node, "a law returns a value")
        if isinstance(node.value, ast.Tuple):
            if len(node.value.elts) != len(results):
                self.fail(node, f"returns {len(node.value.elts)} values, annotated {len(results)}")
            values = ", ".join(self.expr(element) for element in node.value.elts)
            self.emit(indent, f"return {self.entry.result_struct}{{{values}}};")
        else:
            if len(results) != 1:
                self.fail(node, "a tuple-valued law returns a tuple")
            self.emit(indent, f"return {self.expr(node.value)};")

    def render_assign(self, target, value, indent: str, node):
        if isinstance(target, ast.Tuple) and self.tuple_builtin(value) is not None:
            self.render_tuple_builtin(target, value, indent, node)
            return
        if isinstance(target, ast.Tuple):
            callee = _law_of(self.resolve_call(value)) if isinstance(value, ast.Call) else None
            if callee is None or len(callee.results) != len(target.elts):
                self.fail(node, "a tuple assignment takes the result of a tuple-valued law")
            names = []
            for element in target.elts:
                if not isinstance(element, ast.Name):
                    self.fail(node, "a tuple assignment binds names")
                names.append(element.id)
            kinds = [_kind(annotation) for annotation in callee.result_annotations]
            call = self.expr(value)
            if not any(self.declared(name) for name in names):
                bound = ", ".join(self.cname(name) for name in names)
                self.emit(indent, f"auto [{bound}] = {call};")
                for name, kind in zip(names, kinds, strict=True):
                    self.declare(name, kind)
                return
            temporary = self.fresh("result")
            self.emit(indent, f"const auto {temporary} = {call};")
            for k, (name, kind) in enumerate(zip(names, kinds, strict=True)):
                if self.declared(name):
                    self.emit(indent, f"{self.cname(name)} = {temporary}.v{k};")
                else:
                    self.emit(indent, f"auto {self.cname(name)} = {temporary}.v{k};")
                    self.declare(name, kind)
            return
        if isinstance(value, ast.Call) and (callee := _law_of(self.resolve_call(value))) and len(callee.results) > 1:
            self.fail(node, "a tuple-valued law is called in a tuple assignment")
        text, _, kind = self.expression(value)
        if isinstance(target, ast.Name) and not self.declared(target.id):
            self.emit(indent, f"auto {self.cname(target.id)} = {text};")
            self.declare(target.id, kind)
            return
        self.emit(indent, f"{self.target(target)[0]} = {text};")

    @staticmethod
    def tuple_builtin(value) -> str | None:
        """The name of the tuple-valued Warp builtin ``value`` calls, if it calls one."""
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)):
            return None
        func = value.func
        named = isinstance(func.value, ast.Name) and func.value.id == "wp" and func.attr in _TUPLE_BUILTINS
        return func.attr if named else None

    def render_tuple_builtin(self, target: ast.Tuple, value: ast.Call, indent: str, node):
        """Declare the results of a tuple-valued builtin and pass them as its output arguments."""
        name = self.tuple_builtin(value)
        results = _TUPLE_BUILTINS[name]
        if len(target.elts) != len(results) or not all(isinstance(element, ast.Name) for element in target.elts):
            self.fail(node, f"wp.{name} binds {len(results)} names")
        if value.keywords:
            self.fail(node, "keyword arguments")
        outputs = []
        for element, (ctype, kind) in zip(target.elts, results, strict=True):
            if not self.declared(element.id):
                self.emit(indent, f"{ctype} {self.cname(element.id)};")
                self.declare(element.id, kind)
            outputs.append(self.cname(element.id))
        arguments = [self.expr(argument) for argument in value.args] + outputs
        self.emit(indent, f"wp::{name}({', '.join(arguments)});")

    def target(self, node) -> tuple[str, str]:
        """The C++ variable and kind an assignment updates: laws assign whole local values."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and self.kind_of(node.value.id) == _ARRAY:
                self.fail(node, "laws do not write arrays; return the value instead")
            self.fail(node, "laws assign whole values; build the vector instead of writing one component")
        if not isinstance(node, ast.Name):
            self.fail(node, f"assignment target {type(node).__name__}")
        if not self.declared(node.id):
            self.fail(node, f"{node.id} is updated before it is assigned")
        return self.cname(node.id), self.kind_of(node.id)

    def render_if(self, node: ast.If, indent: str):
        self.emit(indent, f"if ({self.expr(node.test)}) {{")
        self.block(node.body, indent + "    ")
        rest = node.orelse
        while len(rest) == 1 and isinstance(rest[0], ast.If):  # elif chain
            self.emit(indent, f"}} else if ({self.expr(rest[0].test)}) {{")
            self.block(rest[0].body, indent + "    ")
            rest = rest[0].orelse
        if rest:
            self.emit(indent, "} else {")
            self.block(rest, indent + "    ")
        self.emit(indent, "}")

    def render_for(self, node: ast.For, indent: str):
        if node.orelse:
            self.fail(node, "for-else")
        iterator = node.iter
        if not (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
            and 1 <= len(iterator.args) <= 3
            and not iterator.keywords
        ):
            self.fail(node, "a loop iterates over range(...)")
        if not isinstance(node.target, ast.Name):
            self.fail(node, "a loop binds one name")
        args = [self.expr(argument) for argument in iterator.args]
        start, stop, step = ("0", args[0], "1") if len(args) == 1 else (args[0], args[1], "1")
        step_value = 1
        if len(args) == 3:
            step = args[2]
            try:
                step_value = int(self.constant_value(iterator.args[2]))
            except Exception:
                self.fail(node, "a loop step is a constant")
            if step_value == 0:
                self.fail(node, "a loop step is nonzero")
        name = node.target.id
        variable = self.fresh("i") if name == "_" else self.cname(name)
        compare = "<" if step_value > 0 else ">"
        increment = "++" + variable if step == "1" else f"{variable} += {step}"
        self.emit(indent, f"for (int {variable} = {start}; {variable} {compare} {stop}; {increment}) {{")
        self.block(node.body, indent + "    ", {name: _SCALAR})
        self.emit(indent, "}")

    # --- expressions ----------------------------------------------------------------------------------------
    def constant_value(self, node):
        """Evaluate a constant expression (a literal, a module constant or ``wp.static``)."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "static"
            and len(node.args) == 1
        ):
            node = node.args[0]
        code = compile(ast.Expression(node), "<law constant>", "eval")
        return eval(code, dict(self.globals))

    def resolve_call(self, node: ast.Call):
        """Return the Python object a call names, or None for builtins this module spells itself."""
        func = node.func
        if isinstance(func, ast.Name):
            if self.declared(func.id):
                return None
            return self.globals.get(func.id)
        if isinstance(func, ast.Attribute):
            try:
                return self.constant_value(func)
            except Exception:
                return None
        return None

    def expr(self, node, context: int = _PREC_TERNARY) -> str:
        text, precedence, _ = self.expression(node)
        return f"({text})" if precedence < context else text

    @staticmethod
    def wrap(operand: tuple[str, int, str], context: int) -> str:
        text, precedence, _ = operand
        return f"({text})" if precedence < context else text

    def binary(self, op, left, right, node) -> tuple[str, int, str]:
        """Render ``left op right``: infix for scalars, Warp's operator function for every other kind."""
        kinds = (left[2], right[2])
        if kinds == (_SCALAR, _SCALAR):
            symbol, precedence = _BINARY[type(op)]
            return f"{self.wrap(left, precedence)} {symbol} {self.wrap(right, precedence + 1)}", precedence, _SCALAR
        function = _OPERATOR_FUNCTIONS.get(type(op))
        if function is None:
            self.fail(node, f"operator {type(op).__name__} on a {kinds[0]} and a {kinds[1]}")
        if function == "mul":
            kind = _product_kind(*kinds)
        elif function == "div" and kinds[0] == _SCALAR:
            kind = kinds[1]
        else:
            kind = kinds[0]
        return f"wp::{function}({left[0]}, {right[0]})", _PREC_POSTFIX, kind

    def expression(self, node) -> tuple[str, int, str]:
        """Return ``(text, precedence, kind)`` of an expression."""
        if isinstance(node, ast.Constant):
            return _literal(node.value), _PREC_ATOM, _SCALAR
        if isinstance(node, ast.Name):
            kind = self.kind_of(node.id)
            if kind is not None:
                return self.cname(node.id), _PREC_ATOM, kind
            if node.id not in self.globals:
                self.fail(node, f"{node.id} is read before it is assigned in this block")
            value = self.globals[node.id]
            if _law_of(value) is not None:
                self.fail(node, f"law {node.id} is called, not passed")
            return _literal(value), _PREC_ATOM, _SCALAR
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int | float):
                    return _literal(-node.operand.value), _PREC_UNARY, _SCALAR
                operand = self.expression(node.operand)
                if operand[2] != _SCALAR:
                    return f"wp::neg({operand[0]})", _PREC_POSTFIX, operand[2]
                # Negation, as Python and a hand-written sweep negate; Warp's own lowering subtracts from zero,
                # which differs in the sign of a zero only.
                text = self.wrap(operand, _PREC_UNARY)
                return f"-({text})" if text.startswith("-") else f"-{text}", _PREC_UNARY, _SCALAR
            if isinstance(node.op, ast.UAdd):
                return self.expression(node.operand)
            if isinstance(node.op, ast.Not):
                return "!" + self.expr(node.operand, _PREC_UNARY), _PREC_UNARY, _SCALAR
            if isinstance(node.op, ast.Invert):
                return "~" + self.expr(node.operand, _PREC_UNARY), _PREC_UNARY, _SCALAR
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                self.fail(node, f"operator {type(node.op).__name__}")
            return self.binary(node.op, self.expression(node.left), self.expression(node.right), node)
        if isinstance(node, ast.BoolOp):
            symbol, precedence = ("&&", _PREC_AND) if isinstance(node.op, ast.And) else ("||", _PREC_OR)
            # Parenthesize nested boolean operators of either kind, for the reader.
            parts = [self.expr(value, _PREC_AND + 1) for value in node.values]
            return f" {symbol} ".join(parts), precedence, _SCALAR
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.fail(node, "chained comparison")
            op = type(node.ops[0])
            if op not in _COMPARE:
                self.fail(node, f"comparison {op.__name__}")
            symbol, precedence = _COMPARE[op]
            left = self.expr(node.left, precedence + 1)
            right = self.expr(node.comparators[0], precedence + 1)
            return f"{left} {symbol} {right}", precedence, _SCALAR
        if isinstance(node, ast.IfExp):
            test = self.expr(node.test, _PREC_OR)
            body = self.expression(node.body)
            orelse = self.expr(node.orelse, _PREC_TERNARY)
            return f"{test} ? {self.wrap(body, _PREC_OR)} : {orelse}", _PREC_TERNARY, body[2]
        if isinstance(node, ast.Call):
            text, kind = self.call(node)
            return text, _PREC_POSTFIX, kind
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.Attribute):
            try:
                return _literal(self.constant_value(node)), _PREC_ATOM, _SCALAR
            except Exception:
                self.fail(node, f"attribute {ast.unparse(node)}")
        self.fail(node, f"expression {type(node).__name__}")
        return "", _PREC_ATOM, _SCALAR

    def subscript(self, node: ast.Subscript) -> tuple[str, int, str]:
        base = node.value
        index = node.slice
        indices = index.elts if isinstance(index, ast.Tuple) else [index]
        rendered = [self.expr(element) for element in indices]
        if isinstance(base, ast.Name) and self.kind_of(base.id) == _ARRAY:
            # A value load, as Warp reads an array element: a reference into memory compiles differently.
            dtype = self.params[base.id].dtype
            address = f"wp::address({self.cname(base.id)}, {', '.join(rendered)})"
            return f"wp::load({address})", _PREC_POSTFIX, _kind(dtype)
        for element in indices:
            try:
                negative = self.constant_value(element) < 0
            except Exception:
                negative = False
            if negative:
                self.fail(node, "a negative index; count from the start")
        operand = self.expression(base)
        target = self.wrap(operand, _PREC_POSTFIX)
        if operand[2] == _MATRIX and len(rendered) == 2:
            return f"{target}.data[{rendered[0]}][{rendered[1]}]", _PREC_POSTFIX, _SCALAR
        if operand[2] == _MATRIX and len(rendered) == 1:
            return f"{target}.get_row({rendered[0]})", _PREC_POSTFIX, _VECTOR
        if operand[2] != _MATRIX and len(rendered) == 1:
            return f"{target}[{rendered[0]}]", _PREC_POSTFIX, _SCALAR
        self.fail(node, "an index of a matrix element or row, or of a vector component")
        return "", _PREC_ATOM, _SCALAR

    def cast(self, ctype: str, node: ast.Call, args: list[str]) -> str:
        """Cast one value; a literal argument renders as a literal (Warp's sources define float(x) as a macro)."""
        if len(args) != 1:
            self.fail(node, "a cast takes one value")
        argument = node.args[0]
        try:
            value = self.constant_value(argument)
        except Exception:
            value = None
        if isinstance(value, bool | int | float) and not isinstance(argument, ast.Name | ast.Attribute):
            if ctype == "float":
                return _float_literal(float(value))
            if ctype == "double":
                return _float_literal(float(value), "double")
            if ctype == "bool":
                return "true" if value else "false"
            if ctype == "int":
                return str(int(value))
        return f"static_cast<{ctype}>({args[0]})"

    def call(self, node: ast.Call) -> tuple[str, str]:
        if node.keywords:
            self.fail(node, "keyword arguments")
        func = node.func
        operands = [self.expression(argument) for argument in node.args]
        args = [operand[0] for operand in operands]
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "wp":
            name = func.attr
            if name == "static":
                return _literal(self.constant_value(node.args[0])), _SCALAR
            if name in _WP_CASTS:
                return self.cast(_WP_CASTS[name], node, args), _SCALAR
            kind = _BUILTIN_KINDS.get(name)
            if kind is None:
                self.fail(node, f"wp.{name} is not a builtin this module renders; add its kind to _BUILTIN_KINDS")
            if kind == _FIRST:
                kind = operands[0][2]
            elif kind == _SECOND:
                kind = operands[1][2]
            return f"wp::{name}({', '.join(args)})", kind
        if isinstance(func, ast.Name):
            if func.id in _CASTS and not self.declared(func.id):
                return self.cast(_CASTS[func.id], node, args), _SCALAR
            if func.id in _PYTHON_BUILTINS and func.id not in self.globals:
                return f"{_PYTHON_BUILTINS[func.id]}({', '.join(args)})", operands[0][2]
            value = self.globals.get(func.id)
            callee = _law_of(value)
            if callee is not None:
                if not callee.cuda:
                    self.fail(node, f"law {callee.name} has no CUDA form")
                if callee.family is not self.entry.family and all(
                    family is not callee.family for family in _required(self.entry.family)
                ):
                    self.fail(node, f"{self.entry.family.name} calls {callee.family.name} without requiring it")
                kind = _kind(callee.result_annotations[0]) if len(callee.results) == 1 else "result"
                return f"{callee.family.name}::{callee.name}({', '.join(args)})", kind
            if isinstance(value, type) and (wp.types.type_is_vector(value) or wp.types.type_is_matrix(value)):
                return self.construct(value, args), _kind(value)
        self.fail(node, f"call of {ast.unparse(func)}: only Warp builtins, Warp types and registered laws")
        return "", _SCALAR

    @staticmethod
    def construct(value_type, args: list[str]) -> str:
        """Construct a vector or matrix of a module-level Warp type; more than four values take an initializer."""
        ctype = _ctype(value_type)
        if len(args) <= 4:
            return f"{ctype}({', '.join(args)})"
        length = value_type._length_
        scalar = _scalar_ctype(value_type._wp_scalar_type_)
        return f"{ctype}(wp::initializer_array<{length}, {scalar}>{{{', '.join(args)}}})"
