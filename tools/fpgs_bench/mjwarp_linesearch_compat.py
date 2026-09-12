# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental, opt-in benchmark workaround for reviewed MJWarp 3.12 line search.

Call ``install()`` before the first solver JIT in an isolated benchmark process.
This rewrites only in-memory source and never changes the installed package.
Unknown sources fail closed. The experimental elliptic extension changes only
friction-loss cost differences beside the reviewed bracket logic; it retains
the inherited cone evaluator. Numerical regressions are not a claim of general
trajectory equivalence.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import inspect
import linecache
import sys
from pathlib import Path
from types import FunctionType, ModuleType

__all__ = ["install", "supports_model"]

_SOURCE_SHA = "509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3"
_PARAMETERS = (
    "ls_iterations",
    "cone_type",
    "fuse_jv",
    "is_sparse",
    "incremental",
    "warn_overflow",
)
_DEPENDENCIES = {
    "solver.py": _SOURCE_SHA,
    "types.py": "a49625a8994db0d3b52dd34f98adac7f70882d473108c4e4793e4830e81ea71c",
    "math.py": "4c58d5e3864f7b841633d403d78e7702bb547359ee846d43810e3b31c63966c2",
    "warp_util.py": "6759883baef16fc9a5222bf0e5c858700d0847392138a8c0d2ae78a2c9a5780a",
}
_INSTALLED = None
_HELPERS = '''@wp.func_native(snippet="""
#if defined(__CUDA_ARCH__)
return nextafterf(a, b);
#else
return __builtin_nextafterf(a, b);
#endif
""")
def _next_float(a: float, b: float) -> float:
  ...


@wp.func
def _point_finite(p: wp.vec3) -> bool:
  return wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])


@wp.func
def _sign_bracket(lo: wp.vec3, hi: wp.vec3, la: float, ha: float) -> bool:
  return (_point_finite(lo) and _point_finite(hi) and wp.isfinite(la) and wp.isfinite(ha)
          and la < ha and lo[1] <= 0.0 and hi[1] >= 0.0)


@wp.func
def _adjacent(lo: wp.vec3, hi: wp.vec3, la: float, ha: float) -> bool:
  return _sign_bracket(lo, hi, la, ha) and _next_float(la, ha) == ha


@wp.func
def _unbracketed_next(p: wp.vec3, alpha: float, proposed: float) -> float:
  result = proposed
  if (_point_finite(p) and wp.isfinite(alpha) and wp.isfinite(proposed)
      and p[2] > 0.0 and p[1] != 0.0 and proposed == alpha):
    direction = wp.where(p[1] < 0.0, 3.4028234663852886e38, -3.4028234663852886e38)
    neighbor = _next_float(alpha, direction)
    if wp.isfinite(neighbor) and neighbor != alpha:
      result = neighbor
  return result


@wp.func
def _acquire_high(lo: wp.vec3, hi: wp.vec3, la: float, ha: float,
                  trial: wp.vec3, ta: float) -> bool:
  return (_point_finite(lo) and _point_finite(hi) and _point_finite(trial)
          and wp.isfinite(la) and wp.isfinite(ha) and wp.isfinite(ta)
          and la <= ha and lo[1] < 0.0 and hi[1] < 0.0
          and trial[1] >= 0.0 and la < ta)


@wp.func
def _acquire_low(lo: wp.vec3, hi: wp.vec3, la: float, ha: float,
                 trial: wp.vec3, ta: float) -> bool:
  return (_point_finite(lo) and _point_finite(hi) and _point_finite(trial)
          and wp.isfinite(la) and wp.isfinite(ha) and wp.isfinite(ta)
          and la <= ha and lo[1] > 0.0 and hi[1] > 0.0
          and trial[1] <= 0.0 and ta < ha)


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] <= 0.0) or (x[1] > y[1] and y[1] >= 0.0)


'''

# These aliases are installed only in the ellipse-private module. The original
# package and the separately compiled pyramidal module keep their old helpers.
_ELLIPTIC_HELPERS = """_original_ellipse_point = _compute_efc_eval_pt_elliptic
_original_ellipse_three = _compute_efc_eval_pt_3alphas_elliptic


@wp.func
def _huber_delta(x0: float, jv: float, alpha: float, f: float, d: float, rf: float) -> float:
  delta = alpha * jv
  x1 = x0 + delta
  if (-rf < x0) and (x0 < rf):
    if (-rf < x1) and (x1 < rf):
      return d * delta * (x0 + 0.5 * delta)
    sign1 = wp.where(x1 <= -rf, -1.0, 1.0)
    gap0 = sign1 * x0 - rf
    return sign1 * f * delta - 0.5 * d * gap0 * gap0
  sign0 = wp.where(x0 <= -rf, -1.0, 1.0)
  if (-rf < x1) and (x1 < rf):
    gap1 = sign0 * x1 - rf
    return sign0 * f * delta + 0.5 * d * gap1 * gap1
  sign1 = wp.where(x1 <= -rf, -1.0, 1.0)
  if sign0 == sign1:
    return sign0 * f * delta
  return sign1 * f * (2.0 * x0 + delta)


@wp.func
def _safe_huber_cost(x0: float, jv: float, alpha: float, f: float, d: float, original: float) -> float:
  delta = alpha * jv
  x1 = x0 + delta
  rf = math.safe_div(f, d)
  if not (wp.isfinite(x0) and wp.isfinite(jv) and wp.isfinite(alpha)
          and wp.isfinite(delta) and wp.isfinite(x1) and wp.isfinite(rf)
          and wp.isfinite(f) and f >= 0.0 and wp.isfinite(d) and d > 0.0):
    return original
  value = _huber_delta(x0, jv, alpha, f, d, rf)
  if not wp.isfinite(value):
    return original
  return value


@wp.func
def _compute_efc_eval_pt_elliptic(
  efcid: int, alpha: float, ne: int, nf: int, impratio_invsqrt: float,
  efc_type: int, d: wp.array[float], frictionloss: wp.array[float],
  x0: float, jv: float, quad: wp.vec3, contact_friction: types.vec5,
  address0: int, quad1: wp.vec3, quad2: wp.vec3,
) -> wp.vec3:
  original = _original_ellipse_point(
    efcid, alpha, ne, nf, impratio_invsqrt, efc_type, d, frictionloss,
    x0, jv, quad, contact_friction, address0, quad1, quad2,
  )
  if efcid >= ne and efcid < ne + nf:
    cost = _safe_huber_cost(x0, jv, alpha, frictionloss[efcid], d[efcid], original[0])
    return wp.vec3(cost, original[1], original[2])
  return original


@wp.func
def _compute_efc_eval_pt_3alphas_elliptic(
  efcid: int, lo: float, hi: float, mid: float, ne: int, nf: int,
  impratio_invsqrt: float, efc_type: int, d: wp.array[float],
  frictionloss: wp.array[float], x0: float, jv: float, quad: wp.vec3,
  contact_friction: types.vec5, address0: int, quad1: wp.vec3, quad2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  a, b, c = _original_ellipse_three(
    efcid, lo, hi, mid, ne, nf, impratio_invsqrt, efc_type, d, frictionloss,
    x0, jv, quad, contact_friction, address0, quad1, quad2,
  )
  if efcid >= ne and efcid < ne + nf:
    f, diagonal = frictionloss[efcid], d[efcid]
    a = wp.vec3(_safe_huber_cost(x0, jv, lo, f, diagonal, a[0]), a[1], a[2])
    b = wp.vec3(_safe_huber_cost(x0, jv, hi, f, diagonal, b[0]), b[1], b[2])
    c = wp.vec3(_safe_huber_cost(x0, jv, mid, f, diagonal, c[0]), c[1], c[2])
  return a, b, c


"""


def _transform(data: bytes) -> str:
    """Reconstruct the reviewed installed factory with exact counted changes."""
    if hashlib.sha256(data).hexdigest() != _SOURCE_SHA:
        raise RuntimeError("Line-search workaround supports only the reviewed stock MJWarp 3.12.0 source")
    original = data.decode()
    nodes = [
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef) and node.name == "_linesearch_iterative_kernel"
    ]
    if len(nodes) != 1:
        raise RuntimeError("Unexpected line-search factory definition")
    node = nodes[0]
    start = min(node.lineno, *(d.lineno for d in node.decorator_list))
    source = "\n".join(original.splitlines()[start - 1 : node.end_lineno]) + "\n"

    def once(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Line-search source expression changed: {old!r}")
        source = source.replace(old, new)

    # The unique name separates upstream cache keys. Its ordinary global cache
    # still participates in Newton's existing deterministic-mode invalidation.
    once("def _linesearch_iterative_kernel(", "def _newton_linesearch_iterative_kernel(")
    once("lo[1] < 0.0 and lo[1] > -gtol", "lo[1] <= 0.0 and lo[1] > -gtol")
    once("hi[1] > 0.0 and hi[1] < gtol", "hi[1] >= 0.0 and hi[1] < gtol")
    once(
        "        lo_better = lo[0] < hi[0]",
        "        lo_better = lo[0] < hi[0] or (lo[0] == hi[0] and wp.abs(lo[1]) < wp.abs(hi[1]))",
    )
    once(
        "    ls_converged = initial_converged",
        "    ls_converged = initial_converged\n    representable_done = bool(False)",
    )
    once(
        "      for _ in range(LS_ITERATIONS):",
        """      for _ in range(LS_ITERATIONS):
        if _adjacent(lo, hi, lo_alpha, hi_alpha) and (lo[0] < 0.0 or hi[0] < 0.0):
          lo_better_early = lo[0] < hi[0] or (lo[0] == hi[0] and wp.abs(lo[1]) < wp.abs(hi[1]))
          alpha = wp.where(lo_better_early, lo_alpha, hi_alpha)
          improvement = -wp.where(lo_better_early, lo[0], hi[0])
          representable_done = True
          ls_converged = True
          break
        valid_bracket = _sign_bracket(lo, hi, lo_alpha, hi_alpha)""",
    )
    once(
        "        mid_alpha = 0.5 * (lo_alpha + hi_alpha)",
        """        mid_alpha = 0.5 * (lo_alpha + hi_alpha)
        if valid_bracket:
          lower_interior = _next_float(lo_alpha, hi_alpha)
          upper_interior = _next_float(hi_alpha, lo_alpha)
          lo_next_alpha = wp.clamp(lo_next_alpha, lower_interior, upper_interior)
          hi_next_alpha = wp.clamp(hi_next_alpha, lower_interior, upper_interior)
        else:
          lo_next_alpha = _unbracketed_next(lo, lo_alpha, lo_next_alpha)
          hi_next_alpha = _unbracketed_next(hi, hi_alpha, hi_next_alpha)""",
    )
    # All three trials are already evaluated. Acquiring a missing sign must
    # precede same-side swaps, which otherwise reject every crossing trial.
    acquisition = "        acquired_lo = bool(False)\n        acquired_hi = bool(False)\n"
    for point in ("lo_next", "hi_next", "mid"):
        acquisition += f"""        if not valid_bracket:
          take_hi = _acquire_high(lo, hi, lo_alpha, hi_alpha, {point}, {point}_alpha)
          take_lo = _acquire_low(lo, hi, lo_alpha, hi_alpha, {point}, {point}_alpha)
          lo = wp.where(take_lo, {point}, lo)
          lo_alpha = wp.where(take_lo, {point}_alpha, lo_alpha)
          hi = wp.where(take_hi, {point}, hi)
          hi_alpha = wp.where(take_hi, {point}_alpha, hi_alpha)
          acquired_lo = acquired_lo or take_lo
          acquired_hi = acquired_hi or take_hi
          valid_bracket = _sign_bracket(lo, hi, lo_alpha, hi_alpha)
"""
    once("        # bracket swapping", acquisition + "\n        # bracket swapping")
    once(
        "        swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next",
        "        swap_lo = acquired_lo or swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next",
    )
    once(
        "        swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next",
        "        swap_hi = acquired_hi or swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next",
    )
    for side, names in (
        ("lo", ("lo_next", "mid", "hi_next")),
        ("hi", ("hi_next", "mid", "lo_next")),
    ):
        for point in names:
            name = f"swap_{side}_{point}"
            old = f"        {name} = _in_bracket({side}, {point})"
            left, sign = ("lo_alpha < ", "<=") if side == "lo" else ("lo_alpha <= ", ">=")
            expression = f"(_point_finite({point}) and {left}{point}_alpha and {point}_alpha < hi_alpha and {point}[1] {sign} 0.0)"
            once(
                old,
                old + f"\n        if valid_bracket:\n          {name} = {expression}",
            )
    once(
        "        # check for convergence\n",
        "        # check for convergence\n        representable_done = _adjacent(lo, hi, lo_alpha, hi_alpha) and (lo[0] < 0.0 or hi[0] < 0.0)\n",
    )
    once(
        "          (not swap_lo and not swap_hi)",
        "          representable_done or (not swap_lo and not swap_hi)",
    )
    return _HELPERS + source


def _ellipse_transform(data: bytes) -> str:
    """Add cost-only ellipse aliases with a distinct immutable factory cache key."""
    source = _transform(data)
    old = "def _newton_linesearch_iterative_kernel("
    if source.count(old) != 1:
        raise RuntimeError("Unexpected private line-search factory definition")
    return _ELLIPTIC_HELPERS + source.replace(old, "def _newton_elliptic_linesearch_iterative_kernel(")


def supports_model(model) -> bool:
    """Return whether the actual model belongs to the tested workaround scope."""
    from mujoco_warp._src import types

    option = getattr(model, "opt", None)
    return (
        option is not None
        and getattr(option, "cone", None) in (types.ConeType.PYRAMIDAL, types.ConeType.ELLIPTIC)
        and getattr(option, "solver", None) == types.SolverType.NEWTON
    )


def install() -> dict:
    """Install the explicit process-local workaround before line-search JIT.

    Returns:
        JSON-safe source/behavior metadata, not a numerical-quality verdict.
    """
    global _INSTALLED  # noqa: PLW0603 - one explicit process-local installation
    from mujoco_warp._src import solver, types, warp_util

    if _INSTALLED is not None:
        if solver._linesearch_iterative is not _INSTALLED[0]:
            raise RuntimeError("Another owner replaced the installed workaround")
        return dict(_INSTALLED[1])
    if importlib.metadata.version("mujoco-warp") != "3.12.0":
        raise RuntimeError("Line-search workaround requires reviewed MJWarp 3.12.0")
    factory = solver._linesearch_iterative_kernel
    if tuple(inspect.signature(factory).parameters) != _PARAMETERS:
        raise RuntimeError("Unexpected MJWarp line-search factory signature")
    implementation = inspect.unwrap(factory)
    original = solver._linesearch_iterative
    for function in (implementation, original):
        if function.__module__ != solver.__name__ or function.__code__.co_filename != solver.__file__:
            raise RuntimeError("MJWarp line-search source already has an unknown override")
    if tuple(inspect.signature(original).parameters) != ("m", "d", "ctx", "fuse_jv"):
        raise RuntimeError("Unexpected MJWarp line-search driver signature")
    if any(key[-1] == hash("_linesearch_iterative_kernel") for key in warp_util._KERNEL_CACHE):
        raise RuntimeError("Install the line-search workaround BEFORE the first line-search JIT")
    source_path = Path(solver.__file__)
    for filename, expected in _DEPENDENCIES.items():
        if hashlib.sha256(source_path.with_name(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unreviewed MJWarp line-search dependency: {filename}")
    data = source_path.read_bytes()
    source = _transform(data)
    ellipse_source = _ellipse_transform(data)
    digest = hashlib.sha256(source.encode()).hexdigest()
    ellipse_digest = hashlib.sha256(ellipse_source.encode()).hexdigest()
    created = []

    def build_private(text: str, sha256: str, factory_name: str):
        name = "_newton_benchmark_mjwarp_linesearch_" + sha256[:16]
        filename = f"<{name}>"
        module = ModuleType(name)
        module.__dict__.update({key: value for key, value in solver.__dict__.items() if not key.startswith("__")})
        module.__file__ = filename
        sys.modules[name] = module
        linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
        created.append((name, filename))
        exec(compile(text, filename, "exec"), module.__dict__)
        driver_globals = dict(original.__globals__)
        driver_globals["_linesearch_iterative_kernel"] = getattr(module, factory_name)
        driver = FunctionType(
            original.__code__, driver_globals, original.__name__, original.__defaults__, original.__closure__
        )
        return module, driver

    try:
        module, driver = build_private(source, digest, "_newton_linesearch_iterative_kernel")
        ellipse_module, ellipse_driver = build_private(
            ellipse_source, ellipse_digest, "_newton_elliptic_linesearch_iterative_kernel"
        )
    except BaseException:
        for name, filename in created:
            sys.modules.pop(name, None)
            linecache.cache.pop(filename, None)
        raise

    @functools.wraps(original)
    def selected(model, data, context, fuse_jv):
        if supports_model(model):
            if model.opt.cone == types.ConeType.ELLIPTIC:
                return ellipse_driver(model, data, context, fuse_jv)
            return driver(model, data, context, fuse_jv)
        return original(model, data, context, fuse_jv)

    record = {
        "enabled": True,
        "experimental": True,
        "patch_id": "newton-bracket-elliptic-huber-v3",
        "source_sha256": _SOURCE_SHA,
        "source_dependencies": dict(_DEPENDENCIES),
        "generated_factory_sha256": digest,
        "generated_elliptic_factory_sha256": ellipse_digest,
        "mujoco_warp_version": "3.12.0",
        "scope": "PYRAMIDAL/ELLIPTIC+NEWTON; unsupported models use the original driver",
        "friction_delta": True,
        "friction_delta_scope": "ELLIPTIC only; original gradients and Hessians",
        "ellipse_evaluator": "Inherited prepared-cone evaluator",
        "general_trajectory_equivalence_validated": False,
        "iteration_budget_changed": False,
        "gradient_tolerance_changed": False,
        "installed_package_modified": False,
        "warning_bits_suppressed": False,
    }
    solver._linesearch_iterative = selected
    _INSTALLED = (selected, record, module, ellipse_module)
    return dict(record)
