# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery of the FeatherPGS law tests (``test_feather_pgs_*_laws``, ``test_feather_pgs_law_source``).

Generated kernels that evaluate a law through its Warp function and through its
CUDA rendering, inputs that exercise each law's branches, bitwise and rounding
comparisons, references compiled from the replaced code (anchored in its module
or extracted from its generator's output), and the record of every unified
rounding variant. It defines no tests.
"""

import inspect
import linecache
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.feather_pgs import (
    contact_laws,
    field_group,
    law_source,
    projection_laws,
    row_kernels,
    row_laws,
)
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.device_preludes import cuda_native, dual_native
from newton._src.solvers.feather_pgs.factor.walk import (
    WALK_FRAME,
    WALK_LINK,
    WalkRelease,
    WalkTrees,
    pack_walk_records,
    release_steps,
    unpack_walk_records,
    walk_leave_kernel,
)
from newton._src.solvers.feather_pgs.friction_patches import PatchRowLaw
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.row_records import DenseRows, MfRows, RowSupport
from newton._src.solvers.feather_pgs.tree_scan import WalkClass
from newton.tests.unittest_utils import add_function_test, get_test_devices

FAMILIES = (contact_laws.CONTACT_LAW, row_laws.ROW_LAW, projection_laws.PROJECTION_LAW, projection_laws.CONE_LAW)
PACKAGE = Path(row_laws.__file__).parent


# ---------------------------------------------------------------------------------------------------------------------
# Generated kernels
# ---------------------------------------------------------------------------------------------------------------------


def exec_kernel(name: str, source: str, namespace: dict):
    """Compile generated Python source as a Warp kernel; linecache lets Warp read the source back."""
    filename = f"<fpgs-law-test-{name}-{abs(hash(source))}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return wp.kernel(namespace[name], module="unique", enable_backward=False)


def _is_bool(annotation) -> bool:
    return annotation in (bool, wp.bool)


def _storage(annotation):
    """Array dtype that carries one value of ``annotation`` (booleans as int32)."""
    return wp.int32 if _is_bool(annotation) else annotation


_TWIN_KERNELS = {}


def twin_kernels(law):
    """Return ``(warp_kernel, native_kernel)`` that evaluate ``law`` per thread through each rendering.

    Both take, per value argument, an array of that type indexed by the thread;
    array arguments pass through unchanged; each result lands in its own array.
    The native kernel calls the family's CUDA function on a CUDA device and its
    CPU twin on the CPU.
    """
    if law in _TWIN_KERNELS:
        return _TWIN_KERNELS[law]
    sig = law_source.law_signature(law)
    namespace = {"wp": wp, "law": law}
    inputs, call_args, native_args = [], [], []
    for k, (name, annotation) in enumerate(sig.parameters.items()):
        type_name = f"T_{k}"
        if wp.types.is_array(annotation):
            namespace[type_name] = annotation
            inputs.append(f"{name}: {type_name}")
            call_args.append(name)
            native_args.append(name)
        else:
            namespace[type_name] = _storage(annotation)
            inputs.append(f"{name}: wp.array[{type_name}]")
            call_args.append(f"{name}[i] != 0" if _is_bool(annotation) else f"{name}[i]")
            native_args.append(f"wp::index({name}, i) != 0" if _is_bool(annotation) else f"wp::index({name}, i)")
    outputs = []
    for j, annotation in enumerate(sig.results):
        namespace[f"R_{j}"] = _storage(annotation)
        outputs.append(f"out_{j}: wp.array[R_{j}]")
    signature = ", ".join([*inputs, *outputs])
    arguments = ", ".join([name.split(":")[0] for name in inputs + outputs])
    names = [f"r_{j}" for j in range(len(sig.results))]
    stores = []
    for j, annotation in enumerate(sig.results):
        if _is_bool(annotation):
            stores.append(f"    out_{j}[i] = wp.where(r_{j}, 1, 0)")
        else:
            stores.append(f"    out_{j}[i] = r_{j}")
    warp_source = (
        f"def warp_twin({signature}):\n    i = wp.tid()\n    {', '.join(names)} = law({', '.join(call_args)})\n"
        + "\n".join(stores)
        + "\n"
    )
    # The native twin: one native call per thread that writes every result.
    if len(sig.results) == 1:
        value = "result"
        native_stores = [f"wp::index(out_0, i) = {'result ? 1 : 0' if _is_bool(sig.results[0]) else 'result'};"]
    else:
        value = "result"
        native_stores = [
            f"wp::index(out_{j}, i) = {f'result.v{j} ? 1 : 0' if _is_bool(annotation) else f'result.v{j}'};"
            for j, annotation in enumerate(sig.results)
        ]
    family = next(f for f in FAMILIES if f.name == sig.family)
    body = (
        f"    const auto {value} = {sig.family}::{sig.name}({', '.join(native_args)});\n    "
        + "\n    ".join(native_stores)
        + "\n"
    )
    namespace["snippet"] = dual_native(body, preludes=family.preludes())
    native_source = (
        f"@wp.func_native(snippet)\ndef native(i: int, {signature}): ...\n\n"
        f"def native_twin({signature}):\n    native(wp.tid(), {arguments})\n"
    )
    kernels_pair = (
        exec_kernel("warp_twin", warp_source, dict(namespace)),
        exec_kernel("native_twin", native_source, dict(namespace)),
    )
    _TWIN_KERNELS[law] = kernels_pair
    return kernels_pair


def launch(kernel, n: int, inputs: dict, sig, device):
    """Launch a twin kernel over ``inputs`` (argument name to numpy array or Warp array) and return its results."""
    args = []
    for name, annotation in sig.parameters.items():
        value = inputs[name]
        if annotation is RowSupport:
            args.append(value)
        elif wp.types.is_array(annotation):
            args.append(
                value if isinstance(value, wp.array) else wp.array(value, dtype=annotation.dtype, device=device)
            )
        else:
            args.append(wp.array(np.asarray(value), dtype=_storage(annotation), device=device))
    outs = [wp.zeros(n, dtype=_storage(annotation), device=device) for annotation in sig.results]
    wp.launch(kernel, dim=n, inputs=args + outs, device=device)
    return [out.numpy() for out in outs]


def _bits(values: np.ndarray, signed_zeros: bool = True) -> np.ndarray:
    """An integer view of ``values`` for bitwise comparison (floats by their bits, NaN payloads unified)."""
    values = np.ascontiguousarray(values)
    if values.dtype.kind == "f":
        flat = values.reshape(values.shape[0], -1)
        if not signed_zeros:
            flat = np.where(flat == 0, np.zeros_like(flat), flat)
        bits = flat.view(np.uint32 if values.dtype == np.float32 else np.uint64).copy()
        bits[np.isnan(flat)] = 0x7FC00000
        return bits
    return values.reshape(values.shape[0], -1)


def mismatches(a: np.ndarray, b: np.ndarray, signed_zeros: bool = True) -> np.ndarray:
    """Rows whose values differ in any bit (NaN equals NaN; without ``signed_zeros``, -0 equals +0)."""
    return np.any(_bits(a, signed_zeros) != _bits(b, signed_zeros), axis=1)


UNIFICATIONS = {}
"""Per unified variant, from the last run of its test: ``(values changed, values compared, largest change in
units of the terms)``, which the phase report quotes."""


def unified(name: str, a: np.ndarray, b: np.ndarray, magnitude: np.ndarray, bound: float) -> float:
    """Record how far a unification moves the values ``a`` to ``b`` and return the largest move in term units."""
    a = np.asarray(a).reshape(np.asarray(a).shape[0], -1)
    b = np.asarray(b).reshape(a.shape)
    magnitude = np.broadcast_to(np.asarray(magnitude).reshape(a.shape[0], -1), a.shape)
    changed = int(mismatches(a, b).sum())
    worst = float(term_ulps(a, b, magnitude).max()) if a.size else 0.0
    UNIFICATIONS[name] = (changed, a.shape[0], worst)
    return worst


def term_ulps(a: np.ndarray, b: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """``|a - b|`` in float32 units of the magnitude of the terms that formed the values.

    A sum that cancels has fewer bits than its terms; a rounding variant of one term
    moves it by some units of that term, which is what a unification may change.
    """
    scale = np.maximum(np.abs(np.asarray(magnitude, dtype=np.float64)), np.finfo(np.float32).tiny)
    gap = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return gap / (scale * np.finfo(np.float32).eps)


def ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Units in the last place between float32 arrays, per element (0 for equal values, inf across signs or NaN)."""
    a32 = np.asarray(a, dtype=np.float32).ravel()
    b32 = np.asarray(b, dtype=np.float32).ravel()

    def ordered(x):
        bits = x.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    gap = np.abs(ordered(a32) - ordered(b32)).astype(np.float64)
    gap[(a32 == b32)] = 0.0
    gap[np.isnan(a32) != np.isnan(b32)] = np.inf
    gap[np.isnan(a32) & np.isnan(b32)] = 0.0
    return gap


CONTRACTIONS = {}
"""Per comparison on a CUDA device of a law with the Warp code it replaces, from the last run of its test:
``(rows that differ, rows compared, largest difference in units of the terms)``."""


def assert_reproduces(test, device, reference, law, magnitude=None, name: str = "", units: float = 4.0):
    """Assert that ``law`` reproduces ``reference``, the values of the Warp code it replaces (rows on axis 0).

    On the CPU the values are bitwise equal: Warp's CPU build rounds every operation as written. A CUDA build
    fuses multiply-adds by the shape of the surrounding code, so the same operations in two kernels may round
    apart: there the values agree within ``units`` float32 units of ``magnitude``, the size of the terms that
    formed them (a scalar, one per row or one per value; by default the values). Integers agree exactly.
    """
    a = np.asarray(reference)
    b = np.asarray(law)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(a.shape)
    if not wp.get_device(device).is_cuda or a.dtype.kind != "f":
        bad = mismatches(a, b)
        test.assertFalse(
            bad.any(), f"{name}: {int(bad.sum())} of {a.shape[0]} differ, first at {np.flatnonzero(bad)[:3]}"
        )
        return
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), name)
    a64 = np.nan_to_num(a.astype(np.float64))
    b64 = np.nan_to_num(b.astype(np.float64))
    scale = np.maximum(np.abs(a64), np.abs(b64)) if magnitude is None else np.asarray(magnitude, dtype=np.float64)
    if scale.ndim == 1 and scale.shape[0] == a.shape[0]:
        scale = scale[:, None]
    worst = float(term_ulps(a64, b64, np.broadcast_to(scale, a.shape)).max()) if a.size else 0.0
    CONTRACTIONS[name] = (int(mismatches(a, b).sum()), a.shape[0], worst)
    test.assertLessEqual(worst, units, name)


# ---------------------------------------------------------------------------------------------------------------------
# Inputs of each law for the twin comparison: random values plus the edge cases its branches test
# ---------------------------------------------------------------------------------------------------------------------

TWIN_SAMPLES = 4096

TWIN_TOLERANCES = {
    "friction_pair": ("radius", 1.0e-5),
    "friction_pair_from_metric": ("radius", 1.0e-5),
    "friction_ball": ("load", 1.0e-5),
    "friction_ball_from_metric": ("load", 1.0e-5),
}
"""Laws whose two renderings agree to their solve's tolerance rather than bitwise: ``(input that scales the
results, or None for the largest result, bound)``. They test convergence and sticking against tolerances, and the
CUDA compiler fuses their multiply-adds differently in Warp's lowering (branches merged with ``wp::where``) and in
the plain C++ rendering, so a rare sample takes the other branch of a test: :data:`TWIN_AGREEING` of the samples
agree within the bound, every sample within a thousand times it. Today's Warp code and hand-written CUDA text of the
pair law differ the same way."""

TWIN_AGREEING = 0.995
"""Fraction of an iterative law's samples whose renderings agree within its :data:`TWIN_TOLERANCES` bound."""


def unit_vectors(rng, n):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v[: n // 16] = np.eye(3)[rng.integers(0, 3, n // 16)] * rng.choice([-1.0, 1.0], (n // 16, 1))
    return v.astype(np.float32)


def scalars(rng, n, scale=1.0):
    x = rng.normal(scale=scale, size=n)
    x[::7] = 0.0
    x[3::11] = rng.choice([1.0e-7, -1.0e-7, 1.0e-6, -1.0e-6, 2.0e-6], size=x[3::11].shape)
    return x.astype(np.float32)


def transforms(rng, count):
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = rng.normal(scale=5.0, size=(count, 3))
    return np.concatenate([p, q], axis=1).astype(np.float32)


def spd3(rng, n):
    a = rng.normal(size=(n, 3, 3))
    w = a @ np.transpose(a, (0, 2, 1)) + 1.0e-3 * np.eye(3)
    w[::13] *= 1.0e-22  # degenerate blocks
    return w.astype(np.float32)


def twin_inputs(name: str, rng, n: int, device) -> dict:
    """Inputs for the twin comparison of law ``name``."""
    ints = lambda lo, hi: rng.integers(lo, hi, n).astype(np.int32)  # noqa: E731
    vec = lambda scale=1.0: rng.normal(scale=scale, size=(n, 3)).astype(np.float32)  # noqa: E731
    shape_body = np.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    body_to_art = np.array([0, 0, 1, -1, 2, 3, 3], dtype=np.int32)
    art_to_world = np.array([0, 1, 1, 0], dtype=np.int32)
    body_q = transforms(rng, 8)
    common = {
        "shape_a": ints(-1, 8),
        "shape_b": ints(-1, 8),
        "body_a": ints(-1, 8),
        "body_b": ints(-1, 8),
        "art_a": ints(-1, 4),
        "art_b": ints(-1, 4),
        "body": ints(0, 7),
        "art": ints(-1, 4),
    }
    if name in ("contact_bodies", "contact_endpoints"):
        return {
            **common,
            "shape_body": shape_body,
            "body_to_articulation": body_to_art,
            "art_to_world": art_to_world,
        }
    if name == "contact_responds":
        return {
            **common,
            "articulation_response_dof_count": np.array([3, 0, 6, 1], dtype=np.int32),
            "body_has_response_dofs": np.array([1, 0, 1, 1, 1, 1, 0], dtype=np.int32),
            "body_flags": np.array([0, 0, 2, 0, 1, 0, 0], dtype=np.int32),
        }
    if name == "contact_geometry":
        contacts = 64
        return {
            "c": rng.integers(0, contacts, n).astype(np.int32),
            "shape0": rng.integers(-1, 8, contacts).astype(np.int32),
            "shape1": rng.integers(-1, 8, contacts).astype(np.int32),
            "point0": rng.normal(scale=0.3, size=(contacts, 3)).astype(np.float32),
            "point1": rng.normal(scale=0.3, size=(contacts, 3)).astype(np.float32),
            "normal": unit_vectors(rng, contacts),
            "margin0": np.abs(rng.normal(scale=0.01, size=contacts)).astype(np.float32),
            "margin1": np.abs(rng.normal(scale=0.01, size=contacts)).astype(np.float32),
            "shape_body": shape_body,
            "body_q": body_q,
        }
    if name.startswith("contact_points"):
        return {
            **common,
            "normal_ab": unit_vectors(rng, n),
            "point0": vec(0.3),
            "point1": vec(0.3),
            "margin0": np.abs(scalars(rng, n, 0.01)),
            "margin1": np.abs(scalars(rng, n, 0.01)),
            "body_q": body_q,
        }
    if name in ("contact_materials", "contact_friction", "contact_restitution"):
        values = np.array([0.5, 0.8, np.nan, -0.5, 1.5, np.inf, 0.0, 0.3], dtype=np.float32)
        return {**common, "shape_mu": rng.uniform(0, 1.5, 8).astype(np.float32), "shape_restitution": values}
    if name == "contact_tangent_basis":
        return {"n": unit_vectors(rng, n)}
    if name == "contact_point":
        return {"translation_a": vec(3.0), "offset_a": vec(0.3), "separation": vec(0.01)}
    if name == "contact_row_frame":
        return {
            "row": ints(0, 4),
            "normal": unit_vectors(rng, n),
            "tangent0": unit_vectors(rng, n),
            "tangent1": unit_vectors(rng, n),
            "point": vec(),
            "friction_point": vec(),
        }
    if name == "jacobian_entry":
        return {
            "motion": rng.normal(size=(n, 6)).astype(np.float32),
            "direction": unit_vectors(rng, n),
            "point": vec(3.0),
            "origin": vec(3.0),
        }
    if name == "contact_seed":
        return {"direction": unit_vectors(rng, n), "lever": vec(3.0)}
    if name == "spin_entry":
        return {"motion": rng.normal(size=(n, 6)).astype(np.float32), "normal": unit_vectors(rng, n)}
    if name == "spin_seed":
        return {"normal": unit_vectors(rng, n)}
    if name == "prescribed_spin_target":
        return {
            **common,
            "normal": unit_vectors(rng, n),
            "prescribed_articulation": np.array([1, 0, 1, 1], dtype=np.int32),
            "body_v_s": rng.normal(size=(8, 6)).astype(np.float32),
        }
    if name == "prescribed_target":
        return {
            **common,
            "point": vec(),
            "direction": unit_vectors(rng, n),
            "prescribed_articulation": np.array([1, 0, 1, 1], dtype=np.int32),
            "articulation_origin": rng.normal(size=(4, 3)).astype(np.float32),
            "body_v_s": rng.normal(size=(8, 6)).astype(np.float32),
        }
    if name == "contact_capped_beta":
        return {
            "phi": scalars(rng, n, 0.05),
            "beta": rng.uniform(0.0, 1.0, n).astype(np.float32),
            "cap": rng.choice(np.array([0.5, 2.0, 0.0, -1.0, np.inf, 1.0e30], dtype=np.float32), n),
            "dt": np.full(n, 1.0 / 240.0, dtype=np.float32),
        }
    if name == "contact_depenetration_cap":
        caps = np.array([2.0, 0.0, np.inf, 0.5, -1.0, 1.0e30, 3.0, 0.25], dtype=np.float32)
        return {**common, "max_depenetration_velocity": caps}
    if name in ("friction_pair", "friction_pair_from_metric"):
        a = rng.uniform(0.01, 4.0, n)
        d = rng.uniform(0.01, 4.0, n)
        c = rng.uniform(-1, 1, n) * np.sqrt(a * d) * rng.uniform(0.0, 0.999, n)
        radius = rng.uniform(0.0, 2.0, n)
        radius[::17] = 0.0
        pair = {
            "residual": rng.normal(size=(n, 2)).astype(np.float32),
            "old": rng.normal(size=(n, 2)).astype(np.float32),
            "radius": radius.astype(np.float32),
            "omega": rng.choice([1.0, 0.8, 1.2], n).astype(np.float32),
        }
        if name == "friction_pair":
            return {"a": a.astype(np.float32), "c": c.astype(np.float32), "d": d.astype(np.float32), **pair}
        # The block's metric (friction_pair_metric's decomposition, in float64).
        scale = np.maximum(a, d)
        a, c, d = a / scale, c / scale, d / scale
        root = np.sqrt((a - d) ** 2 + 4.0 * c * c)
        largest = 0.5 * (a + d + root)
        axis = np.where((a >= d)[:, None], np.stack([largest - d, c], 1), np.stack([c, largest - a], 1))
        length = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = np.where(length > 0.0, axis / np.maximum(length, 1.0e-300), np.array([1.0, 0.0]))
        return {
            "scale": scale.astype(np.float32),
            "largest": largest.astype(np.float32),
            "smallest": np.maximum((a * d - c * c) / largest, 0.0).astype(np.float32),
            "axis": axis.astype(np.float32),
            **pair,
        }
    if name == "ball":
        # The cone's projection: zero coefficients and loads among them (a zero mu_t is the disk's branch).
        mu, mu_t, load = (rng.uniform(0.0, 1.5, n).astype(np.float32) for _ in range(3))
        mu[::7], mu_t[::5], load[::11] = 0.0, 0.0, 0.0
        return {"triple": rng.normal(size=(n, 3)).astype(np.float32), "mu": mu, "mu_t": mu_t, "load": load}
    if name.startswith(("friction_ball", "jacobi_rotation")):
        block = spd3(rng, n).astype(np.float64)
        mu = rng.uniform(0.0, 2.0, n)
        mu[::19] = 0.0
        mu_t = mu * rng.uniform(0.0, 0.2, n)
        mu_t[::23] = 0.0
        load = rng.uniform(0.0, 2.0, n)
        load[::17] = 0.0
        coefficients = np.stack([mu, mu, mu_t], 1)
        triple = {
            "residual": rng.normal(size=(n, 3)).astype(np.float32),
            "old": (rng.normal(size=(n, 3)) * coefficients * load[:, None]).astype(np.float32),
            "mu": mu.astype(np.float32),
            "mu_t": mu_t.astype(np.float32),
            "load": load.astype(np.float32),
            "omega": rng.choice([1.0, 0.8, 1.2], n).astype(np.float32),
        }
        if name == "jacobi_rotation":
            vectors, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
            p, q = np.array([[0, 1], [0, 2], [1, 2]])[rng.integers(0, 3, n)].T
            return {"matrix": block.astype(np.float32), "vectors": vectors.astype(np.float32), "p": p, "q": q}
        if name == "friction_ball_metric":
            return {"block": block.astype(np.float32), **{k: triple[k] for k in ("mu", "mu_t")}}
        if name == "friction_ball":
            return {"block": block.astype(np.float32), **triple}
        # The block's metric (friction_ball_metric's decomposition, in float64).
        scaled = block * coefficients[:, :, None] * coefficients[:, None, :]
        scale = np.maximum(np.einsum("nii->ni", scaled).max(axis=1), 1.0e-20)
        values, vectors = np.linalg.eigh(scaled / scale[:, None, None])
        return {
            "scale": scale.astype(np.float32),
            "values": np.maximum(values, 0.0).astype(np.float32),
            "vectors": vectors.astype(np.float32),
            **triple,
        }
    if name == "speed_row":
        return {
            "speed": np.abs(scalars(rng, n, 3.0)),
            "limit": rng.uniform(0.1, 3.0, n).astype(np.float32),
            "effective_mass": rng.uniform(0.1, 2.0, n).astype(np.float32),
            "old": np.abs(scalars(rng, n)),
        }
    if name in ("velocity_limit_update",):
        return {"old": np.abs(scalars(rng, n)), "step": scalars(rng, n)}
    if name in ("joint_limit_side",):
        lower = scalars(rng, n)
        upper = lower + np.abs(scalars(rng, n))
        lower[::9] = -np.inf
        upper[::10] = np.inf
        return {
            "q": scalars(rng, n),
            "lower": lower,
            "upper": upper,
            "side": ints(0, 2),
            "activation_gap": np.abs(scalars(rng, n, 0.1)),
        }
    if name == "velocity_limit_active":
        limit = scalars(rng, n)
        return {"limit": limit, "velocity": scalars(rng, n), "activation_fraction": np.abs(scalars(rng, n, 0.5))}
    # Generic: scalars by name, kinds and counts as small integers.
    sig = law_source.law_signature(LAWS[name])
    values = {}
    for param, annotation in sig.parameters.items():
        if param in ("kind",):
            values[param] = ints(0, 8)
        elif param in ("inv_dt",):
            values[param] = rng.choice([60.0, 120.0, 1.0 / 7.0e-3, 240.0], n).astype(np.float32)
        elif param in ("dt",):
            values[param] = rng.choice([1.0 / 60.0, 1.0 / 120.0, 7.0e-3], n).astype(np.float32)
        elif param in ("contact_w", "w"):
            values[param] = rng.choice([1.0, 0.5, 0.98, 0.25], n).astype(np.float32)
        elif annotation is wp.vec2:
            values[param] = rng.normal(size=(n, 2)).astype(np.float32)
        elif annotation is wp.vec3:
            values[param] = vec()
        elif annotation in (int, wp.int32):
            values[param] = ints(-2, 6)
        else:
            values[param] = scalars(rng, n)
    return values


def _registered_laws():
    """Every law of :data:`FAMILIES` the modules define, by name (a law they import from another family, the
    spatial kit's, is that family's to test)."""
    families = {family.name for family in FAMILIES}
    laws = {}
    for module in (contact_laws, row_laws, projection_laws):
        for name, value in vars(module).items():
            if isinstance(value, wp.Function):
                try:
                    sig = law_source.law_signature(value)
                except ValueError:
                    continue
                if sig.name == name and sig.family in families:
                    laws[name] = value
    return laws


LAWS = _registered_laws()


CONTACT_ARGUMENTS = {
    "count": "contact_count",
    "point0": "contact_point0",
    "point1": "contact_point1",
    "normal": "contact_normal",
    "shape0": "contact_shape0",
    "shape1": "contact_shape1",
    "margin0": "contact_thickness0",
    "margin1": "contact_thickness1",
    "world": "contact_world",
    "slot": "contact_slot",
    "art0": "contact_art_a",
    "art1": "contact_art_b",
    "path": "contact_path",
    "slots_needed": "contact_slots_needed",
    "shape_body": "shape_body",
    "body_q": "body_q",
    "body_v": "body_v_s",
    "body_mask": "body_response_dof_mask",
    "response_dofs": "articulation_response_dof_count",
    "dof_start": "art_dof_start",
    "origin": "articulation_origin",
    "motion": "joint_S_s",
    "prescribed": "prescribed_articulation",
    "material_mu": "shape_material_mu",
    "material_restitution": "shape_material_restitution",
    "max_depenetration_velocity": "rigid_body_max_depenetration_velocity",
    "incident": "v_hat",
    "beta": "pgs_beta",
    "cfm": "pgs_cfm",
    "row_type": "world_row_type",
    "row_parent": "world_row_parent",
    "row_mu": "world_row_mu",
    "row_beta": "world_row_beta",
    "phi": "world_phi",
    "target": "world_target_velocity",
    "restitution": "world_row_restitution",
}
"""The scenes' names of the :data:`ContactInput` fields: the reference laws take them as separate arguments."""


DENSE_ROW_ARGUMENTS = {
    "cfm": "pgs_cfm",
    "row_type": "world_row_type",
    "row_parent": "world_row_parent",
    "row_mu": "world_row_mu",
    "row_beta": "world_row_beta",
    "phi": "world_phi",
    "target": "world_target_velocity",
    "restitution": "world_row_restitution",
    "rhs": "world_rhs",
    "diag": "world_diag",
    "row_w": "world_row_w",
}
"""The scenes' names of the :data:`DenseRows` fields."""

JOINT_TABLE_ARGUMENTS = {field.name: field.name for field in field_group.fields_of(JointTable)}
"""The scenes' names of the :data:`JointTable` fields: the fields' own."""

MF_ROW_ARGUMENTS = {
    **{field.name: field.attr for field in field_group.fields_of(MfRows) if field.attr.startswith("mf_")},
    "row_type": "mf_row_type",
    "row_parent": "mf_row_parent",
    "row_mu": "mf_row_mu",
    "phi": "mf_phi",
    "restitution": "mf_row_restitution",
    "incident": "mf_row_incident",
    "rhs": "mf_rhs",
    "target": "mf_target_velocity",
    "eps": "mf_row_eps",
    "diag": "mf_diag",
    "sides": "mf_sides",
    "first": "mf_first",
}
"""The scenes' names of the :data:`MfRows` fields: the placement's free-body attributes, and the row store's records
under ``mf_`` names (a scene's free bodies' rows from row 0: ``base`` 0)."""

STRUCT_ARGUMENTS = {
    ContactInput: CONTACT_ARGUMENTS,
    DenseRows: DENSE_ROW_ARGUMENTS,
    JointTable: JOINT_TABLE_ARGUMENTS,
    MfRows: MF_ROW_ARGUMENTS,
    PatchRowLaw: {"patches": "friction_patches", "beta": "friction_anchor_beta"},
}
"""The scenes' field names of each struct a kernel takes in place of separate arguments."""


_ROW_LINES = {}


def row_bias(dt, speculative_scale, contact_w, restitution_threshold, weighted, material) -> row_kernels.RowBias:
    """The rows' law's step parameters (:class:`.row_kernels.RowBias`)."""
    bias = row_kernels.RowBias()
    bias.dt, bias.speculative_scale, bias.contact_w = dt, speculative_scale, contact_w
    bias.restitution_threshold, bias.weighted, bias.material = restitution_threshold, int(weighted), int(material)
    return bias


def row_lines(device) -> wp.array:
    """The row table of a hand-built scene, read by world: its worlds keep their rows on their own lines."""
    key = str(device)
    if key not in _ROW_LINES:
        _ROW_LINES[key] = wp.array(np.arange(1 << 14), dtype=wp.int32, device=device)
    return _ROW_LINES[key]


def bind_struct(struct_type, values, device):
    """An instance of ``struct_type`` from a scene's values by the scenes' names (host values go to ``device``).

    Returns the instance and its arrays by the scenes' names.
    """
    arguments = STRUCT_ARGUMENTS[struct_type]
    fields = {field.name: field for field in field_group.fields_of(struct_type)}
    bound = {}
    for name, argument in arguments.items():
        if argument not in values:
            continue
        value = values[argument]
        if fields[name].dims and not isinstance(value, wp.array):
            value = wp.array(np.asarray(value), dtype=fields[name].dtype, device=device)
        bound[name] = value
    if "sides" in fields and "sides" not in bound and "row_type" in bound:
        # The producers record each row's sides (the packed placement's); a scene without them reads none back.
        bound["sides"] = wp.zeros(bound["row_type"].shape, dtype=wp.vec2i, device=device)
    for name in ("first", "mf_first"):
        if name in fields and name not in bound:
            bound[name] = row_lines(device)
    instance = field_group.upload(struct_type, device, **bound)
    return instance, {
        arguments[name]: value for name, value in bound.items() if isinstance(value, wp.array) and name in arguments
    }


def contact_input(values, device):
    """The :data:`ContactInput` of a scene's values (:func:`bind_struct`), its bodies uncapped unless they say."""
    uncapped = np.full(len(values["body_q"]), np.inf, dtype=np.float32)
    return bind_struct(ContactInput, {"rigid_body_max_depenetration_velocity": uncapped, **values}, device)


def row_support(sides, dofs, offsets, dense, device, worlds=None) -> RowSupport:
    """A packed :class:`.row_records.RowSupport` of test rows: ``sides`` ``[world, row, 2]`` articulation indices (-1
    for none), each articulation's coordinates, first world coordinate and world (``worlds``, default 0), and the
    layout (``dense``: entry ``e`` is coordinate ``e``)."""
    support = RowSupport()
    support.sides = wp.array(np.ascontiguousarray(sides, dtype=np.int32), dtype=wp.vec2i, device=device)
    support.dofs = wp.array(np.asarray(dofs, dtype=np.int32), dtype=wp.int32, device=device)
    support.offset = wp.array(np.asarray(offsets, dtype=np.int32), dtype=wp.int32, device=device)
    worlds = np.zeros(len(dofs), dtype=np.int32) if worlds is None else np.asarray(worlds, dtype=np.int32)
    support.world = wp.array(worlds, dtype=wp.int32, device=device)
    support.dense = int(dense)
    support.packed = 1
    support.first = row_lines(device)
    return support


def group_rows(device) -> RowSupport:
    """The size groups' rows as the producers' placement (:class:`.row_records.RowSupport`, not packed): a producer
    writes an articulation's side of a row at its group, from entry 0."""
    support = row_support(np.full((1, 1, 2), -1, dtype=np.int32), [0], [0], 0, device)
    support.packed = 0
    return support


PREFIX_RECORDS = {
    "row_type": wp.int32,
    "row_parent": wp.int32,
    "row_mu": wp.float32,
    "row_beta": wp.float32,
    "phi": wp.float32,
    "target": wp.float32,
    "sides": wp.vec2i,
}
"""The records the general route's joint row warps write (:func:`prefix_warp_rows`), by their law's names."""


def prefix_warp_rows(
    device, arts: int, size: int, rows: int, *, family="limits", mimics=0, resolved=None, warps_per_block=4, **law
) -> dict:
    """Launch a joint row family's warp of the general route (``family``: the joint limits'
    :func:`.rows.joint_limit_warp_kernel`, the joint velocity limits' ``"velocity"`` or the mimics' ``"mimics"``, at most
    ``mimics`` an articulation), as :meth:`.rows.RowLaws.write` launches it, on the size groups' rows.

    Articulation ``a`` is world ``a`` and group ``a``, its ``size`` coordinates from ``a * size``; its world holds
    ``rows`` rows. ``law`` holds the law's inputs by :data:`.prefix_rows.LAW`'s (and the family's) names; every
    coordinate is a limit coordinate and none is prescribed unless it says. ``resolved`` is the worlds' resolved mask
    (``None``: none). Returns each world's row count (``count``), the Jacobian rows ``J``, the records by name and every
    array the law was given, as numpy arrays.
    """
    from newton._src.solvers.feather_pgs import prefix_rows  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import rows as producers  # noqa: PLC0415

    struct_type, kernel = {
        "limits": (prefix_rows.PrefixLaw, producers.joint_limit_warp_kernel(size, warps_per_block)),
        "velocity": (prefix_rows.PrefixVelocityLaw, producers.velocity_limit_warp_kernel(size, warps_per_block)),
        "mimics": (prefix_rows.PrefixMimicLaw, producers.mimic_warp_kernel(size, warps_per_block, mimics)),
    }[family]

    dofs = arts * size
    law = {"limit_q_index": np.arange(dofs), "prescribed": np.zeros(dofs), **law}
    counter = wp.zeros(arts, dtype=wp.int32, device=device)
    records = {name: wp.zeros((arts, rows), dtype=dtype, device=device) for name, dtype in PREFIX_RECORDS.items()}
    fields = {field.name: field for field in field_group.fields_of(struct_type)}
    law = {
        name: value if not fields[name].dims else wp.array(np.asarray(value), dtype=fields[name].dtype, device=device)
        for name, value in law.items()
    }
    struct = field_group.upload(
        struct_type,
        device,
        slot_counter=counter,
        first=row_lines(device),
        capacity=rows,
        **records,
        **law,
    )
    J = wp.zeros((arts, rows, size), dtype=wp.float32, device=device)
    order = wp.array(np.arange(arts), dtype=wp.int32, device=device)
    wp.launch_tiled(
        kernel,
        dim=[(arts + warps_per_block - 1) // warps_per_block],
        inputs=[
            arts,
            wp.array(size * np.arange(arts), dtype=wp.int32, device=device),
            order,
            order,
            wp.array(np.zeros(0) if resolved is None else np.asarray(resolved), dtype=wp.int32, device=device),
            struct,
            group_rows(device),
        ],
        outputs=[J],
        block_dim=32 * warps_per_block,
        device=device,
    )
    arrays = {name: value for name, value in law.items() if isinstance(value, wp.array)}
    return {
        "count": counter.numpy(),
        "J": J.numpy(),
        **{name: array.numpy() for name, array in {**arrays, **records}.items()},
    }


def local_packed_rows(device, art_world, rows_by_art, incident=None) -> list:
    """The local owners' row operands (:data:`.sweep_routes_local.LOCAL_ABI`): ``packed_Z``, ``row_sides``,
    ``row_incident``, ``response_dofs``, ``articulation_world_dof_offset`` and ``packed_dense`` (1). ``rows_by_art[a]``
    is articulation ``a``'s rows ``[rows, n]`` as the one writer stores them (whitened, :func:`chain_walk`'s ``Z``);
    each world's rows hold its articulations' coordinates in articulation order (dense), every row on every
    articulation of its world. ``incident`` is each world's rows' ``J v_hat`` ``[worlds, rows]`` (zero by default)."""
    art_world = np.asarray(art_world, dtype=np.int32)
    rows_by_art = [np.asarray(rows, dtype=np.float32) for rows in rows_by_art]
    dofs = np.array([rows.shape[1] for rows in rows_by_art], dtype=np.int32)
    worlds, capacity = int(art_world.max()) + 1, rows_by_art[0].shape[0]
    offsets = np.zeros(len(dofs), dtype=np.int32)
    width = np.zeros(worlds, dtype=np.int32)
    for art, world in enumerate(art_world):
        offsets[art], width[world] = width[world], width[world] + dofs[art]
    packed = np.zeros((worlds, capacity, int(width.max())), dtype=np.float32)
    sides = np.full((worlds, capacity, 2), -1, dtype=np.int32)
    for art, world in enumerate(art_world):
        packed[world, :, offsets[art] : offsets[art] + dofs[art]] = rows_by_art[art]
        side = 0 if sides[world, 0, 0] < 0 else 1
        sides[world, :, side] = art
    if incident is None:
        incident = np.zeros((worlds, capacity), dtype=np.float32)
    return [
        wp.array(packed, dtype=wp.float32, device=device),
        wp.array(sides, dtype=wp.vec2i, device=device),
        wp.array(np.asarray(incident, dtype=np.float32), dtype=wp.float32, device=device),
        wp.array(dofs, dtype=wp.int32, device=device),
        wp.array(offsets, dtype=wp.int32, device=device),
        1,
    ]


def _walked_coordinates(solver):
    """Where the placement walks (:mod:`.walk_rows`), ``u_hat`` in the walk's frame, ``[world, world coordinate]``: the
    walk's enter of ``v_hat`` (:func:`.factor.walk.walk_enter_kernel`, the audit's), the coordinates its rows are
    whitened against; ``None`` where nothing walks (no coordinate responds, so no row has a side)."""
    if not getattr(solver.placement, "walks", False):
        return None
    from newton._src.solvers.feather_pgs.factor.walk import walk_enter_kernel  # noqa: PLC0415

    forest, factor = solver.forest, solver.factor
    coordinates = wp.zeros(
        (forest.world_count, max(int(forest.max_world_dofs), 1)), dtype=float, device=solver.model.device
    )
    wp.launch(
        walk_enter_kernel(),
        dim=factor.walk_arts.shape[0],
        inputs=[
            factor.walk_arts,
            factor.walk_view(),
            forest.art_to_world,
            forest.articulation_world_dof_offset,
            solver.v_hat,
            coordinates,
        ],
        device=solver.model.device,
    )
    return coordinates.numpy().astype(np.float64)


def walk_inverse(solver, art: int) -> np.ndarray:
    """``L^-1`` of articulation ``art``'s walk in its physical coordinates (float64 from the walk's float32 records):
    column ``i`` is the walk up of the unit force at coordinate ``i`` (``WALK_UP_TREE``'s steps), the factor the rows
    are whitened by. With ``inverse`` this, ``H^-1 = inverse^T inverse`` (the walk's ``L L^T = H`` with
    ``L = inverse^-1``), a row's footprints are ``z = inverse J^T`` and its response ``H^-1 J^T = inverse^T z``."""
    factor = solver.factor
    d0 = int(solver.forest.articulation_dof_start.numpy()[art])
    j0, j1 = int(solver.model.articulation_start.numpy()[art]), int(solver.forest.articulation_joint_end_host[art])
    path = factor.walk_path_host[j0:j1]
    n = int(solver.forest.articulation_dof_count.numpy()[art])
    data = unpack_walk_records(factor.walk_data.numpy(), d0 + n)[d0:].astype(np.float64)
    # Each body's offset to its parent's frame, as the walk forms it: the float32 difference of the two frame points.
    frames = factor.walk_link.numpy()[j0:j1, WALK_FRAME : WALK_FRAME + 3]
    link = np.array([frames[p] - frames[b] if p >= 0 else np.zeros(3, np.float32) for b, p in enumerate(path[:, 0])])
    link = link.astype(np.float64)
    first = [int(f) for f in path[:, 1]] + [n]
    inverse = np.zeros((n, n))
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        state, p = np.zeros((len(path), 6)), np.zeros(6)
        for b in range(len(path) - 1, -1, -1):
            p += state[b]
            for c in range(first[b + 1] - 1, first[b] - 1, -1):
                z = (x[c] + data[c, :6] @ p) * data[c, 12]
                p -= data[c, 6:12] * z
                inverse[c, i] = z
            up = int(path[b, 0])
            if up >= 0:
                p[3:] -= np.cross(link[b, :3], p[:3])
            if up != b - 1:
                if up >= 0:
                    state[up] += p
                p = np.zeros(6)
    return inverse


def dense_factor32(A) -> np.ndarray:
    """Main's float32 dense factor of the symmetric matrix ``A`` (physical order, ``H + R + K``; main's ``H`` is the float32
    assembly about the origin, :class:`.feather_pgs_mass_reference.MassReference` at ``dtype=np.float32``): the
    leaf-first ``P A P^T = L L^T`` the held tree built before the walk replaced it, over ``A`` rounded to float32, in its
    order: each pivot subtracts its row's squares from the first column on and takes the square root, and each entry
    below subtracts its products and scales by the pivot's reciprocal. A reference for the walk's precision, not a
    solver path. Returns ``L``, float32 lower triangular, leaf first."""
    A = np.asarray(A, dtype=np.float64).astype(np.float32)[::-1, ::-1]
    n = A.shape[0]
    L = np.zeros((n, n), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        for col in range(n):
            value = A[col, col]
            for k in range(col):
                value = np.float32(value - np.float32(L[col, k] * L[col, k]))
            L[col, col] = np.sqrt(value)
            reciprocal = np.float32(1.0) / L[col, col]
            for row in range(col + 1, n):
                value = A[row, col]
                for k in range(col):
                    value = np.float32(value - np.float32(L[row, k] * L[col, k]))
                L[row, col] = np.float32(value * reciprocal)
    return L


def walk_factor(lower) -> np.ndarray:
    """``A = P L^T P`` of a leaf-first factor ``L`` (``P H P^T = L L^T``): lower triangular in coordinate order, the
    walk's ``u = A qd`` of :func:`chain_walk`, ``H = A^T A``; a row's footprints are ``z = A^-T J^T``."""
    lower = np.asarray(lower, dtype=np.float64)
    return lower.T[::-1, ::-1].copy()


def chain_walk(device, lowers) -> WalkTrees:
    """The walk (:class:`.factor.walk.WalkTrees`) of chains whose factors are the given leaf-first ``L`` (one an
    articulation, their coordinates one after another): body ``c`` holds coordinate ``c`` (a chain of at most six),
    ``s_c = e_c``, ``ut_c[k] = A[c, k]`` below the diagonal and ``1 / ell_c = 1 / A[c, c]`` (:func:`walk_factor`), its
    links and frames zero, and the step's columns the factor's, so a wrench seed's ``J`` is its first coordinates' components and
    a covector's footprints and the leave are ``A^-T`` and ``A^-1``. No tree slips."""
    factors = [walk_factor(lower) for lower in lowers]
    sizes = [A.shape[0] for A in factors]
    if max(sizes) > 6:
        raise ValueError("a chain walk holds at most six coordinates an articulation")
    start = np.cumsum([0, *sizes]).astype(np.int32)
    total = int(start[-1])
    s = np.zeros((total, 6), dtype=np.float32)
    ut = np.zeros((total, 6), dtype=np.float32)
    inverse = np.zeros(total, dtype=np.float32)
    now = np.zeros((total, 6), dtype=np.float32)
    path = np.zeros((total, 2), dtype=np.int32)
    dof_body = np.zeros(total, dtype=np.int32)
    for art, A in enumerate(factors):
        for c in range(A.shape[0]):
            d = int(start[art]) + c
            s[d, c] = now[d, c] = 1.0
            ut[d, :c] = A[c, :c]
            inverse[d] = 1.0 / A[c, c]
            path[d] = (c - 1, c)
            dof_body[d] = c
    arts = len(factors)

    def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    walk = WalkTrees()
    walk.data = wp.array(pack_walk_records(s, ut, inverse, np.zeros(total)), dtype=wp.float32, device=device)
    walk.link = wp.zeros((total, WALK_LINK), dtype=wp.float32, device=device)
    walk.path = wp.array(path, dtype=wp.vec2i, device=device)
    walk.now = wp.array(np.asarray(now, dtype=np.float32).reshape(-1, 6), dtype=wp.spatial_vector, device=device)
    walk.now_link = wp.zeros((total, WALK_LINK), dtype=wp.float32, device=device)
    walk.now_com = wp.zeros(total, dtype=wp.vec3, device=device)
    # One geometry's frames, at the world's origin: the factor's and the step's origins agree.
    walk.origin = walk.now_origin = wp.zeros(arts, dtype=wp.vec3, device=device)
    walk.joint_art = ints(np.repeat(np.arange(arts), sizes))
    walk.scratch = wp.zeros(total, dtype=wp.spatial_vector, device=device)
    walk.joint_start, walk.joint_end = ints(start[:-1]), ints(start[1:])
    walk.dof_start, walk.dofs = ints(start[:-1]), ints(sizes)
    walk.dof_body, walk.body_joint = ints(dof_body), ints(np.arange(total))
    # Each chain's articulation at its first coordinate (a free body's tree, found by a sweep).
    walk.body_art = walk.joint_art
    first_art = np.full(total, -1, dtype=np.int32)
    first_art[start[:-1]] = np.arange(arts)
    walk.first_art = ints(first_art)
    release = WalkRelease()
    release.count = ints(np.zeros(arts))
    release.dof = wp.zeros((1, 1), dtype=wp.int32, device=device)
    release.step = release_steps((1, 1, 1), device)
    walk.release = release
    walk.du = wp.zeros(total, dtype=wp.float32, device=device)
    return walk


def diagonal_walk(inverse, device, dofs=None, offsets=None):
    """The walk of each world's dense articulation whose mass matrix is diagonal, ``H = diag(1 / inverse[w])``, as the
    step's one leave walks it (:func:`.factor.walk.walk_leave_kernel`, :func:`leave_walk`): world ``w``'s articulation
    ``w`` at its world's ``offsets[w]`` of ``dofs`` coordinates (global coordinate ``w dofs + d``), a chain of one
    coordinate a body whose columns and lifts are zero, so ``WALK_DOWN`` leaves ``u_c / ell_c = u_c sqrt(m_c^-1)``.
    Returns its :class:`.factor.walk.WalkTrees`."""
    inverse = np.asarray(inverse, np.float32)
    worlds, n = inverse.shape
    dofs = n if dofs is None else int(dofs)
    offsets = np.zeros(worlds, np.int32) if offsets is None else np.asarray(offsets, np.int32)
    first = (np.arange(worlds) * dofs + offsets).astype(np.int32)
    walk = WalkTrees()
    root = np.zeros(worlds * dofs, np.float32)
    for world in range(worlds):
        root[first[world] : first[world] + n] = np.sqrt(inverse[world])
    zero = np.zeros((worlds * dofs, 6), np.float32)
    walk.data = wp.array(pack_walk_records(zero, zero, root, np.zeros(worlds * dofs)), device=device)
    walk.link = wp.zeros((worlds * n, WALK_LINK), dtype=wp.float32, device=device)
    path = np.stack([np.arange(n) - 1, np.arange(n)], axis=1).astype(np.int32)
    walk.path = wp.array(np.tile(path, (worlds, 1)), dtype=wp.vec2i, device=device)
    walk.scratch = wp.zeros(worlds * n, dtype=wp.spatial_vector, device=device)
    starts = (np.arange(worlds) * n).astype(np.int32)
    walk.joint_start = wp.array(starts, device=device)
    walk.joint_end = wp.array(starts + n, device=device)
    walk.dof_start = wp.array(first, device=device)
    walk.dofs = wp.full(worlds, n, dtype=wp.int32, device=device)
    release = WalkRelease()
    release.count = wp.zeros(worlds, dtype=wp.int32, device=device)
    release.dof = wp.zeros((1, 1), dtype=wp.int32, device=device)
    release.step = release_steps((1, 1, 1), device)
    walk.release = release
    walk.du = wp.zeros(worlds * dofs, dtype=wp.float32, device=device)
    return walk


def leave_walk(walk: WalkTrees, velocity, device) -> None:
    """The step's one leave of a hand-built walk (:func:`.factor.walk.walk_leave_kernel`), after a sweep launched by
    hand: each tree's change, which the sweep added into ``walk.du``, walked down onto ``velocity``; ``du`` cleared for
    the next launch."""
    joint_start, joint_end = walk.joint_start.numpy(), walk.joint_end.numpy()
    dofs, path = walk.dofs.numpy(), walk.path.numpy()
    classes: dict = {}
    index = []
    for art in range(joint_start.shape[0]):
        rows = path[joint_start[art] : joint_end[art]]
        table = WalkClass(tuple(int(p) for p in rows[:, 0]), (*(int(f) for f in rows[:, 1]), int(dofs[art])))
        index.append(classes.setdefault(table, len(classes)))
    lanes = 4 if wp.get_device(device).is_cuda else 1
    wp.launch(
        walk_leave_kernel(tuple(classes), True),
        dim=len(index) * lanes,
        inputs=[
            wp.array(np.arange(len(index), dtype=np.int32), device=device),
            lanes,
            walk,
            wp.array(np.asarray(index, dtype=np.int32), device=device),
            walk.du,
            walk.dof_start,
            velocity,
        ],
        device=device,
    )
    walk.du.zero_()


def walked_rows(lower, jacobian) -> np.ndarray:
    """The one writer's rows ``z = A^-T J^T`` of Jacobian rows ``[rows, n]`` on :func:`chain_walk`'s factor."""
    A = walk_factor(lower)
    return np.linalg.solve(A.T, np.asarray(jacobian, dtype=np.float64).T).T.astype(np.float32)


def free_body_incident_gaps(solver) -> tuple[np.ndarray, np.ndarray]:
    """:func:`incident_gaps` of the free-body rows: each row's ``|mf_row_incident - Z^T u_hat|`` over its two bodies'
    six coordinates, ``u_hat`` the walk's enter of ``v_hat`` (:func:`_walked_coordinates`), and its terms' magnitude."""
    incident = solver.placement.row_incident.numpy()[:, solver.placement.body_base :].astype(np.float64)
    Z = solver.placement.mf_Z.numpy().astype(np.float64)
    Z_a, Z_b = Z[..., :6], Z[..., 6:]
    # Each row's sides: its responding bodies' articulations (one-node trees), -1 for none.
    sides, kinds = (
        solver.placement.row_sides.numpy()[:, solver.placement.body_base :],
        solver.placement.row_type.numpy()[:, solver.placement.body_base :],
    )
    walked, offsets = _walked_coordinates(solver), solver.forest.articulation_world_dof_offset.numpy()
    gaps, terms = np.zeros(incident.shape), np.zeros(incident.shape)
    for world, count in enumerate(solver.placement.mf_constraint_count.numpy()):
        for row in range(int(count)):
            if kinds[world, row] == row_kernels.PGS_CONSTRAINT_TYPE_SPEED:
                continue  # A speed row carries no Z.
            term = np.zeros(0)
            for art, Z in ((sides[world, row, 0], Z_a), (sides[world, row, 1], Z_b)):
                if art < 0:
                    continue
                # The walk's coordinates, by coordinate (not reversed): the walked rows' frame.
                first = int(offsets[art])
                term = np.concatenate((term, Z[world, row] * walked[world, first : first + 6]))
            gaps[world, row] = abs(incident[world, row] - term.sum())
            terms[world, row] = np.abs(term).sum()
    return gaps, terms


def row_velocity(jacobian, velocity) -> np.ndarray:
    """``J v`` of rows ``[world, row, coordinate]`` over a velocity by global coordinate (each world's coordinates in
    turn), float32, summed in entry order from zero: the rows' incident velocity as their pass forms it."""
    jacobian = np.asarray(jacobian, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32).reshape(jacobian.shape[0], jacobian.shape[2])
    out = np.zeros(jacobian.shape[:2], dtype=np.float32)
    for coordinate in range(jacobian.shape[2]):
        out += jacobian[:, :, coordinate] * velocity[:, coordinate, None]
    return out


def recomputing(solver):
    """``solver``, recomputing its held state from every step's input: the reference a cached step must match within
    rounding (returns the solver)."""
    step = solver.step

    def recompute_then_step(*args, **kwargs):
        solver.factor.notify_state_changed()
        return step(*args, **kwargs)

    solver.step = recompute_then_step
    return solver


def articulation_factor(mass):
    """The leaf-first factor ``L`` of one articulation with mass matrix ``mass`` (``P mass P^T = L L^T``, ``P``
    reversing the coordinates, float64)."""
    mass = np.asarray(mass, dtype=np.float64)
    return np.linalg.cholesky(mass[::-1, ::-1])


def whitened_rows(jacobian, L) -> np.ndarray:
    """The packed whitened rows ``Z = L^-1 P J^T`` of rows ``jacobian`` ``[..., n]`` on one articulation of leaf-first
    factor ``L``, at their coordinates (factor coordinate ``i`` at entry ``n - 1 - i``), float64."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    flat = jacobian.reshape(-1, jacobian.shape[-1])
    z = np.linalg.solve(L, flat[:, ::-1].T).T[:, ::-1]
    return z.reshape(jacobian.shape)


def packed_coordinates(solver) -> np.ndarray:
    """The world coordinate of every packed row entry ``[world, row, entry]`` (-1 past the row's sides), from the
    rows' sides (:mod:`.packed_rows`: dense rows at their coordinates, else side by side). Every side holds response
    coordinates (:func:`.row_records.row_sides`)."""
    sides, sizes = solver.placement.row_sides.numpy(), solver.forest.articulation_response_dof_count.numpy()
    offsets = solver.forest.articulation_world_dof_offset.numpy()
    dof = np.full((*sides.shape[:2], solver.placement.packed_Z.shape[2]), -1, dtype=np.int32)
    for world, count in enumerate(solver.placement.constraint_count.numpy()):
        for row in range(min(int(count), sides.shape[1])):
            first = 0
            for art in sides[world, row]:
                if art < 0:
                    continue
                n, offset = int(sizes[art]), int(offsets[art])
                if n <= 0 or offset < 0:
                    raise AssertionError(f"world {world} row {row}: side {art} holds no response coordinates")
                at = offset if solver.placement.packed_dense else first
                dof[world, row, at : at + n] = offset + np.arange(n)
                first += n
    return dof


def unpacked_rows(solver) -> tuple[np.ndarray, np.ndarray]:
    """The matrix-free sweeps' dense rows over their worlds' coordinates, from the packed rows (:mod:`.packed_rows`),
    float64 ``[world, row, coordinate]``: the Jacobian ``J`` (``inverse^-1 z`` on each side, the side's entries found by
    their coordinates) and the response ``H^-1 J^T`` (``inverse^T z``), through each side's walk (:func:`walk_inverse`)."""
    dof = packed_coordinates(solver)
    Z = solver.placement.packed_Z.numpy().astype(np.float64)
    sides, sizes = solver.placement.row_sides.numpy(), solver.forest.articulation_response_dof_count.numpy()
    offsets = solver.forest.articulation_world_dof_offset.numpy()
    walks = {}
    shape = (*dof.shape[:2], solver.forest.max_world_dofs)
    jacobian, response = np.zeros(shape), np.zeros(shape)
    for world, count in enumerate(solver.placement.constraint_count.numpy()):
        for row in range(min(int(count), dof.shape[1])):
            for art in sides[world, row]:
                if art < 0:
                    continue
                n, first = int(sizes[art]), int(offsets[art])
                held = (dof[world, row] >= first) & (dof[world, row] < first + n)
                if int(art) not in walks:
                    inverse = walk_inverse(solver, int(art))
                    walks[int(art)] = (np.linalg.inv(inverse), inverse.T)
                lower, response_of = walks[int(art)]
                # The walk's footprints sit at their coordinates (not reversed).
                z = np.zeros(n)
                z[dof[world, row, held] - first] = Z[world, row, held]
                jacobian[world, row, first : first + n] = lower @ z
                response[world, row, first : first + n] = response_of @ z
    return jacobian, response


def incident_gaps(solver) -> tuple[np.ndarray, np.ndarray]:
    """BACKBONE's placement invariant ``|J v - Z^T u|`` at the predictor, over the matrix-free sweeps' dense rows:
    each row's ``|row_incident - Z^T u_hat|`` with ``u_hat`` the walk's enter of ``v_hat`` on each side
    (:func:`_walked_coordinates`), and the magnitude of the
    sum's terms it rounds against (:mod:`.solve_policy`), float64 ``[world, row]``, zero past a world's rows."""
    dof = packed_coordinates(solver)
    Z = solver.placement.packed_Z.numpy().astype(np.float64)
    incident = solver.placement.row_incident.numpy().astype(np.float64)
    sides, sizes = solver.placement.row_sides.numpy(), solver.forest.articulation_response_dof_count.numpy()
    offsets = solver.forest.articulation_world_dof_offset.numpy()
    gaps, terms = np.zeros(dof.shape[:2]), np.zeros(dof.shape[:2])
    walked = _walked_coordinates(solver)
    for world, count in enumerate(solver.placement.constraint_count.numpy()):
        for row in range(min(int(count), dof.shape[1])):
            term = np.zeros(0)
            for art in sides[world, row]:
                if art < 0:
                    continue
                n, first = int(sizes[art]), int(offsets[art])
                held = (dof[world, row] >= first) & (dof[world, row] < first + n)
                # The walked rows' frame: each held entry against the walk's coordinate it sits at.
                term = np.concatenate((term, Z[world, row, held] * walked[world, dof[world, row, held]]))
            gaps[world, row] = abs(incident[world, row] - term.sum())
            terms[world, row] = np.abs(term).sum()
    return gaps, terms


def call(kernel, dim, device, **values):
    """Launch ``kernel`` with arguments by name; numpy inputs become arrays of the argument's dtype.

    A struct argument of :data:`STRUCT_ARGUMENTS` is bound from the values by the scenes' names
    (:func:`bind_struct`), and a producer's placement not given is the size groups' rows (:func:`group_rows`).
    Returns every array argument (and every array of such a struct) by name, so a test reads the outputs it needs.
    """
    args, arrays = [], {}
    for var in kernel.adj.args:
        if var.type is RowSupport and var.label not in values:
            # A producer's placement: the size groups' rows the scenes give (group_rows).
            args.append(group_rows(device))
            continue
        if var.type in STRUCT_ARGUMENTS and var.label not in values:
            value, struct_arrays = bind_struct(var.type, values, device)
            arrays.update(struct_arrays)
            args.append(value)
            continue
        value = values[var.label]
        if wp.types.is_array(var.type):
            if not isinstance(value, wp.array):
                value = wp.array(np.asarray(value), dtype=var.type.dtype, device=device)
            arrays[var.label] = value
        args.append(value)
    wp.launch(kernel, dim=dim, inputs=args, device=device)
    return arrays


def normalized(text: str) -> str:
    return " ".join(text.split())


def anchored(test, module_file: str, text: str):
    """Fail unless ``text`` (whitespace aside) is in the package module ``module_file``; return ``text``."""
    source = (PACKAGE / module_file).read_text(encoding="utf-8")
    test.assertIn(normalized(text), normalized(source), f"{module_file} no longer holds the replaced code")
    return text


def indent_block(body: str) -> str:
    import textwrap  # noqa: PLC0415

    return textwrap.indent(textwrap.dedent(body), "    ")


# ---------------------------------------------------------------------------------------------------------------------
# Native (CUDA text) implementations: each extracted from its generator's output, compiled and compared with the law
# ---------------------------------------------------------------------------------------------------------------------

_CTYPES = {float: "float", int: "int", wp.vec2: "wp::vec2", wp.vec3: "wp::vec3"}


def native_kernel(name: str, inputs: dict, outputs: dict, body: str, preludes=()):
    """A kernel that runs native ``body`` per thread: inputs bound by name, outputs stored from locals of their names."""
    namespace = {"wp": wp}
    params = []
    loads = []
    for k, (var, dtype) in enumerate(inputs.items()):
        namespace[f"I_{k}"] = dtype
        params.append(f"{var}_in: wp.array[I_{k}]")
        loads.append(f"    const {_CTYPES[dtype]} {var} = wp::index({var}_in, i);")
    stores = []
    for k, (var, dtype) in enumerate(outputs.items()):
        namespace[f"O_{k}"] = dtype
        params.append(f"{var}_out: wp.array[O_{k}]")
        stores.append(f"    wp::index({var}_out, i) = {var};")
    snippet_body = "\n".join(loads) + "\n" + body + "\n" + "\n".join(stores) + "\n#endif\n"
    namespace["snippet"] = cuda_native(snippet_body, preludes=tuple(preludes))
    signature = ", ".join(params)
    arguments = ", ".join(p.split(":")[0] for p in params)
    source = (
        f"@wp.func_native(snippet)\ndef native(i: int, {signature}): ...\n\n"
        f"def {name}({signature}):\n    native(wp.tid(), {arguments})\n"
    )
    return exec_kernel(name, source, namespace)


def between(text: str, start: str, end: str, occurrence: int = 0) -> str:
    """The text from ``start`` (included) to ``end`` (excluded), at its ``occurrence``-th start."""
    position = -1
    for _ in range(occurrence + 1):
        position = text.index(start, position + 1)
    return text[position : text.index(end, position)]


def joint_type(name: str) -> int:
    """The integer code of a Newton joint type, by name."""
    return int(getattr(JointType, name))


def register_device_tests(cls, namespace: dict):
    """Add every module-level ``test_*(test, device)`` function of ``namespace`` to ``cls``, once per test device."""
    devices = get_test_devices()
    for name, function in list(namespace.items()):
        if name.startswith("test_") and inspect.isfunction(function):
            add_function_test(cls, name, function, devices=devices)
