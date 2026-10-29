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
from newton._src.solvers.feather_pgs import contact_laws, field_group, kernels, law_source, projection_laws, row_laws
from newton._src.solvers.feather_pgs.contact_input import ContactInput
from newton._src.solvers.feather_pgs.dense_rows import DenseRows, MfRows
from newton._src.solvers.feather_pgs.device_preludes import cuda_native, dual_native
from newton._src.solvers.feather_pgs.friction_patches import PatchRowLaw
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.solver_fields import Mimics
from newton.tests.unittest_utils import add_function_test, get_test_devices

FAMILIES = (contact_laws.CONTACT_LAW, row_laws.ROW_LAW, projection_laws.PROJECTION_LAW)
PACKAGE = Path(kernels.__file__).parent


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
        if wp.types.is_array(annotation):
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
    flags = lambda: rng.integers(0, 2, n).astype(np.int32)  # noqa: E731
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
    if name == "contact_friction_points":
        return {"point_a": vec(), "point_b": vec(), "shared_anchor": flags(), "friction_shared_anchor": flags()}
    if name == "contact_row_frame":
        return {
            "row": ints(0, 3),
            "shared_anchor": flags(),
            "normal": unit_vectors(rng, n),
            "tangent0": unit_vectors(rng, n),
            "tangent1": unit_vectors(rng, n),
            "point_a": vec(),
            "point_b": vec(),
            "friction_a": vec(),
            "friction_b": vec(),
        }
    if name == "jacobian_entry":
        return {
            "motion": rng.normal(size=(n, 6)).astype(np.float32),
            "direction": unit_vectors(rng, n),
            "point": vec(3.0),
            "origin": vec(3.0),
        }
    if name == "free_body_jacobian":
        return {"direction": unit_vectors(rng, n), "point": vec(3.0), "origin": vec(3.0)}
    if name == "prescribed_target":
        return {
            **common,
            "point_a": vec(),
            "point_b": vec(),
            "direction": unit_vectors(rng, n),
            "prescribed_articulation": np.array([1, 0, 1, 1], dtype=np.int32),
            "articulation_origin": rng.normal(size=(4, 3)).astype(np.float32),
            "body_v_s": rng.normal(size=(8, 6)).astype(np.float32),
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
    if name == "speed_row":
        return {
            "speed": np.abs(scalars(rng, n, 3.0)),
            "limit": rng.uniform(0.1, 3.0, n).astype(np.float32),
            "effective_mass": rng.uniform(0.1, 2.0, n).astype(np.float32),
            "old": np.abs(scalars(rng, n)),
            "accumulated": flags(),
        }
    if name in ("velocity_limit_update",):
        return {
            "old": np.abs(scalars(rng, n)),
            "residual": scalars(rng, n),
            "step": scalars(rng, n),
            "accumulated": flags(),
        }
    if name == "velocity_clamp_impulse":
        return {
            "jv": scalars(rng, n, 3.0),
            "limit": rng.uniform(0.1, 3.0, n).astype(np.float32),
            "diagonal": rng.uniform(0.1, 2.0, n).astype(np.float32),
        }
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
        elif param in ("iteration", "start_iteration"):
            values[param] = ints(0, 6)
        elif param == "accumulated":
            values[param] = flags()
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
    "incident": "v_hat",
    "shared_anchor": "contact_shared_anchor",
    "friction_shared_anchor": "contact_friction_shared_anchor",
    "friction_scale": "contact_friction_scale",
    "beta": "pgs_beta",
    "cfm": "pgs_cfm",
    "row_type": "world_row_type",
    "row_parent": "world_row_parent",
    "row_mu": "world_row_mu",
    "row_beta": "world_row_beta",
    "row_cfm": "world_row_cfm",
    "phi": "world_phi",
    "target": "world_target_velocity",
    "restitution": "world_row_restitution",
}
"""The scenes' names of the :data:`ContactInput` fields: the reference laws take them as separate arguments."""


DENSE_ROW_ARGUMENTS = {
    "row_type": "world_row_type",
    "row_parent": "world_row_parent",
    "row_mu": "world_row_mu",
    "row_beta": "world_row_beta",
    "row_cfm": "world_row_cfm",
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

MF_ROW_ARGUMENTS = {field.name: field.attr for field in field_group.fields_of(MfRows)}
"""The scenes' names of the :data:`MfRows` fields: the solver's attributes."""

MIMIC_ARGUMENTS = {field.name: f"mimic_{field.name}" for field in field_group.fields_of(Mimics)}
"""The scenes' names of the :data:`Mimics` fields: ``mimic_`` and the field's name."""

STRUCT_ARGUMENTS = {
    ContactInput: CONTACT_ARGUMENTS,
    DenseRows: DENSE_ROW_ARGUMENTS,
    JointTable: JOINT_TABLE_ARGUMENTS,
    MfRows: MF_ROW_ARGUMENTS,
    Mimics: MIMIC_ARGUMENTS,
    PatchRowLaw: {"patches": "friction_patches", "beta": "friction_anchor_beta"},
}
"""The scenes' field names of each struct a kernel takes in place of separate arguments."""


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
    instance = field_group.upload(struct_type, device, **bound)
    return instance, {
        arguments[name]: value for name, value in bound.items() if isinstance(value, wp.array) and name in arguments
    }


def contact_input(values, device):
    """The :data:`ContactInput` of a scene's values (:func:`bind_struct`)."""
    return bind_struct(ContactInput, values, device)


def position_pass(device) -> list:
    """The trailing operands of ``kernels.compute_world_contact_bias``'s position pass: no velocity pass (0), whose
    world coordinates, packed rows, velocities and restitution threshold it then never reads."""
    return [
        0,
        wp.zeros((1, 1), dtype=wp.int32, device=device),
        wp.full((1, 1, 1), -1, dtype=wp.int32, device=device),
        wp.zeros((1, 1, 1), dtype=wp.float32, device=device),
        wp.zeros(1, dtype=wp.float32, device=device),
        wp.zeros(1, dtype=wp.float32, device=device),
        0.0,
    ]


def packed_world_rows(world_jacobian, device) -> list:
    """Packed rows (:mod:`.packed_rows`) of a dense world-layout Jacobian ``[world, row, coordinate]``: every entry at its
    own coordinate, as ``(packed_dof, packed_J)``."""
    world_jacobian = np.asarray(world_jacobian, dtype=np.float32)
    dof = np.broadcast_to(np.arange(world_jacobian.shape[2], dtype=np.int32), world_jacobian.shape).copy()
    return [wp.array(dof, dtype=wp.int32, device=device), wp.array(world_jacobian, dtype=wp.float32, device=device)]


_FACTOR_STORAGE = []
"""The factors :func:`articulation_factor` addresses, kept alive while the tests run."""


def articulation_factor(mass, device):
    """The :class:`.kernels.BodyFactors` of one articulation on global coordinates ``0..n-1`` with mass matrix
    ``mass``, and its leaf-first factor ``L`` (``P mass P^T = L L^T``, ``P`` reversing the coordinates, float64)."""
    mass = np.asarray(mass, dtype=np.float64)
    n = mass.shape[0]
    L = np.linalg.cholesky(mass[::-1, ::-1])
    storage = wp.array(L.astype(np.float32), dtype=wp.float32, device=device)
    _FACTOR_STORAGE.append(storage)
    factors = kernels.BodyFactors()
    factors.first_art = wp.array([0] + [-1] * (n - 1), dtype=wp.int32, device=device)
    factors.size = wp.array([n], dtype=wp.int32, device=device)
    factors.world_offset = wp.array([0], dtype=wp.int32, device=device)
    factors.address = wp.array([storage.ptr], dtype=wp.uint64, device=device)
    return factors, L


def whitened_rows(jacobian, L) -> np.ndarray:
    """The packed whitened rows ``Z = L^-1 P J^T`` of rows ``jacobian`` ``[..., n]`` on one articulation of leaf-first
    factor ``L``, at their coordinates (factor coordinate ``i`` at entry ``n - 1 - i``), float64."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    flat = jacobian.reshape(-1, jacobian.shape[-1])
    z = np.linalg.solve(L, flat[:, ::-1].T).T[:, ::-1]
    return z.reshape(jacobian.shape)


def unpacked_rows(solver) -> tuple[np.ndarray, np.ndarray]:
    """The matrix-free sweeps' dense rows over their worlds' coordinates, from the packed rows (:mod:`.packed_rows`),
    float64 ``[world, row, coordinate]``: the Jacobian ``J`` and the response ``H^-1 J^T`` (``P^T L^-T z`` on each
    side, the side's entries found by their coordinates)."""
    dof = solver._packed_dof.numpy()
    J, Z = solver._packed_J.numpy().astype(np.float64), solver._packed_Z.numpy().astype(np.float64)
    sides, sizes = solver._row_sides.numpy(), solver.articulation_response_dof_count.numpy()
    groups, offsets = solver.art_group_idx.numpy(), solver.articulation_world_dof_offset.numpy()
    factors = {}
    shape = (*dof.shape[:2], solver.max_world_dofs)
    jacobian, response = np.zeros(shape), np.zeros(shape)
    for world, count in enumerate(solver.constraint_count.numpy()):
        for row in range(min(int(count), dof.shape[1])):
            for art in sides[world, row]:
                if art < 0:
                    continue
                n, first = int(sizes[art]), int(offsets[art])
                if n == 0:
                    continue  # No response coordinates (a prescribed articulation): no entries.
                held = (dof[world, row] >= first) & (dof[world, row] < first + n)
                coordinates = dof[world, row, held]
                jacobian[world, row, coordinates] = J[world, row, held]
                if n not in factors:
                    factors[n] = solver.L_by_size[n].numpy().astype(np.float64)
                L = np.tril(factors[n][groups[art]])
                # Leaf-first positions: factor coordinate i is the side's coordinate n - 1 - i.
                response[world, row, coordinates] = np.linalg.solve(L.T, Z[world, row, held][::-1])[::-1]
    return jacobian, response


def world_restitution_layout(device) -> list:
    """The sparse-diagonal operands of ``kernels.apply_world_contact_restitution_matrix_free`` after ``packed_J``,
    for a packed row (``sparse_diagonal`` 0): never read."""
    return [
        0,
        wp.zeros(1, dtype=wp.int32, device=device),
        wp.zeros(1, dtype=wp.int32, device=device),
        0,
        wp.zeros((1, 1, 1), dtype=wp.float32, device=device),
        wp.full((1, 1, 2), -1, dtype=wp.int32, device=device),
        wp.zeros((1, 1, 4), dtype=wp.float32, device=device),
    ]


def call(kernel, dim, device, **values):
    """Launch ``kernel`` with arguments by name; numpy inputs become arrays of the argument's dtype.

    A struct argument of :data:`STRUCT_ARGUMENTS` is bound from the values by the scenes' names
    (:func:`bind_struct`). Returns every array argument (and every array of such a struct) by name, so a test
    reads the outputs it needs.
    """
    args, arrays = [], {}
    for var in kernel.adj.args:
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


def reference_kernel(name: str, signature: str, body: str, namespace: dict, module_options=None):
    """A Warp kernel whose body is the replaced code, with its data accesses renamed to kernel arguments."""
    source = f"def {name}({signature}):\n{indent_block(body)}"
    kernel = exec_kernel(name, source, {"wp": wp, **namespace})
    if module_options:
        kernel.module.options.update(module_options)
    return kernel


def dedent_block(body: str) -> str:
    import textwrap  # noqa: PLC0415

    return textwrap.dedent(body).lstrip("\n")


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


def run_native(kernel, inputs: dict, outputs: dict, values: dict, n: int, device):
    args = {f"{var}_in": values[var] for var in inputs}
    args.update(
        {
            f"{var}_out": np.zeros(
                (n, 2) if dtype is wp.vec2 else ((n, 3) if dtype is wp.vec3 else n),
                np.float32 if dtype is not int else np.int32,
            )
            for var, dtype in outputs.items()
        }
    )
    out = call(kernel, n, device, **args)
    return {var: out[f"{var}_out"].numpy() for var in outputs}


def between(text: str, start: str, end: str, occurrence: int = 0) -> str:
    """The text from ``start`` (included) to ``end`` (excluded), at its ``occurrence``-th start."""
    position = -1
    for _ in range(occurrence + 1):
        position = text.index(start, position + 1)
    return text[position : text.index(end, position)]


def require_cuda(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("native implementations compile for CUDA only")


def joint_type(name: str) -> int:
    """The integer code of a Newton joint type, by name."""
    return int(getattr(JointType, name))


def register_device_tests(cls, namespace: dict):
    """Add every module-level ``test_*(test, device)`` function of ``namespace`` to ``cls``, once per test device."""
    devices = get_test_devices()
    for name, function in list(namespace.items()):
        if name.startswith("test_") and inspect.isfunction(function):
            add_function_test(cls, name, function, devices=devices)
