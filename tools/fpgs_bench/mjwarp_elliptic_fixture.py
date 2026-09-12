# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build genuine CPU elliptic contacts and an independent FP64 test oracle.

This test-only oracle adds no iterations or acceptance rules to the runtime.
"""

from types import SimpleNamespace as NS

import numpy as np
import warp as wp


def array(value, dtype=float):
    """Allocate explicitly on CPU without initializing a CUDA device."""
    return wp.array(np.asarray(value), dtype=dtype, device="cpu")


def cone_point(x, D, friction, mu):
    """Independent FP64 cone cost, full row gradient/Hessian, and active zone."""
    x, D, friction = (
        np.asarray(x, float),
        np.asarray(D, float),
        np.asarray(friction, float),
    )
    f = friction[: len(x) - 1]
    N = mu * x[0]
    weighted = f * x[1:]
    T = np.linalg.norm(weighted)
    if N >= mu * T:
        return 0.0, np.zeros_like(x), np.zeros((len(x), len(x))), "satisfied"
    if mu * N + T <= 0:
        return 0.5 * np.dot(D * x, x), D * x, np.diag(D), "quadratic"
    assert T > 0 and mu > 0
    dm = D[0] / (mu * mu * (1 + mu * mu))
    r = N - mu * T
    t_gradient = f * f * x[1:] / T
    r_gradient = np.r_[mu, -mu * t_gradient]
    r_hessian = np.zeros((len(x), len(x)))
    r_hessian[1:, 1:] = -mu * (np.diag(f * f) - np.outer(t_gradient, t_gradient)) / T
    gradient = dm * r * r_gradient
    hessian = dm * (np.outer(r_gradient, r_gradient) + r * r_hessian)
    return 0.5 * dm * r * r, gradient, hessian, "cone"


class Reference:
    def __init__(
        self,
        mass,
        jacobian,
        D,
        friction,
        mu,
        aref,
        smooth,
        initial_acceleration,
        search,
    ):
        self.M, self.J, self.D = (
            np.asarray(mass, float),
            np.asarray(jacobian, float),
            np.asarray(D, float),
        )
        self.friction, self.mu = np.asarray(friction, float), float(mu)
        self.aref, self.smooth = np.asarray(aref, float), np.asarray(smooth, float)
        self.initial_acceleration, self.search = (
            np.asarray(initial_acceleration, float),
            np.asarray(search, float),
        )

    def full(self, value):
        x = self.J @ value - self.aref
        cost, gradient, hessian, zone = cone_point(x, self.D, self.friction, self.mu)
        cost += 0.5 * value @ self.M @ value - value @ self.smooth
        gradient = self.M @ value - self.smooth + self.J.T @ gradient
        hessian = self.M + self.J.T @ hessian @ self.J
        return cost, gradient, hessian, zone

    def point(self, alpha):
        cost, gradient, hessian, zone = self.full(self.initial_acceleration + alpha * self.search)
        return cost, gradient @ self.search, self.search @ hessian @ self.search, zone

    def root(self):
        """Test oracle only; no extra iterations are added to the actual kernel."""
        low, high = 0.0, 1.0
        assert self.point(low)[1] < 0
        for _ in range(30):
            if self.point(high)[1] >= 0:
                break
            high *= 2
        else:
            raise AssertionError("Reference failed to bracket finite convex minimizer")
        for _ in range(100):
            middle = (low + high) / 2
            if self.point(middle)[1] < 0:
                low = middle
            else:
                high = middle
        return (low + high) / 2


def fixture(condim=3, impratio=10, slip=1.0, budget=50, direction_scale=1.0, sparse=False):
    """Create actual native MuJoCo elliptic contact rows, then a physical Newton ray."""
    import mujoco
    from mujoco_warp._src import types

    xml = f'''<mujoco><option cone="elliptic" jacobian="dense" impratio="{impratio}"/>
      <worldbody><geom name="ground" type="plane" size="2 2 .1" condim="{condim}" friction=".8 .03 .01"/>
      <body pos="0 0 .11"><freejoint/><geom name="sphere" type="sphere" size=".12" mass="1"
      condim="{condim}" friction=".8 .03 .01"/></body></worldbody></mujoco>'''
    cpu_model = mujoco.MjModel.from_xml_string(xml)
    cpu_data = mujoco.MjData(cpu_model)
    cpu_data.qvel[:] = np.array([2, -1, -0.4, 0.2, -0.3, 0.7]) * slip
    mujoco.mj_forward(cpu_model, cpu_data)
    assert cpu_data.ncon == 1 and cpu_data.nefc == condim
    assert int(cpu_data.contact[0].dim) == condim
    assert np.all(cpu_data.efc_type == int(types.ConstraintType.CONTACT_ELLIPTIC))
    assert np.all(cpu_data.efc_id == 0) and cpu_data.contact[0].efc_address == 0
    nv, n = cpu_model.nv, cpu_data.nefc
    mass = np.empty((nv, nv))
    mujoco.mj_fullM(cpu_model, cpu_data, mass)
    jacobian = cpu_data.efc_J.reshape(n, nv)
    # Match the actual single-precision scalar inputs to the Warp kernel.
    D = cpu_data.efc_D.astype(np.float32)
    friction = cpu_data.contact[0].friction.astype(np.float32)
    inverse = np.float32(1 / np.sqrt(impratio))
    mu = np.float32(friction[0] * inverse)
    jacobian, mass = jacobian.astype(np.float32), mass.astype(np.float32)
    aref, smooth = (
        cpu_data.efc_aref.astype(np.float32),
        cpu_data.qfrc_smooth.astype(np.float32),
    )
    initial = np.zeros(nv, np.float32)
    reference = Reference(mass, jacobian, D, friction, mu, aref, smooth, initial, np.zeros(nv))
    _, gradient, hessian, _ = reference.full(initial)
    assert np.linalg.eigvalsh(hessian).min() > 0
    search = (-np.linalg.solve(hessian, gradient) * direction_scale).astype(np.float32)
    reference.search = search.astype(float)
    assert reference.point(0)[1] < 0
    model = NS(
        nv=nv,
        is_sparse=sparse,
        opt=NS(
            ls_iterations=budget,
            cone=types.ConeType.ELLIPTIC,
            solver=types.SolverType.NEWTON,
            tolerance=array([1e-6]),
            ls_tolerance=array([0.01]),
            impratio_invsqrt=array([inverse]),
            warn_overflow=True,
        ),
        stat=NS(meaninertia=array([cpu_model.stat.meaninertia])),
        block_dim=NS(linesearch_iterative=32),
    )
    row_columns = [np.flatnonzero(row) for row in jacobian]
    counts = np.array([len(columns) for columns in row_columns], np.int32)
    offsets = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
    columns = np.concatenate(row_columns).astype(np.int32)
    sparse_values = np.concatenate([row[cols] for row, cols in zip(jacobian, row_columns, strict=True)])
    data = NS(
        nworld=1,
        njmax=n,
        ne=array([0], int),
        nf=array([0], int),
        nefc=array([n], int),
        qfrc_smooth=array(smooth[None]),
        nacon=array([1], int),
        solver_niter=array([0], int),
        qacc=array(initial[None]),
        overflow=array([0], int),
        contact=NS(
            friction=array(friction[None], types.vec5),
            dim=array([condim], int),
            efc_address=array(np.arange(max(6, n))[None], int),
        ),
        efc=NS(
            type=array(cpu_data.efc_type[None], int),
            id=array(cpu_data.efc_id[None], int),
            J_rownnz=array(counts[None], int),
            J_rowadr=array(offsets[None], int),
            J_colind=array(columns[None, None], int),
            J=array(sparse_values[None, None] if sparse else jacobian[None]),
            D=array(D[None]),
            frictionloss=array(np.zeros((1, n))),
            Ma=array((mass @ initial)[None]),
        ),
    )
    ctx = NS(
        search_unchanged=array([False], bool),
        Jaref=array((jacobian @ initial - aref)[None]),
        search=array(search[None]),
        search_dot=array([np.dot(search, search)]),
        mv=array((mass @ search)[None]),
        jv=array((jacobian @ search)[None]),
        quad=wp.zeros((1, n), dtype=wp.vec3, device="cpu"),
        done=array([False], bool),
        improvement=array([0]),
        alpha=array([0]),
        ls_exhausted=array([False], bool),
    )
    return model, data, ctx, reference, cpu_model, cpu_data


def run_case(driver, condim=3, impratio=10, slip=1.0, *, budget=50, fuse_jv=False, sparse=False):
    """Evaluate the actual selected driver against the independent raw-row law."""
    model, data, ctx, reference, _, _ = fixture(condim, impratio, slip, budget, sparse=sparse)
    driver(model, data, ctx, fuse_jv)
    alpha = float(ctx.alpha.numpy()[0])
    start, point, optimum = reference.point(0), reference.point(alpha), reference.point(reference.root())
    scale = max(1.0, abs(start[0]), abs(optimum[0]), abs(start[1]))
    return {
        "actual_elliptic_rows": int(data.nefc.numpy()[0]),
        "scaled_cost_regret": (point[0] - optimum[0]) / scale,
        "scaled_gradient": abs(point[1]) / scale,
        "cost_change": point[0] - start[0],
        "overflow": int(data.overflow.numpy()[0]),
        "exhausted": bool(ctx.ls_exhausted.numpy()[0]),
        "finite": bool(
            np.isfinite(data.qacc.numpy()).all()
            and np.isfinite(ctx.quad.numpy()).all()
            and np.isfinite([alpha, *start[:3], *point[:3], *optimum[:3]]).all()
        ),
    }
