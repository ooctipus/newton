# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Topology-generated keyed contact rows: a per-world row prefix and direct row ingestion.

The layout (primary tree, secondary free body, parallel row count) comes from
``keyed_row_layout.topology_plan``; nothing here names a robot or a size.

Under persistent patch friction the contact keys are scattered with each row's
type, parent and undivided friction coefficient before the patch link
(``_FrictionPatchState.link``) links the normal rows into their patch rings and
divides the tangent coefficients. The row sources then write the patch row law
(tangent rows at the patch friction location with the anchor's phi and
``friction_anchor_beta``) and read parents and coefficients instead of writing
them, so the links survive until the sweep and the patch finish read them.
"""

from functools import cache, partial

import warp as wp

from ...sim import ModelFlags
from .contact_input import ContactInput, bind
from .friction_patches import PatchRowLaw
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .keyed_row_layout import prefix_kernel
from .keyed_row_layout import topology_plan as direct_topology_plan
from .row_inputs import ParallelRowInputs, RowArgument


@wp.struct
class RowMaps:
    groups: wp.array2d[int]
    starts: wp.array2d[int]


@wp.struct
class PrefixInput:
    starts: wp.array2d[int]
    q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    velocity_limit: wp.array[float]
    q: wp.array[float]
    vhat: wp.array[float]
    counts: wp.array[int]
    bounds: wp.array2d[int]
    velocity_slots: wp.array[int]
    velocity_signs: wp.array[float]
    kind: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    beta: wp.array2d[float]
    cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]
    gap: float
    activation: float
    pgs_beta: float
    pgs_cfm: float
    dt: float


@wp.struct
class KineticRows:
    rowkeys: wp.array2d[int]
    maps: RowMaps
    prefix: PrefixInput
    contact: ContactInput


@wp.kernel(enable_backward=False)
def scatter_contact_keys(data: ContactInput, z: KineticRows):
    """Emit only the original allocated raw contact ID and row direction."""
    contact = wp.tid()
    if contact < wp.min(data.count[0], data.point0.shape[0]):
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            for direction in range(data.slots_needed[contact]):
                z.rowkeys[data.world[contact], data.slot[contact] + direction] = contact * 4 + direction


@wp.kernel(enable_backward=False)
def scatter_patch_contact_keys(data: ContactInput, z: KineticRows):
    """Emit the contact keys with the type, parent and undivided coefficient of each patch row.

    The patch link runs after this launch and before any row is read: it links
    each normal row into its patch ring and divides the tangent coefficient
    among the anchors that received rows.
    """
    contact = wp.tid()
    if contact < wp.min(data.count[0], data.point0.shape[0]):
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            world = data.world[contact]
            slot = data.slot[contact]
            mu = float(0.0)
            materials = int(0)
            if data.shape0[contact] >= 0:
                mu += data.material_mu[data.shape0[contact]]
                materials += 1
            if data.shape1[contact] >= 0:
                mu += data.material_mu[data.shape1[contact]]
                materials += 1
            if materials > 0:
                mu /= float(materials)
            for direction in range(data.slots_needed[contact]):
                row = slot + direction
                z.rowkeys[world, row] = contact * 4 + direction
                if direction == 0:
                    data.row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
                    data.row_parent[world, row] = -1
                    data.row_mu[world, row] = mu
                else:
                    data.row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
                    data.row_parent[world, row] = slot
                    data.row_mu[world, row] = mu * data.friction_scale


def _patch_row_law(source):
    """Turn a point row source into the patch row law, reading the linked parents and coefficients.

    Tangent rows act at the patch friction location for both the Jacobian and
    the prescribed target, carry the anchor's tangential phi, and correct it
    with ``patch_law.beta``. The type, parent and coefficient of every contact
    row stay as :func:`scatter_patch_contact_keys` and the patch link left them.
    """
    edits = (
        (
            "            const wp::vec3 anchor = 0.5f * (pa + pb); pa = anchor; pb = anchor;\n        }\n",
            "            const wp::vec3 anchor = 0.5f * (pa + pb); pa = anchor; pb = anchor;\n        }\n"
            "        if (r > 0) { pa = patch_law.patches.point_a.data[contact]; "
            "pb = patch_law.patches.point_b.data[contact]; }\n",
        ),
        (
            "const float phi = r == 0 ? separation : 0.0f;",
            "const float phi = r == 0 ? separation : patch_law.patches.phi.data[contact][r - 1];",
        ),
        (
            "        c.row_type.data[at] = r == 0 ? 0 : 2;\n"
            "        c.row_parent.data[at] = r == 0 ? -1 : c.slot.data[contact];\n"
            "        c.row_mu.data[at] = r == 0 ? mu : friction;\n"
            "        c.row_beta.data[at] = r == 0 ? c.beta : 0.0f;\n",
            "        if (r > 0) rhs += c.bias_scale * patch_law.beta * phi / c.dt;\n"
            "        c.row_beta.data[at] = r == 0 ? c.beta : patch_law.beta;\n",
        ),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("the keyed row source changed; update the patch row law")
        source = source.replace(old, new)
    return source


def _row_source(*, layout, output="Jr[q]", private_metadata=False, patch=False):
    """Share row geometry and metadata between direct ingestion and the fallback.

    ``patch`` writes the persistent patch row law and reads the linked parents
    and coefficients (see :func:`_patch_row_law`).
    """
    dofs = layout.dofs
    prefix = """
        const int token = ~key, d = token / 2;
        const float sign = (token & 1) == 0 ? 1.0f : -1.0f;
        #pragma unroll
        for (int q = 0; q < __DOFS__; ++q) {
            __PREFIX_VALUE__
        }
    """
    prefix = prefix.replace("__PREFIX_VALUE__", "Jr[q] = q == d ? sign : 0.0f;")
    physical_projection = f"""
            const int n = c.response_dofs.data[art], local = {layout.local("q")};
            if (n != ({layout.primary_test("q")} ? {layout.primary} : {layout.secondary})
                || (c.body_mask.data[body] & (1u << local)) == 0u) continue;
            const wp::spatial_vector S = c.motion.data[c.dof_start.data[art] + local];
            const wp::vec3 linear(S[0], S[1], S[2]), angular(S[3], S[4], S[5]);
            value += sign * wp::dot(direction, linear + wp::cross(angular, point - c.origin.data[art]));
    """
    projection = """
        #pragma unroll
        for (int q = 0; q < __DOFS__; ++q) {
            float value = 0.0f;
            for (int side = 0; side < 2; ++side) {
                const int body = side == 0 ? ba : bb, art = side == 0 ? aa : ab;
                if (body < 0 || art < 0) continue;
                const float sign = side == 0 ? 1.0f : -1.0f;
                const wp::vec3 point = side == 0 ? pa : pb;
                __PROJECT__
            }
            Jr[q] = value;
        }
        """.replace("__PROJECT__", physical_projection)
    incident = f"relative += Jr[q] * c.incident.data[z.maps.starts.data[world * 2 + ({layout.primary_test('q')} ? 0 : 1)] + {layout.local('q')}];"
    if not private_metadata:
        metadata = ""
    elif patch:
        metadata = "built_rhs = rhs;"
    else:
        metadata = (
            "built_rhs = rhs; built_mu = r == 0 ? mu : friction; built_parent = r == 0 ? -1 : c.slot.data[contact];"
        )
    source = (
        r"""
    const int at = world * z.rowkeys.shape[1] + row;
    const int key = z.rowkeys.data[at];
    if (key < 0) {
__PREFIX__
    } else {
        const auto& c = z.contact;
        const int contact = key >> 2, r = key & 3;
        const int sa = c.shape0.data[contact], sb = c.shape1.data[contact];
        const int ba = sa >= 0 ? c.shape_body.data[sa] : -1;
        const int bb = sb >= 0 ? c.shape_body.data[sb] : -1;
        const int aa = c.art0.data[contact], ab = c.art1.data[contact];
        const wp::vec3 normal = -c.normal.data[contact];
        wp::vec3 pa = ba >= 0 ? wp::transform_point(c.body_q.data[ba], c.point0.data[contact]) : c.point0.data[contact];
        wp::vec3 pb = bb >= 0 ? wp::transform_point(c.body_q.data[bb], c.point1.data[contact]) : c.point1.data[contact];
        pa = pa - c.margin0.data[contact] * normal;
        pb = pb + c.margin1.data[contact] * normal;
        const float separation = wp::dot(normal, pa - pb);
        wp::vec3 direction = normal;
        if (r > 0) {
            wp::vec3 t0 = wp::cross(normal, wp::vec3(1.0f, 0.0f, 0.0f));
            if (wp::length_sq(t0) < 1.0e-12f) t0 = wp::cross(normal, wp::vec3(0.0f, 1.0f, 0.0f));
            t0 = wp::normalize(t0);
            direction = r == 1 ? t0 : wp::normalize(wp::cross(normal, t0));
        }
        if (c.shared_anchor != 0 || (r > 0 && c.friction_shared_anchor != 0)) {
            const wp::vec3 anchor = 0.5f * (pa + pb); pa = anchor; pb = anchor;
        }
        float known = 0.0f;
        for (int side = 0; side < 2; ++side) {
            const int body = side == 0 ? ba : bb, art = side == 0 ? aa : ab;
            if (body >= 0 && art >= 0 && c.prescribed.data[art] != 0) {
                const wp::spatial_vector v = c.body_v.data[body];
                const wp::vec3 linear(v[0], v[1], v[2]), angular(v[3], v[4], v[5]);
                const wp::vec3 p = side == 0 ? pa : pb;
                const float value = wp::dot(direction, linear + wp::cross(angular, p - c.origin.data[art]));
                known += side == 0 ? value : -value;
            }
        }
        const float target = -known;
        float mu = 0.0f, restitution = 0.0f; int materials = 0;
        for (int side = 0; side < 2; ++side) {
            const int shape = side == 0 ? sa : sb;
            if (shape >= 0) {
                ++materials; mu += c.material_mu.data[shape];
                const float rest = c.material_restitution.data[shape];
                if (isfinite(rest)) restitution += rest < 0.0f ? 0.0f : (rest > 1.0f ? 1.0f : rest);
            }
        }
        if (materials) { mu /= static_cast<float>(materials); restitution /= static_cast<float>(materials); }
        float friction = mu * c.friction_scale;
        const bool filter = c.friction_pairs_only == 0 || (aa >= 0 && ab >= 0 && c.is_free.data[aa] == 0 && c.is_free.data[ab] == 0);
        if (r > 0 && filter && c.friction_anchor_limit > 0) {
            const int total = wp::min(c.count.data[0], c.point0.shape[0]);
            int rank = 0;
            for (int look = 1; look <= 8; ++look) {
                const int prev = contact - look;
                if (prev < 0 || prev >= total) break;
                if (c.shape0.data[prev] != sa || c.shape1.data[prev] != sb) break;
                ++rank;
            }
            const int next = contact + 1;
            if (rank > 0 || (next < total && c.shape0.data[next] == sa && c.shape1.data[next] == sb)) friction *= 0.5f;
        }
__PROJECTION__
        const float phi = r == 0 ? separation : 0.0f;
        float rhs = -target;
        if (r == 0) {
            rhs += (phi <= 0.0f ? c.bias_scale * c.beta : c.speculative_scale) * phi / c.dt;
            if (restitution > 0.0f) {
                float relative = -target;
                #pragma unroll
                for (int q = 0; q < __DOFS__; ++q) { __INCIDENT__ }
                if (relative < -c.restitution_threshold && (phi <= 1.0e-6f || phi + c.dt * relative <= 1.0e-6f))
                    rhs = -target + restitution * relative;
            }
        }
        c.row_type.data[at] = r == 0 ? 0 : 2;
        c.row_parent.data[at] = r == 0 ? -1 : c.slot.data[contact];
        c.row_mu.data[at] = r == 0 ? mu : friction;
        c.row_beta.data[at] = r == 0 ? c.beta : 0.0f;
        c.row_cfm.data[at] = c.cfm;
        c.phi.data[at] = phi; c.target.data[at] = target;
        c.restitution.data[at] = r == 0 ? restitution : 0.0f;
        c.rhs.data[at] = rhs; c.diag.data[at] = c.cfm;
        __PRIVATE_METADATA__
    }
""".replace("__PREFIX__", prefix)
        .replace("__PROJECTION__", projection)
        .replace("__INCIDENT__", incident)
        .replace("__PRIVATE_METADATA__", metadata)
        .replace("__DOFS__", str(dofs))
        .replace("Jr[q]", output)
    )
    return _patch_row_law(source) if patch else source


def _fallback_source(layout, *, patch=False):
    primary = layout.primary
    secondary = layout.secondary
    dofs = primary + secondary
    limit = layout.parallel_rows
    primary_test = layout.primary_test("q")
    local_a = f"q - {layout.primary_offset}"
    local_b = f"q - {layout.secondary_offset}"
    source = (
        """
    if (counts.data[world] <= $LIMIT && mf_counts.data[world] == 0) return;
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x, stride = blockDim.x;
#else
    const int lane = 0, stride = 1;
#endif
    const int ga = z.maps.groups.data[world * 2], gb = z.maps.groups.data[world * 2 + 1];
    for (int row = lane; row < counts.data[world]; row += stride) {
        float Jr[$DOFS];
"""
        + _row_source(layout=layout, patch=patch)
        + """
        #pragma unroll
        for (int q = 0; q < $DOFS; ++q) {
            if ($PRIMARY_TEST) J16.data[(ga * J16.shape[1] + row) * $NA + $LOCAL_A] = Jr[q];
            else J6.data[(gb * J6.shape[1] + row) * $NB + $LOCAL_B] = Jr[q];
        }
    }
"""
    )

    for key, value in {
        "$LIMIT": limit,
        "$DOFS": dofs,
        "$PRIMARY_TEST": primary_test,
        "$NA": primary,
        "$NB": secondary,
        "$LOCAL_A": local_a,
        "$LOCAL_B": local_b,
    }.items():
        source = source.replace(key, str(value))
    return source


@cache
def get_fallback_kernel(layout):
    """Materialize current physical rows in one pass over fallback worlds only."""
    source = _fallback_source(layout)

    @wp.func_native(source)
    def native(
        world: int,
        logical_lane: int,
        z: KineticRows,
        counts: wp.array[int],
        mf_counts: wp.array[int],
        J16: wp.array3d[float],
        J6: wp.array3d[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def keyed_rows_fallback(
        z: KineticRows, counts: wp.array[int], mf_counts: wp.array[int], J16: wp.array3d[float], J6: wp.array3d[float]
    ):
        world, lane = wp.tid()
        native(world, lane, z, counts, mf_counts, J16, J6)

    return keyed_rows_fallback


@cache
def get_patch_fallback_kernel(layout):
    """Materialize the fallback worlds' rows with the persistent patch row law."""
    source = _fallback_source(layout, patch=True)

    @wp.func_native(source)
    def native(
        world: int,
        logical_lane: int,
        z: KineticRows,
        patch_law: PatchRowLaw,
        counts: wp.array[int],
        mf_counts: wp.array[int],
        J16: wp.array3d[float],
        J6: wp.array3d[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def keyed_rows_patch_fallback(
        z: KineticRows,
        patch_law: PatchRowLaw,
        counts: wp.array[int],
        mf_counts: wp.array[int],
        J16: wp.array3d[float],
        J6: wp.array3d[float],
    ):
        world, lane = wp.tid()
        native(world, lane, z, patch_law, counts, mf_counts, J16, J6)

    return keyed_rows_patch_fallback


def get_direct_parallel_inputs(layout, na, nb, oa, ob, dofs, capacity, *, patch=False):
    """Decode physical endpoints in the ordinary factor stage.

    ``patch`` writes the persistent patch row law and reads every row's parent and
    coefficient as the scatter and the patch link left them.
    """
    if (na, nb, oa, ob, dofs) != (
        layout.primary,
        layout.secondary,
        layout.primary_offset,
        layout.secondary_offset,
        layout.dofs,
    ):
        raise ValueError("physical row layout and parallel operator disagree")
    key = "kinetic_rows.rowkeys.data[off_dense + i]"
    if patch:
        reader = (
            """
            const int row = i;
            const auto& z = kinetic_rows;
            float built_rhs = 0.0f;
    """
            + _row_source(layout=layout, output="Zi[q * ZS]", private_metadata=True, patch=True)
            + f"""
            if ({key} >= 0) s_rhs[i] = built_rhs;
    """
        )
        return ParallelRowInputs(
            stage=None,
            physical_row=reader,
            arguments=(
                RowArgument("kinetic_rows", KineticRows, after="world_row_cfm"),
                RowArgument("patch_law", PatchRowLaw, after="kinetic_rows"),
            ),
            row_type=f"({key} >= 0 ? (({key} & 3) == 0 ? 0 : 2) : world_row_type.data[off_dense + i])",
            rhs=f"{key} < 0 ? rhs_bias.data[off_dense + i] : 0",
            parent="world_row_parent.data[off_dense + i]",
            mu="world_row_mu.data[off_dense + i]",
        )
    reader = (
        """
            const int row = i;
            const auto& z = kinetic_rows;
            float built_rhs = 0.0f, built_mu = 0.0f;
            int built_parent = -1;
    """
        + _row_source(layout=layout, output="Zi[q * ZS]", private_metadata=True)
        + f"""
            if ({key} >= 0) {{
                s_rhs[i] = built_rhs; s_parent[i] = built_parent; s_mu[i] = built_mu;
            }}
    """
    )
    return ParallelRowInputs(
        stage=None,
        physical_row=reader,
        arguments=(RowArgument("kinetic_rows", KineticRows, after="world_row_cfm"),),
        row_type=f"({key} >= 0 ? (({key} & 3) == 0 ? 0 : 2) : world_row_type.data[off_dense + i])",
        rhs=f"{key} < 0 ? rhs_bias.data[off_dense + i] : 0",
        parent=f"{key} < 0 ? world_row_parent.data[off_dense + i] : 0",
        mu=f"{key} < 0 ? world_row_mu.data[off_dense + i] : 0",
    )


@cache
def get_parallel_factory(original, layout):
    """Use the shared parallel policy with the layout's physical row operands.

    ``linked_rows`` selects the persistent patch row law and the policy's
    ring-pooled friction projection together.
    """

    def factory(*args, linked_rows=False, **kwargs):
        inputs = partial(get_direct_parallel_inputs, layout, patch=linked_rows)
        return original(*args, row_inputs=inputs, linked_rows=linked_rows, **kwargs)

    return factory


def _signature(solver):
    names = (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_qd_start",
        "joint_q_start",
        "joint_dof_dim",
        "articulation_start",
        "body_flags",
        "body_world",
    )
    return tuple(getattr(solver.model, name).numpy().tobytes() for name in names)


class KeyedRows:
    """Own the per-world row prefix, direct keyed rows and the complete fallback."""

    def __init__(self, solver):
        self.solver = solver
        self.keyed_rows = True
        self.layout, groups, starts = direct_topology_plan(solver)
        self.prefix_layout = self.layout
        dev, worlds, capacity = solver.model.device, solver.world_count, solver.dense_max_constraints
        self.data, self.maps, self.prefix = KineticRows(), RowMaps(), PrefixInput()
        self.data.rowkeys = wp.empty((worlds, capacity), dtype=int, device=dev)
        self.maps.groups = wp.array(groups, dtype=int, device=dev)
        self.maps.starts = wp.array(starts, dtype=int, device=dev)
        self.contact_input = None
        self._dummy_int = wp.zeros(1, dtype=int, device=dev)
        self._dummy_float = wp.zeros(1, dtype=float, device=dev)
        self._dummy_factor = wp.zeros((1, 1, 1), dtype=float, device=dev)
        self._signature = _signature(solver)
        # Persistent patch friction: the rows carry the patch law, and the patch link owns parents and coefficients.
        self.patch = bool(solver._friction_anchors_enabled)
        self.law = None
        if self.patch:
            self.law = PatchRowLaw()
            self.law.patches = solver._friction_patches.view
            self.law.beta = solver.friction_anchor_beta

    def validate_notification(self, flags=ModelFlags.ALL):
        """Reject static-plan changes before retiring any canonical producer."""
        numeric = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if flags and not (int(flags) & ~int(numeric)):
            return
        if _signature(self.solver) != self._signature:
            raise RuntimeError("Keyed row topology changed; reconstruct the solver before notification")
        if direct_topology_plan(self.solver)[0] != self.prefix_layout:
            raise RuntimeError("Keyed row topology changed; reconstruct the solver")

    def begin_rows(self, state_in, state_aug, dt):
        """Read current motion and predictor with the current held-factor buffers."""
        s, m, p = self.solver, self.maps, self.prefix
        p.starts, p.q, p.vhat = m.starts, state_in.joint_q, s.v_hat
        for name, source in (
            ("q_index", "_joint_limit_q_index"),
            ("counts", "slot_counter"),
            ("bounds", "dense_phase_bounds"),
            ("velocity_slots", "velocity_limit_slot"),
            ("velocity_signs", "velocity_limit_sign"),
            ("kind", "row_type"),
            ("parent", "row_parent"),
            ("mu", "row_mu"),
            ("beta", "row_beta"),
            ("cfm", "row_cfm"),
            ("phi", "phi"),
            ("target", "target_velocity"),
            ("restitution", "row_restitution"),
            ("rhs", "rhs"),
        ):
            value = getattr(s, source)
            if value is None:
                value = self._dummy_float if name == "velocity_signs" else self._dummy_int
            setattr(p, name, value)
        p.lower, p.upper, p.velocity_limit = (
            s.model.joint_limit_lower,
            s.model.joint_limit_upper,
            s.model.joint_velocity_limit,
        )
        p.gap, p.activation, p.pgs_beta, p.pgs_cfm, p.dt = (
            s.joint_limit_activation_gap,
            s.velocity_limit_activation_fraction,
            s.pgs_beta,
            s.pgs_cfm,
            dt,
        )
        self.contact_input = None
        self.data.maps, self.data.prefix = m, p
        prefix = prefix_kernel(self.prefix_layout, PrefixInput, KineticRows)
        wp.launch_tiled(prefix, dim=[s.world_count], inputs=[p, self.data], block_dim=32, device=s.model.device)

    def produce_contacts(self, state_in, state_aug, contacts, dt):
        """Defer current contact construction until final fallback counts are known.

        Under patch friction the contact keys are scattered now, with the type,
        parent and undivided coefficient the patch link reads and rewrites.
        """
        self.contact_input = bind(self.solver, state_in, state_aug, contacts, dt, 1.0)
        self.data.contact = self.contact_input
        if self.patch:
            # Like bind() and the other row producers, read the anchor coefficient every step.
            self.law.beta = self.solver.friction_anchor_beta
            wp.launch(
                scatter_patch_contact_keys,
                dim=self.contact_input.point0.shape[0],
                inputs=[self.contact_input, self.data],
                device=self.solver.model.device,
                max_blocks=self.solver._contact_capacity_blocks,
            )

    def finish_rows(self):
        """Publish current direct rows and fully materialize only fallback worlds."""
        s = self.solver
        if self.contact_input is not None and not self.patch:
            wp.launch(
                scatter_contact_keys,
                dim=self.contact_input.point0.shape[0],
                inputs=[self.contact_input, self.data],
                device=s.model.device,
                max_blocks=s._contact_capacity_blocks,
            )
        na, nb = self.layout.primary, self.layout.secondary
        fallback = get_patch_fallback_kernel(self.layout) if self.patch else get_fallback_kernel(self.layout)
        wp.launch_tiled(
            fallback,
            dim=[s.world_count],
            inputs=[
                *self.parallel_arguments(),
                s.constraint_count,
                s.mf_constraint_count,
                s.J_by_size[na],
                s.J_by_size[nb] if nb else self._dummy_factor,
            ],
            block_dim=32,
            device=s.model.device,
        )

    def parallel_arguments(self):
        """Return the row operands the keyed fallback and parallel tiers take: the rows and the patch law."""
        return [self.data, self.law] if self.patch else [self.data]


def create_owner(solver):
    """Admit one bounded complete numerical law; leave unsupported solvers ordinary."""
    from . import solver_feather_pgs as source  # noqa: PLC0415

    supported = (
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver.pgs_mode == "matrix_free"
        and solver.drive_mode == "augmented"
        and solver.friction_mode == "current"
        and solver.pgs_schedule == "interleaved"
        and solver.enable_joint_limits
        and 0 < solver.mf_gs_parallel_rows <= 128
        and solver.mf_gs_parallel_matrix_free
        and solver.dense_max_constraints >= solver.mf_gs_parallel_rows
        and solver.mf_gs_parallel_sweeps > 0
        and solver.pgs_iterations > 0
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and solver.articulated_contact_response == "immediate"
        and (solver._hinv_jt_writes_world or solver._jy_world_aliased)
        and not solver.fuse_joint_velocity_limits
        and not solver._regularization_enabled
        and not solver._preelim_active
        and not solver._paired_factor_coordinates
        and not solver._has_loop_joints
        and not solver._fused_diagonal_joint_limits
        and not source._COMPACT_CONTACT_BOUNDARY
        and not solver._sparse_diagonal_contact_solve
        and not solver._mimic_count
        and not solver._connect_count
        and solver._contact_w == 1.0
        and not solver._debug_buffers_enabled
        and source._INK_ON
        and not source._INK_CHECK
        and not source._CHECK_ROWS
        and not source._GROUPED_CHECK
        and not source._FPGS_CAPTURE
    )
    if not supported:
        return None
    try:
        return KeyedRows(solver)
    except ValueError:
        return None
