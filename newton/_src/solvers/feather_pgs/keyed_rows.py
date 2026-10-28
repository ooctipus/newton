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
from . import field_group
from .contact_input import ContactInput
from .contact_laws import CONTACT_LAW, contact_friction
from .device_preludes import dual_native, lane_group
from .field_group import Field, FieldGroup, SolverArrays
from .friction_patches import PatchRowLaw, patch_row_law
from .kernel_template import WORLDS, KernelTemplate, substitute
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .keyed_row_layout import prefix_kernel
from .keyed_row_layout import topology_plan as direct_topology_plan
from .law_source import law_preludes
from .prefix_rows import LAW, VELOCITY_LAW
from .row_inputs import ParallelRowInputs, RowArgument
from .row_laws import ROW_LAW

ROW_MAPS = FieldGroup(
    Field("groups", wp.int32, ("world", "size"), owner="maps"),
    Field("starts", wp.int32, ("world", "size"), owner="maps"),
)
"""Each world's size-group index and first coordinate of its articulation of each size."""

RowMaps = field_group.struct("RowMaps", ROW_MAPS, __name__)

PREFIX_INPUT = (
    ROW_MAPS["starts"],
    *LAW,
    *VELOCITY_LAW,
    Field("velocity_slots", wp.int32, ("velocity_row",), attr="velocity_limit_slot"),
    Field("velocity_signs", wp.float32, ("velocity_row",), attr="velocity_limit_sign"),
)
"""The keyed prefix's inputs and the dense rows it writes (:data:`.prefix_rows.LAW`); without joint velocity limits,
their tables take placeholders."""

PrefixInput = field_group.struct("PrefixInput", PREFIX_INPUT, __name__)


KINETIC_ROWS = (
    Field("rowkeys", wp.int32, ("world", "row"), fill=None),
    Field("maps", RowMaps),
    Field("prefix", PrefixInput),
    Field("contact", ContactInput),
)
"""The keyed kernels' input: each dense row's key, the maps, the prefix input and the contact input."""

KineticRows = field_group.struct("KineticRows", KINETIC_ROWS, __name__)


@wp.kernel(enable_backward=False)
def scatter_contact_keys(data: ContactInput, z: KineticRows, law: PatchRowLaw):
    """Emit each allocated contact row's key (raw contact ID and row direction).

    Under persistent patch friction also the type, parent and undivided coefficient of each row: the patch
    link runs after this launch and before any row is read, links each normal row into its patch ring and
    divides the tangent coefficient among the anchors that received rows.
    """
    contact = wp.tid()
    if contact < data.count[0]:
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            world = data.world[contact]
            slot = data.slot[contact]
            for direction in range(data.slots_needed[contact]):
                z.rowkeys[world, slot + direction] = contact * 4 + direction
            if law.patches.enabled != 0:
                mu = contact_friction(data.shape0[contact], data.shape1[contact], data.material_mu)
                for direction in range(data.slots_needed[contact]):
                    row = slot + direction
                    if direction == 0:
                        data.row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
                        data.row_parent[world, row] = -1
                        data.row_mu[world, row] = mu
                    else:
                        data.row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
                        data.row_parent[world, row] = slot
                        data.row_mu[world, row] = mu * data.friction_scale


ROW_PRELUDES = law_preludes(CONTACT_LAW, ROW_LAW)
"""Device preludes of the row source: the contact and row laws it calls (``ContactLaw::...``, ``RowLaw::...``)."""


def _row_source(*, layout, output="Jr[q]", private_metadata=False, patch=False):
    """Share row geometry and metadata between direct ingestion and the fallback.

    A contact row takes its geometry, frame, target, materials and Jacobian entries
    from the contact laws and its rhs from the row laws (:data:`ROW_PRELUDES`). ``patch`` writes the persistent
    patch row law: tangent rows act at the patch friction location for both the
    Jacobian and the prescribed target, carry the anchor's tangential phi and
    correct it with ``patch_law.beta``; every row's type, parent and coefficient
    stay as :func:`scatter_contact_keys` and the patch link left them.
    """
    dofs = layout.dofs
    prefix = substitute(
        """
        const int token = ~key, d = token / 2;
        const float sign = (token & 1) == 0 ? 1.0f : -1.0f;
        #pragma unroll
        for (int q = 0; q < $DOFS; ++q) {
            $OUTPUT = q == d ? sign : 0.0f;
        }
    """,
        DOFS=dofs,
        OUTPUT=output,
    )
    physical_projection = f"""
            const int n = c.response_dofs.data[art], local = {layout.local("q")};
            if (n != ({layout.primary_test("q")} ? {layout.primary} : {layout.secondary})
                || (c.body_mask.data[body] & (1u << local)) == 0u) continue;
            const wp::spatial_vector S = c.motion.data[c.dof_start.data[art] + local];
            value += sign * ContactLaw::jacobian_entry(S, direction, point, c.origin.data[art]);
    """
    projection = substitute(
        """
        #pragma unroll
        for (int q = 0; q < $DOFS; ++q) {
            float value = 0.0f;
            for (int side = 0; side < 2; ++side) {
                const int body = side == 0 ? ba : bb, art = side == 0 ? aa : ab;
                if (body < 0 || art < 0) continue;
                const float sign = side == 0 ? 1.0f : -1.0f;
                const wp::vec3 point = side == 0 ? p0 : p1;
                $PROJECT
            }
            $OUTPUT = value;
        }
        """,
        DOFS=dofs,
        PROJECT=physical_projection,
        OUTPUT=output,
    )
    incident = (
        f"relative += {output} * c.incident.data[z.maps.starts.data[world * 2 + ({layout.primary_test('q')} ? 0 : 1)]"
        f" + {layout.local('q')}];"
    )
    if patch:
        friction_location = (
            "const wp::vec3 fa = patch_law.patches.point_a.data[contact], fb = patch_law.patches.point_b.data[contact];"
        )
        tangent_phi = "patch_law.patches.phi.data[contact][r - 1]"
        tangent_beta = "patch_law.beta"
        row_law = "        c.row_beta.data[at] = r == 0 ? c.beta : patch_law.beta;\n"
    else:
        friction_location = "const auto [fa, fb] = ContactLaw::contact_friction_points(pa, pb, c.shared_anchor, c.friction_shared_anchor);"
        tangent_phi = "0.0f"
        tangent_beta = "0.0f"
        row_law = (
            "        c.row_type.data[at] = r == 0 ? 0 : 2;\n"
            "        c.row_parent.data[at] = r == 0 ? -1 : c.slot.data[contact];\n"
            "        c.row_mu.data[at] = r == 0 ? mu : friction;\n"
            "        c.row_beta.data[at] = r == 0 ? c.beta : 0.0f;\n"
        )
    if not private_metadata:
        metadata = ""
    elif patch:
        metadata = "built_rhs = rhs;"
    else:
        metadata = (
            "built_rhs = rhs; built_mu = r == 0 ? mu : friction; built_parent = r == 0 ? -1 : c.slot.data[contact];"
        )
    return substitute(
        r"""
    const int at = world * z.rowkeys.shape[1] + row;
    const int key = z.rowkeys.data[at];
    if (key < 0) {
${PREFIX}
    } else {
        const auto& c = z.contact;
        const int contact = key >> 2, r = key & 3;
        const int sa = c.shape0.data[contact], sb = c.shape1.data[contact];
        const auto [ba, bb] = ContactLaw::contact_bodies(sa, sb, c.shape_body);
        const int aa = c.art0.data[contact], ab = c.art1.data[contact];
        const auto [normal, pa, pb, separation] = ContactLaw::contact_points(
            c.normal.data[contact], c.point0.data[contact], c.point1.data[contact],
            c.margin0.data[contact], c.margin1.data[contact], ba, bb, c.body_q);
        wp::vec3 t0(0.0f), t1(0.0f);
        if (r > 0) {
            const auto basis = ContactLaw::contact_tangent_basis(normal);
            t0 = basis.v0;
            t1 = basis.v1;
        }
        ${FRICTION_LOCATION}
        const auto [direction, p0, p1] = ContactLaw::contact_row_frame(r, c.shared_anchor, normal, t0, t1, pa, pb, fa, fb);
        const float target = ContactLaw::prescribed_target(ba, aa, bb, ab, p0, p1, direction, c.prescribed, c.origin, c.body_v);
        const auto [mu, restitution] = ContactLaw::contact_materials(sa, sb, c.material_mu, c.material_restitution);
        const float friction = mu * c.friction_scale;
${PROJECTION}
        const float phi = r == 0 ? separation : ${TANGENT_PHI};
        float rhs = RowLaw::row_rhs(r == 0 ? 0 : 2, phi, r == 0 ? c.beta : ${TANGENT_BETA}, target, 1.0f / c.dt,
                                    c.bias_scale, c.speculative_scale, 0.0f, 1.0f).v0;
        if (r == 0 && restitution > 0.0f) {
            float relative = -target;
            #pragma unroll
            for (int q = 0; q < ${DOFS}; ++q) { ${INCIDENT} }
            const auto bounce = RowLaw::contact_restitution_rhs(phi, relative, target, restitution, c.dt,
                                                                c.restitution_threshold);
            if (bounce.v0) rhs = bounce.v1;
        }
${ROW_LAW}        c.row_cfm.data[at] = c.cfm;
        c.phi.data[at] = phi; c.target.data[at] = target;
        c.restitution.data[at] = r == 0 ? restitution : 0.0f;
        c.rhs.data[at] = rhs; c.diag.data[at] = c.cfm;
        ${PRIVATE_METADATA}
    }
""",
        PREFIX=prefix,
        FRICTION_LOCATION=friction_location,
        PROJECTION=projection,
        INCIDENT=incident,
        TANGENT_PHI=tangent_phi,
        TANGENT_BETA=tangent_beta,
        ROW_LAW=row_law,
        PRIVATE_METADATA=metadata,
        DOFS=dofs,
    )


def _fallback_source(layout, *, patch=False):
    primary = layout.primary
    secondary = layout.secondary
    dofs = primary + secondary
    limit = layout.parallel_rows
    primary_test = layout.primary_test("q")
    local_a = f"q - {layout.primary_offset}"
    local_b = f"q - {layout.secondary_offset}"
    source = substitute(
        """
    if (counts.data[world] <= $LIMIT && mf_counts.data[world] == 0) return;
    const int ga = z.maps.groups.data[world * 2], gb = z.maps.groups.data[world * 2 + 1];
    for (int row = LANE; row < counts.data[world]; row += STRIDE) {
        float Jr[$DOFS];
$ROW
        #pragma unroll
        for (int q = 0; q < $DOFS; ++q) {
            if ($PRIMARY_TEST) J16.data[(ga * J16.shape[1] + row) * $NA + $LOCAL_A] = Jr[q];
            else J6.data[(gb * J6.shape[1] + row) * $NB + $LOCAL_B] = Jr[q];
        }
    }
""",
        ROW=_row_source(layout=layout, patch=patch),
        LIMIT=limit,
        DOFS=dofs,
        PRIMARY_TEST=primary_test,
        NA=primary,
        NB=secondary,
        LOCAL_A=local_a,
        LOCAL_B=local_b,
    )
    # A warp per world on CUDA (launched in blocks of 32), one thread on the CPU.
    return dual_native(source, preludes=(lane_group(32), *ROW_PRELUDES))


_FALLBACK_PARAMS = (
    ("z", KineticRows),
    ("counts", wp.array[int]),
    ("mf_counts", wp.array[int]),
    ("J16", wp.array3d[float]),
    ("J6", wp.array3d[float]),
)
"""The fallback's operands; the patch fallback takes the patch row law after the rows."""


@cache
def get_fallback_kernel(layout, patch=False):
    """Materialize current physical rows in one pass over fallback worlds only.

    ``patch`` writes the persistent patch row law, whose struct the kernel takes after the rows.
    """
    params = (_FALLBACK_PARAMS[0], ("patch_law", PatchRowLaw), *_FALLBACK_PARAMS[1:]) if patch else _FALLBACK_PARAMS
    name = "keyed_rows_patch_fallback" if patch else "keyed_rows_fallback"
    return KernelTemplate(name, params, _fallback_source(layout, patch=patch), WORLDS, composed=True).kernel()


def get_direct_parallel_inputs(layout, na, nb, oa, ob, dofs, capacity, *, patch=False):
    """Decode physical endpoints in the ordinary factor stage.

    ``patch`` writes the persistent patch row law and reads every row's parent and
    coefficient as the scatter and the patch link left them; point friction builds
    them with the row.
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
    built = "float built_rhs = 0.0f, built_mu = 0.0f;\n            int built_parent = -1;"
    publish = "{\n                s_rhs[i] = built_rhs; s_parent[i] = built_parent; s_mu[i] = built_mu;\n            }"
    arguments = (RowArgument("kinetic_rows", KineticRows, after="world_row_cfm"),)
    parent = f"{key} < 0 ? world_row_parent.data[off_dense + i] : 0"
    mu = f"{key} < 0 ? world_row_mu.data[off_dense + i] : 0"
    if patch:
        built, publish = "float built_rhs = 0.0f;", "s_rhs[i] = built_rhs;"
        arguments += (RowArgument("patch_law", PatchRowLaw, after="kinetic_rows"),)
        parent, mu = "world_row_parent.data[off_dense + i]", "world_row_mu.data[off_dense + i]"
    reader = (
        f"""
            const int row = i;
            const auto& z = kinetic_rows;
            {built}
    """
        + _row_source(layout=layout, output="Zi[q * ZS]", private_metadata=True, patch=patch)
        + f"""
            if ({key} >= 0) {publish}
    """
    )
    return ParallelRowInputs(
        stage=None,
        physical_row=reader,
        preludes=ROW_PRELUDES,
        arguments=arguments,
        row_type=f"({key} >= 0 ? (({key} & 3) == 0 ? 0 : 2) : world_row_type.data[off_dense + i])",
        rhs=f"{key} < 0 ? rhs_bias.data[off_dense + i] : 0",
        parent=parent,
        mu=mu,
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
        self.data, self.prefix = KineticRows(), PrefixInput()
        self.data.rowkeys = SolverArrays(None, dev).array(KINETIC_ROWS[0], {"world": worlds, "row": capacity})
        self.maps = field_group.upload(RowMaps, dev, groups=groups, starts=starts)
        self.contact_input = None
        # The placeholders of absent prefix tables, allocated before any capture.
        solver._arrays.placeholder(wp.int32)
        solver._arrays.placeholder(wp.float32)
        self._dummy_factor = wp.zeros((1, 1, 1), dtype=float, device=dev)
        self._signature = _signature(solver)
        # Persistent patch friction: the rows carry the patch law, and the patch link owns parents and coefficients.
        self.patch = bool(solver._friction_anchors_enabled)
        self.law = patch_row_law(solver)

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
        s, m = self.solver, self.maps
        # A table the solver does not have (joint velocity limits) takes a placeholder.
        absent = {
            field.name: s._arrays.placeholder(field.dtype)
            for field in PREFIX_INPUT
            if field.owner == "solver" and len(field.dims) == 1 and getattr(s, field.attr or field.name) is None
        }
        p = field_group.bind(
            PrefixInput, self.prefix, solver=s, model=s.model, maps=m, state_in=state_in, dt=dt, **absent
        )
        self.contact_input = None
        self.data.maps, self.data.prefix = m, p
        prefix = prefix_kernel(self.prefix_layout, PrefixInput, KineticRows)
        wp.launch_tiled(prefix, dim=[s.world_count], inputs=[p, self.data], block_dim=32, device=s.model.device)

    def produce_contacts(self, contact_input):
        """Defer current contact construction (the step's bound contact input) until final fallback counts are known.

        Under patch friction the contact keys are scattered now, with the type,
        parent and undivided coefficient the patch link reads and rewrites.
        """
        self.contact_input = contact_input
        self.data.contact = self.contact_input
        # Like the other row producers, bind the anchor coefficient and the patch view (which
        # contact-pool growth replaces) every step.
        self.law = patch_row_law(self.solver)
        if self.patch:
            self._scatter_keys()

    def finish_rows(self):
        """Publish current direct rows and fully materialize only fallback worlds."""
        s = self.solver
        if self.contact_input is not None and not self.patch:
            self._scatter_keys()
        na, nb = self.layout.primary, self.layout.secondary
        fallback = get_fallback_kernel(self.layout, patch=self.patch)
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

    def _scatter_keys(self):
        """Scatter the contact keys: under patch friction before the patch link, else with the final counts."""
        pool = self.solver._contact_pool
        pool.launch(
            scatter_contact_keys,
            pool.launch_rows(),
            inputs=[self.contact_input, self.data, self.law],
            extent=self.contact_input.count,
            max_blocks=self.solver._contact_capacity_blocks,
        )

    def parallel_arguments(self):
        """Return the row operands the keyed fallback and parallel tiers take: the rows and the patch law."""
        return [self.data, self.law] if self.patch else [self.data]


def create_owner(solver):
    """Admit one bounded complete numerical law; leave unsupported solvers ordinary."""
    facts = solver._facts
    supported = (
        facts.forward_cuda
        and facts.matrix_free
        and facts.plain_sweep
        and not facts.regularized
        and solver.enable_joint_limits
        and 0 < solver.mf_gs_parallel_rows <= 128
        and solver.mf_gs_parallel_matrix_free
        and solver.dense_max_constraints >= solver.mf_gs_parallel_rows
        and solver.pgs_iterations > 0
        and (solver._hinv_jt_writes_world or solver._jy_world_aliased)
        and not solver.fuse_joint_velocity_limits
        and not solver._paired_factor_coordinates
        and not solver._has_loop_joints
        and not solver._fused_diagonal_joint_limits
        and not solver._sparse_diagonal_contact_solve
        and not solver._mimic_count
        and not solver._connect_count
        and solver._ink_sizes is not None
    )
    if not supported:
        return None
    try:
        return KeyedRows(solver)
    except ValueError:
        return None
