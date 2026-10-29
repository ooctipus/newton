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
from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_LAW, contact_friction, contact_row_frame
from .dense_rows import RowSupport, jacobian_row
from .device_preludes import warp_stride
from .field_group import Field, FieldGroup, SolverArrays
from .friction_patches import PatchRowLaw, patch_row_law
from .kernel_template import substitute
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .keyed_row_layout import prefix_kernel
from .keyed_row_layout import topology_plan as direct_topology_plan
from .law_source import law_preludes
from .prefix_rows import LAW, VELOCITY_LAW
from .route_table import capable
from .row_build import endpoint_projection, write_contact_row
from .row_inputs import ParallelRowInputs, RowArgument
from .row_laws import ROW_LAW
from .step_plan import JOINT_TOPOLOGY, TopologyPin

ROW_MAPS = FieldGroup(
    Field("groups", wp.int32, ("world", "size"), owner="maps"),
    Field("starts", wp.int32, ("world", "size"), owner="maps"),
    Field("arts", wp.int32, ("world", "size"), owner="maps"),
)
"""Each world's size-group index, first coordinate and index of its articulation of each size."""

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


def _row_source(*, layout, patch=False):
    """Decode a dense row from its key into the parallel tiers' row ``Zi`` and its records.

    A prefix row's key ``~(2 d + side)`` gives a unit coefficient. A contact row takes its geometry, frame, target,
    materials and Jacobian entries from the contact laws and its rhs from the row laws (:data:`ROW_PRELUDES`).
    ``patch`` writes the persistent patch row law: tangent rows act at the patch friction location for both the
    Jacobian and the prescribed target, carry the anchor's tangential phi and correct it with ``patch_law.beta``;
    every row's type, parent and coefficient stay as :func:`scatter_contact_keys` and the patch link left them.
    """
    dofs, output = layout.dofs, "Zi[q * ZS]"
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
    if patch:
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


@wp.kernel(enable_backward=False)
def materialize_general_rows(
    z: KineticRows,
    data: ContactInput,
    law: PatchRowLaw,
    counts: wp.array[int],
    mf_counts: wp.array[int],
    parallel_rows: int,
    primary: int,
    secondary: int,
    primary_offset: int,
    secondary_offset: int,
    support: RowSupport,
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
):
    """Write the rows of the worlds the parallel tiers leave to the general sweep: those of more rows than the tiers
    hold or with matrix-free rows, at each articulation's side (:func:`.dense_rows.jacobian_row`; the packed rows are
    dense, the parallel tiers read them at their coordinates). A warp per world (its lanes over the row coefficients)
    writes every coefficient,
    a prefix row's unit coefficient from its key ``~(2 d + side)`` and a contact row's Jacobian entry at its row
    frame (:func:`.row_build.endpoint_projection`), and a contact row's records and rhs
    (:func:`.row_build.write_contact_row`)."""
    world, lane = wp.tid()
    count = counts[world]
    if count <= parallel_rows and mf_counts[world] == 0:
        return
    dofs = primary + secondary
    for coefficient in range(lane, count * dofs, warp_stride()):
        row, q = coefficient // dofs, coefficient % dofs
        size, local = primary, q - primary_offset
        if q < primary_offset or q >= primary_offset + primary:
            size, local = secondary, q - secondary_offset
        key = z.rowkeys[world, row]
        value = float(0.0)
        if key < 0:
            token = -1 - key
            if q == token // 2:
                value = wp.where(token % 2 == 0, 1.0, -1.0)
                data.sides[world, row] = wp.vec2i(z.maps.arts[world, wp.where(size == primary, 0, 1)], -1)
        else:
            c, r = key // 4, key % 4
            body_a, body_b, normal, point_a, point_b, _phi, tangent0, tangent1, friction_a, friction_b = contact_frame(
                data, law, c
            )
            direction, p0, p1 = contact_row_frame(
                r, data.shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            value = endpoint_projection(data, body_a, data.art0[c], size, local, p0, direction)
            value -= endpoint_projection(data, body_b, data.art1[c], size, local, p1, direction)
            if q == 0:
                write_contact_row(data, law, c, r, True)
        side = wp.where(size == primary, 0, 1)
        art = z.maps.arts[world, side]
        at = jacobian_row(support, z.maps.groups[world, side], art, wp.vec2i(art, -1))
        if size == primary:
            J_a[at[0], row, at[1] + local] = value
        else:
            J_b[at[0], row, at[1] + local] = value


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
        + _row_source(layout=layout, patch=patch)
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


TOPOLOGY = (*JOINT_TOPOLOGY, "joint_dof_dim", "articulation_start", "body_world")
"""The structural model arrays the keyed layout reads."""


class KeyedRows:
    """Own the per-world row prefix, direct keyed rows and the complete fallback."""

    def __init__(self, solver):
        self.solver = solver
        self.keyed_rows = True
        self.layout, groups, starts, arts = direct_topology_plan(solver)
        self.prefix_layout = self.layout
        dev, worlds, capacity = solver.model.device, solver.world_count, solver.dense_max_constraints
        self.data, self.prefix = KineticRows(), PrefixInput()
        self.data.rowkeys = SolverArrays(None, dev).array(KINETIC_ROWS[0], {"world": worlds, "row": capacity})
        self.maps = field_group.upload(RowMaps, dev, groups=groups, starts=starts, arts=arts)
        self.contact_input = None
        # The placeholders of absent prefix tables, allocated before any capture.
        solver._arrays.placeholder(wp.int32)
        solver._arrays.placeholder(wp.float32)
        self._topology = TopologyPin.read(solver.model, TOPOLOGY)
        # Persistent patch friction: the rows carry the patch law, and the patch link owns parents and coefficients.
        self.patch = bool(solver._friction_anchors_enabled)
        self.law = patch_row_law(solver)

    def validate_notification(self, flags=ModelFlags.ALL):
        """Refuse a structural change the keyed layout does not survive (:class:`.step_plan.TopologyPin`)."""
        self._topology.validate(
            self.solver.model, flags, lambda: direct_topology_plan(self.solver)[0] == self.prefix_layout, "Keyed row"
        )

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
        """Bind the step's contact input; :meth:`finish_rows` writes the rows once the counts are final."""
        self.contact_input = contact_input
        self.data.contact = self.contact_input
        # Like the other row producers, bind the anchor coefficient and the patch view (which
        # contact-pool growth replaces) every step.
        self.law = patch_row_law(self.solver)

    def finish_rows(self):
        """Scatter the contact keys, then write the general worlds' rows (:func:`materialize_general_rows`).

        Under patch friction the keys carry each row's type, parent and undivided coefficient, and the general
        worlds' records the same; the patch link, which runs after every row producer, links and divides them.
        """
        s, layout = self.solver, self.layout
        na, nb = layout.primary, layout.secondary
        if self.contact_input is not None:
            pool = s._contact_pool
            pool.launch(
                scatter_contact_keys,
                pool.launch_rows(),
                inputs=[self.contact_input, self.data, self.law],
                extent=self.contact_input.count,
                max_blocks=s._contact_capacity_blocks,
            )
        wp.launch_tiled(
            materialize_general_rows,
            dim=[s.world_count],
            inputs=[
                self.data,
                self.data.contact,
                self.law,
                s.constraint_count,
                s.mf_constraint_count,
                layout.parallel_rows,
                na,
                nb,
                layout.primary_offset,
                layout.secondary_offset,
                s._row_support(),
                s._jacobian_rows(na),
                s._jacobian_rows(nb or na),
            ],
            block_dim=32,
            device=s.model.device,
        )

    def parallel_arguments(self):
        """Return the row operands the keyed fallback and parallel tiers take: the rows and the patch law."""
        return [self.data, self.law] if self.patch else [self.data]


def create_owner(solver):
    """Admit one bounded complete numerical law where the in-kernel parallel tiers (``_ink_sizes``) take the rows: the
    laws keyed rows implement (``route_table.CAPABILITIES``), then their fit."""
    supported = (
        capable("keyed_rows", solver._laws) is None
        and solver._ink_sizes is not None
        and (solver._hinv_jt_writes_world or solver._jy_world_aliased)
        and not solver._fused_diagonal_joint_limits
    )
    if not supported:
        return None
    try:
        return KeyedRows(solver)
    except ValueError:
        return None
