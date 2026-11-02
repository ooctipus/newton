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
from .contact_laws import CONTACT_LAW, CONTACT_SPIN_ROW, contact_friction
from .dense_rows import UNIT_CONTACT, UNIT_LIMIT, unit_source
from .field_group import Field, FieldGroup, SolverArrays
from .friction_patches import PatchRowLaw, patch_row_law
from .kernel_template import substitute
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION
from .keyed_row_layout import prefix_kernel
from .keyed_row_layout import topology_plan as direct_topology_plan
from .law_source import law_preludes
from .prefix_rows import LAW, VELOCITY_LAW
from .route_table import capable
from .row_build import contact_torsion_mu
from .row_inputs import ParallelRowInputs, RowArgument
from .row_laws import ROW_LAW
from .step_plan import JOINT_TOPOLOGY, TopologyPin
from .tree_factor import WALK_FRAME, WALK_LINK

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
    Field("velocity_slots", wp.int32, ("velocity_row",), owner="laws", attr="velocity_limit_slot"),
    Field("velocity_signs", wp.float32, ("velocity_row",), owner="laws", attr="velocity_limit_sign"),
)
"""The keyed prefix's inputs and the dense rows it writes (:data:`.prefix_rows.LAW`); without joint velocity limits,
their tables take placeholders."""

PrefixInput = field_group.struct("PrefixInput", PREFIX_INPUT, __name__)


KINETIC_ROWS = (
    Field("rowkeys", wp.int32, ("world", "row"), fill=None),
    Field("maps", RowMaps),
    Field("prefix", PrefixInput),
    Field("contact", ContactInput),
    Field("first", wp.int32, ("world",)),
)
"""The keyed kernels' input: each dense row's key, the maps, the prefix input and the contact input."""

KineticRows = field_group.struct("KineticRows", KINETIC_ROWS, __name__)


@wp.kernel(enable_backward=False)
def scatter_contact_keys(data: ContactInput, z: KineticRows, law: PatchRowLaw):
    """Emit each allocated contact row's key (raw contact ID and row direction: the normal, the tangents and, under
    contact torsion, the spin row).

    Under persistent patch friction also the type, parent and undivided coefficient of each row: the patch
    link runs after this launch and before any row is read, links each normal row into its patch ring and
    divides the tangent coefficient among the anchors that received rows. A material contact's tangents take its
    friction scale (:func:`.row_build.write_contact_row`).
    """
    contact = wp.tid()
    if contact < data.count[0]:
        if data.path[contact] == 0 and data.slot[contact] >= 0:
            world = data.world[contact]
            line = data.first[world]
            slot = data.slot[contact]
            for direction in range(data.slots_needed[contact]):
                z.rowkeys[line, slot + direction] = contact * 4 + direction
            if law.patches.enabled != 0:
                mu = contact_friction(data.shape0[contact], data.shape1[contact], data.material_mu)
                friction = mu
                if data.material != 0:
                    if data.stiffness[contact] > 0.0 and data.friction_weight[contact] > 0.0:
                        friction *= data.friction_weight[contact]
                for direction in range(data.slots_needed[contact]):
                    row = slot + direction
                    if direction == 0:
                        data.row_type[line, row] = PGS_CONSTRAINT_TYPE_CONTACT
                        data.row_parent[line, row] = -1
                        data.row_mu[line, row] = mu
                    elif direction == CONTACT_SPIN_ROW:
                        data.row_type[line, row] = PGS_CONSTRAINT_TYPE_TORSION
                        data.row_parent[line, row] = slot
                        data.row_mu[line, row] = contact_torsion_mu(data, contact)
                    else:
                        data.row_type[line, row] = PGS_CONSTRAINT_TYPE_FRICTION
                        data.row_parent[line, row] = slot
                        data.row_mu[line, row] = friction


ROW_PRELUDES = law_preludes(CONTACT_LAW, ROW_LAW)
"""Device preludes of the row source: the contact and row laws it calls (``ContactLaw::...``, ``RowLaw::...``)."""


def _row_source(*, layout, patch=False, weighted=False, torsion=False):
    """Decode a dense row from its key and walk its seeds into the parallel tiers' row ``Zi`` (``TIER_WALK``,
    :class:`.response_operator.WalkedRows`), its incident velocity and its records.

    A prefix row's key ``~(2 d + side)`` gives its seed, the covector ``+-e_d`` on the tree holding world coordinate
    ``d``. A contact row takes its geometry, frame, target and materials from the contact laws and its rhs from the row
    laws (:data:`ROW_PRELUDES`); its seed on each body of a tree of the world is the wrench of its direction at its row
    point, ``(d, (p - o) x d)`` at the body's frame ``o`` (the factor's; the spin row's ``(0, n)``), the second
    body's negated. No Jacobian is formed.
    ``patch`` writes the persistent patch row law: tangent rows act at the patch friction location for both the
    seed and the prescribed target, carry the anchor's tangential phi and correct it with ``patch_law.beta``;
    every row's type, parent and coefficient stay as :func:`scatter_contact_keys` and the patch link left them.
    ``weighted`` writes each row's weight (the relative regularization's, 1 for a firing rebound) and, under
    ``contact_compliance``, a material contact's law as :func:`.row_build.write_contact_row` does: its normal's rhs
    and compliance (:func:`.row_laws.contact_material_rhs`, zero on every other row), no restitution and its
    tangents' friction scale; the tiers form its weight from that compliance. Under contact ``torsion`` a key's fourth
    direction is the spin row (:func:`.row_build.write_contact_row`'s): its seed the torque ``(0, n)``, its coefficient
    the shapes' torsional friction, its rhs minus the prescribed spin.
    """
    kind = "r == 0 ? 0 : 2"
    target = "ContactLaw::prescribed_target(ba, aa, bb, ab, row_point, direction, c.prescribed, c.origin, c.body_v)"
    if torsion:
        kind = f"r == 0 ? 0 : (r == 3 ? {int(PGS_CONSTRAINT_TYPE_TORSION)} : 2)"
        target = f"r == 3 ? ContactLaw::prescribed_spin_target(ba, aa, bb, ab, normal, c.prescribed, c.body_v)\n            : {target}"
    prefix = f"""
        const int token = ~key, d = token / 2;
        const float sign = (token & 1) == 0 ? 1.0f : -1.0f;
        const int side = {layout.primary_test("d")} ? 0 : 1, k = {layout.local("d")};
        TIER_WALK(side, walk.dof_body.data[walk.dof_start.data[side == 0 ? art_a : art_b] + k], k, &sign);
    """
    spin = "r == 3" if torsion else "false"
    seeds = f"""
        // Each body's seed on its tree of the world: the row's wrench at the body's frame (the factor's).
        for (int end = 0; end < 2; ++end) {{
            const int body = end == 0 ? ba : bb, art = end == 0 ? aa : ab;
            const int side = art >= 0 && art == art_a ? 0 : (art >= 0 && art == art_b ? 1 : -1);
            if (body < 0 || side < 0) continue;
            const int joint = walk.body_joint.data[body];
            const float* o = walk.link.data + (size_t)joint * {WALK_LINK} + {WALK_FRAME};
            const float sign = end == 0 ? 1.0f : -1.0f;
            const float d0 = sign * direction[0], d1 = sign * direction[1], d2 = sign * direction[2];
            const float r0 = row_point[0] - o[0], r1 = row_point[1] - o[1], r2 = row_point[2] - o[2];
            const bool torque = {spin};
            const float seed[6] = {{torque ? 0.0f : d0, torque ? 0.0f : d1, torque ? 0.0f : d2,
                                    torque ? d0 : r1 * d2 - r2 * d1, torque ? d1 : r2 * d0 - r0 * d2,
                                    torque ? d2 : r0 * d1 - r1 * d0}};
            TIER_WALK(side, joint - walk.joint_start.data[art], -1, seed);
        }}
"""
    if patch:
        friction_location = "const wp::vec3 friction_point = patch_law.patches.point.data[contact];"
        tangent_phi = "patch_law.patches.phi.data[contact][r - 1]"
        if torsion:
            tangent_phi = f"(r == 3 ? 0.0f : {tangent_phi})"
        tangent_beta = "patch_law.beta"
        row_law = "        c.row_beta.data[at] = r == 0 ? normal_beta : patch_law.beta;\n"
    else:
        friction_location = "const wp::vec3 friction_point = point;"
        tangent_phi = "0.0f"
        tangent_beta = "0.0f"
        row_law = (
            f"        c.row_type.data[at] = {kind};\n"
            "        c.row_parent.data[at] = r == 0 ? -1 : c.slot.data[contact];\n"
            "        c.row_mu.data[at] = r == 0 ? mu : friction;\n"
            "        c.row_beta.data[at] = r == 0 ? normal_beta : 0.0f;\n"
        )
    materials = """        const auto [mu, restitution] = ContactLaw::contact_materials(sa, sb, c.material_mu, c.material_restitution);
        const float friction = mu;"""
    rhs = """        float rhs = RowLaw::row_rhs(r == 0 ? 0 : 2, phi, r == 0 ? normal_beta : ${TANGENT_BETA}, target, 1.0f / c.dt,
                                    c.bias_scale, c.speculative_scale, 0.0f, 1.0f).v0;"""
    bounce, weight = "if (bounce.v0) rhs = bounce.v1;", ""
    if weighted:
        materials = """        const auto [mu, rigid_restitution] = ContactLaw::contact_materials(sa, sb, c.material_mu,
                                                                           c.material_restitution);
        const float stiffness = c.material != 0 ? c.stiffness.data[contact] : 0.0f;
        const float restitution = stiffness > 0.0f ? 0.0f : rigid_restitution;
        const float friction = mu
            * (stiffness > 0.0f && c.friction_weight.data[contact] > 0.0f ? c.friction_weight.data[contact] : 1.0f);"""
        rhs = """        const auto row_law = RowLaw::row_rhs(r == 0 ? 0 : 2, phi, r == 0 ? normal_beta : ${TANGENT_BETA}, target,
                                             1.0f / c.dt, c.bias_scale, c.speculative_scale, 0.0f, c.contact_w);
        float rhs = row_law.v0, weight = row_law.v1;"""
        bounce = "if (bounce.v0) { rhs = bounce.v1; weight = 1.0f; }"
        weight = """        float eps = 0.0f;
        if (r == 0 && stiffness > 0.0f) {
            const auto spring = RowLaw::contact_material_rhs(phi, target, stiffness, c.damping.data[contact], c.dt);
            rhs = spring.v0;
            eps = spring.v1;
        }
        c.row_w.data[at] = weight;
        if (c.material != 0) c.row_eps.data[at] = eps;
"""
    rhs = rhs.replace("${TANGENT_BETA}", tangent_beta).replace(
        "RowLaw::row_rhs(r == 0 ? 0 : 2,", f"RowLaw::row_rhs({kind},"
    )
    if torsion:
        # The spin row's coefficient is the shapes' torsional friction.
        spin_mu = "r == 3 ? ContactLaw::contact_friction(sa, sb, c.material_mu_torsional) : "
        materials = materials.replace("const float friction = ", f"const float friction = {spin_mu}")
    if patch:
        metadata = "built_rhs = rhs;"
    else:
        metadata = (
            "built_rhs = rhs; built_mu = r == 0 ? mu : friction; built_parent = r == 0 ? -1 : c.slot.data[contact];"
        )
    return substitute(
        r"""
    const int at = z.first.data[world] * z.rowkeys.shape[1] + row;
    const int key = z.rowkeys.data[at];
    if (key < 0) {
${PREFIX}
    } else {
        const auto& c = z.contact;
        const int contact = key >> 2, r = key & 3;
        const int sa = c.shape0.data[contact], sb = c.shape1.data[contact];
        const auto [ba, bb] = ContactLaw::contact_bodies(sa, sb, c.shape_body);
        const int aa = c.art0.data[contact], ab = c.art1.data[contact];
        const auto [normal, ta, oa, tb, ob, separation, gap] = ContactLaw::contact_points(
            c.normal.data[contact], c.point0.data[contact], c.point1.data[contact],
            c.margin0.data[contact], c.margin1.data[contact], ba, bb, c.body_q);
        wp::vec3 t0(0.0f), t1(0.0f);
        if (r > 0) {
            const auto basis = ContactLaw::contact_tangent_basis(normal);
            t0 = basis.v0;
            t1 = basis.v1;
        }
        const wp::vec3 point = ContactLaw::contact_point(ta, oa, separation);
        ${FRICTION_LOCATION}
        const auto [direction, row_point] = ContactLaw::contact_row_frame(r, normal, t0, t1, point, friction_point);
        const float target = ${TARGET};
${MATERIALS}
${SEEDS}
        const float phi = r == 0 ? gap : ${TANGENT_PHI};
        const float normal_beta = r == 0 ? RowLaw::contact_capped_beta(phi, c.beta, RowLaw::contact_depenetration_cap(
            aa >= 0 && c.response_dofs.data[aa] > 0 ? ba : -1, ab >= 0 && c.response_dofs.data[ab] > 0 ? bb : -1,
            c.max_depenetration_velocity), c.dt) : 0.0f;
${RHS}
        if (r == 0 && restitution > 0.0f) {
            const float relative = incident - target;
            const auto bounce = RowLaw::contact_restitution_rhs(phi, relative, target, restitution, c.dt,
                                                                c.restitution_threshold);
            ${BOUNCE}
        }
${WEIGHT}${ROW_LAW}        c.row_cfm.data[at] = c.cfm;
        c.phi.data[at] = phi; c.target.data[at] = target;
        c.restitution.data[at] = r == 0 ? restitution : 0.0f;
        c.rhs.data[at] = rhs; c.diag.data[at] = c.cfm;
        ${PRIVATE_METADATA}
    }
""",
        PREFIX=prefix,
        MATERIALS=materials,
        RHS=rhs,
        BOUNCE=bounce,
        WEIGHT=weight,
        FRICTION_LOCATION=friction_location,
        SEEDS=seeds,
        TANGENT_PHI=tangent_phi,
        ROW_LAW=row_law,
        PRIVATE_METADATA=metadata,
        TARGET=target,
    )


@wp.kernel(enable_backward=False)
def keyed_general_units(
    z: KineticRows,
    counts: wp.array[int],
    mf_counts: wp.array[int],
    parallel_rows: int,
    primary: int,
    primary_offset: int,
    secondary_offset: int,
    first: wp.array[int],
    dof_start: wp.array[int],
    sides: wp.array2d[wp.vec2i],
    units: wp.array2d[int],
    owner: wp.array[int],
):
    """Hand the worlds the parallel tiers leave to the general sweep (those of more rows than the tiers hold or with
    matrix-free rows) to the one writer (:func:`.walk_rows.walked_rows_kernel`), a thread a world: each row's unit
    source from its key (a contact row's ``4 c + r``, a prefix row's covector ``~(2 d + side)`` at its tree's
    coordinate) and a prefix row's sides; ``owner`` is 1 where the tiers hold the world (the writer skips it), 0 where
    the writer walks its rows. The general worlds' rows then take the walk and the release ``R`` as every walked row.
    It reads no contact input: a step without contacts binds none."""
    world = wp.tid()
    count = counts[world]
    if count <= parallel_rows and mf_counts[world] == 0:
        owner[world] = 1
        return
    owner[world] = 0
    line = first[world]
    for row in range(wp.min(count, units.shape[1])):
        key = z.rowkeys[line, row]
        if key < 0:
            token = -1 - key
            q = token // 2
            side = int(0)
            local = q - primary_offset
            if local < 0 or local >= primary:
                side = 1
                local = q - secondary_offset
            art = z.maps.arts[world, side]
            sides[line, row] = wp.vec2i(art, -1)
            units[line, row] = unit_source(UNIT_LIMIT, 2 * (dof_start[art] + local) + token % 2)
        else:
            units[line, row] = unit_source(UNIT_CONTACT, key)


def _key_kind(key: str, torsion: bool) -> str:
    """The row type of a contact row from its key's direction: the normal's, a tangent's or, under contact torsion,
    the spin row's."""
    if torsion:
        return f"({key} & 3) == 0 ? 0 : (({key} & 3) == 3 ? {int(PGS_CONSTRAINT_TYPE_TORSION)} : 2)"
    return f"({key} & 3) == 0 ? 0 : 2"


def get_direct_parallel_inputs(layout, na, nb, oa, ob, dofs, capacity, *, patch=False, weighted=False, torsion=False):
    """Decode each row from its key and walk its seeds in the tier's row stage (:func:`_row_source`).

    ``patch`` writes the persistent patch row law and reads every row's parent and
    coefficient as the scatter and the patch link left them; point friction builds
    them with the row. ``weighted`` writes the rows' weights and material contacts' law, ``torsion`` decodes the spin
    rows (:func:`_row_source`).
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
        + _row_source(layout=layout, patch=patch, weighted=weighted, torsion=torsion)
        + f"""
            if ({key} >= 0) {publish}
    """
    )
    return ParallelRowInputs(
        read=reader,
        preludes=ROW_PRELUDES,
        arguments=arguments,
        row_type=f"({key} >= 0 ? ({_key_kind(key, torsion)}) : world_row_type.data[off_dense + i])",
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

    def factory(*args, linked_rows=False, weighted=False, **kwargs):
        torsion = kwargs.get("torsion", False)
        inputs = partial(get_direct_parallel_inputs, layout, patch=linked_rows, weighted=weighted, torsion=torsion)
        return original(*args, row_inputs=inputs, linked_rows=linked_rows, weighted=weighted, **kwargs)

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
        dev, worlds, capacity = solver.model.device, solver.forest.world_count, solver.placement.dense_max_constraints
        self.data, self.prefix = KineticRows(), PrefixInput()
        self.data.first = solver.placement.dense_first
        self.data.rowkeys = SolverArrays(None, dev).array(KINETIC_ROWS[0], {"world": worlds, "row": capacity})
        self.maps = field_group.upload(RowMaps, dev, groups=groups, starts=starts, arts=arts)
        # Each world's holder this step: 1 the tiers (their rows walked in the kernel), 0 the one writer.
        self.owner = wp.ones(worlds, dtype=wp.int32, device=dev)
        self.contact_input = None
        # The placeholders of absent prefix tables, allocated before any capture.
        solver._arrays.placeholder(wp.int32)
        solver._arrays.placeholder(wp.float32)
        self._topology = TopologyPin.read(solver.model, TOPOLOGY)
        # Persistent patch friction: the rows carry the patch law, and the patch link owns parents and coefficients.
        self.patch = bool(solver.laws.friction_anchors_enabled)
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
            PrefixInput,
            self.prefix,
            solver=s,
            placement=s.placement,
            factor=s.factor,
            laws=s.laws,
            options=s.options,
            model=s.model,
            maps=m,
            state_in=state_in,
            dt=dt,
            **absent,
        )
        self.contact_input = None
        self.data.maps, self.data.prefix = m, p
        prefix = prefix_kernel(self.prefix_layout, PrefixInput, KineticRows)
        wp.launch_tiled(prefix, dim=[s.forest.world_count], inputs=[p, self.data], block_dim=32, device=s.model.device)

    def produce_contacts(self, contact_input):
        """Bind the step's contact input; :meth:`finish_rows` writes the rows once the counts are final."""
        self.contact_input = contact_input
        self.data.contact = self.contact_input
        # Like the other row producers, bind the anchor coefficient and the patch view (which
        # contact-pool growth replaces) every step.
        self.law = patch_row_law(self.solver)

    def finish_rows(self):
        """Scatter the contact keys, then hand the general worlds' rows to the one writer (:func:`keyed_general_units`).

        Under patch friction the keys carry each row's type, parent and undivided coefficient, and the general
        worlds' records the same; the patch link, which runs after every row producer, links and divides them.
        """
        s, layout = self.solver, self.layout
        na = layout.primary
        if self.contact_input is not None:
            pool = s.placement.contact_pool
            pool.launch(
                scatter_contact_keys,
                pool.launch_rows(),
                inputs=[self.contact_input, self.data, self.law],
                extent=self.contact_input.count,
                max_blocks=s.placement.contact_capacity_blocks,
            )
        wp.launch(
            keyed_general_units,
            dim=s.forest.world_count,
            inputs=[
                self.data,
                s.placement.constraint_count,
                s.placement.mf_constraint_count,
                layout.parallel_rows,
                na,
                layout.primary_offset,
                layout.secondary_offset,
                s.placement.dense_first,
                s.forest.articulation_dof_start,
            ],
            outputs=[s.placement.row_sides, s.placement.unit_source, self.owner],
            device=s.model.device,
        )

    def parallel_arguments(self):
        """Return the row operands the keyed fallback and parallel tiers take: the rows and the patch law."""
        return [self.data, self.law] if self.patch else [self.data]


def create_owner(solver):
    """Admit one bounded complete numerical law for the rows of the in-kernel parallel tiers' class (case K4): the
    laws keyed rows implement (``route_table.CAPABILITIES``), then their fit."""
    supported = (
        solver.routes.case == "K4"
        and capable("keyed_rows", solver.laws.set) is None
        and (solver.placement.hinv_jt_writes_world or solver.placement.jy_world_aliased)
    )
    if not supported:
        return None
    try:
        return KeyedRows(solver)
    except ValueError:
        return None
