# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The ordinary evaluator: the tree factor's stages as serial passes, for the solvers the held tree declines.

:class:`OrdinaryTree` runs behind :class:`.tree_factor.TreeFactor`'s calls: FK/ID (:data:`.tree_passes.fk_id`),
the joint forces with the drives in one pass, the CRBA with K added after the assembly, the dense factor per size
group (diagonal, tiles or loop), the triangular solve per size group (diagonal, tiles or loop) and ``v_hat``, then
``qdd`` from the solved velocity, the integration and the FK publish. Each stage follows the step plan's route of its
size (:class:`.step_plan.StepPlan`). D deletes this module once the held tree closes its gaps.
"""

from __future__ import annotations

from functools import cache

import warp as wp

from ...sim import State
from ...sim.articulation import eval_fk
from . import field_group, response_builds_dense
from .kernel_template import python_kernel
from .kernels import (
    apply_augmented_mass_diagonal_grouped,
    compute_velocity_predictor,
    eval_rigid_tau_and_augmented_drives,
    gather_tau_to_groups,
    integrate_generalized_joints,
    remove_free_root_transport_from_qdd,
    scatter_qdd_from_groups,
    solve_diagonal_mass,
    trisolve_loop,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
)
from .tree_passes import (
    SERIAL_BLOCK_DIM,
    DynamicsData,
    composite_inertia,
    crba_fill,
    fk_id,
    inverse_dynamics_tau,
    refresh_body_inertia,
)

__all__ = ["OrdinaryTree", "triangular_solve_kernel"]


@cache
def triangular_solve_kernel(n_dofs: int, device_arch: str, tile_threads: int = 64) -> wp.Kernel:
    """Build specialized triangular solve kernel for given DOF count.

    Solves H x = b with the leaf-first factor ``P H P^T = L L^T`` by tiled forward and backward substitution, the
    coordinates reversed by indexed loads and stores.
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    TILE_DOF_LAST = wp.constant(int(n_dofs) - 1)

    def trisolve_tiled_template(
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
        qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    ):
        idx = wp.tid()
        reverse = wp.tile_arange(TILE_DOF_LAST, -1, -1, dtype=int)
        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        tau_tile = wp.tile_load_indexed(tau_group[idx], indices=reverse, shape=(TILE_DOF_LOCAL, 1), axis=0)

        # Forward substitution: L * z = P tau
        z_tile = wp.tile_lower_solve(L_tile, tau_tile)

        # Backward substitution: L^T * (P qdd) = z
        Lt_tile = wp.tile_transpose(L_tile)
        qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

        wp.tile_store_indexed(qdd_group[idx], indices=reverse, t=qdd_tile, axis=0)

    return python_kernel(trisolve_tiled_template, f"trisolve_tiled_{n_dofs}_bd{tile_threads}")


class OrdinaryTree:
    """The serial passes behind :class:`.tree_factor.TreeFactor`'s stage calls (see the module docstring).

    It reads the solver's mass matrices ``H_by_size`` (double-buffered with the Jacobians), its velocity work vectors
    and the step plan's routes; everything else is the tree factor's.
    """

    def __init__(self, solver):
        self.solver = solver
        self.model = solver.model
        self.forest = solver.forest
        self.tree = solver.factor
        self.cholesky_kernels_by_size: dict[int, wp.Kernel | None] = {}
        self.triangular_solve_kernels_by_size: dict[int, wp.Kernel | None] = {}

    def build_kernels(self, plan) -> None:
        """Build the stage-2 and stage-3 kernels the plan's tables launch, and only those (one entry per size)."""
        arch, threads = self.model.device.arch, self.solver.tile_threads

        def routed(stage, size, route):
            entry = getattr(plan, stage).get(size)
            return entry is not None and entry.name == route

        sizes = self.forest.size_groups
        self.cholesky_kernels_by_size = {
            size: response_builds_dense.factor_tiles_kernel(size, threads) if routed("factor", size, "tiles") else None
            for size in sizes
        }
        self.triangular_solve_kernels_by_size = {
            size: triangular_solve_kernel(size, arch, threads) if routed("trisolve", size, "tiles") else None
            for size in sizes
        }

    def dynamics(self, state: State | None, control=None, **values) -> DynamicsData:
        """Bind the dynamics passes' struct (:data:`.tree_passes.DynamicsData`).

        ``state`` holds the coordinates the passes read and receives the poses and COM twists they publish
        (``None`` for passes that read neither); the tree factor is the augmented state. ``values`` are the launch's
        switches (zero unless given) and field overrides, such as the prescaled velocities of ``joint_qd``.
        """
        arrays = self.tree._arrays
        values.setdefault("write_inertia_records", 0)
        if state is None:
            for name in ("joint_q", "joint_qd"):
                values.setdefault(name, arrays.placeholder(wp.float32))
            values.setdefault("body_q", arrays.placeholder(wp.transform))
            values.setdefault("body_qd", arrays.placeholder(wp.spatial_vector))
            values.setdefault("body_f", arrays.placeholder(wp.spatial_vector))
        elif "body_f" not in values:
            values["body_f"] = state.body_f if state.body_count else arrays.placeholder(wp.spatial_vector)
        if control is None:
            values.setdefault("joint_f", arrays.placeholder(wp.float32))
        return field_group.bind(
            DynamicsData,
            model=self.model,
            factor=self.tree,
            forest=self.forest,
            state=state,
            aug=self.tree,
            control=control,
            **values,
        )

    def fk_id(self, state_in: State, state_out: State, qd: wp.array, refresh_composite: bool) -> None:
        """FK/ID over each articulation (a refresh step places every body's inertia record for the composite
        inertia), then the output state's body twists."""
        model = self.model
        dynamics = self.dynamics(state_in, joint_qd=qd, write_inertia_records=int(refresh_composite))
        wp.launch(
            fk_id, dim=model.articulation_count, inputs=[dynamics], block_dim=SERIAL_BLOCK_DIM, device=model.device
        )
        if model.body_count:
            wp.launch(
                update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[
                    self.tree.body_v_s,
                    state_in.body_q,
                    model.body_com,
                    self.forest.body_to_articulation,
                    self.tree.articulation_origin,
                ],
                outputs=[state_out.body_qd],
                device=model.device,
            )

    def joint_tau(self, state_in: State, control, dt: float) -> None:
        """Populate ``joint_tau`` and apply the effort-limit clamp.

        Torque-bucket ownership after this routine returns:

        - ``joint_tau`` holds the rigid / passive / Coriolis /
          gravity / external / :attr:`~newton.Control.joint_f` contribution
          summed with the clamped explicit-PD drive contribution ``u0``.
        - The implicit-PD drive response is carried by ``aug_row_K``
          and realized through ``H_tilde^{-1}`` during the linear solve.

        :attr:`~newton.Model.joint_effort_limit` is always applied as
        an **actuator-only** clamp on ``u0`` *before* it is added to
        ``joint_tau``; the rigid / passive / external bucket is left uncapped
        (MuJoCo ``actuatorfrcrange`` / PhysX drive ``maxForce`` convention).
        """
        model, tree = self.model, self.tree
        if self.forest.articulation_max_dofs > 0:
            wp.launch(
                eval_rigid_tau_and_augmented_drives,
                dim=model.articulation_count,
                inputs=[
                    self.dynamics(state_in, control),
                    self.forest.articulation_dof_count,
                    model.joint_target_q_start,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    model.joint_effort_limit,
                    self.forest.articulation_max_dofs,
                    dt,
                ],
                outputs=[tree.aug_row_counts, tree.aug_row_dof_index],
                block_dim=SERIAL_BLOCK_DIM,
                device=model.device,
            )
            return
        # Evaluate joint torques. After this launch `joint_tau` owns
        # the rigid / passive / Coriolis / gravity / external /
        # `control.joint_f` bucket only; the actuator-drive bucket has
        # not been added yet.
        wp.launch(
            inverse_dynamics_tau,
            dim=model.articulation_count,
            inputs=[self.dynamics(state_in, control)],
            block_dim=SERIAL_BLOCK_DIM,
            device=model.device,
        )

    def crba(self, global_flag: int) -> None:
        """The composite inertia and H of the articulations the mass mask selects, K added after the assembly."""
        model, tree, solver = self.model, self.tree, self.solver
        # The composite and every mass consumer read inertia records about the origin: the records of
        # articulations a reuse step's mask selects are placed at their current poses first.
        dynamics = self.dynamics(None)
        if not global_flag:
            wp.launch(
                refresh_body_inertia, dim=model.joint_count, inputs=[dynamics], block_dim=128, device=model.device
            )

        # The composite inertia of the articulations the mass mask selects.
        wp.launch(
            composite_inertia,
            dim=model.articulation_count,
            inputs=[dynamics],
            device=model.device,
            block_dim=SERIAL_BLOCK_DIM,
        )

        # Zeroing H is hygiene, not a correctness requirement: H is only read
        # by the Cholesky kernels, which early-exit when mass_update_mask is 0,
        # and a masked rebuild never relies on a zeroed buffer because
        # crba_fill STORES every structural nonzero (full ancestor
        # chain incl. the whole diagonal) before apply_augmented_mass_diagonal
        # _grouped ADDS K onto that freshly stored diagonal under the same
        # mask. Structural zeros are written by no kernel and stay zero from
        # allocation. Skipping the memset on global_flag==0 steps therefore
        # leaves limit-change refreshes (device-side mask) exact while saving
        # the per-step memset.
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            if solver._H_bufs is None and global_flag:  # not double-buffered
                solver.H_by_size[size].zero_()
            wp.launch(
                crba_fill,
                dim=int(n_arts * size),
                inputs=[
                    dynamics,
                    tree.body_inertia_terms if size in self.forest.free_body_inertia_sizes else tree.body_I_c,
                    self.forest.group_to_art[size],
                    size,
                ],
                outputs=[solver.H_by_size[size]],
                device=model.device,
                block_dim=128,
            )
            # K after the assembly.
            wp.launch(
                apply_augmented_mass_diagonal_grouped,
                dim=n_arts,
                inputs=[
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                    self.forest.articulation_max_dofs,
                    tree.mass_update_mask,
                    tree.aug_row_counts,
                    tree.aug_row_dof_index,
                    tree.aug_row_K,
                ],
                outputs=[solver.H_by_size[size]],
                device=model.device,
            )

    def factor(self, size: int) -> None:
        """Factor one size's mass matrix along the plan's stage-2 route: none for compact, sparse and fused sizes."""
        route = self.solver._plan.factor[size].name
        tree, solver = self.tree, self.solver
        n_arts = self.forest.n_arts_by_size[size]
        operands = [
            solver.H_by_size[size],
            tree.R_by_size[size],
            self.forest.group_to_art[size],
            tree.mass_update_mask,
        ]
        if route == "diagonal":
            # An articulation group known to have diagonal generalized inertia.
            wp.launch(
                response_builds_dense.factor_coordinates_kernel(),
                dim=n_arts * size,
                inputs=[*operands, size],
                outputs=[tree.L_by_size[size]],
                device=self.model.device,
            )
        elif route == "tiles":
            wp.launch_tiled(
                self.cholesky_kernels_by_size[size],
                dim=[n_arts],
                inputs=operands,
                outputs=[tree.L_by_size[size]],
                block_dim=solver.tile_threads,
                device=self.model.device,
            )
        elif route == "loop":
            wp.launch(
                response_builds_dense.factor_articulations_kernel(),
                dim=n_arts,
                inputs=[*operands, size],
                outputs=[tree.L_by_size[size]],
                device=self.model.device,
            )

    def predict(self, state_in: State, qd: wp.array, dt: float) -> None:
        """Stage 3: ``qdd = H^-1 tau`` per size group along the plan's route, then ``v_hat`` and the free-root
        correction."""
        tree = self.tree
        tree.joint_qdd.zero_()
        for size, context in tree.for_sizes():
            with context:
                self._trisolve(size)
        self.velocity_predictor(state_in, qd, dt)

    def velocity_predictor(self, state_in: State, qd: wp.array, dt: float) -> None:
        """``v_hat = qd + dt qdd`` (kinematic coordinates keep ``qd``), then the free-root correction."""
        model, tree, solver = self.model, self.tree, self.solver
        if not model.joint_count:
            return
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[
                qd,
                tree.kinematic_dof_mask,
                dt,
            ],
            outputs=[tree.joint_qdd, solver.v_hat],
            device=model.device,
        )
        tree.correct_free_root_velocity(state_in, qd, dt, solver.v_hat)

    def _trisolve(self, size: int) -> None:
        """Solve one size's unconstrained acceleration along the plan's stage-3 route."""
        route = self.solver._plan.trisolve[size].name
        model, tree = self.model, self.tree
        n_arts = self.forest.n_arts_by_size[size]
        if route == "diagonal":
            # A diagonal generalized inertia's inverse applied to the joint forces.
            wp.launch(
                solve_diagonal_mass,
                dim=n_arts * size,
                inputs=[
                    tree.L_by_size[size],
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                    tree.joint_tau,
                ],
                outputs=[tree.joint_qdd],
                device=model.device,
            )
        elif route == "tiles":
            wp.launch(
                gather_tau_to_groups,
                dim=n_arts,
                inputs=[
                    tree.joint_tau,
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                ],
                outputs=[tree.tau_by_size[size]],
                device=model.device,
            )
            wp.launch_tiled(
                self.triangular_solve_kernels_by_size[size],
                dim=[n_arts],
                inputs=[
                    tree.L_by_size[size],
                    tree.tau_by_size[size],
                ],
                outputs=[tree.qdd_by_size[size]],
                block_dim=self.solver.tile_threads,
                device=model.device,
            )
            wp.launch(
                scatter_qdd_from_groups,
                dim=n_arts,
                inputs=[
                    tree.qdd_by_size[size],
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                ],
                outputs=[tree.joint_qdd],
                device=model.device,
            )
        else:
            wp.launch(
                trisolve_loop,
                dim=n_arts,
                inputs=[
                    tree.L_by_size[size],
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                    tree.joint_tau,
                ],
                outputs=[tree.joint_qdd],
                device=model.device,
            )

    def finish(self, state_in: State, state_out: State, dt: float) -> None:
        """Stage 7: ``qdd`` from the solved velocity, then the integration and the output state's kinematics."""
        self.update_qdd(state_in, dt)
        self.integrate(state_in, state_out, dt)

    def update_qdd(self, state_in: State, dt: float) -> None:
        """``qdd`` from the solved velocity ``v_out``, the free-root transport removed."""
        model, tree = self.model, self.tree
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, tree.kinematic_dof_mask, 1.0 / dt],
            outputs=[self.solver.v_out, tree.joint_qdd],
            device=model.device,
        )
        # Invert jcalc_integrate's free-root transport so it realizes the solved velocity exactly (no launch for
        # fixed-base models).
        if self.forest.free_root_joint_count:
            wp.launch(
                remove_free_root_transport_from_qdd,
                dim=self.forest.free_root_joint_count,
                inputs=[
                    self.forest.free_root_joint_indices,
                    model.joint_qd_start,
                    tree.kinematic_joint_mask,
                    state_in.joint_qd,
                ],
                outputs=[tree.joint_qdd],
                device=model.device,
            )

    def integrate(self, state_in: State, state_out: State, dt: float) -> None:
        """Integrate every joint from ``qdd``, then publish the output state's kinematics."""
        model, tree = self.model, self.tree
        if model.joint_count:
            wp.launch(
                kernel=integrate_generalized_joints,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    tree.kinematic_joint_mask,
                    model.joint_dof_dim,
                    model.body_com,
                    model.joint_X_c,
                    state_in.joint_q,
                    state_in.joint_qd,
                    tree.joint_qdd,
                    dt,
                    self.solver.options.angular_damping,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )
            # Ordinary dynamics derive the next step's kinematics from the state.
            eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
