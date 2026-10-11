# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live numeric controls for shared publication and current integration ownership."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kuka_joint_world, world_scan_publication
from tools.fpgs_bench import test_world_scan_publication as reference
from tools.fpgs_bench.test_kinetic_live_bindings import bound_call


def fixture(device):
    """Use the live two-world unit topology without archived scene snapshots."""
    solver, owner, _first, _second, _control, _contacts, call = bound_call()
    source = call.next_publication
    data = world_scan_publication.PublicationData()
    values = {}
    for name in world_scan_publication.PublicationData.vars:
        value = getattr(source, name)
        if hasattr(value, "numpy"):
            value = wp.array(value.numpy(), dtype=value.dtype, device=device)
        setattr(data, name, value)
        values["v_new" if name == "v_out" else name] = value
    # The CPU descriptor fixture retains a one-body compact-inertia sentinel;
    # materialization tests must allocate the complete output contract.
    data.body_inertia_terms = wp.zeros((solver.model.body_count, 12), dtype=float, device=device)
    values["body_inertia_terms"] = data.body_inertia_terms
    values["body_I_s"].zero_()
    plan = kuka_joint_world.HostPlan(
        **{name: owner.host_plan.host[name] for name in kuka_joint_world.HostPlan.__dataclass_fields__}
    )
    bundle = SimpleNamespace(
        values=values,
        data=data,
        plan=plan,
        device_plan=plan.device_data(device),
        dimensions=(
            solver.model.joint_dof_count,
            len(data.free_root_joint_indices),
            solver.model.joint_count,
            solver.model.articulation_count,
            solver.model.body_count,
        ),
        device=device,
    )
    values["inv_dt"] = 1.0 / data.dt
    for kernel in reference.CHAIN:
        for argument in kernel.adj.args:
            name = argument.label
            if name not in values:
                source = getattr(solver.model, name, None)
                if source is None:
                    source = getattr(solver, name)
                values[name] = wp.array(source.numpy(), dtype=source.dtype, device=device)
    values["v_new"].assign(np.cos(np.arange(solver.model.joint_dof_count)) * 0.2)
    values["joint_qd"].assign(np.sin(np.arange(solver.model.joint_dof_count)) * 0.1)
    for name in reference.OUTPUTS.keys() - {"v_new", "body_I_s", "body_inertia_terms"}:
        values[name].fill_(0 if name == "fk_id_cache_valid" else np.nan)
    return bundle


class TestSharedPublication(unittest.TestCase):
    def check_publication(self, device):
        for refresh in ((0, 0), (1, 0), (0, 1)):
            candidate, expected = fixture(device), fixture(device)
            for bundle in (candidate, expected):
                for name, value in zip(
                    ("materialize_all_body_inertia", "materialize_body_inertia_terms"), refresh, strict=True
                ):
                    bundle.values[name] = value
                    setattr(bundle.data, name, value)
                reference.change_numeric_inputs(bundle)
                bundle.data.angular_damping = bundle.values["angular_damping"]
            reference.original(expected)
            reference.launch(candidate)
            reference.check_outputs(self, candidate, expected)

    def check_resolved(self, device):
        for mask in ((0, 0), (1, 0), (0, 1), (1, 1)):
            with self.subTest(device=device, mask=mask):
                candidate, expected = fixture(device), fixture(device)
                reference.original(expected)
                # Simulate the current early owner publishing generalized state.
                for index, kernel in enumerate(reference.CHAIN[:3]):
                    wp.launch(
                        kernel,
                        candidate.dimensions[index],
                        [candidate.values[a.label] for a in kernel.adj.args],
                        device=device,
                    )
                before = candidate.values["joint_qd"].numpy()
                poisoned = before.copy()
                poisoned[candidate.plan.dof_ids[np.asarray(mask, bool)].ravel()] += 0.7
                candidate.values["joint_qd"].assign(poisoned)
                resolved = wp.array(mask, dtype=int, device=device)
                kernel = world_scan_publication.get_resolved_kernel(str(wp.get_device(device).arch))
                wp.launch_tiled(
                    kernel,
                    dim=[2],
                    inputs=[candidate.device_plan, candidate.data, resolved],
                    block_dim=32,
                    device=device,
                )
                reference.check_outputs(self, candidate, expected)
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.launch_tiled(
                            kernel,
                            dim=[2],
                            inputs=[candidate.device_plan, candidate.data, resolved],
                            block_dim=32,
                            device=device,
                        )
                    wp.capture_launch(capture.graph)
                    reference.check_outputs(self, candidate, expected)

    def test_publication_cpu(self):
        self.check_publication("cpu")

    def test_resolved_cpu(self):
        self.check_resolved("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_publication_cuda(self):
        self.check_publication("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_resolved_cuda(self):
        self.check_resolved("cuda:0")


if __name__ == "__main__":
    unittest.main()
