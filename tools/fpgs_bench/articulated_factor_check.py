# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded fixed-state factor/action diagnostic; never advance a live trajectory.

The optional CUDA invocation is a root-owned component check, not whole-task
timing. Saved source epochs and the original held mass remain distinct from
current right-hand sides. No collision buffers or solver budgets are changed.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import warp as wp


def sha(path):
    """Hash immutable source or snapshot bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dense_mass(parents, slots, inertia, motion, diagonal, counts):
    """Assemble the physical generalized mass independently in FP64."""
    arts, bodies = parents.shape
    dofs = diagonal.shape[1]
    result = np.zeros((arts, dofs, dofs), np.float64)
    for art in range(arts):
        for body in range(int(counts[art])):
            jac = np.zeros((6, dofs), np.float64)
            joint = body
            visited = set()
            while joint >= 0:
                if joint in visited or joint >= bodies:
                    raise ValueError("Invalid ancestor topology")
                visited.add(joint)
                slot = int(slots[art, joint])
                if slot >= 0:
                    jac[:, slot] = motion[art, joint]
                joint = int(parents[art, joint])
            result[art] += jac.T @ inertia[art, body].astype(np.float64) @ jac
        result[art, np.arange(dofs), np.arange(dofs)] += diagonal[art]
    return result


def spatial_from_compact(mass, terms):
    """Reconstruct current individual inertia from the original compact12 owner."""
    mass, terms = np.asarray(mass, np.float64), np.asarray(terms, np.float64)
    if terms.shape != (*mass.shape, 12):
        raise ValueError("Expected one mass and twelve current terms per body")
    result = np.zeros((*mass.shape, 6, 6), np.float64)
    result[..., :3, :3] = mass[..., None, None] * np.eye(3)
    x, y, z = np.moveaxis(terms[..., :3], -1, 0)
    cross = np.zeros((*mass.shape, 3, 3), np.float64)
    cross[..., 0, 1], cross[..., 0, 2] = -z, y
    cross[..., 1, 0], cross[..., 1, 2] = z, -x
    cross[..., 2, 0], cross[..., 2, 1] = -y, x
    result[..., :3, 3:] = -mass[..., None, None] * cross
    result[..., 3:, :3] = mass[..., None, None] * cross
    result[..., 3:, 3:] = terms[..., 3:].reshape(*mass.shape, 3, 3)
    return result


def action_error(matrix, rhs, actual):
    """Gate normalized equation residual, retaining ordinary forward-error diagnostics."""
    h, b, x = (np.asarray(value, np.float64) for value in (matrix, rhs, actual))
    if not all(np.isfinite(value).all() for value in (h, b, x)):
        raise AssertionError("Nonfinite mass/action input or output")
    reference = np.linalg.solve(h, b.swapaxes(1, 2)).swapaxes(1, 2)
    residual = np.einsum("aij,arj->ari", h, x) - b
    scale = np.einsum("aij,arj->ari", abs(h), abs(x)) + abs(b)
    normalized = np.linalg.norm(residual, axis=-1) / np.maximum(np.linalg.norm(scale, axis=-1), 1e-30)
    limit = 128 * 2.0**-24 / (1 - 128 * 2.0**-24)
    result = {
        "max_normalized_residual": float(normalized.max(initial=0)),
        "residual_limit_gamma128": limit,
        "max_absolute_action_error": float(np.max(abs(x - reference), initial=0)),
        "minimum_mass_eigenvalue": float(np.linalg.eigvalsh(h).min()),
    }
    if result["minimum_mass_eigenvalue"] <= 0 or result["max_normalized_residual"] > limit:
        raise AssertionError(f"Held mass action failed: {result}")
    return result


def allocate_factor(module, fixture, device):
    """Bind independent topology and factor output allocations to the actual ABI."""
    parents, slots = fixture["parents"], fixture["slots"]
    arts, bodies = parents.shape
    dofs = fixture["diagonal"].shape[1]
    factor = module.ArticulatedFactorData()
    factor.art_ids = wp.array(np.arange(arts, dtype=np.int32), dtype=int, device=device)
    factor.body_ids = wp.array(np.arange(arts * bodies, dtype=np.int32).reshape(arts, bodies), dtype=int, device=device)
    factor.parents = wp.array(parents, dtype=int, device=device)
    factor.dof_slots = wp.array(slots, dtype=int, device=device)
    factor.global_dofs = wp.array(
        np.where(slots >= 0, slots + np.arange(arts)[:, None] * dofs, -1).astype(np.int32), dtype=int, device=device
    )
    factor.counts = wp.array(fixture["counts"], dtype=int, device=device)
    factor.S = wp.zeros((arts, bodies, 6), dtype=float, device=device)
    factor.U = wp.zeros((arts, bodies, 6), dtype=float, device=device)
    factor.invD = wp.zeros((arts, bodies), dtype=float, device=device)
    factor.valid = wp.zeros(arts, dtype=int, device=device)
    return factor


def factor_inputs(fixture, device):
    """Separate current public inertia/motion/drive inputs from persistent factor state."""
    arts, _bodies = fixture["parents"].shape
    dofs = fixture["diagonal"].shape[1]
    motion = np.zeros((arts, dofs, 6), np.float32)
    for art in range(arts):
        for body in range(int(fixture["counts"][art])):
            slot = fixture["slots"][art, body]
            if slot >= 0:
                motion[art, slot] = fixture["motion"][art, body]
    return [
        wp.array(fixture["inertia"].reshape(-1, 6, 6), dtype=wp.spatial_matrix, device=device),
        wp.array(motion.reshape(-1, 6), dtype=wp.spatial_vector, device=device),
        wp.array(fixture["diagonal"].reshape(-1), dtype=float, device=device),
        wp.array(fixture["refresh"], dtype=int, device=device),
    ]


def run_factor(module, fixture, device, factor=None):
    """Run the real native factor kernel once with no synthetic solver replacement."""
    factor = allocate_factor(module, fixture, device) if factor is None else factor
    arts, bodies = fixture["parents"].shape
    dofs = fixture["diagonal"].shape[1]
    wp.launch_tiled(
        module.get_factor_kernel(bodies, dofs),
        dim=[arts],
        inputs=[factor, *factor_inputs(fixture, device)],
        block_dim=32,
        device=device,
    )
    return factor


def run_action(module, factor, fixture, rhs, device, row_counts=None):
    """Apply the actual held factor to a bounded current set of independent RHS rows."""
    arts, rows, dofs = rhs.shape
    source = wp.array(rhs, dtype=float, device=device)
    output = wp.full(rhs.shape, 12345.0, dtype=float, device=device)
    counts = wp.array(np.full(arts, rows, np.int32) if row_counts is None else row_counts, dtype=int, device=device)
    kernel = module.get_apply_kernel(fixture["parents"].shape[1], dofs)
    wp.launch_tiled(kernel, dim=[arts, rows], inputs=[factor, source, output, counts], block_dim=32, device=device)
    return output.numpy()


def run_live_factor(module, fixture, device, factor=None, *, compact_source=True, poison_unused=False):
    """Exercise the actual compact/full live source selector and drive-R binding."""
    factor = allocate_factor(module, fixture, device) if factor is None else factor
    arts, bodies = fixture["parents"].shape
    dofs = fixture["diagonal"].shape[1]
    public = factor_inputs(fixture, device)
    terms = fixture["compact_terms"].copy()
    body_mass = fixture["body_mass"].copy()
    if poison_unused:
        if compact_source:
            public[0] = wp.full((arts * bodies,), wp.spatial_matrix(float("nan")), device=device)
        else:
            terms[:] = np.nan
            body_mass[:] = np.nan
    wp.launch_tiled(
        module.get_live_factor_kernel(bodies, dofs),
        dim=[arts],
        inputs=[
            factor,
            public[0],
            public[1],
            wp.array(fixture["drive_r"], dtype=float, device=device),
            wp.array(fixture["drive_rows"].reshape(-1), dtype=int, device=device),
            wp.array(fixture["drive_k"], dtype=float, device=device),
            public[3],
            wp.array(terms.reshape(-1, 12), dtype=float, device=device),
            wp.array(body_mass.reshape(-1), dtype=float, device=device),
            int(compact_source),
        ],
        block_dim=32,
        device=device,
    )
    return factor


def load_kuka(path, expected_sha, limit=512):
    """Load only the actual fixed-base hand fields needed by the component check."""
    if sha(path) != expected_sha:
        raise RuntimeError("Saved Kuka snapshot changed")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata_json"].item())
        if metadata["dt"] != 1 / 240 or metadata["pgs_iterations"] != 8:
            raise RuntimeError("Saved Kuka timestep/iteration allowance differs")
        art_ids = saved["group_to_art_23"][:limit].astype(np.int32)
        starts = saved["full_model_articulation_start"][art_ids]
        ends = saved["solve_articulation_joint_end"][art_ids]
        if np.any(ends - starts != 30):
            raise RuntimeError("Expected the original 30-joint/23-DOF Kuka hand")
        joints = starts[:, None] + np.arange(30)
        body_ids = saved["full_model_joint_child"][joints]
        ancestors = saved["full_model_joint_ancestor"][joints]
        parents = np.where(ancestors < 0, -1, ancestors - starts[:, None]).astype(np.int32)
        joint_dofs = saved["full_model_joint_dof_dim"][joints].sum(axis=-1)
        first_dof = saved["full_model_joint_qd_start"][starts]
        global_dofs = saved["full_model_joint_qd_start"][joints]
        slots = np.where(joint_dofs == 1, global_dofs - first_dof[:, None], -1).astype(np.int32)
        if np.any((joint_dofs != 0) & (joint_dofs != 1)):
            raise RuntimeError("Unsupported joint in fixed-base factor fixture")
        dofs = first_dof[:, None] + np.arange(23)
        motion = saved["post3_aug_joint_S_s"][np.maximum(global_dofs, 0)].copy()
        motion[joint_dofs == 0] = 0
        # Global compact refresh deliberately leaves body_I_s stale.  These
        # compact12 terms, not that legacy output allocation, own current I.
        body_mass = saved["full_model_body_mass"][body_ids].copy()
        compact_terms = saved["post3_solver__body_inertia_terms"][body_ids].copy()
        inertia = spatial_from_compact(body_mass, compact_terms).astype(np.float32)
        drive_r = saved["post3_R_23"][: len(art_ids)].copy()
        diagonal = drive_r.copy()
        rows = saved["post3_solver__augmented_drive_row_by_dof"][dofs]
        drive_k = saved["post3_solver_aug_row_K"].copy()
        k = np.zeros_like(diagonal)
        active = rows >= 0
        k[active] = np.maximum(drive_k[rows[active]], 0)
        diagonal += k
        lower = np.tril(saved["post3_L_23"][: len(art_ids)].astype(np.float64))
        current_j = saved["J_23"][: len(art_ids)]
        worlds = saved["solve_art_to_world"][art_ids]
        row_counts = saved["solve_constraint_count"][worlds]
        selected = np.zeros((len(art_ids), 3, 23), np.float32)
        for art, count in enumerate(row_counts):
            selected[art, : min(3, count)] = current_j[art, : min(3, count)]
        return {
            "parents": parents,
            "slots": slots,
            "counts": np.full(len(art_ids), 30, np.int32),
            "inertia": inertia,
            "compact_terms": compact_terms,
            "body_mass": body_mass,
            "motion": motion,
            "diagonal": diagonal,
            "drive_r": drive_r,
            "drive_rows": rows.copy(),
            "drive_k": drive_k,
            "refresh": saved["post3_solver_mass_update_mask"][art_ids].copy(),
            "reference_h": lower @ lower.swapaxes(-1, -2),
            "rhs": np.concatenate((saved["post3_aug_joint_tau"][dofs][:, None, :], selected), axis=1),
            "step": int(metadata["step"]),
            "dt": metadata["dt"],
            "drive_nonzero": int(np.count_nonzero(k)),
        }


def main(argv=None):
    """Run the two source-bound saved epochs; emit no performance acceptance claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists() or sha(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("Use fresh output and the exact saved manifest")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("complete") is not True or manifest.get("source_guard_pass") is not True:
        raise RuntimeError("Saved task capture did not complete its source guards")
    module = importlib.import_module("newton._src.solvers.feather_pgs.articulated_factor")
    expected_module = Path(__file__).resolve().parents[2] / "newton/_src/solvers/feather_pgs/articulated_factor.py"
    if Path(module.__file__).resolve() != expected_module:
        raise RuntimeError("Set PYTHONPATH to this helper's Newton source checkout")
    source_pins = {str(Path(path).resolve()): sha(path) for path in (__file__, module.__file__)}
    records, factor = [], None
    for capture in manifest["captures"]:
        fixture = load_kuka(capture["path"], capture["sha256"])
        if not records and not np.all(fixture["refresh"]):
            raise RuntimeError("Initial component epoch must actually refresh")
        factor = run_live_factor(module, fixture, args.device, factor, poison_unused=True)
        if not np.all(factor.valid.numpy() == 1):
            raise AssertionError("Native factor rejected a captured articulation")
        rhs = np.concatenate(
            (fixture["rhs"], np.broadcast_to(np.eye(23, dtype=np.float32), (len(fixture["counts"]), 23, 23))), axis=1
        )
        actual = run_action(module, factor, fixture, rhs, args.device)
        records.append(
            {
                "step": fixture["step"],
                "dt": fixture["dt"],
                "iterations": 8,
                "articulations": len(fixture["counts"]),
                "refreshed": int(np.count_nonzero(fixture["refresh"])),
                "drive_nonzero": fixture["drive_nonzero"],
                **action_error(fixture["reference_h"], rhs, actual),
            }
        )
    if len(records) != 2 or records[1]["step"] != records[0]["step"] + 1 or records[1]["refreshed"] != 0:
        raise RuntimeError("Expected exactly two adjacent actual refresh/reuse epochs")
    if any(sha(path) != digest for path, digest in source_pins.items()):
        raise RuntimeError("Diagnostic/native source changed")
    args.output.write_text(
        json.dumps(
            {
                "complete": True,
                "scope": "Saved-input held-factor/action accuracy only; not task trajectory or whole timing",
                "source_pins": source_pins,
                "manifest_sha256": args.expected_manifest_sha256,
                "saved_capture_sources": manifest["sources"],
                "records": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
