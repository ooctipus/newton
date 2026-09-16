#!/usr/bin/env bash
# Reuse the fixed paired protocol against the retained performance reference.
# Usage: bash REPRO_CONTACT_ISLANDS_20260916.sh PINNED_WORKTREE NEW_OUTPUT_DIR
set -euo pipefail
test "$#" -eq 2
candidate_root="$1"
git -C "$candidate_root" diff --quiet
git -C "$candidate_root" diff --cached --quiet
uv run --no-project \
    --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
    python "$candidate_root/tools/fpgs_bench/run_contact_islands.py" \
    --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
    --baseline /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 \
    --candidate "$candidate_root" --output-dir "$2" \
    --task keyboard-so101 --gpus 0 1 --rounds 1 --num-envs 4096 \
    --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph \
    --baseline-env FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 \
    --candidate-env FEATHER_PGS_SPARSE_CONTACT_DIRECT=1 \
    --baseline-env FEATHER_PGS_PRISMATIC_PUBLICATION=1 \
    --candidate-env FEATHER_PGS_PRISMATIC_PUBLICATION=1 \
    --baseline-env FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 \
    --candidate-env FEATHER_PGS_COMPACT_CONTACT_BOUNDARY=1 \
    --baseline-env FEATHER_PGS_PRISMATIC_LINEAR_STATE=1 \
    --candidate-env FEATHER_PGS_PRISMATIC_LINEAR_STATE=1 \
    --baseline-env NEWTON_NARROW_PHASE_THREADS_X=4 \
    --candidate-env NEWTON_NARROW_PHASE_THREADS_X=4 \
    --baseline-env FEATHER_PGS_SLEEPING=0 \
    --baseline-env FEATHER_PGS_AWAKE_PIPELINE=0 \
    --baseline-env FEATHER_PGS_CONTACT_ISLANDS=0 \
    --candidate-env FEATHER_PGS_SLEEPING=1 \
    --candidate-env FEATHER_PGS_AWAKE_PIPELINE=1 \
    --candidate-env FEATHER_PGS_CONTACT_ISLANDS=1 \
    --candidate-env FEATHER_PGS_SLEEPING_DIAGNOSTICS=1 \
    --capacity keyboard-so101:fpgs:dense_max_constraints=704 \
    --capacity keyboard-so101:fpgs:rigid_contact_max=147456 \
    --capacity keyboard-so101:fpgs:broad_phase_output_max=57344
