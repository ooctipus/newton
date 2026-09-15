#!/usr/bin/env bash
# Requires the pinned local Lab environment and inherited benchmark artifacts.
# Usage: bash REPRO_CACHED_WORLD_DYNAMICS_20260915.sh whole|node PINNED_WORKTREE NEW_OUTPUT_DIR
set -euo pipefail
test "$#" -eq 3
candidate_root="$2"
test "$(git -C "$candidate_root" rev-parse HEAD)" = f5086ec92c6559435bb37b6a24668268647dac31
case "$1" in
    whole) trace_mode=graph; profile_steps=40 ;;
    node) trace_mode=node; profile_steps=3 ;;
    *) exit 2 ;;
esac
uv run --no-project \
    --python /home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python \
    python /tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py \
    --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
    --baseline /home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915 \
    --candidate "$candidate_root" --output-dir "$3" \
    --task anymald --gpus 0 1 --rounds 1 --num-envs 16384 \
    --warmup-steps 200 --steps 40 --profile-steps "$profile_steps" --trace-mode "$trace_mode" \
    --baseline-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
    --candidate-env FEATHER_PGS_SIMPLE_WORLD_ZERO=1 \
    --baseline-env FEATHER_PGS_FUSED_K1=1 \
    --candidate-env FEATHER_PGS_FUSED_K1=1 \
    --baseline-env FEATHER_PGS_FUSED_K1_CACHE=0 \
    --candidate-env FEATHER_PGS_FUSED_K1_CACHE=1 \
    --capacity anymald:fpgs:dense_max_constraints=72 \
    --capacity anymald:fpgs:rigid_contact_max=212992 \
    --capacity anymald:fpgs:broad_phase_output_max=294912
