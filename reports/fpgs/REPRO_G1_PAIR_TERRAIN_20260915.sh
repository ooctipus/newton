#!/usr/bin/env bash
# Reproduce the complete paired discovery in the original shared workspace.
# This is an UNPROMOTED LOSS, not the retained optimization recipe.
# Requires the pinned local checked adapter and fixed Lab interpreter named below.
# Candidate final report commits preserve measured a68ad0dc runtime bytes.
# Both GPUs must be idle; the original checked driver enforces this.
set -euo pipefail
: "${1:?Pass a fresh output directory outside all source trees}"
exec 'uv' 'run' \
  '--no-project' \
  '--python' '/home/octi/Projects/IsaacLab.wt/contact-reset-20260913/.venv/bin/python' 'python' '/tmp/fpgs-g1-pair-terrain-checked-lDHRWzAJ/run_live_sparse_checked.py' \
  '--isaaclab' '/home/octi/Projects/IsaacLab.wt/contact-reset-20260913' \
  '--baseline' '/home/octi/Projects/newton-fpgs-keyboard-linear-state-20260915' \
  '--candidate' '/home/octi/Projects/newton-fpgs-g1-pair-terrain-20260915' \
  '--task' 'g1' \
  '--gpus' '0' '1' \
  '--rounds' '1' \
  '--seed' '0' \
  '--num-envs' '16384' \
  '--warmup-steps' '200' \
  '--steps' '40' \
  '--profile-steps' '40' \
  '--trace-mode' 'graph' \
  '--capacity' 'g1:fpgs:dense_max_constraints=100' \
  '--capacity' 'g1:fpgs:rigid_contact_max=294912' \
  '--capacity' 'g1:fpgs:broad_phase_output_max=49152' \
  '--capacity' 'g1:fpgs:max_triangle_pairs=1769472' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_CELL_REJECT=1' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_FINITE_QUERY=1' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS=1' \
  '--baseline-env' 'NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_ANALYTIC_MANIFOLD=0' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_FACTOR=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_PACKETS=0' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_PARALLEL_LIMITS=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_LEVEL_UPDATE=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_CONTACT_BLOCK=0' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_METRIC_TANGENTS=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_SMALL_STEP=0' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_LAZY_COMPLIANCE=0' \
  '--baseline-env' 'FEATHER_PGS_G1_KINETIC_STATE=1' \
  '--baseline-env' 'FEATHER_PGS_SPARSE_PRESENT_PORTS=0' \
  '--baseline-env' 'NEWTON_HEIGHTFIELD_PAIR_REDUCER=0' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_CELL_REJECT=1' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_FINITE_QUERY=1' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS=1' \
  '--candidate-env' 'NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_ANALYTIC_MANIFOLD=0' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_FACTOR=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_PACKETS=0' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_PARALLEL_LIMITS=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_LEVEL_UPDATE=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_CONTACT_BLOCK=0' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_METRIC_TANGENTS=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_SMALL_STEP=0' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_LAZY_COMPLIANCE=0' \
  '--candidate-env' 'FEATHER_PGS_G1_KINETIC_STATE=1' \
  '--candidate-env' 'FEATHER_PGS_SPARSE_PRESENT_PORTS=0' \
  '--candidate-env' 'NEWTON_HEIGHTFIELD_PAIR_REDUCER=1' \
  '--output-dir' "$1"
