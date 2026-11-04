#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Fast landing gate: catches regressions in about 15 minutes of one GPU; the full benchmark runs after landings.
#   gate_fast.sh <label> <head tree> <head sha8> <track tree> <track sha8>
#   GPU=pro (default, RTX PRO 6000) | GPU=gb300; GATE_TASKS overrides the task subset; BIG=0 skips the big scenes;
#   HEAD_PATH / TRACK_PATH=<dir> prepend a directory to that arm's PYTHONPATH (e.g. the frozen gpu_components src each
#   tree was built against, when it differs from the venvs' pin).
#   Machine paths: FPGS_OUT, FPGS_LAB, FPGS_PY, FPGS_GPU_LOCKS, FPGS_PRO_UUID (README.md, "Machine paths").
# 1. Speed and validity: one same-window pass per friction law over the route-covering subset: the 11 tasks that
#    together run every structural solver and collision route of the suite, plus anymald (parallel tiers without a
#    velocity-limit law). Bands 0.97-1.03, written before timing.
# 2. Big-scene smoke: coins and hardware, 128 worlds, patch, first 25 frames (coins' tower and hardware's landing),
#    head then track in one window. Each arm first warms its kernel cache in an untimed run: a run that compiles is
#    slow on every frame.
# Run cuda_gate.py beside it on the other GPU (--overlap-timing on the GB300).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${FPGS_OUT:-/home/octi/Projects/IsaacLab.wt/output/fpgs-ws-20261015}
LAB=${FPGS_LAB:-/home/octi/Projects/IsaacLab.wt/isaaclab-fpgs-keyboard-capacity-20261011}
PY=${FPGS_PY:-/home/octi/Projects/IsaacLab.wt/newton-fpgs-framework-cleanup-20261014/.venv/bin/python}
PRO=${FPGS_PRO_UUID:-GPU-883586b6-3100-0610-81e5-3b4c26f45639}
source "${FPGS_GPU_LOCKS:-/tmp/claude-1002/-home-octi-Projects-IsaacLab-wt/9a31c1a9-57ae-4e84-8704-55d2ba3b5586/scratchpad/gpu_locks.sh}"
label=$1 H=$2 hsha=$3 T=$4 tsha=$5
GATE_TASKS=${GATE_TASKS:-"h1-rough kuka allegro franka-cabinet g1 franka cartpole go2-rough keyboard-so101 ant ur10-reach anymald"}
if [ "${GPU:-pro}" = gb300 ]; then LOCK=gb300_exclusive UUID=$GB300 MEM=${MEM_GB:-200}
else LOCK=gpu_exclusive UUID=$PRO MEM=${MEM_GB:-80}; fi
for pair in "$H:$hsha" "$T:$tsha"; do
  tree=${pair%%:*} want=${pair##*:}
  [ "$(git -C "$tree" rev-parse --short=8 HEAD)" = "$want" ] && [ -z "$(git -C "$tree" status --porcelain)" ] \
    || { echo "$tree moved or dirty (want $want)"; exit 1; }
done
TASKS=$(for t in $GATE_TASKS; do printf -- '--task %s ' "$t"; done)
t0=$(date +%s)
for arm in patch point; do
  l=$label; [ $arm = point ] && l=$label-pt
  python3 -c "import json, sys; json.dump({t: [0.97, 1.03] for t in sys.argv[2:]}, open(sys.argv[1], 'w'), indent=1)" \
    "$OUT/bands_${l}_${arm}.json" $GATE_TASKS
  ARM=""; [ $arm = patch ] && ARM="--unset-solver-attr friction_anchor_beta"
  rm -rf "${OUT:?}/ab-${l:?}-head-1" "${OUT:?}/ab-${l:?}-track-1"
  (cd "$LAB" && FPGS_LAB=$LAB $LOCK uv run --no-sync python "$HERE/run_suite_par.py" --gpu "$UUID" \
    --arm "head,$H,$OUT/ab-$l-head-1,,${HEAD_PATH:-}" --arm "track,$T,$OUT/ab-$l-track-1,,${TRACK_PATH:-}" $TASKS $ARM \
    --jobs 6 --gpu-token on --mem-budget-gb "$MEM" --mem-table "$OUT/bench_memory.json" --seed-cache "${SEED:-auto}" \
    > "$OUT/ab-$l-pass-1.log" 2>&1)
  echo "== $(date +%H:%M) $arm: $(tail -1 "$OUT/ab-$l-pass-1.log")"
  (cd "$OUT" && python3 "$HERE/grade_suite.py" --ref "ab-$l-head-1" --cand "ab-$l-track-1" \
    --rel-bands "bands_${l}_${arm}.json" | cut -c1-110)
done
if [ "${BIG:-1}" = 1 ]; then
  for scene in coins hardware; do
    for side in head track; do
      tree=$H site=${HEAD_PATH:-}; [ $side = track ] && tree=$T site=${TRACK_PATH:-}
      for run in warm timed; do
        name=gate-$label-$scene-$side frames="--frames 25 --warmup 5" tag=$label-$side
        [ $run = warm ] && name=$name-warm frames="--frames 2 --warmup 1" tag=$tag-warm
        $LOCK env CUDA_VISIBLE_DEVICES="$UUID" WARP_CACHE_PATH="$OUT/complex/cache-scale" ${site:+PYTHONPATH=$site} \
          timeout 1800 "$PY" "$HERE/complex/run_complex.py" --scene $scene --law patch --world-count 128 \
          --label "gate-$tag" --out-dir "$OUT/complex/results" --newton "$tree" --allow-live-tree --device cuda:0 \
          $frames --no-save-samples > "$OUT/complex/$name.log" 2>&1
      done
      echo "   $scene $side: $(grep -o 'median [0-9.]* ms/frame.*physics flags [^-]*' "$OUT/complex/gate-$label-$scene-$side.log" | cut -c1-140)"
    done
  done
fi
echo "== $(date +%H:%M) gate $label done in $(( ($(date +%s) - t0) / 60 )) min"
