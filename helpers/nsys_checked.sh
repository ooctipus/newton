#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Match the selected Lab nsys_run.sh capture/export/analyzer interface, replacing
# only its hardcoded Python entrypoint with Newton's checked metadata wrapper.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="${FPGS_BENCH_ISAACLAB:?Set the selected Isaac Lab checkout}"
newton="${FPGS_BENCH_NEWTON:?Set the selected Newton checkout}"
run_sha="${FPGS_BENCH_RUN_SHA256:?Set the parent-pinned Lab run_profiled.py SHA256}"
harness="$repo/scripts/benchmarks/fpgs_profile"
out="${OUT_DIR:?Set a fresh output directory outside source trees}"
name=$1; physics=$2; task=$3; shift 3
wrapper_args=()
while true; do
  case "${1:-}" in
    --broad-phase-output-max)
      wrapper_args+=("$1" "${2:?Supply a positive broad-phase output capacity}"); shift 2 ;;
    --mjwarp-linesearch-fix)
      wrapper_args+=("$1"); shift ;;
    *) break ;;
  esac
done
mode="${FPGS_NSYS_TRACE_MODE:-node}"
analysis_args=()
case "$mode" in
  node) graph_trace=node ;;
  graph) graph_trace=graph:host-only; analysis_args+=(--require-graph-trace) ;;
  *) echo "FPGS_NSYS_TRACE_MODE must be node or graph" >&2; exit 2 ;;
esac
mkdir -p "$out"
cd "$repo"
CUDA_VISIBLE_DEVICES=${GPU:-0} nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda-sw,nvtx \
  --cuda-graph-trace="$graph_trace" --cuda-event-trace=false --sample=none --cpuctxsw=none -f true -o "$out/$name" \
  uv run python "$here/checked_capture.py" --isaaclab "$repo" --newton "$newton" \
  --expected-run-sha256 "$run_sha" --checks-output "$out/${name}_checks.json" "${wrapper_args[@]}" -- \
  --physics "$physics" --task "$task" --device cuda:0 \
  --repeats "${REPEATS:-3}" --steps "${STEPS:-100}" --profile-steps "${PROFILE_STEPS:-10}" --output "$out/$name.json" "$@" \
  > "$out/$name.log" 2>&1
nsys export --type sqlite -f true -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null 2>&1
uv run python "$harness/analyze_nsys.py" "$out/$name.sqlite" --json "$out/${name}_analysis.json" --top 60 --sequence \
  "${analysis_args[@]}" > "$out/${name}_analysis.txt" 2>&1
grep -h "RESULT\|FINAL" "$out/$name.log" | cut -c1-400
