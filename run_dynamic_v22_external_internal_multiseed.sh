#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ecs-user/CBO}"
OUT_BASE="${OUT_BASE:-$ROOT/result}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
MAX_JOBS="${MAX_JOBS:-6}"
SEEDS="${SEEDS:-43 44 45}"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

SCHEDULE="${DYNAMIC_SCHEDULE:-1.8:30,40,30:150;1.8:32,38,30:150;2.6:10,20,70:150;3.0:30,60,10:150;2.6:70,20,10:150;1.8:30,40,30:150;2.6:10,20,70:150;3.0:30,60,10:150;1.8:10,80,10:150;3.0:70,10,20:150;2.6:40,40,20:150;1.8:30,40,30:150}"
METHODS="${METHODS:-reduced7_bo_adaptive,reduced7_cbo_lite_pressure_taskmix_counts,reduced7_cbo_lite_internal6_context,reduced7_cbo_lite_internal4_context}"

run_one() {
  local variant="$1"
  local seed="$2"
  local ext_thr="$3"
  local ctx_thr="$4"
  local ctx_k="$5"
  local ext_topk="$6"
  local out="$OUT_BASE/dynamic_v22_${variant}_s${seed}"
  local log="$LOG_DIR/dynamic_v22_${variant}_s${seed}.log"

  if [ -f "$out/dynamic_round_summary.csv" ]; then
    echo "[SKIP] $variant seed=$seed already finished: $out"
    return 0
  fi

  echo "[RUN ] variant=$variant seed=$seed ext_thr=$ext_thr ctx_thr=$ctx_thr ctx_k=$ctx_k ext_topk=$ext_topk"
  python -m new_tr_split \
    --mode dynamic_scenario \
    --selected-keys "$METHODS" \
    --dynamic-schedule "$SCHEDULE" \
    --dynamic-history-mode all_history \
    --bo-interval 240 \
    --fixed-rng \
    --fixed-seed "$seed" \
    --reduced7-latency-weight-bounds 0.1,7.0 \
    --reduced7-queue-weight-bounds 0.0,3.0 \
    --reduced7-risk-scale-bounds 0.0,8.0 \
    --reduced7-cloud-gate-bounds 0.01,0.95 \
    --reduced7-energy-scale-bounds 0.25,2.0 \
    --feedback-score task_effective_backlog_violation \
    --cbo-objective-mode normalized_tradeoff \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy cbo_first \
    --cbo-shared-reference-warmup-rounds 5 \
    --cbo-reference-source-method-key reduced7_cbo_lite_internal4_context \
    --phase-reference-warmup-rounds 5 \
    --cbo-backlog-growth-penalty-weight 0 \
    --cbo-history-select-mode external_internal_threshold \
    --cbo-external-sim-threshold "$ext_thr" \
    --cbo-external-topk "$ext_topk" \
    --cbo-external-min-rows 40 \
    --cbo-external-recent-keep 20 \
    --cbo-context-sim-threshold "$ctx_thr" \
    --cbo-context-k "$ctx_k" \
    --cbo-context-min-rows 40 \
    --cbo-context-weak-fallback-k 40 \
    --cbo-sigma-calibration on \
    --cbo-sigma-calibration-use-in-acq adaptive \
    --cbo-sigma-calibration-eta 0.25 \
    --scheduler-score-norm-mode candidate_minmax_deadline \
    --task-adaptation \
    --output-root "$out" \
    > "$log" 2>&1
  local code=$?
  if [ "$code" -eq 0 ] && [ -f "$out/dynamic_round_summary.csv" ]; then
    echo "[ OK ] variant=$variant seed=$seed out=$out"
  else
    echo "[FAIL] variant=$variant seed=$seed code=$code out=$out log=$log" | tee -a "$OUT_BASE/dynamic_v22_failed_jobs.txt"
  fi
  return "$code"
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 20
  done
}

echo "============================================================"
echo "Dynamic V22 external-internal threshold sweep"
echo "ROOT=$ROOT"
echo "OUT_BASE=$OUT_BASE"
echo "LOG_DIR=$LOG_DIR"
echo "MAX_JOBS=$MAX_JOBS"
echo "SEEDS=$SEEDS"
echo "METHODS=$METHODS"
echo "============================================================"

for seed in $SEEDS; do
  wait_for_slot; run_one "A_ext070_ctx070_k200" "$seed" 0.70 0.70 200 400 &
  wait_for_slot; run_one "B_ext075_ctx070_k200" "$seed" 0.75 0.70 200 400 &
  wait_for_slot; run_one "C_ext065_ctx070_k200" "$seed" 0.65 0.70 200 400 &
  wait_for_slot; run_one "D_ext070_ctx070_k300" "$seed" 0.70 0.70 300 600 &
done

wait

echo "============================================================"
echo "All Dynamic V22 jobs submitted by this script have finished."
echo "Completed summaries:"
for seed in $SEEDS; do
  find "$OUT_BASE" -maxdepth 1 -type d -name "dynamic_v22_*_s${seed}" | sort | while read -r d; do
    if [ -f "$d/dynamic_round_summary.csv" ]; then
      rows=$(python - "$d/dynamic_round_summary.csv" <<'PY'
import sys
import pandas as pd
print(len(pd.read_csv(sys.argv[1], low_memory=False)))
PY
)
      echo "OK rows=$rows $d"
    else
      echo "MISS $d"
    fi
  done
done
echo "Failed jobs file: $OUT_BASE/dynamic_v22_failed_jobs.txt"
echo "============================================================"
