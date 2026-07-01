#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${SEED:-43}"
MAX_JOBS="${MAX_JOBS:-5}"
OUT_ROOT="${OUT_ROOT:-$ROOT/result/dynamic_v32_noise_exploration_parallel_s${SEED}}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/dynamic_v32_noise_exploration_parallel_s${SEED}}"
BANK="$OUT_ROOT/common_reference_bank.json"
SCHEDULE="${DYNAMIC_SCHEDULE:-1.8:30,40,30:150;1.8:30,40,30:150;1.8:40,30,30:150;3.0:10,10,80:150}"

BASELINE_METHODS="reduced7_fixed_mid,reduced7_bo_adaptive"
WORKER_METHODS=(
  "reduced7_cbo_internal4_recent80"
  "reduced7_cbo_internal4_recent80_mean_only"
  "reduced7_cbo_internal4_transfer_weighted_no_reexplore"
  "reduced7_cbo_internal4_transfer_learned_noise"
  "reduced7_cbo_internal4_transfer_learned_noise_mean_only"
)

mkdir -p "$OUT_ROOT" "$LOG_DIR"
cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

COMMON_ARGS=(
  --mode dynamic_scenario
  --dynamic-schedule "$SCHEDULE"
  --dynamic-history-mode all_history
  --bo-interval 240
  --fixed-rng
  --fixed-seed "$SEED"
  --reduced7-latency-weight-bounds 0.1,7.0
  --reduced7-queue-weight-bounds 0.0,3.0
  --reduced7-risk-scale-bounds 0.0,8.0
  --reduced7-cloud-gate-bounds 0.01,0.95
  --reduced7-energy-scale-bounds 0.25,2.0
  --feedback-score task_effective_backlog_violation
  --cbo-objective-mode normalized_tradeoff
  --phase-reference-warmup-rounds 5
  --phase-reference-reuse-mode exact_only
  --cbo-backlog-growth-penalty-weight 0
  --scheduler-score-norm-mode candidate_minmax_deadline
  --task-adaptation
)

echo "============================================================"
echo "V32 noise/exploration mechanism experiment"
echo "SEED=$SEED"
echo "SCHEDULE=$SCHEDULE"
echo "OUT_ROOT=$OUT_ROOT"
echo "MAX_JOBS=$MAX_JOBS"
echo "============================================================"

BASELINE_OUT="$OUT_ROOT/00_baseline_and_reference"
BASELINE_LOG="$LOG_DIR/00_baseline_and_reference.log"
if [ ! -s "$BASELINE_OUT/dynamic_round_summary.csv" ] || [ ! -s "$BANK" ]; then
  echo "[BASELINE] building the neutral reference bank and running ordinary BO"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$BASELINE_METHODS" \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$BANK" \
    --output-root "$BASELINE_OUT" \
    >"$BASELINE_LOG" 2>&1
else
  echo "[SKIP] baseline/reference already complete"
fi

if [ ! -s "$BANK" ]; then
  echo "[FAIL] common reference bank was not created: $BANK" >&2
  exit 1
fi

run_worker() {
  local method="$1"
  local method_out="$OUT_ROOT/$method"
  local method_log="$LOG_DIR/$method.log"
  if [ -s "$method_out/dynamic_round_summary.csv" ]; then
    echo "[SKIP] $method"
    return 0
  fi
  echo "[RUN ] $method"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$method" \
    --cbo-reference-mode load \
    --cbo-reference-file "$BANK" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$method_out" \
    >"$method_log" 2>&1
  echo "[ OK ] $method"
}

pids=()
names=()
failed=0
for method in "${WORKER_METHODS[@]}"; do
  run_worker "$method" &
  pids+=("$!")
  names+=("$method")
  if (( ${#pids[@]} >= MAX_JOBS )); then
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "[FAIL] ${names[$i]}" >&2
        failed=1
      fi
    done
    pids=()
    names=()
  fi
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[FAIL] ${names[$i]}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "At least one parallel method failed; inspect $LOG_DIR" >&2
  exit 1
fi

python analyze_dynamic_v31_cbo_diagnosis.py "$OUT_ROOT" \
  --pattern '*' \
  --merge-parallel \
  --output "$OUT_ROOT/analysis"

echo "============================================================"
echo "Finished: $OUT_ROOT"
echo "Report: $OUT_ROOT/analysis/CBO_V31_深度诊断报告.md"
echo "Logs: $LOG_DIR"
echo "============================================================"
