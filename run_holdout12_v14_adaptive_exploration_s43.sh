#!/usr/bin/env bash
set -uo pipefail

# V14: 12 unseen static scenes, three acquisition variants, BO+CBO per case.
# Usage: MAX_JOBS=5 bash run_holdout12_v14_adaptive_exploration_s43.sh 43

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-5}"
OUT="${OUT:-$ROOT/result/holdout12_v14_adaptive_exploration_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"
METHODS="reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p "$LOG_DIR"
: > "$FAIL_FILE"
cd "$ROOT"

run_case() {
  local variant="$1" mode="$2" lam="$3" probs="$4" scene="$5"
  local case_out="$OUT/$variant/$scene"
  local log_base="$LOG_DIR/${variant}_${scene}"

  if [[ -s "$case_out/pressure_scan_summary_all.csv" ]]; then
    echo "[SKIP] $variant $scene"
    return 0
  fi

  mkdir -p "$case_out"
  echo "[RUN ] $variant $scene acq=$mode"
  python -m new_tr_split \
    --mode pressure_scan \
    --selected-keys "$METHODS" \
    --bo-iterations 500 \
    --bo-interval 240 \
    --session-duration 120000 \
    --fixed-rng --fixed-seed "$SEED" \
    --reduced7-latency-weight-bounds 0.1,7.0 \
    --reduced7-queue-weight-bounds 0.0,3.0 \
    --reduced7-risk-scale-bounds 0.0,8.0 \
    --reduced7-cloud-gate-bounds 0.01,0.95 \
    --reduced7-energy-scale-bounds 0.25,2.0 \
    --feedback-score task_effective_backlog_violation \
    --bo-history-mode recent --bo-recent-window 80 \
    --cbo-objective-mode normalized_tradeoff \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy cbo_first \
    --cbo-shared-reference-warmup-rounds 5 \
    --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
    --cbo-backlog-growth-penalty-weight 0 \
    --cbo-sigma-calibration on \
    --cbo-sigma-calibration-buffer-size 50 \
    --cbo-sigma-calibration-min-samples 10 \
    --cbo-sigma-calibration-use-in-acq "$mode" \
    --cbo-sigma-calibration-eta 0.25 \
    --cbo-adaptive-exploration-beta-max 3.0 \
    --cbo-adaptive-exploration-eta-max 0.25 \
    --cbo-adaptive-exploration-window 30 \
    --cbo-adaptive-exploration-sample-target 80 \
    --cbo-adaptive-exploration-smoothing 0.20 \
    --cbo-adaptive-exploration-progress-pct 0.01 \
    --cbo-adaptive-exploration-reexplore-gain 0.25 \
    --cbo-adaptive-exploration-plausible-margin-mult 2.0 \
    --cbo-adaptive-exploration-backlog-ref 1.0 \
    --cbo-adaptive-exploration-unfinished-ref 0.10 \
    --cbo-adaptive-exploration-trend-ref 0.05 \
    --cbo-adaptive-exploration-max-util-start 0.80 \
    --cbo-sigma-scale-default 4.0 \
    --cbo-sigma-scale-min 1.0 \
    --cbo-sigma-scale-max 6.0 \
    --cbo-sigma-floor 0.03 \
    --scheduler-score-norm-mode candidate_minmax_deadline \
    --task-adaptation \
    --lambda-values "$lam" \
    --task-probs "$probs" \
    --output-root "$case_out" \
    >"${log_base}.stdout.log" \
    2>"${log_base}.stderr.log"

  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[ OK ] $variant $scene"
  else
    echo "[FAIL] $variant $scene exit=$code"
    echo "$variant $scene exit=$code" >> "$FAIL_FILE"
  fi
  return $code
}

variants=("V14-A_diag" "V14-B_soft" "V14-C_adaptive")
modes=("false" "soft" "adaptive")

lams=(
  "1.8" "1.8" "2.6" "3.0"
  "1.8" "2.6" "2.6" "3.0"
  "1.8" "2.6" "3.0" "3.0"
)
probs=(
  "10,70,20" "30,60,10" "40,40,20" "60,20,20"
  "40,20,40" "80,10,10" "40,50,10" "30,20,50"
  "20,10,70" "40,20,40" "10,70,20" "70,10,20"
)
scenes=(
  "loss_lam1p8_rt10_batch70_ai20"
  "loss_lam1p8_rt30_batch60_ai10"
  "loss_lam2p6_rt40_batch40_ai20"
  "loss_lam3p0_rt60_batch20_ai20"
  "tie_lam1p8_rt40_batch20_ai40"
  "tie_lam2p6_rt80_batch10_ai10"
  "tie_lam2p6_rt40_batch50_ai10"
  "tie_lam3p0_rt30_batch20_ai50"
  "win_lam1p8_rt20_batch10_ai70"
  "win_lam2p6_rt40_batch20_ai40"
  "win_lam3p0_rt10_batch70_ai20"
  "win_lam3p0_rt70_batch10_ai20"
)

echo "============================================================"
echo "V14 holdout12 adaptive exploration"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "SEED=$SEED"
echo "MAX_JOBS=$MAX_JOBS"
echo "============================================================"

running=0
failed=0
for vi in 0 1 2; do
  for si in {0..11}; do
    run_case "${variants[$vi]}" "${modes[$vi]}" "${lams[$si]}" "${probs[$si]}" "${scenes[$si]}" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      if ! wait -n; then failed=$((failed + 1)); fi
      running=$((running - 1))
    fi
  done
done

while (( running > 0 )); do
  if ! wait -n; then failed=$((failed + 1)); fi
  running=$((running - 1))
done

summary_count=$(find "$OUT" -name pressure_scan_summary_all.csv -type f | wc -l)
config_count=$(find "$OUT" -name refactor_run_config.json -type f | wc -l)
round_count=$(find "$OUT" -name '*round_summary*.csv' -type f | wc -l)

echo "============================================================"
echo "pressure summaries = $summary_count / 36"
echo "refactor configs   = $config_count / 36"
echo "round summaries    = $round_count / 72"
echo "failed waits       = $failed"
echo "failed records     = $(grep -cve '^$' "$FAIL_FILE" || true)"
echo "result root        = $OUT"
echo "============================================================"

if [[ "$summary_count" -ne 36 || "$config_count" -ne 36 || "$round_count" -ne 72 || -s "$FAIL_FILE" ]]; then
  exit 1
fi
