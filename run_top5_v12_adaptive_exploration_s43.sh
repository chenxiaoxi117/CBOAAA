#!/usr/bin/env bash
set -uo pipefail

# Compare diagnostic-only, fixed-soft, and adaptive exploration on five difficult scenes.
# Usage: MAX_JOBS=3 bash run_top5_v12_adaptive_exploration_s43.sh 43

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-3}"
EXP_TAG="${EXP_TAG:-V12}"
ADAPT_SAMPLE_TARGET="${ADAPT_SAMPLE_TARGET:-40}"
ADAPT_REEXPLORE_GAIN="${ADAPT_REEXPLORE_GAIN:-0.25}"
ADAPT_MARGIN_MULT="${ADAPT_MARGIN_MULT:-1.0}"
OUT="${OUT:-$ROOT/result/top5_v12_adaptive_exploration_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"
METHODS="reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
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
    --cbo-adaptive-exploration-sample-target "$ADAPT_SAMPLE_TARGET" \
    --cbo-adaptive-exploration-smoothing 0.20 \
    --cbo-adaptive-exploration-progress-pct 0.01 \
    --cbo-adaptive-exploration-reexplore-gain "$ADAPT_REEXPLORE_GAIN" \
    --cbo-adaptive-exploration-plausible-margin-mult "$ADAPT_MARGIN_MULT" \
    --cbo-adaptive-exploration-backlog-ref 1.0 \
    --cbo-adaptive-exploration-unfinished-ref 0.10 \
    --cbo-adaptive-exploration-trend-ref 0.05 \
    --cbo-adaptive-exploration-max-util-start 0.80 \
    --cbo-sigma-scale-default 4.0 --cbo-sigma-scale-min 1.0 --cbo-sigma-scale-max 6.0 --cbo-sigma-floor 0.03 \
    --scheduler-score-norm-mode candidate_minmax_deadline --task-adaptation \
    --lambda-values "$lam" --task-probs "$probs" \
    --output-root "$case_out" >"${log_base}.stdout.log" 2>"${log_base}.stderr.log"
  local code=$?
  if [[ $code -eq 0 ]]; then echo "[ OK ] $variant $scene"; else echo "[FAIL] $variant $scene exit=$code"; echo "$variant $scene exit=$code" >> "$FAIL_FILE"; fi
  return $code
}

variants=("${EXP_TAG}-A_diag" "${EXP_TAG}-B_soft" "${EXP_TAG}-C_adaptive")
modes=("false" "soft" "adaptive")
lams=("3.0" "3.0" "3.0" "2.6" "2.6")
probs=("10,40,50" "30,60,10" "10,30,60" "20,10,70" "70,20,10")
scenes=("lambda3p0_rt10_batch40_ai50" "lambda3p0_rt30_batch60_ai10" "lambda3p0_rt10_batch30_ai60" "lambda2p6_rt20_batch10_ai70" "lambda2p6_rt70_batch20_ai10")

echo "V12 adaptive exploration top5 | ROOT=$ROOT | OUT=$OUT | SEED=$SEED | MAX_JOBS=$MAX_JOBS"
running=0; failed=0
for vi in 0 1 2; do
  for si in 0 1 2 3 4; do
    run_case "${variants[$vi]}" "${modes[$vi]}" "${lams[$si]}" "${probs[$si]}" "${scenes[$si]}" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then if ! wait -n; then failed=$((failed + 1)); fi; running=$((running - 1)); fi
  done
done
while (( running > 0 )); do if ! wait -n; then failed=$((failed + 1)); fi; running=$((running - 1)); done

summary_count=$(find "$OUT" -name pressure_scan_summary_all.csv -type f | wc -l)
config_count=$(find "$OUT" -name refactor_run_config.json -type f | wc -l)
round_count=$(find "$OUT" -name '*round_summary*.csv' -type f | wc -l)
echo "pressure summaries = $summary_count / 15"
echo "refactor configs   = $config_count / 15"
echo "round summaries    = $round_count / 30"
echo "failed waits       = $failed"
echo "result root        = $OUT"
if [[ "$summary_count" -ne 15 || "$config_count" -ne 15 || "$round_count" -ne 30 || -s "$FAIL_FILE" ]]; then exit 1; fi
