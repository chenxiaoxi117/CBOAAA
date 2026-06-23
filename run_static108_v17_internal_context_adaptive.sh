#!/usr/bin/env bash
set -uo pipefail

# Static108 V17 internal-context CBO variants with adaptive exploration.
# Default methods:
#   B      = reduced7_bo_adaptive
#   D      = reduced7_cbo_lite_pressure_taskmix_counts + adaptive exploration
#   D-I4   = reduced7_cbo_lite_internal4 + adaptive exploration
#   D-I6C  = reduced7_cbo_lite_internal6_context + adaptive exploration
#   D-I4C  = reduced7_cbo_lite_internal4_context + adaptive exploration
#
# Usage:
#   MAX_JOBS=10 bash run_static108_v17_internal_context_adaptive.sh 43

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-10}"
METHODS="${METHODS:-reduced7_bo_adaptive,reduced7_cbo_lite_pressure_taskmix_counts,reduced7_cbo_lite_internal4,reduced7_cbo_lite_internal6_context,reduced7_cbo_lite_internal4_context}"
REFERENCE_METHOD="${REFERENCE_METHOD:-reduced7_cbo_lite_pressure_taskmix_counts}"
OUT="${OUT:-$ROOT/result/static108_v17_internal_context_adaptive_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"

method_count=$(awk -F',' '{print NF}' <<< "$METHODS")
EXPECTED_ROUND_FILES=$((108 * method_count))

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p "$LOG_DIR"
: > "$FAIL_FILE"
cd "$ROOT"

run_case() {
  local lam="$1" rt="$2" batch="$3" ai="$4"
  local lam_tag="${lam/./p}"
  local scene="rt${rt}_batch${batch}_ai${ai}"
  local case_out="$OUT/lambda_${lam_tag}/$scene"
  local log_base="$LOG_DIR/lambda_${lam_tag}_${scene}"

  if [[ -s "$case_out/pressure_scan_summary_all.csv" ]]; then
    echo "[SKIP] lambda=$lam $scene"
    return 0
  fi

  mkdir -p "$case_out"
  echo "[RUN ] lambda=$lam $scene seed=$SEED methods=$METHODS"
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
    --cbo-reference-source-method-key "$REFERENCE_METHOD" \
    --cbo-backlog-growth-penalty-weight 0 \
    --cbo-sigma-calibration on \
    --cbo-sigma-calibration-buffer-size 50 \
    --cbo-sigma-calibration-min-samples 10 \
    --cbo-sigma-calibration-use-in-acq adaptive \
    --cbo-sigma-calibration-eta 0.25 \
    --cbo-sigma-scale-default 4.0 \
    --cbo-sigma-scale-min 1.0 \
    --cbo-sigma-scale-max 6.0 \
    --cbo-sigma-floor 0.03 \
    --scheduler-score-norm-mode candidate_minmax_deadline \
    --task-adaptation \
    --lambda-values "$lam" \
    --task-probs "$rt,$batch,$ai" \
    --output-root "$case_out" \
    >"${log_base}.stdout.log" \
    2>"${log_base}.stderr.log"

  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[ OK ] lambda=$lam $scene"
  else
    echo "[FAIL] lambda=$lam $scene exit=$code"
    echo "lambda=$lam task_probs=$rt,$batch,$ai exit=$code" >> "$FAIL_FILE"
  fi
  return $code
}

echo "============================================================"
echo "Static108 V17 internal-context adaptive ablation"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "SEED=$SEED"
echo "MAX_JOBS=$MAX_JOBS"
echo "METHODS=$METHODS"
echo "REFERENCE_METHOD=$REFERENCE_METHOD"
echo "EXPECTED_ROUND_FILES=$EXPECTED_ROUND_FILES"
echo "============================================================"

running=0
failed=0
for lam in 1.8 2.6 3.0; do
  for rt in 10 20 30 40 50 60 70 80; do
    for batch in 10 20 30 40 50 60 70 80; do
      ai=$((100 - rt - batch))
      if (( ai < 10 )); then continue; fi

      run_case "$lam" "$rt" "$batch" "$ai" &
      running=$((running + 1))
      if (( running >= MAX_JOBS )); then
        if ! wait -n; then failed=$((failed + 1)); fi
        running=$((running - 1))
      fi
    done
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
echo "Static108 V17 internal-context adaptive ablation finished"
echo "pressure summaries = $summary_count / 108"
echo "refactor configs   = $config_count / 108"
echo "round summaries    = $round_count / $EXPECTED_ROUND_FILES"
echo "failed waits       = $failed"
echo "failed records     = $(grep -cve '^$' "$FAIL_FILE" || true)"
echo "result root        = $OUT"
echo "============================================================"

if [[ "$summary_count" -ne 108 || "$config_count" -ne 108 || "$round_count" -ne "$EXPECTED_ROUND_FILES" || -s "$FAIL_FILE" ]]; then
  exit 1
fi
