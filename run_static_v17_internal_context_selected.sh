#!/usr/bin/env bash
set -uo pipefail

# Selected-scene V17 internal-context CBO ablation.
# This is a quick validation before running full static108.
#
# Default selected scenes include:
#   - previously difficult CBO-vs-BO scenes
#   - balanced middle scenes
#   - RT-heavy, batch-heavy, and AI-heavy edge scenes
#
# Usage:
#   MAX_JOBS=6 bash run_static_v17_internal_context_selected.sh 43

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-6}"
METHODS="${METHODS:-reduced7_bo_adaptive,reduced7_cbo_lite_pressure_taskmix_counts,reduced7_cbo_lite_internal4,reduced7_cbo_lite_internal6_context,reduced7_cbo_lite_internal4_context}"
REFERENCE_METHOD="${REFERENCE_METHOD:-reduced7_cbo_lite_pressure_taskmix_counts}"
OUT="${OUT:-$ROOT/result/static_v17_internal_context_selected12_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"

SCENES=(
  "3.0 10 40 50"
  "3.0 30 60 10"
  "3.0 10 30 60"
  "2.6 20 10 70"
  "2.6 70 20 10"
  "2.6 40 40 20"
  "1.8 10 10 80"
  "1.8 80 10 10"
  "3.0 40 40 20"
  "2.6 10 80 10"
  "1.8 30 30 40"
  "3.0 70 10 20"
)

method_count=$(awk -F',' '{print NF}' <<< "$METHODS")
scene_count="${#SCENES[@]}"
EXPECTED_ROUND_FILES=$((scene_count * method_count))

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
echo "Selected-scene V17 internal-context adaptive ablation"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "SEED=$SEED"
echo "MAX_JOBS=$MAX_JOBS"
echo "METHODS=$METHODS"
echo "REFERENCE_METHOD=$REFERENCE_METHOD"
echo "SCENES=$scene_count"
echo "EXPECTED_ROUND_FILES=$EXPECTED_ROUND_FILES"
echo "============================================================"

running=0
failed=0
for item in "${SCENES[@]}"; do
  read -r lam rt batch ai <<< "$item"
  run_case "$lam" "$rt" "$batch" "$ai" &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    if ! wait -n; then failed=$((failed + 1)); fi
    running=$((running - 1))
  fi
done

while (( running > 0 )); do
  if ! wait -n; then failed=$((failed + 1)); fi
  running=$((running - 1))
done

summary_count=$(find "$OUT" -name pressure_scan_summary_all.csv -type f | wc -l)
config_count=$(find "$OUT" -name refactor_run_config.json -type f | wc -l)
round_count=$(find "$OUT" -name '*round_summary*.csv' -type f | wc -l)

echo "============================================================"
echo "Selected-scene V17 internal-context adaptive ablation finished"
echo "pressure summaries = $summary_count / $scene_count"
echo "refactor configs   = $config_count / $scene_count"
echo "round summaries    = $round_count / $EXPECTED_ROUND_FILES"
echo "failed waits       = $failed"
echo "failed records     = $(grep -cve '^$' "$FAIL_FILE" || true)"
echo "result root        = $OUT"
echo "============================================================"

if [[ "$summary_count" -ne "$scene_count" || "$config_count" -ne "$scene_count" || "$round_count" -ne "$EXPECTED_ROUND_FILES" || -s "$FAIL_FILE" ]]; then
  exit 1
fi
