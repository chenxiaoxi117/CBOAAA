#!/usr/bin/env bash
set -uo pipefail

# Static108 V16 B/C ablation for seed-level checks.
# B: BO + adaptive exploration, no context, no external gate.
# C: CBO internal6 + external gate, no adaptive exploration.
# Usage:
#   MAX_JOBS=10 VARIANT=a  bash run_static108_v16_bc_ablation.sh 43
#   MAX_JOBS=10 VARIANT=b  bash run_static108_v16_bc_ablation.sh 43
#   MAX_JOBS=10 VARIANT=c  bash run_static108_v16_bc_ablation.sh 43
#   MAX_JOBS=10 VARIANT=bc bash run_static108_v16_bc_ablation.sh 43

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-10}"
VARIANT="${VARIANT:-bc}"

case "$VARIANT" in
  a|A)
    METHODS="reduced7_bo_greedy"
    EXPECTED_ROUND_FILES=108
    TAG="a_bo_greedy_pure"
    ;;
  b|B)
    METHODS="reduced7_bo_adaptive"
    EXPECTED_ROUND_FILES=108
    TAG="b_bo_adaptive"
    ;;
  ab|AB|a_b|A_B)
    METHODS="reduced7_bo_greedy,reduced7_bo_adaptive"
    EXPECTED_ROUND_FILES=216
    TAG="ab"
    ;;
  c|C)
    METHODS="reduced7_cbo_lite_pressure_taskmix_counts"
    EXPECTED_ROUND_FILES=108
    TAG="c_cbo_noadaptive"
    ;;
  ac|AC|a_c|A_C)
    METHODS="reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts"
    EXPECTED_ROUND_FILES=216
    TAG="ac"
    ;;
  abc|ABC|a_b_c|A_B_C)
    METHODS="reduced7_bo_greedy,reduced7_bo_adaptive,reduced7_cbo_lite_pressure_taskmix_counts"
    EXPECTED_ROUND_FILES=324
    TAG="abc"
    ;;
  bc|BC|b_c|B_C)
    METHODS="reduced7_bo_adaptive,reduced7_cbo_lite_pressure_taskmix_counts"
    EXPECTED_ROUND_FILES=216
    TAG="bc"
    ;;
  *)
    echo "VARIANT must be a, b, c, ab, ac, abc, or bc" >&2
    exit 2
    ;;
esac

OUT="${OUT:-$ROOT/result/static108_v16_${TAG}_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"

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
  echo "[RUN ] lambda=$lam $scene seed=$SEED variant=$VARIANT methods=$METHODS"
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
echo "Static108 V16 B/C ablation"
echo "ROOT=$ROOT"
echo "OUT=$OUT"
echo "SEED=$SEED"
echo "MAX_JOBS=$MAX_JOBS"
echo "VARIANT=$VARIANT"
echo "METHODS=$METHODS"
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
echo "Static108 V16 B/C ablation finished"
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
