#!/usr/bin/env bash
set -uo pipefail

# 静态高压力36场景：lambda=3.0，比较普通贝叶斯优化与当前最佳V32。
# 与动态实验同时运行时默认只开1个静态进程；最多允许2个。

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
MAX_JOBS="${MAX_JOBS:-1}"
METHODS="reduced7_bo_adaptive,reduced7_cbo_internal4_transfer_learned_noise"
OUT="${OUT:-$ROOT/result/static36_lambda3_v32_s${SEED}}"
LOG_DIR="$OUT/logs"
FAIL_FILE="$OUT/failed_jobs.txt"

if (( MAX_JOBS < 1 || MAX_JOBS > 2 )); then
  echo "MAX_JOBS只能设为1或2；与动态实验同时运行时建议使用1。" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p "$LOG_DIR"
: > "$FAIL_FILE"
cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

COMMON_ARGS=(
  --mode pressure_scan
  --bo-interval 240
  --session-duration 120000
  --fixed-rng
  --fixed-seed "$SEED"
  --reduced7-latency-weight-bounds 0.1,7.0
  --reduced7-queue-weight-bounds 0.0,3.0
  --reduced7-risk-scale-bounds 0.0,8.0
  --reduced7-cloud-gate-bounds 0.01,0.95
  --reduced7-energy-scale-bounds 0.25,2.0
  --feedback-score task_effective_backlog_violation
  --bo-history-mode recent
  --bo-recent-window 80
  --cbo-objective-mode normalized_tradeoff
  --cbo-shared-reference-policy fixed_probe
  --cbo-shared-reference-warmup-rounds 5
  --cbo-backlog-growth-penalty-weight 0
  --scheduler-score-norm-mode candidate_minmax_deadline
  --task-adaptation
)

run_case() {
  local rt="$1" batch="$2" ai="$3"
  local scene="rt${rt}_batch${batch}_ai${ai}"
  local case_out="$OUT/lambda_3p0/$scene"
  local methods_out="$case_out/methods"
  local reference_out="$case_out/reference"
  local bank="$case_out/common_reference_bank.json"
  local log_base="$LOG_DIR/lambda_3p0_${scene}"

  if [ -s "$methods_out/pressure_scan_summary_all.csv" ] && [ -s "$bank" ]; then
    echo "[跳过] lambda=3.0 $scene"
    return 0
  fi

  mkdir -p "$case_out"

  # 只运行2轮固定参数；其作用是用5个中立探测轮次生成共同归一化基准。
  if [ ! -s "$bank" ]; then
    echo "[基准] lambda=3.0 $scene"
    python -m new_tr_split \
      "${COMMON_ARGS[@]}" \
      --selected-keys reduced7_fixed_mid \
      --bo-iterations 2 \
      --cbo-reference-mode calibrate \
      --cbo-reference-calibration-rounds 5 \
      --cbo-reference-min-rounds 3 \
      --cbo-reference-output-file "$bank" \
      --lambda-values 3.0 \
      --task-probs "$rt,$batch,$ai" \
      --output-root "$reference_out" \
      >"${log_base}_reference.stdout.log" \
      2>"${log_base}_reference.stderr.log"
    local reference_code=$?
    if (( reference_code != 0 )) || [ ! -s "$bank" ]; then
      echo "[失败] lambda=3.0 $scene 共享基准生成失败" >&2
      echo "lambda=3.0 task_probs=$rt,$batch,$ai stage=reference exit=$reference_code" >> "$FAIL_FILE"
      return 1
    fi
  fi

  echo "[运行] lambda=3.0 $scene，普通贝叶斯优化＋V32"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$METHODS" \
    --bo-iterations 500 \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --lambda-values 3.0 \
    --task-probs "$rt,$batch,$ai" \
    --output-root "$methods_out" \
    >"${log_base}_methods.stdout.log" \
    2>"${log_base}_methods.stderr.log"
  local code=$?

  if (( code == 0 )) && [ -s "$methods_out/pressure_scan_summary_all.csv" ]; then
    echo "[完成] lambda=3.0 $scene"
  else
    echo "[失败] lambda=3.0 $scene exit=$code" >&2
    echo "lambda=3.0 task_probs=$rt,$batch,$ai stage=methods exit=$code" >> "$FAIL_FILE"
    return 1
  fi
}

echo "============================================================"
echo "静态lambda=3.0高压力36场景"
echo "随机种子：$SEED"
echo "静态并行进程：$MAX_JOBS"
echo "结果目录：$OUT"
echo "============================================================"

running=0
failed=0
for rt in 10 20 30 40 50 60 70 80; do
  for batch in 10 20 30 40 50 60 70 80; do
    ai=$((100 - rt - batch))
    if (( ai < 10 )); then
      continue
    fi

    run_case "$rt" "$batch" "$ai" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      if ! wait -n; then
        failed=$((failed + 1))
      fi
      running=$((running - 1))
    fi
  done
done

while (( running > 0 )); do
  if ! wait -n; then
    failed=$((failed + 1))
  fi
  running=$((running - 1))
done

summary_count=$(find "$OUT" -path '*/methods/pressure_scan_summary_all.csv' -type f | wc -l)
bank_count=$(find "$OUT" -name common_reference_bank.json -type f | wc -l)
round_count=$(find "$OUT" -path '*/methods/*' -name '*round_summary*.csv' -type f | wc -l)
failed_records=$(grep -cve '^$' "$FAIL_FILE" || true)

echo "============================================================"
echo "静态36场景运行结束"
echo "完成场景：$summary_count / 36"
echo "共享基准：$bank_count / 36"
echo "方法结果：$round_count / 72"
echo "失败等待：$failed"
echo "失败记录：$failed_records"
echo "============================================================"

if [[ "$summary_count" -ne 36 || "$bank_count" -ne 36 || "$round_count" -ne 72 || -s "$FAIL_FILE" ]]; then
  exit 1
fi

if ! python analyze_static108_core_v15.py "$OUT" \
    --baseline BO_ADAPTIVE \
    --compare V32_LEARNED_NOISE \
    --output "$OUT/analysis"; then
  echo "分析失败，但36个场景结果仍然完整保留在：$OUT" >&2
  exit 1
fi

echo "分析完成：$OUT/analysis"
