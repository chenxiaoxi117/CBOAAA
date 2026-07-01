#!/usr/bin/env bash
set -euo pipefail

# Single-seed quick sweep. Reuses BO/V32/V33/reference from the completed V33 run.
ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${SEED:-43}"
MAX_JOBS="${MAX_JOBS:-2}"
RUN_TAG="${RUN_TAG:-dynamic_v33_similar_guard6}"
RESULT_ROOT="$ROOT/result/${RUN_TAG}_s${SEED}"
REFERENCE_BANK="$RESULT_ROOT/common_reference_bank.json"
LOG_ROOT="$ROOT/logs/dynamic_v34_threshold_sweep_s${SEED}"
FAIL_FILE="$LOG_ROOT/failed_jobs.txt"

SCHEDULE="${DYNAMIC_SCHEDULE:-2.5:70,20,10:150;2.5:70,20,10:150;2.5:60,30,10:150;3.0:10,80,10:150;3.0:20,70,10:150;3.0:10,80,10:150}"
METHODS=(
  reduced7_cbo_internal4_transfer_v34_gate030
  reduced7_cbo_internal4_transfer_v34_gate045
  reduced7_cbo_internal4_transfer_v34_gate060
  reduced7_cbo_internal4_transfer_v34_gate070
)

if (( MAX_JOBS < 1 || MAX_JOBS > 2 )); then
  echo "MAX_JOBS只能为1或2。" >&2
  exit 2
fi
if [ ! -s "$REFERENCE_BANK" ]; then
  echo "缺少已完成V33实验的共享基准：$REFERENCE_BANK" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
: > "$FAIL_FILE"
cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

COMMON_ARGS=(
  --mode dynamic_scenario
  --dynamic-history-mode all_history
  --dynamic-schedule "$SCHEDULE"
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
  --phase-reference-warmup-rounds 20
  --phase-reference-reuse-mode exact_only
  --cbo-reference-mode load
  --cbo-reference-file "$REFERENCE_BANK"
  --cbo-shared-reference-policy fixed_probe
  --cbo-backlog-growth-penalty-weight 0
  --scheduler-score-norm-mode candidate_minmax_deadline
  --task-adaptation
)

run_method() {
  local method="$1"
  local out="$RESULT_ROOT/$method"
  local log="$LOG_ROOT/${method}.log"
  if [ -s "$out/dynamic_round_summary.csv" ]; then
    echo "[跳过] $method 已完成"
    return 0
  fi
  echo "[启动] $method"
  if python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$method" \
    --output-root "$out" >"$log" 2>&1; then
    if [ -s "$out/dynamic_round_summary.csv" ]; then
      echo "[完成] $method"
      return 0
    fi
  fi
  echo "[失败] $method，日志：$log" >&2
  echo "$method" >> "$FAIL_FILE"
  return 1
}

echo "============================================================"
echo "V34内部相似度阈值快速验证"
echo "种子：$SEED（不重复运行BO、V32、V33）"
echo "阈值：0.30 0.45 0.60 0.70"
echo "并行方法数：$MAX_JOBS"
echo "============================================================"

running=0
failed=0
for method in "${METHODS[@]}"; do
  run_method "$method" &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    if ! wait -n; then
      failed=$((failed + 1))
    fi
    running=$((running - 1))
  fi
done
while (( running > 0 )); do
  if ! wait -n; then
    failed=$((failed + 1))
  fi
  running=$((running - 1))
done

if (( failed > 0 )) || [ -s "$FAIL_FILE" ]; then
  echo "有方法运行失败，暂不生成总报告。" >&2
  exit 1
fi

python analyze_dynamic_v34_threshold_sweep.py "$RESULT_ROOT"

echo "============================================================"
echo "V34阈值快速验证完成"
echo "报告：$RESULT_ROOT/analysis_v34_threshold_sweep/V34单种子阈值快速验证报告.md"
echo "============================================================"
