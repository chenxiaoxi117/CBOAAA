#!/usr/bin/env bash
set -euo pipefail

# 先串行生成中立基准与普通 BO，再并行运行两个 V32 方法。
# 两个 V32 方法共用完全相同的基准文件、随机种子和动态场景。

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${SEED:-43}"
OUT_ROOT="${OUT_ROOT:-$ROOT/result/dynamic_v32_two_methods_parallel_s${SEED}}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/dynamic_v32_two_methods_parallel_s${SEED}}"
REFERENCE_BANK="$OUT_ROOT/common_reference_bank.json"
SCHEDULE="${DYNAMIC_SCHEDULE:-1.8:30,40,30:150;1.8:30,40,30:150;1.8:40,30,30:150;3.0:10,10,80:150}"

BASELINE_METHODS="reduced7_fixed_mid,reduced7_bo_adaptive"
PARALLEL_METHODS=(
  "reduced7_cbo_internal4_transfer_learned_noise"
  "reduced7_cbo_internal4_transfer_learned_noise_mean_only"
)

mkdir -p "$OUT_ROOT" "$LOG_DIR"
cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

# 防止两个 Python 进程各自再占满全部 CPU 核心。
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

echo "[1/3] 串行运行基准方法并生成共享基准文件"
BASELINE_OUT="$OUT_ROOT/00_baseline_and_reference"
BASELINE_LOG="$LOG_DIR/00_baseline_and_reference.log"
if [ ! -s "$BASELINE_OUT/dynamic_round_summary.csv" ] || [ ! -s "$REFERENCE_BANK" ]; then
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$BASELINE_METHODS" \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$REFERENCE_BANK" \
    --output-root "$BASELINE_OUT" \
    >"$BASELINE_LOG" 2>&1
else
  echo "[跳过] 基准结果和共享基准文件已经存在"
fi

if [ ! -s "$REFERENCE_BANK" ]; then
  echo "[失败] 没有生成共享基准文件：$REFERENCE_BANK" >&2
  exit 1
fi

run_one_method() {
  local method="$1"
  local method_out="$OUT_ROOT/$method"
  local method_log="$LOG_DIR/$method.log"

  if [ -s "$method_out/dynamic_round_summary.csv" ]; then
    echo "[跳过] $method 已完成"
    return 0
  fi

  echo "[启动] $method"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --selected-keys "$method" \
    --cbo-reference-mode load \
    --cbo-reference-file "$REFERENCE_BANK" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$method_out" \
    >"$method_log" 2>&1
  echo "[完成] $method"
}

echo "[2/3] 同时启动两个 V32 方法"
run_one_method "${PARALLEL_METHODS[0]}" &
PID_A=$!
run_one_method "${PARALLEL_METHODS[1]}" &
PID_B=$!

FAILED=0
if ! wait "$PID_A"; then
  echo "[失败] ${PARALLEL_METHODS[0]}，请检查日志" >&2
  FAILED=1
fi
if ! wait "$PID_B"; then
  echo "[失败] ${PARALLEL_METHODS[1]}，请检查日志" >&2
  FAILED=1
fi
if (( FAILED != 0 )); then
  exit 1
fi

echo "[3/3] 合并三个目录并生成中文诊断报告"
python analyze_dynamic_v31_cbo_diagnosis.py "$OUT_ROOT" \
  --pattern '*' \
  --merge-parallel \
  --output "$OUT_ROOT/analysis"

echo "实验完成：$OUT_ROOT"
echo "中文报告：$OUT_ROOT/analysis/CBO_V31_深度诊断报告.md"
echo "运行日志：$LOG_DIR"
