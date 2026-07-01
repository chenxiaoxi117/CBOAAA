#!/usr/bin/env bash
set -euo pipefail

# 后续最小验证：
# 1. 种子43补跑固定迁移噪声方法，隔离“自动学习基础噪声”的作用。
# 2. 种子按42、44顺序运行，避免不同种子争抢服务器资源。
# 3. 每个种子先单独生成基准，再并行运行普通贝叶斯优化与最佳V32。

ROOT="${ROOT:-/home/ecs-user/CBO}"
SCHEDULE="${DYNAMIC_SCHEDULE:-1.8:30,40,30:150;1.8:30,40,30:150;1.8:40,30,30:150;3.0:10,10,80:150}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/dynamic_v32_followup_validation}"

REFERENCE_METHOD="reduced7_fixed_mid"
BO_METHOD="reduced7_bo_adaptive"
BEST_METHOD="reduced7_cbo_internal4_transfer_learned_noise"
FIXED_NOISE_METHOD="reduced7_cbo_internal4_transfer_weighted_no_reexplore"

mkdir -p "$LOG_ROOT"
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

seed_root() {
  local seed="$1"
  if [ "$seed" = "43" ]; then
    printf '%s' "$ROOT/result/dynamic_v32_two_methods_parallel_s43"
  else
    printf '%s' "$ROOT/result/dynamic_v32_method_parallel_s${seed}"
  fi
}

run_reference() {
  local seed="$1"
  local out_root
  out_root="$(seed_root "$seed")"
  local out="$out_root/00_fixed_reference"
  local bank="$out_root/common_reference_bank.json"
  local log="$LOG_ROOT/s${seed}_fixed_reference.log"
  mkdir -p "$out_root"

  if [ -s "$out/dynamic_round_summary.csv" ] && [ -s "$bank" ]; then
    echo "[跳过] 种子${seed}固定基准已完成"
    return 0
  fi

  echo "[启动] 种子${seed}：生成共享固定基准"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --fixed-seed "$seed" \
    --selected-keys "$REFERENCE_METHOD" \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$bank" \
    --output-root "$out" \
    >"$log" 2>&1
  echo "[完成] 种子${seed}共享固定基准"
}

run_bo() {
  local seed="$1"
  local out_root
  out_root="$(seed_root "$seed")"
  local bank="$out_root/common_reference_bank.json"
  local out="$out_root/$BO_METHOD"
  local log="$LOG_ROOT/s${seed}_ordinary_bo.log"

  if [ -s "$out/dynamic_round_summary.csv" ]; then
    echo "[跳过] 种子${seed}普通贝叶斯优化已完成"
    return 0
  fi
  if [ ! -s "$bank" ]; then
    echo "[失败] 种子${seed}缺少共享基准：$bank" >&2
    return 1
  fi

  echo "[启动] 种子${seed}：普通贝叶斯优化"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --fixed-seed "$seed" \
    --selected-keys "$BO_METHOD" \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$out" \
    >"$log" 2>&1
  echo "[完成] 种子${seed}普通贝叶斯优化"
}

run_best() {
  local seed="$1"
  local out_root
  out_root="$(seed_root "$seed")"
  local bank="$out_root/common_reference_bank.json"
  local out="$out_root/$BEST_METHOD"
  local log="$LOG_ROOT/s${seed}_best_v32.log"

  if [ -s "$out/dynamic_round_summary.csv" ]; then
    echo "[跳过] 种子${seed}最佳V32已完成"
    return 0
  fi
  if [ ! -s "$bank" ]; then
    echo "[失败] 种子${seed}缺少共享基准：$bank" >&2
    return 1
  fi

  echo "[启动] 种子${seed}：自动学习基础噪声＋自适应探索"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --fixed-seed "$seed" \
    --selected-keys "$BEST_METHOD" \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$out" \
    >"$log" 2>&1
  echo "[完成] 种子${seed}最佳V32"
}

run_fixed_noise_s43() {
  local seed="43"
  local out_root
  out_root="$(seed_root "$seed")"
  local bank="$out_root/common_reference_bank.json"
  local out="$out_root/$FIXED_NOISE_METHOD"
  local log="$LOG_ROOT/s43_fixed_transfer_noise.log"

  if [ -s "$out/dynamic_round_summary.csv" ]; then
    echo "[跳过] 种子43固定迁移噪声方法已完成"
    return 0
  fi
  if [ ! -s "$bank" ]; then
    echo "[失败] 种子43原实验缺少共享基准：$bank" >&2
    return 1
  fi

  echo "[启动] 种子43：固定迁移噪声直接对照"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --fixed-seed "$seed" \
    --selected-keys "$FIXED_NOISE_METHOD" \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$out" \
    >"$log" 2>&1
  echo "[完成] 种子43固定迁移噪声直接对照"
}

wait_pair() {
  local pid_a="$1"
  local name_a="$2"
  local pid_b="$3"
  local name_b="$4"
  local failed=0
  if ! wait "$pid_a"; then
    echo "[失败] $name_a" >&2
    failed=1
  fi
  if ! wait "$pid_b"; then
    echo "[失败] $name_b" >&2
    failed=1
  fi
  if (( failed != 0 )); then
    exit 1
  fi
}

echo "============================================================"
echo "第一步：种子43补跑固定迁移噪声直接对照"
echo "============================================================"
run_fixed_noise_s43
python analyze_dynamic_v31_cbo_diagnosis.py "$(seed_root 43)" \
  --pattern '*' \
  --merge-parallel \
  --output "$(seed_root 43)/analysis"

echo "============================================================"
echo "第二步：种子42，先生成基准，再让两个方法并行"
echo "============================================================"
run_reference 42
run_bo 42 & PID_BO=$!
run_best 42 & PID_V32=$!
wait_pair "$PID_BO" "种子42普通贝叶斯优化" "$PID_V32" "种子42最佳V32"
python analyze_dynamic_v31_cbo_diagnosis.py "$(seed_root 42)" \
  --pattern '*' \
  --merge-parallel \
  --output "$(seed_root 42)/analysis"

echo "============================================================"
echo "第三步：种子44，先生成基准，再让两个方法并行"
echo "============================================================"
run_reference 44
run_bo 44 & PID_BO=$!
run_best 44 & PID_V32=$!
wait_pair "$PID_BO" "种子44普通贝叶斯优化" "$PID_V32" "种子44最佳V32"
python analyze_dynamic_v31_cbo_diagnosis.py "$(seed_root 44)" \
  --pattern '*' \
  --merge-parallel \
  --output "$(seed_root 44)/analysis"

echo "============================================================"
echo "全部完成"
echo "种子42：$ROOT/result/dynamic_v32_method_parallel_s42/analysis/CBO_V31_深度诊断报告.md"
echo "种子43：$ROOT/result/dynamic_v32_two_methods_parallel_s43/analysis/CBO_V31_深度诊断报告.md"
echo "种子44：$ROOT/result/dynamic_v32_method_parallel_s44/analysis/CBO_V31_深度诊断报告.md"
echo "日志目录：$LOG_ROOT"
echo "============================================================"
