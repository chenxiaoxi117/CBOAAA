#!/usr/bin/env bash
set -euo pipefail

# 六阶段专项验收：同时覆盖“完全相同后相似”和“新场景后相似”。
ROOT="${ROOT:-/home/ecs-user/CBO}"
SEEDS="${SEEDS:-42 43 44}"
RUN_TAG="${RUN_TAG:-dynamic_v33_similar_guard6}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
RUN_V32="${RUN_V32:-1}"

BO_METHOD="reduced7_bo_adaptive"
V32_METHOD="reduced7_cbo_internal4_transfer_learned_noise"
V33_METHOD="reduced7_cbo_internal4_transfer_v33_safe_similar"
REFERENCE_METHOD="reduced7_fixed_mid"

# 1初始 -> 2完全相同 -> 3相似；4新场景 -> 5相似 -> 6完全相同。
SCHEDULE="${DYNAMIC_SCHEDULE:-2.5:70,20,10:150;2.5:70,20,10:150;2.5:60,30,10:150;3.0:10,80,10:150;3.0:20,70,10:150;3.0:10,80,10:150}"
REFERENCE_SCHEDULE="${REFERENCE_SCHEDULE:-2.5:70,20,10:1;2.5:70,20,10:1;2.5:60,30,10:1;3.0:10,80,10:1;3.0:20,70,10:1;3.0:10,80,10:1}"

mkdir -p "$LOG_ROOT"
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
  --bo-interval 240
  --fixed-rng
  --reduced7-latency-weight-bounds 0.1,7.0
  --reduced7-queue-weight-bounds 0.0,3.0
  --reduced7-risk-scale-bounds 0.0,8.0
  --reduced7-cloud-gate-bounds 0.01,0.95
  --reduced7-energy-scale-bounds 0.25,2.0
  --feedback-score task_effective_backlog_violation
  --cbo-objective-mode normalized_tradeoff
  --phase-reference-warmup-rounds 20
  --phase-reference-reuse-mode exact_only
  --cbo-backlog-growth-penalty-weight 0
  --scheduler-score-norm-mode candidate_minmax_deadline
  --task-adaptation
)

seed_root() {
  printf '%s' "$ROOT/result/${RUN_TAG}_s$1"
}

run_reference() {
  local seed="$1" root bank out log
  root="$(seed_root "$seed")"
  bank="$root/common_reference_bank.json"
  out="$root/reference_builder"
  log="$LOG_ROOT/s${seed}_reference.log"
  mkdir -p "$root"
  if [ -s "$bank" ]; then
    echo "[跳过] 种子${seed}共享基准已存在"
    return 0
  fi
  echo "[基准] 种子${seed}"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$REFERENCE_SCHEDULE" \
    --fixed-seed "$seed" \
    --selected-keys "$REFERENCE_METHOD" \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$bank" \
    --output-root "$out" >"$log" 2>&1
  test -s "$bank"
}

run_method() {
  local seed="$1" method="$2" tag="$3" root bank out log
  root="$(seed_root "$seed")"
  bank="$root/common_reference_bank.json"
  out="$root/$method"
  log="$LOG_ROOT/s${seed}_${tag}.log"
  if [ -s "$out/dynamic_round_summary.csv" ]; then
    echo "[跳过] 种子${seed} ${tag}已完成"
    return 0
  fi
  echo "[启动] 种子${seed} ${tag}"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$SCHEDULE" \
    --fixed-seed "$seed" \
    --selected-keys "$method" \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$out" >"$log" 2>&1
  echo "[完成] 种子${seed} ${tag}"
}

ROOTS=()
for seed in $SEEDS; do
  echo "============================================================"
  echo "种子${seed}：先共享基准，再并行普通方法与V33，最后运行V32"
  echo "============================================================"
  run_reference "$seed"

  run_method "$seed" "$BO_METHOD" "bo" & pid_bo=$!
  run_method "$seed" "$V33_METHOD" "v33" & pid_v33=$!
  failed=0
  wait "$pid_bo" || failed=1
  wait "$pid_v33" || failed=1
  if (( failed != 0 )); then
    echo "[失败] 种子${seed}并行方法失败" >&2
    exit 1
  fi
  if [ "$RUN_V32" = "1" ]; then
    run_method "$seed" "$V32_METHOD" "v32"
  fi
  ROOTS+=("$(seed_root "$seed")")
done

OUT="$ROOT/result/${RUN_TAG}_multiseed_analysis"
python analyze_dynamic_v33_guard_validation.py "${ROOTS[@]}" --output "$OUT"

echo "============================================================"
echo "V33相似场景专项验收全部完成"
echo "报告：$OUT/V33相似场景负迁移专项验收报告.md"
echo "============================================================"
