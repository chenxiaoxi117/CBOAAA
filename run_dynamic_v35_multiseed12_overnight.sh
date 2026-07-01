#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEEDS="${SEEDS:-42 43 44}"
MAX_JOBS="${MAX_JOBS:-2}"
RUN_TAG="${RUN_TAG:-dynamic_v35_balanced12_mixedorder}"
LOG_ROOT="$ROOT/logs/$RUN_TAG"
FAIL_FILE="$LOG_ROOT/failed_jobs.txt"

# 12 phases: 2 adjacent-similar, 2 delayed-similar, 3 novel and 4 exact repeats.
SCHEDULE="${DYNAMIC_SCHEDULE:-1.5:30,40,30:150;1.5:40,30,30:150;2.0:10,10,80:150;2.5:70,20,10:150;2.0:20,10,70:150;1.5:30,40,30:150;3.0:10,80,10:150;3.0:20,70,10:150;2.0:10,10,80:150;2.5:60,30,10:150;2.5:70,20,10:150;3.0:10,80,10:150}"
# Each unique signature gets 25 rounds so the 20-round reference warmup can
# freeze finite delay and energy references before formal methods start.
REFERENCE_SCHEDULE="${REFERENCE_SCHEDULE:-1.5:30,40,30:25;1.5:40,30,30:25;2.0:10,10,80:25;2.0:20,10,70:25;2.5:70,20,10:25;2.5:60,30,10:25;3.0:10,80,10:25;3.0:20,70,10:25}"

METHODS=(
  reduced7_bo_adaptive
  reduced7_cbo_internal4_transfer_v35_recent_first
  reduced7_cbo_internal4_transfer_learned_noise
)

if (( MAX_JOBS < 1 || MAX_JOBS > 2 )); then
  echo "MAX_JOBS只能为1或2。" >&2
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

validate_bank() {
  local bank="$1"
  python - "$bank" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
obj = json.loads(p.read_text(encoding="utf-8"))
pairs = []
def walk(x):
    if isinstance(x, dict):
        try:
            d = float(x.get("delay_ref"))
            e = float(x.get("energy_per_arrival_ref", x.get("energy_ref")))
            if math.isfinite(d) and math.isfinite(e): pairs.append((d, e))
        except Exception: pass
        for value in x.values(): walk(value)
    elif isinstance(x, list):
        for value in x: walk(value)
walk(obj)
print(f"共享基准有效参考对：{len(pairs)}")
raise SystemExit(0 if len(pairs) >= 8 else 1)
PY
}

validate_method() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
cols = [c for c in d.columns if str(c).startswith("normalized_tradeoff_score")]
n = max((pd.to_numeric(d[c], errors="coerce").notna().sum() for c in cols), default=0)
print(f"有效归一化综合指标：{n}/{len(d)}")
raise SystemExit(0 if len(d) == 1800 and n == 1800 else 1)
PY
}

run_reference() {
  local seed="$1"
  local root bank log
  root="$(seed_root "$seed")"
  bank="$root/common_reference_bank.json"
  log="$LOG_ROOT/s${seed}_reference.log"
  mkdir -p "$root"
  if validate_bank "$bank"; then
    echo "[跳过] 种子${seed}共享基准有效"
    return 0
  fi
  echo "[基准] 种子${seed}：生成8个独立场景参考"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$REFERENCE_SCHEDULE" \
    --fixed-seed "$seed" \
    --selected-keys reduced7_fixed_mid \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$bank" \
    --output-root "$root/reference_builder" >"$log" 2>&1
  if ! validate_bank "$bank"; then
    echo "[失败] 种子${seed}共享基准无效；未运行正式方法" >&2
    return 1
  fi
}

run_method() {
  local seed="$1" method="$2"
  local root bank out csv log
  root="$(seed_root "$seed")"
  bank="$root/common_reference_bank.json"
  out="$root/$method"
  csv="$out/dynamic_round_summary.csv"
  log="$LOG_ROOT/s${seed}_${method}.log"
  if [ -s "$csv" ] && validate_method "$csv"; then
    echo "[跳过] 种子${seed} $method 已完成"
    return 0
  fi
  echo "[启动] 种子${seed} $method"
  if python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$SCHEDULE" \
    --fixed-seed "$seed" \
    --selected-keys "$method" \
    --cbo-reference-mode load \
    --cbo-reference-file "$bank" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$out" >"$log" 2>&1 \
    && [ -s "$csv" ] && validate_method "$csv"; then
    echo "[完成] 种子${seed} $method"
    return 0
  fi
  echo "[失败] 种子${seed} $method；日志：$log" >&2
  echo "seed=$seed method=$method" >> "$FAIL_FILE"
  return 1
}

ROOTS=()
for seed in $SEEDS; do
  echo "============================================================"
  echo "种子${seed}：先校验基准，再最多并行两个方法"
  echo "============================================================"
  run_reference "$seed"

  running=0
  failed=0
  for method in "${METHODS[@]}"; do
    run_method "$seed" "$method" &
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
  if (( failed > 0 )) || [ -s "$FAIL_FILE" ]; then
    echo "种子${seed}存在失败方法，停止后续种子。" >&2
    exit 1
  fi
  ROOTS+=("$(seed_root "$seed")")
done

MULTI_OUT="$ROOT/result/${RUN_TAG}_multiseed_analysis"
python analyze_dynamic_v35_multiseed12.py "${ROOTS[@]}" --output "$MULTI_OUT"

echo "============================================================"
echo "V35三随机种子12阶段实验全部完成"
echo "报告：$MULTI_OUT/V35三随机种子12阶段动态实验报告.md"
echo "============================================================"
