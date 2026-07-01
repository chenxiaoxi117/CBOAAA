#!/usr/bin/env bash
set -euo pipefail

# Corrected one-seed validation: build finite per-scene references first,
# then compare recent80 BO, V32 relation retrieval and V35 recent-first hybrid.
ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${SEED:-43}"
MAX_JOBS="${MAX_JOBS:-2}"
RUN_TAG="${RUN_TAG:-dynamic_v35_objective_fixed}"
OUT="$ROOT/result/${RUN_TAG}_s${SEED}"
REFERENCE_BANK="$OUT/common_reference_bank.json"
LOG_ROOT="$ROOT/logs/${RUN_TAG}_s${SEED}"
FAIL_FILE="$LOG_ROOT/failed_jobs.txt"

SCHEDULE="${DYNAMIC_SCHEDULE:-2.5:70,20,10:150;2.5:70,20,10:150;2.5:60,30,10:150;3.0:10,80,10:150;3.0:20,70,10:150;3.0:10,80,10:150}"
# Four unique signatures, each with more rounds than the 20-round phase warmup.
REFERENCE_SCHEDULE="${REFERENCE_SCHEDULE:-2.5:70,20,10:25;2.5:60,30,10:25;3.0:10,80,10:25;3.0:20,70,10:25}"
METHODS=(
  reduced7_bo_adaptive
  reduced7_cbo_internal4_transfer_learned_noise
  reduced7_cbo_internal4_transfer_v35_recent_first
)

if (( MAX_JOBS < 1 || MAX_JOBS > 2 )); then
  echo "MAX_JOBS只能为1或2。" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$OUT"
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
  --cbo-backlog-growth-penalty-weight 0
  --scheduler-score-norm-mode candidate_minmax_deadline
  --task-adaptation
)

validate_bank() {
  python - "$REFERENCE_BANK" <<'PY'
import json, math, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
obj = json.loads(p.read_text(encoding="utf-8"))
pairs = []
def walk(x):
    if isinstance(x, dict):
        d = x.get("delay_ref")
        e = x.get("energy_per_arrival_ref", x.get("energy_ref"))
        try:
            if math.isfinite(float(d)) and math.isfinite(float(e)):
                pairs.append((float(d), float(e)))
        except Exception:
            pass
        for value in x.values():
            walk(value)
    elif isinstance(x, list):
        for value in x:
            walk(value)
walk(obj)
print(f"共享基准中有效时延/能耗参考对：{len(pairs)}")
raise SystemExit(0 if len(pairs) >= 4 else 1)
PY
}

if ! validate_bank; then
  echo "[基准] 生成4个独立场景的有效归一化参考"
  python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$REFERENCE_SCHEDULE" \
    --selected-keys reduced7_fixed_mid \
    --cbo-reference-mode calibrate \
    --cbo-shared-reference-policy fixed_probe \
    --cbo-reference-output-file "$REFERENCE_BANK" \
    --output-root "$OUT/reference_builder" \
    >"$LOG_ROOT/reference.log" 2>&1
fi
if ! validate_bank; then
  echo "共享基准仍无有效时延/能耗参考，已停止，未运行任何正式方法。" >&2
  exit 1
fi

validate_method() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
cols = [c for c in d.columns if str(c).startswith("normalized_tradeoff_score")]
n = max((pd.to_numeric(d[c], errors="coerce").notna().sum() for c in cols), default=0)
print(f"有效归一化综合指标：{n}/{len(d)}")
raise SystemExit(0 if n == len(d) and n > 0 else 1)
PY
}

run_method() {
  local method="$1"
  local method_out="$OUT/$method"
  local csv="$method_out/dynamic_round_summary.csv"
  local log="$LOG_ROOT/${method}.log"
  if [ -s "$csv" ] && validate_method "$csv"; then
    echo "[跳过] $method 已完成且综合指标有效"
    return 0
  fi
  echo "[启动] $method"
  if python -m new_tr_split \
    "${COMMON_ARGS[@]}" \
    --dynamic-schedule "$SCHEDULE" \
    --selected-keys "$method" \
    --cbo-reference-mode load \
    --cbo-reference-file "$REFERENCE_BANK" \
    --cbo-shared-reference-policy fixed_probe \
    --output-root "$method_out" >"$log" 2>&1 \
    && [ -s "$csv" ] && validate_method "$csv"; then
    echo "[完成] $method"
    return 0
  fi
  echo "[失败或综合指标无效] $method；日志：$log" >&2
  echo "$method" >> "$FAIL_FILE"
  return 1
}

echo "============================================================"
echo "V35归一化目标修复验证：种子$SEED"
echo "方法：BO最近80、V32关系历史、V35相邻相似最近优先"
echo "============================================================"

running=0
failed=0
for method in "${METHODS[@]}"; do
  run_method "$method" &
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
  echo "存在失败方法，未生成汇总报告。" >&2
  exit 1
fi

python analyze_dynamic_v35_objective_fixed.py "$OUT"

echo "============================================================"
echo "V35快速验证完成"
echo "报告：$OUT/analysis/V35归一化目标修复与最近优先验证报告.md"
echo "============================================================"
