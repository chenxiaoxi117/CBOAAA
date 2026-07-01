#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${SEED:-42}"
RESULT_ROOT="$ROOT/result/dynamic_v35_balanced12_mixedorder_s${SEED}"
BANK="$RESULT_ROOT/common_reference_bank.json"
METHOD="reduced7_cbo_internal4_transfer_v36_adjacent_full_trust"
OUT="$RESULT_ROOT/$METHOD"
CSV="$OUT/dynamic_round_summary.csv"
LOG="$ROOT/logs/v36_quick_s${SEED}.log"

SCHEDULE="${DYNAMIC_SCHEDULE:-1.5:30,40,30:150;1.5:40,30,30:150;2.0:10,10,80:150;2.5:70,20,10:150;2.0:20,10,70:150;1.5:30,40,30:150;3.0:10,80,10:150;3.0:20,70,10:150;2.0:10,10,80:150;2.5:60,30,10:150;2.5:70,20,10:150;3.0:10,80,10:150}"

cd "$ROOT"
if [ -f "$ROOT/env.sh" ]; then source "$ROOT/env.sh"; fi
mkdir -p "$ROOT/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

python - "$BANK" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists(): raise SystemExit("缺少共享基准")
obj = json.loads(p.read_text(encoding="utf-8"))
n = 0
def walk(x):
    global n
    if isinstance(x, dict):
        try:
            if math.isfinite(float(x.get("delay_ref"))) and math.isfinite(float(x.get("energy_per_arrival_ref", x.get("energy_ref")))):
                n += 1
        except Exception: pass
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)
walk(obj)
print(f"有效共享参考：{n}")
raise SystemExit(0 if n >= 8 else 1)
PY

echo "[启动] V36种子${SEED}单方法快速验证"
python -m new_tr_split \
  --mode dynamic_scenario \
  --dynamic-history-mode all_history \
  --dynamic-schedule "$SCHEDULE" \
  --bo-interval 240 \
  --fixed-rng \
  --fixed-seed "$SEED" \
  --reduced7-latency-weight-bounds 0.1,7.0 \
  --reduced7-queue-weight-bounds 0.0,3.0 \
  --reduced7-risk-scale-bounds 0.0,8.0 \
  --reduced7-cloud-gate-bounds 0.01,0.95 \
  --reduced7-energy-scale-bounds 0.25,2.0 \
  --feedback-score task_effective_backlog_violation \
  --cbo-objective-mode normalized_tradeoff \
  --phase-reference-warmup-rounds 20 \
  --phase-reference-reuse-mode exact_only \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --selected-keys "$METHOD" \
  --cbo-reference-mode load \
  --cbo-reference-file "$BANK" \
  --cbo-shared-reference-policy fixed_probe \
  --output-root "$OUT" >"$LOG" 2>&1

python - "$CSV" <<'PY'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], low_memory=False)
cols = [c for c in d.columns if str(c).startswith("normalized_tradeoff_score")]
n = max((pd.to_numeric(d[c], errors="coerce").notna().sum() for c in cols), default=0)
print(f"V36有效综合指标：{n}/{len(d)}")
raise SystemExit(0 if len(d) == 1800 and n == 1800 else 1)
PY

python analyze_dynamic_v36_quick.py "$RESULT_ROOT"
echo "[完成] $RESULT_ROOT/analysis_v36/V36相邻相似完全信任快速验收报告.md"
