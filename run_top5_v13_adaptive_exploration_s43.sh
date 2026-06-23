#!/usr/bin/env bash
set -uo pipefail

# V13 uses the V1.4 adaptive controller while preserving the V12 experiment matrix.
ROOT="${ROOT:-/home/ecs-user/CBO}"
SEED="${1:-43}"
export EXP_TAG="V13"
export OUT="${OUT:-$ROOT/result/top5_v13_adaptive_exploration_s${SEED}}"
export ADAPT_SAMPLE_TARGET="80"
export ADAPT_REEXPLORE_GAIN="0.25"
export ADAPT_MARGIN_MULT="2.0"

bash "$ROOT/run_top5_v12_adaptive_exploration_s43.sh" "$SEED"
