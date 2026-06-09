# Dynamic Scenario Patch Notes

This package updates only the four files provided by the user:

- `core_config.py`
- `cli.py`
- `scenario_experiments.py`
- `diagnostics.py`

## New mode

Adds:

```bash
--mode dynamic_scenario
```

The new mode simulates one factory running through multiple workload phases in one continuous BO/CBO run. The BO/CBO agent is initialized once per method and keeps its history across all phases. Fixed-weight baselines keep the same fixed weights throughout the entire dynamic sequence.

## New CLI arguments

```bash
--dynamic-schedule "lambda:RT,Batch,AI:length;lambda:RT,Batch,AI:length;..."
--dynamic-history-mode all_history|recent_window|context_topk
--dynamic-history-window 200
--dynamic-context-topk 100
```

Example phase block:

```text
1.8:10,10,80:200
```

means lambda=1.8, task mix RT=10%, Batch=10%, AI=80%, and phase length=200 BO/CBO iterations.

## Output files

In addition to the normal scenario outputs, dynamic mode writes:

- `dynamic_run_config.json`
- `dynamic_round_summary.csv`
- `dynamic_phase_summary.csv`
- `dynamic_transition_summary.csv`
- `dynamic_repeated_phase_summary.csv`

## Quick test command

Run this after replacing the four files in `new_tr_split`:

```bash
python3 -m py_compile new_tr_split/*.py

python3 -m new_tr_split \
  --mode dynamic_scenario \
  --dynamic-schedule "1.8:10,10,80:5;2.6:10,20,70:5" \
  --selected-keys reduced6_fixed_mid,reduced6_bo_greedy,reduced6_cbo_lite_pressure_prev_unfinished \
  --fixed-rng \
  --fixed-seed 43 \
  --feedback-score window_original \
  --cbo-prediction-guard active \
  --output-root results/dynamic_quick_test
```

Expected checks:

- `dynamic_round_summary.csv` exists.
- `dynamic_phase_summary.csv` exists.
- `phase_id`/`Phase_ID_阶段ID` contains 1 and 2.
- `global_iter`/`Global_Iteration_全局轮次` continues from 1 to 10.
- `phase_iter`/`Phase_Iteration_阶段内轮次` restarts from 1 in phase 2.

## Suggested full 10-phase command

```bash
python3 -m new_tr_split \
  --mode dynamic_scenario \
  --dynamic-schedule "1.8:10,10,80:200;2.6:10,20,70:200;3.0:60,30,10:200;2.6:20,70,10:200;1.8:10,10,80:200;3.0:60,30,10:200;2.6:33,33,34:200;2.6:10,20,70:200;3.0:40,20,40:200;1.8:10,10,80:200" \
  --selected-keys reduced6_fixed_mid,reduced6_fixed_queue_high,reduced6_fixed_risk_high,reduced6_fixed_edge_safe,reduced6_bo_greedy,reduced6_cbo_lite_pressure_only,reduced6_cbo_lite_pressure_prev_unfinished \
  --fixed-rng \
  --fixed-seed 43 \
  --feedback-score window_original \
  --cbo-prediction-guard active \
  --output-root results/dynamic_10phase_seed43
```

## Implementation note

The first version implements dynamic phases by converting the schedule into `CFG.LAMBDA_SCHEDULE` and `CFG.TASK_TYPE_PROB_SCHEDULE`, then calling the normal scenario runner once per method. This keeps BO/CBO history continuous across phases. `dynamic_history_mode` is mapped to existing history knobs as a best-effort first version:

- `all_history`: `BO_HISTORY_MODE=all`
- `recent_window`: `BO_HISTORY_MODE=recent`, `BO_RECENT_WINDOW=dynamic_history_window`
- `context_topk`: `BO_HISTORY_MODE=all`, `CBO_HISTORY_SELECT_MODE=recent_context`, `CBO_CONTEXT_K=dynamic_context_topk`
