# Scheduler Alpha Tradeoff Patch Notes

## Scope
Modified: `D:\CBO\525\修改后的超新代码_tr_residual_topk.py`

This patch changes the bottom-level Boltzmann scheduler node scoring and scheduler-only energy/latency normalization. It does not change the BO feedback objective default.

## Reviewed Scheduler Path
- `split_task_weights(theta_full)` keeps the first 6 theta dimensions as `[W_RT_Latency, W_Batch_Latency, W_AI_Latency, W_RT_Energy, W_Batch_Energy, W_AI_Energy]`.
- `extract_scheduler_controls(theta_full)` still reads extended controls such as `W_Queue`, `W_Risk_Scale`, `Beta_Control`, `Opportunity_Rho`, and `Cloud_Gate`.
- Legacy normal Boltzmann score was in `BoltzmannScheduler.select_node()`:
  `score = energy_w * norm_e + latency_w * norm_l + risk_w * norm_risk`.
- Legacy constrained Boltzmann score was in `ConstrainedBoltzmannScheduler.select_node()`:
  `score = energy_w * norm_e + latency_w * norm_l + risk_w * norm_risk + queue_w * norm_queue`.
- `energy_w` and `latency_w` come from `split_task_weights()` by task type. `risk_w` comes from `CFG.TASK_RISK_WEIGHTS` and, in constrained mode, is multiplied by `W_Risk_Scale`. `queue_w` comes from `extract_scheduler_controls()` as `W_Queue` when `USE_QUEUE_PRESSURE_SCORE=True`.

## Modified Functions
- `ExperimentConfig`: added scheduler tradeoff and scheduler score-normalization defaults.
- `BoltzmannScheduler.__init__()`: reads scheduler EMA/clip/eps parameters from `CFG`.
- `BoltzmannScheduler._compute_fixed_norms()` and `_compute_rolling_norms()`: now report norm debug fields while preserving legacy math.
- Added `BoltzmannScheduler._compute_scheduler_norms()`.
- Added `BoltzmannScheduler._resolve_scheduler_alpha()`.
- Added `BoltzmannScheduler._score_candidate_components()`.
- `BoltzmannScheduler.select_node()`: supports legacy and alpha tradeoff scoring.
- `ConstrainedBoltzmannScheduler._compute_norms_with_queue()`: now uses the shared scheduler normalization path.
- `ConstrainedBoltzmannScheduler.select_node()`: supports legacy and alpha tradeoff scoring.
- `_candidate_scores_for_task()`: static diagnostics use the same scheduler scoring helper.
- `ConnectedFactory.run_continuous()` and `_record_window_log()`: collect last/mean scheduler debug per BO window.
- `aggregate_logs()` and `group_log_to_dataframe()`: export scheduler fields into round summary.
- `_write_refactor_config_snapshot()` and `method_history_policy_map()`: include scheduler config.
- `argparse`: added all requested `--scheduler-*` parameters.
- `run_scenario_group()`: prints `[SCHED-TRADEOFF]`, `[SCHED-NORM]`, and `[SCHED-ROUND]` diagnostics.

## Legacy Compatibility
Default mode is:
- `--scheduler-tradeoff-mode legacy`
- `--scheduler-score-norm-mode legacy`

In this default path, the scheduler keeps the old score formula and still follows the method's existing `norm_mode` (`fixed` or `rolling`). No new alpha score is used unless explicitly enabled from the command line.

## New Score Formulas
`alpha_fixed`:
- `alpha = clip(scheduler_tradeoff_alpha, scheduler_alpha_min, scheduler_alpha_max)`
- `score = alpha * service_component + (1 - alpha) * energy_component`

`alpha_from_ratio`:
- For each task type, `alpha_raw = latency_w / max(latency_w + energy_w, eps)`
- `alpha = clip(alpha_raw, scheduler_alpha_min, scheduler_alpha_max)`
- `score = alpha * service_component + (1 - alpha) * energy_component`

Components:
- `service_component = scheduler_service_latency_weight * norm_l + scheduler_service_risk_weight * risk_w * norm_risk + scheduler_service_queue_weight * queue_w * norm_queue`
- `energy_component = scheduler_energy_weight * norm_e`
- For normal `BoltzmannScheduler`, `norm_queue=0`.
- If `USE_SCORE_RISK=False`, `norm_risk=0`.
- If `USE_QUEUE_PRESSURE_SCORE=False`, `norm_queue=0`.

## Scheduler Score Normalization
`legacy`:
- Preserves existing method behavior. `norm_mode=fixed` uses `CFG.ENERGY_NORM` and `CFG.DELAY_NORM`; `norm_mode=rolling` uses the existing rolling EMA references.

`candidate_median`:
- Per task arrival, within current candidates only:
  - `energy_ref = median(energy_raw over candidates)`
  - `latency_ref = median(latency_total over candidates)`
  - `norm_e = clip(energy_raw / max(energy_ref, eps), 0, scheduler_norm_clip_max)`
  - `norm_l = clip(latency_total / max(latency_ref, eps), 0, scheduler_norm_clip_max)`

`candidate_iqr`:
- Records candidate IQR debug fields.
- Score still uses positive median ratios for lower-is-better ordering:
  - `norm_e = clip(energy_raw / max(median_energy, eps), 0, scheduler_norm_clip_max)`
  - `norm_l = clip(latency_total / max(median_latency, eps), 0, scheduler_norm_clip_max)`
- Very small IQR falls back safely to the same median-ratio behavior.

`rolling_ema`:
- Uses the existing rolling EMA approach, but parameters are now CLI/config controlled:
  - `--scheduler-norm-ema-alpha`, default `0.995`
  - `--scheduler-norm-clip-max`, default `3.0`
  - `--scheduler-norm-eps`, default `1e-6`

## BO Training Objective
`BO_Training_Cost` default remains `Eval_Cost` / existing feedback path. `--cbo-objective-mode` is still present and defaults to `eval_cost`. This patch does not default-enable `normalized_tradeoff`; scheduler alpha is only for bottom-level node selection.

## Debug / Round Summary Fields
`last_score_debug` now includes scheduler tradeoff/norm mode, alpha source, selected normalized metrics, selected service/energy components, selected score, and energy/latency references.

Window-level `perf_log` and round summary include:
- `scheduler_tradeoff_mode`
- `scheduler_score_norm_mode`
- `scheduler_alpha_last`
- `scheduler_alpha_mean`
- `selected_service_component_last`
- `selected_energy_component_last`
- `selected_norm_e_last`
- `selected_norm_l_last`
- `selected_score_last`

These fields were confirmed in the generated quick-test round summary CSV.

## Quick Tests
Passed:
- `python -m py_compile "D:\CBO\525\修改后的超新代码_tr_residual_topk.py"`
- `python "D:\CBO\525\修改后的超新代码_tr_residual_topk.py" --help`
- Legacy 10-round smoke test without new scheduler flags.
- `alpha_fixed + candidate_median` 10-round smoke test.
- `alpha_from_ratio + candidate_median` 10-round smoke test.

The smoke tests used one fixed method (`reduced6_fixed_mid`) to cover scheduler execution and round-summary export quickly.

## Recommended 500-Round Runs
Recommended next two experimental additions, alongside the unchanged legacy baseline:

S1 legacy baseline:
```powershell
python "D:\CBO\525\修改后的超新代码_tr_residual_topk.py" --mode scenario --bo-iterations 500 --session-duration 120000 --selected-keys reduced6_fixed_mid,reduced6_fixed_tuned,reduced6_bo_greedy,reduced6_cbo_lite_pressure_taskmix_counts,direct_greedy_cost,direct_least_load,direct_queue_aware_greedy --output-root "D:\CBO\525\sched_500_legacy"
```

S2 alpha_fixed_service90:
```powershell
python "D:\CBO\525\修改后的超新代码_tr_residual_topk.py" --mode scenario --bo-iterations 500 --session-duration 120000 --selected-keys reduced6_fixed_mid,reduced6_fixed_tuned,reduced6_bo_greedy,reduced6_cbo_lite_pressure_taskmix_counts,direct_greedy_cost,direct_least_load,direct_queue_aware_greedy --output-root "D:\CBO\525\sched_500_alpha_fixed_service90" --scheduler-tradeoff-mode alpha_fixed --scheduler-tradeoff-alpha 0.90 --scheduler-score-norm-mode candidate_median
```

S3 alpha_from_ratio:
```powershell
python "D:\CBO\525\修改后的超新代码_tr_residual_topk.py" --mode scenario --bo-iterations 500 --session-duration 120000 --selected-keys reduced6_fixed_mid,reduced6_fixed_tuned,reduced6_bo_greedy,reduced6_cbo_lite_pressure_taskmix_counts,direct_greedy_cost,direct_least_load,direct_queue_aware_greedy --output-root "D:\CBO\525\sched_500_alpha_from_ratio" --scheduler-tradeoff-mode alpha_from_ratio --scheduler-alpha-min 0.70 --scheduler-alpha-max 0.95 --scheduler-score-norm-mode candidate_median
```

Do not add `--cbo-objective-mode normalized_tradeoff` for these runs. If reference diagnostics are needed, use diagnostic/reference settings while keeping BO training objective as `eval_cost`.
