#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 11636-12014.
# Original command-line interface block.

try:
    from .runtime import *  # noqa: F401,F403
except ImportError:
    from runtime import *  # noqa: F401,F403

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    def _parse_bounds_pair_arg(value, option_name):
        if value is None or str(value).strip() == "":
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lo = float(value[0])
            hi = float(value[1])
            return (lo, hi) if lo <= hi else (hi, lo)
        parts = [p.strip() for p in str(value).split(",")]
        if len(parts) != 2:
            parser.error(f"{option_name} must be formatted as low,high")
        try:
            lo = float(parts[0])
            hi = float(parts[1])
        except ValueError:
            parser.error(f"{option_name} must contain numeric low,high values")
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def _parse_float_list_arg(value, option_name, expected_len):
        if value is None or str(value).strip() == "":
            return None
        parts = [p.strip() for p in str(value).split(",")]
        if len(parts) != int(expected_len):
            parser.error(f"{option_name} must contain exactly {int(expected_len)} comma-separated values")
        try:
            return [float(p) for p in parts]
        except ValueError:
            parser.error(f"{option_name} must contain numeric values")

    def _parse_bool_arg(value):
        if isinstance(value, bool):
            return value
        val = str(value).strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError("expected one of true/false, yes/no, on/off, 1/0")

    parser.add_argument("--mode", choices=["all", "param", "extreme", "scan", "sensitivity", "scenario", "ratio_grid", "pressure_scan", "dynamic_scenario", "offline_noise", "batch_federated"], default="all")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--local-delta", type=float, default=0.08)
    parser.add_argument("--dim", type=str, default="W_RT_Latency")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fixed-rng", action="store_true")
    parser.add_argument("--fixed-seed", type=int, default=None)
    parser.add_argument("--no-boltzmann-random", action="store_true", help="Disable Boltzmann stochastic node selection and choose the minimum score candidate")
    parser.add_argument("--ratio-step", type=int, default=10, help="任务比例网格步长，默认10，即10%%")
    parser.add_argument("--ratio-min", type=int, default=10, help="每类任务最低比例，默认10，即至少10%%")
    parser.add_argument("--output-root", type=str, default=None, help="ratio_grid输出根目录")
    parser.add_argument("--selected-keys", type=str, default=None, help="逗号分隔的方法键；例如 cbo-alpha-direct-prev-unfinished-context 或 cbo-alpha-direct-unfinished-context；默认使用备份版 no-RoundRobin 方法集")
    parser.add_argument("--sensitivity-scenarios", type=str, default="default,rt_high,batch_high,ai_high", help="敏感度场景：default,rt_high,batch_high,ai_high 或 name:RT,Batch,AI")
    parser.add_argument("--sens-seeds", type=int, default=2, help="敏感度分析每个参数点重复的随机种子数")
    parser.add_argument("--sens-windows", type=int, default=3, help="敏感度分析每个 theta 连续评价的窗口数")
    parser.add_argument("--sens-greedy", action="store_true", help="敏感度分析时关闭 Boltzmann 随机，使用确定性机会集合选择")
    parser.add_argument("--pref-norm-mode", choices=["rolling", "fixed"], default="rolling", help="静态节点偏好诊断使用的归一化模式")
    parser.add_argument("--task-adaptation", action="store_true", help="启用 task_node_affinity_factor 环境异构矩阵")
    parser.add_argument("--no-task-adaptation", action="store_true", help="关闭 task_node_affinity_factor，所有任务只使用节点基础 service_rate_gips")
    parser.add_argument("--rt-deadline-factor", type=float, default=None, help="覆盖 RT 任务 deadline_factor，例如 2.5")
    parser.add_argument("--bo-iterations", type=int, default=None, help="覆盖 BO_ITERATIONS，便于快速测试")
    parser.add_argument("--bo-interval", type=float, default=None, help="覆盖 BO_INTERVAL")
    parser.add_argument("--session-duration", type=float, default=None, help="覆盖 SESSION_DURATION")
    parser.add_argument("--feedback-mode", choices=["window", "cohort_complete", "dual"], default="window", help="BO反馈模式：window为旧窗口级反馈；cohort_complete为任务批次完成后反馈；dual为窗口快反馈+批次/分类精反馈替换")
    parser.add_argument("--feedback-score", choices=["window_original", "task_effective", "task_effective_backlog", "task_effective_backlog_violation", "paired_fixed_mid_delta", "legacy_dual", "legacy_cohort"], default=getattr(CFG, "DEFAULT_SCENARIO_FEEDBACK_SCORE", "task_effective_backlog_violation"), help="BO tell 使用的训练反馈；当前默认 task_effective_backlog_violation。paired_fixed_mid_delta 为仿真专用：同窗口 shadow fixed_mid 的 delta cost。")
    parser.add_argument("--cbo-reference-mode", choices=["off", "calibrate", "load", "auto_macro"], default=getattr(CFG, "CBO_REFERENCE_MODE", "calibrate"), help="Scenario normalization reference scale mode")
    parser.add_argument("--cbo-reference-calibration-rounds", type=int, default=30, help="Rounds used to build/freeze scenario reference")
    parser.add_argument("--cbo-reference-min-rounds", type=int, default=5, help="Minimum rounds before reference is considered available")
    parser.add_argument("--cbo-shared-reference-policy", choices=["cbo_first", "fixed_probe"], default=getattr(CFG, "CBO_SHARED_REFERENCE_POLICY", "cbo_first"), help="How to create shared scenario normalization references")
    parser.add_argument("--cbo-shared-reference-warmup-rounds", type=int, default=getattr(CFG, "CBO_SHARED_REFERENCE_WARMUP_ROUNDS", 5), help="CBO-first warm-up windows within the normal BO budget")
    parser.add_argument("--cbo-reference-source-method-key", type=str, default=getattr(CFG, "CBO_REFERENCE_SOURCE_METHOD_KEY", "reduced7_cbo_lite_pressure_taskmix_counts"), help="CBO method used to define shared references")
    parser.add_argument("--cbo-reference-stat", choices=["median", "trimmed_mean", "mean"], default="median", help="Statistic used for reference calibration")
    parser.add_argument("--cbo-reference-trim-pct", type=float, default=0.1, help="Trim percent for trimmed_mean reference")
    parser.add_argument("--cbo-reference-freeze-after-calibration", action="store_true", default=True, help="Freeze reference after calibration window")
    parser.add_argument("--cbo-reference-file", type=str, default="", help="JSON reference file to load")
    parser.add_argument("--cbo-reference-output-file", type=str, default="", help="JSON reference output file")
    parser.add_argument("--phase-reference-warmup-rounds", type=int, default=getattr(CFG, "PHASE_REFERENCE_WARMUP_ROUNDS", 5), help="Dynamic phase reference warm-up windows; new significant phases use this many probe/warm-up rounds")
    parser.add_argument("--cbo-objective-mode", choices=["eval_cost", "diagnostic_only", "normalized_tradeoff"], default=getattr(CFG, "CBO_OBJECTIVE_MODE", "normalized_tradeoff"), help="BO training objective mode")
    parser.add_argument("--cbo-tradeoff-alpha", type=float, default=0.8, help="alpha in alpha*service_norm + (1-alpha)*energy_norm")
    parser.add_argument("--cbo-alpha-min", type=float, default=0.6, help="minimum clipped alpha")
    parser.add_argument("--cbo-alpha-max", type=float, default=0.95, help="maximum clipped alpha")
    parser.add_argument("--cbo-target-success-rate", type=float, default=0.995, help="target SLA success rate for normalized service score")
    parser.add_argument("--cbo-unfinished-penalty-weight", type=float, default=5.0, help="unfinished_rate penalty in service_norm")
    parser.add_argument("--cbo-success-shortfall-weight", type=float, default=2.0, help="success shortfall normalized penalty in service_norm")
    parser.add_argument("--cbo-backlog-growth-penalty-weight", type=float, default=0.0, help="backlog growth rate penalty in service_norm")
    parser.add_argument("--cbo-class-imbalance-weight", type=float, default=0.0, help="class completion imbalance penalty in service_norm")
    parser.add_argument("--cbo-normalized-ratio-clip-min", type=float, default=0.2, help="min clip for normalized ratios")
    parser.add_argument("--cbo-normalized-ratio-clip-max", type=float, default=5.0, help="max clip for normalized ratios")
    parser.add_argument("--scheduler-tradeoff-mode", choices=["legacy", "alpha_fixed", "alpha_from_ratio", "alpha_direct"], default="legacy", help="底层调度器节点 score 的 service-energy tradeoff 模式；默认 legacy 保持旧逻辑")
    parser.add_argument("--scheduler-tradeoff-alpha", type=float, default=0.85, help="alpha_fixed 模式下的 service 权重 alpha")
    parser.add_argument("--scheduler-alpha-min", type=float, default=0.60, help="scheduler alpha 下限")
    parser.add_argument("--scheduler-alpha-max", type=float, default=0.97, help="scheduler alpha 上限")
    parser.add_argument("--scheduler-le-scale", type=float, default=1.0, help="alpha_direct only: scale alpha*norm_l+(1-alpha)*norm_e before risk/queue penalties")
    parser.add_argument("--alpha-direct-bounds", default=getattr(CFG, "ALPHA_DIRECT_BOUNDS", None), help="alpha_direct uniform BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-rt-bounds", default=getattr(CFG, "ALPHA_DIRECT_RT_BOUNDS", None), help="alpha_direct RT BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-batch-bounds", default=getattr(CFG, "ALPHA_DIRECT_BATCH_BOUNDS", None), help="alpha_direct Batch BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-ai-bounds", default=getattr(CFG, "ALPHA_DIRECT_AI_BOUNDS", None), help="alpha_direct AI BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-fixed-theta", type=str, default=None, help="Fixed alpha_direct 6D theta for cbo-alpha-direct/cbo-alpha-direct-no-risk runs")
    parser.add_argument("--reduced7-latency-weight-bounds", default=getattr(CFG, "REDUCED7_LATENCY_WEIGHT_BOUNDS", None), help="reduced7 latency-weight bounds, formatted as low,high")
    parser.add_argument("--reduced7-queue-weight-bounds", default=getattr(CFG, "REDUCED7_QUEUE_WEIGHT_BOUNDS", None), help="reduced7 W_Queue bounds, formatted as low,high")
    parser.add_argument("--reduced7-risk-scale-bounds", default=getattr(CFG, "REDUCED7_RISK_SCALE_BOUNDS", None), help="reduced7 W_Risk_Scale bounds, formatted as low,high")
    parser.add_argument("--reduced7-cloud-gate-bounds", default=getattr(CFG, "REDUCED7_CLOUD_GATE_BOUNDS", None), help="reduced7 Cloud_Gate bounds, formatted as low,high")
    parser.add_argument("--reduced7-energy-scale-bounds", default=getattr(CFG, "REDUCED7_ENERGY_SCALE_BOUNDS", None), help="reduced7 W_Energy_Scale BO bounds, formatted as low,high, e.g. 0.5,3.0")
    parser.add_argument("--scheduler-service-latency-weight", type=float, default=1.0, help="alpha tradeoff 中 norm_l 的系数")
    parser.add_argument("--scheduler-service-risk-weight", type=float, default=1.0, help="alpha 外部 risk_w*norm_risk 惩罚项的额外系数")
    parser.add_argument("--scheduler-service-queue-weight", type=float, default=1.0, help="alpha 外部 queue_w*norm_queue 惩罚项的额外系数")
    parser.add_argument("--scheduler-energy-weight", type=float, default=1.0, help="energy_component 中 norm_e 的系数")
    parser.add_argument("--scheduler-score-norm-mode", choices=["candidate_minmax_deadline", "legacy", "candidate_median", "candidate_iqr", "rolling_ema"], default=getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "candidate_minmax_deadline"), help="底层调度器 score 的 energy/latency 归一化模式；默认对当前候选节点集合按 deadline 和 min-max 归一化")
    parser.add_argument("--scheduler-norm-clip-max", type=float, default=3.0, help="scheduler score normalization clip max")
    parser.add_argument("--scheduler-norm-eps", type=float, default=1e-6, help="scheduler score normalization epsilon")
    parser.add_argument("--scheduler-norm-ema-alpha", type=float, default=0.995, help="rolling_ema scheduler normalization alpha")
    parser.add_argument("--paired-baseline-key", default="reduced6_fixed_mid", help="paired_fixed_mid_delta 使用的 shadow baseline fixed policy key/alias，默认 reduced6_fixed_mid。")
    parser.add_argument("--deploy-policy", choices=["ei", "greedy", "incumbent", "incumbent_safe", "safe", "safe_bo"], default=None, help="SAFEBO 部署策略别名：ei=BO-EI；greedy=BO-greedy posterior mean；incumbent/safe=安全 incumbent。命令行值只覆盖 CBO 类方法，fixed/direct baseline 不受影响。")
    parser.add_argument("--dual-refined-source", choices=["class", "class_weighted", "class_equal", "class_actual", "class_worst", "cohort"], default=None, help="dual模式下延后精反馈的合成方式：class/class_weighted=按设定任务比例合成；class_equal=三类等权；class_actual=按本批实际到达比例；class_worst=最差类别；cohort=原始整体cohort cost")
    parser.add_argument("--dual-class-metric", choices=["completed", "effective", "censored"], default=None, help="dual分类精反馈内部每类指标使用方式：completed=只用已完成任务均值+完成率惩罚；effective/censored=把未完成任务按截尾等待和当前违约计入")
    parser.add_argument("--task-probs", type=str, default=None, help="固定任务比例 RT,Batch,AI，例如 0.2,0.4,0.4 或 20,40,40")
    parser.add_argument("--task-prob-schedule", type=str, default=None, help="分段任务比例，例如 0:4000:20,40,40;4000:9000:40,10,50")
    parser.add_argument("--lambda-schedule", type=str, default=None, help="分段泊松强度，例如 0:4000:1.0,4000:9000:2.2,9000:12000:1.2")
    parser.add_argument("--lambda-values", type=str, default=None, help="pressure_scan 用，逗号分隔，例如 1.0,1.4,1.8,2.2,2.6,3.0")
    parser.add_argument("--dynamic-schedule", type=str, default=getattr(CFG, "DEFAULT_DYNAMIC_SCHEDULE", ""), help="dynamic_scenario 用，格式：lambda:RT,Batch,AI:length;... 例如 1.8:10,10,80:200;2.6:10,20,70:200")
    parser.add_argument("--dynamic-history-mode", choices=["all_history", "recent_window", "context_topk", "state_gated_kernel"], default=getattr(CFG, "DEFAULT_DYNAMIC_HISTORY_MODE", "all_history"), help="dynamic_scenario 的历史使用模式；state_gated_kernel 使用 workload/state/trend 硬门控+乘积核选择历史")
    parser.add_argument("--dynamic-history-window", type=int, default=getattr(CFG, "DEFAULT_DYNAMIC_HISTORY_WINDOW", 200), help="dynamic_scenario recent_window 的窗口大小")
    parser.add_argument("--dynamic-context-topk", type=int, default=getattr(CFG, "DEFAULT_DYNAMIC_CONTEXT_TOPK", 100), help="dynamic_scenario context_topk 的样本数")
    parser.add_argument("--batch-method", choices=["fixed", "local_bo", "independent_bo", "centralized_bo", "federated_bo"], default="independent_bo", help="Static batch experiment method")
    parser.add_argument("--batch-clients", type=int, default=3, help="Number of heterogeneous clients for static batch experiments")
    parser.add_argument("--batch-rounds", type=int, default=20, help="BO/FBO rounds; each round evaluates one batch per active client")
    parser.add_argument("--batch-tasks", type=int, default=120, help="Default task count per client batch")
    parser.add_argument("--batch-client-task-counts", type=str, default=None, help="Comma-separated task counts per client; overrides --batch-tasks per client")
    parser.add_argument("--batch-client-task-probs", type=str, default=None, help="Per-client task mix, formatted as RT,Batch,AI;RT,Batch,AI")
    parser.add_argument("--batch-node-counts", type=str, default=None, help="Comma-separated node counts per generated client")
    parser.add_argument("--batch-topology-profile", choices=["heterogeneous", "edge_small", "edge_large", "cloud_heavy"], default="heterogeneous", help="Generated client topology profile")
    parser.add_argument("--batch-client-config", type=str, default=None, help="Optional JSON file defining clients, nodes, task mix, and scales")
    parser.add_argument("--batch-context-features", type=str, default=",".join(DEFAULT_STATIC_BATCH_CONTEXT), help="Comma-separated static context features; use all or none")
    parser.add_argument("--batch-objective-weights", type=str, default=None, help="Batch objective weights, e.g. delay=1,energy=0.25,lateness=2,violation=6,makespan=0.5,unfinished=20")
    parser.add_argument("--batch-normalization", choices=["batch_reference", "none"], default="batch_reference", help="Batch objective normalization mode")
    parser.add_argument("--batch-objective-clip", type=float, default=10.0, help="Clip max for normalized objective ratios; <=0 disables clipping")
    parser.add_argument("--batch-reuse-mode", choices=["new_each_round", "fixed_per_client"], default="new_each_round", help="Use new deterministic batch per round or reuse one static batch per client")
    parser.add_argument("--batch-order", choices=["generated", "deadline", "type_deadline", "largest_workload", "random"], default="deadline", help="Local dispatch order inside a received batch")
    parser.add_argument("--batch-dispatch-gap", type=float, default=0.01, help="Small local dispatch gap between tasks while create_time remains batch start")
    parser.add_argument("--batch-deterministic-scheduler", type=_parse_bool_arg, default=True, help="Disable Boltzmann sampling inside each batch evaluation for lower-noise BO feedback")
    parser.add_argument("--batch-fed-share-mode", choices=["surrogate", "experience", "hybrid"], default="surrogate", help="Federated BO sharing mode")
    parser.add_argument("--batch-fed-candidates", type=int, default=96, help="Candidate theta count for federated surrogate aggregation")
    parser.add_argument("--batch-fed-beta", type=float, default=None, help="Override federated surrogate UCB beta; default uses CFG.FED_BETA")
    parser.add_argument("--bo-history-mode", choices=["all", "recent", "confidence", "recent_confidence"], default=getattr(CFG, "DEFAULT_BO_HISTORY_MODE", "recent"), help="BO GP训练历史使用方式；备份版默认 recent。all=全部；recent=最近N个；confidence=过滤低可信反馈；recent_confidence=最近+可信度过滤")
    parser.add_argument("--bo-recent-window", type=int, default=getattr(CFG, "DEFAULT_BO_RECENT_WINDOW", 80), help="recent/recent_confidence 模式保留最近多少个BO样本；备份版默认80")
    parser.add_argument("--bo-confidence-min", type=float, default=None, help="confidence/recent_confidence 模式下保留样本的最低反馈可信度")
    parser.add_argument("--bo-confidence-min-samples", type=int, default=None, help="可信过滤后至少保留多少样本，不足时自动使用高可信+最近样本兜底")
    parser.add_argument("--cbo-recent-window", type=int, default=None, help="Override CBO method recent/recent_confidence window only; BO recent window is unchanged")
    parser.add_argument("--cbo-history-select-mode", choices=["recent", "recent_context", "recent_context_elite", "hybrid", "state_gated_kernel", "global_context_threshold", "external_internal_threshold"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_SELECT_MODE", "recent"), help="CBO-only stability history selection mode")
    parser.add_argument("--cbo-context-k", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_K", 50), help="CBO context-nearest historical sample count")
    parser.add_argument("--cbo-context-min-rows", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_MIN_ROWS", 40), help="global_context_threshold: minimum rows before weak/recent fallback")
    parser.add_argument("--cbo-context-recent-keep", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_RECENT_KEEP", 20), help="global_context_threshold: recent rows kept as current-state fallback")
    parser.add_argument("--cbo-context-weak-fallback-k", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_WEAK_FALLBACK_K", 40), help="global_context_threshold: max weak-similarity rows used only when strong rows are insufficient")
    parser.add_argument("--cbo-global-context-sim-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_GLOBAL_CONTEXT_SIM_THRESHOLD", 0.70), help="global_context_threshold: default similarity threshold when --cbo-context-sim-threshold is not positive")
    parser.add_argument("--cbo-external-sim-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_EXTERNAL_INTERNAL_SIM_THRESHOLD", 0.75), help="external_internal_threshold: external-stage similarity threshold")
    parser.add_argument("--cbo-external-topk", type=int, default=getattr(CFG, "DEFAULT_CBO_EXTERNAL_INTERNAL_TOPK", 400), help="external_internal_threshold: max externally similar rows before internal filtering")
    parser.add_argument("--cbo-external-min-rows", type=int, default=getattr(CFG, "DEFAULT_CBO_EXTERNAL_INTERNAL_MIN_ROWS", 40), help="external_internal_threshold: minimum external rows before external weak fallback")
    parser.add_argument("--cbo-external-recent-keep", type=int, default=getattr(CFG, "DEFAULT_CBO_EXTERNAL_INTERNAL_RECENT_KEEP", 20), help="external_internal_threshold: recent rows kept as final current-state fallback")
    parser.add_argument("--cbo-elite-k", type=int, default=getattr(CFG, "DEFAULT_CBO_ELITE_K", 20), help="CBO robust elite theta-region count")
    parser.add_argument("--cbo-diverse-k", type=int, default=getattr(CFG, "DEFAULT_CBO_DIVERSE_K", 20), help="CBO diversity sample count for hybrid history selection")
    parser.add_argument("--cbo-robust-score-mode", choices=["none", "mean", "mean_std", "context_weighted_mean_std"], default=getattr(CFG, "DEFAULT_CBO_ROBUST_SCORE_MODE", "none"), help="Robust elite/incumbent score mode")
    parser.add_argument("--cbo-robust-std-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_ROBUST_STD_WEIGHT", 0.5), help="Std penalty coefficient in robust score")
    parser.add_argument("--cbo-theta-merge-eps", type=float, default=getattr(CFG, "DEFAULT_CBO_THETA_MERGE_EPS", 0.05), help="Theta-region merge epsilon in normalized theta space")
    parser.add_argument("--cbo-context-sim-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_SIM_THRESHOLD", 0.0), help="Minimum context similarity for context-aware history/robust incumbent; 0 disables hard filtering")
    parser.add_argument("--cbo-state-kernel-topk", type=int, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_TOPK", 100), help="state_gated_kernel: top-k gated/product-kernel historical rows")
    parser.add_argument("--cbo-state-kernel-min-rows", type=int, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MIN_ROWS", 20), help="state_gated_kernel: minimum selected rows before fallback")
    parser.add_argument("--cbo-state-kernel-recent-keep", type=int, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_RECENT_KEEP", 20), help="state_gated_kernel: always keep this many recent rows")
    parser.add_argument("--cbo-state-kernel-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_THRESHOLD", 0.05), help="state_gated_kernel: minimum product-kernel similarity")
    parser.add_argument("--cbo-state-kernel-fallback", choices=["recent", "recent_context", "all"], default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_FALLBACK", "recent_context"), help="state_gated_kernel fallback when too few rows pass hard gate")
    parser.add_argument("--cbo-state-kernel-max-workload-dist", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_WORKLOAD_DIST", 3.0), help="state_gated_kernel hard gate max workload group distance")
    parser.add_argument("--cbo-state-kernel-max-state-dist", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_STATE_DIST", 3.0), help="state_gated_kernel hard gate max state group distance")
    parser.add_argument("--cbo-state-kernel-max-trend-dist", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_TREND_DIST", 3.0), help="state_gated_kernel hard gate max trend group distance")
    parser.add_argument("--cbo-state-kernel-max-unfinished-diff", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_UNFINISHED_DIFF", 0.30), help="state_gated_kernel hard gate max unfinished-rate absolute diff")
    parser.add_argument("--cbo-state-kernel-max-backlog-diff", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_BACKLOG_DIFF", 0.50), help="state_gated_kernel hard gate max normalized backlog absolute diff")
    parser.add_argument("--cbo-state-kernel-trend-sign-veto", action="store_true", default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_TREND_SIGN_VETO", True), help="state_gated_kernel: veto opposite trend signs when trends are non-trivial")
    parser.add_argument("--no-cbo-state-kernel-trend-sign-veto", dest="cbo_state_kernel_trend_sign_veto", action="store_false", help="Disable state_gated_kernel trend sign veto")
    parser.add_argument("--cbo-state-kernel-trend-sign-min", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_TREND_SIGN_MIN", 0.02), help="state_gated_kernel: minimum normalized trend magnitude for sign veto")
    parser.add_argument("--cbo-state-kernel-rate-gain", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_RATE_GAIN", 1.0), help="state_gated_kernel: amplify trend/rate distance before kernel similarity; 1 keeps old behavior")
    parser.add_argument("--cbo-state-kernel-rate-power", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_RATE_POWER", 1.0), help="state_gated_kernel: nonlinear power for amplified rate distance; 1 keeps old behavior")
    parser.add_argument("--cbo-state-kernel-max-rate-dist", type=float, default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_MAX_RATE_DIST", 3.0), help="state_gated_kernel hard gate max amplified rate/trend distance")
    parser.add_argument("--cbo-state-kernel-rate-sign-veto", action="store_true", default=getattr(CFG, "DEFAULT_CBO_STATE_KERNEL_RATE_SIGN_VETO", True), help="state_gated_kernel: veto opposite signs for rate/trend features")
    parser.add_argument("--no-cbo-state-kernel-rate-sign-veto", dest="cbo_state_kernel_rate_sign_veto", action="store_false", help="Disable state_gated_kernel rate sign veto")
    parser.add_argument("--cbo-tr-mode", choices=["off", "good_region", "adaptive", "residual_adaptive", "condition_adaptive"], default=getattr(CFG, "DEFAULT_CBO_TR_MODE", "off"), help="CBO trust-region mode")
    parser.add_argument("--cbo-tr-radius-init", type=float, default=getattr(CFG, "TRUST_RADIUS_INIT", 0.10), help="CBO TR initial radius")
    parser.add_argument("--cbo-tr-radius-min", type=float, default=getattr(CFG, "TRUST_RADIUS_MIN", 0.04), help="CBO TR min radius")
    parser.add_argument("--cbo-tr-radius-max", type=float, default=getattr(CFG, "TRUST_RADIUS_MAX", 0.35), help="CBO TR max radius")
    parser.add_argument("--cbo-tr-grow", type=float, default=getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15), help="CBO TR grow factor")
    parser.add_argument("--cbo-tr-shrink", type=float, default=getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92), help="CBO TR shrink factor")
    parser.add_argument("--cbo-tr-update-mode", choices=["best_so_far", "rolling_mean", "ewma_trend"], default=getattr(CFG, "DEFAULT_CBO_TR_UPDATE_MODE", "best_so_far"), help="CBO TR radius update rule")
    parser.add_argument("--cbo-tr-compare-window", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_COMPARE_WINDOW", 30), help="Recent window for rolling/ewma TR trend comparison")
    parser.add_argument("--cbo-tr-baseline-window", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_BASELINE_WINDOW", 60), help="Baseline window before the compare window for TR trend comparison")
    parser.add_argument("--cbo-tr-improve-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_IMPROVE_PCT", 0.015), help="Rolling improvement threshold for TR refine/shrink")
    parser.add_argument("--cbo-tr-worsen-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_WORSEN_PCT", 0.03), help="Rolling worsening threshold for TR expand/grow")
    parser.add_argument("--cbo-tr-deadband-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_DEADBAND_PCT", 0.01), help="Deadband threshold for holding TR radius")
    parser.add_argument("--cbo-tr-update-patience", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_UPDATE_PATIENCE", 2), help="Consecutive trend signals needed before changing TR radius")
    parser.add_argument("--cbo-tr-anchor-mode", choices=["posterior_mean", "robust_elite", "recent_best", "context_best"], default=getattr(CFG, "DEFAULT_CBO_TR_ANCHOR_MODE", "posterior_mean"), help="CBO TR anchor selection mode")
    parser.add_argument("--cbo-robust-incumbent-mode", choices=["off", "recommend_only", "deploy"], default=getattr(CFG, "DEFAULT_CBO_ROBUST_INCUMBENT_MODE", "off"), help="Robust incumbent diagnostic/deploy mode; default off")
    parser.add_argument("--cbo-macro-gate-mode", choices=["off", "topk", "weighted_topk", "hierarchical"], default=getattr(CFG, "DEFAULT_CBO_MACRO_GATE_MODE", "off"), help="CBO macro workload gate mode")
    parser.add_argument("--cbo-macro-k", type=int, default=getattr(CFG, "DEFAULT_CBO_MACRO_K", 100), help="CBO macro workload topK historical samples")
    parser.add_argument("--cbo-macro-total-scale", type=str, default=getattr(CFG, "DEFAULT_CBO_MACRO_TOTAL_SCALE", "auto"), help="Macro total arrivals scale; auto or numeric")
    parser.add_argument("--cbo-macro-lengthscale-total", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_TOTAL", 1.0), help="RBF lengthscale for normalized total arrivals")
    parser.add_argument("--cbo-macro-lengthscale-rt", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_RT", 0.15), help="RBF lengthscale for RT ratio")
    parser.add_argument("--cbo-macro-lengthscale-batch", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_BATCH", 0.15), help="RBF lengthscale for Batch ratio")
    parser.add_argument("--cbo-macro-alpha", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_ALPHA", 1.0), help="Macro similarity exponent")
    parser.add_argument("--cbo-dump-candidates", action="store_true", default=bool(getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES", False)), help="Dump CBO candidate theta diagnostics")
    parser.add_argument("--cbo-dump-candidates-every", type=int, default=getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES_EVERY", 20), help="Dump candidate diagnostics every N iterations")
    parser.add_argument("--cbo-dump-candidates-topn", type=int, default=getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES_TOPN", 30), help="Candidate diagnostic rows per dump, ranked by acquisition plus selected")
    parser.add_argument("--cbo-select-mode", choices=["greedy", "topk_stochastic", "epsilon_greedy", "randomized_ucb"], default=getattr(CFG, "DEFAULT_CBO_SELECT_MODE", "greedy"), help="CBO candidate selection mode; greedy preserves old behavior")
    parser.add_argument("--cbo-topk", type=int, default=getattr(CFG, "DEFAULT_CBO_TOPK", 5), help="Top-K size for stochastic CBO selection")
    parser.add_argument("--cbo-select-temperature", type=float, default=getattr(CFG, "DEFAULT_CBO_SELECT_TEMPERATURE", 0.20), help="Softmax temperature for topK stochastic selection")
    parser.add_argument("--cbo-epsilon", type=float, default=getattr(CFG, "DEFAULT_CBO_EPSILON", 0.10), help="Epsilon for epsilon-greedy selection under triggered exploration")
    parser.add_argument("--cbo-acq-beta", type=float, default=getattr(CFG, "DEFAULT_CBO_ACQ_BETA", 3.0), help="Acquisition beta for mu + beta*sigma scoring")
    parser.add_argument("--cbo-acq-beta-mode", choices=["fixed", "radius_adaptive", "radius_state_adaptive"], default=getattr(CFG, "DEFAULT_CBO_ACQ_BETA_MODE", "fixed"), help="CBO acquisition beta schedule")
    parser.add_argument("--cbo-beta-min", type=float, default=getattr(CFG, "DEFAULT_CBO_BETA_MIN", 0.1), help="Minimum beta for radius-adaptive acquisition")
    parser.add_argument("--cbo-beta-max", type=float, default=getattr(CFG, "DEFAULT_CBO_BETA_MAX", 2.0), help="Maximum beta for radius-adaptive acquisition")
    parser.add_argument("--cbo-radius-beta-power", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_POWER", 1.0), help="Power on normalized TR radius when computing adaptive beta")
    parser.add_argument("--cbo-radius-stable-rebound-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_STABLE_REBOUND_PCT", 0.02), help="Stable rebound threshold for beta diagnostics")
    parser.add_argument("--cbo-radius-unstable-rebound-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04), help="Rebound threshold that boosts radius_state_adaptive beta")
    parser.add_argument("--cbo-radius-surprise-boost-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0), help="Surprise threshold that boosts radius_state_adaptive beta")
    parser.add_argument("--cbo-radius-beta-boost", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_BOOST", 1.5), help="Multiplicative beta boost for unstable radius_state_adaptive state")
    parser.add_argument("--cbo-radius-beta-cap", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_CAP", 3.0), help="Upper cap after radius_state_adaptive beta boost")
    parser.add_argument("--cbo-good-region-guard", choices=["on", "off"], default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_GUARD", "off"), help="Enable alpha_direct deployment fallback to the best rolling good region")
    parser.add_argument("--cbo-good-region-window", type=int, default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_WINDOW", 50), help="Rolling window used to maintain the good-region deployment anchor")
    parser.add_argument("--cbo-good-region-worse-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_WORSE_PCT", 0.03), help="Relative rolling-cost degradation that triggers good-region fallback")
    parser.add_argument("--cbo-good-region-distance-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35), help="Normalized theta distance from good region that triggers fallback")
    parser.add_argument("--cbo-good-region-tr-radius-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15), help="TR radius threshold that triggers good-region fallback")
    parser.add_argument("--cbo-good-region-beta-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_BETA_THRESHOLD", 0.5), help="Acquisition beta threshold that triggers good-region fallback")
    parser.add_argument("--cbo-good-region-guard-mode", choices=["conservative", "distance_only", "performance_only"], default=getattr(CFG, "DEFAULT_CBO_GOOD_REGION_GUARD_MODE", "conservative"), help="Condition set used by the good-region deployment guard")
    parser.add_argument("--cbo-warm-start-history", type=str, default=getattr(CFG, "DEFAULT_CBO_WARM_START_HISTORY", ""), help="CSV file or directory containing bo_warm_history.csv rows to seed target CBO history")
    parser.add_argument("--cbo-warm-start-mode", choices=["none", "all", "similar_topk"], default=getattr(CFG, "DEFAULT_CBO_WARM_START_MODE", "none"), help="Warm-start target CBO from exported BO history; default none")
    parser.add_argument("--cbo-warm-start-topk", type=int, default=getattr(CFG, "DEFAULT_CBO_WARM_START_TOPK", 100), help="similar_topk rows kept by target initial-context distance")
    parser.add_argument("--cbo-warm-start-max-rows", type=int, default=getattr(CFG, "DEFAULT_CBO_WARM_START_MAX_ROWS", 300), help="Maximum compatible warm-start rows injected into a target CBO agent")
    parser.add_argument("--cbo-warm-start-label", type=str, default=getattr(CFG, "DEFAULT_CBO_WARM_START_LABEL", ""), help="Label written into source_scene_label when exporting bo_warm_history.csv")
    parser.add_argument("--cbo-history-denoise-mode", choices=["off", "local_median", "local_outlier_filter", "strict_local_outlier_filter"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_MODE", "off"), help="Denoise BO training targets before GP fit; default off preserves legacy behavior")
    parser.add_argument("--cbo-history-denoise-k", type=int, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_K", 7), help="Max nearest neighbors used for history denoising")
    parser.add_argument("--cbo-history-denoise-radius", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_RADIUS", 0.12), help="Combined normalized theta/context radius for history denoising")
    parser.add_argument("--cbo-history-denoise-min-neighbors", type=int, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_MIN_NEIGHBORS", 3), help="Minimum neighbors required before smoothing a training target")
    parser.add_argument("--cbo-history-denoise-context-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_CONTEXT_WEIGHT", 1.0), help="Context distance weight for history denoising")
    parser.add_argument("--cbo-history-denoise-theta-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_THETA_WEIGHT", 1.0), help="Theta distance weight for history denoising")
    parser.add_argument("--cbo-history-denoise-stat", choices=["median", "trimmed_mean"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_STAT", "median"), help="Robust statistic used for denoised training targets")
    parser.add_argument("--cbo-history-denoise-trim-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_TRIM_PCT", 0.1), help="Two-sided trim fraction for trimmed_mean denoising")
    parser.add_argument("--cbo-history-denoise-apply-to", choices=["local", "warm", "all"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_DENOISE_APPLY_TO", "all"), help="Which selected training rows are eligible for denoising")
    parser.add_argument("--cbo-history-outlier-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_THRESHOLD", 3.0), help="Robust local z threshold for local_outlier_filter")
    parser.add_argument("--cbo-history-outlier-abs-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_ABS_THRESHOLD", 500.0), help="Absolute residual threshold for local_outlier_filter")
    parser.add_argument("--cbo-history-outlier-max-filter-ratio", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_MAX_FILTER_RATIO", 0.2), help="Maximum fraction of training rows filtered by local_outlier_filter")
    parser.add_argument("--cbo-history-outlier-scale", choices=["mad", "iqr", "std"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_SCALE", "mad"), help="Robust scale used by local_outlier_filter")
    parser.add_argument("--cbo-history-outlier-theta-radius", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_THETA_RADIUS", 0.12), help="Strict outlier filter normalized theta radius")
    parser.add_argument("--cbo-history-outlier-context-radius", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_CONTEXT_RADIUS", 0.50), help="Strict outlier filter normalized context radius")
    parser.add_argument("--cbo-history-outlier-min-peers", type=int, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_MIN_PEERS", 3), help="Strict outlier filter minimum leave-one-out peers")
    parser.add_argument("--cbo-history-outlier-use-leave-one-out", type=_parse_bool_arg, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_USE_LEAVE_ONE_OUT", True), help="Whether strict outlier peers exclude the row itself")
    parser.add_argument("--cbo-history-outlier-export-filtered", type=_parse_bool_arg, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_EXPORT_FILTERED", True), help="Export strict outlier filtered record details to CSV")
    parser.add_argument("--cbo-history-outlier-protect-pressure", action="store_true", default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_PROTECT_PRESSURE", False), help="Protect high-pressure strict outlier candidates from filtering")
    parser.add_argument("--cbo-history-outlier-pressure-quantile", type=float, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_PRESSURE_QUANTILE", 0.75), help="Quantile threshold for pressure protection")
    parser.add_argument("--cbo-history-outlier-protect-high-cost-only", type=_parse_bool_arg, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_PROTECT_HIGH_COST_ONLY", True), help="Only protect candidates that are higher-cost than local peers")
    parser.add_argument("--cbo-history-outlier-pressure-fields", type=str, default=getattr(CFG, "DEFAULT_CBO_HISTORY_OUTLIER_PRESSURE_FIELDS", "Avg_Delay,Backlog,unfinished_end,Violation"), help="Comma-separated pressure fields used by strict outlier protection")
    parser.add_argument("--cbo-service-guard-mode", choices=["off", "soft"], default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_MODE", "off"), help="Optional service-aware exploration score guard")
    parser.add_argument("--cbo-service-guard-delay-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_DELAY_PCT", 0.03), help="Delay degradation threshold for service guard")
    parser.add_argument("--cbo-service-guard-backlog-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03), help="Backlog degradation threshold for service guard")
    parser.add_argument("--cbo-surprise-window", type=int, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_WINDOW", 10), help="Window length for residual/surprise diagnostics")
    parser.add_argument("--cbo-surprise-z-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_Z_THRESHOLD", 2.0), help="Standardized prediction-error threshold for residual adaptive TR")
    parser.add_argument("--cbo-surprise-cost-gap-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_COST_GAP_PCT", 0.03), help="Cost gap over recent best required to trigger residual adaptive TR")
    parser.add_argument("--cbo-prediction-guard", choices=["off", "diagnostic", "active", "deploy"], default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD", "off"), help="Prediction-error-aware guard mode. diagnostic only logs whether the guard would trigger; active/deploy may reuse the incumbent when BO is underestimating risk.")
    parser.add_argument("--cbo-prediction-guard-window", type=int, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_WINDOW", 50), help="Recent prediction-error window for prediction guard diagnostics")
    parser.add_argument("--cbo-prediction-guard-min-history", type=int, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_MIN_HISTORY", 20), help="Minimum valid prediction-error records before prediction guard diagnostics become ready")
    parser.add_argument("--cbo-prediction-guard-bias-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_BIAS_THRESHOLD", 300.0), help="Prediction guard triggers when recent positive bias exceeds this threshold")
    parser.add_argument("--cbo-prediction-guard-underestimate-rate", type=float, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_UNDERESTIMATE_RATE", 0.65), help="Prediction guard triggers when recent underestimate rate exceeds this threshold")
    parser.add_argument("--cbo-prediction-guard-start-iter", type=int, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_START_ITER", 200), help="Earliest BO iteration where active prediction guard may change deployment")
    parser.add_argument("--cbo-prediction-guard-bias-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_BIAS_WEIGHT", 1.0), help="Positive prediction-bias weight in active prediction guard risk margin")
    parser.add_argument("--cbo-prediction-guard-mae-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_PREDICTION_GUARD_MAE_WEIGHT", 0.5), help="MAE weight in active prediction guard risk margin")
    parser.add_argument("--cbo-sigma-calibration", choices=["on", "off"], default=getattr(CFG, "DEFAULT_CBO_SIGMA_CALIBRATION", "off"), help="Calibrate CBO posterior sigma from recent prediction residuals")
    parser.add_argument("--cbo-sigma-calibration-buffer-size", type=int, default=getattr(CFG, "DEFAULT_CBO_SIGMA_CALIBRATION_BUFFER_SIZE", 50), help="Maximum recent residual/sigma pairs retained for calibration")
    parser.add_argument("--cbo-sigma-calibration-min-samples", type=int, default=getattr(CFG, "DEFAULT_CBO_SIGMA_CALIBRATION_MIN_SAMPLES", 10), help="Minimum calibration rows before estimating sigma scale")
    parser.add_argument("--cbo-sigma-calibration-use-in-acq", choices=["false", "soft", "adaptive", "true"], default=getattr(CFG, "DEFAULT_CBO_SIGMA_CALIBRATION_USE_IN_ACQ", "false"), help="Use diagnostic-only, fixed-soft, adaptive, or fully calibrated sigma exploration")
    parser.add_argument("--cbo-sigma-calibration-eta", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_CALIBRATION_ETA", 0.25), help="Soft-mode blend weight between raw and calibrated sigma")
    parser.add_argument("--cbo-adaptive-exploration-beta-max", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_BETA_MAX", 3.0), help="Maximum UCB beta used by adaptive exploration")
    parser.add_argument("--cbo-adaptive-exploration-eta-max", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_ETA_MAX", 0.25), help="Maximum calibrated-sigma blend used by adaptive exploration")
    parser.add_argument("--cbo-adaptive-exploration-window", type=int, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_WINDOW", 30), help="Recent cost window used to detect optimization stagnation")
    parser.add_argument("--cbo-adaptive-exploration-sample-target", type=int, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_SAMPLE_TARGET", 80), help="Effective local GP sample count at which data scarcity reaches zero")
    parser.add_argument("--cbo-adaptive-exploration-smoothing", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_SMOOTHING", 0.20), help="EMA update weight for adaptive exploration demand")
    parser.add_argument("--cbo-adaptive-exploration-progress-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_PROGRESS_PCT", 0.01), help="Relative recent cost improvement treated as meaningful progress")
    parser.add_argument("--cbo-adaptive-exploration-reexplore-gain", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_REEXPLORE_GAIN", 0.25), help="Gain applied to stagnation-driven exploration after data scarcity is low")
    parser.add_argument("--cbo-adaptive-exploration-plausible-margin-mult", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_PLAUSIBLE_MARGIN_MULT", 2.0), help="Residual RMSE multiplier used by the adaptive candidate plausibility gate")
    parser.add_argument("--cbo-adaptive-exploration-backlog-ref", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_BACKLOG_REF", 1.0), help="Normalized backlog pressure mapped to full exploration suppression")
    parser.add_argument("--cbo-adaptive-exploration-unfinished-ref", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_UNFINISHED_REF", 0.10), help="Unfinished rate mapped to full exploration suppression")
    parser.add_argument("--cbo-adaptive-exploration-trend-ref", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_TREND_REF", 0.05), help="Positive unfinished-rate trend mapped to full exploration suppression")
    parser.add_argument("--cbo-adaptive-exploration-max-util-start", type=float, default=getattr(CFG, "DEFAULT_CBO_ADAPTIVE_EXPLORATION_MAX_UTIL_START", 0.80), help="Maximum utilization at which exploration suppression begins")
    parser.add_argument("--cbo-sigma-scale-default", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_SCALE_DEFAULT", 4.0), help="Default sigma scale before calibration is ready")
    parser.add_argument("--cbo-sigma-scale-min", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_SCALE_MIN", 1.0), help="Minimum calibrated sigma scale")
    parser.add_argument("--cbo-sigma-scale-max", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_SCALE_MAX", 6.0), help="Maximum calibrated sigma scale")
    parser.add_argument("--cbo-sigma-floor", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_FLOOR", 0.03), help="Floor applied to calibrated posterior sigma")
    parser.add_argument("--cbo-radius-reset", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_RESET", 0.12), help="TR radius after residual/condition soft reset")
    parser.add_argument("--cbo-radius-min-stuck-rounds", type=int, default=getattr(CFG, "DEFAULT_CBO_RADIUS_MIN_STUCK_ROUNDS", 10), help="Rounds stuck near min radius before condition trigger")
    parser.add_argument("--cbo-rebound-window", type=int, default=getattr(CFG, "DEFAULT_CBO_REBOUND_WINDOW", 20), help="Recent window for cost rebound trigger")
    parser.add_argument("--cbo-rebound-threshold-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_REBOUND_THRESHOLD_PCT", 0.03), help="Relative rebound threshold for condition adaptive TR")
    parser.add_argument("--cbo-selection-cooldown", type=int, default=getattr(CFG, "DEFAULT_CBO_SELECTION_COOLDOWN", 5), help="Number of future selections using exploratory selection after a trigger")
    parser.add_argument("--cbo-condition-anchor-switch", choices=["off", "recent_best", "context_best", "robust_elite"], default=getattr(CFG, "DEFAULT_CBO_CONDITION_ANCHOR_SWITCH", "context_best"), help="Temporary anchor override after residual/condition trigger")
    parser.add_argument("--cloud-delay-mult", type=float, default=1.0, help="只作用于云目标节点的传输时延倍率，>1 表示上云更慢")
    parser.add_argument("--cloud-energy-mult", type=float, default=1.0, help="只作用于云目标节点的传输能耗倍率，>1 表示上云更耗能")
    parser.add_argument("--cloud-speed-mult", "--cloud-service-rate-mult", dest="cloud_speed_mult", type=float, default=1.0, help="只作用于云节点服务率的倍率，<1 表示云服务率被削弱")
    parser.add_argument("--export-short-names", action="store_true", help="实验结束后复制一份短英文文件名结果到 output_root/_short_export，并打包 tar.gz")
    args = parser.parse_args()
    print("[BackupDefaults] no RoundRobin default methods = " + ",".join(DEFAULT_SCENARIO_KEYS))
    print(f"[BackupDefaults] BO_ITERATIONS={CFG.BO_ITERATIONS}, BO_INTERVAL={CFG.BO_INTERVAL}, SESSION_DURATION={CFG.SESSION_DURATION}")
    print(f"[BackupDefaults] feedback_score={args.feedback_score}, bo_history_mode={args.bo_history_mode}, bo_recent_window={args.bo_recent_window}, deploy_policy={args.deploy_policy}")
    print(f"[BackupDefaults] cbo_history_select_mode={args.cbo_history_select_mode}, cbo_robust_score_mode={args.cbo_robust_score_mode}, cbo_tr_mode={args.cbo_tr_mode}, cbo_robust_incumbent_mode={args.cbo_robust_incumbent_mode}")
    print(f"[BackupDefaults] cbo_macro_gate_mode={args.cbo_macro_gate_mode}, cbo_macro_k={args.cbo_macro_k}, cbo_dump_candidates={args.cbo_dump_candidates}")
    print(f"[BackupDefaults] cbo_select_mode={args.cbo_select_mode}, cbo_topk={args.cbo_topk}, cbo_surprise_window={args.cbo_surprise_window}, cbo_radius_reset={args.cbo_radius_reset}")
    if args.task_adaptation and args.no_task_adaptation:
        parser.error("--task-adaptation 和 --no-task-adaptation 不能同时使用")
    if args.task_adaptation:
        CFG.USE_TASK_TYPE_ADAPTATION = True
    if args.no_task_adaptation:
        CFG.USE_TASK_TYPE_ADAPTATION = False
    print(f"[TaskAdaptation] task_adaptation={bool(CFG.USE_TASK_TYPE_ADAPTATION)} field=task_node_affinity_factor")
    if args.fixed_rng:
        CFG.USE_FIXED_RNG = True
    if args.fixed_seed is not None:
        CFG.FIXED_RNG_SEED = args.fixed_seed
        CFG.BASE_SEED = args.fixed_seed
    if args.no_boltzmann_random:
        CFG.USE_BOLTZMANN_RANDOM = False
    if args.rt_deadline_factor is not None:
        CFG.TASK_PROPS["RT"]["deadline_factor"] = float(args.rt_deadline_factor)
        CFG.TASK_PROPS["RT"]["deadline"] = get_task_duration_reference(CFG.TASK_PROPS["RT"]) * float(args.rt_deadline_factor)
    if args.bo_iterations is not None:
        CFG.BO_ITERATIONS = int(args.bo_iterations)
    if args.bo_interval is not None:
        CFG.BO_INTERVAL = float(args.bo_interval)
    if args.session_duration is not None:
        CFG.SESSION_DURATION = float(args.session_duration)
    CFG.FEEDBACK_MODE = str(args.feedback_mode)
    CFG.BO_TRAINING_FEEDBACK_SCORE = str(args.feedback_score)
    CFG.CBO_REFERENCE_MODE = str(args.cbo_reference_mode)
    CFG.CBO_REFERENCE_CALIBRATION_ROUNDS = int(args.cbo_reference_calibration_rounds)
    CFG.CBO_REFERENCE_MIN_ROUNDS = int(args.cbo_reference_min_rounds)
    CFG.CBO_SHARED_REFERENCE_POLICY = str(args.cbo_shared_reference_policy)
    CFG.CBO_SHARED_REFERENCE_WARMUP_ROUNDS = int(args.cbo_shared_reference_warmup_rounds)
    CFG.CBO_REFERENCE_SOURCE_METHOD_KEY = str(args.cbo_reference_source_method_key)
    CFG.CBO_REFERENCE_STAT = str(args.cbo_reference_stat)
    CFG.CBO_REFERENCE_TRIM_PCT = float(args.cbo_reference_trim_pct)
    CFG.CBO_REFERENCE_FREEZE_AFTER_CALIBRATION = bool(args.cbo_reference_freeze_after_calibration)
    CFG.CBO_REFERENCE_FILE = str(args.cbo_reference_file)
    CFG.CBO_REFERENCE_OUTPUT_FILE = str(args.cbo_reference_output_file)
    CFG.PHASE_REFERENCE_WARMUP_ROUNDS = int(args.phase_reference_warmup_rounds)
    CFG.CBO_OBJECTIVE_MODE = str(args.cbo_objective_mode)
    CFG.CBO_TRADEOFF_ALPHA = float(args.cbo_tradeoff_alpha)
    CFG.CBO_ALPHA_MIN = float(args.cbo_alpha_min)
    CFG.CBO_ALPHA_MAX = float(args.cbo_alpha_max)
    CFG.CBO_TARGET_SUCCESS_RATE = float(args.cbo_target_success_rate)
    CFG.CBO_UNFINISHED_PENALTY_WEIGHT = float(args.cbo_unfinished_penalty_weight)
    CFG.CBO_SUCCESS_SHORTFALL_WEIGHT = float(args.cbo_success_shortfall_weight)
    CFG.CBO_BACKLOG_GROWTH_PENALTY_WEIGHT = float(args.cbo_backlog_growth_penalty_weight)
    CFG.CBO_CLASS_IMBALANCE_WEIGHT = float(args.cbo_class_imbalance_weight)
    CFG.CBO_NORMALIZED_RATIO_CLIP_MIN = float(args.cbo_normalized_ratio_clip_min)
    CFG.CBO_NORMALIZED_RATIO_CLIP_MAX = float(args.cbo_normalized_ratio_clip_max)
    CFG.SCHEDULER_TRADEOFF_MODE = str(args.scheduler_tradeoff_mode)
    CFG.SCHEDULER_TRADEOFF_ALPHA = float(args.scheduler_tradeoff_alpha)
    CFG.SCHEDULER_ALPHA_MIN = float(args.scheduler_alpha_min)
    CFG.SCHEDULER_ALPHA_MAX = float(args.scheduler_alpha_max)
    CFG.SCHEDULER_LE_SCALE = float(args.scheduler_le_scale)
    alpha_direct_bounds = _parse_bounds_pair_arg(args.alpha_direct_bounds, "--alpha-direct-bounds")
    alpha_direct_rt_bounds = _parse_bounds_pair_arg(args.alpha_direct_rt_bounds, "--alpha-direct-rt-bounds")
    alpha_direct_batch_bounds = _parse_bounds_pair_arg(args.alpha_direct_batch_bounds, "--alpha-direct-batch-bounds")
    alpha_direct_ai_bounds = _parse_bounds_pair_arg(args.alpha_direct_ai_bounds, "--alpha-direct-ai-bounds")
    alpha_direct_fixed_theta = _parse_float_list_arg(args.alpha_direct_fixed_theta, "--alpha-direct-fixed-theta", 6)
    reduced7_latency_weight_bounds = _parse_bounds_pair_arg(args.reduced7_latency_weight_bounds, "--reduced7-latency-weight-bounds")
    reduced7_queue_weight_bounds = _parse_bounds_pair_arg(args.reduced7_queue_weight_bounds, "--reduced7-queue-weight-bounds")
    reduced7_risk_scale_bounds = _parse_bounds_pair_arg(args.reduced7_risk_scale_bounds, "--reduced7-risk-scale-bounds")
    reduced7_cloud_gate_bounds = _parse_bounds_pair_arg(args.reduced7_cloud_gate_bounds, "--reduced7-cloud-gate-bounds")
    reduced7_energy_scale_bounds = _parse_bounds_pair_arg(args.reduced7_energy_scale_bounds, "--reduced7-energy-scale-bounds")
    CFG.ALPHA_DIRECT_BOUNDS = alpha_direct_bounds
    CFG.ALPHA_DIRECT_RT_BOUNDS = alpha_direct_rt_bounds
    CFG.ALPHA_DIRECT_BATCH_BOUNDS = alpha_direct_batch_bounds
    CFG.ALPHA_DIRECT_AI_BOUNDS = alpha_direct_ai_bounds
    CFG.ALPHA_DIRECT_FIXED_THETA = alpha_direct_fixed_theta
    CFG.REDUCED7_LATENCY_WEIGHT_BOUNDS = reduced7_latency_weight_bounds
    CFG.REDUCED7_QUEUE_WEIGHT_BOUNDS = reduced7_queue_weight_bounds
    CFG.REDUCED7_RISK_SCALE_BOUNDS = reduced7_risk_scale_bounds
    CFG.REDUCED7_CLOUD_GATE_BOUNDS = reduced7_cloud_gate_bounds
    CFG.REDUCED7_ENERGY_SCALE_BOUNDS = reduced7_energy_scale_bounds
    CFG.SCHEDULER_SERVICE_LATENCY_WEIGHT = float(args.scheduler_service_latency_weight)
    CFG.SCHEDULER_SERVICE_RISK_WEIGHT = float(args.scheduler_service_risk_weight)
    CFG.SCHEDULER_SERVICE_QUEUE_WEIGHT = float(args.scheduler_service_queue_weight)
    CFG.SCHEDULER_ENERGY_WEIGHT = float(args.scheduler_energy_weight)
    CFG.SCHEDULER_SCORE_NORM_MODE = str(args.scheduler_score_norm_mode)
    CFG.SCHEDULER_NORM_CLIP_MAX = float(args.scheduler_norm_clip_max)
    CFG.SCHEDULER_NORM_EPS = float(args.scheduler_norm_eps)
    CFG.SCHEDULER_NORM_EMA_ALPHA = float(args.scheduler_norm_ema_alpha)
    print(
        f"[SCHED-TRADEOFF] mode={CFG.SCHEDULER_TRADEOFF_MODE} "
        f"alpha={CFG.SCHEDULER_TRADEOFF_ALPHA} alpha_min={CFG.SCHEDULER_ALPHA_MIN} alpha_max={CFG.SCHEDULER_ALPHA_MAX} "
        f"le_scale={CFG.SCHEDULER_LE_SCALE} "
        f"service_latency_weight={CFG.SCHEDULER_SERVICE_LATENCY_WEIGHT} "
        f"service_risk_weight={CFG.SCHEDULER_SERVICE_RISK_WEIGHT} "
        f"service_queue_weight={CFG.SCHEDULER_SERVICE_QUEUE_WEIGHT} "
        f"energy_weight={CFG.SCHEDULER_ENERGY_WEIGHT}",
        flush=True,
    )
    print(
        f"[ALPHA-DIRECT-BOUNDS] uniform={CFG.ALPHA_DIRECT_BOUNDS} "
        f"rt={CFG.ALPHA_DIRECT_RT_BOUNDS} batch={CFG.ALPHA_DIRECT_BATCH_BOUNDS} ai={CFG.ALPHA_DIRECT_AI_BOUNDS} "
        f"fixed_theta={CFG.ALPHA_DIRECT_FIXED_THETA}",
        flush=True,
    )
    print(
        f"[REDUCED7-BOUNDS] latency={CFG.REDUCED7_LATENCY_WEIGHT_BOUNDS} "
        f"queue={CFG.REDUCED7_QUEUE_WEIGHT_BOUNDS} risk={CFG.REDUCED7_RISK_SCALE_BOUNDS} "
        f"cloud={CFG.REDUCED7_CLOUD_GATE_BOUNDS} energy_scale={CFG.REDUCED7_ENERGY_SCALE_BOUNDS}",
        flush=True,
    )
    print(
        f"[SCHED-NORM] mode={CFG.SCHEDULER_SCORE_NORM_MODE} "
        f"clip_max={CFG.SCHEDULER_NORM_CLIP_MAX} eps={CFG.SCHEDULER_NORM_EPS} "
        f"ema_alpha={CFG.SCHEDULER_NORM_EMA_ALPHA}",
        flush=True,
    )
    CFG.PAIRED_BASELINE_KEY = USER_METHOD_ALIASES.get(str(args.paired_baseline_key), USER_METHOD_ALIASES.get(str(args.paired_baseline_key).lower(), str(args.paired_baseline_key)))
    os.environ["BO_TRAINING_FEEDBACK_SCORE"] = str(args.feedback_score)
    os.environ["PAIRED_BASELINE_KEY"] = str(CFG.PAIRED_BASELINE_KEY)
    if args.bo_history_mode is not None:
        CFG.BO_HISTORY_MODE = str(args.bo_history_mode)
        os.environ["BO_HISTORY_MODE"] = str(args.bo_history_mode)
    if args.bo_recent_window is not None:
        CFG.BO_RECENT_WINDOW = int(args.bo_recent_window)
        os.environ["BO_RECENT_WINDOW"] = str(int(args.bo_recent_window))
    if args.bo_confidence_min is not None:
        CFG.BO_CONFIDENCE_MIN = float(args.bo_confidence_min)
        os.environ["BO_CONFIDENCE_MIN"] = str(float(args.bo_confidence_min))
    if args.bo_confidence_min_samples is not None:
        CFG.BO_CONFIDENCE_MIN_SAMPLES = int(args.bo_confidence_min_samples)
        os.environ["BO_CONFIDENCE_MIN_SAMPLES"] = str(int(args.bo_confidence_min_samples))
    if args.cbo_recent_window is not None:
        CFG.CBO_RECENT_WINDOW = int(args.cbo_recent_window)
        os.environ["CBO_RECENT_WINDOW"] = str(int(args.cbo_recent_window))
    CFG.CBO_HISTORY_SELECT_MODE = str(args.cbo_history_select_mode)
    CFG.CBO_CONTEXT_K = int(args.cbo_context_k)
    CFG.CBO_CONTEXT_MIN_ROWS = int(args.cbo_context_min_rows)
    CFG.CBO_CONTEXT_RECENT_KEEP = int(args.cbo_context_recent_keep)
    CFG.CBO_CONTEXT_WEAK_FALLBACK_K = int(args.cbo_context_weak_fallback_k)
    CFG.CBO_GLOBAL_CONTEXT_SIM_THRESHOLD = float(args.cbo_global_context_sim_threshold)
    CFG.CBO_EXTERNAL_INTERNAL_SIM_THRESHOLD = float(args.cbo_external_sim_threshold)
    CFG.CBO_EXTERNAL_INTERNAL_TOPK = int(args.cbo_external_topk)
    CFG.CBO_EXTERNAL_INTERNAL_MIN_ROWS = int(args.cbo_external_min_rows)
    CFG.CBO_EXTERNAL_INTERNAL_RECENT_KEEP = int(args.cbo_external_recent_keep)
    CFG.CBO_ELITE_K = int(args.cbo_elite_k)
    CFG.CBO_DIVERSE_K = int(args.cbo_diverse_k)
    CFG.CBO_ROBUST_SCORE_MODE = str(args.cbo_robust_score_mode)
    CFG.CBO_ROBUST_STD_WEIGHT = float(args.cbo_robust_std_weight)
    CFG.CBO_THETA_MERGE_EPS = float(args.cbo_theta_merge_eps)
    CFG.CBO_CONTEXT_SIM_THRESHOLD = float(args.cbo_context_sim_threshold)
    CFG.CBO_STATE_KERNEL_TOPK = int(args.cbo_state_kernel_topk)
    CFG.CBO_STATE_KERNEL_MIN_ROWS = int(args.cbo_state_kernel_min_rows)
    CFG.CBO_STATE_KERNEL_RECENT_KEEP = int(args.cbo_state_kernel_recent_keep)
    CFG.CBO_STATE_KERNEL_THRESHOLD = float(args.cbo_state_kernel_threshold)
    CFG.CBO_STATE_KERNEL_FALLBACK = str(args.cbo_state_kernel_fallback)
    CFG.CBO_STATE_KERNEL_MAX_WORKLOAD_DIST = float(args.cbo_state_kernel_max_workload_dist)
    CFG.CBO_STATE_KERNEL_MAX_STATE_DIST = float(args.cbo_state_kernel_max_state_dist)
    CFG.CBO_STATE_KERNEL_MAX_TREND_DIST = float(args.cbo_state_kernel_max_trend_dist)
    CFG.CBO_STATE_KERNEL_MAX_UNFINISHED_DIFF = float(args.cbo_state_kernel_max_unfinished_diff)
    CFG.CBO_STATE_KERNEL_MAX_BACKLOG_DIFF = float(args.cbo_state_kernel_max_backlog_diff)
    CFG.CBO_STATE_KERNEL_TREND_SIGN_VETO = bool(args.cbo_state_kernel_trend_sign_veto)
    CFG.CBO_STATE_KERNEL_TREND_SIGN_MIN = float(args.cbo_state_kernel_trend_sign_min)
    CFG.CBO_STATE_KERNEL_RATE_GAIN = float(args.cbo_state_kernel_rate_gain)
    CFG.CBO_STATE_KERNEL_RATE_POWER = float(args.cbo_state_kernel_rate_power)
    CFG.CBO_STATE_KERNEL_MAX_RATE_DIST = float(args.cbo_state_kernel_max_rate_dist)
    CFG.CBO_STATE_KERNEL_RATE_SIGN_VETO = bool(args.cbo_state_kernel_rate_sign_veto)
    CFG.CBO_TR_MODE = str(args.cbo_tr_mode)
    CFG.CBO_TR_RADIUS_INIT = float(args.cbo_tr_radius_init)
    CFG.CBO_TR_RADIUS_MIN = float(args.cbo_tr_radius_min)
    CFG.CBO_TR_RADIUS_MAX = float(args.cbo_tr_radius_max)
    CFG.CBO_TR_GROW = float(args.cbo_tr_grow)
    CFG.CBO_TR_SHRINK = float(args.cbo_tr_shrink)
    CFG.CBO_TR_UPDATE_MODE = str(args.cbo_tr_update_mode)
    CFG.CBO_TR_COMPARE_WINDOW = int(args.cbo_tr_compare_window)
    CFG.CBO_TR_BASELINE_WINDOW = int(args.cbo_tr_baseline_window)
    CFG.CBO_TR_IMPROVE_PCT = float(args.cbo_tr_improve_pct)
    CFG.CBO_TR_WORSEN_PCT = float(args.cbo_tr_worsen_pct)
    CFG.CBO_TR_DEADBAND_PCT = float(args.cbo_tr_deadband_pct)
    CFG.CBO_TR_UPDATE_PATIENCE = int(args.cbo_tr_update_patience)
    CFG.CBO_TR_ANCHOR_MODE = str(args.cbo_tr_anchor_mode)
    CFG.CBO_ROBUST_INCUMBENT_MODE = str(args.cbo_robust_incumbent_mode)
    CFG.CBO_MACRO_GATE_MODE = str(args.cbo_macro_gate_mode)
    CFG.CBO_MACRO_K = int(args.cbo_macro_k)
    CFG.CBO_MACRO_TOTAL_SCALE = str(args.cbo_macro_total_scale)
    CFG.CBO_MACRO_LENGTHSCALE_TOTAL = float(args.cbo_macro_lengthscale_total)
    CFG.CBO_MACRO_LENGTHSCALE_RT = float(args.cbo_macro_lengthscale_rt)
    CFG.CBO_MACRO_LENGTHSCALE_BATCH = float(args.cbo_macro_lengthscale_batch)
    CFG.CBO_MACRO_ALPHA = float(args.cbo_macro_alpha)
    CFG.CBO_DUMP_CANDIDATES = bool(args.cbo_dump_candidates)
    CFG.CBO_DUMP_CANDIDATES_EVERY = int(args.cbo_dump_candidates_every)
    CFG.CBO_DUMP_CANDIDATES_TOPN = int(args.cbo_dump_candidates_topn)
    CFG.CBO_SELECT_MODE = str(args.cbo_select_mode)
    CFG.CBO_TOPK = int(args.cbo_topk)
    CFG.CBO_SELECT_TEMPERATURE = float(args.cbo_select_temperature)
    CFG.CBO_EPSILON = float(args.cbo_epsilon)
    CFG.CBO_ACQ_BETA = float(args.cbo_acq_beta)
    CFG.CBO_ACQ_BETA_MODE = str(args.cbo_acq_beta_mode)
    CFG.CBO_BETA_MIN = float(args.cbo_beta_min)
    CFG.CBO_BETA_MAX = float(args.cbo_beta_max)
    CFG.CBO_RADIUS_BETA_POWER = float(args.cbo_radius_beta_power)
    CFG.CBO_RADIUS_STABLE_REBOUND_PCT = float(args.cbo_radius_stable_rebound_pct)
    CFG.CBO_RADIUS_UNSTABLE_REBOUND_PCT = float(args.cbo_radius_unstable_rebound_pct)
    CFG.CBO_RADIUS_SURPRISE_BOOST_THRESHOLD = float(args.cbo_radius_surprise_boost_threshold)
    CFG.CBO_RADIUS_BETA_BOOST = float(args.cbo_radius_beta_boost)
    CFG.CBO_RADIUS_BETA_CAP = float(args.cbo_radius_beta_cap)
    CFG.CBO_GOOD_REGION_GUARD = str(args.cbo_good_region_guard)
    CFG.CBO_GOOD_REGION_WINDOW = int(args.cbo_good_region_window)
    CFG.CBO_GOOD_REGION_WORSE_PCT = float(args.cbo_good_region_worse_pct)
    CFG.CBO_GOOD_REGION_DISTANCE_THRESHOLD = float(args.cbo_good_region_distance_threshold)
    CFG.CBO_GOOD_REGION_TR_RADIUS_THRESHOLD = float(args.cbo_good_region_tr_radius_threshold)
    CFG.CBO_GOOD_REGION_BETA_THRESHOLD = float(args.cbo_good_region_beta_threshold)
    CFG.CBO_GOOD_REGION_GUARD_MODE = str(args.cbo_good_region_guard_mode)
    CFG.CBO_WARM_START_HISTORY = str(args.cbo_warm_start_history or "")
    CFG.CBO_WARM_START_MODE = str(args.cbo_warm_start_mode)
    CFG.CBO_WARM_START_TOPK = int(args.cbo_warm_start_topk)
    CFG.CBO_WARM_START_MAX_ROWS = int(args.cbo_warm_start_max_rows)
    CFG.CBO_WARM_START_LABEL = str(args.cbo_warm_start_label or "")
    CFG.CBO_HISTORY_DENOISE_MODE = str(args.cbo_history_denoise_mode)
    CFG.CBO_HISTORY_DENOISE_K = int(args.cbo_history_denoise_k)
    CFG.CBO_HISTORY_DENOISE_RADIUS = float(args.cbo_history_denoise_radius)
    CFG.CBO_HISTORY_DENOISE_MIN_NEIGHBORS = int(args.cbo_history_denoise_min_neighbors)
    CFG.CBO_HISTORY_DENOISE_CONTEXT_WEIGHT = float(args.cbo_history_denoise_context_weight)
    CFG.CBO_HISTORY_DENOISE_THETA_WEIGHT = float(args.cbo_history_denoise_theta_weight)
    CFG.CBO_HISTORY_DENOISE_STAT = str(args.cbo_history_denoise_stat)
    CFG.CBO_HISTORY_DENOISE_TRIM_PCT = float(args.cbo_history_denoise_trim_pct)
    CFG.CBO_HISTORY_DENOISE_APPLY_TO = str(args.cbo_history_denoise_apply_to)
    CFG.CBO_HISTORY_OUTLIER_THRESHOLD = float(args.cbo_history_outlier_threshold)
    CFG.CBO_HISTORY_OUTLIER_ABS_THRESHOLD = float(args.cbo_history_outlier_abs_threshold)
    CFG.CBO_HISTORY_OUTLIER_MAX_FILTER_RATIO = float(args.cbo_history_outlier_max_filter_ratio)
    CFG.CBO_HISTORY_OUTLIER_SCALE = str(args.cbo_history_outlier_scale)
    CFG.CBO_HISTORY_OUTLIER_THETA_RADIUS = float(args.cbo_history_outlier_theta_radius)
    CFG.CBO_HISTORY_OUTLIER_CONTEXT_RADIUS = float(args.cbo_history_outlier_context_radius)
    CFG.CBO_HISTORY_OUTLIER_MIN_PEERS = int(args.cbo_history_outlier_min_peers)
    CFG.CBO_HISTORY_OUTLIER_USE_LEAVE_ONE_OUT = bool(args.cbo_history_outlier_use_leave_one_out)
    CFG.CBO_HISTORY_OUTLIER_EXPORT_FILTERED = bool(args.cbo_history_outlier_export_filtered)
    CFG.CBO_HISTORY_OUTLIER_PROTECT_PRESSURE = bool(args.cbo_history_outlier_protect_pressure)
    CFG.CBO_HISTORY_OUTLIER_PRESSURE_QUANTILE = float(args.cbo_history_outlier_pressure_quantile)
    CFG.CBO_HISTORY_OUTLIER_PROTECT_HIGH_COST_ONLY = bool(args.cbo_history_outlier_protect_high_cost_only)
    CFG.CBO_HISTORY_OUTLIER_PRESSURE_FIELDS = str(args.cbo_history_outlier_pressure_fields)
    CFG.CBO_SERVICE_GUARD_MODE = str(args.cbo_service_guard_mode)
    CFG.CBO_SERVICE_GUARD_DELAY_PCT = float(args.cbo_service_guard_delay_pct)
    CFG.CBO_SERVICE_GUARD_BACKLOG_PCT = float(args.cbo_service_guard_backlog_pct)
    CFG.CBO_SURPRISE_WINDOW = int(args.cbo_surprise_window)
    CFG.CBO_SURPRISE_Z_THRESHOLD = float(args.cbo_surprise_z_threshold)
    CFG.CBO_SURPRISE_COST_GAP_PCT = float(args.cbo_surprise_cost_gap_pct)
    CFG.CBO_PREDICTION_GUARD = str(args.cbo_prediction_guard)
    CFG.CBO_PREDICTION_GUARD_WINDOW = int(args.cbo_prediction_guard_window)
    CFG.CBO_PREDICTION_GUARD_MIN_HISTORY = int(args.cbo_prediction_guard_min_history)
    CFG.CBO_PREDICTION_GUARD_BIAS_THRESHOLD = float(args.cbo_prediction_guard_bias_threshold)
    CFG.CBO_PREDICTION_GUARD_UNDERESTIMATE_RATE = float(args.cbo_prediction_guard_underestimate_rate)
    CFG.CBO_PREDICTION_GUARD_START_ITER = int(args.cbo_prediction_guard_start_iter)
    CFG.CBO_PREDICTION_GUARD_BIAS_WEIGHT = float(args.cbo_prediction_guard_bias_weight)
    CFG.CBO_PREDICTION_GUARD_MAE_WEIGHT = float(args.cbo_prediction_guard_mae_weight)
    CFG.CBO_SIGMA_CALIBRATION = str(args.cbo_sigma_calibration)
    CFG.CBO_SIGMA_CALIBRATION_BUFFER_SIZE = int(args.cbo_sigma_calibration_buffer_size)
    CFG.CBO_SIGMA_CALIBRATION_MIN_SAMPLES = int(args.cbo_sigma_calibration_min_samples)
    CFG.CBO_SIGMA_CALIBRATION_USE_IN_ACQ = str(args.cbo_sigma_calibration_use_in_acq)
    CFG.CBO_SIGMA_CALIBRATION_ETA = float(np.clip(args.cbo_sigma_calibration_eta, 0.0, 1.0))
    CFG.CBO_ADAPTIVE_EXPLORATION_BETA_MAX = max(0.0, float(args.cbo_adaptive_exploration_beta_max))
    CFG.CBO_ADAPTIVE_EXPLORATION_ETA_MAX = float(np.clip(args.cbo_adaptive_exploration_eta_max, 0.0, 1.0))
    CFG.CBO_ADAPTIVE_EXPLORATION_WINDOW = max(4, int(args.cbo_adaptive_exploration_window))
    CFG.CBO_ADAPTIVE_EXPLORATION_SAMPLE_TARGET = max(1, int(args.cbo_adaptive_exploration_sample_target))
    CFG.CBO_ADAPTIVE_EXPLORATION_SMOOTHING = float(np.clip(args.cbo_adaptive_exploration_smoothing, 0.0, 1.0))
    CFG.CBO_ADAPTIVE_EXPLORATION_PROGRESS_PCT = max(1e-9, float(args.cbo_adaptive_exploration_progress_pct))
    CFG.CBO_ADAPTIVE_EXPLORATION_REEXPLORE_GAIN = float(np.clip(args.cbo_adaptive_exploration_reexplore_gain, 0.0, 1.0))
    CFG.CBO_ADAPTIVE_EXPLORATION_PLAUSIBLE_MARGIN_MULT = max(0.0, float(args.cbo_adaptive_exploration_plausible_margin_mult))
    CFG.CBO_ADAPTIVE_EXPLORATION_BACKLOG_REF = max(1e-9, float(args.cbo_adaptive_exploration_backlog_ref))
    CFG.CBO_ADAPTIVE_EXPLORATION_UNFINISHED_REF = max(1e-9, float(args.cbo_adaptive_exploration_unfinished_ref))
    CFG.CBO_ADAPTIVE_EXPLORATION_TREND_REF = max(1e-9, float(args.cbo_adaptive_exploration_trend_ref))
    CFG.CBO_ADAPTIVE_EXPLORATION_MAX_UTIL_START = float(np.clip(args.cbo_adaptive_exploration_max_util_start, 0.0, 0.999999))
    CFG.CBO_SIGMA_SCALE_DEFAULT = float(args.cbo_sigma_scale_default)
    CFG.CBO_SIGMA_SCALE_MIN = float(args.cbo_sigma_scale_min)
    CFG.CBO_SIGMA_SCALE_MAX = float(args.cbo_sigma_scale_max)
    CFG.CBO_SIGMA_FLOOR = float(args.cbo_sigma_floor)
    CFG.CBO_RADIUS_RESET = float(args.cbo_radius_reset)
    CFG.CBO_RADIUS_MIN_STUCK_ROUNDS = int(args.cbo_radius_min_stuck_rounds)
    CFG.CBO_REBOUND_WINDOW = int(args.cbo_rebound_window)
    CFG.CBO_REBOUND_THRESHOLD_PCT = float(args.cbo_rebound_threshold_pct)
    CFG.CBO_SELECTION_COOLDOWN = int(args.cbo_selection_cooldown)
    CFG.CBO_CONDITION_ANCHOR_SWITCH = str(args.cbo_condition_anchor_switch)
    if _argv_has_option("--cbo-tr-radius-init"):
        CFG.TRUST_RADIUS_INIT = float(args.cbo_tr_radius_init)
    if _argv_has_option("--cbo-tr-radius-min"):
        CFG.TRUST_RADIUS_MIN = float(args.cbo_tr_radius_min)
    if _argv_has_option("--cbo-tr-radius-max"):
        CFG.TRUST_RADIUS_MAX = float(args.cbo_tr_radius_max)
    if _argv_has_option("--cbo-tr-grow"):
        CFG.TRUST_RADIUS_GROWTH = float(args.cbo_tr_grow)
    if _argv_has_option("--cbo-tr-shrink"):
        CFG.TRUST_RADIUS_SHRINK = float(args.cbo_tr_shrink)
    if args.deploy_policy is not None:
        CFG.DEPLOY_POLICY_ARG = str(args.deploy_policy)
        os.environ["SAFEBO_POLICY"] = str(args.deploy_policy)
        os.environ["SAFEBO_POLICY_ARG"] = str(args.deploy_policy)
    if args.dual_refined_source is not None:
        CFG.DUAL_REFINED_SOURCE = str(args.dual_refined_source)
        os.environ["DUAL_REFINED_SOURCE"] = str(args.dual_refined_source)
    if args.dual_class_metric is not None:
        CFG.DUAL_CLASS_METRIC = str(args.dual_class_metric)
        os.environ["DUAL_CLASS_METRIC"] = str(args.dual_class_metric)
    CFG.CLOUD_DELAY_MULT = float(args.cloud_delay_mult)
    CFG.CLOUD_ENERGY_MULT = float(args.cloud_energy_mult)
    CFG.CLOUD_SPEED_MULT = float(args.cloud_speed_mult)
    CFG.CLOUD_SERVICE_RATE_MULT = float(args.cloud_speed_mult)
    fixed_task_probs = parse_task_probs_arg(args.task_probs)
    if fixed_task_probs is not None:
        CFG.TASK_TYPE_PROBS = fixed_task_probs
    task_prob_schedule = parse_task_prob_schedule_arg(args.task_prob_schedule)
    if task_prob_schedule is not None:
        CFG.TASK_TYPE_PROB_SCHEDULE = task_prob_schedule
    lambda_schedule = parse_lambda_schedule_arg(args.lambda_schedule)
    if lambda_schedule is not None:
        CFG.LAMBDA_SCHEDULE = lambda_schedule
        CFG.ARRIVAL_THRESHOLDS = infer_arrival_thresholds(CFG.LAMBDA_SCHEDULE)
        # 如果用户没手动覆盖 session_duration，则自动扩展到 schedule 末尾。
        if args.session_duration is None:
            CFG.SESSION_DURATION = max(float(x[1]) for x in CFG.LAMBDA_SCHEDULE)
    CFG.DYNAMIC_SCHEDULE = str(args.dynamic_schedule or "")
    CFG.DYNAMIC_HISTORY_MODE = str(args.dynamic_history_mode)
    CFG.DYNAMIC_HISTORY_WINDOW = int(args.dynamic_history_window)
    CFG.DYNAMIC_CONTEXT_TOPK = int(args.dynamic_context_topk)
    CFG.REPEAT_RUNS = max(1, args.repeat)
    if args.mode == "param":
        run_param_analysis(samples=args.samples, local_delta=args.local_delta)
    elif args.mode == "extreme":
        for _ in range(max(1, args.repeat)):
            run_extreme_param_test()
    elif args.mode == "scan":
        run_param_scan(dim_name=args.dim, points=args.points)
    elif args.mode == "sensitivity":
        run_full_sensitivity_analysis(
            points=max(2, args.points),
            seeds=max(1, args.sens_seeds),
            windows=max(1, args.sens_windows),
            scenario_spec=args.sensitivity_scenarios,
            output_dir=args.output_root,
            greedy=bool(args.sens_greedy),
            pref_norm_mode=args.pref_norm_mode,
        )
    elif args.mode == "scenario":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_scenario_method_experiments(repeat_runs=max(1, args.repeat), selected_keys=selected_keys, output_dir=args.output_root)
    elif args.mode == "pressure_scan":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_pressure_scan_experiments(
            repeat_runs=max(1, args.repeat),
            lambda_values=parse_lambda_values_arg(args.lambda_values),
            output_root=args.output_root,
            selected_keys=selected_keys,
            task_probs=parse_task_probs_arg(args.task_probs),
        )
    elif args.mode == "dynamic_scenario":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_dynamic_scenario_experiments(
            repeat_runs=max(1, args.repeat),
            selected_keys=selected_keys,
            output_dir=args.output_root,
            dynamic_schedule=args.dynamic_schedule,
            dynamic_history_mode=args.dynamic_history_mode,
            dynamic_history_window=args.dynamic_history_window,
            dynamic_context_topk=args.dynamic_context_topk,
        )
    elif args.mode == "offline_noise":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_offline_window_noise_diagnostic(
            repeat_runs=max(1, args.repeat),
            selected_keys=selected_keys,
            output_dir=args.output_root,
        )
    elif args.mode == "batch_federated":
        run_static_batch_federated_experiment(
            method=args.batch_method,
            n_clients=max(1, args.batch_clients),
            rounds=max(1, args.batch_rounds),
            task_count=max(1, args.batch_tasks),
            task_counts=args.batch_client_task_counts,
            seed=int(CFG.FIXED_RNG_SEED if args.fixed_seed is None else args.fixed_seed),
            output_root=args.output_root,
            topology_profile=args.batch_topology_profile,
            node_counts=args.batch_node_counts,
            task_probs_by_client=args.batch_client_task_probs,
            client_config_path=args.batch_client_config,
            context_features=args.batch_context_features,
            objective_weights=args.batch_objective_weights,
            normalization=args.batch_normalization,
            objective_clip=args.batch_objective_clip,
            batch_reuse_mode=args.batch_reuse_mode,
            batch_order=args.batch_order,
            scheduler_type="Boltzmann",
            norm_mode=args.pref_norm_mode,
            deterministic_scheduler=bool(args.batch_deterministic_scheduler),
            dispatch_gap=float(args.batch_dispatch_gap),
            fed_share_mode=args.batch_fed_share_mode,
            fed_candidate_count=max(1, args.batch_fed_candidates),
            fed_beta=args.batch_fed_beta,
        )
    elif args.mode == "ratio_grid":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_ratio_grid_experiments(
            repeat_runs=max(1, args.repeat),
            step=args.ratio_step,
            min_ratio=args.ratio_min,
            output_root=args.output_root,
            selected_keys=selected_keys,
        )
    else:
        main()

    if getattr(args, "export_short_names", False):
        export_root = args.output_root or SCENARIO_SAVE_DIR
        export_short_named_results(export_root)
