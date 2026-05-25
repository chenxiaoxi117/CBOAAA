#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 11636-12014.
# Original command-line interface block.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    def _parse_bounds_pair_arg(value, option_name):
        if value is None or str(value).strip() == "":
            return None
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

    parser.add_argument("--mode", choices=["all", "param", "extreme", "scan", "sensitivity", "scenario", "ratio_grid", "pressure_scan", "offline_noise"], default="all")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--local-delta", type=float, default=0.08)
    parser.add_argument("--dim", type=str, default="W_RT_Latency")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fixed-rng", action="store_true")
    parser.add_argument("--fixed-seed", type=int, default=None)
    parser.add_argument("--ratio-step", type=int, default=10, help="任务比例网格步长，默认10，即10%%")
    parser.add_argument("--ratio-min", type=int, default=10, help="每类任务最低比例，默认10，即至少10%%")
    parser.add_argument("--output-root", type=str, default=None, help="ratio_grid输出根目录")
    parser.add_argument("--selected-keys", type=str, default=None, help="逗号分隔的方法键；默认使用备份版 no-RoundRobin 方法集")
    parser.add_argument("--sensitivity-scenarios", type=str, default="default,rt_high,batch_high,ai_high", help="敏感度场景：default,rt_high,batch_high,ai_high 或 name:RT,Batch,AI")
    parser.add_argument("--sens-seeds", type=int, default=2, help="敏感度分析每个参数点重复的随机种子数")
    parser.add_argument("--sens-windows", type=int, default=3, help="敏感度分析每个 theta 连续评价的窗口数")
    parser.add_argument("--sens-greedy", action="store_true", help="敏感度分析时关闭 Boltzmann 随机，使用确定性机会集合选择")
    parser.add_argument("--pref-norm-mode", choices=["rolling", "fixed"], default="rolling", help="静态节点偏好诊断使用的归一化模式")
    parser.add_argument("--task-adaptation", action="store_true", help="启用 type_speed_factor，节点对不同任务类型有差异化速度")
    parser.add_argument("--no-task-adaptation", action="store_true", help="关闭 type_speed_factor，所有任务只使用节点基础 speed")
    parser.add_argument("--rt-deadline-factor", type=float, default=None, help="覆盖 RT 任务 deadline_factor，例如 2.5")
    parser.add_argument("--bo-iterations", type=int, default=None, help="覆盖 BO_ITERATIONS，便于快速测试")
    parser.add_argument("--bo-interval", type=float, default=None, help="覆盖 BO_INTERVAL")
    parser.add_argument("--session-duration", type=float, default=None, help="覆盖 SESSION_DURATION")
    parser.add_argument("--feedback-mode", choices=["window", "cohort_complete", "dual"], default="window", help="BO反馈模式：window为旧窗口级反馈；cohort_complete为任务批次完成后反馈；dual为窗口快反馈+批次/分类精反馈替换")
    parser.add_argument("--feedback-score", choices=["window_original", "task_effective", "task_effective_backlog", "task_effective_backlog_violation", "paired_fixed_mid_delta", "legacy_dual", "legacy_cohort"], default=getattr(CFG, "DEFAULT_SCENARIO_FEEDBACK_SCORE", "window_original"), help="BO tell 使用的训练反馈；默认 window_original，即 BO_Training_Cost=Eval_Cost。paired_fixed_mid_delta 为仿真专用：同窗口 shadow fixed_mid 的 delta cost。")
    parser.add_argument("--cbo-reference-mode", choices=["off", "calibrate", "load", "auto_macro"], default="off", help="Scenario reference baseline mode for normalized metrics")
    parser.add_argument("--cbo-reference-calibration-rounds", type=int, default=30, help="Rounds used to build/freeze scenario reference")
    parser.add_argument("--cbo-reference-min-rounds", type=int, default=5, help="Minimum rounds before reference is considered available")
    parser.add_argument("--cbo-reference-stat", choices=["median", "trimmed_mean", "mean"], default="median", help="Statistic used for reference calibration")
    parser.add_argument("--cbo-reference-trim-pct", type=float, default=0.1, help="Trim percent for trimmed_mean reference")
    parser.add_argument("--cbo-reference-freeze-after-calibration", action="store_true", default=True, help="Freeze reference after calibration window")
    parser.add_argument("--cbo-reference-file", type=str, default="", help="JSON reference file to load")
    parser.add_argument("--cbo-reference-output-file", type=str, default="", help="JSON reference output file")
    parser.add_argument("--cbo-objective-mode", choices=["eval_cost", "diagnostic_only", "normalized_tradeoff"], default="eval_cost", help="BO training objective mode")
    parser.add_argument("--cbo-tradeoff-alpha", type=float, default=0.8, help="alpha in alpha*service_norm + (1-alpha)*energy_norm")
    parser.add_argument("--cbo-alpha-min", type=float, default=0.6, help="minimum clipped alpha")
    parser.add_argument("--cbo-alpha-max", type=float, default=0.95, help="maximum clipped alpha")
    parser.add_argument("--cbo-target-success-rate", type=float, default=0.995, help="target SLA success rate for normalized service score")
    parser.add_argument("--cbo-unfinished-penalty-weight", type=float, default=5.0, help="unfinished_rate penalty in service_norm")
    parser.add_argument("--cbo-success-shortfall-weight", type=float, default=2.0, help="success shortfall normalized penalty in service_norm")
    parser.add_argument("--cbo-backlog-growth-penalty-weight", type=float, default=2.0, help="backlog growth rate penalty in service_norm")
    parser.add_argument("--cbo-class-imbalance-weight", type=float, default=0.0, help="class completion imbalance penalty in service_norm")
    parser.add_argument("--cbo-normalized-ratio-clip-min", type=float, default=0.2, help="min clip for normalized ratios")
    parser.add_argument("--cbo-normalized-ratio-clip-max", type=float, default=5.0, help="max clip for normalized ratios")
    parser.add_argument("--scheduler-tradeoff-mode", choices=["legacy", "alpha_fixed", "alpha_from_ratio", "alpha_direct"], default="legacy", help="底层调度器节点 score 的 service-energy tradeoff 模式；默认 legacy 保持旧逻辑")
    parser.add_argument("--scheduler-tradeoff-alpha", type=float, default=0.85, help="alpha_fixed 模式下的 service 权重 alpha")
    parser.add_argument("--scheduler-alpha-min", type=float, default=0.60, help="scheduler alpha 下限")
    parser.add_argument("--scheduler-alpha-max", type=float, default=0.97, help="scheduler alpha 上限")
    parser.add_argument("--alpha-direct-bounds", type=str, default=None, help="alpha_direct uniform BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-rt-bounds", type=str, default=None, help="alpha_direct RT BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-batch-bounds", type=str, default=None, help="alpha_direct Batch BO bounds, formatted as low,high")
    parser.add_argument("--alpha-direct-ai-bounds", type=str, default=None, help="alpha_direct AI BO bounds, formatted as low,high")
    parser.add_argument("--scheduler-service-latency-weight", type=float, default=1.0, help="alpha tradeoff 中 norm_l 的系数")
    parser.add_argument("--scheduler-service-risk-weight", type=float, default=1.0, help="alpha 外部 risk_w*norm_risk 惩罚项的额外系数")
    parser.add_argument("--scheduler-service-queue-weight", type=float, default=1.0, help="alpha 外部 queue_w*norm_queue 惩罚项的额外系数")
    parser.add_argument("--scheduler-energy-weight", type=float, default=1.0, help="energy_component 中 norm_e 的系数")
    parser.add_argument("--scheduler-score-norm-mode", choices=["legacy", "candidate_median", "candidate_iqr", "rolling_ema"], default="legacy", help="底层调度器 score 的 energy/latency 归一化模式；默认 legacy 保持 norm_mode 行为")
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
    parser.add_argument("--bo-history-mode", choices=["all", "recent", "confidence", "recent_confidence"], default=getattr(CFG, "DEFAULT_BO_HISTORY_MODE", "recent"), help="BO GP训练历史使用方式；备份版默认 recent。all=全部；recent=最近N个；confidence=过滤低可信反馈；recent_confidence=最近+可信度过滤")
    parser.add_argument("--bo-recent-window", type=int, default=getattr(CFG, "DEFAULT_BO_RECENT_WINDOW", 80), help="recent/recent_confidence 模式保留最近多少个BO样本；备份版默认80")
    parser.add_argument("--bo-confidence-min", type=float, default=None, help="confidence/recent_confidence 模式下保留样本的最低反馈可信度")
    parser.add_argument("--bo-confidence-min-samples", type=int, default=None, help="可信过滤后至少保留多少样本，不足时自动使用高可信+最近样本兜底")
    parser.add_argument("--cbo-history-select-mode", choices=["recent", "recent_context", "recent_context_elite", "hybrid"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_SELECT_MODE", "recent"), help="CBO-only stability history selection mode")
    parser.add_argument("--cbo-context-k", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_K", 50), help="CBO context-nearest historical sample count")
    parser.add_argument("--cbo-elite-k", type=int, default=getattr(CFG, "DEFAULT_CBO_ELITE_K", 20), help="CBO robust elite theta-region count")
    parser.add_argument("--cbo-diverse-k", type=int, default=getattr(CFG, "DEFAULT_CBO_DIVERSE_K", 20), help="CBO diversity sample count for hybrid history selection")
    parser.add_argument("--cbo-robust-score-mode", choices=["none", "mean", "mean_std", "context_weighted_mean_std"], default=getattr(CFG, "DEFAULT_CBO_ROBUST_SCORE_MODE", "none"), help="Robust elite/incumbent score mode")
    parser.add_argument("--cbo-robust-std-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_ROBUST_STD_WEIGHT", 0.5), help="Std penalty coefficient in robust score")
    parser.add_argument("--cbo-theta-merge-eps", type=float, default=getattr(CFG, "DEFAULT_CBO_THETA_MERGE_EPS", 0.05), help="Theta-region merge epsilon in normalized theta space")
    parser.add_argument("--cbo-context-sim-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_SIM_THRESHOLD", 0.0), help="Minimum context similarity for context-aware history/robust incumbent; 0 disables hard filtering")
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
    parser.add_argument("--cbo-service-guard-mode", choices=["off", "soft"], default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_MODE", "off"), help="Optional service-aware exploration score guard")
    parser.add_argument("--cbo-service-guard-delay-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_DELAY_PCT", 0.03), help="Delay degradation threshold for service guard")
    parser.add_argument("--cbo-service-guard-backlog-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03), help="Backlog degradation threshold for service guard")
    parser.add_argument("--cbo-surprise-window", type=int, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_WINDOW", 10), help="Window length for residual/surprise diagnostics")
    parser.add_argument("--cbo-surprise-z-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_Z_THRESHOLD", 2.0), help="Standardized prediction-error threshold for residual adaptive TR")
    parser.add_argument("--cbo-surprise-cost-gap-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_COST_GAP_PCT", 0.03), help="Cost gap over recent best required to trigger residual adaptive TR")
    parser.add_argument("--cbo-sigma-floor", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_FLOOR", 1e-6), help="Sigma floor for surprise calculation")
    parser.add_argument("--cbo-radius-reset", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_RESET", 0.12), help="TR radius after residual/condition soft reset")
    parser.add_argument("--cbo-radius-min-stuck-rounds", type=int, default=getattr(CFG, "DEFAULT_CBO_RADIUS_MIN_STUCK_ROUNDS", 10), help="Rounds stuck near min radius before condition trigger")
    parser.add_argument("--cbo-rebound-window", type=int, default=getattr(CFG, "DEFAULT_CBO_REBOUND_WINDOW", 20), help="Recent window for cost rebound trigger")
    parser.add_argument("--cbo-rebound-threshold-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_REBOUND_THRESHOLD_PCT", 0.03), help="Relative rebound threshold for condition adaptive TR")
    parser.add_argument("--cbo-selection-cooldown", type=int, default=getattr(CFG, "DEFAULT_CBO_SELECTION_COOLDOWN", 5), help="Number of future selections using exploratory selection after a trigger")
    parser.add_argument("--cbo-condition-anchor-switch", choices=["off", "recent_best", "context_best", "robust_elite"], default=getattr(CFG, "DEFAULT_CBO_CONDITION_ANCHOR_SWITCH", "context_best"), help="Temporary anchor override after residual/condition trigger")
    parser.add_argument("--cloud-delay-mult", type=float, default=1.0, help="只作用于云目标节点的传输时延倍率，>1 表示上云更慢")
    parser.add_argument("--cloud-energy-mult", type=float, default=1.0, help="只作用于云目标节点的传输能耗倍率，>1 表示上云更耗能")
    parser.add_argument("--cloud-speed-mult", type=float, default=1.0, help="只作用于云节点算力的速度倍率，<1 表示云算力被削弱")
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
    if args.fixed_rng:
        CFG.USE_FIXED_RNG = True
    if args.fixed_seed is not None:
        CFG.FIXED_RNG_SEED = args.fixed_seed
    if args.rt_deadline_factor is not None:
        CFG.TASK_PROPS["RT"]["deadline_factor"] = float(args.rt_deadline_factor)
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
    CFG.CBO_REFERENCE_STAT = str(args.cbo_reference_stat)
    CFG.CBO_REFERENCE_TRIM_PCT = float(args.cbo_reference_trim_pct)
    CFG.CBO_REFERENCE_FREEZE_AFTER_CALIBRATION = bool(args.cbo_reference_freeze_after_calibration)
    CFG.CBO_REFERENCE_FILE = str(args.cbo_reference_file)
    CFG.CBO_REFERENCE_OUTPUT_FILE = str(args.cbo_reference_output_file)
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
    alpha_direct_bounds = _parse_bounds_pair_arg(args.alpha_direct_bounds, "--alpha-direct-bounds")
    alpha_direct_rt_bounds = _parse_bounds_pair_arg(args.alpha_direct_rt_bounds, "--alpha-direct-rt-bounds")
    alpha_direct_batch_bounds = _parse_bounds_pair_arg(args.alpha_direct_batch_bounds, "--alpha-direct-batch-bounds")
    alpha_direct_ai_bounds = _parse_bounds_pair_arg(args.alpha_direct_ai_bounds, "--alpha-direct-ai-bounds")
    CFG.ALPHA_DIRECT_BOUNDS = alpha_direct_bounds
    CFG.ALPHA_DIRECT_RT_BOUNDS = alpha_direct_rt_bounds
    CFG.ALPHA_DIRECT_BATCH_BOUNDS = alpha_direct_batch_bounds
    CFG.ALPHA_DIRECT_AI_BOUNDS = alpha_direct_ai_bounds
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
        f"service_latency_weight={CFG.SCHEDULER_SERVICE_LATENCY_WEIGHT} "
        f"service_risk_weight={CFG.SCHEDULER_SERVICE_RISK_WEIGHT} "
        f"service_queue_weight={CFG.SCHEDULER_SERVICE_QUEUE_WEIGHT} "
        f"energy_weight={CFG.SCHEDULER_ENERGY_WEIGHT}",
        flush=True,
    )
    print(
        f"[ALPHA-DIRECT-BOUNDS] uniform={CFG.ALPHA_DIRECT_BOUNDS} "
        f"rt={CFG.ALPHA_DIRECT_RT_BOUNDS} batch={CFG.ALPHA_DIRECT_BATCH_BOUNDS} ai={CFG.ALPHA_DIRECT_AI_BOUNDS}",
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
    CFG.CBO_HISTORY_SELECT_MODE = str(args.cbo_history_select_mode)
    CFG.CBO_CONTEXT_K = int(args.cbo_context_k)
    CFG.CBO_ELITE_K = int(args.cbo_elite_k)
    CFG.CBO_DIVERSE_K = int(args.cbo_diverse_k)
    CFG.CBO_ROBUST_SCORE_MODE = str(args.cbo_robust_score_mode)
    CFG.CBO_ROBUST_STD_WEIGHT = float(args.cbo_robust_std_weight)
    CFG.CBO_THETA_MERGE_EPS = float(args.cbo_theta_merge_eps)
    CFG.CBO_CONTEXT_SIM_THRESHOLD = float(args.cbo_context_sim_threshold)
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
    CFG.CBO_SERVICE_GUARD_MODE = str(args.cbo_service_guard_mode)
    CFG.CBO_SERVICE_GUARD_DELAY_PCT = float(args.cbo_service_guard_delay_pct)
    CFG.CBO_SERVICE_GUARD_BACKLOG_PCT = float(args.cbo_service_guard_backlog_pct)
    CFG.CBO_SURPRISE_WINDOW = int(args.cbo_surprise_window)
    CFG.CBO_SURPRISE_Z_THRESHOLD = float(args.cbo_surprise_z_threshold)
    CFG.CBO_SURPRISE_COST_GAP_PCT = float(args.cbo_surprise_cost_gap_pct)
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
    elif args.mode == "offline_noise":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_offline_window_noise_diagnostic(
            repeat_runs=max(1, args.repeat),
            selected_keys=selected_keys,
            output_dir=args.output_root,
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
