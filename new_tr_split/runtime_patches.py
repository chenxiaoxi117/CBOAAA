#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 7139-11221.
# Deploy/history/stability overrides, dual feedback, CBO/TR patches, final scenario runner override.

_DUAL_ORIG_USE_COHORT_FEEDBACK = ConnectedFactory._use_cohort_feedback
_DUAL_ORIG_ON_TASK_FINISHED_COHORT = ConnectedFactory._on_task_finished_cohort


def _safebo_env(name, default, cast=str):
    try:
        return cast(os.environ.get(name, default))
    except Exception:
        return default


def _safebo_policy_name(group_cfg=None):
    p = None
    if group_cfg is not None:
        p = group_cfg.get("deploy_policy", None)
    if p is None:
        p = os.environ.get("SAFEBO_POLICY", "ei")
    return str(p or "ei").strip().lower()


def _is_cbo_method_key(group_key, group_cfg=None):
    key = str(group_key or "").lower()
    family = str((group_cfg or {}).get("method_family", "")).lower() if isinstance(group_cfg, dict) else ""
    label = str((group_cfg or {}).get("label", "")).lower() if isinstance(group_cfg, dict) else ""
    return ("cbo" in key) or ("cbo" in family) or ("cbo" in label)


def _deploy_policy_arg():
    val = getattr(CFG, "DEPLOY_POLICY_ARG", None)
    if val is None:
        val = os.environ.get("SAFEBO_POLICY_ARG", None)
    return str(val).strip().lower() if val is not None and str(val).strip() else None


def apply_deploy_policy_override(groups):
    """Apply CLI deploy-policy only to CBO-like methods; fixed/direct baselines stay unchanged."""
    policy = _deploy_policy_arg()
    if not policy:
        return groups
    for group_key, group_cfg in groups.items():
        if _is_cbo_method_key(group_key, group_cfg):
            group_cfg["deploy_policy"] = policy
            group_cfg["deploy_policy_source"] = "cli_override_cbo_only"
        else:
            group_cfg.setdefault("deploy_policy_source", "method_default_or_not_applicable")
    return groups


def method_deploy_policy_map(groups):
    out = {}
    for group_key, group_cfg in (groups or {}).items():
        out[str(group_key)] = {
            "deploy_policy": group_cfg.get("deploy_policy"),
            "deploy_policy_source": group_cfg.get("deploy_policy_source", "method_default"),
            "method_family": group_cfg.get("method_family"),
            "is_cbo_method": bool(_is_cbo_method_key(group_key, group_cfg)),
            "alpha_direct_fixed_theta_enabled": bool(group_cfg.get("alpha_direct_fixed_theta_enabled", False)),
            "alpha_direct_fixed_theta_6d": group_cfg.get("alpha_direct_fixed_theta_6d"),
        }
    return out


def _argv_has_option(option_name):
    """Return True when an argparse option was explicitly supplied.

    Supports both "--x value" and "--x=value" forms. This is needed because
    argparse defaults are non-None for backwards compatibility, while method
    configs may have their own defaults that should only be overridden by an
    explicit CLI request.
    """
    opt = str(option_name)
    return any(str(arg) == opt or str(arg).startswith(opt + "=") for arg in sys.argv[1:])


def _history_mode_arg():
    if not _argv_has_option("--bo-history-mode"):
        return None
    return _cfg_history_mode()


def _recent_window_arg():
    if not _argv_has_option("--bo-recent-window"):
        return None
    return _cfg_recent_window()


def apply_history_policy_override(groups):
    """Apply CLI history settings to CBO-like methods only.

    Method configs such as cbo_lite_group() can define their own history_mode /
    recent_window defaults. A CLI --bo-history-mode or --bo-recent-window should
    override those CBO method-level defaults, while fixed/direct baselines are
    unaffected. If no CLI flag was explicitly supplied, preserve method defaults.
    """
    mode = _history_mode_arg()
    window = _recent_window_arg()
    if mode is None and window is None:
        for group_key, group_cfg in (groups or {}).items():
            group_cfg.setdefault("history_override_source", "method_default")
        return groups
    for group_key, group_cfg in (groups or {}).items():
        if _is_cbo_method_key(group_key, group_cfg):
            if mode is not None:
                group_cfg["history_mode"] = str(mode)
            if window is not None:
                group_cfg["recent_window"] = int(window)
            group_cfg["history_override_source"] = "cli_override_cbo_only"
        else:
            group_cfg.setdefault("history_override_source", "method_default_or_not_applicable")
    return groups


def _cbo_cli_option(name, attr, default=None):
    if not _argv_has_option(name):
        return None
    return getattr(CFG, attr, default)


def apply_cbo_stability_policy_override(groups):
    values = {
        "cbo_history_select_mode": _cbo_cli_option("--cbo-history-select-mode", "CBO_HISTORY_SELECT_MODE", "recent"),
        "cbo_context_k": _cbo_cli_option("--cbo-context-k", "CBO_CONTEXT_K", 50),
        "cbo_elite_k": _cbo_cli_option("--cbo-elite-k", "CBO_ELITE_K", 20),
        "cbo_diverse_k": _cbo_cli_option("--cbo-diverse-k", "CBO_DIVERSE_K", 20),
        "cbo_robust_score_mode": _cbo_cli_option("--cbo-robust-score-mode", "CBO_ROBUST_SCORE_MODE", "none"),
        "cbo_robust_std_weight": _cbo_cli_option("--cbo-robust-std-weight", "CBO_ROBUST_STD_WEIGHT", 0.5),
        "cbo_theta_merge_eps": _cbo_cli_option("--cbo-theta-merge-eps", "CBO_THETA_MERGE_EPS", 0.05),
        "cbo_context_sim_threshold": _cbo_cli_option("--cbo-context-sim-threshold", "CBO_CONTEXT_SIM_THRESHOLD", 0.0),
        "cbo_tr_mode": _cbo_cli_option("--cbo-tr-mode", "CBO_TR_MODE", "off"),
        "cbo_tr_radius_init": _cbo_cli_option("--cbo-tr-radius-init", "CBO_TR_RADIUS_INIT", getattr(CFG, "TRUST_RADIUS_INIT", 0.10)),
        "cbo_tr_radius_min": _cbo_cli_option("--cbo-tr-radius-min", "CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04)),
        "cbo_tr_radius_max": _cbo_cli_option("--cbo-tr-radius-max", "CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35)),
        "cbo_tr_grow": _cbo_cli_option("--cbo-tr-grow", "CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15)),
        "cbo_tr_shrink": _cbo_cli_option("--cbo-tr-shrink", "CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92)),
        "cbo_tr_update_mode": _cbo_cli_option("--cbo-tr-update-mode", "CBO_TR_UPDATE_MODE", "best_so_far"),
        "cbo_tr_compare_window": _cbo_cli_option("--cbo-tr-compare-window", "CBO_TR_COMPARE_WINDOW", 30),
        "cbo_tr_baseline_window": _cbo_cli_option("--cbo-tr-baseline-window", "CBO_TR_BASELINE_WINDOW", 60),
        "cbo_tr_improve_pct": _cbo_cli_option("--cbo-tr-improve-pct", "CBO_TR_IMPROVE_PCT", 0.015),
        "cbo_tr_worsen_pct": _cbo_cli_option("--cbo-tr-worsen-pct", "CBO_TR_WORSEN_PCT", 0.03),
        "cbo_tr_deadband_pct": _cbo_cli_option("--cbo-tr-deadband-pct", "CBO_TR_DEADBAND_PCT", 0.01),
        "cbo_tr_update_patience": _cbo_cli_option("--cbo-tr-update-patience", "CBO_TR_UPDATE_PATIENCE", 2),
        "cbo_tr_anchor_mode": _cbo_cli_option("--cbo-tr-anchor-mode", "CBO_TR_ANCHOR_MODE", "posterior_mean"),
        "cbo_robust_incumbent_mode": _cbo_cli_option("--cbo-robust-incumbent-mode", "CBO_ROBUST_INCUMBENT_MODE", "off"),
        "cbo_macro_gate_mode": _cbo_cli_option("--cbo-macro-gate-mode", "CBO_MACRO_GATE_MODE", "off"),
        "cbo_macro_k": _cbo_cli_option("--cbo-macro-k", "CBO_MACRO_K", 100),
        "cbo_macro_total_scale": _cbo_cli_option("--cbo-macro-total-scale", "CBO_MACRO_TOTAL_SCALE", "auto"),
        "cbo_macro_lengthscale_total": _cbo_cli_option("--cbo-macro-lengthscale-total", "CBO_MACRO_LENGTHSCALE_TOTAL", 1.0),
        "cbo_macro_lengthscale_rt": _cbo_cli_option("--cbo-macro-lengthscale-rt", "CBO_MACRO_LENGTHSCALE_RT", 0.15),
        "cbo_macro_lengthscale_batch": _cbo_cli_option("--cbo-macro-lengthscale-batch", "CBO_MACRO_LENGTHSCALE_BATCH", 0.15),
        "cbo_macro_alpha": _cbo_cli_option("--cbo-macro-alpha", "CBO_MACRO_ALPHA", 1.0),
        "cbo_dump_candidates": _cbo_cli_option("--cbo-dump-candidates", "CBO_DUMP_CANDIDATES", False),
        "cbo_dump_candidates_every": _cbo_cli_option("--cbo-dump-candidates-every", "CBO_DUMP_CANDIDATES_EVERY", 20),
        "cbo_dump_candidates_topn": _cbo_cli_option("--cbo-dump-candidates-topn", "CBO_DUMP_CANDIDATES_TOPN", 30),
        "cbo_select_mode": _cbo_cli_option("--cbo-select-mode", "CBO_SELECT_MODE", "greedy"),
        "cbo_topk": _cbo_cli_option("--cbo-topk", "CBO_TOPK", 5),
        "cbo_select_temperature": _cbo_cli_option("--cbo-select-temperature", "CBO_SELECT_TEMPERATURE", 0.20),
        "cbo_epsilon": _cbo_cli_option("--cbo-epsilon", "CBO_EPSILON", 0.10),
        "cbo_acq_beta": _cbo_cli_option("--cbo-acq-beta", "CBO_ACQ_BETA", 3.0),
        "cbo_acq_beta_mode": _cbo_cli_option("--cbo-acq-beta-mode", "CBO_ACQ_BETA_MODE", "fixed"),
        "cbo_beta_min": _cbo_cli_option("--cbo-beta-min", "CBO_BETA_MIN", 0.1),
        "cbo_beta_max": _cbo_cli_option("--cbo-beta-max", "CBO_BETA_MAX", 2.0),
        "cbo_radius_beta_power": _cbo_cli_option("--cbo-radius-beta-power", "CBO_RADIUS_BETA_POWER", 1.0),
        "cbo_radius_stable_rebound_pct": _cbo_cli_option("--cbo-radius-stable-rebound-pct", "CBO_RADIUS_STABLE_REBOUND_PCT", 0.02),
        "cbo_radius_unstable_rebound_pct": _cbo_cli_option("--cbo-radius-unstable-rebound-pct", "CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04),
        "cbo_radius_surprise_boost_threshold": _cbo_cli_option("--cbo-radius-surprise-boost-threshold", "CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0),
        "cbo_radius_beta_boost": _cbo_cli_option("--cbo-radius-beta-boost", "CBO_RADIUS_BETA_BOOST", 1.5),
        "cbo_radius_beta_cap": _cbo_cli_option("--cbo-radius-beta-cap", "CBO_RADIUS_BETA_CAP", 3.0),
        "cbo_good_region_guard": _cbo_cli_option("--cbo-good-region-guard", "CBO_GOOD_REGION_GUARD", "off"),
        "cbo_good_region_window": _cbo_cli_option("--cbo-good-region-window", "CBO_GOOD_REGION_WINDOW", 50),
        "cbo_good_region_worse_pct": _cbo_cli_option("--cbo-good-region-worse-pct", "CBO_GOOD_REGION_WORSE_PCT", 0.03),
        "cbo_good_region_distance_threshold": _cbo_cli_option("--cbo-good-region-distance-threshold", "CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35),
        "cbo_good_region_tr_radius_threshold": _cbo_cli_option("--cbo-good-region-tr-radius-threshold", "CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15),
        "cbo_good_region_beta_threshold": _cbo_cli_option("--cbo-good-region-beta-threshold", "CBO_GOOD_REGION_BETA_THRESHOLD", 0.5),
        "cbo_good_region_guard_mode": _cbo_cli_option("--cbo-good-region-guard-mode", "CBO_GOOD_REGION_GUARD_MODE", "conservative"),
        "cbo_service_guard_mode": _cbo_cli_option("--cbo-service-guard-mode", "CBO_SERVICE_GUARD_MODE", "off"),
        "cbo_service_guard_delay_pct": _cbo_cli_option("--cbo-service-guard-delay-pct", "CBO_SERVICE_GUARD_DELAY_PCT", 0.03),
        "cbo_service_guard_backlog_pct": _cbo_cli_option("--cbo-service-guard-backlog-pct", "CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03),
        "cbo_surprise_window": _cbo_cli_option("--cbo-surprise-window", "CBO_SURPRISE_WINDOW", 10),
        "cbo_surprise_z_threshold": _cbo_cli_option("--cbo-surprise-z-threshold", "CBO_SURPRISE_Z_THRESHOLD", 2.0),
        "cbo_surprise_cost_gap_pct": _cbo_cli_option("--cbo-surprise-cost-gap-pct", "CBO_SURPRISE_COST_GAP_PCT", 0.03),
        "cbo_sigma_floor": _cbo_cli_option("--cbo-sigma-floor", "CBO_SIGMA_FLOOR", 1e-6),
        "cbo_radius_reset": _cbo_cli_option("--cbo-radius-reset", "CBO_RADIUS_RESET", 0.12),
        "cbo_radius_min_stuck_rounds": _cbo_cli_option("--cbo-radius-min-stuck-rounds", "CBO_RADIUS_MIN_STUCK_ROUNDS", 10),
        "cbo_rebound_window": _cbo_cli_option("--cbo-rebound-window", "CBO_REBOUND_WINDOW", 20),
        "cbo_rebound_threshold_pct": _cbo_cli_option("--cbo-rebound-threshold-pct", "CBO_REBOUND_THRESHOLD_PCT", 0.03),
        "cbo_selection_cooldown": _cbo_cli_option("--cbo-selection-cooldown", "CBO_SELECTION_COOLDOWN", 5),
        "cbo_condition_anchor_switch": _cbo_cli_option("--cbo-condition-anchor-switch", "CBO_CONDITION_ANCHOR_SWITCH", "context_best"),
    }
    any_explicit = any(v is not None for v in values.values())
    for group_key, group_cfg in (groups or {}).items():
        if _is_cbo_method_key(group_key, group_cfg):
            if any_explicit:
                for k, v in values.items():
                    if v is not None:
                        group_cfg[k] = v
                group_cfg["cbo_stability_override_source"] = "cli_override_cbo_only"
            else:
                group_cfg.setdefault("cbo_history_select_mode", "recent")
                group_cfg.setdefault("cbo_robust_score_mode", "none")
                group_cfg.setdefault("cbo_tr_mode", "off")
                group_cfg.setdefault("cbo_robust_incumbent_mode", "off")
                group_cfg.setdefault("cbo_macro_gate_mode", "off")
                group_cfg.setdefault("cbo_dump_candidates", False)
                group_cfg.setdefault("cbo_stability_override_source", "method_default")
        else:
            group_cfg.setdefault("cbo_stability_override_source", "not_applicable")
    return groups


def apply_alpha_direct_fixed_theta_override(groups):
    theta = getattr(CFG, "ALPHA_DIRECT_FIXED_THETA", None)
    if theta is None:
        return groups
    if not isinstance(theta, (list, tuple, np.ndarray)) or len(theta) != 6:
        raise ValueError("ALPHA_DIRECT_FIXED_THETA must be a 6D sequence")
    clip_fn = globals().get("clip_alpha_direct_control_vector")
    theta6 = clip_fn(theta) if callable(clip_fn) else [float(x) for x in theta]
    theta6 = [float(x) for x in list(theta6)[:6]]
    alpha_direct_keys = {"reduced6_cbo_alpha_direct", "reduced6_cbo_alpha_direct_no_risk"}
    for group_key, group_cfg in (groups or {}).items():
        if str(group_key) not in alpha_direct_keys:
            group_cfg.setdefault("alpha_direct_fixed_theta_enabled", False)
            group_cfg.setdefault("alpha_direct_fixed_theta_6d", None)
            continue
        if str(group_cfg.get("control_mode", "")).strip().lower() != "alpha_direct":
            continue
        group_cfg["agent"] = None
        group_cfg["agent_kwargs"] = None
        group_cfg["fixed_theta"] = list(theta6)
        group_cfg["deploy_policy"] = "fixed"
        group_cfg["deploy_policy_source"] = "alpha_direct_fixed_theta_cli"
        group_cfg["scheduler_tradeoff_mode"] = "alpha_direct"
        group_cfg["alpha_direct_fixed_theta_enabled"] = True
        group_cfg["alpha_direct_fixed_theta_6d"] = list(theta6)
    return groups


def method_history_policy_map(groups):
    out = {}
    for group_key, group_cfg in (groups or {}).items():
        out[str(group_key)] = {
            "history_mode": group_cfg.get("history_mode"),
            "recent_window": group_cfg.get("recent_window"),
            "history_override_source": group_cfg.get("history_override_source", "method_default"),
            "method_family": group_cfg.get("method_family"),
            "is_cbo_method": bool(_is_cbo_method_key(group_key, group_cfg)),
            "history_select_mode": group_cfg.get("cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")),
            "effective_history_mode": group_cfg.get("history_mode", _cfg_history_mode("all")),
            "effective_recent_window": group_cfg.get("recent_window", _cfg_recent_window()),
            "context_k": group_cfg.get("cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50)),
            "elite_k": group_cfg.get("cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20)),
            "diverse_k": group_cfg.get("cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20)),
            "robust_score_mode": group_cfg.get("cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")),
            "scheduler_tradeoff_mode": str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy")),
            "scheduler_score_norm_mode": str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_use_score_risk": group_cfg.get("scheduler_use_score_risk", bool(getattr(CFG, "USE_SCORE_RISK", True))),
            "scheduler_tradeoff_alpha": float(getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85)),
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "scheduler_le_scale": float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)),
            "alpha_direct_bounds": getattr(CFG, "ALPHA_DIRECT_BOUNDS", None),
            "alpha_direct_rt_bounds": getattr(CFG, "ALPHA_DIRECT_RT_BOUNDS", None),
            "alpha_direct_batch_bounds": getattr(CFG, "ALPHA_DIRECT_BATCH_BOUNDS", None),
            "alpha_direct_ai_bounds": getattr(CFG, "ALPHA_DIRECT_AI_BOUNDS", None),
            "tr_mode": group_cfg.get("cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")),
            "tr_anchor_mode": group_cfg.get("cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")),
            "tr_update_mode": group_cfg.get("cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far")),
            "tr_compare_window": group_cfg.get("cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)),
            "tr_baseline_window": group_cfg.get("cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)),
            "tr_improve_pct": group_cfg.get("cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015)),
            "tr_worsen_pct": group_cfg.get("cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03)),
            "tr_deadband_pct": group_cfg.get("cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01)),
            "tr_update_patience": group_cfg.get("cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)),
            "robust_incumbent_mode": group_cfg.get("cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")),
            "macro_gate_mode": group_cfg.get("cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")),
            "macro_k": group_cfg.get("cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100)),
            "macro_total_scale": group_cfg.get("cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")),
            "macro_lengthscale_total": group_cfg.get("cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0)),
            "macro_lengthscale_rt": group_cfg.get("cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15)),
            "macro_lengthscale_batch": group_cfg.get("cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15)),
            "macro_alpha": group_cfg.get("cbo_macro_alpha", _cfg_cbo_float("CBO_MACRO_ALPHA", 1.0)),
            "dump_candidates": group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False))),
            "dump_candidates_every": group_cfg.get("cbo_dump_candidates_every", _cfg_cbo_int("CBO_DUMP_CANDIDATES_EVERY", 20)),
            "dump_candidates_topn": group_cfg.get("cbo_dump_candidates_topn", _cfg_cbo_int("CBO_DUMP_CANDIDATES_TOPN", 30)),
            "select_mode": group_cfg.get("cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")),
            "topk": group_cfg.get("cbo_topk", _cfg_cbo_int("CBO_TOPK", 5)),
            "select_temperature": group_cfg.get("cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20)),
            "epsilon": group_cfg.get("cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)),
            "acq_beta": group_cfg.get("cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)),
            "acq_beta_mode": group_cfg.get("cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")),
            "beta_min": group_cfg.get("cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1)),
            "beta_max": group_cfg.get("cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0)),
            "radius_beta_power": group_cfg.get("cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0)),
            "good_region_guard": group_cfg.get("cbo_good_region_guard", _cfg_cbo_str("CBO_GOOD_REGION_GUARD", "off")),
            "good_region_window": group_cfg.get("cbo_good_region_window", _cfg_cbo_int("CBO_GOOD_REGION_WINDOW", 50)),
            "good_region_worse_pct": group_cfg.get("cbo_good_region_worse_pct", _cfg_cbo_float("CBO_GOOD_REGION_WORSE_PCT", 0.03)),
            "good_region_distance_threshold": group_cfg.get("cbo_good_region_distance_threshold", _cfg_cbo_float("CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35)),
            "good_region_tr_radius_threshold": group_cfg.get("cbo_good_region_tr_radius_threshold", _cfg_cbo_float("CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15)),
            "good_region_beta_threshold": group_cfg.get("cbo_good_region_beta_threshold", _cfg_cbo_float("CBO_GOOD_REGION_BETA_THRESHOLD", 0.5)),
            "good_region_guard_mode": group_cfg.get("cbo_good_region_guard_mode", _cfg_cbo_str("CBO_GOOD_REGION_GUARD_MODE", "conservative")),
            "radius_stable_rebound_pct": group_cfg.get("cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)),
            "radius_unstable_rebound_pct": group_cfg.get("cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)),
            "radius_surprise_boost_threshold": group_cfg.get("cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)),
            "radius_beta_boost": group_cfg.get("cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5)),
            "radius_beta_cap": group_cfg.get("cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0)),
            "service_guard_mode": group_cfg.get("cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off")),
            "service_guard_delay_pct": group_cfg.get("cbo_service_guard_delay_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_DELAY_PCT", 0.03)),
            "service_guard_backlog_pct": group_cfg.get("cbo_service_guard_backlog_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03)),
            "surprise_window": group_cfg.get("cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10)),
            "surprise_z_threshold": group_cfg.get("cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)),
            "surprise_cost_gap_pct": group_cfg.get("cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)),
            "radius_reset": group_cfg.get("cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)),
            "radius_min_stuck_rounds": group_cfg.get("cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)),
            "rebound_window": group_cfg.get("cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20)),
            "rebound_threshold_pct": group_cfg.get("cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)),
            "selection_cooldown": group_cfg.get("cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5)),
            "condition_anchor_switch": group_cfg.get("cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best")),
        }
    return out


def _dual_feedback_mode():
    return str(getattr(CFG, "FEEDBACK_MODE", "window")).strip().lower()


def _dual_is_enabled():
    return _dual_feedback_mode() in {"dual", "dual_feedback", "window_cohort", "window_refine"}


def _dual_patch_use_cohort_feedback(self):
    mode = _dual_feedback_mode()
    if mode in {"dual", "dual_feedback", "window_cohort", "window_refine"}:
        return True
    return _DUAL_ORIG_USE_COHORT_FEEDBACK(self)


ConnectedFactory._use_cohort_feedback = _dual_patch_use_cohort_feedback


def _dual_agent_replace_sample(agent, sample_id, refined_cost, refined_meta=None):
    """Replace the provisional window feedback of one BO sample by delayed refined feedback.

    The agent stores y=-cost in local_recent. In dual mode we first append a provisional
    window sample, then later replace that sample's y with the refined cohort/class score.
    The previous window cost is saved for diagnosis/export.
    """
    if agent is None or sample_id is None:
        return False
    refined_y = -float(refined_cost)
    refined_meta = dict(refined_meta or {})
    for rec in reversed(list(getattr(agent, "local_recent", []))):
        if isinstance(rec, dict) and rec.get("sample_id") == sample_id:
            prev_cost = rec.get("feedback_cost", None)
            try:
                prev_cost_float = float(prev_cost) if prev_cost is not None else None
            except Exception:
                prev_cost_float = None
            rec["window_provisional_cost"] = prev_cost_float
            rec["y"] = refined_y
            rec["feedback_cost"] = float(refined_cost)
            rec["feedback_source"] = "dual_refined"
            rec["refined_meta"] = refined_meta
            if prev_cost_float is not None:
                rec["refined_delta_vs_window"] = float(refined_cost) - float(prev_cost_float)
                rec["refined_ratio_vs_window"] = float(refined_cost) / max(1e-12, abs(float(prev_cost_float)))
            else:
                rec["refined_delta_vs_window"] = None
                rec["refined_ratio_vs_window"] = None
            try:
                agent._dual_last_replace_meta = {
                    "sample_id": sample_id,
                    "window_provisional_cost": prev_cost_float,
                    "refined_cost": float(refined_cost),
                    "refined_delta_vs_window": rec.get("refined_delta_vs_window"),
                    "refined_ratio_vs_window": rec.get("refined_ratio_vs_window"),
                }
            except Exception:
                pass
            # Keep incumbent consistent after replacement.
            try:
                best = None
                for rr in getattr(agent, "local_recent", []):
                    if isinstance(rr, dict):
                        if best is None or float(rr.get("y", -1e99)) > float(best.get("y", -1e99)):
                            best = rr
                if best is not None:
                    agent.prev_best_value = float(best.get("y"))
                    agent.prev_best = list(best.get("theta"))
            except Exception:
                pass
            return True
    return False


def _dual_update_class_attrs(cohort, task, delay):
    """Add per-task-type delay/lateness/violation sums to cohort without changing dataclass."""
    ttype = str(getattr(task, "task_type", "Batch"))
    prefix = "rt" if ttype == "RT" else ("ai" if ttype == "AI" else "batch")
    setattr(cohort, f"{prefix}_delay_sum", float(getattr(cohort, f"{prefix}_delay_sum", 0.0)) + float(delay))
    late = max(0.0, float(getattr(task, "finish_time", 0.0)) - float(getattr(task, "deadline", 0.0)))
    setattr(cohort, f"{prefix}_lateness_sum", float(getattr(cohort, f"{prefix}_lateness_sum", 0.0)) + late)
    if float(getattr(task, "finish_time", 0.0)) > float(getattr(task, "deadline", 0.0)):
        setattr(cohort, f"{prefix}_vio_count", int(getattr(cohort, f"{prefix}_vio_count", 0)) + 1)


def _dual_on_task_finished_cohort(self, task, delay):
    cid = getattr(task, "cohort_id", None)
    if cid is not None and cid in getattr(self, "cohorts", {}):
        cohort = self.cohorts.get(cid)
        if cohort is not None and not getattr(cohort, "finalized", False):
            _dual_update_class_attrs(cohort, task, delay)
    return _DUAL_ORIG_ON_TASK_FINISHED_COHORT(self, task, delay)


ConnectedFactory._on_task_finished_cohort = _dual_on_task_finished_cohort


def _dual_ref_probs_from_cfg():
    probs = getattr(CFG, "DUAL_FEEDBACK_REF_PROBS", None)
    if probs is None:
        probs = getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3})
    probs = _normalize_task_probs(probs)
    return {"RT": float(probs.get("RT", 0.0)), "Batch": float(probs.get("Batch", 0.0)), "AI": float(probs.get("AI", 0.0))}


def _dual_class_refined_cost(cohort, row, aggregation=None):
    """Class-normalized refined score.

    The BO/CBO agent still needs one scalar feedback value, but this scalar can be
    composed from separated RT/Batch/AI diagnostics. This function therefore always
    exports per-class details and lets aggregation choose how to combine them.

    aggregation options:
    - class / class_weighted: use configured/nominal task probabilities as weights.
    - class_equal: give RT/Batch/AI equal weights.
    - class_actual: use this cohort's realized arrival ratios; useful for diagnosis, not
      the default, because it reintroduces task-mix randomness into the target.
    - class_worst: use the worst class cost; conservative fairness/safety feedback.
    """
    source = str(aggregation or os.environ.get("DUAL_REFINED_SOURCE", getattr(CFG, "DUAL_REFINED_SOURCE", "class"))).strip().lower()
    unfinished_weight = float(getattr(CFG, "DUAL_CLASS_UNFINISHED_WEIGHT", getattr(CFG, "COHORT_UNFINISHED_PENALTY", 1000.0)))
    include_energy = bool(int(os.environ.get("DUAL_INCLUDE_ENERGY", "1")))
    metric_mode = str(os.environ.get("DUAL_CLASS_METRIC", getattr(CFG, "DUAL_CLASS_METRIC", "completed"))).strip().lower()
    if metric_mode not in {"completed", "effective", "censored"}:
        metric_mode = "completed"

    ref_probs = _dual_ref_probs_from_cfg()
    total_arrivals = max(1, int(getattr(cohort, "rt_arrivals", 0)) + int(getattr(cohort, "batch_arrivals", 0)) + int(getattr(cohort, "ai_arrivals", 0)))
    actual_probs = {
        "RT": int(getattr(cohort, "rt_arrivals", 0)) / total_arrivals,
        "Batch": int(getattr(cohort, "batch_arrivals", 0)) / total_arrivals,
        "AI": int(getattr(cohort, "ai_arrivals", 0)) / total_arrivals,
    }
    equal_probs = {"RT": 1.0 / 3.0, "Batch": 1.0 / 3.0, "AI": 1.0 / 3.0}

    if source in {"class_equal", "equal", "separate_equal"}:
        weights = equal_probs
        aggregation_name = "class_equal"
    elif source in {"class_actual", "actual", "arrival_weighted"}:
        weights = actual_probs
        aggregation_name = "class_actual"
    elif source in {"class_worst", "worst", "max"}:
        weights = None
        aggregation_name = "class_worst"
    else:
        weights = ref_probs
        aggregation_name = "class_weighted"

    class_costs = {}
    class_total = 0.0
    detail = {
        "class_aggregation": aggregation_name,
        "ref_probs": ref_probs,
        "actual_probs": actual_probs,
        "equal_probs": equal_probs,
    }

    for name, prefix in [("RT", "rt"), ("Batch", "batch"), ("AI", "ai")]:
        arrivals = int(getattr(cohort, f"{prefix}_arrivals", 0))
        completed = int(getattr(cohort, f"{prefix}_completed", 0))
        delay_sum = float(getattr(cohort, f"{prefix}_delay_sum", 0.0))
        late_sum = float(getattr(cohort, f"{prefix}_lateness_sum", 0.0))
        vio_count = int(getattr(cohort, f"{prefix}_vio_count", 0))

        if arrivals <= 0:
            avg_delay_completed = 0.0
            avg_late_completed = 0.0
            vio_rate_completed = 0.0
            completion_ratio = None
            avg_delay = 0.0
            avg_late = 0.0
            vio_rate = 0.0
            class_cost = None
        else:
            avg_delay_completed = delay_sum / max(1, completed) if completed > 0 else 0.0
            avg_late_completed = late_sum / max(1, completed) if completed > 0 else 0.0
            vio_rate_completed = vio_count / max(1, completed) if completed > 0 else 0.0
            completion_ratio = completed / max(1, arrivals)

            if metric_mode in {"effective", "censored"}:
                avg_delay = float(row.get(f"{prefix}_censored_avg_delay", avg_delay_completed))
                avg_late = float(row.get(f"{prefix}_effective_avg_lateness", avg_late_completed))
                vio_rate = float(row.get(f"{prefix}_effective_vio_rate", vio_rate_completed))
            else:
                avg_delay = avg_delay_completed
                avg_late = avg_late_completed
                vio_rate = vio_rate_completed

            class_cost = (
                float(CFG.ALPHA_LATENCY) * avg_delay
                + float(CFG.SLA_PENALTY_WEIGHT) * vio_rate
                + float(CFG.LATE_PENALTY_WEIGHT) * avg_late
                + unfinished_weight * (1.0 - completion_ratio)
            )
            class_costs[name] = float(class_cost)
            if weights is not None:
                class_total += float(weights.get(name, 0.0)) * float(class_cost)

        detail[f"{prefix}_class_cost"] = None if class_cost is None else float(class_cost)
        detail[f"{prefix}_avg_delay_completed"] = float(avg_delay_completed)
        detail[f"{prefix}_avg_lateness_completed"] = float(avg_late_completed)
        detail[f"{prefix}_vio_rate_completed"] = float(vio_rate_completed)
        detail[f"{prefix}_avg_delay_used"] = float(avg_delay)
        detail[f"{prefix}_avg_lateness_used"] = float(avg_late)
        detail[f"{prefix}_vio_rate_used"] = float(vio_rate)
        detail[f"{prefix}_metric_mode"] = metric_mode
        detail[f"{prefix}_completion_ratio"] = None if completion_ratio is None else float(completion_ratio)
        detail[f"{prefix}_weight_used"] = None if weights is None else float(weights.get(name, 0.0))

    if aggregation_name == "class_worst":
        class_total = max(class_costs.values()) if class_costs else 0.0

    energy_term = float(row.get("avg_energy_est", 0.0)) if include_energy else 0.0
    pending_term = float(getattr(CFG, "COHORT_PENDING_AREA_WEIGHT", 5.0)) * float(row.get("pending_area_per_task", 0.0))
    refined = energy_term + float(class_total) + pending_term
    detail["metric_mode"] = metric_mode
    detail["energy_term"] = float(energy_term)
    detail["pending_term"] = float(pending_term)
    detail["class_weighted_term"] = float(class_total)
    return float(refined), detail


def _dual_choose_refined_cost(cohort, row):
    source = str(os.environ.get("DUAL_REFINED_SOURCE", getattr(CFG, "DUAL_REFINED_SOURCE", "class"))).strip().lower()
    if source in {"cohort", "cohort_cost", "raw_cohort"}:
        return float(row["cohort_cost"]), {"refined_source": "cohort_cost", "class_aggregation": "none"}
    cost, detail = _dual_class_refined_cost(cohort, row, aggregation=source)
    detail["refined_source"] = "class_cost"
    return float(cost), detail


def _dual_finalize_ready_cohorts(self, now, force=False, reason="all_completed"):
    """Dual-aware cohort finalization.

    Normal cohort_complete mode keeps old behavior: cohort directly tells BO.
    Dual mode: finalized cohort replaces its matching provisional window sample when possible.
    """
    if not self._use_cohort_feedback():
        return []
    finalized_rows = []
    dual = _dual_is_enabled()
    pending = getattr(self, "dual_pending_refinements", None)
    if pending is None:
        self.dual_pending_refinements = {}
        pending = self.dual_pending_refinements
    for cohort in list(self.cohorts.values()):
        if cohort.finalized or cohort.total_tasks <= 0:
            continue
        ready = (cohort.completed_tasks >= cohort.total_tasks)
        if not (ready or force):
            continue
        final_reason = reason if ready else str(reason or "forced")
        row = self._cohort_metrics(cohort, now, reason=final_reason)
        refined_cost, refined_meta = _dual_choose_refined_cost(cohort, row) if dual else (float(row["cohort_cost"]), {"refined_source": "cohort_cost"})
        row["dual_refined_cost"] = float(refined_cost)
        row["dual_refined_source"] = str(refined_meta.get("refined_source", "cohort_cost"))
        for kk, vv in refined_meta.items():
            if kk != "ref_probs":
                row[f"dual_{kk}"] = vv
        row["dual_ref_probs"] = refined_meta.get("ref_probs")
        cohort.finalized = True
        cohort.finalize_time = float(now)
        cohort.finalize_reason = final_reason
        cohort.feedback_cost = float(refined_cost if dual else row["cohort_cost"])
        cohort.confidence = float(row["confidence"])
        self.cohort_finalized_total += 1
        self.cohort_feedback_rows.append(row)
        finalized_rows.append(row)

        if self.scheduler_type == "Boltzmann" and self.agent is not None:
            if dual:
                sid = getattr(cohort, "sample_id", None)
                if sid is None:
                    sid = f"w{int(cohort.window_index)}_c{int(cohort.cohort_id)}"
                    cohort.sample_id = sid
                ok = _dual_agent_replace_sample(self.agent, sid, refined_cost, refined_meta={**row, "sample_id": sid})
                replace_meta = getattr(self.agent, "_dual_last_replace_meta", {}) if self.agent is not None else {}
                row["sample_id"] = sid
                row["dual_replace_success"] = bool(ok)
                row["dual_window_provisional_cost"] = replace_meta.get("window_provisional_cost") if ok else None
                row["dual_refined_delta_vs_window"] = replace_meta.get("refined_delta_vs_window") if ok else None
                row["dual_refined_ratio_vs_window"] = replace_meta.get("refined_ratio_vs_window") if ok else None
                if not ok:
                    pending[sid] = {"cost": float(refined_cost), "meta": {**row, "sample_id": sid}}
                self.scheduler.update_beta(refined_cost)
            else:
                state_arg = cohort.state if getattr(self.agent, "use_state_partition", False) else None
                context_arg = cohort.context if getattr(self.agent, "use_context", False) else None
                self.agent.tell(cohort.theta_control, row["cohort_cost"], state=state_arg, context=context_arg)
                self.scheduler.update_beta(row["cohort_cost"])
    return finalized_rows


ConnectedFactory._finalize_ready_cohorts = _dual_finalize_ready_cohorts


def _dual_apply_pending_refinements(fac, agent):
    pending = getattr(fac, "dual_pending_refinements", {}) or {}
    if not pending:
        return 0
    applied = 0
    for sid in list(pending.keys()):
        item = pending.get(sid) or {}
        if _dual_agent_replace_sample(agent, sid, item.get("cost"), refined_meta=item.get("meta", {})):
            applied += 1
            pending.pop(sid, None)
    return applied


def _safebo_dedup_clip(candidates, low, high, dim):
    out = []
    seen = set()
    for c in candidates:
        if c is None:
            continue
        cc = list(c)
        if len(cc) < dim:
            cc = cc + [cc[-1] if cc else 1.0] * (dim - len(cc))
        cc = [float(min(max(cc[d], low[d]), high[d])) for d in range(dim)]
        key = tuple(round(v, 6) for v in cc)
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
    return out


def _safebo_dedup_clip_with_sources(candidates, sources, low, high, dim, limit=None):
    out = []
    out_sources = []
    seen = set()
    for idx, c in enumerate(candidates):
        if c is None:
            continue
        cc = list(c)
        if len(cc) < dim:
            cc = cc + [cc[-1] if cc else 1.0] * (dim - len(cc))
        cc = [float(min(max(cc[d], low[d]), high[d])) for d in range(dim)]
        key = tuple(round(v, 6) for v in cc)
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
        out_sources.append(str(sources[idx] if idx < len(sources) else "unknown"))
        if limit is not None and len(out) >= int(limit):
            break
    return out, out_sources


def _safebo_candidate_pool(agent, state=None, context=None, n_candidates=None):
    n_candidates = int(n_candidates if n_candidates is not None else _safebo_env("SAFEBO_CANDIDATES", 160, int))
    radius = float(_safebo_env("SAFEBO_TR_RADIUS", 0.12, float))
    dim = int(agent.dim)
    low = agent.bounds[0].tolist()
    high = agent.bounds[1].tolist()
    candidates = []
    sources = []

    for p in getattr(agent, "anchor_points", []) or []:
        try:
            candidates.append(agent._normalize_theta(p))
        except Exception:
            candidates.append(list(p))
        sources.append("anchor_point")

    if getattr(agent, "prev_best", None) is not None:
        candidates.append(list(agent.prev_best))
        sources.append("prev_best")
        center = list(agent.prev_best)
        for _ in range(max(8, n_candidates // 4)):
            cand = []
            for d in range(dim):
                span = (high[d] - low[d]) * radius
                cand.append(center[d] + span * (2.0 * agent.py_rng.random() - 1.0))
            candidates.append(cand)
            sources.append("prev_best_tr")

    try:
        if getattr(agent, "use_context", False):
            _, _, records = agent._training_data(state=state)
            pivot = agent._select_pivot_theta(context, records)
            if pivot is not None:
                candidates.append(list(pivot))
                sources.append("context_pivot")
                for _ in range(max(8, n_candidates // 4)):
                    cand = []
                    for d in range(dim):
                        span = (high[d] - low[d]) * radius
                        cand.append(pivot[d] + span * (2.0 * agent.py_rng.random() - 1.0))
                    candidates.append(cand)
                    sources.append("context_pivot_tr")
            tr_mode = str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
            if tr_mode != "off":
                anchor, anchor_debug = _cbo_resolve_actual_tr_anchor(agent, context, records)
                anchor_mode = str(anchor_debug.get("effective_tr_anchor_mode", getattr(agent, "cbo_tr_anchor_mode", "posterior_mean")) or "posterior_mean")
                anchor_source = str(anchor_debug.get("actual_tr_anchor_source", "no_anchor"))
                if anchor is not None:
                    candidates.append(list(anchor))
                    sources.append("actual_tr_anchor")
                    tr_added = max(8, n_candidates // 3)
                    for _ in range(tr_added):
                        cand = []
                        for d in range(dim):
                            span = (high[d] - low[d]) * float(getattr(agent, "trust_radius", radius))
                            cand.append(anchor[d] + span * (2.0 * agent.py_rng.random() - 1.0))
                        candidates.append(cand)
                        sources.append("trust_region")
                    debug = dict(getattr(agent, "last_debug_info", {}) or {})
                    debug.update({
                        "cbo_tr_mode": tr_mode,
                        "cbo_tr_anchor_mode": anchor_mode,
                        "cbo_tr_radius": float(getattr(agent, "trust_radius", radius)),
                        "cbo_tr_anchor_theta": list(anchor),
                        "cbo_tr_anchor_source": str(anchor_source),
                        "cbo_tr_candidate_count": int(tr_added),
                        "cbo_global_candidate_count": int(n_candidates),
                        "cbo_tr_update_reason": getattr(agent, "cbo_tr_update_reason", "safe_candidate_pool"),
                        "cbo_tr_success_count": int(getattr(agent, "cbo_tr_success_count", 0)),
                        "cbo_tr_failure_count": int(getattr(agent, "cbo_tr_failure_count", 0)),
                    })
                    debug.update(anchor_debug)
                    agent.last_debug_info = debug
                else:
                    debug = dict(getattr(agent, "last_debug_info", {}) or {})
                    debug.update({
                        "cbo_tr_mode": tr_mode,
                        "cbo_tr_anchor_mode": anchor_mode,
                        "cbo_tr_radius": float(getattr(agent, "trust_radius", radius)),
                        "cbo_tr_anchor_theta": None,
                        "cbo_tr_anchor_source": str(anchor_source),
                        "cbo_tr_candidate_count": 0,
                        "cbo_global_candidate_count": int(n_candidates),
                    })
                    debug.update(anchor_debug)
                    agent.last_debug_info = debug
    except Exception:
        pass

    while len(candidates) < n_candidates:
        candidates.append(agent._sample_in_bounds(low, high))
        sources.append("global_random")

    candidates, sources = _safebo_dedup_clip_with_sources(candidates, sources, low, high, dim, limit=max(1, n_candidates))
    try:
        agent._last_candidate_sources = list(sources)
    except Exception:
        pass
    return candidates




def _cbo_numpy_argmax_safe(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0
    return int(np.nanargmax(arr))


def _cbo_force_exploration_active(agent):
    try:
        return int(getattr(agent, "cbo_force_explore_countdown", 0)) > 0
    except Exception:
        return False


def _cbo_select_index_from_scores(agent, mu, sigma, score, default_reason="greedy_posterior_mean"):
    """Select a candidate index from posterior scores.

    Old behavior is preserved when cbo_select_mode=greedy and no residual/condition trigger is active.
    When a trigger is active, topK/epsilon/randomized modes can avoid always choosing rank 1.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    score = np.asarray(score, dtype=float)
    n = int(len(score))
    if n <= 0:
        return 0, "empty_candidate_fallback"
    beta_mode = str(getattr(agent, "cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")) or "fixed").strip().lower()
    # Backward compatibility: fixed beta keeps the old SAFEBO=greedy_mean
    # posterior-mean choice. Adaptive beta modes use the score directly.
    if str(default_reason) == "greedy_posterior_mean" and beta_mode == "fixed":
        greedy_idx = _cbo_numpy_argmax_safe(mu)
    else:
        greedy_idx = _cbo_numpy_argmax_safe(score)
    select_mode = str(getattr(agent, "cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")) or "greedy").strip().lower()
    triggered = _cbo_force_exploration_active(agent) or select_mode in {"topk_stochastic", "epsilon_greedy", "randomized_ucb"}
    if not triggered or select_mode == "greedy":
        try:
            agent.cbo_last_actual_beta_used = float(getattr(agent, "cbo_last_beta_eff", getattr(agent, "cbo_acq_beta", 0.0)))
        except Exception:
            pass
        return greedy_idx, default_reason

    k = max(1, min(int(getattr(agent, "cbo_topk", _cfg_cbo_int("CBO_TOPK", 5))), n))
    finite_score = np.where(np.isfinite(score), score, -1e300)
    top_idx = np.argsort(-finite_score)[:k]
    if len(top_idx) == 0:
        return greedy_idx, default_reason

    # epsilon-greedy: usually greedy, occasionally top-K random.
    if select_mode == "epsilon_greedy":
        eps = float(np.clip(getattr(agent, "cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)), 0.0, 1.0))
        if agent.py_rng.random() >= eps and not _cbo_force_exploration_active(agent):
            return greedy_idx, default_reason
        return int(top_idx[int(agent.py_rng.random() * len(top_idx)) % len(top_idx)]), "epsilon_greedy_topk"

    # randomized UCB: sample a temporary beta and rank mu + beta*sigma.
    if select_mode == "randomized_ucb":
        base_beta = max(0.0, float(getattr(agent, "cbo_last_beta_eff", getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)))))
        beta_sample = float(base_beta * (0.5 + agent.py_rng.random())) if base_beta > 0 else 0.0
        try:
            agent.cbo_last_actual_beta_used = float(beta_sample)
        except Exception:
            pass
        randomized_score = mu + beta_sample * sigma
        return _cbo_numpy_argmax_safe(randomized_score), f"randomized_ucb_beta={beta_sample:.3g}"

    # top-K stochastic: softmax over top-K acquisition scores.
    temp = max(1e-9, float(getattr(agent, "cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20))))
    vals = finite_score[top_idx]
    vals = vals - np.nanmax(vals)
    probs = np.exp(np.clip(vals / temp, -60.0, 60.0))
    if not np.isfinite(probs).all() or float(np.sum(probs)) <= 0:
        return int(top_idx[int(agent.py_rng.random() * len(top_idx)) % len(top_idx)]), "topk_stochastic_uniform"
    probs = probs / np.sum(probs)
    # Use Python RNG for reproducibility with agent stream.
    r = agent.py_rng.random()
    cum = 0.0
    for idx, p in zip(top_idx, probs):
        cum += float(p)
        if r <= cum:
            return int(idx), "topk_stochastic"
    return int(top_idx[-1]), "topk_stochastic_tail"


def _cbo_recent_costs(agent, window=None):
    records = []
    try:
        records = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
    except Exception:
        records = []
    if window is not None and int(window) > 0:
        records = records[-int(window):]
    vals = []
    for r in records:
        try:
            vals.append(-float(r.get("y")))
        except Exception:
            pass
    return [v for v in vals if np.isfinite(v)]


def _cbo_update_residual_condition_state(agent, actual_cost):
    """Update residual/condition diagnostics and possibly reset TR radius.

    This function is called after a true window feedback is observed. It compares the
    selected candidate's predicted cost against actual Eval/Train cost and can trigger
    a temporary exploration mode for the next few selections.
    """
    debug = dict(getattr(agent, "last_debug_info", {}) or {})
    tr_mode = str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    select_mode = str(getattr(agent, "cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")) or "greedy").lower()
    sigma_floor = max(1e-12, float(getattr(agent, "cbo_sigma_floor", _cfg_cbo_float("CBO_SIGMA_FLOOR", 1e-6))))
    mu = debug.get("selected_candidate_mu", debug.get("posterior_mu"))
    sigma = debug.get("selected_candidate_sigma", debug.get("posterior_sigma"))
    predicted_cost = np.nan
    surprise = np.nan
    raw_error = np.nan
    try:
        if mu is not None and np.isfinite(float(mu)):
            predicted_cost = -float(mu)
            raw_error = float(actual_cost) - predicted_cost
            sig = sigma_floor if sigma is None or not np.isfinite(float(sigma)) else max(float(sigma), sigma_floor)
            surprise = raw_error / sig
    except Exception:
        pass

    hist = list(getattr(agent, "cbo_surprise_history", []))
    hist.append({
        "actual_cost": float(actual_cost),
        "predicted_cost": float(predicted_cost) if np.isfinite(predicted_cost) else np.nan,
        "prediction_error": float(raw_error) if np.isfinite(raw_error) else np.nan,
        "surprise": float(surprise) if np.isfinite(surprise) else np.nan,
        "radius": float(getattr(agent, "trust_radius", np.nan)),
    })
    max_hist = max(20, int(getattr(agent, "cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10))) * 5)
    agent.cbo_surprise_history = hist[-max_hist:]

    # radius-min stuck counter
    r = float(getattr(agent, "trust_radius", np.nan))
    r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
    if np.isfinite(r) and r <= r_min * 1.001:
        agent.cbo_radius_min_stuck_count = int(getattr(agent, "cbo_radius_min_stuck_count", 0)) + 1
    else:
        agent.cbo_radius_min_stuck_count = 0

    # Recent rebound relative to recent best.
    rebound_window = max(2, int(getattr(agent, "cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20))))
    recent_costs = _cbo_recent_costs(agent, window=rebound_window)
    recent_best = min(recent_costs) if recent_costs else np.nan
    cost_gap_pct = 0.0
    if np.isfinite(recent_best) and abs(recent_best) > 1e-9:
        cost_gap_pct = (float(actual_cost) - float(recent_best)) / abs(float(recent_best))

    z_thr = float(getattr(agent, "cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)))
    gap_thr = float(getattr(agent, "cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)))
    rebound_thr = float(getattr(agent, "cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)))
    stuck_thr = int(getattr(agent, "cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)))
    residual_trigger = bool(np.isfinite(surprise) and surprise >= z_thr and cost_gap_pct >= gap_thr)
    condition_trigger = bool(cost_gap_pct >= rebound_thr or int(getattr(agent, "cbo_radius_min_stuck_count", 0)) >= stuck_thr)
    trigger = (tr_mode == "residual_adaptive" and residual_trigger) or (tr_mode == "condition_adaptive" and condition_trigger)

    if trigger:
        reset = float(getattr(agent, "cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)))
        r_max = float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
        r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
        agent.trust_radius = float(np.clip(max(reset, r), r_min, r_max))
        agent.cbo_force_explore_countdown = max(int(getattr(agent, "cbo_force_explore_countdown", 0)), int(getattr(agent, "cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5))))
        anchor_switch = str(getattr(agent, "cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best")) or "off").lower()
        if anchor_switch != "off":
            agent.cbo_runtime_anchor_override = anchor_switch
        agent.cbo_tr_update_reason = ("residual_surprise_reset" if residual_trigger else "condition_rebound_or_radius_stuck_reset")
    else:
        # Decrease countdown after each observed feedback.
        if int(getattr(agent, "cbo_force_explore_countdown", 0)) > 0:
            agent.cbo_force_explore_countdown = max(0, int(getattr(agent, "cbo_force_explore_countdown", 0)) - 1)
        if int(getattr(agent, "cbo_force_explore_countdown", 0)) <= 0:
            agent.cbo_runtime_anchor_override = None

    debug.update({
        "cbo_select_mode": select_mode,
        "cbo_topk": int(getattr(agent, "cbo_topk", _cfg_cbo_int("CBO_TOPK", 5))),
        "cbo_select_temperature": float(getattr(agent, "cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20))),
        "cbo_epsilon": float(getattr(agent, "cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10))),
        "cbo_acq_beta": float(getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0))),
        "predicted_cost": float(predicted_cost) if np.isfinite(predicted_cost) else np.nan,
        "actual_cost": float(actual_cost),
        "prediction_error": float(raw_error) if np.isfinite(raw_error) else np.nan,
        "surprise": float(surprise) if np.isfinite(surprise) else np.nan,
        "cost_gap_pct": float(cost_gap_pct),
        "residual_trigger": int(residual_trigger),
        "condition_trigger": int(condition_trigger),
        "radius_min_stuck_count": int(getattr(agent, "cbo_radius_min_stuck_count", 0)),
        "force_explore_countdown": int(getattr(agent, "cbo_force_explore_countdown", 0)),
        "runtime_anchor_override": getattr(agent, "cbo_runtime_anchor_override", None),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
    })
    agent.last_debug_info = debug
    return debug

def _safebo_posterior_mean_theta(agent, state=None, context=None):
    if getattr(agent, "anchor_points", None) and agent.step_count < len(agent.anchor_points):
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "anchor_or_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    try:
        _, _, records = agent._training_data(state=state)
    except Exception:
        records = []

    if len(records) < 2:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "cold_start_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    model_pack = agent.fit_local_gp(state=state)
    if model_pack is None:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "fit_failed_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    candidates = _safebo_candidate_pool(agent, state=state, context=context)
    if not candidates:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "empty_pool_original_ask", "mu": None, "sigma": None, "candidate_count": 1}
    sources = list(getattr(agent, "_last_candidate_sources", []) or [])
    if len(sources) < len(candidates):
        sources += ["posterior_mean_candidate"] * (len(candidates) - len(sources))
    sources = sources[:len(candidates)]

    try:
        gp = model_pack["gp"]
        y_mean = float(model_pack["y_mean"].detach().view(-1)[0].item())
        y_std = float(model_pack["y_std"].detach().view(-1)[0].item())
        bounds_full = model_pack["bounds"]
        xs = torch.tensor([agent._compose_features(t, context=context) for t in candidates], dtype=torch.double)
        xs_norm = torch.clamp(normalize(xs, bounds_full), 0.0, 1.0)
        with torch.no_grad():
            posterior = gp.posterior(xs_norm)
            mu_std = posterior.mean.detach().view(-1)
            var_std = posterior.variance.detach().view(-1)
            mu = mu_std * y_std + y_mean
            sigma = torch.sqrt(torch.clamp(var_std * (y_std ** 2), min=0.0))
            beta_info = _cbo_beta_eff_info(agent)
            score = mu + float(beta_info.get("beta_eff", getattr(agent, "cbo_acq_beta", getattr(agent, "beta_init", 3.0)))) * sigma
            score_np, service_penalty, guard_info = _cbo_service_guard_apply(agent, score.detach().cpu().numpy())
            beta_info.update(guard_info)
            try:
                agent.cbo_last_beta_info = dict(beta_info)
            except Exception:
                pass
            best_idx, select_reason = _cbo_select_index_from_scores(
                agent,
                mu.detach().cpu().numpy(),
                sigma.detach().cpu().numpy(),
                score_np,
                default_reason="greedy_posterior_mean",
            )
        theta = list(candidates[best_idx])
        agent.step_count += 1
        agent.last_theta = list(theta)
        try:
            agent.acq_history.append({
                "step": int(agent.step_count),
                "candidates": [list(c) for c in candidates],
                "acq_values": [float(v) for v in list(score_np)],
                "best_selected": list(theta),
                "model_state_dict": gp.state_dict(),
                "selection_policy": str(select_reason),
            })
        except Exception:
            pass
        if 0 <= best_idx < len(sources) and str(select_reason) != "greedy_posterior_mean":
            sources[best_idx] = str(select_reason)
        recent_records = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
        recent_best = list(max(recent_records, key=lambda r: float(r.get("y", -1e300))).get("theta", [])) if recent_records else None
        robust_theta = None
        try:
            robust_theta, _ = agent._compute_robust_incumbent(context=context)
        except Exception:
            robust_theta = None
        cand_rows, cand_summary = _cbo_candidate_rows(
            agent, candidates, sources,
            mu.detach().cpu().numpy(), sigma.detach().cpu().numpy(), score_np,
            best_idx, selected_reason=str(select_reason),
            deploy_policy="greedy_mean", deploy_source=str(select_reason),
            anchor=(getattr(agent, "cbo_last_actual_anchor_debug", {}) or {}).get("actual_tr_anchor_theta", getattr(agent, "prev_best", None)),
            robust_theta=robust_theta, recent_best=recent_best,
            beta_eff=beta_info.get("beta_eff"), service_penalty=service_penalty,
        )
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        debug.update(cand_summary)
        debug.update(beta_info)
        debug.update({
            "selected_candidate_source": str(sources[best_idx] if 0 <= best_idx < len(sources) else "posterior_mean_candidate"),
            "selected_candidate_mu": float(mu[best_idx].item()),
            "selected_candidate_sigma": float(sigma[best_idx].item()),
            "selected_candidate_acq": float(score_np[best_idx]),
            "selected_candidate_score": float(score_np[best_idx]),
            "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
            "actual_beta_used": float(getattr(agent, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
            "cbo_select_mode": str(getattr(agent, "cbo_select_mode", "greedy")),
            "selected_reason": str(select_reason),
            "candidate_diagnostic_rows": cand_rows,
        })
        agent.last_debug_info = debug
        return theta, {
            "selection": str(select_reason),
            "mu": float(mu[best_idx].item()),
            "sigma": float(sigma[best_idx].item()),
            "candidate_count": int(len(candidates)),
            **cand_summary,
            "selected_candidate_source": str(sources[best_idx] if 0 <= best_idx < len(sources) else "posterior_mean_candidate"),
            "selected_candidate_mu": float(mu[best_idx].item()),
            "selected_candidate_sigma": float(sigma[best_idx].item()),
            "selected_candidate_acq": float(score_np[best_idx]),
            "selected_candidate_score": float(score_np[best_idx]),
            "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
            "actual_beta_used": float(getattr(agent, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
            "cbo_select_mode": str(getattr(agent, "cbo_select_mode", "greedy")),
            "selected_reason": str(select_reason),
            "candidate_diagnostic_rows": cand_rows,
            **beta_info,
        }
    except Exception as e:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "posterior_failed_original_ask_" + type(e).__name__, "mu": None, "sigma": None, "candidate_count": 1}


def _safebo_select_theta(agent, state=None, context=None, group_cfg=None):
    policy = _safebo_policy_name(group_cfg)
    warmup = int(_safebo_env("SAFEBO_WARMUP", 10, int))
    explore_prob = float(_safebo_env("SAFEBO_EXPLORE_PROB", 0.20, float))
    incumbent_available = bool(getattr(agent, "prev_best", None) is not None)
    incumbent_cost = -float(agent.prev_best_value) if getattr(agent, "prev_best_value", None) is not None else None
    incumbent_iter = getattr(agent, "prev_best_iter", None)

    def pack_info(deploy_policy, deploy_source, explore_used, **extra):
        info = {
            "deploy_policy": deploy_policy,
            "deploy_source": deploy_source,
            "used_theta_source": deploy_source,
            "explore_used": int(explore_used),
            "incumbent_available": bool(incumbent_available),
            "incumbent_cost": incumbent_cost,
            "current_candidate_cost": None,
            "current_train_cost": None,
            "best_so_far_cost": incumbent_cost,
            "best_so_far_iter": incumbent_iter,
            "posterior_mu": None,
            "posterior_sigma": None,
            "candidate_count_safe": None,
        }
        info.update(extra)
        return info

    if policy in {"ei", "default", "original", "acq", "explore"}:
        theta = agent.ask(state=state, context=context)
        return theta, pack_info("ei", "acquisition_candidate", 1)

    if policy in {"greedy", "greedy_mean", "posterior_mean", "mean"}:
        theta, info = _safebo_posterior_mean_theta(agent, state=state, context=context)
        extra_info = dict(info or {})
        extra_info.update({"posterior_mu": info.get("mu"), "posterior_sigma": info.get("sigma"), "candidate_count_safe": info.get("candidate_count")})
        return theta, pack_info("greedy_mean", info.get("selection", "greedy_posterior_mean"), 0, **extra_info)

    if policy in {"incumbent", "incumbent_safe", "safe", "safe_bo"}:
        try:
            _, _, records = agent._training_data(state=state)
            n_records = len(records)
        except Exception:
            n_records = 0
        if getattr(agent, "prev_best", None) is None or n_records < warmup:
            theta = agent.ask(state=state, context=context)
            return theta, pack_info("incumbent_safe", "warmup_acquisition_candidate", 1)
        if agent.py_rng.random() < explore_prob:
            theta = agent.ask(state=state, context=context)
            return theta, pack_info("incumbent_safe", "exploration_candidate", 1)
        theta = list(agent.prev_best)
        return theta, pack_info("incumbent_safe", "incumbent_prev_best", 0)

    theta = agent.ask(state=state, context=context)
    return theta, pack_info("unknown_fallback_ei_" + policy, "fallback_acquisition_candidate", 1)


_ORIG_SAFEBO_SELECT_THETA_STABILITY = _safebo_select_theta


def _safebo_select_theta(agent, state=None, context=None, group_cfg=None):
    if agent is not None:
        try:
            agent._active_context = context
        except Exception:
            pass
    theta, info = _ORIG_SAFEBO_SELECT_THETA_STABILITY(agent, state=state, context=context, group_cfg=group_cfg)
    robust_mode = str(getattr(agent, "cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")) if agent is not None else "off").strip().lower()
    robust_theta, robust_info = (None, {"robust_incumbent_available": False, "robust_incumbent_reason": "agent_none"})
    if agent is not None:
        try:
            robust_theta, robust_info = agent._compute_robust_incumbent(context=context)
        except Exception as exc:
            robust_info = {"robust_incumbent_available": False, "robust_incumbent_reason": "robust_error:" + type(exc).__name__}
    robust_info = dict(robust_info or {})
    robust_info.setdefault("robust_incumbent_used", False)
    if robust_mode == "deploy" and robust_theta is not None and robust_info.get("robust_incumbent_available"):
        eval_count = int(robust_info.get("robust_incumbent_eval_count") or 0)
        sim = float(robust_info.get("robust_incumbent_context_similarity") or 0.0)
        threshold = float(getattr(agent, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
        posterior_mu = info.get("posterior_mu")
        predicted_current_cost = -float(posterior_mu) if posterior_mu is not None else np.nan
        robust_score = float(robust_info.get("robust_incumbent_score", np.nan))
        bo_not_clearly_better = (not np.isfinite(predicted_current_cost)) or (np.isfinite(robust_score) and predicted_current_cost >= robust_score * 0.98)
        if (eval_count >= 2 or not robust_info.get("robust_incumbent_available") is False) and sim >= threshold and bo_not_clearly_better:
            theta = list(robust_theta)
            info["deploy_source"] = "robust_incumbent"
            info["used_theta_source"] = "robust_incumbent"
            info["explore_used"] = 0
            robust_info["robust_incumbent_used"] = True
            robust_info["robust_incumbent_reason"] = "deployed_robust_score"
        else:
            robust_info["robust_incumbent_reason"] = f"not_deployed eval_count={eval_count} sim={sim:.3f} bo_not_clearly_better={bo_not_clearly_better}"
    elif robust_mode == "recommend_only" and robust_info.get("robust_incumbent_available"):
        robust_info["robust_incumbent_reason"] = "recommend_only"
    info.update(robust_info)
    if agent is not None:
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        debug.update(dict(getattr(agent, "last_history_debug", {}) or {}))
        debug.update(dict(getattr(agent, "cbo_last_history_denoise_stats", {}) or {}))
        for k in [
            "history_select_mode", "effective_history_mode", "effective_recent_window",
            "selected_recent_count", "selected_macro_count", "selected_context_count", "selected_elite_count",
            "selected_diverse_count", "selected_total_count", "selected_warm_rows_count",
            "selected_local_rows_count", "cbo_warm_start_used_rows", "context_similarity_max",
            "context_similarity_mean", "elite_best_robust_score", "elite_best_eval_count",
            "elite_best_mean_cost", "elite_best_std_cost", "cbo_tr_mode", "cbo_tr_anchor_mode",
            "cbo_tr_radius", "cbo_tr_anchor_theta", "cbo_tr_candidate_count",
            "cbo_global_candidate_count", "cbo_tr_update_reason", "cbo_tr_success_count",
            "cbo_tr_failure_count", "tr_update_mode", "tr_baseline_mean", "tr_current_mean",
            "tr_improve_pct", "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
            "cbo_tr_radius_before_update", "cbo_tr_radius_after_update",
            "cbo_macro_gate_mode", "macro_total_arrivals_norm", "macro_rt_ratio",
            "macro_batch_ratio", "macro_similarity_max", "macro_similarity_mean",
            "macro_similarity_p50", "macro_similarity_p90", "selected_macro_mean_similarity",
            "selected_macro_min_similarity", "selected_macro_max_similarity", "macro_k",
            "macro_lengthscale_total", "macro_lengthscale_rt", "macro_lengthscale_batch",
            "macro_pool_count", "macro_pool_mean_similarity", "macro_pool_min_similarity",
            "macro_pool_max_similarity", "macro_pool_p50_similarity", "macro_pool_p90_similarity",
            "selected_from_macro_pool_count", "selected_outside_macro_pool_count",
            "macro_gate_fallback_used", "macro_gate_fallback_reason",
            "context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool",
            "selected_candidate_source", "selected_candidate_mu", "selected_candidate_sigma",
            "selected_candidate_acq", "selected_candidate_score", "selected_candidate_beta_eff",
            "selected_candidate_rank_by_score", "selected_candidate_rank_by_mu",
            "selected_candidate_rank_by_sigma", "selected_candidate_rank_by_acq",
            "best_mu_candidate_source", "best_acq_candidate_source", "num_candidates",
            "num_tr_candidates", "num_global_candidates", "candidate_diagnostic_rows",
            "cbo_select_mode", "cbo_topk", "cbo_select_temperature", "cbo_epsilon", "cbo_acq_beta",
            "cbo_acq_beta_mode", "beta_eff", "radius_norm", "radius_beta_component",
            "state_beta_boost_used", "state_beta_boost_reason", "actual_score_formula",
            "actual_beta_used", "service_guard_mode", "service_guard_available",
            "service_guard_penalty", "service_guard_reason",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_theta",
            "actual_tr_anchor_reason", "anchor_override_used", "anchor_override_reason",
            "anchor_fallback_used", "anchor_fallback_reason", "anchor_theta_distance_to_prev",
            "anchor_theta_distance_to_robust_elite", "anchor_theta_distance_to_context_best",
            "anchor_theta_distance_to_recent_best", "runtime_anchor_override_reason",
            "predicted_cost", "actual_cost", "prediction_error", "surprise", "cost_gap_pct",
            "residual_trigger", "condition_trigger", "radius_min_stuck_count",
            "force_explore_countdown", "runtime_anchor_override", "cbo_tr_radius_after_update", "selected_reason",
            "cbo_history_denoise_mode", "cbo_history_denoise_k", "cbo_history_denoise_radius",
            "cbo_history_denoise_min_neighbors", "cbo_history_denoise_context_weight",
            "cbo_history_denoise_theta_weight", "cbo_history_denoise_stat",
            "cbo_history_denoise_apply_to", "cbo_history_denoise_raw_rows",
            "cbo_history_denoise_smoothed_rows", "cbo_history_denoise_unsmoothed_rows",
            "cbo_history_denoise_smoothed_ratio", "cbo_history_denoise_neighbor_count_mean",
            "cbo_history_denoise_neighbor_count_max", "cbo_history_denoise_abs_delta_mean",
            "cbo_history_denoise_abs_delta_max", "cbo_history_denoise_y_raw_mean",
            "cbo_history_denoise_y_used_mean",
            "cbo_history_outlier_filter_enabled", "cbo_history_outlier_strict_enabled",
            "cbo_history_outlier_raw_rows",
            "cbo_history_outlier_filtered_rows", "cbo_history_outlier_used_rows",
            "cbo_history_outlier_filter_ratio", "cbo_history_outlier_neighbor_count_mean",
            "cbo_history_outlier_neighbor_count_max", "cbo_history_outlier_theta_radius",
            "cbo_history_outlier_context_radius", "cbo_history_outlier_min_peers",
            "cbo_history_outlier_peer_count_mean", "cbo_history_outlier_peer_count_max",
            "cbo_history_outlier_protect_pressure", "cbo_history_outlier_pressure_quantile",
            "cbo_history_outlier_pressure_fields_available", "cbo_history_outlier_candidate_rows",
            "cbo_history_outlier_protected_rows", "cbo_history_outlier_filtered_rows_before_protection",
            "cbo_history_outlier_filtered_rows_after_protection", "cbo_history_outlier_protected_ratio",
            "cbo_history_outlier_pressure_delay_threshold", "cbo_history_outlier_pressure_backlog_threshold",
            "cbo_history_outlier_pressure_unfinished_threshold", "cbo_history_outlier_pressure_violation_threshold",
            "cbo_history_outlier_residual_mean",
            "cbo_history_outlier_residual_max", "cbo_history_outlier_threshold",
            "cbo_history_outlier_abs_threshold", "cbo_history_outlier_max_filter_ratio",
            "cbo_history_outlier_scale",
        ]:
            if k in debug:
                info[k] = debug.get(k)
        for k, v in _cbo_history_denoise_default_stats(agent, raw_rows=0).items():
            info.setdefault(k, v)
        info.setdefault("history_select_mode", getattr(agent, "cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")))
        info.setdefault("effective_history_mode", getattr(agent, "history_mode", _cfg_history_mode("all")))
        info.setdefault("effective_recent_window", getattr(agent, "recent_window", _cfg_recent_window()))
        info.setdefault("cbo_tr_mode", getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")))
        info.setdefault("cbo_tr_anchor_mode", getattr(agent, "cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")))
        info.setdefault("cbo_tr_radius", float(getattr(agent, "trust_radius", np.nan)))
        if str(getattr(agent, "cbo_macro_gate_mode", "")).strip().lower() == "hierarchical":
            for _pool_key in ["context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool"]:
                if _is_missing_value(info.get(_pool_key)):
                    info[_pool_key] = "macro_pool"
    return theta, info


# ===============================================================
# REFACTOR V1: BO training feedback selector
# ---------------------------------------------------------------
# Eval_Cost:  最终系统评价，默认仍然是 WindowSnapshot.to_metrics() 里的 metrics["cost"]。
# Train_Cost: 真正 tell 给 BO 的单标量反馈。默认等于 Eval_Cost；
#             后续如果比较反馈设计，只改这里，不再散落到 run_scenario_group 里。
# ===============================================================


# ===============================================================
# Reference-normalized metric patch: scenario baseline + tradeoff score
# Defaults preserve old behavior unless explicitly enabled from CLI.
# ===============================================================
def _cbo_metric_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _cbo_metric_clip_ratio(v, lo=None, hi=None):
    x = _cbo_metric_float(v, np.nan)
    if not np.isfinite(x):
        return np.nan
    lo = float(getattr(CFG, "CBO_NORMALIZED_RATIO_CLIP_MIN", 0.2) if lo is None else lo)
    hi = float(getattr(CFG, "CBO_NORMALIZED_RATIO_CLIP_MAX", 5.0) if hi is None else hi)
    return float(np.clip(x, lo, hi))


def _cbo_metric_reference_stat(vals, stat="median", trim_pct=0.1):
    arr = np.array([_cbo_metric_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    stat = str(stat or "median").lower()
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "trimmed_mean":
        arr.sort()
        k = int(np.floor(float(trim_pct) * arr.size))
        if k > 0 and arr.size > 2 * k:
            arr = arr[k:-k]
        return float(np.mean(arr)) if arr.size else np.nan
    return float(np.median(arr))


def _cbo_macro_context_key():
    try:
        lambdas = []
        for item in getattr(CFG, "LAMBDA_SCHEDULE", []):
            if len(item) >= 3:
                lambdas.append(float(item[2]))
        lam = float(np.median(lambdas)) if lambdas else np.nan
    except Exception:
        lam = np.nan
    try:
        probs = get_task_type_probs_at_time(0.0)
        rt = int(round(100 * float(probs.get("RT", 0.0))))
        batch = int(round(100 * float(probs.get("Batch", 0.0))))
        ai = int(round(100 * float(probs.get("AI", 0.0))))
    except Exception:
        rt, batch, ai = 0, 0, 0
    if np.isfinite(lam):
        return f"lambda_{lam:.3g}_mix_{rt}_{batch}_{ai}"
    return f"taskmix_{rt}_{batch}_{ai}"


def _cbo_load_reference_from_file():
    path = str(getattr(CFG, "CBO_REFERENCE_FILE", "") or "").strip()
    if not path:
        return None, "empty_reference_file"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = _cbo_macro_context_key()
        if isinstance(data, dict) and key in data and isinstance(data[key], dict):
            return data[key], "loaded_macro_key"
        if isinstance(data, dict):
            return data, "loaded_single_reference"
        return None, "invalid_reference_json"
    except Exception as e:
        return None, f"load_failed:{type(e).__name__}"


def _cbo_write_reference_if_needed(ref):
    out_path = str(getattr(CFG, "CBO_REFERENCE_OUTPUT_FILE", "") or "").strip()
    if not out_path or not isinstance(ref, dict):
        return
    try:
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        key = ref.get("macro_context_key", _cbo_macro_context_key())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({key: ref}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _cbo_build_reference(records):
    if not records:
        return None
    stat = str(getattr(CFG, "CBO_REFERENCE_STAT", "median"))
    trim = float(getattr(CFG, "CBO_REFERENCE_TRIM_PCT", 0.1))
    ref = {
        "delay_ref": _cbo_metric_reference_stat([r.get("avg_delay") for r in records], stat, trim),
        "energy_per_arrival_ref": _cbo_metric_reference_stat([r.get("energy_per_arrival") for r in records], stat, trim),
        "unfinished_rate_ref": _cbo_metric_reference_stat([r.get("unfinished_rate") for r in records], stat, trim),
        "success_rate_ref": _cbo_metric_reference_stat([r.get("sla_success_rate") for r in records], stat, trim),
        "eval_cost_ref": _cbo_metric_reference_stat([r.get("cost") for r in records], stat, trim),
        "arrivals_ref": _cbo_metric_reference_stat([r.get("arrivals_total") for r in records], stat, trim),
        "created_at_iter": int(len(records)),
        "calibration_rounds": int(len(records)),
        "reference_stat": stat,
        "macro_context_key": _cbo_macro_context_key(),
    }
    for k in ["delay_ref", "energy_per_arrival_ref", "eval_cost_ref"]:
        if not np.isfinite(_cbo_metric_float(ref.get(k))) or abs(_cbo_metric_float(ref.get(k))) < 1e-12:
            ref[k] = np.nan
    return ref


def _cbo_metric_reference_patch(factory, metrics):
    # Compute extra metric diagnostics and optional normalized tradeoff objective.
    # Mutates and returns metrics. Falls back to Eval_Cost if reference unavailable.
    if metrics is None:
        metrics = {}
    ref_mode = str(getattr(CFG, "CBO_REFERENCE_MODE", "off")).lower()
    objective_mode = str(getattr(CFG, "CBO_OBJECTIVE_MODE", "eval_cost")).lower()
    eps = 1e-9

    arrivals = int(_cbo_metric_float(metrics.get("arrivals_total", metrics.get("arrivals", 0)), 0))
    completed = int(_cbo_metric_float(metrics.get("completed_total", metrics.get("task_count", metrics.get("completed", 0))), 0))
    unfinished = int(_cbo_metric_float(metrics.get("unfinished_end", metrics.get("backlog", 0)), 0))
    total_energy = _cbo_metric_float(metrics.get("total_energy", np.nan))
    avg_energy = _cbo_metric_float(metrics.get("avg_energy", np.nan))

    if np.isfinite(total_energy):
        energy_per_arrival = float(total_energy) / max(arrivals, 1)
        energy_metric_source = "Total_Energy_per_arrival"
    else:
        energy_per_arrival = avg_energy
        energy_metric_source = "Avg_Energy"

    unfinished_rate = float(unfinished) / max(arrivals, 1)
    if factory is not None:
        prev_backlog = getattr(factory, "_cbo_prev_backlog_end", None)
        factory._cbo_prev_backlog_end = int(unfinished)
    else:
        prev_backlog = None
    backlog_growth = 0.0 if prev_backlog is None else max(0.0, float(unfinished) - float(prev_backlog))
    backlog_growth_rate = float(backlog_growth) / max(arrivals, 1)

    comp_rt = _cbo_metric_float(metrics.get("completion_ratio_rt", np.nan))
    comp_batch = _cbo_metric_float(metrics.get("completion_ratio_batch", np.nan))
    comp_ai = _cbo_metric_float(metrics.get("completion_ratio_ai", np.nan))
    class_vals = [x for x in [comp_rt, comp_batch, comp_ai] if np.isfinite(x)]
    class_imbalance_available = len(class_vals) == 3
    min_class_success_rate = float(np.min(class_vals)) if class_vals else np.nan
    class_imbalance_penalty = float(np.max(class_vals) - np.min(class_vals)) if len(class_vals) == 3 else np.nan

    metrics.update({
        "macro_context_key": _cbo_macro_context_key(),
        "window_arrivals_total": int(arrivals),
        "window_completed_total": int(completed),
        "window_unfinished_total": int(unfinished),
        "unfinished_rate": float(unfinished_rate),
        "backlog_growth": float(backlog_growth),
        "backlog_growth_rate": float(backlog_growth_rate),
        "energy_per_arrival": float(energy_per_arrival) if np.isfinite(energy_per_arrival) else np.nan,
        "energy_metric_source": str(energy_metric_source),
        "class_imbalance_available": bool(class_imbalance_available),
        "min_class_success_rate": min_class_success_rate,
        "class_imbalance_penalty": class_imbalance_penalty,
    })

    if factory is None:
        ref_records = []
        ref_frozen = False
        ref = None
        ref_status = "no_factory"
    else:
        if not hasattr(factory, "_cbo_reference_records"):
            factory._cbo_reference_records = []
            factory._cbo_reference_frozen = False
            factory._cbo_reference = None
            factory._cbo_reference_status = "new"
        ref_records = factory._cbo_reference_records
        ref_frozen = bool(getattr(factory, "_cbo_reference_frozen", False))
        ref = getattr(factory, "_cbo_reference", None)
        ref_status = getattr(factory, "_cbo_reference_status", "new")

    if ref_mode == "off":
        ref_status = "off"
    elif ref_mode == "load":
        if ref is None:
            ref, ref_status = _cbo_load_reference_from_file()
            if factory is not None:
                factory._cbo_reference = ref
                factory._cbo_reference_frozen = ref is not None
        ref_frozen = ref is not None
    elif ref_mode in {"calibrate", "auto_macro"}:
        if not ref_frozen and factory is not None:
            ref_records.append(dict(metrics))
            n_rec = len(ref_records)
            min_rounds = int(getattr(CFG, "CBO_REFERENCE_MIN_ROUNDS", 5))
            calib_rounds = int(getattr(CFG, "CBO_REFERENCE_CALIBRATION_ROUNDS", 30))
            if n_rec < min_rounds:
                ref_status = "calibrating_min_rounds"
            elif n_rec < calib_rounds:
                ref_status = "calibrating"
                ref = _cbo_build_reference(ref_records)
                factory._cbo_reference = ref
            else:
                ref = _cbo_build_reference(ref_records[-calib_rounds:])
                factory._cbo_reference = ref
                factory._cbo_reference_frozen = bool(getattr(CFG, "CBO_REFERENCE_FREEZE_AFTER_CALIBRATION", True))
                ref_frozen = bool(factory._cbo_reference_frozen)
                ref_status = "frozen" if ref_frozen else "calibrated_unfrozen"
                _cbo_write_reference_if_needed(ref)
        else:
            ref_status = "frozen" if ref_frozen else ref_status
    else:
        ref_status = f"unknown_reference_mode:{ref_mode}"

    ref_available = isinstance(ref, dict)
    delay_ref = _cbo_metric_float(ref.get("delay_ref") if ref_available else np.nan)
    energy_ref = _cbo_metric_float(ref.get("energy_per_arrival_ref") if ref_available else np.nan)
    unfinished_ref = _cbo_metric_float(ref.get("unfinished_rate_ref") if ref_available else np.nan)
    success_ref = _cbo_metric_float(ref.get("success_rate_ref") if ref_available else np.nan)
    eval_ref = _cbo_metric_float(ref.get("eval_cost_ref") if ref_available else np.nan)

    avg_delay = _cbo_metric_float(metrics.get("avg_delay", np.nan))
    eval_cost = _cbo_metric_float(metrics.get("cost", np.nan))
    sla_success = _cbo_metric_float(metrics.get("sla_success_rate", 1.0), 1.0)

    delay_norm = _cbo_metric_clip_ratio(avg_delay / delay_ref) if np.isfinite(delay_ref) and abs(delay_ref) > eps else np.nan
    energy_norm = _cbo_metric_clip_ratio(energy_per_arrival / energy_ref) if np.isfinite(energy_ref) and abs(energy_ref) > eps else np.nan
    eval_cost_norm = _cbo_metric_clip_ratio(eval_cost / eval_ref) if np.isfinite(eval_ref) and abs(eval_ref) > eps else np.nan
    unfinished_norm = _cbo_metric_clip_ratio(unfinished_rate / unfinished_ref) if np.isfinite(unfinished_ref) and unfinished_ref > eps else np.nan

    target_success = float(getattr(CFG, "CBO_TARGET_SUCCESS_RATE", 0.995))
    success_shortfall = max(0.0, target_success - float(sla_success)) if np.isfinite(sla_success) else np.nan
    success_shortfall_norm = success_shortfall / max(1.0 - target_success, eps) if np.isfinite(success_shortfall) else np.nan

    service_norm = np.nan
    if np.isfinite(delay_norm):
        service_norm = float(delay_norm)
        service_norm += float(getattr(CFG, "CBO_UNFINISHED_PENALTY_WEIGHT", 5.0)) * float(unfinished_rate)
        if np.isfinite(success_shortfall_norm):
            service_norm += float(getattr(CFG, "CBO_SUCCESS_SHORTFALL_WEIGHT", 2.0)) * float(success_shortfall_norm)
        service_norm += float(getattr(CFG, "CBO_BACKLOG_GROWTH_PENALTY_WEIGHT", 2.0)) * float(backlog_growth_rate)
        if bool(class_imbalance_available) and np.isfinite(class_imbalance_penalty):
            service_norm += float(getattr(CFG, "CBO_CLASS_IMBALANCE_WEIGHT", 0.0)) * float(class_imbalance_penalty)

    alpha = float(np.clip(float(getattr(CFG, "CBO_TRADEOFF_ALPHA", 0.8)),
                          float(getattr(CFG, "CBO_ALPHA_MIN", 0.6)),
                          float(getattr(CFG, "CBO_ALPHA_MAX", 0.95))))
    normalized_tradeoff_score = np.nan
    if np.isfinite(service_norm) and np.isfinite(energy_norm):
        normalized_tradeoff_score = float(alpha * service_norm + (1.0 - alpha) * energy_norm)

    metrics.update({
        "cbo_reference_mode": str(ref_mode),
        "cbo_reference_available": bool(ref_available),
        "cbo_reference_status": str(ref_status),
        "cbo_reference_round_count": int(len(ref_records)),
        "cbo_reference_frozen": bool(ref_frozen),
        "delay_ref": delay_ref,
        "energy_per_arrival_ref": energy_ref,
        "unfinished_rate_ref": unfinished_ref,
        "success_rate_ref": success_ref,
        "eval_cost_ref": eval_ref,
        "delay_norm": delay_norm,
        "energy_norm": energy_norm,
        "unfinished_norm": unfinished_norm,
        "eval_cost_norm": eval_cost_norm,
        "success_shortfall": success_shortfall,
        "success_shortfall_norm": success_shortfall_norm,
        "service_norm": service_norm,
        "normalized_tradeoff_score": normalized_tradeoff_score,
        "cbo_objective_mode": str(objective_mode),
        "tradeoff_alpha": float(alpha),
        "bo_training_cost_source": "normalized_tradeoff_score" if objective_mode == "normalized_tradeoff" and np.isfinite(normalized_tradeoff_score) else "Eval_Cost_or_feedback_score",
    })
    return metrics


def _cbo_log_reference_fields(perf_log, metrics):
    keys = [
        "macro_context_key", "window_arrivals_total", "window_completed_total", "window_unfinished_total",
        "unfinished_rate", "backlog_growth", "backlog_growth_rate", "energy_per_arrival", "energy_metric_source",
        "class_imbalance_available", "min_class_success_rate", "class_imbalance_penalty",
        "cbo_reference_mode", "cbo_reference_available", "cbo_reference_status", "cbo_reference_round_count", "cbo_reference_frozen",
        "delay_ref", "energy_per_arrival_ref", "unfinished_rate_ref", "success_rate_ref", "eval_cost_ref",
        "delay_norm", "energy_norm", "unfinished_norm", "eval_cost_norm",
        "success_shortfall", "success_shortfall_norm", "service_norm", "normalized_tradeoff_score",
        "cbo_objective_mode", "tradeoff_alpha", "bo_training_cost_source",
    ]
    for k in keys:
        perf_log.setdefault(k, []).append(metrics.get(k, None))

REFACTOR_VERSION = "bo_refactor_v5_paired_delta_crn_tr_residual_topk"


def _feedback_score_mode():
    """读取 BO 训练反馈模式。默认 window_original，等价于 v3 的 metrics['cost']。"""
    mode = getattr(CFG, "BO_TRAINING_FEEDBACK_SCORE", None)
    if mode is None:
        mode = os.environ.get("BO_TRAINING_FEEDBACK_SCORE", "window_original")
    mode = str(mode or "window_original").strip().lower()
    aliases = {
        "window": "window_original",
        "original": "window_original",
        "eval": "window_original",
        "cost": "window_original",
        "simple": "task_effective",
        "simple_effective": "task_effective",
        "effective": "task_effective",
        "effective_simple": "task_effective",
        "task": "task_effective",
        "task_effective_simple": "task_effective",
        "simple_backlog": "task_effective_backlog",
        "effective_backlog": "task_effective_backlog",
        "simple_backlog_violation": "task_effective_backlog_violation",
        "effective_backlog_violation": "task_effective_backlog_violation",
        "task_effective_backlog_violation": "task_effective_backlog_violation",
        "paired": "paired_fixed_mid_delta",
        "paired_delta": "paired_fixed_mid_delta",
        "delta": "paired_fixed_mid_delta",
        "crn": "paired_fixed_mid_delta",
        "crn_delta": "paired_fixed_mid_delta",
        "paired_fixed": "paired_fixed_mid_delta",
        "paired_mid": "paired_fixed_mid_delta",
        "paired_fixed_mid": "paired_fixed_mid_delta",
        "paired_fixed_mid_delta": "paired_fixed_mid_delta",
    }
    return aliases.get(mode, mode)


def _refactor_effective_task_cost(metrics, include_backlog=False, include_violation=False):
    """简化 BO 训练反馈：只保留连续、少重复的三类信号。

    这个函数刻意不使用 class cost、pending area、completion ratio、zero-completion
    penalty 等复杂项，避免 BO 训练目标变成“大杂烩”。

    当前可从窗口聚合指标中稳定获得的信号是：
    - avg_energy: 平均能耗
    - avg_delay: 完成任务平均延迟
    - avg_lateness: 完成任务平均超期时长

    如果 include_backlog=True，仅加入轻量 backlog 项，作为积压保护。
    """
    avg_energy = _safe_float(metrics.get("avg_energy", 0.0), 0.0)
    avg_delay = _safe_float(metrics.get("avg_delay", 0.0), 0.0)
    avg_late = _safe_float(metrics.get("avg_lateness", 0.0), 0.0)
    cost = (
        avg_energy
        + float(getattr(CFG, "ALPHA_LATENCY", 100.0)) * avg_delay
        + float(getattr(CFG, "LATE_PENALTY_WEIGHT", 300.0)) * avg_late
    )
    if include_backlog:
        cost += float(getattr(CFG, "REF_SIMPLE_BACKLOG_WEIGHT", 0.25)) * float(getattr(CFG, "BACKLOG_WEIGHT", 200.0)) * _safe_float(metrics.get("backlog", 0.0), 0.0)
    if include_violation:
        vio = _safe_float(
            metrics.get(
                "effective_violation_rate",
                metrics.get("violation_rate", metrics.get("sla_violation_rate", 0.0)),
            ),
            0.0,
        )
        cost += float(getattr(CFG, "SLA_PENALTY_WEIGHT", 1500.0)) * vio
    return float(cost)




def _paired_delta_enabled():
    """Whether scenario runs should create a shadow baseline window for CRN-style paired feedback."""
    return _feedback_score_mode() in {"paired_fixed_mid_delta", "paired_delta", "crn_delta"}


def _paired_baseline_key_for_group(group_cfg=None):
    """Pick a baseline policy with the same control family as the current group."""
    requested = str(getattr(CFG, "PAIRED_BASELINE_KEY", "") or os.environ.get("PAIRED_BASELINE_KEY", "")).strip()
    if requested:
        return USER_METHOD_ALIASES.get(requested, USER_METHOD_ALIASES.get(requested.lower(), requested))
    mode = str((group_cfg or {}).get("control_mode", "reduced6"))
    if mode == "reduced4":
        return "reduced4_fixed_mid"
    return "reduced6_fixed_mid"


def _run_paired_shadow_baseline(factory_snapshot, group_cfg, ask_state=None, ask_ctx=None, window_end=None):
    """Run a baseline policy on a deep-copied factory state for one identical window.

    Simulation-only CRN diagnostic/training option. It does not claim to be
    available in a real online system. The shadow factory starts from the same
    queues, events, node states, and workload RNG state as the deployed theta.
    """
    baseline_key = _paired_baseline_key_for_group(group_cfg)
    groups = build_scenario_method_groups()
    if baseline_key not in groups:
        raise KeyError(f"Unknown paired baseline key: {baseline_key}")
    baseline_cfg = dict(groups[baseline_key])
    if "fixed_theta" not in baseline_cfg:
        raise ValueError(f"Paired baseline must be a fixed policy, got: {baseline_key}")

    shadow = factory_snapshot
    shadow.agent = None
    shadow.disable_internal_agent_tell = True
    theta_control = list(baseline_cfg["fixed_theta"])
    theta_full = map_group_theta_to_full(theta_control, baseline_cfg)
    shadow.current_control_vector = list(theta_full)
    shadow.current_control_label = "paired_baseline:" + baseline_key
    if window_end is None:
        window_end = float(shadow.current_time) + float(getattr(CFG, "BO_INTERVAL", 40.0))
    _, _, _, _, baseline_metrics, _ = shadow.run_continuous(
        theta_full,
        eval_state=ask_state,
        eval_context=ask_ctx,
        feedback_control=theta_control,
        window_end=window_end,
    )
    return baseline_key, theta_control, baseline_metrics


def _attach_paired_delta_metrics(metrics, baseline_key, baseline_metrics):
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    baseline_cost = float(_safe_float(baseline_metrics.get("cost", np.nan), np.nan))
    delta = eval_cost - baseline_cost
    rel = 100.0 * delta / max(1e-12, abs(baseline_cost)) if np.isfinite(delta) and np.isfinite(baseline_cost) else np.nan
    fields = {
        "paired_baseline_key": str(baseline_key),
        "paired_baseline_cost": baseline_cost,
        "paired_delta_cost": float(delta),
        "paired_delta_relative_pct": float(rel),
        "paired_eval_cost": eval_cost,
        "paired_baseline_reward": float(_safe_float(baseline_metrics.get("reward", -baseline_cost), -baseline_cost)),
        "paired_baseline_avg_delay": float(_safe_float(baseline_metrics.get("avg_delay", np.nan), np.nan)),
        "paired_baseline_avg_energy": float(_safe_float(baseline_metrics.get("avg_energy", np.nan), np.nan)),
        "paired_baseline_sla_success_rate": float(_safe_float(baseline_metrics.get("sla_success_rate", np.nan), np.nan)),
        "paired_baseline_backlog": float(_safe_float(baseline_metrics.get("backlog", np.nan), np.nan)),
        "paired_note": "simulation_only_shadow_baseline_same_window_state_and_rng",
    }
    metrics.update(fields)
    return fields


def log_paired_delta_feedback(fac, metrics):
    keys = [
        "paired_baseline_key", "paired_baseline_cost", "paired_delta_cost", "paired_delta_relative_pct",
        "paired_eval_cost", "paired_baseline_reward", "paired_baseline_avg_delay",
        "paired_baseline_avg_energy", "paired_baseline_sla_success_rate", "paired_baseline_backlog",
        "paired_note",
    ]
    for k in keys:
        fac.perf_log.setdefault(k, []).append(metrics.get(k, None))

def select_bo_training_feedback_cost(metrics, fac=None, group_key=None):
    """统一选择 BO tell 使用的训练 cost。

    返回 (train_cost, feedback_mode, note)。默认返回原始 window cost，保持 v3 行为。
    """
    mode = _feedback_score_mode()
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    objective_mode = str(getattr(CFG, "CBO_OBJECTIVE_MODE", "eval_cost")).strip().lower()
    if objective_mode == "normalized_tradeoff":
        nt = float(_safe_float(metrics.get("normalized_tradeoff_score", np.nan), np.nan))
        if np.isfinite(nt):
            return nt, "normalized_tradeoff", "normalized service-energy tradeoff score"
        return eval_cost, "normalized_tradeoff_missing", "normalized_tradeoff_missing_fallback_to_eval_cost"
    if mode in {"paired_fixed_mid_delta", "paired_delta", "crn_delta"}:
        if "paired_delta_cost" in metrics and np.isfinite(_safe_float(metrics.get("paired_delta_cost"), np.nan)):
            return float(metrics["paired_delta_cost"]), "paired_fixed_mid_delta", "eval_cost_minus_shadow_fixed_mid_cost"
        return eval_cost, "paired_fixed_mid_delta_missing", "paired_delta_missing_fallback_to_eval_cost"
    if mode in {"window_original", "window_cost", "legacy_window"}:
        return eval_cost, "window_original", "same_as_eval_cost"
    if mode in {"task_effective"}:
        return _refactor_effective_task_cost(metrics, include_backlog=False), "task_effective", "energy_delay_lateness_only"
    if mode in {"task_effective_backlog"}:
        return _refactor_effective_task_cost(metrics, include_backlog=True), "task_effective_backlog", "energy_delay_lateness_light_backlog"
    if mode in {"task_effective_backlog_violation"}:
        return _refactor_effective_task_cost(metrics, include_backlog=True, include_violation=True), "task_effective_backlog_violation", "energy_delay_lateness_light_backlog_violation"
    # dual/cohort 的真正反馈路径仍由 legacy 逻辑控制；这里不抢它们的延迟替换机制。
    if mode in {"dual", "cohort", "cohort_complete", "legacy_dual", "legacy_cohort"}:
        return eval_cost, mode, "legacy_feedback_path_kept"
    return eval_cost, "window_original", f"unknown_feedback_score_fallback:{mode}"


def log_bo_training_feedback(fac, metrics, train_cost, mode, note):
    """把 BO 训练反馈显式写入 perf_log，避免和最终评价 cost 混淆。"""
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    for key, value in {
        "eval_cost": eval_cost,
        "bo_training_cost": float(train_cost),
        "bo_training_feedback_score": str(mode),
        "bo_training_feedback_note": str(note),
        "refactor_version": REFACTOR_VERSION,
    }.items():
        fac.perf_log.setdefault(key, []).append(value)


# ===============================================================
# REFACTOR V6: recent / confidence BO, CBO-lite and context ablation helpers
# ---------------------------------------------------------------
# 目标：不改变 BO 冷启动本质，而是降低 noisy window feedback 的误导。
# - recent:        只用最近窗口样本训练 GP，避免很老状态污染。
# - confidence:    对低可信窗口样本过滤/弱化，避免少完成、强偏移窗口误导 GP。
# - CBO-lite:      只使用少量窗口开始状态特征，不再把大量结果指标塞进 context。
# - CBO context ablation: full/load/util/pressure/no_cloud/no_arrival，用于判断哪些状态信息真正有用。
# ===============================================================

# v6.1: CBO-lite/context 信息扩展
# ---------------------------------------------------------------
# 原 v6 CBO-lite 只看 6 维窗口开始压力状态：arrival/backlog/util/queue/cloud-gap。
# 现在新增任务结构与上一窗口任务数，解决 RT-heavy / Batch-heavy / AI-heavy 场景中
# “压力相似但任务含义不同”的问题。
# 注意：这些特征都来自当前窗口开始前可观测信息或外部已知场景配置，不使用本窗口结束结果。
LITE_CONTEXT_FEATURE_NAMES = [
    # 原 6 维压力状态
    "arrival_rate_recent",          # 0: 最近到达率/当前估计到达强度
    "start_backlog",                # 1: 当前窗口开始积压任务数
    "start_avg_util",               # 2: 当前窗口开始平均节点利用率
    "start_max_util",               # 3: 当前窗口开始最大节点利用率
    "start_queue_total",            # 4: 当前窗口开始 ready queue 总数
    "edge_cloud_pressure_gap",       # 5: 边缘压力 - 云压力

    # 当前场景/当前时间段已知任务比例：固定 36 场景时特别有用
    "cfg_rt_prob",                  # 6
    "cfg_batch_prob",               # 7
    "cfg_ai_prob",                  # 8

    # 上一窗口真实到达任务数，按参考量归一化，反映短期任务结构偏移
    "prev_rt_arrivals_norm",        # 9
    "prev_batch_arrivals_norm",     # 10
    "prev_ai_arrivals_norm",        # 11

    # 上一窗口真实到达占比，避免只用 cfg_probs 导致 CBO 看不到随机比例偏移
    "prev_rt_arrival_ratio",        # 12
    "prev_batch_arrival_ratio",     # 13
    "prev_ai_arrival_ratio",        # 14
    "prev_unfinished_rate",         # 15
    "recent_unfinished_rate_mean",  # 16
    "unfinished_rate_trend",        # 17
]
LITE_CONTEXT_BOUNDS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    [5.0, 500.0, 1.0, 1.0, 500.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
]

# v6.1: CBO-lite context 消融配置。
# 原模式全部保留；新增 taskmix / recent_mix / pressure_taskmix / pressure_taskmix_counts 等模式。
# 推荐后续先跑：cbo-pressure, cbo-taskmix, cbo-pressure-taskmix, cbo-pressure-taskmix-counts。
LITE_CONTEXT_MODE_SPECS = {
    # 原始 full：仅 6 维压力状态，保持兼容
    "lite": {"label": "full", "indices": [0, 1, 2, 3, 4, 5]},
    "full": {"label": "full", "indices": [0, 1, 2, 3, 4, 5]},
    "load_only": {"label": "load_only", "indices": [0, 1, 4]},
    "load": {"label": "load_only", "indices": [0, 1, 4]},
    "util_only": {"label": "util_only", "indices": [2, 3]},
    "util": {"label": "util_only", "indices": [2, 3]},
    "pressure_only": {"label": "pressure_only", "indices": [1, 2, 3, 4]},
    "pressure": {"label": "pressure_only", "indices": [1, 2, 3, 4]},
    "no_cloud": {"label": "no_cloud", "indices": [0, 1, 2, 3, 4]},
    "no_arrival": {"label": "no_arrival", "indices": [1, 2, 3, 4, 5]},

    # 新增：任务结构 context
    "taskmix": {"label": "taskmix", "indices": [6, 7, 8]},
    "task_mix": {"label": "taskmix", "indices": [6, 7, 8]},
    "recent_mix": {"label": "recent_mix", "indices": [12, 13, 14]},
    "prev_mix": {"label": "recent_mix", "indices": [12, 13, 14]},
    "prev_counts": {"label": "prev_counts", "indices": [9, 10, 11]},
    "counts": {"label": "prev_counts", "indices": [9, 10, 11]},

    # 新增：压力 + 任务结构。适合验证“RT/Batch/AI比例是否能改善情景区分”。
    "pressure_taskmix": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "pressure_task_mix": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "taskmix_pressure": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "pressure_recent_mix": {"label": "pressure_recent_mix", "indices": [1, 2, 3, 4, 12, 13, 14]},
    "pressure_counts": {"label": "pressure_counts", "indices": [1, 2, 3, 4, 9, 10, 11]},
    "pressure_taskmix_counts": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},
    "pressure_task_mix_counts": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},
    "ptc": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},
    "pressure_prev_unfinished_context": {"label": "pressure_prev_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]},
    "pressure_taskmix_counts_prev_unfinished": {"label": "pressure_prev_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]},
    "prev_unfinished_context": {"label": "pressure_prev_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]},
    "puc_prev": {"label": "pressure_prev_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]},
    "pressure_unfinished_context": {"label": "pressure_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17]},
    "pressure_taskmix_counts_unfinished": {"label": "pressure_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17]},
    "unfinished_context": {"label": "pressure_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17]},
    "puc": {"label": "pressure_unfinished_context", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17]},

    # 新增：全量扩展 context。维度高，只建议小规模对照。
    "full_taskmix": {"label": "full_taskmix", "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
    "full_taskmix_counts": {"label": "full_taskmix_counts", "indices": list(range(15))},
}


def _lite_context_indices(context_mode="lite"):
    mode = str(context_mode or "lite").strip().lower()
    spec = LITE_CONTEXT_MODE_SPECS.get(mode, LITE_CONTEXT_MODE_SPECS["lite"])
    return list(spec["indices"])


def lite_context_feature_names(context_mode="lite"):
    return [LITE_CONTEXT_FEATURE_NAMES[i] for i in _lite_context_indices(context_mode)]


PRESSURE_UNFINISHED_CONTEXT_NAMES = [
    LITE_CONTEXT_FEATURE_NAMES[i]
    for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17]
]

PRESSURE_PREV_UNFINISHED_CONTEXT_NAMES = [
    LITE_CONTEXT_FEATURE_NAMES[i]
    for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15]
]


def lite_context_bounds(context_mode="lite"):
    idx = _lite_context_indices(context_mode)
    lo = [LITE_CONTEXT_BOUNDS[0][i] for i in idx]
    hi = [LITE_CONTEXT_BOUNDS[1][i] for i in idx]
    return [lo, hi]


def slice_lite_context_vector(full_vec, context_mode="lite"):
    idx = _lite_context_indices(context_mode)
    return [float(full_vec[i]) for i in idx]


def _cfg_history_mode(default="all"):
    return str(getattr(CFG, "BO_HISTORY_MODE", os.environ.get("BO_HISTORY_MODE", default)) or default).strip().lower()


def _cfg_recent_window(default=80):
    try:
        return int(getattr(CFG, "BO_RECENT_WINDOW", os.environ.get("BO_RECENT_WINDOW", default)))
    except Exception:
        return int(default)


def _cfg_confidence_min(default=0.35):
    try:
        return float(getattr(CFG, "BO_CONFIDENCE_MIN", os.environ.get("BO_CONFIDENCE_MIN", default)))
    except Exception:
        return float(default)


def _cfg_confidence_min_samples(default=12):
    try:
        return int(getattr(CFG, "BO_CONFIDENCE_MIN_SAMPLES", os.environ.get("BO_CONFIDENCE_MIN_SAMPLES", default)))
    except Exception:
        return int(default)


def _cfg_cbo_history_select_mode(default="recent"):
    return str(getattr(CFG, "CBO_HISTORY_SELECT_MODE", os.environ.get("CBO_HISTORY_SELECT_MODE", default)) or default).strip().lower()


def _cfg_cbo_int(name, default):
    try:
        return int(getattr(CFG, name, os.environ.get(name, default)))
    except Exception:
        return int(default)


def _cfg_cbo_float(name, default):
    try:
        return float(getattr(CFG, name, os.environ.get(name, default)))
    except Exception:
        return float(default)


def _cfg_cbo_str(name, default):
    return str(getattr(CFG, name, os.environ.get(name, default)) or default).strip().lower()


def _cbo_history_denoise_cfg(agent=None):
    get_attr = getattr
    return {
        "mode": str(get_attr(agent, "cbo_history_denoise_mode", get_attr(CFG, "CBO_HISTORY_DENOISE_MODE", "off")) or "off").strip().lower(),
        "k": max(1, int(get_attr(agent, "cbo_history_denoise_k", get_attr(CFG, "CBO_HISTORY_DENOISE_K", 7)))),
        "radius": max(0.0, float(get_attr(agent, "cbo_history_denoise_radius", get_attr(CFG, "CBO_HISTORY_DENOISE_RADIUS", 0.12)))),
        "min_neighbors": max(1, int(get_attr(agent, "cbo_history_denoise_min_neighbors", get_attr(CFG, "CBO_HISTORY_DENOISE_MIN_NEIGHBORS", 3)))),
        "context_weight": max(0.0, float(get_attr(agent, "cbo_history_denoise_context_weight", get_attr(CFG, "CBO_HISTORY_DENOISE_CONTEXT_WEIGHT", 1.0)))),
        "theta_weight": max(0.0, float(get_attr(agent, "cbo_history_denoise_theta_weight", get_attr(CFG, "CBO_HISTORY_DENOISE_THETA_WEIGHT", 1.0)))),
        "stat": str(get_attr(agent, "cbo_history_denoise_stat", get_attr(CFG, "CBO_HISTORY_DENOISE_STAT", "median")) or "median").strip().lower(),
        "trim_pct": float(np.clip(float(get_attr(agent, "cbo_history_denoise_trim_pct", get_attr(CFG, "CBO_HISTORY_DENOISE_TRIM_PCT", 0.1))), 0.0, 0.49)),
        "apply_to": str(get_attr(agent, "cbo_history_denoise_apply_to", get_attr(CFG, "CBO_HISTORY_DENOISE_APPLY_TO", "all")) or "all").strip().lower(),
        "outlier_threshold": max(0.0, float(get_attr(agent, "cbo_history_outlier_threshold", get_attr(CFG, "CBO_HISTORY_OUTLIER_THRESHOLD", 3.0)))),
        "outlier_abs_threshold": max(0.0, float(get_attr(agent, "cbo_history_outlier_abs_threshold", get_attr(CFG, "CBO_HISTORY_OUTLIER_ABS_THRESHOLD", 500.0)))),
        "outlier_max_filter_ratio": float(np.clip(float(get_attr(agent, "cbo_history_outlier_max_filter_ratio", get_attr(CFG, "CBO_HISTORY_OUTLIER_MAX_FILTER_RATIO", 0.2))), 0.0, 1.0)),
        "outlier_scale": str(get_attr(agent, "cbo_history_outlier_scale", get_attr(CFG, "CBO_HISTORY_OUTLIER_SCALE", "mad")) or "mad").strip().lower(),
        "outlier_theta_radius": max(0.0, float(get_attr(agent, "cbo_history_outlier_theta_radius", get_attr(CFG, "CBO_HISTORY_OUTLIER_THETA_RADIUS", 0.12)))),
        "outlier_context_radius": max(0.0, float(get_attr(agent, "cbo_history_outlier_context_radius", get_attr(CFG, "CBO_HISTORY_OUTLIER_CONTEXT_RADIUS", 0.50)))),
        "outlier_min_peers": max(1, int(get_attr(agent, "cbo_history_outlier_min_peers", get_attr(CFG, "CBO_HISTORY_OUTLIER_MIN_PEERS", 3)))),
        "outlier_use_leave_one_out": bool(get_attr(agent, "cbo_history_outlier_use_leave_one_out", get_attr(CFG, "CBO_HISTORY_OUTLIER_USE_LEAVE_ONE_OUT", True))),
        "outlier_export_filtered": bool(get_attr(agent, "cbo_history_outlier_export_filtered", get_attr(CFG, "CBO_HISTORY_OUTLIER_EXPORT_FILTERED", True))),
        "outlier_protect_pressure": bool(get_attr(agent, "cbo_history_outlier_protect_pressure", get_attr(CFG, "CBO_HISTORY_OUTLIER_PROTECT_PRESSURE", False))),
        "outlier_pressure_quantile": float(np.clip(float(get_attr(agent, "cbo_history_outlier_pressure_quantile", get_attr(CFG, "CBO_HISTORY_OUTLIER_PRESSURE_QUANTILE", 0.75))), 0.0, 1.0)),
        "outlier_protect_high_cost_only": bool(get_attr(agent, "cbo_history_outlier_protect_high_cost_only", get_attr(CFG, "CBO_HISTORY_OUTLIER_PROTECT_HIGH_COST_ONLY", True))),
        "outlier_pressure_fields": str(get_attr(agent, "cbo_history_outlier_pressure_fields", get_attr(CFG, "CBO_HISTORY_OUTLIER_PRESSURE_FIELDS", "Avg_Delay,Backlog,unfinished_end,Violation")) or "").strip(),
    }


def _cbo_history_denoise_default_stats(agent=None, raw_rows=0, y_raw=None, y_used=None):
    cfg = _cbo_history_denoise_cfg(agent)
    raw_rows = int(raw_rows or 0)

    def finite_mean(vals):
        if vals is None:
            return 0.0
        arr = np.asarray(vals.detach().cpu().numpy() if hasattr(vals, "detach") else vals, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else 0.0

    return {
        "cbo_history_denoise_mode": cfg["mode"],
        "cbo_history_denoise_k": int(cfg["k"]),
        "cbo_history_denoise_radius": float(cfg["radius"]),
        "cbo_history_denoise_min_neighbors": int(cfg["min_neighbors"]),
        "cbo_history_denoise_context_weight": float(cfg["context_weight"]),
        "cbo_history_denoise_theta_weight": float(cfg["theta_weight"]),
        "cbo_history_denoise_stat": cfg["stat"],
        "cbo_history_denoise_apply_to": cfg["apply_to"],
        "cbo_history_denoise_raw_rows": raw_rows,
        "cbo_history_denoise_smoothed_rows": 0,
        "cbo_history_denoise_unsmoothed_rows": raw_rows,
        "cbo_history_denoise_smoothed_ratio": 0.0,
        "cbo_history_denoise_neighbor_count_mean": 0.0,
        "cbo_history_denoise_neighbor_count_max": 0,
        "cbo_history_denoise_abs_delta_mean": 0.0,
        "cbo_history_denoise_abs_delta_max": 0.0,
        "cbo_history_denoise_y_raw_mean": finite_mean(y_raw),
        "cbo_history_denoise_y_used_mean": finite_mean(y_used if y_used is not None else y_raw),
        "cbo_history_outlier_filter_enabled": int(cfg["mode"] in {"local_outlier_filter", "strict_local_outlier_filter"}),
        "cbo_history_outlier_strict_enabled": int(cfg["mode"] == "strict_local_outlier_filter"),
        "cbo_history_outlier_raw_rows": raw_rows,
        "cbo_history_outlier_filtered_rows": 0,
        "cbo_history_outlier_used_rows": raw_rows,
        "cbo_history_outlier_filter_ratio": 0.0,
        "cbo_history_outlier_neighbor_count_mean": 0.0,
        "cbo_history_outlier_neighbor_count_max": 0,
        "cbo_history_outlier_theta_radius": float(cfg["outlier_theta_radius"]),
        "cbo_history_outlier_context_radius": float(cfg["outlier_context_radius"]),
        "cbo_history_outlier_min_peers": int(cfg["outlier_min_peers"]),
        "cbo_history_outlier_peer_count_mean": 0.0,
        "cbo_history_outlier_peer_count_max": 0,
        "cbo_history_outlier_protect_pressure": int(bool(cfg["outlier_protect_pressure"])),
        "cbo_history_outlier_pressure_quantile": float(cfg["outlier_pressure_quantile"]),
        "cbo_history_outlier_pressure_fields_available": "",
        "cbo_history_outlier_candidate_rows": 0,
        "cbo_history_outlier_protected_rows": 0,
        "cbo_history_outlier_filtered_rows_before_protection": 0,
        "cbo_history_outlier_filtered_rows_after_protection": 0,
        "cbo_history_outlier_protected_ratio": 0.0,
        "cbo_history_outlier_pressure_delay_threshold": np.nan,
        "cbo_history_outlier_pressure_backlog_threshold": np.nan,
        "cbo_history_outlier_pressure_unfinished_threshold": np.nan,
        "cbo_history_outlier_pressure_violation_threshold": np.nan,
        "cbo_history_outlier_residual_mean": 0.0,
        "cbo_history_outlier_residual_max": 0.0,
        "cbo_history_outlier_threshold": float(cfg["outlier_threshold"]),
        "cbo_history_outlier_abs_threshold": float(cfg["outlier_abs_threshold"]),
        "cbo_history_outlier_max_filter_ratio": float(cfg["outlier_max_filter_ratio"]),
        "cbo_history_outlier_scale": cfg["outlier_scale"] if cfg["outlier_scale"] in {"mad", "iqr", "std"} else "mad",
    }


def _cbo_history_denoise_scale(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.ones(0, dtype=float)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, std)
    return np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0).astype(float)


def _cbo_history_denoise_stat(values, stat, trim_pct):
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if str(stat) == "trimmed_mean":
        vals_sorted = np.sort(vals)
        trim_n = int(np.floor(vals_sorted.size * float(trim_pct)))
        if trim_n > 0 and vals_sorted.size - 2 * trim_n >= 2:
            return float(np.mean(vals_sorted[trim_n:-trim_n]))
    return float(np.median(vals))


def _cbo_history_record_sources(records):
    if not records:
        return None
    sources = []
    for rec in records:
        if not isinstance(rec, dict):
            return None
        sources.append("warm" if bool(rec.get("cbo_warm_start_source")) else "local")
    return sources


def _cbo_history_source_eligible(agent, records, raw_rows, apply_to):
    sources = _cbo_history_record_sources(records)
    apply_to = str(apply_to or "all").strip().lower()
    if sources is None and apply_to in {"local", "warm"}:
        warning_key = f"history_denoise_source_unknown_{apply_to}"
        warned = set(getattr(agent, "_cbo_history_denoise_warnings", set()) or set()) if agent is not None else set()
        if warning_key not in warned:
            print(f"[WARN] CBO history denoise apply-to={apply_to} requested but row source metadata is unavailable; falling back to all", flush=True)
            warned.add(warning_key)
            if agent is not None:
                agent._cbo_history_denoise_warnings = warned
        apply_to = "all"
    eligible = np.ones(int(raw_rows), dtype=bool)
    if sources is not None and apply_to == "local":
        eligible = np.asarray([s == "local" for s in sources], dtype=bool)
    elif sources is not None and apply_to == "warm":
        eligible = np.asarray([s == "warm" for s in sources], dtype=bool)
    return eligible, apply_to


def _cbo_history_feature_parts(agent, x):
    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)
    if x_np.ndim != 2:
        return x_np, 0, 0, np.zeros((0, 0), dtype=float), np.zeros((0, 0), dtype=float), np.ones(0), np.ones(0)
    dim = min(max(0, int(getattr(agent, "dim", 0))), x_np.shape[1])
    use_context = bool(getattr(agent, "use_context", False))
    context_dim = min(max(0, int(getattr(agent, "context_dim", 0))) if use_context else 0, max(0, x_np.shape[1] - dim))
    theta_x = x_np[:, :dim] if dim > 0 else np.zeros((x_np.shape[0], 0), dtype=float)
    context_x = x_np[:, dim:dim + context_dim] if context_dim > 0 else np.zeros((x_np.shape[0], 0), dtype=float)
    if dim > 0 and getattr(agent, "bounds", None) is not None:
        bounds = getattr(agent, "bounds")
        bounds_np = bounds.detach().cpu().numpy() if hasattr(bounds, "detach") else np.asarray(bounds, dtype=float)
        theta_scale = np.asarray(bounds_np[1, :dim] - bounds_np[0, :dim], dtype=float)
        theta_scale = np.where(np.isfinite(theta_scale) & (np.abs(theta_scale) > 1e-12), np.abs(theta_scale), 1.0)
    else:
        theta_scale = _cbo_history_denoise_scale(theta_x)
    context_scale = _cbo_history_denoise_scale(context_x)
    return x_np, dim, context_dim, theta_x, context_x, theta_scale, context_scale


def _cbo_history_neighbor_indices(i, theta_x, context_x, theta_scale, context_scale, cfg_vals, raw_rows):
    if theta_x.shape[1] > 0:
        d_theta_vec = (theta_x - theta_x[i]) / theta_scale
        d_theta = np.linalg.norm(d_theta_vec, axis=1)
    else:
        d_theta = np.zeros(raw_rows, dtype=float)
    if context_x.shape[1] > 0:
        d_context_vec = (context_x - context_x[i]) / context_scale
        d_context = np.linalg.norm(d_context_vec, axis=1)
    else:
        d_context = np.zeros(raw_rows, dtype=float)
    dist = float(cfg_vals["theta_weight"]) * d_theta + float(cfg_vals["context_weight"]) * d_context
    dist = np.where(np.isfinite(dist), dist, np.inf)
    idx = np.where(dist <= float(cfg_vals["radius"]))[0]
    if idx.size:
        idx = idx[np.argsort(dist[idx])[:int(cfg_vals["k"])]]
    return idx


def _cbo_history_strict_peer_indices(i, theta_x, context_x, theta_scale, context_scale, cfg_vals, raw_rows):
    if theta_x.shape[1] > 0:
        d_theta_vec = (theta_x - theta_x[i]) / theta_scale
        d_theta = np.linalg.norm(d_theta_vec, axis=1)
    else:
        d_theta = np.zeros(raw_rows, dtype=float)
    if context_x.shape[1] > 0:
        d_context_vec = (context_x - context_x[i]) / context_scale
        d_context = np.linalg.norm(d_context_vec, axis=1)
    else:
        d_context = np.zeros(raw_rows, dtype=float)
    d_theta = np.where(np.isfinite(d_theta), d_theta, np.inf)
    d_context = np.where(np.isfinite(d_context), d_context, np.inf)
    mask = (d_theta <= float(cfg_vals["outlier_theta_radius"])) & (d_context <= float(cfg_vals["outlier_context_radius"]))
    if bool(cfg_vals.get("outlier_use_leave_one_out", True)) and 0 <= int(i) < int(raw_rows):
        mask[int(i)] = False
    idx = np.where(mask)[0]
    if idx.size:
        combined = d_theta[idx] + d_context[idx]
        idx = idx[np.argsort(combined)]
    return idx, d_theta, d_context


def _cbo_history_outlier_scale(values, mode):
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 1e-9
    med = float(np.median(vals))
    mode = str(mode or "mad").strip().lower()
    if mode == "iqr":
        scale = (float(np.percentile(vals, 75)) - float(np.percentile(vals, 25))) / 1.349
    elif mode == "std":
        scale = float(np.std(vals))
    else:
        scale = float(np.median(np.abs(vals - med)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1e-9
    return float(scale)


def _cbo_history_json(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _cbo_history_rec_metric(rec, names, default=np.nan):
    metrics = rec.get("metrics") if isinstance(rec, dict) else None
    for name in names:
        try:
            if isinstance(rec, dict) and name in rec and rec.get(name) is not None:
                return float(rec.get(name))
            if isinstance(metrics, dict) and name in metrics and metrics.get(name) is not None:
                return float(metrics.get(name))
        except Exception:
            continue
    return default


def _cbo_history_pressure_field_names(fields_value):
    fields = []
    for part in str(fields_value or "").split(","):
        name = part.strip()
        if name:
            fields.append(name)
    return fields


def _cbo_history_pressure_aliases(field):
    key = str(field or "").strip()
    low = key.lower()
    if low in {"avg_delay", "delay", "avgdelay"}:
        return ["Avg_Delay", "avg_delay", "delay"]
    if low in {"backlog", "backlog_end"}:
        return ["Backlog", "backlog"]
    if low in {"unfinished_end", "unfinished", "unfinishedend"}:
        return ["unfinished_end", "Unfinished_End", "window_unfinished_total"]
    if low in {"violation", "violation_rate", "vio"}:
        return ["Violation", "Violation_Rate", "violation_rate"]
    return [key]


def _cbo_history_pressure_thresholds(records, cfg_vals):
    fields = _cbo_history_pressure_field_names(cfg_vals.get("outlier_pressure_fields"))
    quantile = float(cfg_vals.get("outlier_pressure_quantile", 0.75))
    thresholds = {}
    values_by_field = {}
    for field in fields:
        aliases = _cbo_history_pressure_aliases(field)
        vals = []
        for rec in records:
            if not isinstance(rec, dict):
                vals.append(np.nan)
                continue
            vals.append(_cbo_history_rec_metric(rec, aliases, np.nan))
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            thresholds[field] = float(np.quantile(finite, quantile))
            values_by_field[field] = arr
    return thresholds, values_by_field


def _cbo_history_pressure_threshold_stat(thresholds, aliases):
    for name in aliases:
        if name in thresholds:
            return float(thresholds[name])
    for key, value in thresholds.items():
        low = str(key).lower()
        if any(str(alias).lower() == low for alias in aliases):
            return float(value)
    return np.nan


def _cbo_history_pressure_protection_for_index(idx_i, detail, thresholds, values_by_field, cfg_vals):
    if not thresholds:
        return False, ""
    if bool(cfg_vals.get("outlier_protect_high_cost_only", True)):
        try:
            # Training y is reward (normally -cost), so higher cost than peers means y_i is lower than median_y.
            if not (float(detail.get("y_i", np.nan)) < float(detail.get("median_y", np.nan))):
                return False, ""
        except Exception:
            return False, ""
    reasons = []
    for field, threshold in thresholds.items():
        arr = values_by_field.get(field)
        if arr is None or idx_i >= len(arr):
            continue
        val = float(arr[idx_i])
        if np.isfinite(val) and np.isfinite(float(threshold)) and val >= float(threshold):
            reasons.append(f"{field}>=p{int(round(100.0 * float(cfg_vals.get('outlier_pressure_quantile', 0.75))))}")
    return bool(reasons), ";".join(reasons)


def _cbo_history_export_filtered_records(agent, selected_details, records):
    try:
        out_dir = os.path.abspath(globals().get("SCENARIO_SAVE_DIR", os.getcwd()))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "outlier_filtered_records.csv")
        columns = [
            "current_iteration", "training_row_index", "source_type", "original_iteration",
            "group_key", "history_mode", "y_i", "median_y", "residual", "scale",
            "d_theta_min", "d_theta_mean", "d_theta_max", "d_context_min",
            "d_context_mean", "d_context_max", "peer_count", "theta", "context",
            "Eval_Cost", "Avg_Delay", "Avg_Energy", "Backlog", "unfinished_end",
            "Violation", "candidate_outlier", "pressure_protected", "final_filtered",
            "protect_reason", "pressure_quantile", "pressure_delay_threshold",
            "pressure_backlog_threshold", "pressure_unfinished_threshold",
            "pressure_violation_threshold",
        ]
        if not selected_details:
            if not os.path.exists(path):
                pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
            if agent is not None:
                agent.cbo_history_outlier_filtered_records_path = path
            return
        current_iteration = int(getattr(agent, "step_count", -1))
        rows = []
        for detail in selected_details:
            idx_i = int(detail.get("index", -1))
            rec = records[idx_i] if 0 <= idx_i < len(records) and isinstance(records[idx_i], dict) else {}
            theta = rec.get("theta", detail.get("theta"))
            context = rec.get("context", detail.get("context"))
            eval_cost = _cbo_history_rec_metric(rec, ["cost", "Eval_Cost", "eval_cost"], np.nan)
            if not np.isfinite(eval_cost):
                try:
                    eval_cost = -float(detail.get("y_i", np.nan))
                except Exception:
                    eval_cost = np.nan
            rows.append({
                "current_iteration": current_iteration,
                "training_row_index": idx_i,
                "source_type": "warm" if bool(rec.get("cbo_warm_start_source")) else "local",
                "original_iteration": rec.get("bo_iter", ""),
                "group_key": rec.get("group_key", ""),
                "history_mode": rec.get("history_mode", ""),
                "y_i": detail.get("y_i", np.nan),
                "median_y": detail.get("median_y", np.nan),
                "residual": detail.get("residual", np.nan),
                "scale": detail.get("scale", np.nan),
                "d_theta_min": detail.get("d_theta_min", np.nan),
                "d_theta_mean": detail.get("d_theta_mean", np.nan),
                "d_theta_max": detail.get("d_theta_max", np.nan),
                "d_context_min": detail.get("d_context_min", np.nan),
                "d_context_mean": detail.get("d_context_mean", np.nan),
                "d_context_max": detail.get("d_context_max", np.nan),
                "peer_count": detail.get("peer_count", 0),
                "theta": _cbo_history_json(theta),
                "context": _cbo_history_json(context),
                "Eval_Cost": eval_cost,
                "Avg_Delay": _cbo_history_rec_metric(rec, ["avg_delay", "Avg_Delay"], np.nan),
                "Avg_Energy": _cbo_history_rec_metric(rec, ["avg_energy", "Avg_Energy"], np.nan),
                "Backlog": _cbo_history_rec_metric(rec, ["backlog", "Backlog"], np.nan),
                "unfinished_end": _cbo_history_rec_metric(rec, ["unfinished_end", "Unfinished_End"], np.nan),
                "Violation": _cbo_history_rec_metric(rec, ["Violation", "Violation_Rate", "violation_rate"], np.nan),
                "candidate_outlier": bool(detail.get("candidate_outlier", True)),
                "pressure_protected": bool(detail.get("pressure_protected", False)),
                "final_filtered": bool(detail.get("final_filtered", True)),
                "protect_reason": detail.get("protect_reason", ""),
                "pressure_quantile": detail.get("pressure_quantile", np.nan),
                "pressure_delay_threshold": detail.get("pressure_delay_threshold", np.nan),
                "pressure_backlog_threshold": detail.get("pressure_backlog_threshold", np.nan),
                "pressure_unfinished_threshold": detail.get("pressure_unfinished_threshold", np.nan),
                "pressure_violation_threshold": detail.get("pressure_violation_threshold", np.nan),
            })
        if rows:
            pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8-sig")
            if agent is not None:
                agent.cbo_history_outlier_filtered_records_path = path
    except Exception as exc:
        warned = set(getattr(agent, "_cbo_history_denoise_warnings", set()) or set()) if agent is not None else set()
        key = "history_outlier_export_failed"
        if key not in warned:
            print(f"[WARN] failed to export CBO outlier filtered records: {type(exc).__name__}: {exc}", flush=True)
            warned.add(key)
            if agent is not None:
                agent._cbo_history_denoise_warnings = warned


def _cbo_history_store_stats(agent, stats):
    if agent is not None:
        agent.cbo_last_history_denoise_stats = dict(stats)
        hist = dict(getattr(agent, "last_history_debug", {}) or {})
        hist.update(stats)
        agent.last_history_debug = hist


def prepare_gp_training_data(agent, x, y, metadata=None, cfg=None):
    records = list(metadata or [])
    try:
        raw_rows = int(y.shape[0])
    except Exception:
        raw_rows = len(records)
    stats = _cbo_history_denoise_default_stats(agent, raw_rows=raw_rows, y_raw=y)
    mode = stats["cbo_history_denoise_mode"]
    if mode == "off" or raw_rows <= 0:
        _cbo_history_store_stats(agent, stats)
        return x, y, records, stats
    if mode not in {"local_median", "local_outlier_filter", "strict_local_outlier_filter"}:
        stats["cbo_history_denoise_mode"] = "off"
        _cbo_history_store_stats(agent, stats)
        return x, y, records, stats

    y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y, dtype=float)
    y_flat = np.asarray(y_np, dtype=float).reshape(-1)
    x_np, dim, context_dim, theta_x, context_x, theta_scale, context_scale = _cbo_history_feature_parts(agent, x)
    if x_np.ndim != 2 or y_flat.size != x_np.shape[0]:
        _cbo_history_store_stats(agent, stats)
        return x, y, records, stats

    cfg_vals = _cbo_history_denoise_cfg(agent)
    eligible, _apply_to = _cbo_history_source_eligible(agent, records, raw_rows, cfg_vals["apply_to"])

    y_used = y_flat.copy()
    neighbor_counts = []
    applied = 0
    residuals = []
    outlier_candidates = []
    outlier_details = {}
    for i in range(raw_rows):
        if not eligible[i] or not np.isfinite(y_flat[i]):
            neighbor_counts.append(1)
            continue
        if mode == "strict_local_outlier_filter":
            idx, d_theta_all, d_context_all = _cbo_history_strict_peer_indices(i, theta_x, context_x, theta_scale, context_scale, cfg_vals, raw_rows)
            min_required = int(cfg_vals["outlier_min_peers"])
        else:
            idx = _cbo_history_neighbor_indices(i, theta_x, context_x, theta_scale, context_scale, cfg_vals, raw_rows)
            d_theta_all = None
            d_context_all = None
            min_required = int(cfg_vals["min_neighbors"])
        idx = np.asarray([j for j in idx.tolist() if np.isfinite(y_flat[j])], dtype=int)
        neighbor_counts.append(int(idx.size))
        if idx.size >= min_required:
            med = float(np.median(y_flat[idx]))
            residual = abs(float(y_flat[i]) - med)
            residuals.append(float(residual))
            if mode == "local_median":
                val = _cbo_history_denoise_stat(y_flat[idx], cfg_vals["stat"], cfg_vals["trim_pct"])
                if np.isfinite(val):
                    applied += 1
                    y_used[i] = float(val)
            else:
                scale = _cbo_history_outlier_scale(y_flat[idx], cfg_vals["outlier_scale"])
                if (
                    residual > float(cfg_vals["outlier_abs_threshold"])
                    and residual > float(cfg_vals["outlier_threshold"]) * float(scale)
                ):
                    outlier_candidates.append((float(residual), int(i)))
                    if mode == "strict_local_outlier_filter":
                        d_theta_peer = d_theta_all[idx] if d_theta_all is not None and idx.size else np.asarray([], dtype=float)
                        d_context_peer = d_context_all[idx] if d_context_all is not None and idx.size else np.asarray([], dtype=float)
                        outlier_details[int(i)] = {
                            "index": int(i),
                            "y_i": float(y_flat[i]),
                            "median_y": float(med),
                            "residual": float(residual),
                            "scale": float(scale),
                            "peer_count": int(idx.size),
                            "d_theta_min": float(np.min(d_theta_peer)) if d_theta_peer.size else np.nan,
                            "d_theta_mean": float(np.mean(d_theta_peer)) if d_theta_peer.size else np.nan,
                            "d_theta_max": float(np.max(d_theta_peer)) if d_theta_peer.size else np.nan,
                            "d_context_min": float(np.min(d_context_peer)) if d_context_peer.size else np.nan,
                            "d_context_mean": float(np.mean(d_context_peer)) if d_context_peer.size else np.nan,
                            "d_context_max": float(np.max(d_context_peer)) if d_context_peer.size else np.nan,
                            "theta": theta_x[i].tolist() if theta_x.shape[1] else [],
                            "context": context_x[i].tolist() if context_x.shape[1] else [],
                        }

    if mode in {"local_outlier_filter", "strict_local_outlier_filter"}:
        max_filter = int(np.floor(float(cfg_vals["outlier_max_filter_ratio"]) * float(raw_rows)))
        outlier_candidates.sort(key=lambda item: item[0], reverse=True)
        selected = outlier_candidates[:max(0, max_filter)]
        pressure_thresholds = {}
        pressure_values = {}
        protected_indices = set()
        protected_rows = 0
        if mode == "strict_local_outlier_filter" and bool(cfg_vals.get("outlier_protect_pressure", False)):
            pressure_thresholds, pressure_values = _cbo_history_pressure_thresholds(records, cfg_vals)
            if not pressure_thresholds:
                warning_key = "history_outlier_pressure_fields_unavailable"
                warned = set(getattr(agent, "_cbo_history_denoise_warnings", set()) or set()) if agent is not None else set()
                if warning_key not in warned:
                    print("[WARN] CBO strict pressure protection enabled but no pressure fields are available; using unprotected strict filter", flush=True)
                    warned.add(warning_key)
                    if agent is not None:
                        agent._cbo_history_denoise_warnings = warned
            else:
                for _residual, idx_i in selected:
                    detail = outlier_details.get(int(idx_i), {})
                    is_protected, reason = _cbo_history_pressure_protection_for_index(int(idx_i), detail, pressure_thresholds, pressure_values, cfg_vals)
                    if is_protected:
                        protected_indices.add(int(idx_i))
                        protected_rows += 1
                        if int(idx_i) in outlier_details:
                            outlier_details[int(idx_i)]["pressure_protected"] = True
                            outlier_details[int(idx_i)]["protect_reason"] = reason
        keep_mask = np.ones(raw_rows, dtype=bool)
        for _residual, idx_i in selected:
            if int(idx_i) not in protected_indices:
                keep_mask[idx_i] = False
        filtered_rows = int(np.sum(~keep_mask))
        min_fit_rows = max(5, int(getattr(agent, "dim", 0)) + 2)
        if raw_rows - filtered_rows < min_fit_rows:
            warning_key = "history_outlier_filter_too_few_rows"
            warned = set(getattr(agent, "_cbo_history_denoise_warnings", set()) or set()) if agent is not None else set()
            if warning_key not in warned:
                print(
                    f"[WARN] CBO {mode} would leave {raw_rows - filtered_rows} rows "
                    f"(< {min_fit_rows}); using unfiltered GP training data",
                    flush=True,
                )
                warned.add(warning_key)
                if agent is not None:
                    agent._cbo_history_denoise_warnings = warned
            keep_mask[:] = True
            filtered_rows = 0
            protected_rows = 0
            protected_indices = set()
        if hasattr(x, "detach"):
            keep_tensor = torch.as_tensor(keep_mask, dtype=torch.bool, device=x.device)
            x_out = x[keep_tensor]
            y_out = y[keep_tensor]
        else:
            x_out = np.asarray(x)[keep_mask]
            y_out = np.asarray(y)[keep_mask]
        records_out = [rec for rec, keep in zip(records, keep_mask.tolist()) if keep]
        residual_arr = np.asarray(residuals, dtype=float)
        selected_ids = {int(idx_i) for _residual, idx_i in selected}
        delay_threshold = _cbo_history_pressure_threshold_stat(pressure_thresholds, ["Avg_Delay", "avg_delay", "delay"])
        backlog_threshold = _cbo_history_pressure_threshold_stat(pressure_thresholds, ["Backlog", "backlog"])
        unfinished_threshold = _cbo_history_pressure_threshold_stat(pressure_thresholds, ["unfinished_end", "Unfinished_End", "window_unfinished_total"])
        violation_threshold = _cbo_history_pressure_threshold_stat(pressure_thresholds, ["Violation", "Violation_Rate", "violation_rate"])
        selected_details = []
        for _residual, idx_i in outlier_candidates:
            idx_i = int(idx_i)
            if idx_i not in outlier_details:
                continue
            detail = dict(outlier_details[idx_i])
            detail["candidate_outlier"] = True
            detail["pressure_protected"] = bool(idx_i in protected_indices)
            detail["final_filtered"] = bool(idx_i in selected_ids and not keep_mask[idx_i])
            if idx_i not in selected_ids and not detail.get("protect_reason"):
                detail["protect_reason"] = "max_filter_ratio_cap"
            detail.setdefault("protect_reason", "")
            detail["pressure_quantile"] = float(cfg_vals.get("outlier_pressure_quantile", np.nan))
            detail["pressure_delay_threshold"] = delay_threshold
            detail["pressure_backlog_threshold"] = backlog_threshold
            detail["pressure_unfinished_threshold"] = unfinished_threshold
            detail["pressure_violation_threshold"] = violation_threshold
            selected_details.append(detail)
        if mode == "strict_local_outlier_filter" and bool(cfg_vals.get("outlier_export_filtered", True)):
            _cbo_history_export_filtered_records(agent, selected_details, records)
        stats.update({
            "cbo_history_outlier_filter_enabled": 1,
            "cbo_history_outlier_strict_enabled": int(mode == "strict_local_outlier_filter"),
            "cbo_history_outlier_raw_rows": int(raw_rows),
            "cbo_history_outlier_filtered_rows": int(filtered_rows),
            "cbo_history_outlier_used_rows": int(raw_rows - filtered_rows),
            "cbo_history_outlier_filter_ratio": float(filtered_rows / max(1, raw_rows)),
            "cbo_history_outlier_neighbor_count_mean": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
            "cbo_history_outlier_neighbor_count_max": int(max(neighbor_counts)) if neighbor_counts else 0,
            "cbo_history_outlier_theta_radius": float(cfg_vals["outlier_theta_radius"]),
            "cbo_history_outlier_context_radius": float(cfg_vals["outlier_context_radius"]),
            "cbo_history_outlier_min_peers": int(cfg_vals["outlier_min_peers"]),
            "cbo_history_outlier_peer_count_mean": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
            "cbo_history_outlier_peer_count_max": int(max(neighbor_counts)) if neighbor_counts else 0,
            "cbo_history_outlier_protect_pressure": int(bool(cfg_vals.get("outlier_protect_pressure", False))),
            "cbo_history_outlier_pressure_quantile": float(cfg_vals.get("outlier_pressure_quantile", 0.75)),
            "cbo_history_outlier_pressure_fields_available": ",".join(sorted(pressure_thresholds.keys())),
            "cbo_history_outlier_candidate_rows": int(len(outlier_candidates)),
            "cbo_history_outlier_protected_rows": int(protected_rows),
            "cbo_history_outlier_filtered_rows_before_protection": int(len(selected)),
            "cbo_history_outlier_filtered_rows_after_protection": int(filtered_rows),
            "cbo_history_outlier_protected_ratio": float(protected_rows / max(1, len(selected))),
            "cbo_history_outlier_pressure_delay_threshold": float(delay_threshold) if np.isfinite(delay_threshold) else np.nan,
            "cbo_history_outlier_pressure_backlog_threshold": float(backlog_threshold) if np.isfinite(backlog_threshold) else np.nan,
            "cbo_history_outlier_pressure_unfinished_threshold": float(unfinished_threshold) if np.isfinite(unfinished_threshold) else np.nan,
            "cbo_history_outlier_pressure_violation_threshold": float(violation_threshold) if np.isfinite(violation_threshold) else np.nan,
            "cbo_history_outlier_residual_mean": float(np.mean(residual_arr)) if residual_arr.size else 0.0,
            "cbo_history_outlier_residual_max": float(np.max(residual_arr)) if residual_arr.size else 0.0,
            "cbo_history_outlier_threshold": float(cfg_vals["outlier_threshold"]),
            "cbo_history_outlier_abs_threshold": float(cfg_vals["outlier_abs_threshold"]),
            "cbo_history_outlier_max_filter_ratio": float(cfg_vals["outlier_max_filter_ratio"]),
            "cbo_history_outlier_scale": cfg_vals["outlier_scale"] if cfg_vals["outlier_scale"] in {"mad", "iqr", "std"} else "mad",
            "cbo_history_denoise_y_used_mean": _cbo_history_denoise_default_stats(agent, raw_rows=raw_rows, y_raw=y, y_used=y_out)["cbo_history_denoise_y_used_mean"],
        })
        _cbo_history_store_stats(agent, stats)
        return x_out, y_out, records_out, stats

    y_out = torch.as_tensor(y_used.reshape(y_np.shape), dtype=y.dtype, device=y.device) if hasattr(y, "device") else y_used.reshape(y_np.shape)
    delta = np.abs(y_used - y_flat)
    finite_delta = delta[np.isfinite(delta)]
    stats.update({
        "cbo_history_denoise_smoothed_rows": int(applied),
        "cbo_history_denoise_unsmoothed_rows": int(raw_rows - applied),
        "cbo_history_denoise_smoothed_ratio": float(applied / max(1, raw_rows)),
        "cbo_history_denoise_neighbor_count_mean": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "cbo_history_denoise_neighbor_count_max": int(max(neighbor_counts)) if neighbor_counts else 0,
        "cbo_history_denoise_abs_delta_mean": float(np.mean(finite_delta)) if finite_delta.size else 0.0,
        "cbo_history_denoise_abs_delta_max": float(np.max(finite_delta)) if finite_delta.size else 0.0,
        "cbo_history_denoise_y_used_mean": _cbo_history_denoise_default_stats(agent, raw_rows=raw_rows, y_raw=y, y_used=y_out)["cbo_history_denoise_y_used_mean"],
    })
    _cbo_history_store_stats(agent, stats)
    return x, y_out, records, stats


def denoise_training_targets(agent, x, y, metadata=None, cfg=None):
    _x_out, y_out, _records_out, stats = prepare_gp_training_data(agent, x, y, metadata=metadata, cfg=cfg)
    return y_out, stats


def _node_count_backlog(nodes):
    return int(sum(len(n.ready_queue) + len(n.running_tasks) for n in nodes))


def _node_count_ready_queue(nodes):
    return int(sum(len(n.ready_queue) for n in nodes))


def _pressure_for_nodes(nodes):
    if not nodes:
        return 0.0
    vals = []
    for n in nodes:
        util = float(n.utilization())
        q = len(n.ready_queue) + len(n.running_tasks)
        denom = max(1.0, float(getattr(n, "cpu_total", 1)) / 8.0)
        vals.append(0.65 * util + 0.35 * min(1.0, q / denom))
    return float(np.mean(vals)) if vals else 0.0


def _last_perf_value(fac, key, default=0.0):
    """读取上一窗口日志值。当前窗口开始时可观测，不使用本窗口未来结果。"""
    try:
        vals = getattr(fac, "perf_log", {}).get(key, [])
        if vals:
            v = vals[-1]
            if v is not None and np.isfinite(float(v)):
                return float(v)
    except Exception:
        pass
    return float(default)


def _finite_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _perf_float_series(fac, key):
    try:
        vals = list(getattr(fac, "perf_log", {}).get(key, []) or [])
    except Exception:
        vals = []
    return [_finite_float(v, np.nan) for v in vals]


def _unfinished_rate_history(fac):
    unfinished_vals = _perf_float_series(fac, "unfinished_end")
    if not unfinished_vals:
        unfinished_vals = _perf_float_series(fac, "backlog")
    arrival_vals = _perf_float_series(fac, "arrivals_total")
    n = min(len(unfinished_vals), len(arrival_vals))
    rates = []
    for unfinished, arrivals in zip(unfinished_vals[-n:], arrival_vals[-n:]):
        if not np.isfinite(unfinished) or not np.isfinite(arrivals):
            continue
        rate = max(0.0, float(unfinished)) / max(float(arrivals), 1.0)
        rates.append(float(np.clip(rate, 0.0, 1.0)))
    return rates


def _mean_or_zero(vals):
    arr = np.array([float(v) for v in vals if np.isfinite(float(v))], dtype=float)
    return float(np.mean(arr)) if arr.size else 0.0


def _unfinished_context_features(fac):
    rates = _unfinished_rate_history(fac)
    hist_n = len(rates)
    if hist_n <= 0:
        return 0.0, 0.0, 0.0, "insufficient_history:no_previous_window"
    prev_rate = float(np.clip(rates[-1], 0.0, 1.0))
    recent_mean = float(np.clip(_mean_or_zero(rates[-20:]), 0.0, 1.0))
    older = rates[:-10][-30:] if hist_n > 10 else []
    if older:
        trend = _mean_or_zero(rates[-10:]) - _mean_or_zero(older)
        status = "ok" if hist_n >= 40 else "insufficient_history:older_window"
    else:
        trend = 0.0
        status = "insufficient_history:older_window"
    return prev_rate, recent_mean, float(np.clip(trend, -1.0, 1.0)), status


def build_lite_context_vector(fac, base_context=None):
    """构造 CBO-lite 的窗口开始状态。

    只使用当前决策前能看到的状态，不使用本窗口结束后的结果指标。
    """
    try:
        m = fac.scenario_monitor.compute_metrics(fac.current_time)
    except Exception:
        m = {}
    arrival_rate = float(m.get("arrival_rate", 0.0))
    nodes = list(getattr(fac, "nodes", []))
    utils = [float(n.utilization()) for n in nodes] if nodes else [0.0]
    backlog = float(_node_count_backlog(nodes))
    queue_total = float(_node_count_ready_queue(nodes))
    edge_nodes = [n for n in nodes if not getattr(n, "is_cloud", False)]
    cloud_nodes = [n for n in nodes if getattr(n, "is_cloud", False)]
    edge_pressure = _pressure_for_nodes(edge_nodes)
    cloud_pressure = _pressure_for_nodes(cloud_nodes)
    gap = float(np.clip(edge_pressure - cloud_pressure, -1.0, 1.0))

    # 当前配置/当前时间段的任务结构。固定 36 场景时它就是该场景的任务比例；
    # 若启用 TASK_TYPE_PROB_SCHEDULE，它会随 current_time 分段变化。
    try:
        cfg_probs = get_task_type_probs_at_time(getattr(fac, "current_time", None))
    except Exception:
        cfg_probs = _normalize_task_probs(getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3}))
    cfg_rt = float(cfg_probs.get("RT", 0.0))
    cfg_batch = float(cfg_probs.get("Batch", 0.0))
    cfg_ai = float(cfg_probs.get("AI", 0.0))

    # 上一窗口实际到达数量与占比。这里用上一轮 perf_log，避免看当前窗口结束结果。
    prev_rt = _last_perf_value(fac, "arrivals_rt", 0.0)
    prev_batch = _last_perf_value(fac, "arrivals_batch", 0.0)
    prev_ai = _last_perf_value(fac, "arrivals_ai", 0.0)
    prev_total = max(0.0, prev_rt + prev_batch + prev_ai)
    # 参考值约等于 λ*BO_INTERVAL，用于把上一窗口任务数压到 0~1。
    # 如果当前 m 里没有 arrival_rate，则回退到 CFG 当前 λ 或 1.0。
    count_ref = max(1.0, float(arrival_rate) * float(getattr(CFG, "BO_INTERVAL", 1.0)))
    if count_ref <= 1.0:
        try:
            lam, _ = fac.workload._get_lambda(getattr(fac, "current_time", 0.0))
            count_ref = max(1.0, float(lam) * float(getattr(CFG, "BO_INTERVAL", 1.0)))
        except Exception:
            count_ref = max(1.0, float(getattr(CFG, "BO_INTERVAL", 1.0)))
    prev_rt_norm = float(np.clip(prev_rt / count_ref, 0.0, 1.0))
    prev_batch_norm = float(np.clip(prev_batch / count_ref, 0.0, 1.0))
    prev_ai_norm = float(np.clip(prev_ai / count_ref, 0.0, 1.0))
    if prev_total > 0:
        prev_rt_ratio = prev_rt / prev_total
        prev_batch_ratio = prev_batch / prev_total
        prev_ai_ratio = prev_ai / prev_total
    else:
        # 第 1 轮没有上一窗口，用配置比例兜底，避免冷启动 context 全 0。
        prev_rt_ratio = cfg_rt
        prev_batch_ratio = cfg_batch
        prev_ai_ratio = cfg_ai

    prev_unfinished_rate, recent_unfinished_rate_mean, unfinished_rate_trend, unfinished_status = _unfinished_context_features(fac)
    try:
        fac._last_unfinished_context_status = unfinished_status
    except Exception:
        pass

    return [
        float(np.clip(arrival_rate, 0.0, 5.0)),
        float(np.clip(backlog, 0.0, 500.0)),
        float(np.clip(np.mean(utils), 0.0, 1.0)),
        float(np.clip(np.max(utils), 0.0, 1.0)),
        float(np.clip(queue_total, 0.0, 500.0)),
        gap,
        float(np.clip(cfg_rt, 0.0, 1.0)),
        float(np.clip(cfg_batch, 0.0, 1.0)),
        float(np.clip(cfg_ai, 0.0, 1.0)),
        prev_rt_norm,
        prev_batch_norm,
        prev_ai_norm,
        float(np.clip(prev_rt_ratio, 0.0, 1.0)),
        float(np.clip(prev_batch_ratio, 0.0, 1.0)),
        float(np.clip(prev_ai_ratio, 0.0, 1.0)),
        prev_unfinished_rate,
        recent_unfinished_rate_mean,
        unfinished_rate_trend,
    ]


def build_context_for_group(fac, group_cfg, base_context=None):
    mode = str(group_cfg.get("context_mode", "legacy") or "legacy").strip().lower()
    if mode in set(LITE_CONTEXT_MODE_SPECS.keys()) | {"state_lite", "cbo_lite"}:
        full_vec = build_lite_context_vector(fac, base_context=base_context)
        if mode in {"state_lite", "cbo_lite"}:
            mode = "lite"
        ctx = slice_lite_context_vector(full_vec, context_mode=mode)
        try:
            label = str(LITE_CONTEXT_MODE_SPECS.get(mode, {}).get("label", mode))
            fac._last_context_mode = label
            fac._last_context_feature_names = lite_context_feature_names(mode)
            fac._last_context_status = getattr(fac, "_last_unfinished_context_status", "ok") if "unfinished" in label else "ok"
        except Exception:
            pass
        return ctx
    try:
        fac._last_context_mode = str(mode)
        fac._last_context_feature_names = list(getattr(CFG, "CONTEXT_FEATURE_NAMES", []))
        fac._last_context_status = "legacy"
    except Exception:
        pass
    return base_context


def compute_feedback_confidence(metrics, group_cfg=None):
    """估计当前窗口 feedback 可信度。

    它不是最终评价指标，只是告诉 BO：这一轮 observation 是否容易被随机到达、
    低完成率、强任务比例偏移污染。
    """
    arrivals = max(0.0, _safe_float(metrics.get("arrivals_total", metrics.get("arrivals", 0.0)), 0.0))
    completed = max(0.0, _safe_float(metrics.get("completed_total", metrics.get("task_count", 0.0)), 0.0))
    unfinished = max(0.0, _safe_float(metrics.get("unfinished_end", metrics.get("backlog", 0.0)), 0.0))
    comp_ratio = completed / max(1.0, arrivals)

    # 样本数量越多，单窗口均值越稳。阈值不宜太高，避免高负载窗口被过度丢弃。
    count_conf = min(1.0, math.log1p(arrivals) / math.log1p(float(getattr(CFG, "BO_CONFIDENCE_TASK_REF", 60.0))))
    complete_conf = min(1.0, comp_ratio / max(1e-9, float(getattr(CFG, "BO_CONFIDENCE_COMPLETION_REF", 0.65))))
    unfinished_ratio = unfinished / max(1.0, arrivals + unfinished)
    unfinished_conf = max(0.20, 1.0 - 0.60 * min(1.0, unfinished_ratio))

    ref = _normalize_task_probs(getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3}))
    actual = {
        "RT": _safe_float(metrics.get("rt_arrival_ratio", 0.0), 0.0),
        "Batch": _safe_float(metrics.get("batch_arrival_ratio", 0.0), 0.0),
        "AI": _safe_float(metrics.get("ai_arrival_ratio", 0.0), 0.0),
    }
    # L1 偏移最大约为 2；这里保留温和惩罚，避免单窗口比例抖动完全支配。
    mix_l1 = sum(abs(float(actual[k]) - float(ref.get(k, 0.0))) for k in ["RT", "Batch", "AI"])
    mix_conf = max(0.25, 1.0 - 0.50 * min(1.0, mix_l1))

    zero_penalty = _safe_float(metrics.get("zero_completion_penalty", 0.0), 0.0)
    zero_conf = 0.25 if zero_penalty > 0 else 1.0

    conf = (
        0.30 * count_conf
        + 0.30 * complete_conf
        + 0.20 * unfinished_conf
        + 0.15 * mix_conf
        + 0.05 * zero_conf
    )
    return float(np.clip(conf, 0.05, 1.0)), {
        "confidence_count": float(count_conf),
        "confidence_completion": float(complete_conf),
        "confidence_unfinished": float(unfinished_conf),
        "confidence_mix": float(mix_conf),
        "confidence_zero": float(zero_conf),
        "confidence_completion_ratio": float(comp_ratio),
        "confidence_unfinished_ratio": float(unfinished_ratio),
        "confidence_task_mix_l1": float(mix_l1),
    }


def log_feedback_confidence(fac, confidence, parts, group_cfg=None):
    data = {"feedback_confidence": float(confidence)}
    data.update(parts or {})
    data["bo_history_mode"] = str(group_cfg.get("history_mode", _cfg_history_mode())) if group_cfg else _cfg_history_mode()
    data["bo_recent_window"] = int(group_cfg.get("recent_window", _cfg_recent_window())) if group_cfg else _cfg_recent_window()
    data["effective_history_mode"] = data["bo_history_mode"]
    data["effective_recent_window"] = data["bo_recent_window"]
    data["history_override_source"] = str(group_cfg.get("history_override_source", "method_default")) if group_cfg else "global_default"
    data["bo_train_sample_count"] = int(getattr(getattr(fac, "agent", None), "last_debug_info", {}).get("training_sample_count", -1)) if getattr(fac, "agent", None) is not None else -1
    data["bo_confidence_min"] = float(group_cfg.get("confidence_min", _cfg_confidence_min())) if group_cfg else _cfg_confidence_min()
    for k, v in data.items():
        fac.perf_log.setdefault(k, []).append(v)


def configure_refactor_agent(agent, group_cfg):
    if agent is None:
        return
    if group_cfg is None:
        group_cfg = {}
    agent.history_mode = str(group_cfg.get("history_mode", _cfg_history_mode("all"))).strip().lower()
    agent.recent_window = int(group_cfg.get("recent_window", _cfg_recent_window()))
    agent.confidence_min = float(group_cfg.get("confidence_min", _cfg_confidence_min()))
    agent.confidence_min_samples = int(group_cfg.get("confidence_min_samples", _cfg_confidence_min_samples()))
    agent.context_mode = str(group_cfg.get("context_mode", "legacy"))
    agent.history_override_source = str(group_cfg.get("history_override_source", "method_default"))
    agent.is_cbo_stability_enabled = bool(_is_cbo_method_key(str(group_cfg.get("group_key", "")), group_cfg))
    agent.cbo_history_select_mode = str(group_cfg.get("cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")) if agent.is_cbo_stability_enabled else "recent").strip().lower()
    agent.cbo_context_k = int(group_cfg.get("cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50)))
    agent.cbo_elite_k = int(group_cfg.get("cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20)))
    agent.cbo_diverse_k = int(group_cfg.get("cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20)))
    agent.cbo_robust_score_mode = str(group_cfg.get("cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) if agent.is_cbo_stability_enabled else "none").strip().lower()
    agent.cbo_robust_std_weight = float(group_cfg.get("cbo_robust_std_weight", _cfg_cbo_float("CBO_ROBUST_STD_WEIGHT", 0.5)))
    agent.cbo_theta_merge_eps = float(group_cfg.get("cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05)))
    agent.cbo_context_sim_threshold = float(group_cfg.get("cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
    agent.cbo_tr_mode = str(group_cfg.get("cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_tr_anchor_mode = str(group_cfg.get("cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean"))).strip().lower()
    agent.cbo_tr_radius_min = float(group_cfg.get("cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
    agent.cbo_tr_radius_max = float(group_cfg.get("cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
    agent.cbo_tr_grow = float(group_cfg.get("cbo_tr_grow", _cfg_cbo_float("CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15))))
    agent.cbo_tr_shrink = float(group_cfg.get("cbo_tr_shrink", _cfg_cbo_float("CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92))))
    agent.cbo_tr_update_mode = str(group_cfg.get("cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far"))).strip().lower()
    agent.cbo_tr_compare_window = int(group_cfg.get("cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)))
    agent.cbo_tr_baseline_window = int(group_cfg.get("cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)))
    agent.cbo_tr_improve_pct = float(group_cfg.get("cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015)))
    agent.cbo_tr_worsen_pct = float(group_cfg.get("cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03)))
    agent.cbo_tr_deadband_pct = float(group_cfg.get("cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01)))
    agent.cbo_tr_update_patience = int(group_cfg.get("cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)))
    agent.cbo_tr_update_patience_count = int(getattr(agent, "cbo_tr_update_patience_count", 0))
    agent.cbo_tr_update_last_signal = str(getattr(agent, "cbo_tr_update_last_signal", ""))
    agent.cbo_tr_cost_history = list(getattr(agent, "cbo_tr_cost_history", []))
    agent.cbo_robust_incumbent_mode = str(group_cfg.get("cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_macro_gate_mode = str(group_cfg.get("cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_macro_k = int(group_cfg.get("cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100)))
    agent.cbo_macro_total_scale = str(group_cfg.get("cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")) or "auto")
    agent.cbo_macro_lengthscale_total = float(group_cfg.get("cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0)))
    agent.cbo_macro_lengthscale_rt = float(group_cfg.get("cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15)))
    agent.cbo_macro_lengthscale_batch = float(group_cfg.get("cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15)))
    agent.cbo_macro_alpha = float(group_cfg.get("cbo_macro_alpha", _cfg_cbo_float("CBO_MACRO_ALPHA", 1.0)))
    agent.cbo_dump_candidates = bool(group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False))))
    agent.cbo_dump_candidates_every = int(group_cfg.get("cbo_dump_candidates_every", _cfg_cbo_int("CBO_DUMP_CANDIDATES_EVERY", 20)))
    agent.cbo_dump_candidates_topn = int(group_cfg.get("cbo_dump_candidates_topn", _cfg_cbo_int("CBO_DUMP_CANDIDATES_TOPN", 30)))
    agent.cbo_select_mode = str(group_cfg.get("cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy"))).strip().lower()
    agent.cbo_topk = int(group_cfg.get("cbo_topk", _cfg_cbo_int("CBO_TOPK", 5)))
    agent.cbo_select_temperature = float(group_cfg.get("cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20)))
    agent.cbo_epsilon = float(group_cfg.get("cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)))
    agent.cbo_acq_beta = float(group_cfg.get("cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)))
    agent.cbo_acq_beta_mode = str(group_cfg.get("cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed"))).strip().lower()
    agent.cbo_beta_min = float(group_cfg.get("cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1)))
    agent.cbo_beta_max = float(group_cfg.get("cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0)))
    agent.cbo_radius_beta_power = float(group_cfg.get("cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0)))
    agent.cbo_good_region_guard = str(group_cfg.get("cbo_good_region_guard", _cfg_cbo_str("CBO_GOOD_REGION_GUARD", "off"))).strip().lower()
    agent.cbo_good_region_window = int(group_cfg.get("cbo_good_region_window", _cfg_cbo_int("CBO_GOOD_REGION_WINDOW", 50)))
    agent.cbo_good_region_worse_pct = float(group_cfg.get("cbo_good_region_worse_pct", _cfg_cbo_float("CBO_GOOD_REGION_WORSE_PCT", 0.03)))
    agent.cbo_good_region_distance_threshold = float(group_cfg.get("cbo_good_region_distance_threshold", _cfg_cbo_float("CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35)))
    agent.cbo_good_region_tr_radius_threshold = float(group_cfg.get("cbo_good_region_tr_radius_threshold", _cfg_cbo_float("CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15)))
    agent.cbo_good_region_beta_threshold = float(group_cfg.get("cbo_good_region_beta_threshold", _cfg_cbo_float("CBO_GOOD_REGION_BETA_THRESHOLD", 0.5)))
    agent.cbo_good_region_guard_mode = str(group_cfg.get("cbo_good_region_guard_mode", _cfg_cbo_str("CBO_GOOD_REGION_GUARD_MODE", "conservative"))).strip().lower()
    agent.cbo_radius_stable_rebound_pct = float(group_cfg.get("cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)))
    agent.cbo_radius_unstable_rebound_pct = float(group_cfg.get("cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)))
    agent.cbo_radius_surprise_boost_threshold = float(group_cfg.get("cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)))
    agent.cbo_radius_beta_boost = float(group_cfg.get("cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5)))
    agent.cbo_radius_beta_cap = float(group_cfg.get("cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0)))
    agent.cbo_service_guard_mode = str(group_cfg.get("cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off"))).strip().lower()
    agent.cbo_service_guard_delay_pct = float(group_cfg.get("cbo_service_guard_delay_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_DELAY_PCT", 0.03)))
    agent.cbo_service_guard_backlog_pct = float(group_cfg.get("cbo_service_guard_backlog_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03)))
    agent.cbo_surprise_window = int(group_cfg.get("cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10)))
    agent.cbo_surprise_z_threshold = float(group_cfg.get("cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)))
    agent.cbo_surprise_cost_gap_pct = float(group_cfg.get("cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)))
    agent.cbo_sigma_floor = float(group_cfg.get("cbo_sigma_floor", _cfg_cbo_float("CBO_SIGMA_FLOOR", 1e-6)))
    agent.cbo_history_denoise_mode = str(group_cfg.get("cbo_history_denoise_mode", _cfg_cbo_str("CBO_HISTORY_DENOISE_MODE", "off"))).strip().lower()
    agent.cbo_history_denoise_k = int(group_cfg.get("cbo_history_denoise_k", _cfg_cbo_int("CBO_HISTORY_DENOISE_K", 7)))
    agent.cbo_history_denoise_radius = float(group_cfg.get("cbo_history_denoise_radius", _cfg_cbo_float("CBO_HISTORY_DENOISE_RADIUS", 0.12)))
    agent.cbo_history_denoise_min_neighbors = int(group_cfg.get("cbo_history_denoise_min_neighbors", _cfg_cbo_int("CBO_HISTORY_DENOISE_MIN_NEIGHBORS", 3)))
    agent.cbo_history_denoise_context_weight = float(group_cfg.get("cbo_history_denoise_context_weight", _cfg_cbo_float("CBO_HISTORY_DENOISE_CONTEXT_WEIGHT", 1.0)))
    agent.cbo_history_denoise_theta_weight = float(group_cfg.get("cbo_history_denoise_theta_weight", _cfg_cbo_float("CBO_HISTORY_DENOISE_THETA_WEIGHT", 1.0)))
    agent.cbo_history_denoise_stat = str(group_cfg.get("cbo_history_denoise_stat", _cfg_cbo_str("CBO_HISTORY_DENOISE_STAT", "median"))).strip().lower()
    agent.cbo_history_denoise_trim_pct = float(group_cfg.get("cbo_history_denoise_trim_pct", _cfg_cbo_float("CBO_HISTORY_DENOISE_TRIM_PCT", 0.1)))
    agent.cbo_history_denoise_apply_to = str(group_cfg.get("cbo_history_denoise_apply_to", _cfg_cbo_str("CBO_HISTORY_DENOISE_APPLY_TO", "all"))).strip().lower()
    agent.cbo_history_outlier_threshold = float(group_cfg.get("cbo_history_outlier_threshold", _cfg_cbo_float("CBO_HISTORY_OUTLIER_THRESHOLD", 3.0)))
    agent.cbo_history_outlier_abs_threshold = float(group_cfg.get("cbo_history_outlier_abs_threshold", _cfg_cbo_float("CBO_HISTORY_OUTLIER_ABS_THRESHOLD", 500.0)))
    agent.cbo_history_outlier_max_filter_ratio = float(group_cfg.get("cbo_history_outlier_max_filter_ratio", _cfg_cbo_float("CBO_HISTORY_OUTLIER_MAX_FILTER_RATIO", 0.2)))
    agent.cbo_history_outlier_scale = str(group_cfg.get("cbo_history_outlier_scale", _cfg_cbo_str("CBO_HISTORY_OUTLIER_SCALE", "mad"))).strip().lower()
    agent.cbo_history_outlier_theta_radius = float(group_cfg.get("cbo_history_outlier_theta_radius", _cfg_cbo_float("CBO_HISTORY_OUTLIER_THETA_RADIUS", 0.12)))
    agent.cbo_history_outlier_context_radius = float(group_cfg.get("cbo_history_outlier_context_radius", _cfg_cbo_float("CBO_HISTORY_OUTLIER_CONTEXT_RADIUS", 0.50)))
    agent.cbo_history_outlier_min_peers = int(group_cfg.get("cbo_history_outlier_min_peers", _cfg_cbo_int("CBO_HISTORY_OUTLIER_MIN_PEERS", 3)))
    agent.cbo_history_outlier_use_leave_one_out = bool(group_cfg.get("cbo_history_outlier_use_leave_one_out", getattr(CFG, "CBO_HISTORY_OUTLIER_USE_LEAVE_ONE_OUT", True)))
    agent.cbo_history_outlier_export_filtered = bool(group_cfg.get("cbo_history_outlier_export_filtered", getattr(CFG, "CBO_HISTORY_OUTLIER_EXPORT_FILTERED", True)))
    agent.cbo_history_outlier_protect_pressure = bool(group_cfg.get("cbo_history_outlier_protect_pressure", getattr(CFG, "CBO_HISTORY_OUTLIER_PROTECT_PRESSURE", False)))
    agent.cbo_history_outlier_pressure_quantile = float(group_cfg.get("cbo_history_outlier_pressure_quantile", _cfg_cbo_float("CBO_HISTORY_OUTLIER_PRESSURE_QUANTILE", 0.75)))
    agent.cbo_history_outlier_protect_high_cost_only = bool(group_cfg.get("cbo_history_outlier_protect_high_cost_only", getattr(CFG, "CBO_HISTORY_OUTLIER_PROTECT_HIGH_COST_ONLY", True)))
    agent.cbo_history_outlier_pressure_fields = str(group_cfg.get("cbo_history_outlier_pressure_fields", getattr(CFG, "CBO_HISTORY_OUTLIER_PRESSURE_FIELDS", "Avg_Delay,Backlog,unfinished_end,Violation")))
    agent.cbo_last_history_denoise_stats = _cbo_history_denoise_default_stats(agent, raw_rows=0)
    agent.cbo_radius_reset = float(group_cfg.get("cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)))
    agent.cbo_radius_min_stuck_rounds = int(group_cfg.get("cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)))
    agent.cbo_rebound_window = int(group_cfg.get("cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20)))
    agent.cbo_rebound_threshold_pct = float(group_cfg.get("cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)))
    agent.cbo_selection_cooldown = int(group_cfg.get("cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5)))
    agent.cbo_condition_anchor_switch = str(group_cfg.get("cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best"))).strip().lower()
    agent.cbo_force_explore_countdown = int(getattr(agent, "cbo_force_explore_countdown", 0))
    agent.cbo_radius_min_stuck_count = int(getattr(agent, "cbo_radius_min_stuck_count", 0))
    agent.cbo_surprise_history = list(getattr(agent, "cbo_surprise_history", []))
    if agent.cbo_tr_mode != "off":
        agent.use_trust_region = True
    agent.trust_radius = float(group_cfg.get("cbo_tr_radius_init", getattr(CFG, "TRUST_RADIUS_INIT", getattr(agent, "trust_radius", 0.10))))
    agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0))
    agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0))
    agent.cbo_tr_update_reason = str(getattr(agent, "cbo_tr_update_reason", "init"))


def agent_tell_with_feedback_meta(agent, theta, cost, state=None, context=None, metrics=None, bo_iter=None, group_key=None, group_cfg=None, confidence=None, parts=None):
    agent.tell(theta, cost, state=state, context=context)
    try:
        rec = None
        for r in reversed(list(getattr(agent, "local_recent", []))):
            if isinstance(r, dict):
                rec = r
                break
        if rec is not None:
            rec["feedback_confidence"] = float(confidence if confidence is not None else 1.0)
            rec["bo_iter"] = int(bo_iter) if bo_iter is not None else None
            rec["group_key"] = str(group_key) if group_key is not None else None
            rec["history_mode"] = str(getattr(agent, "history_mode", _cfg_history_mode()))
            rec["feedback_metrics_meta"] = dict(parts or {})
            if isinstance(metrics, dict):
                rec["metrics"] = dict(metrics)
                for macro_key in [
                    "arrivals_total", "arrivals_rt", "arrivals_batch", "arrivals_ai",
                    "rt_arrival_ratio", "batch_arrival_ratio", "ai_arrival_ratio",
                    "completed_total", "task_count", "unfinished_end", "backlog",
                ]:
                    if macro_key in metrics:
                        rec[macro_key] = metrics.get(macro_key)
            try:
                theta_norm = agent._normalize_theta(theta)
                prev_best = getattr(agent, "prev_best", None)
                prev_best_value = getattr(agent, "prev_best_value", None)
                if (
                    prev_best is not None
                    and prev_best_value is not None
                    and np.allclose(np.array(prev_best, dtype=float), np.array(theta_norm, dtype=float), atol=1e-9, rtol=1e-9)
                    and abs(float(prev_best_value) + float(cost)) <= 1e-9 * max(1.0, abs(float(cost)))
                ):
                    agent.prev_best_iter = int(bo_iter) + 1 if bo_iter is not None else None
            except Exception:
                pass
    except Exception:
        pass


_ORIG_AGENT_COLLECT_SAMPLES = FederatedBOAgent._collect_samples


def _cbo_record_cost(rec):
    try:
        return -float(rec.get("y", np.nan))
    except Exception:
        return np.nan


def _cbo_theta_norm(agent, theta):
    theta = np.asarray(agent._normalize_theta(theta), dtype=float)
    low = np.asarray(agent.bounds[0].tolist(), dtype=float)
    high = np.asarray(agent.bounds[1].tolist(), dtype=float)
    denom = np.maximum(1e-12, high - low)
    return np.clip((theta - low) / denom, 0.0, 1.0)


def _cbo_context_similarity(agent, context, rec):
    if context is None or rec.get("context") is None:
        return 1.0 if context is None else 0.0
    try:
        return float(agent._context_similarity(context, rec.get("context")))
    except Exception:
        return 0.0


def _cbo_all_records(agent):
    records = []
    for bucket in getattr(agent, "local_archive", {}).values():
        records.extend([agent._unpack_sample(s) for s in bucket])
    records.extend([agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])])
    return records


def _cbo_rec_value(rec, keys, default=np.nan):
    if rec is None:
        return default
    metrics = rec.get("metrics") if isinstance(rec, dict) else None
    for key in keys:
        try:
            if isinstance(rec, dict) and key in rec and rec.get(key) is not None:
                return float(rec.get(key))
            if isinstance(metrics, dict) and key in metrics and metrics.get(key) is not None:
                return float(metrics.get(key))
        except Exception:
            continue
    return default


def _cbo_macro_total_scale(agent, records):
    mode = str(getattr(agent, "cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")) or "auto").strip().lower()
    if mode != "auto":
        try:
            return max(1e-12, float(mode))
        except Exception:
            return 1.0
    totals = []
    for rec in records or []:
        macro = _cbo_macro_context_from_record(agent, rec, records=None, scale=1.0, allow_auto=False)
        total_raw = macro.get("total_arrivals_raw")
        if np.isfinite(total_raw) and total_raw > 0:
            totals.append(float(total_raw))
    if not totals:
        return 1.0
    vals = np.asarray(totals, dtype=float)
    return max(1.0, float(np.nanpercentile(vals, 90)))


def _cbo_macro_context_from_record(agent, rec, records=None, scale=None, allow_auto=True):
    if scale is None:
        scale = _cbo_macro_total_scale(agent, records or []) if allow_auto else 1.0
    rt = _cbo_rec_value(rec, ["arrivals_rt", "Prev_RT_Arrivals", "window_rt_arrivals", "rt_arrivals"], np.nan)
    batch = _cbo_rec_value(rec, ["arrivals_batch", "Prev_Batch_Arrivals", "window_batch_arrivals", "batch_arrivals"], np.nan)
    ai = _cbo_rec_value(rec, ["arrivals_ai", "Prev_AI_Arrivals", "window_ai_arrivals", "ai_arrivals"], np.nan)
    total = _cbo_rec_value(rec, ["arrivals_total", "task_count", "generated_total", "completed_total"], np.nan)
    if not np.isfinite(total):
        if np.isfinite(rt) or np.isfinite(batch) or np.isfinite(ai):
            total = float(np.nan_to_num(rt, nan=0.0) + np.nan_to_num(batch, nan=0.0) + np.nan_to_num(ai, nan=0.0))
    rt_ratio = _cbo_rec_value(rec, ["rt_arrival_ratio", "prev_rt_arrival_ratio", "RT_Ratio"], np.nan)
    batch_ratio = _cbo_rec_value(rec, ["batch_arrival_ratio", "prev_batch_arrival_ratio", "Batch_Ratio"], np.nan)
    if np.isfinite(total) and total > 0:
        if not np.isfinite(rt_ratio) and np.isfinite(rt):
            rt_ratio = float(rt) / max(1e-12, float(total))
        if not np.isfinite(batch_ratio) and np.isfinite(batch):
            batch_ratio = float(batch) / max(1e-12, float(total))
    elif isinstance(rec, dict) and rec.get("context") is not None and getattr(agent, "use_context", False):
        try:
            names = lite_context_feature_names(getattr(agent, "context_mode", "pressure_taskmix_counts"))
            ctx = list(agent._normalize_context(rec.get("context")))
            name_to_val = {str(n): float(ctx[i]) for i, n in enumerate(names[:len(ctx)])}
            rt_norm = name_to_val.get("prev_rt_arrivals_norm", np.nan)
            batch_norm = name_to_val.get("prev_batch_arrivals_norm", np.nan)
            ai_norm = name_to_val.get("prev_ai_arrivals_norm", np.nan)
            norm_total = float(np.nan_to_num(rt_norm, nan=0.0) + np.nan_to_num(batch_norm, nan=0.0) + np.nan_to_num(ai_norm, nan=0.0))
            if norm_total > 0:
                total = norm_total
                rt_ratio = float(np.nan_to_num(rt_norm, nan=0.0)) / norm_total
                batch_ratio = float(np.nan_to_num(batch_norm, nan=0.0)) / norm_total
            rt_ratio = name_to_val.get("prev_rt_arrival_ratio", rt_ratio)
            batch_ratio = name_to_val.get("prev_batch_arrival_ratio", batch_ratio)
        except Exception:
            pass
    total_norm = float(total) / max(1e-12, float(scale)) if np.isfinite(total) else np.nan
    return {
        "total_arrivals_raw": float(total) if np.isfinite(total) else np.nan,
        "total_arrivals_norm": float(total_norm) if np.isfinite(total_norm) else np.nan,
        "rt_ratio": float(rt_ratio) if np.isfinite(rt_ratio) else np.nan,
        "batch_ratio": float(batch_ratio) if np.isfinite(batch_ratio) else np.nan,
    }


def _cbo_macro_context_from_context(agent, context, records=None, scale=None):
    pseudo = {"context": context}
    if context is not None and getattr(agent, "use_context", False):
        try:
            names = lite_context_feature_names(getattr(agent, "context_mode", "pressure_taskmix_counts"))
            ctx = list(agent._normalize_context(context))
            for i, name in enumerate(names[:len(ctx)]):
                pseudo[str(name)] = float(ctx[i])
        except Exception:
            pass
    return _cbo_macro_context_from_record(agent, pseudo, records=records, scale=scale)


def _cbo_macro_similarity(agent, macro_a, macro_b):
    try:
        vals = []
        for key, ls_name, default in [
            ("total_arrivals_norm", "cbo_macro_lengthscale_total", 1.0),
            ("rt_ratio", "cbo_macro_lengthscale_rt", 0.15),
            ("batch_ratio", "cbo_macro_lengthscale_batch", 0.15),
        ]:
            a = float(macro_a.get(key, np.nan))
            b = float(macro_b.get(key, np.nan))
            if not np.isfinite(a) or not np.isfinite(b):
                vals.append(1e6)
            else:
                ls = max(1e-12, float(getattr(agent, ls_name, default)))
                vals.append(((a - b) / ls) ** 2)
        sim = float(np.exp(-0.5 * float(np.sum(vals))))
        alpha = max(1e-12, float(getattr(agent, "cbo_macro_alpha", 1.0)))
        return float(sim ** alpha)
    except Exception:
        return 0.0


def _cbo_record_identity(agent, rec):
    try:
        theta = _cbo_theta_norm(agent, rec.get("theta", []))
        theta_key = tuple(np.round(theta, 8).tolist())
    except Exception:
        theta_key = tuple()
    try:
        ctx = np.asarray(agent._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(agent, "use_context", False) else np.zeros(0)
        ctx_key = tuple(np.round(ctx, 8).tolist())
    except Exception:
        ctx_key = tuple()
    return (
        int(rec.get("bo_iter", -1) if rec.get("bo_iter", -1) is not None else -1),
        str(rec.get("group_key", "")),
        theta_key,
        ctx_key,
    )


def _cbo_cluster_records(agent, records, context=None):
    eps = max(1e-9, float(getattr(agent, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
    mode = str(getattr(agent, "cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) or "none").lower()
    std_weight = float(getattr(agent, "cbo_robust_std_weight", _cfg_cbo_float("CBO_ROBUST_STD_WEIGHT", 0.5)))
    clusters = []
    for idx, rec in enumerate(records):
        theta = rec.get("theta")
        if theta is None:
            continue
        tn = _cbo_theta_norm(agent, theta)
        best_j, best_d = None, None
        for j, cluster in enumerate(clusters):
            d = float(np.linalg.norm(tn - cluster["theta_norm"]))
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j is not None and best_d <= eps:
            clusters[best_j]["records"].append(rec)
            n = len(clusters[best_j]["records"])
            clusters[best_j]["theta_norm"] = ((clusters[best_j]["theta_norm"] * (n - 1)) + tn) / n
        else:
            clusters.append({"theta_norm": tn, "records": [rec]})
    out = []
    for cluster in clusters:
        recs = cluster["records"]
        costs = np.asarray([_cbo_record_cost(r) for r in recs], dtype=float)
        costs = costs[np.isfinite(costs)]
        if costs.size == 0:
            continue
        sims = np.asarray([_cbo_context_similarity(agent, context, r) for r in recs], dtype=float)
        sims = np.clip(np.nan_to_num(sims, nan=0.0), 0.0, None)
        if float(np.sum(sims)) <= 1e-12:
            sims = np.ones(len(recs), dtype=float)
        cost_all = np.asarray([_cbo_record_cost(r) for r in recs], dtype=float)
        valid = np.isfinite(cost_all)
        cost_valid = cost_all[valid]
        sim_valid = sims[valid]
        sim_valid = sim_valid / max(1e-12, float(np.sum(sim_valid)))
        cw_mean = float(np.sum(sim_valid * cost_valid))
        cw_std = float(np.sqrt(np.sum(sim_valid * (cost_valid - cw_mean) ** 2)))
        mean_cost = float(np.mean(cost_valid))
        std_cost = float(np.std(cost_valid))
        if mode == "mean":
            robust_score = mean_cost
        elif mode == "mean_std":
            robust_score = mean_cost + std_weight * std_cost
        elif mode == "context_weighted_mean_std":
            robust_score = cw_mean + std_weight * cw_std
        else:
            robust_score = np.nan
        best_rec = min(recs, key=lambda r: _cbo_record_cost(r) if np.isfinite(_cbo_record_cost(r)) else float("inf"))
        rep_theta = list(best_rec.get("theta", []))
        contexts = [np.asarray(agent._normalize_context(r.get("context")), dtype=float) for r in recs if r.get("context") is not None and getattr(agent, "use_context", False)]
        mean_context = np.mean(np.vstack(contexts), axis=0).tolist() if contexts else None
        out.append({
            "records": recs,
            "eval_count": int(len(recs)),
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "min_cost": float(np.min(cost_valid)),
            "max_cost": float(np.max(cost_valid)),
            "recent_mean_cost": float(np.mean(cost_valid[-min(len(cost_valid), 20):])),
            "context_weighted_mean_cost": cw_mean,
            "context_weighted_std_cost": cw_std,
            "robust_score": float(robust_score) if np.isfinite(robust_score) else np.nan,
            "representative_theta": rep_theta,
            "mean_context": mean_context,
            "context_similarity_to_current": _cbo_context_similarity(agent, context, {"context": mean_context}) if mean_context is not None else (1.0 if context is None else 0.0),
        })
    return out


def _compute_robust_incumbent(self, context=None):
    mode = str(getattr(self, "cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) or "none").lower()
    if mode == "none":
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "robust_score_mode_none"}
    macro_mode = str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    macro_records = list(getattr(self, "_last_macro_pool_records", []) or [])
    source_pool = "macro_pool" if macro_mode == "hierarchical" and macro_records else "all_records"
    records = macro_records if source_pool == "macro_pool" else _cbo_all_records(self)
    clusters = _cbo_cluster_records(self, records, context=context)
    clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
    if not clusters:
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "no_valid_clusters"}
    threshold = float(getattr(self, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
    if threshold > 0:
        clusters = [c for c in clusters if float(c.get("context_similarity_to_current", 0.0)) >= threshold]
    if not clusters:
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "context_threshold_filtered_all"}
    multi = [c for c in clusters if int(c.get("eval_count", 0)) >= 2]
    pool = multi if multi else clusters
    best = min(pool, key=lambda c: float(c.get("robust_score", float("inf"))))
    info = {
        "robust_incumbent_available": True,
        "robust_incumbent_score": float(best.get("robust_score", np.nan)),
        "robust_incumbent_eval_count": int(best.get("eval_count", 0)),
        "robust_incumbent_mean_cost": float(best.get("mean_cost", np.nan)),
        "robust_incumbent_std_cost": float(best.get("std_cost", np.nan)),
        "robust_incumbent_context_similarity": float(best.get("context_similarity_to_current", np.nan)),
        "robust_incumbent_theta": list(best.get("representative_theta", [])),
        "robust_incumbent_used": False,
        "robust_incumbent_reason": "recommend_only",
        "robust_incumbent_source_pool": source_pool,
    }
    return list(best.get("representative_theta", [])), info


FederatedBOAgent._compute_robust_incumbent = _compute_robust_incumbent


def _refactor_collect_samples(self, state=None):
    records = list(_ORIG_AGENT_COLLECT_SAMPLES(self, state=state))
    if not records:
        return records
    context = getattr(self, "_active_context", None)
    select_mode = str(getattr(self, "cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")) or "recent").strip().lower()
    recent_window = max(2, int(getattr(self, "recent_window", _cfg_recent_window())))

    def set_debug(pool, recent_count=None, context_count=0, elite_count=0, diverse_count=0, sims=None, elite=None,
                  macro_count=0, macro_sims=None, macro_current=None, macro_pool_sims=None,
                  selected_macro_sims=None, selected_from_macro_pool_count=None, selected_outside_macro_pool_count=0,
                  macro_gate_fallback_used=False, macro_gate_fallback_reason="",
                  context_selection_source_pool="all_records", elite_selection_source_pool="all_records",
                  tr_anchor_source_pool="all_records"):
        sims = list(sims or [])
        macro_sims = list(macro_sims or [])
        macro_pool_sims = list(macro_pool_sims or [])
        selected_macro_sims = list(selected_macro_sims if selected_macro_sims is not None else macro_sims[:int(macro_count)])
        elite = elite or {}
        macro_current = macro_current or {}
        if selected_from_macro_pool_count is None:
            selected_from_macro_pool_count = int(macro_count)
        warm_count = sum(1 for rec in list(pool or []) if _cbo_is_warm_record(rec))
        local_count = int(len(list(pool or [])) - warm_count)
        self.last_history_debug = {
            "history_select_mode": select_mode,
            "effective_history_mode": str(getattr(self, "history_mode", _cfg_history_mode("all"))),
            "effective_recent_window": int(recent_window),
            "selected_recent_count": int(recent_count if recent_count is not None else min(len(getattr(self, "local_recent", [])), recent_window)),
            "selected_macro_count": int(macro_count),
            "selected_context_count": int(context_count),
            "selected_elite_count": int(elite_count),
            "selected_diverse_count": int(diverse_count),
            "selected_total_count": int(len(pool)),
            "selected_warm_rows_count": int(warm_count),
            "selected_local_rows_count": int(local_count),
            "cbo_warm_start_used_rows": int(warm_count),
            "cbo_macro_gate_mode": str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off"))),
            "macro_total_arrivals_norm": macro_current.get("total_arrivals_norm"),
            "macro_rt_ratio": macro_current.get("rt_ratio"),
            "macro_batch_ratio": macro_current.get("batch_ratio"),
            "macro_similarity_max": float(max(macro_sims)) if macro_sims else np.nan,
            "macro_similarity_mean": float(np.mean(macro_sims)) if macro_sims else np.nan,
            "macro_similarity_p50": float(np.percentile(macro_sims, 50)) if macro_sims else np.nan,
            "macro_similarity_p90": float(np.percentile(macro_sims, 90)) if macro_sims else np.nan,
            "selected_macro_mean_similarity": float(np.mean(selected_macro_sims)) if selected_macro_sims else np.nan,
            "selected_macro_min_similarity": float(np.min(selected_macro_sims)) if selected_macro_sims else np.nan,
            "selected_macro_max_similarity": float(np.max(selected_macro_sims)) if selected_macro_sims else np.nan,
            "macro_pool_count": int(len(macro_pool_sims)),
            "macro_pool_mean_similarity": float(np.mean(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_min_similarity": float(np.min(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_max_similarity": float(np.max(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_p50_similarity": float(np.percentile(macro_pool_sims, 50)) if macro_pool_sims else np.nan,
            "macro_pool_p90_similarity": float(np.percentile(macro_pool_sims, 90)) if macro_pool_sims else np.nan,
            "selected_from_macro_pool_count": int(selected_from_macro_pool_count),
            "selected_outside_macro_pool_count": int(selected_outside_macro_pool_count),
            "macro_gate_fallback_used": bool(macro_gate_fallback_used),
            "macro_gate_fallback_reason": str(macro_gate_fallback_reason),
            "context_selection_source_pool": str(context_selection_source_pool),
            "elite_selection_source_pool": str(elite_selection_source_pool),
            "tr_anchor_source_pool": str(tr_anchor_source_pool),
            "macro_k": int(getattr(self, "cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100))),
            "macro_lengthscale_total": float(getattr(self, "cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0))),
            "macro_lengthscale_rt": float(getattr(self, "cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15))),
            "macro_lengthscale_batch": float(getattr(self, "cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15))),
            "context_similarity_max": float(max(sims)) if sims else np.nan,
            "context_similarity_mean": float(np.mean(sims)) if sims else np.nan,
            "elite_best_robust_score": elite.get("robust_score"),
            "elite_best_eval_count": elite.get("eval_count"),
            "elite_best_mean_cost": elite.get("mean_cost"),
            "elite_best_std_cost": elite.get("std_cost"),
        }

    mode = str(getattr(self, "history_mode", _cfg_history_mode("all")) or "all").strip().lower()
    if mode in {"all", "legacy", "none"}:
        set_debug(records)
        return records

    # 按插入顺序近似时间顺序。local_recent 本身已经是时间顺序，archive 在前，recent 在后。
    min_keep = max(2, int(getattr(self, "confidence_min_samples", _cfg_confidence_min_samples())))
    min_conf = float(getattr(self, "confidence_min", _cfg_confidence_min()))

    macro_mode = str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    if select_mode in {"recent_context", "recent_context_elite", "hybrid"} or macro_mode != "off":
        recent = [self._unpack_sample(s) for s in list(getattr(self, "local_recent", []))[-recent_window:]]
        all_records = _cbo_all_records(self)
        threshold = float(getattr(self, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
        context_k = max(0, int(getattr(self, "cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50))))
        elite_k = max(0, int(getattr(self, "cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20))))
        diverse_k = max(0, int(getattr(self, "cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20))))
        macro_k = max(0, int(getattr(self, "cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100))))
        macro_scale = _cbo_macro_total_scale(self, all_records)
        macro_current = _cbo_macro_context_from_context(self, context, records=all_records, scale=macro_scale)
        macro_scored = []
        if macro_mode != "off" and macro_k > 0:
            for rec in all_records:
                rec_macro = _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale)
                sim_m = _cbo_macro_similarity(self, macro_current, rec_macro)
                macro_scored.append((sim_m, rec))
            macro_scored.sort(key=lambda x: x[0], reverse=True)
        macro_records = [rec for _, rec in macro_scored[:macro_k]]
        macro_sims = [float(s) for s, _ in macro_scored]
        macro_pool_sims = [float(s) for s, _ in macro_scored[:macro_k]]
        if macro_mode == "hierarchical":
            macro_pool = list(macro_records)
            macro_pool_keys = {_cbo_record_identity(self, r) for r in macro_pool}
            fallback_records = []
            fallback_used = False
            fallback_reason = ""
            if len(macro_pool) < min_keep:
                needed = max(0, min_keep - len(macro_pool))
                recent_tail = [self._unpack_sample(s) for s in list(getattr(self, "local_recent", []))[-recent_window:]]
                for rec in reversed(recent_tail):
                    if _cbo_record_identity(self, rec) not in macro_pool_keys:
                        fallback_records.append(rec)
                        if len(fallback_records) >= needed:
                            break
                fallback_used = bool(fallback_records)
                fallback_reason = f"macro_pool_below_min_keep added_recent={len(fallback_records)}" if fallback_used else "macro_pool_below_min_keep_no_recent_available"
            self._last_macro_pool_records = list(macro_pool)
            self._last_macro_pool_keys = set(macro_pool_keys)

            macro_recent_keep = min(recent_window, max(min_keep, min(50, max(1, macro_k))))
            recent_scored = []
            for rec in recent:
                sim_m = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                if _cbo_record_identity(self, rec) in macro_pool_keys:
                    recent_scored.append((sim_m, rec))
            recent_scored.sort(key=lambda x: (x[0], int(x[1].get("bo_iter", -1) or -1)), reverse=True)
            macro_recent = [rec for _, rec in recent_scored[:macro_recent_keep]]
            if len(macro_recent) < min_keep:
                need = min_keep - len(macro_recent)
                existing = {_cbo_record_identity(self, r) for r in macro_recent}
                for rec in reversed(recent):
                    key = _cbo_record_identity(self, rec)
                    if key in existing:
                        continue
                    macro_recent.append(rec)
                    existing.add(key)
                    fallback_used = True
                    if not fallback_reason:
                        fallback_reason = "recent_keep_fallback"
                    if len(macro_recent) >= min_keep or len(macro_recent) >= macro_recent_keep + need:
                        break

            scored = []
            context_records = []
            if select_mode in {"recent_context", "recent_context_elite", "hybrid"}:
                for rec in macro_pool:
                    sim = _cbo_context_similarity(self, context, rec)
                    if threshold <= 0 or sim >= threshold:
                        scored.append((sim, rec))
                scored.sort(key=lambda x: x[0], reverse=True)
                context_records = [rec for _, rec in scored[:context_k]]

            clusters = _cbo_cluster_records(self, macro_pool, context=context)
            elite_clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
            elite_clusters.sort(key=lambda c: float(c.get("robust_score", float("inf"))))
            elite_records = []
            if select_mode in {"recent_context_elite", "hybrid"}:
                for c in elite_clusters[:elite_k]:
                    if c.get("records"):
                        elite_records.append(c["records"][0])

            diverse_records = []
            if select_mode == "hybrid" and diverse_k > 0:
                used_ids = {_cbo_record_identity(self, r) for r in (macro_recent + context_records + elite_records)}
                candidates = [r for r in macro_pool if _cbo_record_identity(self, r) not in used_ids]
                selected_norms = [_cbo_theta_norm(self, r.get("theta", [])) for r in macro_recent + context_records + elite_records if r.get("theta") is not None]
                while candidates and len(diverse_records) < diverse_k:
                    best_i, best_score = 0, -1.0
                    for idx, rec in enumerate(candidates):
                        tn = _cbo_theta_norm(self, rec.get("theta", []))
                        d_theta = min([float(np.linalg.norm(tn - s)) for s in selected_norms], default=1.0)
                        d_context = 1.0 - _cbo_context_similarity(self, context, rec)
                        score = d_theta + 0.25 * d_context
                        if score > best_score:
                            best_i, best_score = idx, score
                    rec = candidates.pop(best_i)
                    diverse_records.append(rec)
                    selected_norms.append(_cbo_theta_norm(self, rec.get("theta", [])))

            merged = []
            for priority, block in [(0, macro_recent), (1, context_records), (2, elite_records), (3, diverse_records), (4, fallback_records)]:
                for rec in block:
                    rec = dict(rec)
                    rec["_cbo_select_priority"] = priority
                    rec["_cbo_context_similarity"] = _cbo_context_similarity(self, context, rec)
                    rec["_cbo_macro_similarity"] = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                    merged.append(rec)
            dedup = {}
            eps = max(1e-9, float(getattr(self, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
            for rec in merged:
                tn = _cbo_theta_norm(self, rec.get("theta", []))
                cn = np.asarray(self._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(self, "use_context", False) else np.zeros(0)
                key = tuple(np.round(np.concatenate([tn / eps, cn / max(eps, 1e-9)]), 0).astype(int).tolist())
                old = dedup.get(key)
                if old is None:
                    dedup[key] = rec
                else:
                    old_score = (old.get("_cbo_select_priority", 9), -float(old.get("_cbo_macro_similarity", 0.0)), -float(old.get("_cbo_context_similarity", 0.0)), -int(old.get("bo_iter", -1) or -1))
                    new_score = (rec.get("_cbo_select_priority", 9), -float(rec.get("_cbo_macro_similarity", 0.0)), -float(rec.get("_cbo_context_similarity", 0.0)), -int(rec.get("bo_iter", -1) or -1))
                    if new_score < old_score:
                        dedup[key] = rec
            pool = list(dedup.values())
            max_train = macro_recent_keep + (context_k if select_mode in {"recent_context", "recent_context_elite", "hybrid"} else 0) + (elite_k if select_mode in {"recent_context_elite", "hybrid"} else 0) + (diverse_k if select_mode == "hybrid" else 0)
            pool.sort(key=lambda r: (int(r.get("_cbo_select_priority", 9)), -float(r.get("_cbo_macro_similarity", 0.0)), -float(r.get("_cbo_context_similarity", 0.0)), -int(r.get("bo_iter", -1) or -1)))
            pool = pool[:max(2, max_train)]
            pool_keys = {_cbo_record_identity(self, r) for r in pool}
            selected_macro_sims = [float(r.get("_cbo_macro_similarity", np.nan)) for r in pool if _cbo_record_identity(self, r) in macro_pool_keys and np.isfinite(float(r.get("_cbo_macro_similarity", np.nan)))]
            outside_count = sum(1 for r in pool if _cbo_record_identity(self, r) not in macro_pool_keys)
            best_elite = elite_clusters[0] if elite_clusters else {}
            set_debug(
                pool,
                recent_count=len(macro_recent),
                macro_count=len(selected_macro_sims),
                context_count=len(context_records),
                elite_count=len(elite_records),
                diverse_count=len(diverse_records),
                sims=[s for s, _ in scored],
                elite=best_elite,
                macro_sims=macro_sims,
                macro_pool_sims=macro_pool_sims,
                selected_macro_sims=selected_macro_sims,
                selected_from_macro_pool_count=sum(1 for k in pool_keys if k in macro_pool_keys),
                selected_outside_macro_pool_count=outside_count,
                macro_gate_fallback_used=fallback_used or outside_count > 0,
                macro_gate_fallback_reason=fallback_reason,
                context_selection_source_pool="macro_pool",
                elite_selection_source_pool="macro_pool",
                tr_anchor_source_pool="macro_pool",
                macro_current=macro_current,
            )
            return list(pool)

        self._last_macro_pool_records = []
        self._last_macro_pool_keys = set()
        scored = []
        context_records = []
        if select_mode in {"recent_context", "recent_context_elite", "hybrid"}:
            for rec in all_records:
                sim = _cbo_context_similarity(self, context, rec)
                if threshold <= 0 or sim >= threshold:
                    scored.append((sim, rec))
            scored.sort(key=lambda x: x[0], reverse=True)
            context_records = [rec for _, rec in scored[:context_k]]
        clusters = _cbo_cluster_records(self, all_records, context=context)
        elite_clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
        elite_clusters.sort(key=lambda c: float(c.get("robust_score", float("inf"))))
        elite_records = []
        if select_mode in {"recent_context_elite", "hybrid"}:
            for c in elite_clusters[:elite_k]:
                if c.get("records"):
                    elite_records.append(c["records"][0])
        diverse_records = []
        if select_mode == "hybrid" and diverse_k > 0:
            used_ids = {id(r) for r in (recent + context_records + elite_records)}
            candidates = [r for r in all_records if id(r) not in used_ids]
            selected_norms = [_cbo_theta_norm(self, r.get("theta", [])) for r in recent + context_records + elite_records if r.get("theta") is not None]
            while candidates and len(diverse_records) < diverse_k:
                best_i, best_score = 0, -1.0
                for idx, rec in enumerate(candidates):
                    tn = _cbo_theta_norm(self, rec.get("theta", []))
                    d_theta = min([float(np.linalg.norm(tn - s)) for s in selected_norms], default=1.0)
                    d_context = 1.0 - _cbo_context_similarity(self, context, rec)
                    score = d_theta + 0.25 * d_context
                    if score > best_score:
                        best_i, best_score = idx, score
                rec = candidates.pop(best_i)
                diverse_records.append(rec)
                selected_norms.append(_cbo_theta_norm(self, rec.get("theta", [])))
        merged = []
        for priority, block in [(0, recent), (1, macro_records), (2, context_records), (3, elite_records), (4, diverse_records)]:
            for rec in block:
                rec = dict(rec)
                rec["_cbo_select_priority"] = priority
                rec["_cbo_context_similarity"] = _cbo_context_similarity(self, context, rec)
                if macro_mode != "off":
                    rec["_cbo_macro_similarity"] = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                merged.append(rec)
        dedup = {}
        eps = max(1e-9, float(getattr(self, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
        for rec in merged:
            tn = _cbo_theta_norm(self, rec.get("theta", []))
            cn = np.asarray(self._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(self, "use_context", False) else np.zeros(0)
            key = tuple(np.round(np.concatenate([tn / eps, cn / max(eps, 1e-9)]), 0).astype(int).tolist())
            old = dedup.get(key)
            if old is None:
                dedup[key] = rec
            else:
                old_score = (old.get("_cbo_select_priority", 9), -float(old.get("_cbo_context_similarity", 0.0)), -int(old.get("bo_iter", -1) or -1))
                new_score = (rec.get("_cbo_select_priority", 9), -float(rec.get("_cbo_context_similarity", 0.0)), -int(rec.get("bo_iter", -1) or -1))
                if new_score < old_score:
                    dedup[key] = rec
        pool = list(dedup.values())
        max_train = recent_window + (macro_k if macro_mode != "off" else 0) + (context_k if select_mode in {"recent_context", "recent_context_elite", "hybrid"} else 0) + (elite_k if select_mode in {"recent_context_elite", "hybrid"} else 0) + (diverse_k if select_mode == "hybrid" else 0)
        pool.sort(key=lambda r: (int(r.get("_cbo_select_priority", 9)), -float(r.get("_cbo_context_similarity", 0.0)), -int(r.get("bo_iter", -1) or -1)))
        pool = pool[:max(2, max_train)]
        best_elite = elite_clusters[0] if elite_clusters else {}
        set_debug(
            pool,
            recent_count=len(recent),
            macro_count=len(macro_records),
            context_count=len(context_records),
            elite_count=len(elite_records),
            diverse_count=len(diverse_records),
            sims=[s for s, _ in scored],
            elite=best_elite,
            macro_sims=macro_sims,
            macro_pool_sims=macro_pool_sims,
            selected_from_macro_pool_count=len(macro_records),
            selected_outside_macro_pool_count=0,
            context_selection_source_pool="all_records",
            elite_selection_source_pool="all_records",
            tr_anchor_source_pool="all_records",
            macro_current=macro_current,
        )
        return list(pool)

    pool = records
    if mode in {"recent", "recent_only", "recent_confidence", "confidence_recent", "decay"}:
        pool = pool[-recent_window:]

    if mode in {"confidence", "conf", "recent_confidence", "confidence_recent"}:
        keep = [r for r in pool if float(r.get("feedback_confidence", 1.0)) >= min_conf]
        if len(keep) >= min_keep:
            pool = keep
        else:
            # 样本太少时不能硬过滤，否则 BO 直接失明。保留置信度最高的一批 + 最近样本兜底。
            ranked = sorted(pool, key=lambda r: float(r.get("feedback_confidence", 1.0)), reverse=True)
            top = ranked[:min(len(ranked), min_keep)]
            tail = pool[-min_keep:]
            merged = []
            seen = set()
            for r in top + tail:
                key = id(r)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            pool = merged

    set_debug(pool)
    return list(pool)


FederatedBOAgent._collect_samples = _refactor_collect_samples


_ORIG_AGENT_TELL_STABILITY = FederatedBOAgent.tell
_ORIG_AGENT_ASK_CONTEXTUAL_STABILITY = FederatedBOAgent._ask_contextual


def _cbo_tr_update_params(agent):
    return {
        "mode": str(getattr(agent, "cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far")) or "best_so_far").strip().lower(),
        "compare_window": max(1, int(getattr(agent, "cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)))),
        "baseline_window": max(1, int(getattr(agent, "cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)))),
        "improve_pct": float(getattr(agent, "cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015))),
        "worsen_pct": float(getattr(agent, "cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03))),
        "deadband_pct": float(getattr(agent, "cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01))),
        "patience": max(1, int(getattr(agent, "cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)))),
        "r_min": float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04)))),
        "r_max": float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35)))),
        "grow": float(getattr(agent, "cbo_tr_grow", _cfg_cbo_float("CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15)))),
        "shrink": float(getattr(agent, "cbo_tr_shrink", _cfg_cbo_float("CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92)))),
    }


def _cbo_ewma(values, alpha=None):
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return np.nan
    if alpha is None:
        alpha = 2.0 / (len(vals) + 1.0)
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    cur = vals[0]
    for v in vals[1:]:
        cur = alpha * float(v) + (1.0 - alpha) * cur
    return float(cur)


def _cbo_apply_tr_signal(agent, params, signal, reason, radius_before):
    mode = str(params.get("mode", "best_so_far"))
    patience = max(1, int(params.get("patience", 1)))
    actionable = signal in {"improve", "worse"}
    prev_signal = str(getattr(agent, "cbo_tr_update_last_signal", "") or "")
    if actionable:
        count = int(getattr(agent, "cbo_tr_update_patience_count", 0))
        count = count + 1 if prev_signal == signal else 1
    else:
        count = 0
    agent.cbo_tr_update_last_signal = signal
    agent.cbo_tr_update_patience_count = int(count)

    radius_after = float(radius_before)
    applied = False
    final_reason = reason
    if actionable and count >= patience:
        if signal == "improve":
            radius_after = max(float(params["r_min"]), float(radius_before) * float(params["shrink"]))
            agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0)) + 1
        else:
            radius_after = min(float(params["r_max"]), float(radius_before) * float(params["grow"]))
            agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0)) + 1
        agent.trust_radius = float(radius_after)
        agent.cbo_tr_update_patience_count = 0
        applied = True
    elif actionable:
        final_reason = f"{reason}_pending_patience"

    return {
        "tr_update_mode": mode,
        "tr_update_signal": signal,
        "tr_update_patience_count": int(getattr(agent, "cbo_tr_update_patience_count", count)),
        "cbo_tr_update_reason": final_reason,
        "cbo_tr_radius_before_update": float(radius_before),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", radius_after)),
        "tr_update_applied": int(applied),
    }


def _cbo_update_tr_radius_after_tell(agent, cost, prev_best_value=None, radius_before=None):
    params = _cbo_tr_update_params(agent)
    mode = params["mode"]
    if mode not in {"best_so_far", "rolling_mean", "ewma_trend"}:
        mode = "best_so_far"
        params["mode"] = mode
    if radius_before is None or not np.isfinite(float(radius_before)):
        radius_before = float(getattr(agent, "trust_radius", np.nan))
    radius_before = float(radius_before)
    agent.trust_radius = float(np.clip(float(getattr(agent, "trust_radius", radius_before)), params["r_min"], params["r_max"]))

    hist = list(getattr(agent, "cbo_tr_cost_history", []))
    if np.isfinite(float(cost)):
        hist.append(float(cost))
    max_len = max(200, (params["compare_window"] + params["baseline_window"]) * 4)
    agent.cbo_tr_cost_history = hist[-max_len:]

    info = {
        "tr_update_mode": mode,
        "tr_baseline_mean": np.nan,
        "tr_current_mean": np.nan,
        "tr_improve_pct": np.nan,
        "tr_worse_pct": np.nan,
        "tr_update_signal": "none",
        "tr_update_patience_count": int(getattr(agent, "cbo_tr_update_patience_count", 0)),
        "cbo_tr_update_reason": "tr_update_not_run",
        "cbo_tr_radius_before_update": float(radius_before),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", radius_before)),
    }

    if mode == "best_so_far":
        y_val = -float(cost)
        improved = prev_best_value is None or (np.isfinite(float(prev_best_value)) and y_val >= float(prev_best_value))
        if improved:
            agent.trust_radius = min(params["r_max"], radius_before * params["grow"])
            agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0)) + 1
            signal = "best_so_far_improve"
            reason = "best_so_far_improved_grow"
        else:
            agent.trust_radius = max(params["r_min"], radius_before * params["shrink"])
            agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0)) + 1
            signal = "best_so_far_worse"
            reason = "best_so_far_worse_shrink"
        info.update({
            "tr_update_signal": signal,
            "tr_update_patience_count": 0,
            "cbo_tr_update_reason": reason,
            "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
        })
    else:
        need = params["compare_window"] + params["baseline_window"]
        if len(hist) < need:
            info.update({
                "tr_update_signal": "insufficient_history",
                "cbo_tr_update_reason": f"{mode}_insufficient_history",
                "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
            })
        else:
            baseline = hist[-need:-params["compare_window"]]
            current = hist[-params["compare_window"]:]
            if mode == "ewma_trend":
                alpha = 2.0 / (params["compare_window"] + 1.0)
                baseline_mean = _cbo_ewma(baseline, alpha=alpha)
                current_mean = _cbo_ewma(current, alpha=alpha)
            else:
                baseline_mean = float(np.mean(baseline))
                current_mean = float(np.mean(current))
            if np.isfinite(baseline_mean) and abs(float(baseline_mean)) > 1e-12 and np.isfinite(current_mean):
                improve_pct = (baseline_mean - current_mean) / abs(baseline_mean)
                worse_pct = (current_mean - baseline_mean) / abs(baseline_mean)
            else:
                improve_pct = np.nan
                worse_pct = np.nan
            info.update({
                "tr_baseline_mean": float(baseline_mean) if np.isfinite(baseline_mean) else np.nan,
                "tr_current_mean": float(current_mean) if np.isfinite(current_mean) else np.nan,
                "tr_improve_pct": float(improve_pct) if np.isfinite(improve_pct) else np.nan,
                "tr_worse_pct": float(worse_pct) if np.isfinite(worse_pct) else np.nan,
            })
            if not np.isfinite(improve_pct) or not np.isfinite(worse_pct):
                signal, reason = "invalid_trend", f"{mode}_invalid_trend"
            elif improve_pct > params["improve_pct"]:
                signal, reason = "improve", f"{mode}_improved_refine"
            elif worse_pct > params["worsen_pct"]:
                signal, reason = "worse", f"{mode}_worse_expand"
            elif abs(current_mean - baseline_mean) / max(1e-12, abs(baseline_mean)) <= params["deadband_pct"]:
                signal, reason = "deadband", f"{mode}_hold_deadband"
            else:
                signal, reason = "hold", f"{mode}_hold_between_thresholds"
            info.update(_cbo_apply_tr_signal(agent, params, signal, reason, radius_before))
            info.update({
                "tr_baseline_mean": float(baseline_mean) if np.isfinite(baseline_mean) else np.nan,
                "tr_current_mean": float(current_mean) if np.isfinite(current_mean) else np.nan,
                "tr_improve_pct": float(improve_pct) if np.isfinite(improve_pct) else np.nan,
                "tr_worse_pct": float(worse_pct) if np.isfinite(worse_pct) else np.nan,
            })

    agent.cbo_tr_update_reason = str(info.get("cbo_tr_update_reason", "tr_update_unknown"))
    debug = dict(getattr(agent, "last_debug_info", {}) or {})
    debug.update(info)
    agent.last_debug_info = debug
    return info


def _stability_tell(self, theta, cost, state=None, context=None):
    prev_best_value = getattr(self, "prev_best_value", None)
    tr_mode = str(getattr(self, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    managed_tr = tr_mode in {"adaptive", "residual_adaptive", "condition_adaptive"}
    radius_before = float(getattr(self, "trust_radius", np.nan))
    old_use_trust_region = bool(getattr(self, "use_trust_region", False))
    if managed_tr:
        self.use_trust_region = False
    try:
        _ORIG_AGENT_TELL_STABILITY(self, theta, cost, state=state, context=context)
    finally:
        if managed_tr:
            self.use_trust_region = old_use_trust_region
    if tr_mode in {"adaptive", "residual_adaptive", "condition_adaptive"}:
        _cbo_update_tr_radius_after_tell(self, float(cost), prev_best_value=prev_best_value, radius_before=radius_before)
        if tr_mode in {"residual_adaptive", "condition_adaptive"}:
            _cbo_update_residual_condition_state(self, float(cost))
    elif tr_mode == "good_region":
        self.cbo_tr_update_reason = "good_region_fixed_radius"


FederatedBOAgent.tell = _stability_tell


def _cbo_choose_tr_anchor(agent, mode, context, records, candidate_scores=None):
    low = np.asarray(agent.bounds[0].tolist(), dtype=float)
    high = np.asarray(agent.bounds[1].tolist(), dtype=float)
    mode = str(mode or "posterior_mean").lower()
    macro_mode = str(getattr(agent, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    macro_records = list(getattr(agent, "_last_macro_pool_records", []) or [])
    anchor_records = macro_records if macro_mode == "hierarchical" and macro_records else list(records or [])
    source_pool = "macro_pool" if macro_mode == "hierarchical" and macro_records else "all_records"
    if mode == "robust_elite":
        theta, info = agent._compute_robust_incumbent(context=context)
        if theta is not None:
            info = dict(info or {})
            info.setdefault("tr_anchor_source_pool", source_pool)
            return list(np.clip(np.asarray(theta, dtype=float), low, high)), "robust_elite", info
    if mode == "recent_best":
        if source_pool == "macro_pool":
            recent = sorted([r for r in anchor_records if r.get("theta") is not None], key=lambda r: int(r.get("bo_iter", -1) or -1))
        else:
            recent = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
        if recent:
            best = max(recent, key=lambda r: float(r.get("y", -1e300)))
            return list(best.get("theta", [])), "recent_best", {"tr_anchor_source_pool": source_pool}
    if mode == "context_best":
        scored = [(_cbo_context_similarity(agent, context, r), r) for r in anchor_records if r.get("theta") is not None]
        if scored:
            scored.sort(key=lambda x: (x[0], float(x[1].get("y", -1e300))), reverse=True)
            return list(scored[0][1].get("theta", [])), "context_best", {"tr_anchor_source_pool": source_pool}
    if candidate_scores is not None and len(candidate_scores) > 0:
        try:
            idx = int(np.argmax(candidate_scores))
            return None, "posterior_mean", {"posterior_anchor_idx": idx, "tr_anchor_source_pool": source_pool}
        except Exception:
            pass
    pivot = agent._select_pivot_theta(context, anchor_records)
    if pivot is not None:
        return list(pivot), "posterior_mean_pivot", {"tr_anchor_source_pool": source_pool}
    if getattr(agent, "prev_best", None) is not None:
        return list(agent.prev_best), "posterior_mean_prev_best_fallback", {"tr_anchor_source_pool": source_pool}
    return None, "no_anchor", {"tr_anchor_source_pool": source_pool}


def _cbo_theta_distance(agent, a, b):
    try:
        if a is None or b is None:
            return np.nan
        return float(np.linalg.norm(_cbo_theta_norm(agent, a) - _cbo_theta_norm(agent, b)))
    except Exception:
        return np.nan


def _cbo_anchor_for_distance(agent, mode, context, records):
    try:
        theta, source, _ = _cbo_choose_tr_anchor(agent, mode, context, records)
        return theta if str(source) == str(mode) else None
    except Exception:
        return None


def _cbo_resolve_actual_tr_anchor(agent, context, records):
    configured_mode = str(getattr(agent, "cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")) or "posterior_mean").strip().lower()
    override_mode = getattr(agent, "cbo_runtime_anchor_override", None)
    override_mode = None if _is_missing_value(override_mode) else str(override_mode).strip().lower()
    effective_mode = str(override_mode or configured_mode or "posterior_mean").strip().lower()
    anchor, anchor_source, anchor_info = _cbo_choose_tr_anchor(agent, effective_mode, context, records)
    anchor_info = dict(anchor_info or {})
    fallback_used = bool(anchor is None or (effective_mode in {"robust_elite", "context_best", "recent_best"} and str(anchor_source) != effective_mode))
    fallback_reason = "" if not fallback_used else f"{effective_mode}_unavailable_used_{anchor_source}"
    override_used = bool(override_mode)
    reason_parts = []
    if override_used:
        reason_parts.append(f"runtime_override:{override_mode}")
    if fallback_used:
        reason_parts.append(f"fallback:{fallback_reason}")
    if not reason_parts:
        reason_parts.append("configured_anchor")

    prev_theta = getattr(agent, "last_theta", None)
    robust_theta = _cbo_anchor_for_distance(agent, "robust_elite", context, records)
    context_theta = _cbo_anchor_for_distance(agent, "context_best", context, records)
    recent_theta = _cbo_anchor_for_distance(agent, "recent_best", context, records)
    debug = {
        "configured_tr_anchor_mode": configured_mode,
        "effective_tr_anchor_mode": effective_mode,
        "actual_tr_anchor_mode": str(anchor_source),
        "actual_tr_anchor_source": str(anchor_source),
        "actual_tr_anchor_theta": list(anchor) if anchor is not None else None,
        "actual_tr_anchor_reason": ";".join(reason_parts),
        "anchor_override_used": int(override_used),
        "anchor_override_reason": f"runtime_anchor_override={override_mode}" if override_used else "",
        "anchor_fallback_used": int(fallback_used),
        "anchor_fallback_reason": fallback_reason,
        "anchor_theta_distance_to_prev": _cbo_theta_distance(agent, anchor, prev_theta),
        "anchor_theta_distance_to_robust_elite": _cbo_theta_distance(agent, anchor, robust_theta),
        "anchor_theta_distance_to_context_best": _cbo_theta_distance(agent, anchor, context_theta),
        "anchor_theta_distance_to_recent_best": _cbo_theta_distance(agent, anchor, recent_theta),
        "runtime_anchor_override": override_mode,
        "runtime_anchor_override_reason": f"runtime_anchor_override={override_mode}" if override_used else "",
        "tr_anchor_source_pool": str(anchor_info.get("tr_anchor_source_pool", "all_records")),
    }
    try:
        agent.cbo_last_actual_anchor_debug = dict(debug)
    except Exception:
        pass
    return (list(anchor) if anchor is not None else None), debug


def _cbo_radius_norm(agent):
    try:
        r = float(getattr(agent, "trust_radius", np.nan))
        r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
        r_max = float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
        if not np.isfinite(r) or not np.isfinite(r_min) or not np.isfinite(r_max) or abs(r_max - r_min) <= 1e-12:
            return np.nan
        return float(np.clip((r - r_min) / (r_max - r_min), 0.0, 1.0))
    except Exception:
        return np.nan


def _cbo_beta_eff_info(agent):
    mode = str(getattr(agent, "cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")) or "fixed").strip().lower()
    base_beta = max(0.0, float(getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0))))
    beta_min = max(0.0, float(getattr(agent, "cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1))))
    beta_max = max(beta_min, float(getattr(agent, "cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0))))
    power = max(1e-12, float(getattr(agent, "cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0))))
    radius_norm = _cbo_radius_norm(agent)
    if mode == "fixed":
        beta_eff = base_beta
        radius_component = base_beta
        formula = "posterior_mu + cbo_acq_beta * posterior_sigma"
    else:
        rn = 0.0 if not np.isfinite(radius_norm) else float(radius_norm)
        radius_component = float(beta_min + (beta_max - beta_min) * (rn ** power))
        beta_eff = radius_component
        formula = "posterior_mu + beta_eff * posterior_sigma"

    boost_used = False
    boost_reason = "none"
    if mode == "radius_state_adaptive":
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        reasons = []
        unstable_rebound = float(getattr(agent, "cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)))
        surprise_thr = float(getattr(agent, "cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)))
        cost_gap = _safe_float(debug.get("cost_gap_pct"), 0.0)
        surprise = _safe_float(debug.get("surprise"), np.nan)
        prediction_error = _safe_float(debug.get("prediction_error"), np.nan)
        predicted_cost = _safe_float(debug.get("predicted_cost"), np.nan)
        if np.isfinite(cost_gap) and cost_gap >= unstable_rebound:
            reasons.append("rebound")
        if np.isfinite(surprise) and surprise >= surprise_thr:
            reasons.append("surprise")
        if bool(int(debug.get("residual_trigger", 0) or 0)):
            reasons.append("residual_trigger")
        if bool(int(debug.get("condition_trigger", 0) or 0)):
            reasons.append("condition_trigger")
        stable_gap = float(getattr(agent, "cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)))
        if np.isfinite(prediction_error) and np.isfinite(predicted_cost) and prediction_error > stable_gap * max(1.0, abs(predicted_cost)):
            reasons.append("actual_gt_predicted")
        if reasons:
            boost = max(1.0, float(getattr(agent, "cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5))))
            cap = max(0.0, float(getattr(agent, "cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0))))
            beta_eff = min(float(beta_eff) * boost, cap)
            boost_used = True
            boost_reason = "+".join(reasons)
    info = {
        "cbo_acq_beta_mode": mode,
        "beta_eff": float(beta_eff),
        "radius_norm": float(radius_norm) if np.isfinite(radius_norm) else np.nan,
        "radius_beta_component": float(radius_component),
        "state_beta_boost_used": int(boost_used),
        "state_beta_boost_reason": boost_reason,
        "actual_score_formula": formula,
    }
    try:
        agent.cbo_last_beta_eff = float(beta_eff)
        agent.cbo_last_beta_info = dict(info)
        agent.cbo_last_actual_beta_used = float(beta_eff)
    except Exception:
        pass
    return info


def _cbo_service_guard_apply(agent, score):
    mode = str(getattr(agent, "cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off")) or "off").strip().lower()
    score = np.asarray(score, dtype=float)
    penalty = np.zeros_like(score, dtype=float)
    reason = "off" if mode == "off" else "not_available"
    info = {
        "service_guard_mode": mode,
        "service_guard_available": False,
        "service_guard_penalty": 0.0,
        "service_guard_reason": reason,
    }
    return score - penalty, penalty, info


def _cbo_good_region_recent_mean(agent, window):
    costs = []
    for v in list(getattr(agent, "cbo_eval_cost_history", []) or []):
        fv = _safe_float(v, np.nan)
        if np.isfinite(fv):
            costs.append(float(fv))
    if len(costs) < max(1, int(window)):
        return np.nan
    return float(np.mean(costs[-max(1, int(window)):]))


def _cbo_good_region_guard_enabled(agent, group_cfg):
    if agent is None or not isinstance(group_cfg, dict):
        return False
    if str(group_cfg.get("control_mode", "")).strip().lower() != "alpha_direct":
        return False
    val = group_cfg.get("cbo_good_region_guard", getattr(agent, "cbo_good_region_guard", _cfg_cbo_str("CBO_GOOD_REGION_GUARD", "off")))
    return str(val or "off").strip().lower() in {"on", "true", "1", "yes", "enabled"}


def _alpha_direct_expanded_6d(theta, group_cfg=None):
    expand_fn = globals().get("expand_alpha_direct_control_vector")
    if callable(expand_fn):
        try:
            return list(expand_fn(theta, group_cfg=group_cfg))
        except TypeError:
            return list(expand_fn(theta))
        except Exception:
            pass
    clip_fn = globals().get("clip_alpha_direct_control_vector")
    return list(clip_fn(theta)) if callable(clip_fn) else list(theta)


def _cbo_apply_good_region_deployment_guard(agent, candidate_theta, safe_info, group_cfg):
    """Optionally deploy the best rolling good-region theta instead of a risky alpha_direct candidate."""
    candidate_theta = list(candidate_theta) if candidate_theta is not None else None
    enabled = _cbo_good_region_guard_enabled(agent, group_cfg)
    window = max(1, int(group_cfg.get("cbo_good_region_window", getattr(agent, "cbo_good_region_window", _cfg_cbo_int("CBO_GOOD_REGION_WINDOW", 50)))))
    mode = str(group_cfg.get("cbo_good_region_guard_mode", getattr(agent, "cbo_good_region_guard_mode", _cfg_cbo_str("CBO_GOOD_REGION_GUARD_MODE", "conservative"))) or "conservative").strip().lower()
    if mode not in {"conservative", "distance_only", "performance_only"}:
        mode = "conservative"
    good_theta = getattr(agent, "good_region_anchor_theta", None)
    good_cost = getattr(agent, "good_region_best_rolling50_cost", None)
    available = bool(good_theta is not None and good_cost is not None and np.isfinite(float(good_cost)))
    distance = _cbo_theta_distance(agent, candidate_theta, good_theta) if available else np.nan
    recent_mean = _cbo_good_region_recent_mean(agent, window)
    worse_pct = np.nan
    if available and np.isfinite(recent_mean) and abs(float(good_cost)) > 1e-12:
        worse_pct = (float(recent_mean) - float(good_cost)) / abs(float(good_cost))
    tr_radius = safe_info.get("cbo_tr_radius_after_update", safe_info.get("cbo_tr_radius", getattr(agent, "trust_radius", np.nan)))
    beta_eff = safe_info.get("beta_eff", safe_info.get("selected_candidate_beta_eff", getattr(agent, "cbo_last_beta_eff", np.nan)))
    tr_radius_f = _safe_float(tr_radius, np.nan)
    beta_eff_f = _safe_float(beta_eff, np.nan)
    selected_source = str(safe_info.get("selected_candidate_source", safe_info.get("selected_source", "")) or "")
    distance_threshold = float(group_cfg.get("cbo_good_region_distance_threshold", getattr(agent, "cbo_good_region_distance_threshold", _cfg_cbo_float("CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35))))
    worse_threshold = float(group_cfg.get("cbo_good_region_worse_pct", getattr(agent, "cbo_good_region_worse_pct", _cfg_cbo_float("CBO_GOOD_REGION_WORSE_PCT", 0.03))))
    tr_threshold = float(group_cfg.get("cbo_good_region_tr_radius_threshold", getattr(agent, "cbo_good_region_tr_radius_threshold", _cfg_cbo_float("CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15))))
    beta_threshold = float(group_cfg.get("cbo_good_region_beta_threshold", getattr(agent, "cbo_good_region_beta_threshold", _cfg_cbo_float("CBO_GOOD_REGION_BETA_THRESHOLD", 0.5))))

    checks = []
    if available and np.isfinite(distance) and distance > distance_threshold:
        checks.append("distance")
    if available and np.isfinite(worse_pct) and worse_pct > worse_threshold:
        checks.append("rolling_worse")
    if np.isfinite(tr_radius_f) and tr_radius_f > tr_threshold:
        checks.append("tr_radius")
    if np.isfinite(beta_eff_f) and beta_eff_f > beta_threshold:
        checks.append("beta_eff")
    if available and selected_source == "global_random" and np.isfinite(distance) and distance > distance_threshold:
        checks.append("global_random_far")

    if mode == "distance_only":
        active_checks = [c for c in checks if c in {"distance", "global_random_far"}]
    elif mode == "performance_only":
        active_checks = [c for c in checks if c in {"rolling_worse", "tr_radius", "beta_eff"}]
    else:
        active_checks = checks

    triggered = bool(enabled and available and active_checks)
    deployed_theta = list(good_theta) if triggered else list(candidate_theta)
    candidate_theta_6d = _alpha_direct_expanded_6d(candidate_theta, group_cfg) if candidate_theta is not None else None
    deployed_theta_6d = _alpha_direct_expanded_6d(deployed_theta, group_cfg) if deployed_theta is not None else None
    good_theta_6d = _alpha_direct_expanded_6d(good_theta, group_cfg) if good_theta is not None else None
    guard_info = {
        "good_region_guard_enabled": int(enabled),
        "good_region_guard_triggered": int(triggered),
        "good_region_guard_reason": "+".join(active_checks) if triggered else ("no_good_region" if enabled and not available else "pass"),
        "candidate_theta_6d": candidate_theta_6d,
        "deployed_theta_6d": deployed_theta_6d,
        "good_region_theta_6d": good_theta_6d,
        "good_region_cost": float(good_cost) if available else np.nan,
        "good_region_iter": getattr(agent, "good_region_best_iter", None),
        "good_region_window_start": getattr(agent, "good_region_window_start", None),
        "good_region_window_end": getattr(agent, "good_region_window_end", None),
        "distance_to_good_region": float(distance) if np.isfinite(distance) else np.nan,
        "candidate_selected_source": selected_source or None,
        "deployed_source": "good_region_guard" if triggered else (selected_source or safe_info.get("deploy_source")),
        "candidate_tr_radius": float(tr_radius_f) if np.isfinite(tr_radius_f) else np.nan,
        "candidate_beta_eff": float(beta_eff_f) if np.isfinite(beta_eff_f) else np.nan,
        "guard_fallback_type": "good_region" if triggered else "none",
    }
    return deployed_theta, guard_info


def _cbo_update_good_region_memory(agent, iteration, theta, eval_cost, safe_info):
    if agent is None:
        return safe_info
    window = max(1, int(getattr(agent, "cbo_good_region_window", _cfg_cbo_int("CBO_GOOD_REGION_WINDOW", 50))))
    costs = list(getattr(agent, "cbo_eval_cost_history", []) or [])
    try:
        costs.append(float(eval_cost))
    except Exception:
        costs.append(np.nan)
    agent.cbo_eval_cost_history = costs
    rolling = np.nan
    if len(costs) >= window:
        recent = np.asarray(costs[-window:], dtype=float)
        if np.isfinite(recent).all():
            rolling = float(np.mean(recent))
            best = getattr(agent, "good_region_best_rolling50_cost", None)
            if best is None or not np.isfinite(float(best)) or rolling < float(best):
                agent.good_region_best_iter = int(iteration)
                agent.good_region_best_rolling50_cost = float(rolling)
                agent.good_region_anchor_theta = list(theta) if theta is not None else None
                agent.good_region_anchor_source = str(safe_info.get("deployed_source", safe_info.get("selected_candidate_source", safe_info.get("deploy_source", "selected_theta"))))
                agent.good_region_window_start = int(iteration) - window + 1
                agent.good_region_window_end = int(iteration)
    best_cost = getattr(agent, "good_region_best_rolling50_cost", None)
    anchor = getattr(agent, "good_region_anchor_theta", None)
    available = bool(anchor is not None and best_cost is not None and np.isfinite(float(best_cost)))
    gap = np.nan
    if available and np.isfinite(float(best_cost)) and abs(float(best_cost)) > 1e-12:
        try:
            gap = (float(eval_cost) - float(best_cost)) / abs(float(best_cost))
        except Exception:
            gap = np.nan
    safe_info.update({
        "good_region_available": int(available),
        "good_region_best_iter": getattr(agent, "good_region_best_iter", None),
        "good_region_best_rolling50_cost": float(best_cost) if available else np.nan,
        "good_region_anchor_theta": list(anchor) if anchor is not None else None,
        "good_region_anchor_source": getattr(agent, "good_region_anchor_source", None),
        "good_region_cost": float(best_cost) if available else np.nan,
        "good_region_theta_6d": list(anchor) if anchor is not None else None,
        "good_region_iter": getattr(agent, "good_region_best_iter", None),
        "good_region_window_start": getattr(agent, "good_region_window_start", None),
        "good_region_window_end": getattr(agent, "good_region_window_end", None),
        "distance_to_good_region_anchor": _cbo_theta_distance(agent, theta, anchor),
        "distance_to_good_region": _cbo_theta_distance(agent, theta, anchor),
        "current_vs_good_region_gap_pct": float(gap) if np.isfinite(gap) else np.nan,
    })
    return safe_info


def _cbo_rank_ascending(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks


def _cbo_candidate_rows(agent, candidates, sources, mu, sigma, score, selected_idx, selected_reason, deploy_policy=None, deploy_source=None, anchor=None, robust_theta=None, recent_best=None, beta_eff=None, service_penalty=None):
    rows = []
    if candidates is None:
        return rows, {}
    mu = np.asarray(mu if mu is not None else [np.nan] * len(candidates), dtype=float)
    sigma = np.asarray(sigma if sigma is not None else [np.nan] * len(candidates), dtype=float)
    score = np.asarray(score if score is not None else [np.nan] * len(candidates), dtype=float)
    service_penalty = np.asarray(service_penalty if service_penalty is not None else [0.0] * len(candidates), dtype=float)
    ranks_mu = _cbo_rank_ascending(-mu) if len(mu) else []
    ranks_sigma = _cbo_rank_ascending(-sigma) if len(sigma) else []
    ranks_score = _cbo_rank_ascending(-score) if len(score) else []
    ranks_acq = ranks_score
    best_mu_idx = int(np.nanargmax(mu)) if len(mu) and np.isfinite(mu).any() else None
    best_acq_idx = int(np.nanargmax(score)) if len(score) and np.isfinite(score).any() else None
    names = list(getattr(CFG, "FEATURE_NAMES", []))
    selected_idx = int(selected_idx) if selected_idx is not None else -1
    anchor_debug = dict(getattr(agent, "cbo_last_actual_anchor_debug", {}) or {})
    beta_info = dict(getattr(agent, "cbo_last_beta_info", {}) or {})
    beta_value = beta_eff if beta_eff is not None else beta_info.get("beta_eff")
    for i, theta in enumerate(candidates):
        row = {
            "candidate_id": int(i),
            "candidate_source": str(sources[i] if i < len(sources) else "unknown"),
            "theta": _safe_json(list(theta)),
            "control_vector": _safe_json(list(theta)),
            "posterior_mu": float(mu[i]) if i < len(mu) and np.isfinite(mu[i]) else np.nan,
            "posterior_sigma": float(sigma[i]) if i < len(sigma) and np.isfinite(sigma[i]) else np.nan,
            "acquisition_score": float(score[i]) if i < len(score) and np.isfinite(score[i]) else np.nan,
            "beta_eff": float(beta_value) if beta_value is not None and np.isfinite(float(beta_value)) else np.nan,
            "score": float(score[i]) if i < len(score) and np.isfinite(score[i]) else np.nan,
            "rank_by_mu": int(ranks_mu[i]) if len(ranks_mu) else None,
            "rank_by_sigma": int(ranks_sigma[i]) if len(ranks_sigma) else None,
            "rank_by_score": int(ranks_score[i]) if len(ranks_score) else None,
            "rank_by_acq": int(ranks_acq[i]) if len(ranks_acq) else None,
            "is_selected": int(i == selected_idx),
            "selected_reason": str(selected_reason),
            "deploy_policy": deploy_policy,
            "deploy_source": deploy_source,
            "service_guard_available": beta_info.get("service_guard_available", False),
            "service_guard_penalty": float(service_penalty[i]) if i < len(service_penalty) and np.isfinite(service_penalty[i]) else 0.0,
            "service_guard_reason": beta_info.get("service_guard_reason", "off"),
            "actual_tr_anchor_mode": anchor_debug.get("actual_tr_anchor_mode"),
            "actual_tr_anchor_source": anchor_debug.get("actual_tr_anchor_source"),
            "actual_tr_anchor_theta": _safe_json(anchor_debug.get("actual_tr_anchor_theta")),
            "anchor_override_used": anchor_debug.get("anchor_override_used"),
            "anchor_fallback_used": anchor_debug.get("anchor_fallback_used"),
            "distance_to_tr_anchor": _cbo_theta_distance(agent, theta, anchor),
            "distance_to_robust_incumbent": _cbo_theta_distance(agent, theta, robust_theta),
            "distance_to_recent_best": _cbo_theta_distance(agent, theta, recent_best),
            "cbo_tr_mode": str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off"))),
            "cbo_tr_radius": float(getattr(agent, "trust_radius", np.nan)),
        }
        for j, name in enumerate(names[:len(theta)]):
            row[str(name)] = float(theta[j])
        rows.append(row)
    summary = {
        "selected_candidate_id": selected_idx if selected_idx >= 0 else None,
        "selected_source": str(sources[selected_idx]) if 0 <= selected_idx < len(sources) else None,
        "selected_mu": float(mu[selected_idx]) if 0 <= selected_idx < len(mu) and np.isfinite(mu[selected_idx]) else np.nan,
        "selected_sigma": float(sigma[selected_idx]) if 0 <= selected_idx < len(sigma) and np.isfinite(sigma[selected_idx]) else np.nan,
        "selected_acq": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_score": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_candidate_score": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_candidate_beta_eff": float(beta_value) if beta_value is not None and np.isfinite(float(beta_value)) else np.nan,
        "best_mu_candidate_source": str(sources[best_mu_idx]) if best_mu_idx is not None and best_mu_idx < len(sources) else None,
        "best_acq_candidate_source": str(sources[best_acq_idx]) if best_acq_idx is not None and best_acq_idx < len(sources) else None,
        "num_candidates": int(len(candidates)),
        "num_tr_candidates": int(sum(1 for s in sources if str(s) == "trust_region")),
        "num_global_candidates": int(sum(1 for s in sources if str(s) == "global_random")),
        "selected_rank_by_mu": int(ranks_mu[selected_idx]) if 0 <= selected_idx < len(ranks_mu) else None,
        "selected_rank_by_sigma": int(ranks_sigma[selected_idx]) if 0 <= selected_idx < len(ranks_sigma) else None,
        "selected_rank_by_score": int(ranks_score[selected_idx]) if 0 <= selected_idx < len(ranks_score) else None,
        "selected_rank_by_acq": int(ranks_acq[selected_idx]) if 0 <= selected_idx < len(ranks_acq) else None,
        "selected_candidate_rank_by_score": int(ranks_score[selected_idx]) if 0 <= selected_idx < len(ranks_score) else None,
        "selected_candidate_rank_by_sigma": int(ranks_sigma[selected_idx]) if 0 <= selected_idx < len(ranks_sigma) else None,
    }
    summary.update(anchor_debug)
    summary.update(beta_info)
    return rows, summary


def _stability_ask_contextual(self, state=None, context=None):
    tr_mode = str(getattr(self, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    self._active_context = context
    if tr_mode == "off":
        theta = _ORIG_AGENT_ASK_CONTEXTUAL_STABILITY(self, state=state, context=context)
        hist = dict(getattr(self, "last_history_debug", {}) or {})
        hist.update(dict(getattr(self, "cbo_last_history_denoise_stats", {}) or {}))
        self.last_debug_info = {**getattr(self, "last_debug_info", {}), **hist,
                                "cbo_tr_mode": "off", "cbo_tr_anchor_mode": getattr(self, "cbo_tr_anchor_mode", "posterior_mean"),
                                "cbo_tr_radius": float(getattr(self, "trust_radius", np.nan)),
                                "cbo_tr_anchor_theta": None, "cbo_tr_candidate_count": 0,
                                "cbo_global_candidate_count": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0),
                                "cbo_tr_update_reason": getattr(self, "cbo_tr_update_reason", "off"),
                                "cbo_tr_success_count": int(getattr(self, "cbo_tr_success_count", 0)),
                                "cbo_tr_failure_count": int(getattr(self, "cbo_tr_failure_count", 0)),
                                "selected_candidate_source": getattr(self, "last_debug_info", {}).get("selected_candidate_source", "acquisition_candidate"),
                                "selected_candidate_mu": getattr(self, "last_debug_info", {}).get("posterior_mu"),
                                "selected_candidate_sigma": getattr(self, "last_debug_info", {}).get("posterior_sigma"),
                                "selected_candidate_acq": getattr(self, "last_debug_info", {}).get("selected_candidate_acq"),
                                "selected_candidate_rank_by_mu": getattr(self, "last_debug_info", {}).get("selected_candidate_rank_by_mu"),
                                "selected_candidate_rank_by_acq": getattr(self, "last_debug_info", {}).get("selected_candidate_rank_by_acq"),
                                "best_mu_candidate_source": getattr(self, "last_debug_info", {}).get("best_mu_candidate_source"),
                                "best_acq_candidate_source": getattr(self, "last_debug_info", {}).get("best_acq_candidate_source"),
                                "num_candidates": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0),
                                "num_tr_candidates": 0,
                                "num_global_candidates": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0)}
        return theta

    self.step_count += 1
    low = self.bounds[0].tolist()
    high = self.bounds[1].tolist()
    base_debug = {
        "step": int(self.step_count),
        "state": str(state) if state is not None else None,
        "context": self._normalize_context(context) if context is not None else ([0.0] * self.context_dim if self.use_context else []),
        "training_sample_count": 0,
        "recent_sample_count": int(len(self.local_recent)),
        "archive_sample_count": int(sum(len(v) for v in self.local_archive.values())),
        "neighbor_k": int(self.neighbor_k),
        "topk_history": [],
        "topk_similarity": [],
        "pivot": None,
        "trust_radius": float(self.trust_radius),
        "best_selected": None,
        "candidate_count": 0,
    }
    if self.anchor_points and self.step_count <= len(self.anchor_points):
        theta = list(self.anchor_points[self.step_count - 1])
        self.last_theta = list(theta)
        self.acq_history.append({"step": int(self.step_count), "candidates": [theta], "acq_values": [], "best_selected": theta})
        self.last_debug_info = {**base_debug, **dict(getattr(self, "last_history_debug", {}) or {}), "best_selected": list(theta), "candidate_count": 1}
        return theta
    x, y, records = self._training_data(state=state)
    base_debug["training_sample_count"] = int(len(records))
    model_pack = self.fit_local_gp(state=state)
    if len(x) < 2 or model_pack is None:
        theta = self._sample_in_bounds(low, high)
        self.last_theta = list(theta)
        self.last_debug_info = {**base_debug, **dict(getattr(self, "last_history_debug", {}) or {}), "best_selected": list(theta), "candidate_count": 1}
        return theta
    gp = model_pack["gp"]
    y_mean = model_pack["y_mean"]
    y_std = model_pack["y_std"]
    bounds_full = model_pack["bounds"]
    global_count = 52
    tr_count = 76
    candidates = self._random_candidates(low, high, global_count)
    candidate_sources = ["global_random"] * len(candidates)
    anchor, anchor_debug = _cbo_resolve_actual_tr_anchor(self, context, records)
    anchor_mode = str(anchor_debug.get("effective_tr_anchor_mode", getattr(self, "cbo_tr_anchor_mode", "posterior_mean")) or "posterior_mean").lower()
    anchor_source = str(anchor_debug.get("actual_tr_anchor_source", "no_anchor"))
    anchor_info = {"tr_anchor_source_pool": anchor_debug.get("tr_anchor_source_pool", "all_records")}
    if anchor is None:
        anchor = self._select_pivot_theta(context, records) or getattr(self, "prev_best", None)
    if anchor is not None:
        base_debug["pivot"] = list(anchor)
        for _ in range(tr_count):
            cand = []
            for d in range(self.dim):
                span = (high[d] - low[d]) * float(self.trust_radius)
                cand.append(min(max(anchor[d] + span * (2.0 * self.py_rng.random() - 1.0), low[d]), high[d]))
            candidates.append(cand)
            candidate_sources.append("trust_region")
    unique, seen = [], set()
    unique_sources = []
    for src_idx, c in enumerate(candidates):
        key = tuple(round(float(v), 6) for v in c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        unique_sources.append(candidate_sources[src_idx] if src_idx < len(candidate_sources) else "unknown")
    candidates = unique[:160] or [self._sample_in_bounds(low, high)]
    candidate_sources = unique_sources[:len(candidates)] if unique_sources else ["fallback"] * len(candidates)
    mu, sigma, score = self._contextual_scores(gp, y_mean, y_std, bounds_full, candidates, context)
    beta_info = _cbo_beta_eff_info(self)
    score = np.asarray(mu, dtype=float) + float(beta_info.get("beta_eff", getattr(self, "cbo_acq_beta", self.beta_init))) * np.asarray(sigma, dtype=float)
    score, service_penalty, guard_info = _cbo_service_guard_apply(self, score)
    beta_info.update(guard_info)
    try:
        self.cbo_last_beta_info = dict(beta_info)
    except Exception:
        pass
    best_idx, select_reason = _cbo_select_index_from_scores(self, mu, sigma, score, default_reason="max_acquisition_score")
    best = list(candidates[best_idx])
    if self.last_theta is not None and self.theta_momentum > 0.0:
        best = [self.theta_momentum * self.last_theta[d] + (1.0 - self.theta_momentum) * best[d] for d in range(self.dim)]
    best = [min(max(best[d], low[d]), high[d]) for d in range(self.dim)]
    self.last_theta = list(best)
    self.acq_history.append({
        "step": int(self.step_count),
        "candidates": [list(c) for c in candidates],
        "acq_values": [float(v) for v in score.tolist()],
        "best_selected": list(best),
        "model_state_dict": gp.state_dict(),
    })
    hist = dict(getattr(self, "last_history_debug", {}) or {})
    hist.update(dict(getattr(self, "cbo_last_history_denoise_stats", {}) or {}))
    recent_records = [self._unpack_sample(s) for s in getattr(self, "local_recent", [])]
    recent_best = list(max(recent_records, key=lambda r: float(r.get("y", -1e300))).get("theta", [])) if recent_records else None
    robust_theta = None
    try:
        robust_theta, _robust_info_for_diag = self._compute_robust_incumbent(context=context)
    except Exception:
        robust_theta = None
    candidate_rows, candidate_summary = _cbo_candidate_rows(
        self, candidates, candidate_sources, mu, sigma, score, best_idx,
        selected_reason=str(select_reason),
        deploy_policy=str(tr_mode), deploy_source=str(select_reason),
        anchor=anchor, robust_theta=robust_theta, recent_best=recent_best,
        beta_eff=beta_info.get("beta_eff"), service_penalty=service_penalty,
    )
    self.last_debug_info = {
        **base_debug,
        **hist,
        **candidate_summary,
        **anchor_debug,
        **beta_info,
        "best_selected": list(best),
        "candidate_count": int(len(candidates)),
        "posterior_mu": float(mu[best_idx]),
        "posterior_sigma": float(sigma[best_idx]),
        "selected_candidate_source": str(select_reason if str(select_reason) != "max_acquisition_score" else (candidate_sources[best_idx] if best_idx < len(candidate_sources) else "unknown")),
        "selected_candidate_mu": float(mu[best_idx]),
        "selected_candidate_sigma": float(sigma[best_idx]),
        "selected_candidate_acq": float(score[best_idx]),
        "selected_candidate_score": float(score[best_idx]),
        "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
        "actual_beta_used": float(getattr(self, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
        "cbo_select_mode": str(getattr(self, "cbo_select_mode", "greedy")),
        "selected_reason": str(select_reason),
        "candidate_diagnostic_rows": candidate_rows,
        "cbo_tr_mode": str(tr_mode),
        "cbo_tr_anchor_mode": str(anchor_mode),
        "cbo_tr_radius": float(self.trust_radius),
        "cbo_tr_anchor_theta": list(anchor) if anchor is not None else None,
        "cbo_tr_anchor_source": str(anchor_source),
        "tr_anchor_source_pool": str((anchor_info or {}).get("tr_anchor_source_pool", hist.get("tr_anchor_source_pool", "all_records"))),
        "cbo_tr_candidate_count": int(tr_count if anchor is not None else 0),
        "cbo_global_candidate_count": int(global_count),
        "cbo_tr_update_reason": getattr(self, "cbo_tr_update_reason", "candidate_generation"),
        "cbo_tr_success_count": int(getattr(self, "cbo_tr_success_count", 0)),
        "cbo_tr_failure_count": int(getattr(self, "cbo_tr_failure_count", 0)),
    }
    return list(best)


FederatedBOAgent._ask_contextual = _stability_ask_contextual


def reduced6_lite_context_agent_kwargs(use_trust_region=False, anchor_mode="none", context_mode="lite"):
    kwargs = reduced6_agent_kwargs(use_context=True, use_trust_region=use_trust_region, anchor_mode=anchor_mode)
    kwargs["use_state_partition"] = False
    kwargs["context_dim"] = len(lite_context_feature_names(context_mode))
    kwargs["context_bounds"] = lite_context_bounds(context_mode)
    return kwargs


USER_METHOD_ALIASES = {
    "fixed_mid": "reduced6_fixed_mid",
    "fixed_tuned": "reduced6_fixed_tuned",
    "fixed_queue_high": "reduced6_fixed_queue_high",
    "fixed_risk_high": "reduced6_fixed_risk_high",
    "fixed_edge_safe": "reduced6_fixed_edge_safe",
    "bo-ei": "reduced6_bo_ei",
    "bo_ei": "reduced6_bo_ei",
    "boei": "reduced6_bo_ei",
    "bo-greedy": "reduced6_bo_greedy",
    "bo_greedy": "reduced6_bo_greedy",
    "bogreedy": "reduced6_bo_greedy",
    "bo-greedy-recent": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_recent": "reduced6_bo_greedy_recent_conf",
    "bo-greedy-confidence": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_confidence": "reduced6_bo_greedy_recent_conf",
    "bo-greedy-recent-confidence": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_recent_confidence": "reduced6_bo_greedy_recent_conf",
    "cbo-lite": "reduced6_cbo_lite_recent_conf",
    "cbo_lite": "reduced6_cbo_lite_recent_conf",
    "cbo-lite-greedy": "reduced6_cbo_lite_recent_conf",
    "cbo_lite_greedy": "reduced6_cbo_lite_recent_conf",
    "cbo-lite-recent-confidence": "reduced6_cbo_lite_recent_conf",
    "cbo_lite_recent_confidence": "reduced6_cbo_lite_recent_conf",
    "cbo-full": "reduced6_cbo_lite_full",
    "cbo_full": "reduced6_cbo_lite_full",
    "cbo-lite-full": "reduced6_cbo_lite_full",
    "cbo_lite_full": "reduced6_cbo_lite_full",
    "cbo-load": "reduced6_cbo_lite_load_only",
    "cbo_load": "reduced6_cbo_lite_load_only",
    "cbo-lite-load": "reduced6_cbo_lite_load_only",
    "cbo_lite_load": "reduced6_cbo_lite_load_only",
    "cbo-util": "reduced6_cbo_lite_util_only",
    "cbo_util": "reduced6_cbo_lite_util_only",
    "cbo-lite-util": "reduced6_cbo_lite_util_only",
    "cbo_lite_util": "reduced6_cbo_lite_util_only",
    "cbo-pressure": "reduced6_cbo_lite_pressure_only",
    "cbo_pressure": "reduced6_cbo_lite_pressure_only",
    "cbo-lite-pressure": "reduced6_cbo_lite_pressure_only",
    "cbo_lite_pressure": "reduced6_cbo_lite_pressure_only",
    "cbo-no-cloud": "reduced6_cbo_lite_no_cloud",
    "cbo_no_cloud": "reduced6_cbo_lite_no_cloud",
    "cbo-lite-no-cloud": "reduced6_cbo_lite_no_cloud",
    "cbo_lite_no_cloud": "reduced6_cbo_lite_no_cloud",
    "cbo-no-arrival": "reduced6_cbo_lite_no_arrival",
    "cbo_no_arrival": "reduced6_cbo_lite_no_arrival",
    "cbo-lite-no-arrival": "reduced6_cbo_lite_no_arrival",
    "cbo_lite_no_arrival": "reduced6_cbo_lite_no_arrival",
    "cbo-taskmix": "reduced6_cbo_lite_taskmix",
    "cbo_taskmix": "reduced6_cbo_lite_taskmix",
    "cbo-lite-taskmix": "reduced6_cbo_lite_taskmix",
    "cbo_lite_taskmix": "reduced6_cbo_lite_taskmix",
    "cbo-recent-mix": "reduced6_cbo_lite_recent_mix",
    "cbo_recent_mix": "reduced6_cbo_lite_recent_mix",
    "cbo-prev-counts": "reduced6_cbo_lite_prev_counts",
    "cbo_prev_counts": "reduced6_cbo_lite_prev_counts",
    "cbo-pressure-taskmix": "reduced6_cbo_lite_pressure_taskmix",
    "cbo_pressure_taskmix": "reduced6_cbo_lite_pressure_taskmix",
    "cbo-ptask": "reduced6_cbo_lite_pressure_taskmix",
    "cbo_ptask": "reduced6_cbo_lite_pressure_taskmix",
    "cbo-pressure-recent-mix": "reduced6_cbo_lite_pressure_recent_mix",
    "cbo_pressure_recent_mix": "reduced6_cbo_lite_pressure_recent_mix",
    "cbo-pressure-counts": "reduced6_cbo_lite_pressure_counts",
    "cbo_pressure_counts": "reduced6_cbo_lite_pressure_counts",
    "cbo-pressure-taskmix-counts": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo_pressure_taskmix_counts": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo-ptc": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo_ptc": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo-alpha-direct": "reduced6_cbo_alpha_direct",
    "cbo_alpha_direct": "reduced6_cbo_alpha_direct",
    "alpha-direct": "reduced6_cbo_alpha_direct",
    "alpha_direct": "reduced6_cbo_alpha_direct",
    "cbo-alpha-direct-no-risk": "reduced6_cbo_alpha_direct_no_risk",
    "cbo_alpha_direct_no_risk": "reduced6_cbo_alpha_direct_no_risk",
    "alpha-direct-no-risk": "reduced6_cbo_alpha_direct_no_risk",
    "alpha_direct_no_risk": "reduced6_cbo_alpha_direct_no_risk",
    "ad-no-risk": "reduced6_cbo_alpha_direct_no_risk",
    "cbo-alpha-direct-unfinished-context": "reduced6_cbo_alpha_direct_unfinished_context",
    "cbo_alpha_direct_unfinished_context": "reduced6_cbo_alpha_direct_unfinished_context",
    "alpha-direct-unfinished-context": "reduced6_cbo_alpha_direct_unfinished_context",
    "alpha_direct_unfinished_context": "reduced6_cbo_alpha_direct_unfinished_context",
    "ad-unfinished-context": "reduced6_cbo_alpha_direct_unfinished_context",
    "ad_unfinished_context": "reduced6_cbo_alpha_direct_unfinished_context",
    "cbo-alpha-direct-prev-unfinished-context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "cbo_alpha_direct_prev_unfinished_context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "alpha-direct-prev-unfinished-context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "alpha_direct_prev_unfinished_context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "ad-prev-unfinished-context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "ad_prev_unfinished_context": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "cbo-alpha-direct-prev-unfinished": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "ad-prev-unfinished": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "cbo-alpha-direct-prev-unfinished-risk0": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "cbo_alpha_direct_prev_unfinished_risk0": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "alpha-direct-prev-unfinished-risk0": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "ad-prev-unfinished-risk0": "reduced6_cbo_alpha_direct_prev_unfinished_context",
    "cbo-full-taskmix": "reduced6_cbo_lite_full_taskmix",
    "cbo_full_taskmix": "reduced6_cbo_lite_full_taskmix",
    "cbo-full-taskmix-counts": "reduced6_cbo_lite_full_taskmix_counts",
    "cbo_full_taskmix_counts": "reduced6_cbo_lite_full_taskmix_counts",
    "cbo-greedy": "reduced6_cbo_greedy_legacy",
    "cbo_greedy": "reduced6_cbo_greedy_legacy",
    "cbo-tr-greedy": "reduced6_cbo_tr_greedy_legacy",
    "cbo_tr_greedy": "reduced6_cbo_tr_greedy_legacy",
    "round-robin-direct": "direct_round_robin",
    "round_robin_direct": "direct_round_robin",
    "roundrobin-direct": "direct_round_robin",
    "rr-direct": "direct_round_robin",
    "greedy-direct-cost": "direct_greedy_cost",
    "greedy_direct_cost": "direct_greedy_cost",
    "direct-greedy-cost": "direct_greedy_cost",
    "least-load-direct": "direct_least_load",
    "least_load_direct": "direct_least_load",
    "leastload-direct": "direct_least_load",
    "queue-aware-greedy-direct": "direct_queue_aware_greedy",
    "queue_aware_greedy_direct": "direct_queue_aware_greedy",
    "queueaware-greedy-direct": "direct_queue_aware_greedy",
    "dpp-greedy-direct": "direct_queue_aware_greedy",
    "dpp_greedy_direct": "direct_queue_aware_greedy",
    # Backwards-compatible aliases for old names.
    "reduced6_vanilla_bo_anchor": "reduced6_bo_ei",
    "reduced6_context_bo_anchor": "reduced6_cbo_greedy_legacy",
    "reduced6_context_tr_bo_anchor": "reduced6_cbo_tr_greedy_legacy",
}


def normalize_selected_method_keys(selected_keys):
    """支持用户输入短别名，同时保留原有完整 group key。"""
    if selected_keys is None:
        return None
    out = []
    for raw in selected_keys:
        k = str(raw).strip()
        if not k:
            continue
        out.append(USER_METHOD_ALIASES.get(k, USER_METHOD_ALIASES.get(k.lower(), k)))
    # 去重但保持顺序。
    seen = set()
    dedup = []
    for k in out:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def _dual_mark_last_sample(agent, sample_id, window_cost, state=None, context=None):
    for rec in reversed(list(getattr(agent, "local_recent", []))):
        if isinstance(rec, dict) and rec.get("sample_id") is None:
            rec["sample_id"] = sample_id
            rec["feedback_cost"] = float(window_cost)
            rec["feedback_source"] = "window_provisional"
            rec["feedback_state"] = str(state) if state is not None else None
            rec["feedback_context"] = list(context) if context is not None else None
            return True
    return False


def _cbo_dump_candidate_diagnostics(output_dir, iteration, safe_info, group_key=None):
    rows = list(safe_info.get("candidate_diagnostic_rows") or [])
    if not rows:
        return
    try:
        diag_dir = os.path.join(output_dir, "candidate_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        topn = max(1, int(getattr(CFG, "CBO_DUMP_CANDIDATES_TOPN", 30)))
        selected_id = safe_info.get("selected_candidate_id")
        clean_rows = []
        for row in rows:
            r = dict(row)
            r["iteration"] = int(iteration)
            r["group_key"] = str(group_key) if group_key is not None else None
            r["deploy_policy"] = safe_info.get("deploy_policy", r.get("deploy_policy"))
            r["deploy_source"] = safe_info.get("deploy_source", r.get("deploy_source"))
            clean_rows.append(r)
        selected_rows = [r for r in clean_rows if int(r.get("is_selected", 0) or 0) == 1]
        top_rows = sorted(clean_rows, key=lambda r: int(r.get("rank_by_acq") or 10**9))[:topn]
        if selected_rows:
            sel = dict(selected_rows[0])
            def fill_missing(key, value):
                if _is_missing_value(safe_info.get(key)):
                    safe_info[key] = value
            fill_missing("selected_candidate_id", sel.get("candidate_id"))
            fill_missing("selected_candidate_source", sel.get("candidate_source"))
            fill_missing("selected_source", sel.get("candidate_source"))
            fill_missing("selected_candidate_mu", sel.get("posterior_mu"))
            fill_missing("selected_mu", sel.get("posterior_mu"))
            fill_missing("selected_candidate_sigma", sel.get("posterior_sigma"))
            fill_missing("selected_sigma", sel.get("posterior_sigma"))
            fill_missing("selected_candidate_acq", sel.get("acquisition_score"))
            fill_missing("selected_acq", sel.get("acquisition_score"))
            fill_missing("selected_candidate_score", sel.get("score", sel.get("acquisition_score")))
            fill_missing("selected_score", sel.get("score", sel.get("acquisition_score")))
            fill_missing("selected_candidate_beta_eff", sel.get("beta_eff"))
            fill_missing("selected_candidate_rank_by_mu", sel.get("rank_by_mu"))
            fill_missing("selected_rank_by_mu", sel.get("rank_by_mu"))
            fill_missing("selected_candidate_rank_by_score", sel.get("rank_by_score"))
            fill_missing("selected_rank_by_score", sel.get("rank_by_score"))
            fill_missing("selected_candidate_rank_by_sigma", sel.get("rank_by_sigma"))
            fill_missing("selected_rank_by_sigma", sel.get("rank_by_sigma"))
            fill_missing("selected_candidate_rank_by_acq", sel.get("rank_by_acq"))
            fill_missing("selected_rank_by_acq", sel.get("rank_by_acq"))
        merged = []
        seen = set()
        for r in selected_rows + top_rows:
            cid = int(r.get("candidate_id", -1))
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(r)
        cand_path = os.path.join(diag_dir, f"candidates_iter_{int(iteration):04d}.csv")
        pd.DataFrame(merged).to_csv(cand_path, index=False, encoding="utf-8-sig")

        runtime_override_reason = safe_info.get("runtime_anchor_override_reason")
        anchor_override_reason = safe_info.get("anchor_override_reason")
        runtime_override_mode = safe_info.get("runtime_anchor_override")
        actual_anchor_reason = str(safe_info.get("actual_tr_anchor_reason") or "")
        try:
            override_used = (not _is_missing_value(safe_info.get("anchor_override_used"))) and int(float(safe_info.get("anchor_override_used"))) != 0
        except Exception:
            override_used = False
        if _is_missing_value(runtime_override_reason) and (not _is_missing_value(runtime_override_mode)) and (override_used or "runtime_override" in actual_anchor_reason):
            runtime_override_reason = f"runtime_anchor_override={runtime_override_mode}"
        if _is_missing_value(anchor_override_reason) and not _is_missing_value(runtime_override_reason):
            anchor_override_reason = runtime_override_reason

        summary = {
            "iteration": int(iteration),
            "group_key": str(group_key) if group_key is not None else None,
            "selected_candidate_id": safe_info.get("selected_candidate_id", selected_id),
            "selected_source": safe_info.get("selected_source", safe_info.get("selected_candidate_source")),
            "selected_mu": safe_info.get("selected_mu", safe_info.get("selected_candidate_mu")),
            "selected_sigma": safe_info.get("selected_sigma", safe_info.get("selected_candidate_sigma")),
            "selected_acq": safe_info.get("selected_acq", safe_info.get("selected_candidate_acq")),
            "selected_score": safe_info.get("selected_score", safe_info.get("selected_candidate_score")),
            "selected_beta_eff": safe_info.get("selected_candidate_beta_eff", safe_info.get("beta_eff")),
            "best_mu_candidate_source": safe_info.get("best_mu_candidate_source"),
            "best_acq_candidate_source": safe_info.get("best_acq_candidate_source"),
            "num_candidates": safe_info.get("num_candidates", safe_info.get("candidate_count")),
            "num_tr_candidates": safe_info.get("num_tr_candidates", safe_info.get("cbo_tr_candidate_count")),
            "num_global_candidates": safe_info.get("num_global_candidates", safe_info.get("cbo_global_candidate_count")),
            "selected_rank_by_mu": safe_info.get("selected_rank_by_mu", safe_info.get("selected_candidate_rank_by_mu")),
            "selected_rank_by_score": safe_info.get("selected_rank_by_score", safe_info.get("selected_candidate_rank_by_score")),
            "selected_rank_by_sigma": safe_info.get("selected_rank_by_sigma", safe_info.get("selected_candidate_rank_by_sigma")),
            "selected_rank_by_acq": safe_info.get("selected_rank_by_acq", safe_info.get("selected_candidate_rank_by_acq")),
            "actual_tr_anchor_mode": safe_info.get("actual_tr_anchor_mode"),
            "actual_tr_anchor_source": safe_info.get("actual_tr_anchor_source"),
            "actual_tr_anchor_theta": _safe_json(safe_info.get("actual_tr_anchor_theta")),
            "anchor_override_used": safe_info.get("anchor_override_used"),
            "anchor_override_reason": anchor_override_reason,
            "anchor_fallback_used": safe_info.get("anchor_fallback_used"),
            "anchor_fallback_reason": safe_info.get("anchor_fallback_reason"),
            "runtime_anchor_override_reason": runtime_override_reason,
        }
        summary_path = os.path.join(diag_dir, "candidate_selection_summary.csv")
        summary_df = pd.DataFrame([summary])
        if os.path.exists(summary_path):
            try:
                old_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
                summary_df = pd.concat([old_summary, summary_df], ignore_index=True, sort=False)
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            except Exception:
                # If an older run left a mismatched header/row shape, restart this compact diagnostic file.
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[WARN] failed to dump candidate diagnostics: {type(exc).__name__}: {exc}", flush=True)


def _cbo_ws_json(value, default=None):
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except Exception:
        pass
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _cbo_ws_float(value, default=np.nan):
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _cbo_ws_int(value, default=0):
    try:
        if value is None:
            return default
        out = int(float(value))
        return out
    except Exception:
        return default


def _cbo_warm_context_feature_names(group_cfg):
    try:
        mode = str((group_cfg or {}).get("context_mode", "legacy") or "legacy").strip().lower()
        if mode in set(LITE_CONTEXT_MODE_SPECS.keys()) | {"state_lite", "cbo_lite"}:
            if mode in {"state_lite", "cbo_lite"}:
                mode = "lite"
            return list(lite_context_feature_names(mode))
    except Exception:
        pass
    try:
        agent_kwargs = (group_cfg or {}).get("agent_kwargs", {}) or {}
        if int(agent_kwargs.get("context_dim", 0) or 0) <= 0:
            return []
    except Exception:
        pass
    return list(getattr(CFG, "CONTEXT_FEATURE_NAMES", []))


def _cbo_warm_history_files(path_or_dir):
    spec = str(path_or_dir or "").strip()
    if not spec:
        return []
    path = os.path.abspath(spec)
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        print(f"[WARN] CBO warm-start history path not found: {path}", flush=True)
        return []
    direct = os.path.join(path, "bo_warm_history.csv")
    if os.path.isfile(direct):
        return [direct]
    found = []
    for root, _, files in os.walk(path):
        if "bo_warm_history.csv" in files:
            found.append(os.path.join(root, "bo_warm_history.csv"))
    return sorted(found)


def _cbo_row_compatible(row, target):
    checks = [
        ("control_dim", _cbo_ws_int(row.get("control_dim"), -1), int(target["control_dim"])),
        ("context_dim", _cbo_ws_int(row.get("context_dim"), -1), int(target["context_dim"])),
        ("context_feature_names", list(_cbo_ws_json(row.get("context_feature_names"), []) or []), list(target["context_feature_names"])),
        ("scheduler_tradeoff_mode", str(row.get("scheduler_tradeoff_mode", "")), str(target["scheduler_tradeoff_mode"])),
        ("scheduler_score_norm_mode", str(row.get("scheduler_score_norm_mode", "")), str(target["scheduler_score_norm_mode"])),
    ]
    for name, got, want in checks:
        if got != want:
            return False, f"{name} mismatch source={got} target={want}"
    src_le = _cbo_ws_float(row.get("scheduler_le_scale"), np.nan)
    tgt_le = float(target["scheduler_le_scale"])
    if not np.isfinite(src_le) or abs(src_le - tgt_le) > 1e-9:
        return False, f"scheduler_le_scale mismatch source={src_le} target={tgt_le}"
    return True, "ok"


def _cbo_select_warm_rows(rows, mode, target_context, topk, max_rows):
    mode = str(mode or "none").strip().lower()
    rows = list(rows)
    if mode == "similar_topk" and target_context is not None:
        tgt = np.asarray(list(target_context), dtype=float)
        scored = []
        for idx, row in enumerate(rows):
            ctx = _cbo_ws_json(row.get("context_vector"), None)
            if not isinstance(ctx, (list, tuple)):
                dist = float("inf")
            else:
                arr = np.asarray(list(ctx), dtype=float)
                if arr.size != tgt.size:
                    dist = float("inf")
                else:
                    dist = float(np.linalg.norm(arr - tgt))
            scored.append((dist, idx, row))
        scored.sort(key=lambda x: (x[0], x[1]))
        rows = [r for _, _, r in scored[:max(1, int(topk))]]
    return rows[:max(0, int(max_rows))]


def _cbo_warm_record_from_row(agent, row):
    theta = _cbo_ws_json(row.get("control_theta"), None)
    if not isinstance(theta, (list, tuple)):
        theta = _cbo_ws_json(row.get("deployed_theta"), None)
    context = _cbo_ws_json(row.get("context_vector"), None)
    cost = _cbo_ws_float(row.get("BO_Training_Cost"), np.nan)
    if not np.isfinite(cost):
        cost = _cbo_ws_float(row.get("Eval_Cost"), np.nan)
    if not isinstance(theta, (list, tuple)) or not np.isfinite(cost):
        return None
    rec = agent._pack_sample(list(theta), -float(cost), state=None, context=context)
    rec["feedback_confidence"] = 1.0
    rec["bo_iter"] = _cbo_ws_int(row.get("iteration"), None)
    rec["group_key"] = str(row.get("selected_key", row.get("method", "warm_source")) or "warm_source")
    rec["history_mode"] = str(getattr(agent, "history_mode", _cfg_history_mode()))
    rec["cbo_warm_start_source"] = True
    rec["source_scene_label"] = str(row.get("source_scene_label", "") or "")
    rec["result_file_path"] = str(row.get("result_file_path", "") or "")
    metrics = {
        "cost": _cbo_ws_float(row.get("Eval_Cost"), np.nan),
        "bo_training_cost": float(cost),
        "avg_delay": _cbo_ws_float(row.get("Avg_Delay"), np.nan),
        "avg_energy": _cbo_ws_float(row.get("Avg_Energy"), np.nan),
        "unfinished_end": _cbo_ws_float(row.get("unfinished_end"), np.nan),
        "unfinished_rate": _cbo_ws_float(row.get("unfinished_rate"), np.nan),
    }
    rec["metrics"] = {k: v for k, v in metrics.items() if not (isinstance(v, float) and not np.isfinite(v))}
    return rec


def _cbo_inject_warm_start(agent, group_key, group_cfg, target_context=None):
    mode = str(getattr(CFG, "CBO_WARM_START_MODE", "none") or "none").strip().lower()
    status = {
        "cbo_warm_start_enabled": bool(mode != "none"),
        "cbo_warm_start_mode": mode,
        "cbo_warm_start_loaded_rows": 0,
        "cbo_warm_start_history_path": str(getattr(CFG, "CBO_WARM_START_HISTORY", "") or ""),
    }
    if agent is None or mode == "none" or not _is_cbo_method_key(group_key, group_cfg):
        return status
    target = {
        "control_dim": int(getattr(agent, "dim", 0) or 0),
        "context_dim": int(getattr(agent, "context_dim", 0) or 0),
        "context_feature_names": _cbo_warm_context_feature_names(group_cfg),
        "scheduler_tradeoff_mode": str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy")),
        "scheduler_score_norm_mode": str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
        "scheduler_le_scale": float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)),
    }
    files = _cbo_warm_history_files(getattr(CFG, "CBO_WARM_START_HISTORY", ""))
    if not files:
        print("[WARN] CBO warm-start enabled but no bo_warm_history.csv files were found", flush=True)
        return status
    rows = []
    warn_counts = {}
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] failed to read warm-start history {csv_path}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for row in df.to_dict("records"):
            ok, reason = _cbo_row_compatible(row, target)
            if not ok:
                warn_counts[reason] = warn_counts.get(reason, 0) + 1
                continue
            row["_warm_history_file"] = csv_path
            rows.append(row)
    for reason, count in sorted(warn_counts.items())[:8]:
        print(f"[WARN] skipped {count} CBO warm-start rows: {reason}", flush=True)
    rows = _cbo_select_warm_rows(
        rows,
        mode=mode,
        target_context=target_context,
        topk=int(getattr(CFG, "CBO_WARM_START_TOPK", 100)),
        max_rows=int(getattr(CFG, "CBO_WARM_START_MAX_ROWS", 300)),
    )
    loaded = 0
    archive_key = "CBO_WARM_START"
    local_capacity = int(getattr(getattr(agent, "local_recent", None), "maxlen", 0) or 0)
    recent_budget = max(0, min(local_capacity, len(rows)))
    archive_rows = rows[:-recent_budget] if recent_budget else rows
    recent_rows = rows[-recent_budget:] if recent_budget else []
    for row in archive_rows:
        rec = _cbo_warm_record_from_row(agent, row)
        if rec is None:
            continue
        agent.local_archive[archive_key].append(rec)
        loaded += 1
    for row in recent_rows:
        rec = _cbo_warm_record_from_row(agent, row)
        if rec is None:
            continue
        agent.local_recent.append(rec)
        loaded += 1
    all_recs = []
    for bucket in getattr(agent, "local_archive", {}).values():
        all_recs.extend(list(bucket))
    all_recs.extend(list(getattr(agent, "local_recent", [])))
    warm_recs = [r for r in all_recs if isinstance(r, dict) and bool(r.get("cbo_warm_start_source"))]
    if warm_recs:
        best = max(warm_recs, key=lambda r: float(r.get("y", -1e300)))
        agent.prev_best = list(best.get("theta", []))
        agent.prev_best_value = float(best.get("y", -1e300))
        agent.prev_best_iter = _cbo_ws_int(best.get("bo_iter"), None)
    status["cbo_warm_start_loaded_rows"] = int(loaded)
    status["cbo_warm_start_history_path"] = ";".join(files)
    try:
        agent.cbo_warm_start_loaded_rows = int(loaded)
        agent.cbo_warm_start_history_path = status["cbo_warm_start_history_path"]
        agent.cbo_warm_start_mode = mode
    except Exception:
        pass
    print(
        f"[CBO-WARM-START] method={group_key} mode={mode} loaded_rows={loaded} files={len(files)}",
        flush=True,
    )
    return status


def _cbo_is_warm_record(rec):
    return isinstance(rec, dict) and bool(rec.get("cbo_warm_start_source"))


def _cbo_warm_log_get(log, key, idx, default=None):
    vals = log.get(key, []) if isinstance(log, dict) else []
    if idx < len(vals):
        return vals[idx]
    return default


def _cbo_warm_json_dump(value):
    try:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
    except Exception:
        pass
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _cbo_warm_group_dims(group_cfg):
    agent_kwargs = (group_cfg or {}).get("agent_kwargs", {}) or {}
    try:
        control_dim = int(len((group_cfg or {}).get("fixed_theta", []))) if (group_cfg or {}).get("fixed_theta") is not None else int(agent_kwargs.get("dim", 0) or 0)
    except Exception:
        control_dim = int(agent_kwargs.get("dim", 0) or 0)
    try:
        context_dim = int(agent_kwargs.get("context_dim", 0) or 0)
    except Exception:
        context_dim = 0
    return control_dim, context_dim


def export_bo_warm_history_csv(group_logs, output_dir=None, selected_keys=None, groups=None):
    output_dir = os.path.abspath(output_dir or SCENARIO_SAVE_DIR)
    os.makedirs(output_dir, exist_ok=True)
    source_label = str(getattr(CFG, "CBO_WARM_START_LABEL", "") or "")
    if not source_label:
        source_label = os.path.basename(os.path.abspath(output_dir))
    task_probs = dict(getattr(CFG, "TASK_TYPE_PROBS", {}) or {})
    lambda_schedule = list(getattr(CFG, "LAMBDA_SCHEDULE", []) or [])
    lambda_value = ""
    if len(lambda_schedule) == 1:
        try:
            lambda_value = float(lambda_schedule[0][2])
        except Exception:
            lambda_value = ""
    rows = []
    for group_key, info in (group_logs or {}).items():
        group_cfg = (groups or {}).get(group_key, {}) if isinstance(groups, dict) else {}
        control_dim, context_dim = _cbo_warm_group_dims(group_cfg)
        fallback_context_names = _cbo_warm_context_feature_names(group_cfg)
        result_path = os.path.join(output_dir, f"{group_key}_round_summary_轮次汇总.csv")
        for repeat_idx, log in enumerate(info.get("logs", []) or [], start=1):
            n = len(log.get("time", [])) if isinstance(log, dict) else 0
            for i in range(n):
                ctx_names = _cbo_warm_log_get(log, "context_feature_names", i, fallback_context_names)
                if not isinstance(ctx_names, (list, tuple)):
                    ctx_names = fallback_context_names
                control_theta = _cbo_warm_log_get(log, "theta_control_deployed", i, None)
                alpha_theta = _cbo_warm_log_get(log, "alpha_direct_control_vector_6d", i, None)
                deployed_theta = alpha_theta if isinstance(alpha_theta, (list, tuple)) else control_theta
                eval_cost = _cbo_warm_log_get(log, "eval_cost", i, None)
                if eval_cost is None:
                    reward = _cbo_warm_log_get(log, "reward", i, None)
                    eval_cost = -_cbo_ws_float(reward, np.nan) if reward is not None else np.nan
                bo_cost = _cbo_warm_log_get(log, "bo_training_cost", i, eval_cost)
                rows.append({
                    "iteration": int(i + 1),
                    "repeat_idx": int(repeat_idx),
                    "method": str(group_key),
                    "selected_key": str(group_key),
                    "control_dim": int(control_dim),
                    "context_dim": int(context_dim),
                    "scheduler_tradeoff_mode": str(_cbo_warm_log_get(log, "scheduler_tradeoff_mode", i, getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy"))),
                    "scheduler_score_norm_mode": str(_cbo_warm_log_get(log, "scheduler_score_norm_mode", i, getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy"))),
                    "scheduler_le_scale": _cbo_ws_float(_cbo_warm_log_get(log, "scheduler_le_scale", i, getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)), float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0))),
                    "lambda_value": lambda_value,
                    "lambda_schedule": _cbo_warm_json_dump(lambda_schedule),
                    "task_probs": _cbo_warm_json_dump(task_probs),
                    "context_vector": _cbo_warm_json_dump(_cbo_warm_log_get(log, "context_vector", i, None)),
                    "context_feature_names": _cbo_warm_json_dump(list(ctx_names or [])),
                    "control_theta": _cbo_warm_json_dump(control_theta),
                    "deployed_theta": _cbo_warm_json_dump(deployed_theta),
                    "Alpha_Direct_Control_Vector_6D": _cbo_warm_json_dump(alpha_theta),
                    "BO_Training_Cost": _cbo_ws_float(bo_cost, np.nan),
                    "Eval_Cost": _cbo_ws_float(eval_cost, np.nan),
                    "Avg_Delay": _cbo_ws_float(_cbo_warm_log_get(log, "avg_delay", i, np.nan), np.nan),
                    "Avg_Energy": _cbo_ws_float(_cbo_warm_log_get(log, "avg_energy", i, np.nan), np.nan),
                    "unfinished_end": _cbo_ws_float(_cbo_warm_log_get(log, "unfinished_end", i, np.nan), np.nan),
                    "unfinished_rate": _cbo_ws_float(_cbo_warm_log_get(log, "unfinished_rate", i, np.nan), np.nan),
                    "source_scene_label": source_label,
                    "result_file_path": result_path,
                })
    path = os.path.join(output_dir, "bo_warm_history.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[CBO-WARM-EXPORT] rows={len(rows)} path={path}", flush=True)
    return path


def run_scenario_group(seed, group_key, group_cfg):
    group_cfg["group_key"] = group_key
    fac = ConnectedFactory(fid=0, name=group_cfg["label"], seed=seed, node_config=CFG.NODES_CFG, scheduler_type=group_cfg.get("scheduler_type", "Boltzmann"), norm_mode=group_cfg.get("norm_mode", "rolling"))
    fac.reset(use_batch=False)
    fac.agent = create_scenario_agent(group_cfg, seed)
    configure_refactor_agent(fac.agent, group_cfg)
    fac.perf_log["group_key"] = group_key
    fac.perf_log["group_label"] = group_cfg["label"]
    old_scheduler_tradeoff_mode = str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy"))
    method_scheduler_tradeoff_mode = group_cfg.get("scheduler_tradeoff_mode")
    if method_scheduler_tradeoff_mode:
        CFG.SCHEDULER_TRADEOFF_MODE = str(method_scheduler_tradeoff_mode)
    old_use_score_risk = bool(getattr(CFG, "USE_SCORE_RISK", True))
    method_use_score_risk = group_cfg.get("scheduler_use_score_risk", None)
    if method_use_score_risk is not None:
        CFG.USE_SCORE_RISK = bool(method_use_score_risk)
    warm_start_status = {
        "cbo_warm_start_enabled": False,
        "cbo_warm_start_mode": str(getattr(CFG, "CBO_WARM_START_MODE", "none") or "none"),
        "cbo_warm_start_loaded_rows": 0,
        "cbo_warm_start_history_path": str(getattr(CFG, "CBO_WARM_START_HISTORY", "") or ""),
    }
    if fac.agent is not None:
        try:
            ws_state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
            ws_base_ctx = fac.scenario_monitor.get_context_vector(fac.current_time)
            ws_ctx = build_context_for_group(fac, group_cfg, base_context=ws_base_ctx)
            ws_ctx = ws_ctx if getattr(fac.agent, "use_context", False) else None
            warm_start_status = _cbo_inject_warm_start(fac.agent, group_key, group_cfg, target_context=ws_ctx)
        except Exception as exc:
            print(f"[WARN] CBO warm-start injection failed for {group_key}: {type(exc).__name__}: {exc}", flush=True)
    # v6.2 runtime logging: per-method and per-iteration elapsed time.
    runtime_group_t0 = time.perf_counter()
    runtime_group_wall_t0 = time.time()
    fac.perf_log["runtime_group_start_unix"] = [float(runtime_group_wall_t0)]
    print(
        f"[DEPLOY] method={group_key} deploy_policy={_safebo_policy_name(group_cfg)} "
        f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
        f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
        f"feedback={getattr(CFG, 'BO_TRAINING_FEEDBACK_SCORE', 'window_original')}",
        flush=True,
    )
    print(
        f"[HISTORY] method={group_key} "
        f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
        f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
        f"source={group_cfg.get('history_override_source', 'method_default')}",
        flush=True,
    )
    print(
        f"[SCHED-TRADEOFF] mode={getattr(CFG, 'SCHEDULER_TRADEOFF_MODE', 'legacy')} "
        f"alpha={getattr(CFG, 'SCHEDULER_TRADEOFF_ALPHA', 0.85)} "
        f"alpha_min={getattr(CFG, 'SCHEDULER_ALPHA_MIN', 0.60)} "
        f"alpha_max={getattr(CFG, 'SCHEDULER_ALPHA_MAX', 0.97)} "
        f"le_scale={getattr(CFG, 'SCHEDULER_LE_SCALE', 1.0)} "
        f"use_score_risk={getattr(CFG, 'USE_SCORE_RISK', True)} "
        f"service_latency_weight={getattr(CFG, 'SCHEDULER_SERVICE_LATENCY_WEIGHT', 1.0)} "
        f"service_risk_weight={getattr(CFG, 'SCHEDULER_SERVICE_RISK_WEIGHT', 1.0)} "
        f"service_queue_weight={getattr(CFG, 'SCHEDULER_SERVICE_QUEUE_WEIGHT', 1.0)} "
        f"energy_weight={getattr(CFG, 'SCHEDULER_ENERGY_WEIGHT', 1.0)}",
        flush=True,
    )
    print(
        f"[SCHED-NORM] mode={getattr(CFG, 'SCHEDULER_SCORE_NORM_MODE', 'legacy')} "
        f"clip_max={getattr(CFG, 'SCHEDULER_NORM_CLIP_MAX', 3.0)} "
        f"eps={getattr(CFG, 'SCHEDULER_NORM_EPS', 1e-6)} "
        f"ema_alpha={getattr(CFG, 'SCHEDULER_NORM_EMA_ALPHA', 0.995)}",
        flush=True,
    )
    if _is_cbo_method_key(group_key, group_cfg):
        print(
            f"[CBO-STABILITY] method={group_key} "
            f"history_select_mode={group_cfg.get('cbo_history_select_mode', _cfg_cbo_history_select_mode())} "
            f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
            f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
            f"context_k={group_cfg.get('cbo_context_k', _cfg_cbo_int('CBO_CONTEXT_K', 50))} "
            f"elite_k={group_cfg.get('cbo_elite_k', _cfg_cbo_int('CBO_ELITE_K', 20))} "
            f"diverse_k={group_cfg.get('cbo_diverse_k', _cfg_cbo_int('CBO_DIVERSE_K', 20))} "
            f"robust_score_mode={group_cfg.get('cbo_robust_score_mode', _cfg_cbo_str('CBO_ROBUST_SCORE_MODE', 'none'))} "
            f"tr_mode={group_cfg.get('cbo_tr_mode', _cfg_cbo_str('CBO_TR_MODE', 'off'))} "
            f"tr_anchor_mode={group_cfg.get('cbo_tr_anchor_mode', _cfg_cbo_str('CBO_TR_ANCHOR_MODE', 'posterior_mean'))} "
            f"robust_incumbent_mode={group_cfg.get('cbo_robust_incumbent_mode', _cfg_cbo_str('CBO_ROBUST_INCUMBENT_MODE', 'off'))}",
            flush=True,
        )
        print(
            f"[CBO-MACRO-GATE] method={group_key} "
            f"macro_gate_mode={group_cfg.get('cbo_macro_gate_mode', _cfg_cbo_str('CBO_MACRO_GATE_MODE', 'off'))} "
            f"macro_k={group_cfg.get('cbo_macro_k', _cfg_cbo_int('CBO_MACRO_K', 100))} "
            f"lengthscale_total={group_cfg.get('cbo_macro_lengthscale_total', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_TOTAL', 1.0))} "
            f"lengthscale_rt={group_cfg.get('cbo_macro_lengthscale_rt', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_RT', 0.15))} "
            f"lengthscale_batch={group_cfg.get('cbo_macro_lengthscale_batch', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_BATCH', 0.15))} "
            f"history_select_mode={group_cfg.get('cbo_history_select_mode', _cfg_cbo_history_select_mode())} "
            f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())}",
            flush=True,
        )
        print(
            f"[CBO-ACQ-BETA] method={group_key} "
            f"beta_mode={group_cfg.get('cbo_acq_beta_mode', _cfg_cbo_str('CBO_ACQ_BETA_MODE', 'fixed'))} "
            f"beta_min={group_cfg.get('cbo_beta_min', _cfg_cbo_float('CBO_BETA_MIN', 0.1))} "
            f"beta_max={group_cfg.get('cbo_beta_max', _cfg_cbo_float('CBO_BETA_MAX', 2.0))} "
            f"radius_beta_power={group_cfg.get('cbo_radius_beta_power', _cfg_cbo_float('CBO_RADIUS_BETA_POWER', 1.0))} "
            f"select_mode={group_cfg.get('cbo_select_mode', _cfg_cbo_str('CBO_SELECT_MODE', 'greedy'))} "
            f"tr_mode={group_cfg.get('cbo_tr_mode', _cfg_cbo_str('CBO_TR_MODE', 'off'))} "
            f"tr_anchor_mode={group_cfg.get('cbo_tr_anchor_mode', _cfg_cbo_str('CBO_TR_ANCHOR_MODE', 'posterior_mean'))} "
            f"tr_update_mode={group_cfg.get('cbo_tr_update_mode', _cfg_cbo_str('CBO_TR_UPDATE_MODE', 'best_so_far'))}",
            flush=True,
        )

    is_reduced = group_cfg.get("control_mode") in {"reduced4", "reduced6", "alpha_direct"}
    fac.disable_internal_agent_tell = bool(is_reduced and fac.agent is not None)

    for i in range(CFG.BO_ITERATIONS):
        runtime_iter_t0 = time.perf_counter()
        state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
        base_ctx = fac.scenario_monitor.get_context_vector(fac.current_time)
        ctx = build_context_for_group(fac, group_cfg, base_context=base_ctx)
        safe_info = {"deploy_policy": "fixed", "deploy_source": "fixed_theta", "explore_used": 0, "posterior_mu": None, "posterior_sigma": None, "candidate_count_safe": None}
        safe_info.update(warm_start_status)

        if fac.agent is None:
            theta_control = list(group_cfg["fixed_theta"])
            ask_state = state
            ask_ctx = ctx
        else:
            ask_state = state if getattr(fac.agent, "use_state_partition", False) else None
            ask_ctx = ctx if getattr(fac.agent, "use_context", False) else None
            theta_control, safe_info = _safebo_select_theta(fac.agent, state=ask_state, context=ask_ctx, group_cfg=group_cfg)
            safe_info.update(warm_start_status)
            safe_info.setdefault("cbo_warm_start_used_rows", safe_info.get("selected_warm_rows_count", 0))
            if bool(group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False)))):
                every = max(1, int(group_cfg.get("cbo_dump_candidates_every", getattr(CFG, "CBO_DUMP_CANDIDATES_EVERY", 20))))
                if ((i + 1) % every == 0) or ((i + 1) == int(CFG.BO_ITERATIONS)):
                    _cbo_dump_candidate_diagnostics(SCENARIO_SAVE_DIR, i + 1, safe_info, group_key=group_key)

        candidate_theta_control = list(theta_control)
        deployed_theta_control = list(theta_control)
        candidate_theta_6d = _alpha_direct_expanded_6d(candidate_theta_control, group_cfg) if group_cfg.get("control_mode") == "alpha_direct" else None
        deployed_theta_6d = _alpha_direct_expanded_6d(deployed_theta_control, group_cfg) if group_cfg.get("control_mode") == "alpha_direct" else None
        alpha_direct_fixed_enabled = bool(group_cfg.get("alpha_direct_fixed_theta_enabled", False))
        alpha_direct_fixed_theta_6d = group_cfg.get("alpha_direct_fixed_theta_6d")
        guard_info = {
            "good_region_guard_enabled": int(False),
            "good_region_guard_triggered": int(False),
            "good_region_guard_reason": "off",
            "candidate_theta_6d": candidate_theta_6d,
            "deployed_theta_6d": deployed_theta_6d,
            "good_region_theta_6d": None,
            "good_region_cost": np.nan,
            "good_region_iter": None,
            "good_region_window_start": None,
            "good_region_window_end": None,
            "distance_to_good_region": np.nan,
            "candidate_selected_source": safe_info.get("selected_candidate_source", safe_info.get("selected_source")),
            "deployed_source": safe_info.get("selected_candidate_source", safe_info.get("deploy_source")),
            "candidate_tr_radius": safe_info.get("cbo_tr_radius_after_update", safe_info.get("cbo_tr_radius")),
            "candidate_beta_eff": safe_info.get("beta_eff", safe_info.get("selected_candidate_beta_eff")),
            "guard_fallback_type": "none",
            "alpha_direct_fixed_theta_enabled": int(alpha_direct_fixed_enabled),
            "alpha_direct_fixed_theta_6d": list(alpha_direct_fixed_theta_6d) if alpha_direct_fixed_enabled and alpha_direct_fixed_theta_6d is not None else None,
        }
        if fac.agent is not None and _is_cbo_method_key(group_key, group_cfg):
            deployed_theta_control, guard_info = _cbo_apply_good_region_deployment_guard(
                fac.agent, candidate_theta_control, safe_info, group_cfg
            )
        theta_control = list(deployed_theta_control)
        safe_info.update(guard_info)
        theta_full = map_group_theta_to_full(theta_control, group_cfg)
        paired_shadow = None
        paired_window_end = None
        if _paired_delta_enabled():
            paired_window_end = float(fac.current_time) + float(getattr(CFG, "BO_INTERVAL", 40.0))
            try:
                _agent_ref = fac.agent
                fac.agent = None  # avoid copying torch generator / GP state into the shadow
                paired_shadow = copy.deepcopy(fac)
            finally:
                fac.agent = _agent_ref

        fac.current_control_vector = list(theta_full)
        fac.current_control_label = group_cfg.get("label", group_key)
        _, _, _, _, metrics, _ = fac.run_continuous(
            theta_full,
            eval_state=ask_state if fac.agent is not None else state,
            eval_context=ask_ctx if fac.agent is not None else ctx,
            feedback_control=theta_control,
        )
        if paired_shadow is not None:
            try:
                baseline_key, baseline_theta_control, baseline_metrics = _run_paired_shadow_baseline(
                    paired_shadow, group_cfg, ask_state=ask_state, ask_ctx=ask_ctx, window_end=paired_window_end
                )
                _attach_paired_delta_metrics(metrics, baseline_key, baseline_metrics)
            except Exception as e:
                metrics["paired_note"] = "paired_shadow_failed:" + type(e).__name__ + ":" + str(e)
                metrics["paired_baseline_key"] = _paired_baseline_key_for_group(group_cfg)
                metrics["paired_baseline_cost"] = np.nan
                metrics["paired_delta_cost"] = np.nan
                metrics["paired_delta_relative_pct"] = np.nan
        else:
            metrics["paired_note"] = "paired_delta_disabled"
        train_cost, train_feedback_mode, train_feedback_note = select_bo_training_feedback_cost(metrics, fac=fac, group_key=group_key)
        metrics["bo_training_cost"] = float(train_cost)
        metrics["bo_training_feedback_score"] = str(train_feedback_mode)
        metrics["bo_training_feedback_note"] = str(train_feedback_note)
        safe_info["current_candidate_cost"] = float(metrics.get("cost", np.nan))
        safe_info["current_deployed_cost"] = float(metrics.get("cost", np.nan))
        safe_info["current_train_cost"] = float(train_cost)
        safe_info["best_so_far_cost"] = -float(fac.agent.prev_best_value) if (fac.agent is not None and getattr(fac.agent, "prev_best_value", None) is not None) else safe_info.get("best_so_far_cost")
        safe_info["best_so_far_iter"] = getattr(fac.agent, "prev_best_iter", safe_info.get("best_so_far_iter")) if fac.agent is not None else safe_info.get("best_so_far_iter")
        feedback_confidence, confidence_parts = compute_feedback_confidence(metrics, group_cfg=group_cfg)
        metrics["feedback_confidence"] = float(feedback_confidence)
        metrics.update(confidence_parts)
        log_bo_training_feedback(fac, metrics, train_cost, train_feedback_mode, train_feedback_note)
        log_feedback_confidence(fac, feedback_confidence, confidence_parts, group_cfg=group_cfg)
        log_paired_delta_feedback(fac, metrics)

        if is_reduced and fac.agent is not None:
            state_arg = ask_state if getattr(fac.agent, "use_state_partition", False) else None
            context_arg = ask_ctx if getattr(fac.agent, "use_context", False) else None
            if _dual_is_enabled():
                # Provisional window tell: fast but noisy. A finalized cohort later replaces this sample.
                sample_id = f"w{int(i)}_c{metrics.get('cohort_id')}"
                agent_tell_with_feedback_meta(fac.agent, theta_control, train_cost, state=state_arg, context=context_arg, metrics=metrics, bo_iter=i, group_key=group_key, group_cfg=group_cfg, confidence=feedback_confidence, parts=confidence_parts)
                _dual_mark_last_sample(fac.agent, sample_id, train_cost, state=state_arg, context=context_arg)
                # Attach sample id to current cohort, so finalized precise feedback can update the same sample.
                try:
                    cur = fac.cohorts.get(metrics.get("cohort_id"))
                    if cur is not None:
                        cur.sample_id = sample_id
                except Exception:
                    pass
                applied = _dual_apply_pending_refinements(fac, fac.agent)
                fac.perf_log.setdefault("dual_window_provisional_cost", []).append(float(train_cost))
                fac.perf_log.setdefault("dual_refinement_applied_count", []).append(int(applied))
                fac.scheduler.update_beta(train_cost)
            elif not fac._use_cohort_feedback():
                agent_tell_with_feedback_meta(fac.agent, theta_control, train_cost, state=state_arg, context=context_arg, metrics=metrics, bo_iter=i, group_key=group_key, group_cfg=group_cfg, confidence=feedback_confidence, parts=confidence_parts)
                fac.scheduler.update_beta(train_cost)

        safe_info["best_so_far_cost"] = -float(fac.agent.prev_best_value) if (fac.agent is not None and getattr(fac.agent, "prev_best_value", None) is not None) else safe_info.get("best_so_far_cost")
        safe_info["best_so_far_iter"] = getattr(fac.agent, "prev_best_iter", safe_info.get("best_so_far_iter")) if fac.agent is not None else safe_info.get("best_so_far_iter")
        if fac.agent is not None and _is_cbo_method_key(group_key, group_cfg):
            tell_debug = dict(getattr(fac.agent, "last_debug_info", {}) or {})
            for post_tell_key in [
                "tr_update_mode", "tr_baseline_mean", "tr_current_mean", "tr_improve_pct",
                "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
                "cbo_tr_update_reason", "cbo_tr_radius_before_update", "cbo_tr_radius_after_update",
                "cbo_tr_success_count", "cbo_tr_failure_count", "predicted_cost", "actual_cost",
                "prediction_error", "surprise", "cost_gap_pct", "residual_trigger",
                "condition_trigger", "radius_min_stuck_count", "force_explore_countdown",
                "runtime_anchor_override",
            ]:
                if post_tell_key in tell_debug:
                    safe_info[post_tell_key] = tell_debug.get(post_tell_key)
            override_used = False
            try:
                override_used = (not _is_missing_value(safe_info.get("anchor_override_used"))) and int(float(safe_info.get("anchor_override_used"))) != 0
            except Exception:
                override_used = False
            override_mode = safe_info.get("runtime_anchor_override")
            if _is_missing_value(override_mode) and override_used:
                override_mode = safe_info.get("actual_tr_anchor_mode")
            if _is_missing_value(safe_info.get("runtime_anchor_override_reason")) and not _is_missing_value(override_mode):
                safe_info["runtime_anchor_override_reason"] = f"runtime_anchor_override={override_mode}"
            if _is_missing_value(safe_info.get("anchor_override_reason")) and override_used:
                safe_info["anchor_override_reason"] = f"runtime_anchor_override={override_mode}"
        if fac.agent is not None and _is_cbo_method_key(group_key, group_cfg):
            safe_info = _cbo_update_good_region_memory(fac.agent, i + 1, theta_control, float(metrics.get("cost", np.nan)), safe_info)
        for diag_key in [
            "selected_candidate_source", "selected_candidate_mu", "selected_candidate_sigma",
            "selected_candidate_acq", "selected_candidate_score", "selected_candidate_beta_eff",
            "selected_candidate_rank_by_score", "selected_candidate_rank_by_mu",
            "selected_candidate_rank_by_sigma", "selected_candidate_rank_by_acq",
            "best_mu_candidate_source", "best_acq_candidate_source", "num_candidates",
            "num_tr_candidates", "num_global_candidates", "cbo_macro_gate_mode",
            "macro_total_arrivals_norm", "macro_rt_ratio", "macro_batch_ratio",
            "macro_similarity_max", "macro_similarity_mean", "macro_similarity_p50",
            "macro_similarity_p90", "selected_macro_count", "selected_macro_mean_similarity",
            "selected_macro_min_similarity", "selected_macro_max_similarity", "macro_k",
            "macro_lengthscale_total", "macro_lengthscale_rt", "macro_lengthscale_batch",
            "macro_pool_count", "macro_pool_mean_similarity", "macro_pool_min_similarity",
            "macro_pool_max_similarity", "macro_pool_p50_similarity", "macro_pool_p90_similarity",
            "selected_from_macro_pool_count", "selected_outside_macro_pool_count",
            "macro_gate_fallback_used", "macro_gate_fallback_reason",
            "context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool",
            "cbo_select_mode", "cbo_topk", "cbo_select_temperature", "cbo_epsilon", "cbo_acq_beta",
            "cbo_acq_beta_mode", "beta_eff", "radius_norm", "radius_beta_component",
            "state_beta_boost_used", "state_beta_boost_reason", "actual_score_formula",
            "actual_beta_used", "service_guard_mode", "service_guard_available",
            "service_guard_penalty", "service_guard_reason",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_theta",
            "actual_tr_anchor_reason", "anchor_override_used", "anchor_override_reason",
            "anchor_fallback_used", "anchor_fallback_reason", "anchor_theta_distance_to_prev",
            "anchor_theta_distance_to_robust_elite", "anchor_theta_distance_to_context_best",
            "anchor_theta_distance_to_recent_best", "runtime_anchor_override_reason",
            "good_region_available", "good_region_best_iter", "good_region_best_rolling50_cost",
            "good_region_anchor_theta", "good_region_anchor_source",
            "distance_to_good_region_anchor", "current_vs_good_region_gap_pct",
            "good_region_guard_enabled", "good_region_guard_triggered", "good_region_guard_reason",
            "candidate_theta_6d", "deployed_theta_6d", "good_region_theta_6d",
            "good_region_cost", "good_region_iter", "good_region_window_start", "good_region_window_end",
            "distance_to_good_region", "candidate_selected_source", "deployed_source",
            "candidate_tr_radius", "candidate_beta_eff", "guard_fallback_type",
            "alpha_direct_fixed_theta_enabled", "alpha_direct_fixed_theta_6d",
            "tr_update_mode", "tr_baseline_mean", "tr_current_mean", "tr_improve_pct",
            "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
            "cbo_tr_radius_before_update",
            "predicted_cost", "actual_cost", "prediction_error", "surprise", "cost_gap_pct",
            "residual_trigger", "condition_trigger", "radius_min_stuck_count", "force_explore_countdown",
            "runtime_anchor_override", "cbo_tr_radius_after_update", "selected_reason",
            "cbo_warm_start_enabled", "cbo_warm_start_mode", "cbo_warm_start_loaded_rows",
            "cbo_warm_start_used_rows", "selected_warm_rows_count", "selected_local_rows_count",
            "cbo_warm_start_history_path",
            "cbo_history_denoise_mode", "cbo_history_denoise_k", "cbo_history_denoise_radius",
            "cbo_history_denoise_min_neighbors", "cbo_history_denoise_context_weight",
            "cbo_history_denoise_theta_weight", "cbo_history_denoise_stat",
            "cbo_history_denoise_apply_to", "cbo_history_denoise_raw_rows",
            "cbo_history_denoise_smoothed_rows", "cbo_history_denoise_unsmoothed_rows",
            "cbo_history_denoise_smoothed_ratio", "cbo_history_denoise_neighbor_count_mean",
            "cbo_history_denoise_neighbor_count_max", "cbo_history_denoise_abs_delta_mean",
            "cbo_history_denoise_abs_delta_max", "cbo_history_denoise_y_raw_mean",
            "cbo_history_denoise_y_used_mean",
            "cbo_history_outlier_filter_enabled", "cbo_history_outlier_strict_enabled",
            "cbo_history_outlier_raw_rows",
            "cbo_history_outlier_filtered_rows", "cbo_history_outlier_used_rows",
            "cbo_history_outlier_filter_ratio", "cbo_history_outlier_neighbor_count_mean",
            "cbo_history_outlier_neighbor_count_max", "cbo_history_outlier_theta_radius",
            "cbo_history_outlier_context_radius", "cbo_history_outlier_min_peers",
            "cbo_history_outlier_peer_count_mean", "cbo_history_outlier_peer_count_max",
            "cbo_history_outlier_protect_pressure", "cbo_history_outlier_pressure_quantile",
            "cbo_history_outlier_pressure_fields_available", "cbo_history_outlier_candidate_rows",
            "cbo_history_outlier_protected_rows", "cbo_history_outlier_filtered_rows_before_protection",
            "cbo_history_outlier_filtered_rows_after_protection", "cbo_history_outlier_protected_ratio",
            "cbo_history_outlier_pressure_delay_threshold", "cbo_history_outlier_pressure_backlog_threshold",
            "cbo_history_outlier_pressure_unfinished_threshold", "cbo_history_outlier_pressure_violation_threshold",
            "cbo_history_outlier_residual_mean",
            "cbo_history_outlier_residual_max", "cbo_history_outlier_threshold",
            "cbo_history_outlier_abs_threshold", "cbo_history_outlier_max_filter_ratio",
            "cbo_history_outlier_scale",
        ]:
            safe_info.setdefault(diag_key, None)
        for k, v in safe_info.items():
            if k == "candidate_diagnostic_rows":
                continue
            fac.perf_log.setdefault(k, []).append(v)
        fac.perf_log.setdefault("theta_control_deployed", []).append(list(theta_control))
        fac.perf_log.setdefault("theta_full_deployed", []).append(list(theta_full))
        fac.perf_log.setdefault("theta_full_feature_names", []).append(list(getattr(CFG, "FEATURE_NAMES", [])))
        fac.perf_log.setdefault("control_vector_meaning", []).append("deployed_full_theta")
        fac.perf_log.setdefault("scheduler_use_score_risk", []).append(bool(getattr(CFG, "USE_SCORE_RISK", True)))
        if group_cfg.get("control_mode") == "alpha_direct":
            alpha_control = _alpha_direct_expanded_6d(theta_control, group_cfg)
            while len(alpha_control) < 6:
                alpha_control.append(None)
            fac.perf_log.setdefault("alpha_direct_control_vector_6d", []).append(list(alpha_control[:6]))
            fac.perf_log.setdefault("alpha_direct_feature_names", []).append(list(globals().get("ALPHA_DIRECT_FEATURE_NAMES", [
                "Alpha_RT", "Alpha_Batch", "Alpha_AI", "W_Queue", "W_Risk_Scale", "Cloud_Gate"
            ])))
            fac.perf_log.setdefault("alpha_direct_alpha_rt", []).append(alpha_control[0])
            fac.perf_log.setdefault("alpha_direct_alpha_batch", []).append(alpha_control[1])
            fac.perf_log.setdefault("alpha_direct_alpha_ai", []).append(alpha_control[2])
            fac.perf_log.setdefault("alpha_direct_w_queue", []).append(alpha_control[3])
            fac.perf_log.setdefault("alpha_direct_w_risk_scale", []).append(alpha_control[4])
            fac.perf_log.setdefault("alpha_direct_cloud_gate", []).append(alpha_control[5])
        else:
            fac.perf_log.setdefault("alpha_direct_control_vector_6d", []).append(None)
            fac.perf_log.setdefault("alpha_direct_feature_names", []).append(None)
            fac.perf_log.setdefault("alpha_direct_alpha_rt", []).append(None)
            fac.perf_log.setdefault("alpha_direct_alpha_batch", []).append(None)
            fac.perf_log.setdefault("alpha_direct_alpha_ai", []).append(None)
            fac.perf_log.setdefault("alpha_direct_w_queue", []).append(None)
            fac.perf_log.setdefault("alpha_direct_w_risk_scale", []).append(None)
            fac.perf_log.setdefault("alpha_direct_cloud_gate", []).append(None)
        fac.perf_log.setdefault("agent_use_context", []).append(bool(getattr(fac.agent, "use_context", False)) if fac.agent is not None else False)

        # v6.2 runtime logging: append one elapsed-time sample per BO window.
        runtime_iter_elapsed = time.perf_counter() - runtime_iter_t0
        fac.perf_log.setdefault("runtime_iter_elapsed_sec", []).append(float(runtime_iter_elapsed))
        fac.perf_log.setdefault("runtime_iter_elapsed_min", []).append(float(runtime_iter_elapsed / 60.0))

        if (i + 1) % 20 == 0 or (i + 1) == int(CFG.BO_ITERATIONS):
            print(
                f"[SCHED-ROUND] iter={i + 1} "
                f"tradeoff_mode={(fac.perf_log.get('scheduler_tradeoff_mode', [getattr(CFG, 'SCHEDULER_TRADEOFF_MODE', 'legacy')]) or [None])[-1]} "
                f"score_norm_mode={(fac.perf_log.get('scheduler_score_norm_mode', [getattr(CFG, 'SCHEDULER_SCORE_NORM_MODE', 'legacy')]) or [None])[-1]} "
                f"alpha_last={(fac.perf_log.get('scheduler_alpha_last', [None]) or [None])[-1]} "
                f"selected_service_component={(fac.perf_log.get('selected_service_component_last', [None]) or [None])[-1]} "
                f"selected_energy_component={(fac.perf_log.get('selected_energy_component_last', [None]) or [None])[-1]} "
                f"selected_norm_l={(fac.perf_log.get('selected_norm_l_last', [None]) or [None])[-1]} "
                f"selected_norm_e={(fac.perf_log.get('selected_norm_e_last', [None]) or [None])[-1]} "
                f"selected_score={(fac.perf_log.get('selected_score_last', [None]) or [None])[-1]}",
                flush=True,
            )
            print(
                f"[DEPLOY-ROUND] method={group_key} iter={i + 1} "
                f"policy={safe_info.get('deploy_policy')} source={safe_info.get('deploy_source')} "
                f"explore_used={safe_info.get('explore_used')} "
                f"best_so_far_cost={safe_info.get('best_so_far_cost')} current_cost={safe_info.get('current_train_cost')}",
                flush=True,
            )
            if _is_cbo_method_key(group_key, group_cfg):
                print(
                    f"[CBO-STABILITY-ROUND] iter={i + 1} "
                    f"selected_total={safe_info.get('selected_total_count')} "
                    f"recent={safe_info.get('selected_recent_count')} "
                    f"context={safe_info.get('selected_context_count')} "
                    f"elite={safe_info.get('selected_elite_count')} "
                    f"diverse={safe_info.get('selected_diverse_count')} "
                    f"robust_score={safe_info.get('robust_incumbent_score')} "
                    f"robust_eval_count={safe_info.get('robust_incumbent_eval_count')} "
                    f"tr_radius={safe_info.get('cbo_tr_radius')} "
                    f"tr_update_mode={safe_info.get('tr_update_mode')} "
                    f"tr_before={safe_info.get('cbo_tr_radius_before_update')} "
                    f"tr_after={safe_info.get('cbo_tr_radius_after_update')} "
                    f"tr_signal={safe_info.get('tr_update_signal')} "
                    f"tr_reason={safe_info.get('cbo_tr_update_reason')} "
                    f"tr_mode={safe_info.get('cbo_tr_mode')} "
                    f"current_cost={safe_info.get('current_train_cost')} "
                    f"best_so_far_cost={safe_info.get('best_so_far_cost')}",
                    flush=True,
                )
                print(
                    f"[CBO-MACRO-ROUND] iter={i + 1} "
                    f"macro=[{safe_info.get('macro_total_arrivals_norm')},{safe_info.get('macro_rt_ratio')},{safe_info.get('macro_batch_ratio')}] "
                    f"macro_pool={safe_info.get('macro_pool_count')} "
                    f"selected_macro={safe_info.get('selected_macro_count')} "
                    f"sim_mean={safe_info.get('macro_similarity_mean')} "
                    f"sim_max={safe_info.get('macro_similarity_max')} "
                    f"selected_source={safe_info.get('selected_candidate_source', safe_info.get('selected_source'))} "
                    f"rank_mu={safe_info.get('selected_candidate_rank_by_mu', safe_info.get('selected_rank_by_mu'))} "
                    f"rank_acq={safe_info.get('selected_candidate_rank_by_acq', safe_info.get('selected_rank_by_acq'))} "
                    f"current_cost={safe_info.get('current_train_cost')}",
                    flush=True,
                )
                print(
                    f"[CBO-ACQ-ROUND] iter={i + 1} "
                    f"tr_radius={safe_info.get('cbo_tr_radius_after_update', safe_info.get('cbo_tr_radius'))} "
                    f"radius_norm={safe_info.get('radius_norm')} "
                    f"beta_eff={safe_info.get('beta_eff')} "
                    f"boost_used={safe_info.get('state_beta_boost_used')} "
                    f"boost_reason={safe_info.get('state_beta_boost_reason')} "
                    f"actual_anchor_source={safe_info.get('actual_tr_anchor_source')} "
                    f"selected_source={safe_info.get('selected_candidate_source', safe_info.get('selected_source'))} "
                    f"rank_mu={safe_info.get('selected_candidate_rank_by_mu', safe_info.get('selected_rank_by_mu'))} "
                    f"rank_score={safe_info.get('selected_candidate_rank_by_score', safe_info.get('selected_rank_by_score'))} "
                    f"rank_acq={safe_info.get('selected_candidate_rank_by_acq', safe_info.get('selected_rank_by_acq'))} "
                    f"current_cost={safe_info.get('current_train_cost')} "
                    f"surprise={safe_info.get('surprise')}",
                    flush=True,
                )
        if (i + 1) % 10 == 0:
            print(f"  [{group_cfg['label']} | SAFEBO={safe_info.get('deploy_policy')}/{safe_info.get('deploy_source')} | feedback={_dual_feedback_mode()}] Iteration {i + 1}/{CFG.BO_ITERATIONS}")

    if fac._use_cohort_feedback() and bool(getattr(CFG, "COHORT_FORCE_FINALIZE_AT_RUN_END", True)):
        fac._finalize_ready_cohorts(fac.current_time, force=True, reason="run_end")
        if fac.agent is not None and _dual_is_enabled():
            _dual_apply_pending_refinements(fac, fac.agent)
    # v6.2 runtime logging: total method runtime and compact dimension metadata.
    runtime_group_elapsed = time.perf_counter() - runtime_group_t0
    agent_kwargs = group_cfg.get("agent_kwargs", {}) or {}
    try:
        control_dim = int(len(group_cfg.get("fixed_theta", []))) if group_cfg.get("fixed_theta") is not None else int(agent_kwargs.get("dim", 0) or 0)
    except Exception:
        control_dim = int(agent_kwargs.get("dim", 0) or 0)
    try:
        context_dim = int(agent_kwargs.get("context_dim", 0) or 0)
    except Exception:
        context_dim = 0
    total_model_dim = int(control_dim + context_dim)
    fac.perf_log["runtime_group_elapsed_sec"] = [float(runtime_group_elapsed)]
    fac.perf_log["runtime_group_elapsed_min"] = [float(runtime_group_elapsed / 60.0)]
    fac.perf_log["runtime_group_sec_per_iter"] = [float(runtime_group_elapsed / max(1, int(CFG.BO_ITERATIONS)))]
    fac.perf_log["runtime_control_dim"] = [int(control_dim)]
    fac.perf_log["runtime_context_dim"] = [int(context_dim)]
    fac.perf_log["runtime_total_model_dim"] = [int(total_model_dim)]
    fac.perf_log["cohort_feedback_debug_rows"] = list(getattr(fac, "cohort_feedback_rows", []))
    if method_scheduler_tradeoff_mode:
        CFG.SCHEDULER_TRADEOFF_MODE = old_scheduler_tradeoff_mode
    if method_use_score_risk is not None:
        CFG.USE_SCORE_RISK = old_use_score_risk
    return fac.perf_log
