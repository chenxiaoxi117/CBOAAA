#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 4447-5508.
# Log flattening, allocation diagnostics, metric summaries, and plots.

def _safe_json(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _log_get_safe(log, key, i, default=None):
    vals = log.get(key, [])
    if isinstance(vals, (list, tuple, np.ndarray)) and i < len(vals):
        return vals[i]
    return default


def _log_get_non_nan(log, key, i, default=None):
    val = _log_get_safe(log, key, i, default)
    try:
        if isinstance(val, float) and np.isnan(val):
            return default
    except Exception:
        pass
    return val


def _log_get_ffill(log, key, i, default=None):
    vals = log.get(key, [])
    if isinstance(vals, (list, tuple, np.ndarray)) and vals:
        idx = min(i, len(vals) - 1)
        for j in range(idx, -1, -1):
            val = vals[j]
            try:
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    continue
            except Exception:
                pass
            return val
    return default


def _is_missing_value(v):
    if v is None:
        return True
    try:
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _first_present(*values, default=None):
    for v in values:
        if not _is_missing_value(v):
            return v
    return default


def _theta_feature_value(theta_full, feature_names, name, default=None):
    if not isinstance(theta_full, (list, tuple, np.ndarray)):
        return default
    try:
        names = list(feature_names or getattr(CFG, "FEATURE_NAMES", []))
        idx = names.index(name)
        if idx < len(theta_full):
            return float(theta_full[idx])
    except Exception:
        pass
    return default


def _global_energy_scale_value(w_rt_energy, w_batch_energy, w_ai_energy, default=None):
    vals = [w_rt_energy, w_batch_energy, w_ai_energy]
    if any(_is_missing_value(v) for v in vals):
        return default
    try:
        vals = [float(v) for v in vals]
        if max(vals) - min(vals) <= 1e-9:
            return vals[0]
    except Exception:
        pass
    return default


def group_log_to_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    alphas = log.get("alpha", [])
    controls = log.get("control_vector", [])
    allocs = log.get("alloc", [])
    bsf_reward = best_so_far(log.get("reward", []))
    rows = []
    for i in range(n):
        theta = alphas[i] if i < len(alphas) else [None] * CFG.DIM_THETA
        theta = list(theta) if isinstance(theta, (list, tuple, np.ndarray)) else [None] * CFG.DIM_THETA
        while len(theta) < CFG.DIM_THETA:
            theta.append(None)
        control = controls[i] if i < len(controls) else []
        alloc = allocs[i] if i < len(allocs) else []
        selected_candidate_source = _first_present(
            _log_get_ffill(log, "selected_candidate_source", i, None),
            _log_get_ffill(log, "selected_source", i, None),
            _log_get_ffill(log, "best_mu_candidate_source", i, None),
            _log_get_ffill(log, "best_acq_candidate_source", i, None),
        )
        if _is_missing_value(selected_candidate_source):
            deploy_src = str(_log_get_ffill(log, "deploy_source", i, "") or "")
            if "greedy_posterior_mean" in deploy_src:
                selected_candidate_source = "posterior_mean_candidate"
            elif "acquisition" in deploy_src:
                selected_candidate_source = "acquisition_candidate"
            elif deploy_src:
                selected_candidate_source = deploy_src
        theta_full_deployed = _log_get_safe(log, "theta_full_deployed", i, control)
        alpha_direct_control = _log_get_safe(log, "alpha_direct_control_vector_6d", i, None)
        if _is_missing_value(alpha_direct_control):
            theta_control = _log_get_safe(log, "theta_control_deployed", i, None)
            if isinstance(theta_control, (list, tuple, np.ndarray)) and len(theta_control) == 6:
                alpha_direct_control = list(theta_control)
            elif isinstance(theta_full_deployed, (list, tuple, np.ndarray)) and len(theta_full_deployed) >= 11:
                alpha_direct_control = [
                    theta_full_deployed[0], theta_full_deployed[1], theta_full_deployed[2],
                    theta_full_deployed[6], theta_full_deployed[7], theta_full_deployed[10],
                ]
        if not isinstance(alpha_direct_control, (list, tuple, np.ndarray)):
            alpha_direct_control = [None] * 6
        alpha_direct_control = list(alpha_direct_control)[:6]
        while len(alpha_direct_control) < 6:
            alpha_direct_control.append(None)
        alpha_feature_names = _log_get_safe(log, "alpha_direct_feature_names", i, globals().get("ALPHA_DIRECT_FEATURE_NAMES", [
            "Alpha_RT", "Alpha_Batch", "Alpha_AI", "W_Queue", "W_Risk_Scale", "Cloud_Gate"
        ]))
        theta_full_feature_names = _log_get_safe(log, "theta_full_feature_names", i, list(getattr(CFG, "FEATURE_NAMES", [])))
        w_rt_latency = _first_present(
            _log_get_safe(log, "W_RT_Latency_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_RT_Latency"),
        )
        w_batch_latency = _first_present(
            _log_get_safe(log, "W_Batch_Latency_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_Batch_Latency"),
        )
        w_ai_latency = _first_present(
            _log_get_safe(log, "W_AI_Latency_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_AI_Latency"),
        )
        w_queue_value = _first_present(
            _log_get_safe(log, "W_Queue_last", i, None),
            _log_get_safe(log, "w_queue", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_Queue"),
        )
        w_risk_scale_value = _first_present(
            _log_get_safe(log, "W_Risk_Scale_last", i, None),
            _log_get_safe(log, "w_risk_scale", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_Risk_Scale"),
        )
        cloud_gate_value = _first_present(
            _log_get_safe(log, "Cloud_Gate_last", i, None),
            _log_get_safe(log, "cloud_gate", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "Cloud_Gate"),
        )
        w_rt_energy_last = _first_present(
            _log_get_safe(log, "W_RT_Energy_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_RT_Energy"),
        )
        w_batch_energy_last = _first_present(
            _log_get_safe(log, "W_Batch_Energy_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_Batch_Energy"),
        )
        w_ai_energy_last = _first_present(
            _log_get_safe(log, "W_AI_Energy_last", i, None),
            _theta_feature_value(theta_full_deployed, theta_full_feature_names, "W_AI_Energy"),
        )
        w_energy_scale_last = _first_present(
            _log_get_safe(log, "W_Energy_Scale_last", i, None),
            _global_energy_scale_value(w_rt_energy_last, w_batch_energy_last, w_ai_energy_last),
        )
        macro_gate_mode_row = _log_get_non_nan(log, "cbo_macro_gate_mode", i, str(getattr(CFG, "CBO_MACRO_GATE_MODE", "off")))
        default_source_pool = "macro_pool" if str(macro_gate_mode_row).strip().lower() == "hierarchical" else None
        actual_anchor_reason_row = _log_get_ffill(log, "actual_tr_anchor_reason", i, None)
        runtime_anchor_override_row = _log_get_ffill(log, "runtime_anchor_override", i, None)
        anchor_override_used_row = _log_get_ffill(log, "anchor_override_used", i, None)
        anchor_override_reason_row = _log_get_ffill(log, "anchor_override_reason", i, None)
        runtime_anchor_override_reason_row = _log_get_ffill(log, "runtime_anchor_override_reason", i, None)
        try:
            anchor_override_used_bool = (not _is_missing_value(anchor_override_used_row)) and int(float(anchor_override_used_row)) != 0
        except Exception:
            anchor_override_used_bool = False
        if _is_missing_value(runtime_anchor_override_reason_row):
            reason_text = str(actual_anchor_reason_row or "")
            if (not _is_missing_value(runtime_anchor_override_row)) and (anchor_override_used_bool or "runtime_override" in reason_text):
                runtime_anchor_override_reason_row = f"runtime_anchor_override={runtime_anchor_override_row}"
        if _is_missing_value(anchor_override_reason_row) and not _is_missing_value(runtime_anchor_override_reason_row):
            anchor_override_reason_row = runtime_anchor_override_reason_row
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Dynamic_Mode_动态模式": _log_get_safe(log, "dynamic_mode", i, None),
            "Dynamic_History_Mode_动态历史模式": _log_get_safe(log, "dynamic_history_mode", i, None),
            "Global_Iteration_全局轮次": _log_get_safe(log, "dynamic_global_iter", i, i + 1),
            "Phase_ID_阶段ID": _log_get_safe(log, "dynamic_phase_id", i, None),
            "Phase_Name_阶段名称": _log_get_safe(log, "dynamic_phase_name", i, None),
            "Phase_Iteration_阶段内轮次": _log_get_safe(log, "dynamic_phase_iter", i, None),
            "Phase_Lambda_阶段到达率": _log_get_safe(log, "dynamic_phase_lambda", i, None),
            "Phase_RT_Prob_阶段RT比例": _log_get_safe(log, "dynamic_phase_rt_prob", i, None),
            "Phase_Batch_Prob_阶段Batch比例": _log_get_safe(log, "dynamic_phase_batch_prob", i, None),
            "Phase_AI_Prob_阶段AI比例": _log_get_safe(log, "dynamic_phase_ai_prob", i, None),
            "Time_时间": log.get("time", [None] * n)[i],
            "Reward_奖励": log.get("reward", [None] * n)[i],
            "Eval_Cost_最终评估Cost": log.get("eval_cost", [None] * n)[i] if i < len(log.get("eval_cost", [])) else (-_safe_float(log.get("reward", [np.nan] * n)[i])),
            "BO_Training_Cost_BO训练Cost": log.get("bo_training_cost", [None] * n)[i] if i < len(log.get("bo_training_cost", [])) else None,
            "BO_Training_Feedback_训练反馈模式": log.get("bo_training_feedback_score", [None] * n)[i] if i < len(log.get("bo_training_feedback_score", [])) else None,
            "BO_Training_Feedback_Note_训练反馈说明": log.get("bo_training_feedback_note", [None] * n)[i] if i < len(log.get("bo_training_feedback_note", [])) else None,
            "Paired_Baseline_Key_配对基线": log.get("paired_baseline_key", [None] * n)[i] if i < len(log.get("paired_baseline_key", [])) else None,
            "Paired_Baseline_Cost_配对基线Cost": log.get("paired_baseline_cost", [None] * n)[i] if i < len(log.get("paired_baseline_cost", [])) else None,
            "Paired_Delta_Cost_相对基线DeltaCost": log.get("paired_delta_cost", [None] * n)[i] if i < len(log.get("paired_delta_cost", [])) else None,
            "Paired_Delta_Relative_Pct_相对基线百分比": log.get("paired_delta_relative_pct", [None] * n)[i] if i < len(log.get("paired_delta_relative_pct", [])) else None,
            "Paired_Note_配对说明": log.get("paired_note", [None] * n)[i] if i < len(log.get("paired_note", [])) else None,
            "Refactor_Version_重构版本": log.get("refactor_version", [None] * n)[i] if i < len(log.get("refactor_version", [])) else None,
            "Feedback_Confidence_反馈可信度": log.get("feedback_confidence", [None] * n)[i] if i < len(log.get("feedback_confidence", [])) else None,
            "Confidence_Completion_Ratio_可信度完成率": log.get("confidence_completion_ratio", [None] * n)[i] if i < len(log.get("confidence_completion_ratio", [])) else None,
            "Confidence_Unfinished_Ratio_可信度未完成比例": log.get("confidence_unfinished_ratio", [None] * n)[i] if i < len(log.get("confidence_unfinished_ratio", [])) else None,
            "Confidence_Task_Mix_L1_任务比例偏移": log.get("confidence_task_mix_l1", [None] * n)[i] if i < len(log.get("confidence_task_mix_l1", [])) else None,
            "BO_History_Mode_历史模式": _log_get_non_nan(log, "bo_history_mode", i, str(getattr(CFG, "BO_HISTORY_MODE", "recent"))),
            "BO_Recent_Window_最近历史窗口": log.get("bo_recent_window", [None] * n)[i] if i < len(log.get("bo_recent_window", [])) else None,
            "effective_history_mode": _log_get_non_nan(log, "effective_history_mode", i, str(getattr(CFG, "BO_HISTORY_MODE", "recent"))),
            "effective_recent_window": log.get("effective_recent_window", [None] * n)[i] if i < len(log.get("effective_recent_window", [])) else None,
            "history_override_source": _log_get_non_nan(log, "history_override_source", i, "unknown"),
            "bo_train_sample_count": log.get("bo_train_sample_count", [None] * n)[i] if i < len(log.get("bo_train_sample_count", [])) else None,
            "BO_Confidence_Min_最小可信度": log.get("bo_confidence_min", [None] * n)[i] if i < len(log.get("bo_confidence_min", [])) else None,
            "Best_So_Far_Reward_历史最优奖励": bsf_reward[i] if i < len(bsf_reward) else None,
            "deploy_policy": _log_get_safe(log, "deploy_policy", i, None),
            "deploy_source": _log_get_safe(log, "deploy_source", i, None),
            "explore_used": _log_get_safe(log, "explore_used", i, None),
            "incumbent_available": _log_get_safe(log, "incumbent_available", i, None),
            "incumbent_cost": _log_get_safe(log, "incumbent_cost", i, None),
            "current_candidate_cost": _log_get_safe(log, "current_candidate_cost", i, None),
            "current_train_cost": _log_get_safe(log, "current_train_cost", i, None),
            "best_so_far_cost": _log_get_safe(log, "best_so_far_cost", i, None),
            "best_so_far_iter": _log_get_safe(log, "best_so_far_iter", i, None),
            "used_theta_source": _log_get_safe(log, "used_theta_source", i, None),
            "posterior_mu": _log_get_safe(log, "posterior_mu", i, None),
            "posterior_sigma": _log_get_safe(log, "posterior_sigma", i, None),
            "candidate_count_safe": _log_get_safe(log, "candidate_count_safe", i, None),
            "history_select_mode": _log_get_non_nan(log, "history_select_mode", i, str(getattr(CFG, "CBO_HISTORY_SELECT_MODE", "recent"))),
            "selected_recent_count": _log_get_ffill(log, "selected_recent_count", i, None),
            "selected_macro_count": _log_get_ffill(log, "selected_macro_count", i, None),
            "selected_context_count": _log_get_ffill(log, "selected_context_count", i, None),
            "selected_elite_count": _log_get_ffill(log, "selected_elite_count", i, None),
            "selected_diverse_count": _log_get_ffill(log, "selected_diverse_count", i, None),
            "selected_total_count": _log_get_ffill(log, "selected_total_count", i, None),
            "external_gate_mode": _log_get_ffill(log, "external_gate_mode", i, None),
            "external_gate_enabled": _log_get_ffill(log, "external_gate_enabled", i, None),
            "external_gate_raw_count": _log_get_ffill(log, "external_gate_raw_count", i, None),
            "external_gate_passed_count": _log_get_ffill(log, "external_gate_passed_count", i, None),
            "external_gate_selected_count": _log_get_ffill(log, "external_gate_selected_count", i, None),
            "external_gate_fallback_used": _log_get_ffill(log, "external_gate_fallback_used", i, None),
            "external_gate_fallback_reason": _log_get_ffill(log, "external_gate_fallback_reason", i, None),
            "external_similarity_max": _log_get_ffill(log, "external_similarity_max", i, None),
            "external_similarity_mean": _log_get_ffill(log, "external_similarity_mean", i, None),
            "selected_external_similarity_mean": _log_get_ffill(log, "selected_external_similarity_mean", i, None),
            "state_kernel_enabled": _log_get_ffill(log, "state_kernel_enabled", i, None),
            "state_kernel_topk": _log_get_ffill(log, "state_kernel_topk", i, None),
            "state_kernel_min_rows": _log_get_ffill(log, "state_kernel_min_rows", i, None),
            "state_kernel_recent_keep": _log_get_ffill(log, "state_kernel_recent_keep", i, None),
            "state_kernel_threshold": _log_get_ffill(log, "state_kernel_threshold", i, None),
            "state_kernel_raw_records": _log_get_ffill(log, "state_kernel_raw_records", i, None),
            "state_kernel_passed_count": _log_get_ffill(log, "state_kernel_passed_count", i, None),
            "state_kernel_rejected_count": _log_get_ffill(log, "state_kernel_rejected_count", i, None),
            "state_kernel_reject_reason_counts": _log_get_ffill(log, "state_kernel_reject_reason_counts", i, None),
            "state_kernel_selected_count": _log_get_ffill(log, "state_kernel_selected_count", i, None),
            "state_kernel_fallback_used": _log_get_ffill(log, "state_kernel_fallback_used", i, None),
            "state_kernel_fallback_reason": _log_get_ffill(log, "state_kernel_fallback_reason", i, None),
            "state_kernel_similarity_max": _log_get_ffill(log, "state_kernel_similarity_max", i, None),
            "state_kernel_similarity_mean": _log_get_ffill(log, "state_kernel_similarity_mean", i, None),
            "state_kernel_similarity_p50": _log_get_ffill(log, "state_kernel_similarity_p50", i, None),
            "state_kernel_selected_similarity_mean": _log_get_ffill(log, "state_kernel_selected_similarity_mean", i, None),
            "state_kernel_selected_similarity_min": _log_get_ffill(log, "state_kernel_selected_similarity_min", i, None),
            "state_kernel_selected_similarity_max": _log_get_ffill(log, "state_kernel_selected_similarity_max", i, None),
            "state_kernel_current_total_norm": _log_get_ffill(log, "state_kernel_current_total_norm", i, None),
            "state_kernel_current_rt_ratio": _log_get_ffill(log, "state_kernel_current_rt_ratio", i, None),
            "state_kernel_current_batch_ratio": _log_get_ffill(log, "state_kernel_current_batch_ratio", i, None),
            "state_kernel_current_backlog_norm": _log_get_ffill(log, "state_kernel_current_backlog_norm", i, None),
            "state_kernel_current_unfinished_rate": _log_get_ffill(log, "state_kernel_current_unfinished_rate", i, None),
            "state_kernel_current_backlog_trend": _log_get_ffill(log, "state_kernel_current_backlog_trend", i, None),
            "state_kernel_current_unfinished_trend": _log_get_ffill(log, "state_kernel_current_unfinished_trend", i, None),
            "state_kernel_current_cost_trend": _log_get_ffill(log, "state_kernel_current_cost_trend", i, None),
            "state_kernel_current_delay_trend": _log_get_ffill(log, "state_kernel_current_delay_trend", i, None),
            "state_kernel_rate_gain": _log_get_ffill(log, "state_kernel_rate_gain", i, None),
            "state_kernel_rate_power": _log_get_ffill(log, "state_kernel_rate_power", i, None),
            "state_kernel_max_rate_dist": _log_get_ffill(log, "state_kernel_max_rate_dist", i, None),
            "state_kernel_rate_distance_mean": _log_get_ffill(log, "state_kernel_rate_distance_mean", i, None),
            "state_kernel_rate_distance_p50": _log_get_ffill(log, "state_kernel_rate_distance_p50", i, None),
            "state_kernel_rate_distance_max": _log_get_ffill(log, "state_kernel_rate_distance_max", i, None),
            "state_kernel_selected_rate_distance_mean": _log_get_ffill(log, "state_kernel_selected_rate_distance_mean", i, None),
            "state_kernel_rate_sign_veto": _log_get_ffill(log, "state_kernel_rate_sign_veto", i, None),
            "state_kernel_selected_phase_counts": _log_get_ffill(log, "state_kernel_selected_phase_counts", i, None),
            "cbo_warm_start_enabled": _log_get_ffill(log, "cbo_warm_start_enabled", i, False),
            "cbo_warm_start_mode": _log_get_ffill(log, "cbo_warm_start_mode", i, "none"),
            "cbo_warm_start_loaded_rows": _log_get_ffill(log, "cbo_warm_start_loaded_rows", i, 0),
            "cbo_warm_start_used_rows": _log_get_ffill(log, "cbo_warm_start_used_rows", i, 0),
            "selected_warm_rows_count": _log_get_ffill(log, "selected_warm_rows_count", i, 0),
            "selected_local_rows_count": _log_get_ffill(log, "selected_local_rows_count", i, None),
            "cbo_warm_start_history_path": _log_get_ffill(log, "cbo_warm_start_history_path", i, None),
            "cbo_history_denoise_mode": _log_get_ffill(log, "cbo_history_denoise_mode", i, str(getattr(CFG, "CBO_HISTORY_DENOISE_MODE", "off"))),
            "cbo_history_denoise_k": _log_get_ffill(log, "cbo_history_denoise_k", i, int(getattr(CFG, "CBO_HISTORY_DENOISE_K", 7))),
            "cbo_history_denoise_radius": _log_get_ffill(log, "cbo_history_denoise_radius", i, float(getattr(CFG, "CBO_HISTORY_DENOISE_RADIUS", 0.12))),
            "cbo_history_denoise_min_neighbors": _log_get_ffill(log, "cbo_history_denoise_min_neighbors", i, int(getattr(CFG, "CBO_HISTORY_DENOISE_MIN_NEIGHBORS", 3))),
            "cbo_history_denoise_context_weight": _log_get_ffill(log, "cbo_history_denoise_context_weight", i, float(getattr(CFG, "CBO_HISTORY_DENOISE_CONTEXT_WEIGHT", 1.0))),
            "cbo_history_denoise_theta_weight": _log_get_ffill(log, "cbo_history_denoise_theta_weight", i, float(getattr(CFG, "CBO_HISTORY_DENOISE_THETA_WEIGHT", 1.0))),
            "cbo_history_denoise_stat": _log_get_ffill(log, "cbo_history_denoise_stat", i, str(getattr(CFG, "CBO_HISTORY_DENOISE_STAT", "median"))),
            "cbo_history_denoise_apply_to": _log_get_ffill(log, "cbo_history_denoise_apply_to", i, str(getattr(CFG, "CBO_HISTORY_DENOISE_APPLY_TO", "all"))),
            "cbo_history_denoise_raw_rows": _log_get_ffill(log, "cbo_history_denoise_raw_rows", i, 0),
            "cbo_history_denoise_smoothed_rows": _log_get_ffill(log, "cbo_history_denoise_smoothed_rows", i, 0),
            "cbo_history_denoise_unsmoothed_rows": _log_get_ffill(log, "cbo_history_denoise_unsmoothed_rows", i, 0),
            "cbo_history_denoise_smoothed_ratio": _log_get_ffill(log, "cbo_history_denoise_smoothed_ratio", i, 0.0),
            "cbo_history_denoise_neighbor_count_mean": _log_get_ffill(log, "cbo_history_denoise_neighbor_count_mean", i, 0.0),
            "cbo_history_denoise_neighbor_count_max": _log_get_ffill(log, "cbo_history_denoise_neighbor_count_max", i, 0),
            "cbo_history_denoise_abs_delta_mean": _log_get_ffill(log, "cbo_history_denoise_abs_delta_mean", i, 0.0),
            "cbo_history_denoise_abs_delta_max": _log_get_ffill(log, "cbo_history_denoise_abs_delta_max", i, 0.0),
            "cbo_history_denoise_y_raw_mean": _log_get_ffill(log, "cbo_history_denoise_y_raw_mean", i, 0.0),
            "cbo_history_denoise_y_used_mean": _log_get_ffill(log, "cbo_history_denoise_y_used_mean", i, 0.0),
            "cbo_history_outlier_filter_enabled": _log_get_ffill(log, "cbo_history_outlier_filter_enabled", i, int(str(getattr(CFG, "CBO_HISTORY_DENOISE_MODE", "off")) in {"local_outlier_filter", "strict_local_outlier_filter"})),
            "cbo_history_outlier_strict_enabled": _log_get_ffill(log, "cbo_history_outlier_strict_enabled", i, int(str(getattr(CFG, "CBO_HISTORY_DENOISE_MODE", "off")) == "strict_local_outlier_filter")),
            "cbo_history_outlier_raw_rows": _log_get_ffill(log, "cbo_history_outlier_raw_rows", i, 0),
            "cbo_history_outlier_filtered_rows": _log_get_ffill(log, "cbo_history_outlier_filtered_rows", i, 0),
            "cbo_history_outlier_used_rows": _log_get_ffill(log, "cbo_history_outlier_used_rows", i, 0),
            "cbo_history_outlier_filter_ratio": _log_get_ffill(log, "cbo_history_outlier_filter_ratio", i, 0.0),
            "cbo_history_outlier_neighbor_count_mean": _log_get_ffill(log, "cbo_history_outlier_neighbor_count_mean", i, 0.0),
            "cbo_history_outlier_neighbor_count_max": _log_get_ffill(log, "cbo_history_outlier_neighbor_count_max", i, 0),
            "cbo_history_outlier_theta_radius": _log_get_ffill(log, "cbo_history_outlier_theta_radius", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_THETA_RADIUS", 0.12))),
            "cbo_history_outlier_context_radius": _log_get_ffill(log, "cbo_history_outlier_context_radius", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_CONTEXT_RADIUS", 0.50))),
            "cbo_history_outlier_min_peers": _log_get_ffill(log, "cbo_history_outlier_min_peers", i, int(getattr(CFG, "CBO_HISTORY_OUTLIER_MIN_PEERS", 3))),
            "cbo_history_outlier_peer_count_mean": _log_get_ffill(log, "cbo_history_outlier_peer_count_mean", i, 0.0),
            "cbo_history_outlier_peer_count_max": _log_get_ffill(log, "cbo_history_outlier_peer_count_max", i, 0),
            "cbo_history_outlier_protect_pressure": _log_get_ffill(log, "cbo_history_outlier_protect_pressure", i, int(bool(getattr(CFG, "CBO_HISTORY_OUTLIER_PROTECT_PRESSURE", False)))),
            "cbo_history_outlier_pressure_quantile": _log_get_ffill(log, "cbo_history_outlier_pressure_quantile", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_PRESSURE_QUANTILE", 0.75))),
            "cbo_history_outlier_pressure_fields_available": _log_get_ffill(log, "cbo_history_outlier_pressure_fields_available", i, ""),
            "cbo_history_outlier_candidate_rows": _log_get_ffill(log, "cbo_history_outlier_candidate_rows", i, 0),
            "cbo_history_outlier_protected_rows": _log_get_ffill(log, "cbo_history_outlier_protected_rows", i, 0),
            "cbo_history_outlier_filtered_rows_before_protection": _log_get_ffill(log, "cbo_history_outlier_filtered_rows_before_protection", i, 0),
            "cbo_history_outlier_filtered_rows_after_protection": _log_get_ffill(log, "cbo_history_outlier_filtered_rows_after_protection", i, 0),
            "cbo_history_outlier_protected_ratio": _log_get_ffill(log, "cbo_history_outlier_protected_ratio", i, 0.0),
            "cbo_history_outlier_pressure_delay_threshold": _log_get_ffill(log, "cbo_history_outlier_pressure_delay_threshold", i, None),
            "cbo_history_outlier_pressure_backlog_threshold": _log_get_ffill(log, "cbo_history_outlier_pressure_backlog_threshold", i, None),
            "cbo_history_outlier_pressure_unfinished_threshold": _log_get_ffill(log, "cbo_history_outlier_pressure_unfinished_threshold", i, None),
            "cbo_history_outlier_pressure_violation_threshold": _log_get_ffill(log, "cbo_history_outlier_pressure_violation_threshold", i, None),
            "cbo_history_outlier_residual_mean": _log_get_ffill(log, "cbo_history_outlier_residual_mean", i, 0.0),
            "cbo_history_outlier_residual_max": _log_get_ffill(log, "cbo_history_outlier_residual_max", i, 0.0),
            "cbo_history_outlier_threshold": _log_get_ffill(log, "cbo_history_outlier_threshold", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_THRESHOLD", 3.0))),
            "cbo_history_outlier_abs_threshold": _log_get_ffill(log, "cbo_history_outlier_abs_threshold", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_ABS_THRESHOLD", 500.0))),
            "cbo_history_outlier_max_filter_ratio": _log_get_ffill(log, "cbo_history_outlier_max_filter_ratio", i, float(getattr(CFG, "CBO_HISTORY_OUTLIER_MAX_FILTER_RATIO", 0.2))),
            "cbo_history_outlier_scale": _log_get_ffill(log, "cbo_history_outlier_scale", i, str(getattr(CFG, "CBO_HISTORY_OUTLIER_SCALE", "mad"))),
            "cbo_macro_gate_mode": macro_gate_mode_row,
            "macro_total_arrivals_norm": _log_get_ffill(log, "macro_total_arrivals_norm", i, None),
            "macro_rt_ratio": _log_get_ffill(log, "macro_rt_ratio", i, None),
            "macro_batch_ratio": _log_get_ffill(log, "macro_batch_ratio", i, None),
            "macro_similarity_max": _log_get_ffill(log, "macro_similarity_max", i, None),
            "macro_similarity_mean": _log_get_ffill(log, "macro_similarity_mean", i, None),
            "macro_similarity_p50": _log_get_ffill(log, "macro_similarity_p50", i, None),
            "macro_similarity_p90": _log_get_ffill(log, "macro_similarity_p90", i, None),
            "selected_macro_mean_similarity": _log_get_ffill(log, "selected_macro_mean_similarity", i, None),
            "selected_macro_min_similarity": _log_get_ffill(log, "selected_macro_min_similarity", i, None),
            "selected_macro_max_similarity": _log_get_ffill(log, "selected_macro_max_similarity", i, None),
            "macro_pool_count": _log_get_ffill(log, "macro_pool_count", i, None),
            "macro_pool_mean_similarity": _log_get_ffill(log, "macro_pool_mean_similarity", i, None),
            "macro_pool_min_similarity": _log_get_ffill(log, "macro_pool_min_similarity", i, None),
            "macro_pool_max_similarity": _log_get_ffill(log, "macro_pool_max_similarity", i, None),
            "macro_pool_p50_similarity": _log_get_ffill(log, "macro_pool_p50_similarity", i, None),
            "macro_pool_p90_similarity": _log_get_ffill(log, "macro_pool_p90_similarity", i, None),
            "selected_from_macro_pool_count": _log_get_ffill(log, "selected_from_macro_pool_count", i, None),
            "selected_outside_macro_pool_count": _log_get_ffill(log, "selected_outside_macro_pool_count", i, None),
            "macro_gate_fallback_used": _log_get_ffill(log, "macro_gate_fallback_used", i, None),
            "macro_gate_fallback_reason": _log_get_ffill(log, "macro_gate_fallback_reason", i, None),
            "context_selection_source_pool": _first_present(_log_get_ffill(log, "context_selection_source_pool", i, None), default_source_pool),
            "elite_selection_source_pool": _first_present(_log_get_ffill(log, "elite_selection_source_pool", i, None), default_source_pool),
            "tr_anchor_source_pool": _first_present(_log_get_ffill(log, "tr_anchor_source_pool", i, None), default_source_pool),
            "macro_k": _log_get_ffill(log, "macro_k", i, None),
            "macro_lengthscale_total": _log_get_ffill(log, "macro_lengthscale_total", i, None),
            "macro_lengthscale_rt": _log_get_ffill(log, "macro_lengthscale_rt", i, None),
            "macro_lengthscale_batch": _log_get_ffill(log, "macro_lengthscale_batch", i, None),
            "context_similarity_max": _log_get_ffill(log, "context_similarity_max", i, None),
            "context_similarity_mean": _log_get_ffill(log, "context_similarity_mean", i, None),
            "elite_best_robust_score": _log_get_ffill(log, "elite_best_robust_score", i, None),
            "elite_best_eval_count": _log_get_ffill(log, "elite_best_eval_count", i, None),
            "elite_best_mean_cost": _log_get_ffill(log, "elite_best_mean_cost", i, None),
            "elite_best_std_cost": _log_get_ffill(log, "elite_best_std_cost", i, None),
            "robust_incumbent_available": _log_get_ffill(log, "robust_incumbent_available", i, None),
            "robust_incumbent_score": _log_get_ffill(log, "robust_incumbent_score", i, None),
            "robust_incumbent_eval_count": _log_get_ffill(log, "robust_incumbent_eval_count", i, None),
            "robust_incumbent_mean_cost": _log_get_ffill(log, "robust_incumbent_mean_cost", i, None),
            "robust_incumbent_std_cost": _log_get_ffill(log, "robust_incumbent_std_cost", i, None),
            "robust_incumbent_context_similarity": _log_get_ffill(log, "robust_incumbent_context_similarity", i, None),
            "robust_incumbent_theta": _safe_json(_log_get_ffill(log, "robust_incumbent_theta", i, None)),
            "robust_incumbent_used": _log_get_ffill(log, "robust_incumbent_used", i, None),
            "robust_incumbent_reason": _log_get_ffill(log, "robust_incumbent_reason", i, None),
            "cbo_tr_mode": _log_get_non_nan(log, "cbo_tr_mode", i, str(getattr(CFG, "CBO_TR_MODE", "off"))),
            "cbo_tr_anchor_mode": _log_get_non_nan(log, "cbo_tr_anchor_mode", i, str(getattr(CFG, "CBO_TR_ANCHOR_MODE", "posterior_mean"))),
            "cbo_tr_radius": _log_get_ffill(log, "cbo_tr_radius", i, None),
            "cbo_tr_anchor_theta": _safe_json(_log_get_ffill(log, "cbo_tr_anchor_theta", i, None)),
            "tr_update_mode": _log_get_ffill(log, "tr_update_mode", i, None),
            "tr_baseline_mean": _log_get_ffill(log, "tr_baseline_mean", i, None),
            "tr_current_mean": _log_get_ffill(log, "tr_current_mean", i, None),
            "tr_improve_pct": _log_get_ffill(log, "tr_improve_pct", i, None),
            "tr_worse_pct": _log_get_ffill(log, "tr_worse_pct", i, None),
            "tr_update_signal": _log_get_ffill(log, "tr_update_signal", i, None),
            "tr_update_patience_count": _log_get_ffill(log, "tr_update_patience_count", i, None),
            "cbo_tr_radius_before_update": _log_get_ffill(log, "cbo_tr_radius_before_update", i, None),
            "actual_tr_anchor_mode": _log_get_ffill(log, "actual_tr_anchor_mode", i, None),
            "actual_tr_anchor_source": _log_get_ffill(log, "actual_tr_anchor_source", i, None),
            "actual_tr_anchor_theta": _safe_json(_log_get_ffill(log, "actual_tr_anchor_theta", i, None)),
            "actual_tr_anchor_reason": actual_anchor_reason_row,
            "anchor_override_used": anchor_override_used_row,
            "anchor_override_reason": anchor_override_reason_row,
            "anchor_fallback_used": _log_get_ffill(log, "anchor_fallback_used", i, None),
            "anchor_fallback_reason": _log_get_ffill(log, "anchor_fallback_reason", i, None),
            "anchor_theta_distance_to_prev": _log_get_ffill(log, "anchor_theta_distance_to_prev", i, None),
            "anchor_theta_distance_to_robust_elite": _log_get_ffill(log, "anchor_theta_distance_to_robust_elite", i, None),
            "anchor_theta_distance_to_context_best": _log_get_ffill(log, "anchor_theta_distance_to_context_best", i, None),
            "anchor_theta_distance_to_recent_best": _log_get_ffill(log, "anchor_theta_distance_to_recent_best", i, None),
            "cbo_tr_candidate_count": _log_get_ffill(log, "cbo_tr_candidate_count", i, None),
            "cbo_global_candidate_count": _log_get_ffill(log, "cbo_global_candidate_count", i, None),
            "cbo_tr_update_reason": _log_get_ffill(log, "cbo_tr_update_reason", i, None),
            "cbo_tr_success_count": _log_get_ffill(log, "cbo_tr_success_count", i, None),
            "cbo_tr_failure_count": _log_get_ffill(log, "cbo_tr_failure_count", i, None),
            "cbo_select_mode": _log_get_non_nan(log, "cbo_select_mode", i, str(getattr(CFG, "CBO_SELECT_MODE", "greedy"))),
            "cbo_topk": _log_get_ffill(log, "cbo_topk", i, None),
            "cbo_select_temperature": _log_get_ffill(log, "cbo_select_temperature", i, None),
            "cbo_epsilon": _log_get_ffill(log, "cbo_epsilon", i, None),
            "cbo_acq_beta": _log_get_ffill(log, "cbo_acq_beta", i, None),
            "cbo_acq_beta_mode": _log_get_ffill(log, "cbo_acq_beta_mode", i, None),
            "beta_eff": _log_get_ffill(log, "beta_eff", i, None),
            "radius_norm": _log_get_ffill(log, "radius_norm", i, None),
            "radius_beta_component": _log_get_ffill(log, "radius_beta_component", i, None),
            "state_beta_boost_used": _log_get_ffill(log, "state_beta_boost_used", i, None),
            "state_beta_boost_reason": _log_get_ffill(log, "state_beta_boost_reason", i, None),
            "actual_score_formula": _log_get_ffill(log, "actual_score_formula", i, None),
            "selected_candidate_score": _first_present(_log_get_ffill(log, "selected_candidate_score", i, None), _log_get_ffill(log, "selected_score", i, None)),
            "selected_candidate_beta_eff": _log_get_ffill(log, "selected_candidate_beta_eff", i, None),
            "selected_candidate_rank_by_score": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_score", i, None), _log_get_ffill(log, "selected_rank_by_score", i, None)),
            "selected_candidate_rank_by_sigma": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_sigma", i, None), _log_get_ffill(log, "selected_rank_by_sigma", i, None)),
            "service_guard_mode": _log_get_ffill(log, "service_guard_mode", i, None),
            "service_guard_available": _log_get_ffill(log, "service_guard_available", i, None),
            "service_guard_penalty": _log_get_ffill(log, "service_guard_penalty", i, None),
            "service_guard_reason": _log_get_ffill(log, "service_guard_reason", i, None),
            "good_region_available": _log_get_ffill(log, "good_region_available", i, None),
            "good_region_best_iter": _log_get_ffill(log, "good_region_best_iter", i, None),
            "good_region_best_rolling50_cost": _log_get_ffill(log, "good_region_best_rolling50_cost", i, None),
            "good_region_anchor_theta": _safe_json(_log_get_ffill(log, "good_region_anchor_theta", i, None)),
            "good_region_anchor_source": _log_get_ffill(log, "good_region_anchor_source", i, None),
            "distance_to_good_region_anchor": _log_get_ffill(log, "distance_to_good_region_anchor", i, None),
            "current_vs_good_region_gap_pct": _log_get_ffill(log, "current_vs_good_region_gap_pct", i, None),
            "good_region_guard_enabled": _log_get_ffill(log, "good_region_guard_enabled", i, None),
            "good_region_guard_triggered": _log_get_ffill(log, "good_region_guard_triggered", i, None),
            "good_region_guard_reason": _log_get_ffill(log, "good_region_guard_reason", i, None),
            "candidate_theta_6d": _safe_json(_log_get_safe(log, "candidate_theta_6d", i, None)),
            "deployed_theta_6d": _safe_json(_log_get_safe(log, "deployed_theta_6d", i, None)),
            "good_region_theta_6d": _safe_json(_log_get_ffill(log, "good_region_theta_6d", i, None)),
            "good_region_cost": _log_get_ffill(log, "good_region_cost", i, None),
            "good_region_iter": _log_get_ffill(log, "good_region_iter", i, None),
            "good_region_window_start": _log_get_ffill(log, "good_region_window_start", i, None),
            "good_region_window_end": _log_get_ffill(log, "good_region_window_end", i, None),
            "distance_to_good_region": _log_get_ffill(log, "distance_to_good_region", i, None),
            "candidate_selected_source": _log_get_ffill(log, "candidate_selected_source", i, None),
            "deployed_source": _log_get_ffill(log, "deployed_source", i, None),
            "candidate_tr_radius": _log_get_ffill(log, "candidate_tr_radius", i, None),
            "candidate_beta_eff": _log_get_ffill(log, "candidate_beta_eff", i, None),
            "guard_fallback_type": _log_get_ffill(log, "guard_fallback_type", i, None),
            "alpha_direct_fixed_theta_enabled": _log_get_ffill(log, "alpha_direct_fixed_theta_enabled", i, None),
            "alpha_direct_fixed_theta_6d": _safe_json(_log_get_safe(log, "alpha_direct_fixed_theta_6d", i, None)),
            "predicted_cost": _log_get_ffill(log, "predicted_cost", i, None),
            "actual_cost": _log_get_ffill(log, "actual_cost", i, None),
            "prediction_error": _log_get_ffill(log, "prediction_error", i, None),
            "surprise": _log_get_ffill(log, "surprise", i, None),
            "prediction_error_valid": _log_get_ffill(log, "prediction_error_valid", i, None),
            "prediction_error_skipped_reason": _log_get_ffill(log, "prediction_error_skipped_reason", i, None),
            "cost_gap_pct": _log_get_ffill(log, "cost_gap_pct", i, None),
            "prediction_guard_mode": _log_get_ffill(log, "prediction_guard_mode", i, None),
            "prediction_guard_enabled": _log_get_ffill(log, "prediction_guard_enabled", i, None),
            "prediction_guard_history_count": _log_get_ffill(log, "prediction_guard_history_count", i, None),
            "prediction_guard_recent_bias": _log_get_ffill(log, "prediction_guard_recent_bias", i, None),
            "prediction_guard_recent_mae": _log_get_ffill(log, "prediction_guard_recent_mae", i, None),
            "prediction_guard_underestimate_rate": _log_get_ffill(log, "prediction_guard_underestimate_rate", i, None),
            "prediction_guard_surprise_abs_mean": _log_get_ffill(log, "prediction_guard_surprise_abs_mean", i, None),
            "prediction_guard_should_trigger": _log_get_ffill(log, "prediction_guard_should_trigger", i, None),
            "prediction_guard_reason": _log_get_ffill(log, "prediction_guard_reason", i, None),
            "prediction_guard_window": _log_get_ffill(log, "prediction_guard_window", i, None),
            "prediction_guard_min_history": _log_get_ffill(log, "prediction_guard_min_history", i, None),
            "prediction_guard_bias_threshold": _log_get_ffill(log, "prediction_guard_bias_threshold", i, None),
            "prediction_guard_underestimate_threshold": _log_get_ffill(log, "prediction_guard_underestimate_threshold", i, None),
            "prediction_guard_start_iter": _log_get_ffill(log, "prediction_guard_start_iter", i, None),
            "prediction_guard_bias_weight": _log_get_ffill(log, "prediction_guard_bias_weight", i, None),
            "prediction_guard_mae_weight": _log_get_ffill(log, "prediction_guard_mae_weight", i, None),
            "prediction_guard_active_triggered": _log_get_ffill(log, "prediction_guard_active_triggered", i, None),
            "prediction_guard_active_reason": _log_get_ffill(log, "prediction_guard_active_reason", i, None),
            "prediction_guard_risk_margin": _log_get_ffill(log, "prediction_guard_risk_margin", i, None),
            "prediction_guard_predicted_candidate_cost": _log_get_ffill(log, "prediction_guard_predicted_candidate_cost", i, None),
            "prediction_guard_incumbent_cost": _log_get_ffill(log, "prediction_guard_incumbent_cost", i, None),
            "prediction_guard_predicted_improvement": _log_get_ffill(log, "prediction_guard_predicted_improvement", i, None),
            "prediction_guard_fallback_source": _log_get_ffill(log, "prediction_guard_fallback_source", i, None),
            "residual_trigger": _log_get_ffill(log, "residual_trigger", i, None),
            "condition_trigger": _log_get_ffill(log, "condition_trigger", i, None),
            "radius_min_stuck_count": _log_get_ffill(log, "radius_min_stuck_count", i, None),
            "force_explore_countdown": _log_get_ffill(log, "force_explore_countdown", i, None),
            "runtime_anchor_override": runtime_anchor_override_row,
            "runtime_anchor_override_reason": runtime_anchor_override_reason_row,
            "cbo_tr_radius_after_update": _log_get_ffill(log, "cbo_tr_radius_after_update", i, None),
            "selected_reason": _log_get_ffill(log, "selected_reason", i, None),
            "selected_candidate_source": selected_candidate_source,
            "selected_candidate_mu": _first_present(_log_get_ffill(log, "selected_candidate_mu", i, None), _log_get_ffill(log, "selected_mu", i, None)),
            "selected_candidate_sigma": _first_present(_log_get_ffill(log, "selected_candidate_sigma", i, None), _log_get_ffill(log, "selected_sigma", i, None)),
            "selected_candidate_acq": _first_present(_log_get_ffill(log, "selected_candidate_acq", i, None), _log_get_ffill(log, "selected_acq", i, None)),
            "selected_candidate_rank_by_mu": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_mu", i, None), _log_get_ffill(log, "selected_rank_by_mu", i, None)),
            "selected_candidate_rank_by_acq": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_acq", i, None), _log_get_ffill(log, "selected_rank_by_acq", i, None)),
            "best_mu_candidate_source": _log_get_ffill(log, "best_mu_candidate_source", i, None),
            "best_acq_candidate_source": _log_get_ffill(log, "best_acq_candidate_source", i, None),
            "num_candidates": _log_get_ffill(log, "num_candidates", i, None),
            "num_tr_candidates": _log_get_ffill(log, "num_tr_candidates", i, None),
            "num_global_candidates": _log_get_ffill(log, "num_global_candidates", i, None),
            "Avg_Delay_平均时延": log.get("avg_delay", [None] * n)[i],
            "Avg_Delay_RT_实时平均时延": log.get("avg_delay_rt", [None] * n)[i],
            "Avg_Delay_Batch_批量平均时延": log.get("avg_delay_batch", [None] * n)[i],
            "Avg_Delay_AI_AI平均时延": log.get("avg_delay_ai", [None] * n)[i],
            "SLA_Success_Rate_SLA成功率": log.get("sla_success_rate", [None] * n)[i],
            "Violation_Rate_违约率": log.get("vio_rate", [None] * n)[i],
            "Avg_Energy_平均能耗": log.get("avg_energy", [None] * n)[i],
            "Total_Energy_优化目标能耗": log.get("total_energy", [None] * n)[i],
            "Total_Energy_Real_真实常开能耗": log.get("total_energy_real", [None] * n)[i] if i < len(log.get("total_energy_real", [])) else None,
            "Compute_Dynamic_Energy_动态计算能耗": log.get("compute_dynamic_energy", [None] * n)[i] if i < len(log.get("compute_dynamic_energy", [])) else None,
            "Compute_Idle_Energy_目标空闲能耗": log.get("compute_idle_energy", [None] * n)[i] if i < len(log.get("compute_idle_energy", [])) else None,
            "Transmission_Energy_传输能耗": log.get("transmission_energy", [None] * n)[i] if i < len(log.get("transmission_energy", [])) else None,
            "Cumulative_Energy_累计目标能耗": log.get("cumulative_energy", [None] * n)[i],
            "Cumulative_Energy_Real_累计真实能耗": log.get("cumulative_energy_real", [None] * n)[i] if i < len(log.get("cumulative_energy_real", [])) else None,
            "Backlog_积压任务数": log.get("backlog", [None] * n)[i],
            "macro_context_key": _log_get_safe(log, "macro_context_key", i, None),
            "window_arrivals_total": _log_get_safe(log, "window_arrivals_total", i, None),
            "window_completed_total": _log_get_safe(log, "window_completed_total", i, None),
            "window_unfinished_total": _log_get_safe(log, "window_unfinished_total", i, None),
            "unfinished_rate": _log_get_safe(log, "unfinished_rate", i, None),
            "backlog_growth": _log_get_safe(log, "backlog_growth", i, None),
            "backlog_growth_rate": _log_get_safe(log, "backlog_growth_rate", i, None),
            "energy_per_arrival": _log_get_safe(log, "energy_per_arrival", i, None),
            "energy_metric_source": _log_get_safe(log, "energy_metric_source", i, None),
            "class_imbalance_available": _log_get_safe(log, "class_imbalance_available", i, None),
            "min_class_success_rate": _log_get_safe(log, "min_class_success_rate", i, None),
            "class_imbalance_penalty": _log_get_safe(log, "class_imbalance_penalty", i, None),
            "cbo_reference_mode": _log_get_safe(log, "cbo_reference_mode", i, None),
            "cbo_reference_available": _log_get_safe(log, "cbo_reference_available", i, None),
            "cbo_reference_status": _log_get_safe(log, "cbo_reference_status", i, None),
            "cbo_reference_round_count": _log_get_safe(log, "cbo_reference_round_count", i, None),
            "cbo_reference_frozen": _log_get_safe(log, "cbo_reference_frozen", i, None),
            "phase_id": _log_get_safe(log, "phase_id", i, None),
            "phase_name": _log_get_safe(log, "phase_name", i, None),
            "phase_iter": _log_get_safe(log, "phase_iter", i, None),
            "phase_signature": _log_get_safe(log, "phase_signature", i, None),
            "active_reference_id": _log_get_safe(log, "active_reference_id", i, None),
            "reference_source": _log_get_safe(log, "reference_source", i, None),
            "is_calibration_window": _log_get_safe(log, "is_calibration_window", i, None),
            "calibration_window_label": _log_get_safe(log, "calibration_window_label", i, None),
            "phase_reference_cache_status": _log_get_safe(log, "phase_reference_cache_status", i, None),
            "phase_reference_warmup_rounds": _log_get_safe(log, "phase_reference_warmup_rounds", i, None),
            "phase_reference_is_new_scene": _log_get_safe(log, "phase_reference_is_new_scene", i, None),
            "phase_reference_base_phase_id": _log_get_safe(log, "phase_reference_base_phase_id", i, None),
            "delay_ref": _log_get_safe(log, "delay_ref", i, None),
            "energy_per_arrival_ref": _log_get_safe(log, "energy_per_arrival_ref", i, None),
            "energy_norm_ref": _log_get_safe(log, "energy_norm_ref", i, None),
            "unfinished_rate_ref": _log_get_safe(log, "unfinished_rate_ref", i, None),
            "backlog_ref": _log_get_safe(log, "backlog_ref", i, None),
            "backlog_growth_ref": _log_get_safe(log, "backlog_growth_ref", i, None),
            "backlog_growth_rate_ref": _log_get_safe(log, "backlog_growth_rate_ref", i, None),
            "rt_violation_rate_ref": _log_get_safe(log, "rt_violation_rate_ref", i, None),
            "success_rate_ref": _log_get_safe(log, "success_rate_ref", i, None),
            "eval_cost_ref": _log_get_safe(log, "eval_cost_ref", i, None),
            "delay_norm": _log_get_safe(log, "delay_norm", i, None),
            "energy_norm": _log_get_safe(log, "energy_norm", i, None),
            "unfinished_norm": _log_get_safe(log, "unfinished_norm", i, None),
            "backlog_norm": _log_get_safe(log, "backlog_norm", i, None),
            "backlog_growth_norm": _log_get_safe(log, "backlog_growth_norm", i, None),
            "backlog_growth_rate_norm": _log_get_safe(log, "backlog_growth_rate_norm", i, None),
            "rt_violation_norm": _log_get_safe(log, "rt_violation_norm", i, None),
            "eval_cost_norm": _log_get_safe(log, "eval_cost_norm", i, None),
            "success_shortfall": _log_get_safe(log, "success_shortfall", i, None),
            "success_shortfall_norm": _log_get_safe(log, "success_shortfall_norm", i, None),
            "service_norm": _log_get_safe(log, "service_norm", i, None),
            "normalized_tradeoff_score": _log_get_safe(log, "normalized_tradeoff_score", i, None),
            "cbo_objective_mode": _log_get_safe(log, "cbo_objective_mode", i, None),
            "tradeoff_alpha": _log_get_safe(log, "tradeoff_alpha", i, None),
            "bo_training_cost_source": _log_get_safe(log, "bo_training_cost_source", i, None),
            "scheduler_tradeoff_mode": _log_get_safe(log, "scheduler_tradeoff_mode", i, None),
            "scheduler_score_norm_mode": _log_get_safe(log, "scheduler_score_norm_mode", i, None),
            "scheduler_use_score_risk": _log_get_safe(log, "scheduler_use_score_risk", i, None),
            "scheduler_le_scale": _log_get_safe(log, "scheduler_le_scale", i, None),
            "scheduler_alpha_last": _log_get_safe(log, "scheduler_alpha_last", i, None),
            "scheduler_alpha_mean": _log_get_safe(log, "scheduler_alpha_mean", i, None),
            "alpha_rt": _log_get_safe(log, "alpha_rt", i, None),
            "alpha_batch": _log_get_safe(log, "alpha_batch", i, None),
            "alpha_ai": _log_get_safe(log, "alpha_ai", i, None),
            "w_queue": _log_get_safe(log, "w_queue", i, None),
            "w_queue_effective": _log_get_safe(log, "w_queue_effective", i, None),
            "w_risk_scale": _log_get_safe(log, "w_risk_scale", i, None),
            "cloud_gate": _log_get_safe(log, "cloud_gate", i, None),
            "W_RT_Latency": w_rt_latency,
            "W_Batch_Latency": w_batch_latency,
            "W_AI_Latency": w_ai_latency,
            "W_Queue": w_queue_value,
            "W_Risk_Scale": w_risk_scale_value,
            "Cloud_Gate": cloud_gate_value,
            "W_Energy_Scale": w_energy_scale_last,
            "W_RT_Energy": w_rt_energy_last,
            "W_Batch_Energy": w_batch_energy_last,
            "W_AI_Energy": w_ai_energy_last,
            "W_RT_Latency_last": w_rt_latency,
            "W_Batch_Latency_last": w_batch_latency,
            "W_AI_Latency_last": w_ai_latency,
            "W_Queue_last": w_queue_value,
            "W_Risk_Scale_last": w_risk_scale_value,
            "Cloud_Gate_last": cloud_gate_value,
            "W_Energy_Scale_last": w_energy_scale_last,
            "W_RT_Energy_last": w_rt_energy_last,
            "W_Batch_Energy_last": w_batch_energy_last,
            "W_AI_Energy_last": w_ai_energy_last,
            "selected_latency_component_last": _log_get_safe(log, "selected_latency_component_last", i, None),
            "latency_contribution_last": _log_get_safe(log, "latency_contribution_last", i, None),
            "energy_contribution_last": _log_get_safe(log, "energy_contribution_last", i, None),
            "service_contribution_last": _log_get_safe(log, "service_contribution_last", i, None),
            "risk_contribution_last": _log_get_safe(log, "risk_contribution_last", i, None),
            "queue_contribution_last": _log_get_safe(log, "queue_contribution_last", i, None),
            "cloud_contribution_last": _log_get_safe(log, "cloud_contribution_last", i, None),
            "selected_energy_component_last": _log_get_safe(log, "selected_energy_component_last", i, None),
            "selected_risk_penalty_last": _log_get_safe(log, "selected_risk_penalty_last", i, None),
            "selected_queue_penalty_last": _log_get_safe(log, "selected_queue_penalty_last", i, None),
            "selected_latency_energy_component_last": _log_get_safe(log, "selected_latency_energy_component_last", i, None),
            "selected_latency_energy_component_unscaled_last": _log_get_safe(log, "selected_latency_energy_component_unscaled_last", i, None),
            "selected_latency_energy_component_scaled_last": _log_get_safe(log, "selected_latency_energy_component_scaled_last", i, None),
            "selected_service_component_last": _log_get_safe(log, "selected_service_component_last", i, None),  # Deprecated alias.
            "selected_norm_e_last": _log_get_safe(log, "selected_norm_e_last", i, None),
            "selected_norm_l_last": _log_get_safe(log, "selected_norm_l_last", i, None),
            "selected_norm_risk_last": _log_get_safe(log, "selected_norm_risk_last", i, None),
            "selected_norm_queue_last": _log_get_safe(log, "selected_norm_queue_last", i, None),
            "selected_score_last": _log_get_safe(log, "selected_score_last", i, None),
            "deadline_filter_reject_ratio": _log_get_safe(log, "deadline_filter_reject_ratio", i, None),
            "fallback_risk_used_ratio": _log_get_safe(log, "fallback_risk_used_ratio", i, None),
            "cloud_candidate_ratio": _log_get_safe(log, "cloud_candidate_ratio", i, None),
            "cloud_selected_ratio": _log_get_safe(log, "cloud_selected_ratio", i, None),
            "scheduler_score_min_last": _log_get_safe(log, "scheduler_score_min_last", i, None),
            "scheduler_score_max_last": _log_get_safe(log, "scheduler_score_max_last", i, None),
            "scheduler_score_gap_best_2nd_last": _log_get_safe(log, "scheduler_score_gap_best_2nd_last", i, None),
            "scheduler_score_gap_min_max_last": _log_get_safe(log, "scheduler_score_gap_min_max_last", i, None),
            "boltzmann_top1_prob_last": _log_get_safe(log, "boltzmann_top1_prob_last", i, None),
            "boltzmann_selected_prob_last": _log_get_safe(log, "boltzmann_selected_prob_last", i, None),
            "boltzmann_entropy_last": _log_get_safe(log, "boltzmann_entropy_last", i, None),
            "boltzmann_entropy_norm_last": _log_get_safe(log, "boltzmann_entropy_norm_last", i, None),
            "opportunity_candidate_count_last": _log_get_safe(log, "opportunity_candidate_count_last", i, None),
            "Zero_Completion_Penalty_零完成惩罚": log.get("zero_completion_penalty", [None] * n)[i],
            "Beta_Boltzmann系数": log.get("beta", [None] * n)[i],
            "Control_Label_控制标签": log.get("control_label", [None] * n)[i] if i < len(log.get("control_label", [])) else None,
            "Control_Vector_控制向量": _safe_json(control),
            "Control_Vector_Meaning": _log_get_safe(log, "control_vector_meaning", i, "deployed_full_theta"),
            "Alpha_Direct_Control_Vector_6D": _safe_json(alpha_direct_control),
            "Alpha_Direct_Feature_Names": _safe_json(alpha_feature_names),
            "Theta_Full_Deployed_11D": _safe_json(theta_full_deployed),
            "Theta_Full_Feature_Names": _safe_json(theta_full_feature_names),
            "Alpha_RT": alpha_direct_control[0],
            "Alpha_Batch": alpha_direct_control[1],
            "Alpha_AI": alpha_direct_control[2],
            "W_Queue_alpha_direct": alpha_direct_control[3],
            "W_Risk_Scale_alpha_direct": alpha_direct_control[4],
            "Cloud_Gate_alpha_direct": alpha_direct_control[5],
            "Alloc_By_Type_分任务节点分配": _safe_json(log.get("alloc_by_type", [None] * n)[i] if i < len(log.get("alloc_by_type", [])) else None),
            "Context_Label_情景标签": log.get("context_label", [None] * n)[i] if i < len(log.get("context_label", [])) else None,
            "Context_Vector_情景向量": _safe_json(log.get("context_vector", [None] * n)[i] if i < len(log.get("context_vector", [])) else None),
            "context_mode": _log_get_safe(log, "context_mode", i, None),
            "context_status": _log_get_safe(log, "context_status", i, None),
            "context_feature_names": _safe_json(_log_get_safe(log, "context_feature_names", i, None)),
            "External_Context_Vector_外部情景向量": _safe_json(log.get("external_context_vector", [None] * n)[i] if i < len(log.get("external_context_vector", [])) else None),
            "external_context_feature_names": _safe_json(_log_get_safe(log, "external_context_feature_names", i, None)),
            "Arrival_Rate_到达率": log.get("arrival_rate", [None] * n)[i],
            "Avg_Util_平均利用率": log.get("avg_util", [None] * n)[i],
            "RT_Ratio_实时占比": log.get("rt_arrival_ratio", [None] * n)[i],
            "Batch_Ratio_批量占比": log.get("batch_arrival_ratio", [None] * n)[i],
            "AI_Ratio_AI占比": log.get("ai_arrival_ratio", [None] * n)[i],
            "Arrivals_Total_本轮到达任务数": log.get("arrivals_total", [None] * n)[i],
            "Arrivals_RT_本轮实时任务数": log.get("arrivals_rt", [None] * n)[i],
            "Arrivals_Batch_本轮批量任务数": log.get("arrivals_batch", [None] * n)[i],
            "Arrivals_AI_本轮AI任务数": log.get("arrivals_ai", [None] * n)[i],
            "Completed_Total_本轮完成任务数": log.get("completed_total", [None] * n)[i],
            "Completed_RT_本轮完成实时任务数": log.get("completed_rt", [None] * n)[i],
            "Completed_Batch_本轮完成批量任务数": log.get("completed_batch", [None] * n)[i],
            "Completed_AI_本轮完成AI任务数": log.get("completed_ai", [None] * n)[i],

            # Per-class window diagnostics.
            "RT_Window_Avg_Lateness_实时窗口平均超期": log.get("avg_lateness_rt", [None] * n)[i] if i < len(log.get("avg_lateness_rt", [])) else None,
            "Batch_Window_Avg_Lateness_批任务窗口平均超期": log.get("avg_lateness_batch", [None] * n)[i] if i < len(log.get("avg_lateness_batch", [])) else None,
            "AI_Window_Avg_Lateness_AI窗口平均超期": log.get("avg_lateness_ai", [None] * n)[i] if i < len(log.get("avg_lateness_ai", [])) else None,
            "RT_Window_Vio_Rate_实时窗口违约率": log.get("vio_rate_rt", [None] * n)[i] if i < len(log.get("vio_rate_rt", [])) else None,
            "Batch_Window_Vio_Rate_批任务窗口违约率": log.get("vio_rate_batch", [None] * n)[i] if i < len(log.get("vio_rate_batch", [])) else None,
            "AI_Window_Vio_Rate_AI窗口违约率": log.get("vio_rate_ai", [None] * n)[i] if i < len(log.get("vio_rate_ai", [])) else None,
            "RT_Window_Completion_Ratio_实时窗口完成比例": log.get("completion_ratio_rt", [None] * n)[i] if i < len(log.get("completion_ratio_rt", [])) else None,
            "Batch_Window_Completion_Ratio_批任务窗口完成比例": log.get("completion_ratio_batch", [None] * n)[i] if i < len(log.get("completion_ratio_batch", [])) else None,
            "AI_Window_Completion_Ratio_AI窗口完成比例": log.get("completion_ratio_ai", [None] * n)[i] if i < len(log.get("completion_ratio_ai", [])) else None,
            "RT_Window_Avg_Energy_实时窗口平均能耗": log.get("avg_energy_rt", [None] * n)[i] if i < len(log.get("avg_energy_rt", [])) else None,
            "Batch_Window_Avg_Energy_批任务窗口平均能耗": log.get("avg_energy_batch", [None] * n)[i] if i < len(log.get("avg_energy_batch", [])) else None,
            "AI_Window_Avg_Energy_AI窗口平均能耗": log.get("avg_energy_ai", [None] * n)[i] if i < len(log.get("avg_energy_ai", [])) else None,
            "RT_Window_Class_Cost_实时窗口分类Cost": log.get("window_rt_class_cost", [None] * n)[i] if i < len(log.get("window_rt_class_cost", [])) else None,
            "Batch_Window_Class_Cost_批任务窗口分类Cost": log.get("window_batch_class_cost", [None] * n)[i] if i < len(log.get("window_batch_class_cost", [])) else None,
            "AI_Window_Class_Cost_AI窗口分类Cost": log.get("window_ai_class_cost", [None] * n)[i] if i < len(log.get("window_ai_class_cost", [])) else None,
            "Unfinished_End_轮末未完成任务数": log.get("unfinished_end", [None] * n)[i],
            "Feedback_Mode_反馈模式": log.get("feedback_mode", [None] * n)[i] if i < len(log.get("feedback_mode", [])) else None,
            "Cohort_ID_任务批次ID": log.get("cohort_id", [None] * n)[i] if i < len(log.get("cohort_id", [])) else None,
            "Cohort_Arrivals_批次到达任务数": log.get("cohort_arrivals", [None] * n)[i] if i < len(log.get("cohort_arrivals", [])) else None,
            "Cohort_Feedback_Count_本轮反馈批次数": log.get("cohort_feedback_count", [None] * n)[i] if i < len(log.get("cohort_feedback_count", [])) else None,
            "Cohort_Feedback_Cost_Mean_本轮批次反馈平均Cost": log.get("cohort_feedback_cost_mean", [None] * n)[i] if i < len(log.get("cohort_feedback_cost_mean", [])) else None,
            "Cohort_Active_Count_活跃批次数": log.get("cohort_active_count", [None] * n)[i] if i < len(log.get("cohort_active_count", [])) else None,
            "Cohort_Finalized_Total_累计已反馈批次数": log.get("cohort_finalized_total", [None] * n)[i] if i < len(log.get("cohort_finalized_total", [])) else None,
            "Cohort_Pending_Tasks_批次未完成任务数": log.get("cohort_pending_tasks", [None] * n)[i] if i < len(log.get("cohort_pending_tasks", [])) else None,
            "Training_Sample_Count_建模样本数": log.get("training_sample_count", [None] * n)[i],
            "Recent_Sample_Count_最近样本数": log.get("recent_sample_count", [None] * n)[i],
            "Archive_Sample_Count_归档样本数": log.get("archive_sample_count", [None] * n)[i],
            "Neighbor_K_相似邻居数": log.get("neighbor_k", [None] * n)[i],
            "Candidate_Count_候选点数量": log.get("candidate_count", [None] * n)[i],
            "Trust_Radius_TR半径": log.get("trust_radius", [None] * n)[i],
            "Pivot_局部搜索中心": _safe_json(log.get("pivot_theta", [None] * n)[i] if i < len(log.get("pivot_theta", [])) else None),
            "TopK_History_参考历史点": _safe_json(log.get("topk_history", [None] * n)[i] if i < len(log.get("topk_history", [])) else None),
            "TopK_Similarity_相似度列表": _safe_json(log.get("topk_similarity", [None] * n)[i] if i < len(log.get("topk_similarity", [])) else None),
            "Best_Selected_本轮最终选点": _safe_json(log.get("best_selected_theta", [None] * n)[i] if i < len(log.get("best_selected_theta", [])) else None),
        }
        bilingual_feature_names = [
            "W_RT_Latency_RT时延权重",
            "W_Batch_Latency_Batch时延权重",
            "W_AI_Latency_AI时延权重",
            "W_RT_Energy_RT能耗权重",
            "W_Batch_Energy_Batch能耗权重",
            "W_AI_Energy_AI能耗权重",
        ]
        for idx, name in enumerate(bilingual_feature_names):
            row[name] = theta[idx] if idx < len(theta) else None
        if isinstance(alloc, (list, tuple)):
            row["Alloc_Counts_节点分配统计"] = _safe_json(alloc)
            for node_idx, value in enumerate(alloc):
                row[f"Alloc_Node_{node_idx}_节点{node_idx}分配数"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_context_debug_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    rows = []
    for i in range(n):
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Time_时间": log.get("time", [None] * n)[i],
            "Context_Label_情景标签": log.get("context_label", [None] * n)[i] if i < len(log.get("context_label", [])) else None,
            "Context_Vector_情景向量": _safe_json(log.get("context_vector", [None] * n)[i] if i < len(log.get("context_vector", [])) else None),
            "Arrival_Rate_到达率": log.get("arrival_rate", [None] * n)[i],
            "Avg_Util_平均利用率": log.get("avg_util", [None] * n)[i],
            "Backlog_积压任务数": log.get("backlog", [None] * n)[i],
            "Violation_Rate_违约率": log.get("vio_rate", [None] * n)[i],
            "RT_Ratio_实时占比": log.get("rt_arrival_ratio", [None] * n)[i],
            "Batch_Ratio_批量占比": log.get("batch_arrival_ratio", [None] * n)[i],
            "AI_Ratio_AI占比": log.get("ai_arrival_ratio", [None] * n)[i],
            "Training_Sample_Count_建模样本数": log.get("training_sample_count", [None] * n)[i],
            "Recent_Sample_Count_最近样本数": log.get("recent_sample_count", [None] * n)[i],
            "Archive_Sample_Count_归档样本数": log.get("archive_sample_count", [None] * n)[i],
            "Neighbor_K_相似邻居数": log.get("neighbor_k", [None] * n)[i],
            "Candidate_Count_候选点数量": log.get("candidate_count", [None] * n)[i],
            "Trust_Radius_TR半径": log.get("trust_radius", [None] * n)[i],
            "Pivot_局部搜索中心": _safe_json(log.get("pivot_theta", [None] * n)[i] if i < len(log.get("pivot_theta", [])) else None),
            "TopK_History_参考历史点": _safe_json(log.get("topk_history", [None] * n)[i] if i < len(log.get("topk_history", [])) else None),
            "TopK_Similarity_相似度列表": _safe_json(log.get("topk_similarity", [None] * n)[i] if i < len(log.get("topk_similarity", [])) else None),
            "Best_Selected_本轮最终选点": _safe_json(log.get("best_selected_theta", [None] * n)[i] if i < len(log.get("best_selected_theta", [])) else None),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_alloc_debug_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    allocs = log.get("alloc", [])
    rows = []
    for i in range(n):
        alloc = allocs[i] if i < len(allocs) else []
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Time_时间": log.get("time", [None] * n)[i],
            "Arrivals_Total_本轮到达任务数": log.get("arrivals_total", [None] * n)[i],
            "Completed_Total_本轮完成任务数": log.get("completed_total", [None] * n)[i],
            "Unfinished_End_轮末未完成任务数": log.get("unfinished_end", [None] * n)[i],
            "Alloc_Counts_节点分配统计": _safe_json(alloc),
        }
        if isinstance(alloc, (list, tuple)):
            for node_idx, value in enumerate(alloc):
                row[f"Alloc_Node_{node_idx}_节点{node_idx}分配数"] = value
        rows.append(row)
    return pd.DataFrame(rows)


# ===============================================================
# 诊断增强：每轮平均指标、cohort 学习曲线、按任务类型节点分配
# ===============================================================
PLOT_COLOR_CYCLE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#E69F00", "#000000", "#8A2BE2", "#A52A2A"
]


def _log_get(log, key, i, default=None):
    vals = log.get(key, []) if isinstance(log, dict) else []
    if isinstance(vals, list) and i < len(vals):
        return vals[i]
    return default


def _json_load_maybe(x):
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _normalize_alloc_by_type(value, node_count):
    """把 log 里的 alloc_by_type 统一整理成 RT/Batch/AI -> 节点计数列表。"""
    empty = {t: [0.0 for _ in range(node_count)] for t in TASK_TYPE_ORDER}
    value = _json_load_maybe(value)
    if not isinstance(value, dict):
        return empty, True
    out = {}
    missing = False
    for task_type in TASK_TYPE_ORDER:
        vals = value.get(task_type, value.get(task_type.lower(), None))
        if vals is None:
            out[task_type] = [0.0 for _ in range(node_count)]
            missing = True
            continue
        if isinstance(vals, np.ndarray):
            vals = vals.tolist()
        if not isinstance(vals, (list, tuple)):
            out[task_type] = [0.0 for _ in range(node_count)]
            missing = True
            continue
        clean = [_safe_float(v, 0.0) for v in list(vals)[:node_count]]
        if len(clean) < node_count:
            clean += [0.0 for _ in range(node_count - len(clean))]
        out[task_type] = clean
    return out, missing


def _node_meta(node_idx):
    cfg = CFG.NODES_CFG[int(node_idx)]
    return {
        "Node_ID_节点": int(node_idx),
        "Node_Type_节点类型": str(cfg.get("node_type", cfg.get("role", ""))),
        "Node_Role_节点角色": str(cfg.get("role", "")),
        "Node_Workshop_车间": int(cfg.get("workshop", cfg.get("site", -1))),
        "Node_Is_Cloud_是否云": bool(_node_is_cloud(cfg)),
        "Node_Capacity_Slots": int(get_node_capacity_slots(cfg)),
        "Node_Num_Cores_Metadata": int(cfg.get("num_cores", cfg.get("cpu", 0))),
        "Node_Service_Rate_GIPS": float(cfg.get("service_rate_gips", cfg.get("speed", 0.0))),
        "Node_Memory_GB": float(cfg.get("memory_gb", 0.0)),
        "Node_Accelerator_Type": str(cfg.get("accelerator_type", "none")),
        "Node_Task_Affinity_RT": float(get_node_task_affinity_factor(cfg, "RT")),
        "Node_Task_Affinity_Batch": float(get_node_task_affinity_factor(cfg, "Batch")),
        "Node_Task_Affinity_AI": float(get_node_task_affinity_factor(cfg, "AI")),
        "Node_RT_Reserved_Slots": int(get_reserved_slots(cfg, "RT")),
    }


def build_alloc_by_type_debug_dataframe(log, group_key, group_label, run_index=None):
    """长表：每轮、每类任务、每个节点一行，用于判断 BO 是否改变了任务类型级分配。"""
    n = len(log.get("time", []))
    node_count = len(CFG.NODES_CFG)
    rows = []
    for i in range(n):
        alloc_raw = _log_get(log, "alloc_by_type", i, default=None)
        alloc_by_type, missing_alloc_by_type = _normalize_alloc_by_type(alloc_raw, node_count)
        control_vec = _json_load_maybe(_log_get(log, "control_vector", i, default=[]))
        if not isinstance(control_vec, (list, tuple)):
            control_vec = []
        base_row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Run_Index_重复编号": run_index,
            "Iteration_轮次": i + 1,
            "Time_时间": _log_get(log, "time", i, None),
            "Feedback_Mode_反馈模式": _log_get(log, "feedback_mode", i, None),
            "Reward_评分": _log_get(log, "reward", i, None),
            "Cost_代价": -_safe_float(_log_get(log, "reward", i, np.nan)),
            "Avg_Delay_平均时延": _log_get(log, "avg_delay", i, None),
            "Avg_Energy_平均能耗": _log_get(log, "avg_energy", i, None),
            "Violation_Rate_违约率": _log_get(log, "vio_rate", i, None),
            "SLA_Success_Rate_SLA成功率": _log_get(log, "sla_success_rate", i, None),
            "Backlog_积压": _log_get(log, "backlog", i, None),
            "Arrivals_Total_总到达": _log_get(log, "arrivals_total", i, 0),
            "Completed_Total_总完成": _log_get(log, "completed_total", i, 0),
            "RT_Arrivals_实时到达": _log_get(log, "arrivals_rt", i, None),
            "Batch_Arrivals_批任务到达": _log_get(log, "arrivals_batch", i, None),
            "AI_Arrivals_AI到达": _log_get(log, "arrivals_ai", i, None),
            "RT_Ratio_实时占比": _log_get(log, "rt_arrival_ratio", i, None),
            "Batch_Ratio_批任务占比": _log_get(log, "batch_arrival_ratio", i, None),
            "AI_Ratio_AI占比": _log_get(log, "ai_arrival_ratio", i, None),
            "Missing_Alloc_By_Type_缺失类型分配": bool(missing_alloc_by_type),
            "Control_Vector_控制向量": _safe_json(list(control_vec)),
        }
        for idx, name in enumerate(REDUCED4_FEATURE_NAMES if 'REDUCED4_FEATURE_NAMES' in globals() else ["Theta0", "Theta1", "Theta2", "Theta3"]):
            base_row[f"Control_{name}"] = _safe_float(control_vec[idx], np.nan) if idx < len(control_vec) else np.nan
        for task_type in TASK_TYPE_ORDER:
            counts = alloc_by_type.get(task_type, [0.0 for _ in range(node_count)])
            task_total = float(np.nansum(counts))
            cloud_total = 0.0
            edge_total = 0.0
            for node_idx, count in enumerate(counts):
                if _node_is_cloud(CFG.NODES_CFG[int(node_idx)]):
                    cloud_total += float(count)
                else:
                    edge_total += float(count)
            for node_idx, count in enumerate(counts):
                row = dict(base_row)
                row.update(_node_meta(node_idx))
                row.update({
                    "Task_Type_任务类型": task_type,
                    "Alloc_Count_分配数": float(count),
                    "Task_Type_Total_该类型总分配": task_total,
                    "Alloc_Ratio_In_Task_该类型内占比": float(count) / task_total if task_total > 0 else 0.0,
                    "Task_Type_Cloud_Total_该类型云分配": cloud_total,
                    "Task_Type_Edge_Total_该类型边缘分配": edge_total,
                    "Task_Type_Cloud_Ratio_该类型云占比": cloud_total / task_total if task_total > 0 else 0.0,
                    "Task_Type_Edge_Ratio_该类型边缘占比": edge_total / task_total if task_total > 0 else 0.0,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def build_alloc_by_type_summary_dataframe(alloc_type_df):
    if alloc_type_df is None or alloc_type_df.empty:
        return pd.DataFrame()
    group_cols = [
        "Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型",
        "Node_ID_节点", "Node_Role_节点角色", "Node_Workshop_车间", "Node_Is_Cloud_是否云"
    ]
    summary = alloc_type_df.groupby(group_cols, dropna=False)["Alloc_Count_分配数"].sum().reset_index()
    totals = summary.groupby(["Group_Key_方法键", "Task_Type_任务类型"], dropna=False)["Alloc_Count_分配数"].sum().reset_index()
    totals = totals.rename(columns={"Alloc_Count_分配数": "Task_Type_Total_该类型总分配"})
    summary = summary.merge(totals, on=["Group_Key_方法键", "Task_Type_任务类型"], how="left")
    summary["Alloc_Ratio_In_Task_该类型内占比"] = summary.apply(
        lambda r: float(r["Alloc_Count_分配数"]) / float(r["Task_Type_Total_该类型总分配"])
        if float(r["Task_Type_Total_该类型总分配"]) > 0 else 0.0,
        axis=1,
    )
    cloud_rows = []
    for (g, label, t), sub in summary.groupby(["Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型"], dropna=False):
        total = float(sub["Alloc_Count_分配数"].sum())
        cloud = float(sub.loc[sub["Node_Is_Cloud_是否云"] == True, "Alloc_Count_分配数"].sum())
        edge = total - cloud
        cloud_rows.append({
            "Group_Key_方法键": g,
            "Group_Label_方法名称": label,
            "Task_Type_任务类型": t,
            "Task_Type_Total_该类型总分配": total,
            "Cloud_Total_云分配": cloud,
            "Edge_Total_边缘分配": edge,
            "Cloud_Ratio_云占比": cloud / total if total > 0 else 0.0,
            "Edge_Ratio_边缘占比": edge / total if total > 0 else 0.0,
        })
    cloud_df = pd.DataFrame(cloud_rows)
    summary = summary.merge(
        cloud_df,
        on=["Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型", "Task_Type_Total_该类型总分配"],
        how="left",
    )
    return summary


def plot_alloc_by_type_summary(summary_df, save_dir, group_key, group_label):
    if summary_df is None or summary_df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)
    task_types = [t for t in TASK_TYPE_ORDER if t in set(summary_df["Task_Type_任务类型"])]
    node_ids = sorted(summary_df["Node_ID_节点"].dropna().astype(int).unique().tolist())
    pivot = summary_df.pivot_table(index="Task_Type_任务类型", columns="Node_ID_节点", values="Alloc_Count_分配数", aggfunc="sum", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(task_types), dtype=float)
    for j, node_id in enumerate(node_ids):
        vals = np.array([float(pivot.loc[t, node_id]) if t in pivot.index and node_id in pivot.columns else 0.0 for t in task_types])
        if np.sum(vals) <= 0:
            continue
        role = str(CFG.NODES_CFG[int(node_id)].get("role", ""))
        ax.bar(task_types, vals, bottom=bottom, label=f"N{node_id}-{role}", color=PLOT_COLOR_CYCLE[j % len(PLOT_COLOR_CYCLE)])
        bottom += vals
    ax.set_title(f"{group_label} - Allocation by Task Type")
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Allocated Tasks")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{group_key}_alloc_by_type_stacked_任务类型节点堆叠图.png"), dpi=300)
    plt.close(fig)

    cloud_df = summary_df.drop_duplicates(["Group_Key_方法键", "Task_Type_任务类型"])[["Task_Type_任务类型", "Cloud_Ratio_云占比", "Edge_Ratio_边缘占比"]].copy()
    cloud_df = cloud_df.set_index("Task_Type_任务类型").reindex(task_types).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(task_types))
    width = 0.35
    ax.bar(x - width / 2, cloud_df["Edge_Ratio_边缘占比"].fillna(0.0).values, width, label="Edge", color="#0072B2")
    ax.bar(x + width / 2, cloud_df["Cloud_Ratio_云占比"].fillna(0.0).values, width, label="Cloud", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(task_types)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title(f"{group_label} - Edge vs Cloud Ratio by Task Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{group_key}_alloc_by_type_cloud_ratio_任务类型云边占比.png"), dpi=300)
    plt.close(fig)


def plot_alloc_by_type_method_compare(all_summary_df, save_dir):
    if all_summary_df is None or all_summary_df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)
    method_labels = all_summary_df[["Group_Key_方法键", "Group_Label_方法名称"]].drop_duplicates().sort_values("Group_Key_方法键")
    ratio_rows = []
    for _, method in method_labels.iterrows():
        g = method["Group_Key_方法键"]
        label = method["Group_Label_方法名称"]
        sub_g = all_summary_df[all_summary_df["Group_Key_方法键"] == g]
        for task_type in TASK_TYPE_ORDER:
            sub_t = sub_g[sub_g["Task_Type_任务类型"] == task_type]
            if sub_t.empty:
                cloud_ratio, total = np.nan, 0.0
            else:
                one = sub_t.iloc[0]
                cloud_ratio = _safe_float(one.get("Cloud_Ratio_云占比", np.nan))
                total = _safe_float(one.get("Task_Type_Total_该类型总分配", 0.0), 0.0)
            ratio_rows.append({
                "Group_Key_方法键": g,
                "Group_Label_方法名称": label,
                "Task_Type_任务类型": task_type,
                "Cloud_Ratio_云占比": cloud_ratio,
                "Task_Type_Total_该类型总分配": total,
            })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(os.path.join(save_dir, "alloc_by_type_method_cloud_ratio_compare_方法云占比对比.csv"), index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TASK_TYPE_ORDER))
    methods = method_labels["Group_Key_方法键"].tolist()
    width = 0.8 / max(1, len(methods))
    for j, g in enumerate(methods):
        sub = ratio_df[ratio_df["Group_Key_方法键"] == g]
        vals = []
        for t in TASK_TYPE_ORDER:
            row = sub[sub["Task_Type_任务类型"] == t]
            vals.append(float(row["Cloud_Ratio_云占比"].iloc[0]) if not row.empty else np.nan)
        label = method_labels.loc[method_labels["Group_Key_方法键"] == g, "Group_Label_方法名称"].iloc[0]
        color = get_method_style(g, {"label": label}, fallback_idx=j).get("color", PLOT_COLOR_CYCLE[j % len(PLOT_COLOR_CYCLE)])
        ax.bar(x - 0.4 + width / 2 + j * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_TYPE_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cloud Allocation Ratio")
    ax.set_title("Cloud Allocation Ratio by Method and Task Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "alloc_by_type_method_cloud_ratio_compare_方法云占比对比.png"), dpi=300)
    plt.close(fig)


def plot_round_mean_energy_delay_score(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """每轮重复平均后的能耗、时延、评分曲线。"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    phase_ranges = get_bo_phase_ranges()
    metric_specs = [
        ("avg_energy", "Avg Energy per Completed Task", "Avg Energy"),
        ("avg_delay", "Avg Delay", "Delay (s)"),
        ("reward", "Score / Reward", "Reward (higher is better)"),
    ]
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        style = get_method_style(group_key, info, fallback_idx=idx2)
        for ax, (key, title, ylabel) in zip(axes, metric_specs):
            vals = log.get(key, [])
            if not vals:
                continue
            vals = ema_smooth(vals, weight=0.80)
            ax.plot(np.arange(1, len(vals) + 1), vals, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.45)
    for ax in axes:
        for phase in phase_ranges[:-1]:
            ax.axvline(phase["iter_end"], color="#555555", linestyle=":", alpha=0.8)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("BO Iteration")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_round_mean_energy_delay_score_每轮平均能耗时延评分.png"), dpi=300)
    plt.close(fig)


def plot_cohort_learning_curves(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """画 cohort_cost 和 best-so-far cohort_reward，诊断 BO 学习反馈。"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        frames = []
        for raw_log in info.get("logs", []):
            cdf = build_cohort_feedback_dataframe(raw_log, group_key, info["label"])
            if not cdf.empty:
                frames.append(cdf)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "Window_Index_窗口序号" not in df.columns:
            continue
        df["Window_Index_窗口序号"] = pd.to_numeric(df["Window_Index_窗口序号"], errors="coerce")
        df["Cohort_Cost_批次Cost"] = pd.to_numeric(df["Cohort_Cost_批次Cost"], errors="coerce")
        df = df.dropna(subset=["Window_Index_窗口序号", "Cohort_Cost_批次Cost"])
        if df.empty:
            continue
        mean_df = df.groupby("Window_Index_窗口序号", as_index=False)["Cohort_Cost_批次Cost"].mean().sort_values("Window_Index_窗口序号")
        x = mean_df["Window_Index_窗口序号"].astype(int).values
        cost = mean_df["Cohort_Cost_批次Cost"].values.astype(float)
        reward = -cost
        bsf_reward = best_so_far(reward.tolist())
        style = get_method_style(group_key, info, fallback_idx=idx2)
        axes[0].plot(x, ema_smooth(cost.tolist(), weight=0.75), label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
        axes[1].plot(x, bsf_reward, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
    axes[0].set_title("Cohort Feedback Cost by BO Iteration")
    axes[0].set_ylabel("Cohort Cost (lower is better)")
    axes[1].set_title("Best-so-far Cohort Reward")
    axes[1].set_ylabel("Best-so-far Cohort Reward")
    axes[1].set_xlabel("BO Iteration / Feedback Window")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_cohort_learning_curves_批次反馈学习曲线.png"), dpi=300)
    plt.close(fig)


def plot_theta_trajectory(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """画低维控制变量轨迹，自动兼容 reduced4 / reduced6。"""
    # 先扫描所有 control_vector，判断最大维度。
    max_dim = 0
    for info in group_logs.values():
        log0 = aggregate_logs(info.get("logs", [])) if info.get("logs") else info.get("mean", {})
        for c in log0.get("control_vector", []):
            cc = _json_load_maybe(c)
            if isinstance(cc, (list, tuple, np.ndarray)):
                max_dim = max(max_dim, len(cc))
                break
    if max_dim <= 0:
        return
    names = _control_feature_names_for_vector([np.nan] * max_dim) if '_control_feature_names_for_vector' in globals() else [f"Theta{i}" for i in range(max_dim)]
    fig, axes = plt.subplots(max_dim, 1, figsize=(13, max(3.0 * max_dim, 8)), sharex=True)
    if max_dim == 1:
        axes = [axes]
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        controls = log.get("control_vector", [])
        if not controls:
            continue
        arr = []
        for c in controls:
            cc = _json_load_maybe(c)
            if isinstance(cc, np.ndarray):
                cc = cc.tolist()
            if isinstance(cc, (list, tuple)):
                row = []
                for v in list(cc)[:max_dim]:
                    row.append(_safe_float(v))
                while len(row) < max_dim:
                    row.append(np.nan)
                arr.append(row)
        if not arr:
            continue
        arr = np.array(arr, dtype=float)
        style = get_method_style(group_key, info, fallback_idx=idx2)
        x = np.arange(1, arr.shape[0] + 1)
        for d in range(min(max_dim, arr.shape[1])):
            axes[d].plot(x, arr[:, d], label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
    for d, ax in enumerate(axes):
        label_name = names[d] if d < len(names) else f"Theta{d}"
        ax.set_title(f"Control Trajectory - {label_name}")
        ax.set_ylabel(label_name)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("BO Iteration")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_theta_trajectory_控制参数轨迹.png"), dpi=300)
    plt.close(fig)

def _as_clean_numeric_list(values):
    """把 None / 字符串 / 非数值项转成可安全 np 统计的 float list。"""
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    out = []
    for v in values:
        try:
            if v is None:
                out.append(np.nan)
            else:
                fv = float(v)
                out.append(fv if np.isfinite(fv) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def _safe_nanmean(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _safe_nanmax(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmax(arr))


def _safe_last(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    for v in reversed(vals):
        if not np.isnan(v):
            return float(v)
    return np.nan


def _recovery_iteration(backlog_seq, phase, threshold=1.0):
    start = max(0, phase["iter_start"] - 1)
    end = min(len(backlog_seq), phase["iter_end"])
    for idx in range(start, end):
        try:
            value = backlog_seq[idx]
            if value is None:
                continue
            if float(value) <= float(threshold):
                return idx + 1
        except Exception:
            continue
    return np.nan


def build_key_metric_summary_dataframe(group_logs):
    """核心指标统计：整体 + 阶段 + 峰值 + 恢复。

    修复点：aggregate_logs 对某些非数值字段会产生 None；这里统一转为 np.nan，
    避免 Colab / NumPy 2.x 下 np.nanmean([None, None]) 报错。
    """
    rows = []
    phases = get_bo_phase_ranges()
    for group_key, info in group_logs.items():
        log = aggregate_logs(info["logs"])
        reward = _as_clean_numeric_list(log.get("reward", []))
        delay = _as_clean_numeric_list(log.get("avg_delay", []))
        energy = _as_clean_numeric_list(log.get("avg_energy", []))
        total_energy = _as_clean_numeric_list(log.get("total_energy", []))
        vio = _as_clean_numeric_list(log.get("vio_rate", []))
        sla = _as_clean_numeric_list(log.get("sla_success_rate", []))
        backlog = _as_clean_numeric_list(log.get("backlog", []))
        completion = _as_clean_numeric_list(log.get("completion_rate", []))
        completion_ratio = _as_clean_numeric_list(log.get("completion_ratio", []))
        util = _as_clean_numeric_list(log.get("avg_util", []))
        arrivals_total = _as_clean_numeric_list(log.get("arrivals_total", []))
        completed_total = _as_clean_numeric_list(log.get("completed_total", []))
        backlog_growth = _as_clean_numeric_list(log.get("backlog_growth_rate", []))
        cumulative_energy = _as_clean_numeric_list(log.get("cumulative_energy", []))
        cumulative_energy_real = _as_clean_numeric_list(log.get("cumulative_energy_real", []))
        w_rt_latency = _as_clean_numeric_list(log.get("W_RT_Latency_last", []))
        w_batch_latency = _as_clean_numeric_list(log.get("W_Batch_Latency_last", []))
        w_ai_latency = _as_clean_numeric_list(log.get("W_AI_Latency_last", []))
        w_queue_value = _as_clean_numeric_list(log.get("W_Queue_last", log.get("w_queue", [])))
        w_risk_scale_value = _as_clean_numeric_list(log.get("W_Risk_Scale_last", log.get("w_risk_scale", [])))
        cloud_gate_value = _as_clean_numeric_list(log.get("Cloud_Gate_last", log.get("cloud_gate", [])))
        w_energy_scale_last = _as_clean_numeric_list(log.get("W_Energy_Scale_last", []))
        w_rt_energy_last = _as_clean_numeric_list(log.get("W_RT_Energy_last", []))
        w_batch_energy_last = _as_clean_numeric_list(log.get("W_Batch_Energy_last", []))
        w_ai_energy_last = _as_clean_numeric_list(log.get("W_AI_Energy_last", []))
        total_arrivals = float(np.nansum(arrivals_total)) if arrivals_total else 0.0
        total_completed = float(np.nansum(completed_total)) if completed_total else 0.0

        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": info["label"],
            "Overall_Mean_Reward_整体平均评分": _safe_nanmean(reward),
            "Overall_Mean_Cost_整体平均代价": _safe_nanmean([-x if not np.isnan(x) else np.nan for x in reward]),
            "Overall_Mean_Avg_Delay_整体平均时延": _safe_nanmean(delay),
            "Overall_Mean_Avg_Energy_整体平均能耗": _safe_nanmean(energy),
            "Overall_Mean_Violation_整体平均违约率": _safe_nanmean(vio),
            "Overall_Mean_SLA_整体平均SLA成功率": _safe_nanmean(sla),
            "Overall_Mean_Backlog_整体平均积压": _safe_nanmean(backlog),
            "Overall_Mean_Utilization_Load_Calibration": _safe_nanmean(util),
            "Overall_Total_Completion_Ratio_Load_Calibration": total_completed / max(total_arrivals, 1.0),
            "Overall_Mean_Window_Completion_Ratio_Load_Calibration": _safe_nanmean(completion_ratio),
            "Overall_Mean_Backlog_Growth_Rate_Load_Calibration": _safe_nanmean(backlog_growth),
            "Overall_Peak_Backlog_整体积压峰值": _safe_nanmax(backlog),
            "Final_Reward_最终评分": _safe_last(reward),
            "Final_Avg_Delay_最终平均时延": _safe_last(delay),
            "Final_Avg_Energy_最终平均能耗": _safe_last(energy),
            "Final_Backlog_最终积压": _safe_last(backlog),
            "Final_Cumulative_Objective_Energy_最终累计目标能耗": _safe_last(cumulative_energy),
            "Final_Cumulative_Real_Energy_最终累计真实能耗": _safe_last(cumulative_energy_real),
            "Mean_W_RT_Latency_平均RT时延权重": _safe_nanmean(w_rt_latency),
            "Final_W_RT_Latency_最终RT时延权重": _safe_last(w_rt_latency),
            "Mean_W_Batch_Latency_平均Batch时延权重": _safe_nanmean(w_batch_latency),
            "Final_W_Batch_Latency_最终Batch时延权重": _safe_last(w_batch_latency),
            "Mean_W_AI_Latency_平均AI时延权重": _safe_nanmean(w_ai_latency),
            "Final_W_AI_Latency_最终AI时延权重": _safe_last(w_ai_latency),
            "Mean_W_Queue_平均队列权重": _safe_nanmean(w_queue_value),
            "Final_W_Queue_最终队列权重": _safe_last(w_queue_value),
            "Mean_W_Risk_Scale_平均风险权重": _safe_nanmean(w_risk_scale_value),
            "Final_W_Risk_Scale_最终风险权重": _safe_last(w_risk_scale_value),
            "Mean_Cloud_Gate_平均云门控": _safe_nanmean(cloud_gate_value),
            "Final_Cloud_Gate_最终云门控": _safe_last(cloud_gate_value),
            "Mean_W_Energy_Scale_last_平均全局能耗权重": _safe_nanmean(w_energy_scale_last),
            "Final_W_Energy_Scale_last_最终全局能耗权重": _safe_last(w_energy_scale_last),
            "Mean_W_RT_Energy_last_平均RT能耗权重": _safe_nanmean(w_rt_energy_last),
            "Final_W_RT_Energy_last_最终RT能耗权重": _safe_last(w_rt_energy_last),
            "Mean_W_Batch_Energy_last_平均Batch能耗权重": _safe_nanmean(w_batch_energy_last),
            "Final_W_Batch_Energy_last_最终Batch能耗权重": _safe_last(w_batch_energy_last),
            "Mean_W_AI_Energy_last_平均AI能耗权重": _safe_nanmean(w_ai_energy_last),
            "Final_W_AI_Energy_last_最终AI能耗权重": _safe_last(w_ai_energy_last),
        }
        for phase in phases:
            pidx = phase["phase_idx"]
            def seg(vals):
                return _as_clean_numeric_list(_slice_metric_by_phase(vals, phase))
            for name, vals, cn in [
                ("Reward", reward, "评分"), ("Avg_Delay", delay, "平均时延"), ("Avg_Energy", energy, "平均能耗"),
                ("Total_Energy", total_energy, "窗口总能耗"), ("Violation", vio, "违约率"), ("SLA", sla, "SLA成功率"),
                ("Backlog", backlog, "积压"), ("Completion_Rate", completion, "完成率"),
            ]:
                ss = seg(vals)
                row[f"Phase{pidx}_{name}_Mean_{cn}均值"] = _safe_nanmean(ss)
                row[f"Phase{pidx}_{name}_Final_{cn}末值"] = _safe_last(ss)
                row[f"Phase{pidx}_{name}_Peak_{cn}峰值"] = _safe_nanmax(ss)
            if pidx >= 2:
                row[f"Phase{pidx}_Backlog_Recovery_Iter_积压恢复轮次"] = _recovery_iteration(backlog, phase, threshold=1.0)
        rows.append(row)
    return pd.DataFrame(rows)

def save_extra_diagnostics(group_logs):
    """统一保存增强诊断输出。"""
    all_alloc_type_summary_frames = []
    for group_key, info in group_logs.items():
        alloc_type_frames = []
        for run_idx, raw_log in enumerate(info.get("logs", []), start=1):
            atdf = build_alloc_by_type_debug_dataframe(raw_log, group_key=group_key, group_label=info["label"], run_index=run_idx)
            if not atdf.empty:
                alloc_type_frames.append(atdf)
        if alloc_type_frames:
            alloc_type_df = pd.concat(alloc_type_frames, ignore_index=True)
            alloc_type_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_alloc_by_type_debug_任务类型节点分配调试.csv"), index=False)
            alloc_type_summary_df = build_alloc_by_type_summary_dataframe(alloc_type_df)
            alloc_type_summary_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_alloc_by_type_summary_任务类型节点分配汇总.csv"), index=False)
            all_alloc_type_summary_frames.append(alloc_type_summary_df)
            plot_alloc_by_type_summary(alloc_type_summary_df, save_dir=SCENARIO_SAVE_DIR, group_key=group_key, group_label=info["label"])
    if all_alloc_type_summary_frames:
        all_summary = pd.concat(all_alloc_type_summary_frames, ignore_index=True)
        all_summary.to_csv(os.path.join(SCENARIO_SAVE_DIR, "alloc_by_type_all_methods_summary_全部方法任务类型分配汇总.csv"), index=False)
        plot_alloc_by_type_method_compare(all_summary, SCENARIO_SAVE_DIR)
    key_summary = build_key_metric_summary_dataframe(group_logs)
    key_summary.to_csv(os.path.join(SCENARIO_SAVE_DIR, "key_metric_summary_核心指标统计.csv"), index=False)
    plot_round_mean_energy_delay_score(group_logs, save_dir=SCENARIO_SAVE_DIR)
    plot_cohort_learning_curves(group_logs, save_dir=SCENARIO_SAVE_DIR)
    plot_theta_trajectory(group_logs, save_dir=SCENARIO_SAVE_DIR)

def plot_group_alloc_heatmaps(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario"):
    group_items = list(group_logs.items())
    fig, axes = plt.subplots(len(group_items), 1, figsize=(12, 2.6 * max(1, len(group_items))))
    if len(group_items) == 1:
        axes = [axes]
    for ax, (group_key, info) in zip(axes, group_items):
        log = aggregate_logs(info["logs"])
        alloc = np.array(log.get("alloc", []), dtype=float) if log.get("alloc") else np.zeros((1, len(CFG.NODES_CFG)))
        if alloc.ndim == 1:
            alloc = alloc.reshape(1, -1)
        im = ax.imshow(alloc.T, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_title(f"Node Allocation Heatmap - {info['label']}")
        ax.set_xlabel("BO Iteration")
        ax.set_ylabel("Node Index")
        ax.set_yticks(np.arange(len(CFG.NODES_CFG)))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_alloc_heatmaps.png"), dpi=300)
    plt.close(fig)

def plot_group_task_delay_bars(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario"):
    labels = []
    rt_vals = []
    batch_vals = []
    ai_vals = []
    for _, info in group_logs.items():
        log = aggregate_logs(info["logs"])
        labels.append(info["label"])
        rt_vals.append(float(np.nanmean(log.get("avg_delay_rt", [np.nan]))))
        batch_vals.append(float(np.nanmean(log.get("avg_delay_batch", [np.nan]))))
        ai_vals.append(float(np.nanmean(log.get("avg_delay_ai", [np.nan]))))
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    ax.bar(x - width, rt_vals, width=width, label="RT")
    ax.bar(x, batch_vals, width=width, label="Batch")
    ax.bar(x + width, ai_vals, width=width, label="AI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("Average Delay by Task Type Across Control Groups")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_task_delay_bars.png"), dpi=300)
    plt.close(fig)
