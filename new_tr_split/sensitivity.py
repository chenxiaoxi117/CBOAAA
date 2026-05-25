#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 3989-4445.
# Sensitivity-analysis evaluation and report helpers.

SENSITIVITY_SCENARIO_PRESETS = {
    "default": None,  # 使用当前 CFG.TASK_TYPE_PROBS
    "balanced": {"RT": 0.34, "Batch": 0.33, "AI": 0.33},
    "rt_high": {"RT": 0.70, "Batch": 0.20, "AI": 0.10},
    "batch_high": {"RT": 0.10, "Batch": 0.70, "AI": 0.20},
    "ai_high": {"RT": 0.20, "Batch": 0.10, "AI": 0.70},
}


def _normalize_task_probs(probs):
    vals = {t: float(probs.get(t, 0.0)) for t in TASK_TYPE_ORDER}
    total = sum(vals.values())
    if total <= 0:
        return dict(CFG.TASK_TYPE_PROBS)
    return {t: vals[t] / total for t in TASK_TYPE_ORDER}


def _parse_sensitivity_scenarios(spec):
    """解析敏感度场景。

    支持：
    - default,rt_high,batch_high,ai_high
    - 自定义 name:RT,Batch,AI，例如 mix1:0.2,0.5,0.3
    """
    if not spec:
        spec = "default,rt_high,batch_high,ai_high"
    scenarios = []
    for raw in str(spec).split(','):
        raw = raw.strip()
        if not raw:
            continue
        if ':' in raw:
            name, vals = raw.split(':', 1)
            parts = [float(x) for x in vals.replace('/', ',').split(',') if x.strip()]
            if len(parts) != 3:
                raise ValueError(f"自定义场景 {raw} 需要三个比例：RT,Batch,AI")
            probs = _normalize_task_probs({"RT": parts[0], "Batch": parts[1], "AI": parts[2]})
            scenarios.append((name.strip(), probs))
        else:
            key = raw.lower()
            if key not in SENSITIVITY_SCENARIO_PRESETS:
                raise ValueError(f"未知敏感度场景 {raw}. 可选: {list(SENSITIVITY_SCENARIO_PRESETS)} 或 name:rt,batch,ai")
            preset = SENSITIVITY_SCENARIO_PRESETS[key]
            probs = dict(CFG.TASK_TYPE_PROBS) if preset is None else _normalize_task_probs(preset)
            scenarios.append((key, probs))
    return scenarios


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _mean_or_nan(values):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _std_or_nan(values):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if vals else np.nan


def _sum_alloc_by_type(log):
    """把多个窗口的分任务分配计数相加。"""
    result = {t: [0 for _ in CFG.NODES_CFG] for t in TASK_TYPE_ORDER}
    for item in log.get("alloc_by_type", []):
        if not isinstance(item, dict):
            continue
        for t in TASK_TYPE_ORDER:
            vals = item.get(t, [])
            for i, v in enumerate(vals[:len(CFG.NODES_CFG)]):
                try:
                    result[t][i] += int(v)
                except Exception:
                    pass
    return result


def _allocation_top_nodes(alloc_counts, k=3):
    order = sorted(range(len(alloc_counts)), key=lambda i: (-alloc_counts[i], i))
    return order[:k]


def _eval_theta_windows(theta, seed, windows=3, scenario_name="default", task_probs=None):
    """固定一个 theta 连续运行若干 BO 窗口，用于敏感度诊断。

    注意：这里不调用 agent.ask，因此不是 BO 搜索；只是评价某个 theta 在同一任务分布下的平均表现。
    """
    old_probs = dict(CFG.TASK_TYPE_PROBS)
    if task_probs is not None:
        CFG.TASK_TYPE_PROBS = _normalize_task_probs(task_probs)
    try:
        fac = ConnectedFactory(fid=0, name=f"Sensitivity_{scenario_name}", seed=seed, node_config=CFG.NODES_CFG)
        fac.reset(use_batch=False)
        for _ in range(max(1, int(windows))):
            fac.current_control_label = "SensitivityFixedTheta"
            fac.run_continuous(theta)
        log = fac.perf_log
        rewards = log.get("reward", [])
        costs = [-_safe_float(x) for x in rewards]
        alloc_by_type = _sum_alloc_by_type(log)
        row = {
            "Seed": int(seed),
            "Windows": int(windows),
            "Mean_Cost": _mean_or_nan(costs),
            "Std_Cost_Window": _std_or_nan(costs),
            "Mean_Reward": _mean_or_nan(rewards),
            "Mean_Avg_Delay": _mean_or_nan(log.get("avg_delay", [])),
            "Mean_Avg_Delay_RT": _mean_or_nan(log.get("avg_delay_rt", [])),
            "Mean_Avg_Delay_Batch": _mean_or_nan(log.get("avg_delay_batch", [])),
            "Mean_Avg_Delay_AI": _mean_or_nan(log.get("avg_delay_ai", [])),
            "Mean_Avg_Energy": _mean_or_nan(log.get("avg_energy", [])),
            "Mean_Total_Energy": _mean_or_nan(log.get("total_energy", [])),
            "Final_Cumulative_Energy": _safe_float(log.get("cumulative_energy", [np.nan])[-1]) if log.get("cumulative_energy") else np.nan,
            "Mean_SLA_Success": _mean_or_nan(log.get("sla_success_rate", [])),
            "Mean_Violation_Rate": _mean_or_nan(log.get("vio_rate", [])),
            "Mean_Backlog": _mean_or_nan(log.get("backlog", [])),
            "Mean_Arrivals": _mean_or_nan(log.get("arrivals_total", [])),
            "Mean_Completed": _mean_or_nan(log.get("completed_total", [])),
            "Alloc_By_Type_JSON": json.dumps(alloc_by_type, ensure_ascii=False),
            "Alloc_RT_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("RT", []), 3)),
            "Alloc_Batch_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("Batch", []), 3)),
            "Alloc_AI_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("AI", []), 3)),
        }
        return row
    finally:
        CFG.TASK_TYPE_PROBS = old_probs


def _candidate_scores_for_task(theta, task_type, origin_node_id=0, seed=0, norm_mode="rolling"):
    """静态计算某任务类型在代表性 origin 下的候选节点排序。

    这个函数不抽样、不执行任务，只复用调度器的 score/feasibility/opportunity 逻辑，
    用来回答：改变某个参数是否改变了该任务的节点偏好排序。
    """
    props = CFG.TASK_PROPS[task_type]
    task = Task(
        id=f"pref-{task_type}-{origin_node_id}",
        create_time=0.0,
        data_size=props["data"],
        cpu_req=props["cpu"],
        duration_base=props["dur"],
        task_type=task_type,
        deadline_factor=props["deadline_factor"],
        origin_node_id=int(origin_node_id),
    )
    nodes = [Node(cfg) for cfg in CFG.NODES_CFG]
    scheduler = ConstrainedBoltzmannScheduler(np_rng=np.random.default_rng(resolve_base_seed(seed, stream=810)), norm_mode=norm_mode)
    scheduler.current_time = 0.0
    latency_weights, energy_weights, _ = split_task_weights(theta)
    controls = extract_scheduler_controls(theta)
    scheduler.beta = float(controls.get("beta", scheduler.beta))
    latency_w = float(latency_weights.get(task_type, 1.0))
    energy_w = float(energy_weights.get(task_type, 1.0))
    base_risk_w = float(CFG.TASK_RISK_WEIGHTS.get(task_type, CFG.DEADLINE_WEIGHT))
    risk_w = base_risk_w * float(controls.get("risk_scale", getattr(CFG, "RISK_SCALE_DEFAULT", 1.0)))
    queue_w = float(controls.get("queue_w", getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0))) if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True) else 0.0

    raw_infos = [scheduler._node_score(task, idx, node, latency_w, energy_w, risk_w) for idx, node in enumerate(nodes)]
    candidates, feasibility_debug = scheduler._apply_feasibility_filter(task, nodes, raw_infos, controls)
    if not candidates:
        candidates = [min(raw_infos, key=lambda c: c.get("latency_total", 1e18))]
        feasibility_debug["fallback_reason"] = "empty_after_all_filters_static_diag"
    norm_e, norm_l, norm_r, norm_q, norm_debug = scheduler._compute_norms_with_queue(task, candidates)
    alpha, _alpha_source, tradeoff_mode = scheduler._resolve_scheduler_alpha(task_type, latency_w, energy_w)
    for i, c in enumerate(candidates):
        c["norm_e"] = float(norm_e[i])
        c["norm_l"] = float(norm_l[i])
        c["norm_risk"] = float(norm_r[i])
        c["norm_queue"] = float(norm_q[i])
        score, latency_energy_component, energy_component = scheduler._score_candidate_components(
            c, latency_w, energy_w, risk_w, queue_w=queue_w, alpha=alpha, tradeoff_mode=tradeoff_mode
        )
        c["latency_energy_component"] = latency_energy_component
        c["base_latency_energy_score"] = latency_energy_component
        c["service_component"] = latency_energy_component  # Deprecated alias.
        c["energy_component"] = energy_component
        c["score"] = score
    opportunity_candidates, opportunity_debug = scheduler._apply_opportunity_window(candidates, controls)
    opp_nodes = {int(c["node_idx"]) for c in opportunity_candidates}
    sorted_candidates = sorted(candidates, key=lambda c: (float(c.get("score", np.inf)), int(c.get("node_idx", 999))))
    candidate_rank = {int(c["node_idx"]): i + 1 for i, c in enumerate(sorted_candidates)}
    sorted_opp = sorted(opportunity_candidates, key=lambda c: (float(c.get("score", np.inf)), int(c.get("node_idx", 999))))
    opp_rank = {int(c["node_idx"]): i + 1 for i, c in enumerate(sorted_opp)}

    rows = []
    for c in sorted_candidates:
        node_idx = int(c["node_idx"])
        node_cfg = CFG.NODES_CFG[node_idx]
        rows.append({
            "Task_Type": task_type,
            "Origin_Node": int(origin_node_id),
            "Node": node_idx,
            "Node_Role": str(node_cfg.get("role", "")),
            "Node_Workshop": _get_node_site_from_cfg(node_cfg, default_site=node_idx // 2),
            "Is_Cloud": bool(_node_is_cloud(node_cfg)),
            "Rank_Candidate": int(candidate_rank[node_idx]),
            "Rank_Opportunity": int(opp_rank[node_idx]) if node_idx in opp_rank else np.nan,
            "In_Opportunity": int(node_idx in opp_nodes),
            "Score": float(c.get("score", np.nan)),
            "Latency": float(c.get("latency_total", np.nan)),
            "Energy": float(c.get("energy_raw", np.nan)),
            "Risk": float(c.get("deadline_risk", np.nan)),
            "Queue": float(c.get("queue_pressure", np.nan)),
            "Slack": float(c.get("slack", np.nan)),
            "Predicted_Finish": float(c.get("predicted_finish", np.nan)),
            "Feasible_Count": int(len(candidates)),
            "Opportunity_Count": int(len(opportunity_candidates)),
            "Fallback_Reason": feasibility_debug.get("fallback_reason"),
            "Cloud_Pressure": feasibility_debug.get("cloud_pressure"),
        })
    return rows


def _diagnose_node_preferences(theta, seed=0, origins=None, norm_mode="rolling"):
    if origins is None:
        origins = [cfg["id"] for cfg in CFG.NODES_CFG if (not _node_is_cloud(cfg)) and cfg.get("id") in [0, 2, 4, 6, 8]]
        if not origins:
            origins = [0]
    rows = []
    for task_type in TASK_TYPE_ORDER:
        for origin in origins:
            rows.extend(_candidate_scores_for_task(theta, task_type, origin_node_id=origin, seed=seed, norm_mode=norm_mode))
    return rows


def _rank_map(df, task_type, origin, rank_col="Rank_Candidate"):
    sub = df[(df["Task_Type"] == task_type) & (df["Origin_Node"] == origin)]
    out = {}
    for _, r in sub.iterrows():
        val = r.get(rank_col, np.nan)
        if pd.notna(val):
            out[int(r["Node"])] = int(val)
    return out


def _topk_from_rank_map(rank_map, k=3):
    return [node for node, _ in sorted(rank_map.items(), key=lambda kv: (kv[1], kv[0]))[:k]]


def _kendall_distance_rank_maps(a, b):
    nodes = sorted(set(a.keys()) | set(b.keys()))
    n = len(nodes)
    if n <= 1:
        return 0.0
    missing_rank = n + 1
    inv = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = nodes[i], nodes[j]
            da = (a.get(ni, missing_rank) - a.get(nj, missing_rank))
            db = (b.get(ni, missing_rank) - b.get(nj, missing_rank))
            if da == 0 or db == 0:
                continue
            total += 1
            if da * db < 0:
                inv += 1
    return float(inv / total) if total > 0 else 0.0


def _preference_compare_rows(base_pref_df, pref_df, scenario_name, dimension, value, theta_id):
    rows = []
    for task_type in TASK_TYPE_ORDER:
        origins = sorted(set(base_pref_df[base_pref_df["Task_Type"] == task_type]["Origin_Node"].tolist()) |
                         set(pref_df[pref_df["Task_Type"] == task_type]["Origin_Node"].tolist()))
        for origin in origins:
            base_rank = _rank_map(base_pref_df, task_type, origin)
            cur_rank = _rank_map(pref_df, task_type, origin)
            if not base_rank or not cur_rank:
                continue
            base_top1 = _topk_from_rank_map(base_rank, 1)[0]
            cur_top1 = _topk_from_rank_map(cur_rank, 1)[0]
            base_top3 = set(_topk_from_rank_map(base_rank, 3))
            cur_top3 = set(_topk_from_rank_map(cur_rank, 3))
            union = base_top3 | cur_top3
            jaccard_dist = 1.0 - (len(base_top3 & cur_top3) / max(1, len(union)))
            rows.append({
                "Scenario": scenario_name,
                "Dimension": dimension,
                "Value": value,
                "Theta_ID": theta_id,
                "Task_Type": task_type,
                "Origin_Node": int(origin),
                "Top1_Base": int(base_top1),
                "Top1_Current": int(cur_top1),
                "Top1_Changed": int(base_top1 != cur_top1),
                "Top3_Base": json.dumps(sorted(base_top3)),
                "Top3_Current": json.dumps(sorted(cur_top3)),
                "Top3_Jaccard_Distance": float(jaccard_dist),
                "Kendall_Distance": float(_kendall_distance_rank_maps(base_rank, cur_rank)),
            })
    return rows


def _spearman_corr_safe(x, y):
    try:
        ser = pd.Series(x, dtype="float64")
        val = ser.corr(pd.Series(y, dtype="float64"), method="spearman")
        return float(val) if pd.notna(val) else np.nan
    except Exception:
        return np.nan


def run_full_sensitivity_analysis(points=5, seeds=2, windows=3, scenario_spec="default,rt_high,batch_high,ai_high", output_dir=None, greedy=False, pref_norm_mode="rolling"):
    """保留 11 维控制量，做参数敏感度 + 节点偏好诊断。

    输出文件：
    - sensitivity_eval_raw.csv：每个 theta、seed 的窗口评价原始结果；
    - sensitivity_eval_summary.csv：跨 seed 聚合后的性能结果；
    - sensitivity_preference_raw.csv：每个 theta 下 RT/Batch/AI 的候选节点排序；
    - sensitivity_preference_compare.csv：相对 baseline 的排序变化；
    - sensitivity_dimension_summary.csv：每个参数维度的综合敏感度结论。
    """
    root = output_dir or os.path.abspath("sensitivity_full_11d")
    os.makedirs(root, exist_ok=True)
    old_random = CFG.USE_BOLTZMANN_RANDOM
    if greedy:
        CFG.USE_BOLTZMANN_RANDOM = False
    try:
        bounds = get_control_bounds(CFG.DIM_THETA)
        low = np.array(bounds[0], dtype=float)
        high = np.array(bounds[1], dtype=float)
        base_theta = default_control_vector(fill=1.5)
        scenarios = _parse_sensitivity_scenarios(scenario_spec)
        eval_rows = []
        pref_rows = []
        pref_cmp_rows = []

        for scenario_name, probs in scenarios:
            print(f"\n=== Sensitivity scenario: {scenario_name}, probs={probs} ===")
            # baseline preference for each scenario; static preference does not depend on task probs,
            # but keeping scenario in output makes downstream analysis easier.
            base_pref_rows = _diagnose_node_preferences(base_theta, seed=CFG.BASE_SEED, norm_mode=pref_norm_mode)
            base_pref_df = pd.DataFrame(base_pref_rows)
            for r in base_pref_rows:
                rr = dict(r)
                rr.update({"Scenario": scenario_name, "Dimension": "BASE", "Value": np.nan, "Theta_ID": "BASE"})
                rr.update(theta_to_named_dict(base_theta))
                pref_rows.append(rr)

            theta_jobs = []
            theta_jobs.append(("BASE", "BASE", np.nan, list(base_theta)))
            for dim_idx, dim_name in enumerate(CFG.FEATURE_NAMES):
                xs = np.linspace(low[dim_idx], high[dim_idx], max(2, int(points)))
                for v in xs:
                    theta = list(base_theta)
                    theta[dim_idx] = float(v)
                    theta_id = f"{dim_name}={float(v):.6g}"
                    theta_jobs.append((dim_name, theta_id, float(v), theta))

            for job_idx, (dim_name, theta_id, value, theta) in enumerate(theta_jobs, start=1):
                print(f"  [{scenario_name}] {job_idx}/{len(theta_jobs)} {theta_id}")
                # 静态节点偏好诊断：每个 theta 只做一次。
                pref = _diagnose_node_preferences(theta, seed=CFG.BASE_SEED, norm_mode=pref_norm_mode)
                pref_df = pd.DataFrame(pref)
                for r in pref:
                    rr = dict(r)
                    rr.update({"Scenario": scenario_name, "Dimension": dim_name, "Value": value, "Theta_ID": theta_id})
                    rr.update(theta_to_named_dict(theta))
                    pref_rows.append(rr)
                pref_cmp_rows.extend(_preference_compare_rows(base_pref_df, pref_df, scenario_name, dim_name, value, theta_id))

                for sidx in range(max(1, int(seeds))):
                    seed = CFG.BASE_SEED + sidx
                    row = _eval_theta_windows(theta, seed=seed, windows=windows, scenario_name=scenario_name, task_probs=probs)
                    row.update({"Scenario": scenario_name, "Dimension": dim_name, "Value": value, "Theta_ID": theta_id})
                    row.update(theta_to_named_dict(theta))
                    eval_rows.append(row)

        eval_raw = pd.DataFrame(eval_rows)
        pref_raw = pd.DataFrame(pref_rows)
        pref_cmp = pd.DataFrame(pref_cmp_rows)

        eval_raw.to_csv(os.path.join(root, "sensitivity_eval_raw.csv"), index=False)
        pref_raw.to_csv(os.path.join(root, "sensitivity_preference_raw.csv"), index=False)
        pref_cmp.to_csv(os.path.join(root, "sensitivity_preference_compare.csv"), index=False)

        group_cols = ["Scenario", "Dimension", "Theta_ID", "Value"]
        metric_cols = [
            "Mean_Cost", "Mean_Reward", "Mean_Avg_Delay", "Mean_Avg_Delay_RT", "Mean_Avg_Delay_Batch", "Mean_Avg_Delay_AI",
            "Mean_Avg_Energy", "Mean_Total_Energy", "Final_Cumulative_Energy", "Mean_SLA_Success", "Mean_Violation_Rate", "Mean_Backlog",
            "Mean_Arrivals", "Mean_Completed"
        ]
        agg_spec = {m: ["mean", "std"] for m in metric_cols if m in eval_raw.columns}
        eval_summary = eval_raw.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
        eval_summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in eval_summary.columns]
        eval_summary.to_csv(os.path.join(root, "sensitivity_eval_summary.csv"), index=False)

        dim_rows = []
        for scenario_name, _ in scenarios:
            base_cost_vals = eval_raw[(eval_raw["Scenario"] == scenario_name) & (eval_raw["Dimension"] == "BASE")]["Mean_Cost"].tolist()
            base_cost = _mean_or_nan(base_cost_vals)
            for dim_name in CFG.FEATURE_NAMES:
                sub = eval_summary[(eval_summary["Scenario"] == scenario_name) & (eval_summary["Dimension"] == dim_name)].copy()
                if sub.empty:
                    continue
                values = sub["Value_"].tolist() if "Value_" in sub.columns else sub["Value"].tolist()
                cost_mean_col = "Mean_Cost_mean"
                cost_std_col = "Mean_Cost_std"
                costs = sub[cost_mean_col].tolist() if cost_mean_col in sub.columns else []
                delays = sub["Mean_Avg_Delay_mean"].tolist() if "Mean_Avg_Delay_mean" in sub.columns else []
                energies = sub["Mean_Avg_Energy_mean"].tolist() if "Mean_Avg_Energy_mean" in sub.columns else []
                vios = sub["Mean_Violation_Rate_mean"].tolist() if "Mean_Violation_Rate_mean" in sub.columns else []
                cost_range = float(np.nanmax(costs) - np.nanmin(costs)) if costs else np.nan
                delay_range = float(np.nanmax(delays) - np.nanmin(delays)) if delays else np.nan
                energy_range = float(np.nanmax(energies) - np.nanmin(energies)) if energies else np.nan
                vio_range = float(np.nanmax(vios) - np.nanmin(vios)) if vios else np.nan
                avg_seed_std = _mean_or_nan(sub[cost_std_col].tolist()) if cost_std_col in sub.columns else np.nan
                rel_cost_range = cost_range / max(abs(base_cost), 1e-9) if np.isfinite(cost_range) and np.isfinite(base_cost) else np.nan
                noise_ratio = avg_seed_std / max(cost_range, 1e-9) if np.isfinite(avg_seed_std) and np.isfinite(cost_range) else np.nan
                cmp_sub = pref_cmp[(pref_cmp["Scenario"] == scenario_name) & (pref_cmp["Dimension"] == dim_name)]
                mean_kendall = _mean_or_nan(cmp_sub.get("Kendall_Distance", [])) if not cmp_sub.empty else np.nan
                top1_change = _mean_or_nan(cmp_sub.get("Top1_Changed", [])) if not cmp_sub.empty else np.nan
                top3_dist = _mean_or_nan(cmp_sub.get("Top3_Jaccard_Distance", [])) if not cmp_sub.empty else np.nan

                if (np.isfinite(rel_cost_range) and rel_cost_range < 0.02 and
                    np.isfinite(top1_change) and top1_change < 0.10 and
                    np.isfinite(mean_kendall) and mean_kendall < 0.05):
                    decision = "weak_fix_candidate"
                elif np.isfinite(noise_ratio) and noise_ratio > 1.0 and (not np.isfinite(rel_cost_range) or rel_cost_range < 0.05):
                    decision = "noise_dominated"
                elif np.isfinite(top1_change) and top1_change >= 0.25:
                    decision = "preference_sensitive_keep_or_ablate"
                elif np.isfinite(rel_cost_range) and rel_cost_range >= 0.05:
                    decision = "performance_sensitive_keep"
                else:
                    decision = "unclear_need_repeat"

                dim_rows.append({
                    "Scenario": scenario_name,
                    "Dimension": dim_name,
                    "Base_Cost": base_cost,
                    "Cost_Range": cost_range,
                    "Relative_Cost_Range": rel_cost_range,
                    "Avg_Seed_Cost_Std": avg_seed_std,
                    "Seed_Noise_To_Effect_Ratio": noise_ratio,
                    "Cost_Spearman_With_Value": _spearman_corr_safe(values, costs),
                    "Delay_Range": delay_range,
                    "Energy_Range": energy_range,
                    "Violation_Range": vio_range,
                    "Mean_Kendall_Distance": mean_kendall,
                    "Top1_Change_Rate": top1_change,
                    "Top3_Jaccard_Distance": top3_dist,
                    "Decision": decision,
                })
        dim_summary = pd.DataFrame(dim_rows)
        dim_summary.to_csv(os.path.join(root, "sensitivity_dimension_summary.csv"), index=False)
        print(f"\n=== Sensitivity finished. Outputs saved to: {root} ===")
        return dim_summary
    finally:
        CFG.USE_BOLTZMANN_RANDOM = old_random
