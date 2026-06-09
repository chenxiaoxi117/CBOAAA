#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 11228-11629.
# Offline noise diagnostics and short-name result export.

def _noise_numeric_array(values):
    out = []
    for v in values or []:
        try:
            x = float(v)
            if np.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def _offline_default_fixed_keys():
    return [
        "reduced6_fixed_mid",
        "reduced6_fixed_tuned",
        "reduced6_fixed_queue_high",
        "reduced6_fixed_risk_high",
        "reduced6_fixed_edge_safe",
    ]


def _extract_cost_series_from_log(log):
    """Return per-window evaluation cost.

    Preferred source is eval_cost if the refactor logging path filled it.
    Fallback is -reward, which is equivalent to metrics["cost"] in the
    original window feedback path.
    """
    eval_cost = log.get("eval_cost", []) if isinstance(log, dict) else []
    arr = _noise_numeric_array(eval_cost)
    if len(arr) > 0:
        return arr
    reward = log.get("reward", []) if isinstance(log, dict) else []
    r = _noise_numeric_array(reward)
    return -r


def run_offline_window_noise_diagnostic(repeat_runs=10, selected_keys=None, output_dir=None):
    """Offline diagnostic: estimate whether window-level feedback noise is large.

    This is NOT an online training method. It repeatedly evaluates fixed policies
    under the same scenario configuration and compares:
      1. within-policy per-window cost volatility;
      2. between-policy mean gaps;
      3. paired same-seed/same-iteration deltas, approximating a CRN-style check;
      4. rank stability across seeds.

    Use this to answer: are differences among fixed policies larger than the
    per-window noise BO receives as one observation?
    """
    global SCENARIO_SAVE_DIR
    old_save_dir = SCENARIO_SAVE_DIR
    root = output_dir or os.path.abspath("offline_window_noise_diagnostic")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    SCENARIO_SAVE_DIR = root

    groups = build_scenario_method_groups()
    if selected_keys is None:
        selected_keys = _offline_default_fixed_keys()
    selected_keys = normalize_selected_method_keys([str(k).strip() for k in selected_keys if str(k).strip()])
    selected_keys = [k for k in selected_keys if k in groups and groups[k].get("agent") is None and "fixed_theta" in groups[k]]
    if not selected_keys:
        raise ValueError("offline_noise requires fixed policy keys. Example: --selected-keys fixed_mid,fixed_tuned,fixed_queue_high,fixed_risk_high,fixed_edge_safe")

    config_payload = {
        "diagnostic": "offline_window_noise",
        "refactor_version": REFACTOR_VERSION,
        "selected_keys": selected_keys,
        "repeat_runs": int(max(1, repeat_runs)),
        "bo_iterations": int(CFG.BO_ITERATIONS),
        "bo_interval": float(CFG.BO_INTERVAL),
        "session_duration": float(CFG.SESSION_DURATION),
        "lambda_schedule": list(getattr(CFG, "LAMBDA_SCHEDULE", [])),
        "task_type_probs": dict(getattr(CFG, "TASK_TYPE_PROBS", {})),
        "use_task_type_adaptation": bool(getattr(CFG, "USE_TASK_TYPE_ADAPTATION", False)),
        "cloud_delay_mult": float(getattr(CFG, "CLOUD_DELAY_MULT", 1.0)),
        "cloud_energy_mult": float(getattr(CFG, "CLOUD_ENERGY_MULT", 1.0)),
        "cloud_speed_mult": float(getattr(CFG, "CLOUD_SPEED_MULT", 1.0)),
        "note": "This diagnostic does not train BO. It quantifies fixed-policy window-cost variance and policy gap/noise ratios.",
    }
    with open(os.path.join(root, "offline_window_noise_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    sample_rows = []
    run_mean_rows = []
    print("=== Offline Window Noise Diagnostic ===")
    print(f"methods={selected_keys}, repeats={repeat_runs}, output={root}")
    try:
        for run_idx in range(max(1, repeat_runs)):
            seed = CFG.BASE_SEED + run_idx
            print(f"[Noise repeat {run_idx + 1}/{max(1, repeat_runs)}] seed={seed}")
            for key in selected_keys:
                cfg = groups[key]
                log = run_scenario_group(seed, key, cfg)
                costs = _extract_cost_series_from_log(log)
                rewards = _noise_numeric_array(log.get("reward", []))
                arrivals = _noise_numeric_array(log.get("arrivals_total", []))
                completed = _noise_numeric_array(log.get("completed_total", []))
                backlog = _noise_numeric_array(log.get("backlog", []))
                feedback_conf = _noise_numeric_array(log.get("feedback_confidence", []))
                n = int(len(costs))
                for i in range(n):
                    sample_rows.append({
                        "Seed": int(seed),
                        "Repeat_Index": int(run_idx),
                        "Method_Key": key,
                        "Method_Label": cfg.get("label", key),
                        "Iteration": int(i + 1),
                        "Window_Cost": float(costs[i]),
                        "Reward": float(rewards[i]) if i < len(rewards) else np.nan,
                        "Arrivals": float(arrivals[i]) if i < len(arrivals) else np.nan,
                        "Completed": float(completed[i]) if i < len(completed) else np.nan,
                        "Backlog": float(backlog[i]) if i < len(backlog) else np.nan,
                        "Feedback_Confidence": float(feedback_conf[i]) if i < len(feedback_conf) else np.nan,
                    })
                if n > 0:
                    run_mean_rows.append({
                        "Seed": int(seed),
                        "Repeat_Index": int(run_idx),
                        "Method_Key": key,
                        "Method_Label": cfg.get("label", key),
                        "Run_Mean_Cost": float(np.nanmean(costs)),
                        "Run_Std_Window_Cost": float(np.nanstd(costs, ddof=1)) if n > 1 else 0.0,
                        "Run_CV_Window_Cost": float(np.nanstd(costs, ddof=1) / max(1e-12, abs(np.nanmean(costs)))) if n > 1 else 0.0,
                    })

        sample_df = pd.DataFrame(sample_rows)
        run_df = pd.DataFrame(run_mean_rows)
        sample_path = os.path.join(root, "offline_window_noise_samples.csv")
        run_path = os.path.join(root, "offline_window_noise_run_means.csv")
        sample_df.to_csv(sample_path, index=False)
        run_df.to_csv(run_path, index=False)

        method_rows = []
        for key, sub in sample_df.groupby("Method_Key"):
            costs = sub["Window_Cost"].astype(float).to_numpy()
            run_sub = run_df[run_df["Method_Key"] == key]
            mean_cost = float(np.nanmean(costs)) if len(costs) else np.nan
            std_window = float(np.nanstd(costs, ddof=1)) if len(costs) > 1 else 0.0
            std_run_mean = float(np.nanstd(run_sub["Run_Mean_Cost"].astype(float), ddof=1)) if len(run_sub) > 1 else 0.0
            method_rows.append({
                "Method_Key": key,
                "Method_Label": str(sub["Method_Label"].iloc[0]),
                "Mean_Window_Cost": mean_cost,
                "Std_Window_Cost": std_window,
                "CV_Window_Cost": std_window / max(1e-12, abs(mean_cost)) if np.isfinite(mean_cost) else np.nan,
                "Mean_Run_Cost": float(np.nanmean(run_sub["Run_Mean_Cost"].astype(float))) if len(run_sub) else np.nan,
                "Std_Run_Mean_Cost": std_run_mean,
                "Num_Windows": int(len(costs)),
                "Num_Repeats": int(len(run_sub)),
            })
        method_df = pd.DataFrame(method_rows).sort_values("Mean_Window_Cost")
        method_path = os.path.join(root, "offline_window_noise_method_summary.csv")
        method_df.to_csv(method_path, index=False)

        # Paired same-seed/same-iteration policy deltas.
        pair_rows = []
        for i, key_a in enumerate(selected_keys):
            for key_b in selected_keys[i + 1:]:
                a = sample_df[sample_df["Method_Key"] == key_a][["Seed", "Iteration", "Window_Cost"]].rename(columns={"Window_Cost": "Cost_A"})
                b = sample_df[sample_df["Method_Key"] == key_b][["Seed", "Iteration", "Window_Cost"]].rename(columns={"Window_Cost": "Cost_B"})
                merged = pd.merge(a, b, on=["Seed", "Iteration"], how="inner")
                if merged.empty:
                    continue
                delta = merged["Cost_A"].astype(float) - merged["Cost_B"].astype(float)
                mean_a = float(np.nanmean(merged["Cost_A"].astype(float)))
                mean_b = float(np.nanmean(merged["Cost_B"].astype(float)))
                mean_delta = float(np.nanmean(delta))
                std_delta = float(np.nanstd(delta, ddof=1)) if len(delta) > 1 else 0.0
                pooled_std = float(np.sqrt(0.5 * (np.nanvar(merged["Cost_A"].astype(float), ddof=1) + np.nanvar(merged["Cost_B"].astype(float), ddof=1)))) if len(delta) > 1 else 0.0
                best_mean = min(mean_a, mean_b)
                pair_rows.append({
                    "Method_A": key_a,
                    "Method_B": key_b,
                    "Mean_Cost_A": mean_a,
                    "Mean_Cost_B": mean_b,
                    "Mean_Delta_A_minus_B": mean_delta,
                    "Std_Paired_Delta": std_delta,
                    "Abs_Delta_Over_StdDelta": abs(mean_delta) / max(1e-12, std_delta),
                    "Abs_Delta_Over_PooledWindowStd": abs(mean_delta) / max(1e-12, pooled_std),
                    "Relative_Gap_Pct_of_Best": 100.0 * abs(mean_delta) / max(1e-12, abs(best_mean)),
                    "Paired_Win_Rate_A_Lower": float(np.nanmean(delta < 0.0)),
                    "Num_Paired_Windows": int(len(delta)),
                })
        pair_df = pd.DataFrame(pair_rows)
        pair_path = os.path.join(root, "offline_window_noise_pairwise.csv")
        pair_df.to_csv(pair_path, index=False)

        # Rank stability by repeat/run mean.
        rank_rows = []
        if not run_df.empty:
            for seed, sub in run_df.groupby("Seed"):
                sub_sorted = sub.sort_values("Run_Mean_Cost")
                for rank, (_, row) in enumerate(sub_sorted.iterrows(), start=1):
                    rank_rows.append({
                        "Seed": int(seed),
                        "Method_Key": row["Method_Key"],
                        "Run_Mean_Cost": float(row["Run_Mean_Cost"]),
                        "Rank_LowerCostBetter": int(rank),
                        "Is_Top1": int(rank == 1),
                    })
        rank_df = pd.DataFrame(rank_rows)
        rank_path = os.path.join(root, "offline_window_noise_rank_by_seed.csv")
        rank_df.to_csv(rank_path, index=False)
        top1_df = pd.DataFrame()
        if not rank_df.empty:
            top1_df = rank_df.groupby("Method_Key", as_index=False).agg(
                Mean_Rank=("Rank_LowerCostBetter", "mean"),
                Std_Rank=("Rank_LowerCostBetter", "std"),
                Top1_Count=("Is_Top1", "sum"),
                Repeat_Count=("Is_Top1", "count"),
            )
            top1_df["Top1_Rate"] = top1_df["Top1_Count"] / top1_df["Repeat_Count"].clip(lower=1)
            top1_df = top1_df.sort_values(["Mean_Rank", "Top1_Rate"], ascending=[True, False])
            top1_df.to_csv(os.path.join(root, "offline_window_noise_rank_stability.csv"), index=False)

        # A compact textual report.
        report_lines = []
        report_lines.append("Offline window-noise diagnostic")
        report_lines.append("================================")
        report_lines.append(f"Output directory: {root}")
        report_lines.append(f"Methods: {', '.join(selected_keys)}")
        report_lines.append(f"Repeats: {int(max(1, repeat_runs))}, BO windows per repeat: {int(CFG.BO_ITERATIONS)}")
        if not method_df.empty:
            best_row = method_df.iloc[0]
            worst_row = method_df.iloc[-1]
            spread = float(worst_row["Mean_Window_Cost"] - best_row["Mean_Window_Cost"])
            typical_std = float(method_df["Std_Window_Cost"].median())
            report_lines.append("")
            report_lines.append(f"Best mean fixed policy: {best_row['Method_Key']} cost={best_row['Mean_Window_Cost']:.4f}")
            report_lines.append(f"Worst mean fixed policy: {worst_row['Method_Key']} cost={worst_row['Mean_Window_Cost']:.4f}")
            report_lines.append(f"Fixed-policy mean spread: {spread:.4f}")
            report_lines.append(f"Median per-window std within policy: {typical_std:.4f}")
            report_lines.append(f"Spread / median per-window std: {spread / max(1e-12, typical_std):.4f}")
            if spread / max(1e-12, typical_std) < 1.0:
                report_lines.append("Interpretation: policy gaps are smaller than a typical single-window fluctuation; one-window BO observations are likely noisy.")
            elif spread / max(1e-12, typical_std) < 2.0:
                report_lines.append("Interpretation: policy gaps are visible but close to the single-window noise scale; confidence/recent filtering may help.")
            else:
                report_lines.append("Interpretation: policy gaps are larger than typical single-window noise; BO should be able to learn if feedback attribution is otherwise clean.")
        report_lines.append("")
        report_lines.append("Key files:")
        report_lines.append("- offline_window_noise_samples.csv: one row per method/seed/window")
        report_lines.append("- offline_window_noise_method_summary.csv: per-method mean/std/CV")
        report_lines.append("- offline_window_noise_pairwise.csv: paired same-seed/same-iteration deltas")
        report_lines.append("- offline_window_noise_rank_stability.csv: rank stability across seeds")
        with open(os.path.join(root, "offline_window_noise_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print("\n".join(report_lines))
        return {
            "samples": sample_df,
            "method_summary": method_df,
            "pairwise": pair_df,
            "rank": rank_df,
            "top1": top1_df,
        }
    finally:
        SCENARIO_SAVE_DIR = old_save_dir


# ===============================================================
# v6.1: short filename export helper
# ---------------------------------------------------------------
# 不改原始结果文件；在 output_dir/_short_export 里复制一份英文短名，并写 filename_mapping.csv。
# 用法：命令行加 --export-short-names。
# ===============================================================

def _short_clean_ascii_name(s):
    repl = {
        "核心指标统计": "key_metrics", "实验汇总": "summary", "轮次汇总": "round",
        "情景调试": "ctx", "节点分配调试": "alloc_node",
        "任务类型节点分配汇总": "alloc_type_sum", "任务类型节点分配调试": "alloc_type_dbg",
        "任务类型节点堆叠图": "alloc_type_stack", "任务类型云边占比": "alloc_cloud_ratio",
        "方法云占比对比": "cloud_ratio_cmp", "批次反馈学习曲线": "learn_curve",
        "每轮平均能耗时延评分": "round_metrics", "控制参数轨迹": "theta_traj",
        "全部方法任务类型分配汇总": "alloc_type_all",
    }
    for k, v in repl.items():
        s = str(s).replace(k, v)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "file"


def _short_scene_part(part):
    part = str(part)
    if part == "window_original":
        return "win"
    if part == "paired_fixed_mid_delta":
        return "paired"
    m = re.match(r"lam(\d+)p(\d+)_RT(\d+)_Batch(\d+)_AI(\d+)", part)
    if m:
        return f"l{m.group(1)}{m.group(2)}_r{m.group(3)}_b{m.group(4)}_a{m.group(5)}"
    m = re.match(r"RT(\d+)_Batch(\d+)_AI(\d+)", part)
    if m:
        return f"r{m.group(1)}_b{m.group(2)}_a{m.group(3)}"
    return _short_clean_ascii_name(part)[:80]


def _short_file_name_for_export(filename):
    p = os.path.basename(str(filename))
    stem, ext = os.path.splitext(p)
    method_map = {
        "reduced6_fixed_mid": "fm", "reduced6_fixed_tuned": "ft",
        "reduced6_fixed_queue_high": "fq", "reduced6_fixed_risk_high": "fr", "reduced6_fixed_edge_safe": "fe",
        "reduced6_bo_greedy": "bo", "reduced6_bo_ei": "boei",
        "reduced6_cbo_lite_full": "cbo_full", "reduced6_cbo_lite_pressure_only": "cbo_p",
        "reduced6_cbo_lite_load_only": "cbo_l", "reduced6_cbo_lite_util_only": "cbo_u",
        "reduced6_cbo_lite_no_cloud": "cbo_nc", "reduced6_cbo_lite_no_arrival": "cbo_na",
        "reduced6_cbo_lite_taskmix": "cbo_tm", "reduced6_cbo_lite_recent_mix": "cbo_rm",
        "reduced6_cbo_lite_prev_counts": "cbo_cnt", "reduced6_cbo_lite_pressure_taskmix": "cbo_pt",
        "reduced6_cbo_lite_pressure_recent_mix": "cbo_prm", "reduced6_cbo_lite_pressure_counts": "cbo_pc",
        "reduced6_cbo_lite_pressure_taskmix_counts": "cbo_ptc", "reduced6_cbo_lite_full_taskmix": "cbo_ftm",
        "reduced6_cbo_lite_full_taskmix_counts": "cbo_ftc",
        "direct_round_robin": "rr_direct",
        "direct_greedy_cost": "greedy_direct",
        "direct_least_load": "leastload_direct",
        "direct_queue_aware_greedy": "qaware_direct",
    }
    suffix_map = {
        "round_summary_轮次汇总": "round", "context_debug_情景调试": "ctx",
        "alloc_debug_节点分配调试": "alloc_node", "alloc_by_type_summary_任务类型节点分配汇总": "alloc_type_sum",
        "alloc_by_type_debug_任务类型节点分配调试": "alloc_type_dbg",
        "alloc_by_type_stacked_任务类型节点堆叠图": "alloc_type_stack",
        "alloc_by_type_cloud_ratio_任务类型云边占比": "alloc_cloud_ratio",
    }
    exact_map = {
        "key_metric_summary_核心指标统计": "key_metrics", "scenario_experiment_summary_实验汇总": "scene_summary",
        "scenario_phase_summary": "phase_summary", "refactor_run_config": "run_config",
        "alloc_by_type_all_methods_summary_全部方法任务类型分配汇总": "alloc_type_all",
        "alloc_by_type_method_cloud_ratio_compare_方法云占比对比": "cloud_ratio_cmp",
        "scenario_theta_trajectory_控制参数轨迹": "theta_traj", "scenario_convergence": "scene_conv",
        "scenario_best_so_far": "scene_best", "scenario_alloc_heatmaps": "scene_alloc_heatmap",
        "scenario_task_delay_bars": "scene_task_delay",
        "scenario_round_mean_energy_delay_score_每轮平均能耗时延评分": "scene_round_metrics",
        "scenario_cohort_learning_curves_批次反馈学习曲线": "scene_learn",
    }
    if stem in exact_map:
        return exact_map[stem] + ext
    for long_m, short_m in method_map.items():
        if stem.startswith(long_m + "_"):
            rest = stem[len(long_m) + 1:]
            for long_suf, short_suf in suffix_map.items():
                if rest == long_suf:
                    return f"{short_m}_{short_suf}{ext}"
            return f"{short_m}_{_short_clean_ascii_name(rest)[:60]}{ext}"
    return _short_clean_ascii_name(stem)[:90] + ext


def export_short_named_results(output_dir, export_dir=None, make_tar=True):
    """复制 output_dir 中的常用结果文件到短文件名目录，避免 Windows/网盘下载丢长路径。"""
    import shutil
    import csv
    import tarfile
    root = os.path.abspath(output_dir or SCENARIO_SAVE_DIR)
    if not os.path.exists(root):
        print(f"[WARN] short export skipped; missing output_dir={root}")
        return None
    export_dir = os.path.abspath(export_dir or os.path.join(root, "_short_export"))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    keep_ext = {".csv", ".png", ".json", ".log", ".txt"}
    mapping = []
    for base, _, files in os.walk(root):
        # 不把导出目录再递归导出一遍
        if os.path.abspath(base).startswith(os.path.abspath(export_dir)):
            continue
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in keep_ext:
                continue
            src = os.path.join(base, name)
            rel = os.path.relpath(src, root)
            parts = rel.split(os.sep)
            short_parts = [_short_scene_part(x) for x in parts[:-1]]
            short_name = _short_file_name_for_export(parts[-1])
            dst_dir = os.path.join(export_dir, *short_parts)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, short_name)
            stem, dst_ext = os.path.splitext(dst)
            k = 2
            while os.path.exists(dst):
                dst = f"{stem}_{k}{dst_ext}"
                k += 1
            shutil.copy2(src, dst)
            mapping.append({"original_relpath": rel, "short_relpath": os.path.relpath(dst, export_dir)})
    mapping_path = os.path.join(export_dir, "filename_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_relpath", "short_relpath"])
        writer.writeheader()
        writer.writerows(mapping)
    tar_path = None
    if make_tar:
        tar_path = export_dir.rstrip(os.sep) + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_dir, arcname=os.path.basename(export_dir))
    print(f"[OK] short export files={len(mapping)} dir={export_dir}" + (f" tar={tar_path}" if tar_path else ""))
    return export_dir
