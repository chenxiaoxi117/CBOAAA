#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 3622-3983.
# Legacy output saving, smoothing, plotting, and batch helpers.

def save_detailed_data(baseline_log, fed_log, rr_log=None):
    logs = [("Baseline", baseline_log), ("Federated", fed_log)]
    if rr_log: logs.append(("RoundRobin", rr_log))
    for label, log in logs:
        alphas = log.get("alpha", [])
        feature_cols = {}
        for idx, name in enumerate(CFG.FEATURE_NAMES):
            feature_cols[name] = [a[idx] if isinstance(a, (list, tuple)) and len(a) > idx else None for a in alphas]
        df_dict = {
            "Time": log["time"],
            "Reward": log.get("reward", []),
            "Total_Energy": log["total_energy"],
            "Cumulative_Energy": log["cumulative_energy"],
            "Avg_Latency": log["avg_delay"],
            "Avg_Latency_RT": log.get("avg_delay_rt", []),
            "Avg_Latency_Batch": log.get("avg_delay_batch", []),
            "Avg_Latency_AI": log.get("avg_delay_ai", []),
            "Violation_Rate": log.get("vio_rate", []),
            "SLA_Success_Rate": log["sla_success_rate"],
            "Avg_Energy_Per_Task": log.get("avg_energy", []),
            "Avg_Earliness": log.get("avg_earliness", []),
            "Avg_Lateness": log.get("avg_lateness", []),
        }
        df_dict.update(feature_cols)
        df = pd.DataFrame(df_dict)
        csv_path = os.path.join(SAVE_DIR, f"detailed_data_{label}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed data for {label} to {csv_path}")

def smooth(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

def ema_smooth(scalars, weight=0.85):
    if not scalars: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_comparison(baseline_log, fed_log, rr_log=None):
    min_len = min(len(baseline_log['reward']), len(fed_log['reward']))
    if rr_log: min_len = min(min_len, len(rr_log['reward']))
    iters = np.arange(min_len)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax1 = axes[0]
    ax1.grid(True, linestyle='-', alpha=0.7)
    if rr_log:
        lat_rr = ema_smooth(rr_log['avg_delay'][:min_len], weight=0.85)
        ax1.plot(iters, lat_rr, label="Round Robin", color='orange', linestyle='--')
    lat_base = ema_smooth(baseline_log['avg_delay'][:min_len], weight=0.85)
    lat_fed = ema_smooth(fed_log['avg_delay'][:min_len], weight=0.85)
    ax1.plot(iters, lat_base, label="BO-Local (Baseline)", color='blue', linestyle='-.')
    ax1.plot(iters, lat_fed, label="Proposed Algorithm (BO-Fed)", color='green', linewidth=2.5, linestyle='-')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Average Latency (s)")
    ax1.set_title("Latency comparison")
    ax1.legend(loc='upper right')
    ax2 = axes[1]
    ax2.grid(True, linestyle='-', alpha=0.7)
    if rr_log:
        ce_rr = np.array(rr_log['cumulative_energy'][:min_len])
        ce_base = np.array(baseline_log['cumulative_energy'][:min_len])
        ce_fed = np.array(fed_log['cumulative_energy'][:min_len])
        savings_base = ce_rr - ce_base
        savings_fed = ce_rr - ce_fed
        ax2.plot(iters, savings_base, label="BO-Local Savings", color='blue', linestyle='-.')
        ax2.plot(iters, savings_fed, label="Proposed Algorithm Savings", color='purple', linewidth=2.0)
        ax2.set_ylabel("Cumulative Energy Savings (J)")
    else:
        ax2.plot(iters, baseline_log['cumulative_energy'][:min_len], label="BO-Local", color='blue', linestyle='-.')
        ax2.plot(iters, fed_log['cumulative_energy'][:min_len], label="Proposed Algorithm", color='purple', linewidth=2.0)
        ax2.set_ylabel("Cumulative Energy (J)")
    ax2.set_xlabel("Iterations")
    ax2.set_title("Cumulative Energy Savings Comparison")
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "paper_style_convergence.png"), dpi=300)
def plot_best_so_far(baseline_log, fed_log, rr_log=None):
    def bsf(seq):
        if not seq: return []
        out = []
        m = -float("inf")
        for x in seq:
            m = max(m, x)
            out.append(m)
        return out
    min_len = min(len(baseline_log.get("reward", [])), len(fed_log.get("reward", [])))
    if rr_log: min_len = min(min_len, len(rr_log.get("reward", [])))
    iters = np.arange(min_len)
    r_base = bsf(baseline_log["reward"][:min_len])
    r_fed = bsf(fed_log["reward"][:min_len])
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(iters, r_base, label="BO-Local Best-So-Far", color='blue', linestyle='-.')
    ax.plot(iters, r_fed, label="BO-Fed Best-So-Far", color='green', linewidth=2.0)
    if rr_log:
        r_rr = bsf(rr_log["reward"][:min_len])
        ax.plot(iters, r_rr, label="Round-Robin Best-So-Far", color='orange', linestyle='--')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("Best-So-Far Reward")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "trend_bsf.png"), dpi=300)
def plot_convergence_metrics(baseline_log, fed_log, rr_log=None):
    min_len = min(len(baseline_log['time']), len(fed_log['time']))
    if rr_log: min_len = min(min_len, len(rr_log['time']))
    iters = np.arange(min_len)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    s_base = ema_smooth(baseline_log['sla_success_rate'][:min_len], weight=0.85)
    s_fed = ema_smooth(fed_log['sla_success_rate'][:min_len], weight=0.85)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].plot(iters[:len(s_base)], s_base, label="BO-Local", color='blue', linestyle='-.')
    axes[0].plot(iters[:len(s_fed)], s_fed, label="BO-Fed", color='green')
    if rr_log:
        s_rr = ema_smooth(rr_log['sla_success_rate'][:min_len], weight=0.85)
        axes[0].plot(iters[:len(s_rr)], s_rr, label="Round-Robin", color='orange', linestyle='--')
    axes[0].set_title("SLA Success Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right')
    e_base = ema_smooth(baseline_log.get('avg_energy', [])[:min_len], weight=0.85)
    e_fed = ema_smooth(fed_log.get('avg_energy', [])[:min_len], weight=0.85)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    if e_base:
        axes[1].plot(iters[:len(e_base)], e_base, label="BO-Local Avg Energy", color='blue', linestyle='-.')
    if e_fed:
        axes[1].plot(iters[:len(e_fed)], e_fed, label="BO-Fed Avg Energy", color='purple')
    if rr_log and rr_log.get('avg_energy'):
        e_rr = ema_smooth(rr_log['avg_energy'][:min_len], weight=0.85)
        axes[1].plot(iters[:len(e_rr)], e_rr, label="Round-Robin Avg Energy", color='orange', linestyle='--')
    axes[1].set_title("Average Energy per Task (J)")
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "convergence_metrics.png"), dpi=300)

def plot_acq_process(acq_history):
    if not acq_history: return
    steps_to_show = acq_history[-4:]
    fig, axes = plt.subplots(len(steps_to_show), 1, figsize=(10, 4 * len(steps_to_show)))
    if len(steps_to_show) == 1: axes = [axes]
    for i, step_data in enumerate(steps_to_show):
        ax = axes[i]
        cands = np.array(step_data["candidates"])
        acq_vals = np.array(step_data["acq_values"])
        best = step_data["best_selected"]
        if len(cands) > 0 and len(cands) == len(acq_vals):
            sc = ax.scatter(cands[:, 0], cands[:, 1], c=acq_vals, cmap='viridis', s=50)
            plt.colorbar(sc, ax=ax)
        if best is not None and len(best) >= 2:
            ax.scatter(best[0], best[1], color='red', s=200, marker='*')
        ax.set_title(f"BO Step {step_data['step']}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "acq_process.png"))

# ==========================================
# 7. 主流程 (Main)
# ==========================================
# 你可以把主流程理解为两层：
# - 内层：run_continuous() 在一个时间窗口内做事件驱动仿真。
# - 外层：run_baseline_batch()/run_federated_batch() 每轮先 ask theta，再跑一个窗口，再 tell 反馈。
def aggregate_logs(logs):
    if not logs:
        return {}
    min_len = min(len(l.get("reward", [])) for l in logs)
    result = {}
    all_keys = set()
    for l in logs:
        all_keys.update(l.keys())
    for key in all_keys:
        if key == "state":
            continue
        val0 = next((l.get(key) for l in logs if isinstance(l.get(key), list)), None)
        if not isinstance(val0, list):
            continue
        if min_len == 0:
            result[key] = []
            continue
        # build matrix padded to min_len so shorter lists won't raise IndexError
        matrix = []
        for l in logs:
            lst = l.get(key, []) or []
            row = []
            for j in range(min_len):
                if j < len(lst):
                    row.append(lst[j])
                else:
                    row.append(None)
            matrix.append(row)

        # find a non-None sample to determine element shape
        first_non_none = None
        for row in matrix:
            for v in row:
                if v is not None:
                    first_non_none = v
                    break
            if first_non_none is not None:
                break
        if first_non_none is None:
            result[key] = [None] * min_len
            continue
        if key in {
            "deploy_policy", "deploy_source", "used_theta_source",
            "tr_update_mode", "tr_update_signal", "cbo_tr_update_reason",
            "selected_reason", "selected_candidate_source", "selected_source",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_reason",
            "anchor_fallback_reason", "runtime_anchor_override",
            "state_beta_boost_reason", "service_guard_mode", "service_guard_reason",
            "macro_context_key", "energy_metric_source", "cbo_reference_mode", "cbo_reference_status", "cbo_objective_mode", "bo_training_cost_source",
            "scheduler_tradeoff_mode", "scheduler_score_norm_mode",
            "control_vector_meaning", "alpha_direct_feature_names", "theta_full_feature_names",
        }:
            result[key] = [next((row[j] for row in matrix if row[j] is not None), None) for j in range(min_len)]
            continue

        try:
            if isinstance(first_non_none, (list, tuple, np.ndarray)):
                inner_len = len(first_non_none)
                arr = np.full((len(matrix), min_len, inner_len), np.nan, dtype=float)
                for i in range(len(matrix)):
                    for j in range(min_len):
                        v = matrix[i][j]
                        if v is None:
                            continue
                        arr[i, j, :] = np.array(v, dtype=float)
                mean_vals = np.nanmean(arr, axis=0)
                result[key] = mean_vals.tolist()
            else:
                arr = np.full((len(matrix), min_len), np.nan, dtype=float)
                for i in range(len(matrix)):
                    for j in range(min_len):
                        v = matrix[i][j]
                        if v is None:
                            arr[i, j] = np.nan
                        else:
                            try:
                                arr[i, j] = float(v)
                            except Exception:
                                arr[i, j] = np.nan
                mean_vals = np.nanmean(arr, axis=0)
                result[key] = mean_vals.tolist()
        except Exception:
            result[key] = [None] * min_len
    return result

def summarize_metrics(logs, label):
    valid_logs = [
        l for l in logs
        if l.get("reward") and l.get("avg_delay") and l.get("sla_success_rate") and l.get("cumulative_energy")
    ]
    if not valid_logs:
        print(f"[{label}] Metrics Summary: no valid logs")
        return

    rewards = [float(np.mean(l["reward"])) for l in valid_logs]
    delays = [float(np.mean(l["avg_delay"])) for l in valid_logs]
    sla_success = [float(np.mean(l["sla_success_rate"])) for l in valid_logs]
    total_energy = [float(l["cumulative_energy"][-1]) for l in valid_logs]
    print(f"[{label}] Metrics Summary: Avg Reward: {np.mean(rewards):.3f}, Avg Delay: {np.mean(delays):.3f}s, SLA Success: {np.mean(sla_success):.3f}, Total Energy: {np.mean(total_energy):.2f} J")

def _eval_theta_once(theta, batch_size, seed):
    fac = ConnectedFactory(fid=0, name="Eval", seed=seed, node_config=CFG.NODES_CFG)
    fac.reset(use_batch=False)
    horizon = max(CFG.BO_INTERVAL, float(batch_size) / max(1e-9, CFG.BATCH_POISSON_LAMBDA))
    fac.run_continuous(theta, window_end=horizon)
    log = fac.perf_log
    idx = -1
    res = {}
    res.update(theta_to_named_dict(theta))
    res.update({
        "Reward": log["reward"][idx],
        "Total_Energy": log["total_energy"][idx],
        "Avg_Latency": log["avg_delay"][idx],
        "Avg_Latency_RT": log.get("avg_delay_rt", [None])[idx],
        "Avg_Latency_Batch": log.get("avg_delay_batch", [None])[idx],
        "Avg_Latency_AI": log.get("avg_delay_ai", [None])[idx],
        "SLA_Success_Rate": log["sla_success_rate"][idx],
        "Violation_Rate": log.get("vio_rate", [None])[idx],
        "Avg_Energy_Per_Task": log.get("avg_energy", [None])[idx],
    })
    return res

def run_param_analysis(samples=40, local_delta=0.08):
    bounds = get_control_bounds(CFG.DIM_THETA)
    low = torch.tensor(bounds[0], dtype=torch.double)
    high = torch.tensor(bounds[1], dtype=torch.double)
    base = default_control_vector(fill=1.5)
    sample_gen = torch.Generator().manual_seed(resolve_base_seed(CFG.BASE_SEED, stream=700))
    thetas = []
    thetas.append(base)
    dim = len(low)
    for _ in range(samples):
        r = torch.rand(dim, dtype=torch.double, generator=sample_gen)
        t = (low + r * (high - low)).tolist()
        thetas.append(t)
    for d in range(dim):
        td = base.copy()
        td[d] = float(max(low[d], base[d] * (1.0 - local_delta)))
        thetas.append(td)
        td2 = base.copy()
        td2[d] = float(min(high[d], base[d] * (1.0 + local_delta)))
        thetas.append(td2)
    rows = []
    for t in thetas:
        r = _eval_theta_once(t, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "param_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    cols_theta = list(CFG.FEATURE_NAMES)
    cols_metric = ["Reward", "Total_Energy", "Avg_Latency", "SLA_Success_Rate"]
    corr = {}
    for m in cols_metric:
        corr[m] = []
        for c in cols_theta:
            s = df[[c, m]].corr(method="spearman").iloc[0, 1]
            corr[m].append(s)
    corr_df = pd.DataFrame(corr, index=cols_theta)
    out_corr = os.path.join(SAVE_DIR, "param_sensitivity_corr.csv")
    corr_df.to_csv(out_corr)

def run_extreme_param_test():
    prev_fixed = CFG.USE_FIXED_RNG
    CFG.USE_FIXED_RNG = True
    extremes = [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 0.8, 0.8, 0.8, 2.5, 2.0],
        [0.8, 3.0, 0.8, 2.5, 0.8, 2.0],
        [0.8, 0.8, 3.0, 2.0, 2.0, 0.8],
        [2.5, 1.5, 1.2, 1.0, 2.5, 2.0],
    ]
    rows = []
    for t in extremes:
        r = _eval_theta_once(t, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "extreme_param_test.csv")
    df.to_csv(out_csv, index=False)
    CFG.USE_FIXED_RNG = prev_fixed

def run_param_scan(dim_name="RT_E", points=8):
    names = list(CFG.FEATURE_NAMES)
    if dim_name not in names: dim_idx = 0
    else: dim_idx = names.index(dim_name)
    bounds = get_control_bounds(CFG.DIM_THETA)
    low = torch.tensor(bounds[0], dtype=torch.double)
    high = torch.tensor(bounds[1], dtype=torch.double)
    base = default_control_vector(fill=1.5)
    xs = np.linspace(float(low[dim_idx]), float(high[dim_idx]), points)
    rows = []
    for v in xs:
        th = base.copy()
        th[dim_idx] = float(v)
        r = _eval_theta_once(th, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, f"scan_{dim_name}.csv")
    df.to_csv(out_csv, index=False)
