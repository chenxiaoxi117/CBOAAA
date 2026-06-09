from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 配置区
# ============================================================

ROOT = Path(r"D:\CBOv2\results\新原始107静态数据\results\full_ratio_sweep_compare_107_seed43")
OUT = Path(r"C:\Users\POPchen\Desktop\duiibi_new_gap_107")
OUT.mkdir(parents=True, exist_ok=True)

ROLLING = 50

BASELINE_KEY = "reduced6_fixed_mid"
BASELINE_LABEL = "Fixed-mid"

# 图里默认不放 RoundRobin 和 edge-safe，避免图被拉爆/过乱
EXCLUDE_METHOD_KEYS = {
    "direct_round_robin",
    "reduced6_fixed_edge_safe",
}

LABEL_MAP = {
    "reduced6_fixed_mid": "Fixed-mid",
    "reduced6_fixed_tuned": "Fixed-tuned",
    "reduced6_fixed_risk_high": "Fixed-risk-high",
    "reduced6_fixed_queue_high": "Fixed-queue-high",
    "reduced6_fixed_edge_safe": "Fixed-edge-safe",

    "direct_least_load": "LeastLoad-direct",
    "direct_greedy_cost": "Greedy-direct",
    "direct_queue_aware_greedy": "QueueAwareGreedy-direct",
    "direct_round_robin": "RoundRobin-direct",

    "reduced6_bo_greedy": "BO-greedy",
    "reduced6_cbo_lite_pressure_only": "CBO-pressure",
    "reduced6_cbo_lite_pressure_prev_unfinished": "CBO-prev-unfinished",
}

# 图例顺序
METHOD_ORDER = [
    "reduced6_fixed_mid",
    "reduced6_fixed_tuned",
    "reduced6_fixed_risk_high",
    "reduced6_fixed_queue_high",

    "direct_least_load",
    "direct_greedy_cost",
    "direct_queue_aware_greedy",

    "reduced6_bo_greedy",
    "reduced6_cbo_lite_pressure_only",
    "reduced6_cbo_lite_pressure_prev_unfinished",
]

# 分场景图只画这些，避免 107 张图太乱
SCENE_PLOT_METHOD_KEYS = [
    "reduced6_bo_greedy",
    "reduced6_cbo_lite_pressure_only",
    "reduced6_cbo_lite_pressure_prev_unfinished",
]

METHOD_COLS = [
    "Group_Key_方法键",
    "method",
    "Method",
    "method_key",
]

ITER_COLS = [
    "Iteration_轮次",
    "Iteration",
    "iter",
    "round",
]

METRIC_CANDIDATES = {
    "cost": [
        "Eval_Cost_最终评估Cost",
        "Eval_Cost",
        "Cost",
    ],
    "delay": [
        "Avg_Delay_平均时延",
        "Average_Delay_平均时延",
        "Mean_Delay_平均时延",
        "Avg_Delay",
        "Average_Delay",
        "Mean_Delay",
        "Delay",
        "Latency",
    ],
    "energy": [
        "Avg_Energy_平均能耗",
        "Average_Energy_平均能耗",
        "Mean_Energy_平均能耗",
        "Avg_Energy",
        "Average_Energy",
        "Mean_Energy",
        "Energy",
        "Total_Energy_总能耗",
        "Total_Energy",
    ],
}

# ============================================================
# 工具函数
# ============================================================

def label_of(key):
    return LABEL_MAP.get(key, key)

def pick_col(df, candidates, loose_keys=None):
    for c in candidates:
        if c in df.columns:
            return c

    if loose_keys:
        for c in df.columns:
            low = str(c).lower()
            if any(str(k).lower() in low for k in loose_keys):
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().sum() > 0:
                    return c
    return None

def parse_scene(path: Path):
    s = str(path)
    m = re.search(r"lam(?P<lam>\d+p\d+)_RT(?P<rt>\d+)_(?:B|Batch)(?P<b>\d+)_AI(?P<ai>\d+)", s)
    if not m:
        return None

    lam = float(m.group("lam").replace("p", "."))
    rt = int(m.group("rt"))
    batch = int(m.group("b"))
    ai = int(m.group("ai"))

    vals = {
        "RT-heavy": rt,
        "Batch-heavy": batch,
        "AI-heavy": ai,
    }
    best = max(vals, key=vals.get)
    task_group = best if vals[best] >= 50 else "Mixed"

    lam_tag = str(lam).replace(".", "p")
    scene_key = f"lam{lam_tag}_RT{rt}_B{batch}_AI{ai}"

    return {
        "scene_key": scene_key,
        "lambda": lam,
        "rt": rt,
        "batch": batch,
        "ai": ai,
        "task_group": task_group,
    }

def find_files(root):
    files = []
    for pat in ["*round_summary*csv", "*轮次汇总*csv"]:
        for p in root.rglob(pat):
            s = str(p).lower()
            if "_short_export" in s:
                continue
            if "analysis" in s:
                continue
            files.append(p)

    # 去重
    seen = set()
    out = []
    for p in files:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return sorted(out)

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

def method_sort_key(label):
    try:
        key = {label_of(k): i for i, k in enumerate(METHOD_ORDER)}
        return key.get(label, 999)
    except Exception:
        return 999

# ============================================================
# 读取数据
# ============================================================

def load_all_records():
    files = find_files(ROOT)

    print("[ROOT]", ROOT)
    print("[FILES]", len(files))

    rows = []
    inventory = []

    for p in files:
        scene = parse_scene(p)
        if scene is None:
            continue

        try:
            df = read_csv(p)
        except Exception as e:
            print("[WARN] read failed:", p, e)
            continue

        method_col = pick_col(df, METHOD_COLS)
        if method_col is None:
            continue

        iter_col = pick_col(df, ITER_COLS)
        if iter_col is None:
            df["_iter_tmp"] = np.arange(1, len(df) + 1)
            iter_col = "_iter_tmp"

        metric_cols = {}
        for metric, cands in METRIC_CANDIDATES.items():
            loose = None
            if metric == "cost":
                loose = ["cost"]
            elif metric == "delay":
                loose = ["delay", "latency", "时延", "延迟"]
            elif metric == "energy":
                loose = ["energy", "能耗"]

            col = pick_col(df, cands, loose_keys=loose)
            if col is not None:
                metric_cols[metric] = col

        if "cost" not in metric_cols:
            continue

        method_keys = sorted(df[method_col].dropna().astype(str).unique())

        inventory.append({
            "file": str(p),
            "scene_key": scene["scene_key"],
            "lambda": scene["lambda"],
            "rt": scene["rt"],
            "batch": scene["batch"],
            "ai": scene["ai"],
            "task_group": scene["task_group"],
            "rows": len(df),
            "method_col": method_col,
            "iter_col": iter_col,
            "method_count": len(method_keys),
            "methods": "|".join(method_keys),
            "cost_col": metric_cols.get("cost", ""),
            "delay_col": metric_cols.get("delay", ""),
            "energy_col": metric_cols.get("energy", ""),
        })

        for key in method_keys:
            if key in EXCLUDE_METHOD_KEYS:
                continue

            sub = df[df[method_col].astype(str) == key].copy()
            if sub.empty:
                continue

            rec = pd.DataFrame()
            rec["iter"] = pd.to_numeric(sub[iter_col], errors="coerce")
            rec["method_key"] = key
            rec["method"] = label_of(key)

            rec["scene_key"] = scene["scene_key"]
            rec["lambda"] = scene["lambda"]
            rec["rt"] = scene["rt"]
            rec["batch"] = scene["batch"]
            rec["ai"] = scene["ai"]
            rec["task_group"] = scene["task_group"]

            for metric, col in metric_cols.items():
                rec[metric] = pd.to_numeric(sub[col], errors="coerce")

            rec = rec.dropna(subset=["iter", "cost"])
            if rec.empty:
                continue

            rec = rec.sort_values("iter").reset_index(drop=True)

            for metric in ["cost", "delay", "energy"]:
                if metric in rec.columns:
                    rec[f"roll50_{metric}"] = rec[metric].rolling(ROLLING, min_periods=ROLLING).mean()

            rows.append(rec)

    if not rows:
        raise RuntimeError("No records loaded. 请检查 ROOT 或 CSV 文件名。")

    raw = pd.concat(rows, ignore_index=True)
    inv = pd.DataFrame(inventory)

    return raw, inv

# ============================================================
# 统计
# ============================================================

def add_gap_vs_baseline(raw, metric):
    roll_col = f"roll50_{metric}"
    if roll_col not in raw.columns:
        return pd.DataFrame()

    base = raw[raw["method_key"] == BASELINE_KEY][["scene_key", "iter", roll_col]].copy()
    base = base.rename(columns={roll_col: "baseline_roll50"})

    merged = raw.merge(base, on=["scene_key", "iter"], how="left")
    merged = merged.dropna(subset=[roll_col, "baseline_roll50"]).copy()

    merged["gap_pct"] = 100.0 * (merged[roll_col] - merged["baseline_roll50"]) / merged["baseline_roll50"]
    merged["metric"] = metric

    return merged

def scene_method_summary(raw):
    rows = []

    for (scene, method_key), g in raw.groupby(["scene_key", "method_key"]):
        g = g.sort_values("iter")
        first = g.iloc[0]

        rec = {
            "scene_key": scene,
            "lambda": first["lambda"],
            "rt": first["rt"],
            "batch": first["batch"],
            "ai": first["ai"],
            "task_group": first["task_group"],
            "method_key": method_key,
            "method": first["method"],
            "rows": len(g),
        }

        for metric in ["cost", "delay", "energy"]:
            if metric not in g.columns:
                continue

            x = pd.to_numeric(g[metric], errors="coerce")
            rec[f"mean_{metric}"] = x.mean()
            rec[f"first100_{metric}"] = x.head(100).mean()
            rec[f"tail100_{metric}"] = x.tail(100).mean()
            rec[f"last50_{metric}"] = x.tail(50).mean()

            r = pd.to_numeric(g.get(f"roll50_{metric}", pd.Series(dtype=float)), errors="coerce")
            rec[f"final_roll50_{metric}"] = r.dropna().iloc[-1] if len(r.dropna()) else np.nan
            rec[f"min_roll50_{metric}"] = r.min()

        rows.append(rec)

    sm = pd.DataFrame(rows)

    # 给 summary 加 Fixed-mid gap
    for metric in ["cost", "delay", "energy"]:
        val = f"final_roll50_{metric}"
        if val not in sm.columns:
            continue

        base = sm[sm["method_key"] == BASELINE_KEY][["scene_key", val]].rename(columns={val: f"baseline_{val}"})
        sm = sm.merge(base, on="scene_key", how="left")
        sm[f"gap_vs_Fixed-mid_{metric}_pct"] = 100.0 * (sm[val] - sm[f"baseline_{val}"]) / sm[f"baseline_{val}"]

    return sm

def aggregate_summary(sm, group_cols):
    metric_cols = [
        c for c in sm.columns
        if c.startswith(("mean_", "first100_", "tail100_", "last50_", "final_roll50_", "min_roll50_", "gap_vs_"))
    ]

    out = (
        sm.groupby(group_cols + ["method_key", "method"])
        .agg(
            scene_count=("scene_key", "nunique"),
            **{c: (c, "mean") for c in metric_cols},
        )
        .reset_index()
    )

    if "final_roll50_cost" in out.columns:
        out = out.sort_values(group_cols + ["final_roll50_cost"])

    return out

# ============================================================
# 绘图
# ============================================================

def plot_gap_curve(gap_df, metric, group_col=None, group_value=None, outpath=None, title=None):
    d = gap_df[gap_df["metric"] == metric].copy()

    if group_col is not None:
        d = d[d[group_col] == group_value].copy()

    if d.empty:
        return

    # 先按 scene 内 gap，再聚合成平均 gap
    curve = (
        d.groupby(["iter", "method_key", "method"])["gap_pct"]
        .mean()
        .reset_index()
    )

    curve["sort_key"] = curve["method"].map(method_sort_key)
    curve = curve.sort_values(["sort_key", "method"])

    plt.figure(figsize=(11, 6))

    for method, g in curve.groupby("method", sort=False):
        gg = g.sort_values("iter")
        # baseline 自己是 0 线，不画也行；这里跳过，避免图例占位
        if method == BASELINE_LABEL:
            continue
        plt.plot(gg["iter"], gg["gap_pct"], linewidth=1.8, label=method)

    plt.axhline(0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Iteration")
    plt.ylabel(f"Rolling{ROLLING} gap vs {BASELINE_LABEL} (%)")
    plt.title(title or f"{metric.upper()} gap vs {BASELINE_LABEL}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_value_curve(raw, metric, group_col=None, group_value=None, outpath=None, title=None):
    roll_col = f"roll50_{metric}"
    if roll_col not in raw.columns:
        return

    d = raw.dropna(subset=[roll_col]).copy()
    if group_col is not None:
        d = d[d[group_col] == group_value].copy()

    if d.empty:
        return

    curve = (
        d.groupby(["iter", "method_key", "method"])[roll_col]
        .mean()
        .reset_index()
        .rename(columns={roll_col: "value"})
    )

    curve["sort_key"] = curve["method"].map(method_sort_key)
    curve = curve.sort_values(["sort_key", "method"])

    plt.figure(figsize=(11, 6))

    for method, g in curve.groupby("method", sort=False):
        gg = g.sort_values("iter")
        plt.plot(gg["iter"], gg["value"], linewidth=1.8, label=method)

    plt.xlabel("Iteration")
    plt.ylabel(f"Rolling{ROLLING} {metric}")
    plt.title(title or f"{metric.upper()} rolling{ROLLING}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def plot_scene_cbo_gap(gap_df, metric):
    d = gap_df[gap_df["metric"] == metric].copy()
    d = d[d["method_key"].isin(SCENE_PLOT_METHOD_KEYS)].copy()

    if d.empty:
        return

    scene_dir = OUT / "scene_gap_plots" / metric
    scene_dir.mkdir(parents=True, exist_ok=True)

    for scene, gscene in d.groupby("scene_key"):
        plt.figure(figsize=(10, 5))

        for method, g in gscene.groupby("method"):
            gg = g.sort_values("iter")
            plt.plot(gg["iter"], gg["gap_pct"], linewidth=1.8, label=method)

        plt.axhline(0, linestyle="--", linewidth=1, color="black")
        plt.xlabel("Iteration")
        plt.ylabel(f"Rolling{ROLLING} gap vs {BASELINE_LABEL} (%)")
        plt.title(f"{scene} | {metric.upper()} gap vs {BASELINE_LABEL}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(scene_dir / f"{scene}_{metric}_cbo_gap.png", dpi=180)
        plt.close()

# ============================================================
# 主流程
# ============================================================

def main():
    raw, inventory = load_all_records()

    inventory.to_csv(OUT / "file_inventory.csv", index=False, encoding="utf-8-sig")
    raw.to_csv(OUT / "all_loaded_round_records.csv", index=False, encoding="utf-8-sig")

    method_inventory = (
        raw.groupby(["method_key", "method"])
        .agg(
            scene_count=("scene_key", "nunique"),
            rows=("cost", "size"),
        )
        .reset_index()
        .sort_values("method")
    )
    method_inventory.to_csv(OUT / "method_inventory.csv", index=False, encoding="utf-8-sig")

    print("\n[METHOD INVENTORY]")
    print(method_inventory.to_string(index=False))

    sm = scene_method_summary(raw)
    sm.to_csv(OUT / "scene_method_summary.csv", index=False, encoding="utf-8-sig")

    overall = aggregate_summary(sm, [])
    by_lambda = aggregate_summary(sm, ["lambda"])
    by_task = aggregate_summary(sm, ["task_group"])

    overall.to_csv(OUT / "method_summary_overall.csv", index=False, encoding="utf-8-sig")
    by_lambda.to_csv(OUT / "method_summary_by_lambda.csv", index=False, encoding="utf-8-sig")
    by_task.to_csv(OUT / "method_summary_by_task_group.csv", index=False, encoding="utf-8-sig")

    # gap records
    gap_parts = []
    for metric in ["cost", "delay", "energy"]:
        if metric in raw.columns:
            g = add_gap_vs_baseline(raw, metric)
            if not g.empty:
                gap_parts.append(g)

    gap_all = pd.concat(gap_parts, ignore_index=True)
    gap_all.to_csv(OUT / "rolling50_gap_vs_fixed_mid_records.csv", index=False, encoding="utf-8-sig")

    # gap curve csv
    gap_curve = (
        gap_all.groupby(["metric", "iter", "method_key", "method"])["gap_pct"]
        .mean()
        .reset_index()
    )
    gap_curve.to_csv(OUT / "rolling50_gap_curve_overall.csv", index=False, encoding="utf-8-sig")

    gap_curve_lambda = (
        gap_all.groupby(["metric", "lambda", "iter", "method_key", "method"])["gap_pct"]
        .mean()
        .reset_index()
    )
    gap_curve_lambda.to_csv(OUT / "rolling50_gap_curve_by_lambda.csv", index=False, encoding="utf-8-sig")

    gap_curve_task = (
        gap_all.groupby(["metric", "task_group", "iter", "method_key", "method"])["gap_pct"]
        .mean()
        .reset_index()
    )
    gap_curve_task.to_csv(OUT / "rolling50_gap_curve_by_task_group.csv", index=False, encoding="utf-8-sig")

    # 绘图
    for metric in ["cost", "delay", "energy"]:
        if metric not in raw.columns:
            continue

        metric_dir = OUT / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        plot_value_curve(
            raw,
            metric,
            outpath=metric_dir / f"{metric}_overall_rolling50_value.png",
            title=f"New data | overall rolling{ROLLING} {metric}",
        )

        plot_gap_curve(
            gap_all,
            metric,
            outpath=metric_dir / f"{metric}_overall_gap_pct_vs_fixed_mid.png",
            title=f"New data | overall rolling{ROLLING} {metric} gap vs Fixed-mid",
        )

        for lam in sorted(raw["lambda"].dropna().unique()):
            tag = str(lam).replace(".", "p")
            plot_gap_curve(
                gap_all,
                metric,
                group_col="lambda",
                group_value=lam,
                outpath=metric_dir / f"{metric}_lambda_{tag}_gap_pct_vs_fixed_mid.png",
                title=f"New data | lambda={lam} | rolling{ROLLING} {metric} gap vs Fixed-mid",
            )

        for tg in sorted(raw["task_group"].dropna().unique()):
            tag = str(tg).replace("-", "_")
            plot_gap_curve(
                gap_all,
                metric,
                group_col="task_group",
                group_value=tg,
                outpath=metric_dir / f"{metric}_task_{tag}_gap_pct_vs_fixed_mid.png",
                title=f"New data | {tg} | rolling{ROLLING} {metric} gap vs Fixed-mid",
            )

        # 分场景只画 CBO/BO gap
        plot_scene_cbo_gap(gap_all, metric)

    # Excel 汇总
    xlsx = OUT / "new_data_gap_107_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        method_inventory.to_excel(writer, sheet_name="method_inventory", index=False)
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_lambda.to_excel(writer, sheet_name="by_lambda", index=False)
        by_task.to_excel(writer, sheet_name="by_task_group", index=False)
        gap_curve.to_excel(writer, sheet_name="gap_curve_overall", index=False)
        gap_curve_lambda.to_excel(writer, sheet_name="gap_curve_lambda", index=False)
        gap_curve_task.to_excel(writer, sheet_name="gap_curve_task", index=False)

    print("\n" + "=" * 100)
    print("[OVERALL COST SUMMARY]")
    cols = [
        "method", "method_key", "scene_count",
        "final_roll50_cost",
        "gap_vs_Fixed-mid_cost_pct",
        "final_roll50_delay",
        "gap_vs_Fixed-mid_delay_pct",
        "final_roll50_energy",
        "gap_vs_Fixed-mid_energy_pct",
    ]
    cols = [c for c in cols if c in overall.columns]
    print(overall[cols].sort_values("final_roll50_cost").to_string(index=False))

    print("\n" + "=" * 100)
    print("[CBO ONLY]")
    cbo = overall[overall["method_key"].str.contains("cbo", case=False, na=False)].copy()
    print(cbo[cols].sort_values("final_roll50_cost").to_string(index=False))

    print("\n[DONE]")
    print("Output:", OUT)
    print("Excel:", xlsx)
    print("Main figures:")
    print(" ", OUT / "cost" / "cost_overall_gap_pct_vs_fixed_mid.png")
    print(" ", OUT / "delay" / "delay_overall_gap_pct_vs_fixed_mid.png")
    print(" ", OUT / "energy" / "energy_overall_gap_pct_vs_fixed_mid.png")

if __name__ == "__main__":
    main()
