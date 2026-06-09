from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\CBO\v6_3lambda_36_context500_results_107_plus_timeout\v6_3lambda_36_context500")
OUT = Path(r"C:\Users\POPchen\Desktop\duiibi_old_REAL_gap_107")
OUT.mkdir(parents=True, exist_ok=True)

ROLLING = 50
BASELINE_KEY = "reduced6_fixed_mid"
BASELINE_LABEL = "Fixed-mid"

LABEL_MAP = {
    "reduced6_fixed_mid": "Fixed-mid",
    "reduced6_fixed_tuned": "Fixed-tuned",
    "reduced6_fixed_risk_high": "Fixed-risk-high",
    "reduced6_bo_greedy": "BO-greedy",
    "reduced6_cbo_lite_pressure_only": "CBO-pressure",
    "reduced6_cbo_lite_full_taskmix_counts": "CBO-full",
}

METHOD_ORDER = [
    "reduced6_fixed_mid",
    "reduced6_fixed_tuned",
    "reduced6_fixed_risk_high",
    "reduced6_bo_greedy",
    "reduced6_cbo_lite_pressure_only",
    "reduced6_cbo_lite_full_taskmix_counts",
]

METHOD_COLS = ["Group_Key_方法键", "method", "Method", "method_key"]
ITER_COLS = ["Iteration_轮次", "Iteration", "iter", "round"]

METRIC_CANDIDATES = {
    "cost": ["Eval_Cost_最终评估Cost", "Eval_Cost", "Cost"],
    "delay": ["Avg_Delay_平均时延", "Average_Delay_平均时延", "Avg_Delay", "Delay", "Latency"],
    "energy": ["Avg_Energy_平均能耗", "Average_Energy_平均能耗", "Avg_Energy", "Energy", "Total_Energy"],
}

def label_of(k):
    return LABEL_MAP.get(k, k)

def pick_col(df, candidates, loose=None):
    for c in candidates:
        if c in df.columns:
            return c
    if loose:
        for c in df.columns:
            low = str(c).lower()
            if any(x.lower() in low for x in loose):
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().sum() > 0:
                    return c
    return None

def parse_scene(p):
    s = str(p)
    m = re.search(r"lam(?P<lam>\d+p\d+)_RT(?P<rt>\d+)_(?:B|Batch)(?P<b>\d+)_AI(?P<ai>\d+)", s)
    if not m:
        return None
    lam = float(m.group("lam").replace("p", "."))
    rt = int(m.group("rt"))
    b = int(m.group("b"))
    ai = int(m.group("ai"))
    vals = {"RT-heavy": rt, "Batch-heavy": b, "AI-heavy": ai}
    tg = max(vals, key=vals.get)
    task_group = tg if vals[tg] >= 50 else "Mixed"
    return {
        "scene_key": f"lam{str(lam).replace('.', 'p')}_RT{rt}_B{b}_AI{ai}",
        "lambda": lam,
        "rt": rt,
        "batch": b,
        "ai": ai,
        "task_group": task_group,
    }

def find_files(root):
    files = []
    seen = set()

    # ???????? task_effective ????? 6 ??? round_summary?
    # ??? task_effective/_short_export ???????
    allowed_prefixes = [
        "reduced6_bo_greedy",
        "reduced6_cbo_lite_full_taskmix_counts",
        "reduced6_cbo_lite_pressure_only",
        "reduced6_fixed_mid",
        "reduced6_fixed_risk_high",
        "reduced6_fixed_tuned",
    ]

    for p in root.rglob("*.csv"):
        name = p.name.lower()

        # ??????????? task_effective???? _short_export
        if p.parent.name.lower() != "task_effective":
            continue

        if ("round_summary" not in name) and ("????" not in name):
            continue

        if not any(name.startswith(prefix.lower()) for prefix in allowed_prefixes):
            continue

        if p not in seen:
            files.append(p)
            seen.add(p)

    return sorted(files)


def load():
    files = find_files(ROOT)
    print("[ROOT]", ROOT)
    print("[FILES]", len(files))

    parts = []
    inv = []

    for p in files:
        scene = parse_scene(p)
        if scene is None:
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except Exception:
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
            loose = ["cost"] if metric == "cost" else ["delay", "latency", "时延", "延迟"] if metric == "delay" else ["energy", "能耗"]
            col = pick_col(df, cands, loose)
            if col:
                metric_cols[metric] = col

        if "cost" not in metric_cols:
            continue

        methods = sorted(df[method_col].dropna().astype(str).unique())

        # 只读旧数据真正有的重点方法，避免误读分析汇总 CSV
        methods = [m for m in methods if m in METHOD_ORDER]
        if not methods:
            continue

        inv.append({
            "file": str(p),
            "scene_key": scene["scene_key"],
            "lambda": scene["lambda"],
            "methods": "|".join(methods),
            "rows": len(df),
        })

        for key in methods:
            sub = df[df[method_col].astype(str) == key].copy()
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

            parts.append(rec)

    if not parts:
        raise SystemExit("No records loaded. 检查旧 root 下 CSV 是否包含 Group_Key_方法键 / Eval_Cost。")

    return pd.concat(parts, ignore_index=True), pd.DataFrame(inv)

def add_gap(raw, metric):
    roll = f"roll50_{metric}"
    base = raw[raw["method_key"] == BASELINE_KEY][["scene_key", "iter", roll]].rename(columns={roll: "baseline"})
    d = raw.merge(base, on=["scene_key", "iter"], how="left")
    d = d.dropna(subset=[roll, "baseline"]).copy()
    d["gap_pct"] = 100 * (d[roll] - d["baseline"]) / d["baseline"]
    d["metric"] = metric
    return d

def summarize(raw):
    rows = []
    for (scene, key), g in raw.groupby(["scene_key", "method_key"]):
        g = g.sort_values("iter")
        f = g.iloc[0]
        r = {
            "scene_key": scene,
            "lambda": f["lambda"],
            "rt": f["rt"],
            "batch": f["batch"],
            "ai": f["ai"],
            "task_group": f["task_group"],
            "method_key": key,
            "method": f["method"],
            "rows": len(g),
        }
        for metric in ["cost", "delay", "energy"]:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce")
            rr = pd.to_numeric(g[f"roll50_{metric}"], errors="coerce")
            r[f"mean_{metric}"] = x.mean()
            r[f"tail100_{metric}"] = x.tail(100).mean()
            r[f"last50_{metric}"] = x.tail(50).mean()
            r[f"final_roll50_{metric}"] = rr.dropna().iloc[-1]
            r[f"min_roll50_{metric}"] = rr.min()
        rows.append(r)

    sm = pd.DataFrame(rows)
    for metric in ["cost", "delay", "energy"]:
        val = f"final_roll50_{metric}"
        base = sm[sm["method_key"] == BASELINE_KEY][["scene_key", val]].rename(columns={val: f"base_{metric}"})
        sm = sm.merge(base, on="scene_key", how="left")
        sm[f"gap_vs_Fixed-mid_{metric}_pct"] = 100 * (sm[val] - sm[f"base_{metric}"]) / sm[f"base_{metric}"]
    return sm

def aggregate(sm, groups):
    cols = [c for c in sm.columns if c.startswith(("mean_", "tail100_", "last50_", "final_roll50_", "min_roll50_", "gap_vs_"))]
    return (
        sm.groupby(groups + ["method_key", "method"])
        .agg(scene_count=("scene_key", "nunique"), **{c: (c, "mean") for c in cols})
        .reset_index()
        .sort_values(groups + ["final_roll50_cost"])
    )

def plot_gap(gap, metric, group_col=None, group_val=None, outpath=None, title=None):
    d = gap[gap["metric"] == metric].copy()
    if group_col:
        d = d[d[group_col] == group_val].copy()
    if d.empty:
        return
    curve = d.groupby(["iter", "method_key", "method"])["gap_pct"].mean().reset_index()
    order = {label_of(k): i for i, k in enumerate(METHOD_ORDER)}
    curve["sort"] = curve["method"].map(order).fillna(999)
    curve = curve.sort_values(["sort", "method"])

    plt.figure(figsize=(11, 6))
    for method, g in curve.groupby("method", sort=False):
        if method == BASELINE_LABEL:
            continue
        gg = g.sort_values("iter")
        plt.plot(gg["iter"], gg["gap_pct"], linewidth=1.8, label=method)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel(f"Rolling{ROLLING} gap vs Fixed-mid (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def main():
    raw, inv = load()
    inv.to_csv(OUT / "file_inventory.csv", index=False, encoding="utf-8-sig")
    raw.to_csv(OUT / "all_loaded_round_records.csv", index=False, encoding="utf-8-sig")

    mi = raw.groupby(["method_key", "method"]).agg(scene_count=("scene_key", "nunique"), rows=("cost", "size")).reset_index()
    print("\n[METHOD INVENTORY]")
    print(mi.to_string(index=False))
    mi.to_csv(OUT / "method_inventory.csv", index=False, encoding="utf-8-sig")

    sm = summarize(raw)
    sm.to_csv(OUT / "scene_method_summary.csv", index=False, encoding="utf-8-sig")

    overall = aggregate(sm, [])
    by_lam = aggregate(sm, ["lambda"])
    by_task = aggregate(sm, ["task_group"])

    overall.to_csv(OUT / "method_summary_overall.csv", index=False, encoding="utf-8-sig")
    by_lam.to_csv(OUT / "method_summary_by_lambda.csv", index=False, encoding="utf-8-sig")
    by_task.to_csv(OUT / "method_summary_by_task_group.csv", index=False, encoding="utf-8-sig")

    gaps = []
    for metric in ["cost", "delay", "energy"]:
        if metric in raw.columns:
            gaps.append(add_gap(raw, metric))
    gap = pd.concat(gaps, ignore_index=True)
    gap.to_csv(OUT / "rolling50_gap_vs_fixed_mid_records.csv", index=False, encoding="utf-8-sig")

    for metric in ["cost", "delay", "energy"]:
        mdir = OUT / metric
        mdir.mkdir(parents=True, exist_ok=True)
        plot_gap(gap, metric, outpath=mdir / f"{metric}_overall_gap_pct_vs_fixed_mid.png",
                 title=f"Old REAL data | overall rolling{ROLLING} {metric} gap vs Fixed-mid")
        for lam in sorted(raw["lambda"].unique()):
            tag = str(lam).replace(".", "p")
            plot_gap(gap, metric, "lambda", lam, mdir / f"{metric}_lambda_{tag}_gap_pct_vs_fixed_mid.png",
                     f"Old REAL data | lambda={lam} | rolling{ROLLING} {metric} gap vs Fixed-mid")
        for tg in sorted(raw["task_group"].unique()):
            tag = tg.replace("-", "_")
            plot_gap(gap, metric, "task_group", tg, mdir / f"{metric}_task_{tag}_gap_pct_vs_fixed_mid.png",
                     f"Old REAL data | {tg} | rolling{ROLLING} {metric} gap vs Fixed-mid")

    xlsx = OUT / "old_REAL_data_gap_107_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        mi.to_excel(writer, sheet_name="method_inventory", index=False)
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_lam.to_excel(writer, sheet_name="by_lambda", index=False)
        by_task.to_excel(writer, sheet_name="by_task_group", index=False)

    print("\n" + "=" * 100)
    print("[OVERALL COST SUMMARY]")
    cols = ["method", "method_key", "scene_count", "final_roll50_cost", "gap_vs_Fixed-mid_cost_pct",
            "final_roll50_delay", "gap_vs_Fixed-mid_delay_pct", "final_roll50_energy", "gap_vs_Fixed-mid_energy_pct"]
    cols = [c for c in cols if c in overall.columns]
    print(overall[cols].sort_values("final_roll50_cost").to_string(index=False))

    print("\n" + "=" * 100)
    print("[CBO ONLY]")
    cbo = overall[overall["method_key"].str.contains("cbo", case=False, na=False)]
    print(cbo[cols].sort_values("final_roll50_cost").to_string(index=False))

    print("\n[DONE]")
    print("Output:", OUT)
    print("Excel:", xlsx)

if __name__ == "__main__":
    main()
