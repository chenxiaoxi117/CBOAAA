from pathlib import Path
import math
import re
import pandas as pd
import numpy as np

ROOT = Path(r"D:\CBOv2\results\pressure_context_overnight_validation")
OUTDIR = ROOT / "analysis_prediction_gap_overnight"
OUTDIR.mkdir(parents=True, exist_ok=True)

def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def find_col(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        key = str(n).lower()
        if key in lower:
            return lower[key]
    for n in names:
        key = str(n).lower()
        for c in df.columns:
            if key in str(c).lower():
                return c
    return None

def infer_seed(path):
    m = re.search(r"seed(\d+)", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else None

def infer_scene(path):
    s = str(path)
    for key in ["P0_RT60_Batch30_AI10", "P1_RT50_Batch40_AI10", "P2_AI70_RT10_Batch20"]:
        if key in s:
            return key
    return "unknown"

def infer_method(path):
    s = str(path).lower()
    if "fixed_tuned" in s:
        return "fixed_tuned"
    if "cbo5d_prev_unfinished" in s or "prev_unfinished" in s:
        return "cbo5d_prev_unfinished"
    if "cbo6d_pressure_transition" in s or "pressure_transition" in s:
        return "cbo6d_pressure_transition"
    if "cbo4d_pressure_only" in s or "pressure_only" in s:
        return "cbo4d_pressure_only"
    return "unknown"

def stage_df(df, stage):
    if stage == "all":
        return df
    if stage == "first50":
        return df.iloc[:50]
    if stage == "first100":
        return df.iloc[:100]
    if stage == "101_200":
        return df.iloc[100:200]
    if stage == "201_350":
        return df.iloc[200:350]
    if stage == "tail100":
        return df.iloc[-100:]
    if stage == "last50":
        return df.iloc[-50:]
    return df

def summarize_stage(df, stage):
    g = stage_df(df, stage)

    actual_col = find_col(g, ["Eval_Cost_最终评估Cost", "Eval_Cost", "eval_cost"])
    mu_col = find_col(g, ["selected_candidate_mu"])
    sigma_col = find_col(g, ["selected_candidate_sigma"])

    if actual_col is None or mu_col is None:
        return {
            "stage": stage,
            "rows": len(g),
            "valid_prediction_rows": 0,
            "valid_prediction_rate": 0.0,
        }

    actual = pd.to_numeric(g[actual_col], errors="coerce")
    mu = pd.to_numeric(g[mu_col], errors="coerce")
    predicted = -mu
    error = actual - predicted

    valid = actual.notna() & predicted.notna() & error.notna()
    actual = actual[valid]
    predicted = predicted[valid]
    error = error[valid]

    row = {
        "stage": stage,
        "rows": len(g),
        "valid_prediction_rows": int(len(error)),
        "valid_prediction_rate": float(len(error) / max(1, len(g))),
    }

    if len(error) == 0:
        return row

    row.update({
        "actual_cost_mean": float(actual.mean()),
        "predicted_cost_mean": float(predicted.mean()),
        "prediction_error_bias": float(error.mean()),
        "prediction_error_mae": float(error.abs().mean()),
        "prediction_error_rmse": float(math.sqrt((error ** 2).mean())),
        "underestimate_rate": float((error > 0).mean()),
        "large_abs_error_gt_500": float((error.abs() > 500).mean()),
        "large_abs_error_gt_1000": float((error.abs() > 1000).mean()),
    })

    if sigma_col is not None:
        sigma = pd.to_numeric(g.loc[valid.index[valid], sigma_col], errors="coerce")
        sigma = sigma.replace([np.inf, -np.inf], np.nan).clip(lower=1e-9)
        surprise = error / sigma
        surprise = surprise.replace([np.inf, -np.inf], np.nan).dropna()
        if len(surprise):
            row.update({
                "surprise_abs_mean": float(surprise.abs().mean()),
                "surprise_rate_abs_gt_2": float((surprise.abs() > 2).mean()),
                "positive_surprise_rate_gt_2": float((surprise > 2).mean()),
            })

    # 服务压力相关性
    for key, names in {
        "delay": ["Avg_Delay_平均时延", "Avg_Delay"],
        "backlog": ["Backlog_积压任务数", "Backlog"],
        "unfinished": ["Unfinished_End_轮末未完成任务数", "Unfinished_End", "window_unfinished_total"],
    }.items():
        c = find_col(g, names)
        if c is not None:
            x = pd.to_numeric(g.loc[valid.index[valid], c], errors="coerce")
            if x.notna().sum() > 5:
                row[f"corr_error_{key}"] = float(error.corr(x))

    return row

files = [
    p for p in ROOT.rglob("*.csv")
    if "round_summary" in p.name.lower()
    and "_short_export" not in str(p).lower()
]

rows = []
for p in files:
    df = read_csv_safe(p)
    scene = infer_scene(p)
    method = infer_method(p)
    seed = infer_seed(p)

    if method == "fixed_tuned":
        continue

    for stage in ["all", "first100", "101_200", "201_350", "tail100", "last50"]:
        r = summarize_stage(df, stage)
        r.update({
            "scene": scene,
            "method": method,
            "seed": seed,
            "path": str(p),
        })
        rows.append(r)

summary = pd.DataFrame(rows)
summary.to_csv(OUTDIR / "prediction_gap_all_runs_stage_summary.csv", index=False, encoding="utf-8-sig")

agg_cols = [
    "prediction_error_bias",
    "prediction_error_mae",
    "underestimate_rate",
    "surprise_abs_mean",
    "surprise_rate_abs_gt_2",
    "positive_surprise_rate_gt_2",
    "corr_error_backlog",
    "corr_error_unfinished",
    "corr_error_delay",
]

agg_rows = []
for (scene, method, stage), g in summary.groupby(["scene", "method", "stage"]):
    row = {
        "scene": scene,
        "method": method,
        "stage": stage,
        "n_runs": len(g),
        "seeds": ",".join(str(int(x)) for x in sorted(g["seed"].dropna().unique())),
    }
    for c in agg_cols:
        if c in g.columns:
            row[c + "_mean"] = float(pd.to_numeric(g[c], errors="coerce").mean())
            row[c + "_std"] = float(pd.to_numeric(g[c], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0
    agg_rows.append(row)

agg = pd.DataFrame(agg_rows)
agg.to_csv(OUTDIR / "prediction_gap_aggregate.csv", index=False, encoding="utf-8-sig")

# 只抽 last50/tail100，方便看后期谁低估更严重
late = agg[agg["stage"].isin(["tail100", "last50"])].copy()
late.to_csv(OUTDIR / "prediction_gap_late_focus.csv", index=False, encoding="utf-8-sig")

report_lines = []
report_lines.append("# Prediction gap overnight analysis\n")
report_lines.append(f"Root: `{ROOT}`")
report_lines.append(f"Round summary files parsed: {len(files)}\n")

report_lines.append("## Late-stage focus\n")
show_cols = [
    "scene", "method", "stage", "n_runs", "seeds",
    "prediction_error_bias_mean",
    "prediction_error_mae_mean",
    "underestimate_rate_mean",
    "surprise_rate_abs_gt_2_mean",
    "positive_surprise_rate_gt_2_mean",
    "corr_error_backlog_mean",
    "corr_error_unfinished_mean",
]
show_cols = [c for c in show_cols if c in late.columns]
report_lines.append(late[show_cols].sort_values(["scene", "stage", "method"]).to_string(index=False))

report_lines.append("\n\n## Interpretation\n")
report_lines.append("- prediction_error_bias > 0 means BO underestimates actual cost.")
report_lines.append("- underestimate_rate > 0.6 means most rounds are worse than BO predicted.")
report_lines.append("- positive_surprise_rate_gt_2 high means confident underestimation.")
report_lines.append("- corr_error_backlog high means prediction error is strongly tied to backlog/unfinished risk.")

(OUTDIR / "prediction_gap_overnight_report.md").write_text("\n".join(report_lines), encoding="utf-8")

print("Done.")
print("Output:", OUTDIR)
print(late[show_cols].sort_values(["scene", "stage", "method"]).to_string(index=False))