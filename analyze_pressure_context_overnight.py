from pathlib import Path
import math
import re
import pandas as pd
import numpy as np

ROOT = Path(r"D:\CBOv2\results\pressure_context_overnight_validation")
OUTDIR = ROOT / "analysis_pressure_context_overnight"
OUTDIR.mkdir(parents=True, exist_ok=True)

MANIFEST = ROOT / "runs_manifest.csv"


def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def find_round_summary(run_dir: Path):
    if not run_dir.exists():
        return None
    files = [
        p for p in run_dir.rglob("*.csv")
        if "round_summary" in p.name.lower()
        and "_short_export" not in str(p).lower()
    ]
    if not files:
        return None
    scored = []
    for p in files:
        try:
            df = read_csv_safe(p)
            scored.append((abs(len(df) - 500), -len(df), len(str(p)), p))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort()
    return scored[0][3]


def find_col(df: pd.DataFrame, candidates, fuzzy=True):
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower:
            return lower[key]
    if fuzzy:
        for name in candidates:
            key = str(name).lower()
            for c in df.columns:
                if key in str(c).lower():
                    return c
    return None


def series_num(df, candidates):
    c = find_col(df, candidates)
    if c is None:
        return None, None
    return c, pd.to_numeric(df[c], errors="coerce")


def mean_part(s, start=None, end=None):
    s = pd.to_numeric(s, errors="coerce")
    if start is None and end is None:
        x = s
    elif start is None:
        x = s.iloc[:end]
    elif end is None:
        x = s.iloc[start:]
    else:
        x = s.iloc[start:end]
    x = x.dropna()
    return float(x.mean()) if len(x) else math.nan


def rolling50_info(s):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rolling(50, min_periods=50).mean()
    valid = r.dropna()
    if valid.empty:
        return math.nan, math.nan, math.nan, math.nan
    min_idx = valid.idxmin()
    rmin = float(valid.loc[min_idx])
    rfinal = float(valid.iloc[-1])
    rebound = (rfinal - rmin) / rmin * 100.0 if rmin else math.nan
    return rmin, int(min_idx) + 1, rfinal, rebound


def infer_seed(x):
    text = str(x)
    m = re.search(r"seed(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def normalize_method(method_name, selected_key, output_root):
    text = " ".join([str(method_name), str(selected_key), str(output_root)]).lower()
    if "fixed_tuned" in text:
        return "fixed_tuned"
    if "prev_unfinished" in text or "5d" in text:
        return "cbo5d_prev_unfinished"
    if "transition" in text or "6d" in text:
        return "cbo6d_pressure_transition"
    if "pressure_only" in text or "4d" in text:
        return "cbo4d_pressure_only"
    return str(method_name or selected_key or "unknown")


def summarize_run(meta, csv_path: Path):
    df = read_csv_safe(csv_path)

    cost_col, cost = series_num(df, ["Eval_Cost", "eval_cost", "cost", "Cost"])
    bo_col, bo = series_num(df, ["BO_Training_Cost", "bo_training_cost"])

    if cost is None:
        raise ValueError(f"No cost column found in {csv_path}")

    rmin, rmin_iter, rfinal, rebound = rolling50_info(cost)

    row = dict(meta)
    row.update({
        "round_summary_path": str(csv_path),
        "rows": int(len(df)),
        "cost_col": cost_col or "",
        "mean_cost": mean_part(cost),
        "first50": mean_part(cost, 0, 50),
        "first100": mean_part(cost, 0, 100),
        "stage_101_200": mean_part(cost, 100, 200),
        "stage_201_350": mean_part(cost, 200, 350),
        "tail100": mean_part(cost, -100, None),
        "last50": mean_part(cost, -50, None),
        "rolling50_min": rmin,
        "rolling50_min_iter": rmin_iter,
        "rolling50_final": rfinal,
        "rebound_pct": rebound,
    })

    if bo is not None:
        diff = (pd.to_numeric(cost, errors="coerce") - pd.to_numeric(bo, errors="coerce")).abs()
        row["bo_training_cost_col"] = bo_col
        row["bo_eval_mismatch_count"] = int((diff > 1e-8).sum())
        row["bo_eval_max_abs_diff"] = float(diff.max())
    else:
        row["bo_training_cost_col"] = ""
        row["bo_eval_mismatch_count"] = math.nan
        row["bo_eval_max_abs_diff"] = math.nan

    metric_defs = {
        "delay": ["Avg_Delay", "avg_delay", "Delay"],
        "energy": ["Avg_Energy", "avg_energy", "Energy"],
        "backlog": ["Backlog", "backlog", "Avg_Backlog"],
        "unfinished": ["unfinished_end", "Unfinished_End", "window_unfinished_total", "unfinished"],
        "violation": ["Violation", "Violation_Rate", "violation_rate", "sla_violation_rate"],
        "arrivals_total": ["Arrivals_Total", "arrivals_total", "window_arrivals_total"],
        "arrivals_rt": ["Arrivals_RT", "arrivals_rt", "rt_arrivals"],
        "arrivals_batch": ["Arrivals_Batch", "arrivals_batch", "batch_arrivals"],
        "arrivals_ai": ["Arrivals_AI", "arrivals_ai", "ai_arrivals"],
        "context_dim": ["context_dim"],
        "model_input_dim": ["model_input_dim", "total_model_dim"],
        "control_dim": ["control_dim"],
    }

    for key, names in metric_defs.items():
        col, s = series_num(df, names)
        row[f"{key}_col"] = col or ""
        if s is None:
            row[f"{key}_mean"] = math.nan
            row[f"{key}_first100"] = math.nan
            row[f"{key}_tail100"] = math.nan
            row[f"{key}_last50"] = math.nan
            row[f"{key}_last"] = math.nan
        else:
            row[f"{key}_mean"] = mean_part(s)
            row[f"{key}_first100"] = mean_part(s, 0, 100)
            row[f"{key}_tail100"] = mean_part(s, -100, None)
            row[f"{key}_last50"] = mean_part(s, -50, None)
            row[f"{key}_last"] = float(s.dropna().iloc[-1]) if len(s.dropna()) else math.nan

    # String diagnostics
    for key, names in {
        "context_mode": ["context_mode"],
        "context_feature_names": ["context_feature_names"],
        "selected_key": ["selected_key"],
        "scheduler_tradeoff_mode": ["scheduler_tradeoff_mode"],
    }.items():
        col = find_col(df, names)
        if col is not None and len(df):
            vals = df[col].dropna().astype(str).unique().tolist()
            row[key] = "|".join(vals[:5])
        else:
            row[key] = ""

    return row


# Load manifest
runs = []
if MANIFEST.exists():
    mf = read_csv_safe(MANIFEST)
    for _, r in mf.iterrows():
        output_root = Path(str(r.get("output_root", "")))
        selected_key = str(r.get("selected_key", ""))
        method_name = str(r.get("method_name", ""))
        scene = str(r.get("scene", r.get("scene_name", "")))
        scene_name = str(r.get("scene_name", scene))
        seed = r.get("seed", None)
        try:
            seed = int(seed)
        except Exception:
            seed = infer_seed(output_root)

        method = normalize_method(method_name, selected_key, output_root)
        status = str(r.get("status", ""))

        csv_path = find_round_summary(output_root)
        runs.append({
            "scene": scene,
            "scene_name": scene_name,
            "seed": seed,
            "method": method,
            "selected_key_manifest": selected_key,
            "output_root": str(output_root),
            "manifest_status": status,
            "csv_path": csv_path,
        })
else:
    for p in ROOT.rglob("*.csv"):
        if "round_summary" in p.name.lower() and "_short_export" not in str(p).lower():
            seed = infer_seed(p)
            method = normalize_method("", p.name, p)
            scene = ""
            for part in p.parts:
                if part.startswith("P") and ("RT" in part or "AI" in part or "Batch" in part):
                    scene = part
            runs.append({
                "scene": scene,
                "scene_name": scene,
                "seed": seed,
                "method": method,
                "selected_key_manifest": "",
                "output_root": str(p.parent),
                "manifest_status": "found_by_recurse",
                "csv_path": p,
            })

summary_rows = []
missing_rows = []
for r in runs:
    csv_path = r.pop("csv_path")
    if csv_path is None:
        missing_rows.append(r)
        continue
    try:
        summary_rows.append(summarize_run(r, csv_path))
    except Exception as e:
        rr = dict(r)
        rr["error"] = type(e).__name__ + ": " + str(e)
        missing_rows.append(rr)

summary = pd.DataFrame(summary_rows)
missing = pd.DataFrame(missing_rows)

summary.to_csv(OUTDIR / "pressure_context_run_summary.csv", index=False, encoding="utf-8-sig")
missing.to_csv(OUTDIR / "pressure_context_missing_or_failed.csv", index=False, encoding="utf-8-sig")

# Pairwise comparisons by scene + seed
metrics = [
    "mean_cost", "first50", "first100", "stage_101_200", "stage_201_350",
    "tail100", "last50", "rolling50_min", "rolling50_final",
    "delay_mean", "delay_tail100", "delay_last50",
    "energy_mean", "energy_tail100", "energy_last50",
    "backlog_mean", "backlog_tail100", "backlog_last50",
    "unfinished_mean", "unfinished_tail100", "unfinished_last50",
]

pair_rows = []
if not summary.empty:
    idx = summary.set_index(["scene", "seed", "method"], drop=False)
    for (scene, seed), g in summary.groupby(["scene", "seed"]):
        methods = set(g["method"])
        base4 = g[g["method"] == "cbo4d_pressure_only"]
        fixed = g[g["method"] == "fixed_tuned"]
        if base4.empty:
            continue
        base4 = base4.iloc[0]
        fixed_row = fixed.iloc[0] if not fixed.empty else None

        for _, target in g.iterrows():
            if target["method"] == "cbo4d_pressure_only":
                continue
            row = {
                "scene": scene,
                "seed": seed,
                "target_method": target["method"],
                "base_method": "cbo4d_pressure_only",
            }
            for m in metrics:
                if m in summary.columns:
                    b = base4.get(m, np.nan)
                    t = target.get(m, np.nan)
                    row[f"{m}_base4"] = b
                    row[f"{m}_target"] = t
                    row[f"{m}_gain_vs_4d_pct"] = (b - t) / b * 100.0 if pd.notna(b) and abs(b) > 1e-12 and pd.notna(t) else np.nan
            if fixed_row is not None:
                for m in ["mean_cost", "first100", "tail100", "last50", "rolling50_min"]:
                    f = fixed_row.get(m, np.nan)
                    t = target.get(m, np.nan)
                    row[f"{m}_fixed"] = f
                    row[f"{m}_gap_vs_fixed_pct"] = (t - f) / f * 100.0 if pd.notna(f) and abs(f) > 1e-12 and pd.notna(t) else np.nan
            pair_rows.append(row)

pairwise = pd.DataFrame(pair_rows)
pairwise.to_csv(OUTDIR / "pressure_context_pairwise_vs_4d.csv", index=False, encoding="utf-8-sig")

# Aggregate summaries
agg_metrics = [
    "mean_cost", "first100", "stage_101_200", "stage_201_350",
    "tail100", "last50", "rolling50_min", "rolling50_final", "rebound_pct",
    "delay_mean", "delay_last50", "energy_mean", "energy_last50",
    "backlog_mean", "backlog_last50", "unfinished_mean", "unfinished_last50",
]

agg_rows = []
for (scene, method), g in summary.groupby(["scene", "method"]):
    row = {
        "scene": scene,
        "method": method,
        "n_runs": int(len(g)),
        "seeds": ",".join(str(int(x)) for x in sorted(g["seed"].dropna().unique())),
    }
    for m in agg_metrics:
        if m in g.columns:
            row[f"{m}_mean"] = float(pd.to_numeric(g[m], errors="coerce").mean())
            row[f"{m}_std"] = float(pd.to_numeric(g[m], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0
    agg_rows.append(row)

aggregate = pd.DataFrame(agg_rows)
aggregate.to_csv(OUTDIR / "pressure_context_aggregate_by_scene_method.csv", index=False, encoding="utf-8-sig")

pair_agg_rows = []
if not pairwise.empty:
    for (scene, target_method), g in pairwise.groupby(["scene", "target_method"]):
        row = {
            "scene": scene,
            "target_method": target_method,
            "n_pairs": int(len(g)),
            "seeds": ",".join(str(int(x)) for x in sorted(g["seed"].dropna().unique())),
        }
        for m in ["first100", "stage_101_200", "stage_201_350", "tail100", "last50", "rolling50_min", "mean_cost"]:
            c = f"{m}_gain_vs_4d_pct"
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce")
                row[f"{m}_gain_vs_4d_mean_pct"] = float(vals.mean())
                row[f"{m}_gain_vs_4d_std_pct"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
                row[f"{m}_better_ratio"] = float((vals > 0).mean())
        for m in ["first100", "tail100", "last50", "rolling50_min", "mean_cost"]:
            c = f"{m}_gap_vs_fixed_pct"
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce")
                row[f"{m}_gap_vs_fixed_mean_pct"] = float(vals.mean())
        pair_agg_rows.append(row)

pair_agg = pd.DataFrame(pair_agg_rows)
pair_agg.to_csv(OUTDIR / "pressure_context_pairwise_aggregate.csv", index=False, encoding="utf-8-sig")

# Data quality
quality = []
for _, r in summary.iterrows():
    quality.append({
        "scene": r["scene"],
        "seed": r["seed"],
        "method": r["method"],
        "rows": r["rows"],
        "bo_eval_mismatch_count": r.get("bo_eval_mismatch_count", np.nan),
        "bo_eval_max_abs_diff": r.get("bo_eval_max_abs_diff", np.nan),
        "arrivals_total_sum": r.get("arrivals_total_mean", np.nan) * r.get("rows", np.nan),
        "context_dim_last": r.get("context_dim_last", np.nan),
        "model_input_dim_last": r.get("model_input_dim_last", np.nan),
        "context_mode": r.get("context_mode", ""),
        "path": r.get("round_summary_path", ""),
    })

quality_df = pd.DataFrame(quality)
quality_df.to_csv(OUTDIR / "pressure_context_quality_check.csv", index=False, encoding="utf-8-sig")

# Markdown report
lines = []
lines.append("# Pressure context overnight validation report\n")
lines.append(f"Root: `{ROOT}`\n")
lines.append(f"Runs found: {len(summary)}")
lines.append(f"Missing/failed entries: {len(missing)}\n")

if not missing.empty:
    lines.append("## Missing / failed\n")
    lines.append(missing.to_string(index=False))
    lines.append("")

lines.append("## Data quality highlights\n")
if not quality_df.empty:
    bad_rows = quality_df[(quality_df["rows"] != 500) | (quality_df["bo_eval_mismatch_count"].fillna(0) != 0)]
    if bad_rows.empty:
        lines.append("- All parsed runs have 500 rows and BO_Training_Cost == Eval_Cost.")
    else:
        lines.append("- Some runs have row-count or BO/Eval mismatch issues:")
        lines.append(bad_rows[["scene", "seed", "method", "rows", "bo_eval_mismatch_count", "bo_eval_max_abs_diff"]].to_string(index=False))
lines.append("")

lines.append("## Aggregate by scene and method\n")
if not aggregate.empty:
    show_cols = [
        "scene", "method", "n_runs", "seeds",
        "mean_cost_mean", "first100_mean", "tail100_mean", "last50_mean",
        "rolling50_min_mean", "rebound_pct_mean",
        "delay_last50_mean", "backlog_last50_mean", "unfinished_last50_mean",
    ]
    show_cols = [c for c in show_cols if c in aggregate.columns]
    lines.append(aggregate[show_cols].sort_values(["scene", "method"]).to_string(index=False))
lines.append("")

lines.append("## Pairwise gains vs 4D pressure_only\n")
if not pair_agg.empty:
    show_cols = [
        "scene", "target_method", "n_pairs", "seeds",
        "mean_cost_gain_vs_4d_mean_pct",
        "first100_gain_vs_4d_mean_pct",
        "tail100_gain_vs_4d_mean_pct",
        "last50_gain_vs_4d_mean_pct",
        "rolling50_min_gain_vs_4d_mean_pct",
        "last50_better_ratio",
        "tail100_better_ratio",
    ]
    show_cols = [c for c in show_cols if c in pair_agg.columns]
    lines.append(pair_agg[show_cols].sort_values(["scene", "target_method"]).to_string(index=False))
lines.append("")

lines.append("## Gap vs fixed_tuned\n")
if not pair_agg.empty:
    show_cols = [
        "scene", "target_method", "n_pairs",
        "mean_cost_gap_vs_fixed_mean_pct",
        "first100_gap_vs_fixed_mean_pct",
        "tail100_gap_vs_fixed_mean_pct",
        "last50_gap_vs_fixed_mean_pct",
        "rolling50_min_gap_vs_fixed_mean_pct",
    ]
    show_cols = [c for c in show_cols if c in pair_agg.columns]
    if len(show_cols) > 3:
        lines.append(pair_agg[show_cols].sort_values(["scene", "target_method"]).to_string(index=False))
    else:
        lines.append("No fixed_tuned paired runs found for some/all scenes.")
lines.append("")

lines.append("## Suggested reading of results\n")
lines.append("- Positive `gain_vs_4d` means the method is better than 4D pressure_only.")
lines.append("- Positive `gap_vs_fixed` means the method is worse than fixed_tuned; negative means it beats fixed_tuned.")
lines.append("- Focus first on `tail100` and `last50`; this is the stability question.")
lines.append("- Then check `rolling50_min`; this tells whether the method ever found a better region.")
lines.append("- Check delay/backlog/unfinished to see whether gains come from service quality rather than only energy.")

report = "\n".join(lines)
(OUTDIR / "pressure_context_overnight_report.md").write_text(report, encoding="utf-8")

print("\nAnalysis completed.")
print(f"Output directory:\n{OUTDIR}")
print("\nKey files:")
for name in [
    "pressure_context_run_summary.csv",
    "pressure_context_pairwise_vs_4d.csv",
    "pressure_context_aggregate_by_scene_method.csv",
    "pressure_context_pairwise_aggregate.csv",
    "pressure_context_quality_check.csv",
    "pressure_context_overnight_report.md",
]:
    print(OUTDIR / name)

print("\nQuick aggregate preview:")
if not pair_agg.empty:
    cols = [
        "scene", "target_method", "n_pairs",
        "tail100_gain_vs_4d_mean_pct",
        "last50_gain_vs_4d_mean_pct",
        "rolling50_min_gain_vs_4d_mean_pct",
        "last50_better_ratio",
    ]
    cols = [c for c in cols if c in pair_agg.columns]
    print(pair_agg[cols].sort_values(["scene", "target_method"]).to_string(index=False))
else:
    print("No pairwise aggregate generated.")