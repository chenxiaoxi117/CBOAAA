from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"D:\CBOv2\results\transfer_cbo_pressure")
OUTDIR = ROOT / "analysis_warm_start_transfer"
FIGDIR = OUTDIR / "figures"

RUNS = {
    "similar_cold": ROOT / "target_similar_cold_lam2p6_RT50_Batch40_AI10",
    "similar_warm": ROOT / "target_similar_warm_lam2p6_RT50_Batch40_AI10",
    "dissimilar_cold": ROOT / "target_dissimilar_cold_lam3p0_RT10_Batch20_AI70",
    "dissimilar_warm": ROOT / "target_dissimilar_warm_lam3p0_RT10_Batch20_AI70",
}

PAIRS = {
    "similar": ("similar_cold", "similar_warm"),
    "dissimilar": ("dissimilar_cold", "dissimilar_warm"),
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def find_round_summary(run_dir: Path) -> Path:
    candidates = [
        p for p in run_dir.rglob("*.csv")
        if "round_summary" in p.name.lower()
        and "_short_export" not in str(p).lower()
    ]

    if not candidates:
        raise FileNotFoundError(f"No round_summary csv found under: {run_dir}")

    scored = []
    for p in candidates:
        try:
            df = read_csv_safely(p)
            scored.append((len(df), len(str(p)), p))
        except Exception:
            continue

    if not scored:
        raise FileNotFoundError(f"Found candidates but none readable under: {run_dir}")

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def find_col(df: pd.DataFrame, candidates: List[str], fuzzy: bool = True) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if fuzzy:
        for name in candidates:
            key = name.lower()
            for c in cols:
                if key in str(c).lower():
                    return c

    return None


def metric_col(df: pd.DataFrame, metric: str) -> Optional[str]:
    mapping = {
        "iter": [
            "Iteration_轮次", "Iteration", "iteration", "Iter", "iter", "Round", "round"
        ],
        "cost": [
            "Eval_Cost", "eval_cost", "BO_Training_Cost", "Cost", "cost"
        ],
        "delay": [
            "Avg_Delay", "Average_Delay", "avg_delay", "Delay", "delay"
        ],
        "energy": [
            "Avg_Energy", "Average_Energy", "avg_energy", "Energy", "energy"
        ],
        "backlog": [
            "Backlog", "Avg_Backlog", "avg_backlog", "Backlog_End", "backlog"
        ],
        "unfinished": [
            "unfinished_end", "Unfinished_End", "Unfinished", "unfinished",
            "Unfinished_Tasks", "unfinished_tasks"
        ],
        "violation": [
            "Violation", "violation", "Violation_Rate", "SLA_Violation", "sla_violation"
        ],
        "sla": [
            "SLA", "sla", "SLA_Satisfaction", "Completion_Rate", "completion_rate"
        ],
        "warm_mode": [
            "cbo_warm_start_mode", "warm_start_mode"
        ],
        "warm_loaded": [
            "cbo_warm_start_loaded_rows", "warm_start_loaded_rows"
        ],
        "warm_used": [
            "cbo_warm_start_used_rows", "warm_start_used_rows"
        ],
        "warm_selected": [
            "selected_warm_rows_count", "warm_rows_count"
        ],
        "local_selected": [
            "selected_local_rows_count", "local_rows_count"
        ],
    }
    return find_col(df, mapping.get(metric, []))


def numeric_series(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def window_mean(s: pd.Series, n: int, mode: str) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return math.nan
    if mode == "first":
        return float(s.head(n).mean())
    if mode == "last":
        return float(s.tail(n).mean())
    raise ValueError(mode)


def rolling_info(df: pd.DataFrame, cost_col: str, iter_col: Optional[str], window: int = 50) -> Tuple[float, float, float, float]:
    cost = pd.to_numeric(df[cost_col], errors="coerce")
    roll = cost.rolling(window, min_periods=window).mean()

    if roll.dropna().empty:
        return math.nan, math.nan, math.nan, math.nan

    min_idx = roll.idxmin()
    rolling_min = float(roll.loc[min_idx])
    rolling_final = float(roll.dropna().iloc[-1])

    if iter_col and iter_col in df.columns:
        iter_series = pd.to_numeric(df[iter_col], errors="coerce")
        min_iter = float(iter_series.loc[min_idx]) if pd.notna(iter_series.loc[min_idx]) else float(min_idx + 1)
    else:
        min_iter = float(min_idx + 1)

    rebound_pct = (
        (rolling_final - rolling_min) / rolling_min * 100.0
        if rolling_min and pd.notna(rolling_min)
        else math.nan
    )
    return rolling_min, min_iter, rolling_final, rebound_pct


def summarize_run(name: str, df: pd.DataFrame, path: Path) -> Dict:
    iter_col = metric_col(df, "iter")
    cost_col = metric_col(df, "cost")

    if cost_col is None:
        raise ValueError(f"{name}: cannot find Eval_Cost column. Columns={list(df.columns)}")

    cost = numeric_series(df, cost_col)
    rolling_min, rolling_min_iter, rolling_final, rebound_pct = rolling_info(df, cost_col, iter_col)

    row = {
        "run": name,
        "rows": len(df),
        "round_summary_path": str(path),
        "cost_col": cost_col,
        "iter_col": iter_col or "",
        "mean_eval_cost": float(cost.mean()),
        "first50_eval_cost": window_mean(cost, 50, "first"),
        "first100_eval_cost": window_mean(cost, 100, "first"),
        "tail100_eval_cost": window_mean(cost, 100, "last"),
        "last50_eval_cost": window_mean(cost, 50, "last"),
        "rolling50_min": rolling_min,
        "rolling50_min_iter": rolling_min_iter,
        "rolling50_final": rolling_final,
        "rebound_pct": rebound_pct,
    }

    for key, label in [
        ("delay", "avg_delay"),
        ("energy", "avg_energy"),
        ("backlog", "backlog"),
        ("unfinished", "unfinished"),
        ("violation", "violation"),
        ("sla", "sla"),
    ]:
        col = metric_col(df, key)
        s = numeric_series(df, col)
        row[f"{label}_col"] = col or ""
        if s is not None:
            row[f"{label}_mean"] = float(s.mean())
            row[f"{label}_first100"] = window_mean(s, 100, "first")
            row[f"{label}_tail100"] = window_mean(s, 100, "last")
            row[f"{label}_last50"] = window_mean(s, 50, "last")
        else:
            row[f"{label}_mean"] = math.nan
            row[f"{label}_first100"] = math.nan
            row[f"{label}_tail100"] = math.nan
            row[f"{label}_last50"] = math.nan

    for key in ["warm_mode", "warm_loaded", "warm_used", "warm_selected", "local_selected"]:
        col = metric_col(df, key)
        row[f"{key}_col"] = col or ""
        if col is None:
            row[f"{key}_unique_or_mean"] = ""
            row[f"{key}_max"] = math.nan
        else:
            ser = df[col]
            num = pd.to_numeric(ser, errors="coerce")
            if num.notna().any():
                row[f"{key}_unique_or_mean"] = float(num.mean())
                row[f"{key}_max"] = float(num.max())
            else:
                vals = sorted({str(x) for x in ser.dropna().unique()})
                row[f"{key}_unique_or_mean"] = json.dumps(vals, ensure_ascii=False)
                row[f"{key}_max"] = math.nan

    return row


def align_pair(cold_df: pd.DataFrame, warm_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    cold_cost_col = metric_col(cold_df, "cost")
    warm_cost_col = metric_col(warm_df, "cost")
    cold_iter_col = metric_col(cold_df, "iter")
    warm_iter_col = metric_col(warm_df, "iter")

    if cold_cost_col is None or warm_cost_col is None:
        raise ValueError("Cannot find cost column in pair.")

    c = cold_df.copy()
    w = warm_df.copy()

    if cold_iter_col and warm_iter_col:
        c["_iter_key"] = pd.to_numeric(c[cold_iter_col], errors="coerce")
        w["_iter_key"] = pd.to_numeric(w[warm_iter_col], errors="coerce")
        merged = pd.merge(
            c[["_iter_key", cold_cost_col]],
            w[["_iter_key", warm_cost_col]],
            on="_iter_key",
            how="inner",
            suffixes=("_cold", "_warm"),
        )
        mode = "iteration"
    else:
        n = min(len(c), len(w))
        merged = pd.DataFrame({
            "_iter_key": range(1, n + 1),
            f"{cold_cost_col}_cold": pd.to_numeric(c[cold_cost_col].iloc[:n], errors="coerce").to_numpy(),
            f"{warm_cost_col}_warm": pd.to_numeric(w[warm_cost_col].iloc[:n], errors="coerce").to_numpy(),
        })
        mode = "row_index"

    cold_col = [x for x in merged.columns if x.endswith("_cold")][0]
    warm_col = [x for x in merged.columns if x.endswith("_warm")][0]

    merged["cold_eval_cost"] = pd.to_numeric(merged[cold_col], errors="coerce")
    merged["warm_eval_cost"] = pd.to_numeric(merged[warm_col], errors="coerce")
    merged["delta_warm_minus_cold"] = merged["warm_eval_cost"] - merged["cold_eval_cost"]
    merged["rolling50_delta"] = merged["delta_warm_minus_cold"].rolling(50, min_periods=50).mean()

    return merged[["_iter_key", "cold_eval_cost", "warm_eval_cost", "delta_warm_minus_cold", "rolling50_delta"]], mode


def compare_pair(pair_name: str, cold_name: str, warm_name: str, dfs: Dict[str, pd.DataFrame], summaries: pd.DataFrame) -> Dict:
    cold_df = dfs[cold_name]
    warm_df = dfs[warm_name]
    merged, align_mode = align_pair(cold_df, warm_df)

    cold_summary = summaries.set_index("run").loc[cold_name]
    warm_summary = summaries.set_index("run").loc[warm_name]

    def gain(metric: str) -> float:
        cold_val = float(cold_summary[metric])
        warm_val = float(warm_summary[metric])
        return (cold_val - warm_val) / cold_val if cold_val else math.nan

    row = {
        "pair": pair_name,
        "cold_run": cold_name,
        "warm_run": warm_name,
        "align_mode": align_mode,
        "paired_rows": len(merged),

        "cold_mean": cold_summary["mean_eval_cost"],
        "warm_mean": warm_summary["mean_eval_cost"],
        "delta_mean_summary": warm_summary["mean_eval_cost"] - cold_summary["mean_eval_cost"],
        "gain_mean": gain("mean_eval_cost"),

        "cold_first50": cold_summary["first50_eval_cost"],
        "warm_first50": warm_summary["first50_eval_cost"],
        "delta_first50": warm_summary["first50_eval_cost"] - cold_summary["first50_eval_cost"],
        "gain_first50": gain("first50_eval_cost"),

        "cold_first100": cold_summary["first100_eval_cost"],
        "warm_first100": warm_summary["first100_eval_cost"],
        "delta_first100": warm_summary["first100_eval_cost"] - cold_summary["first100_eval_cost"],
        "gain_first100": gain("first100_eval_cost"),

        "cold_tail100": cold_summary["tail100_eval_cost"],
        "warm_tail100": warm_summary["tail100_eval_cost"],
        "delta_tail100": warm_summary["tail100_eval_cost"] - cold_summary["tail100_eval_cost"],
        "gain_tail100": gain("tail100_eval_cost"),

        "cold_last50": cold_summary["last50_eval_cost"],
        "warm_last50": warm_summary["last50_eval_cost"],
        "delta_last50": warm_summary["last50_eval_cost"] - cold_summary["last50_eval_cost"],
        "gain_last50": gain("last50_eval_cost"),

        "cold_rolling50_min": cold_summary["rolling50_min"],
        "warm_rolling50_min": warm_summary["rolling50_min"],
        "delta_rolling50_min": warm_summary["rolling50_min"] - cold_summary["rolling50_min"],
        "gain_rolling50_min": gain("rolling50_min"),

        "paired_delta_mean": float(merged["delta_warm_minus_cold"].mean()),
        "paired_delta_median": float(merged["delta_warm_minus_cold"].median()),
        "warm_better_ratio": float((merged["delta_warm_minus_cold"] < 0).mean()),
        "paired_first100_delta": float(merged["delta_warm_minus_cold"].head(100).mean()),
        "paired_tail100_delta": float(merged["delta_warm_minus_cold"].tail(100).mean()),
        "paired_last50_delta": float(merged["delta_warm_minus_cold"].tail(50).mean()),
    }

    if row["gain_first100"] > 0 and row["gain_tail100"] >= -0.01 and row["gain_last50"] >= -0.01:
        label = "transfer_success"
    elif row["gain_first100"] < -0.01 or row["gain_tail100"] < -0.01 or row["gain_last50"] < -0.01:
        label = "negative_transfer"
    else:
        label = "neutral"

    row["label"] = label
    return row


def plot_eval_curve(pair_name: str, cold_name: str, warm_name: str, dfs: Dict[str, pd.DataFrame]) -> Path:
    cold_df = dfs[cold_name]
    warm_df = dfs[warm_name]
    cold_cost_col = metric_col(cold_df, "cost")
    warm_cost_col = metric_col(warm_df, "cost")

    cold_cost = pd.to_numeric(cold_df[cold_cost_col], errors="coerce").reset_index(drop=True)
    warm_cost = pd.to_numeric(warm_df[warm_cost_col], errors="coerce").reset_index(drop=True)
    x_cold = range(1, len(cold_cost) + 1)
    x_warm = range(1, len(warm_cost) + 1)

    plt.figure(figsize=(11, 6))
    plt.plot(x_cold, cold_cost, alpha=0.25, label=f"{cold_name} raw")
    plt.plot(x_warm, warm_cost, alpha=0.25, label=f"{warm_name} raw")
    plt.plot(x_cold, cold_cost.rolling(50, min_periods=50).mean(), linewidth=2, label=f"{cold_name} rolling50")
    plt.plot(x_warm, warm_cost.rolling(50, min_periods=50).mean(), linewidth=2, label=f"{warm_name} rolling50")
    plt.xlabel("Iteration")
    plt.ylabel("Eval_Cost")
    plt.title(f"{pair_name}: Cold vs Warm Eval_Cost")
    plt.legend()
    plt.tight_layout()

    out = FIGDIR / f"{pair_name}_eval_cost_curve.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_delta_curve(pair_name: str, merged: pd.DataFrame) -> Path:
    plt.figure(figsize=(11, 6))
    plt.plot(merged["_iter_key"], merged["delta_warm_minus_cold"], alpha=0.25, label="raw delta")
    plt.plot(merged["_iter_key"], merged["rolling50_delta"], linewidth=2, label="rolling50 delta")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Warm Eval_Cost - Cold Eval_Cost")
    plt.title(f"{pair_name}: Paired Delta Curve (below 0 means warm better)")
    plt.legend()
    plt.tight_layout()

    out = FIGDIR / f"{pair_name}_delta_curve.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_window_bar(summaries: pd.DataFrame) -> Path:
    runs = ["similar_cold", "similar_warm", "dissimilar_cold", "dissimilar_warm"]
    metrics = ["first100_eval_cost", "tail100_eval_cost", "last50_eval_cost"]
    labels = ["first100", "tail100", "last50"]

    data = summaries.set_index("run").loc[runs, metrics]

    plt.figure(figsize=(11, 6))
    ax = data.plot(kind="bar", figsize=(11, 6))
    ax.set_xticklabels(runs, rotation=20, ha="right")
    ax.set_ylabel("Eval_Cost")
    ax.set_title("Cold/Warm Window Metrics")
    ax.legend(labels)
    plt.tight_layout()

    out = FIGDIR / "first100_tail100_last50_bar.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_service_breakdown(summaries: pd.DataFrame) -> Path:
    # Normalize warm/cold service metrics inside each pair so different units can share one chart.
    rows = []
    service_metrics = [
        ("avg_delay_last50", "Delay last50"),
        ("avg_energy_last50", "Energy last50"),
        ("backlog_last50", "Backlog last50"),
        ("unfinished_last50", "Unfinished last50"),
    ]

    idx = summaries.set_index("run")

    for pair_name, (cold, warm) in PAIRS.items():
        for col, label in service_metrics:
            if col not in idx.columns:
                continue
            cold_val = idx.loc[cold, col]
            warm_val = idx.loc[warm, col]
            if pd.notna(cold_val) and cold_val != 0 and pd.notna(warm_val):
                rows.append({
                    "pair_metric": f"{pair_name}-{label}",
                    "cold": 1.0,
                    "warm": warm_val / cold_val,
                })

    if not rows:
        return Path("")

    df = pd.DataFrame(rows).set_index("pair_metric")

    plt.figure(figsize=(12, 6))
    ax = df.plot(kind="bar", figsize=(12, 6))
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Normalized value, cold = 1.0")
    ax.set_title("Service Breakdown, Warm vs Cold")
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    plt.tight_layout()

    out = FIGDIR / "service_breakdown_normalized_bar.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def write_report(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    selected_paths: Dict[str, Path],
    fig_paths: List[Path],
) -> Path:
    out = OUTDIR / "warm_start_transfer_report.md"

    lines = []
    lines.append("# Warm-start / Experience-sharing Transfer Report\n")
    lines.append("## Selected round_summary files\n")
    for name, path in selected_paths.items():
        lines.append(f"- **{name}**: `{path}`")
    lines.append("")

    lines.append("## Run summary\n")
    cols = [
        "run", "rows", "mean_eval_cost", "first100_eval_cost",
        "tail100_eval_cost", "last50_eval_cost", "rolling50_min",
        "rolling50_min_iter", "rebound_pct",
        "avg_delay_last50", "avg_energy_last50", "backlog_last50",
        "unfinished_last50",
        "warm_mode_unique_or_mean", "warm_loaded_unique_or_mean",
        "warm_selected_unique_or_mean", "local_selected_unique_or_mean",
    ]
    show_cols = [c for c in cols if c in summaries.columns]
    lines.append(summaries[show_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Pairwise comparison\n")
    comp_cols = [
        "pair", "label",
        "cold_first100", "warm_first100", "gain_first100",
        "cold_tail100", "warm_tail100", "gain_tail100",
        "cold_last50", "warm_last50", "gain_last50",
        "warm_better_ratio",
        "paired_delta_mean",
        "paired_first100_delta",
        "paired_tail100_delta",
        "paired_last50_delta",
    ]
    show_comp_cols = [c for c in comp_cols if c in comparisons.columns]
    lines.append(comparisons[show_comp_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Figures\n")
    for p in fig_paths:
        if p and str(p):
            lines.append(f"- `{p}`")
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- `gain > 0` means warm-start is better than cold-start.")
    lines.append("- `delta = warm - cold`; delta below 0 means warm-start is better.")
    lines.append("- Similar transfer is successful if first100 improves and tail100/last50 do not regress.")
    lines.append("- Dissimilar transfer is negative if warm-start worsens first100, tail100, or last50 by more than about 1%.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}
    selected_paths: Dict[str, Path] = {}

    print("Finding round_summary files...")
    for name, run_dir in RUNS.items():
        path = find_round_summary(run_dir)
        df = read_csv_safely(path)

        dfs[name] = df
        selected_paths[name] = path
        print(f"  {name}: rows={len(df)} path={path}")

    print("\nSummarizing runs...")
    summary_rows = []
    for name, df in dfs.items():
        summary_rows.append(summarize_run(name, df, selected_paths[name]))

    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(OUTDIR / "warm_start_run_summary.csv", index=False, encoding="utf-8-sig")

    print("\nComparing cold vs warm pairs...")
    comparison_rows = []
    merged_by_pair = {}

    for pair_name, (cold, warm) in PAIRS.items():
        merged, _ = align_pair(dfs[cold], dfs[warm])
        merged_by_pair[pair_name] = merged
        merged.to_csv(OUTDIR / f"{pair_name}_paired_delta.csv", index=False, encoding="utf-8-sig")
        comparison_rows.append(compare_pair(pair_name, cold, warm, dfs, summaries))

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(OUTDIR / "warm_start_pairwise_comparison.csv", index=False, encoding="utf-8-sig")

    print("\nGenerating figures...")
    fig_paths: List[Path] = []
    for pair_name, (cold, warm) in PAIRS.items():
        fig_paths.append(plot_eval_curve(pair_name, cold, warm, dfs))
        fig_paths.append(plot_delta_curve(pair_name, merged_by_pair[pair_name]))
    fig_paths.append(plot_window_bar(summaries))
    service_fig = plot_service_breakdown(summaries)
    if str(service_fig):
        fig_paths.append(service_fig)

    figures_index = FIGDIR / "figures_index.md"
    figures_index.write_text(
        "\n".join([f"- `{p}`" for p in fig_paths]),
        encoding="utf-8",
    )

    report = write_report(summaries, comparisons, selected_paths, fig_paths)

    print("\n=== Short summary ===")
    for _, row in comparisons.iterrows():
        print(
            f"{row['pair']}: "
            f"first100 cold={row['cold_first100']:.3f}, warm={row['warm_first100']:.3f}, "
            f"gain={row['gain_first100']*100:.2f}%; "
            f"tail100 cold={row['cold_tail100']:.3f}, warm={row['warm_tail100']:.3f}, "
            f"gain={row['gain_tail100']*100:.2f}%; "
            f"label={row['label']}"
        )

    print("\nOutputs:")
    print(f"  {OUTDIR / 'warm_start_run_summary.csv'}")
    print(f"  {OUTDIR / 'warm_start_pairwise_comparison.csv'}")
    print(f"  {report}")
    print(f"  {FIGDIR}")


if __name__ == "__main__":
    main()