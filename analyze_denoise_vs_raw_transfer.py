from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


RAW_ROOT = Path(r"D:\CBOv2\results\transfer_cbo_pressure_extended_overnight")
DENOISE_ROOT = Path(r"D:\CBOv2\results\transfer_cbo_pressure_denoise_R050_M2_CW03")
OUTDIR = DENOISE_ROOT / "analysis_compare_raw"

PAIRS = [
    "P1_RT_sim_RT60_to_RT50",
    "P2_RT_to_AI_neg_RT60_to_AI70",
]

SEEDS = [43, 44, 45]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, names: list[str], fuzzy: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if fuzzy:
        for n in names:
            key = n.lower()
            for c in df.columns:
                if key in str(c).lower():
                    return c
    return None


def cost_col(df: pd.DataFrame) -> str:
    c = find_col(df, ["Eval_Cost", "eval_cost", "BO_Training_Cost", "Cost", "cost"])
    if c is None:
        raise ValueError(f"Cannot find cost column. columns={list(df.columns)}")
    return c


def metric_col(df: pd.DataFrame, kind: str) -> str | None:
    mapping = {
        "delay": ["Avg_Delay", "avg_delay", "Delay", "delay"],
        "energy": ["Avg_Energy", "avg_energy", "Energy", "energy"],
        "backlog": ["Backlog", "Avg_Backlog", "backlog", "Backlog_End"],
        "unfinished": ["unfinished_end", "Unfinished_End", "unfinished", "Unfinished"],
        "violation": ["Violation", "violation", "Violation_Rate"],
        "denoise_ratio": ["cbo_history_denoise_smoothed_ratio"],
        "denoise_smoothed": ["cbo_history_denoise_smoothed_rows"],
        "denoise_raw": ["cbo_history_denoise_raw_rows"],
        "denoise_neighbor_mean": ["cbo_history_denoise_neighbor_count_mean"],
        "denoise_neighbor_max": ["cbo_history_denoise_neighbor_count_max"],
        "denoise_delta_mean": ["cbo_history_denoise_abs_delta_mean"],
        "denoise_delta_max": ["cbo_history_denoise_abs_delta_max"],
        "warm_loaded": ["cbo_warm_start_loaded_rows"],
        "warm_selected": ["selected_warm_rows_count"],
        "local_selected": ["selected_local_rows_count"],
    }
    return find_col(df, mapping[kind])


def find_run_dir(root: Path, seed: int, pair: str, kind: str) -> Path:
    base = root / "targets" / f"seed{seed}" / pair
    if not base.exists():
        raise FileNotFoundError(f"Pair dir not found: {base}")

    matches = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(f"{kind}_")]
    if not matches:
        raise FileNotFoundError(f"No {kind}_* dir under {base}")

    matches.sort(key=lambda p: len(str(p)))
    return matches[0]


def find_round_summary(run_dir: Path) -> Path:
    candidates = [
        p for p in run_dir.rglob("*.csv")
        if "round_summary" in p.name.lower()
        and "_short_export" not in str(p).lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No round_summary under {run_dir}")

    scored = []
    for p in candidates:
        try:
            df = read_csv(p)
            scored.append((abs(len(df) - 500), -len(df), len(str(p)), p))
        except Exception:
            continue

    if not scored:
        raise FileNotFoundError(f"No readable round_summary under {run_dir}")

    scored.sort()
    return scored[0][3]


def s_num(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def wmean(s: pd.Series, n: int, where: str) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return math.nan
    if where == "first":
        return float(s.head(n).mean())
    if where == "last":
        return float(s.tail(n).mean())
    raise ValueError(where)


def rolling_min_final(s: pd.Series, window: int = 50) -> tuple[float, int, float, float]:
    roll = pd.to_numeric(s, errors="coerce").rolling(window, min_periods=window).mean()
    valid = roll.dropna()
    if valid.empty:
        return math.nan, -1, math.nan, math.nan
    min_idx = int(valid.idxmin())
    rmin = float(roll.loc[min_idx])
    rfinal = float(valid.iloc[-1])
    rebound = (rfinal - rmin) / rmin if rmin else math.nan
    return rmin, min_idx + 1, rfinal, rebound


def summarize_run(root_name: str, pair: str, seed: int, kind: str, run_dir: Path, summary_path: Path) -> dict:
    df = read_csv(summary_path)
    c = cost_col(df)
    cost = pd.to_numeric(df[c], errors="coerce")
    rmin, rmin_iter, rfinal, rebound = rolling_min_final(cost)

    row = {
        "root": root_name,
        "pair": pair,
        "seed": seed,
        "kind": kind,
        "rows": len(df),
        "run_dir": str(run_dir),
        "round_summary": str(summary_path),
        "mean_cost": float(cost.mean()),
        "first50_cost": wmean(cost, 50, "first"),
        "first100_cost": wmean(cost, 100, "first"),
        "tail100_cost": wmean(cost, 100, "last"),
        "last50_cost": wmean(cost, 50, "last"),
        "rolling50_min": rmin,
        "rolling50_min_iter": rmin_iter,
        "rolling50_final": rfinal,
        "rebound_pct": rebound * 100 if pd.notna(rebound) else math.nan,
    }

    for key in ["delay", "energy", "backlog", "unfinished", "violation"]:
        col = metric_col(df, key)
        s = s_num(df, col)
        row[f"{key}_col"] = col or ""
        if s is None:
            row[f"{key}_mean"] = math.nan
            row[f"{key}_tail100"] = math.nan
            row[f"{key}_last50"] = math.nan
        else:
            row[f"{key}_mean"] = float(s.mean())
            row[f"{key}_tail100"] = wmean(s, 100, "last")
            row[f"{key}_last50"] = wmean(s, 50, "last")

    for key in [
        "denoise_ratio",
        "denoise_smoothed",
        "denoise_raw",
        "denoise_neighbor_mean",
        "denoise_neighbor_max",
        "denoise_delta_mean",
        "denoise_delta_max",
        "warm_loaded",
        "warm_selected",
        "local_selected",
    ]:
        col = metric_col(df, key)
        s = s_num(df, col)
        row[f"{key}_col"] = col or ""
        if s is None:
            row[f"{key}_mean"] = math.nan
            row[f"{key}_last50"] = math.nan
            row[f"{key}_max"] = math.nan
        else:
            row[f"{key}_mean"] = float(s.mean())
            row[f"{key}_last50"] = wmean(s, 50, "last")
            row[f"{key}_max"] = float(s.max())

    return row


def paired_compare(root_name: str, pair: str, seed: int, cold_path: Path, warm_path: Path) -> dict:
    cold_df = read_csv(cold_path)
    warm_df = read_csv(warm_path)
    cc = cost_col(cold_df)
    wc = cost_col(warm_df)

    n = min(len(cold_df), len(warm_df))
    cold = pd.to_numeric(cold_df[cc].iloc[:n], errors="coerce").reset_index(drop=True)
    warm = pd.to_numeric(warm_df[wc].iloc[:n], errors="coerce").reset_index(drop=True)
    delta = warm - cold

    def gain(cold_val: float, warm_val: float) -> float:
        return (cold_val - warm_val) / cold_val if cold_val else math.nan

    cold_mean = float(cold.mean())
    warm_mean = float(warm.mean())
    cold_first100 = wmean(cold, 100, "first")
    warm_first100 = wmean(warm, 100, "first")
    cold_tail100 = wmean(cold, 100, "last")
    warm_tail100 = wmean(warm, 100, "last")
    cold_last50 = wmean(cold, 50, "last")
    warm_last50 = wmean(warm, 50, "last")

    row = {
        "root": root_name,
        "pair": pair,
        "seed": seed,
        "paired_rows": n,
        "cold_mean": cold_mean,
        "warm_mean": warm_mean,
        "mean_gain": gain(cold_mean, warm_mean),
        "cold_first100": cold_first100,
        "warm_first100": warm_first100,
        "first100_gain": gain(cold_first100, warm_first100),
        "cold_tail100": cold_tail100,
        "warm_tail100": warm_tail100,
        "tail100_gain": gain(cold_tail100, warm_tail100),
        "cold_last50": cold_last50,
        "warm_last50": warm_last50,
        "last50_gain": gain(cold_last50, warm_last50),
        "warm_better_ratio": float((delta < 0).mean()),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_median": float(delta.median()),
        "paired_first100_delta": float(delta.head(100).mean()),
        "paired_tail100_delta": float(delta.tail(100).mean()),
        "paired_last50_delta": float(delta.tail(50).mean()),
    }

    for key in ["delay", "energy", "backlog", "unfinished", "violation"]:
        ccol = metric_col(cold_df, key)
        wcol = metric_col(warm_df, key)
        if ccol and wcol:
            cs = pd.to_numeric(cold_df[ccol].iloc[:n], errors="coerce").reset_index(drop=True)
            ws = pd.to_numeric(warm_df[wcol].iloc[:n], errors="coerce").reset_index(drop=True)
            d = ws - cs
            row[f"{key}_tail100_delta"] = float(d.tail(100).mean())
            row[f"{key}_last50_delta"] = float(d.tail(50).mean())
        else:
            row[f"{key}_tail100_delta"] = math.nan
            row[f"{key}_last50_delta"] = math.nan

    return row


def label_pair(row: pd.Series) -> str:
    if row["first100_gain_mean"] > 0 and row["tail100_gain_mean"] > 0 and row["last50_gain_mean"] > 0:
        return "positive_transfer"
    if row["tail100_gain_mean"] < -0.01 or row["last50_gain_mean"] < -0.01:
        return "negative_or_late_regression"
    if abs(row["first100_gain_mean"]) <= 0.01 and abs(row["tail100_gain_mean"]) <= 0.01:
        return "neutral"
    return "mixed"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    roots = {
        "raw": RAW_ROOT,
        "denoise": DENOISE_ROOT,
    }

    run_rows = []
    pair_rows = []

    selected = []

    for root_name, root in roots.items():
        for pair in PAIRS:
            for seed in SEEDS:
                cold_dir = find_run_dir(root, seed, pair, "cold")
                warm_dir = find_run_dir(root, seed, pair, "warm")
                cold_summary = find_round_summary(cold_dir)
                warm_summary = find_round_summary(warm_dir)

                selected.append({
                    "root": root_name,
                    "pair": pair,
                    "seed": seed,
                    "cold_dir": str(cold_dir),
                    "warm_dir": str(warm_dir),
                    "cold_summary": str(cold_summary),
                    "warm_summary": str(warm_summary),
                })

                run_rows.append(summarize_run(root_name, pair, seed, "cold", cold_dir, cold_summary))
                run_rows.append(summarize_run(root_name, pair, seed, "warm", warm_dir, warm_summary))
                pair_rows.append(paired_compare(root_name, pair, seed, cold_summary, warm_summary))

    selected_df = pd.DataFrame(selected)
    run_df = pd.DataFrame(run_rows)
    pair_df = pd.DataFrame(pair_rows)

    agg = (
        pair_df
        .groupby(["root", "pair"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            first100_gain_mean=("first100_gain", "mean"),
            first100_gain_std=("first100_gain", "std"),
            tail100_gain_mean=("tail100_gain", "mean"),
            tail100_gain_std=("tail100_gain", "std"),
            last50_gain_mean=("last50_gain", "mean"),
            last50_gain_std=("last50_gain", "std"),
            mean_gain_mean=("mean_gain", "mean"),
            warm_better_ratio_mean=("warm_better_ratio", "mean"),
            paired_delta_mean=("paired_delta_mean", "mean"),
            paired_first100_delta_mean=("paired_first100_delta", "mean"),
            paired_tail100_delta_mean=("paired_tail100_delta", "mean"),
            paired_last50_delta_mean=("paired_last50_delta", "mean"),
            delay_tail100_delta_mean=("delay_tail100_delta", "mean"),
            delay_last50_delta_mean=("delay_last50_delta", "mean"),
            energy_tail100_delta_mean=("energy_tail100_delta", "mean"),
            energy_last50_delta_mean=("energy_last50_delta", "mean"),
            backlog_tail100_delta_mean=("backlog_tail100_delta", "mean"),
            backlog_last50_delta_mean=("backlog_last50_delta", "mean"),
            unfinished_tail100_delta_mean=("unfinished_tail100_delta", "mean"),
            unfinished_last50_delta_mean=("unfinished_last50_delta", "mean"),
        )
    )

    agg["label"] = agg.apply(label_pair, axis=1)

    raw_agg = agg[agg["root"] == "raw"].set_index("pair")
    den_agg = agg[agg["root"] == "denoise"].set_index("pair")

    compare_rows = []
    for pair in PAIRS:
        r = raw_agg.loc[pair]
        d = den_agg.loc[pair]
        compare_rows.append({
            "pair": pair,
            "raw_first100_gain": r["first100_gain_mean"],
            "denoise_first100_gain": d["first100_gain_mean"],
            "delta_first100_gain": d["first100_gain_mean"] - r["first100_gain_mean"],
            "raw_tail100_gain": r["tail100_gain_mean"],
            "denoise_tail100_gain": d["tail100_gain_mean"],
            "delta_tail100_gain": d["tail100_gain_mean"] - r["tail100_gain_mean"],
            "raw_last50_gain": r["last50_gain_mean"],
            "denoise_last50_gain": d["last50_gain_mean"],
            "delta_last50_gain": d["last50_gain_mean"] - r["last50_gain_mean"],
            "raw_warm_better_ratio": r["warm_better_ratio_mean"],
            "denoise_warm_better_ratio": d["warm_better_ratio_mean"],
            "delta_warm_better_ratio": d["warm_better_ratio_mean"] - r["warm_better_ratio_mean"],
            "raw_label": r["label"],
            "denoise_label": d["label"],
        })

    compare_df = pd.DataFrame(compare_rows)

    # Denoise diagnostics summary, only denoise root.
    denoise_diag_cols = [
        "root", "pair", "seed", "kind",
        "denoise_ratio_mean", "denoise_ratio_last50", "denoise_ratio_max",
        "denoise_neighbor_mean_mean", "denoise_neighbor_mean_last50",
        "denoise_neighbor_max_mean", "denoise_neighbor_max_max",
        "denoise_delta_mean_mean", "denoise_delta_mean_last50",
        "denoise_delta_max_mean", "denoise_delta_max_max",
        "warm_loaded_mean", "warm_selected_mean", "local_selected_mean",
    ]
    denoise_diag_cols = [c for c in denoise_diag_cols if c in run_df.columns]
    denoise_diag = run_df[run_df["root"] == "denoise"][denoise_diag_cols].copy()

    selected_df.to_csv(OUTDIR / "selected_files.csv", index=False, encoding="utf-8-sig")
    run_df.to_csv(OUTDIR / "raw_vs_denoise_run_summary.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(OUTDIR / "raw_vs_denoise_pair_seed_comparison.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUTDIR / "raw_vs_denoise_pair_aggregate_summary.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUTDIR / "denoise_vs_raw_gain_delta.csv", index=False, encoding="utf-8-sig")
    denoise_diag.to_csv(OUTDIR / "denoise_diagnostics_summary.csv", index=False, encoding="utf-8-sig")

    report = OUTDIR / "denoise_vs_raw_transfer_report.md"
    lines = []
    lines.append("# Denoise vs Raw Warm-start Transfer Report\n")
    lines.append("## Aggregate transfer summary\n")
    show_cols = [
        "root", "pair", "n_seeds",
        "first100_gain_mean", "tail100_gain_mean", "last50_gain_mean",
        "warm_better_ratio_mean", "label",
    ]
    lines.append(agg[show_cols].to_markdown(index=False))
    lines.append("\n## Denoise gain delta relative to raw\n")
    lines.append(compare_df.to_markdown(index=False))
    lines.append("\n## Denoise diagnostics\n")
    if not denoise_diag.empty:
        lines.append(denoise_diag.head(20).to_markdown(index=False))
    else:
        lines.append("No denoise diagnostics found.")
    lines.append("\n## Reading guide\n")
    lines.append("- gain > 0 means warm-start is better than cold-start.")
    lines.append("- delta_gain > 0 means denoise improves warm-start transfer compared with raw.")
    lines.append("- For P1, ideal result is first100 gain staying positive while tail100/last50 improve versus raw.")
    lines.append("- For P2, ideal result is less negative tail100/last50; if still negative, context gating is still required.")
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Aggregate summary ===")
    print(agg[show_cols].to_string(index=False))

    print("\n=== Denoise vs raw gain delta ===")
    print(compare_df.to_string(index=False))

    print("\nOutputs:")
    print(OUTDIR)
    print(report)


if __name__ == "__main__":
    main()