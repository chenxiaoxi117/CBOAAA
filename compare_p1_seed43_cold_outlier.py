from pathlib import Path
import math
import pandas as pd


RUNS = {
    "raw_cold": Path(
        r"D:\CBOv2\results\transfer_cbo_pressure_extended_overnight\targets\seed43\P1_RT_sim_RT60_to_RT50\cold_lam2p6_RT50_Batch40_AI10"
    ),
    "local_median_cold": Path(
        r"D:\CBOv2\results\transfer_cbo_pressure_denoise_R050_M2_CW03\targets\seed43\P1_RT_sim_RT60_to_RT50\cold_lam2p6_RT50_Batch40_AI10"
    ),
    "outlier_filter_cold": Path(
        r"D:\CBOv2\results\transfer_cbo_pressure_outlier_filter_seed43\targets\seed43\P1_RT_sim_RT60_to_RT50\cold_lam2p6_RT50_Batch40_AI10"
    ),
}

OUTDIR = Path(r"D:\CBOv2\results\transfer_cbo_pressure_outlier_filter_seed43\analysis_quick")
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


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


def find_col(df: pd.DataFrame, names, fuzzy=True):
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


def num(df, col):
    if col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def mean_window(s: pd.Series, n: int, part: str):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return math.nan
    if part == "first":
        return float(s.head(n).mean())
    if part == "last":
        return float(s.tail(n).mean())
    raise ValueError(part)


def rolling_info(s: pd.Series, window=50):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rolling(window, min_periods=window).mean()
    valid = r.dropna()
    if valid.empty:
        return math.nan, math.nan, math.nan
    idx = valid.idxmin()
    rmin = float(r.loc[idx])
    rfinal = float(valid.iloc[-1])
    rebound = (rfinal - rmin) / rmin * 100 if rmin else math.nan
    return rmin, idx + 1, rebound


def summarize(name: str, path: Path):
    df = read_csv(path)
    cost_col = find_col(df, ["Eval_Cost", "eval_cost", "BO_Training_Cost", "Cost"])
    if cost_col is None:
        raise ValueError(f"{name}: cannot find cost column")

    cost = pd.to_numeric(df[cost_col], errors="coerce")
    rmin, rmin_iter, rebound = rolling_info(cost)

    row = {
        "run": name,
        "rows": len(df),
        "path": str(path),
        "mean_cost": float(cost.mean()),
        "first50": mean_window(cost, 50, "first"),
        "first100": mean_window(cost, 100, "first"),
        "tail100": mean_window(cost, 100, "last"),
        "last50": mean_window(cost, 50, "last"),
        "rolling50_min": rmin,
        "rolling50_min_iter": rmin_iter,
        "rebound_pct": rebound,
    }

    metrics = {
        "delay": ["Avg_Delay", "avg_delay", "Delay"],
        "energy": ["Avg_Energy", "avg_energy", "Energy"],
        "backlog": ["Backlog", "Avg_Backlog", "backlog"],
        "unfinished": ["unfinished_end", "Unfinished_End", "unfinished"],
        "violation": ["Violation", "violation", "Violation_Rate"],
        "filtered_rows": ["cbo_history_outlier_filtered_rows"],
        "filter_ratio": ["cbo_history_outlier_filter_ratio"],
        "residual_max": ["cbo_history_outlier_residual_max"],
        "neighbor_mean": ["cbo_history_outlier_neighbor_count_mean"],
        "neighbor_max": ["cbo_history_outlier_neighbor_count_max"],
    }

    for key, names in metrics.items():
        col = find_col(df, names)
        s = num(df, col)
        row[f"{key}_col"] = col or ""
        if s is None:
            row[f"{key}_mean"] = math.nan
            row[f"{key}_last50"] = math.nan
            row[f"{key}_max"] = math.nan
        else:
            row[f"{key}_mean"] = float(s.mean())
            row[f"{key}_last50"] = mean_window(s, 50, "last")
            row[f"{key}_max"] = float(s.max())

    return row


rows = []
selected = {}

for name, run_dir in RUNS.items():
    path = find_round_summary(run_dir)
    selected[name] = path
    rows.append(summarize(name, path))

summary = pd.DataFrame(rows)
summary.to_csv(OUTDIR / "p1_seed43_cold_raw_median_outlier_summary.csv", index=False, encoding="utf-8-sig")

base = summary.set_index("run")

compare_rows = []
for target in ["local_median_cold", "outlier_filter_cold"]:
    for metric in ["mean_cost", "first50", "first100", "tail100", "last50", "rolling50_min"]:
        raw_val = base.loc["raw_cold", metric]
        tgt_val = base.loc[target, metric]
        gain = (raw_val - tgt_val) / raw_val if raw_val else math.nan
        compare_rows.append({
            "target": target,
            "metric": metric,
            "raw": raw_val,
            "target_value": tgt_val,
            "gain_vs_raw": gain,
            "gain_vs_raw_pct": gain * 100 if pd.notna(gain) else math.nan,
        })

compare = pd.DataFrame(compare_rows)
compare.to_csv(OUTDIR / "p1_seed43_cold_gain_vs_raw.csv", index=False, encoding="utf-8-sig")

print("\nSelected files:")
for k, p in selected.items():
    print(f"{k}: {p}")

print("\nSummary:")
print(summary[[
    "run", "rows", "mean_cost", "first50", "first100",
    "tail100", "last50", "rolling50_min", "rolling50_min_iter",
    "rebound_pct", "delay_mean", "energy_mean", "backlog_mean",
    "unfinished_mean", "filter_ratio_mean", "filter_ratio_max",
    "filtered_rows_mean", "filtered_rows_max", "residual_max_max"
]].to_string(index=False))

print("\nGain vs raw:")
print(compare.to_string(index=False))

print("\nOutput:")
print(OUTDIR)