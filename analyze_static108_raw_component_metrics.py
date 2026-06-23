#!/usr/bin/env python3
"""Analyze raw component metrics for static108 BO/CBO runs.

Unlike analyze_static108_metric_breakdown.py, this reads the original
round_summary CSV files so delay, energy, and violation fields are not limited
by what was previously copied into the pairs file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENTS = (
    ("001-050", 1, 50),
    ("051-100", 51, 100),
    ("101-200", 101, 200),
    ("201-300", 201, 300),
    ("301-400", 301, 400),
    ("401-500", 401, 500),
    ("451-500", 451, 500),
    ("all500", 1, 500),
)

METHOD_LABELS = {
    "reduced7_bo_greedy": "BO",
    "reduced7_bo_adaptive": "BO_ADAPTIVE",
    "reduced7_cbo_lite_pressure_taskmix_counts": "CBO",
}

RAW_METRICS = {
    "eval_cost": ["Eval_Cost_", "eval_cost", "cost"],
    "normalized_tradeoff": ["normalized_tradeoff_score"],
    "backlog": ["Backlog_", "backlog"],
    "unfinished_rate": ["unfinished_rate", "window_unfinished_rate"],
    "avg_delay": ["avg_delay", "Avg_Delay_", "average_delay"],
    "effective_avg_delay": ["effective_avg_delay", "window_censored_avg_delay"],
    "delay_deadline_norm": ["window_delay_deadline_norm", "delay_norm"],
    "avg_energy": ["avg_energy", "energy_per_task"],
    "energy_per_arrival": ["window_energy_per_arrival", "energy_per_arrival"],
    "energy_norm": ["window_energy_norm", "energy_norm"],
    "total_energy": ["total_energy"],
    "rt_violation_rate": ["window_rt_violation_rate", "rt_violation_rate", "RT_vio_rate", "rt_vio_rate"],
    "effective_violation_rate": ["effective_vio_rate", "vio_rate", "violation_rate", "deadline_violation_rate"],
    "rt_avg_delay": ["rt_avg_delay", "avg_delay_rt"],
    "batch_avg_delay": ["batch_avg_delay", "avg_delay_batch"],
    "ai_avg_delay": ["ai_avg_delay", "avg_delay_ai"],
    "rt_vio_rate": ["rt_vio_rate"],
    "batch_vio_rate": ["batch_vio_rate"],
    "ai_vio_rate": ["ai_vio_rate"],
    "rt_avg_energy": ["avg_energy_rt", "rt_avg_energy"],
    "batch_avg_energy": ["avg_energy_batch", "batch_avg_energy"],
    "ai_avg_energy": ["avg_energy_ai", "ai_avg_energy"],
}


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def infer_seed(root: Path) -> str:
    match = re.search(r"_s(\d+)(?:\D*$|$)", root.name)
    return match.group(1) if match else root.name


def infer_method(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    for prefix, label in METHOD_LABELS.items():
        if name.startswith(prefix):
            return prefix, label
    return None, None


def find_metric_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        c = lower_map.get(str(cand).lower())
        if c is not None:
            return c
    for cand in candidates:
        cand_l = str(cand).lower()
        for col in cols:
            if str(col).lower().startswith(cand_l):
                return col
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_entries(roots: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for root in roots:
        seed = infer_seed(root)
        for path in root.rglob("*round_summary*.csv"):
            method_key, method = infer_method(path)
            if method is None:
                continue
            rel = path.relative_to(root)
            lambda_tag = next((part for part in rel.parts if part.startswith("lambda_")), rel.parts[0])
            scene = next((part for part in rel.parts if part.startswith("rt")), rel.parts[1] if len(rel.parts) > 1 else "")
            entries.append({
                "seed": seed,
                "lambda_tag": lambda_tag,
                "scene": scene,
                "scenario": f"{lambda_tag}/{scene}",
                "method_key": method_key,
                "method": method,
                "path": path,
            })
    return entries


def segment_rows(entries: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detected = []
    for entry in entries:
        df = pd.read_csv(entry["path"], low_memory=False)
        col_map = {metric: find_metric_col(df, candidates) for metric, candidates in RAW_METRICS.items()}
        detected.append({
            "seed": entry["seed"],
            "scenario": entry["scenario"],
            "method": entry["method"],
            **{f"{k}_col": v for k, v in col_map.items()},
        })
        series = {metric: numeric(df, col) for metric, col in col_map.items()}
        for label, start, end in SEGMENTS:
            if len(df) < start:
                sl = slice(0, 0)
                sample_count = 0
            else:
                stop = min(end, len(df))
                sl = slice(start - 1, stop)
                sample_count = stop - start + 1
            row = {
                "seed": entry["seed"],
                "lambda_tag": entry["lambda_tag"],
                "scene": entry["scene"],
                "scenario": entry["scenario"],
                "method": entry["method"],
                "method_key": entry["method_key"],
                "segment": label,
                "sample_count": sample_count,
            }
            for metric, values in series.items():
                row[f"{metric}_mean"] = finite_mean(values.iloc[sl])
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(detected)


def pair_methods(segment_df: pd.DataFrame, baseline: str, compare: str) -> pd.DataFrame:
    index_cols = ["seed", "scenario", "segment"]
    base = segment_df[segment_df["method"] == baseline].set_index(index_cols)
    comp = segment_df[segment_df["method"] == compare].set_index(index_cols)
    common = base.index.intersection(comp.index)
    metric_cols = [c for c in segment_df.columns if c.endswith("_mean") and c not in {"sample_count"}]
    rows = []
    for key in common:
        seed, scenario, segment = key
        lambda_tag, scene = scenario.split("/", 1) if "/" in scenario else ("", scenario)
        row = {"seed": seed, "lambda_tag": lambda_tag, "scene": scene, "scenario": scenario, "segment": segment}
        for metric in metric_cols:
            if metric not in base.columns or metric not in comp.columns:
                continue
            b = pd.to_numeric(pd.Series([base.loc[key, metric]]), errors="coerce").iloc[0]
            c = pd.to_numeric(pd.Series([comp.loc[key, metric]]), errors="coerce").iloc[0]
            delta = c - b
            row[f"baseline_{metric}"] = b
            row[f"compare_{metric}"] = c
            row[f"delta_{metric}"] = delta
            row[f"relative_pct_{metric}"] = 100.0 * delta / abs(b) if np.isfinite(b) and abs(b) > 1e-12 else np.nan
            row[f"compare_win_{metric}"] = bool(np.isfinite(delta) and delta < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    delta_cols = [c for c in pairs.columns if c.startswith("delta_")]
    for segment, group in pairs.groupby("segment", sort=False):
        for delta_col in delta_cols:
            metric = delta_col.removeprefix("delta_")
            rel_col = f"relative_pct_{metric}"
            win_col = f"compare_win_{metric}"
            delta = pd.to_numeric(group[delta_col], errors="coerce")
            valid = delta[np.isfinite(delta)]
            if valid.empty:
                continue
            rel = pd.to_numeric(group[rel_col], errors="coerce") if rel_col in group.columns else pd.Series(np.nan, index=group.index)
            rows.append({
                "segment": segment,
                "metric": metric.removesuffix("_mean"),
                "pair_count": int(valid.shape[0]),
                "compare_better_count": int(group.loc[valid.index, win_col].sum()) if win_col in group.columns else int((valid < 0).sum()),
                "compare_worse_count": int((valid > 0).sum()),
                "mean_delta_compare_minus_baseline": finite_mean(valid),
                "median_delta_compare_minus_baseline": finite_median(valid),
                "mean_relative_pct": finite_mean(rel.loc[valid.index]),
                "median_relative_pct": finite_median(rel.loc[valid.index]),
            })
    return pd.DataFrame(rows)


def worst_scenes(pairs: pd.DataFrame, topn: int) -> pd.DataFrame:
    rows = []
    delta_cols = [c for c in pairs.columns if c.startswith("delta_")]
    for (scenario, segment), group in pairs.groupby(["scenario", "segment"], sort=False):
        for delta_col in delta_cols:
            metric = delta_col.removeprefix("delta_").removesuffix("_mean")
            delta = finite_mean(pd.to_numeric(group[delta_col], errors="coerce"))
            rel_col = "relative_pct_" + delta_col.removeprefix("delta_")
            rel = finite_mean(pd.to_numeric(group[rel_col], errors="coerce")) if rel_col in group.columns else np.nan
            rows.append({"scenario": scenario, "segment": segment, "metric": metric, "mean_delta": delta, "mean_relative_pct": rel})
    df = pd.DataFrame(rows)
    focus = [
        "eval_cost", "avg_delay", "effective_avg_delay", "delay_deadline_norm",
        "avg_energy", "energy_per_arrival", "energy_norm", "rt_violation_rate",
        "effective_violation_rate", "backlog", "unfinished_rate",
    ]
    out = []
    for segment in ["001-050", "451-500", "all500"]:
        for metric in focus:
            sub = df[(df["segment"] == segment) & (df["metric"] == metric)].dropna(subset=["mean_delta"])
            if sub.empty:
                continue
            out.append(sub.sort_values("mean_delta", ascending=False).head(topn))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_roots", nargs="+", type=Path)
    parser.add_argument("--baseline", default="BO")
    parser.add_argument("--compare", default="CBO")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--topn", type=int, default=15)
    args = parser.parse_args()

    roots = [p.resolve() for p in args.result_roots]
    output = (args.output or (roots[0].parent / "static108_raw_component_analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    entries = load_entries(roots)
    if not entries:
        raise SystemExit("No supported round_summary files found.")

    segments, detected = segment_rows(entries)
    pairs = pair_methods(segments, baseline=str(args.baseline), compare=str(args.compare))
    summary = summarize_pairs(pairs)
    worst = worst_scenes(pairs, topn=int(args.topn))

    detected.to_csv(output / "detected_metric_columns.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(output / "raw_component_segment_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / f"{str(args.compare).lower()}_vs_{str(args.baseline).lower()}_raw_component_pairs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / f"{str(args.compare).lower()}_vs_{str(args.baseline).lower()}_raw_component_by_segment.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(output / f"{str(args.compare).lower()}_vs_{str(args.baseline).lower()}_raw_component_worst_scenes.csv", index=False, encoding="utf-8-sig")

    print(f"round files loaded: {len(entries)}")
    print(f"paired rows: {len(pairs)}")
    print(f"comparison: {args.compare} - {args.baseline}")
    print(f"output: {output}")
    focus = summary[summary["metric"].isin([
        "eval_cost", "normalized_tradeoff", "avg_delay", "effective_avg_delay",
        "delay_deadline_norm", "avg_energy", "energy_per_arrival", "energy_norm",
        "rt_violation_rate", "effective_violation_rate", "backlog", "unfinished_rate",
    ])]
    print(focus.to_string(index=False))
    missing_counts = detected.filter(like="_col").isna().sum().sort_values(ascending=False)
    print("\nMissing detected columns:")
    print(missing_counts.to_string())


if __name__ == "__main__":
    main()
