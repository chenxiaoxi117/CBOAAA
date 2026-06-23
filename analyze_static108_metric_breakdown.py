#!/usr/bin/env python3
"""Break down static108 CBO-vs-BO differences by raw metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "eval_cost_mean": "Eval_Cost",
    "normalized_tradeoff_mean": "NormScore",
    "backlog_mean": "Backlog",
    "unfinished_rate_mean": "UnfinishedRate",
    "deadline_violation_mean": "DeadlineViolation",
    "avg_latency_mean": "AvgLatency",
    "energy_cost_mean": "EnergyCost",
}


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def finite_median(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def load_pairs(path: Path) -> pd.DataFrame:
    if path.is_dir():
        candidates = sorted(path.glob("*pairs.csv"))
        preferred = path / "cbo_vs_bo_pairs.csv"
        if preferred.exists():
            return pd.read_csv(preferred, low_memory=False)
        if not candidates:
            raise SystemExit(f"No *pairs.csv found in {path}")
        return pd.read_csv(candidates[0], low_memory=False)
    return pd.read_csv(path, low_memory=False)


def summarize_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, group in df.groupby("segment", sort=False):
        for metric, label in METRIC_LABELS.items():
            delta_col = f"delta_{metric}"
            rel_col = f"relative_pct_{metric}"
            base_col = f"baseline_{metric}"
            comp_col = f"compare_{metric}"
            if delta_col not in group.columns:
                continue
            delta = pd.to_numeric(group[delta_col], errors="coerce")
            rel = pd.to_numeric(
                group[rel_col] if rel_col in group.columns else pd.Series(np.nan, index=group.index),
                errors="coerce",
            )
            base = pd.to_numeric(
                group[base_col] if base_col in group.columns else pd.Series(np.nan, index=group.index),
                errors="coerce",
            )
            comp = pd.to_numeric(
                group[comp_col] if comp_col in group.columns else pd.Series(np.nan, index=group.index),
                errors="coerce",
            )
            valid = delta[np.isfinite(delta)]
            if valid.empty:
                continue
            rows.append({
                "segment": segment,
                "metric": label,
                "pair_count": int(valid.shape[0]),
                "cbo_better_count": int((valid < 0).sum()),
                "cbo_worse_count": int((valid > 0).sum()),
                "bo_mean": finite_mean(base.loc[valid.index]),
                "cbo_mean": finite_mean(comp.loc[valid.index]),
                "mean_delta_cbo_minus_bo": finite_mean(valid),
                "median_delta_cbo_minus_bo": finite_median(valid),
                "mean_relative_pct": finite_mean(rel.loc[valid.index]),
                "median_relative_pct": finite_median(rel.loc[valid.index]),
            })
    return pd.DataFrame(rows)


def summarize_worst_scenes(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    rows = []
    for (scenario, segment), group in df.groupby(["scenario", "segment"], sort=False):
        row = {"scenario": scenario, "segment": segment}
        for metric, label in METRIC_LABELS.items():
            delta_col = f"delta_{metric}"
            rel_col = f"relative_pct_{metric}"
            if delta_col not in group.columns:
                continue
            delta = pd.to_numeric(group[delta_col], errors="coerce")
            rel = pd.to_numeric(
                group[rel_col] if rel_col in group.columns else pd.Series(np.nan, index=group.index),
                errors="coerce",
            )
            row[f"{label}_delta"] = finite_mean(delta)
            row[f"{label}_relative_pct"] = finite_mean(rel)
        rows.append(row)
    scene_df = pd.DataFrame(rows)
    out = []
    for segment in ["001-050", "451-500", "all500"]:
        sub = scene_df[scene_df["segment"] == segment].copy()
        for label in ["Backlog", "UnfinishedRate", "DeadlineViolation", "AvgLatency", "EnergyCost", "Eval_Cost"]:
            col = f"{label}_delta"
            if col not in sub.columns or sub[col].dropna().empty:
                continue
            tmp = sub.sort_values(col, ascending=False).head(topn).copy()
            tmp.insert(0, "rank_metric", label)
            out.append(tmp[["rank_metric", "segment", "scenario", col, f"{label}_relative_pct"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir_or_pairs_csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--topn", type=int, default=15)
    args = parser.parse_args()

    src = args.analysis_dir_or_pairs_csv.resolve()
    output = args.output or (src if src.is_dir() else src.parent)
    output.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(src)
    segment = summarize_by_segment(pairs)
    worst = summarize_worst_scenes(pairs, args.topn)

    segment.to_csv(output / "raw_metric_breakdown_by_segment.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(output / "raw_metric_worst_scenes.csv", index=False, encoding="utf-8-sig")

    print(f"output: {output}")
    print("\n=== Raw metric breakdown by segment ===")
    print(segment.to_string(index=False))
    if not worst.empty:
        print("\n=== Worst raw-metric scenes ===")
        print(worst.head(args.topn * 6).to_string(index=False))


if __name__ == "__main__":
    main()
