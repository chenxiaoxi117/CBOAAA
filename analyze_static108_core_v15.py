#!/usr/bin/env python3
"""Aggregate static108 V15 BO/CBO results across seeds.

The script expects one or more result roots produced by
run_static108_v15_adaptive.sh with METHOD_SET=core.
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


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def finite_quantile(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else np.nan


def infer_seed(root: Path) -> str:
    match = re.search(r"_s(\d+)(?:\D*$|$)", root.name)
    return match.group(1) if match else root.name


def infer_method(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    for prefix, label in METHOD_LABELS.items():
        if name.startswith(prefix):
            return prefix, label
    return None, None


def find_col(df: pd.DataFrame, exact: str | None = None, prefixes: tuple[str, ...] = ()) -> str | None:
    if exact and exact in df.columns:
        return exact
    for prefix in prefixes:
        for col in df.columns:
            if str(col).startswith(prefix):
                return col
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def metric_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "eval_cost": find_col(df, prefixes=("Eval_Cost_", "eval_cost")),
        "normalized_tradeoff": find_col(df, exact="normalized_tradeoff_score"),
        "backlog": find_col(df, prefixes=("Backlog_", "backlog")),
        "unfinished_rate": find_col(df, exact="unfinished_rate", prefixes=("unfinished_rate",)),
        "deadline_violation": find_col(
            df,
            exact="deadline_violation_rate",
            prefixes=("Deadline_Violation_", "deadline_violation", "deadline_miss", "violation_rate"),
        ),
        "avg_latency": find_col(df, prefixes=("Avg_Latency_", "avg_latency", "Latency_")),
        "energy_cost": find_col(df, prefixes=("Energy_Cost_", "energy_cost", "Energy_")),
    }


def load_entries(roots: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for root in roots:
        root = root.resolve()
        seed = infer_seed(root)
        for path in root.rglob("*round_summary*.csv"):
            method_key, method = infer_method(path)
            if method is None:
                continue
            rel = path.relative_to(root)
            if len(rel.parts) < 2:
                continue
            lambda_tag = next((part for part in rel.parts if part.startswith("lambda_")), rel.parts[0])
            scene = next((part for part in rel.parts if part.startswith("rt")), rel.parts[1])
            df = pd.read_csv(path, low_memory=False)
            entries.append(
                {
                    "root": root,
                    "seed": seed,
                    "lambda_tag": lambda_tag,
                    "scene": scene,
                    "scenario": f"{lambda_tag}/{scene}",
                    "method_key": method_key,
                    "method": method,
                    "path": path,
                    "df": df,
                }
            )
    return entries


def load_scene_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    scenes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            scenes.add(item)
    return scenes


def build_segment_metrics(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        df = entry["df"]
        cols = metric_columns(df)
        series = {name: numeric(df, col) for name, col in cols.items()}
        for label, start, end in SEGMENTS:
            if len(df) < start:
                sample_count = 0
                sl = slice(0, 0)
            else:
                stop = min(end, len(df))
                sample_count = stop - start + 1
                sl = slice(start - 1, stop)
            row = {
                "seed": entry["seed"],
                "lambda_tag": entry["lambda_tag"],
                "scene": entry["scene"],
                "scenario": entry["scenario"],
                "method": entry["method"],
                "method_key": entry["method_key"],
                "segment": label,
                "start": start,
                "end": end,
                "sample_count": sample_count,
                "path": str(entry["path"]),
            }
            for name, values in series.items():
                row[f"{name}_mean"] = finite_mean(values.iloc[sl])
            rows.append(row)
    return pd.DataFrame(rows)


def pair_methods(segment_df: pd.DataFrame, baseline: str, compare: str) -> pd.DataFrame:
    metrics = [
        "eval_cost_mean",
        "normalized_tradeoff_mean",
        "backlog_mean",
        "unfinished_rate_mean",
        "deadline_violation_mean",
        "avg_latency_mean",
        "energy_cost_mean",
    ]
    index_cols = ["seed", "scenario", "segment"]
    base = segment_df[segment_df["method"] == baseline].set_index(index_cols)
    comp = segment_df[segment_df["method"] == compare].set_index(index_cols)
    common = base.index.intersection(comp.index)
    rows = []
    for key in common:
        seed, scenario, segment = key
        lambda_tag, scene = scenario.split("/", 1) if "/" in scenario else ("", scenario)
        row = {
            "seed": seed,
            "lambda_tag": lambda_tag,
            "scene": scene,
            "scenario": scenario,
            "segment": segment,
        }
        for metric in metrics:
            if metric not in base.columns or metric not in comp.columns:
                continue
            base_v = pd.to_numeric(pd.Series([base.loc[key, metric]]), errors="coerce").iloc[0]
            comp_v = pd.to_numeric(pd.Series([comp.loc[key, metric]]), errors="coerce").iloc[0]
            delta = comp_v - base_v
            rel = delta / abs(base_v) if np.isfinite(base_v) and base_v != 0 else np.nan
            row[f"baseline_{metric}"] = base_v
            row[f"compare_{metric}"] = comp_v
            row[f"delta_{metric}"] = delta
            row[f"relative_pct_{metric}"] = 100.0 * rel
            row[f"compare_win_{metric}"] = bool(np.isfinite(delta) and delta < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "eval_cost_mean",
        "normalized_tradeoff_mean",
        "backlog_mean",
        "unfinished_rate_mean",
        "deadline_violation_mean",
        "avg_latency_mean",
        "energy_cost_mean",
    ]
    rows = []
    for segment, group in pair_df.groupby("segment", sort=False):
        for metric in metrics:
            delta_col = f"delta_{metric}"
            rel_col = f"relative_pct_{metric}"
            win_col = f"compare_win_{metric}"
            if delta_col not in group.columns:
                continue
            deltas = pd.to_numeric(group[delta_col], errors="coerce")
            rels = pd.to_numeric(group[rel_col], errors="coerce")
            valid = deltas[np.isfinite(deltas)]
            if valid.empty:
                continue
            rows.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "pair_count": int(valid.shape[0]),
                    "scene_seed_count": int(valid.shape[0]),
                    "unique_scene_count": int(group.loc[valid.index, "scenario"].nunique()),
                    "seed_count": int(group.loc[valid.index, "seed"].nunique()),
                    "compare_wins_lower_is_better": int(group.loc[valid.index, win_col].sum()),
                    "mean_delta_compare_minus_baseline": finite_mean(valid),
                    "median_delta_compare_minus_baseline": finite_quantile(valid, 0.50),
                    "mean_relative_pct": finite_mean(rels.loc[valid.index]),
                    "median_relative_pct": finite_quantile(rels.loc[valid.index], 0.50),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_scene(pair_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ("eval_cost_mean", "normalized_tradeoff_mean")
    rows = []
    for (scenario, segment), group in pair_df.groupby(["scenario", "segment"], sort=False):
        lambda_tag, scene = scenario.split("/", 1) if "/" in scenario else ("", scenario)
        row = {
            "lambda_tag": lambda_tag,
            "scene": scene,
            "scenario": scenario,
            "segment": segment,
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in metrics:
            delta_col = f"delta_{metric}"
            rel_col = f"relative_pct_{metric}"
            win_col = f"compare_win_{metric}"
            if delta_col not in group.columns:
                continue
            deltas = pd.to_numeric(group[delta_col], errors="coerce")
            row[f"{metric}_compare_wins"] = int(group.loc[deltas.index, win_col].sum())
            row[f"{metric}_mean_delta"] = finite_mean(deltas)
            row[f"{metric}_mean_relative_pct"] = finite_mean(pd.to_numeric(group[rel_col], errors="coerce"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", default="BO", help="Baseline method label, e.g. BO or BO_ADAPTIVE")
    parser.add_argument("--compare", default="CBO", help="Compared method label, e.g. CBO")
    parser.add_argument("--scene-list", type=Path, default=None, help="Optional newline-separated scenario filter, e.g. lambda_1p8/rt10_batch10_ai80")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.result_roots]
    output = args.output
    if output is None:
        output = roots[0].parent / "static108_v15_core_analysis"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    entries = load_entries(roots)
    scene_filter = load_scene_filter(args.scene_list.resolve() if args.scene_list else None)
    if scene_filter is not None:
        entries = [e for e in entries if e["scenario"] in scene_filter]
    if not entries:
        raise SystemExit("No BO/CBO round summary files found.")

    segments = build_segment_metrics(entries)
    pairs = pair_methods(segments, baseline=str(args.baseline), compare=str(args.compare))
    summary = summarize_pairs(pairs)
    by_scene = summarize_by_scene(pairs)

    segments.to_csv(output / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    pair_tag = f"{str(args.compare).lower()}_vs_{str(args.baseline).lower()}"
    pairs.to_csv(output / f"{pair_tag}_pairs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / f"{pair_tag}_by_segment.csv", index=False, encoding="utf-8-sig")
    by_scene.to_csv(output / f"{pair_tag}_by_scene.csv", index=False, encoding="utf-8-sig")

    print(f"round files loaded: {len(entries)}")
    if scene_filter is not None:
        print(f"scene filter loaded: {len(scene_filter)}")
        print(f"scenarios after filter: {len(set(e['scenario'] for e in entries))}")
    print(f"paired rows: {len(pairs)}")
    print(f"comparison: {args.compare} - {args.baseline}")
    print(f"output: {output}")
    focus = summary[summary["metric"].isin(["eval_cost_mean", "normalized_tradeoff_mean"])]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
