#!/usr/bin/env python3
"""Matched static108 A/B/C/D ablation analyzer.

Variants:
  A = BO fixed-beta / no adaptive
  B = BO adaptive / no context
  C = CBO internal6 + external gate / no adaptive
  D = CBO internal6 + external gate + adaptive

Each variant is loaded from an explicit root+method pair, avoiding ambiguity
when different roots contain the same method label.
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

METHOD_PREFIXES = {
    "BO": "reduced7_bo_greedy",
    "BO_ADAPTIVE": "reduced7_bo_adaptive",
    "CBO": "reduced7_cbo_lite_pressure_taskmix_counts",
    "CBO_I4": "reduced7_cbo_lite_internal4",
    "CBO_I6CTX": "reduced7_cbo_lite_internal6_context",
    "CBO_I4CTX": "reduced7_cbo_lite_internal4_context",
}

METRICS = {
    "eval_cost": ("Eval_Cost_", "eval_cost"),
    "normalized_tradeoff": ("normalized_tradeoff_score",),
    "backlog": ("Backlog_", "backlog"),
    "unfinished_rate": ("unfinished_rate", "window_unfinished_rate"),
    "avg_delay": ("avg_delay", "Avg_Delay_"),
    "energy_per_arrival": ("window_energy_per_arrival", "energy_per_arrival"),
    "energy_norm": ("window_energy_norm", "energy_norm"),
    "rt_violation_rate": ("window_rt_violation_rate", "rt_violation_rate"),
    "effective_violation_rate": ("effective_vio_rate", "violation_rate", "deadline_violation_rate"),
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
    m = re.search(r"_s(\d+)(?:\D*$|$)", root.name)
    return m.group(1) if m else root.name


def load_scene_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    scenes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            scenes.add(item)
    return scenes


def find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    for cand in candidates:
        cand_l = cand.lower()
        for col in cols:
            if str(col).lower().startswith(cand_l):
                return col
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_variant_rows(variant: str, root: Path, method_label: str, scene_filter: set[str] | None) -> pd.DataFrame:
    prefix = METHOD_PREFIXES.get(method_label, method_label)
    rows = []
    seed = infer_seed(root)
    # Match the exact method prefix. A loose glob such as
    # ``reduced7_cbo_lite_internal4*`` also matches
    # ``reduced7_cbo_lite_internal4_context`` and silently mixes variants.
    for path in root.rglob(f"{prefix}_round_summary*.csv"):
        rel = path.relative_to(root)
        lambda_tag = next((part for part in rel.parts if part.startswith("lambda_")), rel.parts[0])
        scene = next((part for part in rel.parts if part.startswith("rt")), rel.parts[1] if len(rel.parts) > 1 else "")
        scenario = f"{lambda_tag}/{scene}"
        if scene_filter is not None and scenario not in scene_filter:
            continue
        df = pd.read_csv(path, low_memory=False)
        col_map = {name: find_col(df, candidates) for name, candidates in METRICS.items()}
        series = {name: numeric(df, col) for name, col in col_map.items()}
        for segment, start, end in SEGMENTS:
            if len(df) < start:
                sl = slice(0, 0)
                n = 0
            else:
                stop = min(end, len(df))
                sl = slice(start - 1, stop)
                n = stop - start + 1
            row = {
                "variant": variant,
                "method": method_label,
                "seed": seed,
                "lambda_tag": lambda_tag,
                "scene": scene,
                "scenario": scenario,
                "segment": segment,
                "sample_count": n,
                "path": str(path),
            }
            for metric, values in series.items():
                row[f"{metric}_mean"] = finite_mean(values.iloc[sl])
            rows.append(row)
    return pd.DataFrame(rows)


def make_pairs(data: pd.DataFrame, baseline: str, compare: str) -> pd.DataFrame:
    idx = ["seed", "scenario", "segment"]
    base = data[data["variant"] == baseline].drop_duplicates(idx).set_index(idx)
    comp = data[data["variant"] == compare].drop_duplicates(idx).set_index(idx)
    common = base.index.intersection(comp.index)
    metric_cols = [c for c in data.columns if c.endswith("_mean")]
    rows = []
    for key in common:
        seed, scenario, segment = key
        lambda_tag, scene = scenario.split("/", 1) if "/" in scenario else ("", scenario)
        row = {"baseline": baseline, "compare": compare, "seed": seed, "lambda_tag": lambda_tag, "scene": scene, "scenario": scenario, "segment": segment}
        for metric in metric_cols:
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
    for (baseline, compare, segment), group in pairs.groupby(["baseline", "compare", "segment"], sort=False):
        for delta_col in delta_cols:
            metric = delta_col.removeprefix("delta_")
            delta = pd.to_numeric(group[delta_col], errors="coerce")
            valid = delta[np.isfinite(delta)]
            if valid.empty:
                continue
            rel_col = "relative_pct_" + metric
            win_col = "compare_win_" + metric
            rel = pd.to_numeric(group[rel_col], errors="coerce") if rel_col in group.columns else pd.Series(np.nan, index=group.index)
            rows.append({
                "comparison": f"{compare} - {baseline}",
                "baseline": baseline,
                "compare": compare,
                "segment": segment,
                "metric": metric.removesuffix("_mean"),
                "pair_count": int(valid.shape[0]),
                "unique_scene_count": int(group.loc[valid.index, "scenario"].nunique()),
                "compare_wins_lower_is_better": int(group.loc[valid.index, win_col].sum()) if win_col in group.columns else int((valid < 0).sum()),
                "mean_delta_compare_minus_baseline": finite_mean(valid),
                "median_delta_compare_minus_baseline": finite_quantile(valid, 0.50),
                "mean_relative_pct": finite_mean(rel.loc[valid.index]),
                "median_relative_pct": finite_quantile(rel.loc[valid.index], 0.50),
            })
    return pd.DataFrame(rows)


def parse_variant_spec(spec: str) -> tuple[str, Path, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Variant spec must be NAME=ROOT=METHOD, e.g. A=result/...=BO")
    name, root, method = parts
    return name, Path(root), method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", type=parse_variant_spec, required=True, help="NAME=ROOT=METHOD")
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", action="append", default=[], help="BASE:COMPARE, e.g. A:B")
    args = parser.parse_args()

    scene_filter = load_scene_filter(args.scene_list.resolve() if args.scene_list else None)
    frames = []
    for name, root, method in args.variant:
        df = load_variant_rows(name, root.resolve(), method, scene_filter)
        if df.empty:
            raise SystemExit(f"No rows loaded for variant {name} from {root} method={method}")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    comparisons = args.comparison or ["A:B", "A:C", "A:D", "B:D", "C:D", "B:C"]
    pair_frames = []
    for spec in comparisons:
        base, comp = spec.split(":", 1)
        pair_frames.append(make_pairs(data, base, comp))
    pairs = pd.concat(pair_frames, ignore_index=True)
    summary = summarize_pairs(pairs)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "abcd_segment_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "abcd_pairs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "abcd_summary.csv", index=False, encoding="utf-8-sig")

    print(f"variants loaded: {', '.join(sorted(data['variant'].unique()))}")
    if scene_filter is not None:
        print(f"scene filter loaded: {len(scene_filter)}")
    print(f"scenarios loaded: {data['scenario'].nunique()}")
    print(f"paired rows: {len(pairs)}")
    print(f"output: {output}")
    focus = summary[summary["metric"].isin(["eval_cost", "normalized_tradeoff", "backlog", "unfinished_rate", "avg_delay", "energy_norm"])]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
