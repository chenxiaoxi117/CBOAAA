#!/usr/bin/env python3
"""Analyze V11 A/B/C sigma ablations by convergence segment and uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("V11-A_off", "V11-B_diag", "V11-C_soft")
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


def find_col(df: pd.DataFrame, exact: str | None = None, prefix: str | None = None):
    if exact and exact in df.columns:
        return exact
    if prefix:
        for col in df.columns:
            if str(col).startswith(prefix):
                return col
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def finite_quantile(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else np.nan


def load_entries(root: Path):
    entries = []
    for path in root.rglob("*round_summary*.csv"):
        rel = path.relative_to(root)
        if len(rel.parts) < 3 or rel.parts[0] not in VARIANTS:
            continue
        name = path.name
        if name.startswith("reduced7_cbo_lite_pressure_taskmix_counts"):
            method = "CBO"
        elif name.startswith("reduced7_bo_greedy"):
            method = "BO"
        else:
            continue
        df = pd.read_csv(path, low_memory=False)
        entries.append({
            "variant": rel.parts[0],
            "scene": rel.parts[1],
            "method": method,
            "path": path,
            "df": df,
        })
    return entries


def scene_thresholds(entries):
    out = {}
    scenes = sorted({e["scene"] for e in entries if e["method"] == "CBO"})
    for scene in scenes:
        backlog, unfinished = [], []
        for entry in entries:
            if entry["method"] != "CBO" or entry["scene"] != scene:
                continue
            df = entry["df"]
            backlog.extend(numeric(df, find_col(df, prefix="Backlog_")).tolist())
            unfinished.extend(numeric(df, find_col(df, exact="unfinished_rate")).tolist())
        out[scene] = {
            "backlog": finite_quantile(backlog, 0.90),
            "unfinished": max(0.05, finite_quantile(unfinished, 0.90)) if np.isfinite(finite_quantile(unfinished, 0.90)) else 0.05,
        }
    return out


def segment_rows(entries, thresholds):
    rows = []
    for entry in entries:
        df = entry["df"]
        eval_cost = numeric(df, find_col(df, prefix="Eval_Cost_"))
        norm = numeric(df, find_col(df, exact="normalized_tradeoff_score"))
        backlog = numeric(df, find_col(df, prefix="Backlog_"))
        unfinished = numeric(df, find_col(df, exact="unfinished_rate"))
        for label, start, end in SEGMENTS:
            sl = slice(start - 1, min(end, len(df)))
            cost_seg = eval_cost.iloc[sl]
            norm_seg = norm.iloc[sl]
            backlog_seg = backlog.iloc[sl]
            unfinished_seg = unfinished.iloc[sl]
            diffs = np.diff(cost_seg.to_numpy(dtype=float))
            positive = diffs[np.isfinite(diffs) & (diffs > 0)]
            th = thresholds.get(entry["scene"], {"backlog": np.nan, "unfinished": 0.05})
            rows.append({
                "variant": entry["variant"],
                "scene": entry["scene"],
                "method": entry["method"],
                "segment": label,
                "start": start,
                "end": end,
                "sample_count": int(min(end, len(df)) - start + 1) if len(df) >= start else 0,
                "eval_cost_mean": finite_mean(cost_seg),
                "normalized_tradeoff_mean": finite_mean(norm_seg),
                "positive_step_rebound_mean": finite_mean(positive),
                "positive_step_rebound_count": int(len(positive)),
                "backlog_high_threshold": th["backlog"],
                "high_backlog_count": int(np.sum(backlog_seg > th["backlog"])) if np.isfinite(th["backlog"]) else 0,
                "unfinished_high_threshold": th["unfinished"],
                "high_unfinished_count": int(np.sum(unfinished_seg > th["unfinished"])),
            })
    return pd.DataFrame(rows)


def uncertainty_rows(entries):
    rows = []
    for entry in entries:
        if entry["method"] != "CBO":
            continue
        df = entry["df"]
        raw_z = numeric(df, find_col(df, exact="raw_surprise"))
        cal_z = numeric(df, find_col(df, exact="calibrated_surprise"))
        scale = numeric(df, find_col(df, exact="sigma_scale"))
        raw_sigma = numeric(df, find_col(df, exact="raw_sigma"))
        cal_sigma = numeric(df, find_col(df, exact="sigma_calibrated"))
        acq_sigma = numeric(df, find_col(df, exact="sigma_acq"))
        for label, start, end in SEGMENTS:
            sl = slice(start - 1, min(end, len(df)))
            rz, cz = raw_z.iloc[sl], cal_z.iloc[sl]
            rs, cs, acs = raw_sigma.iloc[sl], cal_sigma.iloc[sl], acq_sigma.iloc[sl]
            scale_seg = scale.iloc[sl]
            valid_raw = rz[np.isfinite(rz)]
            valid_cal = cz[np.isfinite(cz)]
            ratio_acq = acs / rs.replace(0, np.nan)
            ratio_cal = cs / rs.replace(0, np.nan)
            rows.append({
                "variant": entry["variant"],
                "scene": entry["scene"],
                "segment": label,
                "raw_surprise_mean": finite_mean(valid_raw),
                "raw_surprise_abs_mean": finite_mean(np.abs(valid_raw)),
                "calibrated_surprise_mean": finite_mean(valid_cal),
                "calibrated_surprise_abs_mean": finite_mean(np.abs(valid_cal)),
                "raw_2sigma_exceed_rate": finite_mean(np.abs(valid_raw) > 2.0),
                "calibrated_2sigma_exceed_rate": finite_mean(np.abs(valid_cal) > 2.0),
                "sigma_scale_mean": finite_mean(scale_seg),
                "sigma_scale_p95": finite_quantile(scale_seg, 0.95),
                "sigma_acq_over_raw_mean": finite_mean(ratio_acq),
                "sigma_acq_over_raw_p95": finite_quantile(ratio_acq, 0.95),
                "sigma_calibrated_over_raw_mean": finite_mean(ratio_cal),
                "sigma_calibrated_over_raw_p95": finite_quantile(ratio_cal, 0.95),
            })
    return pd.DataFrame(rows)


def pairwise_rows(segment_df: pd.DataFrame):
    metrics = ("eval_cost_mean", "normalized_tradeoff_mean", "positive_step_rebound_mean", "high_backlog_count", "high_unfinished_count")
    cbo = segment_df[segment_df["method"] == "CBO"]
    rows = []
    for variant in VARIANTS[1:]:
        for segment in [s[0] for s in SEGMENTS]:
            a = cbo[(cbo.variant == VARIANTS[0]) & (cbo.segment == segment)].set_index("scene")
            b = cbo[(cbo.variant == variant) & (cbo.segment == segment)].set_index("scene")
            common = a.index.intersection(b.index)
            for metric in metrics:
                av = pd.to_numeric(a.loc[common, metric], errors="coerce")
                bv = pd.to_numeric(b.loc[common, metric], errors="coerce")
                delta = bv - av
                rel = delta / av.abs().replace(0, np.nan)
                rows.append({
                    "variant_vs_A": variant,
                    "segment": segment,
                    "metric": metric,
                    "scene_count": int(len(common)),
                    "wins_lower_is_better": int(np.sum(delta < 0)),
                    "mean_delta": finite_mean(delta),
                    "median_delta": finite_quantile(delta, 0.50),
                    "mean_relative_pct": 100.0 * finite_mean(rel),
                })
    return pd.DataFrame(rows)


def main():
    global VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--variants", nargs=3, default=list(VARIANTS), metavar=("A", "B", "C"), help="Three variant directory names; A is the pairwise baseline")
    args = parser.parse_args()
    VARIANTS = tuple(args.variants)

    root = args.result_root.resolve()
    output = (args.output or (root / "analysis_sigma_abc")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = load_entries(root)
    if not entries:
        raise SystemExit(f"No round summary CSV files found under {root}")

    thresholds = scene_thresholds(entries)
    segments = segment_rows(entries, thresholds)
    uncertainty = uncertainty_rows(entries)
    pairwise = pairwise_rows(segments)

    segments.to_csv(output / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    uncertainty.to_csv(output / "uncertainty_metrics.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output / "pairwise_vs_A.csv", index=False, encoding="utf-8-sig")

    print(f"round files: {len(entries)}")
    print(f"output: {output}")
    print(pairwise[pairwise["metric"].isin(["eval_cost_mean", "normalized_tradeoff_mean"])].to_string(index=False))


if __name__ == "__main__":
    main()
