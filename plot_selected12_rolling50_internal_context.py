#!/usr/bin/env python3
"""Plot selected12 rolling-50 curves for internal-context CBO ablations.

The script accepts variant specs in the same NAME=ROOT=METHOD style used by
analyze_static108_abcd_matched.py. It plots per-scene rolling means and an
overall selected12 average for normalized_tradeoff_score by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_PREFIXES = {
    "BO": "reduced7_bo_greedy",
    "BO_ADAPTIVE": "reduced7_bo_adaptive",
    "CBO": "reduced7_cbo_lite_pressure_taskmix_counts",
    "CBO_I4": "reduced7_cbo_lite_internal4",
    "CBO_I6CTX": "reduced7_cbo_lite_internal6_context",
    "CBO_I4CTX": "reduced7_cbo_lite_internal4_context",
}

DEFAULT_VARIANTS = [
    "A0=result/static108_v11_sigma_calibrated_s43=BO",
    "B=result/static108_v16_bc_s43=BO_ADAPTIVE",
    "D0=result/static108_v16_bc_s43=CBO",
    "D=result/static108_v15_adaptive_core_s43=CBO",
    "D4=result/static_v17_internal_context_selected12_newmods_s43=CBO_I4",
    "D6C=result/static_v17_internal_context_selected12_newmods_s43=CBO_I6CTX",
    "D4C=result/static_v17_internal_context_selected12_newmods_s43=CBO_I4CTX",
]

LABELS = {
    "A0": "BO",
    "B": "BO adaptive",
    "D0": "CBO old",
    "C0": "CBO old",
    "D": "CBO old + adaptive",
    "D4": "CBO internal4",
    "D6C": "CBO internal6+ctx",
    "D4C": "CBO internal4+ctx",
    "N4": "CBO internal4",
    "N6C": "CBO internal6+ctx",
    "N4C": "CBO internal4+ctx",
    "A4": "CBO internal4 + adaptive",
    "A6C": "CBO internal6+ctx + adaptive",
    "A4C": "CBO internal4+ctx + adaptive",
}

COLORS = {
    "B": "#1f77b4",
    "C0": "#ff7f0e",
    "D0": "#ff7f0e",
    "D": "#2ca02c",
    "D4": "#9467bd",
    "D6C": "#8c564b",
    "D4C": "#d62728",
    "N4": "#9467bd",
    "N6C": "#8c564b",
    "N4C": "#d62728",
    "A4": "#17becf",
    "A6C": "#bcbd22",
    "A4C": "#d62728",
    "A0": "#7f7f7f",
}

HIGHLIGHT_VARIANTS = {"D4C", "A4C"}


def parse_variant(spec: str) -> tuple[str, Path, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Variant must be NAME=ROOT=METHOD")
    name, root, method = parts
    return name, Path(root), METHOD_PREFIXES.get(method, method)


def load_scene_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    scenes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            scenes.add(item)
    return scenes


def find_metric_col(df: pd.DataFrame, metric: str) -> str:
    if metric in df.columns:
        return metric
    lower = {str(c).lower(): c for c in df.columns}
    hit = lower.get(metric.lower())
    if hit is not None:
        return hit
    metric_l = metric.lower()
    for col in df.columns:
        if str(col).lower().startswith(metric_l):
            return col
    raise KeyError(f"Metric column not found: {metric}")


def infer_scene(root: Path, path: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    lambda_tag = next((p for p in rel.parts if p.startswith("lambda_")), "")
    scene = next((p for p in rel.parts if p.startswith("rt")), "")
    scenario = f"{lambda_tag}/{scene}" if lambda_tag and scene else str(rel.parent)
    return lambda_tag, scene, scenario


def load_variant(name: str, root: Path, prefix: str, metric: str, scene_filter: set[str] | None) -> pd.DataFrame:
    frames = []
    # Match the exact method prefix. For example, internal4 must not also
    # pick up internal4_context files.
    for path in sorted(root.rglob(f"{prefix}_round_summary*.csv")):
        lambda_tag, scene, scenario = infer_scene(root, path)
        if scene_filter is not None and scenario not in scene_filter:
            continue
        df = pd.read_csv(path, low_memory=False)
        col = find_metric_col(df, metric)
        values = pd.to_numeric(df[col], errors="coerce")
        frame = pd.DataFrame(
            {
                "variant": name,
                "lambda_tag": lambda_tag,
                "scene": scene,
                "scenario": scenario,
                "iteration": np.arange(1, len(df) + 1),
                "value": values.to_numpy(dtype=float),
                "path": str(path),
            }
        )
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No rows loaded for {name}: root={root}, prefix={prefix}")
    return pd.concat(frames, ignore_index=True)


def scene_sort_key(scenario: str) -> tuple[float, int, int, int, str]:
    m = re.search(r"lambda_([0-9p.]+)/rt(\d+)_batch(\d+)_ai(\d+)", scenario)
    if not m:
        return (999.0, 999, 999, 999, scenario)
    lam = float(m.group(1).replace("p", "."))
    return (lam, int(m.group(2)), int(m.group(3)), int(m.group(4)), scenario)


def scene_title(scenario: str) -> str:
    m = re.search(r"lambda_([0-9p.]+)/rt(\d+)_batch(\d+)_ai(\d+)", scenario)
    if not m:
        return scenario
    lam = m.group(1).replace("p", ".")
    return f"lambda={lam}, RT={m.group(2)}, Batch={m.group(3)}, AI={m.group(4)}"


def plot_scene(group: pd.DataFrame, scenario: str, output: Path, rolling: int, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=140)
    for variant, vdf in group.groupby("variant", sort=False):
        vdf = vdf.sort_values("iteration")
        curve = vdf["value"].rolling(rolling, min_periods=max(5, rolling // 5)).mean()
        ax.plot(
            vdf["iteration"],
            curve,
            label=LABELS.get(variant, variant),
            linewidth=2.4 if variant in HIGHLIGHT_VARIANTS else 1.7,
            color=COLORS.get(variant),
            alpha=1.0 if variant in HIGHLIGHT_VARIANTS else 0.82,
        )
    ax.set_title(f"Rolling{rolling} {metric} - {scene_title(scenario)}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Rolling{rolling} mean")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    safe = scenario.replace("/", "__").replace(",", "_").replace("=", "_")
    fig.savefig(output / f"{safe}_rolling{rolling}_{metric}.png")
    plt.close(fig)


def plot_overall(data: pd.DataFrame, output: Path, rolling: int, metric: str) -> None:
    scene_count = int(data["scenario"].nunique())
    if scene_count == 108:
        scope = "full108"
    elif scene_count == 12:
        scope = "selected12"
    else:
        scope = f"{scene_count}scenes"
    avg = (
        data.groupby(["variant", "iteration"], as_index=False)["value"]
        .mean()
        .sort_values(["variant", "iteration"])
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=150)
    for variant, vdf in avg.groupby("variant", sort=False):
        curve = vdf["value"].rolling(rolling, min_periods=max(5, rolling // 5)).mean()
        ax.plot(
            vdf["iteration"],
            curve,
            label=LABELS.get(variant, variant),
            linewidth=2.8 if variant in HIGHLIGHT_VARIANTS else 1.9,
            color=COLORS.get(variant),
            alpha=1.0 if variant in HIGHLIGHT_VARIANTS else 0.82,
        )
    ax.set_title(f"{scope} average rolling{rolling} {metric}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Rolling{rolling} mean")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    primary = output / f"{scope}_average_rolling{rolling}_{metric}.png"
    fig.savefig(primary)
    if scope != "overall":
        fig.savefig(output / f"overall_average_rolling{rolling}_{metric}.png")
    plt.close(fig)


def save_summary(data: pd.DataFrame, output: Path, rolling: int) -> None:
    rows = []
    for (scenario, variant), group in data.groupby(["scenario", "variant"], sort=False):
        group = group.sort_values("iteration")
        roll = group["value"].rolling(rolling, min_periods=max(5, rolling // 5)).mean()
        rows.append(
            {
                "scenario": scenario,
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "rolling_last": float(roll.iloc[-1]),
                "raw_all_mean": float(group["value"].mean()),
                "raw_last50_mean": float(group["value"].tail(50).mean()),
                "raw_first50_mean": float(group["value"].head(50).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output / "rolling50_curve_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, default=None, help="Optional scenario list. Omit for all scenes.")
    parser.add_argument("--output", type=Path, default=Path("result/static_v17_selected12_rolling50_figs"))
    parser.add_argument("--metric", default="normalized_tradeoff_score")
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument("--variant", action="append", type=parse_variant, default=[])
    args = parser.parse_args()

    variants = args.variant or [parse_variant(spec) for spec in DEFAULT_VARIANTS]
    scene_filter = load_scene_filter(args.scene_list if args.scene_list and args.scene_list.exists() else None)
    frames = []
    for name, root, prefix in variants:
        frames.append(load_variant(name, root, prefix, args.metric, scene_filter))
    data = pd.concat(frames, ignore_index=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "selected12_rolling_input_long.csv", index=False, encoding="utf-8-sig")
    save_summary(data, output, args.rolling)
    plot_overall(data, output, args.rolling, args.metric)
    for scenario in sorted(data["scenario"].unique(), key=scene_sort_key):
        plot_scene(data[data["scenario"] == scenario], scenario, output, args.rolling, args.metric)

    print(f"variants: {', '.join([name for name, _, _ in variants])}")
    print(f"scenarios: {data['scenario'].nunique()}")
    print(f"rows: {len(data)}")
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
