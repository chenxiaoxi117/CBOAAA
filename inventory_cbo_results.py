#!/usr/bin/env python3
"""Inventory CBO/BO experiment result directories.

Run on the server from /home/ecs-user/CBO:
  python inventory_cbo_results.py result --output result/cbo_result_inventory.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


METHOD_PATTERNS = {
    "DIRECT_GREEDY_COST": "direct_greedy_cost",
    "DIRECT_QUEUE_AWARE_GREEDY": "direct_queue_aware_greedy",
    "REDUCED7_FIXED_MID": "reduced7_fixed_mid",
    "REDUCED7_FIXED_TUNED": "reduced7_fixed_tuned",
    "REDUCED6_FIXED_MID": "reduced6_fixed_mid",
    "REDUCED6_FIXED_TUNED": "reduced6_fixed_tuned",
    "REDUCED4_FIXED_MID": "reduced4_fixed_mid",
    "REDUCED4_FIXED_TUNED": "reduced4_fixed_tuned",
    "REDUCED9_FIXED_MID": "reduced9_fixed_mid",
    "REDUCED9_FIXED_TUNED": "reduced9_fixed_tuned",
    "A0_BO": "reduced7_bo_greedy",
    "B_BO_ADAPTIVE": "reduced7_bo_adaptive",
    "CBO_OLD": "reduced7_cbo_lite_pressure_taskmix_counts",
    "D6C_INTERNAL6_CONTEXT": "reduced7_cbo_lite_internal6_context",
    "D4C_INTERNAL4_CONTEXT": "reduced7_cbo_lite_internal4_context",
    "D4_INTERNAL4": "reduced7_cbo_lite_internal4",
}


def infer_seed(path: Path) -> str:
    for part in path.parts:
        m = re.search(r"_s(\d+)(?:\D*$|$)", part)
        if m:
            return m.group(1)
    return ""


def infer_root(result_root: Path, file_path: Path) -> Path:
    parts = file_path.relative_to(result_root).parts
    for i, part in enumerate(parts):
        if part == "lambda_1p80" or part == "lambda_2p60" or part == "lambda_3p00":
            prefix = parts[: max(0, i - 2)]
            return result_root / Path(*prefix) if prefix else result_root
    for i, part in enumerate(parts):
        if re.fullmatch(r"lambda_\d+p\d+", part) or re.fullmatch(r"lambda_\d+(?:\.\d+)?", part):
            prefix = parts[:i]
            return result_root / Path(*prefix) if prefix else result_root
    return file_path.parent


def infer_scenario(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    lam = next((p for p in rel.parts if p.startswith("lambda_")), "")
    scene = next((p for p in rel.parts if p.startswith("rt")), "")
    return f"{lam}/{scene}" if lam and scene else str(file_path.parent.relative_to(root))


def read_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def config_summary(root: Path) -> dict:
    cfgs = list(root.rglob("refactor_run_config.json"))
    if not cfgs:
        return {}
    cfg = read_config(cfgs[0])
    flat = {}

    def visit(prefix: str, value):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            flat[prefix] = value

    visit("", cfg)
    keys = [
        "bo_iterations",
        "bo_interval",
        "fixed_seed",
        "cbo_sigma_calibration_use_in_acq",
        "cbo_history_select_mode",
        "cbo_reference_source_method_key",
        "cbo_backlog_growth_penalty_weight",
    ]
    out = {}
    for want in keys:
        for k, v in flat.items():
            if k.endswith(want):
                out[want] = v
                break
    out["config_files"] = len(cfgs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path, nargs="?", default=Path("result"))
    parser.add_argument("--output", type=Path, default=Path("result/cbo_result_inventory.csv"))
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    by_root: dict[Path, dict] = defaultdict(lambda: defaultdict(set))
    unmatched_prefix_counts: dict[str, int] = defaultdict(int)
    round_files = list(result_root.rglob("*round_summary*.csv"))

    for path in round_files:
        name = path.name
        if name.startswith("round_summary_concat") or name.startswith("round_summary_completed"):
            continue
        method_label = None
        method_prefix = None
        for label, prefix in METHOD_PATTERNS.items():
            if name.startswith(prefix):
                method_label = label
                method_prefix = prefix
                break
        if method_label is None:
            unmatched_prefix = name.split("_round_summary", 1)[0]
            unmatched_prefix_counts[unmatched_prefix] += 1
            continue
        root = infer_root(result_root, path)
        scenario = infer_scenario(root, path)
        item = by_root[root]
        item["methods"].add(method_label)
        item[f"{method_label}_files"].add(str(path))
        item[f"{method_label}_scenarios"].add(scenario)
        item[f"{method_label}_prefix"].add(method_prefix)

    rows = []
    for root, data in sorted(by_root.items(), key=lambda kv: str(kv[0])):
        cfg = config_summary(root)
        row = {
            "root": str(root.relative_to(Path.cwd())) if root.is_relative_to(Path.cwd()) else str(root),
            "seed": infer_seed(root),
            "method_labels": ",".join(sorted(data["methods"])),
        }
        for label in METHOD_PATTERNS:
            row[f"{label}_round_files"] = len(data.get(f"{label}_files", set()))
            row[f"{label}_scenes"] = len(data.get(f"{label}_scenarios", set()))
        row.update(cfg)
        rows.append(row)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 240)
    print(f"round files scanned: {len(round_files)}")
    print(f"roots found: {len(df)}")
    print(f"output: {args.output.resolve()}")
    if unmatched_prefix_counts:
        print("\nunmatched round_summary method prefixes:")
        for prefix, count in sorted(unmatched_prefix_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
            print(f"  {count:5d}  {prefix}")
    if not df.empty:
        score_cols = [c for c in df.columns if c.endswith("_scenes")]
        show = df[["root", "seed", "method_labels", *score_cols]].copy()
        print(show.sort_values("root").to_string(index=False))


if __name__ == "__main__":
    main()
