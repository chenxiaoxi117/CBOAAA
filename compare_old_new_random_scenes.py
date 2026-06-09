from pathlib import Path
import re
import random
import json
import numpy as np
import pandas as pd

# ============================================================
# 1. 路径配置
# ============================================================

OLD_ROOT = Path(r"D:\CBO\v6_3lambda_36_context500_results_107_plus_timeout\v6_3lambda_36_context500")

NEW_ROOT = Path(r"D:\CBOv2\results\新原始107静态数据\results\full_ratio_sweep_compare_107_seed43")

OUT = Path(r"C:\Users\POPchen\Desktop\duiibi")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260601
RANDOM_SCENE_N = 12
ROLLING = 50

# 如果你后面想指定固定场景，就填这里；留空就是随机抽
MANUAL_SCENES = [
    # "lam2p6_RT60_B30_AI10",
    # "lam3p0_RT70_B20_AI10",
]

# ============================================================
# 2. 列名候选
# ============================================================

METHOD_COL_CANDIDATES = [
    "Group_Key_方法键",
    "method",
    "Method",
    "method_key",
]

ITER_COL_CANDIDATES = [
    "Iteration_轮次",
    "Iteration",
    "iter",
    "round",
]

METRIC_CANDIDATES = {
    "cost": [
        "Eval_Cost_最终评估Cost",
        "Eval_Cost",
        "Cost",
    ],
    "bo_training_cost": [
        "BO_Training_Cost_BO训练Cost",
        "BO_Training_Cost",
        "bo_training_cost",
    ],
    "delay": [
        "Avg_Delay_平均时延",
        "Average_Delay_平均时延",
        "Mean_Delay_平均时延",
        "Avg_Delay",
        "Average_Delay",
        "Mean_Delay",
        "Delay",
        "Latency",
    ],
    "energy": [
        "Avg_Energy_平均能耗",
        "Average_Energy_平均能耗",
        "Mean_Energy_平均能耗",
        "Avg_Energy",
        "Average_Energy",
        "Mean_Energy",
        "Energy",
        "Total_Energy_总能耗",
        "Total_Energy",
    ],
    "backlog": [
        "Backlog",
        "Avg_Backlog",
        "Backlog_积压",
        "Queue_Backlog",
        "unfinished_end",
        "Unfinished_End",
    ],
    "rt_delay": [
        "RT_Delay",
        "RT_Avg_Delay",
        "Avg_RT_Delay",
        "avg_delay_rt",
        "RT_Delay_平均时延",
    ],
    "batch_delay": [
        "Batch_Delay",
        "Batch_Avg_Delay",
        "Avg_Batch_Delay",
        "avg_delay_batch",
        "Batch_Delay_平均时延",
    ],
    "ai_delay": [
        "AI_Delay",
        "AI_Avg_Delay",
        "Avg_AI_Delay",
        "avg_delay_ai",
        "AI_Delay_平均时延",
    ],
}

# 只是用于输出时好看，不影响识别
LABEL_MAP = {
    "reduced6_fixed_mid": "Fixed-mid",
    "reduced6_fixed_tuned": "Fixed-tuned",
    "reduced6_fixed_risk_high": "Fixed-risk-high",
    "reduced6_fixed_queue_high": "Fixed-queue-high",
    "reduced6_fixed_edge_safe": "Fixed-edge-safe",
    "direct_least_load": "LeastLoad-direct",
    "direct_greedy_cost": "Greedy-direct",
    "direct_queue_aware_greedy": "QueueAwareGreedy-direct",
    "direct_round_robin": "RoundRobin-direct",
    "reduced6_bo_greedy": "BO-greedy",
    "reduced6_cbo_lite_pressure_only": "CBO-pressure",
    "reduced6_cbo_lite_pressure_prev_unfinished": "CBO-prev-unfinished",
    "reduced6_cbo_lite_pressure_taskmix_counts": "CBO-taskmix-counts",
    "reduced6_cbo_lite_full_taskmix_counts": "CBO-full",
}

FIXED_KEYS = {
    "reduced6_fixed_mid",
    "reduced6_fixed_tuned",
    "reduced6_fixed_risk_high",
    "reduced6_fixed_queue_high",
    "reduced6_fixed_edge_safe",
}

CBO_KEY_HINTS = [
    "cbo",
    "CBO",
]

# ============================================================
# 3. 工具函数
# ============================================================

def method_label(method_key: str) -> str:
    return LABEL_MAP.get(method_key, method_key)


def pick_col(df: pd.DataFrame, candidates, loose_keys=None):
    for c in candidates:
        if c in df.columns:
            return c

    if loose_keys:
        for c in df.columns:
            low = str(c).lower()
            if any(str(k).lower() in low for k in loose_keys):
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().sum() > 0:
                    return c

    return None


def parse_scene_from_path(path: Path):
    s = str(path)
    m = re.search(r"lam(?P<lam>\d+p\d+)_RT(?P<rt>\d+)_(?:B|Batch)(?P<b>\d+)_AI(?P<ai>\d+)", s)
    if not m:
        return None

    lam = float(m.group("lam").replace("p", "."))
    rt = int(m.group("rt"))
    batch = int(m.group("b"))
    ai = int(m.group("ai"))

    vals = {
        "RT-heavy": rt,
        "Batch-heavy": batch,
        "AI-heavy": ai,
    }
    best = max(vals, key=vals.get)
    task_group = best if vals[best] >= 50 else "Mixed"

    # ???/??????
    # ?????? lam1p8_RT10_Batch10_AI80
    # ?????? lam1p8_RT10_B10_AI80
    # ????? lam1p8_RT10_B10_AI80??? old/new ????
    lam_tag = str(lam).replace(".", "p")
    canonical_scene_key = f"lam{lam_tag}_RT{rt}_B{batch}_AI{ai}"

    return {
        "scene_key": canonical_scene_key,
        "lambda": lam,
        "rt": rt,
        "batch": batch,
        "ai": ai,
        "task_group": task_group,
    }


def find_round_files(root: Path):
    files = []
    for p in root.rglob("*round_summary*csv"):
        s = str(p).lower()
        if "_short_export" in s:
            continue
        if "analysis" in s:
            continue
        files.append(p)
    return sorted(files)


def read_csv_safe(path: Path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False), ""
    except Exception as e:
        return None, repr(e)


def scan_root(root: Path, root_label: str):
    rows = []
    files = find_round_files(root)

    for p in files:
        scene = parse_scene_from_path(p)
        if scene is None:
            continue

        df, err = read_csv_safe(p)
        if df is None:
            rows.append({
                "root_label": root_label,
                "file": str(p),
                "scene_key": scene["scene_key"],
                "ok": False,
                "err": err,
                "rows": 0,
                "method_col": "",
                "methods": "",
            })
            continue

        method_col = pick_col(df, METHOD_COL_CANDIDATES)
        if method_col is None:
            rows.append({
                "root_label": root_label,
                "file": str(p),
                "scene_key": scene["scene_key"],
                "ok": False,
                "err": "method_col_not_found",
                "rows": len(df),
                "method_col": "",
                "methods": "",
            })
            continue

        methods = sorted(df[method_col].dropna().astype(str).unique())

        rows.append({
            "root_label": root_label,
            "file": str(p),
            "scene_key": scene["scene_key"],
            "lambda": scene["lambda"],
            "rt": scene["rt"],
            "batch": scene["batch"],
            "ai": scene["ai"],
            "task_group": scene["task_group"],
            "ok": True,
            "err": "",
            "rows": len(df),
            "method_col": method_col,
            "method_count": len(methods),
            "methods": "|".join(methods),
            "columns_json": json.dumps(list(df.columns), ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def choose_scenes(old_scan: pd.DataFrame, new_scan: pd.DataFrame):
    old_scenes = set(old_scan.loc[old_scan["ok"], "scene_key"].dropna().astype(str))
    new_scenes = set(new_scan.loc[new_scan["ok"], "scene_key"].dropna().astype(str))

    common = sorted(old_scenes & new_scenes)

    if MANUAL_SCENES:
        selected = [s for s in MANUAL_SCENES if s in common]
    else:
        rnd = random.Random(RANDOM_SEED)
        selected = rnd.sample(common, min(RANDOM_SCENE_N, len(common)))

    selected_df = pd.DataFrame({"scene_key": selected})
    return common, selected_df


def load_records_for_selected(root: Path, root_label: str, selected_scenes):
    files = find_round_files(root)
    parts = []
    file_rows = []

    for p in files:
        scene = parse_scene_from_path(p)
        if scene is None:
            continue
        if scene["scene_key"] not in selected_scenes:
            continue

        df, err = read_csv_safe(p)
        if df is None:
            continue

        method_col = pick_col(df, METHOD_COL_CANDIDATES)
        if method_col is None:
            continue

        iter_col = pick_col(df, ITER_COL_CANDIDATES)
        if iter_col is None:
            df["_iter_tmp"] = np.arange(1, len(df) + 1)
            iter_col = "_iter_tmp"

        cost_col = pick_col(df, METRIC_CANDIDATES["cost"], loose_keys=["cost"])
        if cost_col is None:
            continue

        methods = sorted(df[method_col].dropna().astype(str).unique())

        file_rows.append({
            "root_label": root_label,
            "file": str(p),
            "scene_key": scene["scene_key"],
            "rows": len(df),
            "method_col": method_col,
            "iter_col": iter_col,
            "cost_col": cost_col,
            "methods": "|".join(methods),
        })

        for method_key in methods:
            sub = df[df[method_col].astype(str) == method_key].copy()
            if sub.empty:
                continue

            sub = sub.sort_values(iter_col).reset_index(drop=True)

            rec = pd.DataFrame()
            rec["root_label"] = root_label
            rec["scene_key"] = scene["scene_key"]
            rec["lambda"] = scene["lambda"]
            rec["rt"] = scene["rt"]
            rec["batch"] = scene["batch"]
            rec["ai"] = scene["ai"]
            rec["task_group"] = scene["task_group"]
            rec["method_key"] = method_key
            rec["method_label"] = method_label(method_key)
            rec["iter"] = pd.to_numeric(sub[iter_col], errors="coerce")
            rec["cost"] = pd.to_numeric(sub[cost_col], errors="coerce")

            for metric, candidates in METRIC_CANDIDATES.items():
                if metric == "cost":
                    continue

                loose = None
                if metric == "delay":
                    loose = ["delay", "latency", "时延", "延迟"]
                elif metric == "energy":
                    loose = ["energy", "能耗"]
                elif metric == "backlog":
                    loose = ["backlog", "unfinished", "积压"]
                elif metric == "bo_training_cost":
                    loose = ["bo_training", "training_cost"]

                col = pick_col(sub, candidates, loose_keys=loose)
                if col is not None:
                    rec[metric] = pd.to_numeric(sub[col], errors="coerce")
                else:
                    rec[metric] = np.nan

            rec = rec.dropna(subset=["iter", "cost"])

            # ?????
            # rec ??? DataFrame ???????? scalar ?????
            # pandas ??????? iter/cost ?????????
            # ????????????????????????
            if rec.empty:
                continue

            rec["root_label"] = root_label
            rec["scene_key"] = scene["scene_key"]
            rec["lambda"] = scene["lambda"]
            rec["rt"] = scene["rt"]
            rec["batch"] = scene["batch"]
            rec["ai"] = scene["ai"]
            rec["task_group"] = scene["task_group"]
            rec["method_key"] = method_key
            rec["method_label"] = method_label(method_key)

            rec = rec.sort_values("iter").reset_index(drop=True)

            for metric in [
                "cost",
                "delay",
                "energy",
                "backlog",
                "rt_delay",
                "batch_delay",
                "ai_delay",
                "bo_training_cost",
            ]:
                if metric in rec.columns:
                    rec[f"roll50_{metric}"] = rec[metric].rolling(ROLLING, min_periods=ROLLING).mean()

            parts.append(rec)

    if not parts:
        raise RuntimeError(f"No records loaded from {root_label}")

    return pd.concat(parts, ignore_index=True), pd.DataFrame(file_rows)


def summarize_scene_method(raw: pd.DataFrame):
    rows = []

    for (root_label, scene_key, method_key), g in raw.groupby(["root_label", "scene_key", "method_key"]):
        g = g.sort_values("iter")
        first = g.iloc[0]

        rec = {
            "root_label": root_label,
            "scene_key": scene_key,
            "lambda": first["lambda"],
            "rt": first["rt"],
            "batch": first["batch"],
            "ai": first["ai"],
            "task_group": first["task_group"],
            "method_key": method_key,
            "method_label": first["method_label"],
            "rows": len(g),
        }

        for metric in [
            "cost",
            "delay",
            "energy",
            "backlog",
            "rt_delay",
            "batch_delay",
            "ai_delay",
            "bo_training_cost",
        ]:
            if metric not in g.columns:
                continue

            x = pd.to_numeric(g[metric], errors="coerce")
            if x.notna().sum() == 0:
                continue

            rec[f"mean_{metric}"] = x.mean()
            rec[f"first100_{metric}"] = x.head(100).mean()
            rec[f"tail100_{metric}"] = x.tail(100).mean()
            rec[f"last50_{metric}"] = x.tail(50).mean()

            rcol = f"roll50_{metric}"
            if rcol in g.columns:
                rx = pd.to_numeric(g[rcol], errors="coerce")
                rec[f"final_roll50_{metric}"] = rx.dropna().iloc[-1] if rx.dropna().shape[0] else np.nan
                rec[f"min_roll50_{metric}"] = rx.min()

        if "bo_training_cost" in g.columns:
            a = pd.to_numeric(g["cost"], errors="coerce")
            b = pd.to_numeric(g["bo_training_cost"], errors="coerce")
            rec["bo_eval_abs_diff_mean"] = (a - b).abs().mean()
            rec["bo_eval_abs_diff_max"] = (a - b).abs().max()

        rows.append(rec)

    return pd.DataFrame(rows)


def aggregate_overall(sm: pd.DataFrame):
    metric_cols = [
        c for c in sm.columns
        if c.startswith(("mean_", "first100_", "tail100_", "last50_", "final_roll50_", "min_roll50_"))
    ]

    out = (
        sm.groupby(["root_label", "method_key", "method_label"])
        .agg(
            scene_count=("scene_key", "nunique"),
            rows_mean=("rows", "mean"),
            **{c: (c, "mean") for c in metric_cols},
        )
        .reset_index()
    )

    if "final_roll50_cost" in out.columns:
        out = out.sort_values(["root_label", "final_roll50_cost"])

    return out


def add_baseline_gaps(sm: pd.DataFrame):
    out = sm.copy()

    # 对每个 root/scene 内，找 fixed-mid/fixed-tuned/best-fixed
    for metric in ["cost", "delay", "energy", "backlog"]:
        val = f"final_roll50_{metric}"
        if val not in out.columns:
            continue

        # Fixed-mid
        for baseline_key, baseline_name in [
            ("reduced6_fixed_mid", "Fixed-mid"),
            ("reduced6_fixed_tuned", "Fixed-tuned"),
        ]:
            base = out[out["method_key"] == baseline_key][["root_label", "scene_key", val]].rename(
                columns={val: f"{baseline_name}_{val}"}
            )
            out = out.merge(base, on=["root_label", "scene_key"], how="left")
            bcol = f"{baseline_name}_{val}"
            out[f"gap_vs_{baseline_name}_{metric}_pct"] = 100.0 * (out[val] - out[bcol]) / out[bcol]
            out[f"gain_vs_{baseline_name}_{metric}_pct"] = -out[f"gap_vs_{baseline_name}_{metric}_pct"]

        # Best fixed
        fixed = out[out["method_key"].isin(FIXED_KEYS)].copy()
        best = fixed.groupby(["root_label", "scene_key"])[val].min().reset_index().rename(
            columns={val: f"Best-fixed_{val}"}
        )
        out = out.merge(best, on=["root_label", "scene_key"], how="left")
        bcol = f"Best-fixed_{val}"
        out[f"gap_vs_Best-fixed_{metric}_pct"] = 100.0 * (out[val] - out[bcol]) / out[bcol]
        out[f"gain_vs_Best-fixed_{metric}_pct"] = -out[f"gap_vs_Best-fixed_{metric}_pct"]

    return out


def pairwise_gap_summary(sm_gap: pd.DataFrame):
    rows = []
    for metric in ["cost", "delay", "energy", "backlog"]:
        for baseline in ["Fixed-mid", "Fixed-tuned", "Best-fixed"]:
            gap_col = f"gap_vs_{baseline}_{metric}_pct"
            gain_col = f"gain_vs_{baseline}_{metric}_pct"
            if gap_col not in sm_gap.columns:
                continue

            tmp = (
                sm_gap.groupby(["root_label", "method_key", "method_label"])
                .agg(
                    scenes=("scene_key", "nunique"),
                    mean_gap_pct=(gap_col, "mean"),
                    median_gap_pct=(gap_col, "median"),
                    mean_gain_pct=(gain_col, "mean"),
                    positive_gain_count=(gain_col, lambda x: int((x > 0).sum())),
                )
                .reset_index()
            )
            tmp["metric"] = metric
            tmp["baseline"] = baseline
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def old_vs_new_same_method(sm: pd.DataFrame):
    rows = []

    metrics = ["cost", "delay", "energy", "backlog", "rt_delay", "batch_delay", "ai_delay"]

    detail_parts = []

    for metric in metrics:
        val = f"final_roll50_{metric}"
        if val not in sm.columns:
            continue

        piv = sm.pivot_table(
            index=["scene_key", "lambda", "task_group", "method_key", "method_label"],
            columns="root_label",
            values=val,
            aggfunc="mean",
        ).reset_index()

        if "old_original_cost" not in piv.columns or "new_modified_cost" not in piv.columns:
            continue

        piv[f"new_minus_old_{metric}"] = piv["new_modified_cost"] - piv["old_original_cost"]
        piv[f"new_vs_old_gap_pct_{metric}"] = 100.0 * (
            piv["new_modified_cost"] - piv["old_original_cost"]
        ) / piv["old_original_cost"]
        piv["metric"] = metric

        detail_parts.append(piv)

        gap_col = f"new_vs_old_gap_pct_{metric}"

        for (method_key, method_label), g in piv.groupby(["method_key", "method_label"]):
            rows.append({
                "metric": metric,
                "method_key": method_key,
                "method_label": method_label,
                "scenes": g["scene_key"].nunique(),
                "mean_new_vs_old_gap_pct": g[gap_col].mean(),
                "median_new_vs_old_gap_pct": g[gap_col].median(),
                "new_better_count": int((g[gap_col] < 0).sum()),
            })

        for (lam, method_key, method_label), g in piv.groupby(["lambda", "method_key", "method_label"]):
            rows.append({
                "metric": metric,
                "group_type": "lambda",
                "group": lam,
                "method_key": method_key,
                "method_label": method_label,
                "scenes": g["scene_key"].nunique(),
                "mean_new_vs_old_gap_pct": g[gap_col].mean(),
                "median_new_vs_old_gap_pct": g[gap_col].median(),
                "new_better_count": int((g[gap_col] < 0).sum()),
            })

        for (tg, method_key, method_label), g in piv.groupby(["task_group", "method_key", "method_label"]):
            rows.append({
                "metric": metric,
                "group_type": "task_group",
                "group": tg,
                "method_key": method_key,
                "method_label": method_label,
                "scenes": g["scene_key"].nunique(),
                "mean_new_vs_old_gap_pct": g[gap_col].mean(),
                "median_new_vs_old_gap_pct": g[gap_col].median(),
                "new_better_count": int((g[gap_col] < 0).sum()),
            })

    if detail_parts:
        detail = pd.concat(detail_parts, ignore_index=True)
        detail.to_csv(OUT / "old_vs_new_same_method_scene_detail.csv", index=False, encoding="utf-8-sig")

    return pd.DataFrame(rows)


def focus_tables(sm_gap: pd.DataFrame):
    # 重点输出 CBO / BO / fixed 的 cost 对比
    focus_mask = (
        sm_gap["method_key"].isin(FIXED_KEYS)
        | sm_gap["method_key"].str.contains("cbo", case=False, na=False)
        | sm_gap["method_key"].str.contains("bo", case=False, na=False)
    )
    focus = sm_gap[focus_mask].copy()

    keep_cols = [
        "root_label", "scene_key", "lambda", "task_group",
        "method_key", "method_label",
        "final_roll50_cost", "last50_cost", "tail100_cost",
        "final_roll50_delay", "final_roll50_energy", "final_roll50_backlog",
        "gap_vs_Fixed-mid_cost_pct",
        "gap_vs_Fixed-tuned_cost_pct",
        "gap_vs_Best-fixed_cost_pct",
        "bo_eval_abs_diff_mean",
        "bo_eval_abs_diff_max",
    ]
    keep_cols = [c for c in keep_cols if c in focus.columns]
    return focus[keep_cols].sort_values(["scene_key", "root_label", "final_roll50_cost"])


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("[STEP 1] scan roots")
    old_scan = scan_root(OLD_ROOT, "old_original_cost")
    new_scan = scan_root(NEW_ROOT, "new_modified_cost")

    old_scan.to_csv(OUT / "old_root_file_inventory.csv", index=False, encoding="utf-8-sig")
    new_scan.to_csv(OUT / "new_root_file_inventory.csv", index=False, encoding="utf-8-sig")

    method_inventory = []
    for root_label, scan in [("old_original_cost", old_scan), ("new_modified_cost", new_scan)]:
        for _, row in scan[scan["ok"]].iterrows():
            methods = str(row["methods"]).split("|") if pd.notna(row.get("methods", "")) else []
            for m in methods:
                method_inventory.append({
                    "root_label": root_label,
                    "scene_key": row["scene_key"],
                    "method_key": m,
                    "method_label": method_label(m),
                })

    method_inv = pd.DataFrame(method_inventory)
    method_summary = (
        method_inv.groupby(["root_label", "method_key", "method_label"])
        .agg(scene_count=("scene_key", "nunique"))
        .reset_index()
        .sort_values(["root_label", "method_label"])
    )
    method_summary.to_csv(OUT / "method_inventory_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[METHOD INVENTORY]")
    print(method_summary.to_string(index=False))

    common_scenes, selected_df = choose_scenes(old_scan, new_scan)
    selected_df.to_csv(OUT / "selected_random_scenes.csv", index=False, encoding="utf-8-sig")

    print("\n[SCENE COUNTS]")
    print("old scenes:", old_scan.loc[old_scan["ok"], "scene_key"].nunique())
    print("new scenes:", new_scan.loc[new_scan["ok"], "scene_key"].nunique())
    print("common scenes:", len(common_scenes))
    print("selected scenes:")
    print(selected_df.to_string(index=False))

    selected_scenes = set(selected_df["scene_key"].astype(str))

    print("\n[STEP 2] load selected scene records")
    old_raw, old_file = load_records_for_selected(OLD_ROOT, "old_original_cost", selected_scenes)
    new_raw, new_file = load_records_for_selected(NEW_ROOT, "new_modified_cost", selected_scenes)

    raw = pd.concat([old_raw, new_raw], ignore_index=True)
    file_info = pd.concat([old_file, new_file], ignore_index=True)

    raw.to_csv(OUT / "selected_all_round_records.csv", index=False, encoding="utf-8-sig")
    file_info.to_csv(OUT / "selected_file_info.csv", index=False, encoding="utf-8-sig")

    print("\n[LOADED SELECTED RECORDS]")
    loaded = (
        raw.groupby(["root_label", "method_key", "method_label"])
        .agg(rows=("cost", "size"), scenes=("scene_key", "nunique"))
        .reset_index()
        .sort_values(["root_label", "method_label"])
    )
    print(loaded.to_string(index=False))
    loaded.to_csv(OUT / "selected_loaded_method_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[STEP 3] summarize")
    sm = summarize_scene_method(raw)
    sm.to_csv(OUT / "selected_scene_method_summary.csv", index=False, encoding="utf-8-sig")

    sm_gap = add_baseline_gaps(sm)
    sm_gap.to_csv(OUT / "selected_scene_method_summary_with_gaps.csv", index=False, encoding="utf-8-sig")

    overall = aggregate_overall(sm)
    overall.to_csv(OUT / "selected_aggregate_overall.csv", index=False, encoding="utf-8-sig")

    pairwise = pairwise_gap_summary(sm_gap)
    pairwise.to_csv(OUT / "selected_pairwise_gap_summary.csv", index=False, encoding="utf-8-sig")

    oldnew = old_vs_new_same_method(sm)
    oldnew.to_csv(OUT / "selected_old_vs_new_same_method_summary.csv", index=False, encoding="utf-8-sig")

    focus = focus_tables(sm_gap)
    focus.to_csv(OUT / "selected_focus_cbo_fixed_scene_table.csv", index=False, encoding="utf-8-sig")

    # Excel 汇总
    xlsx = OUT / "compare_old_new_random12_report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        selected_df.to_excel(writer, sheet_name="selected_scenes", index=False)
        method_summary.to_excel(writer, sheet_name="method_inventory", index=False)
        loaded.to_excel(writer, sheet_name="loaded_methods", index=False)
        overall.to_excel(writer, sheet_name="overall", index=False)
        pairwise.to_excel(writer, sheet_name="pairwise_gap", index=False)
        oldnew.to_excel(writer, sheet_name="old_vs_new", index=False)
        focus.to_excel(writer, sheet_name="focus_scene_table", index=False)

    print("\n" + "=" * 100)
    print("[OVERALL final_roll50_cost]")
    cols = [
        "root_label", "method_label", "method_key", "scene_count",
        "final_roll50_cost", "final_roll50_delay", "final_roll50_energy",
        "min_roll50_cost",
    ]
    cols = [c for c in cols if c in overall.columns]
    print(overall[cols].sort_values(["root_label", "final_roll50_cost"]).to_string(index=False))

    print("\n" + "=" * 100)
    print("[PAIRWISE GAP vs Fixed-mid, cost]")
    if not pairwise.empty:
        sub = pairwise[(pairwise["baseline"] == "Fixed-mid") & (pairwise["metric"] == "cost")]
        print(sub.sort_values(["root_label", "mean_gap_pct"]).to_string(index=False))

    print("\n" + "=" * 100)
    print("[OLD vs NEW same method: cost/delay/energy]")
    if not oldnew.empty:
        sub = oldnew[oldnew["metric"].isin(["cost", "delay", "energy"])]
        print(sub.sort_values(["metric", "method_label"]).to_string(index=False))

    print("\n[DONE]")
    print("Output:", OUT)
    print("Excel:", xlsx)


if __name__ == "__main__":
    main()