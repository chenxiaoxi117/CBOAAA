from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


TRANSFER_ROOT = Path(r"D:\CBOv2\results\transfer_cbo_pressure")
OUTDIR = TRANSFER_ROOT / "analysis_warm_start_transfer" / "gap_to_tuned"
FIGDIR = OUTDIR / "figures"

# Target scenes only. Source is not compared here.
SCENES = {
    "similar_RT50": {
        "scene_tag": "lam2p6_RT50_Batch40_AI10",
        "cold_dir": TRANSFER_ROOT / "target_similar_cold_lam2p6_RT50_Batch40_AI10",
        "warm_dir": TRANSFER_ROOT / "target_similar_warm_lam2p6_RT50_Batch40_AI10",
    },
    "dissimilar_AI70": {
        "scene_tag": "lam3p0_RT10_Batch20_AI70",
        "cold_dir": TRANSFER_ROOT / "target_dissimilar_cold_lam3p0_RT10_Batch20_AI70",
        "warm_dir": TRANSFER_ROOT / "target_dissimilar_warm_lam3p0_RT10_Batch20_AI70",
    },
}

# 搜索 fixed_tuned 的范围。可以按你机器上的实际结果目录增删。
SEARCH_ROOTS = [
    Path(r"D:\CBOv2\results"),
    Path(r"D:\CBO\新的结果"),
    Path(r"D:\CBO\525"),
    Path(r"D:\CBO"),
]
MANUAL_TUNED_PATHS = {
    "lam2p6_RT50_Batch40_AI10": Path(
        r"D:\CBO\v6_3lambda_36_context500_results_107_plus_timeout\v6_3lambda_36_context500\lam2p6_RT50_Batch40_AI10\task_effective\reduced6_fixed_tuned_round_summary_轮次汇总.csv"
    ),
    "lam3p0_RT10_Batch20_AI70": Path(
        r"D:\CBO\v6_3lambda_36_context500_results_107_plus_timeout\v6_3lambda_36_context500\lam3p0_RT10_Batch20_AI70\task_effective\reduced6_fixed_tuned_round_summary_轮次汇总.csv"
    ),
}

def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def find_col(df: pd.DataFrame, names: List[str], fuzzy: bool = True) -> Optional[str]:
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


def iter_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["Iteration_轮次", "Iteration", "iteration", "iter", "round", "Round"])


def cost_col(df: pd.DataFrame) -> str:
    c = find_col(df, ["Eval_Cost", "eval_cost", "BO_Training_Cost", "Cost", "cost"])
    if c is None:
        raise ValueError(f"Cannot find cost column. Columns={list(df.columns)}")
    return c


def metric_col(df: pd.DataFrame, kind: str) -> Optional[str]:
    mapping = {
        "delay": ["Avg_Delay", "avg_delay", "Delay", "delay"],
        "energy": ["Avg_Energy", "avg_energy", "Energy", "energy"],
        "backlog": ["Backlog", "Avg_Backlog", "backlog", "Backlog_End"],
        "unfinished": ["unfinished_end", "Unfinished_End", "unfinished", "Unfinished"],
        "violation": ["Violation", "violation", "Violation_Rate"],
        "sla": ["SLA", "sla", "SLA_Satisfaction"],
    }
    return find_col(df, mapping[kind])


def find_round_summary(run_dir: Path) -> Path:
    files = [
        p for p in run_dir.rglob("*.csv")
        if "round_summary" in p.name.lower()
        and "_short_export" not in str(p).lower()
    ]
    if not files:
        raise FileNotFoundError(f"No round_summary found under {run_dir}")

    scored = []
    for p in files:
        try:
            df = read_csv_safely(p)
            scored.append((len(df), len(str(p)), p))
        except Exception:
            continue

    if not scored:
        raise FileNotFoundError(f"round_summary candidates found but none readable under {run_dir}")

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def looks_like_fixed_tuned(path: Path, df: Optional[pd.DataFrame] = None) -> bool:
    s = str(path).lower()
    if "fixed_tuned" in s or "reduced6_fixed_tuned" in s:
        return True

    if df is not None:
        for c in df.columns:
            if str(c).lower() in ("group", "method", "method_name", "selected_key"):
                vals = " ".join(str(x).lower() for x in df[c].dropna().unique()[:20])
                if "fixed_tuned" in vals or "reduced6_fixed_tuned" in vals:
                    return True
    return False


def find_fixed_tuned_for_scene(scene_tag: str) -> Path:
    manual = MANUAL_TUNED_PATHS.get(scene_tag)
    if manual is not None:
        if manual.exists():
            return manual
        raise FileNotFoundError(f"Manual fixed_tuned path does not exist for {scene_tag}: {manual}")

    candidates: List[Tuple[int, int, Path]] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        # 先用 ASCII scene_tag 过滤路径，避免中文路径硬编码问题。
        for p in root.rglob("*.csv"):
            sp = str(p)
            if scene_tag not in sp:
                continue
            if "_short_export" in sp.lower():
                continue
            if "round_summary" not in p.name.lower():
                continue

            try:
                df = read_csv_safely(p)
            except Exception:
                continue

            if len(df) < 100:
                continue

            is_fixed = looks_like_fixed_tuned(p, df)
            if not is_fixed:
                continue

            # 优先 500 行、路径短的 fixed_tuned。
            row_score = abs(len(df) - 500)
            candidates.append((row_score, len(sp), p))

    if not candidates:
        raise FileNotFoundError(f"No fixed_tuned round_summary found for scene_tag={scene_tag}")

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def align_to_tuned(run_df: pd.DataFrame, tuned_df: pd.DataFrame) -> pd.DataFrame:
    r_iter = iter_col(run_df)
    t_iter = iter_col(tuned_df)
    r_cost = cost_col(run_df)
    t_cost = cost_col(tuned_df)

    r = run_df.copy()
    t = tuned_df.copy()

    if r_iter and t_iter:
        r["_iter_key"] = pd.to_numeric(r[r_iter], errors="coerce")
        t["_iter_key"] = pd.to_numeric(t[t_iter], errors="coerce")
        merged = pd.merge(
            r[["_iter_key", r_cost]],
            t[["_iter_key", t_cost]],
            on="_iter_key",
            how="inner",
            suffixes=("_run", "_tuned"),
        )
        run_col = [c for c in merged.columns if c.endswith("_run")][0]
        tuned_col = [c for c in merged.columns if c.endswith("_tuned")][0]
    else:
        n = min(len(r), len(t))
        merged = pd.DataFrame({
            "_iter_key": range(1, n + 1),
            "run_cost": pd.to_numeric(r[r_cost].iloc[:n], errors="coerce").to_numpy(),
            "tuned_cost": pd.to_numeric(t[t_cost].iloc[:n], errors="coerce").to_numpy(),
        })
        run_col = "run_cost"
        tuned_col = "tuned_cost"

    merged["run_cost"] = pd.to_numeric(merged[run_col], errors="coerce")
    merged["tuned_cost"] = pd.to_numeric(merged[tuned_col], errors="coerce")
    merged["gap_to_tuned"] = (merged["run_cost"] - merged["tuned_cost"]) / merged["tuned_cost"]
    merged["rolling50_gap_to_tuned"] = merged["gap_to_tuned"].rolling(50, min_periods=50).mean()
    merged["rolling50_run_cost"] = merged["run_cost"].rolling(50, min_periods=50).mean()
    merged["rolling50_tuned_cost"] = merged["tuned_cost"].rolling(50, min_periods=50).mean()
    return merged[[
        "_iter_key",
        "run_cost",
        "tuned_cost",
        "gap_to_tuned",
        "rolling50_gap_to_tuned",
        "rolling50_run_cost",
        "rolling50_tuned_cost",
    ]]


def window_mean(s: pd.Series, n: int, mode: str) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return math.nan
    if mode == "first":
        return float(s.head(n).mean())
    if mode == "last":
        return float(s.tail(n).mean())
    raise ValueError(mode)


def summarize_gap(scene: str, run_name: str, aligned: pd.DataFrame, run_path: Path, tuned_path: Path) -> Dict:
    gap = pd.to_numeric(aligned["gap_to_tuned"], errors="coerce")
    roll = pd.to_numeric(aligned["rolling50_gap_to_tuned"], errors="coerce")

    return {
        "scene": scene,
        "run": run_name,
        "rows": len(aligned),
        "run_path": str(run_path),
        "tuned_path": str(tuned_path),
        "mean_gap_pct": float(gap.mean() * 100),
        "median_gap_pct": float(gap.median() * 100),
        "first50_gap_pct": float(window_mean(gap, 50, "first") * 100),
        "first100_gap_pct": float(window_mean(gap, 100, "first") * 100),
        "tail100_gap_pct": float(window_mean(gap, 100, "last") * 100),
        "last50_gap_pct": float(window_mean(gap, 50, "last") * 100),
        "better_than_tuned_ratio": float((gap < 0).mean()),
        "rolling50_gap_min_pct": float(roll.min() * 100) if roll.notna().any() else math.nan,
        "rolling50_gap_final_pct": float(roll.dropna().iloc[-1] * 100) if roll.notna().any() else math.nan,
    }


def compare_warm_cold_gap(scene: str, cold_aligned: pd.DataFrame, warm_aligned: pd.DataFrame) -> Dict:
    c = cold_aligned[["_iter_key", "gap_to_tuned"]].rename(columns={"gap_to_tuned": "cold_gap"})
    w = warm_aligned[["_iter_key", "gap_to_tuned"]].rename(columns={"gap_to_tuned": "warm_gap"})
    m = pd.merge(c, w, on="_iter_key", how="inner")
    m["gap_improvement"] = m["cold_gap"] - m["warm_gap"]  # >0 means warm closer to tuned

    return {
        "scene": scene,
        "rows": len(m),
        "mean_gap_improvement_pct": float(m["gap_improvement"].mean() * 100),
        "first50_gap_improvement_pct": float(window_mean(m["gap_improvement"], 50, "first") * 100),
        "first100_gap_improvement_pct": float(window_mean(m["gap_improvement"], 100, "first") * 100),
        "tail100_gap_improvement_pct": float(window_mean(m["gap_improvement"], 100, "last") * 100),
        "last50_gap_improvement_pct": float(window_mean(m["gap_improvement"], 50, "last") * 100),
        "warm_closer_to_tuned_ratio": float((m["gap_improvement"] > 0).mean()),
    }


def plot_gap_curve(scene: str, cold_aligned: pd.DataFrame, warm_aligned: pd.DataFrame) -> Path:
    plt.figure(figsize=(11, 6))

    plt.plot(
        cold_aligned["_iter_key"],
        cold_aligned["gap_to_tuned"] * 100,
        alpha=0.18,
        label="cold raw gap",
    )
    plt.plot(
        warm_aligned["_iter_key"],
        warm_aligned["gap_to_tuned"] * 100,
        alpha=0.18,
        label="warm raw gap",
    )
    plt.plot(
        cold_aligned["_iter_key"],
        cold_aligned["rolling50_gap_to_tuned"] * 100,
        linewidth=2,
        label="cold rolling50 gap",
    )
    plt.plot(
        warm_aligned["_iter_key"],
        warm_aligned["rolling50_gap_to_tuned"] * 100,
        linewidth=2,
        label="warm rolling50 gap",
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Gap to fixed_tuned (%)")
    plt.title(f"{scene}: gap to fixed_tuned, cold vs warm")
    plt.legend()
    plt.tight_layout()

    out = FIGDIR / f"{scene}_gap_to_tuned_curve.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_gap_improvement(scene: str, cold_aligned: pd.DataFrame, warm_aligned: pd.DataFrame) -> Path:
    c = cold_aligned[["_iter_key", "gap_to_tuned"]].rename(columns={"gap_to_tuned": "cold_gap"})
    w = warm_aligned[["_iter_key", "gap_to_tuned"]].rename(columns={"gap_to_tuned": "warm_gap"})
    m = pd.merge(c, w, on="_iter_key", how="inner")
    m["gap_improvement"] = (m["cold_gap"] - m["warm_gap"]) * 100
    m["rolling50_gap_improvement"] = m["gap_improvement"].rolling(50, min_periods=50).mean()

    plt.figure(figsize=(11, 6))
    plt.plot(m["_iter_key"], m["gap_improvement"], alpha=0.22, label="raw gap improvement")
    plt.plot(m["_iter_key"], m["rolling50_gap_improvement"], linewidth=2, label="rolling50 gap improvement")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Cold gap - Warm gap to tuned (%)")
    plt.title(f"{scene}: warm-start improvement in gap to fixed_tuned")
    plt.legend()
    plt.tight_layout()

    out = FIGDIR / f"{scene}_gap_improvement_curve.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_window_gap_bar(gap_summary: pd.DataFrame) -> Path:
    keep_cols = ["first100_gap_pct", "tail100_gap_pct", "last50_gap_pct"]
    df = gap_summary.copy()
    df["label"] = df["scene"] + "_" + df["run"]
    data = df.set_index("label")[keep_cols]

    ax = data.plot(kind="bar", figsize=(12, 6))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Gap to fixed_tuned (%)")
    ax.set_title("Window gap to fixed_tuned")
    ax.set_xticklabels(data.index, rotation=25, ha="right")
    plt.tight_layout()

    out = FIGDIR / "window_gap_to_tuned_bar.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def write_report(
    gap_summary: pd.DataFrame,
    gap_improvement: pd.DataFrame,
    selected: Dict[str, Dict[str, Path]],
    figs: List[Path],
) -> Path:
    report = OUTDIR / "gap_to_tuned_report.md"
    lines = []

    lines.append("# Warm-start Gap-to-fixed_tuned Analysis\n")

    lines.append("## Selected files\n")
    for scene, paths in selected.items():
        lines.append(f"### {scene}")
        for k, p in paths.items():
            lines.append(f"- **{k}**: `{p}`")
    lines.append("")

    lines.append("## Gap summary\n")
    cols = [
        "scene", "run",
        "mean_gap_pct", "first100_gap_pct", "tail100_gap_pct", "last50_gap_pct",
        "rolling50_gap_min_pct", "rolling50_gap_final_pct",
        "better_than_tuned_ratio",
    ]
    lines.append(gap_summary[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Warm-start improvement in gap to tuned\n")
    lines.append("Positive values mean warm-start is closer to fixed_tuned than cold-start.")
    lines.append(gap_improvement.to_markdown(index=False))
    lines.append("")

    lines.append("## Figures\n")
    for p in figs:
        lines.append(f"- `{p}`")
    lines.append("")

    lines.append("## Reading guide\n")
    lines.append("- `gap_to_tuned = (method - fixed_tuned) / fixed_tuned`.")
    lines.append("- Gap below 0 means the method is better than fixed_tuned for that window.")
    lines.append("- `gap_improvement = cold_gap - warm_gap`; positive means warm-start reduces the gap to fixed_tuned.")
    lines.append("- This view is more suitable than raw cold-vs-warm curves when judging practical closeness to the tuned baseline.")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    gap_rows = []
    improvement_rows = []
    selected_paths: Dict[str, Dict[str, Path]] = {}
    figs: List[Path] = []

    for scene, info in SCENES.items():
        print(f"\n=== {scene} ===")

        cold_path = find_round_summary(info["cold_dir"])
        warm_path = find_round_summary(info["warm_dir"])
        tuned_path = find_fixed_tuned_for_scene(info["scene_tag"])

        cold_df = read_csv_safely(cold_path)
        warm_df = read_csv_safely(warm_path)
        tuned_df = read_csv_safely(tuned_path)

        selected_paths[scene] = {
            "cold": cold_path,
            "warm": warm_path,
            "fixed_tuned": tuned_path,
        }

        print(f"cold: {cold_path}")
        print(f"warm: {warm_path}")
        print(f"tuned: {tuned_path}")

        cold_aligned = align_to_tuned(cold_df, tuned_df)
        warm_aligned = align_to_tuned(warm_df, tuned_df)

        cold_aligned.to_csv(OUTDIR / f"{scene}_cold_gap_to_tuned.csv", index=False, encoding="utf-8-sig")
        warm_aligned.to_csv(OUTDIR / f"{scene}_warm_gap_to_tuned.csv", index=False, encoding="utf-8-sig")

        gap_rows.append(summarize_gap(scene, "cold", cold_aligned, cold_path, tuned_path))
        gap_rows.append(summarize_gap(scene, "warm", warm_aligned, warm_path, tuned_path))
        improvement_rows.append(compare_warm_cold_gap(scene, cold_aligned, warm_aligned))

        figs.append(plot_gap_curve(scene, cold_aligned, warm_aligned))
        figs.append(plot_gap_improvement(scene, cold_aligned, warm_aligned))

    gap_summary = pd.DataFrame(gap_rows)
    gap_improvement = pd.DataFrame(improvement_rows)

    gap_summary.to_csv(OUTDIR / "gap_to_tuned_summary.csv", index=False, encoding="utf-8-sig")
    gap_improvement.to_csv(OUTDIR / "gap_improvement_summary.csv", index=False, encoding="utf-8-sig")

    figs.append(plot_window_gap_bar(gap_summary))

    fig_index = FIGDIR / "figures_index.md"
    fig_index.write_text("\n".join(f"- `{p}`" for p in figs), encoding="utf-8")

    report = write_report(gap_summary, gap_improvement, selected_paths, figs)

    print("\n=== Gap improvement summary ===")
    for _, r in gap_improvement.iterrows():
        print(
            f"{r['scene']}: "
            f"first100 gap improvement={r['first100_gap_improvement_pct']:.2f} pp, "
            f"tail100={r['tail100_gap_improvement_pct']:.2f} pp, "
            f"last50={r['last50_gap_improvement_pct']:.2f} pp, "
            f"warm_closer_ratio={r['warm_closer_to_tuned_ratio']:.3f}"
        )

    print("\nOutputs:")
    print(f"  {OUTDIR / 'gap_to_tuned_summary.csv'}")
    print(f"  {OUTDIR / 'gap_improvement_summary.csv'}")
    print(f"  {report}")
    print(f"  {FIGDIR}")


if __name__ == "__main__":
    main()