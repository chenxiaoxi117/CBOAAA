from pathlib import Path
import math
import pandas as pd
import numpy as np

CSV_PATH = Path(
    r"D:\CBOv2\results\pressure_context_overnight_validation\targets\seed43\P1_RT50_Batch40_AI10\cbo5d_prev_unfinished_lam2p6_RT50_Batch40_AI10\reduced6_cbo_lite_pressure_prev_unfinished_round_summary_轮次汇总.csv"
)

OUTDIR = CSV_PATH.parent / "prediction_gap_analysis_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def find_col(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        key = str(n).lower()
        if key in lower:
            return lower[key]
    for n in names:
        key = str(n).lower()
        for c in df.columns:
            if key in str(c).lower():
                return c
    return None


def num_series(df, names):
    c = find_col(df, names)
    if c is None:
        return None, None
    return c, pd.to_numeric(df[c], errors="coerce")


def choose_prediction_columns(df):
    """
    Prefer already-derived predicted_cost if it has values.
    If not, reconstruct from selected_candidate_mu.
    """
    pred_col, pred = num_series(df, ["predicted_cost"])
    if pred is not None and pred.notna().sum() > 0:
        return pred_col, pred, "predicted_cost"

    mu_col, mu = num_series(df, ["selected_candidate_mu"])
    if mu is not None and mu.notna().sum() > 0:
        return mu_col, -mu, "reconstructed_from_selected_candidate_mu"

    post_mu_col, post_mu = num_series(df, ["posterior_mu"])
    if post_mu is not None and post_mu.notna().sum() > 0:
        return post_mu_col, -post_mu, "reconstructed_from_posterior_mu"

    return None, None, "missing"


def choose_sigma_columns(df):
    sig_col, sig = num_series(df, ["selected_candidate_sigma"])
    if sig is not None and sig.notna().sum() > 0:
        return sig_col, sig, "selected_candidate_sigma"

    post_sig_col, post_sig = num_series(df, ["posterior_sigma"])
    if post_sig is not None and post_sig.notna().sum() > 0:
        return post_sig_col, post_sig, "posterior_sigma"

    return None, None, "missing"


def stage_slice(df, name):
    if name == "all":
        return df
    if name == "first50":
        return df.iloc[:50]
    if name == "first100":
        return df.iloc[:100]
    if name == "101_200":
        return df.iloc[100:200]
    if name == "201_350":
        return df.iloc[200:350]
    if name == "tail100":
        return df.iloc[-100:]
    if name == "last50":
        return df.iloc[-50:]
    return df


def summarize_stage(df, stage):
    g = stage_slice(df, stage).copy()

    actual_col, actual = num_series(
        g,
        ["Eval_Cost_最终评估Cost", "Eval_Cost", "eval_cost", "current_candidate_cost"],
    )
    pred_col, pred, pred_source = choose_prediction_columns(g)
    sig_col, sigma, sigma_source = choose_sigma_columns(g)

    if actual is None or pred is None:
        return {
            "stage": stage,
            "rows": len(g),
            "valid_prediction_rows": 0,
            "valid_prediction_rate": 0.0,
            "note": "missing actual or predicted cost",
        }

    err = actual - pred

    valid = (
        actual.notna()
        & pred.notna()
        & err.notna()
        & np.isfinite(actual)
        & np.isfinite(pred)
        & np.isfinite(err)
    )

    e = err[valid].astype(float)
    a = actual[valid].astype(float)
    p = pred[valid].astype(float)

    if len(e) == 0:
        return {
            "stage": stage,
            "rows": len(g),
            "valid_prediction_rows": 0,
            "valid_prediction_rate": 0.0,
            "note": "no valid prediction rows",
        }

    row = {
        "stage": stage,
        "rows": len(g),
        "valid_prediction_rows": int(len(e)),
        "valid_prediction_rate": float(len(e) / max(1, len(g))),
        "actual_col": actual_col,
        "pred_col": pred_col,
        "pred_source": pred_source,
        "sigma_col": sig_col or "",
        "sigma_source": sigma_source,
        "actual_cost_mean": float(a.mean()),
        "predicted_cost_mean": float(p.mean()),
        "prediction_error_bias": float(e.mean()),
        "prediction_error_mae": float(e.abs().mean()),
        "prediction_error_rmse": float(math.sqrt((e ** 2).mean())),
        "prediction_error_median": float(e.median()),
        "prediction_error_p25": float(e.quantile(0.25)),
        "prediction_error_p75": float(e.quantile(0.75)),
        "underestimate_rate_actual_gt_pred": float((e > 0).mean()),
        "overestimate_rate_actual_lt_pred": float((e < 0).mean()),
        "large_abs_error_rate_gt_500": float((e.abs() > 500).mean()),
        "large_abs_error_rate_gt_1000": float((e.abs() > 1000).mean()),
    }

    if sigma is not None:
        s = pd.to_numeric(sigma[valid], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        surprise = e / s.clip(lower=1e-9)
        surprise = surprise.replace([np.inf, -np.inf], np.nan).dropna()

        if len(surprise) > 0:
            row.update(
                {
                    "surprise_mean": float(surprise.mean()),
                    "surprise_abs_mean": float(surprise.abs().mean()),
                    "surprise_rate_abs_gt_2": float((surprise.abs() > 2).mean()),
                    "surprise_rate_abs_gt_3": float((surprise.abs() > 3).mean()),
                    "positive_surprise_rate_gt_2": float((surprise > 2).mean()),
                    "negative_surprise_rate_lt_minus_2": float((surprise < -2).mean()),
                }
            )

    return row


df = read_csv_safe(CSV_PATH)

actual_col, actual = num_series(
    df,
    ["Eval_Cost_最终评估Cost", "Eval_Cost", "eval_cost", "current_candidate_cost"],
)
pred_col, pred, pred_source = choose_prediction_columns(df)
sig_col, sigma, sigma_source = choose_sigma_columns(df)

if actual is None or pred is None:
    raise RuntimeError("Cannot find actual cost or prediction columns.")

df["_actual_cost_used"] = actual
df["_predicted_cost_used"] = pred
df["_prediction_error_used"] = actual - pred

if sigma is not None:
    df["_sigma_used"] = sigma
    df["_surprise_used"] = df["_prediction_error_used"] / pd.to_numeric(sigma, errors="coerce").clip(lower=1e-9)
else:
    df["_sigma_used"] = np.nan
    df["_surprise_used"] = np.nan

stages = ["all", "first50", "first100", "101_200", "201_350", "tail100", "last50"]
summary = pd.DataFrame([summarize_stage(df, s) for s in stages])
summary.to_csv(OUTDIR / "prediction_gap_stage_summary.csv", index=False, encoding="utf-8-sig")

# 输出 top 100 最大预测误差轮次
useful_cols = []
for names in [
    ["Iteration_轮次", "Iteration", "iter"],
    ["Eval_Cost_最终评估Cost", "Eval_Cost"],
    ["_predicted_cost_used"],
    ["_prediction_error_used"],
    ["_surprise_used"],
    ["selected_candidate_mu"],
    ["selected_candidate_sigma"],
    ["selected_candidate_acq"],
    ["selected_candidate_source"],
    ["selected_reason"],
    ["Avg_Delay_平均时延", "Avg_Delay"],
    ["Backlog_积压任务数", "Backlog"],
    ["Unfinished_End_轮末未完成任务数", "unfinished_end"],
    ["cbo_tr_radius"],
    ["beta_eff"],
    ["candidate_beta_eff"],
    ["context_mode"],
    ["Context_Vector_情景向量"],
    ["Control_Vector_控制向量"],
]:
    c = find_col(df, names)
    if c is not None and c not in useful_cols:
        useful_cols.append(c)

top = df[useful_cols].copy()
top["_abs_prediction_error"] = df["_prediction_error_used"].abs()
top = top.sort_values("_abs_prediction_error", ascending=False)
top.head(100).to_csv(OUTDIR / "top100_prediction_errors.csv", index=False, encoding="utf-8-sig")

# 输出每轮预测误差序列
ts_cols = useful_cols + ["_actual_cost_used", "_predicted_cost_used", "_prediction_error_used", "_sigma_used", "_surprise_used"]
ts_cols = [c for c in ts_cols if c in df.columns]
df[ts_cols].to_csv(OUTDIR / "prediction_gap_timeseries.csv", index=False, encoding="utf-8-sig")

print("Done.")
print("Input:", CSV_PATH)
print("Output:", OUTDIR)
print("Prediction source:", pred_source, "| col:", pred_col)
print("Sigma source:", sigma_source, "| col:", sig_col)
print(summary.to_string(index=False))