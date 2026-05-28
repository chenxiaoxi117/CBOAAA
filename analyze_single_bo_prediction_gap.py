from pathlib import Path
import math
import pandas as pd
import numpy as np

CSV_PATH = Path(r"D:\CBOv2\results\pressure_context_overnight_validation\targets\seed43\P1_RT50_Batch40_AI10\cbo5d_prev_unfinished_lam2p6_RT50_Batch40_AI10\reduced6_cbo_lite_pressure_prev_unfinished_round_summary_轮次汇总.csv")
OUTDIR = CSV_PATH.parent / "prediction_gap_analysis"
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
        if str(n).lower() in lower:
            return lower[str(n).lower()]
    for n in names:
        key = str(n).lower()
        for c in df.columns:
            if key in str(c).lower():
                return c
    return None

def num(df, names):
    c = find_col(df, names)
    if c is None:
        return None, None
    return c, pd.to_numeric(df[c], errors="coerce")

def stage_slice(df, name):
    n = len(df)
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

    pred_col, pred = num(g, ["predicted_cost"])
    actual_col, actual = num(g, ["actual_cost", "Eval_Cost", "eval_cost", "cost"])
    err_col, err = num(g, ["prediction_error"])
    surprise_col, surprise = num(g, ["surprise"])

    # 如果 prediction_error 缺失，但 predicted/actual 存在，就重算
    if err is None and pred is not None and actual is not None:
        err = actual - pred
        err_col = "actual_cost - predicted_cost"

    if pred is None or actual is None or err is None:
        return {
            "stage": stage,
            "rows": len(g),
            "valid_prediction_rows": 0,
            "valid_prediction_rate": 0.0,
            "note": "missing predicted_cost/actual_cost/prediction_error",
        }

    valid = pred.notna() & actual.notna() & err.notna() & np.isfinite(pred) & np.isfinite(actual) & np.isfinite(err)
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

    abs_e = e.abs()
    sq_e = e ** 2

    row = {
        "stage": stage,
        "rows": len(g),
        "valid_prediction_rows": int(len(e)),
        "valid_prediction_rate": float(len(e) / max(1, len(g))),
        "actual_cost_mean": float(a.mean()),
        "predicted_cost_mean": float(p.mean()),
        "prediction_error_bias": float(e.mean()),
        "prediction_error_mae": float(abs_e.mean()),
        "prediction_error_rmse": float(math.sqrt(sq_e.mean())),
        "prediction_error_median": float(e.median()),
        "prediction_error_p25": float(e.quantile(0.25)),
        "prediction_error_p75": float(e.quantile(0.75)),
        "underestimate_rate_actual_gt_pred": float((e > 0).mean()),
        "overestimate_rate_actual_lt_pred": float((e < 0).mean()),
        "large_abs_error_rate_gt_500": float((abs_e > 500).mean()),
        "large_abs_error_rate_gt_1000": float((abs_e > 1000).mean()),
    }

    if surprise is not None:
        s = pd.to_numeric(surprise[valid], errors="coerce")
        s = s[np.isfinite(s)]
        if len(s):
            row.update({
                "surprise_mean": float(s.mean()),
                "surprise_abs_mean": float(s.abs().mean()),
                "surprise_rate_abs_gt_2": float((s.abs() > 2).mean()),
                "surprise_rate_abs_gt_3": float((s.abs() > 3).mean()),
                "positive_surprise_rate_gt_2": float((s > 2).mean()),
            })
        else:
            row.update({
                "surprise_mean": np.nan,
                "surprise_abs_mean": np.nan,
                "surprise_rate_abs_gt_2": np.nan,
                "surprise_rate_abs_gt_3": np.nan,
                "positive_surprise_rate_gt_2": np.nan,
            })

    return row

df = read_csv_safe(CSV_PATH)

# 如果没有 predicted_cost，但有 selected_candidate_mu，就补算
pred_col = find_col(df, ["predicted_cost"])
mu_col = find_col(df, ["selected_candidate_mu", "posterior_mu"])
if pred_col is None and mu_col is not None:
    df["predicted_cost"] = -pd.to_numeric(df[mu_col], errors="coerce")

actual_col = find_col(df, ["actual_cost", "Eval_Cost", "eval_cost", "cost"])
pred_col = find_col(df, ["predicted_cost"])
err_col = find_col(df, ["prediction_error"])

if err_col is None and pred_col is not None and actual_col is not None:
    df["prediction_error"] = pd.to_numeric(df[actual_col], errors="coerce") - pd.to_numeric(df[pred_col], errors="coerce")

stages = ["all", "first50", "first100", "101_200", "201_350", "tail100", "last50"]
summary = pd.DataFrame([summarize_stage(df, s) for s in stages])
summary.to_csv(OUTDIR / "prediction_gap_stage_summary.csv", index=False, encoding="utf-8-sig")

# 导出有效预测行，方便看具体哪些轮次低估严重
pred_col = find_col(df, ["predicted_cost"])
actual_col = find_col(df, ["actual_cost", "Eval_Cost", "eval_cost", "cost"])
err_col = find_col(df, ["prediction_error"])
sur_col = find_col(df, ["surprise"])
iter_col = find_col(df, ["Iteration", "iter", "round", "bo_iter"])

cols = []
for c in [iter_col, actual_col, pred_col, err_col, sur_col,
          find_col(df, ["Avg_Delay", "avg_delay"]),
          find_col(df, ["Backlog", "backlog"]),
          find_col(df, ["unfinished_end", "Unfinished_End"]),
          find_col(df, ["selected_candidate_source"]),
          find_col(df, ["cbo_tr_radius", "tr_radius"]),
          find_col(df, ["beta_eff"])]:
    if c is not None and c not in cols:
        cols.append(c)

valid_df = df[cols].copy() if cols else df.copy()
if err_col is not None:
    valid_df["_abs_prediction_error"] = pd.to_numeric(df[err_col], errors="coerce").abs()
    valid_df = valid_df.sort_values("_abs_prediction_error", ascending=False)

valid_df.head(100).to_csv(OUTDIR / "top100_prediction_errors.csv", index=False, encoding="utf-8-sig")

print("Done.")
print("Input:", CSV_PATH)
print("Output:", OUTDIR)
print(summary.to_string(index=False))