# -*- coding: utf-8 -*-
"""
Used Car Price Interval Model (no-clean version)

假设已输入的是清洗后的 Full_dataset_cleaned.csv：
- 类型 / 缺失值 / 极值裁剪已经在脚本外完成
- 本脚本只做：
    * 特征工程（车龄、里程派生、频数特征等）
    * CatBoost 分位数模型 (P10/P50/P90) —— log-space 训练
    * 年度市场系数（Hedonic + EWMA）
    * 时间感知非对称 CQR 校准
    * 输出 3 个 CatBoost 模型 + cqr_meta.json
- 预留 train_deep_block(...) 接口给 CNN / Transformer 使用
"""

import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from catboost import CatBoostRegressor, Pool
from joblib import dump
from tqdm import tqdm

# =========================
# 固定路径（Windows 绝对路径）
# =========================
CSV_PATH = r"C:\YOUR\PATH\Full_dataset_cleaned.csv"     # TODO: 改成你自己的路径
SAVE_DIR = r"C:\YOUR\PATH\used_car_price_super"         # TODO: 改成你自己的路径

# 列名，对齐 Full_dataset
TARGET_COL = "Price"
DATE_COL   = "listing_date"     # 如果数据里没有这个列，会自动退化为基于 Brand/Model/Year 的分组切分
BRAND_COL  = "Brand"
MODEL_COL  = "Model"
YEAR_COL   = "Year"
AGE_COL    = "Age"
MILE_COL   = "Kilometer"
HP_COL     = "Max Power"
ENGINE_COL = "Engine"
GEAR_COL   = "Transmission"
FUEL_COL   = "Fuel Type"
SEATS_COL  = "Seats"

# 训练与评估配置
RANDOM_STATE      = 42
TEST_SIZE_BY_TIME = 0.2
N_SPLITS          = 5
ALPHA             = 0.20  # 目标 80% 置信区间（即 P10~P90）

# CatBoost 参数（稍偏保守，训练时间长一点没关系）
CATBOOST_PARAMS = dict(
    depth=8,
    learning_rate=0.035,
    l2_leaf_reg=6.0,
    loss_function="Quantile:alpha={alpha}",
    iterations=5000,
    random_seed=RANDOM_STATE,
    border_count=254,
    verbose=100,          # 打印内部迭代进度
    thread_count=-1,
    od_type='Iter',
    od_wait=300,
    subsample=0.9,
    rsm=0.9,
    bootstrap_type='Bernoulli'
)

# ================= 工具函数 =================
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def safe_log1p(x):
    x = np.maximum(x, 0)
    return np.log1p(x)

def pick_existing_columns(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]

def make_age(df: pd.DataFrame) -> pd.Series:
    """车龄（双保险：即使 Age 列有一点问题，也会重新算一遍）"""
    if AGE_COL in df.columns:
        age = pd.to_numeric(df[AGE_COL], errors="coerce")
    elif YEAR_COL in df.columns:
        current_year = datetime.now().year
        age = current_year - pd.to_numeric(df[YEAR_COL], errors="coerce")
    else:
        age = pd.Series(np.nan, index=df.index)
    return age.where(age >= 0, np.nan)

def age_bin(a):
    if pd.isna(a):
        return "age:Unknown"
    if a < 3:
        return "age:0-3"
    if a < 8:
        return "age:3-8"
    return "age:8+"

def finite_sample_quantile(scores, q: float) -> float:
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0:
        return 0.0
    rank = int(math.ceil((n + 1) * q)) - 1
    rank = min(max(rank, 0), n - 1)
    return float(s[rank])

def evaluate_point(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(
        np.abs((y_true - y_pred) / np.maximum(1e-8, y_true))
    ) * 100.0
    return mae, mape

# ---------- period & market ----------
def get_period_series(df: pd.DataFrame) -> pd.Series:
    """period：优先用 listing_date 的年份，否则用 Year"""
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        p = d.dt.year.astype("Int64").astype(str)
    elif YEAR_COL in df.columns:
        p = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64").astype(str)
    else:
        p = pd.Series(["Unknown"] * len(df))
    return p.fillna("Unknown")

def make_period_bin_from_series(period_series: pd.Series) -> pd.Series:
    """period_bin：两年一组，比如 2014-2015、2016-2017 ..."""
    yy = pd.to_numeric(period_series, errors="coerce")
    out = []
    for v in yy:
        if np.isnan(v):
            out.append("Unknown")
        else:
            lo = int(v) // 2 * 2
            out.append(f"{lo}-{lo+1}")
    return pd.Series(out, index=period_series.index)

def compute_residual_by_period(y_log_true, y_log_pred, period_series, agg="median"):
    r = y_log_true - y_log_pred
    tmp = pd.DataFrame({"r": r, "period": period_series})
    if agg == "median":
        R = tmp.groupby("period")["r"].median()
    else:
        R = tmp.groupby("period")["r"].mean()
    R = R - R.mean()
    return R.to_dict()

def fit_time_dummy(R_dict):
    """把按年份的 log 残差映射为 (gamma_t, M_t_raw)"""
    gamma = dict(R_dict)
    M = {k: float(np.exp(v)) for k, v in gamma.items()}
    return gamma, M

def smooth_market_index(gamma_dict, alpha=0.25):
    """对年度残差做 EWMA 平滑，返回平滑后的 multiplicative index"""
    keys = sorted([k for k in gamma_dict.keys() if str(k).isdigit()])
    sm, last = {}, 0.0
    for k in keys:
        g = float(gamma_dict[k])
        last = alpha * g + (1.0 - alpha) * last
        sm[k] = last
    # 如果有非数字 key，就直接原样 exponent
    for k in gamma_dict:
        if k not in sm:
            sm[k] = float(gamma_dict[k])
    return {k: float(np.exp(v)) for k, v in sm.items()}

# ================= 特征工程 =================
def build_feature_df(df_raw: pd.DataFrame):
    """
    输入：已经清洗好的原始 DataFrame
    输出：
        X: 特征矩阵
        y: 目标 (Price)
        cat_cols: 类别特征列名列表
    """
    df = df_raw.copy()

    # 车龄（用 Age / Year 双保险）
    df["car_age"] = make_age(df)

    # 数值派生
    if MILE_COL in df.columns:
        df[MILE_COL] = pd.to_numeric(df[MILE_COL], errors="coerce")
        df["log1p_mileage"] = safe_log1p(df[MILE_COL])

    if (MILE_COL in df.columns) and ("car_age" in df.columns):
        age_eps = df["car_age"].replace(0, 0.25)
        df["avg_km_per_year"] = df[MILE_COL] / age_eps

    if (HP_COL in df.columns) and (GEAR_COL in df.columns):
        df[HP_COL] = pd.to_numeric(df[HP_COL], errors="coerce")
        is_auto = (df[GEAR_COL] == "Automatic").astype(int)
        df["hp_x_auto"] = df[HP_COL] * is_auto

    if (HP_COL in df.columns) and ("avg_km_per_year" in df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["hp_div_avgkm"] = df[HP_COL] / np.where(
                df["avg_km_per_year"] > 0, df["avg_km_per_year"], np.nan
            )

    if (HP_COL in df.columns) and (ENGINE_COL in df.columns):
        df[ENGINE_COL] = pd.to_numeric(df[ENGINE_COL], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["power_per_cc"] = df[HP_COL] / np.where(
                df[ENGINE_COL] > 0, df[ENGINE_COL], np.nan
            )

    if (ENGINE_COL in df.columns) and (SEATS_COL in df.columns):
        df[SEATS_COL] = pd.to_numeric(df[SEATS_COL], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["cc_per_seat"] = df[ENGINE_COL] / np.where(
                df[SEATS_COL] > 0, df[SEATS_COL], np.nan
            )

    # 频数 / 稀缺度特征
    if BRAND_COL in df.columns:
        cnt_brand = df[BRAND_COL].astype(str).map(
            df[BRAND_COL].astype(str).value_counts()
        )
        df["brand_count"] = cnt_brand
    if MODEL_COL in df.columns:
        cnt_model = df[MODEL_COL].astype(str).map(
            df[MODEL_COL].astype(str).value_counts()
        )
        df["model_count"] = cnt_model
    if (BRAND_COL in df.columns) and (MODEL_COL in df.columns):
        bm = df[BRAND_COL].astype(str) + "§" + df[MODEL_COL].astype(str)
        cnt_bm = bm.map(bm.value_counts())
        df["brand_model_count"] = cnt_bm

    # 候选列
    numeric_candidates = pick_existing_columns(
        df,
        [
            YEAR_COL,
            AGE_COL,
            MILE_COL,
            HP_COL,
            ENGINE_COL,
            SEATS_COL,
            "car_age",
            "log1p_mileage",
            "avg_km_per_year",
            "hp_x_auto",
            "hp_div_avgkm",
            "power_per_cc",
            "cc_per_seat",
            "brand_count",
            "model_count",
            "brand_model_count",
        ],
    )
    categorical_candidates = pick_existing_columns(
        df, [BRAND_COL, MODEL_COL, GEAR_COL, FUEL_COL]
    )

    # 目标
    assert TARGET_COL in df.columns, f"CSV 缺少目标列: {TARGET_COL}"
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 组装 X
    X = pd.DataFrame(index=df.index)

    # 数值列：中位数填补 + 缺失指示
    for col in numeric_candidates:
        col_num = pd.to_numeric(df[col], errors="coerce")
        miss_flag = col_num.isna().astype(int)
        med = np.nanmedian(col_num)
        X[col] = np.where(np.isnan(col_num), med, col_num)
        X[col + "_missing"] = miss_flag

    # 类别列
    for col in categorical_candidates:
        X[col] = df[col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})

    # period & bin
    X["period"] = get_period_series(df)
    X["period_bin"] = make_period_bin_from_series(X["period"])

    # 把 period / period_bin 也作为类别特征
    if "period" in X.columns:
        X["period"] = X["period"].astype(str)
    if "period_bin" in X.columns:
        X["period_bin"] = X["period_bin"].astype(str)

    cat_cols = categorical_candidates.copy()
    for c in ["period", "period_bin"]:
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)

    keep = ~y.isna()
    return (
        X.loc[keep].reset_index(drop=True),
        y.loc[keep].reset_index(drop=True),
        cat_cols,
    )

# =============== 切分 ===============
def time_based_split(df: pd.DataFrame, date_col: str, test_ratio=0.2):
    d = pd.to_datetime(df[date_col], errors="coerce")
    order = np.argsort(d.fillna(d.min()))
    n = len(df)
    cutoff = int((1.0 - test_ratio) * n)
    return order[:cutoff], order[cutoff:]

def build_groups_for_split(df: pd.DataFrame):
    parts = []
    if BRAND_COL in df.columns:
        parts.append(df[BRAND_COL].astype(str))
    if MODEL_COL in df.columns:
        parts.append(df[MODEL_COL].astype(str))
    if YEAR_COL in df.columns:
        parts.append(df[YEAR_COL].astype(str))
    if not parts:
        return None
    g = parts[0]
    for p in parts[1:]:
        g = g.str.cat(p, sep="__")
    return g

def group_based_split(df: pd.DataFrame, groups: pd.Series, n_splits=5, random_state=42):
    idx = np.arange(len(df))
    if groups is None or (hasattr(groups, "isna") and groups.isna().all()):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        tr, va = next(kf.split(idx))
        return tr, va
    else:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, va in gkf.split(idx, groups=groups):
            return tr, va

# =============== CatBoost 训练 ===============
def train_single_quantile_model(X_tr, y_tr_log, X_va, y_va_log, alpha: float):
    """
    训练单个分位数模型（log-space），内部自动识别 dtype=object 的列为类别特征。
    """
    params = CATBOOST_PARAMS.copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(**params)

    # 1) 统一识别类别列（以 dtype=object 为准）
    cats = [c for c in X_tr.columns if X_tr[c].dtype == "object"]

    # 2) 强制类别列为字符串
    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)

    # 3) 入池
    train_pool = Pool(X_tr, y_tr_log, cat_features=cats)
    val_pool   = Pool(X_va, y_va_log, cat_features=cats)

    print(f"\n[alpha={alpha:.2f}] Cat features used: {cats}")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=100)
    preds_log = model.predict(val_pool)
    return model, preds_log

def train_all_quantiles_with_progress(X_tr, y_tr_log, X_va, y_va_log):
    """
    三个分位数模型 P10/P50/P90，以 tqdm 显示进度条。
    """
    quantiles = [0.10, 0.50, 0.90]
    models = {}
    val_preds = {}
    for alpha in tqdm(quantiles, desc="Training CatBoost quantile models"):
        m, p = train_single_quantile_model(
            X_tr.copy(), y_tr_log, X_va.copy(), y_va_log, alpha
        )
        models[alpha] = m
        val_preds[alpha] = p
    return models, val_preds

# =============== 时间感知 非对称 CQR ===============
def build_group_keys_for_cqr(df_va: pd.DataFrame):
    a = (
        df_va["car_age"].apply(age_bin)
        if "car_age" in df_va.columns
        else pd.Series(["age:Unknown"] * len(df_va))
    )
    f = (
        df_va[FUEL_COL]
        if FUEL_COL in df_va.columns
        else pd.Series(["fuel:Unknown"] * len(df_va))
    )
    t = (
        df_va[GEAR_COL]
        if GEAR_COL in df_va.columns
        else pd.Series(["gear:Unknown"] * len(df_va))
    )
    pb = (
        df_va["period_bin"]
        if "period_bin" in df_va.columns
        else pd.Series(["Unknown"] * len(df_va))
    )
    key = (
        a.astype(str)
        .str.cat(f.astype(str), sep="|")
        .str.cat(t.astype(str), sep="|")
        .str.cat(pb.astype(str), sep="|")
    )
    return key

def cqr_asymmetric_global_and_group(
    y_va, p10_m, p90_m, group_keys, min_group_size=120
):
    """
    非对称 CQR：对 (p10_m, p90_m) 做蒙德里安式校准（总体 + 分组）。
    这里用的是已经乘上市场系数后的 p10_m / p90_m。
    """
    y_va = np.asarray(y_va, dtype=float)
    p10_m = np.asarray(p10_m, dtype=float)
    p90_m = np.asarray(p90_m, dtype=float)

    s_lo_all = p10_m - y_va  # 下尾偏差
    s_hi_all = y_va - p90_m  # 上尾偏差

    q_lo_global = finite_sample_quantile(s_lo_all, 1.0 - ALPHA)
    q_hi_global = finite_sample_quantile(s_hi_all, 1.0 - ALPHA)

    q_lo_groups, q_hi_groups = {}, {}
    uniq = pd.Series(group_keys).unique()
    for g in uniq:
        idx = (group_keys == g)
        if idx.sum() < min_group_size:
            continue
        q_lo_groups[g] = finite_sample_quantile(s_lo_all[idx], 1.0 - ALPHA)
        q_hi_groups[g] = finite_sample_quantile(s_hi_all[idx], 1.0 - ALPHA)

    return (q_lo_global, q_hi_global), (q_lo_groups, q_hi_groups)

def apply_asymmetric_cqr(
    p10_m, p90_m, group_keys, qlo_global, qhi_global, qlo_groups, qhi_groups
):
    p10_m = np.asarray(p10_m, dtype=float)
    p90_m = np.asarray(p90_m, dtype=float)

    qlo = np.array([qlo_groups.get(k, qlo_global) for k in group_keys])
    qhi = np.array([qhi_groups.get(k, qhi_global) for k in group_keys])

    lo = p10_m - qlo
    hi = p90_m + qhi
    return lo, hi

# =============== 预留的 CNN / Transformer 模块 ===============
def train_deep_block(X_tr, y_tr, X_va, y_va):
    """
    预留接口：
    - 将来可以在这里塞入 CNN / Transformer 等深度模型；
    - 推荐直接使用原始 df_raw 或你们另外构造的序列 / 图像特征；
    - 返回值可以是任意 dict，后面会被写入 meta["deep_block"]["info"]。

    当前版本：不做任何事，直接返回 None。
    """
    return None

# =============== 主流程 ===============
if __name__ == "__main__":
    ensure_dir(SAVE_DIR)

    # 1) 读数据 + 特征工程
    df_raw = pd.read_csv(CSV_PATH)
    X_all, y_all, cat_cols = build_feature_df(df_raw)

    # 2) 切分训练 / 验证
    if DATE_COL in df_raw.columns:
        tr_idx, va_idx = time_based_split(
            df_raw.loc[y_all.index], DATE_COL, TEST_SIZE_BY_TIME
        )
        split_mode = "time"
    else:
        groups = build_groups_for_split(df_raw.loc[y_all.index])
        tr_idx, va_idx = group_based_split(
            df_raw.loc[y_all.index], groups, N_SPLITS, RANDOM_STATE
        )
        split_mode = "group" if groups is not None else "kfold"

    X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
    X_va = X_all.iloc[va_idx].reset_index(drop=True)
    y_va = y_all.iloc[va_idx].reset_index(drop=True)

    # 3) log-space 训练
    y_tr_log = np.log(np.maximum(1e-6, y_tr.values))
    y_va_log = np.log(np.maximum(1e-6, y_va.values))

    print("Training CatBoost Quantile models (P10/P50/P90)...")
    models, val_log_preds = train_all_quantiles_with_progress(
        X_tr, y_tr_log, X_va, y_va_log
    )

    m_p10 = models[0.10]
    m_p50 = models[0.50]
    m_p90 = models[0.90]
    va_p10_log = val_log_preds[0.10]
    va_p50_log = val_log_preds[0.50]
    va_p90_log = val_log_preds[0.90]

    # 4) log -> 原尺度
    va_p10 = np.exp(va_p10_log)
    va_p50 = np.exp(va_p50_log)
    va_p90 = np.exp(va_p90_log)

    # ---- 市场系数（按 period）----
    period_va = X_va["period"]
    R_t = compute_residual_by_period(y_va_log, va_p50_log, period_va, agg="median")
    gamma_t, M_t_raw = fit_time_dummy(R_t)
    M_t_smooth = smooth_market_index(gamma_t, alpha=0.25)

    def map_M(per):
        per = str(per)
        if per in M_t_smooth:
            return M_t_smooth[per]
        if per in M_t_raw:
            return M_t_raw[per]
        return 1.0

    M_vec = period_va.map(map_M).astype(float).values
    va_p10_m = va_p10 * M_vec
    va_p50_m = va_p50 * M_vec
    va_p90_m = va_p90 * M_vec

    # ---- 时间感知 非对称 CQR ----
    group_keys = build_group_keys_for_cqr(X_va)
    (q_lo_g, q_hi_g), (q_lo_groups, q_hi_groups) = \
        cqr_asymmetric_global_and_group(
            y_va.values, va_p10_m, va_p90_m, group_keys, min_group_size=120
        )
    cqr_low, cqr_high = apply_asymmetric_cqr(
        va_p10_m, va_p90_m, group_keys, q_lo_g, q_hi_g, q_lo_groups, q_hi_groups
    )

    # 5) 评估
    mae, mape = evaluate_point(y_va.values, va_p50_m)
    coverage = np.mean(
        (y_va.values >= cqr_low) & (y_va.values <= cqr_high)
    ) * 100.0
    avg_width = np.mean(cqr_high - cqr_low)

    print("\n=== Validation Report (Overall, with Market Index + asym CQR) ===")
    print(f"Samples: {len(y_va)} | Split: {split_mode}")
    print(f"P50 MAE:  {mae:,.2f}")
    print(f"P50 MAPE: {mape:.2f}%")
    print(f"CQR {int((1-ALPHA)*100)}% Coverage: {coverage:.2f}%")
    print(f"CQR Avg Interval Width: {avg_width:,.2f}")

    # 6) 按区间宽度比做 selective 报告（可选）
    print("\n=== Selective Pricing Report (by interval width ratio) ===")
    width = cqr_high - cqr_low
    wr = width / np.maximum(1e-6, va_p50_m)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mask = wr <= thr
        if not mask.any():
            continue
        mae_thr, mape_thr = evaluate_point(y_va.values[mask], va_p50_m[mask])
        print(
            f"width_ratio <= {thr:.2f}: "
            f"n={mask.sum():4d}, MAPE={mape_thr:6.2f}%"
        )

    # 7) 预留 CNN / Transformer 模块（当前不做任何事）
    deep_info = train_deep_block(X_tr, y_tr.values, X_va, y_va.values)

    # 8) 保存模型与 meta
    ensure_dir(SAVE_DIR)
    dump(m_p10, os.path.join(SAVE_DIR, "catboost_p10.joblib"))
    dump(m_p50, os.path.join(SAVE_DIR, "catboost_p50.joblib"))
    dump(m_p90, os.path.join(SAVE_DIR, "catboost_p90.joblib"))

    meta = {
        "columns": list(X_tr.columns),
        "categorical_cols": [c for c in cat_cols if c in X_tr.columns],
        "categorical_indices": [
            X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns
        ],
        "alpha": ALPHA,
        "market_index": {
            "period_unit": "year",
            "gamma_t": gamma_t,
            "M_t_raw": M_t_raw,
            "M_t_smooth": M_t_smooth,
            "ewma_alpha": 0.25,
        },
        "cqr_after_market": {
            "type": "asymmetric",
            "q_lo_global": q_lo_g,
            "q_hi_global": q_hi_g,
            "q_lo_groups": q_lo_groups,
            "q_hi_groups": q_hi_groups,
            "group_key_def": "age_bin(car_age)|fuel_type|transmission|period_bin",
        },
        "external_ts_blend": {
            "enable": False,
            "blend_weight": 0.5,
            "file": os.path.join(SAVE_DIR, "external_market_forecast.json"),
        },
        "target": TARGET_COL,
        "date_col": DATE_COL if DATE_COL in df_raw.columns else None,
        "split": {
            "mode": split_mode,
            "train_size": int(len(y_tr)),
            "valid_size": int(len(y_va)),
        },
        # 假定这几列已经在数据清洗阶段做过 Winsorize
        "winsorized_cols": [
            c for c in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL]
            if c in df_raw.columns
        ],
        "deep_block": {
            "enabled": deep_info is not None,
            "info": deep_info,
        },
    }

    with open(
        os.path.join(SAVE_DIR, "cqr_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModels & meta saved to: {SAVE_DIR}")
