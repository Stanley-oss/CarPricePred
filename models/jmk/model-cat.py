# -*- coding: utf-8 -*-
"""
Used Car Price Interval Model (v3):
CatBoost 分位数 + 市场系数(年度/季度) + 时间感知 Mondrian-CQR + Selective 报表
并预留“外部时间序列融合”接口

基线来自上一版：
- CatBoost 分位数训练与特征工程、切分与 CQR 架构保持一致
- 参考：原模型头部与参数/工具函数结构（请见历史版本） 
"""

import os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold, KFold

# =========================
# 固定路径（Windows 绝对路径）
# =========================
CSV_PATH = r"C:\data\used_cars.csv"
SAVE_DIR = r"C:\models\used_car_price_v3"

# 列名对齐（与你的数据一致）
TARGET_COL = "price"
DATE_COL   = "listing_date"   # 没有也没关系；若无则用 YEAR_COL 代表 period

BRAND_COL  = "brand"
MODEL_COL  = "model"
YEAR_COL   = "year"
AGE_COL    = "age"
MILE_COL   = "milage"         # 注意拼写
HP_COL     = "max_power"
ENGINE_COL = "engine"
GEAR_COL   = "transmission"
FUEL_COL   = "fuel_type"
SEATS_COL  = "seats"

# 训练与评估配置
RANDOM_STATE = 42
TEST_SIZE_BY_TIME = 0.2
N_SPLITS = 5
ALPHA = 0.20  # 目标区间 80%（P10~P90）

# CatBoost 超参（与旧版一致，已验证稳定）
CATBOOST_PARAMS = dict(
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Quantile:alpha={alpha}",
    iterations=3000,
    random_seed=RANDOM_STATE,
    border_count=254,
    verbose=False,
    thread_count=-1,
    od_type='Iter',
    od_wait=200
)

# =============== 工具函数（与旧版一致 + 少量新增） ===============
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    return float(m.group(0)) if m else np.nan

def winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi), lo, hi

def safe_log1p(x):
    x = np.maximum(x, 0)
    return np.log1p(x)

def standardize_enum(series, mapping, default="Unknown"):
    s = series.astype(str).str.strip().str.lower()
    return s.map(mapping).fillna(default)

def pick_existing_columns(df, cols):
    return [c for c in cols if c in df.columns]

def make_age(df):
    if AGE_COL in df.columns:
        age = pd.to_numeric(df[AGE_COL], errors="coerce")
    elif YEAR_COL in df.columns:
        current_year = datetime.now().year
        age = current_year - pd.to_numeric(df[YEAR_COL], errors="coerce")
    else:
        age = pd.Series(np.nan, index=df.index)
    return age.where(age >= 0, np.nan)

def age_bin(a):
    if pd.isna(a): return "age:Unknown"
    if a < 3: return "age:0-3"
    if a < 8: return "age:3-8"
    return "age:8+"

def build_groups_for_split(df):
    parts = []
    if BRAND_COL in df.columns: parts.append(df[BRAND_COL].astype(str))
    if MODEL_COL in df.columns: parts.append(df[MODEL_COL].astype(str))
    if YEAR_COL  in df.columns: parts.append(df[YEAR_COL].astype(str))
    if not parts: return None
    g = parts[0]
    for p in parts[1:]:
        g = g.str.cat(p, sep="__")
    return g

def finite_sample_quantile(scores, q):
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0: return 0.0
    rank = int(math.ceil((n + 1) * q)) - 1
    rank = min(max(rank, 0), n - 1)
    return float(s[rank])

def evaluate_point(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, y_true))) * 100.0
    return mae, mape

# ---------- 新增：period 相关 ----------
def get_period_series(df):
    """
    返回字符串 period：优先 DATE_COL 的年份；否则 YEAR_COL；都没有则 'Unknown'
    """
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        p = d.dt.year.astype("Int64").astype(str)
    elif YEAR_COL in df.columns:
        p = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64").astype(str)
    else:
        p = pd.Series(["Unknown"]*len(df))
    return p.fillna("Unknown")

def compute_residual_by_period(y_log_true, y_log_pred, period_series, agg="median"):
    r = y_log_true - y_log_pred
    df = pd.DataFrame({"r": r, "period": period_series})
    if agg == "median":
        R = df.groupby("period")["r"].median()
    else:
        R = df.groupby("period")["r"].mean()
    R = R - np.average(R, weights=None)  # 中心化，基期≈0
    return R.to_dict()  # {period: residual}

def fit_time_dummy(R_dict):
    """
    最简时间虚拟项：γ_t := R_t（已中心化）
    返回 {period: gamma_t} 与对应的指数 M_t = exp(gamma_t)
    """
    gamma = dict(R_dict)
    M = {k: float(np.exp(v)) for k, v in gamma.items()}
    return gamma, M

def smooth_market_index(gamma_dict, method="EWMA", alpha=0.3):
    """
    对 γ_t 做时间平滑，返回 M_t 平滑版
    假定 period 为 'YYYY' 可排序；异常值忽略
    """
    keys = sorted([k for k in gamma_dict.keys() if k.isdigit()])
    sm = {}
    last = 0.0
    for k in keys:
        g = float(gamma_dict[k])
        last = alpha * g + (1 - alpha) * last
        sm[k] = last
    M_smooth = {k: float(np.exp(v)) for k, v in sm.items()}
    return M_smooth

# =============== 特征工程（延续旧版） ===============
def build_feature_df(df_raw):
    df = df_raw.copy()

    # 单位清洗
    for col in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL, YEAR_COL, AGE_COL, SEATS_COL]:
        if col in df.columns:
            df[col] = df[col].apply(to_number)

    # 类别标准化
    if GEAR_COL in df.columns:
        df[GEAR_COL] = standardize_enum(
            df[GEAR_COL],
            {"a":"Automatic","auto":"Automatic","automatic":"Automatic",
             "m":"Manual","man":"Manual","manual":"Manual"},
            default="Unknown"
        )
    if FUEL_COL in df.columns:
        df[FUEL_COL] = standardize_enum(
            df[FUEL_COL],
            {"petrol":"Petrol","gasoline":"Petrol",
             "diesel":"Diesel",
             "cng":"Other","lpg":"Other","hybrid":"Other","electric":"Other","other":"Other"},
            default="Other"
        )

    # 极值裁剪
    for col in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL]:
        if col in df.columns:
            df[col], _, _ = winsorize_series(df[col], 0.01, 0.99)

    # 车龄
    df["car_age"] = make_age(df)

    # 数值派生
    if MILE_COL in df.columns:
        df["log1p_mileage"] = safe_log1p(df[MILE_COL])
    if (MILE_COL in df.columns) and ("car_age" in df.columns):
        age_eps = df["car_age"].replace(0, 0.25)
        df["avg_km_per_year"] = df[MILE_COL] / age_eps
    if (HP_COL in df.columns) and (GEAR_COL in df.columns):
        is_auto = (df[GEAR_COL] == "Automatic").astype(int)
        df["hp_x_auto"] = df[HP_COL] * is_auto
    if (HP_COL in df.columns) and ("avg_km_per_year" in df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["hp_div_avgkm"] = df[HP_COL] / np.where(df["avg_km_per_year"] > 0, df["avg_km_per_year"], np.nan)
    if (HP_COL in df.columns) and (ENGINE_COL in df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["power_per_cc"] = df[HP_COL] / np.where(df[ENGINE_COL] > 0, df[ENGINE_COL], np.nan)
    if (ENGINE_COL in df.columns) and (SEATS_COL in df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["cc_per_seat"] = df[ENGINE_COL] / np.where(df[SEATS_COL] > 0, df[SEATS_COL], np.nan)

    # 候选列
    numeric_candidates = pick_existing_columns(df, [
        YEAR_COL, AGE_COL, MILE_COL, HP_COL, ENGINE_COL, SEATS_COL,
        "car_age", "log1p_mileage", "avg_km_per_year",
        "hp_x_auto", "hp_div_avgkm", "power_per_cc", "cc_per_seat"
    ])
    categorical_candidates = pick_existing_columns(df, [
        BRAND_COL, MODEL_COL, GEAR_COL, FUEL_COL
    ])

    # 目标
    assert TARGET_COL in df.columns, f"CSV必须包含目标列: {TARGET_COL}"
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 组装 X
    X = pd.DataFrame(index=df.index)
    for col in numeric_candidates:
        col_num = pd.to_numeric(df[col], errors="coerce")
        miss_flag = col_num.isna().astype(int)
        med = np.nanmedian(col_num)
        X[col] = np.where(np.isnan(col_num), med, col_num)
        X[col + "_missing"] = miss_flag
    for col in categorical_candidates:
        col_cat = df[col].astype(str)
        X[col] = col_cat.replace({"nan":"Unknown","None":"Unknown"})

    # period 字段（年）
    X["period"] = get_period_series(df)

    cat_cols = categorical_candidates.copy()
    # 丢掉 y 缺失
    keep = ~y.isna()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), cat_cols

# =============== 切分（与旧版一致） ===============
def time_based_split(df, date_col, test_ratio=0.2):
    d = pd.to_datetime(df[date_col], errors="coerce")
    order = np.argsort(d.fillna(d.min()))
    n = len(df)
    cutoff = int((1.0 - test_ratio) * n)
    return order[:cutoff], order[cutoff:]

def group_based_split(df, groups, n_splits=5, random_state=42):
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
def cat_idx_from_cols(X, cat_cols):
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

def train_quantile_model(X_tr, y_tr, X_va, alpha, cat_idx):
    params = CATBOOST_PARAMS.copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(**params)
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx if cat_idx else None)
    val_pool   = Pool(X_va, y_va=None, cat_features=cat_idx if cat_idx else None)
    model.fit(train_pool)
    preds = model.predict(val_pool)
    return model, preds

# =============== CQR（时间感知 Mondrian） ===============
def build_group_keys_for_cqr(df_va):
    # 年龄段 + 燃油 + 变速箱 + period（时间感知！）
    a = df_va["car_age"].apply(age_bin) if "car_age" in df_va.columns else pd.Series(["age:Unknown"]*len(df_va))
    f = df_va[FUEL_COL] if FUEL_COL in df_va.columns else pd.Series(["fuel:Unknown"]*len(df_va))
    t = df_va[GEAR_COL] if GEAR_COL in df_va.columns else pd.Series(["gear:Unknown"]*len(df_va))
    p = df_va["period"] if "period" in df_va.columns else pd.Series(["period:Unknown"]*len(df_va))
    key = a.astype(str).str.cat(f.astype(str), sep="|").str.cat(t.astype(str), sep="|").str.cat(p.astype(str), sep="|")
    return key

def cqr_global_and_group(y_va, p10, p90, group_keys, min_group_size=50):
    s_all = np.maximum(p10 - y_va, y_va - p90)
    q_hat_global = finite_sample_quantile(s_all, 1.0 - ALPHA)
    q_hat_groups = {}
    for g in pd.Series(group_keys).unique():
        idx = (group_keys == g).values
        n_g = int(idx.sum())
        if n_g >= min_group_size:
            s_g = np.maximum(p10[idx] - y_va[idx], y_va[idx] - p90[idx])
            q_hat_groups[g] = finite_sample_quantile(s_g, 1.0 - ALPHA)
    return q_hat_global, q_hat_groups

def apply_cqr_with_groups(p10, p90, group_keys, q_hat_global, q_hat_groups):
    q = np.array([q_hat_groups.get(k, q_hat_global) for k in group_keys])
    return p10 - q, p90 + q

# =============== 主流程 ===============
if __name__ == "__main__":
    ensure_dir(SAVE_DIR)

    # 读取
    df_raw = pd.read_csv(CSV_PATH)
    X_all, y_all, cat_cols = build_feature_df(df_raw)

    # 切分（优先时间，否则 GroupKFold），与旧版一致
    if DATE_COL in df_raw.columns:
        tr_idx, va_idx = time_based_split(df_raw.loc[y_all.index], DATE_COL, TEST_SIZE_BY_TIME)
        split_mode = "time"
    else:
        groups = build_groups_for_split(df_raw.loc[y_all.index])
        tr_idx, va_idx = group_based_split(df_raw.loc[y_all.index], groups, N_SPLITS, RANDOM_STATE)
        split_mode = "group" if groups is not None else "kfold"

    X_tr, y_tr = X_all.iloc[tr_idx].reset_index(drop=True), y_all.iloc[tr_idx].reset_index(drop=True)
    X_va, y_va = X_all.iloc[va_idx].reset_index(drop=True), y_all.iloc[va_idx].reset_index(drop=True)

    # 对数空间训练与预测（承接旧版做法）
    y_tr_log = np.log(np.maximum(1e-6, y_tr))
    y_va_log = np.log(np.maximum(1e-6, y_va))

    cat_idx = cat_idx_from_cols(X_tr, cat_cols)

    print("Training CatBoost Quantile models (alpha=0.10/0.50/0.90)...")
    m_p10, va_p10_log = train_quantile_model(X_tr, y_tr_log, X_va, alpha=0.10, cat_idx=cat_idx)
    m_p50, va_p50_log = train_quantile_model(X_tr, y_tr_log, X_va, alpha=0.50, cat_idx=cat_idx)
    m_p90, va_p90_log = train_quantile_model(X_tr, y_tr_log, X_va, alpha=0.90, cat_idx=cat_idx)

    # 反变换
    va_p10 = np.exp(va_p10_log)
    va_p50 = np.exp(va_p50_log)
    va_p90 = np.exp(va_p90_log)

    # ---------- A：年度市场系数（享乐时间虚拟项） ----------
    # 在日志空间计算残差并按 period 聚合
    period_va = X_va["period"]
    R_t = compute_residual_by_period(y_va_log.values, va_p50_log, period_va, agg="median")
    gamma_t, M_t_raw = fit_time_dummy(R_t)
    M_t_smooth = smooth_market_index(gamma_t, method="EWMA", alpha=0.35)  # 可调

    # 给验证集每条样本找到对应的市场系数（优先平滑版，无则回退 raw，再无则 1.0）
    def map_M(per):
        if per in M_t_smooth: return M_t_smooth[per]
        if per in M_t_raw:    return M_t_raw[per]
        return 1.0
    M_vec = period_va.map(map_M).astype(float).values

    va_p10_m = va_p10 * M_vec
    va_p50_m = va_p50 * M_vec
    va_p90_m = va_p90 * M_vec

    # ---------- B：时间感知 Mondrian-CQR（加入 period） ----------
    group_keys = build_group_keys_for_cqr(X_va)  # 带 period
    q_hat_global, q_hat_groups = cqr_global_and_group(y_va.values, va_p10_m, va_p90_m, group_keys)
    cqr_low, cqr_high = apply_cqr_with_groups(va_p10_m, va_p90_m, group_keys, q_hat_global, q_hat_groups)

    # ---------- C：评估（总体 + 切片 + Selective 报表） ----------
    mae, mape = evaluate_point(y_va.values, va_p50_m)
    coverage = np.mean((y_va.values >= cqr_low) & (y_va.values <= cqr_high)) * 100.0
    avg_width = np.mean(cqr_high - cqr_low)

    print("\n=== Validation Report (Overall, with Market Index) ===")
    print(f"Samples: {len(y_va)} | Split: {split_mode}")
    print(f"P50 MAE:  {mae:,.2f}")
    print(f"P50 MAPE: {mape:.2f}%")
    print(f"CQR {int((1-ALPHA)*100)}% Coverage: {coverage:.2f}%")
    print(f"CQR Avg Interval Width: {avg_width:,.2f}")

    # 切片（fuel × transmission × age_bin × period）
    print("\n=== Slice Report (fuel × transmission × age_bin × period) ===")
    df_slice = pd.DataFrame({
        "y": y_va.values,
        "p50": va_p50_m,
        "lo": cqr_low,
        "hi": cqr_high,
        "key": group_keys.values
    })
    for k, g in df_slice.groupby("key"):
        n = len(g)
        cov = np.mean((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])) * 100.0
        w = np.mean(g["hi"] - g["lo"])
        mae_k, mape_k = evaluate_point(g["y"].values, g["p50"].values)
        print(f"{k:>45s} | n={n:4d} | MAPE={mape_k:6.2f}% | Cov={cov:6.2f}% | Width={w:,.2f}")

    # 创新②：Selective 报表（高置信自动定价）
    print("\n=== Selective Pricing Report (by interval width ratio) ===")
    width = cqr_high - cqr_low
    wr = width / np.maximum(1e-6, va_p50_m)  # 区间相对宽度
    for thr in [0.10, 0.15, 0.20, 0.25]:
        mask = wr <= thr
        if mask.sum() == 0:
            print(f"WR≤{thr:.2f}: none")
            continue
        mae_s, mape_s = evaluate_point(y_va.values[mask], va_p50_m[mask])
        print(f"WR≤{thr:.2f}: cover={mask.mean()*100:5.1f}% | MAE={mae_s:,.2f} | MAPE={mape_s:5.2f}%")

    # ---------- 保存模型与元数据 ----------
    dump(m_p10, os.path.join(SAVE_DIR, "catboost_p10.joblib"))
    dump(m_p50, os.path.join(SAVE_DIR, "catboost_p50.joblib"))
    dump(m_p90, os.path.join(SAVE_DIR, "catboost_p90.joblib"))

    meta = {
        "columns": list(X_tr.columns),
        "categorical_cols": [c for c in cat_cols if c in X_tr.columns],
        "categorical_indices": cat_idx,
        "alpha": ALPHA,

        # 市场指数
        "market_index": {
            "period_unit": "year",
            "gamma_t": gamma_t,         # 未平滑
            "M_t_raw": M_t_raw,         # e^{gamma}
            "M_t_smooth": M_t_smooth,   # EWMA 平滑后（默认线上用）
            "ewma_alpha": 0.35
        },

        # CQR（时间感知）
        "cqr_after_market": {
            "q_hat_global": q_hat_global,
            "q_hat_groups": q_hat_groups,
            "group_key_def": "age_bin(car_age)|fuel_type|transmission|period"
        },

        # 外部时间序列融合接口（占位；线上可覆盖）
        "external_ts_blend": {
            "enable": False,
            "blend_weight": 0.5,   # 0~1：外部系数的权重
            "file": os.path.join(SAVE_DIR, "external_market_forecast.json")
            # 期望文件格式：{"period_unit":"year","M_t":{"2026":1.03,"2027":0.98,...}}
        },

        "target": TARGET_COL,
        "date_col": DATE_COL if DATE_COL in df_raw.columns else None,
        "split": {
            "mode": split_mode,
            "train_size": int(len(y_tr)),
            "valid_size": int(len(y_va))
        },
        "winsorized_cols": [c for c in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL] if c in df_raw.columns]
    }

    with open(os.path.join(SAVE_DIR, "cqr_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModels & meta saved to: {SAVE_DIR}")
    print("推理提示：先出 p10/p50/p90 → 乘市场系数 M_t（默认用平滑版）→ 按分组取 q_hat → 返回区间与点价。")

