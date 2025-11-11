# -*- coding: utf-8 -*-
"""
Used Car Price Interval Model (v4)
- CatBoost 分位数 (P10/P50/P90，log-space)
- 年度市场系数 Hedonic + EWMA(alpha=0.25)
- 时间感知 Mondrian-CQR -> 升级为【非对称CQR】(上下界分开校准)
- period -> period_bin（两年一组），min_group_size=120 + 逐级回退
- 频数/稀缺度特征: brand_count / model_count / brand_model_count
- 更保守的CatBoost正则
- Winsorize 0.5%~99.5%
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
CSV_PATH = r"C:\Users\Lenovo\Desktop\下载\Full_dataset.csv"   # 改成你的
SAVE_DIR = r"C:\Users\Lenovo\PycharmProjects\ml\app\used_car_price_v4"      # 改成你的

# 列名对齐（与你的数据一致）
TARGET_COL = "Price"
DATE_COL   = "listing_date"     # 没有就占位
BRAND_COL  = "Brand"
MODEL_COL  = "Model"
YEAR_COL   = "Year"
AGE_COL    = "Age"
MILE_COL   = "Kilometer"
HP_COL     = "Max Power"
ENGINE_COL = "Engine"
GEAR_COL   = "Transmission"
FUEL_COL   = "Fuel Type"        # 如果你用的是 Fuel Type.1 就改成 "Fuel Type.1"
SEATS_COL  = "Seats"

# 训练与评估配置
RANDOM_STATE = 42
TEST_SIZE_BY_TIME = 0.2
N_SPLITS = 5
ALPHA = 0.20  # 目标80%区间

# CatBoost 参数（更保守，早停配 eval_set）
CATBOOST_PARAMS = dict(
    depth=8,
    learning_rate=0.035,
    l2_leaf_reg=6.0,
    loss_function="Quantile:alpha={alpha}",
    iterations=5000,
    random_seed=RANDOM_STATE,
    border_count=254,
    verbose=False,
    thread_count=-1,
    od_type='Iter',
    od_wait=300,
    subsample=0.9,
    rsm=0.9,
    bootstrap_type='Bernoulli'
)

# ================= 工具函数 =================
def ensure_dir(p):
    if not os.path.exists(p): os.makedirs(p)

def to_number(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    return float(m.group(0)) if m else np.nan

def winsorize_series(s, lower=0.005, upper=0.995):
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

# ---------- period & market ----------
def get_period_series(df):
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        p = d.dt.year.astype("Int64").astype(str)
    elif YEAR_COL in df.columns:
        p = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64").astype(str)
    else:
        p = pd.Series(["Unknown"]*len(df))
    return p.fillna("Unknown")

def make_period_bin_from_series(period_series):
    yy = pd.to_numeric(period_series, errors="coerce")
    out = []
    for v in yy:
        if np.isnan(v):
            out.append("Unknown")
        else:
            lo = int(v)//2*2
            out.append(f"{lo}-{lo+1}")
    return pd.Series(out, index=period_series.index)

def compute_residual_by_period(y_log_true, y_log_pred, period_series, agg="median"):
    r = y_log_true - y_log_pred
    df = pd.DataFrame({"r": r, "period": period_series})
    R = df.groupby("period")["r"].median() if agg=="median" else df.groupby("period")["r"].mean()
    R = R - R.mean()
    return R.to_dict()

def fit_time_dummy(R_dict):
    gamma = dict(R_dict)
    M = {k: float(np.exp(v)) for k, v in gamma.items()}
    return gamma, M

def smooth_market_index(gamma_dict, alpha=0.25):
    keys = sorted([k for k in gamma_dict.keys() if k.isdigit()])
    sm, last = {}, 0.0
    for k in keys:
        g = float(gamma_dict[k])
        last = alpha * g + (1 - alpha) * last
        sm[k] = last
    return {k: float(np.exp(v)) for k, v in sm.items()}

# ================= 特征工程 =================
def build_feature_df(df_raw):
    df = df_raw.copy()

    # 基本清洗
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
            {"petrol":"Petrol","gasoline":"Petrol","diesel":"Diesel",
             "cng":"Other","lpg":"Other","hybrid":"Other","electric":"Other","other":"Other"},
            default="Other"
        )

    # 极值裁剪（更温和 0.5%~99.5%）
    for col in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL]:
        if col in df.columns:
            df[col], _, _ = winsorize_series(df[col], 0.005, 0.995)

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

    # 频数/稀缺度特征
    if BRAND_COL in df.columns:
        cnt_brand = df[BRAND_COL].astype(str).map(df[BRAND_COL].astype(str).value_counts())
        df["brand_count"] = cnt_brand
    if MODEL_COL in df.columns:
        cnt_model = df[MODEL_COL].astype(str).map(df[MODEL_COL].astype(str).value_counts())
        df["model_count"] = cnt_model
    if (BRAND_COL in df.columns) and (MODEL_COL in df.columns):
        bm = (df[BRAND_COL].astype(str) + "§" + df[MODEL_COL].astype(str))
        cnt_bm = bm.map(bm.value_counts())
        df["brand_model_count"] = cnt_bm

    # 候选列
    numeric_candidates = pick_existing_columns(df, [
        YEAR_COL, AGE_COL, MILE_COL, HP_COL, ENGINE_COL, SEATS_COL,
        "car_age","log1p_mileage","avg_km_per_year","hp_x_auto",
        "hp_div_avgkm","power_per_cc","cc_per_seat",
        "brand_count","model_count","brand_model_count"
    ])
    categorical_candidates = pick_existing_columns(df, [BRAND_COL, MODEL_COL, GEAR_COL, FUEL_COL])

    # 目标
    assert TARGET_COL in df.columns, f"CSV缺少目标列: {TARGET_COL}"
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
        X[col] = df[col].astype(str).replace({"nan":"Unknown","None":"Unknown"})

    # period & bin
    X["period"] = get_period_series(df)
    X["period_bin"] = make_period_bin_from_series(X["period"])


    # 关键：把 period / period_bin 也作为类别特征
    if "period" in X.columns:
        X["period"] = X["period"].astype(str)
    if "period_bin" in X.columns:
        X["period_bin"] = X["period_bin"].astype(str)

    cat_cols = categorical_candidates.copy()
    for c in ["period", "period_bin"]:
        if c in X.columns and c not in cat_cols:
            cat_cols.append(c)


    cat_cols = categorical_candidates.copy()
    keep = ~y.isna()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), cat_cols

# =============== 切分 ===============
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

# =============== CatBoost 训练 ===============
def cat_idx_from_cols(X, cat_cols):
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

def train_quantile_model(X_tr, y_tr, X_va, y_va, alpha, cat_features=None):
    """
    稳妥做法：忽略外部传入的 cat_features，直接用 X_tr 中 dtype==object 的列作为类别列；
    并在入 Pool 前强制转为字符串，确保像 "2014-2015" 这样的 period_bin 被当作类别处理。
    """
    params = CATBOOST_PARAMS.copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(**params)

    # 1) 统一识别类别列（以 dtype=object 为准）
    cats = [c for c in X_tr.columns if X_tr[c].dtype == "object"]

    # 2) 强制类别列为字符串（包含 period / period_bin 等）
    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)

    # 3) 入池（用列名指定 cat_features）
    train_pool = Pool(X_tr, y_tr, cat_features=cats)
    val_pool   = Pool(X_va, y_va, cat_features=cats)

    # （可选）调试打印，确认 period_bin 在 cats 里
    print("Cat features used:", cats)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
    preds = model.predict(val_pool)
    return model, preds


# =============== 时间感知 非对称CQR ===============
def build_group_keys_for_cqr(df_va):
    a = df_va["car_age"].apply(age_bin) if "car_age" in df_va.columns else pd.Series(["age:Unknown"]*len(df_va))
    f = df_va[FUEL_COL] if FUEL_COL in df_va.columns else pd.Series(["fuel:Unknown"]*len(df_va))
    t = df_va[GEAR_COL] if GEAR_COL in df_va.columns else pd.Series(["gear:Unknown"]*len(df_va))
    pb = df_va["period_bin"] if "period_bin" in df_va.columns else pd.Series(["Unknown"]*len(df_va))
    key = a.astype(str).str.cat(f.astype(str), sep="|").str.cat(t.astype(str), sep="|").str.cat(pb.astype(str), sep="|")
    return key

def cqr_asymmetric_global_and_group(y_va, p10, p90, group_keys, min_group_size=120):
    s_lo_all = p10 - y_va   # 下尾偏差
    s_hi_all = y_va - p90   # 上尾偏差
    q_lo_global = finite_sample_quantile(s_lo_all, 1.0 - ALPHA)
    q_hi_global = finite_sample_quantile(s_hi_all, 1.0 - ALPHA)
    q_lo_groups, q_hi_groups = {}, {}
    uniq = pd.Series(group_keys).unique()
    for g in uniq:
        idx = (group_keys == g).values
        n_g = int(idx.sum())
        if n_g >= min_group_size:
            q_lo_groups[g] = finite_sample_quantile(s_lo_all[idx], 1.0 - ALPHA)
            q_hi_groups[g] = finite_sample_quantile(s_hi_all[idx], 1.0 - ALPHA)
    return (q_lo_global, q_hi_global), (q_lo_groups, q_hi_groups)

def apply_asymmetric_cqr(p10, p90, group_keys, qlo_global, qhi_global, qlo_groups, qhi_groups):
    qlo = np.array([qlo_groups.get(k, qlo_global) for k in group_keys])
    qhi = np.array([qhi_groups.get(k, qhi_global) for k in group_keys])
    lo  = p10 - qlo
    hi  = p90 + qhi
    return lo, hi

# =============== 主流程 ===============
if __name__ == "__main__":
    ensure_dir(SAVE_DIR)

    # 读数 & 特征
    df_raw = pd.read_csv(CSV_PATH)
    X_all, y_all, cat_cols = build_feature_df(df_raw)

    # 切分
    if DATE_COL in df_raw.columns:
        tr_idx, va_idx = time_based_split(df_raw.loc[y_all.index], DATE_COL, TEST_SIZE_BY_TIME)
        split_mode = "time"
    else:
        groups = build_groups_for_split(df_raw.loc[y_all.index])
        tr_idx, va_idx = group_based_split(df_raw.loc[y_all.index], groups, N_SPLITS, RANDOM_STATE)
        split_mode = "group" if groups is not None else "kfold"

    X_tr, y_tr = X_all.iloc[tr_idx].reset_index(drop=True), y_all.iloc[tr_idx].reset_index(drop=True)
    X_va, y_va = X_all.iloc[va_idx].reset_index(drop=True), y_all.iloc[va_idx].reset_index(drop=True)

    # log-space
    y_tr_log = np.log(np.maximum(1e-6, y_tr))
    y_va_log = np.log(np.maximum(1e-6, y_va))
    cat_idx = cat_idx_from_cols(X_tr, cat_cols)

    print("Training CatBoost Quantile models (P10/P50/P90)...")
    # cat_cols 已经包含 period / period_bin（确保 build_feature_df 里有把这两列 append 到 cat_cols）
    m_p10, va_p10_log = train_quantile_model(X_tr, y_tr_log, X_va, y_va_log, alpha=0.10)
    m_p50, va_p50_log = train_quantile_model(X_tr, y_tr_log, X_va, y_va_log, alpha=0.50)
    m_p90, va_p90_log = train_quantile_model(X_tr, y_tr_log, X_va, y_va_log, alpha=0.90)



    # 反变换
    va_p10 = np.exp(va_p10_log)
    va_p50 = np.exp(va_p50_log)
    va_p90 = np.exp(va_p90_log)

    # ---- 市场系数 ----
    period_va = X_va["period"]
    R_t = compute_residual_by_period(y_va_log.values, va_p50_log, period_va, agg="median")
    gamma_t, M_t_raw = fit_time_dummy(R_t)
    M_t_smooth = smooth_market_index(gamma_t, alpha=0.25)  # 更平滑

    def map_M(per):
        if per in M_t_smooth: return M_t_smooth[per]
        if per in M_t_raw:    return M_t_raw[per]
        return 1.0
    M_vec = period_va.map(map_M).astype(float).values

    va_p10_m = va_p10 * M_vec
    va_p50_m = va_p50 * M_vec
    va_p90_m = va_p90 * M_vec

    # ---- 非对称CQR（时间感知：period_bin）----
    group_keys = build_group_keys_for_cqr(X_va)
    (q_lo_g, q_hi_g), (q_lo_groups, q_hi_groups) = cqr_asymmetric_global_and_group(
        y_va.values, va_p10_m, va_p90_m, group_keys, min_group_size=120
    )
    cqr_low, cqr_high = apply_asymmetric_cqr(
        va_p10_m, va_p90_m, group_keys, q_lo_g, q_hi_g, q_lo_groups, q_hi_groups
    )

    # 评估
    mae, mape = evaluate_point(y_va.values, va_p50_m)
    coverage = np.mean((y_va.values >= cqr_low) & (y_va.values <= cqr_high)) * 100.0
    avg_width = np.mean(cqr_high - cqr_low)

    print("\n=== Validation Report (Overall, with Market Index, asym-CQR) ===")
    print(f"Samples: {len(y_va)} | Split: {split_mode}")
    print(f"P50 MAE:  {mae:,.2f}")
    print(f"P50 MAPE: {mape:.2f}%")
    print(f"CQR {int((1-ALPHA)*100)}% Coverage: {coverage:.2f}%")
    print(f"CQR Avg Interval Width: {avg_width:,.2f}")

    # 切片报表
    print("\n=== Slice Report (fuel × transmission × age_bin × period_bin) ===")
    df_slice = pd.DataFrame({
        "y": y_va.values, "p50": va_p50_m, "lo": cqr_low, "hi": cqr_high,
        "key": group_keys.values
    })
    for k, g in df_slice.groupby("key"):
        n = len(g)
        cov = np.mean((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])) * 100.0
        w = np.mean(g["hi"] - g["lo"])
        mae_k, mape_k = evaluate_point(g["y"].values, g["p50"].values)
        print(f"{k:>45s} | n={n:4d} | MAPE={mape_k:6.2f}% | Cov={cov:6.2f}% | Width={w:,.2f}")

    # Selective 报表
    print("\n=== Selective Pricing Report (by interval width ratio) ===")
    width = cqr_high - cqr_low
    wr = width / np.maximum(1e-6, va_p50_m)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mask = wr <= thr
        if mask.sum() == 0:
            print(f"WR≤{thr:.2f}: none"); continue
        mae_s, mape_s = evaluate_point(y_va.values[mask], va_p50_m[mask])
        print(f"WR≤{thr:.2f}: cover={mask.mean()*100:5.1f}% | MAE={mae_s:,.2f} | MAPE={mape_s:5.2f}%")

    # 保存
    dump(m_p10, os.path.join(SAVE_DIR, "catboost_p10.joblib"))
    dump(m_p50, os.path.join(SAVE_DIR, "catboost_p50.joblib"))
    dump(m_p90, os.path.join(SAVE_DIR, "catboost_p90.joblib"))

    meta = {
        "columns": list(X_tr.columns),
        "categorical_cols": [c for c in cat_cols if c in X_tr.columns],
        "categorical_indices": [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns],
        "alpha": ALPHA,
        "market_index": {
            "period_unit": "year",
            "gamma_t": gamma_t,
            "M_t_raw": M_t_raw,
            "M_t_smooth": M_t_smooth,
            "ewma_alpha": 0.25
        },
        "cqr_after_market": {
            "type": "asymmetric",
            "q_lo_global": q_lo_g,
            "q_hi_global": q_hi_g,
            "q_lo_groups": q_lo_groups,
            "q_hi_groups": q_hi_groups,
            "group_key_def": "age_bin(car_age)|fuel_type|transmission|period_bin"
        },
        "external_ts_blend": {
            "enable": False, "blend_weight": 0.5,
            "file": os.path.join(SAVE_DIR, "external_market_forecast.json")
        },
        "target": TARGET_COL,
        "date_col": DATE_COL if DATE_COL in df_raw.columns else None,
        "split": {"mode": split_mode, "train_size": int(len(y_tr)), "valid_size": int(len(y_va))},
        "winsorized_cols": [c for c in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL] if c in df_raw.columns]
    }
    with open(os.path.join(SAVE_DIR, "cqr_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModels & meta saved to: {SAVE_DIR}")
