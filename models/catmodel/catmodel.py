import json
import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from joblib import dump
from sklearn.model_selection import GroupKFold, KFold

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, "model")
CSV_PATH = os.path.join(CURRENT_DIR, "../../datasets/Full_dataset.csv")
TARGET_COL = "Price"
DATE_COL   = "listing_date"

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

RANDOM_STATE = 42
TEST_SIZE_BY_TIME = 0.2
N_SPLITS = 5
ALPHA = 0.2    # CQR 覆盖率 1-ALPHA

# ===== CatBoost 基础超参 =====
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
    od_type="Iter",
    od_wait=300,
    subsample=0.9,
    rsm=0.9,
    bootstrap_type="Bernoulli",
)


# ===== 小工具 =====
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def to_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    return float(m.group(0)) if m else np.nan


def winsorize_series(s, lower=0.005, upper=0.995):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = (s.quantile(lower), s.quantile(upper))
    return (s.clip(lo, hi), lo, hi)


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
    if pd.isna(a):
        return "age:Unknown"
    a = float(a)
    if a < 3:
        return "age:0-3"
    if a < 8:
        return "age:3-8"
    return "age:8+"


def finite_sample_quantile(scores, q):
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0:
        return 0.0
    rank = int(math.ceil((n + 1) * q)) - 1
    rank = min(max(rank, 0), n - 1)
    return float(s[rank])


def evaluate_point(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(
        np.abs((y_true - y_pred) / np.maximum(1e-8, y_true))
    ) * 100.0
    return mae, mape


def get_period_series(df):
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        p = d.dt.year.astype("Int64").astype(str)
    elif YEAR_COL in df.columns:
        p = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64").astype(str)
    else:
        p = pd.Series(["Unknown"] * len(df))
    return p.fillna("Unknown")


def make_period_bin_from_series(period_series):
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
    df = pd.DataFrame({"r": r, "period": period_series})
    if agg == "median":
        R = df.groupby("period")["r"].median()
    else:
        R = df.groupby("period")["r"].mean()
    R = R - R.mean()
    return R.to_dict()


def fit_time_dummy(R_dict):
    gamma = dict(R_dict)
    M = {k: float(np.exp(v)) for k, v in gamma.items()}
    return gamma, M


def smooth_market_index(gamma_dict, alpha=0.25):
    keys = sorted([k for k in gamma_dict.keys() if k.isdigit()])
    sm = {}
    last = 0.0
    for k in keys:
        g = float(gamma_dict[k])
        last = alpha * g + (1 - alpha) * last
        sm[k] = last
    return {k: float(np.exp(v)) for k, v in sm.items()}


# ===== 特征工程 =====
def build_feature_df(df_raw):
    df = df_raw.copy()

    # 数值列
    for col in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL, YEAR_COL, AGE_COL, SEATS_COL]:
        if col in df.columns:
            df[col] = df[col].apply(to_number)

    # 变速箱 / 燃料类型标准化
    if GEAR_COL in df.columns:
        df[GEAR_COL] = standardize_enum(
            df[GEAR_COL],
            {
                "a": "Automatic",
                "auto": "Automatic",
                "automatic": "Automatic",
                "m": "Manual",
                "man": "Manual",
                "manual": "Manual",
            },
            default="Unknown",
        )

    if FUEL_COL in df.columns:
        df[FUEL_COL] = standardize_enum(
            df[FUEL_COL],
            {
                "petrol": "Petrol",
                "gasoline": "Petrol",
                "diesel": "Diesel",
                "cng": "Other",
                "lpg": "Other",
                "hybrid": "Other",
                "electric": "Other",
                "other": "Other",
            },
            default="Other",
        )

    # winsorize 目标价 / 里程 / 马力 / 排量
    for col in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL]:
        if col in df.columns:
            df[col], _, _ = winsorize_series(df[col], 0.005, 0.995)

    # 衍生特征
    df["car_age"] = make_age(df)

    if MILE_COL in df.columns:
        df["log1p_mileage"] = safe_log1p(df[MILE_COL])

    if MILE_COL in df.columns and "car_age" in df.columns:
        age_eps = df["car_age"].replace(0, 0.25)
        df["avg_km_per_year"] = df[MILE_COL] / age_eps

    if HP_COL in df.columns and GEAR_COL in df.columns:
        is_auto = (df[GEAR_COL] == "Automatic").astype(int)
        df["hp_x_auto"] = df[HP_COL] * is_auto

    if HP_COL in df.columns and "avg_km_per_year" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["hp_div_avgkm"] = df[HP_COL] / np.where(
                df["avg_km_per_year"] > 0, df["avg_km_per_year"], np.nan
            )

    if HP_COL in df.columns and ENGINE_COL in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["power_per_cc"] = df[HP_COL] / np.where(
                df[ENGINE_COL] > 0, df[ENGINE_COL], np.nan
            )

    if ENGINE_COL in df.columns and SEATS_COL in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["cc_per_seat"] = df[ENGINE_COL] / np.where(
                df[SEATS_COL] > 0, df[SEATS_COL], np.nan
            )

    # 频数特征
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

    if BRAND_COL in df.columns and MODEL_COL in df.columns:
        bm = df[BRAND_COL].astype(str) + "§" + df[MODEL_COL].astype(str)
        cnt_bm = bm.map(bm.value_counts())
        df["brand_model_count"] = cnt_bm

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

    assert TARGET_COL in df.columns, f"CSV缺少目标列: {TARGET_COL}"

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    X = pd.DataFrame(index=df.index)

    # 数值列 + 缺失标记
    for col in numeric_candidates:
        col_num = pd.to_numeric(df[col], errors="coerce")
        miss_flag = col_num.isna().astype(int)
        med = np.nanmedian(col_num)
        X[col] = np.where(np.isnan(col_num), med, col_num)
        X[col + "_missing"] = miss_flag

    # 类别列
    for col in categorical_candidates:
        X[col] = (
            df[col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
        )

    # 时间 period / period_bin
    X["period"] = get_period_series(df)
    X["period_bin"] = make_period_bin_from_series(X["period"])

    X["period"] = X["period"].astype(str)
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


# ===== 划分训练 / 验证 =====
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


# ===== CQR 分组 key =====
def build_group_keys_for_cqr(df_va):
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


def cqr_asymmetric_ratio_global_and_group(
    y_true, p50, p10, p90, group_keys, alpha=ALPHA, min_group_size=120
):
    """
    用“相对误差”(s/p50) 做 CQR 校准：
    s_lo = max(0, p10 - y), s_hi = max(0, y - p90)，再除以 p50。
    """
    y_true = np.asarray(y_true)
    p50 = np.asarray(p50)
    p10 = np.asarray(p10)
    p90 = np.asarray(p90)

    eps = 1e-6
    s_lo_all = np.maximum(0.0, p10 - y_true) / np.maximum(eps, p50)
    s_hi_all = np.maximum(0.0, y_true - p90) / np.maximum(eps, p50)

    q_lo_global = finite_sample_quantile(s_lo_all, 1.0 - alpha)
    q_hi_global = finite_sample_quantile(s_hi_all, 1.0 - alpha)

    q_lo_groups = {}
    q_hi_groups = {}
    g_keys = pd.Series(group_keys).astype(str).values
    uniq = np.unique(g_keys)

    for g in uniq:
        idx = g_keys == g
        n_g = int(idx.sum())
        if n_g >= min_group_size:
            q_lo_groups[g] = finite_sample_quantile(
                s_lo_all[idx], 1.0 - alpha
            )
            q_hi_groups[g] = finite_sample_quantile(
                s_hi_all[idx], 1.0 - alpha
            )

    return (q_lo_global, q_hi_global), (q_lo_groups, q_hi_groups)


def apply_asymmetric_cqr_ratio(
    p50, p10, p90, group_keys, q_lo_global, q_hi_global, q_lo_groups, q_hi_groups
):
    """
    预测时：lo = p10 - q_lo * p50, hi = p90 + q_hi * p50
    """
    p50 = np.asarray(p50)
    p10 = np.asarray(p10)
    p90 = np.asarray(p90)
    g_keys = pd.Series(group_keys).astype(str).values

    qlo = np.array([q_lo_groups.get(k, q_lo_global) for k in g_keys])
    qhi = np.array([q_hi_groups.get(k, q_hi_global) for k in g_keys])

    lo = np.maximum(0.0, p10 - qlo * p50)
    hi = p90 + qhi * p50
    return lo, hi


# ===== 品牌 / 车型 校准 =====
def compute_brand_model_calibration(
    y_true,
    p50_m,
    X_va,
    lam_age=10.0,
    lam_bm=15.0,
    lam_brand=20.0,
    min_cnt_age=3,
    min_cnt_bm=5,
    min_cnt_brand=10,
):
    """
    在时间市场系数校正后的 P50 上，再按 Brand/Model(+age_bin) 做残差校准。
    返回一个 dict，后面存到 meta 里。
    """
    if BRAND_COL not in X_va.columns:
        return {}

    brand = X_va[BRAND_COL].astype(str).fillna("Unknown")
    if MODEL_COL in X_va.columns:
        model = X_va[MODEL_COL].astype(str).fillna("Unknown")
    else:
        model = pd.Series(["Unknown"] * len(X_va))
    if "car_age" in X_va.columns:
        age_series = X_va["car_age"]
    else:
        age_series = pd.Series([np.nan] * len(X_va))
    age_bin_series = age_series.apply(age_bin)

    y_true = np.asarray(y_true)
    p50_m = np.asarray(p50_m)
    log_r = np.log(np.maximum(y_true, 1e-6)) - np.log(np.maximum(p50_m, 1e-6))
    global_log_med = float(np.median(log_r))

    df = pd.DataFrame(
        {
            "brand": brand.values,
            "model": model.values,
            "age_bin": age_bin_series.values,
            "log_r": log_r,
        }
    )

    def _build_level(group_cols, lam, min_cnt):
        res = {}
        grp = df.groupby(group_cols)["log_r"]
        for key, s in grp:
            s = s.dropna()
            n = int(s.size)
            if n < min_cnt:
                continue
            med = float(s.median())
            w = n / (n + lam)
            log_c = (1.0 - w) * global_log_med + w * med
            coef = float(np.exp(log_c))
            if isinstance(key, tuple):
                key_str = "|".join(str(k) for k in key)
            else:
                key_str = str(key)
            res[key_str] = coef
        return res

    lvl_bma = _build_level(["brand", "model", "age_bin"], lam_age, min_cnt_age)
    lvl_bm = _build_level(["brand", "model"], lam_bm, min_cnt_bm)
    lvl_brand = _build_level(["brand"], lam_brand, min_cnt_brand)

    cal = {
        "global_log_median": global_log_med,
        "levels": {
            "brand_model_age": lvl_bma,
            "brand_model": lvl_bm,
            "brand": lvl_brand,
        },
    }
    return cal


def get_brand_model_multiplier(brand, model, age_bin_str, cal_cfg):
    """
    给定 brand / model / age_bin，从校准表里取一个倍率：
    先找 Brand+Model+age_bin，其次 Brand+Model，再其次 Brand，最后 1.0。
    """
    if not cal_cfg:
        return 1.0
    levels = cal_cfg.get("levels", {})
    lvl_bma = levels.get("brand_model_age", {})
    lvl_bm = levels.get("brand_model", {})
    lvl_brand = levels.get("brand", {})

    b = str(brand) if brand is not None else "Unknown"
    m = str(model) if model is not None else "Unknown"
    ab = str(age_bin_str)

    k1 = f"{b}|{m}|{ab}"
    if k1 in lvl_bma:
        return float(lvl_bma[k1])

    k2 = f"{b}|{m}"
    if k2 in lvl_bm:
        return float(lvl_bm[k2])

    if b in lvl_brand:
        return float(lvl_brand[b])

    return 1.0


# ===== A+B：样本加权（全局长尾 + 品牌内价格分布） =====
def compute_global_tail_weights(y_log, n_bins=30, clip_min=0.5, clip_max=3.0):
    """
    B：全局长尾加权。
    在 log(price) 上做直方图，密度高的区域权重低，密度低的尾部权重高。
    """
    y_log = np.asarray(y_log, dtype=float)
    n = y_log.size
    if n == 0:
        return np.ones(0, dtype=float)

    hist, edges = np.histogram(y_log, bins=n_bins)
    hist = hist.astype(float) + 1e-6  # 防止除零
    total = hist.sum()

    # 为每个样本找到所在 bin
    bin_idx = np.searchsorted(edges, y_log, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(hist) - 1)

    freq = hist[bin_idx] / total  # 近似概率质量
    w = 1.0 / (freq + 1e-8)
    w /= np.mean(w)  # 均值归一到 1
    w = np.clip(w, clip_min, clip_max)
    return w


def compute_brand_price_weights(
    y, brand_series, min_brand_samples=40, max_bins=10, clip_min=0.5, clip_max=3.0
):
    """
    A：品牌内价格分布加权。
    对每个品牌单独做价格直方图，品牌内部价格区间稀疏的样本权重大，
    价格集中（常见价位）的样本权重小。
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    w = np.ones(n, dtype=float)

    brands = pd.Series(brand_series).astype(str).fillna("Unknown").values

    for b in np.unique(brands):
        idx = np.where(brands == b)[0]
        if idx.size < min_brand_samples:
            # 该品牌样本太少，不做复杂加权，全部权重=1
            continue

        y_b = y[idx]
        y_b = y_b[np.isfinite(y_b)]
        if y_b.size < min_brand_samples or np.allclose(y_b.min(), y_b.max()):
            continue

        # Freedman–Diaconis 规则确定 bin 宽度
        q25, q75 = np.percentile(y_b, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            continue
        h = 2.0 * iqr / (y_b.size ** (1.0 / 3.0))
        if h <= 0:
            continue

        data_range = y_b.max() - y_b.min()
        n_bins = int(np.ceil(data_range / h))
        n_bins = max(1, min(max_bins, n_bins))

        hist, edges = np.histogram(y_b, bins=n_bins)
        hist = hist.astype(float) + 1e-6
        total = hist.sum()

        bin_idx_b = np.searchsorted(edges, y[idx], side="right") - 1
        bin_idx_b = np.clip(bin_idx_b, 0, len(hist) - 1)

        freq_b = hist[bin_idx_b] / total
        w_b = 1.0 / (freq_b + 1e-8)
        w_b /= np.mean(w_b)
        w_b = np.clip(w_b, clip_min, clip_max)

        w[idx] = w_b

    return w


def build_sample_weights(y_tr, y_tr_log, brand_tr):
    """
    综合 A（品牌内）+ B（全局）的样本权重。
    """
    w_global = compute_global_tail_weights(y_tr_log)
    w_brand = compute_brand_price_weights(y_tr, brand_tr)

    w = w_global * w_brand
    w /= np.mean(w)  # 全局再归一一次
    return w


# ===== CatBoost 训练 =====
def train_quantile_model(X_tr, y_tr, X_va, y_va, alpha, sample_weight=None):
    params = CATBOOST_PARAMS.copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"

    # CatBoost 直接吃字符串类别
    cats = [c for c in X_tr.columns if X_tr[c].dtype == "object"]
    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)

    if sample_weight is not None:
        train_pool = Pool(X_tr, y_tr, cat_features=cats, weight=sample_weight)
    else:
        train_pool = Pool(X_tr, y_tr, cat_features=cats)

    val_pool = Pool(X_va, y_va, cat_features=cats)

    print("Cat features used:", cats)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

    preds = model.predict(val_pool)
    return model, preds


# ===== 主流程 =====
if __name__ == "__main__":
    ensure_dir(SAVE_DIR)

    df_raw = pd.read_csv(CSV_PATH)
    X_all, y_all, cat_cols = build_feature_df(df_raw)

    # 为了划分一致，这里用 build_feature_df 之后的索引
    df_for_split = df_raw.loc[y_all.index].reset_index(drop=True)

    # 划分 train / valid
    if DATE_COL in df_for_split.columns:
        tr_idx, va_idx = time_based_split(
            df_for_split, DATE_COL, TEST_SIZE_BY_TIME
        )
        split_mode = "time"
    else:
        groups = build_groups_for_split(df_for_split)
        tr_idx, va_idx = group_based_split(
            df_for_split, groups, N_SPLITS, RANDOM_STATE
        )
        split_mode = "group" if groups is not None else "kfold"

    X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
    X_va = X_all.iloc[va_idx].reset_index(drop=True)
    y_va = y_all.iloc[va_idx].reset_index(drop=True)

    # log 目标
    y_tr_log = np.log(np.maximum(1e-6, y_tr))
    y_va_log = np.log(np.maximum(1e-6, y_va))

    # ===== A+B 样本权重（只作用于训练集）=====
    if BRAND_COL in df_for_split.columns:
        brand_tr = df_for_split.iloc[tr_idx][BRAND_COL]
    else:
        brand_tr = pd.Series(["Unknown"] * len(tr_idx))

    sample_weight = build_sample_weights(
        y_tr.values.astype(float), y_tr_log.values.astype(float), brand_tr
    )

    print(
        f"Sample weight stats: "
        f"mean={sample_weight.mean():.3f}, "
        f"min={sample_weight.min():.3f}, "
        f"max={sample_weight.max():.3f}"
    )

    # ===== 训练 CatBoost Quantile 模型 =====
    print("Training CatBoost Quantile models (P10/P50/P90)...")
    m_p10, va_p10_log = train_quantile_model(
        X_tr, y_tr_log, X_va, y_va_log, 0.1, sample_weight
    )
    m_p50, va_p50_log = train_quantile_model(
        X_tr, y_tr_log, X_va, y_va_log, 0.5, sample_weight
    )
    m_p90, va_p90_log = train_quantile_model(
        X_tr, y_tr_log, X_va, y_va_log, 0.9, sample_weight
    )

    va_p10 = np.exp(va_p10_log)
    va_p50 = np.exp(va_p50_log)
    va_p90 = np.exp(va_p90_log)

    # ===== 时间市场系数 =====
    period_va = X_va["period"]
    R_t = compute_residual_by_period(
        y_va_log.values, va_p50_log, period_va, agg="median"
    )
    gamma_t, M_t_raw = fit_time_dummy(R_t)
    M_t_smooth = smooth_market_index(gamma_t, alpha=0.25)

    def map_M(per):
        if per in M_t_smooth:
            return M_t_smooth[per]
        if per in M_t_raw:
            return M_t_raw[per]
        return 1.0

    M_vec = period_va.map(map_M).astype(float).values
    va_p10_m = va_p10 * M_vec
    va_p50_m = va_p50 * M_vec
    va_p90_m = va_p90 * M_vec

    # ===== 品牌 / 车型 校准（基于 M_t 后的 P50）=====
    brand_model_cal = compute_brand_model_calibration(
        y_va.values, va_p50_m, X_va
    )

    brand_va = X_va[BRAND_COL].astype(str)
    model_va = X_va[MODEL_COL].astype(str)
    age_bin_va = X_va["car_age"].apply(age_bin)

    coef_vec = np.array(
        [
            get_brand_model_multiplier(
                brand_va.iloc[i],
                model_va.iloc[i],
                age_bin_va.iloc[i],
                brand_model_cal,
            )
            for i in range(len(y_va))
        ]
    )

    va_p10_mb = va_p10_m * coef_vec
    va_p50_mb = va_p50_m * coef_vec
    va_p90_mb = va_p90_m * coef_vec

    # ===== 相对误差 CQR（基于“时间+品牌校准后”的预测）=====
    group_keys = build_group_keys_for_cqr(X_va)
    (q_lo_g, q_hi_g), (q_lo_groups, q_hi_groups) = (
        cqr_asymmetric_ratio_global_and_group(
            y_va.values,
            va_p50_mb,
            va_p10_mb,
            va_p90_mb,
            group_keys,
            alpha=ALPHA,
            min_group_size=120,
        )
    )
    cqr_low, cqr_high = apply_asymmetric_cqr_ratio(
        va_p50_mb,
        va_p10_mb,
        va_p90_mb,
        group_keys,
        q_lo_g,
        q_hi_g,
        q_lo_groups,
        q_hi_groups,
    )

    # ===== 验证集报告 =====
    mae, mape = evaluate_point(y_va.values, va_p50_mb)
    coverage = (
        np.mean((y_va.values >= cqr_low) & (y_va.values <= cqr_high)) * 100.0
    )
    avg_width = np.mean(cqr_high - cqr_low)

    print("\n=== Validation Report (with market + brand/model calibration + weights) ===")
    print(f"Samples: {len(y_va)} | Split: {split_mode}")
    print(f"P50 MAE:  {mae:,.2f}")
    print(f"P50 MAPE: {mape:.2f}%")
    print(f"CQR {int((1 - ALPHA) * 100)}% Coverage: {coverage:.2f}%")
    print(f"CQR Avg Interval Width: {avg_width:,.2f}")

    print("\n=== Slice Report (fuel × transmission × age_bin × period_bin) ===")
    df_slice = pd.DataFrame(
        {
            "y": y_va.values,
            "p50": va_p50_mb,
            "lo": cqr_low,
            "hi": cqr_high,
            "key": group_keys.values,
        }
    )
    for k, g in df_slice.groupby("key"):
        n = len(g)
        cov = (
            np.mean((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])) * 100.0
        )
        w = np.mean(g["hi"] - g["lo"])
        mae_k, mape_k = evaluate_point(g["y"].values, g["p50"].values)
        print(
            f"{k:>45s} | n={n:4d} | MAPE={mape_k:6.2f}% | "
            f"Cov={cov:6.2f}% | Width={w:,.2f}"
        )

    print("\n=== Selective Pricing Report (by interval width ratio) ===")
    width = cqr_high - cqr_low
    wr = width / np.maximum(1e-6, va_p50_mb)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mask = wr <= thr
        if mask.sum() == 0:
            print(f"WR≤{thr:.2f}: none")
            continue
        mae_s, mape_s = evaluate_point(
            y_va.values[mask], va_p50_mb[mask]
        )
        print(
            f"WR≤{thr:.2f}: cover={mask.mean() * 100:5.1f}% | "
            f"MAE={mae_s:,.2f} | MAPE={mape_s:5.2f}%"
        )

    # ===== 保存模型 & meta =====
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
        # 注意：这其实是“时间+品牌校准后”的 CQR
        "cqr_after_market": {
            "type": "asymmetric_ratio",
            "q_lo_global": q_lo_g,
            "q_hi_global": q_hi_g,
            "q_lo_groups": q_lo_groups,
            "q_hi_groups": q_hi_groups,
            "group_key_def": "age_bin(car_age)|fuel_type|transmission|period_bin",
        },
        "brand_model_calibration": brand_model_cal,
        "external_ts_blend": {
            "enable": False,
            "blend_weight": 0.5,
            "file": os.path.join(SAVE_DIR, "external_market_forecast.json"),
        },
        "target": TARGET_COL,
        "date_col": DATE_COL if DATE_COL in df_for_split.columns else None,
        "split": {
            "mode": split_mode,
            "train_size": int(len(y_tr)),
            "valid_size": int(len(y_va)),
        },
        "winsorized_cols": [
            c
            for c in [TARGET_COL, MILE_COL, HP_COL, ENGINE_COL]
            if c in df_raw.columns
        ],
        "sample_weighting": {
            "global_tail": {"n_bins": 30, "clip": [0.5, 3.0]},
            "brand_price": {
                "min_brand_samples": 40,
                "max_bins": 10,
                "clip": [0.5, 3.0],
            },
        },
    }

    with open(
        os.path.join(SAVE_DIR, "cqr_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModels & meta saved to: {SAVE_DIR}")
