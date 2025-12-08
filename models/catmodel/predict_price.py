import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
CSV_PATH = os.path.join(CURRENT_DIR, "../../datasets/Full_dataset.csv")
META_PATH = os.path.join(MODEL_DIR, "cqr_meta.json")  # cqr_meta.json 的路径，这是训练阶段导出的 meta 信息
MODEL_P10 = os.path.join(MODEL_DIR, "catboost_p10.joblib")
MODEL_P50 = os.path.join(MODEL_DIR, "catboost_p50.joblib")
MODEL_P90 = os.path.join(MODEL_DIR, "catboost_p90.joblib")

MAX_HALF_WIDTH = 60000.0  # 最终展示区间的半宽：6 万


# ===== 一些小工具 =====
def clamp_interval_to_60k(p50, lo_raw, hi_raw, max_half=MAX_HALF_WIDTH):
    """
    把原始区间 (lo_raw, hi_raw) 以 p50 为中心截断到最多 ±max_half。
    返回截断后的 (lo_show, hi_show) 以及原始半宽 half_raw。
    """
    mid = float(p50)
    lo_raw = float(lo_raw)
    hi_raw = float(hi_raw)

    half_raw = max(mid - lo_raw, hi_raw - mid)
    half_new = min(half_raw, max_half)

    lo_show = mid - half_new
    hi_show = mid + half_new
    return lo_show, hi_show, half_raw


def to_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    return float(m.group(0)) if m else np.nan


def safe_log1p(v):
    v = 0 if v is None else v
    return np.log1p(max(0.0, float(v)))


def age_bin(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "age:Unknown"
    a = float(a)
    if a < 3:
        return "age:0-3"
    if a < 8:
        return "age:3-8"
    return "age:8+"


def standardize_fuel(x):
    s = str(x).strip().lower()
    if s in ["petrol", "gasoline", "p", "油", "汽油"]:
        return "Petrol"
    if s in ["diesel", "d", "柴油"]:
        return "Diesel"
    return "Other"


def standardize_gear(x):
    s = str(x).strip().lower()
    if s in ["a", "auto", "automatic", "自动", "at"]:
        return "Automatic"
    if s in ["m", "man", "manual", "手动", "mt"]:
        return "Manual"
    return "Unknown"


# ===== 载入模型 & meta =====
_meta = json.load(open(META_PATH, encoding="utf-8"))

_cols = _meta["columns"]
_cat_cols = _meta.get("categorical_cols", [])
_cat_idx = _meta.get("categorical_indices", [])
alpha = float(_meta.get("alpha", 0.20))

market_info = _meta["market_index"]
M_t_smooth = market_info.get("M_t_smooth", {})
M_t_raw = market_info.get("M_t_raw", {})
ewma_alpha = market_info.get("ewma_alpha", 0.25)

# CQR 参数（相对误差模式）
cqr_info = _meta["cqr_after_market"]
cqr_type = cqr_info.get("type", "asymmetric_ratio")
q_lo_global = float(cqr_info.get("q_lo_global", 0.0))
q_hi_global = float(cqr_info.get("q_hi_global", 0.0))
q_lo_groups = cqr_info.get("q_lo_groups", {})
q_hi_groups = cqr_info.get("q_hi_groups", {})
group_key_def = cqr_info.get(
    "group_key_def", "age_bin|fuel_type|transmission|period_bin"
)

# 品牌 / 车型 校准表
bm_cal = _meta.get("brand_model_calibration", {})
bm_levels = bm_cal.get("levels", {}) if bm_cal else {}
BM_AGE_COEF = bm_levels.get("brand_model_age", {})
BM_COEF = bm_levels.get("brand_model", {})
BRAND_COEF = bm_levels.get("brand", {})

# 外部时间序列融合（可选，目前默认关掉）
ext_cfg = _meta.get("external_ts_blend", {"enable": False})
EXT_ENABLE = bool(ext_cfg.get("enable", False))
EXT_WEIGHT = float(ext_cfg.get("blend_weight", 0.5))
EXT_FILE = ext_cfg.get("file", "")

m_p10 = load(MODEL_P10)
m_p50 = load(MODEL_P50)
m_p90 = load(MODEL_P90)


# ===== 分组 CQR 的逐级回退 =====
def hierarchical_qhat_asym(key_full):
    """
    从 q_lo_groups/q_hi_groups 里，按 key_full 逐级回退找一对 (q_lo, q_hi)。
    这里保持你原来的逻辑：先用 full key；不行就合并 period_bin / age_bin。
    """
    if key_full in q_lo_groups and key_full in q_hi_groups:
        return float(q_lo_groups[key_full]), float(q_hi_groups[key_full])

    parts = key_full.split("|")
    if len(parts) == 4:
        # 去掉 period_bin，只看前三段
        k3_prefix = "|".join(parts[:3])
        cand = [gk for gk in q_lo_groups.keys() if gk.startswith(k3_prefix + "|")]
        if len(cand) > 0:
            lo_vals = [q_lo_groups[c] for c in cand if c in q_lo_groups]
            hi_vals = [q_hi_groups[c] for c in cand if c in q_hi_groups]
            if len(lo_vals) > 0 and len(hi_vals) > 0:
                return float(np.median(lo_vals)), float(np.median(hi_vals))

        # 再去掉 age_bin，只看 fuel|gear
        k2 = "|".join(parts[1:3])
        cand2 = [gk for gk in q_lo_groups.keys() if ("|" + k2) in gk]
        if len(cand2) > 0:
            lo_vals = [q_lo_groups[c] for c in cand2 if c in q_lo_groups]
            hi_vals = [q_hi_groups[c] for c in cand2 if c in q_hi_groups]
            if len(lo_vals) > 0 and len(hi_vals) > 0:
                return float(np.median(lo_vals)), float(np.median(hi_vals))

    return q_lo_global, q_hi_global


def make_period_bin_from_year_or_date(year, listing_date):
    if listing_date:
        try:
            y = pd.to_datetime(listing_date).year
        except Exception:
            y = year
    else:
        y = year
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return "Unknown"
    lo = int(y) // 2 * 2
    return f"{lo}-{lo+1}"


# ===== 品牌 / 车型 校准倍率 =====
def get_brand_model_multiplier_for_predict(brand, model, age_val):
    """
    预测侧：给定 brand / model / age，按层级从校准表里取一个倍率。
    """
    ab = age_bin(age_val)
    b = str(brand) if brand is not None else "Unknown"
    m = str(model) if model is not None else "Unknown"

    k1 = f"{b}|{m}|{ab}"
    if k1 in BM_AGE_COEF:
        return float(BM_AGE_COEF[k1])

    k2 = f"{b}|{m}"
    if k2 in BM_COEF:
        return float(BM_COEF[k2])

    if b in BRAND_COEF:
        return float(BRAND_COEF[b])

    return 1.0


# ===== 构造一行特征 =====
def build_row(d):
    brand = d.get("brand")
    model = d.get("model")

    year = to_number(d.get("year"))
    age = to_number(d.get("age"))
    if age is None or (isinstance(age, float) and np.isnan(age)):
        if year is not None and not np.isnan(year):
            age = datetime.now().year - year
        else:
            age = np.nan

    milage = to_number(d.get("milage"))
    engine = to_number(d.get("engine"))
    max_power = to_number(d.get("max_power"))
    seats = to_number(d.get("seats"))

    fuel_type = standardize_fuel(d.get("fuel_type"))
    gear = standardize_gear(d.get("transmission"))
    listing_date = d.get("listing_date", None)

    period = (
        str(int(year)) if (year is not None and not np.isnan(year)) else "Unknown"
    )
    period_bin = make_period_bin_from_year_or_date(year, listing_date)

    log1p_mileage = safe_log1p(milage)
    age_safe = (
        0.25
        if (age is None or age == 0 or (isinstance(age, float) and np.isnan(age)))
        else float(age)
    )

    avg_km_per_year = milage if milage == milage else np.nan
    avg_km_per_year = (
        avg_km_per_year / age_safe
        if (avg_km_per_year == avg_km_per_year)
        else np.nan
    )

    is_auto = 1 if gear == "Automatic" else 0
    hp_x_auto = (max_power if max_power == max_power else 0.0) * is_auto
    hp_div_avgkm = (
        max_power / avg_km_per_year
        if (max_power == max_power and avg_km_per_year and avg_km_per_year > 0)
        else np.nan
    )
    power_per_cc = (
        max_power / engine
        if (max_power == max_power and engine and engine > 0)
        else np.nan
    )
    cc_per_seat = (
        engine / seats
        if (engine == engine and seats and seats > 0)
        else np.nan
    )

    row = {}

    def put_num(name, val):
        row[name] = float(val) if (val is not None and val == val) else 0.0
        row[name + "_missing"] = 0 if (val is not None and val == val) else 1

    # 数值特征 + 缺失标记
    for name, val in [
        ("Year", year),
        ("Age", age),
        ("Kilometer", milage),
        ("Max Power", max_power),
        ("Engine", engine),
        ("Seats", seats),
        ("car_age", age),
        ("log1p_mileage", log1p_mileage),
        ("avg_km_per_year", avg_km_per_year),
        ("hp_x_auto", hp_x_auto),
        ("hp_div_avgkm", hp_div_avgkm),
        ("power_per_cc", power_per_cc),
        ("cc_per_seat", cc_per_seat),
        # 频数特征在预测侧无真实值，设为 0 + missing=1
        ("brand_count", np.nan),
        ("model_count", np.nan),
        ("brand_model_count", np.nan),
    ]:
        if name in _cols or (name + "_missing") in _cols:
            put_num(name, val)

    # 类别特征
    if "Brand" in _cols:
        row["Brand"] = str(brand) if brand is not None else "Unknown"
    if "Model" in _cols:
        row["Model"] = str(model) if model is not None else "Unknown"
    if "Transmission" in _cols:
        row["Transmission"] = gear
    if "Fuel Type" in _cols:
        row["Fuel Type"] = fuel_type
    if "period" in _cols:
        row["period"] = period
    if "period_bin" in _cols:
        row["period_bin"] = period_bin

    final = []
    for c in _cols:
        if c in row:
            final.append(row[c])
        else:
            if c.endswith("_missing"):
                final.append(1)
            else:
                final.append("Unknown" if c in _cat_cols else 0.0)

    X = pd.DataFrame([final], columns=_cols)
    return X, period, period_bin, age, brand, model


# ===== 市场系数（含外部融合）=====
def get_market_multiplier(period):
    if period in M_t_smooth:
        m_internal = float(M_t_smooth[period])
    elif period in M_t_raw:
        m_internal = float(M_t_raw[period])
    else:
        m_internal = 1.0

    # 可选：外部时序融合（目前默认关闭）
    if EXT_ENABLE and EXT_FILE and os.path.exists(EXT_FILE):
        try:
            ext = json.load(open(EXT_FILE, encoding="utf-8"))
            m_ext = float(ext.get("M_t", {}).get(period, m_internal))
            w = np.clip(EXT_WEIGHT, 0.0, 1.0)
            return (m_internal ** (1 - w)) * (m_ext ** w)
        except Exception:
            pass
    return m_internal


# ===== 对外预测接口 =====
def predict_price(d):
    """
    d 是一个 dict，键可以是：
    brand, model, year, age, milage,
    fuel_type, engine, max_power, transmission, seats,
    listing_date(可选，YYYY-MM-DD)。
    """
    X, period, period_bin, age_val, brand, model = build_row(d)

    # CatBoost 模型输出 log(price)
    p10_log = m_p10.predict(X)
    p50_log = m_p50.predict(X)
    p90_log = m_p90.predict(X)

    p10 = float(np.exp(np.asarray(p10_log).item()))
    p50 = float(np.exp(np.asarray(p50_log).item()))
    p90 = float(np.exp(np.asarray(p90_log).item()))

    # 时间市场系数
    M = get_market_multiplier(period)
    p10_m, p50_m, p90_m = p10 * M, p50 * M, p90 * M

    # Brand / Model 校准系数
    bm_coef = get_brand_model_multiplier_for_predict(brand, model, age_val)
    p10_mb, p50_mb, p90_mb = p10_m * bm_coef, p50_m * bm_coef, p90_m * bm_coef

    # CQR 分组 key
    akey = age_bin(age_val)
    fuel = standardize_fuel(d.get("fuel_type"))
    gear = standardize_gear(d.get("transmission"))
    key_full = f"{akey}|{fuel}|{gear}|{period_bin}"

    # 逐级回退拿相对误差 q_lo_ratio / q_hi_ratio
    qlo_ratio, qhi_ratio = hierarchical_qhat_asym(key_full)

    # 相对误差 CQR：先算“原始区间”
    lo_raw = max(0.0, p10_mb - qlo_ratio * p50_mb)
    hi_raw = p90_mb + qhi_ratio * p50_mb

    # 再做“展示截断”：最大只显示 ±6 万
    lo_show, hi_show, half_raw = clamp_interval_to_60k(p50_mb, lo_raw, hi_raw)

    wr_show = (hi_show - lo_show) / max(1e-6, p50_mb)
    wr_raw = (hi_raw - lo_raw) / max(1e-6, p50_mb)

    return {
        "p50": float(p50_mb),
        # 给前端展示的区间
        "lo": float(lo_show),
        "hi": float(hi_show),
        "wr": float(wr_show),
        # 内部真实区间（方便调试）
        "lo_raw": float(lo_raw),
        "hi_raw": float(hi_raw),
        "wr_raw": float(wr_raw),
        "group_key": key_full,
        "period": period,
        "period_bin": period_bin,
        "market_multiplier": float(M),
        "brand_model_coef": float(bm_coef),
    }


# 简单自测
if __name__ == "__main__":
    demo = {
        "brand": "Toyota",
        "model": "Corolla",
        "year": 2019,
        "age": None,
        "milage": 45000,
        "fuel_type": "Petrol",
        "engine": 1798,
        "max_power": 138,
        "transmission": "Automatic",
        "seats": 5,
    }
    out = predict_price(demo)
    print("\nPrediction:")
    for k, v in out.items():
        print(f"{k}: {v}")
