# predict_price.py
# -*- coding: utf-8 -*-

import os, json, re, math
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load

# =========================
# 固定路径（按你的环境改这三行）
# =========================
MODEL_DIR   = r"C:\Users\Lenovo\PycharmProjects\ml\app\used_car_price_v3"
META_PATH   = os.path.join(MODEL_DIR, "cqr_meta.json")
MODEL_P10   = os.path.join(MODEL_DIR, "catboost_p10.joblib")
MODEL_P50   = os.path.join(MODEL_DIR, "catboost_p50.joblib")
MODEL_P90   = os.path.join(MODEL_DIR, "catboost_p90.joblib")

# =============== 小工具（与训练对齐） ===============
def to_number(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", ""))
    return float(m.group(0)) if m else np.nan

def safe_log1p(v):
    v = 0 if v is None else v
    return np.log1p(max(0.0, float(v)))

def age_bin(a):
    if a is None or (isinstance(a, float) and np.isnan(a)): return "age:Unknown"
    a = float(a)
    if a < 3:  return "age:0-3"
    if a < 8:  return "age:3-8"
    return "age:8+"

def standardize_fuel(x):
    s = str(x).strip().lower()
    if s in ["petrol", "gasoline"]: return "Petrol"
    if s in ["diesel"]:             return "Diesel"
    return "Other"

def standardize_gear(x):
    s = str(x).strip().lower()
    if s in ["a", "auto", "automatic"]: return "Automatic"
    if s in ["m", "man", "manual"]:     return "Manual"
    return "Unknown"

def finite_sample_quantile(scores, q):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0: return 0.0
    rank = int(math.ceil((n + 1) * q)) - 1
    rank = min(max(rank, 0), n - 1)
    return float(s[rank])

# =============== 载入模型与元数据 ===============
_meta = json.load(open(META_PATH, "r", encoding="utf-8"))
_cols             = _meta["columns"]                 # 训练时的列顺序
_cat_cols         = _meta.get("categorical_cols", [])
_cat_idx          = _meta.get("categorical_indices", [])
alpha             = float(_meta.get("alpha", 0.20))

market_info       = _meta["market_index"]
M_t_smooth        = market_info.get("M_t_smooth", {})
M_t_raw           = market_info.get("M_t_raw", {})
ewma_alpha        = market_info.get("ewma_alpha", 0.35)

cqr_info          = _meta["cqr_after_market"]
q_hat_global      = float(cqr_info["q_hat_global"])
q_hat_groups      = cqr_info.get("q_hat_groups", {})
group_key_def     = cqr_info.get("group_key_def", "age_bin|fuel|transmission|period")

ext_cfg           = _meta.get("external_ts_blend", {"enable": False})
EXT_ENABLE        = bool(ext_cfg.get("enable", False))
EXT_WEIGHT        = float(ext_cfg.get("blend_weight", 0.5))
EXT_FILE          = ext_cfg.get("file", "")

m_p10 = load(MODEL_P10)
m_p50 = load(MODEL_P50)
m_p90 = load(MODEL_P90)

# =============== 分组回退策略（逐级回退） ===============
def hierarchical_qhat(key_full):
    """key: age_bin|fuel|gear|period -> 逐级去 period -> 去 age_bin -> 仅 fuel|gear -> 全局"""
    if key_full in q_hat_groups:
        return float(q_hat_groups[key_full])

    parts = key_full.split("|")
    if len(parts) == 4:
        # 去掉 period
        k3 = "|".join(parts[:3])
        cand = [gk for gk in q_hat_groups.keys() if gk.startswith(k3+"|")]
        if k3 in q_hat_groups: return float(q_hat_groups[k3])  # 兼容可能已存三段键
        if len(cand) > 0:
            # 用该三段键的分位（退而求其次：同前缀的中位数）
            vals = [q_hat_groups[c] for c in cand]
            return float(np.median(vals))

        # 再去掉 age_bin -> fuel|gear
        k2 = "|".join(parts[1:3])
        cand2 = [gk for gk in q_hat_groups.keys() if ("|"+k2) in gk]
        if k2 in q_hat_groups: return float(q_hat_groups[k2])
        if len(cand2) > 0:
            vals = [q_hat_groups[c] for c in cand2]
            return float(np.median(vals))

    return q_hat_global

# =============== 生成一行与训练一致的特征 ===============
def build_feature_row(input_dict):
    """
    input_dict 例子：
    {
      "brand":"Toyota","model":"Corolla",
      "year":2019,"age":4,"milage":45000,
      "fuel_type":"Petrol","engine":1798,"max_power":138,
      "transmission":"Automatic","seats":5,
      "listing_date": None   # 可选
    }
    """
    # 读取输入
    brand = input_dict.get("brand")
    model = input_dict.get("model")

    year  = to_number(input_dict.get("year"))
    age   = to_number(input_dict.get("age"))
    if age is None or np.isnan(age):
        # 若未给 age，用 year 推
        if year is not None and not np.isnan(year):
            age = datetime.now().year - year
        else:
            age = np.nan

    milage     = to_number(input_dict.get("milage"))
    engine     = to_number(input_dict.get("engine"))
    max_power  = to_number(input_dict.get("max_power"))
    seats      = to_number(input_dict.get("seats"))
    fuel_type  = standardize_fuel(input_dict.get("fuel_type"))
    gear       = standardize_gear(input_dict.get("transmission"))

    # period：优先 listing_date 的年；否则用 year；否则 "Unknown"
    listing_date = input_dict.get("listing_date", None)
    if listing_date:
        try:
            period = pd.to_datetime(listing_date).year
        except Exception:
            period = int(year) if year == year else None
    else:
        period = int(year) if year == year else None
    period = str(period) if period is not None else "Unknown"

    # 派生（与训练一致）
    log1p_mileage = safe_log1p(milage)
    age_safe = 0.25 if (age is None or age == 0 or np.isnan(age)) else float(age)
    avg_km_per_year = (milage if milage is not None and not np.isnan(milage) else np.nan)
    avg_km_per_year = (avg_km_per_year / age_safe) if avg_km_per_year==avg_km_per_year else np.nan

    is_auto = 1 if gear == "Automatic" else 0
    hp_x_auto = (max_power if max_power==max_power else 0.0) * is_auto
    hp_div_avgkm = (max_power/avg_km_per_year) if (max_power==max_power and avg_km_per_year and avg_km_per_year>0) else np.nan
    power_per_cc = (max_power/engine) if (max_power==max_power and engine and engine>0) else np.nan
    cc_per_seat  = (engine/seats) if (engine==engine and seats and seats>0) else np.nan

    # 构造与训练同名的列（缺的就填 Unknown/0，并生成 *_missing 标志）
    row = {}
    # 数值主列
    def put_num(name, val):
        row[name] = float(val) if (val is not None and val==val) else 0.0
        row[name + "_missing"] = 0 if (val is not None and val==val) else 1

    for name, val in [
        ("year", year),
        ("age", age),
        ("milage", milage),
        ("max_power", max_power),
        ("engine", engine),
        ("seats", seats),
        ("car_age", age),
        ("log1p_mileage", log1p_mileage),
        ("avg_km_per_year", avg_km_per_year),
        ("hp_x_auto", hp_x_auto),
        ("hp_div_avgkm", hp_div_avgkm),
        ("power_per_cc", power_per_cc),
        ("cc_per_seat", cc_per_seat),
    ]:
        if name in _cols or (name + "_missing") in _cols:
            put_num(name, val)

    # 类别列
    if "brand" in _cols: row["brand"] = str(brand) if brand is not None else "Unknown"
    if "model" in _cols: row["model"] = str(model) if model is not None else "Unknown"
    if "transmission" in _cols: row["transmission"] = gear
    if "fuel_type" in _cols:     row["fuel_type"]   = fuel_type

    # period
    if "period" in _cols: row["period"] = period

    # 齐备到训练列顺序
    # 任何训练时存在但 row 没给的列，补默认值
    final = []
    for c in _cols:
        if c in row:
            final.append(row[c])
        else:
            # 缺失列：数值列默认 0，类别列默认 "Unknown"
            if c.endswith("_missing"):
                final.append(1)  # 未提供 -> 标记缺失
            else:
                final.append("Unknown" if c in _cat_cols else 0.0)

    X = pd.DataFrame([final], columns=_cols)
    return X, period, age  # age 用于分组的 age_bin

# =============== 市场系数（含外部时序融合） ===============
def get_market_multiplier(period):
    # 内部指数
    if period in M_t_smooth:
        m_internal = float(M_t_smooth[period])
    elif period in M_t_raw:
        m_internal = float(M_t_raw[period])
    else:
        m_internal = 1.0

    # 外部序列（可选）
    if EXT_ENABLE and EXT_FILE and os.path.exists(EXT_FILE):
        try:
            ext = json.load(open(EXT_FILE, "r", encoding="utf-8"))
            ext_unit = ext.get("period_unit", "year")
            ext_map  = ext.get("M_t", {})
            m_ext = float(ext_map.get(period, m_internal))
            # 几何加权更合理（防止负值），若你想线性可自行改
            w = np.clip(EXT_WEIGHT, 0.0, 1.0)
            m_final = (m_internal ** (1 - w)) * (m_ext ** w)
            return m_final
        except Exception:
            pass
    return m_internal

# =============== 预测主函数 ===============
def predict_price(input_dict):
    """
    返回：
      {
        "p50": float, "lo": float, "hi": float,
        "wr": float,  # 相对宽度 (hi-lo)/p50
        "group_key": "...",
        "period": "YYYY",
        "market_multiplier": float
      }
    """
    X, period, age_val = build_feature_row(input_dict)

    # 1) 三模型预测（注意：模型学的是 log(price)）
    # 这里不用 Pool，也能被 CatBoost 接收 DataFrame（若报列类型问题再换 Pool）
    p10_log = m_p10.predict(X)
    p50_log = m_p50.predict(X)
    p90_log = m_p90.predict(X)

    p10 = float(np.exp(np.asarray(p10_log).item()))
    p50 = float(np.exp(np.asarray(p50_log).item()))
    p90 = float(np.exp(np.asarray(p90_log).item()))


    # 2) 乘市场系数
    M = get_market_multiplier(period)
    p10_m = p10 * M
    p50_m = p50 * M
    p90_m = p90 * M

    # 3) 按“时间感知 Mondrian-CQR”做区间校准
    akey = age_bin(age_val)
    fuel = standardize_fuel(input_dict.get("fuel_type"))
    gear = standardize_gear(input_dict.get("transmission"))
    key_full = f"{akey}|{fuel}|{gear}|{period}"
    qhat = hierarchical_qhat(key_full)

    lo = p10_m - qhat
    hi = p90_m + qhat
    lo = max(0.0, float(lo))
    hi = float(hi)
    wr = (hi - lo) / max(1e-6, p50_m)

    return {
        "p50": float(p50_m),
        "lo": lo,
        "hi": hi,
        "wr": float(wr),
        "group_key": key_full,
        "period": period,
        "market_multiplier": float(M)
    }

# =============== 示例调用（你可以删掉这段） ===============
if __name__ == "__main__":
    demo = {
        "brand":"Toyota",
        "model":"Corolla",
        "year":2019,
        "age":  2025-2019,      # 也可以直接填 6
        "milage": 45000,        # km
        "fuel_type":"Petrol",
        "engine": 1798,         # cc
        "max_power": 138,       # bhp
        "transmission":"Automatic",
        "seats":5,
        # "listing_date":"2021-06-15"  # 可选；不填就用 year
    }
    out = predict_price(demo)
    print("\nPrediction:")
    for k,v in out.items():
        print(f"{k}: {v}")
