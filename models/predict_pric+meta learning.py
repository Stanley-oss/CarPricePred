# predict_price.py
# -*- coding: utf-8 -*-

import os, json, re, math
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load
from meta_fewshot import FewShotConfig, FewShotKnnMeta



# 路径
MODEL_DIR   = r"C:\Users\Lenovo\PycharmProjects\ml\app\used_car_price_v4"             # 改成你的
META_PATH   = os.path.join(MODEL_DIR, "cqr_meta.json")
MODEL_P10   = os.path.join(MODEL_DIR, "catboost_p10.joblib")
MODEL_P50   = os.path.join(MODEL_DIR, "catboost_p50.joblib")
MODEL_P90   = os.path.join(MODEL_DIR, "catboost_p90.joblib")

# 小工具
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
    if s in ["petrol", "gasoline", "p", "油", "汽油"]: return "Petrol"
    if s in ["diesel", "d", "柴油"]:                   return "Diesel"
    return "Other"

def standardize_gear(x):
    s = str(x).strip().lower()
    if s in ["a","auto","automatic","自动","at"]: return "Automatic"
    if s in ["m","man","manual","手动","mt"]:     return "Manual"
    return "Unknown"

# 载入模型与meta
_meta = json.load(open(META_PATH, "r", encoding="utf-8"))
_cols       = _meta["columns"]
_cat_cols   = _meta.get("categorical_cols", [])
_cat_idx    = _meta.get("categorical_indices", [])
alpha       = float(_meta.get("alpha", 0.20))

market_info = _meta["market_index"]
M_t_smooth  = market_info.get("M_t_smooth", {})
M_t_raw     = market_info.get("M_t_raw", {})
ewma_alpha  = market_info.get("ewma_alpha", 0.25)

cqr_info    = _meta["cqr_after_market"]
cqr_type    = cqr_info.get("type", "asymmetric")
q_lo_global = float(cqr_info.get("q_lo_global", 0.0))
q_hi_global = float(cqr_info.get("q_hi_global", 0.0))
q_lo_groups = cqr_info.get("q_lo_groups", {})
q_hi_groups = cqr_info.get("q_hi_groups", {})
group_key_def = cqr_info.get("group_key_def", "age_bin|fuel_type|transmission|period_bin")

ext_cfg     = _meta.get("external_ts_blend", {"enable": False})
EXT_ENABLE  = bool(ext_cfg.get("enable", False))
EXT_WEIGHT  = float(ext_cfg.get("blend_weight", 0.5))
EXT_FILE    = ext_cfg.get("file", "")

m_p10 = load(MODEL_P10)
m_p50 = load(MODEL_P50)
m_p90 = load(MODEL_P90)

# 分组回退（逐级）
def hierarchical_qhat_asym(key_full):
    if key_full in q_lo_groups and key_full in q_hi_groups:
        return float(q_lo_groups[key_full]), float(q_hi_groups[key_full])

    parts = key_full.split("|")
    # 去掉 period_bin
    if len(parts) == 4:
        k3_prefix = "|".join(parts[:3])
        cand = [gk for gk in q_lo_groups.keys() if gk.startswith(k3_prefix+"|")]
        if len(cand) > 0:
            lo_vals = [q_lo_groups[c] for c in cand if c in q_lo_groups]
            hi_vals = [q_hi_groups[c] for c in cand if c in q_hi_groups]
            if len(lo_vals)>0 and len(hi_vals)>0:
                return float(np.median(lo_vals)), float(np.median(hi_vals))
        # 再去掉 age_bin -> 仅 fuel|gear
        k2 = "|".join(parts[1:3])
        cand2 = [gk for gk in q_lo_groups.keys() if ("|"+k2) in gk]
        if len(cand2) > 0:
            lo_vals = [q_lo_groups[c] for c in cand2 if c in q_lo_groups]
            hi_vals = [q_hi_groups[c] for c in cand2 if c in q_hi_groups]
            if len(lo_vals)>0 and len(hi_vals)>0:
                return float(np.median(lo_vals)), float(np.median(hi_vals))

    return q_lo_global, q_hi_global

# period_bin
def make_period_bin_from_year_or_date(year, listing_date):
    if listing_date:
        try:
            y = pd.to_datetime(listing_date).year
        except Exception:
            y = year
    else:
        y = year
    if y is None or (isinstance(y, float) and np.isnan(y)): return "Unknown"
    lo = int(y)//2*2
    return f"{lo}-{lo+1}"

# 构造一行与训练一致的特征
def build_row(d):
    brand = d.get("brand")
    model = d.get("model")
    year  = to_number(d.get("year"))
    age   = to_number(d.get("age"))
    if age is None or (isinstance(age,float) and np.isnan(age)):
        if year is not None and not np.isnan(year):
            age = datetime.now().year - year
        else:
            age = np.nan
    milage    = to_number(d.get("milage"))
    engine    = to_number(d.get("engine"))
    max_power = to_number(d.get("max_power"))
    seats     = to_number(d.get("seats"))
    fuel_type = standardize_fuel(d.get("fuel_type"))
    gear      = standardize_gear(d.get("transmission"))
    listing_date = d.get("listing_date", None)

    period = str(int(year)) if year==year and year is not None else "Unknown"
    period_bin = make_period_bin_from_year_or_date(year, listing_date)

    log1p_mileage = safe_log1p(milage)
    age_safe = 0.25 if (age is None or age==0 or np.isnan(age)) else float(age)
    avg_km_per_year = (milage if milage==milage else np.nan)
    avg_km_per_year = (avg_km_per_year/age_safe) if avg_km_per_year==avg_km_per_year else np.nan
    is_auto = 1 if gear=="Automatic" else 0
    hp_x_auto = (max_power if max_power==max_power else 0.0) * is_auto
    hp_div_avgkm = (max_power/avg_km_per_year) if (max_power==max_power and avg_km_per_year and avg_km_per_year>0) else np.nan
    power_per_cc = (max_power/engine) if (max_power==max_power and engine and engine>0) else np.nan
    cc_per_seat  = (engine/seats) if (engine==engine and seats and seats>0) else np.nan

    row = {}
    def put_num(name, val):
        row[name] = float(val) if (val is not None and val==val) else 0.0
        row[name + "_missing"] = 0 if (val is not None and val==val) else 1

    # 与训练列对齐：能放的都放，没在训练列中的忽略
    for name, val in [
        ("year", year), ("age", age), ("milage", milage), ("max_power", max_power),
        ("engine", engine), ("seats", seats), ("car_age", age), ("log1p_mileage", log1p_mileage),
        ("avg_km_per_year", avg_km_per_year), ("hp_x_auto", hp_x_auto), ("hp_div_avgkm", hp_div_avgkm),
        ("power_per_cc", power_per_cc), ("cc_per_seat", cc_per_seat),
        # 频数特征在预测侧没有先验值，设为0并标记缺失
        ("brand_count", np.nan), ("model_count", np.nan), ("brand_model_count", np.nan),
    ]:
        if name in _cols or (name + "_missing") in _cols:
            put_num(name, val)

    if "brand" in _cols: row["brand"] = str(brand) if brand is not None else "Unknown"
    if "model" in _cols: row["model"] = str(model) if model is not None else "Unknown"
    if "transmission" in _cols: row["transmission"] = gear
    if "fuel_type" in _cols:     row["fuel_type"]   = fuel_type
    if "period" in _cols:        row["period"]      = period
    if "period_bin" in _cols:    row["period_bin"]  = period_bin

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
    return X, period, period_bin, age

# 市场系数（含外部序列融合）
def get_market_multiplier(period):
    if period in M_t_smooth: m_internal = float(M_t_smooth[period])
    elif period in M_t_raw:  m_internal = float(M_t_raw[period])
    else: m_internal = 1.0

    # 可选外部融合
    if EXT_ENABLE and EXT_FILE and os.path.exists(EXT_FILE):
        try:
            ext = json.load(open(EXT_FILE, "r", encoding="utf-8"))
            m_ext = float(ext.get("M_t", {}).get(period, m_internal))
            w = np.clip(EXT_WEIGHT, 0.0, 1.0)
            return (m_internal ** (1-w)) * (m_ext ** w)
        except Exception:
            pass
    return m_internal


# ==== Few-shot 元学习 / 迁移模块 ====
# 注意把 csv_path 改成你自己 Full_dataset.csv 的实际路径
_FEWSHOT_CFG = FewShotConfig(
    csv_path=r"C:\Users\Lenovo\Desktop\下载\Full_dataset.csv",  # ← 这里改成和 newcat.py 里用的一样，注意这里需要更改！！！
    min_group_size=30,     # 认为样本>=30 就不需要 few-shot 调整
    max_support_size=200,  # 支持集最多取 200 条
    k_neighbors=50,        # KNN 取 50 个邻居
    new_car_max_age=3.0,   # 车龄<=3 年大致认为是“新车”
)

_FEWSHOT_ADAPTER = FewShotKnnMeta(_FEWSHOT_CFG)



def _base_predict_price(d):
    X, period, period_bin, age_val = build_row(d)

    # 模型输出 log(price)
    p10_log = m_p10.predict(X); p50_log = m_p50.predict(X); p90_log = m_p90.predict(X)
    p10 = float(np.exp(np.asarray(p10_log).item()))
    p50 = float(np.exp(np.asarray(p50_log).item()))
    p90 = float(np.exp(np.asarray(p90_log).item()))

    # 市场系数
    M = get_market_multiplier(period)
    p10_m, p50_m, p90_m = p10*M, p50*M, p90*M

    # 分组键（period_bin）
    akey = age_bin(age_val)
    fuel = standardize_fuel(d.get("fuel_type"))
    gear = standardize_gear(d.get("transmission"))
    key_full = f"{akey}|{fuel}|{gear}|{period_bin}"

    # 非对称 CQR 逐级回退
    qlo, qhi = hierarchical_qhat_asym(key_full)

    lo = max(0.0, p10_m - qlo)
    hi = p90_m + qhi
    wr = (hi - lo) / max(1e-6, p50_m)

    return {
        "p50": float(p50_m),
        "lo": float(lo),
        "hi": float(hi),
        "wr": float(wr),
        "group_key": key_full,
        "period": period,
        "period_bin": period_bin,
        "market_multiplier": float(M)
    }
def predict_price(d):
    """
    带 few-shot 迁移学习的最终预测函数：

    1. 先调用 _base_predict_price(d) 做 CatBoost + 市场系数 + CQR；
    2. 再根据训练集中 (brand, model, year) 的样本量，自动对“小样本新车”做 KNN few-shot 迁移；
    3. 对于样本已经很多的老车款，元学习模块自动不做调整。
    """
    base = _base_predict_price(d)
    out  = _FEWSHOT_ADAPTER.maybe_adapt(d, base)
    return out



# demo
if __name__ == "__main__":
    demo = {
        "brand":"Toyota","model":"Corolla","year":2019,"age":None,
        "milage":45000,"fuel_type":"Petrol","engine":1798,"max_power":138,
        "transmission":"Automatic","seats":5
    }
    out = predict_price(demo)
    print("\nPrediction:")
    for k,v in out.items():
        print(f"{k}: {v}")
