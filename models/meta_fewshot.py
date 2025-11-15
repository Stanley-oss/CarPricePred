# meta_fewshot.py
# -*- coding: utf-8 -*-
"""
Few-shot meta-learning adapter (KNN based) for used car price model.

目标：
- 解决“新出产车型数据太少”的问题；
- 对 (Brand, Model, Year) 样本数很少且车龄 Age 较小的车款，使用同品牌同车型的历史样本做 KNN 迁移；
- 样本足够多的老车款则直接用 CatBoost+CQR 的原始预测，不做调整。

数据列假设与 Full_dataset.csv 一致：
Brand, Model, Year, Age, Kilometer, Fuel Type, Engine, Max Power, Transmission, Seats, Price
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
import re
from datetime import datetime


# ===================== 配置 =====================

@dataclass
class FewShotConfig:
    csv_path: str                  # Full_dataset.csv 的路径
    min_group_size: int = 30       # (Brand, Model, Year) 样本数 >= 这个值就认为“不少”，不做 few-shot
    max_support_size: int = 200    # 支持集最多使用多少条样本
    k_neighbors: int = 50          # KNN 中最多取多少邻居
    new_car_max_age: float = 3.0   # Age <= 这个阈值认为是“新车”，才会考虑 few-shot 迁移


# ===================== 工具函数 =====================

_num_pattern = re.compile(r"[-+]?\d*\.?\d+")


def _to_number(x):
    """把各种字符串/数字安全地转为 float，失败返回 np.nan"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x)
    m = _num_pattern.search(s.replace(",", ""))
    return float(m.group(0)) if m else np.nan


def _norm_str(s):
    """统一品牌/型号字符串：小写 + 去前后空格"""
    if s is None:
        return ""
    return str(s).strip().lower()


# ===================== Few-shot KNN Meta 类 =====================

class FewShotKnnMeta:
    """
    使用全量 Full_dataset.csv 作为“经验库”，按 (Brand, Model, Year) 统计样本量：

    - 如果 Age > new_car_max_age：认为不是新车 -> 不做任何调整；
    - 如果 Age <= new_car_max_age 但该 (Brand, Model, Year) 的样本数 >= min_group_size：
        说明其实数据也不算少 -> 也不做调整；
    - 否则：
        从同品牌同车型的历史样本中，按 Age / Kilometer / Engine / Max Power / Seats 做 KNN，
        用邻居的真实价格分布来微调当前车的预测中位数 p50，再整体缩放 (lo, hi) 区间。
    """

    def __init__(self, cfg: FewShotConfig):
        self.cfg = cfg

        # 训练数据缓存
        self._loaded = False
        self._df: pd.DataFrame = None
        self._X: np.ndarray = None          # 归一化后的特征矩阵
        self._y: np.ndarray = None          # 价格
        self._feature_means: np.ndarray = None
        self._feature_stds: np.ndarray = None

        self._group_counts: Dict[str, int] = {}
        self._group_ids: np.ndarray = None  # 每行对应的 group_id（brand||model||year）

    # ---------- 懒加载全量数据 ----------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_full_dataset()
        self._loaded = True

    def _load_full_dataset(self):
        cfg = self.cfg
        df = pd.read_csv(cfg.csv_path)

        # 对应 Full_dataset.csv 的列名
        df = df.rename(columns={
            "Brand": "brand",
            "Model": "model",
            "Year": "year",
            "Age": "age",
            "Kilometer": "milage",
            "Fuel Type": "fuel_type",
            "Engine": "engine",
            "Max Power": "max_power",
            "Transmission": "transmission",
            "Seats": "seats",
            "Price": "price",
        })

        # 数值化
        for col in ["year", "age", "milage", "engine", "max_power", "seats", "price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 如果 Age 缺失，用“当前年份 - Year”近似补一下车龄
        now_year = datetime.now().year
        mask_age_na = df["age"].isna() & df["year"].notna()
        df.loc[mask_age_na, "age"] = now_year - df.loc[mask_age_na, "year"]

        # 基本清洗：价格必须存在
        df = df[df["price"].notna()].copy()

        # 统一 brand/model Key
        df["brand_key"] = df["brand"].apply(_norm_str)
        df["model_key"] = df["model"].apply(_norm_str)
        df["year_int"] = df["year"].astype("Int64")

        # group_id = brand||model||year
        df["_group_id"] = (
            df["brand_key"].fillna("") + "||" +
            df["model_key"].fillna("") + "||" +
            df["year_int"].fillna(-1).astype(int).astype(str)
        )

        # 统计每个 (brand, model, year) 的样本量
        self._group_counts = df["_group_id"].value_counts().to_dict()
        self._group_ids = df["_group_id"].to_numpy()

        # 特征矩阵（KNN 用）
        feat_cols = ["age", "milage", "engine", "max_power", "seats"]
        X_raw = df[feat_cols].to_numpy(dtype=float)

        # 用列均值填 NaN
        col_means = np.nanmean(X_raw, axis=0)
        idx_nan = np.where(~np.isfinite(X_raw))
        X_raw[idx_nan] = np.take(col_means, idx_nan[1])

        # 标准化
        col_stds = np.nanstd(X_raw, axis=0)
        col_stds[col_stds == 0] = 1.0
        X_norm = (X_raw - col_means) / col_stds

        self._df = df
        self._X = X_norm
        self._y = df["price"].to_numpy(dtype=float)
        self._feature_means = col_means
        self._feature_stds = col_stds

    # ---------- 对单条样本做“也许”元学习调整 ----------

    def maybe_adapt(self, d: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
        """
        d: 你在 predict_price 里传进来的原始 dict（包含 brand/model/year/age/milage/engine/max_power/seats 等）
        base: _base_predict_price(d) 的输出字典，必须至少包含：
              - "p50", "lo", "hi", "wr"
        返回：可能经过 few-shot 调整后的结果字典
        """
        self._ensure_loaded()
        cfg = self.cfg

        # ---------- 从输入样本中取出关键信息 ----------
        brand_raw = d.get("brand") or d.get("Brand")
        model_raw = d.get("model") or d.get("Model")
        year_raw = d.get("year") or d.get("Year")
        age_raw = d.get("age") or d.get("Age")

        brand_key = _norm_str(brand_raw)
        model_key = _norm_str(model_raw)
        year_val = _to_number(year_raw)
        age_val = _to_number(age_raw)

        # 尽量保证 Age 有值
        if not np.isfinite(age_val):
            if np.isfinite(year_val):
                age_val = datetime.now().year - year_val

        # 如果 Age 仍然拿不到，就放弃 few-shot
        if not np.isfinite(age_val):
            return base

        # 只对“新车”做 few-shot
        if age_val > cfg.new_car_max_age:
            return base

        # group_id = brand||model||year_int
        year_int = int(year_val) if np.isfinite(year_val) else -1
        group_id = f"{brand_key}||{model_key}||{year_int}"
        n_group = self._group_counts.get(group_id, 0)

        # 如果这个 (brand, model, year) 样本数已经不少了，也不用 few-shot
        if n_group >= cfg.min_group_size:
            return base

        # brand / model 至少要有，才能找“同车型家族”的 support
        if not brand_key or not model_key:
            return base

        df = self._df

        # 从全量数据中，选同品牌同车型（不限制年份）的支持集
        mask_family = (df["brand_key"] == brand_key) & (df["model_key"] == model_key)
        family_idx = np.where(mask_family)[0]

        if len(family_idx) == 0:
            # 训练集中完全没有这个品牌或型号
            return base

        # 控制支持集大小：优先选年份离当前车近的样本
        if len(family_idx) > cfg.max_support_size:
            year_train = df["year_int"].to_numpy()[family_idx].astype(float)
            if np.isfinite(year_val):
                year_diff = np.abs(year_train - year_val)
                order = np.argsort(year_diff)
                family_idx = family_idx[order[:cfg.max_support_size]]
            else:
                family_idx = family_idx[:cfg.max_support_size]

        # ---------- 构造当前样本的特征向量 ----------
        x_query_raw = np.array([
            age_val,
            _to_number(d.get("milage") or d.get("Kilometer")),
            _to_number(d.get("engine") or d.get("Engine")),
            _to_number(d.get("max_power") or d.get("Max Power")),
            _to_number(d.get("seats") or d.get("Seats")),
        ], dtype=float)

        # 用训练集均值填补 NaN
        for j in range(len(x_query_raw)):
            if not np.isfinite(x_query_raw[j]):
                x_query_raw[j] = self._feature_means[j]

        x_query_norm = (x_query_raw - self._feature_means) / self._feature_stds

        # ---------- 在同品牌同车型家族里做 KNN ----------
        X_family = self._X[family_idx]
        diff = X_family - x_query_norm
        dist = np.sqrt(np.sum(diff * diff, axis=1))

        order = np.argsort(dist)
        k = min(cfg.k_neighbors, len(order))
        knn_idx = family_idx[order[:k]]

        y_knn = self._y[knn_idx]
        y_knn = y_knn[np.isfinite(y_knn)]
        if len(y_knn) < 5:
            # 邻居太少，效果不稳定，放弃调整
            return base

        q10_local = float(np.quantile(y_knn, 0.10))
        q50_local = float(np.quantile(y_knn, 0.50))
        q90_local = float(np.quantile(y_knn, 0.90))

        # ---------- 与基础预测融合 ----------
        p50_base = float(base.get("p50", np.nan))
        lo_base = float(base.get("lo", np.nan))
        hi_base = float(base.get("hi", np.nan))

        if not np.isfinite(p50_base) or p50_base <= 0:
            return base

        # 样本越少 & 车越新 -> 权重越大
        w_size = max(0.0, min(1.0, (cfg.min_group_size - n_group) / float(cfg.min_group_size)))
        w_age = max(0.0, min(1.0, (cfg.new_car_max_age - age_val) / float(cfg.new_car_max_age)))
        w = w_size * w_age

        if w <= 0.0:
            return base

        # 融合后的中位数，夹在本家族 q10~q90 之间
        p50_meta = (1.0 - w) * p50_base + w * q50_local
        p50_meta = float(np.clip(p50_meta, q10_local, q90_local))

        scale = p50_meta / p50_base if p50_base > 0 else 1.0
        lo_meta = lo_base * scale
        hi_meta = hi_base * scale

        # 组装输出
        out = dict(base)  # 复制一份
        out["p50_before_meta"] = p50_base
        out["lo_before_meta"] = lo_base
        out["hi_before_meta"] = hi_base

        out["p50"] = float(p50_meta)
        out["lo"] = float(max(0.0, lo_meta))
        out["hi"] = float(max(out["lo"], hi_meta))
        out["wr"] = float((out["hi"] - out["lo"]) / max(1e-6, out["p50"]))

        out["meta_info"] = {
            "enabled": True,
            "is_new_car": True,
            "age": float(age_val),
            "n_group": int(n_group),
            "w_meta": float(w),
            "k_neighbors": int(k),
            "support_size": int(len(family_idx)),
            "q10_local": q10_local,
            "q50_local": q50_local,
            "q90_local": q90_local,
        }
        return out
