import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FewShotConfig:
    csv_path: str
    min_group_size: int = 30
    max_support_size: int = 200
    k_neighbors: int = 50
    new_car_max_age: float = 3.0
    iqr_to_width_factor: float = 1.8
    min_uncertainty_scale: float = 0.7
    max_uncertainty_scale: float = 1.6

# ===================== 工具函数 =====================

_NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

def _to_number(x: Any) -> float:
    """
    将输入安全转换为 float。
    支持：数字、包含数字的字符串。
    异常：None 或无法解析时返回 np.nan。
    """
    if x is None:
        return np.nan
    if isinstance(x, (float, int, np.number)):
        return float(x) if not np.isnan(x) else np.nan
    
    s = str(x).replace(",", "")
    match = _NUM_PATTERN.search(s)
    return float(match.group(0)) if match else np.nan


def _normalize_string(s: Any) -> str:
    """标准化字符串：转小写并去除首尾空格。"""
    if s is None:
        return ""
    return str(s).strip().lower()


# ===================== Few-shot KNN Meta 类 =====================

class FewShotKnnMeta:
    """
    基于 KNN 的 Few-Shot 元学习适配器。
    
    逻辑：
    1. 针对新车 (Age <= 3) 且 历史样本稀缺 (< 30) 的情况启用。
    2. 在同品牌、同型号的历史数据中，搜索特征相似的 "邻居"。
    3. 利用邻居的价格分布 (IQR, Median) 修正主模型的预测结果。
    """

    def __init__(self, cfg: FewShotConfig, feature_weights: dict[str, float] | None = None):
        self.cfg = cfg
        
        # 内部状态
        self._loaded = False
        self._df: pd.DataFrame | None = None
        self._X: np.ndarray | None = None          # 归一化特征矩阵
        self._y: np.ndarray | None = None          # 价格标签
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        
        # 索引与元数据
        self._group_counts: dict[str, int] = {}
        self._group_ids: np.ndarray | None = None # 对应每一行的 group_id
        
        # 特征定义
        self._feat_cols = ["age", "milage", "engine", "max_power", "seats"]
        
        # 初始化特征权重
        self._feature_weights = self._init_feature_weights(feature_weights)

    def _init_feature_weights(self, weights_dict: dict[str, float] | None) -> np.ndarray:
        """计算并归一化特征权重向量"""
        weights_dict = weights_dict or {}
        w_list = [float(weights_dict.get(col, 1.0)) for col in self._feat_cols]
        w_arr = np.array([max(0.0, w) for w in w_list], dtype=float)
        
        if not np.any(w_arr > 0):
            w_arr = np.ones_like(w_arr)
        
        # 归一化使其均值为 1
        return w_arr / w_arr.mean()

    def _ensure_loaded(self):
        """惰性加载数据"""
        if not self._loaded:
            self._load_full_dataset()
            self._loaded = True

    def _load_full_dataset(self):
        """加载并预处理全量训练数据 (经验库)"""
        df = pd.read_csv(self.cfg.csv_path)

        # 1. 字段重命名标准化
        column_map = {
            "Brand": "brand", "Model": "model", "Year": "year", 
            "Age": "age", "Kilometer": "milage", "Fuel Type": "fuel_type",
            "Engine": "engine", "Max Power": "max_power", 
            "Transmission": "transmission", "Seats": "seats", "Price": "price"
        }
        df = df.rename(columns=column_map)

        # 2. 数值转换
        numeric_cols = ["year", "age", "milage", "engine", "max_power", "seats", "price"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. 缺失值处理 (Age 缺失用 Year 补全)
        now_year = datetime.now().year
        mask_age_na = df["age"].isna() & df["year"].notna()
        df.loc[mask_age_na, "age"] = now_year - df.loc[mask_age_na, "year"]

        # 移除无价格样本
        df = df.dropna(subset=["price"]).copy()

        # 4. 生成索引 Key
        df["brand_key"] = df["brand"].apply(_normalize_string)
        df["model_key"] = df["model"].apply(_normalize_string)
        df["year_int"] = df["year"].astype("Int64").fillna(-1)

        # 构造 Group ID: brand||model||year
        # 用于快速判断样本是否稀缺
        df["_group_id"] = (
            df["brand_key"].fillna("") + "||" +
            df["model_key"].fillna("") + "||" +
            df["year_int"].astype(str)
        )

        # 5. 统计样本分布
        self._group_counts = df["_group_id"].value_counts().to_dict()
        self._group_ids = df["_group_id"].to_numpy()

        # 6. 准备 KNN 特征矩阵
        X_raw = df[self._feat_cols].to_numpy(dtype=float)
        
        # 填充 NaN (列均值)
        col_means = np.nanmean(X_raw, axis=0)
        inds = np.where(np.isnan(X_raw))
        X_raw[inds] = np.take(col_means, inds[1])
        
        # Z-Score 标准化
        col_stds = np.nanstd(X_raw, axis=0)
        col_stds[col_stds == 0] = 1.0  # 防止除以 0
        X_norm = (X_raw - col_means) / col_stds

        # 保存状态
        self._df = df
        self._X = X_norm
        self._y = df["price"].to_numpy(dtype=float)
        self._feature_means = col_means
        self._feature_stds = col_stds

    def maybe_adapt(self, d: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()

        if (self._df is None or 
            self._feature_means is None or 
            self._feature_stds is None or 
            self._X is None or 
            self._y is None):
            return base
        
        df = self._df 
        feature_means = self._feature_means
        feature_stds = self._feature_stds
        X_full = self._X
        y_full = self._y

        cfg = self.cfg

        # --- 1. 提取关键信息 ---
        brand_key = _normalize_string(d.get("brand") or d.get("Brand"))
        model_key = _normalize_string(d.get("model") or d.get("Model"))
        year_val = _to_number(d.get("year") or d.get("Year"))
        age_val = _to_number(d.get("age") or d.get("Age"))

        # 尝试补全 Age
        if np.isnan(age_val) and not np.isnan(year_val):
            age_val = datetime.now().year - year_val

        # --- 2. 资格校验 (Early Exit) ---
        if np.isnan(age_val):
            return base

        if age_val > cfg.new_car_max_age:
            return base

        year_int = int(year_val) if not np.isnan(year_val) else -1
        group_id = f"{brand_key}||{model_key}||{year_int}"
        n_group = self._group_counts.get(group_id, 0)

        if n_group >= cfg.min_group_size:
            return base
        
        if not brand_key or not model_key:
            return base

        # --- 3. 构建支持集 (Support Set) ---
        mask_family = (df["brand_key"] == brand_key) & (df["model_key"] == model_key)
        family_idx = np.where(mask_family)[0]

        if len(family_idx) == 0:
            return base

        if len(family_idx) > cfg.max_support_size:
            if not np.isnan(year_val):
                year_train = df["year_int"].to_numpy()[family_idx].astype(float)
                year_diff = np.abs(year_train - year_val)
                top_k_indices = np.argsort(year_diff)[:cfg.max_support_size]
                family_idx = family_idx[top_k_indices]
            else:
                family_idx = family_idx[:cfg.max_support_size]

        # --- 4. KNN 搜索 ---
        query_vals = [
            age_val,
            _to_number(d.get("milage") or d.get("Kilometer")),
            _to_number(d.get("engine") or d.get("Engine")),
            _to_number(d.get("max_power") or d.get("Max Power")),
            _to_number(d.get("seats") or d.get("Seats"))
        ]
        
        x_query_raw = np.array(query_vals, dtype=float)
        
        mask_nan = np.isnan(x_query_raw)
        x_query_raw[mask_nan] = feature_means[mask_nan]
        
        # 标准化
        x_query_norm = (x_query_raw - feature_means) / feature_stds

        # 计算加权欧氏距离 (使用本地变量 X_full)
        X_family = X_full[family_idx]
        diff = (X_family - x_query_norm) * self._feature_weights
        dist = np.sqrt(np.sum(diff ** 2, axis=1))

        # 取最近邻
        k = min(cfg.k_neighbors, len(dist))
        nearest_indices = np.argsort(dist)[:k]
        knn_global_idx = family_idx[nearest_indices]

        # 使用本地变量 y_full
        y_knn = y_full[knn_global_idx]
        y_knn = y_knn[~np.isnan(y_knn)]

        if len(y_knn) < 5:
            return base

        # --- 5. 计算统计量与融合权重 ---
        q10_local, q25_local, q50_local, q75_local, q90_local = np.quantile(
            y_knn, [0.10, 0.25, 0.50, 0.75, 0.90]
        )
        iqr_local = max(0.0, q75_local - q25_local)

        p50_base = float(base.get("p50", np.nan))
        lo_base = float(base.get("lo", np.nan))
        hi_base = float(base.get("hi", np.nan))

        if np.isnan(p50_base) or p50_base <= 0:
            return base

        w_size = max(0.0, min(1.0, (cfg.min_group_size - n_group) / float(cfg.min_group_size)))
        w_age = max(0.0, min(1.0, (cfg.new_car_max_age - age_val) / float(cfg.new_car_max_age)))
        w = w_size * w_age

        if w <= 0.0:
            return base

        # --- 6. 融合与区间调整 ---
        p50_meta = (1.0 - w) * p50_base + w * q50_local
        p50_meta = float(np.clip(p50_meta, q10_local, q90_local))

        width_base = max(0.0, hi_base - lo_base)
        s_unc = 1.0
        if iqr_local > 0.0 and width_base > 0.0:
            width_target = iqr_local * cfg.iqr_to_width_factor
            ratio = width_target / width_base
            s_unc = float(np.clip(ratio, cfg.min_uncertainty_scale, cfg.max_uncertainty_scale))

        # 计算新的区间 (保留原有逻辑顺序)
        delta_lo_base = max(0.0, p50_base - lo_base)
        delta_hi_base = max(0.0, hi_base - p50_base)
        
        # 调试用：逻辑1计算结果
        lo_meta_iqr = p50_meta - (delta_lo_base * s_unc)
        hi_meta_iqr = p50_meta + (delta_hi_base * s_unc)

        # 实际用：逻辑2计算结果 (整体缩放)
        scale = p50_meta / p50_base if p50_base > 0 else 1.0
        lo_meta = lo_base * scale
        hi_meta = hi_base * scale
        
        lo_meta = float(max(0.0, lo_meta))
        hi_meta = float(max(lo_meta, hi_meta))
        wr_meta = (hi_meta - lo_meta) / max(1e-6, p50_meta)

        # --- 7. 组装输出 ---
        out = base.copy()
        out.update({
            "p50_before_meta": p50_base,
            "lo_before_meta": lo_base,
            "hi_before_meta": hi_base,
            "p50": p50_meta,
            "lo": lo_meta,
            "hi": hi_meta,
            "wr": wr_meta,
            "meta_info": {
                "enabled": True,
                "is_new_car": True,
                "age": float(age_val),
                "n_group": int(n_group),
                "w_meta": float(w),
                "k_neighbors": int(k),
                "support_size": int(len(family_idx)),
                "q10_local": float(q10_local),
                "q50_local": float(q50_local),
                "q90_local": float(q90_local),
                "iqr_local": float(iqr_local),
                "uncertainty_scale": float(s_unc),
                "_debug_lo_iqr": lo_meta_iqr, 
                "_debug_hi_iqr": hi_meta_iqr
            }
        })
        
        if self._feature_weights is not None:
            out["meta_info"]["feature_weights"] = dict(zip(self._feat_cols, self._feature_weights,strict=True))

        return out