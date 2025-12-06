基于分位数回归和少样本元学习的二手车价格预测

二手车价格预测：CatBoost 分位数回归 + 市场指数校准 + CQR + 少样本 KNN 元学习插件。

本仓库包含二手车价格预测系统的推理流程。

核心思想：

使用三个 CatBoost 分位数模型预测对数价格的 P10 / P50 / P90。

使用随时间变化的市场指数调整预测值。

使用共形分位数回归 (CQR) 校准预测区间。

对于新的和少样本的二手车模型，应用基于 KNN 的元学习器 (meta_fewshot.py)，利用同品牌同型号车辆的历史价格来优化最终价格范围。

仓库结构

```text
.
├─ predict_price.py        # Main inference script (入口脚本)
├─ meta_fewshot.py         # Few-shot meta-learning adapter (KNN-based)
├─ cqr_meta.json           # Trained meta config: features, market index, CQR stats
├─ catboost_p10.joblib     # CatBoost model for 10% quantile (log-price)
├─ catboost_p50.joblib     # CatBoost model for 50% quantile (log-price)
├─ catboost_p90.joblib     # CatBoost model for 90% quantile (log-price)
└─ Full_dataset.csv        # Full training dataset used as "experience base" for KNN
