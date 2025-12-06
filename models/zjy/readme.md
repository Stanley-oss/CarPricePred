Used Car Price Prediction with Quantile Regression & Few-Shot Meta-Learning

二手车价格预测：CatBoost 分位数回归 + 市场指数校准 + CQR + Few-Shot KNN 元学习插件。

This repository contains the **inference pipeline** for a used-car price prediction system.  
The core idea:

1. Use three CatBoost quantile models to predict **P10 / P50 / P90** of the log-price.  
2. Adjust predictions with a **time-varying market index**.
3. Calibrate the prediction interval using **Conformal Quantile Regression (CQR)**.
4. For **new + few-shot car models**, apply a **KNN-based meta-learner** (`meta_fewshot.py`) to refine the final price range using same-brand & same-model historical neighbors.

---

Repository Structure

```text
.
├─ predict_price.py        # Main inference script (入口脚本)
├─ meta_fewshot.py         # Few-shot meta-learning adapter (KNN-based)
├─ cqr_meta.json           # Trained meta config: features, market index, CQR stats
├─ catboost_p10.joblib     # CatBoost model for 10% quantile (log-price)
├─ catboost_p50.joblib     # CatBoost model for 50% quantile (log-price)
├─ catboost_p90.joblib     # CatBoost model for 90% quantile (log-price)
└─ Full_dataset.csv        # Full training dataset used as "experience base" for KNN
