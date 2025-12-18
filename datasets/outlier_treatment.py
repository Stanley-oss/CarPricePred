import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

FILE_PATH = r"./Full_dataset.csv"
RANDOM_STATE = 42

# 1. Load data
if os.path.exists(FILE_PATH):
    df = pd.read_csv(FILE_PATH)
else:
    raise FileNotFoundError("No dataset file found")

# 2. Field filtering
cols_to_keep = [
    'Brand', 'Model', 'Age', 'Kilometer', 'Fuel Type', 
    'Transmission', 'Engine', 'Max Power', 'Seats', 'Price'
]
cols_to_keep = [c for c in cols_to_keep if c in df.columns]
df = df[cols_to_keep].copy()

# 3. Basic cleaning
if 'Brand' in df.columns:
    df['Brand'] = df['Brand'].fillna('Unknown')
if 'Model' in df.columns:
    df['Model'] = df['Model'].fillna('Unknown')

if df['Max Power'].dtype == 'O':
    df['Max Power'] = pd.to_numeric(
        df['Max Power'].astype(str).str.extract(r'(\d+\.?\d*)')[0], 
        errors='coerce'
    )

df['Log_Price'] = np.log1p(df['Price'])

# Category Code
cat_cols = ['Fuel Type', 'Transmission']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')
        df[col + '_Code'] = LabelEncoder().fit_transform(df[col].astype(str))

# KNN interpolation
print("KNN interpolation")
cols_num = ['Engine', 'Max Power', 'Seats']
cols_num = [c for c in cols_num if c in df.columns]

df_raw_vis = df.copy()
imputer = KNNImputer(n_neighbors=5)

df[cols_num] = imputer.fit_transform(df[cols_num])
print("KNN Completed")

plt.figure(figsize=(10, 6))
sns.kdeplot(df_raw_vis['Max Power'].dropna(), color='gray', linestyle='--', label='Before Imputation', fill=True, alpha=0.3)
sns.kdeplot(df['Max Power'], color='blue', label='After KNN', fill=False, linewidth=2)
plt.title("KNN Imputation Effect")
plt.legend()
plt.savefig('1_KNN_Imputation_Effect.png', dpi=300, bbox_inches='tight')

# Feature engineering (retaining all features)
df['Age_Squared'] = df['Age'] ** 2
df['Km_per_Year'] = df['Kilometer'] / (df['Age'] + 0.1)
df['Power_per_Seat'] = df['Max Power'] / (df['Seats'] + 1e-5)

# Determine the list of numerical features for modelling
base_features = ['Age', 'Kilometer', 'Engine', 'Max Power', 'Seats', 'Fuel Type_Code', 'Transmission_Code']
fe_features = base_features + ['Age_Squared', 'Km_per_Year', 'Power_per_Seat']
fe_features = [f for f in fe_features if f in df.columns]

final_features = fe_features 

# Correlation Heatmap (Full Feature Display)
plt.figure(figsize=(12, 10))

final_corr = df[final_features + ['Log_Price']].corr()
mask = np.triu(np.ones_like(final_corr, dtype=bool))

sns.heatmap(final_corr, mask=mask, annot=True, cmap='coolwarm', 
            vmin=-1, vmax=1, fmt='.2f', square=True)

plt.title("Feature Correlation Matrix (All Features Kept)")
plt.savefig('2_Correlation_Analysis_Matrix.png', dpi=300, bbox_inches='tight')

# 双层异常检测
df_fe = df.copy()

# Layer 1: Isolation Forest 
iso_feats = [f for f in ['Age', 'Kilometer', 'Engine', 'Max Power'] if f in df.columns]
iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
iso_labels = iso.fit_predict(df_fe[iso_feats])
mask_layer1 = (iso_labels == 1)

# Layer 2: GBDT 
X_clean = df_fe[final_features]
y_clean = df_fe['Log_Price']

# （分位数预测）
gbr_low = GradientBoostingRegressor(loss='quantile', alpha=0.01, n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
gbr_high = GradientBoostingRegressor(loss='quantile', alpha=0.99, n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
gbr_low.fit(X_clean, y_clean)
gbr_high.fit(X_clean, y_clean)

mask_layer2 = (y_clean >= gbr_low.predict(X_clean)) & (y_clean <= gbr_high.predict(X_clean))
mask_both = mask_layer1 & mask_layer2
print("过滤完成")

# 消融实验
print("消融实验开始")

def get_cv_rmse(df_subset, name=""):
    if len(df_subset) < 10: 
        return 0
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmses = []
    X = df_subset[final_features]
    y = df_subset['Log_Price']
    
    for train_idx, test_idx in kf.split(X, y):
        model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=RANDOM_STATE)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        rmses.append(root_mean_squared_error(y.iloc[test_idx], y_pred))
    
    print(f"   - {name:<20} Avg RMSE: {np.mean(rmses):.5f}")
    return np.mean(rmses)

results = {
    '1. Baseline': get_cv_rmse(df_fe, "Baseline"),
    '2. Layer 1 Only': get_cv_rmse(df_fe[mask_layer1], "Layer 1 Only"),
    '3. Layer 2 Only': get_cv_rmse(df_fe[mask_layer2], "Layer 2 Only"),
    '4. Both Layers': get_cv_rmse(df_fe[mask_both], "Both Layers")
}

ablation_df = pd.DataFrame(list(results.items()), columns=['Method', 'RMSE'])

plt.figure(figsize=(10, 6))
bars = sns.barplot(
    x='Method', y='RMSE', data=ablation_df, 
    palette=['#d9d9d9', '#a1d99b', '#74c476', '#238b45'], 
    edgecolor='black'
)

plt.title("Ablation Study: Cleaning Strategy Comparison", fontsize=14)
plt.ylabel("RMSE (Lower is Better)")
plt.xticks(rotation=15)
plt.ylim(0, ablation_df['RMSE'].max() * 1.15) 

for i, (_, row) in enumerate(ablation_df.iterrows()):
    bars.text(i, row.RMSE + 0.002, f'{row.RMSE:.4f}', 
              color='black', ha="center", fontweight='bold', fontsize=11)

plt.savefig('3_Ablation_Study.png', dpi=300, bbox_inches='tight')

# 输出 
df_final = df_fe[mask_both].copy()

meta_cols = [c for c in ['Brand', 'Model', 'Price'] if c in df_final.columns]
output_cols = meta_cols + final_features + ['Log_Price']

output_file = "Processed_Data_All_Features.csv"
df_final[output_cols].to_csv(output_file, index=False)