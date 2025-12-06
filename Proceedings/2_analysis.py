import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Dataset/Full_dataset.csv")

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

numeric_cols = ['Year', 'Age', 'Kilometer', 'Engine', 'Max Power', 'Seats', 'Price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 相关性热力图
plt.figure(figsize=(10, 8))
numeric_df = df[numeric_cols].copy()
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# 回归分析
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
features = ['Kilometer', 'Age', 'Max Power']
for i, feature in enumerate(features):
    sns.regplot(x=feature, y='Price', data=df, ax=axes[i], 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    axes[i].set_title(f'Regression: Price vs {feature}')
plt.tight_layout()
plt.show()

# 每个特征的分布
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, color="#4C72B0")
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.show()

# 价格箱线图
plt.figure(figsize=(6, 5))
sns.boxplot(x=df["Price"], color="#DD8452")
plt.title("Price Box Graph")
plt.tight_layout()
plt.show()

# 数值特征标准化箱线图
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

plt.figure(figsize=(10, 6))
sns.boxplot(data=scaled_df, palette="Set2")
plt.title("All feature Box Graph")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Pairplot
subset = ['Price', 'Age', 'Kilometer', 'Max Power']
sns.pairplot(df[subset], diag_kind='kde', plot_kws={'alpha': 0.7})
plt.show()