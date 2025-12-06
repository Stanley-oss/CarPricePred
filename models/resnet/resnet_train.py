import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CSV_PATH = "../../datasets/Full_dataset.csv"


def seed_everything(seed=42):
    # 1. Python 原生随机
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Numpy 随机 (用于数据处理)
    np.random.seed(seed)

    # 3. PyTorch 随机 (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多显卡

    # 4. 保证卷积算法确定性 (会稍微牺牲一点速度，但保证结果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


# ==========================================
# 1. 核心模块: ResNet Block
# ==========================================
class ResNetBlock(nn.Module):
    """
    ResNet 残差块
    作用: 防止深层网络梯度消失，允许模型学习从'基础特征'到'价格'的微细非线性映射。
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)  # Skip Connection


# ==========================================
# 2. 主模型: Tabular ResNet (仅预测物理残值)
# ==========================================
class TabularResNet(nn.Module):
    def __init__(
        self,
        num_numerical,
        cat_dims,
        embedding_dims,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.2,
    ):
        super().__init__()

        # 类别嵌入 (Model, Transmission 等，不含 Brand/Fuel)
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(cat_dims, embedding_dims, strict=True)])
        total_emb_dim = sum(embedding_dims)

        # 数值特征归一化
        self.num_bn = nn.BatchNorm1d(num_numerical)

        # 输入投影
        input_dim = total_emb_dim + num_numerical
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

        # ResNet 主干
        self.resnet_layers = nn.ModuleList([ResNetBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_blocks)])

        # 输出层 (输出 Log Core Price)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_num, x_cat):
        emb_outputs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outputs, dim=1)
        x_num = self.num_bn(x_num)
        x = torch.cat([x_emb, x_num], dim=1)
        x = self.input_proj(x)
        for layer in self.resnet_layers:
            x = layer(x)
        return self.head(x)


# ==========================================
# 3. 数据集定义
# ==========================================
class CarDataset(data.Dataset):
    def __init__(self, X_num, X_cat, y_core, bias_brand, bias_fuel, raw_price):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y_core = torch.tensor(y_core, dtype=torch.float32)
        self.bias_brand = torch.tensor(bias_brand, dtype=torch.float32)
        self.bias_fuel = torch.tensor(bias_fuel, dtype=torch.float32)
        self.raw_price = torch.tensor(raw_price, dtype=torch.float32)

    def __len__(self):
        return len(self.y_core)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y_core[idx],
            self.bias_brand[idx],
            self.bias_fuel[idx],
            self.raw_price[idx],
        )


# ==========================================
# 4. 数据处理策略 (Global Bias + Clean Training)
# ==========================================
def get_data_strategy(file_path, batch_size=64):
    print("Loading & Cleaning Data...")
    df = pd.read_csv(file_path)

    # 补上缺失值
    df["Max Power"] = pd.to_numeric(df["Max Power"], errors="coerce")
    for col in ["Engine", "Max Power"]:
        df[col] = df[col].fillna(df[col].median())

    # 转换为Log，第一次缓解长尾影响
    df["log_price"] = np.log1p(df["Price"])
    global_mean_log = df["log_price"].mean()

    # =======================================================
    # Bias 计算
    # 这里是全局的偏置
    # =========================================================
    print("Computing Global Biases (Brand & Fuel)...")
    le_brand = LabelEncoder()
    df["Brand_ID"] = le_brand.fit_transform(df["Brand"])  # 把品牌转换成编号
    # Brand Bias = Brand Mean - Global Mean
    brand_map = df.groupby("Brand_ID")["log_price"].mean() - global_mean_log  # 计算每个品牌相对于平均的偏置
    df["bias_brand"] = df["Brand_ID"].map(brand_map)  # 再把算好的偏置加到数据里

    le_fuel = LabelEncoder()  # 一样的操作针对燃油类型再来一遍
    df["Fuel_ID"] = le_fuel.fit_transform(df["Fuel Type"])
    # Fuel Bias = Fuel Mean - Global Mean
    fuel_map = df.groupby("Fuel_ID")["log_price"].mean() - global_mean_log
    df["bias_fuel"] = df["Fuel_ID"].map(fuel_map)

    # =======================================================
    # 4.3 计算核心残差 (Core Residual)
    # y_core = 除去了品牌和燃油类型溢价后的纯粹车价，第二次缓解长尾的影响
    # =======================================================
    df["y_core_raw"] = df["log_price"] - df["bias_brand"] - df["bias_fuel"]

    # 把其他的标签也变成数值
    le_model = LabelEncoder()
    df["Model_ID"] = le_model.fit_transform(df["Model"])
    le_trans = LabelEncoder()
    df["Trans_ID"] = le_trans.fit_transform(df["Transmission"])

    cat_cols = ["Model_ID", "Trans_ID"]
    X_cat = df[cat_cols].values
    X_num = df.drop(
        columns=[
            "Price",
            "log_price",
            "Brand",
            "Model",
            "Fuel Type",
            "Transmission",
            "Brand_ID",
            "Fuel_ID",
            "Model_ID",
            "Trans_ID",
            "bias_brand",
            "bias_fuel",
            "y_core_raw",
        ]
    ).values

    # =======================================================
    # 准备数据集：这里仅仅使用没有溢价的那部分基础车型算出正常的价钱
    # =======================================================
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 截断来自10%最离谱豪车的数据，第三次缓解长尾影响（经过实验10%是比较合理的）
    train_core_values = df.iloc[train_idx]["y_core_raw"]
    core_threshold = train_core_values.quantile(0.90)

    # 过滤掉离谱贵车
    clean_mask = train_core_values <= core_threshold
    final_train_idx = train_idx[clean_mask]

    print("Form Training Set")
    print(f"Original Train Size: {len(train_idx)}")
    print(f"Filtered Train Size: {len(final_train_idx)} (Only 'Basic' Logic used for Backbone)")
    print(f"Test Set Size:{len(test_idx)} (Includes ALL cars, luxury & basic)")

    # 辅助函数: 根据索引打包数据
    def pack_data(idx_list):
        return (
            X_num[idx_list],
            X_cat[idx_list],
            df.iloc[idx_list]["y_core_raw"].values,
            df.iloc[idx_list]["bias_brand"].values,
            df.iloc[idx_list]["bias_fuel"].values,
            df.iloc[idx_list]["Price"].values,
        )

    train_data = pack_data(final_train_idx)  # 只包含基础车
    test_data = pack_data(test_idx)  # 包含豪车和基础车

    # 标准化 (Fit on Clean Train, Transform on All)
    scaler_x = StandardScaler()
    scaler_x.fit(train_data[0])
    X_num_train = scaler_x.transform(train_data[0])
    X_num_test = scaler_x.transform(test_data[0])

    scaler_y = StandardScaler()
    y_core_train = scaler_y.fit_transform(train_data[2].reshape(-1, 1)).flatten()
    y_core_test = scaler_y.transform(test_data[2].reshape(-1, 1)).flatten()

    # Dataloaders
    train_ds = CarDataset(
        X_num_train,
        train_data[1],
        y_core_train,
        train_data[3],
        train_data[4],
        train_data[5],
    )
    test_ds = CarDataset(X_num_test, test_data[1], y_core_test, test_data[3], test_data[4], test_data[5])

    cat_dims = [len(le_model.classes_), len(le_trans.classes_)]
    emb_dims = [min(50, (d + 1) // 2) for d in cat_dims]  # 计算有多少个嵌入维度

    return (
        (
            data.DataLoader(train_ds, batch_size, shuffle=True),
            data.DataLoader(test_ds, batch_size, shuffle=False),
        ),
        X_num.shape[1],
        cat_dims,
        emb_dims,
        scaler_y,
        le_brand,
        le_fuel,
    )


# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    seed_everything(42)
    BATCH_SIZE = 64
    LR = 2e-3
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据
    (train_dl, test_dl), n_num, cat_dims, emb_dims, scaler_y, le_brand, le_fuel = get_data_strategy(CSV_PATH, BATCH_SIZE)

    model = TabularResNet(n_num, cat_dims, emb_dims, hidden_dim=256, num_blocks=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.MSELoss()

    print("Training Backbone")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_num, x_cat, y_core, _, _, _ in train_dl:
            x_num, x_cat, y_core = x_num.to(DEVICE), x_cat.to(DEVICE), y_core.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_num, x_cat).squeeze()
            loss = criterion(pred, y_core)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Train MSE (Core Logic): {total_loss / len(train_dl):.4f}")

    # 6.最终评估
    print("Evaluation on FULL Test Set (Basic + Luxury)")
    model.eval()
    results = []

    header_printed = 1
    with torch.no_grad():
        for x_num, x_cat, _y_core, bias_b, bias_f, raw_price in test_dl:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            # 1. ResNet 预测"普通车况价值" (Normalized Log scale)
            pred_core_norm = model(x_num, x_cat).cpu().numpy().flatten()
            # 2. 反归一化
            pred_core_log = scaler_y.inverse_transform(pred_core_norm.reshape(-1, 1)).flatten()
            # 3. 加上之前算好的 Bias (Formula: Core + Brand + Fuel)
            final_log_pred = pred_core_log + bias_b.numpy() + bias_f.numpy()
            final_pred_price = np.expm1(final_log_pred)
            actual_price = raw_price.numpy()
            # 收集结果用于计算整体指标
            batch_res = np.vstack([final_pred_price, actual_price, bias_b.numpy(), bias_f.numpy()]).T
            results.append(batch_res)
            # 打印样本分析
            if header_printed <= 10:
                print("-" * 100)
                print(f"{'Type':<10}|'Brand Coeff'|'Fuel Coeff'|'Base Price'|'Final Pred'|'Actual'")
                print("-" * 100)
                for j in range(min(10, len(final_pred_price))):
                    # 将 Log Bias 转为倍数显示 (e.g., 0.69 -> x2.0)
                    b_coeff = np.exp(bias_b[j])
                    f_coeff = np.exp(bias_f[j])
                    base_p = np.expm1(pred_core_log[j])

                    # 简单判断这个样本是否算豪车
                    car_type = "Luxury" if bias_b[j] > 0.3 else "Basic"

                    print(
                        f"{car_type:<10}|x{b_coeff:.2f}:<15|x{f_coeff:.2f}|{base_p:,.0f}|{final_pred_price[j]:,.0f}|{actual_price[j]:,.0f}"
                    )
                print("-" * 100)
                header_printed += 1

    # 汇总计算
    all_res = np.vstack(results)
    preds = all_res[:, 0]
    targets = all_res[:, 1]

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    print("\nFinal Results on Model:")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")


if __name__ == "__main__":
    main()
