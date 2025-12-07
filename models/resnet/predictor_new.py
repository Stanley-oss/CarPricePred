import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
CSV_PATH = os.path.join(CURRENT_DIR, "../../datasets/Full_dataset.csv")
BATCH_SIZE = 64
LR = 2e-3
EPOCHS = 100


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
    def __init__(self, X_num, X_cat, y_core, composite_bias, raw_price):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y_core = torch.tensor(y_core, dtype=torch.float32)
        # 这里只存一个最终计算好的 bias (包含了 Brand/Model/Fuel 的综合影响)
        self.composite_bias = torch.tensor(composite_bias, dtype=torch.float32)
        self.raw_price = torch.tensor(raw_price, dtype=torch.float32)

    def __len__(self):
        return len(self.y_core)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y_core[idx],
            self.composite_bias[idx],
            self.raw_price[idx],
        )


# ==========================================
# 4. 数据处理策略 (Global Bias + Clean Training)
# ==========================================
class ResnetCarPricePredictor:
    def __init__(self, model_dir=MODEL_DIR, csv_path=CSV_PATH):
        self.model_dir = model_dir
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 文件路径
        self.model_path = os.path.join(model_dir, "resnet_model.pth")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

        # 状态占位符
        self.model = None
        self.preprocessor_data = {}  # 存放 scaler, encoders, bias maps, medians

        # 创建目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def get_data_strategy(self, file_path, batch_size=64):
        """
        重构后的数据处理，增加返回值以便保存预处理参数
        """
        print("Loading & Cleaning Data...")
        df = pd.read_csv(file_path)

        # 1. 强制全小写标准化
        for col in ["Brand", "Model", "Fuel Type", "Transmission"]:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # 2. 构造组合键 (Brand + Model)
        df["Brand_Model"] = df["Brand"] + "|" + df["Model"]

        df["Max Power"] = pd.to_numeric(df["Max Power"], errors="coerce")
        impute_values = {}
        for col in ["Engine", "Max Power"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            impute_values[col] = median_val

        # 转换为Log，第一次缓解长尾影响
        df["log_price"] = np.log1p(df["Price"])
        global_mean_log = df["log_price"].mean()

        # =======================================================
        # 层级 Bias 计算
        # 优先使用 (Brand+Model) 的均值，如果样本少，回退到 Brand 均值
        # =======================================================
        print("Computing Hierarchical Biases...")

        # A. 计算 Brand Bias
        le_brand = LabelEncoder()
        df["Brand_ID"] = le_brand.fit_transform(df["Brand"])
        brand_map = (df.groupby("Brand_ID")["log_price"].mean() - global_mean_log).to_dict()

        # B. 计算 Model (Brand_Model) Bias
        # 我们只信任样本数 > 5 的车型偏置，否则容易过拟合
        model_counts = df["Brand_Model"].value_counts()
        valid_models = model_counts[model_counts >= 5].index

        # 计算每个 Brand_Model 的 Bias
        bm_group = df.groupby("Brand_Model")["log_price"].mean()
        bm_bias_map = (bm_group - global_mean_log).to_dict()

        # C. 计算 Fuel Bias (作为独立叠加项)
        le_fuel = LabelEncoder()
        df["Fuel_ID"] = le_fuel.fit_transform(df["Fuel Type"])
        fuel_map = (df.groupby("Fuel_ID")["log_price"].mean() - global_mean_log).to_dict()

        # D. 为每行数据分配 Bias
        # 逻辑：Baseline = Global + Fuel_Bias + (Model_Bias if exists else Brand_Bias)

        def get_hierarchical_bias(row):
            f_bias = fuel_map.get(row["Fuel_ID"], 0.0)

            # 尝试获取车型级 Bias
            bm_key = row["Brand_Model"]
            if bm_key in bm_bias_map and model_counts[bm_key] >= 5:
                # 找到了具体的车型偏置 (例如 Toyota|Corolla 的偏置)
                base_bias = bm_bias_map[bm_key]
            else:
                # 没找到，或者样本太少，回退到 Brand 偏置 (例如 Toyota 均值)
                base_bias = brand_map.get(row["Brand_ID"], 0.0)

            return f_bias + base_bias

        df["composite_bias"] = df.apply(get_hierarchical_bias, axis=1)

        # y_core 是去除了这些偏置后的残差
        df["y_core_raw"] = df["log_price"] - global_mean_log - df["composite_bias"]

        # 把其他的标签也变成数值
        le_model = LabelEncoder()
        df["Model_ID"] = le_model.fit_transform(df["Model"])
        le_trans = LabelEncoder()
        df["Trans_ID"] = le_trans.fit_transform(df["Transmission"])

        cat_cols = ["Model_ID", "Trans_ID"]
        X_cat = df[cat_cols].values

        # 定义数值列顺序，确保训练和预测一致
        num_cols = ["Year", "Age", "Kilometer", "Engine", "Max Power", "Seats"]
        X_num = df[num_cols].values

        # =======================================================
        # 准备数据集：这里仅仅使用没有溢价的那部分基础车型算出正常的价钱
        # =======================================================
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # 截断来自10%最离谱豪车的数据，第三次缓解长尾影响（经过实验10%是比较合理的）
        train_core_values = df.iloc[train_idx]["y_core_raw"]
        core_threshold = train_core_values.quantile(0.95)  # 稍微放宽一点

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
                df.iloc[idx_list]["composite_bias"].values,  # 只传综合 Bias
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
        )
        test_ds = CarDataset(
            X_num_test,
            test_data[1],
            y_core_test,
            test_data[3],
            test_data[4],
        )

        cat_dims = [len(le_model.classes_), len(le_trans.classes_)]
        emb_dims = [min(50, (d + 1) // 2) for d in cat_dims]  # 计算有多少个嵌入维度

        # 收集预处理所需的所有对象
        preprocessor_pack = {
            "impute_values": impute_values,
            "global_mean_log": global_mean_log,
            "brand_map": brand_map,
            "bm_bias_map": bm_bias_map,  # 车型 Bias Map
            "valid_models": set(valid_models),  # 有效车型集合
            "fuel_map": fuel_map,
            "le_brand": le_brand,
            "le_fuel": le_fuel,
            "le_model": le_model,
            "le_trans": le_trans,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "cat_dims": cat_dims,
            "emb_dims": emb_dims,
            "n_num": X_num.shape[1],
            "num_cols": num_cols,  # 记录数值列顺序
        }

        return (
            data.DataLoader(train_ds, batch_size, shuffle=True),
            data.DataLoader(test_ds, batch_size, shuffle=False),
            preprocessor_pack,
        )

    def train_model(self):
        """
        核心训练逻辑，若模型不存在则重新训练
        """
        seed_everything(42)

        # 1. 获取数据与预处理器
        (train_dl, test_dl, prep_pack) = self.get_data_strategy(self.csv_path, BATCH_SIZE)

        # 2. 保存预处理参数到内存和磁盘
        self.preprocessor_data = prep_pack
        with open(self.preprocessor_path, "wb") as f:
            pickle.dump(prep_pack, f)
        print(f"Preprocessor saved to {self.preprocessor_path}")

        # 3. 初始化模型
        n_num = prep_pack["n_num"]
        cat_dims = prep_pack["cat_dims"]
        emb_dims = prep_pack["emb_dims"]

        model = TabularResNet(n_num, cat_dims, emb_dims, hidden_dim=256, num_blocks=3).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        criterion = nn.MSELoss()

        print("Training Backbone...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for x_num, x_cat, y_core, _, _ in train_dl:
                x_num, x_cat, y_core = x_num.to(self.device), x_cat.to(self.device), y_core.to(self.device)
                optimizer.zero_grad()
                pred = model(x_num, x_cat).squeeze()
                loss = criterion(pred, y_core)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Train MSE (Core Logic): {total_loss / len(train_dl):.4f}")

        # 4. 保存模型
        torch.save(model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
        self.model = model

        # 最终评估
        self.evaluate(test_dl)

    def evaluate(self, test_dl):
        print("Evaluation on FULL Test Set (Basic + Luxury)")
        scaler_y = self.preprocessor_data["scaler_y"]
        self.model.eval()
        results = []
        header_printed = 1
        with torch.no_grad():
            for x_num, x_cat, _y_core, comp_bias, raw_price in test_dl:
                x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
                # 1. ResNet 预测"普通车况价值" (Normalized Log scale)
                pred_core_norm = self.model(x_num, x_cat).cpu().numpy().flatten()
                # 2. 反归一化
                pred_core_log = scaler_y.inverse_transform(pred_core_norm.reshape(-1, 1)).flatten()

                # 预测 log = Core + Composite Bias
                final_log_pred = pred_core_log + comp_bias.numpy() + self.preprocessor_data["global_mean_log"]
                final_pred_price = np.expm1(final_log_pred)
                actual_price = raw_price.numpy()

                batch_res = np.vstack([final_pred_price, raw_price.numpy()]).T
                results.append(batch_res)
                # 打印样本分析
                if header_printed <= 10:
                    print("-" * 100)
                    print(f"{'Type':<10}|{'Comp. Coeff':<15}|{'Base Price':<15}|{'Final Pred':<15}|{'Actual':<15}")
                    print("-" * 100)
                    for j in range(min(10, len(final_pred_price))):
                        # 将 Log Bias 转为倍数显示 (e.g., 0.69 -> x2.0)
                        # 这里的 comp_bias 已经是 Brand + Model + Fuel 的总和
                        total_bias_val = comp_bias[j].item()
                        total_coeff = np.exp(total_bias_val)
                        base_p = np.expm1(pred_core_log[j])

                        # 简单判断这个样本是否算豪车
                        car_type = "Luxury" if total_bias_val > 0.3 else "Basic"

                        print(
                            f"{car_type:<10}|x{total_coeff:.2f}|{base_p:<15,.0f}|{final_pred_price[j]:<15,.0f}|{actual_price[j]:<15,.0f}"
                        )
                    print("-" * 100)
                    header_printed += 1

        # 汇总计算
        all_res = np.vstack(results)
        mae = np.mean(np.abs(all_res[:, 0] - all_res[:, 1]))
        rmse = np.sqrt(np.mean((all_res[:, 0] - all_res[:, 1]) ** 2))

        print("\nFinal Results on Model:")
        print(f"MAE : {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")

    def load_model(self):
        """
        加载已有模型和预处理器
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            return False

        print("Loading existing model and preprocessors...")
        # 1. 加载预处理器
        with open(self.preprocessor_path, "rb") as f:
            self.preprocessor_data = pickle.load(f)

        # 2. 重新构建模型结构
        n_num = self.preprocessor_data["n_num"]
        cat_dims = self.preprocessor_data["cat_dims"]
        emb_dims = self.preprocessor_data["emb_dims"]

        self.model = TabularResNet(n_num, cat_dims, emb_dims, hidden_dim=256, num_blocks=3).to(self.device)

        # 3. 加载权重
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        return True

    def initialize(self):
        """
        启动入口：检查并加载，或者重新训练
        """
        if not self.load_model():
            print("No saved model found. Starting training...")
            self.train_model()
        else:
            print("Model loaded successfully.")

    def predict_price(self, data_dict):
        """
        单条数据预测接口
        data_dict: 包含车辆信息的字典
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # 1. 构造数据
        # 转小写，防止匹配失败
        brand = str(data_dict.get("brand")).strip().lower()
        model_str = str(data_dict.get("model")).strip().lower()
        fuel = str(data_dict.get("fuel_type")).strip().lower()
        trans = str(data_dict.get("transmission")).strip().lower()
        row_data = {
            "Brand": brand,
            "Model": model_str,
            "Year": data_dict.get("year"),
            "Age": data_dict.get("age"),
            "Kilometer": data_dict.get("milage"),  # 对应 Dataset 的 Kilometer
            "Fuel Type": fuel,
            "Engine": data_dict.get("engine"),
            "Max Power": data_dict.get("max_power"),
            "Transmission": trans,
            "Seats": data_dict.get("seats"),
        }

        # 构造 DataFrame
        df = pd.DataFrame([row_data])

        # 2. 提取预处理器
        prep = self.preprocessor_data

        # 3. 数值特征预处理 (缺失值填充)
        df["Max Power"] = pd.to_numeric(df["Max Power"], errors="coerce")
        df["Engine"] = df["Engine"].fillna(prep["impute_values"]["Engine"])
        df["Max Power"] = df["Max Power"].fillna(prep["impute_values"]["Max Power"])

        # 4. 提取数值特征并归一化
        X_num = df[prep["num_cols"]].values
        X_num = prep["scaler_x"].transform(X_num)

        # 5. 类别特征编码与偏置查找
        # 注意：生产环境中如果遇到训练集没见过的 Brand/Model，LabelEncoder 会报错
        # 这里使用 try-except (ValueError/KeyError) 捕获未知标签的情况，并做降级处理

        # 1. Fuel Bias
        try:
            fuel_id = int(prep["le_fuel"].transform([fuel])[0])
            fuel_bias = prep["fuel_map"].get(fuel_id, 0.0)
        except:
            fuel_bias = 0.0

        # 2. Base Bias (Model 优先, Brand 兜底)
        bm_key = f"{brand}|{model_str}"
        base_bias = 0.0
        used_strategy = "Unknown"

        if bm_key in prep["bm_bias_map"] and bm_key in prep["valid_models"]:
            # 命中车型 Bias
            base_bias = prep["bm_bias_map"][bm_key]
            used_strategy = f"Model Specific ({bm_key})"
        else:
            # 回退到品牌 Bias
            try:
                brand_id = int(prep["le_brand"].transform([brand])[0])
                base_bias = prep["brand_map"].get(brand_id, 0.0)
                used_strategy = f"Brand Fallback ({brand})"
            except:
                used_strategy = "Global Mean (No Match)"
                base_bias = 0.0

        final_composite_bias = base_bias + fuel_bias

        # 3. Embedding ID 查找
        try:
            model_id = int(prep["le_model"].transform([model_str])[0])
        except (ValueError, KeyError, IndexError):
            model_id = 0

        try:
            trans_id = int(prep["le_trans"].transform([trans])[0])
        except (ValueError, KeyError, IndexError):
            trans_id = 0

        X_cat = np.array([[model_id, trans_id]])

        # 6. 转 Tensor 并预测
        x_num_tensor = torch.tensor(X_num, dtype=torch.float32).to(self.device)
        x_cat_tensor = torch.tensor(X_cat, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # 预测 Core
            pred_core_norm = self.model(x_num_tensor, x_cat_tensor).cpu().numpy().flatten()

            # 反归一化
            pred_core_log = prep["scaler_y"].inverse_transform(pred_core_norm.reshape(-1, 1)).flatten()[0]

            # 还原
            final_log_pred = pred_core_log + final_composite_bias + prep["global_mean_log"]
            final_price = np.expm1(final_log_pred)

            print("-" * 40)
            print("Debug Info (Hierarchical):")
            print(f"Input: {brand} {model_str}")
            print(f"Strategy: {used_strategy}")
            print(f"Base Bias: {base_bias:.4f} | Fuel Bias: {fuel_bias:.4f}")
            print(f"Core Residual: {pred_core_log:.4f}")
            print(f"Global Mean: {prep['global_mean_log']:.4f}")
            print("-" * 40)
        return float(final_price)


# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    predictor = ResnetCarPricePredictor()

    predictor.initialize()

    (train_dl, test_dl, prep_pack) = predictor.get_data_strategy(CSV_PATH, BATCH_SIZE)
    predictor.evaluate(test_dl)

    payload = {
        "brand": "Toyota",
        "model": "Corolla",
        "year": 2019,
        "age": 4,
        "milage": 45000,
        "fuel_type": "Petrol",
        "engine": 1798,
        "max_power": 138,
        "transmission": "Automatic",
        "seats": 5,
    }

    try:
        price = predictor.predict_price(payload)
        print("\n" + "=" * 40)
        print(f"Predicted Price for Sample: {price:,.2f}")
        print("=" * 40)
    except Exception as e:
        print(f"Prediction failed: {e}")
