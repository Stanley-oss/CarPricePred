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

        # 补上缺失值 (并记录中位数用于推理)
        df["Brand"] = df["Brand"].astype(str).str.strip()
        df["Fuel Type"] = df["Fuel Type"].astype(str).str.strip()
        df["Model"] = df["Model"].astype(str).str.strip()
        df["Transmission"] = df["Transmission"].astype(str).str.strip()

        # 补全缺失值
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
        # Bias 计算
        # 这里是全局的偏置
        # =========================================================
        print("Computing Global Biases (Brand & Fuel)...")
        le_brand = LabelEncoder()
        df["Brand_ID"] = le_brand.fit_transform(df["Brand"])  # 把品牌转换成编号
        # Brand Bias = Brand Mean - Global Mean
        brand_map = (df.groupby("Brand_ID")["log_price"].mean() - global_mean_log).to_dict()  # 计算每个品牌相对于平均的偏置
        df["bias_brand"] = df["Brand_ID"].map(brand_map)  # 再把算好的偏置加到数据里

        le_fuel = LabelEncoder()  # 一样的操作针对燃油类型再来一遍
        df["Fuel_ID"] = le_fuel.fit_transform(df["Fuel Type"])

        # Fuel Bias = Fuel Mean - Global Mean
        fuel_map = (df.groupby("Fuel_ID")["log_price"].mean() - global_mean_log).to_dict()
        df["bias_fuel"] = df["Fuel_ID"].map(fuel_map)

        # =======================================================
        # 4.3 计算核心残差 (Core Residual)
        # y_core = 除去了品牌和燃油类型溢价后的纯粹车价，第二次缓解长尾的影响
        df["y_core_raw"] = df["log_price"] - df["bias_brand"] - df["bias_fuel"]

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
        test_ds = CarDataset(
            X_num_test,
            test_data[1],
            y_core_test,
            test_data[3],
            test_data[4],
            test_data[5],
        )

        cat_dims = [len(le_model.classes_), len(le_trans.classes_)]
        emb_dims = [min(50, (d + 1) // 2) for d in cat_dims]  # 计算有多少个嵌入维度

        # 收集预处理所需的所有对象
        preprocessor_pack = {
            "impute_values": impute_values,
            "global_mean_log": global_mean_log,
            "brand_map": brand_map,  # 现在是 Dict
            "fuel_map": fuel_map,  # 现在是 Dict
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
            for x_num, x_cat, y_core, _, _, _ in train_dl:
                x_num, x_cat, y_core = (
                    x_num.to(self.device),
                    x_cat.to(self.device),
                    y_core.to(self.device),
                )
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
            for x_num, x_cat, _y_core, bias_b, bias_f, raw_price in test_dl:
                x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
                # 1. ResNet 预测"普通车况价值" (Normalized Log scale)
                pred_core_norm = self.model(x_num, x_cat).cpu().numpy().flatten()
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
            print("No saved model found or files missing. Starting training...")
            self.train_model()
        else:
            print("Model loaded successfully.")

    def predict_price(self, data_dict):
        """
        单条数据预测接口
        data_dict: 包含车辆信息的字典
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call initialize() first.")

        # 1. 构造数据
        row_data = {
            "Brand": str(data_dict.get("brand")).strip(),  # 只去空格
            "Model": str(data_dict.get("model")).strip(),
            "Year": data_dict.get("year"),
            "Age": data_dict.get("age"),
            "Kilometer": data_dict.get("milage"),  # 对应 Dataset 的 Kilometer
            "Fuel Type": str(data_dict.get("fuel_type")).strip(),  # 对应 Dataset 的 Fuel Type
            "Engine": data_dict.get("engine"),
            "Max Power": data_dict.get("max_power"),
            "Transmission": str(data_dict.get("transmission")).strip(),
            "Seats": data_dict.get("seats"),
            # listing_date 暂时未在模型中使用，忽略
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

        # Brand Bias
        try:
            # transform 返回 numpy array
            brand_id_numpy = prep["le_brand"].transform(df["Brand"])[0]
            brand_id = int(brand_id_numpy)
            # 从保存的 Series 中查找 Bias, 找不到则用 0
            brand_bias = prep["brand_map"].get(brand_id, 0.0)
        except (ValueError, KeyError, IndexError) as e:
            print(f"[Warn] Brand mismatch for '{df['Brand'].iloc[0]}': {e}")
            # 遇到未知品牌或数据错误，降级处理
            brand_bias = 0.0

        # Fuel Bias
        try:
            fuel_id_numpy = prep["le_fuel"].transform(df["Fuel Type"])[0]
            fuel_id = int(fuel_id_numpy)
            fuel_bias = prep["fuel_map"].get(fuel_id, 0.0)
        except (ValueError, KeyError, IndexError):
            fuel_bias = 0.0

        # Model & Trans 编码
        # 简单处理：如果遇到未知 Model，随机指派一个(或者指派众数)，这里取 0
        try:
            model_id = int(prep["le_model"].transform(df["Model"])[0])
        except (ValueError, KeyError, IndexError):
            model_id = 0

        try:
            trans_id = int(prep["le_trans"].transform(df["Transmission"])[0])
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

            # [Debug Print]
            # 这会让你确信 bias 到底有没有取到
            print("-" * 30)
            print("Debug Info:")
            print(f"Brand: {row_data['Brand']} | Bias: {brand_bias}")
            print(f"Fuel : {row_data['Fuel Type']} | Bias: {fuel_bias}")
            print(f"Core Log: {pred_core_log:.4f}")
            print("-" * 30)

            # 加上 Bias
            final_log_pred = pred_core_log + brand_bias + fuel_bias

            # Exp 还原价格
            final_price = np.expm1(final_log_pred)

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
