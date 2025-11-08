import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


#生成一个假数据
def generate_data(seq_len=50, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    y = np.sin(x) + 0.1 * np.random.randn(num_samples)
    data = []
    for i in range(len(y) - seq_len):
        data.append((y[i:i + seq_len], y[i + seq_len]))
    X = np.array([d[0] for d in data])
    Y = np.array([d[1] for d in data])
    X = X[:, :, np.newaxis]  # (samples, seq_len, 1)
    Y = Y[:, np.newaxis]  # (samples, 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


seq_len = 20
X, Y = generate_data(seq_len=seq_len, num_samples=500)

# 划分训练，验证
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]


class BaseRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: (B, T, hidden)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


class BaseLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class BaseGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = criterion(model(X_val), Y_val).item()
            print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")


#训练
print("=== RNN ===")
rnn_model = BaseRNN()
train_model(rnn_model, X_train, Y_train, X_val, Y_val)

print("=== LSTM ===")
lstm_model = BaseLSTM()
train_model(lstm_model, X_train, Y_train, X_val, Y_val)

print("=== GRU ===")
gru_model = BaseGRU()
train_model(gru_model, X_train, Y_train, X_val, Y_val)


# 简单预测

sample_input = X_val[0:1]  # (1, seq_len, 1)
pred_rnn = rnn_model(sample_input).item()
pred_lstm = lstm_model(sample_input).item()
pred_gru = gru_model(sample_input).item()
print(f"RNN pred: {pred_rnn:.4f}, LSTM pred: {pred_lstm:.4f}, GRU pred: {pred_gru:.4f}")
print(f"Ground truth: {Y_val[0].item():.4f}")
