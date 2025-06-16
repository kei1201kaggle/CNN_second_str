import torch
import torch.nn as nn
import torch.optim as optim

# 1. ダミーデータ（2次元の特徴量×100件）を生成
torch.manual_seed(0)
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)  # ラベル: 0 or 1

# 2. モデル定義
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 入力2次元 → 出力1次元（確率）

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # 確率に変換

model = LogisticRegression()

# 3. 損失関数と最適化手法
criterion = nn.BCELoss()  # バイナリクロスエントロピー
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 学習ループ
for epoch in range(100):
    y_pred = model(X)                  # 予測
    loss = criterion(y_pred, y)        # 損失
    loss.backward()                    # 勾配計算
    optimizer.step()                   # パラメータ更新
    optimizer.zero_grad()              # 勾配リセット

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. 精度確認（予測値が0.5以上なら1とみなす）
with torch.no_grad():
    y_pred = model(X)
    predicted = (y_pred > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print(f"Accuracy: {accuracy.item():.2%}")
