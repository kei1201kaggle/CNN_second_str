import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


label_map = {'H': 0, 'E': 1, '-': 2}

# データセットクラス
class SequenceDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = label_map[parts[1]]
                features = list(map(float, parts[2:]))
                self.data.append((features, label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, label = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#理論部分
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)

#データ読み込み
train_dataset = SequenceDataset("nmers_n7_scores_validation.txt")
test_dataset = SequenceDataset("nmers_n7_scores.txt")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)


model = LogisticRegressionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#学習
for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#テスト精度の確認
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {correct / total:.2%}")
