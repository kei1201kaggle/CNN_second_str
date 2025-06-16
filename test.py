import torch
import torch.nn as nn
import torch.optim as optim

# read data
X_data = []
y_data = []
 
label_map = {"E": 0, "H": 1, "-": 2}  

with open("nmers_n7_scores.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        label_char = parts[1]
        x1 = float(parts[2])
        x2 = float(parts[3])

        X_data.append([x1, x2])
        y_data.append(label_map[label_char])


X = torch.tensor(X_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32).reshape(-1).long()




#learn

class MultiClassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)

model = MultiClassClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


epochs = 10000
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % (epochs // 10) == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")