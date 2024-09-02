import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成单调递增数据集
def create_monotonic_data(seq_length, size):
    data = []
    start = np.random.uniform(0, 1)
    for _ in range(size):
        seq = np.linspace(start, start + seq_length - 1, seq_length)
        start += seq_length
        data.append(seq)
    return np.array(data, dtype=np.float32)

# RNN模型定义
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集
seq_length = 20
data_size = 100
data = create_monotonic_data(seq_length, data_size)
data = torch.from_numpy(data).unsqueeze(-1).to(device)

# 模型、损失函数和优化器
input_size = 1
hidden_size = 20
num_layers = 5
output_size = 1

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 10000
losses = []

for epoch in range(num_epochs):
    for seq in data:
        seq = seq.unsqueeze(0)  # 增加批次维度
        target = seq[:, -1, :]  # 目标是序列的最后一个元素
        output = model(seq)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
