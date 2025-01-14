# Spuriosity Didn’t Kill the Classifier: Using Invariant Predictions to Harness Spurious Features

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 辅助函数
def rademacher(beta, size):
    """生成 Rademacher 随机变量，+1 的概率为 `beta`，-1 的概率为 `1-beta`。"""
    return np.where(np.random.rand(size) < beta, 1, -1)

def generate_dataset(beta_e, num_samples):
    """根据 `beta_e` 和样本数量生成数据集。"""
    Y = rademacher(0.5, num_samples)  # 生成标签 Y ~ Rad(0.5)
    X_S = Y * rademacher(0.75, num_samples)  # 稳定特征，与 Y 高度相关
    X_U = Y * rademacher(beta_e, num_samples)  # 不稳定特征，与 Y 的相关性由 beta_e 决定
    X = np.stack((X_S, X_U), axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor((Y + 1) / 2, dtype=torch.float32).unsqueeze(1)

# 参数设置
num_samples = 1000  # 每个域的样本数量
train_betas = [0.95, 0.7]  # 训练域的相关性
val_beta = 0.6  # 验证域的相关性
test_beta = 0.1  # 测试域的相关性
batch_size = 32  # DataLoader 的批量大小

# 生成数据集
train_domains = [generate_dataset(beta, num_samples) for beta in train_betas]
val_domain = generate_dataset(val_beta, num_samples)
test_domain = generate_dataset(test_beta, num_samples)

# 准备 DataLoader
X_train = torch.cat([X for X, _ in train_domains], dim=0)
Y_train = torch.cat([Y for _, Y in train_domains], dim=0)
train_datasets = [TensorDataset(X, Y) for X, Y in train_domains]
val_dataset = TensorDataset(*val_domain)
test_dataset = TensorDataset(*test_domain)
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_datasets]
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # 使用 Sigmoid 激活函数进行二分类

# 初始化稳定模型
model = SimpleNN(input_size=1, hidden_size=8, output_size=1)  # 输入为 X_S
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练稳定模型
num_epochs = 50
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch[:, 0:1])  # 使用 X_S 训练
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch[:, 0:1])
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'第 {epoch+1}/{num_epochs} 轮，验证损失: {val_loss:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

# 加载最佳模型
model.load_state_dict(best_model_state)

# 初始化不稳定模型
modelU = SimpleNN(input_size=1, hidden_size=8, output_size=1)  # 输入为 X_U
criterionU = nn.BCELoss()
optimizerU = optim.Adam(modelU.parameters(), lr=1e-4)

# 使用伪标签训练不稳定模型
for epoch in range(num_epochs):
    modelU.train()
    model.eval()
    for X_batch, _ in test_loader:
        optimizerU.zero_grad()
        Y_batch = model(X_batch[:, 0:1])  # 从 X_S 生成伪标签
        Y_batch = (Y_batch > 0.5).float()
        outputs = modelU(X_batch[:, 1:2])  # 使用 X_U 训练
        lossU = criterionU(outputs, Y_batch)
        lossU.backward()
        optimizerU.step()

# 测试模型
model.eval()
modelU.eval()
def evaluate(which_loader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in which_loader:
            Y_stable = model(X_batch[:, 0:1])
            Y_stable_logit = torch.logit(Y_stable, eps=1e-6)
            Y_stable = (Y_stable > 0.5).float()
            Y_unstable = modelU(X_batch[:, 1:2])
            Y_unstable_logit = torch.logit(Y_unstable, eps=1e-6)
            combined_prediction = torch.sigmoid(Y_stable_logit + Y_unstable_logit)
            predicted = (combined_prediction > 0.5).float()
            correct += (predicted == Y_batch).sum().item()
            total += Y_batch.size(0)
    accuracy = correct / total
    print(f'{name} 准确率: {accuracy:.4f}')

# 在训练集和测试集上评估模型
evaluate(train_loader, '训练集')
evaluate(test_loader, '测试集')
