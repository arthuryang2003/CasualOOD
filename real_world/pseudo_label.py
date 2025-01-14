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
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    Y_tensor = (Y_tensor + 1) / 2
    return X_tensor, Y_tensor

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

# Get an iterator from the DataLoader
data_iter = iter(train_loader)

# Get the first batch (or sample) from the iterator
X_sample, Y_sample = next(data_iter)

# Print the sample
print("X_sample:\n", X_sample.shape)
print("Y_sample:\n", Y_sample.shape)

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
        outputs = model(X_batch[:, 0:1])  
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

# Training loop
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    modelU.train()
    model.eval()
    for X_batch, _ in test_loader:
        optimizerU.zero_grad()  # Clear gradients
        Y_batch = model(X_batch[:,0:1])  # Forward pass
        Y_batch = (Y_batch > 0.5).float()
        outputs = modelU(X_batch[:,1:2])  # Forward pass
        lossU = criterionU(outputs, Y_batch)  # Compute loss
        lossU.backward()  # Backward pass
        optimizerU.step()  # Update weights
    
    # Validation
    modelU.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            Y_batch = model(X_batch[:,0:1])  # Forward pass
            Y_batch = (Y_batch > 0.5).float()
            outputs = modelU(X_batch[:,1:2])  # Forward pass
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_stateU = modelU.state_dict()
    best_model_stateU = modelU.state_dict()

# Load the best model for testing
modelU.load_state_dict(best_model_stateU)

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

def f_multiclass(which_loader, name, num_classes):
    PY = torch.zeros(num_classes)  # Prior for each class
    e_matrix = torch.zeros(num_classes, num_classes)  # Confusion matrix

    with torch.no_grad():
        # First pass: Compute confusion matrix and priors
        for X_batch, Y_batch in which_loader:
            PY += Y_batch.sum(dim=0)  # Summing the true labels
            Y_pred = F.softmax(model(X_batch[:, 0:1]), dim=1)  # Stable model predictions
            Y_pred_hard = (Y_pred == Y_pred.max(dim=1, keepdim=True)[0]).float()

            outputs = F.softmax(modelU(X_batch[:, 1:2]), dim=1)  # Unstable model predictions
            for k in range(num_classes):
                for k_prime in range(num_classes):
                    e_matrix[k, k_prime] += (
                        (Y_batch[:, k_prime] * Y_pred_hard[:, k]).sum().item()
                    )

    # Normalize confusion matrix and prior
    e_matrix = e_matrix / e_matrix.sum(dim=0, keepdim=True)
    PY = PY / PY.sum()

    correct = 0
    total = 0
    OOD = 0
    with torch.no_grad():
        # Second pass: Use adjusted predictions
        for X_batch, Y_batch in which_loader:
            Y_stable = F.softmax(model(X_batch[:, 0:1]), dim=1)  # Stable model
            Xlogit = torch.log(Y_stable + 1e-6)  # Avoid log(0)

            Y_unstable = F.softmax(modelU(X_batch[:, 1:2]), dim=1)  # Unstable model
            Y_unstable_corrected = torch.matmul(Y_unstable, torch.inverse(e_matrix))

            Ulogit = torch.log(Y_unstable_corrected + 1e-6)

            prior_logit = torch.log(PY / (1 - PY) + 1e-6)  # Class priors

            # Combine stable and unstable logits
            combined_logit = Xlogit + Ulogit - prior_logit
            predict = torch.softmax(combined_logit, dim=1)

            # Convert predictions to hard labels
            predicted = (predict == predict.max(dim=1, keepdim=True)[0]).float()
            OOD += predicted.sum(dim=1).mean().item()
            correct += (predicted == Y_batch).all(dim=1).sum().item()
            total += Y_batch.size(0)

    OOD = OOD / total
    accuracy = correct / total
    print(name + f' Accuracy: {accuracy:.4f}')

f_multiclass(train_loader, 'train', num_classes=5)
f_multiclass(test_loader, 'test', num_classes=5)


def finetune_unstable_with_pseudo_labels(stable_model, unstable_model, train_target_iter, optimizer, lr_scheduler, epoch, args, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.1f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.1f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs],
        prefix="Unstable Model Fine-tuning Epoch: [{}]".format(epoch)
    )

    # 切换到训练模式
    unstable_model.train()
    end = time.time()

    # 获取目标域数据并选择10%用于微调
    img_t_all, labels_t_all, d_t_all, _ = next(train_target_iter)
    img_t_all = img_t_all.to(device)
    labels_t_all = labels_t_all.to(device)
    d_t_all = d_t_all.to(device)

    # 随机选取目标域数据的10%作为训练数据
    target_train_size = int(0.1 * len(img_t_all))  # 10%的数据用于训练
    target_train_idx = torch.randperm(len(img_t_all))[:target_train_size]
    target_train_data = img_t_all[target_train_idx]
    target_train_labels = labels_t_all[target_train_idx]

    # 剩下的90%数据用于验证
    target_val_data = img_t_all[target_train_size:]
    target_val_labels = labels_t_all[target_train_size:]

    # 使用稳定模型生成伪标签
    stable_model.eval()
    with torch.no_grad():
        # 获取稳定模型的输出并生成伪标签
        stable_logits = stable_model.classifier(extract_features(stable_model, target_train_data, d_t_all[target_train_idx])[0])
        pseudo_labels = torch.argmax(stable_logits, dim=1)

    # 微调不稳定模型
    unstable_model.train()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        unstable_model.train()

        # 计时并加载数据
        data_time.update(time.time() - end)
        img_t, labels_t, d_t, _ = next(train_target_iter)
        img_t = img_t.to(device)
        labels_t = labels_t.to(device)

        # 只使用从目标域选出的训练数据（10%）
        target_train_data = img_t[:target_train_size].to(device)
        pseudo_labels = pseudo_labels.to(device)  # 使用伪标签

        # 提取不稳定特征（style）
        _, style = extract_features(unstable_model, target_train_data, d_t[:target_train_size])  # 提取可变特征

        # 使用不稳定特征进行分类
        logits = unstable_model.classifier(style)  # 分类

        # 计算交叉熵损失与伪标签之间的损失
        loss = F.cross_entropy(logits, pseudo_labels)

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        # 更新统计指标
        acc = accuracy(logits, pseudo_labels)[0]
        cls_losses.update(loss.item(), target_train_data.size(0))
        cls_accs.update(acc.item(), target_train_data.size(0))

        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        # 每隔一定频率打印进度信息
        if i % args.print_freq == 0:
            # 切换到评估模式
            unstable_model.eval()

            with torch.no_grad():
                # 使用不稳定特征模型进行验证
                val_logits = unstable_model(target_val_data)
                val_loss = F.cross_entropy(val_logits, target_val_labels)
                val_acc = accuracy(val_logits, target_val_labels)[0]
                progress.display(i)

                # 记录验证的损失和准确率
                wandb.log({
                    "Unstable Model Fine-tuning Loss": cls_losses.avg,
                    "Unstable Model Fine-tuning Accuracy": cls_accs.avg,
                    "Validation Loss": val_loss.item(),
                    "Validation Accuracy": val_acc.item(),
                })

    print(f"Epoch [{epoch}] Fine-tuning Loss = {cls_losses.avg:.4f}, Accuracy = {cls_accs.avg * 100:.2f}%")


def combined_inference(stable_model, unstable_model, which_loader, num_classes):
    # 初始化先验分布和混淆矩阵
    PY = torch.zeros(num_classes).to(device)  # 类别先验分布
    e_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵

    # 第一遍：计算混淆矩阵和先验分布
    stable_model.eval()
    unstable_model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in which_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # 稳定模型预测
            Y_pred_stable = F.softmax(stable_model(X_batch[:, 0:1]), dim=1)
            Y_pred_stable_hard = (Y_pred_stable == Y_pred_stable.max(dim=1, keepdim=True)[0]).float()

            # 不稳定模型预测
            Y_pred_unstable = F.softmax(unstable_model(X_batch[:, 1:2]), dim=1)

            # 更新先验分布
            PY += Y_batch.sum(dim=0)

            # 更新混淆矩阵
            for k in range(num_classes):
                for k_prime in range(num_classes):
                    e_matrix[k, k_prime] += (
                        (Y_batch[:, k_prime] * Y_pred_stable_hard[:, k]).sum().item()
                    )

    # 归一化混淆矩阵和先验分布
    e_matrix = e_matrix / e_matrix.sum(dim=0, keepdim=True)
    PY = PY / PY.sum()

    # 第二遍：使用调整后的不稳定模型预测
    correct = 0
    total = 0
    OOD = 0
    with torch.no_grad():
        for X_batch, Y_batch in which_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # 稳定模型预测
            Y_stable = F.softmax(stable_model(X_batch[:, 0:1]), dim=1)
            Xlogit = torch.log(Y_stable + 1e-6)

            # 不稳定模型预测并调整
            Y_unstable = F.softmax(unstable_model(X_batch[:, 1:2]), dim=1)
            Y_unstable_corrected = torch.matmul(Y_unstable, torch.inverse(e_matrix))
            Ulogit = torch.log(Y_unstable_corrected + 1e-6)

            # 类别先验
            prior_logit = torch.log(PY / (1 - PY) + 1e-6)

            # 联合预测
            combined_logit = Xlogit + Ulogit - prior_logit
            predict = torch.softmax(combined_logit, dim=1)

            # 转为硬标签
            predicted = (predict == predict.max(dim=1, keepdim=True)[0]).float()
            OOD += predicted.sum(dim=1).mean().item()
            correct += (predicted == Y_batch).all(dim=1).sum().item()
            total += Y_batch.size(0)

    # 输出准确率和 OOD
    OOD = OOD / total
    accuracy = correct / total
    return accuracy, OOD


# 定义伪标签生成函数
def generate_pseudo_labels(test_loader, stable_model, device):
    pseudo_labels = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_S = X_batch[:, :args.z_dim - args.style_dim].to(device)
            Y_hat = stable_model(X_S)  # 使用稳定特征生成伪标签
            pseudo_labels.append((X_batch, torch.argmax(Y_hat, dim=1)))  # 多类情况下使用argmax生成伪标签
    return pseudo_labels

# 定义伪标签微调函数
def pseudo_label_finetune(test_loader, unstable_model, stable_model, optimizer, criterion, device):
    unstable_model.train()
    pseudo_labels = generate_pseudo_labels(test_loader, stable_model, device)
    for X_batch, Y_hat in pseudo_labels:
        X_U = X_batch[:, args.z_dim - args.style_dim:].to(device)  # 使用可变特征 X_U
        Y_hat = Y_hat.to(device)
        optimizer.zero_grad()
        outputs = unstable_model(X_U)
        loss = criterion(outputs, Y_hat)  # 使用伪标签进行损失计算
        loss.backward()
        optimizer.step()