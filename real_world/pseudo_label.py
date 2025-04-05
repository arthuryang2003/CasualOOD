import torch
import torch.nn.functional as F
from itertools import chain


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_inference(model, test_loader,num_classes):
    # 初始化先验分布和混淆矩阵
    PY_raw = torch.zeros(num_classes).to(device)  # 未归一化的先验分布
    model.eval()
    e_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵

    # 计算混淆矩阵和先验分布
    model.eval()
    test_iter = chain(*test_loader)  # <-- 这里拼接
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            data = batch[0].to(device)

            # 通过稳定模型提取特征并预测
            z_u,z_s,u_logits,s_logits,tilde_s_logits,combined_logits=model.encode(data)

            # softmax 后的概率分布
            stable_pred = F.softmax(u_logits, dim=1)

            # 转为 one-hot 编码
            stable_pred_hard = torch.argmax(stable_pred, dim=1)
            stable_pred_onehot = F.one_hot(stable_pred_hard, num_classes=num_classes).float()

            # 累加未归一化先验
            PY_raw += stable_pred_onehot.sum(dim=0)


    # 计算归一化的 P_Y
    PY = PY_raw / PY_raw.sum()

    # 计算混淆矩阵 e = P_Y_raw^T * Normalize(P_Y)
    e_matrix = PY_raw.unsqueeze(1) @ F.normalize(PY.unsqueeze(0), p=1, dim=1)

    test_iter = chain(*test_loader)  # <-- 这里拼接
    # 第二遍：使用调整后的不稳定模型预测
    correct = 0
    total = 0
    OOD = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            # 解包数据，只取前两个（data 和 labels）
            data =batch[0].to(device)
            labels = batch[1].to(device)
            # decoupler model inference to decouple content and style
            z_u,z_s,u_logits,s_logits,tilde_s_logits,combined_logits=model.encode(data)

            # Stable model prediction using content (z_content)

            stable_pred_softmax = F.softmax(u_logits, dim=1)
            stable_pred_hard = torch.argmax(stable_pred_softmax, dim=1)

            # Unstable model prediction using style (z_style)

            unstable_pred_softmax = F.softmax(tilde_s_logits, dim=1)

            # Apply least squares correction to the unstable model's output
            unstable_pred_corrected = least_squares_correction(unstable_pred_softmax, e_matrix)

            # Logits for combining stable and unstable model predictions
            stable_logit = torch.log(stable_pred_softmax + 1e-6)
            unstable_logit = torch.log(unstable_pred_corrected + 1e-6)

            # Combined logits
            combined_logit = stable_logit + unstable_logit - torch.log(PY + 1e-6)

            # Convert combined logits to probabilities
            predict = F.softmax(combined_logit, dim=1)


            # 转换为硬标签（单标签分类选择最大概率）
            predicted = torch.argmax(predict, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 输出准确率
    accuracy = correct / total*100.0
    return accuracy


# 最小二乘优化
def least_squares_correction(Y_unstable, e_matrix):
    # 初始校正值（随机或均匀分布）
    p = torch.ones_like(Y_unstable) / Y_unstable.size(1)  # 初始概率分布

    # 迭代优化
    for i in range(1000):
        gradient = torch.matmul(e_matrix, p.T) - Y_unstable.T
        p = p - 0.01 * gradient.T  # 学习率 0.01
        p = F.softmax(p, dim=1)  # 确保 p 满足概率分布约束

    return p
