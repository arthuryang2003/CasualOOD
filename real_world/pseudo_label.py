import torch
from extract_features import extract_features
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_inference(model, test_loader,num_classes):
    # 初始化先验分布和混淆矩阵
    PY = torch.zeros(num_classes).to(device)  # 类别先验分布
    e_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵

    # 计算混淆矩阵和先验分布
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 解包数据，只取前两个（data 和 labels）
            data, labels,domains = batch[:3]
            data = data.to(device)
            labels = labels.to(device)
            domains=domains.to(device)
            one_hot_labels = F.one_hot(labels, num_classes)

            # decoupler model inference to decouple content and style
            z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(data)



            stable_pred_softmax = F.softmax(u_logits, dim=1)  # Softmax for multi-class classification
            stable_pred_hard = torch.argmax(stable_pred_softmax, dim=1)

            # 更新先验分布
            PY += one_hot_labels.sum(dim=0)

            # 计算混淆矩阵
            for i in range(len(labels)):  # 对每一个样本
                true_label = labels[i].item()  # 真实标签
                predicted_label = stable_pred_hard[i].item()  # 预测标签
                e_matrix[true_label, predicted_label] += 1


    # 归一化混淆矩阵和先验分布

    e_matrix = e_matrix / e_matrix.sum(dim=1, keepdim=True)
    # # 打印混淆矩阵
    # print(f"Iteration {batch_idx + 1}, Confusion Matrix:\n{e_matrix}")

    PY = PY / PY.sum()

    # 第二遍：使用调整后的不稳定模型预测
    correct = 0
    total = 0
    OOD = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 解包数据，只取前两个（data 和 labels）
            data, labels,domains = batch[:3]
            data = data.to(device)
            labels = labels.to(device)
            domains=domains.to(device)

            # decoupler model inference to decouple content and style
            z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(data)

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
