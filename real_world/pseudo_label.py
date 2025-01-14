import torch
from extract_features import extract_features
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_inference(stable_model, unstable_model, test_loader,num_classes):
    # 初始化先验分布和混淆矩阵
    PY = torch.zeros(num_classes).to(device)  # 类别先验分布
    e_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵

    # 第一遍：计算混淆矩阵和先验分布
    stable_model.eval()
    unstable_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 解包数据，只取前两个（data 和 labels）
            data, labels = batch[:2]
            data = data.to(device)
            labels = labels.to(device)
            if labels.dim() == 1:
                labels = F.one_hot(labels, num_classes=num_classes).float()

            # compute output
            u = torch.ones([len(data)]).long().to(device)
            # 提取稳定特征和不稳定特征
            stable_features, _ = extract_features(stable_model, data, u)  # 提取稳定特征
            _, unstable_features = extract_features(unstable_model, data, u)  # 提取不稳定特征

            # 稳定模型预测（使用 sigmoid）
            Y_pred_stable = torch.sigmoid(stable_model.stable_classifier(stable_features))
            Y_pred_stable_hard = (Y_pred_stable > 0.5).float()

            # 更新先验分布
            PY += labels.sum(dim=0)

            e_matrix += torch.matmul(labels.T, Y_pred_stable_hard)


    # 归一化混淆矩阵和先验分布
    e_matrix = e_matrix / e_matrix.sum(dim=0, keepdim=True)
    PY = PY / PY.sum()

    # 第二遍：使用调整后的不稳定模型预测
    correct = 0
    total = 0
    OOD = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 解包数据，只取前两个（data 和 labels）
            data, labels = batch[:2]
            data = data.to(device)
            labels = labels.to(device)
            # 转换为 one-hot 编码
            if labels.dim() == 1:
                labels = F.one_hot(labels, num_classes=num_classes).float()

            u = torch.ones([len(data)]).long().to(device)
            # 提取稳定特征和不稳定特征
            stable_features, _ = extract_features(stable_model, data, u)  # 提取稳定特征
            _, unstable_features = extract_features(unstable_model, data, u)  # 提取不稳定特征


            # 稳定模型预测（sigmoid 输出概率）
            Y_stable = torch.sigmoid(stable_model.stable_classifier(stable_features))
            Xlogit = torch.logit(Y_stable, eps=1e-6)

            # 不稳定模型预测并调整（sigmoid 输出 + 偏差校正）
            Y_unstable = torch.sigmoid(unstable_model.unstable_classifier(unstable_features))
            Y_unstable_corrected = torch.matmul(Y_unstable, torch.inverse(e_matrix))
            Y_unstable_corrected = torch.clamp(Y_unstable_corrected, min=0, max=1)
            Ulogit = torch.logit(Y_unstable_corrected, eps=1e-6)

            # 联合预测：组合稳定模型和不稳定模型的对数几率
            combined_logit = Xlogit + Ulogit - torch.log(PY + 1e-6)

            # 转换为概率分布
            predict = torch.sigmoid(combined_logit)


            # 转换为硬标签（单标签分类选择最大概率）
            predicted = torch.argmax(predict, dim=1)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            total += labels.size(0)

    # 输出准确率

    accuracy = correct / total
    return accuracy

