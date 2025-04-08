import torch
import torch.nn.functional as F
from itertools import chain
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def combined_inference(model, test_loader, num_classes):
    model.eval()
    test_iter = chain(*test_loader)

    if num_classes == 2:
        # ==== 二分类推理逻辑 ====
        PY = 0.
        n1 = 0
        n = 0
        e0 = 0.
        e1 = 0.

        with torch.no_grad():
            for data, labels in test_iter:
                data = data.to(device)
                labels = labels.to(device).float()

                z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(data)
                Y_stable = torch.sigmoid(u_logits).squeeze()
                Y_stable_hard = torch.argmax(Y_stable, dim=1)
                # Y_unstable = torch.sigmoid(tilde_s_logits).squeeze()
                # Y_unstable_hard = torch.argmax(Y_stable, dim=1)

                PY += Y_stable_hard.sum().item()
                n1 += Y_stable_hard.sum().item()
                n += Y_stable_hard.size(0)

                e0 += ((1 - Y_stable_hard) * (1 - Y_stable_hard)).sum().item()
                e1 += (Y_stable_hard * Y_stable_hard).sum().item()

        e0 = e0 / (n - n1 + 1e-6)
        e1 = e1 / (n1 + 1e-6)
        PY = PY / n

        # 第二遍预测
        correct = 0
        total = 0
        OOD = 0
        test_iter = chain(*test_loader)
        with torch.no_grad():
            for data, labels in test_iter:
                data = data.to(device)
                labels = labels.to(device).float()

                z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(data)
                Y_stable = torch.sigmoid(u_logits).squeeze()
                Y_unstable = torch.sigmoid(tilde_s_logits).squeeze()

                Xlogit = torch.logit(Y_stable, eps=1e-6)
                Y_unstable_corrected = (Y_unstable + e0 - 1) / (e1 + e0 - 1 + 1e-6)
                Y_unstable_corrected = torch.clamp(Y_unstable_corrected, min=0, max=1)
                Ulogit = torch.logit(Y_unstable_corrected, eps=1e-6)

                combined_logit = Xlogit + Ulogit - np.log(PY / (1 - PY + 1e-6))
                predict = torch.sigmoid(combined_logit)
                predicted = torch.argmax(predict, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                OOD += predicted.sum().item()

        acc = correct / total * 100.0
        print(f"[Binary Combined Inference] Accuracy: {acc:.2f}%, OOD rate: {OOD/total:.4f}")
        return acc

    else:
        # ==== 多分类推理逻辑 ====
        PY_raw = torch.zeros(num_classes).to(device)
        test_iter = chain(*test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iter):
                data = batch[0].to(device)
                z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(data)

                stable_pred = F.softmax(u_logits, dim=1)
                stable_pred_hard = torch.argmax(stable_pred, dim=1)
                stable_pred_onehot = F.one_hot(stable_pred_hard, num_classes=num_classes).float()
                PY_raw += stable_pred_onehot.sum(dim=0)

        PY = PY_raw / PY_raw.sum()
        e_matrix = PY_raw.unsqueeze(1) @ F.normalize(PY.unsqueeze(0), p=1, dim=1)

        correct = 0
        total = 0
        test_iter = chain(*test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_iter):
                data = batch[0].to(device)
                labels = batch[1].to(device)

                z_u, z_s, u_logits, s_logits, tilde_s_logits, _ = model.encode(data)
                stable_pred_softmax = F.softmax(u_logits, dim=1)
                unstable_pred_softmax = F.softmax(tilde_s_logits, dim=1)

                unstable_pred_corrected = least_squares_correction(unstable_pred_softmax, e_matrix)

                stable_logit = torch.log(stable_pred_softmax + 1e-6)
                unstable_logit = torch.log(unstable_pred_corrected + 1e-6)
                combined_logit = stable_logit + unstable_logit - torch.log(PY + 1e-6)
                predict = F.softmax(combined_logit, dim=1)

                predicted = torch.argmax(predict, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100.0
        print(f"[Multiclass Combined Inference] Accuracy: {accuracy:.2f}%")
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
