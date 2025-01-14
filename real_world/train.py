
import random
import time
import warnings
import argparse
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy
import wandb
from common.utils import ForeverDataIterator
from extract_features import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_stable(train_source_iter: ForeverDataIterator,
                 train_target_iter: ForeverDataIterator,
                 model, optimizer: SGD, lr_scheduler: LambdaLR,
                 epoch: int, args: argparse.Namespace, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.1f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.1f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs],
        prefix="Stable Model Epoch: [{}]".format(epoch)
    )

    # 切换到训练模式
    model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()

        # 计时并加载数据
        data_time.update(time.time() - end)
        img_s, labels_s, d_s, _ = next(train_source_iter)
        img_t, labels_t, d_t, _ = next(train_target_iter)

        # 将图像和标签数据移至GPU
        img_s = img_s.to(device)
        img_t = img_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # 将源域和目标域的图像合并
        img_all = torch.cat([img_s, img_t], 0)
        d_all = torch.cat([d_s, d_t], 0).to(device)
        label_all = torch.cat([labels_s, labels_t], 0)

        # 初始化各个损失项的列表
        losses_cls = []
        losses_kl = []
        z_all = []# 保存所有的潜变量z
        y_t = None
        y_s = [] # 用于保存源域的logits
        labels_s = []# 用于保存源域的标签
        x_all = []# 用于保存源域和目标域的特征

        # 遍历每个域，针对每个源域进行训练
        for id in range(args.n_domains):  # 遍历每个源域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域

            if not is_target:
                continue

            # 获取当前域的样本数据
            index = d_all == id  # 获取当前域的样本
            label_dom = label_all[index]
            img_dom = img_all[index]
            d_dom = d_all[index]

            # 提取稳定特征（content）
            content, _ = extract_features(model, img_dom, d_dom,is_target)  # 提取不变特征

            # 使用稳定特征进行分类
            logits = model.stable_classifier(content)  # 分类

            if not is_target:  # only source
                # 计算交叉熵损失
                losses_cls.append(F.cross_entropy(logits, label_dom))
                y_s.append(logits)
                labels_s.append(label_dom)
            else:
                y_t = logits  # 如果是目标域，保存logit以便计算目标域的损失

        # 分类损失
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = mean_loss_cls

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(mean_loss_cls.item(), y_s.size(0))  # 更新分类损失
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        # 每隔一定频率打印进度信息
        if i % args.print_freq == 0:
            # 切换到评估模式
            model.eval()

            # 提取目标域数据
            img_t = img_all[d_all == args.n_domains - 1]
            labels_t = label_all[d_all == args.n_domains - 1]

            with torch.no_grad():
                # 使用稳定特征模型进行预测
                y = model(img_t, d_all[d_all == args.n_domains - 1])
                cls_t_acc = accuracy(y, labels_t)[0]  # 计算目标域准确率
                val_accs.update(cls_t_acc.item(), img_t.size(0))

            # 切换回训练模式
            model.train()

            # 打印进度并记录日志
            progress.display(i)
            wandb.log({
                "Stable Model Train Loss": cls_losses.avg,
                "Stable Model Train Accuracy": cls_accs.avg,
                "Target Domain Accuracy": cls_t_acc.item(),
                "Train Source Acc": cls_accs.avg,
                "Train Source Cls Loss": cls_losses.avg,
            })

def train_unstable(train_source_iter: ForeverDataIterator,
                   train_target_iter: ForeverDataIterator,
                   model, optimizer: SGD, lr_scheduler: LambdaLR,
                   epoch: int, args: argparse.Namespace, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.1f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.1f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs],
        prefix="Unstable Model Epoch: [{}]".format(epoch)
    )

    # 切换到训练模式
    model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()

        # 计时并加载数据
        data_time.update(time.time() - end)
        img_s, labels_s, d_s, _ = next(train_source_iter)
        img_t, labels_t, d_t, _ = next(train_target_iter)

        # 将图像和标签数据移至GPU
        img_s = img_s.to(device)
        img_t = img_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # 将源域和目标域的图像合并
        img_all = torch.cat([img_s, img_t], 0)
        d_all = torch.cat([d_s, d_t], 0).to(device)
        label_all = torch.cat([labels_s, labels_t], 0)

        # 初始化各个损失项的列表
        losses_cls = []
        z_all = []  # 保存所有的潜变量z
        y_t = None
        y_s = []  # 用于保存源域的logits
        labels_s = []  # 用于保存源域的标签

        # 遍历每个域，针对每个源域进行训练
        for id in range(args.n_domains):  # 遍历每个源域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域


            # 获取当前域的样本数据
            index = d_all == id  # 获取当前域的样本
            label_dom = label_all[index]
            img_dom = img_all[index]
            d_dom = d_all[index]

            # 提取不稳定特征（style）
            _, style = extract_features(model, img_dom, d_dom,is_target)  # 提取可变特征

            # 使用不稳定特征进行分类
            logits = model.unstable_classifier(style)  # 分类

            if not is_target:  # only source
                # 计算交叉熵损失
                losses_cls.append(F.cross_entropy(logits, label_dom))
                y_s.append(logits)
                labels_s.append(label_dom)
            else:
                y_t = logits# 如果是目标域，保存logit以便计算目标域的损失

        # 分类损失
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = mean_loss_cls

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(mean_loss_cls.item(), y_s.size(0))  # 更新分类损失
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        # 每隔一定频率打印进度信息
        if i % args.print_freq == 0:
            # 切换到评估模式
            model.eval()

            # 提取目标域数据
            img_t = img_all[d_all == args.n_domains - 1]
            labels_t = label_all[d_all == args.n_domains - 1]

            with torch.no_grad():
                # 使用不稳定特征模型进行预测
                y = model(img_t, d_all[d_all == args.n_domains - 1])
                cls_t_acc = accuracy(y, labels_t)[0]  # 计算目标域准确率
                val_accs.update(cls_t_acc.item(), img_t.size(0))

            # 切换回训练模式
            model.train()

            # 打印进度并记录日志
            progress.display(i)

            wandb.log({
                "Unstable Model Train Loss": mean_loss_cls.avg,
                "Unstable Model Train Accuracy": cls_accs.avg,
                "Target Domain Accuracy": cls_t_acc.item(),
            })


def finetune_unstable_with_pseudo_labels(stable_model, unstable_model, train_target_iter,
                                        optimizer, lr_scheduler, epoch, args, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.1f')  # 分类准确率
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
    target_val_domains = d_t_all[target_train_size:] # 提取验证数据对应的域信息

    # 提取目标域数据
    target_val_data = target_val_data.to(device)
    target_val_labels = target_val_labels.to(device)
    target_val_domains = target_val_domains.to(device)

    # 使用稳定模型生成伪标签
    stable_model.eval()
    with torch.no_grad():
        # 获取稳定模型的输出并生成伪标签
        stable_logits = stable_model.stable_classifier(extract_features(stable_model, target_train_data, d_t_all[target_train_idx])[0])
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
        _, style = extract_features(unstable_model, target_train_data, d_t[:target_train_size])  # 提取不变特征

        # 使用不稳定特征进行分类
        logits = unstable_model.unstable_classifier(style)  # 分类

        # 计算交叉熵损失与伪标签之间的损失
        loss = F.cross_entropy(logits, pseudo_labels)

        # 更新统计指标
        acc = accuracy(logits, pseudo_labels)[0]
        cls_losses.update(loss.item(), target_train_data.size(0))
        cls_accs.update(acc.item(), target_train_data.size(0))

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()


        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        # 每隔一定频率打印进度信息
        if i % args.print_freq == 0:
            # 切换到评估模式
            unstable_model.eval()

            with torch.no_grad():
                # 使用不稳定特征模型进行验证
                val_logits = unstable_model(target_val_data, target_val_domains)  # 补上域信息
                val_loss = F.cross_entropy(val_logits, target_val_labels)
                val_acc = accuracy(val_logits, target_val_labels)[0]
                progress.display(i)
            unstable_model.train()

            progress.display(i)

            # 记录验证的损失和准确率
            wandb.log({
                "Unstable Model Fine-tuning Loss": cls_losses.avg,
                "Unstable Model Fine-tuning Accuracy": cls_accs.avg,
                "Validation Loss": val_loss.item(),
                "Validation Accuracy": val_acc.item(),
            })
