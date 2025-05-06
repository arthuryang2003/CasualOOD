
import random
import time
import warnings
import argparse
from audioop import error

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.metric import accuracy
import wandb
from common.utils import ForeverDataIterator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_mmd(x: torch.Tensor, domain_labels: torch.Tensor,
                kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma=None) -> torch.Tensor:
    """
    输入:
        x: (N, D) 特征（例如 z_u）
        domain_labels: (N,) 域标签（整型张量）
    输出:
        mmd_loss: 标量张量，表示不同域之间的 MMD 平均差异
    """
    unique_domains = domain_labels.unique()
    domain_features = [x[domain_labels == dom] for dom in unique_domains]

    mmd_loss = 0.
    count = 0
    for i in range(len(domain_features)):
        for j in range(i + 1, len(domain_features)):
            xi = domain_features[i]
            xj = domain_features[j]
            if xi.size(0) < 2 or xj.size(0) < 2:
                continue
            mmd_loss += _mmd_pairwise(xi, xj, kernel_mul, kernel_num, fix_sigma)
            count += 1

    return mmd_loss / max(count, 1)


def _gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
    total = torch.cat([source, target], dim=0)
    n_samples = total.size(0)
    L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth = torch.clamp(bandwidth, min=1e-3)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernels = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernels)  # (N+M, N+M)


def _mmd_pairwise(source, target, kernel_mul, kernel_num, fix_sigma):
    n = source.size(0)
    m = target.size(0)
    kernels = _gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)

    XX = kernels[:n, :n].mean()
    YY = kernels[n:, n:].mean()
    XY = kernels[:n, n:].mean()
    YX = kernels[n:, :n].mean()
    return XX + YY - XY - YX

def compute_conditional_MI(zu, zs, y, num_classes):
    batch_size, feat_dim = zu.size()

    # one-hot
    one_hot = F.one_hot(y, num_classes=num_classes).float()

    # 计算每个类别的均值
    sum_zs = one_hot.T @ zs  # [num_classes, feat_dim]
    count_zs = one_hot.sum(dim=0, keepdim=True).T + 1e-6
    mean_zs = sum_zs / count_zs

    # 每个样本对应的同类均值
    mean_zs_per_sample = mean_zs[y]  # [batch_size, feat_dim]

    # 残差
    diff = zs - mean_zs_per_sample  # [batch_size, feat_dim]

    # 乘上zu
    weighted_diff = zu * diff  # 元素乘 [batch_size, feat_dim]

    # 求所有样本均值
    avg_weighted_diff = weighted_diff.mean(dim=0)  # [feat_dim]

    # 最后取L1范数
    loss_MI = torch.norm(avg_weighted_diff, p=1)

    return loss_MI


def CasualOOD_train(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
                    model, optimizer: torch.optim.SGD,
                    lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, args: argparse.Namespace,
                    total_iter: int, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('total', ':4.2f')  # 解藕损失
    KL_losses = AverageMeter('KL', ':4.2f')  # 解藕损失
    MI_losses = AverageMeter('MI', ':4.2f')  # 解藕损失
    stable_cls_losses = AverageMeter('Cls_u', ':4.2f')
    unstable_cls_losses = AverageMeter('Cls_s', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    MMD_losses = AverageMeter('MMD', ':4.2f')
    DomCls_losses = AverageMeter('DomCls', ':4.2f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs,stable_cls_losses,unstable_cls_losses, KL_losses,MI_losses,MMD_losses, DomCls_losses,val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        # 从源域中获取一批数据
        train_minibatches = next(train_source_iter)  # list of (x, y)
        val_minibatches = next(val_iter)  # list of (x, y)

        # 将不同 domain 的数据合并
        img_train = torch.cat([d[0] for d in train_minibatches])
        labels_train = torch.cat([d[1] for d in train_minibatches])

        img_val = torch.cat([d[0] for d in val_minibatches])
        labels_val = torch.cat([d[1] for d in val_minibatches])

        domains_train = torch.cat([d for x, y, d in train_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        domains_train= domains_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)

        logits = combined_logits

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        if args.loss_selection_mode == "add":
            # 分别计算 stable 与 unstable 分类器的损失后相加
            loss_cls = loss_cls_u + loss_cls_s

        elif args.loss_selection_mode == "concat":

            loss_cls = F.cross_entropy(logits, labels_train)

        else:
            raise ValueError(f"Unsupported loss_selection_mode: {args.loss_selection_mode}")

        # === MMD Loss: Encourage z_u independence from domain ===
        loss_mmd = compute_mmd(z_u, domains_train)

        # === Domain Classification Loss on z_s ===
        domain_logits = model.domain_classifier(z_s)
        loss_domain_cls = F.cross_entropy(domain_logits, domains_train)


        # 解耦损失（互信息近似）
        if args.mi_type == 'conditional':
            loss_MI = compute_conditional_MI(z_u, z_s, labels_train, args.num_classes)
        else:  # 'cosine'
            sim = F.cosine_similarity(z_u, z_s, dim=1)
            loss_MI = torch.mean(sim ** 2)

        # KL散度损失（不稳定特征）
        q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
        log_qz = q_dist.log_prob(s_logits)
        loss_kl = -log_qz.mean()

        # 总损失 = 分类 + KL + 互信息
        loss = (loss_cls
                # + args.decouple_alpha * loss_kl
                + args.decouple_beta * loss_MI
                + args.mmd_lambda * loss_mmd
                + args.domain_lambda * loss_domain_cls)
        # 分类准确率
        cls_acc = accuracy(logits, labels_train)[0]

        # 统计指标更新
        cls_losses.update(loss_cls.item(), logits.size(0))
        stable_cls_losses.update(loss_cls_u.item(), logits.size(0))
        unstable_cls_losses.update(loss_cls_s.item(), logits.size(0))
        cls_accs.update(cls_acc.item(), logits.size(0))
        total_losses.update(loss.item(), logits.size(0))
        KL_losses.update(loss_kl.item(), logits.size(0))
        MI_losses.update(loss_MI.item(), logits.size(0))
        MMD_losses.update(loss_mmd.item(), logits.size(0))
        DomCls_losses.update(loss_domain_cls.item(), logits.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)

            with torch.no_grad():
                y = model(img_val)
                cls_t_acc = accuracy(y, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Train Val Acc": cls_t_acc.item(),
                "Train Acc": cls_acc.item(),
                "Train Loss": loss.item(),
                "Train Cls Loss": loss_cls.item(),
                "Train KL": args.decouple_alpha *loss_kl.item(),
                "Train MI Loss": args.decouple_beta *loss_MI.item(),

            })

def CasualOOD_finetune(train_target_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
                       model, optimizer: torch.optim.SGD,
                       lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, args: argparse.Namespace,
                       total_iter: int, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('total', ':4.2f')  # 总损失
    stable_cls_losses = AverageMeter('Cls_u', ':4.2f')
    unstable_cls_losses = AverageMeter('Cls_s', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses, cls_accs, val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )

    # Switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # Measure data loading time
        data_time.update(time.time() - end)


        # 从目标域中获取一批数据
        train_minibatches = next(train_target_iter)  # list of (x, y)
        val_minibatches = next(val_iter)

        # 合并不同 domain 的样本
        img_train = torch.cat([d[0] for d in train_minibatches])
        # labels_train = torch.cat([d[1] for d in train_minibatches])
        img_val = torch.cat([d[0] for d in val_minibatches])
        labels_val = torch.cat([d[1] for d in val_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)

        # 特征提取与伪标签生成
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        stable_pred_hard=F.softmax(u_logits,dim=1)
        pseudo_labels = torch.argmax(stable_pred_hard, dim=1)

        # 分类损失（仅不稳定分支）
        if args.finetune_logits == 'tilde':
            logits = tilde_s_logits
        elif args.finetune_logits == 'combined':
            logits = combined_logits
        else:
            raise NotImplementedError

        loss_cls = F.cross_entropy(logits, pseudo_labels)

        # 准确率计算
        cls_acc = accuracy(logits, pseudo_labels)[0]

        # 总损失
        loss = loss_cls

        # 更新指标
        cls_losses.update(loss.item(), logits.size(0))
        unstable_cls_losses.update(loss.item(), logits.size(0))
        cls_accs.update(cls_acc.item(), logits.size(0))
        total_losses.update(loss.item(), logits.size(0))

        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # 更新时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)

            with torch.no_grad():
                y = model(img_val)
                cls_t_acc = accuracy(y, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Finetune Val Acc": cls_t_acc.item(),
                "Finetune Acc": cls_acc.item(),
                "Finetune Loss": loss.item()
            })

def CasualOOD_train1(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
                    model, optimizer: torch.optim.SGD,
                    lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, args: argparse.Namespace,
                    total_iter: int, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('total', ':4.2f')  # 解藕损失
    KL_losses = AverageMeter('KL', ':4.2f')  # 解藕损失
    MI_losses = AverageMeter('MI', ':4.2f')  # 解藕损失
    stable_cls_losses = AverageMeter('Cls_u', ':4.2f')
    unstable_cls_losses = AverageMeter('Cls_s', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    MMD_losses = AverageMeter('MMD', ':4.2f')
    DomCls_losses = AverageMeter('DomCls', ':4.2f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs,stable_cls_losses,unstable_cls_losses, KL_losses,MI_losses,MMD_losses, DomCls_losses,val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        # 从源域中获取一批数据
        train_minibatches = next(train_source_iter)  # list of (x, y)
        val_minibatches = next(val_iter)  # list of (x, y)

        # 将不同 domain 的数据合并
        img_train = torch.cat([d[0] for d in train_minibatches])
        labels_train = torch.cat([d[1] for d in train_minibatches])

        img_val = torch.cat([d[0] for d in val_minibatches])
        labels_val = torch.cat([d[1] for d in val_minibatches])

        domains_train = torch.cat([d for x, y, d in train_minibatches])
        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        domains_train= domains_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        logits = u_logits

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        loss_cls = loss_cls_u
        # === MMD Loss: Encourage z_u independence from domain ===
        loss_mmd = compute_mmd(z_u, domains_train)

        # === Domain Classification Loss on z_s ===
        domain_logits = model.domain_classifier(z_s)
        loss_domain_cls = F.cross_entropy(domain_logits, domains_train)

        # 解耦损失（互信息近似）
        if args.mi_type == 'conditional':
            loss_MI = compute_conditional_MI(z_u, z_s, labels_train, args.num_classes)
        else:  # 'cosine'
            sim = F.cosine_similarity(z_u, z_s, dim=1)
            loss_MI = torch.mean(sim ** 2)

        # KL散度损失（不稳定特征）
        q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
        log_qz = q_dist.log_prob(s_logits)
        loss_kl = -log_qz.mean()


        # 总损失 = 分类 + KL + 互信息
        loss = (loss_cls
                # + args.decouple_alpha * loss_kl
                + args.decouple_beta * loss_MI
                + args.mmd_lambda * loss_mmd
                + args.domain_lambda * loss_domain_cls)
        # 分类准确率
        cls_acc = accuracy(logits, labels_train)[0]

        # 统计指标更新
        cls_losses.update(loss_cls.item(), logits.size(0))
        stable_cls_losses.update(loss_cls_u.item(), logits.size(0))
        unstable_cls_losses.update(loss_cls_s.item(), logits.size(0))
        cls_accs.update(cls_acc.item(), logits.size(0))
        total_losses.update(loss.item(), logits.size(0))
        KL_losses.update(loss_kl.item(), logits.size(0))
        MI_losses.update(args.decouple_beta *loss_MI.item(), logits.size(0))
        MMD_losses.update(args.mmd_lambda * loss_mmd.item(), logits.size(0))
        DomCls_losses.update(args.domain_lambda *loss_domain_cls.item(), logits.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)

            with torch.no_grad():
                _,_,u_logits,_,_,_=model.encode(img_val)
                cls_t_acc = accuracy(u_logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Train Phase 1 Val Acc": cls_t_acc.item(),
                "Train Phase 1 Acc": cls_acc.item(),
                "Train Phase 1 Loss": loss.item(),
                "Train Phase 1 Cls Loss": loss_cls.item(),
                "Train Phase 1 KL": args.decouple_alpha *loss_kl.item(),
                "Train Phase 1 MI Loss": args.decouple_beta *loss_MI.item(),

            })



def CasualOOD_train2(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
                    model, optimizer: torch.optim.SGD,
                    lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, args: argparse.Namespace,
                    total_iter: int, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('total', ':4.2f')  # 解藕损失
    KL_losses = AverageMeter('KL', ':4.2f')  # 解藕损失
    MI_losses = AverageMeter('MI', ':4.2f')  # 解藕损失
    stable_cls_losses = AverageMeter('Cls_u', ':4.2f')
    unstable_cls_losses = AverageMeter('Cls_s', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs,unstable_cls_losses, val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        # 从源域中获取一批数据
        train_minibatches = next(train_source_iter)  # list of (x, y)
        val_minibatches = next(val_iter)  # list of (x, y)

        # 将不同 domain 的数据合并
        img_train = torch.cat([d[0] for d in train_minibatches])
        labels_train = torch.cat([d[1] for d in train_minibatches])

        img_val = torch.cat([d[0] for d in val_minibatches])
        labels_val = torch.cat([d[1] for d in val_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        # 分类损失（仅不稳定分支）
        if args.finetune_logits == 'tilde':
            logits = tilde_s_logits
        elif args.finetune_logits == 'combined':
            logits = combined_logits
        else:
            raise NotImplementedError

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        loss_cls = F.cross_entropy(logits, labels_train)

        loss = loss_cls

        # 准确率计算
        cls_acc = accuracy(logits, labels_train)[0]

        # 统计指标更新
        cls_losses.update(loss_cls.item(), logits.size(0))
        stable_cls_losses.update(loss_cls_u.item(), logits.size(0))
        unstable_cls_losses.update(loss_cls_s.item(), logits.size(0))
        cls_accs.update(cls_acc.item(), logits.size(0))
        total_losses.update(loss.item(), logits.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)


            with torch.no_grad():
                z_u,z_s,u_logits,s_logits,tilde_s_logits,combined_logits=model.encode(img_val)
                cls_t_acc = accuracy(tilde_s_logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Train Phase 2 Val Acc": cls_t_acc.item(),
                "Train Phase 2 Acc": cls_acc.item(),
                "Train Phase 2 Loss": loss.item(),
                "Train Phase 2 Cls Loss": loss_cls.item(),
            })


def ERM_train(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
              model, optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
              epoch: int, args: argparse.Namespace, total_iter: int):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':4.2f')
    accs = AverageMeter('Acc', ':3.2f')
    val_accs = AverageMeter('Val Acc', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, accs, val_accs],
        prefix="ERM Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        data_time.update(time.time() - end)

        minibatches = next(train_source_iter)
        x = torch.cat([d[0] for d in minibatches]).to(device)
        y = torch.cat([d[1] for d in minibatches]).to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc = accuracy(logits, y)[0]
        losses.update(loss.item(), x.size(0))
        accs.update(acc.item(), x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            val_minibatches = next(val_iter)
            x_val = torch.cat([d[0] for d in val_minibatches]).to(device)
            y_val = torch.cat([d[1] for d in val_minibatches]).to(device)

            with torch.no_grad():
                val_logits = model(x_val)
                val_acc = accuracy(val_logits, y_val)[0]
                val_accs.update(val_acc.item(), x_val.size(0))

            model.train()
            progress.display(i)

            wandb.log({
                "ERM Train Loss": loss.item(),
                "ERM Train Acc": acc.item(),
                "ERM Val Acc": val_acc.item()
            })


def IRM_train(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
              model, optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
              epoch: int, args: argparse.Namespace, total_iter: int):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':4.2f')
    accs = AverageMeter('Acc', ':3.2f')
    penalties = AverageMeter('Penalty', ':4.2f')
    val_accs = AverageMeter('Val Acc', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, penalties, accs, val_accs],
        prefix="IRM Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        data_time.update(time.time() - end)

        minibatches = next(train_source_iter)
        x = torch.cat([d[0] for d in minibatches]).to(device)
        y = torch.cat([d[1] for d in minibatches]).to(device)

        loss, nll, penalty = model.get_penalized_loss(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        model.update_count += 1


        logits = model(x)
        acc = accuracy(logits, y)[0]

        losses.update(loss.item(), x.size(0))
        accs.update(acc.item(), x.size(0))
        penalties.update(penalty, x.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            val_minibatches = next(val_iter)
            x_val = torch.cat([d[0] for d in val_minibatches]).to(device)
            y_val = torch.cat([d[1] for d in val_minibatches]).to(device)

            with torch.no_grad():
                val_logits = model(x_val)
                val_acc = accuracy(val_logits, y_val)[0]
                val_accs.update(val_acc.item(), x_val.size(0))

            model.train()
            progress.display(i)

            wandb.log({
                "IRM Train Loss": loss.item(),
                "IRM Train Acc": acc.item(),
                "IRM Penalty": penalty,
                "IRM Val Acc": val_acc.item()
            })
