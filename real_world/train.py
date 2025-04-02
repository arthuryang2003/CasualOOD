
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs,stable_cls_losses,unstable_cls_losses, KL_losses,MI_losses, val_accs],
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
        img_train = torch.cat([x for x, y in train_minibatches])
        labels_train = torch.cat([y for x, y in train_minibatches])

        img_val = torch.cat([x for x, y in val_minibatches])
        labels_val = torch.cat([y for x, y in val_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)

        logits = combined_logits

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        loss_cls = F.cross_entropy(logits, labels_train)
        # loss_cls =loss_cls_u+loss_cls_s

        # 解耦损失（互信息近似）
        sim = F.cosine_similarity(z_u, z_s, dim=1)
        loss_MI = torch.mean(sim ** 2)

        # KL散度损失（不稳定特征）
        q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
        log_qz = q_dist.log_prob(s_logits)
        loss_kl = -log_qz.mean()

        # 总损失 = 分类 + KL + 互信息
        loss = loss_cls + args.decouple_alpha * loss_kl + args.decouple_beta * loss_MI

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
        [batch_time, data_time, cls_losses, total_losses, cls_accs, stable_cls_losses, unstable_cls_losses, val_accs],
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
        img_train = torch.cat([x for x, y in train_minibatches]).to(device)
        labels_train = torch.cat([y for x, y in train_minibatches])  # 用于伪标签评估可选
        img_val = torch.cat([x for x, y in val_minibatches]).to(device)
        labels_val = torch.cat([y for x, y in val_minibatches]).to(device)

        # 特征提取与伪标签生成
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        pseudo_labels = torch.argmax(u_logits, dim=1)

        # 分类损失（仅不稳定分支）
        logits = combined_logits
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
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs,stable_cls_losses,unstable_cls_losses, KL_losses,MI_losses, val_accs],
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
        img_train = torch.cat([x for x, y in train_minibatches])
        labels_train = torch.cat([y for x, y in train_minibatches])

        img_val = torch.cat([x for x, y in val_minibatches])
        labels_val = torch.cat([y for x, y in val_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        logits = u_logits

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        loss_cls = loss_cls_u

        # 解耦损失（互信息近似）
        sim = F.cosine_similarity(z_u, z_s, dim=1)
        loss_MI = torch.mean(sim ** 2)

        # KL散度损失（不稳定特征）
        q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
        log_qz = q_dist.log_prob(s_logits)
        loss_kl = -log_qz.mean()

        # 总损失 = 分类 + KL + 互信息
        loss = loss_cls + args.decouple_alpha * loss_kl + args.decouple_beta * loss_MI

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
        img_train = torch.cat([x for x, y in train_minibatches])
        labels_train = torch.cat([y for x, y in train_minibatches])

        img_val = torch.cat([x for x, y in val_minibatches])
        labels_val = torch.cat([y for x, y in val_minibatches])

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(img_train)
        logits =  tilde_s_logits

        # 各类损失项
        loss_cls_u = F.cross_entropy(u_logits, labels_train)
        loss_cls_s = F.cross_entropy(tilde_s_logits, labels_train)
        loss_cls = F.cross_entropy(logits, labels_train)

        # 总损失 = 分类
        loss = loss_cls

        # 分类准确率
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

