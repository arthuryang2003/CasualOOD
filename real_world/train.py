
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
        img_train, labels_train, d_train, _ = next(train_source_iter)
        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        d_train = d_train.to(device)

        # 初始化各个损失项的列表
        losses_cls = []
        losses_cls_u = []
        losses_cls_s = []
        losses_MI = []
        losses_KL = []  # KL散度损失
        z_all = []  # 保存所有的不变特征 z_u
        y_s = []  # 用于保存源域的logits
        labels_s = []  # 用于保存源域的标签

        for id in range(args.n_domains):  # 遍历每个域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域
            if is_target:
                continue
            index = d_train == id  # 获取当前域的样本
            label_dom = labels_train[index]
            img_dom = img_train[index]
            d_dom = d_train[index]

            # 特征提取

            z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(img_dom)

            logits=u_logits+tilde_s_logits

            loss_cls_u=F.cross_entropy(u_logits, label_dom)

            loss_cls_s=F.cross_entropy(tilde_s_logits, label_dom)

            # 分类损失
            # loss_cls = loss_cls_u+loss_cls_s
            loss_cls=F.cross_entropy(logits, label_dom)

            # 计算解藕损失（互信息损失）
            # 计算不变特征和虚假特征之间的余弦相似度
            sim = F.cosine_similarity(z_u, z_s, dim=1)
            # 通过L2损失强制二者解耦（相似度越小越好）
            loss_MI = torch.mean(sim ** 2)

            # 计算可变特征的KL散度损失
            # 目标是让不稳定预测标签（基于z_s）接近标准正态分布
            q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
            log_qz = q_dist.log_prob(s_logits)
            loss_kl = -log_qz.mean()

            # 解藕总损失 = 解藕正则化损失 + KL损失

            losses_cls.append(loss_cls)
            losses_cls_u.append(loss_cls_u)
            losses_cls_s.append(loss_cls_s)
            losses_MI.append(loss_MI)
            losses_KL.append(loss_kl)
            y_s.append(logits)
            labels_s.append(label_dom)
            z_all.append(z_u)


        # 计算总的分类损失、解藕损失和KL损失
        mean_cls_losses = torch.stack(losses_cls, 0).mean()
        mean_cls_u = torch.stack(losses_cls_u, 0).mean()
        mean_cls_s = torch.stack(losses_cls_s, 0).mean()
        mean_MI_losses = torch.stack(losses_MI, 0).mean()
        mean_kl_losses = torch.stack(losses_KL, 0).mean()

        # 总损失 = 分类损失 + 解藕损失
        loss = mean_cls_losses + args.decouple_alpha * mean_kl_losses+args.decouple_beta*mean_MI_losses

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        # 更新统计数据
        cls_losses.update(mean_cls_losses.item(), y_s.size(0))  # 更新分类损失
        stable_cls_losses.update(mean_cls_u.item(), y_s.size(0))
        unstable_cls_losses.update(mean_cls_s.item(), y_s.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率
        total_losses.update(loss.item(), y_s.size(0))  # 更新总损失
        KL_losses.update(mean_kl_losses.item(), y_s.size(0))
        MI_losses.update(mean_MI_losses.item(), y_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 更新学习率

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)

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
                "Train Cls Loss": mean_cls_losses.item(),
                "Train KL": mean_kl_losses.item(),
                "Train MI Loss": mean_MI_losses.item(),

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
        img_train, labels_train, d_train, _ = next(train_source_iter)
        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        d_train = d_train.to(device)

        # 初始化各个损失项的列表
        losses_cls = []
        losses_cls_u = []
        losses_cls_s = []
        losses_MI = []
        losses_KL = []  # KL散度损失
        z_all = []  # 保存所有的不变特征 z_u
        y_s = []  # 用于保存源域的logits
        labels_s = []  # 用于保存源域的标签

        for id in range(args.n_domains):  # 遍历每个域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域
            if is_target:
                continue
            index = d_train == id  # 获取当前域的样本
            label_dom = labels_train[index]
            img_dom = img_train[index]
            d_dom = d_train[index]

            # 特征提取

            z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(img_dom)

            logits=u_logits

            loss_cls_u=F.cross_entropy(u_logits, label_dom)

            loss_cls_s=F.cross_entropy(tilde_s_logits, label_dom)

            # 分类损失
            loss_cls = loss_cls_u

            # 计算解藕损失（互信息损失）
            # 计算不变特征和虚假特征之间的余弦相似度
            sim = F.cosine_similarity(z_u, z_s, dim=1)
            # 通过L2损失强制二者解耦（相似度越小越好）
            loss_MI = torch.mean(sim ** 2)

            # 计算可变特征的KL散度损失
            # 目标是让不稳定预测标签（基于z_s）接近标准正态分布
            q_dist = torch.distributions.Normal(torch.zeros_like(s_logits), torch.ones_like(s_logits))
            log_qz = q_dist.log_prob(s_logits)
            loss_kl = -log_qz.mean()

            # 解藕总损失 = 解藕正则化损失 + KL损失

            losses_cls.append(loss_cls)
            losses_cls_u.append(loss_cls_u)
            losses_cls_s.append(loss_cls_s)
            losses_MI.append(loss_MI)
            losses_KL.append(loss_kl)
            y_s.append(logits)
            labels_s.append(label_dom)
            z_all.append(z_u)


        # 计算总的分类损失、解藕损失和KL损失
        mean_cls_losses = torch.stack(losses_cls, 0).mean()
        mean_cls_u = torch.stack(losses_cls_u, 0).mean()
        mean_cls_s = torch.stack(losses_cls_s, 0).mean()
        mean_MI_losses = torch.stack(losses_MI, 0).mean()
        mean_kl_losses = torch.stack(losses_KL, 0).mean()

        # 总损失 = 分类损失 + 解藕损失
        loss = mean_cls_losses + args.decouple_alpha * mean_kl_losses+args.decouple_beta*mean_MI_losses

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        # 更新统计数据
        cls_losses.update(mean_cls_losses.item(), y_s.size(0))  # 更新分类损失
        stable_cls_losses.update(mean_cls_u.item(), y_s.size(0))
        unstable_cls_losses.update(mean_cls_s.item(), y_s.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率
        total_losses.update(loss.item(), y_s.size(0))  # 更新总损失
        KL_losses.update(mean_kl_losses.item(), y_s.size(0))
        MI_losses.update(mean_MI_losses.item(), y_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 更新学习率

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)

            with torch.no_grad():
                _,_,u_logits,_,_=model.encode(img_val)
                cls_t_acc = accuracy(u_logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Train Phase 1 Val Acc": cls_t_acc.item(),
                "Train Phase 1 Acc": cls_acc.item(),
                "Train Phase 1 Loss": loss.item(),
                "Train Phase 1 Cls Loss": mean_cls_losses.item(),
                "Train Phase 1 KL": mean_kl_losses.item(),
                "Train Phase 1 MI Loss": mean_MI_losses.item(),

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
        img_train, labels_train, d_train, _ = next(train_target_iter)
        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        d_train = d_train.to(device)

        # 初始化各个损失项的列表
        losses_cls = []
        losses_cls_u = []
        losses_cls_s = []
        losses_MI = []
        losses_KL = []  # KL散度损失
        z_all = []  # 保存所有的不变特征 z_u
        y_s = []  # 用于保存目标域的logits
        labels_s = []  # 用于保存目标域的标签

        # 只针对目标域进行训练
        index = d_train == args.n_domains - 1  # 目标域样本

        # # 打印 encoder 的所有参数
        # print("Encoder Parameters:")
        # for name, param in model.encoder.named_parameters():
        #     print(f"{name}: {param.data}")

        # 特征提取
        z_u, z_s, u_logits, s_logits, tilde_s_logits = model.encode(img_train)

        # 生成伪标签
        pseudo_labels = torch.argmax(u_logits, dim=1)
        pseudo_labels = pseudo_labels.to(device)  # 使用伪标签

        loss_cls_s = F.cross_entropy(tilde_s_logits, pseudo_labels)

        logits =  tilde_s_logits
        
        # 分类损失
        loss_cls = loss_cls_s

        # 解藕总损失 = 解藕正则化损失 + KL损失
        losses_cls.append(loss_cls)
        # losses_cls_u.append(loss_cls_u)
        losses_cls_s.append(loss_cls_s)

        y_s.append(logits)
        labels_s.append(pseudo_labels)
        z_all.append(z_u)

        # 计算总的分类损失、解藕损失和KL损失
        mean_cls_losses = torch.stack(losses_cls, 0).mean()

        mean_cls_s = torch.stack(losses_cls_s, 0).mean()


        # 总损失 = 分类损失
        loss = mean_cls_losses

        # 合并目标域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        # 更新统计数据
        cls_losses.update(mean_cls_losses.item(), y_s.size(0))  # 更新分类损失
        unstable_cls_losses.update(mean_cls_s.item(), y_s.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率
        total_losses.update(loss.item(), y_s.size(0))  # 更新总损失

        # 计算梯度并执行 SGD 步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 更新学习率

        # 测量经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)

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
        img_train, labels_train, d_train, _ = next(train_source_iter)
        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        d_train = d_train.to(device)

        # 初始化各个损失项的列表
        losses_cls = []
        losses_cls_u = []
        losses_cls_s = []
        losses_MI = []
        losses_KL = []  # KL散度损失
        z_all = []  # 保存所有的不变特征 z_u
        y_s = []  # 用于保存源域的logits
        labels_s = []  # 用于保存源域的标签

        for id in range(args.n_domains):  # 遍历每个域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域
            if is_target:
                continue
            index = d_train == id  # 获取当前域的样本
            label_dom = labels_train[index]
            img_dom = img_train[index]
            d_dom = d_train[index]

            # 特征提取

            z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(img_dom)

            logits=tilde_s_logits

            loss_cls_u=F.cross_entropy(u_logits, label_dom)

            loss_cls_s=F.cross_entropy(tilde_s_logits, label_dom)

            # 分类损失
            loss_cls = loss_cls_s


            # 解藕总损失 = 解藕正则化损失 + KL损失

            losses_cls.append(loss_cls)
            losses_cls_u.append(loss_cls_u)
            losses_cls_s.append(loss_cls_s)
            y_s.append(logits)
            labels_s.append(label_dom)



        # 计算总的分类损失、解藕损失和KL损失
        mean_cls_losses = torch.stack(losses_cls, 0).mean()
        mean_cls_s = torch.stack(losses_cls_s, 0).mean()


        # 总损失 = 分类损失 + 解藕损失
        loss = mean_cls_losses

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        # 更新统计数据
        cls_losses.update(mean_cls_losses.item(), y_s.size(0))  # 更新分类损失
        unstable_cls_losses.update(mean_cls_s.item(), y_s.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率
        total_losses.update(loss.item(), y_s.size(0))  # 更新总损失

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # 更新学习率

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            model.eval()
            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)

            with torch.no_grad():
                z_u,z_s,u_logits,s_logits,tilde_s_logits=model.encode(img_val)
                cls_t_acc = accuracy(tilde_s_logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            # 记录训练过程的指标
            wandb.log({
                "Train Phase 2 Val Acc": cls_t_acc.item(),
                "Train Phase 2 Acc": cls_acc.item(),
                "Train Phase 2 Loss": loss.item(),
                "Train Phase 2 Cls Loss": mean_cls_losses.item(),

            })

