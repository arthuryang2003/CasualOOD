
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


def train_VAE(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
          model, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, total_iter, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')  # 重建损失
    vae_losses = AverageMeter('VAE', ':4.2f')  # VAE损失
    kl_losses = AverageMeter('KL', ':4.2f')  # KL散度损失
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, vae_losses, recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )
    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(),
                                                                 torch.eye(args.z_dim).cuda())

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
        losses_kl = []
        z_all = []  # 保存所有的潜变量z
        y_s = []  # 用于保存源域的logits
        labels_s = []  # 用于保存源域的标签
        x_all = []  # 用于保存源域和目标域的特征
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
            x_dom = model.backbone(img_dom, track_bn=is_target)
            z, tilde_z, mu, log_var, logdet_u, logit = model.encode(x_dom, u=d_dom, track_bn=is_target)

            # VAE KL Loss
            q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
            kl = (log_qz.sum(dim=1) - log_pz).mean()

            # 设置一个平滑的C值，以便在迭代过程中平滑调节
            C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()  # 计算KL损失


            losses_cls.append(F.cross_entropy(logit, label_dom))
            y_s.append(logit)
            labels_s.append(label_dom)

            losses_kl.append(loss_kl)
            x_all.append(x_dom)
            z_all.append(z)

        x_all = torch.cat(x_all, 0)  # 将所有域的特征合并
        z_all = torch.cat(z_all, 0)  # 合并所有的潜变量
        x_all_hat = model.decode(z_all)  # 解码潜变量，恢复图像

        # vae loss
        mean_loss_recon = F.mse_loss(x_all, x_all_hat, reduction='sum') / len(x_all)
        mean_loss_kl = torch.stack(losses_kl, dim=0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl

        # source classification
        mean_loss_cls = torch.stack(losses_cls, 0).mean()


        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = mean_loss_cls \
               + args.lambda_vae * mean_loss_vae

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(mean_loss_cls.item(), y_s.size(0))  # 更新分类损失
        recon_losses.update(mean_loss_recon.item(), x_all.size(0))  # 更新重建损失
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))  # 更新VAE损失
        kl_losses.update(mean_loss_kl.item(), x_all.size(0))  # 更新KL损失
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
                y = model(img_val,d_val)
                cls_t_acc = accuracy(y, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))
            model.train()

            progress.display(i)

            wandb.log({
                "Train Val Acc": cls_t_acc.item(),
                "Train Source Acc": cls_acc.item(),
                "Train Source Cls Loss": mean_loss_cls.item(),
                "Train Reconstruction Loss": mean_loss_recon.item(),
                "Train VAE Loss": mean_loss_vae.item(),
                "Train KL": mean_loss_kl.item(),
            })


def train_decoupler(train_source_iter: ForeverDataIterator, val_iter: ForeverDataIterator,
                    model, optimizer: torch.optim.SGD,
                    lr_scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int, args: argparse.Namespace,
                    total_iter: int, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    total_losses = AverageMeter('total', ':4.2f')  # 解藕损失
    KL_losses = AverageMeter('KL', ':4.2f')  # 解藕损失
    MI_losses = AverageMeter('MI', ':4.2f')  # 解藕损失

    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, total_losses,cls_accs, KL_losses,MI_losses, val_accs],
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

            logits = model(img_dom)  # 获得分类结果和解藕后的特征
            z_u,z_s=model.extrect_feature(img_dom)
            # 分类损失
            loss_cls = F.cross_entropy(logits, label_dom)

            # 计算解藕损失（互信息损失）
            # 计算不变特征和虚假特征之间的余弦相似度
            sim = F.cosine_similarity(z_u, z_s, dim=1)
            # 通过L2损失强制二者解耦（相似度越小越好）
            loss_MI = torch.mean(sim ** 2)

            # 计算可变特征的KL散度损失
            # 目标是使虚假特征的分布接近标准正态分布
            q_dist = torch.distributions.Normal(torch.zeros_like(z_s), torch.ones_like(z_s))
            log_qz = q_dist.log_prob(z_s)
            loss_kl = -log_qz.mean()

            # 解藕总损失 = 解藕正则化损失 + KL损失

            losses_cls.append(loss_cls)
            losses_MI.append(loss_MI)
            losses_KL.append(loss_kl)
            y_s.append(logits)
            labels_s.append(label_dom)
            z_all.append(z_u)

        # 计算总的分类损失、解藕损失和KL损失
        mean_cls_losses = torch.stack(losses_cls, 0).mean()
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

def train_stable(train_source_iter: ForeverDataIterator,
                 val_iter: ForeverDataIterator,
                 decoupler, stable_classifier, optimizer: SGD, lr_scheduler: LambdaLR,
                 epoch: int, args: argparse.Namespace, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs, val_accs],
        prefix="Stable Classifier Epoch: [{}]".format(epoch)
    )

    # 切换到训练模式
    stable_classifier.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        stable_classifier.train()

        # 计时并加载数据
        data_time.update(time.time() - end)
        img_s, labels_s, d_s, _ = next(train_source_iter)

        # 从源域中获取一批数据
        img_train, labels_train, d_train, _ = next(train_source_iter)

        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        d_train = d_train.to(device)


        # 初始化各个损失项的列表
        losses_cls = []
        y_s = [] # 用于保存源域的logits
        labels_s = []# 用于保存源域的标签


        # 遍历每个域，针对每个源域进行训练
        for id in range(args.n_domains):  # 遍历每个源域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域
            if is_target:
                continue
            index = d_train == id  # 获取当前域的样本
            label_dom = labels_train[index]
            img_dom = img_train[index]
            d_dom = d_train[index]

            # 提取稳定特征（content）
            content, _ = decoupler.extract_feature(img_dom)  # 提取不变特征

            # 使用稳定特征进行分类
            logits = stable_classifier.classifier(content)  # 分类

            losses_cls.append(F.cross_entropy(logits, label_dom))
            y_s.append(logits)
            labels_s.append(label_dom)


        # 分类损失
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # 总损失 = 分类损失
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
            stable_classifier.eval()

            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)


            with torch.no_grad():
                # 提取稳定特征（content）
                content, _ = decoupler.extract_feature(img_dom)  # 提取不变特征
                # 使用稳定特征进行分类
                logits = stable_classifier.classifier(content)  # 分类

                cls_t_acc = accuracy(logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))

            # 切换回训练模式
            stable_classifier.train()

            # 打印进度并记录日志
            progress.display(i)

            wandb.log({
                "Stable Classifier Train Loss": mean_loss_cls.item(),
                "Stable Classifier Train Accuracy": cls_acc.item(),
                "Stable Classifier Val Accuracy": cls_t_acc.item(),

            })

def train_unstable(train_source_iter: ForeverDataIterator,
                   val_iter: ForeverDataIterator,
                   decoupler,stable_classifier,unstable_classifier, optimizer: SGD, lr_scheduler: LambdaLR,
                   epoch: int, args: argparse.Namespace, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs, val_accs],
        prefix="Unstable Classifier Epoch: [{}]".format(epoch)
    )

    stable_classifier.eval()

    # 切换到训练模式
    unstable_classifier.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        stable_classifier.train()

        # 计时并加载数据
        data_time.update(time.time() - end)
        img_s, labels_s, d_s, _ = next(train_source_iter)

        # 从源域中获取一批数据
        img_train, labels_train, d_train, _ = next(train_source_iter)

        img_val, labels_val, d_val, _ = next(val_iter)

        # 将图像和标签数据移至GPU
        img_train = img_train.to(device)
        labels_train = labels_train.to(device)
        d_train = d_train.to(device)


        # 初始化各个损失项的列表
        losses_cls = []
        y_s = [] # 用于保存源域的logits
        labels_s = []# 用于保存源域的标签


        # 遍历每个域，针对每个源域进行训练
        for id in range(args.n_domains):  # 遍历每个源域
            domain_id = id
            is_target = domain_id == args.n_domains - 1  # 判断是否为目标域
            if is_target:
                continue
            index = d_train == id  # 获取当前域的样本
            label_dom = labels_train[index]
            img_dom = img_train[index]
            d_dom = d_train[index]

            # 提取稳定特征（content）
            content, _ = decoupler.extract_feature(img_dom) # 提取不变特征

            # 通过稳定分类器生成伪标签
            with torch.no_grad():
                stable_logits = stable_classifier.classifier(content)

            # 生成伪标签
            pseudo_labels = torch.argmax(stable_logits, dim=1)
            pseudo_labels = pseudo_labels.to(device)  # 使用伪标签

            # 计算伪标签与真实标签之间的准确率
            correct = (pseudo_labels == label_dom).sum().item()  # 计算匹配的样本数
            total = label_dom.size(0)  # 总样本数
            stab_acc = correct / total * 100  # 准确率百分比

            # 提取不稳定特征（style）
            _, style = decoupler.extract_feature(img_dom)  # 提取可变特征
            # 使用不稳定特征进行分类
            logits = unstable_classifier.classifier(style)  # 分类

            losses_cls.append(F.cross_entropy(logits, pseudo_labels))
            y_s.append(logits)
            labels_s.append(pseudo_labels)

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

            print("stab_acc:", stab_acc)
            # 切换到评估模式
            unstable_classifier.eval()

            # 将图像和标签数据移至GPU
            img_val = img_val.to(device)
            labels_val = labels_val.to(device)
            d_val = d_val.to(device)

            with torch.no_grad():
                # 提取不稳定特征（style）
                _, style = decoupler.extract_feature(img_dom)  # 提取可变特征
                # 使用不稳定特征进行分类
                logits = unstable_classifier.classifier(style)  # 分类

                cls_t_acc = accuracy(logits, labels_val)[0]
                val_accs.update(cls_t_acc.item(), img_val.size(0))

            # 切换回训练模式
            unstable_classifier.train()

            # 打印进度并记录日志
            progress.display(i)

            wandb.log({
                "Unstable Classifier Train Loss": mean_loss_cls.item(),
                "Unstable Classifier Train Accuracy": cls_acc.item(),
                "Unstable Classifier Val  Accuracy": cls_t_acc.item(),

            })


