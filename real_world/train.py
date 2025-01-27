
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
    recon_losses = AverageMeter('Rec', ':4.2f')# 重建损失
    vae_losses = AverageMeter('VAE', ':4.2f') # VAE损失
    kl_losses = AverageMeter('KL', ':4.2f')# KL散度损失
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    ent_losses = AverageMeter('Ent', ':4.2f')# 熵损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses, recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Stable Model Epoch: [{}]".format(epoch)
    )

    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())

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

            # 获取当前域的样本数据
            index = d_all == id  # 获取当前域的样本
            label_dom = label_all[index] if not is_target else None
            img_dom = img_all[index]
            d_dom = d_all[index]

            # 特征提取
            x_dom = model.backbone(img_dom, track_bn=is_target)
            z, tilde_z, mu, log_var, logdet_u, _ = model.encode(x_dom, u=d_dom, track_bn=is_target)

            # VAE KL Loss
            q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
            kl = (log_qz.sum(dim=1) - log_pz).mean()

             # 设置一个平滑的C值，以便在迭代过程中平滑调节
            C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()# 计算KL损失

            # 提取稳定特征（content）
            content, _ = extract_features(model, img_dom, d_dom,is_target)  # 提取不变特征

            # 使用稳定特征进行分类
            logits = model.classifier(content)  # 分类


            if not is_target:  # only source
                # 计算交叉熵损失
                losses_cls.append(F.cross_entropy(logits, label_dom))
                y_s.append(logits)
                labels_s.append(label_dom)
            else:
                y_t = logits  # 如果是目标域，保存logit以便计算目标域的损失

            losses_kl.append(loss_kl)
            x_all.append(x_dom)
            z_all.append(z)

        x_all = torch.cat(x_all, 0) # 将所有域的特征合并
        z_all = torch.cat(z_all, 0)# 合并所有的潜变量
        x_all_hat = model.decode(z_all)# 解码潜变量，恢复图像

        # vae loss
        mean_loss_recon = F.mse_loss(x_all, x_all_hat, reduction='sum') / len(x_all)
        mean_loss_kl = torch.stack(losses_kl, dim=0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl

        # 分类损失
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # entropy loss
        loss_ent = torch.tensor(0.).to(device)
        if args.lambda_ent > 0:
            output_t = y_t
            entropy = F.cross_entropy(output_t, torch.softmax(output_t, dim=1), reduction='none').detach()
            index = torch.nonzero((entropy < args.entropy_thr).float()).squeeze(-1)
            select_output_t = output_t[index]
            if len(select_output_t) > 0:
                loss_ent = F.cross_entropy(select_output_t, torch.softmax(select_output_t, dim=1))

        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = mean_loss_cls \
               + args.lambda_vae * mean_loss_vae \
               + args.lambda_ent * loss_ent

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(mean_loss_cls.item(), y_s.size(0))  # 更新分类损失
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率

        recon_losses.update(mean_loss_recon.item(), x_all.size(0))# 更新重建损失
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))# 更新VAE损失
        ent_losses.update(loss_ent.item(), y_t.size(0))# 更新熵损失
        kl_losses.update(mean_loss_kl.item(), x_all.size(0))# 更新KL损失

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
                # 使用稳定特征模型进行预测
                y = model(img_t, d_all[d_all == args.n_domains - 1])
                cls_t_acc = accuracy(y, labels_t)[0]  # 计算目标域准确率
                val_accs.update(cls_t_acc.item(), img_t.size(0))

            # 切换回训练模式
            model.train()

            # 打印进度并记录日志
            progress.display(i)

            wandb.log({
                "Stable Model Train Loss": mean_loss_cls.item(),
                "Stable Model Train Accuracy": cls_acc.item(),
                "Stable Model Target Domain Accuracy": cls_t_acc.item(),
                "Stable Model Train Reconstruction Loss": mean_loss_recon.item(),
                "Stable Model Train VAE Loss": mean_loss_vae.item(),
                "Stable Model Entropy Loss": loss_ent.item(),
                "Stable Model Train KL": mean_loss_kl.item(),

            })

def train_unstable(stable_model,train_source_iter: ForeverDataIterator,
                   train_target_iter: ForeverDataIterator,
                   unstable_model, optimizer: SGD, lr_scheduler: LambdaLR,
                   epoch: int, args: argparse.Namespace, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')# 重建损失
    vae_losses = AverageMeter('VAE', ':4.2f') # VAE损失
    kl_losses = AverageMeter('KL', ':4.2f')# KL散度损失
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    ent_losses = AverageMeter('Ent', ':4.2f')# 熵损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f')  # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses, recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Unstable Model Epoch: [{}]".format(epoch)
    )

    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())

    for param in stable_model.parameters():
        param.requires_grad = False

    stable_model.eval()

    # 切换到训练模式
    unstable_model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        unstable_model.train()

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

            # 获取当前域的样本数据
            index = d_all == id  # 获取当前域的样本
            label_dom = label_all[index] if not is_target else None
            img_dom = img_all[index]
            d_dom = d_all[index]

            # 特征提取
            x_dom = unstable_model.backbone(img_dom, track_bn=is_target)
            z, tilde_z, mu, log_var, logdet_u, _ = unstable_model.encode(x_dom, u=d_dom, track_bn=is_target)

            # VAE KL Loss
            q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
            kl = (log_qz.sum(dim=1) - log_pz).mean()

             # 设置一个平滑的C值，以便在迭代过程中平滑调节
            C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()# 计算KL损失

            # 提取不稳定特征（style）
            _, style = extract_features(unstable_model, img_dom, d_dom,is_target)  # 提取可变特征

            # 使用不稳定特征进行分类
            logits = unstable_model.classifier(style)  # 分类

            if not is_target:  # only source
                # 计算交叉熵损失
                losses_cls.append(F.cross_entropy(logits, label_dom))
                y_s.append(logits)
                labels_s.append(label_dom)
            else:
                # 使用稳定模型生成伪标签
                with torch.no_grad():
                    # 获取稳定模型的输出并生成伪标签
                    stable_logits = stable_model(img_dom, d_dom)
                    pseudo_labels = torch.argmax(stable_logits, dim=1)
                pseudo_labels = pseudo_labels.to(device)  # 使用伪标签
                # 计算伪标签与真实标签之间的准确率
                correct = (pseudo_labels == target_train_labels).sum().item()  # 计算匹配的样本数
                total = target_train_labels.size(0)  # 总样本数
                stab_acc = correct / total * 100  # 准确率百分比

                # 计算交叉熵损失
                losses_cls.append(F.cross_entropy(logits, pseudo_labels))
                y_s.append(logits)
                labels_s.append(pseudo_labels)
                y_t = logits# 如果是目标域，保存logit以便计算目标域的损失

            losses_kl.append(loss_kl)
            x_all.append(x_dom)
            z_all.append(z)

        x_all = torch.cat(x_all, 0) # 将所有域的特征合并
        z_all = torch.cat(z_all, 0)# 合并所有的潜变量
        x_all_hat = unstable_model.decode(z_all)# 解码潜变量，恢复图像

        # vae loss
        mean_loss_recon = F.mse_loss(x_all, x_all_hat, reduction='sum') / len(x_all)
        mean_loss_kl = torch.stack(losses_kl, dim=0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl

        # 分类损失
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # entropy loss
        loss_ent = torch.tensor(0.).to(device)
        if args.lambda_ent > 0:
            output_t = y_t
            entropy = F.cross_entropy(output_t, torch.softmax(output_t, dim=1), reduction='none').detach()
            index = torch.nonzero((entropy < args.entropy_thr).float()).squeeze(-1)
            select_output_t = output_t[index]
            if len(select_output_t) > 0:
                loss_ent = F.cross_entropy(select_output_t, torch.softmax(select_output_t, dim=1))


        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = mean_loss_cls \
               + args.lambda_vae * mean_loss_vae \
               + args.lambda_ent * loss_ent

        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]

        cls_losses.update(mean_loss_cls.item(), y_s.size(0))  # 更新分类损失
        cls_accs.update(cls_acc.item(), y_s.size(0))  # 更新分类准确率

        recon_losses.update(mean_loss_recon.item(), x_all.size(0))# 更新重建损失
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))# 更新VAE损失
        ent_losses.update(loss_ent.item(), y_t.size(0))# 更新熵损失
        kl_losses.update(mean_loss_kl.item(), x_all.size(0))# 更新KL损失

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
            unstable_model.eval()

            # 提取目标域数据
            img_t = img_all[d_all == args.n_domains - 1]
            labels_t = label_all[d_all == args.n_domains - 1]

            with torch.no_grad():
                # 使用不稳定特征模型进行预测
                y = unstable_model(img_t, d_all[d_all == args.n_domains - 1])
                cls_t_acc = accuracy(y, labels_t)[0]  # 计算目标域准确率
                val_accs.update(cls_t_acc.item(), img_t.size(0))

            # 切换回训练模式
            unstable_model.train()

            # 打印进度并记录日志
            progress.display(i)

            wandb.log({
                "Unstable Model Train Loss": mean_loss_cls.item(),
                "Unstable Model Train Accuracy": cls_acc.item(),
                "Unstable Model Target Domain Accuracy": cls_t_acc.item(),
                "Unstable Model Train Reconstruction Loss": mean_loss_recon.item(),
                "Unstable Model Train VAE Loss": mean_loss_vae.item(),
                "Unstable Model Entropy Loss": loss_ent.item(),
                "Unstable Model Train KL": mean_loss_kl.item(),
            })


def finetune_unstable_with_pseudo_labels(stable_model, unstable_model, train_target_iter,
                                        optimizer, lr_scheduler, epoch, args, total_iter):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')# 重建损失
    vae_losses = AverageMeter('VAE', ':4.2f') # VAE损失
    kl_losses = AverageMeter('KL', ':4.2f')# KL散度损失
    ent_losses = AverageMeter('Ent', ':4.2f')  # 熵损失
    cls_losses = AverageMeter('Cls', ':4.2f')  # 分类损失
    cls_accs = AverageMeter('Cls Acc', ':3.2f')  # 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.2f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses, recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Unstable Model Fine-tuning Epoch: [{}]".format(epoch)
    )
    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())

    for param in stable_model.parameters():
        param.requires_grad = False

    # 切换到训练模式
    stable_model.eval()
    unstable_model.train()
    end = time.time()

    for i in range(args.iters_per_epoch):
        total_iter += 1
        unstable_model.train()

        # 计时并加载数据
        data_time.update(time.time() - end)

        # 获取目标域数据
        img_t_all, labels_t_all, d_t_all, _ = next(train_target_iter)
        img_t_all = img_t_all.to(device)
        labels_t_all = labels_t_all.to(device)
        d_t_all = d_t_all.to(device)

        # 选取目标域数据的一部分作为训练数据
        target_train_size = int(args.target_split_ratio * len(img_t_all))
        target_train_data = img_t_all[:target_train_size]
        target_train_domains = d_t_all[:target_train_size]
        target_train_labels = labels_t_all[:target_train_size]

        target_train_labels=target_train_labels.to(device)
        target_train_data=target_train_data.to(device)
        target_train_domains=target_train_domains.to(device)

        # 剩下的部分数据用于验证
        target_val_data = img_t_all[target_train_size:]
        target_val_labels = labels_t_all[target_train_size:]
        target_val_domains = d_t_all[target_train_size:]

        # 提取目标域数据
        target_val_data = target_val_data.to(device)
        target_val_labels = target_val_labels.to(device)
        target_val_domains = target_val_domains.to(device)



        # 使用稳定模型生成伪标签

        with torch.no_grad():
            # 获取稳定模型的输出并生成伪标签
            stable_logits = stable_model(target_train_data, target_train_domains)
            pseudo_labels = torch.argmax(stable_logits, dim=1)

        # 计算伪标签与真实标签之间的准确率
        correct = (pseudo_labels == target_train_labels).sum().item()  # 计算匹配的样本数
        total = target_train_labels.size(0)  # 总样本数
        stab_acc = correct / total * 100  # 准确率百分比


        pseudo_labels = pseudo_labels.to(device)  # 使用伪标签

        # 特征提取
        x_dom= unstable_model.backbone(target_train_data, track_bn=True)
        z, tilde_z, mu, log_var, logdet_u, _ = unstable_model.encode(x_dom, u=target_train_domains, track_bn=True)

        # VAE KL Loss
        q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
        log_qz = q_dist.log_prob(z)
        log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
        kl = (log_qz.sum(dim=1) - log_pz).mean()

        # 设置一个平滑的C值，以便在迭代过程中平滑调节
        C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)

        loss_kl = args.beta * (kl - C).abs()  # 计算KL损失


        # 提取不稳定特征（style）
        _, style = extract_features(unstable_model, target_train_data, target_train_domains,True)  # 提取不变特征

        # 使用不稳定特征进行分类
        logits = unstable_model.classifier(style)  # 分类

        y_t = logits  # 如果是目标域，保存logit以便计算目标域的损失

        x_hat = unstable_model.decode(z)# 解码潜变量，恢复图像

        # vae loss
        loss_recon = F.mse_loss(x_dom, x_hat, reduction='sum') / len(x_dom)

        loss_vae = loss_recon + loss_kl

        # 计算交叉熵损失与伪标签之间的损失
        loss_cls = F.cross_entropy(logits, pseudo_labels)

        # entropy loss
        loss_ent = torch.tensor(0.).to(device)
        if args.lambda_ent > 0:
            output_t = y_t
            entropy = F.cross_entropy(output_t, torch.softmax(output_t, dim=1), reduction='none').detach()
            index = torch.nonzero((entropy < args.entropy_thr).float()).squeeze(-1)
            select_output_t = output_t[index]
            if len(select_output_t) > 0:
                loss_ent = F.cross_entropy(select_output_t, torch.softmax(select_output_t, dim=1))

        # 总损失 = 分类损失 + VAE损失 + 熵损失
        loss = loss_cls  \
            + args.lambda_vae * loss_vae  \
            + args.lambda_ent * loss_ent

        # 更新统计指标
        acc = accuracy(logits, pseudo_labels)[0]
        cls_losses.update(loss_cls.item(), target_train_data.size(0))# 更新分类损失
        recon_losses.update(loss_recon.item(), x_dom.size(0))  # 更新重建损失
        vae_losses.update(loss_vae.item(), x_dom.size(0))# 更新VAE损失
        ent_losses.update(loss_ent.item(), y_t.size(0))# 更新熵损失
        kl_losses.update(loss_kl.item(), x_dom.size(0))# 更新KL损失
        cls_accs.update(acc.item(), target_train_data.size(0))# 更新分类准确率

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
            unstable_model.eval()

            with torch.no_grad():
                # 使用不稳定特征模型进行验证
                val_logits = unstable_model(target_val_data, target_val_domains)
                val_loss = F.cross_entropy(val_logits, target_val_labels)
                val_acc = accuracy(val_logits, target_val_labels)[0]
                val_accs.update(val_acc.item(), target_val_data.size(0))

            unstable_model.train()


            progress.display(i)

            # 记录验证的损失和准确率
            wandb.log({

                "Unstable Model Fine-tuning Loss": loss_cls.item(),
                "Unstable Model Fine-tuning Accuracy": acc.item(),
                "Unstable Model Fine-tuning Reconstruction Loss": loss_recon.item(),
                "Unstable Model Fine-tuning VAE Loss": loss_vae.item(),
                "Unstable Model Fine-tuning Entropy Loss": loss_ent.item(),
                "Unstable Model Fine-tuning Train KL": loss_kl.item(),
                "Unstable Model Fine-tuning Validation Accuracy": val_acc.item(),
                "Unstable Model Fine-tuning Validation Loss": val_loss.item(),
            })
