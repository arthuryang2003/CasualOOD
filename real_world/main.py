import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

import utils
from common.modules.networks import iVAE
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=(args.n_domains-1)*args.train_batch_size,
                                     num_workers=args.workers, drop_last=True,
                                     #sampler=_make_balanced_sampler(train_source_dataset.domain_ids)
                                     shuffle=True,
                                     )
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True,
                                     )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    classifier = iVAE(args, backbone_net=backbone).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    print(optimizer.param_groups[0]['lr'], ' *** lr')
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(optimizer.param_groups[0]['lr'], ' *** lr')
    test_logger = '%s/test.txt' % (args.log)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return
    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    total_iter = 0
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr(), optimizer.param_groups[0]['lr'])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer,
              lr_scheduler, epoch, args, total_iter, backbone)
        total_iter += args.iters_per_epoch
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        print(' * Val Acc@1 %.3f' % (acc1))
        wandb.log({"Val Acc": acc1})
        if args.data.lower() == "domainnet":
            acc1 = utils.validate(test_loader, classifier, args, device)
        wandb.log({"Test Acc": acc1})
        message = '(epoch %d): Test Acc@1 %.3f' % (epoch+1, acc1)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        wandb.run.summary["best_accuracy"] = best_acc1

    print("best_acc1 = {:3.1f}".format(best_acc1))
    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("Final Best test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model,  optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, total_iter, backbone):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')# 重建损失
    vae_losses = AverageMeter('VAE', ':4.2f') # VAE损失
    kl_losses = AverageMeter('KL', ':4.2f')# KL散度损失
    cls_losses = AverageMeter('Cls', ':4.2f') # 分类损失
    ent_losses = AverageMeter('Ent', ':4.2f')# 熵损失
    cls_accs = AverageMeter('Cls Acc', ':3.1f')# 分类准确率
    val_accs = AverageMeter('Val Acc', ':3.1f') # 验证准确率
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses, recon_losses, kl_losses, cls_accs, val_accs],
        prefix="Epoch: [{}]".format(epoch)
    )
    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(), torch.eye(args.z_dim).cuda())

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        # 从源域和目标域中获取一批数据
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
        for id in range(args.n_domains):# 遍历每个域
            domain_id = id
            is_target = domain_id == args.n_domains-1# 判断是否为目标域
            """
            if id == 0:
                index = (d_all != target_domain_id)
            else:
                index = (d_all == target_domain_id)
            """
            index = d_all == id# 获取当前域的样本
            label_dom = label_all[index] if not is_target else None
            img_dom = img_all[index]
            d_dom = d_all[index]

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
            loss_kl = args.beta * (kl - C).abs()# 计算KL损失

            if not is_target:  # only source
                losses_cls.append(F.cross_entropy(logit, label_dom))
                y_s.append(logit)
                labels_s.append(label_dom)
            else:
                y_t = logit# 如果是目标域，保存logit以便计算目标域的损失

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

        # source classification
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
        loss = mean_loss_cls  \
            + args.lambda_vae * mean_loss_vae  \
            + args.lambda_ent * loss_ent
        
        # 合并源域标签和预测结果，计算分类准确率
        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]
        cls_losses.update(mean_loss_cls.item(), y_s.size(0)) # 更新分类损失
        recon_losses.update(mean_loss_recon.item(), x_all.size(0))# 更新重建损失
        cls_accs.update(cls_acc.item(), y_s.size(0))# 更新分类准确率
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))# 更新VAE损失
        ent_losses.update(loss_ent.item(), y_t.size(0))# 更新熵损失
        kl_losses.update(mean_loss_kl.item(), x_all.size(0))# 更新KL损失
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()# 更新学习率

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            # not used in training
            model.eval()
            img_t = img_all[d_all==args.n_domains-1]
            labels_t = label_all[d_all==args.n_domains-1]
            with torch.no_grad():
                y = model(img_t, d_all[d_all==args.n_domains-1])
                cls_t_acc = accuracy(y, labels_t)[0]
                val_accs.update(cls_t_acc.item(), img_t.size(0))
            model.train()

            progress.display(i)

            wandb.log({
                "Train Target Acc": cls_t_acc.item(),
                "Train Source Acc": cls_acc.item(),
                "Train Source Cls Loss": mean_loss_cls.item(),
                "Train Reconstruction Loss": mean_loss_recon.item(),
                "Train VAE Loss": mean_loss_vae.item(),
                "Entropy Loss": loss_ent.item(),
                "Train KL": mean_loss_kl.item(),
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # 数据集参数

    # 数据集根路径，默认值为'../da_datasets/domainnet'
    parser.add_argument('--root', type=str, default='../da_datasets/domainnet',
                        help='root path of dataset')   
    # 数据集名称，默认为'DomainNet'，用户可选择已有的所有数据集
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    # 源域名称，多个源域用逗号分隔，默认为'i,p,q,r,s'
    parser.add_argument('-s', '--source', help='source domain(s)', default='i,p,q,r,s')
    # 目标域名称，默认为'c'
    parser.add_argument('-t', '--target', help='target domain(s)', default='c')
    # 训练时的图片缩放模式，默认为'default'
    parser.add_argument('--train-resizing', type=str, default='default')
    # 验证集图片缩放模式，默认为'default'
    parser.add_argument('--val-resizing', type=str, default='default') 
    # 图片缩放后的尺寸，默认为224
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing') 
    # 是否在训练时进行随机水平翻转，默认为False
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    # 图片归一化时的均值，默认值为ImageNet预训练模型的均值
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    # 图片归一化时的标准差，默认值为ImageNet预训练模型的标准差
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # 模型参数
    # 选择模型的架构，默认为'resnet101'
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    # 颈部（bottleneck）层的维度，默认为2048
    parser.add_argument('--bottleneck-dim', default=2048, type=int,
                        help='Dimension of bottleneck')
    # 是否在特征提取后去掉池化层
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
     # 是否从头开始训练模型（不使用预训练权重）
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # 损失函数中源任务和目标任务的权衡超参数
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    
    # 训练参数
    # 批大小，默认为32
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    # 初始学习率，默认为0.01
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    # 学习率调度器的gamma参数，用于调整学习率
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    # 学习率衰减系数
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
     # 动量因子，默认为0.9
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # 权重衰减（L2正则化系数），默认为5e-4
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    # 数据加载的工作线程数，默认为2
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    # 训练的总周期数，默认为40
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    # 每个训练周期的迭代次数，默认为2500
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    # 打印频率，每100次打印一次
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    # 评估频率，每100次评估一次
    parser.add_argument('-e', '--eval-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    
    # 随机种子和评估选项
    # 随机种子，用于初始化训练，默认None
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # 是否输出每个类别的准确率
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    # 日志和检查点设置
    # 日志文件保存路径，默认为'logs'
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    # 设置训练阶段（train/test/analysis），默认为'train'
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    
    # 模型超参数
    # 隐空间维度，默认为64
    parser.add_argument('--z_dim', type=int, default=64, metavar='N')
    # 训练时的批大小，默认为16
    parser.add_argument('--train_batch_size', default=16, type=int)
    # 风格空间维度，默认为16
    parser.add_argument('--s_dim', type=int, default=16, metavar='N')
    # 隐藏层维度，默认为4096
    parser.add_argument('--hidden_dim', type=int, default=4096, metavar='N')
    # VAE损失中的beta系数，默认为1
    parser.add_argument('--beta', type=float, default=1., metavar='N')
    # 实验名称
    parser.add_argument('--name', type=str, default='', metavar='N')
    # 流网络类型，默认为'ddsf'
    parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
    # 流网络的维度
    parser.add_argument('--flow_dim', type=int, default=16, metavar='N')
    # 流网络的层数
    parser.add_argument('--flow_nlayer', type=int, default=2, metavar='N')
    # 初始化值
    parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
    # 初始化值
    parser.add_argument('--flow_bound', type=int, default=5, metavar='N')
    # 流网络的bin数
    parser.add_argument('--flow_bins', type=int, default=8, metavar='N')
    # 流网络的顺序
    parser.add_argument('--flow_order', type=str, default='linear', metavar='N')
    # 网络类型，默认为'dirt'
    parser.add_argument('--net', type=str, default='dirt', metavar='N')
    # 流网络的数量
    parser.add_argument('--n_flow', type=int, default=2, metavar='N')
    # VAE损失的权重，默认为1e-3
    parser.add_argument('--lambda_vae', type=float, default=1e-3, metavar='N')
    # 分类损失的权重，默认为1
    parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
    # 熵损失的权重，默认为0.1
    parser.add_argument('--lambda_ent', type=float, default=0.1, metavar='N')
    # 熵阈值
    parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
    # 最大C值
    parser.add_argument('--C_max', type=float, default=20., metavar='N')
    # 停止迭代的C值
    parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')    

    # 解析命令行参数
    args = parser.parse_args()

    # 构建模型ID，并设置日志路径
    model_id = f"{args.data}_{args.target}/{args.name}-lam_vae_{args.lambda_vae}-lambda_{args.lambda_ent}-D_{args.s_dim}/{args.z_dim}"
    args.log = os.path.join(args.log, model_id)

    # 分割源域和目标域
    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    args.n_domains = len(args.source) + len(args.target)

    # 设置输入维度和隐藏层维度
    args.input_dim = 2048
    if 'pacs' in args.root:
        args.input_dim = 512
        args.hidden_dim = 256
    args.norm_id = args.n_domains - 1
    args.c_dim = args.z_dim - args.s_dim

    wandb.init(
        project="domain_adaptation_partial_identifiability",
        group=args.name,
    )
    wandb.config.update(args)

    main(args)
