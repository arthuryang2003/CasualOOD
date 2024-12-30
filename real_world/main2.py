import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from common.modules.networks import iVAE
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义伪标签生成函数
def generate_pseudo_labels(test_loader, stable_model, device):
    pseudo_labels = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_S = X_batch[:, :args.z_dim - args.style_dim].to(device)
            Y_hat = stable_model(X_S)  # 使用稳定特征生成伪标签
            pseudo_labels.append((X_batch, torch.argmax(Y_hat, dim=1)))  # 多类情况下使用argmax生成伪标签
    return pseudo_labels

# 定义伪标签微调函数
def pseudo_label_finetune(test_loader, unstable_model, stable_model, optimizer, criterion, device):
    unstable_model.train()
    pseudo_labels = generate_pseudo_labels(test_loader, stable_model, device)
    for X_batch, Y_hat in pseudo_labels:
        X_U = X_batch[:, args.z_dim - args.style_dim:].to(device)  # 使用可变特征 X_U
        Y_hat = Y_hat.to(device)
        optimizer.zero_grad()
        outputs = unstable_model(X_U)
        loss = criterion(outputs, Y_hat)  # 使用伪标签进行损失计算
        loss.backward()
        optimizer.step()

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')

    cudnn.benchmark = True

    # 数据加载和预处理
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=(args.n_domains-1)*args.train_batch_size,
                                     num_workers=args.workers, drop_last=True, shuffle=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # 创建稳定模型和不稳定模型
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    stable_model = iVAE(args, backbone_net=backbone).to(device)  # 用于提取不变特征
    unstable_model = iVAE(args, backbone_net=backbone).to(device)  # 用于提取可变特征

    # 定义优化器和学习率调度器
    optimizer = SGD(unstable_model.get_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # 载入最佳检查点
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        unstable_model.load_state_dict(checkpoint)

    # 测试阶段
    if args.phase == 'test':
        acc1 = utils.validate(test_loader, unstable_model, args, device)
        print(f"初始测试准确率: {acc1:.4f}")

        # 添加伪标签微调
        print("开始伪标签微调...")
        pseudo_label_finetune(
            test_loader=test_loader,
            unstable_model=unstable_model,
            stable_model=stable_model,  # 使用不同的稳定模型
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),  # 多分类问题使用交叉熵损失
            device=device
        )

        acc2 = utils.validate(test_loader, unstable_model, args, device)
        print(f"伪标签微调后准确率: {acc2:.4f}")
        return

    # 训练阶段
    if args.phase == 'train':
        best_acc1 = 0.
        total_iter = 0
        for epoch in range(args.epochs):
            print("lr:", lr_scheduler.get_last_lr(), optimizer.param_groups[0]['lr'])

            # 训练
            train_one_epoch(train_source_loader, train_target_loader, stable_model, unstable_model, optimizer, epoch, args, total_iter, backbone)
            total_iter += args.iters_per_epoch

            # 验证
            acc1 = utils.validate(val_loader, unstable_model, args, device)
            print(' * 验证准确率 %.3f' % (acc1))

            # 保存检查点
            torch.save(unstable_model.state_dict(), logger.get_checkpoint_path('latest'))
            if acc1 > best_acc1:
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(acc1, best_acc1)

        print("最佳验证准确率 = {:3.1f}".format(best_acc1))
        return

def train_one_epoch(train_source_loader, train_target_loader, stable_model, unstable_model, optimizer, epoch, args, total_iter, backbone):
    stable_model.train()
    unstable_model.train()
    for (X_s, Y_s), (X_t, Y_t) in zip(train_source_loader, train_target_loader):
        # 提取不变特征（从源域）
        X_s, Y_s = X_s.to(device), Y_s.to(device)
        z_s, _, _, _, _, _ = stable_model.encode(X_s, Y_s)
        
        # 提取可变特征（从目标域）
        X_t, Y_t = X_t.to(device), Y_t.to(device)
        _, z_t, _, _, _, _ = unstable_model.encode(X_t, Y_t)
        
        # 用不变特征训练稳定模型
        stable_model_loss = F.cross_entropy(stable_model(z_s), Y_s)
        
        # 用可变特征训练不稳定模型
        unstable_model_loss = F.cross_entropy(unstable_model(z_t), Y_t)
        
        # 总损失
        loss = stable_model_loss + unstable_model_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    parser.add_argument('--root', type=str, default='../da_datasets/office-home', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome', choices=utils.get_dataset_names(), help='dataset: ' + ' | '.join(utils.get_dataset_names()) + ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', default='Ar,Cl,Pr')
    parser.add_argument('-t', '--target', help='target domain(s)', default='Rw')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224, help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=utils.get_model_names(), help='backbone architecture: ' + ' | '.join(utils.get_model_names()) + ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=2048, type=int, help='Dimension of bottleneck')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float, help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2000, type=int, help='Number of iterations per epoch')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'analysis'], help="When phase is 'test', only test the model. When phase is 'analysis', only analysis the model.")
    parser.add_argument('--z_dim', type=int, default=128, metavar='N', help='latent feature dimension')
    parser.add_argument('--style_dim', type=int, default=64, metavar='N', help='style feature dimension')
    parser.add_argument('--lambda_vae', type=float, default=1e-4, metavar='N', help='VAE loss coefficient')
    args = parser.parse_args()
    main(args)
