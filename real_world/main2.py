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

from extract_features import extract_features
from pseudo_label import combined_inference
from train import  train_stable,train_unstable,train_VAE,train_decoupler

import utils
from common.modules.networks import iVAE,Classifier,Decoupler
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['WANDB_MODE'] = 'disabled'

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

    # 数据加载和预处理
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, val_dataset, test_dataset, args.num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=(args.n_domains-1)*args.train_batch_size,
                                     num_workers=args.workers, drop_last=True,
                                     #sampler=_make_balanced_sampler(train_source_dataset.domain_ids)
                                     shuffle=True,
                                     )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    val_iter = ForeverDataIterator(val_loader)

    # 通过目标数据集计算类别数量
    num_classes = args.num_classes


    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    decoupler_model=Decoupler(args, backbone_net=backbone).to(device)
    # define optimizer and lr scheduler
    decoupler_optimizer = SGD(decoupler_model.get_parameters(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    print(decoupler_optimizer.param_groups[0]['lr'], ' *** lr')
    lr_scheduler = LambdaLR(decoupler_optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(decoupler_optimizer.param_groups[0]['lr'], ' *** lr')

    # 初始化稳定分类器和不稳定分类器
    stable_classifier = Classifier(args, input_size=args.c_dim).to(device)  # 稳定模型分类器，输入为c_dim
    unstable_classifier = Classifier(args, input_size=args.s_dim).to(device)  # 不稳定模型分类器，输入为s_dim

    # 稳定分类器的优化器和学习率调度器
    stable_optimizer = SGD(stable_classifier.parameters(),
                           lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    print(stable_optimizer.param_groups[0]['lr'], ' *** lr for stable classifier')
    stable_lr_scheduler = LambdaLR(stable_optimizer,
                                   lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(stable_optimizer.param_groups[0]['lr'], ' *** lr for stable classifier after scheduler')

    # 不稳定分类器的优化器和学习率调度器
    unstable_optimizer = SGD(unstable_classifier.parameters(),
                             lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    print(unstable_optimizer.param_groups[0]['lr'], ' *** lr for unstable classifier')
    unstable_lr_scheduler = LambdaLR(unstable_optimizer,
                                     lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(unstable_optimizer.param_groups[0]['lr'], ' *** lr for unstable classifier after scheduler')



    test_logger = '%s/test.txt' % (args.log)

    if args.phase != 'train':
        stable_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_stable')))
        unstable_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_unstable')))
        decoupler_model.load_state_dict(torch.load(logger.get_checkpoint_path('best_decoupler')))

    if args.phase == 'test':
        acc1, _ = utils.validate(test_loader, decoupler_model, args, 0, device)
        print("decoupler Best test_acc1 = {:3.2f}".format(acc1))

        # 调用 validate_classifier 函数进行验证
        acc2, acc3 = utils.validate_classifier(decoupler_model, stable_classifier, unstable_classifier, test_loader,
                                               args, device)
        print("Final Best Stable Classifier test_acc2 = {:3.2f}".format(acc2))
        print("Final Best Unstable Classifier test_acc3 = {:3.2f}".format(acc3))

        combined_acc = combined_inference(decoupler_model, stable_classifier, unstable_classifier, test_loader, num_classes)

        print("Final Combined Model Test Acc = {:3.2f}".format(combined_acc))
        return

    # start training
    total_iter = 0
    best_acc1=0.
    for epoch in range(args.decoupler_epochs):
        print("lr:", lr_scheduler.get_last_lr(), decoupler_optimizer.param_groups[0]['lr'])
        # train for one epoch
        train_decoupler(train_source_iter, val_iter, decoupler_model, decoupler_optimizer,
              lr_scheduler, epoch, args, total_iter, backbone)

        # evaluate on validation set
        acc1 = utils.validate_decoupler(val_loader, decoupler_model, args, device)
        print("acc1 = {:3.4f}".format(acc1))
        wandb.log({"decoupler Val Acc": acc1})
        message = '(epoch %d): decoupler Val Acc %.3f' % (epoch+1, acc1)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(decoupler_model.state_dict(), logger.get_checkpoint_path('latest_decoupler'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest_decoupler'), logger.get_checkpoint_path('best_decoupler'))

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.4f}".format(best_acc1))
    # evaluate on test set
    decoupler_model.load_state_dict(torch.load(logger.get_checkpoint_path('best_decoupler')))
    acc1 = utils.validate_decoupler(test_loader, decoupler_model, args,device)
    print("decoupler Best test_acc1 = {:3.2f}".format(acc1))

    for param in decoupler_model.parameters():
        param.requires_grad = False

    best_acc2 = 0.
    total_iter = 0
    for epoch in range(args.unstable_epochs):

        # 训练不稳定特征模型
        train_stable(train_source_iter, val_iter,decoupler_model, stable_classifier, stable_optimizer, stable_lr_scheduler, epoch, args, total_iter)


        # 调用 validate_classifier 函数进行验证
        acc2, _ = utils.validate_classifier(decoupler_model, stable_classifier, unstable_classifier, val_loader,
                                                       args, device)

        wandb.log({"Stable Classifier Val Acc": acc2})

        message = '(epoch %d): Stable Classifier Val Acc %.3f' % (epoch + 1, acc2)
        print(message)
        record = open(test_logger, 'a')
        record.write(message + '\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(stable_classifier.state_dict(), logger.get_checkpoint_path('latest_stable'))

        if acc2 > best_acc2:
            shutil.copy(logger.get_checkpoint_path('latest_stable'), logger.get_checkpoint_path('best_stable'))
        best_acc2= max(acc2, best_acc2)

        wandb.run.summary["best_stable_accuracy"] = best_acc2

    print("best_acc2 = {:3.2f}".format(best_acc2))

    # 加载最佳稳定模型并评估
    stable_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_stable')))
    # 调用 validate_classifier 函数进行验证
    acc2, _ = utils.validate_classifier(decoupler_model, stable_classifier, unstable_classifier, test_loader,
                                        args, device)
    print("Best Stable Classifier test_acc2 = {:3.2f}".format(acc2))

    for param in stable_classifier.parameters():
        param.requires_grad = False

    best_acc3 = 0.
    total_iter = 0
    for epoch in range(args.unstable_epochs):

        # 训练不稳定特征模型
        train_unstable(train_source_iter, val_iter,decoupler_model, stable_classifier,unstable_classifier, unstable_optimizer, unstable_lr_scheduler, epoch, args, total_iter)


        # 调用 validate_classifier 函数进行验证
        _, acc3 = utils.validate_classifier(decoupler_model, stable_classifier, unstable_classifier, val_loader,
                                                       args, device)

        wandb.log({"Unstable Classifier Val Acc": acc3})

        message = '(epoch %d): Unstable Classifier Val Acc %.3f' % (epoch + 1, acc3)
        print(message)
        record = open(test_logger, 'a')
        record.write(message + '\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(unstable_classifier.state_dict(), logger.get_checkpoint_path('latest_unstable'))

        if acc3 > best_acc3:
            shutil.copy(logger.get_checkpoint_path('latest_unstable'), logger.get_checkpoint_path('best_unstable'))
        best_acc3= max(acc3, best_acc3)

        wandb.run.summary["best_accuracy"] = best_acc3

    print("best_acc3 = {:3.2f}".format(best_acc3))

    # 加载最佳稳定模型并评估
    unstable_classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best_unstable')))

    # 调用 validate_classifier 函数进行验证
    acc2,acc3 = utils.validate_classifier(decoupler_model, stable_classifier, unstable_classifier, test_loader,
                                        args, device)
    print("Final Best Stable Classifier test_acc2 = {:3.2f}".format(acc2))
    print("Final Best Unstable Classifier test_acc3 = {:3.2f}".format(acc3))

    for param in unstable_classifier.parameters():
        param.requires_grad = False

    # 评估组合模型

    combined_acc= combined_inference(decoupler_model,stable_classifier, unstable_classifier, test_loader, num_classes)

    # 分别打印准确率
    print("Final Combined Model Test Acc = {:3.2f}".format(combined_acc))

    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # 数据集参数
    parser.add_argument('--root', type=str, default='../../da_datasets/pacs-vae',
                        help='root path of dataset')   
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: PACS)')
    parser.add_argument('-s', '--source', help='source domain(s)', default='C,P,A')
    parser.add_argument('-t', '--target', help='target domain(s)', default='S')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default') 
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing') 
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')


    # 模型参数
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=2048, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # 训练参数
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N', help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-e', '--eval-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')

    # 随机种子和评估选项
    parser.add_argument('--seed', default=5, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # 模型超参数
    parser.add_argument('--z_dim', type=int, default=64, metavar='N')
    # parser.add_argument('--c_dim', type=int, default=32, metavar='N')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--s_dim', type=int, default=32, metavar='N')
    parser.add_argument('--hidden_dim', type=int, default=4096, metavar='N')
    parser.add_argument('--beta', type=float, default=1., metavar='N')
    parser.add_argument('--name', type=str, default='ours_PACS_KL_dim_8', metavar='N')
    parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
    parser.add_argument('--flow_dim', type=int, default=16, metavar='N')
    parser.add_argument('--flow_nlayer', type=int, default=2, metavar='N')
    parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
    parser.add_argument('--flow_bound', type=int, default=5, metavar='N')
    parser.add_argument('--flow_bins', type=int, default=8, metavar='N')
    parser.add_argument('--flow_order', type=str, default='linear', metavar='N')
    parser.add_argument('--net', type=str, default='dirt', metavar='N')
    parser.add_argument('--n_flow', type=int, default=2, metavar='N')
    parser.add_argument('--lambda_vae', type=float, default=5e-5, metavar='N')
    parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
    parser.add_argument('--lambda_ent', type=float, default=0.1, metavar='N')
    parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
    parser.add_argument('--C_max', type=float, default=15., metavar='N')
    parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')
    parser.add_argument('--decouple_alpha', type=float, default=1., metavar='N')
    parser.add_argument('--decouple_beta', type=float, default=1., metavar='N')
    parser.add_argument('--stable_epochs', type=int, default=1, metavar='N',
                        help='number of stable epochs to run')
    parser.add_argument('--unstable_epochs', type=int, default=1, metavar='N',
                        help='number of unstable epochs to run')
    parser.add_argument('--decoupler_epochs', type=int, default=1, metavar='N',
                        help='number of decoupler epochs to run')
    parser.add_argument('--target_split_ratio', type=float, default=0.8, metavar='N',
                        help='ratio of target domain data used for training set (rest for testing)')

    args = parser.parse_args()
    model_id = f"{args.data}_{args.target}/{args.name}"
    args.log = os.path.join(args.log, model_id)

    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    args.n_domains = len(args.source) + len(args.target)
    args.input_dim = 2048
    if 'pacs-vae' in args.root:
        args.input_dim = 512
        args.hidden_dim = 256
    args.norm_id = args.n_domains - 1
    args.c_dim = args.z_dim - args.s_dim

    wandb.init(
        project="CasualOOD",
        group=args.name,
    )
    wandb.config.update(args)

    main(args)

