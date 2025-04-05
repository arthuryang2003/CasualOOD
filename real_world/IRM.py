import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os


import datasets.datasets as datasets
import datasets.misc as misc

from datasets import datasets,misc
from datasets.fast_data_loader import InfiniteDataLoader, FastDataLoader
from common.modules.networks import CasualOOD, IRMNet

sys.path.append('.')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from pseudo_label import combined_inference
from train import CasualOOD_train, CasualOOD_finetune, CasualOOD_train1, IRM_train

import utils

from utils import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['WANDB_MODE'] = 'disabled'


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.target, args)
    else:
        raise NotImplementedError

    train_source_dataset, val_source_dataset, val_target_dataset, test_dataset = [], [], [], []
    for i, env in enumerate(dataset):
        if dataset.ENVIRONMENTS[i] in args.source:
            val_s, train_s = misc.split_dataset(env, int(len(env) * args.source_split_ratio), misc.seed_hash(args.seed, i))
            train_source_dataset.append(train_s)
            val_source_dataset.append(val_s)
        elif dataset.ENVIRONMENTS[i] in args.target:
            val_t, train_t = misc.split_dataset(env, int(len(env) * args.target_split_ratio), misc.seed_hash(args.seed, i))
            val_target_dataset.append(val_t)
            test_dataset.append(train_t)
            test_dataset.append(val_t)

    train_source_loader = [InfiniteDataLoader(env, None, args.batch_size, args.workers) for env in train_source_dataset]
    val_source_loader = [FastDataLoader(env, args.batch_size, args.workers) for env in val_source_dataset]
    test_loader = [FastDataLoader(env, args.batch_size, args.workers) for env in test_dataset]

    train_source_iter = ForeverDataIterator(train_source_loader)
    val_source_iter = ForeverDataIterator(val_source_loader)

    args.num_classes = dataset.num_classes
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    model = IRMNet(args, backbone_net=backbone).to(device)

    optimizer = SGD(model.get_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    test_logger = '%s/test.txt' % (args.log)

    if args.phase == 'test':
        model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train')))
        acc = utils.validate(test_loader, model, args, device)
        print("Test Accuracy = {:.2f}".format(acc))
        return

    best_acc = 0.
    total_iter = 0
    for epoch in range(args.train_epochs):
        print("lr:", lr_scheduler.get_last_lr())
        IRM_train(train_source_iter, val_source_iter, model, optimizer, lr_scheduler, epoch, args, total_iter)

        acc = utils.validate(val_source_loader, model, args, device)
        print("Val Acc = {:.4f}".format(acc))
        wandb.log({"IRM Val Acc": acc})
        with open(test_logger, 'a') as f:
            f.write(f"(epoch {epoch+1}): IRM Val Acc {acc:.3f}\n")

        torch.save(model.state_dict(), logger.get_checkpoint_path('latest_model'))
        if acc > best_acc:
            shutil.copy(logger.get_checkpoint_path('latest_model'), logger.get_checkpoint_path('best_model_train'))
        best_acc = max(best_acc, acc)

    print("Best Val Accuracy: {:.2f}".format(best_acc))
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train')))
    test_acc = utils.validate(test_loader, model, args, device)
    print("Test Accuracy = {:.2f}".format(test_acc))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CasualOOD')
    # 数据集参数
    # parser.add_argument('--root', type=str, default='../../da_datasets/pacs',
    #                     help='root path of dataset')
    # parser.add_argument('-d', '--data', metavar='DATA', default='PACS', choices=utils.get_dataset_names(),
    #                     help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
    #                          ' (default: PACS)')

    parser.add_argument('--dataset', type=str, default="ColoredMNIST")
    parser.add_argument('--data_dir', type=str,default='./data')

    parser.add_argument('-s', '--source', help='source domain(s)', default='C,P,A')
    parser.add_argument('-t', '--target', help='target domain(s)', default='S')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--data_augmentation', type=bool, default=True,
                        help='apply data augmentation')
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
    parser.add_argument('-i', '--iters-per-epoch', default=100, type=int,
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
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N')
    parser.add_argument('--name', type=str, default='test', metavar='N')

    parser.add_argument('--decouple_alpha', type=float, default=1., metavar='N')
    parser.add_argument('--decouple_beta', type=float, default=1., metavar='N')

    parser.add_argument('--train_epochs', type=int, default=1, metavar='N',
                        help='number of train epochs to run')
    parser.add_argument('--finetune_epochs', type=int, default=1, metavar='N',
                        help='number of finetune epochs to run')
    parser.add_argument('--source_split_ratio', type=float, default=0.8, metavar='N',
                        help='ratio of source domain data used for training set (rest for testing)')

    parser.add_argument('--target_split_ratio', type=float, default=0.2, metavar='N',
                        help='ratio of target domain data used for training set (rest for testing)')

    args = parser.parse_args()
    model_id = f"{args.dataset}_{args.target}/{args.name}"
    args.log = os.path.join(args.log, model_id)

    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    args.n_domains = len(args.source) + len(args.target)

    args.norm_id = args.n_domains - 1
    # args.c_dim = args.z_dim - args.s_dim

    wandb.init(
        project="CasualOOD",
        group=args.name,
    )
    wandb.config.update(args)

    main(args)

