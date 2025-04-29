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
from common.modules.networks import CasualOOD

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
from train import CasualOOD_train, CasualOOD_finetune, CasualOOD_train1, CasualOOD_train2

import utils

from utils import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['WANDB_MODE'] = 'disabled'

#两阶段
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

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.target, args)
    else:
        raise NotImplementedError

    train_source_dataset = []
    train_target_dataset = []
    val_source_dataset = []
    val_target_dataset = []
    test_dataset = []
    for env_i, env in enumerate(dataset):

        if dataset.ENVIRONMENTS[env_i]  in args.source:
            train_s,val_s = misc.split_dataset(env,
                                          int(len(env) * args.source_split_ratio),
                                          misc.seed_hash(args.seed, env_i))
            train_source_dataset.append(train_s)
            val_source_dataset.append(val_s)

        elif dataset.ENVIRONMENTS[env_i]  in args.target:
            train_t,val_t = misc.split_dataset(env,
                                                int(len(env) * args.target_split_ratio),
                                                misc.seed_hash(args.seed, env_i))
            train_target_dataset.append(train_t)
            val_target_dataset.append(val_t)

            test_dataset.append(train_t)
            test_dataset.append(val_t)


    train_source_loader = [InfiniteDataLoader(
        dataset=env,  # 这里不需要列表展开
        weights=None,  # 如果没有特定的 sample 权重，可以设为 None
        batch_size=args.batch_size,
        num_workers=args.workers
    ) for i, env in enumerate(train_source_dataset)]

    train_target_loader = [InfiniteDataLoader(
        dataset=env,  # 这里不需要列表展开
        weights=None,  # 如果没有特定的 sample 权重，可以设为 None
        batch_size=args.batch_size,
        num_workers=args.workers
    )for i, env in enumerate(train_target_dataset)]

    val_source_loader = [FastDataLoader(
        dataset=env,  # 这里不需要列表展开
        batch_size=args.batch_size,
        num_workers=args.workers
    ) for i, env in enumerate(val_source_dataset)]

    val_target_loader = [FastDataLoader(
        dataset=env,  # 这里不需要列表展开
        batch_size=args.batch_size,
        num_workers=args.workers
    )for i, env in enumerate(val_target_dataset)]

    test_loader = [FastDataLoader(
        dataset=env,  # 这里不需要列表展开
        batch_size=args.batch_size,
        num_workers=args.workers
    )for i, env in enumerate(test_dataset)]

    train_source_iter = ForeverDataIterator(train_source_loader)
    val_source_iter = ForeverDataIterator(val_source_loader)
    val_target_iter = ForeverDataIterator(val_target_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    args.num_classes=dataset.num_classes
    # 通过目标数据集计算类别数量
    num_classes = dataset.num_classes

    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    model=CasualOOD(args, backbone_net=backbone).to(device)
    # define optimizer and lr scheduler
    phase1_optimizer = SGD(model.get_parameters_train_phase1(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    print(phase1_optimizer.param_groups[0]['lr'], ' *** lr')
    phase1_lr_scheduler = LambdaLR(phase1_optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(phase1_optimizer.param_groups[0]['lr'], ' *** lr')

    # define optimizer and lr scheduler
    phase2_optimizer = SGD(model.get_parameters_train_phase2(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    print(phase2_optimizer.param_groups[0]['lr'], ' *** lr')
    phase2_lr_scheduler = LambdaLR(phase2_optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(phase2_optimizer.param_groups[0]['lr'], ' *** lr')

    # define finetune optimizer and lr scheduler
    finetune_optimizer = SGD(model.get_finetune_parameters(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    print(finetune_optimizer.param_groups[0]['lr'], ' *** lr')
    finetune_lr_scheduler = LambdaLR(finetune_optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(finetune_optimizer.param_groups[0]['lr'], ' *** lr')


    test_logger = '%s/test.txt' % (args.log)
    print(test_logger)

    if args.phase != 'train':
        model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train1')))
    if args.phase == 'analysis':
        model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_test')))
        model.eval()
        print("==> Running GradCAM analysis on disentangled features...")

        gradcam_dir = os.path.join(args.log, "gradcam_disentangled")
        os.makedirs(gradcam_dir, exist_ok=True)

        def find_last_conv(model):
            for layer in reversed(list(model.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
            raise ValueError("No Conv2d layer found in model")

        target_layer = find_last_conv(model.backbone_net)
        cam = GradCAM(model, target_layer=target_layer)

        def cam_to_pil(cam_np):
            if isinstance(cam_np, torch.Tensor):
                cam_tensor = cam_np.detach().cpu()
            else:
                cam_tensor = torch.from_numpy(cam_np)
            if cam_tensor.ndim == 2:
                cam_tensor = cam_tensor.unsqueeze(0)
            elif cam_tensor.ndim == 3 and cam_tensor.shape[0] != 1:
                raise ValueError(f"Expected CAM shape [1, H, W] or [H, W], but got {cam_tensor.shape}")
            return to_pil_image(cam_tensor, mode='F')

        def auto_color_map(img_tensor):
            if img_tensor.shape[0] == 3:
                return img_tensor
            elif img_tensor.shape[0] == 2:
                r, g = img_tensor[0:1], img_tensor[1:2]
                b = torch.zeros_like(r)
                return torch.cat([r, g, b], dim=0)
            elif img_tensor.shape[0] == 1:
                return img_tensor.repeat(3, 1, 1)
            else:
                raise ValueError(f"Unsupported image shape: {img_tensor.shape}")

        def run_cam(img, class_idx, logit_fn, retain=False):
            img = img.clone().detach().to(device).requires_grad_(True)
            z_u, z_s, *_ = model.encode(img)
            logits = logit_fn(z_u, z_s)
            return cam(class_idx, scores=logits, retain_graph=retain)

        selected_samples = {0: [], 1: []}
        max_per_class = 5

        test_iter = ForeverDataIterator(test_loader)
        while len(selected_samples[0]) < max_per_class or len(selected_samples[1]) < max_per_class:
            data, labels = next(test_iter)[0]
            data, labels = data.to(device), labels.to(device)

            # 保证梯度追踪
            data.requires_grad_()
            z_u, z_s, *_ = model.encode(data)
            logit_u = model.predict_u(z_u)
            pred_u = logit_u.argmax(dim=1)

            for i in range(data.size(0)):
                label = labels[i].item()
                pred = pred_u[i].item()
                if label in [0, 1] and label == pred and len(selected_samples[label]) < max_per_class:
                    selected_samples[label].append((data[i].unsqueeze(0), label))

        # 对筛选结果进行GradCAM分析
        for label_class in [0, 1]:
            for i, (img, label) in enumerate(selected_samples[label_class]):
                img = img.to(device).requires_grad_()

                # 视觉展示准备
                img_vis = auto_color_map(img[0].detach().cpu() * 0.229 + 0.485)
                img_vis = torch.clamp(img_vis, 0, 1)

                with torch.enable_grad():
                    z_u, z_s, *_ = model.encode(img)
                    tilde_z_s = model.domain_influence(z_s)
                    drop_z_s=model.drop_spurious_features(z_s)
                    logit_u = model.predict_u(z_u)
                    logit_s = model.predict_s(z_s)
                    logit_tilde_s = model.predict_tilde_s(tilde_z_s)
                    logit_drop_s = model.predict_tilde_s(drop_z_s)
                    class_idx_u = logit_u.argmax(dim=1).item()
                    class_idx_s = logit_s.argmax(dim=1).item()
                    class_idx_t = logit_tilde_s.argmax(dim=1).item()
                    class_idx_drop = logit_drop_s.argmax(dim=1).item()

                    cam_map_u = cam(class_idx_u, scores=logit_u, retain_graph=True)
                    cam_map_s = cam(class_idx_s, scores=logit_s, retain_graph=True)
                    cam_map_tilde_s = cam(class_idx_t, scores=logit_tilde_s, retain_graph=True)
                    cam_map_drop_s = cam(class_idx_drop, scores=logit_drop_s, retain_graph=True)

                # 热力图叠加
                heatmap_u = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_u[0]), alpha=0.5)
                heatmap_s = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_s[0]), alpha=0.5)
                heatmap_tilde_s = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_tilde_s[0]), alpha=0.5)
                heatmap_diff = overlay_mask(to_pil_image(img_vis), cam_to_pil(cam_map_drop_s[0]), alpha=0.5)

                # 可视化保存
                fig, axs = plt.subplots(1, 5, figsize=(16, 4))
                axs[0].imshow(to_pil_image(img_vis))
                axs[0].set_title(f"Original({label})")
                axs[1].imshow(heatmap_u)
                axs[1].set_title(f"GradCAM: z_u({class_idx_u})")
                axs[2].imshow(heatmap_s)
                axs[2].set_title(f"GradCAM: z_s({class_idx_s})")
                axs[3].imshow(heatmap_tilde_s)
                axs[3].set_title(f"GradCAM: z_s'({class_idx_t})")
                axs[4].imshow(heatmap_diff)
                axs[4].set_title(f"GradCAM: drop_z_s({class_idx_drop})")

                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                save_path = os.path.join(gradcam_dir, f"selected_class{label_class}_{i}.png")
                plt.savefig(save_path)
                plt.close()

        mask_sigmoid = torch.sigmoid(model.mask).detach().cpu().numpy().squeeze()

        # 绘图：mask sigmoid 后的通道权重
        plt.figure(figsize=(12, 3))
        plt.bar(range(len(mask_sigmoid)), mask_sigmoid)
        plt.title("Mask Channel Weights after Sigmoid")
        plt.xlabel("Channel index")
        plt.ylabel("Gate value (sigmoid)")
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, "mask_weights.png"))
        plt.close()
        return
    if args.phase == 'test':
        model.set_requires_grad(False)

        # start test and finetune
        total_iter = 0
        best_acc2 = 0.
        for epoch in range(args.finetune_epochs):
            print("lr:", finetune_lr_scheduler.get_last_lr(), finetune_optimizer.param_groups[0]['lr'])
            # train for one epoch
            CasualOOD_finetune(train_target_iter, val_target_iter, model, finetune_optimizer,
                               finetune_lr_scheduler, epoch, args, total_iter, backbone)

            # evaluate on validation set
            acc2 = combined_inference(model, val_target_loader, num_classes)
            acc3 = utils.validate(val_target_loader, model, args, device)
            print("acc2 = {:3.4f}".format(acc2))
            print("acc3 = {:3.4f}".format(acc3))
            wandb.log({"Model Val Acc": acc2})
            message = '(epoch %d): Model Val Acc %.3f' % (epoch + 1, acc2)
            print(message)
            record = open(test_logger, 'a')
            record.write(message + '\n')
            record.close()

            # remember best acc@1 and save checkpoint
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest_model'))
            if acc2 > best_acc2:
                shutil.copy(logger.get_checkpoint_path('latest_model'), logger.get_checkpoint_path('best_model_test'))

            best_acc2 = max(acc2, best_acc2)

        print("best_acc2 = {:3.4f}".format(best_acc2))
        # evaluate on test set
        model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_test')))
        acc2 = combined_inference(model, test_loader, num_classes)
        acc3 = utils.validate(test_loader, model, args, device)
        print("acc3 = {:3.4f}".format(acc3))
        print("Test Phase Best test_acc = {:3.2f}".format(acc2))

        acc3 = utils.validate_ulogits(test_loader, model, args, device)
        print("base acc = {:3.4f}".format(acc3))

        logger.close()

        return



    model.set_requires_grad_phase1()
    # start training
    total_iter = 0
    best_acc1=0.
    for epoch in range(args.train_epochs):
        print("lr:", phase1_lr_scheduler.get_last_lr(), phase1_optimizer.param_groups[0]['lr'])
        # train for one epoch
        CasualOOD_train1(train_source_iter, val_source_iter, model, phase1_optimizer,
              phase1_lr_scheduler, epoch, args, total_iter, backbone)

        # evaluate on validation set
        acc1 = utils.validate_ulogits(val_source_loader, model, args, device)
        print("phase 1 acc1 = {:3.4f}".format(acc1))
        wandb.log({"Model phase 1 Val Acc": acc1})
        message = '(epoch %d): Model phase 1 Val Acc %.3f' % (epoch+1, acc1)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest_model'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest_model'), logger.get_checkpoint_path('best_model_train1'))

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.4f}".format(best_acc1))
    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train1')))
    acc1 = utils.validate_ulogits(test_loader, model, args,device)
    print("Train Phase 1 Best test_acc1 = {:3.2f}".format(acc1))

    acc3 = utils.validate_ulogits(test_loader, model, args, device)
    print("base acc = {:3.4f}".format(acc3))

    # model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train1')))


    model.set_requires_grad_phase2()
    # start training
    total_iter = 0
    best_acc1=0.
    for epoch in range(args.finetune_epochs):
        print("lr:", phase2_lr_scheduler.get_last_lr(), phase2_optimizer.param_groups[0]['lr'])
        # train for one epoch
        CasualOOD_train2(train_source_iter, val_source_iter, model, phase2_optimizer,
              phase2_lr_scheduler, epoch, args, total_iter, backbone)

        # evaluate on validation set
        acc1 = utils.validate(val_source_loader, model, args, device)
        print("phase 2 acc1 = {:3.4f}".format(acc1))
        wandb.log({"Model phase 2 Val Acc": acc1})
        message = '(epoch %d): Model phase 2 Val Acc %.3f' % (epoch+1, acc1)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest_model'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest_model'), logger.get_checkpoint_path('best_model_train2'))

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.4f}".format(best_acc1))
    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_train2')))
    acc1 = utils.validate(test_loader, model, args,device)
    print("Train Phase 2 Best test_acc1 = {:3.2f}".format(acc1))

    acc3 = utils.validate_ulogits(test_loader, model, args, device)
    print("base acc = {:3.4f}".format(acc3))

    model.set_requires_grad(False)

    # start test and finetune
    total_iter = 0
    best_acc2=0.
    for epoch in range(args.finetune_epochs):
        print("lr:", finetune_lr_scheduler.get_last_lr(), finetune_optimizer.param_groups[0]['lr'])
        # train for one epoch
        CasualOOD_finetune(train_target_iter, val_target_iter, model, finetune_optimizer,
                        finetune_lr_scheduler, epoch, args, total_iter, backbone)

        # evaluate on validation set
        acc3 = combined_inference(model, val_target_loader, num_classes)
        acc2 = utils.validate(val_target_loader, model, args, device)
        print("acc2 = {:3.4f}".format(acc2))
        print("acc3 = {:3.4f}".format(acc3))
        wandb.log({"Model Val Acc": acc2})
        message = '(epoch %d): Model Val Acc %.3f' % (epoch+1, acc2)
        print(message)
        record = open(test_logger, 'a')
        record.write(message+'\n')
        record.close()

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), logger.get_checkpoint_path('latest_model'))
        if acc2 > best_acc2:
            shutil.copy(logger.get_checkpoint_path('latest_model'), logger.get_checkpoint_path('best_model_test'))

        best_acc2 = max(acc2, best_acc2)

    print("best_acc2 = {:3.4f}".format(best_acc2))
    # evaluate on test set
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best_model_test')))
    acc3 = combined_inference(model, test_loader, num_classes)
    acc2 = utils.validate(test_loader, model, args, device)
    print("acc3 = {:3.4f}".format(acc3))
    print("Test Phase Best test_acc = {:3.2f}".format(acc2))

    acc3 = utils.validate_ulogits(test_loader, model, args, device)
    print("base acc = {:3.4f}".format(acc3))

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
    parser.add_argument('--data_dir', type=str, default='./data')

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

    parser.add_argument('--combine_method', type=str, default='logits', choices=['logits', 'features'],
                        help="How to combine inference results: 'logits' or 'features'")

    parser.add_argument('--mi_type', type=str, default='conditional', choices=['conditional', 'cosine'],
                        help="Mutual information type: 'conditional' or 'cosine'")

    parser.add_argument('--finetune_logits', type=str, default='tilde', choices=['tilde', 'combined'],
                        help="Which logits to use in finetune phase: 'tilde' or 'combined'")

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
