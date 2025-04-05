"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import wilds
import numpy as np
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from torchvision.utils import save_image
from itertools import chain

class ForeverDataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iterator = zip(*data_loader)

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = zip(*self.data_loader)
            data = next(self.iterator)
        return data

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        if model_name == 'MINST_CNN' :
            backbone = models.__dict__[model_name]
        else :
            backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone



def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])



def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_len = sum(len(loader) for loader in val_loader)
    progress = ProgressMeter(
        total_len,
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    val_iter = chain(*val_loader)  # <-- 这里拼接
    with torch.no_grad():
        end = time.time()
        for i, (images, target)  in enumerate(val_iter):

            # images = torch.cat([b[0] for b in data], dim=0).to(device)
            # target = torch.cat([b[1] for b in data], dim=0).to(device)
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


def validate_add_logits(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    total_len = sum(len(loader) for loader in val_loader)
    stable_losses = AverageMeter('Loss', ':.4e')
    combined_losses = AverageMeter('CombLoss', ':.4e')
    stable_top1 = AverageMeter('Acc@1', ':6.2f')
    combined_top1 = AverageMeter('CombAcc@1', ':6.2f')
    progress = ProgressMeter(
        total_len,
        [batch_time,combined_losses, combined_top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images = torch.cat([b[0] for b in data], dim=0).to(device)
            target = torch.cat([b[1] for b in data], dim=0).to(device)

            z_u, z_s, u_logits, s_logits, tilde_s_logits,_ = model.encode(images)
            stable_loss = F.cross_entropy(u_logits, target)
            combined_logits=u_logits+tilde_s_logits
            combined_loss = F.cross_entropy(combined_logits, target)

            # measure accuracy and record loss
            acc2, = accuracy(combined_logits, target, topk=(1,))
            if confmat:
                confmat.update(target, combined_logits.argmax(1))

            combined_losses.update(combined_loss.item(), images.size(0))
            combined_top1.update(acc2.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if confmat:
            print(confmat.format(args.class_names))

    return combined_top1.avg

def validate_ulogits(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_len = sum(len(loader) for loader in val_loader)
    progress = ProgressMeter(
        total_len,
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    val_iter = chain(*val_loader)  # <-- 这里拼接
    with torch.no_grad():
        end = time.time()
        for i, (images, target)  in enumerate(val_iter):

            # images = torch.cat([b[0] for b in data], dim=0).to(device)
            # target = torch.cat([b[1] for b in data], dim=0).to(device)
            images = images.to(device)
            target = target.to(device)

            z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = model.encode(images)
            output=u_logits
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


# def combined_inference(stable_model, unstable_model, test_loader,num_classes):
#     # 初始化先验分布和混淆矩阵
#     PY = torch.zeros(num_classes).to(device)  # 类别先验分布
#     e_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵
#
#     # 第一遍：计算混淆矩阵和先验分布
#     stable_model.eval()
#     unstable_model.eval()
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             # 解包数据，只取前两个（data 和 labels）
#             data, labels = batch[:2]
#             data = data.to(device)
#             labels = labels.to(device)
#             if labels.dim() == 1:
#                 labels = F.one_hot(labels, num_classes=num_classes).float()
#
#             # compute output
#             u = torch.ones([len(data)]).long().to(device)
#             # 提取稳定特征和不稳定特征
#             stable_features, _ = extract_features(stable_model, data, u)  # 提取稳定特征
#             _, unstable_features = extract_features(unstable_model, data, u)  # 提取不稳定特征
#
#             # 稳定模型预测
#             Y_pred_stable = F.softmax(stable_model.stable_classifier(stable_features), dim=1)
#             Y_pred_stable_hard = (Y_pred_stable == Y_pred_stable.max(dim=1, keepdim=True)[0]).float()
#
#             # 更新先验分布
#             PY += labels.sum(dim=0)
#
#             e_matrix += torch.matmul(labels.T, Y_pred_stable_hard)
#
#             # # 更新混淆矩阵
#             # for k in range(num_classes):
#             #     for k_prime in range(num_classes):
#             #         e_matrix[k, k_prime] += (
#             #             (labels[:, k_prime] * Y_pred_stable_hard[:, k]).sum().item()
#             #         )
#
#     # 归一化混淆矩阵和先验分布
#     e_matrix = e_matrix / e_matrix.sum(dim=0, keepdim=True)
#     PY = PY / PY.sum()
#
#     # 第二遍：使用调整后的不稳定模型预测
#     correct = 0
#     total = 0
#     OOD = 0
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             # 解包数据，只取前两个（data 和 labels）
#             data, labels = batch[:2]
#             data = data.to(device)
#             labels = labels.to(device)
#             # 转换为 one-hot 编码
#             if labels.dim() == 1:
#                 labels = F.one_hot(labels, num_classes=num_classes).float()
#
#             u = torch.ones([len(data)]).long().to(device)
#             # 提取稳定特征和不稳定特征
#             stable_features, _ = extract_features(stable_model, data, u)  # 提取稳定特征
#             _, unstable_features = extract_features(unstable_model, data, u)  # 提取不稳定特征
#
#
#             # 稳定模型预测
#             Y_stable = F.softmax(stable_model.stable_classifier(stable_features), dim=1)
#             Xlogit = torch.log(Y_stable + 1e-6)
#
#             # 调整不稳定模型预测
#             Y_unstable = F.softmax(unstable_model.unstable_classifier(unstable_features), dim=1)
#             Y_unstable_corrected = torch.matmul(Y_unstable, torch.inverse(e_matrix))
#             Ulogit = torch.log(Y_unstable_corrected + 1e-6)
#
#             # 计算类别先验的对数
#             prior_logit = torch.log(PY / (1 - PY) + 1e-6)
#
#             # 联合预测
#             combined_logit = Xlogit + Ulogit - prior_logit
#             predict = torch.softmax(combined_logit, dim=1)
#
#
#
#             # 转换为硬标签
#             predicted = (predict == predict.max(dim=1, keepdim=True)[0]).float()
#             OOD += predicted.sum(dim=1).mean().item()
#             correct += (predicted == labels).all(dim=1).sum().item()
#             total += labels.size(0)
#
#     # 输出准确率和 OOD
#     OOD = OOD / total
#     accuracy = correct / total
#     return accuracy
