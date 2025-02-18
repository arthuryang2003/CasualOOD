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
from extract_features import extract_features

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
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


def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    return Dataset()


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets + ['Digits']

class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, split_indices, domain_id):
        self.original_dataset = original_dataset
        self.split_indices = split_indices
        self.domain_id = domain_id

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        # Get the index from the split indices and fetch the corresponding sample
        real_idx = self.split_indices[idx]
        return self.original_dataset[real_idx]

class UniformDataset(torch.utils.data.Dataset):
    def __init__(self, datasets=[]):
        super(UniformDataset, self).__init__()
        self.datasets = datasets
        total = 0
        for ds in datasets:
            total += len(ds)
        self.total = total
        self.domain_id = 0

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        #domain_id = np.random.choice(len(self.datasets))
        domain_id = self.domain_id % len(self.datasets)
        self.domain_id += 1
        self.domain_id = self.domain_id % len(self.datasets)
        idx = idx % len(self.datasets[domain_id])
        return self.datasets[domain_id][idx]



def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)

    elif dataset_name in datasets.__dict__:
        # Check if the dataset is PACS or another dataset
        dataset = datasets.__dict__[dataset_name]
        all_domains = source + target
        assert target[0] not in source

        # Define the concat_dataset function
        def concat_dataset(tasks, **kwargs):
            domain_ids = []
            dataset_list = []
            for task in tasks:
                dt = dataset(task=task, domain_index=all_domains.index(task), **kwargs)
                domain_ids += [all_domains.index(task)] * len(dt)
                dataset_list.append(dt)
            # x = ConcatDataset(dataset_list)
            x = UniformDataset(dataset_list)
            x.domain_ids = domain_ids
            return x

        def concat_and_split_dataset(tasks, split_ratio=0.8, **kwargs):

            # Initialize domain_ids and dataset lists for both train and val
            train_domain_ids = []
            val_domain_ids = []
            train_dataset_list = []
            val_dataset_list = []

            # Loop over each task and split the dataset
            for task in tasks:
                # Load dataset for the task
                ds = dataset(task=task, domain_index=all_domains.index(task), **kwargs)

                # Access the samples from the dataset directly (dt.samples contains the actual data)
                samples = ds.samples

                # Calculate the split index based on the split_ratio
                total_samples = len(samples)
                split_index = int(total_samples * split_ratio)

                # Split the dataset into train and val by iterating over each item
                train_indices = list(range(split_index))  # First part for training
                val_indices = list(range(split_index, total_samples))  # Remaining part for validation

                dt = SplitDataset(ds, train_indices, all_domains.index(task))
                dv=SplitDataset(ds, val_indices, all_domains.index(task))

                train_domain_ids += [all_domains.index(task)] * len(dt)
                val_domain_ids += [all_domains.index(task)] * len(dv)

                # Create new SplitDataset instances for train and val sets
                train_dataset_list.append(dt)
                val_dataset_list.append(dv)

            # Create new datasets for train and validation sets
            train_dataset = UniformDataset(train_dataset_list)
            val_dataset = UniformDataset(val_dataset_list)

            # Assign domain_ids to both train and val datasets
            train_dataset.domain_ids = train_domain_ids
            val_dataset.domain_ids = val_domain_ids

            return train_dataset, val_dataset

        if dataset_name == 'PACS':
            # PACS dataset loading
            train_source_dataset = concat_dataset(root=root, tasks=source, download=True, split='train', phase='train',
                                                  transform=train_source_transform)
            val_dataset = concat_dataset(root=root, tasks=source, download=True, split='test', phase='val',
                                         transform=val_transform)

            # train_source_dataset = concat_and_split_dataset(tasks=source, split_ratio=0.8, root=root,
            #                                                 download=True, phase='train',
            #                                                 transform=train_source_transform)[0]
            # val_dataset = concat_and_split_dataset(tasks=source, split_ratio=0.8, root=root,
            #                                        download=True, phase='val', transform=val_transform)[1]
        else:
            train_source_dataset = concat_and_split_dataset(tasks=source, split_ratio=0.8, root=root,
                                                            download=True, phase='train',
                                                            transform=train_source_transform)[0]
            val_dataset = concat_and_split_dataset(tasks=source, split_ratio=0.8, root=root,
                                                   download=True, phase='val', transform=val_transform)[1]


        test_dataset = concat_dataset(root=root, tasks=target, download=True, phase='test', transform=val_transform)
        # Extract class names and number of classes
        class_names = test_dataset.datasets[0].classes
        num_classes = len(class_names)


    else:
        # load datasets from wilds
        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        class_names = None
        train_source_dataset = convert_from_wilds_dataset(dataset.get_subset('train', transform=train_source_transform))
        train_target_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=train_target_transform))
        val_dataset = test_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=val_transform))
    return train_source_dataset, val_dataset, test_dataset, num_classes, class_names


def validate_classifier(vae_model, stable_classifier, unstable_classifier, val_loader, args, device):
    # 定义统计指标
    batch_time = AverageMeter('Time', ':6.3f')
    stable_losses = AverageMeter('Loss', ':6.3f')
    unstable_losses = AverageMeter('Loss', ':6.3f')
    top1_stable = AverageMeter('Stable Acc@1', ':6.2f')
    top1_unstable = AverageMeter('Unstable Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, stable_losses,unstable_losses, top1_stable, top1_unstable],
        prefix='Validation: ')

    # 切换到评估模式
    stable_classifier.eval()
    unstable_classifier.eval()

    with torch.no_grad():
        end = time.time()

        for i, data in enumerate(val_loader):
            images, target = data[0], data[1]
            images = images.to(device)
            target = target.to(device)

            # 使用VAE模型提取特征
            content, style = extract_features(vae_model, images, target)

            # 使用稳定分类器进行预测
            stable_output = stable_classifier(content)
            stable_loss = F.cross_entropy(stable_output, target)

            # 生成伪标签
            pseudo_labels = torch.argmax(stable_output, dim=1)
            pseudo_labels = pseudo_labels.to(device)

            # 使用不稳定分类器进行预测
            unstable_output = unstable_classifier(style)
            unstable_loss = F.cross_entropy(unstable_output, pseudo_labels)

            # 计算准确率
            stable_acc, = accuracy(stable_output, target, topk=(1,))
            unstable_acc, = accuracy(unstable_output, pseudo_labels, topk=(1,))

            # 更新统计
            stable_losses.update(stable_loss.item(), images.size(0))
            unstable_losses.update(unstable_loss.item(), images.size(0))
            top1_stable.update(stable_acc.item(), images.size(0))
            top1_unstable.update(unstable_acc.item(), images.size(0))

            # 计时
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)
    progress.display(i)
    return top1_stable.avg, top1_unstable.avg


def validate_vae(val_loader, model, args, total_iter,device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses_vae = AverageMeter('VAE', ':4.4f')
    losses_cls = AverageMeter('Cls', ':4.4f')
    losses_kl = AverageMeter('KL', ':4.4f')
    losses_recon = AverageMeter('Rec', ':4.4f')
    total_loss = AverageMeter('Loss', ':4.4f')

    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,losses_vae, losses_cls, losses_kl, losses_recon, total_loss, top1],
        prefix='Test: ')

    normal_distribution = torch.distributions.MultivariateNormal(torch.zeros(args.z_dim).cuda(),
                                                                 torch.eye(args.z_dim).cuda())

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images = data[0]
            target = data[1]
            u=data[2]
            images = images.to(device)
            target = target.to(device)
            u =u.to(device)

            # Feature extraction
            x = model.backbone(images)
            z, tilde_z, mu, log_var, logdet_u, logit = model.encode(x, u=u)

            # Classification loss
            cls_loss = F.cross_entropy(logit, target)

            # VAE KL loss
            q_dist = torch.distributions.Normal(mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
            log_qz = q_dist.log_prob(z)
            log_pz = normal_distribution.log_prob(tilde_z) + logdet_u
            kl = (log_qz.sum(dim=1) - log_pz).mean()

            # Smooth C value to adjust KL loss during training
            C = torch.clamp(torch.tensor(args.C_max) / args.C_stop_iter * total_iter, 0, args.C_max)
            loss_kl = args.beta * (kl - C).abs()  # KL loss

            # Reconstruction loss
            x_hat = model.decode(z)
            recon_loss = F.mse_loss(x, x_hat, reduction='sum') / len(x)

            # VAE loss (KL loss + reconstruction loss)
            mean_loss_vae = recon_loss + loss_kl



            # Total loss (Classification loss + VAE loss)
            total_val_loss = cls_loss + args.lambda_vae * mean_loss_vae

            # Update meters
            losses_vae.update(mean_loss_vae.item(), images.size(0))
            losses_cls.update(cls_loss.item(), images.size(0))
            losses_kl.update(loss_kl.item(), images.size(0))
            losses_recon.update(recon_loss.item(), images.size(0))
            total_loss.update(total_val_loss.item(), images.size(0))

            # Measure accuracy
            acc1 = accuracy(logit, target)[0]
            top1.update(acc1.item(), images.size(0))

            # Confusion matrix update
            if confmat:
                confmat.update(target, logit.argmax(1))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        if confmat:
            print(confmat.format(args.class_names))


    progress.display(i)
    return top1.avg, total_loss.avg


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



def save_pair_images(images_a, images_b, filename):
    images = []
    nrow = int(np.sqrt(len(images_a)))*2
    for im_a, im_b in zip(images_a, images_b):
        images.append(im_a)
        images.append(im_b)
    images = torch.stack(images, 0).detach().cpu()
    save_image(images, filename, nrow=nrow, normalize=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def validate_decoupler(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
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
            images = data[0]
            target = data[1]
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
