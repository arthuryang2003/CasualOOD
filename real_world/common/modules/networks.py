import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools
from torch import autograd


class CasualOOD(nn.Module):
    def __init__(self, args, backbone_net=None):
        super(CasualOOD, self).__init__()

        self.args = args
        self.backbone_net = backbone_net

        # latent space dimensions
        self.z_dim = args.z_dim  # 总潜在空间的维度
        self.s_dim = args.z_dim  # 虚假特征的维度
        self.c_dim = args.z_dim  # 不变特征的维度

        dim = args.hidden_dim

        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())

        # Define the encoder to map features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.backbone_net.out_features, dim),  # 假设backbone的输出是out_features维度
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, self.z_dim)  # 潜在空间的维度是z_dim
        )

        self.projection_phi = nn.Sequential(
            nn.Linear(self.z_dim, self.c_dim),
            nn.BatchNorm1d(self.c_dim),
            nn.ReLU()
        )  # Invariant features

        self.projection_psi = nn.Sequential(
            nn.Linear(self.z_dim, self.s_dim),
            nn.BatchNorm1d(self.s_dim),
            nn.ReLU()
        )  # Spurious features

        # Classifiers for stable (content) and unstable (style) features
        self.classifier_u = nn.Sequential(
            nn.Linear(self.c_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

        self.classifier_s = nn.Sequential(
            nn.Linear(self.s_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

        self.classifier_tilde_s = nn.Sequential(
            nn.Linear(self.s_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.z_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

        self.mask = nn.Parameter(torch.ones(self.s_dim))  # 初始化为全1

        self.classifier_combined = nn.Sequential(
            nn.Linear(self.s_dim+self.z_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

    def set_requires_grad(self, requires_grad):
        """
        在训练时启用所有层的梯度，测试时只启用mask的梯度，其他层冻结。
        """
        for name, param in self.named_parameters():
            if name == "mask":  # 只对mask层启用梯度
                param.requires_grad = True
            else:
                param.requires_grad = requires_grad  # 其他层冻结

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = requires_grad
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = requires_grad

    def set_requires_grad_phase1(self):
        """
        第一阶段:
        - 冻结 mask 参数
        - 冻结 classifier_tilde_s 中的所有参数
        - 其他层保持可训练
        """
        for name, param in self.named_parameters():
            if name == "mask":
                param.requires_grad = False
            elif "classifier_tilde_s" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_requires_grad_phase2(self):
        """
        第二阶段:
        - 启用 mask 参数
        - 启用 classifier_tilde_s 中的所有参数
        - 其他层（如需要继续训练也可保持 True，如果希望第二阶段只训练这两部分，则可将其他层设为 False）
        """
        for name, param in self.named_parameters():
            if name == "mask":
                param.requires_grad = True
            elif "classifier_tilde_s" in name:
                param.requires_grad = True
            elif "classifier_combined" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    if "classifier_tilde_s" in m.__class__.__name__:
                        m.track_running_stats = True
                    elif "classifier_combined" in m.__class__.__name__:
                        m.track_running_stats = True
                    else:
                        m.track_running_stats = False

    def backbone(self, x):
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict_combined(self, tilde_z):
        logits = self.classifier_combined(tilde_z)
        return logits

    def predict_u(self, z_u):
        u_logits = self.classifier(z_u)
        # u_logits = self.classifier_u(z_u)
        return u_logits

    def predict_s(self, z_s):
        s_logits = self.classifier(z_s)
        # s_logits = self.classifier_s(z_s)
        return s_logits

    def predict_tilde_s(self, tilde_z_s):
        tilde_s_logits = self.classifier(tilde_z_s)
        # tilde_s_logits = self.classifier_tilde_s(tilde_z_s)
        return tilde_s_logits

    def drop_spurious_features(self, z_s):
        mask = torch.sigmoid(self.mask)
        drop_z_s = (1 - mask) * z_s
        return drop_z_s

    def domain_influence(self, z_s, hard=False):

        mask = torch.sigmoid(self.mask)

        # The influence on the spurious features can be represented by the transformation of z_s
        tilde_z_s = mask * z_s

        return tilde_z_s

    def forward(self, x):
        z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits = self.encode(x)
        logits = combined_logits
        # logits=u_logits+tilde_s_logits
        return logits

    def encode(self, x):
        # Step 1: Extract features using the backbone
        x_feat = self.backbone(x)  # The output of the backbone network

        # Step 2: Project to the latent space
        z = self.encoder(x_feat)

        # Step 3: Apply projections to decouple into invariant and spurious features
        z_u = self.projection_phi(z)  # Invariant features
        z_s = self.projection_psi(z)  # Spurious features

        # De-influence z_s using Gumbel-Softmax with a learnable temperature
        tilde_z_s = self.domain_influence(z_s)  # Remove the domain influence; back to Gaussian

        # 合并 zu 和 tilde_zs
        tilde_z = torch.cat([z_u, tilde_z_s], dim=1)

        combined_logits = self.classifier_combined(tilde_z)

        # Get logits
        u_logits = self.predict_u(z_u)
        s_logits = self.predict_s(z_s)
        tilde_s_logits = self.predict_tilde_s(tilde_z_s)
        # combined_logits = u_logits+tilde_s_logits
        return z_u, z_s, u_logits, s_logits, tilde_s_logits,combined_logits


    def get_parameters(self, base_lr=1.0):
        """返回优化器所需的参数列表，支持为不同模块设置不同的学习率"""

        # Use itertools.chain() to combine parameters from different layers
        base_params = itertools.chain(self.encoder.parameters(),
                                      self.projection_phi.parameters(),
                                      self.projection_psi.parameters(),
                                      self.classifier.parameters(),
                                      self.classifier_combined.parameters(),
                                      self.classifier_u.parameters(),
                                      self.classifier_s.parameters(),
                                      self.classifier_tilde_s.parameters()
                                      )

        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},  # backbone使用较小的学习率
            {"params": base_params, "lr": 1.0 * base_lr},  # projection_phi, projection_psi, classifier使用默认学习率
            {"params": self.mask, "lr": 1.0 * base_lr}  # 只训练temperature
        ]

        return params


    def get_finetune_parameters(self, base_lr=1.0):
        """返回优化器所需的参数列表，支持为不同模块设置不同的学习率"""

        # Use itertools.chain() to combine parameters from different layers
        base_params = itertools.chain(
                                      self.classifier_combined.parameters(),
                                      self.classifier_tilde_s.parameters())

        params = [
            {"params": base_params, "lr": 1.0 * base_lr},  # projection_phi, projection_psi, classifier使用默认学习率
            {"params": self.mask, "lr": 1.0 * base_lr}  # 只训练temperature
        ]

        return params

    def get_parameters_train_phase2(self, base_lr=1.0):
        """只训练 temperature 和 classifier_tilde_s，冻结其他参数"""

        params = [
            {"params": self.classifier_combined.parameters(), "lr": 1.0 * base_lr},  # 训练
            {"params": self.classifier_tilde_s.parameters(), "lr": 1.0 * base_lr},  # 训练 classifier_tilde_s
            {"params": self.mask, "lr": 1.0 * base_lr}  # 训练 temperature
        ]

        return params

    def get_parameters_train_phase1(self, base_lr=1.0):
        """只训练 temperature 和 classifier_tilde_s，冻结其他参数"""
        # Use itertools.chain() to combine parameters from different layers
        base_params = itertools.chain(self.encoder.parameters(),
                                      self.projection_phi.parameters(),
                                      self.projection_psi.parameters(),
                                      self.classifier_u.parameters(),
                                      self.classifier_s.parameters())
        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},  # backbone使用较小的学习率
            {"params": base_params, "lr": 1.0 * base_lr},  # projection_phi, projection_psi, classifier使用默认学习率
        ]

        return params



class ERMNet(nn.Module):
    def __init__(self, args, backbone_net=None):
        super(ERMNet, self).__init__()
        self.args = args
        self.backbone_net = backbone_net

        dim = args.hidden_dim
        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_net.out_features, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out

    def backbone(self, x):
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict(self, x):
        return self.forward(x)

    def get_parameters(self, base_lr=1.0):
        return [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},
            {"params": self.classifier.parameters(), "lr": 1.0 * base_lr}
        ]

class IRMNet(ERMNet):
    def __init__(self, args, backbone_net=None):
        super(IRMNet, self).__init__(args, backbone_net)
        self.update_count = 0
        # 设置默认 IRM 参数（如果 args 中没有定义）
        self.irm_lambda = getattr(args, 'irm_lambda', 1.0)
        self.irm_anneal_iters = getattr(args, 'irm_anneal_iters', 500)

    def irm_penalty(self, logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    def get_penalized_loss(self, x,y):
        logits = self.forward(x)
        batch_size = x.size(0)
        assert batch_size % 2 == 0, "IRM penalty 计算需要偶数样本（交叉组合）"

        nll = F.cross_entropy(logits, y)
        penalty = self.irm_penalty(logits, y)

        penalty_weight = self.irm_lambda if self.update_count >= self.irm_anneal_iters else 1.0
        total_loss = nll + penalty_weight * penalty

        return total_loss, nll.item(), penalty.item()