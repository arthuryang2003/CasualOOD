import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import itertools


class Classifier(nn.Module):
    def __init__(self, args, input_size):
        super(Classifier, self).__init__()

        # 使用args初始化hidden_dim和num_classes
        dim = args.hidden_dim
        self.num_classes = args.num_classes

        # 确保input_size（即c_dim或s_dim）已传入
        if input_size is None:
            raise ValueError("Input size (c_dim or s_dim) must be provided.")

        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_size, dim),  # 输入为c_dim或s_dim
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, self.num_classes)  # 输出类别数
        )

    def forward(self, x):
        return self.classifier(x)



class Decoupler(nn.Module):
    def __init__(self, args, backbone_net=None):
        super(Decoupler, self).__init__()

        self.args = args
        self.backbone_net = backbone_net

        # latent space dimensions
        self.z_dim = args.z_dim  # 总潜在空间的维度
        self.s_dim = args.s_dim  # 虚假特征的维度
        self.c_dim = self.z_dim - self.s_dim  # 不变特征的维度

        dim = args.hidden_dim

        # Define the backbone (feature extractor)
        self.backbone_net = backbone_net

        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)), nn.Flatten())

        # Define the encoder to map features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.backbone_net.out_features, dim),  # 假设backbone的输出是out_features维度
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, self.z_dim)  # 潜在空间的维度是z_dim
        )

        # Projection layers for decoupling the features into invariant and spurious
        self.projection_phi = nn.Linear(self.z_dim, self.c_dim)  # Project to invariant features
        self.projection_psi = nn.Linear(self.z_dim, self.s_dim)  # Project to spurious features

        # Classifiers for stable (content) and unstable (style) features
        self.classifier_stable = nn.Sequential(
            nn.Linear(self.c_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

        self.classifier_unstable = nn.Sequential(
            nn.Linear(self.s_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, args.num_classes)
        )

    def predict_stable(self, z_u):

        return self.classifier_stable(z_u)

    def predict_unstable(self, z_s):

        return self.classifier_unstable(z_s)



    def forward(self, x):

        # 获取解藕后的不变特征和虚假特征
        z_u, _ = self.extract_feature(x)

        # 通过不变特征进行分类
        logits = self.classifier_stable(z_u)

        return logits

    def extract_feature(self, x):
        """
        提取特征并进行解藕，将特征解耦为不变特征（z_u）和虚假特征（z_s）
        """
        # Step 1: Extract features using the backbone
        x_feat = self.backbone(x)  # The output of the backbone network

        # Step 2: Project to the latent space
        z = self.encoder(x_feat)

        # Step 3: Apply projections to decouple into invariant and spurious features
        z_u = self.projection_phi(z)  # Invariant features
        z_s = self.projection_psi(z)  # Spurious features

        return z_u, z_s


    def get_parameters(self, base_lr=1.0):
        """返回优化器所需的参数列表，支持为不同模块设置不同的学习率"""

        # 通过 itertools.chain() 可以连接多个层的参数
        base_params = itertools.chain(self.encoder.parameters(),
                                      self.projection_phi.parameters(),
                                      self.projection_psi.parameters(),
                                      self.classifier_stable.parameters(),
                                      self.classifier_unstable.parameters())

        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},  # backbone 使用较小的学习率
            {"params": base_params, "lr": 1.0 * base_lr},  # projection_phi, projection_psi, classifier 使用默认学习率
        ]

        return params

    def backbone(self, x):
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out


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

        # Define the backbone (feature extractor)
        self.backbone_net = backbone_net

        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())

        # Define the encoder to map features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.backbone_net.out_features, dim),  # 假设backbone的输出是out_features维度
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, self.z_dim)  # 潜在空间的维度是z_dim
        )

        # Projection layers for decoupling the features into invariant and spurious
        self.projection_phi = nn.Linear(self.z_dim, self.c_dim)  # Project to invariant features
        self.projection_psi = nn.Linear(self.z_dim, self.s_dim)  # Project to spurious features

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

        # GumbelSoftmaxLayer as a learnable network layer
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Start with a temperature of 1.0

    def set_requires_grad(self, requires_grad):
        """
        在训练时启用所有层的梯度，测试时只启用temperature的梯度，其他层冻结。
        """
        for name, param in self.named_parameters():
            if name == "temperature":  # 只对temperature层启用梯度
                param.requires_grad = True
            else:
                param.requires_grad = requires_grad  # 其他层冻结

    def backbone(self, x):
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict_u(self, z_u):
        u_logits = self.classifier(z_u)
        return u_logits

    def predict_s(self, z_s):
        s_logits = self.classifier(z_s)
        return s_logits

    def predict_tilde_s(self, tilde_z_s):
        tilde_s_logits = self.classifier(tilde_z_s)
        return tilde_s_logits

    def domain_influence(self, z_s, hard=False):
        """
        使用Gumbel-Softmax和学习到的温度调整spurious特征。
        """
        # Apply Gumbel-Softmax with learnable temperature
        gumbel_softmax = F.gumbel_softmax(z_s, tau=self.temperature, hard=hard)

        # The influence on the spurious features can be represented by the transformation of z_s
        tilde_z_s = gumbel_softmax * z_s

        return tilde_z_s

    def forward(self, x):
        z_u, z_s, u_logits, s_logits, tilde_s_logits = self.encode(x)
        logits = u_logits + tilde_s_logits
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

        # Get logits
        u_logits = self.predict_u(z_u)
        s_logits = self.predict_s(z_s)
        tilde_s_logits = self.predict_tilde_s(tilde_z_s)

        return z_u, z_s, u_logits, s_logits, tilde_s_logits

    def extract_feature(self, x):
        """
        提取特征并解耦为不变特征（z_u）和虚假特征（z_s）
        """
        # Step 1: Extract features using the backbone
        x_feat = self.backbone(x)  # The output of the backbone network

        # Step 2: Project to the latent space
        z = self.encoder(x_feat)

        # Step 3: Apply projections to decouple into invariant and spurious features
        z_u = self.projection_phi(z)  # Invariant features
        z_s = self.projection_psi(z)  # Spurious features

        return z_u, z_s

    def get_parameters(self, base_lr=1.0):
        """返回优化器所需的参数列表，支持为不同模块设置不同的学习率"""

        # Use itertools.chain() to combine parameters from different layers
        base_params = itertools.chain(self.encoder.parameters(),
                                      self.projection_phi.parameters(),
                                      self.projection_psi.parameters(),
                                      self.classifier.parameters())

        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},  # backbone使用较小的学习率
            {"params": base_params, "lr": 1.0 * base_lr},  # projection_phi, projection_psi, classifier使用默认学习率
            {"params": self.temperature, "lr": 1.0 * base_lr}  # 只训练temperature
        ]

        return params



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, hidden_dim=1024):
        super(MLP, self).__init__()
        model = []
        model += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers):
            model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class iVAE(nn.Module):
    def __init__(self, args, backbone_net=None):
        super(iVAE, self).__init__()
        self.args = args
        self.backbone_net = backbone_net
        self.pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)), nn.Flatten())

        # latent space: [0:self.c_dim] [self.c_dim:self_z_dim]
        self.z_dim = args.z_dim
        self.s_dim = args.s_dim
        self.c_dim = self.z_dim - self.s_dim

        flow_dim = args.flow_dim
        flow_nlayer = args.flow_nlayer
        flow = args.flow
        dim = args.hidden_dim
        self.encoder = nn.Sequential(nn.Linear(self.backbone_net.out_features, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(), nn.Dropout())
        self.fc_mu = nn.Sequential(nn.Linear(dim, self.z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(dim, self.z_dim))

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.ReLU(),
                                     nn.Linear(dim, self.backbone_net.out_features))

        # 全局分类器
        if args.arch == 'resnet18':
            self.classifier = nn.Sequential(
                        nn.Linear(self.z_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(),
                        nn.Linear(dim, args.num_classes)
            )
        else:
            self.classifier = nn.Sequential(nn.Linear(self.z_dim, args.num_classes))

        # # 新增稳定和不稳定分类器
        # self.stable_classifier = nn.Sequential(
        #     nn.Linear(self.c_dim, dim),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(dim, args.num_classes)  # 输出类别数
        # )
        #
        # self.unstable_classifier = nn.Sequential(
        #     nn.Linear(self.s_dim, dim),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(dim, args.num_classes)  # 输出类别数
        # )
        #

        self.flow_type = flow
        self.u_embedding = nn.Embedding(10, 1024)
        assert flow in ['ddsf', 'dsf', 'sf', 'nsf']
        if flow == 'sf':
            self.domain_flow = SigmoidFlow(flow_dim)
        elif flow == 'dsf':
            self.domain_flow = DenseSigmoidFlow(1, flow_dim, 1)
        elif flow == 'ddsf':
            self.domain_flow = DDSF(flow_nlayer, 1, flow_dim, 1)
        if self.flow_type in ['sf', 'dsf', 'ddsf']:
            domain_num_params = self.domain_flow.num_params * self.s_dim
            self.domain_mlp = MLP(1024, domain_num_params)

        # print(self.encoder, self.fc_mu, self.fc_logvar)
        # print(self.decoder, self.classifier, self.stable_classifier, self.unstable_classifier)


        self.lambda_vae = args.lambda_vae

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = track
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def extract_feature(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x, track_bn)
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        tilde_z, _ = self.domain_influence(z, u) 
        return tilde_z

    def encode(self, x, u, track_bn=False):

        self.track_bn_stats(track_bn)

        # sample z
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_logvar(h) # 计算潜变量 z 的均值和对数方差
        if self.training or self.lambda_vae != 0:
            z = self.reparameterize(mu, log_var)# 使用 reparameterization trick 采样 z
        else:
            z = mu
        
        # de-influence u
        tilde_z, logdet_u = self.domain_influence(z, u) # remove the domain influcence; back to Gaussian            

        # get logits
        logit = self.predict(tilde_z, track_bn=track_bn)

        return z, tilde_z, mu, log_var, logdet_u, logit # tilde_z is for domain adversarial, tilde_tilde_z is for KL p, z is for reconstruction and KL q. 


    def domain_influence(self, z, u):

        if self.flow_type == 'nsf':
            zcont = z[:, :-self.s_dim]
            tilde_zs = self.domain_flow(z[:, -self.s_dim:], u)

        else:
            domain_embedding = self.u_embedding(u)  # B,h_dim
            B, _ = domain_embedding.size()
            dsparams = self.domain_mlp(domain_embedding)  # B, ndim
            dsparams = dsparams.view(B, self.s_dim, -1)
            zcont = z[:,:self.c_dim]
            tilde_zs, logdet = self.domain_flow(z[:,-self.s_dim:], dsparams)

        tilde_z = torch.cat([zcont, tilde_zs], 1)

        return tilde_z, logdet

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, u, track_bn=False):
        self.track_bn_stats(track_bn)
        x = self.backbone(x)
        _, _, _, _, _, logit = self.encode(x, u=u)
        if self.training:
            raise NotImplementedError
            return tilde_z, logit
        else:
            return logit

    def backbone(self, x, track_bn=False):
        self.track_bn_stats(track_bn)
        out = self.backbone_net(x)
        if len(out.size()) > 2:
            out = self.pool_layer(out)
        return out

    def predict(self, z, track_bn=False):
        self.track_bn_stats(track_bn)
        return self.classifier(z)

    def predict_stable(self, z_stable,track_bn=False):
        """
        使用稳定分类器进行预测
        """
        self.track_bn_stats(track_bn)
        return self.stable_classifier(z_stable)

    def predict_unstable(self, z_unstable,track_bn=False):
        """
        使用不稳定分类器进行预测
        """
        self.track_bn_stats(track_bn)
        return self.unstable_classifier(z_unstable)

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        base_params = itertools.chain(self.encoder.parameters(), self.fc_mu.parameters(),
                                      self.fc_logvar.parameters(), self.decoder.parameters(),
                                      self.classifier.parameters(), 
                                      self.u_embedding.parameters(),
                                      self.domain_flow.parameters(),
                                      self.domain_mlp.parameters(),
                                      )
        params = [
            {"params": self.backbone_net.parameters(), "lr": 0.1 * base_lr},
            {"params": base_params, "lr": 1.0 * base_lr},
        ]
        return params



class BaseFlow(nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)

class SigmoidFlow(BaseFlow):

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.num_params = 3 * num_ds_dim
        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)

    def forward(self, x, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim:3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        return xnew.squeeze(-1)

class DenseSigmoidFlow(BaseFlow):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: softplus_(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=3)
        self.act_u = lambda x: softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))
        self.num_params = 3 * hidden_dim + in_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, dsparams, logdet=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        inv = np.log(np.exp(1 - delta) - 1)
        ndim = self.hidden_dim
        pre_u = self.u_[None, None, :, :] + dsparams[:, :, -self.in_dim:][:, :, None, :]
        pre_w = self.w_[None, None, :, :] + dsparams[:, :, 2 * ndim:3 * ndim][:, :, None, :]
        a = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)

        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(pre_w, dim=3) + \
            logsigmoid(pre_sigm[:,:,None,:]) + \
            logsigmoid(-pre_sigm[:,:,None,:]) + log(a[:,:,None,:])
        # n, d, d2, dh
        
        logj = logj[:,:,:,:,None] + F.log_softmax(pre_u, dim=3)[:,:,None,:,:]
        # n, d, d2, dh, d1
        
        logj = log_sum_exp(logj,3).sum(3)
        # n, d, d2, d1
        
        logdet_ = logj + np.log(1-delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped+1))[:,:,:,None]
        
        if logdet is None:
            logdet = logdet_.new_zeros(logdet_.shape[0], logdet_.shape[1], 1, 1)
        
        logdet = log_sum_exp(
            logdet_[:,:,:,:,None] + logdet[:,:,None,:,:], 3
        ).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0
        
        return xnew.squeeze(-1), logdet
        

    def extra_repr(self):
        return 'input_dim={in_dim}, output_dim={out_dim}'.format(**self.__dict__)

class DDSF(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(DDSF, self).__init__()
        blocks = [DenseSigmoidFlow(in_dim, hidden_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, hidden_dim)]
        blocks += [DenseSigmoidFlow(hidden_dim, hidden_dim, out_dim)]
        self.num_params = 0
        for block in blocks:
            self.num_params += block.num_params
        self.model = nn.ModuleList(blocks)

    def forward(self, x, dsparams):
        start = 0
        _logdet = None

        for block in self.model:
            block_dsparams = dsparams[:, :, start:start + block.num_params]
            x, _logdet = block(x, block_dsparams, logdet=_logdet)
            start = start + block.num_params

        logdet = _logdet[:,:,0,0].sum(1)

        return x, logdet


def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper

def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A,maximum, axis, True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max    
    return B
