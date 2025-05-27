"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    LeNet 改写版，结构与 MNIST_CNN 一致：
    - 4个卷积层 + GroupNorm
    - ReLU激活
    - AdaptiveAvgPool + Flatten 输出 128维
    - 可复制分类头用于多任务学习或IRM等算法
    """
    def __init__(self, input_shape=(2, 28, 28), num_classes=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.GroupNorm(8, 64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(8, 128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(8, 128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity()  # 可以根据算法替换为非线性激活

        self.out_features = 128
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.out_features, self.num_classes)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        feat = self.activation(x)
        return feat

    def copy_head(self):
        """复制一份新的分类器头，用于多环境训练"""
        return nn.Linear(self.out_features, self.num_classes)


# class LeNet(nn.Sequential):
#     def __init__(self, num_classes=2):
#         super(LeNet, self).__init__(
#             nn.Conv2d(2, 20, kernel_size=5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.Dropout2d(p=0.5),
#             nn.MaxPool2d(2),
#             nn.ReLU(),
#             nn.Flatten(start_dim=1),
#             nn.Linear(50 * 4 * 4, 500),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#         )
#         self.num_classes = num_classes
#         self.out_features = 500
#
#     def copy_head(self):
#         return nn.Linear(500, self.num_classes)


class DTN(nn.Sequential):
    def __init__(self, num_classes=10):
        super(DTN, self).__init__(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.num_classes = num_classes
        self.out_features = 512

    def copy_head(self):
        return nn.Linear(512, self.num_classes)



def lenet(pretrained=False, **kwargs):
    """LeNet model from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 28 x 28.

    """
    return LeNet(**kwargs)


def dtn(pretrained=False, **kwargs):
    """ DTN model

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 32 x 32.

    """
    return DTN(**kwargs)