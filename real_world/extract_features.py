import torch
from common.modules.networks import iVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, x, u, is_target=False):
    """
    从输入数据中提取解耦的内容（content）和风格（style）特征。
    """
    if u is not None:
        u = u.to(device)  # 如果 u 不为空，将其移动到正确的设备上

    # 确保数据在传入 backbone 前被展平
    x = model.backbone(x,track_bn=is_target)
    x = x.view(x.size(0), -1)  # 展平为 [batch_size, features]

    # 使用模型进行编码，提取特征
    z, _, _, _, _, _ = model.encode(x, u,track_bn=is_target)
    content = z[:, :model.c_dim]  # 不变特征
    style = z[:, model.c_dim:]   # 可变特征

    return content,style

