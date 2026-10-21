import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from objectives import fca_loss, daefca_loss, convcca_loss, daefca02_loss
from fimdmf_ops import fca_loss_ste, daefca_loss_ste


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid()
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepFCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, device=torch.device('cpu')):
        super(DeepFCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()  # 构建网络模型--多层感知机
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = fca_loss(outdim_size, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)  # 把视图x1输入到模型1中，得到输出output1
        output2 = self.model2(x2)

        return output1, output2


class DeepAEFCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, device=torch.device('cpu')):
        super(DeepAEFCA, self).__init__()
        self.encoder1 = MlpNet(layer_sizes1, input_size1).double()
        self.encoder2 = MlpNet(layer_sizes2, input_size2).double()

        self.decoder1 = MlpNet(layer_sizes1[-2::-1] + [input_size1], layer_sizes1[-1]).double()
        self.decoder2 = MlpNet(layer_sizes2[-2::-1] + [input_size2], layer_sizes2[-1]).double()

        self.loss = daefca_loss(outdim_size, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.encoder1(x1)
        output2 = self.encoder2(x2)

        re1 = self.decoder1(output1)
        re2 = self.decoder2(output2)

        return output1, output2, re1, re2


class DeepAEFCA02(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, device=torch.device('cpu')):
        super(DeepAEFCA02, self).__init__()
        self.encoder1 = MlpNet(layer_sizes1, input_size1).double()
        self.encoder2 = MlpNet(layer_sizes2, input_size2).double()

        self.decoder1 = MlpNet(layer_sizes1[-2::-1] + [input_size1], layer_sizes1[-1]).double()
        self.decoder2 = MlpNet(layer_sizes2[-2::-1] + [input_size2], layer_sizes2[-1]).double()

        self.loss = daefca02_loss(outdim_size, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.encoder1(x1)
        output2 = self.encoder2(x2)

        re1 = self.decoder1(output1)
        re2 = self.decoder2(output2)

        return output1, output2, re1, re2


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # cs = [1, 32, 64, 128]
        cs = [1, 64, 128, 256]

        self.conv1 = nn.Sequential(
            nn.Conv2d(cs[0], cs[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(cs[1]),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cs[1], cs[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cs[2]),
            nn.ReLU(True),
            nn.Conv2d(cs[2], cs[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(cs[2]),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(cs[2], cs[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cs[3]),
            nn.ReLU(True),
            nn.Conv2d(cs[3], cs[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(cs[3]),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(cs[3], 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.fc(x)
        return x


class ConvCCA(nn.Module):
    def __init__(self, layer_sizes1, input_size1, layer_sizes2, input_size2):
        super(ConvCCA, self).__init__()

        self.model1 = ConvNet()
        self.model2 = ConvNet()

        self.loss = convcca_loss(outdim_size=10, use_all_singular_values=False, device=torch.device('cpu')).loss

    def forward(self, x1, x2):
        # feature * batch_size
        z1 = self.model1(x1)
        z2 = self.model2(x2)

        return z1, z2


class ConvFCA(nn.Module):
    def __init__(self, layer_sizes1, input_size1, layer_sizes2, input_size2):
        super(ConvFCA, self).__init__()

        self.model1 = ConvNet()
        self.model2 = ConvNet()

        self.loss = fca_loss(outdim_size=50, device=torch.device('cpu')).loss

    def forward(self, x1, x2):
        # feature * batch_size
        z1 = self.model1(x1)
        z2 = self.model2(x2)

        return z1, z2


# ======================================================================
# 可微 STE 版模型：结构与原 DeepFCA / DeepAEFCA 完全一致，仅把损失换成 STE
# ======================================================================
class DeepFCA_STE(nn.Module):
    """用 fca_loss_ste 替换 fca_loss，其余结构不变"""

    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size,
                 device=torch.device('cpu'), tau=0.05):
        super(DeepFCA_STE, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss_fn = fca_loss_ste(outdim_size, device, tau=tau)
        self.loss = self.loss_fn.loss

    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        return output1, output2


class DeepAEFCA_STE(nn.Module):
    """用 daefca_loss_ste 替换 daefca_loss，其余结构不变"""

    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size,
                 device=torch.device('cpu'), tau=0.025, lam=0.0):
        super(DeepAEFCA_STE, self).__init__()
        self.encoder1 = MlpNet(layer_sizes1, input_size1).double()
        self.encoder2 = MlpNet(layer_sizes2, input_size2).double()

        self.decoder1 = MlpNet(layer_sizes1[-2::-1] + [input_size1], layer_sizes1[-1]).double()
        self.decoder2 = MlpNet(layer_sizes2[-2::-1] + [input_size2], layer_sizes2[-1]).double()

        self.loss_fn = daefca_loss_ste(outdim_size, device, tau=tau, lam=lam)
        self.loss = self.loss_fn.loss

    def forward(self, x1, x2):
        output1 = self.encoder1(x1)
        output2 = self.encoder2(x2)

        re1 = self.decoder1(output1)
        re2 = self.decoder2(output2)

        return output1, output2, re1, re2
