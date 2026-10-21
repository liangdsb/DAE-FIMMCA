import torch
import numpy as np
import torch.nn.functional as F
from utilssss import u, feature_sort


class fca_loss():
    def __init__(self, outdim_size, device):
        self.outdim_size = outdim_size
        self.device = device


    def get_umax(self, data1, data2):
        sorted_index1, sorted_index2 = feature_sort(data1, data2)
        umax = 0
        k_best = 0
        for k in range(5, int(len(data1) / 2) + 1):  # k = 区间内元素个数
            u0 = u(sorted_index1, sorted_index2, k)
            if u0 > umax:
                umax = u0
                k_best = k
        return umax

    def get_view1_and_view2_average_u_x(self, view1, view2):
        # a = []  # 用于装50000张图像之间的包含度
        b = []  # 用于装一张view1和一张view2之间的包含度
        for n in range(view2.shape[0]):
            b.append(self.get_umax(view1[n, :], view2[n, :]))

            # 求b的平均值
            b_sum = sum(b)
            b_len = len(b)
            b_avg = b_sum / b_len
            # a.append(b_avg)
        return b_avg

    def loss(self, H1, H2):
        H1 = H1.detach().cpu().numpy()
        H2 = H2.detach().cpu().numpy()
        corr = torch.tensor(self.get_view1_and_view2_average_u_x(H1, H2))
        return -corr


class daefca_loss():
    def __init__(self, outdim_size, device):
        self.outdim_size = outdim_size
        self.device = device


    def reconstruction_loss(self, x, r):
        # return F.l1_loss(r, x)
        return F.mse_loss(x, r)

    def get_umax(self, data1, data2):
        sorted_index1, sorted_index2 = feature_sort(data1, data2)
        umax = 0
        k_best = 0
        for k in range(5, int(len(data1) / 2) + 1):  # k = 区间内元素个数
            u0 = u(sorted_index1, sorted_index2, k)
            if u0 > umax:
                umax = u0
                k_best = k
        return umax

    def get_view1_and_view2_average_u_x(self, view1, view2):
        # a = []  # 用于装50000张图像之间的包含度
        b = []  # 用于装一张view1和一张view2之间的包含度
        for n in range(view2.shape[0]):
            b.append(self.get_umax(view1[n, :], view2[n, :]))

            # 求b的平均值
            b_sum = sum(b)
            b_len = len(b)
            b_avg = b_sum / b_len
            # a.append(b_avg)
        return b_avg

    def loss(self, H1, H2, x1, x2, re1, re2):
        H1 = H1.detach().cpu().numpy()
        H2 = H2.detach().cpu().numpy()
        # corr = self.get_view1_and_view2_average_u_x(H1, H2)
        # corr = self.get_view1_and_view2_average_u_x(H1, H2) + (self.reconstruction_loss(x1, re1) + self.reconstruction_loss(x2, re2)) * 0.001
        corr = torch.tensor(self.get_view1_and_view2_average_u_x(H1, H2))
        return -corr


class daefca02_loss():
    def __init__(self, outdim_size, device):
        self.outdim_size = outdim_size
        self.device = device
        # print(device)

    def reconstruction_loss(self, x, r):
        # return F.l1_loss(r, x)
        return F.mse_loss(x, r)

    def loss(self, x1, x2, re1, re2):
        # corr = self.get_view1_and_view2_average_u_x(H1, H2)
        corr = (self.reconstruction_loss(x1, re1) + self.reconstruction_loss(x2, re2)) * 0.001
        # corr = torch.tensor(self.get_view1_and_view2_average_u_x(H1, H2))
        return -corr


class convcca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()  # H1和H2转置
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)  # H1矩阵的行

        m = H1.size(1)  # H2矩阵的列
        #         print(H1.size())

        # H1bar和H2bar是深度神经网络的两个输出
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)  # H1.mean(dim=1)： 压缩第二维度的数据 unsqueeze(dim=1)：扩张第二维度的维度
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())  # torch.matmul做乘法
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO='U')
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO='U')
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]  # torch.gt(a,b)得到a中比b中元素大的位置
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
        #         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r1).to(
                self.device))  # regularization for more stability
            U, V = torch.linalg.eigh(trace_TT, UPLO='U')
            U = torch.where(U > eps, U, (torch.ones(U.shape).double() * eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr
