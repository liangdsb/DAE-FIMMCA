"""
可微模糊包含单调决策隶属函数 (Differentiable FIMMCA / DFIMDMF)

背景
----
原实现 (objectives.py / utilssss.py) 里的 FIMMCA 由四个不可微算子组成:
  1. argsort                 (feature_sort)
  2. 集合交集计数             (np.intersect1d + len)
  3. 区间取最小              (min)
  4. 对 k 取最大             (max)
因此原 fca_loss / daefca_loss 里 H1/H2 被 .detach() 后转 numpy,
返回的 corr 是与网络无关的新张量, 导致 loss.backward() 对编码器是零梯度。

本文件提供:
  * 可微(软)版本  dfimdmf()          —— 纯 torch, 梯度可回传
  * 精确(硬)版本  hard_fimdmf()      —— numpy, 与原 objectives.get_umax 数值一致
  * STE 版本      fimdmf_ste()       —— 前向用精确值, 反向用软版本梯度
  * 损失类        fca_loss_ste / daefca_loss_ste, 接口与原 fca_loss / daefca_loss 一致

用法
----
    loss_fn = fca_loss_ste(outdim_size, device, tau=0.05)
    loss = loss_fn.loss(H1, H2)     # 与原 fca_loss.loss(H1, H2) 相同签名
    loss.backward()                 # 此时梯度可回传到编码器
    loss_fn.step_tau()              # 可选: 每个 epoch 退火, 逐步逼近硬 FIMMCA

注意
----
  * 软秩已归一化到 [0,1]; 实测 tau 越小软值越贴近硬值、梯度越大,
    0.02~0.08 区间较佳, 默认取 0.05, 可用 step_tau() 退火到 0.02。
  * 输入 x, y 形状为 (B, p), p = 隐层维度(如 100)。
"""

import numpy as np
import torch
import torch.nn.functional as F


# ======================================================================
# 一、可微（软）实现
# ======================================================================
def soft_rank(x, tau=1.0):
    """
    可微软排序: rank[i] ≈ 1 + #{j : x[j] < x[i]}
    x: (B, p) -> (B, p)
    """
    d = x.unsqueeze(-1) - x.unsqueeze(-2)      # (B, p, p), d[..., i, j] = x_i - x_j
    less = torch.sigmoid(d / tau)              # 软指示 x_i > x_j
    rank = less.sum(dim=-1) + 0.5              # 自比较项贡献 0.5, 补回偏移
    return rank


def _normalize_rank(rank):
    """把软秩(约 [0.5, p-0.5]) 归一化到 [0,1], 使 soft_interval 与 soft_rank 同量纲"""
    p = rank.shape[-1]
    return (rank - 0.5) / max(p - 1.0, 1.0)


def soft_interval(rank, lo, hi, tau):
    """可微"是否落在区间 [lo, hi]"指示器; rank/lo/hi 均已归一化到 [0,1]"""
    return torch.sigmoid((rank - lo) / tau) * torch.sigmoid((hi - rank) / tau)


def soft_inclusion(rx, ry, k, tau):
    """
    固定分区大小 k 下的软包含度: u_v = sum_i inx_v[i] * iny_v[i] / k
    rx, ry 为归一化后的软秩 (∈ [0,1]); 只用完整块 (p // k 个), 丢弃余数
    """
    p = rx.shape[-1]
    nblocks = p // k
    u_v = []
    for v in range(nblocks):
        lo = (v * k) / (p - 1.0)              # 归一化块边界
        hi = ((v + 1) * k - 1.0) / (p - 1.0)
        inx = soft_interval(rx, lo, hi, tau)  # (B, p)
        iny = soft_interval(ry, lo, hi, tau)
        overlap = (inx * iny).sum(dim=-1)     # (B,) 软"共有元素数"
        u_v.append(overlap / k)
    return torch.stack(u_v, dim=0)            # (nblocks, B)


def dfimdmf(x, y, tau=1.0, k_list=None):
    """
    可微 FIMMCA: mean_batch [ softmax_k [ softmin_v u_v ] ]
    x, y: (B, p) -> 标量
    """
    p = x.shape[-1]
    if k_list is None:
        k_list = list(range(5, p // 2 + 1))
    if not k_list:
        k_list = [1]

    # 软秩归一化到 [0,1], 避免 soft_interval 的 sigmoid 在 rank(1~100) 尺度下饱和为零梯度
    rx = _normalize_rank(soft_rank(x, tau))
    ry = _normalize_rank(soft_rank(y, tau))

    u_k = []
    for k in k_list:
        u_v = soft_inclusion(rx, ry, k, tau)                      # (nblocks, B)
        # 软最小 over blocks: -tau * logsumexp(-u_v / tau)
        u_k.append(-tau * torch.logsumexp(-u_v / tau, dim=0))     # (B,)
    u_k = torch.stack(u_k, dim=0)                                 # (nK, B)
    # 软最大 over k: tau * logsumexp(u_k / tau)
    u_max = tau * torch.logsumexp(u_k / tau, dim=0)               # (B,)
    return u_max.mean()                                           # 标量


# ======================================================================
# 二、精确（硬）实现 —— 数值与原 objectives.get_umax 一致
# ======================================================================
def _hard_u(sx, sy, k):
    u_i = []
    for i in range(0, len(sx), k):
        if i + k > len(sx):
            break
        inter = len(np.intersect1d(sx[i:i + k], sy[i:i + k]))
        u_i.append(float(inter / k))
    return min(u_i)


def _hard_umax(x, y):
    sx = np.argsort(x)
    sy = np.argsort(y)
    umax = 0.0
    for k in range(5, int(len(x) / 2) + 1):
        u0 = _hard_u(sx, sy, k)
        if u0 > umax:
            umax = u0
    return umax


def hard_fimdmf(x, y):
    """
    精确 FIMMCA: x, y 为 (B, p) numpy -> 标量(批平均)
    数值与原 get_view1_and_view2_average_u_x 一致
    """
    vals = [_hard_umax(x[n], y[n]) for n in range(x.shape[0])]
    return float(np.mean(vals))


# ======================================================================
# 三、STE: 前向精确 / 反向软梯度
# ======================================================================
def fimdmf_ste(x, y, tau):
    """
    STE(直通估计): 前向返回精确硬 FIMMCA 值, 反向沿软 dfimdmf 的梯度回传。
    利用 `soft - soft.detach()` 前向为 0、反向贡献 soft 梯度的技巧, 避免在
    backward 里嵌套调用 autograd.grad(会触发引擎重入问题)。
    x, y: (B, p) -> 标量
    """
    hard = hard_fimdmf(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    hard = torch.as_tensor(hard, device=x.device, dtype=x.dtype)
    soft = dfimdmf(x, y, tau=tau)
    return hard + (soft - soft.detach())


# ======================================================================
# 四、损失类 —— 直接替换原 fca_loss / daefca_loss
# ======================================================================
class fca_loss_ste:
    """与原 fca_loss 相同的 .loss(H1, H2) 接口, 但梯度可回传"""

    def __init__(self, outdim_size, device, tau=0.05, anneal=0.98):
        self.outdim_size = outdim_size
        self.device = device
        self.tau = tau
        self.anneal = anneal

    def loss(self, H1, H2):
        corr = fimdmf_ste(H1, H2, self.tau)
        return -corr

    def step_tau(self):
        self.tau = max(self.tau * self.anneal, 0.02)


class daefca_loss_ste:
    """与原 daefca_loss 相同的 .loss(H1, H2, x1, x2, re1, re2) 接口"""

    def __init__(self, outdim_size, device, tau=0.05, anneal=0.98, lam=0.005):
        self.outdim_size = outdim_size
        self.device = device
        self.tau = tau
        self.anneal = anneal
        self.lam = lam          # 重建项权重, 0=纯 FIMMCA; 建议 0.001~0.1

    def reconstruction_loss(self, x, r):
        return F.mse_loss(x, r)

    def loss(self, H1, H2, x1, x2, re1, re2):
        # FIMMCA 相关度 (最大化)
        corr = fimdmf_ste(H1, H2, self.tau)
        # 可选重建项 (最小化), 防止特征塌缩、保留判别信息; lam=0 时退化为纯 FIMMCA
        recon = self.reconstruction_loss(x1, re1) + self.reconstruction_loss(x2, re2)
        return -corr + self.lam * recon

    def step_tau(self):
        self.tau = max(self.tau * self.anneal, 0.02)
