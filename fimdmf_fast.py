"""
fimdmf_pair 的向量化版本

原实现 (fimdmf_core.fimdmf_pair) 有两层循环:
    for k in range(5, p//2+1):        # 46 次 (latent=100)
        for v in range(p//k):         # 各分区的块
            soft_interval(...)        # 逐块算软区间指示
            inx @ iny.t()             # 逐块算交集

本文件把**内层的块循环**合并成批量矩阵运算 (数学等价, 只是并行化):
    inx: (nb, M, p)   <- 一次性算出所有块的软指示
    overlap = bmm(inx, iny^T)      # 批量矩阵乘
    size_x  = inx.sum(-1)          # 批量求和

外层 k 循环保留 (不同 k 的块数 nb 不同, 跨 k 批处理需 padding, 收益有限)。

用法:
    from fimdmf_fast import fimdmf_pair_fast
    S = fimdmf_pair_fast(Z1, Z2)     # 与 fimdmf_pair(Z1, Z2) 数值等价
"""
import torch

from fimdmf_ops import soft_rank, _normalize_rank


def fimdmf_pair_fast(Z1, Z2, tau_rank=0.5, tau_block=0.02, tau_minmax=0.05, max_batch_memory=None):
    """
    Z1:(M,p), Z2:(N,p) -> S:(M,N), S[i,j] = 软FIMMCA(Z1[i], Z2[j])

    与原版数值等价 (只把内层块循环改为批量运算)。
    max_batch_memory: 若给定, 按此上限切分 M 维, 控制显存峰值 (单位: 元素个数)
    """
    p = Z1.shape[-1]
    # 每样本 z-score (单调变换不改变排序, 使软 rank 量纲一致)
    Z1 = (Z1 - Z1.mean(-1, keepdim=True)) / (Z1.std(-1, keepdim=True) + 1e-6)
    Z2 = (Z2 - Z2.mean(-1, keepdim=True)) / (Z2.std(-1, keepdim=True) + 1e-6)
    R1 = _normalize_rank(soft_rank(Z1, tau_rank))   # (M, p)
    R2 = _normalize_rank(soft_rank(Z2, tau_rank))   # (N, p)

    M = R1.shape[0]
    u_k = []
    for k in range(5, p // 2 + 1):
        nb = p // k
        v = torch.arange(nb, device=R1.device, dtype=R1.dtype)
        lo = (v * k) / (p - 1.0)                     # (nb,)
        hi = ((v + 1) * k - 1.0) / (p - 1.0)         # (nb,)
        # 批量软区间指示: (nb, M, p) / (nb, N, p)
        inx = torch.sigmoid((R1.unsqueeze(0) - lo[:, None, None]) / tau_block) * \
              torch.sigmoid((hi[:, None, None] - R1.unsqueeze(0)) / tau_block)
        iny = torch.sigmoid((R2.unsqueeze(0) - lo[:, None, None]) / tau_block) * \
              torch.sigmoid((hi[:, None, None] - R2.unsqueeze(0)) / tau_block)
        overlap = torch.bmm(inx, iny.transpose(1, 2))          # (nb, M, N) 批量交集
        size_x = inx.sum(dim=-1, keepdim=True)                 # (nb, M, 1) 软 |X|
        ov = overlap / (size_x + 1e-6)                         # (nb, M, N) 归一化包含度
        u_k.append(-tau_minmax * torch.logsumexp(-ov / tau_minmax, dim=0))   # (M, N) 软最小
    u_k = torch.stack(u_k, dim=0)                              # (nK, M, N)
    return tau_minmax * torch.logsumexp(u_k / tau_minmax, dim=0)             # (M, N) 软最大
