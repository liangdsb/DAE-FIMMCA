"""
FIMMCA 度量学习（方案 A）：监督对比 + 修好的软 FIMMCA
=====================================================

思路
----
用「软 FIMMCA」作为样本间的相似度度量，监督对比损失让「同类样本的 FIMMCA 大、异类样本的 FIMMCA 小」；
训练后冻结编码器，用「硬 FIMMCA-1NN」评测。

这是单视图、带标签的方法（用数据集的 view1 + 标签做监督对比），不是论文原版的多视图 DAE-FIMMCA。
核心点：软 FIMMCA 代理经过两处修复后（① 分母用软 |X| 归一化 ② 拆分三个 tau），
其判别性从 0.003 恢复到接近硬 FIMMCA 的 0.061，因此能作为可微的训练信号。

运行
----
    # 默认跑 USPS（3 seed、20 轮、全量评测）
    python fimdmf_core.py

    # 指定数据集（自动带出正确的路径和输入维度）
    python fimdmf_core.py --dataset mnist
    python fimdmf_core.py --dataset coil100 --epochs 15

    # 快速测试（1 seed、3 轮、跳过基线）
    python fimdmf_core.py --dataset usps --seeds 0 --epochs 3 --no-baseline

    # 大数据集限制评测规模（MNIST/Fashion/CIFAR10 全量评测很慢，建议限制）
    python fimdmf_core.py --dataset mnist --eval-train 5000 --eval-test 2000

可用数据集: usps, mnist, fashion, coil100, cifar10, cmu, yaleb
"""
import gzip, time, sys, logging, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
from fimdmf_ops import soft_rank, _normalize_rank, soft_interval
from fimdmf_fast import fimdmf_pair_fast


# ==================== 数据集注册表（按需增删）====================
_BASE = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA"
DATASETS = {
    # 名称: (数据文件(view1), 输入维度)
    'usps':    dict(path=_BASE + r"\test_with_usps\usps_view1_speckle.gz",              in_dim=256),
    'mnist':   dict(path=_BASE + r"\test_with_noisymnist\my_noisymnist_view1_white_mask.gz", in_dim=784),
    'fashion': dict(path=_BASE + r"\test_with_fashionmnist\fashionmnist_view1.gz",       in_dim=784),
    'coil100': dict(path=_BASE + r"\test_with_coil100\coil_100_view1_speckle.gz",        in_dim=1024),
    'cifar10': dict(path=_BASE + r"\test_with_cifar10\cifar10_view1.gz",                 in_dim=1024),
    'cmu':     dict(path=_BASE + r"\test_cmu\cmu_lu_view1_white_mask.gz",                in_dim=1024),
    'yaleb':   dict(path=_BASE + r"\test_extended\pie_view1_white_mask_fmmca.gz",        in_dim=2304),
}

# ==================== 训练配置（按需修改）====================
OUT_DIM = 100       # 隐层维度（即 FIMMCA 作用向量的长度）
BATCH_SIZE = 64
LR = 1e-3
TAU_RANK = 0.5      # soft_rank 软度（中等，保证梯度）
TAU_BLOCK = 0.02    # soft_interval 软度（小，保证块判别性，实测 0.02 判别性≈硬 FIMMCA）
TAU_MINMAX = 0.05   # soft min/max 软度（小，保留"瓶颈"结构）
TAU_C = 0.1         # 监督对比损失的温度
# =============================================================


# ==================== pairwise 软 FIMMCA（训练用）====================
def fimdmf_pair(Z1, Z2, tau_rank=TAU_RANK, tau_block=TAU_BLOCK, tau_minmax=TAU_MINMAX):
    """Z1:(M,p), Z2:(N,p) -> S:(M,N), S[i,j]=软FIMMCA(Z1[i], Z2[j])

    训练统一走向量化实现 (fimdmf_fast.fimdmf_pair_fast):
    与原逐块循环版数值等价(已验证差异 ~1e-16, 即机器精度), 但提速约 2.9 倍。
    原实现保留为 fimdmf_pair_loop, 供数值校验使用。
    """
    return fimdmf_pair_fast(Z1, Z2, tau_rank, tau_block, tau_minmax)


def fimdmf_pair_loop(Z1, Z2, tau_rank=TAU_RANK, tau_block=TAU_BLOCK, tau_minmax=TAU_MINMAX):
    """逐块循环, 未向量化"""
    p = Z1.shape[-1]
    Z1 = (Z1 - Z1.mean(-1, keepdim=True)) / (Z1.std(-1, keepdim=True) + 1e-6)
    Z2 = (Z2 - Z2.mean(-1, keepdim=True)) / (Z2.std(-1, keepdim=True) + 1e-6)
    R1 = _normalize_rank(soft_rank(Z1, tau_rank))   # (M, p)
    R2 = _normalize_rank(soft_rank(Z2, tau_rank))   # (N, p)
    u_k = []
    for k in range(5, p // 2 + 1):
        nb = p // k
        ov = []
        for v in range(nb):
            lo = (v * k) / (p - 1.0)
            hi = ((v + 1) * k - 1.0) / (p - 1.0)
            inx = soft_interval(R1, lo, hi, tau_block)   # (M, p)
            iny = soft_interval(R2, lo, hi, tau_block)   # (N, p)
            overlap = inx @ iny.t()                       # (M, N) 软共有元素数
            size_x = inx.sum(dim=-1, keepdim=True)        # (M, 1) 软 |X|
            ov.append(overlap / (size_x + 1e-6))          # (M, N) 归一化包含度 ∈ [0,1]
        ov = torch.stack(ov, dim=0)                       # (nb, M, N)
        u_k.append(-tau_minmax * torch.logsumexp(-ov / tau_minmax, dim=0))   # (M, N) 软最小 over blocks
    u_k = torch.stack(u_k, dim=0)                         # (nK, M, N)
    return tau_minmax * torch.logsumexp(u_k / tau_minmax, dim=0)   # (M, N) 软最大 over k


# ==================== 监督对比损失 ====================
def supervised_contrastive_loss(S, labels, tau_c=TAU_C):
    """S:(N,N) 相似度, labels:(N,) -> 标量"""
    labels = labels.to(S.device)
    N = S.shape[0]
    eye = torch.eye(N, device=S.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1 - eye)
    S = (S / tau_c) - S.max(dim=1, keepdim=True).values.detach()
    expS = torch.exp(S) * (1 - eye)
    denom = expS.sum(dim=1, keepdim=True)
    pos_sum = (expS * pos_mask).sum(dim=1, keepdim=True)
    loss = -torch.log((pos_sum + 1e-8) / (denom + 1e-8))
    n_pos = pos_mask.sum(dim=1)
    return loss[n_pos > 0].mean()


# ==================== 编码器 ====================
class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==================== 硬 FIMMCA-1NN 评测（向量化）====================
def rank_matrix(X):
    """X:(n,p) -> (n,p), rank_X[m,j] = 元素 j 在 X[m] 升序中的位置"""
    r = np.argsort(X, axis=1)
    rank = np.empty_like(r)
    np.put_along_axis(rank, r, np.arange(X.shape[1])[None, :], axis=1)
    return rank


def fimdmf_batch_umax(a, rank_B):
    """a:(p,) 测试样本, rank_B:(n,p) 训练秩矩阵 -> (n,) 每个训练样本与 a 的硬 FIMMCA。
    用 one-hot 矩阵乘法一次性算所有训练样本的块交集，与逐对 get_umax 精确等价（已验证）。"""
    p = a.shape[0]
    ra = np.argsort(a)
    rank_a = np.empty(p, dtype=rank_B.dtype)
    rank_a[ra] = np.arange(p)
    n = rank_B.shape[0]
    umax = np.zeros(n, dtype=np.float32)
    for k in range(5, p // 2 + 1):
        nb = p // k
        a_block = rank_a // k                                   # (p,)
        b_block = rank_B // k                                   # (n, p)
        B_same = (b_block == a_block[None, :]).astype(np.float32)            # (n, p)
        A = (a_block[:, None] == np.arange(nb)[None, :]).astype(np.float32)  # (p, nb)
        counts = B_same @ A                                     # (n, nb) 每块交集数
        u_k = counts.min(axis=1) / k                            # (n,)
        umax = np.maximum(umax, u_k)
    return umax


def prepare_train_blocks(rank_B):
    """【优化用】一次性预计算各 k 下的训练块索引, 供所有测试样本复用。

    原实现在每个测试样本 × 每个 k 上都要算一次 (n,p) 的整数除法 rank_B//k,
    这是评测的主要开销之一。这里只算一次, 全测试集共享。
    返回值: [(k, 块索引矩阵 (n,p)), ...]
    """
    p = rank_B.shape[1]
    n = rank_B.shape[0]
    nb_max = p // 5
    dtype = np.int8 if nb_max < 128 else np.int16      # 块号范围 [0, p//5]
    return [(k, (rank_B // k).astype(dtype)) for k in range(5, p // 2 + 1)]


def fimdmf_batch_umax_fast(rank_a, blocks, p):
    """【优化版】a 的秩向量 + 预计算的训练块 -> (n,) 每个训练样本与 a 的硬 FIMMCA。

    数学上与 fimdmf_batch_umax 等价:
        counts[i,v] = #{ j : trainblock[i,j]==v 且 testblock[j]==v }
    但避免了逐块的 (n,p) float32 类型转换与 (n,p)@(p,nb) 矩阵乘,
    改用按块计数, 并对所有测试样本复用训练块索引。
    """
    n = blocks[0][1].shape[0]
    umax = np.zeros(n, dtype=np.float32)
    for k, B in blocks:
        nb = p // k
        A = (rank_a // k).astype(B.dtype)              # (p,) 测试块索引
        counts = np.empty((n, nb), dtype=np.int32)
        for v in range(nb):
            sel = (A == v)                              # 落在测试块 v 的位置
            counts[:, v] = np.count_nonzero(B[:, sel] == v, axis=1)
        np.maximum(umax, counts.min(axis=1) / k, out=umax)
    return umax


def classify(z_train, y_train, z_test, y_test, n_train=None, n_test=None):
    """硬 FIMMCA-1NN 评测：每个测试样本找 FIMMCA 最大的训练样本，取标签。
    n_train/n_test 为 None 时用全量，否则用前 N 个（用于大数据集快速评测）。
    已向量化: 训练块索引预计算一次并全测试集复用; 测试秩批量预计算。"""
    if n_train is None:
        n_train = z_train.shape[0]
    if n_test is None:
        n_test = z_test.shape[0]
    rank_B = rank_matrix(z_train[:n_train])
    rank_Z = rank_matrix(z_test[:n_test])          # 批量算测试秩, 避免逐个 argsort
    p = rank_B.shape[1]
    blocks = prepare_train_blocks(rank_B)          # 只算一次
    count = 0
    for t in range(n_test):
        sims = fimdmf_batch_umax_fast(rank_Z[t], blocks, p)
        if y_test[t] == y_train[np.argmax(sims)]:
            count += 1
    return count / n_test


def classify_loop(z_train, y_train, z_test, y_test, n_train=None, n_test=None):
    """【原版实现, 保留作参考/校验】逐个测试样本调用未优化的 fimdmf_batch_umax"""
    if n_train is None:
        n_train = z_train.shape[0]
    if n_test is None:
        n_test = z_test.shape[0]
    rank_B = rank_matrix(z_train[:n_train])
    count = 0
    for n in range(n_test):
        sims = fimdmf_batch_umax(z_test[n], rank_B)
        if y_test[n] == y_train[np.argmax(sims)]:
            count += 1
    return count / n_test


# ==================== 训练 ====================
def train(encoder, train_x, train_y_t, epochs, batch_size, device):
    opt = torch.optim.Adam(encoder.parameters(), lr=LR)
    N = train_x.shape[0]
    for epoch in range(epochs):
        encoder.train()
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            zb = encoder(train_x[idx])
            S = fimdmf_pair(zb, zb)
            loss = supervised_contrastive_loss(S, train_y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"[epoch {epoch + 1}/{epochs}] loss={tot / max(nb, 1):.4f}", flush=True)


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="FIMMCA 度量学习/监督对比")
    parser.add_argument('--dataset', type=str, default='fashion', choices=list(DATASETS.keys()),
                        help="数据集: " + ', '.join(DATASETS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help="随机种子列表")
    parser.add_argument('--epochs', type=int, default=20, help="训练轮数")
    parser.add_argument('--no-baseline', action='store_true', help="跳过基线（随机初始化）评测")
    parser.add_argument('--eval-train', type=int, default=None, help="评测用训练参考样本数(默认全量)")
    parser.add_argument('--eval-test', type=int, default=None, help="评测用测试样本数(默认全量)")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"数据集: {args.dataset} | 输入维度: {cfg['in_dim']} | 设备: {device}", flush=True)

    d1 = load_pickle(gzip.open(cfg['path'], 'rb'))
    train_x = d1[0][0].astype(np.float32)
    test_x = d1[2][0].astype(np.float32)
    y_train = d1[0][1].astype(np.int64)
    y_test = d1[2][1].astype(np.int64)
    print(f"数据: train={train_x.shape}, test={test_x.shape}", flush=True)

    train_x_t = torch.tensor(train_x).to(device)
    train_y_t = torch.tensor(y_train)
    test_x_t = torch.tensor(test_x).to(device)

    def encode(encoder):
        encoder.eval()
        with torch.no_grad():
            return encoder(train_x_t).cpu().numpy(), encoder(test_x_t).cpu().numpy()

    # 基线（随机初始化）
    acc_base = None
    if not args.no_baseline:
        torch.manual_seed(args.seeds[0])
        enc0 = Encoder(cfg['in_dim'], OUT_DIM).to(device)
        t0 = time.time()
        z_train, z_test = encode(enc0)
        acc_base = classify(z_train, y_train, z_test, y_test, args.eval_train, args.eval_test)
        print(f"基线(随机初始化): {acc_base * 100:.2f}%  (评测 {time.time() - t0:.0f}s)", flush=True)

    # 多 seed 训练 + 评测
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        encoder = Encoder(cfg['in_dim'], OUT_DIM).to(device)
        print(f"\n----- seed = {seed} -----", flush=True)
        train(encoder, train_x_t, train_y_t, args.epochs, BATCH_SIZE, device)
        t0 = time.time()
        z_train, z_test = encode(encoder)
        acc = classify(z_train, y_train, z_test, y_test, args.eval_train, args.eval_test)
        results.append(acc)
        print(f"seed={seed}: {acc * 100:.2f}%  (评测 {time.time() - t0:.0f}s)", flush=True)

    # 汇总
    arr = np.array(results)
    print("\n" + "=" * 50, flush=True)
    print(f"数据集: {args.dataset}", flush=True)
    print(f"最终: {arr.mean() * 100:.2f}% ± {arr.std() * 100:.2f}%  (seeds={[f'{x * 100:.2f}' for x in results]})", flush=True)
    if acc_base is not None:
        print(f"基线: {acc_base * 100:.2f}%", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
