"""
FIMMCA 多视图自监督对比
==========================================

思路
----
用两个视图 view1、view2（同一张图的不同噪声/变换），共用同一个编码器 f。
用软 FIMMCA 作为「样本间相似度」，InfoNCE 自监督对比损失：
  * 正样本对：view1[i] 与 view2[i]（同一张图）→ FIMMCA 要尽量大；
  * 负样本对：view1[i] 与 view2[j≠i]（不同图）→ FIMMCA 要尽量小。
负样本防止了"所有样本对齐到一样"的塌缩（这是原版跨视图 FIMMCA 塌缩的原因）。
训练后冻结编码器，用硬 FIMMCA-1NN 在 view1 上评测。

重要: 默认「共享编码器」(SimCLR 式)
--------------------------------
实测两视图若用相互独立的随机编码器，同一张图的跨视图正样本与负样本几乎无差别
(硬 FIMMCA gap 仅 0.0057, 信噪比 0.15), InfoNCE 的 softmax 近乎均匀, 训不动。
改成共享编码器后 gap 提升到 0.059 (信噪比 1.5, 强 10~23 倍), 训练才有信号。
如需复现旧的独立编码器版本, 加 --separate。

这是「多视图 + 自监督（无标签）+ FIMMCA」的版本

"""
import gzip, time, sys, logging, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
# 软 FIMMCA / 编码器 / 向量化评测
from fimdmf_core import fimdmf_pair, Encoder, rank_matrix, fimdmf_batch_umax, classify


def enc_labels(y):
    """字符串标签(如 cmu 的 '01'..'68')转整数编码; 数值标签原样返回"""
    if y.dtype.kind in ('U', 'S', 'O'):
        _, inv = np.unique(y, return_inverse=True)
        return inv.astype(np.int64)
    return y.astype(np.int64)


# ==================== 数据集注册表（view1 + view2）====================
_BASE = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA"
DATASETS = {
    # 名称: (view1 文件, view2 文件, 输入维度)
    'usps':    dict(v1=_BASE + r"\test_with_usps\usps_view1.gz",             v2=_BASE + r"\test_with_usps\usps_view2.gz",             in_dim=256),
    'mnist':   dict(v1=_BASE + r"\test_with_noisymnist\my_noisymnist_view1.gz", v2=_BASE + r"\test_with_noisymnist\my_noisymnist_view2.gz", in_dim=784),
    'fashion': dict(v1=_BASE + r"\test_with_fashionmnist\fashionmnist_view1.gz",      v2=_BASE + r"\test_with_fashionmnist\fashionmnist_view2.gz",      in_dim=784),
    'coil100': dict(v1=_BASE + r"\test_with_coil100\coil_100_view1_white_mask.gz",       v2=_BASE + r"\test_with_coil100\coil_100_view2_white_mask.gz",       in_dim=1024),
    'cifar10': dict(v1=_BASE + r"\test_with_cifar10\cifar10_view1.gz",                v2=_BASE + r"\test_with_cifar10\cifar10_view2.gz",                in_dim=1024),
    'cmu':     dict(v1=_BASE + r"\test_cmu\cmu_lu_view1_white_mask.gz",               v2=_BASE + r"\test_cmu\cmu_lu_view2_white_mask.gz",               in_dim=1024),
    'yaleb':   dict(v1=_BASE + r"\test_extended\pie_view1_white_mask_fmmca.gz",       v2=_BASE + r"\test_extended\pie_view2_white_mask_fmmca.gz",       in_dim=2304),
}

# ==================== 训练配置 ====================
OUT_DIM = 100
BATCH_SIZE = 64
LR = 1e-4
TAU_C = 0.1         # InfoNCE 对比损失温度
# (软 FIMMCA 的 tau_rank/tau_block/tau_minmax 沿用 fimdmf_pair 的默认值 0.5/0.02/0.05)
# ================================================


# ==================== InfoNCE 自监督对比损失 ====================
def infonce_loss(S, tau_c=TAU_C):
    """S:(N,N) 跨视图软 FIMMCA 相似度, 对角线是正样本(同一图的两视图)。
    loss = -mean_i log( exp(S[i,i]/τ) / sum_j exp(S[i,j]/τ) )"""
    N = S.shape[0]
    S = (S / tau_c) - S.max(dim=1, keepdim=True).values.detach()   # 数值稳定
    expS = torch.exp(S)
    pos = expS.diag()                       # 正样本(对角线)
    denom = expS.sum(dim=1)                 # 分母含对角线
    return -torch.log((pos + 1e-8) / (denom + 1e-8)).mean()


# ==================== 训练（两个视图共用一个编码器 f）====================
def train(f1, f2, v1, v2, epochs, batch_size, device, lr=LR):
    # 共享编码器时 f2 is f1，参数只取一份，避免 Adam 重复注册参数组
    params = list(f1.parameters()) if (f2 is f1) else list(f1.parameters()) + list(f2.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    N = v1.shape[0]
    for epoch in range(epochs):
        f1.train()
        f2.train()
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            Z1 = f1(v1[idx])               # (B, outdim) view1 隐表示
            Z2 = f2(v2[idx])               # (B, outdim) view2 隐表示
            S = fimdmf_pair(Z1, Z2)        # (B, B) 跨视图软 FIMMCA
            loss = infonce_loss(S)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"[epoch {epoch + 1}/{epochs}] loss={tot / max(nb, 1):.4f}", flush=True)


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="FIMMCA 多视图自监督对比")
    parser.add_argument('--dataset', type=str, default='cmu', choices=list(DATASETS.keys()),
                        help="数据集: " + ', '.join(DATASETS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help="随机种子列表")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--lr', type=float, default=LR,
                        help="学习率(默认 1e-4;")
    parser.add_argument('--no-baseline', action='store_true', help="跳过基线评测")
    parser.add_argument('--eval-train', type=int, default=None, help="评测用训练参考样本数(默认全量)")
    parser.add_argument('--eval-test', type=int, default=None, help="评测用测试样本数(默认全量)")
    parser.add_argument('--separate', action='store_true',
                        help="两个视图用相互独立的编码器(默认共享同一编码器, 跨视图信号强 10 倍以上)")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"数据集: {args.dataset} | 输入维度: {cfg['in_dim']} | 设备: {device}", flush=True)

    d1 = load_pickle(gzip.open(cfg['v1'], 'rb'))
    d2 = load_pickle(gzip.open(cfg['v2'], 'rb'))
    # 统一展平成 (样本数, 特征维)：部分 view2 文件存的是 3D(如 28x28)，需 flatten 后与 view1 对齐
    v1_train = d1[0][0].astype(np.float32).reshape(d1[0][0].shape[0], -1)
    v2_train = d2[0][0].astype(np.float32).reshape(d2[0][0].shape[0], -1)
    v1_test = d1[2][0].astype(np.float32).reshape(d1[2][0].shape[0], -1)
    y_train = enc_labels(d1[0][1])
    y_test = enc_labels(d1[2][1])
    assert v1_train.shape == v2_train.shape, f"view1/view2 训练形状不一致: {v1_train.shape} vs {v2_train.shape}"
    print(f"数据: view1 train={v1_train.shape}, view2 train={v2_train.shape}, test={v1_test.shape}", flush=True)

    v1_train_t = torch.tensor(v1_train).to(device)
    v2_train_t = torch.tensor(v2_train).to(device)
    v1_test_t = torch.tensor(v1_test).to(device)

    def encode_view1(f1):
        f1.eval()
        with torch.no_grad():
            return f1(v1_train_t).cpu().numpy(), f1(v1_test_t).cpu().numpy()

    # 基线（f1 随机初始化）
    acc_base = None
    if not args.no_baseline:
        torch.manual_seed(args.seeds[0])
        f1 = Encoder(cfg['in_dim'], OUT_DIM).to(device)
        t0 = time.time()
        z_train, z_test = encode_view1(f1)
        acc_base = classify(z_train, y_train, z_test, y_test, args.eval_train, args.eval_test)
        print(f"基线(随机初始化): {acc_base * 100:.2f}%  (评测 {time.time() - t0:.0f}s)", flush=True)

    # 多 seed 训练 + 评测
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        f1 = Encoder(cfg['in_dim'], OUT_DIM).to(device)
        # 默认共享编码器: 同一网络处理两个视图, 正样本对(同图跨视图)才能凸显;
        # 独立编码器会把两视图映射成互不相关的编码, InfoNCE 几乎无信号(实测信噪比 0.15 vs 1.5)
        f2 = Encoder(cfg['in_dim'], OUT_DIM).to(device) if args.separate else f1
        mode = "独立编码器" if args.separate else "共享编码器"
        print(f"\n----- seed = {seed} ({mode}) -----", flush=True)
        train(f1, f2, v1_train_t, v2_train_t, args.epochs, BATCH_SIZE, device, args.lr)
        t0 = time.time()
        z_train, z_test = encode_view1(f1)
        acc = classify(z_train, y_train, z_test, y_test, args.eval_train, args.eval_test)
        results.append(acc)
        print(f"seed={seed}: {acc * 100:.2f}%  (评测 {time.time() - t0:.0f}s)", flush=True)

    # 汇总
    arr = np.array(results)
    print("\n" + "=" * 50, flush=True)
    print(f"数据集: {args.dataset} (多视图自监督对比)", flush=True)
    print(f"最终: {arr.mean() * 100:.2f}% ± {arr.std() * 100:.2f}%  (seeds={[f'{x * 100:.2f}' for x in results]})", flush=True)
    if acc_base is not None:
        print(f"基线: {acc_base * 100:.2f}%", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
