"""
全部方法 (不含 LRR-CCA) 在 5 个数据集上的计算时间

方法: CCA, RCCA, GRCCA, DCCA, DCCAE, ConvCCA, DAE-FIMMCA
口径: 训练(fit) + 编码(训练集 + 测试集) 的总耗时, 单位秒
配置: latent=10 (线性/浅层方法), latent=100 (DAE-FIMMCA); epochs=40
      训练集统一上限 8000 (MNIST 原始 50000, 否则不可行)

运行: python exp_4_time_cost.py
      python exp_4_time_cost.py --datasets usps coil100
"""
import gzip, sys, time, argparse, logging, math
import numpy as np
import torch

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
from exp_1_comparison import (run_cca, run_rcca, run_grcca,
                                run_dcca, run_dccae, run_convcca, run_ours)
from fimdmf_core import Encoder
from fimdmf_fast import fimdmf_pair_fast


def run_ours_fast(d, latent, epochs, bs, lr):
    """与 exp_1_comparison.run_ours 相同, 但用向量化的 fimdmf_pair_fast"""
    torch.manual_seed(0)
    x1 = torch.tensor(d['train_1']).to(DEV)
    x2 = torch.tensor(d['train_2']).to(DEV)
    xt = torch.tensor(d['test_1']).to(DEV)
    f = Encoder(x1.shape[1], latent).to(DEV)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    N = x1.shape[0]
    for ep in range(epochs):
        f.train(); perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            S = fimdmf_pair_fast(f(x1[idx]), f(x2[idx]))
            S = (S / 0.1) - S.max(1, keepdim=True).values.detach()
            e = torch.exp(S)
            loss = -torch.log((e.diag() + 1e-8) / (e.sum(1) + 1e-8)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    f.eval()
    with torch.no_grad():
        return f(x1).cpu().numpy(), f(xt).cpu().numpy()

_B = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA"
DATASETS = {
    'coil100': (_B + r"\test_with_coil100\coil_100_view1_white_mask.gz",   _B + r"\test_with_coil100\coil_100_view2_white_mask.gz"),
    'yaleb':   (_B + r"\test_extended\pie_view1_white_mask_fmmca.gz",      _B + r"\test_extended\pie_view2_white_mask_fmmca.gz"),
    'mnist':   (_B + r"\test_with_noisymnist\my_noisymnist_view1_white_mask.gz", _B + r"\test_with_noisymnist\my_noisymnist_view2_white_mask.gz"),
    'usps':    (_B + r"\test_with_usps\usps_view1_white_mask.gz",          _B + r"\test_with_usps\usps_view2_white_mask.gz"),
    'cmu':     (_B + r"\test_cmu\cmu_lu_view1_white_mask.gz",              _B + r"\test_cmu\cmu_lu_view2_white_mask.gz"),
}
LATENT = 10
OURS_LATENT = 100
MAX_TRAIN = 8000
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def enc_labels(y):
    if y.dtype.kind in ('U', 'S', 'O'):
        _, inv = np.unique(y, return_inverse=True)
        return inv.astype(np.int64)
    return y.astype(np.int64)


def build(name, max_train=MAX_TRAIN):
    p1, p2 = DATASETS[name]
    with gzip.open(p1, 'rb') as f:
        d1 = load_pickle(f)
    with gzip.open(p2, 'rb') as f:
        d2 = load_pickle(f)
    fl = lambda a: a.astype(np.float32).reshape(a.shape[0], -1)
    x1, x2 = fl(d1[0][0]), fl(d2[0][0])
    ytr = enc_labels(d1[0][1])
    x1t, yte = fl(d1[2][0]), enc_labels(d1[2][1])
    if x1.shape[0] > max_train:
        sel = np.random.RandomState(0).choice(x1.shape[0], max_train, replace=False)
        x1, x2, ytr = x1[sel], x2[sel], ytr[sel]
    return dict(train_1=x1, train_2=x2, test_1=x1t, test_2=x1t, train_y=ytr, test_y=yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, nargs='+', default=list(DATASETS.keys()))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-train', type=int, default=MAX_TRAIN)
    args = ap.parse_args()

    METHODS = ['CCA', 'RCCA', 'GRCCA', 'DCCA', 'DCCAE', 'ConvCCA', 'DAE-FIMMCA']
    print(f"计算时间 (秒) | epochs={args.epochs} | latent: 前6个={LATENT}, DAE-FIMMCA={OURS_LATENT}")
    print(f"口径: fit/训练 + transform/编码(训练集+测试集) | 训练集上限={args.max_train}\n")

    res = {}
    for name in args.datasets:
        data = build(name, args.max_train)
        indim = data['train_1'].shape[1]
        side = int(round(math.sqrt(indim)))
        print(f"--- {name} (train={data['train_1'].shape} test={data['test_1'].shape}) ---", flush=True)
        for m in METHODS:
            t0 = time.perf_counter()
            try:
                if m == 'CCA':
                    run_cca(data, LATENT)
                elif m == 'RCCA':
                    run_rcca(data, LATENT)
                elif m == 'GRCCA':
                    run_grcca(data, LATENT)
                elif m == 'DCCA':
                    run_dcca(data, LATENT, args.epochs, 64, args.lr, indim)
                elif m == 'DCCAE':
                    run_dccae(data, LATENT, args.epochs, 64, args.lr, indim)
                elif m == 'ConvCCA':
                    run_convcca(data, LATENT, args.epochs, 64, args.lr, side=side)
                elif m == 'DAE-FIMMCA':
                    run_ours_fast(data, OURS_LATENT, args.epochs, 64, args.lr)
                t = time.perf_counter() - t0
                res[(m, name)] = t
                print(f"  {m:12s} {t:9.2f} s", flush=True)
            except Exception as ex:
                import traceback
                print(f"  {m:12s} 失败: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()
        print(flush=True)

    print("=" * 74)
    print("汇总 (秒)")
    hdr = "".join(f"{d:>12}" for d in args.datasets)
    print(f"{'方法':<14s}{hdr}{'平均':>12}")
    print("-" * 74)
    for m in METHODS:
        vals = [res.get((m, d)) for d in args.datasets]
        row = "".join(f"{v:12.2f}" if v is not None else f"{'-':>12}" for v in vals)
        ok = [v for v in vals if v is not None]
        avg = f"{np.mean(ok):.2f}" if ok else "-"
        print(f"{m:<14s}{row}{avg:>12}")
    print("=" * 74)


if __name__ == "__main__":
    main()
