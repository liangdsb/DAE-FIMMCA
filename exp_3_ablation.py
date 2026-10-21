"""
消融实验：对应论文 Table 4 的第 2 行与第 3 行
=============================================

论文 Table 4 每行由两个开关组合而成：目标函数 (MSE / FIMMCA) × 分类策略 (SVM / FIMMCA-1NN)。
本脚本实现其中的两行：

  第 2 行  MSE 目标 + FIMMCA-1NN 分类
          两个编码器 + 两个解码器, 仅用重建误差训练, 不含任何跨视图相关项;
          训练完成后用 FIMMCA-1NN 分类。

  第 3 行  FIMMCA 目标 + SVM 分类
          两个编码器用多视图自监督对比 (FIMMCA 作相似度的 InfoNCE) 训练;
          训练完成后把隐表示交给线性 SVM 分类。

  第 3 行与论文完整模型(第 4 行)共用同一套训练流程, 唯一差别在于分类策略,
  因此二者的性能差可直接归因于分类策略本身。

统一设定: latent=100, lr=1e-4, 60 轮, 评测 2000 训练参考 x 1000 测试样本,
          数据使用各目录的 white_mask (白块遮挡) 变体。

运行:
    python exp_3_ablation.py --dataset yaleb          # 单个数据集
    python exp_3_ablation.py --all                    # 全部 5 个数据集
    python exp_3_ablation.py --dataset usps --only R2 # 只跑第 2 行
"""
import gzip, sys, time, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from DeepFCAModel import MlpNet
from fimdmf_core import fimdmf_pair, Encoder, classify

_B = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA"
DATASETS = {
    'coil100': dict(v1=_B + r"\test_with_coil100\coil_100_view1_white_mask.gz",   v2=_B + r"\test_with_coil100\coil_100_view2_white_mask.gz",   in_dim=1024),
    'yaleb':   dict(v1=_B + r"\test_extended\pie_view1_white_mask_fmmca.gz",      v2=_B + r"\test_extended\pie_view2_white_mask_fmmca.gz",      in_dim=2304),
    'mnist':   dict(v1=_B + r"\test_with_noisymnist\my_noisymnist_view1_white_mask.gz", v2=_B + r"\test_with_noisymnist\my_noisymnist_view2_white_mask.gz", in_dim=784),
    'usps':    dict(v1=_B + r"\test_with_usps\usps_view1_white_mask.gz",          v2=_B + r"\test_with_usps\usps_view2_white_mask.gz",          in_dim=256),
    'cmu':     dict(v1=_B + r"\test_cmu\cmu_lu_view1_white_mask.gz",              v2=_B + r"\test_cmu\cmu_lu_view2_white_mask.gz",              in_dim=1024),
}
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT = 100
N_REF, N_TEST = 2000, 1000
HIDDEN = (1024, 1024, 1024)


def enc_labels(y):
    if y.dtype.kind in ('U', 'S', 'O'):
        _, inv = np.unique(y, return_inverse=True)
        return inv.astype(np.int64)
    return y.astype(np.int64)


def load_ds(name, max_train=8000):
    cfg = DATASETS[name]
    with gzip.open(cfg['v1'], 'rb') as f:
        d1 = load_pickle(f)
    with gzip.open(cfg['v2'], 'rb') as f:
        d2 = load_pickle(f)
    fl = lambda a: a.astype(np.float32).reshape(a.shape[0], -1)
    x1tr, x2tr = fl(d1[0][0]), fl(d2[0][0])
    ytr = enc_labels(d1[0][1])
    x1te, yte = fl(d1[2][0]), enc_labels(d1[2][1])
    # 训练集上限, 保证各数据集可比且可行
    if x1tr.shape[0] > max_train:
        sel = np.random.RandomState(0).choice(x1tr.shape[0], max_train, replace=False)
        x1tr, x2tr, ytr = x1tr[sel], x2tr[sel], ytr[sel]
    return x1tr, x2tr, ytr, x1te, yte


def mk_enc(indim):
    """编码器: indim -> ... -> LATENT"""
    return MlpNet(list(HIDDEN) + [LATENT], indim)


def mk_dec(outdim):
    """解码器: LATENT -> ... -> outdim"""
    return MlpNet(list(HIDDEN) + [outdim], LATENT)


def svm_acc(ztr, ytr, zte, yte, C=1.0):
    clf = LinearSVC(C=C, dual=False, max_iter=5000)
    clf.fit(ztr, ytr)
    return accuracy_score(yte, clf.predict(zte))


def run_R2(ds, epochs, lr):
    """第 2 行: MSE 目标(纯自编码器) + FIMMCA-1NN 分类

    两个编码器仅由重建误差驱动, 不含任何跨视图相关项。
    """
    x1a, x2a, ytr, x1e, yte = load_ds(ds)
    indim = x1a.shape[1]
    torch.manual_seed(0)
    x1 = torch.tensor(x1a).double().to(DEV)
    x2 = torch.tensor(x2a).double().to(DEV)
    x1t = torch.tensor(x1e).double().to(DEV)

    e1, e2 = mk_enc(indim).double().to(DEV), mk_enc(indim).double().to(DEV)
    d1_, d2_ = mk_dec(indim).double().to(DEV), mk_dec(indim).double().to(DEV)
    params = list(e1.parameters()) + list(e2.parameters()) + \
             list(d1_.parameters()) + list(d2_.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    N = x1.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            o1, o2 = e1(x1[idx]), e2(x2[idx])
            loss = F.mse_loss(d1_(o1), x1[idx]) + F.mse_loss(d2_(o2), x2[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    e1.eval()
    with torch.no_grad():
        ztr = e1(x1).cpu().numpy()
        zte = e1(x1t).cpu().numpy()
    return classify(ztr, ytr, zte, yte, min(N_REF, len(ztr)), min(N_TEST, len(zte))), 'FIMMCA-1NN'


def run_R3(ds, epochs, lr):
    """第 3 行: FIMMCA 目标(多视图自监督对比) + SVM 分类

    与论文完整模型共用同一训练流程, 仅把分类策略换成线性 SVM。
    """
    x1a, x2a, ytr, x1e, yte = load_ds(ds)
    indim = x1a.shape[1]
    torch.manual_seed(0)
    x1 = torch.tensor(x1a).float().to(DEV)
    x2 = torch.tensor(x2a).float().to(DEV)
    x1t = torch.tensor(x1e).float().to(DEV)

    f = Encoder(indim, LATENT).float().to(DEV)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    N = x1.shape[0]
    for ep in range(epochs):
        f.train(); perm = torch.randperm(N)
        for i in range(0, N, 64):
            idx = perm[i:i + 64]
            if len(idx) < 2:
                continue
            S = fimdmf_pair(f(x1[idx]), f(x2[idx]))
            S = (S / 0.1) - S.max(1, keepdim=True).values.detach()
            e = torch.exp(S)
            loss = -torch.log((e.diag() + 1e-8) / (e.sum(1) + 1e-8)).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    f.eval()
    with torch.no_grad():
        ztr = f(x1).cpu().numpy()
        zte = f(x1t).cpu().numpy()
    return svm_acc(ztr[:N_REF], ytr[:N_REF], zte[:N_TEST], yte[:N_TEST]), 'SVM'


VARIANTS = [
    ('R2', 'MSE 目标 (纯自编码器) + FIMMCA-1NN', run_R2),
    ('R3', 'FIMMCA 目标 (多视图对比) + SVM',      run_R3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='yaleb', choices=list(DATASETS.keys()))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--only', type=str, default=None, help="只跑指定行 (R2/R3)")
    args = ap.parse_args()

    names = list(DATASETS.keys()) if args.all else [args.dataset]
    print(f"消融实验 (Table 4 第 2、3 行) | latent={LATENT} lr={args.lr} epochs={args.epochs}")
    print(f"评测: 2000 训练参考 x 1000 测试 | 数据 white_mask 变体\n")
    print(f"{'数据集':<10s}{'行':<5s}{'配置':<44s}{'分类器':<13s}{'准确率':>9s}")
    print("-" * 84)

    rows = {}
    for ds in names:
        for key, desc, fn in VARIANTS:
            if args.only and key != args.only:
                continue
            t0 = time.time()
            try:
                acc, clf = fn(ds, args.epochs, args.lr)
                rows[(ds, key)] = acc
                print(f"{ds:<10s}{key:<5s}{desc:<44s}{clf:<13s}{acc*100:8.2f}%   ({time.time()-t0:.0f}s)", flush=True)
            except Exception as ex:
                import traceback
                print(f"{ds:<10s}{key:<5s}{desc:<44s}失败: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()
        print("-" * 84, flush=True)

    print("\n汇总 (ACC %):")
    print(f"{'数据集':<10s}{'第2行: MSE+FIMMCA-1NN':>24s}{'第3行: FIMMCA+SVM':>22s}")
    print("-" * 58)
    for ds in names:
        a, b = rows.get((ds, 'R2')), rows.get((ds, 'R3'))
        g = lambda x: f"{x*100:.2f}" if x is not None else "-"
        print(f"{ds:<10s}{g(a):>24s}{g(b):>22s}")


if __name__ == "__main__":
    main()
