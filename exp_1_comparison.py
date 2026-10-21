"""
干净 USPS 数据集上的多方法对比实验

数据: test_with_usps/usps_clean_view1.gz (干净原图)
      test_with_usps/usps_clean_view2.gz (干净图 ±10° 旋转)
协议: 所有方法统一 —— 提取 view1 隐特征 -> 线性 SVM 分类 -> 测试集准确率

方法:
  CCA      cca_zoo.linear.CCA
  RCCA     随机傅里叶特征 + CCA   (Lopez-Paz et al. 2014 的随机非线性 CCA; 环境无官方实现, 本脚本实现)
  GRCCA    cca_zoo.linear.GRCCA  (Tuzhilina et al. 的原作者实现)
  DCCA     两个 MLP 编码器 + 标准 DCCA 损失 (objectives.convcca_loss, 即 Andrew et al. 2013)
  DCCAE    两个 MLP 编码器 + 解码器 + DCCA 损失 + λ·重建 (Wang et al. 2015)
  ConvCCA  两个 ConvNet + DCCA 损失 (除输入尺寸适配 16x16 外, 与项目 ConvCCA.py 一致)
  Ours     多视图自监督对比 + 共享编码器 (dae_fimmca)

运行:
    python exp_1_comparison.py
    python exp_1_comparison.py --epochs 50 --latent 10
"""
import gzip, sys, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from DeepFCAModel import MlpNet, ConvNet
from objectives import convcca_loss
from fimdmf_core import fimdmf_pair, Encoder

V1 = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA\test_with_usps\usps_clean_view1.gz"
V2 = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA\test_with_usps\usps_clean_view2.gz"
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0


# ---------------- 数据 ----------------
def load_data():
    with gzip.open(V1, 'rb') as f:
        d1 = load_pickle(f)
    with gzip.open(V2, 'rb') as f:
        d2 = load_pickle(f)
    out = {}
    for i, name in enumerate(['train', 'val', 'test']):
        out[f'{name}_1'] = d1[i][0].astype(np.float32).reshape(d1[i][0].shape[0], -1)
        out[f'{name}_2'] = d2[i][0].astype(np.float32).reshape(d2[i][0].shape[0], -1)
        out[f'{name}_y'] = d1[i][1].astype(np.int64)
    return out


def svm_acc(ztr, ytr, zte, yte, C=1.0):
    clf = LinearSVC(C=C, dual=False, max_iter=5000)
    clf.fit(ztr, ytr)
    return accuracy_score(yte, clf.predict(zte))


# ---------------- 通用深度训练循环 ----------------
def train_deep(loss_fn, params, x1, x2, epochs, batch_size, lr, forward_fn):
    """forward_fn(x1_batch, x2_batch) -> loss"""
    opt = torch.optim.Adam(params, lr=lr)
    N = x1.shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            loss = forward_fn(x1[idx], x2[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        hist.append(tot / max(nb, 1))
    return hist


# ---------------- 各方法 ----------------
def run_cca(d, latent):
    from cca_zoo.linear import CCA
    m = CCA(latent_dimensions=latent)
    m.fit([d['train_1'], d['train_2']])
    return m.transform([d['train_1'], d['train_2']])[0], m.transform([d['test_1'], d['test_2']])[0]


def run_rcca(d, latent, n_comp=500, gamma=None):
    """随机傅里叶特征(RBF) + CCA —— Lopez-Paz et al. 2014 随机非线性 CCA"""
    from sklearn.kernel_approximation import RBFSampler
    from cca_zoo.linear import CCA
    g = gamma if gamma else 1.0 / d['train_1'].shape[1]
    r1 = RBFSampler(gamma=g, n_components=n_comp, random_state=SEED)
    r2 = RBFSampler(gamma=g, n_components=n_comp, random_state=SEED + 1)
    t1 = r1.fit_transform(d['train_1']); t2 = r2.fit_transform(d['train_2'])
    e1 = r1.transform(d['test_1']);      e2 = r2.transform(d['test_2'])
    m = CCA(latent_dimensions=latent)
    m.fit([t1, t2])
    return m.transform([t1, t2])[0], m.transform([e1, e2])[0]


def run_grcca(d, latent):
    from cca_zoo.linear import GRCCA
    m = GRCCA(latent_dimensions=latent)
    m.fit([d['train_1'], d['train_2']])
    return m.transform([d['train_1'], d['train_2']])[0], m.transform([d['test_1'], d['test_2']])[0]


def run_dcca(d, latent, epochs, bs, lr, indim, layers=(1024, 1024, 1024)):
    torch.manual_seed(SEED)
    e1 = MlpNet(list(layers) + [latent], indim).double().to(DEV)
    e2 = MlpNet(list(layers) + [latent], indim).double().to(DEV)
    lf = convcca_loss(outdim_size=latent, use_all_singular_values=False, device=DEV).loss
    x1 = torch.tensor(d['train_1']).double().to(DEV)
    x2 = torch.tensor(d['train_2']).double().to(DEV)
    train_deep(lf, list(e1.parameters()) + list(e2.parameters()), x1, x2, epochs, bs, lr,
               lambda a, b: lf(e1(a), e2(b)))
    e1.eval()
    with torch.no_grad():
        ztr = e1(torch.tensor(d['train_1']).double().to(DEV)).cpu().numpy()
        zte = e1(torch.tensor(d['test_1']).double().to(DEV)).cpu().numpy()
    return ztr, zte


def run_dccae(d, latent, epochs, bs, lr, indim, lam=0.5, layers=(1024, 1024, 1024)):
    torch.manual_seed(SEED)
    e1 = MlpNet(list(layers) + [latent], indim).double().to(DEV)
    e2 = MlpNet(list(layers) + [latent], indim).double().to(DEV)
    dec1 = MlpNet(list(layers)[::-1] + [indim], latent).double().to(DEV)
    dec2 = MlpNet(list(layers)[::-1] + [indim], latent).double().to(DEV)
    lf = convcca_loss(outdim_size=latent, use_all_singular_values=False, device=DEV).loss
    x1 = torch.tensor(d['train_1']).double().to(DEV)
    x2 = torch.tensor(d['train_2']).double().to(DEV)

    def fwd(a, b):
        o1, o2 = e1(a), e2(b)
        rec = F.mse_loss(dec1(o1), a) + F.mse_loss(dec2(o2), b)
        return lf(o1, o2) + lam * rec

    train_deep(None, list(e1.parameters()) + list(e2.parameters()) +
               list(dec1.parameters()) + list(dec2.parameters()), x1, x2, epochs, bs, lr, fwd)
    e1.eval()
    with torch.no_grad():
        ztr = e1(torch.tensor(d['train_1']).double().to(DEV)).cpu().numpy()
        zte = e1(torch.tensor(d['test_1']).double().to(DEV)).cpu().numpy()
    return ztr, zte


def run_convcca(d, latent, epochs, bs, lr, side=16):
    torch.manual_seed(SEED)
    c1, c2 = ConvNet().double().to(DEV), ConvNet().double().to(DEV)
    lf = convcca_loss(outdim_size=10, use_all_singular_values=False, device=DEV).loss
    def prep(X): return torch.tensor(X.reshape(-1, 1, side, side)).double().to(DEV)
    x1, x2 = prep(d['train_1']), prep(d['train_2'])
    train_deep(lf, list(c1.parameters()) + list(c2.parameters()), x1, x2, epochs, bs, lr,
               lambda a, b: lf(c1(a).squeeze(), c2(b).squeeze()))
    c1.eval()
    with torch.no_grad():
        ztr = c1(prep(d['train_1'])).squeeze().cpu().numpy()
        zte = c1(prep(d['test_1'])).squeeze().cpu().numpy()
    return ztr, zte


def run_ours(d, latent, epochs, bs, lr=5e-5):
    """多视图自监督对比 + 共享编码器 (与 dae_fimmca 一致)"""
    torch.manual_seed(SEED)
    f = Encoder(d['train_1'].shape[1], latent).float().to(DEV)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    x1 = torch.tensor(d['train_1']).to(DEV)
    x2 = torch.tensor(d['train_2']).to(DEV)
    N = x1.shape[0]
    for ep in range(epochs):
        f.train(); perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            S = fimdmf_pair(f(x1[idx]), f(x2[idx]))
            S = (S / 0.1) - S.max(1, keepdim=True).values.detach()
            e = torch.exp(S)
            loss = -torch.log((e.diag() + 1e-8) / (e.sum(1) + 1e-8)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    f.eval()
    with torch.no_grad():
        ztr = f(torch.tensor(d['train_1']).to(DEV)).cpu().numpy()
        zte = f(torch.tensor(d['test_1']).to(DEV)).cpu().numpy()
    return ztr, zte


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--latent', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--only', type=str, default=None, help="只跑某个方法")
    args = ap.parse_args()

    d = load_data()
    print(f"数据: train={d['train_1'].shape} test={d['test_1'].shape} 类别={len(np.unique(d['train_y']))}")
    print(f"协议: 提取 view1 隐特征 -> 线性SVM -> 测试集准确率 | latent={args.latent} epochs={args.epochs} lr={args.lr}\n")
    indim = d['train_1'].shape[1]
    ytr, yte = d['train_y'], d['test_y']

    METHODS = [
        ('CCA',     lambda: run_cca(d, args.latent)),
        ('RCCA',    lambda: run_rcca(d, args.latent)),
        ('GRCCA',   lambda: run_grcca(d, args.latent)),
        ('DCCA',    lambda: run_dcca(d, args.latent, args.epochs, args.batch, args.lr, indim)),
        ('DCCAE',   lambda: run_dccae(d, args.latent, args.epochs, args.batch, args.lr, indim)),
        ('ConvCCA', lambda: run_convcca(d, args.latent, args.epochs, args.batch, args.lr)),
        ('Ours',    lambda: run_ours(d, args.latent, args.epochs, args.batch, args.lr)),
    ]
    results = []
    for name, fn in METHODS:
        if args.only and name != args.only:
            continue
        t0 = time.time()
        try:
            ztr, zte = fn()
            acc = svm_acc(ztr, ytr, zte, yte)
            results.append((name, acc))
            print(f"  {name:8s} acc={acc*100:6.2f}%   ({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            import traceback
            print(f"  {name:8s} 失败: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 46)
    print(f"{'方法':<10s}{'准确率':>12s}")
    print("-" * 46)
    for name, acc in results:
        print(f"{name:<10s}{acc*100:11.2f}%")
    print("=" * 46)


if __name__ == "__main__":
    main()
