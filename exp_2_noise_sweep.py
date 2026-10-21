"""
噪声水平扫描曲线 (USPS, 椒盐噪声)

设计: view1 = 干净原图(固定), view2 = 干净图 + 椒盐噪声(density 变化)
      density 定义为「被污染的像素比例」, 其中一半变 0(椒)、一半变 1(盐)
      每个噪声档位下, 所有方法在同一份数据上重新训练并评测
协议: 统一用线性 SVM 分类 (与前面的对比实验一致)

运行:
    python exp_2_noise_sweep.py                      # 全部 7 个方法 x 5 档
    python exp_2_noise_sweep.py --levels 0 0.1 0.2   # 指定档位
    python exp_2_noise_sweep.py --methods CCA Ours   # 指定方法
"""
import gzip, sys, time, argparse, logging
import numpy as np
import torch

sys.path.insert(0, r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA")
logging.getLogger("Pytorch").setLevel(logging.WARNING)

from test_with_noisymnist.utils import load_pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from exp_1_comparison import (run_cca, run_rcca, run_grcca,
                                run_dcca, run_dccae, run_convcca, run_ours)

CLEAN = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA\test_with_usps\usps_clean.gz"
LATENT = 10          # 与其他方法统一(对比表用的维度); Ours 单独用 100 见 --ours-latent
SEED = 0


def salt_pepper_batch(X, density, rng):
    """椒盐噪声: 以 density 的比例污染像素, 一半->0(椒), 一半->1(盐)"""
    out = X.copy()
    mask = rng.rand(*X.shape) < density
    vals = rng.rand(*X.shape)
    out[mask & (vals < 0.5)] = 0.0
    out[mask & (vals >= 0.5)] = 1.0
    return out


def svm_acc(ztr, ytr, zte, yte, C=1.0):
    clf = LinearSVC(C=C, dual=False, max_iter=5000)
    clf.fit(ztr, ytr)
    return accuracy_score(yte, clf.predict(zte))


def build_data(density, ours_latent):
    """生成指定噪声档位下的数据字典 (供 exp_1_comparison 的各方法函数使用)"""
    with gzip.open(CLEAN, 'rb') as f:
        d = load_pickle(f)
    fl = lambda a: a.astype(np.float32).reshape(a.shape[0], -1)
    x1tr, x1te = fl(d[0][0]), fl(d[2][0])
    ytr, yte = d[0][1].astype(np.int64), d[2][1].astype(np.int64)
    rng = np.random.RandomState(SEED + int(density * 1000))
    x2tr = salt_pepper_batch(x1tr, density, rng)
    x2te = x1te     # 评测用 view1(干净), 噪声只用于训练时的第二视图
    return dict(train_1=x1tr, train_2=x2tr, test_1=x1te, test_2=x2te,
                train_y=ytr, test_y=yte)


def main():
    ap = argparse.ArgumentParser()
    # 最低档用 0.05 而非 0: density=0 时 view1==view2, CCA 系方法会退化(相关系数恒为1, 投影不唯一)
    ap.add_argument('--levels', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument('--methods', type=str, nargs='+',
                    default=['CCA', 'RCCA', 'GRCCA', 'DCCA', 'DCCAE', 'ConvCCA', 'DAE-FIMMCA'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ours-latent', type=int, default=100, help="Ours 的隐层维度")
    ap.add_argument('--ours-lr', type=float, default=1e-4)
    args = ap.parse_args()

    indim = 256
    print(f"噪声扫描 | USPS 椒盐 | 档位={args.levels} | epochs={args.epochs} | 协议=统一SVM")
    print(f"latent: 其他方法={LATENT}, Ours={args.ours_latent}\n")

    results = {}
    for density in args.levels:
        d = build_data(density, args.ours_latent)
        print(f"--- density={density} (污染像素 {density*100:.0f}%) ---", flush=True)
        for m in args.methods:
            t0 = time.time()
            try:
                if m == 'CCA':
                    ztr, zte = run_cca(d, LATENT)
                elif m == 'RCCA':
                    ztr, zte = run_rcca(d, LATENT)
                elif m == 'GRCCA':
                    ztr, zte = run_grcca(d, LATENT)
                elif m == 'DCCA':
                    ztr, zte = run_dcca(d, LATENT, args.epochs, 64, args.lr, indim)
                elif m == 'DCCAE':
                    ztr, zte = run_dccae(d, LATENT, args.epochs, 64, args.lr, indim)
                elif m == 'ConvCCA':
                    ztr, zte = run_convcca(d, LATENT, args.epochs, 64, args.lr)
                elif m == 'DAE-FIMMCA':
                    ztr, zte = run_ours(d, args.ours_latent, args.epochs, 64, args.ours_lr)
                else:
                    print(f"  未知方法 {m}"); continue
                acc = svm_acc(ztr[:2000], d['train_y'][:2000], zte[:1000], d['test_y'][:1000])
                results[(m, density)] = acc
                print(f"  {m:12s} {acc*100:6.2f}%   ({time.time()-t0:.0f}s)", flush=True)
            except Exception as ex:
                import traceback
                print(f"  {m:12s} 失败: {type(ex).__name__}: {ex}", flush=True)
                traceback.print_exc()

    # 汇总: 行=方法, 列=噪声档位
    print("\n" + "=" * 78)
    print("汇总 ACC (%)")
    hdr = "".join(f"{l:>11}" for l in args.levels)
    print(f"{'方法':<14s}{hdr}")
    print("-" * 78)
    for m in args.methods:
        row = "".join(f"{results[(m,l)]*100:11.2f}" if (m, l) in results else f"{'-':>11}"
                      for l in args.levels)
        print(f"{m:<14s}{row}")
    print("=" * 78)
    print("\n(可直接用于画图: x 轴=噪声密度, y 轴=准确率, 每条线=一个方法)")


if __name__ == "__main__":
    main()
