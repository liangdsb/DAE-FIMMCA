"""
生成论文中使用的 USPS 干净/旋转数据集
======================================

论文中 "干净 USPS" 一栏的实验数据并非原始 .gz 中自带, 而是由本脚本从现有数据派生。

流程(两阶段, 顺序执行)
----------------------
  阶段 1  干净数据
      来源: test_with_usps/usps_view1.gz (其 view1 是未加噪声的原图, 已实测验证)
      输出: test_with_usps/usps_clean.gz
            两个视图为同一份干净图
  阶段 2  干净 + 小角度旋转
      来源: 阶段 1 的输出
      输出: test_with_usps/usps_clean_view1.gz  (干净原图)
            test_with_usps/usps_clean_view2.gz  (干净图随机旋转 ±10°)
      说明: 若两视图完全相同, CCA 系方法会退化(典型相关系数恒为 1、投影解不唯一),
            故对第二视图施加信息无损的良性变换, 使多视图任务非平凡

运行
----
    python make_datasets.py
"""
import gzip, os, sys
import numpy as np
import cv2

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

BASE = r"D:\Pycharm\Project\CCA-zoo\DAE-FIMMCA\test_with_usps"
SRC_CLEAN = os.path.join(BASE, "usps_view1.gz")
DST_CLEAN = os.path.join(BASE, "usps_clean.gz")
DST_ROT1 = os.path.join(BASE, "usps_clean_view1.gz")
DST_ROT2 = os.path.join(BASE, "usps_clean_view2.gz")

SIDE = 16          # USPS 为 16x16
MAX_ANGLE = 10.0   # 第二视图旋转角度范围 ±10°
SEED = 0


def load_pickle(f):
    try:
        return thepickle.load(f, encoding='latin1')
    except TypeError:
        return thepickle.load(f)


def rotate_batch(X, angle, seed):
    """X:(n,256) in [0,1] -> 每样本随机旋转 [-angle, angle] 度 (float32, 无量化损失)"""
    rng = np.random.RandomState(seed)
    out = np.empty_like(X, dtype=np.float32)
    c = (SIDE * 0.5, SIDE * 0.5)
    for i in range(X.shape[0]):
        img = X[i].reshape(SIDE, SIDE).astype(np.float32)
        a = rng.uniform(-angle, angle)
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        out[i] = cv2.warpAffine(img, M, (SIDE, SIDE),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1)
    return out


def stage1_clean():
    """由 usps_view1.gz 生成 usps_clean.gz (view1 = view2 = 干净原图)"""
    print("=" * 60)
    print("[阶段 1] 生成干净数据")
    print("=" * 60)
    with gzip.open(SRC_CLEAN, 'rb') as f:
        src = load_pickle(f)

    clean = [[src[i][0], src[i][1]] for i in range(3)]

    for name, (x, y) in zip(['train', 'val', 'test'], clean):
        assert len(x) == len(y), f"{name} 特征/标签数不一致"
        assert np.isfinite(x).all(), f"{name} 含 NaN/Inf"
        assert x.max() <= 1.0 and x.min() >= 0.0, f"{name} 取值超出 [0,1]"
        # 干净性判据: 椒盐噪声会在图上散布大量精确 0/1 像素(占 9%~17%),
        # 干净图只在个别亮图上偶有一两个恰好=255 的像素, 故按比例判定
        assert (x == 1.0).mean() < 1e-3, f"{name} 精确=1 像素比例过高, 可能不是干净数据"
        print(f"  {name}: x={x.shape} y={y.shape} 值域=[{x.min():.3f}, {x.max():.3f}] "
              f"精确1.0占比={(x == 1.0).mean():.2e} 类别数={len(np.unique(y))}")

    with gzip.open(DST_CLEAN, 'wb') as f:
        thepickle.dump(clean, f)
    print(f"  已生成: {DST_CLEAN}")

    with gzip.open(DST_CLEAN, 'rb') as f:
        back = load_pickle(f)
    for i in range(3):
        assert np.array_equal(back[i][0], clean[i][0])
        assert np.array_equal(back[i][1], clean[i][1])
    print(f"  回读验证通过: {[(b[0].shape, len(b[1])) for b in back]}\n")


def stage2_clean_rot():
    """由 usps_clean.gz 生成 view1(干净) + view2(±10°旋转)"""
    print("=" * 60)
    print("[阶段 2] 生成干净 + 小角度旋转数据")
    print("=" * 60)
    with gzip.open(DST_CLEAN, 'rb') as f:
        src = load_pickle(f)

    v1 = [[src[i][0], src[i][1]] for i in range(3)]
    v2 = [[rotate_batch(src[i][0], MAX_ANGLE, SEED + i), src[i][1]] for i in range(3)]

    for name, (x, _) in zip(['train', 'val', 'test'], v2):
        assert np.isfinite(x).all(), f"{name} 含 NaN/Inf"
        assert x.min() >= 0 and x.max() <= 1.0, f"{name} 超出 [0,1]"
    for i, name in enumerate(['train', 'val', 'test']):
        print(f"  {name}: view1 与 view2 平均像素差 = {np.abs(v1[i][0] - v2[i][0]).mean():.4f}")

    for dst, data in [(DST_ROT1, v1), (DST_ROT2, v2)]:
        with gzip.open(dst, 'wb') as f:
            thepickle.dump(data, f)
        print(f"  已生成: {dst}")

    for dst in [DST_ROT1, DST_ROT2]:
        with gzip.open(dst, 'rb') as f:
            back = load_pickle(f)
        assert len(back) == 3
    print("  回读验证通过\n")


def main():
    stage1_clean()
    stage2_clean_rot()
    print("=" * 60)
    print("全部完成。用法: usps_clean_view1.gz / usps_clean_view2.gz 作为两视图")
    print("=" * 60)


if __name__ == "__main__":
    main()
