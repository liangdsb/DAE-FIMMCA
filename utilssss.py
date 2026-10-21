import gzip
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle


def load_data(path1, path2, num):
    try:
        with gzip.open(path1, 'rb') as fi:
            images = np.frombuffer(fi.read(), dtype=np.int8, offset=16)
            # images = data_i.reshape(784, 60000)
            images = images.reshape(num, 784)
        with gzip.open(path2, 'rb') as fl:
            labels = np.frombuffer(fl.read(), dtype=np.int8, offset=8)
            # labels = data_l.reshape(1, 60000)
            # labels = data_l
    except:
        print("The file directory doesn't exist!")
    return images, labels


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """

    try:
        ret = thepickle.load(f, encoding='latin1')  # 将文件f反序列化返回给ret
        # ret = thepickle.load(open(f,"rb"), encoding='utf-8')  # 将文件f反序列化返回给ret
    except TypeError:
        ret = thepickle.load(f)

    return ret


def svm_classify_mark(data, C):
    train_data, train_label = data[0]
    valid_data, valid_label = data[1]
    test_data, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def rotate_Img(image, angle):
    """

    :param image: 输入处理图像
    :param angle: 旋转角度
    :return:
    """
    angle = np.random.uniform(-angle, angle)
    h, w = image.shape[:2]
    matRotate = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1)
    image = image.astype("uint8")
    dst = cv2.warpAffine(image, matRotate, (w, h))
    return dst


def rotate_Img_batch(images, angle):
    i = 0
    rotate_images = []
    while i < images.shape[0]:
        image = rotate_Img(images[i, :].reshape(28, 28), angle)
        rotate_images.append(image)
        i = i + 1
    rotate_images = np.array(rotate_images)
    return rotate_images.reshape(images.shape[0], -1)


def salt_pepper_noise(image, prob=0.1):
    """
    添加椒盐噪声
    :param image: 输入图像
    :param prob: 噪声比
    :return: 带有椒盐噪声的图像
    """
    salt = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                salt[i][j] = 0
            elif rdn > thres:
                salt[i][j] = 255
            else:
                salt[i][j] = image[i][j]
    return salt


def salt_pepper_noise_batch(images, prob=0.2):
    i = 0
    noisy_images = []
    while i < images.shape[0]:
        image = salt_pepper_noise(images[i, :].reshape(28, 28))
        noisy_images.append(image)
        i = i + 1
    noisy_images = np.array(noisy_images)
    return noisy_images.reshape(images.shape[0], -1)


def showData(datas, labels):
    """可视化数据"""

    datas = datas.reshape(datas.shape[0], 28, 28)
    fig = plt.figure()  # 生成图框

    # for i, c in enumerate(np.random.randint(0, 1000, 25)):  # 随机取0，1000里的25张图片
    for i, c in enumerate(np.arange(25)):
        plt.subplot(5, 5, i + 1)
        plt.tight_layout()  # 调整间距
        plt.imshow(datas[c], interpolation='none', cmap='gray')
        plt.title("数字标签: {}".format(labels[c]))
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def showSingleImage(datas, labels):
    """可视化数据"""

    fig = plt.figure()  # 生成图框
    datas = datas.reshape(28, 28)
    plt.imshow(datas, interpolation='none', cmap='gray')
    plt.title("数字标签: {}".format(labels))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def toNormalize(data):
    data = data / 255
    return data


def getTwoView(data):
    # data1和data2数据归一化
    # data随机旋转[-45,45] 作为view1
    view1 = rotate_Img_batch(data, 0)
    view1 = toNormalize(view1)

    # data加[0,1]椒盐噪声噪声 作为view2
    view2 = salt_pepper_noise_batch(data)
    view2 = toNormalize(view2)
    return view1, view2


import math


def Image_reduction_01(data1, data2):
    """
    图像数据约简：
    1、选取一个合适的m：为分析局部相关性的区间的大小。难点：如何选择合适的m
    2、将data1和data2分为k份----k = data1.length/m
    3、迭代k次 分析每个m块的相关性
    4、每i in range（k）次进行一次feature_sort,
        4.1 将k个m块分区间，迭代不同区间
            4.1.1 调用u分析包含度
        4.2若包含度大于等于0.5 则将mark对应位置上的值置1 否则置0
    5、将mark作为svm的参数，训练svm模型
    :param data1: 视图1数据---一维数组，每一列为各个图像样本的特征属性
    :param data2: 视图2数据
    :return: 约简后对应的数据结构mark---用一个一维数组表示，shape与data1相同，
            每一列的值为0或1，
            0 表示无模糊单调关系  1 表示有模糊单调关系
    """
    mark = np.zeros(len(data1), dtype=int)
    monotone = np.zeros(data1.shape, np.uint8)
    sorted_index1, sorted_index2 = feature_sort(data1, data2)
    for k in range(1, int(len(data1) / 2) + 1):  # k = 区间内元素个数
        if len(data1) % k == 0:
            p = len(data1) / k  # p = 区间个数
        else:
            p = math.ceil(len(data1) / k) - 1

        if u(sorted_index1, sorted_index2, k) >= 0.5:
            monotone[k] = 1
        else:
            monotone[k] = 0

    count = 0
    # print(monotone)
    for h in range(int(len(data1) / 2), int(len(data1) / 2) - 2, -1):
        if monotone[h] == 0:
            break
        elif monotone[h] == 1:
            count = count + 1
    if count >= 2:
        # 有模糊单调关系
        mark[0:len(data1)] = 1
    else:
        mark[0:len(data1)] = 0

    return mark

def Image_reduction_02(data1, data2):
    """
    图像数据约简：
    1、选取一个合适的m：为分析局部相关性的区间的大小。难点：如何选择合适的m
    2、将data1和data2分为k份----k = data1.length/m
    3、迭代k次 分析每个m块的相关性
    4、每i in range（k）次进行一次feature_sort,
        4.1 将k个m块分区间，迭代不同区间
            4.1.1 调用u分析包含度
        4.2若包含度大于等于0.5 则将mark对应位置上的值置1 否则置0
    5、将mark作为svm的参数，训练svm模型
    :param data1: 视图1数据---一维数组，每一列为各个图像样本的特征属性
    :param data2: 视图2数据
    :return: 约简后对应的数据结构mark---用一个一维数组表示，shape与data1相同，
            每一列的值为0或1，
            0 表示无模糊单调关系  1 表示有模糊单调关系
    """
    # mark = np.zeros(len(data1), dtype=int)
    monotone = np.zeros(data1.shape, np.uint8)
    sorted_index1, sorted_index2 = feature_sort(data1, data2)
    for k in range(1, int(len(data1) / 2) + 1):  # k = 区间内元素个数
        if len(data1) % k == 0:
            p = len(data1) / k  # p = 区间个数
        else:
            p = math.ceil(len(data1) / k) - 1

        if u(sorted_index1, sorted_index2, k) >= 0.5:
            monotone[k] = 1
        else:
            monotone[k] = 0

    count = 0
    # print(monotone)
    for h in range(int(len(data1) / 2), int(len(data1) / 2) - 2, -1):
        if monotone[h] == 0:
            break
        elif monotone[h] == 1:
            count = count + 1
    if count >= 2:
        # 有模糊单调关系
        print("有关系")
        # mark[0:len(data1)] = 1
    else:
        print("没有关系")
        # mark[0:len(data1)] = 0

    # return mark

def feature_sort(feature1, feature2):
    """
    比较feature值的大小，交换坐标位置：
    1、初始化index1和index2，长度为feature长度，值为[0,feature.length-1]
    2、将feature1和feature2转为list
    3、选取一个合适的排序算法
    4、得到sorted_index1和sorted_index2
    :param feature1: 是一个1维数组
    :param feature2: 是一个1维数组
    :return: index1和index2 分别是feature1值和feature2值按从小到大排序后对应的变换后的坐标
    """

    return np.array(np.argsort(feature1)), np.array(np.argsort(feature2))


# test-----feature_sort
feature1 = np.array([0, 0, 0, 0.7, 0.9, 0.4, 0, 0, 0])
feature2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
feature_sort(feature1, feature2)
print(np.array(np.argsort(feature1)))

def u(sorted_index1, sorted_index2, k):
    """
    计算包含度：
    :param sorted_index1: 一维的整型数组
    :param sorted_index2: 同上
    :return: 包含度 类型为 float
    """
    u_i = []  # 用来存储各个区间的包含度
    for i in range(0, len(sorted_index1), k):
        # 计算sorted_index[i:i+k]和sorted_index2[i:i+k]的包含度
        if i + k > len(sorted_index1):
            break
            # intersect_len = len(np.intersect1d(sorted_index1[i:len(sorted_index1)], sorted_index2[i:len(sorted_index2)]))
            # u_i.append(float(intersect_len / k))
        else:
            intersect_len = len(np.intersect1d(sorted_index1[i:i + k], sorted_index2[i:i + k]))
            u_i.append(float(intersect_len / k))
    # print(u_i)
    u_k = min(u_i)  # u_k 表示在各个区间元素为k时的包含度，这里称它为整体包含度吧，因为要取各区间最小的那个包含度
    return u_k

    # test-----u
def u_max(sorted_index1, sorted_index2, k):
    """
    计算包含度：
    :param sorted_index1: 一维的整型数组
    :param sorted_index2: 同上
    :return: 包含度 类型为 float
    """
    u_i = []  # 用来存储各个区间的包含度
    for i in range(0, len(sorted_index1), k):
        # 计算sorted_index[i:i+k]和sorted_index2[i:i+k]的包含度
        if i + k > len(sorted_index1):
            break
            # intersect_len = len(np.intersect1d(sorted_index1[i:len(sorted_index1)], sorted_index2[i:len(sorted_index2)]))
            # u_i.append(float(intersect_len / k))
        else:
            intersect_len = len(np.intersect1d(sorted_index1[i:i + k], sorted_index2[i:i + k]))
            u_i.append(float(intersect_len / k))
    # print(u_i)
    u_k = max(u_i)  # u_k 表示在各个区间元素为k时的包含度，这里称它为整体包含度吧，因为要取各区间最小的那个包含度
    return u_k
# a = np.array([1,3,4,7,5,6,8,9,10,0])
# b= np.array([1,2,4,7,3,5,9,8,0,10])
# print(u(a,b,6))
