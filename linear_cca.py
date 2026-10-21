import numpy


class linear_cca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-4 #正则化参数
        r2 = 1e-4

        m = H1.shape[0] #m表示H1矩阵的行数
        o1 = H1.shape[1] #o1表示H1矩阵的列数
        o2 = H2.shape[1] #o2表示H2矩阵的列数

        self.m[0] = numpy.mean(H1, axis=0) #计算H1每一列的均值，对应着view1的均值
        self.m[1] = numpy.mean(H2, axis=0) #计算H2每一列的均值
        H1bar = H1 - numpy.tile(self.m[0], (m, 1)) #得到中心化数据矩阵H1bar
        H2bar = H2 - numpy.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * numpy.dot(H1bar.T,
                                                 H1bar) + r1 * numpy.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * numpy.dot(H2bar.T,
                                                 H2bar) + r2 * numpy.identity(o2)

        [D1, V1] = numpy.linalg.eigh(SigmaHat11) #D1是特征值，V1是特征向量
        [D2, V2] = numpy.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = numpy.dot(
            numpy.dot(V1, numpy.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = numpy.dot(
            numpy.dot(V2, numpy.diag(D2 ** -0.5)), V2.T)

        Tval = numpy.dot(numpy.dot(SigmaHat11RootInv,
                                   SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = numpy.linalg.svd(Tval) #Tval的奇异值分解
        V = V.T
        self.w[0] = numpy.dot(SigmaHat11RootInv, U[:, 0:outdim_size])  #对应的是view1的投影矩阵
        self.w[1] = numpy.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0) #中心化数据
        result = numpy.dot(result, self.w[idx]) #将原始数据投影
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)
