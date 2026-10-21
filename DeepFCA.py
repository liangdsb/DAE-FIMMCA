import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepFCAModel import DeepFCA
# from utils import load_data, svm_classify, dcca_knn
import time
import logging
import matplotlib.pyplot as plt

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)


class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par,
                 device=torch.device('cpu')):
        self.model = nn.DataParallel(model)  # 好像是可以使用多个GPU平行计算，可是我没有NIVIDIA显卡
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # from torch.autograd import Variable
        # Variable(x1, requires_grad=True).to(self.device, non_blocking=True)
        # Variable(x2, requires_grad=True).to(self.device, non_blocking=True)

        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            # Variable(vx1, requires_grad=True).to(self.device, non_blocking=True)
            # Variable(vx2, requires_grad=True).to(self.device, non_blocking=True)
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            # Variable(tx1, requires_grad=True).to(self.device, non_blocking=True)
            # Variable(tx2, requires_grad=True).to(self.device, non_blocking=True)
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()  # 切换网络模式为训练模式
            # batch_idxs为二维数组batch_idxs[data_size/batch_size][batch_size]
            # 下面的语句完成数据分组
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            # batch_idx是组号
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()  # 梯度清零
                batch_x1 = x1[batch_idx, :]  # 取样本x1中的第batch_idx组数据
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)  # 把batch_x1和batch_x2丢进网络模型，得到两个输出
                loss = self.loss(o1, o2)  # 计算两个输出的损失
                train_losses.append(loss.item())
                loss.requires_grad_(True)
                loss.backward()  # 误差反向传播
                self.optimizer.step()
            train_loss = np.mean(train_losses)  # 每迭代一次计算一次平均损失

            # info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()  # 切换网络模式为验证模式
                    val_loss = self.test(vx1, vx2)
                    # info_string += " - val_loss: {:.4f}".format(val_loss)
                    # if val_loss < best_val_loss:
                    #     self.logger.info(
                    #         "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1,
                    #                                                                                          best_val_loss,
                    #                                                                                          val_loss,
                    #                                                                                          checkpoint))
                    #     best_val_loss = val_loss
                    #     torch.save(self.model.state_dict(), checkpoint)
                    # else:
                    #     self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                    #         epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            # self.logger.info(info_string.format(
            #     epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)  # 将样本x1和样本x2输入到训练好的网络中，得到损失度和网络输出结果
            self.train_linear_cca(outputs[0], outputs[1])  # 把outputs的第一行与第二行放入cca中

        # checkpoint_ = torch.load(checkpoint)
        # self.model.load_state_dict(checkpoint_)

        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            # self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            # self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():  # 禁用梯度
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")

                # try cca-zoo
                from cca_zoo.models import CCA
                model = CCA(latent_dims=10)
                print(model.fit((outputs[0], outputs[1])).score((outputs[0], outputs[1])))

                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()  # 切换网络模式为验证模式，防止网络权值发生变化
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


def DFCA(data1, data2):
    start_time = time.time()

    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]
    train1 = train1.reshape(-1, 28*28)
    val1 = val1.reshape(-1, 28*28)
    test1 = test1.reshape(-1,28*28)
    val2 = val2.reshape(-1, 28*28)
    test2 = test2.reshape(-1,28*28)
    train1 = torch.tensor(train1)
    train2 = torch.tensor(train2)
    val1 = torch.tensor(val1)
    val2 = torch.tensor(val2)
    test1 = torch.tensor(test1)
    test2 = torch.tensor(test2)
    ############
    # Parameters Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")
    # device = torch.device('cuda')
    # print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 100

    # size of the input for view 1 and view 2
    input_shape1 = train1.shape[1]
    input_shape2 = train1.shape[1]
    # input_shape1 = 392
    # input_shape2 = 392

    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 10
    batch_size = 800

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    # use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False
    # end of parameters section
    ############

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/


    # Building, training, and producing the new features by DCCA
    model = DeepFCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, device=device).double()  # 构建DCCA模型
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    solver.fit(train1, train2, val1, val2, test1, test2)  # 训练网络的参数
    # TODO: Save l_cca model if needed

    set_size = [0, train1.size(0), train1.size(
        0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]
    loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat(
        [train2, val2, test2], dim=0), apply_linear_cca)

    new_data = []
    # print(outputs)
    # print(len(outputs))
    for idx in range(3):
        new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :],
                         outputs[1][set_size[idx]:set_size[idx + 1], :], data1[idx][1]])
    # new_data.append(outputs[0], outputs[1], data1[2][1])

    return new_data
