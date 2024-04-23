import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from bp_net import MyNet
from lib.datasets import PaviaU
from lib.stackedDAE import StackedDAE

sys.path.append("..")


def best_map(L1, L2):
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)  # 逐元素比较大小
    G = np.zeros((nClass, nClass))  # 生成一个0矩阵
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):  # 10
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def acc_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] == c_x[:])
    acc = err_x.astype(float) / (gt_s.shape[0])
    return acc


def train(args):
    # according to the released code, mnist data is multiplied by 0.02
    # 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
    # so add a customized Scale transform to multiple by 5.1
    train_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae = StackedDAE(input_dim=103, z_dim=9, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)
    print(sdae)

    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
                  num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="mse")
    sdae.fit(train_loader, test_loader, lr=args.lr,
             num_epochs=args.epochs, corrupt=0.2, loss_type="mse")
    sdae.save_model("./model/PaviaU/sdae.pt")


def predict(args):
    sdae = StackedDAE(input_dim=103, z_dim=10, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)
    print(sdae)

    data_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False, predict=True),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae.load_model('./model/PaviaU/sdae.pt')
    sdae.predict_PaviaU(data_loader)


def kmeans(args):
    print('kmeans聚类：')
    sdae = StackedDAE(input_dim=103, z_dim=9, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)

    data_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False, predict=True, dataset=True),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae.load_model('./model/PaviaU/sdae.pt')
    data, label = sdae.get_encoder(data_loader)
    cluster = KMeans(n_clusters=9, random_state=1).fit(data)
    pred = cluster.labels_
    print('SDAE后聚类：')
    print('准确率：', acc_rate(label, pred))
    NMI = metrics.normalized_mutual_info_score(pred, label)
    print('NMI: ', NMI)

    data = pd.read_csv(os.path.join(
        './dataset/PaviaU/raw/PaviaU.csv'), header=None)
    data = data.values
    label = data[:, -1].astype(int)
    data = data[:, :-1]
    cluster = KMeans(n_clusters=9, random_state=1).fit(data)
    pred = cluster.labels_
    print('原始数据聚类：')
    print('准确率：', acc_rate(label, pred))  # 0.24619093539054968
    NMI = metrics.normalized_mutual_info_score(pred, label)
    print('NMI: ', NMI)  # 0.15086213992440017


def svm(args):
    print('svm分类：')
    sdae = StackedDAE(input_dim=103, z_dim=9, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)

    data_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False, predict=True),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae.load_model('./model/PaviaU/sdae.pt')
    data, label = sdae.get_encoder(data_loader)
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.9)
    # 模型训练与拟合
    clf = SVC(kernel='rbf', gamma=0.125, C=2)
    clf.fit(data_train, label_train)
    pred = clf.predict(data_test)
    accuracy = metrics.accuracy_score(label_test, pred)*100
    print('SDAE后分类：', accuracy)

    data = pd.read_csv(os.path.join(
        './dataset/PaviaU/raw/PaviaU.csv'), header=None)
    data = data.values
    label = data[:, -1].astype(int)
    data = data[:, :-1]
    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.9)
    # 模型训练与拟合
    clf = SVC(kernel='rbf', gamma=0.125, C=2)
    clf.fit(data_train, label_train)
    pred = clf.predict(data_test)
    accuracy = metrics.accuracy_score(label_test, pred)*100
    print('原始数据分类：', accuracy)


def sc(args):  # 谱聚类
    print('谱聚类：')
    sdae = StackedDAE(input_dim=103, z_dim=9, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)

    data_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False, predict=True),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae.load_model('./model/PaviaU/sdae.pt')
    data, label = sdae.get_encoder(data_loader)  # 42776
    data_ = []
    label_ = []
    tmp = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(label)):
        if tmp[label[i]] < 900:
            data_.append(data[i])
            label_.append(label[i])
            tmp[label[i]] += 1
    data_ = np.array(data_)
    label_ = np.array(label_)
    cluster = SpectralClustering(
        n_clusters=9, assign_labels='kmeans', random_state=1).fit(data_)
    pred = cluster.labels_
    print('SDAE后聚类：')
    print('准确率：', acc_rate(label_, pred))
    print('NMI: ', metrics.normalized_mutual_info_score(pred, label_))

    data = pd.read_csv(os.path.join(
        './dataset/PaviaU/raw/PaviaU.csv'), header=None)
    data = data.values
    label = data[:, -1].astype(int)
    data = data[:, :-1]
    data_ = []
    label_ = []
    tmp = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(label)):
        if tmp[label[i]] < 900:
            data_.append(data[i])
            label_.append(label[i])
            tmp[label[i]] += 1
    data_ = np.array(data_)
    label_ = np.array(label_)
    cluster = SpectralClustering(
        n_clusters=9, assign_labels='kmeans', random_state=1).fit(data_)
    pred = cluster.labels_
    print('原始数据聚类：')
    print('准确率：', acc_rate(label_, pred))
    print('NMI: ', metrics.normalized_mutual_info_score(pred, label_))
    
    # data = np.split(data, block)
    # label = np.split(label, block)
    # acc, nmi = 0, 0
    # for i in range(block):
    #     cluster = SpectralClustering(
    #         n_clusters=9, assign_labels='discretize', random_state=0).fit(data[i])
    #     pred = cluster.labels_
    #     acc += acc_rate(label[i], pred)
    #     nmi += metrics.normalized_mutual_info_score(pred, label[i])
    # print('原始数据聚类：')
    # print('准确率：', acc / block)
    # print('NMI: ', nmi / block)


def bp(args):
    def load_data(data, label):
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        data = TensorDataset(data, label)
        train_size = int(0.5 * len(data))
        test_size = len(data) - train_size
        train_data, test_data = random_split(data, [train_size, test_size], generator=torch.manual_seed(0))
        return train_data, test_data
    
    def train(train_loader, model):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
        closs = nn.CrossEntropyLoss()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            loss = closs(pred, labels)
            loss.backward()
            optimizer.step()

    def test(test_loader, model):
        correct = 0
        total = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print(' 准确率：{}%'.format(correct * 100 / total))
        return correct * 100 / total
    
    print('bp神经网络：')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sdae = StackedDAE(input_dim=103, z_dim=9, binary=False,
                      encodeLayer=[200, 200, 1000], decodeLayer=[1000, 200, 200], activation="relu",
                      dropout=0)

    data_loader = torch.utils.data.DataLoader(
        PaviaU('./dataset/PaviaU', train=False, predict=True, dataset=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae.load_model('./model/PaviaU/sdae.pt')
    data, label = sdae.get_encoder(data_loader)
    label = label - 1
    label = label.astype(np.int64)
    
    batch_size = 10
    input_size = 9
    hidden_size = 9
    num_classes = 9
    epoch = 100
    train_data, test_data = load_data(data, label)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = MyNet(input_size, hidden_size, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, model)
        best = max(test(test_loader, model), best)
    print('SDAE后分类后最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()

    data = pd.read_csv(os.path.join(
        './dataset/PaviaU/raw/PaviaU.csv'), header=None)
    data = data.values
    label = data[:, -1].astype(np.int64)
    data = data[:, :-1]
    label = label - 1
    
    input_size = 103
    hidden_size = 9

    train_data, test_data = load_data(data, label)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = MyNet(input_size, hidden_size, num_classes).to(device)
    best = 0
    for i in range(epoch):
        print('epoch:', i, end='')
        train(train_loader, model)
        best = max(test(test_loader, model), best)
    print('原始数据分类最高准确率：{}%'.format(best))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    # train(args)
    # predict(args)
    # kmeans(args)
    # svm(args)
    # sc(args)
    bp(args)
