import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import datasets, transforms

from lib.denoisingAutoencoder import DenoisingAutoencoder
from lib.ops import BCELoss, MSELoss
from lib.utils import Dataset, masking_noise


def buildNetwork(layers, activation="relu", dropout=0):
    """_summary_

    Args:
        layers (_type_): the dims of each layer. [784 500 500 2000]
        activation (str, optional): active function. Defaults to "relu".
        dropout (int, optional): dropout. Defaults to 0.

    Returns:
        nn.Sequential: net
    """
    net = []
    for i in range(1, len(layers)):  # 逐层建立网络
        net.append(nn.Linear(layers[i-1], layers[i]))  # 全连接层
        if activation == "relu":  # relu激活函数
            net.append(nn.ReLU())
        elif activation == "sigmoid":  # sigmoid激活函数
            net.append(nn.Sigmoid())
        if dropout > 0:  # 残差模块
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch//100))
    toprint = True
    for param_group in optimizer.param_groups:
        if param_group["lr"] != lr:
            param_group["lr"] = lr
            if toprint:
                print("Switching to learning rate %f" % lr)
                toprint = False


class StackedDAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
                 encodeLayer=[400], decodeLayer=[400], activation="relu",
                 dropout=0, tied=False):
        """_summary_
        Args:
            input_dim (int, optional): input dim. Defaults to 784.
            z_dim (int, optional): output dim. Defaults to 10.
            binary (bool, optional): _description_. Defaults to True.
            encodeLayer (list, optional): _description_. Defaults to [400].
            decodeLayer (list, optional): _description_. Defaults to [400].
            activation (str, optional): active function. Defaults to "relu".
            dropout (int, optional): dropout. Defaults to 0.
            tied (bool, optional): _description_. Defaults to False.
        """
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork(  # 构建encoder
            [input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork(  # 构建decoder
            [z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)  # 增加z_dim层
        self._dec = nn.Linear(decodeLayer[-1], input_dim)  # 增加input_dim层
        self._dec_act = None  # Sigmoid active function
        if binary:
            self._dec_act = nn.Sigmoid()
        # print(self.layers)

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x):
        loss = -torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10)) +
                                     (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))

        return loss

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def pretrain(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.2, loss_type="cross-entropy"):
        trloader = trainloader
        valoader = validloader
        daeLayers = []
        for l in range(1, len(self.layers)):  # train by each layer
            infeatures = self.layers[l-1]
            outfeatures = self.layers[l]
            if l != len(self.layers)-1:
                dae = DenoisingAutoencoder(  # 降噪自编码器
                    infeatures, outfeatures, activation=self.activation, dropout=corrupt)
            else:
                dae = DenoisingAutoencoder(
                    infeatures, outfeatures, activation="none", dropout=0)
            print(dae)
            if l == 1:
                dae.fit(trloader, valoader, lr=lr, batch_size=batch_size,
                        num_epochs=num_epochs, corrupt=corrupt, loss_type=loss_type)
            else:
                if self.activation == "sigmoid":
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size,
                            num_epochs=num_epochs, corrupt=corrupt, loss_type="cross-entropy")
                else:
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size,
                            num_epochs=num_epochs, corrupt=corrupt, loss_type="mse")
            data_x = dae.encodeBatch(trloader)
            valid_x = dae.encodeBatch(valoader)
            trainset = Dataset(data_x, data_x)
            trloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            validset = Dataset(valid_x, valid_x)
            valoader = torch.utils.data.DataLoader(
                validset, batch_size=1000, shuffle=False, num_workers=0)
            daeLayers.append(dae)  # size=4
        self.copyParam(daeLayers)

    def copyParam(self, daeLayers):
        if self.dropout == 0:
            every = 2
        else:
            every = 3
        # input layer
        # copy encoder weight
        self.encoder[0].weight.data.copy_(daeLayers[0].weight.data)
        self.encoder[0].bias.data.copy_(daeLayers[0].bias.data)
        self._dec.weight.data.copy_(daeLayers[0].deweight.data)
        self._dec.bias.data.copy_(daeLayers[0].vbias.data)

        for l in range(1, len(self.layers)-2):  # layers:[784 500 500 2000 10]
            # copy encoder weight
            self.encoder[l*every].weight.data.copy_(daeLayers[l].weight.data)
            self.encoder[l*every].bias.data.copy_(daeLayers[l].bias.data)

            # copy decoder weight
            self.decoder[-(l-1)*every -
                         2].weight.data.copy_(daeLayers[l].deweight.data)
            self.decoder[-(l-1)*every -
                         2].bias.data.copy_(daeLayers[l].vbias.data)

        # z layer
        self._enc_mu.weight.data.copy_(daeLayers[-1].weight.data)
        self._enc_mu.bias.data.copy_(daeLayers[-1].bias.data)
        self.decoder[0].weight.data.copy_(daeLayers[-1].deweight.data)
        self.decoder[0].bias.data.copy_(daeLayers[-1].vbias.data)

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10, corrupt=0.3,
            loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Stacked Denoising Autoencoding Layer=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        if loss_type == "mse":
            criterion = MSELoss()
        elif loss_type == "cross-entropy":
            criterion = BCELoss()

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()  # [256, 784]
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)  # 变量化
            z, outputs = self.forward(inputs)

            valid_recon_loss = criterion(outputs, inputs)
            total_loss += valid_recon_loss.data * len(inputs)
            total_num += inputs.size()[0]

        min_train_loss = 100.0
        min_valid_loss = 100.0
        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))
        self.train()  # change to train mode
        for epoch in range(num_epochs):
            # train 1 epoch
            adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                z, outputs = self.forward(inputs_corr)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.data*len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs = self.forward(inputs)

                valid_recon_loss = criterion(outputs, inputs)
                valid_loss += valid_recon_loss.data * len(inputs)

            print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f" % (
                epoch+1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))
            min_valid_loss = min(min_valid_loss, valid_loss / len(validloader.dataset))
            min_train_loss = min(min_train_loss, train_loss / len(trainloader.dataset))
        with open('./logs/PaviaU.log', 'w') as f:
            f.write('min train loss: {}, min valid loss: {}'.format(min_train_loss, min_valid_loss))

    def predict_mnist(self, validloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()  # [256, 784]
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)  # 变量化
            z, outputs = self.forward(inputs)
            i = 1
            plt.figure(figsize=(28,28))
            for j in range(inputs.size(0)):
                ax = plt.subplot(32, 16, i)
                plt.imshow(inputs[j].data.cpu().numpy().reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
                ax = plt.subplot(32, 16, i)
                plt.imshow(outputs[j].data.cpu().numpy().reshape(28,28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                i += 1
            plt.savefig('./picture/{}.jpg'.format(batch_idx + 1))            

    def predict_PaviaU(self, validloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        print('img start...')
        img = []
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()  # [610 * 340, 103]
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)  # 变量化
            z, outputs = self.forward(inputs)
            i = 1
            for j in range(inputs.size(0)):
                # if _[j] != 0:
                img.append(outputs[j].data.cpu().numpy())
                # else:
                #     img.append([0 for i in range(103)])
        img = np.array(img).T
        
        print('data start...')
        data = pd.read_csv('./dataset/PaviaU/raw/PaviaUAllData.csv', header=None)
        data = np.array(data.values[:,:-1]).T
        
        plt.figure(figsize=(610,340), dpi=5)
        plt.axis('off')
        for i in range(len(img)):
            print('{}th start...'.format(i+1), end='')
            ax = plt.subplot(1, 2, 1)
            plt.gray()
            ax.set_title('autoencoder')
            plt.imshow(img[i].reshape(610, 340))
            ax = plt.subplot(1, 2, 2)
            plt.gray()
            ax.set_title('original')
            plt.imshow(data[i].reshape(610, 340))
            plt.savefig('./picture/PaviaU/{}.jpg'.format(i+1))
            print(', {}th picture has been done!'.format(i+1))
                  
    def get_encoder(self, validloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            
        encoder = []
        label = []
        for inputs, _ in validloader:
            inputs = inputs.view(inputs.size(0), -1).float()  # [610 * 340, 103]
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)  # 变量化
            z, outputs = self.forward(inputs)
            for j in range(z.size(0)):
                # if _[j] != 0:
                encoder.append(z[j].data.cpu().numpy())
                label.append(_[j].data.cpu().numpy())
                # else:
                #     encoder.append([0 for i in range(103)])
        encoder = np.array(encoder)
        label = np.array(label)
        return encoder, label
        