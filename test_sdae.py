import argparse
import sys

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms

from lib.datasets import MNIST
from lib.stackedDAE import StackedDAE

sys.path.append("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    # according to the released code, mnist data is multiplied by 0.02
    # 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
    # so add a customized Scale transform to multiple by 5.1
    train_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
                      encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                      dropout=0)
    print(sdae)
    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
                  num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="mse")
    sdae.fit(train_loader, test_loader, lr=args.lr,
             num_epochs=args.epochs, corrupt=0.2, loss_type="mse")
    sdae.save_model("./model/mnist/sdae.pt")

    sdae.load_model('./model/mnist/sdae.pt')
    sdae.predict_mnist(test_loader)
