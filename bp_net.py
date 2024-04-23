# coding=utf-8
import torch
from torch import nn


class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out