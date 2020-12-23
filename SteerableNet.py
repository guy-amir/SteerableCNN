import torch.nn as nn
from SteerableCNN import SteerableCNN


class SteerableNet(nn.Module):
    def __init__(self, filter_size, truncation, beta):

        super(SteerableNet, self).__init__()
        padding_size = int((filter_size - 1) / 2)
        self.filters1 = SteerableCNN(1, 140, filter_size, padding=padding_size, truncation=truncation, beta=beta, angular_freqs=None)
        self.linear_comb = nn.Conv2d(self.filters1.out_channels(), 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(729,10)

    def forward(self, x):
        x = self.filters1(x)
        x = self.linear_comb(x)
        x = self.sigmoid(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
