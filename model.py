import torch
import torch.nn as nn


class OutputLayer(nn.Module):
    def __init__(self, in_channels, number_of_classes):
        super().__init__()
        self.output_conv1 = nn.Conv2d(in_channels, (number_of_classes + 1), kernel_size=3)
        self.output_conv2 = nn.Conv2d(in_channels, 4, kernel_size=3)

    def forward(self, x):
        return torch.cat((self.output_conv1(x), self.output_conv2(x)), dim=1)


class Net(nn.Module):
    def __init__(self, number_of_classes):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            # 236 x 236 x 3
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(),
            # 232 x 232 x 16
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 116 x 116 x 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            # 112 x 112 x 32
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56 x 56 x 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            # 52 x 52 x 64
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 26 x 26 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            # 24 x 24 x 128
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12 x 12 x 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            # 10 x 10 x 128
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5 x 5 x 128
            OutputLayer(128, number_of_classes)
        )

    def forward(self, x):
        return self.model(x)
