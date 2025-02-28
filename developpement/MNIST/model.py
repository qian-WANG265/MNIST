import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc2 = nn.Linear(1600, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))       # Second convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling
        x = nn.Flatten()(x)         # Flatten the output of the convolution layers
        x = F.relu(self.fc2(x))         # First fully connected layer followed by
        x = self.fc3(x)
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(16 * 4 * 4,-1)
        return x