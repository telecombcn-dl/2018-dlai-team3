import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """ This network maps z (withdrawn from prior distribution) to the pdf in train data with dim `out_dim` """
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=(2, 2)))
        # cnn output size = (input_size - kernel_size + 2*padding)/stride + 1
        # feature map size = [ (28-3+2*0)/(1) + 1 = 26 ] / 2 = 13
        x = F.relu(self.conv2(x))
        # feature map size = (13-2)/1 + 1 = 12
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class Discriminator(nn.Module):
    """ This network classifies its input as either real or fake """

    def __init__(self, input_dim=10, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h1 = F.leaky_relu(self.fc1(x), 0.3)
        h2 = F.leaky_relu(self.fc2(h1), 0.3)
        h3 = self.out_fc(h2)
        return h3
