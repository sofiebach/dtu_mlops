from torch import nn
import torch.nn.functional as F
import torch

kernel_size = 5
channel_sizes = [1, 6, 16]
hidden_sizes = [256, 120, 84]
output_size = 10
dropout_rate = 0.2

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size)
        self.conv2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # 5*5 from image dimension
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3 = nn.Sequential(nn.Linear(hidden_sizes[2], output_size),nn.LogSoftmax(dim=1))
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.dropout(F.max_pool2d(F.relu(self.conv1(x)), (2, 2)))
        #print('x shape', x.shape)
        # If the size is a square, you can specify with a single number
        x = self.dropout(F.max_pool2d(F.relu(self.conv2(x)), 2))
        #print('x shape2', x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        #print('x shape3', x.shape)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

print('test')