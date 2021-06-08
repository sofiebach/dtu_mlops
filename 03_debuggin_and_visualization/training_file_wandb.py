from torch import nn
import torch.nn.functional as F
import torch
import sys
import argparse
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
import wandb


wandb.init()
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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('./.pytorch/MNIST_data/', download=True, train=True, transform=transform)
test_set = datasets.MNIST('./.pytorch/MNIST_data/', download=True, train=False, transform=transform)

print("Training day and night")
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=0.003)
# add any additional argument that you want
args = parser.parse_args(sys.argv[2:])
print(args)
        
model = MyAwesomeModel()
wandb.watch(model, log_freq=100)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
steps = 0

train_losses, test_losses = [], []
loss_epoch = []
epoch_no = []
for e in range(epochs):
    print("Starting epoch ", e+1)
    running_loss = 0
    for images, labels in trainloader:
        model.train()
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss})
        train_losses.append(loss.item())
    wandb.log({"Input images" : [wandb.Image(i) for i in images]})

    print('Loss: ', np.mean(train_losses))

    # for epoch vs loss plot 
    epoch_no.append(e+1)
    loss_epoch.append(np.mean(train_losses))

torch.save(model.state_dict(), 'checkpoint.pth') # save model 
plt.plot(epoch_no, loss_epoch, label='Training loss')
plt.xlabel('Epoch number')
plt.legend()
plt.show()   