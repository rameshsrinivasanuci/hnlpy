#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:05:45 2020

@author: jenny
"""

# this scripts build a fully connected neural network using SSVEP power to predict accuracy


import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np


class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integer):
        y_onehot = torch.zeros(self.num_classes)
        y_onehot[integer]=1
        return y_onehot
np.random.normal(-1,1,(100,1,1,121))

# Write my own data structure
class MyTrainingDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.expand_dims(np.load(self.root + '/data.npy'), axis=1)
        # self.data = self.data*(np.random.normal(0,100,(len(self.data),1,121)))
        self.targets = np.load(self.root + '/target_rt.npy').astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

root = '/home/jenny/pdmattention/ssvep'
# Load the training set
train_set = MyTrainingDataset('/home/jenny/pdmattention/ssvep',
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                             ]),
                           target_transform = None,)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

# Load the test set
test_set = MyTrainingDataset('/home/jenny/pdmattention/ssvep/test',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ]),
                             target_transform=None, )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# # Sample the data loader
# data, target = next(iter(train_loader))
# data.shape
# plt.imshow(data[1][0])
# torch.argmax(target[1])

# build the neural network
net = torch.nn.Sequential(torch.nn.Linear(121, 300),
                          torch.nn.ReLU(), #this is an activation function
                          torch.nn.Linear(300, 300),
                          torch.nn.ReLU(), #this is an activation function
                          torch.nn.Linear(300, 3)).cuda()

# calculate the loss function
mse_loss = torch.nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# define train_step
def train_step(x, t, net, opt_fn, loss_fn):
    y = net(x.cuda())
    loss = loss_fn(y, t.cuda())
    loss.backward()
    opt_fn.step()
    opt_fn.zero_grad()
    return loss

x,t = next(iter(train_loader))
x = x.float()
loss_ = train_step(x.view(-1,121), t, net, opt, mse_loss)

# make prediction
x,t = next(iter(test_loader))
x = x.float()
y = net(x.view(-1,121).cuda())

# torch.argmax(y[1])
#
# torch.argmax(t[1])




train_accuracy = []
test_accuracy = []
for epoch in range(50):
    net.train()  # training mode
    for x, t in iter(train_loader):
        x=x.float()
        loss_ = train_step(x.view(-1, 121), t, net, opt, mse_loss)

    acc_batch = []
    for x, t in iter(train_loader):
        x = x.float()
        y = net(x.view(-1, 121).cuda())
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        acc_batch.append(batch_accuracy)
    train_accuracy.append(torch.mean(torch.FloatTensor(acc_batch)))

    acc_batch = []
    net.eval()  # evaluation mode
    for x, t in iter(test_loader):
        x = x.float()
        y = net(x.view(-1, 121).cuda())  # This is necessary because the data has shape [1,28,28], but the input layer is [784]
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        acc_batch.append(batch_accuracy)
    test_accuracy.append(torch.mean(torch.FloatTensor(acc_batch)))
    print('Accuracy:', torch.mean(torch.FloatTensor(acc_batch)))

plt.figure()
plt.plot(train_accuracy, label = 'training')
plt.plot(test_accuracy, label = 'testing')
plt.title('two-layer fully conneted neural network')