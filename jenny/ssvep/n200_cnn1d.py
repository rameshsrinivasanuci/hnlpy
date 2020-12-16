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
from itertools import cycle, islice

root = '/home/jenny/pdmattention/ssvep'

class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integer):
        y_onehot = torch.zeros(self.num_classes)
        y_onehot[integer]=1
        return y_onehot


# Write my own data structure

# Write my own data structure
class MyTrainingDataset(Dataset):
    def __init__(self, root, transforms=None, target_transforms=None):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.data = np.load(self.root + '/data_n200bychan.npy')
        # self.data2 = np.expand_dims(np.load(self.root + '/n200param.npy'), axis=1)
        # self.data = np.hstack((self.data1,self.data2))
        self.targets = np.load(self.root + '/target.npy').astype(int)

        # oversample the inaccurate trials
        self.incorrect_ind = np.where(self.targets == 0)[0]
        self.correct_ind = np.where(self.targets == 1)[0]
        self.testlist = list(islice(cycle(self.incorrect_ind), 6430))
        self.oversample = self.data[self.testlist,:,:]
        self.data = np.vstack((self.data, self.oversample))
        self.targets = np.hstack((self.targets, np.zeros(6430))).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transforms:
            sample = self.transforms(sample)
        if self.target_transforms:
            target = self.target_transforms(target)

        return sample, target

class MyTestingDataset(Dataset):
    def __init__(self, root, transforms=None, target_transforms=None):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.data = np.load(self.root + '/data_n200bychan.npy')
        # self.data2 = np.expand_dims(np.load(self.root + '/n200param.npy'), axis=1)
        # self.data = np.hstack((self.data1,self.data2))
        self.targets = np.load(self.root + '/target.npy').astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transforms:
            sample = self.transforms(sample)
        if self.target_transforms:
            target = self.target_transforms(target)

        return sample, target


# Load the training set
train_set = MyTrainingDataset('/home/jenny/pdmattention/ssvep',
                              transforms=
                                  transforms.ToTensor(),
                           target_transforms = None,)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)





# Load the test set
test_set = MyTestingDataset('/home/jenny/pdmattention/ssvep/test',
                             transforms=transforms.Compose([transforms.ToTensor(),]),
                             target_transforms=None, )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# Sample the data loader
data, target = next(iter(train_loader))
data.shape

# buildling the network

class CNN1D(torch.nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(119,300, kernel_size=2, stride=1)
        self.conv2 = torch.nn.Conv1d(300, 110, 4, 1)
        self.dropout1 = torch.nn.Dropout(p = .25)
        self.dropout2 = torch.nn.Dropout(p = 0.5)
        self.pool = torch.nn.MaxPool1d((2))
        self.fc1 = torch.nn.Linear(27280,2)  #247
    def forward(self, x):
        x = torch.squeeze(x, axis=1)
        x = torch.transpose(x, 1,2)
        x = self.conv1(x)
        x = torch.relu(x)
        x= self.conv2(x)
        x = self.pool(x)
        # self.linear_input_size = x.view(-1, x.shape[-1]).shape[0]
        # self.print(self.linear_input_size)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
net = CNN1D().cuda()


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


# test it out

x,t = next(iter(train_loader))
x = x.float()

loss_ = train_step(x, t, net, opt, mse_loss)


x,t = next(iter(test_loader))
x = x.float()
y = net(x.cuda())


# go through epochs

train_accuracy = []
test_accuracy = []
train_precision = []
test_precision = []
for epoch in range(50):
    net.train()  # training mode
    # for x, t in iter(test_loader):
    #     x=x.float()
        # loss_ = train_step(x.view(-1, 121), t, net, opt, mse_loss)

    acc_batch = []
    prc_batch = []

    for x, t in iter(train_loader):
        x = x.float()
        loss_ = train_step(x, t, net, opt, mse_loss)
        y = net(x.cuda())
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        tp = torch.sum(torch.logical_and(t.cuda() == y.argmax(1).cuda(), t.cuda() == 1).float())
        fp = torch.sum(torch.logical_and(y.argmax(1).cuda() == 1, t.cuda() == 0).float())
        batch_precision = tp / (tp + fp)
        acc_batch.append(batch_accuracy)
        prc_batch.append(batch_precision)
    train_accuracy.append(torch.mean(torch.FloatTensor(acc_batch)))
    train_precision.append(torch.mean(torch.FloatTensor(prc_batch)))
    print('Loss:', loss_)
    print('Train Acc:', torch.mean(torch.FloatTensor(acc_batch)))
    print('Train Prc:', torch.mean(torch.FloatTensor(prc_batch)))

    acc_batch = []
    prc_batch = []

    net.eval()  # evaluation mode
    for x, t in iter(test_loader):
        x = x.float()
        y = net(x.cuda())  # This is necessary because the data has shape [1,28,28], but the input layer is [784]
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        tp = torch.sum(torch.logical_and(t.cuda() == y.argmax(1).cuda(), t.cuda() == 1).float())
        fp = torch.sum(torch.logical_and(y.argmax(1).cuda() == 1, t.cuda() == 0).float())
        batch_precision = tp / (tp + fp)
        acc_batch.append(batch_accuracy)
        prc_batch.append(batch_precision)
    test_accuracy.append(torch.mean(torch.FloatTensor(acc_batch)))
    test_precision.append(torch.mean(torch.FloatTensor(prc_batch)))
    print('Test Accuracy:', torch.mean(torch.FloatTensor(acc_batch)))
    print('Test Prc:', torch.mean(torch.FloatTensor(prc_batch)))

plt.figure()
plt.plot(train_accuracy, ls = '--', label = 'training accuracy')
plt.plot(test_accuracy, ls = '--',label = 'testinging accuracy')
plt.plot(train_precision, label = 'training precision')
plt.plot(test_precision, label = 'testing precision')
plt.plot()
plt.legend(loc='best')
plt.title('CNN1D using N200 time series by channel')

