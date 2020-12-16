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

class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integer):
        y_onehot = torch.zeros(self.num_classes)
        y_onehot[integer]=1
        return y_onehot


# Write my own data structure
class MyTrainingDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data = np.expand_dims(np.load(self.root + '/data_n200adapted.npy'), axis=1)
        # self.data2 = np.expand_dims(np.load(self.root + '/n200param.npy'), axis=1)
        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        # self.data = np.dstack((self.data1, self.data2))
        self.targets = np.load(self.root + '/target_rt.npy').astype(int)
        # oversampling the low acc data
        # self.data= np.dstack((self.data, self.data3))
        # self.data = self.data*(np.random.normal(0,100,(len(self.data),1,244)))
        # self.ind = np.load(self.root + '/dropout_acc.npy')
        # self.data = np.delete(self.data,self.ind, axis=0)
        #
        # # oversample the inaccurate trials
        # self.incorrect_ind = np.where(self.targets == 0)[0]
        # self.correct_ind = np.where(self.targets == 1)[0]
        # self.testlist = list(islice(cycle(self.incorrect_ind), 6430))
        # self.oversample = self.data[self.testlist,:,:]
        # self.data = np.vstack((self.data, self.oversample))
        # self.targets = np.hstack((self.targets, np.zeros(6430))).astype(int)
        #


        # self.targets = np.hstack((self.targets, np.zeros(6430))).astype(int)





        # self.data = np.expand_dims(np.load(self.root + '/data_real.npy'), axis=1)

        # self.data = self.data*(np.random.normal(0,100,(len(self.data),1,242)))

        # self.targets = np.delete( self.targets,self.ind, axis=0)


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

class MyTestingDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data = np.expand_dims(np.load(self.root + '/data_n200 adapted .npy'), axis=1)
        # self.data2 = np.expand_dims(np.load(self.root + '/n200param.npy'), axis=1)
        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        # self.data = np.dstack((self.data1, self.data2))
        self.targets = np.load(self.root + '/target_rt.npy').astype(int)




        # self.data = np.expand_dims(np.load(self.root + '/data_real.npy'), axis=1)

        # self.data = self.data*(np.random.normal(0,100,(len(self.data),1,242)))

        # self.targets = np.delete( self.targets,self.ind, axis=0)


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
test_set = MyTestingDataset('/home/jenny/pdmattention/ssvep/test',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ]),
                             target_transform=None, )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# Sample the data loader
data, target = next(iter(train_loader))
data.shape
# plt.imshow(data[1][0])
# torch.argmax(target[1])

# build the neural network
# net = torch.nn.Sequential(torch.nn.Linear(121, 242),
#                           torch.nn.ReLU(), #this is an activation function
#                           torch.nn.Linear(242, 242),
#                           torch.nn.ReLU(), #this is an activation function
#                           torch.nn.Linear(242, 2)).cuda()

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(242,600)
        self.dropout = torch.nn.Dropout(p = .5)
        self.fc2 = torch.nn.Linear(600,600)
        self.fc3 = torch.nn.Linear(600,3)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = net().cuda()


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
loss_ = train_step(x.view(-1,242), t, net, opt, mse_loss)

# make prediction
x,t = next(iter(test_loader))
x = x.float()
y = net(x.view(-1,242).cuda())
# model_arch = make_dot(y.mean(), params = dict(net.named_parameters())
# Source(model_arch).render(root)
# make_dot(y, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")
# torch.argmax(y[1])
#
# torch.argmax(t[1])




train_accuracy = []
test_accuracy = []

train_precision = []
test_precision = []

for epoch in range(100):
    net.train()  # training mode
    # for x, t in iter(test_loader):
    #     x=x.float()
        # loss_ = train_step(x.view(-1, 121), t, net, opt, mse_loss)

    acc_batch = []
    prc_batch = []
    for x, t in iter(train_loader):
        x = x.float()
        loss_ = train_step(x.view(-1, 242), t, net, opt, mse_loss)
        y = net(x.view(-1, 242).cuda())
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        tp = torch.sum(torch.logical_and(t.cuda() == y.argmax(1).cuda(), t.cuda() ==1).float())
        fp = torch.sum(torch.logical_and(y.argmax(1).cuda()==1, t.cuda() ==0).float())
        batch_precision = tp / (tp+fp)
        acc_batch.append(batch_accuracy)
        prc_batch.append(batch_precision)
    train_accuracy.append(torch.mean(torch.FloatTensor(acc_batch)))
    train_precision.append(torch.mean(torch.FloatTensor(prc_batch)))
    print('Loss:',loss_)
    print('Train Acc:', torch.mean(torch.FloatTensor(acc_batch)))
    print('Train Prc:', torch.mean(torch.FloatTensor(prc_batch)))

    acc_batch = []
    prc_batch = []
    net.eval()  # evaluation mode
    for x, t in iter(test_loader):
        x = x.float()
        y = net(x.view(-1, 242).cuda())  # This is necessary because the data has shape [1,28,28], but the input layer is [784]
        batch_accuracy = torch.mean((t.cuda() == y.argmax(1).cuda()).float())
        tp = torch.sum(torch.logical_and(t.cuda() == y.argmax(1).cuda(), t.cuda() ==1).float())
        fp = torch.sum(torch.logical_and(y.argmax(1).cuda()==1, t.cuda() ==0).float())
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
# plt.plot(train_precision, label = 'training precision')
# plt.plot(test_precision, label = 'testing precision')
plt.plot()
plt.legend(loc='best')
plt.title('two-layer FCN using 242 features on RT')



plt.figure()
plt.plot(train_accuracy, label = 'training',color = 'red')
plt.plot(test_accuracy, label = 'testing', color = 'green')
plt.legend(loc='best')
plt.title('Using Random Noise on Accuracy')
#
