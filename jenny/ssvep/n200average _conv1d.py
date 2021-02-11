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


#
# # randomly split the all dataset
# import random
# from sklearn.model_selection import train_test_split
# ind = np.arange(0,15172)
# x_train, x_test = train_test_split(ind, test_size=0.2)
# train_n200new = n200all[x_train,:]
# np.save('/home/ramesh/pdmattention/ssvep/n200new', train_n200new)
# test_n200new = n200all[x_test,:]
# np.save('/home/ramesh/pdmattention/ssvep/test/n200new', test_n200new)
#
# train_twofreq = data_twofreqall[x_train,:]
# np.save('/home/ramesh/pdmattention/ssvep/data_twofreqnew', train_twofreq)
# test_twofreq = data_twofreqall[x_test,:]
# np.save('/home/ramesh/pdmattention/ssvep/test/data_twofreqnew', test_twofreq)
#
# train_target = targetall [x_train]
# np.save('/home/ramesh/pdmattention/ssvep/target_new', train_target)
# test_target = targetall [x_test]
# np.save('/home/ramesh/pdmattention/ssvep/test/target_new', test_target)
#
#
# train_data = data[indices]
# test_data = np.delete(data,indices)
#

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

        self.data1 = np.expand_dims(np.load(self.root + '/data_twofreqnew.npy'), axis=1)
        # self.data = np.expand_dims(np.load(self.root + '/Data_freqs.npy'), axis=1)
        self.data2 = np.expand_dims(np.load(self.root + '/n200new.npy'), axis=1)
        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        self.data = np.dstack((self.data1, self.data2))
        # self.data = np.concatenate((self.data1, self.data2), axis=1)


        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        # self.data = np.dstack((self.data1, self.data2))
        self.targets = np.load(self.root + '/target_new.npy').astype(int)
        # oversampling the low acc data
        # self.data= np.dstack((self.data, self.data3))
        # self.data = self.data*(np.random.normal(0,100,(len(self.data),1,244)))
        # self.ind = np.load(self.root + '/dropout_acc.npy')
        # self.data = np.delete(self.data,self.ind, axis=0)
        #
        self.incorrect_ind = np.where(self.targets == 0)[0]
        self.correct_ind = np.where(self.targets == 1)[0]
        self.testlist = list(islice(cycle(self.incorrect_ind), 6711))
        self.oversample = self.data[self.testlist,:,:]
        self.data = np.vstack((self.data, self.oversample))
        self.targets = np.hstack((self.targets, np.zeros(6711))).astype(int)





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

        # self.data = np.expand_dims(np.load(self.root + '/data_n200adapted.npy'), axis=1)
        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        # self.data = np.dstack((self.data1, self.data2))
        self.targets = np.load(self.root + '/target_new.npy').astype(int)

        self.data1 = np.expand_dims(np.load(self.root + '/data_twofreqnew.npy'), axis=1)
        # self.data = np.expand_dims(np.load(self.root + '/Data_freqs.npy'), axis=1)
        self.data2 = np.expand_dims(np.load(self.root + '/n200new.npy'), axis=1)
        self.data = np.dstack((self.data1, self.data2))
        # self.data = np.expand_dims(np.load(self.root + '/Data_freqs.npy'), axis=1)
        # self.data2 = np.expand_dims(np.load(self.root + '/n200param.npy'), axis=1)
        # self.data3 = np.expand_dims(np.load(self.root + '/data_n200raw.npy'), axis=1)
        # self.data = np.dstack((self.data1, self.data2))
        # self.data = np.concatenate((self.data1, self.data2), axis=1)

        # self.incorrect_ind = np.where(self.targets == 0)[0]
        # self.correct_ind = np.where(self.targets == 1)[0]
        # self.testlist = list(islice(cycle(self.incorrect_ind), 6430))
        # self.oversample = self.data[self.testlist,:,:]
        # self.data = np.vstack((self.data, self.oversample))



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


class CNN1D(torch.nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(1,6, kernel_size=4, stride=1)
        self.conv2 = torch.nn.Conv1d(6, 110, 4, 1)
        self.dropout1 = torch.nn.Dropout(p = .25)
        self.dropout2 = torch.nn.Dropout(p = 0.5)
        self.pool = torch.nn.MaxPool1d((2))
        self.fc1 = torch.nn.Linear(13090,2)  #247
    def forward(self, x):
        x = torch.squeeze(x, axis=1)
        # x = torch.transpose(x, 1,2)
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
        # x = torch.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
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
plt.title('CNN1D using raw SSVEP+N200 parameters as 1 channel')

