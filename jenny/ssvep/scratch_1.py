
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

# Write my own data structure
class MyTrainingDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.expand_dims(np.load(self.root + '/data.npy'), axis=1)
        self.targets = np.load(self.root + '/target.npy').astype(int)

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

net = torch.nn.Sequential(torch.nn.Linear(121, 100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100,2),
                        torch.nn.ReLU())
mse_loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters())

def train_step(x, t, net, opt_fn, loss_fn):
    y = net(x)
    loss = loss_fn(y, t)
    loss.backward()
    opt_fn.step()
    opt_fn.zero_grad()
    return loss

x,t = next(iter(train_loader))
x=x.float()
train_step(x.view(-1,121), t, net, opt, mse_loss)

for x,t in iter(test_loader):
    x = x.float()
    y = net(x.view(-1,121))

y.argmax(1)
torch.mean((t == y.argmax(1)).float())


acc_hist_train = []
acc_hist_test = []
for epoch in range(50):
    acc_batch = []
    for x,t in iter(train_loader):
        x = x.float()
        loss_ = train_step(x.view(-1,121), t, net, opt, mse_loss)
        y = net(x.view(-1,121))
        acc_batch.append(torch.mean((t == y.argmax(1)).float()))
    acc_hist_train.append(torch.mean(torch.FloatTensor(acc_batch)))
    print(loss_)

    acc_batch = []
    for x,t in iter(test_loader):
        loss_ = 0
        x = x.float()
        y = net(x.view(-1,121))
        acc_batch.append(torch.mean((t == y.argmax(1)).float()))
    acc_hist_test.append(torch.mean(torch.FloatTensor(acc_batch)))
    print(acc_hist_test[-1])

acc_hist_train
acc_hist_test
plt.figure()
plt.plot(acc_hist_train, label = 'training')
plt.plot(acc_hist_test, label = 'testing')
plt.title('single-layer fully conneted neural network')