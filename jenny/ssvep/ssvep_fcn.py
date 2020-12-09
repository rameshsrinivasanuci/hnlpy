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


class toOneHot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, integer):
        y_onehot = torch.zeros(self.num_classes)
        y_onehot[integer]=1
        return y_onehot

# Load the dataset
class MyTrainingDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(self.root[0])
        self.targets = np.load(self.root[1])

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






