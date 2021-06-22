import os
import tarfile
import torch
import numpy as np
import math

import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F

import matplotlib.pyplot as plt
%matplotlib inline

!pip install torchsummary
from torchsummary import summary

from scipy.io import loadmat
import shutil
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
#from util.misc import CSVLogger
import csv
#from util.cutout import Cutout

from tqdm import tqdm

# image preprocess
normaliz = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
                                     
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normaliz)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normaliz])
    
num_classes = 10
train_dataset = datasets.SVHN(root='data/',
                              split='train',
                              transform=train_transform,
                              download=True)

extra_dataset = datasets.SVHN(root='data/',
                              split='extra',
                              transform=train_transform,
                              download=True)

# Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
train_dataset.data = data
train_dataset.labels = labels

test_dataset = datasets.SVHN(root='data/',
                             split='test',
                             transform=test_transform,
                             download=True)
                             
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
