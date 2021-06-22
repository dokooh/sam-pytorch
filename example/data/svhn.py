import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout

class Svhn:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        # image preprocess
        normaliz = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                             std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

        train_transform = transforms.Compose([])
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normaliz)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normaliz])

        train_set = torchvision.datasets.SVHN(root='./data',
                                          split='train',
                                          transform=train_transform,
                                          download=True)

        extra_set = datasets.SVHN(root='./data',
                                      split='extra',
                                      transform=train_transform,
                                      download=True)

        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        data = np.concatenate([train_set.data, extra_set.data], axis=0)
        labels = np.concatenate([train_set.labels, extra_set.labels], axis=0)
        train_set.data = data
        train_set.labels = labels

        test_set = datasets.SVHN(root='./data',
                                     split='test',
                                     transform=test_transform,
                                     download=True)

        # Data Loader (Input Pipeline)
        self.train = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=2)

        self.test = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=2)
        
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def _get_statistics(self):
        train_set = torchvision.datasets.SVHN(root='./svhn', split='train', download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
