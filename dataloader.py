#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#filename = 'data/train.csv'


class DigitDataset(Dataset):
    def __init__(self,filename,transforms=True):
        #super(DigitDataset,self).__init__()
        self.df = pd.read_csv(filename)
        if transforms == True:
            self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(25),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=0., std=1.)
                    ])
        else:
            self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)
                    ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        item = np.array(self.df.iloc[idx]).astype(np.float32)
        x = item[1:].reshape((28,28))
        x = self.transforms(x)
        y = item[0]
        return x, y


def show_samples(batch):
    fig, axes = plt.subplots(2,4,figsize=((8,5)))
    axes = axes.ravel()
    for i in range(8):
        axes[i].imshow(batch[0][i].squeeze())
        axes[i].set_title(int(batch[1][i]))
 

