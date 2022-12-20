# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:00:16 2022

@author: marko
"""
import torch

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        print("in mednistdataset")
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index], self.image_files[index]
    
def cut_image(img):
    img = img[:,0:120, 0:120]
    return img