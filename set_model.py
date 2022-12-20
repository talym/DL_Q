# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:54:08 2022

@author: marko
"""

from monai.networks.nets import DenseNet121
import torchvision.models as models
import torch


def setModel(model_type = "DenseNet121", num_class = 2, device = "cpu"):
    
    if(model_type == "DenseNet121"):
        print("DenseNet121")
        model = DenseNet121(pretrained=True, spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    elif(model_type == "ResNet18"):
        print("ResNet18")
        resnet = models.resnet18(pretrained=True)
        # change first layer
        resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # change last layer
        fc_in = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(fc_in, num_class)    
        model = resnet.to(device)
    print(model_type)
    return(model)

def losFunc(loss_function_type = "BCEWithLogitsLoss" , device = "cpu"):
    weights = [1, 560/280]
    class_weights = torch.FloatTensor(weights).to(device)
    if(loss_function_type == "BCEWithLogitsLoss"):
        loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    elif(loss_function_type == "CrossEntropyLoss"):
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif(loss_function_type == "BCELoss"):        
        loss_function = torch.nn.BCELoss(weight=class_weights)
    return loss_function