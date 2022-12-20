# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:15:00 2022

@author: marko
"""
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import classification_report


from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ResizeWithPadOrCrop,
)
from MedNISTDataset import MedNISTDataset
from getdata import getData
from set_model import setModel
from set_model import losFunc

if __name__ == "__main__":
   

    num_class, train_x,train_y, val_x, val_y, test_x, test_y, root_dir, class_names  = getData()
    
    #Define MONAI transforms
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ResizeWithPadOrCrop(spatial_size=[160,160]),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )    
    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ResizeWithPadOrCrop(spatial_size=[160,160]),ScaleIntensity()])
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])        
    
    #Define MONAI Dataset and Dataloader to pre-process data
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_ds_val = MedNISTDataset(train_x, train_y, val_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=300, shuffle=True, num_workers=10)
    train_loader_val = DataLoader(
        train_ds_val, batch_size=300, shuffle=True, num_workers=10)
    
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=300, num_workers=10)
    
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(
    test_ds, batch_size=300, num_workers=10)
    
    #Define network and optimizer    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device) 
    model_type = "DenseNet121"
    model = setModel(model_type = model_type)
    loss_function_type = "BCEWithLogitsLoss"
    loss_function = losFunc(loss_function_type = loss_function_type)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    max_epochs = 200
    val_interval = 1
    auc_metric = ROCAUCMetric()
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    auc_metric_values = []
    acc_metric_values = []
    tr_auc_metric_values = []
    tr_acc_metric_values = []    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = torch.nn.functional.one_hot(labels.to(torch.int64), 2)
            labels = labels.float()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                #on train
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                i = 0
                for tr_data in train_loader_val:
                    i+=1
                    val_images, val_labels = (
                        tr_data[0].to(device),
                        tr_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                print("i", i)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                print("result", result)
                auc_metric.reset()
                del y_pred_act, y_onehot
                tr_auc_metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                tr_acc_metric_values.append(acc_metric)
                
                #on val
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                i = 0
                for val_data in val_loader:
                    i+=1
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                print("i", i)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                print("result", result)
                auc_metric.reset()
                del y_pred_act, y_onehot
                auc_metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                acc_metric_values.append(acc_metric)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
    
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    
    #Plot the loss and metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Av Loss " + model_type + " " + loss_function_type)
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(auc_metric_values))]
    y = [auc_metric_values, tr_auc_metric_values]
    plt.xlabel("epoch")
    plt.plot(x, np.transpose(y))
    plt.subplot(1, 3, 3)
    plt.title("Val ACC")
    x = [val_interval * (i + 1) for i in range(len(acc_metric_values))]
    y = [acc_metric_values, tr_acc_metric_values]
    plt.xlabel("epoch")
    plt.plot(x, np.transpose(y))
    plt.show()

    #Evaluate the model on test dataset
    model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
    
    #on train
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for tr_data in train_loader:
            test_images, test_labels = (
                tr_data[0].to(device),
                tr_data[1].to(device),
            )
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))
    
    #on test
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))
    
    
   