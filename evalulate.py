# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:12:14 2022

@author: marko
"""
import pandas as pd
if __name__ == "__main__":
    
    
    import os
    import torch
    from sklearn.metrics import classification_report
    
    
    from monai.data import decollate_batch, DataLoader
    from monai.transforms import (
        Activations,
        EnsureChannelFirst,
        Compose,
        LoadImage,
        ScaleIntensity,
        ResizeWithPadOrCrop,
    )
    from MedNISTDataset import MedNISTDataset
    from getdata import getData
    from set_model import setModel
    
    #define directories 
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir, "best_metric_modelResNet18_BCEWithLogitsLoss_pretrained_200.pth")
    
    class_names = ["good", "bad"]
    num_class = len(class_names)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = setModel("ResNet18")
    
    #Get pretrained model 
    model.load_state_dict(torch.load(model_path))
    
    #upload data 
    num_class, train_x,train_y, val_x, val_y, test_x, test_y, root_dir, class_names  = getData()
    print(test_x)

    #Define MONAI transforms, Dataset and Dataloader to pre-process data
    
    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ResizeWithPadOrCrop(spatial_size=[160,160]), ScaleIntensity()])

    train_ds = MedNISTDataset(train_x, train_y, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)
    y_pred_trans = Compose([Activations(softmax=True)])

    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)
    
    #on train
    model.eval()
    y_true = []
    y_pred = []
    all_test_name = []
    with torch.no_grad():
        y_pred_ = torch.tensor([], dtype=torch.float32, device=device)
        for tr_data in train_loader:
            test_images, test_labels, test_name = (
                tr_data[0].to(device),
                tr_data[1].to(device),
                tr_data[2],
            )            
            y_pred_ = torch.cat([y_pred_, model(test_images)], dim=0)
            pred = model(test_images).argmax(dim=1)
            
            for i in range(len(pred)):                
                head_tail = os.path.split(test_name[i])            
                test_name[i] = head_tail[len(head_tail)-1] 
                all_test_name.append(test_name[i])
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())  
    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred_)]
    print(len(all_test_name), len(y_true), len(y_pred))
    f = pd.DataFrame({"image name":all_test_name, "label": y_true, "Prediction": y_pred, "Prob":y_pred_act})  
    f.to_csv("train_pred.csv")
    print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))
    
    #on test
    model.eval()
    y_true = []
    y_pred = []
    all_test_name = []
    with torch.no_grad():
        y_pred_ = torch.tensor([], dtype=torch.float32, device=device)

        for test_data in test_loader:
            test_images, test_labels, test_name = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2],
            )
            pred = model(test_images).argmax(dim=1)
            y_pred_ = torch.cat([y_pred_, model(test_images)], dim=0)

            for i in range(len(pred)):                
                head_tail = os.path.split(test_name[i])            
                test_name[i] = head_tail[len(head_tail)-1] 
                all_test_name.append(test_name[i])
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred_)]
    f = pd.DataFrame({"image name":all_test_name, "label": y_true, "Prediction": y_pred , "Prob":y_pred_act})  
    f.to_csv("test_pred.csv")
    print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))