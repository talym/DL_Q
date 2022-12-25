# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:14:44 2022

@author: marko
"""

import pandas as pd
    
import os
import torch
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

from mil.models import APR, AttentionDeepPoolingMil, MILES
from mil.models import RandomForestClassifier, SVC
from mil.metrics import AUC
from mil.metrics import Specificity
from mil.utils.utils import bags2instances
from mil.validators import KFold
from mil.bag_representation import MILESMapping
from mil.validators import LeaveOneOut
from mil.trainer import Trainer
from sklearn.model_selection import train_test_split
from mil.preprocessing import StandarizerBagsList

    
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
from set_model import setModel
import glob

def evaluate_slices(root_folder, volume, device, model):
    
    volume_path = os.path.join(root_folder, volume)
    s_path = os.path.join(volume_path, "slices")
    test_x  = glob.glob(os.path.join(s_path,"*.png"))
    test_y = [0]*len(test_x)

    #Define MONAI transforms, Dataset and Dataloader to pre-process data    
    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ResizeWithPadOrCrop(spatial_size=[160,160]), ScaleIntensity()])
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)
    
    y_pred_trans = Compose([Activations(softmax=True)])

    model.eval()
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
                y_pred.append(pred[i].item())
    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred_)]
    Prob=[]
    for yi in y_pred_act:
        Prob.append(yi.cpu().detach().numpy()[0])
    f = pd.DataFrame({"image name":all_test_name, "Prediction": y_pred , "Prob":Prob})  
    f.to_csv(os.path.join(s_path,"slices.csv"))
    return f
    
    
def volume_quality_score_av(qual_eval):
        score = 0
        number_of_s = qual_eval.shape[0]
        for i in range(number_of_s):
            weight = (1 - qual_eval["Prob"][i])#*(abs((number_of_s/2 - abs(i - number_of_s/2)))/(number_of_s/2))
            score += weight
        score /= number_of_s  
        return score
    
def present_evaluation(results_file, score):
    df = pd.read_csv(results_file)   
    fpr, tpr, thresholds =sklearn.metrics.roc_curve(df["ImageQuality"].fillna(0), df[score].abs())
    roc_auc = sklearn.metrics.roc_auc_score(df["ImageQuality"].fillna(0), df[score].abs())
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    print(thresholds[(fpr < 0.35).sum()])
    chosen_tr = thresholds[(fpr < 0.35).sum()]
    df.loc[df[score] > chosen_tr, 'Pred'] = 1
    df.loc[df[score] <= chosen_tr, 'Pred'] = 0
    ac = sklearn.metrics.balanced_accuracy_score(df["ImageQuality"].fillna(0), df['Pred'])
    df.to_csv(results_file, index=False)
    print(ac)

        
def evaluate_volume_quality(root_folder, results_file, list_of_volumes):
    results_df = pd.read_csv(results_file)    
    results_df["Qaulity Score Av. Slices"] = ""
    score_av_list = []
    for vol_path in list_of_volumes:
        head_tail = os.path.split(vol_path)
        volume_name = head_tail[len(head_tail)-1] 
        volume_path = os.path.join(root_folder, volume_name)
        s_path = os.path.join(volume_path, "slices")
        qual_eval = pd.read_csv(os.path.join(s_path,"slices.csv"))
        score_av = volume_quality_score_av(qual_eval)
        score_av_list.append(score_av)
        
        try:
            index = results_df[results_df['file']==volume_name].index.tolist()[0]
        except:
            print("Cannot find volume: ", volume_name)
        else:
            results_df.loc[index, 'Qaulity Score Av. Slices'] = score_av
    results_df.to_csv(results_file, index=False)
    return score_av_list

def train_mil(root_folder, list_of_volumes):
    results_df = pd.read_csv(results_file)
    y = []
    bags = []
    vol_names = []
    for vol_path in list_of_volumes:
        head_tail = os.path.split(vol_path)
        volume_name = head_tail[len(head_tail)-1] 
        volume_path = os.path.join(root_folder, volume_name)
        s_path = os.path.join(volume_path, "slices")
        qual_eval = pd.read_csv(os.path.join(s_path,"slices.csv"))
        bag = np.array([a for a in qual_eval["Prob"] if a !=-1])
        bag = np.expand_dims(bag, 1)

        bags.append(bag)
        try:
            index = results_df[results_df['file']==volume_name].index.tolist()[0]
        except:
            print("Cannot find volume: ", volume_name)
        else:
            vol_names.append(volume_name)
            if(results_df['ImageQuality'][index]!=1):
                y.append(0)  
            else:
                y.append(1)   

    bags_train, bags_test, y_train,  y_test, name_train, name_test = train_test_split(bags,y, vol_names, train_size=0.66, shuffle=False)    
    
    results_df["Train"]="No"
    for n in name_train:
        ks = results_df['file'].str.contains(n)
        results_df.loc[ks, 'Train']  = "Yes"
    
    # instantiate trainer
    trainer = Trainer()
    
    # preparing trainer
    metrics = ['acc', AUC]
    #model = SVC(kernel='linear', C=1, class_weight='balanced')
    model = RandomForestClassifier()
    
    pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', MILESMapping())]
    trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)
 
    
    # fitting trainer
    valid = LeaveOneOut()
    history = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)
    
    # printing validation results for each fold
    print(history['metrics_val'])
    
    # predicting metrics for the test set
    print(trainer.predict_metrics(bags_test, y_test))
    out = trainer.predict(bags_test)
    print(out)
    results_df["Pred MIL"]=""
    for n,o in zip(name_test, out):
        ks = results_df['file'].str.contains(n)
        results_df.loc[ks, 'Pred MIL']  = o
    
    results_df.to_csv(results_file, index=False)

    
if __name__ == "__main__":
    #global parameters 
    ROOT_FOLDER = "C:\\Users\\marko\\first rotation\\project\\ResultsSeg_vTry"
    list_of_volumes = glob.glob(os.path.join(ROOT_FOLDER,"*.nii.gz"))
    results_folder = os.path.join(ROOT_FOLDER, "results")
    results_file = os.path.join(results_folder, "combined_results.csv")        
    volume = "Pat549_Se07_Res0.7813_0.7813_Spac5.0.nii.gz"
    MODELS_ROOT = "C:\\Users\\marko\\first rotation\\project\\DL"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setModel("ResNet18")      
    #Get pretrained model 
    model_path = os.path.join(MODELS_ROOT, "best_metric_modelResNet18_BCEWithLogitsLoss_pretrained_200.pth")
    model.load_state_dict(torch.load(model_path))
    #run_type = "Single"
    #run_type = "Batch"   
    run_type = "None"
    
    if(run_type == "Single"):
        qual_eval = evaluate_slices(ROOT_FOLDER, volume, device, model)
        score = volume_quality_score_av(qual_eval)
        print(score)        
    elif(run_type == "Batch"):
        slice_eval_list = []
        for vol_path in list_of_volumes:
            head_tail = os.path.split(vol_path)
            chosen_vol_name = head_tail[len(head_tail)-1] 
            qual_eval = evaluate_slices(ROOT_FOLDER, chosen_vol_name, device, model)

    score_av_list = evaluate_volume_quality(ROOT_FOLDER, results_file, list_of_volumes)
    present_evaluation(results_file, "Qaulity Score Av. Slices")

    #multiple instance learning
    train_mil(ROOT_FOLDER, list_of_volumes)
    
    
