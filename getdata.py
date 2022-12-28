# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 23:04:49 2022

@author: marko
"""

import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import csv


from monai.config import print_config

from monai.utils import set_determinism

def getData():
    # Copyright 2020 MONAI Consortium
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #     http://www.apache.org/licenses/LICENSE-2.0
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.


    
    
    print_config()
    
    root_dir = os.getcwd()
    data_dir  = os.path.join(root_dir, "SliceClass\\TrainValidImg")
    test_data_dir  = os.path.join(root_dir, "SliceClass\\Test_img")
    training_validation_data_file = os.path.join(root_dir, "SliceClass\\data_2classes.csv")
    test_data_file = os.path.join(root_dir, "SliceClass\\data_test_2class.csv")
    print(root_dir, data_dir )
    
    set_determinism(seed=0)
    
    #upload data 
    class_names = ["good", "bad"]
    num_class = len(class_names)
    image_files = [[],[]] 
    image_files_val = [[],[]]
    image_files_test = [[],[]]
    with open(training_validation_data_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            decision = ''
            img_name = ''
            train_test = ''
            for (k,v) in row.items(): # go over each column name and value                  
                if(k == "decision"): decision = v
                if(k == "img name"): 
                    v = v.replace('.','_')
                    v = v.replace( '_png','.png')
                    img_name = v
                    file_path = os.path.join(data_dir, img_name)
                    if os.path.isfile(file_path)!=True:
                        print("File doesn't exist ")
                        img_name = ''
                if(k == "TrainTest"): train_test = v
            if(train_test == '0' and img_name!=''):
                if(decision == "y"): 
                    image_files[0].append(file_path)
                if(decision == "n"): 
                    image_files[1].append(file_path)
            else:
                if(train_test == '1' and img_name!=''):
                    if(decision == "y"): 
                        image_files_val[0].append(os.path.join(data_dir, img_name))
                    if(decision == "n"): 
                        image_files_val[1].append(os.path.join(data_dir, img_name))
    with open(test_data_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            decision = ''
            img_name = ''
            for (k,v) in row.items(): # go over each column name and value                  
                if(k == "decision"): decision = v
                if(k == "img name"): 
                    v = v.replace('.','_')
                    v = v.replace( '_png','.png')
                    img_name = v
                    file_path = os.path.join(test_data_dir, img_name)
                    if os.path.exists(file_path)!=True:
                        print("File doesn't exist ")
                        img_name = ''
            if(img_name!=''):
                if(decision == "y"): 
                    image_files_test[0].append(file_path)
                if(decision == "n"): 
                    image_files_test[1].append(file_path)
 
                    
    num_each = [len(image_files[i]) for i in range(num_class)]
    num_each_val = [len(image_files_val[i]) for i in range(num_class)]
    num_each_test = [len(image_files_test[i]) for i in range(num_class)]
    image_files_list = []
    image_files_list_val = []
    image_files_list_test = []
    image_class = []
    image_class_val = []
    image_class_test = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_files_list_val.extend(image_files_val[i])
        image_files_list_test.extend(image_files_test[i])

        image_class.extend([i] * num_each[i])
        image_class_val.extend([i] * num_each_val[i])
        image_class_test.extend([i] * num_each_test[i])
    num_total = len(image_class) + len(image_class_val)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    
    train_x= image_files_list
    train_y  = image_class
    val_x = image_files_list_val
    val_y = image_class_val
    test_x = image_files_list_test
    test_y = image_class_test
    print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")
    
    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts train: {num_each}")
    print(f"Label counts val: {num_each_val}")
    
    plt.subplots(3, 3, figsize=(8, 8))
    for i, k in enumerate(np.random.randint(len(image_class), size=9)):
        im = PIL.Image.open(image_files_list[k])
        arr = np.array(im)
        plt.subplot(3, 3, i + 1)
        plt.xlabel(class_names[image_class[k]])
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
    
    return num_class, train_x,train_y, val_x, val_y, test_x, test_y, root_dir, class_names