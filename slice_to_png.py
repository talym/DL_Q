# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:57:00 2022

@author: marko
"""
import os
import nibabel as nib
import numpy as np
import imageio
import glob


ROOT_FOLDER = os.path.join("C:\\Users\\marko\\first rotation\\project\\ResultsSeg_vTry\\")

def volume_to_slices_png(root_folder, volume_name):
    volume_path = os.path.join(ROOT_FOLDER, volume_name)
    volume_image_path = os.path.join(volume_path, "data.nii.gz")
    slices_folder = os.path.join(volume_path,"slices")
    try:
        os.mkdir(slices_folder)
    except:
        print("mkdir failure")
    
    img_load = nib.load(volume_image_path)
    img_vol = np.array(img_load.dataobj)
    slices_axis = np.argmin(img_vol.shape)
    for i in range(img_vol.shape[slices_axis]):
        print('slice num : {}'.format(i))
        img = img_vol.take(indices=i, axis=slices_axis)
        img = 255 * (img - np.min(img[:])) / (np.max(img[:]) - np.min(img[:]))  # image normalization
        imageio.imwrite(os.path.join(slices_folder,'{}.png'.format(i)), img)
    

if __name__ == "__main__":
    #run_type = "Single"
    run_type = "Batch"    
    
    if(run_type == "Single"):
        volume_to_slices_png(ROOT_FOLDER, "Pat549_Se07_Res0.7813_0.7813_Spac5.0.nii.gz")
    else:
        list_of_volumes = glob.glob(os.path.join(ROOT_FOLDER,"*.nii.gz"))
        for vol_path in list_of_volumes:
            head_tail = os.path.split(vol_path)
            chosen_vol_name = head_tail[len(head_tail)-1] 
            volume_to_slices_png(ROOT_FOLDER, chosen_vol_name)

    
