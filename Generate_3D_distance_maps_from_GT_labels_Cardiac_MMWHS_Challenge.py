## Generate 3D Distance Maps for Boundary Loss
## Generate the distance maps from GT labels of cardiac MMWHS challenge dataset (Train & val)

import monai

import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
import time

import os
import shutil
import tempfile
import glob

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    CenterSpatialCropd,
    LoadImage,
    LoadImaged,
    SaveImage,
    SaveImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

import torch
import torch.nn.functional as F
from typing import Tuple


root_dir = os.getcwd()
print(f"root dir is: {root_dir}")

## Define Data Paths

## For CT Datasets
data_dir = os.path.join(root_dir, "Data_3D/CT/Data_MMWHS")

## MMWHS CT Dataset
train_imgs = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [
    {"image": img_name, "label": label_name}
    for img_name, label_name in zip(train_imgs, train_labels)
]

## MapTransform for transforming initial label values to discrete consecutive integers
# convert the label pixel value 
class ConvertLabel(MapTransform):
    
    #[0. 205. 420. 500. 550. 600. 820. 850.]

    def __call__(self, data):                
        d = dict(data)                             
        for key in self.keys:            
            d[key][d[key]==205] = 1
            d[key][d[key]==420] = 2
            d[key][d[key]==500] = 3
            d[key][d[key]==550] = 4
            d[key][d[key]==600] = 5
            d[key][d[key]==820] = 6
            d[key][d[key]==850] = 7
            d[key][d[key]==421] = 0 ## have a pixel with value 421, which should be a mistake
        
        return d


## Function to convert classes to one-hot labels
def class_to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:    
    # convert to tensor    
    labels_tensor = torch.from_numpy(labels) # Label shape: (512,512,z): z varies by patient
    # one-hot encoding
    label_onehot_encoded = F.one_hot(labels_tensor.to(torch.int64), num_classes=num_classes) # (512,512,z,num_classes)
    label_onehot_encoded = label_onehot_encoded.permute(3,0,1,2) # (num_classes,512,512,z)
    
    return label_onehot_encoded.numpy()


## Function to convert one-hot labels to distance maps
def dist_map_transform(one_hot_labels: np.ndarray, num_classes: int, 
                       resolution: Tuple[float, float, float]=None, dtype=None) -> np.ndarray:
    
    # same shape as one-hot enc GT label (num_classes,512,512,z)
    dist_map = np.zeros_like(one_hot_labels, dtype=dtype)    
                
    for nc in range(1, num_classes):  # Leave background (negative classes) blank      
        posmask = one_hot_labels[nc].astype(np.bool)        
        
        # Leave background (negative classes) blank
        # We DO NOT use the background for training
        if posmask.any():            
            negmask = ~posmask            
                    
            dist_map[nc] = (eucl_distance(negmask, sampling=resolution) * negmask) - ((eucl_distance(posmask, sampling=resolution) - 1) * posmask)                            
                   
    return dist_map  # shape: (num_classes,512,512,z): z varies by patient


## Load Train Images, Ground Truth (GT) Labels, convert GT Labels to Distance Maps and save the distance maps

# The following is for Train Labels ONLY (includes Validation also) 
# MMWHS does not have test images - just train and validation
loader = LoadImaged(keys=("image", "label"))
add_ch = AddChanneld(keys=["image", "label"])
spc = Spacingd(keys=("image", "label"),
               pixdim=(1.5,1.5,2.0),
               mode=("bilinear", "nearest"),              
              ) # convert (1,512,512,z) -> (1,167,167,z')
converted_label = ConvertLabel(keys='label')
orient = Orientationd(keys=["image", "label"], axcodes="RAS")

distmap_transforms = Compose([loader,add_ch,converted_label,spc,orient])

start_time = time.time()

for i in range(len(data_dicts)):
    
    data_dict = distmap_transforms(data_dicts[i])  
                                
    ## Save Distance Maps from GT Labels
    label = data_dict["label"] # np.ndarray: (512,512,z) where z varies by patient
    img = data_dict["image"] # np.ndarray: (512,512,z) where z varies by patient
                        
    # one-hot encoded GT labels
    # num_classes = 8 (7 sub-structures + Bkg) or -1 to infer num_classes from highest class no.(Ex: 7)
    
    # convert (1,167,167,z') -> (167,167,z') for class_to_onehot func
    label = np.squeeze(label,axis=0) 
    img = np.squeeze(img,axis=0)
    
    label_one_hot = class_to_onehot(label, num_classes=-1) # np.ndarray: (num_classes,512,512,z)
    #print("One-hot encoded labels shape: ", np.shape(label_one_hot))  
                
    # Apply Distance Transform to one-hot GT Labels
    num_classes = int(np.max(label)+1)
    
    # pixel spacing along x,y,z axis
    rx = 1.0
    ry = 1.0
    rz = 1.0
    
    dist_map = dist_map_transform(label_one_hot, num_classes=num_classes, resolution=(rx,ry,rz), dtype=np.float32)
    #print("Distance Map shape: ", np.shape(dist_map)) # np.ndarray: (num_classes,512,512,z)

    ##############################################################################################################
    
    ## Save Distance Maps
    out_dir = os.path.join(data_dir, "distmapsTr") 
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    saver = SaveImage(output_dir=out_dir, output_postfix='dist_map', output_ext='.nii.gz', 
                      output_dtype=np.float32, resample=False, scale=None, dtype=None, squeeze_end_dims=True, 
                      data_root_dir='', separate_folder=False, print_log=False
                     )
        
    if i < 9: saver(dist_map, meta_data={'filename_or_obj':'ct_train_100'+str(i+1)})
    else: saver(dist_map, meta_data={'filename_or_obj':'ct_train_10'+str(i+1)})


end_time = time.time()
print("Time taken: ", (end_time-start_time))



