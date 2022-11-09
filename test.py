# pyright: reportPrivateImportUsage=false
import os
import torch
import numpy as np
from torchmanager_monai import Manager, metrics

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
import data

from monai.transforms import LoadImage, SaveImage


data_dir = '/home/neil/Lab_work/Cardiac_Image_Segmentation/Data_3D/CT/Data_MMWHS'
img_size = (96,96,96)

## Basic UNETR
# manager = Manager.from_checkpoint("experiments/CT_Dice_Loss_challenge_data_BasicUNetr.exp/best.model")

## UNETR with Deep Supervision ONLY
# manager = Manager.from_checkpoint("experiments/CT_Dice_Loss_challenge_data_DeepSuperOnly.exp/best.model")

## UNETR with Self Distillation Original - Dice + KL Div Loss
# manager = Manager.from_checkpoint("experiments/CT_Dice_KL_Loss_challenge_data_SelfDist_Original.exp/best.model")

## UNETR with Self Distillation with Distance Maps - Dice + KL Div + Boundary Loss
manager = Manager.from_checkpoint("experiments/CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp/best.model")

# Initialize Metrics
dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")

metric_fns: dict[str, metrics.Metric] = {
    "val_dice": dice_fn,
    "val_hd": hd_fn
    } 

manager.loss_fn = None
manager.metric_fns = metric_fns

if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
else: model = manager.model

manager.model = model
print(f'The best Dice score occurs at {manager.current_epoch + 1} epoch number')

# load validation dataset - Load MMWHS Challenge validation Data
## validation dataset with (image,label) for regional losses
# _, validation_dataset, num_classes = data.load_challenge(data_dir, img_size=img_size, train_split=4, show_verbose=True)

## validation dataset with (image,label,dist_maps) for Boundary Loss
_, validation_dataset, num_classes = data.load_challenge_boun(data_dir, img_size=img_size, train_split=4, show_verbose=True)
        
validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate, num_workers=12, pin_memory=True)
print("Validation Data for Self Distillation Loaded ...")

summary = manager.test(validation_dataset, device=torch.device("cuda:0"), use_multi_gpus=False, show_verbose=True)
print(summary)

## Generate Model Predictions
patient_id = 2 # Select patient for whom to generate predictions

preds = manager.predict(validation_dataset, device=torch.device("cuda:0"), use_multi_gpus=False, show_verbose=True)
# print(preds[patient_id].shape)
preds_1 = preds[patient_id].squeeze(0)
# print(preds_1.shape)
preds_f = torch.argmax(preds_1, dim=0)
# print(preds_f.shape)

## Save Model Predictions
out_dir = os.path.join(data_dir, "Predicted_Labels") 

## Define your case
# case = 'basic_UNETR'
# case = 'deepSuper_UNETR'
# case = 'selfDistil_UNETR'
case = 'selfDistilDistMap_UNETR'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
saver = SaveImage(
    output_dir=out_dir, output_postfix='pred_label_'+case, output_ext='.nii.gz', 
    output_dtype=np.float32, resample=False, scale=None, dtype=None, squeeze_end_dims=True, 
    data_root_dir='', separate_folder=False, print_log=False, channel_dim=None
    )
    
saver(preds_f, meta_data={'filename_or_obj':'CT'})

load_path = os.path.join(out_dir,'CT_pred_label_'+case+'.nii.gz')
loader = LoadImage()

img = loader(load_path)
print(f"Loaded Image Shape: {img[0].shape}")

# Print the Dice/HD of all classes for all samples
print(f"Entire Val Dice Matrix (no of samples x classes): {manager.metric_fns['val_dice'].results.squeeze(1)}")
print(f"Entire Val HD Matrix (no of samples x classes): {manager.metric_fns['val_hd'].results.squeeze(1)}")

# Mean of Dice/HD for all samples in validation set (all classes)
print(f"Mean Val Dice (for all classes of all samples): {manager.metric_fns['val_dice'].results.squeeze(1).mean()}")
print(f"Mean Val HD (for all classes of all samples): {manager.metric_fns['val_hd'].results.squeeze(1).mean()}")

# Std Dev of Dice/HD for all samples in validation set (all classes)
print(f"Std of Val Dice (for all classes of all samples): {manager.metric_fns['val_dice'].results.squeeze(1).std()}")
print(f"Std of Val HD (for all classes of all samples): {manager.metric_fns['val_hd'].results.squeeze(1).std()}")

num_foreground_classes = num_classes - 1
print(f"No. of Foreground classes: {num_foreground_classes}")

for class_id in range(num_foreground_classes):
    # Mean of Dice/HD for class_id of all samples in validation set (individual classes)
    print(f"Mean Val Dice of foreground class {class_id} across all val samples: {manager.metric_fns['val_dice'].results.squeeze(1).mean(0)[class_id]}")
    print(f"Mean Val HD of foreground class {class_id} across all val samples:: {manager.metric_fns['val_hd'].results.squeeze(1).mean(0)[class_id]}")

    # Std of Dice/HD for class_id of all samples in validation set (individual classes)
    print(f"Std of Val Dice of foreground class {class_id} across all val samples: {manager.metric_fns['val_dice'].results.squeeze(1).std(0)[class_id]}")
    print(f"Std of Val HD of foreground class {class_id} across all val samples: {manager.metric_fns['val_hd'].results.squeeze(1).std(0)[class_id]}")

