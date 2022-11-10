# Self-Distillation of U-shaped Networks for 3D Cardiac Image Segmentation with Shape Priors

This is the official PyTorch implementation of our paper "SELF-DISTILLATION OF U-SHAPED NETWORKS FOR 3D CARDIAC IMAGE SEGMENTATION WITH SHAPE PRIORS" that is under review for IEEE ISBI 2023.

## Requirements
* python >= 3.9
* pytorch >= 1.12
* [torchmanager](https://github.com/kisonho/torchmanager) >= 1.0.6
* [monai](https://monai.io/) >= 0.9.0

## Get Started
The following steps are required to replicate our work:

1. Download dataset.
* MMWHS Dataset - Download [MMWHS data](https://zmiclab.github.io/zxh/0/mmwhs/) and save the CT images `ct_train_..._image.nii.gz` in `Data_3D/CT/Data_MMWHS/imagesTr` directory and the CT ground-truth (GT) segmentation labels `ct_train_..._label.nii.gz` in `Data_3D/CT/Data_MMWHS/labelsTr` directory. 

2. Generate shape priors (level-set based distance maps) from downloaded dataset.
* Run the file `Generate_3D_distance_maps_from_GT_labels_Cardiac_MMWHS_Challenge.py` to generate the shape priors (distance maps) `ct_train_..._dist_map.nii.gz` and save the CT ground-truth (GT) segmentation shape priors (distance maps) in `Data_3D/CT/Data_MMWHS/distmapsTr` directory.

3. Generate Train and Validation datasets.
* We divided the entire dataset (20 patients) into 80% training (16 patients) and 20% validation (4 patients).
* The file `challenge.py` generates the training and validation datasets and performs the necessary transforms for training and validation images and labels.
* The file `challenge_dist_map.py` generates the training and validation datasets and performs the necessary transforms for training and validation images, labels and shape priors (distance maps).

## Training

1. Train the model
```
python train.py --epochs 100 --learning_rate 0.001 --expid 1 --print_every 20
```

## Testing

1. Test the model

```
python test.py 
```

## Notes
* 

