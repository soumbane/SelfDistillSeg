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
* The functions in the above files are called from the main training script. 

## Training

1. Train the four models either with their respective training script. An example of training with deep supervision + self-distillation with shape priors is:

```
python train_SelfDistil_DistMaps.py /path_to_data/Data_3D/CT/Data_MMWHS /path_to_last_trained_model/trained_models/last_CT_challenge_data_SelfDistil_DistMaps.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp --training_split 4 --device cuda:0 --show_verbose
```

2. Another way to train is by running the shell script `train.sh` with the required models:

```
. train.sh
```

## Testing

1. Test with the model with the highest validation dice score during training. This model is saved as `best.model` in the `experiments/` folder.
* For example: the best model while training with shape priors (distance maps) `train_SelfDistil_DistMaps.py` is saved in `experiments/CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp/best.model`

```
python test.py 
```

* The path to the `best.model` checkpoint is given in the `test.py` file. An example of testing with the best model for `SelfDistil_DistMaps` is:
```
manager = Manager.from_checkpoint("experiments/CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp/best.model")
```

## Notes
* Use the `def load(...)` function from `data/challenge.py` to load the training and validation images and labels for `train_basicUNETR.py`, `train_DeepSuperOnly.py`, `train_SelfDistil_Original.py` and `test.py` as follows:

```
# load dataset - Load MMWHS Challenge Data
training_dataset, validation_dataset, num_classes = data.load_challenge(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose)        
```

* Use the `def loadBoun(...)` function from `data/challenge_dist_map.py` to load the training and validation images, labels and shape priors (distance maps) for `train_SelfDistil_DistMaps.py` and `test.py` as follows:

```
# load dataset - Load MMWHS Challenge Data
training_dataset, validation_dataset, num_classes = data.load_challenge_boun(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose)
```
