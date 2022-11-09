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

3. Generate Train, Validation and Test datasets from the generated X matrix.
* We divided the entire dataset in chronological order with 80% training, 10% validation and 10% testing.
* Run the file `generate_training_data.py` to generate the processed files `train.npz, val.npz, test.npz` from X matrix and save the processed files in `data/COVID_JHU/processed` or `data/COVID_NYT/processed`. Use the `confirmed` or `deaths` in the argument to generate infected and death cases processed files respectively.
```
# For JHU Daily Infected cases data
python generate_training_data.py --traffic_df_filename "data/COVID_JHU/covid19_confirmed_US_51_states_X_matrix_final.csv" 

# For NYT Daily Death cases data
python generate_training_data.py --traffic_df_filename "data/COVID_NYT/covid19_NYT_deaths_US_51_states_X_matrix_final.csv"
```

## Training

1. Define paths and hyper-parameters in configuration files.
* Refer to the files `config/COVID_JHU.conf` and `config/COVID_NYT.conf` for the data paths, hyper-parameters and model configurations used for training and testing. 
* The `sensors_distance` in the config files indicate the path to the adjacency matrix W.

2. Train the model
```
python train.py --epochs 100 --learning_rate 0.001 --expid 1 --print_every 20
```

## Testing

1. The pre-trained models could be found in `checkpoints/pretrained_models`
* Refer to the required folder `JHU or NYT`, `Infected or Deaths` for infected or death cases respectively and our model is in folder `STST`

2. Test the model
* An example for testing with `COVID_JHU` dataset's daily infected cases and `COVID_NYT` dataset's daily death cases with our model `STST` (name in code for STSGT model) is given here. The `... _best_model.pth` indicates the model with the lowest Mean Absolute Error (MAE) on the validation set. 
```
# For JHU Daily Infected cases data with our trained model
python test.py --checkpoint "checkpoints/pretrained_models/JHU_States_Infected/STST/exp_2_1654.67_best_model.pth"

# For NYT Daily Death cases data with our trained model
python test.py --checkpoint "checkpoints/pretrained_models/NYT_States_Deaths/STST/exp_1_19.06_best_model.pth"
```

## Notes
* Please choose the correct configuration file with the `DATASET` variable in both `train.py` and `test.py`.

