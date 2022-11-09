## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE - Basic UNETR
# python train_basicUNETR.py /path_to_data/Data_3D/CT/Data_MMWHS /path_to_last_trained_model/trained_models/last_CT_challenge_data_BasicUnetr.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_Dice_Loss_challenge_data_BasicUNetr.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE - UNETR with Deep Supervision
# python train_DeepSuperOnly.py /path_to_data/Data_3D/CT/Data_MMWHS /path_to_last_trained_model/trained_models/last_CT_challenge_data_DeepSuperOnly.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_Dice_Loss_challenge_data_DeepSuperOnly.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div - UNETR with Self Distillation Original
# python train_SelfDistil_Original.py /path_to_data/Data_3D/CT/Data_MMWHS /path_to_last_trained_model/trained_models/last_CT_challenge_data_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_Dice_KL_Loss_challenge_data_SelfDist_Original.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div + Boundary - UNETR with Self Distillation with Distance Maps
python train_SelfDistil_DistMaps.py /path_to_data/Data_3D/CT/Data_MMWHS /path_to_last_trained_model/trained_models/last_CT_challenge_data_SelfDistil_DistMaps.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp --training_split 4 --device cuda:0 --show_verbose

