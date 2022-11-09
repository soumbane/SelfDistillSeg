"""
Main training script to train a UNETR with Self Distillation with Distance Maps on MMWHS Challenge dataset for CT ONLY
"""
# pyright: reportPrivateImportUsage=false
from csv import writer
import logging, os, torch
from pyrsistent import b
from typing import Union
from torch.nn import MSELoss, KLDivLoss

from monai.data.dataloader import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss

from torch.backends import cudnn

import data
from configs import TrainingConfig
from torchmanager_monai import Manager, metrics
from networks import SelfDistillUNETRWithDictOutput as SelfDistilUNETR
from torchmanager import callbacks, losses

from loss_functions import Self_Distillation_Loss_Dice, Self_Distillation_Loss_Boundary, BoundaryLoss, Self_Distillation_Loss_KL, Self_Distillation_Loss_L2


if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_ckpt_dir = os.path.join(config.experiment_dir, "best.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")
    
    # load dataset - Load MMWHS Challenge Data
    in_channels = 1
    training_dataset, validation_dataset, num_classes = data.load_challenge_boun(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose)
        
    training_dataset = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate)

    ##########################################################################################################
    ## Initialize the model
    model = SelfDistilUNETR(in_channels, num_classes, img_size=config.img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
    
    ##########################################################################################################
    
    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) # lr used by challenge winner

    # initialize learning rate scheduler
    lr_step = max(int(config.epochs / 6), 1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5)

    # Initilialize Loss Functions
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

    # Hyper-parameters for Self Distillation
    alpha_KL: float = 0.05  # weight of KL Div Loss Term; (1-alpha_KL) for DiceCE Loss Term - Exp ID 1
    # lambda_feat: float = 1e-4  # weight of L2 Loss Term between feature maps
    temperature: int = 2 # divided by temperature (T) to smooth logits before softmax (required for KL Div)
    beta_distMap: float = 1e-6 # weight of Boundary Loss Term

    ## For Multiple Losses
    # Deep Supervision or Self Distillation from GT Labels to decoders  
    # For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main,out_dec2,out_dec3,out_dec4) [input]
    loss_dice = Self_Distillation_Loss_Dice([
        losses.Loss(DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)), #out_main and GT labels
        losses.Loss(DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)), #out_dec4 and GT labels
        losses.Loss(DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)), #out_dec3 and GT labels
        losses.Loss(DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)), #out_dec2 and GT labels
        ], target="out", weight=(1.0-alpha_KL))

    # Deep Supervision or Self Distillation from GT Dist_Maps to decoders  
    # For DiceCE Loss between GT Dist_Maps(target="dist_map") [target] and softmax(out_main,out_dec2,out_dec3,out_dec4) [input]
    loss_boundary = Self_Distillation_Loss_Boundary([
        losses.Loss(BoundaryLoss(include_background=False)), #out_main and GT Dist_Maps
        losses.Loss(BoundaryLoss(include_background=False)), #out_dec4 and GT Dist_Maps
        losses.Loss(BoundaryLoss(include_background=False)), #out_dec3 and GT Dist_Maps
        losses.Loss(BoundaryLoss(include_background=False)), #out_dec2 and GT Dist_Maps
        ], target="dist_map", weight=beta_distMap)

    # Self Distillation from deepest encoder/decoder (out_enc4/out_dec2) to shallower encoders/decoders (out_enc2/out_dec3,out_enc3/out_dec4)  
    # For KL Div between softmax(out_dec2) [target] and log_softmax((out_dec3,out_dec4)) [input]
    loss_KL = Self_Distillation_Loss_KL([
        losses.Loss(KLDivLoss(reduction="batchmean", log_target=False)), #out_dec3/out_enc2 and out_dec2/out_enc4
        losses.Loss(KLDivLoss(reduction="batchmean", log_target=False)), #out_dec4/out_enc3 and out_dec2/out_enc4
    ], weight=alpha_KL, include_background=False, T=temperature) # pass the entire dict NOT just "out"

    # For L2 loss between feature maps/hints dec3_f [target] and feature maps (dec1_f,dec2_f) [input]
    '''loss_L2 = Self_Distillation_Loss_L2([
        losses.Loss(MSELoss(reduction="mean")), #dec1_f and dec3_f
        losses.Loss(MSELoss(reduction="mean")), #dec2_f and dec3_f
    ], weight=lambda_feat) # pass the entire dict NOT just "out"'''

    loss_fn = {
        "dice": loss_dice,
        "boundary": loss_boundary,
        "KL": loss_KL
    }
   
    # Initialize Metrics
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

    hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")

    metric_fns: dict[str, metrics.Metric] = {
        "val_dice": dice_fn,
        "val_hd": hd_fn
        } 

    post_labels = data.transforms.AsDiscrete(to_onehot=num_classes)
    post_predicts = data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes)

    # compile manager
    manager = Manager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size) # type: ignore

    ## All callbacks defined below
    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)

    last_ckpt_callback = callbacks.LastCheckpoint(manager, last_ckpt_dir)
    besti_ckpt_callback = callbacks.BestCheckpoint("dice", manager, best_ckpt_dir)
    lr_scheduler_callback = callbacks.LrSchedueler(lr_scheduler, tf_board_writer=tensorboard_callback.writer)
    
    ##############################################################################################################
    # Increase Boundary Loss weight by this amount every epoch
    incr_beta_distMap = 1e-3

    def getw_dist_map(e):
        return (loss_fn["boundary"].weight + incr_beta_distMap)

    weights_callback_boundary = callbacks.LambdaDynamicWeight(getw_dist_map, loss_fn["boundary"], writer=tensorboard_callback.writer, name='boundary_weight')

    ##############################################################################################################

    # Final callbacks list
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback, lr_scheduler_callback, weights_callback_boundary]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test with last model
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

    # save and test with best model on validation dataset  
    manager = Manager.from_checkpoint("experiments/CT_Dice_KL_Boundary_Loss_challenge_data_SelfDist_DistMaps.exp/best.model") # for Self Distillation with Distance Maps

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

