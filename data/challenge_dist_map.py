# pyright: reportPrivateImportUsage=false
import glob
import os
from monai.data import CacheDataset
import numpy as np
from typing import Union, Tuple

from .transforms import (
    AddChanneld,
    Compose,
    ConvertLabel,
    LoadImaged,
    AsChannelFirstd,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    CenterSpatialCropd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

def loadBoun(data_dir: str, img_size: Tuple[int, ...]=(96,96,96), scale_intensity_ranged: bool = True, train_split: int = 4, show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, int]:
    """
    Load dataset

    - Parameters:
        - data_dir: A `str` of data directory
        - img_size: An `int` of image size
        - show_verbose: A `bool` of flag to show loading progress
    - Returns: A `tuple` of training `DataLoader`, validation `DataLoader`, and the number of classes in `int`
    """
    # load images and labels
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    train_dist_maps = sorted(
        glob.glob(os.path.join(data_dir, "distmapsTr", "*.nii.gz")))
    
    data_dicts = [
        {"image": image_name, "label": label_name, "dist_map": dist_map_name}
        for image_name, label_name, dist_map_name in zip(train_images, train_labels, train_dist_maps)
    ]
    train_data_dicts, val_data_dicts = data_dicts[train_split:], data_dicts[:train_split]

    ##############################################################################################################

    # transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label", "dist_map"]),
        AddChanneld(keys=["image", "label"]),
        AsChannelFirstd(keys=("dist_map")), # same as data_dict['dist_map'].permute(3, 0, 1, 2)
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ), # already done while generating dist maps
        Orientationd(keys=["image", "label", "dist_map"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label", "dist_map"], roi_size=(128,128,128)),
    ]
    if scale_intensity_ranged:
        train_transforms += [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # for ct onlys
        ]
    train_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        SpatialPadd(keys=["image", "label", "dist_map"], spatial_size=img_size, mode='reflect'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label", "dist_map"],
            label_key="label",
            spatial_size=img_size,  # 16*n
            pos=1,
            neg=0, # neg=0 & pos=1 to always pick a foreground voxel as center for random crop
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label", "dist_map"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label", "dist_map"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore

    ################################################################################################

    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label", "dist_map"]),
        AddChanneld(keys=["image", "label"]),
        AsChannelFirstd(keys=("dist_map")), # same as data_dict['dist_map'].permute(3, 0, 1, 2)
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ), # already done while generating dist maps
        Orientationd(keys=["image", "label", "dist_map"], axcodes="RAS")
    ]
    
    if scale_intensity_ranged:
        val_transforms += [
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]
    val_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        ToTensord(keys=["image", "label", "dist_map"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore

    ##########################################################################

    # get datasets
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )
    val_ds = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        progress=show_verbose
    )

    num_classes = 8

    return train_ds, val_ds, num_classes

