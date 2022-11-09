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
    CropForegroundd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

def load(data_dir: str, img_size: Tuple[int, ...]=(96,96,96), scale_intensity_ranged: bool = True, train_split: int = 4, show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, int]:
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
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    train_data_dicts, val_data_dicts = data_dicts[train_split:], data_dicts[:train_split]

    #############################################################################################################
    ## Test out the different transforms for MMWHS challenge dataset
    '''loader = LoadImaged(keys=("image","label"))
    data_dict = loader(train_data_dicts[0])

    #print(len(val_data_dicts))
    #print(len(train_data_dicts))
    print(f"image shape: {data_dict['image'].shape}")
    print(f"label shape: {data_dict['label'].shape}")
    #print(f"Dist Map shape: {data_dict['dist_map'].shape}")  
    raise ValueError


    label = data_dict["label"] # np.ndarray: (512,512,177)
    print("Original GT Labels shape: ", np.shape(label))   
    print("Unique Labels: ", np.unique(label))
    print("Max value of Label: ", np.max(label)) 

    conv_label = ConvertLabel(keys='label')
    label = conv_label(data_dict["label"])
    print("Original GT Labels shape: ", np.shape(label))   
    print("Unique Labels: ", np.unique(label))
    print("Max value of Label: ", np.max(label))

    raise ValueError
    
    add_c = AddChanneld(keys=('image','label'))
    data_dict_ch_added = add_c(data_dict)

    print(f"image shape after channel added: {data_dict_ch_added['image'].shape}")
    print(f"label shape after channel added: {data_dict_ch_added['label'].shape}")
    # print(f"Dist Map shape with NO channel added: {data_dict_ch_added['dist_map'].shape}") 
    raise ValueError

    #data_dict_ch_added['dist_map'] = torch.from_numpy(data_dict_ch_added['dist_map']).permute(3, 0, 1, 2)
    #print(f"Dist Map shape after permute: {data_dict_ch_added['dist_map'].shape}") 

    to_tensor = ToTensord(keys=keys)
    converted_to_tensor = to_tensor(data_dict_ch_added)
    
    print(f"image shape after channel added: {converted_to_tensor['image'].shape}")
    print(f"label shape after channel added: {converted_to_tensor['label'].shape}")
    print(f"Dist Map shape with NO channel added: {converted_to_tensor['dist_map'].permute(3, 0, 1, 2).shape}")'''

    ##############################################################################################################

    # transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        #AsChannelFirstd(keys=("dist_map")), # same as data_dict['dist_map'].permute(3, 0, 1, 2)
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=(128,128,128)),
    ]
    if scale_intensity_ranged:
        train_transforms += [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # for ct onlys
        ]
    train_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode='reflect'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,  # 16*n
            pos=1,
            neg=0, # neg=0 & pos=1 to always pick a foreground voxel as center for random crop; neg=1 used by Junhao
            num_samples=1, #num_samples=1 used by Junhao
            image_key="image",
            image_threshold=0,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="label", select_fn=lambda x: x > 0),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore
    
    #########################################################################################
    '''## Debug
    tT = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode='constant'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,  # 16*n
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
    ]
    
    train_transforms = Compose(tT)

    data_dict = train_transforms(train_data_dicts)

    print(len(train_data_dicts[0]))
    # print(data_dict)
    print(f"image shape: {data_dict[0]['image'].shape}")
    print(f"label shape: {data_dict[0]['label'].shape}")
    print(f"dist map shape: {data_dict[0]['dist_map'].shape}")'''


    ################################################################################################

    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        #AsChannelFirstd(keys=("dist_map")), # same as data_dict['dist_map'].permute(3, 0, 1, 2)
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if scale_intensity_ranged:
        val_transforms += [
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]
    val_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        ToTensord(keys=["image", "label"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore

    ################################################################################################
    '''data_dict = val_transforms(val_data_dicts[0])

    print(len(val_data_dicts[0]))
    # print(data_dict)
    print(f"image shape: {data_dict['image'].shape}")
    print(f"label shape: {data_dict['label'].shape}")
    print(f"dist map shape: {data_dict['dist_map'].shape}")

    raise ValueError'''

    ################################################################################################

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

