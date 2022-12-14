# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from monai.networks.blocks import UpSample
from monai.utils import UpsampleMode

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp

class SelfDistilUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        Examples::
            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')
             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)
            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        ############
        # Encoders #
        ############
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        ############
        # Decoders #
        ############

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        #######################################################
        # ONLY Deconv Block of Decoders for Self-Distillation #
        #######################################################
        upsample_kernel_size = 2
        upsample_stride = upsample_kernel_size

        self.transp_conv_dec5 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.transp_conv_dec4 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.transp_conv_dec3 = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        #########################################
        # Upsample blocks for Self Distillation #
        #########################################
        
        self.deep_2 = DeepUp(
        spatial_dims = 3,
        in_channels = feature_size * 8,
        out_channels = out_channels,
        scale_factor = 8
        )

        self.deep_3 = DeepUp(
        spatial_dims = 3,
        in_channels = feature_size * 4,
        out_channels = out_channels,
        scale_factor = 4
        )

        self.deep_4 = DeepUp(
        spatial_dims = 3,
        in_channels = feature_size * 2,
        out_channels = out_channels,
        scale_factor = 2
        )

        #############
        # Out block #
        #############

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x    

    def forward(self, x_in):
        # Main ViT Transformer
        x, hidden_states_out = self.vit(x_in)

        #################################################
        # Encoders and Upsamplers for Self Distillation #
        #################################################

        # For Dice CE , KL Div and L2 loss between labels and each upsampled encoder output #
        
        # Encoder 1
        enc1 = self.encoder1(x_in)
        # print(f"Encoder 1 shape before upsampling: {enc1.shape}")  
        
        x2 = hidden_states_out[3]
        # Encoder 2
        enc2 = self.encoder2(self.proj_feat(x2))
        # print(f"Encoder 2 shape before upsampling: {enc2.shape}")
        
        # Upsample Encoder 2
        out_enc2 = self.deep_4(enc2) 
        # print(f"Encoder 2 shape after upsampling: {out_enc2.shape}")

        x3 = hidden_states_out[6]
        # Encoder 3
        enc3 = self.encoder3(self.proj_feat(x3))
        # print(f"Encoder 3 shape before upsampling: {enc3.shape}")       
        
        # Upsample Encoder 3
        out_enc3 = self.deep_3(enc3) 
        # print(f"Encoder 3 shape after upsampling: {out_enc3.shape}")

        x4 = hidden_states_out[9]
        # Encoder 4
        enc4 = self.encoder4(self.proj_feat(x4))
        # print(f"Encoder 4 shape before upsampling: {enc4.shape}")       
        
        # Upsample Encoder 4
        out_enc4 = self.deep_2(enc4) 
        # print(f"Encoder 4 shape after upsampling: {out_enc4.shape}")

        #######################################################################################################
        #######################################################################################################
        
        ################################################
        # Decoders and Upsamplers for Deep Supervision #
        ################################################

        # Decoder 3
        dec4 = self.proj_feat(x)
    
        dec3 = self.decoder5(dec4, enc4)
        # print(f"Decoder 3 output shape: {dec3.shape}")

        # Upsample decoder 3
        out_dec4 = self.transp_conv_dec5(dec4)
        out_dec4 = self.deep_2(out_dec4)
        # print(f"Decoder 3 upsampled shape: {out_dec4.shape}")

        # Decoder 2
        dec2 = self.decoder4(dec3, enc3)
        # print(f"Decoder 2 output shape: {dec2.shape}")

        # Upsample decoder 2
        out_dec3 = self.transp_conv_dec4(dec3)
        out_dec3 = self.deep_3(out_dec3)
        # print(f"Decoder 2 upsampled shape: {out_dec3.shape}")

        # Decoder 1
        dec1 = self.decoder3(dec2, enc2)
        # print(f"Decoder 1 output shape: {dec1.shape}")

        # Upsample decoder 1
        out_dec2 = self.transp_conv_dec3(dec2)
        out_dec2 = self.deep_4(out_dec2) 
        # print(f"Decoder 1 upsampled shape: {out_dec2.shape}")
                              
        # Prepare output layers        
        out = self.decoder2(dec1, enc1)
        # print(f"Main model output shape before 1x1x1 conv: {out.shape}")

        out_main = self.out(out) # Upper classifier
        # print(f"Main model output shape: {out_main.shape}")
                      
        # For traditional Self Distillation (Self Distil)
        if self.training:
            ## For Deep Supervision, Self Distillation Original and Self Distillation with GT Distance Maps
            out = (out_main, out_dec4, out_dec3, out_dec2, out_enc2, out_enc3, out_enc4)  # on decoder and encoder side
            
            ## For Basic UNETR ONLY
            # out = out_main  
        else:
            out = out_main

        return out


if __name__ == '__main__':
    unetr_with_self_distil = SelfDistilUNETR(
        in_channels = 1,
        out_channels = 8,
        img_size = (96,96,96)
        )

    x1 = torch.rand((1, 1, 96, 96, 96)) # (B,num_ch,x,y,z)
    print("Self Distil UNetr input shape: ", x1.shape)

    x3 = unetr_with_self_distil(x1)
    print("Self Distil UNetr output shape: ", x3[0].shape)

