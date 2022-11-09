from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers.utils import get_act_layer, get_norm_layer

from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He

DropPath = nn.Dropout3d

# ConvNeXt v2
# 1.Block中 depthwise kernel_size 为 5 2
# 2.LayerNorm 为 instanceNorm
# 3.Block中激活函数为 leakyrelu
# 4.Block中 drop_path_rate 为 0
# 5.downsampling patch_embed 为 2
# 6.depths = [3,3,3,3]
# 7.dims = [48,96,192,384]

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, norm_name = "instance"):
        super(Block, self).__init__()
        # 3D ConvNeXt depthwise conv 7,3
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        self.norm = get_norm_layer(name=norm_name, spatial_dims=3, channels=dim)

        self.pwconv1 = nn.Conv3d(dim, 4 * dim, 1)

        self.act = get_act_layer(name="LeakyReLU")

        self.pwconv2 = nn.Conv3d(4 * dim, dim, 1)

        # gamma ??? B,C,Y,H,W
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):  
    def __init__(self, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],patch_embed=4,norm_name="instance"
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1,patch_embed,patch_embed), stride=(1,patch_embed,patch_embed)),
            get_norm_layer(name=norm_name, spatial_dims=3, channels=dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    get_norm_layer(name=norm_name, spatial_dims=3, channels=dims[i]),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, norm_name=norm_name) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        for i_layer in range(4):
            layer = get_norm_layer(name=norm_name, spatial_dims=3, channels=dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x
