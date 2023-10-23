#from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mmcv.cnn import ConvModule

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint

from mm_custom.registry import MODELS



@MODELS.register_module()
class UnetPlusPlus(BaseModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        interpolate_mode='nearest',
        attention_type=None,
        with_cp=False,
    ):
        super().__init__()

        self.with_cp = with_cp

        # reverse channels to start from head of encoder
        in_channels = in_channels[::-1]

        # computing blocks input and output channels
        head_channels = in_channels[0]
        self.in_channels = [head_channels] + list(out_channels[:-1])
        self.skip_channels = list(in_channels[1:]) + [0]
        self.out_channels = out_channels

        # combine decoder keyword arguments
        kwargs = dict(norm_cfg=norm_cfg, act_cfg=act_cfg, attention_type=attention_type, interpolate_mode=interpolate_mode, with_cp=with_cp)


        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)

        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )

        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1


    def forward(self, features):

        #features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )

        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])

        res_features = []

        for depth in range(self.depth, -1, -1):
            res_features.append(dense_x[f'x_{0}_{depth}'])

        return res_features

        #return dense_x[f"x_{0}_{self.depth}"]
    






class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        interpolate_mode='nearest',
        attention_type=None,
        with_cp=False,
    ):
        super().__init__()

        self.with_cp = with_cp

        self.interpolate_mode = interpolate_mode

    


        self.conv1 = ConvModule(
            in_channels=in_channels+skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        


        self.conv2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        if not attention_type is None:
            self.attention1 = SCSEModule(in_channels=in_channels+skip_channels)
            self.attention2 = SCSEModule(in_channels=out_channels)
        else:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode=self.interpolate_mode) # nearest/bilinear
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        if self.with_cp:
            x = checkpoint(self.conv1, x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        x = self.attention2(x)
        return x




class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)



