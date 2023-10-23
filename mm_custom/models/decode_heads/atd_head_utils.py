from typing import Sequence


#import monai
#from monai.networks.nets.unet import UNet

from mmengine.model import BaseModule, BaseModel
from mmengine.structures import BaseDataElement


from mm_custom.registry import MODELS
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor
import math



@MODELS.register_module()
class ClipSinePositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset


    def forward(self, feat: Tensor, mask: Tensor = None) -> Tensor:

        B, T, C, H, W = feat.shape

        if mask is None:
            mask = torch.zeros((B, T, H, W), device=feat.device, dtype=torch.bool)

        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)  # not_mask【bath,t，h,w】1代表时间列的索引，cumsum累加计算，得到位置id
        y_embed = not_mask.cumsum(2, dtype=torch.float32)  # h
        x_embed = not_mask.cumsum(3, dtype=torch.float32)  # w

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=feat.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        dim_t_z = torch.arange((self.num_feats * 2), dtype=torch.float32, device=feat.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t  # [b,t,h,w]->[b,t,h,w,d] xy编码的d长度是位置编码向量长度的一半
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z # z用编码向量长度，然后和xy编码相加
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w

        return pos


    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str