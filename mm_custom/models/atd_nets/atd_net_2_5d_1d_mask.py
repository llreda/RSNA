from typing import Sequence
#import monai
#from monai.networks.nets.unet import UNet

from mmengine.model import BaseModule, BaseModel
from mmengine.structures import BaseDataElement

from mm_custom.registry import MODELS

import torch
import torch.nn.functional as F
import torch.nn as nn


@MODELS.register_module()
class ATDNetMask(BaseModel):
    def __init__(self, 
                 backbone: dict,
                 neck: dict,
                 decode_head: dict,
                 metainfo=None,
                 init_cfg=None):
        super().__init__(init_cfg)


        self.backbone = MODELS.build(backbone)
        self.neck_extra = MODELS.build(neck)
        self.neck_masks = MODELS.build(neck)
        self.decode_head = MODELS.build(decode_head)

        self.metainfo = metainfo



    def forward(self, data_batch, mode):

        # tensor 
        batch_inputs = data_batch['inputs'] # [B, T, H, W, C]

        B, T, H, W, C = batch_inputs.shape

        inputs = batch_inputs

        inputs = inputs.permute(0, 1, 4, 2, 3).flatten(0, 1) # [B*T, C, H, W]

        # backbone
        outputs = self.backbone(inputs)

        # neck
        features_masks = self.neck_masks(outputs) 
        features_extra = self.neck_extra(outputs)

        features = [features_masks, features_extra]


        if mode == 'loss':
            decoder_inputs = {
                'inputs': features,
                'target_dic': data_batch['target_dic'],
                'weight_dic': data_batch['weight_dic'],
                'mask_dic': data_batch['mask_dic'],
                'image_level_target_dic': data_batch['image_level_target_dic'],
            }

            loss_dict = self.decode_head(decoder_inputs, mode)
            return loss_dict
        else:

            decoder_inputs = {
                'inputs': features,
            }            

            data_samples = self.decode_head(decoder_inputs, mode)
            return data_samples
    



