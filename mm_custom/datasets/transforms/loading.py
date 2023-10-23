import numpy as np
import os
import cv2

from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor


from mm_custom.registry import TRANSFORMS

import pydicom
import numpy as np

import torch
import torch.nn.functional as F


from .util import *


@TRANSFORMS.register_module()
class LoadImageSub(BaseTransform):
    def __init__(self, in_channel, scale) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.scale = scale
        

    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:

        root, patient_id, series_id = results['root'], results['patient_id'], results['series_id']
        
        series_folder = os.path.join(root, 'test_images', str(patient_id), str(series_id))

        file_names = os.listdir(series_folder)


        if patient_id == '3124' and series_id == '5842':
            file_names = [file_name for file_name in file_names if file_name.split('.')[0] != '514']


        file_names = sorted(file_names, key=lambda x : int(x.split('.')[0]))

        file_names = np.array(file_names)
        extract_indexs = get_extract_indexes(len(file_names), self.scale[0])
        file_names = file_names[extract_indexs]

        imgs = []
        for file_name in file_names:
            file_path = os.path.join(series_folder, file_name)
            img = dicom_to_image(pydicom.dcmread(file_path))
            imgs.append(img)
    
        imgs = np.stack(imgs, axis=0) # shape: [d, h, w]

        imgs = torch.tensor(imgs).unsqueeze(0).unsqueeze(0).float()
        imgs = F.interpolate(imgs, size=self.scale, mode='trilinear')
        imgs = imgs.squeeze(0).squeeze(0).numpy().astype(np.uint8) # shape [D, H, W]
        

        D, H, W = imgs.shape

        C = self.in_channel
        T = D // C
        assert C * T == D
        
        imgs = imgs.reshape(T, C, H, W).transpose(0, 2, 3, 1) # [T, H, W, C]

        results['img'] = imgs    # shape [T, H, W, C]

        return results





@TRANSFORMS.register_module()
class LoadImage(BaseTransform):
    def __init__(self, clip_num, in_channel) -> None:
        super().__init__()
        self.clip_num = clip_num
        self.in_channel = in_channel


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:
        
        #patient_id, series_id = results['patient_id'], results['series_id']

        img_path_list = results['img_path_list']
        imgs = []

        if img_path_list[0].endswith('.png'):
            for file_name in img_path_list:
                img = cv2.imread(file_name)[:,:,0].astype(np.uint8)
                imgs.append(img)
        
        elif img_path_list[0].endswith('.dcm'):
        
            for file_name in img_path_list:
                img = dicom_to_image(pydicom.dcmread(file_name))
                imgs.append(img)

        else:
            Exception
        
        imgs = np.stack(imgs, axis=0) # [T*C, H, W]
        

        # TC, H, W = imgs.shape
        # assert self.clip_num * self.in_channel == TC
    
        # imgs = imgs.reshape(self.clip_num, self.in_channel, H, W)
        # imgs = imgs.transpose(0, 2, 3, 1) # [T, C, H, W] -> [T, H, W, C]
            

        results['img'] = imgs    # shape [T, H, W, C]

        return results



@TRANSFORMS.register_module()
class LoadMask(BaseTransform):
    def __init__(self, clip_num, in_channel) -> None:
        super().__init__()
        self.clip_num = clip_num
        self.in_channel = in_channel


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:


        mask_path_list_dict = results['mask_path_list_dict']

        mask_dic = {}

        for mask_name, mask_path_list in mask_path_list_dict.items():
            
            masks = []

            for mask_path in mask_path_list:
                mask = cv2.imread(mask_path)[:, :, 0]
                masks.append(mask)

            masks = np.stack(masks, axis=0) # [TC, H, W]

            TC, H, W = masks.shape
            assert self.clip_num * self.in_channel == TC

            mask_dic[mask_name] = masks.astype(np.uint8)

        results['mask_dic'] = mask_dic

        return results

