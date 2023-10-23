import numpy as np
import os
#import nrrd

from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor


from mm_custom.registry import TRANSFORMS

import albumentations as A




@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    def __init__(self) -> None:
        super().__init__()


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:
        
        packet_results = dict()

        data_batch = dict()

        # inputs
        if 'img' in results:
            img = results['img']    
            img = to_tensor(img).contiguous()
            img = (img / 255. - 0.5) / 2.  # TODO  这个地方应该设成*2
            data_batch['inputs'] = img


        # target_dic
        if 'target_dic' in results:
            target_dic = results['target_dic']

            for k, v in target_dic.items():
                target_dic[k] = to_tensor(v)
            
            data_batch['target_dic'] = target_dic

        # image_level_target_dic
        if 'image_level_target_dic' in results:
            image_level_target_dic = results['image_level_target_dic']

            for k, v in image_level_target_dic.items():
                image_level_target_dic[k] = to_tensor(v)
            
            data_batch['image_level_target_dic'] = image_level_target_dic
        
        

        # weight_dic
        if 'weight_dic' in results:
            data_batch['weight_dic'] = results['weight_dic']



        if 'mask_dic' in results:
            mask_dic = results['mask_dic']

            for k, v in mask_dic.items():
                mask_dic[k] = to_tensor(v)
            
            data_batch['mask_dic'] = mask_dic


        # meta info
        data_batch['patient_id'] = results['patient_id']
        data_batch['series_id'] = results['series_id']


        packet_results['data_batch'] = data_batch
        
        return packet_results
    






    
@TRANSFORMS.register_module()
class PackInputsSub(BaseTransform):
    def __init__(self) -> None:
        super().__init__()


    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:
        
        packet_results = dict()

        data_batch = dict()

        # inputs
        if 'img' in results:
            img = results['img']    
            img = to_tensor(img).contiguous()
            img = (img / 255. - 0.5) / 2.
            data_batch['inputs'] = img


        # meta info
        #data_batch['aortic_hu_type'] = results['aortic_hu_type']
        #data_batch['aortic_hu'] = results['aortic_hu']

        data_batch['patient_id'] = results['patient_id']
        data_batch['series_id'] = results['series_id']


        packet_results['data_batch'] = data_batch
        
        return packet_results