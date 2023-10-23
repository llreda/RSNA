import numpy as np
import os
import pandas as pd
import math

from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform
from mm_custom.registry import DATASETS, TRANSFORMS

from mmengine.dataset import BaseDataset
from .transforms.util import get_extract_indexes


@DATASETS.register_module()
class DatasetSub(BaseDataset):
    def __init__(self,
                 clip_num,
                 in_channel,
                 root,
                 **kwargs
                 ):

        self.in_channel = in_channel
        self.clip_num = clip_num
        self.extract_num = in_channel * clip_num
        self.root = root

        super().__init__(**kwargs)



    def load_data_list(self) -> List[dict]:

        data_list = []

        patient_ids = os.listdir(os.path.join(self.root, 'test_images'))

        for patient_id in patient_ids:
            series_ids = os.listdir(os.path.join(self.root, 'test_images', patient_id))

            for series_id in series_ids:
                series_folder = os.path.join(self.root, 'test_images', patient_id, series_id)
                file_names = os.listdir(series_folder)
                
                if patient_id == '3124' and series_id == '5842':
                    file_names = [file_name for file_name in file_names if file_name.split('.')[0] != '514']
                
                file_names = sorted(file_names, key=lambda x : int(x.split('.')[0]))

                file_names = np.array(file_names)
                extract_indexs = get_extract_indexes(len(file_names), self.extract_num)
                file_names = file_names[extract_indexs]
                img_path_list = [os.path.join(f'{self.root}/test_images/{patient_id}/{series_id}', file_name) for file_name in file_names]


                data_info = dict(
                    img_path_list=img_path_list,
                    patient_id=(int)(patient_id),
                    series_id=(int)(series_id),
                )

                data_list.append(data_info)

        return data_list#[:20]


