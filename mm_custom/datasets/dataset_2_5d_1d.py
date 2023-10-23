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
class Dataset(BaseDataset):
    def __init__(self,
                 clip_num,
                 in_channel,
                 root,
                 mask_root,
                 fold,
                 data_version,
                 png_root=None,
                 **kwargs
                 ):

        
        self.clip_num = clip_num
        self.extract_num = in_channel * clip_num
        self.root = root
        self.mask_root = mask_root
        self.png_root = png_root
        self.fold = fold
        self.in_channel = in_channel
        if self.png_root is not None:
            self.img_root = self.png_root
        else:
            self.img_root = os.path.join(self.root, 'train_images')
        self.data_version = data_version

        self.train_df = pd.read_csv(f'{root}/train.csv')
        self.train_weight_df = pd.read_csv(f'./data/processed/train_weight.csv')
        self.train_series_meta_df = pd.read_csv(f'{root}/train_series_meta.csv')
        self.image_level_labels_df = pd.read_csv(f'{root}/image_level_labels.csv')
        


        # self.reversed_series_ids = {}
        # with open(f'{process_root}/reversed_z.txt', 'r') as f:
        #     for line in f.readlines():
        #         patient_id, series_id = line.strip('\n').split(',')
        #         patient_id, series_id = int(patient_id), int(series_id)
        #         self.reversed_series_ids[series_id] = patient_id

        super().__init__(**kwargs)



    def prepare_target_dic(self, patient_id):

        labels = self.train_df[self.train_df.patient_id == patient_id].values[0][1:]

        bowel_labels = labels[0:2]
        extravasation_labels = labels[2:4]
        kidney_labels = labels[4:7]
        liver_labels = labels[7:10]
        spleen_labels = labels[10:13]


        any_injury_label = labels[13]
        any_injury_labels = np.zeros(shape=2)
        any_injury_labels[any_injury_label] = 1



        target_dic = {}

        # binary
        target_dic['bowel'] = bowel_labels
        target_dic['extravasation'] = extravasation_labels
        
        # triple
        target_dic['kidney'] = kidney_labels
        target_dic['liver'] = liver_labels
        target_dic['spleen'] = spleen_labels

        # any_injury
        target_dic['any_injury'] = any_injury_labels


        return target_dic


    def prepare_weight_dic(self, patient_id):

        weight = self.train_weight_df[self.train_weight_df.patient_id == patient_id].values[0][-6:]

        weight_dic = {}

        for i, name in enumerate(self.metainfo['all_class_names']):
            weight_dic[name] = weight[i]


        return weight_dic



    def prepare_image_level_target_dic(self, patient_id, series_id, files_prefix):
        
        # 0: 无损伤 1: 有损伤
        image_level_target_dic = {
            'Active_Extravasation': np.zeros(self.clip_num),
            'Bowel': np.zeros(self.clip_num)
        }

        img_indexes = files_prefix

        df = self.image_level_labels_df[(self.image_level_labels_df.patient_id == patient_id) & (self.image_level_labels_df.series_id == series_id)]

        injury_names = df.injury_name.unique()

        for injury_name in injury_names:
            temp_df = df[df.injury_name == injury_name]
            
            if temp_df.shape[0] == 0:
                continue

            instance_numbers = temp_df.instance_number.values

            is_injury = np.isin(img_indexes, instance_numbers)

            #image_level_target_dic[injury_name][is_injury] = 1


            is_injury = np.reshape(is_injury, (self.clip_num, self.in_channel))
            keep = np.sum(is_injury, axis=-1)
            keep = keep > 0

            image_level_target_dic[injury_name][keep] = 1


        return image_level_target_dic




    def load_data_list(self) -> List[dict]:
        
        data_list = []

        self.train_val_df = pd.read_csv(f'./data/processed/train_val_{self.data_version}.csv')

        if self.test_mode:
            df = self.train_val_df[self.train_val_df[f'fold_{self.fold}'] == 'val']#.head(10)
        else:
            df = self.train_val_df[self.train_val_df[f'fold_{self.fold}'] == 'train']


        patient_ids = df.patient_id.values


        for patient_id in patient_ids:

            target_dic = self.prepare_target_dic(patient_id)
            weight_dic = self.prepare_weight_dic(patient_id)


            patient_folder = os.path.join(self.img_root, str(patient_id))

            series_ids = os.listdir(patient_folder)

            for series_id in series_ids:
                series_id = int(series_id)
                files = sorted(os.listdir(f'{self.img_root}/{patient_id}/{series_id}'), key=lambda x : int(x.split('.')[0]))
                
                files_prefix = np.array([int(file.split('.')[0]) for file in files])
                extract_indexs = get_extract_indexes(len(files_prefix), self.extract_num)
                files_prefix = files_prefix[extract_indexs]

                image_level_target_dic = self.prepare_image_level_target_dic(patient_id, series_id, files_prefix)
                
                files = np.array(files)
                files = files[extract_indexs]
                img_path_list = [os.path.join(f'{self.img_root}/{patient_id}/{series_id}', file_name) for file_name in files]

                mask_path_list_dict = {}
                for mask_name in self.metainfo['mask_names']:
                    mask_path_list = [os.path.join(f'{self.mask_root}/{patient_id}/{series_id}', mask_name, f'{file_prefix}.png') for file_prefix in files_prefix]
                    mask_path_list = [os.path.join(f'{self.mask_root}/{patient_id}/{series_id}', mask_name, f'{file_prefix}.png') for file_prefix in files_prefix]
                    mask_path_list_dict[mask_name] = mask_path_list


                data_info = {}

                data_info = dict(
                    
                    img_path_list=img_path_list,
                    mask_path_list_dict=mask_path_list_dict,
                   
                    patient_id=patient_id,
                    series_id=series_id,

                    target_dic=target_dic,
                    weight_dic=weight_dic,
                    image_level_target_dic=image_level_target_dic,

                    metainfo=self.metainfo
                )

                data_list.append(data_info)


        return data_list#[:20]









