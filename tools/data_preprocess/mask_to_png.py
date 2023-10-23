import os
import cv2
import glob
import pydicom
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
import torch.nn.functional as F

from util import dicom_to_image


from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
import random
import nibabel as nib


from mmengine.config import Config

cfg = Config.fromfile('./configs/setting.py')

root = cfg.root
mask_root = cfg.mask_root

total_segmentator_root = cfg.total_segmentator_root


SIZE = (512, 512)



if not os.path.exists(mask_root):
     os.makedirs(mask_root)


patient_ids = os.listdir(f'{root}/train_images')


def load_mask(filepath, downsample_rate=1):
    # 加载mask，其shape与img对齐，为[n, 512, 512]
    img = nib.load(filepath).get_fdata()
    img = np.transpose(img, [1, 0, 2])
    img = np.rot90(img, 1, (1,2))
    img = img[::-1,:,:]
    img = np.transpose(img, [1, 0, 2])
    img = img[::downsample_rate, ::downsample_rate, ::downsample_rate]
    return img




reversed_series_ids = {}
with open(f'./data/processed/reversed_z.txt', 'r') as f:
    for line in f.readlines():
        patient_id, series_id = line.strip('\n').split(',')
        patient_id, series_id = int(patient_id), int(series_id)
        reversed_series_ids[series_id] = patient_id



def process_single_patient(i):
    
    patient_id = patient_ids[i]

    patient_folder = os.path.join(root, 'train_images', patient_id)

    series_ids = os.listdir(patient_folder)

    for series_id in series_ids:
		
        mask_folder = os.path.join(total_segmentator_root, patient_id, series_id)


        files = sorted(os.listdir(f'{root}/train_images/{patient_id}/{series_id}'), key=lambda x : int(x.split('.')[0]))
        files_prefix = [int(file.split('.')[0]) for file in files]        
		
        ### load pred masks
        # kidney 肾脏
        kidney_left_mask = load_mask(f'{mask_folder}/kidney_left.nii.gz')
        kidney_right_mask = load_mask(f'{mask_folder}/kidney_right.nii.gz')
        kidney_mask = kidney_left_mask + kidney_right_mask

        # liver  脾脏
        liver_mask = load_mask(f'{mask_folder}/liver.nii.gz')

        # spleen 肝脏
        spleen_mask = load_mask(f'{mask_folder}/spleen.nii.gz')


        # bowel 肠
        colon_mask = load_mask(f'{mask_folder}/colon.nii.gz')
        duodenum_mask = load_mask(f'{mask_folder}/duodenum.nii.gz')
        small_bowel_mask = load_mask(f'{mask_folder}/small_bowel.nii.gz')
        esophagus_mask = load_mask(f'{mask_folder}/esophagus.nii.gz')

        bowel_mask = colon_mask + duodenum_mask + small_bowel_mask + esophagus_mask

        names = ['kidney', 'liver', 'spleen', 'bowel']
        masks = [kidney_mask, liver_mask, spleen_mask, bowel_mask]


        for name, mask in zip(names, masks):
            mask = mask.astype(np.float32)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = mask.astype(np.uint8)

            if series_id in reversed_series_ids:
                mask = mask[::-1, :, :]


            organ_folder = os.path.join(mask_root, patient_id, series_id, name)
            if not os.path.exists(organ_folder):
                os.makedirs(organ_folder)


            for i in range(mask.shape[0]):
                file_prefix = files_prefix[i]
                file_path = os.path.join(organ_folder, f'{file_prefix}.png')               
                cv2.imwrite(file_path, mask[i])



#with Pool(processes=cpu_count()) as p:
with Pool(processes=10) as p:
	with tqdm(total=len(patient_ids)) as pbar:
		for i, _ in enumerate(p.imap_unordered(process_single_patient, iter(torch.randperm(len(patient_ids)).tolist()))):
			pbar.update()

