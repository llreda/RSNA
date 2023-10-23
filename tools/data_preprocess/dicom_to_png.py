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



from mmengine.config import Config

cfg = Config.fromfile('./configs/setting.py')

root = cfg.root
png_root = cfg.png_root


patient_ids = os.listdir(f'{root}/train_images')




def process_single_patient(i):
    
    patient_id = patient_ids[i]

    patient_folder = os.path.join(root, 'train_images', patient_id)

    series_ids = os.listdir(patient_folder)

    for series_id in series_ids:
        series_folder = os.path.join(patient_folder, series_id)
		
        out_folder = f'{png_root}/{patient_id}/{series_id}'
        if not os.path.exists(out_folder):
                os.makedirs(out_folder)

        file_names = os.listdir(series_folder)
        file_names = sorted(file_names, key=lambda x : int(x.split('.')[0]))
        

        for file_name in file_names:
                file_path = os.path.join(series_folder, file_name)
                img = dicom_to_image(pydicom.dcmread(file_path))

                file_name = file_name.replace('.dcm', '.png')
                out_file_path = f'{out_folder}/{file_name}'

                cv2.imwrite(out_file_path, img)






#with Pool(processes=cpu_count()) as p:
with Pool(processes=10) as p:
	with tqdm(total=len(patient_ids)) as pbar:
		for i, _ in enumerate(p.imap_unordered(process_single_patient, iter(torch.randperm(len(patient_ids)).tolist()))):
			pbar.update()
