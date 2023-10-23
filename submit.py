# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import datetime
import torch
import numpy as np
import pandas as pd
import cv2


from typing import Dict, List, Optional, Tuple, Union

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.dataset import BaseDataset


import pydicom
import numpy as np

import torch
import torch.nn.functional as F


from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
import random



from mmengine.config import Config

config = Config.fromfile('./configs/setting.py')

root = config.root
cfg_files_list = config.cfg_files_list
load_froms_list = config.load_froms_list



SUBMISSION_DIR = config.SUBMISSION_DIR

if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR)


batch_size = 4
num_workers = 4



def load_cfg(cfg_file, load_from):
    cfg = Config.fromfile(cfg_file)
    cfg.launcher = 'pytorch'
    cfg.load_from = load_from
    cfg.test_dataloader.dataset.root = root    

    return cfg    



def load_runner(cfg):
    runner = Runner(
        model=cfg.model,
        work_dir='submit',
        load_from=cfg.load_from,
        test_dataloader=cfg.test_dataloader,
        test_cfg=cfg.test_cfg,
        test_evaluator=cfg.test_evaluator
    )

    runner.load_or_resume()
    runner.model = runner.model.cuda()
    runner.model.eval()

    return runner



def load_cfgs(cfg_files, load_froms):
    cfgs = [load_cfg(cfg_file, load_from) for cfg_file, load_from in zip(cfg_files, load_froms)]
    return cfgs


def load_runners(cfgs):
    runners = [load_runner(cfg) for cfg in cfgs]
    return runners



assert len(cfg_files_list) == len(load_froms_list)

all_model_num = 0
folds_num = len(cfg_files_list)
all_pred_probs = []

class_names = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']

for fold in range(folds_num):
    cfg_files = cfg_files_list[fold]
    load_froms = load_froms_list[fold]

    model_num = len(load_froms)
    
    all_model_num += model_num

    assert len(cfg_files) == len(load_froms)

    # load cfgs
    cfgs = load_cfgs(cfg_files, load_froms)

    # load runners
    runners = load_runners(cfgs)

    test_dataloader = runners[0].test_dataloader
    N = len(test_dataloader.dataset)
    batch_size = test_dataloader.batch_size
    pred_probs = np.zeros(shape=(N, 13))
    patient_ids, series_ids = [], []

    for batch, data in enumerate(test_dataloader):
        with torch.no_grad():
            # runners
            for i in range(model_num):

                if i == 0:
                    patient_ids.append(data['data_batch']['patient_id'].numpy())
                    series_ids.append(data['data_batch']['series_id'].numpy())

                results_batch = runners[i].model.test_step(data)

                for ii, result in enumerate(results_batch):

                    idx = batch * batch_size + ii

                    pred_prob = []
                    for name in class_names:
                        pred_prob.append(result.pred_prob_dic[name])
                    
                    pred_prob = torch.cat(pred_prob)

                    pred_probs[idx] += pred_prob.cpu().detach().numpy()
    
    patient_ids = np.concatenate(patient_ids).reshape(-1, 1).astype(int)
    series_ids = np.concatenate(series_ids).reshape(-1, 1).astype(int)
    
    
    all_pred_probs.append(pred_probs)
    
    del runners
    torch.cuda.empty_cache()



new_pred_probs = None


for fold in range(folds_num):
    if fold == 0:
        new_pred_probs = all_pred_probs[fold].copy()
    else:
        new_pred_probs += all_pred_probs[fold].copy()

pred_probs = new_pred_probs / float(all_model_num)

data = np.concatenate([patient_ids, series_ids, pred_probs], axis=1) # [N, col_nums]

column_names = ['patient_id', 'series_id']

binary_class_names = ['bowel', 'extravasation']
triple_class_names = ['kidney', 'liver', 'spleen']
any_injury_class_names = ['any_injury']

binary_injury_types = ['healthy', 'injury']
triple_injury_types = ['healthy', 'low', 'high']

for name in class_names:
    if name in binary_class_names:
        injury_types = binary_injury_types
    elif name in triple_class_names:
        injury_types = triple_injury_types
    else:
        raise NotImplementedError
    
    for injury_type in injury_types:
        column_names.append(f'{name}_{injury_type}')

pred_df = pd.DataFrame(data=data, columns=column_names)
pred_df = pred_df.groupby(['patient_id'], as_index=False).mean()
del pred_df['series_id']


sub_df = pd.read_csv(f'{root}/sample_submission.csv')
for patient_id in pred_df.patient_id.values:
    if patient_id in sub_df.patient_id.values:
        sub_df.loc[sub_df.patient_id == patient_id] = pred_df[pred_df.patient_id == patient_id].values

sub_df.to_csv(f"{SUBMISSION_DIR}/submission.csv", index=False)


