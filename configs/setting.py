

root = '/data/competitions/wangh/kaggle/2023/RSNA-2023-Abdominal-Trauma-Detection'

process_root = '/data/competitions/wangh/kaggle/2023/RSNA-2023-Abdominal-Trauma-Detection/process_release'
#process_root = '/data/competitions/wangh/kaggle/2023/RSNA-2023-Abdominal-Trauma-Detection/process_release_test'


SUBMISSION_DIR = './submission'



# config_files_list contains all models that you want ensemble, each one has a corresponding checkpoint in load_froms_list
# In each sub list in cfg_files_list, the pipelines are the same, that is, the sequence length used is the same (can share a dataloader)，
# but different data folds are used
cfg_files_list = [
    [
        'checkpoints/atd_cfg_b_fold0_data_v2-0.0001-24x384x256-data_v2-fold_0/atd_cfg_b_fold0_data_v2.py',
    ],
    [
        'checkpoints/atd_cfg_b_fold1_data_v2-0.0001-32x384x256-data_v2-fold_1/atd_cfg_b_fold1_data_v2.py',
    ],
    [
        'checkpoints/atd_cfg_b_fold2_data_v2-0.0001-48x384x256-data_v2-fold_2/atd_cfg_b_fold2_data_v2.py'
    ]
]
    
load_froms_list = [
    [
        'checkpoints/atd_cfg_b_fold0_data_v2-0.0001-24x384x256-data_v2-fold_0/iter_1400.pth',
    ],
    [
        'checkpoints/atd_cfg_b_fold1_data_v2-0.0001-32x384x256-data_v2-fold_1/iter_1400.pth',
    ],
    [
        'checkpoints/atd_cfg_b_fold2_data_v2-0.0001-48x384x256-data_v2-fold_2/iter_1400.pth'
    ]
]


total_segmentator_root = f'{process_root}/my-total-segmentator'

mask_root = f'{process_root}/train_masks_png'
png_root = f'{process_root}/train_images_png'



fold = 0
scale = (24 * 3, 512, 512) # [d, h, w]

data_version = 'v2'   # v2, v3, v4


