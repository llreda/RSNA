_base_ = [
    './_base_/default_runtime.py',
    './setting.py'
]

root = _base_.root
mask_root = _base_.mask_root
png_root = _base_.png_root

scale = _base_.scale
fold = _base_.fold


data_version = _base_.data_version


crop_scale = (384, 256) # [w, h]
#crop_scale = (512, 384)


in_channel = 3

T = scale[0] // in_channel
clip_num = scale[0] // in_channel


max_lr = 1e-4

max_iters = 1400
warmup_iters = 100

val_interval = 200


val_begin = max_iters

#val_interval = 10
#val_begin = 10


batch_size = 4
num_workers = 8



metainfo = dict(
    all_class_names=['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury'],

    class_names=['bowel', 'extravasation', 'kidney', 'liver', 'spleen'],
    out_channels=[2, 2, 3, 3, 3],

    mask_names=['bowel', 'kidney', 'liver', 'spleen'],
    extra_names = ['extravasation'],

    metric_class_names=['bowel', 'extravasation', 'kidney', 'liver', 'spleen', 'any_injury'],

    image_level_class_names = ['Bowel', 'Active_Extravasation'],
    binary_class_names=['bowel', 'extravasation'],
)



log_processor=dict(by_epoch=False)


#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)


################################ model ################################

model = dict(
    type='mm_custom.ATDNetMask',
    

    # backbone
    backbone=dict(
        type='mm_custom.InternImage',
        core_op='DCNv3',
        channels=112,
        depths=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        #drop_rate=0.2,
        mlp_ratio=4.,
        drop_path_rate=0.4, # 0.4
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth'
        )
    ),


    # neck
    neck=dict(
        type='mm_custom.UnetPlusPlus',

        #in_channels=[80 * 2, 80 * 4, 80 * 8],
        #out_channels=[256, 256, 256],   # 上采样2倍后的结果，stride [8, 16, 32] -> [4, 8, 16]

        in_channels=[112 * 2, 112 * 4, 112 * 8],
        out_channels=[256, 256, 256],   # 上采样2倍后的结果，stride [8, 16, 32] -> [4, 8, 16]

        norm_cfg=norm_cfg,
        act_cfg=dict(type='Swish'),  # ReLU GELU Swish PReLU
        interpolate_mode='bilinear', # nearest bilinear
        attention_type=None,        # None scse
        with_cp=True,
    ),  

    
    decode_head=dict(
        type='mm_custom.AtdHeadALLMask',

        transformer_decoder_cfg_masks=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=1,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)
                ),
            init_cfg=None
        ),

        transformer_decoder_cfg_extra=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=1,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)
                ),
            init_cfg=None
        ),

        clip_num=clip_num,
        feat_channels=256,
        mask_channels=256,
        in_channels=[256, 256, 256],   # 最高分辨率是mask_feature的大小
        # in_channels=[80, 80 * 2, 80 * 4, 80 * 8],   # 最高分辨率是mask_feature的大小
        mask_weight=3, # default: 3
        metainfo=metainfo       
    ),


    metainfo=metainfo
)



################################ data #################################



train_pipeline = [
    dict(type='mm_custom.LoadImage', clip_num=clip_num, in_channel=in_channel),
    dict(type='mm_custom.ResizeImgs', scale=(clip_num, scale[1], scale[2]), in_channel=in_channel),
    
    dict(type='mm_custom.LoadMask', clip_num=clip_num, in_channel=in_channel),
    dict(type='mm_custom.ResizeMaskDic', scale=(clip_num, scale[1], scale[2])),
    
    dict(type='mm_custom.CropValid', min_area_thres=0.1, min_ratio_thres=1.4, max_ratio_thres=1.4, mean_ratio=1.4, rescaled_size=crop_scale,  pixel_thresh=25, ratio_thresh=0.02, padding_size=(6, 6)),

    dict(type='mm_custom.RandomCropResize', prob=0.2,
                 scale=(0.7, 1.3),
                 ratio=(1.2, 1.4),
                 height=crop_scale[1],
                 width=crop_scale[0]
    ),


    dict(type='mm_custom.RandomRotate', prob=0.2, degree=20, pad_val=0, seg_pad_val=0),
    dict(type='mm_custom.RandomFlip'),

    dict(type='mm_custom.PackInputs'),
]




val_pipeline = [
    dict(type='mm_custom.LoadImage', clip_num=clip_num, in_channel=in_channel),
    dict(type='mm_custom.ResizeImgs', scale=(clip_num, scale[1], scale[2]), in_channel=in_channel),
    dict(type='mm_custom.CropValid', min_area_thres=0.1, min_ratio_thres=1.4, max_ratio_thres=1.4, mean_ratio=1.4, rescaled_size=crop_scale,  pixel_thresh=25, ratio_thresh=0.02, padding_size=(6, 6)),
    dict(type='mm_custom.PackInputs'),
]



test_pipeline = [
    dict(type='mm_custom.LoadImage', clip_num=clip_num, in_channel=in_channel),
    dict(type='mm_custom.ResizeImgs', scale=(clip_num, scale[1], scale[2]), in_channel=in_channel),
    dict(type='mm_custom.CropValid', min_area_thres=0.1, min_ratio_thres=1.4, max_ratio_thres=1.4, mean_ratio=1.4, rescaled_size=crop_scale,  pixel_thresh=25, ratio_thresh=0.02, padding_size=(6, 6)),
    dict(type='mm_custom.PackInputs'),
]




train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='mm_custom.Dataset',
        clip_num=clip_num,
        in_channel=in_channel,
        root=root, 
        mask_root=mask_root,
        png_root=png_root,
        fold=fold,
        data_version=data_version,
        metainfo=metainfo,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True),
)


train_cfg = dict(
    by_epoch=False, 
    max_iters=max_iters, 
    val_interval=val_interval,
    val_begin=val_begin,
)



val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='mm_custom.Dataset',
        clip_num=clip_num,
        in_channel=in_channel,
        root=root, 
        mask_root=mask_root,
        png_root=png_root,
        fold=fold,
        data_version=data_version,
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
)



val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='mm_custom.Metric',
    metainfo=metainfo,
)



test_cfg = dict(type='TestLoop')

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='mm_custom.DatasetSub',
        clip_num=clip_num,
        in_channel=in_channel,
        root=root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
)

test_evaluator = dict(
    type='mm_custom.Metric',
    metainfo=metainfo,
)



### optimizer
optimizer_AdamW = dict(
    type='AdamW',
    lr=max_lr,
    #weight_decay=1e-2,
)



optim_wrapper = dict(
    type='OptimWrapper', # 'AmpOptimWrapper', 'OptimWrapper'
    optimizer=optimizer_AdamW,

    clip_grad=None,
    #accumulative_counts=accumulative_counts,
)


### scheduler
param_scheduler = [

    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=warmup_iters,
    ),

    dict(
        type='CosineAnnealingLR',
        T_max=max_iters-warmup_iters,
        eta_min=1e-7, #1e-6,
        by_epoch=False,
        begin=warmup_iters,
        end=max_iters
    )
]



# default_hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=max_iters, max_keep_ckpts=1),

    # print log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50),
)


