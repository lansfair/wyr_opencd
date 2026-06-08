dataset_type = 'LEVIRCD_Dataset'
data_root = '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/LEVIR-CD'
crop_size = (1024, 1024)

# LEVIR-CD 是 RGB 3 通道，用 ImageNet 均值方差
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
)

train_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),  
    dict(
        type='MultiImgNormalizeMultibandImage',    
        mean=img_norm_cfg['mean'],
        std=img_norm_cfg['std']),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),  
    dict(
        type='MultiImgNormalizeMultibandImage',     
        mean=img_norm_cfg['mean'],
        std=img_norm_cfg['std']),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            split='train',
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator