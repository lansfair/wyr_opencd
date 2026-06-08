backbone_inchannels = 12
crop_size = (
    256,
    256,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.olmoearthfm.models',
    ])
data_preprocessor = dict(
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    test_cfg=dict(size_divisor=32),
    type='DualInputSegDataPreProcessor')
data_root = '/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/'
dataset_type = 'OSCD_Dataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=True, type='LoggerHook'),
    visualization=dict(
        draw=True, interval=1, show=False, type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/mnt/ht2-nas2/EO_test/wry/open-cd/work_dirs/olmoearth-base_upernet_1xb16-50e_oscd-s2-256x256/best_mIoU_epoch_43.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=768,
        in_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=0.4,
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='mmseg.FCNHead'),
    backbone=dict(
        init_cfg=dict(
            checkpoint=
            '/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth',
            type='Pretrained'),
        modality='sentinel2_l2a',
        model_config_path=
        '/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/config.json',
        num_timesteps=1,
        out_channels=768,
        patch_size=4,
        pooling_type='mean',
        type='OlmoEarthBackbone'),
    backbone_inchannels=12,
    data_preprocessor=dict(
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        test_cfg=dict(size_divisor=32),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=1.0,
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='mmseg.UPerHead'),
    neck=dict(
        embed_dim=768,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        policy='abs_diff',
        rescales=[
            4,
            2,
            1,
            0.5,
        ],
        type='FeatureFusionPyramid'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        128,
        128,
    )),
    train_cfg=dict(),
    type='SiamEncoderDecoder')
model_config_path = '/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/config.json'
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_timesteps = 1
olmoearth_fm_checkpoint = '/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base/weights.pth'
olmoearth_model_dir = '/mnt/ht2-nas2/EO_test/model/OlmoEarth-v1-Base'
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        anneal_strategy='cos',
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_max=0.001,
        pct_start=0.0,
        type='OneCycleLR'),
]
patch_area = 0.0256
patch_size = 4
randomness = dict(seed=0)
resume = False
s2_band_bandwidths = [
    65,
    35,
    30,
    115,
    15,
    15,
    20,
    20,
    90,
    180,
    20,
    20,
]
s2_band_stats = dict(
    mean=[
        1117.2,
        1041.8,
        946.5,
        2301.2,
        1199.1,
        2003.0,
        2374.0,
        2599.7,
        1820.6,
        1118.2,
        1353.7,
        732.1,
    ],
    std=[
        736.0,
        684.8,
        620.0,
        1545.5,
        791.9,
        1341.3,
        1595.4,
        1750.1,
        1216.5,
        736.7,
        897.3,
        475.1,
    ])
s2_band_wavelengths = [
    490,
    560,
    665,
    842,
    705,
    740,
    783,
    860,
    1610,
    2190,
    440,
    940,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/',
        official_format=True,
        pipeline=[
            dict(type='MultiImgLoadGeoTiffImageFromFile'),
            dict(
                mean=[
                    1117.2,
                    1041.8,
                    946.5,
                    2301.2,
                    1199.1,
                    2003.0,
                    2374.0,
                    2599.7,
                    1820.6,
                    1118.2,
                    1353.7,
                    732.1,
                ],
                std=[
                    736.0,
                    684.8,
                    620.0,
                    1545.5,
                    791.9,
                    1341.3,
                    1595.4,
                    1750.1,
                    1216.5,
                    736.7,
                    897.3,
                    475.1,
                ],
                type='MultiImgNormalizeMultibandImage'),
            dict(type='MultiImgLoadOSCDAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        split='test',
        type='OSCD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),
    dict(
        mean=[
            1117.2,
            1041.8,
            946.5,
            2301.2,
            1199.1,
            2003.0,
            2374.0,
            2599.7,
            1820.6,
            1118.2,
            1353.7,
            732.1,
        ],
        std=[
            736.0,
            684.8,
            620.0,
            1545.5,
            791.9,
            1341.3,
            1595.4,
            1750.1,
            1216.5,
            736.7,
            897.3,
            475.1,
        ],
        type='MultiImgNormalizeMultibandImage'),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            data_root='/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/',
            official_format=True,
            pipeline=[
                dict(type='MultiImgLoadGeoTiffImageFromFile'),
                dict(type='MultiImgLoadOSCDAnnotations'),
                dict(
                    mean=[
                        1117.2,
                        1041.8,
                        946.5,
                        2301.2,
                        1199.1,
                        2003.0,
                        2374.0,
                        2599.7,
                        1820.6,
                        1118.2,
                        1353.7,
                        732.1,
                    ],
                    std=[
                        736.0,
                        684.8,
                        620.0,
                        1545.5,
                        791.9,
                        1341.3,
                        1595.4,
                        1750.1,
                        1216.5,
                        736.7,
                        897.3,
                        475.1,
                    ],
                    type='MultiImgNormalizeMultibandImage'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        256,
                        256,
                    ),
                    type='MultiImgRandomCrop'),
                dict(
                    direction='horizontal',
                    prob=0.5,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
                dict(type='MultiImgPackSegInputs'),
            ],
            split='train',
            type='OSCD_Dataset'),
        times=10,
        type='RepeatDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(
        mean=[
            1117.2,
            1041.8,
            946.5,
            2301.2,
            1199.1,
            2003.0,
            2374.0,
            2599.7,
            1820.6,
            1118.2,
            1353.7,
            732.1,
        ],
        std=[
            736.0,
            684.8,
            620.0,
            1545.5,
            791.9,
            1341.3,
            1595.4,
            1750.1,
            1216.5,
            736.7,
            897.3,
            475.1,
        ],
        type='MultiImgNormalizeMultibandImage'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            256,
            256,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/',
        official_format=True,
        pipeline=[
            dict(type='MultiImgLoadGeoTiffImageFromFile'),
            dict(
                mean=[
                    1117.2,
                    1041.8,
                    946.5,
                    2301.2,
                    1199.1,
                    2003.0,
                    2374.0,
                    2599.7,
                    1820.6,
                    1118.2,
                    1353.7,
                    732.1,
                ],
                std=[
                    736.0,
                    684.8,
                    620.0,
                    1545.5,
                    791.9,
                    1341.3,
                    1595.4,
                    1750.1,
                    1216.5,
                    736.7,
                    897.3,
                    475.1,
                ],
                type='MultiImgNormalizeMultibandImage'),
            dict(type='MultiImgLoadOSCDAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        split='test',
        type='OSCD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    save_dir=
    '/mnt/ht2-nas2/EO_test/wry/open-cd/projects/olmoearthfm/infer-results',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = '/mnt/ht2-nas2/EO_test/wry/open-cd/work_dirs/olmoearth-base_upernet_1xb16-50e_oscd-s2-256x256'
