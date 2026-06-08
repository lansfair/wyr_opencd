backbone_inchannels = 13
copernicus_fm_checkpoint = '/mnt/ht2-nas2/EO_test/cyz/Copernicus-FM/weights/CopernicusFM_ViT_base_varlang_e100.pth'
crop_size = (
    256,
    256,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.copernicusfm',
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
    visualization=dict(draw=True, interval=1, type='CDVisualizationHook'))
default_scope = 'opencd'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/mnt/ht2-nas2/EO_test/wry/open-cd/work_dirs/copernicusfm-base_upernet_1xb16-50e_oscd-s2-256x256/best_mIoU_epoch_21.pth'
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
        arch='base',
        band_bandwidths=[
            20,
            65,
            35,
            30,
            15,
            15,
            20,
            115,
            20,
            20,
            30,
            90,
            180,
        ],
        band_wavelengths=[
            440,
            490,
            560,
            665,
            705,
            740,
            783,
            842,
            860,
            940,
            1370,
            1610,
            2190,
        ],
        frozen_exclude=[],
        init_cfg=dict(
            checkpoint=
            '/mnt/ht2-nas2/EO_test/cyz/Copernicus-FM/weights/CopernicusFM_ViT_base_varlang_e100.pth',
            type='Pretrained'),
        input_mode='spectral',
        kernel_size=16,
        norm_eval=True,
        patch_area=0.0256,
        type='mmseg.CopernicusFMBackbone',
        var_option='spectrum'),
    backbone_inchannels=13,
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
norm_cfg = dict(requires_grad=True, type='SyncBN')
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
randomness = dict(seed=0)
resume = False
s2_band_bandwidths = [
    20,
    65,
    35,
    30,
    15,
    15,
    20,
    115,
    20,
    20,
    30,
    90,
    180,
]
s2_band_stats = dict(
    mean=[
        1353.7,
        1117.2,
        1041.8,
        946.5,
        1199.1,
        2003.0,
        2374.0,
        2301.2,
        2599.7,
        732.1,
        12.1,
        1820.6,
        1118.2,
    ],
    std=[
        897.3,
        736.0,
        684.8,
        620.0,
        791.9,
        1341.3,
        1595.4,
        1545.5,
        1750.1,
        475.1,
        98.3,
        1216.5,
        736.7,
    ])
s2_band_wavelengths = [
    440,
    490,
    560,
    665,
    705,
    740,
    783,
    842,
    860,
    940,
    1370,
    1610,
    2190,
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
                    1353.7,
                    1117.2,
                    1041.8,
                    946.5,
                    1199.1,
                    2003.0,
                    2374.0,
                    2301.2,
                    2599.7,
                    732.1,
                    12.1,
                    1820.6,
                    1118.2,
                ],
                std=[
                    897.3,
                    736.0,
                    684.8,
                    620.0,
                    791.9,
                    1341.3,
                    1595.4,
                    1545.5,
                    1750.1,
                    475.1,
                    98.3,
                    1216.5,
                    736.7,
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
            1353.7,
            1117.2,
            1041.8,
            946.5,
            1199.1,
            2003.0,
            2374.0,
            2301.2,
            2599.7,
            732.1,
            12.1,
            1820.6,
            1118.2,
        ],
        std=[
            897.3,
            736.0,
            684.8,
            620.0,
            791.9,
            1341.3,
            1595.4,
            1545.5,
            1750.1,
            475.1,
            98.3,
            1216.5,
            736.7,
        ],
        type='MultiImgNormalizeMultibandImage'),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        dataset=dict(
            data_root='/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/',
            official_format=True,
            pipeline=[
                dict(type='MultiImgLoadGeoTiffImageFromFile'),
                dict(type='MultiImgLoadOSCDAnnotations'),
                dict(
                    mean=[
                        1353.7,
                        1117.2,
                        1041.8,
                        946.5,
                        1199.1,
                        2003.0,
                        2374.0,
                        2301.2,
                        2599.7,
                        732.1,
                        12.1,
                        1820.6,
                        1118.2,
                    ],
                    std=[
                        897.3,
                        736.0,
                        684.8,
                        620.0,
                        791.9,
                        1341.3,
                        1595.4,
                        1545.5,
                        1750.1,
                        475.1,
                        98.3,
                        1216.5,
                        736.7,
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
            1353.7,
            1117.2,
            1041.8,
            946.5,
            1199.1,
            2003.0,
            2374.0,
            2301.2,
            2599.7,
            732.1,
            12.1,
            1820.6,
            1118.2,
        ],
        std=[
            897.3,
            736.0,
            684.8,
            620.0,
            791.9,
            1341.3,
            1595.4,
            1545.5,
            1750.1,
            475.1,
            98.3,
            1216.5,
            736.7,
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
                    1353.7,
                    1117.2,
                    1041.8,
                    946.5,
                    1199.1,
                    2003.0,
                    2374.0,
                    2301.2,
                    2599.7,
                    732.1,
                    12.1,
                    1820.6,
                    1118.2,
                ],
                std=[
                    897.3,
                    736.0,
                    684.8,
                    620.0,
                    791.9,
                    1341.3,
                    1595.4,
                    1545.5,
                    1750.1,
                    475.1,
                    98.3,
                    1216.5,
                    736.7,
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
    '/mnt/ht2-nas2/EO_test/wry/open-cd/projects/copernicusfm/infer-results',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = '/mnt/ht2-nas2/EO_test/wry/open-cd/work_dirs/copernicusfm-base_upernet_1xb16-50e_oscd-s2-256x256'
