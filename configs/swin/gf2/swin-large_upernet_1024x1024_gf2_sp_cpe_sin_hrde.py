"""GF2-GID-15 spatial-parallel Swin configuration with sinusoidal CPE and HRDE."""
_base_ = [
    '../../_base_/models/upernet_swin.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

dataset_type = 'BaseSegDataset'
data_root = 'data/01_1024_rgb_split'

classes = ['industrial_land', 'urban_residential', 'rural_residential',
           'traffic_land', 'paddy_field', 'irrigated_land', 'dry_cropland',
           'garden_land', 'arbor_forest', 'shrub_land', 'natural_meadow',
           'artificial_meadow', 'river', 'lake', 'pond']

palette = [[0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
           [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
           [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
           [0, 127, 191], [0, 127, 255], [0, 100, 155]]

crop_size = (1024, 1024)

data_preprocessor = dict(
    type='SPSegDataPreProcessor_hw',
    size=crop_size)

# Set initialization through `load_from` or the CLI when an external checkpoint is available.
model = dict(
    type='SPEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer_sp_layer_cpe',
        init_cfg=None,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        frozen_stages=-1,
        sinusoidal=True,
    ),
    decode_head=dict(
        type='UPerHead_HRDE_std',  # 标准 UPerHead + HRDE
        in_channels=[192, 384, 768, 1536],
        num_classes=15,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=15,
    )
)

custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=1.0,
        depths=(2, 2, 18, 2),
        custom_keys={
            'bias': dict(decay_mult=0.),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    constructor='LayerDecayOptimizerConstructorSwin'
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=2000,
        by_epoch=False,
    )
]

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=2000, val_interval=200)
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200,
                    max_keep_ckpts=5, save_best='mIoU'),
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

num_batch_size = 4
train_dataloader = dict(
    batch_size=num_batch_size,
    num_workers=num_batch_size,
    persistent_workers=True,
    sampler=dict(type='SPInfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/img', seg_map_path='train/mask'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        reduce_zero_label=True,
        metainfo=dict(classes=classes, palette=palette),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/img', seg_map_path='val/mask'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        reduce_zero_label=True,
        metainfo=dict(classes=classes, palette=palette),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/img', seg_map_path='test/mask'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        reduce_zero_label=True,
        metainfo=dict(classes=classes, palette=palette),
    ),
)

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

find_unused_parameters = True
