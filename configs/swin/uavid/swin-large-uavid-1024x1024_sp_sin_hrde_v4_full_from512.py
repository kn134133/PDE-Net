"""UAVid spatial-parallel Swin configuration with SIN and HRDE V4."""
_base_ = [
    '../../_base_/models/upernet_swin.py',
    '../../_base_/datasets/uavid_1024x1024_sp.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

custom_imports = dict(
    imports=['mmseg.datasets', 'mmseg.models', 'mmcv_custom'],
    allow_failed_imports=False)

crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SPSegDataPreProcessor_hw',
    size=crop_size)

# Set initialization through `load_from` or the CLI when an external checkpoint is available.
model = dict(
    type='SPEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer_sp_layer',
        init_cfg=None,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        frozen_stages=-1,
        drop_path_rate=0.3,
    ),
    decode_head=dict(
        type='UPerHead_SIN_HRDE_V4',
        in_channels=[192, 384, 768, 1536],
        num_classes=8,
        dropout_ratio=0.1,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=5.0)
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        num_classes=8,
        dropout_ratio=0.1,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='DiceLoss', loss_weight=2.0)
        ]
    ),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
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
            'norm': dict(decay_mult=0.),
            'spattns': dict(lr_mult=1.0),
            'sin': dict(lr_mult=2.0),
            'hrde': dict(lr_mult=2.0),
        }),
    constructor='LayerDecayOptimizerConstructorSwin'
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0,
         begin=1500, end=8000, by_epoch=False),
]

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=8000, val_interval=500)
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500,
                    max_keep_ckpts=2, save_best='mIoU'),
)

# Same as baseline training
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root='data/uavid/1024',
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root='data/uavid/ori',
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root='data/uavid/ori',
    )
)

find_unused_parameters = True

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
