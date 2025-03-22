_base_ = ['../_base_/seg_default_runtime.py'] 
backend_args = None
custom_imports = dict(imports=['mmdet.datasets.kitti_dataset'], allow_failed_imports=False)

##############################################################################
# 1. 预训练权重
##############################################################################
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/' \
             'vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

##############################################################################
# 2. 缩小图像尺寸 & batch size
##############################################################################
# 将原先 1120 大幅降低到 384 (或更小 224)。
# batch_size改为1，以尽量降低一次forward/backward的显存消耗。
base_img_size = 384

##############################################################################
# 3. detection任务 + GiT 模型
##############################################################################
det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    # 这里的80只是“原预留”类别数
    num_vocal=base_img_size * 2 + 1 + 80 + 1,
    max_decoder_length=5,
    global_only_image=True
)

model = dict(
    type='GiT',
    # 多任务列表，为避免空序列
    support_tasks=['detection','semantic_segmentation','instance_segmentation','caption','grounding'],
    # 需要的话可尝试 use_checkpoints=False 看是否更省显存
    use_checkpoints=True,
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    bert_embed=dict(type='bert-base', hidden_size=768, pretrain_path='./bert_embed.pt'),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=255,
        pad_size_divisor=224
    ),
    backbone=dict(
        type='ViTGiT',
        arch='base',
        # 缩小img_size
        img_size=base_img_size,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,  # 若OOM仍严重，也可尝试改False
        new_more_layers=['win','win','win','win','win','win'],
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.')
    ),
    head_list=dict(
        detection_head=dict(
            type='GiTDetHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[dict(type='PointsL1Cost', weight=5.0, box_format='xywh')],
                )
            ),
            test_cfg=dict(max_per_img=100)
        )
    )
)

##############################################################################
# 4. Data Pipeline: 仍用 KittiDataset，但将 base_img_size=384
##############################################################################
det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AddMetaInfo',
        meta_dict=dict(
            task_name='detection',
            head_cfg=dict(
                num_classes=3,
                num_vocal=(base_img_size * 2 + 1) + 3 + 1,
                num_bins=base_img_size * 2,
                dec_length=5,
                arg_max_inference=True
            ),
            git_cfg=det_cfgs
        )
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                # 384 => 384
                dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)
            ],
            [
                # 这里也可改小: (400->256, 600->320等)
                dict(type='RandomChoiceResize', scales=[(400, 4200), (600, 4200)], keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(256, 320), allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)
            ]
        ]
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor',
                   'flip', 'flip_direction','task_name','head_cfg','git_cfg')
    ),
]
# test pipeline 同理
det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AddMetaInfo',
        meta_dict=dict(
            task_name='detection',
            head_cfg=dict(
                num_classes=3,
                num_vocal=(base_img_size * 2 + 1) + 3 + 1,
                num_bins=base_img_size * 2,
                dec_length=5
            ),
            git_cfg=det_cfgs
        )
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',
                   'task_name','head_cfg','git_cfg')
    )
]

train_dataloader = dict(
    batch_size=1,  # 改为1
    num_workers=1, # 也可降低num_workers减少CPU开销
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='KittiDataset',
        data_root='',
        ann_file='data/kitti/training/label_2',
        data_prefix=dict(img='data/kitti/training/image_2'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=det_train_pipeline,
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='',
        ann_file='data/kitti/training/label_2',
        data_prefix=dict(img='data/kitti/training/image_2'),
        test_mode=True,
        pipeline=det_test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

##############################################################################
# 5. 优化器与训练设置
##############################################################################
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.layers.6': dict(lr_mult=0.2286),
            'backbone.layers.7': dict(lr_mult=0.3571),
            'backbone.layers.8': dict(lr_mult=0.4858),
            'backbone.layers.9': dict(lr_mult=0.6143),
            'backbone.layers.10': dict(lr_mult=0.7429),
            'backbone.layers.11': dict(lr_mult=0.8714),
            'backbone.layers.12': dict(lr_mult=1.0),
            'backbone.layers.13': dict(lr_mult=1.0),
            'backbone.layers.14': dict(lr_mult=1.0),
            'backbone.layers.15': dict(lr_mult=1.0),
            'backbone.layers.16': dict(lr_mult=1.0),
            'backbone.layers.17': dict(lr_mult=1.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

val_evaluator = dict(
    type='KittiMetric',
    classes=['Car','Pedestrian','Cyclist'],
    format_only=True,
    outfile_prefix='work_dirs/kitti_result/kitti',
    
)
test_evaluator = val_evaluator

max_iters = 120000  # 也可减少迭代次数
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=max_iters,
        eta_min=2e-6,
        begin=0,
        end=max_iters,
        by_epoch=False
    )
]

auto_scale_lr = dict(base_batch_size=4)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(type='LogProcessor', window_size=200, by_epoch=False)
