# single_occupancy_base.py 
_base_ = ['../_base_/seg_default_runtime.py']

# ---------------------------
# 基础设置与预训练权重配置
# ---------------------------
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

custom_imports = dict(
    imports=[
        'mmdet.datasets.nuscenes_occ',
        'mmdet.datasets.pipelines.load_annotations_3d_e2e',
        'mmdet.datasets.pipelines.generate_occ_flow_labels',
        'mmdet.models.dense_heads.git_occ_head',
        # 新增：加载前摄像头图像的流水线模块
        'mmdet.datasets.pipelines.load_front_camera_image',
        'mmdet.datasets.pipelines.resize_for_occ_input',
       # 'mmdet.evaluation.metrics.occ_eval'
    ],
    allow_failed_imports=False
)

# ---------------------------
# Occupancy（occ）相关配置
# ---------------------------
# 参考 UniAD 配置中 occflow_grid_conf 的设置
occ_cfgs = dict(
    mode='occupancy_prediction',
    grid_conf=dict(
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5],
        zbound=[-5.0, 3.0, 8.0],
    ),
    ignore_index=255,
    only_vehicle=True,
    occ_n_future=4,
    # 以下参数与语义分割配置中保持一致
    grid_resolution_perwin=(1, 1),
    samples_grids_eachwin=1,
    grid_interpolate=False,
    num_vocal=30524,
    global_only_image=False,
    use_vocab_list=False
)

# 定义 occ head 额外参数（这里单独配置也可，二者数值保持一致）
occ_head_cfg = dict(
    num_classes=2,
    grid_resolution_perwin=(1, 1),
    grid_interpolate=False,
    num_vocal=30524,
    samples_grids_eachwin=32,
    global_only_image=True,
    use_vocab_list=False,
)

# ---------------------------
# 模型配置
# ---------------------------
model = dict(
    type='GiT',
    support_tasks=['occupancy_prediction'],
    use_checkpoints=True,
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    bert_embed=dict(type='bert-base', hidden_size=768, pretrain_path='./bert_embed.pt'),
    data_preprocessor=dict(
        type='CustomOccDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        meta_keys=('occ_future_ann_infos', 'task_name', 'head_cfg', 'git_cfg',)  # occ head 需要的额外信息
    ),
    backbone=dict(
        type='ViTGiT',
        arch='base',
        img_size=1120,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.')
    ),
    head_list=dict(
    occupancy_prediction_head=dict(
        type='OccHead',
        
        num_classes=2,
        ignore_index=255,
        dec_length=20,
        beam_num=2
    )
)
)

# ---------------------------
# 数据预处理流水线
# ---------------------------
train_pipeline = [
    dict(
        type='LoadAnnotations3D_E2E',
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=False
    ),
    dict(
        type='LoadFrontCameraImageFromFile',
        to_float32=True,
        color_type='color',
        img_root='data/nuscenes/images'
    ),
    dict(
        type='ResizeForOccInput',
        target_size=(224, 224) 
    ),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occ_cfgs['grid_conf'],
        ignore_index=occ_cfgs['ignore_index'],
        only_vehicle=occ_cfgs['only_vehicle'],
        filter_invisible=True,
        deal_instance_255=False
    ),
    dict(
        type='AddMetaInfo',
        meta_dict=dict(
            task_name='occupancy_prediction',
            head_cfg=occ_head_cfg,
            git_cfg=occ_cfgs
        )
    ),
    dict(
        type='PackOccInputs',
        meta_keys=('occ_future_ann_infos', 'task_name', 'head_cfg', 'git_cfg')
    ),
]

test_pipeline = [
    dict(
        type='LoadAnnotations3D_E2E',
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=False
    ),
    dict(
        type='LoadFrontCameraImageFromFile',
        to_float32=True,
        color_type='color',
        img_root='data/nuscenes/images'
    ),
    dict(
        type='ResizeForOccInput',
        target_size=(224, 224)  # 这里设定宽度1568，高度896
    ),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occ_cfgs['grid_conf'],
        ignore_index=occ_cfgs['ignore_index'],
        only_vehicle=occ_cfgs['only_vehicle'],
        filter_invisible=False
    ),
    dict(
        type='AddMetaInfo',
        meta_dict=dict(
            task_name='occupancy_prediction',
            head_cfg=occ_head_cfg,
            git_cfg=occ_cfgs
        )
    ),
    dict(
        type='PackOccInputs',
        meta_keys=('occ_future_ann_infos', 'task_name', 'head_cfg', 'git_cfg')
    ),
]

# ---------------------------
# 数据集及数据加载器配置
# ---------------------------
data_root = 'data/nuscenes/'
ann_file_train = 'data/infos/nuscenes_infos_temporal_train.pkl'
ann_file_val   = 'data/infos/nuscenes_infos_temporal_val.pkl'
ann_file_test  = 'data/infos/nuscenes_infos_temporal_val.pkl'

train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesOccDataset',
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        load_interval=1,
        test_mode=False,
        queue_length=4,
        occ_n_future=occ_cfgs.get('occ_n_future', 4)
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesOccDataset',
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True,
        queue_length=4,
        occ_n_future=occ_cfgs.get('occ_n_future', 4)
    )
)

test_dataloader = val_dataloader

# ---------------------------
# 优化器、学习率调度等训练配置
# ---------------------------
max_iters = 200000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
         milestones=[10000, 15000], gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# ---------------------------
# 评估配置
# ---------------------------
val_evaluator = dict(type='OCCEvaluator', ann_file='data/coco_2014/annotations/coco_karpathy_test_gt.json')
test_evaluator = val_evaluator
