# /home/UNT/yz0370/projects/GiT/configs/GiT/single_semanticseg_base_nuScenes.py
_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

# 根据nuScenes图像尺寸调整
base_img_size = 896  # 原始1120可能过大，调整为适合nuScenes的尺寸

# 语义分割配置（nuScenes专用）
semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_vocal=16+1,  # nuScenes的16个语义类别+1
    max_decoder_length=16,
    global_only_image=True)

model = dict(
    type='GiT',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 'caption', 'grounding'],
    use_checkpoints=True,
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    bert_embed=dict(type='bert-base', hidden_size=768, pretrain_path='./bert_embed.pt'),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=0,  # nuScenes通常使用0作为忽略值
        pad_size_divisor=224),
    backbone=dict(
        type='ViTGiT',
        arch='base',
        img_size=base_img_size,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.')),
    head_list=dict(
        semantic_segmentation_head=dict(type='GiTSemSegHead')))

# 数据增强流水线
semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations', reduce_zero_label=False),  # nuScenes不需要减1
    dict(type='AddMetaInfo', meta_dict=dict(
        task_name='semantic_segmentation',
        head_cfg=dict(
            num_classes=16,  # 实际语义类别数
            num_vocal=17,    # 16+1
            dec_length=16,
            dec_pixel_resolution=[4, 4],
            arg_max_inference=True,
            ignore_index=0),  # 根据实际忽略标签调整
        git_cfg=semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672*x*0.1), int(672*x*0.1)) for x in range(10,21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.9),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    dict(type='SegLoadAnnotations', reduce_zero_label=False),
    dict(type='AddMetaInfo', meta_dict=dict(
        task_name='semantic_segmentation',
        head_cfg=dict(
            num_classes=16,
            num_vocal=17,
            dec_length=16,
            dec_pixel_resolution=[4, 4],
            ignore_index=0),
        git_cfg=semseg_cfgs)),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

# 数据加载配置
# 在single_semanticseg_base_nuScenes.py中找到以下部分并修改：

train_dataloader = dict(
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='NuScenesDataset',
                data_root='/home/UNT/yz0370/projects/GiT/data/nuscenes',  # ← 改为绝对路径
                data_prefix=dict(
                    img_path='samples/CAM_FRONT',
                    seg_map_path='segmentation/CAM_FRONT'),
                ann_file='annotations/train.json',
                pipeline=semseg_train_pipeline)
        ]))

val_dataloader = dict(
    dataset=dict(
        type='NuScenesDataset',
        data_root='/home/UNT/yz0370/projects/GiT/data/nuscenes',  # ← 改为绝对路径
        data_prefix=dict(
            img_path='samples/CAM_FRONT',
            seg_map_path='segmentation/CAM_FRONT'),
        ann_file='annotations/val.json',
        test_mode=True,
        pipeline=semseg_test_pipeline))
test_dataloader = val_dataloader

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),  # 调整初始学习率
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
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# 评估配置
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    ignore_index=0)  # 忽略背景类
test_evaluator = val_evaluator

# 训练调度
max_iters = 120000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
param_scheduler = [dict(
    type='CosineAnnealingLR',
    T_max=max_iters,
    eta_min=2e-6,
    begin=0,
    end=max_iters,
    by_epoch=False)]

# 系统配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
auto_scale_lr = dict(base_batch_size=16)  # 根据实际batch_size调整

custom_imports = dict(
    imports=['mmdet.datasets.nuscenes_occ'],
    allow_failed_imports=False)
