# _base_/occ_default_runtime.py

default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
# 对于 occupancy 任务，你可以选择合适的 visualizer
visualizer = dict(
    type='DetLocalVisualizer',  # 或者自定义一个 OccupancyVisualizer
    vis_backends=vis_backends,
    name='visualizer'
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# 如果 occupancy 任务不需要 TTA 模型，也可以不定义或定义一个空的 tta_model
tta_model = dict(type='OccTTAModel')  # 如果你有专门的 occupancy TTA 模块，否则可以去掉
