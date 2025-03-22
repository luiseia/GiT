#!/usr/bin/env python
"""
测试 GenerateOccFlowLabels 流水线模块
"""

# 如果无法从 mmengine.transforms 导入 Compose，则定义一个简单的版本
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

# 接下来导入 GenerateOccFlowLabels 模块
from mmdet.datasets.pipelines.generate_occ_flow_labels import GenerateOccFlowLabels
import torch

def main():
    # 构造一个模拟的输入字典，模拟从数据集中 get_data_info 得到的结果
    data = {
        'sample_idx': 'dummy_token',
        'scene_token': 'dummy_scene',
        'frame_idx': 0,
        'timestamp': 123456789,
        'sweeps': None,
        # 构造 occ_future_ann_infos 为一个列表，包含 5 个元素（这里用 None 模拟）
        'occ_future_ann_infos': [None] * 5,
        'lidar2ego_rotation': None,
        'lidar2ego_translation': None,
        'ego2global_rotation': None,
        'ego2global_translation': None,
        # 此处不包含 'img' 键，测试 PackSegInputs 是否能生成 dummy tensor
    }

    # 定义流水线，仅包含 GenerateOccFlowLabels 模块
    pipeline = Compose([
        {
            'type': 'GenerateOccFlowLabels',
            'grid_conf': {'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5], 'zbound': [-5, 3, 8]},
            'ignore_index': 255,
            'only_vehicle': True,
            'filter_invisible': True,
            'deal_instance_255': False
        }
    ])

    # mmengine 的 Registry 系统要求构建对象时通过 TRANSFORMS.build(cfg)；为了简化测试，我们手动构造对象：
    occ_generator = GenerateOccFlowLabels(
        grid_conf={'xbound': [-50, 50, 0.5], 'ybound': [-50, 50, 0.5], 'zbound': [-5, 3, 8]},
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=True,
        deal_instance_255=False
    )

    # 模拟流水线处理：先调用 GenerateOccFlowLabels 模块
    out = occ_generator(data)
    # 如果你希望在流水线中还打包数据，则可以先调用 PackSegInputs 模块，
    # 但此处我们单独测试 GenerateOccFlowLabels

    print("Final keys in output:", list(out.keys()))
    if 'gt_segmentation' in out:
        print("gt_segmentation shape:", out['gt_segmentation'].shape)
    else:
        print("gt_segmentation not found in output.")

if __name__ == '__main__':
    main()
