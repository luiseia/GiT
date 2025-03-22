#!/usr/bin/env python
"""
测试整个流水线模块，验证各模块的 transform() 方法是否按预期工作。
该流水线包括：
  - LoadAnnotations3D_E2E
  - GenerateOccFlowLabels
  - AddMetaInfo
  - PackSegInputs
请根据实际情况修改各模块参数。
"""

import torch
import mmcv
from mmengine.registry import TRANSFORMS

# 定义一个简单的 Compose，用于依次调用各模块的 transform() 方法
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, data):
        for t in self.transforms:
            data = t.transform(data)
        return data

    def __call__(self, data):
        return self.transform(data)

# 导入各个流水线模块
from mmdet.datasets.pipelines.load_annotations_3d_e2e import LoadAnnotations3D_E2E
from mmdet.datasets.pipelines.generate_occ_flow_labels import GenerateOccFlowLabels
from mmdet.datasets.pipelines.add_meta_info import AddMetaInfo
from mmdet.datasets.pipelines.pack_occ_inputs import PackSegInputs

def main():
    # 构造一个模拟的输入字典，模拟 NuScenesOccDataset.get_data_info() 的输出
    data = {
        'sample_idx': 'dummy_token',
        'scene_token': 'dummy_scene',
        'frame_idx': 0,
        'timestamp': 123456789,
        'sweeps': None,
        # 模拟 occ_future_ann_infos 为一个包含 5 个元素的列表（每个元素这里用一个 dummy 字典）
        'occ_future_ann_infos': [{
            'gt_bboxes_3d': None,
            'gt_names_3d': None,
            'gt_inds': None,
            'visibility_tokens': None,
        }] * 5,
        'lidar2ego_rotation': None,
        'lidar2ego_translation': None,
        'ego2global_rotation': None,
        'ego2global_translation': None,
    }

    # 构造流水线各模块的对象
    load_ann = LoadAnnotations3D_E2E(
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=False
    )
    occ_generator = GenerateOccFlowLabels(
        grid_conf={'xbound': [-50, 50, 0.5],
                   'ybound': [-50, 50, 0.5],
                   'zbound': [-5, 3, 8]},
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=True,
        deal_instance_255=False
    )
    add_meta = AddMetaInfo(
        meta_dict={
            'task_name': 'occupancy_prediction',
            'head_cfg': {'grid_resolution_perwin': (14, 14)},
            'git_cfg': {'ignore_index': 255}
        }
    )
    pack_inputs = PackSegInputs(
        meta_keys=('occ_future_ann_infos', 'task_name', 'head_cfg', 'git_cfg')
    )

    # 将所有流水线模块组装到 Compose 中
    pipeline = Compose([
        load_ann,
        occ_generator,
        add_meta,
        pack_inputs
    ])

    # 调用流水线对模拟数据进行处理
    output = pipeline(data)

    # 输出最终结果的关键信息
    print("Final output keys:", list(output.keys()))
    if 'inputs' in output:
        print("inputs shape:", output['inputs'].shape)
    if 'data_samples' in output:
        print("data_samples:", output['data_samples'])
    if 'gt_segmentation' in output:
        print("gt_segmentation shape:", output['gt_segmentation'].shape)

if __name__ == '__main__':
    main()
