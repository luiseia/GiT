#!/usr/bin/env python
"""
测试 PackSegInputs 流水线步骤

该脚本构造一个测试字典，并测试 PackSegInputs 的功能，
分别测试带有 'img_path'、带有 'token' 情况以及不带任何图像数据的情况。
"""

import os
import json
import torch
import mmcv
import numpy as np
import tempfile
from mmdet.datasets.pipelines.pack_occ_inputs import PackSegInputs


def test_with_img_path():
    # 模拟一个输入字典，其中包含图像路径（请修改为你实际存在的图像路径）
    data = {
        'sample_idx': 'dummy_token',
        'scene_token': 'dummy_scene',
        'frame_idx': 0,
        'timestamp': 123456789,
        'sweeps': None,
        'occ_future_ann_infos': [
            {
                'gt_bboxes_3d': None,
                'gt_names_3d': None,
                'gt_inds': None,
                'visibility_tokens': None
            }
        ] * 5,
        'lidar2ego_rotation': None,
        'lidar2ego_translation': None,
        'ego2global_rotation': None,
        'ego2global_translation': None,
        # 这里加入图像路径（请确保该路径存在，否则会报错或走到 dummy tensor 分支）
        'img_path': '/home/UNT/yz0370/projects/GiT/data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg'
    }
    print("=== Test with 'img_path' ===")
    packer = PackSegInputs(meta_keys=('occ_future_ann_infos',))
    out = packer(data)
    print("Final keys:", list(out.keys()))
    print("inputs tensor shape:", out.get('inputs').shape)
    print("data_samples:", out.get('data_samples'))
    print()


def test_without_img_path():
    # 模拟一个输入字典，不包含 'img' 或 'img_path'
    data = {
        'sample_idx': 'dummy_token',
        'scene_token': 'dummy_scene',
        'frame_idx': 0,
        'timestamp': 123456789,
        'sweeps': None,
        'occ_future_ann_infos': [
            {
                'gt_bboxes_3d': None,
                'gt_names_3d': None,
                'gt_inds': None,
                'visibility_tokens': None
            }
        ] * 5,
        'lidar2ego_rotation': None,
        'lidar2ego_translation': None,
        'ego2global_rotation': None,
        'ego2global_translation': None,
    }
    print("=== Test without 'img_path' or 'img' ===")
    packer = PackSegInputs(meta_keys=('occ_future_ann_infos',))
    out = packer(data)
    print("Final keys:", list(out.keys()))
    print("inputs tensor shape:", out.get('inputs').shape)
    print("data_samples:", out.get('data_samples'))
    print()


def test_with_token():
    # 测试通过 token 索引加载图像的情况
    print("=== Test with 'token' ===")
    # 创建一个临时 dummy 图像（尺寸为 224x224，3 通道）
    dummy_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    # 使用临时文件保存图像
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_file:
        dummy_img_path = img_file.name
    mmcv.imwrite(dummy_img, dummy_img_path)

    # 构造临时 token 索引文件，其中包含 token 对应的文件路径
    token_data = [
        {
            "token": "dummy_token",
            "filename": dummy_img_path  # 此处使用绝对路径
        }
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as token_file:
        json.dump(token_data, token_file)
        token_file_path = token_file.name

    data = {
        'token': 'dummy_token',
        'sample_idx': 'dummy_token',
        'scene_token': 'dummy_scene',
        'frame_idx': 0,
        'timestamp': 123456789,
        'sweeps': None,
        'occ_future_ann_infos': [
            {
                'gt_bboxes_3d': None,
                'gt_names_3d': None,
                'gt_inds': None,
                'visibility_tokens': None
            }
        ] * 5,
        'lidar2ego_rotation': None,
        'lidar2ego_translation': None,
        'ego2global_rotation': None,
        'ego2global_translation': None,
    }
    # 初始化 PackSegInputs 时传入 token_file_path 和 data_root（此处不需要 data_root，因为 filename 为绝对路径）
    packer = PackSegInputs(
        meta_keys=('occ_future_ann_infos',),
        token_file_path=token_file_path,
        data_root=None
    )
    out = packer(data)
    print("Final keys:", list(out.keys()))
    print("inputs tensor shape:", out.get('inputs').shape)
    print("data_samples:", out.get('data_samples'))

    # 清理临时文件
    os.remove(token_file_path)
    os.remove(dummy_img_path)
    print()


def main():
    test_with_img_path()
    test_without_img_path()
    test_with_token()


if __name__ == "__main__":
    main()
