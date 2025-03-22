#!/usr/bin/env python
"""
测试 NuScenesOccDataset 流水线起点

此脚本从配置文件中加载数据集部分配置，删除不需要的 "type" 键后构造 NuScenesOccDataset 实例，
并打印第一个样本的关键信息以验证数据加载和预处理流程是否正确。
"""

import os
from mmengine import Config
from mmdet.datasets.nuscenes_occ import NuScenesOccDataset

def main():
    # 加载配置文件（确保项目根目录在 PYTHONPATH 中）
    cfg = Config.fromfile("configs/GiT/single_occupancy_base.py")
    
    # 从 train_dataloader 中提取数据集配置
    dataset_cfg = cfg.train_dataloader['dataset']
    print(f"[DEBUG] Using dataset config:\n{dataset_cfg}\n")
    
    # 删除 'type' 键，因为它仅用于注册构造时使用
    dataset_cfg.pop("type", None)
    
    # 构造数据集实例
    dataset = NuScenesOccDataset(**dataset_cfg)
    
    # 获取第一个样本（调用 __getitem__ 内部会调用 prepare_train_data/get_data_info）
    sample = dataset[0]
    
    # 输出样本中的键及部分内容，观察 debug 信息
    print("\nFinal sample keys:", list(sample.keys()))
    if "inputs" in sample:
        print("inputs tensor shape:", sample["inputs"].shape)
    else:
        print("No 'inputs' key found in sample!")
    
    if "data_samples" in sample:
        print("data_samples type:", type(sample["data_samples"]))
        # 如果 data_samples 是列表，直接打印或迭代列表的每个元素
        if isinstance(sample["data_samples"], dict):
            print("data_samples content:")
            for key, value in sample["data_samples"].items():
                print(f"  {key}: {value}")
        elif isinstance(sample["data_samples"], list):
            print("data_samples content (list):")
            for i, item in enumerate(sample["data_samples"]):
                print(f"  item {i}: {item}")
        else:
            print("data_samples content:", sample["data_samples"])
    else:
        print("No 'data_samples' key found in sample!")


if __name__ == "__main__":
    main()
