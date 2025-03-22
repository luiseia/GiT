import os
import os.path as osp
import random

from mmengine.registry import DATASETS
from .base_det_dataset import BaseDetDataset

@DATASETS.register_module()
class KittiDataset(BaseDetDataset):
    """A minimal KITTI 2D detection dataset, built upon BaseDetDataset.

    1) 只关注 3 类: Car, Pedestrian, Cyclist.
    2) `ann_file` 指向存放txt标注的文件夹（label_2）。
    3) `data_prefix` 中的 `img` 字段指向图像文件夹（image_2）。
    4) 每个 txt 文件格式: class, trunc, occ, alpha, x1, y1, x2, y2, ...
    5) 在验证 / 测试模式下（`test_mode=True`），仅随机抽取 50 张图像，用于快速验证 / 测试。
    """

    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist')
    }

    def load_data_list(self):
        """从 `self.ann_file` 文件夹读取标注，再与 `data_prefix['img']` 对应图像配对。"""
        data_list = []

        # 1) 计算标注文件夹的绝对路径
        label_dir = self.ann_file  # e.g. data/kitti/training/label_2
        # 2) 计算图像文件夹的绝对路径
        img_dir = self.data_prefix['img']  # e.g. data/kitti/training/image_2
        print('label_dir', self.data_root, label_dir, img_dir)

        # 获取所有 .txt 标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        label_files.sort()

        for label_file in label_files:
            stem = osp.splitext(label_file)[0]  # e.g. "000000"
            label_path = osp.join(label_dir, label_file)

            # 读取标注
            bboxes = []
            labels = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                splits = line.strip().split()
                if len(splits) < 8:
                    continue
                cls_str = splits[0]
                x1, y1, x2, y2 = map(float, splits[4:8])
                if cls_str not in self.METAINFO['classes']:
                    continue
                cls_idx = self.METAINFO['classes'].index(cls_str)
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_idx)

            # 生成图像信息
            img_file = osp.join(img_dir, stem + '.png')  # KITTI中的图像扩展名通常是 .png
            data_info = {
                'img_id': stem,       # or int(stem) if你想转换为数值
                'img_path': img_file,
                'height': 375,        # KITTI 默认高度，可根据实际情况修改
                'width': 1242,        # 同上
                'instances': []
            }

            # 生成 instance 列表
            for bbox, label in zip(bboxes, labels):
                inst = {
                    'bbox': bbox,         # [x1, y1, x2, y2]
                    'bbox_label': label,  # 类别索引
                    'ignore_flag': 0      # KITTI中通常没有ignore信息，就设为0
                }
                data_info['instances'].append(inst)

            data_list.append(data_info)

        # 写死：若处于验证/测试模式，只随机取 50 张
        if self.test_mode:
            max_val_imgs = 50  # 固定写死
            if len(data_list) > max_val_imgs:
                data_list = random.sample(data_list, max_val_imgs)

        return data_list
