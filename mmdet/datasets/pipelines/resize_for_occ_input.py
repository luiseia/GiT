# /home/UNT/yz0370/projects/GiT/mmdet/datasets/pipelines/resize_for_occ_input.py
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ResizeForOccInput(BaseTransform):
    """将输入图像调整为目标尺寸以满足 occupancy prediction 的需求。

    本模块主要用于将加载后的原始图像（例如 1600×900）调整为一个目标尺寸，
    例如 (1568, 896)，使得经过 backbone 得到的 patch embedding（如 70×70）
    在划分成更细网格时（例如 window_size=14）能保证每个窗口尺寸为整数。

    Args:
        target_size (tuple[int, int]): 目标输出尺寸，格式为 (width, height)。
            例如： (1568, 896)。
    """
    def __init__(self, target_size: tuple):
        self.target_size = target_size

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            # 使用 mmcv.imresize 调整图像大小
            img = results['img']
            resized_img = mmcv.imresize(img, self.target_size, interpolation='bilinear')
            results['img'] = resized_img
            # 更新图像尺寸信息，如果存在的话
            results['img_shape'] = resized_img.shape[:2]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_size={self.target_size})"
