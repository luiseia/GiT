# mmdet/models/dense_heads/occ_head.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from mmengine.model import BaseModule


from mmdet.structures import SampleList


from typing import Dict, List, Tuple, Callable

from mmengine.structures import InstanceData


from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList, DataSample
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

class OccHead(nn.Module):
    def __init__(self,
                 bev_proj_dim: int = 256,
                 num_classes: int = 2,
                 ignore_index: int = 255,
                 max_length: int = 100,
                 tokenizer=None):
        """
        Args:
            bev_proj_dim (int): BEV 特征的通道数。
            num_classes (int): 占用预测任务的类别数。
            ignore_index (int): 忽略标签的值，默认 255。
            max_length (int): token 序列的最大长度。
            tokenizer: 用于 token 化坐标文本的 tokenizer 对象。
        """
        super().__init__()
        self.bev_proj_dim = bev_proj_dim
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.conv = nn.Sequential(
            nn.Conv2d(self.bev_proj_dim, self.bev_proj_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_proj_dim, self.num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> dict:
        occ_feats = x  # 如有需要，可对多尺度特征进行聚合
        occ_preds = self.conv(occ_feats)
        return {"occ_preds": occ_preds, "occ_feats": occ_feats}

    # ----------------------- 辅助函数 -----------------------
    def _extract_occupied_coordinates(self, gt_occ_seg: torch.Tensor) -> List[Tuple[int, int]]:
        """
        从单个样本的 occ segmentation 真值中提取被占用区域的像素坐标。

        Args:
            gt_occ_seg (torch.Tensor): 形状 (H, W) 或 (C, H, W) 的占用分割图，
                                       其中 1 表示占用，0 表示空闲，ignore_index 表示忽略区域。

        Returns:
            List[Tuple[int, int]]: 被占用区域的 (x, y) 坐标列表。
        """
        # 若输入 tensor 维度大于 2，则默认选择第一个通道
        if gt_occ_seg.dim() > 2:
            gt_occ_seg = gt_occ_seg[0]
        valid_mask = (gt_occ_seg == 1)
        coords_tensor = torch.nonzero(valid_mask, as_tuple=False)  # (num_coords, 2)
        coords = [(int(x), int(y)) for x, y in coords_tensor]
        return coords

    def _format_coordinates_to_text(self, coords: List[Tuple[int, int]]) -> str:
        """
        将坐标列表转换为文本字符串，格式为 "x1,y1; x2,y2; ...; xn,yn"
        """
        coord_strs = [f"{x},{y}" for x, y in coords]
        text = "; ".join(coord_strs)
        return text

    def _tokenize_occ_text(self, text: str, tokenizer) -> dict:
        """
        使用指定的 tokenizer 对文本进行 token 化。

        Args:
            text (str): 输入的坐标文本字符串。
            tokenizer: 用于 token 化的 tokenizer 对象。

        Returns:
            dict: 包含 token 化结果，如 'input_ids' 和 'attention_mask'（均为张量格式）。
        """
        tokenized = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokenized

    # ----------------------- 核心接口方法 -----------------------
    def prepare_occ_target_tokens(self, gt_occ_seg: torch.Tensor, tokenizer=None) -> dict:
        """
        整合上述步骤：从 occ segmentation 真值中提取被占用区域的坐标，
        构造格式化文本，再利用 tokenizer 得到 token 序列。

        Args:
            gt_occ_seg (torch.Tensor): 形状 (H, W) 的占用分割图。
            tokenizer: 用于 token 化的 tokenizer 对象，默认为 None，此时使用 self.tokenizer。

        Returns:
            dict: 包含 token 化结果（例如 'input_ids', 'attention_mask'）的字典。
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        coords = self._extract_occupied_coordinates(gt_occ_seg)
        text = self._format_coordinates_to_text(coords)
        tokenized = self._tokenize_occ_text(text, tokenizer)
        return tokenized

    def get_targets_based_on_reference(self,
                                         reference_preds: List[torch.Tensor],
                                         batch_gt_occ_seg: List[torch.Tensor],
                                         batch_img_metas: List[dict],
                                         tokenizer=None
                                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        针对每个样本，从 BEV 的 occ 真值中提取被占用区域的坐标，经过文本构造和 token 化，
        得到输入 tokens、目标 tokens 及 token 权重信息。同时统计正负样本数量。

        Args:
            reference_preds (List[torch.Tensor]): 来自前摄像头的参考点预测，形状为 (grid_num, 2)。
            batch_gt_occ_seg (List[torch.Tensor]): 每个样本的 occ segmentation 真值，形状 (H, W) 或 (C, H, W)。
            batch_img_metas (List[dict]): 每个样本的元信息（本例中可能不使用）。
            tokenizer: 用于 token 化的 tokenizer 对象，默认为 None，此时使用 self.tokenizer。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
                - input_tokens: (B, L) 张量，作为模型输入的 token 序列（此处示例直接复用目标 tokens）。
                - target_tokens: (B, L) 张量，目标 token 序列。
                - token_weights: (B, L) 张量，对应 token 的有效性权重。
                - num_total_pos: 正样本总数（有效 token 数）。
                - num_total_neg: 负样本总数（PAD token 数）。
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        input_tokens_list = []
        target_tokens_list = []
        token_weights_list = []
        num_total_pos = 0
        num_total_neg = 0

        batch_size = len(batch_gt_occ_seg)
        for i in range(batch_size):
            gt_occ = batch_gt_occ_seg[i].to(reference_preds[i].device)
            tokenized = self.prepare_occ_target_tokens(gt_occ, tokenizer)
            target_tokens = tokenized['input_ids'].squeeze(0)            # (L,)
            token_weights = tokenized['attention_mask'].squeeze(0).float()   # (L,)

            target_tokens_list.append(target_tokens)
            token_weights_list.append(token_weights)

            num_total_pos += int(token_weights.sum().item())
            num_total_neg += int((token_weights == 0).sum().item())

            # 此处示例直接复用目标 tokens 作为输入 tokens
            input_tokens_list.append(target_tokens)

        input_tokens = torch.stack(input_tokens_list, dim=0)    # (B, L)
        target_tokens = torch.stack(target_tokens_list, dim=0)    # (B, L)
        token_weights = torch.stack(token_weights_list, dim=0)    # (B, L)

        return input_tokens, target_tokens, token_weights, num_total_pos, num_total_neg
