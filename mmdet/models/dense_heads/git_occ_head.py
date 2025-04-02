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

@MODELS.register_module()
class OccHead(BaseModule):
    r"""OCC head for BEV occupancy prediction based on bbox representation.
    
    This head predicts occupancy bboxes in BEV space (e.g., 200×200 grid) using a
    token-based autoregressive decoding approach. The target token sequence for each
    occupancy bbox includes:
        [class_token, cx_token, cy_token, w_token, h_token, theta_token]
    where theta represents the rotation angle.
    
    The ground truth bboxes are provided in BEV space.
    """
    def __init__(self,
                 num_classes: int,
                 ignore_index: int,
                 dec_length: int,
                 beam_num: int,
                 num_bins: int,
                 num_vocal: int,
                 num_queries: int,
                 bev_shape: Tuple[int, int],
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 train_cfg: ConfigType = None,
                 test_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        """
        OCC head for BEV occupancy prediction based on bbox representation.
        Args:
            num_classes (int): Number of classes.
            ignore_index (int): Ignore index for loss.
            dec_length (int): Number of decoding steps.
            beam_num (int): Beam search number.
            num_bins (int): Number of bins for discretization.
            num_vocal (int): Vocabulary size (num_bins + 1).
            num_queries (int): Number of candidate grid queries.
            bev_shape (tuple[int,int]): BEV occupancy map size, e.g., (200, 200).
            d_model (int): Token feature dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            train_cfg (ConfigType, optional): Training config.
            test_cfg (ConfigType, optional): Testing config.
            init_cfg (OptMultiConfig, optional): Initialization config.
        """
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dec_length = dec_length
        self.beam_num = beam_num
        self.num_bins = num_bins
        self.num_vocal = num_vocal
        self.num_queries = num_queries
        self.bev_shape = bev_shape
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # 若有训练配置，则构建匹配器
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner must be provided in train_cfg.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('OCC head does not use sampler.')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 初始化 OCC head 所需的各个子模块
        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize transformer modules and BEV mapping layers.

        In this OCC head implementation, we initialize:
        - A transformer encoder/decoder block to serve as the backbone for token decoding.
        - A BEV mapping layer that projects input tokens (from multi-camera features) into BEV space.
        - Optionally, a positional embedding for BEV tokens.

        These modules are placeholders and can be further adjusted based on experimental needs.
        """
        # Feature dimension for tokens
        d_model = 256  
        nhead = 8      # Number of attention heads
        num_layers = 6 # Number of transformer layers

        # Initialize a transformer encoder layer.
        # Note: In a complete design, you might use a decoder or a combination of encoder-decoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # BEV mapping layer: maps input token features to BEV space features.
        # This is a simple linear projection which can be replaced by a more complex module if needed.
        self.bev_mapping = nn.Linear(d_model, d_model)

        # Optional: learnable positional embedding for BEV tokens.
        # Assume BEV occupancy map is 200x200, so total positions=200*200.
        self.bev_pos_embed = nn.Parameter(torch.zeros(1, 200 * 200, d_model))

        # Initialize weights for the BEV mapping layer and positional embeddings.
        nn.init.xavier_uniform_(self.bev_mapping.weight)
        nn.init.zeros_(self.bev_mapping.bias)
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

    def init_weights(self) -> None:
        """Initialize weights for OCC head layers.
        
        This method initializes:
        - The transformer encoder layers using Xavier uniform initialization.
        - The BEV mapping layer using Xavier uniform for weights and zeros for bias.
        - The BEV positional embedding using truncated normal initialization.
        """
        # Initialize transformer parameters.
        for name, param in self.transformer.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                # For biases or 1D parameters, use zeros initialization.
                nn.init.zeros_(param)
        
        # Initialize BEV mapping layer weights.
        nn.init.xavier_uniform_(self.bev_mapping.weight)
        if self.bev_mapping.bias is not None:
            nn.init.zeros_(self.bev_mapping.bias)
        
        # Initialize BEV positional embedding.
        # Use truncated normal initialization with standard deviation 0.02.
        try:
            nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)
        except AttributeError:
            # If trunc_normal_ is not available, fall back to normal initialization.
            nn.init.normal_(self.bev_pos_embed, std=0.02)


    def reset_hyparameter(self, cfgs: dict):
        """Reset hyper-parameters dynamically.
        
        Besides common hyper-parameters, ensure to set up additional parameters for OCC,
        e.g., number of bins for theta (rotation) discretization.
        """
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        # Loss function using cross entropy over discretized tokens.
        self.loss_reg = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.num_vocal)
    
    def get_targets_based_on_reference(self,
                             bev_bbox_gt_list: List[Tensor],
                             batch_gt_instances: InstanceList,
                             batch_img_metas: List[dict]) -> tuple:
        """
        Compute regression and classification targets for a batch of images in BEV space.
        
        Args:
            bev_bbox_gt_list (list[Tensor]): Each element is a tensor of BEV occupancy bboxes,
                with shape (num_bboxes, 5) where each bbox is represented as [cx, cy, w, h, theta],
                and all values are normalized to [0, 1] relative to the BEV map (e.g., 200×200).
            batch_gt_instances (list[InstanceData]): Ground truth instances for each image,
                including fields such as 'labels'.
            batch_img_metas (list[dict]): Meta information for each image, which should include
                the BEV map size (e.g., under key 'bev_shape').
        
        Returns:
            tuple: A tuple containing:
                - input_tokens_tensor (Tensor): Input tokens for autoregressive training,
                shape (bs, num_queries, 5).
                - targets_tokens_tensor (Tensor): Full target token sequence,
                shape (bs, num_queries, 6) (with token order: [class, cx, cy, w, h, theta]).
                - tokens_weights_tensor (Tensor): Token weights, shape (bs, num_queries, 6).
                - num_total_pos (int): Total number of positive samples across images.
                - num_total_neg (int): Total number of negative samples across images.
        """
        (input_tokens_list, targets_tokens_list, tokens_weights_list,
        pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single_based_on_bev,
            bev_bbox_gt_list,
            batch_gt_instances,
            batch_img_metas
        )
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        # 假设所有图像的查询数一致，因此可以直接 stack
        input_tokens_tensor = torch.stack(input_tokens_list)
        targets_tokens_tensor = torch.stack(targets_tokens_list)
        tokens_weights_tensor = torch.stack(tokens_weights_list)
         
        return (input_tokens_tensor, targets_tokens_tensor, tokens_weights_tensor, num_total_pos, num_total_neg)


    def _get_targets_single_based_on_bev(self, bev_bbox_gt: Tensor,
                                     gt_instances: InstanceData,
                                     img_meta: dict) -> tuple:
        """
        Generate targets for one image in BEV space.
        
        Args:
            bev_bbox_gt (Tensor): Ground truth occupancy bboxes in BEV space,
                shape (num_bboxes, 5), each represented as [cx, cy, w, h, theta],
                with values normalized to [0, 1].
            gt_instances (InstanceData): Ground truth instance data that contains
                fields such as 'labels' (shape: [num_bboxes]).
            img_meta (dict): Meta info for the BEV map, should contain key 'bev_shape'
                as (H, W), e.g., (200, 200).

        Returns:
            tuple: (input_tokens, targets_tokens, token_weights, pos_inds, neg_inds)
                - input_tokens: Tensor, shape (num_candidates, 5), used as input for autoregressive training.
                - targets_tokens: Tensor, shape (num_candidates, 6), full target token sequence.
                - token_weights: Tensor, shape (num_candidates, 6), weights for each token.
                - pos_inds: Tensor, indices of positive samples.
                - neg_inds: Tensor, indices of negative samples.
        """
        # 假设 BEV map 大小，例如 (200, 200)
        bev_H, bev_W =(200, 200) # 如 (200, 200)
        
        # 模拟生成候选的 grid 位置，这里假设已有 self.num_queries 个候选位置，
        # 每个位置在 BEV 空间归一化坐标下 (cx, cy) ∈ [0,1]
        # 实际中候选位置可以来自于特定设计，例如均匀采样或学习得到的 grid.

        ## generate sampled grids
        grid_H_win, grid_W_win = 1,1
        window_size = 14
        patch_H, patch_W =  14, 14# patch resolution of the whole bev
        assert patch_H % window_size == 0 and patch_W % window_size == 0, "padding inner \
        window is not implemented, patch scale must be a multiple of window size"
        win_H = patch_H // window_size
        win_W = patch_W // window_size
        grid_H, grid_W = grid_H_win * win_H, grid_W_win * win_W

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, grid_H - 1, grid_H, dtype=torch.float32, device=current_device),
            torch.linspace(0, grid_W - 1, grid_W, dtype=torch.float32, device=current_device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid_int_position = grid.clone()
        batch_size = 3
        # normalize grids and construct start task identifier tokens
        grid_scale = grid.new_zeros((batch_size, 1, 1, 2))
        grid_scale[:, :, :, 0] = grid_W
        grid_scale[:, :, :, 1] = grid_H
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / grid_scale
        bev_grid_reference = grid.view(batch_size, -1, 2).detach() # bs, grid_num, 2
        import math

        # 假设 self.num_queries 是一个完全平方数
        n = int(math.sqrt(self.num_queries))
        assert n * n == self.num_queries, "num_queries 应该是完全平方数以便均匀采样"
        # 使用 linspace 在 [0,1] 范围内均匀生成 n 个点
        x = torch.linspace(0, 1, n, device=bev_bbox_gt.device)
        y = torch.linspace(0, 1, n, device=bev_bbox_gt.device)
        # 生成网格
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        # 将网格点堆叠成 (num_queries, 2) 的张量
        reference_pred = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        num_candidates = reference_pred.size(0)

        
        # 构造用于匹配的候选实例，assigner 会基于这些候选点与 ground truth 匹配
        reference_pred_instances = InstanceData(points=reference_pred)
        
        # 为了匹配，将 ground truth bbox 只取前 4 个参数 ([cx, cy, w, h])，忽略 theta
        gt_bboxes_for_matching = bev_bbox_gt[:, :4]
        # 更新 gt_instances 中的 bboxes字段，使得 assigner 接收到的 bbox 形状为 (N, 4)
        gt_instances.bboxes = gt_bboxes_for_matching
        # 调用 assigner 进行正负样本匹配
        assign_result = self.assigner.assign(
            pred_instances=reference_pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        
        # 正样本: assign_result.gt_inds > 0, 负样本: assign_result.gt_inds == 0
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        
        # 对正样本，根据 assign_result 得到对应的 ground truth 索引（减 1）
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        # 提取匹配到的 ground truth bboxes，形状为 (num_pos, 5)
        pos_gt_bboxes = bev_bbox_gt[pos_assigned_gt_inds.long(), :]
        
        # 构造目标 token序列，序列长度为6: [class_token, cx, cy, w, h, theta]
        # 初始化为 ignore token，即 self.num_vocal - 1
        targets_tokens = bev_bbox_gt.new_full((num_candidates, 6), self.num_vocal - 1, dtype=torch.long)
        
        # 假设 gt_instances.labels 包含正样本对应的类别标签，形状 (num_bboxes,)
        gt_labels = gt_instances.labels  # 类型应为长整型
        
        # 初始化所有候选的类别，默认设置为背景类别（或 ignore 标记）
        labels = bev_bbox_gt.new_full((num_candidates,), self.num_classes, dtype=torch.long)
        # 对正样本，设置真实的类别标签
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        
        # 类别 token：将标签加上一个偏移量（与 detection head 类似），例如 self.num_bins + 1
        class_tokens = labels + self.num_bins + 1
        
        # 对于正样本，将 BEV bbox 的参数离散化：
        # 这里假设所有参数均归一化到 [0, 1]，乘以 self.num_bins 后四舍五入，并裁剪到 [0, self.num_bins]
        pos_tokens = (pos_gt_bboxes * self.num_bins).round().long().clamp(min=0, max=self.num_bins)
        
        # 填入 token序列，第一个 token 为类别，后续 5 个 token为 bbox 参数（cx, cy, w, h, theta）
        targets_tokens[:, 0] = class_tokens
        targets_tokens[pos_inds, 1:] = pos_tokens
        
        # 构造 token 权重，对于正样本权重为 1，其余为 0
        token_weights = bev_bbox_gt.new_ones((num_candidates, 6))
        token_weights[neg_inds] = 0
        
        # 自回归训练时，输入 token 通常是目标 token 序列去掉最后一个 token
        input_tokens = targets_tokens[:, :-1]  # shape (num_candidates, 5)
        
        return input_tokens, targets_tokens, token_weights, pos_inds, neg_inds


    def loss(self, 
             all_layer_pred_seq_logits: Tensor,
             all_layer_target_tokens: List[Tensor],
             all_layer_token_weights: List[Tensor],
             num_total_pos: List[int], 
             num_total_neg: List[int],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute overall loss for OCC head.
        
        This method organizes the batch data and delegates to loss_by_feat.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        
        loss_inputs = (all_layer_pred_seq_logits,
                       all_layer_target_tokens,
                       all_layer_token_weights,
                       num_total_pos,
                       num_total_neg,
                       batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                    all_layer_pred_seq_logits: Tensor,
                    all_layer_target_tokens: List[Tensor],
                    all_layer_token_weights: List[Tensor],
                    num_total_pos: List[int],
                    num_total_neg: List[int],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """
        Loss function for outputs from each decoder layer for OCC head.
        
        For OCC head, the target token sequence is of length 6:
            [class_token, cx_token, cy_token, w_token, h_token, theta_token]
        Only the outputs from the last decoder layer are used as the primary loss,
        while losses from intermediate layers are computed as auxiliary losses.
        
        Args:
            all_layer_pred_seq_logits (Tensor): Predicted logits from the autoregressive head,
                with shape (num_decoder_layers, bs, num_queries, max_token_len, vocab_size).
            all_layer_target_tokens (List[Tensor]): Ground truth token sequences for each decoder layer,
                with shape (num_decoder_layers, bs, num_queries, max_token_len).
            all_layer_token_weights (List[Tensor]): Weights for each token,
                with shape (num_decoder_layers, bs, num_queries, max_token_len).
            num_total_pos (List[int]): Total number of positive samples across images.
            num_total_neg (List[int]): Total number of negative samples across images.
            batch_gt_instances (InstanceList): Ground truth instances for the batch.
            batch_img_metas (List[dict]): Meta information for each image.
            batch_gt_instances_ignore (OptInstanceList): Currently must be None.
        
        Returns:
            Dict[str, Tensor]: A dictionary of loss components, including:
                - 'loss_cls': loss from the last decoder layer for classification.
                - 'loss_reg': loss from the last decoder layer for regression.
                - Additional auxiliary losses from intermediate layers, e.g., 'd0.loss_cls', 'd0.loss_reg', etc.
        """
        # OCC head does not support ignoring any ground truth instances in this setting.
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports batch_gt_instances_ignore set to None.'

        # 利用 multi_apply 调用 loss_by_feat_single 对每个 decoder 层分别计算损失，
        # 返回的 losses_cls 和 losses_reg 均为列表，每个元素对应一个 decoder 层的损失结果。
        losses_cls, losses_reg = multi_apply(
            self.loss_by_feat_single,
            all_layer_pred_seq_logits,
            all_layer_target_tokens,
            all_layer_token_weights,
            num_total_pos,
            num_total_neg,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas
        )

        loss_dict = dict()
        # 以最后一层的损失作为最终输出损失
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_reg'] = losses_reg[-1]

        # 同时记录其他层的辅助损失，便于训练时监控
        num_dec_layer = 0
        for loss_cls_i, loss_reg_i in zip(losses_cls[:-1], losses_reg[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
            loss_dict[f'd{num_dec_layer}.loss_reg'] = loss_reg_i[0]
            num_dec_layer += 1

        return loss_dict



    def loss_by_feat_single(self, 
                         pred_seq_logits: Tensor, 
                         targets_tokens_tensor: Tensor,
                         tokens_weights_tensor: Tensor,
                         num_total_pos: int, 
                         num_total_neg: int,
                         batch_gt_instances: InstanceList,
                         batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """
        Compute loss for outputs from a single decoder layer for OCC head.
        
        For OCC head, the target token sequence is of length 6:
            [class_token, cx_token, cy_token, w_token, h_token, theta_token]
        where:
        - The classification token is at index 0, with vocabulary indices offset by (self.num_bins + 1).
        - The regression tokens are at indices 1 to 5.
        
        Args:
            pred_seq_logits (Tensor): Predicted logits from the autoregressive head,
                shape (bs, num_queries, max_token_len, vocab_size). Here max_token_len = 6.
            targets_tokens_tensor (Tensor): Ground truth token sequences,
                shape (bs, num_queries, 6).
            tokens_weights_tensor (Tensor): Weights for each token, shape (bs, num_queries, 6).
            num_total_pos (int): Total number of positive samples.
            num_total_neg (int): Total number of negative samples.
            batch_gt_instances (InstanceList): Ground truth instances.
            batch_img_metas (list[dict]): Image meta information.
        
        Returns:
            Tuple[Tensor]: A tuple containing loss for classification and regression.
        """
        num_imgs, num_queries = pred_seq_logits.shape[:2]
        
        # Split classification and regression logits.
        # Classification logits: use token at position 0. For class prediction,
        # the effective vocabulary starts from index (self.num_bins + 1).
        pred_seq_cls_logits = pred_seq_logits[:, :, 0, self.num_bins + 1:].reshape(-1, self.num_classes + 1)
        
        # Regression logits: tokens from positions 1 to 5 (for cx, cy, w, h, theta).
        pred_seq_reg_logits = pred_seq_logits[:, :, 1:, :self.num_bins + 1].reshape(-1, self.num_bins + 1)
        
        # Construct a weighted average factor:
        # 每个查询都用于分类损失，只有正样本需要回归损失（每个正样本对应 5 个回归 token）。
        avg_factor = num_imgs * num_queries + num_total_pos * (5)
        avg_factor = reduce_mean(pred_seq_logits.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)
        
        # 将 token 权重和目标展开到一维
        tokens_weights_tensor = tokens_weights_tensor.view(-1)
        targets_tokens_tensor = targets_tokens_tensor.view(-1)
        
        # 对于权重为 0 的 token，将对应的目标设置为 ignore id（self.num_vocal）
        ignore_token_ids = torch.nonzero(tokens_weights_tensor == 0.).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.num_vocal
        
        # 重新将目标 token转换为 (N, 6) 的形状
        targets_tokens_tensor = targets_tokens_tensor.view(-1, 6)
        
        # 分类目标：取第 0 个 token，并减去偏移量（self.num_bins + 1）
        cls_targets_tokens_tensor = targets_tokens_tensor[:, 0] - (self.num_bins + 1)
        
        # 回归目标：取 tokens 1 到 5，拉平成一维
        reg_targets_tokens_tensor = targets_tokens_tensor[:, 1:].reshape(-1)
        
        # 计算交叉熵损失（self.loss_reg 为 CrossEntropyLoss）
        loss_cls = self.loss_reg(pred_seq_cls_logits, cls_targets_tokens_tensor) / avg_factor
        loss_reg = self.loss_reg(pred_seq_reg_logits, reg_targets_tokens_tensor) / avg_factor

        return loss_cls, loss_reg


    def decoder_inference(self, 
                        layers_module, 
                        patch_embed: Tensor, 
                        patch_mask: Tensor, 
                        text_embed: Tensor, 
                        text_mask: Tensor, 
                        grid_pos_embed: Tensor, 
                        grid_mask: Tensor, 
                        references: Tensor, 
                        bert_embed_func: Callable, 
                        task_embedding: Tensor, 
                        vocabulary_embed: Tensor, 
                        grid_interpolate: bool=True, 
                        global_only_image: bool=True) -> Dict:
        """
        Autoregressive decoding to generate BEV occupancy bbox tokens.
        
        For OCC head, the target token sequence represents BEV occupancy bbox parameters:
            [class_token, cx_token, cy_token, w_token, h_token, theta_token]
        The decoder sequentially generates tokens over self.dec_length+1 steps (with self.dec_length=5).
        An additional BEV mapping layer (self.bev_mapping) can be applied to the decoded tokens 
        to better align them with BEV space.
        """
        pre_kv_list = []
        patch_resolution = patch_embed.shape[1:3]
        grid_resolution_perwin = [grid_mask.shape[i+1] // (patch_resolution[i] // layers_module[0].window_size) for i in range(2)]
        batch_size, query_num = references.shape[:2]
        references = references[:, :, :2]  # assume normalized BEV grid coordinates in [0,1]
        image_patch = patch_embed 
        grid_token = grid_pos_embed.clone() 
        grid_mask = grid_mask.flatten(1)

        # Compute local feature tokens from image patch (if grid_interpolate is True)
        grid_interpolate_feats = [] if grid_interpolate else None
        for layer_id, layer in enumerate(layers_module):
            if grid_interpolate:
                # compute tokens of local image feature
                input_img_patch = image_patch.permute(0, 3, 1, 2) 
                # map normalized BEV coordinates to [-1,1] for grid_sample
                grid_position = references.unsqueeze(2) * 2 - 1 
                grid_local_feat = F.grid_sample(input_img_patch, grid_position, align_corners=False)
                grid_interpolate_feats.append(grid_local_feat.squeeze(-1).permute(0, 2, 1))
            # calculate number of patches in current window or use global if window_size <=0
            window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
            scope_len = window_patch_num
            # generate attention mask for text if provided
            if text_embed is not None:
                observe_num = window_patch_num + text_embed.shape[1]
                scope_len += text_embed.shape[1]
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)
                text_len = observe_num - window_patch_num
                attn_mask[window_patch_num:observe_num, window_patch_num:observe_num] = \
                    torch.triu(torch.ones(text_len, text_len, device=grid_token.device), diagonal=1)
            else:
                observe_num = window_patch_num
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)
            
            # Perform image feature interaction using transformer layer
            image_patch, text_embed, inter_kv = layer.img_forward(
                image_patch, text_embed, attn_mask[:observe_num, :observe_num].bool(), patch_mask, text_mask,
                return_intermediate=True)
            pre_kv_list.append(inter_kv)
        
        outputs_coords = []  # list to collect regression tokens for each decoding step
        outputs_classes = None  # will hold classification probabilities

        # Start autoregressive decoding loop:
        # pos_id from 0 to self.dec_length (total steps = self.dec_length + 1, e.g., 6 steps)
        for pos_id in range(0, self.dec_length + 1):
            # For pos_id==0: initial token embedding is the grid token (local image info)
            if pos_id == 0:
                input_embed = grid_token.view(batch_size * query_num, 1, -1)
            # For pos_id==1: use task identifier token via BERT embedding
            elif pos_id == 1:
                input_embed = bert_embed_func(
                    inputs_embeds=task_embedding[None, None, :].repeat(batch_size * query_num, 1, 1))
            x = input_embed
            # Iterate over transformer layers for decoding
            for layer_id, layer in enumerate(layers_module):
                if layer.window_size <= 0 and global_only_image:
                    continue  # skip if only global image interaction is desired
                x = x.view(batch_size, query_num, 1, -1)
                # Update local image token for the first token of each layer if using grid interpolation
                if grid_interpolate and pos_id == 0:
                    x += grid_interpolate_feats[layer_id].unsqueeze(2)
                # Re-add positional embedding to the first token before each layer (except first layer)
                if layer_id > 0 and pos_id == 0:
                    x += grid_pos_embed.view(batch_size, query_num, 1, -1)
                
                # Generate attention mask based on current step pos_id and window patch settings
                window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
                unit_grid_num = np.prod(grid_resolution_perwin) if layer.window_size > 0 else query_num
                observe_num = window_patch_num + (text_embed.shape[1] if text_embed is not None else 0)
                attn_mask = torch.zeros(unit_grid_num, observe_num, device=input_embed.device)
                iter_pad_masks = (1. - torch.eye(unit_grid_num, device=input_embed.device)).repeat(1, pos_id+1)
                attn_mask = torch.cat([attn_mask, iter_pad_masks], dim=1)

                # Update token representation via token_forward
                x, pre_kv_update = layer.token_forward(
                    image_patch=image_patch, 
                    grid_token=x, 
                    grid_position=references, 
                    idx=pos_id,
                    attn_mask=attn_mask.bool(), 
                    patch_mask=patch_mask, 
                    grid_mask=grid_mask, 
                    text_mask=text_mask,  
                    pre_kv=pre_kv_list[layer_id])
                pre_kv_list[layer_id] = pre_kv_update
                x = x.view(batch_size * query_num, 1, -1)
                
                # For pos_id==0, no decoding occurs, just propagate information
                if pos_id == 0:
                    continue
                # At the last layer of the transformer, decode current token via argmax over vocabulary
                if layer_id == (len(layers_module) - 1):
                    logits = (x @ vocabulary_embed.transpose(0, 1))[:, -1, :]
                    if pos_id == 1:
                        # For pos_id==1: predict classification token; effective vocabulary is from offset (self.num_bins+1)
                        start_offset = self.num_bins + 1
                        current_logits = logits[:, start_offset: start_offset + self.num_classes + 1]
                        outputs_classes = current_logits.softmax(dim=-1)[:, :-1]  # soft probabilities
                        pred_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                    elif pos_id > 1 and pos_id <= self.dec_length:
                        # For pos_id > 1: predict regression tokens for BEV bbox (cx, cy, w, h, theta)
                        start_offset = 0
                        current_logits = logits[:, :self.num_bins + 1]
                        pred_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                        outputs_coords.append(pred_token)
                    else:
                        raise RuntimeError('OCC head expects exactly 6 tokens (1 classification + 5 regression tokens).')
            
            # Update input embedding for next decoding step using predicted token and BERT embedding function
            if pos_id > 0:
                input_embed = bert_embed_func(
                    inputs_embeds=vocabulary_embed[(pred_token + start_offset)],
                    past_key_values_length=pos_id)
        
        # Reshape classification outputs to (batch_size, query_num, -1)
        outputs_classes = outputs_classes.view(batch_size, query_num, -1)
        # Concatenate regression tokens (5 tokens per candidate), then normalize by self.num_bins to get continuous values
        outputs_coords = torch.cat(outputs_coords, dim=-1).view(batch_size, query_num, -1) / self.num_bins
        
        # For example,可以对中心坐标进行残差校正（假设 references 为初始候选中心）
        outputs_coords[:, :, :2] = outputs_coords[:, :, :2] - 0.5 + references
        
        output_dict = {'outputs_classes': outputs_classes, 'outputs_coords': outputs_coords}
        
        return output_dict


    def predict(self, 
                outputs_classes: Tensor, 
                outputs_coords: Tensor,
                batch_data_samples: SampleList, 
                rescale: bool = False) -> InstanceList:
        """
        Post-process the decoding outputs to produce final BEV occupancy bbox predictions.
        
        Args:
            outputs_classes (Tensor): Classification scores of the last layer,
                shape (bs, num_queries, num_classes+1), representing the soft scores.
            outputs_coords (Tensor): Regression outputs of the last layers,
                shape (bs, num_queries, 5), where each row represents
                [cx, cy, w, h, theta] in normalized BEV coordinates (e.g., [0,1]).
            batch_data_samples (SampleList): Data samples that contain meta information.
            rescale (bool): If True, convert predictions to original BEV coordinate scale.
        
        Returns:
            InstanceList: A list of InstanceData for each image, where each InstanceData
            contains the predicted occupancy bboxes (in BEV space), their scores and labels.
        """
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = outputs_classes[img_id]
            bbox_pred = outputs_coords[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_single(cls_score, bbox_pred, img_meta, rescale)
            result_list.append(results)
        return result_list


    def _predict_single(self, cls_score: Tensor, bbox_pred: Tensor,
                        img_meta: dict, rescale: bool = True) -> InstanceData:
        """
        Convert the outputs from the last decoder layer into final BEV occupancy bbox predictions.
        
        For each BEV occupancy bbox, the prediction includes:
            - A class score and label (from cls_score).
            - Continuous bbox parameters [cx, cy, w, h, theta] (from bbox_pred).
        
        Args:
            cls_score (Tensor): Classification logits with shape (num_queries, num_classes+1).
            bbox_pred (Tensor): Regression outputs with shape (num_queries, 5) representing
                [cx, cy, w, h, theta] in normalized BEV coordinates.
            img_meta (dict): Meta information, should contain key 'bev_shape' as (H, W).
            rescale (bool): If True, map normalized predictions to BEV coordinate scale.
        
        Returns:
            InstanceData: Containing fields:
                - bboxes (Tensor): Final occupancy bboxes with shape (num_instances, 5).
                - scores (Tensor): Classification scores for each bbox.
                - labels (Tensor): Predicted class labels.
        """
        # Get maximum number of predictions per image from test config
        max_per_img = self.test_cfg.get('max_per_img', cls_score.shape[0])
        
        # Flatten classification scores and select top-k predictions
        cls_score_flat = cls_score.reshape(-1)
        scores, indexes = cls_score_flat.topk(max_per_img)
        
        # Derive predicted labels and corresponding indices for bbox_pred
        # 假设类别预测采用 argmax，类别值从 0 到 self.num_classes-1
        det_labels = indexes % (self.num_classes + 1)  # 这里包含背景类别或 ignore 类别
        bbox_index = torch.div(indexes, (self.num_classes + 1), rounding_mode='trunc')

        
        # Select the corresponding regression predictions (each has 5 values: cx, cy, w, h, theta)
        bbox_pred = bbox_pred[bbox_index]
        
        # Map normalized bbox parameters to BEV coordinate space if needed.
        # 假设 img_meta['bev_shape'] 包含 BEV 图的高度和宽度，例如 (200, 200)
        bev_H, bev_W = img_meta.get('bev_shape', (200, 200))
        rescale = False
        if rescale:
            # 将中心坐标和尺寸从 [0,1] 映射到 BEV 图尺度
            bbox_pred[:, 0] = bbox_pred[:, 0] * bev_W   # cx
            bbox_pred[:, 1] = bbox_pred[:, 1] * bev_H   # cy
            bbox_pred[:, 2] = bbox_pred[:, 2] * bev_W   # width
            bbox_pred[:, 3] = bbox_pred[:, 3] * bev_H   # height
            # theta 可以保持归一化状态或根据需求映射到实际角度（例如 [-pi, pi]），这里暂保持不变
        # Optionally,进行边界值裁剪
        bbox_pred[:, 0].clamp_(min=0, max=bev_W)
        bbox_pred[:, 1].clamp_(min=0, max=bev_H)
        
        # 构造输出结果，使用 InstanceData 封装预测结果
        results = InstanceData()
        results.bboxes = bbox_pred  # shape (num_instances, 5)
        results.scores = scores
        results.labels = det_labels
        return results



