# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence
import os.path as osp


from scipy.optimize import linear_sum_assignment
import torch
import numpy as np

from mmengine.evaluator import BaseMetric
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmdet.registry import METRICS




@METRICS.register_module()
class Occ2DBoxMetric(BaseMetric):
    """评估2D带旋转角的bbox的自定义指标类。

    预测框和真值框均采用 [cx, cy, w, h, theta] 表示（theta 单位为弧度）。
    该评测类通过计算预测框和真值框之间的IoU矩阵，并利用匈牙利算法进行
    最优匹配，再统计匹配后的IoU，进而计算平均IoU、precision、recall和F1分数。

    Args:
        iou_mode (str): IoU计算模式，目前支持 'rbox'。
        iou_thresh (float): 判断匹配为真阳性(TP)的IoU阈值，默认0.5。
        output_dir (str, optional): 若指定，将保存预测结果到该目录。
        format_only (bool): 若为True，则仅格式化输出，不进行评测。
        prefix (str, optional): 指标名称前缀，用于多任务区分。
    """
    def __init__(self,
                 iou_mode: str = 'rbox',
                 iou_thresh: float = 0.5,
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(prefix=prefix, **kwargs)
        self.iou_mode = iou_mode
        self.iou_thresh = iou_thresh
        self.output_dir = output_dir
        self.format_only = format_only
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        处理一个batch的预测结果和数据样本。

        对于每个样本：
         - 从 sample 中获取预测框 (pred_2d_box_occ) 和真值框 (gt_2d_box_occ)，
           每个均为 shape (N, 5)，表示 [cx, cy, w, h, theta]。
         - 将 tensor 转换为 numpy 数组（如有必要）。
         - 计算预测框与真值框之间的IoU矩阵。
         - 利用匈牙利算法在IoU矩阵上做最佳匹配，得到一组匹配对及对应的IoU。
         - 将每个样本的匹配结果（匹配的IoU、预测框总数和真值框总数）存入 self.results，
           供后续在 compute_metrics() 中聚合计算指标。
        """
        for sample in data_samples:
            pred_boxes = sample['pred_2d_box_occ']['bboxes']
            gt_boxes = sample['bev_bbox_gt']
           
            self.visualize_and_save_boxes(pred_boxes, gt_boxes,filename=sample['image_id'])
            # 转换为 numpy 数组
            if pred_boxes is not None and isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()

            # 保存预测结果（如果output_dir不为空）
            if self.output_dir is not None:
                self._save_pred(sample)

            # 如果仅做格式化或缺少预测/真值，则跳过评测
            if self.format_only or pred_boxes is None or gt_boxes is None or \
               len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # 1. 计算预测框与真值框之间的 IoU 矩阵
            ious_matrix = self.compute_ious_matrix(pred_boxes, gt_boxes)

            # 2. 利用匈牙利算法匹配预测框和真值框
            row_inds, col_inds = self.hungarian_match(ious_matrix)
            matched_ious = ious_matrix[row_inds, col_inds]

            # 3. 保存当前样本的匹配信息
            self.results.append({
                'matched_ious': matched_ious,
                'num_pred': pred_boxes.shape[0],
                'num_gt': gt_boxes.shape[0]
            })

    @classmethod
    def visualize_and_save_boxes(self, pred_boxes, gt_boxes, 
                                 save_dir="/home/UNT/yz0370/projects/GiT/visualization/results_occ", 
                                 filename="visualization_occ.png"):
        """
        将预测框和真值框可视化，并保存到指定路径。

        Args:
            pred_boxes (np.ndarray or torch.Tensor): 预测框数组，形状 (P, 5)，每行格式为 [cx, cy, w, h, theta]。
            gt_boxes (np.ndarray or torch.Tensor): 真值框数组，形状 (G, 5)。
            save_dir (str): 保存图像的文件夹路径。
            filename (str): 保存的图像文件名。
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        # 转换为 numpy 数组（如果输入为 torch.Tensor）
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制预测框（红色）
        for box in pred_boxes:
            poly = self._rbox2poly(box)
            polygon = Polygon(poly, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)

        # 绘制真值框（绿色）
        for box in gt_boxes:
            poly = self._rbox2poly(box)
            polygon = Polygon(poly, closed=True, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(polygon)

        # 设置标题、坐标轴及显示比例
        ax.set_title("Predicted (red) vs Ground Truth (green) Boxes")
        ax.set_aspect('equal')
        ax.autoscale_view()

        # 保存图像
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Visualization saved to {save_path}")



    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        聚合所有样本的结果，并计算最终评测指标。

        Returns:
            Dict[str, float]: 包含平均 IoU、precision、recall 和 F1 分数的字典。
        """
        logger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'预测结果已保存至 {self.output_dir}')
            return {}

        all_matched_ious = []
        total_pred, total_gt = 0, 0

        for r in results:
            if 'matched_ious' not in r:
                continue
            matched_ious = r['matched_ious']
            all_matched_ious.extend(matched_ious)
            total_pred += r['num_pred']
            total_gt += r['num_gt']

        # 平均IoU计算
        if len(all_matched_ious) == 0:
            mean_iou = 0.0
        else:
            mean_iou = float(np.mean(all_matched_ious))

        # 基于设定阈值统计 TP, FP, FN
        tp = sum(iou > self.iou_thresh for iou in all_matched_ious)
        fp = total_pred - tp
        fn = total_gt - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        logger.info(f'[Occ2DBoxMetric] mean IoU={mean_iou:.4f}, '
                    f'precision={precision:.4f}, recall={recall:.4f}, f1_score={f1:.4f}')

        metrics = {
            'mIoU': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics

    @staticmethod
    def compute_ious_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """
        计算预测框和真值框之间的 IoU 矩阵。

        Args:
            pred_boxes (np.ndarray): shape (P, 5) 的预测框，每行 [cx, cy, w, h, theta]。
            gt_boxes (np.ndarray): shape (G, 5) 的真值框。

        Returns:
            np.ndarray: shape (P, G)，其中 [i, j] 表示 pred_boxes[i] 与 gt_boxes[j] 的 IoU。
        """
        P = pred_boxes.shape[0]
        G = gt_boxes.shape[0]
        ious = np.zeros((P, G), dtype=np.float32)
        for i in range(P):
            for j in range(G):
                ious[i, j] = Occ2DBoxMetric._rotated_iou_2d(pred_boxes[i], gt_boxes[j])
        return ious

    @staticmethod
    def hungarian_match(ious: np.ndarray):
        """
        基于匈牙利算法在 IoU 矩阵上进行匹配。

        Args:
            ious (np.ndarray): IoU矩阵，形状为 (P, G)。

        Returns:
            tuple: (row_inds, col_inds)，分别为匹配到的预测框和真值框的索引数组。
        """
        cost = 1 - ious  # 将 IoU 转换为代价矩阵（代价越低匹配越好）
        row_inds, col_inds = linear_sum_assignment(cost)
        return row_inds, col_inds

    @staticmethod
    def _rotated_iou_2d(box1, box2):
        """
        计算两个2D旋转框的IoU。

        每个框的格式为 [cx, cy, w, h, theta]（theta为弧度）。
        该函数先将旋转框转换为多边形，再计算多边形的交并面积。

        Args:
            box1 (array-like): 第一个旋转框。
            box2 (array-like): 第二个旋转框。

        Returns:
            float: IoU值。
        """
        poly1 = Occ2DBoxMetric._rbox2poly(box1)
        poly2 = Occ2DBoxMetric._rbox2poly(box2)
        inter_area = Occ2DBoxMetric._polygon_intersection_area(poly1, poly2)
        area1 = Occ2DBoxMetric._polygon_area(poly1)
        area2 = Occ2DBoxMetric._polygon_area(poly2)
        union_area = area1 + area2 - inter_area
        if union_area < 1e-6:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _rbox2poly(box):
        """
        将旋转框 [cx, cy, w, h, theta] 转换为四边形的四个顶点。

        Args:
            box (array-like): [cx, cy, w, h, theta]

        Returns:
            list: 四个顶点组成的列表，每个顶点为 [x, y]。
        """
        cx, cy, w, h, theta = box
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # 定义相对于中心点的四个角点（顺时针顺序）
        x1 = (w / 2) * cos_t - (h / 2) * sin_t + cx
        y1 = (w / 2) * sin_t + (h / 2) * cos_t + cy

        x2 = (w / 2) * cos_t - (-h / 2) * sin_t + cx
        y2 = (w / 2) * sin_t + (-h / 2) * cos_t + cy

        x3 = (-w / 2) * cos_t - (-h / 2) * sin_t + cx
        y3 = (-w / 2) * sin_t + (-h / 2) * cos_t + cy

        x4 = (-w / 2) * cos_t - (h / 2) * sin_t + cx
        y4 = (-w / 2) * sin_t + (h / 2) * cos_t + cy

        poly = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return poly

    @staticmethod
    def _polygon_area(poly):
        """
        计算多边形的面积（顶点按顺序排列）。

        Args:
            poly (list): 多边形各顶点的列表。

        Returns:
            float: 多边形的面积。
        """
        area = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        return abs(area) / 2.0

    @staticmethod
    def _polygon_intersection_area(poly1, poly2):
        """
        计算两个多边形的交集面积。

        注：这里提供一个极简示例，实际应用中建议使用 shapely 等库计算精确交集面积。

        Args:
            poly1 (list): 第一个多边形顶点列表。
            poly2 (list): 第二个多边形顶点列表。

        Returns:
            float: 两个多边形的交集面积。
        """
        # 示例：直接返回 min(area1, area2)*0.5，仅作为占位实现
        area1 = Occ2DBoxMetric._polygon_area(poly1)
        area2 = Occ2DBoxMetric._polygon_area(poly2)
        return min(area1, area2) * 0.5

    def _save_pred(self, sample):
        """
        可选：将预测结果保存到 output_dir 中。

        示例中将预测框写入文本文件，每行记录一个框的 [cx, cy, w, h, theta]。
        """
        if self.output_dir is None:
            return
        img_path = sample['image_id']
        basename = osp.splitext(osp.basename(img_path))[0]
        save_file = osp.join(self.output_dir, f'{basename}_pred.txt')
        pred_boxes = getattr(sample, 'pred_2d_box_occ', None)
        if pred_boxes is not None:
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            with open(save_file, 'w') as f:
                for box in pred_boxes:
                    f.write(' '.join(map(str, box)) + '\n')
