import os
import os.path as osp
from typing import Dict, Sequence, Optional, List

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

from mmdet.registry import METRICS

@METRICS.register_module()
class KittiMetric(BaseMetric):
    """Minimal KITTI 2D metric,同时可以在Python内打印AP和将结果保存为KITTI官方格式.

    - 当 `format_only=True` 时，会将推理结果写入 .txt 文件（每张图一个），
      同时在 Python 内根据简化策略计算 per-class AP（单一IoU阈值），
      并返回到日志中。
    - 由于 KITTI 不同类别有不同 IoU (如 Car 0.7, Ped/Cyc 0.5)，
      我们做一个简化示例：按类别判断 IoU 阈值，用 VOC 11 点法或 40 点法计算 AP。
    - 注意：真正 KITTI 官方还需要区分 easy / moderate / hard，截断/遮挡等。
      这里只做最小化演示。

    Args:
        classes (List[str]): KITTI包含的类别，例如['Car','Pedestrian','Cyclist']。
        format_only (bool): 是否写出结果 + 计算AP. 如果为False，仅示例抛异常。
        outfile_prefix (str|None): 写出 .txt 文件的前缀（含路径），
            例如 'kitti_out/kitti' => kitti_out/kitti_000000.txt ...
        collect_device (str): DDP收集结果用的设备。
        prefix (str): 日志前缀。
    """

    default_prefix: Optional[str] = 'KITTI'

    def __init__(self,
                 classes: Sequence[str],
                 format_only: bool = True,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.CLASSES = list(classes)
        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        if not self.format_only:
            raise NotImplementedError(
                '此示例只示范 format_only=True 并在Python内简单算AP。')

    def process(self, data_batch, data_samples):
        """处理每个batch的推理结果并保存到 `self.results`."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            bboxes = pred['bboxes'].cpu().numpy()    # shape (n,4), xyxy
            labels = pred['labels'].cpu().numpy()    # shape (n,)
            scores = pred['scores'].cpu().numpy()    # shape (n,)
            img_id = data_sample.get('img_id', None)
            if img_id is None:
                img_id = data_sample.get('sample_idx', -1)

            # 同时需要 GT 用于 AP 计算 (仅2D)
            # 这里假设 data_sample['gt_instances'] 中有 bboxes, labels
            # 若KITTI某些信息不在, 要看你实际数据
            gt = data_sample.get('gt_instances', None)
            if gt is not None:
                gt_bboxes = gt.get('bboxes', None)
                gt_labels = gt.get('labels', None)
                if gt_bboxes is not None:
                    gt_bboxes = gt_bboxes.cpu().numpy()
                if gt_labels is not None:
                    gt_labels = gt_labels.cpu().numpy()
            else:
                gt_bboxes = None
                gt_labels = None

            self.results.append(dict(
                img_id=img_id,
                pred_bboxes=bboxes,    # 预测框
                pred_labels=labels,
                pred_scores=scores,
                gt_bboxes=gt_bboxes,   # 真值框
                gt_labels=gt_labels,
            ))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """保存结果为 KITTI 格式并在 Python 内做简易的 AP."""
        logger: MMLogger = MMLogger.get_current_instance()

        if not self.format_only:
            raise NotImplementedError(
                '当前示例只在 format_only=True 时写出文件 + 简化AP计算.')

        if self.outfile_prefix is None:
            raise ValueError('outfile_prefix 不能是 None, 需要指定输出路径.')

        logger.info(f'Saving KITTI txt results to prefix={self.outfile_prefix}')

        out_dir = osp.dirname(self.outfile_prefix)
        os.makedirs(out_dir, exist_ok=True)

        # 1) 写出 .txt 文件
        for res in results:
            img_id = res['img_id']
            # 强制转 int(若是str可转int), 也可做判断
            img_id_int = int(img_id)
            txt_filename = f'{self.outfile_prefix}_{img_id_int:06d}.txt'

            lines = []
            for bbox, label, score in zip(res['pred_bboxes'], res['pred_labels'], res['pred_scores']):
                cls_name = self.CLASSES[label]
                x1, y1, x2, y2 = bbox
                line = (f'{cls_name} 0 0 0 '
                        f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} '
                        f'-1 -1 -1 -1000 -1000 -1000 -10 {score:.3f}')
                lines.append(line)

            with open(txt_filename, 'w') as f:
                f.write('\n'.join(lines))

        logger.info('Done! KITTI 2D results saved. You can evaluate with devkit. ' +
                    'Now do a python-level AP as DEMO:')

        # 2) 在python里做一个非常简化的 KITTI AP 计算
        #    (不区分easy/mod/hard, 只区分类 & IoU阈值)
        #    - Car: 0.7
        #    - Pedestrian: 0.5
        #    - Cyclist: 0.5
        #    - VOC 11point method

        class_to_iou = {
            'Car': 0.7,
            'Pedestrian': 0.5,
            'Cyclist': 0.5,
        }

        # 把所有 pred, gt 按类别分组
        cls_pred = {cls: [] for cls in self.CLASSES}
        cls_gt = {cls: [] for cls in self.CLASSES}

        # group (img_id, bboxes, scores) for each class
        for res in results:
            img_id = res['img_id']

            # GT
            gt_boxes = res.get('gt_bboxes', None)
            gt_labels = res.get('gt_labels', None)
            if gt_boxes is None or gt_labels is None:
                continue
            for box, lab in zip(gt_boxes, gt_labels):
                cls_name = self.CLASSES[lab]
                cls_gt[cls_name].append((img_id, box))

            # PRED
            for box, lab, sc in zip(res['pred_bboxes'], res['pred_labels'], res['pred_scores']):
                cls_name = self.CLASSES[lab]
                cls_pred[cls_name].append((img_id, box, sc))

        # compute AP for each class
        ap_dict = {}
        for cls_name in self.CLASSES:
            iou_thr = class_to_iou.get(cls_name, 0.5)

            # 1) gather GT => {img_id: [boxes], ...}
            gt_img_dict = {}
            for (imgid, gbox) in cls_gt[cls_name]:
                gt_img_dict.setdefault(imgid, []).append(gbox)

            # 2) gather Pred => [(img_id, box, score)] sort desc
            preds = cls_pred[cls_name]
            preds.sort(key=lambda x: x[2], reverse=True)

            # 3) match
            tp = []
            fp = []
            # total gt
            total_gts = len(cls_gt[cls_name])

            matched = set()  # (img_id, gt_idx) used
            for (imgid, pbox, score) in preds:
                gboxes = gt_img_dict.get(imgid, [])
                # find best iou
                best_iou = 0
                best_i = -1
                for i, gbox in enumerate(gboxes):
                    iou_ = self._iou_xyxy(pbox, gbox)
                    if iou_ > best_iou:
                        best_iou = iou_
                        best_i = i
                if best_iou >= iou_thr and (imgid, best_i) not in matched:
                    tp.append(1)
                    fp.append(0)
                    matched.add((imgid, best_i))
                else:
                    tp.append(0)
                    fp.append(1)

            tp = np.cumsum(tp)
            fp = np.cumsum(fp)

            recall = tp / (total_gts + 1e-6)
            precision = tp / (tp + fp + 1e-6)

            ap = self._voc_ap(recall, precision, use_11_point=True)
            ap_dict[f'{cls_name}_AP'] = float(ap)

        # mean ap
        mAP = np.mean(list(ap_dict.values())) if ap_dict else 0.0
        ap_dict['mAP'] = float(mAP)

        logger.info(f"Python-level AP summary: {ap_dict}")

        return ap_dict

    def _voc_ap(self, recall, precision, use_11_point=False):
        """Compute AP using 11-point method or standard integral method."""
        if use_11_point:
            # 11-point
            ap = 0.0
            for t in np.arange(0., 1.1, 0.1):
                p = precision[recall >= t]
                if p.size > 0:
                    ap += np.max(p)
                else:
                    ap += 0.
            ap /= 11.0
        else:
            # integrate all points
            # first add boundary
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # for i in range(mpre.size - 1, 0, -1):
            #     mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            # area sum
            # ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
            # (如COCO 这个仅是演示)

            # 这里为了简化就写个 11-point
            ap = 0.0
            for t in np.arange(0., 1.1, 0.1):
                p = mpre[mrec >= t]
                if p.size > 0:
                    ap += np.max(p)
                else:
                    ap += 0.
            ap /= 11.0
        return ap

    def _iou_xyxy(self, box1, box2):
        """Compute IoU of two bboxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0., x2 - x1)
        inter_h = max(0., y2 - y1)
        inter_area = inter_w * inter_h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        if union <= 0:
            return 0
        return inter_area / union
