import os
import json
import tempfile
from typing import List, Optional, Dict, Tuple

import torch
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS

import numpy as np
import cv2
import matplotlib.pyplot as plt


@METRICS.register_module()
class OCCEvaluator(BaseMetric):
    """OccEvaluator 用于评估 GiT 生成的 occ 预测文本，与 ground truth（已包含在结果中的文本）进行对比。

    Args:
        ann_file (str): 虽然这里不需要单独加载 ground truth，但保持接口一致。
        collect_device (str): 结果收集时使用的设备，"cpu" 或 "gpu"，默认为 "cpu"。
        prefix (Optional[str]): 评估指标名称前缀。
    """
    def __init__(self,
                 ann_file: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file

    def process(self, data_batch, data_samples) -> None:
        """处理单个 batch 的数据，将预测 occ 文本和对应的 ground truth（通过 occ 分割图提取）存入 self.results。"""
        for i, data_sample in enumerate(data_samples):
            result = dict()
            # 预测结果存储在 pred_occ 字段中（文本格式，例如 "100,105; 100,105; ..."）
            result['pred_occ_text'] = data_sample.get('pred_occ')
            # 从 ground truth occ segmentation 提取坐标，并格式化为文本
            gt_occ_seg = data_sample.get('gt_occ_seg')
            coords = self._extract_occupied_coordinates(gt_occ_seg)
            result['gt_occ_text'] = self._format_coordinates_to_text(coords)
            # 如果 data_sample 中没有 image_id，则假定使用当前索引作为 image_id
            result['image_id'] = data_sample['image_id']
            
            self.results.append(result)

    def _extract_occupied_coordinates(self, gt_occ_seg: torch.Tensor) -> List[Tuple[int, int]]:
        """
        从 occ segmentation 真值中提取被占用区域的像素坐标。

        Args:
            gt_occ_seg (torch.Tensor): 形状 (H, W) 或 (C, H, W) 的 occ 分割图，
                其中 1 表示占用，0 表示空闲，ignore_index 表示忽略区域。

        Returns:
            List[Tuple[int, int]]: 被占用区域的 (x, y) 坐标列表。
        """
        # 如果 tensor 维度大于 2，假设第一个维度为通道，选择第一个通道处理
        if gt_occ_seg.dim() > 2:
            gt_occ_seg = gt_occ_seg[0]
        valid_mask = (gt_occ_seg == 1)
        coords_tensor = torch.nonzero(valid_mask, as_tuple=False)  # 形状 (num_coords, 2)
        coords = [(int(x), int(y)) for x, y in coords_tensor]
        return coords

    def _format_coordinates_to_text(self, coords: List[Tuple[int, int]]) -> str:
        """
        将坐标列表转换为文本字符串，格式为 "x1,y1; x2,y2; ...; xn,yn"

        Args:
            coords (List[Tuple[int, int]]): 坐标列表。

        Returns:
            str: 由坐标组成的文本字符串。
        """
        coord_strs = [f"{x},{y}" for x, y in coords]
        text = "; ".join(coord_strs)
        return text

    def compute_metrics(self, results: List) -> Dict:
        """计算预测结果与真实标签之间的评估指标。

        Args:
            results (List): 每个 batch 处理后的结果列表。

        Returns:
            Dict: 评估指标的字典。
        """
        # 保存结果到临时文件中
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = save_result(
                result=results,
                result_dir=temp_dir,
                filename='occ_pred',
                remove_duplicate='image_id'
            )
            # 调用 occ_text_eval 进行评估（注意此处不再需要 ann_file）
            eval_results = occ_text_eval(result_file)
        return eval_results


def save_result(result, result_dir, filename, remove_duplicate='') -> str:
    """保存预测结果为 JSON 文件，用于评估。

    Args:
        result: 待保存的结果列表。
        result_dir (str): 结果保存的目录路径。
        filename (str): 文件名（不包含扩展名）。
        remove_duplicate (str): 根据指定字段移除重复记录（例如 'image_id'）。

    Returns:
        str: 最终结果文件的完整路径。
    """
    if remove_duplicate:
        result_new = []
        id_list = []
        for res in result:
            key = res.get(remove_duplicate)
            if key not in id_list:
                id_list.append(key)
                result_new.append(res)
        result = result_new

    final_result_file = os.path.join(result_dir, f'{filename}.json')
    print(f'结果文件保存至: {final_result_file}')
    with open(final_result_file, 'w') as f:
        json.dump(result, f)
    return final_result_file


def convert_occ_text_to_matrix(occ_text: str, matrix_size=(200, 200)) -> np.ndarray:
    """
    将 occ 文本转换为二值矩阵。
    文本格式通常为 "x1,y1; x2,y2; ..."，
    若 token 格式异常（例如空 token 或缺少逗号），则忽略该 token。

    Args:
        occ_text (str): occ 文本。
        matrix_size (tuple): 输出矩阵的尺寸，默认 200×200。

    Returns:
        np.ndarray: 二值矩阵，坐标对应的位置填 1，其余为 0。
    """
    matrix = np.zeros(matrix_size, dtype=np.uint8)
    if occ_text is None or occ_text.strip() == "":
        return matrix
    tokens = occ_text.split(';')
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # 根据逗号分割
        parts = token.split(',')
        if len(parts) < 2:
            continue
        try:
            x = int(parts[0].strip())
            y = int(parts[1].strip())
            # 注意：矩阵索引为 [y, x]
            if 0 <= x < matrix_size[1] and 0 <= y < matrix_size[0]:
                matrix[y, x] = 1
        except Exception as e:
            continue
    return matrix


def occ_predict_visualization(results):
    """
    可视化推理结果。
    对于每个 record，通过 record['image_id'] 与前摄像头路径拼接加载图片，
    并将 record['gt_occ_text'] 与 record['pred_occ_text'] 转换成 200×200 的二值矩阵，
    然后每 3 个 record 为一组，将前摄像头图片、GT occ 矩阵与 Pred occ 矩阵并排显示，
    并在每行标题中显示对应的 image_id，
    最后将该组可视化结果保存至指定目录。
    
    Args:
        results (List): 结果列表，每个元素包含 'image_id'、'pred_occ_text' 和 'gt_occ_text'。
    """
    # 定义前摄像头图片所在路径
    cam_front_path = './data/nuscenes/samples/CAM_FRONT/'
    # 定义可视化结果保存目录
    save_dir = "/home/UNT/yz0370/projects/GiT/visualization/results_occ"
    os.makedirs(save_dir, exist_ok=True)

    group_size = 3
    num_records = len(results)
    groups = [results[i:i+group_size] for i in range(0, num_records, group_size)]

    for group_idx, group in enumerate(groups):
        num_rows = len(group)
        # 每行 3 列：前摄像头图片、GT occ 矩阵、Pred occ 矩阵
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        # 如果只有一行，确保 axs 是二维数组
        if num_rows == 1:
            axs = np.expand_dims(axs, axis=0)
        for row_idx, record in enumerate(group):
            image_id = record.get('image_id')
            pred_occ_text = record.get('pred_occ_text', "")
            gt_occ_text = record.get('gt_occ_text', "")

            # 1. 加载前摄像头图片
            image_path = os.path.join(cam_front_path, image_id)
            image = cv2.imread(image_path)
            if image is not None:
                # cv2 默认读取为 BGR，转换为 RGB 以便 matplotlib 正确显示
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 若图片加载失败，则用空白图代替
                image = np.zeros((200, 200, 3), dtype=np.uint8)

            # 2. 将 occ 文本转换为 200x200 的二值矩阵
            gt_matrix = convert_occ_text_to_matrix(gt_occ_text)
            pred_matrix = convert_occ_text_to_matrix(pred_occ_text)

            # 在同一行中显示三张图，标题中包含 image_id
            # 前摄像头图片
            ax = axs[row_idx, 0]
            ax.imshow(image)
            ax.set_title(f"{image_id} - Front")
            ax.axis('off')

            # Ground Truth occ 矩阵
            ax = axs[row_idx, 1]
            ax.imshow(gt_matrix, cmap='gray')
            ax.set_title(f"{image_id} - GT OCC")
            ax.axis('off')

            # Predicted occ 矩阵
            ax = axs[row_idx, 2]
            ax.imshow(pred_matrix, cmap='gray')
            ax.set_title(f"{image_id} - Pred OCC")
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"occ_visualization_group_{group_idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization group {group_idx} to {save_path}")


def occ_text_eval(results_file: str) -> dict:
    """评估 occ 文本预测结果与 ground truth 的对比情况。

    此函数直接从结果文件中读取每个样本的预测和真实 occ 文本，
    然后计算各项文本相似性指标，结构参考 COCOEvalCap，
    并在评估过程中调用 occ_predict_visualization 进行结果可视化。

    Args:
        results_file (str): 包含预测结果的 JSON 文件路径，每条记录应包含 'image_id'、'pred_occ_text' 和 'gt_occ_text' 字段。

    Returns:
        dict: 包含评估指标的字典，例如 {'Bleu_1': 0.XX, 'METEOR': 0.XX, ...}。
    """
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = json.load(f)

    # 构建字典，每个 image_id 对应 ground truth 和预测文本列表，
    # 每个元素以字典格式保存，包含 'caption' 字段
    gts = {}
    res = {}
    for record in results:
        image_id = record['image_id']
        gt_text = record['gt_occ_text']
        pred_text = record['pred_occ_text']
        # 注意这里增加了 'image_id' 和 'id' 字段（可根据评估工具需要调整）
        gts[image_id] = [{'image_id': image_id, 'caption': gt_text, 'id': image_id}]
        res[image_id] = [{'image_id': image_id, 'caption': pred_text, 'id': image_id}]

    # 使用自定义简单分词函数进行分词（直接按分号和逗号分割）
    def simple_tokenize(data):
        """
        对 occ 文本进行简单分词，但保留坐标对信息：
        1. 先以分号分割出各个坐标对字符串。
        2. 对每个坐标对字符串按逗号拆分，得到 x 和 y 值。
        3. 用下划线将 x 和 y 连接为一个 token，例如 "100_91"。
        4. 最后将所有 token 用空格连接返回。
        """
        tokenized_data = {}
        for key, captions in data.items():
            tokenized_data[key] = []
            for cap_dict in captions:
                caption = cap_dict['caption']
                # 以分号分割坐标对（假设每个坐标对之间用分号隔开）
                pairs = caption.split(';')
                tokens = []
                for pair in pairs:
                    pair = pair.strip()
                    if not pair:
                        continue
                    # 以逗号分割出坐标，期望得到两个部分：x 和 y
                    coords = pair.split(',')
                    if len(coords) == 2:
                        x = coords[0].strip()
                        y = coords[1].strip()
                        # 使用下划线将 x 和 y 合并为一个 token
                        tokens.append(f"{x}_{y}")
                    else:
                        # 如果格式不符合预期，直接加入原始文本
                        tokens.append(pair)
                tokenized_caption = ' '.join(tokens)
                tokenized_data[key].append(tokenized_caption)
        return tokenized_data

    gts = simple_tokenize(gts)
    res = simple_tokenize(res)

    # 设置各项评估指标
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    # from pycocoevalcap.spice.spice import Spice  # 可选的 Spice 指标

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
        # , (Spice(), "SPICE")
    ]

    eval_results = {}
    # 分别计算每个指标得分
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_results[m] = sc
                print("%s: %0.3f" % (m, sc))
        else:
            eval_results[method] = score
            print("%s: %0.3f" % (method, score))

    # 调用可视化函数展示推理结果
    occ_predict_visualization(results)

    return eval_results
