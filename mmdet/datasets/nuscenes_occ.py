import copy
import pickle
import mmcv
import numpy as np
import torch  # 用于构造 tensor
from typing import Optional, Dict, List, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import FileClient
from mmdet.registry import DATASETS

# 为了计算旋转矩阵及 yaw 等信息，需要引入 Quaternion
from nuscenes.eval.common.utils import Quaternion
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes



# 如果需要使用其它 nuScenes API，可在此处引入

@DATASETS.register_module()
class NuScenesOccDataset(BaseDataset):
    """Dataset for occupancy prediction.
    
    此数据集加载包含时序信息的 nuScenes 注释文件（PKL 格式），
    并按照多帧（当前帧及未来帧）构造 occ 任务所需的额外信息，
    例如：各帧间的坐标变换、未来检测标注等信息，以供 occ 标签生成流水线使用.
    
    Args:
        data_root (str): nuScenes 数据根目录.
        ann_file (str): 包含 temporal annotation 的 PKL 文件路径.
        pipeline (list[dict]): 数据处理流水线.
        load_interval (int): 下采样间隔，默认 1.
        test_mode (bool): 是否测试模式.
        queue_length (int): 使用的帧数（例如历史帧数量），默认 4.
        occ_n_future (int): 用于 occ 标签的未来帧数，默认 4.
        file_client_args (dict): mmcv.FileClient 配置.
        metainfo (dict, optional): 数据集级元信息.
    """
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: list,
                 load_interval=1,
                 test_mode=False,
                 queue_length=4,
                 occ_n_future=4,
                 file_client_args=dict(backend='disk'),
                 metainfo: Optional[Dict] = None,
                 occ_receptive_field: int = 3,         # past + current
                 occ_only_total_frames: int = 7,         # 用于评估等（比如 7 帧的总数）
                 **kwargs):
        self.data_root = data_root
        self.ann_file = ann_file
        self.load_interval = load_interval
        self.test_mode = test_mode
        self.queue_length = queue_length
        self.occ_n_future = occ_n_future
        self.file_client_args = file_client_args
        self.box_mode_3d = 'lidar'


        # 以下两个 occ 参数供后续 occ 数据处理使用
        self.occ_receptive_field = occ_receptive_field
        self.occ_only_total_frames = occ_only_total_frames

        # 存储所有排序后的原始注释信息
        self._raw_data_infos = None

        # 如果 metainfo 中没有提供类别信息，可设置默认的 nuScenes 类别
        if metainfo is None or 'CLASSES' not in metainfo:
            self.CLASSES = [
                "car",
                "truck",
                "construction_vehicle",
                "bus",
                "trailer",
                "barrier",
                "motorcycle",
                "bicycle",
                "pedestrian",
                "traffic_cone",
            ]
        else:
            self.CLASSES = metainfo['CLASSES']

        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            metainfo=metainfo if metainfo is not None else dict(),
            test_mode=test_mode,
            **kwargs
        )

    def load_data_list(self) -> List[dict]:
        """从 ann_file 加载注释信息，并返回数据列表."""
        file_client = FileClient(**self.file_client_args)
        data_bytes = file_client.get(self.ann_file)
        data = pickle.loads(data_bytes)

        # 对注释信息按照时间戳排序，并下采样
        data_infos = sorted(data['infos'], key=lambda x: x['timestamp'])
        data_infos = data_infos[:: self.load_interval]
        self._raw_data_infos = data_infos

        # 可设置 dataset-level 的元信息
        self._metainfo = {
            'dataset_type': 'NuScenesOcc',
            'version': data.get('metadata', {}).get('version', 'v1.0'),
            'info_source': 'nuScenes_infos',
        }

        # 返回每个数据项仅包含一个索引，后续在 get_data_info 中根据索引构建完整样本
        data_list = [{'idx': i} for i in range(len(data_infos))]
        # print(f"[DEBUG] load_data_list: Loaded {len(data_infos)} samples from {self.ann_file}.")
        return data_list


    def get_data_info(self, data_item: dict) -> Optional[dict]:
        """根据 data_list 中的元素构建样本字典."""
        # print(f"[DEBUG] get_data_info: Received data_item = {data_item}")
        idx = data_item['idx']
        if idx < 0 or idx >= len(self._raw_data_infos):
            # print(f"[DEBUG] get_data_info: Index {idx} 超出范围！")
            return None
        info = self._raw_data_infos[idx]
        # print(f"[DEBUG] get_data_info: Sample token = {info.get('token', 'N/A')}, timestamp = {info.get('timestamp', 'N/A')}, scene = {info.get('scene_token', 'N/A')}")

        # 构造基础信息
        input_dict = dict(
            sample_idx=info['token'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            sweeps=info.get('sweeps', []),
            lidar2ego_rotation=info['lidar2ego_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            ego2global_translation=info['ego2global_translation'],
        )

        # **新增 img_filename**
        if 'cams' in info:
            input_dict['img_filename'] = [
                cam_info['data_path'] for cam_info in info['cams'].values()
            ]
            # print(f"[DEBUG] get_data_info: Loaded {len(input_dict['img_filename'])} image filenames.")
        else:
            input_dict['img_filename'] = []
            # print("[DEBUG] get_data_info: No camera images found.")

        # --- 处理 OCC 相关数据 ---
        prev_indices, future_indices = self.occ_get_temporal_indices(idx, self.occ_receptive_field, self.occ_n_future)
        all_frames = prev_indices + [idx] + future_indices
        has_invalid_frame = (-1 in all_frames[:self.occ_only_total_frames])
        input_dict['occ_has_invalid_frame'] = has_invalid_frame
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0

        # 计算 OCC 未来帧的变换信息
        future_frames = [idx] + future_indices
        occ_transforms = self.occ_get_transforms(future_frames)
        input_dict.update(occ_transforms)

        # 未来帧检测标注信息
        input_dict['occ_future_ann_infos'] = self.get_future_detection_infos(future_frames)

        # print(f"[DEBUG] get_data_info: Constructed input_dict keys: {list(input_dict.keys())}")
        return input_dict



    # def _get_occ_ann_info(self, idx: int) -> dict:
    #     """模仿 UniAD 的 occ_get_detection_ann_info，返回用于生成 occ 标签的检测信息."""
    #     info = self._raw_data_infos[idx]
    #     ann_dict = dict(
    #         gt_bboxes_3d = info['gt_boxes'],      # 3D 框信息
    #         gt_names_3d    = info['gt_names'],      # 类别名称
    #         gt_inds        = info['gt_inds'],       # 实例 id
    #         visibility_tokens = info.get('visibility_tokens', None)
    #     )
    #     num_boxes = len(info.get('gt_boxes', [])) if info.get('gt_boxes', None) is not None else 0
    #     # print(f"[DEBUG] _get_occ_ann_info: For idx {idx}, token = {info.get('token', 'N/A')}, found {num_boxes} boxes.")
    #     return ann_dict

    def occ_get_temporal_indices(self, idx: int, receptive_field: int, n_future: int) -> (List[int], List[int]):
        """计算历史帧和未来帧的索引，若不在同一 scene 则返回 -1."""
        current_scene = self._raw_data_infos[idx]['scene_token']
        previous_indices = []
        for t in range(-receptive_field + 1, 0):
            idx_t = idx + t
            if idx_t >= 0 and self._raw_data_infos[idx_t]['scene_token'] == current_scene:
                previous_indices.append(idx_t)
            else:
                previous_indices.append(-1)
        future_indices = []
        n_total = len(self._raw_data_infos)
        for t in range(1, n_future + 1):
            idx_t = idx + t
            if idx_t < n_total and self._raw_data_infos[idx_t]['scene_token'] == current_scene:
                future_indices.append(idx_t)
            else:
                future_indices.append(-1)
        # print(f"[DEBUG] occ_get_temporal_indices: For idx {idx}, prev: {previous_indices}, future: {future_indices}")
        return previous_indices, future_indices

    def occ_get_transforms(self, indices: List[int], data_type=torch.float32) -> Dict[str, List]:
        """对给定索引列表，计算每帧的 lidar->ego 与 ego->global 的变换矩阵及平移向量."""
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []
        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self._raw_data_infos[index]
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']
                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)
                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))
        return {
            'occ_l2e_r_mats': l2e_r_mats,
            'occ_l2e_t_vecs': l2e_t_vecs,
            'occ_e2g_r_mats': e2g_r_mats,
            'occ_e2g_t_vecs': e2g_t_vecs,
        }

    def get_future_detection_infos(self, future_frames: List[int]) -> List[Optional[dict]]:
        """对给定未来帧索引，返回每一帧的检测标注信息."""
        detection_infos = []
        for frame in future_frames:
            if frame >= 0:
                detection_infos.append(self.occ_get_detection_ann_info(frame))
            else:
                detection_infos.append(None)
        return detection_infos



    def occ_get_detection_ann_info(self, idx: int) -> dict:
        info = self._raw_data_infos[idx].copy()
        gt_bboxes_3d = info['gt_boxes'].copy()
        gt_names_3d = info['gt_names'].copy()
        gt_inds = info['gt_inds'].copy()
        gt_vis_tokens = info.get('visibility_tokens', None)

        # 将原始 NumPy 数组转换为支持 rotate/translate 的数据结构
        boxes = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)
        )
        # 如果目标模式不是 lidar，则进行转换；否则不转换
        if self.box_mode_3d != 'lidar':
            boxes = boxes.convert_to(self.box_mode_3d)
        # 此时 boxes 就是支持 rotate/translate 方法的对象
        gt_bboxes_3d = boxes

        # 将类别名称转换为索引
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        ann_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_vis_tokens=gt_vis_tokens,
        )
        return ann_results


    def prepare_train_data(self, idx):
        data_info = self.get_data_info({'idx': idx})
        if data_info is None:
            return None
        return self.pipeline(data_info)

    def prepare_test_data(self, idx):
        data_info = self.get_data_info({'idx': idx})
        if data_info is None:
            return None
        return self.pipeline(data_info)

    def __getitem__(self, idx):
        data_info = self.get_data_info({'idx': idx})
        if data_info is None:
            return None
        output = self.pipeline(data_info)
        return output
