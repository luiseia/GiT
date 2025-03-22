# mmdet/datasets/pipelines/load_annotations_3d_e2e.py
from mmengine.registry import TRANSFORMS
import numpy as np

# 如果你已有 mmdet3d 中的 LoadAnnotations3D 基类，可直接继承
# 否则可以根据需要继承 mmcv.transforms.BaseTransform 或 mmengine.transforms.BaseTransform
# 如果没有从 __init__.py 导出，就直接从 loading.py 导入
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
 

@TRANSFORMS.register_module()
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    """加载 3D 注释和未来帧的注释信息。

    Args:
        with_future_anns (bool): 是否加载未来帧的注释信息，默认 False。
        with_ins_inds_3d (bool): 是否加载实例索引信息，默认 False。
        ins_inds_add_1 (bool): 是否将实例索引加 1（使其从 1 开始），默认 False。
        其他参数通过 kwargs 传递给父类。
    """
    def __init__(self,
                 with_future_anns=False,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d
        self.ins_inds_add_1 = ins_inds_add_1

    def _load_future_anns(self, results):
        """加载 occ_future_ann_infos 中未来帧的 3D 注释信息。

        遍历 results['occ_future_ann_infos'] 中每一帧的注释，
        并分别将 gt_bboxes_3d、gt_labels_3d、gt_inds 和 gt_vis_tokens 保存到对应的 key 中。
        """
        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        gt_vis_tokens = []
        for ann_info in results.get('occ_future_ann_infos', []):
            
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info.get('gt_bboxes_3d', None))
                gt_labels_3d.append(ann_info.get('gt_labels_3d', None))
                ann_gt_inds = ann_info.get('gt_inds', None)
                if (ann_gt_inds is not None) and self.ins_inds_add_1:
                    # 假设 ann_gt_inds 为 NumPy 数组或列表，可以直接加 1
                    ann_gt_inds = np.array(ann_gt_inds) + 1
                gt_inds_3d.append(ann_gt_inds)
                gt_vis_tokens.append(ann_info.get('gt_vis_tokens', None))
            else:
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                gt_vis_tokens.append(None)

        results['future_gt_bboxes_3d'] = gt_bboxes_3d
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        results['future_gt_vis_tokens'] = gt_vis_tokens
        return results

    def _load_ins_inds_3d(self, results):
        """加载当前帧的实例索引，并根据需要对索引加 1。"""
        # 假设原始注释在 results['ann_info'] 中，并且包含 'gt_inds'
        ann_info = results.get('ann_info', {})
        ann_gt_inds = ann_info.get('gt_inds')
        if ann_gt_inds is None:
            return results

        # 复制一份防止修改原始数据
        ann_gt_inds = np.array(ann_gt_inds).copy()
        # 移除原始的 gt_inds，避免重复加载
        results['ann_info'].pop('gt_inds', None)

        if self.ins_inds_add_1:
            ann_gt_inds = ann_gt_inds + 1

        results['gt_inds'] = ann_gt_inds
        return results

    def __call__(self, results):
        # 直接从 occ_future_ann_infos 获取当前帧标注
      
        occ_infos = results['occ_future_ann_infos']
        if occ_infos[0] is None:
            raise KeyError("当前帧的标注信息为空！请检查 occ_future_ann_infos[0] 的数据。")
        # 将当前帧标注作为 ann_info 传递给父类
        results['ann_info'] = occ_infos[0]

        # 调用父类加载基本的 3D 注释（例如 gt_bboxes_3d, gt_labels_3d 等）
        results = super().__call__(results)

        # 根据配置进一步加载未来帧注释和实例索引
        if self.with_future_anns:
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)

        # 如果存在用于规划的 occ_future_ann_infos_for_plan，可在此添加处理逻辑
        if 'occ_future_ann_infos_for_plan' in results:
            results = self._load_future_anns_plan(results)

        return results
    

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str += (f', with_future_anns={self.with_future_anns}, '
                     f'with_ins_inds_3d={self.with_ins_inds_3d}, '
                     f'ins_inds_add_1={self.ins_inds_add_1}')
        return repr_str
