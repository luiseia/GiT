import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor
from mmdet.datasets.transforms.formatting import PackOccInputs
from mmdet.registry import MODELS

@MODELS.register_module()
class CustomOccDataPreprocessor(BaseDataPreprocessor):
    """自定义的 occupancy 预处理器。

    工作流程：
      1. 检查输入数据中是否存在 'inputs' 键；如果没有则调用 PackOccInputs 模块进行打包，
         该模块会将图像（或 dummy tensor）包装成列表。
      2. 将返回的列表转换为 Tensor（若列表中有 numpy 数组则先转换为 Tensor），
         并检查并调整图像维度（HWC -> CHW），最后堆叠成 4D tensor（B, C, H, W），以满足 backbone 的输入要求。
      3. 如果配置了归一化参数（mean、std），则对输入进行归一化处理。
    """
    def __init__(self,
                 mean=None,
                 std=None,
                 meta_keys: tuple = ('occ_future_ann_infos', 'task_name', 'head_cfg', 'git_cfg')):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std is not None else None
        
        self.meta_keys = meta_keys

        # print('CustomOccDataPreprocessor self.meta_keys', self.meta_keys)
        self.pack_occ_inputs = PackOccInputs(meta_keys=self.meta_keys)

        # print("[DEBUG] CustomOccDataPreprocessor initialized.")

    def forward(self, data: dict, training: bool = False) -> dict:
        # 如果数据中没有 'inputs' 键，则调用 PackOccInputs 进行打包
        # if 'inputs' not in data:
        #     # print("[DEBUG] CustomOccDataPreprocessor: 'inputs' not found, calling PackOccInputs...")
        #     data = self.pack_occ_inputs(data)
        # else:
        #     print("[DEBUG] CustomOccDataPreprocessor: 'inputs' already exists.")
        
        # 如果 'inputs' 是列表，则转换其中的每个元素为 Tensor（若为 numpy.ndarray），并检查维度是否为 CHW
        if isinstance(data['inputs'], list):
            new_inputs = []
            for inp in data['inputs']:
                # 如果不是 tensor，则转换为 tensor
                if not torch.is_tensor(inp):
                    inp = torch.from_numpy(inp)
                # 如果输入形状为 (H, W, C)（即 ndim == 3 且最后一维为 1 或 3），转换为 (C, H, W)
                if inp.ndim == 3 and inp.shape[2] in [1, 3]:
                    inp = inp.permute(2, 0, 1)
                new_inputs.append(inp)
            data['inputs'] = torch.stack(new_inputs, dim=0)
            # print("[DEBUG] CustomOccDataPreprocessor: Stacked inputs into tensor with shape", data['inputs'].shape)
        else:
            print("[DEBUG] CustomOccDataPreprocessor: inputs is already a tensor.")

        # 进行归一化处理（如果设置了 mean/std）
        if self.mean is not None and self.std is not None:
            data['inputs'] = (data['inputs'] - self.mean.to(data['inputs'].device)) / self.std.to(data['inputs'].device)
        
        return data

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
                f"meta_keys={self.meta_keys})")
