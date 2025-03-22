import os
import numpy as np
import mmcv
from mmengine.registry import TRANSFORMS
from mmengine.fileio import FileClient  # 重要修改

@TRANSFORMS.register_module()
class LoadFrontCameraImageFromFile:
    """
    用于加载前摄像头图像的流水线类。

    该流水线假设输入字典中的 key 'img_filename' 为一个列表，
    列表中的第一个元素即为前摄像头图像的文件路径。加载后将图像数据、
    图像尺寸、归一化配置等信息写入结果字典，以便后续数据处理和模型输入使用。

    Args:
        to_float32 (bool): 是否将图像转换为 float32。默认 False。
        color_type (str): 图像读取时的颜色类型。默认 'unchanged'。
        file_client_args (dict): 文件客户端参数。默认使用 {'backend': 'disk'}。
        img_root (str): 图像根目录，加载时会将 img_root 与文件路径拼接。
    """
    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 img_root=''):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = FileClient(**self.file_client_args)  # 重要修改
        self.img_root = img_root

    def __call__(self, results: dict) -> dict:
        # 取出前摄像头图像文件名
        front_img_filename = results['img_filename'][0]
        
        # 如果文件名已经以 '.' 或 'data/' 开头，则直接使用它，否则拼接 img_root
        if front_img_filename.startswith('.') or front_img_filename.startswith('data/'):
            front_img_path = front_img_filename
        else:
            front_img_path = os.path.join(self.img_root, front_img_filename)
        
        # 读取图像
        if self.file_client_args['backend'] == 'petrel':
            img_bytes = self.file_client.get(front_img_path)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        else:
            img = mmcv.imread(front_img_path, self.color_type)
        
        if self.to_float32:
            img = img.astype(np.float32)
        
        # 检查图像尺寸是否为 1600x900
        expected_shape = (900, 1600, 3)
        if img.shape != expected_shape:
            raise RuntimeError(f"Image shape {img.shape} is not equal to expected {expected_shape} for file {front_img_path}.")

        results['filename'] = front_img_filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False
        )
        return results



    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(to_float32={self.to_float32}, "
                f"color_type='{self.color_type}', img_root='{self.img_root}')")
