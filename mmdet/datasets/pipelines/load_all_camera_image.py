import os
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from mmengine.registry import TRANSFORMS
from mmengine.fileio import FileClient  # 重要修改

@TRANSFORMS.register_module()
class LoadAllCameraImageFromFile:
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
        """
        读取6个相机视角的图片，将它们拼接成2×3的大图。
        
        拼接顺序:
         - 第一行: 前左（index=2），前（index=0），前右（index=1）
         - 第二行: 后左（index=4），后（index=3），后右（index=5）
         
        :param results: 包含 'img_filename'（长度为6的list）的字典
        :return: 更新后的 results，'img' 为拼接后的大图
        """
        
        img_paths = results['img_filename']  # 这里是长度为6的列表
        # 我们希望的顺序（根据你的描述和给定index的含义）
        reorder_index = [2, 0, 1, 4, 3, 5]
        
        # 依次读取6张图像，并检查尺寸
        imgs = []
        expected_shape = (900, 1600, 3)  # (height, width, channels)
        for idx in range(len(img_paths)):
            one_img_filename = img_paths[idx]
            
            # 如果文件名以 '.' 或 'data/' 开头，则直接使用它，否则拼接 img_root
            if one_img_filename.startswith('.') or one_img_filename.startswith('data/'):
                one_img_path = one_img_filename
            else:
                one_img_path = os.path.join(self.img_root, one_img_filename)
            
            # 读取图像
            if self.file_client_args['backend'] == 'petrel' and self.file_client is not None:
                img_bytes = self.file_client.get(one_img_path)
                img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            else:
                img = mmcv.imread(one_img_path, self.color_type)
            
            if self.to_float32:
                img = img.astype(np.float32)
                
            # 检查图像尺寸是否为 1600x900
            if img.shape != expected_shape:
                raise RuntimeError(
                    f"Image shape {img.shape} != expected {expected_shape} "
                    f"for file {one_img_path}."
                )
            
            imgs.append(img)
        
        # 现在根据指定顺序重排 [2, 0, 1, 4, 3, 5]
        reordered_imgs = [imgs[i] for i in reorder_index]
        
        # 分别拼接第一行(前左, 前, 前右)和第二行(后左, 后, 后右)
        row1 = np.concatenate([reordered_imgs[0], reordered_imgs[1], reordered_imgs[2]], axis=1)
        row2 = np.concatenate([reordered_imgs[3], reordered_imgs[4], reordered_imgs[5]], axis=1)
        
        # 再把两行竖向拼起来
        big_img = np.concatenate([row1, row2], axis=0)
        
        # 将结果保存到 results
        # filename 可以选择保留为原先的全部文件名列表
        results['filename'] = front_img_filename = results['img_filename'][0]
        # 替换为拼接后的大图
        
        results['img'] = big_img
        for fname in results.get('img_filename', []):
            # 这里通过判断路径中是否含有 '/CAM_FRONT/' 和 '__CAM_FRONT__' 来确定前摄像头图像
            if '/CAM_FRONT/' in fname and '__CAM_FRONT__' in fname:
                cam_front_file = fname
                break
        base_name = os.path.basename(cam_front_file)
        name, ext = os.path.splitext(base_name)
        
        results.update({
        'save_name': name , # 新增 BEV 2D 框参数
        'ext': ext
        })
        visualize_big_img(results)
        results['img_shape'] = big_img.shape
        results['ori_shape'] = big_img.shape
        results['pad_shape'] = big_img.shape
        results['scale_factor'] = 1.0
        
        num_channels = 1 if len(big_img.shape) < 3 else big_img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False
        )
        
        return results



    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(to_float32={self.to_float32}, "
                f"color_type='{self.color_type}', img_root='{self.img_root}')")

def visualize_big_img(results):
    """
    可视化 results['img'] 大图，并将其保存到指定路径。
    """
    # 从 results 中取得大图
    big_img = results['img']
    
    # 如果你读入的是 BGR 通道，为保证可视化颜色正常，可转换为 RGB
    big_img_rgb = mmcv.bgr2rgb(big_img)
    big_img_rgb = big_img_rgb / 255.0
    # 显示合并后的大图
    plt.imshow(big_img_rgb)
    plt.title('Merged 6-View Image')
    plt.axis('off')
    plt.show()
    
    # 保存到指定目录
    save_dir = "/home/UNT/yz0370/projects/GiT/visualization/all_6_img"
    os.makedirs(save_dir, exist_ok=True)  # 若不存在就创建
    
    name = results['save_name']
    ext = results['ext']
    new_name = name + 'merged_6_view' + ext        
    # 设置保存文件名，这里示例写为 merged_6_view.jpg
    save_path = os.path.join(save_dir, new_name)
    
    # 注意：mmcv.imwrite 默认是 BGR，如果希望直接用原图保存，可以用 big_img
    mmcv.imwrite(big_img, save_path)
    # print(f"合并后的大图已保存至: {save_path}")
    plt.close()