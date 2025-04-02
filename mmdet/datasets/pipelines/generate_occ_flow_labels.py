import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import os

from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import calculate_birds_eye_view_parameters

@TRANSFORMS.register_module()
class GenerateOccFlowLabels(BaseTransform):
    """Generate occupancy (and flow) labels from multi-frame 3D annotations.
    
    模仿 UniAD 的 occflow 标签生成逻辑，计算 BEV 参数并生成 BEV 分割图。
    
    Args:
        grid_conf (dict): BEV 网格配置，例如 {'xbound': [-50, 50, 0.5], ...}
        ignore_index (int): 忽略标签的索引。
        only_vehicle (bool): 是否仅生成车辆的标签。
        filter_invisible (bool): 是否过滤掉不可见的目标。
        deal_instance_255 (bool): 备用参数（目前不启用）。
    """
    def __init__(self,
             grid_conf,
             ignore_index,
             only_vehicle,
             filter_invisible,
             deal_instance_255=False):
        self.grid_conf = grid_conf
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible
        self.deal_instance_255 = deal_instance_255
        # print(f"[DEBUG] GenerateOccFlowLabels initialized with grid_conf: {grid_conf}, ignore_index: {ignore_index}")
        
        # 调用项目中已有的 calculate_birds_eye_view_parameters
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        )
        
        # Convert torch.Tensor to NumPy arrays if necessary
        self.bev_resolution = self.bev_resolution.cpu().numpy().astype(np.float32)
        self.bev_start_position = self.bev_start_position.cpu().numpy().astype(np.float32)
        self.bev_dimension = self.bev_dimension.cpu().numpy().astype(np.int32)
        
        # 类别过滤（这里仅举例车辆类别，你可以根据需要修改）
        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']
        if only_vehicle:
            self.filter_cls_ids = np.array([nusc_classes.index(cls) for cls in vehicle_classes])
        else:
            self.filter_cls_ids = np.arange(len(nusc_classes))

    
    def transform(self, results: dict) -> dict:
        print("[DEBUG] GenerateOccFlowLabels: Received results keys:", list(results.keys()))
        
        # 这里你可以根据检测框等信息生成 BEV 标签，以下仅生成一个全0的 dummy 标签
        bev_width, bev_height = self.bev_dimension
        gt_segmentation = torch.zeros((bev_height, bev_width), dtype=torch.long)
        
        results['gt_segmentation'] = gt_segmentation
        print("[DEBUG] GenerateOccFlowLabels: Set gt_segmentation with shape", gt_segmentation.shape, list(results.keys()))
        
        return results
  


    def reframe_boxes(self, boxes, t_init, t_curr):
        """
        将当前帧的 3D 检测框（boxes）从当前坐标系转换到参考帧坐标系。

        参数：
            boxes: 检测框数据，要求最终为支持 rotate 和 translate 方法的对象，
                如果传入的是 NumPy 数组，则先转换为 BaseInstance3DBoxes 对象。
            t_init (dict): 参考帧的转换信息，包含：
                - 'l2e_r': 参考帧的 lidar→ego 旋转矩阵
                - 'l2e_t': 参考帧的 lidar→ego 平移向量
                - 'e2g_r': 参考帧的 ego→global 旋转矩阵
                - 'e2g_t': 参考帧的 ego→global 平移向量
            t_curr (dict): 当前帧的转换信息（同上）。

        返回：
            boxes: 转换到参考帧坐标系下的 boxes 对象。
        """

        # --- 第一步：从当前 ego 坐标系转换到全局坐标系 ---
        # 当前帧的 lidar→ego 转换
        l2e_r_curr = t_curr['l2e_r']
        l2e_t_curr = t_curr['l2e_t']
        boxes.rotate(l2e_r_curr.T)  # 假设 rotate 为右乘，故传入转置矩阵
        boxes.translate(l2e_t_curr)
        # 当前帧的 ego→global 转换
        e2g_r_curr = t_curr['e2g_r']
        e2g_t_curr = t_curr['e2g_t']
        boxes.rotate(e2g_r_curr.T)
        boxes.translate(e2g_t_curr)

        # --- 第二步：从全局坐标系转换到参考帧的 ego 坐标系 ---
        # 参考帧的 ego→global 转换的逆变换
        e2g_r_init = t_init['e2g_r']
        e2g_t_init = t_init['e2g_t']
        boxes.translate(-e2g_t_init)                # 先逆平移
        inv_e2g_r_init = np.linalg.inv(e2g_r_init)    # 计算逆旋转矩阵
        boxes.rotate(inv_e2g_r_init.T)              # 再逆旋转

        # 参考帧的 lidar→ego 转换的逆变换
        l2e_r_init = t_init['l2e_r']
        l2e_t_init = t_init['l2e_t']
        boxes.translate(-l2e_t_init)                # 先逆平移
        inv_l2e_r_init = np.linalg.inv(l2e_r_init)    # 计算逆旋转矩阵
        boxes.rotate(inv_l2e_r_init.T)              # 再逆旋转

        return boxes



    def __call__(self, results: dict) -> dict:
        """
        根据未来帧的 3D 检测标注信息和变换矩阵，生成 BEV 分割、实例、中心、偏移、流等标签。
        （这里的 gt_segmentation 即 occ 分割图 gt_occ_seg）
        """
        # 用于将 ignore_index 转换为特殊标记
        SPECIAL_INDEX = -20
        # 提取未来帧的检测信息
        all_gt_bboxes_3d = results.get('future_gt_bboxes_3d', None)
        all_gt_labels_3d = results.get('future_gt_labels_3d', None)
        all_gt_inds = results.get('future_gt_inds', None)
        all_vis_tokens = results.get('future_gt_vis_tokens', None)
        num_frames = len(all_gt_bboxes_3d)
        
        # 提取变换矩阵
        l2e_r_mats = results.get('occ_l2e_r_mats', None)
        l2e_t_vecs = results.get('occ_l2e_t_vecs', None)
        e2g_r_mats = results.get('occ_e2g_r_mats', None)
        e2g_t_vecs = results.get('occ_e2g_t_vecs', None)

        # 参考帧（取序列第一帧）
        t_ref = {
            'l2e_r': l2e_r_mats[0],
            'l2e_t': l2e_t_vecs[0],
            'e2g_r': e2g_r_mats[0],
            'e2g_t': e2g_t_vecs[0],
        }

        segmentations = []   # 存放每帧生成的 BEV 分割图
        instances = []       # 存放每帧生成的实例图
        gt_future_boxes = [] # 存放转换后每帧的检测框
        gt_future_labels = []# 存放每帧对应的类别信息

        # 这里只处理一帧数据（假设 i=0）
        i = 0
        gt_bboxes_3d = all_gt_bboxes_3d[i]
        gt_labels_3d = all_gt_labels_3d[i]
        ins_inds = all_gt_inds[i]
        vis_tokens = all_vis_tokens[i] if all_vis_tokens is not None else None

        t_curr = {
            'l2e_r': l2e_r_mats[i],
            'l2e_t': l2e_t_vecs[i],
            'e2g_r': e2g_r_mats[i],
            'e2g_t': e2g_t_vecs[i],
        }
        # 转换 3D 框到参考帧
        ref_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr)
        gt_future_boxes.append(ref_bboxes_3d)
        gt_future_labels.append(gt_labels_3d)

        # 初始化 BEV 分割图和实例图
        segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int32)
        instance = np.zeros((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int32)

        if self.only_vehicle:
            vehicle_mask = np.isin(gt_labels_3d, self.filter_cls_ids)
            ref_bboxes_3d = ref_bboxes_3d[vehicle_mask]
            gt_labels_3d = gt_labels_3d[vehicle_mask]
            ins_inds = ins_inds[vehicle_mask]
            if vis_tokens is not None:
                vis_tokens = vis_tokens[vehicle_mask]

        if self.filter_invisible and vis_tokens is not None:
            visible_mask = (vis_tokens != 1)  # vis_tokens==1 表示不可见
            ref_bboxes_3d = ref_bboxes_3d[visible_mask]
            gt_labels_3d = gt_labels_3d[visible_mask]
            ins_inds = ins_inds[visible_mask]

        # 将 3D 框转换为 BEV 下的 2D 框参数 [cx, cy, w, h, theta]
        if len(ref_bboxes_3d.tensor) > 0:
            # 改为使用过滤后的 ref_bboxes_3d
            current_boxes = ref_bboxes_3d  
            
            boxes_np = (
                current_boxes.tensor.cpu().numpy()
                
            )
          
            # 转换 3D 框到 BEV 2D 框
            bev_boxes = np.zeros((boxes_np.shape[0], 5), dtype=np.float32)
            bev_boxes[:, 0] = (boxes_np[:, 0] - self.bev_start_position[0]) / self.bev_resolution[0]  # cx
            bev_boxes[:, 1] = (boxes_np[:, 1] - self.bev_start_position[1]) / self.bev_resolution[1]  # cy
            bev_boxes[:, 2] = boxes_np[:, 3] / self.bev_resolution[0]  # w
            bev_boxes[:, 3] = boxes_np[:, 4] / self.bev_resolution[1]  # h
            bev_boxes[:, 4] = boxes_np[:, 6]  # theta
            # 1. 将 theta 从弧度转换为角度，并归一化到 [0, 360]
            theta_deg = np.degrees(bev_boxes[:, 4]) % 360
            bev_boxes[:, 4] = theta_deg

            # 2. 归一化各列到 [0, 1]
            #   - cx: 除以 BEV 图像宽度
            #   - cy: 除以 BEV 图像高度
            #   - w:  除以 BEV 图像宽度
            #   - h:  除以 BEV 图像高度
            #   - theta: 除以 360
            bev_boxes[:, 0] = bev_boxes[:, 0] / self.bev_dimension[0]  # normalized cx
            bev_boxes[:, 1] = bev_boxes[:, 1] / self.bev_dimension[1]  # normalized cy
            bev_boxes[:, 2] = bev_boxes[:, 2] / self.bev_dimension[0]  # normalized width
            bev_boxes[:, 3] = bev_boxes[:, 3] / self.bev_dimension[1]  # normalized height
            bev_boxes[:, 4] = bev_boxes[:, 4] / 360.0   
           
            bbox_corners = ref_bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy()
            bbox_corners = np.round(
                (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0)
                / self.bev_resolution[:2]
            ).astype(np.int32)
            
            for j, gt_ind in enumerate(ins_inds):
                if gt_ind == self.ignore_index:
                    gt_ind = SPECIAL_INDEX
                poly_region = bbox_corners[j]
                cv2.fillPoly(segmentation, [poly_region], 1)
                cv2.fillPoly(instance, [poly_region], int(gt_ind))
            segmentations.append(segmentation)
            instances.append(instance)

        gt_instances = build_gt_instances_from_bev_bbox(bev_boxes, 1)
        # 更新 results，只保留 gt_2dbbox 信息
        bev_boxes = torch.from_numpy(bev_boxes).to(current_boxes.tensor.device)
        results.update({
            'bev_bbox_gt': bev_boxes , # 新增 BEV 2D 框参数
            'gt_instances': gt_instances
        })
        
        

        
        ###可视化
        segmentations = torch.from_numpy(np.stack(segmentations, axis=0)).long()
        # segmentations = generate_all_ones(segmentations)
        flip_segmentation = flip_segmentations(segmentations)
        flip_masked_segmentation = flip_segmentation
        generateoccflowlabels_visualization(results, flip_segmentation, flip_masked_segmentation, segmentations)
        # ----------------- 新增部分：可视化 gt_2dbbox -----------------
        # 从 results['img_filename'] 中获取 CAM_FRONT 的图像文件名
        cam_front_file = None
        for fname in results.get('img_filename', []):
            # 这里通过判断路径中是否含有 '/CAM_FRONT/' 和 '__CAM_FRONT__' 来确定前摄像头图像
            if '/CAM_FRONT/' in fname and '__CAM_FRONT__' in fname:
                cam_front_file = fname
                break

        if cam_front_file is not None and bev_boxes.numel() > 0:
            name = results['save_name']
            ext = results['ext']
            new_name = name + 'occflow' + ext 
            vis_save_path = os.path.join('/home/UNT/yz0370/projects/GiT/visualization/occ_2d_box', new_name)
            # 调用可视化函数
            visualize_gt_2dbbox(bev_boxes, self.bev_dimension, vis_save_path)
        else:
            print("未在 results['img_filename'] 中找到 CAM_FRONT 图像或无有效 2D 框信息。")
        # --------------------------------------------------------------

        return results


from mmengine.structures import InstanceData
import torch

def build_gt_instances_from_bev_bbox(bev_bbox_gt: torch.Tensor, label: int) -> InstanceData:
    """
    Construct ground truth instances (gt_instances) from BEV occupancy bbox ground truth.
    
    Args:
        bev_bbox_gt (Tensor): A tensor of shape (N, 5) containing BEV occupancy bbox ground truth,
                              where each bbox is represented as [cx, cy, w, h, theta] and values are normalized.
        label (int): The class label for all bounding boxes (e.g., vehicle class id).
        
    Returns:
        InstanceData: An object that contains at least:
            - bboxes (Tensor): The original bev_bbox_gt tensor.
            - labels (Tensor): A tensor of shape (N,) with all values set to the given label.
    """
     # 如果输入是 numpy 数组，先转换为 tensor
    if isinstance(bev_bbox_gt, np.ndarray):
        bev_bbox_gt = torch.from_numpy(bev_bbox_gt)
    num_bboxes = bev_bbox_gt.size(0)
    # 创建一个标签张量，所有 bbox 都属于同一类别
    labels = torch.full((num_bboxes,), label, dtype=torch.long, device=bev_bbox_gt.device)
    
    # 构造 InstanceData 对象，包含 bboxes 和 labels
    gt_instances = InstanceData(bboxes=bev_bbox_gt, labels=labels)
    return gt_instances


import cv2
import numpy as np

def visualize_gt_2dbbox(gt_2dbbox, bev_dimension, save_path):
    """
    可视化归一化后的 gt_2dbbox，在 BEV 图像上绘制所有 2D 检测框，并保存到指定路径。
    
    参数:
      gt_2dbbox: numpy 数组，形状为 [N, 5]，每一行为 [norm_cx, norm_cy, norm_w, norm_h, norm_theta]，
                 其中归一化后的 norm_cx, norm_cy 分别为 BEV 图像中心的比例（0~1），
                 norm_w, norm_h 为检测框宽高相对于 BEV 图像尺寸的比例，
                 norm_theta 为检测框角度归一化后的值（对应 0~360 度）。
      bev_dimension: (width, height) 表示 BEV 图像尺寸（像素）
      save_path: 生成的可视化图像保存路径
    """
    width, height = bev_dimension[:2]
    # 创建一张白色背景图像
    bev_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    for box in gt_2dbbox:
        norm_cx, norm_cy, norm_w, norm_h, norm_theta = box
        # 反归一化到像素坐标
        cx = norm_cx * width
        cy = norm_cy * height
        w  = norm_w * width
        h  = norm_h * height
        # 反归一化 theta: norm_theta * 360 得到角度，再转换为弧度
        theta = np.deg2rad(norm_theta * 360)

        # 计算半宽、半高
        hw, hh = w / 2.0, h / 2.0
        # 定义未旋转的角点（相对于中心）
        corners = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh]
        ])
        # 构造旋转矩阵（theta 为弧度）
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        # 旋转角点并平移到 BEV 图像坐标
        rotated_corners = np.dot(corners, R.T)
        corners_absolute = rotated_corners + np.array([cx, cy])
        pts = corners_absolute.astype(np.int32).reshape((-1, 1, 2))
        # 绘制红色边框，线宽设置为1
        cv2.polylines(bev_img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        # 绘制蓝色中心点，半径设置为1
        center_pt = (int(cx), int(cy))
        cv2.circle(bev_img, center_pt, radius=1, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(save_path, bev_img)
    # print(f"Visualization saved at {save_path}")


def generateoccflowlabels_visualization(results, flip_segmentation, flip_masked_segmentation, segmentations):
    """
    可视化 occ flow labels。

    该函数首先从 results['img_filename'] 中获取六个摄像头的图片路径，
    加载这六张图片；然后将传入的三个 segmentation（翻转后的、翻转后的掩码处理后的、原始）
    转换为灰度图（0/1 二值图乘以 255），并在每幅分割图中心画一个小正方形，
    最后将六张摄像头图片（以两行三列排列）以及三个 segmentation 图（排列在下方）
    组合到一张图上，并保存到目录：
    /home/UNT/yz0370/projects/GiT/visualization/generateoccflow/
    """
    # 从 results 中获取六个摄像头的图片路径
    img_filenames = results.get('img_filename', [])
    if len(img_filenames) != 6:
        print("Warning: img_filename 中图片数量不是6！")
    
    cam_imgs = []
    for path in img_filenames:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 若图片加载失败，则使用一个空白图片
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        cam_imgs.append(img)
    
    # 辅助函数：将 segmentation 转换为 numpy 灰度图，并在中心绘制小正方形
    def get_segmentation_image(seg):
        if isinstance(seg, torch.Tensor):
            seg_np = seg[0].cpu().numpy().astype(np.uint8) * 255
        else:
            seg_np = seg[0].astype(np.uint8) * 255
        
        # 在分割图中心画一个小正方形
        H, W = seg_np.shape
        square_size = 20  # 正方形边长
        center_x, center_y = W // 2, H // 2
        top_left = (center_x - square_size // 2, center_y - square_size // 2)
        bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
        cv2.rectangle(seg_np, top_left, bottom_right, color=127, thickness=2)
        
        return seg_np
    
    # 转换三个 segmentation 图
    flip_seg_img = get_segmentation_image(flip_segmentation)
    flip_masked_seg_img = get_segmentation_image(flip_masked_segmentation)
    orig_seg_img = get_segmentation_image(segmentations)
    
    # 创建 3行3列的子图布局：上两行显示六个摄像头图片，第三行显示三个分割图
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # 定义新的显示顺序，对应的索引为：
    # CAM_FRONT_LEFT: 原索引 2
    # CAM_FRONT:      原索引 0
    # CAM_FRONT_RIGHT:原索引 1
    # CAM_BACK_LEFT:  原索引 4
    # CAM_BACK:       原索引 3
    # CAM_BACK_RIGHT: 原索引 5
    order = [2, 0, 1, 4, 3, 5]

    # 显示六张摄像头图片（按照新的顺序）
    for i in range(6):
        idx = order[i]
        axs[i].imshow(cam_imgs[idx])
        axs[i].set_title(os.path.basename(img_filenames[idx]))
        axs[i].axis('off')

    
    # 显示三个 segmentation 图
    axs[6].imshow(flip_seg_img, cmap='gray')
    axs[6].set_title("Flipped OCC Segmentation")
    axs[6].axis('off')
    
    axs[7].imshow(flip_masked_seg_img, cmap='gray')
    axs[7].set_title("Flipped Masked OCC Segmentation")
    axs[7].axis('off')
    
    axs[8].imshow(orig_seg_img, cmap='gray')
    axs[8].set_title("Original OCC Segmentation")
    axs[8].axis('off')
    
    plt.tight_layout()
    save_dir = "/home/UNT/yz0370/projects/GiT/visualization/generateoccflow"
    os.makedirs(save_dir, exist_ok=True)
    
    # 从 results['filename'] 中提取文件名并去除扩展名
    if 'filename' in results:
        base_name = os.path.basename(results['filename'])
        name_without_ext, _ = os.path.splitext(base_name)
    else:
        name_without_ext = "generateoccflowlabels_visualization"
    
    # 用提取出的文件名生成保存路径
    save_path = os.path.join(save_dir, f"{name_without_ext}_generateoccflowlabels_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)
    # print(f"Saved generateoccflowlabels_visualization to {save_path}")



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_all_ones(segmentations):
    """
    生成一个与 segmentations 形状一致的全1矩阵。

    参数:
        segmentations: 分割图数据，类型可以是 torch.Tensor 或 numpy.ndarray。
    返回:
        与 segmentations 形状一致的全1矩阵，类型与输入保持一致。
    """
    if isinstance(segmentations, torch.Tensor):
        return torch.ones_like(segmentations)
    else:
        return np.ones_like(segmentations)


def flip_segmentations(segmentations):
    """
    将输入的 segmentation 上下（垂直）翻转。
    
    Args:
        segmentations: torch.Tensor 或 numpy.ndarray，
            若为 tensor，形状一般为 (N, H, W) 或 (H, W)；
            若为 numpy 数组，同理。
    
    Returns:
        翻转后的 segmentation，与输入类型一致。
    """
    if isinstance(segmentations, torch.Tensor):
        # 对于 tensor，假定高度在倒数第二个维度（即 dim=-2）
        return torch.flip(segmentations, dims=[-2])
    elif isinstance(segmentations, np.ndarray):
        if segmentations.ndim == 3:
            # 若为 (N, H, W)，翻转 axis=1
            return np.flip(segmentations, axis=1)
        elif segmentations.ndim == 2:
            # 若为 (H, W)，直接使用 np.flipud
            return np.flipud(segmentations)
        else:
            raise ValueError("Unsupported numpy array shape for segmentation.")
    else:
        raise TypeError("Unsupported type for segmentations. Expected torch.Tensor or numpy.ndarray.")
