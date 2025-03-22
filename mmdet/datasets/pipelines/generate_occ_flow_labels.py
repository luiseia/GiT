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
        vehicle_classes = ['car', 'bus', 'construction_vehicle', 'truck', 'trailer']
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

    def visualize_segmentation(self, segmentation: np.ndarray, save_path: str = 'segmentation.png') -> None:
            """
            使用 matplotlib 显示并保存分割图像。
            
            Args:
                segmentation (np.ndarray): 2D 分割图，像素值为整数（例如 0/1 或其他类别索引）。
                save_path (str): 保存图像的文件路径，默认 'segmentation.png'。
            """
            plt.figure(figsize=(8, 8))
            # 如果 segmentation 中类别较少，可用 'jet'、'nipy_spectral' 或 'gray' 等 cmap，根据需要调整
            plt.imshow(segmentation, cmap='gray')
            plt.title("BEV Segmentation")
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()


    def apply_front_camera_mask(self, occ_seg: torch.Tensor) -> torch.Tensor:
        """
        对 occ_seg（尺寸形如 (N, H, W)）应用梯形掩码，仅保留前摄像头对应的区域，
        将梯形外的部分置为未占用（这里用 0 表示未占用）。
        
        Args:
            occ_seg (torch.Tensor): occ 分割图，形状 (num_frames, H, W)
            
        Returns:
            torch.Tensor: 掩码处理后的 occ 分割图，形状不变。
        """
        # 获取尺寸信息
        N, H, W = occ_seg.shape
        
        # 设计梯形区域（换到上方）：
        #   - 顶边覆盖整个宽度（[0, W-1]），对应车辆正前方近处；
        #   - 底边较窄（例如位于宽度的 25% ~ 75% 区间），对应远处
        pts = np.array([
            [0, 0],                     # 顶左：图像上边缘左侧
            [W - 1, 0],                 # 顶右：图像上边缘右侧
            [int(W * 0.75), int(H * 0.5)],  # 底右：位于图像中部偏右
            [int(W * 0.25), int(H * 0.5)]   # 底左：位于图像中部偏左
        ], dtype=np.int32)
        
        # 在空白图像上填充梯形区域（值为1，其余区域为0）
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        
        # 将 numpy mask 转换为 tensor，并扩展到所有帧上
        mask_tensor = torch.from_numpy(mask).to(occ_seg.device)
        mask_tensor = mask_tensor.unsqueeze(0).expand(N, -1, -1)
        
        # 将 occ_seg 中 mask 外部置 0（未占用），mask 内保持原值
        occ_seg_masked = occ_seg * mask_tensor
        return occ_seg_masked


    def __call__(self, results: dict) -> dict:
        """
        根据未来帧的 3D 检测标注信息和变换矩阵，生成 BEV 分割、实例、中心、偏移、流等标签。
        （这里的 gt_segmentation 即 occ 分割图 gt_occ_seg）
        """
        # 用于将 ignore_index 转换为特殊标记
        SPECIAL_INDEX = -20
        # print(results.keys())
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

        segmentations = []  # 存放每帧生成的 BEV 分割图
        instances = []      # 存放每帧生成的实例图
        gt_future_boxes = []   # 存放转换后每帧的检测框
        gt_future_labels = []  # 存放每帧对应的类别信息

        for i in range(num_frames):
            gt_bboxes_3d = all_gt_bboxes_3d[i]
            gt_labels_3d = all_gt_labels_3d[i]
            ins_inds = all_gt_inds[i]
            vis_tokens = all_vis_tokens[i] if all_vis_tokens is not None else None

            if gt_bboxes_3d is None:
                segmentation = np.ones((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int64) * self.ignore_index
                instance = np.ones((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int64) * self.ignore_index
            else:
                t_curr = {
                    'l2e_r': l2e_r_mats[i],
                    'l2e_t': l2e_t_vecs[i],
                    'e2g_r': e2g_r_mats[i],
                    'e2g_t': e2g_t_vecs[i],
                }
                ref_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr)
                gt_future_boxes.append(ref_bboxes_3d)
                gt_future_labels.append(gt_labels_3d)

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

                if len(ref_bboxes_3d.tensor) > 0:
                    # 计算检测框在 BEV 下的 2D 坐标
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
                else:
                    segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int32) * self.ignore_index
                    instance = np.zeros((self.bev_dimension[1], self.bev_dimension[0]), dtype=np.int32) * self.ignore_index
            
            if 4 > i >= 1:
                # print(f"[DEBUG] Visualizing segmentation for frame {i}")
                self.visualize_segmentation(segmentation, save_path=f'segmentation_frame_{i}.png')

            segmentations.append(segmentation)
            instances.append(instance)

        # 将所有帧的 segmentation 和 instance 堆叠后转换为 tensor
        segmentations = torch.from_numpy(np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()

        instance_centerness, instance_offset, instance_flow, instance_backward_flow = self.center_offset_flow(
            instances, all_gt_inds, ignore_index=self.ignore_index, sigma=3.0
        )

        invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
        instance_centerness[invalid_mask] = self.ignore_index


        # 添加当前帧的所有3d box参数文本格式
        # 这里我们假设当前帧为 gt_future_boxes 列表中的第 0 帧
        if len(gt_future_boxes) > 0:
            current_boxes = gt_future_boxes[0]
            # 如果 current_boxes 有 tensor 属性，则提取其中的数值，
            # 这里假设 current_boxes.tensor 是一个 [N, 7] 的数组，表示每个 3d box 的参数
            if hasattr(current_boxes, 'tensor'):
                boxes_np = current_boxes.tensor.cpu().numpy() if isinstance(current_boxes.tensor, torch.Tensor) else current_boxes.tensor
                # 格式化每个 box 的参数为一行，参数之间用逗号分隔，每个参数保留两位小数
                boxes_text = "\n".join([", ".join([f"{val:.2f}" for val in box]) for box in boxes_np])
            else:
                boxes_text = str(current_boxes)
        else:
            boxes_text = "No 3d boxes available"
        # 将生成的文本添加到 results 中，键名可自定义，比如 "3dbox_text"
        results['3dbox_text'] = boxes_text

        # 关键修改部分：
        # 使用前摄像头对应的梯形掩码对生成的 occ 分割图（即 gt_occ_seg）进行处理，
        # 将不在梯形内的区域置为 0（未占用）
        results['filename'] = '1111.jpg'
        segmentations = generate_all_ones(segmentations)
        flip_segmentation = flip_segmentations(segmentations)
        flip_masked_segmentation = self.apply_front_camera_mask(flip_segmentation)
        
        
        
        generateoccflowlabels_visualization(results, flip_segmentation, flip_masked_segmentation, segmentations)

        
        # 更新 results，注意这里暂时将 gt_segmentation 设置为处理后的 occ 分割图
        results['gt_occ_has_invalid_frame'] = results.pop('occ_has_invalid_frame', None)
        results['gt_occ_img_is_valid'] = results.pop('occ_img_is_valid', None)
        results.update({
            'gt_segmentation': flip_masked_segmentation,
            'gt_instance': instances,
            'gt_centerness': instance_centerness,
            'gt_offset': instance_offset,
            'gt_flow': instance_flow,
            'gt_backward_flow': instance_backward_flow,
            'gt_future_boxes': gt_future_boxes,
            'gt_future_labels': gt_future_labels
        })
        return results



    def center_offset_flow(self, instance_img, all_gt_inds, ignore_index=255, sigma=3.0):
        """
        根据 BEV 下的实例图计算目标中心热图、像素偏移、前向流（future displacement）以及后向流（backward flow）。

        Args:
            instance_img (torch.Tensor): 实例图，形状为 (seq_len, H, W)，其中每个像素的值为实例 id。
            all_gt_inds (list): 长度为 seq_len 的列表，每个元素为当前帧目标实例 id 的数组（或 list）。
            ignore_index (int): 忽略的标签值，默认 255。
            sigma (float): 高斯扩散参数，用于计算中心热图，默认 3.0。

        Returns:
            tuple: (center_label, offset_label, future_displacement_label, backward_flow)
                - center_label: torch.Tensor，形状为 (seq_len, 1, H, W)，表示目标中心处的高响应。
                - offset_label: torch.Tensor，形状为 (seq_len, 2, H, W)，表示每个像素到目标中心的横纵向偏移。
                - future_displacement_label: torch.Tensor，形状为 (seq_len, 2, H, W)，表示前一帧到当前帧目标中心的位移。
                - backward_flow: torch.Tensor，形状为 (seq_len, 2, H, W)，表示当前帧到前一帧目标中心的位移（与 future_displacement 方向相反）。
        """
        seq_len, H, W = instance_img.shape
        device = instance_img.device

        # 初始化输出 tensor，注意这里所有值初始化为 ignore_index（或0）
        center_label = torch.zeros(seq_len, 1, H, W, device=device)
        offset_label = ignore_index * torch.ones(seq_len, 2, H, W, device=device, dtype=torch.float)
        future_disp_label = ignore_index * torch.ones(seq_len, 2, H, W, device=device, dtype=torch.float)
        backward_flow = ignore_index * torch.ones(seq_len, 2, H, W, device=device, dtype=torch.float)

        # 生成坐标网格：x 表示行索引（垂直坐标），y 表示列索引（水平坐标）
        x = torch.arange(H, dtype=torch.float, device=device).view(H, 1).expand(H, W)
        y = torch.arange(W, dtype=torch.float, device=device).view(1, W).expand(H, W)

        # 收集所有帧中出现的目标 id
        all_ids = []
        for inds in all_gt_inds:
            if inds is not None:
                # 若 inds 为 tensor，则转换为 list
                if isinstance(inds, torch.Tensor):
                    all_ids.extend(inds.tolist())
                else:
                    all_ids.extend(inds)
        unique_ids = np.unique(np.array(all_ids))

        # 对每个独立的实例 id 进行处理
        for instance_id in unique_ids:
            instance_id = int(instance_id)
            prev_xc = None
            prev_yc = None
            prev_mask = None
            for t in range(seq_len):
                # 得到当前帧中该实例的掩码，形状 (H, W)
                instance_mask = (instance_img[t] == instance_id)
                if instance_mask.sum() == 0:
                    # 当前帧中不存在该实例，则重置之前的中心信息
                    prev_xc = None
                    prev_yc = None
                    prev_mask = None
                    continue

                # 计算当前帧中该实例在 BEV 图像下的中心（均值坐标）
                xc = x[instance_mask].mean()
                yc = y[instance_mask].mean()

                # 计算偏移图：每个像素到目标中心的距离
                off_x = xc - x
                off_y = yc - y

                # 计算高斯响应作为中心热图
                g = torch.exp(-((off_x ** 2 + off_y ** 2) / (sigma ** 2)))
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)

                # 对于属于该目标的像素，更新偏移值
                offset_label[t, 0][instance_mask] = off_x[instance_mask]
                offset_label[t, 1][instance_mask] = off_y[instance_mask]

                # 若在上一帧中也存在该实例，则计算帧间位移
                if prev_xc is not None and prev_mask is not None:
                    delta_x = xc - prev_xc
                    delta_y = yc - prev_yc
                    # 将上一帧中该实例的像素，赋值为从上一帧到当前帧的位移
                    future_disp_label[t - 1, 0][prev_mask] = delta_x
                    future_disp_label[t - 1, 1][prev_mask] = delta_y
                    # 后向流为负值
                    backward_flow[t - 1, 0][instance_mask] = -delta_x
                    backward_flow[t - 1, 1][instance_mask] = -delta_y

                # 更新上一帧的中心信息
                prev_xc = xc
                prev_yc = yc
                prev_mask = instance_mask

        return center_label, offset_label, future_disp_label, backward_flow

    def visualize_instances(self, instances, vis_root=''):
        """
        可视化生成的实例分割图，将每一帧的实例图保存为彩色图像，并合成视频便于查看调试。

        Args:
            instances (list or np.ndarray): 每个元素为一帧的实例分割图（二维 numpy 数组）。
            vis_root (str): 保存可视化结果的目录路径。如果为空字符串，则不保存。
        """
        import os
        import cv2

        # 如果指定了保存目录，则创建目录
        if vis_root is not None and vis_root != '':
            os.makedirs(vis_root, exist_ok=True)

        # 遍历每一帧，将实例图保存为彩色图像
        for i, ins in enumerate(instances):
            # 确保输入为 uint8 类型
            ins_uint8 = ins.astype(np.uint8)
            # 应用 COLORMAP_JET 颜色映射，将灰度图转换为彩色图
            colored_ins = cv2.applyColorMap(ins_uint8, cv2.COLORMAP_JET)
            save_path = os.path.join(vis_root, f'{i}.png')
            cv2.imwrite(save_path, colored_ins)

        # 合成视频：如果实例图非空，则将所有帧合成为视频保存
        if len(instances) > 0:
            # 假设所有帧尺寸相同，取第一帧的尺寸 (height, width)
            height, width = instances[0].shape
            video_path = os.path.join(vis_root, 'instances_video.avi')
            # 注意：cv2.VideoWriter 需要视频尺寸格式为 (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            fps = 4  # 设定帧率，可根据需要调整
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for ins in instances:
                ins_uint8 = ins.astype(np.uint8)
                colored_ins = cv2.applyColorMap(ins_uint8, cv2.COLORMAP_JET)
                video_writer.write(colored_ins)
            video_writer.release()


    def __repr__(self):
        return (f"{self.__class__.__name__}(grid_conf={self.grid_conf}, ignore_index={self.ignore_index}, "
                f"only_vehicle={self.only_vehicle}, filter_invisible={self.filter_invisible}, "
                f"deal_instance_255={self.deal_instance_255})")
    

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    print(f"Saved generateoccflowlabels_visualization to {save_path}")


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
