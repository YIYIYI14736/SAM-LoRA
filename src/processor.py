from src.segment_anything.utils.transforms import ResizeLongestSide  # 导入 ResizeLongestSide，用于调整图像大小
from src.lora import LoRA_sam  # 导入 LoRA_sam 模型
import numpy as np  # 导入 NumPy，用于数值计算
import torch  # 导入 PyTorch，用于深度学习计算
import PIL  # 导入 PIL，用于图像处理
from typing import Optional, Tuple  # 导入可选和元组类型提示

class Samprocessor:
    def __init__(self, sam_model: LoRA_sam, default_prompt_mode: str = "pt_box"):

        super().__init__()  # 初始化父类
        self.model = sam_model  # 保存传入的 SAM 模型实例
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)  # 初始化图像变换器
        self.reset_image()  # 重置图像状态
        self.default_prompt_mode = default_prompt_mode

    def __call__(self, image: np, original_size: tuple, box: list, points: list, prompt_mode: Optional[str] = None) -> dict:

        # 为了便于维护和使用，在dataloader.py处不传入prompt_mode
        # 使得直接调用上面的default_prompt_mode，可以在训练代码中，直接选择对应的提示模式
        if prompt_mode is None:
            prompt_mode = self.default_prompt_mode
            
        # 处理图像和边界框提示
        image_torch = self.process_image(image, original_size)  # 处理图像

        if prompt_mode == "no":
            inputs = {
                "image": image_torch,  # 添加处理后的图像
                "original_size": original_size,  # 添加原始图像大小
            }
            return inputs

        elif prompt_mode == "box":
            box_torch = self.process_prompt(box, original_size)  # 处理边界框提示
            inputs = {
                "image": image_torch,  # 添加处理后的图像
                "original_size": original_size,  # 添加原始图像大小
                "boxes": box_torch,  # 添加处理后的边界框
            }
            return inputs
        
        elif prompt_mode == "full_box": # 全框提示，可以与无提示进行对比实验
            box_torch = self.process_full_box(original_size)  # 处理边界框提示
            inputs = {
                "image": image_torch,  # 添加处理后的图像
                "original_size": original_size,  # 添加原始图像大小
                "boxes": box_torch,  # 添加处理后的边界框
            }
            return inputs

        else:
            # 转换输入提示
            box_torch = self.process_prompt(box, original_size)  # 处理边界框提示

            # 处理点提示
            point_coords, point_labels = self.process_points(points, original_size)

            # 构建输入字典
            inputs = {
                    "image": image_torch,  # 添加处理后的图像
                    "original_size": original_size,  # 添加原始图像大小
                    "boxes": box_torch,  # 添加处理后的边界框
                    "point_coords": point_coords,  # 添加点坐标 (B, N, 2)
                    "point_labels": point_labels,  # 添加点标签 (B, N)
            }

            return inputs  # 返回输入字典

    def process_image(self, image: np, original_size: tuple) -> torch.tensor:

        nd_image = np.array(image)  # 将图像转换为 NumPy 数组
        input_image = self.transform.apply_image(nd_image)  # 应用图像变换
        input_image_torch = torch.as_tensor(input_image, device=self.device)  # 转换为张量并移动到相应设备
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  # 调整维度并增加批次维度
        return input_image_torch  # 返回处理后的图像张量

    def process_prompt(self, box: list, original_size: tuple) -> torch.tensor:

        box_torch = None  # 初始化边界框张量
        # 设置框提示
        box_np = torch.tensor(box).float().numpy()
        sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, original_size=original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        return box_torch  # 返回处理后的边界框张量

    def process_points(self, points: list, original_size: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理点提示，返回符合SAM要求的point_coords和point_labels。
        
        Args:
            points: 点坐标列表，形状为 (N, 2) 或 (B, N, 2)
            original_size: 原始图像大小 (H, W)
            
        Returns:
            point_coords: 形状为 (B, N, 2) 的点坐标张量
            point_labels: 形状为 (B, N) 的点标签张量，1表示前景点，0表示背景点
        """
        points = torch.tensor(points, dtype=torch.float).to(self.device)
        
        # 确保points是3D张量 (B, N, 2)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (N, 2) -> (1, N, 2)
        
        transform = ResizeLongestSide(self.model.image_encoder.img_size)
        
        # 对每个batch中的点进行坐标变换
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        # apply_coords_torch 期望输入形状为 (N, 2)，需要对每个batch分别处理
        transformed_coords = []
        for b in range(batch_size):
            coords = transform.apply_coords_torch(points[b], original_size)
            transformed_coords.append(coords)
        
        point_coords = torch.stack(transformed_coords, dim=0)  # (B, N, 2)
        
        # 创建点标签，全部设为1（前景点）
        point_labels = torch.ones((batch_size, num_points), dtype=torch.float, device=self.device)  # (B, N)

        return point_coords, point_labels  # 返回处理后的点坐标和标签

    def process_full_box(self, original_size: tuple) -> torch.tensor:

        box_torch = None  # 初始化边界框张量
        H, W = original_size
        box_np = np.array([0, 0, W, H], dtype=np.float32)

        sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, original_size=original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        return box_torch  # 返回处理后的边界框张量

    @property
    def device(self) -> torch.device:
            # 返回模型所在的设备（CPU 或 GPU）
        return self.model.device

    def reset_image(self) -> None:
        """重置当前设置的图像。"""
        self.is_image_set = False  # 标记图像未设置
        self.features = None  # 重置特征
        self.orig_h = None  # 重置原始高度
        self.orig_w = None  # 重置原始宽度
        self.input_h = None  # 重置输入高度
        self.input_w = None  # 重置输入宽度


