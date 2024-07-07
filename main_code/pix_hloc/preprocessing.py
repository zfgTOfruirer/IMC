#zfg 数据处理
import gc
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple
from torchvision import transforms
import re 
import cv2 
from torch import nn
import numpy as np
from tqdm import tqdm

class OrientationPredictor(nn.Module):
    """
    定义一个用于预测图像方向的神经网络模型类
    """

    def __init__(self):
        super().__init__()  # 调用父类的构造函数

        from timm import create_model as timm_create_model  # 导入timm库中的模型创建函数
        from torch.utils import model_zoo  # 导入模型下载模块

        def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
            """
            重命名模型层的辅助函数

            Args:
                state_dict (Dict[str, Any]): 模型状态字典
                rename_in_layers (Dict[str, Any]): 需要重命名的层名称映射

            Returns:
                Dict[str, Any]: 重命名后的状态字典
            """
            result = {}  # 初始化结果字典
            for key, value in state_dict.items():  # 遍历状态字典
                for key_r, value_r in rename_in_layers.items():  # 遍历需要重命名的层名称
                    key = re.sub(key_r, value_r, key)  # 使用正则表达式替换层名称
                result[key] = value  # 将重命名后的层添加到结果字典
            return result  # 返回重命名后的状态字典

        # 创建预训练模型
        self.model = timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4)  # 创建模型
        state_dict = model_zoo.load_url(
            "https://github.com/ternaus/check_orientation/releases/download/v0.0.3/2020-11-16_resnext50_32x4d.zip", 
            progress=True, 
            map_location="cpu"
        )["state_dict"]  # 下载模型权重
        self.model.load_state_dict(rename_layers(state_dict, {"model.": ""}))  # 加载并重命名模型权重

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 确定设备为GPU或CPU

        self.model = torch.nn.Sequential(self.model, torch.nn.Softmax(dim=1))  # 将模型和Softmax层组合成顺序模型
        self.model = self.model.eval().to(device=self.device)  # 设置模型为评估模式并移动到指定设备

        self.transforms = transforms.Compose((
            transforms.ToTensor(),  # 转换为张量
            transforms.Resize(size=(224, 224)),  # 调整大小为224x224
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 标准化
        ))

        self.angles = torch.as_tensor([0, 90, 180, 270], device=self.device)  # 定义角度张量

    @torch.no_grad()
    def forward(self, image):
        """
        前向传播函数，预测图像的旋转角度

        Returns:
            Tensor: 预测的角度
            Tensor: 模型的输出logits
        """
        logits = self.model(image)  # 通过模型获取logits
        result = self.angles[torch.argmax(logits, dim=-1)]  # 根据最大logits索引获取角度
        return result, logits  # 返回预测的角度和logits


def get_rotated_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
    """
    根据给定的角度旋转图像（四舍五入到90度的倍数）

    Args:
        image (np.ndarray): 输入图像
        angle (float): 旋转角度

    Returns:
        Tuple[np.ndarray, float]: 旋转后的图像及实际旋转的角度
    """
    if angle < 0.0:
        angle += 360  # 确保角度为正值
    angle = (round(angle / 90.0) * 90) % 360  # 将角度四舍五入到90度的倍数

    # 根据角度旋转图像
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, int(angle)  # 返回旋转后的图像及实际旋转的角度


def rotate_image_dir(
    input_dir: Path, output_dir: Path, image_list: List[str], same_original_shapes) -> Tuple[Dict[str, Any], bool]:
    """
    预处理输入目录中的图像并将其保存到输出目录

    Args:
        input_dir (Path): 包含图像文件夹的输入目录
        output_dir (Path): 保存预处理图像的输出目录
        image_list (List[str]): 图像文件名列表

    Returns:
        Tuple[Dict[str, Any], bool]: 映射图像文件名到旋转角度的字典及所有图像是否具有相同的形状
    """
    # 旋转图像
    rotation_angles = {}  # 初始化旋转角度字典
    n_rotated = 0  # 记录旋转图像数量

    same_rotated_shapes = True  # 假设所有旋转后的图像形状相同
    prev_shape = None  # 记录前一个图像的形状

    deep_orientation = OrientationPredictor()  # 创建图像方向预测模型实例

    for image_fn in tqdm(image_list, desc=f"Rotating {input_dir.name}", ncols=80):
        img_path = output_dir / "images" / image_fn  # 获取图像路径

        image = cv2.imread(str(img_path))  # 读取图像
        image = deep_orientation.transforms(image)  # 应用变换
        image = image.to(device=deep_orientation.device).unsqueeze(0)  # 移动到设备并添加批次维度
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # 使用混合精度加速
            angle, _ = deep_orientation(image)  # 预测角度
            angle = angle.squeeze(0).cpu().numpy()  # 获取角度值

        image = cv2.imread(str(img_path))  # 再次读取原始图像
        image, angle = get_rotated_image(image, int(angle))  # 旋转图像

        if prev_shape is not None:
            same_rotated_shapes &= prev_shape == image.shape  # 检查旋转后图像形状是否相同

        prev_shape = image.shape  # 更新前一个图像形状

        if angle != 0:
            n_rotated += 1  # 记录非0旋转角度的图像数量

        cv2.imwrite(str(output_dir / "images_rotated" / image_fn), image)  # 保存旋转后的图像

        rotation_angles[image_fn] = angle  # 记录图像文件名和旋转角度

    # 释放CUDA内存
    del deep_orientation
    gc.collect()

    same_shape = same_original_shapes  # 返回原始形状是否相同
    return rotation_angles, same_shape  # 返回旋转角度字典和形状一致性标志
