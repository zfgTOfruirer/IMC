# zfg 完成图像预处理---图像旋转---生成图像对---提取图像特征（sift/disk/aliked）---匹配图像特征（lightglue完成sift/disk/alike特征的匹配）---多个模型（sift/disk/aliked）特征和匹配结果的合并---关键点旋转---3D重建（pixsfm/hloc算法）---优化稀疏模型

import sys

sys.path.append('../main_code')

from pathlib import Path
from copy import deepcopy
import logging
import os
import warnings
from typing import Any, Dict
import numpy as np

from hloc import extract_features  # 从 hloc 模块中导入特征提取函数

from pixhloc.pipeline import Pipeline

pixsfm_config = r'..\main_code\pixsfm.yaml'  # pixsfm 的配置文件路径

# zfg 超参数设置
configs = {
    "sift": {  # 配置 SIFT 模型
        "features": {  # 特征提取相关配置
            "model": {"name": "sift"},  # 模型名称为 SIFT
            "options": {
                "first_octave": -1,  # 从第一个八度开始
                "peak_threshold": 0.00667,  # 峰值检测阈值
                "backend": "pycolmap"  # 使用 pycolmap 作为后端
            },
            "output": "feats-sift",  # 输出特征文件名称
            "preprocessing": {
                "grayscale": True,  # 是否转换为灰度图
                "resize_max": 1600  # 最大尺寸调整
            },
        },
        "matches": {  # 特征匹配相关配置
            "output": "matches-sift-lightglue",  # 输出匹配文件名称
            "model": {
                "features": "sift",  # 使用 SIFT 特征
                "name": "lightglue",  # 使用 LightGlue 进行匹配
                "weights": "sift_lightglue",  # LightGlue 的预训练权重
                "filter_threshold": 0.1,  # 过滤阈值
                "width_confidence": -1,  # 宽度置信度（未使用）
                "depth_confidence": -1,  # 深度置信度（未使用）
                "mp": True  # 是否使用多处理
            },
        },
    },

    "disk": {  # 配置 disk 模型
        "features": {  # 特征提取相关配置
            "output": "feats-disk",  # 输出特征文件名称
            "model": {
                "name": "disk",  # 模型名称
                "max_keypoints": 5000,  # 最大关键点数量
            },
            "preprocessing": {
                "grayscale": False,  # 不转换为灰度图
                "resize_max": 1600,  # 最大尺寸调整
            },
        },
        "matches": {  # 特征匹配相关配置
            "output": "matches-disk-lightglue",  # 输出匹配文件名称
            "model": {
                "features": "disk",  # 使用 DISK 特征
                "name": "lightglue",  # 使用 LightGlue 进行匹配
                "weights": "disk_lightglue",  # LightGlue 的预训练权重
                "filter_threshold": 0.1,  # 过滤阈值
                "width_confidence": -1,  # 宽度置信度（未使用）
                "depth_confidence": -1,  # 深度置信度（未使用）
                "mp": True  # 是否使用多处理
            },
        },
    },

    "aliked2k": {  # 配置 ALIKED 模型
        "features": {  # 特征提取相关配置
            "output": "feats-aliked2k",  # 输出特征文件名称
            "model": {
                "name": "aliked",  # 模型名称为 ALIKED
                "model_name": "aliked-n16",  # ALIKED 的具体模型名称
                "max_num_keypoints": 2048,  # 最大关键点数量
                "detection_threshold": 0.0,  # 检测阈值
                "force_num_keypoints": False,  # 是否强制关键点数量
            },
            "preprocessing": {
                "resize_max": 1600,  # 最大尺寸调整
            },
        },
        "matches": {  # 特征匹配相关配置
            "output": "matches-aliked2k-lightglue",  # 输出匹配文件名称
            "model": {
                "features": "aliked",  # 使用 ALIKED 特征
                "name": "lightglue",  # 使用 LightGlue 进行匹配
                "filter_threshold": 0.1,  # 过滤阈值
                "width_confidence": -1,  # 宽度置信度（未使用）
                "depth_confidence": -1,  # 深度置信度（未使用）
                "mp": True  # 是否使用多处理
            },
        },
    },
}


def setup_logger():
    """设置日志记录器的函数"""
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    if not len(logger.handlers):
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False

    # 设置numexpr库的日志级别为ERROR
    numexpr_logger = logging.getLogger("numexpr")
    numexpr_logger.setLevel(logging.ERROR)

    # 设置filelock库的日志级别为ERROR
    filelock_logger = logging.getLogger("filelock")
    filelock_logger.setLevel(logging.ERROR)

    # 设置urllib3库的日志级别为ERROR
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.ERROR)

    # 设置h5py库的日志级别为ERROR
    h5py_logger = logging.getLogger("h5py")
    h5py_logger.setLevel(logging.ERROR)

    # 忽略transformers库中的FutureWarning
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.models.vit.feature_extraction_vit"
    )

    # 忽略torch库中的FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.data.dataloader")


# zfg 从目录中读取数据并存储到字典中
def get_data_from_dict(data_dir: str) -> Dict[str, Any]:
    """
    参数:
        data_dir (str): 包含数据文件的目录路径

    返回:
        Dict[str, Any]: 存储数据的嵌套字典，结构为 {dataset: {scene: [image1, image2, ...]}}
    """
    data_dict = {}  # 初始化一个空字典来存储数据
    # 打开名为 "sample_submission.csv" 的文件，读取内容
    with open(os.path.join(data_dir, "sample_submission.csv"), "r") as f:
        # 遍历文件中的每一行
        for i, l in enumerate(f):
            # 跳过头部（第一行）
            if l and i > 0:
                # 分割每一行的数据，提取 image, dataset, 和 scene 列
                image, dataset, scene, _, _ = l.strip().split(",")
                # 如果字典中还没有这个 dataset 的键，创建一个新的键值对
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                # 如果字典中这个 dataset 下还没有这个 scene 的键，创建一个新的键值对
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                # 将 image 添加到相应的 dataset 和 scene 下的列表中
                data_dict[dataset][scene].append(image)
    return data_dict  # 返回构建的嵌套字典


# zfg 将数组转换为字符串格式，以分号分隔
def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


# zfg 创建提交文件
def create_submission(out_results: Dict[str, Any], data_dict: Dict[str, Any], fname: str):
    """
    参数:
        out_results (Dict[str, Any]): 包含结果数据的字典，结构为 {dataset: {scene: {image: {"R": rotation_matrix, "t": translation_vector}}}}
        data_dict (Dict[str, Any]): 包含输入数据的字典，结构为 {dataset: {scene: [image1, image2, ...]}}
        fname (str): 输出提交文件的文件名

    该函数根据输入数据和结果数据构建提交文件，并写入到指定文件名中。
    """
    n_images_total = 0  # 统计总图像数量
    n_images_written = 0  # 统计写入结果的图像数量
    with open(fname, "w") as f:  # 打开输出文件
        # 写入文件头部
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        # 遍历 data_dict 中的每个数据集
        for dataset in data_dict:
            # 获取该数据集对应的结果
            res = out_results.get(dataset, {})
            # 遍历数据集中每个场景
            for scene in data_dict[dataset]:
                # 获取该场景对应的结果，如果没有则使用默认值
                scene_res = res[scene] if scene in res else {"R": {}, "t": {}}
                # 遍历场景中的每个图像
                for image in data_dict[dataset][scene]:
                    n_images_total += 1  # 总图像数量加一
                    if image in scene_res:
                        # 如果图像有对应的结果
                        R = np.array(scene_res[image]["R"]).reshape(-1)
                        T = np.array(scene_res[image]["t"]).reshape(-1)
                        n_images_written += 1  # 写入结果的图像数量加一
                    else:
                        # 如果图像没有对应的结果，使用默认值
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    # 写入图像的相关信息到文件
                    f.write(f"{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")


# zfg 用于存储路径的类
class DataPaths:
    def __init__(self, data_dir: str, output_dir: str, dataset: str, scene: str):
        """
        参数:
            data_dir (str): 输入数据的根目录
            output_dir (str): 输出数据的根目录
            dataset (str): 数据集的名称
            scene (str): 场景的名称
        """
        # 输入目录，存储测试数据
        self.input_dir = Path(f"{data_dir}/test/{dataset}")
        # 场景目录，存储输出数据
        self.scene_dir = Path(output_dir) / dataset / scene
        # 图像目录，存储场景的图像
        self.image_dir = self.scene_dir / "images"
        self.dataset = dataset  # 数据集名称
        self.scene = scene  # 场景名称

        # SFM（Structure from Motion）目录，存储稀疏重建数据
        self.sfm_dir = self.scene_dir / "sparse"
        # 图像对路径，存储图像对信息
        self.pairs_path = self.scene_dir / "pairs.txt"
        # 特征检索文件路径，存储特征检索结果
        self.features_retrieval = self.scene_dir / "features_retrieval.h5"
        # 特征文件路径，存储图像特征
        self.features_path = self.scene_dir / "features.h5"
        # 匹配文件路径，存储图像匹配结果
        self.matches_path = self.scene_dir / "matches.h5"

        # 旋转匹配相关目录和文件
        self.rotated_image_dir = self.scene_dir / "images_rotated"  # 旋转后的图像目录
        self.rotated_features_path = self.scene_dir / "features_rotated.h5"  # 旋转后的特征文件路径

        # PixSfM相关目录
        self.cache = Path(output_dir) / "cache"  # 缓存目录

        # 创建目录，确保所有需要的目录存在
        self.scene_dir.mkdir(parents=True, exist_ok=True)  # 创建场景目录
        self.image_dir.mkdir(parents=True, exist_ok=True)  # 创建图像目录
        self.sfm_dir.mkdir(parents=True, exist_ok=True)  # 创建SFM目录
        self.rotated_image_dir.mkdir(parents=True, exist_ok=True)  # 创建旋转图像目录
        self.cache.mkdir(parents=True, exist_ok=True)  # 创建缓存目录


# zfg 主函数
def main():
    setup_logger()  # 设置日志记录
    # 设置输出目录路径
    output_dir = Path("/media/laplace/CA4960E6F5A4B1BD/E/2024-06-18 IMC2024 讲解班/temp/output")
    output_dir.mkdir(exist_ok=True, parents=True)  # 如果路径不存在则创建

    # 读取并设置配置
    confs = [configs[c] for c in ['sift', 'aliked2k', 'disk']]  # 读取特定配置
    config = {
        "features": [c["features"] for c in confs],  # 特征配置
        "matches": [c["matches"] for c in confs],  # 匹配配置
        "retrieval": extract_features.confs['dinov2_salad'],  # 检索配置
        "n_retrieval": 32,  # 检索的数量
    }

    # 从目录中获取数据字典
    data_dict = get_data_from_dict(Path('../IMC2024/image-matching-challenge-2024'))
    print(f"data_dict:\n{data_dict}")
    out_results = {}  # 初始化输出结果字典
    for dataset in data_dict:
        if dataset not in out_results:
            out_results[dataset] = {}

        # 设置数据路径
        scene = dataset  # 场景名称
        paths = DataPaths(
            data_dir=Path('../IMC2024/image-matching-challenge-2024'),  # 数据目录
            output_dir=output_dir,  # 输出目录
            dataset=dataset,  # 数据集名称
            scene=scene,  # 场景名称
        )
        img_list = [Path(p).name for p in data_dict[dataset][scene]]  # 获取图像列表
        out_results[dataset][scene] = {}

        if not paths.image_dir.exists():
            # 如果图像目录不存在，跳过
            continue
        # 定义并运行pipeline
        pipeline = Pipeline(config=config, paths=paths, img_list=img_list, pixsfm_config=pixsfm_config)
        pipeline.run()
        sparse_model = pipeline.sparse_model  # 获取稀疏模型

        if sparse_model is None:
            continue

        # 保存结果
        # 遍历稀疏模型中的所有图像项，为每张图像保存其旋转变换矩阵和平移向量
        for _, im in sparse_model.images.items():
            # 构建图像在输出结果中的路径
            img_name = os.path.join('test', dataset, "images", im.name)
            # 为当前图像保存旋转矩阵R和平移向量t
            out_results[dataset][scene][img_name] = {"R": deepcopy(im.rotmat()), "t": deepcopy(np.array(im.tvec))}
        # 删除场景目录以节省空间（注释掉以保留目录）
        # shutil.rmtree(paths.scene_dir)

    # 创建提交文件
    create_submission(out_results, data_dict, "submission.csv")


# 调用主函数
if __name__ == "__main__":
    main()
