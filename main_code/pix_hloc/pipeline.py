#zfg 算法处理流程Pipeline
import gc
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import h5py
import numpy as np
import pycolmap
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from kornia.enhance import equalize_clahe

from hloc import (
    extract_features,
    localize_sfm,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
    match_features,
)
from hloc.utils.database import COLMAPDatabase
from hloc.utils.io import list_h5_names

from pixhloc.preprocessing import rotate_image_dir
from pixhloc.concatenate import concat_features, concat_matches

class Pipeline:
    def __init__(self, config, paths, img_list, pixsfm_config):
        self.config = config  # 存储配置字典
        self.paths = paths  # 存储路径对象
        self.img_list = img_list  # 存储图像列表
        self.same_shapes = False  # 初始化same_shapes标志为False
        self.sparse_model = None  # 初始化稀疏模型为None
        self.pixsfm_config = pixsfm_config  # 存储PixSfM配置

        # 确保特征和匹配的数量相等，以便进行集成匹配
        assert len(self.config["features"]) == len(self.config["matches"]), "特征和匹配的数量必须相等，以便进行集成匹配。"

        self.rotation_angles = {}  # 初始化旋转角度字典
        self.n_rotated = 0  # 初始化旋转图像计数

    #zfg 数据预处理
    def preprocess(self) -> None:
        """预处理输入目录中的图像并将其保存到输出目录"""
        same_original_shapes = True  # 初始化同形状标志
        prev_shape = None  # 初始化前一个图像的形状

        # 遍历图像文件名列表，进行图像的读取和处理
        for image_fn in tqdm(self.img_list, desc=f"Rescaling {self.paths.input_dir.name}", ncols=80):
            img_path = self.paths.input_dir / "images" / image_fn  # 获取图像路径
            image = cv2.imread(str(img_path))  # 读取图像

            # 检查图像形状是否相同
            if prev_shape is not None:
                same_original_shapes &= prev_shape == image.shape  # 更新同形状标志，只有所有图像形状相同时为True

            prev_shape = image.shape  # 更新前一个图像的形状

            # 如果数据集包含透明对象，进行特殊处理
            if "transp_obj" in self.paths.dataset:
                image = to_tensor(np.ascontiguousarray(image[..., ::-1]))  # 将图像转换为张量并改变颜色通道顺序
                image = equalize_clahe(image, clip_limit=40.0, grid_size=(8, 8))  # 进行直方图均衡化
                image = image.squeeze(0).permute(1, 2, 0).numpy()[..., ::-1] * 255  # 转换回NumPy数组并还原颜色通道顺序

            cv2.imwrite(str(self.paths.scene_dir / "images" / image_fn), image)  # 保存处理后的图像

        self.same_shapes = same_original_shapes  # 更新同形状标志

    #zfg 图像旋转
    def perform_rotation(self) -> None:
        """旋转图像。"""
        self.rotation_angles, self.same_shapes = rotate_image_dir(
            input_dir=self.paths.input_dir,  # 输入图像目录
            output_dir=self.paths.scene_dir,  # 输出图像目录
            image_list=self.img_list,  # 图像文件名列表
            same_original_shapes=self.same_shapes,  # 是否为同形状
        )

    #zfg 得到图像对
    def get_pairs(self) -> None:
        """获取要匹配的图像对。"""
        n_retrieval = self.config["n_retrieval"]  # 获取检索数量

        # 如果图像数量少于检索数量，使用穷尽搜索生成图像对
        if len(self.img_list) < n_retrieval:
            pairs_from_exhaustive.main(output=self.paths.pairs_path, image_list=self.img_list)
            return

        image_dir = self.paths.rotated_image_dir  # 获取旋转后的图像目录

        # 提取图像特征用于检索
        extract_features.main(
            conf=self.config["retrieval"],
            image_dir=image_dir,
            image_list=self.img_list,
            feature_path=self.paths.features_retrieval,
        )

        # 从检索结果中生成图像对
        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=self.config["n_retrieval"],
            output=self.paths.pairs_path,
        )

    #zfg 特征提取
    def extract_features(self) -> None:
        """提取图像特征。"""
        feature_path = self.paths.rotated_features_path  # 获取旋转后的特征路径
        image_dir = self.paths.rotated_image_dir  # 获取旋转后的图像目录

        # 如果存在, 则删除
        if feature_path.exists():
            feature_path.unlink()

        # 遍历配置中的每个特征提取配置
        for config in self.config["features"]:
            ens_feature_path = feature_path.parent / f'{config["output"]}.h5'

            # 如果存在, 则删除
            if ens_feature_path.exists():
                ens_feature_path.unlink()

            extract_features.main(
                conf=config,
                image_dir=image_dir,
                image_list=self.img_list,
                feature_path=ens_feature_path,
            )

    #zfg 特征匹配
    def match_features(self) -> None:
        """匹配图像特征。"""
        feature_path = self.paths.rotated_features_path  # 获取旋转后的特征路径

        # 如果匹配文件存在，删除它
        if self.paths.matches_path.exists():
            self.paths.matches_path.unlink()  # 删除匹配文件

        # 遍历配置中的每个特征匹配配置
        for feat_config, match_config in zip(self.config["features"], self.config["matches"]):
            ens_feature_path = feature_path.parent / f'{feat_config["output"]}.h5'  # 获取特征文件路径
            ens_match_path = self.paths.matches_path.parent / f'{match_config["output"]}.h5'  # 获取匹配文件路径

            if ens_match_path.exists():
                ens_match_path.unlink()  # 如果匹配文件存在，删除它

            # 调用 match_features.main 函数进行特征匹配
            match_features.main(
                conf=match_config,  # 匹配配置
                pairs=self.paths.pairs_path,  # 图像对路径
                features=ens_feature_path,  # 特征文件路径
                matches=ens_match_path,  # 匹配文件路径
            )

    #zfg 特征合并
    def create_ensemble(self) -> None:
        """合并特征和匹配。"""
        feature_path = self.paths.features_path  # 获取特征路径
        feature_path = self.paths.rotated_features_path  # 获取旋转后的特征路径

        # 复制第一个特征和匹配到最终输出
        shutil.copyfile(
            self.paths.features_path.parent / f'{self.config["features"][0]["output"]}.h5',
            feature_path,  # 复制第一个特征文件到旋转后的特征路径
        )
        shutil.copyfile(
            self.paths.matches_path.parent / f'{self.config["matches"][0]["output"]}.h5',
            self.paths.matches_path,  # 复制第一个匹配文件到匹配路径
        )

        # 合并剩余的特征和匹配
        for i in range(1, len(self.config["features"])):
            feat_path = (
                    self.paths.features_path.parent / f'{self.config["features"][i]["output"]}.h5'
            )  # 获取当前特征文件路径
            match_path = (
                    self.paths.matches_path.parent / f'{self.config["matches"][i]["output"]}.h5'
            )  # 获取当前匹配文件路径

            # 合并特征
            concat_features(
                features1=feature_path,  # 第一个特征文件路径
                features2=feat_path,  # 第二个特征文件路径
                out_path=feature_path,  # 输出合并后的特征文件路径
            )

            # 合并匹配
            concat_matches(
                matches1_path=self.paths.matches_path,  # 第一个匹配文件路径
                matches2_path=match_path,  # 第二个匹配文件路径
                ensemble_features_path=feature_path,  # 合并特征文件路径
                out_path=self.paths.matches_path,  # 输出合并后的匹配文件路径
            )

        # 写入图像对文件
        pairs = sorted(list(list_h5_names(self.paths.matches_path)))  # 获取并排序所有图像对
        with open(self.paths.pairs_path, "w") as f:  # 打开图像对文件
            for pair in pairs:  # 遍历所有图像对
                p = pair.split("/")  # 拆分图像对
                f.write(f"{p[0]} {p[1]}\n")  # 写入图像对到文件

    #zfg 关键点处理
    def rotate_keypoints(self) -> None:
        """在旋转匹配后旋转关键点。"""
        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)  # 复制旋转后的特征文件到最终特征文件路径

        # 将旋转的关键点写入
        with h5py.File(str(self.paths.features_path), "r+", libver="latest") as f:  # 以读写模式打开最终特征文件
            for image_fn, angle in self.rotation_angles.items():  # 遍历旋转角度字典中的每个图像和角度
                if angle == 0:  # 如果旋转角度为0，跳过
                    continue

                self.n_rotated += 1  # 增加旋转图像计数

                keypoints = f[image_fn]["keypoints"].__array__()  # 获取当前图像的关键点数组
                y_max, x_max = cv2.imread(str(self.paths.rotated_image_dir / image_fn)).shape[:2]  # 获取旋转后图像的尺寸

                new_keypoints = np.zeros_like(keypoints)  # 初始化新的关键点数组
                if angle == 90:  # 如果旋转角度为90度
                    # 将关键点旋转-90度
                    # ==> (x,y) 变为 (y, x_max - x)
                    new_keypoints[:, 0] = keypoints[:, 1]  # x坐标变为y坐标
                    new_keypoints[:, 1] = x_max - keypoints[:, 0] - 1  # y坐标变为x_max - x坐标 - 1
                elif angle == 180:  # 如果旋转角度为180度
                    # 将关键点旋转180度
                    # ==> (x,y) 变为 (x_max - x, y_max - y)
                    new_keypoints[:, 0] = x_max - keypoints[:, 0] - 1  # x坐标变为x_max - x坐标 - 1
                    new_keypoints[:, 1] = y_max - keypoints[:, 1] - 1  # y坐标变为y_max - y坐标 - 1
                elif angle == 270:  # 如果旋转角度为270度
                    # 将关键点旋转+90度
                    # ==> (x,y) 变为 (y_max - y, x)
                    new_keypoints[:, 0] = y_max - keypoints[:, 1] - 1  # x坐标变为y_max - y坐标 - 1
                    new_keypoints[:, 1] = keypoints[:, 0]  # y坐标变为x坐标
                f[image_fn]["keypoints"][...] = new_keypoints  # 将新的关键点写入文件

    #zfg sfm处理（PixSfm算法 + Hloc算法 两种算法提供不同的重建方式）
    def sfm(self) -> None:
        """运行sfm"""
        import pycolmap  # 导入pycolmap库
        # 初始化种子以确保pycolmap的随机性
        arr = np.ones((10, 2))
        pycolmap.fundamental_matrix_estimation(arr, arr)  # 估计基础矩阵，确保pycolmap的随机性初始化

        # 获取图片路径
        image_dir = self.paths.image_dir

        # 设置相机模式
        camera_mode = pycolmap.CameraMode.AUTO  # 默认相机模式为AUTO
        if self.same_shapes:
            camera_mode = pycolmap.CameraMode.SINGLE  # 如果图像形状相同，则设置相机模式为SINGLE

        # 如果满足以下条件，使用pixsfm进行处理
        self.pixsfm = (
                len(self.img_list) <= 9999  # 图像数量少于等于9999
                and (self.n_rotated == 0)  # 没有旋转图像
                and ("transp" not in self.paths.dataset)  # 数据集中不包含透明对象
        )

        if self.n_rotated != 0:
            logging.info(f"检测到 {self.n_rotated} 张旋转图像, 不使用pixsfm")  # 如果有旋转图像，不使用pixsfm

        gc.collect()  # 进行垃圾回收

        if self.pixsfm:
            from omegaconf import OmegaConf  # 导入OmegaConf库
            from pixsfm.refine_hloc import PixSfM  # 导入PixSfM库
            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)  # 如果缓存目录不存在，创建它

            conf = OmegaConf.load(self.pixsfm_config)  # 加载pixsfm配置
            refiner = PixSfM(conf=conf)  # 创建PixSfM对象
            sparse_model, _ = refiner.run(
                output_dir=Path(self.paths.sfm_dir),
                image_dir=Path(image_dir),
                pairs_path=Path(self.paths.pairs_path),
                features_path=Path(self.paths.features_path),
                matches_path=Path(self.paths.matches_path),
                cache_path=Path(self.paths.cache),
                verbose=False,
                camera_mode=camera_mode,
            )  # 运行PixSfM

            if sparse_model is not None:
                sparse_model.write(Path(self.paths.sfm_dir))  # 如果生成了稀疏模型，将其写入文件

            # 清理缓存
            for file in Path(self.paths.cache).glob("*"):
                file.unlink()  # 删除缓存中的所有文件

            # 子进程将SfM模型写入磁盘 => 在主进程中加载模型
            if self.paths.sfm_dir.exists():
                try:
                    self.sparse_model = pycolmap.Reconstruction(self.paths.sfm_dir)  # 从文件加载重建的模型
                except ValueError:
                    logging.warning(f"无法从 {self.paths.sfm_dir} 重建/读取模型。")
                    self.sparse_model = None  # 如果无法加载模型，设置稀疏模型为None

        else:
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 3  # 设置mapper的最小模型大小为3
            mapper_options.min_num_matches = 10  # 设置mapper的最小匹配数量为10

            self.sparse_model = reconstruction.main(
                sfm_dir=self.paths.sfm_dir,
                image_dir=image_dir,
                image_list=self.img_list,
                pairs=self.paths.pairs_path,
                features=self.paths.features_path,
                matches=self.paths.matches_path,
                camera_mode=camera_mode,
                verbose=False,
                mapper_options=mapper_options,
            )  # 使用HLoc进行SfM重建

        if self.sparse_model is not None:
            self.sparse_model.write(self.paths.sfm_dir)  # 如果生成了稀疏模型，将其写入文件

        gc.collect()  # 进行垃圾回收

    def localize_unregistered(self) -> None:
        """尝试定位未注册的图像"""

        # 如果稀疏模型不存在，输出日志信息并返回
        if self.sparse_model is None:
            logging.info("未重建稀疏模型，跳过定位")
            return

        # 获取已注册图像的名称
        reg_image_names = [im.name for imid, im in self.sparse_model.images.items()]

        # 找出未注册的图像
        missing = list(set(self.img_list) - set(reg_image_names))
        logging.info(f"找到 {len(missing)} 张未注册图像")

        # 如果没有未注册的图像，返回
        if len(missing) == 0:
            return

        # 如果使用PixSfM并且存在refined_keypoints.h5文件，则使用PixSfM的关键点
        if self.pixsfm and (self.paths.sfm_dir / "refined_keypoints.h5").exists():
            features = self.paths.sfm_dir / "refined_keypoints.h5"
            database_path = self.paths.sfm_dir / "hloc" / "database.db"
        else:
            # 否则使用HLoc的关键点
            features = self.paths.features_path
            database_path = self.paths.sfm_dir / "database.db"

        # 连接到COLMAP数据库
        db = COLMAPDatabase.connect(database_path)
        for img_name in missing:
            est_conf = pycolmap.AbsolutePoseEstimationOptions()  # 初始化绝对位姿估计配置
            refine_conf = pycolmap.AbsolutePoseRefinementOptions()  # 初始化绝对位姿精炼配置

            # 从数据库中获取图像ID
            ((image_id,),) = db.execute("SELECT image_id FROM images WHERE name=?", (img_name,))
            # 从数据库中获取相机ID
            ((camera_id,),) = db.execute("SELECT camera_id FROM images WHERE image_id=?", (image_id,))

            if camera_id in self.sparse_model.cameras:
                # 如果相机已经在稀疏模型中，重用相机
                camera = self.sparse_model.cameras[camera_id]
            else:
                # 根据图像元数据推断相机
                camera = pycolmap.infer_camera_from_image(self.paths.image_dir / img_name)
                camera.camera_id = camera_id  # 设置相机ID
                self.sparse_model.add_camera(camera)  # 将相机添加到稀疏模型中

                est_conf.estimate_focal_length = True  # 配置估计焦距
                refine_conf.refine_focal_length = True  # 配置精炼焦距
                refine_conf.refine_extra_params = True  # 配置精炼额外参数

            # 绝对位姿估计和精炼配置
            conf = {
                "estimation": est_conf.todict(),
                "refinement": refine_conf.todict(),
            }

            # 进行定位
            q = [(img_name, camera)]
            logs = localize_sfm.main(
                self.sparse_model,
                q,
                self.paths.pairs_path,
                features,
                self.paths.matches_path,
                self.paths.scene_dir / "loc.txt",
                covisibility_clustering=True,  # 使用可见性聚类
                ransac_thresh=10,  # 设置RANSAC阈值为10
                config=conf,
            )

            # 处理定位结果
            for q, v in logs["loc"].items():
                if v["best_cluster"] is None:
                    # 无法定位, 跳过
                    continue

                v = v["log_clusters"][v["best_cluster"]]  # 选出最佳聚类
                im = pycolmap.Image(
                    q,
                    tvec=v["PnP_ret"]["tvec"],  # 设置平移向量
                    qvec=v["PnP_ret"]["qvec"],  # 设置四元数向量
                    id=image_id,
                    camera_id=camera_id,
                )
                im.registered = True  # 标记图像已注册
                self.sparse_model.add_image(im)  # 将图像添加到稀疏模型中

        # 将稀疏模型写入文件
        self.sparse_model.write(self.paths.sfm_dir)

    def run(self) -> None:
        self.preprocess() # 预处理
        self.perform_rotation() # 旋转图像
        self.get_pairs() # 获取图像对
        self.extract_features() # 提取特征
        self.match_features() # 匹配特征
        self.create_ensemble() # ensemble
        self.rotate_keypoints() # 旋转关键点, 对应旋转图像
        self.sfm() # sfm
        self.localize_unregistered() # 定位未注册的图像