# zfg 特征提取+特征匹配
import logging
from pathlib import Path
from typing import Tuple

import h5py as h5
import numpy as np
from hloc.utils.io import find_pair, list_h5_names
from tqdm import tqdm


# zfg 特征合并
def concat_features(features1: Path, features2: Path, out_path: Path) -> None:
    """将两个HDF5文件中的特征合并到一个HDF5文件中。

    参数:
        features1 (Path): 第一个HDF5文件的路径。
        features2 (Path): 第二个HDF5文件的路径。
        out_path (Path): 输出HDF5文件的路径。
    """
    # 读取特征
    img_list = list_h5_names(features1) + list_h5_names(features2)  # 获取两个HDF5文件中的所有数据集名称
    img_list = list(set(img_list))  # 去重
    ensemble_features = {}  # 用于存储合并后的特征

    with h5.File(features1, "r") as f1:  # 以只读模式打开第一个HDF5文件
        with h5.File(features2, "r") as f2:  # 以只读模式打开第二个HDF5文件
            for img in tqdm(img_list, desc="concatenating features", ncols=80):  # 遍历所有图像名称，并显示进度条
                # 如果图片在第一个文件中，获取其关键点和分数，否则为空数组
                kpts1 = f1[img]["keypoints"][:] if img in f1.keys() else np.array([])  # 从第一个文件中读取关键点
                kpts2 = f2[img]["keypoints"][:] if img in f2.keys() else np.array([])  # 从第二个文件中读取关键点

                scores1 = f1[img]["scores"][:] if img in f1.keys() else np.array([])  # 从第一个文件中读取分数
                scores2 = f2[img]["scores"][:] if img in f2.keys() else np.array([])  # 从第二个文件中读取分数

                n_feats1 = len(kpts1) if img in f1.keys() else 0  # 计算第一个文件中的特征数量
                n_feats2 = len(kpts2) if img in f2.keys() else 0  # 计算第二个文件中的特征数量

                # 合并关键点和分数
                keypoints = np.concatenate([kpts1, kpts2], axis=0)  # 合并两个文件中的关键点
                scores = np.concatenate([scores1, scores2], axis=0)  # 合并两个文件中的分数

                ensemble_features[img] = {
                    "keypoints": keypoints,  # 存储合并后的关键点
                    "scores": scores,  # 存储合并后的分数
                    "counts": [n_feats1, n_feats2],  # 记录每个文件中各自的特征数量
                }

    # 写入特征到新的HDF5文件
    ens_kp_ds = h5.File(out_path, "w")  # 以写模式打开输出HDF5文件
    for img in ensemble_features:  # 遍历所有图像名称
        ens_kp_ds.create_group(img)  # 创建图像组
        for k in ensemble_features[img].keys():  # 遍历每个图像的特征键
            ens_kp_ds[img].create_dataset(k, data=ensemble_features[img][k])  # 创建数据集并写入数据

    ens_kp_ds.close()  # 关闭输出文件


# zfg 匹配处理
def reverse_matches(
        matches: np.ndarray, scores: np.ndarray, num_kpts1: int, num_kpts2: int
) -> Tuple[np.ndarray, np.ndarray]:
    """针对于match2匹配match1,需要做一次反转

    参数:
        matches (np.ndarray): 匹配索引数组，长度为图像1中的关键点数量，每个值要么是-1要么是图像2中的匹配索引
        scores (np.ndarray): 匹配得分数组，长度为图像1中的关键点数量
        num_kpts1 (int): 图像1中的关键点数量
        num_kpts2 (int): 图像2中的关键点数量

    返回:
        Tuple[np.ndarray, np.ndarray]: 反转后的匹配索引数组和匹配得分数组
    """
    rev_matches = np.ones(num_kpts2) * -1  # 初始化反转后的匹配索引数组，默认值为-1
    rev_scores = np.zeros(num_kpts2)  # 初始化反转后的匹配得分数组，默认值为0

    assert len(matches) == num_kpts1, "匹配的数量必须等于图像1中的关键点数量"  # 确认matches数组长度等于图像1中的关键点数量
    assert np.max(matches) < num_kpts2, "匹配必须是图像2中关键点的索引"  # 确认matches数组中的最大值小于图像2中的关键点数量

    # matches是一个长度为num_kpts1的列表，每个值要么是-1要么是图像2中的匹配索引
    for i, m in enumerate(matches):  # 遍历matches数组
        if m != -1:  # 如果匹配索引不等于-1
            rev_matches[m] = i  # 在反转后的匹配索引数组中，将对应位置的值设为当前索引
            rev_scores[m] = scores[i]  # 在反转后的匹配得分数组中，将对应位置的值设为当前得分

    return rev_matches.astype(int), rev_scores  # 返回反转后的匹配索引数组和匹配得分数组


# zfg 特征提取
def extract_matches(
        matches: np.ndarray, features: np.ndarray, name0: str, name1: str, idx=0
) -> Tuple[np.ndarray, np.ndarray]:
    """从一对图像中提取匹配。

    参数:
        matches (np.ndarray): 图像之间的匹配。
        features (np.ndarray): 图像的合并特征。
        name0 (str): 图像0的名称。
        name1 (str): 图像1的名称。
        idx (int, optional): 在合并特征中的图像索引。默认为0。

    返回:
        Tuple[np.ndarray, np.ndarray]: 匹配和分数。
    """
    nkpts0 = features[name0]["counts"][idx]  # 获取图像0的关键点数量
    nkpts1 = features[name1]["counts"][idx]  # 获取图像1的关键点数量

    try:
        p, rev = find_pair(matches, name0, name1)  # 尝试找到图像对的匹配索引和反转标志
    except ValueError:  # 如果找不到匹配对，则捕获ValueError异常
        m = np.ones(nkpts0) * -1  # 初始化匹配数组，默认值为-1
        sc = np.zeros(nkpts0)  # 初始化匹配得分数组，默认值为0
        return m, sc  # 返回初始化的匹配数组和得分数组

    m = matches[p]["matches0"].__array__()  # 获取匹配数组
    sc = matches[p]["matching_scores0"].__array__()  # 获取匹配得分数组

    return reverse_matches(m, sc, nkpts1, nkpts0) if rev else (m, sc)  # 如果需要反转，则调用reverse_matches，否则直接返回匹配和得分数组


# zfg 特征匹配合并
def concat_matches(
        matches1_path: Path, matches2_path: Path, ensemble_features_path: Path, out_path: Path
):
    """合并两个h5文件中的匹配并写入新的h5文件。

    参数:
        matches1_path (Path): 第一个h5文件的路径。
        matches2_path (Path): 第二个h5文件的路径。
        ensemble_features_path (Path): 合并特征的h5文件路径。
        out_path (Path): 输出h5文件的路径。
    """
    ensemble_matches = {}  # 用于存储合并后的匹配

    with h5.File(matches1_path, "r") as matches1:  # 以只读模式打开第一个HDF5文件
        with h5.File(matches2_path, "r") as matches2:  # 以只读模式打开第二个HDF5文件
            with h5.File(ensemble_features_path, "r") as ensemble_features:  # 以只读模式打开合并特征的HDF5文件
                # 获取所有唯一的图像对
                pairs = list_h5_names(matches1_path) + list_h5_names(matches2_path)  # 获取两个HDF5文件中的所有图像对
                pairs = [sorted(p.split("/"))[0] + "/" + sorted(p.split("/"))[1] for p in pairs]  # 对图像对进行排序并格式化
                pairs = sorted(list(set(pairs)))  # 去重并排序

                logging.info(f"找到 {len(pairs)} 个唯一对")  # 记录日志：找到的唯一图像对数量
                logging.info(f"matches1 中的对数: {len(list_h5_names(matches1_path))}")  # 记录日志：第一个文件中的图像对数量
                logging.info(f"matches2 中的对数: {len(list_h5_names(matches2_path))}")  # 记录日志：第二个文件中的图像对数量

                for pair in tqdm(pairs, desc="concatenating matches", ncols=80):  # 遍历所有图像对，并显示进度条
                    name0, name1 = pair.split("/")  # 获取图像对中的两个图像名称

                    # 准备字典
                    if name0 not in ensemble_matches:  # 如果字典中没有name0，添加name0
                        ensemble_matches[name0] = {}
                    if name1 not in ensemble_matches[name0]:  # 如果字典中name0下没有name1，添加name1
                        ensemble_matches[name0][name1] = {}

                    # 获取matches1中的匹配
                    m1, sc1 = extract_matches(matches1, ensemble_features, name0, name1, idx=0)  # 提取第一个文件中的匹配和得分

                    # 获取matches2中的匹配
                    m2, sc2 = extract_matches(matches2, ensemble_features, name0, name1, idx=1)  # 提取第二个文件中的匹配和得分

                    # 合并匹配
                    offset = ensemble_features[name1]["counts"][0]  # 获取图像1在合并特征中的偏移量
                    m2 += offset * np.where(m2 != -1, 1, 0)  # 如果匹配索引不等于-1，则添加偏移量

                    ensemble_matches[name0][name1]["matches0"] = np.concatenate([m1, m2], axis=0)  # 合并匹配数组

                    ensemble_matches[name0][name1]["matching_scores0"] = np.concatenate([sc1, sc2], axis=0)  # 合并匹配得分数组

    # 写入合并的匹配到新的h5文件
    ens_matches_ds = h5.File(out_path, "w")  # 以写模式打开输出HDF5文件
    for img1 in ensemble_matches:  # 遍历合并后的匹配字典
        ens_matches_ds.create_group(img1)  # 创建图像1的组
        for img2 in ensemble_matches[img1].keys():  # 遍历图像1下的所有图像2
            ens_matches_ds[img1].create_group(img2)  # 创建图像2的组
            for k in ensemble_matches[img1][img2].keys():  # 遍历图像对中的所有键
                ens_matches_ds[img1][img2].create_dataset(k, data=ensemble_matches[img1][img2][k])  # 创建数据集并写入数据

    ens_matches_ds.close()  # 关闭输出文件
