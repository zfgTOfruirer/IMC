﻿# **Kaggle:** **Image Matching Challenge 2024 - Hexathlon 解决方案**



## 比赛结果：排名top 2.5%，获得银牌。



<div align="center">
  <img src="images/fig-1.png" alt="IMC银牌"   width="65%"/>
</div>




成员laplace\_zfg，noshakeplz，Emily，Yinghao，Wang。

作者（laplace\_zfg）主要负责其中，图像特征匹配，以及3D重建。具体由SuperGlue 模型完成特征匹配，PixSfM算法对完成特征匹配得到得关键点进行3D重建生成稀疏模型。实际实验中发现SuperGlue的加入使整个推理运行时间太长，超过了kaggle限定的推理时间，因此后续采用LightGlue完成模型的特征匹配。并且为提高鲁棒性，应对赛题对于不同场景3D重建的需求，加入Hloc算法，确保在不同条件下均能得到高质量的重建结果



## **项目背景：**

本次竞赛的目标是通过使用多场景图像集构建精确的3D场景。参与者需要开发模型，生成精确的空间表示，不论图像来源如何——无论是无人机拍摄的图像，还是在密林中、夜晚或其他六种问题类别中的场景。



## 项目方案：



### **数据预处理：**



#### **1. 预处理图像**

遍历图像文件名列表，读取并处理图像，对图像进行直方图均衡化（如果数据集包含透明对象），并将处理后的图像保存到指定目录。预处理步骤确保所有输入图像具有一致的质量和格式，有助于后续步骤中特征提取和匹配的准确性。



#### **2. 图像旋转**

使用了一个预训练的图像方向预测模型（模型输出4个类别，分别为0度、90度、180度和270度）。对输入图像进行旋转处理，并记录旋转角度。旋转图像是为了确保所有图像在相同的方向上进行处理，避免因图像方向不同而导致特征匹配不准确的问题。



#### **3. 生成图像对**

如果图像数量少于指定阈值，使用穷尽搜索生成图像对；否则，提取图像特征用于检索，并从检索结果中生成图像对。生成图像对是为了后续特征匹配做准备，通过检索生成的图像对可以有效减少计算量并提高匹配精度。



### **特征工程：**



#### **1. 提取图像特征**

调用特征提取函数，提取每张图像的特征并保存。特征提取是后续特征匹配和3D重建的基础，高质量的特征可以显著提高匹配的准确性和重建的效果。



**（1）SIFT 特征提取：**

使用 SIFT 模型提取特征，配置参数包括初始八度、峰值检测阈值等。将图像转换为灰度图，并调整大小。



**（2）DISK 特征提取：**

使用 DISK 模型提取特征，最大关键点数量为 5000。不将图像转换为灰度图，只调整大小。



**（3）ALIKED 特征提取：**

使用 ALIKED 模型提取特征，最大关键点数量为 2048。不将图像转换为灰度图，只调整大小。这些提取的特征文件分别存储在不同的 HDF5 文件中，供后续的特征匹配步骤使用。



#### **2. 匹配图像特征**

调用特征匹配函数，匹配图像对中的特征并保存匹配结果。特征匹配是为了找到图像间的对应点，这些对应点是进行3D重建的关键。



**（1）SIFT 特征匹配：**

使用LightGlue 模型匹配 SIFT 特征，预训练权重为 `sift\_lightglue`。输出匹配文件名为 `matches-sift-lightglue.h5`。

​	

**（2）DISK 特征匹配：**

使用 LightGlue 模型匹配 DISK 特征，预训练权重为 `disk\_lightglue`。输出匹配文件名为 `matches-disk-lightglue.h5`。



**（3）ALIKED 特征匹配：**

使用 LightGlue 模型匹配 ALIKED 特征，预训练权重为 `aliked\_lightglue`。输出匹配文件名为 `matches-aliked2k-lightglue.h5`。

上述步骤为每个特征提取模型生成了对应的特征匹配文件。



#### **3. 合并多个模型的特征和匹配**

将多个特征和匹配文件合并成一个，生成最终的特征和匹配结果。合并特征和匹配结果是为了集成不同特征提取算法的优点，提高整体匹配的准确性。

（1）首先复制第一个特征和匹配文件到最终输出位置（如 SIFT 模型的特征和匹配文件）。

（2）对于剩余的模型（DISK 和 ALIKED），依次进行特征和匹配的合并：

调用 concat\_features 函数，将新模型的特征与已有特征合并。

调用 concat\_matches 函数，将新模型的匹配结果与已有匹配合并。

通过上述步骤，我们将所有模型的特征和匹配结果整合在一起，生成了一个集成的特征和匹配文件。



#### **4. 旋转关键点**

根据记录的旋转角度旋转关键点，确保关键点的坐标与旋转后的图像一致。旋转关键点是为了保证在旋转后的图像中，关键点的位置仍然准确，从而保证后续3D重建的正确性。



### **3D重建：**

#### **1. 结构化运动（SfM）**

使用PixSfM进行稀疏3D重建，或者在不满足PixSfM条件时使用HLoc完成3D重建，并将保存生成的稀疏模型。使用PixSfM和HLoc可以提供不同的重建方式，确保在不同条件下均能得到高质量的重建结果。



#### **2. 定位未注册的图像**

利用已经注册的图像生成的相机姿态，使用绝对姿态估计，对在稀疏模型中未注册的图像尝试重新定位。如果成功并将它们添加到模型中。定位未注册的图像是为了尽可能将所有输入图像都包含在3D模型中，提高重建的完整性。



#### **3. 最终3D重建效果**

**多个角度拍摄的原图（提交的测试集输入）**

<div align="center">
  <img src="images/3D_original.png" alt="IMC银牌"   width="50%"/>
</div>

**3D重建稀疏模型**

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="images/3D_0.png" alt="效果0" width="32%" />
  <img src="images/3D_1.png" alt="效果1" width="32%" />
  <img src="images/3D_2.png" alt="效果2" width="32%" />
</div>



**完成3D稀疏模型的重建后，最终提交的文件应包含每个测试图像的姿态信息（即相机在拍摄该图像时的外部参数，包括旋转矩阵和平移向量）。核心评估指标是相机中心位置的平均准确性 (mean Average Accuracy, mAA)。**



## **项目目录结构：**



**lib:相关依赖包**

**main\_code:模型代码**

**model:预训练模型**

**temp:代码运行相关结果**

**inference.py：模型推理部分**

**README:项目说明**



## **推理环境：** 

搭载12th Gen Intel(R) Core(TM) i5-12500H处理器和16 GB RAM内存的Win11操作系统。RXT 3060 显卡。



## **总结：**

在IMC2024竞赛中，我们通过多种图像特征提取模型（包括SIFT、DISK和ALIKED）提取特征，并使用LightGlue进行特征匹配，将匹配结果合并，旋转关键点并进行结构化运动（SfM）重建3D模型。利用PixSfM和HLoc进行稀疏重建，并通过绝对姿态估计重新定位未注册图像，最终生成完整的3D场景。这一流程有效处理了多种复杂场景，包括历史建筑、自然环境和透明反射物体，显著提高了图像匹配和3D重建的精度和完整性。











