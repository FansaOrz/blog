---
title: ROME学习笔记
date: 2024-04-14 22:43:27
tags: [视觉重建, 机器学习, MLP, 计算机视觉, 论文解读]
math: true
categories: 视觉重建
excerpt: "RoMe: Towards Large Scale Road Surface Reconstruction via Mesh Representation"
---

# 相关文档
* 代码链接：https://github.com/DRosemei/RoMe
* 论文链接：https://arxiv.org/pdf/2306.11368

<p align="center">{% asset_img rome.png  %}</p>

# Introduction
* 提出 waypioint 采样方法，每个只渲染局部的区域，然后再合并到一起
* 同时可以估计外参
* 作者提到道路表面的重建可以辅助 BEV 模型的训练和验证，目前主要分为两种方法：传统的方法和基于 NeRF 的方法
  + 传统方法包括 MVS，可以生成稠密点云，但要求有清晰的纹理特征。在处理均匀纹理时可能出现噪点或者不完整的结果（空洞？），并且计算量很大
  + NeRF-based 的方法，使用 MLP 做隐式表达，输入一组带 pose 的图像，可以生成对应的高质量纹理，但需要大量的 GPU 资源，不太好适应大尺度的场景。
* 因此 RoME 的方法是：
  + 提出了一个 2D 的隐式路面表达方法，来实现道路表面的重建
  + 使用 waypoint 采样的方法，来降低内存和时间复杂度
# Approaches

<p align="center">{% asset_img method.png method %}</p>

* 主要分成三个部分，Mesh 初始化、waypoints 采样、和优化

## Mesh Initialization

* 文章使用ORB-SLAM2的pose来初始化mesh。文章使用ORB-SLAM得到相机的每一帧位姿，同时使用Mask2Former来分割语义信息。
* Mesh初始化的方法主要参考：[StreetSurf](https://ventusff.github.io/streetsurf_web/)
  + 水平延伸自车位姿来得到半稠密的点云，然后把些点放到一个MLP网络中，输入xy的坐标，输出z值（用MLP来拟合一个地面表示）。并对xy的值做位置编码，用MLP是为了通过调整PE的评率来控制路面的平滑度
$$
z = MLP(PE(x, y))
$$

## Waypoint Sampling

* 这里使用**最远点采样**的方法对输入的相机pose做了一次采样，每次选取部分采样点后，就只用这几个图片训练当前epoch就行了，而不需要全部图片。
<p align="center">{% asset_img waypoint_sample.png %}</p>

* **个人认为**，这里的采样除了可以加速以外，还有一个原因是因为用的ORB-SLAM的位姿，这个位姿会有误差，如果全部图片都用上的话，模型不太能把位置误差估计出来，导致有重影。而经过位置采样后，每个epoch中，每个范围只由一张image来训练，其他的误差不会引入进来，最后渲染的mesh就会更清晰。



## Optimization

* 优化了两个部分，外参优化和RGB+语义的Mesh优化

### 外参优化
- 旋转矩阵用轴和轴角表示。在优化中同时估计旋转和平移，来实现外参优化。
- 使用Rodrigues公式，将轴角表示的旋转矩阵转换为旋转向量
$$
R = I + \frac{\sin(\alpha)}{\alpha}  \phi^{\hat{}} + \frac{1 - \cos(\alpha)}{\alpha^2}  (\phi^{\hat{}})^2
$$
其中$(*)^{\hat{}}$表示向量转成反对称矩阵

### Mesh优化
- 使用当前mesh来渲染waypoint的图像，然后RGB使用L1 Loss，语义信息使用交叉熵Loss

# Result

<p align="center">{% asset_img result_1.png%}</p>

<p align="center">{% asset_img result_2.png%}</p>

<p align="center">{% asset_img result_3.png%}</p>
