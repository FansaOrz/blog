---
layout: post
title: LetsGo阅读笔记（有LiDAR辅助的适用于大范围停车场的重建框架）
date: 2025-02-05 15:56:15
tags: [视觉重建, 机器学习, MLP, 深度学习, 论文解读]
math: true
categories: 视觉重建
excerpt: "LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives"
---

<p align="center">{% asset_img cover.png %}</p>

<mark>**总结**：Lidar-IMU-Camera构建点云地图，转成mesh，得到RGB深度，渲染3DGS</mark>

# 相关文档
* 项目链接：https://zhaofuq.github.io/LetsGo/
* 代码链接：https://github.com/zhaofuq/LOD-3DGS
* 论文链接：https://dl.acm.org/doi/pdf/10.1145/3687762
# Introduction
* 本文的主要目标场景是停车场，停车场有光照条件差、表面无纹理和反射表面的问题，传统的SFM和MVS方法在这种场景中没法提取足够的特征，导致重建效果不佳。基于LiDAR的方法当出现反光表面和透明玻璃时，几何估计也会有比较大的问题，并且激光会更稀疏，会丢失高频细节。
* 本文的主要特点是把LiDAR的点云集成到3DGS的框架中，并利用LiDAR的深度先验计算depth正则项，减少重建后的“浮点”。
* 由于场景过大，有可能导致GPU OOM，因此作者还提出了一个多分辨率的3DGS表示。低分辨率3DGS捕捉场景的粗略结构；高分辨率3DGS捕捉细节。同时在渲染阶段，作者引入了一种**级别选择策略**，通过考虑3DGS和渲染视点之间的距离来优化视觉质量和设备性能之间的平衡。相对于原始的3DGS，该方法渲染速度快了4倍。
# Method

<p align="center">{% asset_img method.png %}</p>

## Initial Mesh Reconsruction（mesh初始化）

* 作者使用LVI-SLAM得到全局点云地图+RGB颜色，然后使用泊松重建得到初始的mesh。**同时把mesh和原始的点云对比较，删掉不正确的mesh面。**
* **个人认为，这里是不是用NVIDIA的NKSR来生成mesh效果会更好？**

## Gaussian Splatting with LiDAR Inputs（有LiDAR辅助的Gaussian Splatting）

* 考虑到原始的点云地图会有一些噪点，作者这里在泊松重建后的mesh上重新采样一组点云作为深度信息。并且利用采样后的深度信息与GS的depth信息额外计算了一个深度loss，和RGBloss一起训练。
  $$
  \mathcal{L}_{total} = \mathcal{L}_{rgb} = \lambda_{depth} \mathcal{L}_{depth}
  $$

## Multi-Resolution Represtration（多分辨率表示）

* 对点云地图做多分辨率的降采样，最精细的分辨率是4cm，然后8cm, 16cm, 32cm... 直到点云个数少于1万个。
* 每个分辨率用一个独立的GS模型表示，在渲染时根据视点距离选择合适的分辨率。
<p align="center">{% asset_img render.png %}</p>

# Expeirment

<p align="center">{% asset_img result.png %}</p>
