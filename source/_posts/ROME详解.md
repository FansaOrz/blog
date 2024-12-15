---
title: ROME学习笔记
date: 2024-04-14 22:43:27
tags: [视觉重建, 机器学习, MLP, 计算机视觉, 论文解读]
math: true
categories: 视觉重建
excerpt: "RoMe: Towards Large Scale Road Surface Reconstruction via Mesh Representation"
---

# 相关文档

- 代码链接：https://github.com/DRosemei/RoMe
- 论文链接：https://arxiv.org/pdf/2306.11368

<p align="center">{% asset_img rome.png  %}</p>


# Introduction

- 提出 waypioint 采样方法，每个只渲染局部的区域，然后再合并到一起
- 同时可以估计外参
- 作者提到道路表面的重建可以辅助 BEV 模型的训练和验证，目前主要分为两种方法：传统的方法和基于 NeRF 的方法
  - 传统方法包括 MVS，可以生成稠密点云，但要求有清晰的纹理特征。在处理均匀纹理时可能出现噪点或者不完整的结果（空洞？），并且计算量很大
  - NeRF-based 的方法，使用 MLP 做隐式表达，输入一组带 pose 的图像，可以生成对应的高质量纹理，但需要大量的 GPU 资源，不太好适应大尺度的场景。
- 因此 RoME 的方法是：
  - 提出了一个 2D 的隐式路面表达方法，来实现道路表面的重建
  - 使用 waypoint 采样的方法，来降低内存和时间复杂度

# Approaches

<p align="center">{% asset_img method.png method %}</p>

- 主要分成三个部分，waypoints 采样、Mesh 初始化和优化
- 常用的术语和定义为：
  - 3D point cloud：点云，包含了空间中的所有点，每个点都有对应的坐标和颜色信息

##
