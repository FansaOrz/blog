---
layout: post
title: SGS-SLAM阅读笔记（带语义的3DGS SLAM框架）
date: 2025-02-11 16:50:33
tags: [视觉重建, SLAM, 空间感知, NeRF, 3DGS, 机器学习, MLP, 论文解读]
math: true
categories: 视觉重建
excerpt: "SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM"
---
<p align="center">{% asset_img sgsslam.png %}</p>

# 相关文档
- 论文链接：https://arxiv.org/pdf/2402.03246
- 代码链接：https://github.com/ShuhongLL/SGS-SLAM

# Introduction
- 目前基于NeRF的方法实现视觉稠密SLAM的方法有很多了。NeRF-based的SLAM引入MLP作为场景的隐式表示，存在几个问题：
  1. MLP建模时，物体边缘处会过于平滑，使得物体的形状信息丢失，且使得物体在场景中很难被准确的分割出来。
  2. 对于更大的场景，MLP的模型容易出现灾难性的遗忘。合并新场景会导致之前学习的模型精度降低。
  3. NeRF-based的方法计算效率低，因为整个场景由一个或者多个MLP模型表示，因此当更新场景时，需要大量的模型调整。
- 本文是基于3DGS实现的SLAM方法，3DGS渲染速度快，优化时也是直接优化每个高斯核的参数，而不像NeRF一样由像素差异来优化MLP参数。并且这种直接优化的方式，使得可以直接在高斯核中增加新的通道参数用于优化。使用明确的空间和语义信息也有助于相机位姿的跟踪。
- 作者还引入了一个二级调整策略，当识别到物体重新出现在场景中时，关键帧会被选择。

# Method
<p align="center">{% asset_img method.png %}</p>

## Multi-Channel Gaussian Representation