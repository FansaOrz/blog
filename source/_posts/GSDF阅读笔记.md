---
layout: post
title: GSDF阅读笔记
date: 2024-12-27 22:54:04
tags: [视觉重建, 机器学习, MLP, 论文解读]
math: true
categories: 视觉重建
excerpt: "GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction"
---
<p align="center">{% asset_img excerpt.png %}</p>

# 相关文档
* 项目链接：https://city-super.github.io/GSDF/
* 代码链接：https://github.com/city-super/GSDF
* 论文链接：https://arxiv.org/pdf/2403.16964
# Introduction
* 作者提出，目前的视觉重建方法在物体的几何精度和画面的渲染精度上有一个“优先级匹配”的问题，目前二者是耦合在一起的，如果我们为这个系统整体添加正则项或者约束，可能某一个方面的质量会提升，而另一个方面则会下降。
* 目前已经有一些方法试图解决这个问题，例如：Neusg和Sugar都是用2维的扁平高斯基元进行表面重建。Sugar使用了强制二元不透明度，Neusg使用联合学习的NeuS模型来正则化属性，但是这些约束都导致了渲染质量的下降。
* 然而，诸如Adaptive Shell（自适应壳[Adaptive shells for efficient neural radiance field rendering]），Binary Occupancy Field（二元占用场[Binary opacity grids: Capturing fine geometric detail for mesh-based view synthesis]）和Scaffold-GS都表明，通过融合几何引导，可以生成具有良好正则化的空间结构，进而显著的提升渲染质量。（TODO：没太看懂这块，先往后看，再回来修改）
* 基于以上见解，作者提出了一个同时优化的双分支（Dual-branches）系统，分别是用于做渲染的3DGS分支和用于做表面重建的SDF分支。主要做了三个改进：
    - 从3DGS分支的光栅化深度来引导SDF分支的光线采样，加强提渲染的效率和避免局部最小值
    - 用SDF来控制3DFS的密度，引导3DGS在表面附近的增长，在其他地方剪枝
    - 对齐两个branch的几何结构（深度和法向）
