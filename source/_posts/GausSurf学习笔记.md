---
title: GausSurf学习笔记（重建过程中引入MVS的patch match方法，和3dgs互相促进效果）
date: 2024-12-07 23:56:57
tags: [视觉重建, 机器学习, MLP, 论文解读]
math: true
categories: 视觉重建
excerpt: "GausSurf: Geometry-Guided 3D Gaussian Splatting for Surface Reconstruction"
---

<p align="center">{% asset_img gaussurf.png %}</p>

# 相关文档
* 项目链接：https://jiepengwang.github.io/GausSurf/
* 代码链接：https://github.com/jiepengwang/GausSurf
* 论文链接：https://arxiv.org/pdf/2411.19454
# Introduction
* 3DGS主要用于新视角合成，而不是表面重建，因此不能生成高质量的表面重建。SuGaR通过对高斯函数正则化使其平坦；2DGS将场景表示为一组2D高斯盘，并使用表面法线正则化来重建高质量的表面。但以上问题作者认为质量都有限而且优化速度慢。

* 文章提出，自然场景通常由两种类型的区域组成：
    - 纹理丰富的
    - 无纹理的
    - 对于纹理丰富的，使用多视图一致性约束来引导优化过程；对于无纹理的，结合预训练模型中的法线先验来提供补充监督信号。通过有效的整合这些几何先验，文章的方法实现了高质量和快速表面重建。

* 为了提高优化效率和精度，文章通过迭代合并来强制多视图的一致性。在训练GausSurf时，除了利用输入图像的监督损失外，还将立体匹配引入到高斯函数的优化中。具体来说：
    - 使用patch-matching算法来细化深度值和法线图，这样可以匹配多视图图像，以便在优化高斯函数时准确定位表面的位置。
    - 随后，增强的深度和法线将作为几何引导和监督信号，进一步指导高斯函数的优化。

* 传统的MVS里也包含了PatchMatch步骤，但只是在sfm得到的稀疏点云中进行一次。这里的系数点云存在较多的错误点和噪声，因此导致MVS重建质量下降。同时，文章在PatchMatch中还采用了额外的几何验证策略，其中跨多视图的差异超过阈值时，将该点视为错误点而丢弃，这些被丢弃的图像区域意味着他们不包含足够的纹理来进行块匹配，并且无法产生可靠的深度和法线。这些像素归类为无纹理的区域，并在这些区域中使用额外的法线先验用于优化。

* 总结文章的贡献：
    - 整合了传统的MVS中的PatchMatch方法和法线先验信息，增强重建保真度和计算效率；
# Related Work
* 作者提到一篇：[Gsdf: 3dgs meets sdf for improved rendering and reconstruction](https://arxiv.org/pdf/2403.16964)，是把SDF和3DGS结合起来的文章，可以关注一下。
# Method

## 基于PatchMatch的几何引导

* 文章提出了一种reginement and supervision scheme方法，主要是用MVS的几何指导来优化高斯，同时生成更准确的深度和法线图，用于后续MVS优化时的先验。
* 具体来说，
    - 首先根据超参数（训练步数），先预训练一版高斯函数，作为初始化。
    - 然后，为所有的训练图像渲染深度图和法线图，然后把这些渲染后的结果送到PatchMatch算法中，进行细化，得到更精确的深度和法线图。
    - 接下来，这些细化之后的图像再送到高斯模型里，优化指定步数。
    - 迭代的重复这个过程，直到优化收敛。
<p align="center">{% asset_img patchmatch_based.png %}</p>

* **PatchMatch细化** 从高斯函数中提取深度和法线图。首先把图像中每一个像素的深度和法线方向歘博导其相邻的像素上，按照从上到下，从左到右的顺序。然后对于每一个传播后的像素，使用其本身的深度-法线对和传播后的深度法线对，得到其与相邻视图的块相似度（NCC），然后保留具有更高块相似度的深度法线对。传播后的深度和发现方向会用随机噪声来增强。在传播和patchmatch之后，在做额外的几何验证，来检查不同图像的深度和法线图之间的一致性。 **如果不同视图之间的深度或法线差异大于预先制定的阈值，则该深度或法线将视为不可靠，并从此轮的细化结果中删除。** 整个PatchMatch过层根据相邻视图之间的块一致性来细化深度图，从而大大提高了深度质量。

* **深度监督** PatchMatch之后的深度图用来监督高斯函数的优化，使用L1 loss计算：
$$
\mathcal{l}_d = \sum |d_p - d_i|
$$

## 基于法线先验的几何引导

* 作者发现在光滑的表面区域，法线先验可以提供高质量的估计，但是在尖锐特征的地方就会给出太过于平滑的估计，这和PatchMatch正好相反，PatchMatch在边缘可以提供高质量的估计。这里的法线估计方法是用的：[Stablenormal: Reducing diffusion variance for stable and sharp normal](https://stable-x.github.io/StableNormal/)
<p align="center">{% asset_img normal_prior.png %}</p>

## Loss设计和表面提取

* loss共分为三类：颜色loss，深度loss，法线loss，深度和法线一致性loss。
* 表面提取沿用的TSDF
# Experiment

<p align="center">{% asset_img result_1.png %}</p>

<p align="center">{% asset_img result.png %}</p>
