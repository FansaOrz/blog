---
layout: post
title: SuGaR阅读笔记
date: 2024-12-27 23:40:56
tags: [视觉重建, 机器学习, MLP, 论文解读]
math: true
hide: true
categories: 视觉重建
excerpt: "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering 一种准确快速的从3DGS中提取mesh的方法"
---

<p align="center">{% asset_img sugar.png %}</p>

# 相关文档
* 项目链接：https://anttwo.github.io/sugar/
* 代码链接：https://github.com/Anttwo/SuGaR
* 论文链接：https://arxiv.org/pdf/2311.12775
# Introduction
- 作者首先表明，3DGS优化后的结果，高斯函数没有一个有序的结构，并且和场景的实际表面不能很好的对齐。当然除了表面之外，整个场景也需要Mesh的表示形式，这样可以用一些其他的3D建模软件来编辑。
- 针对这个问题，作者首先提出正则化项来鼓励高斯在场景的表面上均匀分布，以此来捕捉到更好的表面结构。首先做一个假设，高斯函数函数是平坦的，并且在场景表面上均匀分布的。然后从高斯函数中提取体密度，并优化它和实际计算出来的高斯密度的差，最终让高斯函数在表面上均匀分布。（具体做法需要往后再看一下）。
- 