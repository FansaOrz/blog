---
layout: post
title: PGSR阅读笔记（深度无偏估计+多视角一致性）
date: 2025-02-02 20:14:35
tags: [视觉重建, 机器学习, MLP, 深度学习, 论文解读]
math: true
categories: 视觉重建
excerpt: "PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction"
---
<p align="center">{% asset_img cover.png %}</p>

# 相关文档
* 项目链接：https://zju3dv.github.io/pgsr/
* 论文链接：https://arxiv.org/abs/2406.06521
* 代码链接：https://github.com/zju3dv/PGSR
* 视频介绍：
  + [B站：【学习记录】3dgs几何精度六月进展](https://www.bilibili.com/video/BV1hm421V7w1/?vd_source=9629687338410a5ccaa5e1a595d0f17d)
  + [B站：AnySyn3D-Webinar007-PGSR](https://www.bilibili.com/video/BV1DBsTe4Eb2/?spm_id_from=333.337.search-card.all.click&vd_source=9629687338410a5ccaa5e1a595d0f17d)
# Introduction
* 3DGS由于无序性和不规则性，使其无法准确的获得场景的几何表面信息。并且3DGS只关注图像重建的loss，而没有考虑到几何重建的loss。
* 本文的作者提出了一种**深度的无偏估计**方法以获取更准确的深度值。之前的方法是在相机坐标系的z方向上，对每个高斯分布做累加得到深度值，这会导致深度对应曲面的问题，如图2所示。3DGS是一个椭球，深度做累积的话，得到的深度是一个曲面，而不是一个平面。

<p align="center">{% asset_img unbias_depth.png %}</p>

* 为解决这个问题，作者将3D高斯压缩成平面，并得到**高斯平面的法向量**和**相机到平面的距离**。通过将二者相除，可以得到相机光线和高斯平面的交点，改点到相机的距离为真正的深度值。
* 除此之外，还引入了单视角和多视角的正则项来优化每个像素的平面参数。
  + 单视角正则项是假设相邻像素通常属于同一个平面。所以可以从相邻像素的深度中估计法向量并与当前像素的法向量保持一致。（在图像边缘处这个假设通常不成立，因此需要做边缘检测并把这部分区域的权重降低）
  + 多视角正则项是确保在不同视角下的几何一致性。具体实现后文会提到。
* 最后，作者还引入两个光度系数来补偿图像中的光照变化，进一步提高重建质量。
# Method

<p align="center">{% asset_img method.png %}</p>

## Planar-based Gaussian Splatting Representation

* **Flattening 3D Gaussian**
  + 原始的3D高斯的协方差矩阵为 $\mathbf{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T$，其中 $\mathbf{R}_i$ 表示椭球三个方向的正交基，$\mathbf{S}_i$ 为沿着三个方向的尺度。这里把椭球压缩成平面，就是把椭球沿着最小尺度的方向压缩，也就是把这个最小的scale作为了loss中的一项：
  $$
  L_s = || \min(s_1, s_2, s_3) ||_1
  $$
* **Unbiased Depth Rendering**

<p align="center">{% asset_img unbiased.png %}</p>

  + 取scale最小的方向 $\mathbf{n}_i$ 作为平面法向量，利用相机到global坐标系的旋转变换 $\mathbf{R}_c$ 变换到相机坐标系下，并利用 $\alpha$-blending的方法，得到当前像素所有高斯的整体法向量：
  $$
  \mathbf{N} = \sum_{i \in N} \mathbf{R}_c^T\mathbf{n}_i \alpha_i \prod_{j = 1}^{i-1} (1 - \alpha_j)
  $$
  + 每个高斯核到相机的距离为 $d_i = (\mathbf{R}_c^T(\mathbf{\mu}_i - \mathbf{T}_c))^T(\mathbf{R_c^T\mathbf{n_i}})$，其中 $\mathbf{T}_c$为相机中心在世界坐标系下的位置，$\mu_i$ 为高斯核的世界坐标系下的位置。再次利用 $\alpha$-blending的方法，得到当前像素所有高斯的整体深度：
  $$
  \mathcal{D} = \sum_{i \in N} d_i \alpha_i \prod_{j = 1}^{i-1} (1 - \alpha_j)
  $$
  + 得到法向量$\mathbf{N}$和$\mathcal{D}$后，利用下式得到**交点到相机的距离，即为真正的深度值**：
  $$
  \mathbf{D}(\mathbf{p}) = \frac{\mathcal{D}}{\mathbf{N}(\mathbf{p})\mathbf{K}^{-1}\tilde{\mathbf{p}}}
  $$
  + 其中 $\mathbf{K}$ 为相机内参，$\tilde{\mathbf{p}}$为归一化后的像素坐标。
* 这样计算深度的方法有两个好处：
  + 这种深度和高斯平面一致，以前的方法是基于高斯的Z做$\alpha$=blending，因此深度是曲面。
  + 由于每条射线的累积权重可能小于1，以前的渲染方法受到权重累积的影响，可能导致深度更靠近相机侧。相比之下，本文的深度是通过将渲染原点到平面的距离除以法线得到的，有效地消除了权重累积系数的影响。

## Geometric Regularization

* **Single-view Regularization**
  + **局部平面性假设**：一个像素及其相邻像素可以被视为一个近似平面。渲染深度图后，四个相邻点进行采样。利用它们的深度计算平面的法线。然后，最小化此法线图与渲染的法线图之间的差异，确保局部深度和法线之间的几何一致性。
  + 对于每个像素，取其四个相邻像素，并拿到相机坐标系下其对应的3D点 $\left\{\mathbf{P}_j | j = 1, ..., 4 \right\}$，并计算它们对应平面的法向量：
  $$
  \mathbf{N}_d(\mathbf{p}) = \frac{(\mathbf{P}_1 - \mathbf{P}_0) \times (\mathbf{P}_3 - \mathbf{P}_2)}{|(\mathbf{P}_1 - \mathbf{P}_0) \times (\mathbf{P}_3 - \mathbf{P}_2)|}
  $$
  + 最终得到单视角一致性的loss为：
  $$
  \boldsymbol{L}_{\text {svgeom }}=\frac{1}{W} \sum_{\boldsymbol{p} \in W}(1-\overline{\nabla \boldsymbol{I}})^2\left\|\boldsymbol{N}_d(\boldsymbol{p})-\boldsymbol{N}(\boldsymbol{p})\right\|_1
  $$
  其中 $\overline{\nabla \boldsymbol{I}}$ 为归一化之后的图像梯度，范围在0到1之间，用以控制边缘区域的权重。
* **Multi-View Geometric Consistency**
  + 如下图所示，对于关键帧和相邻帧，分别渲染法向量和相机到平面的距离。对于关键帧中的特定像素 $\mathbf{p}_r$，其对应法向量为$\mathbf{n}_r$，距离为$d_r$。关键帧中的像素 $\mathbf{p}_r$ 可以通过单应矩阵 $\mathbf{H}_{rn}$ 投影到相邻帧中的像素 $\mathbf{p}_n$。
  $$
  \mathbf{H}_{rn} = \mathbf{K}_n(\mathbf{R}_{rn} - \frac{\mathbf{T}_{rn}\mathbf{n}_t^T}{d_r})\mathbf{K}_r^{-1}
  $$
  + 此时计算出重投影误差，即可作为多视角几何一致性loss：
  $$
  \mathbf{L}_{mvgeom} = \frac{1}{V}\sum_{\mathbf{p}_r \in W} \omega(\mathbf{p}_r) \phi(\mathbf{p}_r) \\

  \phi(\mathbf{p}_r) = || \mathbf{p}_r - \mathbf{H}_{nr}\mathbf{H}_{rn}\mathbf{p}_r || \\

  w\left(\boldsymbol{p}_r\right)= \begin{cases}1 / \exp \left(\phi\left(\boldsymbol{p}_r\right)\right), & \text { if } \phi\left(\boldsymbol{p}_r\right)<1 \\ 0, & \text { if } \phi\left(\boldsymbol{p}_r\right)>=1\end{cases}
  $$
  其中$\omega(\boldsymbol{p}_r)$表示权重，如果两次重投影误差太大，则可能是有遮挡，此时就降低该权重。
<p align="center">{% asset_img multi-view.png %}</p>

* **Multi-View Photometric Consistency**
  + 从MVS中获得灵感，作者取像素周围7x7的像素块，将其映射到相邻帧中，并将其转换成灰度图，使用NCC（normalized cross correlation）计算两个patch之间的光度误差作为一个loss项：
  $$
  \boldsymbol{L}_{m v r g b}=\frac{1}{V} \sum_{\boldsymbol{p}_r \in W} w\left(\boldsymbol{p}_r\right)\left(1-\operatorname{NCC}\left(\boldsymbol{I}_r\left(\boldsymbol{p}_r\right), \boldsymbol{I}_n\left(\boldsymbol{H}_{r n} \boldsymbol{p}_r\right)\right)\right)
  $$
*  **Geometric Regularization Loss**
   *  最终得到几何相关的loss项：

    $$
    \mathbf{L}_{geom} = \lambda_2 \mathbf{L}_{svgeom} + \lambda_3 \mathbf{L}_{mvrgb} + \lambda_4 \mathbf{L}_{mvgeom}
    $$

## Exposure Compensation Image Loss

* 作者认为，之前3DGS没考虑光照变化，**因此在实际场景中会出现浮动的噪点**。作者为每个图像添加两个曝光系数$a, b$，用这两个系数可以计算出有曝光补偿的图像：
  $$
  \mathbf{I}_i^a = \exp(a_i)\mathbf{I}_i^r + b_i
  $$
* 对应的loss如下：
  $$
  \begin{gathered}
  \boldsymbol{L}_{r g b}=(1-\lambda) \boldsymbol{L}_1\left(\tilde{\boldsymbol{I}}-\boldsymbol{I}_i\right)+\lambda \boldsymbol{L}_{S S I M}\left(\boldsymbol{I}_i^r-\boldsymbol{I}_i\right) . \\
  \tilde{\boldsymbol{I}}= \begin{cases}\boldsymbol{I}_i^a, & \text { if } \boldsymbol{L}_{S S I M}\left(\boldsymbol{I}_i^r-\boldsymbol{I}_i\right)<0.5 \\
  \boldsymbol{I}_i^r, & \text { if } \boldsymbol{L}_{S S I M}\left(\boldsymbol{I}_i^r-\boldsymbol{I}_i\right)>=0.5\end{cases}
  \end{gathered}
  $$
  + 其中 $\boldsymbol{I}_i$ 为真实图像。
# Experiment

<p align="center">{% asset_img table.png %}</p>
<p align="center">{% asset_img result_1.png %}</p>
<p align="center">{% asset_img result_2.png %}</p>

## 实测数据

<p align="center">{% asset_img chair.png %}</p>
