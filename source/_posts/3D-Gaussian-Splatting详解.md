---
title: 3D Gaussian Splatting学习笔记
date: 2024-04-13 13:59:38
tags: [视觉重建, 机器学习, MLP, 计算机视觉, 论文解读]
math: true
categories: 视觉重建
excerpt: 3D Gaussian Splatting for Real-Time Radiance Field Rendering. 对3D Gaussian Splatting的原理和代码进行详细解读
---

# 相关文档

## 3DGS

- 项目链接：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- 代码链接：https://github.com/graphdeco-inria/gaussian-splatting

- 论文链接：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf

# NeRF 和 3DGS 的区别

<p align="center">{% asset_img nerf_3dgs.png nerf_3dgs %}</p>

# 什么是 Splatting

## 定义

- 一种体渲染的方法：从 3D 物体渲染到 2D 平面

- Ray-casting 是被动的（NeRF）

  - 计算出每个像素点受到发光粒子的影响来生成图像

- Splatting 是主动的

  - 计算出每个发光粒子如何影响像素点

# 为什么选择 3D 高斯椭球

- 很好的数学性质

  - 仿射变换后高斯核仍然闭合

  - 3D 降维到 2D 后（沿着某一个轴积分（z 轴））仍然为高斯

- 定义：

  - 椭球高斯函数：

$$G(x) = \frac{1}{\sqrt{2\pi^k|\Sigma|}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

    - 其中$\Sigma$表示协方差矩阵，半正定，$|\Sigma|$是其行列式，$\mu$表示均量

## 3D gaussian 为什么是椭球？

- 对于椭球高斯函数：$G(x) = \frac{1}{\sqrt{2\pi^k|\Sigma|}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$来说，${\sqrt{2\pi^k|\Sigma|}}$始终是一个常数，变量只存在于${-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$中，当${(x-\mu)^T\Sigma^{-1}(x-\mu)}=constant$时，其可以展开为

$$
\begin{align}
constant &= (x-\mu)^T\Sigma^{-1}(x-\mu) \\
         &= \frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} + \frac{(z-\mu_z)^2}{\sigma_z^2} - \frac{2\sigma_{xy}(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}-\frac{2\sigma_{xz}(x-\mu_x)(z-\mu_z)}{\sigma_x\sigma_z}-\frac{2\sigma_{yz}(y-\mu_y)(z-\mu_z)}{\sigma_y\sigma_z}
\end{align} \\

equal \quad to:          Ax^2+By^2+Cz^2+2Dxy+2Exz+2Fyz=1


$$

其中,

$$
\Sigma = \begin{pmatrix}
\sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\
\sigma_{xy} & \sigma_y^2 & \sigma_{yz} \\
\sigma_{xz} & \sigma_{yz} & \sigma_z^2
\end{pmatrix}
$$

最终就是一个**椭圆面**的表示形式。而${(x-\mu)^T\Sigma^{-1}(x-\mu)}$是有取值范围的，所以整体表现出来就是一个**实心的椭球体**

# 如何利用旋转和缩放计算协方差矩阵

**对应于 cuda 代码中的 computeConv3D**

- 高斯分布：

       - $\boldsymbol{x}\sim N(\mu, \Sigma)$

       - 均值：$\mu_x, \mu_y, \mu_z$

       - 协方差矩阵：$\Sigma = \begin{pmatrix}

  \sigma*x^2 & \sigma*{xy} & \sigma*{xz} \\
  \sigma*{yx} & \sigma*y^2 & \sigma*{yz} \\
  \sigma*{zx} & \sigma*{zy} & \sigma_z^2
  \end{pmatrix}$

- 高斯分布的仿射变换：

  - $\boldsymbol{w} = A\boldsymbol{x}+b$

  - $\boldsymbol{w} \sim N(A\mu+b, A\Sigma A^T)$

  - 协方差和 b 没有关系

- 标准高斯分布：

  - $\boldsymbol{x} \sim N(\bold{0}, I)$

  - 均值：$\bold{0}$

  - 协方差矩阵：$\boldsymbol{\Sigma} = \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{pmatrix}$

  - $\boldsymbol{\Sigma} = \boldsymbol{A} \cdot \boldsymbol{I} \cdot \boldsymbol{A}^T$ **任意高斯可以看作是标准高斯通过仿射变换得到**

- $\boldsymbol{A}=RS$，其中$\boldsymbol{R}$是旋转矩阵，$\boldsymbol{S}$是缩放矩阵，因此

$$
\begin{align}
\boldsymbol{\Sigma} &= \boldsymbol{A} \cdot \boldsymbol{I} \cdot \boldsymbol{A}^T\\
&=\boldsymbol{R}\boldsymbol{S}\boldsymbol{S}^T\boldsymbol{R}^T
\end{align}
$$

# Splatting 过程介绍——从 3D 到像素

## 变换矩阵介绍

### 观测变换

<p align="center">{% asset_img 3d_viewing_trans.png 3d_viewing_trans %}</p>

- 从世界坐标系到相机坐标系，仿射变换

- $\boldsymbol{w} = \boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}$

### 投影变换

- 3D 到 2D

- 透视投影，和 Z 轴有关

- 正交投影，和 Z 轴无关

<p align="center">{% asset_img projection.png projection %}</p>

#### 正交投影

- 对于一个立方体$[l, r]\times[b, t]\times[f, n]$，将其平移到原点，然后缩放为$[-1, 1]\times[-1, 1]\times[-1, 1]$的正方体

- 投影矩阵：

$$
\begin{bmatrix}
2/(r-l) & 0 & 0 & 0 \\
0 & 2/(t-b) & 0 & 0 \\
0 & 0 & 2/(n-f) & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & -(r+l)/2 \\
0 & 1 & 0 & -(t+b)/2 \\
0 & 0 & 1 & -(n+f)/2 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

<p align="center">{% asset_img orthographic_projection.png orthographic_projection %}</p>

#### 透视投影

- 有远小近大的原则，投影出来是一个锥体。因此需要先把锥体压成立方体，然后再进行正交投影

<p align="center">{% asset_img persp_projection.png persp_projection %}</p>

- 透视投影到立体的过程：

$$
\boldsymbol{M}_{persp->ortho} =
\begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

- **注意：透视投影是非线性的，不是仿射变换**

### 视口变换

- 不管是透视投影还是正交投影，最终得到的都是$[-1, 1]^3$范围内的立方体，然后进行视口变换将其映射回图片的大小 height \* width，恢复出原始比例

<p align="center">{% asset_img view_point.png view_point %}</p>

$$
\boldsymbol{M}_{viewpoint} =
\begin{bmatrix}
\frac{w}{2} & 0 & 0 & \frac{w}{2} \\
0 & \frac{h}{2} & 0 & \frac{h}{2} \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

## 3D 高斯的观测变换过程，如何把物理坐标系的物体映射到像素空间

**对应代码中的 computeCov2D**

### 第一步：从物理坐标系到相机坐标系

- 物理坐标系：

  - 高斯核的中心点：$\boldsymbol{t}_k = [t_x, t_y, t_z]^T$

  - 高斯核对应的高斯分布：$r^{,,}_k(t) = G_{V_k^{,,}}(t-t_k)$

  - 其中$V_k^{,,}$是协方差矩阵

- 相机坐标系：

  - 高斯核中心：$\boldsymbol{\mu}_k = [\mu_x, \mu_y, \mu_z]^T$

  - 高斯核对应的高斯分布：$r^{,}_k(t) = G_{V_k^{,}}(\mu-\boldsymbol{\mu}_k)$

  - 均值：$\boldsymbol{\mu}_k = Wt_k+d$

  - 协方差矩阵：$V_k^{,} = WV_k^{,,}W^T$

- 这一步就是简单做个仿射变换，3D 空间的坐标变换

### 第二步：从相机坐标系到像素空间

- 相机坐标系：

  - 高斯核中心：$\boldsymbol{\mu}_k = [\mu_x, \mu_y, \mu_z]^T$

  - 协方差矩阵：$V_k^{,}

- 投影变换：（非线性的变换，需要做泰勒展开做局部的线性近似）

  - 高斯核中心：$\boldsymbol{x}_k = [x_x, x_y, x_z]^T$

  - 高斯核对应的高斯分布：$r_k(t) = G_{V_k}(x-\boldsymbol{x}_k)$

  - 均值：$\boldsymbol{x}_k = m(\mu_k)$（对均值可以直接做非线性变换，不需要局部线性近似）

  - 协方差矩阵：$V_k = \boldsymbol{J}V_k^{,}\boldsymbol{J}^T$

  - 其中$\boldsymbol{J} = \frac{\partial m(\mu_k)}{\partial \mu}$，雅可比矩阵

  - 至此，协方差矩阵$V_k = \boldsymbol{J}WV_k^{,,}W^T\boldsymbol{J}^T$

#### 如何求得雅可比矩阵？

- 已知：

$$
\boldsymbol{M}_{persp->ortho} =
\begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

- 视锥中一个点:$[x y z 1]^T$对其应用投影变换后：

$$
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
1
\end{bmatrix} =
\begin{bmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & -1 & 0
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix} =
\begin{bmatrix}
nx \\
ny \\
(n+f)z-nf \\
z
\end{bmatrix} =
\begin{bmatrix}
\frac{nx}{z} \\
\frac{ny}{z} \\
(n+f)-\frac{nf}{z} \\
1
\end{bmatrix}


$$

- 由此，雅可比矩阵为：

$$
\boldsymbol{J} =
\begin{bmatrix}
\frac{df_1}{dx} & \frac{df_1}{dy} & \frac{df_1}{dz} \\
\frac{df_2}{dx} & \frac{df_2}{dy} & \frac{df_2}{dz} \\
\frac{df_3}{dx} & \frac{df_3}{dy} & \frac{df_3}{dz} \\
\end{bmatrix} =
\begin{bmatrix}
\frac{n}{z} & 0 & -\frac{nx}{z^2} \\
0 & \frac{n}{z} & -\frac{ny}{z^2} \\
0 & 0 & -\frac{nf}{z^2} \\
\end{bmatrix}
$$

## 3D 高斯球的颜色表示

这里是把高斯椭球的颜色表示成了一个颜色球，这个颜色球从不同角度观察，都可以看到不同的颜色。但是颜色球的颜色应该怎么描述呢？这里就是使用球谐函数来对其做近似，然后从不同角度对其做渲染

<p align="center">{% asset_img sh_01.png sh_01 %}</p>

### 球谐函数介绍

- 任何一个球面坐标的函数，都可以用多个球谐函数来近似

- $f(t) \approx \sum_l\sum^l_{m=-l}c_l^my_l^m(\theta, \phi)$

- 其中，$l$表示当前的阶数，$c_l^m$是各项系数，$y_l^m$是基函数

<p align="center">{% asset_img sh.png sh %}</p>

$$
\begin{align}
f(t) &\approx \sum_l\sum^l_{m=-l}c_l^my_l^m(\theta, \phi) \\
= &c_0^0y_0^0 + \\
&c_1^{-1}y_1^{-1} + c_1^{0}y_1^{0} + c_1^{1}y_1^{1}+ \\
&c_2^{-2}y_2^{-2} + c_2^{-1}y_2^{-1} + c_2^{0}y_2^{0} + c_2^{1}y_2^{1} + c_2^{2}y_2^{2} +\\
&...
\end{align}
$$

- 其中各个基函数为：

$$
y_0^0 = \sqrt{\frac{1}{4\pi}} \\
y_1^{-1} = \sqrt{\frac{3}{4\pi}}\frac{y}{r}\\
y_1^{0} = \sqrt{\frac{3}{4\pi}}\frac{z}{r}\\
... \\
$$

- 由此，当 xyz 固定时，基函数就是固定的，但是系数是变化的。当系数维度为 3 时，共有 16 个系数，**即一个高斯球，有 16 个颜色系数要估计**

## 使用$\alpha$ -blending 计算像素的颜色

**与 NERF 中公式相同**

$$
\begin{align}
C &= T_i \alpha_i c_i \\
 &= \sum_{i=1}^N T_i(1-e^{-\sigma_i\delta_i}c_i)
\end{align} \\
where: T_i = r^{-\sum_{j=1}^{i-1}\sigma_i\delta_j}
$$

- 其中：

  - $T_i$：在 s 点之前，光线没有被阻挡的概率

  - $\alpha_i$：在 s 点处，光线被遮挡的概率。包含两点：1. 高斯椭球本身的不透明度；2. 该像素点距离高斯椭球中心越远，高斯椭球对其影响越小

  - $c_i$：在 s 点处，高斯球的颜色。即通过球谐函数近似得到的颜色

**注意，此时是在 2D 平面上计算像素值**

- 当一轮迭代之后，参数都固定了，此时就可以计算像素的颜色值。3D 高斯在投影到 2D 平面上后，还是一个高斯分布。利用之前的理论，当${(x-\mu)^T\Sigma^{-1}(x-\mu)}=constant$时，在二维上表现为一个椭圆：$\frac{(x-\mu_1)^2}{\sigma_1^2} + \frac{(y-\mu_2)^2}{\sigma_2^2} - \frac{2\sigma_{xy}(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} = constant$。此时$\mu_1$和$\mu_2$为高斯中心点在像素平面上的坐标，$\sigma_1$和$\sigma_2$为二维高斯协方差矩阵对角线上的元素。

# 代码解析

- 已添加注释到：https://github.com/FansaOrz/gaussian-splatting

## diff-gaussian-rasterization 解析

<p align="center">{% asset_img rasterization.png rasterization %}</p>

### 预处理

- 批量并行处理高斯点，为光栅化做准备

```C++
// 共划分了P/256个线程块，每个线程块包含256个线程
// 创建的是一维线程块，每个高斯核对应一个线程
preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,
        means3D,
        scales,
        rotations,
        opacities,
        ...
        );
```

- 每个线程中执行了以下步骤：
<p align="center">{% asset_img thead.png thread %}</p>

#### 剔除视锥外的点

```C++
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
    ...) {

}
```

#### 计算高斯核对应的 tile

<p align="center">{% asset_img grid.png tile %}</p>

# 其他

## 名词解释：

- EWA:
- NDC 坐标系：Normalized Device Coordinates，标准化设备坐标系
- footprint（足迹）：椭球投影到 2D 平面上的扩散
- NVS: novel view synthesis，新视角合成

## 常见疑问

1. 为什么 3d 高斯降维到 2d？
   - 因为 splatting 算法就是要把 3d 的 voxel 投射到像素屏幕然后做叠加
2. 怎么降到 2d？
   - 沿着 z 做积分
3. 为什么选择高斯？
   - 因为高斯变成 2d 还是高斯，协方差矩阵直接拿掉 3 行 3 列即可（这个相当重要，直接省略了积分了！！！！）
4. 为什么协方差矩阵不用做平移缩放？
   - 感觉应该是 splatting 在投影矩阵之后的那个空间里做，点要做视口变换是为了和后面的像素匹配，做后面的 tile 渲染

# 参考文档

- [b 站视频：【较真系列】讲人话-3d gaussian splatting 全解(原理+代码+公式)](https://www.bilibili.com/video/BV1zi421v7Dr?spm_id_from=333.788.player.switch&vd_source=9629687338410a5ccaa5e1a595d0f17d)

- [A Survey on 3D Gaussian Splatting](https://arxiv.org/pdf/2401.03890)
