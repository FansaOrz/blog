---
layout: post
title: Cross-Attention流程介绍
date: 2024-8-11 20:32:51
tags: [机器学习, 基础知识, 数学, 神经网络, 深度学习, Attention, Transformer]
math: true
categories: 机器学习
excerpt: Cross-Attention介绍
---

# Self-Attention

$$
Softmax(\frac{Q \cdot K^T}{\sqrt(d)}) \cdot V
$$

## 举例

* 图像中如何使用Self-Attention

<p align="center">{% asset_img image_attentionss.png %}</p>

* 假设图像分成了4块，每块都有RGB三个通道，将其展开可以得到一个4x3的“矩阵”（每一块图像看做一个元素）

<p align="center">{% asset_img image_blocks.png %}</p>

* 现在想根据这个4x3的矩阵计算出对应的QKV，我们首先先随机初始化QKV的权重$W^Q, W^K, W^V$，并将其与图像矩阵相乘，得到QKV。此时的QKV维度均为$4 \times 2$：

<p align="center">{% asset_img first_cal.png %}</p>

* 对于上式$Softmax(\frac{Q \cdot K^T}{\sqrt(d)}) \cdot V$来说，首先计算$Q \cdot K^T$。由于Q和K的维度均为$4 \times 2$，因此计算后维度为$4 \times 4$，每个数字代表两个矩阵之间的相似程度。因此$Q \cdot K^T$可以解释为相似性度量。

* 对$Q \cdot K^T$执行Softmax，并和矩阵V相乘，得到最终的输出。**和矩阵V相乘可以理解为注意力机制，即根据Q和K的相似程度对V进行加权**。
<p align="center">{% asset_img softmax.png %}</p>

<p align="center">{% asset_img V.png %}</p>

* 以上就是Self-Attention的计算过程。完全根据自身的像素块，计算互相之间的相似程度，然后根据相似程度对V进行加权。

* 最后要把Self-Attention的结果映射回原始的图像维度，增加一个$W^{out}$用于映射。并把映射的结果和原始图像进行相加，得到最终的输出。
    - 相加的原因：
        * 原始的每个像素也需要对自己给予很大的关注度
        * 这样Self-Attention可以控制什么时候跳过self-attention操作

<p align="center">{% asset_img projection.png %}</p>

<p align="center">{% asset_img add_x.png %}</p>

# Cross-Attention
* Cross-Attention和Self-Attention不同，这里只由Q是从原始的图像输入中计算而来，K和V是条件信息的投影。

* K和V以文字的输入形式举例，

<p align="center">{% asset_img cross-attention.png %}</p>

<p align="center">{% asset_img encoder.png %}</p>

<p align="center">{% asset_img shape.png %}</p>

* 例如输入文字内容为“I love mountains”，Encoder之后的输入维度为$3 \times 6$，Token长度是3，每个token都编码为6维向量，batch size为1

<p align="center">{% asset_img cross_attention_QKV.png %}</p>

* 此时得到了$Q \in 4 \times 2$ 和 $K \in 3 \times 2, V \in 3 \times 2$。 继续执行和Self-Attention一样的操作。首先计算$ Q \cdot K^T $。

<p align="center">{% asset_img cross_01.png %}</p>

- SoftMax

<p align="center">{% asset_img cross_02.png %}</p>

- 最后是和V相乘，这里可以理解为每个像素，对于每个token的关注度，或者参考的权重值是多少。

<p align="center">{% asset_img cross_03.png %}</p>

- 使用$W^{out}$来把cross-attention的结果映射回原始的图像维度。

<p align="center">{% asset_img cross_04.png %}</p>

- 再和输入的图像相加，得到最终的输出。

# 参考文档
* [动画解释 Self Attention & Cross Attention](https://www.bilibili.com/video/BV1Ke411X7t7/?spm_id_from=333.337.search-card.all.click&vd_source=9629687338410a5ccaa5e1a595d0f17d)
