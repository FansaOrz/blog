---
title: UNET和FPN
date: 2024-12-15 10:39:39
tags: [机器学习, 基础知识, 数学, 神经网络, 深度学习]
math: true
categories: 机器学习
excerpt: Unet和FPN的介绍
---

# FPN(Feature Pyramid Network)
* 论文链接：[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

* 这篇论文是用特征金字塔做目标检测的。FPN网络将特征金字塔模型引入Faster R-CNN中，在不牺牲内存和检测速度的情况下，达到了SOTA的效果，同时对**小物体**的检测也获得不错的效果。

* 在目标检测任务重，不同尺度下的目标识别是一个很大的挑战。之前多数的object detection方法都是只采用顶层特征做预测（卷积的最后一层），**底层的特征语义信息比较少，但目标位置准确；顶层的特征语义信息比较丰富，但目标位置不够准确**。

* 有以下四种方法：
<p align="center">{% asset_img method.png %}</p>

* 图a，使用图像金字塔的方式生成多尺度特征。先生成很多不同分辨率的图像，然后对不同尺度下，通过ConvNet前向过程生成各自的feature map
* 图b，CNN网络最常用的结构，只用最后一层feature map做预测
* 图c，SSD的做法，从conv4开始每一层的feature都做预测，就得到了多尺度下的特征，但这样每一层预测时都没用到最顶层的特征，而最顶层的特征信息对小物体检测效果更好
* 图d，FPN的做法，多了一个上采样的过程，然后和浅层的特征融合，在独立做预测。

<p align="center">{% asset_img fpn.png %}</p>

* 一个是自底向上的线路，一个是自顶向下的线路。放大的区域就是横向连接。这里1x1的卷积核主要作用是降维，不改变尺寸大小。

<p align="center">{% asset_img fpn_struct.png %}</p>

# UNET
* 论文链接：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

<p align="center">{% asset_img unet.png %}</p>

* 前半部分就是特征提取，后半部分是上采样。也是Encoder-Decoder结构。
    - Encoder：由两个3x3的卷积层再加上一个2x2的池化层构成下采样的模块
    - Decoder：有一个上采样的卷积层（反卷积）+特征拼接+两个3x3的卷积层构成的上采样的模块
# 两种方法的异同
* 同：都使用了“自底向上”、“横向连接”及“自顶向下”的结构，从而对多尺度特征图进行融合，即将高层的语义信息与低层的几何细节结合。另外，融合后都会再经过一层卷积。
* 异：
    1.  FPN对多尺度特征图融合的方式是element-wise add（每个元素对应相加），而UNet采用的是concate；
    2. FPN对多尺度特征图都进行了预测，而UNet仅在（由上至下）最后一层进行预测，而且这一层通常还需要进行一次resize才能恢复到原图尺寸；
    3. FPN对高层特征图采用的放大方式是插值，而UNet通常还会使用**转置卷积**，通过网络自学习的方式来进行上采样；
    4. FPN的高层特征放大2倍后与低层的尺寸恰好一致，而在UNet中通常不一致，还需要对低层特征做crop使得与放大后的高层特征尺寸一致；
    5. FPN在下采样时的卷积带有padding，分辨率的下降仅由stirde决定，而UNet的卷积通常不带padding，使得分辨率下降在stride的基础上还会额外的减小。也就是说，FPN的“由下至上”和“由下至上”是对称结构，而UNet其实是非对称的，这也是导致4和2中最后提到的原因‘；
    6. FPN在特征层融合后经过一层卷积是为了消除上采样过程中产生的混叠效应带来的影响，而UNet中还起到了压缩通道的作用（也是由于UNet融合特征层时采用的是concate，因此需要压缩通道减少计算量）；
    7. FPN主要针对detection任务，而UNet针对segmentation任务，前者通常作为一个模块嵌入到网络结构中，而后者本身就是一种网络模型结构。
# 参考文档
* [FPN与U-Net](https://blog.csdn.net/weixin_51015047/article/details/121242275)
* [目标检测的FPN和Unet有差别吗?](https://www.zhihu.com/question/351279839)
* [【原理篇】一文读懂FPN(Feature Pyramid Networks)](https://blog.csdn.net/Eyesleft_being/article/details/120989953)