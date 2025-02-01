---
layout: post
title: CLIP论文阅读笔记
date: 2025-02-01 17:13:03
tags: [机器学习, 深度学习, 计算机视觉, 论文解读, Transformer, 多模态, 图片分类]
math: true
categories: 机器学习
excerpt: Learning Transferable Visual Models From Natural Language Supervision
---
<p align="center">{% asset_img clip.png %}</p>

# 相关文档
* CLIP（Contrastive Language-Image Pre-training）
* 项目链接：https://openai.com/index/clip/
* 代码链接：https://github.com/openai/CLIP
* 论文链接：https://arxiv.org/abs/2103.00020
* 视频介绍：[B站跟李沐学AI: CLIP 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1SL4y1s7LQ?spm_id_from=333.788.videopod.sections&vd_source=9629687338410a5ccaa5e1a595d0f17d)
# Introduction and Motivating Work
* 直接从原始的文本中预训练一个模型，过去几年在NLP领域中已经取得了革命性的成功，例如BERT, GPT等。并且NLP中都是自监督的目标函数，和下游任务是无关的，因此并不需要针对下游任务做任何的输出头和特殊处理，
* 然而在计算机视觉领域，主流的做法仍然是针对特定的类别去预训练一个模型。因此作者希望把NLP中的这类方法应用在计算机视觉中，在视觉中预训练一个模型，然后可以zero-shot的迁移到其他的视觉任务中。
* 之前也有一些利用自然语言做弱监督的工作，作者认为这些工作的效果不好，主要是差在了规模上。这里的规模既包括数据的规模，又包括模型的规模。CLIP的工作共采用了**4亿组图片文字对**，同时模型的规模也提上去了。作者这里共测试了8个视觉模型，最终发现**迁移学习的效果基本上和模型的参数大小是呈正相关的。**
# Approach

## Natural Language Supervision

<p align="center">{% asset_img method.png %}</p>

* **CLIP文章的核心，就是用自然语言的监督信号，来训练一个比较好的视觉模型。**这并不是一个新的idea，之前也有类似的方法，但是叫法比较混乱，实际上都是用文本做一个训练的信号，并且规模不够大。
* 为什么要用自然语言做监督，来训练一个视觉模型呢？
  1. 不需要再去标注这些数据了，不需要先固定类别的数量。并且监督信号不是1-N了，而是一些文本，因此模型的输入输出的自由度会比较高。
  2. 训练时会把文字和图片绑定到一起，模型学习到的特征一个是多模态的特征，因此就比较容易去做zero-shot的迁移学习。而如果只是一个图像特征的话，很难在后处理步骤将其和文本关联到一起。

## Creating a Sufficiently Large Dataset

* 目前已经有3个文本-图像的数据集：MS-COCO，Visual Genome，YFCC100M。前两个标注质量比较高，但是数据量只有10w个左右，YFCC100M有1亿数据左右，但是质量比较差。因此作者团队自己制作了4亿组数据，称为WIT(WebImageText)。

## Selecting an Efficient Pre-training Method

* 作者首先尝试了类似VirTex的方法，图像用CNN，文本用Transformer。训练任务是给定一张图片，预测图片对应的文本。
* 为什么要用对比学习呢？
  + 给定一张图片，来预测其描述。需要逐字逐句的输出，这个过程很难实现，因为一张图片的描述有很多种，模型可能的输出有很多，训练速度会很慢。
  + 但如果把任务变成了一个对比学习的任务，那么只需要判断这个图片和对应的文字是不是正确的配对，而不需要逐字逐句的去输出图片的描述。
* 作者发现，如果把预测型的目标函数换成对比型的目标函数，整个训练速度就提高了4倍。

<p align="center">{% asset_img efficient.png %}</p>

* 以下是整个CLIP的伪代码：
  + 首先有图片的输入$I$和文本的输入$T$
  + 分别利用两个编码器，得到图像和文本的特征
  + 对视觉和文本的特征，分别应用一个投射矩阵，使得模型学习如何将单模态的特征映射成多模态的特征，然后应用L2的归一化，得到最终用来对比的特征$I_e$和$T_e$
  + 计算两类特征之间的cosine similarities，得到最后用于分类的logits
  + 创建gt_label，计算交叉熵loss
  + 最后将Image和Text的loss算平均，最为最后的loss
  
<p align="center">{% asset_img numpy.png %}</p>

# Experiments
* 实验内容太多了，这里就不列举了，建议看原文。
<p align="center">{% asset_img experiment.png %}</p>
