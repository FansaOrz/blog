---
layout: post
title: NaVid阅读笔记（Vision-Language-Navigation的端到端导航模型）
date: 2025-01-12 14:00:59
tags: [视觉重建, 机器学习, VLN, VLA, Transformer, 深度学习, 论文解读, BERT]
math: true
categories: VLA
excerpt: "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation. "
---

# 相关文档
* 项目链接：https://pku-epic.github.io/NaVid/
* 代码链接：https://github.com/jzhzhang/NaVid-VLN-CE
* 论文链接：https://arxiv.org/pdf/2402.15852

# Introduction
- 文章的目标是输入一组人类的指引语言，以及根据相机的画面信息，直接输出导航指令。
- 如果理解复杂的视觉信息以及理解详细的指令，是一件十分困难的事情。