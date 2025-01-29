---
layout: post
title: UC Berkeley CS285-RL(Lecture 1—Introduction)
date: 2025-01-27 15:48:13
tags: [机器学习, 基础知识, UCB, CS285, 课程, 数学, 神经网络, 深度学习, 强化学习]
math: true
categories: 强化学习
excerpt: UC Berkeley CS285-RL课程第一章学习笔记
---
<p align="center">{% asset_img no1.png %}</p>

- 视频链接：https://www.youtube.com/watch?v=SupFHGbytvA&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=1
- 资料：https://rail.eecs.berkeley.edu/deeprlcourse/

# 监督学习和强化学习

- 监督学习：
    - 监督学习的数据会默认数据是独立同分布的假设：
        - x的label不会影响另一个x的label
        - 并且他们的分布是相同的，及从输入x到输出y的映射函数，对所有的样本都是相同的
    - 对于每一个输入数据，都需要知道其对应输出的ground truth。**（但对于抓取任务来说，很难对每个物体找到最佳的抓取位置，而且从不同的位置抓取都可以完成该任务，因此并不适合使用监督学习）**
- 强化学习：
    - 强化学习的数据并不满足**独立同分布**，并且之前的输出会影响到未来的输入（有时序性）
    - 通常不知道gt的结果，而是只知道当前任务是成功还是失败，或者说我们只知道当前action的reward

<p align="center">{% asset_img no2.png %}</p>

- 强化学习的过程是一个时序的过程，Agent在t时刻采取$action_t$和环境进行交互，此时世界发生变化，并计算出当前状态$state_{t+1}$的$reward_{t+1}$。
    - 注意，在每个时刻，RL并不会计算每个event是好是坏，而是用奖励函数来描述当前action。但即使$state_{t+1}$是一个非常好的状态，也并不是$action_t$的作用，而是历史所有$action$的作用。
- 强化学习考虑的目标，是所有时刻reward的求和最大化，而不是某一个时刻的。

<p align="center">{% asset_img no3.png %}</p>

# example

- 文生图中也可以用RL增强效果，例如先由模型生成一个初始版本的图片，然后结合LLaVA生成图片的描述，并与input word计算reward，最终不断迭代。

<p align="center">{% asset_img no4.png %}</p>

# Data-Driven和RL之间的差别
- Data-Driven AI
    - 更擅长从已有的数据中重新索引并给出答案，但并不会基于已有的数据创造新的内容。因此也导致这种方法，必须给出足够多的数据，覆盖范围足够广，才能有比较好的效果。
- Reinforcement Learning
    - 设置的目标是解决一个明确的问题，因此如何利用历史信息来解决当前的问题是其主要目标，因此在数据的利用方式上和DD的方法是不一样的。
    - 但要考虑如何处理大规模的数据

<p align="center">{% asset_img no5.png %}</p>

# RL目前的仍未解决的问题
1. 深度学习可以很好的从巨量的数据中学习，也有很好的强化学习的优化方法，但目前还没找到既能使用数据又能使用深度学习的出色方法。
2. 人类的学习速度非常快，而深度强化学习的学习速度仍然很慢。
3. 人类可以重复使用过去的知识，而强化学习中的迁移学习仍然是一个悬而未决的问题
4. 奖励函数应该如何设计并不明确
5. 预测的作用应该是什么并不明确

<p align="center">{% asset_img no6.png %}</p>
