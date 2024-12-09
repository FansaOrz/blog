---
title: Transformer学习笔记
date: 2024-12-09 22:19:30
tags: [机器学习, 深度学习, 计算机视觉, 论文解读]
math: true
categories: 机器学习
excerpt: Transformer介绍
---

# 先验知识

## 迁移学习

- 迁移学习(Transfer Learning)是深度学习领域的一个重要研究方向。它通过利用已有模型的预训练参数来解决新任务的学习问题。

- 预训练是一种从头开始训练模型的方式：所有的模型权重都被随机初始化，然后在没有任何先验知识的情况下开始训练：

<p align="center">{% asset_img pretraining.svg pretraining %}</p>

- 这个过程不仅需要海量的训练数据，而且时间和经济成本都非常高。因此大部分情况下，我们是将别人预训练好的模型权重通过迁移学习应用到自己的模型中，即使用自己的任务语料对模型进行“二次训练”，通过微调参数使得模型适用于新任务

- 迁移学习的好处是：

  - 预训练时模型很可能已经见过与任务类似的数据集，通过微调可以激发出模型在预训练过程中获得的知识，将基于海量数据获得的统计理解能力应用于我们的任务；

  - 由于模型已经在大量数据上进行过预训练，微调时只需要很少的数据量就可以达到不错的性能；

  - 在自己任务上获得优秀性能所需的时间和计算成本都可以很小。

- 举例：我们可以选择一个在大规模英文语料上预训练好的模型，使用 arXiv 语料进行微调，以生成一个面向学术/研究领域的模型。这个微调的过程只需要很少的数据：我们相当于将预训练模型已经获得的知识“迁移”到了新的领域，因此被称为迁移学习。

- **在绝大部分情况下，我们都应该尝试找到一个尽可能接近我们任务的预训练模型，然后微调它，也就是所谓的“站在巨人的肩膀上”。**

# Transform 结构

- 标准的 Transformer 模型主要由两个模块构成：
  - 编码器(Encoder)：负责理解输入文本，为每个输入构造对应的语义表示（语义特征）
  - 解码器(Decoder)：负责生成输出，使用 Encoder 输出的语义表示结合其他输入来生成目标序列

<p align="center">{% asset_img transformers_blocks.svg Transformer结构 %}</p>

- 这两个模块可以根据任务的需求而单独使用：

  - **纯 Encoder 模型**：适用于只需要理解输入语义的任务，例如：句子分类、命名实体识别；

  - **纯 Decoder 模型**：适用于生成式任务，例如文本生成

  - **Encoder-Decoder 模型或 Seq2Seq 模型**：适用于需要基于输入的生成式任务，例如翻译、摘要。

- 原始结构

- Transformer 模型本来是为了翻译任务而设计的。在训练过程中，**Encoder 接受源语言的句子作为输入**，而 **Decoder 则接受目标语言的翻译作为输入**。

  - 在 Encoder 中，由于翻译一个词语需要依赖于上下文，因此注意力层可以访问句子中的所有词语；
  - 而 Decoder 是顺序地进行解码，在生成每个词语时，注意力层只能访问前面已经生成的单词。

- 举例，假设翻译模型当前已经预测出了三个词语，我们会把这三个词语作为输入送入 Decoder，然后 Decoder 结合 Encoder 所有的源语言输入来预测第四个词语。

        实际训练中为了加快速度，会把整个目标序列都送入Decoder，然后在注意力层中通过Mask遮盖掉未来的词语来防止信息泄露。

<p align="center">{% asset_img transformers.svg transformers %}</p>

# 注意力机制

## Attention

- NLP 神经网络模型的本质就是对输入文本进行编码，常规的做法是首先对句子进行分词，然后将每个词语(token)都转化为对应的词向量（token embedding），这样文本就有转换为一个由词语向量组成的矩阵$\boldsymbol{X} = (\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_n)$，其中$\boldsymbol{x}_i$表示第$i$个词语的词向量，维度为$d$，故$\boldsymbol{X}$的维度为$\boldsymbol{X} \in \mathbb{R}^{n \times d}$。

- 《Attention is All You Need》**直接使用 Attention 机制编码整个文本**。相比 RNN 要逐步递归才能获取全局信息，而 CNNN 只能获取局部信息，需要通过层叠来增大感受野，Attention 机制一步到位获取了全局信息：

$$
\boldsymbol{y}_t = f(\boldsymbol{x}_t, \boldsymbol{A}, \boldsymbol{B})
$$

- 其中$\boldsymbol{A}, \boldsymbol{B}$是另外的词语序列（矩阵），如果取$\boldsymbol{A} = \boldsymbol{B} = \boldsymbol{X}$，就是 self-attention，\*\*即直接将$\boldsymbol{x}_t$与自身序列中每个词语进行比较，最后算出$\boldsymbol{y}_t$。

### Scaled Dot-Product Attention

<p align="center">{% asset_img scaled_dot_production_attention.png scaled_dot_production_attention %}</p>

- Scaled Dot-Product Attention 是一种计算两个向量的相似度的方法，它通过将两个向量进行**点积来衡量它们之间的相似度**。共包含两个步骤：

  - **计算注意力权重**：使用某种相似度函数度量每个 query 向量和所有 key 向量之间的关联程度。对于长度为$m$的 Query 序列和长度为$n$的 Key 序列，该步骤会生成一个$m \times n$的注意力分数矩阵。

    - 这里使用点积作为相似度函数，相似的 queries 和 keys 会有较大的点积。

    - 由于点积可以产生任意大的数字，这会破坏训练过程的稳定性，因此注意力分数还需要乘以一个**缩放因子**来标准化它们的方差，然后用 softmax 标准化。这样就得到了最终的注意力权重$\omega_{ij}$，表示第$i$个 query 与第$j$个 key 的相似度。

  - **更新 token embedding**：将权重$\omega_{ij}$与 value 向量$\boldsymbol{v}_1,..., \boldsymbol{v}_n$相乘以获得第$i$个 query 向量更新后的语义表示$\boldsymbol{x}_i^{,} = \sum_j\omega_{ij}\boldsymbol{v}_j$。

- 形式化表示为：

$$
\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\operatorname{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^{\top}}{\sqrt{d_k}}\right) \boldsymbol{V}
$$

- 其中$\boldsymbol{Q} \in \mathbb{R}^{m \times d_k}, \boldsymbol{K} \in \mathbb{R}^{n \times d_k}, \boldsymbol{V} \in \mathbb{R}^{n \times d_v}$ 分别是 query、key、value 向量序列。如果忽略 softmax 激活函数，实际上它就是三个$m \times d_k, d_k \times n, n \times d_v$矩阵相乘，得到一个$m \times d_v$ 的矩阵，也就是将$m \times d_k$的序列$Q$编码成了一个新的$m \times d_v$的序列。

- 将上面的公式拆开来看：

$$
\operatorname{Attention}\left(\boldsymbol{q}_t, \boldsymbol{K}, \boldsymbol{V}\right)=\sum_{s=1}^n \frac{1}{Z} \exp \left(\frac{\left\langle\boldsymbol{q}_t, \boldsymbol{k}_s\right\rangle}{\sqrt{d_k}}\right) \boldsymbol{v}_s
$$

- pytorch 实现：

```python
import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

```

- **注意！** 上面的做法会带来一个问题：当$Q$和$K$序列相同时，注意力机制会为上下文中的**相同单词**分配非常大的分数（点积为 1），而在实践中，**相关词往往比相同词更重要**。因此，多头注意力 (Multi-head Attention) 出现了！

# 参考文档

- [transformers 快速入门](https://transformers.run/c1/transformer/#transformer-%E7%9A%84%E7%BB%93%E6%9E%84)
