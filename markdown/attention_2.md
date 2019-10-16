# BERT

[TOC]



---

# 1. Intro

BERT, Bidirectional Encoder Representation from Transformers.

分析全称的关键字

- Bidirectional: 双向提取上下文，而非常见的单向Left-to-Right提取上下文
- Encoder Representation of Transformers：基于Transformer模型中的Encoder结构，而Transformer中的Encoder本身就是双向的。

总的来说，BERT即以Transformer的Encoder结构为底子，构造出来一个更大更深的网络模型结构，并以此作为NLP领域大多数任务的预训练模型，取得了很多state-of-art的成果。

# 2. 背景

使用语言模型预训练的结果时，通常有如下两种策略：

- feature-based

    主体网络结构是task-specific的，仅用pre-trained representations作为额外的features

- fine-tuning

    主体网络结构为pre-trained model，仅使用task-specific结构作为下游输出，然后对整个模型依据task进行fine-tuning

BERT隶属于fine-tuning的策略，其预训练好的模型后面直接接task-specific的输出即可。

在BERT之前，模型们都是使用单向language model作为预训练的目标函数，这就导致对于一些任务来说，之前的模型们效果并不好。

因此BERT采用Masked Language Model (MLM)来作为预训练的目标函数，进而把前面和后面的context融合到一一起，增强预训练模型的效果。

# 3. 模型结构

BERT的模型结构实际上就是：多层的双向Transformer Encoder。

（关于Transformer的详细解释可查看之前的文章 [从Attention到BERT（二）：Transformer](https://zhuanlan.zhihu.com/p/85313597)）

BERT实际上有两个版本的预训练模型，分别是

（L: layers, H: hidden size, A: number of self-attention heads）

- BERT-base：L=12, H=768, A=12, Total Parameters=110M
- BERT-large：L=24, H=1024, A=16, Total Parameters=340M

而BERT模型的训练主要分两步：Pre-training和Fine-Tuning

## 3.1 Pre-training

BERT会在unlabeled data上训练Mask Lauguage Model (MLM)和Next Sentence Prediction (NSP)两个任务。

MLM是为了训练token-level的语义理解，NSP是为了训练sentence-level的语义理解。

对于MLM task来说，训练数据生成器会随机选取15%的token用于mask。如果一个token被选中，那么有80%的概率会被替换成 [MASK] token，有10%的概率被替换成其他随机的token，有10%的概率保持不变。

对于NSP task来说，每个训练样本会包括A和B两个sentence。其中B有50%的概率是A之后的sentence，有50%的概率不是A之后的sentence。

## 3.2 Fine-Tuning

先用预训练好的参数初始化模型，然后使用task-specific labeled data对整个模型的所有参数进行end-to-end的fine-tuning

# 4. Refer

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)