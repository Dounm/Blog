# 从Attention到BERT（一）：Attention Mechanism

本系列文章是我的学习笔记，包括如下三个部分：

- 从Attention到BERT（一）：Attention Mechanism
- 从Attention到BERT（二）：Transformer
- 从Attention到BERT（三）：BERT

如有不足，还请多多指教。

---

[TOC]

## 1. 原理 

Attention是用于提升基于RNN（LSTM/GRU）的Encoder+Decoder模型效果的机制（Mechanism），所以一般称为Attention Mechanism。

在传统的Encoder-Decoder结构中，Encoder把输入序列编码成**统一的语义特征`C`**再Decode。因此，**`C`中必须包含原始序列的所有信息，它的长度就会限制模型性能**。

Attention机制中，在**每个阶段输入不同的`c`**来解决该问题。每个`c`**经过训练后会自动选出**与当前要输出的`y`最相关的上下文信息

![1](https://pic3.zhimg.com/80/v2-ba462bb981cf15a190ecf47029b20072_hd.jpg)



## 2. 机器翻译的例子

以机器翻译为例，加入了Attention机制的translation model如下图所示 ![1570196867287](https://pic1.zhimg.com/80/v2-0c1a968887e7b22af1652fadb00497a4_hd.jpg)

在上图中，

- $\alpha_0^1$是标量，也是$h^1$对应的权重/概率

  为保证所有概率和为1，需要经过softmax

- Decoder阶段，$c^1$和前一个词（如`machine`）一起作为下一阶段的输入



那么如何计算权重呢？

![1570197313473](https://pic2.zhimg.com/80/v2-f343b16f5851d01cce9969fe18d34015_hd.jpg)

以左上图为例
$$
\alpha_0^1 = match(z^0, h^1) \\ \alpha_0^2 = match(z^0, h^2) \\ \alpha_0^3 = match(z^0, h^3) \\ \alpha_0^4 = match(z^0, h^4)
$$
而`match()`函数有如下可能情况

1. cosine similarity of $z$ and $h$
2. small NN whose input is $z$ and $h$, output a scalar $\alpha$
3. $\alpha = h^TWz$

其中2,3都有weight需要通过训练得到



## 3. Attention的通用框架

### 3.1 介绍

对于一个attention过程而言，可看做是如下图所示的过程![1570198083791](https://pic4.zhimg.com/80/v2-8d799f650a7f877cc9a2aa87a3333b9f_hd.jpg)

**mapping a query and a set of key-value pairs to an output(Attention Value) **

**（weight assigned to each value is computed by a compatibility function of the query with the corresponding key.）** 

注意

- query/key/value都是**向量**
- output则是**标量**，是value的 **weight sum**



针对于1.2节所示的例子而言，而说

- query: `z0`,`z1`
- key: `h1`,`h2`,`h3`,`h4`
- value: `h1`,`h2`,`h3`,`h4`



### 3.2 序列的角度

更进一步，我们从sequence的角度再来看一下这个Attention的框架

所有的`query`是一个sequence，所有的`key`在一起是一个sequence，所有的`value`在一起是一个sequence。

Attention过程就是：

遍历`query seq`，对其中的每个`query token`，计算其与`key seq`里每一个`key token`的相关关系作为权重。

因为每一个`key token`对应一个`value token`，所以最终`attention value`就是所有`value`的**weight sum**。



### 3.3 总结

总结而言，Attention流程的公式如下：
$$
Attention(Q,K,V)=softmax(sim(Q,K))V
$$


## 4. Attention的变种

Soft/Hard Attention

- soft attention：传统attention，可被嵌入到模型中去进行训练并传播梯度
- hard attention：不计算所有输出，依据概率对encoder的输出采样，在反向传播时需采用蒙特卡洛进行梯度估计



Global/Local Attention

- global attention：传统attention，对所有encoder输出进行计算
- local attention：介于soft和hard之间，会预测一个位置并选取一个窗口进行计算



Self Attention

又叫intra-attention。传统attention是计算Q和K之间的依赖关系，而**self attention的Q/K/V相同**，`query_seq`/`key_seq`/`val_seq`都是同一个sequence。

即对该sequence，遍历其中的每个token作为query，计算其与seq中的其余token之间的相关性，然后以之作为该token的embedding。



## 5. Reference

- [Attention-based Model - 李宏毅 ](<http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html>)

- [【NLP】Attention原理和源码解析 - 李入魔的文章 - 知乎 ](https://zhuanlan.zhihu.com/p/43493999)
- [模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用](https://zhuanlan.zhihu.com/p/31547842)