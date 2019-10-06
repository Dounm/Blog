# 从Attention到BERT（二）：Transformer

本系列文章是我的学习笔记，包括如下三个部分：

- 从Attention到BERT（一）：Attention Mechanism
- 从Attention到BERT（二）：Transformer
- 从Attention到BERT（三）：BERT

如有不足，还请多多指教。

---

[TOC]

## 1. 介绍

传统的sequence transduction model都是基于复杂的recurrent or convolutional NN。

以recurrrent model为例，其主要思路是把计算量沿着输入输出序列的位置分解。

> recurrent models typically factor computation along the symbol positions of the input and output sequences.

因此，这种**有序计算的特性就阻止了训练的并行**。因为对于长序列来说，因为序列的每个token都需要内存存储embedding，内存的大小就限制了batch size。

而Transformer模型的提出，就是为了解决传统模型难以并行训练的特点的。



## 2. Transformer的模型架构

![1570274619258](https://pic3.zhimg.com/80/v2-4ff256f06a702786cc815ee580f2e5c6_hd.jpg)



### 2.1 Encoder：左侧

由`N=6`层相同的layer组成。

每个layer由两个sub-layer组成：

1. multi-head self-attention mechanism
2. simple position-wise fully connected feed-forward network

此外每个sub-layer还都有【residual connection】和【normalization】

所以整的来说，每个sublayer的输出为：
$$
LayerNorm(x + Sublayer(x))
$$


### 2.2 Decoder：右侧

decoder同样由`N=6`个相同的层组成。

除了encoder的两层之外，还添加了一层multi-head attention over the output of the encoder stack。

具体而言：

- 输入：**encoder的输出 + 对应`i-1`位置的decoder的输出**。

  中间的multi-head attention并非self-attention，其`K`/`V`来自encoder，但`Q`来自`i-1`位置的decoder

- 输出：对应`i`位置的词的概率分布

- 解码过程：**encoder可以并行计算，一次性全encode出来。但decoder仍然需要像普通的rnn一样，一步步解码出来。**因为要用上个位置的输入当作attention的query。



除此之外，注意decoder部分最下面的是**masked multi-head attention**。因为训练时output sequence都是已知的，我们要**屏蔽(mask)掉当前位置后面位置的信息**，确保其不会接触未来的信息



## 3. Transformer中的Attention

### 3.1 Scaled Dot-Product Attention

![1570275297646](https://pic3.zhimg.com/80/v2-04b9fda4710d0651ed9f6e689dae67ce_hd.jpg)

attention的计算流程是：**由Query Token和每一个Key Token计算（经由`match()`）出来多个weight，然后计算weight与Value Token的weight sum**。

此处Scaled Dot-Product Attention也是类似，其**`match()`函数是点积**（即图中MatMul）



通常是对每个Query Token分别计算其Attention Value，但此处为了并行，我们**同时计算多个Query Token**。

令Query Token和Key Token是`d_k`维，Value Token是`d_v`维，同时计算`q_size`个Query Token，且Key/Val Sequence中共`p`个Key/Val Token。

然后把`q_size`个Query Token组成矩阵`Q(q_size * d_k)`，`p`个Key/Val Token组成矩阵`K(p * d_k)`, `V(p * d_v)`，那么上图计算流程如下
$$
Attention(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$
注：

- softmax()是沿着`p`维计算的

- 传统的Dot-Product Attention并没有scale factor $\frac{1}{\sqrt{d_k}}$，之所以缩放是为了减少点积结果的方差（类似normalize）

  > 大方差会导致weight有的大有的小，softmax会导致大的更大，小的更小（详见Attention Is All You Need 3.2.1）

### 3.2 Multi-Head Attention

![1570281607246](https://pic2.zhimg.com/80/v2-1eca9c1fd05b788fb86ef02a8185934d_hd.jpg)

一般attention中，`Q`/`K`/`V`都是相同维度`d_model`的向量。但paper中发现对`Q`/`K`/`V`做一下线性变换(即矩阵乘法)更好。

线性变换后，`Q`/`K`/`V`的维度就分别变成了`d_k`,`d_k`,`d_v`，然后经由Scaled Dot-Product Attention之后，得出`h`个`d_v`维的attention_value，对应`h`个Query Token。

我们把这`h`个attention_value给**concat在一起**(`h*d_v`维向量)，再做一次线性变换(仍变换为`d_model`维向量)，就得到了某个Query Token的attention_value的值



所以，整体流程如下： 

1. 一个Query Token,Key Token,Value Token(都是`d_model`维)，分别经过`h`次**不同且可学习的线性映射**后，得到`h`个`Q(d_k)`,`K(d_k)`,`V(d_v)`（`d_k = d_v = d_model/h`）。
2. `h`个`Q`,`K`,`V`经由`h`个Scaled Dot-Product Attention得到`h`个attention_value(`d_v`)
3. 把这`h`个`d_v`维的attention_value给concat(`h*d_v`)到一起，经过线性映射，再次形成`d_model`维的向量。

其对应的公式为  
$$
\begin{aligned} \text { MultiHead }(Q, K, V) &=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{h}\right) W^{O} \\ \text { where head }_{i} &=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right) \end{aligned} \\ W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{k}} , W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{k}} , W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}} \\ W^{O} \in \mathbb{R}^{hd_{v} \times d_{model}}
$$
注：

- 共有$h*3+1$个线性变换矩阵，它们都可学习



### 3.3 Transformer中Attention的应用 

- decoder的中间部分的Multi-Head Attention，其`Q`来自上一层的decoder，但`K`/`V`来自encoder。

  Output Sequence中的每个position都能访问到Input Sequence中的所有position的信息。

- encoder中Multi-Head Attention是self-attention

  encoder中的每个position都能访问到上一层encoder的所有position的信息。

- decoder中也类似

  decoder每个position都能访问到该位置以及该位置之前的所有position的信息，训练时后面的position会被mask掉。



## 4. Transformer中的其他子结构

### 4.1 Position-Wise Feed-Forward Networks 

每一层由两个sublayer，

- attention sublayer

- fully connected feed-forward network：对每个position分别执行.

  长为`n`的seq，共`n`个position，所以输入是`n*d_model`

其公式如下：
$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$
即两个FC之间夹着一个ReLU。 

注意，对于同一layer的不同position而言，其所作的线性变换是一致的（即`W1`/`W2`不变）；但不同Layer的线性变换（矩阵）却不是一样的。



### 4.2 Embeddings & Softmax 

Transformer和其余模型一样，

- 都要用learned embeddings来把input/output seq里的token变成`d_model`维的向量；

  `vocab_dim → d_model`

- 也要用learned transformation和softmax把decoder的输出变成概率。

  `d_model → vocab_dim`

注意，第2节的Figure 1中的**input/output embedding 和 softmax之前的linear transformation是相同的矩阵**。



### 4.3 Positional Encoding

#### 4.3.1 基本介绍

因为Transformer模型中没有Recurrence/Convolution的缘故，所以无法获取sequence的顺序和位置信息（类似于词袋模型）。

而位置信息又对模型的效果影响很大，因此我们必须得传入一些relative or absolute position of tokens in the sequence。 

因此，Transformer引入了【Positional Encoding】来编码输入输出序列的位置信息。

（【Positional Encodings】和【word embedding】都是`d_model`维，这两个在Transformer中是**累加**在一起的。）



Transformer用的是sine/cosine function来计算positional encoding。 
$$
\begin{array}{l}{P E_{(p o s, 2 i)}=\sin \left(\operatorname{pos} / 10000^{2 i / d_{model}} \right)} \\ {P E_{(p o s, 2 i+1)}=\cos \left(\operatorname{pos} / 10000^{2 i / d_{model}} \right)}\end{array}
$$
其中

- `pos`是position，代表是sequence的第几个token

- `i`是第几维（共`d_model`维）

- $PE_{(pos,2i)}$和$PE_{pos,2i+1}$的下标$2i/2i+1$代表的是偶数和奇数的意思。

  令$f(x)=pos/{10000^{x/d_{model}}}$，具体的计算方法是遍历$i={0,1,...,d_{model-1}}$ ，当$i$是偶数时，计算$sin(f(i))$；当$i$是奇数时，计算$cos(f(i-1))$。

- `d_model`维的positional encoding vector中，每一维的频率和波长都不一样。

  第0/1维是$sin(pos)$和$cos(pos)$，波长为$2\pi$；最后一维根据$d_{model}$奇偶不同可能是$sin(pos/10000)$或$cos(pos/10000)$，波长为$10000*2\pi$。




#### 4.3.2 三角函数用作positional encoding的性质与优势

- sin/cos的值域为$[-1,1]$

  无论是多长的序列，其值域都在$[-1,1]$之间，就不会出现[extrapolation](<https://en.wikipedia.org/wiki/Extrapolation> )的问题

  

- 三角函数公式
  $$
  \begin{array}{l}{\sin (\alpha+\beta)=\sin (\alpha) \cos (\beta)+\cos (\alpha) \sin (\beta)} \\ {\cos (\alpha+\beta)=\cos (\alpha) \cos (\beta)-\sin (\alpha) \sin (\beta)}\end{array}
  $$
  从relative position的角度考虑

  ​	对于任意固定的$k$而言，$PE_{pos+k}$都可以看作是$PE_{pos}$的**线性组合**。

  ​	推导过程如下：

  ​	令$a=10000^{2i/d_{model}}$，则$a$是一个常量。 
  $$
  PE_{pos+k,i}=sin(\frac{pos+k}{a})=sin{(\frac{pos}{a}+\frac{k}{a})} \\= sin(\frac{pos}{a})cos(\frac{k}{a})+cos(\frac{pos}{a})sin(\frac{k}{a})
  $$
  ​	因为$k$也是固定的，所以$cos(k/a)$和$sin(k/a)$都是常量，令其分别为$u$,$v$，则 
  $$
  PE_{pos+k,i}=u*PE_{pos,i}+v*PE_{pos,i+1}
  $$
  ​	对于固定的k而言，无论sequence多长，两个position之间的差值也不会变。这就保证了relative position的不变性。

  

  从absolute position的角度考虑

  ​	如果把某一position与position 0相比较，因为position 0始终不变，**相对位置就变成了绝对位置**，相当于也encode了绝对位置的信息。

  

- 三角函数具有周期性

  因此对于确定的某一维$i$而言，相隔整数周期的两个position相同。

  但因为**不同维周期不同**，所以并不会出现两个position的encoding vector完全一致的情况。



Positional Encoding的实现代码参见[Positional Encoding](http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding) 。

#### 4.3.3 可学习的positional encoding

positional embedding也可以学习得到。据paper所说，效果与sin/cos一致。但之所以选sin/cos，是因为其可以**extrapolate比所有train sequence更长的序列**。



## 5. Reference

- [Attention Is All You Need](<https://arxiv.org/abs/1706.03762>)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [【NLP】Transformer详解](https://zhuanlan.zhihu.com/p/44121378)
- [从《Convolutional Sequence to Sequence Learning》到《Attention Is All You Need》](https://zhuanlan.zhihu.com/p/27464080)