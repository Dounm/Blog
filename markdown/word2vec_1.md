# Word2Vec-知其然知其所以然（二）：模型详解

---

[TOC]

## 4 基于Hierarchical Softmax的模型

Word2vec中的两个重要模型是：CBOW模型（Continuous Bag-of-Words Model）和Skip-gram模型（Continuous Skip-gram模型）。对于这两个模型，Word2vec给出了两套框架，分别是基于Hierarchical Softmax和Negative Sampling来设计的。本节先介绍Hierarchical Softmax框架。

### 4.1 CBOW模型

CBOW模型全名为 $Continous \;bag-of-words$。之所以叫 $bag-of-words$ 是因为输入层到投影层的操作由『拼接』变成了『叠加』，对于『叠加而言』，无所谓词的顺序，所以称为词袋 $bag-of-words$ 模型。

![2](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/3.png)
上图中：

- 上下文$Context(w)$：由$w$前后各$c$个词构成
- 输入层：包含$2c$个词的词向量
- 投影层：讲输入层的$2c$个词做『累加求和』。$X_w=\sum_{i=1}^{2c}v(Context(w_i)) \in \mathbb{R}^m$
- 输出层：输出层对应的是一棵Huffman树。该Huffman树以语料中出现的词为叶子节点，以各词在语料中**出现的次数当做权值**构造出来的。该树中，叶子节点总共有$N$个（$N=|D|$）。


对比CBOW模型结构和上面的神经概率语言模型的模型结构，区别在于：

- 输入层到投影层：CBOW是『累加』，神经概率语言是『拼接』
- 隐藏层：CBOW无隐藏层，神经概率语言有
- 输出层：CBOW是『树形结构』，神经概率语言是『线性结构』

#### 4.1.1 梯度计算

首先是如下几个符号的意思：

- $p^w$：从根节点到$w$对应的叶子节点的路径
- $l^w$：路径$p^w$上包含的节点的个数
- $p_1^w,p_2^w,\dots,p_{l^w}^w$：路径$p^w$上对应的节点
- $d_2^w,d_3^w,\dots,d_{l^w}^w\in \{0,1\}$：路径上的节点对应的Huffman编码，根节点不对应编码
- $\theta_1^w,\theta_2^w,\dots,\theta_{l^w}^w\in \mathbb{R}^m$：路径上的**『非叶子节点』**对应的词向量

从根节点出发到某个叶子节点的路径上，每次分支都可视为进行了一次『二分类』。默认左边（编码为0）是负类，右边（编码为1）是正类。

- 分为正类的概率：$\sigma(X_w^T\theta)=\frac{1}{1+e^{-X_w^T\theta}}$
- 分为负类的概率：$1-\sigma(X_w^T\theta)$

其中的$\theta$即当前『非叶子结点』的对应的词向量。

所以Hierarchical Softmax的思想就是：
**对于词典$D$中的任意词$w$，Huffman树中必存在一条从根节点到词$w$对应叶子节点的路径$p^w$。路径$p^w$上存在$l^w-1$个分支，将每个分支看做一次二分类，每次分类就产生一个概率，讲这些概率连乘，即$p(w|Context(w))$。**
$$
p(w|Context(w))=\prod_{j=2}^{l^w}p(d_j^w|X_w,\theta_{j-1}^w) \\
其中 \\ 
p(d_j^w|X_w,\theta_{j-1}^w) = 
\begin{cases} 
\sigma(X_w^T\theta_{j-1}^w) & d_j^w=0 \\ 
1-\sigma(X_w^T\theta_{j-1}^w) & d_j^w=1
\end{cases} \\
= [\sigma(X_w^T\theta_{j-1}^w)]^{1-d_j^w} \cdot [1-\sigma(X_w^T\theta_{j-1}^w)]^{d_j^w}
$$

带入最大似然的公式，得
$$
\mathcal{L}=\sum_{w\in C}log\prod_{j=2}^{l^w}\{[\sigma(X_w^T\theta_{j-1}^w)]^{1-d_j^w} \cdot [1-\sigma(X_w^T\theta_{j-1}^w)]^{d_j^w}\} \\ 
= \sum_{w\in C}\sum_{j=2}^{l^w}\{(1-d_j^w) \cdot log[\sigma(X_w^T\theta_{j-1}^w)]+d_j^w \cdot log[1-\sigma(X_w^T\theta_{j-1}^w)] \}
$$

令$\mathcal{L}(w,j)$等于如下内容：
$$
\mathcal{L}(w,j)=(1-d_j^w) \cdot log[\sigma(X_w^T\theta_{j-1}^w)]+d_j^w \cdot log[1-\sigma(X_w^T\theta_{j-1}^w)]
$$

因为要求最大似然，所以对上式采用『随机梯度上升法』。

首先对$\theta$求导：
$$
\frac{\partial\mathcal{L}(w,j)}{\partial\theta_{j-1}^w} = \frac{\partial}{\partial\theta_{j-1}^w}\{(1-d_j^w) \cdot log[\sigma(X_w^T\theta_{j-1}^w)]+d_j^w \cdot log[1-\sigma(X_w^T\theta_{j-1}^w)]\} \\ 
利用[log\,\sigma(x)]'=1-\sigma(x),[log(1-\sigma(x))]'=-\sigma(x) 得\\ 
=(1-d_j^w)[1-\sigma(X_w^T\theta_{j-1}^w)]X_w-d_j^w\sigma(X_w^T\theta_{j-1}^w)X_w \\ 
= \{(1-d_j^w)[1-\sigma(X_w^T\theta_{j-1}^w)]-d_j^w\sigma(X_w^T\theta_{j-1}^w)\}X_w \\ 
= [1-d_j^w-\sigma(X_w^T\theta_{j-1}^w)]X_w
$$

因为$X_w$在$L(w,j)$的表达式中和$\theta$是对称的，所以：
$$
\frac{\partial\mathcal{L}(w,j)}{\partial X_w} = [1-d_j^w-\sigma(X_w^T\theta_{j-1}^w)]\theta_{j-1}^w
$$

在对词向量进行更新时，因为$X_w$表示的是$Context(w)$中各词词向量的叠加，所以$X_w$的更新也要贡献到$Context(w)$中的每个词的词向量上去
$$
v(\bar w):=v(\bar w)+\eta\sum_{j=2}^{l^w}\frac{\partial\mathcal{L}(w,j)}{\partial X_w},\bar w\in Context(w)
$$

#### 4.1.2 伪代码
![3](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/4.png)
注意，步骤3.3/3.4不能交换位置，$\theta$应该等$e$更新后再更新。


### 4.2 Skip-gram模型

Skip-gram模型是已知当前词$w$，对其上下文$Context(w)$中的词进行预测。
![4](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/5.png)
网络结构：

- 输入层：当前样本的中心词$w$的词向量
- 投影层：其实没什么作用
- 输出层：类似CBOW模型，也是一颗Huffman树

#### 4.2.1 梯度计算
$$
p(Context(w)|w)=\prod_{u\in Context(w)}p(u|w) \\ 
基于前面的Hierarchical\; Softmax思想 \\ 
p(u|w)=\prod_{j=2}^{l^u}p(d_j^u|v(w),\theta_{j-1}^u) \\ 
= [\sigma(v(w)^T\theta_{j-1}^u)]^{1-d_j^u}\cdot [1-\sigma(v(w)^T\theta_{j-1}^u)]^{d_j^u}
$$

因此带入最大似然函数，得
$$
\mathcal{L}=\sum_{w\in C}log\prod_{u\in Context(w)}\prod_{j=2}^{l^u}\{[\sigma(v(w)^T\theta_{j-1}^u)]^{1-d_j^u}\cdot [1-\sigma(v(w)^T\theta_{j-1}^u)]^{d_j^u}\} \\
= \sum_{w\in C}log\sum_{u\in Context(w)}\sum_{j=2}^{l^u}\{(1-d_j^u)\cdot log[\sigma(v(w)^T\theta_{j-1}^u)] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)]\}
$$

令$\mathcal{L}(w,u,j)$等于如下内容：
$$
\mathcal{L}(w,u,j)=(1-d_j^u)\cdot log[\sigma(v(w)^T\theta_{j-1}^u)] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)]
$$

计算$\mathcal{L}(w,u,j)$对$\theta_{j-1}^{u}$的梯度：
$$
\frac{\partial\mathcal{L}(w,u,j)}{\partial\theta_{j-1}^u} = \frac{\partial}{\partial\theta_{j-1}^u} \{(1-d_j^u)\cdot log[\sigma(v(w)^T\theta_{j-1}^u)] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)]\}\\ 
利用[log\,\sigma(x)]'=1-\sigma(x),[log(1-\sigma(x))]'=-\sigma(x) 得\\ 
= (1-d_j^u)[1-\sigma(v(w)^T\theta_{j-1}^u)]v(w)-d_j^u\sigma(v(w)^T\theta_{j-1}^u)v(w) \\
= \{(1-d_j^u)[1-\sigma(v(w)^T\theta_{j-1}^u)]-d_j^u\sigma(v(w)^T\theta_{j-1}^u)\}v(w) \\
= [1-d_j^u-\sigma(v(w)^T\theta_{j-1}^u)]v(w)
$$

同样由于参数$v(w)$与参数$\theta$是对称的，所以
$$
\frac{\partial\mathcal{L}(w,u,j)}{\partial v(w)} = [1-d_j^u-\sigma(v(w)^T\theta_{j-1}^u)]\theta_{j-1}^u
$$

所以梯度更新公式如下：
$$
\theta_{j-1}^u := \theta_{j-1}^u + \eta[1-d_j^u-\sigma(v(w)^T\theta_{j-1}^u)]v(w) \\
v(w) := v(w) + \eta\sum_{u\in Context(w)}\sum_{j=2}^{l^u}[1-d_j^u-\sigma(v(w)^T\theta_{j-1}^u)]\theta_{j-1}^u
$$

#### 4.2.2 伪代码

skip-gram的伪代码如下：
![5](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/6.png)

但是在Word2vec的源码中，并非等$Context(w)$中所有词都处理完后才更新为v(w)，而是每处理完$Context(w)$中的一个词$u$就更新一次$v(w)$。
![6](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/7.png)


## 5 基于Negative Sampling的模型

### 5.1 如何选取负样本

选取负样本需要按照一定的概率分布，Word2vec的作者们测试发现**最佳的分布是$\frac{3}{4}$次幂的$Unigram \;distribution$**。

啥是$Unigram \;distribution$？$Unigram$来自于$Unigram\;Model$（即一元模型），它认为语料库中所有的词出现的概率都是互相独立的。所以$Unigram \;distribution$就是按照在语料库中随机选择，因此高频词被选中的概率大，低频词被选中的概率小，这也很符合逻辑。

概率分布公式如下：
$$
p(w)=\frac{[count(w)]^{\frac{3}{4}}}{\sum_{u \in D}[count(u)]^{\frac{3}{4}}}
$$

### 5.2 CBOW模型

CBOW模型中，我们是已知词$w$的上下文$Context(w)$，需要预测$w$。

设我们已经选好了一个关于$w$的负样本子集$NEG(w)$，并且定义了对于词典$D$中的任意词$w'$，都有
$$
L^w(w')=
\begin{cases} 
1 & w'=w \\ 
0 & w'\neq w
\end{cases} 
$$

对于一个给定的正样本$(Context(w),w)$，我们希望最大化
$$
g(w)=\prod_{u\in \{w\}\bigcup NEG(w)} p(u|Context(w)) \\
其中\\
p(u|Context(w))=
\begin{cases} 
\sigma(X_w^T\theta^u) & L^w(u)=1 \\ 
1-\sigma(X_w^T\theta^u) & L^w(u)=0
\end{cases} \\
= [\sigma(X_w^T\theta^u)]^{L^w(u)} \cdot [1-\sigma(X_w^T\theta^u)]^{1-L^w(u)}
$$

#### 5.2.1 梯度计算

所以，
$$
g(w)=\sigma(X_w^T\theta^w)\prod_{u\in NEG(w)} [1-\sigma(X_w^T\theta^u)]
$$

为什么要最大化$g(w)$？因为$\sigma(X_w^T\theta^w)$表示的是上下文为$Context(w)$时，预测中心词为$w$的概率；而$\sigma(X_w^T\theta^u)$表示的是上下文为$Context(w)$时，预测中心词为$u$的概率（即一个二分类）。因此最大化$g(w)$即相当于增大正样本的概率，同时降低负样本的概率，而这就是我们所期望的。

> 上面这段话仅仅是方便理解$g(w)$的作用。其实此处的$g(w)$代表的不再是$p(w|Context(w))$，而是$P(D|w,Context(w))$。即不再求条件概率分布，而是联合概率分布，这正是NCE的思想。

> 如果要形式化的证明为什么选择$g(w)$作为求最大似然的对象，请阅读参考资料4。

所以对于给定语料库$C$来说，整体的优化目标即为最大化$G=\prod_{w\in C}g(w)$。则$Loss$函数为：
$$
\mathcal{L}=log G=log\prod_{w\in C}g(w)=\sum_{w \in C} log\,g(w) \\
= \sum_{w \in C} log \prod_{u\in \{w\}\bigcup NEG(w)}  \{[\sigma(X_w^T\theta^u)]^{L^w(u)} \cdot [1-\sigma(X_w^T\theta^u)]^{1-L^w(u)}\} \\ 
= \sum_{w \in C} \sum_{u\in \{w\}\bigcup NEG(w)}  \{L^w(u) \cdot log[\sigma(X_w^T\theta^u)] + [1-L^w(u)] \cdot log[1-\sigma(X_w^T\theta^u)]\}
$$

令$\mathcal{L}(w,u)$等于如下式子：
$$
\mathcal{L}(w,u)=L^w(u) \cdot log[\sigma(X_w^T\theta^u)] + [1-L^w(u)] \cdot log[1-\sigma(X_w^T\theta^u)]
$$

采用随机梯度上升法，计算梯度
$$
\frac{\partial\mathcal{L}(w,u)}{\partial\theta^u}=\frac{\partial}{\partial\theta^u} \{L^w(u) \cdot log[\sigma(X_w^T\theta^u)] + [1-L^w(u)] \cdot log[1-\sigma(X_w^T\theta^u)]\} \\
利用[log\,\sigma(x)]'=1-\sigma(x),[log(1-\sigma(x))]'=-\sigma(x) 得\\
= L^w(u)[1-\sigma(X_w^T\theta^u)]X_w - [1-L^w(u)]\sigma(X_w^T\theta^u)X_w \\
= [L^w(u)-\sigma(X_w^T\theta^u)]X_w
$$

根据对称性：
$$
\frac{\partial\mathcal{L}(w,u)}{\partial X_w}=[L^w(u)-\sigma(X_w^T\theta^u)]\theta^u
$$

所以，对于词向量的更新为：
$$
v(w'):=v(w')+\eta\sum_{u\in \{w\}\bigcup NEG(w)}\frac{\partial\mathcal{L}(w,u)}{\partial X_w}\;,w'\in Context(w)
$$

#### 5.2.2 伪代码
![8](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/8.png)

### 5.3 Skip-gram模型

思路类似，略。详细推理见参考资料1。



## 6 总结

在Word2Vec的训练过程中，每个word vectors都被要求为相邻上下文中的word的出现作预测，所以尽管我们是随机初始化Word vectors，但是这些vectors最终仍然能通过上面的预测行为捕获到word之间的语义关系，从而训练到较好的word vectors。

正如同上文所暗示的，Word2Vec尽量让具有相同上下文的word的向量相似，从而获得较好的vector representation的相似性。这种相似性有时候是线性的，例如$vec(King)-vec(Man)+vec(Queue)$的结果会与$vec(Woman)$相似，即Word2vec可以学习到词与词之间语义上的联系。

另外，由于Word2Vec采用了非常多的方法简化网络结构，简化训练流程，导致Word2Vec可以很轻易的训练超大的训练集。据参考资料3说，一个优化后的单机实现版的Word2Vec算法可以在一天时间内训练100 bililion words。

### 6.1 残留问题

- 为什么求最大似然就可以得到较好的词向量？
- 为什么所有的二分类采用的都是逻辑回归，都是Sigmoid函数？

## 7 X2Vec

### 7.1 Paragraph2Vec

和Word2vec一样，像Sentence/Paragrah/Document都可以表示为向量，这些统称为Paragraph2vec（之所以取Paragraph是表示可长可短，长可为一个Doc，短可为一个Sentence，原理都是一样的）。

传统的求Paragraph2Vec的方法有很多，例如将一个句子/短语的所有的word的向量加起来，就可以当做句子/短语向量。而对于文档级别的来说，最经典常见的定长向量就是Bag-of-Words了。Bag-of-Words即以词典大小作为文档向量维度，然后以word出现次数作为对应元素的值。

Bag-of-Words有两个局限性：其一是丢弃了word与word之间的顺序，另一个就是忽略了word的语义关系。因此Google后来提出了Paragraph2Vec的方法（见参考资料6），大致解决了这些问题。

Paragraph2Vec的模型中，这个Paragraph Vectors的被训练出来的主要目的是用来预测paragraph中的word。其结构图如下：
![9](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/9.png)

相比于word2vec，训练过程中新增了paragraph id，即给每个paragraph一个唯一的id。Paragraph id也会被映射为一个向量，即paragraph vector。paragraph vector与word vector的维数虽一样，但是来自于两个不同的向量空间，之间并不能执行计算相似度之类的操作。在之后的计算里，paragraph vector和word vector累加或者连接起来，作为输出层softmax的输入，如上图所示。

训练的方法就是随机梯度下降和BP算法。随机梯度下降的每一步，我们都是从一个随机的paragraph中采样出一个定长的Context，然后计算梯度并更新。对于相同的paragraph来说，paragraph vector是共享的。这相当于每次在预测单词的概率时，都利用了整个句子的语义。

注意在预测阶段，对于一个新的paragraph，先给它一个paragraph id，然后利用原先就已经训练好的word vectors和神经网络参数，**使用梯度下降不断的训练待预测句子**。收敛之后，就得到了待预测句子的paragraph vector。

### 7.2 推荐系统中的X2Vec

Wrod2Vec的思想启发了很多人，并且在推荐系统也大有作为。在Youtube于2016公布的论文（参考8）中谈到了在他们的视频推荐系统中是如何使用Word2Vec的思想的。

Youtube的推荐系统架构分为两部分：*Candidate Generation*和*ranking*。*Candidate Generation*就是从巨量的视频库里面挑选出挑选出几百部与user最相关的视频，然后*ranking*在对这几百部进行排序，挑选出最合适的推荐给用户。

其中，*Candidate Generation*他们以前用的是矩阵分解，现在用的是如下图所示的架构。
![10](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/word2vec/10.png)

如上图右上角所示，他们将推荐看做成了一个超大规模的多分类。分类的问题就是在给定用户$(user)U$和用户上下文$(context)C$的情况下，在视频库$(corpus)V$的百万视频中对一个观看记录进行分类，每个视频看做一类。
> We pose recommendation as extreme multiclass classification where the prediction problem becomes accurately classifing a specific video watch $w_t$ at time $t$ among millions of videos $i$ (classes) from a corpus $V$ based on a user $U$ and context $C$.

$$
p(w_t=i|U,C)=\frac{e^{v_iu}}{\sum_{j \in V}e_{v_ju}}
$$
公式如上面所示，其中$u\in \mathbb{R}^n$代表了$(user,context)$二元组的高维向量表示，$v_j\in \mathbb{R}^n$代表了候选视频的向量表示。

其实上面的公式就是求了个Softmax，所以同理，也是因为Softmax分母计算复杂度太高，所以Youtube也采用了Negative Sampling和Hierarchical Softmax两种优化方法。

除此之外，就训练数据的标签Label而言，虽然Youtube有赞或踩的选项，但是他们却没有用这些作为标签，而是把用户看完一个视频当做一个正样本。这是因为显式的赞或踩不够多，而隐式的看完视频的历史记录够多，所以使用隐式反馈就可以加强一些长尾的推荐效果。

上图左上角代表的是预测。每次预测的时候，输入user vector $u$和video vectors $v_j$，然后选取top $N$最近邻作为输出。

上图最下方的输入层代表的是各种各样的X2Vec。watch vector是以用户历史观看的视频的video vector做平均得到的。而search vector则是对用户的历史搜索记录做处理，先将query处理成unigram或bigram，然后再向量化，最后取平均得到的。再右侧的就是一些其他的特征了。

这些乱七八糟的拼接起来，经过好几层全连接的RELU层，到最顶层就得到了user vector $u$，然后带入Softmax的公式训练，在通过梯度下降和BP算法最终将更新传递回输入层。

## 7 参考资料

1. [word2vec中的数学原理详解](http://www.cnblogs.com/peghoty/p/3857839.html)
2. Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector space[J]. arXiv preprint arXiv:1301.3781, 2013.
3. Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[C]//Advances in neural information processing systems. 2013: 3111-3119.
4. Goldberg Y, Levy O. word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method[J]. arXiv preprint arXiv:1402.3722, 2014.
5. [word2vec: negative sampling (in layman term)?](http://stackoverflow.com/questions/27860652/word2vec-negative-sampling-in-layman-term/27864657#27864657)
6. Le Q V, Mikolov T. Distributed Representations of Sentences and Documents[C]//ICML. 2014, 14: 1188-1196.
7. [语义分析的一些方法(二)](http://www.flickering.cn/ads/2015/02/%E8%AF%AD%E4%B9%89%E5%88%86%E6%9E%90%E7%9A%84%E4%B8%80%E4%BA%9B%E6%96%B9%E6%B3%95%E4%BA%8C/)
8. Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations[C]//Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016: 191-198.

---

[^1]: Mikolov T, Sutskever I, Chen K, et al. Distributed representations of words and phrases and their compositionality[C]//Advances in neural information processing systems. 2013: 3111-3119.
