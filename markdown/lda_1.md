# LDA算法理解（二）：Gibbs Sampling

---

系列文章分为两部分：

1. LDA算法理解（一）：数学基础
2. LDA算法理解（二）：Gibbs Sampling

---

[TOC]

## 3 Gibbs Sampling吉布斯采样
吉布斯采样是马尔科夫链蒙特卡罗法（Markov Chain Monte Carlo, MCMC）的一种。

### 3.1 Monte Carlo蒙特卡罗法
蒙特卡罗法即通过模拟采样的方式来让你获得想要的值。
举个例子，在一个正方形内，以正方形的中心为原点，边长为直径画一个圆（即正方形的内切圆）。
向正方形内均匀撒米，那么按道理来说圆内的米粒C与正方形内的米粒S满足：$\frac{C}{S} \approx \frac{\pi(\frac{d}{2})^2}{d^2}$
因此对于参数$\pi$来说，我们可以用$\pi\approx \frac{4C}{S}$来对它进行估计。

### 3.2 Markov Chain马尔科夫链
马尔科夫链即根据转移矩阵去转移的随机过程（马尔科夫过程）。
如下图就是一个典型的马尔科夫过程
![马尔科夫链](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/lda/1.png)
该状态转移图的转移矩阵如下图所示：
![转移矩阵](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/lda/2.png)
其中，$i,j,k,l$表示的马尔科夫链上的状态。$pij$表示从状态$i$到状态$j$转移的可能性。
现在利用向量$\pi=(i,j,k,l)$表示当前所处的状态。开始的时候$\pi_0=(1,0,0,0)$表示最开始处于状态$i$。
那么下一次转移后$\pi$变为$\pi_1=\pi_0*P=[P_{ii},P_{ij},P_{ik},P_{il}]$，此处向量$\pi_1$其实就是第一次转移之后的状态分布，即有$P_{ii}$的概率身处状态$i$，有$P_{ij}$的概率身处状态$j$。


#### 3.2.1 平稳状态分布Stationary Distribution
有一种情况，即向量$\pi$在经过大量的转移后达到一个稳定状态，之后即使再怎么转移$\pi$的值也不会改变了。此时$\pi$即成为平稳状态分布。（如果在平稳状态下我们继续在马尔科夫链中转移$n$次，那么即$n$次转移中位于状态$i$的次数为$\pi*P$）。
要达到这个平稳状态分布需要满足一些条件，即$\pi P=\pi$(也即$\pi_iP_{ij}=\pi_jP_{ji}$，这两个条件等价)。
举例而言，![马尔科夫例子](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/lda/3.png)
如果初始状态$\pi=(1,0,0)$的话，在多次乘以转移矩阵$p$之后，$\pi$最终等于$(0.625,0.3125,0.0625)$。这也就意味着如果我们在$\pi$收敛后继续转移状态的话，我们10000次转移，大约有6250次位于 Bull 状态，3125次位于 Bear 状态，625词位于 Stagnant 状态。
其实$\pi$就是一个概率分布，而我们构造出某个马尔科夫链（即转移矩阵）使得收敛到平稳状态分布后采样出来的结果满足$\pi$这个概率分布。
因此如果我们想求某个概率分布$P(X)$的话，我们就可以构造一个马尔科夫链来使得最终平稳状态分布就是概率分布$P(X)$，从而在无需明显求出$P(X)$表达式的情况下获取其采样结果。

### 3.3 Metropolis-Hasting算法
MH算法目的：根据一个需求的概率分布$P(x)$生成（采样）一系列的样本状态点。
为达到这一目的，MH算法通过构造马尔科夫链来使得该马尔科夫链最终的平稳分布为$P(X)$，然后再进行采样即可。
对于平稳状态分布的条件而言：
$$
P(x)p(x->x')=P(x')p(x'->x)
$$
$$
\frac{p(x->x')}{p(x'->x)} = \frac{P(x')}{P(x)}
$$
将转移概率$p(x->x')$分解为**建议概率$g(x->x')$和接受概率$A(x->x')$**，即$p(x->x')=g(x->x')A(x->x')$。建议概率是我们给出状态$x$后转移到状态$x'$的条件概率，而接受概率则是接受状态$x'$的条件概率。
则整理可得，
$$
\alpha=\frac{A(x->x')}{A(x'->x)}=\frac{P(x')}{P(x)}\frac{g(x'->x)}{g(x->x')}
$$
这样我们得到了接受率$\alpha$，代表的含义是：从状态$x$到状态$x'$的接受概率与从状态$x'$到状态$x$的接受概率的比率。对于$\alpha$来说，如果它大于1，就寿命下次要转移的状态X'比当前状态Xi可能性更大，那么我们就按照建议概率$g(x->x')$所建议的，转移到$x'$。如果它不大于1，例如为0.7，那么我们就有0.7的概率接受建议概率的建议转移到$x'$，有0.3的概率拒绝建议概率的建议仍然留在$x$处。
所以最终的话，我们倾向于留在高概率密度的地方，然后仅偶尔跑到低概率状态的地方（这也就是MH算法直观上的运行机理）。
MH算法将接受率$\alpha$添加了个上界1，得到如下的公式：
$$
\alpha=A(x->x')=min\{1, \frac{g(x'->x)}{g(x->x')}\}
$$
因此，MH算法的步骤为：
1. 选择任意一个状态点$Xt$作为初始状态。
2. 选择任意一个概率分布作为建议概率$g(x|y)$（建议概率是条件概率，而且必须满足$g(x|y)=g(y|x)$。通常会选择以$y$点为中心的正态分布）
3. 根据$g(X'|Xt)$生成下次状态点$X'$，计算接受率$\alpha$
4. 如果$\alpha==1$，则说明$X'$比$Xt$更有可能，因此就接受这次转移，下次状态点的状态就是X'
5. 如果$\alpha<1$，则以$\alpha$的概率接受这次转移。
6. 继续循环3-5步即可。

### 3.4 Gibbs Sampling算法
MH默认的流程是任意选择转移概率$g(x)$，然后利用接受率$\alpha$来使得采样最终收敛于$p(x)$。但是如果我选择足够好的$g(x)$，使得$g(x)$每次发出的建议都是符合$P(x)$分布的建议，那么我就一直接受就行了(此时即接受率恒为1)。Gibbs Sampling采用的就是这种方式。

对于多维随机变量的概率分布$p(\vec{x})$而言，选择**完全条件概率full conditionals** 作为建议概率，
$$
p(x_j|x_{-j})=p(x_j|x_1,\dots,x_{j-1},x_{j+1},\dots,x_n)=\frac{p(x_1,\dots,x_n)}{p(x_1,\dots ,x_{j-1},x_{j+1},\dots,x_n)}
$$
此时可以证明的是接受率$\alpha$恒等于1，即$g(x)$每次发出建议都是符合联合概率分布的，因此我们只需要一直接受建议即可。
证明如下：
![证明1](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/lda/4.png)

所以，对于多维随机变量的概率分布而言，一旦其完全条件概率full conditionals 可用，则可以采用$n​$维向量轮流每个维度循环的方式来迭代达到平衡。
![](https://raw.githubusercontent.com/Dounm/TheFarmOfDounm/master/resources/images/lda/5.png)

### 3.5 LDA模型的Gibbs Sampling应用
由2.5节可知，LDA模型所要计算的是$p(\vec{z}|\vec{w})$。
其中$\vec{w}$是文档集中的单词，是已知的可观测变量，所以我们把它当做已知值，则LDA想要的概率分布就是$p(\vec{z})$。而又因为$\vec{z}$是多维随机变量，结合前面Gibbs Sampling的思想，所以我们选取其完全条件概率$p(z_i|\vec{z}_{-i})$作为马尔科夫链的转移概率。此时考虑$\vec{w}$的因素，所以我们要用$p(z_i|\vec{z}_{-i},\vec{w})$作为转移概率。

#### 3.5.1 整个文本训练集生成的联合概率

要得到完全条件概率$p(z_i|\vec{z}_{-i},\vec{w})$，我们必须得到整个文本集生成的联合概率分布$p(\vec{w},\vec{z})$。
$$
p(\vec{w},\vec{z})=p(\vec{w}|\vec{z})p(\vec{z}))
$$
即
$$
p(\vec{w},\vec{z}|\vec{\alpha},\vec{\beta})=p(\vec{w}|\vec{z},\vec{\beta})p(\vec{z}|\vec{\alpha})
$$
我们将两个因子分开处理。

对于第一项因子$p(\vec{w}|\vec{z},\vec{\beta})$来说，先考虑$p(\vec{w}|\vec{z},\phi)$（先不考虑$\beta$的因素，将$\phi$看做常数），则
$$
p(\vec{w}|\vec{z},\phi)=\prod_{i=1}^Wp(w_i|z_i)=\prod_{i=1}^w\phi_{z_i,w_i}
$$
注意$W$是文档集中的所有词的个数（并非词表中的词个数$V$），$\vec{w}$是文档集中的词向量，$\vec{z}$是文档集中与词向量所对应的每个主题值。

我们将这一项分解成两个连乘，一个相对于词表，一个相对于topic。
首先，对于topic确定的情况下，$p(\vec{w}|z_i=k,\phi)=\prod_{t=1}^V \phi_{k,t}^{n_k^t}​$。$n_k^t​$表示的是整个文档集中隶属于topic k的词t的个数。
又因为同一个词可能被不同的topic生成（只是生成的概率不同而已，例如topic "农业"生成词"小麦"的概率可能为0.5，但topic“军事”生成词"小麦"的概率就可能只是0.01），那么我们在添上对于topic的连乘，即$p(\vec{w}|\vec{z},\phi)=\prod_{k=1}^K \prod_{t=1}^V \phi_{k,t}^{n_k^t}​$

所以
$$
p(\vec{w}|\vec{z},\vec{\beta})=\int p(\vec{w}|\vec{z},\phi) p(\phi|\vec{\beta}) d\phi
$$
而 $p(\phi_k|\vec{\beta})=Dir(\phi_k|\vec{\beta})=\frac{1}{B(\alpha)}\prod_{t=1}^V \phi_{k,t}^{\beta_t-1}​$
所以 $p(\phi|\vec{\beta})=\prod_{k=1}^K Dir(\phi_k|\vec{\beta})=\prod_{k=1}^K \frac{1}{B(\alpha)}\prod_{t=1}^V \phi_{k,t}^{\beta_t-1}​$
所以
$$
p(\vec{w}|\vec{z},\vec{\beta})=\int p(\vec{w}|\vec{z},\phi) \cdot p(\phi|\vec{\beta}) d\phi \\ 
= \int \prod_{k=1}^K \prod_{t=1}^V \phi_{k,t}^{n_k^t} \cdot \prod_{k=1}^K \frac{1}{B(\vec{\beta})}\prod_{t=1}^V \phi_{k,t}^{\beta_t-1} d\phi \\ 
= \prod_{k=1}^K \frac{1}{B(\vec{\beta})}  \int \prod_{t=1}^V \phi_{k,t}^{n_k^t+\beta_t-1} d\phi \\ 
= \prod_{k=1}^K \frac{B(\vec{n_k}+\vec{\beta})}{B(\vec{\beta})} 
$$
其中$\vec{n_k}=(n_k^0,n_k^1 \dots n_k^V)$。


对于因子2来说，同理可得：
$$
p(\vec{z}|\theta) = \prod_{i=1}^W p(z_i|d_i)=\prod_{m=1}^M \prod_{k=1}^K p(z_i=k|d_i=m)=\prod_{m=1}^M \prod_{k=1}^K \theta_{m,k}^{n_m^k}
$$
 $n_m^k$表示的是第m个文档中属于第k个主题的词的个数。
同样积分可得：
$$
p(\vec{z}|\vec{\alpha})=\int p(\vec{z}|\theta)p(\theta|\vec{\alpha}) d\theta \\ 
=\prod_{m=1}^M \frac{1}{B(\vec{\alpha})} \int \prod_{k=1}^K \theta_{m,k}^{n_m^k+\alpha_k-1} d\vec{\theta_m} \\
= \prod_{m=1}^M \frac{B(\vec{n_m}+\vec{\alpha})}{B(\vec{\alpha})} 
$$
其中$\vec{n_m}=(n_m^1,n_m^2\dots n_m^K)$

所以，文档集中生成的联合分布为
$$
p(\vec{w},\vec{z}|\vec{\alpha},\vec{\beta})=p(\vec{w}|\vec{z},\vec{\beta})p(\vec{z}|\vec{\alpha}) \\
= \prod_{k=1}^K \frac{B(\vec{n_k}+\vec{\beta})}{B(\vec{\beta})} \cdot \prod_{m=1}^M \frac{B(\vec{n_m}+\vec{\alpha})}{B(\vec{\alpha})}
$$


#### 3.5.2 Collapsed Gibbs Sampling公式

为清晰起见我们更改下联合分布式子的符号表示，将$B(x)$改为$\Delta(x)$。则

$$
p(\vec{w},\vec{z}|\vec{\alpha},\vec{\beta})=p(\vec{w}|\vec{z},\vec{\beta})p(\vec{z}|\vec{\alpha}) \\
= \prod_{k=1}^K \frac{\Delta(\vec{n_k}+\vec{\beta})}{\Delta(\vec{\beta})} \cdot \prod_{m=1}^M \frac{\Delta(\vec{n_m}+\vec{\alpha})}{\Delta(\vec{\alpha})}
$$


其中：

- $\vec{n_k}=(n_k^0,n_k^1 \dots n_k^V)$，$n_k^t$表示的是整个文档集中隶属于topic k的词t的个数。
-  $\vec{n_m}=(n_m^1,n_m^2\dots n_m^K)$，$n_m^k$表示的是第m个文档中属于第k个主题的词的个数。
- $\Delta(\vec{\alpha})=\frac{\prod_{k=1}^K \Gamma(\alpha_k)}{\Gamma(\sum_{k=1}^K \alpha_k)}$

我们所要求的完全条件概率如下：
$$
p(z_x=k|\vec{z_{-x}},\vec{w})=\frac{p(\vec{w},\vec{z})}{p(\vec{w},\vec{z_{-x}})}  
$$
注意，对于这个公式而言，

- 因为当前仅处理文档集中的第$x$个单词$W_x$（若该单词位于第$m$篇文档，且其隐含主题为第$k$个主题，该单词是词表中的第$i$个词），因此与$topic_k$和$doc_m$无关的均被视为常数忽略，因此消去了$\prod$符号。
- 而分母的$\Delta(\vec{\beta}),\Delta(\vec{\alpha})$也由于分子分母都有而消去了。


则因此
$$
p(z_x=k|\vec{z_{-x}},\vec{w})=\frac{p(\vec{w},\vec{z})}{p(\vec{w},\vec{z_{-x}})} \\ 
\propto \frac{\Delta(\vec{n_k}+\vec{\beta})}{\Delta(\vec{n_{k,-i}}+\vec{\beta})} \cdot \frac{\Delta(\vec{n_m}+\vec{\alpha})}{\Delta(\vec{n_{m,-i}}+\vec{\alpha})}
$$

首先我们来看第一个因子的分母部分，
$$
\Delta(\vec{n_{k,-i}}+\vec{\beta})=\frac{\prod_{t=1}^V \Gamma(n_{t,-i}+\beta_t)}{\Gamma(\sum_{t=1}^V n_{t,-i}+\beta_t)} \\ 
= \frac{\Gamma(n_1+\beta_1)\Gamma(n_2+\beta_2) \dots \Gamma(n_i-1+\beta_i) \dots \Gamma(n_V+\beta_V)}{\Gamma(\sum_{t=1}^V n_{t,-i}+\beta_t)} \\ \text{(只有第i个单词需要减1，即减去当前单词的影响)}
$$
上面公式中，V为字典中词的个数。

同理，分子部分的话，
$$
\Delta(\vec{n_k}+\vec{\beta})=\frac{\prod_{t=1}^V \Gamma(n_t+\beta_t)}{\Gamma(\sum_{t=1}^V n_t+\beta_t)} \\ 
= \frac{\Gamma(n_1+\beta_1)\Gamma(n_2+\beta_2) \dots \Gamma(n_i+\beta_i) \dots \Gamma(n_V+\beta_V)}{\Gamma(\sum_{t=1}^V n_t+\beta_t)} 
$$

因此，第一个因子即如下：
$$
\frac{\Delta(\vec{n_k}+\vec{\beta})}{\Delta(\vec{n_{k,-i}}+\vec{\beta})} = \frac{\Gamma(n_i+\beta_i)}{\Gamma(n_i-1+\beta_i)} \cdot \frac{\Gamma(\sum_{t=1}^V n_{t,-i}+\beta_t)}{\Gamma(\sum_{t=1}^V n_t+\beta_t)} \\ 
(基于\Gamma(x+1)=x\Gamma(x)) \\ 
= (n_i-1+\beta_i) \cdot \frac{1}{\sum_{t=1}^V (n_{t,-i}+\beta_t)} = \frac{n_i-1+\beta_i}{\sum_{t=1}^V (n_{t,-i}+\beta_t)} 
$$

同理，第二个因子为：
$$
\frac{\Delta(\vec{n_m}+\vec{\alpha})}{\Delta(\vec{n_{m,-i}}+\vec{\alpha})} = \frac{\Gamma(n_k+\alpha_k)}{\Gamma(n_k-1+\alpha_k)} \cdot \frac{\Gamma(\sum_{t=1}^K n_{t,-i}+\alpha_t)}{\Gamma(\sum_{t=1}^K n_t+\alpha_t)} \\ 
(k是z_i=k的那个k号主题) \\ 
= \frac{n_k-1+\alpha_i}{\sum_{t=1}^K (n_{t,-i}+\alpha_t)} 
$$

所以，最终我们得到了如下结果：
$$
p(z_i=k|\vec{z_{-i}},\vec{w})=\frac{p(\vec{w},\vec{z})}{p(\vec{w},\vec{z_{-i}})} \\
= \frac{n_i-1+\beta_i}{\sum_{t=1}^V (n_{t,-i}+\beta_t)} \cdot \frac{n_k-1+\alpha_i}{\sum_{t=1}^K (n_{t,-i}+\alpha_t)} \\
(通常我们把超参数\vec{\alpha}和\vec{\beta}的每个元素值设为相同值，即对称超参数) \\
= \frac{n_{k,-i}^{(t)}+\beta}{\sum_{t=1}^V n_{k,-i}^{(t)}+V\beta)} \cdot \frac{n_{m,-i}^{(t)}+\alpha}{\sum_{t=1}^K (n_{m,-i}^{(t)}+K\alpha)}
$$

其中，

- $n_{k,-i}^{(t)}$即$n_i-1$，表示第k个topic的第i个单词个数-1
- $n_{m,-i}^{(t)}$即$n_k-1$，表示第m个文档的第k个主题词数-1

#### 3.5.3 计算隐含主题的概率分布

上一节中我们已经的出来采样公式，所以在多次循环采样后，我们就能得到文档集中的每个词的主题赋值。
然后根据2.5节LDA的目的（二）所说，知道了每个词的主题，我们利用Dirichlet分布的期望公式，就可以得到隐含主题的两个概率分布矩阵$\phi$和$\theta$了。
$$
\theta_{mat}=[\vec{\theta_1}, \vec{\theta_2} \dots \vec{\theta_M}] 
$$
则第m篇文章的第k个主题：
$$
\theta_{m,k} = \frac{n_{m,k}+\alpha_k}{\sum_{i=1}^K (n_{m,i}+\alpha_i)} \\ (引入对称超参数) \\ 
= \frac{n_{m,k}+\alpha}{\sum_{i=1}^K n_{m,i}+K\alpha)}
$$

同理，
$$
\phi_{mat}=[\vec{\phi_1}, \vec{\phi_2} \dots \vec{\phi_K}] 
$$
第k个主题的第w个词：
$$
\phi_{m,k} = \frac{n_{k,w}+\beta_w}{\sum_{i=1}^V (n_{k,i}+\beta_i)} \\ (引入对称超参数) \\ 
= \frac{n_{k,w}+\beta}{\sum_{i=1}^V n_{k,i}+V\beta)}
$$


到此为止，LDA算法的所要达到的目的我们已经获取到了，LDA算法的大致流程也已经都写了下来。另外，LDA的计算方法除了Gibbs Sampling之外还有一种方法是变分贝叶斯，大家有兴趣可以去了解一下思想。


## 4 参考资料
- [LDA漫游指南-马晨](http://yuedu.baidu.com/ebook/d0b441a8ccbff121dd36839a)
- Gibbs Sampling for the Uninitiated
- [通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)
- 机器学习-周志华
- [Metropolis-Hasting algorithm-Wikipedia](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- Introduction to Stationary Distribution
- Probabilistic Topic Models



[^1]: 证明详见参考文献：LDA漫游指南。
