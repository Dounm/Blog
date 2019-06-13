# LDA算法理解（一）：数学基础

---

系列文章分为两部分：

1. LDA算法理解（一）：数学基础
2. LDA算法理解（二）：Gibbs Sampling

---

[TOC]

## 1 LDA（Latent Dirichlet Allocation）概述
隐含狄利克雷分布（Latent Dirichlet Allocation, LDA）算法由David Blei, Andrew Ng, Jordan Michaell于2003年提出，是一种主题模型。
作用是：**将文档集中每篇文档的主题以概率分布的形式给出**。
在获得了每篇文档上的隐含主题的概率分布后，我们就可以根据主题分布对文本进行处理（例如主题聚类和文本分类）。
其中一篇文档可以包含多个主题，文档中的每个词都由其中的某个主题生成。

### 1.1 基本术语解释
|名词|含义|
|----|----|
|文档Document|非传统意义上的文档。<br />LDA是词袋（bag of words）模型。将文档看作是是一组词，词与词之间没有先后顺序。|
|文档集|训练集中所有文档的集合（设共有$M$个文档）|
|词word|英文中即一个单词，中文中即独立意义的中文词|
|词表|文档集中所有出现过的不重复的词的集合（设共有$V$个词）。|
|**主题Topic**|一个主题表现为该主题相关的一些词的集合，我们可以用$V$维的词表向量来表示主题$k$。<br />向量的第$i$个元素表示的就是word $i$在该主题$k$下的概率，不出现在该主题$k$的词值就为0。<br />向量的所有元素取值之和为1。|
|**主题分布**|对于一篇文档而言，LDA认为其包含了多个主题（设共有$K$个主题）。<br />举例而言，对于某文档而言，主题1占20%的可能，主题2占50%的可能等等。<br />主题分布即K个主题在文档上的分布|
### 1.2 LDA的目的（一）
LDA的使用情景是：
**对于一个文档集，我要在文档集中提取出$K$个主题来（$K$大小可以随意指定），然后我想知道文档集中每篇文档在相对于这$K$个主题的主题分布，从而根据这些主题分布来对文档集中的文档进行一些处理（求解文本之间的相似度，对文本自动打标签）。**

为了完成这个目标：
1. 我们首先得确定这$K$个主题都是啥（即这$K$个主题对应的$V$维词表向量分布）
2. 然后得确定这$K$个主题在每个文档上的主体分布（即$M$个文档对应的$K$维主题分布）

所以**LDA的程序最终需要求解出的结果即为两个矩阵**： 

|矩阵符号|维度|含义|
|----|----|----|
| $\theta$ |  $M*K$ | 代表的$M$篇文档，每篇文档上的主题分布。$\theta_i$代表的是第$i$篇文档上的主题向量。 |
| $\phi$ | $K*V$| 代表$K$个主题，每个主题上的词频。$\phi_k$代表的是第$k$个主题上的词表向量 |

## 2 数学基础
LDA从生成模型的角度来看待文档和话题。

举例而言，通常人类写文章的步骤如下：

1. 选择一些与该文章相关的主题，例如军事和农业
2. 然后基于选定的主题遣词造句等等。

在LDA模型中，一篇文档生成的方式如下：

1. 从狄利克雷分布$\alpha$中取样生成文档$i$的**主题分布$\theta_i$**
2. 从狄利克雷分布$\beta$中取样生成主题$k$的**词表分布$\phi_k$**
3. 对于文档i中的每个单词位置来说，执行下列操作：
4. 从主题的多项式分布$\theta_i$ 中取样生成文档$i$的第$j$个词的主题$z_{i,j}$
5. 从词语的多项式分布$\phi_{z_i,j}$采样生成最终词语$w_{i,j}$

### 2.1 多项式分布Multinomial Distribuition
注意上面所描写的文档的生成方式中，暗示了无论是**主题分布**还是**词表分布** 其形式都是**多项式分布**。
多项式分布定义如下：
> 设$A_1,A_2,\dots,A_n$为某一试验的完备事件群，即事件$A_1,A_2,\dots,A_n$两两互斥，其和为完备事件群。
其中$A_1,A_2,\dots,A_n$的概率分别是$p_1,p_2,\dots,p_n$。
将该事件独立地重复N次，以$X_i$记为这N次试验中事件$A_i$出现的次数，则$X=(X_1,X_2,...,X_n)$是一个$n$维随机向量（$X_i$的取值范围为都是非负整数，且和为$N$）。
 多维随机变量$X$的概率分布即为多项分布：
$$
 P(x_1,x_2,...,x_k;n,p_1,p_2,...,p_k)=\frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}
$$

我们来就多项式分布的定义分析下主体分布和词表分布：

|多项式分布定义|主题分布$\theta_i$|词表分布$\phi_k$|
|---|---|---|
|基本事件|对于当前文档$i$的某个空白位置选择主题|对于当前文档的某个空白位置，在该空白位置主题已定的情况下，选择该空白位置填充词|
|基本事件的执行次数$N$|当前文档空白位置的个数|当前文档被分配给主题k的空白位置的个数|
|完备事件群$A_1,A_2,...,A_n$|$K$个主题构成完备事件群|$V$个词构成完备事件群|
|$p_1,p_2,\dots,p_n$|$\theta_{i1},\theta_{i2},...,\theta_{iK}$|$\phi_{i1},\phi_{i2},...,\phi_{iV}$|
|多维随机变量$X=(X_1,X_2,...,X_n)$|$X=(Topic_1, Topic_2 \dots Topic_K)$。$X_k$：当前文档$i$的空白位置中，分配给$Topic_k$的空白位置的个数|$X=(Word_1, Word_2 \dots Word_V)$。$X_j$：当前文档$i$的空白位置中，分配给$Topic_k$的空白位置中，$Word_j$的个数|
|$X \sim Multi(N;p_1,p_2,...,p_n)$|$X \sim Multi(L;\theta_{i1},\theta_{i2}\dots\theta_{iK}) $（设当前文档有$L$个空白位置）|$X \sim Multi(L_k;\phi_{i1},\phi_{i2}\dots\phi_{iV}) $（设当前文档被分配给主题$k$的空白位置的个数为$L_k$）|

注意：对于LDA而言，*我们最终所要求的两个矩阵，就是由主题分布和词表分布两个多项式分布的参数的参数构成*。
所以，我们要求的就是**多项式分布的参数**。

### 2.2 Gamma函数
普通的阶乘仅适用于正整数域，而Gamma函数即为阶乘的一般形式，将阶乘拓展到正实数域。
Gamma函数形式：$\Gamma(x)=\int_0^{+\infty}e^{-t}t^{x-1}dt(x>0)$
Gamma函数具有如下性质：

- $\Gamma(x+1)=x\Gamma(x)$
- $\Gamma(n)=(n-1)!$

Gamma函数是定义在实数域上的阶乘运算，将阶乘这个操作从离散扩展到了连续。
**任何用到离散阶乘的地方，都可以借助Gamma函数将概念从离散扩展到连续。**

### 2.3 二项分布与Beta分布
#### 2.3.1 二项分布
二项分布即是重复了$n$次的伯努利分布，其概率密度函数为$P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}$
注意：二项式分布是**离散概率分布**，并且其中也出现了阶乘。

#### 2.3.2 Beta分布
Beta分布$X \sim Beta(\alpha, \beta)$，指的是定义在区间$(0,1)$上的**连续概率分布**，他有两个参数$\alpha$和$\beta$。
其概率密度函数如下所示：
$$
f(x;\alpha,\beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{\int_0^1u^{\alpha-1}(1-u)^{\beta-1}du}
$$
对于上式而言，我们观察可以得出，**分母部分是一个归一化参数**。因为对于连续概率分布而言，其概率密度函数必然要保证在定义域内的基本为1（否则不能称之为概率密度函数）。

因此，我们可以假设**Beta分布其实是我们为了某个目的认为构造出来的概率分布（仅为了帮助理解）**：
我们先构造了Beta分布概率密度函数的一部分，即分子$x^{\alpha-1}(1-x)^{\beta-1}$。然后为了使得概率密度函数积分为1，给概率密度函数添加了一个分母$B(a,b)=\int_0^1\mu^{\alpha-1}(1-\mu)^{\beta-1}d\mu​$。

上面公式通常会写作以下形式，即将分母部分利用$B(\alpha,\beta)$来显示：
$$
f(x;\alpha,\beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{\int_0^1u^{\alpha-1}(1-u)^{\beta-1}du} \\
=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$
注意，我们可以证明$B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$[^1]。

对比一下二项式分布和Beta分布，我们可以发现他们的概率密度函数在剔除掉系数之后很相似。而且二项式分布的系数是$\binom{n}{k}$，带有阶乘；Beta分布的系数是$\frac{1}{B(\alpha,\beta)}=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$，带有Gamma函数。
因此，这其实就相当于**二项式分布借由Gamma函数从离散扩充到了连续**。
#### 2.3.3 Beta分布的期望
如果$p \sim Beta(t|\alpha,\beta)$，则
$$
E(p)=\int_0^1t \cdot Beta(t|\alpha, \beta)dt \\  
=\int_0^1t\cdot \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}t^{\alpha-1}(1-t)^{\beta-1}dt \\
= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\int_0^1 t^{\alpha}(1-t)^{\beta-1}dt
$$
对于式子$\int_0^1 t^{\alpha}(1-t)^{\beta-1}dt$而言，我们联想到$Beta(t|\alpha+1,\beta)$的概率密度函数为
$$
f(x;\alpha,\beta)=\frac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)}t^\alpha(1-t)^{\beta-1}
$$
，则因为概率密度函数积分为1，所以
$$
\int_0^1\frac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)}t^\alpha(1-t)^{\beta-1}dt=1
$$
将上式带回到$E(p)$中，并根据$Gamma$函数的性质可得
$$
E(p)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\cdot\frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)}=\frac{\alpha}{\alpha+\beta}
$$
因此，Beta分布的均值就可以用$\frac{\alpha}{\alpha+\beta}$来估计。

#### 2.3.4 多项式分布与狄利克雷分布
对比一下上述所提到的四个分布，如下表所示

|分布名称|概率密度函数|参数|
|---|---|---|
|二项分布|$P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}$|参数为$p$和$1-p$，参数约束为：$p+(1-p)=1$|
|多项分布| $P(x_1,x_2,...,x_k;n,p_1,p_2,...,p_k)=\frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}$ |参数为$p_1,p_2\dots p_n$，参数约束为：$p_1+p_2+\dots+p_n=1$|
|$Beta$分布| $f(x;\alpha,\beta)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}$，其中$\frac{1}{B(\alpha,\beta)}=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$|参数为$\alpha$和$\beta$|
|$Dirichlet$分布|$f(x_1,x_2\cdots x_k;\alpha_1,\alpha_2\cdots \alpha_k)=\frac{1}{B(\alpha)}\prod_{i=1}^kx_i^{\alpha^i-1}$其中，$B(\alpha)=\frac{\prod_{i=1}^k\Gamma(\alpha^i)}{\Gamma(\sum_{i=1}^k\alpha^i)}$|参数为$\alpha_1,\alpha_2\dots \alpha_k$|

由上述表格内容可以看出：

- 多项分布是二项分布在多维变量上的推广 
- $Dirichlet$分布式$Beta$分布在多维变量上的推广　

#### 2.3.5 Dirichlet分布的期望
参见Beta分布的期望，因为$Dirichlet$分布是Beta分布在多维变量的推广，所以我们可以得出如下结论（证明参考Beta分布的期望证明，略）：
$$
E(\vec{p})=(\frac{\alpha_1}{\sum_{i=1}^K\alpha_i},\frac{\alpha_2}{\sum_{i=1}^K\alpha_i} \dots \frac{\alpha_K}{\sum_{i=1}^K\alpha_i})
$$

### 2.4 共轭先验分布
#### 2.4.1 贝叶斯定理
$$
p(\theta|x)=\frac{p(x|\theta)p(\theta)}{p(x)}=\frac{p(x|\theta)p(\theta)}{\int p(x|\theta)p(\theta)d\theta}\propto p(x|\theta)p(\theta)
$$
这个公式中：$\theta$表示参数，$x$是已观测到的数据。

- $p(\theta|x)$：**后验概率**，在已经观测到了数据$x$的情况下，参数为$\theta$的概率。
- $p(\theta)$：**先验概率**，在没有观测到数据x的情况下，参数为$\theta$的概率。
- $p(x|\theta)$：**似然函数**，参数为$\theta$的情况下，产生观测数据为$x$的概率。
- $P(x)$：**归一化常数**，通常不会直接求，而是忽略掉，最后利用归一化（概率密度函数积分必须为1）来处理，例如上式中所示。

由贝叶斯定理可知，**后验分布$\propto$似然函数*先验分布**。

而共轭先验分布的定义如下：
**如果先验分布和似然函数可以使得先验分布和后验分布具有相同的形式，则称先验分布是似然函数的共轭先验分布**

### 2.4.2 Beta分布是二项式分布的共轭先验分布
因为多项式分布和$Dirichlet$分布式二项式分布和$Beta$分布的多维推广，所以我们在此只证明$Beta$分布式二项分布的共轭先验分布。

要证明$Beta$分布式二项分布的共轭先验分布，则根据共轭先验分布的定义，似然函数是二项分布的形式，先验分布是$Beta$分布的形式，我们想要让后验分布也是$Beta$分布的形式。
证明：
1. 似然函数$p(x|\theta)$是二项分布，$\theta$即为二项分布的参数$p$，似然函数即为
$$
L=\binom{s+f}{s}p^s(1-p)^f
$$
其中$s$表示$n$次试验中成功的次数，$f$表示$n$次试验中失败的次数，$n=s+f$。
2. 先验分布$p(\theta)$是$Beta$分布，即$\theta$是$beta$分布以$\alpha$和$\beta$为参数的结果，先验分布即为
$$
P(p|\alpha,\beta)=\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}
$$
其中$B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$
3. 根据**后验分布$\propto$似然函数*先验分布**可得：
$$
P(p|s,f,\alpha,\beta)=\frac{\binom{s+f}{s}p^s(1-p)^f\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}}{\int_{q=0}^1\binom{s+f}{s}p^s(1-p)^f\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}dq} \\ = \frac{p^{s+\alpha-1}(1-p)^{f+\beta-1}}{\int_{q=0}^1p^{s+\alpha-1}(1-p)^{f+\beta-1}dq} \\ = \frac{p^{s+\alpha-1}(1-p)^{f+\beta-1}}{B(s+\alpha,f+\beta)}
$$

因此，我们会发现后验分布也是Beta分布。
而且**先验分布是$X\sim Beta(\alpha,\beta)$，后验分布则是$X\sim Beta(\alpha+s, \beta+f)$**。
超参数$\alpha$和$\beta$在基于观测到的数据$s$和$f$后发生了改变，变成了$\alpha+s$和$\beta+f$，但形式上仍然是$Beta$分布。
如果以后再有新的数据的话，我们仍然可以在了$\alpha+s$和$\beta+f$的基础上继续更新超参数。

#### 2.4.3 Dirichlet分布是多项式分布的共轭先验分布
**先验分布是$Dir(\vec{p}|\vec{a})$，后验分布就变成了$Dir(\vec{p}|\vec{\alpha}+\vec{x})$**。
注意：

- $\vec{p}$：多项式分布的参数，同时也是$Dirichlet$分布的结果/随机变量。
- $\vec{\alpha}$：$Dirichlet$分布的参数。
- $\vec{x}$：观测到的数据。第$i$维代表事件$i$发生的次数（如果将$Beta$分布看做是二维的$Dirichlet$分布的话，那么$x_1=s,x_2=f$）。
$\vec{p},\vec{\alpha},\vec{x}$这三个向量维度相同。

### 2.5 LDA的目的（二）
我们在*LDA的目的（一）*中提到，LDA目的最终要求出的是两个矩$\theta$和$\phi$，这两个矩阵又都是多项式分布的参数。而根据共轭先验分布，多项式分布的参数即为$Dirichlet$分布的结果/随机变量，因此我们可以**用$Dirichlet$分布随机变量的期望来估计多项式分布的参数**。
结合上面所提到的，$Dirichlet$分布的期望公式如下：
$$
E(\vec{p})=(\frac{\alpha_1}{\sum_{i=1}^K\alpha_i},\frac{\alpha_2}{\sum_{i=1}^K\alpha_i} \dots \frac{\alpha_K}{\sum_{i=1}^K\alpha_i})
$$
根据共轭先验分布的结论，$Dirichlet$分布的参数有最开始的$\vec{\alpha}$变成了$\vec{\alpha}+\vec{x}$，其中$\vec{\alpha}$是我们预先设定好的参数，$\vec{x}$则是训练数据中隶属于每个主题词的个数（如果将$Beta$分布看做是二维的$Dirichlet$分布的话，$\vec{x}$为二维，值分别是$x_1=s,x_2=f$）。因此我们只需要识别出对于文档中的每个词来说，该词属于哪个隐含主题，然后就可以按照上述公式来计算出来两个矩阵。

**因此，我们所要求解的就是$p(\vec{z}|\vec{w})$，即文档集中的每个词背后所隐含的主题。**

注意：$\vec{w}$是文档集中的词向量，$\vec{z}$是文档集中与词向量所对应的每个主题值。
举例而言，如果对于文档集，只有一个文档，该文档分词后有5个词，"aaa bbb ccc ddd aaa"。
然后我们要从文档集中提取出来3个主题$topic0,topic1,topic2$。
词"aaa"被赋予的隐含主题为$topic0$，词"bbb"被赋予的隐含主题为$topic2$，词"ccc"被赋予的隐含主题维$topic0$，词"ddd"被赋予的隐含主题为$topic1$。
则$\vec{w}=(aaa,bbb,ccc,ddd,aaa), \vec{z}=(topic0，topic2,topic0,topic1,topic0)$。
$\vec{w}$和$\vec{z}$的维度都是整个文档集中词的个数（重复词不合并）。

根据条件概率公式
$$
p(\vec{z}|\vec{w})=\frac{p(\vec{w},\vec{z})}{p(\vec{w})}
$$
但是对于该公式而言，对其分母利用离散概率分布求解边缘概率的方法进行展开。
$$
p(\vec{w})=\sum_zp(\vec{w},\vec{z})=\prod_{i=1}^n\sum_{k=1}^Kp(w_i|z_i=k)p(z_i=k)
$$
其中$n$是文档集中所有词的个数，也即为$\vec{w}$和$\vec{z}$的维度（对于前面所举出的文档集的例子，$n=5$），$K$是索要提取出的隐含主题的个数（前面例子中$K=3$）。
因此对于分母而言，其计算量高达$K^n$，难以计算。
所以我们采用**Gibbs Sampling**的方法来计算。