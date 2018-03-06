---
layout: post
title: 模式识别中的最小风险分类决策
tags: 模式识别系列
description: 从疾病诊断的例子出发，介绍了决策风险的概念，详细推导了最小风险决策的平均损失公式，并说明了何种决策能使得风险最小。
---

[前面这篇文章](/2018/02/15/minimum-misclassification-rate)探讨的是使得分类错误率最小的分类原则。但当我们根据已知信息做分类的时候，不同的决策结果通常伴随着不同的风险，比如说在对人做病情诊断时，患病与不患病对应两种状态，我们假设分别为 $$\mathcal{C}_1$$ 和 $$\mathcal{C}_2$$ 。在诊断过程中收集到的病情特征用 $$x$$ 表示，并且特征空间被分为两个部分 $$R_1$$ 和 $$R_2$$ ，当 $$x\in R_1$$ 时，诊断为 $$\mathcal{C}_1$$ ，即患病，否则诊断为 $$\mathcal{C}_2$$ 未患病。

那么病情特征和诊断结果就有四种组合方式，这里用四个元组来表示，即

$$(x\in R_1, \mathcal{C}_2),\quad (x\in R_2, \mathcal{C}_1), \quad (x\in R_1, \mathcal{C}_1),\quad (x\in R_2, \mathcal{C}_2)$$

不难发现，前两种为误诊，但它们造成的后果是不一样的，如果未患病者给诊成了患病，人最多心里上的压力大点，但如果没有查出来潜在疾病，耽误了治疗实际，就会造成更大的损失。为了量化这种损失，我们定义，把属于 $$R_i$$ 的特征分类到 $$\mathcal{C}_j$$ 的损失为 $$\lambda_{ij}$$ ，那么对于任意特征，作出决策 $$\mathcal{C}_i$$ 的平均损失就为

$$
\begin{aligned}
Loss(\mathcal{C}_i) &= \lambda_{1i}p(x\in R_1,\mathcal{C}_i) + \lambda_{2i} p(x\in R_2, \mathcal{C}_i)\\
&=\int_{x\in R_1} \lambda_{1i} p(x, \mathcal{C}_i) \mathbf{d}x+ \int_{x\in R_2} \lambda_{2i} p(x, \mathcal{C}_i)\mathbf{d}x\\
&=\int_{R_1}f_1(x)\mathbf{d}x + \int_{R_2}f_2(x)\mathbf{d}x
\end{aligned}
$$

如果在 $$x\in R_2$$ 上定义 $$f_1(x) = 0$$ ，以及在 $$x\in R_1 $$ 上定义 $$f_2(x) = 0$$ 。那么就有

$$
\begin{aligned}
Loss(\mathcal{C}_i) &=\int_{R_1}f_1(x)\mathbf{d}x + \int_{R_2}f_2(x)\mathbf{d}x\\
&=\int_{R_1 + R_2} f_1(x) + f_2(x)\mathbf{d}x\\
&= \int_{R_1+R_2} \sum_{j=1}^2 \lambda_{ji} p(x, \mathcal{C}_i) \mathbf{d}x \\
&= \int_{R_1+R_2} \sum_{j=1}^2 \lambda_{ji} p(\mathcal{C}_i\mid x)p(x) \mathbf{d}x
\end{aligned}
$$

更一般地，对于多类问题，平均损失为

$$
Loss(\mathcal{C}_i) =  \int_R \sum_{j=1}^n \lambda_{ji} p(\mathcal{C}_i\mid x)p(x) \mathbf{d}x
$$

其中 $$R$$ 为整个特征空间。在上述积分项中，若令

$$
Loss(\mathcal{C}_i, x) = \sum_{j=1}^n \lambda_{ji} p(\mathcal{C}_i\mid x)p(x)
$$

则有

$$
\begin{aligned}
Loss(\mathcal{C}_i) &= \int_R Loss(\mathcal{C}_i , x) \mathbf{d}x\\
&=\sum_{j=1}^n\int_{x\in R_j} Loss(C_i,  x)\mathcal{d}x
\end{aligned}
$$

显然，为了使平均损失最小，需要先计算 $$Loss(\mathcal{C}_i, x), i=1,2,,,n$$ 的每个值，然后找到其中的最小值 $$Loss(\mathcal{C}_m)$$ 所在的类别，并将 $$x$$ 分类到 $$\mathcal{C}_m$$ （原理类似我们在[最小错误率决策规则](/2018/02/15/minimum-misclassification-rate)中的处理方法）。
