---
Adaboost 算法
---

#### 训练集

\(T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}\)，其中 \(x_i \in \mathcal{X} \subset \mathbf{R}^n\)，\(y_i \in \mathcal{Y} \subset \{-1, 1\}\)。

#### 算法过程

1. 初始化训练集数据权重分布

\[
  D_1 = (w_{11}, w_{12}, ..., w_{1N})，w_{1i} = \frac 1 N ,i = 1...N
  \]

2. 对于 m = 1 ... M
  a. 利用训练集 \(T\) 和权重分布 \(D_m\) ，训练得到弱学习器 \(G(x): \mathcal{X} \rightarrow \{-1, 1\}\)
  b. 计算 \(G_m(x)\) 在训练数据上的误差率
\[
  e_m = \sum_{i = 1}^N w_{mi} I(G(x_i) \ne y_i )
    \]
  c. 计算 \(G_m(x)\) 的权重
  \[
    \alpha_m = \frac 1 2 \ln {\frac {1-e_m} {e_m}}
    \]
  d. 更新训练集数据权重分布 \(D_{m+1}\)，
  \[
    w_{m+1,i} = \frac {w_{mi}} {Z_m} \exp(-\alpha_m y_i G_m(x_i)), i = 1...N
    \]
    其中规范化因子 \(Z_m = \sum_{i}^N w_{mi} \exp(-\alpha_m y_i G_m(x_i))\)
  >注： 权重更新可以用下面的逻辑来解释：当 \(G_m(x_i) = y_i\) 时，即对于正确分类的样本，需要降低它的权重，而此时 \(\exp(-\alpha_m y_i G_m(x_i))\) 小于 1，用它乘以 \(w_{mi}\) 得到的值显然小于 \(w_{mi}\)，假设它是 \(\bar{w}_{m+1, i}\)，于是这就达到了降低权重的效果。反过来当样本被错误分类时，需要增加权重，而此时\(\exp(-\alpha_m y_i G_m(x_i))\) 大于 1，于是乘以它又产生了增加权重的效果。但是更新后的权重之和 \(\sum \bar{w}_{m+1, i}\) 不一定等于 1，为了使权重分布是一个概率分布，需要乘以一个系数使得所有权重之和等于 1，于是需要让每个更新后的权重除以 \(\sum \bar{w}_{m+1, i}\)，也就是 \(Z_m\) ，所以 \(Z_m\) 又被称作规范化因子。

  3. 弱学习器线性组合
  \[
    f(x) = \sum_{m=1}^M \alpha_m G_m(x)
    \]
    利用符号函数得到最终分类器 
    \[
      G(x) = sign(f(x))
      \]
  
  