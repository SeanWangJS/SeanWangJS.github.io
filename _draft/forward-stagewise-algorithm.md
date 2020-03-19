---
前向分步算法
---

#### 训练集数据

\(T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}\)

#### 损失函数

\(L(y, f(x))\)

#### 基学习器 

\(b(x, \gamma)\)

#### 算法过程

1. 初始化 \(f_0(x) = 0\)
2. 对于 m = 1 ... M
 a. 假设第 m-1 步的模型函数为 \(f_{m-1}(x)\)，定义第 m 步的模型
 \[
   f_m(x) = f_{m-1}(x) + \beta b(x, \gamma)
   \]
  这里的 \(\beta, \gamma\) 分别是基学习器的权重系数和参数，它们现在都是未知量，需要通过极小化在训练集上的损失函数来确定，定义总体损失
  \[
    l = \sum_{i=1}^N L(y_i, f_m(x_i))
    \]
  于是需要优化的问题形式为
 \[(\beta_m, \gamma_m) = \arg\min_{\beta, \gamma}l\]
 b. 更新模型
 \[
   f_m(x) = f_{m-1}(x) + \beta_m b(x, \gamma_m)
   \]

3. 最终的模型
\[
  f(x) = f_M(x) = \sum_{m = 1}^M\beta_m b(x, \gamma_m)
  \]

#### 与 Adaboost 的联系

当前向分步算法采用指数损失函数时，即
\[
  L(y, f(x)) = \exp(- y f(x))
  \]
算法过程中的第 2 步里面，模型在训练集上的损失函数为
\[
  l=\sum_{i=1}^N \exp\left[-y_i (f_{m-1}(x_i) + \beta b(x, \gamma))\right]
  \]
  
如果令 \(G(x) = b(x, \gamma)\)，则
\[
  \begin{aligned} l &=\sum_{i=1}^N \exp\left[-y_i (f_{m-1}(x) + \beta G(x))\right] \\ 
  &= \sum_{i=1}^N \exp(-y_i f_{m-1}(x) )  \exp(-y_i \beta G(x))
  \end{aligned}
  \]

同时，优化问题也变为
\[
  (\beta_m, G_m(x)) = \arg\min_{\beta, G(x)} l
  \]
令 \(\bar{w}_{mi} = \exp(-y_i f_{m-1}(x) )\)，则有
\[
  l = \sum_{i=1}^N \bar{w}_{mi} \exp(-y_i \beta G(x_i))
  \]

如果把 \(G(x_i) = y_i\) 和 \(G(x_i) \ne y_i\) 的情况分开求和，则
\[
  \begin{aligned}
  l &=\sum_{G(x_i) = y_i}\bar{w}_{mi} \exp(-\beta) + \sum_{G(x_i) \ne y_i} \bar{w}_{mi} \exp(\beta)\\
  &=e^{-\beta} \sum_{G(x_i) = y_i}\bar{w}_{mi} + e^{\beta}\sum_{G(x_i) \ne y_i}\bar{w}_{mi} 
  \end{aligned}
  \]

然后再利用指示函数对上式进行变换

\[
  \begin{aligned}
  l &=e^{-\beta} \sum_{i=1}^N \bar{w}_{mi}\mathbb{I}(G(x_i)=y_i) + e^{\beta} \sum_{i=1}^N \bar{w}_{mi} \mathbb{I} (G(x_i) \ne y_i)\\
  &=e^{-\beta} \sum_{i=1}^N \bar{w}_{mi}(1-\mathbb{I}(G(x_i)\ne y_i)) + e^{\beta} \sum_{i=1}^N \bar{w}_{mi} \mathbb{I} (G(x_i) \ne y_i)\\
  &=e^{-\beta} \sum_{i=1}^N \bar{w}_{mi}- e^{-\beta} \sum_{i=1}^N \bar{w}_{mi}\mathbb{I}(G(x_i)\ne y_i) + e^{\beta} \sum_{i=1}^N \bar{w}_{mi} \mathbb{I} (G(x_i) \ne y_i)
  \end{aligned}
  \]

  令 \(\epsilon_m = \sum_{i=1}^N \bar{w}_{mi} \mathbb{I} (G(x_i) \ne y_i)\)，并考虑到 \(\sum_{i=1}^N \bar{w}_{mi}=1\)，上式可简化为
\[
    l = e^{-\beta} (1-\epsilon_m) + e^{\beta} \epsilon_m
    \]

为了得到 \(\beta\) 的最优值，令损失函数对 \(\beta\) 求导
\[
  \frac{\partial l}{\partial \beta} = - e^{-\beta}(1-\epsilon_m) + e^{\beta}\epsilon_m
  \]

再令导数值等于 0

\[
  \begin{aligned}
    &\frac{\partial l} {\partial \beta} \mid_{\beta=\beta_m}= 0\\
    \Rightarrow &- e^{-\beta_m}(1-\epsilon_m) + e^{\beta_m}\epsilon_m = 0\\
    \Rightarrow & e^{2\beta_m}  =  \frac{1-\epsilon_m} {\epsilon_m}\\
    \Rightarrow &\beta_m = \frac 1 2 \ln \frac{1-\epsilon_m} {\epsilon_m}
  \end{aligned}
  \]

观察 [Adaboost](Adaboost.md) 的算法过程，可以发现，\(\epsilon_m\) 的定义，其实就是 Adaboost 第 m 步训练的弱分类器 \(G_m(x)\) 在训练集上的误差率，它们之间只相差一个规范化因子，而 \(\beta_m\) 的形式与 Adaboost 第 m 个弱分类器的权重系数 \(\alpha_m\) 一样。于是，通过以上推导，可以把 Adaboost 看作是前向分步算法采用指数损失函数时的特例。




  



