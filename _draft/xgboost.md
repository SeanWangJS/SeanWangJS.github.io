---
XGBoost 算法
---

#### 训练集

\[
    \mathcal{D} = \{(x_i, y_i)\},\quad x_i \in R^m, y_i \in R , i=1..n
    \]

#### 模型

\[
    \hat{y}_i = \phi(x_i) = \sum_{k=1}^K f_k(x_i), \quad f_k \in \mathcal{F}
    \]

这里的每个 \(f_k\) 都是决策树函数空间 \(\mathcal{F}\) 上的一个实例。而 \(\mathcal{F}\) 可以表示为 \(\mathcal{F} = \{w_{q(x)}\}\)，其中 \(q(x): R^m \rightarrow \{1..T\}, w \in R^T\)。解释一下：\(q(x)\) 是决策树的结构，它把 \(x\) 传输到树的 \(T\) 个叶节点之一，\(w\) 存储了每个叶节点的值，是一个\(T\) 维向量，通过 \(w_{q(x)}\) 得到样本的预测值，所以 \(w_{q(x)}\) 实际上就是一颗决策树。

#### 正则化目标函数

\[
    L(\phi) = \sum_{i=1}^n l(y_i, \hat{y_i}) + \sum_{k=1}^K \Omega(f_k)
    \]

其中 \(\Omega(f) = \gamma T + \frac 1 2 \lambda \mid\mid w \mid\mid^2\) 表示正则项，模型越复杂，正则项的值越大，因此在损失函数中加入正则项有利于控制模型的复杂度，从而抑制过拟合。

#### 梯度提升树

模型的迭代形式

\[
    \hat{y}_i = \phi_t(x_i) = \phi_{t-1}(x_i) + f_t(x_i)
    \]

第 t 步迭代的目标函数

\[
    L(f_t) = \sum_{i=1}^n l(y_i, \hat{y}_i^{t-1} + f_t(x_i)) + \Omega(f_t)
    \]

这可以看作是对 \(f_t\) 的泛函，对 \(l\) 二阶泰勒展开

\[
    l(y_i, \hat{y}_i^{t-1} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{t-1}) + g_i f_t(x_i) +\frac 1 2 h_i f_t^2(x_i)
    \]

其中\(g_i = \frac { \partial l(y_i, \hat{y}^{t-1}_i)}{\partial \hat{y}^{t-1}_i} \)，\(h_i = \frac { \partial^2 l(y_i, \hat{y}^{t-1}_i)}{\partial (\hat{y}^{t-1}_i)^2} \)。

代入到目标函数后

\[
    \begin{aligned}
    L(f_t) &= \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{t-1}) + g_i f_t(x_i) 
    +\frac 1 2 h_i f_t^2(x_i)\right] + \Omega(f_t)\\
    &=\sum_{i=1}^n l(y_i, \hat{y}_i^{t-1})  + \sum_{i=1}^n \left[  g_i f_t(x_i) 
    +\frac 1 2 h_i f_t^2(x_i)\right] + \Omega(f_t)
    \end{aligned}
    \]

去掉与 \(f_t\) 无关的常量，优化目标不会变

\[
    \arg \min_{f_t} L^{(t)} = \arg\min_{f_t} \tilde{L}^{(t)}
    \]

\[
    \tilde{L}(f_t) = \sum_{i=1}^n \left[  g_i f_t(x_i) 
    +\frac 1 2 h_i f_t^2(x_i)\right] + \Omega(f_t)
    \]

对于一个样本 \(x_i\)，如果它被决策树 \(f_t\) 分配到了第 \(j\) 个叶节点，也就是说 \(q(x_i) = j\)，这时 \(f_t(x_i) = w_j\)，现在对所有样本按叶节点进行分组，定义集合 \(I_j = \{i \mid q(x_i) = j\}\)，那么目标函数又可以写成

\[
    \begin{aligned}
    &\tilde{L}(f_t) = \sum_{j=1}^T \left[\sum_{i \in I_j} g_i f_t(x_i) + \frac 1 2 \sum_{i \in I_j} h_i f_t^2(x_i) \right] + \Omega(f_t) \\
    &\Rightarrow \tilde{L}^{(t)}(w, q)= \sum_{j=1}^T \left[\sum_{i \in I_j} g_i w_j + \frac 1 2 \sum_{i \in I_j} h_i w_j^2 \right] + \gamma T + \frac 1 2 \lambda \sum_{j=1}^T w_j^2 \\
    &\Rightarrow \tilde{L}^{(t)}(w, q)= \sum_{j=1}^T \left[\sum_{i \in I_j} g_i w_j + \frac 1 2 \left(\sum_{i \in I_j}h_i+\lambda\right)   w_j^2 \right] + \gamma T
    \end{aligned}
    \]

上面的转换其实就是把损失对 \(f_t\) 的泛函转换成为 \(w, q\) 的函数，为了获得最优解，对 \(w_k\) 求偏导，并令其等于 0

\[
    \begin{aligned}
    &\frac{\partial \tilde{L}^{(t)}}{\partial w_k} = \sum_{i\in I_k}g_i + \left(\sum_{i\in I_k} h_i + \lambda\right)w_k = 0\\
    &\Rightarrow w_k^*= -\frac{\sum_{i\in I_k} g_i}{\sum_{i\in I_k} h_i + \lambda}
    \end{aligned}
    \]

再将 \(w^*\)代入 \(\tilde{L}^{(t)}(w, q)\) 得到关于 \(q\) 的泛函

\[
    \begin{aligned}
    \tilde{L}^{(t)}(q) &= \sum_{j=1}^T \left[\sum_{i \in I_j} g_i \left(-\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}\right) + \frac 1 2 \left(\sum_{i \in I_j}h_i+\lambda\right)   \left(-\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}\right)^2 \right] + \gamma T\\
    &=-\frac 1 2 \sum_{j=1}^T \left[ \frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j} h_i + \lambda} \right] + \gamma T
    \end{aligned}
    \]

再定义 
\[
    G_j =\sum_{i\in I_j} g_i ,\quad H_j = \sum_{i\in I_j} h_i
    \]

上述损失又可以化简为 
\[
    \tilde{L}^{(t)}(q) = -\frac 1 2 \sum_{j=1}^T \left[ \frac {G_j^2}{H_j+\lambda} \right] + \gamma T
    \]

上式只与 \(q\) 有关，也就是说任意给出一颗决策树，可以利用上式计算本轮迭代的总体损失。通过枚举所有决策树结构，选取令总体损失最小的 \(q\)，便完成了本轮优化。

但实际情况下，枚举所有的决策树结构是不可能的，实际上，决策树的优化有一类比较通用的贪心算法，即通过计算分裂增益来寻找最佳的分裂点，比如 ID3 使用信息增益，C4.5 使用信息增益比，那么在这里，其实就有一个现成的分裂增益评价指标，即 \(\tilde{L}\) 在分裂前后的差值。注意，当前情况下我们的树有 \(T\) 个节点，从中任选一个节点进行分裂，假设编号为 \(k\) ，把 \(I_k\) 分成了两个部分 \(I_L\) 和 \(I_R\)，那么分裂后的损失为

\[
    \tilde{L}^{(t)}(\hat{q}) =  -\frac 1 2 \sum_{j=1,j\ne k}^{T}  \left[ \frac {G_j^2}{H_j+\lambda} \right] - \frac {G_L^2}{H_L + \lambda} -\frac {G_R^2}{H_R + \lambda} + \gamma (T+1)
    \]

其中
\[
    G_L = \sum_{i \in I_L} g_i,\quad H_L = \sum_{i \in I_L}h_i,\quad G_R = \sum_{i \in I_R} g_i,\quad H_R = \sum_{i -\in I_R}h_i
    \]

于是，分裂前后的总体损失增益就为 

\[
    \tilde{L}_{split} =\tilde{L}^{(t)}(q) - \tilde{L}^{(t)}(\hat{q})= \frac {G_L^2}{H_L + \lambda} +\frac {G_R^2}{H_R + \lambda} -\frac{G_k^2}{H_k+\lambda}- \gamma 
    \]

#### 精确贪心算法

精确的贪心算法严格按照贪心策略，在特征和值两个维度上搜索。算法过程如下

1. 首先设分裂前的节点上的样本集合为 \(I\)，样本特征维度等于 \(m\)；
2. 初始化：损失增益变量 \(gain = 0\)，最佳分裂特征 \(d = 1\)，最佳分裂点 \(z = \min(\mathbf{x}_d)\)，这里的 \(\mathbf{x}_d\) 是所有样本第 \(d\) 个特征组成的向量；
3. 计算 
\[
    G = \sum_{i \in I} g_i,\quad H = \sum_{i \in I}h_i
    \]
4. 对于 \(k = 1...m\)：
a. 假设在第 \(k\) 个特征上分裂，为了确定在哪个值上分裂，先对第 \(k\) 个特征上的所有值 \(\{x_{jk}\}_{j=1}^n\) 进行排序，得到排序后的向量 \(\{\bar{x}_{jk}\}_{j=1}^n\)；
b. 假设分裂后的两部分样本分别为 \(I_L\) 和 \(I_R\)，初始化 \(G_L = 0, H_L = 0\)，对于 \(j = 1... n\)：
    (1). 计算 
    \[
        G_L \leftarrow G_L + g_j, H_L \leftarrow H_L + h_j
    \]

    再根据 \(G\) 的定义，得到 
    \[
        G_R \leftarrow G - G_L, H_R \leftarrow H - H_L
        \]
    
    计算损失增益
    \[
        \Delta L =  \frac {G_L^2}{H_L + \lambda} +\frac {G_R^2}{H_R + \lambda} -\frac{G^2}{H+\lambda}- \gamma 
        \]
    
    当 \(\Delta L > gain\) 时，更新 \(gain = \Delta L, d = k, z = \bar{x}_{jk}\)；
5. 使用第 \(d\) 个特征对样本进行分裂，分裂点为 \(z\)。

#### 近似贪心算法