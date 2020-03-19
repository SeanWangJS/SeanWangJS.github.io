---
title: 梯度提升树算法
---

#### 训练集数据

\(S = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}\)

#### 损失函数

\(L(y, f(x))\)

#### 基学习器

决策树 \(T(x, \Theta) = \sum_{j=1}^J \gamma_j \mathbb{I}(x_i \in R_j)\)，其中 \(\Theta = \{R_j, \gamma_j\}_{j=1}^J\)

#### 模型

\[
    \hat{y} = f(x) = \sum_{m=1}^M T(x, \Theta_m)
    \]

#### 目标函数 

\[
    obj = \sum_{i=1}^N L(y_i, \hat{y}_i)
    \]

目标函数的迭代形式

\[
    obj_{m} = \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + T(x, \Theta_m)) 
    \]

#### 负梯度残差近似

对目标函数泰勒展开，取到二阶导数

\[
    \begin{aligned}
    obj_{m} &= \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + T(x_i, \Theta_m)) \\
    &= \sum_{i=1}^N \left[ L(y_i, f_{m-1}(x_i)) + \frac{\partial L}{\partial f_{m-1}(x_i)} T(x_i, \Theta_m) + \frac 1 2 \frac{\partial^2 L}{\partial f_{m-1}^2(x_i)} T^2(x_i, \Theta_m)\right]
    \end{aligned}
    \]

由于我们的目标是寻找最优的 \(T(x, \Theta_m)\) 使得 \(obj_m\) 取到最小值，于是对 \(T(x, \Theta)\) 求导，并令导数等于零

\[
    \begin{aligned}
    &\frac{\partial obj_m}{\partial T} = 0 \\
    &\Rightarrow \sum_{i=1}^N \left(\frac{\partial L}{\partial f_{m-1}(x_i)} + \frac{\partial^2 L}{\partial f_{m-1}^2(x_i)} T(x_i, \Theta_m)\right) = 0 \\
    &\Rightarrow \frac {\partial L}{\partial f_{m-1}} + \frac{\partial^2 L}{\partial f^2_{m-1}} T(x, \Theta_m) = 0
    \end{aligned}
    \]

令 \(g = \nabla_{f_{m-1}}L, h = \nabla^2_{f_{m-1}}L \)，则有 
\[
    h T(x, \Theta_m) = -g
    \]

也就是说当前要训练的弱学习器 \(T(x, \Theta_m)\) 与目标函数 \(obj_m\) 对当前模型 \(f_{m-1}(x)\) 的负梯度之间只相差一个常系数 \(h\)。

