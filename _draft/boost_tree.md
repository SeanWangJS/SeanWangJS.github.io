---
title: 提升树算法
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

如果存在 \(T^*(x, \Theta_m)\) 对于所有 \(x_i\) 都有 \(y_i = f_{m-1}(x_i) + T^*(x, \Theta_m)\)，则目标函数可以取到最小值 0。于是第 m 步迭代的优化方法便是寻找决策树 \(T(x, \Theta_m)\) 拟合残差 \(r_{mi} = y_i - f_{m-1}(x_i)\)。



#### 训练过程

1. 初始化模型 \(f_0(x) = 0\)
2. 对于 m = 1 ... M
    a. 对训练集中的所有数据，计算残差 \(r_{mi} = y_i - f_{m-1}(x_i), i = 1...N\)。
    b. 利用残差训练弱学习器 \(T(x, \Theta_m)\)。
    c. 更新模型 \(f_m(x) = f_{m-1}(x) + T(x, \Theta_m)\)
3. 最终模型
    \[
        f(x) = \sum_{=m1}^M T(x, \Theta_m)
        \]
