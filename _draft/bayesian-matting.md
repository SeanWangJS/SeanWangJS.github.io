---
title: 贝叶斯抠图算法
---

后验概率

\[
P(F, B, \alpha \mid C) =\frac{ P(C \mid F, B, \alpha) P(F, B, \alpha )  }{P(C)}
    \]

目标函数

\[
    L(F, B, \alpha) = \ln P(C\mid F, B, \alpha) + \ln P(F) + \ln P(B) + \ln P(\alpha)
    \]

假设

\[
P(C \mid F, B, \alpha) \thicksim N(\bar C , \sigma_C^2)
    \]

其中

\[
N(\bar C, \sigma_C^2) = \exp\left(-\frac{1}{2\sigma_C^2} (C - \bar C)^T (C - \bar C)\right)
   \]

以及 

\[
\bar C =\alpha F +(1 - \alpha) B    
    \]

所以

\[
\ln P(C \mid F, B, \alpha) = - \frac{1}{2\sigma_C^2}  (C - \bar C)^T (C - \bar C)
    \]

假设

\[
    P(F) \thicksim N(\bar F, \Sigma_F)
    \]

其中 

\[
    \begin{aligned}
\bar F &= \frac 1 {W_F} \sum_{i \in N} w^F_i F_i \\
\Sigma_F &= \frac 1 {W_F} \sum_{i \in N} w^F_i (F_i - \bar F) (F_i - \bar F)^T
\end{aligned}
    \]

权重

\[
    \begin{aligned}
w^F_i &= \alpha_i g_i = \alpha_i N(0, 8)\\
W_F &= \sum_{i \in N} w_i^F
\end{aligned}
    \]

所以

\[
\ln P(F) = - \frac 1 2(F - \bar F)^T\Sigma_F^{-1} (F - \bar F)
    \]

假设

\[
P(B) \thicksim N(\bar B, \Sigma_B)
    \]

同理可得

\[
\ln P(B) = -\frac 1 2 (B - \bar B)^T \Sigma_B^{-1} (B - \bar B)
    \]

其中

\[
    w^B_i =(1- \alpha_i) g_i = (1-\alpha_i )N(0, 8)
    \]

得到目标函数

\[
    \begin{aligned}
L(F, B, \alpha) &= - \frac{1}{2\sigma_C^2}  (C - \alpha F - (1 - \alpha) B)^T (C - \alpha F - (1 - \alpha) B)\\
&- \frac 1 2(F - \bar F)^T\Sigma_F^{-1} (F - \bar F)-\frac 1 2 (B - \bar B)^T \Sigma_B^{-1} (B - \bar B)
\end{aligned}
    \]

首先令 \\(\alpha\\) 为常数，对 \\(F, B\\) 分别求导

\[
    \frac{\partial L}{\partial F} = \frac{\alpha}{\sigma_C^2}(C - \alpha F - (1-\alpha)B) - \Sigma_F^{-1}(F - \bar F)
    \]


\[
    \frac{\partial L}{\partial B} = \frac{(1-\alpha)}{\sigma_C^2}(C - \alpha F - (1-\alpha)B) - \Sigma_B^{-1}(B - \bar B)
    \]

令

\[
    \frac{\partial L}{\partial F} = 0, \frac{\partial L}{\partial B} = 0
    \]

得到

\[
\left(\frac{\alpha^2 I}{\sigma_C^2} + \Sigma_F^{-1}\right) F + \frac{\alpha I}{\sigma_C^2}(1-\alpha)B = \frac{\alpha}{\sigma_C^2}C + \Sigma_F^{-1}\bar F
    \]

\[
\frac{(1-\alpha)\alpha I}{\sigma_C^2}F +\left(\frac{(1-\alpha)^2 I}{\sigma_C^2} + \Sigma_B^{-1} \right)B = \frac{(1-\alpha)I}{\sigma_C^2}C + \Sigma_B^{-1} \bar B
    \]

还可以写成矩阵形式

\[
    \left[
        \begin{aligned}
        \frac{\alpha^2 I}{\sigma_C^2} + \Sigma_F^{-1}&\quad \frac{\alpha I}{\sigma_C^2}(1-\alpha)\\
        \frac{(1-\alpha)\alpha I}{\sigma_C^2}&\quad \frac{(1-\alpha)^2 I}{\sigma_C^2} + \Sigma_B^{-1}
        \end{aligned}
        \right]
        \left[
            \begin{aligned}
            F\\ B
            \end{aligned}
            \right]
            =\left[
            \begin{aligned}
            \frac{\alpha}{\sigma_C^2}C + \Sigma_F^{-1}\bar F\\ \frac{(1-\alpha)I}{\sigma_C^2}C + \Sigma_B^{-1} \bar B
            \end{aligned}
            \right]
    \]

求解上述线性方程组，获得 \\(F, B\\)

然后固定 \\(F, B\\)，对 \\(\alpha\\) 求导

\[
\frac{\partial L}{\partial \alpha} = \frac 1 {\sigma_C^2} (C- \alpha F - (1-\alpha)B )^T (-F + B)   =0
    \]

化简可得

\[
\alpha = \frac{(C -B)^T(F - B) }{(F - B)^T(F - B)}
    \]

