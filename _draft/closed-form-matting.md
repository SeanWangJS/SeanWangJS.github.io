---
title: Closed Form Matting
---

\[
    I_i = \alpha_i F_i + (1 - \alpha_i) B_i
    \]

\[
I_i =\alpha_i F_i + B_i - \alpha_i B_i
    \]

\[
\alpha_i = a I_i + b    
    \]

\\(a = \frac 1 {F - B}\\)
\\(b = -\frac B {F - B}\\)

\[
\alpha_i = \frac {I_i} {F - B}  - \frac B {F - B}
    \]

损失函数

\[
L(\alpha, a, b) = \sum_{j \in I}\left(\sum_{i \in \omega_j} (\alpha_i - a_j I_i - b_j)^2 + \epsilon a_j^2\right)
    \]

\[
L(\alpha, a, b) = \sum_{k = 1}^n \left(
    \sum_{i=1}^9 (\alpha_{ki} - a_k I_{ki}- b_k)^2 + \epsilon a_k^2
    \right)
    \]

\[
L(\alpha, a, b) = \sum_{k=1}^n \left(
    \left|
    \left[
    \begin{aligned}
    a_k I_{k1} + b_k \\...\\ a_k I_{k9} + b_k\\\sqrt{\epsilon}a_k
    \end{aligned}
    \right]
    -
    \left[
        \begin{aligned}
        \alpha_{k1}\\...\\\alpha_{k9} \\0
        \end{aligned}
        \right]
     \right|^2
    \right)    
    \]



\[
L(\alpha, a, b) = \sum_{k=1}^n \left(
    \left|
    \left[
    \begin{aligned}
    I_{k1} &\quad 1 \\...\\ I_{k9} &\quad 1\\\sqrt{\epsilon}&\quad 0
    \end{aligned}
    \right]
    \cdot \left[\begin{aligned}
    a_k \\ b_k
    \end{aligned}\right]
    -
    \left[
        \begin{aligned}
        \alpha_{k1}\\...\\\alpha_{k9} \\0
        \end{aligned}
        \right]
     \right|^2
     \right)
    \]

令 

\[
G_k = \left[
    \begin{aligned}
    I_{k1} &\quad 1 \\...\\ I_{k9} &\quad 1\\\sqrt{\epsilon}&\quad 0
    \end{aligned}
    \right]
    \]

\[
    \bar{\alpha}_k = \left[
        \begin{aligned}
        \alpha_{k1}\\...\\\alpha_{k9} \\0
        \end{aligned}
        \right]
    \]

则有

\[
L(\alpha, a, b) = \sum_{k=1}^n \left(G_k \cdot \left[\begin{aligned}
    a_k \\ b_k
    \end{aligned}\right] - \bar{\alpha}_k\right)^2    
\]

令 

\[
    \begin{aligned}
    J_k &= \left(G_k \cdot c_k- \bar{\alpha}_k\right)^2    \\
    &=  \left(G_k \cdot c_k- \bar{\alpha}_k\right)^T  \left(G_k \cdot c_k- \bar{\alpha}_k\right)
\end{aligned}
    \]

\[
\nabla_{c_k} J_k =2 G_k^T (G_k\cdot c_k - \bar{\alpha}_k) = 0
    \]

\[
c_k = (G_k^T G_k)^{-1} G_k^T \bar{\alpha}_k
\]

\[
    \begin{aligned}
G_k \cdot c_k - \bar{\alpha}_k&= G_k (G_k^T G_k)^{-1} G_k^T \bar {\alpha}_k - \bar{\alpha}_k \\
&= [G_k (G_k^T G_k)^{-1} G_k^T  - I] \bar{\alpha}_k\\
&= \bar{G}_k \bar{\alpha}_k
\end{aligned}
    \]

其中

\[
\bar{G}_k = G_k (G_k^T G_k)^{-1} G_k^T  - I
    \]

\[
L(\alpha, a, b) = \sum_{k=1}^n (G_k \cdot c_k - \bar{\alpha}_k)^2 = \sum_{k=1}^n \bar{\alpha}_k^T \bar{G}_k^T \bar{G_k}\bar{\alpha}_k    
    \]


\[
    \begin{aligned}
    \bar{G}_k^T \bar{G}_k &= (G_k (G_k^T G_k)^{-1} G_k^T  - I)^T (G_k (G_k^T G_k)^{-1} G_k^T  - I)\\
    &=(G_k (G_k^T G_k)^{-1} G_k^T  - I) (G_k (G_k^T G_k)^{-1} G_k^T  - I)\\
    &=G_k (G_k^T G_k)^{-1} G_k^T G_k (G_k^T G_k)^{-1} G_k^T - 2G_k (G_k^T G_k)^{-1} G_k^T + I\\
    &=I - G_k (G_k^T G_k)^{-1} G_k^T
    \end{aligned}
    \]

\[
    \bar{G}_k^T \bar{G}_k = -\bar{G}_k 
    \]

\[
    (\bar{G}_k^T+I)\bar{G}_k = 0
    \]

\[
    a^T M^T M a = a^T (M^T M) \cdot a
    \]

---

\[
    \begin{aligned}
F_i &= \beta_i^F F_1 + (1- \beta_i^F)F_2\\
B_i &= \beta_i^B B_1 + (1- \beta_i^B)B_2
\end{aligned}
    \]

代入

\[
    I_i^c = \alpha_i F_i^c + (1 - \alpha_i) B_i^c
    \]

得到

\[
   I_i^c =  \alpha_i \beta_i^F F_1^c + \alpha_i (1 - \beta_i^F) F_2^c + (1-\alpha_i) \beta_i^B B_1^c + (1-\alpha_i)(1 - \beta_i^B) B_2^c
    \]

\[
I_i^c - B_2^c = \alpha_i \beta_i^F F_1^c - \alpha_i \beta_i^F F_2^c + \alpha_i F_2^c + \beta_i^B B_1^c - \alpha_i \beta_i^B B_1^c  + \alpha_i \beta_i^B B_2^c -\beta_i^B B_2^c - \alpha_i B_2^c 
    \]

\[
I_i^c - B_2^c = \alpha_i \beta_i^F (F_1^c - F_2^c) +( \beta_i^B - \alpha_i \beta_i^B) (B_1^c  - B_2^c)   + \alpha_i (F_2^c- B_2^c )
    \]

\[
\left[(F_1^c - F_2^c)\quad (B_1^c - B_2^c) \quad (F_2^c - B_2^c)\right]\cdot \left[\begin{aligned}
\alpha_i \beta_i^F \\ \beta_i^B - \alpha_i \beta_i^B\\ \alpha_i
\end{aligned}\right] = I_i^c - B_2^c
    \]


\[
\left[
    \begin{aligned}
    (F_1^1 - F_2^1)\quad (B_1^1 - B_2^1) \quad (F_2^1 - B_2^1)\\
    (F_1^2 - F_2^2)\quad (B_1^2 - B_2^2) \quad (F_2^2 - B_2^2)\\
    (F_1^3 - F_2^3)\quad (B_1^3 - B_2^3) \quad (F_2^3 - B_2^3)
    \end{aligned}
    \right]
\cdot \left[\begin{aligned}
\alpha_i \beta_i^F \\ \beta_i^B - \alpha_i \beta_i^B\\ \alpha_i
\end{aligned}\right] = \left[
    \begin{aligned}
    I_i^1 - B_2^1\\I_i^2 - B_2^2\\I_i^3 - B_2^3
    \end{aligned}
    \right]
    \]

令

\[
    H = \left[
    \begin{aligned}
    (F_1^1 - F_2^1)\quad (B_1^1 - B_2^1) \quad (F_2^1 - B_2^1)\\
    (F_1^2 - F_2^2)\quad (B_1^2 - B_2^2) \quad (F_2^2 - B_2^2)\\
    (F_1^3 - F_2^3)\quad (B_1^3 - B_2^3) \quad (F_2^3 - B_2^3)
    \end{aligned}
    \right]
    \]

\[
H
\cdot \left[\begin{aligned}
\alpha_i \beta_i^F \\ \beta_i^B - \alpha_i \beta_i^B\\ \alpha_i
\end{aligned}\right] = \left[
    \begin{aligned}
    I_i^1 - B_2^1\\I_i^2 - B_2^2\\I_i^3 - B_2^3
    \end{aligned}
    \right]
    \]

\[
 \left[\begin{aligned}
\alpha_i \beta_i^F \\ \beta_i^B - \alpha_i \beta_i^B\\ \alpha_i
\end{aligned}\right] = 
H^{-1}
\cdot
\left[
    \begin{aligned}
    I_i^1 - B_2^1\\I_i^2 - B_2^2\\I_i^3 - B_2^3
    \end{aligned}
    \right]
    \]

假设 \\(H^{-1}\\) 的最后一行等于 \\([a^1\quad a^2 \quad a^3]\\)，于是

\[
    \begin{aligned}
\alpha_i &= a^1 (I_i^1 - B_2^1) + a^2 (I_i^2 - B_2^2) + a^3 (I_i^3 - B_2^3)\\
&= \sum_{j=1}^3 a^j I_i^j + b
\end{aligned}
    \]

其中

\[
b = \sum_{i = 1}^ 3 a^i B_2^i
    \]

$$
J(\alpha, a, b) = \sum_{j\in W} \left(
    \sum_{i \in w_j} \left(
        \alpha_i - \sum_{t=1}^3 a_j^t I_i^t - b_j
        \right)^2 + \epsilon \sum_{t=1}^3(a_j^t)^2
    \right)
$$

$$
\left[
    \begin{aligned}
    \sum_{t=1}^3 a_j^t I_1^t + b_j \\
    \sum_{t=1}^3 a_j^t I_2^t + b_j \\
    \sum_{t=1}^3 a_j^t I_3^t + b_j \\
    ...\\
    \sum_{t=1}^3 a_j^t I_9^t + b_j \\
    \sqrt{\epsilon} a_j^1 \\
    \sqrt{\epsilon} a_j^2 \\
    \sqrt{\epsilon} a_j^3
    \end{aligned}
    \right]
    -
    \left[
        \begin{aligned}
        \alpha_1\\
        \alpha_2\\
        \alpha_3\\
        ... \\
        \alpha_9\\
        0 \\
        0 \\
        0
        \end{aligned}
        \right]
$$

$$
\left[
    \begin{aligned}
    I_1^1 \quad I_1^2 \quad I_1^3 \quad 1 \\
    I_2^1 \quad I_2^2 \quad I_2^3 \quad 1 \\
    I_3^1 \quad I_3^2 \quad I_3^3 \quad 1 \\
    ...\\
    I_9^1 \quad I_9^2 \quad I_9^3 \quad 1 \\
    \sqrt{\epsilon} \quad 0 \quad 0 \quad 0 \\
    0 \quad \sqrt{\epsilon} \quad 0 \quad 0 \\
    0 \quad 0 \quad \sqrt{\epsilon} \quad 0
    \end{aligned}
    \right]
    \left[
        \begin{aligned}
        a_j^1 \\
        a_j^2 \\
        a_j^3 \\
        b_j
        \end{aligned}
        \right]
         -
    \left[
        \begin{aligned}
        \alpha_1\\
        \alpha_2\\
        \alpha_3\\
        ... \\
        \alpha_9\\
        0 \\
        0 \\
        0
        \end{aligned}
        \right]
$$

---

\[
L(\alpha, a, b) = \sum_{k=1}^n \left(
    \left|
    \left[
    \begin{aligned}
    a_k I_{k1} + b_k \\...\\ a_k I_{k9} + b_k
    \end{aligned}
    \right]
    -
    \left[
        \begin{aligned}
        \alpha_{k1}\\...\\\alpha_{k9}
        \end{aligned}
        \right]
     \right|^2
     +\epsilon a_k^2
    \right)    
    \]

\[
L(\alpha, a, b) = \sum_{k=1}^n \left(
    \left|
    \left[
    \begin{aligned}
    I_{k1} &\quad 1 \\...\\ I_{k9} &\quad 1
    \end{aligned}
    \right]
    \cdot \left[\begin{aligned}
    a_k \\ b_k
    \end{aligned}\right]
    -
    \left[
        \begin{aligned}
        \alpha_{k1}\\...\\\alpha_{k9} 
        \end{aligned}
        \right]
     \right|^2
     \right)
     +\sum_{k=1}^n \epsilon a_k^2
    \]

\[
L(\alpha, a , b) = \sum_{k=1}^n \parallel
    G_k \cdot c_k - \bar{\alpha}_{k}
    \parallel^2 + \sum_{k=1}^n \epsilon_k^2
    \]



---

优化问题

\[
    \begin{aligned}
    &\min \alpha^T L \alpha\\
    s.t. \quad &\alpha_i = 0 \quad for  \quad i\in BG\\
    &\alpha_i = 1 \quad for  \quad i\in FG
    \end{aligned}
    \]

拉格朗日函数

\[
l(\alpha, \lambda) = \alpha^T L \alpha + \sum_{i\in BG}\lambda_i \alpha_i + \sum_{i\in FG} \lambda_i (1-\alpha_i)    
\]

进一步

\[
    l(\alpha, \lambda) = \alpha^T L \alpha + \lambda \alpha + C
    \]

其中 \\(C\\) 为常数，\\(\lambda\\) 在索引 \\(i \notin BG \land i\notin FG\\) 上的值为 0。

优化条件

\[
\frac{\partial L}{\partial \alpha} = L \alpha + \lambda = 0
    \]

即求解线性方程组

