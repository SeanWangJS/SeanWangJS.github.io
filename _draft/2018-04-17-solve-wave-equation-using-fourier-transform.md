---
layout: post
title: 使用傅立叶变换求解波动方程
description:
tags: 傅立叶变换 Fourier-Transform 波动方程
---

### 一维情况

最简单的情况下，一维波动方程的形式为

$$
\frac{\partial^2 u}{\partial t^2} = v^2 \frac{\partial^2 u}{\partial x^2}
$$

其中 \\(v\\) 是波速。下面我们应用傅立叶变换将时域上的方程转换到频域上

$$
-\omega^2 \mathcal{F}(u) = v^2 \frac{\partial^2}{\partial x^2}\mathcal{F}(u)
$$

这里我们用到了傅立叶变换的微分性质

$$
\begin{aligned}
&\mathcal{F}[f'(t)] = i\omega \mathcal{F}[f(t)]\\
&\mathcal{F}[f^n(t)] = i^n \omega^n \mathcal{F}[f(t)]
\end{aligned}
$$

并且 \\(\mathcal{F}(u)\\) 是关于 \\(x\\) 和频率 \\(\omega\\) 的函数，在这里令其为

$$
\mathcal{F}(u) = \hat {u} (x, \omega)
$$

如果固定 \\(\omega\\) ，并令 \\(y\\) 为关于 \\(x\\) 的函数， \\(y(x) = \hat u(x,\omega)\\) ，则有

$$
-\omega^2 y = v^2 \frac{\mathbf{d}^2 y}{\mathbf{d} x^2}
$$

这是一个二阶常微分方程，设其解具有形式 \\(y=e^{rx}\\)，代入后可得特征方程为

$$
v^2 r^2 + \omega^2  = 0
$$

解之，得到

$$
r = \pm i\frac{w}{v}
$$

也就是说，该特征方程具有两个相异得复数根，于是常微分方程的通解为

$$
y(x) = C_1 \cos (\frac {\omega}{v}x) + C_2 \sin(\frac{\omega}{v}x)
$$

其中 \\(C_1, C_2\\) 是与 \\(\omega\\) 相关的系数，可由初始条件和边界条件确定

$$
\begin{aligned}
C_1 = C_1(\omega)\\
C_2 = C_2(\omega)
\end{aligned}
$$

所以，\\(u\\) 的傅立叶变换解可以写成下面的形式

$$
\hat u(x, \omega) = C_1(\omega) \cos (\frac {\omega}{v}x) + C_2(\omega) \sin(\frac{\omega}{v}x)
$$

下面我们就具体地来计算一个问题，设计算区间为 $$x\in [-10,0]$$ ，并且

$$
\left\{\begin{aligned}
&u(x, 0) = 0\\
&\frac{\partial u}{\partial t}(x,0)= \sin(x)
\end{aligned}
\right.
$$

也就是说，一开始，整个区间处于静止状态，.........。对上面两个方程同样进行时域傅立叶变换

$$
\left\{\begin{aligned}
&\hat u(x, \omega)|_ {\omega = 0} = 0\\
&i\omega \hat u(x, \omega)|_ {x=-10} = \sqrt{2\pi}\delta (\omega)
\end{aligned}
\right.
$$

