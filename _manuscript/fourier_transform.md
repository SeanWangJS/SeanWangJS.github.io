#### 脉冲函数的傅里叶变换

考虑脉冲函数

\[
  \delta(t) = \left\{ 
    \begin{aligned}
  &\infin, \quad t = 0\\
  &0,\quad\,\,\, t \ne 0
  \end{aligned}\right.
  \]

设其满足条件 \(\int_{-\infty}^{\infty}\delta(t) \mathrm{t} = 1\)。推导 \(\delta(t)\) 的傅里叶变换：

\[
  \mathcal{F}(\delta(t)) = F(\mu) = \int_{-\infty}^{\infty} \delta(t)e^{-i2\pi \mu t}\mathrm{d}t
    \]

#### 周期脉冲函数

定义在零点处的脉冲函数为 \(\delta(t)\)，则相应的，位于点 \(x\) 处的脉冲函数为 \(\delta (t - x)\)，两个不同位置的脉冲函数的和也可以看作是一个具有两个脉冲的函数，于是以 \(\Delta T\) 为周期的多脉冲函数可以表示为 

\[
  s_{\Delta T} (t) = \sum_{-\infty}^{\infty} \delta(t - n \Delta T)
  \]

#### 采样

对函数 \(f(t)\) 进行采样的数学表示为，使用周期乘以函数本身，即 

\[
  \widetilde f(t) =  \sum_{-\infty}^{\infty} f(t) \delta(t - n\Delta T)
  \]

设 \(f(t)\) 的傅里叶变换为 \(F(\mu)\)，周期脉冲函数的傅里叶变换为 \(S(\mu)\)，那么根据卷积定理可知，它们的乘积的傅里叶变换就为两个函数在频率域的卷积
