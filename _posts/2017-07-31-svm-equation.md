---
layout: default
---

## 支持向量机
在n维空间上定义一个超平面

$$\omega^T x + b = 0$$

### 定义：函数间隔

对于一个特征 F，通常使用空间中的一点来表示，如果已知其分类类型，那么这就成为一个训练数据，用二元组表示为：$$ (x^{(i)}, y^{(i)}) $$。定义训练数据与超平面的函数间隔

$$\hat{\gamma}^{(i)} = y^{(i)} ( \omega^T x^{(i)} + b)$$

将点 F 代入函数 $$\omega^T x + b$$ 中，实际上是在判断此点与上述超平面的位置关系。其正负号决定了该点位于超平面的哪一侧，而其绝对值则度量了该点与超平面的距离。
如果规定 $$\omega^T x^{(i)} + b > 0 $$ 时，$$ y^{(i)} = 1 $$，反之 $$ y^{(i)} = -1 $$。那么显然，$$\hat{\gamma}^{(i)}$$ 始终为正。于是 $$\hat{\gamma}^{(i)}$$ 就可以看作点 F 到超平面的相对距离。
若有一组特征及其分类结果组成的训练数据 $$S = \{(x^{(i)}, y^{(i)}) | i = 1,,,m\}$$ ，那么最小函数间隔就为

$$ \hat{\gamma} = \min\limits_{i = 1,,,m} \hat{\gamma}^{(i)} $$

### 定义：几何间隔

![](/resources/2017-07-31-svm/geo-margin.png)

可以证明，以 $$\omega$$ 为系数的超平面本身垂直于向量 $$\omega$$ 。假设点A的坐标为 $$x^{(i)}$$ ，到超平面的距离为 $$ \bar{\gamma}^{(i)} $$。那么过点A，作垂直于超平面的直线，交点为B，AB两点满足下面的关系

$$ x^{(i)} = x_B + \omega \bar{\gamma}^{(i)} $$

然后根据 $$ \omega^T x_B + b = 0 $$ ， 可得

$$ \bar{\gamma}^{(i)} = \frac{\omega^T}{||\omega||} x^{(i)} + \frac{b}{||\omega||} $$

同函数间隔类似，规定，当 $$\bar{\gamma}^{(i)} > 0$$ 时，$$y^{(i)} = 1$$ ，否则 $$y^{(i)} = -1$$。然后定义

$$\gamma^{(i)} = y^{(i)} \bar{\gamma}^{(i)} = y^{(i)} \left(\frac{\omega^T}{||\omega||} x^{(i)} + \frac{b}{||\omega||} \right)$$

显然，$$\gamma^{(i)} > 0$$ 恒成立。这里把 $$\gamma^{(i)}$$ 称为点A到超平面的几何间隔。值得注意的是，几何间隔实际就是点到超平面的距离。同样，对于一组训练数据集，定义最小几何间隔

$$\gamma = \min\limits_{i = 1,,,m} \gamma^{(i)}$$

### 最优间隔分类器

所谓分类器其实就是一个超平面，其作用是在空间中将两类数据分开，可以想象，这样的超平面如果存在，那么很可能不止一个。而我们想要的其实是满足下述条件的超平面：对于一组训练数据，使得最小几何间隔最大，即所谓的最优间隔分类器，如下图所示

![](/resources/2017-07-31-svm/max-margin.png)

也就是说，要求解下面的优化问题

$$
\begin{align*}
  \begin{array} {cc}
    \max\limits_{\omega, b}\quad \gamma \\ 
    s.t.  \qquad\!y^{(i)} \left(\frac{\omega^T}{||\omega||} x^{(i)} + \frac{b}{||\omega||}\right) \geq \gamma \qquad i = 1,,,m
  \end{array}
\end{align*}
$$

考虑到函数间隔和几何间隔的定义，可以发现

$$\hat{\gamma}^{(i)} = ||\omega|| \gamma^{(i)}$$
 
于是，原问题可以重新表述为

$$
\begin{align*}
  \begin{array} {cc}
    \max\limits_{\omega, b}\quad \frac{\hat\gamma}{||\omega||} \\ 
    s.t.  \qquad\!y^{(i)} \left(\frac{\omega^T}{||\omega||} x^{(i)} + \frac{b}{||\omega||}\right) \geq \hat\gamma \qquad i = 1,,,m
  \end{array}
\end{align*}
$$

这就将原本由几何间隔表达的问题，转换成了用函数间隔来表达。
这样做的原因在于：几何间隔是点到超平面的绝对距离，而函数间隔是相对距离，可以通过缩放 $$\omega, b$$ 这两个参数来调整。显然这一缩放操作，并不会影响超平面的位置。所以我们可以通过缩放将 $$\hat\gamma$$ 设置为1，而不改变整个系统本质。
所以对 $$\frac{\hat{\gamma}^{(i)}}{||\omega||}$$ 的最大化，就是最大化 $$\frac{1}{||\omega||}$$，也等价于最小化 $$\frac 1 2 \omega^2$$。所以原问题又转换为

$$
\begin{align*}
  \begin{array} {cc}
    \max\limits_{\omega, b}\quad \frac 1 2 \omega^2 \\ 
    s.t.  \qquad\!y^{(i)} \left(\frac{\omega^T}{||\omega||} x^{(i)} + \frac{b}{||\omega||}\right) \geq 1 \qquad i = 1,,,m
  \end{array}
\end{align*}
$$

### 拉格朗日对偶问题

考虑同时带有等式约束和不等式约束的优化问题（这里称之为原问题）

$$
\begin{align*}
  \begin{array} {cc}
    \min\limits_{\omega}\quad f(\omega) \\ 
    s.t. \qquad g_i(\omega) \leq 0 \quad i = 1,,,m\\
	\qquad h_i(\omega) = 0 \quad i = 1,,,k
	
  \end{array}
\end{align*}
$$

建立拉格朗日函数

$$L(\omega, \alpha, \beta) = f(\omega) + \sum_{i=1}^m \alpha_i g_i(\omega) + \sum_{i=1}^k \beta_i h_i(\omega)$$ 

进而考虑下述优化问题

$$\Theta_P(\omega) = \max\limits_{\alpha, \beta} L(\omega, \alpha, \beta) \quad \alpha_i > 0$$

如果存在 $$g_i(\omega) > 0$$ 或者 $$h_i(\omega) \neq 0$$，那么我们可以找到适当 $$\alpha, \beta$$ 的使得 $$\Theta_P(\omega) = \infty$$。反之，如果 $$\omega$$ 满足原问题的所有约束条件，那么 $$L(\omega, \alpha, \beta)$$ 的最大值就等于 $$f(\omega)$$。
于是有

$$
\begin{equation}
  \Theta_P(\omega) = \left\{
    \begin{aligned}
	  f(\omega) \quad if \quad h_i(\omega) = 0 \quad and\quad g_i(\omega) \geq 0 \quad for \quad all \quad i\\ 
	  \infty \qquad otherwise
	\end{aligned}
    
  \right.
\end{equation}
$$

从而原问题可以表述为


$$\min\limits_\omega \Theta_P(\omega) = \min\limits_\omega \max\limits_{\alpha, \beta} L(\omega, \alpha, \beta)$$


假设 $$p^* = \min\limits_\omega \Theta_P(\omega)$$ 为原问题的解。另一方面，定义原问题的对偶问题

$$
d^* = \max\limits_{\alpha, \beta} \min\limits_\omega L(\omega, \alpha, \beta)
$$

可以证明 $$d^* \leq p^*$$ 恒成立。并且在满足特定情况的条件下
$$d^* = p^*$$

假设 $$f, g_i(\omega)$$ 是凸函数，$$h_i(\omega)$$ 可以表示成 $$h_i(\omega) = a_i \omega + b$$，且存在 $$\omega$$ 使得 $$g_i(\omega) < 0$$。
在这种假设下，必然存在 $$\omega^*$$ 是原问题的解，$$\alpha^*, \beta^*$$ 是对偶问题的解，并且 $$d^* = p^* = L(\omega^*, \alpha^*, \beta^*)$$，
而且 $$\omega^*, \alpha^*, \beta^*$$ 还应满足KKT条件，即

$$
\begin{align*}
  \begin{array} {cc}
    \frac{\partial L^*}{\partial \omega_i} = 0, i = 1,,,n \quad(1)\\
	\frac{\partial L^*}{\partial \beta_i} = 0, i = 1,,,n \quad (2)\\
	\alpha_i^* g_i(\omega) = 0 ,i = 1,,,m \quad (3)\\
	g_i(\omega) \leq 0, i = 1,,,m \quad (4) \\ 
	\alpha_i \geq 0, i=1,,,m \quad (5)
	
  \end{array}
\end{align*}
$$

特别注意第(3)个条件，如果 $$\alpha_i^* > 0$$，那么 $$g_i(\omega) = 0$$。

### 最优间隔分类器

现在回到最优间隔分类器的讨论，之前已经提出了下述的等价优化问题

$$
\begin{align*}
  \begin{array} {cc}
    \min\limits_{\omega}\quad f(\omega) \\ 
    s.t. \qquad g_i(\omega) \leq 0 \quad i = 1,,,m\\
	\qquad h_i(\omega) = 0 \quad i = 1,,,k
	
  \end{array}
\end{align*}
$$

若设 $$ g_i(\omega) = -y^{(i)} (\omega^T x^{(i)} + b) + 1$$，那么约束条件可以表述为 $$g_i(\omega) \leq 0 \quad i = 1,,,m$$

![](/resources/2017-07-31-svm/max-margin2.png)

考虑到只有当数据点的函数间隔为1（即函数间隔取得最小值）时才有 $$g_i(\omega) = 0$$。所以对于大部分的点（如图所示），都有 $$g_i(\omega) > 0$$ ，也就是说 $$\alpha_i = 0$$ 。而使得 $$\alpha_i > 0$$ 的点即所谓的**支持向量** 。

现在利用拉格朗日乘数法建立拉格朗日函数

$$
L(\omega, b, \alpha) = \frac 1 2 \omega^2 + \sum_{i=1}^m \alpha_i [-y^{(i)} (\omega^T x^{(i)} + b) + 1]
$$

仿照前面的讨论，原问题表述为

$$
p^* = \min\limits_{\omega, b} \Theta_P (\omega, b, \alpha) = \min\limits_{\omega, b} \max\limits_\alpha L(\omega, b, \alpha)
$$

相应的对偶问题

$$
d^* = \max\limits_{\alpha} \Theta_D (\omega, b, \alpha) = \max\limits_\alpha \min\limits_{\omega, b} L(\omega, b, \alpha)
$$

我们首先求解 $$\min\limits_{\omega, b} L(\omega, b, \alpha)$$ ，通过求导建立方程组

$$
\begin{align*}
  \begin{array}{cc}
    \Delta_\omega L = \omega - \sum_{i=1}^m \alpha_i y^{(i)}x^{(i)} = 0\\
	\frac{\partial L}{\partial b} = \sum_{i=1}^m \alpha_i y^{(i)} = 0
  \end{array}
\end{align*}
$$

解得

$$
\begin{align*}
  \begin{array}{cc}
	\omega = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}\\
	\sum_{i=1}^m \alpha_i y^{(i)} = 0
  \end{array}
\end{align*}
$$

然后将 $$\omega$$ 代回到拉格朗日函数，得到

$$
W(\alpha) = L(\omega, b, \alpha) = -\frac 1 2 \sum_{i, j=1}^m \alpha_i \alpha_j y^{(i)}y^{(j)} <x^{(i)}, x^{(j)}> + \sum_{i=1}^m \alpha_i
$$

其中 $$ <x^{(i)}, x^{(j)}>$$表示两个量的内积。于是对偶问题的解 

$$d^* = \max\limits_\alpha \min\limits_{\omega, b} L(\omega, b, \alpha) = \max\limits_\alpha W(\alpha)$$

其中 $$\alpha$$ 具有约束

$$
\begin{align*}
  \begin{array}{cc}
	\alpha_i \geq 0, \quad i = 1,,,m\\
	\sum_{i=1}^m \alpha_i y^{(i)} = 0
  \end{array}
\end{align*}
$$

### 软间隔分类期

![](/resources/2017-07-31-svm/soft-margin.png)

考虑两种情形，第一种如上图所示，有一个数据点产生异常。如果要使产生的超平面将两类数据完全分开，即如虚线所示，但这条线对数据的分割明显不如实线，也就是说可疑数据的干扰影响了分类器的正确判断。另外一种情况是，数据点集合本身就无法完全分割。这样一来就需要对算法进行改进，使其能适当允许数据偏离。定义如下优化：

$$
\begin{align*}
  \begin{array} {cc}
    \min\limits_{\omega, b}\quad \frac 1 2 \omega^2 + C \sum_{i=1}^m \xi_i \\ 
    s.t. \qquad y^{(i)}(\omega^T x^{(i)} + b \geq 1 -\xi) \quad i = 1,,,m\\
	\qquad \xi \geq 0 \quad i = 1,,,m
  \end{array}
\end{align*}
$$

也就是说，允许某些点到超平面的函数距离小于1，甚至越过超平面，但是其代价是在目标函数产生适当增量， $$C$$ 为适当参数，应在实际训练中调节，最终产生对各方面都有适当顾及的分割面。然后建立拉格朗日函数

$$
L(\omega,b,\alpha,\gamma, \xi) = \frac 1 2 \omega^2 + C \sum_{i=1}^m \xi_i + \sum_{i=1}^m \alpha_i [-y^{(i)}(\omega^T x^{(i)} + b) - \xi_i + 1] - \sum_{i=1}^m \xi_i \gamma_i
$$

其中 $$\gamma$$ 恒大于等于0。 

然后再考虑原问题与相应的对偶问题

$$
p^* = \min\limits_{\omega, b, \xi} \Theta_P (\omega, b, \alpha, \xi,\gamma) = \min\limits_{\omega, b,\xi} \max\limits_{\alpha, \gamma \geq 0} L(\omega, b, \alpha, \xi, \gamma)
$$

$$
d^* = \max\limits_{\alpha,\gamma} \Theta_D (\omega, b, \alpha,\xi,\gamma) = \max\limits_{\alpha, \gamma} \min\limits_{\omega, b, \xi} L(\omega, b, \alpha, \xi, \gamma)
$$

对参数求导解问题 $$\min\limits_{\omega, b, \xi} L(\omega, b, \alpha, \xi, \gamma)$$

$$
\begin{align*}
  \begin{array} {cc}
    \Delta_\omega L = \omega - \sum_{i=1}^m \alpha_i y^{(i)}x^{(i)} = 0\\
	\frac{\partial L}{\partial b} = \sum_{i=1}^m \alpha_i y^{(i)} = 0\\
	\frac{\partial L}{\partial \xi_j} = C -\alpha_j - \gamma_j = 0
  \end{array}
\end{align*}
$$

第1、2个方程与前面的推导一样，唯一的区别是增加了一个条件 $$C-\alpha_i - \gamma_i = 0$$，由于 $$\gamma \geq 0$$ 始终成立，所以 $$\alpha_i < C$$。
于是最终，我们要求解的问题更新为

$$
\begin{align*}
  \begin{array} {cc}
    \max\limits_{\alpha}\quad W(\alpha) \\ 
    s.t. \quad 0 \leq \alpha_i \leq  C \quad i = 1,,,m\\
	\qquad \sum_{i=1}^m \alpha_i y^{(i)} = 0 \quad i = 1,,,m
  \end{array}
\end{align*}
$$

现在来考虑软间隔问题优化的KKT条件(3)，这里分两种情况来讨论:
1、当 $$1-y^{(i)}(\omega^T x^{(i)} + b) \leq 0$$，这时 $$\xi_i=0$$，属于线性可分情况，那么 $$\alpha_i g_i(\omega) leq 0$$从而当 $$y^{(i)}(\omega^T x^{(i)} + b) > 1$$ 时，$$\alpha_i=0$$，而当时 $$y^{(i)}(\omega^T x^{(i)} + b) = 1$$，$$0 < \alpha_i < C$$

2、当 $$1-y^{(i)}(\omega^T x^{(i)} + b) > 0$$，这时 $$\xi_i>0$$，即遇到不可分数据，而原问题要求最大化拉格朗日函数 L，于是需要 $$\gamma_i = 0$$，那么就有 $$\alpha_i = C$$。

结合以上两种情况，可以得出 $$\alpha_i$$ 作为最优解应满足的条件
 
$$
\begin{align*}
  \begin{array} {cc}
    y^{(i)}(\omega^T x^{(i)} + b) > 1  => \alpha_i = 0\\
	y^{(i)}(\omega^T x^{(i)} + b) = 1  => 0 < \alpha_i < C\\
	y^{(i)}(\omega^T x^{(i)} + b) < 1  => \alpha_i = C
  \end{array}
\end{align*}
$$ 

通过以上的分析，我们得到了支持向量机应该求解的优化问题，然而这只是第一步，怎样高效的求解上述问题将是下一阶段的话题。

