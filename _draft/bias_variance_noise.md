#### 数据集

定义总体数据空间为 \(\Omega\)，由多个训练集合构成的集合 \(\mathbb{S}\)

#### 模型

对于每一个训练集 \(D\subset \mathbb{S}\)，可以得到模型 \(f(x;D)\)，对于\(\mathbb{S}\) 来说，得到的期望模型为 \(E_{D\subset\mathbb{S}}[f(x;D)]\)，简写成 \(E[f(x;D)]\)。

#### 方差

不同训练集训练得到的模型对样本的预测值方差  

\[
  var(x) = E[\left(f(x,D) - E[f(x;D)]\right)^2]
  \]

#### 偏差

不同训练集得到的期望模型对样本的预测值与真实值之间的差异
\[
  bias(x) = (y -E[f(x, D)])^2
  \]

#### 噪声

不同训练集里面对某一样本的观测值与真实值之间期望差异

\[
 \epsilon(x) = E[(\hat{y} - y)^2]
  \]

#### 泛化误差

定义损失函数 \(L(y, f(x))\)，泛化误差是模型对总体数据的预测期望损失
\[
  E = \iint_{\Omega} p(x, y)L(y, f(x)) \mathrm{d}x\mathrm{d}y
  \]

如果采用平方损失函数 \(L(y, f(x)) = (y - f(x))^2\)
泛化损失
\[
  E = \iint_{\Omega} p(x, y) \left(y - f(x)\right)^2\mathrm{d}x\mathrm{d}y
  \]
上式可以看作 \(E\) 对 \(f(x)\) 的泛函，假定当 \(f(x) = f^*(x)\) 时，E 可以取到最小值，并将 \(f(x)\) 拆分成两个部分 \(f(x) = f^*(x) + \epsilon\eta(x) \)，这里的 \(\eta(x)\) 可以是任意函数，\(\epsilon\) 是它的系数。于是
\[
    E(\epsilon) = \iint_{\Omega} p(x, y) \left(y - f^*(x) - \epsilon \eta(x)\right)^2\mathrm{d}x\mathrm{d}y
  \]

根据前面的假设， \(\epsilon=0\) 使 \(E\) 达到极值，也就是说
\[\frac{\partial E}{\partial \epsilon} \mid_{\epsilon=0} = 0\]

那么

\[
  \begin{aligned}&-2\iint_{\Omega} p(x, y)(y - f^*(x) ) \eta(x) \mathrm{d}x\mathrm{d}y = 0 \\
  &\Rightarrow \int \left(\int p(x, y)(y-f^*(x)) \mathrm{d}y\right) \eta(x)\mathrm{d}x = 0
  \end{aligned}
  \]

由于对于任意的 \(\eta(x)\)，上式都成立，所以，内部积分应该始终为零
\[
  \begin{aligned}
  &\int p(x, y)(y-f^*(x)) \mathrm{d}y = 0 \\
  &\Rightarrow \int p(x, y) y \mathrm{d}y = f^*(x)\int p(x, y) \mathrm{d}y\\
  &\Rightarrow f^*(x) = \frac{\int p(x, y) y \mathrm{d}y}{p(x)} = \int p(y\mid x) y \mathrm{d}y
  \end{aligned}
  \]

这样就得到了使得泛化损失最小的解，也就是最优解，它可以看作是 \(y\) 关于 \(x\) 的条件均值 \(E(y\mid x)\)。

把泛化误差按下面的方式展开

\[
  \begin{aligned}
  E&=\iint_{\Omega}p(x,y)(y - f(x))^2  \mathrm{d}x\mathrm{d}y \\
  &= \iint_{\Omega}p(x, y) (y - f^*(x) + f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y\\
  &= \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  +\iint_{\Omega} p(x, y) (f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y + \iint_{\Omega} p(x, y)2 (y-f^*(x))(f^*(x)-f(x)) \mathrm{d}x\mathrm{d}y
  \end{aligned}
  \]

其中第二项

\[
  \begin{aligned}\iint_{\Omega} p(x, y) (f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y &= \int (f^*(x) - f(x))^2 \int p(x, y) \mathrm{d}y \mathrm{d}x\\
  &=\int (f^*(x) - f(x))^2 p(x)\mathrm{d}x 
  \end{aligned}
  \]

第三项

\[
  \begin{aligned}
  &\iint_{\Omega}2 p(x, y) (y-f^*(x))(f^*(x)-f(x)) \mathrm{d}x\mathrm{d}y \\
  &=\iint_{\Omega} 2p(x,y) y f^*(x)\mathrm{d}x\mathrm{d}y -
   \iint_{\Omega} 2p(x, y)y f(x)\mathrm{d}x\mathrm{d}y -
   \iint_{\Omega}2p(x, y) (f^*(x))^2\mathrm{d}x\mathrm{d}y + 
   \iint_{\Omega}2p(x, y) f^*(x)f(x) \mathrm{d}x\mathrm{d}y\\
   &= 2\int \left(\int p(x, y)y\mathrm{d}y\right) f^*(x)\mathrm{d}x - 2 \int \left(\int p(x, y)y\mathrm{d}y\right)f(x)\mathrm{d}x -2 \int \left(\int p(x, y)\mathrm{d}y\right) (f^*(x))^2\mathrm{d}x + 2 \int \left(\int p(x, y)\mathrm{d}y\right) f^*(x) f(x)\mathrm{d}x\\
   &=2 \int f^*(x) p(x) f^*(x)\mathrm{d}x - 2 \int f^*(x) p(x) f(x)  \mathrm{d}x-2\int p(x)(f^*(x))^2\mathrm{d}x + 2 \int p(x) f^*(x)f(x)
  \mathrm{d}x\\
  &=0
  \end{aligned}
  \]

于是 
\[
  E = \int (f^*(x) - f(x))^2 p(x)\mathrm{d}x + \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  \]

假设一个由多个训练集合构成的集合 \(\mathbb{S}\)，对于其中的每一个训练集 \(D\subset \mathbb{S}\)，可以得到模型 \(f(x;D)\)，于是对于\(\mathbb{S}\) 来说，得到的平均模型为 \(E_{D\subset\mathbb{S}}[f(x;D)]\)，简写成 \(E[f(x;D)]\)。再用下面的方式展开第一项 
\[
  \begin{aligned}
  &\int (f^*(x) - f(x))^2 p(x)\mathrm{d}x \\&= \int (f^*(x) -E[f(x;D)] + E[f(x;D)]- f(x;D))^2 p(x)\mathrm{d}x \\
  &= \int (f^*(x) - E[f(x;D)])^2  p(x)\mathrm{d}x + \int (E[f(x;D)] -f(x;D))^2  p(x)\mathrm{d}x + \int (f^*(x) - E[f(x;D)]) (E[f(x;D)] -f(x;D)) p(x)\mathrm{d}x
  \end{aligned}
  \]

其中第三项
\[
   \begin{aligned}
  &\int (f^*(x) - E[f(x;D)]) (E[f(x;D)] -f(x;D)) p(x)\mathrm{d}x\\
  &=
  \end{aligned}
  \]