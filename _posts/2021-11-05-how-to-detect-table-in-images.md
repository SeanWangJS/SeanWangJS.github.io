---
title: 如何检测出图片中的表格
tags: 霍夫变换 直线检测 表格检测
---

通常，在我们使用 OCR 识别图片的时候，模型本身是无法知晓文档结构的，对于那些包含表格的文档，常出现的情况是识别出文字，但丢失了原有的表格结构。针对这一情况，我们来开发一个专门检测图片中表格的算法。

表格检测的具体思路很简单，考虑到表格是由多条线条构成的，因此我们可以首先把这些线条检测出来。在数字图像处理技术中，有一类比较经典的直线检测算法，被称为霍夫变换（Hough transform）。它的具体原理是利用直线极坐标方程

$$
  r = x \cos \theta + y \sin \theta
  $$

将 \\(x-y\\) 平面上的一个点 \\((x_0, y_0)\\) 变换成 \\(\theta-r\\) 平面上的曲线 \\(r = x_0 \cos \theta + y_0 \sin \theta\\)

![](/resources/2021-11-05-how-to-detect-table-in-images/table-detection_polar-point.png)

也就是说，平面上的每个点，都对应着这样一条曲线。而我们知道，平面上的每个点有无数条直线穿过，那么也可以说，这些直线构成的簇对应着这样一条曲线。更进一步地说，由于这是 \\(\theta-r\\) 平面上的曲线，所以这条曲线上的每个点 \\((\theta_0, r_0)\\) 都对应着 \\(x-y\\) 平面上的一条直线

$$
  r_0 = x\cos \theta_0 + y \sin \theta_0
  $$

如果我们再在平面上找到另外一个点 \\((x_1, y_1)\\)，然后画出它变换后的曲线

![](/resources/2021-11-05-how-to-detect-table-in-images/table-detection_intersect.png)

可以看到，这两个点的变换曲线有一个交点，显然，这个交点对应的直线就是经过 \\((x_0, y_0)\\) 和 \\((x_1, y_1)\\) 的直线。

现在，如果我们把 \\(x-y\\) 平面的一条直线上的所有点都变换到 \\(\theta-r\\) 平面上，则可以预见的是，这些曲线将全部相交于同一点。反过来想，假如我们把 \\(x-y\\) 平面上**明显的点**都不分青红皂白的全部映射到 \\(\theta-r\\) 平面上，然后找到这些曲线的所有交点，并统计相交的曲线数量，那么显然，有大量曲线相交的点对应的 \\(x-y\\) 平面上的直线就是我们要找的直线。

![](/resources/2021-11-05-how-to-detect-table-in-images/./table-detection_table.png)

OpenCV 库为我们提供了 HoughLines 和 HoughLinesP 方法可以轻松实现直线检测。但距离我们的目标还差的远，以上图为例，我们使用如下代码检测图片上的直线，并绘制出来

```python
img_canny = cv2.Canny(img, 50, 150)
lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, 10, minLineLength=10, maxLineGap=2)
```

![](/resources/2021-11-05-how-to-detect-table-in-images/./table-detection_table-lines.png)

然而，该算法不仅检测出了表格线，还把部分文字中的线段也识别出来了，因此，接下来我们将想办法把这些误识别的线段给排除掉。

稍加观察可以发现，文字上的线段很多都是孤立的，另一部分是成团聚集在一个很小的区域，而表格线则阡陌交通，互相连接。也就是说，假如我们能将这些线段进行分类，把相互连接的线段归为同一组，那么就能根据每组线段的分布特征，来确定是不是表格线了。所以，现在的问题就变成了把相互连接的线段分到同一个组。如果大家熟悉图算法，可以立即联想到这其实是一个求图连通分量的问题。

将线段看作 Vertex，将相互连接的线段看作 Edge，那么这些线段就构成了一个 Graph。为了简单起见，我们使用邻接矩阵来表示此 Graph，生成邻接矩阵的方法很简单，对所有线段进行两两比较，如果两线段相连，则将矩阵的相应位置设为 1，至于如何判断两线段相连，我们后面再做介绍。

连通分量就是由所有能够相互连接的 Vertex 组成的子图，我们的任务就是找到所有的孤立子图，下面我们采用深度优先搜索来解决该问题：

1. 初始化数组 visited，长度等于 Vertex 数量，全部设置为 0，标记所有访问过的节点；
2. 初始化变量 compoent_id = 1，作为连通分量id；
3. 声明一个空栈 stack 用于存储 Vertex id，然后将 0 入栈；
4. 如果 stack 不为空，则循环：
  a. 将 stack 顶部的 Vertex id 出栈，然后查看邻接矩阵的相应行，如果该行的所有元素都为 0，则说明该 Vertex 是孤立的，将 visited 的相应位置标设为 component_id，然后 component_id 自增 1。如果该行含有非 0 元素，则将该行的所有非 0 位置入栈，并将 visited 的相应位置标设为 component_id，以表示它们属于同一个连通分量；
  b. 如果此时 stack 为空，则检查 visited 数组是否含有 0 元素，如果有，则将第一个 0 元素位置编号入栈，且 compoenent_id 自增 1。

下面给出算法伪码：

> connect_components(neighbor_matrix):

> &emsp; init: visited, compoent_id, stack = [0]

> &emsp; loop while stack not empty:

> &emsp; &emsp; i = stack.pop()

> &emsp; &emsp; isconnect = neighbor_matrix[i]

> &emsp; &emsp; if isconnect all 0:

> &emsp; &emsp; &emsp; visited[i] = component_id

> &emsp; &emsp; &emsp; component_id += 1

> &emsp; &emsp; for j in position where isconnect != 0:

> &emsp; &emsp; &emsp; visited[j] = component_id

> &emsp; &emsp; if stack is empty:

> &emsp; &emsp; &emsp; push the first non-zero vistied index to stack

> &emsp; &emsp; &emsp; component_id += 1

这样一来，visited 数组就包含了每个 Vertex 属于哪个连通分量的信息，对于每一个连通分量，我们找到它所占的矩形区域，然后计算该区域的面积，根据常识来看，表格的面积比其他区域的面积要大的多，因此可以将这些大面积的矩形区域作为表格位置。以10000作为面积阈值，检测的结果如下

![](/resources/2021-11-05-how-to-detect-table-in-images/./table-detection_table-detect.png)

最后，我们来看一下计算两条线段是否连接的方法，主要参考自[该文档](https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf)，该算法计算两条线段的最短距离，常用于碰撞检测，这里我们实现一个简化的版本就足够了。

我们令 \\(p_0, p_1\\) 为一条线段的起点和终点，\\(q_0, q_1\\) 为另一条线段的起点和终点，那么，这两条线段上的点的参数方程分别为 

$$
  \begin{aligned}
  p = (1 - s) p_0 + s p_1\\
  q = (1 - t) q_0 + t q_1
  \end{aligned}
  $$

其中 \\(s, t\\) 为参数，取值范围为 \\([0, 1]\\)。这两个点间的距离可以表示为

$$
  r(s, t) = \|p-q\|^2
  $$

这里我们稍加变换

$$
  \begin{aligned}
  r(s, t) &= \|p_0 - q_0 + (p_1 - p_0)s - (q_1 - q_0) t\|^2\\
  &=\|p_1 - p_0\|^2 s^2 +\|q_1 - q_0\| t^2 -2 (p_1 - p_0)\cdot (q_1 - q_0) st + 2(p_0 - q_0) \cdot (p_1 - p_0)s - 2(p_0 - q_0) \cdot (q_1 - q_0) t + \|p_0 - q_0\|^2\\
  &= a s^2 + bt^2 - 2c st + 2ds - 2et + f
  \end{aligned}
  $$

其中 

$$
  \begin{aligned}
  a &= \|p_1 - p_0\|^2\\
  b &= \|q_1 - q_0\|^2\\
  c &= (p_1 - p_0)\cdot (q_1 - q_0)\\
  d &= (p_0 - q_0) \cdot (p_1 - p_0)\\
  e &= (p_0 - q_0) \cdot (q_1 - q_0)\\
  f &= \|p_0 - q_0\|^2
  \end{aligned}
$$

以上就是两条线段上任意两个点之间的距离公式，于是计算两条线段的最短距离问题就转化成了计算函数 \\(r(s, t)\\) 最小值的最优化问题

$$
  \min_{r, s \in [0, 1]} r(s, t)
  $$

现在，我们对 \\(r(s, t)\\) 求梯度，并令其为 0

$$
  \begin{aligned}
  &\nabla r = 0\\
  \Rightarrow &\left( \frac{\partial r}{\partial s}, \frac{\partial r} {\partial t}  \right) =0\\
  \Rightarrow & (2as -2ct + 2d, 2bt -2cs -2e) = 0
  \end{aligned}
  $$

求解可得

$$
  \hat{s} = -\frac{bd - ce}{ab - c^2}, \hat{t} = \frac{ae - cd} {ab - c^2}
  $$

首先容易看到，上述解没有考虑约束条件，如果最优解 \\((\hat{s}, \hat{t})\\) 位于 \\([0, 1]\\) 之外，那么就还需要考虑边界上的函数值，也就是以下几个函数的最优解

$$
  \begin{aligned}
  r(0, t) &= bt^2 -2et + f\\
  r(1, t) &=  b t^2 - (2c+2e) t + a + 2 d + f \\
  r(s, 0) &= a s^2 + 2 d s+ f\\
  r(s, 1) &= a s^2 + (2d -2c) s+ b - 2 e + f 
  \end{aligned}
  $$

同样，对它们求导，并令其等于 0，可以得到最优解

$$
  \begin{aligned}
  t_0 &= \frac{e} {b}\\
  t_1 &= \frac{c+e} {b}\\
  s_0 &= -\frac{d}{a}\\
  s_1 &= \frac{c -d} {a}
  \end{aligned}
  $$

这时，\\(t_0, t_1, s_0, s_1\\) 仍有可能落在约束区间之外，所以我们还需要计算曲线的最优边界值，即 

$$
  r(0, 0), \quad r(1, 0), \quad r(0, 1), \quad r(1, 1)
  $$

另一方面可以看到，理论最优解 \\(\hat{s}, \hat{t}\\) 存在退化情况，即当 \\(ab -c^2 =0\\) 的时候失效。根据 \\(a, b, c\\) 的定义，也就是说 

$$
  \|p_1 - p_0\|^2 \|q_1 - q_0\|^2 - ((p_1 - p_0)\cdot (q_1 - q_0))^2 = 0
  $$

根据拉格朗日恒等式

$$
    \|\mathbf{a}\|^2 \|\mathbf{b}\| - (\mathbf{a} \cdot \mathbf{b})^2 = \|\mathbf{a} \times \mathbf{b}\|
  $$

于是有 

$$
  (p_1 - p_0)\times (q_1 - q_0) = 0
  $$

两条线段的方向向量叉积为 0，即两条线段平行。在这种情况下，无法得到 \\(\hat{s}, \hat{t}\\)，只有通过边界上的函数值来求解。

总结来说，为了求解两条线段的最短距离，我们应该首先判断它们是不是平行线，如果不是，我们计算 \\(\hat{s}, \hat{t}\\)，判断它们是否满足约束，如果满足，则最优解为 \\(r(\hat{s}, \hat{t})\\)，否则，我们再计算后面的 \\(t_0, t_1, s_0, s_1\\) ，根据是否满足约束，分别计算 \\(r(0, t_0), r(1, t_1), r(s_0, 0), r(s_1, 1)\\)，对于那些不满足约束的，再计算 \\(r(0, 0),  r(1, 0), r(0, 1), r(1, 1)\\)。最后，比较所有满足约束的解，取最小值即可。而如果两条线是平行线，那么除了计算 \\(r(\hat{s}, \hat{t})\\) 外，按上述步骤也可以得到最优解。



