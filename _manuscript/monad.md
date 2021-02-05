---
title: 如何理解“Monad是自函子范畴上的幺半群”
---

##### 群的概念

设存在集合 \(G\)，以及二元运算 \(\circ: G\times G \rightarrow G\)，如果满足下述条件

* 封闭性：\(\forall x, y \in G\) 都有 \(x \circ y \in G\)
* 结合性： \(\forall x, y, z \in G \)， 都有 \(x \circ y \circ z = x \circ (y \circ z)\)
* 单位元： \(\exist e \in G\)，\(\forall x \in G\)，使得 \(x\circ e = e \circ x= x\)
* 逆元：\(\forall x \in G, \quad\exist x^{-1} \in G\) ，使得 \(x\circ x^{-1} = e\)

则构成了一个群 \((G, \circ)\)。

##### 群同构

假设有两个群 \((G_1, \circ)\)  和 \((G_2, \star)\)，以及双射映射 \(\phi: G_1 \rightarrow G_2\) ，满足以下条件

* \(\forall x, y \in G_1\)， 都有 \(\phi(x\circ y) = \phi(x)\star \phi(y)\)

那么， 群 \((G_1, \circ)\) 和 \((G_2, \star)\) 同构。

##### 半群和幺半群

在群的四个必要条件中，如果只满足封闭性和结合性，则构成半群，如果还满足单位元，则构成幺半群。

<!-- 按照一般的说法，半群是在结合性二元运算下闭合的集合构成的代数结构，这里的 **结合性** 二元运算指的是如下性质

\[(x \circ y) \circ z = x \circ (y \circ z)\]

也就是说，对于集合 \(S\)，以及映射 \(f: S\times S \rightarrow S\)，如果满足下述条件

* \(\forall x, y, z \in S\)，都有 \(f(f(x, y), z) = f(x, f(y, z))\)

则 \((S, f)\) 构成一个半群。

通常，我们在展示运算的结合性的时候，是通过比较两种计算顺序的结果值来进行的，比如乘法的结合性，导致了 \((ab)c = a(bc)\)，但从更广义的角度来看，结合性还可以通过群的同构来定义。

假设存在群 \((G, \circ)\)，根据群的定义，可以推断

* \(\exist e \in G, \forall x \in G\)，使得 \(x \circ e = x\)
* \(\forall x \in G, \exist x^{-1} \in G\)，使得 \(x \circ x^{-1} = e\) 

现在，我们构造一个集合 \(G^2 = \{(x, y) \mid x, y \in G\}\)，即集合 \(G\) 的笛卡尔积，并且定义映射 \(\otimes: G^2 \times G^2 \rightarrow G^2\)，使其拥有下述计算性质

\[\forall x, y, z, w \in G, \quad (x, y) \otimes (z,w) = (x \circ z, y \circ w)\]

可以证明装备有二元运算 \(\otimes\) 的集合 \(G^2\) 构成一个群 \((G^2, \otimes)\)，它的单位元是 \((e, e)\)，任意元素 \((x, y)\) 的逆元是 \((x^{-1}, y^{-1})\)。

接下来，我们构造另外一个集合 \(G^3 = G \times G^2\)，这是 \(G\) 和 \(G^2\) 的笛卡尔积，也可以写作 \(G \times (G \times G)\)，以及二元运算 \(\star: G^3\times G^3 \rightarrow G^3\)，且

\[
\forall x, y, z, u, v, w \in G, \quad  (x, (y, z)) \star (u, (v, w)) = (x \circ u, (y, z) \otimes (v, w))
  \]

可以证明装备有二元运算 \(\star\) 的集合 \(G \times G^2\) 仍然构成一个群 \((G\times G^2, \star)\)。

然后，我们再以另外一种组合方式构造群 \(G^3 = G^2 \times G\)，即 \((G\times G)\times G\)，以及二元运算 \(\oplus: G^3 \times G^3 \rightarrow G^3\)

\[
  \forall x, y, z, u, v, w \in G, \quad  ((x, y), z) \oplus ((u, v), w) = ((x,y)\otimes (u, v) , z \circ w)
  \]

同样可以证明装备有二元运算 \(\oplus\) 的集合 \(G^2 \times G\) 构成群 \((G^2\times G, \oplus)\)。

最后，我们将运算 \(\circ\) 定义为二元组构造函数，即 

\[
  x \circ y = (x, y)
  \]

那么运算的结合性将导致 

\[
  ((x, y), z) = (x, (y, z))
  \]

显然，\(((x, y), z) \in (G\times G)\times G, \quad (x, (y, z)) \in G \times (G \times G)\)，于是可以证明群 \((G^3, \star)\) 同构于群 \((G^3, \oplus)\)。 -->

##### 态射

态射是一种对数学对象之间关系的高层次抽象，比如集合中的函数关系 \(f: S_1 \rightarrow S_2 \)，群的同构关系 \(\phi: G_1 \rightarrow G_2\) 等等，它们的共同点是都从一个对象指向了另一个对象，因此我们用术语 **箭头** 来命名这种关系，所以我们称态射是两个对象之间的箭头。

同函数类似，态射之间也可以组合，可以看作是以态射为对象的二元运算，比如 \(\phi: G_1 \rightarrow G_2\) 和 \(\psi: G_2 \rightarrow G_3\) 之间的组合 \(\psi \circ \phi : G_1\rightarrow G_3\)。

##### 范畴

范畴是由对象、态射以及态射复合这三个概念构成的抽象数学结构，定义范畴为 \(\mathbb{C}\)，对象集合为 \(ob(\mathbb{C})\)，态射集合为 \(hom(\mathbb{C})\)，其中态射复合需要满足下述两个条件
* 封闭性：\(\forall f, g \in hom(\mathbb{C})\)，有 \(f \circ g \in hom(\mathbb{C})\)
* 结合性：\(\forall a, b, c, d\in \mathbb{C}\)，设 \(f: a \rightarrow b, g: b \rightarrow c, h: c \rightarrow d\)，有  \(f\circ g \circ h = f \circ (g \circ h)\)
* 单元性：\(\forall x \in ob(\mathbb{C}), \exist I_x: x\rightarrow x, \forall a, b \in ob(\mathbb{C}), f: a \rightarrow b\)，使得 \(I_b \circ f = f \circ I_a = f\)

幺半群可以被视为只有一个对象的范畴。设幺半群 \(\mathbb{M}\)，以及二元运算 \(\mu: \mathbb{M}\times \mathbb{M} \rightarrow \mathbb{M}\)，根据定义，\(\mu\) 满足封闭性和结合性，且 \(\mathbb{M}\) 中含有关于 \(\mu\) 的单位元 \(e\)。

再设只有一个对象 \(p\) 的范畴 \(\mathbb{C}\)，根据定义，范畴 \(\mathbb{C}\) 的态射集合都是 \(p\)到 \(p\) 的态射，即 \(\{f \mid f: p\rightarrow p\}\)，并且这些态射复合满足封闭性，结合性以及单元性。可见，幺半群的二元运算与态射复合这两个概念拥有相同的性质，若将态射看作是幺半群里面的元素，则幺半群与单对象范畴具有相同的结构，因此可以将幺半群和单对象范畴看作是同一种概念的一体两面。

|幺半群|单对象范畴|
|--|--|
|元素|态射|
|二元运算|态射复合|
|单位元|恒等态射|

现在，假设单对象范畴 \(\mathbb{C}\) 中的对象是一个幺半群 \(\mathbb{M}\)，则 \(\mathbb{C}\) 中的态射 \(f: \mathbb{M}\rightarrow \mathbb{M}\) 从群的角度来看就是自同构映射。

##### 函子

以范畴作为对象，范畴与范畴之间的态射又被称为**函子**，函子将一个范畴的对象和态射映射到另一个范畴的对象和态射。设函子 \(F: C \rightarrow C'\)，则
* \(\forall X \in C, \exist X' \in C'\)，使得 \(F(X) = X'\)
* \(\forall f: X \rightarrow Y , \exist g \in hom(D)\)，使得 \(F(f) =g: F(X) \rightarrow F(Y) \)
* 函子 \(F\) 将 \(C\) 中的单位态射映射成了 \(C'\) 中的单位态射： \(F(id_C) = id_{C'}\)
* \(\forall f, g \in (C), F(f\circ g) = F(f)\circ F(g)\)

如果一个函子将某个范畴映射到自身，即 \(F: C\rightarrow C\)，则该函子又被称为**自函子**。

##### 幺半范畴

幺半范畴是在范畴的基础上装备了幺半群结构，根据幺半群的定义，它有一个满足封闭性、结合性以及单元性的二元运算 \(\otimes\)，那么类似地，我们在范畴 \(\mathbb{C}\) 上定义二元运算 \(\otimes : \mathbb{C}\times \mathbb{C} \rightarrow \mathbb{C}\)，满足如下性质 

* 封闭性：\(\forall X, Y \in \mathbb{C}\)， 有 \(X\otimes Y \in \mathbb{C}\)
* 结合性：\(\forall X, Y, Z \in \mathbb{C}\)，有 \(X \otimes Y\otimes Z = X\otimes (Y \otimes Z)\)
* 单元性：\(\exist I \in \mathbb{C},\forall X \in \mathbb{C}\)，有 \(X \otimes I = I \otimes X = X\)

##### 自函子范畴



##### 自函子范畴上的幺半群