---
title: XGBoost 算法原理详解(2)——近似贪心优化算法
---

#### 动机

使用精确贪心算法的问题在于每一个样本都会被作为候选分割点来计算一次损失增益，这带来了高昂的时间代价，为了解决这一问题，一个很直接的想法就是使用训练集 \(\mathcal{X}\) 中的少部分样本来作为候选分割点，这里假设为 \(S = \lbrace s_1, s_2, s_3 ,..., s_l\rbrace ,s_i \in\mathcal{X}\)。有了候选分割样本之后，剩下的步骤和之前描述的精确贪心算法是类似的，不过某些细节需要特殊处理，这里简单描述一下：

考虑使用第 \(k\) 个特征作为分裂特征，此时对所有候选分割样本进行排序，得到 \(s_{1k}< s_{2k}<  s_{3k}<...< s_{lk} \)，为了计算分裂增益，首先分组计算统计量 \(G_{kv}\)和 \(H_{kv}\)，这里的 \(v = 1..l-1\)，具体计算公式为 

\[
  G_{kv} = \sum_{j\in \lbrace j \mid s_{vk} < x_{jk} \le s_{{v+1},k} \rbrace} g_j ,\quad H_{kv} = \sum_{j\in \lbrace j \mid s_{vk} < x_{jk} \le s_{{v+1},k} \rbrace} h_j 
  \]

也就是说 \(G_{kv}\) 和 \(H_{kv}\) 是第 \(v\) 和 第 \(v+1\) 个候选分割点之间的所有样本的\(g\) 和 \(h\) 的求和。后续的过程就和精确贪心算法一样了，只需要把对 \(g_i,h_i\) 的累加换成对 \(G_{kv},H_{kv}\) 的累加即可。