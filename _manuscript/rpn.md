##### Matcher

利用 gt box 和 anchor 计算得到 iou 矩阵之后，开始生成 gt label，也就是每个 anchor 的标签值，根据原论文，正样本可以由两种规则得到，第一种是与 gt box 有最大 iou 的 anchor，也就是说，假如对于一个 gt box，出现多个 anchor 与之相交，那么取 iou 最大的 anchor 作为正样本。第二种规则是 gt box 的 iou 大于 0.7 的 anchor。而负样本则直接取与所有 gt box 的 iou 小于 0.3 的 anchor。最后剩下的 anchor 不参与 loss 计算。

在 detectron2 中，以上内容的实现代码位于 detectron2/modeling/matcher.py 中，其核心代码如下 

```python
matched_vals, matches = match_quality_matrix.max(dim=0)

match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
  low_high = (matched_vals >= low) & (matched_vals < high)
  match_labels[low_high] = l

return matches, match_labels
```

这里的 match_quality_matrix 就是 iou 矩阵，其类型为 torch.Tensor，shape 为 M by N（（M 为 gt box 的数量，也就是图片中的待检测物体数量，N 为 anchor 的数量）。首先找到 iou 矩阵每列的最大值及其索引号，也就是与每个 anchor 有最大 iou 的 gt box 编号，然后新建一个长度为 N 的数组并用 1 填充，这一步其实就是在实现正样本的第一个规则。后面这个 for 循环用到的数组 labels = [0, -1, 1]，分别表示负样本、忽略样本和正样本的标签，thresholds = [-inf, 0.3, 0.7, inf] 是默认的 iou 阈值。

在第一个循环中，l = 0，low = -inf，high=0.3，将所有与 gt box 的 iou 小于 0.3 的 anchor 设为 0，表示负样本；
在第二个循环中，l = 1，low = 0.3，high=0.7，将所有与 gt box 的 iou 大于等于 0.3 小于 0.7 的 anchor 设为 -1（在后续计算中将被忽略）；
在第三个循环中，l = 2，low = 0.7，high = inf，将所有与 gt box 的 iou 大于等于 0.7 的anchor 设为 1，表示正样本。

最后返回的两个值分别为 gt box 索引数组，以及标签数组。

##### anchors 采样

经过 Matcher 后返回了两个数组, 它们的长度都为 anchor 数量, 这里用 N 表示, 第一个数组的每个值为 与当前索引 anchor 有最大 iou 的 gt box 编号, 第二个数组的每个值为当前索引 anchor 的样本标签, 1 表示正样本, 0 表示负样本, -1 表示忽略样本. 利用第一个数组, 以及 gt box 数组, 可以生成每个 anchor 对应的 gt box, 显然, 这里每个 gt box 可以有多个 anchor 与其匹配. 

##### RPN 损失函数计算

代码位置: detectron2/modeling/proposal_generator/rpn.py

```python
def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor], 
        pred_anchor_deltas: List[torch.Tensor], 
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
```

原函数计算的是一个批次的损失和, 所以每个参数都是 List 类型的, 为了简便起见, 我们接下来的分析只考虑一张图, 所以将此函数视作

```python
def losses(
        self,
        anchors: Boxes,
        pred_objectness_logits: torch.Tensor,
        gt_labels: torch.Tensor, 
        pred_anchor_deltas: torch.Tensor, 
        gt_boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
```

参数含义如下
* anchors 是所有的锚点; 
* pred_objectness_logits 是网络输出的每个 anchor 的对象包含概率预测值;
* gt_labels 是其对应的标签;
* pred_anchor_deltas 是网络输出的锚框偏移量预测值
* gt_boxes 是每个 anchor 对应的真实边界框



经过 Mather 和采样之后得到了 gt_labels 和 gt_boxes, 其中 gt_labels 类型为 List[torch.Tensor], 是一维张量列表, gt_boxes 类型为 List[torch.Tensor] 是二维张量列表, 列表内的每个张量都代表一张图, 所以这里我们把列表展开单独对一张图进行分析. 

##### 推理阶段

