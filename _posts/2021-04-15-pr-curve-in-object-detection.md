---
title: 对目标检测模型 PR 曲线的原理分析和实现
tags: 目标检测 PR曲线
---
PR曲线其实就是以准确率(Precise)为横轴，召回率(Recall)为纵轴的曲线，定义很简单，但是我们立即会产生一个疑问，那就是对于一个确定的模型，怎么准确率咋还能变化呢？所以要理解 PR 曲线，首先需要知道准确率和召回率是怎么变化的。

我们知道，准确率指的是准确检测的物体占总检测物体的比例（需要注意的是，我们的分析都是建立在单一类别之上的），比如一次测试过程，我们检测到了 N 个物体，但实际上只有 P 个物体是被准确检测到的，其他都属于误检，那么准确率就是 P / N。而召回率则是准确检测的物体占总的物体的比例，还是同样的测试，假如这批样本中共有 M 个物体，那么召回率就等于 P / M。准确率的分母是检测到的物体总数量，召回率的分母是实际物体总数量，所以准确率又叫查准率，召回率又叫查全率。

一般目标检测模型的输出会附带一个自信度(confidence)，介于 0 到 1 之间，自信度越大表示其检测准确的概率越大，那么，我们就需要抉择了，如果一个检测实例的自信度等于 0.6，是否认为其检测准确？0.5 呢？ 0.4 呢？显然，这是一个不好回答的问题，0.8 太严格，0.2 太宽松，都不好。于是，我们只有劳烦一下，在每个自信度上分析一次。显然在每个自信度级别上计算的准确率和召回率是不一样的，这就是为什么会有一根曲线的原因了。

PR曲线的原理就是这样，接下来我们考虑具体的实现问题。首先，我们需要一张表来记录检测结果，其 header 如下

|gt_id|pred_id|iou|confidence|
|--|--|--|--|--|--|

依次为 Ground Truth id、预测id、交并比以及自信度，对于一张图片来说，其物体实例有 M 个，检测结果有 N 个，为了将 pred 和 gt 对应起来，我们需要计算它们的 IOU 矩阵，这是一个 M by N 的矩阵，其第 i 行第 j 列的值是第 i 个gt 与第 j 个 pred 的交并比，然后我们需要找到每列的最大值，也就是每个 pred 匹配上的 gt，但是这里存在一个问题，有可能同一个 gt 被多个 pred 匹配上了，这时就需要对 IOU 排序，以最大的 IOU 作为正样本。以 detectron2 提供的工具函数 pairwise_iou 为例，代码如下

```python
pred_boxes: Boxes = ...
gt_boxes: Boxes = ...
iou_matrix: torch.Tensor = pairewise_iou(pred_boxes, gt_boxes)
ious, gt_ids = iou_matrix(dim=0)
```

为了构建上表，还需要生成 pred_id 以及 confidence，其中 pred_id 可以按预测实例的顺序生成，为了区分图片，可以在前缀上加上图片id

```python
df_i = pd.DataFrame({
        "gt_id": [f"{image_id}_{i}" for i in gt_ids.cpu().numpy().tolist()],
        "predict_id": [f"{image_id}_{i}" for i in range(len(gt_ids))],
        "iou": ious.cpu().numpy().tolist(),
        "confidence": output["instances"].scores.cpu().numpy().tolist()
    }
```

当然，这是一张图片的结果，对于多张图片，还需要拼接连续

```python
df=pd.concat((df, df_i), axis=0)
```

在得到类似于下面这张表之后，接下来开始划分正预测，即 True Positive。

|index|gt_id|predict_id|iou|confidence|
|--|--|--|--|--|--|
|0|0006990_1|0006990_0|0.922708|0.999956|
|1|0006990_5|0006990_1|0.880718|0.999952|
|2|0006990_14|0006990_2|0.986290|0.999944|
|3|0006990_26|0006990_3|0.942946|0.999930|
|4|0006990_24|0006990_4|0.926946|0.999927|

首先，需要确定一个 IOU 阈值，当 gt 与 pred 的 iou 大于该值的时候，则认为此预测为 TP，比如，将其设为 0.5

```python
df["TP"] = df["confidence"] > 0.5
```

然后，我们来解决同一个 gt 匹配多个 pred 的问题，解决方法是将最大 IOU 的 pred 设为 TP，从编程的角度来看，我们首先按 gt_id 分组，然后找到每组的最大 IOU，这时对应的 predict_id 就是最佳匹配，将它们设为 TP

```python
max_iou=df[["gt_id", "iou"]].groupby("gt_id").max()
best_match_df=df[["gt_id", "iou", "predict_id"]].merge(max_iou, how="inner", on = ["gt_id", "iou"])
best_match_df=best_match_df.drop(columns=["iou"])
best_match_df["best_match"]=pd.Series([True for _ in range(len(best_match_df))])
```

得到下表

|index|gt_id|predict_id|best_match|
|--|--|--|--|
|0|0006990_1|0006990_0|True|
|1|0006990_5|0006990_1|True|
|2|0006990_14|0006990_2|True|
|3|0006990_26|0006990_3|True|
|4|0006990_24|0006990_4|True|

然后再以 gt_id 和 predict_id 为key，merge 回 df 表，这里采用 outer merge，让非最佳匹配对的 best_match 为 False

```python
df = df.merge(best_match_df, how="outer", on=["gt_id", "predict_id"]).fillna(False)
```

此时某些行的 TP 列和 best_match 列可能不同，也就是说，某些 pred 的 IOU 虽然大于阈值，但是还有比它更好的匹配框，所以应该将其 TP 设为 False，表现为 TP 列 best_match 列的逻辑与关系运算

```python
df["TP"] = df["TP"] & df["best_match"]
```

这样我们就完成了对预测数据的处理，接下来开始计算每个 confidence 下的 Precise 和 Recall。首先按 confidence 从大到小排序，并生成每个 confidence 作为阈值的情况下的 TP 数量和 pred 数量，也就是前面提到的 P 和 N， 而 gt 数量 N 是常数。

```python
df = df.sort_values(by="confidence", ascending=False)

N = len(pd.unique(df["gt_id"]))

M = pd.Series([i + 1 for i in range(len(df))])

P = pd.Series([tp[0:i+1].sum() for i in range(len(df))])

precise = P / M
recall = P / N
```

所得的 PR 曲线一般如下所示，其特点是，准确率越高，召回率就越低，这是显而易见的，当 confidence 阈值设置得很大的时候，大部分正预测都将是准确的，而因为门槛太高，导致很少有 pred 达到要求，所以占总的 gt 数量极少，于是召回率就低。

![](/resources/2021-04-15-pr-curve-in-object-detection/pr_curve.png)