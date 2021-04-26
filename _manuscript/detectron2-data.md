---
title: Detectron2 代码研究——数据篇
tags: Detectron2 计算机视觉 深度学习
---

作为一种通用的 cv 深度学习框架，Detectron2 遵循约定大于配置的理念，定义了一套可以用于目标检测、实例分割、关键点检测等视觉任务的数据格式，大致如下

```json
{
  "file_name": ,
  "height": ,
  "width": ,
  "image_id": ,
  "annotations": [
    {
      "bbox": [],
      "bbox_mode": ,
      "category_id": ,
      "segmentations": [[]],
      "keypoints": [],
      "iscrowd": 
    }
  ]
}
```

其中 file_name, height, width, image_id 等字段对于所有任务都是必要的，annotations 里面的字段可以看具体的任务类型进行取舍，详细的解释可以参见[官方文档](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts)。我们这里要说明的是，如果想利用 Detectron2 里面的工具对数据进行处理，就需要在加载数据的时候输出上述标准格式。

总体来说，Detectron2 中的数据需要经过以下几个步骤：
1. 利用数据读取函数，生成标准数据格式字典列表 (dataset)；
2. 利用数据字典列表构造数据加载器(dataloader)，这一步还需要传入 DatasetMapper，用于读取图片并实施数据增强操作；
3. 从数据加载器中循环取出批量数据，并输入模型进行训练或者推理。

可以看到，这和 pytorch 中的训练模式是一样的，只不过 Detectron2 专用于解决图像领域的问题，所以可以针对图像数据集预先定义大量专用的处理方法。下面我们介绍一些比较有用的模块

##### 数据集注册

对于一些大家耳熟能详的数据集，比如 PascalVOC、MS COCO 等，Detectron2 为我们提供了内建的数据集加载方法，以 PascalVOC 为例，首先我们需要将自己的数据存储成 PascalVOC 要求的格式，然后注册即可，十分方便

```python
from detectron2.data.datasets.pascal_voc import 

register_pascal_voc(name="custom_voc_train", dirname="/path/to/my_data", split="train", year=2007, class_names = ["cat1", "cat2"])
```

这里的参数中， name 是我们注册的数据集名称，在框架的执行代码中，将通过这个名称拿到我们注册的数据集实例，比如 

```python
dataset = DatasetCatalog.get("custom_voc_train")
```

返回的结果就是前面我们提到的标准数据字典列表。参数 dirname 是我们的数据集所在文件目录，split 指的是 "train" 还是 "val"，year 是 pascal voc 的年份，对于我们自己的数据没什么用，class_names 当然指的是数据中所有类别的名称列表。

类似的还有专门注册 COCO 格式的函数

```python
from detectron2.data.datasets.coco import register_coco_instances

register_coco_instances(name="custom_coco_train", metadata={}, json_file="/path/to/coco_text_train.json", image_root="/path/to/images")
```

要使用这些工具的前提就是将数据存储成对应的格式，否则就只有自己写加载函数。然后使用 DatasetCatalog 和 MetadataCatalog 进行注册，比如

```python
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

def load_data():
  ...
  return ... # List[dict]

DatasetCatalog.register("custom_data_train", lambda: load_data())
MetadataCatalog.get("custom_data_train").set(thing_classes=["cat1", "cat2"])
```

可以看到，DatasetCatalog 其实就是一个字典类型的对象，它接收 key 为 str，value 为 lambda 表达式的键值对。

##### 训练过程中的数据加载流程梳理

1、读取数据字典列表

只要是被注册过的数据集，在代码的任何位置都可以通过 DatasetCatalog 的 get 方法进行读取

```python
dataset = DatasetCatalog.get(dataset_name)
```

除此之外，我们还可以通过 Detectron2 提供的工具函数 get_detection_dataset_dicts 读取多个数据集且合并成单个数据集

```python
# /detectron2/data/build.py
def get_detection_dataset_dicts(dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None)
```

这里的 dataset_names 如果是一个 str，将加载此名称的数据集，而如果是一个 str 列表，则会读取多个数据集且合并到单个数据集，filter_empty 参数用于指定是否过滤掉 annotations 为空的数据，min_keypoints 只适用于关键点检测，表示关键点数量小于该值的样本将被过滤掉，proposal_files 指的是预先生成的 Region Proposal 文件位置，对于 RPN 网络来说这是不需要的。

2、数据加载器

在 detectron2 中，数据加载器是对 pytorch DataLoader 的简单封装，为了讲清楚这一部分，我们先来分析 pytorch 的数据加载机制。下面给出 DataLoader 类的初始化方法

```python
# torch.util.data.DataLoader
def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None)
```

首先是 dataset 参数，按官方文档的说法，它的类型可以是 map-style 的或者 iterable-style 的，如果它是 iterable-stype 的，那么需要实现自 IterableDataset，而如果是 map-style 的，则需要保证调用 dataset[i] 时能够返回某个元素，调用 len(dataset) 时能够返回数据集长度，也就是说，实现了 \__len__() 和 \__getitem__() 这两个方法的类。

batch_size 没什么好说的，就是批量大小。shuffle 指的是在从 dataset 中拉取数据的时候，是否采用乱序的方式，需要注意的是，如果 dataset 是 iterable-style 的，shuffle=True 会 raise exception，因为迭代器没法乱序。

sampler 是数据取样器，它是 torch.utils.data.Sampler 的子类，定义了以何种顺序读取 dataset 里面的数据，比如顺序读取 (SequentialSampler)，随机读取 (RandomSampler)等等，同 batch_size 类似，如果 dataset 是 iterable-style 的，那么 sampler 必须为 None，否则会 raise exception，因为迭代器只有一种顺序。另外，在 shuffle 为 True 的情况下，如果 sampler 不为 None，也会 raise exception，因为，同一个数据加载器不可能同时以两种顺序返回数据。

batch_sampler 是批量抽样器，它需要接收一个 sampler 对象作为其内部成员，并按指定的批量大小抽取数据。 类似的道理，batch_sampler 不能用于 iterable-style 的 dataset，不能指定 shuffle=True，不能同时设置 sampler。另外，如果指定了 batch_sampler，那么不能再指定 batch_size，因为 batch_sampler 已经规定了批量大小。

num_workers 指定了数据加载进程数量，如果等于 0，那么将默认在主进程中加载数据。

collate_fn 是一个 callable 对象，定义了如何根据一批样本构建小批量张量的方法，举例来说，下面这个函数将 Tensor 列表组合成一个维度更高的张量来表示整批数据

```python
def collate_fn(batch: List[torch.Tensor]):
  return torch.stack(batch, 0)
```

这是 pytorch 默认的小批量张量构造方法，它要求每个批次中的所有张量维度都相同，这意味着每张图片都需要被 resize 到固定尺寸。如果我们不想这样做，那么就需要传入一个自定义的 collate_fn，比如

```python
def collate_fn(batch: List[torch.Tensor]):
  return batch
```

pin_memory 主要和张量在内存与显存中的移动有关，我们知道，Tensor 对象一开始被分配到内存中，如果主内存空间不足，那么将采用虚拟内存把数据交换到磁盘上，这样一来，当 Tensor 被发送到 cuda 中进行计算的时候，必须先从磁盘中读取数据，从而导致时延。为了避免主内存把张量数据交换到虚拟内存中，可以将 pin_memory 设为 True，显然，这会导致内存占用量增加，所以默认情况下这一功能是关闭的。

drop_last 选项设置是否在最后一个批次不够的时候丢弃该批次图片，因为总的样本数量一般除不尽 batch_size，这就导致最后一个批次的样本数量小于 batch_size，当设置 drop_last 为 True 时，将丢弃这个批次的样本。

以上就是对 DataLoader 主要参数的解释，可以看到这里的灵活性太高了，反而容易搞得人无从下手的感觉，在 detectron2 中，对它进行了一些简化处理，核心函数为 /detectron2/data/build.py 文件里面的 build_batch_data_loader

```python
def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0
)
```

其中 dataset 和 sampler 同 DataLoader 中的要求类似，total_batch_size 等于 GPU 数量 * batch_size，如果只有一个GPU，那么它的含义与 batch_size 相同。aspect_ratio_grouping 是一个新的参数，它用来设置是否将高宽比接近的图片尽量放在一起返回给调用端，要理解这里的用意就需要先了解 detectron2 对同批次不同尺寸的图片是如何处理的，我们知道，DataLoader 的默认 collate_fn 会将一个批次的所有 tensor 进行 stack 成一个大的 tensor，但是，当这些 tensor 的维度不尽相同的时候，stack 就会失败。而 detectron2 使用的 collete_fn 则不进行 stack，而是简单地将 tensor 合并成列表返回，但是使用模型进行 forward 的时候，这些张量总得变换成一个大的张量才行，detectron2 给出的方法是在传给模型之前，先将这些张量使用零填充到相同的尺寸。这时，为了减小填充宽度，需要这些图片的高宽比尽量相似，所以出现了 aspect_ratio_grouping 这个参数来进行控制。

针对 train 和 eval 阶段的数据差异性，detectron2 在 build_batch_data_loader 的基础上还提供了两个工具函数 build_detection_train_loader 和 build_detection_test_loader 用于加载训练数据和测试数据

```python
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )

@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
```

可以看到，build_detection_train_loader 主要做了几件事情，首先是把 list 类型的 dataset 封装成 Dataset 类型的，然后再加入 mapper 进一步封装成 MapDataset (如果mapper 不为None 的话)，之后再处理 sampler，最后调用 build_batch_data_loader 返回 DataLoader，build_detection_test_loader 也大致类似，唯一不同的是 sampler 的处理方法。

从这两个方法的 @configuature 注解可以知道，它们支持通过配置的方式生成，具体看 _train_loader_from_config 和 _test_loader_from_config 两个函数

```python 
def _train_loader_from_config(cfg, *, mapper=None, dataset=None, sampler=None)

def _test_loader_from_config(cfg, dataset_name, mapper=None)
```

也就是说，如果我们不想自己构造 build_detection_train/test_loader 所需的这几个参数，可以直接传入 cfg，它会自动将任务分派给 _train/test_loader_from_config 方法，比如

```python
cfg = get_cfg()
cfg.DATASETS.TRAIN = ("voc_2007_train", )
train_loader = build_detection_train_loader(cfg)
```

当然，如果想自定义 mapper 和 sampler，也可以传进去。

3、数据增强

数据增强是为了扩展数据的多样性，从而使得模型具有更好的泛化能力，特别是在训练数据不足的情况下。
