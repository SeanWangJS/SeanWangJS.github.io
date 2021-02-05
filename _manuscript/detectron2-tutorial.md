---
title: Detectron2 入门
---

#### 简介

Detectron2 是 facebook 开源的，建立在 pytorch 之上，并专注于目标检测、实例分割、语义分割等图像处理任务的框架。我们知道，pytorch 是一种通用的神经网络框架，我们可以用它来加载数据集、搭建神经网络、训练模型。但事实上，由于神经网络的广泛应用性，像 pytorch 这样的框架无法对我们的实际任务做太多假设，比如 cv 的数据来自于图像集，nlp 的数据来自于语料库，作为通用框架的 pytorch 只能提供一个抽象的 Dataset 接口，而让使用者自行实现不同数据集的加载逻辑。

为了减轻广大 cver 的负担，Detectron2 针对图像处理任务的共通性，专门提供了一系列工具来简化 pytorch 的使用流程，包括数据加载、数据增强、神经网络搭建、模型训练等等。

Detetron2 的安装和普通的 python 库类似，根据自己的操作系统和 pytorch 以及 cuda 版本选择对应的预编译版本就行了，需要注意的是目前 detectron2 不支持 windows 系统，其他就不再赘述了。

#### 数据集定义和注册

神经网络训练都遵循这样一个套路，即定义数据集、定义数据加载器、定义模型、训练、存储模型。所以我们首先来看一看 detectron2 中的数据集定义模块。在 pytorch 里面，用户数据集需要实现 Dataset 类才能被加载，其中 \_\_getitem__ 方法定义了加载方法以及返回项目。但在 Detectron2 中，数据读取和返回的内容是固定的，读取的当然就是图片，而返回项目则是预先约定的字典列表，其中每个字典都是一张图片包含的信息，Detectron2 定义了自己的标准数据集字典格式(Standard Datasets Dicts)，大致内容如下（更详细的信息请参考[文档](https://detectron2.readthedocs.io/tutorials/datasets.html)）

```javascript
{
  "file_name": ,            // 文件路径
  "height": ,               // 图片高度
  "width": ,                // 图片宽度
  "image_id": ,             // 图片id
  "annotations": [          
    {
      "bbox": ,             // 边界框坐标
      "bbox_mode": ,        // 边界框坐标模式， xyxy 或者 xywh
      "category_id": ,      // 目标类别
      "segmentation": ,     
      "keypoints": 
    }

  ],
  "sem_seg_file_name": ,
  "pan_seg_file_name": ,
  "segments_info": [
    {
      "id": ,
      "category_id": 
    }
  ]
}
```

当然，Detectron2 对数据集格式的约定很宽松，我们可以按实际情况修改部分内容，比如单纯的目标检测任务就不用 segmentation 相关的字段。定义了数据集之后，下一步就是读取实际的训练文件并返回上述约定格式的对象，对于那些相当出名的通用数据集，比如 COCO、PASCAL_VOC 等，Detectron2 还为它们都逐个实现了一个加载函数，比如这里的[pascal_voc.py](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/pascal_voc.py)。这样一来，就给了我们两个选项，一是在标注数据的时候就按照通用的格式存储，这样可以复用框架现成的代码，二是自己定义标注格式，但要自己实现加载代码。

```python
from detectron2.data.datasets.pascal_voc import load_voc_instances

dataset: List[Dict] = load_voc_instances("/opt/dataset/VOC2007", "train", ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
```

上面的代码演示了内置的读取 PascalVOC 格式数据集的方法。

Detectron2 封装了训练过程，因此数据集读取方式可以作为一种配置参数传入。为了实现这一特性，Detectron2 将读取方法注册成全局可访问函数，然后将注册的 key 传给配置。具体使用方法如下 

```python
from detectron2.data import DatasetCatalog

DatasetCatalog.register("pascal_voc_train", lambda: load_voc_instances(...))
```

上面的 lambda 表达式定义了具体的数据集读取行为，需要注意，在用 DatasetCatalog 注册函数时，被注册函数必须是无参的。然后将上面注册的数据集名称传入配置对象，就相当于告知 Detectron2 封装的训练器该加载哪些数据。

```python
cfg.DATASETS.TRAIN = ("pascal_voc_train", )
```

#### 数据加载器和数据增强

同 pytorch 中的过程类似，定义了数据集(dataset) 后，还需要一个数据集加载器(dataloader) 才能得到可以输入神经网络模型的张量数据。在 pytorch 中，我们通常是用下面的方法来获取数据加载器

```python
dataset = ...
batch_size = ...
sampler = ...
num_workers= ...
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers = num_workers)
```

而 Detectron2 则封装了上述方法，提供了两个工具方法 build_detection_train_loader 和 build_detection_test_loader 来分别加载训练数据和测试数据，并额外增加了一个数据增强功能的入口，其中 build_detection_train_loader 的函数接口为

```python 
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
)
```

此方法的返回类型正是 DataLoader。这里的 mapper 参数，其实就是数据增强方法，如果我们查看 build_detection_train_loader 的实现

```python
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
```

可以发现，dataset 被 MapDataset 装饰，这里的 MapDataset 是 Dataset 的派生类，它的 \__getitem__ 方法从原本的 dataset 中取出数据后，通过 mapper 进行了一次转换，从而实现数据增强。

#### 配置神经网络

Detectron2 封装了一些常用的神经网络模型，比如针对目标检测任务的 faster_rcnn，针对实例分割的 mask_rcnn 等。为了获取这些内置的网络模型，Detectron2 使用配置的方式来指定具体的网络以及训练时的超参数。

```python
from detectron2.config import get_cfg

cfg = get_cfg()
```

这里的 get_cfg 方法是获取配置对象的入口。在得到配置对象之后，我们可以设置各种参数

```python
## 设置神经网络模型
config_file=model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
cfg.merge_from_file(config_file)

## 设置训练集名称，也就是前面注册的数据集加载 lambda 表达式
cfg.DATASETS.TRAIN = ("pascal_voc_train", )

## 设置验证集名称
cfg.DATASETS.TEST = ("val_set")

## 设置每个GPU设备的 batch_size
cfg.SOLVER.IMS_PER_BATCH=2

## 设置初始学习速率
cfg.SOLVER.BASE_LR = 0.00025

## 设置迭代次数
cfg.SOLVER.MAX_ITER=300

## 设置类别数量
cfg.MODEL.CLASSES = 1

## 设置初始网络权重
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
```

当然，cfg 对象的参数还有很多，具体可见[官方文档](https://detectron2.readthedocs.io/modules/config.html#detectron2.config.CfgNode)。