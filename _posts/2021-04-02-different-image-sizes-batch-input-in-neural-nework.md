---
title: 如何在神经网络中实现不同尺寸图片的批量输入
tags: 神经网络 PyTorch
---

当我们使用 VGG 或者 ResNet 做图片分类的时候，最后一层接的全连接层的输出尺寸必然是恒定值，即潜在类别数量，尽管骨架卷积网络不要求输入的图片具有相同的尺寸，但为了接上全连接层，也必须将所有输入图片都变形到相同的尺寸。所以一般来讲，在数据加载阶段会有一个 Resize 操作

```python
transform = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    ...
  ]
)
```

这种强行将所有图片变换到一致大小的操作会造成图片中物体的几何变形，在 SPPNet 的论文中，作者认为这对图片识别精度不利，于是提出了在输入图片尺寸不一致的情况下，如何输出等长特征的新结构，这就是 SPP 层，示意图如下

![](/resources/2021-04-02-different-image-sizes-batch-input-in-neural-nework/spp_layer.png)

本文的重点不是讲解SPP层原理，而是从技术角度讨论一下，为了应用类似 SPP 层这种结构，图片输入应该作什么样的调整。不难想象，如果图片是一张一张地连续输入骨架网络，生成不同尺寸的特征图，再经过 SPP 层输出相同尺寸的特征向量，这没有任何困难。但是我们知道，对于优化问题来说，批量样本输入的效果要好于单个样本，所以一般神经网络的数据加载模块都会设置 batch_size，现在的问题就在于，如果每个样本的尺寸不同，是无法作为一个整体进行张量计算的，简单来说的话，就是像下面这样的二维数组无法进行矩阵运算。

$$
  \left[
  \begin{aligned}
    a &\quad b & c\\
    e &\quad f
  \end{aligned}
  \right]
  $$

而为了让图片能够批量输入卷积网络，必然将它们变换到同一尺寸，既然缩放和裁切的方法会使提取的特征不准确，那么就只剩下填充这一条路了。为了将批量图片填充到相同的尺寸，我们首先找到高和宽的最大值，然后对每张图片进行补 0 填充，具体代码如下

```python
def preprocess(self, batched_inputs: List[torch.Tensor]):
        """
            Args:
              batch_inputs: 图片张量列表
            Return:
              padded_images: 填充后的批量图片张量
              image_sizes_orig: 原始图片尺寸信息
        """
        ## 保留原始图片尺寸
        image_sizes_orig = [[image.shape[-2], image.shape[-1]] for image in batched_inputs]
        ## 找到最大尺寸
        max_size = max([max(image_size[0], image_size[1]) for image_size in image_sizes_orig])
        
        ## 构造批量形状 (batch_size, channel, max_size, max_size)
        batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], max_size, max_size)

        padded_images = batched_inputs[0].new_full(batch_shape, 0.0)
        for padded_img, img in zip(padded_images, batched_inputs):
            h, w=img.shape[1:]
            padded_img[..., :h, :w].copy_(img)

        return padded_images, np.array(image_sizes_orig)
```

使用上述的前处理步骤，同批次所有不同尺寸的图片都被填充成了相同尺寸，没有产生扭曲变形，输入卷积网络后能够得到正常的特征，但是这些特征图是包含了大量无效信息的，也就是填充的部分，因此在传给 SPP 层之前，应该对特征图进行裁剪，需要借助原始图片尺寸，来计算特征图中的有效信息范围，如下图所示

![](/resources/2021-04-02-different-image-sizes-batch-input-in-neural-nework/feature_map_crop.png)

```python
def postprocess(self, padded_images: torch.Tensor, feature_maps: torch.Tensor, image_sizes_orig: np.array):
        """
            Args:
                padded_images: 填充后的图片张量
                feature_maps: 特征图张量
                image_size_orig: 原图尺寸
        """
        padded_size = padded_images.shape[-2:]
        feature_size=feature_maps.shape[-2:]

        ratio = feature_size[0] / float(padded_size[0])
        image_sizes_on_feature = (image_sizes_orig * ratio).astype(np.int16)
        
        crops = []
        for image, size in zip(feature_maps, image_sizes_on_feature):
            size = (self.odd(size[0]), self.odd(size[1]))
            if size[0] <= 2 or size[1] <= 2:
                continue
            crop=image[:, :size[0], :size[1]]
            crops.append(crop)

        return crops
  
```

将裁剪后的特征图输入 SPP 层，输出固定维度的特征向量，即可正常向后传递。




