---
title: 如何一步步地实现各种深度的 ResNet
tags: 深度学习 残差网络 PyTorch
---

ResNet 通过建立短路连接，实现了一般神经网络难以模拟的恒等映射，其常用的具体架构一般有 resnet18，resnet34，resnet50，resnet101 和 resnet152 这五种，本文将从代码层面详细分析如何搭建这些结构。

首先，从最粗略的层面来看，各种 resnet 都有一个大致相同的结构，可以说是骨架中的骨架，即（conv -> bn -> relu -> maxpool -> 4 x res_layer），其简化代码如下

```python
class ResNet(nn.Module):
  def __init__(self):

    self.conv1 = nn.Conv2d()
    self.bn1 = nn.BatchNorm2d()
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d()
    self.layer1 = res_layer()
    self.layer2 = res_layer()
    self.layer3 = res_layer()
    self.layer4 = res_layer()

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    return x

```

通过调节各个 res layer 的残差块数量, 就可以实现不同深度的版本. 比如常用的 resnet 分布如下表所示

||layer1|layer2|layer3|layer4|
|--|--|--|--|--|
|resnet18|2|2|2|2|
|resnet34|3|4|6|3|
|resnet50|3|4|6|3|
|resnet101|3|4|23|3|
|resnet152|3|8|36|3|

而残差块又分为两种类型, 即 basic block 和 bottleneck block, 其结构参考下图所示

![](/resources/2021-04-26-resnet-implementation/resnet_block-type.png)

其中 basic block 由两个核大小相同的卷积层组成, 而 bottleneck block 则有 3 个卷积层, 其中第一层和第三层为 1 by 1 卷积. 另一个重要的区别是 basic block 两个卷积层的输出通道相同, 而 bottleneck block 的最后一层输出通道是前一层输出通道的 4 倍. 

我们首先来看 basic block 的简化代码

```python 
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    self.conv1 = conv3x3(in_channels, out_channels)
    self.bn1 = nn.BatchNorm2d()
    self.relu = nn.ReLU()
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d()
    if in_channels == out_channels:
      self.shortcut=nn.Sequential(
        nn.Conv2d(out_channels, out_channels),
        nn.BatchNorm2d()
      )
    
  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.shortcut:
      identity = self.shortcut(identity)
    
    x = x + identity
    x = self.relu(x)

    return x
```

在这里， 由于输入通道数可能和输出通道数不同， 所以利用 shortcut 层将输入张量的维度变换到卷积层的输出维度, 以便实现两者相加。

bottleneck block 的实现也大同小异， 只不过需要注意最后一层的通道扩展

```python
class Bottleneck(nn.Module):

  expansion = 4 ## 最后一层的输出通道扩展倍数

  def __init__(self, in_channels, out_channels):
    self.conv1 = conv1x1(in_channels, out_channels)
    self.bn1 = nn.BatchNorm2d()
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d()
    self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
    self.bn3 = nn.BatchNorm2d()
    self.relu = nn.ReLU()
    if in_channels != out_channenls * self.expansion
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * self.expansion),
        nn.BatchNorm2d(out_channels * self.expansion)
      )

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    
    x = self.conv3(x)
    x = self.bn3(x)

    if self.shortcut:
      identity = self.shortcut(x)
    
    x = x + identity
    x = self.relu(x)

    return x

```

根据原论文中的叙述, basic block 一般作为 resnet18 和 resnet34 的残差块, 其他架构使用 bottleneck block. 所以在 resnet18 和 resnet34 中，定义 res_layer 函数如下

```python
def res_layer(in_channel, out_channel, num_layers):
  layers = []

  for _ in range(num_layers):
    layer=BasicBlock(in_channel, out_channel)
    layers.append(block)
    in_channel = out_channel

  return nn.Sequential(*layers)
```

在使用的时候，只需要注意前层的 out_channel 与后层的 in_channel 匹配就行了，比如

```python
layer1 = res_layer(64, 64, 2)
layer2 = res_layer(64, 128, 2)
```

而其他 resnet 结构中，由于使用了 bottlenet block，所以需要考虑通道扩展的情况

```python
def res_layer(in_channel, out_channel, num_layers):
  layers = []

  for _ in range(num_layers):
    layer = Bottleneck(in_channel, out_channel)
    layers.append(layer)
    in_channel = 4 * out_channel
  
  return nn.Sequential(*layers)

layers1 = res_layer(64, 64, 3)
in_channel = 64 * 4
layers2 = res_layer(in_channel, 128, 4)
```

稍加修改，上面定义的两个 res_layer 可以统一成一个函数

```python
def res_layer(block, in_channel, out_channel, num_layers):
  layers = []

  for _ in range(len(num_layers)):
    layer = block(in_channel, out_channel)
    layers.append(layer)
    in_channel = block.expansion * out_channel
  
  return nn.Sequential(*layers)
```

这里使用了 block.expansion 来引用通道扩展系数，对于 BasicBlock 来说，其 expansion 等于 1。

如果我们查看 torchvision.model 自带的 resnet 结构，会一时看不出来它们各层的 res layer 的 in_channel 和 out_channel 有何规律，实际上，它们是有一定规则的，如下图所示

![](/resources/2021-04-26-resnet-implementation/resnet_res-layers.png)

可以发现具体的规则很简单

1. in_channel 从 64 开始；
2. 每个 layer 的第一个 block 的 out_channel 依次为 64，128，256，512；
3. 后层 in_channel 等于前层 out_channel。

以上的讨论为了不搞得太复杂，只涉及到了卷积网络的输入输出通道，丝毫没有提及 kernel_size, stride, padding 等量，也就是说特征图的尺寸变化规律。我们这里可以先抛出结果

1. 第一个卷积层使图片尺寸减半；
2. 第一个池化层使图片尺寸减半；
3. 对于 res_layer，如果输入输出通道一样，则不改变特征图尺寸，否则特征图尺寸减半。

总的来说，各个阶段的特征图尺寸如下所示 (其中的 e 代表 expansion)

![](/resources/2021-04-26-resnet-implementation/resnet_arch.png)

特征图尺寸的参数由 kernel_size, stride 和 padding 来控制，根据公式 

\[
  M = \frac {N + 2 p - k}{s} + 1
  \]

当 \(M = N\)，即特征图尺寸不变时，可以有 \(k = 3, p = 1, s = 1\)，当 \(M = \frac N 2\) 时，可以有 \(k = 3, p = 1, s = 2\)。

对于 basic block 来说，它有两个 3x3 卷积层，只需要在第一个卷积层控制特征图尺寸变化，第二个卷积层不改变通道数和特征图尺寸

```python
class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1=torch.nn.Conv2d(in_channels, out_channels, kernel_size =3, stride=stride, padding=1)
        self.bn1=torch.nn.BatchNorm2d(out_channels)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2=torch.nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                torch.nn.BatchNorm2d(out_channels)                
            )
        else:
            self.shortcut = None
```

这里我们额外增加了一个参数 stride，它作用于第一个卷积层和 shortcut 连接（如果需要的话）。当 stride 参数等于 2 的时候，第一个卷积层将令特征图尺寸减半，这时，如果需要 shortcut 连接，则它必须将输入特征图的尺寸变换到第二个卷积层的输出大小，也必然要求 stride=2，所以这里需要统一控制。

而对于 bottleneck block 来说，它用第二个卷积层来改变特征图大小

```python
class Bottleneck(torch.nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = torch.nn.ReLU()

        if in_channels != out_channels * self.expansion:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2(out_channels * self.expansion)
            )
        else:
            self.shortcut = None
```

同 basic block 一样，shortcut 连接层的 stride 由参数控制，与中间卷积层保持一致。

有了 BasicBlock 和 Bottleneck 的完整定义后，我们就可以完成 res_layer 的编写了。res_layer 由多个 Block 组成，我们使用第一个 Block 来实现特征图尺寸的变化，因此后续 Block 的 stride 均为 1。之所以将 stride 作为参数传入，是因为是否让第一个 block 将特征图尺寸减半由上层代码来确定

```python
def res_layer(block, in_channels, out_channels, num_layer, stride):
  layers = []

  layers.append(
    block(in_channels, out_channels, stride=stride)
  )
  in_channels = out_channels * block.expansion
  for _ in range(1, len(num_layer)):
    layer = block(in_channels, out_channels, stride=1)
    layers.append(layer)
    in_channels = block.expansion * out_channels
  
  return nn.Sequential(*layers)
```

最后，我们便可以补全各种深度的 ResNet 了

```python
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=res_layer(BasicBlock, 64, 64, num_layer=2, stride=1) ## 第一个 block 不改变特征图尺寸
        self.layer2=res_layer(BasicBlock, 64, 128, num_layer=2, stride=2)
        self.layer3=res_layer(BasicBlock, 128, 256, num_layer=2, stride=2)
        self.layer4=res_layer(BasicBlock, 256, 512, num_layer=2, stride=2)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=res_layer(Bottleneck, 64, 64, num_layer=2, stride=1) ## 第一个 block 不改变特征图尺寸
        self.layer2=res_layer(Bottleneck, 64 * 4, 128, num_layer=2, stride=2)
        self.layer3=res_layer(Bottleneck, 128 * 4, 256, num_layer=2, stride=2)
        self.layer4=res_layer(Bottleneck, 256 * 4, 512, num_layer=2, stride=2)
```

从上面的代码来看，不同深度的 ResNet 有很多重复的代码，因此我们可以让不同的部分用参数来表达

```python
class ResNet(nn.Module):
    def __init__(self, block, num_layers):
        super(ResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(block, 64, 64, num_layers[0], 1)
        self.layer2=self._make_layer(block, 64 * block.expansion, 128, num_layers[1], 2)
        self.layer3=self._make_layer(block, 128 * block.expansion, 256, num_layers[2], 2)
        self.layer4=self._make_layer(block, 256 * block.expansion, 512, num_layers[3], 2)
```

这样一来，各种深度的 ResNet 创建方法可以简化为

```python
def resnet18():
  return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
  return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
  return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
  return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
  return ResNet(Bottleneck, [3, 8, 36, 3])

```

当然，我们可以根据实际情况调整各层 res_layer 的 block 数量从而实现其他深度的 ResNet，原论文中一千多层的神经网络就是这么来的，而残差层的结构保证了它在极深的情况下不会出现严重的退化问题。

最后，附一张 ResNet-18 的详细结构图，其中白色框是特征图张量，红色框是池化层，蓝色框是卷积层，可以看到这里有 17 个卷积层，再加上分类时的全连接层，共有 18 个卷积层，也许这就是 ResNet-18 的含义吧

![](/resources/2021-04-26-resnet-implementation/resnet_resnet-18.png)

> 声明：本文的代码实现参考了 torchvision 以及 detectron2 的 resnet 实现 
