---
title: 面向组合子编程在复杂业务逻辑中的工程化实现——多源图片素材请求模块
tags: 面向组合子编程
layout: post
---

现有项目需要处理大量的图片，这些图片有的保存在文件系统，有的放在对象存储，甚至还有一部分需要实时地从网上下载。在不同的条件下可能需要处理这些图片中的一部分或全部，比如需求 A
>A: 按文件系统 - 对象存储 - 网络的顺序依次处理，在这期间如果满足中止条件，则放弃后续图片。

对于这样的需求，我们可以给出下面的伪代码

```python
// A
is_stopped=false
for image in file_system:
  is_stopped = check()
  if(is_stopped) {
    break
  }
  handle(image)

for image in ossfs:
  is_stopped=check()
  if(is_stopped){
    break
  }
  handle(image)

for image in web:
  is_stopped=check()
  if(is_stopped){
    break
  }
  handle(image)

```

接下来我们考虑需求 B

> B: 对于条件p，先采用文件系统图片，后采用网络图片，对于条件q，先采用网络图片，后采用文件系统图片。

于是相应的伪代码就是

```python
// B
if(p){
  is_stopped=false
  for image in file_system:
    is_stopped = check()
    if(is_stopped) {
      break
    }
    handle(image)

  for image in web:
    is_stopped=check()
    if(is_stopped){
      break
    }
    handle(image)

}else if(q) {
  is_stopped=false
  for image in web:
    is_stopped=check()
    if(is_stopped){
      break
    }
    handle(image)

  for image in file_system:
    is_stopped = check()
    if(is_stopped) {
      break
    }
    handle(image)
  
}

```

可以看到，如果还有其他的规则，那我们还需要再写更多类似的逻辑，冗长且繁琐。事实上，我们要做的事情本质上就是下面伪码所示的过程

```python
is_stopped=false
for image in image_source:
  is_stopped=check()
  if(is_stopped) {
    break
  }
  handle(image)
```

之所以实现需求那么麻烦是图源的选择逻辑和图片的处理逻辑高耦合的结果。按照直击本质的思路，我们定义一个接口专门提供图片

```java
public interface ImageIterator{

  Image next();

}
```

再针对不同的图源各自实现图片获取逻辑

```java
public class FSImageIterator implements ImageIterator{

  public Image next() {

    // read image from file system

  }

}

public class OSSImageIterator implements ImageIterator{

  public Image next() {
    // load image from oss
  }

}

public class WebImageIterator implements ImageIterator{

  public Image next() {
     // download image from web 
  }

}
```

上面三个类就是我们的基本组合子，接下来我们实现一个复合组合子

```java
public class SequenceImageIterator implements ImageIterator{

  private List<ImageIterator> iters;

  public Image next() {

    if(iters is empty) { //  return image end_flag if list is empty
      return end_flag;
    }

    iter=iters.get(0)
    image=iter.next()
    if(image is end_flag) { // if current image_iterator is finished, remove it and return empty image
      iters.remove(0)
      return empty image // empty image is not end_flag
    }

    return image;
  }
}
```

这里的 SequenceImageIterator 组合子实现了从多个 ImageIterator 实例中依次获取图片的逻辑，有了这个组合子，我们就能实现前面的 A，B 两个需求了。首先是统一的图片处理逻辑

```java
public void run(ImageIterator iter) {
  while(true) {

    Image image=iter.next()
    if(image is end_flag) {
      break;
    }

    if(image is empty) {
      continue;
    }

    if(check()) {
      break;
    }

    handle(image)

  }
}

```

针对需求A

```java
// A
ImageIterator iterA = new SeqenceImageIterator(
  new FSImageIterator(),
  new OSSImageIterator(),
  new WebImageIterator()
);

run(iterA);
```

针对需求B

```java
ImageIterator iterB;
switch(condition_flag) {
  case p: 
    iterB = new SeqenceImageIterator(new FSImageIterator(),new WebImageIterator());
  case q:
    iterB = new SeqenceImageIterator(new WebImageIterator(), new FSImageIterator());  
  default:
    iterB = new SeqenceImageIterator(new EmptyImageIterator());
}

run(iterB);
```

在上面的代码中，我们遇到了一个陌生的 EmptyImageIterator 类，其实它可以看作是这个组合子系统中的单位元，它在任何情况下都只返回 empty image，看似毫无意义，但是如果没有这个类，我们就还需要额外的代码来处理条件既不是p 也不是 q 的情况。

可见，针对不同得需求，我们只需要配置不同的组合子，而图片处理逻辑完全不需要修改，其优势是显而易见的。首先是图片处理功能和图片获取功能可以由不同的人员同时开发，他们之间的唯一联系就是 ImageIterator 接口，这也是一种面向接口编程。第二个显著优势是便于测试，图片处理功能的开发者完全可以事先mock ImageIterator，根据一些本地图片来测试功能代码。而它最重要的优势在于其扩展性，比如我们再增加一个需求C

> C: 有一个可变的参数 c，当它等于 p 时，使用文件系统图片，当它等于 q 时，使用网络图片

可以想象，如果采用本文最开始的朴素方法，实现逻辑是很复杂的。采用组合子框架，我们先实现一个复合组合子

```java
public class ChangableImageIterator implements ImageIterator{

  private ImageIterator iter1;
  private ImageIterator iter2;
  private CWrapper c;

  public Image next() {

    if(c.get() == p) {
      return iter1.next();
    }else if(c.get() == q) {
      return iter2.next();
    }

    return empty image;

  }

}
```

然后再组合

```java
ImageIterator iter=new ChangableImageIterator(new FSImageIterator(), new WebImageIterator);

run(iter);
```

稳。

