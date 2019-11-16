---
title: opencv 图像处理之污点修复
tags: opencv java 图像处理
---

Photoshop 里面有一个很实用的工具，叫做污点修复画笔，只需轻轻一点，就能将图片上的小瑕疵抹掉。这一神奇的功能我们可以使用 opencv 的 inpaint 函数轻松实现。

```java
//Class: Photo
public static void inpaint(Mat src, Mat inpaintMask, Mat dst, double inpaintRadius, int flags);
```

参数解释如下：
> src: 源图像 mat
> inpaintMask: 区域蒙版，其大小和 src 相同，8位1通道类型，无关的像素值为0，需要被修复的像素值不为 0。
> dst: 输出图像 mat
> inpaintRadius: 修复像素时被算法考虑的相关像素半径
> flags: 具体的修复算法，可选 Photo.INPAINT_NS 或者 Photo.INPAINT_TELEA

