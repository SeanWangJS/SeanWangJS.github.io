---
layout: post
title: OpenCV 图像处理——描边
tags: opencv 图像处理
---

##### 实现效果

![](/resources/2020-03-28-opencv-image-process-stroke/preview.png)

##### 关键函数

```java
/**
 * Imgproc.java
 * @param src 原图
   @param dst 目标图
   @param kernel 膨胀参数
 */
public static void dilate(Mat src, Mat dst, Mat kernel)
```

##### 实现思路

1. 分离出 alpha 通道；

```java
Core.split(img, channels);
Mat alpha = channels.get(3);
```

2. 应用 dilate 函数膨胀；

```java
Mat kernel = Mat.ones(thick, thick, CvType.CV_32S)
dilate(alpha, alpha, kernel)
```

![](/resources/2020-03-28-opencv-image-process-stroke/alpha_stroke.png)
3. 为原始图片填充背景色；
这一步没有开箱即用的工具方法，需要首先生成纯色图片，然后与原图逐像素以 alpha 通道的值作为权重求和

```java
// 对于每个像素
dst[0] = img[0] * alpha + background[0] * (1 - alpha);
```

![](/resources/2020-03-28-opencv-image-process-stroke/pi_background.png)
4. 以膨胀后的 alpha 通道作为填充图的 alpha 通道

```java
Core.split(img_width_backgound, channels);
channels.add(alpha);
Core.merge(channels, dstMat);
```

![](/resources/2020-03-28-opencv-image-process-stroke/add_background.png)