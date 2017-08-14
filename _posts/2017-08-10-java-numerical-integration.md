---
layout: default
---

## 数值积分 Java 实现

之前在SICP上看到一个很厉害的东西，考虑三个问题：
1. 计算从 a 到 b 间所有整数的和
2. 计算从 a 到 b 间所有整数的三次方之和
3. 求解序列 
$$
\frac{1}{1 \cdot 3} + \frac{1}{5 \cdot 7} + \frac{1}{9 \cdot 11} + ....
$$

下面的三段代码可以分别解决这三个问题

```scheme
(define (sum-integers a b)
  (cond ((> a b) 0)
        (else (+ a (sum-integers (+ a 1) b)))))
		
(define (sum-cubes a b)
  (define (cube k)
  (* k k k))
  (cond ((> a b) 0)
        (else (+ (cube a) (sum-cubes (+ a 1) b))))
  )

(define (pi-sum a b)
  (cond ((> a b) 0)
        (else (+ (/ 1.0 (* a (+ a 2))) (pi-sum (+ a 4) b)))))
```

虽然解决的是不同的问题，但是在代码中却存在相似的结构，即如下所示

```scheme
(define (<func-name> a b)
  (cond ((> a b) 0)
        (else (+ (<func> a) (<func-name> (<inc> a) b)))))
```

也就是说，同样是求和过程，但是不同的细节操作可以表达出不同的算法需求，如果我们将特定的操作(比如func、inc) 这些函数作为参数传入高阶函数，那么就能得到更具表达力的过程。

```scheme
(define (sum func inc a b)
  (cond ((> a b) 0)
        (else (+ (func a) (sum func inc (inc a) b)))))
```

使用 sum 函数作为基本构件，可以重写前面的问题，这里以第三个为例

```scheme
(define (pi-sum2 a b)
  (sum (lambda (x) (+ (/ 1.0 (* x (+ x 2)))))
       (lambda (x) (+ x 4))
       a
       b))
```

所以这个 sum 函数就很厉害了，它可以定义一大类求和问题。考虑到许多数值积分其实也都是求和过程，例如复化梯形积分公式

$$
\int_a^b f(x)\mathrm{d}x = \frac h 2 f(a) + \frac h 2 f(b) + h * \sum_{i=1}^{n-1}f(x_i),\quad h = \frac{b - a}{n}, \quad x_i = a + i h
$$

下面为使用 sum 函数的代码实现

```scheme
(define (integral2 f a b n)
  (define h (/ (- b a) n))
  (define (g i)
    (f (+ a (* i h))))
  (+ (* (/ h 2) (g 0))
     (* (/ h 2) (g n))
     (* h (sum g
          (lambda (x) (+ x 1))
          1
          (- n 1))))
  )
```
注意在上面的代码中，为了方便，使用了 $$g(i) = f(x_i) = f(a + i h)$$ 
### Java 版实现

既然从 version 8 开始，Java 也支持 lambda 表达式了，那么使用 Java 写这样的函数也不是难事。首先是定义 sum 函数

```java
double sum(Function<Integer, Double> func, Function<Integer, Integer> inc, 
  int a, int b) {
        if(a > b)
            return 0;
        return func.apply(a) + sum(func, inc, inc.apply(a), b);
    }
```

为了求解效率，下面我们使用4阶的复化 Newton-Cotes 公式来求积分。有的书上直接抛出了最终的公式形态，我觉得不太适用于代码编写，所以在此之前先看看基本的 Newton-Cotes 公式

$$
\int_{t}^{t + h} f(x)\mathrm{d}x = \sum_{i = 0}^m C_i^{(m)} f(x_i)
$$

其中 $$x_i = t + i * d,\quad d = \frac h m$$， $$C_i^{m}$$ 是 m 阶 Newton-Cotes 公式的系数，若 $$m =4$$，则有

$$C_i = \frac 7 {90} , \frac{32}{90} ,\frac{12}{90}, \frac{32}{90} , \frac{7}{90}, \quad i = 0,1,2,3,4$$
 
于是

$$
\int_{t}^{t + h} f(x)\mathrm{d}x = \frac h {90} [7 f(t) + 32 f(t + d) + 12 f(t + 2d) + 32 f(t + 3d) + 7 f(t + 4d)]
$$
 
由于基本的积分公式只在很小的区间上才能得到较精确的值。对于大段积分区间，更好的处理方式是将其分割为许多小段，然后在每个小段上应用上面的公式，这就产生了复化求积公式

$$
\int_a^b f(x)\mathrm{d}x = \sum_{i=0}^{n-1} \int_{x_i}^{x_i +h} f(x)\mathrm{d}x 
$$

然后定义

$$
g(i) = \int_{x_i}^{x_i +h} f(x)\mathrm{d}x 
$$

于是有

$$
\int_a^b f(x)\mathrm{d}x = \sum_{i=0}^{n-1} g(i)
$$

这种简单的形式非常利于代码实现，下面给出了 Java 代码

```java
public double integral(Function<Double, Double> f, 
  double a, double b, int n) {
        double h = (b - a) / (double) n;
        double d = h / 4.0;
        Function<Integer, Double> g = i -> h / 90.0 
		* (7 * (f.apply(a + i * h) + f.apply(a + i * h + 4 * d))
                + 32 * (f.apply(a + i * h + d) + f.apply(a + i * h + 3 * d))
                + 12 * f.apply(a + i * h + 2 * d));
        return sum(g, x -> x+1, 0, n - 1);
    }
```


















