#### 公式

\(x, y\) 为向量
\[
  y = softmax(x)
  \]

\[
y_j = \frac{\exp(x_j)}{\sum_i \exp(x_i)}
\]

\(X, Y\) 为矩阵
\[
  Y = softmax(X)
  \]

  \[
    Y_i = softmax(X_i)
    \]

#### pytorch 实现

```python

def softmax(X):
  X_exp = X.exp()
  rowSum = X_exp.sum(dim=1, keepdim=True)
  return X_exp / rowSum
```