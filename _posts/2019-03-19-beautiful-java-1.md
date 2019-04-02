---
layout: post
title: Java 之美--过程封装
tags: Java
---

当我对 Java 还用得不是很熟练的时候，拥有 lambda 表达式的 jdk 8 发布了。于是怀着强烈的好奇心，去了解了一些 lambda 演算方面的知识，对于半路出家的我来说，仿佛来到了一个全新的世界。

在一开始，我并不知道如何在合适的地方去使用这个特性，仿佛它只是普通方法的一个时髦替代品而已，但我知道事实并非如此。直到后来读到了 SICP 这本书前面的章节，才算是有所领悟。可是，作为一个 Javaer，肯定还是想用 Java 来实现才会觉得踏实。下面我们就来看看，如何利用 lambda 表达式来抽象行为。

首先我们从最简单的数组求和开始

```java
public double sum(double[] arr) {
    return sum(arr, 0);
} 

private double sum(double[] arr, int pointer) {

    if(pointer >= arr.length) {
        return 0;
    }
    return arr[pointer] + sum(arr, pointer + 1);

}
```

这里，我们定义了一公一私两个 sum 方法，其中的公有 sum 暴露给调用者，而私有 sum 才是真正干活的地方。私有 sum 接受两个参数，第一个是需要求和的数组，第二个是指针变量。在方法中，我们首先判断指针是否越界，如果是的话，则返回 0，否则返回指针指向的当前值与数组后面值的和。

可以看到，在这个 sum 例子中。我们并没有显式的循环遍历数组的每个元素并把它们加起来。而是采用了递归的方式，每次函数调用只做一次加法，如果展开来看的话，它的过程就像下面这样

```
1. sum({1,2,3,4,5})
2. 1 + sum({2,3,4,5})
3. 1 + (2 + sum({3,4,5}))
4. 1 + (2 + (3 + sum(4,5)))
5. 1 + (2 + (3 + (4 + sum(5))))
6. 1 + (2 + (3 + 9))
7. 1 + (2 + 12)
8. 1 + 14
9. 15
```

类似于前面这个例子，下面我们需要另一种求和，即计算数组的平方和，当然也很简单

```java
public double squareSum(int[] arr) {
    return squareSum(arr, 0);
}

private double squareSum(int[] arr, int pointer) {
    if(pointer >= arr.length) {
        return 0;
    }
    return Math.pow(arr[pointer], 2) + squareSum(arr, pointer + 1);
}
```

然后，我们还想计算数组的三次方和，一样的道理

```java
public double cubicSum(int[] arr) {
    return cubicSum(arr, 0);
}

private double cubicSum(int[] arr, int pointer) {
    if(pointer >= arr.length) {
        return 0;
    }
    return Math.pow(arr[pointer], 3) + cubicSum(arr, pointer + 1);
}
```

可以看到，我们的需求仿佛源源不断，而每一次总是在类似的逻辑上修修补补，显得十分低效。回顾前面的三个例子，除了函数名以外，唯一的变化就是对当前取出来的数的处理方式，也就是下面这三种

```
arr[pointer]
Math.pow(arr[pointer], 2)
Math.pow(arr[pointer], 3)
```

第一个是什么都不做，第二个计算平方，第三个计算三次方。于是我们就想，能不能把这种行为作为参数传递给一个通用的函数呢？答案是显然的

```java

private double generalSum(int[] arr, int pointer, Function<Double, Double> f) {

  if(pointer >= arr.length) {
    return 0;
  }

  return f.apply(arr[pointer]) + generalSum(arr, pointer + 1, f);

}

```

在这里，我们定义了一个通用的求和方法，它除了接受之前的两个参数外，还需要我们提供期望的对数据的变换方式，其实这就可以用 lambda 表达式搞定，使用起来就像下面这样

```java
public double sum(int[] arr) {
    return generalSum(arr, 0, x -> x);
}

public double squareSum(int[] arr) {
    return generalSum(arr, 0, x -> x * x);
}

public double cubicSum(int[] arr) {
    return generalSum(arr, 0, x -> x * x * x);
}

```

与之前的实现相比，这一次我们的代码量缩减不少，而且还具备扩展的能力，只需要修改第三个参数，便可实现不同的处理逻辑。

接下来，为了说明这种方式可以有多灵活，我们考虑一个数值积分的例子。其实积分也是一种求和，对于一个函数来说，我们只需知道它某些点上的值，便可近似的计算它在相应区间上的积分值，以最简单的梯形公式为例

$$
    \int_a^b f(x) \mathbf{d}x \approx \frac{f(a) + f(b)} {2}(b - a) = \frac{b-a} {2}f(a) + \frac{b-a}{2} f(b)
$$

可以看到这其实就是一个最简单的求和。当然为了数值解的精确性，积分点之间的间隔，也就是 \\(b - a\\) 的值越小越好。但是如果积分区间本身就很长，那最好的方式就是把这很长的区间分割成许多小段，在每一个小段上应用梯形公式，然后把所有段上的积分加起来，这就是所谓的复化求积。

$$
    \begin{aligned}
    \int_a^b f(x) \mathbf{d}x &= \sum_{i=1}^n \int_{s_i}^{s_{i+1}} f(x)\mathbf{d}x \\ &= \sum_{i=1}^n \frac h 2 (f(s_i) + f(s_{i+1})) \\
    &= -\frac h 2(f(a) + f(b)) + h\sum_{i=1}^{n} f(s_i)
    \end{aligned}
$$

上式中的 \\(n\\) 为分割的小段数量，\\(h\\) 为小段的长度，\\(s_i\\) 为第 \\(i\\) 段的左端点，它们有如下的关系

$$
    h = \frac {b - a} n 
$$

$$
  s_i = a + h * (i - 1)
$$

然后，我们把刚才开发的求和函数用到数值积分的程序中

```java

public double integral(Function<Double, Double> f, double a, double b, int n) {

  double h = (b - a) / (double)n;
  int[] indexes = IntStream.range(1, n + 1).toArray();
  return - h / 2.0 * (f.apply(a) + f.apply(b)) +
    h * generalSum(indexes, 0, i -> f.apply(a + h * (i - 1)));

}

```

在上面的代码中，我们把分割小段的编号作为求和的数组，并在 lambda 表达式中封装了怎样通过编号计算函数值的方法，于是通过复用之前的求和函数，一段数值积分程序就轻易地表达出来了。当然，单纯的用上述递归代码还存在一些问题，比如调用层次过大会引起 Stack overflow，但是本篇文章先不关注这些。

接下来我们看一个更实用点的例子。在 Java 与数据库交互的时候，我们通常会写一些类来专门处理数据的增删查改等操作，这就是所谓的数据接入对象(DAO)。一般来讲，一个 Java pojo 类和数据库中的一个表对应，它们之间通过一个 DAO 相联系。这个 DAO 里面定义了一些方法，比如 

```java
// 向数据表中插入一条记录
public boolean insert(User user);

// 更新一条记录
public boolean update(User user);

// 删除一条记录
public boolean delete(User user);
```

这些方法可以接收一个 pojo 对象然后对数据表进行操作。下面，我们也实现一个 DAO，但是为了通用性，我们不限制传入对象的类型，而是用一个 Map 来代替，第一版的实现如下

```java
public class UntypedDAO{

  /**
  * 数据库连接提供者
  */ 
  private ConnectionProvider cp;

  /**
  * 数据表名称
  */ 
  private String tableName;

  /**
  * 数据表的字段名
  */
  private String[] fields;

  public UntypedDAO(ConnectionProvider cp, String tableName,  String[] fields) {
      this.cp = cp;
      this.tableName = tableName;
      this.fields = fields;
  }

  /**
  * 使用数据定义语言 ddl 创建数据表
  */
  public void createFromDDL(String ddl) {
      String sql = String.format("create table %1$s (%2$s) charset=utf8", tableName, ddl);
      try(Connection connection = cp.getConnection();
          Statement stat = connection.createStatement()
      ) {
          stat.executeUpdate(sql);
      }catch (SQLException e) {
          e.printStackTrace();
      }

  }

  /**
  * 插入一条记录
  */
  public void insert(Map<String, Object> record) {
    String sql = ...;

    try(Connection connection = cp.getConnection();
        Statement stat = connection.createStatement()
    ){
        stat.executeUpdate(sql);
    }    
  }

  /**
  * 更新一条记录
  */
  public void update(Map<String, Object> record) {
    String sql = ...;

    try(Connection connection = cp.getConnection();
        Statement stat = connection.createStatement();
    ) {
        stat.executeUpdate(sql);   
    }
  }

}

```

在上面的代码中，我们实现了三个方法，分别是创建数据表，插入一条数据，更新一条数据，它们都有一个共同的模式，即构造 sql 语句、获得数据库连接、执行 sql 这三个步骤。而获取连接、执行sql 明显是重复的过程，因此通常我们会将这些重复的代码单独提出来写一个方法，就像下面这样

```java
public void execute(String sql) {
    try(Connection connection = cp.getConnection();
        Statement stat = connection.createStatement();
    ) {
        stat.executeUpdate(sql);   
    }
}
```

然后其他方法只需要传入 sql 参数就可以复用这段代码了。但是这种方式存在一个很严重的问题，那就是它限制了 statement 的表达能力，因为我们知道 statement 其实有很多方法，而现在却只用到了 executeUpdate 这一种。比如要实现批量插入数据，那么我们会用到 executeBatch 方法，就像下面这样

```java
public void insert(List<Map<String, Object>> records) {

    try(Connection conn = cp.getConnection();
        Statement stat = conn.createStatement();
    ) {
        for (Map<String, Object> record : records) {
            String sql = ...;
            stat.addBatch(sql);
        }
        stat.executeBatch();
        stat.clearBatch();
    }catch (SQLException e) {
        e.printStackTrace();
    }
}
```

这就没法调用 execute(sql) 执行了，为了解决这一问题，我们可以考虑先定义 statement 的行为，并使用 lambda 表达式封装，然后再传给数据库连接代码片段，代码如下

```java

public class UntypedDAO{

  /**
  * 数据库连接提供者
  */ 
  private ConnectionProvider cp;

  /**
  * 数据表名称
  */ 
  private String tableName;

  /**
  * 数据表的字段名
  */
  private String[] fields;

  public UntypedDAO(ConnectionProvider cp, String tableName,  String[] fields) {
      this.cp = cp;
      this.tableName = tableName;
      this.fields = fields;
  }

  private void consume(Consumer<Statement> consumer) {
        try(Connection connection = cp.getConnection();
            Statement stat = connection.createStatement()
        ) {
            connection.setAutoCommit(false);
            consumer.accept(stat);
            connection.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

  /**
  * 使用数据定义语言 ddl 创建数据表
  */
  public void createFromDDL(String ddl) {
      String sql = String.format("create table %1$s (%2$s) charset=utf8", tableName, ddl);
      consume(stat -> stat.executeUpdate(sql));
  }

  /**
  * 插入一条记录
  */
  public void insert(Map<String, Object> record) {
      String sql = ...;
      consume(stat -> stat.executeUpdate(sql));
  }

  /**
  * 更新一条记录
  */
  public void update(Map<String, Object> record) {
      String sql = ...;
      consume(stat -> stat.executeUpdate(sql));
  }

  /**
  * 批量插入记录
  */
  public void insert(List<Map<String, Object>> records) {

    consume(stat -> {
      for (Map<String, Object> record : records) {
            String sql = ...;
            stat.addBatch(sql);
        }
      stat.executeBatch();
      stat.clearBatch();
    });

  }
  
}

```

上面的 consume 方法为具体的实现提供 statment 实例，并且负责连接的打开和关闭。而具体的数据库操作逻辑则被封装到 lambda 表达式。可以发现，这种实现既能复用原本重复的代码，又能非常灵活的使用 statement 提供的各种基础设施。通过这个例子，我们能看到 lambda 表达式确实能为 java 编程带来更优雅的姿势。