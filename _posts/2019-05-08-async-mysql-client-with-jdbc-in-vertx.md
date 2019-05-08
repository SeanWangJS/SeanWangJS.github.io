---
layout: post
title: 在 Vertx 中使用 jdbc 实现异步 Mysql 客户端
tags: 异步编程 vertx
---

最近在折腾 vertx，发现这是一款相当有意思的框架，不过也存在一些坑，比如在使用 vertx-mysql-postgresql-client 这个依赖的时候，postgresql client 倒是正常，但 mysql client 总是出现关于 'caching_sha2_password' 的错误，原因好像是 mysql 8 的远程连接授权方式有变动，总之就是连不上，google 了一下也没什么好的解决方法，没有办法那就只能自己用 jdbc 连了。由于只是一个临时的方案，所以写得比较简单，这里我们只实现 query 方法

```scala
def query(sql: String, handler: Handler[AsyncResult[ResultSet]]): Unit
```

由于 vertx 中一切都是异步的，所以 query 方法也不例外，我们不会让调用者阻塞等待方法返回值，而是传给 query 方法一个 handler，一旦从数据库查询完成，便使用 handler 来处理掉。这里的 ResultSet 是查询结果，但是 vertx 里面的类，和 java.sql 中的 ResultSet 不一样，所以需要一次转换

```scala
import java.sql.{ResultSet => QueryResult}

private def queryResultToResultSet(result: QueryResult): ResultSet = {

    val meta = result.getMetaData

    val names = List.range(1, meta.getColumnCount + 1)
      .map(i => meta.getColumnName(i))

    val list = new util.ArrayList[JsonArray]()

    result.next()
    do{
      val array = new JsonArray(names.map(name => result.getObject(name)).asJava)
      list.add(array)
    }while(result.next())

    new ResultSet(names.asJava, list, null)

  }
```

这里为了防止名称冲突，我们把 java.sql 中的 ResultSet 重命名成 QueryResult。然后把返回的列名用列表表示，每一行的内容用 JsonArray 表示，最终可以 new 一个 ResultSet。接下来我们看 query 方法的具体实现

```scala
def query(sql: String, handler: Handler[AsyncResult[ResultSet]]): Unit = {

    val conn = cp.getConnection
    val stat = conn.createStatement()
    try{
      val result = stat.executeQuery(sql)
      handler.handle(Future.succeededFuture[ResultSet](queryResultToResultSet(result)))
    } catch {
      case e:Exception =>
        handler.handle(Future.failedFuture(e))
    } finally {
      close(stat, conn)
    }

}

```

这里我们使用连接池获得数据库连接，然后分两种情况处理结果，如果没有发生异常，则用 Future.succeededFuture 封装结果，因为 Future 继承自 AsyncResult，所以 handler 的 handle 方法自然能接收 Future 对象。如果查询失败，则用 Future.failedFuture 封装异常，同样用 handler 处理。最终关闭连接即可。使用方法和 vertx-postgresql-mysql-client 中的客户端一样

```scala
Future.<ResultSet>future(f -> {
            String sql = ...;
            client.query(sql, f);
        })
        .setHandler(res -> {
            if(res.succeeded()){
                System.out.println(res.result().toJson().toString());
            }else {
                System.out.println("failure");
            }
        });
```