---
title: 面向组合子编程的应用实践—-影视节目名称匹配
tags: 面向组合子编程
---

最近有个项目，需要根据用户输入的影视节目名称来查询我们数据库里面对应的节目，数据表结构如下所示，用户事先是不知道我们这边数据库里的节目名称的，比如 "xxxx 第3季"，不同的用户可能会输入不同的名称，例如 “xxxx3”、“xxxx：3”、“xxxx 第三季” 等等，这是很正常的现象，但却为我们的查询工作带来了困难。

|id|name|
|--|--|
|1|abdc|
|2|cbsd|
|..|..|

在这里，我想记录一下我的解决方案。我们把问题简化一下，分几个层次来讨论。首先，假如用户的输入与数据表的记录名称一模一样，那么只需要一句 sql 就搞定了。接下来，如果用户输入的名称始终被数据表中相应节目的名称完全包含，比如用户输入 "abc"，而表中的记录为 "ab,c1"，这时我们没法通过 sql 来查询，甚至连缩小范围都做不到，对于这种情况，我们可以建立所有节目的倒排索引表，让每一个字符都对应一组id，然后把用户的输入拆分成字符组，逐一查询，再把得到的 id 列表做一个交集，得到一个候选的节目集合，利用这个候选集，我们可以做更精细的匹配。

|token|id_list|
|--|--|
|a|1,3,10,12,..|
|b|1,2,3,4,21...|
|..|..|

现在的问题是，用户的输入是我们没法预设的，按上述方法查询倒排索引表可能得到一个空集。所以，我们首先应该对用户的输入进行一些预处理，一般来讲，一档节目的名称可以分解成主标题（必）、副标题（选）、季数（选）以及一些连接符号。一般来讲，匹配上主标题是最关键的，剩下的关键字即便无法匹配，我们也可以作为候选集返回给用户自行选择。

那么如何提取用户输入节目名称的主标题呢？这就没有定论了，根据经验，用户输入主要有下面几种形式

1. "xxxx: xxx"、"xxxx：xxx"、"xxxx-xxx"
2. "xxxx2: xxx"、"xxxx2：xxx"、"xxxx3-xxx"
3. "xxxx2"
4. "xxxx 第x部"、 "xxxx 第x季"
5. xxxx(xxx)
6. xxxx（xxx）
7. ...

其中每一种情况都可以用一个正则表达式来匹配

1. "(.+)[: |：|-]"
2. "(.+)\\d{1}[: |：|-]"
3. "(.+)\d{1}$"
4. "(.+)第.+[季|部]"
5. "(.+)\\(.+\\)"
6. "(.+)（.+）"

当然我们不可能枚举出每一种模式，所以如何方便对模式进行扩展是我们在写程序的时候需要仔细考虑的问题。下面我们开始讨论如何实现提取主标题的功能，首先定义接口

```scala
trait Matcher {

  def tryMatch(title: String): Option[String]

}
```

这里的 Matcher 特质有一个 tryMatch 方法负责提取字符串的部分内容，它是一个比提取主标题这一功能更宽泛的能力，可以在实现类（这里我们称之为组合子）中指定提取任意我们感兴趣的内容，包括主标题。对于前面我们提到的用正则表达式来提取主标题，对应的 Matcher 实现如下

```scala
class RegexMatcher(regex: String, group: Int) extends Matcher {

  private val pattern: Pattern = Pattern.compile(regex)

  override def tryMatch(title: String): Option[String] = {

    pattern.matcher(title) match {
      case matcher if matcher.find() => Some(matcher.group(group).trim)
      case matcher if !matcher.find() => None
    }

  }
}
```

每一个正则表达式都可以定义一个 RegexMatcher 实例，来提取对应模式的主标题。很显然这些模式都是互斥的，我们需要逐一尝试才有可能找到最终的结果，为了抽象这一特点，我们实现一个 AnyMatcher 类，它由多个 Matcher 实例组合得到，当其中一个 Matcher 提取到有效内容时立即返回该内容

```scala
class AnyMatcher(matchers: Matcher*) extends Matcher {

  override def tryMatch(title: String): Option[String] = matchers.view.map(_.tryMatch(title)).find(_.nonEmpty).flatten

}
```

测试一下

```java
public class MatcherTest{

  @Test
  public void anyMatcher(){

    Matcher m1 = new RegexMatcher("(.+)（.+）", 1);
    Matcher m2 = new RegexMatcher("(.+)第.+[季|部]", 1);
    Matcher m3 = new RegexMatcher(".+", 0);

    Seq<Matcher> seq = JavaConverters.asScalaIteratorConverter(Arrays.asList(m1, m2, m3).iterator()).asScala().toSeq();
    AnyMatcher any = new AnyMatcher(seq);
    assertEquals(any.tryMatch("你好（hello）").get(), "你好");
    assertEquals(any.tryMatch("你好 第一季").get(), "你好");
    assertEquals(any.tryMatch("你好").get(), "你好");

  }

}
```

很稳！使用 RegexMatcher 实例组合得到的 Matcher 覆盖了所有能通过正则表达式提取的情况。但上面的测试中有一行

```java
Matcher m3 = new RegexMatcher(".+", 0);
```

它的语义是将输入原封不动的返回，我们可以把这种语义单独定义成组合子

```scala
class IdMatcher extends Matcher{
  override def tryMatch(title: String): Option[String] = Some(title)
}
```

接下来经过更细致的分析用户的输入和节目库中的名称，我们发现，对于一些内部含有标点符号的名称，类似 "xxx，xx"，它整个就是主标题，但是用户输入不一定含有这个逗号，所以需要把这些标点移除，当然首先是把它们取出来，为此我们定义一个提取字符的组合子

```scala
class CharMatcher(char: Char) extends Matcher {
  override def tryMatch(title: String): Option[String] = {
    String.valueOf(title.toCharArray.filter(c => c == char)) match {
      case str if !str.isEmpty => Some(str)
      case str if str.isEmpty => None
    }
  }
}
```

然后定义一个取反的组合子，它把传入的组合子提取的字符串取补集

```scala
class NotMatcher(matcher: Matcher) extends Matcher{
  override def tryMatch(title: String): Option[String] = {
    ...
  }
}
```

将 CharMatcher 实例传入 NotMatcher 便能得到移除字符的组合子

```java
CharMatcher cm = new CharMatcher('，');
NotMatcher nm = new NotMatcher(cm);
nm.tryMatch("xxxx，xxx");
```

移除字符和提取主标题这两个操作可以不分先后，但必须都进行，这种组合方式又可以抽象出新的组合子 AllMatcher

```scala
class AllMatcher(matchers: Matcher*) extends Matcher{

  override def tryMatch(title: String): Option[String] = {
    val options = matchers.map(matcher => matcher.tryMatch(title))
    options.find(opt => opt.isEmpty) match {
      case Some(_) => return None
      case None =>
    }

    options
      .map(opt => opt.get)
      .reduce((str1, str2) => Strs.LCS(str1, str2)) match {
        case str if str.isEmpty = None
        case str if !str.isEmpty = Some(str)
      }

  }
}
```

它的实现逻辑是返回内部所有组合子提取的字符串的最长公共子序列。到目前为止，我们定义了 RegexMatcher，CharMatcher，NotMatcher，AnyMatcher，AllMatcher 这 5 种组合子，为了方便创建对象，我们定义一批工厂方法

```scala
object Matcher{

  def id = new IdMatcher

  def not(m: Matcher) = new NotMatcher(m)

  def char(c: Char) = new CharMatcher(c)

  def regex(rgx: String, group: Int) = new RegexMatcher(rgx, group)

  def any(matchers: Matcher*) = new AnyMatcher(matchers:_*)

  def all(matchers: Matcher*) = new AllMatcher(matchers:_*)

}
```

现在，我们可以组合出一个具有满足前面所提需求的提取器

```scala
val matcher: Matcher = all(
    any(
      regex("(.+)（.+）", 1), 
      regex("(.+)第.+[季|部]", 1), 
      regex("(.+)\\d{1}[:|：|-]", 1), 
      regex("(.+)[:|：|-]", 1), 
      regex("(.+)\\d{1}$", 1)
      id
  ),
    not(char('，')), 
    not(char('！')), 
    not(char('·')), 
    not(char(' ')) 
  )
```

利用组合子的设计，在使用的过程中，如果发现了一些我们未曾意识到的模式，也能很容易的扩展。
