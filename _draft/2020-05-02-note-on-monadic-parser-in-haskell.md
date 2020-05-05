---
title: 笔记：monadic parser in haskell
tags: 笔记
---

解析器定义 

```haskell
newtype Parser a = Parser (String -> [(a, String)])
```

这里定义解析器的类型为一个函数，它接收一个 String 类型值，并返回一个数组，其中的元素类型为 (a, String)。也就是说，解析器是一个函数，它把输入的字符串数据转换成 a 和 String 的二元组列表，其中 a 是参数类型，由使用者指定。比如下面的名为 item 的解析器

```haskell
item :: Parser (Char -> [(Char, String)])
item = Parser (\cs -> case cs of 
                      "" -> []
                      (c: cs) -> [(c, cs)])
```

这里就把 a 的实际类型指定为 Char，而这个解析器的工作内容就是把空字符串转换成空数组，把非空字符串分割成第一个字符和剩余字符串两部分，并作为二元组的列表返回。

为了应用解析器，需要定义一个辅助函数把由 Parser 包裹的真实函数暴露出来

```haskell
parse:: (Parser a) -> String -> [(a, String)]
parse (Parser p) = p
```

于是便可以用下面这种方式来应用解析器

```haskell
str = "hello haskell"
parse item str
{-输出: [('h',"ello haskell")]-}
```

接下来的事实表明 Parser 是一个 Monad

```haskell
instance Monad Parser where 
  return a = Parser (\cs -> [(a, cs)])
  p >>= f = Parser (\cs -> concat [parse (f a) cs' |
                          (a, cs') <- parse p cs])
```

这里的 return 接收一个值 a（注意与前面的类型参数 a 相区分），返回一个解析器，此解析器的作用是，接收一个字符串 cs，然后返回由 a 和 cs 组成的二元组列表。比如

```haskell
p = return 1:: Parser Int
```