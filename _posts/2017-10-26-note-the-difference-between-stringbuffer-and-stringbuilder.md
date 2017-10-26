---
layout: default
---

## Note: Java(JDK 1.9) StringBuffer 与 StringBuilder 的区别

StringBuffer 与 StringBuilder 都是为了更高效的操纵字符串而生，原因在于 String 类是不可变量，对其进行任何修改都会产生新的对象，而 StringBuffer 和 StringBuilder 则能够有效避免这一问题。但既然区分了两个类，就说明它们之间存在差别。

查看源码，发现 StringBuffer 与 StringBuilder 都继承 AbstractStringBuilder。下面从源码层面来对比两者

|StringBuffer|StringBuilder|
|:--|:--|
|public StringBuffer(int capacity) |public StringBuilder(int capacity) |
|public StringBuffer(String str) |public StringBuilder(String str) |
|public StringBuffer(CharSequence seq) |public StringBuilder(CharSequence seq) |
|public synchronized int length() ||
|public synchronized int capacity() ||
|public synchronized void ensureCapacity(int minimumCapacity) ||
|public synchronized void trimToSize() ||
|public synchronized void setLength(int newLength) ||
|public synchronized char charAt(int index) ||
|public synchronized int codePointAt(int index) ||
|public synchronized int codePointBefore(int index) ||
|public synchronized int codePointCount(int beginIndex, int endIndex) ||
|public synchronized int offsetByCodePoints(int index, int codePointOffset) ||
|public synchronized void getChars(int srcBegin, int srcEnd, char[] dst,int dstBegin)||
|public synchronized void setCharAt(int index, char ch) ||
|public synchronized StringBuffer append(Object obj) |public StringBuilder append(Object obj) |
|public synchronized StringBuffer append(String str) |public StringBuilder append(String str) |
|public synchronized StringBuffer append(StringBuffer sb) |public StringBuilder append(StringBuffer sb) |
|synchronized StringBuffer append(AbstractStringBuilder asb) ||
|public synchronized StringBuffer append(CharSequence s) |public StringBuilder append(CharSequence s) |
|public synchronized StringBuffer append(CharSequence s, int start, int end)|public StringBuilder append(CharSequence s, int start, int end) |
|public synchronized StringBuffer append(char[] str) |public StringBuilder append(char[] str) |
|public synchronized StringBuffer append(char[] str, int offset, int len) |public StringBuilder append(char[] str, int offset, int len) |
|public synchronized StringBuffer append(boolean b) |public StringBuilder append(boolean b) |
|public synchronized StringBuffer append(char c) |public StringBuilder append(char c) |
|public synchronized StringBuffer append(int i) |public StringBuilder append(int i) |
|public synchronized StringBuffer appendCodePoint(int codePoint) |public StringBuilder appendCodePoint(int codePoint) |
|public synchronized StringBuffer append(long lng) |public StringBuilder append(long lng) |
|public synchronized StringBuffer append(float f) |public StringBuilder append(float f) |
|public synchronized StringBuffer append(double d) |public StringBuilder append(double d) |
|public synchronized StringBuffer delete(int start, int end) |public StringBuilder delete(int start, int end) |
|public synchronized StringBuffer deleteCharAt(int index) |public StringBuilder deleteCharAt(int index) |
|public synchronized StringBuffer replace(int start, int end, String str) |public StringBuilder replace(int start, int end, String str) |
|public synchronized String substring(int start) ||
|public synchronized CharSequence subSequence(int start, int end) ||
|public synchronized String substring(int start, int end) ||
|public synchronized StringBuffer insert(int index, char[] str, int offset, int len)|public StringBuilder insert(int index, char[] str, int offset, int len)|
|public synchronized StringBuffer insert(int offset, Object obj) |public StringBuilder insert(int offset, Object obj) |
|public synchronized StringBuffer insert(int offset, String str) |public StringBuilder insert(int offset, String str) |
|public synchronized StringBuffer insert(int offset, char[] str) |public StringBuilder insert(int offset, char[] str) |
|public StringBuffer insert(int dstOffset, CharSequence s) |public StringBuilder insert(int dstOffset, CharSequence s) |
|public synchronized StringBuffer insert(int dstOffset, CharSequence s, int start, int end)|public StringBuilder insert(int dstOffset, CharSequence s, int start, int end)|
|public  StringBuffer insert(int offset, boolean b) |public StringBuilder insert(int offset, boolean b) |
|public synchronized StringBuffer insert(int offset, char c) |public StringBuilder insert(int offset, char c) |
|public StringBuffer insert(int offset, int i) |public StringBuilder insert(int offset, int i) |
|public StringBuffer insert(int offset, long l) |public StringBuilder insert(int offset, long l) |
|public StringBuffer insert(int offset, float f) |public StringBuilder insert(int offset, float f) |
|public StringBuffer insert(int offset, double d) |public StringBuilder insert(int offset, double d) |
|public int indexOf(String str) |public int indexOf(String str) |
|public synchronized int indexOf(String str, int fromIndex) |public int indexOf(String str, int fromIndex) |
|public int lastIndexOf(String str) |public int lastIndexOf(String str) |
|public synchronized int lastIndexOf(String str, int fromIndex) |public int lastIndexOf(String str, int fromIndex) |
|public synchronized StringBuffer reverse() |public StringBuilder reverse() |
|public synchronized String toString() |public String toString() |
|private synchronized void writeObject(java.io.ObjectOutputStream s)|private void writeObject(java.io.ObjectOutputStream s)|
|private void readObject(java.io.ObjectInputStream s)|private void readObject(java.io.ObjectInputStream s)|
|synchronized void getBytes(byte dst[], int dstBegin, byte coder) ||

可以看到，StringBuffer 在几乎每个方法前面都有 synchronized 关键字修饰，即这个类是线程安全的，而在实现上两者都大致相同，比如 append(String str) 方法

```java

//StringBuffer
public synchronized StringBuffer append(String str) {
        toStringCache = null;
        super.append(str);
        return this;
    }
//StringBuilder
    public StringBuilder append(String str) {
        super.append(str);
        return this;
    }
```

由于线程同步需要额外的性能消耗，所以在单线程环境中更适合使用 StringBuilder。
