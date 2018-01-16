---
layout: post
title: UDP 协议校验和计算 
---

UDP 协议不具备差错恢复机制，但能通过校验和判断网络传输是否出现丢包。UDP 的校验和由三部分内容求和得到，分别是伪首部、首部以及数据报文，如下图所示

![](/resources/2017-10-19-udp-protocol-checksum/udp2.png)

为了弄清楚这些字段究竟是什么东西，下面我们使用 wireshark 来抓取一个 UDP 包来详细分析。为了制造这个 UDP 包，使用如下代码来向某 ip 地址发送一段数据（这个 ip 不一定非得实际存在，我们只需要观察基于 UDP 协议封装的数据，所以只要能被 wireshark 获取就行）

```java
import java.io.IOException;
import java.net.*;

/**
 * Created by wangx on 2017/10/19.
 */
public class Client {

    private DatagramSocket socket;

    public Client() throws SocketException {
        socket = new DatagramSocket();
    }

    public void run() throws IOException {
        InetAddress ip = InetAddress.getByName("11.111.111.111");
        String message = "hello UDP";
        byte[] buffer = message.getBytes();
        DatagramPacket sendPacket = new DatagramPacket(buffer, buffer.length, ip, 12345);
        socket.send(sendPacket);

    }

    public static void main(String[] args) throws IOException {
        new Client().run();
    }

}
```

在运行之前，启动 wireshark 捕获，之后找到 Destination 栏为 11.111.111.111 的 UDP 行，类似于下图所示

![](/resources/2017-10-19-udp-protocol-checksum/wireshark.png)

然后找到这个 UDP 包的详细信息

![](/resources/2017-10-19-udp-protocol-checksum/detail.png)

以及这段信息的十六进制表示

![](/resources/2017-10-19-udp-protocol-checksum/hex.png)

有了以上这些内容，剩下的就是对照着最开始的图表来寻找各个参数的值了( wireshark 一个十分好用的功能就是点选上面的人类可读内容，其十六进制值会在下面高亮显示)。我用下表来表示

|key|human|hex|
|:--|:----|:--|
|Source|192.168.1.106|c0a8 016a|
|Destination|11.111.111.111|0b6f 6f6f|
|Protocol|UDP(17)|11|
|Length|17|11|
|Source Port|63549|f83d|
|Destination Port|12345|3039|
|Length|17|11|
|Checksum|0xb12d|b12d|
|Data|hello UDP|6865 6c6c 6f20 5544 5000|

然后就可以开始着手校验和的计算了，但在这之前还应注意，上表中有一项 Checksum ，这是发送方根据发送内容计算出来校验和，接受方需要根据收到的内容重新计算一遍校验和，然后再对比两者。所以在接收方计算时应该忽略这里 Checksum 项。

校验和的计算规则很简单，就是将上表中所有的 16 进制数加起来，之后取反码。有一点需要注意的是，如果遇到最高位进位，那么需要对结果进行回卷，意思是

![](/resources/2017-10-19-udp-protocol-checksum/backscroll.png)

简单来说，就是将要进的那一位加到尾部，上面是以二进制演示的，对于16进制同样适用。

参考：

《计算机网络——自顶向下方法》

[如何计算UDP/TCP检验和checksum](http://blog.csdn.net/lanhy999/article/details/51123626)