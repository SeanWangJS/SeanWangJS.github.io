---
layout: default
---

## 使用行为树实现应用程序命令行解析工具

不知道是不是我一开始想多了，在做本科毕业论文的时候，为了更方便地运行程序，绞尽脑汁写了一个命令行解析器。而后又经过几次迭代，感觉成熟了不少，但是回过头发现好像并用不着这样的解析器。。。当然主体程序跟现在要讨论的内容不相关就不说了，只是花了精力实现的东西还是希望记录下思路。

行为树的原理，简单来说，系统的状态从根节点出发，接收外部信息，每接受一个信息，就按照预定义的跳转表进行状态跳转。当所有信息都处理完之后，最终停留的那个状态一般都有个动作，运行之后便完成了对程序的控制。

所以实际上要完成两个阶段的任务，首先是利用待实现的命令行语句构建一颗行为树，语句里的每个单词都是一个状态跳转指示。然后实现一个语句解析器，负责读取语句，并进行相应的状态跳转。

### 构造行为树

预定义一颗行为树的方法有很多种，主要是表现形式的问题。我这里使用json格式的配置文件来作说明如下：

```json
{  
  "START": {
    "transitions": {
	  "cmd1": "CMD1",
	  "cmd2": "CMD2",
	  "cmd3": "CMD3"
    }
  }
}
```

在上面的配置中，大写单词如"START"， "CMD1"， "CDM2"，"CMD3"，就是所谓的节点。在"transitions" 对象中定义了跳转条件，即从"START"开始，如果输入"cmd1"，则跳转到"CMD1"节点，依此类推。而对于后续的输入，当然还要继续定义，例如：

```json
{
  "CMD1": {
    "transitions": {
	  "a": "CMD1_A"
    }
  },  
  "CMD2": {
    "transitions": {
	  "b": "CMD2_B"
    }
  }
}
```


使用这种方式可以将所有命令语句都用一颗行为树来表达，并且具有很好的灵活性，可以随时修改。这算是大体完成了第一阶段的任务，很简单。

接下来需要加载配置文件，那么首先要通过代码建立起行为树的模型，也就是规定其包含的内容与拥有的行为。像二叉树一样，只需包含根节点，然后通过各节点之间的链接关系就可以访问到树上的每个节点。行为树也不必完全拥有其树上所有节点的引用，而只需要知道最关键的信息，也就是当前节点和起始节点。行为树所拥有的动作当然就是执行命令语句，并且在语句执行完成后回复到起始状态，以便下一条语句的执行。粗略的代码如下：

```java
public class BehaviorTree{

	private Node start;
	private Node current;
	
	public BehaviorTree(Node start) {
		this.start = start;
		current = start;
	}

	public void execute(String command) {
		//command execution code
	}

	public void reset() {
		current = start;
	}
	
}
```

而每个节点都包含一个跳转表，即在特定指令单词下，返回下一个状态节点

```java
public class Node {
	private HashMap<String, Node> transitions = new HashMap<>();
	
	public Node next(String word) {
		return transitions.get(word);
	}
	
	public void addTransition(String trigger, Node target) {
		transitions.put(trigger, target);
	}
	
	public void run() {
	        //run
	}
}
```

节点类的addTransition()方法需要在构造节点对象的时候调用，其依据就是前面所述的行为树配置文件中的跳转规则。而next()方法则根据传入的指令返回跳转表中的相应后续节点。最后还有个run()方法，这只在叶节点（又称为动作节点）中才实现，当然一般节点也可以输出些调试信息。

现在解释下命令语句的执行过程：当调用行为树的execute()方法时，传入命令，首先将语句分解成一个词组，然后调用current节点的next()方法，传入第一个单词，这将返回下一个节点，将其绑定为current。接着传入下一个单词，循环这一过程，直到所有单词都输入，最后得到的节点就是需要的动作节点，最后run()方法实现了这句指令想要执行的操作。

```java
public void excute(String command) {
	String[] words = command.trim().split("\\s+");
	for(String word : words) {
		current = current.next(word);
	}
	current.run();	
        current = start;
}
```

好了，介绍完大致思路之后，就要开始构建行为树了，一种很自然的想法是——为每个状态定义一个类（继承自Node类），例如我们可以定义 class Start extends Node{}，class CMD1 extends Node{}，class CMD2 extends Node{}等等，然后再为每个类创建实例，并配置跳转条件，类似下面这种

```java
start.addTransition("cmd1", cmd1);
start.addTransition("cmd2", cmd2);
cmd1.addTransition("a", cmd1_a);
//....
```

想当初写着这样的代码的时候内心是崩溃的，还使用了好长时间。最后幡然醒悟，为何要为这些中间节点专门写个类，简直多此一举。简单的做法是创建很多Node实例作为状态集合，将这些对象放入哈希表map，其键值由到达这些状态所需要的字符串命令组成。例如，将对象cmd1_a的键值设为"cmd1_a"，这种方式规定的键值显然是唯一的。

前面提到的 json 配置文件就是生成行为树的一种很好的方式，因为它定义了所有的节点，以及节点间的跳转规则。于是就可以想办法把这种配置解析成 java 类，从而组装成一棵树。下面的代码首先将所有的节点找到，并以名称为键放入哈希表

```java
String json = new String(new Files.readAllBytes(Paths.get(uri)));
JsonObject configJson = gson.fromJson(json, JsonObject.class);
HashMap<String, Node> nodes = new HashMap<>();
configJson.entrySet().forEach(e -> {
			String name = e.getKey();
			nodes.put(name, new Node());
		});
```

其中的 gson 为 google 的 json 解析工具。
最后为每个节点配置跳转规则，便完成了配置文件的解析。

```java
configJson.entrySet().forEach(e -> {
		String name = e.getKey();
		JsonObject properties = e.getValue().getAsJsonObject();
		JsonElement transitions = properties.get("transitions");
		if(transitions != null) {
			transitions.getAsJsonObject().entrySet().forEach(t -> {
				String trigger = t.getKey();
				String targetName = t.getValue().getAsString();
				nodes.get(name).addTransition(trigger, nodes.get(targetName));
			});
		}
	});
```
