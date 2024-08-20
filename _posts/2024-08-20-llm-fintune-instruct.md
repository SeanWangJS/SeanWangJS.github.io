---
title: LLM 指令微调方法
tags: LLM 模型训练
---

## 介绍

经过无监督预训练的模型已经具备基本的语言理解能力，也就是说，给它输入一段不完整的文本，它能够比较准确的预测下一个词是什么，但也仅此而已。此时的模型遵循指令的能力还很弱，也基本没有什么对话能力，在早期的 GPT3 时代，让模型完成一些任务的方法是构造一个相对特殊的 prompt，比如让它写一个新闻

```
Title: United Methodists Agree to Historic Split
Subtitle: Those who oppose gay marriage will form their own denomination
Article:
```

这里的 `Article: ` 表明了这是一个还没有完成的文本，如果模型理解了前面的内容，那么就该知道接下来应该生成什么。

受到这种方法的启发，后期的模型在完成了预训练之后，一般会再使用特定格式的 prompt 来对模型进一步训练，从而增强模型在这种格式输入下的表现，比如 [alpaca](https://github.com/tatsu-lab/stanford_alpaca) 项目中的指令模板

```
Below is an instruction that describes a task. Write a response that appropriately completes the request
        
### Instruction:
{instruction}

### Response: 
```

以及 [vicuna](https://github.com/lm-sys/FastChat) 项目中的指令模板

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 

### Human: {instruction} 

### ASSISTANT:
```

让模型适应固定指令格式的训练阶段就叫做**指令微调**(instruction tuning)，通过这种方式训练，模型可以更准确的理解用户的意图，从而给出生成更合理的回复。

当然，后来各家模型都遵循 OpenAI 的调用规范，使用 `role` 和 `content` 来表示对话的角色和内容，比如

```python
messages=[
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hello, how are you?"},
  {"role": "user", "content": "I'm fine, thanks, and you?"},
]
```

然后再使用自家的格式化规范来生成 prompt，比如 [gemma](https://huggingface.co/google/gemma-2b-it) 模型的格式化结果

```
<bos><start_of_turn>user
hello<end_of_turn>
<start_of_turn>model
hello, how are you?<end_of_turn>
<start_of_turn>user
i'm fine, thanks, and you?<end_of_turn>
```

由于这种规范实在太好用，所以 huggingface 也在 transformers 的 Tokenizer 中提供了 `apply_chat_template` 方法，可以直接将对话内容格式化

```python
tokenizer.apply_chat_template(messages, tokenize=False)
```

## 数据集处理

微调阶段的数据集一般都是有监督数据，也就是说，数据集中的每个样本都有一个 context 序列和一个 label 序列，我们的训练目标就是让模型的输出尽可能接近 label 序列。无论原始的数据格式是怎样的，我们都需要将每个样本转换成如下所示的序列格式

```
| context  |  label     |
o o o o o o x x x x x x x
```

以 alpaca 数据集为例，它的样本格式如下:

```json
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
}
```

其中 `instruction` 和 `input` 共同构成 context，`output` 构成 label，利用 transformers tokenizer的 `apply_chat_template` 方法，我们可以将这个样本进行转换

```python
def generate_text(sample: dict):
    if "input" in sample:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
{sample["instruction"]}. Here are the inputs: {sample["input"]}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
{sample["instruction"]}."""

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": sample["output"]}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text
```

如果 tokenizer 本身没有 chat_template（这在比较新的模型上不太常见），那么我们可以自己设置一个

```python
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

这里的表达式是 jinja2 模板语言，更详细的信息可以参考 transformers 的[官方文档](https://huggingface.co/docs/transformers/main/en/chat_templating)。利用上述模板，我们可以将样本格式化为

```
<|im_start|>user
Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
Give three tips for staying healthy.. Here are the inputs: <|im_end|>
<|im_start|>assistant
1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.

2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.

3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.<|im_end|>
```

这里 `<|im_start|>assistant` 以及前面的部分就是 context 序列，后面的部分就是 label 序列，当然，不同 chat_template 的标志符号不同。

接下来我们就需要对整个序列进行 tokenization

```python
tokenizer.encode(text, return_tensors="pt", max_length = 1024, truncation=True)
```

然后构造成 batch，但是这里存在几种不同的路线选择。我们知道 batch 里面每个样本张量的维度必须是一样的，如果样本 token 序列的长度不同，需要把不足的部分填充，也就是常说的 padding，比如下面给出的例子

```
x x x x x x x o o o o o o o o * * * 
x x x x x x x x x x o o o o o o o o 
x x x x o o o o o o o o o o * * * *
```

其中 `*` 就是 padding 的部分。一般常规的方法就是这样了，但是显然，填充的部分消耗了计算量，却没有在计算损失的时候提供任何贡献，因此这是一种低效的做法。更加高效的做法是类似于预训练阶段的数据处理方式，把所有文本序列拼接在一起，然后按照一定的 `max_length` 和 `batch_size` 对 token 序列进行切分，这样就可以避免填充的问题了，这种方式也叫 packed，生成的 batch 类似下面这样

```
x x x x x x x o o o o o o o o x x x 
x x x x x x x o o o o o o o o x x x 
x o o o o o o o o o o
```

但是这样做有一个明显的缺陷，就是在切分的时候，可能会把一个样本的 context 和 label 切分到两个 tensor 中，并且指令模板的特殊字符也不一定在 tensor 的开头或结尾。后面我们将会看到这对训练稍微有点负面影响。

最后一种解决方案在 max_length 限制下，尽可能多的拼接样本序列，使得 padding 数量尽量少，并且不对序列进行切分，这样可以保证每个样本的 context 和 label 都在同一个 tensor 中，我把这种方式称为 `concat`。

## 损失函数

> 这一节涉及原理性的内容，和具体的代码关系不大，如果不感兴趣可以跳过。

自回归语言模型的预测模式是给定一个上下文 token 序列，预测下一个 token 的分布，然后对这个分布进行采样后将新的 token 拼接到原序列后面，继续预测下一个 token 分布，直到达到最大序列长度或者得到终止 token。设上下文 token 序列符号表示为 $x_1, x_2,...,x_c$，那么自回归语言模型的推理过程本质上是在不断计算下列条件分布

$$
\begin{aligned}
&p(x_{c+1} \mid x_1, x_2,...,x_c)\\
&p(x_{c+2} \mid x_1, x_2,...,x_{c+1})\\
&\cdots\\
&p(x_{c+k} \mid x_1, x_2,...,x_{c+k-1})\\
\end{aligned}
$$

其中 $k$ 表示 label token 序列长度。我们可以将 label token 序列的联合分布关于上下文序列的条件概率表示为

$$
p(x_{label}\mid x_{ctx};\theta)
$$

其中 $x_{label}$ 表示 label token 序列随机变量，$x_{ctx}$ 表示上下文 token 序列随机变量，$\theta$ 表示模型参数。在真实样本条件下，对于未知的模型参数，我们可以得到似然函数

$$
\mathcal{L}(\theta) = p(x_{label} = X_{label} \mid x_{ctx} = X_{ctx}; \theta)
$$

这里的 $X_{label}$ 和 $X_{ctx}$ 表示 label token 序列和上下文 token 序列，即 $(X_{ctx}, X_{label})$ 是一个训练样本。

利用贝叶斯公式，我们可以将上述似然函数变换为（注意这里我们为了简化推导，省略了 $\theta$）

$$
\begin{aligned}
\mathcal{L}(\theta) &= p(label\mid ctx) \\&= p(x_{c+1}, x_{c+2},...,x_{c+k} \mid x_1, x_2,...,x_c) \\
&= p(x_{c+2},...,x_{c+k} \mid x_1, x_2,..., x_{c+1}) p(x_{c+1} \mid x_1, x_2,...,x_c)\\
&= p(x_{c+3},...,x_{c+k} \mid x_1, x_2,..., x_{c+2}) p(x_{c+2} \mid x_1, x_2,...,x_c, x_{c+1}) p(x_{c+1} \mid x_1, x_2,...,x_c)\\
&= p(x_{c+4},...,x_{c+k} \mid x_1, x_2,...,x_{x+3}) \prod_{i=1}^{3} p(x_{c+i} \mid x_1, x_2,...,x_{c+i-1}) \\
&\cdots\\
&= p(x_{c+k}\mid x_1, x_2,...,x_{c+k-1}) \prod_{i=1}^{k-1} p(x_{c+i} \mid x_1, x_2,...,x_{c+i-1})\\
&= \prod_{i=1}^k p(x_{c+i} \mid x_1, x_2,...,x_{c+i-1})

\end{aligned}
$$

显然，为了获得最佳的语言模型参数，我们需要最大化似然函数，即

$$
\theta^{best} = \argmax_{\theta} \mathcal{L}(\theta)
$$

而为了简化计算，我们通常会取似然函数的对数形式，即

$$
\log(\mathcal{L}(\theta)) = \sum_{i=1}^k \log(p(x_{c+i} \mid x_1, x_2,...,x_{c+i-1};\theta))
$$

对于损失函数来讲，根据其定义，我们可以为上式添加一个负号，使得计算目标为损失函数最小化，所以自回归语言模型的损失函数就是负对数似然函数。

至于为什么在实际代码中优化目标函数一般是交叉熵，简单来说，当语言模型同时满足平稳性和遍历性条件时，模型输出与标签之间的交叉熵为（具体可以参考[这篇文章](https://seanwangjs.github.io/2024/01/09/ppl.html)）

$$
H(\eta, \xi) = \lim_{n \to \infty} -\frac{1}{n} \sum_{i=1}^n \log p_{\xi}(X_{i} \mid X_{\lt i})
$$

可以看到上述交叉熵的公式与负对数似然函数只差了一个系数 $\frac 1 n$，原因在于交叉熵在整个样本集上取得平均，而我们推导的负对数似然函数只是针对单个样本的，所以两者实际上是等价的。但值得注意的是，在 PyTorch 实现中，`torch.nn.functional` 模块下的 `cross_entroy` 和 `nll_loss` 这两个函数，`cross_entropy` 会自动对输入进行 softmax 操作，而 `nll_loss` 不会，也就是说 `cross_entroy` 的输入是未归一化的 logits，而 `nll_loss` 的输入是已经归一化的概率分布。下面这个例子应该能说明情况

```python
import torch
import torch.nn.functional as F

logits = torch.randn(1, 10, 100)
probs = F.softmax(logits, dim=-1)
labels = torch.randint(0, 100, (1, 10))

print(F.cross_entropy(logits.view(-1, 100), labels.view(-1)))
print(F.nll_loss(torch.log(probs.view(-1, 100)), labels.view(-1)))
```

## 代码实现

接下来我们以 GPT2 为例，使用 alpaca 数据集进行指令微调，让原本只能进行文本补全的 GPT2 模型具备回答问题的能力。之所以选择 GPT2 是因为，首先它是一个比较小巧的模型，最小的版本只有 270M 参数，可以在一般的笔记本上训练，另外它的预训练模型没有指令模板，方便展示对其赋予指令理解能力的过程。

### 设置 chat_format

在近期版本的 transformers 框架中，为了解决不同模型的指令格式不同的问题，统一为模型提供了 `apply_chat_format` 接口，可以将对话内容格式化到模型能识别的指令形式。由于 GPT2 模型没有经过指令微调，所以我们需要自己设置一个 chat_format，这里借鉴 trl 框架的 https://github.com/huggingface/trl/blob/main/trl/models/utils.py 文件中的代码并进行简化

```python
from typing import Tuple
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )

```


## 参考链接

* https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-gemma-0444d46d821c

* https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2

* https://huggingface.co/docs/transformers/main/en/chat_templating

* https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/