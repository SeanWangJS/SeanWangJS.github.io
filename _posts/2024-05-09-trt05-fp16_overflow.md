---
title: TensorRT 使用指南（5）：FP16 精度溢出的解决方案
tags: TensorRT
---

TensorRT 在 fp16 精度下可以获得相对于 fp32 大概 2 倍的速度提升，一般情况下，精度损失较小。但如果模型结构比较特殊，比如连续多个卷积层之间没有 Normalize 操作，则有很大概率使得激活值超出 fp16 的表示范围（也就是绝对值大于 65504），这时模型会输出 NaN。解决这个问题的方式很简单，直接将溢出层的精度提升到 fp32 即可。下面我们来详细说明。

首先，我们需要找到哪些层溢出了，也就是激活张量超过了 fp16 的表示范围，为了保险起见，可以将最大绝对值设置成 10000，也就是说如果激活值的绝对值超过 10000，我们就认为这个层溢出了。为此我们需要 TensorRT 提供的 Polygraphy 工具将 ONNX 模型转换一下，将每一层的输出都保存下来，参考下面的代码

```python
import onnx
from polygraphy.backend.onnx import modify_outputs
from polygraphy import constants

inputs = model.graph.input
input_names = [input.name for input in inputs]

model = onnx.load("/path/to/model.onnx")
modified_model = modify_outputs(model, outputs=constants.MARK_ALL, exclude_outputs=input_names)

onnx.save_model(modified_model, '/path/to/model_all_output.onnx')
```

这里最关键的是 `modify_outputs` 方法，`constants.MARK_ALL` 表示将所有层的输出都作为模型的输出，然后再将输入排除掉。

接下来使用 `polygraphy run` 命令来运行模型

```bash
polygraphy run model_all_output.onnx --onnxrt --load-inputs inputs.json --save-outputs outputs.json
```

注意，这里的 `inputs.json` 请使用真实的数据，而不是随机生成的数据，因为我们需要找到真实数据下哪些层溢出了。它可以通过 `polygraphy` 的 `json` 模块来生成

```python
from typing import OrderedDict
from polygraphy.json import save_json

input_tensor = ... ## 真实的输入张量
d = OrderedDict()
d["x"] = input_tensor
data = [d]
save_json(data, "/path/to/inputs.json")
```

接下来，我们分析 `outputs.json` 数据，同样，使用 `polygraphy` 的 `json` 模块来读取

```python
outputs_all = load_json("/path/to/outputs.json")
data = outputs_all[0][1][0]
overflow_output_names = []
for key in data:
    out_data = data[key]
    max_abs = np.max(np.abs(out_data))
    if max_abs > 10000:
        overflow_output_names.append(key)
```

这里的 `overflow_output_names` 就是溢出激活值的名称，接下来我们需要将输出这些张量的层的精度提升到 fp32，可以参考下面的代码

```python
import tensorrt as trt

onnx_path = "/path/to/model.onnx"
engine_path = "/path/to/model_fp16.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
parser = trt.OnnxParser(network, logger)

profile.set_shape("x", 
                  min = (1, 3, 512, 512),
                  opt = (1, 3, 512, 512),
                  max = (1, 3, 512, 512))

with open(onnx_path, "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
config.add_optimization_profile(profile)
## 设置 fp16 精度
config.set_flag(trt.BuilderFlag.FP16)
## 设置 profiling_verbosity 为 DETAILED，方便后面检查生成的 Engine 各算子的详细信息（主要是精度类型）
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

## 将溢出层的精度设置到 fp32
for idx in range(network.num_layers):
    layer = network.get_layer(idx)
    output_name = layer.get_output(0).name
    if output_name in overflow_output_names:
        print(f"Set {layer.name} to FP32, output name: {output_name}")
        layer.precision = trt.float32
        layer.set_output_type(0, trt.DataType.FLOAT)

## 设置 OBEY_PRECISION_CONSTRAINTS，强制 TensorRT 服从我们的精度设置
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

engineString = builder.build_serialized_network(network, config)

with open(engine_path, "wb") as f:
    f.write(engineString)
```

具体的代码逻辑已经在注释中说明了，下面我们检查一下生成的 Engine

```python
with open(engine_path, "rb") as f:
    engine=trt.Runtime(logger).deserialize_cuda_engine(f.read())
    inspector = engine.create_engine_inspector()

print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
```

通过打印的信息，我们可以查看哪些层的精度被设置成了 fp32。

最后需要提一下，在确定溢出层的时候，我们只用了一个真实数据，这可能不够全面，所以在实际使用的时候，最好有一批真实数据，类似于 int8 量化时的标定过程，找到所有可能的溢出层。