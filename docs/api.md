# DPCS-B API 文档

本文档详细介绍了双路径协调系统(DPCS-B)的程序接口(API)，帮助开发者了解如何在自己的项目中集成和使用DPCS-B的功能。

## 1. 概述

DPCS-B提供了多种层次的API，从高级系统API到底层模块API，满足不同的使用需求。

## 2. 系统级API

### 2.1 DualPathCoordinationSystem 类

系统的主要入口类，提供了整个系统的核心功能。

#### 2.1.1 初始化

```python
from dpcs.system import DualPathCoordinationSystem

# 默认配置初始化
dpcs = DualPathCoordinationSystem()

# 自定义配置初始化
config = {
    'input_size': 256,
    'embedding_dim': 768,
    'hidden_size': 512,
    'output_size': 128,
    'use_blockchain': True,
    'da_shards': 16,
    'rollup_type': 'zk',
    'rollup_batch_size': 100
}
dpcs = DualPathCoordinationSystem(config)
```

#### 2.1.2 主要方法

##### process()

处理输入数据，核心处理方法。

```python
# 处理文本输入
result = dpcs.process("需要处理的文本", input_type="text")

# 处理张量输入
import torch
tensor_input = torch.randn(1, 256)  # 创建示例张量
result = dpcs.process(tensor_input, input_type="tensor")

# 处理结构化数据
data = {
    "features": [1.2, 3.4, 5.6, 7.8],
    "categorical": ["A", "B"],
    "timestamp": 1630000000
}
result = dpcs.process(data, input_type="dict")
```

参数:
- `input_data`: 输入数据，可以是字符串、张量或字典
- `input_type`: 输入类型，可选值为"text"、"tensor"、"dict"，默认为"tensor"

返回:
- 处理结果，类型取决于输入和处理路径

##### set_mode()

设置处理模式，决定使用哪种路径。

```python
# 设置为左脑路径
dpcs.set_mode("left")

# 设置为右脑路径
dpcs.set_mode("right")

# 设置为双路径模式
dpcs.set_mode("dual")
```

参数:
- `mode`: 处理模式，可选值为"left"、"right"、"dual"

##### set_blockchain()

启用或禁用区块链功能。

```python
# 启用区块链
dpcs.set_blockchain(True)

# 禁用区块链
dpcs.set_blockchain(False)
```

参数:
- `use_blockchain`: 布尔值，表示是否使用区块链

##### optimize_model()

优化模型计算。

```python
# 执行模型优化
dpcs.optimize_model()
```

返回:
- 布尔值，表示优化是否成功

##### manage_memory()

内存管理优化。

```python
# 管理内存，返回最佳批处理大小
batch_size = dpcs.manage_memory(input_size=256, target_memory_usage=0.8)
```

参数:
- `input_size`: 输入大小
- `target_memory_usage`: 目标内存使用率，默认为0.8

返回:
- 最佳批处理大小

##### enable_parallel()

启用并行处理。

```python
# 启用数据并行
dpcs.enable_parallel()

# 指定设备ID
dpcs.enable_parallel(device_ids=[0, 1])

# 启用分布式并行
dpcs.enable_parallel(distributed=True)
```

参数:
- `device_ids`: 设备ID列表，默认为None（使用所有可用设备）
- `distributed`: 是否使用分布式并行，默认为False

返回:
- 布尔值，表示是否成功启用并行处理

##### optimize_blockchain_operations()

优化区块链操作。

```python
# 优化区块链操作
optimization_result = dpcs.optimize_blockchain_operations()
```

返回:
- 优化结果字典

### 2.2 CaseAdapter 类

案例适配器类，用于运行外部示例。

#### 2.2.1 初始化

```python
from dpcs.examples.case_adapter import CaseAdapter

# 默认初始化
adapter = CaseAdapter()

# 指定DPCS系统实例
from dpcs.system import DualPathCoordinationSystem
dpcs = DualPathCoordinationSystem()
adapter = CaseAdapter(dpcs_system=dpcs)
```

#### 2.2.2 主要方法

##### run_case()

运行案例文件。

```python
# 运行特定案例文件
adapter.run_case("examples/basic_usage.py")

# 指定处理模式和区块链设置
adapter.run_case("examples/forecast_example.py", mode="left", use_blockchain=False)
```

参数:
- `main_file_path`: 主文件路径
- `mode`: 处理模式，可选值为"left"、"right"、"dual"，默认为"dual"
- `use_blockchain`: 是否使用区块链，默认为True

返回:
- 布尔值，表示运行是否成功

## 3. 模块级API

### 3.1 左脑模块 (SRMT)

结构化强化学习模块的API。

#### 3.1.1 初始化

```python
from dpcs.modules.srmt import SRMT

# 初始化
srmt = SRMT(input_size=256, hidden_size=512, output_size=128)
```

#### 3.1.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建输入张量
input_tensor = torch.randn(1, 256)

# 前向传播
output, value = srmt(input_tensor)
```

参数:
- `x`: 输入张量

返回:
- `output`: 输出张量
- `value`: 价值估计

### 3.2 右脑模块 (CAMEL Agent)

语义理解与语言生成模块的API。

#### 3.2.1 初始化

```python
from dpcs.modules.camel import CAMELAgent

# 初始化
camel = CAMELAgent(
    model_name="gpt2",
    embedding_dim=768,
    hidden_size=512,
    output_size=128
)
```

#### 3.2.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建输入嵌入
embedding = torch.randn(1, 768)

# 前向传播
output, task_plan = camel(embedding)
```

参数:
- `x`: 输入嵌入张量

返回:
- `output`: 输出张量
- `task_plan`: 任务规划张量

##### generate_text()

生成文本。

```python
# 生成文本
text = camel.generate_text(embedding, max_length=100, temperature=0.7)
```

参数:
- `embedding`: 输入嵌入
- `max_length`: 最大生成长度，默认为50
- `temperature`: 生成温度，默认为0.7

返回:
- 生成的文本字符串

### 3.3 中脑模块 (Spatial Detector)

路由选择器的API。

#### 3.3.1 初始化

```python
from dpcs.modules.spatial import SpatialDetector

# 初始化
detector = SpatialDetector(input_size=256, hidden_size=128)
```

#### 3.3.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建输入张量
input_tensor = torch.randn(1, 256)

# 前向传播
mode_probs = detector(input_tensor)
```

参数:
- `x`: 输入张量

返回:
- 各处理模式的概率分布

##### detect_mode()

检测处理模式。

```python
# 检测模式
selected_mode, probs = detector.detect_mode(input_tensor)
```

参数:
- `input_data`: 输入数据

返回:
- `selected_mode`: 选择的处理模式
- `probs`: 各模式的概率

### 3.4 胼胝体模块 (Corpus Callosum)

信息融合模块的API。

#### 3.4.1 初始化

```python
from dpcs.modules.callosum import CorpusCallosum

# 初始化
callosum = CorpusCallosum(feature_dim=128, fusion_dim=256)
```

#### 3.4.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建左右脑特征
left_features = torch.randn(1, 128)
right_features = torch.randn(1, 128)

# 融合特征
fused_output = callosum(left_features, right_features)
```

参数:
- `left_features`: 左脑特征
- `right_features`: 右脑特征

返回:
- 融合后的特征

### 3.5 小脑模块 (Cerebellum Synchronizer)

时序同步模块的API。

#### 3.5.1 初始化

```python
from dpcs.modules.cerebellum import CerebellumSynchronizer

# 初始化
cerebellum = CerebellumSynchronizer(
    input_dim=128,
    hidden_dim=256,
    lstm_layers=2
)
```

#### 3.5.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建输入张量
input_tensor = torch.randn(1, 128)

# 前向传播
output = cerebellum(input_tensor, time_steps=5)
```

参数:
- `x`: 输入张量
- `time_steps`: 时间步长，默认为5

返回:
- 同步后的输出

##### detect_pattern()

检测时间序列模式。

```python
# 创建序列
sequence = torch.randn(10, 128)

# 检测模式
pattern_info = cerebellum.detect_pattern(sequence, pattern_length=5)
```

参数:
- `sequence`: 输入序列
- `pattern_length`: 模式长度，默认为5

返回:
- 模式信息字典

### 3.6 额叶模块 (Prefrontal Cortex)

执行控制模块的API。

#### 3.6.1 初始化

```python
from dpcs.modules.prefrontal import PrefrontalCortexModule

# 初始化
prefrontal = PrefrontalCortexModule(
    input_size=128,
    hidden_size=256,
    output_size=128,
    num_heads=4
)
```

#### 3.6.2 主要方法

##### forward()

前向传播方法。

```python
import torch

# 创建输入张量
module_output = torch.randn(1, 128)
synchronized_output = torch.randn(1, 128)

# 前向传播
control, meta_output = prefrontal(module_output, synchronized_output)
```

参数:
- `module_output`: 模块输出
- `synchronized_output`: 同步输出

返回:
- `control`: 控制信号
- `meta_output`: 元认知输出

## 4. 区块链API

### 4.1 数据可用性层 (DA Layer)

#### 4.1.1 初始化

```python
from dpcs.blockchain.da_layer import DataAvailabilityLayer

# 初始化
da_layer = DataAvailabilityLayer(shard_count=16, sampling_ratio=0.1)
```

#### 4.1.2 主要方法

##### store_data()

存储数据。

```python
# 存储数据
data_id = "example_data_123"
data = {"key": "value", "numbers": [1, 2, 3]}
metadata = {"type": "example", "timestamp": 1630000000}

proof = da_layer.store_data(data_id, data, metadata)
```

参数:
- `data_id`: 数据ID
- `data`: 数据内容
- `metadata`: 元数据，默认为None

返回:
- 数据可用性证明

##### get_data()

获取数据。

```python
# 获取数据
retrieved_data = da_layer.get_data(data_id)
```

参数:
- `data_id`: 数据ID

返回:
- 存储的数据

### 4.2 计算聚合层 (Rollup Layer)

#### 4.2.1 初始化

```python
from dpcs.blockchain.rollup_layer import RollupLayer
from dpcs.blockchain.da_layer import DataAvailabilityLayer

# 初始化数据可用性层
da_layer = DataAvailabilityLayer()

# 初始化计算聚合层
rollup = RollupLayer(
    rollup_type="zk",  # 或 "optimistic"
    batch_size=100,
    da_layer=da_layer
)
```

#### 4.2.2 主要方法

##### add_transaction()

添加事务。

```python
# 添加事务
tx = {
    'input_id': 'input_123',
    'result_summary': {'mean': 0.5, 'std': 0.1}
}

result = rollup.add_transaction(tx)
```

参数:
- `tx`: 事务数据

返回:
- 添加结果

##### process_batch()

处理批次。

```python
# 处理批次
batch_result = rollup.process_batch()
```

返回:
- 批处理结果

## 5. 接口API

### 5.1 文件上传界面

#### 5.1.1 启动界面

```python
from dpcs.interface.file_upload import main as launch_gui

# 启动图形界面
launch_gui()
```

### 5.2 命令行工具

```bash
# 显示帮助
dpcs --help

# 运行示例
dpcs examples/basic_usage.py

# 指定处理模式
dpcs examples/forecast_example.py --mode left

# 禁用区块链
dpcs examples/cost_tracking_example.py --no-blockchain

# 启动图形界面
dpcs --gui

# 列出可用示例
dpcs --list
```

## 6. 工具函数API

### 6.1 内存管理

```python
from dpcs.utils.memory import MemoryManager

# 初始化内存管理器
memory_manager = MemoryManager(target_usage=0.8)

# 估计内存使用
stats = memory_manager.estimate_memory_usage(model, input_size=256)

# 计算最佳批处理大小
batch_size = memory_manager.compute_optimal_batch_size("model_name")
```

### 6.2 并行处理

```python
from dpcs.utils.parallel import enable_data_parallel, enable_model_parallel

# 启用数据并行
parallel_model = enable_data_parallel(model, device_ids=[0, 1])

# 启用模型并行
model_parts = enable_model_parallel(model, num_gpus=2)
```

### 6.3 优化工具

```python
from dpcs.utils.optimization import quantize_model, apply_mixed_precision

# 量化模型
quantized_model = quantize_model(model)

# 应用混合精度
model, optimizer, scaler = apply_mixed_precision(model, optimizer)
```

## 7. 错误处理

DPCS-B API在遇到错误时会抛出不同类型的异常：

```python
from dpcs.system import DualPathCoordinationSystem
from dpcs.exceptions import DPCSError, InputError, ProcessingError, BlockchainError

# 错误处理示例
try:
    dpcs = DualPathCoordinationSystem()
    result = dpcs.process(input_data)
except InputError as e:
    print(f"输入错误: {e}")
except ProcessingError as e:
    print(f"处理错误: {e}")
except BlockchainError as e:
    print(f"区块链错误: {e}")
except DPCSError as e:
    print(f"一般错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 8. 配置API

### 8.1 配置管理

```python
from dpcs.config.default import get_config, get_flat_config

# 获取全部配置
config = get_config()

# 获取特定部分的配置
srmt_config = get_config('srmt')

# 获取展平的配置
flat_config = get_flat_config()

# 获取特定配置项
value = get_flat_config('system.version')
```

## 9. 版本兼容性

DPCS-B API的版本遵循语义化版本规范(SemVer)，主要版本号的变更可能会引入不兼容的API更改。

```python
from dpcs import __version__

# 检查版本
print(f"DPCS-B 版本: {__version__}")
```

## 10. 最佳实践

### 10.1 性能优化

- 对于大规模任务，使用`optimize_model()`方法进行优化
- 使用`manage_memory()`调整批处理大小，避免内存溢出
- 对于计算密集型任务，启用`enable_parallel()`进行并行处理

```python
# 性能优化示例
dpcs = DualPathCoordinationSystem()
dpcs.optimize_model()
batch_size = dpcs.manage_memory(input_size=256)
dpcs.enable_parallel()
```

### 10.2 错误处理

- 总是使用try-except捕获可能的异常
- 对于特定类型的错误使用专门的处理方式
- 记录错误信息以便调试

```python
try:
    result = dpcs.process(data)
except Exception as e:
    logging.error(f"处理错误: {e}")
    # 适当的错误恢复或回退策略
```

### 10.3 资源管理

- 处理完成后释放不需要的资源
- 对于大模型，考虑使用上下文管理器控制生命周期
- 合理设置区块链的批处理大小

```python
# 资源管理示例
dpcs.optimize_blockchain_operations()
# 处理完成后
gc.collect()
torch.cuda.empty_cache()
```

## 11. 示例

### 11.1 基本处理流程

```python
from dpcs.system import DualPathCoordinationSystem

# 初始化系统
dpcs = DualPathCoordinationSystem()

# 处理文本输入
text_input = "这是一个示例文本，需要进行处理和分析。"
result = dpcs.process(text_input, input_type="text")
print(f"处理结果: {result}")

# 处理结构化数据
structured_data = {
    "numeric_features": [1.2, 3.4, 5.6],
    "categorical_features": ["A", "B"],
    "time_series": [10, 20, 30, 40, 50]
}
result = dpcs.process(structured_data, input_type="dict")
print(f"处理结果: {result}")
```

### 11.2 区块链集成示例

```python
from dpcs.system import DualPathCoordinationSystem
from dpcs.blockchain.da_layer import DataAvailabilityLayer
from dpcs.blockchain.rollup_layer import RollupLayer

# 初始化区块链组件
da_layer = DataAvailabilityLayer(shard_count=16)
rollup_layer = RollupLayer(rollup_type="zk", batch_size=100, da_layer=da_layer)

# 初始化DPCS系统
config = {
    'use_blockchain': True,
    'da_shards': 16,
    'rollup_type': 'zk',
    'rollup_batch_size': 100
}
dpcs = DualPathCoordinationSystem(config)

# 处理数据并记录到区块链
for i in range(10):
    data = f"Sample data {i}"
    result = dpcs.process(data, input_type="text")
    
    # 查看区块链记录
    if i % 5 == 0 and dpcs.rollup_layer:
        batch_result = dpcs.rollup_layer.process_batch()
        print(f"批处理结果: {batch_result}")
```

### 11.3 运行外部示例

```python
from dpcs.examples.case_adapter import CaseAdapter
from dpcs.system import DualPathCoordinationSystem

# 初始化系统
dpcs = DualPathCoordinationSystem()

# 创建案例适配器
adapter = CaseAdapter(dpcs_system=dpcs)

# 运行库存管理示例
print("运行基本库存示例:")
adapter.run_case("examples/basic_usage.py")

# 运行预测示例，使用左脑路径
print("\n运行预测示例 (左脑路径):")
adapter.run_case("examples/forecast_example.py", mode="left")

# 运行成本跟踪示例，不使用区块链
print("\n运行成本跟踪示例 (无区块链):")
adapter.run_case("examples/cost_tracking_example.py", use_blockchain=False)
```

## 12. API更新与版本控制

DPCS-B API遵循语义化版本控制，版本号格式为X.Y.Z：

- X：主版本号，不兼容的API更改
- Y：次版本号，向后兼容的功能性新增
- Z：修订号，向后兼容的问题修正

API的更新和变更将在每个版本的发布说明中详细记录。开发者应定期查看版本发布说明以了解API的最新变化。