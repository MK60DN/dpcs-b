# DPCS-B 示例文档

本文档详细介绍了DPCS-B框架提供的示例程序，包括每个示例的功能、使用方法和核心代码解析。

## 1. 概述

DPCS-B框架提供了一系列示例，展示如何在不同应用场景中使用该框架。这些示例从简单到复杂，帮助用户快速上手并理解系统的核心功能。

## 2. 示例列表

### 2.1 基本使用示例 (basic_usage.py)

这个示例展示了库存系统的基本功能，包括添加订单、处理需求和查看库存状态。

#### 2.1.1 功能介绍

- 创建库存系统实例
- 添加和处理订单需求
- 检查库存状态和服务水平

#### 2.1.2 使用方法

```bash
# 命令行运行
dpcs examples/basic_usage.py

# 或者在Python中导入
from examples.basic_usage import main
main()
```

#### 2.1.3 核心代码解析

```python
# 创建库存系统实例
inventory_system = InventorySystem()

# 添加订单需求
inventory_system.addOrder(30)
inventory_system.addOrder(50)
inventory_system.addOrder(70)

# 处理订单需求
inventory_system.checkDemand()

# 检查库存状态
inventory_system.getInventoryStatus()
```

#### 2.1.4 DPCS-B集成

当通过DPCS-B框架运行时，以下过程会发生：

1. 输入数据经由中脑模块分析，选择合适的处理路径
2. 库存计算由左脑结构化模块处理
3. 输出结果由额叶模块整合
4. 所有数据和计算过程可选择性地记录到区块链

### 2.2 库存控制策略示例 (inventory_control_example.py)

这个示例展示了不同的库存控制策略((R,S)和(Q,R))及其效果对比。

#### 2.2.1 功能介绍

- 演示(R,S)策略：当库存低于订货点时，补货到最大库存量
- 演示(Q,R)策略：当库存低于订货点时，固定订货量补货
- 比较两种策略的效果

#### 2.2.2 使用方法

```bash
# 命令行运行
dpcs examples/inventory_control_example.py

# 或者在Python中导入
from examples.inventory_control_example import main
main()
```

#### 2.2.3 核心代码解析

```python
# 创建(R,S)策略控制器
rs_control = InventoryControl(strategy="RS")

# 模拟库存消耗
rs_control.inventory = 40  # 低于订货点

# 应用库存控制策略
rs_control.controlInventory()

# 创建(Q,R)策略控制器
qr_control = InventoryControl(strategy="QR")

# 应用库存控制策略
qr_control.controlInventory()
```

#### 2.2.4 DPCS-B集成

DPCS-B框架对该示例的增强：

1. 策略选择过程由左脑模块优化
2. 决策计算更精确，考虑多种因素
3. 结果可通过区块链验证，确保策略执行的透明性和一致性

### 2.3 需求预测示例 (forecast_example.py)

这个示例展示了如何使用需求预测系统基于历史数据预测未来需求。

#### 2.3.1 功能介绍

- 使用历史订单数据进行需求预测
- 处理异常数据的影响
- 分析预测准确性

#### 2.3.2 使用方法

```bash
# 命令行运行
dpcs examples/forecast_example.py

# 或者在Python中导入
from examples.forecast_example import main
main()
```

#### 2.3.3 核心代码解析

```python
# 创建预测系统
forecast_system = InventorySystemWithForecast(forecast_period=3)

# 添加历史数据
for demand in [45, 52, 48]:
    forecast_system.updateOrderHistory(demand)

# 进行预测
forecast_system.forecastDemand()
```

#### 2.3.4 DPCS-B集成

DPCS-B框架对预测功能的增强：

1. 左脑模块提供先进的时间序列分析
2. 右脑模块可处理文本描述的市场因素
3. 胼胝体模块融合两种分析方法
4. 小脑模块确保时序数据的正确处理
5. 预测结果可通过区块链跟踪其准确性历史

### 2.4 成本跟踪示例 (cost_tracking_example.py)

这个示例展示了如何跟踪和分析不同类型的库存成本。

#### 2.4.1 功能介绍

- 跟踪订货成本、持有成本和缺货成本
- 可视化成本历史和结构
- 分析成本趋势

#### 2.4.2 使用方法

```bash
# 命令行运行
dpcs examples/cost_tracking_example.py

# 或者在Python中导入
from examples.cost_tracking_example import main
main()
```

#### 2.4.3 核心代码解析

```python
# 创建成本跟踪代理
cost_agent = CostTrackingAgent(
    name="配送中心1",
    order_cost_per_unit=10,
    holding_cost_per_unit=2,
    stockout_cost_per_unit=50
)

# 更新成本
cost_agent.updateTotalCost(order_quantity=50, shortage_quantity=0)

# 绘制成本历史
cost_agent.plotCostHistory()
```

#### 2.4.4 DPCS-B集成

DPCS-B框架对成本分析的增强：

1. 使用左脑模块进行精确的成本计算和优化
2. 使用右脑模块生成详细的成本分析报告
3. 通过区块链记录所有成本数据，确保审计可追溯性

### 2.5 分布式代理示例 (distributed_agents_example.py)

这个示例展示了多个库存点之间的调拨和协作机制。

#### 2.5.1 功能介绍

- 模拟多个库存点之间的协作
- 实现库存调拨机制
- 处理调拨价格选择

#### 2.5.2 使用方法

```bash
# 命令行运行
dpcs examples/distributed_agents_example.py

# 或者在Python中导入
from examples.distributed_agents_example import main
main()
```

#### 2.5.3 核心代码解析

```python
# 创建多个分布式库存代理
agents = [
    DistributedInventoryAgent(name="仓库A", forecast_demand=50, inventory=100),
    DistributedInventoryAgent(name="仓库B", forecast_demand=70, inventory=80),
    DistributedInventoryAgent(name="仓库C", forecast_demand=60, inventory=120)
]

# 发送调拨请求
shortage = current_demand - agent.inventory
agent.request(shortage)

# 其他代理响应调拨请求
offers = []
for other_agent in agents:
    if other_agent != agent:
        offer = other_agent.checkTransship(shortage)
        offers.append(offer)

# 选择最佳调拨方案
best_offer = min(offers, key=lambda x: x['offer_price'])
```

#### 2.5.4 DPCS-B集成

DPCS-B框架对分布式代理的增强：

1. 中脑模块帮助选择最佳调拨决策
2. 使用胼胝体模块在结构化数据和语义理解间切换
3. 通过区块链记录所有调拨交易，确保透明性
4. 使用小脑模块协调多个代理的时序动作

### 2.6 综合系统示例 (integrated_system_example.py)

这个示例整合了所有组件，构建完整的库存管理解决方案。

#### 2.6.1 功能介绍

- 集成库存系统、控制策略、预测和成本跟踪
- 比较不同库存策略的性能
- 提供完整的可视化分析

#### 2.6.2 使用方法

```bash
# 命令行运行
dpcs examples/integrated_system_example.py

# 或者在Python中导入
from examples.integrated_system_example import main
main()
```

#### 2.6.3 核心代码解析

```python
# 创建综合系统
rs_system = IntegratedInventorySystem("RS策略中心", control_strategy="RS")
qr_system = IntegratedInventorySystem("QR策略中心", control_strategy="QR")

# 处理需求
for period in range(1, num_periods + 1):
    # 生成随机需求
    demand = generate_demand(period)
    
    # 两个系统处理相同需求
    rs_result = rs_system.process_demand(demand)
    qr_result = qr_system.process_demand(demand)
    
    # 对比结果
    compare_results(rs_result, qr_result)

# 绘制对比图表
plot_comparison(rs_system, qr_system)
```

#### 2.6.4 DPCS-B集成

DPCS-B框架对综合系统的增强：

1. 双路径协调系统处理复杂的决策流程
2. 区块链组件记录整个流程中的数据和决策
3. 多模块协作提升系统性能和可解释性

### 2.7 交互式模拟GUI (interactive_simulation.py)

这个示例提供了一个图形用户界面，允许用户自定义参数并模拟库存管理过程。

#### 2.7.1 功能介绍

- 用户可自定义库存参数、成本参数和需求参数
- 实时显示模拟结果和图表
- 支持不同策略的对比

#### 2.7.2 使用方法

```bash
# 命令行运行
dpcs examples/interactive_simulation.py

# 或者在Python中导入
from examples.interactive_simulation import main
main()
```

#### 2.7.3 核心代码解析

```python
# 创建GUI应用
class InventorySimulationGUI:
    def __init__(self, root):
        # 设置GUI组件
        self.setup_ui()
        
    def start_simulation(self):
        # 获取用户设置的参数
        parameters = self.get_parameters()
        
        # 运行模拟
        results = self.run_simulation(parameters)
        
        # 更新图表
        self.update_charts(results)
        
    def run_simulation(self, parameters):
        # 初始化系统
        inventory_system = InventorySystem()
        control_system = InventoryControl(strategy=parameters['strategy'])
        
        # 运行模拟
        # ...
        
        return results
```

#### 2.7.4 DPCS-B集成

DPCS-B框架对交互式模拟的增强：

1. 双路径处理提供更丰富的模拟能力
2. 用户可以选择不同的处理模式
3. 区块链组件可选择性地记录模拟过程
4. 结果通过左右脑路径分别处理，提供多角度分析

## 3. 自定义示例指南

### 3.1 创建自己的示例

您可以基于提供的示例创建自己的库存管理应用。步骤如下：

1. 创建新的Python文件，放在`examples/`目录下
2. 导入需要的模块
3. 实现主要功能
4. 定义`main()`函数作为入口点

```python
# examples/my_custom_example.py
from dpcs.modules.inventory_system import InventorySystem
from dpcs.modules.inventory_control import InventoryControl
# 导入其他需要的模块

def main():
    # 实现自定义功能
    
if __name__ == "__main__":
    main()
```

### 3.2 上传和运行自定义示例

有两种方式上传和运行自定义示例：

#### 3.2.1 通过命令行

```bash
# 直接运行自定义示例
dpcs path/to/my_custom_example.py
```

#### 3.2.2 通过图形界面

1. 运行`dpcs --gui`启动图形界面
2. 点击"浏览"按钮上传自定义示例文件
3. 选择主执行文件
4. 配置处理模式和区块链设置
5. 点击"运行"按钮

### 3.3 集成DPCS-B功能

要充分利用DPCS-B框架的功能，可以在自定义示例中添加以下代码：

```python
# 导入DPCS-B组件
from dpcs.system import DualPathCoordinationSystem

def main():
    # 创建DPCS-B系统实例
    dpcs = DualPathCoordinationSystem()
    
    # 配置处理模式
    dpcs.set_mode("dual")  # 或 "left" 或 "right"
    
    # 使用DPCS-B处理输入
    result = dpcs.process("输入数据", input_type="text")
    
    # 使用处理结果
    print(f"处理结果: {result}")
```

## 4. 示例性能优化

所有示例都可以通过DPCS-B框架获得性能优化。以下是一些优化技巧：

### 4.1 计算优化

```python
# 优化模型计算
dpcs.optimize_model()
```

### 4.2 内存管理

```python
# 优化内存使用
batch_size = dpcs.manage_memory(input_size=256)
```

### 4.3 并行处理

```python
# 启用并行处理
dpcs.enable_parallel()
```

### 4.4 区块链优化

```python
# 优化区块链操作
dpcs.optimize_blockchain_operations()
```

## 5. 示例故障排除

### 5.1 常见问题

1. **示例运行失败**
   - 检查依赖是否正确安装
   - 确保文件路径正确
   - 查看错误日志

2. **性能问题**
   - 使用`optimize_model()`方法
   - 减少批处理大小
   - 检查内存使用

3. **区块链相关问题**
   - 检查区块链配置
   - 可能需要增加批处理大小
   - 考虑禁用区块链进行测试

### 5.2 获取帮助

如果遇到无法解决的问题，可以：

1. 查阅完整文档
2. 在GitHub仓库提交issue
3. 联系DPCS-B开发团队获取支持