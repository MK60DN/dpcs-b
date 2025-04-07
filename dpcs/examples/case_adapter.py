"""
案例适配器 - 使DPCS-B框架能够直接运行库存管理示例文件
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path

# 添加父目录到路径，以便导入DPCS-B模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.system import DualPathCoordinationSystem


class CaseAdapter:
    """案例适配器，将外部示例适配到DPCS-B框架"""

    def __init__(self, dpcs_system=None):
        """初始化适配器"""
        self.dpcs = dpcs_system or DualPathCoordinationSystem()
        self.examples_dir = Path("./examples")
        self.module_mapping = {}
        self.initialized_modules = False

    def initialize_module_mapping(self):
        """初始化模块映射"""
        if self.initialized_modules:
            return

        # 预定义的库存管理模块映射
        self.module_mapping = {
            "inventory_system": {"path": "inventory_system", "class": "InventorySystem"},
            "inventory_control": {"path": "inventory_control", "class": "InventoryControl"},
            "forecast_system": {"path": "forecast_system", "class": "InventorySystemWithForecast"},
            "cost_tracking": {"path": "cost_tracking", "class": "CostTrackingAgent"},
            "distributed_agent": {"path": "distributed_agent", "class": "DistributedInventoryAgent"},
            "port_demand": {"path": "port_demand", "class": "PortDemand"}
        }

        # 动态检测examples目录中的模块
        if self.examples_dir.exists():
            for py_file in self.examples_dir.glob("*.py"):
                module_name = py_file.stem

                # 如果不是以"__"开头的Python文件
                if not module_name.startswith("__"):
                    # 尝试导入模块并查找类
                    try:
                        spec = importlib.util.spec_from_file_location(f"examples.{module_name}", py_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # 查找模块中定义的类
                        classes = inspect.getmembers(module, inspect.isclass)
                        if classes:
                            # 使用第一个找到的类
                            class_name = classes[0][0]
                            self.module_mapping[module_name] = {
                                "path": module_name,
                                "class": class_name
                            }
                    except Exception:
                        # 如果导入失败，跳过
                        pass

        self.initialized_modules = True

    def setup_paths(self):
        """设置路径以便能够导入示例模块"""
        if str(self.examples_dir) not in sys.path:
            sys.path.insert(0, str(self.examples_dir))

        # 确保examples目录存在
        self.examples_dir.mkdir(exist_ok=True)

        # 确保examples/__init__.py存在
        init_file = self.examples_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    def load_module(self, module_name):
        """加载指定的模块"""
        self.initialize_module_mapping()
        self.setup_paths()

        if module_name not in self.module_mapping:
            raise ImportError(f"未知模块: {module_name}")

        module_info = self.module_mapping[module_name]
        module_path = module_info["path"]

        # 尝试从examples目录导入
        try:
            module = importlib.import_module(f"examples.{module_path}")
            return module
        except ImportError:
            # 如果导入失败，创建模块存根
            return self._create_module_stub(module_name, module_info)

    def _create_module_stub(self, module_name, module_info):
        """创建模块存根"""
        class_name = module_info["class"]

        # 创建一个空的模块对象
        module = type('module', (), {})

        # 根据模块类型创建不同的存根类
        if module_name == "inventory_system":
            module.InventorySystem = self._create_inventory_system_stub()
        elif module_name == "inventory_control":
            module.InventoryControl = self._create_inventory_control_stub()
        elif module_name == "forecast_system":
            module.InventorySystemWithForecast = self._create_forecast_system_stub()
        elif module_name == "cost_tracking":
            module.CostTrackingAgent = self._create_cost_tracking_stub()
        elif module_name == "distributed_agent":
            module.DistributedInventoryAgent = self._create_distributed_agent_stub()
        elif module_name == "port_demand":
            module.PortDemand = self._create_port_demand_stub()
        else:
            # 创建通用存根
            setattr(module, class_name, type(class_name, (), {}))

        return module

    def _create_inventory_system_stub(self):
        """创建库存系统存根"""
        from collections import deque

        class InventorySystemStub:
            def __init__(self):
                self.demands = deque()
                self.backOff = 0
                self.backOffAmount = 0
                self.fulFilled = 0
                self.fulFilledAmount = 0
                self.serviceLevel = 0
                self.inventory = 100
                self.inTransit = 50

            def checkDemand(self):
                # 调用DPCS-B处理需求
                dpcs_input = {
                    "type": "inventory_check",
                    "demands": list(self.demands),
                    "inventory": self.inventory,
                    "inTransit": self.inTransit
                }

                result = self.dpcs.process(dpcs_input)

                # 更新状态
                if isinstance(result, dict):
                    for key in ['inventory', 'inTransit', 'fulFilled', 'fulFilledAmount',
                                'backOff', 'backOffAmount', 'serviceLevel']:
                        if key in result:
                            setattr(self, key, result[key])

                # 原始实现保留作为后备
                if not self.demands:
                    print("订单队列为空，结束处理。")
                    return

                while self.demands:
                    order_demand = self.demands.popleft()
                    print(f"处理订单需求: {order_demand}")

                    if order_demand <= self.inventory:
                        self.inventory -= order_demand
                        self.fulFilled += 1
                        self.fulFilledAmount += order_demand
                        print(f"库存满足订单需求，当前库存: {self.inventory}")
                    else:
                        if order_demand <= self.inventory + self.inTransit:
                            remaining_demand = order_demand - self.inventory
                            self.inventory = 0
                            self.inTransit -= remaining_demand
                            self.fulFilled += 1
                            self.fulFilledAmount += order_demand
                            print(f"部分需求由在途库存满足，当前在途库存: {self.inTransit}")
                        else:
                            shortage_amount = order_demand - (self.inventory + self.inTransit)
                            self.backOff += 1
                            self.backOffAmount += shortage_amount
                            self.inventory = 0
                            self.inTransit = 0
                            print(f"无法满足订单，缺货量: {shortage_amount}")

                self.serviceLevel = self.fulFilled / (self.fulFilled + self.backOff) if (
                                                                                                    self.fulFilled + self.backOff) > 0 else 0
                print(f"更新服务水平: {self.serviceLevel}")

            def addOrder(self, demand):
                self.demands.append(demand)
                print(f"添加新订单需求: {demand}")

            def getInventoryStatus(self):
                print(f"当前库存: {self.inventory}, 在途库存: {self.inTransit}")

        # 将DPCS实例注入存根类
        setattr(InventorySystemStub, 'dpcs', self.dpcs)

        return InventorySystemStub

    def _create_inventory_control_stub(self):
        """创建库存控制存根"""

        class InventoryControlStub:
            def __init__(self, strategy="RS"):
                self.inventory = 100
                self.inTransit = 50
                self.maxInventory = 200
                self.reorderPoint = 50
                self.reorderPointQ = 60
                self.orderQuantity = 30
                self.replenishmentLeadTime = 5
                self.currentStrategy = strategy

            def RS(self):
                # 调用DPCS-B处理RS策略
                dpcs_input = {
                    "type": "inventory_control",
                    "strategy": "RS",
                    "inventory": self.inventory,
                    "reorderPoint": self.reorderPoint,
                    "maxInventory": self.maxInventory
                }

                result = self.dpcs.process(dpcs_input)

                # 更新状态
                if isinstance(result, dict) and 'inventory' in result:
                    self.inventory = result['inventory']
                    if 'inTransit' in result:
                        self.inTransit = result['inTransit']

                # 原始实现保留作为后备
                if self.inventory < self.reorderPoint:
                    order_amount = self.maxInventory - self.inventory
                    self.inventory += order_amount
                    self.inTransit += order_amount
                    print(f"(R,S)策略：库存低于订货点，订货量: {order_amount}")

            def QR(self):
                # 调用DPCS-B处理QR策略
                dpcs_input = {
                    "type": "inventory_control",
                    "strategy": "QR",
                    "inventory": self.inventory,
                    "reorderPointQ": self.reorderPointQ,
                    "orderQuantity": self.orderQuantity
                }

                result = self.dpcs.process(dpcs_input)

                # 更新状态
                if isinstance(result, dict) and 'inventory' in result:
                    self.inventory = result['inventory']
                    if 'inTransit' in result:
                        self.inTransit = result['inTransit']

                # 原始实现保留作为后备
                if self.inventory < self.reorderPointQ:
                    self.inventory += self.orderQuantity
                    self.inTransit += self.orderQuantity
                    print(f"(Q,R)策略：库存低于订货点，固定订货量: {self.orderQuantity}")

            def controlInventory(self):
                if self.currentStrategy == "RS":
                    self.RS()
                elif self.currentStrategy == "QR":
                    self.QR()
                else:
                    print("未知策略，无法执行库存控制。")

        # 将DPCS实例注入存根类
        setattr(InventoryControlStub, 'dpcs', self.dpcs)

        return InventoryControlStub

    def _create_forecast_system_stub(self):
        """创建预测系统存根"""
        from collections import deque

        class InventorySystemWithForecastStub:
            def __init__(self, forecast_period=5, shortage_threshold=20):
                self.orderHistory = deque()
                self.forecastPeriod = forecast_period
                self.inventory = 100
                self.inTransit = 50
                self.shortageThreshold = shortage_threshold

            def updateOrderHistory(self, demand):
                # 调用DPCS-B处理预测
                dpcs_input = {
                    "type": "forecast_update",
                    "demand": demand,
                    "current_history": list(self.orderHistory),
                    "forecast_period": self.forecastPeriod
                }

                result = self.dpcs.process(dpcs_input)

                # 更新状态
                if isinstance(result, dict) and 'updated_history' in result:
                    self.orderHistory = deque(result['updated_history'])
                else:
                    # 原始实现保留作为后备
                    if len(self.orderHistory) >= self.forecastPeriod:
                        self.orderHistory.popleft()
                    self.orderHistory.append(demand)

            def forecastDemand(self):
                # 调用DPCS-B处理预测
                dpcs_input = {
                    "type": "forecast_demand",
                    "history": list(self.orderHistory),
                    "forecast_period": self.forecastPeriod
                }

                result = self.dpcs.process(dpcs_input)

                # 使用结果
                if isinstance(result, dict) and 'forecast' in result:
                    forecast = result['forecast']
                    print(f"预测需求量: {forecast}")
                    return forecast

                # 原始实现保留作为后备
                if len(self.orderHistory) < self.forecastPeriod:
                    print("历史数据不足，无法预测。")
                    return

                avg_demand = sum(self.orderHistory) / self.forecastPeriod
                print(f"预测需求量: {avg_demand}")
                return avg_demand

        # 将DPCS实例注入存根类
        setattr(InventorySystemWithForecastStub, 'dpcs', self.dpcs)

        return InventorySystemWithForecastStub

    def _create_cost_tracking_stub(self):
        """创建成本跟踪存根"""
        import matplotlib.pyplot as plt

        class CostTrackingAgentStub:
            def __init__(self, name, order_cost_per_unit, holding_cost_per_unit, stockout_cost_per_unit):
                self.name = name
                self.orderCostPerUnit = order_cost_per_unit
                self.holdingCostPerUnit = holding_cost_per_unit
                self.stockoutCostPerUnit = stockout_cost_per_unit
                self.totalCost = 0
                self.costHistory = []

            def updateTotalCost(self, order_quantity, shortage_quantity):
                # 调用DPCS-B处理成本计算
                dpcs_input = {
                    "type": "cost_calculation",
                    "name": self.name,
                    "order_quantity": order_quantity,
                    "shortage_quantity": shortage_quantity,
                    "order_cost_per_unit": self.orderCostPerUnit,
                    "holding_cost_per_unit": self.holdingCostPerUnit,
                    "stockout_cost_per_unit": self.stockoutCostPerUnit
                }

                result = self.dpcs.process(dpcs_input)

                # 更新状态
                if isinstance(result, dict) and 'total_cost' in result:
                    self.totalCost = result['total_cost']
                    self.costHistory.append(self.totalCost)
                else:
                    # 原始实现保留作为后备
                    order_cost = self.orderCostPerUnit * order_quantity
                    holding_cost = self.holdingCostPerUnit * 100
                    stockout_cost = self.stockoutCostPerUnit * shortage_quantity

                    self.totalCost = order_cost + holding_cost + stockout_cost
                    self.costHistory.append(self.totalCost)

            def plotCostHistory(self):
                plt.plot(self.costHistory, label=f"{self.name} Total Cost")
                plt.xlabel("Time Period")
                plt.ylabel("Total Cost")
                plt.title(f"Cost History of {self.name}")
                plt.legend()
                plt.show()

        # 将DPCS实例注入存根类
        setattr(CostTrackingAgentStub, 'dpcs', self.dpcs)

        return CostTrackingAgentStub

    def _create_distributed_agent_stub(self):
        """创建分布式库存代理存根"""
        import random

        class DistributedInventoryAgentStub:
            def __init__(self, name, forecast_demand, inventory):
                self.name = name
                self.forecastDemand = forecast_demand
                self.inventory = inventory

            def request(self, shortage_amount):
                # 调用DPCS-B处理请求
                dpcs_input = {
                    "type": "transshipment_request",
                    "name": self.name,
                    "shortage_amount": shortage_amount
                }

                result = self.dpcs.process(dpcs_input)

                # 原始实现保留作为后备
                print(f"{self.name} 发送调拨请求: {shortage_amount}")

                return result

            def checkTransship(self, request):
                # 调用DPCS-B处理调拨
                dpcs_input = {
                    "type": "transshipment_check",
                    "name": self.name,
                    "request": request,
                    "inventory": self.inventory
                }

                result = self.dpcs.process(dpcs_input)

                # 如果有结果，使用它
                if isinstance(result, dict) and 'offer' in result:
                    return result['offer']

                # 原始实现保留作为后备
                offer = {
                    "supplier_name": self.name,
                    "offer_amount": min(self.inventory, request),
                    "offer_price": random.uniform(10, 20)
                }

                print(f"{self.name} 发送调拨反馈: {offer}")
                return offer

        # 将DPCS实例注入存根类
        setattr(DistributedInventoryAgentStub, 'dpcs', self.dpcs)

        return DistributedInventoryAgentStub

    def _create_port_demand_stub(self):
        """创建需求端口存根"""

        class PortDemandStub:
            def __init__(self):
                self.demand_queue = []

            def receive_demand(self, demand):
                # 调用DPCS-B处理需求接收
                dpcs_input = {
                    "type": "port_demand",
                    "demand": demand
                }

                result = self.dpcs.process(dpcs_input)

                # 更新需求队列
                self.demand_queue.append(demand)

                # 原始实现保留作为后备
                print(f"接收到需求订单：{demand} 件")

        # 将DPCS实例注入存根类
        setattr(PortDemandStub, 'dpcs', self.dpcs)

        return PortDemandStub

    def run_case(self, main_file_path, mode="dual", use_blockchain=True):
        """运行一个案例文件"""
        self.setup_paths()

        # 设置DPCS配置
        if hasattr(self.dpcs, 'set_mode'):
            self.dpcs.set_mode(mode)

        if hasattr(self.dpcs, 'set_blockchain'):
            self.dpcs.set_blockchain(use_blockchain)

        # 确保路径是Path对象
        if isinstance(main_file_path, str):
            main_file_path = Path(main_file_path)

        # 如果是相对路径，相对于examples目录
        if not main_file_path.is_absolute():
            main_file_path = self.examples_dir / main_file_path

        # 检查文件是否存在
        if not main_file_path.exists():
            raise FileNotFoundError(f"找不到主文件: {main_file_path}")

        # 添加模块重定向
        sys.meta_path.insert(0, self)

        try:
            # 动态导入主模块
            module_name = main_file_path.stem
            spec = importlib.util.spec_from_file_location(f"examples.{module_name}", main_file_path)
            main_module = importlib.util.module_from_spec(spec)
            sys.modules[f"examples.{module_name}"] = main_module

            # 执行模块
            spec.loader.exec_module(main_module)

            # 调用main函数（如果存在）
            if hasattr(main_module, "main"):
                main_module.main()
            else:
                print(f"警告: {main_file_path}中未找到main()函数")

            return True

        except Exception as e:
            print(f"运行案例时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # 移除模块重定向
            if self in sys.meta_path:
                sys.meta_path.remove(self)

    # 实现importlib.abc.MetaPathFinder和importlib.abc.Loader的接口
    def find_spec(self, fullname, path, target=None):
        """查找模块规范"""
        # 只处理examples包下的模块
        if not fullname.startswith("examples."):
            return None

        # 获取模块名
        module_name = fullname.split(".")[-1]

        # 检查是否是我们要处理的模块
        self.initialize_module_mapping()
        for mapped_name in self.module_mapping:
            if module_name == mapped_name or module_name == self.module_mapping[mapped_name]["path"]:
                # 创建和返回ModuleSpec
                return importlib.machinery.ModuleSpec(fullname, self)

        return None

    def create_module(self, spec):
        """创建模块"""
        # 让importlib创建默认模块
        return None

    def exec_module(self, module):
        """执行模块"""
        # 获取模块名
        fullname = module.__name__
        module_name = fullname.split(".")[-1]

        # 加载对应的模块或存根
        for mapped_name, info in self.module_mapping.items():
            if module_name == mapped_name or module_name == info["path"]:
                # 加载模块
                loaded_module = self.load_module(mapped_name)

                # 复制所有属性到目标模块
                for name in dir(loaded_module):
                    if not name.startswith("__"):
                        setattr(module, name, getattr(loaded_module, name))

                break


def main():
    """主函数，用于测试案例适配器"""
    adapter = CaseAdapter()

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="DPCS-B库存管理案例适配器")
    parser.add_argument('file', nargs='?', help="要运行的案例文件路径")
    parser.add_argument('--mode', choices=['left', 'right', 'dual'], default='dual', help="DPCS-B处理模式")
    parser.add_argument('--no-blockchain', action='store_true', help="禁用区块链")

    args = parser.parse_args()

    if args.file:
        # 运行指定的案例文件
        adapter.run_case(args.file, mode=args.mode, use_blockchain=not args.no_blockchain)
    else:
        print("请指定要运行的案例文件")
        print("示例: python -m dpcs.examples.case_adapter examples/basic_usage.py")


if __name__ == "__main__":
    main()