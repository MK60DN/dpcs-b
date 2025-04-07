"""
综合示例 - 整合所有组件构建完整的库存管理系统
"""

from dpcs.modules.inventory_system import InventorySystem
from dpcs.modules.inventory_control import InventoryControl
from dpcs.modules.forecast_system import InventorySystemWithForecast
from dpcs.modules.cost_tracking import CostTrackingAgent
from dpcs.modules.distributed_agent import DistributedInventoryAgent
from dpcs.modules.port_demand import PortDemand

import random
import matplotlib.pyplot as plt
import numpy as np


class IntegratedInventorySystem:
    def __init__(self, location_name, control_strategy="RS"):
        """初始化综合库存系统"""
        self.name = location_name
        self.base_system = InventorySystem()
        self.control = InventoryControl(strategy=control_strategy)
        self.forecast = InventorySystemWithForecast(forecast_period=5)
        self.cost_agent = CostTrackingAgent(
            name=location_name,
            order_cost_per_unit=10,
            holding_cost_per_unit=2,
            stockout_cost_per_unit=50
        )
        self.distributed_agent = DistributedInventoryAgent(
            name=location_name,
            forecast_demand=60,
            inventory=100
        )

        # 同步初始库存
        self.sync_inventory()

        # 跟踪历史数据
        self.demand_history = []
        self.inventory_history = []
        self.order_history = []
        self.service_level_history = []

    def sync_inventory(self):
        """同步各组件间的库存数据"""
        inventory = self.base_system.inventory
        self.control.inventory = inventory
        self.forecast.inventory = inventory
        self.distributed_agent.inventory = inventory

    def process_demand(self, demand):
        """处理订单需求"""
        print(f"\n{self.name} - 处理需求: {demand}")

        # 更新需求历史和预测
        self.demand_history.append(demand)
        self.forecast.updateOrderHistory(demand)

        # 添加订单到基础系统
        self.base_system.addOrder(demand)
        self.base_system.checkDemand()

        # 使用库存控制策略
        self.control.inventory = self.base_system.inventory
        self.control.controlInventory()
        order_quantity = 0

        # 计算订货量
        if self.control.currentStrategy == "RS":
            if self.base_system.inventory < self.control.reorderPoint:
                order_quantity = self.control.maxInventory - self.base_system.inventory
        else:  # QR策略
            if self.base_system.inventory < self.control.reorderPointQ:
                order_quantity = self.control.orderQuantity

        # 记录订货
        self.order_history.append(order_quantity)

        # 计算缺货量
        shortage_quantity = self.base_system.backOffAmount

        # 更新成本
        self.cost_agent.updateTotalCost(order_quantity, shortage_quantity)

        # 更新库存
        self.base_system.inventory += order_quantity

        # 同步库存数据
        self.sync_inventory()

        # 更新历史记录
        self.inventory_history.append(self.base_system.inventory)
        self.service_level_history.append(self.base_system.serviceLevel)

        # 预测下一期需求
        if len(self.demand_history) >= self.forecast.forecastPeriod:
            forecast_demand = sum(self.demand_history[-self.forecast.forecastPeriod:]) / self.forecast.forecastPeriod
            self.distributed_agent.forecastDemand = forecast_demand
            print(f"{self.name} - 预测下一期需求: {forecast_demand:.2f}")

        # 返回处理结果
        return {
            "satisfied": demand - shortage_quantity,
            "shortage": shortage_quantity,
            "order_placed": order_quantity,
            "current_inventory": self.base_system.inventory,
            "service_level": self.base_system.serviceLevel
        }

    def plot_history(self):
        """绘制历史数据图表"""
        periods = range(1, len(self.demand_history) + 1)

        plt.figure(figsize=(12, 10))

        # 绘制需求和库存历史
        plt.subplot(3, 1, 1)
        plt.plot(periods, self.demand_history, 'r-', marker='o', label='需求')
        plt.plot(periods, self.inventory_history, 'b-', marker='s', label='库存')
        plt.plot(periods, self.order_history, 'g--', marker='^', label='订货量')
        plt.title(f"{self.name} - 需求、库存和订货历史")
        plt.xlabel("周期")
        plt.ylabel("数量")
        plt.grid(True)
        plt.legend()

        # 绘制服务水平历史
        plt.subplot(3, 1, 2)
        plt.plot(periods, self.service_level_history, 'purple', marker='d')
        plt.title(f"{self.name} - 服务水平历史")
        plt.xlabel("周期")
        plt.ylabel("服务水平")
        plt.grid(True)
        plt.ylim(0, 1.1)

        # 绘制成本历史
        plt.subplot(3, 1, 3)
        plt.plot(periods, self.cost_agent.costHistory, 'orange', marker='*')
        plt.title(f"{self.name} - 成本历史")
        plt.xlabel("周期")
        plt.ylabel("成本")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_summary(self):
        """获取系统运行摘要"""
        summary = {
            "name": self.name,
            "total_demand": sum(self.demand_history),
            "final_inventory": self.base_system.inventory,
            "avg_service_level": np.mean(self.service_level_history) if self.service_level_history else 0,
            "total_cost": self.cost_agent.totalCost,
            "total_orders": sum(self.order_history),
            "control_strategy": self.control.currentStrategy
        }
        return summary


def main():
    # 创建两个使用不同策略的综合库存系统
    rs_system = IntegratedInventorySystem("RS策略中心", control_strategy="RS")
    qr_system = IntegratedInventorySystem("QR策略中心", control_strategy="QR")

    # 创建共享的需求端口
    port = PortDemand()

    # 模拟20个周期
    num_periods = 20
    print(f"开始模拟 {num_periods} 个周期...")

    for period in range(1, num_periods + 1):
        print(f"\n===== 第 {period} 期 =====")

        # 生成随机需求 (包含一些季节性波动)
        base_demand = 60
        seasonality = 20 * np.sin(period * np.pi / 10)  # 周期性波动
        noise = random.randint(-10, 10)  # 随机波动
        demand = max(10, int(base_demand + seasonality + noise))

        # 接收需求
        port.receive_demand(demand)
        current_demand = port.demand_queue[-1]
        print(f"当期需求: {current_demand}")

        # 两个系统处理相同的需求
        rs_result = rs_system.process_demand(current_demand)
        qr_result = qr_system.process_demand(current_demand)

        # 显示处理结果对比
        print("\n本期结果对比:")
        print(
            f"RS策略中心 - 满足: {rs_result['satisfied']}/{current_demand}, 缺货: {rs_result['shortage']}, 订货: {rs_result['order_placed']}, 库存: {rs_result['current_inventory']}")
        print(
            f"QR策略中心 - 满足: {qr_result['satisfied']}/{current_demand}, 缺货: {qr_result['shortage']}, 订货: {qr_result['order_placed']}, 库存: {qr_result['current_inventory']}")

    # 显示模拟结果
    print("\n\n===== 模拟结束 =====")

    # 获取两个系统的摘要
    rs_summary = rs_system.get_summary()
    qr_summary = qr_system.get_summary()

    print("\n系统摘要对比:")
    print(f"总需求: {rs_summary['total_demand']}")
    print(f"{'指标':<15} {'RS策略中心':<15} {'QR策略中心':<15}")
    print("-" * 45)
    print(f"{'最终库存':<15} {rs_summary['final_inventory']:<15} {qr_summary['final_inventory']:<15}")
    print(f"{'平均服务水平':<15} {rs_summary['avg_service_level']:.2f}{'':<13} {qr_summary['avg_service_level']:.2f}")
    print(f"{'总成本':<15} {rs_summary['total_cost']:<15.2f} {qr_summary['total_cost']:<15.2f}")
    print(f"{'总订货量':<15} {rs_summary['total_orders']:<15} {qr_summary['total_orders']:<15}")

    # 绘制两个系统的历史对比图
    rs_system.plot_history()
    qr_system.plot_history()

    # 绘制两个系统的直接对比
    plot_comparison(rs_system, qr_system)


def plot_comparison(system1, system2):
    """绘制两个系统的直接对比图"""
    periods = range(1, len(system1.demand_history) + 1)

    plt.figure(figsize=(15, 12))

    # 对比库存水平
    plt.subplot(3, 1, 1)
    plt.plot(periods, system1.inventory_history, 'b-', marker='o', label=f"{system1.name} 库存")
    plt.plot(periods, system2.inventory_history, 'r-', marker='s', label=f"{system2.name} 库存")
    plt.plot(periods, system1.demand_history, 'g--', label="需求")
    plt.title("库存水平对比")
    plt.xlabel("周期")
    plt.ylabel("数量")
    plt.legend()
    plt.grid(True)

    # 对比服务水平
    plt.subplot(3, 1, 2)
    plt.plot(periods, system1.service_level_history, 'b-', marker='o', label=f"{system1.name}")
    plt.plot(periods, system2.service_level_history, 'r-', marker='s', label=f"{system2.name}")
    plt.title("服务水平对比")
    plt.xlabel("周期")
    plt.ylabel("服务水平")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)

    # 对比成本
    plt.subplot(3, 1, 3)

    # 计算累计成本
    system1_cumulative_cost = np.cumsum(system1.cost_agent.costHistory)
    system2_cumulative_cost = np.cumsum(system2.cost_agent.costHistory)

    plt.plot(periods, system1_cumulative_cost, 'b-', marker='o', label=f"{system1.name}")
    plt.plot(periods, system2_cumulative_cost, 'r-', marker='s', label=f"{system2.name}")
    plt.title("累计成本对比")
    plt.xlabel("周期")
    plt.ylabel("累计成本")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 绘制订货策略对比
    plt.figure(figsize=(12, 6))

    width = 0.35
    x = np.arange(len(periods))

    plt.bar(x - width / 2, system1.order_history, width, label=f"{system1.name}")
    plt.bar(x + width / 2, system2.order_history, width, label=f"{system2.name}")

    plt.xlabel("周期")
    plt.ylabel("订货量")
    plt.title("订货策略对比")
    plt.xticks(x, periods)
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()