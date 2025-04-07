"""
基本用法示例 - 演示如何使用库存系统的基本功能
"""

from dpcs.modules.inventory_system import InventorySystem


def main():
    # 创建库存系统实例
    inventory_system = InventorySystem()

    # 检查初始库存状态
    print("初始状态:")
    inventory_system.getInventoryStatus()

    # 添加一些订单需求
    print("\n添加订单需求:")
    inventory_system.addOrder(30)
    inventory_system.addOrder(50)
    inventory_system.addOrder(70)

    # 处理订单需求
    print("\n处理订单需求:")
    inventory_system.checkDemand()

    # 检查处理后的库存状态
    print("\n处理后状态:")
    inventory_system.getInventoryStatus()
    print(f"服务水平: {inventory_system.serviceLevel:.2f}")
    print(f"满足订单数: {inventory_system.fulFilled}")
    print(f"未满足订单数: {inventory_system.backOff}")

    # 添加超出库存能力的订单
    print("\n添加超出库存能力的订单:")
    inventory_system.addOrder(200)
    inventory_system.checkDemand()

    # 检查最终状态
    print("\n最终状态:")
    inventory_system.getInventoryStatus()
    print(f"服务水平: {inventory_system.serviceLevel:.2f}")
    print(f"满足订单数: {inventory_system.fulFilled}")
    print(f"未满足订单数: {inventory_system.backOff}")
    print(f"未满足需求总量: {inventory_system.backOffAmount}")


if __name__ == "__main__":
    main()