"""
库存控制策略示例 - 演示如何使用不同的库存控制策略 (R,S) 和 (Q,R)
"""

from dpcs.modules.inventory_control import InventoryControl
from dpcs.modules.inventory_system import InventorySystem


def main():
    # 创建库存系统实例
    inventory_system = InventorySystem()

    # 创建使用(R,S)策略的库存控制实例
    rs_control = InventoryControl(strategy="RS")
    print("使用(R,S)策略的库存控制:")
    print(f"初始库存: {rs_control.inventory}, 在途库存: {rs_control.inTransit}")
    print(f"订货点: {rs_control.reorderPoint}, 最大库存: {rs_control.maxInventory}")

    # 模拟消耗库存
    print("\n模拟消耗库存:")
    rs_control.inventory = 40  # 设置库存低于订货点
    print(f"消耗后库存: {rs_control.inventory} (低于订货点{rs_control.reorderPoint})")

    # 控制库存 - 应用(R,S)策略
    print("\n应用(R,S)策略:")
    rs_control.controlInventory()
    print(f"补货后库存: {rs_control.inventory}, 在途库存: {rs_control.inTransit}")

    # 创建使用(Q,R)策略的库存控制实例
    qr_control = InventoryControl(strategy="QR")
    print("\n\n使用(Q,R)策略的库存控制:")
    print(f"初始库存: {qr_control.inventory}, 在途库存: {qr_control.inTransit}")
    print(f"订货点Q: {qr_control.reorderPointQ}, 固定订货量: {qr_control.orderQuantity}")

    # 模拟消耗库存
    print("\n模拟消耗库存:")
    qr_control.inventory = 50  # 设置库存低于订货点Q
    print(f"消耗后库存: {qr_control.inventory} (低于订货点Q {qr_control.reorderPointQ})")

    # 控制库存 - 应用(Q,R)策略
    print("\n应用(Q,R)策略:")
    qr_control.controlInventory()
    print(f"补货后库存: {qr_control.inventory}, 在途库存: {qr_control.inTransit}")

    # 比较两种策略
    print("\n\n两种策略比较:")
    print(f"(R,S)策略 - 补货后库存: {rs_control.inventory}")
    print(f"(Q,R)策略 - 补货后库存: {qr_control.inventory}")
    print("(R,S)策略适合需求波动较大的情况，每次补货都会补到最大库存")
    print("(Q,R)策略适合需求稳定的情况，每次补货量固定")


if __name__ == "__main__":
    main()