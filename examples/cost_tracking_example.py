"""
成本跟踪示例 - 演示如何使用成本跟踪代理
"""

from dpcs.modules.cost_tracking import CostTrackingAgent
import matplotlib.pyplot as plt


def main():
    # 创建成本跟踪代理实例
    cost_agent = CostTrackingAgent(
        name="配送中心1",
        order_cost_per_unit=10,
        holding_cost_per_unit=2,
        stockout_cost_per_unit=50
    )

    print(f"成本跟踪代理: {cost_agent.name}")
    print(f"订货成本: {cost_agent.orderCostPerUnit}/单位")
    print(f"持有成本: {cost_agent.holdingCostPerUnit}/单位")
    print(f"缺货成本: {cost_agent.stockoutCostPerUnit}/单位")

    # 模拟10个周期的订货和可能的缺货情况
    print("\n模拟10个周期的库存管理:")

    order_quantities = [50, 0, 70, 30, 0, 100, 0, 60, 20, 80]
    shortage_quantities = [0, 0, 0, 10, 5, 0, 0, 0, 15, 0]

    for period, (order_qty, shortage_qty) in enumerate(zip(order_quantities, shortage_quantities), 1):
        print(f"\n第{period}期:")
        print(f"订货量: {order_qty}, 缺货量: {shortage_qty}")

        # 更新成本
        cost_agent.updateTotalCost(order_qty, shortage_qty)

        # 计算当期各类成本
        order_cost = cost_agent.orderCostPerUnit * order_qty
        holding_cost = cost_agent.holdingCostPerUnit * 100  # 假设每期持有100单位
        stockout_cost = cost_agent.stockoutCostPerUnit * shortage_qty
        total_cost = order_cost + holding_cost + stockout_cost

        print(f"订货成本: {order_cost}")
        print(f"持有成本: {holding_cost}")
        print(f"缺货成本: {stockout_cost}")
        print(f"总成本: {total_cost}")
        print(f"累计总成本: {cost_agent.totalCost}")

    # 直接绘制成本历史
    print("\n绘制成本历史图表:")
    cost_agent.plotCostHistory()

    # 手动绘制更详细的成本分析图表
    print("\n绘制详细成本分析图表:")

    # 计算各期的成本明细
    periods = list(range(1, 11))
    order_costs = [cost_agent.orderCostPerUnit * qty for qty in order_quantities]
    holding_costs = [cost_agent.holdingCostPerUnit * 100 for _ in range(10)]
    stockout_costs = [cost_agent.stockoutCostPerUnit * qty for qty in shortage_quantities]
    total_costs = [o + h + s for o, h, s in zip(order_costs, holding_costs, stockout_costs)]

    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 绘制堆叠条形图
    plt.bar(periods, order_costs, label='订货成本')
    plt.bar(periods, holding_costs, bottom=order_costs, label='持有成本')
    plt.bar(periods, stockout_costs, bottom=[o + h for o, h in zip(order_costs, holding_costs)], label='缺货成本')

    # 绘制总成本折线
    plt.plot(periods, total_costs, 'r-', marker='o', linewidth=2, label='总成本')

    # 添加图表标题和标签
    plt.title(f"{cost_agent.name}的成本结构分析")
    plt.xlabel("时间周期")
    plt.ylabel("成本")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 为每个条形添加总成本标签
    for i, cost in enumerate(total_costs):
        plt.text(periods[i], cost + 10, f"{cost}", ha='center')

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()