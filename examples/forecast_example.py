"""
需求预测示例 - 演示如何使用需求预测系统
"""

from dpcs.modules.forecast_system import InventorySystemWithForecast


def main():
    # 创建带有预测功能的库存系统实例
    forecast_system = InventorySystemWithForecast(forecast_period=3, shortage_threshold=30)

    print(f"初始化预测系统 - 预测周期: {forecast_system.forecastPeriod}, 缺货阈值: {forecast_system.shortageThreshold}")
    print(f"初始库存: {forecast_system.inventory}, 在途库存: {forecast_system.inTransit}")

    # 尝试在没有足够历史数据的情况下进行预测
    print("\n尝试预测 (不足历史数据):")
    forecast_system.forecastDemand()

    # 添加历史订单数据
    print("\n添加历史订单数据:")
    demands = [45, 52, 48]
    for i, demand in enumerate(demands, 1):
        print(f"添加第{i}期订单需求: {demand}")
        forecast_system.updateOrderHistory(demand)

    # 使用累积的历史数据进行预测
    print("\n有足够历史数据后进行预测:")
    forecast_system.forecastDemand()

    # 添加一个新的异常需求值
    print("\n添加一个异常需求值:")
    new_demand = 80
    print(f"添加新订单需求: {new_demand}")
    forecast_system.updateOrderHistory(new_demand)

    # 查看预测如何变化
    print("\n添加异常值后的预测:")
    forecast_system.forecastDemand()

    # 模拟一段时间的需求数据
    print("\n\n模拟一段时间的需求数据:")
    # 重新初始化预测系统
    forecast_system = InventorySystemWithForecast(forecast_period=5)

    # 模拟10个周期的需求数据
    demands = [50, 55, 48, 52, 58, 60, 65, 70, 62, 68]
    forecasts = []

    for period, demand in enumerate(demands, 1):
        print(f"\n第{period}期:")
        print(f"实际需求: {demand}")
        forecast_system.updateOrderHistory(demand)

        # 从第5期开始有足够的历史数据进行预测
        if period >= 5:
            forecast_system.forecastDemand()
            # 存储预测值用于后续分析
            if period > 5:
                forecasts.append(sum(demands[period - 6:period - 1]) / 5)

    # 简单分析预测准确性（从第6期开始）
    if len(forecasts) > 0:
        actual_demands = demands[5:]
        print("\n\n预测准确性分析:")
        for i, (forecast, actual) in enumerate(zip(forecasts, actual_demands), 6):
            error = ((actual - forecast) / actual) * 100
            print(f"第{i}期 - 预测: {forecast:.2f}, 实际: {actual}, 误差: {error:.2f}%")

        # 计算平均预测误差
        avg_error = sum([abs(((a - f) / a) * 100) for f, a in zip(forecasts, actual_demands)]) / len(forecasts)
        print(f"\n平均预测误差: {avg_error:.2f}%")


if __name__ == "__main__":
    main()