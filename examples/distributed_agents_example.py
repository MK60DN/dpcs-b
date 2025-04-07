"""
分布式库存代理示例 - 演示多个库存代理之间的调拨和协作
"""

from dpcs.modules.distributed_agent import DistributedInventoryAgent
from dpcs.modules.port_demand import PortDemand
import random


def main():
    # 创建三个分布式库存代理
    agents = [
        DistributedInventoryAgent(name="仓库A", forecast_demand=50, inventory=100),
        DistributedInventoryAgent(name="仓库B", forecast_demand=70, inventory=80),
        DistributedInventoryAgent(name="仓库C", forecast_demand=60, inventory=120)
    ]

    # 创建需求端口
    port = PortDemand()

    print("初始化分布式库存代理:")
    for agent in agents:
        print(f"{agent.name}: 预测需求={agent.forecastDemand}, 当前库存={agent.inventory}")

    # 模拟5个周期的运行
    print("\n开始模拟:")

    for period in range(1, 6):
        print(f"\n===== 第{period}期 =====")

        # 生成随机需求
        demands = [
            random.randint(40, 80),
            random.randint(60, 90),
            random.randint(50, 70)
        ]

        # 处理每个代理的需求
        for i, (agent, demand) in enumerate(zip(agents, demands)):
            print(f"\n{agent.name}处理需求: {demand}")

            # 接收需求
            port.receive_demand(demand)
            current_demand = port.demand_queue[-1]

            # 检查库存是否足够
            if agent.inventory >= current_demand:
                print(f"{agent.name}库存充足，满足需求: {current_demand}")
                agent.inventory -= current_demand
            else:
                # 库存不足，发起调拨请求
                shortage = current_demand - agent.inventory
                print(f"{agent.name}库存不足，缺少: {shortage}单位")

                # 发送调拨请求
                agent.request(shortage)

                # 其他代理响应调拨请求
                offers = []
                for j, other_agent in enumerate(agents):
                    if j != i:  # 不要自己响应自己的请求
                        offer = other_agent.checkTransship(shortage)
                        offers.append(offer)

                # 选择最佳调拨方案
                if offers:
                    print("\n收到调拨方案:")
                    for offer in offers:
                        print(f"- {offer['supplier_name']}提供{offer['offer_amount']}单位，价格: {offer['offer_price']:.2f}")

                    # 找出价格最低的方案
                    best_offer = min(offers, key=lambda x: x['offer_price'])
                    print(
                        f"\n选择方案: {best_offer['supplier_name']}提供{best_offer['offer_amount']}单位，价格: {best_offer['offer_price']:.2f}")

                    # 执行调拨
                    transship_amount = best_offer['offer_amount']
                    for other_agent in agents:
                        if other_agent.name == best_offer['supplier_name']:
                            other_agent.inventory -= transship_amount

                    # 更新当前代理库存
                    received_amount = min(transship_amount, shortage)
                    agent.inventory += received_amount
                    shortage -= received_amount

                    print(f"{agent.name}通过调拨获得{received_amount}单位")
                    if shortage > 0:
                        print(f"{agent.name}仍然缺少{shortage}单位")
                else:
                    print(f"无可用调拨方案，{agent.name}缺少{shortage}单位")

                # 处理需求
                fulfilled_amount = current_demand - max(0, shortage)
                agent.inventory -= min(agent.inventory, fulfilled_amount)
                print(f"{agent.name}最终满足需求: {fulfilled_amount}/{current_demand}单位")

            # 更新预测需求
            new_forecast = (agent.forecastDemand * 0.7) + (current_demand * 0.3)
            agent.forecastDemand = int(new_forecast)
            print(f"{agent.name}更新预测需求: {agent.forecastDemand}")

        # 显示每个周期结束时的库存状态
        print("\n本期结束后的库存状态:")
        for agent in agents:
            print(f"{agent.name}: 库存={agent.inventory}, 预测需求={agent.forecastDemand}")

    # 分析整体表现
    print("\n\n===== 模拟结束 =====")
    print("最终库存状态:")
    for agent in agents:
        print(f"{agent.name}: 库存={agent.inventory}, 预测需求={agent.forecastDemand}")


if __name__ == "__main__":
    main()