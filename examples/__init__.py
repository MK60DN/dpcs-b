"""
DPCS-B 示例集合
包含展示框架功能的实用示例
"""

# 版本标记
__version__ = '0.1.0'

# 导出示例列表
EXAMPLES = [
    'basic_usage',
    'inventory_control_example',
    'forecast_example',
    'cost_tracking_example',
    'distributed_agents_example',
    'integrated_system_example',
    'interactive_simulation'
]


def get_example_info(example_name):
    """
    获取示例信息

    Args:
        example_name: 示例名称

    Returns:
        示例描述字典
    """
    example_descriptions = {
        'basic_usage': '演示库存系统的基本功能，包括添加订单、处理需求和查看库存状态',
        'inventory_control_example': '演示不同的库存控制策略，比较(R,S)策略和(Q,R)策略的效果',
        'forecast_example': '演示需求预测系统的功能，展示如何基于历史数据进行需求预测',
        'cost_tracking_example': '演示成本跟踪功能，跟踪订货成本、持有成本和缺货成本',
        'distributed_agents_example': '演示分布式库存代理之间的调拨和协作，展示多个库存点如何处理库存不平衡',
        'integrated_system_example': '演示完整的综合库存管理系统，集成所有组件构建端到端的解决方案',
        'interactive_simulation': '提供交互式GUI界面进行库存管理模拟，允许用户自定义各种参数'
    }

    if example_name in example_descriptions:
        return {
            'name': example_name,
            'description': example_descriptions[example_name],
            'file': f"{example_name}.py"
        }
    else:
        return {
            'name': example_name,
            'description': "未知示例",
            'file': f"{example_name}.py"
        }


def list_examples():
    """
    列出所有示例

    Returns:
        示例信息列表
    """
    return [get_example_info(example) for example in EXAMPLES]


# 当直接运行此模块时显示所有示例
if __name__ == "__main__":
    print("DPCS-B 示例库")
    print("=" * 50)
    for example in list_examples():
        print(f"{example['name']}:")
        print(f"  {example['description']}")
        print(f"  文件: {example['file']}")
        print("-" * 50)