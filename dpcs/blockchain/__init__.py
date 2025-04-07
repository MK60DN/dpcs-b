"""
DPCS-B 区块链组件包
包含数据可用性层和计算聚合层的实现
"""

# 导出区块链组件
from dpcs.blockchain.da_layer import DataAvailabilityLayer, DataAvailabilityProver
from dpcs.blockchain.rollup_layer import RollupLayer, ZKProver

# 区块链组件配置
BLOCKCHAIN_CONFIG = {
    'default_shards': 16,
    'default_sampling_ratio': 0.1,
    'default_rollup_type': 'zk',
    'default_batch_size': 100,
    'challenge_period': 86400  # 24小时（以秒为单位）
}