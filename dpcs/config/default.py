"""
默认配置文件
包含系统各模块的默认参数设置
"""

# 系统全局配置
SYSTEM_CONFIG = {
    'name': 'DPCS-B',
    'version': '0.1.0',
    'description': '双路径协调系统',
    'debug_mode': False,
    'log_level': 'INFO'
}

# 计算层配置
COMPUTATION_CONFIG = {
    'input_size': 256,
    'embedding_dim': 768,
    'hidden_size': 512,
    'output_size': 128
}

# 左脑模块配置
SRMT_CONFIG = {
    'hidden_size': 256,
    'use_layernorm': True,
    'activation': 'relu',
    'learning_rate': 0.0003,
    'gamma': 0.99,
    'epsilon': 0.2
}

# 右脑模块配置
CAMEL_CONFIG = {
    'model_name': 'gpt2',
    'hidden_size': 512,
    'max_memory_size': 10,
    'num_attention_heads': 8
}

# 中脑模块配置
SPATIAL_DETECTOR_CONFIG = {
    'hidden_size': 128,
    'learning_rate': 0.01,
    'history_window': 10
}

# 胼胝体模块配置
CORPUS_CALLOSUM_CONFIG = {
    'fusion_dim': 256,
    'num_heads': 4
}

# 小脑模块配置
CEREBELLUM_CONFIG = {
    'hidden_dim': 128,
    'lstm_layers': 2,
    'time_constants': [1, 2, 4, 8, 16],
    'max_memory_length': 50
}

# 额叶模块配置
PREFRONTAL_CONFIG = {
    'hidden_size': 256,
    'num_heads': 4,
    'confidence_threshold': 0.7
}

# 区块链配置
BLOCKCHAIN_CONFIG = {
    'use_blockchain': True,
    'da_shards': 16,
    'sampling_ratio': 0.1,
    'rollup_type': 'zk',
    'rollup_batch_size': 100,
    'challenge_period': 86400  # 24小时
}

# 优化配置
OPTIMIZATION_CONFIG = {
    'use_mixed_precision': True,
    'use_jit': True,
    'use_quantization': True,
    'target_memory_usage': 0.8,
    'use_parallel': True,
    'use_distributed': False
}

# 构建默认合并配置
DEFAULT_CONFIG = {
    'system': SYSTEM_CONFIG,
    'computation': COMPUTATION_CONFIG,
    'srmt': SRMT_CONFIG,
    'camel': CAMEL_CONFIG,
    'spatial_detector': SPATIAL_DETECTOR_CONFIG,
    'corpus_callosum': CORPUS_CALLOSUM_CONFIG,
    'cerebellum': CEREBELLUM_CONFIG,
    'prefrontal': PREFRONTAL_CONFIG,
    'blockchain': BLOCKCHAIN_CONFIG,
    'optimization': OPTIMIZATION_CONFIG
}


def get_config(config_name=None):
    """
    获取配置

    Args:
        config_name: 配置名称

    Returns:
        config: 配置字典
    """
    if config_name is None:
        return DEFAULT_CONFIG

    if config_name in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[config_name]

    raise ValueError(f"Unknown config name: {config_name}")


def flatten_config():
    """
    展平配置为单层字典

    Returns:
        flat_config: 展平后的配置字典
    """
    flat_config = {}

    for section, config in DEFAULT_CONFIG.items():
        for key, value in config.items():
            flat_key = f"{section}.{key}"
            flat_config[flat_key] = value

    return flat_config


def get_flat_config(key=None):
    """
    获取展平后的配置

    Args:
        key: 配置键

    Returns:
        value: 配置值
    """
    flat_config = flatten_config()

    if key is None:
        return flat_config

    if key in flat_config:
        return flat_config[key]

    raise ValueError(f"Unknown config key: {key}")