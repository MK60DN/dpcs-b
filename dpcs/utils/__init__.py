"""
DPCS-B 工具函数包
包含内存管理、并行处理和优化工具
"""

# 导出工具函数
from dpcs.utils.memory import MemoryManager
from dpcs.utils.parallel import enable_data_parallel, enable_model_parallel
from dpcs.utils.optimization import quantize_model, apply_mixed_precision