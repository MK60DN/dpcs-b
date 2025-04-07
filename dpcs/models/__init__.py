"""
DPCS-B 核心模块包
包含模拟人脑不同区域功能的六大神经认知模块
"""

# 导出核心模块，方便直接导入
from dpcs.modules.srmt import SRMT
from dpcs.modules.camel import CAMELAgent
from dpcs.modules.spatial import SpatialDetector
from dpcs.modules.callosum import CorpusCallosum
from dpcs.modules.cerebellum import CerebellumSynchronizer
from dpcs.modules.prefrontal import PrefrontalCortexModule

# 定义模块映射，便于通过名称动态加载模块
MODULE_MAPPING = {
    'left': SRMT,
    'right': CAMELAgent,
    'spatial': SpatialDetector,
    'callosum': CorpusCallosum,
    'cerebellum': CerebellumSynchronizer,
    'prefrontal': PrefrontalCortexModule
}