#!/usr/bin/env python
"""
DPCS-B: 双路径协调系统 (Dual Path Coordination System)
基于脑科学启发的计算架构，结合区块链技术的创新AI框架
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 版本信息
VERSION = '0.1.0'

# 基本依赖
requirements = [
    'numpy>=1.20.0',
    'torch>=1.9.0',
    'matplotlib>=3.4.0',
    'tqdm>=4.62.0',
    'pandas>=1.3.0',
]

# 区块链相关依赖
blockchain_requirements = [
    'pycryptodome>=3.10.0',  # 加密功能
    'merkletools>=1.0.3',    # Merkle树实现
]

# 用户界面相关依赖
ui_requirements = [
    'tk',                    # 基础GUI工具包
]

# 示例代码相关依赖
examples_requirements = [
    'deque-tools>=0.1.0',    # 双端队列工具
]

# 聚合所有依赖
all_requirements = (
    requirements +
    blockchain_requirements +
    ui_requirements +
    examples_requirements
)

# 开发和测试依赖
dev_requirements = [
    'pytest>=6.2.5',
    'pytest-cov>=2.12.1',
    'flake8>=3.9.2',
    'mypy>=0.910',
    'black>=21.6b0',
    'isort>=5.9.2',
]

setup(
    name="dpcs-b",
    version=VERSION,
    author="DPCS-B Team",
    author_email="info@dpcs-b.example.com",
    description="基于脑科学启发的双路径协调系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/dpcs-b",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'blockchain': blockchain_requirements,
        'ui': ui_requirements,
        'examples': examples_requirements,
        'dev': dev_requirements,
        'all': all_requirements,
    },
    entry_points={
        'console_scripts': [
            'dpcs=dpcs.run_example:main',
        ],
    },
    include_package_data=True,
    package_data={
        'dpcs': ['config/*.json'],
    },
)