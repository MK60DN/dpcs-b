# DPCS-B 安装指南

本文档提供了安装DPCS-B框架的详细说明，包括环境要求、安装步骤以及可能遇到的问题解决方案。

## 系统要求

- **操作系统**：Windows 10+, macOS 10.14+, 或 Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**：Python 3.8 或更高版本
- **RAM**：最小 4GB，推荐 8GB+
- **GPU**：可选，但对于大规模处理任务推荐使用CUDA兼容的NVIDIA GPU
- **磁盘空间**：至少 2GB 可用空间

## 安装方法

### 方法一：从PyPI安装（推荐）

```bash
# 安装基本版本
pip install dpcs-b

# 安装完整版本（包含所有功能）
pip install dpcs-b[all]
```

### 方法二：从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/dpcs-b.git
cd dpcs-b

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖并安装DPCS-B
pip install -e .
```

### 选择性安装

DPCS-B支持按需安装不同的功能模块：

```bash
# 仅安装区块链功能
pip install dpcs-b[blockchain]

# 仅安装用户界面
pip install dpcs-b[ui]

# 仅安装示例代码
pip install dpcs-b[examples]

# 安装开发工具
pip install dpcs-b[dev]
```

## 验证安装

安装完成后，可以运行以下命令验证安装是否成功：

```bash
# 验证命令行工具
dpcs --list

# 或者运行Python代码
python -c "from dpcs.system import DualPathCoordinationSystem; print('安装成功!')"
```

## 常见问题

### PyTorch安装问题

如果在安装过程中遇到PyTorch相关的问题，可以尝试直接从PyTorch官网安装匹配您系统的版本：

```bash
# CUDA支持版本（如果有NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

安装后，继续安装DPCS-B：

```bash
pip install dpcs-b
```

### 缺少依赖问题

如果您在运行时遇到缺少依赖的错误，可以尝试手动安装缺失的包：

```bash
pip install -r requirements.txt
```

或者重新安装DPCS-B的完整版本：

```bash
pip install dpcs-b[all]
```

### GUI相关问题

如果您在使用GUI界面时遇到问题：

#### 在Linux系统上

确保已安装tkinter：

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo dnf install python3-tkinter
```

#### 在MacOS系统上

如果使用Homebrew安装的Python，确保已安装tkinter：

```bash
brew install python-tk
```

## 开发环境设置

如果您计划为DPCS-B贡献代码或进行开发，建议安装开发环境：

```bash
# 安装开发依赖
pip install -e .[dev]

# 安装pre-commit钩子
pre-commit install
```

## 更多帮助

如果您在安装过程中遇到其他问题，请查阅以下资源：

- 项目GitHub仓库：https://github.com/yourusername/dpcs-b
- 项目文档：https://dpcs-b.readthedocs.io/
- 创建Issue：https://github.com/yourusername/dpcs-b/issues