# 贡献指南

非常感谢您考虑为DPCS-B项目做出贡献！本文档提供了参与贡献的指南和流程介绍。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
  - [报告Bug](#报告bug)
  - [功能请求](#功能请求)
  - [代码贡献](#代码贡献)
- [开发流程](#开发流程)
  - [环境设置](#环境设置)
  - [代码风格](#代码风格)
  - [测试](#测试)
  - [文档](#文档)
- [Pull Request流程](#pull-request流程)
- [版本发布流程](#版本发布流程)
- [联系方式](#联系方式)

## 行为准则

本项目遵循开源社区的行为准则，我们期望所有参与者尊重彼此并保持专业态度。不当行为包括但不限于：

- 使用侮辱性语言或图像
- 人身攻击或政治攻击
- 公开或私下骚扰
- 未经明确许可发布他人私人信息
- 其他不道德或不专业的行为

项目维护者有权利和责任删除、编辑或拒绝不符合本行为准则的评论、提交、代码、wiki编辑、issues和其他贡献内容。

## 如何贡献

### 报告Bug

如果您发现了Bug，请按照以下步骤报告：

1. 在项目的Issues页面搜索已存在的相关问题，避免重复报告
2. 如果没有找到相关问题，创建一个新的issue
3. 使用清晰的标题和详细描述
4. 包含以下信息：
   - 使用的DPCS-B版本
   - 操作系统和Python版本
   - 重现步骤
   - 期望行为与实际行为
   - 相关截图或日志

### 功能请求

如果您有新功能的想法，请：

1. 检查功能是否已在路线图中或已有相关issue
2. 创建一个新的issue，标签为"enhancement"
3. 详细描述您想要的功能及其潜在用例
4. 如有可能，提供实现思路或伪代码

### 代码贡献

如果您想为项目代码做出贡献，请按照以下步骤：

1. Fork项目仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request

## 开发流程

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/dpcs-b.git
cd dpcs-b

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -e .[dev]

# 安装pre-commit钩子
pre-commit install
```

### 代码风格

我们遵循PEP 8编码规范，并使用以下工具确保代码质量：

- **flake8**: 代码风格检查
- **black**: 代码格式化
- **isort**: 导入排序
- **mypy**: 类型检查

```bash
# 运行代码格式化
black dpcs tests examples
isort dpcs tests examples

# 运行代码风格检查
flake8 dpcs tests examples

# 运行类型检查
mypy dpcs
```

### 测试

所有新功能和修复都应该有相应的测试。我们使用pytest进行测试：

```bash
# 运行所有测试
pytest

# 运行特定测试模块
pytest tests/test_modules/test_spatial.py

# 运行测试并生成覆盖率报告
pytest --cov=dpcs tests/
```

### 文档

所有公共API和函数都应该有清晰的文档字符串。我们遵循Google风格的文档字符串格式：

```python
def function_name(param1, param2):
    """
    函数简短描述.
    
    详细描述，可以包含多行文本.
    
    Args:
        param1: 第一个参数的描述
        param2: 第二个参数的描述
        
    Returns:
        返回值的描述
        
    Raises:
        ValueError: 可能引发的异常及条件
    """
    # 函数实现
```

## Pull Request流程

1. 确保您的PR针对的是`develop`分支，而不是`main`分支
2. 更新或添加相关的测试
3. 更新文档（如果需要）
4. 确保所有CI检查都通过
5. 准备好针对反馈进行更改
6. 确保PR标题简洁明了

PR模板将帮助您提供所有必要信息。

## 版本发布流程

DPCS-B遵循[语义化版本](https://semver.org/)规范:

- 主版本号(X.0.0): 不兼容的API修改
- 次版本号(0.X.0): 向后兼容的功能性新增
- 修订号(0.0.X): 向后兼容的问题修正

发布流程由项目维护者管理。

## 联系方式

如有任何问题或需要帮助，请通过以下方式联系：

- GitHub issues
- 项目邮箱: info@dpcs-b.example.com

感谢您的贡献！