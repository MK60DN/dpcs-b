#!/usr/bin/env python
"""
DPCS-B 类脑AI框架 - 示例运行工具

该工具允许用户运行库存管理示例文件，并通过DPCS-B框架处理。
"""

import os
import sys
import argparse
from pathlib import Path

# 导入案例适配器
from dpcs.examples.case_adapter import CaseAdapter
from dpcs.system import DualPathCoordinationSystem
from dpcs.interface.file_upload import main as launch_gui


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DPCS-B 类脑AI框架 - 示例运行工具")
    parser.add_argument('file', nargs='?', help="要运行的示例文件")
    parser.add_argument('--mode', choices=['left', 'right', 'dual'], default='dual',
                        help="DPCS-B处理模式 (左脑/右脑/双路径)")
    parser.add_argument('--no-blockchain', action='store_true',
                        help="禁用区块链功能")
    parser.add_argument('--gui', action='store_true',
                        help="启动图形界面")
    parser.add_argument('--list', action='store_true',
                        help="列出可用的示例")

    args = parser.parse_args()

    # 创建DPCS-B系统实例
    dpcs = DualPathCoordinationSystem()

    # 创建案例适配器
    adapter = CaseAdapter(dpcs)

    # 处理不同选项
    if args.gui:
        # 启动图形界面
        launch_gui()
        return

    if args.list:
        # 列出可用示例
        list_examples()
        return

    if args.file:
        # 运行指定的示例文件
        file_path = args.file
        print(f"运行示例: {file_path}")
        print(f"处理模式: {args.mode}")
        print(f"区块链功能: {'禁用' if args.no_blockchain else '启用'}")
        print("-" * 50)

        # 运行示例
        success = adapter.run_case(
            file_path,
            mode=args.mode,
            use_blockchain=not args.no_blockchain
        )

        if success:
            print("\n示例运行成功!")
        else:
            print("\n示例运行失败!")
            sys.exit(1)
    else:
        # 如果没有指定文件且没有其他选项，显示帮助
        parser.print_help()
        print("\n可用示例:")
        list_examples()


def list_examples():
    """列出可用的示例"""
    examples_dir = Path("./examples")

    if not examples_dir.exists():
        print("找不到examples目录!")
        return

    # 查找示例文件
    example_files = sorted(examples_dir.glob("*.py"))

    if not example_files:
        print("没有找到任何示例文件!")
        return

    print("\n可用示例:")
    for i, file in enumerate(example_files, 1):
        # 跳过__init__.py和其他非示例文件
        if file.name.startswith("__"):
            continue

        # 提取示例描述
        description = extract_description(file)
        print(f"{i}. {file.name} - {description}")

    print("\n使用方法:")
    print("  python -m dpcs.run_example examples/basic_usage.py")
    print("  python -m dpcs.run_example examples/integrated_system_example.py --mode left")
    print("  python -m dpcs.run_example --gui")


def extract_description(file_path):
    """从Python文件提取描述"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # 只读取前1000个字符

        # 查找文档字符串
        if '"""' in content:
            docstring = content.split('"""')[1].strip()
            # 取第一行
            return docstring.split('\n')[0]

        # 如果没有docstring，尝试注释
        if '#' in content:
            for line in content.split('\n'):
                if line.strip().startswith('#'):
                    return line.strip().lstrip('#').strip()
    except Exception:
        pass

    return "无描述"


if __name__ == "__main__":
    main()