"""
文件上传界面 - 允许用户上传案例文件并在DPCS-B框架中运行
"""

import os
import sys
import importlib.util
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import shutil
import json
import threading
import time
from pathlib import Path

# 添加父目录到路径，以便导入DPCS-B模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.system import DualPathCoordinationSystem


class FileUploadInterface:
    def __init__(self, root):
        """初始化文件上传界面"""
        self.root = root
        self.root.title("DPCS-B 类脑AI框架 - 文件上传与运行")
        self.root.geometry("1000x700")

        # 设置主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建DPCS-B系统实例
        self.dpcs = DualPathCoordinationSystem()

        # 创建界面组件
        self.create_upload_frame()
        self.create_file_list_frame()
        self.create_execution_frame()
        self.create_output_frame()

        # 运行标志
        self.running = False

        # 存储上传的文件
        self.uploaded_files = {}
        self.example_module = None

        # 临时目录，用于存储上传的文件
        self.temp_dir = Path("./temp_uploads")
        self.temp_dir.mkdir(exist_ok=True)

        # 示例目录，用于存储导入的示例
        self.example_dir = Path("./examples")
        self.example_dir.mkdir(exist_ok=True)

        # 创建examples/__init__.py以确保它是一个包
        init_file = self.example_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    def create_upload_frame(self):
        """创建文件上传框架"""
        upload_frame = ttk.LabelFrame(self.main_frame, text="文件上传", padding="10")
        upload_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # 单个文件上传
        ttk.Label(upload_frame, text="选择Python文件上传:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.single_file_button = ttk.Button(upload_frame, text="浏览...", command=self.upload_single_file)
        self.single_file_button.grid(row=0, column=1, padx=5, pady=5)

        # 多文件上传
        ttk.Label(upload_frame, text="选择多个文件上传:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.multi_file_button = ttk.Button(upload_frame, text="浏览...", command=self.upload_multiple_files)
        self.multi_file_button.grid(row=1, column=1, padx=5, pady=5)

        # 目录上传
        ttk.Label(upload_frame, text="选择目录上传:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.dir_button = ttk.Button(upload_frame, text="浏览...", command=self.upload_directory)
        self.dir_button.grid(row=2, column=1, padx=5, pady=5)

        # 清除按钮
        self.clear_button = ttk.Button(upload_frame, text="清除所有文件", command=self.clear_files)
        self.clear_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    def create_file_list_frame(self):
        """创建文件列表框架"""
        file_frame = ttk.LabelFrame(self.main_frame, text="已上传文件", padding="10")
        file_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # 创建带滚动条的列表框
        self.file_list = tk.Listbox(file_frame, width=60, height=10)
        scrollbar = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)

        self.file_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 配置列表响应双击
        self.file_list.bind("<Double-1>", self.on_file_double_click)

        # 配置网格权重
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)

    def create_execution_frame(self):
        """创建执行控制框架"""
        exec_frame = ttk.LabelFrame(self.main_frame, text="执行控制", padding="10")
        exec_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        # 主文件选择
        ttk.Label(exec_frame, text="主执行文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.main_file_var = tk.StringVar()
        self.main_file_combo = ttk.Combobox(exec_frame, textvariable=self.main_file_var, width=30)
        self.main_file_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # 分析模式
        ttk.Label(exec_frame, text="分析模式:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.analysis_mode_var = tk.StringVar(value="dual")
        mode_frame = ttk.Frame(exec_frame)
        mode_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Radiobutton(mode_frame, text="左脑路径", variable=self.analysis_mode_var, value="left").pack(side=tk.LEFT,
                                                                                                     padx=5)
        ttk.Radiobutton(mode_frame, text="右脑路径", variable=self.analysis_mode_var, value="right").pack(side=tk.LEFT,
                                                                                                      padx=5)
        ttk.Radiobutton(mode_frame, text="双路径", variable=self.analysis_mode_var, value="dual").pack(side=tk.LEFT,
                                                                                                    padx=5)

        # 使用区块链
        self.use_blockchain_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(exec_frame, text="使用区块链记录", variable=self.use_blockchain_var).grid(row=2, column=0,
                                                                                           columnspan=2, sticky="w",
                                                                                           padx=5, pady=5)

        # 运行按钮
        self.run_button = ttk.Button(exec_frame, text="运行", command=self.run_example)
        self.run_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

        # 运行状态
        self.status_var = tk.StringVar(value="就绪")
        status_frame = ttk.Frame(exec_frame)
        status_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(exec_frame, orient="horizontal", length=200, mode="determinate",
                                        variable=self.progress_var)
        self.progress.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # 配置网格权重
        exec_frame.columnconfigure(1, weight=1)

    def create_output_frame(self):
        """创建输出显示框架"""
        output_frame = ttk.LabelFrame(self.main_frame, text="输出", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # 创建输出文本框
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, width=80, height=15)
        output_scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)

        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_scrollbar.grid(row=0, column=1, sticky="ns")

        # 清除输出按钮
        clear_output_button = ttk.Button(output_frame, text="清除输出",
                                         command=lambda: self.output_text.delete(1.0, tk.END))
        clear_output_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # 保存输出按钮
        save_output_button = ttk.Button(output_frame, text="保存输出", command=self.save_output)
        save_output_button.grid(row=1, column=0, padx=5, pady=5, sticky="e")

        # 配置网格权重
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # 配置主框架的网格权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

    def upload_single_file(self):
        """上传单个文件"""
        file_path = filedialog.askopenfilename(
            title="选择Python文件",
            filetypes=[("Python文件", "*.py"), ("所有文件", "*.*")]
        )

        if file_path:
            self._process_upload_file(file_path)

    def upload_multiple_files(self):
        """上传多个文件"""
        file_paths = filedialog.askopenfilenames(
            title="选择多个Python文件",
            filetypes=[("Python文件", "*.py"), ("所有文件", "*.*")]
        )

        for file_path in file_paths:
            self._process_upload_file(file_path)

    def upload_directory(self):
        """上传整个目录"""
        dir_path = filedialog.askdirectory(title="选择目录")

        if dir_path:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        self._process_upload_file(file_path)

    def _process_upload_file(self, file_path):
        """处理上传的文件"""
        file_name = os.path.basename(file_path)

        # 复制文件到临时目录
        dest_path = os.path.join(self.temp_dir, file_name)
        shutil.copy2(file_path, dest_path)

        # 存储文件信息
        self.uploaded_files[file_name] = {
            'path': dest_path,
            'original_path': file_path,
            'name': file_name,
            'size': os.path.getsize(file_path),
            'time': time.time()
        }

        # 更新文件列表
        self._update_file_list()
        self._update_main_file_combo()

        self.output_text.insert(tk.END, f"已上传文件: {file_name}\n")

    def _update_file_list(self):
        """更新文件列表显示"""
        self.file_list.delete(0, tk.END)

        for file_name, file_info in sorted(self.uploaded_files.items()):
            size_kb = file_info['size'] / 1024
            self.file_list.insert(tk.END, f"{file_name} ({size_kb:.1f} KB)")

    def _update_main_file_combo(self):
        """更新主文件下拉列表"""
        python_files = [f for f in self.uploaded_files.keys() if f.endswith('.py')]
        self.main_file_combo['values'] = python_files

        if python_files and not self.main_file_var.get():
            # 如果有main.py或run.py，优先选择
            main_candidates = ['main.py', 'run.py', 'app.py']
            for candidate in main_candidates:
                if candidate in python_files:
                    self.main_file_var.set(candidate)
                    break
            else:
                # 否则选择第一个
                self.main_file_var.set(python_files[0])

    def clear_files(self):
        """清除所有上传的文件"""
        if messagebox.askyesno("确认", "确定要清除所有上传的文件吗？"):
            self.uploaded_files = {}
            self.file_list.delete(0, tk.END)
            self.main_file_combo['values'] = []
            self.main_file_var.set("")

            # 清空临时目录
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            self.output_text.insert(tk.END, "已清除所有文件\n")

    def on_file_double_click(self, event):
        """双击文件列表中的文件"""
        selection = self.file_list.curselection()
        if selection:
            index = selection[0]
            file_entry = self.file_list.get(index)
            file_name = file_entry.split(" (")[0]

            if file_name in self.uploaded_files:
                file_path = self.uploaded_files[file_name]['path']
                self._preview_file(file_path)

    def _preview_file(self, file_path):
        """预览文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 创建预览窗口
            preview_window = tk.Toplevel(self.root)
            preview_window.title(f"预览: {os.path.basename(file_path)}")
            preview_window.geometry("800x600")

            # 添加文本框和滚动条
            preview_frame = ttk.Frame(preview_window, padding="10")
            preview_frame.pack(fill=tk.BOTH, expand=True)

            preview_text = tk.Text(preview_frame, wrap=tk.NONE)
            x_scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
            y_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
            preview_text.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)

            preview_text.grid(row=0, column=0, sticky="nsew")
            y_scrollbar.grid(row=0, column=1, sticky="ns")
            x_scrollbar.grid(row=1, column=0, sticky="ew")

            preview_frame.columnconfigure(0, weight=1)
            preview_frame.rowconfigure(0, weight=1)

            # 插入文件内容
            preview_text.insert(tk.END, content)
            preview_text.config(state=tk.DISABLED)  # 设为只读

        except Exception as e:
            messagebox.showerror("错误", f"无法预览文件: {str(e)}")

    def run_example(self):
        """运行示例"""
        if self.running:
            messagebox.showinfo("提示", "当前有任务正在运行，请等待完成")
            return

        main_file = self.main_file_var.get()
        if not main_file:
            messagebox.showerror("错误", "请选择主执行文件")
            return

        if main_file not in self.uploaded_files:
            messagebox.showerror("错误", f"找不到主执行文件: {main_file}")
            return

        # 创建并设置examples目录结构
        self._setup_example_environment()

        # 获取分析模式
        analysis_mode = self.analysis_mode_var.get()
        use_blockchain = self.use_blockchain_var.get()

        # 配置DPCS参数
        dpcs_config = {
            'mode': analysis_mode,
            'use_blockchain': use_blockchain
        }

        # 启动运行线程
        self.running = True
        self.status_var.set("正在运行...")
        self.progress_var.set(0)

        run_thread = threading.Thread(
            target=self._run_example_thread,
            args=(main_file, dpcs_config)
        )
        run_thread.daemon = True
        run_thread.start()

    def _setup_example_environment(self):
        """设置示例环境"""
        # 清空示例目录（保留__init__.py）
        for item in os.listdir(self.example_dir):
            item_path = os.path.join(self.example_dir, item)
            if os.path.isfile(item_path) and item != "__init__.py":
                os.remove(item_path)
            elif os.path.isdir(item_path) and item != "__pycache__":
                shutil.rmtree(item_path)

        # 复制上传的文件到示例目录
        for file_info in self.uploaded_files.values():
            dest_path = os.path.join(self.example_dir, file_info['name'])
            shutil.copy2(file_info['path'], dest_path)

        # 创建modules子目录（如果需要）
        modules_dir = os.path.join(self.example_dir, "modules")
        os.makedirs(modules_dir, exist_ok=True)

        # 创建modules/__init__.py
        modules_init = os.path.join(modules_dir, "__init__.py")
        if not os.path.exists(modules_init):
            with open(modules_init, 'w') as f:
                f.write("# 模块初始化文件\n")

    def _run_example_thread(self, main_file, dpcs_config):
        """在单独的线程中运行示例"""
        try:
            self._add_output(f"开始运行 {main_file}...\n")
            self._add_output(f"DPCS-B配置: 模式={dpcs_config['mode']}, 使用区块链={dpcs_config['use_blockchain']}\n")
            self._add_output("=" * 50 + "\n")

            # 更新DPCS-B配置
            if hasattr(self.dpcs, 'set_mode'):
                self.dpcs.set_mode(dpcs_config['mode'])

            if hasattr(self.dpcs, 'set_blockchain'):
                self.dpcs.set_blockchain(dpcs_config['use_blockchain'])

            # 导入并运行示例
            main_module_path = os.path.join(self.example_dir, main_file)

            # 使用importlib动态导入模块
            module_name = f"examples.{main_file[:-3]}"  # 去掉.py后缀
            spec = importlib.util.spec_from_file_location(module_name, main_module_path)
            self.example_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self.example_module

            # 重定向标准输出到GUI
            original_stdout = sys.stdout

            class StdoutRedirector:
                def __init__(self, output_callback):
                    self.output_callback = output_callback

                def write(self, text):
                    self.output_callback(text)
                    # 保持一些输出仍然在控制台显示
                    original_stdout.write(text)

                def flush(self):
                    original_stdout.flush()

            sys.stdout = StdoutRedirector(self._add_output)

            try:
                # 执行模块
                spec.loader.exec_module(self.example_module)

                # 调用main函数（如果存在）
                if hasattr(self.example_module, "main"):
                    self.example_module.main()
                else:
                    self._add_output(f"警告: {main_file}中未找到main()函数\n")

                self._add_output("\n" + "=" * 50 + "\n")
                self._add_output("运行完成!\n")
                self.status_var.set("完成")
                self.progress_var.set(100)

            finally:
                # 恢复标准输出
                sys.stdout = original_stdout

        except Exception as e:
            error_msg = f"运行时错误: {str(e)}\n"
            self._add_output(error_msg)
            self.status_var.set("错误")

            # 显示更详细的错误信息
            import traceback
            tb = traceback.format_exc()
            self._add_output(f"错误详情:\n{tb}\n")

        finally:
            self.running = False

    def _add_output(self, text):
        """添加文本到输出窗口（线程安全）"""
        self.root.after(0, lambda: self._add_output_unsafe(text))

    def _add_output_unsafe(self, text):
        """添加文本到输出窗口（非线程安全）"""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.update()

    def save_output(self):
        """保存输出到文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存输出",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.get(1.0, tk.END))
                messagebox.showinfo("成功", "输出已成功保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存输出时出错: {str(e)}")


def main():
    """主函数，启动文件上传界面"""
    root = tk.Tk()
    app = FileUploadInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()