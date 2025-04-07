import torch
import gc
import sys
import os
import psutil
import numpy as np
import random
from typing import Dict, List, Any, Optional, Union, Tuple


class MemoryManager:
    """内存管理器"""

    def __init__(self, target_usage=0.8):
        self.target_usage = target_usage
        self.estimated_per_sample = {}
        self.optimal_batch_sizes = {}

    def estimate_memory_usage(self, model, input_size, device='cuda'):
        """
        估计模型内存使用情况

        Args:
            model: PyTorch模型
            input_size: 输入大小
            device: 设备类型

        Returns:
            memory_stats: 内存统计信息
        """
        if not torch.cuda.is_available() and device == 'cuda':
            return {
                'error': 'CUDA not available',
                'suggestion': 'Use CPU device instead'
            }

        # 清理缓存
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            init_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process(os.getpid())
            init_memory = process.memory_info().rss

        # 创建示例输入
        if isinstance(input_size, int):
            sample_input = torch.randn(1, input_size)
        elif isinstance(input_size, tuple) or isinstance(input_size, list):
            sample_input = torch.randn(1, *input_size)
        else:
            raise ValueError(f"Unsupported input_size: {input_size}")

        if device == 'cuda':
            sample_input = sample_input.cuda()
            model = model.cuda()

        # 前向传播
        try:
            with torch.no_grad():
                _ = model(sample_input)

            # 计算内存占用
            if device == 'cuda':
                final_memory = torch.cuda.memory_allocated()
                memory_used = final_memory - init_memory

                # 估计每个样本的内存
                del sample_input
                torch.cuda.empty_cache()

                # 尝试批处理
                batch_size = 10
                batch_input = torch.randn(batch_size, *sample_input.shape[1:]).cuda()
                batch_init_memory = torch.cuda.memory_allocated()
                _ = model(batch_input)
                batch_final_memory = torch.cuda.memory_allocated()
                batch_memory_used = batch_final_memory - batch_init_memory

                per_sample_estimate = batch_memory_used / batch_size
            else:
                process = psutil.Process(os.getpid())
                final_memory = process.memory_info().rss
                memory_used = final_memory - init_memory

                # 估计每个样本的内存
                del sample_input
                gc.collect()

                # 尝试批处理
                batch_size = 10
                batch_input = torch.randn(batch_size, *sample_input.shape[1:])
                process = psutil.Process(os.getpid())
                batch_init_memory = process.memory_info().rss
                _ = model(batch_input)
                process = psutil.Process(os.getpid())
                batch_final_memory = process.memory_info().rss
                batch_memory_used = batch_final_memory - batch_init_memory

                per_sample_estimate = batch_memory_used / batch_size

            # 存储估计值
            model_name = model.__class__.__name__
            self.estimated_per_sample[model_name] = per_sample_estimate

            return {
                'model_name': model_name,
                'memory_used': memory_used,
                'per_sample_estimate': per_sample_estimate,
                'device': device
            }
        except Exception as e:
            return {
                'error': str(e),
                'model_name': model.__class__.__name__,
                'device': device
            }
        finally:
            # 清理
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

    def compute_optimal_batch_size(self, model_name, available_memory=None, safety_factor=0.9):
        """
        计算最佳批处理大小

        Args:
            model_name: 模型名称
            available_memory: 可用内存
            safety_factor: 安全因子

        Returns:
            optimal_batch_size: 最佳批处理大小
        """
        if model_name not in self.estimated_per_sample:
            return {
                'error': f'No memory estimate for model: {model_name}',
                'suggestion': 'Run estimate_memory_usage first'
            }

        per_sample = self.estimated_per_sample[model_name]

        # 如果未指定可用内存，获取系统信息
        if available_memory is None:
            if torch.cuda.is_available():
                device_info = torch.cuda.get_device_properties(0)
                total_memory = device_info.total_memory
                available_memory = total_memory * self.target_usage
            else:
                total_memory = psutil.virtual_memory().total
                available_memory = total_memory * self.target_usage

        # 计算最佳批大小
        optimal_batch_size = int((available_memory * safety_factor) / per_sample)
        optimal_batch_size = max(1, optimal_batch_size)  # 至少为1

        # 存储结果
        self.optimal_batch_sizes[model_name] = optimal_batch_size

        return {
            'model_name': model_name,
            'optimal_batch_size': optimal_batch_size,
            'available_memory': available_memory,
            'per_sample_memory': per_sample
        }

    def enable_gradient_checkpointing(self, model):
        """
        启用梯度检查点

        Args:
            model: PyTorch模型

        Returns:
            bool: 是否成功
        """
        # 检查是否支持梯度检查点
        if not hasattr(model, 'modules'):
            return False

        from torch.utils.checkpoint import checkpoint_sequential

        # 查找适合检查点的序列模块
        sequential_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Sequential) and len(list(module.children())) > 1:
                sequential_modules.append((name, module))

        if not sequential_modules:
            return False

        # 应用检查点
        for name, module in sequential_modules:
            setattr(model, name.replace('.', '_') + '_checkpointed',
                    lambda x, m=module: checkpoint_sequential(m, 3, x))

        return True

    def get_memory_stats(self):
        """
        获取内存统计信息

        Returns:
            memory_stats: 内存统计信息
        """
        stats = {
            'system': {
                'total_ram': psutil.virtual_memory().total,
                'available_ram': psutil.virtual_memory().available,
                'ram_usage_percent': psutil.virtual_memory().percent
            }
        }

        # 添加CUDA信息
        if torch.cuda.is_available():
            cuda_stats = {}
            for i in range(torch.cuda.device_count()):
                device_info = torch.cuda.get_device_properties(i)
                cuda_stats[f'device_{i}'] = {
                    'name': device_info.name,
                    'total_memory': device_info.total_memory,
                    'allocated_memory': torch.cuda.memory_allocated(i),
                    'reserved_memory': torch.cuda.memory_reserved(i),
                    'utilization': torch.cuda.utilization(i)
                }
            stats['cuda'] = cuda_stats

        # 添加进程信息
        process = psutil.Process(os.getpid())
        stats['process'] = {
            'pid': process.pid,
            'rss': process.memory_info().rss,
            'vms': process.memory_info().vms,
            'cpu_percent': process.cpu_percent(interval=0.1)
        }

        # 添加Python对象信息
        stats['python'] = {
            'gc_objects': len(gc.get_objects()),
            'tensors': sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))
        }

        return stats

    def clear_memory(self):
        """
        清理内存

        Returns:
            bool: 是否成功
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def diagnose_memory_leak(self, model, input_generator, iterations=10):
        """
        诊断内存泄漏

        Args:
            model: PyTorch模型
            input_generator: 输入生成器函数
            iterations: 迭代次数

        Returns:
            diagnosis: 诊断结果
        """
        if not torch.cuda.is_available():
            return {
                'error': 'CUDA not available',
                'suggestion': 'Use CPU device instead'
            }

        # 清理初始状态
        gc.collect()
        torch.cuda.empty_cache()

        # 跟踪内存使用
        memory_usage = []
        tensor_counts = []

        try:
            for i in range(iterations):
                # 记录初始状态
                initial_memory = torch.cuda.memory_allocated()
                initial_tensors = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

                # 执行前向传播
                inputs = input_generator()
                with torch.no_grad():
                    outputs = model(inputs)
                    del outputs

                # 清理
                del inputs
                gc.collect()
                torch.cuda.empty_cache()

                # 记录最终状态
                final_memory = torch.cuda.memory_allocated()
                final_tensors = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))

                # 记录差异
                memory_usage.append(final_memory - initial_memory)
                tensor_counts.append(final_tensors - initial_tensors)

            # 分析结果
            memory_growth = np.array(memory_usage)
            tensor_growth = np.array(tensor_counts)

            has_leak = np.any(memory_growth[-3:] > 1024)  # 最后3次迭代存在明显增长

            diagnosis = {
                'has_memory_leak': has_leak,
                'memory_growth': memory_growth.tolist(),
                'tensor_growth': tensor_growth.tolist(),
                'iterations': iterations
            }

            # 提供建议
            if has_leak:
                diagnosis['suggestion'] = [
                    'Check for tensors not being properly deleted',
                    'Ensure all intermediate results are released',
                    'Consider using gradient checkpointing',
                    'Reduce batch size or model size'
                ]

            return diagnosis
        except Exception as e:
            return {
                'error': str(e),
                'suggestion': 'Error during diagnosis'
            }
        finally:
            # 最终清理
            gc.collect()
            torch.cuda.empty_cache()


def estimate_batch_size(model, input_size, target_memory_usage=0.8):
    """
    估计最佳批处理大小的简便函数

    Args:
        model: PyTorch模型
        input_size: 输入大小
        target_memory_usage: 目标内存使用率

    Returns:
        optimal_batch_size: 最佳批处理大小
    """
    memory_manager = MemoryManager(target_usage=target_memory_usage)
    stats = memory_manager.estimate_memory_usage(model, input_size)

    if 'error' in stats:
        return 1  # 出错时默认为1

    model_name = stats['model_name']
    batch_size_info = memory_manager.compute_optimal_batch_size(model_name)

    if 'error' in batch_size_info:
        return 1

    return batch_size_info['optimal_batch_size']


def enable_mixed_precision(model=None):
    """
    启用混合精度训练

    Args:
        model: 可选的PyTorch模型

    Returns:
        result: 启用结果
    """
    if not torch.cuda.is_available():
        return {
            'status': 'error',
            'message': 'CUDA not available for mixed precision'
        }

    try:
        # 初始化混合精度训练
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        # 如果提供了模型，将其转换为CUDA
        if model is not None:
            model = model.cuda()

        return {
            'status': 'success',
            'scaler': scaler,
            'autocast': autocast,
            'message': 'Mixed precision enabled'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to enable mixed precision: {str(e)}'
        }


def apply_memory_optimization(model, config=None):
    """
    应用内存优化技术

    Args:
        model: PyTorch模型
        config: 优化配置

    Returns:
        optimized_model: 优化后的模型
    """
    if config is None:
        config = {
            'use_mixed_precision': True,
            'use_gradient_checkpointing': True,
            'optimize_batch_size': True,
            'target_memory_usage': 0.8
        }

    memory_manager = MemoryManager(target_usage=config.get('target_memory_usage', 0.8))
    optimizations_applied = []

    # 启用混合精度
    if config.get('use_mixed_precision', True) and torch.cuda.is_available():
        mixed_precision_result = enable_mixed_precision(model)
        if mixed_precision_result['status'] == 'success':
            optimizations_applied.append('mixed_precision')

    # 启用梯度检查点
    if config.get('use_gradient_checkpointing', True):
        checkpointing_success = memory_manager.enable_gradient_checkpointing(model)
        if checkpointing_success:
            optimizations_applied.append('gradient_checkpointing')

    # 优化批处理大小
    if config.get('optimize_batch_size', True):
        if hasattr(model, 'input_size'):
            input_size = model.input_size
        else:
            # 默认输入大小
            input_size = config.get('input_size', 256)

        optimal_batch_size = estimate_batch_size(
            model,
            input_size,
            config.get('target_memory_usage', 0.8)
        )

        optimizations_applied.append(f'optimal_batch_size_{optimal_batch_size}')

    return {
        'model': model,
        'optimizations_applied': optimizations_applied,
        'memory_manager': memory_manager
    }


class DataLoader:
    """高效数据加载器"""

    def __init__(self, dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.num_workers = num_workers

        # 创建索引
        self.indices = list(range(len(dataset)))
        self.current_position = 0

        # 如果启用打乱
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        self.current_position = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_position >= len(self.indices):
            raise StopIteration

        # 获取批次索引
        batch_indices = self.indices[self.current_position:self.current_position + self.batch_size]
        self.current_position += self.batch_size

        # 加载数据
        batch = [self.dataset[i] for i in batch_indices]

        # 转换为张量批次
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # 假设是(输入，标签)对
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            if isinstance(inputs[0], torch.Tensor):
                inputs = torch.stack(inputs)
            if isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)

            # 如果启用pin_memory
            if self.pin_memory:
                inputs = inputs.pin_memory()
                if isinstance(labels, torch.Tensor):
                    labels = labels.pin_memory()

            return inputs, labels
        else:
            # 单一输入
            if isinstance(batch[0], torch.Tensor):
                batch = torch.stack(batch)
                if self.pin_memory:
                    batch = batch.pin_memory()
            return batch

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


class CachedDataset:
    """缓存数据集"""

    def __init__(self, dataset, cache_size=1000, device='cuda'):
        self.dataset = dataset
        self.cache_size = min(cache_size, len(dataset))
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 初始化缓存
        self.cache = {}
        self.access_count = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 预热缓存
        self._warm_cache()

    def _warm_cache(self):
        """预热缓存"""
        indices = list(range(min(self.cache_size, len(self.dataset))))
        for i in indices:
            self.cache[i] = self._load_to_device(self.dataset[i])
            self.access_count[i] = 0

    def _load_to_device(self, item):
        """将项目加载到设备"""
        if isinstance(item, torch.Tensor):
            return item.to(self.device)
        elif isinstance(item, tuple) and all(isinstance(x, torch.Tensor) for x in item):
            return tuple(x.to(self.device) for x in item)
        return item

    def __getitem__(self, idx):
        """获取数据项"""
        if idx in self.cache:
            # 缓存命中
            self.cache_hits += 1
            self.access_count[idx] += 1
            return self.cache[idx]

        # 缓存未命中
        self.cache_misses += 1
        item = self._load_to_device(self.dataset[idx])

        # 如果缓存已满，移除最少访问的项
        if len(self.cache) >= self.cache_size:
            min_idx = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_idx]
            del self.access_count[min_idx]

        # 添加到缓存
        self.cache[idx] = item
        self.access_count[idx] = 1

        return item

    def __len__(self):
        return len(self.dataset)

    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def resize_cache(self, new_size):
        """调整缓存大小"""
        if new_size < len(self.cache):
            # 需要减少缓存
            items_to_remove = len(self.cache) - new_size

            # 按访问次数排序
            sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])

            # 移除最少访问的项
            for i in range(items_to_remove):
                if i < len(sorted_items):
                    idx = sorted_items[i][0]
                    del self.cache[idx]
                    del self.access_count[idx]

        self.cache_size = new_size
        return len(self.cache)


def monitor_memory_usage(interval=5.0):
    """
    监控内存使用情况的装饰器

    Args:
        interval: 监控间隔（秒）

    Returns:
        decorator: 装饰器函数
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 创建监控线程
            import threading
            import time

            stop_event = threading.Event()
            memory_stats = []

            def monitor_thread():
                while not stop_event.is_set():
                    # 记录内存使用情况
                    process = psutil.Process(os.getpid())
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_info = process.memory_info()

                    if torch.cuda.is_available():
                        cuda_memory = {
                            'allocated': torch.cuda.memory_allocated(),
                            'reserved': torch.cuda.memory_reserved()
                        }
                    else:
                        cuda_memory = None

                    memory_stats.append({
                        'timestamp': time.time(),
                        'rss': memory_info.rss,
                        'vms': memory_info.vms,
                        'cpu_percent': cpu_percent,
                        'cuda_memory': cuda_memory
                    })

                    time.sleep(interval)

            # 启动监控线程
            monitor = threading.Thread(target=monitor_thread)
            monitor.daemon = True
            monitor.start()

            try:
                # 执行原函数
                result = func(*args, **kwargs)
                return result
            finally:
                # 停止监控线程
                stop_event.set()
                monitor.join()

                # 返回监控结果
                result.memory_stats = memory_stats

        return wrapper

    return decorator


def optimize_for_inference(model):
    """
    针对推理优化模型

    Args:
        model: PyTorch模型

    Returns:
        optimized_model: 优化后的模型
    """
    # 设置为评估模式
    model.eval()

    # 如果可能，使用TorchScript
    try:
        traced_model = torch.jit.script(model)
        model = traced_model
    except Exception as e:
        print(f"Could not create TorchScript model: {e}")

    # 量化模型
    try:
        from torch.quantization import quantize_dynamic
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        return {
            'model': quantized_model,
            'optimizations': ['eval_mode', 'torchscript', 'quantization']
        }
    except Exception as e:
        print(f"Could not quantize model: {e}")
        return {
            'model': model,
            'optimizations': ['eval_mode', 'torchscript']
        }