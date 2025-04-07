import torch
import torch.nn as nn
import torch.distributed as dist
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple


class ParallelManager:
    """并行处理管理器"""

    def __init__(self):
        self.parallel_mode = None
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.world_size = 1
        self.rank = 0
        self.initialized = False

    def initialize_data_parallel(self, model, device_ids=None):
        """
        初始化数据并行

        Args:
            model: 模型
            device_ids: 设备ID列表

        Returns:
            parallel_model: 并行处理模型
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for data parallel',
                'model': model
            }

        if device_ids is None:
            device_ids = list(range(self.device_count))

        if len(device_ids) <= 1:
            return {
                'status': 'warning',
                'message': 'At least 2 devices needed for data parallel',
                'model': model
            }

        try:
            parallel_model = nn.DataParallel(model, device_ids=device_ids)
            self.parallel_mode = 'data_parallel'
            self.initialized = True

            return {
                'status': 'success',
                'message': f'Model initialized with DataParallel on {len(device_ids)} devices',
                'model': parallel_model,
                'devices': device_ids
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize DataParallel: {str(e)}',
                'model': model
            }

    def initialize_distributed(self, model, backend='nccl', init_method='env://', world_size=None, rank=None):
        """
        初始化分布式训练

        Args:
            model: 模型
            backend: 后端
            init_method: 初始化方法
            world_size: 世界大小
            rank: 等级

        Returns:
            parallel_model: 分布式模型
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for distributed training',
                'model': model
            }

        if not dist.is_available():
            return {
                'status': 'error',
                'message': 'Distributed package not available',
                'model': model
            }

        # 确定world_size和rank
        if world_size is None:
            if 'WORLD_SIZE' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
            else:
                world_size = 1

        if rank is None:
            if 'RANK' in os.environ:
                rank = int(os.environ['RANK'])
            else:
                rank = 0

        try:
            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method=init_method,
                                        world_size=world_size, rank=rank)

            # 设置设备
            torch.cuda.set_device(rank % torch.cuda.device_count())

            # 创建DistributedDataParallel模型
            model = model.cuda()
            parallel_model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank % torch.cuda.device_count()]
            )

            self.parallel_mode = 'distributed'
            self.world_size = world_size
            self.rank = rank
            self.initialized = True

            return {
                'status': 'success',
                'message': f'Model initialized with DistributedDataParallel (rank {rank}/{world_size})',
                'model': parallel_model,
                'world_size': world_size,
                'rank': rank
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize DistributedDataParallel: {str(e)}',
                'model': model
            }

    def initialize_model_parallel(self, model, split_layer=None):
        """
        初始化模型并行

        Args:
            model: 模型
            split_layer: 分割层

        Returns:
            result: 模型并行结果
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for model parallel',
                'model': model
            }

        if self.device_count < 2:
            return {
                'status': 'error',
                'message': 'At least 2 GPU devices needed for model parallel',
                'model': model
            }

        # 从模型获取所有层
        layers = []
        for name, module in model.named_children():
            layers.append((name, module))

        if not layers:
            return {
                'status': 'error',
                'message': 'Model has no child modules for model parallel',
                'model': model
            }

        # 如果未指定分割层，则尝试在中间分割
        if split_layer is None:
            split_idx = len(layers) // 2
            split_layer = layers[split_idx][0]
        else:
            split_idx = None
            for i, (name, _) in enumerate(layers):
                if name == split_layer:
                    split_idx = i
                    break

        if split_idx is None:
            return {
                'status': 'error',
                'message': f'Split layer {split_layer} not found in model',
                'model': model
            }

        try:
            # 创建模型并行的两部分
            part1_layers = layers[:split_idx]
            part2_layers = layers[split_idx:]

            class ModelParallelPart1(nn.Module):
                def __init__(self, layers):
                    super(ModelParallelPart1, self).__init__()
                    for name, module in layers:
                        setattr(self, name, module)
                    self.layer_names = [name for name, _ in layers]

                def forward(self, x):
                    for name in self.layer_names:
                        x = getattr(self, name)(x)
                    return x

            class ModelParallelPart2(nn.Module):
                def __init__(self, layers):
                    super(ModelParallelPart2, self).__init__()
                    for name, module in layers:
                        setattr(self, name, module)
                    self.layer_names = [name for name, _ in layers]

                def forward(self, x):
                    for name in self.layer_names:
                        x = getattr(self, name)(x)
                    return x

            # 创建模型的两部分
            part1 = ModelParallelPart1(part1_layers).cuda(0)
            part2 = ModelParallelPart2(part2_layers).cuda(1)

            class ModelParallel(nn.Module):
                def __init__(self, part1, part2):
                    super(ModelParallel, self).__init__()
                    self.part1 = part1
                    self.part2 = part2

                def forward(self, x):
                    # 确保输入在第一个设备上
                    if x.device != self.part1.parameters().__next__().device:
                        x = x.cuda(0)

                    # 第一部分前向传播
                    output_part1 = self.part1(x)

                    # 在设备之间传输数据
                    output_part1 = output_part1.cuda(1)

                    # 第二部分前向传播
                    output = self.part2(output_part1)

                    return output

            parallel_model = ModelParallel(part1, part2)
            self.parallel_mode = 'model_parallel'
            self.initialized = True

            return {
                'status': 'success',
                'message': f'Model initialized with Model Parallel, split at {split_layer}',
                'model': parallel_model,
                'part1': part1,
                'part2': part2,
                'split_layer': split_layer
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize Model Parallel: {str(e)}',
                'model': model
            }

    def initialize_pipeline_parallel(self, model, num_chunks=2):
        """
        初始化流水线并行

        Args:
            model: 模型
            num_chunks: 分块数量

        Returns:
            result: 流水线并行结果
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for pipeline parallel',
                'model': model
            }

        if self.device_count < 2:
            return {
                'status': 'error',
                'message': 'At least 2 GPU devices needed for pipeline parallel',
                'model': model
            }

        try:
            # 导入torch.distributed.pipeline.sync
            try:
                from torch.distributed.pipeline.sync import Pipe
            except ImportError:
                return {
                    'status': 'error',
                    'message': 'Pipeline parallelism requires torch.distributed.pipeline',
                    'model': model
                }

            # 将模型分块
            chunks = []
            layers = list(model.children())
            chunk_size = max(1, len(layers) // num_chunks)

            for i in range(0, len(layers), chunk_size):
                chunk_layers = layers[i:i + chunk_size]
                chunk = nn.Sequential(*chunk_layers)
                chunks.append(chunk)

            # 创建流水线模型
            devices = [i % self.device_count for i in range(len(chunks))]
            pipe_model = Pipe(nn.Sequential(*chunks), chunks=len(chunks), devices=devices)

            self.parallel_mode = 'pipeline_parallel'
            self.initialized = True

            return {
                'status': 'success',
                'message': f'Model initialized with Pipeline Parallel using {len(chunks)} chunks',
                'model': pipe_model,
                'chunks': chunks,
                'devices': devices
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize Pipeline Parallel: {str(e)}',
                'model': model
            }

    def initialize_hybrid_parallel(self, model, split_layer=None, world_size=None, rank=None):
        """
        初始化混合并行（模型并行+数据并行）

        Args:
            model: 模型
            split_layer: 分割层
            world_size: 世界大小
            rank: 等级

        Returns:
            result: 混合并行结果
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for hybrid parallel',
                'model': model
            }

        if self.device_count < 2:
            return {
                'status': 'error',
                'message': 'At least 2 GPU devices needed for hybrid parallel',
                'model': model
            }

        try:
            # 首先应用模型并行
            mp_result = self.initialize_model_parallel(model, split_layer)
            if mp_result['status'] != 'success':
                return mp_result

            # 然后在每个设备上应用分布式数据并行
            mp_model = mp_result['model']
            dp_result = self.initialize_distributed(mp_model, world_size=world_size, rank=rank)

            if dp_result['status'] == 'success':
                self.parallel_mode = 'hybrid_parallel'
                return {
                    'status': 'success',
                    'message': 'Model initialized with Hybrid Parallel (Model+Data)',
                    'model': dp_result['model'],
                    'model_parallel_info': mp_result,
                    'data_parallel_info': dp_result
                }
            else:
                return dp_result
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to initialize Hybrid Parallel: {str(e)}',
                'model': model
            }

    def get_optimal_parallel_strategy(self, model, input_size, device_count=None):
        """
        获取最佳并行策略

        Args:
            model: 模型
            input_size: 输入大小
            device_count: 设备数量

        Returns:
            strategy: 最佳并行策略
        """
        if not torch.cuda.is_available():
            return {
                'status': 'warning',
                'message': 'CUDA not available',
                'recommended_strategy': 'none'
            }

        if device_count is None:
            device_count = self.device_count

        if device_count <= 1:
            return {
                'status': 'warning',
                'message': 'Only one GPU available',
                'recommended_strategy': 'none'
            }

        # 尝试评估模型大小
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # 以MB为单位

        # 估计单样本内存占用
        try:
            # 创建示例输入
            if isinstance(input_size, int):
                sample_input = torch.randn(1, input_size)
            elif isinstance(input_size, tuple) or isinstance(input_size, list):
                sample_input = torch.randn(1, *input_size)
            else:
                raise ValueError(f"Unsupported input_size: {input_size}")

            sample_input = sample_input.cuda()
            model = model.cuda()

            # 清理缓存
            torch.cuda.empty_cache()
            init_memory = torch.cuda.memory_allocated()

            # 前向传播
            with torch.no_grad():
                _ = model(sample_input)

            # 计算内存占用
            final_memory = torch.cuda.memory_allocated()
            sample_memory = final_memory - init_memory

            # 评估最佳策略
            if model_size > 10000:  # 超过10GB的大模型
                recommendation = 'model_parallel'
                reason = 'Model is very large (>10GB), model parallelism recommended'
            elif model_size > 5000:  # 超过5GB的中型模型
                if device_count >= 4:
                    recommendation = 'hybrid_parallel'
                    reason = 'Model is large (>5GB) with multiple GPUs, hybrid parallelism recommended'
                else:
                    recommendation = 'model_parallel'
                    reason = 'Model is large (>5GB), model parallelism recommended'
            elif sample_memory > 1024 * 1024 * 1024:  # 单样本超过1GB
                recommendation = 'pipeline_parallel'
                reason = 'Single sample memory usage is high, pipeline parallelism recommended'
            else:
                recommendation = 'data_parallel'
                reason = 'Model size and memory usage are moderate, data parallelism recommended'

            return {
                'status': 'success',
                'recommended_strategy': recommendation,
                'reason': reason,
                'model_size_mb': model_size,
                'sample_memory_bytes': sample_memory,
                'device_count': device_count
            }
        except Exception as e:
            # 如果评估失败，返回基于模型大小的简单建议
            if model_size > 5000:
                recommendation = 'model_parallel'
                reason = 'Model is large (>5GB), model parallelism recommended'
            else:
                recommendation = 'data_parallel'
                reason = 'Default recommendation based on model size'

            return {
                'status': 'warning',
                'message': f'Error during evaluation: {str(e)}',
                'recommended_strategy': recommendation,
                'reason': reason,
                'model_size_mb': model_size,
                'device_count': device_count
            }

    def clean_up(self):
        """
        清理并行环境

        Returns:
            bool: 是否成功
        """
        if self.parallel_mode == 'distributed' and dist.is_initialized():
            dist.destroy_process_group()

        self.initialized = False
        self.parallel_mode = None

        return True


def distribute_batch(batch, devices):
    """
    将批次数据分布到多个设备

    Args:
        batch: 批次数据
        devices: 设备列表

    Returns:
        distributed_batch: 分布式批次
    """
    if not isinstance(batch, torch.Tensor):
        return batch

    # 计算每个设备的数据大小
    batch_size = batch.size(0)
    chunk_size = batch_size // len(devices)

    # 处理不能整除的情况
    remainder = batch_size % len(devices)
    chunks = []

    start = 0
    for i, device in enumerate(devices):
        # 最后一个设备处理剩余数据
        if i == len(devices) - 1:
            end = batch_size
        else:
            end = start + chunk_size + (1 if i < remainder else 0)

        chunks.append(batch[start:end].to(device))
        start = end

    return chunks


def sync_gradients(model):
    """
    在多设备之间同步梯度

    Args:
        model: 模型

    Returns:
        bool: 是否成功
    """
    if not torch.cuda.is_available():
        return False

    # 检查是否使用DataParallel
    if isinstance(model, nn.DataParallel):
        # DataParallel自动同步梯度
        return True

    # 检查是否使用DistributedDataParallel
    if isinstance(model, nn.parallel.DistributedDataParallel):
        # DistributedDataParallel自动同步梯度
        return True

    # 对于自定义并行实现，手动同步梯度
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= torch.cuda.device_count()

    return True


def initialize_parallel_environment(model, strategy='auto', input_size=None, **kwargs):
    """
    初始化并行环境的便捷函数

    Args:
        model: 模型
        strategy: 并行策略，'auto', 'data', 'model', 'pipeline', 'distributed', 'hybrid'
        input_size: 输入大小（用于自动策略）
        **kwargs: 其他参数

    Returns:
        result: 初始化结果
    """
    parallel_manager = ParallelManager()

    # 自动确定最佳策略
    if strategy == 'auto':
        if input_size is None:
            # 尝试从模型获取输入大小
            if hasattr(model, 'input_size'):
                input_size = model.input_size
            else:
                # 默认输入大小
                input_size = 256

        strategy_info = parallel_manager.get_optimal_parallel_strategy(model, input_size)
        strategy = strategy_info['recommended_strategy']

        if strategy == 'none':
            return {
                'status': 'warning',
                'message': 'No parallel strategy applicable',
                'model': model,
                'strategy_info': strategy_info
            }

    # 根据选择的策略初始化
    if strategy == 'data' or strategy == 'data_parallel':
        return parallel_manager.initialize_data_parallel(model, **kwargs)
    elif strategy == 'model' or strategy == 'model_parallel':
        return parallel_manager.initialize_model_parallel(model, **kwargs)
    elif strategy == 'pipeline' or strategy == 'pipeline_parallel':
        return parallel_manager.initialize_pipeline_parallel(model, **kwargs)
    elif strategy == 'distributed':
        return parallel_manager.initialize_distributed(model, **kwargs)
    elif strategy == 'hybrid':
        return parallel_manager.initialize_hybrid_parallel(model, **kwargs)
    else:
        return {
            'status': 'error',
            'message': f'Unknown parallel strategy: {strategy}',
            'model': model
        }


def apply_parallel_inference(model, inputs, strategy='data_parallel', batch_size=32):
    """
    应用并行推理

    Args:
        model: 模型
        inputs: 输入数据
        strategy: 并行策略
        batch_size: 批处理大小

    Returns:
        outputs: 输出结果
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        # 单设备推理
        with torch.no_grad():
            return model(inputs)

    # 数据并行推理
    if strategy == 'data_parallel':
        # 创建DataParallel模型（如果尚未创建）
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)

        with torch.no_grad():
            return model(inputs)

    # 批次分割推理
    elif strategy == 'batch_split':
        devices = list(range(torch.cuda.device_count()))
        chunks = distribute_batch(inputs, devices)

        # 在每个设备上并行推理
        results = []
        for i, chunk in enumerate(chunks):
            with torch.cuda.device(devices[i]):
                with torch.no_grad():
                    output = model(chunk)
                    results.append(output)

        # 合并结果
        if all(isinstance(r, torch.Tensor) for r in results):
            return torch.cat(results, dim=0)
        else:
            return results

    # 大批量分批次推理
    elif strategy == 'large_batch':
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("Input must be a tensor for large_batch strategy")

        batch_size = inputs.size(0)
        sub_batch_size = batch_size // torch.cuda.device_count()

        results = []
        for i in range(0, batch_size, sub_batch_size):
            end = min(i + sub_batch_size, batch_size)
            sub_inputs = inputs[i:end]

            with torch.no_grad():
                output = model(sub_inputs)
                results.append(output)

        # 合并结果
        if all(isinstance(r, torch.Tensor) for r in results):
            return torch.cat(results, dim=0)
        else:
            return results

    else:
        raise ValueError(f"Unknown inference strategy: {strategy}")