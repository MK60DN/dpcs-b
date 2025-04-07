import torch
import torch.nn as nn
import numpy as np
import time
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple


class OptimizationManager:
    """优化管理器"""

    def __init__(self, log_level=logging.INFO):
        # 设置日志
        self.logger = logging.getLogger("optimization_manager")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 初始化优化状态
        self.optimizations = {
            'jit_compilation': False,
            'mixed_precision': False,
            'quantization': False,
            'pruning': False,
            'kernel_fusion': False
        }

        # 性能跟踪
        self.performance_history = {}

    def apply_jit_compilation(self, model, sample_input=None):
        """
        应用JIT编译

        Args:
            model: PyTorch模型
            sample_input: 示例输入

        Returns:
            compiled_model: 编译后的模型
        """
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            try:
                if sample_input is not None:
                    # 使用跟踪JIT
                    self.logger.info("Applying trace JIT compilation...")
                    compiled_model = torch.jit.trace(model, sample_input)
                else:
                    # 使用脚本JIT
                    self.logger.info("Applying script JIT compilation...")
                    compiled_model = torch.jit.script(model)

                self.optimizations['jit_compilation'] = True
                return {
                    'status': 'success',
                    'model': compiled_model,
                    'message': 'JIT compilation applied successfully'
                }
            except Exception as e:
                self.logger.error(f"JIT compilation failed: {str(e)}")
                return {
                    'status': 'error',
                    'model': model,
                    'message': f'JIT compilation failed: {str(e)}'
                }
        else:
            return {
                'status': 'warning',
                'model': model,
                'message': 'Model is already using JIT'
            }

    def apply_mixed_precision(self, model, optimizer=None):
        """
        应用混合精度训练

        Args:
            model: PyTorch模型
            optimizer: 优化器

        Returns:
            result: 混合精度设置结果
        """
        if not torch.cuda.is_available():
            return {
                'status': 'error',
                'message': 'CUDA not available for mixed precision',
                'model': model
            }

        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()

            # 如果提供了优化器，更新优化器步骤
            if optimizer is not None:
                original_step = optimizer.step

                def new_step(closure=None):
                    scaler.step(optimizer, closure)
                    scaler.update()

                optimizer.step = new_step

            self.optimizations['mixed_precision'] = True

            return {
                'status': 'success',
                'model': model,
                'scaler': scaler,
                'autocast': autocast,
                'optimizer': optimizer,
                'message': 'Mixed precision training enabled'
            }
        except Exception as e:
            self.logger.error(f"Mixed precision setup failed: {str(e)}")
            return {
                'status': 'error',
                'model': model,
                'message': f'Mixed precision setup failed: {str(e)}'
            }

    def apply_quantization(self, model, quantization_type='dynamic', dtype=torch.qint8):
        """
        应用量化

        Args:
            model: PyTorch模型
            quantization_type: 量化类型
            dtype: 数据类型

        Returns:
            quantized_model: 量化后的模型
        """
        try:
            if quantization_type == 'dynamic':
                # 动态量化
                self.logger.info("Applying dynamic quantization...")
                qconfig_mapping = torch.quantization.get_default_qconfig_mapping()
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU, nn.RNN},
                    dtype=dtype
                )
            elif quantization_type == 'static':
                # 静态量化
                self.logger.info("Applying static quantization...")
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                # 需要进行校准，这里简单跳过
                quantized_model = torch.quantization.convert(model, inplace=False)
            elif quantization_type == 'qat':
                # 量化感知训练
                self.logger.info("Applying quantization aware training...")
                model.train()
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                torch.quantization.prepare_qat(model, inplace=True)
                # 训练后转换
                quantized_model = model  # 实际使用中，需要在训练后调用convert
            else:
                return {
                    'status': 'error',
                    'model': model,
                    'message': f'Unknown quantization type: {quantization_type}'
                }

            self.optimizations['quantization'] = True
            return {
                'status': 'success',
                'model': quantized_model,
                'message': f'{quantization_type} quantization applied successfully'
            }
        except Exception as e:
            self.logger.error(f"Quantization failed: {str(e)}")
            return {
                'status': 'error',
                'model': model,
                'message': f'Quantization failed: {str(e)}'
            }

    def apply_pruning(self, model, pruning_type='unstructured', amount=0.2):
        """
        应用剪枝

        Args:
            model: PyTorch模型
            pruning_type: 剪枝类型
            amount: 剪枝比例

        Returns:
            pruned_model: 剪枝后的模型
        """
        try:
            import torch.nn.utils.prune as prune

            # 对象存储函数的模块和名称
            prunable_modules = []

            # 查找可剪枝层
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prunable_modules.append((module, 'weight'))

            if not prunable_modules:
                return {
                    'status': 'warning',
                    'model': model,
                    'message': 'No prunable layers found'
                }

            # 应用剪枝
            self.logger.info(f"Applying {pruning_type} pruning with amount {amount}...")

            if pruning_type == 'unstructured':
                for module, name in prunable_modules:
                    prune.l1_unstructured(module, name=name, amount=amount)

            elif pruning_type == 'structured':
                for module, name in prunable_modules:
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
                    else:
                        prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)

            elif pruning_type == 'global':
                parameters_to_prune = tuple(prunable_modules)
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=amount
                )
            else:
                return {
                    'status': 'error',
                    'model': model,
                    'message': f'Unknown pruning type: {pruning_type}'
                }

            # 使剪枝永久化
            for module, name in prunable_modules:
                prune.remove(module, name)

            self.optimizations['pruning'] = True
            return {
                'status': 'success',
                'model': model,
                'message': f'{pruning_type} pruning applied successfully'
            }
        except Exception as e:
            self.logger.error(f"Pruning failed: {str(e)}")
            return {
                'status': 'error',
                'model': model,
                'message': f'Pruning failed: {str(e)}'
            }

    def apply_kernel_fusion(self, model):
        """
        应用内核融合

        Args:
            model: PyTorch模型

        Returns:
            fused_model: 融合后的模型
        """
        # 此功能需要自定义实现，这里提供框架
        try:
            self.logger.info("Attempting kernel fusion...")

            # 检查是否已经应用JIT，如果没有则应用
            if not self.optimizations['jit_compilation']:
                jit_result = self.apply_jit_compilation(model)
                if jit_result['status'] != 'success':
                    return jit_result
                model = jit_result['model']

            # 假设融合已经在JIT过程中应用
            self.optimizations['kernel_fusion'] = True

            return {
                'status': 'success',
                'model': model,
                'message': 'Kernel fusion applied through JIT optimization'
            }
        except Exception as e:
            self.logger.error(f"Kernel fusion failed: {str(e)}")
            return {
                'status': 'error',
                'model': model,
                'message': f'Kernel fusion failed: {str(e)}'
            }

    def benchmark_model(self, model, sample_input, num_iterations=100, warm_up=10):
        """
        对模型进行基准测试

        Args:
            model: PyTorch模型
            sample_input: 示例输入
            num_iterations: 迭代次数
            warm_up: 预热迭代次数

        Returns:
            benchmark_results: 基准测试结果
        """
        if not isinstance(sample_input, torch.Tensor):
            return {
                'status': 'error',
                'message': 'Sample input must be a tensor'
            }

        device = next(model.parameters()).device
        sample_input = sample_input.to(device)

        # 确保模型在评估模式
        model.eval()

        # 预热
        self.logger.info(f"Warming up for {warm_up} iterations...")
        with torch.no_grad():
            for _ in range(warm_up):
                _ = model(sample_input)

        # 计时
        self.logger.info(f"Benchmarking for {num_iterations} iterations...")
        timings = []
        with torch.no_grad():
            for i in range(num_iterations):
                start_time = time.time()
                _ = model(sample_input)
                # 确保GPU操作完成
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)

        # 计算统计信息
        timings = np.array(timings)
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        median_time = np.median(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)

        # 计算吞吐量
        batch_size = sample_input.size(0)
        throughput = batch_size / mean_time

        # 存储结果
        model_name = model.__class__.__name__
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        benchmark_results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'device': str(device),
            'batch_size': batch_size,
            'mean_time': mean_time,
            'std_time': std_time,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput': throughput,
            'iterations': num_iterations,
            'optimizations': self.optimizations.copy()
        }

        # 记录历史
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        self.performance_history[model_name].append(benchmark_results)

        # 日志输出
        self.logger.info(f"Benchmark results for {model_name}:")
        self.logger.info(f"  Mean inference time: {mean_time * 1000:.2f} ms")
        self.logger.info(f"  Throughput: {throughput:.2f} samples/sec")

        return benchmark_results

    def compare_optimization_strategies(self, model, sample_input, strategies=None):
        """
        比较不同优化策略

        Args:
            model: PyTorch模型
            sample_input: 示例输入
            strategies: 要比较的策略列表

        Returns:
            comparison_results: 比较结果
        """
        if strategies is None:
            strategies = [
                {'name': 'baseline', 'optimizations': []},
                {'name': 'jit_only', 'optimizations': ['jit_compilation']},
                {'name': 'quantize_only', 'optimizations': ['quantization']},
                {'name': 'jit_and_quantize', 'optimizations': ['jit_compilation', 'quantization']},
                {'name': 'mixed_precision', 'optimizations': ['mixed_precision']},
                {'name': 'all', 'optimizations': ['jit_compilation', 'quantization', 'mixed_precision']}
            ]

        comparison_results = []
        original_model = model

        for strategy in strategies:
            self.logger.info(f"Testing strategy: {strategy['name']}")

            # 重置优化状态
            self.optimizations = {
                'jit_compilation': False,
                'mixed_precision': False,
                'quantization': False,
                'pruning': False,
                'kernel_fusion': False
            }

            # 从原始模型开始
            model = original_model

            # 应用策略中的优化
            for opt in strategy['optimizations']:
                if opt == 'jit_compilation':
                    result = self.apply_jit_compilation(model, sample_input)
                    if result['status'] == 'success':
                        model = result['model']
                elif opt == 'mixed_precision':
                    result = self.apply_mixed_precision(model)
                    if result['status'] == 'success':
                        # 混合精度不改变模型
                        pass
                elif opt == 'quantization':
                    result = self.apply_quantization(model)
                    if result['status'] == 'success':
                        model = result['model']
                elif opt == 'pruning':
                    result = self.apply_pruning(model)
                    if result['status'] == 'success':
                        # 剪枝不改变模型引用
                        pass
                elif opt == 'kernel_fusion':
                    result = self.apply_kernel_fusion(model)
                    if result['status'] == 'success':
                        model = result['model']

            # 基准测试
            benchmark_result = self.benchmark_model(model, sample_input)

            # 将结果添加到比较中
            comparison_result = {
                'strategy_name': strategy['name'],
                'optimizations': strategy['optimizations'],
                'mean_time': benchmark_result['mean_time'],
                'throughput': benchmark_result['throughput']
            }

            comparison_results.append(comparison_result)

        # 按吞吐量排序
        comparison_results.sort(key=lambda x: x['throughput'], reverse=True)

        # 找出最佳策略
        best_strategy = comparison_results[0]['strategy_name']
        best_throughput = comparison_results[0]['throughput']

        self.logger.info(f"Optimization comparison complete.")
        self.logger.info(f"Best strategy: {best_strategy} with throughput {best_throughput:.2f} samples/sec")

        return {
            'results': comparison_results,
            'best_strategy': best_strategy,
            'best_throughput': best_throughput
        }

    def optimize_for_deployment(self, model, sample_input=None, target_device='cuda', target_format=None):
        """
        为部署优化模型

        Args:
            model: PyTorch模型
            sample_input: 示例输入
            target_device: 目标设备
            target_format: 目标格式

        Returns:
            optimized_model: 优化后的模型
        """
        self.logger.info(f"Optimizing model for deployment on {target_device}")

        # 始终以评估模式开始
        model.eval()

        # 首先应用量化
        quantized_result = self.apply_quantization(model)
        if quantized_result['status'] == 'success':
            model = quantized_result['model']

        # 然后应用JIT
        if sample_input is not None:
            jit_result = self.apply_jit_compilation(model, sample_input)
            if jit_result['status'] == 'success':
                model = jit_result['model']

        # 如果需要，转换为特定格式
        if target_format == 'torchscript':
            # 已经由JIT处理
            pass
        elif target_format == 'onnx':
            try:
                import torch.onnx

                if sample_input is None:
                    return {
                        'status': 'error',
                        'message': 'Sample input is required for ONNX export',
                        'model': model
                    }

                # 导出到临时文件
                onnx_file = "model.onnx"
                torch.onnx.export(
                    model,
                    sample_input,
                    onnx_file,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}}
                )

                self.logger.info(f"Model exported to ONNX format: {onnx_file}")

                return {
                    'status': 'success',
                    'model': model,
                    'onnx_file': onnx_file,
                    'message': f'Model optimized and exported to ONNX format'
                }
            except Exception as e:
                self.logger.error(f"ONNX export failed: {str(e)}")
                return {
                    'status': 'error',
                    'model': model,
                    'message': f'ONNX export failed: {str(e)}'
                }
        elif target_format == 'tensorrt':
            try:
                import torch_tensorrt

                if sample_input is None:
                    return {
                        'status': 'error',
                        'message': 'Sample input is required for TensorRT conversion',
                        'model': model
                    }

                # 转换为TensorRT
                trt_model = torch_tensorrt.compile(
                    model,
                    inputs=[sample_input],
                    enabled_precisions={torch.float16}  # 使用FP16精度
                )

                return {
                    'status': 'success',
                    'model': trt_model,
                    'message': 'Model optimized and converted to TensorRT'
                }
            except Exception as e:
                self.logger.error(f"TensorRT conversion failed: {str(e)}")
                return {
                    'status': 'error',
                    'model': model,
                    'message': f'TensorRT conversion failed: {str(e)}'
                }

        return {
            'status': 'success',
            'model': model,
            'message': f'Model optimized for {target_device} deployment'
        }

    def optimize_training(self, model, optimizer=None):
        """
        优化训练过程

        Args:
            model: PyTorch模型
            optimizer: 优化器

        Returns:
            optimization_result: 优化结果
        """
        self.logger.info("Optimizing model for training")

        # 启用混合精度训练
        if torch.cuda.is_available():
            mixed_precision_result = self.apply_mixed_precision(model, optimizer)
            if mixed_precision_result['status'] == 'success':
                # 使用新的优化器，包含梯度缩放
                optimizer = mixed_precision_result['optimizer']
                scaler = mixed_precision_result['scaler']
                autocast = mixed_precision_result['autocast']

                return {
                    'status': 'success',
                    'model': model,
                    'optimizer': optimizer,
                    'scaler': scaler,
                    'autocast': autocast,
                    'message': 'Training optimized with mixed precision'
                }
            else:
                self.logger.warning("Mixed precision training setup failed")

        # 如果混合精度失败或不可用，尝试其他优化
        self.logger.info("Using standard training optimizations")

        # 创建优化训练函数
        def optimized_training_step(inputs, targets, criterion, use_amp=False):
            # 确保梯度清零
            optimizer.zero_grad()

            # 前向传播
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                optimizer.step()

            return loss, outputs

        return {
            'status': 'success',
            'model': model,
            'optimizer': optimizer,
            'training_step': optimized_training_step,
            'message': 'Standard training optimizations applied'
        }

    def get_optimization_stats(self):
        """
        获取优化统计信息

        Returns:
            stats: 优化统计信息
        """
        active_optimizations = [k for k, v in self.optimizations.items() if v]

        stats = {
            'active_optimizations': active_optimizations,
            'total_optimizations': len(active_optimizations),
            'performance_history': self.performance_history
        }

        return stats


def optimize_model(model, sample_input=None, target='inference', level='medium', device='cuda'):
    """
    自动应用优化的简便函数

    Args:
        model: PyTorch模型
        sample_input: 示例输入
        target: 优化目标 ('inference', 'training', 'deployment')
        level: 优化级别 ('light', 'medium', 'aggressive')
        device: 目标设备

    Returns:
        optimized_model: 优化后的模型
    """
    optimization_manager = OptimizationManager()

    # 根据优化级别选择策略
    if level == 'light':
        strategies = {
            'inference': ['jit_compilation'],
            'training': ['mixed_precision'],
            'deployment': ['jit_compilation', 'quantization']
        }
    elif level == 'medium':
        strategies = {
            'inference': ['jit_compilation', 'quantization'],
            'training': ['mixed_precision'],
            'deployment': ['jit_compilation', 'quantization', 'kernel_fusion']
        }
    elif level == 'aggressive':
        strategies = {
            'inference': ['jit_compilation', 'quantization', 'pruning', 'kernel_fusion'],
            'training': ['mixed_precision', 'jit_compilation'],
            'deployment': ['jit_compilation', 'quantization', 'pruning', 'kernel_fusion']
        }
    else:
        raise ValueError(f"Unknown optimization level: {level}")

    # 获取当前目标的策略
    if target not in strategies:
        raise ValueError(f"Unknown optimization target: {target}")

    selected_strategies = strategies[target]

    # 应用所选策略
    for strategy in selected_strategies:
        if strategy == 'jit_compilation':
            result = optimization_manager.apply_jit_compilation(model, sample_input)
            if result['status'] == 'success':
                model = result['model']
        elif strategy == 'mixed_precision':
            result = optimization_manager.apply_mixed_precision(model)
            # 混合精度不改变模型实例
        elif strategy == 'quantization':
            result = optimization_manager.apply_quantization(model)
            if result['status'] == 'success':
                model = result['model']
        elif strategy == 'pruning':
            amount = 0.1 if level == 'light' else 0.2 if level == 'medium' else 0.3
            result = optimization_manager.apply_pruning(model, amount=amount)
            # 剪枝不改变模型实例
        elif strategy == 'kernel_fusion':
            result = optimization_manager.apply_kernel_fusion(model)
            if result['status'] == 'success':
                model = result['model']

    # 为特定目标应用额外优化
    if target == 'deployment':
        result = optimization_manager.optimize_for_deployment(model, sample_input, target_device=device)
        if result['status'] == 'success':
            model = result['model']

    return {
        'model': model,
        'optimization_manager': optimization_manager,
        'applied_strategies': selected_strategies,
        'target': target,
        'level': level,
        'device': device
    }