import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.modules.cerebellum import CerebellumSynchronizer


class TestCerebellumSynchronizer(unittest.TestCase):
    """小脑时序同步模块测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.input_dim = 64
        self.hidden_dim = 128
        self.lstm_layers = 2
        self.batch_size = 2
        self.cerebellum = CerebellumSynchronizer(
            self.input_dim,
            self.hidden_dim,
            self.lstm_layers
        )

        # 创建测试张量
        self.input_tensor = torch.randn(self.batch_size, self.input_dim)
        self.sequence = torch.randn(self.batch_size, 5, self.input_dim)  # [batch, seq_len, dim]

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.cerebellum.input_dim, self.input_dim)
        self.assertEqual(self.cerebellum.hidden_dim, self.hidden_dim)
        self.assertEqual(self.cerebellum.lstm_layers, self.lstm_layers)
        self.assertIsInstance(self.cerebellum.time_encoder, torch.nn.Linear)
        self.assertIsInstance(self.cerebellum.lstm, torch.nn.LSTM)
        self.assertIsInstance(self.cerebellum.output_layer, torch.nn.Linear)
        self.assertIsInstance(self.cerebellum.rhythm_controller, torch.nn.Parameter)
        self.assertIsInstance(self.cerebellum.phase_param, torch.nn.Parameter)
        self.assertEqual(len(self.cerebellum.time_constants), 5)  # 检查时间常数数量
        self.assertEqual(self.cerebellum.temporal_memory, [])

    def test_forward(self):
        """测试前向传播"""
        # 前向传播
        output = self.cerebellum(self.input_tensor)

        # 验证输出维度
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # 测试不同时间步长
        output_long = self.cerebellum(self.input_tensor, time_steps=10)
        self.assertEqual(output_long.shape, (self.batch_size, self.input_dim))

    def test_synchronize(self):
        """测试同步功能"""
        # 执行同步
        output = self.cerebellum.synchronize(
            self.input_tensor,
            time_steps=5,
            rhythm_factor=0.8
        )

        # 验证输出维度
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # 测试不同节奏因子的影响
        output_slower = self.cerebellum.synchronize(
            self.input_tensor,
            time_steps=5,
            rhythm_factor=0.5
        )

        # 应该有不同的输出值
        self.assertFalse(torch.allclose(output, output_slower))

    def test_multiscale_time(self):
        """测试多尺度时间表示"""
        # 获取_implement_multiscale_time方法
        multiscale_time_method = self.cerebellum._implement_multiscale_time

        # 执行多尺度时间表示
        time_constants = [1.0, 2.0, 4.0]
        multiscale_repr = multiscale_time_method(
            time_constants,
            self.input_tensor
        )

        # 验证输出维度：[batch_size, num_scales, signal_dim]
        self.assertEqual(multiscale_repr.shape,
                         (self.batch_size, len(time_constants), self.input_dim))

        # 验证不同时间常数的表示应该不同
        for i in range(len(time_constants) - 1):
            for j in range(i + 1, len(time_constants)):
                # 不同时间尺度的表示应该不同
                self.assertFalse(
                    torch.allclose(multiscale_repr[:, i, :], multiscale_repr[:, j, :])
                )

    def test_process_sequence(self):
        """测试时间序列处理"""
        # 处理序列
        output_sequence = self.cerebellum.process_sequence(self.sequence)

        # 验证输出维度应与输入序列相同
        self.assertEqual(output_sequence.shape, self.sequence.shape)

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output_sequence).any())
        self.assertFalse(torch.isinf(output_sequence).any())

    def test_temporal_memory_update(self):
        """测试时序记忆更新"""
        # 初始记忆应为空
        self.assertEqual(len(self.cerebellum.temporal_memory), 0)

        # 创建编码状态
        encoded_state = torch.randn(self.batch_size, self.hidden_dim)

        # 更新记忆
        self.cerebellum._update_temporal_memory(encoded_state)

        # 验证记忆更新
        self.assertEqual(len(self.cerebellum.temporal_memory), 1)

        # 确保记忆是张量
        self.assertIsInstance(self.cerebellum.temporal_memory[0], torch.Tensor)

        # 测试记忆大小限制
        for _ in range(self.cerebellum.max_memory_length + 10):
            self.cerebellum._update_temporal_memory(torch.randn(self.batch_size, self.hidden_dim))

        # 验证记忆不超过限制大小
        self.assertLessEqual(len(self.cerebellum.temporal_memory),
                             self.cerebellum.max_memory_length)

    def test_detect_pattern(self):
        """测试时间序列模式检测"""
        # 创建一个有明显模式的序列
        pattern = torch.randn(10, self.input_dim)
        repeated_pattern = torch.cat([pattern, pattern, pattern], dim=0)

        # 检测模式
        pattern_info = self.cerebellum.detect_pattern(repeated_pattern, pattern_length=10)

        # 验证返回结果
        self.assertIn('has_pattern', pattern_info)

        # 如果检测到模式
        if pattern_info['has_pattern']:
            self.assertIn('pattern_start_idx', pattern_info)
            self.assertIn('pattern_length', pattern_info)
            self.assertIn('pattern_frequency', pattern_info)

        # 测试无模式情况
        random_sequence = torch.randn(15, self.input_dim)
        random_pattern_info = self.cerebellum.detect_pattern(random_sequence, pattern_length=10)

        # 可能检测不到明显模式
        if not random_pattern_info['has_pattern']:
            self.assertIn('message', random_pattern_info)

    def test_handle_message_sync_request(self):
        """测试处理同步请求消息"""
        # 模拟synchronize方法
        mock_output = torch.randn(self.batch_size, self.input_dim)
        with patch.object(self.cerebellum, 'synchronize', return_value=mock_output):
            # 创建消息
            message = {
                'data': {
                    'signals': self.input_tensor.numpy().tolist(),
                    'time_steps': 5,
                    'rhythm_factor': 0.8
                },
                'metadata': {'type': 'sync_request'}
            }

            # 处理消息
            response = self.cerebellum._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertIn('synchronized_output', response)
            self.assertEqual(np.array(response['synchronized_output']).shape,
                             mock_output.detach().numpy().shape)

    def test_handle_message_pattern_detection_request(self):
        """测试处理模式检测请求消息"""
        # 模拟detect_pattern方法
        mock_pattern_info = {
            'has_pattern': True,
            'pattern_start_idx': 0,
            'pattern_length': 5,
            'pattern_frequency': 2,
            'avg_interval': 5.0,
            'pattern_strength': 0.8
        }
        with patch.object(self.cerebellum, 'detect_pattern', return_value=mock_pattern_info):
            # 创建消息
            message = {
                'data': {
                    'sequence': self.sequence.numpy().tolist(),
                    'pattern_length': 5
                },
                'metadata': {'type': 'pattern_detection_request'}
            }

            # 处理消息
            response = self.cerebellum._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertIn('pattern_info', response)
            self.assertEqual(response['pattern_info'], mock_pattern_info)

    def test_handle_message_sequence_processing_request(self):
        """测试处理序列处理请求消息"""
        # 模拟process_sequence方法
        mock_output = torch.randn(self.batch_size, 5, self.input_dim)
        with patch.object(self.cerebellum, 'process_sequence', return_value=mock_output):
            # 创建消息
            message = {
                'data': {
                    'sequence': self.sequence.numpy().tolist()
                },
                'metadata': {'type': 'sequence_processing_request'}
            }

            # 处理消息
            response = self.cerebellum._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertIn('processed_sequence', response)
            self.assertEqual(np.array(response['processed_sequence']).shape,
                             mock_output.detach().numpy().shape)

    def test_handle_message_unknown_type(self):
        """测试处理未知类型消息"""
        # 创建消息
        message = {
            'metadata': {'type': 'unknown_type'}
        }

        # 处理消息
        response = self.cerebellum._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'unknown_message_type')

    def test_handle_message_invalid_data(self):
        """测试处理无效数据的消息"""
        # 创建消息 - 缺少必要数据
        message = {
            'data': {},
            'metadata': {'type': 'sync_request'}
        }

        # 处理消息
        response = self.cerebellum._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'error')
        self.assertIn('message', response)
        self.assertIn('Missing', response['message'])


if __name__ == '__main__':
    unittest.main()