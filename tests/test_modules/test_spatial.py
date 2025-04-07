import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.modules.spatial import SpatialDetector, ModeAdaptation


class TestSpatialDetector(unittest.TestCase):
    """中脑路由选择器(Spatial Detector)测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.input_size = 64
        self.detector = SpatialDetector(self.input_size)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.detector.input_size, self.input_size)
        self.assertIsInstance(self.detector.fc1, torch.nn.Linear)
        self.assertIsInstance(self.detector.ln1, torch.nn.LayerNorm)
        self.assertIsInstance(self.detector.fc2, torch.nn.Linear)
        self.assertIsInstance(self.detector.ln2, torch.nn.LayerNorm)
        self.assertIsInstance(self.detector.fc3, torch.nn.Linear)
        self.assertIsNone(self.detector.history_probs)
        self.assertEqual(self.detector.mode_history, [])
        self.assertIsInstance(self.detector.mode_adaptation, ModeAdaptation)

    def test_forward(self):
        """测试前向传播"""
        # 创建输入张量
        batch_size = 2
        x = torch.randn(batch_size, self.input_size)

        # 前向传播
        output = self.detector(x)

        # 验证输出维度和范围
        self.assertEqual(output.shape, (batch_size, 3))  # 三种模式：左脑、右脑、双脑
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # 概率范围
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(batch_size)))  # 概率和为1

        # 验证历史更新
        self.assertIsNotNone(self.detector.history_probs)
        self.assertTrue(torch.allclose(self.detector.history_probs, output.detach()))

    def test_detect_mode(self):
        """测试模式检测"""
        # 创建输入张量
        x = torch.randn(1, self.input_size)

        # 检测模式
        mode, probs = self.detector.detect_mode(x)

        # 验证返回值
        self.assertIn(mode, ["left", "right", "dual"])
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

        # 验证历史更新
        self.assertEqual(len(self.detector.mode_history), 1)
        self.assertEqual(self.detector.mode_history[0], mode)

    def test_update_performance(self):
        """测试性能更新"""
        # 更新各模式性能
        self.assertTrue(self.detector.update_performance("left", 0.8))
        self.assertTrue(self.detector.update_performance("right", 0.6))
        self.assertTrue(self.detector.update_performance("dual", 0.9))

        # 无效模式
        self.assertFalse(self.detector.update_performance("invalid", 0.7))

        # 验证性能记录
        mode_adaptation = self.detector.mode_adaptation
        self.assertEqual(len(mode_adaptation.mode_performance["left"]), 1)
        self.assertEqual(len(mode_adaptation.mode_performance["right"]), 1)
        self.assertEqual(len(mode_adaptation.mode_performance["dual"]), 1)
        self.assertEqual(mode_adaptation.mode_performance["left"][0], 0.8)
        self.assertEqual(mode_adaptation.mode_performance["right"][0], 0.6)
        self.assertEqual(mode_adaptation.mode_performance["dual"][0], 0.9)

    def test_get_performance_stats(self):
        """测试获取性能统计"""
        # 添加一些历史记录
        self.detector.mode_history = ["left", "right", "dual", "left", "left"]
        self.detector.update_performance("left", 0.8)
        self.detector.update_performance("right", 0.6)
        self.detector.update_performance("dual", 0.9)

        # 获取统计信息
        stats = self.detector.get_performance_stats()

        # 验证统计结果
        self.assertEqual(stats["mode_counts"]["left"], 3)
        self.assertEqual(stats["mode_counts"]["right"], 1)
        self.assertEqual(stats["mode_counts"]["dual"], 1)
        self.assertEqual(stats["total_decisions"], 5)
        self.assertIn("current_bias", stats)
        self.assertIn("performance", stats)

    def test_reset_history(self):
        """测试重置历史"""
        # 添加一些历史记录
        self.detector.history_probs = torch.tensor([[0.3, 0.3, 0.4]])
        self.detector.mode_history = ["left", "right", "dual"]

        # 重置历史
        self.assertTrue(self.detector.reset_history())

        # 验证重置结果
        self.assertIsNone(self.detector.history_probs)
        self.assertEqual(self.detector.mode_history, [])

    def test_handle_message_detection_request(self):
        """测试处理检测请求消息"""
        # 模拟detect_mode方法
        with patch.object(self.detector, 'detect_mode', return_value=("left", np.array([[0.6, 0.3, 0.1]]))):
            # 创建消息
            message = {
                'data': torch.randn(1, self.input_size),
                'metadata': {'type': 'detection_request'}
            }

            # 处理消息
            response = self.detector._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertEqual(response['selected_mode'], 'left')
            self.assertEqual(len(response['probabilities']), 3)

    def test_handle_message_performance_update(self):
        """测试处理性能更新消息"""
        # 模拟update_performance方法
        with patch.object(self.detector, 'update_performance', return_value=True):
            # 创建消息
            message = {
                'data': {'mode': 'left', 'performance': 0.8},
                'metadata': {'type': 'performance_update'}
            }

            # 处理消息
            response = self.detector._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertEqual(response['message'], 'Performance updated')

    def test_handle_message_stats_request(self):
        """测试处理统计请求消息"""
        # 模拟get_performance_stats方法
        mock_stats = {
            "mode_counts": {"left": 3, "right": 1, "dual": 1},
            "total_decisions": 5,
            "current_bias": {"left": 0.1, "right": 0.0, "dual": 0.05},
            "performance": {"left": [0.8], "right": [0.6], "dual": [0.9]}
        }
        with patch.object(self.detector, 'get_performance_stats', return_value=mock_stats):
            # 创建消息
            message = {
                'metadata': {'type': 'stats_request'}
            }

            # 处理消息
            response = self.detector._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertEqual(response['stats'], mock_stats)

    def test_handle_message_unknown_type(self):
        """测试处理未知类型消息"""
        # 创建消息
        message = {
            'metadata': {'type': 'unknown_type'}
        }

        # 处理消息
        response = self.detector._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'unknown_message_type')


class TestModeAdaptation(unittest.TestCase):
    """处理模式适应机制测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.adaptation = ModeAdaptation(learning_rate=0.01, history_window=5)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.adaptation.learning_rate, 0.01)
        self.assertEqual(self.adaptation.history_window, 5)
        self.assertEqual(self.adaptation.mode_performance, {'left': [], 'right': [], 'dual': []})
        self.assertEqual(self.adaptation.mode_bias, {'left': 0.0, 'right': 0.0, 'dual': 0.0})

    def test_update_performance(self):
        """测试更新性能"""
        # 更新多次性能
        for i in range(7):  # 超过历史窗口大小
            self.adaptation.update_performance('left', 0.7 + i * 0.02)

        # 验证历史窗口限制
        self.assertEqual(len(self.adaptation.mode_performance['left']), 5)

        # 检查最新的值是否保留
        self.assertAlmostEqual(self.adaptation.mode_performance['left'][-1], 0.7 + 6 * 0.02)

    def test_get_mode_bias(self):
        """测试获取模式偏好"""
        # 设置不同的性能
        self.adaptation.update_performance('left', 0.9)
        self.adaptation.update_performance('right', 0.6)
        self.adaptation.update_performance('dual', 0.8)

        # 获取偏好
        bias = self.adaptation.get_mode_bias()

        # 验证偏好值
        self.assertIsInstance(bias, dict)
        self.assertIn('left', bias)
        self.assertIn('right', bias)
        self.assertIn('dual', bias)

        # "left"的性能最好，应该有正偏好
        self.assertGreater(bias['left'], 0.0)


if __name__ == '__main__':
    unittest.main()