import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.modules.callosum import CorpusCallosum, MultiHeadAttention


class TestCorpusCallosum(unittest.TestCase):
    """胼胝体信息融合模块测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.feature_dim = 64
        self.fusion_dim = 128
        self.batch_size = 2
        self.callosum = CorpusCallosum(self.feature_dim, self.fusion_dim)

        # 创建测试张量
        self.left_features = torch.randn(self.batch_size, self.feature_dim)
        self.right_features = torch.randn(self.batch_size, self.feature_dim)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.callosum.feature_dim, self.feature_dim)
        self.assertEqual(self.callosum.fusion_dim, self.fusion_dim)
        self.assertIsInstance(self.callosum.left_proj, torch.nn.Linear)
        self.assertIsInstance(self.callosum.right_proj, torch.nn.Linear)
        self.assertIsInstance(self.callosum.fusion_layer, torch.nn.Linear)
        self.assertIsInstance(self.callosum.gate_layer, torch.nn.Linear)
        self.assertIsInstance(self.callosum.alpha_param, torch.nn.Parameter)
        self.assertIsInstance(self.callosum.multihead_attention, MultiHeadAttention)
        self.assertIsInstance(self.callosum.fusion_mode_selector, torch.nn.Sequential)

    def test_forward(self):
        """测试前向传播"""
        # 前向传播
        output = self.callosum(self.left_features, self.right_features)

        # 验证输出维度
        self.assertEqual(output.shape, (self.batch_size, self.fusion_dim))

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_align_semantics(self):
        """测试语义对齐功能"""
        # 执行语义对齐
        output = self.callosum.align_semantics(self.left_features, self.right_features)

        # 验证输出维度
        self.assertEqual(output.shape, (self.batch_size, self.fusion_dim))

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_semantic_alignment_score(self):
        """测试语义对齐分数"""
        # 计算对齐分数
        score = self.callosum.semantic_alignment_score(self.left_features, self.right_features)

        # 验证分数类型和范围
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

        # 测试相同特征的对齐分数
        same_score = self.callosum.semantic_alignment_score(self.left_features, self.left_features)
        self.assertGreaterEqual(same_score, score)  # 自身对齐应该更好

    def test_handle_message_fusion_request(self):
        """测试处理融合请求消息"""
        # 模拟forward方法
        mock_output = torch.randn(self.batch_size, self.fusion_dim)
        with patch.object(self.callosum, 'forward', return_value=mock_output):
            # 创建消息
            message = {
                'data': {
                    'left_features': self.left_features.numpy().tolist(),
                    'right_features': self.right_features.numpy().tolist()
                },
                'metadata': {'type': 'fusion_request'}
            }

            # 处理消息
            response = self.callosum._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertIn('fused_output', response)
            self.assertEqual(np.array(response['fused_output']).shape,
                             mock_output.detach().numpy().shape)

    def test_handle_message_alignment_score_request(self):
        """测试处理对齐分数请求消息"""
        # 模拟semantic_alignment_score方法
        with patch.object(self.callosum, 'semantic_alignment_score', return_value=0.75):
            # 创建消息
            message = {
                'data': {
                    'left_features': self.left_features.numpy().tolist(),
                    'right_features': self.right_features.numpy().tolist()
                },
                'metadata': {'type': 'alignment_score_request'}
            }

            # 处理消息
            response = self.callosum._handle_message(message)

            # 验证响应
            self.assertEqual(response['status'], 'success')
            self.assertIn('alignment_score', response)
            self.assertEqual(response['alignment_score'], 0.75)

    def test_handle_message_unknown_type(self):
        """测试处理未知类型消息"""
        # 创建消息
        message = {
            'metadata': {'type': 'unknown_type'}
        }

        # 处理消息
        response = self.callosum._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'unknown_message_type')

    def test_handle_message_invalid_data(self):
        """测试处理无效数据的消息"""
        # 创建消息 - 缺少右特征
        message = {
            'data': {
                'left_features': self.left_features.numpy().tolist()
            },
            'metadata': {'type': 'fusion_request'}
        }

        # 处理消息
        response = self.callosum._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'error')
        self.assertIn('message', response)
        self.assertIn('Missing', response['message'])


class TestMultiHeadAttention(unittest.TestCase):
    """多头注意力机制测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.hidden_size = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10
        self.mha = MultiHeadAttention(self.hidden_size, self.num_heads)

        # 创建测试张量
        self.query = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.key = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.value = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.mha.hidden_size, self.hidden_size)
        self.assertEqual(self.mha.num_heads, self.num_heads)
        self.assertEqual(self.mha.head_size, self.hidden_size // self.num_heads)
        self.assertIsInstance(self.mha.q_linear, torch.nn.Linear)
        self.assertIsInstance(self.mha.k_linear, torch.nn.Linear)
        self.assertIsInstance(self.mha.v_linear, torch.nn.Linear)
        self.assertIsInstance(self.mha.out_linear, torch.nn.Linear)

    def test_forward(self):
        """测试前向传播"""
        # 前向传播
        output = self.mha(self.query, self.key, self.value)

        # 验证输出维度
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))

        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_mask(self):
        """测试注意力掩码（选择性测试）"""
        # 此测试仅在MultiHeadAttention支持掩码时有效
        if hasattr(self.mha, 'apply_mask'):
            # 创建掩码
            mask = torch.ones(self.batch_size, self.seq_len, self.seq_len)
            mask[:, :, self.seq_len // 2:] = 0  # 掩盖后半部分

            # 前向传播（带掩码）
            output_masked = self.mha.apply_mask(self.query, self.key, self.value, mask)

            # 前向传播（不带掩码）
            output_normal = self.mha(self.query, self.key, self.value)

            # 验证两个输出不同
            self.assertFalse(torch.allclose(output_masked, output_normal))
        else:
            # 如果不支持掩码，则跳过此测试
            self.skipTest("MultiHeadAttention does not support masking")


if __name__ == '__main__':
    unittest.main()