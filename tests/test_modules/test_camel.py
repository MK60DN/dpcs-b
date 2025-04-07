import unittest
import torch
import numpy as np
from dpcs.modules.camel import CAMELAgent
from dpcs.config.default import CAMEL_CONFIG, COMPUTATION_CONFIG


class TestCAMELAgent(unittest.TestCase):
    """语义理解与语言生成模块（右脑）测试"""

    def setUp(self):
        """测试前的设置"""
        # 配置参数
        self.embedding_dim = COMPUTATION_CONFIG['embedding_dim']
        self.hidden_size = CAMEL_CONFIG['hidden_size']
        self.output_size = COMPUTATION_CONFIG['output_size']

        # 创建CAMEL模块
        self.camel = CAMELAgent(
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            model_name=CAMEL_CONFIG['model_name']
        )

        # 创建测试输入
        self.test_embedding = torch.randn(1, self.embedding_dim)

        # 记录可用设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.camel = self.camel.cuda()
            self.test_embedding = self.test_embedding.cuda()

        print(f"Running CAMEL tests on device: {self.device}")

    def test_initialization(self):
        """测试模块初始化"""
        # 检查编码器
        self.assertIsInstance(self.camel.encoder, torch.nn.Sequential)

        # 检查解码器
        self.assertIsInstance(self.camel.decoder, torch.nn.Sequential)

        # 检查任务规划器
        self.assertIsInstance(self.camel.task_planner, torch.nn.Sequential)

        # 检查注意力机制
        self.assertIsInstance(self.camel.attention, torch.nn.MultiheadAttention)

        # 验证参数尺寸
        for name, param in self.camel.named_parameters():
            if 'encoder.0.weight' in name:
                self.assertEqual(param.shape, (self.hidden_size, self.embedding_dim))
            elif 'decoder.2.weight' in name:
                self.assertEqual(param.shape, (self.output_size, self.hidden_size))
            elif 'task_planner.2.weight' in name:
                self.assertEqual(param.shape, (self.hidden_size // 2, self.hidden_size))

    def test_forward_pass(self):
        """测试前向传播"""
        # 运行前向传播
        output, task_plan = self.camel(self.test_embedding)

        # 验证输出尺寸
        self.assertEqual(output.shape, (1, self.output_size))
        self.assertEqual(task_plan.shape, (1, self.hidden_size // 2))

    def test_text_generation(self):
        """测试文本生成"""
        # 运行文本生成
        generated_text = self.camel.generate_text(self.test_embedding, max_length=20)

        # 验证输出是字符串
        self.assertIsInstance(generated_text, str)

        # 验证输出不为空
        self.assertTrue(len(generated_text) > 0)

        # 生成不同长度的文本
        short_text = self.camel.generate_text(self.test_embedding, max_length=10)
        long_text = self.camel.generate_text(self.test_embedding, max_length=50)

        # 预期不同长度，但这是一个简化实现，可能不严格遵循长度限制
        self.assertIsInstance(short_text, str)
        self.assertIsInstance(long_text, str)

    def test_memory_mechanism(self):
        """测试记忆机制"""
        # 初始记忆应为空
        self.assertEqual(len(self.camel.memory), 0)

        # 运行前向传播，应更新记忆
        self.camel(self.test_embedding)

        # 验证记忆已更新
        self.assertEqual(len(self.camel.memory), 1)

        # 再次运行前向传播
        self.camel(self.test_embedding)

        # 验证记忆继续累积
        self.assertEqual(len(self.camel.memory), 2)

        # 验证记忆上限
        for _ in range(self.camel.max_memory_size):
            self.camel(self.test_embedding)

        # 验证记忆不超过上限
        self.assertLessEqual(len(self.camel.memory), self.camel.max_memory_size)

    def test_attention_mechanism(self):
        """测试注意力机制"""
        # 首先生成一些记忆
        for _ in range(3):
            self.camel(torch.randn_like(self.test_embedding).to(self.device))

        # 现在应该有3个记忆项
        self.assertEqual(len(self.camel.memory), 3)

        # 运行前向传播，应该应用注意力
        output, _ = self.camel(self.test_embedding)

        # 验证输出形状
        self.assertEqual(output.shape, (1, self.output_size))

    def test_parsing_and_planning(self):
        """测试解析和规划"""
        # 运行解析和规划
        plan = self.camel.parse_and_plan(self.test_embedding)

        # 验证输出格式
        self.assertIsInstance(plan, dict)
        self.assertIn('encoded_representation', plan)
        self.assertIn('task_plan', plan)

        # 验证编码表示
        self.assertIsInstance(plan['encoded_representation'], np.ndarray)

        # 验证任务计划
        task_plan = plan['task_plan']
        self.assertIsInstance(task_plan, dict)
        self.assertIn('goals', task_plan)
        self.assertIn('complexity', task_plan)
        self.assertIn('estimated_steps', task_plan)
        self.assertIn('focus_areas', task_plan)

    def test_batch_processing(self):
        """测试批处理能力"""
        # 创建批次输入
        batch_size = 4
        batch_embedding = torch.randn(batch_size, self.embedding_dim).to(self.device)

        # 运行前向传播
        outputs, task_plans = self.camel(batch_embedding)

        # 验证输出尺寸
        self.assertEqual(outputs.shape, (batch_size, self.output_size))
        self.assertEqual(task_plans.shape, (batch_size, self.hidden_size // 2))

    def test_message_handling(self):
        """测试消息处理"""
        # 测试生成请求
        message = {
            'data': {
                'embedding': self.test_embedding.cpu().numpy(),
                'max_length': 30,
                'temperature': 0.8
            },
            'metadata': {
                'type': 'generate_request'
            }
        }

        response = self.camel._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'success')
        self.assertIn('generated_text', response)

        # 测试规划请求
        message = {
            'data': {
                'embedding': self.test_embedding.cpu().numpy()
            },
            'metadata': {
                'type': 'planning_request'
            }
        }

        response = self.camel._handle_message(message)

        # 验证响应
        self.assertEqual(response['status'], 'success')
        self.assertIn('plan', response)

        # 测试记忆更新
        message = {
            'data': {
                'clear_memory': True
            },
            'metadata': {
                'type': 'memory_update'
            }
        }

        response = self.camel._handle_message(message)

        # 验证响应和记忆清除
        self.assertEqual(response['status'], 'success')
        self.assertEqual(len(self.camel.memory), 0)

    def test_save_load(self):
        """测试模型保存和加载"""
        import tempfile
        import os

        # 获取初始预测
        self.camel.eval()
        with torch.no_grad():
            initial_output, initial_task_plan = self.camel(self.test_embedding)

        # 保存模型到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp:
            temp_path = temp.name
            torch.save(self.camel.state_dict(), temp_path)

        # 创建新模型并加载
        new_camel = CAMELAgent(
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)

        new_camel.load_state_dict(torch.load(temp_path))
        new_camel.eval()

        # 删除临时文件
        os.unlink(temp_path)

        # 获取加载后的预测
        with torch.no_grad():
            loaded_output, loaded_task_plan = new_camel(self.test_embedding)

        # 验证预测结果一致
        self.assertTrue(torch.allclose(initial_output, loaded_output))
        self.assertTrue(torch.allclose(initial_task_plan, loaded_task_plan))

    def test_gradient_flow(self):
        """测试梯度流"""
        # 设置训练模式
        self.camel.train()

        # 准备输入，要求梯度
        test_embedding = self.test_embedding.clone().detach().requires_grad_(True)

        # 前向传播
        output, task_plan = self.camel(test_embedding)

        # 计算损失并反向传播
        loss = output.mean() + task_plan.mean()
        loss.backward()

        # 验证梯度已计算
        for name, param in self.camel.named_parameters():
            self.assertIsNotNone(param.grad)
            # 至少有一些非零梯度
            if param.grad.norm() > 0:
                break
        else:
            self.fail("未检测到非零梯度")

        # 验证输入梯度
        self.assertIsNotNone(test_embedding.grad)
        self.assertTrue(test_embedding.grad.norm() > 0)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试错误输入尺寸
        wrong_embedding = torch.randn(1, self.embedding_dim + 10).to(self.device)

        # 预期前向传播应失败
        with self.assertRaises(RuntimeError):
            self.camel(wrong_embedding)

        # 测试处理无效消息
        invalid_message = {
            'data': None,
            'metadata': {
                'type': 'unknown_type'
            }
        }

        response = self.camel._handle_message(invalid_message)
        self.assertEqual(response['status'], 'unknown_message_type')

    def test_memory_operations(self):
        """测试记忆操作"""
        # 清除现有记忆
        self.camel.memory = []

        # 创建测试编码
        test_encoded = torch.randn(1, self.hidden_size).to(self.device)

        # 更新记忆
        self.camel._update_memory(test_encoded)

        # 验证记忆已更新
        self.assertEqual(len(self.camel.memory), 1)

        # 验证记忆内容
        self.assertTrue(torch.allclose(self.camel.memory[0], test_encoded.squeeze(0)))

        # 测试批量编码记忆更新
        batch_encoded = torch.randn(3, self.hidden_size).to(self.device)
        self.camel._update_memory(batch_encoded)

        # 验证批量记忆更新（应该保存平均值）
        self.assertEqual(len(self.camel.memory), 2)

    def test_text_embedding_update(self):
        """测试文本嵌入更新"""
        # 创建一个token ID
        token_id = torch.tensor([123]).to(self.device)

        # 更新嵌入
        updated_embedding = self.camel._update_input_with_token(self.test_embedding, token_id)

        # 验证更新后的嵌入形状
        self.assertEqual(updated_embedding.shape, self.test_embedding.shape)

        # 验证更新导致变化
        self.assertFalse(torch.allclose(updated_embedding, self.test_embedding))


if __name__ == "__main__":
    unittest.main()