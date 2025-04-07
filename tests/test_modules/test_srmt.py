import unittest
import torch
import numpy as np
from dpcs.modules.srmt import SRMT
from dpcs.config.default import SRMT_CONFIG, COMPUTATION_CONFIG


class TestSRMT(unittest.TestCase):
    """结构化强化学习模块（左脑）测试"""

    def setUp(self):
        """测试前的设置"""
        # 配置参数
        self.input_size = COMPUTATION_CONFIG['input_size']
        self.hidden_size = SRMT_CONFIG['hidden_size']
        self.output_size = COMPUTATION_CONFIG['output_size']

        # 创建SRMT模块
        self.srmt = SRMT(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )

        # 创建测试输入
        self.test_input = torch.randn(1, self.input_size)

        # 记录可用设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.srmt = self.srmt.cuda()
            self.test_input = self.test_input.cuda()

        print(f"Running SRMT tests on device: {self.device}")

    def test_initialization(self):
        """测试模块初始化"""
        # 检查特征提取器
        self.assertIsInstance(self.srmt.feature_extractor, torch.nn.Sequential)

        # 检查策略网络
        self.assertIsInstance(self.srmt.policy_net, torch.nn.Sequential)

        # 检查价值网络
        self.assertIsInstance(self.srmt.value_net, torch.nn.Sequential)

        # 验证参数尺寸
        for name, param in self.srmt.named_parameters():
            if 'feature_extractor.0.weight' in name:
                self.assertEqual(param.shape, (self.hidden_size, self.input_size))
            elif 'policy_net.2.weight' in name:
                self.assertEqual(param.shape, (self.output_size, self.hidden_size // 2))
            elif 'value_net.2.weight' in name:
                self.assertEqual(param.shape, (1, self.hidden_size // 2))

    def test_forward_pass(self):
        """测试前向传播"""
        # 运行前向传播
        policy_output, value_output = self.srmt(self.test_input)

        # 验证输出尺寸
        self.assertEqual(policy_output.shape, (1, self.output_size))
        self.assertEqual(value_output.shape, (1, 1))

        # 验证策略输出范围 (由于tanh激活函数，应在-1到1之间)
        self.assertTrue(torch.all(policy_output >= -1.0))
        self.assertTrue(torch.all(policy_output <= 1.0))

    def test_predict(self):
        """测试预测方法"""
        # 运行预测
        prediction = self.srmt.predict(self.test_input)

        # 验证输出类型和形状
        self.assertIsInstance(prediction, dict)
        self.assertIn('policy_output', prediction)
        self.assertIn('value', prediction)
        self.assertIn('confidence', prediction)

        # 验证输出尺寸
        self.assertEqual(prediction['policy_output'].shape, (1, self.output_size))
        self.assertEqual(prediction['value'].shape, (1, 1))

        # 验证置信度范围
        self.assertTrue(0.0 <= prediction['confidence'] <= 1.0)

    def test_action_selection(self):
        """测试动作选择"""
        # 测试确定性动作选择
        action = self.srmt.select_action(self.test_input, deterministic=True)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (1, self.output_size))

        # 测试随机动作选择
        action = self.srmt.select_action(self.test_input, deterministic=False)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (1, self.output_size))

        # 多次随机选择，验证结果不同
        actions = []
        for _ in range(5):
            action = self.srmt.select_action(self.test_input, deterministic=False)
            actions.append(action.cpu().numpy())

        actions = np.array(actions)
        # 至少有一对结果不同
        self.assertTrue(np.any(np.std(actions, axis=0) > 0))

    def test_training(self):
        """测试训练过程"""
        # 设置训练模式
        self.srmt.train()

        # 创建优化器
        optimizer = torch.optim.Adam(self.srmt.parameters(), lr=0.001)

        # 创建模拟数据
        states = torch.randn(10, self.input_size).to(self.device)
        actions = torch.randn(10, self.output_size).to(self.device)
        rewards = torch.randn(10, 1).to(self.device)
        next_states = torch.randn(10, self.input_size).to(self.device)
        dones = torch.zeros(10, 1).to(self.device)

        # 初始参数
        old_params = [param.clone().detach() for param in self.srmt.parameters()]

        # 运行几次训练步骤
        for _ in range(3):
            self.srmt.optimize(optimizer, states, actions, rewards, next_states, dones)

        # 新参数
        new_params = [param.clone().detach() for param in self.srmt.parameters()]

        # 验证参数已更新
        params_changed = False
        for old, new in zip(old_params, new_params):
            if not torch.allclose(old, new):
                params_changed = True
                break

        self.assertTrue(params_changed)

    def test_evaluation(self):
        """测试评估模式"""
        # 设置评估模式
        self.srmt.eval()

        # 记录初始梯度需求
        initial_requires_grad = {}
        for name, param in self.srmt.named_parameters():
            initial_requires_grad[name] = param.requires_grad

        # 使用torch.no_grad()运行
        with torch.no_grad():
            policy_output, value_output = self.srmt(self.test_input)

        # 验证输出
        self.assertEqual(policy_output.shape, (1, self.output_size))
        self.assertEqual(value_output.shape, (1, 1))

        # 验证梯度需求没有改变
        for name, param in self.srmt.named_parameters():
            self.assertEqual(param.requires_grad, initial_requires_grad[name])

    def test_batch_processing(self):
        """测试批处理能力"""
        # 创建批次输入
        batch_size = 32
        batch_input = torch.randn(batch_size, self.input_size).to(self.device)

        # 运行前向传播
        policy_outputs, value_outputs = self.srmt(batch_input)

        # 验证输出尺寸
        self.assertEqual(policy_outputs.shape, (batch_size, self.output_size))
        self.assertEqual(value_outputs.shape, (batch_size, 1))

    def test_save_load(self):
        """测试模型保存和加载"""
        import tempfile
        import os

        # 获取初始预测
        self.srmt.eval()
        with torch.no_grad():
            initial_policy, initial_value = self.srmt(self.test_input)

        # 保存模型到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp:
            temp_path = temp.name
            torch.save(self.srmt.state_dict(), temp_path)

        # 创建新模型并加载
        new_srmt = SRMT(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(self.device)

        new_srmt.load_state_dict(torch.load(temp_path))
        new_srmt.eval()

        # 删除临时文件
        os.unlink(temp_path)

        # 获取加载后的预测
        with torch.no_grad():
            loaded_policy, loaded_value = new_srmt(self.test_input)

        # 验证预测结果一致
        self.assertTrue(torch.allclose(initial_policy, loaded_policy))
        self.assertTrue(torch.allclose(initial_value, loaded_value))

    def test_gradient_flow(self):
        """测试梯度流"""
        # 设置训练模式
        self.srmt.train()

        # 准备输入，要求梯度
        test_input = self.test_input.clone().detach().requires_grad_(True)

        # 前向传播
        policy_output, value_output = self.srmt(test_input)

        # 计算损失并反向传播
        loss = policy_output.mean() + value_output.mean()
        loss.backward()

        # 验证梯度已计算
        for name, param in self.srmt.named_parameters():
            self.assertIsNotNone(param.grad)
            # 至少有一些非零梯度
            if param.grad.norm() > 0:
                break
        else:
            self.fail("未检测到非零梯度")

        # 验证输入梯度
        self.assertIsNotNone(test_input.grad)
        self.assertTrue(test_input.grad.norm() > 0)

    def test_feature_extraction(self):
        """测试特征提取层"""
        # 提取特征
        features = self.srmt.extract_features(self.test_input)

        # 验证特征尺寸
        self.assertEqual(features.shape, (1, self.hidden_size))

        # 验证特征激活正确应用
        # (ReLU后应该没有负值)
        self.assertTrue(torch.all(features >= 0))

    def test_error_handling(self):
        """测试错误处理"""
        # 测试错误输入尺寸
        wrong_input = torch.randn(1, self.input_size + 10).to(self.device)

        # 预期前向传播应失败
        with self.assertRaises(RuntimeError):
            self.srmt(wrong_input)

    def test_update_hyperparameters(self):
        """测试超参数更新"""
        # 初始学习率
        initial_lr = self.srmt.learning_rate

        # 更新超参数
        new_lr = 0.0005
        self.srmt.update_hyperparameters(learning_rate=new_lr, gamma=0.98)

        # 验证更新成功
        self.assertEqual(self.srmt.learning_rate, new_lr)
        self.assertEqual(self.srmt.gamma, 0.98)


if __name__ == "__main__":
    unittest.main()