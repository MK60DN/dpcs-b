import unittest
import torch
import numpy as np
import os
import json
import tempfile
from dpcs.system import DualPathCoordinationSystem
from dpcs.modules.srmt import SRMT
from dpcs.modules.camel import CAMELAgent
from dpcs.modules.spatial import SpatialDetector
from dpcs.modules.callosum import CorpusCallosum
from dpcs.modules.cerebellum import CerebellumSynchronizer
from dpcs.modules.prefrontal import PrefrontalCortexModule
from dpcs.blockchain.da_layer import DataAvailabilityLayer
from dpcs.blockchain.rollup_layer import RollupLayer
from dpcs.utils.memory import MemoryManager
from dpcs.utils.parallel import ParallelManager
from dpcs.utils.optimization import OptimizationManager
from dpcs.config.default import DEFAULT_CONFIG


class TestDualPathCoordinationSystem(unittest.TestCase):
    def setUp(self):
        """测试前的设置"""
        # 使用默认配置创建系统
        self.config = DEFAULT_CONFIG['computation']
        self.system = DualPathCoordinationSystem(self.config)

        # 创建测试输入
        self.test_tensor_input = torch.randn(1, self.config['input_size'])
        self.test_text_input = "这是一个测试输入，用于验证双路径协调系统的功能"

        # 记录可用设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running tests on device: {self.device}")

    def test_system_initialization(self):
        """测试系统初始化"""
        # 验证模块初始化
        self.assertIsInstance(self.system.srmt, SRMT)
        self.assertIsInstance(self.system.camel, CAMELAgent)
        self.assertIsInstance(self.system.spatial_detector, SpatialDetector)
        self.assertIsInstance(self.system.corpus_callosum, CorpusCallosum)
        self.assertIsInstance(self.system.cerebellum, CerebellumSynchronizer)
        self.assertIsInstance(self.system.prefrontal_cortex, PrefrontalCortexModule)

        # 验证区块链组件初始化
        if self.system.config.get('use_blockchain', False):
            self.assertIsInstance(self.system.da_layer, DataAvailabilityLayer)
            self.assertIsInstance(self.system.rollup_layer, RollupLayer)

        # 验证配置
        self.assertEqual(self.system.config['input_size'], self.config['input_size'])
        self.assertEqual(self.system.config['hidden_size'], self.config['hidden_size'])
        self.assertEqual(self.system.config['output_size'], self.config['output_size'])

    def test_process_tensor_input(self):
        """测试处理张量输入"""
        # 处理张量输入
        output = self.system.process(self.test_tensor_input, input_type="tensor")

        # 验证输出
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(1), self.config['output_size'])

        # 验证模式选择
        self.assertIn(self.system.mode_history[-1], ["left", "right", "dual"])

    def test_process_text_input(self):
        """测试处理文本输入"""
        # 处理文本输入
        output = self.system.process(self.test_text_input, input_type="text")

        # 验证输出
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(1), self.config['output_size'])

        # 验证模式选择
        self.assertIn(self.system.mode_history[-1], ["left", "right", "dual"])

    def test_mode_selection(self):
        """测试模式选择功能"""
        # 处理多个不同输入，统计模式选择
        modes = []
        for _ in range(10):
            # 随机生成不同特征的输入
            if np.random.rand() > 0.5:
                # 结构化数据偏向
                input_data = torch.randn(1, self.config['input_size'])
                input_data[:, :self.config['input_size'] // 2] *= 3  # 强化左半部分特征
                output = self.system.process(input_data, input_type="tensor")
            else:
                # 文本数据偏向
                input_data = "这是一个偏向语言理解的测试" + "文本内容" * np.random.randint(1, 5)
                output = self.system.process(input_data, input_type="text")

            modes.append(self.system.mode_history[-1])

        # 验证模式分布（应该不是单一模式）
        unique_modes = set(modes)
        self.assertGreater(len(unique_modes), 1)

    def test_message_passing(self):
        """测试模块间消息传递"""
        # 发送测试消息
        message_id = self.system._send_message(
            source_module="test",
            target_module="srmt",
            data={"test_data": 1.0},
            metadata={"type": "test_message"}
        )

        # 验证消息已加入队列
        self.assertGreater(len(self.system.message_queue), 0)

        # 处理消息
        self.system._process_messages()

        # 验证消息已处理（队列应该为空）
        self.assertEqual(len(self.system.message_queue), 0)

    def test_blockchain_data_flow(self):
        """测试区块链数据流"""
        if not self.system.config.get('use_blockchain', False):
            self.skipTest("Blockchain functionality disabled")

        # 创建测试事务
        transaction_id = "test_tx_123"
        input_data = {"test_key": "test_value"}
        processing_result = torch.randn(1, self.config['output_size'])

        # 执行区块链数据流处理
        result = self.system._process_blockchain_data_flow(
            transaction_id,
            input_data,
            processing_result
        )

        # 验证结果
        self.assertIn('input_id', result)
        self.assertIn('input_proof', result)

        # 验证DA层记录
        self.assertIn(result['input_id'], self.system.da_layer.metadata_index)

        # 检查是否达到批处理大小
        if len(self.system.rollup_layer.pending_txs) >= self.system.rollup_layer.batch_size:
            self.assertIn('batch_id', result)
            self.assertIn('batch_proof', result)

    def test_system_state(self):
        """测试系统状态维护"""
        # 处理输入
        output = self.system.process(self.test_tensor_input)

        # 验证系统状态已更新
        self.assertIn('last_input_id', self.system.current_state)
        self.assertIn('last_mode', self.system.current_state)
        self.assertIn('prefrontal_state', self.system.current_state)
        self.assertIn('metacognition', self.system.current_state)

        # 验证状态内容
        self.assertEqual(self.system.current_state['last_mode'], self.system.mode_history[-1])

    def test_optimization(self):
        """测试性能优化方法"""
        # 测试混合精度训练
        if torch.cuda.is_available():
            result = self.system.optimize_model()
            self.assertTrue(result)

        # 测试内存管理
        batch_size = self.system.manage_memory(self.config['input_size'])
        self.assertGreaterEqual(batch_size, 1)

        # 测试并行处理
        if torch.cuda.device_count() > 1:
            result = self.system.enable_parallel()
            self.assertTrue(result)

    def test_serialization(self):
        """测试系统序列化"""
        # 处理输入生成状态
        self.system.process(self.test_tensor_input)

        # 保存状态到临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp_path = temp.name
            state_dict = {
                'mode_history': self.system.mode_history,
                'current_state': self.system.current_state
            }
            json.dump(state_dict, temp)

        # 创建新系统并加载状态
        new_system = DualPathCoordinationSystem(self.config)
        with open(temp_path, 'r') as f:
            loaded_state = json.load(f)
            new_system.mode_history = loaded_state['mode_history']
            new_system.current_state = loaded_state['current_state']

        # 删除临时文件
        os.unlink(temp_path)

        # 验证状态加载正确
        self.assertEqual(new_system.mode_history, self.system.mode_history)
        self.assertEqual(new_system.current_state['last_mode'], self.system.current_state['last_mode'])

    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效输入类型
        with self.assertRaises(ValueError):
            self.system.process(self.test_tensor_input, input_type="invalid_type")

        # 测试空输入
        with self.assertRaises(ValueError):
            self.system.process("")

    def test_end_to_end(self):
        """端到端测试"""
        # 执行完整流程
        input_data = "这是一个完整流程的测试，验证系统各组件的协同工作"
        output = self.system.process(input_data, input_type="text")

        # 验证输出
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(1), self.config['output_size'])

        # 访问系统状态
        mode = self.system.current_state['last_mode']
        self.assertIn(mode, ["left", "right", "dual"])

        # 验证元认知输出
        self.assertIn('metacognition', self.system.current_state)

        # 打印结果摘要
        print(f"End-to-end test complete")
        print(f"Selected mode: {mode}")
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    unittest.main()