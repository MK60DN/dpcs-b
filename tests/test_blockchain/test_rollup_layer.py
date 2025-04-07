import unittest
import json
import time
import random
import sys
import os
from unittest.mock import patch, MagicMock, Mock

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.blockchain.rollup_layer import RollupLayer, ZKProver
from dpcs.blockchain.da_layer import DataAvailabilityLayer


class TestRollupLayer(unittest.TestCase):
    """计算聚合层测试类"""

    def setUp(self):
        """测试前的初始化"""
        # 创建模拟DA层
        self.mock_da_layer = MagicMock(spec=DataAvailabilityLayer)
        self.mock_da_layer.store_data.return_value = {
            'data_id': 'test_id',
            'timestamp': time.time(),
            'proof': {'merkle_root': 'test_root'}
        }

        # 创建Rollup层实例
        self.rollup_type = "zk"  # 零知识Rollup
        self.batch_size = 5
        self.rollup = RollupLayer(
            rollup_type=self.rollup_type,
            batch_size=self.batch_size,
            da_layer=self.mock_da_layer
        )

        # 创建测试事务
        self.test_tx = {
            'input_id': 'input_123',
            'mode': 'left',
            'result_summary': {'mean': 0.5, 'std': 0.1}
        }

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.rollup.rollup_type, self.rollup_type)
        self.assertEqual(self.rollup.batch_size, self.batch_size)
        self.assertEqual(self.rollup.da_layer, self.mock_da_layer)
        self.assertEqual(self.rollup.pending_txs, [])
        self.assertEqual(self.rollup.processed_batches, [])
        self.assertEqual(self.rollup.challenge_periods, [])
        self.assertIsNotNone(self.rollup.state_root)

        # 验证ZK Rollup的证明生成器
        self.assertIsInstance(self.rollup.prover, ZKProver)

        # 测试乐观Rollup
        rollup_opt = RollupLayer(rollup_type="optimistic", batch_size=10)
        self.assertEqual(rollup_opt.rollup_type, "optimistic")
        self.assertIsNone(rollup_opt.prover)

    def test_add_transaction(self):
        """测试添加事务"""
        # 添加单个事务
        result = self.rollup.add_transaction(self.test_tx)

        # 验证返回结果
        self.assertIn('tx_id', result)
        self.assertEqual(result['status'], 'pending')

        # 验证事务添加到队列
        self.assertEqual(len(self.rollup.pending_txs), 1)
        self.assertEqual(self.rollup.pending_txs[0]['input_id'], 'input_123')

        # 验证事务添加了时间戳和ID
        self.assertIn('timestamp', self.rollup.pending_txs[0])
        self.assertIn('tx_id', self.rollup.pending_txs[0])

    def test_process_batch_zk(self):
        """测试处理ZK Rollup批次"""
        # 替换ZKProver为Mock
        mock_prover = MagicMock()
        mock_prover.generate_proof.return_value = {'proof_data': 'test_zk_proof'}
        self.rollup.prover = mock_prover

        # 添加足够的事务以触发批处理
        for i in range(self.batch_size):
            tx = {
                'input_id': f'input_{i}',
                'mode': 'left' if i % 2 == 0 else 'right',
                'result_summary': {'mean': random.random(), 'std': random.random() * 0.2}
            }
            self.rollup.add_transaction(tx)

        # 确保有足够事务
        self.assertEqual(len(self.rollup.pending_txs), self.batch_size)

        # 直接调用处理批次
        batch_result = self.rollup.process_batch()

        # 验证批处理结果
        self.assertIsInstance(batch_result, dict)
        self.assertIn('batch_id', batch_result)
        self.assertIn('new_state_root', batch_result)
        self.assertIn('batch_size', batch_result)
        self.assertIn('proof', batch_result)

        # 验证批大小
        self.assertEqual(batch_result['batch_size'], self.batch_size)

        # 验证事务队列已清空
        self.assertEqual(len(self.rollup.pending_txs), 0)

        # 验证批次已添加到处理记录
        self.assertEqual(len(self.rollup.processed_batches), 1)

        # 验证调用了ZK证明生成
        mock_prover.generate_proof.assert_called_once()

    def test_process_batch_optimistic(self):
        """测试处理乐观Rollup批次"""
        # 创建乐观Rollup实例
        optimistic_rollup = RollupLayer(
            rollup_type="optimistic",
            batch_size=self.batch_size,
            da_layer=self.mock_da_layer
        )

        # 添加足够的事务以触发批处理
        for i in range(self.batch_size):
            tx = {
                'input_id': f'input_{i}',
                'mode': 'dual' if i % 2 == 0 else 'right',
                'result_summary': {'mean': random.random(), 'std': random.random() * 0.2}
            }
            optimistic_rollup.add_transaction(tx)

        # 确保有足够事务
        self.assertEqual(len(optimistic_rollup.pending_txs), self.batch_size)

        # 直接调用处理批次
        batch_result = optimistic_rollup.process_batch()

        # 验证批处理结果
        self.assertIsInstance(batch_result, dict)
        self.assertIn('batch_id', batch_result)
        self.assertIn('new_state_root', batch_result)
        self.assertIn('batch_size', batch_result)
        self.assertIn('challenge_period_start', batch_result)
        self.assertIn('challenge_period_end', batch_result)

        # 验证批大小
        self.assertEqual(batch_result['batch_size'], self.batch_size)

        # 验证事务队列已清空
        self.assertEqual(len(optimistic_rollup.pending_txs), 0)

        # 验证批次已添加到处理记录
        self.assertEqual(len(optimistic_rollup.processed_batches), 1)

        # 验证创建了争议期
        self.assertEqual(len(optimistic_rollup.challenge_periods), 1)

    def test_process_batch_no_transactions(self):
        """测试处理空批次"""
        # 确保没有待处理事务
        self.rollup.pending_txs = []

        # 处理批次
        result = self.rollup.process_batch()

        # 验证返回结果
        self.assertEqual(result, {'status': 'no_transactions'})

    def test_process_batch_with_da_layer(self):
        """测试使用DA层处理批次"""
        # 添加足够的事务以触发批处理
        for i in range(self.batch_size):
            tx = {
                'input_id': f'input_{i}',
                'mode': 'left',
                'result_summary': {'mean': random.random()}
            }
            self.rollup.add_transaction(tx)

        # 处理批次
        batch_result = self.rollup.process_batch()

        # 验证DA层被调用存储结果
        self.mock_da_layer.store_data.assert_called()

    def test_start_challenge_period(self):
        """测试启动争议期"""
        # 创建乐观Rollup实例
        optimistic_rollup = RollupLayer(
            rollup_type="optimistic",
            batch_size=self.batch_size
        )

        # 启动争议期
        batch_id = "batch_123"
        batch_data = {"transactions": [self.test_tx]}
        result = optimistic_rollup._start_challenge_period(batch_id, batch_data)

        # 验证结果
        self.assertIn('batch_id', result)
        self.assertEqual(result['batch_id'], batch_id)
        self.assertIn('start_time', result)
        self.assertIn('end_time', result)
        self.assertGreater(result['end_time'], result['start_time'])
        self.assertIn('challenges', result)
        self.assertEqual(result['challenges'], [])

        # 验证争议期添加到列表
        self.assertEqual(len(optimistic_rollup.challenge_periods), 1)

    def test_submit_challenge(self):
        """测试提交争议"""
        # 创建乐观Rollup实例
        optimistic_rollup = RollupLayer(
            rollup_type="optimistic",
            batch_size=self.batch_size
        )

        # 添加一个活跃的争议期
        batch_id = "batch_123"
        challenge_period = optimistic_rollup._start_challenge_period(batch_id, {"transactions": [self.test_tx]})

        # 提交争议
        challenge_data = {
            'batch_id': batch_id,
            'transaction_index': 0,
            'reason': 'Invalid computation',
            'evidence': {'expected': 0.5, 'actual': 0.4}
        }
        result = optimistic_rollup.submit_challenge(challenge_data)

        # 验证结果
        self.assertEqual(result['status'], 'challenge_submitted')
        self.assertEqual(result['batch_id'], batch_id)

        # 验证争议添加到期间
        self.assertEqual(len(optimistic_rollup.challenge_periods[0]['challenges']), 1)
        self.assertEqual(optimistic_rollup.challenge_periods[0]['challenges'][0]['reason'],
                         'Invalid computation')

    def test_submit_challenge_expired(self):
        """测试提交过期争议"""
        # 创建乐观Rollup实例
        optimistic_rollup = RollupLayer(
            rollup_type="optimistic",
            batch_size=self.batch_size
        )

        # 添加一个已过期的争议期
        batch_id = "batch_123"
        challenge_period = {
            'batch_id': batch_id,
            'start_time': time.time() - 86500,  # 超过24小时
            'end_time': time.time() - 100,  # 已结束
            'challenges': []
        }
        optimistic_rollup.challenge_periods.append(challenge_period)

        # 提交争议
        challenge_data = {
            'batch_id': batch_id,
            'transaction_index': 0,
            'reason': 'Invalid computation',
            'evidence': {'expected': 0.5, 'actual': 0.4}
        }
        result = optimistic_rollup.submit_challenge(challenge_data)

        # 验证结果
        self.assertEqual(result['status'], 'challenge_period_expired')

        # 验证争议未添加
        self.assertEqual(len(optimistic_rollup.challenge_periods[0]['challenges']), 0)

    def test_verify_batch(self):
        """测试验证批次"""
        # 添加一个批次
        batch_id = "verify_batch_123"
        new_state_root = "new_root_hash"
        proof = {"proof_data": "test_proof"}
        batch_data = {
            'batch_id': batch_id,
            'new_state_root': new_state_root,
            'batch_size': 5,
            'proof': proof,
            'timestamp': time.time()
        }
        self.rollup.processed_batches.append(batch_data)

        # 验证批次
        result = self.rollup.verify_batch(batch_id, proof)

        # 验证结果
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['batch_id'], batch_id)

    def test_verify_batch_invalid(self):
        """测试验证无效批次"""
        # 添加一个批次
        batch_id = "verify_batch_123"
        new_state_root = "new_root_hash"
        proof = {"proof_data": "test_proof"}
        batch_data = {
            'batch_id': batch_id,
            'new_state_root': new_state_root,
            'batch_size': 5,
            'proof': proof,
            'timestamp': time.time()
        }
        self.rollup.processed_batches.append(batch_data)

        # 验证批次（无效证明）
        invalid_proof = {"proof_data": "invalid_proof"}

        # 修改验证方法返回False
        with patch.object(self.rollup.prover, 'verify_proof', return_value=False):
            result = self.rollup.verify_batch(batch_id, invalid_proof)

            # 验证结果
            self.assertFalse(result['is_valid'])
            self.assertEqual(result['batch_id'], batch_id)

    def test_verify_nonexistent_batch(self):
        """测试验证不存在的批次"""
        batch_id = "nonexistent_batch"
        proof = {"proof_data": "test_proof"}

        # 验证不存在的批次
        result = self.rollup.verify_batch(batch_id, proof)

        # 验证结果
        self.assertEqual(result['status'], 'batch_not_found')

    def test_get_batch_info(self):
        """测试获取批次信息"""
        # 添加一个批次
        batch_id = "info_batch_123"
        batch_data = {
            'batch_id': batch_id,
            'new_state_root': "root_hash",
            'batch_size': 3,
            'transactions': [
                {'tx_id': 'tx1', 'input_id': 'input1'},
                {'tx_id': 'tx2', 'input_id': 'input2'},
                {'tx_id': 'tx3', 'input_id': 'input3'}
            ],
            'timestamp': time.time()
        }
        self.rollup.processed_batches.append(batch_data)

        # 获取批次信息
        result = self.rollup.get_batch_info(batch_id)

        # 验证结果
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['batch_id'], batch_id)
        self.assertEqual(result['batch_size'], 3)
        self.assertIn('timestamp', result)
        self.assertIn('transactions', result)
        self.assertEqual(len(result['transactions']), 3)

    def test_get_nonexistent_batch_info(self):
        """测试获取不存在的批次信息"""
        batch_id = "nonexistent_batch"

        # 获取不存在的批次信息
        result = self.rollup.get_batch_info(batch_id)

        # 验证结果
        self.assertEqual(result['status'], 'batch_not_found')

    def test_get_rollup_stats(self):
        """测试获取Rollup统计信息"""
        # 添加一些待处理事务
        for i in range(3):
            self.rollup.add_transaction({
                'input_id': f'stats_input_{i}',
                'mode': 'left',
                'result_summary': {'mean': random.random()}
            })

        # 添加一些已处理批次
        for i in range(2):
            self.rollup.processed_batches.append({
                'batch_id': f'stats_batch_{i}',
                'batch_size': 5,
                'timestamp': time.time() - i * 3600
            })

        # 获取统计信息
        stats = self.rollup.get_rollup_stats()

        # 验证结果
        self.assertIsInstance(stats, dict)
        self.assertIn('pending_tx_count', stats)
        self.assertIn('processed_batch_count', stats)
        self.assertIn('total_processed_tx_count', stats)
        self.assertIn('rollup_type', stats)
        self.assertIn('last_state_root', stats)

        # 验证数值正确性
        self.assertEqual(stats['pending_tx_count'], 3)
        self.assertEqual(stats['processed_batch_count'], 2)
        self.assertEqual(stats['rollup_type'], self.rollup_type)

    def test_compute_batch_state(self):
        """测试计算批次状态"""
        # 创建一批事务
        transactions = [
            {
                'tx_id': f'tx_{i}',
                'input_id': f'input_{i}',
                'mode': 'left' if i % 2 == 0 else 'right',
                'result_summary': {'mean': 0.1 * i}
            }
            for i in range(5)
        ]

        # 计算批次状态
        old_state_root = self.rollup.state_root
        new_state = self.rollup._compute_batch_state(transactions, old_state_root)

        # 验证状态计算
        self.assertIsInstance(new_state, dict)
        self.assertIn('state_root', new_state)
        self.assertIn('tx_count', new_state)
        self.assertIn('state_update', new_state)

        # 验证状态根发生变化
        self.assertNotEqual(new_state['state_root'], old_state_root)

        # 验证事务数量正确
        self.assertEqual(new_state['tx_count'], 5)


class TestZKProver(unittest.TestCase):
    """零知识证明生成器测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.prover = ZKProver()

        # 创建测试数据
        self.test_data = {
            'batch_id': 'test_batch',
            'transactions': [
                {'tx_id': 'tx1', 'result': 0.5},
                {'tx_id': 'tx2', 'result': 0.7}
            ],
            'old_state_root': 'old_root_hash',
            'new_state_root': 'new_root_hash'
        }

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.prover, ZKProver)

    def test_generate_proof(self):
        """测试生成证明"""
        # 生成证明
        proof = self.prover.generate_proof(self.test_data)

        # 验证证明结构
        self.assertIsInstance(proof, dict)
        self.assertIn('proof_data', proof)
        self.assertIn('public_inputs', proof)
        self.assertIn('verification_key', proof)

    def test_verify_proof(self):
        """测试验证证明"""
        # 生成证明
        proof = self.prover.generate_proof(self.test_data)

        # 验证证明
        is_valid = self.prover.verify_proof(proof, self.test_data)

        # 验证结果
        self.assertTrue(is_valid)

    def test_verify_invalid_proof(self):
        """测试验证无效证明"""
        # 生成证明
        proof = self.prover.generate_proof(self.test_data)

        # 修改数据
        modified_data = self.test_data.copy()
        modified_data['new_state_root'] = 'tampered_root_hash'

        # 验证证明
        is_valid = self.prover.verify_proof(proof, modified_data)

        # 验证结果
        self.assertFalse(is_valid)

    def test_compute_proof_hash(self):
        """测试计算证明哈希"""
        # 计算哈希
        hash_result = self.prover._compute_proof_hash(self.test_data)

        # 验证哈希
        self.assertIsInstance(hash_result, str)
        self.assertGreater(len(hash_result), 0)

        # 验证相同数据产生相同哈希
        hash_repeat = self.prover._compute_proof_hash(self.test_data)
        self.assertEqual(hash_result, hash_repeat)

        # 验证不同数据产生不同哈希
        modified_data = self.test_data.copy()
        modified_data['new_state_root'] = 'different_root'
        different_hash = self.prover._compute_proof_hash(modified_data)
        self.assertNotEqual(hash_result, different_hash)


if __name__ == '__main__':
    unittest.main()