import unittest
import json
import time
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.blockchain.da_layer import DataAvailabilityLayer, DataAvailabilityProver


class TestDataAvailabilityLayer(unittest.TestCase):
    """数据可用性层测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.shard_count = 16
        self.sampling_ratio = 0.1
        self.da_layer = DataAvailabilityLayer(
            shard_count=self.shard_count,
            sampling_ratio=self.sampling_ratio
        )

        # 创建测试数据
        self.test_data_id = "test_data_123"
        self.test_data_simple = "test data string"
        self.test_data_dict = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        self.test_metadata = {"type": "test", "timestamp": time.time()}

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.da_layer.shards), self.shard_count)
        self.assertEqual(self.da_layer.metadata_index, {})
        self.assertEqual(self.da_layer.shard_count, self.shard_count)
        self.assertEqual(self.da_layer.sampling_ratio, self.sampling_ratio)
        self.assertEqual(self.da_layer.hot_storage, {})
        self.assertEqual(self.da_layer.warm_storage, {})
        self.assertEqual(self.da_layer.cold_storage, {})
        self.assertEqual(self.da_layer.access_count, {})
        self.assertFalse(self.da_layer.use_compression)
        self.assertIsInstance(self.da_layer.da_prover, DataAvailabilityProver)

    def test_store_data_simple(self):
        """测试存储简单数据"""
        # 存储简单数据
        proof = self.da_layer.store_data(
            self.test_data_id,
            self.test_data_simple,
            self.test_metadata
        )

        # 验证返回的证明
        self.assertIsInstance(proof, dict)
        self.assertEqual(proof['data_id'], self.test_data_id)
        self.assertIn('timestamp', proof)
        self.assertIn('proof', proof)

        # 验证数据存储
        self.assertIn(self.test_data_id, self.da_layer.metadata_index)
        self.assertIn(self.test_data_id, self.da_layer.hot_storage)
        self.assertEqual(self.da_layer.hot_storage[self.test_data_id]['data'], self.test_data_simple)

        # 验证元数据
        stored_metadata = self.da_layer.metadata_index[self.test_data_id]['metadata']
        self.assertEqual(stored_metadata, self.test_metadata)

        # 验证分片存储
        has_shards = False
        for shard_list in self.da_layer.shards:
            for shard_id, _, _ in shard_list:
                if shard_id == self.test_data_id:
                    has_shards = True
                    break
            if has_shards:
                break
        self.assertTrue(has_shards)

    def test_store_data_complex(self):
        """测试存储复杂数据"""
        # 存储字典数据
        proof = self.da_layer.store_data(
            self.test_data_id,
            self.test_data_dict,
            self.test_metadata
        )

        # 验证返回的证明
        self.assertIsInstance(proof, dict)

        # 验证数据存储
        self.assertIn(self.test_data_id, self.da_layer.metadata_index)
        self.assertIn(self.test_data_id, self.da_layer.hot_storage)
        self.assertEqual(self.da_layer.hot_storage[self.test_data_id]['data'], self.test_data_dict)

    def test_get_data(self):
        """测试获取数据"""
        # 存储数据
        self.da_layer.store_data(
            self.test_data_id,
            self.test_data_dict,
            self.test_metadata
        )

        # 获取数据
        retrieved_data = self.da_layer.get_data(self.test_data_id)

        # 验证数据
        self.assertEqual(retrieved_data, self.test_data_dict)

        # 验证访问计数更新
        self.assertEqual(self.da_layer.access_count[self.test_data_id], 2)  # 存储 + 获取

    def test_get_nonexistent_data(self):
        """测试获取不存在的数据"""
        # 获取不存在的数据
        retrieved_data = self.da_layer.get_data("nonexistent_id")

        # 验证返回None
        self.assertIsNone(retrieved_data)

    def test_verify_availability(self):
        """测试验证数据可用性"""
        # 存储数据
        proof = self.da_layer.store_data(
            self.test_data_id,
            self.test_data_simple,
            self.test_metadata
        )

        # 验证数据可用性
        is_available = self.da_layer.verify_availability(self.test_data_id, proof['proof'])

        # 验证结果
        self.assertTrue(is_available)

        # 验证不存在的数据
        is_nonexistent_available = self.da_layer.verify_availability("nonexistent_id", proof['proof'])
        self.assertFalse(is_nonexistent_available)

    def test_implement_tiered_storage(self):
        """测试分层存储实现"""
        # 存储多个数据
        for i in range(5):
            data_id = f"test_data_{i}"
            self.da_layer.store_data(data_id, f"Data {i}", {"index": i})

        # 所有数据应该在热存储中
        self.assertEqual(len(self.da_layer.hot_storage), 5)
        self.assertEqual(len(self.da_layer.warm_storage), 0)
        self.assertEqual(len(self.da_layer.cold_storage), 0)

        # 手动修改部分数据的最后访问时间以模拟时间流逝
        current_time = time.time()
        for i in range(3):
            data_id = f"test_data_{i}"
            self.da_layer.hot_storage[data_id]['last_access'] = current_time - 7200  # 2小时前

        # 实现分层存储
        result = self.da_layer.implement_tiered_storage()

        # 验证结果
        self.assertTrue(result)

        # 验证数据迁移到温存储
        self.assertEqual(len(self.da_layer.hot_storage), 2)  # 应该剩下 2 个
        self.assertEqual(len(self.da_layer.warm_storage), 3)  # 应该迁移 3 个
        self.assertEqual(len(self.da_layer.cold_storage), 0)  # 没有冷存储

    def test_data_compression(self):
        """测试数据压缩"""
        # 启用数据压缩
        result = self.da_layer.enable_data_compression()
        self.assertTrue(result)
        self.assertTrue(self.da_layer.use_compression)

        # 存储可压缩数据
        large_data = {"large_array": [i for i in range(1000)]}
        proof = self.da_layer.store_data("compressed_data", large_data, {"compressed": True})

        # 获取数据
        retrieved_data = self.da_layer.get_data("compressed_data")

        # 验证数据完整性
        self.assertEqual(retrieved_data, large_data)

    def test_storage_stats(self):
        """测试存储统计"""
        # 存储一些数据
        for i in range(10):
            data_id = f"stats_data_{i}"
            self.da_layer.store_data(data_id, f"Stats Data {i}", {"stats_index": i})

        # 获取存储统计
        stats = self.da_layer.get_storage_stats()

        # 验证统计信息
        self.assertIsInstance(stats, dict)
        self.assertIn('hot_storage_count', stats)
        self.assertIn('warm_storage_count', stats)
        self.assertIn('cold_storage_count', stats)
        self.assertIn('total_data_count', stats)
        self.assertIn('total_shard_count', stats)

        # 验证数值正确性
        self.assertEqual(stats['hot_storage_count'], 10)
        self.assertEqual(stats['total_data_count'], 10)

    def test_storage_tier_transition(self):
        """测试存储层级转换"""
        # 存储数据
        data_id = "tier_test_data"
        self.da_layer.store_data(data_id, "Tier Test Data", {"tier": "test"})

        # 数据应该在热存储中
        self.assertIn(data_id, self.da_layer.hot_storage)

        # 手动将数据移动到温存储
        data = self.da_layer.hot_storage[data_id]
        self.da_layer.warm_storage[data_id] = data
        del self.da_layer.hot_storage[data_id]

        # 验证数据不在热存储而在温存储
        self.assertNotIn(data_id, self.da_layer.hot_storage)
        self.assertIn(data_id, self.da_layer.warm_storage)

        # 获取数据应该将其移回热存储
        retrieved_data = self.da_layer.get_data(data_id)

        # 验证数据移回热存储
        self.assertIn(data_id, self.da_layer.hot_storage)
        self.assertNotIn(data_id, self.da_layer.warm_storage)

    def test_serialization_methods(self):
        """测试序列化方法"""
        # 测试字典序列化
        dict_data = {"test": "value", "number": 123}
        serialized_dict = self.da_layer._serialize_data(dict_data)
        self.assertIsInstance(serialized_dict, str)
        self.assertEqual(json.loads(serialized_dict), dict_data)

        # 测试字符串序列化
        string_data = "test string"
        serialized_string = self.da_layer._serialize_data(string_data)
        self.assertEqual(serialized_string, string_data)

        # 测试数字序列化
        number_data = 12345
        serialized_number = self.da_layer._serialize_data(number_data)
        self.assertEqual(serialized_number, str(number_data))

        # 测试NumPy数组序列化
        numpy_data = np.array([1, 2, 3, 4, 5])
        serialized_numpy = self.da_layer._serialize_data(numpy_data)
        self.assertIsInstance(serialized_numpy, str)
        self.assertEqual(json.loads(serialized_numpy), numpy_data.tolist())


class TestDataAvailabilityProver(unittest.TestCase):
    """数据可用性证明生成器测试类"""

    def setUp(self):
        """测试前的初始化"""
        self.shard_count = 16
        self.prover = DataAvailabilityProver(self.shard_count)

        # 创建测试数据
        self.test_data = "Test data for proving availability"

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.prover.shard_count, self.shard_count)

    def test_generate_proof(self):
        """测试生成证明"""
        # 生成证明
        proof = self.prover.generate_proof(self.test_data)

        # 验证证明结构
        self.assertIsInstance(proof, dict)
        self.assertIn('merkle_root', proof)
        self.assertIn('sample_proofs', proof)
        self.assertIn('shard_count', proof)

        # 验证证明内容
        self.assertIsInstance(proof['merkle_root'], str)
        self.assertIsInstance(proof['sample_proofs'], list)
        self.assertEqual(proof['shard_count'], self.shard_count)

        # 验证抽样证明
        for sample_proof in proof['sample_proofs']:
            self.assertIn('index', sample_proof)
            self.assertIn('merkle_proof', sample_proof)
            self.assertIsInstance(sample_proof['index'], int)
            self.assertIsInstance(sample_proof['merkle_proof'], list)

    def test_compute_merkle_tree(self):
        """测试计算Merkle树"""
        # 创建数据块
        data_chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]

        # 计算Merkle树
        merkle_tree = self.prover._compute_merkle_tree(data_chunks)

        # 验证Merkle树的基本属性
        self.assertIsInstance(merkle_tree, list)
        self.assertGreater(len(merkle_tree), len(data_chunks))

        # 验证叶子节点的哈希
        for i, chunk in enumerate(data_chunks):
            chunk_hash = self.prover._hash(chunk)
            self.assertEqual(merkle_tree[i], chunk_hash)

    def test_hash_function(self):
        """测试哈希函数"""
        # 测试空字符串哈希
        empty_hash = self.prover._hash("")
        self.assertIsInstance(empty_hash, str)
        self.assertEqual(len(empty_hash), 64)  # SHA-256 哈希长度

        # 测试具体数据哈希
        data_hash = self.prover._hash("specific data")
        self.assertIsInstance(data_hash, str)
        self.assertEqual(len(data_hash), 64)

        # 测试哈希的确定性
        data_hash_repeat = self.prover._hash("specific data")
        self.assertEqual(data_hash, data_hash_repeat)

        # 测试不同数据的哈希不同
        different_hash = self.prover._hash("different data")
        self.assertNotEqual(data_hash, different_hash)


if __name__ == '__main__':
    unittest.main()