import time
import hashlib
import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union


class DataAvailabilityLayer:
    """数据可用性层实现"""

    def __init__(self, shard_count=16, sampling_ratio=0.1):
        self.shards = [[] for _ in range(shard_count)]
        self.metadata_index = {}
        self.shard_count = shard_count
        self.sampling_ratio = sampling_ratio

        # 存储层级
        self.hot_storage = {}  # 热存储（最近访问）
        self.warm_storage = {}  # 温存储（一般访问）
        self.cold_storage = {}  # 冷存储（很少访问）

        # 访问计数
        self.access_count = {}

        # 数据压缩标志
        self.use_compression = False

        # 数据可用性证明生成器
        self.da_prover = DataAvailabilityProver(shard_count)

    def store_data(self, data_id: str, data: Any, metadata: Optional[Dict] = None) -> Dict:
        """
        将数据分片存储

        Args:
            data_id: 数据ID
            data: 数据内容
            metadata: 元数据

        Returns:
            proof: 数据可用性证明
        """
        # 如果使用压缩，先压缩数据
        if self.use_compression and isinstance(data, (dict, list)):
            data = self._compress_data(data)

        # 序列化数据
        serialized_data = self._serialize_data(data)

        # 数据编码和纠删码应用
        encoded_data = self._apply_erasure_coding(serialized_data)

        # 分片数据
        data_shards = self._split_into_shards(encoded_data)
        for i, shard in enumerate(data_shards):
            self.shards[i % self.shard_count].append((data_id, i, shard))

        # 存储元数据
        self.metadata_index[data_id] = {
            'timestamp': time.time(),
            'size': len(serialized_data),
            'shards': self.shard_count,
            'metadata': metadata or {}
        }

        # 添加到热存储
        self.hot_storage[data_id] = {
            'data': data,
            'last_access': time.time()
        }
        self.access_count[data_id] = 1

        # 生成并返回证明
        proof = self.da_prover.generate_proof(serialized_data)

        return {
            'data_id': data_id,
            'timestamp': self.metadata_index[data_id]['timestamp'],
            'proof': proof
        }

    def verify_availability(self, data_id: str, proof: Dict) -> bool:
        """
        验证数据可用性

        Args:
            data_id: 数据ID
            proof: 数据可用性证明

        Returns:
            is_valid: 验证结果
        """
        if data_id not in self.metadata_index:
            return False

        # 执行随机抽样验证
        sample_indices = self._generate_random_indices(
            self.shard_count,
            int(self.shard_count * self.sampling_ratio)
        )

        # 验证抽样的分片
        for idx in sample_indices:
            shard_data = self._get_shard_by_index(data_id, idx)
            if not self._verify_shard(shard_data, proof, idx):
                return False

        # 更新访问计数
        if data_id in self.access_count:
            self.access_count[data_id] += 1

        # 更新存储层级
        self._update_storage_tier(data_id)

        return True

    def get_data(self, data_id: str) -> Optional[Any]:
        """
        获取数据

        Args:
            data_id: 数据ID

        Returns:
            data: 数据内容
        """
        # 从各存储层获取数据
        if data_id in self.hot_storage:
            data = self.hot_storage[data_id]['data']
            self.hot_storage[data_id]['last_access'] = time.time()
        elif data_id in self.warm_storage:
            data = self.warm_storage[data_id]['data']
            # 移至热存储
            self.hot_storage[data_id] = {
                'data': data,
                'last_access': time.time()
            }
            del self.warm_storage[data_id]
        elif data_id in self.cold_storage:
            data = self.cold_storage[data_id]['data']
            # 移至热存储
            self.hot_storage[data_id] = {
                'data': data,
                'last_access': time.time()
            }
            del self.cold_storage[data_id]
        else:
            # 重建数据
            data = self._rebuild_data(data_id)
            if data is None:
                return None

            # 添加到热存储
            self.hot_storage[data_id] = {
                'data': data,
                'last_access': time.time()
            }

        # 更新访问计数
        if data_id in self.access_count:
            self.access_count[data_id] += 1
        else:
            self.access_count[data_id] = 1

        # 更新存储层级
        self._update_storage_tier(data_id)

        # 解压数据（如果使用了压缩）
        if self.use_compression and isinstance(data, str) and data.startswith('compressed:'):
            data = self._decompress_data(data)

        return data

    def implement_tiered_storage(self):
        """
        实现分层存储策略

        Returns:
            bool: 是否成功
        """
        # 设置时间阈值
        current_time = time.time()
        warm_threshold = current_time - 3600  # 1小时
        cold_threshold = current_time - 86400  # 24小时

        # 移动热存储到温存储
        for data_id in list(self.hot_storage.keys()):
            if self.hot_storage[data_id]['last_access'] < warm_threshold:
                self.warm_storage[data_id] = self.hot_storage[data_id]
                del self.hot_storage[data_id]

        # 移动温存储到冷存储
        for data_id in list(self.warm_storage.keys()):
            if self.warm_storage[data_id]['last_access'] < cold_threshold:
                self.cold_storage[data_id] = self.warm_storage[data_id]
                del self.warm_storage[data_id]

        return True

    def enable_data_compression(self):
        """
        启用数据压缩

        Returns:
            bool: 是否成功
        """
        self.use_compression = True
        return True

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            stats: 存储统计信息
        """
        return {
            'hot_storage_count': len(self.hot_storage),
            'warm_storage_count': len(self.warm_storage),
            'cold_storage_count': len(self.cold_storage),
            'total_data_count': len(self.metadata_index),
            'total_shard_count': sum(len(shard) for shard in self.shards)
        }

    def _serialize_data(self, data: Any) -> str:
        """将数据序列化为字符串"""
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        elif isinstance(data, (int, float, str, bool)):
            return str(data)
        elif isinstance(data, bytes):
            return data.decode('utf-8', errors='replace')
        elif isinstance(data, np.ndarray):
            return json.dumps(data.tolist())
        else:
            # 简化实现，实际系统应支持更多数据类型
            return str(data)

    def _apply_erasure_coding(self, data: str) -> List[str]:
        """应用纠删码"""
        # 简化实现，实际系统应使用Reed-Solomon编码
        # 这里使用简单的复制冗余
        chunk_size = max(1, len(data) // (self.shard_count // 2))
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 确保有足够的块
        while len(chunks) < self.shard_count:
            chunks.append("")

        return chunks

    def _split_into_shards(self, encoded_data: List[str]) -> List[str]:
        """将编码后的数据分片"""
        return encoded_data

    def _generate_random_indices(self, max_index: int, count: int) -> List[int]:
        """生成随机索引"""
        return random.sample(range(max_index), min(count, max_index))

    def _get_shard_by_index(self, data_id: str, index: int) -> Optional[str]:
        """根据索引获取分片数据"""
        for shard_list in self.shards:
            for shard_id, shard_index, shard_data in shard_list:
                if shard_id == data_id and shard_index == index:
                    return shard_data
        return None

    def _verify_shard(self, shard_data: Optional[str], proof: Dict, index: int) -> bool:
        """验证分片"""
        if shard_data is None:
            return False

        # 简化实现，实际系统应使用密码学验证
        # 这里只检查分片存在
        return True

    def _rebuild_data(self, data_id: str) -> Optional[Any]:
        """重建数据"""
        if data_id not in self.metadata_index:
            return None

        # 收集所有分片
        shards = []
        for i in range(self.shard_count):
            shard = self._get_shard_by_index(data_id, i)
            if shard is not None:
                shards.append((i, shard))

        # 检查是否有足够的分片重建数据
        if len(shards) < self.shard_count // 2:
            return None

        # 根据索引排序分片
        shards.sort(key=lambda x: x[0])

        # 合并分片
        reconstructed_data = ''.join(shard for _, shard in shards)

        # 尝试解析JSON
        try:
            return json.loads(reconstructed_data)
        except:
            return reconstructed_data

    def _update_storage_tier(self, data_id: str):
        """更新存储层级"""
        # 简化实现，实际系统应根据访问频率和访问模式动态调整
        access_count = self.access_count.get(data_id, 0)

        # 访问计数范围划分存储层级
        if access_count > 10:
            # 频繁访问，放入热存储
            if data_id in self.warm_storage:
                self.hot_storage[data_id] = self.warm_storage[data_id]
                self.hot_storage[data_id]['last_access'] = time.time()
                del self.warm_storage[data_id]
            elif data_id in self.cold_storage:
                self.hot_storage[data_id] = self.cold_storage[data_id]
                self.hot_storage[data_id]['last_access'] = time.time()
                del self.cold_storage[data_id]

    def _compress_data(self, data: Any) -> str:
        """压缩数据"""
        # 简化实现，实际系统应使用真实压缩算法
        serialized = json.dumps(data)
        return f"compressed:{serialized}"

    def _decompress_data(self, compressed_data: str) -> Any:
        """解压数据"""
        # 简化实现，实际系统应使用真实解压算法
        if not compressed_data.startswith('compressed:'):
            return compressed_data

        serialized = compressed_data[11:]  # 移除 'compressed:' 前缀
        try:
            return json.loads(serialized)
        except:
            return serialized


class DataAvailabilityProver:
    """数据可用性证明生成器"""

    def __init__(self, shard_count=16):
        self.shard_count = shard_count

    def generate_proof(self, data: str) -> Dict[str, Any]:
        """
        生成数据可用性证明

        Args:
            data: 数据

        Returns:
            proof: 证明
        """
        # 应用纠删码（简化实现）
        chunk_size = max(1, len(data) // (self.shard_count // 2))
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 确保有足够的块
        while len(chunks) < self.shard_count:
            chunks.append("")

        encoded_data = chunks

        # 计算Merkle树（简化实现）
        merkle_tree = self._compute_merkle_tree(encoded_data)
        merkle_root = merkle_tree[0]

        # 生成随机抽样点
        sample_indices = self._generate_random_indices(
            self.shard_count,
            min(self.shard_count, 10)
        )

        # 生成抽样证明
        sample_proofs = []
        for idx in sample_indices:
            shard = encoded_data[idx] if idx < len(encoded_data) else ""
            merkle_proof = self._generate_merkle_proof(merkle_tree, idx)
            sample_proofs.append({
                'index': idx,
                'merkle_proof': merkle_proof
            })

        return {
            'merkle_root': merkle_root,
            'sample_proofs': sample_proofs,
            'shard_count': self.shard_count
        }

    def _compute_merkle_tree(self, data_chunks: List[str]) -> List[str]:
        """计算Merkle树"""
        # 计算数据块哈希
        leaves = [self._hash(chunk) for chunk in data_chunks]

        # 确保叶子节点数量为2的幂
        while len(leaves) & (len(leaves) - 1) != 0:
            leaves.append(self._hash(""))

        # 构建Merkle树
        tree = leaves.copy()
        level_size = len(leaves)
        tree_size = 2 * level_size - 1
        tree_index = level_size

        # 从叶子向上构建
        while level_size > 1:
            for i in range(0, level_size, 2):
                if i + 1 < level_size:
                    parent_hash = self._hash(tree[i] + tree[i + 1])
                else:
                    parent_hash = self._hash(tree[i] + tree[i])

                tree.append(parent_hash)
                tree_index += 1

            level_size = level_size // 2

        return tree

    def _generate_merkle_proof(self, merkle_tree: List[str], index: int) -> List[str]:
        """生成Merkle证明"""
        # 简化实现，实际系统应生成完整的Merkle证明
        return [merkle_tree[0]]  # 仅返回根哈希

    def _generate_random_indices(self, max_index: int, count: int) -> List[int]:
        """生成随机索引"""
        return random.sample(range(max_index), min(count, max_index))

    def _hash(self, data: str) -> str:
        """计算哈希值"""
        return hashlib.sha256(data.encode()).hexdigest()