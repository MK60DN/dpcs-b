import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union


class SpatialDetector(nn.Module):
    """中脑路由选择器"""

    def __init__(self, input_size, hidden_size=128):
        super(SpatialDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 特征提取网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # 左脑、右脑、双脑三种模式

        # 历史性能记录
        self.history_probs = None
        self.mode_history = []
        self.performance_history = {}

        # 初始化模式性能记录
        self.mode_adaptation = ModeAdaptation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，预测处理模式概率

        Args:
            x: 输入特征

        Returns:
            mode_probs: 各处理模式的概率分布
        """
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))

        # 原始模式概率
        mode_logits = self.fc3(x)
        mode_probs = F.softmax(mode_logits, dim=1)

        # 考虑历史偏好
        if hasattr(self, 'history_probs') and self.history_probs is not None:
            # 结合历史偏好，实现平滑过渡
            mode_probs = 0.8 * mode_probs + 0.2 * self.history_probs

        # 更新历史
        self.history_probs = mode_probs.detach()

        return mode_probs

    def detect_mode(self, input_data: torch.Tensor) -> Tuple[str, np.ndarray]:
        """
        检测当前应该使用的处理模式

        Args:
            input_data: 输入数据

        Returns:
            selected_mode: 选择的处理模式
            probs: 各模式的概率
        """
        with torch.no_grad():
            # 提取特征
            features = F.gelu(self.ln1(self.fc1(input_data)))
            features = F.gelu(self.ln2(self.fc2(features)))

            # 计算模式概率
            logits = self.fc3(features)
            probs = F.softmax(logits, dim=1)

            # 考虑历史偏好和性能适应
            if hasattr(self, 'history_probs') and self.history_probs is not None:
                probs = 0.8 * probs + 0.2 * self.history_probs

            # 获取模式适应性调整
            bias = self.mode_adaptation.get_mode_bias()
            bias_tensor = torch.tensor([[bias['left'], bias['right'], bias['dual']]])

            # 应用偏好调整
            adjusted_probs = probs * (1.0 + 0.5 * bias_tensor)
            adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)

            # 选择模式
            mode_idx = torch.argmax(adjusted_probs, dim=1).item()
            modes = ["left", "right", "dual"]
            selected_mode = modes[mode_idx]

            # 更新历史
            self.history_probs = adjusted_probs
            self.mode_history.append(selected_mode)

            return selected_mode, adjusted_probs.numpy()

    def update_performance(self, mode: str, performance_score: float):
        """
        更新模式性能记录

        Args:
            mode: 处理模式
            performance_score: 性能评分

        Returns:
            bool: 更新是否成功
        """
        if mode not in ["left", "right", "dual"]:
            return False

        # 更新性能记录
        self.mode_adaptation.update_performance(mode, performance_score)

        return True

    def get_performance_stats(self):
        """
        获取性能统计信息

        Returns:
            dict: 性能统计信息
        """
        mode_counts = {"left": 0, "right": 0, "dual": 0}
        for mode in self.mode_history:
            mode_counts[mode] += 1

        stats = {
            "mode_counts": mode_counts,
            "total_decisions": len(self.mode_history),
            "current_bias": self.mode_adaptation.get_mode_bias(),
            "performance": self.mode_adaptation.mode_performance
        }

        return stats

    def reset_history(self):
        """
        重置历史记录

        Returns:
            bool: 重置是否成功
        """
        self.history_probs = None
        self.mode_history = []
        return True

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'detection_request':
            return self._process_detection_request(data)
        elif message_type == 'performance_update':
            return self._process_performance_update(data)
        elif message_type == 'stats_request':
            return self._process_stats_request()

        return {'status': 'unknown_message_type'}

    def _process_detection_request(self, data):
        """处理模式检测请求"""
        if not isinstance(data, torch.Tensor):
            if isinstance(data, list) or isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            else:
                return {'status': 'error', 'message': 'Invalid data format'}

        # 执行模式检测
        selected_mode, probs = self.detect_mode(data)

        return {
            'status': 'success',
            'selected_mode': selected_mode,
            'probabilities': probs.tolist()
        }

    def _process_performance_update(self, data):
        """处理性能更新请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        mode = data.get('mode', None)
        performance = data.get('performance', None)

        if mode is None or performance is None:
            return {'status': 'error', 'message': 'Missing required fields'}

        # 更新性能
        success = self.update_performance(mode, float(performance))

        return {
            'status': 'success' if success else 'error',
            'message': 'Performance updated' if success else 'Failed to update performance'
        }

    def _process_stats_request(self):
        """处理统计信息请求"""
        stats = self.get_performance_stats()

        return {
            'status': 'success',
            'stats': stats
        }


class ModeAdaptation:
    """处理模式适应机制"""

    def __init__(self, learning_rate=0.01, history_window=10):
        self.mode_performance = {
            'left': [],
            'right': [],
            'dual': []
        }
        self.learning_rate = learning_rate
        self.history_window = history_window
        self.mode_bias = {
            'left': 0.0,
            'right': 0.0,
            'dual': 0.0
        }

    def update_performance(self, mode, performance):
        """
        更新模式性能记录

        Args:
            mode: 处理模式
            performance: 性能评分

        Returns:
            None
        """
        self.mode_performance[mode].append(performance)

        # 保持窗口大小
        if len(self.mode_performance[mode]) > self.history_window:
            self.mode_performance[mode].pop(0)

        # 更新模式偏好
        self._adjust_bias()

    def get_mode_bias(self):
        """
        获取当前模式偏好

        Returns:
            dict: 模式偏好字典
        """
        return self.mode_bias

    def _adjust_bias(self):
        """
        根据历史表现调整模式偏好

        Returns:
            None
        """
        avg_performance = {}

        # 计算平均性能
        for mode in self.mode_performance:
            if self.mode_performance[mode]:
                avg_performance[mode] = sum(self.mode_performance[mode]) / len(self.mode_performance[mode])
            else:
                avg_performance[mode] = 0.0

        # 计算性能差异
        if all(avg_performance.values()):
            total_performance = sum(avg_performance.values())
            for mode in self.mode_bias:
                target_bias = avg_performance[mode] / total_performance
                self.mode_bias[mode] += self.learning_rate * (target_bias - self.mode_bias[mode])