import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional, List, Union


class CerebellumSynchronizer(nn.Module):
    """小脑时序同步模块"""

    def __init__(self, input_dim, hidden_dim=128, lstm_layers=2):
        super(CerebellumSynchronizer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # 时间编码器
        self.time_encoder = nn.Linear(input_dim, hidden_dim)

        # LSTM网络
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # 节奏控制器 - 可学习参数
        self.rhythm_controller = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.phase_param = nn.Parameter(torch.zeros(1))

        # 多尺度时间表示
        self.time_constants = [1, 2, 4, 8, 16]  # 不同时间尺度的时间常数

        # 时序记忆
        self.temporal_memory = []
        self.max_memory_length = 50

    def forward(self, x: torch.Tensor, time_steps: int = 5) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征
            time_steps: 时间步长

        Returns:
            output: 同步后的输出
        """
        batch_size = x.size(0)

        # 时间编码
        encoded = self.time_encoder(x)

        # 准备序列输入
        rhythm = self.rhythm_controller.repeat(batch_size, time_steps, 1)
        sequence = encoded.unsqueeze(1).repeat(1, time_steps, 1) * rhythm

        # LSTM处理
        lstm_out, _ = self.lstm(sequence)

        # 节奏控制
        rhythmic_output = lstm_out[:, -1, :] * (1 + 0.1 * torch.sin(self.phase_param))

        # 生成输出
        output = self.output_layer(rhythmic_output)

        # 更新时序记忆
        self._update_temporal_memory(encoded.detach())

        return output

    def synchronize(self, signals: torch.Tensor, time_steps: int = 5, rhythm_factor: float = 0.8) -> torch.Tensor:
        """
        对信号进行时间同步

        Args:
            signals: 输入信号
            time_steps: 时间步长
            rhythm_factor: 节奏因子

        Returns:
            output: 同步后的输出
        """
        batch_size = signals.size(0)

        # 时间编码
        encoded = self.time_encoder(signals)

        # 准备序列输入
        rhythm = self.rhythm_controller * rhythm_factor
        rhythm = rhythm.repeat(batch_size, time_steps, 1)
        sequence = encoded.unsqueeze(1).repeat(1, time_steps, 1) * rhythm

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(sequence)

        # 节奏控制
        rhythmic_output = lstm_out[:, -1, :] * (1 + 0.1 * torch.sin(self.phase_param))

        # 生成输出
        output = self.output_layer(rhythmic_output)

        return output

    def _implement_multiscale_time(self, time_constants: List[float], input_signal: torch.Tensor) -> torch.Tensor:
        """
        实现多尺度时间表示

        Args:
            time_constants: 时间常数列表
            input_signal: 输入信号

        Returns:
            multiscale_repr: 多尺度表示
        """
        batch_size = input_signal.size(0)
        signal_dim = input_signal.size(1)
        num_scales = len(time_constants)

        # 初始化多尺度表示
        multiscale_repr = torch.zeros(batch_size, num_scales, signal_dim).to(input_signal.device)

        # 上一时刻的表示 (初始为0)
        prev_repr = torch.zeros_like(multiscale_repr)

        # 应用指数滤波
        for k, tau in enumerate(time_constants):
            decay_factor = torch.exp(-1.0 / tau)
            multiscale_repr[:, k, :] = decay_factor * prev_repr[:, k, :] + (1 - decay_factor) * input_signal

        return multiscale_repr

    def process_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        处理时间序列

        Args:
            sequence: 输入序列 [batch_size, seq_len, input_dim]

        Returns:
            output: 处理后的序列
        """
        batch_size, seq_len, _ = sequence.shape

        # 时间编码
        encoded_sequence = self.time_encoder(sequence)

        # 多尺度时间表示
        multiscale_repr = []
        for t in range(seq_len):
            time_repr = self._implement_multiscale_time(
                self.time_constants,
                sequence[:, t, :]
            )
            multiscale_repr.append(time_repr)

        # 拼接多尺度表示
        multiscale_tensor = torch.cat([r.unsqueeze(1) for r in multiscale_repr], dim=1)

        # 展平多尺度表示，以便LSTM处理
        flat_repr = multiscale_tensor.view(batch_size, seq_len, -1)

        # LSTM处理
        lstm_out, _ = self.lstm(flat_repr)

        # 最终输出
        output = []
        for t in range(seq_len):
            out_t = self.output_layer(lstm_out[:, t, :])
            output.append(out_t)

        return torch.stack(output, dim=1)

    def _update_temporal_memory(self, encoded_state: torch.Tensor):
        """
        更新时序记忆

        Args:
            encoded_state: 编码状态

        Returns:
            None
        """
        # 将编码状态添加到记忆
        if isinstance(encoded_state, torch.Tensor):
            self.temporal_memory.append(encoded_state.detach().mean(dim=0))

        # 限制记忆长度
        if len(self.temporal_memory) > self.max_memory_length:
            self.temporal_memory.pop(0)

    def detect_pattern(self, sequence: torch.Tensor, pattern_length: int = 5) -> Dict[str, Any]:
        """
        检测时间序列中的模式

        Args:
            sequence: 输入序列
            pattern_length: 模式长度

        Returns:
            pattern_info: 模式信息
        """
        with torch.no_grad():
            # 编码序列
            encoded_sequence = self.time_encoder(sequence)

            # 提取子序列作为可能的模式
            patterns = []
            for i in range(len(encoded_sequence) - pattern_length + 1):
                pattern = encoded_sequence[i:i + pattern_length]
                patterns.append(pattern)

            # 寻找重复模式
            pattern_scores = []

            for i, pattern in enumerate(patterns):
                matches = []

                for j, other_pattern in enumerate(patterns):
                    if i != j:
                        # 计算相似度
                        similarity = F.cosine_similarity(
                            pattern.view(1, -1),
                            other_pattern.view(1, -1)
                        ).item()

                        if similarity > 0.8:  # 相似度阈值
                            matches.append((j, similarity))

                pattern_scores.append((i, len(matches), matches))

            # 找出最频繁的模式
            pattern_scores.sort(key=lambda x: x[1], reverse=True)

            if not pattern_scores or pattern_scores[0][1] == 0:
                return {
                    'has_pattern': False,
                    'message': 'No significant patterns detected'
                }

            best_pattern_idx = pattern_scores[0][0]
            best_pattern = patterns[best_pattern_idx]

            # 计算模式周期
            matches = pattern_scores[0][2]
            if matches:
                match_indices = [m[0] for m in matches]
                if len(match_indices) > 1:
                    intervals = [match_indices[i + 1] - match_indices[i] for i in range(len(match_indices) - 1)]
                    avg_interval = sum(intervals) / len(intervals)
                else:
                    avg_interval = None
            else:
                avg_interval = None

            return {
                'has_pattern': True,
                'pattern_start_idx': best_pattern_idx,
                'pattern_length': pattern_length,
                'pattern_frequency': len(matches),
                'avg_interval': avg_interval,
                'pattern_strength': pattern_scores[0][1] / len(patterns) if patterns else 0
            }

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'sync_request':
            return self._process_sync_request(data)
        elif message_type == 'pattern_detection_request':
            return self._process_pattern_detection_request(data)
        elif message_type == 'sequence_processing_request':
            return self._process_sequence_processing_request(data)

        return {'status': 'unknown_message_type'}

    def _process_sync_request(self, data):
        """处理同步请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        signals = data.get('signals', None)
        time_steps = data.get('time_steps', 5)
        rhythm_factor = data.get('rhythm_factor', 0.8)

        if signals is None:
            return {'status': 'error', 'message': 'Missing signals'}

        # 确保是张量
        if not isinstance(signals, torch.Tensor):
            signals = torch.tensor(signals, dtype=torch.float)

        # 执行同步
        output = self.synchronize(signals, time_steps, rhythm_factor)

        return {
            'status': 'success',
            'synchronized_output': output.detach().numpy()
        }

    def _process_pattern_detection_request(self, data):
        """处理模式检测请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        sequence = data.get('sequence', None)
        pattern_length = data.get('pattern_length', 5)

        if sequence is None:
            return {'status': 'error', 'message': 'Missing sequence'}

        # 确保是张量
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float)

        # 检测模式
        pattern_info = self.detect_pattern(sequence, pattern_length)

        return {
            'status': 'success',
            'pattern_info': pattern_info
        }

    def _process_sequence_processing_request(self, data):
        """处理序列处理请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        sequence = data.get('sequence', None)

        if sequence is None:
            return {'status': 'error', 'message': 'Missing sequence'}

        # 确保是张量
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float)

        # 处理序列
        output = self.process_sequence(sequence)

        return {
            'status': 'success',
            'processed_sequence': output.detach().numpy()
        }