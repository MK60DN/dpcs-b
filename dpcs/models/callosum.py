import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, Optional, List, Union


class CorpusCallosum(nn.Module):
    """胼胝体信息融合模块"""

    def __init__(self, feature_dim, fusion_dim=256):
        super(CorpusCallosum, self).__init__()

        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim

        # 投影网络
        self.left_proj = nn.Linear(feature_dim, fusion_dim)
        self.right_proj = nn.Linear(feature_dim, fusion_dim)

        # 融合网络
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        self.gate_layer = nn.Linear(fusion_dim * 2, fusion_dim)

        # 可学习的融合权重参数
        self.alpha_param = nn.Parameter(torch.zeros(1))

        # 多头注意力融合
        self.multihead_attention = MultiHeadAttention(hidden_size=fusion_dim, num_heads=4)

        # 融合模式选择器
        self.fusion_mode_selector = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 3)  # 三种融合模式：加权、门控、注意力
        )

    def forward(self, left_features: torch.Tensor, right_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，融合左右脑特征

        Args:
            left_features: 左脑特征
            right_features: 右脑特征

        Returns:
            fused_output: 融合后的特征
        """
        # 投影到共同空间
        left_encoded = self.left_proj(left_features)
        right_encoded = self.right_proj(right_features)

        # 计算对齐矩阵
        alignment_scores = torch.matmul(left_encoded, right_encoded.transpose(-2, -1)) / math.sqrt(
            left_encoded.size(-1))
        alignment_matrix = F.softmax(alignment_scores, dim=-1)

        # 对齐表示
        left_aligned = torch.matmul(alignment_matrix, right_encoded)

        # 融合信息
        # 1. 加权融合
        alpha = torch.sigmoid(self.alpha_param)
        weighted_fusion = alpha * left_encoded + (1 - alpha) * left_aligned

        # 2. 门控融合
        gate = torch.sigmoid(self.gate_layer(torch.cat([left_encoded, left_aligned], dim=-1)))
        gated_fusion = gate * left_encoded + (1 - gate) * left_aligned

        # 3. 注意力融合
        attentive_fusion = self.multihead_attention(left_encoded, right_encoded, right_encoded)

        # 组合多种融合结果
        fusion_input = torch.cat([weighted_fusion, gated_fusion], dim=-1)
        fused_output = self.fusion_layer(fusion_input)

        # 残差连接
        fused_output = fused_output + 0.1 * attentive_fusion

        return fused_output

    def align_semantics(self, srmt_output: torch.Tensor, camel_output: torch.Tensor) -> torch.Tensor:
        """
        对齐并融合左右脑的输出

        Args:
            srmt_output: 左脑输出
            camel_output: 右脑输出

        Returns:
            fused_output: 融合后的输出
        """
        # 投影到共享空间
        left_encoded = self.left_proj(srmt_output)
        right_encoded = self.right_proj(camel_output)

        # 计算对齐矩阵
        alignment_scores = torch.matmul(left_encoded, right_encoded.transpose(-2, -1)) / math.sqrt(
            left_encoded.size(-1))
        alignment_matrix = F.softmax(alignment_scores, dim=-1)

        # 对齐表示
        left_aligned = torch.matmul(alignment_matrix, right_encoded)

        # 决定融合模式
        fusion_features = torch.cat([left_encoded, left_aligned], dim=-1)
        fusion_mode_logits = self.fusion_mode_selector(fusion_features)
        fusion_mode_weights = F.softmax(fusion_mode_logits, dim=-1)

        # 计算不同融合模式的结果
        # 1. 加权融合
        alpha = torch.sigmoid(self.alpha_param)
        weighted_fusion = alpha * left_encoded + (1 - alpha) * left_aligned

        # 2. 门控融合
        gate = torch.sigmoid(self.gate_layer(fusion_features))
        gated_fusion = gate * left_encoded + (1 - gate) * left_aligned

        # 3. 注意力融合
        attentive_fusion = self.multihead_attention(left_encoded, right_encoded, right_encoded)

        # 融合各种模式的结果
        mode1_weight = fusion_mode_weights[:, 0:1]
        mode2_weight = fusion_mode_weights[:, 1:2]
        mode3_weight = fusion_mode_weights[:, 2:3]

        # 加权组合
        combined_fusion = (
                mode1_weight * weighted_fusion +
                mode2_weight * gated_fusion +
                mode3_weight * attentive_fusion
        )

        # 最终融合层
        fusion_input = torch.cat([combined_fusion, left_encoded], dim=-1)
        fused_output = self.fusion_layer(fusion_input)

        return fused_output

    def semantic_alignment_score(self, left_features: torch.Tensor, right_features: torch.Tensor) -> float:
        """
        计算左右脑特征之间的语义对齐分数

        Args:
            left_features: 左脑特征
            right_features: 右脑特征

        Returns:
            alignment_score: 对齐分数
        """
        with torch.no_grad():
            # 投影到共享空间
            left_encoded = self.left_proj(left_features)
            right_encoded = self.right_proj(right_features)

            # 归一化表示
            left_norm = F.normalize(left_encoded, p=2, dim=-1)
            right_norm = F.normalize(right_encoded, p=2, dim=-1)

            # 计算余弦相似度
            similarity = torch.matmul(left_norm, right_norm.transpose(-2, -1))

            # 取平均作为整体对齐分数
            alignment_score = similarity.mean().item()

        return alignment_score

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'fusion_request':
            return self._process_fusion_request(data)
        elif message_type == 'alignment_score_request':
            return self._process_alignment_score_request(data)

        return {'status': 'unknown_message_type'}

    def _process_fusion_request(self, data):
        """处理融合请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        left_features = data.get('left_features', None)
        right_features = data.get('right_features', None)

        if left_features is None or right_features is None:
            return {'status': 'error', 'message': 'Missing required features'}

        # 确保是张量
        if not isinstance(left_features, torch.Tensor):
            left_features = torch.tensor(left_features, dtype=torch.float)
        if not isinstance(right_features, torch.Tensor):
            right_features = torch.tensor(right_features, dtype=torch.float)

        # 执行融合
        fused_output = self.forward(left_features, right_features)

        return {
            'status': 'success',
            'fused_output': fused_output.detach().numpy()
        }

    def _process_alignment_score_request(self, data):
        """处理对齐分数请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        left_features = data.get('left_features', None)
        right_features = data.get('right_features', None)

        if left_features is None or right_features is None:
            return {'status': 'error', 'message': 'Missing required features'}

        # 确保是张量
        if not isinstance(left_features, torch.Tensor):
            left_features = torch.tensor(left_features, dtype=torch.float)
        if not isinstance(right_features, torch.Tensor):
            right_features = torch.tensor(right_features, dtype=torch.float)

        # 计算对齐分数
        alignment_score = self.semantic_alignment_score(left_features, right_features)

        return {
            'status': 'success',
            'alignment_score': alignment_score
        }


class MultiHeadAttention(nn.Module):
    """多头注意力实现"""

    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # 查询、键、值的线性变换
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出线性变换
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # 重塑并线性变换
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.out_linear(context)

        return output