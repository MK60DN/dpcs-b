import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Union


class CAMELAgent(nn.Module):
    """语义理解与语言生成模块 (右脑)"""

    def __init__(self, model_name="gpt2", embedding_dim=768, hidden_size=512, output_size=128):
        super(CAMELAgent, self).__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

        self.task_planner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        # 记忆机制
        self.memory = []
        self.max_memory_size = 10

        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (embedding_dim)

        Returns:
            output: 输出向量 (output_size)
            task_plan: 任务规划向量 (hidden_size//2)
        """
        encoded = self.encoder(x)

        # 如果有记忆，应用注意力机制
        if self.memory:
            memory_tensor = torch.stack(self.memory) if len(self.memory) > 1 else self.memory[0].unsqueeze(0)

            # 调整张量维度以适配多头注意力
            encoded_q = encoded.unsqueeze(0) if encoded.dim() == 2 else encoded
            memory_kv = memory_tensor.unsqueeze(1) if memory_tensor.dim() == 2 else memory_tensor

            # 应用注意力机制
            attention_output, _ = self.attention(encoded_q, memory_kv, memory_kv)

            # 合并当前表示和注意力输出
            encoded = encoded + attention_output.squeeze(0)

        # 任务规划
        task_plan = self.task_planner(encoded)

        # 输出生成
        output = self.decoder(encoded)

        # 更新记忆
        self._update_memory(encoded.detach())

        return output, task_plan

    def generate_text(self, embedding, max_length=50, temperature=0.7):
        """
        基于输入生成文本

        Args:
            embedding: 输入嵌入
            max_length: 最大生成长度
            temperature: 生成温度

        Returns:
            str: 生成的文本
        """
        # 这里是简化版本的生成逻辑
        # 真实实现应该集成大型语言模型或Transformer解码器

        # 获取初始表示
        with torch.no_grad():
            encoded = self.encoder(embedding)

        # 用于存储生成的token ID
        generated_ids = []

        # 生成循环
        for _ in range(max_length):
            with torch.no_grad():
                # 获取输出分布
                output, _ = self.forward(embedding)

                # 应用温度
                logits = output / temperature

                # 采样下一个token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.squeeze(0), 1)

                generated_ids.append(next_token.item())

                # 如果生成了结束标记，提前结束
                if next_token.item() == 50256:  # EOS token for GPT
                    break

                # 准备下一次迭代的输入
                embedding = self._update_input_with_token(embedding, next_token)

        # 将token ID转换回文本(伪实现)
        generated_text = self._ids_to_text(generated_ids)

        return generated_text

    def _update_memory(self, encoded_state):
        """更新记忆机制"""
        # 将编码状态添加到记忆中
        if isinstance(encoded_state, torch.Tensor):
            if encoded_state.dim() > 1 and encoded_state.size(0) > 1:
                # 如果是批处理，取平均
                encoded_state = encoded_state.mean(dim=0, keepdim=True)
            self.memory.append(encoded_state.squeeze(0))

        # 限制记忆大小
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def _update_input_with_token(self, previous_embedding, token_id):
        """更新输入嵌入以包括新token"""
        # 这是一个简化实现，实际应使用正确的词嵌入
        # 在真实系统中应该通过语言模型的token嵌入层获取表示
        token_embedding = torch.randn(self.embedding_dim)

        # 更新输入
        updated_embedding = 0.9 * previous_embedding + 0.1 * token_embedding

        return updated_embedding

    def _ids_to_text(self, token_ids):
        """将token ID转换为文本(伪实现)"""
        # 这是一个伪实现，真实系统应集成tokenizer
        return f"Generated text with {len(token_ids)} tokens"

    def parse_and_plan(self, text_embedding):
        """
        解析输入并生成任务计划

        Args:
            text_embedding: 文本嵌入

        Returns:
            dict: 包含解析结果和计划的字典
        """
        with torch.no_grad():
            # 编码输入
            encoded = self.encoder(text_embedding)

            # 生成任务计划
            task_plan = self.task_planner(encoded)

            # 将计划转换为结构化表示
            structured_plan = self._plan_to_structure(task_plan)

        return {
            'encoded_representation': encoded.numpy(),
            'task_plan': structured_plan
        }

    def _plan_to_structure(self, plan_tensor):
        """将计划张量转换为结构化表示(伪实现)"""
        # 这是一个伪实现，实际应根据任务空间设计结构化表示方法
        plan_vector = plan_tensor.squeeze().numpy()

        # 提取计划要素
        plan = {
            'goals': ['理解输入', '提取关键信息', '生成回复'],
            'complexity': float(plan_vector.mean()),
            'estimated_steps': max(1, int(plan_vector.sum() % 10)),
            'focus_areas': ['语义理解', '上下文联系', '回复生成']
        }

        return plan

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'generate_request':
            return self._process_generate_request(data)
        elif message_type == 'planning_request':
            return self._process_planning_request(data)
        elif message_type == 'memory_update':
            return self._process_memory_update(data)

        return {'status': 'unknown_message_type'}

    def _process_generate_request(self, data):
        """处理生成请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        embedding = data.get('embedding', None)
        max_length = data.get('max_length', 50)
        temperature = data.get('temperature', 0.7)

        if embedding is None:
            return {'status': 'error', 'message': 'Missing embedding'}

        # 确保embedding是tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)

        # 生成文本
        generated_text = self.generate_text(embedding, max_length, temperature)

        return {'status': 'success', 'generated_text': generated_text}

    def _process_planning_request(self, data):
        """处理规划请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        embedding = data.get('embedding', None)

        if embedding is None:
            return {'status': 'error', 'message': 'Missing embedding'}

        # 确保embedding是tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)

        # 解析并规划
        plan_result = self.parse_and_plan(embedding)

        return {'status': 'success', 'plan': plan_result}

    def _process_memory_update(self, data):
        """处理记忆更新请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        # 清空记忆
        if data.get('clear_memory', False):
            self.memory = []
            return {'status': 'success', 'message': 'Memory cleared'}

        # 更新记忆容量
        if 'memory_size' in data:
            self.max_memory_size = max(1, int(data['memory_size']))

        # 添加新记忆
        new_memory = data.get('new_memory', None)
        if new_memory is not None:
            if not isinstance(new_memory, torch.Tensor):
                new_memory = torch.tensor(new_memory, dtype=torch.float)
            self._update_memory(new_memory)

        return {
            'status': 'success',
            'memory_size': len(self.memory),
            'max_memory_size': self.max_memory_size
        }