import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union


class SRMT(nn.Module):
    """结构化强化学习模块 (左脑)"""

    def __init__(self, input_size, hidden_size=256, output_size=128):
        super(SRMT, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 实例变量初始化
        self.training_steps = 0
        self.is_training = False
        self.optimizer = None
        self.loss_history = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            action: 策略网络输出
            value: 价值网络输出
        """
        features = self.feature_extractor(x)
        action = self.policy_net(features)
        value = self.value_net(features)
        return action, value

    def learn(self, states, actions, rewards, next_states, dones, gamma=0.99, epsilon=0.2):
        """
        使用PPO算法学习

        Args:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次
            dones: 终止标志批次
            gamma: 折扣因子
            epsilon: PPO裁剪参数

        Returns:
            dict: 包含损失信息的字典
        """
        self.is_training = True

        # 确保优化器已初始化
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        # 转换输入为张量
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(next_states)
        if not isinstance(dones, torch.Tensor):
            dones = torch.FloatTensor(dones)

        # 计算旧策略的动作概率
        with torch.no_grad():
            old_actions, old_values = self(states)
            old_log_probs = self._compute_log_probs(old_actions, actions)

            # 计算状态价值估计
            _, next_values = self(next_states)
            next_values = next_values.squeeze(-1)

            # 计算GAE优势
            advantages = self._compute_gae(rewards, old_values.squeeze(-1), next_values, dones)

            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算回报
            returns = advantages + old_values.squeeze(-1)

        # PPO更新
        for _ in range(10):  # 多次迭代更新
            # 计算新策略的动作概率
            new_actions, new_values = self(states)
            new_log_probs = self._compute_log_probs(new_actions, actions)

            # 计算概率比率
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # 计算裁剪后的目标函数
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(new_values.squeeze(-1), returns.detach())

            # 熵损失（促进探索）
            entropy_loss = -self._compute_entropy(new_actions).mean()

            # 总损失
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            # 梯度更新
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

        # 记录损失
        loss_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        self.loss_history.append(loss_info)

        self.training_steps += 1
        self.is_training = False

        return loss_info

    def _compute_log_probs(self, actions, taken_actions):
        """计算动作的对数概率"""
        # 简化实现，实际应根据动作空间类型设计
        action_mean = actions
        action_std = torch.ones_like(action_mean) * 0.1

        # 假设动作服从正态分布
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(taken_actions).sum(dim=-1)

        return log_probs

    def _compute_entropy(self, actions):
        """计算策略熵"""
        # 简化实现，实际应根据动作空间类型设计
        action_mean = actions
        action_std = torch.ones_like(action_mean) * 0.1

        # 假设动作服从正态分布
        dist = torch.distributions.Normal(action_mean, action_std)
        entropy = dist.entropy().sum(dim=-1)

        return entropy

    def _compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """计算广义优势估计 (GAE)"""
        gae = 0
        advantages = torch.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'parameter_update':
            self._update_parameters(data)
        elif message_type == 'learning_request':
            self._process_learning_request(data)
        elif message_type == 'evaluation_request':
            self._process_evaluation_request(data)

        return {'status': 'processed', 'message_id': metadata.get('message_id')}

    def _update_parameters(self, data):
        """更新模型参数"""
        if not data or not isinstance(data, dict):
            return False

        for name, param_data in data.items():
            if hasattr(self, name) and isinstance(getattr(self, name), nn.Parameter):
                param = getattr(self, name)
                param.data = torch.tensor(param_data)

        return True

    def _process_learning_request(self, data):
        """处理学习请求"""
        if not data or not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        # 提取学习数据
        states = data.get('states', [])
        actions = data.get('actions', [])
        rewards = data.get('rewards', [])
        next_states = data.get('next_states', [])
        dones = data.get('dones', [])

        # 执行学习
        result = self.learn(states, actions, rewards, next_states, dones)

        return {'status': 'success', 'result': result}

    def _process_evaluation_request(self, data):
        """处理评估请求"""
        if not data or not isinstance(data, torch.Tensor):
            if isinstance(data, list) or isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            else:
                return {'status': 'error', 'message': 'Invalid data format'}

        # 执行评估
        with torch.no_grad():
            action, value = self(data)

        return {
            'status': 'success',
            'action': action.numpy(),
            'value': value.numpy()
        }