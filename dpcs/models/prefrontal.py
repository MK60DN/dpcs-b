import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Union


class PrefrontalCortexModule(nn.Module):
    """额叶执行控制模块"""

    def __init__(self, input_size, hidden_size=256, output_size=128, num_heads=4):
        super(PrefrontalCortexModule, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads

        # 信息整合层
        self.integration_layer = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 意识形成机制
        self.consciousness_former = nn.MultiheadAttention(hidden_size, num_heads)
        self.consciousness_norm = nn.LayerNorm(hidden_size)

        # 执行控制网络
        self.executive_control = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

        # 工作记忆
        self.working_memory = nn.Parameter(torch.zeros(hidden_size))
        self.wm_attention = nn.MultiheadAttention(output_size, num_heads)
        self.wm_gate = nn.Linear(output_size * 2, output_size)

        # 元认知评估
        self.metacognition = nn.Sequential(
            nn.Linear(output_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

        # 任务控制器
        self.task_controller = TaskController(hidden_size, output_size)

        # 决策信心估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, module_output: torch.Tensor, synchronized_output: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            module_output: 模块输出
            synchronized_output: 同步输出

        Returns:
            control: 控制信号
            meta_output: 元认知输出
        """
        # 整合信息
        integrated_features = self.integration_layer(
            torch.cat([module_output, synchronized_output], dim=-1)
        )

        # 形成意识表示
        consciousness_output = self.form_consciousness(integrated_features)

        # 执行控制
        control, updated_memory, meta_output = self.execute_control(consciousness_output)

        # 更新工作记忆
        self.working_memory.data = updated_memory.detach()

        return control, meta_output

    def form_consciousness(self, integrated_features: torch.Tensor) -> torch.Tensor:
        """
        形成意识表示

        Args:
            integrated_features: 整合后的特征

        Returns:
            consciousness_output: 意识表示
        """
        # 自注意力处理
        consciousness_input = integrated_features.unsqueeze(0)
        consciousness_output, _ = self.consciousness_former(
            consciousness_input,
            consciousness_input,
            consciousness_input
        )
        consciousness_output = consciousness_output.squeeze(0)

        # 残差连接
        consciousness_output = self.consciousness_norm(
            consciousness_output + integrated_features
        )

        return consciousness_output

    def execute_control(self, consciousness_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行控制生成

        Args:
            consciousness_output: 意识表示

        Returns:
            control: 控制信号
            updated_memory: 更新后的工作记忆
            meta_output: 元认知输出
        """
        # 生成控制信号
        control = self.executive_control(consciousness_output)

        # 工作记忆交互
        wmq = control.unsqueeze(0)
        wmk = self.working_memory.unsqueeze(0)
        wmv = self.working_memory.unsqueeze(0)
        wm_output, wm_attn = self.wm_attention(wmq, wmk, wmv)
        wm_output = wm_output.squeeze(0)

        # 更新工作记忆
        gates = torch.sigmoid(self.wm_gate(torch.cat([control, wm_output], dim=-1)))
        updated_memory = gates * wm_output + (1 - gates) * self.working_memory

        # 元认知评估
        meta_input = torch.cat([control, wm_attn.squeeze(0)], dim=-1)
        meta_output = self.metacognition(meta_input)

        return control, updated_memory, meta_output

    def evaluate_task(self, task_description: torch.Tensor) -> Dict[str, Any]:
        """
        评估任务并生成执行计划

        Args:
            task_description: 任务描述

        Returns:
            task_plan: 任务执行计划
        """
        return self.task_controller.plan_task(task_description)

    def estimate_confidence(self, control_signal: torch.Tensor) -> float:
        """
        估计决策信心

        Args:
            control_signal: 控制信号

        Returns:
            confidence: 信心评分
        """
        with torch.no_grad():
            confidence = self.confidence_estimator(control_signal)
        return confidence.item()

    def _implement_metacognition(self, control_signal, processing_results, confidence_threshold=0.7):
        """
        实现元认知评估

        Args:
            control_signal: 控制信号
            processing_results: 处理结果
            confidence_threshold: 信心阈值

        Returns:
            metacognition_output: 元认知输出
        """
        # 提取处理特征
        features = torch.cat([control_signal, processing_results], dim=-1)

        # 评估处理质量
        quality_assessment = self.metacognition(features)

        # 计算处理信心
        confidence = torch.sigmoid(self.confidence_estimator(quality_assessment))

        # 决定是否需要额外处理
        needs_reprocessing = confidence < confidence_threshold

        # 生成元认知输出
        metacognition_output = {
            'confidence': confidence.item(),
            'quality_assessment': quality_assessment.detach().numpy(),
            'needs_reprocessing': needs_reprocessing.item()
        }

        return metacognition_output

    def _handle_message(self, message):
        """处理模块间消息"""
        data = message.get('data', None)
        metadata = message.get('metadata', {})
        message_type = metadata.get('type', 'unknown')

        if message_type == 'control_request':
            return self._process_control_request(data)
        elif message_type == 'task_evaluation_request':
            return self._process_task_evaluation_request(data)
        elif message_type == 'metacognition_request':
            return self._process_metacognition_request(data)

        return {'status': 'unknown_message_type'}

    def _process_control_request(self, data):
        """处理控制请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        module_output = data.get('module_output', None)
        synchronized_output = data.get('synchronized_output', None)

        if module_output is None or synchronized_output is None:
            return {'status': 'error', 'message': 'Missing required outputs'}

        # 确保是张量
        if not isinstance(module_output, torch.Tensor):
            module_output = torch.tensor(module_output, dtype=torch.float)
        if not isinstance(synchronized_output, torch.Tensor):
            synchronized_output = torch.tensor(synchronized_output, dtype=torch.float)

        # 执行控制
        control, meta_output = self.forward(module_output, synchronized_output)

        return {
            'status': 'success',
            'control': control.detach().numpy(),
            'meta_output': meta_output.detach().numpy()
        }

    def _process_task_evaluation_request(self, data):
        """处理任务评估请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        task_description = data.get('task_description', None)

        if task_description is None:
            return {'status': 'error', 'message': 'Missing task description'}

        # 确保是张量
        if not isinstance(task_description, torch.Tensor):
            task_description = torch.tensor(task_description, dtype=torch.float)

        # 评估任务
        task_plan = self.evaluate_task(task_description)

        return {
            'status': 'success',
            'task_plan': task_plan
        }

    def _process_metacognition_request(self, data):
        """处理元认知请求"""
        if not isinstance(data, dict):
            return {'status': 'error', 'message': 'Invalid data format'}

        control_signal = data.get('control_signal', None)
        processing_results = data.get('processing_results', None)
        confidence_threshold = data.get('confidence_threshold', 0.7)

        if control_signal is None or processing_results is None:
            return {'status': 'error', 'message': 'Missing required signals'}

        # 确保是张量
        if not isinstance(control_signal, torch.Tensor):
            control_signal = torch.tensor(control_signal, dtype=torch.float)
        if not isinstance(processing_results, torch.Tensor):
            processing_results = torch.tensor(processing_results, dtype=torch.float)

        # 执行元认知评估
        metacognition_output = self._implement_metacognition(
            control_signal,
            processing_results,
            confidence_threshold
        )

        return {
            'status': 'success',
            'metacognition_output': metacognition_output
        }


class TaskController:
    """任务控制器"""

    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 任务规划网络
        self.planner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )

        # 任务分解网络
        self.decomposer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

        # 优先级估计网络
        self.priority_estimator = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.GELU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )

    def plan_task(self, task_description: torch.Tensor) -> Dict[str, Any]:
        """
        计划任务

        Args:
            task_description: 任务描述

        Returns:
            plan: 任务计划
        """
        with torch.no_grad():
            # 生成任务规划
            plan_features = self.planner(task_description)

            # 分解任务
            subtasks = self.decomposer(plan_features)

            # 估计优先级
            priority = self.priority_estimator(subtasks)

            # 构建任务计划
            plan = {
                'plan_features': plan_features.numpy(),
                'subtasks': subtasks.numpy(),
                'priority': priority.item(),
                'estimated_steps': int(priority.item() * 10) + 1
            }

        return plan