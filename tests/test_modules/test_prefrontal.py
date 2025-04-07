import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpcs.modules.prefrontal import PrefrontalCortexModule, TaskController


class TestPrefrontalCortexModule(unittest.TestCase):
    """Prefrontal cortex executive control module test class"""

    def setUp(self):
        """Initialize test environment"""
        self.input_size = 64
        self.hidden_size = 128
        self.output_size = 64
        self.num_heads = 4
        self.batch_size = 2

        self.prefrontal = PrefrontalCortexModule(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.num_heads
        )

        # Create test tensors
        self.module_output = torch.randn(self.batch_size, self.input_size)
        self.synchronized_output = torch.randn(self.batch_size, self.input_size)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.prefrontal.input_size, self.input_size)
        self.assertEqual(self.prefrontal.hidden_size, self.hidden_size)
        self.assertEqual(self.prefrontal.output_size, self.output_size)
        self.assertEqual(self.prefrontal.num_heads, self.num_heads)

        self.assertIsInstance(self.prefrontal.integration_layer, torch.nn.Sequential)
        self.assertIsInstance(self.prefrontal.consciousness_former, torch.nn.MultiheadAttention)
        self.assertIsInstance(self.prefrontal.consciousness_norm, torch.nn.LayerNorm)
        self.assertIsInstance(self.prefrontal.executive_control, torch.nn.Sequential)
        self.assertIsInstance(self.prefrontal.working_memory, torch.nn.Parameter)
        self.assertIsInstance(self.prefrontal.wm_attention, torch.nn.MultiheadAttention)
        self.assertIsInstance(self.prefrontal.wm_gate, torch.nn.Linear)
        self.assertIsInstance(self.prefrontal.metacognition, torch.nn.Sequential)
        self.assertIsInstance(self.prefrontal.task_controller, TaskController)
        self.assertIsInstance(self.prefrontal.confidence_estimator, torch.nn.Sequential)

    def test_forward(self):
        """Test forward pass"""
        # Forward pass
        control, meta_output = self.prefrontal(
            self.module_output,
            self.synchronized_output
        )

        # Verify output dimensions
        self.assertEqual(control.shape, (self.batch_size, self.output_size))
        self.assertEqual(meta_output.shape, (self.batch_size, self.output_size))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(control).any())
        self.assertFalse(torch.isinf(control).any())
        self.assertFalse(torch.isnan(meta_output).any())
        self.assertFalse(torch.isinf(meta_output).any())

    def test_form_consciousness(self):
        """Test consciousness formation"""
        # Get integrated features
        integrated_features = self.prefrontal.integration_layer(
            torch.cat([self.module_output, self.synchronized_output], dim=-1)
        )

        # Form consciousness
        consciousness_output = self.prefrontal.form_consciousness(integrated_features)

        # Verify output dimensions
        self.assertEqual(consciousness_output.shape, integrated_features.shape)

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(consciousness_output).any())
        self.assertFalse(torch.isinf(consciousness_output).any())

    def test_execute_control(self):
        """Test execute control functionality"""
        # Get integrated features and form consciousness
        integrated_features = self.prefrontal.integration_layer(
            torch.cat([self.module_output, self.synchronized_output], dim=-1)
        )
        consciousness_output = self.prefrontal.form_consciousness(integrated_features)

        # Execute control
        control, updated_memory, meta_output = self.prefrontal.execute_control(consciousness_output)

        # Verify output dimensions
        self.assertEqual(control.shape, (self.batch_size, self.output_size))
        self.assertEqual(updated_memory.shape, self.prefrontal.working_memory.shape)
        self.assertEqual(meta_output.shape, (self.batch_size, self.output_size))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(control).any())
        self.assertFalse(torch.isinf(control).any())
        self.assertFalse(torch.isnan(updated_memory).any())
        self.assertFalse(torch.isinf(updated_memory).any())
        self.assertFalse(torch.isnan(meta_output).any())
        self.assertFalse(torch.isinf(meta_output).any())

    def test_update_working_memory(self):
        """Test working memory update"""
        # Create test tensors
        current_memory = torch.randn_like(self.prefrontal.working_memory)
        new_input = torch.randn(self.batch_size, self.output_size)

        # Update working memory
        updated_memory = self.prefrontal.update_working_memory(current_memory, new_input)

        # Verify output dimensions
        self.assertEqual(updated_memory.shape, current_memory.shape)

        # Verify memory has changed
        self.assertFalse(torch.allclose(updated_memory, current_memory))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(updated_memory).any())
        self.assertFalse(torch.isinf(updated_memory).any())

    def test_evaluate_confidence(self):
        """Test confidence evaluation"""
        # Create test tensor
        control_signal = torch.randn(self.batch_size, self.output_size)

        # Evaluate confidence
        confidence = self.prefrontal.evaluate_confidence(control_signal)

        # Verify output dimensions
        self.assertEqual(confidence.shape, (self.batch_size, 1))

        # Verify confidence range (should be between 0 and 1)
        self.assertTrue((confidence >= 0).all() and (confidence <= 1).all())

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(confidence).any())
        self.assertFalse(torch.isinf(confidence).any())

    def test_assess_metacognition(self):
        """Test metacognitive assessment"""
        # Create test tensors
        integrated_features = self.prefrontal.integration_layer(
            torch.cat([self.module_output, self.synchronized_output], dim=-1)
        )
        consciousness_output = self.prefrontal.form_consciousness(integrated_features)
        control, _, _ = self.prefrontal.execute_control(consciousness_output)

        # Assess metacognition
        meta_output = self.prefrontal.assess_metacognition(control, consciousness_output)

        # Verify output dimensions
        self.assertEqual(meta_output.shape, (self.batch_size, self.output_size))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(meta_output).any())
        self.assertFalse(torch.isinf(meta_output).any())

    def test_generate_control_signal(self):
        """Test control signal generation"""
        # Create test tensor
        consciousness_output = torch.randn(self.batch_size, self.hidden_size)

        # Generate control signal
        control_signal = self.prefrontal.generate_control_signal(consciousness_output)

        # Verify output dimensions
        self.assertEqual(control_signal.shape, (self.batch_size, self.output_size))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(control_signal).any())
        self.assertFalse(torch.isinf(control_signal).any())

    def test_handle_message_control_request(self):
        """Test handling control request message"""
        # Mock forward method
        mock_control = torch.randn(self.batch_size, self.output_size)
        mock_meta = torch.randn(self.batch_size, self.output_size)
        with patch.object(self.prefrontal, 'forward', return_value=(mock_control, mock_meta)):
            # Create message
            message = {
                'data': {
                    'module_output': self.module_output.numpy().tolist(),
                    'synchronized_output': self.synchronized_output.numpy().tolist()
                },
                'metadata': {'type': 'control_request'}
            }

            # Handle message
            response = self.prefrontal._handle_message(message)

            # Verify response
            self.assertEqual(response['status'], 'success')
            self.assertIn('control_signal', response)
            self.assertIn('metacognition', response)
            self.assertEqual(np.array(response['control_signal']).shape,
                             mock_control.detach().numpy().shape)
            self.assertEqual(np.array(response['metacognition']).shape,
                             mock_meta.detach().numpy().shape)

    def test_handle_message_metacognition_request(self):
        """Test handling metacognition request message"""
        # Mock assess_metacognition method
        mock_meta = torch.randn(self.batch_size, self.output_size)
        with patch.object(self.prefrontal, 'assess_metacognition', return_value=mock_meta):
            # Create message
            message = {
                'data': {
                    'control': torch.randn(self.batch_size, self.output_size).numpy().tolist(),
                    'consciousness': torch.randn(self.batch_size, self.hidden_size).numpy().tolist()
                },
                'metadata': {'type': 'metacognition_request'}
            }

            # Handle message
            response = self.prefrontal._handle_message(message)

            # Verify response
            self.assertEqual(response['status'], 'success')
            self.assertIn('metacognition', response)
            self.assertEqual(np.array(response['metacognition']).shape,
                             mock_meta.detach().numpy().shape)

    def test_handle_message_confidence_request(self):
        """Test handling confidence request message"""
        # Mock evaluate_confidence method
        mock_confidence = torch.tensor([[0.85], [0.92]])
        with patch.object(self.prefrontal, 'evaluate_confidence', return_value=mock_confidence):
            # Create message
            message = {
                'data': {
                    'control_signal': torch.randn(self.batch_size, self.output_size).numpy().tolist()
                },
                'metadata': {'type': 'confidence_request'}
            }

            # Handle message
            response = self.prefrontal._handle_message(message)

            # Verify response
            self.assertEqual(response['status'], 'success')
            self.assertIn('confidence', response)
            self.assertEqual(len(response['confidence']), self.batch_size)

    def test_handle_message_unknown_type(self):
        """Test handling unknown message type"""
        # Create message
        message = {
            'metadata': {'type': 'unknown_type'}
        }

        # Handle message
        response = self.prefrontal._handle_message(message)

        # Verify response
        self.assertEqual(response['status'], 'unknown_message_type')

    def test_handle_message_invalid_data(self):
        """Test handling message with invalid data"""
        # Create message - missing required data
        message = {
            'data': {
                'module_output': self.module_output.numpy().tolist()
                # Missing synchronized_output
            },
            'metadata': {'type': 'control_request'}
        }

        # Handle message
        response = self.prefrontal._handle_message(message)

        # Verify response
        self.assertEqual(response['status'], 'error')
        self.assertIn('message', response)
        self.assertIn('Missing', response['message'])


class TestTaskController(unittest.TestCase):
    """Task Controller test class"""

    def setUp(self):
        """Initialize test environment"""
        self.hidden_size = 128
        self.output_size = 64
        self.batch_size = 2
        self.task_controller = TaskController(self.hidden_size, self.output_size)

        # Create test tensor
        self.input_tensor = torch.randn(self.batch_size, self.hidden_size)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.task_controller.hidden_size, self.hidden_size)
        self.assertEqual(self.task_controller.output_size, self.output_size)
        self.assertIsInstance(self.task_controller.task_network, torch.nn.Sequential)
        self.assertIsInstance(self.task_controller.priority_network, torch.nn.Sequential)

    def test_forward(self):
        """Test forward pass"""
        # Forward pass
        task_vector, priorities = self.task_controller(self.input_tensor)

        # Verify output dimensions
        self.assertEqual(task_vector.shape, (self.batch_size, self.output_size))
        self.assertEqual(priorities.shape, (self.batch_size, 1))

        # Verify priority range (should be between 0 and 1)
        self.assertTrue((priorities >= 0).all() and (priorities <= 1).all())

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(task_vector).any())
        self.assertFalse(torch.isinf(task_vector).any())
        self.assertFalse(torch.isnan(priorities).any())
        self.assertFalse(torch.isinf(priorities).any())

    def test_get_task_representation(self):
        """Test task representation extraction"""
        # Get task representation
        task_vector = self.task_controller.get_task_representation(self.input_tensor)

        # Verify output dimensions
        self.assertEqual(task_vector.shape, (self.batch_size, self.output_size))

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(task_vector).any())
        self.assertFalse(torch.isinf(task_vector).any())

    def test_get_task_priority(self):
        """Test task priority extraction"""
        # Get task priority
        priorities = self.task_controller.get_task_priority(self.input_tensor)

        # Verify output dimensions
        self.assertEqual(priorities.shape, (self.batch_size, 1))

        # Verify priority range (should be between 0 and 1)
        self.assertTrue((priorities >= 0).all() and (priorities <= 1).all())

        # Verify no NaN or inf values
        self.assertFalse(torch.isnan(priorities).any())
        self.assertFalse(torch.isinf(priorities).any())


if __name__ == '__main__':
    unittest.main()