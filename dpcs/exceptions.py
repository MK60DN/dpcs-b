"""
异常类模块 - 定义DPCS-B框架的异常类型
"""


class DPCSError(Exception):
    """
    DPCS-B基础异常类，所有框架异常的基类
    """
    def __init__(self, message="DPCS-B框架出现错误"):
        self.message = message
        super().__init__(self.message)


class InputError(DPCSError):
    """
    输入错误异常，当输入数据无效时抛出
    """
    def __init__(self, message="输入数据无效", input_type=None, details=None):
        self.input_type = input_type
        self.details = details
        message_with_details = f"{message}"
        if input_type:
            message_with_details += f" (类型: {input_type})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class ProcessingError(DPCSError):
    """
    处理错误异常，当处理过程中出现错误时抛出
    """
    def __init__(self, message="处理过程出错", module=None, phase=None, details=None):
        self.module = module
        self.phase = phase
        self.details = details
        message_with_details = f"{message}"
        if module:
            message_with_details += f" (模块: {module})"
        if phase:
            message_with_details += f" (阶段: {phase})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class BlockchainError(DPCSError):
    """
    区块链错误异常，当区块链操作失败时抛出
    """
    def __init__(self, message="区块链操作失败", component=None, operation=None, details=None):
        self.component = component
        self.operation = operation
        self.details = details
        message_with_details = f"{message}"
        if component:
            message_with_details += f" (组件: {component})"
        if operation:
            message_with_details += f" (操作: {operation})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class ConfigurationError(DPCSError):
    """
    配置错误异常，当配置无效时抛出
    """
    def __init__(self, message="配置无效", parameter=None, expected=None, received=None):
        self.parameter = parameter
        self.expected = expected
        self.received = received
        message_with_details = f"{message}"
        if parameter:
            message_with_details += f" (参数: {parameter})"
        if expected and received:
            message_with_details += f" - 期望: {expected}, 收到: {received}"
        elif expected:
            message_with_details += f" - 期望: {expected}"
        elif received:
            message_with_details += f" - 收到: {received}"
        self.message = message_with_details
        super().__init__(self.message)


class ResourceError(DPCSError):
    """
    资源错误异常，当资源不足或不可用时抛出
    """
    def __init__(self, message="资源不足或不可用", resource_type=None, details=None):
        self.resource_type = resource_type
        self.details = details
        message_with_details = f"{message}"
        if resource_type:
            message_with_details += f" (资源类型: {resource_type})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class ModuleError(DPCSError):
    """
    模块错误异常，当特定模块发生错误时抛出
    """
    def __init__(self, message="模块错误", module_name=None, method=None, details=None):
        self.module_name = module_name
        self.method = method
        self.details = details
        message_with_details = f"{message}"
        if module_name:
            message_with_details += f" (模块: {module_name})"
        if method:
            message_with_details += f" (方法: {method})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class ValidationError(DPCSError):
    """
    验证错误异常，当验证失败时抛出
    """
    def __init__(self, message="验证失败", validation_type=None, details=None):
        self.validation_type = validation_type
        self.details = details
        message_with_details = f"{message}"
        if validation_type:
            message_with_details += f" (验证类型: {validation_type})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class IntegrationError(DPCSError):
    """
    集成错误异常，当与外部系统集成失败时抛出
    """
    def __init__(self, message="集成失败", system=None, operation=None, details=None):
        self.system = system
        self.operation = operation
        self.details = details
        message_with_details = f"{message}"
        if system:
            message_with_details += f" (系统: {system})"
        if operation:
            message_with_details += f" (操作: {operation})"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)


class TimeoutError(DPCSError):
    """
    超时错误异常，当操作超时时抛出
    """
    def __init__(self, message="操作超时", operation=None, timeout=None, details=None):
        self.operation = operation
        self.timeout = timeout
        self.details = details
        message_with_details = f"{message}"
        if operation:
            message_with_details += f" (操作: {operation})"
        if timeout:
            message_with_details += f" (超时: {timeout}秒)"
        if details:
            message_with_details += f" - {details}"
        self.message = message_with_details
        super().__init__(self.message)