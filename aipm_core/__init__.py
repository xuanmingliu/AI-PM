"""
AI-Product Manager Core Framework
包含五个核心模块：感知、决策、执行、学习、交互
"""

from .perception_module import PerceptionModule
from .decision_module import DecisionModule
from .execution_module import ExecutionModule
from .learning_module import LearningModule
from .interaction_module import InteractionModule
from .framework import AIPMFramework

__all__ = [
    'PerceptionModule',
    'DecisionModule', 
    'ExecutionModule',
    'LearningModule',
    'InteractionModule',
    'AIPMFramework'
]