"""
SFT core module - moved from /sft/core/ to /core/sft/

This module contains all SFT-related core functionality including:
- Configuration classes
- Trainer base classes and factories
- Evaluation and logging
- Model management
"""

from .config import (
    BackendType,
    DatasetConfig,
    LoggingConfig,
    ModelConfig,
    PrecisionType,
    SFTConfig,
    TaskType,
    TrainingConfig,
)
from .config_loader import SFTConfigLoader
from .evaluator import SFTEvaluator
from .logging import SFTLogger
from .trainer_base import SFTTrainerBase

__all__ = [
    # Config classes
    "SFTConfig",
    "TaskType",
    "PrecisionType",
    "BackendType",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "LoggingConfig",
    # Core classes
    "SFTTrainerBase",
    "SFTEvaluator",
    "SFTLogger",
    "SFTConfigLoader",
]
