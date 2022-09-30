# Copyright (c) OpenMMLab. All rights reserved.
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .uniform_assigner import UniformAssigner

__all__ = [
    'BaseAssigner', 'AssignResult',
    'UniformAssigner'
]
