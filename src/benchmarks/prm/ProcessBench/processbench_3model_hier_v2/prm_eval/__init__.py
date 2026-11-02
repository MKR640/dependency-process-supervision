from .base_prm import BasePRM
from .qwen3_math_prm import Qwen3MathPRM


PRM_REGISTRY = {
    "qwen3-math-prm-4b": Qwen3MathPRM,
}


__all__ = [
    "BasePRM",
    "Qwen3MathPRM",
    "PRM_REGISTRY",
]
