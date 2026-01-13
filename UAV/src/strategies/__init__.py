"""Strategy package initializer.

Exports the registry helpers and ensures built-in strategies are registered.
"""
from src.strategies.transfer_base import (
    STRATEGY_REGISTRY,
    BaseTransferStrategy,
    create_strategy,
    register_strategy,
)

# Import built-in strategies to populate registry
from src.strategies import finetune, feature_extract  # noqa: F401

__all__ = [
    "STRATEGY_REGISTRY",
    "BaseTransferStrategy",
    "create_strategy",
    "register_strategy",
]
