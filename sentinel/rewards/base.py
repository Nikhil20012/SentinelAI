"""Base interface for reward computation.

Reward functions are separate from environments so that the same environment
can be used with different reward formulations during ablation studies.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):
    """Abstract base for reward functions."""

    @abstractmethod
    def compute(
        self,
        observations: dict[str, Any],
        actions: dict[str, Any],
        next_observations: dict[str, Any],
        info: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Compute rewards for all agents given a transition.

        Args:
            observations: pre-step observations {level: {agent_id: obs}}
            actions: actions taken {level: {agent_id: action}}
            next_observations: post-step observations
            info: environment info dict (contains resource usage, incidents, etc.)

        Returns:
            rewards: {level: {agent_id: float}}
        """

    @abstractmethod
    def reward_components(
        self,
        observations: dict[str, Any],
        actions: dict[str, Any],
        next_observations: dict[str, Any],
        info: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Return individual reward terms before weighting.

        Same signature as compute(), but returns per-term breakdowns for
        logging and ablation. Keys are term names (e.g. "detection_accuracy",
        "resource_cost").
        """
