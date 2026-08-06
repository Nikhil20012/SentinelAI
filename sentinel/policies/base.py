"""Base interface for allocation policies.

Both RL agents and heuristic baselines implement this interface so that
the training and evaluation scripts can treat them interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base for all allocation policies."""

    @abstractmethod
    def select_actions(
        self,
        observations: dict[str, Any],
        action_masks: dict[str, Any],
    ) -> dict[str, Any]:
        """Choose actions for all agents given current observations and masks.

        Args:
            observations: {level: {agent_id: obs}}
            action_masks: {level: {agent_id: mask}}

        Returns:
            actions: {level: {agent_id: action}}
        """

    @abstractmethod
    def train(self, config: dict) -> None:
        """Run the training loop. No-op for heuristic baselines."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist policy state to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore policy state from disk."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this policy (used in configs and logging)."""
