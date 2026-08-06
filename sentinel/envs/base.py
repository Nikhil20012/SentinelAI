"""Base interface for SentinelAI simulation environments.

Every environment provides a PettingZoo-compatible ParallelEnv with three
hierarchy levels (global, zone, camera). Observations and actions are returned
as nested dicts keyed by agent id.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Abstract base for all SentinelAI environments.

    Subclasses must implement the core step/reset loop and expose the
    observation and action spaces that the three hierarchy levels need.
    """

    @abstractmethod
    def observation_spaces(self) -> dict:
        """Return observation spaces keyed by hierarchy level.

        Expected keys: "global", "zone", "camera".
        Each value is a gymnasium.spaces.Space (or dict of spaces for
        multi-agent levels).
        """

    @abstractmethod
    def action_spaces(self) -> dict:
        """Return action spaces keyed by hierarchy level.

        Expected keys: "global", "zone", "camera".
        """

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment and return initial observations.

        Returns a nested dict: {level: {agent_id: observation}}.
        """

    @abstractmethod
    def step(self, actions: dict[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one timestep with actions from all hierarchy levels.

        Args:
            actions: nested dict {level: {agent_id: action}}

        Returns:
            observations: {level: {agent_id: obs}}
            rewards: {level: {agent_id: float}}
            terminateds: {level: {agent_id: bool}}
            truncateds: {level: {agent_id: bool}}
            infos: {level: {agent_id: dict}}
        """

    @abstractmethod
    def action_masks(self) -> dict[str, Any]:
        """Return current valid action masks per agent.

        Used by the RL training loop to mask infeasible actions before
        the policy samples.
        """

    @abstractmethod
    def render(self) -> dict[str, Any] | None:
        """Return a serializable snapshot of the current state for logging.

        Used by the episode recorder. Returns None if rendering is disabled.
        """

    @property
    @abstractmethod
    def num_zones(self) -> int:
        """Number of zones in this environment instance."""

    @property
    @abstractmethod
    def max_cameras_per_zone(self) -> int:
        """Maximum cameras per zone (used for padding inactive agents)."""

    @property
    @abstractmethod
    def num_cameras(self) -> int:
        """Total number of active cameras across all zones."""
