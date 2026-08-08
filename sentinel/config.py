"""Configuration dataclasses for SentinelAI.

Configs are loaded from YAML files and validated into these structures.
Using dataclasses instead of raw dicts so that typos and missing fields
are caught early rather than surfacing as KeyErrors during training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ZoneConfig:
    """Configuration for a single zone in the environment."""
    zone_type: str = "commercial"  # commercial, residential, highway, parking
    max_cameras: int = 15
    base_anomaly_rate: float = 0.1


@dataclass
class WeatherConfig:
    """Markov chain parameters for weather transitions."""
    states: list[str] = field(default_factory=lambda: ["clear", "rain", "fog"])
    # transition_matrix[i][j] = P(state j | state i)
    transition_matrix: list[list[float]] = field(default_factory=lambda: [
        [0.90, 0.07, 0.03],  # clear -> clear/rain/fog
        [0.15, 0.75, 0.10],  # rain -> clear/rain/fog
        [0.20, 0.10, 0.70],  # fog -> clear/rain/fog
    ])
    initial_state: str = "clear"


@dataclass
class FailureConfig:
    """Parameters for infrastructure failure injection."""
    camera_dropout_prob: float = 0.005
    camera_dropout_duration: int = 20  # steps
    gpu_failure_prob: float = 0.002
    gpu_failure_duration: int = 50
    gpu_failure_capacity_loss: float = 0.3  # fraction of zone GPU lost
    network_congestion_prob: float = 0.003
    network_congestion_duration: int = 30
    network_congestion_bandwidth_loss: float = 0.4


@dataclass
class ResourceConfig:
    """Global and per-zone resource budgets."""
    total_gpu_budget: float = 100.0
    total_bandwidth_budget: float = 100.0


@dataclass
class DomainRandomizationConfig:
    """Ranges for domain randomization during training."""
    enabled: bool = False
    camera_count_range: list[int] = field(default_factory=lambda: [10, 60])
    gpu_budget_range: list[float] = field(default_factory=lambda: [60.0, 120.0])
    bandwidth_budget_range: list[float] = field(default_factory=lambda: [60.0, 120.0])
    anomaly_rate_multiplier_range: list[float] = field(default_factory=lambda: [0.5, 2.0])
    failure_prob_multiplier_range: list[float] = field(default_factory=lambda: [0.0, 3.0])


@dataclass
class EnvironmentConfig:
    """Full environment configuration."""
    type: str = "urban"
    num_zones: int = 4
    zones: list[ZoneConfig] = field(default_factory=lambda: [
        ZoneConfig(zone_type="commercial", max_cameras=15, base_anomaly_rate=0.12),
        ZoneConfig(zone_type="residential", max_cameras=15, base_anomaly_rate=0.06),
        ZoneConfig(zone_type="highway", max_cameras=15, base_anomaly_rate=0.10),
        ZoneConfig(zone_type="parking", max_cameras=15, base_anomaly_rate=0.04),
    ])
    max_cameras_per_zone: int = 15
    episode_length: int = 500
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    failures: FailureConfig = field(default_factory=FailureConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    spatial_correlation_decay: int = 5  # steps before neighbor boost fades
    spatial_correlation_boost: float = 0.3  # added anomaly prob for neighbors


@dataclass
class RewardWeights:
    """Weights for reward function terms. Used across all hierarchy levels."""
    detection_accuracy: float = 1.0
    incident_catch_rate: float = 2.0
    resource_cost: float = 0.5
    latency_penalty: float = 0.3
    # global-only
    resource_waste: float = 0.4
    starvation_penalty: float = 1.5
    # zone-only
    budget_overshoot: float = 1.0
    tier_misallocation: float = 0.8


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # filled in at runtime
    git_commit: str = ""
    wandb_run_id: str = ""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return the raw dict.

    Use this for loading configs before converting to typed dataclasses.
    Keeps the loading logic simple and separate from validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def env_config_from_dict(raw: dict[str, Any]) -> EnvironmentConfig:
    """Build an EnvironmentConfig from a raw YAML dict.

    Handles nested objects (zones, weather, failures, etc.) that
    yaml.safe_load returns as plain dicts.
    """
    config = EnvironmentConfig()
    config.type = raw.get("type", config.type)
    config.num_zones = raw.get("num_zones", config.num_zones)
    config.max_cameras_per_zone = raw.get("max_cameras_per_zone", config.max_cameras_per_zone)
    config.episode_length = raw.get("episode_length", config.episode_length)
    config.spatial_correlation_decay = raw.get(
        "spatial_correlation_decay", config.spatial_correlation_decay
    )
    config.spatial_correlation_boost = raw.get(
        "spatial_correlation_boost", config.spatial_correlation_boost
    )

    if "zones" in raw:
        config.zones = [ZoneConfig(**z) for z in raw["zones"]]

    if "weather" in raw:
        config.weather = WeatherConfig(**raw["weather"])

    if "failures" in raw:
        config.failures = FailureConfig(**raw["failures"])

    if "resources" in raw:
        config.resources = ResourceConfig(**raw["resources"])

    if "domain_randomization" in raw:
        config.domain_randomization = DomainRandomizationConfig(**raw["domain_randomization"])

    return config
