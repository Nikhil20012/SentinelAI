"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from sentinel.config import (
    EnvironmentConfig,
    WeatherConfig,
    env_config_from_dict,
    load_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "environments"


class TestDefaultConfig:
    """Verify that default EnvironmentConfig values are sane."""

    def test_default_has_four_zones(self):
        config = EnvironmentConfig()
        assert config.num_zones == 4
        assert len(config.zones) == 4

    def test_zone_types_are_distinct(self):
        config = EnvironmentConfig()
        types = [z.zone_type for z in config.zones]
        assert len(set(types)) == len(types)

    def test_weather_transition_matrix_is_stochastic(self):
        config = WeatherConfig()
        for row in config.transition_matrix:
            assert abs(sum(row) - 1.0) < 1e-6, f"Row does not sum to 1: {row}"

    def test_max_cameras_per_zone_is_positive(self):
        config = EnvironmentConfig()
        assert config.max_cameras_per_zone > 0
        for zone in config.zones:
            assert zone.max_cameras > 0


class TestYamlLoading:
    """Verify that YAML configs load and convert correctly."""

    def test_load_urban_normal(self):
        raw = load_config(CONFIGS_DIR / "urban_normal.yaml")
        assert raw["type"] == "urban"
        assert raw["num_zones"] == 4
        assert len(raw["zones"]) == 4

    def test_convert_urban_normal_to_dataclass(self):
        raw = load_config(CONFIGS_DIR / "urban_normal.yaml")
        config = env_config_from_dict(raw)
        assert isinstance(config, EnvironmentConfig)
        assert config.num_zones == 4
        assert config.weather.initial_state == "clear"
        # normal scenario has no failures
        assert config.failures.camera_dropout_prob == 0.0

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_zone_configs_preserved(self):
        raw = load_config(CONFIGS_DIR / "urban_normal.yaml")
        config = env_config_from_dict(raw)
        zone_types = [z.zone_type for z in config.zones]
        assert "commercial" in zone_types
        assert "highway" in zone_types

    def test_domain_randomization_disabled_in_normal(self):
        raw = load_config(CONFIGS_DIR / "urban_normal.yaml")
        config = env_config_from_dict(raw)
        assert config.domain_randomization.enabled is False
