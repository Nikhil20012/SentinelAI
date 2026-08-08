"""Tests for the urban surveillance environment."""

import numpy as np

from sentinel.config import (
    DomainRandomizationConfig,
    EnvironmentConfig,
    env_config_from_dict,
    load_config,
)
from sentinel.envs.constants import (
    NUM_BUDGET_LEVELS,
    NUM_QUALITY_LEVELS,
    NUM_TIERS,
)
from sentinel.envs.urban import UrbanEnvironment


def make_env(config: EnvironmentConfig | None = None) -> UrbanEnvironment:
    """Create an environment with defaults or a provided config."""
    if config is None:
        config = EnvironmentConfig()
    return UrbanEnvironment(config)


def sample_random_actions(env: UrbanEnvironment, rng: np.random.Generator) -> dict:
    """Sample valid random actions for all agents."""
    spaces = env.action_spaces()
    actions = {"global": {}, "zone": {}, "camera": {}}
    for level, agents in spaces.items():
        for agent_id, space in agents.items():
            actions[level][agent_id] = space.sample()
    return actions


class TestSpaces:
    """Verify observation and action space dimensions are consistent."""

    def test_global_obs_shape(self):
        env = make_env()
        spaces = env.observation_spaces()
        config = EnvironmentConfig()
        expected_dim = 3 * config.num_zones + 6
        assert spaces["global"]["global"].shape == (expected_dim,)

    def test_zone_obs_shape(self):
        env = make_env()
        spaces = env.observation_spaces()
        config = EnvironmentConfig()
        expected_dim = 6 * config.max_cameras_per_zone + 4
        for z in range(config.num_zones):
            assert spaces["zone"][f"zone_{z}"].shape == (expected_dim,)

    def test_camera_obs_shape(self):
        env = make_env()
        spaces = env.observation_spaces()
        config = EnvironmentConfig()
        for z in range(config.num_zones):
            for c in range(config.max_cameras_per_zone):
                assert spaces["camera"][f"cam_{z}_{c}"].shape == (9,)

    def test_global_action_space(self):
        env = make_env()
        spaces = env.action_spaces()
        config = EnvironmentConfig()
        assert len(spaces["global"]["global"].nvec) == config.num_zones
        assert all(n == NUM_BUDGET_LEVELS for n in spaces["global"]["global"].nvec)

    def test_zone_action_space(self):
        env = make_env()
        spaces = env.action_spaces()
        config = EnvironmentConfig()
        for z in range(config.num_zones):
            nvec = spaces["zone"][f"zone_{z}"].nvec
            assert len(nvec) == config.max_cameras_per_zone
            assert all(n == NUM_TIERS for n in nvec)

    def test_camera_action_space(self):
        env = make_env()
        spaces = env.action_spaces()
        config = EnvironmentConfig()
        for z in range(config.num_zones):
            for c in range(config.max_cameras_per_zone):
                nvec = spaces["camera"][f"cam_{z}_{c}"].nvec
                assert list(nvec) == [NUM_QUALITY_LEVELS] * 3

    def test_agent_counts(self):
        env = make_env()
        spaces = env.observation_spaces()
        config = EnvironmentConfig()
        assert len(spaces["global"]) == 1
        assert len(spaces["zone"]) == config.num_zones
        total_cams = config.num_zones * config.max_cameras_per_zone
        assert len(spaces["camera"]) == total_cams


class TestReset:
    """Verify reset produces valid initial observations."""

    def test_returns_correct_structure(self):
        env = make_env()
        obs = env.reset(seed=42)
        assert "global" in obs
        assert "zone" in obs
        assert "camera" in obs

    def test_observations_match_spaces(self):
        env = make_env()
        obs = env.reset(seed=42)
        spaces = env.observation_spaces()
        for level in ["global", "zone", "camera"]:
            for agent_id, space in spaces[level].items():
                assert agent_id in obs[level], f"Missing obs for {agent_id}"
                assert obs[level][agent_id].shape == space.shape
                assert obs[level][agent_id].dtype == np.float32

    def test_deterministic_with_seed(self):
        env = make_env()
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        for level in obs1:
            for agent_id in obs1[level]:
                np.testing.assert_array_equal(
                    obs1[level][agent_id], obs2[level][agent_id]
                )

    def test_different_seeds_differ(self):
        env = make_env()
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=99)
        # Anomaly scores should differ (global obs includes zone anomaly means)
        assert not np.array_equal(
            obs1["global"]["global"], obs2["global"]["global"]
        )

    def test_active_camera_count(self):
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        expected = sum(z.max_cameras for z in config.zones)
        assert env.num_cameras == expected


class TestStep:
    """Verify step mechanics work correctly."""

    def test_step_returns_correct_structure(self):
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        actions = sample_random_actions(env, rng)
        obs, rewards, terms, truncs, infos = env.step(actions)

        assert "global" in obs
        assert "zone" in obs
        assert "camera" in obs

    def test_step_obs_match_spaces(self):
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        actions = sample_random_actions(env, rng)
        obs, _, _, _, _ = env.step(actions)

        spaces = env.observation_spaces()
        for level in ["global", "zone", "camera"]:
            for agent_id, space in spaces[level].items():
                assert obs[level][agent_id].shape == space.shape

    def test_rewards_are_float(self):
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        actions = sample_random_actions(env, rng)
        _, rewards, _, _, _ = env.step(actions)

        for level in rewards:
            for agent_id, r in rewards[level].items():
                assert isinstance(r, (int, float))

    def test_episode_terminates(self):
        config = EnvironmentConfig()
        config.episode_length = 10
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        for step_i in range(10):
            actions = sample_random_actions(env, rng)
            _, _, _, truncateds, _ = env.step(actions)

        # After episode_length steps, should be truncated
        assert truncateds["global"]["global"] is True

    def test_episode_not_done_early(self):
        config = EnvironmentConfig()
        config.episode_length = 100
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        actions = sample_random_actions(env, rng)
        _, _, _, truncateds, _ = env.step(actions)
        assert truncateds["global"]["global"] is False


class TestBudgetAllocation:
    """Verify global controller's budget allocation works."""

    def test_even_allocation(self):
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        n = config.num_zones

        # All zones get "moderate" (index 1, fraction 0.25)
        action = np.ones(n, dtype=np.int64)
        zone_tiers = {
            f"zone_{z}": np.full(config.max_cameras_per_zone, 3, dtype=np.int64)
            for z in range(n)
        }
        actions = {
            "global": {"global": action},
            "zone": zone_tiers,
            "camera": {
                f"cam_{z}_{c}": np.zeros(3, dtype=np.int64)
                for z in range(n)
                for c in range(config.max_cameras_per_zone)
            },
        }
        env.step(actions)

        # Total fraction = 4 * 0.25 = 1.0, no normalization needed
        expected_gpu = 0.25 * config.resources.total_gpu_budget
        np.testing.assert_allclose(env._zone_gpu_budget, expected_gpu, rtol=1e-5)

    def test_over_allocation_normalizes(self):
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        n = config.num_zones

        # All zones get "critical" (index 3, fraction 0.50)
        # Total = 4 * 0.50 = 2.0, should normalize to 0.25 each
        action = np.full(n, 3, dtype=np.int64)
        zone_tiers = {
            f"zone_{z}": np.full(config.max_cameras_per_zone, 3, dtype=np.int64)
            for z in range(n)
        }
        actions = {
            "global": {"global": action},
            "zone": zone_tiers,
            "camera": {
                f"cam_{z}_{c}": np.zeros(3, dtype=np.int64)
                for z in range(n)
                for c in range(config.max_cameras_per_zone)
            },
        }
        env.step(actions)

        total_allocated = env._zone_gpu_budget.sum()
        np.testing.assert_allclose(
            total_allocated, config.resources.total_gpu_budget, rtol=1e-5
        )


class TestInactiveCameras:
    """Verify inactive cameras are handled correctly."""

    def test_inactive_cameras_have_zero_anomaly(self):
        config = EnvironmentConfig()
        # Set first zone to only 5 cameras (max is 15)
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        env.reset(seed=42)

        # Cameras 5-14 in zone 0 should have zero anomaly
        assert (env._anomaly[0, 5:] == 0.0).all()

    def test_inactive_cameras_have_zero_obs(self):
        config = EnvironmentConfig()
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        obs = env.reset(seed=42)

        # Inactive camera should have all-zero observation
        inactive_obs = obs["camera"]["cam_0_10"]
        np.testing.assert_array_equal(inactive_obs, np.zeros(9, dtype=np.float32))

    def test_active_cameras_have_nonzero_obs(self):
        env = make_env()
        obs = env.reset(seed=42)

        # At least anomaly_score should be nonzero for most active cameras
        active_obs = obs["camera"]["cam_0_0"]
        assert active_obs[0] > 0.0  # anomaly_score


class TestResourceCosts:
    """Verify resource cost computation."""

    def test_min_quality_costs(self):
        """Minimum quality should produce minimum costs."""
        env = make_env()
        env.reset(seed=42)

        # After reset, all cameras are at (0,0,0)
        # GPU cost = 6.0 * 0.3 * 0.2 * 0.3 = 0.108
        expected = 6.0 * 0.3 * 0.2 * 0.3
        for z in range(env.num_zones):
            for c in range(env.max_cameras_per_zone):
                if env._active[z, c]:
                    np.testing.assert_allclose(
                        env._gpu_costs[z, c], expected, rtol=1e-5
                    )

    def test_inactive_cameras_zero_cost(self):
        config = EnvironmentConfig()
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        assert (env._gpu_costs[0, 5:] == 0.0).all()
        assert (env._bw_costs[0, 5:] == 0.0).all()


class TestFullEpisode:
    """Run a complete episode to catch any runtime errors."""

    def test_full_episode_random_actions(self):
        config = EnvironmentConfig()
        config.episode_length = 50
        env = UrbanEnvironment(config)
        obs = env.reset(seed=42)
        rng = np.random.default_rng(42)

        for _ in range(50):
            actions = sample_random_actions(env, rng)
            obs, rewards, terms, truncs, infos = env.step(actions)

        # Should have completed the episode
        assert truncs["global"]["global"] is True

    def test_render_returns_state(self):
        env = make_env()
        env.reset(seed=42)
        state = env.render()
        assert state is not None
        assert "timestep" in state
        assert "anomaly" in state
        assert state["anomaly"].shape == (env.num_zones, env.max_cameras_per_zone)

    def test_action_masks_shape(self):
        env = make_env()
        env.reset(seed=42)
        masks = env.action_masks()
        spaces = env.action_spaces()

        for level in ["global", "zone", "camera"]:
            for agent_id, space in spaces[level].items():
                expected_len = int(space.nvec.sum())
                assert masks[level][agent_id].shape == (expected_len,)


class TestFromYamlConfig:
    """Verify environment works when loaded from YAML config."""

    def test_load_and_run(self):
        raw = load_config("configs/environments/urban_normal.yaml")
        config = env_config_from_dict(raw)
        env = UrbanEnvironment(config)
        obs = env.reset(seed=42)
        rng = np.random.default_rng(42)

        actions = sample_random_actions(env, rng)
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Basic sanity
        assert obs["global"]["global"].shape[0] > 0
        assert env.num_cameras > 0


class TestAnomalyGeneration:
    """Tests for Poisson-based anomaly generation with time and spatial effects."""

    def test_scores_in_valid_range(self):
        """All anomaly scores should be in [0, 1]."""
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        for _ in range(20):
            actions = sample_random_actions(env, rng)
            env.step(actions)
            assert env._anomaly.min() >= 0.0
            assert env._anomaly.max() <= 1.0

    def test_not_uniform_distribution(self):
        """Anomaly scores should cluster near 0 (calm) with occasional spikes,
        not be uniformly distributed."""
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        all_scores = []
        for _ in range(50):
            actions = sample_random_actions(env, rng)
            env.step(actions)
            active = env._active
            all_scores.extend(env._anomaly[active].tolist())

        scores = np.array(all_scores)
        # Most scores should be low (calm cameras dominate)
        low_fraction = (scores < 0.3).mean()
        assert low_fraction > 0.4, f"Expected most scores low, got {low_fraction:.2f} below 0.3"

    def test_time_of_day_modulation(self):
        """Commercial zones should produce higher anomaly rates at noon than midnight."""
        config = EnvironmentConfig()
        config.episode_length = 1000
        env = UrbanEnvironment(config)

        # Collect mean anomaly at "noon" (time_of_day=0.5)
        env.reset(seed=42)
        env._time_of_day = 0.5  # noon
        env._generate_anomalies()
        #noon_mean = env._anomaly[0, :env._cams_per_zone[0]].mean()

        # Collect mean anomaly at "midnight" (time_of_day=0.0)
        env.reset(seed=42)
        env._time_of_day = 0.0  # midnight
        env._generate_anomalies()
        #midnight_mean = env._anomaly[0, :env._cams_per_zone[0]].mean()

        # Run enough samples to get stable means
        noon_samples, midnight_samples = [], []
        for _ in range(200):
            env._time_of_day = 0.5
            env._generate_anomalies()
            noon_samples.append(env._anomaly[0, :env._cams_per_zone[0]].mean())
            env._time_of_day = 0.0
            env._generate_anomalies()
            midnight_samples.append(env._anomaly[0, :env._cams_per_zone[0]].mean())

        # Commercial zone (index 0) peaks during business hours
        assert np.mean(noon_samples) > np.mean(midnight_samples), (
            f"Noon mean {np.mean(noon_samples):.3f} should exceed "
            f"midnight mean {np.mean(midnight_samples):.3f} for commercial zone"
        )

    def test_spatial_correlation_boosts_neighbors(self):
        """After a high-anomaly event, adjacent cameras should get boosted rates."""
        config = EnvironmentConfig()
        env = UrbanEnvironment(config)
        env.reset(seed=42)

        # Force a high anomaly on camera 5 in zone 0
        env._anomaly[0, 5] = 0.95
        env._update_spatial_boost()

        # Neighbors (4 and 6) should have nonzero boost
        assert env._spatial_boost[0, 4] > 0.0, "Left neighbor should be boosted"
        assert env._spatial_boost[0, 6] > 0.0, "Right neighbor should be boosted"
        # Non-neighbors should have zero boost (still first step)
        assert env._spatial_boost[0, 2] == 0.0, "Distant camera should not be boosted"

    def test_spatial_boost_decays(self):
        """Spatial boost should decay over time if no new incidents occur."""
        config = EnvironmentConfig()
        env = UrbanEnvironment(config)
        env.reset(seed=42)

        # Inject a boost
        env._spatial_boost[0, 3] = 0.4

        # Clear all anomalies so no new boost is added
        env._anomaly[:] = 0.0
        env._update_spatial_boost()

        # Boost should have decayed
        assert env._spatial_boost[0, 3] < 0.4, "Boost should decay"
        assert env._spatial_boost[0, 3] > 0.0, "Boost should not vanish in one step"

    def test_hotspots_spawn_over_time(self):
        """Running many steps should eventually spawn at least one hotspot."""
        config = EnvironmentConfig()
        config.episode_length = 500
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        max_hotspots_seen = 0
        for _ in range(200):
            actions = sample_random_actions(env, rng)
            env.step(actions)
            max_hotspots_seen = max(max_hotspots_seen, len(env._hotspots))

        assert max_hotspots_seen > 0, "Should have spawned at least one hotspot in 200 steps"

    def test_hotspots_bounce_at_edges(self):
        """Hotspot center should stay within valid camera indices."""
        config = EnvironmentConfig()
        env = UrbanEnvironment(config)
        env.reset(seed=42)

        # Force a hotspot at the edge moving outward
        env._hotspots = [{
            "zone": 0,
            "center": 0,
            "intensity": 3.0,
            "direction": -1,
            "remaining": 10,
        }]
        env._update_hotspots()

        # Should have bounced: center stays at 0, direction flips to 1
        h = [x for x in env._hotspots if x["zone"] == 0]
        assert len(h) >= 1
        assert h[0]["center"] >= 0
        assert h[0]["direction"] == 1

    def test_inactive_cameras_still_zero(self):
        """Inactive cameras should always have zero anomaly even with new generation."""
        config = EnvironmentConfig()
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        for _ in range(20):
            actions = sample_random_actions(env, rng)
            env.step(actions)
            assert (env._anomaly[0, 5:] == 0.0).all()

    def test_zone_types_produce_different_patterns(self):
        """Highway and residential zones should have noticeably different
        anomaly rate profiles across a simulated day."""
        config = EnvironmentConfig()
        env = UrbanEnvironment(config)

        # Highway is zone index 2, residential is index 1 in default config
        highway_rates = []
        residential_rates = []
        for hour_frac in np.linspace(0, 1, 24, endpoint=False):
            env.reset(seed=42)
            env._time_of_day = hour_frac
            rates = env._compute_base_rates()
            n_hw = env._cams_per_zone[2]
            n_res = env._cams_per_zone[1]
            highway_rates.append(rates[2, :n_hw].mean())
            residential_rates.append(rates[1, :n_res].mean())

        # The profiles should differ (different peak hours)
        highway_rates = np.array(highway_rates)
        residential_rates = np.array(residential_rates)
        peak_highway = np.argmax(highway_rates)
        peak_residential = np.argmax(residential_rates)
        assert peak_highway != peak_residential, (
            "Highway and residential should peak at different hours"
        )


class TestActionMasking:
    """Tests for tier-based camera masking and budget-based zone masking."""

    def test_camera_mask_minimal_tier(self):
        """Minimal tier should only allow the lowest option per dimension."""
        env = make_env()
        env.reset(seed=42)
        # Force camera 0 in zone 0 to minimal tier
        env._tiers[0, 0] = 0
        mask = env._build_camera_mask(0, 0)
        # mask layout: [res0, res1, res2, fps0, fps1, fps2, mod0, mod1, mod2]
        expected = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask, expected)

    def test_camera_mask_normal_tier(self):
        """Normal tier allows indices 0-1 for all three dimensions."""
        env = make_env()
        env.reset(seed=42)
        env._tiers[0, 0] = 1
        mask = env._build_camera_mask(0, 0)
        expected = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask, expected)

    def test_camera_mask_elevated_tier(self):
        """Elevated: res up to 2, fps/model up to 1."""
        env = make_env()
        env.reset(seed=42)
        env._tiers[0, 0] = 2
        mask = env._build_camera_mask(0, 0)
        expected = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask, expected)

    def test_camera_mask_priority_tier(self):
        """Priority tier should allow everything."""
        env = make_env()
        env.reset(seed=42)
        env._tiers[0, 0] = 3
        mask = env._build_camera_mask(0, 0)
        expected = np.ones(9, dtype=np.int8)
        np.testing.assert_array_equal(mask, expected)

    def test_camera_mask_inactive(self):
        """Inactive cameras should only allow the lowest option per dimension."""
        config = EnvironmentConfig()
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        # Camera 10 in zone 0 is inactive
        mask = env._build_camera_mask(0, 10)
        expected = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask, expected)

    def test_zone_mask_generous_budget(self):
        """With a large budget, all tiers should be available for active cameras."""
        env = make_env()
        env.reset(seed=42)
        # Give zone 0 a very large budget
        env._zone_gpu_budget[0] = 1000.0
        env._zone_bw_budget[0] = 1000.0
        # All cameras at minimal currently
        env._tiers[0, :] = 0
        mask = env._build_zone_mask(0)

        # Check first active camera: all 4 tiers should be valid
        active_cam_tiers = mask[:4]
        np.testing.assert_array_equal(
            active_cam_tiers, np.ones(4, dtype=np.int8)
        )

    def test_zone_mask_tight_budget(self):
        """With a very tight budget, only minimal should be available."""
        env = make_env()
        env.reset(seed=42)
        # Tiny budget: only enough for all cameras at minimal
        env._zone_gpu_budget[0] = 0.5
        env._zone_bw_budget[0] = 0.5
        env._tiers[0, :] = 0
        mask = env._build_zone_mask(0)

        # For each active camera, check which tiers are allowed
        n_active = int(env._cams_per_zone[0])
        for c in range(n_active):
            offset = c * 4
            # Minimal should always be valid
            assert mask[offset] == 1
            # Priority (tier 3, cost=6.0) should be masked with 0.5 budget
            assert mask[offset + 3] == 0

    def test_zone_mask_inactive_camera_locked_to_minimal(self):
        """Inactive cameras should only have minimal tier available."""
        config = EnvironmentConfig()
        config.zones[0].max_cameras = 5
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        env._zone_gpu_budget[0] = 1000.0
        env._zone_bw_budget[0] = 1000.0
        mask = env._build_zone_mask(0)

        # Camera index 10 (inactive) should only allow minimal
        offset = 10 * 4
        expected = np.array([1, 0, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(mask[offset:offset + 4], expected)

    def test_tier_enforcement_clamps_actions(self):
        """Camera actions should be clamped to tier limits even if
        the policy outputs higher values."""
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        n = config.num_zones
        k = config.max_cameras_per_zone

        # Set all cameras to minimal tier
        zone_actions = {
            f"zone_{z}": np.zeros(k, dtype=np.int64)
            for z in range(n)
        }
        # But camera actions request max quality (2, 2, 2)
        camera_actions = {
            f"cam_{z}_{c}": np.array([2, 2, 2], dtype=np.int64)
            for z in range(n)
            for c in range(k)
        }
        actions = {
            "global": {"global": np.ones(n, dtype=np.int64)},
            "zone": zone_actions,
            "camera": camera_actions,
        }
        env.step(actions)

        # All cameras should be clamped to (0, 0, 0) since minimal tier
        for z in range(n):
            for c in range(k):
                if env._active[z, c]:
                    assert env._res[z, c] == 0
                    assert env._fps[z, c] == 0
                    assert env._model[z, c] == 0

    def test_elevated_tier_partial_clamp(self):
        """Elevated tier allows res=2 but limits fps and model to 1."""
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        n = config.num_zones
        k = config.max_cameras_per_zone

        zone_actions = {
            f"zone_{z}": np.full(k, 2, dtype=np.int64)  # elevated tier
            for z in range(n)
        }
        camera_actions = {
            f"cam_{z}_{c}": np.array([2, 2, 2], dtype=np.int64)
            for z in range(n)
            for c in range(k)
        }
        actions = {
            "global": {"global": np.ones(n, dtype=np.int64)},
            "zone": zone_actions,
            "camera": camera_actions,
        }
        env.step(actions)

        # res should stay at 2, fps and model clamped to 1
        for z in range(n):
            for c in range(k):
                if env._active[z, c]:
                    assert env._res[z, c] == 2
                    assert env._fps[z, c] == 1
                    assert env._model[z, c] == 1

    def test_full_action_masks_structure(self):
        """action_masks() should return the full nested dict with correct shapes."""
        env = make_env()
        env.reset(seed=42)
        masks = env.action_masks()
        spaces = env.action_spaces()

        for level in ["global", "zone", "camera"]:
            for agent_id, space in spaces[level].items():
                expected_len = int(space.nvec.sum())
                assert masks[level][agent_id].shape == (expected_len,)
                # Every mask should have at least one valid option per dimension
                assert masks[level][agent_id].sum() > 0

    def test_masks_respect_tier_after_step(self):
        """After a step that sets tiers, camera masks should reflect them."""
        env = make_env()
        env.reset(seed=42)
        config = EnvironmentConfig()
        n = config.num_zones
        k = config.max_cameras_per_zone

        # Set zone 0 cameras to normal tier (1), zone 1+ to priority
        zone_actions = {}
        for z in range(n):
            if z == 0:
                zone_actions[f"zone_{z}"] = np.ones(k, dtype=np.int64)
            else:
                zone_actions[f"zone_{z}"] = np.full(k, 3, dtype=np.int64)

        actions = {
            "global": {"global": np.ones(n, dtype=np.int64)},
            "zone": zone_actions,
            "camera": {
                f"cam_{z}_{c}": np.zeros(3, dtype=np.int64)
                for z in range(n)
                for c in range(k)
            },
        }
        env.step(actions)
        masks = env.action_masks()

        # Zone 0 camera 0: normal tier, should not allow index 2 for any dim
        cam_mask = masks["camera"]["cam_0_0"]
        assert cam_mask[2] == 0  # res=1080p masked
        assert cam_mask[5] == 0  # fps=30 masked
        assert cam_mask[8] == 0  # model=heavy masked

        # Zone 1 camera 0: priority tier, everything open
        cam_mask_p = masks["camera"]["cam_1_0"]
        np.testing.assert_array_equal(cam_mask_p, np.ones(9, dtype=np.int8))


class TestWeather:
    """Tests for the Markov weather state machine."""

    def test_weather_transitions_occur(self):
        """Running many steps should produce at least one weather change."""
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        states_seen = {env._weather}
        for _ in range(200):
            env.step(sample_random_actions(env, rng))
            states_seen.add(env._weather)
        assert len(states_seen) > 1, "Weather should transition at least once"

    def test_fog_reduces_confidence(self):
        """Fog weather should lower detection confidence compared to clear."""
        env = make_env()
        env.reset(seed=42)
        env._weather = 0  # clear
        env._update_confidence()
        clear_conf = env._confidence[env._active].mean()

        env._weather = 2  # fog
        env._update_confidence()
        fog_conf = env._confidence[env._active].mean()
        assert fog_conf < clear_conf

    def test_rain_boosts_highway_anomaly(self):
        """Rain should increase anomaly rates for highway zones."""
        env = make_env()
        env.reset(seed=42)
        env._weather = 0
        rates_clear = env._compute_base_rates()
        env._weather = 1  # rain
        rates_rain = env._compute_base_rates()

        # Highway is zone index 2 in the default 4-zone config
        hw_z = 2
        n = int(env._cams_per_zone[hw_z])
        assert rates_rain[hw_z, :n].sum() > rates_clear[hw_z, :n].sum()

    def test_weather_stays_valid(self):
        """Weather index should always be 0, 1, or 2."""
        env = make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        for _ in range(100):
            env.step(sample_random_actions(env, rng))
            assert 0 <= env._weather <= 2


class TestFailures:
    """Tests for infrastructure failure injection and recovery."""

    def test_camera_dropout_deactivates(self):
        """A camera in dropout should be marked inactive."""
        env = make_env()
        env.reset(seed=42)
        env._camera_dropout[0, 0] = 5
        env._update_failures()
        assert not env._active[0, 0]

    def test_camera_recovers_after_countdown(self):
        """Camera should become active again when dropout countdown hits 0."""
        config = EnvironmentConfig()
        # Zero injection prob so recovery isn't overwritten by a new failure
        config.failures.camera_dropout_prob = 0.0
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        env._camera_dropout[0, 0] = 1
        env._update_failures()
        assert env._active[0, 0], "Camera should recover when dropout expires"

    def test_gpu_failure_reduces_effective_budget(self):
        """Active GPU failure should reduce the effective zone budget."""
        env = make_env()
        env.reset(seed=42)
        env._zone_gpu_budget[0] = 50.0
        env._gpu_failure_remaining[0] = 10
        env._gpu_failure_loss[0] = 0.3
        effective = env._effective_zone_gpu_budget(0)
        assert abs(effective - 35.0) < 0.01

    def test_network_congestion_reduces_bandwidth(self):
        """Active network congestion should reduce effective bandwidth."""
        env = make_env()
        env.reset(seed=42)
        env._zone_bw_budget[0] = 100.0
        env._net_congestion_remaining[0] = 10
        env._net_congestion_loss[0] = 0.4
        effective = env._effective_zone_bw_budget(0)
        # 100 * (1 - 0.4) * weather_modifier(clear=1.0) = 60
        assert abs(effective - 60.0) < 0.01

    def test_failures_inject_with_high_prob(self):
        """With high failure probability, failures should appear quickly."""
        config = EnvironmentConfig()
        config.failures.camera_dropout_prob = 0.5
        config.failures.gpu_failure_prob = 0.3
        config.failures.network_congestion_prob = 0.3
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        rng = np.random.default_rng(42)
        saw_dropout = False
        saw_gpu = False
        for _ in range(50):
            env.step(sample_random_actions(env, rng))
            if (env._camera_dropout > 0).any():
                saw_dropout = True
            if (env._gpu_failure_remaining > 0).any():
                saw_gpu = True
        assert saw_dropout
        assert saw_gpu

    def test_failure_clears_on_expiry(self):
        """GPU failure loss should reset to 0 when the countdown expires."""
        config = EnvironmentConfig()
        config.failures.gpu_failure_prob = 0.0
        env = UrbanEnvironment(config)
        env.reset(seed=42)
        env._gpu_failure_remaining[0] = 1
        env._gpu_failure_loss[0] = 0.3
        env._update_failures()
        assert env._gpu_failure_remaining[0] == 0
        assert env._gpu_failure_loss[0] == 0.0


class TestDomainRandomization:
    """Tests for domain randomization during reset."""

    def test_camera_counts_vary(self):
        """Multiple resets with DR enabled should produce different counts."""
        config = EnvironmentConfig()
        config.domain_randomization = DomainRandomizationConfig(
            enabled=True,
            camera_count_range=[4, 12],
            gpu_budget_range=[50.0, 80.0],
            bandwidth_budget_range=[50.0, 80.0],
        )
        env = UrbanEnvironment(config)
        counts = set()
        for seed in range(20):
            env.reset(seed=seed)
            counts.add(env.num_cameras)
        assert len(counts) > 1

    def test_camera_counts_within_bounds(self):
        """Camera count should stay within the configured range."""
        config = EnvironmentConfig()
        config.domain_randomization = DomainRandomizationConfig(
            enabled=True,
            camera_count_range=[5, 10],
        )
        env = UrbanEnvironment(config)
        for seed in range(30):
            env.reset(seed=seed)
            assert 5 <= env.num_cameras <= 10

    def test_budget_varies(self):
        """GPU budget should vary across resets with DR enabled."""
        config = EnvironmentConfig()
        config.domain_randomization = DomainRandomizationConfig(
            enabled=True,
            gpu_budget_range=[40.0, 90.0],
            bandwidth_budget_range=[40.0, 90.0],
        )
        env = UrbanEnvironment(config)
        budgets = set()
        for seed in range(20):
            env.reset(seed=seed)
            budgets.add(round(config.resources.total_gpu_budget, 1))
        assert len(budgets) > 1

    def test_dr_disabled_keeps_constant(self):
        """With DR disabled, camera count should be constant across resets."""
        config = EnvironmentConfig()
        config.domain_randomization.enabled = False
        env = UrbanEnvironment(config)
        counts = set()
        for seed in range(10):
            env.reset(seed=seed)
            counts.add(env.num_cameras)
        assert len(counts) == 1


class TestScenarioPresets:
    """Verify all scenario preset configs load and run."""

    PRESETS = [
        "urban_normal",
        "urban_rush_hour",
        "urban_resource_crunch",
        "urban_infrastructure_stress",
        "urban_incident_cascade",
        "urban_mixed",
    ]

    def test_all_presets_load_and_step(self):
        for preset in self.PRESETS:
            raw = load_config(f"configs/environments/{preset}.yaml")
            config = env_config_from_dict(raw)
            env = UrbanEnvironment(config)
            obs = env.reset(seed=42)
            rng = np.random.default_rng(42)
            obs, _, _, _, _ = env.step(sample_random_actions(env, rng))
            spaces = env.observation_spaces()
            for level in spaces:
                for aid, space in spaces[level].items():
                    assert obs[level][aid].shape == space.shape, (
                        f"{preset}: {aid} shape mismatch"
                    )

    def test_resource_crunch_has_lower_budget(self):
        raw = load_config("configs/environments/urban_resource_crunch.yaml")
        config = env_config_from_dict(raw)
        assert config.resources.total_gpu_budget == 50.0

    def test_stress_has_failures_enabled(self):
        raw = load_config(
            "configs/environments/urban_infrastructure_stress.yaml"
        )
        config = env_config_from_dict(raw)
        assert config.failures.camera_dropout_prob > 0

    def test_mixed_has_dr_enabled(self):
        raw = load_config("configs/environments/urban_mixed.yaml")
        config = env_config_from_dict(raw)
        assert config.domain_randomization.enabled is True
