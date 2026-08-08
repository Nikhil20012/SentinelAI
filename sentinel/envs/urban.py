"""Urban surveillance camera network environment.

A PettingZoo-style parallel environment where cameras in a city surveillance
network need adaptive compute allocation. Three hierarchy levels (global, zone,
camera) observe and act simultaneously each timestep.

Current implementation (Step 1): basic skeleton with uniform random anomaly
scores, fixed weather, no failures, no domain randomization, no action mask
enforcement. The goal is a correct and testable interface that later steps
build on incrementally.
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np

from sentinel.config import EnvironmentConfig
from sentinel.envs.base import BaseEnvironment
from sentinel.envs.constants import (
    ANOMALY_BETA_CALM,
    ANOMALY_BETA_EVENT,
    ANOMALY_INCIDENT_THRESHOLD,
    BUDGET_FRACTIONS,
    BW_SCALE,
    FPS_BW_WEIGHT,
    FPS_GPU_WEIGHT,
    GPU_SCALE,
    HOTSPOT_INTENSITY,
    HOTSPOT_LIFETIME_RANGE,
    HOTSPOT_SPAWN_PROB,
    HOTSPOT_SPREAD,
    MODEL_GPU_WEIGHT,
    NUM_BUDGET_LEVELS,
    NUM_QUALITY_LEVELS,
    NUM_TIERS,
    NUM_WEATHER_STATES,
    RESOLUTION_BW_WEIGHT,
    RESOLUTION_GPU_WEIGHT,
    TIER_ACTION_LIMITS,
    TIER_MAX_BW_COST,
    TIER_MAX_GPU_COST,
    TIME_PROFILES,
)

logger = logging.getLogger(__name__)


def _interpolate_time_profile(zone_type: str, hour: float) -> float:
    """Linearly interpolate the 24-hour anomaly rate profile for a zone type."""
    profile = TIME_PROFILES.get(zone_type, TIME_PROFILES["commercial"])
    idx = int(hour) % 24
    next_idx = (idx + 1) % 24
    frac = hour - int(hour)
    return profile[idx] * (1.0 - frac) + profile[next_idx] * frac


class UrbanEnvironment(BaseEnvironment):
    """Simulated urban camera network with hierarchical resource allocation.

    Hierarchy:
        Global controller  - allocates GPU/bandwidth budgets across zones
        Zone controllers   - assign resource-access tiers to cameras
        Camera agents      - choose resolution, FPS, and model tier
    """

    def __init__(self, config: EnvironmentConfig):
        self._config = config
        self._n_zones = config.num_zones
        self._max_cams = config.max_cameras_per_zone
        self._episode_len = config.episode_length

        self._cams_per_zone = np.array(
            [z.max_cameras for z in config.zones], dtype=np.int32
        )

        # Observation and action space dimensions
        self._global_obs_dim = 3 * self._n_zones + 6
        self._zone_obs_dim = 6 * self._max_cams + 4
        self._camera_obs_dim = 9

        self._obs_spaces = self._build_obs_spaces()
        self._act_spaces = self._build_action_spaces()

        # State arrays, populated by reset()
        self._anomaly: np.ndarray | None = None
        self._res: np.ndarray | None = None
        self._fps: np.ndarray | None = None
        self._model: np.ndarray | None = None
        self._confidence: np.ndarray | None = None
        self._active: np.ndarray | None = None
        self._tiers: np.ndarray | None = None
        self._zone_gpu_budget: np.ndarray | None = None
        self._zone_bw_budget: np.ndarray | None = None
        self._gpu_costs: np.ndarray | None = None
        self._bw_costs: np.ndarray | None = None
        self._spatial_boost: np.ndarray | None = None
        self._hotspots: list[dict] = []

        self._timestep = 0
        self._time_of_day = 0.0
        self._weather = 0
        self._rng: np.random.Generator | None = None

    # -- Space definitions --

    def _build_obs_spaces(self) -> dict:
        global_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self._global_obs_dim,), dtype=np.float32
        )
        zone_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self._zone_obs_dim,), dtype=np.float32
        )
        camera_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self._camera_obs_dim,), dtype=np.float32
        )

        return {
            "global": {"global": global_space},
            "zone": {f"zone_{z}": zone_space for z in range(self._n_zones)},
            "camera": {
                f"cam_{z}_{c}": camera_space
                for z in range(self._n_zones)
                for c in range(self._max_cams)
            },
        }

    def _build_action_spaces(self) -> dict:
        global_action = gym.spaces.MultiDiscrete(
            [NUM_BUDGET_LEVELS] * self._n_zones
        )
        zone_action = gym.spaces.MultiDiscrete(
            [NUM_TIERS] * self._max_cams
        )
        camera_action = gym.spaces.MultiDiscrete(
            [NUM_QUALITY_LEVELS] * 3
        )

        return {
            "global": {"global": global_action},
            "zone": {f"zone_{z}": zone_action for z in range(self._n_zones)},
            "camera": {
                f"cam_{z}_{c}": camera_action
                for z in range(self._n_zones)
                for c in range(self._max_cams)
            },
        }

    def observation_spaces(self) -> dict:
        return self._obs_spaces

    def action_spaces(self) -> dict:
        return self._act_spaces

    # -- Properties --

    @property
    def num_zones(self) -> int:
        return self._n_zones

    @property
    def max_cameras_per_zone(self) -> int:
        return self._max_cams

    @property
    def num_cameras(self) -> int:
        if self._active is None:
            return 0
        return int(self._active.sum())

    # -- Core loop --

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._rng = np.random.default_rng(seed)
        self._timestep = 0
        self._time_of_day = self._rng.uniform(0.0, 1.0)
        self._weather = 0

        shape = (self._n_zones, self._max_cams)

        # All cameras start at minimum quality
        self._res = np.zeros(shape, dtype=np.int32)
        self._fps = np.zeros(shape, dtype=np.int32)
        self._model = np.zeros(shape, dtype=np.int32)

        # Which cameras are active (based on configured count per zone)
        self._active = np.zeros(shape, dtype=bool)
        for z in range(self._n_zones):
            self._active[z, : self._cams_per_zone[z]] = True

        # All cameras start at priority tier (no restrictions)
        self._tiers = np.full(shape, NUM_TIERS - 1, dtype=np.int32)

        # Spatial correlation tracking and moving hotspots
        self._spatial_boost = np.zeros(shape, dtype=np.float32)
        self._hotspots = []

        # Even budget split across zones
        frac = 1.0 / self._n_zones
        self._zone_gpu_budget = np.full(
            self._n_zones, frac * self._config.resources.total_gpu_budget
        )
        self._zone_bw_budget = np.full(
            self._n_zones, frac * self._config.resources.total_bandwidth_budget
        )

        self._generate_anomalies()
        self._update_confidence()
        self._update_resource_costs()

        logger.debug(
            "Environment reset: %d zones, %d active cameras, seed=%s",
            self._n_zones, self.num_cameras, seed,
        )
        return self._build_observations()

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict, dict, dict, dict, dict]:
        self._apply_actions(actions)

        self._timestep += 1
        self._time_of_day = (self._time_of_day + 1.0 / self._episode_len) % 1.0

        self._update_hotspots()
        self._generate_anomalies()
        self._update_spatial_boost()
        self._update_confidence()
        self._update_resource_costs()

        obs = self._build_observations()
        rewards = self._placeholder_rewards()

        done = self._timestep >= self._episode_len
        terminateds = self._constant_agent_dict(False)
        truncateds = self._constant_agent_dict(done)
        infos = self._build_infos()

        return obs, rewards, terminateds, truncateds, infos

    # -- Action processing --

    def _apply_actions(self, actions: dict[str, Any]) -> None:
        """Apply actions from all three hierarchy levels."""

        # Global: update zone budgets
        global_action = np.asarray(actions["global"]["global"])
        fractions = BUDGET_FRACTIONS[global_action]
        total = fractions.sum()
        if total > 1.0:
            fractions = fractions / total
        self._zone_gpu_budget = fractions * self._config.resources.total_gpu_budget
        self._zone_bw_budget = fractions * self._config.resources.total_bandwidth_budget

        # Zone: update camera tier assignments
        for z in range(self._n_zones):
            zone_action = np.asarray(actions["zone"][f"zone_{z}"])
            self._tiers[z] = zone_action

        # Camera: update quality settings, clamped to tier limits
        for z in range(self._n_zones):
            for c in range(self._max_cams):
                if not self._active[z, c]:
                    continue
                cam_action = np.asarray(actions["camera"][f"cam_{z}_{c}"])
                limits = TIER_ACTION_LIMITS[self._tiers[z, c]]
                self._res[z, c] = min(int(cam_action[0]), int(limits[0]))
                self._fps[z, c] = min(int(cam_action[1]), int(limits[1]))
                self._model[z, c] = min(int(cam_action[2]), int(limits[2]))

    # -- World simulation --

    def _generate_anomalies(self) -> None:
        """Generate anomaly scores from zone-aware rates.

        Each camera's effective anomaly rate comes from:
        1. Zone base rate (config) * time-of-day multiplier (zone type profile)
        2. Moving hotspot influence (if any active nearby)
        3. Spatial correlation boost (from recent neighbor incidents)

        The rate drives a mixture model: with probability=rate the camera
        draws from a high-anomaly Beta distribution, otherwise from a
        low-anomaly Beta distribution.
        """
        rates = self._compute_base_rates()
        rates += self._spatial_boost
        rates = np.clip(rates, 0.0, 1.0)

        is_event = self._rng.random(size=rates.shape) < rates
        calm = self._rng.beta(
            ANOMALY_BETA_CALM[0], ANOMALY_BETA_CALM[1], size=rates.shape
        )
        event = self._rng.beta(
            ANOMALY_BETA_EVENT[0], ANOMALY_BETA_EVENT[1], size=rates.shape
        )
        self._anomaly = np.where(is_event, event, calm).astype(np.float32)
        self._anomaly[~self._active] = 0.0

    def _compute_base_rates(self) -> np.ndarray:
        """Per-camera anomaly rates from zone config, time-of-day, and hotspots."""
        rates = np.zeros((self._n_zones, self._max_cams), dtype=np.float32)
        hour = self._time_of_day * 24.0

        for z in range(self._n_zones):
            zone_cfg = self._config.zones[z]
            time_mult = _interpolate_time_profile(zone_cfg.zone_type, hour)
            rates[z, : self._cams_per_zone[z]] = zone_cfg.base_anomaly_rate * time_mult

        # Hotspot contribution: cameras near the center get elevated rates
        for h in self._hotspots:
            z = h["zone"]
            center = h["center"]
            base = self._config.zones[z].base_anomaly_rate
            lo = max(0, center - HOTSPOT_SPREAD)
            hi = min(int(self._cams_per_zone[z]), center + HOTSPOT_SPREAD + 1)
            for c in range(lo, hi):
                if self._active[z, c]:
                    dist = abs(c - center)
                    # Linear falloff from hotspot center
                    scale = 1.0 - dist / (HOTSPOT_SPREAD + 1)
                    rates[z, c] += h["intensity"] * base * scale

        return rates

    def _update_spatial_boost(self) -> None:
        """Cameras adjacent to incidents get a temporary anomaly rate boost.

        Existing boosts decay each step. New incidents add fresh boost to
        immediate left/right neighbors within the same zone.
        """
        decay_steps = self._config.spatial_correlation_decay
        if decay_steps > 0:
            self._spatial_boost *= 1.0 - 1.0 / decay_steps
        else:
            self._spatial_boost[:] = 0.0

        boost = self._config.spatial_correlation_boost
        incidents = (
            (self._anomaly > ANOMALY_INCIDENT_THRESHOLD) & self._active
        ).astype(np.float32) * boost

        # Shift incident signal to neighboring positions
        self._spatial_boost[:, 1:] += incidents[:, :-1]
        self._spatial_boost[:, :-1] += incidents[:, 1:]

        self._spatial_boost = np.clip(self._spatial_boost, 0.0, 0.5)
        self._spatial_boost[~self._active] = 0.0

    def _update_hotspots(self) -> None:
        """Manage moving hotspot lifecycle: advance existing, expire old, spawn new."""
        surviving = []
        for h in self._hotspots:
            h["remaining"] -= 1
            if h["remaining"] <= 0:
                continue
            # Move the center, bounce off zone edges
            h["center"] += h["direction"]
            n_cams = int(self._cams_per_zone[h["zone"]])
            if h["center"] < 0:
                h["center"] = 0
                h["direction"] = 1
            elif h["center"] >= n_cams:
                h["center"] = n_cams - 1
                h["direction"] = -1
            surviving.append(h)
        self._hotspots = surviving

        for z in range(self._n_zones):
            n_cams = int(self._cams_per_zone[z])
            if n_cams == 0:
                continue
            if self._rng.random() < HOTSPOT_SPAWN_PROB:
                lifetime = int(
                    self._rng.integers(HOTSPOT_LIFETIME_RANGE[0], HOTSPOT_LIFETIME_RANGE[1])
                )
                self._hotspots.append({
                    "zone": z,
                    "center": int(self._rng.integers(0, n_cams)),
                    "intensity": float(HOTSPOT_INTENSITY),
                    "direction": int(self._rng.choice([-1, 1])),
                    "remaining": lifetime,
                })

    def _update_confidence(self) -> None:
        """Detection confidence as a function of camera quality settings.

        Higher resolution, FPS, and model tier produce higher confidence.
        """
        quality = (self._res + self._fps + self._model).astype(np.float32) / 6.0
        noise = self._rng.uniform(
            -0.1, 0.1, size=quality.shape
        ).astype(np.float32)
        self._confidence = np.clip(0.5 + 0.3 * quality + noise, 0.0, 1.0)
        self._confidence[~self._active] = 0.0

    def _update_resource_costs(self) -> None:
        """Recompute per-camera GPU and bandwidth costs from current quality."""
        self._gpu_costs = (
            GPU_SCALE
            * RESOLUTION_GPU_WEIGHT[self._res]
            * FPS_GPU_WEIGHT[self._fps]
            * MODEL_GPU_WEIGHT[self._model]
        ).astype(np.float32)
        self._gpu_costs[~self._active] = 0.0

        self._bw_costs = (
            BW_SCALE
            * RESOLUTION_BW_WEIGHT[self._res]
            * FPS_BW_WEIGHT[self._fps]
        ).astype(np.float32)
        self._bw_costs[~self._active] = 0.0

    # -- Observation construction --

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:
        zone_gpu_usage = self._gpu_costs.sum(axis=1)
        zone_bw_usage = self._bw_costs.sum(axis=1)
        total_gpu = self._config.resources.total_gpu_budget
        total_bw = self._config.resources.total_bandwidth_budget

        # Count incidents per zone (cameras above anomaly threshold)
        zone_incidents = (
            (self._anomaly > ANOMALY_INCIDENT_THRESHOLD) & self._active
        ).sum(axis=1).astype(np.float32)

        global_obs = self._build_global_obs(
            zone_gpu_usage, zone_bw_usage, zone_incidents, total_gpu, total_bw
        )
        zone_obs = {
            f"zone_{z}": self._build_zone_obs(z, zone_incidents[z])
            for z in range(self._n_zones)
        }
        camera_obs = {}
        for z in range(self._n_zones):
            zone_gpu_util = (
                zone_gpu_usage[z] / self._zone_gpu_budget[z]
                if self._zone_gpu_budget[z] > 0 else 0.0
            )
            zone_bw_util = (
                zone_bw_usage[z] / self._zone_bw_budget[z]
                if self._zone_bw_budget[z] > 0 else 0.0
            )
            # Neighbor mean excludes the camera itself
            zone_anomaly = self._anomaly[z]
            zone_active = self._active[z]
            n_active = zone_active.sum()

            for c in range(self._max_cams):
                cam_id = f"cam_{z}_{c}"
                if not self._active[z, c]:
                    camera_obs[cam_id] = np.zeros(
                        self._camera_obs_dim, dtype=np.float32
                    )
                    continue

                if n_active > 1:
                    neighbor_sum = zone_anomaly[zone_active].sum() - zone_anomaly[c]
                    neighbor_mean = neighbor_sum / (n_active - 1)
                else:
                    neighbor_mean = 0.0

                camera_obs[cam_id] = np.array([
                    self._anomaly[z, c],
                    self._res[z, c] / 2.0,
                    self._fps[z, c] / 2.0,
                    self._model[z, c] / 2.0,
                    self._confidence[z, c],
                    float(zone_gpu_util),
                    float(zone_bw_util),
                    self._tiers[z, c] / 3.0,
                    float(neighbor_mean),
                ], dtype=np.float32)

        return {
            "global": {"global": global_obs},
            "zone": zone_obs,
            "camera": camera_obs,
        }

    def _build_global_obs(
        self,
        zone_gpu_usage: np.ndarray,
        zone_bw_usage: np.ndarray,
        zone_incidents: np.ndarray,
        total_gpu: float,
        total_bw: float,
    ) -> np.ndarray:
        zone_anomaly_means = np.zeros(self._n_zones, dtype=np.float32)
        for z in range(self._n_zones):
            active = self._active[z]
            if active.any():
                zone_anomaly_means[z] = self._anomaly[z, active].mean()

        # Normalize zone-level features
        zone_gpu_norm = zone_gpu_usage / total_gpu
        zone_incident_norm = zone_incidents / self._max_cams

        # Weather one-hot
        weather_onehot = np.zeros(NUM_WEATHER_STATES, dtype=np.float32)
        weather_onehot[self._weather] = 1.0

        return np.concatenate([
            [float(self._gpu_costs.sum() / total_gpu)],
            [float(self._bw_costs.sum() / total_bw)],
            zone_anomaly_means,
            zone_gpu_norm.astype(np.float32),
            zone_incident_norm,
            [self._time_of_day],
            weather_onehot,
        ]).astype(np.float32)

    def _build_zone_obs(self, zone_idx: int, incident_count: float) -> np.ndarray:
        z = zone_idx
        total_gpu = self._config.resources.total_gpu_budget
        total_bw = self._config.resources.total_bandwidth_budget

        # Count failed cameras (none in skeleton, placeholder for later)
        failure_count = 0.0

        return np.concatenate([
            [self._zone_gpu_budget[z] / total_gpu],
            [self._zone_bw_budget[z] / total_bw],
            self._anomaly[z].astype(np.float32),
            (self._res[z] / 2.0).astype(np.float32),
            (self._fps[z] / 2.0).astype(np.float32),
            (self._model[z] / 2.0).astype(np.float32),
            self._confidence[z].astype(np.float32),
            self._active[z].astype(np.float32),
            [incident_count / self._max_cams],
            [failure_count / self._max_cams],
        ]).astype(np.float32)

    # -- Rewards --

    def _placeholder_rewards(self) -> dict[str, dict[str, float]]:
        """Placeholder zero rewards. Real rewards come from the reward module."""
        return self._constant_agent_dict(0.0)

    # -- Action masking --

    def action_masks(self) -> dict[str, Any]:
        """Return masks where 1 = valid action, 0 = masked.

        For MultiDiscrete spaces, the mask is a flat 1D array of length
        sum(nvec). Each segment corresponds to one discrete dimension.

        Global controller: all budget levels always valid.
        Zone controllers: tier masked if its max cost would exceed zone budget
            headroom (computed assuming other cameras keep their current tier).
        Camera agents: per-dimension masking from assigned tier limits.
        """
        masks: dict[str, dict[str, np.ndarray]] = {
            "global": {},
            "zone": {},
            "camera": {},
        }

        # Global: no constraints on budget level assignment
        global_space = self._act_spaces["global"]["global"]
        masks["global"]["global"] = np.ones(
            int(global_space.nvec.sum()), dtype=np.int8
        )

        # Zone: budget-based tier masking
        for z in range(self._n_zones):
            masks["zone"][f"zone_{z}"] = self._build_zone_mask(z)

        # Camera: tier-based per-dimension masking
        for z in range(self._n_zones):
            for c in range(self._max_cams):
                masks["camera"][f"cam_{z}_{c}"] = self._build_camera_mask(z, c)

        return masks

    def _build_zone_mask(self, zone_idx: int) -> np.ndarray:
        """Build tier masks for all cameras in a zone based on budget headroom.

        For each camera, compute how much budget is available assuming all
        other cameras stay at their current tier. Mask any tier whose max
        cost would exceed that headroom.
        """
        z = zone_idx
        mask = np.zeros(NUM_TIERS * self._max_cams, dtype=np.int8)

        # Cost of each camera's current tier assignment
        current_gpu = TIER_MAX_GPU_COST[self._tiers[z]] * self._active[z]
        current_bw = TIER_MAX_BW_COST[self._tiers[z]] * self._active[z]
        total_gpu = current_gpu.sum()
        total_bw = current_bw.sum()

        for c in range(self._max_cams):
            offset = c * NUM_TIERS

            if not self._active[z, c]:
                mask[offset] = 1  # inactive cameras locked to minimal
                continue

            # Budget available for this camera = total - everyone else's cost
            headroom_gpu = self._zone_gpu_budget[z] - (total_gpu - current_gpu[c])
            headroom_bw = self._zone_bw_budget[z] - (total_bw - current_bw[c])

            for t in range(NUM_TIERS):
                if (
                    TIER_MAX_GPU_COST[t] <= headroom_gpu + 1e-6
                    and TIER_MAX_BW_COST[t] <= headroom_bw + 1e-6
                ):
                    mask[offset + t] = 1

            # Minimal is always valid regardless of budget
            mask[offset] = 1

        return mask

    def _build_camera_mask(self, zone_idx: int, cam_idx: int) -> np.ndarray:
        """Build per-dimension action mask for a camera based on its assigned tier."""
        mask = np.zeros(NUM_QUALITY_LEVELS * 3, dtype=np.int8)

        if not self._active[zone_idx, cam_idx]:
            # Inactive: only the lowest option per dimension
            mask[0] = 1
            mask[NUM_QUALITY_LEVELS] = 1
            mask[2 * NUM_QUALITY_LEVELS] = 1
            return mask

        tier = self._tiers[zone_idx, cam_idx]
        limits = TIER_ACTION_LIMITS[tier]

        for dim in range(3):
            offset = dim * NUM_QUALITY_LEVELS
            for i in range(int(limits[dim]) + 1):
                mask[offset + i] = 1

        return mask

    # -- Rendering / logging --

    def render(self) -> dict[str, Any] | None:
        """Serializable snapshot of current state for episode recording."""
        if self._anomaly is None:
            return None
        return {
            "timestep": self._timestep,
            "time_of_day": self._time_of_day,
            "weather": self._weather,
            "anomaly": self._anomaly.copy(),
            "res": self._res.copy(),
            "fps": self._fps.copy(),
            "model": self._model.copy(),
            "confidence": self._confidence.copy(),
            "active": self._active.copy(),
            "tiers": self._tiers.copy(),
            "zone_gpu_budget": self._zone_gpu_budget.copy(),
            "zone_bw_budget": self._zone_bw_budget.copy(),
            "gpu_costs": self._gpu_costs.copy(),
            "bw_costs": self._bw_costs.copy(),
            "spatial_boost": self._spatial_boost.copy(),
            "num_hotspots": len(self._hotspots),
        }

    def _build_infos(self) -> dict[str, dict[str, dict]]:
        """Per-agent info dicts. Includes resource usage for reward computation."""
        base_info = {
            "timestep": self._timestep,
            "total_gpu_usage": float(self._gpu_costs.sum()),
            "total_bw_usage": float(self._bw_costs.sum()),
            "total_gpu_budget": self._config.resources.total_gpu_budget,
            "total_bw_budget": self._config.resources.total_bandwidth_budget,
        }
        # Every agent gets the same info dict for now. The reward module
        # can pull what it needs from here.
        return self._constant_agent_dict(base_info)

    # -- Helpers --

    def _constant_agent_dict(self, value: Any) -> dict[str, dict[str, Any]]:
        """Build the nested {level: {agent_id: value}} structure."""
        return {
            "global": {"global": value},
            "zone": {f"zone_{z}": value for z in range(self._n_zones)},
            "camera": {
                f"cam_{z}_{c}": value
                for z in range(self._n_zones)
                for c in range(self._max_cams)
            },
        }
