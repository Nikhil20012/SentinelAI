"""Constants for SentinelAI simulation environments.

Defines the discrete option sets (resolutions, FPS, model tiers, budget levels,
resource access tiers) and the resource cost model that determines how much GPU
and bandwidth each quality combination consumes.
"""

import numpy as np

# Camera quality settings (indexed 0, 1, 2)
RESOLUTIONS = [480, 720, 1080]
FPS_LEVELS = [5, 15, 30]
MODEL_TIERS = ["lightweight", "standard", "heavy"]
NUM_QUALITY_LEVELS = 3

# Budget levels the global controller assigns to zones
BUDGET_LEVELS = ["low", "moderate", "high", "critical"]
NUM_BUDGET_LEVELS = 4
BUDGET_FRACTIONS = np.array([0.15, 0.25, 0.35, 0.50])

# Resource access tiers the zone controller assigns to cameras
TIER_NAMES = ["minimal", "normal", "elevated", "priority"]
NUM_TIERS = 4

# Max allowed action index per dimension (res, fps, model) for each tier.
# Row = tier index, columns = (max_res, max_fps, max_model).
TIER_ACTION_LIMITS = np.array([
    [0, 0, 0],  # minimal: 480p / 5fps / lightweight only
    [1, 1, 1],  # normal: up to 720p / 15fps / standard
    [2, 1, 1],  # elevated: up to 1080p / 15fps / standard
    [2, 2, 2],  # priority: full action space
])

# Resource cost model.
# GPU cost = GPU_SCALE * res_weight * fps_weight * model_weight
# BW cost  = BW_SCALE  * res_weight * fps_weight
#
# Calibrated so that ~40% of cameras at max quality exhausts the default
# budget of 100. With 40 active cameras:
#   40 * 6.0 * 1.0 * 1.0 * 1.0 = 240 (all max, 2.4x budget)
#   16 * 6.0 + 24 * 6.0 * 0.018 = 98.6 (40% max, 60% min, just under budget)
GPU_SCALE = 6.0
BW_SCALE = 4.0

RESOLUTION_GPU_WEIGHT = np.array([0.3, 0.6, 1.0])
FPS_GPU_WEIGHT = np.array([0.2, 0.5, 1.0])
MODEL_GPU_WEIGHT = np.array([0.3, 0.6, 1.0])

RESOLUTION_BW_WEIGHT = np.array([0.3, 0.6, 1.0])
FPS_BW_WEIGHT = np.array([0.2, 0.5, 1.0])

# Weather
WEATHER_STATES = ["clear", "rain", "fog"]
NUM_WEATHER_STATES = 3

# Zone types
ZONE_TYPES = ["commercial", "residential", "highway", "parking"]

# A camera with anomaly_score above this counts as an incident
ANOMALY_INCIDENT_THRESHOLD = 0.7

# Time-of-day anomaly rate multipliers per zone type.
# 24 values (one per hour), linearly interpolated for fractional hours.
# These multiply the zone's base_anomaly_rate.
TIME_PROFILES = {
    "commercial": [
        0.20, 0.20, 0.20, 0.20, 0.20, 0.30,
        0.50, 0.70, 0.90, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00, 1.00, 1.00, 0.90,
        0.70, 0.50, 0.40, 0.30, 0.30, 0.20,
    ],
    "residential": [
        0.30, 0.20, 0.20, 0.20, 0.20, 0.30,
        0.50, 0.70, 0.80, 0.50, 0.30, 0.30,
        0.30, 0.30, 0.30, 0.40, 0.50, 0.70,
        0.90, 1.00, 0.90, 0.70, 0.50, 0.40,
    ],
    "highway": [
        0.20, 0.15, 0.10, 0.10, 0.15, 0.30,
        0.60, 0.90, 1.00, 0.80, 0.50, 0.40,
        0.40, 0.40, 0.40, 0.50, 0.70, 1.00,
        0.90, 0.60, 0.40, 0.30, 0.25, 0.20,
    ],
    "parking": [
        0.20, 0.15, 0.10, 0.10, 0.10, 0.15,
        0.30, 0.50, 0.70, 0.80, 0.80, 0.80,
        0.80, 0.80, 0.80, 0.70, 0.60, 0.50,
        0.40, 0.30, 0.25, 0.20, 0.20, 0.20,
    ],
}

# Moving hotspot parameters
HOTSPOT_SPAWN_PROB = 0.02
HOTSPOT_INTENSITY = 3.0
HOTSPOT_SPREAD = 2  # cameras affected in each direction from center
HOTSPOT_LIFETIME_RANGE = (20, 60)

# Anomaly score Beta distribution parameters
# Calm cameras produce low scores, event cameras produce high scores
ANOMALY_BETA_CALM = (1.5, 8.0)
ANOMALY_BETA_EVENT = (5.0, 2.0)

# Maximum resource cost a camera can incur at each tier.
# Used by zone controllers to check whether a tier assignment fits the budget.
# Derived from TIER_ACTION_LIMITS and the cost weight tables.
TIER_MAX_GPU_COST = np.array([
    GPU_SCALE
    * RESOLUTION_GPU_WEIGHT[lim[0]]
    * FPS_GPU_WEIGHT[lim[1]]
    * MODEL_GPU_WEIGHT[lim[2]]
    for lim in TIER_ACTION_LIMITS
])

TIER_MAX_BW_COST = np.array([
    BW_SCALE * RESOLUTION_BW_WEIGHT[lim[0]] * FPS_BW_WEIGHT[lim[1]]
    for lim in TIER_ACTION_LIMITS
])
