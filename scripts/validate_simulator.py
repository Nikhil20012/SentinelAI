"""Validate the urban environment simulator dynamics.

Generates plots confirming that anomaly generation, weather transitions,
time-of-day profiles, spatial correlation, infrastructure failures, and
domain randomization all behave as configured. Output goes to
evaluation/simulator_validation/.

Run from project root:
    python scripts/validate_simulator.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.config import (
    DomainRandomizationConfig,
    EnvironmentConfig,
    FailureConfig,
    ZoneConfig,
)
from sentinel.envs.constants import WEATHER_STATES, ZONE_TYPES
from sentinel.envs.urban import UrbanEnvironment

logging.basicConfig(level=logging.WARNING)

OUTPUT_DIR = Path("evaluation/simulator_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def sample_actions(env):
    """Random actions for all agents."""
    spaces = env.action_spaces()
    actions = {"global": {}, "zone": {}, "camera": {}}
    for level, agents in spaces.items():
        for aid, space in agents.items():
            actions[level][aid] = space.sample()
    return actions


def plot_anomaly_distribution():
    """Plot the anomaly score distribution across many steps.

    Expected: bimodal with a large peak near 0 (calm cameras) and a
    smaller peak near 0.7-0.8 (event cameras).
    """
    print("  Anomaly score distribution...")
    config = EnvironmentConfig()
    config.episode_length = 2000
    env = UrbanEnvironment(config)
    env.reset(seed=42)

    scores = []
    for _ in range(500):
        env.step(sample_actions(env))
        active = env._active
        scores.extend(env._anomaly[active].tolist())

    scores = np.array(scores)

    fig, ax = plt.subplots()
    ax.hist(scores, bins=80, density=True, alpha=0.7, color="#4c72b0")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution (500 steps, all cameras)")
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.5, label="Incident threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "anomaly_distribution.png")
    plt.close(fig)


def plot_time_of_day_profiles():
    """Plot anomaly base rates across 24 hours for each zone type.

    Each zone type should show a distinct daily pattern matching
    the configured TIME_PROFILES.
    """
    print("  Time-of-day profiles...")
    config = EnvironmentConfig()
    env = UrbanEnvironment(config)

    hours = np.linspace(0, 24, 96, endpoint=False)
    rates_by_zone = {zt: [] for zt in ZONE_TYPES}

    for hour in hours:
        env.reset(seed=42)
        env._time_of_day = hour / 24.0
        base_rates = env._compute_base_rates()
        for z, zcfg in enumerate(config.zones):
            n = int(env._cams_per_zone[z])
            if n > 0:
                rates_by_zone[zcfg.zone_type].append(
                    base_rates[z, :n].mean()
                )

    fig, ax = plt.subplots()
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    for (zt, rates), color in zip(rates_by_zone.items(), colors):
        ax.plot(hours, rates, label=zt.capitalize(), color=color, linewidth=2)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Anomaly Base Rate")
    ax.set_title("Anomaly Rate by Time of Day and Zone Type")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "time_of_day_profiles.png")
    plt.close(fig)


def plot_weather_transitions():
    """Run many steps and count weather state frequencies.

    Expected: frequencies should roughly match the stationary
    distribution of the Markov chain.
    """
    print("  Weather transition frequencies...")
    config = EnvironmentConfig()
    config.episode_length = 5000
    env = UrbanEnvironment(config)
    env.reset(seed=42)

    counts = np.zeros(3)
    transitions = np.zeros((3, 3))
    prev = env._weather
    for _ in range(3000):
        env.step(sample_actions(env))
        curr = env._weather
        counts[curr] += 1
        transitions[prev, curr] += 1
        prev = curr

    # Normalize
    freq = counts / counts.sum()
    trans_freq = transitions / transitions.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # State frequencies
    ax = axes[0]
    bars = ax.bar(WEATHER_STATES, freq, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylabel("Frequency")
    ax.set_title("Weather State Frequencies (3000 steps)")
    for bar, f in zip(bars, freq):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{f:.3f}", ha="center", fontsize=10,
        )

    # Transition matrix
    ax = axes[1]
    im = ax.imshow(trans_freq, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(WEATHER_STATES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(WEATHER_STATES)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Observed Transition Probabilities")
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{trans_freq[i, j]:.2f}",
                ha="center", va="center", fontsize=11,
                color="white" if trans_freq[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weather_transitions.png")
    plt.close(fig)


def plot_spatial_correlation():
    """Show how anomaly scores propagate to neighbors after an incident.

    Force a high anomaly on one camera and track neighbor scores
    over subsequent steps.
    """
    print("  Spatial correlation propagation...")
    config = EnvironmentConfig()
    config.episode_length = 200
    # No failures to keep it clean
    config.failures = FailureConfig(
        camera_dropout_prob=0.0,
        gpu_failure_prob=0.0,
        network_congestion_prob=0.0,
    )
    env = UrbanEnvironment(config)
    env.reset(seed=42)

    target_cam = 7  # inject anomaly here
    n_steps = 30
    boosts = {d: [] for d in range(-4, 5)}

    for step in range(n_steps):
        if step == 5:
            # Inject a strong anomaly on step 5
            env._anomaly[0, target_cam] = 0.95
            env._update_spatial_boost()

        env.step(sample_actions(env))

        for d in boosts:
            c = target_cam + d
            if 0 <= c < int(env._cams_per_zone[0]):
                boosts[d].append(float(env._spatial_boost[0, c]))
            else:
                boosts[d].append(0.0)

    fig, ax = plt.subplots()
    for d in [-2, -1, 0, 1, 2]:
        label = f"cam {target_cam + d} (d={d:+d})"
        ax.plot(range(n_steps), boosts[d], label=label, linewidth=1.5)

    ax.axvline(5, color="red", linestyle="--", alpha=0.5, label="Incident injected")
    ax.set_xlabel("Step")
    ax.set_ylabel("Spatial Boost")
    ax.set_title(f"Spatial Correlation: Boost Propagation from Camera {target_cam}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_correlation.png")
    plt.close(fig)


def plot_failure_rates():
    """Run with high failure rates and plot failure occurrences over time."""
    print("  Infrastructure failure rates...")
    config = EnvironmentConfig()
    config.episode_length = 300
    config.failures = FailureConfig(
        camera_dropout_prob=0.02,
        camera_dropout_duration=15,
        gpu_failure_prob=0.01,
        gpu_failure_duration=30,
        gpu_failure_capacity_loss=0.3,
        network_congestion_prob=0.01,
        network_congestion_duration=20,
        network_congestion_bandwidth_loss=0.4,
    )
    env = UrbanEnvironment(config)
    env.reset(seed=42)

    steps = []
    dropout_counts = []
    gpu_fail_zones = []
    net_cong_zones = []
    active_counts = []

    for t in range(300):
        env.step(sample_actions(env))
        steps.append(t)
        dropout_counts.append(int((env._camera_dropout > 0).sum()))
        gpu_fail_zones.append(int((env._gpu_failure_remaining > 0).sum()))
        net_cong_zones.append(int((env._net_congestion_remaining > 0).sum()))
        active_counts.append(env.num_cameras)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(steps, dropout_counts, label="Cameras offline", color="#c44e52")
    ax.plot(steps, active_counts, label="Active cameras", color="#4c72b0")
    ax.set_ylabel("Count")
    ax.set_title("Camera Availability Over Time")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, gpu_fail_zones, label="Zones with GPU failure", color="#dd8452")
    ax.plot(
        steps, net_cong_zones,
        label="Zones with network congestion", color="#8172b2",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Zone Count")
    ax.set_title("Infrastructure Failures Over Time")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_rates.png")
    plt.close(fig)


def plot_domain_randomization():
    """Show distributions of camera counts and budgets across many resets."""
    print("  Domain randomization distributions...")
    config = EnvironmentConfig()
    config.domain_randomization = DomainRandomizationConfig(
        enabled=True,
        camera_count_range=[10, 55],
        gpu_budget_range=[50.0, 120.0],
        bandwidth_budget_range=[50.0, 120.0],
    )
    env = UrbanEnvironment(config)

    cam_counts = []
    gpu_budgets = []
    for seed in range(200):
        env.reset(seed=seed)
        cam_counts.append(env.num_cameras)
        gpu_budgets.append(config.resources.total_gpu_budget)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(cam_counts, bins=20, color="#4c72b0", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Total Active Cameras")
    ax.set_ylabel("Count (out of 200 resets)")
    ax.set_title("Camera Count Distribution (DR enabled)")

    ax = axes[1]
    ax.hist(gpu_budgets, bins=20, color="#55a868", alpha=0.7, edgecolor="black")
    ax.set_xlabel("GPU Budget")
    ax.set_ylabel("Count (out of 200 resets)")
    ax.set_title("GPU Budget Distribution (DR enabled)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "domain_randomization.png")
    plt.close(fig)


def plot_weather_effects_on_confidence():
    """Show how weather state affects detection confidence."""
    print("  Weather effects on confidence...")
    config = EnvironmentConfig()
    env = UrbanEnvironment(config)

    weather_confs = {}
    for w_idx, w_name in enumerate(WEATHER_STATES):
        confs = []
        for _ in range(200):
            env.reset(seed=None)
            env._weather = w_idx
            # Set cameras to medium quality so confidence varies meaningfully
            env._res[:] = 1
            env._fps[:] = 1
            env._model[:] = 1
            env._update_confidence()
            confs.extend(env._confidence[env._active].tolist())
        weather_confs[w_name] = np.array(confs)

    fig, ax = plt.subplots()
    positions = range(len(WEATHER_STATES))
    data = [weather_confs[w] for w in WEATHER_STATES]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([w.capitalize() for w in WEATHER_STATES])
    ax.set_ylabel("Detection Confidence")
    ax.set_title("Detection Confidence by Weather State (medium quality)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weather_confidence.png")
    plt.close(fig)


def plot_hotspot_effect():
    """Show how a hotspot elevates anomaly scores near its center."""
    print("  Hotspot effect on anomaly rates...")
    config = EnvironmentConfig()
    config.episode_length = 200
    config.failures = FailureConfig(
        camera_dropout_prob=0.0,
        gpu_failure_prob=0.0,
        network_congestion_prob=0.0,
    )
    env = UrbanEnvironment(config)
    env.reset(seed=42)

    # Inject a hotspot at camera 7 in zone 0
    env._hotspots = [{
        "zone": 0,
        "center": 7,
        "intensity": 3.0,
        "direction": 1,
        "remaining": 50,
    }]

    # Collect mean anomaly per camera over several steps
    n_cams = int(env._cams_per_zone[0])
    cam_means = np.zeros(n_cams)
    n_samples = 100

    for _ in range(n_samples):
        env._generate_anomalies()
        cam_means += env._anomaly[0, :n_cams]

    cam_means /= n_samples

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#c44e52" if abs(i - 7) <= 2 else "#4c72b0" for i in range(n_cams)]
    ax.bar(range(n_cams), cam_means, color=colors, alpha=0.8, edgecolor="black")
    ax.axvline(7, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Camera Index")
    ax.set_ylabel("Mean Anomaly Score")
    ax.set_title("Hotspot Effect: Elevated Anomaly Near Center (cam 7)")
    ax.annotate(
        "Hotspot center", xy=(7, cam_means[7]),
        xytext=(10, cam_means[7] + 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10, color="red",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hotspot_effect.png")
    plt.close(fig)


def main():
    print("Generating simulator validation plots...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    plot_anomaly_distribution()
    plot_time_of_day_profiles()
    plot_weather_transitions()
    plot_spatial_correlation()
    plot_failure_rates()
    plot_domain_randomization()
    plot_weather_effects_on_confidence()
    plot_hotspot_effect()

    plots = list(OUTPUT_DIR.glob("*.png"))
    print(f"\nDone. Generated {len(plots)} plots:")
    for p in sorted(plots):
        print(f"  {p}")


if __name__ == "__main__":
    main()