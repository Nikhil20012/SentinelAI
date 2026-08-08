<div align="center">

# 🤖 SentinelAI

**Hierarchical multi-agent RL for adaptive compute allocation in distributed vision systems**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Multi--Agent_Env-4B8BBE?style=flat-square&logo=python&logoColor=white)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Simulation-0081A7?style=flat-square&logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![Ray RLlib](https://img.shields.io/badge/Ray_RLlib-MAPPO-028CF0?style=flat-square&logo=ray&logoColor=white)](https://docs.ray.io/en/latest/rllib/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Experiments-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Redis](https://img.shields.io/badge/Redis-Streams-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
[![gRPC](https://img.shields.io/badge/gRPC-Inference_Serving-244c5a?style=flat-square&logo=google&logoColor=white)](https://grpc.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B5CF6?style=flat-square)](https://shap.readthedocs.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=flat-square&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=flat-square&logo=grafana&logoColor=white)](https://grafana.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

`IN PROGRESS`

</div>

---

Surveillance camera networks waste compute on idle cameras while under-serving
cameras that need it during incidents. SentinelAI uses a three-level RL
hierarchy to dynamically allocate resolution, frame rate, and model complexity
across a distributed camera network based on real-time activity, weather
conditions, and resource constraints.

## Problem

A city operating a distributed camera network allocates fixed compute to every
camera regardless of what is happening in its field of view. This means:

1. **Wasted resources.** Cameras watching empty parking lots at 2 AM run at the
   same quality as cameras covering an active incident. GPU and bandwidth are
   burned on frames that contain nothing useful.

2. **Missed detections.** During multi-zone incidents the compute budget cannot
   shift to where it is needed. Cameras that should be running at high
   resolution stay at baseline quality because the budget is spread uniformly.

3. **No coordination.** When neighboring cameras detect related activity, there
   is no mechanism to jointly escalate. Each camera operates in isolation.

## Approach

A three-level RL hierarchy decomposes the allocation problem so that each level
handles a different scope of decision-making:

1. **Global controller** observes system-wide activity and allocates GPU and
   bandwidth budgets to each zone.
2. **Zone controllers** observe all cameras within their zone and assign
   resource-access tiers (minimal, normal, elevated, priority) to each camera.
3. **Camera agents** observe local conditions and choose resolution, FPS, and
   model tier within the constraints of their assigned tier.

All three levels are trained with MAPPO (centralized training, decentralized
execution) on a custom PettingZoo environment with time-varying anomaly
patterns, weather effects, spatial correlation, and infrastructure failures.

## Architecture

```
                  Digital Twin Simulation (PettingZoo ParallelEnv)
     Time-of-day / Weather / Spatial anomalies / Infrastructure failures
                                    |
                                    v
                    +-------------------------------+
                    |      Global Controller        |
                    |   Allocates zone budgets       |
                    |   (discrete budget levels)     |
                    +-------+---+---+---+-----------+
                            |   |   |   |
               Zone A ------+   |   |   +------ Zone D
               Zone B ----------+   +---------- Zone C
                            |           |
                            v           v
                    +---------------+  +---------------+
                    | Zone Ctrl     |  | Zone Ctrl     |
                    | Assigns tiers |  | Assigns tiers |
                    | per camera    |  | per camera    |
                    +-------+-------+  +-------+-------+
                            |                  |
              +------+------+------+     +-----+-----+
              |      |      |      |     |     |     |
              v      v      v      v     v     v     v
            Cam 1  Cam 2  Cam 3  Cam 4  ...  Cam N-1 Cam N
            (res, fps, model tier within assigned tier)
                                    |
                                    v
                    +-------------------------------+
                    |    Inference + Streaming       |
                    |    Redis Streams / gRPC        |
                    +-------+-----------+-----------+
                            |           |
               +------------+     +-----+--------+
               |                  |              |
               v                  v              v
         Explainability     Prometheus       React Dashboard
         (SHAP, distill,   + Grafana        + Digital Twin
          counterfactual,                      Playback
          decision replay)

              W&B tracks all training experiments
```

## Tech stack

| Tool | Role |
|---|---|
| PettingZoo + Gymnasium | Multi-agent simulation environment |
| Ray RLlib | Hierarchical MAPPO training (centralized training, decentralized execution) |
| Weights & Biases | Experiment tracking, hyperparameter sweeps, training curves |
| Redis Streams | Low-latency state buffering and action streaming |
| gRPC | Inference serving for trained policies |
| SHAP + scikit-learn | Decision explainability (selective SHAP, policy distillation) |
| Prometheus + Grafana | Production monitoring, alerting, RL system health |
| FastAPI | Explainability and status API endpoints |
| React | Operations dashboard with digital twin playback |
| Docker Compose | Full-stack containerization |

## Current status

Phase 1: Simulator (complete)

- [x] Project structure and configuration system
- [x] Base interfaces (BaseEnvironment, BasePolicy, BaseReward)
- [x] YAML-driven environment configs
- [x] Config loading and validation tests
- [x] Urban environment (PettingZoo ParallelEnv)
- [x] Anomaly generation (Poisson, spatial correlation, moving hotspots)
- [x] Resource budgets and action masking
- [x] Weather state machine
- [x] Infrastructure failure model
- [x] Domain randomization
- [x] Scenario presets
- [x] Simulator validation plots

## Build roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Simulator | In progress |
| 2 | RL training + baselines + evaluation | Not started |
| 3 | Explainability | Not started |
| - | Research checkpoint (go/no-go) | - |
| 4 | Infrastructure (Redis, gRPC, monitoring) | Not started |
| 5 | Dashboard + polish + report | Not started |

## Project structure

```
sentinel/                Core library
  envs/                  Simulation environments
  policies/              RL agents and heuristic baselines
  rewards/               Reward functions
  config.py              Typed configuration dataclasses
configs/                 YAML experiment and environment configs
training/                Training and evaluation entry points
tests/                   Unit and integration tests
docs/                    Architecture docs and development log
scripts/                 Setup and utility scripts
```

Additional directories (serving, streaming, api, dashboard, docker, evaluation,
paper) will be added as their respective phases begin.

## Setup

```bash
git clone https://github.com/Nikhil20012/SentinelAI.git
cd SentinelAI
bash scripts/setup.sh
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Author

**Nikhil Bharadwaj Yellapragada**
<br>
MS Data Analytics Engineering, Northeastern University

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikhil-bharadwaj-yellapragada-48321a211/)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:yellapragada.n@northeastern.edu)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Nikhil20012)

## License

MIT. See [LICENSE](LICENSE) for details.
