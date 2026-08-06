# Development Log

Running record of implementation decisions, debugging notes, and observations.


## Project Setup

**Date**: 2026-08-04

Initialized project structure. Starting with Phase 1 (simulator).

Architecture is frozen per the final architecture document. The research question:
can hierarchical multi-agent RL improve adaptive compute allocation in distributed
vision systems under constrained resources?

Key decisions locked in:
- Three-level hierarchy: global (budget allocation), zone (tier assignment), camera (quality selection)
- PettingZoo ParallelEnv with fixed max-N cameras and active/inactive masking
- Fixed number of zones across episodes; domain randomization over everything else
- Research-first build order with go/no-go gate at week 6
- Experiment 2 (hierarchy depth comparison) is the critical milestone, target week 4

Starting with the environment since everything downstream depends on it.
