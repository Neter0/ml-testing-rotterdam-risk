"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    crash_count = 0
    min_distance = float("inf")

    for step in time_series:

        if step.get("crashed", False):
            crash_count = 1

        ego = step.get("ego")
        if ego is None:
            continue

        pos = ego.get("pos")
        if pos is None or len(pos) < 2:
            continue
        x, y = float(pos[0]), float(pos[1])

        for car in step.get("others", []):
            pos2 = car.get("pos")
            if pos2 is None or len(pos2) < 2:
                continue
            x2, y2 = float(pos2[0]), float(pos2[1])

            distance = float(np.hypot(x2 - x, y2 - y))
            if distance < min_distance:
                min_distance = distance

    return {
        "crash_count": int(crash_count),
        "min_distance": float(min_distance)
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    crash = int(objectives.get("crash_count", 0))
    min_distance = float(objectives.get("min_distance", float("inf")))

    if crash > 0:
        return -1.0

    return min_distance


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    new_config = copy.deepcopy(cfg)

    methods = [
        "vehicles_count",
        "lanes_count",
        "initial_spacing",
        "ego_spacing"
    ]

    selected = rng.choice(methods)

    if selected == "vehicles_count":
        mutate_vehicle_count(new_config, param_spec, rng)
    elif selected == "lanes_count":
        mutate_lanes_count(new_config, param_spec, rng)
    elif selected == "initial_spacing":
        mutate_spacing(new_config, param_spec, rng)
    elif selected == "ego_spacing":
        mutate_ego_spacing(new_config, param_spec, rng)

    clamp_params(new_config, param_spec)
    clamp_lane_id(new_config)

    return new_config

def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x

def clamp_params(config, param_spec):
    for key, value in param_spec.items():
        if key not in config:
            continue
        low, high = value["min"], value["max"]
        if value["type"] == "int":
            config[key] = int(clamp(float(config[key]), float(low), float(high)))
        elif value["type"] == "float":
            config[key] = float(clamp(float(config[key]), float(low), float(high)))

def clamp_lane_id(config):
    x = int(config["lanes_count"])
    x = max(1, x)
    initial = float(config["initial_lane_id"])
    config["initial_lane_id"] = int(clamp(initial, 0.0, float(x - 1)))
    return

def mutate_vehicle_count(config, param_spec, rng):
    param_vehicles_count = param_spec["vehicles_count"]
    low, high = int(param_vehicles_count["min"]), int(param_vehicles_count["max"])
    current = int(config.get("vehicles_count", low))

    change = int(rng.integers(1, 5))
    direction = 1
    if rng.random() < 0.5:
        direction = -1

    new_count = float(current + direction * change)
    config["vehicles_count"] = int(clamp(new_count, float(low), float(high)))

def mutate_lanes_count(config, param_spec, rng):
    param_lanes_count = param_spec["lanes_count"]
    low, high = int(param_lanes_count["min"]), int(param_lanes_count["max"])
    current = int(config.get("lanes_count", low))

    change = int(rng.integers(1, 5))
    direction = 1
    if rng.random() < 0.5:
        direction = -1

    new_count = float(current + direction * change)
    config["lanes_count"] = int(clamp(new_count, float(low), float(high)))
    clamp_lane_id(config)

def mutate_spacing(config, param_spec, rng):
    param_initial_spacing = param_spec["initial_spacing"]
    low, high = float(param_initial_spacing["min"]), float(param_initial_spacing["max"])
    current = float(config.get("initial_spacing", low))

    x = float(rng.random() * 0.5 + 0.75)
    new_spacing = float(current * x)
    config["initial_spacing"] = float(clamp(new_spacing, low, high))

def mutate_ego_spacing(config, param_spec, rng):
    param_ego_spacing = param_spec["ego_spacing"]
    low, high = float(param_ego_spacing["min"]), float(param_ego_spacing["max"])
    current = float(config.get("ego_spacing", low))

    x = float(rng.random() * 0.5)
    if rng.random() < 0.5:
        x = -x

    new_spacing = float(current + x)
    config["ego_spacing"] = float(clamp(new_spacing, low, high))



# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    current_cfg = sample_random_config(base_cfg, param_spec, rng)
    # current_cfg = dict(base_cfg)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    if int(obj.get("crash_count", 0)) == 0 and crashed:
        obj["count_crashed"] = 1
    cur_fit = compute_fitness(obj)
    print("\n")
    print(f"Initial fitness: {cur_fit}")

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_ts = ts
    evaluations = 1

    history = [best_fit]

    if int(obj.get("crash_count", 0)) > 0 or crashed:
        print("Initial configuration crashed!")
        return {
            "best_cfg": best_cfg,
            "best_objectives": best_obj,
            "best_fitness": best_fit,
            "best_seed_base": best_seed_base,
            "best_time_series": best_ts,
            "history": history,
            "evaluations": evaluations,
        }

    for i in range(int(iterations)):
        print(f"Running iteration {i+1}/{iterations}")
        best_neighbor_cfg = None
        best_neighbor_obj = None
        best_neighbor_fit = float("inf")
        best_neighbor_seed = None
        best_neighbor_ts = None

        for j in range(int(neighbors_per_iter)):
            x_cfg = mutate_config(current_cfg, param_spec, rng)
            x_seed_base = int(rng.integers(1_000_000))
            x_crashed, x_ts = run_episode(env_id, x_cfg, policy, defaults, x_seed_base)
            x_obj = compute_objectives_from_time_series(x_ts)
            if int(x_obj.get("crash_count", 0)) == 0 and x_crashed:
                x_obj["count_crashed"] = 1
            x_fit = float(compute_fitness(x_obj))
            evaluations += 1

            if x_fit < best_neighbor_fit:
                best_neighbor_fit = x_fit
                best_neighbor_cfg = x_cfg
                best_neighbor_obj = x_obj
                best_neighbor_seed = x_seed_base
                best_neighbor_ts = x_ts

            if int(x_obj.get("crash_count", 0)) > 0 or x_crashed:
                print(f"Crash fitness: {x_fit}")
                x_obj["crash_count"] = 1
                return {
                    "best_cfg": copy.deepcopy(x_cfg),
                    "best_objectives": dict(x_obj),
                    "best_fitness": float(x_fit),
                    "best_seed_base": int(x_seed_base),
                    "best_time_series": x_ts,
                    "history": history + [float(x_fit)],
                    "evaluations": evaluations,
                }

        if best_neighbor_fit < best_fit:
            best_cfg = copy.deepcopy(best_neighbor_cfg)
            best_obj = dict(best_neighbor_obj)
            best_fit = float(best_neighbor_fit)
            best_seed_base = int(best_neighbor_seed)
            best_ts = best_neighbor_ts

        history.append(best_fit)
        print(f"Best fitness: {best_fit}")

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "best_time_series": best_ts,
        "history": history,
        "evaluations": evaluations,
    }

def sample_random_config(base_cfg, param_spec, rng):
    cfg = {}
    lanes = None
    if "lanes_count" in param_spec:
        s = param_spec["lanes_count"]
        lanes = int(rng.integers(s["min"], s["max"] + 1))
        cfg["lanes_count"] = lanes

    for k, s in param_spec.items():
        if k == "lanes_count":
            continue
        if k == "initial_lane_id":
            lanes = lanes or 3
            cfg[k] = int(rng.integers(0, lanes))
            continue
        if s["type"] == "int":
            cfg[k] = int(rng.integers(s["min"], s["max"] + 1))
        elif s["type"] == "float":
            cfg[k] = float(rng.uniform(s["min"], s["max"]))

    for k, v in base_cfg.items():
        if k not in cfg:
            cfg[k] = v
    return cfg