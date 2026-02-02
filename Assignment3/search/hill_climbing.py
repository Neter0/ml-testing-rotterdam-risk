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
import time
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
    
    # Enhanced fitness with time-to-collision (TTC)
    min_ttc = float("inf")  # Time-to-collision metric
    same_lane_min_distance = float("inf")  # Track same-lane proximity separately
    min_closing_distance = float("inf")  # Distance only when vehicles are actively approaching

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
        ego_speed = float(ego.get("speed", 0.0))
        ego_lane = ego.get("lane_id", -1)

        for car in step.get("others", []):
            pos2 = car.get("pos")
            if pos2 is None or len(pos2) < 2:
                continue

            x2, y2 = float(pos2[0]), float(pos2[1])
            distance = float(np.hypot(x2 - x, y2 - y))

            if distance < min_distance:
                min_distance = distance

            # Calculate TTC and same-lane distance
            car_lane = car.get("lane_id", -1)
            if ego_lane == car_lane and ego_lane != -1:
                if distance < same_lane_min_distance:
                    same_lane_min_distance = distance

            # TTC and closing distance — only meaningful when vehicles are approaching
            car_speed = float(car.get("speed", 0.0))
            dx = x2 - x  # positive if other car is ahead
            relative_velocity_x = ego_speed - car_speed  # positive if ego is faster

            # Approaching = ego is faster AND other car is ahead, OR ego is slower AND other car is behind
            approaching = (dx > 0 and relative_velocity_x > 0.1) or (dx < 0 and relative_velocity_x < -0.1)

            if approaching and distance < 100:
                closing_speed = abs(relative_velocity_x)
                ttc = distance / closing_speed
                if ttc < min_ttc:
                    min_ttc = ttc
                if distance < min_closing_distance:
                    min_closing_distance = distance

    return {
        "crash_count": int(crash_count),
        "min_distance": float(min_distance),
        "min_ttc": float(min_ttc),
        "same_lane_min_distance": float(same_lane_min_distance),
        "min_closing_distance": float(min_closing_distance)
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
    same_lane_dist = float(objectives.get("same_lane_min_distance", float("inf")))
    closing_dist = float(objectives.get("min_closing_distance", float("inf")))

    if crash > 0:
        return -1.0

    # Weighted combination of distance metrics
    w_general = 0.3
    w_same_lane = 0.4
    w_closing = 0.3

    fitness = w_general * min_distance

    if same_lane_dist < float("inf"):
        fitness += w_same_lane * same_lane_dist
    else:
        fitness += w_same_lane * min_distance

    if closing_dist < float("inf"):
        fitness += w_closing * closing_dist
    else:
        fitness += w_closing * min_distance

    return float(fitness)


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
    """
    new_config = copy.deepcopy(cfg)

    # Multi-parameter mutation with probability
    if rng.random() < 0.7:
        # Single parameter mutation
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
    else:
        # Multi-parameter mutation
        num_params = int(rng.integers(2, 4))
        all_methods = ["vehicles_count", "lanes_count", "initial_spacing", "ego_spacing"]
        selected_methods = rng.choice(all_methods, size=num_params, replace=False)
        
        for method in selected_methods:
            if method == "vehicles_count":
                mutate_vehicle_count(new_config, param_spec, rng)
            elif method == "lanes_count":
                mutate_lanes_count(new_config, param_spec, rng)
            elif method == "initial_spacing":
                mutate_spacing(new_config, param_spec, rng)
            elif method == "ego_spacing":
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
    """
    rng = np.random.default_rng(seed)

    # Track runtime and search statistics
    start_time = time.time()
    stats = {
        "acceptances": 0,
        "rejections": 0,
        "restarts": 0,
        "crashes_found": 0,
        "crash_iteration": None,
        "crash_scenarios": []
    }

    # Track all fitness values for comparison
    all_fitness_values = []
    first_crash_evaluation = None

    # Start from random initial configuration
    current_cfg = sample_random_config(base_cfg, param_spec, rng)

    # Evaluate initial solution
    seed_base = int(rng.integers(0, 1_000_000_000))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    if int(obj.get("crash_count", 0)) == 0 and crashed:
        obj["crash_count"] = 1

    cur_fit = compute_fitness(obj)
    all_fitness_values.append(cur_fit)

    print("\n")
    print(f"Initial fitness: {cur_fit}")

    # Track best solution found
    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_ts = ts

    # current fit tracking
    current_fit = float(cur_fit)
    evaluations = 1
    history = [best_fit]

    # Best non-crash tracking
    best_non_crash = {
        "config": None,
        "objectives": None,
        "fitness": float('inf'),
        "seed": None
    }

    if int(obj.get("crash_count", 0)) > 0 or crashed:
        print("Initial configuration crashed!")
        stats["crashes_found"] = 1
        stats["crash_iteration"] = 0
        stats["crash_scenarios"].append({
            "config": copy.deepcopy(current_cfg), 
            "objectives": dict(obj), 
            "seed": seed_base,
            "evaluation_number": evaluations
        })
        first_crash_evaluation = evaluations
        end_time = time.time()
        
        return generate_return_dict(
            best_cfg, best_obj, best_fit, best_seed_base, best_ts,
            history, evaluations, stats, start_time, end_time,
            all_fitness_values, first_crash_evaluation, best_non_crash
        )

    # Random restart logic
    no_improvement_count = 0
    max_no_improvement = 15

    for i in range(int(iterations)):
        print(f"Running iteration {i+1}/{iterations}")

        if no_improvement_count >= max_no_improvement:
            print(f"   → No improvement for {max_no_improvement} iterations. Random restart!")
            stats["restarts"] += 1
            current_cfg = sample_random_config(base_cfg, param_spec, rng)
            seed_base = int(rng.integers(0, 1_000_000_000))
            crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
            obj = compute_objectives_from_time_series(ts)
            if int(obj.get("crash_count", 0)) == 0 and crashed:
                obj["crash_count"] = 1
            current_fit = compute_fitness(obj)
            all_fitness_values.append(current_fit)
            evaluations += 1
            no_improvement_count = 0

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
                x_obj["crash_count"] = 1

            x_fit = float(compute_fitness(x_obj))
            all_fitness_values.append(x_fit)
            evaluations += 1

            if x_fit < best_neighbor_fit:
                best_neighbor_fit = x_fit
                best_neighbor_cfg = x_cfg
                best_neighbor_obj = x_obj
                best_neighbor_seed = x_seed_base
                best_neighbor_ts = x_ts

            if int(x_obj.get("crash_count", 0)) > 0 or x_crashed:
                print(f"   → Found crash with fitness: {x_fit}")
                x_obj["crash_count"] = 1
                stats["crashes_found"] += 1
                if stats["crash_iteration"] is None:
                    stats["crash_iteration"] = i + 1
                if first_crash_evaluation is None:
                    first_crash_evaluation = evaluations
                stats["crash_scenarios"].append({
                    "config": copy.deepcopy(x_cfg), 
                    "objectives": dict(x_obj), 
                    "seed": x_seed_base,
                    "evaluation_number": evaluations
                })
                end_time = time.time()
                
                return generate_return_dict(
                    copy.deepcopy(x_cfg), dict(x_obj), float(x_fit), int(x_seed_base), x_ts,
                    history + [float(x_fit)], evaluations, stats, start_time, end_time,
                    all_fitness_values, first_crash_evaluation, best_non_crash
                )

        # Track best non-crash
        if best_neighbor_obj and int(best_neighbor_obj.get("crash_count", 0)) == 0:
            if best_neighbor_fit < best_non_crash["fitness"]:
                best_non_crash = {
                    "config": copy.deepcopy(best_neighbor_cfg),
                    "objectives": dict(best_neighbor_obj),
                    "fitness": best_neighbor_fit,
                    "seed": best_neighbor_seed
                }

        # Accept neighbor and update current state
        if best_neighbor_cfg is not None and best_neighbor_fit < current_fit:
            current_cfg = copy.deepcopy(best_neighbor_cfg)
            current_fit = float(best_neighbor_fit)
            no_improvement_count = 0
            stats["acceptances"] += 1
            print(f"   → Accepted neighbor, new fitness: {current_fit}")

            # Update global best if this is the best so far
            if best_neighbor_fit < best_fit and best_neighbor_obj is not None:
                best_cfg = copy.deepcopy(best_neighbor_cfg)
                best_obj = dict(best_neighbor_obj)
                best_fit = float(best_neighbor_fit)
                best_seed_base = int(best_neighbor_seed) if best_neighbor_seed is not None else best_seed_base
                best_ts = best_neighbor_ts
                print(f"   → NEW GLOBAL BEST: {best_fit}")
        else:
            no_improvement_count += 1
            stats["rejections"] += 1
            print(f"   → No improvement (count: {no_improvement_count})")

        history.append(best_fit)

    print(f"\nFinal best fitness: {best_fit}")
    end_time = time.time()
    
    return generate_return_dict(
        best_cfg, best_obj, best_fit, best_seed_base, best_ts,
        history, evaluations, stats, start_time, end_time,
        all_fitness_values, first_crash_evaluation, best_non_crash
    )


def generate_return_dict(best_cfg, best_obj, best_fit, best_seed_base, best_ts,
                        history, evaluations, stats, start_time, end_time,
                        all_fitness_values, first_crash_evaluation, best_non_crash):
    """Generate standardized return dictionary with all comparison metrics."""
    runtime = end_time - start_time
    
    print_metrics_summary(stats, evaluations, start_time, end_time, best_obj, best_cfg)
    
    return {
        # Failure Discovery
        "collision_found": stats["crashes_found"] > 0,
        "num_crashes": stats["crashes_found"],
        "min_distance_achieved": best_obj.get("min_distance", float('inf')),
        
        # Best scenario
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed": best_seed_base,
        "best_time_series": best_ts,
        
        # All crashes
        "crash_scenarios": stats["crash_scenarios"],
        "best_non_crash": best_non_crash if stats["crashes_found"] == 0 else None,
        
        # Efficiency
        "runtime_seconds": runtime,
        "total_evaluations": evaluations,
        "evaluations_to_first_crash": first_crash_evaluation,
        "evaluations_per_second": evaluations / runtime if runtime > 0 else 0,
        
        # Convergence
        "best_fitness_history": history,
        "all_fitness_values": all_fitness_values,
        
        # Method-specific
        "method": "Hill Climbing",
        "method_stats": {
            "acceptances": stats["acceptances"],
            "rejections": stats["rejections"],
            "restarts": stats["restarts"],
            "crash_iteration": stats["crash_iteration"]
        },
        
        # Legacy compatibility
        "stats": stats,
        "history": history,
        "evaluations": evaluations,
        "best_seed_base": best_seed_base
    }


def print_metrics_summary(stats, evaluations, start_time, end_time, best_obj, best_cfg):
    """Print comprehensive search metrics."""
    runtime = end_time - start_time
    print("\n" + "="*80)
    print("HILL CLIMBING METRICS SUMMARY")
    print("="*80)

    # Failure discovery
    print("\n[1] FAILURE DISCOVERY:")
    print(f"   • Collision found: {'YES' if stats['crashes_found'] > 0 else 'NO'}")
    print(f"   • Number of crashes: {stats['crashes_found']}")
    if stats['crash_iteration'] is not None:
        print(f"   • First crash at iteration: {stats['crash_iteration']}")
    print(f"   • Minimum distance achieved: {best_obj.get('min_distance', float('inf')):.2f}m")

    # Scenario characteristics
    print("\n[2] SCENARIO CHARACTERISTICS:")
    if stats['crash_scenarios']:
        print(f"   • Critical scenario configuration:")
        cfg = stats['crash_scenarios'][0]['config']
        print(f"     - vehicles_count: {cfg.get('vehicles_count')}")
        print(f"     - lanes_count: {cfg.get('lanes_count')}")
        print(f"     - initial_spacing: {cfg.get('initial_spacing'):.2f}")
        print(f"     - ego_spacing: {cfg.get('ego_spacing'):.2f}")
        print(f"     - initial_lane_id: {cfg.get('initial_lane_id')}")
        print(f"   • Reproducibility seed: {stats['crash_scenarios'][0]['seed']}")
    else:
        print(f"   • Best scenario configuration:")
        print(f"     - vehicles_count: {best_cfg.get('vehicles_count')}")
        print(f"     - lanes_count: {best_cfg.get('lanes_count')}")
        print(f"     - initial_spacing: {best_cfg.get('initial_spacing'):.2f}")
        print(f"     - ego_spacing: {best_cfg.get('ego_spacing'):.2f}")
        print(f"     - initial_lane_id: {best_cfg.get('initial_lane_id')}")

    # Efficiency
    print("\n[3] EFFICIENCY:")
    print(f"   • Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print(f"   • Total scenario evaluations: {evaluations}")
    print(f"   • Evaluations per second: {evaluations/runtime:.2f}")
    if stats['crash_iteration'] is not None:
        evals_to_crash = sum([1 + i * 10 for i in range(stats['crash_iteration'])])
        print(f"   • Evaluations to reach crash: ~{evals_to_crash}")

    # Search effectiveness
    total_decisions = stats['acceptances'] + stats['rejections']
    if total_decisions > 0:
        acceptance_rate = 100 * stats['acceptances'] / total_decisions
        print("\n[4] SEARCH EFFECTIVENESS:")
        print(f"   • Acceptance rate: {acceptance_rate:.1f}% ({stats['acceptances']}/{total_decisions})")
        print(f"   • Rejections: {stats['rejections']}")
        print(f"   • Random restarts: {stats['restarts']}")

    # Objective breakdown
    print("\n[5] OBJECTIVE METRICS:")
    print(f"   • Min distance: {best_obj.get('min_distance', float('inf')):.2f}m")
    print(f"   • Same-lane min distance: {best_obj.get('same_lane_min_distance', float('inf')):.2f}m")
    print(f"   • Min closing distance: {best_obj.get('min_closing_distance', float('inf')):.2f}m")
    print(f"   • Min TTC: {best_obj.get('min_ttc', float('inf')):.2f} timesteps")

    print("="*80 + "\n")


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
