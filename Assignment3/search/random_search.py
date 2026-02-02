from tqdm import trange
import numpy as np
import copy
import time
from envs.highway_env_utils import record_video_episode, run_episode

# Import from hill_climbing to reuse objective computation
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hill_climbing import compute_objectives_from_time_series, compute_fitness

class RandomSearch:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, n_scenarios=50, n_eval=1, seed=42):
        print(f"Running Random Search for {n_scenarios} scenarios...")
        rng = np.random.default_rng(seed)
        
        # Tracking for comparison metrics
        start_time = time.time()
        crash_log = []
        all_fitness_values = []
        best_fitness_history = []
        current_best_fitness = float('inf')
        first_crash_evaluation = None
        total_evaluations = 0
        
        # Track ALL evaluated scenarios for pattern analysis (Section 2.4)
        all_scenarios_evaluated = []
        
        # Statistics tracking (comparable to HC)
        stats = {
            "improvements_found": 0,  # Equivalent to HC's acceptances
            "no_improvements": 0,      # Equivalent to HC's rejections
            "crashes_found": 0,
            "crash_scenario_number": None
        }
        
        # Best non-crash scenario tracking
        best_non_crash = {
            "config": None,
            "objectives": None,
            "fitness": float('inf'),
            "seed": None
        }
        
        # Best overall tracking
        best_cfg = None
        best_objectives = None
        best_fitness = float('inf')
        best_seed = None
        
        print("\n")
        
        for i in range(n_scenarios):
            print(f"[Scenario {i+1}/{n_scenarios}] Evaluating random configuration...")
            cfg = self.sample_random_config(rng)
            
            # Print configuration being tested
            print(f"  → Config: vehicles={cfg.get('vehicles_count')}, lanes={cfg.get('lanes_count')}, "
                  f"spacing={cfg.get('initial_spacing'):.2f}, ego_spacing={cfg.get('ego_spacing'):.2f}, "
                  f"lane_id={cfg.get('initial_lane_id')}")
            
            for j in range(n_eval):
                s = int(rng.integers(1e9))
                crashed, ts = run_episode(self.env_id, cfg, self.policy, self.defaults, s)
                
                # Compute objectives and fitness for comparison
                obj = compute_objectives_from_time_series(ts)
                if int(obj.get("crash_count", 0)) == 0 and crashed:
                    obj["crash_count"] = 1
                
                fit = compute_fitness(obj)
                total_evaluations += 1
                all_fitness_values.append(fit)
                
                # Store ALL evaluated scenarios for pattern analysis
                all_scenarios_evaluated.append({
                    "config": copy.deepcopy(cfg),
                    "objectives": dict(obj),
                    "fitness": fit,
                    "seed": s,
                    "crashed": crashed,
                    "evaluation_number": total_evaluations
                })
                
                # Track best fitness progression and improvements
                previous_best = current_best_fitness
                if fit < current_best_fitness:
                    current_best_fitness = fit
                    best_cfg = copy.deepcopy(cfg)
                    best_objectives = dict(obj)
                    best_fitness = fit
                    best_seed = s
                    stats["improvements_found"] += 1
                    print(f"  → ✨ NEW BEST: fitness={fit:.2f}, min_distance={obj.get('min_distance', float('inf')):.2f}m")
                    print(f"     (Improvement: {previous_best - fit:.2f})")
                else:
                    stats["no_improvements"] += 1
                
                best_fitness_history.append(current_best_fitness)
                
                print(f"  → Fitness: {fit:.2f}, Min distance: {obj.get('min_distance', float('inf')):.2f}m, "
                      f"Crash: {'YES' if crashed else 'NO'}")
                
                if crashed:
                    print(f"  → 💥 COLLISION DETECTED!")
                    
                    crash_entry = {
                        "cfg": copy.deepcopy(cfg), 
                        "seed": s,
                        "objectives": dict(obj),
                        "fitness": fit,
                        "evaluation_number": total_evaluations,
                        "config": copy.deepcopy(cfg)
                    }
                    crash_log.append(crash_entry)
                    
                    stats["crashes_found"] += 1
                    if stats["crash_scenario_number"] is None:
                        stats["crash_scenario_number"] = i + 1
                    if first_crash_evaluation is None:
                        first_crash_evaluation = total_evaluations
                    
                    # Print immediate crash summary
                    print("\n" + "-"*80)
                    print("CRASH DETECTED - IMMEDIATE SUMMARY")
                    print("-"*80)
                    print(f"Crash #{stats['crashes_found']} found at scenario {i+1}, evaluation {total_evaluations}")
                    print(f"Configuration:")
                    print(f"  - vehicles_count: {cfg.get('vehicles_count')}")
                    print(f"  - lanes_count: {cfg.get('lanes_count')}")
                    print(f"  - initial_spacing: {cfg.get('initial_spacing'):.2f}")
                    print(f"  - ego_spacing: {cfg.get('ego_spacing'):.2f}")
                    print(f"  - initial_lane_id: {cfg.get('initial_lane_id')}")
                    print(f"Reproducibility seed: {s}")
                    print("-"*80 + "\n")
                    
                    record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")
                    break
                else:
                    # Track best non-crash scenario
                    if fit < best_non_crash["fitness"]:
                        best_non_crash = {
                            "config": copy.deepcopy(cfg),
                            "objectives": dict(obj),
                            "fitness": fit,
                            "seed": s
                        }
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Determine best overall scenario
        if crash_log:
            best_crash = min(crash_log, key=lambda x: x["fitness"])
            best_cfg = best_crash["cfg"]
            best_objectives = best_crash["objectives"]
            best_seed = best_crash["seed"]
            best_fitness = best_crash["fitness"]
        
        print(f"\nFinal best fitness: {best_fitness}")
        
        # Analyze patterns for Section 2.4
        pattern_analysis = self.analyze_patterns(all_scenarios_evaluated)
        
        # Print final comprehensive metrics summary
        self.print_metrics_summary(
            stats,
            total_evaluations,
            start_time,
            end_time,
            best_objectives, 
            best_cfg,
            first_crash_evaluation,
            pattern_analysis
        )
        
        # Return standardized format for comparison
        return {
            # Failure Discovery (Section 2.1)
            "collision_found": len(crash_log) > 0,
            "num_crashes": len(crash_log),
            "min_distance_achieved": best_objectives.get("min_distance", float('inf')) if best_objectives else float('inf'),
            "distinct_failing_scenarios": crash_log,
            
            # Best scenario (Section 2.2)
            "best_cfg": best_cfg,
            "best_objectives": best_objectives,
            "best_fitness": best_fitness,
            "best_seed": best_seed,
            "most_critical_scenario": {
                "config": best_cfg,
                "objectives": best_objectives,
                "seed": best_seed,
                "fitness": best_fitness
            },
            
            # All crashes (Section 2.2)
            "crash_scenarios": crash_log,
            "best_non_crash": best_non_crash if not crash_log else None,
            
            # Efficiency (Section 2.3)
            "runtime_seconds": runtime,
            "total_evaluations": total_evaluations,
            "evaluations_to_first_crash": first_crash_evaluation,
            "evaluations_per_second": total_evaluations / runtime if runtime > 0 else 0,
            
            # Qualitative observations (Section 2.4)
            "pattern_analysis": pattern_analysis,
            "all_scenarios_evaluated": all_scenarios_evaluated,
            
            # Convergence (Section 2.4)
            "best_fitness_history": best_fitness_history,
            "all_fitness_values": all_fitness_values,
            "convergence_data": {
                "fitness_per_evaluation": all_fitness_values,
                "best_fitness_progression": best_fitness_history,
                "distance_progression": [s["objectives"]["min_distance"] for s in all_scenarios_evaluated]
            },
            
            # Exploration metrics (Section 2.5)
            "exploration_metrics": {
                "improvements_found": stats["improvements_found"],
                "no_improvements": stats["no_improvements"],
                "improvement_rate": stats["improvements_found"] / total_evaluations * 100 if total_evaluations > 0 else 0
            },
            
            # Method-specific
            "method": "Random Search",
            "method_stats": {
                "n_scenarios": n_scenarios,
                "n_eval_per_scenario": n_eval,
                "crashes_found": stats["crashes_found"],
                "crash_scenario_number": stats["crash_scenario_number"]
            },
            
            # Legacy compatibility
            "stats": stats,
            "crash_log": crash_log
        }

    def analyze_patterns(self, all_scenarios):
        """Analyze patterns in crash vs non-crash scenarios (Section 2.4)"""
        crash_scenarios = [s for s in all_scenarios if s["crashed"]]
        non_crash_scenarios = [s for s in all_scenarios if not s["crashed"]]
        
        def compute_pattern_stats(scenarios):
            if not scenarios:
                return None
            
            configs = [s["config"] for s in scenarios]
            vehicles = [c.get("vehicles_count") for c in configs]
            lanes = [c.get("lanes_count") for c in configs]
            spacings = [c.get("initial_spacing") for c in configs]
            ego_spacings = [c.get("ego_spacing") for c in configs]
            lane_ids = [c.get("initial_lane_id") for c in configs]
            distances = [s["objectives"]["min_distance"] for s in scenarios if s["objectives"]["min_distance"] != float('inf')]
            
            from collections import Counter
            lane_counter = Counter(lane_ids)
            
            return {
                "count": len(scenarios),
                "avg_vehicles_count": np.mean(vehicles),
                "avg_lanes_count": np.mean(lanes),
                "avg_initial_spacing": np.mean(spacings),
                "avg_ego_spacing": np.mean(ego_spacings),
                "vehicles_range": [min(vehicles), max(vehicles)],
                "spacing_range": [min(spacings), max(spacings)],
                "ego_spacing_range": [min(ego_spacings), max(ego_spacings)],
                "most_common_lanes": lane_counter.most_common(3),
                "avg_min_distance": np.mean(distances) if distances else float('inf'),
                "min_distance_range": [min(distances), max(distances)] if distances else [float('inf'), float('inf')]
            }
        
        return {
            "crash_patterns": compute_pattern_stats(crash_scenarios),
            "non_crash_patterns": compute_pattern_stats(non_crash_scenarios),
            "total_scenarios": len(all_scenarios)
        }

    def print_metrics_summary(self, stats, evaluations, start_time, end_time, best_obj, best_cfg, first_crash_eval, pattern_analysis):
        """Print comprehensive search metrics for Random Search (matching HC format)."""
        runtime = end_time - start_time
        print("\n" + "="*80)
        print("RANDOM SEARCH METRICS SUMMARY")
        print("="*80)
        
        # Failure discovery (Section 2.1)
        print("\n[1] FAILURE DISCOVERY:")
        print(f"   • Collision found: {'YES' if stats['crashes_found'] > 0 else 'NO'}")
        print(f"   • Number of crashes: {stats['crashes_found']}")
        if stats['crash_scenario_number'] is not None:
            print(f"   • First crash at scenario: {stats['crash_scenario_number']}")
        if first_crash_eval:
            print(f"   • First crash at evaluation: {first_crash_eval}")
        if best_obj:
            print(f"   • Minimum distance achieved: {best_obj.get('min_distance', float('inf')):.2f}m")
        
        # Scenario characteristics (Section 2.2)
        print("\n[2] SCENARIO CHARACTERISTICS:")
        if stats['crashes_found'] > 0 and best_cfg:
            print(f"   • Most critical scenario configuration:")
            print(f"     - vehicles_count: {best_cfg.get('vehicles_count')}")
            print(f"     - lanes_count: {best_cfg.get('lanes_count')}")
            print(f"     - initial_spacing: {best_cfg.get('initial_spacing'):.2f}")
            print(f"     - ego_spacing: {best_cfg.get('ego_spacing'):.2f}")
            print(f"     - initial_lane_id: {best_cfg.get('initial_lane_id')}")
        elif best_cfg:
            print(f"   • Best scenario configuration (no crash):")
            print(f"     - vehicles_count: {best_cfg.get('vehicles_count')}")
            print(f"     - lanes_count: {best_cfg.get('lanes_count')}")
            print(f"     - initial_spacing: {best_cfg.get('initial_spacing'):.2f}")
            print(f"     - ego_spacing: {best_cfg.get('ego_spacing'):.2f}")
            print(f"     - initial_lane_id: {best_cfg.get('initial_lane_id')}")
        
        # Efficiency (Section 2.3)
        print("\n[3] EFFICIENCY:")
        print(f"   • Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print(f"   • Total scenario evaluations: {evaluations}")
        print(f"   • Evaluations per second: {evaluations/runtime:.2f}")
        if first_crash_eval:
            print(f"   • Evaluations to reach crash: {first_crash_eval}")
        
        # Search effectiveness (Section 2.5)
        total_attempts = stats['improvements_found'] + stats['no_improvements']
        if total_attempts > 0:
            improvement_rate = 100 * stats['improvements_found'] / total_attempts
            print("\n[4] SEARCH EFFECTIVENESS:")
            print(f"   • Improvements found: {stats['improvements_found']}/{total_attempts}")
            print(f"   • Improvement rate: {improvement_rate:.1f}%")
            print(f"   • Scenarios without improvement: {stats['no_improvements']}")
            print(f"   • Crashes found: {stats['crashes_found']}")
        
        # Objective breakdown (Section 2.2)
        if best_obj:
            print("\n[5] OBJECTIVE METRICS:")
            print(f"   • Min distance: {best_obj.get('min_distance', float('inf')):.2f}m")
            print(f"   • Same-lane min distance: {best_obj.get('same_lane_min_distance', float('inf')):.2f}m")
            print(f"   • Min closing distance: {best_obj.get('min_closing_distance', float('inf')):.2f}m")
            print(f"   • Min TTC: {best_obj.get('min_ttc', float('inf')):.2f} timesteps")
        
        # Pattern analysis (Section 2.4)
        print("\n[6] PATTERN ANALYSIS:")
        if pattern_analysis["crash_patterns"]:
            cp = pattern_analysis["crash_patterns"]
            print(f"   • Crash scenarios ({cp['count']} total):")
            print(f"     - Avg vehicles: {cp['avg_vehicles_count']:.1f} (range: {cp['vehicles_range'][0]}-{cp['vehicles_range'][1]})")
            print(f"     - Avg spacing: {cp['avg_initial_spacing']:.2f} (range: {cp['spacing_range'][0]:.2f}-{cp['spacing_range'][1]:.2f})")
            print(f"     - Avg ego spacing: {cp['avg_ego_spacing']:.2f} (range: {cp['ego_spacing_range'][0]:.2f}-{cp['ego_spacing_range'][1]:.2f})")
            print(f"     - Most common lanes: {cp['most_common_lanes']}")
        
        if pattern_analysis["non_crash_patterns"]:
            ncp = pattern_analysis["non_crash_patterns"]
            print(f"   • Non-crash scenarios ({ncp['count']} total):")
            print(f"     - Avg vehicles: {ncp['avg_vehicles_count']:.1f} (range: {ncp['vehicles_range'][0]}-{ncp['vehicles_range'][1]})")
            print(f"     - Avg spacing: {ncp['avg_initial_spacing']:.2f} (range: {ncp['spacing_range'][0]:.2f}-{ncp['spacing_range'][1]:.2f})")
            print(f"     - Avg ego spacing: {ncp['avg_ego_spacing']:.2f} (range: {ncp['ego_spacing_range'][0]:.2f}-{ncp['ego_spacing_range'][1]:.2f})")
            print(f"     - Avg min distance: {ncp['avg_min_distance']:.2f}m")
        
        print("="*80 + "\n")

    def sample_random_config(self, rng):
        from search.base_search import ScenarioSearch
        return ScenarioSearch.sample_random_config(self, rng)
