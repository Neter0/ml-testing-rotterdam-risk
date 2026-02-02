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
        
        # Best non-crash scenario tracking
        best_non_crash = {
            "config": None,
            "objectives": None,
            "fitness": float('inf'),
            "seed": None
        }
        
        for i in trange(n_scenarios, desc="Random search"):
            cfg = self.sample_random_config(rng)
            
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
                
                # Track best fitness progression
                if fit < current_best_fitness:
                    current_best_fitness = fit
                best_fitness_history.append(current_best_fitness)
                
                if crashed:
                    print(f"💥 Collision: scenario {i}, seed={s}, fitness={fit:.2f}")
                    crash_log.append({
                        "cfg": copy.deepcopy(cfg), 
                        "seed": s,
                        "objectives": dict(obj),
                        "fitness": fit,
                        "evaluation_number": total_evaluations
                    })
                    
                    if first_crash_evaluation is None:
                        first_crash_evaluation = total_evaluations
                    
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
            best_scenario = min(crash_log, key=lambda x: x["fitness"])
            best_cfg = best_scenario["cfg"]
            best_objectives = best_scenario["objectives"]
            best_seed = best_scenario["seed"]
            best_fitness = best_scenario["fitness"]
        else:
            best_cfg = best_non_crash["config"]
            best_objectives = best_non_crash["objectives"]
            best_seed = best_non_crash["seed"]
            best_fitness = best_non_crash["fitness"]
        
        # Print metrics summary
        self.print_metrics_summary(
            crash_log, 
            best_objectives, 
            best_cfg, 
            runtime, 
            total_evaluations,
            first_crash_evaluation
        )
        
        # Return standardized format for comparison
        return {
            # Failure Discovery
            "collision_found": len(crash_log) > 0,
            "num_crashes": len(crash_log),
            "min_distance_achieved": best_objectives.get("min_distance", float('inf')) if best_objectives else float('inf'),
            
            # Best scenario
            "best_cfg": best_cfg,
            "best_objectives": best_objectives,
            "best_fitness": best_fitness,
            "best_seed": best_seed,
            
            # All crashes
            "crash_scenarios": crash_log,
            "best_non_crash": best_non_crash if not crash_log else None,
            
            # Efficiency
            "runtime_seconds": runtime,
            "total_evaluations": total_evaluations,
            "evaluations_to_first_crash": first_crash_evaluation,
            "evaluations_per_second": total_evaluations / runtime if runtime > 0 else 0,
            
            # Convergence
            "best_fitness_history": best_fitness_history,
            "all_fitness_values": all_fitness_values,
            
            # Method-specific
            "method": "Random Search",
            "method_stats": {
                "n_scenarios": n_scenarios,
                "n_eval_per_scenario": n_eval
            }
        }

    def print_metrics_summary(self, crash_log, best_obj, best_cfg, runtime, evaluations, first_crash_eval):
        """Print comprehensive search metrics for Random Search."""
        print("\n" + "="*80)
        print("RANDOM SEARCH METRICS SUMMARY")
        print("="*80)
        
        # Failure discovery
        print("\n[1] FAILURE DISCOVERY:")
        print(f"   • Collision found: {'YES' if crash_log else 'NO'}")
        print(f"   • Number of crashes: {len(crash_log)}")
        if first_crash_eval:
            print(f"   • First crash at evaluation: {first_crash_eval}")
        if best_obj:
            print(f"   • Minimum distance achieved: {best_obj.get('min_distance', float('inf')):.2f}m")
        
        # Scenario characteristics
        print("\n[2] SCENARIO CHARACTERISTICS:")
        if crash_log:
            print(f"   • Critical scenario configuration:")
            cfg = crash_log[0]['cfg']
            print(f"     - vehicles_count: {cfg.get('vehicles_count')}")
            print(f"     - lanes_count: {cfg.get('lanes_count')}")
            print(f"     - initial_spacing: {cfg.get('initial_spacing'):.2f}")
            print(f"     - ego_spacing: {cfg.get('ego_spacing'):.2f}")
            print(f"     - initial_lane_id: {cfg.get('initial_lane_id')}")
            print(f"   • Reproducibility seed: {crash_log[0]['seed']}")
        elif best_cfg:
            print(f"   • Best scenario configuration (no crash):")
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
        if first_crash_eval:
            print(f"   • Evaluations to reach crash: {first_crash_eval}")
        
        # Objective breakdown
        if best_obj:
            print("\n[4] OBJECTIVE METRICS:")
            print(f"   • Min distance: {best_obj.get('min_distance', float('inf')):.2f}m")
            print(f"   • Same-lane min distance: {best_obj.get('same_lane_min_distance', float('inf')):.2f}m")
            print(f"   • Min closing distance: {best_obj.get('min_closing_distance', float('inf')):.2f}m")
            print(f"   • Min TTC: {best_obj.get('min_ttc', float('inf')):.2f} timesteps")
        
        print("="*80 + "\n")

    def sample_random_config(self, rng):
        from search.base_search import ScenarioSearch
        return ScenarioSearch.sample_random_config(self, rng)
