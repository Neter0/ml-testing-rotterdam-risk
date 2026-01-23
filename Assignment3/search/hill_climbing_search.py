import numpy as np
from tqdm import trange
from search.base_search import ScenarioSearch
from search.hill_climbing import hill_climb
from envs.highway_env_utils import record_video_episode


class HillClimbingSearch(ScenarioSearch):
    def run_search(
        self,
        n_scenarios = 50,
        seed = 0,
        iterations = 100,
        neighbors_per_iteration = 10,
    ):
        rng = np.random.default_rng(seed)
        crashes = []

        for i in trange(int(n_scenarios), desc="Hill Climbing search"):
            run_seed = int(rng.integers(1_000_000))
            result = hill_climb(
                env_id=self.env_id,
                base_cfg=self.base_cfg,
                param_spec=self.param_spec,
                policy=self.policy,
                defaults=self.defaults,
                seed=run_seed,
                iterations=iterations,
                neighbors_per_iter=neighbors_per_iteration,
            )

            if int(result["best_objectives"].get("crash_count", 0)) > 0:
                print(f"Crash: scenario {i}, seed={run_seed}\n")
                record_video_episode(
                    self.env_id,
                    result["best_cfg"],
                    self.policy,
                    self.defaults,
                    result["best_seed_base"],
                    out_dir="videos",
                )
                crashes.append(result)
                print()
            else:
                print(f"No Crash: scenario {i}, seed={run_seed}")

        return crashes