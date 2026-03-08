# engine/rl_agent.py
"""
PPO Reinforcement Learning Agent — Real-Time Re-routing
---------------------------------------------------------
The RL agent monitors live conditions and decides WHETHER and HOW to
re-route vehicles mid-delivery when conditions change.

Architecture:
  • Gym-compatible custom environment: VRPEnv
  • State  : per-vehicle position, remaining stops, traffic multiplier,
             time elapsed, capacity remaining, priority urgency
  • Action : reassign next stop (discrete: which stop to visit next)
  • Reward : -travel_time + priority_bonus - deadline_violation_penalty
  • Algorithm : PPO via stable-baselines3

Usage:
    from engine.rl_agent import RLReRouter
    agent = RLReRouter()
    agent.train(data, result, timesteps=10_000)
    agent.save("ml_model/ppo_router")

    # Real-time: given updated traffic, get re-routed sequence
    new_routes = agent.reroute(data, result, traffic_multiplier=1.4)
"""

import numpy as np
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Stable-Baselines3 + Gymnasium ────────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed.")
    print("    Run: pip install stable-baselines3 gymnasium")


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Gymnasium Environment
# ══════════════════════════════════════════════════════════════════════════════
if SB3_AVAILABLE:
    class VRPEnv(gym.Env):
        """
        Single-vehicle routing environment for PPO training.
        One episode = one vehicle completing its assigned stops.

        State vector (per step):
          [current_node / N,           # normalized current position
           stops_remaining / N,        # progress indicator
           traffic_multiplier,         # 1.0=normal, >1=slower
           time_elapsed / 480,         # fraction of 8-hr shift used
           capacity_remaining / C,     # weight capacity fraction
           next_deadline_urgency,      # (latest_arrival - now) / 480
           avg_priority_score]         # 0=LOW..1=URGENT for remaining stops

        Action: integer index into remaining stops list
        """

        metadata = {"render_modes": []}

        PRIORITY_SCORE = {"URGENT": 1.0, "HIGH": 0.67, "MEDIUM": 0.33, "LOW": 0.0}

        def __init__(
            self,
            data: Dict,
            vehicle_capacity_kg: float = 300.0,
            traffic_multiplier:  float = 1.0,
            max_stops:           int   = 30,   # pad/truncate to fixed action space
        ):
            super().__init__()
            self.data                = data
            self.vehicle_capacity    = vehicle_capacity_kg
            self.base_traffic        = traffic_multiplier
            self.max_stops           = max_stops
            self.n_nodes             = len(data["time_matrix"])

            # Fixed-size action space (pad remaining stops list to max_stops)
            self.action_space      = spaces.Discrete(max_stops)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(7,), dtype=np.float32
            )
            self._reset_state()

        def _reset_state(self):
            self.current_node       = 0        # depot
            self.time_elapsed       = 0.0
            self.capacity_remaining = self.vehicle_capacity
            self.remaining_stops    = list(range(1, len(self.data["stops"]) + 1))
            self.visited            = []
            self.traffic            = self.base_traffic + np.random.uniform(-0.1, 0.3)

        def _get_obs(self) -> np.ndarray:
            N = self.n_nodes
            n_rem = len(self.remaining_stops)

            if n_rem == 0:
                return np.zeros(7, dtype=np.float32)

            # Urgency of the most urgent remaining stop
            deadlines = []
            priorities = []
            for node in self.remaining_stops:
                if node - 1 < len(self.data["stops"]):
                    s = self.data["stops"][node - 1]
                    deadlines.append(s["latest_arrival"])
                    priorities.append(self.PRIORITY_SCORE.get(s["priority"], 0.0))

            min_deadline  = min(deadlines) if deadlines else 480
            avg_priority  = np.mean(priorities) if priorities else 0.0
            urgency       = max(0.0, (min_deadline - self.time_elapsed) / 480.0)

            return np.array([
                self.current_node / max(N, 1),
                n_rem / max(self.max_stops, 1),
                min(self.traffic / 2.0, 1.0),
                min(self.time_elapsed / 480.0, 1.0),
                self.capacity_remaining / self.vehicle_capacity,
                min(urgency, 1.0),
                avg_priority
            ], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._reset_state()
            return self._get_obs(), {}

        def step(self, action: int):
            if not self.remaining_stops:
                return self._get_obs(), 0.0, True, False, {}

            # Map action index → actual stop (clamp to valid range)
            action_idx = min(action, len(self.remaining_stops) - 1)
            next_node  = self.remaining_stops[action_idx]

            # Travel time (affected by traffic)
            travel = (self.data["time_matrix"][self.current_node][next_node]
                      * self.traffic)

            # Service time at stop
            stop_idx = next_node - 1
            service  = 0
            weight   = 0
            deadline = 480
            priority = "LOW"
            if stop_idx < len(self.data["stops"]):
                s        = self.data["stops"][stop_idx]
                service  = s["service_time"]
                weight   = s["weight_kg"]
                deadline = s["latest_arrival"]
                priority = s["priority"]

            self.time_elapsed       += travel + service
            self.capacity_remaining -= weight
            self.current_node        = next_node
            self.visited.append(next_node)
            self.remaining_stops.remove(next_node)

            # ── Reward shaping ────────────────────────────────────────────────
            # Penalize travel time
            reward = -travel / 60.0

            # Bonus for serving high-priority stops
            priority_bonus = {"URGENT": 5.0, "HIGH": 3.0, "MEDIUM": 1.0, "LOW": 0.0}
            reward += priority_bonus.get(priority, 0.0)

            # Deadline violation penalty
            if self.time_elapsed > deadline:
                violation = (self.time_elapsed - deadline) / 60.0
                reward -= violation * 2.0

            # Capacity violation — terminal penalty
            if self.capacity_remaining < 0:
                reward -= 20.0
                return self._get_obs(), reward, True, False, {}

            # Episode ends when all stops visited or shift over
            done = len(self.remaining_stops) == 0 or self.time_elapsed >= 480

            # Bonus for completing all stops
            if done and len(self.remaining_stops) == 0:
                reward += 10.0

            return self._get_obs(), reward, done, False, {}


# ══════════════════════════════════════════════════════════════════════════════
#  RLReRouter — high-level API
# ══════════════════════════════════════════════════════════════════════════════
class RLReRouter:
    """
    Wraps the PPO agent with train / save / load / reroute API.
    Degrades gracefully to OR-Tools result if SB3 not available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional["PPO"] = None
        self._available = SB3_AVAILABLE
        if model_path and SB3_AVAILABLE:
            self.load(model_path)

    def train(
        self,
        data: Dict,
        total_timesteps: int = 50_000,
        n_envs: int = 4,
        traffic_multiplier: float = 1.0,
        verbose: int = 0
    ) -> "PPO":
        """
        Trains a PPO agent on the given day's routing problem.

        Parameters
        ----------
        data            : preprocessed day data dict
        total_timesteps : PPO training steps (50k = quick demo, 500k = production)
        n_envs          : parallel environments for faster training
        traffic_multiplier : base traffic level for training scenarios
        """
        if not self._available:
            print("⚠️  SB3 not available — skipping RL training")
            return None

        max_stops = max(20, len(data["stops"]))

        def make_env():
            return VRPEnv(
                data=data,
                vehicle_capacity_kg=300.0,
                traffic_multiplier=traffic_multiplier,
                max_stops=max_stops
            )

        vec_env   = make_vec_env(make_env, n_envs=n_envs)
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=verbose,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,     # entropy bonus encourages exploration
        )
        self.model.learn(total_timesteps=total_timesteps)
        vec_env.close()
        print(f"✅ PPO agent trained for {total_timesteps:,} timesteps")
        return self.model

    def save(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            self.model.save(path)
            print(f"✅ PPO model saved to {path}")

    def load(self, path: str):
        if not self._available:
            return
        if os.path.exists(path + ".zip") or os.path.exists(path):
            self.model = PPO.load(path)
            print(f"✅ PPO model loaded from {path}")
        else:
            print(f"⚠️  Model not found at {path}")

    def reroute(
        self,
        data: Dict,
        original_result: Dict,
        traffic_multiplier: float = 1.0,
        vehicle_id: Optional[int] = None
    ) -> Dict:
        """
        Re-routes one or all vehicles given updated traffic conditions.

        Parameters
        ----------
        data               : preprocessed day data dict
        original_result    : current optimizer result
        traffic_multiplier : current traffic (1.0=normal, 1.5=heavy, 2.0=severe)
        vehicle_id         : if set, re-route only this vehicle; else all

        Returns
        -------
        Updated result dict with re-routed sequences and updated metrics
        """
        if not self._available or self.model is None:
            print("⚠️  No RL model available — returning original routes")
            return original_result

        updated_routes = []
        max_stops = max(20, len(data["stops"]))

        for route in original_result["optimized_routes"]:
            if vehicle_id is not None and route["vehicle_id"] != vehicle_id:
                updated_routes.append(route)
                continue

            # Build a sub-data dict for this vehicle's assigned stops
            assigned_node_ids = route["stop_sequence"]
            if not assigned_node_ids:
                updated_routes.append(route)
                continue

            stop_ids_set  = set(assigned_node_ids)
            vehicle_stops = [s for s in data["stops"] if s["id"] in stop_ids_set]

            # Create sub-matrices for this vehicle's stops + depot
            node_ids  = [0] + assigned_node_ids
            n         = len(data["time_matrix"])
            tm        = np.array(data["time_matrix"])
            dm        = np.array(data["dist_matrix"])
            sub_time  = tm[np.ix_(node_ids, node_ids)].tolist()
            sub_dist  = dm[np.ix_(node_ids, node_ids)].tolist()

            # Re-map stop IDs
            id_map = {old: new for new, old in enumerate(node_ids)}
            remapped = []
            for s in vehicle_stops:
                s2 = dict(s)
                s2["id"] = id_map[s["id"]]
                remapped.append(s2)

            sub_data = {
                "stops": remapped,
                "time_matrix": sub_time,
                "dist_matrix": sub_dist
            }

            # Run RL policy
            env  = VRPEnv(
                sub_data,
                traffic_multiplier=traffic_multiplier,
                max_stops=max(max_stops, len(remapped))
            )
            obs, _ = env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(int(action))

            # Reverse-map node IDs back to original
            reverse_map = {v: k for k, v in id_map.items()}
            new_sequence = [reverse_map[n] for n in env.visited if n != 0]

            # Recompute metrics with traffic
            new_time = sum(
                data["time_matrix"][new_sequence[i - 1] if i > 0 else 0][new_sequence[i]]
                * traffic_multiplier
                for i in range(len(new_sequence))
            )
            new_dist = sum(
                data["dist_matrix"][new_sequence[i - 1] if i > 0 else 0][new_sequence[i]]
                for i in range(len(new_sequence))
            )

            updated_routes.append({
                **route,
                "stop_sequence":    new_sequence,
                "num_stops":        len(new_sequence),
                "travel_time_min":  round(new_time, 1),
                "distance_km":      round(new_dist, 2),
                "rerouted":         True,
                "traffic_factor":   traffic_multiplier
            })

        result = dict(original_result)
        result["optimized_routes"]    = updated_routes
        result["rerouting_applied"]   = True
        result["traffic_multiplier"]  = traffic_multiplier
        return result

    def get_training_status(self) -> Dict:
        return {
            "sb3_available": self._available,
            "model_loaded":  self.model is not None,
            "policy":        "PPO MlpPolicy" if self.model else None
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("Install: pip install stable-baselines3 gymnasium")
    else:
        with open("data/processed/day1_mostlikely.json") as f:
            data = json.load(f)
        with open("data/results/day1_result.json") as f:
            result = json.load(f)

        agent = RLReRouter()
        print("🚀 Training PPO agent (quick 5k steps demo)...")
        agent.train(data, total_timesteps=5_000, verbose=1)
        agent.save("ml_model/ppo_router_day1")

        print("\n🔄 Simulating heavy traffic re-routing (1.5x multiplier)...")
        new_result = agent.reroute(data, result, traffic_multiplier=1.5)

        for r in new_result["optimized_routes"]:
            tag = "🔄 REROUTED" if r.get("rerouted") else "  original"
            print(f"  {tag} Vehicle {r['vehicle_id']}: "
                  f"{r['num_stops']} stops | {r['travel_time_min']} min")