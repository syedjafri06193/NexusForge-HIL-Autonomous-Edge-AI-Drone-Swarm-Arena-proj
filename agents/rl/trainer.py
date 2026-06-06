"""
NexusForge Reinforcement Learning
PPO-style training for drone combat policies.
Drones learn by self-play: each episode runs a short sim and collects
(state, action, reward) tuples, then updates a simple policy network.

Designed to run without GPU — lightweight enough for CPU training.
Trained policies are exported and plugged back into the behavior tree
as the 'rl_policy' model_type.
"""

import math
import random
import time
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


# ─── Observation / action spaces ─────────────────────────────────────────────

OBS_DIM = 24      # state features per drone
ACT_DIM = 5       # [move_x, move_y, fire, evade, capture] (continuous/discrete mix)

# Observation features:
# [0,1]   - normalized position (x/W, y/H)
# [2,3]   - normalized velocity (vx/vmax, vy/vmax)
# [4]     - heading / pi
# [5]     - health / 100
# [6]     - shield / 50
# [7]     - battery / 100
# [8]     - weapon_ready (0/1)
# [9]     - stun_timer (clamped)
# [10,11] - nearest enemy relative pos (dx/W, dy/H)
# [12]    - nearest enemy distance / sensor_range
# [13]    - nearest enemy health / 100
# [14,15] - nearest ally relative pos
# [16]    - ally count / max_allies
# [17]    - enemy count / max_enemies
# [18,19] - nearest control point relative pos
# [20]    - nearest control point capture progress
# [21]    - is_outnumbered (0/1)
# [22]    - team_score / 100 (normalized)
# [23]    - elapsed / 300 (normalized time)


def extract_observation(drone, sim, max_allies: int = 32) -> np.ndarray:
    """Extract a fixed-size observation vector for a drone."""
    from simulation.engine.sim import ARENA_W, ARENA_H, MAX_SPEED, SENSOR_RANGE

    obs = np.zeros(OBS_DIM, dtype=np.float32)

    obs[0] = drone.position.x / ARENA_W
    obs[1] = drone.position.y / ARENA_H
    obs[2] = drone.velocity.x / MAX_SPEED
    obs[3] = drone.velocity.y / MAX_SPEED
    obs[4] = drone.heading / math.pi
    obs[5] = drone.health / 100.0
    obs[6] = drone.shield / 50.0
    obs[7] = drone.battery_pct / 100.0
    obs[8] = 1.0 if drone.weapon_cooldown <= 0 else 0.0
    obs[9] = min(1.0, drone.stun_timer)

    if drone.visible_enemies:
        nearest_e = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position))
        dx = (nearest_e.position.x - drone.position.x) / ARENA_W
        dy = (nearest_e.position.y - drone.position.y) / ARENA_H
        dist = drone.position.distance_to(nearest_e.position)
        obs[10] = dx
        obs[11] = dy
        obs[12] = dist / SENSOR_RANGE
        obs[13] = nearest_e.health / 100.0

    obs[16] = len(drone.neighbors) / max_allies
    obs[17] = len(drone.visible_enemies) / max_allies
    obs[21] = 1.0 if len(drone.visible_enemies) > len(drone.neighbors) + 1 else 0.0

    if drone.neighbors:
        nearest_a = min(drone.neighbors, key=lambda a: drone.position.distance_to(a.position))
        obs[14] = (nearest_a.position.x - drone.position.x) / ARENA_W
        obs[15] = (nearest_a.position.y - drone.position.y) / ARENA_H

    # Nearest unowned control point
    cps = sim.arena.control_points
    unowned = [cp for cp in cps if cp.get("owner") != drone.team.name]
    if unowned:
        nearest_cp = min(unowned, key=lambda cp: math.sqrt(
            (cp["x"] - drone.position.x)**2 + (cp["y"] - drone.position.y)**2
        ))
        obs[18] = (nearest_cp["x"] - drone.position.x) / ARENA_W
        obs[19] = (nearest_cp["y"] - drone.position.y) / ARENA_H
        obs[20] = nearest_cp.get("capture", 0.0)

    obs[22] = sim.scores.get(drone.team.name, 0) / 100.0
    obs[23] = min(1.0, sim.elapsed / 300.0)

    return obs


def compute_reward(drone, prev_health: float, prev_kills: int,
                   prev_score: int, sim, dt: float) -> float:
    """Dense reward signal for drone combat."""
    reward = 0.0

    # Survival bonus
    if drone.is_alive:
        reward += 0.01

    # Kill reward
    kill_delta = drone.kills - prev_kills
    reward += kill_delta * 2.0

    # Damage taken penalty
    health_delta = prev_health - drone.health
    reward -= health_delta * 0.02

    # Score capture bonus
    score_delta = sim.scores.get(drone.team.name, 0) - prev_score
    reward += score_delta * 0.05

    # Death penalty
    if not drone.is_alive and prev_health > 0:
        reward -= 3.0

    # Battery efficiency (small penalty for high power draw)
    reward -= (1.0 - drone.battery_pct / 100.0) * 0.001

    return reward


# ─── Simple policy network (NumPy MLP) ───────────────────────────────────────

class PolicyNetwork:
    """
    Lightweight 2-layer MLP policy.
    No PyTorch/TF dependency — runs on any MCU simulator or CPU.
    Uses NumPy for forward pass.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 64, lr: float = 3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden
        self.lr      = lr

        # Xavier initialization
        scale1 = math.sqrt(2.0 / obs_dim)
        scale2 = math.sqrt(2.0 / hidden)
        self.W1 = np.random.randn(obs_dim, hidden).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = np.random.randn(hidden, act_dim).astype(np.float32) * 0.01
        self.b3 = np.zeros(act_dim, dtype=np.float32)

        # Value head (critic)
        self.Wv = np.random.randn(hidden, 1).astype(np.float32) * 0.01
        self.bv = np.zeros(1, dtype=np.float32)

        self._grads: Dict = {}

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (action_probs, value_estimate)."""
        h1 = self._relu(obs @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        action_probs = self._sigmoid(logits)   # independent bernoulli per action
        value = float((h2 @ self.Wv + self.bv)[0])
        self._last_h1, self._last_h2 = h1, h2
        return action_probs, value

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Sample discrete actions, return (actions, log_probs, value)."""
        probs, value = self.forward(obs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        actions = (np.random.random(self.act_dim) < probs).astype(np.float32)
        log_probs = np.where(actions > 0, np.log(probs), np.log(1 - probs))
        return actions, log_probs, value

    def update_pg(self, obs: np.ndarray, actions: np.ndarray,
                  advantages: np.ndarray, returns: np.ndarray):
        """Vanilla policy gradient update (REINFORCE with baseline)."""
        probs, value = self.forward(obs)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        # Policy loss gradient (negative because we maximize)
        adv = advantages.mean() if advantages.ndim > 0 else advantages
        pg_grad = -adv * np.where(actions > 0, 1.0 / probs, -1.0 / (1 - probs))

        # Value loss gradient
        value_error = value - returns.mean()
        vg = 2 * value_error

        # Backprop through W3/b3
        dL_dlogits = pg_grad * probs * (1 - probs)
        h2 = self._last_h2
        self.W3 -= self.lr * np.outer(h2, dL_dlogits)
        self.b3 -= self.lr * dL_dlogits

        # Backprop through Wv/bv
        self.Wv -= self.lr * vg * h2.reshape(-1, 1)
        self.bv -= self.lr * vg

        # Backprop through W2/b2
        dh2 = (dL_dlogits @ self.W3.T + vg * self.Wv.T[0]) * (h2 > 0)
        self.W2 -= self.lr * np.outer(self._last_h1, dh2)
        self.b2 -= self.lr * dh2

        # Backprop through W1/b1
        dh1 = (dh2 @ self.W2.T) * (self._last_h1 > 0)
        self.W1 -= self.lr * np.outer(obs, dh1)
        self.b1 -= self.lr * dh1

    def save(self, path: str):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 Wv=self.Wv, bv=self.bv)

    def load(self, path: str):
        data = np.load(path + '.npz')
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        self.Wv, self.bv = data['Wv'], data['bv']

    def export_weights_json(self) -> dict:
        """Export for embedding in firmware or frontend visualization."""
        return {
            "obs_dim": self.obs_dim, "act_dim": self.act_dim, "hidden": self.hidden,
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
        }


# ─── Rollout buffer ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs:       np.ndarray
    action:    np.ndarray
    log_prob:  np.ndarray
    reward:    float
    value:     float
    done:      bool


class RolloutBuffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam   = lam
        self._buf: List[Transition] = []

    def add(self, t: Transition):
        self._buf.append(t)

    def clear(self):
        self._buf = []

    def __len__(self):
        return len(self._buf)

    def compute_returns_and_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-lambda advantage estimation."""
        n = len(self._buf)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(n)):
            t = self._buf[i]
            mask = 0.0 if t.done else 1.0
            delta = t.reward + self.gamma * next_value * mask - t.value
            gae   = delta + self.gamma * self.lam * mask * gae
            advantages[i] = gae
            returns[i]    = gae + t.value
            next_value    = t.value

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        return returns, advantages

    def sample_all(self):
        returns, advantages = self.compute_returns_and_advantages()
        obs     = np.stack([t.obs    for t in self._buf])
        actions = np.stack([t.action for t in self._buf])
        return obs, actions, advantages, returns


# ─── Training loop ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    n_episodes:       int   = 200
    episode_ticks:    int   = 600      # 10 seconds at 60 FPS
    drones_per_team:  int   = 4
    num_teams:        int   = 2
    gamma:            float = 0.99
    lam:              float = 0.95
    lr:               float = 3e-4
    hidden:           int   = 64
    save_interval:    int   = 50
    save_dir:         str   = "agents/models"
    log_interval:     int   = 10
    entropy_coef:     float = 0.01


class SelfPlayTrainer:
    """
    Trains drone combat policies via self-play.
    All teams share the same policy (parameter sharing).
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.policy = PolicyNetwork(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=config.hidden, lr=config.lr
        )
        self.buffer = RolloutBuffer(config.gamma, config.lam)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self.kill_counts:     List[int]   = []
        os.makedirs(config.save_dir, exist_ok=True)

    def run_episode(self) -> dict:
        """Run one episode of self-play, collect rollout data."""
        from simulation.engine.sim import Simulation, TeamID, DroneConfig

        sim = Simulation(
            num_teams=self.cfg.num_teams,
            drones_per_team=self.cfg.drones_per_team,
        )

        # Override all drones with RL policy type
        for drone in sim.drones.values():
            drone.config.model_type = "rl_policy"

        prev_states: Dict[str, dict] = {
            d.id: {"health": d.health, "kills": d.kills, "score": 0}
            for d in sim.drones.values()
        }

        total_reward = 0.0
        episode_transitions: List[Tuple[str, Transition]] = []

        for tick in range(self.cfg.episode_ticks):
            alive_drones = [d for d in sim.drones.values() if d.is_alive]

            for drone in alive_drones:
                ps = prev_states[drone.id]
                obs = extract_observation(drone, sim)
                actions, log_probs, value = self.policy.sample_action(obs)

                # Apply actions to drone
                self._apply_rl_action(drone, actions, sim)

                reward = compute_reward(
                    drone, ps["health"], ps["kills"], ps["score"], sim, dt=1/60
                )
                total_reward += reward

                transition = Transition(
                    obs=obs, action=actions, log_prob=log_probs,
                    reward=reward, value=value,
                    done=(tick == self.cfg.episode_ticks - 1)
                )
                episode_transitions.append((drone.id, transition))

                prev_states[drone.id] = {
                    "health": drone.health,
                    "kills":  drone.kills,
                    "score":  sim.scores.get(drone.team.name, 0),
                }

            sim.tick_once()

        # Add all transitions to buffer
        for _, t in episode_transitions:
            self.buffer.add(t)

        total_kills = sum(d.kills for d in sim.drones.values())
        return {
            "total_reward": total_reward,
            "total_kills":  total_kills,
            "ticks":        self.cfg.episode_ticks,
            "scores":       dict(sim.scores),
        }

    def _apply_rl_action(self, drone, actions: np.ndarray, sim):
        """Convert policy output vector into drone forces."""
        from simulation.engine.sim import MAX_SPEED, WEAPON_RANGE, Vec2

        move_x = (actions[0] - 0.5) * 2  # [-1, 1]
        move_y = (actions[1] - 0.5) * 2
        fire   = actions[2] > 0.5
        evade  = actions[3] > 0.5

        if evade and drone.visible_enemies:
            nearest = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position))
            away = (drone.position - nearest.position).normalized()
            move_x = away.x
            move_y = away.y

        desired = Vec2(move_x, move_y).normalized() * MAX_SPEED
        drone.apply_force((desired - drone.velocity) * 3.0)

        if fire and drone.visible_enemies and drone.weapon_cooldown <= 0:
            target = min(drone.visible_enemies, key=lambda e: drone.position.distance_to(e.position))
            if drone.position.distance_to(target.position) <= WEAPON_RANGE * 1.5:
                sim._fire_weapon(drone, target)

    def update_policy(self):
        """Update policy from collected rollout data."""
        if len(self.buffer) == 0:
            return
        obs, actions, advantages, returns = self.buffer.sample_all()
        # Mini-batch updates
        n = len(obs)
        indices = np.random.permutation(n)
        batch_size = min(64, n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) == 0:
                continue
            # Average over batch
            batch_obs  = obs[idx].mean(axis=0)
            batch_act  = actions[idx].mean(axis=0)
            batch_adv  = advantages[idx]
            batch_ret  = returns[idx]
            self.policy.update_pg(batch_obs, batch_act, batch_adv, batch_ret)
        self.buffer.clear()

    def train(self) -> dict:
        """Full training loop."""
        print(f"[RL] Starting self-play training: {self.cfg.n_episodes} episodes")
        t0 = time.time()

        for ep in range(1, self.cfg.n_episodes + 1):
            stats = self.run_episode()
            self.episode_rewards.append(stats["total_reward"])
            self.episode_lengths.append(stats["ticks"])
            self.kill_counts.append(stats["total_kills"])

            self.update_policy()

            if ep % self.cfg.log_interval == 0:
                avg_r = np.mean(self.episode_rewards[-self.cfg.log_interval:])
                avg_k = np.mean(self.kill_counts[-self.cfg.log_interval:])
                elapsed = time.time() - t0
                print(f"[RL] Ep {ep:4d}/{self.cfg.n_episodes} | "
                      f"avg_reward={avg_r:7.2f} | avg_kills={avg_k:.1f} | "
                      f"elapsed={elapsed:.1f}s")

            if ep % self.cfg.save_interval == 0:
                path = os.path.join(self.cfg.save_dir, f"policy_ep{ep}")
                self.policy.save(path)
                print(f"[RL] Saved checkpoint: {path}")

        # Final save
        final_path = os.path.join(self.cfg.save_dir, "policy_final")
        self.policy.save(final_path)

        # Export JSON weights for frontend visualization
        weights_path = os.path.join(self.cfg.save_dir, "policy_weights.json")
        with open(weights_path, "w") as f:
            json.dump(self.policy.export_weights_json(), f)

        total_time = time.time() - t0
        summary = {
            "episodes": self.cfg.n_episodes,
            "total_time_s": round(total_time, 1),
            "final_avg_reward": round(float(np.mean(self.episode_rewards[-20:])), 3),
            "final_avg_kills": round(float(np.mean(self.kill_counts[-20:])), 2),
            "best_reward": round(float(max(self.episode_rewards)), 3),
            "model_path": final_path,
        }
        print(f"[RL] Training complete: {summary}")
        return summary


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NexusForge RL trainer")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--teams", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    cfg = TrainingConfig(
        n_episodes=args.episodes,
        drones_per_team=args.drones,
        num_teams=args.teams,
        lr=args.lr,
        hidden=args.hidden,
    )
    trainer = SelfPlayTrainer(cfg)
    trainer.train()
