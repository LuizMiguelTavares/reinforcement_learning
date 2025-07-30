#!/usr/bin/env python3
"""
Enhanced Q-learning Grid-World (Python 3.8+)
-------------------------------------------
• Adaptive ε-greedy, reward shaping, 4/8-direction moves
• Clustered/random obstacles (seeded)
• Large grids (tested 300x300)
• Combined visual: greedy path & value-map with arrows
"""

import math, time
from collections import deque
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle

# ---------------------------------------------------------------------
# GridWorld environment
# ---------------------------------------------------------------------
class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float = 0.2,
        obstacle_mode: str = "cluster",            # 'random' | 'cluster'
        seed: Optional[int] = None,
        cluster_size: int = 5,
        grid_map: Optional[np.ndarray] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        allow_diagonal: bool = True,
        reward_goal: float = 100.0,
        reward_obstacle: float = -100.0,
        reward_step: float = -0.001,
        use_reward_shaping: bool = False,
        shaping: Optional[str] = "euclidean",      # None | 'manhattan' | 'euclidean'
        allow_diagonal_obstacle: bool = False,
        min_dist_nearby_obstacle: int = 3,
        safety_nearby_obstacle_gain: float = 0.0,
        energy_consumption_gain: float = 0.0,
    ):
        
        # Build grid --------------------------------------------------
        if grid_map is not None:
            self.grid = grid_map.copy()
            self.width, self.height = self.grid.shape[1], self.grid.shape[0]
        else:
            self.width, self.height = width, height
            self.obstacle_density = obstacle_density
            self.obstacle_mode = obstacle_mode
            self.cluster_size = cluster_size
            self.rng = np.random.RandomState(seed)
            self.grid = np.zeros((height, width), dtype=np.int8)
            self._populate_obstacles()
        
        self.start = tuple(start) if start else (0, 0)
        self.goal = tuple(goal) if goal else (self.height - 1, self.width - 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        self.grid_explored = np.zeros_like(self.grid, dtype=np.int32)
        self.grid_explored[self.start] = 1

        # Basic Rewards
        self.reward_goal = reward_goal
        self.reward_obstacle = reward_obstacle
        self.reward_step = reward_step

        # Reward shaping
        self.use_reward_shaping = use_reward_shaping
        self.shaping = shaping

        # Safety
        self.safety_nearby_obstacle_gain = safety_nearby_obstacle_gain
        self.safety_nearby_obstacle = (True if safety_nearby_obstacle_gain > 0 else False)
        self.min_dist_nearby_obstacle = min_dist_nearby_obstacle
        
        # Energy consumption
        self.energy_consumption_gain = abs(energy_consumption_gain)

        self.allow_diag = allow_diagonal
        self.diag_cost = -abs(math.sqrt(2)*reward_step)
        self.allow_diag_obstacle = allow_diagonal_obstacle

        self.angle = -1 # Angle initialization

        if self.safety_nearby_obstacle:
            self.precompute_nearby_obstacles_reward()

        # Action set --------------------------------------------------
        if allow_diagonal:
            self.actions: List[Tuple[int, int]] = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
        else:
            self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)
        self.agent_pos: Optional[Tuple[int, int]] = None

    # -----------------------------------------------------------------
    def _populate_obstacles(self) -> None:
        total = self.width * self.height
        n_obs = int(total * self.obstacle_density)
        if n_obs == 0:
            return
        if self.obstacle_mode == "random":
            idx = self.rng.choice(total, size=n_obs, replace=False)
            self.grid.flat[idx] = 1
            return
        # cluster mode
        placed = 0
        while placed < n_obs:
            center_r = self.rng.randint(0, self.height)
            center_c = self.rng.randint(0, self.width)
            for _ in range(self.cluster_size):
                if placed >= n_obs:
                    break
                dr = self.rng.randint(-self.cluster_size, self.cluster_size + 1)
                dc = self.rng.randint(-self.cluster_size, self.cluster_size + 1)
                r, c = center_r + dr, center_c + dc
                if 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    placed += 1

    # -----------------------------------------------------------------
    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start
        self.angle = -1
        return self.agent_pos
    
    def reset_new_position(self, pos: Tuple[int, int]) -> bool:
        """Reset the agent to a new position."""
        if pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width:
            raise ValueError("Position out of bounds.")
        if self.grid[pos] == 1:
            return False
        self.agent_pos = pos
        self.angle = -1
        return True

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        r, c = self.agent_pos
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        # invalid move (wall or obstacle) -----------------------------
        if not (0 <= nr < self.height and 0 <= nc < self.width) or self.grid[nr, nc] == 1:
            next_state = (r, c) # Stay in place
            reward, done = self.reward_obstacle, False
            return next_state, reward, done

        self.grid_explored[nr, nc] += 1

        angle = math.atan2(dr, dc)
        turn_angle = 0.0

        if self.angle == -1:
            turn_angle = 0.0
            self.angle = math.atan2(dr, dc)
        else:
            turn_angle = angle - self.angle
            turn_angle = (turn_angle + math.pi) % (2 * math.pi) - math.pi

        if not self.allow_diag_obstacle and self.allow_diag:
            try:
                corner1 = self.grid[nr, c]
            except IndexError:
                corner1 = 0 # Não vou contar valores fora do grid como obstáculos
            try:
                corner2 = self.grid[r, nc]
            except IndexError:
                corner2 = 0

            if corner1 == 1 or corner2 == 1:
                next_state = (r, c)
                reward, done = self.reward_obstacle, False
                return next_state, reward, done
    
        next_state = (nr, nc)
        self.agent_pos = next_state
        if next_state == self.goal:
            reward, done = self.reward_goal, True
        else:
            if action >=4 and self.allow_diag:
                reward = self.diag_cost
            else:
                reward = self.reward_step
            done = False

        # potential-based shaping ------------------------------------
        if self.shaping in ("manhattan", "euclidean") and self.use_reward_shaping:
            def dist(s):
                if self.shaping == "manhattan":
                    return abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1])
                return math.hypot(s[0] - self.goal[0], s[1] - self.goal[1])
            reward += dist((r, c)) - dist(next_state)
        
        # nearby obstacles safety check ------------------------------
        if self.safety_nearby_obstacle:
            if next_state in self.nearby_obstacles_reward:
                reward += self.nearby_obstacles_reward[next_state]
            else:
                print(f"Warning: {next_state} not precomputed in nearby obstacles reward.")
        
        reward += -self.energy_consumption_gain * turn_angle / math.pi  # Penalty for turning

        return next_state, reward, done
    
    def precompute_nearby_obstacles_reward(self) -> None:
        """Precompute nearby obstacles for safety checks."""
        self.nearby_obstacles_reward = {}
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 1:
                    continue
                reward = 0.0
                for dr in range(-self.min_dist_nearby_obstacle, self.min_dist_nearby_obstacle + 1):
                    for dc in range(-self.min_dist_nearby_obstacle, self.min_dist_nearby_obstacle + 1):
                        if self.shaping == "manhattan":
                            distance = abs(dr) + abs(dc)
                        elif self.shaping == "euclidean":
                            distance = math.hypot(dr, dc)
                        else:
                            distance = math.hypot(dr, dc) # Euclidean distance default
                        
                        if distance > self.min_dist_nearby_obstacle:
                            continue

                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 1:
                            reward += distance
                self.nearby_obstacles_reward[(r, c)] = -reward * self.safety_nearby_obstacle_gain


# ---------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------
class QLearningAgent:
    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.9,
        schedule: str = "adaptive",           # 'adaptive' | 'exponential'
    ):
        self.env = env
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.min_eps, self.decay = epsilon, min_epsilon, epsilon_decay
        self.schedule = schedule
        self.Q = np.zeros((env.height, env.width, env.num_actions))

    def choose_action(self, state: Tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        r, c = state
        q = self.Q[r, c]
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done) -> None:
        r0, c0 = s
        r1, c1 = s2
        target = r if done else r + self.gamma * self.Q[r1, c1].max()
        self.Q[r0, c0, a] += self.alpha * (target - self.Q[r0, c0, a])

    def decay_epsilon(self, success_rate: float, threshold: float = 0.8, episodes: int = 5000, ep: int = 0) -> None:
        if self.schedule == "adaptive":
            if success_rate >= threshold and self.epsilon > self.min_eps:
                self.epsilon = max(self.min_eps, self.epsilon * self.decay)
        elif self.schedule == "mix":
            if ep/episodes < 0.8:
                self.epsilon = 0.95
            else:
                self.epsilon = max(self.min_eps, self.epsilon * self.decay)
        else:
            self.epsilon = max(self.min_eps, self.epsilon * self.decay)


# ---------------------------------------------------------------------
# Training & helpers
# ---------------------------------------------------------------------
def train(
    env,
    agent,
    episodes: int = 5000,
    window: int = 100,
    print_every: int = 100,
    threshold: float = 0.8,
    change_start_percentage: int = 0.2,
):
    rewards, successes, ep_durations = deque(maxlen=window), deque(maxlen=window), []
    t_start = time.perf_counter()

    if change_start_percentage > 1:
        print("Warning: change_start_percentage should be between 0 and 1. Setting to 0.")
        change_start_percentage = 0

    so = 0
    sn = 0

    for ep in range(1, episodes + 1):
        ep_begin = time.perf_counter()

        if change_start_percentage <= np.random.rand():
            s, tot, done = env.reset(), 0.0, False
            so += 1
        else:
            r, c = np.random.randint(0, env.height), np.random.randint(0, env.width)
            while not env.reset_new_position((r, c)):
                r, c = np.random.randint(0, env.height), np.random.randint(0, env.width)
            
            s, tot, done = env.agent_pos, 0.0, False
            sn += 1
            
        while not done:
            a = agent.choose_action(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s, tot = s2, tot + r

        # timing ------------------------------------------------------
        ep_time = time.perf_counter() - ep_begin
        ep_durations.append(ep_time)

        rewards.append(tot)
        successes.append(1 if done else 0)
        agent.decay_epsilon(float(np.mean(successes)), threshold, episodes, ep)

        if ep % print_every == 0:
            print(
                f"Ep {ep:5d} | AvgR={np.mean(rewards):7.2f} | "
                f"Succ={np.mean(successes)*100:5.1f}% | "
                f"ε={agent.epsilon:.3f} | "
                f"EpTime={ep_time:.3f}s"
            )

    total_time = time.perf_counter() - t_start
    print(f"\nTraining finished in {total_time:.2f} seconds "
          f"({total_time/episodes:.3f} s/episode on average).")

    print(f"Start changes: {(sn/episodes)*100:.1f}%, No start changes: {so/episodes*100:.1f}%")

    return list(ep_durations) 


def greedy_path(env: GridWorld, agent: QLearningAgent, limit: int = 1000):
    backup = agent.epsilon
    agent.epsilon = 0.0
    s, path = env.reset(), [env.start]
    for _ in range(limit):
        if s == env.goal:
            break
        a = agent.choose_action(s)
        s, _, done = env.step(a)
        path.append(s)
        if done:
            break
    agent.epsilon = backup
    return path

# ---------------------------------------------------------------------
# Combined visualisation
# ---------------------------------------------------------------------
def draw_q_heatmap(ax, env, agent):
    V = agent.Q.max(axis=2)  # best over actions
    cmap_val = "turbo" if "turbo" in plt.colormaps() else "plasma"
    V_mask = np.ma.masked_where(env.grid == 1, V)
    im = ax.imshow(V_mask, cmap=cmap_val, origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.imshow(np.ma.masked_where(env.grid == 0, env.grid),
              cmap="gray_r", origin="lower", vmin=0, vmax=1, alpha=1, interpolation="nearest")

    ax.scatter(env.start[1], env.start[0], marker="o", c="lime", s=100, zorder=5)
    ax.scatter(env.goal[1],  env.goal[0],  marker="*", c="red",  s=150, zorder=5)
    ax.set_title("State Values")
    ax.set_xticks([]); ax.set_yticks([])

def path_image(env: GridWorld, path):
    img = np.ones((env.height, env.width, 3))
    img[env.grid == 1] = (0, 0, 0)

    sr, sc = env.start
    gr, gc = env.goal
    img[sr, sc] = (0, 1, 0)   # start (green)
    img[gr, gc] = (1, 0, 0)   # goal  (red)

    for step in path:
        r, c = step[:2]  # supports (r,c) or (r,c,angle)
        if (r, c) not in ((sr, sc), (gr, gc)) and env.grid[r, c] == 0:
            img[r, c] = (0.5, 0.5, 1)
    return img

def combined_vis(env: GridWorld, agent: QLearningAgent, path):
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    # --- (0,0) Greedy path ------------------------------------------
    img = path_image(env, path)
    axs[0, 0].imshow(img, origin="lower", interpolation="nearest")
    axs[0, 0].set_xlim([-0.5, env.width-0.5])
    axs[0, 0].set_ylim([-0.5, env.height-0.5])
    axs[0, 0].set_xlabel("x (cols)")
    axs[0, 0].set_ylabel("y (rows)")
    axs[0, 0].set_title("Greedy Path (Cartesian view)")
    axs[0, 0].set_xticks(range(env.width))
    axs[0, 0].set_yticks(range(env.height))

    # --- (0,1) Q heatmap + greedy policy arrows ----------------------
    draw_q_heatmap(axs[0, 1], env, agent)

    # --- (1,0) Exploration (log) ------------------------------------
    max_lin = max(1, int(env.grid_explored.max()))
    draw_explored_heatmap(axs[1, 0], env, log_scale=True,
                          vmin=0, vmax=np.log10(max_lin + 1))

    # --- (1,1) Exploration (linear) ---------------------------------
    draw_explored_heatmap(axs[1, 1], env, log_scale=False,
                          vmin=0, vmax=max_lin)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Heat-map of exploration frequency
# ---------------------------------------------------------------------
def draw_explored_heatmap(
    ax,
    env: GridWorld,
    log_scale: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    # 1) Prepare data
    explored = env.grid_explored.astype(float)
    if log_scale:
        explored = np.log10(explored + 1)   # keeps zeros at 0

    # 2) Mask obstacles
    explored_mask = np.ma.masked_where(env.grid == 1, explored)

    # 3) Heatmap
    im = ax.imshow(
        explored_mask,
        cmap="inferno",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label="Visits (log₁₀)" if log_scale else "Visits")

    # 4) Obstacles + markers
    ax.imshow(
        np.ma.masked_where(env.grid == 0, env.grid),
        cmap="gray_r",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        alpha=1,
    )
    ax.scatter(env.start[1], env.start[0], marker="o", c="lime", s=100, zorder=5)
    ax.scatter(env.goal[1],  env.goal[0],  marker="*", c="red",  s=150, zorder=5)

    ax.set_title("Exploration Heat-map (log)" if log_scale else "Exploration Heat-map (linear)")
    ax.set_xticks([]); ax.set_yticks([])

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    width = 9
    height = 7

    obstacle_map = np.load("obs_grids/map3_paper.npy", allow_pickle=False)
    print("Obstacle map loaded successfully.")
    obstacle_map[:] = obstacle_map[::-1, :]

    # obstacle_map = obstacle_map.T
    print(obstacle_map)
    use_reward_shaping = True

    min_dist_nearby_obstacle = 3
    safety_nearby_obstacle_gain = 1
    energy_consumption_gain = 0.6
    reward_step = -0.1

    allow_diagonal_obstacle = False

    env = GridWorld(
        width=width,
        height=height,
        obstacle_density=0.22,
        obstacle_mode="cluster",
        grid_map=obstacle_map,
        seed=30,
        allow_diagonal=True,
        allow_diagonal_obstacle=allow_diagonal_obstacle,
        shaping="euclidean",
        reward_step=-0.001,
        safety_nearby_obstacle_gain=safety_nearby_obstacle_gain,
        min_dist_nearby_obstacle=min_dist_nearby_obstacle,
        energy_consumption_gain=energy_consumption_gain,
        use_reward_shaping=use_reward_shaping,
    )
    
    print(f"{env.grid}\n")

    agent = QLearningAgent(
        env,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.9,
        schedule="mix",
    )
    
    episodes = 400

    change_start_percentage = 1
    durations = train(env, agent, episodes=episodes, print_every=int(episodes/100), change_start_percentage=change_start_percentage)

    path = greedy_path(env, agent)
    # print(path)
    combined_vis(env, agent, path)
