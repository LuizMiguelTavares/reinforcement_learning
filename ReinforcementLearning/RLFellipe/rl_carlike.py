#!/usr/bin/env python3

import math, time
from collections import deque
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# GridWorld environment
# ---------------------------------------------------------------------
class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float = 0.2,
        obstacle_mode: str = "cluster",
        grid_map: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        cluster_size: int = 5,
        start: Optional[Tuple[int, int, float]] = None,
        goal: Optional[Tuple[int, int, float]] = None,
        allow_diagonal: bool = False,
        diagonal_cost: Optional[float] = None,
        reward_goal: float = 100.0,
        reward_obstacle: float = -10.0,
        reward_step: float = -0.001,
        shaping: Optional[str] = "euclidean",
        allow_diagonal_obstacle: bool = False,
        safety_nearby_obstacle: bool = True,
        allow_only_forward: bool = False,
        min_dist_nearby_obstacle: int = 2,
        safety_nearby_obstacle_gain: float = 0.6,
        energy_consumption_gain: float = 0.6,
        backward_penalty: float = 0.6
    ):
        self.width, self.height = width, height
        self.obstacle_density = obstacle_density
        self.obstacle_mode = obstacle_mode
        self.num_angles = 8  # Number of angles for car-like robot
        self.cluster_size = cluster_size
        self.rng = np.random.RandomState(seed)
        self.reward_goal = reward_goal
        self.reward_obstacle = reward_obstacle
        self.reward_step = reward_step
        self.shaping = shaping
        self.allow_diag = allow_diagonal
        self.diag_cost = (
            diagonal_cost if diagonal_cost is not None
            else (math.sqrt(2) if allow_diagonal else 1.0)
        )
        self.allow_diag_obstacle = allow_diagonal_obstacle
        self.safety_nearby_obstacle = safety_nearby_obstacle
        self.min_dist_nearby_obstacle = min_dist_nearby_obstacle
        self.safety_nearby_obstacle_gain = safety_nearby_obstacle_gain
        self.energy_consumption_gain = energy_consumption_gain
        self.backward_penalty = backward_penalty
        self.allow_only_forward = allow_only_forward

        # Build grid --------------------------------------------------
        if grid_map is not None:
            self.grid = grid_map.copy()
            self.width, self.height = self.grid.shape[1], self.grid.shape[0]
        else:
            self.grid = np.zeros((height, width), dtype=np.int8)
            self._populate_obstacles()

        self.start = (0, 0, self.angle_to_idx(0.0)) if start is None else (start[0], start[1], self.angle_to_idx(start[2]))
        self.goal  = (height - 1, width - 1, self.angle_to_idx(0.0)) if goal is None else (goal[0], goal[1], self.angle_to_idx(goal[2]))
        self.grid[self.start[0:2]] = 0
        self.grid[self.goal[0:2]] = 0

        if self.safety_nearby_obstacle:
            self.precompute_nearby_obstacles_reward()

        if self.allow_only_forward:
            self.num_actions = 5
        else:
            self.num_actions = 10

        self.invalid_action_buffer = 0

        # Precompute action lookup table -----------------------------
        self._action_lookup = np.empty((self.num_angles, self.num_actions, 3), dtype=int)
        for ang_idx in range(self.num_angles):
            for a in range(self.num_actions):
                self._action_lookup[ang_idx, a] = self._raw_action(ang_idx, a)

        self.agent_pos: Optional[Tuple[int, int, float]] = None

    def angle_to_idx(self, ang: float) -> int:
        step = 2 * math.pi / self.num_angles
        return int(((ang + math.pi) % (2 * math.pi)) // step)

    def idx_to_angle(self, idx: int) -> float:
        step = 2 * math.pi / self.num_angles
        return -math.pi + idx * step

    def _raw_action(self, ang_idx: int, action: int):
        ang = self.idx_to_angle(ang_idx)

        def rint(x):
            return int(np.round(x))

        options = []

        if self.allow_only_forward:
            options = [
                (rint(math.sin(ang + math.pi/4)),  rint(math.cos(ang + math.pi/4)),   +1),
                (rint(math.sin(ang)),              rint(math.cos(ang)),               +1),
                (rint(math.sin(ang)),              rint(math.cos(ang)),                0),
                (rint(math.sin(ang)),              rint(math.cos(ang)),               -1),
                (rint(math.sin(ang - math.pi/4)),  rint(math.cos(ang - math.pi/4)),   -1),]
        else:
            options = [
                (rint(math.sin(ang + math.pi/4)),  rint(math.cos(ang + math.pi/4)),   +1),
                (rint(math.sin(ang)),              rint(math.cos(ang)),               +1),
                (rint(math.sin(ang)),              rint(math.cos(ang)),                0),
                (rint(math.sin(ang)),              rint(math.cos(ang)),               -1),
                (rint(math.sin(ang - math.pi/4)),  rint(math.cos(ang - math.pi/4)),   -1),
                (-rint(math.sin(ang + math.pi/4)), -rint(math.cos(ang + math.pi/4)),  +1),
                (-rint(math.sin(ang)),             -rint(math.cos(ang)),              +1),
                (-rint(math.sin(ang)),             -rint(math.cos(ang)),               0),
                (-rint(math.sin(ang)),             -rint(math.cos(ang)),              -1),
                (-rint(math.sin(ang - math.pi/4)), -rint(math.cos(ang - math.pi/4)),  -1),]

        dr, dc, d_idx = options[action]
        new_idx = (ang_idx + d_idx) % self.num_angles
        return dr, dc, new_idx

    def action(self, ang_idx: int, action: int):
        return tuple(self._action_lookup[ang_idx, action])

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
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
    
    def return_pose(self) -> Tuple[int, int, float]:
        """Return the current position of the agent."""
        if self.agent_pos is None:
            raise ValueError("Agent position is not set. Call reset() first.")
        return self.agent_pos

    def set_new_start(self, pos: Tuple[int, int, int]) -> None:
        """Set a new starting position for the agent."""
        if not (0 <= pos[0] < self.height and 0 <= pos[1] < self.width):
            raise ValueError("Starting position out of bounds.")
        if self.grid[pos[0], pos[1]] == 1:
            raise ValueError("Starting position cannot be on an obstacle.")
        self.agent_pos = (pos[0], pos[1], pos[2])
        self.start = (pos[0], pos[1], pos[2])

    def is_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Check if the given position is an obstacle."""
        r, c = pos
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return self.grid[r, c] == 1

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool]:
        r, c, ang_idx = self.agent_pos
        dr, dc, new_idx = self.action(ang_idx, action)
        nr, nc = r + dr, c + dc
        turn_angle = ((new_idx - ang_idx) % self.num_angles) * (2*math.pi/self.num_angles)

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
                next_state = (r, c, ang_idx)
                reward, done = self.reward_obstacle, False
                return next_state, reward, done

        # invalid move (wall or obstacle) -----------------------------
        if not (0 <= nr < self.height and 0 <= nc < self.width) or self.grid[nr, nc] == 1:
            next_state = (r, c, ang_idx)
            reward, done = self.reward_obstacle, False
            self.invalid_action_buffer += 1
            return next_state, reward, done
        else:
            next_state = (nr, nc, new_idx)
            self.agent_pos = next_state
            if next_state == self.goal:
                reward, done = self.reward_goal, True
            else:
                step_pen = (
                    -abs(self.diag_cost)
                    if self.allow_diag and dr and dc and self.reward_step < 0
                    else self.reward_step
                )
                reward, done = step_pen, False

        # potential-based shaping ------------------------------------
        # if self.shaping in ("manhattan", "euclidean"):
        #     def dist(s):
        #         if self.shaping == "manhattan":
        #             return abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1])
        #         return math.hypot(s[0] - self.goal[0], s[1] - self.goal[1])
        #     reward += dist((r, c)) - dist(next_state[0:2])

        # nearby obstacles safety check ------------------------------
        if self.safety_nearby_obstacle:
            if (next_state[0], next_state[1]) in self.nearby_obstacles_reward:
                reward += self.nearby_obstacles_reward[next_state[0:2]]

        if action >= 5:
            reward += -self.backward_penalty  # Penalty for moving backward

        reward += -self.energy_consumption_gain * turn_angle / math.pi  # Penalty for turning

        self.invalid_action_buffer = 0

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
        self.Q = np.zeros((env.height, env.width, env.num_angles, env.num_actions))

    def choose_action(self, state):
        r, c, ang_idx = state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        q = self.Q[r, c, ang_idx]
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        r0, c0, a0 = s
        r1, c1, a1 = s2
        target = r if done else r + self.gamma * self.Q[r1, c1, a1].max()
        self.Q[r0, c0, a0, a] += self.alpha * (target - self.Q[r0, c0, a0, a])

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
    change_beginning: bool = True,
    change_beginning_value: int = 10,
):
    rewards, successes, ep_durations = deque(maxlen=window), deque(maxlen=window), []
    t_start = time.perf_counter()
    ep_count = 0

    for ep in range(1, episodes + 1):
        ep_begin = time.perf_counter()
        if ep_count >=  change_beginning_value and change_beginning:
            # Find a new random start
            is_obstacle = True
            while is_obstacle:
                start_row = np.random.randint(0, env.height)
                start_col = np.random.randint(0, env.width)
                start_angle = np.random.randint(0, env.num_angles)
                is_obstacle = env.is_obstacle((start_row, start_col))

            env.set_new_start((start_row, start_col, start_angle))
            s, tot, done = env.return_pose(), 0.0, False
        else:
            s, tot, done = env.reset(), 0.0, False

        while not done:
            a = agent.choose_action(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s, tot = s2, tot + r

            if env.invalid_action_buffer > 3*env.num_actions:
                # print("Failed!")
                break

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
            rewards.clear()
            successes.clear()
        ep_count += 1

    total_time = time.perf_counter() - t_start
    print(f"\nTraining finished in {total_time:.2f} seconds "
          f"({total_time/episodes:.3f} s/episode on average).")

    return list(ep_durations)

def greedy_path(env: GridWorld, agent: QLearningAgent, limit: int = 1000, pos=None):
    backup = agent.epsilon
    agent.epsilon = 0.0

    if pos is None:
        s = env.reset()
        path = [env.start]
    else:
        r, c, ang = pos
        s = (int(r), int(c),
             env.angle_to_idx(ang) if isinstance(ang, float) else int(ang))
        env.agent_pos = s
        path = [s]

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
    V = agent.Q.max(axis=3).max(axis=2)  # best over actions, then over angles
    cmap_val = "turbo" if "turbo" in plt.colormaps() else "plasma"
    V_mask = np.ma.masked_where(env.grid == 1, V)
    im = ax.imshow(V_mask, cmap=cmap_val, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.imshow(np.ma.masked_where(env.grid == 0, env.grid),
              cmap="gray_r", origin="lower", vmin=0, vmax=1, alpha=1)

    ax.scatter(env.start[1], env.start[0], marker="o", c="lime", s=100, zorder=5)
    ax.scatter(env.goal[1],  env.goal[0],  marker="*", c="red",  s=150, zorder=5)
    ax.set_title("State Values")
    ax.set_xticks([]); ax.set_yticks([])

def path_image(env: GridWorld, path):
    img = np.ones((env.height, env.width, 3))
    img[env.grid == 1] = (0, 0, 0)

    sr, sc = env.start[0], env.start[1]
    gr, gc = env.goal[0],  env.goal[1]
    img[sr, sc] = (0, 1, 0)   # start (green)
    img[gr, gc] = (1, 0, 0)   # goal  (red)

    for r, c, _ in path:
        if (r, c) not in ((sr, sc), (gr, gc)) and env.grid[r, c] == 0:
            img[r, c] = (0.5, 0.5, 1)
    return img

def draw_path_with_angles(ax, env: GridWorld, path, every=1, scale=0.4):
    xs, ys, us, vs = [], [], [], []
    for r, c, ang_idx in path[::every]:
        ang = env.idx_to_angle(ang_idx)
        xs.append(c)                 # x = column
        ys.append(r)                 # y = row
        us.append(math.cos(ang)*scale)
        vs.append(math.sin(ang)*scale)   # NO minus now (we'll use origin='lower')
    ax.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=1,
              width=0.01, color='yellow', zorder=6)

def combined_vis(env: GridWorld, agent: QLearningAgent, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: path + angles, Cartesian (0,0 bottom-left) ---
    img = path_image(env, path)
    ax1.imshow(img, origin="lower", interpolation='nearest')  # <- key change
    draw_path_with_angles(ax1, env, path, every=1, scale=0.35)

    ax1.set_xlim([-0.5, env.width-0.5])
    ax1.set_ylim([-0.5, env.height-0.5])
    ax1.set_xlabel("x (cols)")
    ax1.set_ylabel("y (rows)")
    ax1.set_title("Greedy Path + Orientation (Cartesian view)")
    ax1.set_xticks(range(env.width))
    ax1.set_yticks(range(env.height))

    # --- Right: heatmap (also Cartesian) ---
    draw_q_heatmap(ax2, env, agent)  # inside, also use origin='lower'
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    width = 30
    height = 30

    only_forward = False 
    # importing a grid world from pickle file
    obstacle_map = np.load("obs_grids/map2_paper.npy", allow_pickle=False)
    print("Obstacle map loaded successfully.")
    obstacle_map[:] = obstacle_map[::-1, :]

    # obstacle_map = obstacle_map.T
    print(obstacle_map)

    # Map 1
    # start = (3, 2, math.pi/2)  # Starting position (y, x, angle)
    # goal = (5, 4, math.pi/2)  # Goal position (y, x, angle)

    # Map 2
    # start = (2, 0, -math.pi/2)  # Starting position (y, x, angle)
    # goal = (2, 2, -math.pi)   # Goal position (y, x, angle)

    # Map2 2
    # goal = (1, 2, -math.pi)   # Goal position (y, x, angle)

    # Maps paper
    start = (0, 0, math.pi/4)  # Starting position (y, x, angle)
    goal = (12, 12, math.pi/4)   # Goal position (y, x, angle)

    env = GridWorld(
        width=width,
        height=height,
        obstacle_density=0.22,
        obstacle_mode="cluster",
        grid_map=obstacle_map,
        start=start,
        goal=goal,
        seed=40,
        allow_diagonal=True,
        allow_only_forward=only_forward,
        shaping="euclidean",
        reward_step=-0.001,
        safety_nearby_obstacle_gain=0.0,
        energy_consumption_gain=0.6,
        backward_penalty=2,
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

    episodes = 500
    change_beginning_value = 10
    change_beginning = False ### Está com problema de bug quando muda o início

    durations = train(env, agent, episodes=episodes, print_every=int(episodes/100), change_beginning=change_beginning, change_beginning_value=change_beginning_value)

    path = greedy_path(env, agent)
    # pos = (1, 7, -math.pi/2)

    # path2 = greedy_path(env, agent, pos=pos)
    # print(path)
    combined_vis(env, agent, path)
    # combined_vis(env, agent, path2)