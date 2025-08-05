#!/usr/bin/env python3

import math
import time
from collections import deque
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import hashlib

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
        start: Optional[Tuple[int, int, float]] = None,
        goal: Optional[Tuple[int, int, float]] = None,
        reward_goal: float = 100.0,
        reward_obstacle: float = -100.0,
        reward_step: float = -0.001,
        use_reward_shaping: bool = False,
        # None | 'manhattan' | 'euclidean'
        shaping: Optional[str] = "euclidean",
        allow_diagonal_obstacle: bool = False,
        allow_only_forward: bool = False,
        min_dist_nearby_obstacle: int = 3,
        safety_nearby_obstacle_gain: float = 0.0,
        energy_consumption_gain: float = 0.0,
    ):
        self.num_angles = 8

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

        self.start = (0, 0, self.angle_to_idx(0.0)) if start is None else (
            start[0], start[1], self.angle_to_idx(start[2]))
        self.goal = (self.height - 1, self.width - 1, self.angle_to_idx(0.0)
                     ) if goal is None else (goal[0], goal[1], self.angle_to_idx(goal[2]))
        self.grid[self.start[0:2]] = 0
        self.grid[self.goal[0:2]] = 0

        self.grid_explored = np.zeros(
            (self.height, self.width, self.num_angles), dtype=np.int32)
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
        self.safety_nearby_obstacle = (
            True if safety_nearby_obstacle_gain > 0 else False)
        self.min_dist_nearby_obstacle = min_dist_nearby_obstacle

        # Energy consumption
        self.energy_consumption_gain = abs(energy_consumption_gain)

        self.diag_cost = -abs(math.sqrt(2)*reward_step)
        self.allow_diag_obstacle = allow_diagonal_obstacle
        self.allow_only_forward = allow_only_forward

        self.angle = -1  # Angle initialization

        if self.safety_nearby_obstacle:
            self.precompute_nearby_obstacles_reward()

        if self.allow_only_forward:
            self.num_actions = 3
        else:
            self.num_actions = 4

        # Precompute action lookup table -----------------------------
        self._action_lookup = np.empty(
            (self.num_angles, self.num_actions, 3), dtype=int)
        for ang_idx in range(self.num_angles):
            for a in range(self.num_actions):
                self._action_lookup[ang_idx, a] = self._raw_action(ang_idx, a)

        self.agent_pos: Optional[Tuple[int, int, float]] = None

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
                dr = self.rng.randint(-self.cluster_size,
                                      self.cluster_size + 1)
                dc = self.rng.randint(-self.cluster_size,
                                      self.cluster_size + 1)
                r, c = center_r + dr, center_c + dc
                if 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] == 0:
                    self.grid[r, c] = 1
                    placed += 1

    # -----------------------------------------------------------------
    def reset(self) -> Tuple[int, int, int]:
        self.agent_pos = self.start
        self.angle = -1
        return self.agent_pos

    def reset_new_position(self, pos: Tuple[int, int, int]) -> bool:
        """Reset the agent to a new position."""
        if pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width:
            raise ValueError("Position out of bounds.")
        if self.grid[pos[0], pos[1]] == 1:
            return False
        self.agent_pos = pos
        self.angle = -1
        return True

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
                (rint(math.sin(ang)),  rint(math.cos(ang)),   0),
                (0, 0, +1),
                (0, 0, -1),]
        else:
            options = [
                (rint(math.sin(ang)),  rint(math.cos(ang)),   0),
                (-rint(math.sin(ang)),  -rint(math.cos(ang)),   0),
                (0, 0, +1),
                (0, 0, -1),]

        dr, dc, d_idx = options[action]
        new_idx = (ang_idx + d_idx) % self.num_angles
        return dr, dc, new_idx

    def action(self, ang_idx: int, a: int):
        dr, dc, new_idx = self._action_lookup[ang_idx, a]
        return int(dr), int(dc), int(new_idx)

    # --- in step() --------------------------------------------------
    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool]:
        r, c, ang_idx = self.agent_pos
        dr, dc, new_idx = self.action(ang_idx, action)
        nr, nc = r + dr, c + dc
        turn_angle = ((new_idx - ang_idx) % self.num_angles) * \
            (2*math.pi/self.num_angles)

        # invalid move (wall or obstacle)
        if not (0 <= nr < self.height and 0 <= nc < self.width) or self.grid[nr, nc] == 1:
            next_state = (r, c, ang_idx)
            reward, done = self.reward_obstacle, False
            return next_state, reward, done

        if not self.allow_diag_obstacle:
            corner1 = self.grid[nr,
                                c] if 0 <= nr < self.height and 0 <= c < self.width else 0
            corner2 = self.grid[r,
                                nc] if 0 <= r < self.height and 0 <= nc < self.width else 0
            if corner1 == 1 or corner2 == 1:
                next_state = (r, c, ang_idx)
                reward, done = self.reward_obstacle, False
                return next_state, reward, done

        next_state = (nr, nc, new_idx)
        self.agent_pos = next_state
        self.grid_explored[nr, nc, new_idx] += 1

        # initialize reward/done before adding step/shaping
        reward, done = 0.0, False
        if next_state == self.goal:
            reward, done = self.reward_goal, True

        # step penalty
        reward += self.reward_step

        # potential-based shaping
        if self.shaping in ("manhattan", "euclidean") and self.use_reward_shaping:
            if self.shaping == "manhattan":
                d0 = abs(r - self.goal[0]) + abs(c - self.goal[1])
                d1 = abs(nr - self.goal[0]) + abs(nc - self.goal[1])
            else:
                d0 = math.hypot(r - self.goal[0], c - self.goal[1])
                d1 = math.hypot(nr - self.goal[0], nc - self.goal[1])
            reward += (d0 - d1)

        # nearby obstacles safety (dict is keyed by (r,c), not angle)
        if self.safety_nearby_obstacle:
            reward += self.nearby_obstacles_reward.get((nr, nc), 0.0)

        # energy/turn penalty
        reward += -self.energy_consumption_gain * turn_angle / math.pi

        # Not moving penalty
        if dr == 0 and dc == 0:
            reward += self.reward_step

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
                            # Euclidean distance default
                            distance = math.hypot(dr, dc)

                        if distance > self.min_dist_nearby_obstacle:
                            continue

                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] == 1:
                            reward += distance
                self.nearby_obstacles_reward[(
                    r, c)] = -reward * self.safety_nearby_obstacle_gain


# ---------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------
class QLearningAgent:
    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        min_epsilon: float = 0.1,
        max_epsilon: float = 0.9,
        # first 30% of the train the epsilon value will be the max_epsilon
        max_epsilon_band: float = 0.3,
        # last 10% of the train the epsilon value will be the min_epsilon
        min_epsilon_band: float = 0.1,
    ):
        self.env = env
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.min_eps, self.max_eps = max_epsilon, min_epsilon, max_epsilon
        self.max_epsilon_band = max_epsilon_band
        self.min_epsilon_band = min_epsilon_band
        self.Q = np.zeros(
            (env.height, env.width, env.num_angles, env.num_actions))

    def choose_action(self, state: Tuple[int, int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        r, c, ang_idx = state
        q = self.Q[r, c, ang_idx]
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    def choose_action_bias(self, state, force_move=False) -> int:
        move_idxs = [0] if self.env.allow_only_forward else [0, 1]
        if force_move:
            # exploration still allowed, but among moves only
            if np.random.rand() < self.epsilon:
                return int(np.random.choice(move_idxs))
            r, c, k = state
            q = self.Q[r, c, k]
            return move_idxs[int(np.argmax(q[move_idxs]))]
        else:
            return self.choose_action(state)

    def update(self, s, a, r, s2, done) -> None:
        r0, c0, k0 = s
        r1, c1, k1 = s2
        target = r if done else r + self.gamma * self.Q[r1, c1, k1].max()
        self.Q[r0, c0, k0, a] += self.alpha * (target - self.Q[r0, c0, k0, a])

    def decay_epsilon(self, episodes: int = 5000, ep: int = 0) -> None:
        min_band = int(episodes * (1 - self.min_epsilon_band))
        max_band = int(episodes * self.max_epsilon_band)
        if ep < max_band:
            self.epsilon = self.max_eps
        elif ep < min_band:
            self.epsilon = self.max_eps - \
                ((self.max_eps - self.min_eps) /
                 (min_band - max_band)) * (ep - max_band)
        else:
            self.epsilon = self.min_eps
        pass

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
    max_steps: int = 1000,
    change_start_percentage: int = 0.2,
):
    rewards, successes, ep_durations = deque(
        maxlen=window), deque(maxlen=window), []
    t_start = time.perf_counter()

    if change_start_percentage > 1:
        print("Warning: change_start_percentage should be between 0 and 1. Setting to 0.")
        change_start_percentage = 0

    so = 0
    sn = 0
    group_success = []
    group_time_begin = time.perf_counter()
    ep_time_mean = []
    actions_per_episode = []
    consecutive_turns = []
    stuck_count = []
    for ep in range(1, episodes + 1):
        ep_begin = time.perf_counter()

        if change_start_percentage <= np.random.rand():
            s, tot, done = env.reset(), 0.0, False
            so += 1
        else:
            r, c = np.random.randint(
                0, env.height), np.random.randint(0, env.width)
            k = np.random.randint(env.num_angles)
            while not env.reset_new_position((r, c, k)):
                r, c = np.random.randint(
                    0, env.height), np.random.randint(0, env.width)
                k = np.random.randint(env.num_angles)

            s, tot, done = env.agent_pos, 0.0, False
            sn += 1

        steps = 0
        same_place = 0
        same_place_more_than_10 = 0
        force_move = False
        bump_streak = 0
        got_stuck = 0
        while not done and steps < max_steps:
            a = agent.choose_action_bias(s, force_move=force_move)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            if s2[0] == s[0] and s2[1] == s[1] and s2[2] != s[2]:
                # agent is in the same place but with different angle
                same_place += 1
            elif s2[0] == s[0] and s2[1] == s[1] and s2[2] == s[2]:
                # agent is in the same place and same angle
                bump_streak += 1
            else:
                bump_streak = 0
                same_place = 0
                force_move = False

            if same_place > 10:
                same_place_more_than_10 += 1
                force_move = True

            if bump_streak > 10 and force_move:
                # agent is stuck in the same place for too long, force move
                got_stuck += 1
                force_move = False

            s, tot = s2, tot + r

            steps += 1

        # timing ------------------------------------------------------
        ep_time = time.perf_counter() - ep_begin
        ep_time_mean.append(ep_time)
        ep_durations.append(ep_time)
        actions_per_episode.append(steps)
        consecutive_turns.append(same_place_more_than_10)
        stuck_count.append(got_stuck)

        rewards.append(tot)
        successes.append(1 if done else 0)
        group_success.append(1 if done else 0)
        agent.decay_epsilon(episodes, ep)

        if ep % print_every == 0:
            group_duration = time.perf_counter() - group_time_begin
            group_time_begin = time.perf_counter()
            print(
                f"Ep {ep:5d} | AvgR={np.mean(rewards):7.2f} | "
                f"ActionsMean={np.mean(actions_per_episode):5.1f} | "
                f"ActionsStDev={np.std(actions_per_episode):5.1f} | "
                f"Succ={np.mean(successes)*100:5.1f}% | "
                f"GroupSucc={np.mean(group_success)*100:5.1f}% | "
                f"ConsecutiveTurnsMean={np.mean(consecutive_turns):.1f} | "
                f"ConsecutiveTurnsStDev={np.std(consecutive_turns):.1f} | "
                f"StuckCountMean={np.mean(stuck_count):.1f} | "
                f"StuckCountStDev={np.std(stuck_count):.1f} | "
                f"ε={agent.epsilon:.3f} | "
                f"EpTime={np.mean(ep_time_mean):.3f}s | "
                f"GroupTime={group_duration:.3f}s | "
            )
            ep_time_mean.clear()
            group_success.clear()
            actions_per_episode.clear()
            consecutive_turns.clear()
            stuck_count.clear()

    total_time = time.perf_counter() - t_start
    print(f"\nTraining finished in {total_time:.2f} seconds "
          f"({total_time/episodes:.3f} s/episode on average).")

    print(
        f"Start changes: {(sn/episodes)*100:.1f}%, No start changes: {so/episodes*100:.1f}%")

    return list(ep_durations)


def train_adaptative(
    env,
    agent,
    window: int = 100,
    max_steps: int = 1000,
    inc_step: float = 0.02,
    success_window: int = 10,
    change_start_percentage: int = 0.2,
    success_countdown: int = 5,
):
    rewards, successes, ep_durations = deque(
        maxlen=window), deque(maxlen=window), []
    t_start = time.perf_counter()

    if change_start_percentage > 1:
        print("Warning: change_start_percentage should be between 0 and 1. Setting to 0.")
        change_start_percentage = 0

    so = 0
    sn = 0
    group_success = []
    max_group_success = 0.0
    max_group_success_countdown = 0
    group_time_begin = time.perf_counter()
    ep_time_mean = []
    actions_per_episode = []
    consecutive_turns = []
    stuck_count = []
    save_epsilon = []
    agent.epsilon = agent.max_eps
    save_epsilon.append(agent.epsilon)
    success_window_count = []
    metrics_rows = []
    ep = 0
    trained = False
    while not trained:
        ep += 1
        ep_begin = time.perf_counter()

        if change_start_percentage <= np.random.rand():
            s, tot, done = env.reset(), 0.0, False
            so += 1
        else:
            r, c = np.random.randint(
                0, env.height), np.random.randint(0, env.width)
            k = np.random.randint(env.num_angles)
            while not env.reset_new_position((r, c, k)):
                r, c = np.random.randint(
                    0, env.height), np.random.randint(0, env.width)
                k = np.random.randint(env.num_angles)

            s, tot, done = env.agent_pos, 0.0, False
            sn += 1

        steps = 0
        same_place = 0
        same_place_more_than_10 = 0
        force_move = False
        bump_streak = 0
        got_stuck = 0
        while not done and steps < max_steps:
            a = agent.choose_action_bias(s, force_move=force_move)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            if s2[0] == s[0] and s2[1] == s[1] and s2[2] != s[2]:
                # agent is in the same place but with different angle
                same_place += 1
            elif s2[0] == s[0] and s2[1] == s[1] and s2[2] == s[2]:
                # agent is in the same place and same angle
                bump_streak += 1
            else:
                bump_streak = 0
                same_place = 0
                force_move = False

            if same_place > 10:
                same_place_more_than_10 += 1
                force_move = True

            if bump_streak > 10 and force_move:
                # agent is stuck in the same place for too long, force move
                got_stuck += 1
                force_move = False

            s, tot = s2, tot + r

            steps += 1

        success_window_count.append(1 if done else 0)

        # timing ------------------------------------------------------
        ep_time = time.perf_counter() - ep_begin
        ep_time_mean.append(ep_time)
        ep_durations.append(ep_time)
        actions_per_episode.append(steps)
        consecutive_turns.append(same_place_more_than_10)
        stuck_count.append(got_stuck)

        rewards.append(tot)
        successes.append(1 if done else 0)
        group_success.append(1 if done else 0)
        # agent.decay_epsilon(episodes, ep) # No decay in adaptative training

        if ep % success_window == 0:
            group_duration = time.perf_counter() - group_time_begin
            group_time_begin = time.perf_counter()
            group_success_mean = np.mean(group_success)
            print(
                f"Ep {ep:5d} | AvgR={np.mean(rewards):7.2f} | "
                f"ActionsMean={np.mean(actions_per_episode):5.1f} | "
                f"ActionsStDev={np.std(actions_per_episode):5.1f} | "
                f"Succ={np.mean(successes)*100:5.1f}% | "
                f"GroupSucc={group_success_mean*100:5.1f}% | "
                f"ConsecutiveTurnsMean={np.mean(consecutive_turns):.1f} | "
                f"ConsecutiveTurnsStDev={np.std(consecutive_turns):.1f} | "
                f"StuckCountMean={np.mean(stuck_count):.1f} | "
                f"StuckCountStDev={np.std(stuck_count):.1f} | "
                f"ε={agent.epsilon:.3f} | "
                f"EpTime={np.mean(ep_time_mean):.3f}s | "
                f"GroupTime={group_duration:.3f}s | "
            )

            if group_success_mean > max_group_success:
                max_group_success = group_success_mean
                max_group_success_countdown -= 2
            else:
                max_group_success_countdown += 1

            if max_group_success_countdown > success_countdown:
                agent.epsilon = max(agent.min_eps, agent.epsilon - inc_step)

            row = [
                ep,
                float(np.mean(rewards)),
                float(np.mean(actions_per_episode)),
                float(np.std(actions_per_episode)),
                float(np.mean(successes) * 100.0),
                float(np.mean(group_success) * 100.0),
                float(np.mean(consecutive_turns)),
                float(np.std(consecutive_turns)),
                float(np.mean(stuck_count)),
                float(np.std(stuck_count)),
                float(agent.epsilon),
                float(np.mean(ep_time_mean)),
                float(group_duration),
            ]
            metrics_rows.append(row)

            ep_time_mean.clear()
            group_success.clear()
            actions_per_episode.clear()
            consecutive_turns.clear()
            stuck_count.clear()

            # break loop if epsilon reached minimum
            if agent.epsilon <= agent.min_eps:
                trained = True
                print(
                    f"Training finished at episode {ep} with epsilon {agent.epsilon:.3f}.")

    total_time = time.perf_counter() - t_start
    print(f"\nTraining finished in {total_time:.2f} seconds "
          f"({total_time/ep:.3f} s/episode on average).")

    print(
        f"Start changes: {(sn/ep)*100:.1f}%, No start changes: {so/ep*100:.1f}%")
    metrics_mat = np.asarray(metrics_rows, dtype=np.float32)

    return list(ep_durations), save_epsilon, metrics_mat


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
    V = agent.Q.max(axis=3).max(axis=2)  # best over actions, then over angles
    cmap_val = "turbo" if "turbo" in plt.colormaps() else "plasma"
    V_mask = np.ma.masked_where(env.grid == 1, V)
    im = ax.imshow(V_mask, cmap=cmap_val, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.imshow(np.ma.masked_where(env.grid == 0, env.grid),
              cmap="gray_r", origin="lower", vmin=0, vmax=1, alpha=1)

    ax.scatter(env.start[1], env.start[0],
               marker="o", c="lime", s=100, zorder=5)
    ax.scatter(env.goal[1],  env.goal[0],
               marker="*", c="red",  s=150, zorder=5)
    ax.set_title("State Values")
    ax.set_xticks([])
    ax.set_yticks([])


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
        # NO minus now (we'll use origin='lower')
        vs.append(math.sin(ang)*scale)
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


def save_artifacts(path: str, env, agent, note: str = "", train_metrics: np.ndarray = None):
    def _sha1(arr):
        import hashlib
        h = hashlib.sha1()
        h.update(arr.tobytes())
        return h.hexdigest()

    np.savez_compressed(
        path,
        grid=env.grid.astype(np.uint8),
        goal=np.array(env.goal, dtype=np.int16),
        start_train=np.array(env.start, dtype=np.int16),
        Q=agent.Q.astype(np.float32),
        action_table=env._action_lookup.astype(np.int8),
        num_angles=np.int16(env.num_angles),
        num_actions=np.int16(env.num_actions),
        allow_only_forward=np.uint8(env.allow_only_forward),
        allow_diagonal_obstacle=np.uint8(env.allow_diag_obstacle),
        # store strings as fixed-length unicode (safe with allow_pickle=False)
        grid_sha1=np.array([_sha1(env.grid)], dtype="U64"),
        note=np.array([note], dtype="U512"),
        train_metrics=(train_metrics.astype(
            np.float32) if train_metrics is not None else np.zeros((0, 13), np.float32)),
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    width = 9
    height = 7

    obstacle_map = np.load("obs_grids/map3_paper.npy", allow_pickle=False)
    print("Obstacle map loaded successfully.")
    obstacle_map[:] = obstacle_map[::-1, :]

    use_reward_shaping = True

    min_dist_nearby_obstacle = 3
    safety_nearby_obstacle_gain = 0
    energy_consumption_gain = 0
    reward_step = -0.1

    allow_diagonal_obstacle = False
    only_forward = False

    env = GridWorld(
        width=width,
        height=height,
        obstacle_density=0.22,
        obstacle_mode="cluster",
        grid_map=obstacle_map,
        seed=30,
        allow_diagonal_obstacle=allow_diagonal_obstacle,
        allow_only_forward=only_forward,
        shaping="euclidean",
        reward_step=-0.001,
        safety_nearby_obstacle_gain=safety_nearby_obstacle_gain,
        min_dist_nearby_obstacle=min_dist_nearby_obstacle,
        energy_consumption_gain=energy_consumption_gain,
        use_reward_shaping=use_reward_shaping,
    )

    print(f"{env.grid}\n")

    max_epsilon_band = 0.7
    min_epsilon_band = 0.1
    max_epsilon = 0.9
    min_epsilon = 0.05
    agent = QLearningAgent(
        env,
        alpha=0.1,
        gamma=0.99,
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        max_epsilon_band=max_epsilon_band,
        min_epsilon_band=min_epsilon_band,
    )

    episodes = 2500

    change_start_percentage = 0.7
    max_steps = env.height * env.width * 2
    inc_step = 0.01

    # durations = train(env, agent, episodes=episodes, print_every=int(episodes/100), change_start_percentage=change_start_percentage, max_steps=max_steps)

    durations, save_epsilon, matrix_mat = train_adaptative(
        env, agent, success_window=200, change_start_percentage=change_start_percentage, max_steps=max_steps, inc_step=inc_step)
    # save_epsilon = np.array(save_epsilon)
    # Save epsilon to a file
    # np.save("epsilon_decay.npy", save_epsilon)

    # plot the epsilons
    # plt.figure(figsize=(10, 5))
    # plt.plot(save_epsilon, label="Epsilon", color="blue")
    # plt.xlabel("Episode")
    # plt.ylabel("Epsilon")
    # plt.title("Epsilon Decay Over Episodes")
    # plt.grid()
    # plt.legend()
    # plt.show()

    path = greedy_path(env, agent)
    # print(path)
    combined_vis(env, agent, path)
    save_artifacts("agent.npz", env, agent,
                   note="Q+actions with map/goal/start", train_metrics=matrix_mat)
