import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import time


class Environment:
    def __init__(self,
                 grid_size: tuple[int, int],
                 start: Optional[tuple[int, int]] = None,
                 goal: Optional[tuple[int, int]] = None,
                 action_type: str = '8d',
                 use_obstacles: bool = True,
                 obstacle_ratio: float = 0.2,
                 seed: int = 24,
                 obstacle_map: Optional[np.ndarray] = None,
                 ) -> None:

        # Action Settings
        self.actions, self.actions_symbols = self._define_actions(action_type)
        self.n_actions: int = len(self.actions)

        # Grid Settings
        self.grid_size: tuple[int, int] = grid_size
        self.n_states: int = grid_size[0] * grid_size[1]
        self.Q: np.ndarray = np.zeros((self.n_states, self.n_actions))

        # Task Settings
        self.start: tuple[int, int] = start if start is not None else (0, 0)
        self.goal: tuple[int, int] = goal if goal is not None else (
            grid_size[0] - 1, grid_size[1] - 1)

        # Obstacles
        self.seed = seed

        if use_obstacles:
            if obstacle_map is not None:
                self.load_obstacle_map(obstacle_map)
            else:
                self.obstacles = self._generate_obstacles(obstacle_ratio)
        else:
            self.obstacles = None

    def _step(self, state, position, action):
        dx, dy = self.actions[action]
        next_pos = (position[0] + dx, position[1] + dy)
        valid = self._valid_pos(position, next_pos)

        reward = self._calculate_reward(next_pos, action, valid)
        finish = ((next_pos == self.goal)
                  or not valid) if self.kill_on_invalid else next_pos == self.goal
        next_state = next_pos[0] * self.grid_size[1] + \
            next_pos[1] if valid else state

        return next_state, (next_pos if valid else position), reward, finish

    def _valid_pos(self, pr_position: tuple[int, int], position: tuple[int, int]) -> bool:

        # Out of bounds check
        inside = 0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]
        if not inside:
            return False

        if self.obstacles is not None:
            # Colision check
            if self.obstacles[position]:
                return False

            # Diagonal corner check
            pr_row, pr_col = pr_position
            new_row, new_col = position

            corner1_is_obstacle = self.obstacles[pr_row, new_col]
            corner2_is_obstacle = self.obstacles[new_row, pr_col]

            if corner1_is_obstacle or corner2_is_obstacle:
                return False

        return True

    def _calculate_reward(self, position: tuple[int, int], action: int, valid: bool) -> float:

        if not valid:
            return -100.0

        if position == self.goal:
            return 100.0

        if self.reward_type == 'constant':
            return -0.1

        elif self.reward_type == 'goal_distance':
            distance = np.linalg.norm(
                np.array(position) - np.array(self.goal))
            return -0.1 * distance

        elif self.reward_type == 'min_mov':
            mov = np.linalg.norm(self.actions[action])
            return -0.1 * mov
        elif self.reward_type == 'v1':
            rd = -0.1 * np.linalg.norm(
                np.array(position) - np.array(self.goal))

            if self.obstacles is not None:
                obs_pos = np.argwhere(self.obstacles)
                if obs_pos.size == 0:
                    return 0.0

                dists = np.linalg.norm(obs_pos - np.array(position), axis=1)
                nearest = np.partition(dists, 8)[:8]

                ro = np.sum(np.minimum(0,
                                       -1 * (4 - nearest)))

                reward = ro + rd
                return reward

            return rd

        else:
            raise ValueError(f"Invalid reward type.")

    def _egreedy_update(self) -> None:
        # Update epsilon
        if self.epsilon > self.epsilonf:
            if self.egreedy_type == 'linear':
                self.epsilon = self.epsilon - \
                    ((self.epsilon0 - self.epsilonf) /
                     (self.n_episodes * self.epsilonf_percentage))
            elif self.egreedy_type == 'exponential':
                decay_ratio = (self.epsilonf/self.epsilon0)**(1 /
                                                              (self.n_episodes * self.epsilonf_percentage))
                self.epsilon *= decay_ratio
            else:
                raise ValueError("Invalid egreedy type.")

        self.epsilon = max(self.epsilonf, self.epsilon)

    def _define_actions(self, action_type: str) -> tuple[np.ndarray, tuple[str, ...]]:
        action_dict = {
            '4d': np.array([(-1, 0), (0, 1), (1, 0), (0, -1)]),  # 4 directions
            '8d': np.array(
                [(-1, 0), (0, 1), (1, 0), (0, -1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
            ),  # 8 directions
        }
        symbols_dict = {
            '4d': ("↑", "→", "↓", "←"),
            '8d': ("↑", "→", "↓", "←", "↖", "↗", "↙", "↘"),
        }

        return action_dict[action_type], symbols_dict[action_type]

    def _generate_obstacles(self, obstacle_ratio: float) -> np.ndarray:
        obstacle_rng = np.random.default_rng(self.seed)
        mask = obstacle_rng.random(self.grid_size) < obstacle_ratio
        mask[self.start] = False
        mask[self.goal] = False
        return mask

    def training(self,
                 n_episodes: int = 1000,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon0: float = 1.0,
                 epsilonf: float = 0.1,
                 epsilonf_percentage: float = 0.8,
                 egreedy_type: str = 'linear',
                 reward_type: str = 'constant',
                 kill_on_invalid: bool = True,
                 verbose_interval: int = 1000
                 ) -> None:

        self.epsilon = epsilon0
        self.epsilon0 = epsilon0
        self.epsilonf = epsilonf
        self.n_episodes = n_episodes
        self.egreedy_type = egreedy_type
        self.epsilonf_percentage = epsilonf_percentage
        self.reward_type = reward_type
        self.kill_on_invalid = kill_on_invalid

        self.epsilon_history = []

        pr_Q_mean = self.mean_q_value

        time_start = time.time()
        for episode in range(self.n_episodes):
            position = self.start
            state = self.start[0] * self.grid_size[1] + self.start[1]
            finish = False

            self._egreedy_update()
            self.epsilon_history.append(self.epsilon)

            while not finish:
                # Action selection
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.n_actions)
                else:
                    action = np.argmax(self.Q[state])

                next_state, next_position, reward, finish = self._step(
                    state, position, action)

                best_next_action = np.argmax(self.Q[next_state])
                updated_q_value = (1-alpha) * self.Q[state][action] + alpha * (
                    reward + gamma * self.Q[next_state][best_next_action])
                self.Q[state][action] = updated_q_value

                state = next_state
                position = next_position

            dif_mean_q_value = (self.mean_q_value - pr_Q_mean)
            pr_Q_mean = self.mean_q_value

            # Logging the episode details
            if verbose_interval > 0 and episode % verbose_interval == 0:

                print(f"Episode: {episode:5d} | Mean QValue: {self.mean_q_value:.6f} | "
                      f"Dif Mean QValue: {dif_mean_q_value:.6f} | Epsilon: {self.epsilon:.2f}")

            if 0 < dif_mean_q_value < 1e-5 and episode > 0:
                print("Convergence reached.")
                break

        time_end = time.time()
        print(f"Training completed in {time_end - time_start:.2f} seconds.")

    @property
    def policy(self) -> np.ndarray:
        policy = np.full(self.grid_size, "", dtype=object)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state_index = i * self.grid_size[1] + j
                best_action = np.argmax(self.Q[state_index])
                policy[i, j] = self.actions_symbols[best_action]

        policy[self.start] = "S"
        policy[self.goal] = "G"
        if self.obstacles is not None:
            policy[self.obstacles] = "X"

        return policy

    @property
    def mean_q_value(self) -> float:
        return np.mean(self.Q)

    def plot_epsilon_decay(self) -> None:

        if not self.epsilon_history:
            print("Epsilon history is empty. Please run training first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.epsilon_history)),
                 self.epsilon_history, label='Epsilon Value')
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_policy(self, figsize: tuple[int, int] = (6, 6)) -> None:
        rows, cols = self.grid_size
        # Compute centers of each cell
        X, Y = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
        U = np.zeros((rows, cols))
        V = np.zeros((rows, cols))
        # Build vector field
        for r in range(rows):
            for c in range(cols):
                if (r, c) == self.start or (r, c) == self.goal or (self.obstacles is not None and self.obstacles[r, c]):
                    U[r, c] = 0
                    V[r, c] = 0
                else:
                    idx = r * cols + c
                    a = int(np.argmax(self.Q[idx]))
                    dx, dy = self.actions[a]
                    U[r, c] = dy
                    V[r, c] = dx

        fig, ax = plt.subplots(figsize=figsize)
        # Draw grid lines
        ax.set_xticks(np.arange(cols + 1))
        ax.set_yticks(np.arange(rows + 1))
        ax.grid(True)
        # Plot arrows at cell centers
        ax.quiver(X, Y, U, V, pivot='middle', angles='xy', scale_units='xy')
        # Plot start, goal and obstacles at cell centers
        ax.scatter(self.start[1] + 0.5, self.start[0] +
                   0.5, marker='o', s=100, label='Start')
        ax.scatter(self.goal[1] + 0.5, self.goal[0] +
                   0.5, marker='*', s=100, label='Goal')
        if self.obstacles is not None:
            obs = np.argwhere(self.obstacles)
            ax.scatter(obs[:, 1] + 0.5, obs[:, 0] + 0.5,
                       marker='s', s=100, label='Obstacle')

        # --- Aqui começa a rota ótima ---
        path = [self.start]
        pos = self.start
        state = pos[0] * cols + pos[1]
        max_steps = 2 * rows * cols
        for _ in range(max_steps):
            a = int(np.argmax(self.Q[state]))
            _, next_pos, _, finish = self._step(state, pos, a)
            path.append(next_pos)
            pos = next_pos
            state = pos[0] * cols + pos[1]
            if finish:
                break
        # extrai coordenadas e plota linha vermelha
        xs = [c + 0.5 for (r, c) in path]
        ys = [r + 0.5 for (r, c) in path]
        ax.plot(xs, ys, 'r-', linewidth=2)
        # --- Fim da rota ótima ---

        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title('Policy visualization')
        plt.tight_layout()
        plt.show()

    def save(self, file: Union[str, Path]):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def evaluate_policy(self,
                        start_positions: Optional[list[tuple[int, int]]] = None,
                        max_steps: Optional[int] = None
                        ) -> dict[tuple[int, int], bool]:

        successes: dict[tuple[int, int], bool] = {}

        # monta lista de starts se não fornecida
        if start_positions is None:
            n_rows, n_cols = self.grid_size
            start_positions = [
                (i, j)
                for i in range(n_rows) for j in range(n_cols)
                if not self.obstacles[i, j] and (i, j) != self.goal
            ] if self.obstacles is not None else [
                (i, j)
                for i in range(self.grid_size[0])
                for j in range(self.grid_size[1])
                if (i, j) != self.goal
            ]

        # define limite de passos
        if max_steps is None:
            n_rows, n_cols = self.grid_size
            max_steps = 2 * n_rows * n_cols

        # para cada posição inicial, simula episódio determinístico
        for pos in start_positions:
            current_pos = pos
            state = current_pos[0]*self.grid_size[1] + current_pos[1]
            success = False

            for _ in range(max_steps):
                # ação greedy (sempre escolher argmax)
                action = int(np.argmax(self.Q[state]))
                next_state, next_pos, _, finish = self._step(
                    state, current_pos, action)

                current_pos = next_pos
                state = next_state

                if finish:
                    # se terminou no goal, sucesso; se terminou por inválido, falha
                    success = (current_pos == self.goal)
                    break

            successes[pos] = success

        return successes

    def load_obstacle_map(self,
                          obstacle_map: np.ndarray | str,
                          ) -> None:

        if isinstance(obstacle_map, np.ndarray):
            arr = obstacle_map
        else:
            raise ValueError(f"Formato não suportado para obstáculos.")

        if arr.shape != self.grid_size:
            raise ValueError(
                f"Mapa de obstáculos tem shape {arr.shape}, esperado {self.grid_size}.")
        # converte para bool
        self.obstacles = (arr != 0)
