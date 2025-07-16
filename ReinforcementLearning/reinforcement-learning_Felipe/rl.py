import numpy as np
from environment import Environment
import pickle
import time
from draw_obs_grid import draw_obs_grid

if __name__ == '__main__':

    nx = 7
    ny = 9

    grid_size = (nx, ny)

    f = open("obs_grids/sala.pkl", "rb")
    obstacle_map = pickle.load(f)
    print("Obstacle map loaded successfully.")

    # obstacle_map = draw_obs_grid(ny, nx)

    env = Environment(grid_size=grid_size, obstacle_map=obstacle_map)

    print(f"\nStart Training...")
    env.training(
        n_episodes=10000,
        alpha=0.1,
        gamma=0.9,
        epsilon0=1,
        epsilonf=0.5,
        egreedy_type='exponential',
        reward_type='v1',
        kill_on_invalid=False,
        verbose_interval=100
    )
    print(f"\nTraining Complete.")

    date_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"data/env_{date_str}.pkl"
    env.save(file_name)

    results = env.evaluate_policy()
    n_success = sum(results.values())
    print(f"{n_success} de {len(results)} posições iniciais chegaram ao objetivo.")

    env.plot_policy()
