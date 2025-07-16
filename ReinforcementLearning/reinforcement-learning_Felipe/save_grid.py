# save_grid.py

import pickle
import os
from draw_obs_grid import draw_obs_grid
import numpy as np

if __name__ == "__main__":
    GRID_WIDTH = 9
    GRID_HEIGHT = 7

    print(f"Abrindo grade interativa de {GRID_WIDTH}x{GRID_HEIGHT}...")

    mapa_de_obstaculos = draw_obs_grid(nx=GRID_WIDTH, ny=GRID_HEIGHT)

    print("\nGrade definida. A matriz resultante é:")
    print(mapa_de_obstaculos)

    output_dir = "obs_grids"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "mapa_de_obstaculos.pkl")

    with open(file_path, 'wb') as f:
        pickle.dump(mapa_de_obstaculos, f)

    print(f"\nMatriz salva com sucesso em: {file_path}")

    print("\n--- Verificação ---")
    try:
        with open(file_path, 'rb') as f:
            mapa_carregado = pickle.load(f)
        print("Arquivo carregado com sucesso. Conteúdo:")
        print(mapa_carregado)

        if np.array_equal(mapa_de_obstaculos, mapa_carregado):
            print("\nVerificação bem-sucedida: A matriz salva é idêntica à original.")
        else:
            print("\nErro na verificação: A matriz salva é diferente da original.")

    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo: {e}")
