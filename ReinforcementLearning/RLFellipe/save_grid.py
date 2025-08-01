# save_grid.py

import os
import numpy as np
from draw_obs_grid import draw_obs_grid

if __name__ == "__main__":
    GRID_WIDTH = 10  # X
    GRID_HEIGHT = 10  # Y

    print(f"Abrindo grade interativa de {GRID_WIDTH}x{GRID_HEIGHT}...")

    mapa_de_obstaculos = draw_obs_grid(nx=GRID_WIDTH, ny=GRID_HEIGHT)
    mapa_de_obstaculos[:] = mapa_de_obstaculos[::-1, :]

    print("\nGrade definida. A matriz resultante é:")
    print(mapa_de_obstaculos)

    output_dir = "obs_grids"
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "mapa_de_obstaculos.npy")
    np.save(file_path, mapa_de_obstaculos)  # <-- trocado

    print(f"\nMatriz salva com sucesso em: {file_path}")

    print("\n--- Verificação ---")
    try:
        mapa_carregado = np.load(file_path, allow_pickle=False)
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
