import numpy as np
import sys


def ler_arquivo_npy(caminho_do_arquivo):
    """
    Lê um arquivo .npy e imprime seu conteúdo e informações.

    Args:
        caminho_do_arquivo (str): O caminho para o arquivo .npy a ser lido.
    """
    try:
        # Tenta carregar os dados do arquivo .npy
        dados = np.load(caminho_do_arquivo)

        print(f"Arquivo '{caminho_do_arquivo}' carregado com sucesso!")
        print("-" * 30)

        # Imprime informações sobre o array
        print(f"Formato (Shape): {dados.shape}")
        print(f"Tipo de Dados (Dtype): {dados.dtype}")
        print(f"Número de Dimensões: {dados.ndim}")
        print(f"Número total de elementos: {dados.size}")

        print("-" * 30)
        print("Conteúdo do arquivo:")

        # Imprime o conteúdo do array
        print(dados)

    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_do_arquivo}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")


if __name__ == "__main__":
    # Verifique se o caminho do arquivo foi passado como argumento na linha de comando
    if len(sys.argv) > 1:
        caminho = sys.argv[1]
    else:
        # Se nenhum arquivo for passado, use um nome de arquivo de exemplo.
        # **IMPORTANTE**: Altere "sua_trajetoria.npy" para o nome real do seu arquivo.
        caminho = "last_rewards.npy"
        print(
            f"Nenhum arquivo especificado. Tentando carregar o arquivo de exemplo: '{caminho}'")
        print("Uso: python seu_script.py <caminho_para_o_arquivo.npy>")

    ler_arquivo_npy(caminho)
