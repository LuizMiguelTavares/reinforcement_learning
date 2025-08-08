import pickle
import pandas as pd

# Define o nome do arquivo a ser lido
NOME_ARQUIVO = 'last_reward.pkl'


def analisar_recompensas(nome_arquivo):
    """
    Lê um arquivo pickle contendo um dicionário de listas de recompensas
    e imprime uma análise estatística completa.
    """
    try:
        # Abre o arquivo em modo de leitura binária ('rb')
        with open(nome_arquivo, 'rb') as f:
            # Carrega os dados do arquivo
            dados_recompensa = pickle.load(f)
        print(f"Arquivo '{nome_arquivo}' carregado com sucesso.\n")

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{nome_arquivo}' não foi encontrado.")
        print("Por favor, certifique-se de que o arquivo está na mesma pasta que o script ou forneça o caminho correto.")
        return
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao ler o arquivo: {e}")
        return

    # --- Abordagem Recomendada com Pandas ---
    print("--- Análise Estatística Completa (usando Pandas) ---")

    # Converte o dicionário de listas diretamente para um DataFrame do Pandas
    # As chaves do dicionário viram colunas e as listas viram as linhas.
    df = pd.DataFrame(dados_recompensa)

    # O método .describe() calcula automaticamente as principais estatísticas
    # para todas as colunas numéricas.
    estatisticas = df.describe()

    print(estatisticas)


# Executa a função de análise
if __name__ == "__main__":
    analisar_recompensas(NOME_ARQUIVO)
