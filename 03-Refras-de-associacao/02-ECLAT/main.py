import pandas as pd  # Biblioteca para manipular dados em tabelas
from pyECLAT import ECLAT  # Biblioteca para encontrar padrões de compra (análise de itens frequentes)

# Carrega os dados de transações de um arquivo CSV, sem cabeçalho
base_mercado = pd.read_csv('mercado.csv', header=None)

# Cria uma instância do algoritmo ECLAT para análise de associação
eclat = ECLAT(data=base_mercado)

# Exibe a tabela binária, onde 1 indica a presença de um item em uma transação
print(eclat.df_bin)

# Exibe a lista de itens únicos presentes nas transações
print(eclat.uniq_)

# Aplica o algoritmo ECLAT para encontrar combinações de itens frequentes
# min_support: frequência mínima de suporte para considerar a combinação
# min_combination: número mínimo de itens por combinação
# max_combination: número máximo de itens por combinação
indices, suporte = eclat.fit(min_support=0.3, min_combination=1, max_combination=3)

# Exibe os índices das transações onde cada combinação ocorre
print(indices)

# Exibe o suporte (frequência relativa) de cada combinação encontrada
print(suporte)