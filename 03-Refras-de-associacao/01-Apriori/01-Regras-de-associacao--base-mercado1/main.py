import pandas as pd  # Biblioteca para manipular dados em tabelas
from apyori import apriori  # Biblioteca para realizar análise de regras de associação

# Carrega os dados do arquivo CSV que contém transações de supermercado
# 'header=None' indica que o arquivo não possui cabeçalho
base_mercado = pd.read_csv('mercado.csv', header=None)

# Prepara os dados em uma lista de listas, onde cada sublista representa uma transação
transacoes = []  # Lista para armazenar as transações
for i in range(len(base_mercado)):  # Para cada linha na base de dados
    # Adiciona os itens da transação como strings
    transacoes.append([str(base_mercado.values[i, j]) for j in range(base_mercado.shape[1])])

# Aplica o algoritmo Apriori para encontrar regras de associação
# min_support: frequência mínima de um item nas transações
# min_confidence: confiança mínima para uma regra ser válida
# min_lift: impacto mínimo de uma regra em relação ao acaso
regras = apriori(transacoes, min_support=0.3, min_confidence=0.8, min_lift=2)

# Converte as regras geradas pelo Apriori em uma lista
resultados = list(regras)

# Exibe informações de uma regra específica
print(resultados[2][0])  # Conjunto de itens envolvidos na regra
print(resultados[2][1])  # Suporte da regra (frequência do conjunto de itens)

# Acessa os detalhes da regra (consequentes e métricas de confiança e lift)
r = resultados[2][2]  # Detalhes das métricas

print(r[1])  # Confiança da regra (quão provável o consequente ocorre dado o antecedente)

print(r[2][0])  # Lift da primeira regra (quão relevante é a regra em relação ao acaso)
print(r[2][1])  # Lift da segunda regra (se houver mais de uma associada)