import pandas as pd  # Biblioteca para manipular dados em tabelas
from apyori import apriori  # Biblioteca para realizar análise de regras de associação

# Carrega os dados de transações do arquivo CSV, sem cabeçalho
base_mercado = pd.read_csv('mercado.csv', header=None)

# Converte as transações em uma lista de listas (cada transação é uma sublista)
transacoes = []  # Armazena todas as transações
for i in range(len(base_mercado)):  # Itera por cada linha na tabela
    # Cada linha (transação) é transformada em uma lista de strings
    transacoes.append([str(base_mercado.values[i, j]) for j in range(base_mercado.shape[1])])

# Executa o algoritmo Apriori para encontrar padrões nas transações
regras = apriori(
    transacoes, 
    min_support=0.3,  # Suporte mínimo: frequência de um item nas transações
    min_confidence=0.8,  # Confiança mínima: chance de "B" acontecer dado "A"
    min_lift=2  # Lift mínimo: força da regra em relação ao acaso
)

# Converte as regras encontradas em uma lista para manipulação
resultados = list(regras)

# Inicializa listas para armazenar os componentes das regras
A = []  # Itens do lado esquerdo da regra
B = []  # Itens do lado direito da regra
suporte = []  # Suporte da regra
confianca = []  # Confiança da regra
lift = []  # Lift da regra

# Itera sobre as regras geradas
for resultado in resultados:
    s = resultado[1]  # Suporte da regra atual
    result_rules = resultado[2]  # Detalhes das regras associadas

    for result_rule in result_rules:  # Para cada regra individual
        a = list(result_rule[0])  # Itens no lado esquerdo (A)
        b = list(result_rule[1])  # Itens no lado direito (B)
        c = result_rule[2]  # Confiança da regra
        l = result_rule[3]  # Lift da regra

        # Adiciona os valores às respectivas listas
        A.append(a)
        B.append(b)
        suporte.append(s)
        confianca.append(c)
        lift.append(l)

# Cria um DataFrame com as informações das regras geradas
rules_df = pd.DataFrame({
    'A': A,  # Lado esquerdo da regra
    'B': B,  # Lado direito da regra
    'suporte': suporte,  # Suporte da regra
    'confianca': confianca,  # Confiança da regra
    'lift': lift  # Lift da regra
})

# Ordena o DataFrame por 'lift' em ordem decrescente e exibe o resultado
print(rules_df.sort_values(by='lift', ascending=False))