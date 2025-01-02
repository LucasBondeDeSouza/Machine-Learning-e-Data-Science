import pandas as pd  # Biblioteca para manipular dados em formato de tabela
from apyori import apriori  # Biblioteca para criar regras de associação entre itens

# Carrega os dados de transações de um arquivo CSV sem cabeçalho
base_mercado2 = pd.read_csv('mercado2.csv', header=None)

# Converte os dados para uma lista de listas, onde cada sublista representa uma transação
transacoes = []  # Lista para armazenar transações
for i in range(base_mercado2.shape[0]):  # Para cada linha no arquivo
    transacoes.append([str(base_mercado2.values[i, j]) for j in range(base_mercado2.shape[1])])

# Aplica o algoritmo Apriori para encontrar padrões nos dados
regras = apriori(
    transacoes, 
    min_support=0.003,  # Frequência mínima dos itens nas transações
    min_confidence=0.2,  # Confiança mínima para validar as regras
    min_lift=3  # Fator mínimo de relevância (impacto) da regra
)

# Transforma os resultados do Apriori em uma lista
resultados = list(regras)

# Listas para armazenar os componentes das regras de associação
A = []  # Lista de itens antecedentes
B = []  # Lista de itens consequentes
suporte = []  # Suporte das regras
confianca = []  # Confiança das regras
lift = []  # Impacto (lift) das regras

# Itera pelos resultados para extrair os detalhes das regras
for resultado in resultados:
    s = resultado[1]  # Suporte da regra
    result_rules = resultado[2]  # Detalhes das regras geradas

    for result_rule in result_rules:  # Itera pelas regras específicas
        a = list(result_rule[0])  # Itens antecedentes (A)
        b = list(result_rule[1])  # Itens consequentes (B)
        c = result_rule[2]  # Confiança da regra
        l = result_rule[3]  # Impacto (lift) da regra
        A.append(a)  # Adiciona os itens antecedentes à lista
        B.append(b)  # Adiciona os itens consequentes à lista
        suporte.append(s)  # Adiciona o suporte à lista
        confianca.append(c)  # Adiciona a confiança à lista
        lift.append(l)  # Adiciona o lift à lista

# Cria um DataFrame com os resultados das regras
rules_df = pd.DataFrame({
    'A': A,  # Antecedentes
    'B': B,  # Consequentes
    'suporte': suporte,  # Suporte
    'confianca': confianca,  # Confiança
    'lift': lift  # Impacto (lift)
})

# Ordena as regras pelo nível de confiança e exibe o resultado
print(rules_df.sort_values(by='confianca', ascending=False))