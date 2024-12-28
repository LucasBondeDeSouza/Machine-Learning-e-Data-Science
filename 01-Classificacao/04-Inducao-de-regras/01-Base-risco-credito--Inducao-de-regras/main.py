# Importando o módulo Orange, uma biblioteca poderosa para aprendizado de máquina
# e manipulação de dados, especialmente para tarefas de classificação.
import Orange
import Orange.classification

# Carregando a base de dados 'risco_credito_regras.csv' no formato esperado pelo Orange.
# Essa base de dados contém informações para classificar o risco de crédito (baixo, moderado, alto).
base_risco_credito = Orange.data.Table('risco_credito_regras.csv')

# Criando um aprendiz de regras baseado no algoritmo CN2 (uma técnica de aprendizado supervisionado).
# Este algoritmo gera regras compreensíveis a partir dos dados fornecidos.
cn2 = Orange.classification.rules.CN2SDLearner()

# Aplicando o aprendiz à base de dados para criar um conjunto de regras que podem prever
# a classe de risco de crédito com base nos atributos fornecidos.
regras_risco_credito = cn2(base_risco_credito)

# Abaixo temos um código comentado que, se ativado, exibe as regras geradas pelo CN2.
# Essas regras ajudam a entender como as previsões são feitas.
# for regras in regras_risco_credito.rule_list:
#     print(regras)

# Simulação: usando as regras geradas para prever o risco de crédito para novos casos.
previsoes = regras_risco_credito([
    ['boa', 'alta', 'nenhuma', 'acima_35'],  # Caso 1: Cliente com boa história, dívida alta, sem garantias, renda acima de 35
    ['ruim', 'alta', 'adequada', '0_15']    # Caso 2: Cliente com história ruim, dívida alta, garantias adequadas, renda abaixo de 15
])

# Exibindo as previsões para os casos simulados.
# O loop percorre as previsões geradas pelo modelo (índices numéricos) e traduz
# esses índices nos valores correspondentes às classes de risco de crédito (ex.: "alto", "baixo").
for i in previsoes:
    print(base_risco_credito.domain.class_var.values[i])
